import asyncio
import html
import logging
import os

import gspread
from dotenv import load_dotenv
from google import genai
from google.genai import types
from oauth2client.service_account import ServiceAccountCredentials
from pydantic import BaseModel
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import time
import random
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Configurazione (le credenziali NON vanno scritte nel codice: usare un file
# .env nella stessa cartella, con queste chiavi)
#
#   TELEGRAM_BOT_TOKEN=xxxxx
#   GEMINI_API_KEY=xxxxx
#   SPREADSHEET_ID=xxxxx
#   GOOGLE_CREDENTIALS_FILE=credentials.json   (opzionale, default sotto)
# ---------------------------------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
GOOGLE_CREDENTIALS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Zittisce i log molto verbosi di alcune librerie di terze parti
logging.getLogger("httpx").setLevel(logging.WARNING)

ai_client = genai.Client(api_key=GEMINI_API_KEY)


class TypeAnalysis(BaseModel):
    tipo: str
    traduzione: str
    sinonimi: list[str]
    esempi: list[str]


class WordAnalysis(BaseModel):
    analisi: list[TypeAnalysis]


HEADER_ROW = ["Parola", "Tipo", "Traduzione", "Sinonimi", "Esempi"]

# Il foglio viene aperto una sola volta all'avvio e riutilizzato,
# invece di riautenticarsi ad ogni messaggio.
_sheet = None


def _init_sheet():
    """Autentica e apre il foglio Google Sheets (chiamata bloccante, va
    eseguita fuori dall'event loop async)."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1

    # Aggiunge l'intestazione se il foglio è vuoto
    if not sheet.get_all_values():
        sheet.append_row(HEADER_ROW)

    return sheet


def get_sheet():
    global _sheet
    if _sheet is None:
        _sheet = _init_sheet()
    return _sheet


#def _call_gemini(word: str) -> WordAnalysis:
    """Chiamata bloccante a Gemini, da eseguire in un thread separato."""
    prompt = (
        f"Analizza la parola inglese '{word}'. Se la parola ha più funzioni "
        f"grammaticali (es. sia sostantivo che verbo), crea una voce separata "
        f"per ciascuna. Per ogni funzione grammaticale indica: il tipo "
        f"(sostantivo, verbo, aggettivo, avverbio), la traduzione in italiano "
        f"specifica per quel significato, 3 sinonimi e 2 frasi di esempio con "
        f"traduzione."
    )
    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=WordAnalysis,
        ),
    )
    # Con response_schema impostato, la libreria fornisce già l'oggetto
    # validato tramite .parsed: non serve rifare il parsing manuale del JSON.
    return response.parsed

def _call_gemini(word: str, max_retries: int = 5, base_delay: float = 2.0) -> WordAnalysis:
    """Chiamata bloccante a Gemini, da eseguire in un thread separato.
    Riprova automaticamente in caso di errori temporanei (503, 429, ecc.)."""
    prompt = (
        f"Analizza la parola inglese '{word}'. Se la parola ha più funzioni "
        f"grammaticali (es. sia sostantivo che verbo), crea una voce separata "
        f"per ciascuna. Per ogni funzione grammaticale indica: il tipo "
        f"(sostantivo, verbo, aggettivo, avverbio), la traduzione in italiano "
        f"specifica per quel significato, 3 sinonimi e 2 frasi di esempio con "
        f"traduzione."
    )

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            response = ai_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=WordAnalysis,
                ),
            )
            return response.parsed

        except genai_errors.ServerError as e:
            last_exception = e
            status_code = getattr(e, "code", None) or getattr(e, "status_code", None)

            # Se non è un errore temporaneo (es. 400, 401, 403), non ha senso ritentare
            if status_code not in {429, 500, 503, 504} and status_code is not None:
                raise

            if attempt == max_retries:
                break

            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            logger.warning(
                "Gemini non disponibile per '%s' (tentativo %d/%d, status=%s). Riprovo tra %.1fs...",
                word, attempt, max_retries, status_code, delay,
            )
            time.sleep(delay)

    logger.error("Gemini non disponibile per '%s' dopo %d tentativi", word, max_retries)
    raise last_exception


def _append_rows_to_sheet(rows: list[list[str]]) -> None:
    """Scrittura bloccante su Google Sheets, da eseguire in un thread separato.
    Scrive tutte le righe in un'unica chiamata API."""
    sheet = get_sheet()
    sheet.append_rows(rows)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Inviami una parola in inglese per analizzarla e salvarla nel foglio Google!"
    )


async def process_word(update: Update, context: ContextTypes.DEFAULT_TYPE):
    word = update.message.text.strip()

    # Validazione minima: deve essere una singola parola alfabetica
    if not word or not word.replace("-", "").isalpha() or " " in word:
        await update.message.reply_text(
            "⚠️ Per favore inviami una singola parola inglese (senza spazi o numeri)."
        )
        return

    processing_msg = await update.message.reply_text(f"Elaborazione di '{word}' in corso...")

    try:
        # Le chiamate bloccanti vengono spostate in un thread separato
        # per non fermare l'event loop del bot mentre attendono.
        analysis = await asyncio.to_thread(_call_gemini, word)

        rows = []
        reply_blocks = []
        for entry in analysis.analisi:
            sinonimi = ", ".join(entry.sinonimi)
            esempi = " | ".join(entry.esempi)

            rows.append([word, entry.tipo, entry.traduzione, sinonimi, esempi])

            # Uso HTML invece di Markdown ed escaping esplicito: evita crash
            # se la parola/traduzione contiene caratteri speciali (_, *, `, ecc.)
            reply_blocks.append(
                f"🏷️ <b>Tipo:</b> {html.escape(entry.tipo)}\n"
                f"🇮🇹 <b>Traduzione:</b> {html.escape(entry.traduzione)}\n"
                f"🔄 <b>Sinonimi:</b> {html.escape(sinonimi)}\n"
                f"📝 <b>Esempi:</b>\n{html.escape(esempi)}"
            )

        await asyncio.to_thread(_append_rows_to_sheet, rows)

        reply_message = (
            f"✅ <b>Salvato nel Foglio Google!</b> ({len(rows)} righe)\n\n"
            f"📌 <b>Parola:</b> {html.escape(word)}\n\n"
            + "\n\n".join(reply_blocks)
        )
        await processing_msg.edit_text(reply_message, parse_mode=ParseMode.HTML)

    except Exception:
        # Il dettaglio tecnico va nei log, non all'utente (evita di
        # esporre informazioni interne o rendere il messaggio illeggibile).
        logger.exception("Errore durante l'elaborazione della parola '%s'", word)
        await processing_msg.edit_text(
            "⚠️ Si è verificato un errore durante l'elaborazione. Riprova più tardi."
        )


def main():
    # Inizializza il foglio subito, così un eventuale errore di credenziali
    # emerge all'avvio e non al primo messaggio ricevuto.
    get_sheet()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_word))

    logger.info("Bot avviato...")
    app.run_polling()


if __name__ == "__main__":
    main()