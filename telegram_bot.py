import logging
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.constants import ParseMode
load_dotenv()
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.api.app import _load_pipeline, run_pipeline, PIPELINE
from urllib.parse import urlparse

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("TelegramBot")

_BOT_NAME    = "FactGuard AI"
_BOT_TAGLINE = "Fake News Detection System"

_SPECTRUM_META: dict[str, tuple[str, str, str]] = {
    "SUPPORTED":           ("✅", "VERIFIED",           "Supported by credible evidence"),
    "PARTIALLY_SUPPORTED": ("⚠️", "PARTIALLY TRUE",     "Core facts are real, but details differ"),
    "UNSUPPORTED":         ("⚪", "UNVERIFIABLE",        "No strong evidence to confirm or deny"),
    "UNVERIFIABLE":        ("⚪", "UNVERIFIABLE",        "No strong evidence to confirm or deny"),
    "MISLEADING_FRAMING":  ("🟠", "MISLEADING",         "Facts are twisted or out of context"),
    "CONTRADICTED":        ("❌", "FALSE / REFUTED",    "Actively refuted by available evidence"),
    "FALSE_FABRICATED":    ("🔴", "FALSE / FABRICATED", "Entirely fabricated — contradicts documented reality"),
    "UNKNOWN":             ("❓", "INCONCLUSIVE",        "Insufficient data to reach a verdict"),
}

_RISK_META: dict[str, tuple[str, str]] = {
    "HIGH":   ("🔴", "manipulative or biased"),
    "MEDIUM": ("🟡", "moderately biased"),
    "LOW":    ("🟢", "neutral and objective"),
}

def _escape(text: str) -> str:
    special = r"\_*[]()~`>#+-=|{}.!"
    out = []
    for ch in str(text):
        if ch in special:
            out.append(f"\\{ch}")
        else:
            out.append(ch)
    return "".join(out)

def _verdict_line(label: str) -> str:
    emoji, display, desc = _SPECTRUM_META.get(
        label.upper(), _SPECTRUM_META["UNKNOWN"]
    )
    return f"{emoji} *{_escape(display)}*  —  _{_escape(desc)}_"


def _risk_line(risk: str) -> str:
    emoji, text = _RISK_META.get(risk.strip().upper(), ("⚪", risk))
    return f"{emoji} {_escape(text)}"


def _subjectivity_label(score: float) -> str:
    if score < 0.30:
        return "Objective — language is largely factual and neutral"
    if score < 0.55:
        return "Mixed — blends facts with subjective framing"
    return "Highly subjective — opinion-driven and emotional"

def format_result(result: dict) -> str:
    if "error" in result and "claim" not in result:
        return f"❌ *Error*\n\n{_escape(result['error'])}"

    claim = result.get("claim", "").strip()
    verdict = result.get("final_verdict", {})
    manip = result.get("manipulation_analysis", {})
    narrative = result.get("counter_narrative", "").strip()
    sources = result.get("sources", [])

    label = verdict.get("spectrum") or verdict.get("label", "UNKNOWN")
    emoji, title, desc = _SPECTRUM_META.get(label.upper(), _SPECTRUM_META["UNKNOWN"])

    risk = manip.get("manipulation_risk", "LOW").strip().upper()
    risk_emoji, risk_desc = _RISK_META.get(risk, ("⚪", risk))
    style_label = _subjectivity_label(manip.get("subjectivity_score", 0))

    L = [
        f"🔍 *FACT\\-CHECK REPORT*",
        "",
        f"*Claim analysed:* _{_escape(claim)}_",
        "",
        f"*VERDICT:* {emoji} *{_escape(title)}*  {_escape(desc)}",
        "",
        f"📊 *MANIPULATION ANALYSIS*",
        "",
        f"  • *Style:* {_escape(style_label)}",
        f"  • *Risk:* {risk_emoji} {_escape(risk)} \\ — {_escape(risk_desc)}",
        "",
        f"🤖 *AI REASONING*",
        "",
        f"{_escape(verdict.get('explanation', 'No explanation provided.'))}",
    ]

    if narrative:
        L += ["", f"💡 *COUNTER\\-NARRATIVE*", "", f"{_escape(narrative)}"]

    if sources:
        L += ["", "🌐 *SOURCES*", ""]
        for src in sources[:3]:
            # Support both new dict format {"url": ..., "title": ...}
            # and legacy plain-string format (URL only).
            if isinstance(src, dict):
                url   = src.get("url", "")
                title = src.get("title", "").strip()
            else:
                url   = src
                title = ""
            if not url:
                continue
            if not title:
                parsed = urlparse(url)
                title  = parsed.netloc or url
            # Truncate very long titles so the line stays readable in Telegram.
            if len(title) > 60:
                title = title[:57] + "..."
            L.append(f"  • [{_escape(title)}]({url})")

    return "\n".join(L)

def _main_menu_keyboard() -> InlineKeyboardMarkup:
    """Beautiful and organized menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📖 How to use", callback_data="menu_howto"),
            InlineKeyboardButton("⚖️ Verdict Scale", callback_data="menu_scale")
        ],
        [
            InlineKeyboardButton("🔬 How it works", callback_data="menu_ai"),
            InlineKeyboardButton("💡 Examples", callback_data="menu_examples")
        ]
    ])

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    first_name = _escape(user.first_name) if user and user.first_name else "there"

    text = (
        f"🛡️ *{_escape(_BOT_NAME)}*\n"
        f"_{_escape(_BOT_TAGLINE)}_\n\n"
        f"Welcome, {first_name} 👋\n"
        f"Send any news or claim — I'll analyze it for you\.\n\n"
        f"*What you'll get:*\n"
        f"  • 🔍 Fact verification\n"
        f"  • ⚠️ Manipulation detection\n"
        f"  • 🧠 Reasoned conclusions\n\n"
        f"👇 Select an option below or just send a claim to start"
    )

    await update.message.reply_text(
        text,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=_main_menu_keyboard(),
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    text = (
        "*HELP*\n\n"
        "*How to use:*\n"
        "Type or paste any news claim as a message and send it\\.\n\n"
        "*Verdicts:*\n"
        + "".join(
            f"  {emoji} {display} — {desc}\n"
            for emoji, display, desc in dict.fromkeys(
                (v[0], v[1], v[2]) for v in _SPECTRUM_META.values()
            )
        )
        + "\n"
        "*Manipulation risk levels:*\n"
        "  🔴 HIGH — alarming or emotionally charged language\n"
        "  🟡 MEDIUM — moderately biased phrasing\n"
        "  🟢 LOW — neutral, predominantly factual tone\n\n"
        "*Commands:*\n"
        "  /start — welcome screen\n"
        "  /help — this message\n"
        "  /about — about the AI pipeline\n"
        "  /examples — example claims to try\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def about_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /about."""
    text = (
        f"*About {_escape(_BOT_NAME)}*\n\n"
        "Built on a *Triple\\-Signal AI Pipeline*:\n\n"
        "*1\\. RoBERTa Classifier*\n"
        "   Analyses writing style and linguistic patterns for an initial verdict\\.\n\n"
        "*2\\. Sentiment Analyzer \\(GoEmotions\\)*\n"
        "   Measures emotional tone, subjectivity, and manipulation risk using "
        "a fine\\-tuned RoBERTa model across 28 emotion categories\\.\n\n"
        "*3\\. RAG Evidence Retrieval*\n"
        "   Searches Wikipedia and the web in real\\-time\\. Runs a dedicated "
        "refutation search to actively find sources that contradict the claim\\.\n\n"
        "*4\\. Mistral 7B \\(Supreme Judge\\)*\n"
        "   Receives all three signals, applies a Hard Relevance Gate, "
        "credibility tier hierarchy \\(Tier 1\\-5\\), Credibility Gap detection, "
        "Scenario A/B distinction \\(lack of support vs\\.  active contradiction\\), "
        "and returns a granular Truth Spectrum verdict\\.\n\n"
        "_When Mistral overrides the classifier, a Scientific Justification "
        "is always provided in the explanation\\._\n\n"
        "_Built as a thesis project\\._"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def examples_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /examples."""
    text = (
        "*EXAMPLE CLAIMS*\n\n"
        "Copy and paste any of these to try the bot:\n\n"
        "🔴 _Vaccines have been proven to cause autism_\n\n"
        "🟢 _Sweden joined NATO in March 2024_\n\n"
        "🟠 _5G towers were installed to spread COVID\\-19_\n\n"
        "🟡 _The WHO confirmed that coffee reduces cancer risk_\n\n"
        "🔴 _Bluetooth radiation changes human DNA_\n\n"
        "⚪ _A new study proves intermittent fasting doubles lifespan_\n\n"
        "_Just send the text directly — no extra formatting needed\\._"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses from the main menu."""
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "menu_howto":
        text = (
            "📌 *How to use*\n\n"
            "Just send any news claim or headline as a message\\.\n\n"
            "You'll get a full report with:\n"
            "  • ⚖️ Truth verdict\n"
            "  • 🧩 Claim breakdown \\(if needed\\)\n"
            "  • ⚠️ Manipulation \\& tone analysis\n"
            "  • 🧠 AI reasoning \\& explanation\n"
            "  • 🔄 Evidence\\-based counter\\-narrative\n"
            "  • 📚 Sources\n\n"
            "✉️ No commands needed \\- just type and send"
        )

    elif data == "menu_scale":
        rows = "\n\n".join(
            f" {emoji} *{_escape(display)}* \\- {_escape(desc)}"
            for emoji, display, desc in dict.fromkeys(
                (v[0], v[1], v[2]) for v in _SPECTRUM_META.values()
            )
        )
        text = f"*Verdict Scale*\n\n{rows}"

    elif data == "menu_ai":
        text = (
            "*How the AI works*\n\n"
            "The pipeline runs three signals:\n\n"
            "*1\\. RoBERTa classifier*\n"
            "   Gives an initial REAL/FAKE verdict based on writing style\\.\n\n"
            "*2\\. Sentiment Analyzer*\n"
            "   Measures emotional tone, subjectivity score, and manipulation "
            "risk across 28 emotion categories\\.\n\n"
            "*3\\. RAG retriever*\n"
            "   Searches Wikipedia and the web in real time\\. "
            "Also fires a dedicated refutation search to actively find "
            "sources that say the claim is false\\.\n\n"
            "*4\\. Mistral 7B judge*\n"
            "   Receives all signals, applies a Hard Relevance Gate, "
            "credibility tier weighting, Credibility Gap detection, "
            "and Scenario A/B distinction\\. Returns a Truth Spectrum verdict\\.\n\n"
            "_If Mistral overrides the classifier, a Scientific Justification "
            "is always stated in the explanation\\._"
        )

    elif data == "menu_examples":
        text = (
            "*Example claims to try*\n\n"
            f"  • Vaccines have been proven to cause autism\n\n"
            f"  • Sweden joined NATO in March 2024\n\n"
            f"  • 5G towers were installed to spread COVID\\-19\n\n"
            f"  • The WHO confirmed that coffee reduces cancer risk\n\n"
            f"  • Bluetooth radiation changes human DNA\n\n"
            "💬 Send any of these as a message to see a full report\\."
        )

    else:
        text = "_Unknown option\\._"

    await query.edit_message_text(
        text,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=_main_menu_keyboard(),
    )


async def claim_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    claim = (update.message.text or "").strip()
    if not claim:
        return

    if len(claim) < 10:
        await update.message.reply_text(
            "✏️ _Please send a full claim or headline \\(at least 10 characters\\)\\._",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    status_msg = await update.message.reply_text(
        "⏳ *Processing Request\.\.\.*\n"
        "_Analyzing writing style, searching web evidence, and generating reasoning\._",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    try:
        result    = run_pipeline(claim)
        formatted = format_result(result)
        await status_msg.edit_text(formatted, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as exc:
        logger.exception("Handler error for claim '%s': %s", claim[:60], exc)
        await status_msg.edit_text(
            f"❌ *Error*\n\n{_escape(str(exc))}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Telegram update caused error: %s", context.error, exc_info=True)

async def _post_init(application) -> None:
    commands = [
        BotCommand("start",    "Welcome screen and main menu"),
        BotCommand("help",     "How to use the bot"),
        BotCommand("examples", "Example claims to try"),
        BotCommand("about",    "About the AI pipeline"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands registered.")


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print(
            "[TelegramBot] ERROR: TELEGRAM_BOT_TOKEN is not set.\n"
            "  Windows : set TELEGRAM_BOT_TOKEN=<your_token>\n"
            "  Linux   : export TELEGRAM_BOT_TOKEN=<your_token>\n"
            "  Get a token from @BotFather on Telegram."
        )
        sys.exit(1)

    if not PIPELINE:
        logger.info("Loading pipeline components (one-time startup)...")
        _load_pipeline()
        logger.info("Pipeline ready. Starting Telegram bot...")

    app = (
        ApplicationBuilder()
        .token(token)
        .post_init(_post_init)
        .build()
    )

    app.add_handler(CommandHandler("start",    start_handler))
    app.add_handler(CommandHandler("help",     help_handler))
    app.add_handler(CommandHandler("about",    about_handler))
    app.add_handler(CommandHandler("examples", examples_handler))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, claim_handler))
    app.add_error_handler(error_handler)

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()