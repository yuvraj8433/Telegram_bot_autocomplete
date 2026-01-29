from transformers import pipeline
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import torch 
import os

# 🔹 Load model once (important)
generator = pipeline(
    "text-generation",
    model="gpt2",
    device_map="auto"
)

def autocomplete_sentences(prompt, max_new_tokens=50, num_options=3, temperature=0.7):
    results = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_options,
        temperature=temperature,
        do_sample=True,
        top_k=20,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    completions = [
        r["generated_text"].replace(prompt, "").strip()
        for r in results
    ]
    return completions

# 🔹 Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Radhe Radhe 🙏\nSend me a sentence and I’ll autocomplete it 🤖"
    )

# 🔹 Handle user text
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text
    completions = autocomplete_sentences(prompt)

    reply = ""
    for i, c in enumerate(completions, 1):
        reply += f"{i}. {prompt} {c}\n\n"

    await update.message.reply_text(reply)

# 🔹 Main runner
if __name__ == "__main__":
    BOT_TOKEN = os.getenv("BOT_TOKEN")   # 🔐 secure

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 Bot is running...")
    app.run_polling()
    