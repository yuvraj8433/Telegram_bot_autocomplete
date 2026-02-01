from transformers import pipeline
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import torch 
import os
import re

# 🔹 Load model once (important)
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1   # 👈 FORCE CPU
    model_kwargs={"torch_dtype": torch.float32}
)

def autocomplete_sentences(
    prompt,
    max_new_tokens=100,   # thoda zyada
    num_options=1,
    temperature=0.7
):
    results = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_options,
        temperature=temperature,
        do_sample=True,
        top_k=20,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    completions = []

    for r in results:
        text = r["generated_text"].replace(prompt, "").strip()

        # ✅ last complete sentence tak cut
        sentences = re.findall(r'.*?[.!?]', text)
        if sentences:
            text = sentences[-1] if len(sentences) == 1 else " ".join(sentences)

        completions.append(text.strip())

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
    
