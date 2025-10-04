# bot_ribera_conversacional.py
import os
import json
import time
import asyncio
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

import numpy as np
import faiss
from dotenv import load_dotenv
import openai

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

# --- CONFIGURACIÓ ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
openai.api_key = OPENAI_API_KEY

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
TOKEN_LOG_FILE = os.path.join(DATA_DIR, "token_log.json")

TOP_K = 5
CONTEXT_EXPIRY = 5 * 60
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
MAX_RESPONSE_TOKENS = 450
TEMPERATURE = 0.5  # una mica creatiu i simpàtic

SYSTEM_PROMPT = (
    "Ets un bot expert en la Ribera d'Ebre: Ginestar, Benissanet, Tivissa, Rasquera i Miravet. "
    "RESPON amb gràcia, curiositats i exemples divertits sempre que sigui possible, "
    "però només basant-te en la informació dels fragments proporcionats. "
    "Si no tens informació, RESPON exactament: \"No tinc informació al corpus sobre això.\""
)

# --- MEMÒRIA ---
user_context = {}  # user_id -> {"last_time", "last_query", "topics_covered": set()}

# --- CARREGAR CORPUS ---
corpus = []
corpus_texts = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            entry.setdefault("title", f"fragment_{i}")
            entry.setdefault("summary", "")
            entry.setdefault("long_summary", "")
            entry.setdefault("population", "")
            entry.setdefault("id", str(i))
            corpus.append(entry)
            corpus_texts.append(" ".join([entry.get("title", ""), entry.get("summary", ""), entry.get("long_summary", "")]))
        except json.JSONDecodeError:
            print(f"⚠️ Línia descartada al corpus: {line[:120]}")

# --- CARREGAR / CREAR EMBEDDINGS ---
if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
else:
    embeddings = np.zeros((0, 1536), dtype=np.float32)

if embeddings.shape[0] != len(corpus):
    print("ℹ️ Generant embeddings per al corpus...")
    new_embs = []
    for entry in corpus:
        text = " ".join([entry.get("title",""), entry.get("summary",""), entry.get("long_summary","")])
        try:
            resp = openai.embeddings.create(input=text, model=EMBED_MODEL)
            emb = np.array(resp.data[0].embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            new_embs.append(emb)
        except Exception:
            new_embs.append(np.zeros((1536,), dtype=np.float32))
    embeddings = np.vstack(new_embs).astype(np.float32)
    np.save(EMBEDDINGS_FILE, embeddings)

# --- FAISS INDEX ---
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# --- FUNCIONS AUXILIARS ---
def clean_expired_context():
    now = time.time()
    expired = [uid for uid, ctx in user_context.items() if now - ctx.get("last_time", 0) > CONTEXT_EXPIRY]
    for uid in expired:
        del user_context[uid]

def needs_expansion(text: str):
    triggers = ["detall", "detalls", "explica", "aprofund", "més informació", "explica'm", "amplia"]
    t = text.lower()
    return any(x in t for x in triggers)

def extract_topics(prompt: str):
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": f"Extreu fins a 5 temes o paraules-clau separades per coma d'aquest text:\n{prompt}"}],
            temperature=0,
            max_tokens=60
        )
        raw = resp.choices[0].message.content
        topics = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()][:5]
        return topics
    except Exception:
        return []

def semantic_search(query: str, top_k=TOP_K, population: str = None, topics: list = None):
    try:
        qemb = np.array(openai.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(qemb)
        if norm > 0:
            qemb = qemb / norm
    except Exception:
        return []

    qemb = qemb.reshape(1, -1)
    D, I = index.search(qemb, top_k * 6)
    candidate_indices = [int(i) for i in I[0] if i != -1]

    results = []
    for idx in candidate_indices:
        frag = corpus[idx]
        score = float(D[0][candidate_indices.index(idx)])
        if population and population.lower() in frag.get("population", "").lower():
            score += 0.3
        if topics:
            text = " ".join([frag.get("title",""), frag.get("summary",""), frag.get("long_summary","")]).lower()
            for t in topics:
                if t.lower() in text:
                    score += 0.1
        results.append((score, frag))
    results.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    unique_results = []
    for sc, fr in results:
        if fr.get("id") in seen:
            continue
        seen.add(fr.get("id"))
        unique_results.append((sc, fr))
        if len(unique_results) >= top_k:
            break
    unique_results = [(s,f) for (s,f) in unique_results if f.get("summary") or f.get("long_summary")]
    return unique_results[:top_k]

def format_fragments_for_prompt(frags):
    parts = []
    for score, f in frags:
        parts.append({
            "id": f.get("id"),
            "title": f.get("title"),
            "summary": f.get("summary")[:800],
            "long_summary": (f.get("long_summary") or "")[:1200],
            "population": f.get("population",""),
            "score": round(float(score), 4)
        })
    return parts

def build_user_prompt(question: str, fragments_for_prompt: list, expand=False):
    fragments_text = ""
    for i, f in enumerate(fragments_for_prompt, start=1):
        fragments_text += f"F{i}. Títol: {f['title']} (id: {f['id']})\nResum: {f['summary']}\n\n"

    instruction = (
        "INSTRUCCIONS:\n"
        "- RESPON UTILITZANT NOMÉS la informació dels fragments F1..Fn.\n"
        "- Sigues simpàtic, curiós i didàctic.\n"
        "- Si no pots respondre amb aquests fragments, RESPON exactament: \"No tinc informació al corpus sobre això.\"\n"
        "- Dona una resposta breu (3-8 línies) i al final llista les fonts F{i} (Títol).\n\n"
    )
    if expand:
        instruction += "- L'usuari vol més detalls; si hi ha 'long_summary', afegeix 1-2 frases addicionals.\n\n"

    prompt = f"{SYSTEM_PROMPT}\n\n{instruction}\nFRAGMENTS:\n{fragments_text}\nPREGUNTA DE L'USUARI: {question}\nRESPON:"
    return prompt

def log_tokens(user_id, tokens_used, cost):
    try:
        if os.path.exists(TOKEN_LOG_FILE):
            with open(TOKEN_LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = {}
        if str(user_id) not in log:
            log[str(user_id)] = {"tokens": 0, "euros": 0.0}
        log[str(user_id)]["tokens"] += tokens_used
        log[str(user_id)]["euros"] += cost
        with open(TOKEN_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- FUNCIO PRINCIPAL ---
def ask_openai(question: str, user_id=None, population=None):
    clean_expired_context()
    topics = extract_topics(question)
    frags = semantic_search(question, top_k=TOP_K, population=population, topics=topics)
    if not frags:
        return "No tinc informació al corpus sobre això."

    fragments_for_prompt = format_fragments_for_prompt(frags)
    expand = needs_expansion(question)
    user_prompt = build_user_prompt(question, fragments_for_prompt, expand=expand)

    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS
        )
        text = resp.choices[0].message.content.strip()
        if user_id:
            user_context[user_id] = {
                "last_time": time.time(),
                "topics_covered": set(topics),
                "last_query": question
            }
        return text
    except Exception as e:
        return f"Error amb OpenAI: {e}"

# --- TELEGRAM HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Hola! Sóc el bot de la Ribera d'Ebre. Pregunta'm coses curioses o històriques i t'ho explicaré amb gràcia! /forget per esborrar memòria."
    )

async def forget_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_context.pop(user_id, None)
    await update.message.reply_text("🗑️ Memòria d'usuari esborrada.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.message.from_user.id
    if user_text.strip().lower() in ["/forget", "esborra memòria", "esborra"]:
        user_context.pop(user_id, None)
        await update.message.reply_text("🗑️ Memòria eliminada.")
        return
    resp = await asyncio.to_thread(ask_openai, user_text, user_id)
    await update.message.reply_text(resp)

# --- HEALTHCHECK SERVER ---
def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    class HealthHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_error(404, "Not Found")
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# --- CONFIGURACIÓ BOT TELEGRAM ---
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("forget", forget_cmd))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

if __name__ == "__main__":
    print("🤖 Bot conversacional de la Ribera d'Ebre en execució!")
    app.run_polling()
