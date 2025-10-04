import os
import json
import asyncio
import time
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, HTTPServer

import numpy as np
import faiss
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import openai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓ ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "Ets un bot expert en la Ribera d'Ebre: Ginestar, Benissanet, Tivissa, Rasquera i Miravet. "
    "Respón amablement i també com ChatGPT. "
    "Si l'usuari et diu hola, saluda com ChatGPT."
)

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
TOKEN_LOG_FILE = os.path.join(DATA_DIR, "token_log.json")

TOP_K = 5
RESUME_TOKENS = 150
CONTEXT_EXPIRY = 300  # segons

# --- MEMÒRIA TEMPORAL ---
user_context = {}  # clau: user_id, valor: {"last_topic", "last_embedding", "last_time", "topics_covered"}

# --- CARREGAR CORPUS ---
corpus = []
if os.path.exists(CORPUS_FILE):
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                corpus.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Línia descartada al carregar corpus: {line}")
print(f"📚 {len(corpus)} documents carregats del corpus.")

# --- CARREGAR EMBEDDINGS ---
if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✅ Carregats {embeddings.shape[0]} embeddings existents.")
else:
    embeddings = np.zeros((0, 1536), dtype=np.float32)

# --- CARREGAR FAISS HNSW (més escalable) ---
d = embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32)  # 32 és la mida de l’eficiència HNSW
index.hnsw.efConstruction = 40
if embeddings.shape[0] > 0:
    index.add(embeddings)
print(f"📈 FAISS index inicialitzat amb {index.ntotal} vectors.")

# --- FUNCIONS AUXILIARS ---
def needs_expansion(user_query: str) -> bool:
    triggers = ["detall", "detalls", "explica", "explicació", "amplia", "llarg", "més informació", "aprofund"]
    return any(t in user_query.lower() for t in triggers)

def semantic_search(query, top_k=TOP_K, topics=None, population=None):
    try:
        resp = openai.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        D, I = index.search(q_emb, top_k * 3)
    except Exception:
        return []

    candidates = [corpus[i] for i in I[0]]

    # Prioritza població
    if population:
        pop_fragments = [f for f in candidates if population.lower() in f.get("population","").lower()]
        other_fragments = [f for f in candidates if population.lower() not in f.get("population","").lower()]
        candidates = pop_fragments + other_fragments

    # Filtrat per topics
    if topics:
        filtered = [
            f for f in candidates
            if any(
                t.lower() in f.get("summary","").lower() or
                t.lower() in f.get("long_summary","").lower() or
                t.lower() in [x.lower() for x in f.get("topics",[])]
                for t in topics
            )
        ]
        candidates = filtered if filtered else candidates

    # Boost de títol
    candidates.sort(key=lambda x: bool(x.get("title","")), reverse=True)

    # Truncament segur de fragments
    result = []
    for f in candidates:
        summary = f.get("summary", "")
        if len(summary.split()) > 200:  # truncament 200 paraules
            summary = " ".join(summary.split()[:200]) + "..."
        f_copy = f.copy()
        f_copy["summary"] = summary
        result.append(f_copy)
    return result[:top_k]

def summarize_fragments(fragments, expand=False):
    parts = []
    for f in fragments:
        base = f"📌 {f.get('title','[Sense títol]')}\n{f.get('summary','')}"
        if expand and f.get("long_summary"):
            base += f"\n🔎 Detalls: {f['long_summary']}"
        parts.append(base)
    return "\n\n".join(parts)

def log_tokens(user_id, tokens_used, cost):
    log = {}
    if os.path.exists(TOKEN_LOG_FILE):
        with open(TOKEN_LOG_FILE, "r", encoding="utf-8") as f:
            try:
                log = json.load(f)
            except:
                log = {}
    if str(user_id) not in log:
        log[str(user_id)] = {"tokens":0,"euros":0.0}
    log[str(user_id)]["tokens"] += tokens_used
    log[str(user_id)]["euros"] += cost
    with open(TOKEN_LOG_FILE,"w",encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def expire_context():
    now = time.time()
    to_delete = []
    for uid, ctx in user_context.items():
        if now - ctx.get("last_time",0) > CONTEXT_EXPIRY:
            to_delete.append(uid)
    for uid in to_delete:
        del user_context[uid]
    threading.Timer(60, expire_context).start()  # revisa cada minut

expire_context()

# --- FUNCIO PRINCIPAL DEL BOT ---
def ask_openai(prompt, user_id=None, strict_corpus=True, population=None):
    if user_id and user_id in user_context:
        last_topic = user_context[user_id].get("last_topic","")
        for poble in ["Ginestar","Benissanet","Tivissa","Rasquera","Miravet"]:
            if poble.lower() in last_topic.lower():
                population = poble
                break

    # Extreure topics
    try:
        resp_topics = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user",
                       "content":f"Extreu fins a 5 temes principals d'aquest text, separats per comes:\n{prompt}"}],
            temperature=0
        )
        topics = [t.strip() for t in resp_topics.choices[0].message.content.split(",") if t.strip()][:5]
    except:
        topics = []

    if user_id and user_id in user_context:
        covered = user_context[user_id].get("topics_covered", set())
        topics = [t for t in topics if t not in covered]

    fragments = semantic_search(prompt, topics=topics, population=population)
    if not fragments:
        if strict_corpus:
            return "⚠️ No tinc informació concreta al corpus sobre això."
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
                max_tokens=500
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"S'ha produït un error amb OpenAI: {e}"

    expand = needs_expansion(prompt)
    summary = summarize_fragments(fragments, expand=expand)

    # Actualitzar context
    try:
        emb_topic = np.array(openai.embeddings.create(input=prompt, model="text-embedding-3-small").data[0].embedding, dtype=np.float32)
        if user_id:
            if user_id not in user_context:
                user_context[user_id] = {
                    "last_topic": prompt,
                    "last_embedding": emb_topic,
                    "last_time": time.time(),
                    "topics_covered": set(topics)
                }
            else:
                user_context[user_id]["last_topic"] = prompt
                user_context[user_id]["last_embedding"] = emb_topic
                user_context[user_id]["last_time"] = time.time()
                user_context[user_id]["topics_covered"].update(topics)
    except:
        pass

    user_prompt = f"Utilitza només la informació següent extreta dels arxius del CERE:\n\n{summary}\n\nResposta a la pregunta de l'usuari: {prompt}"
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
            max_tokens=600
        )
        usage = resp.usage
        if user_id:
            cost = (usage.total_tokens / 1000) * 0.001
            log_tokens(user_id, usage.total_tokens, cost)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"S'ha produït un error amb OpenAI: {e}"

# --- HANDLERS TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Hola! Sóc un bot expert de la Miscel·lània del CERE (Ribera d'Ebre). Com puc ajudar-te avui?"
    )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.message.from_user.id
    resp = await asyncio.to_thread(ask_openai, user_text, user_id=user_id)
    await update.message.reply_text(resp)

# --- CONFIGURACIÓ BOT ---
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

# --- HEALTHCHECK HTTP SERVER ---
def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    class HealthHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-type","text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_error(404, "Not Found")
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"🌐 Healthcheck HTTP server listening on port {port}")
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# --- INICI BOT ---
if __name__ == "__main__":
    print("🤖 Bot amb FAISS HNSW, memòria expirable i logging actiu...")
    app.run_polling()
