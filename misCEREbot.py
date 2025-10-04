import os
import json
import asyncio
import time
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
    "Ets un bot expert en la Ribera d'Ebre (Ginestar, Benissanet, Tivissa, Rasquera i Miravet). "
    "Respón amb estil patxetí, proper i directe, aportant dades històriques del corpus. "
    "Cada fet històric ha de portar la seva font citada (F1, F2…) amb títol resumit. "
    "Si l’usuari demana detalls, amplia la resposta amb més informació disponible, sinó respon breu. "
    "Incorpora algunes paraules del diccionari patxetí de manera natural dins del text."
)

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
TOKEN_LOG_FILE = os.path.join(DATA_DIR, "token_log.json")
DICCIONARI_PATXETI_FILE = os.path.join(DATA_DIR, "diccionari_patxeti.json")

# --- CARREGAR DICCIONARI PATXETÍ ---
with open(DICCIONARI_PATXETI_FILE, "r", encoding="utf-8") as f:
    DICCIONARI_PATXETI = json.load(f)

TOP_K = 5
CONTEXT_EXPIRY = 600  # 10 minuts

# --- MEMÒRIA TEMPORAL ---
user_context = {}  # user_id -> {"last_topic": str, "last_embedding": np.array, "last_time": timestamp, "topics_covered": set()}

# --- CARREGAR CORPUS ---
corpus = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            corpus.append(entry)
        except json.JSONDecodeError:
            print(f"Línia descartada: {line}")

# --- CARREGAR / CREAR EMBEDDINGS ---
if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Carregats {embeddings.shape[0]} embeddings existents.")
else:
    embeddings = np.zeros((0, 1536), dtype=np.float32)

existing_count = embeddings.shape[0]
new_docs = corpus[existing_count:]

if new_docs:
    new_embeddings = []
    for entry in new_docs:
        text = " ".join([
            entry.get("title",""),
            entry.get("summary",""),
            entry.get("long_summary",""),
            " ".join(entry.get("topics", []))
        ])
        try:
            resp = openai.embeddings.create(input=text, model="text-embedding-3-small")
            new_embeddings.append(np.array(resp.data[0].embedding, dtype=np.float32))
        except Exception:
            new_embeddings.append(np.zeros((1536,), dtype=np.float32))
    if new_embeddings:
        embeddings = np.vstack([embeddings, np.array(new_embeddings, dtype=np.float32)])
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"Embeddings nous guardats. Total embeddings: {embeddings.shape[0]}")

# --- CREAR FAISS INDEX ---
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# --- FUNCIONS AUXILIARS ---
def clean_expired_context():
    now = time.time()
    expired = [uid for uid, ctx in user_context.items() if now - ctx.get("last_time", 0) > CONTEXT_EXPIRY]
    for uid in expired:
        del user_context[uid]

def needs_expansion(user_query: str) -> bool:
    triggers = ["detall", "detalls", "explica", "explicació", "amplia", "llarg", "més informació", "aprofund"]
    return any(t in user_query.lower() for t in triggers)

def semantic_search(query, top_k=TOP_K, topics=None, population=None):
    try:
        resp = openai.embeddings.create(input=query, model="text-embedding-3-small")
        q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    except Exception:
        return []

    D, I = index.search(q_emb, top_k*3)
    candidates = [corpus[i] for i in I[0]]

    if population:
        pop_fragments = [f for f in candidates if population.lower() in f.get("population","").lower()]
        other_fragments = [f for f in candidates if population.lower() not in f.get("population","").lower()]
        candidates = pop_fragments + other_fragments

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

    return [f for f in candidates if f.get("summary") or f.get("long_summary")][:top_k]

def summarize_fragments(fragments, expand=False):
    parts = []
    for idx, f in enumerate(fragments, start=1):
        base = f"{idx}. {f.get('title','')}"
        if f.get("years"):
            base += f" (Període: {f['years']})"
        base += f"\n{f.get('summary','')}"
        if expand and f.get("long_summary"):
            base += f"\nDetalls: {f.get('long_summary')}"
        base += f"\nFont: F{idx} ({f.get('title','')})"
        parts.append(base)
    return "\n\n".join(parts)

def log_tokens(user_id, tokens_used, cost):
    try:
        if os.path.exists(TOKEN_LOG_FILE):
            with open(TOKEN_LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = {}
        if str(user_id) not in log:
            log[str(user_id)] = {"tokens":0, "euros":0.0}
        log[str(user_id)]["tokens"] += tokens_used
        log[str(user_id)]["euros"] += cost
        with open(TOKEN_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- FUNCIO PRINCIPAL ---
def ask_openai(prompt, user_id=None, strict_corpus=True, population=None):
    clean_expired_context()

    if not population and user_id and user_id in user_context:
        last_topic = user_context[user_id].get("last_topic","")
        for poble in ["Ginestar","Benissanet","Tivissa","Rasquera","Miravet"]:
            if poble.lower() in last_topic.lower():
                population = poble
                break

    try:
        resp_topics = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":f"Extreu fins a 5 temes principals d'aquest text per fer cerca semàntica, separats per comes:\n{prompt}"}],
            temperature=0
        )
        topics = [t.strip() for t in resp_topics.choices[0].message.content.split(",") if t.strip()][:5]
    except Exception:
        topics = []

    if user_id and user_id in user_context:
        covered = user_context[user_id].get("topics_covered", set())
        topics = [t for t in topics if t not in covered]

    fragments = semantic_search(prompt, topics=topics, population=population)
    if not fragments:
        if strict_corpus:
            return "Escolta’m, però no tinc informació concreta al corpus sobre això."
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":prompt}
                ],
                max_tokens=500
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error amb OpenAI: {e}"

    expand = needs_expansion(prompt)
    summary = summarize_fragments(fragments, expand=expand)

    try:
        emb_topic = np.array(openai.embeddings.create(input=prompt, model="text-embedding-3-small").data[0].embedding, dtype=np.float32)
        if user_id:
            ctx = user_context.setdefault(user_id, {})
            ctx["last_topic"] = prompt
            ctx["last_embedding"] = emb_topic
            ctx["last_time"] = time.time()
            ctx.setdefault("topics_covered", set()).update(topics)
    except Exception:
        pass

    user_prompt = f"Aquí tens la informació trobada al corpus:\n\n{summary}\n\nPregunta: {prompt}"

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":user_prompt}
            ],
            max_tokens=600
        )
        usage = getattr(resp, "usage", None)
        if usage and user_id:
            cost = (usage.total_tokens / 1000) * 0.001
            log_tokens(user_id, usage.total_tokens, cost)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error amb OpenAI: {e}"

# --- TELEGRAM HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola, patxetí! Sóc el bot del Miscel·lania CERE de la Ribera d’Ebre. Pregunta’m el que vulguis sobre Miravet, Rasquera, Tivissa, Ginestar i Benissanet. També pots provar amb curiositats de la zona"
    )

async def forget_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_context:
        del user_context[user_id]
    await update.message.reply_text("Memòria d’usuari esborrada!")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.message.from_user.id
    # Consulta al corpus / OpenAI
    resp = await asyncio.to_thread(ask_openai, user_text, user_id=user_id)
    await update.message.reply_text(resp)


# --- CONFIGURACIÓ BOT ---
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("forget", forget_cmd))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

# --- INICI BOT ---
if __name__ == "__main__":
    print("Bot patxetí amb memòria temporal i cites històriques actiu...")
    app.run_polling()
