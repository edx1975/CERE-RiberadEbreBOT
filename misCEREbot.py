import os
import json
import asyncio
import time
import numpy as np
import faiss
from difflib import get_close_matches
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
    "Ets un bot expert en la Ribera d'Ebre (14 pobles). "
    "Respón amb estil patxetí, proper i directe, aportant dades històriques del corpus. "
    "Cita la font només quan l'usuari pregunta un tema concret."
)

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
TOKEN_LOG_FILE = os.path.join(DATA_DIR, "token_log.json")

TOP_K = 5
CONTEXT_EXPIRY = 600  # 10 minuts
user_context = {}

POBLES_RIBERA = [
    "Ginestar","Benissanet","Tivissa","Rasquera","Miravet",
    "Móra d’Ebre","Flix","Ascó","La Palma d’Ebre","Batea",
    "Corbera d’Ebre","La Fatarella","Cambrils","Vinebre"
]

# --- CARREGAR CORPUS ---
corpus = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            corpus.append(json.loads(line))
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

def summarize_fragments(fragments, expand=False, list_mode=False, max_items=10, user_id=None, specific_topic=False):
    if not fragments:
        return ["Escolta, però no tinc informació concreta sobre això."]
    
    max_chars = 1200
    if user_id and user_context.get(user_id, {}).get("last_topic"):
        max_chars = 4000

    parts = []

    if list_mode:
        for f in fragments[:max_items]:
            title = f.get("title","Sense títol")
            text = f.get("long_summary","") or f.get("summary","")
            text = text.replace("\n"," ").strip()
            if len(text) > 250:
                text = text[:250].rsplit(".",1)[0] + "."
            parts.append(f"{title}: {text}")
        message = "\n\n".join(parts)
    else:
        for f in fragments:
            text = f.get("long_summary","") or f.get("summary","")
            if expand and f.get("summary"):
                text += " " + f.get("summary")
            text = text.replace("\n"," ").strip()
            if specific_topic:
                title = f.get("title","Sense títol")
                text = f"{text} (Font: F1: {title})"
            parts.append(text)
        body = " ".join(parts)
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(".",1)[0] + "."
        message = body

    return [message]

def verify_location(user_input):
    all_places = list({entry.get("population","").lower() for entry in corpus})
    matches = get_close_matches(user_input.lower(), all_places, n=1, cutoff=0.6)
    return matches[0].capitalize() if matches else None

def ask_openai(prompt, user_id=None, strict_corpus=True, population=None):
    clean_expired_context()

    correct_loc = verify_location(prompt)
    if correct_loc and (not population or correct_loc.lower() != population.lower()):
        population = correct_loc

    if not population and user_id and user_id in user_context:
        last_topic = user_context[user_id].get("last_topic","")
        for poble in POBLES_RIBERA:
            if poble.lower() in last_topic.lower():
                population = poble
                break

    list_keywords = ["llistat", "llista", "coses", "histories", "curiositats", "plants", "menjars", "fetes"]
    is_list = any(k in prompt.lower() for k in list_keywords)

    specific_topic = False
    for poble in POBLES_RIBERA:
        if poble.lower() in prompt.lower():
            specific_topic = True
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
            return ["Escolta’m, però no tinc informació concreta al corpus sobre això."]
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":prompt}
                ],
                max_tokens=500
            )
            return [resp.choices[0].message.content.strip()]
        except Exception as e:
            return [f"Error amb OpenAI: {e}"]

    expand = needs_expansion(prompt)
    msgs = summarize_fragments(fragments, expand=expand, list_mode=is_list, user_id=user_id, specific_topic=specific_topic)

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

    return msgs

# --- TELEGRAM HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola! Sóc el bot del Miscel·lania CERE de la Ribera d’Ebre. "
        "Pregunta’m el que vulguis sobre els 14 pobles o curiositats locals, i t'ho explicaré de manera natural."
    )

async def forget_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_context:
        del user_context[user_id]
    await update.message.reply_text("He esborrat la teva memòria temporal.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.message.from_user.id

    is_list = any(k in user_text.lower() for k in ["llistat", "llista", "coses", "histories", "curiositats", "plants", "menjars", "fetes"])

    fragments = await asyncio.to_thread(ask_openai, user_text, user_id=user_id)

    msgs = summarize_fragments(fragments, expand=needs_expansion(user_text), list_mode=is_list, user_id=user_id)

    await update.message.reply_text(msgs[0])

# --- CONFIGURACIÓ BOT ---
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("forget", forget_cmd))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

# --- INICI BOT ---
if __name__ == "__main__":
    print("Bot patxetí amb memòria temporal i respostes naturals actiu...")
    app.run_polling()
