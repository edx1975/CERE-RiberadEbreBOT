"""
Bot Telegram per preguntar sobre el corpus de la Ribera d'Ebre.
Requisits:
pip install python-telegram-bot==20.4 openai numpy faiss-cpu rapidfuzz python-dotenv

Configura a .env:
- TELEGRAM_TOKEN
- OPENAI_API_KEY
"""

import os, json, signal, asyncio
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz
import faiss
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# ---------- CONFIG ----------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- MEMÒRIA USUARI ----------
user_memory = {}  # Diccionari global per guardar la memòria de cada usuari

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus_original.jsonl")
EMB_NPY = os.path.join(DATA_DIR, "embeddings_G.npy")
FAISS_IDX = os.path.join(DATA_DIR, "faiss_index_G.index")

TOP_K = 5
MAX_CONTEXT_DOCS = 5
USER_MEMORY_SIZE = 6
MAX_TOKENS = 3500
MAX_MESSAGE_LENGTH = 4096  # Telegram max per segment

# ---------- CARREGA CORPUS ----------
def load_jsonl(path: str):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs

print("Carregant corpus...")
docs = load_jsonl(CORPUS_FILE)
print(f"Corpus carregat: {len(docs)} documents")

# ---------- FAISS INDEX ----------
use_faiss = False
index = None
embeddings = None
if Path(FAISS_IDX).exists() and Path(EMB_NPY).exists():
    try:
        embeddings = np.load(EMB_NPY)
        index = faiss.read_index(FAISS_IDX)
        use_faiss = True
        print("Faiss carregat.")
    except Exception as e:
        print("Error carregant FAISS, fallback a text-match:", e)
else:
    print("No hi ha FAISS, s'utilitzarà text-match.")

# ---------- MEMÒRIA USUARI ----------
def push_user_memory(chat_id, question, answer, docs_used):
    """ Guarda la memòria de l'usuari """
    m = user_memory.setdefault(chat_id, {
        "history": [],
        "last_docs": [],
        "last_title": None,
        "last_mode": None,
        "active_title": None
    })
    m["history"].append((question, answer))
    if len(m["history"]) > USER_MEMORY_SIZE:
        m["history"].pop(0)
    m["last_docs"] = docs_used
    if docs_used:
        m["active_title"] = docs[docs_used[0]].get("title") if docs_used[0] < len(docs) else None

# ---------- UTILS ----------
def get_embedding(text: str) -> np.ndarray:
    """Crida a l'API d'OpenAI per obtenir embedding"""
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def semantic_search(query: str, docs, embeddings=None, index=None, top_k=5):
    """Retorna els indexes dels documents més rellevants"""
    query_lower = query.lower()
    if index is not None and embeddings is not None:
        q_emb = np.array([get_embedding(query)], dtype=np.float32)
        D, I = index.search(q_emb, top_k)
        return [int(i) for i in I[0] if i != -1]
    
    # fallback text-match
    scores = []
    for i, d in enumerate(docs):
        ttext = " ".join(d.get("topics", []) + [d.get("title",""), d.get("summary",""), d.get("summary_long","")]).lower()
        score = fuzz.token_set_ratio(query_lower, ttext)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i,_ in scores[:top_k]]

# ---------- LLM ----------
SYSTEM_PROMPT = """
Ets una IA experta en la història i temes dels pobles de la Ribera d'Ebre.
HAS DE RESPONDRE NOMÉS AMB LA INFORMACIÓ DEL CORPUS.
Si no pots respondre, diu 'No ho sé segons el corpus'.
- Pregunta específica: usa només l'article rellevant.
- Pregunta general: resumeix info de diversos articles, sense repetir.
- Si l'usuari demana /mes, amplia amb detalls i títols.
Sigues amable i clar.
"""

def build_context_for_docs(doc_indexes):
    """Construeix el context a partir dels documents seleccionats"""
    parts = []
    for idx in doc_indexes:
        d = docs[idx]
        parts.append(
            f"== ARTICLE {idx} ==\n"
            f"Title: {d.get('title')}\n"
            f"Summary: {d.get('summary')}\n"
            f"Long: {d.get('summary_long')}\n"
        )
    return "\n\n".join(parts)

def call_llm_with_context(user_query, doc_indexes, temperature=0.0, max_tokens=800):
    """Crida a l'LLM amb context dels documents"""
    context_text = build_context_for_docs(doc_indexes)
    prompt = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user", "content":
            f"Context documents:\n\n{context_text}\n\n"
            f"Pregunta: {user_query}\n\n"
            f"Respon en català. Si et manca info, diu 'No ho sé segons el corpus'."}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- Fragmentació de missatges llargs ---
async def send_long_message(chat_id, text, app):
    """Telegram no accepta missatges massa llargs, així que els fragmentem"""
    chunks = [text[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]
    for chunk in chunks:
        await app.bot.send_message(chat_id=chat_id, text=chunk)
# ---------- WRAPPER TELEGRAM ----------
# --- Detector de mode ---
def detect_mode(text, memory):
    t = text.lower()
    # Mode conversa / chat
    if any(x in t for x in ["com estàs", "què et sembla", "ets", "tu", "com et trobes"]):
        return "chat"
    # Mode detall de font (/mes o hi ha articles previs)
    elif "/mes" in t or (memory and memory.get("last_docs")):
        return "source_detail"
    # Mode resum general
    else:
        return "summary"

# --- Crida a LLM en mode conversa ---
def call_llm_chat_mode(user_query):
    prompt = [
        {"role":"system", "content":
         "Ets una IA amable i propera del territori de la Ribera d’Ebre. Parla en català occidental (Terres de l’Ebre)."},
        {"role":"user", "content": user_query}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.7,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# --- Handler principal ---
async def telegram_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text
    memory = user_memory.get(chat_id, {})

    mode = detect_mode(text, memory)

    if mode == "chat":
        reply = call_llm_chat_mode(text)
    elif mode == "source_detail":
        docs_to_use = memory.get("last_docs", [])
        if docs_to_use:
            reply = call_llm_with_context(text, docs_to_use)
        else:
            reply = "No tinc informació prèvia. Fes-me primer una pregunta general."
    else:  # summary
        docs_to_use = semantic_search(text, docs, embeddings, index, top_k=MAX_CONTEXT_DOCS)
        reply = call_llm_with_context(text, docs_to_use)
        # Afegir fonts al final
        titles = [docs[i].get("title", "") for i in docs_to_use]
        if titles:
            reply += "\n\nFonts: " + ", ".join(f"«{t}»" for t in titles if t)
        # Actualitzar memòria
        memory["last_docs"] = docs_to_use
        memory["active_title"] = titles[0] if titles else None
        user_memory[chat_id] = memory

    # Fragmentar i enviar
    await send_long_message(chat_id, reply, context)

# --- /mes handler ---
async def more_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    memory = user_memory.get(chat_id, {})
    if not memory.get("last_docs"):
        await update.message.reply_text("No hi ha res anterior per ampliar. Fes-me primer una pregunta general.")
        return

    active_title = memory.get("active_title", "")
    if not active_title:
        await update.message.reply_text("No tinc article actiu per ampliar.")
        return

    # Demanar més detalls sobre l'article actiu
    text = f"Explica més detalls sobre l'article: {active_title}"
    reply = call_llm_with_context(text, memory["last_docs"])
    reply += "\n\nContinua amb /mes si vols més informació."

    await send_long_message(chat_id, reply, context)

# --- Bot runner ---
def run_bot():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Posa TELEGRAM_TOKEN a l'entorn")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text(
        "Hola! Sóc l'assistent de la Ribera d'Ebre. Pregunta'm sobre el corpus o temes generals."
    )))
    app.add_handler(CommandHandler("mes", more_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, telegram_message_handler))

    stop_event = asyncio.Event()
    def _sigterm_handler(signum, frame):
        print("Tancant per senyal...")
        stop_event.set()
    signal.signal(signal.SIGINT, _sigterm_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("Bot engegat. Waiting for messages...")
    app.run_polling()

if __name__ == "__main__":
    run_bot()
