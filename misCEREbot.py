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

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus_original.jsonl")
EMB_NPY = os.path.join(DATA_DIR, "embeddings_G.npy")
FAISS_IDX = os.path.join(DATA_DIR, "faiss_index_G.index")

TOP_K = 5
MAX_CONTEXT_DOCS = 6
USER_MEMORY_SIZE = 6
MAX_TOKENS = 3500

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
user_memory = {}

def push_user_memory(chat_id, question, answer, docs_used):
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_title": None, "last_mode": None})
    m["history"].append((question, answer))
    if len(m["history"]) > USER_MEMORY_SIZE:
        m["history"].pop(0)
    m["last_docs"] = docs_used

# ---------- UTILS ----------
def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def semantic_search(query: str, docs, embeddings=None, index=None, top_k=5):
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
    parts = []
    for idx in doc_indexes:
        d = docs[idx]
        parts.append(f"== ARTICLE {idx} ==\nTitle: {d.get('title')}\nSummary: {d.get('summary')}\nLong: {d.get('summary_long')}\n")
    return "\n\n".join(parts)

def call_llm_with_context(user_query, doc_indexes, temperature=0.0, max_tokens=800):
    context_text = build_context_for_docs(doc_indexes)
    prompt = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user", "content": f"Context documents:\n\n{context_text}\n\nPregunta: {user_query}\n\nRespon en català. Si et manca info, diu 'No ho sé segons el corpus'."}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# ---------- WRAPPER TELEGRAM ----------
async def telegram_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    
    # Mode conversa casual
    salutacions = ["hola","bon dia","bona tarda","bona nit","ei"]
    if any(text.lower().startswith(s) for s in salutacions):
        await update.message.reply_text("Hola! Com et puc ajudar avui?")
        return
    
    # Mostra "..rumiant.." abans del resum
    await update.message.reply_text("..rumiant..")
    
    # Generem la resposta
    reply = handle_message(chat_id, text, docs, embeddings, index)
    await update.message.reply_text(reply)

# ---------- FUNCIO GENERAL handle_message ----------
def handle_message(chat_id, text, docs, embeddings, index):
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_title": None, "last_mode": None})
    text_lower = text.lower()

    # Detecta si l'usuari escriu /id o títol exacte
    is_direct = False
    doc_indexes = []

    # Mode cita directa si /n
    if text_lower.startswith("/"):
        try:
            n = int(text_lower[1:])-1
            if 0 <= n < len(docs):
                is_direct = True
                doc_indexes = [n]
        except:
            pass
    else:
        # Comprovem si coincideix exactament amb algun title
        for idx, d in enumerate(docs):
            if text.strip().lower() == d.get("title","").lower():
                is_direct = True
                doc_indexes = [idx]
                break

    if is_direct and doc_indexes:
        # Mode cita directa
        d = docs[doc_indexes[0]]
        summary = d.get("summary","")
        m["last_docs"] = doc_indexes
        m["last_title"] = d.get("title","")
        m["last_mode"] = "direct"
        return f"**{d.get('title','')}**\n\n{summary}\n\nVols que t'ampliï amb /mes?"
    
    # Mode resum general
    doc_indexes = semantic_search(text, docs, embeddings, index, top_k=MAX_CONTEXT_DOCS)
    summaries = [docs[i].get("summary","") for i in doc_indexes]
    combined_summary = "\n\n".join(summaries)
    # Afegim fonts utilitzades
    fonts = ", ".join(f"/{i+1} {docs[i].get('title','')}" for i in doc_indexes)
    m["last_docs"] = doc_indexes
    m["last_mode"] = "summary"
    return f"{combined_summary}\n\nFonts: {fonts}"

# ---------- /mes handler ----------
async def more_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    m = user_memory.get(chat_id)
    if not m or not m.get("last_docs") or m.get("last_mode") != "direct":
        await update.message.reply_text("No tinc context previ. Digues-me sobre què vols informació.")
        return
    idx = m["last_docs"][0]
    d = docs[idx]
    long_text = d.get("summary_long","")
    
    # Dividim per pàgines de 3500 chars
    chunk_size = 3500
    if "current_page" not in m:
        m["current_page"] = 0
    start = m["current_page"] * chunk_size
    end = start + chunk_size
    chunk = long_text[start:end]
    total_pages = (len(long_text)//chunk_size)+1
    page_number = m["current_page"]+1
    m["current_page"] += 1
    await update.message.reply_text(f"{chunk}\n\n({page_number}/{total_pages})\nVols continuar?")

# ---------- RUN BOT ----------
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
