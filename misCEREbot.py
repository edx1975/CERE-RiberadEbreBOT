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
from rapidfuzz import fuzz, process
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

# ---------- COINCIDÈNCIES EXACTES ----------
def find_exact_matches(query: str, docs, threshold=80):
    """
    Retorna índexs amb coincidència forta al title, summary, summary_long o topics.
    """
    q = query.lower()
    matches = []
    for i, d in enumerate(docs):
        ttext = " ".join(
            d.get("topics", []) +
            [d.get("title",""), d.get("summary",""), d.get("summary_long","")]
        ).lower()
        score = fuzz.token_set_ratio(q, ttext)
        if score >= threshold:
            matches.append((i, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches]

# ---------- SEMANTIC SEARCH ----------
def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

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

# Detector de mode
def detect_mode(text, memory):
    t = text.lower()
    # Mode conversa casual
    if any(x in t for x in ["com estàs", "què et sembla", "ets", "tu", "com et trobes"]):
        return "chat"
    # Mode /mes o ampliació de fonts
    elif "/mes" in t or (memory and memory.get("last_docs")):
        return "source_detail"
    # Mode resum general
    else:
        return "summary"

# Funció per a mode conversa
def call_llm_chat_mode(user_query):
    prompt = [
        {"role":"system","content":"Ets una IA amable i propera del territori de la Ribera d’Ebre. Parla en català occidental (Terres de l’Ebre)."},
        {"role":"user","content": user_query}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.7,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# Funció general amb context (summary o source_detail)
def call_llm_with_context(user_query, docs):
    context_text = "\n".join(d.get("content","") for d in docs)
    prompt = [
        {"role":"system","content":"Ets una IA experta, amable i precisa. Resumeix la informació amb claredat i inclou cites si escau."},
        {"role":"user","content": f"{user_query}\n\nContext:\n{context_text}"}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.2,
        max_tokens=400
    )
    return resp.choices[0].message.content.strip()

def semantic_search(query: str, docs, embeddings=None, index=None, top_k=5):
    """
    Retorna els índexs dels documents més rellevants.
    1) Si FAISS i embeddings existeixen, fem cerca semàntica.
    2) Si no, fem fallback fuzzy sobre title + summary + topics.
    """
    query_lower = query.lower()
    
    if index is not None and embeddings is not None:
        q_emb = np.array([get_embedding(query)], dtype=np.float32)
        D, I = index.search(q_emb, top_k)
        return [int(i) for i in I[0] if i != -1]
    
    # fallback text-match amb topics
    scores = []
    for i, d in enumerate(docs):
        ttext = " ".join(
            d.get("topics", []) + [d.get("title",""), d.get("summary",""), d.get("summary_long","")]
        ).lower()
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

# ---------- MEMÒRIA USUARI ----------
user_memory = {}

def push_user_memory(chat_id, question, answer, docs_used):
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": []})
    m["history"].append((question, answer))
    if len(m["history"]) > USER_MEMORY_SIZE:
        m["history"].pop(0)
    m["last_docs"] = docs_used
    
# ---------- TELEGRAM HANDLERS ----------
from rapidfuzz import process

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    print(f"[{chat_id}] user: {text}")

    # ---------- Memòria ----------
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_mode": "summary"})

    # ---------- Salutacions ----------
    salutacions = ["hola", "bon dia", "bona tarda", "bona nit", "ei"]
    if any(text.lower().startswith(s) for s in salutacions):
        await update.message.reply_text("Hola! Com et puc ajudar avui?")
        return

    # ---------- Detectar si l'usuari vol un article concret (/id o títol) ----------
    doc_idx = None
    # 1️⃣ Comprovar si és /n
    if text.startswith("/"):
        try:
            idx = int(text[1:]) - 1
            if 0 <= idx < len(m.get("last_docs", [])):
                doc_idx = m["last_docs"][idx]
        except ValueError:
            pass
    # 2️⃣ Comprovar coincidència amb títol (fuzzy)
    if doc_idx is None:
        candidates = [(i, docs[i].get("title","")) for i in range(len(docs))]
        match = process.extractOne(text, dict(candidates), scorer=fuzz.token_set_ratio)
        if match and match[1] >= 80:  # threshold ajustable
            doc_idx = match[2]  # index real

    # ---------- Mode cita directa si hi ha doc_idx ----------
    if doc_idx is not None:
        doc = docs[doc_idx]
        reply = f"Segons l'article «{doc.get('title')}»:\n\n{doc.get('summary')}\n\nVols que t'ampliï amb /mes?"
        m["last_mode"] = "source_detail"
        m["active_doc"] = doc_idx
        await update.message.reply_text(reply)
        return

    # ---------- Mode resum general ----------
    docs_to_use = semantic_search(text, docs, embeddings, index, top_k=MAX_CONTEXT_DOCS)
    reply = call_llm_with_context(text, docs_to_use)

    # Afegir fonts numerades al final
    if docs_to_use:
        reply += "\n\nFonts (fes clic per ampliar):"
        for i, idx in enumerate(docs_to_use):
            title = docs[idx].get("title", "")
            reply += f"\n/{i+1}: {title}"

    # ---------- Actualitzar memòria ----------
    m["last_docs"] = docs_to_use
    m["last_mode"] = "summary"

    await update.message.reply_text(reply)


    # ---------- Mode resum general ----------
    docs_to_use = semantic_search(text, docs, embeddings, index, top_k=MAX_CONTEXT_DOCS)
    reply = call_llm_with_context(text, docs_to_use)

    # Afegir fonts numerades
    if docs_to_use:
        reply += "\n\nFonts (fes clic per ampliar):"
        for i, idx in enumerate(docs_to_use):
            title = docs[idx].get("title", "")
            reply += f"\n/{i+1}: {title}"

    # ---------- Actualitzar memòria ----------
    m["last_docs"] = docs_to_use
    m["last_mode"] = "summary"

    await update.message.reply_text(reply)


async def more_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    m = user_memory.get(chat_id, {})
    
    if not m or "active_doc" not in m:
        await update.message.reply_text("No tinc context previ d'un article concret. Digues-me sobre què vols informació.")
        return

    doc_idx = m["active_doc"]
    user_q = "Expandeix la resposta anterior i afegeix títols de les fonts utilitzades."
    try:
        reply = call_llm_with_context(user_q, [doc_idx], temperature=0.0, max_tokens=1200)
    except Exception as e:
        reply = f"S'ha produït un error: {e}"

    await update.message.reply_text(reply)


    await update.message.reply_text(reply)

# ---------- RUN BOT ----------
def run_bot():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Posa TELEGRAM_TOKEN a l'entorn")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text(
        "Hola! Sóc l'assistent de la Ribera d'Ebre. Pregunta'm sobre el corpus o temes generals.")
    app.add_handler(CommandHandler("mes", more_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

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
