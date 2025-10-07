"""
Bot Telegram per preguntar sobre el corpus de la Ribera d'Ebre.
Requisits:
pip install python-telegram-bot==20.4 openai numpy faiss-cpu rapidfuzz python-dotenv

Configura a .env:
- TELEGRAM_TOKEN
- OPENAI_API_KEY
"""

import os, json, signal, asyncio, math
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
MAX_CHARS_PER_PAGE = 3500  # limit per message in mode cita

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
def find_exact_matches(query: str, docs, threshold=90):
    """
    Retorna índexs amb coincidència forta al title (o topics).
    """
    q = query.lower()
    matches = []
    for i, d in enumerate(docs):
        ttext = " ".join([d.get("title","")] + d.get("topics", [])).lower()
        score = fuzz.ratio(q, ttext)
        if score >= threshold:
            matches.append((i, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches]

# ---------- SEMANTIC SEARCH ----------
def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

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

def semantic_search(query: str, docs, embeddings=None, index=None, top_k=5):
    """
    Retorna els índexs dels documents més rellevants.
    """
    query_lower = query.lower()
    if index is not None and embeddings is not None:
        q_emb = np.array([get_embedding(query)], dtype=np.float32)
        D, I = index.search(q_emb, top_k)
        return [int(i) for i in I[0] if i != -1]
    
    # fallback text-match amb topics + title + summary
    scores = []
    for i, d in enumerate(docs):
        ttext = " ".join(d.get("topics", []) + [d.get("title",""), d.get("summary",""), d.get("summary_long","")]).lower()
        score = fuzz.token_set_ratio(query_lower, ttext)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i,_ in scores[:top_k]]
    # ---------- MEMÒRIA USUARI ----------
user_memory = {}

def push_user_memory(chat_id, question, answer, docs_used):
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_mode": None, "last_pages": {}})
    m["history"].append((question, answer))
    if len(m["history"]) > USER_MEMORY_SIZE:
        m["history"].pop(0)
    m["last_docs"] = docs_used

# ---------- LLM ----------
SYSTEM_PROMPT = """
Ets una IA experta en la història i temes dels pobles de la Ribera d'Ebre.
HAS DE RESPONDRE NOMÉS AMB LA INFORMACIÓ DEL CORPUS.
Si no pots respondre, diu 'No ho sé segons el corpus'.
- Pregunta específica: mostra l'article amb títol + summary + /mes
- Pregunta general: resumeix info de diversos articles, sense repetir
- Si l'usuari demana /mes, amplia amb summary_long
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

# ---------- FUNCIONS DE MODES ----------
def is_exact_title_match(text):
    matches = find_exact_matches(text, docs, threshold=95)
    return matches[0] if matches else None

def split_text_into_pages(text, max_chars=MAX_CHARS_PER_PAGE):
    pages = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # intentar tallar en salt de línia
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos != -1 and newline_pos > start:
                end = newline_pos
        pages.append(text[start:end].strip())
        start = end
    return pages

# ---------- TELEGRAM HANDLERS ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola! Sóc l'assistent de la Ribera d'Ebre. "
        "Pregunta'm sobre el corpus o temes generals (riuades, guerra civil, pobles...)."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    m = user_memory.get(chat_id, {})

    # ---------- Modes ----------
    # 1. Exact title / cita directa
    exact_idx = is_exact_title_match(text)
    if exact_idx is not None:
        doc = docs[exact_idx]
        summary = doc.get("summary","")
        # Dividir en pàgines
        pages = split_text_into_pages(summary)
        m["last_mode"] = "cite"
        m["last_docs"] = [exact_idx]
        m["last_pages"] = {"pages": pages, "current": 0}
        user_memory[chat_id] = m
        text_reply = f"**{doc.get('title')}**\n\n{pages[0]}"
        if len(pages) > 1:
            text_reply += f"\n\n(1/{len(pages)})\nVols que t'ampliï amb /mes?"
        await update.message.reply_text(text_reply)
        return

    # 2. Continuació /mes d'un article concret
    if text.startswith("/mes") and m.get("last_mode") == "cite":
        pages = m["last_pages"]["pages"]
        current = m["last_pages"]["current"]
        if current + 1 < len(pages):
            current += 1
            m["last_pages"]["current"] = current
            user_memory[chat_id] = m
            text_reply = f"{pages[current]}\n\n({current+1}/{len(pages)})"
            if current + 1 < len(pages):
                text_reply += "\nVols continuar amb /mes?"
            await update.message.reply_text(text_reply)
        else:
            await update.message.reply_text("Has arribat al final de l'article.")
        return

    # 3. Resum general
    await update.message.reply_text("..rumiant..")
    docs_to_use = semantic_search(text, docs, embeddings, index, top_k=MAX_CONTEXT_DOCS)
    reply = call_llm_with_context(text, docs_to_use)
    # Afegim fonts utilitzades
    fonts = [f"/{i+1} {docs[i].get('title','')}" for i in docs_to_use]
    reply += "\n\nFonts: " + ", ".join(fonts)
    m["last_mode"] = "summary"
    m["last_docs"] = docs_to_use
    user_memory[chat_id] = m
    await update.message.reply_text(reply)

# ---------- RUN BOT ----------
def run_bot():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Posa TELEGRAM_TOKEN a l'entorn")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
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

