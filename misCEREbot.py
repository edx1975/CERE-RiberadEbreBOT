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

# ---------- EMBEDDINGS ----------
def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

# ---------- SEMANTIC SEARCH ----------
def semantic_search(query: str, docs, embeddings=None, index=None, top_k=5):
    query_lower = query.lower()
    if index is not None and embeddings is not None:
        q_emb = np.array([get_embedding(query)], dtype=np.float32)
        D, I = index.search(q_emb, top_k)
        return [int(i) for i in I[0] if i != -1]
    
    # fallback fuzzy
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

# ---------- MEMÒRIA ----------
user_memory = {}

def push_user_memory(chat_id, question, answer, docs_used):
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_mode": "summary"})
    m["history"].append((question, answer))
    if len(m["history"]) > USER_MEMORY_SIZE:
        m["history"].pop(0)
    m["last_docs"] = docs_used
    m["last_mode"] = "summary"

# ---------- TELEGRAM HANDLERS ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola! Sóc l'assistent de la Ribera d'Ebre. "
        "Pregunta'm sobre el corpus o temes generals (riuades, guerra civil, pobles...)."
    )

def push_user_memory(chat_id, question, answer, docs_used):
    """Actualitza la memòria de l'usuari amb pregunta, resposta i docs."""
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "message_counter": 0})
    m["history"].append((question, answer))
    if len(m["history"]) > USER_MEMORY_SIZE:
        m["history"].pop(0)
    m["last_docs"] = docs_used
    # Incrementa el contador de missatges
    m["message_counter"] += 1

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    m = user_memory.get(chat_id, {})

    # Detectar mode: conversa casual, resum general, o cita directa
    mode = detect_mode(text, m)

    if mode == "chat":
        reply = call_llm_chat_mode(text)
        docs_to_use = []

    elif mode == "source_detail":
        docs_to_use = m.get("last_docs", [])
        if not docs_to_use:
            docs_to_use = semantic_search(text, docs, embeddings, index)
        reply = call_llm_with_context(text, docs_to_use)

    else:  # summary general
        docs_to_use = semantic_search(text, docs, embeddings, index)
        reply = call_llm_with_context(text, docs_to_use)
        # Afegir fonts amb ID
        if docs_to_use:
            reply += "\n\nFonts utilitzades:\n" + "\n".join(f"/{i+1}" for i in docs_to_use)

    # Actualitza memòria
    push_user_memory(chat_id, text, reply, docs_to_use)

    # Afegir índex de missatge [x/y]
    counter = user_memory[chat_id]["message_counter"]
    total = len(user_memory[chat_id]["history"])
    reply += f"\n\n[{counter}/{total}]"

    await update.message.reply_text(reply)

async def more_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    m = user_memory.get(chat_id)
    if not m or not m.get("last_docs"):
        await update.message.reply_text("No tinc context previ. Digues-me sobre què vols informació.")
        return

    last_docs = m["last_docs"]
    user_q = "Expandeix la resposta anterior i afegeix títols de les fonts utilitzades."
    try:
        reply = call_llm_with_context(user_q, last_docs, temperature=0.0, max_tokens=1200)
    except Exception as e:
        reply = f"S'ha produït un error: {e}"

    # Actualitza memòria i comptador
    push_user_memory(chat_id, "/mes", reply, last_docs)

    counter = user_memory[chat_id]["message_counter"]
    total = len(user_memory[chat_id]["history"])
    reply += f"\n\n[{counter}/{total}]"

    await update.message.reply_text(reply)


# ---------- RUN BOT ----------
def run_bot():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Posa TELEGRAM_TOKEN a l'entorn")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
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
