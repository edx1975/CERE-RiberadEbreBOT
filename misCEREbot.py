import os, json, signal, asyncio, re
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

# ---------- EMBEDDINGS & FAISS ----------
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

# ---------- CERCA ----------

def find_exact_matches(query: str, docs, threshold=80):
    q = query.lower()
    matches = []
    for i, d in enumerate(docs):
        ttext = " ".join(d.get("topics", []) + [d.get("title",""), d.get("summary",""), d.get("summary_long","")]).lower()
        score = fuzz.token_set_ratio(q, ttext)
        if score >= threshold:
            matches.append((i, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches]

def semantic_search(query: str, docs, embeddings=None, index=None, top_k=5):
    query_lower = query.lower()
    if index is not None and embeddings is not None:
        q_emb = np.array([get_embedding(query)], dtype=np.float32)
        D, I = index.search(q_emb, top_k)
        return [int(i) for i in I[0] if i != -1]
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
- Pregunta específica: mostra només l'article rellevant.
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
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "message_counter":0})
    m["history"].append((question, answer))
    if len(m["history"]) > USER_MEMORY_SIZE:
        m["history"].pop(0)
    m["last_docs"] = docs_used
    m["message_counter"] += 1


# ---------- MODES I DETECCIÓ ----------
def detect_mode(text, memory):
    t = text.lower()
    if any(x in t for x in ["com estàs","què et sembla","ets","tu","com et trobes"]):
        return "chat"
    elif re.match(r"^/\d+$", t) and memory and memory.get("last_docs"):
        return "source_detail_id"
    elif "/mes" in t and memory and memory.get("last_docs"):
        return "more"
    else:
        return "summary"

# ---------- HANDLER PRINCIPAL ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    
    # Memòria i comptador missatges
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "message_counter":0})
    m["message_counter"] += 1
    msg_index = m["message_counter"]
    
    # Salutacions
    salutacions = ["hola","bon dia","bona tarda","bona nit","ei","ep","xeic","xéc"]
    if any(text.lower().startswith(s) for s in salutacions):
        await update.message.reply_text(f"[{msg_index}] Hola! Com et puc ajudar avui?")
        return
    
    # Detectem mode
    mode = detect_mode(text, m)
    
    if mode == "chat":
        reply = call_llm_chat_mode(text)
    elif mode == "more":
        last_docs = m.get("last_docs", [])
        user_q = "Expandeix la resposta anterior i afegeix títols de les fonts utilitzades."
        reply = call_llm_with_context(user_q, last_docs, temperature=0.0, max_tokens=1200)
    elif mode == "source_detail_id":
        n = int(text[1:]) - 1
        last_docs = m.get("last_docs", [])
        if 0 <= n < len(last_docs):
            idx = last_docs[n]
            doc = docs[idx]
            short = doc.get("summary","")
            title = doc.get("title","")
            reply = f"[{msg_index}] Segons l'article «{title}»:\n\n{short}\n\nVols que t'ampliï amb /mes?"
            m["last_docs"] = [idx]  # només aquest article ara
        else:
            reply = f"[{msg_index}] Número de font desconegut. Tria /1, /2, ..."
    else:  # summary general
        docs_to_use = semantic_search(text, docs, embeddings, index, top_k=MAX_CONTEXT_DOCS)
        reply = call_llm_with_context(text, docs_to_use)
        if docs_to_use:
            # Afegim fonts resumides amb /n
            fonts_list = [f"/{i+1}" for i in range(len(docs_to_use))]
            titles = [docs[i].get("title","") for i in docs_to_use]
            reply += "\n\nFonts: " + ", ".join(f"{fonts_list[i]} {titles[i]}" for i in range(len(fonts_list)))
        m["last_docs"] = docs_to_use

    # Resposta amb index missatge
    await update.message.reply_text(f"[{msg_index}] {reply}")

# ---------- /start ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola! Sóc l'assistent de la Ribera d'Ebre. "
        "Pregunta'm sobre el corpus o temes generals (riuades, guerra civil, pobles...)."
    )

# ---------- /mes ----------
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
    
    m["message_counter"] += 1
    msg_index = m["message_counter"]
    await update.message.reply_text(f"[{msg_index}] {reply}")

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

