"""
Bot Telegram per preguntar sobre el corpus de la Ribera d'Ebre.
Requisits:
pip install python-telegram-bot==20.4 openai numpy faiss-cpu rapidfuzz python-dotenv

Configura a .env:
- TELEGRAM_TOKEN
- OPENAI_API_KEY
"""

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

# ---------- COINCIDÈNCIES EXACTES ----------
def find_exact_matches(query: str, docs, threshold=80):
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

def semantic_search(query: str, docs, embeddings=None, index=None, top_k=5, town=None):
    """
    Retorna els índexs dels documents més rellevants.
    Filtra per poble si town és indicat.
    """
    filtered_docs = []
    filtered_indexes = []

    for i, d in enumerate(docs):
        towns = [t.lower() for t in d.get("population", [])]
        if town:
            if town.lower() in towns:
                filtered_docs.append(d)
                filtered_indexes.append(i)
        else:
            filtered_docs.append(d)
            filtered_indexes.append(i)

    query_lower = query.lower()

    # Si FAISS i embeddings existeixen
    if index is not None and embeddings is not None:
        q_emb = np.array([get_embedding(query)], dtype=np.float32)
        D, I = index.search(q_emb, top_k)
        # Només retornem índexs dins filtered_indexes
        result = [i for i in I[0] if i != -1 and i in filtered_indexes]
        return result[:top_k]
    
    print(f"[semantic_search] town={town}, query='{query}', total_docs={len(docs)}")

    # fallback text-match amb topics, title, summary, summary_long
    scores = []
    for idx, d in zip(filtered_indexes, filtered_docs):
        ttext = " ".join(
            d.get("topics", []) + [d.get("title",""), d.get("summary",""), d.get("summary_long","")]
        ).lower()
        score = fuzz.token_set_ratio(query_lower, ttext)
        scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i,_ in scores[:top_k]]

    print(f"[semantic_search] Fallback scores top5={scores[:5]}")


# ---------- Detecció de poble i tema ----------
def detect_town(question, docs):
    """
    Retorna el poble detectat dins de la pregunta.
    Busca coincidències amb la llista de pobles del corpus (campo 'population' o 'town').
    """
    q_lower = question.lower()
    for d in docs:
        towns = [t.lower() for t in d.get("population", [])]
        for t in towns:
            if t in q_lower:
                return t
    return None

def detect_topic(question, keywords=None):
    """
    Retorna el tema detectat segons paraules clau.
    """
    if not keywords:
        keywords = ["riuades", "església", "guerra civil", "castell", "riu", "pobles", "moli", "mercat"]
    q_lower = question.lower()
    for kw in keywords:
        if kw in q_lower:
            return kw
    return None

def update_context(memory, detected_town, detected_topic):
    """
    Si poble o tema canvia, reinicia el context de memòria.
    """
    reset_context = False

    if detected_town and detected_town != memory.get("last_town"):
        memory["last_town"] = detected_town
        reset_context = True

    if detected_topic and detected_topic != memory.get("last_topic"):
        memory["last_topic"] = detected_topic
        reset_context = True

    if reset_context:
        memory["last_docs"] = []
        memory["active_doc"] = None
        memory["current_page"] = 0
        memory["last_mode"] = "summary"


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
    print(f"[call_llm_with_context] user_query='{user_query}', doc_indexes={doc_indexes}")
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

def detect_mode(text, memory):
    t = text.lower()
    # Mode conversa casual
    if any(x in t for x in ["com estàs", "què et sembla", "ets", "tu", "com et trobes"]):
        return "chat"
    # Mode /mes o ampliació de fonts
    elif "/mes" in t or (memory and memory.get("last_docs")):
        return "source_detail"
    else:
        return "summary"

# ---------- MEMÒRIA USUARI ----------
user_memory = {}

def push_user_memory(chat_id, question, answer, docs_used, active_doc=None, last_mode="summary"):
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_mode": last_mode, "active_doc": active_doc, "current_page": 0})
    m["history"].append((question, answer))
    if len(m["history"]) > USER_MEMORY_SIZE:
        m["history"].pop(0)
    m["last_docs"] = docs_used
    m["last_mode"] = last_mode
    m["active_doc"] = active_doc
    m["current_page"] = 0
    
# ---------- TELEGRAM HANDLERS ----------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    print(f"[{chat_id}] user: {text}")

    # ---------- Memòria ----------
    m = user_memory.setdefault(chat_id, {
        "history": [],
        "last_docs": [],
        "last_mode": "summary",
        "active_doc": None,
        "last_town": None,
        "last_topic": None
    })

    # ---------- Salutacions ----------
    salutacions = ["hola", "bon dia", "bona tarda", "bona nit", "ei"]
    if any(text.lower().startswith(s) for s in salutacions):
        await update.message.reply_text(
            "Hola! Sóc un bot del Miscel·lània del CERE. Ara mateix tinc info general de la Ribera i concreta de Benissanet, Miravet, Rasquera, Ginestar i Tivissa. Com et puc ajudar avui?"
        )
        return

    # ---------- Detectem poble i tema ----------
    detected_town = detect_town(text, docs)
    detected_topic = detect_topic(text)

    # --- Mantenim poble si s'infereix pel context ---
    if not detected_town and m.get("last_town"):
        if re.search(r"\b(seu|seva|seves|seus|del poble|del municipi|de la vila|la seva|i a|i el|i la)\b", text.lower()):
            detected_town = m["last_town"]
        elif len(text.split()) <= 5:
            detected_town = m["last_town"]

    # --- Mantenim tema si s'infereix pel context ---
    if not detected_topic and m.get("last_topic"):
        if re.search(r"\b(i|també|altres|l’altra|el mateix|a més)\b", text.lower()):
            detected_topic = m["last_topic"]
        elif detected_town and not detect_topic(text):
            detected_topic = m["last_topic"]

    # Actualitzem memòria
    if detected_town:
        m["last_town"] = detected_town
    if detected_topic:
        m["last_topic"] = detected_topic

    # --- 🔍 DEBUG: mostra quin context s'està usant ---
    print(f"[{chat_id}] → Context actiu: town='{m.get('last_town')}', topic='{m.get('last_topic')}'")

    # Combinar context per reforçar la cerca semàntica
    query_text = text
    if detected_town and detected_topic:
        query_text += f" {detected_town} {detected_topic}"
    elif detected_town:
        query_text += f" {detected_town}"
    elif detected_topic:
        query_text += f" {detected_topic}"

    update_context(m, detected_town, detected_topic)

    # ---------- Detectar si l'usuari vol un article concret (/n o títol exacte) ----------
    doc_idx = None
    if text.startswith("/"):
        try:
            idx = int(text[1:]) - 1
            if 0 <= idx < len(docs):
                doc_idx = idx
        except ValueError:
            pass
    else:
        for i, d in enumerate(docs):
            if text.strip().lower() == d.get("title", "").lower():
                doc_idx = i
                break

    if doc_idx is not None:
        doc = docs[doc_idx]
        summary = doc.get("summary", "")
        if len(summary) > 3500:
            summary = summary[:3500] + "…"
        reply = f"Segons l'article «{doc.get('title')}»\n\nResum: {summary}\n\nVols que t'ampliï amb /mes?"
        m["last_mode"] = "source_detail"
        m["active_doc"] = doc_idx
        m["current_page"] = 0
        await update.message.reply_text(reply)
        return

    # ---------- Mode resum general ----------
    await update.message.reply_text("..rumiant..")
    
    print(f"[{chat_id}] DEBUG detectat → town={detected_town}, topic={detected_topic}, query_text='{query_text}'")

    docs_to_use = semantic_search(
        query_text, docs, embeddings, index,
        top_k=MAX_CONTEXT_DOCS, town=detected_town
    )
    reply = call_llm_with_context(query_text, docs_to_use)

    # ---------- Afegir fonts numerades ----------
    if docs_to_use:
        reply += "\n\nFonts (fes clic per ampliar):"
        for i, idx in enumerate(docs_to_use):
            title = docs[idx].get("title", "")
            reply += f"\n/{i+1}: {title}"

    # ---------- Cerca profunda si LLM diu que no ho sap ----------
    if "No ho sé segons el corpus" in reply and detected_town:
        snippet, doc_idx = deep_search_by_town(detected_town, docs)
        if snippet:
            doc = docs[doc_idx]
            reply = (
                f"Segons l'article «{doc.get('title')}» a {detected_town}:\n\n"
                f"{snippet}\n\nVols que t'ampliï amb /mes?"
            )
            m["active_doc"] = doc_idx
            m["last_mode"] = "source_detail"
            m["current_page"] = 0
            m["last_town"] = detected_town

    # ---------- Actualitzar memòria ----------
    m["last_docs"] = docs_to_use
    if m.get("last_mode") != "source_detail":
        m["last_mode"] = "summary"
        m["active_doc"] = None

    await update.message.reply_text(reply)


# ---------- Handler per /1, /2, /3, ... ----------
async def numbered_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    command = update.message.text.strip()
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_mode": "summary", "active_doc": None, "current_page": 0})

    import re
    match = re.match(r"/(\d+)", command)
    if not match:
        return
    num = int(match.group(1))
    last_docs = m.get("last_docs", [])
    if num < 1 or num > len(last_docs):
        await update.message.reply_text("No reconec aquesta font. Torna-ho a provar després d'una cerca.")
        return

    doc_idx = last_docs[num - 1]
    doc = docs[doc_idx]
    summary = doc.get("summary","")
    if len(summary) > 3500:
        summary = summary[:3500] + "…"

    m["active_doc"] = doc_idx
    m["last_mode"] = "source_detail"
    m["current_page"] = 0

    await update.message.reply_text(
        f"Segons l'article «{doc.get('title')}»\n\nResum: {summary}\n\nVols veure el resum sencer de l'article amb /mes?"
    )


# ---------- /mes handler ----------
async def more_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    m = user_memory.get(chat_id, {})

    if not m or m.get("last_mode") != "source_detail" or m.get("active_doc") is None:
        await update.message.reply_text("No tinc context previ d'un article concret. Digues-me sobre què vols informació.")
        return

    idx = m["active_doc"]
    doc = docs[idx]
    long_text = doc.get("summary_long", "")
    author = doc.get("author", "Autor desconegut")

    chunk_size = 3500
    start = m.get("current_page", 0) * chunk_size
    end = start + chunk_size
    chunk = long_text[start:end]

    total_pages = (len(long_text) - 1) // chunk_size + 1
    page_number = m.get("current_page", 0) + 1

    if chunk:
        m["current_page"] += 1
        if page_number == total_pages:
            await update.message.reply_text(f"{chunk}\n\nFinal de l’article.\nAutor de l'article original: {author}")
            m["current_page"] = 0
        else:
            await update.message.reply_text(f"{chunk}\n\n({page_number}/{total_pages})\nVols continuar amb /mes?")
    else:
        await update.message.reply_text("He mostrat tot el contingut de l'article.")
        m["current_page"] = 0


# ---------- Handler per /1, /2, /3, ... ----------
async def numbered_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    command = update.message.text.strip()
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_mode": "summary", "active_doc": None, "current_page": 0})

    import re
    match = re.match(r"/(\d+)", command)
    if not match:
        return
    num = int(match.group(1))
    last_docs = m.get("last_docs", [])
    if num < 1 or num > len(last_docs):
        await update.message.reply_text("No reconec aquesta font. Torna-ho a provar després d'una cerca.")
        return

    doc_idx = last_docs[num - 1]
    doc = docs[doc_idx]
    summary = doc.get("summary","")
    if len(summary) > 3500:
        summary = summary[:3500] + "…"

    m["active_doc"] = doc_idx
    m["last_mode"] = "source_detail"
    m["current_page"] = 0

    await update.message.reply_text(
        f"Segons l'article «{doc.get('title')}»\n\nResum: {summary}\n\nVols veure el resum sencer de l'article amb /mes?"
    )


def deep_search_by_town(town_name, docs):
    """
    Busca coincidències dins del long_summary dels articles que contenen el nom del poble.
    Retorna el primer fragment rellevant i l'índex del document.
    """
    town_lower = town_name.lower()
    for i, d in enumerate(docs):
        # Comprovem si el poble apareix en topics, title o summary llarg
        combined_text = " ".join(d.get("topics", []) + [d.get("title",""), d.get("summary_long","")]).lower()
        if town_lower in combined_text:
            # Busquem el fragment dins del long_summary
            long_text = d.get("summary_long","")
            # Opcional: agafem les primeres 500-1000 caràcters que continguin el poble
            idx = long_text.lower().find(town_lower)
            if idx != -1:
                start = max(0, idx - 50)
                end = min(len(long_text), idx + 300)
                snippet = long_text[start:end].strip()
                return snippet, i
    return None, None

# ---------- START HANDLER ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hola! Sóc l'assistent de la Ribera d'Ebre. Pregunta'm sobre el corpus o temes generals.")


# ---------- RUN BOT ----------
def run_bot():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Posa TELEGRAM_TOKEN a l'entorn")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mes", more_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.COMMAND, numbered_command_handler))

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
