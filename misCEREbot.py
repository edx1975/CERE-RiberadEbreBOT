import os
import json
import asyncio
import time
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import openai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
import faiss



load_dotenv()

# --- CONFIGURACIÓ ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "Ets un bot expert en la Ribera d'Ebre: Ginestar, Benissanet, Tivissa, Rasquera i Miravet. "
    "Respón amablement i també com ChatGPT."
)

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
TERM_MAP_FILE = os.path.join(DATA_DIR, "term_map_clean.json")
TOKEN_LOG_FILE = os.path.join(DATA_DIR, "token_log.json")

TOP_K = 5
CONTEXT_EXPIRY = 100  # segons
SVD_DIM = 100  # dimensions reduïdes TF-IDF

# --- MEMÒRIA TEMPORAL ---
user_context = {}  # clau: user_id, valor: {"last_topic", "last_embedding", "last_time", "topics_covered"}

# --- TERM MAP / SINÒNIMS ---
if os.path.exists(TERM_MAP_FILE):
    with open(TERM_MAP_FILE, "r", encoding="utf-8") as f:
        TERM_MAP = json.load(f)
else:
    TERM_MAP = {
        "indrets històrics": ["castell", "església", "pont", "monument", "ruïnes"],
        "munició": ["bales", "pistoles", "tancs"],
        "persones": ["habitants", "personatges", "figures"],
    }

# --- CARREGAR CORPUS ---
corpus = []
corpus_texts = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            corpus.append(entry)
            # concatenem camps per TF-IDF
            corpus_texts.append(" ".join([entry.get("title",""), entry.get("summary",""), entry.get("long_summary","")]))
        except json.JSONDecodeError:
            print(f"⚠️ Línia descartada: {line}")

# --- CARREGAR EMBEDDINGS EXISTENTS ---
if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✅ Carregats {embeddings.shape[0]} embeddings existents.")
else:
    embeddings = np.zeros((0, 1536), dtype=np.float32)

# --- CARREGAR HNSW FAISS ---
d = embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.efConstruction = 100
index.hnsw.efSearch = 100
if embeddings.shape[0] > 0:
    index.add(embeddings)


# --- TF-IDF + SVD lleuger ---
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(corpus_texts)
svd = TruncatedSVD(n_components=min(SVD_DIM, tfidf_matrix.shape[1]), random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

# --- FUNCIONS AUXILIARS ---
def needs_expansion(user_query: str) -> bool:
    triggers = ["detall", "detalls", "explica", "explicació", "amplia", "llarg", "més informació", "aprofund"]
    return any(t in user_query.lower() for t in triggers)

def expand_query(query):
    expanded_terms = [query]
    for key, synonyms in TERM_MAP.items():
        if key in query.lower():
            expanded_terms.extend(synonyms)
    return " ".join(expanded_terms)

def truncate_text(text, max_chars=1000):
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_punct = max(cut.rfind("."), cut.rfind("\n"), cut.rfind("!"), cut.rfind("?"))
    if last_punct > 50:
        return cut[:last_punct+1]
    last_space = cut.rfind(" ")
    return cut[:last_space] if last_space > 50 else cut

def log_tokens(user_id, tokens_used, cost):
    if os.path.exists(TOKEN_LOG_FILE):
        with open(TOKEN_LOG_FILE, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {}
    if str(user_id) not in log:
        log[str(user_id)] = {"tokens": 0, "euros": 0.0}
    log[str(user_id)]["tokens"] += tokens_used
    log[str(user_id)]["euros"] += cost
    with open(TOKEN_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def clean_expired_context():
    now = time.time()
    expired_users = [uid for uid, ctx in user_context.items()
                     if now - ctx.get("last_time", 0) > CONTEXT_EXPIRY]
    for uid in expired_users:
        del user_context[uid]

def semantic_search(query, top_k=TOP_K, topics=None, population=None):
    # expandim amb sinònims
    q_text = expand_query(query)

    # Embedding
    resp = openai.embeddings.create(input=q_text, model="text-embedding-3-small")
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)

    # FAISS search
    D, I = index.search(q_emb, top_k*3)
    candidates = [corpus[i] for i in I[0]]

    # Filtrat per població
    if population:
        pop_fragments = [f for f in candidates if population.lower() in f.get("population","").lower()]
        other_fragments = [f for f in candidates if population.lower() not in f.get("population","").lower()]
        candidates = pop_fragments + other_fragments

    # Filtrat per topics + TF-IDF+SVD semàntic
    if topics:
        filtered = []
        for f in candidates:
            text = " ".join([f.get("title",""), f.get("summary",""), f.get("long_summary","")])
            tfidf_vec = vectorizer.transform([text])
            svd_vec = svd.transform(tfidf_vec)
            score = 0
            for t in topics:
                if t.lower() in text.lower():
                    score += 1
            if score > 0:
                filtered.append(f)
        candidates = filtered if filtered else candidates

    # Boost títol
    def boost_score(f):
        score = 0
        title = f.get("title","").lower()
        if any(word in title for word in query.lower().split()):
            score -= 1
        return score

    candidates.sort(key=boost_score)
    return [f for f in candidates if f.get("summary") or f.get("long_summary")][:top_k]

def summarize_fragments(fragments, expand=False):
    parts = []
    for f in fragments:
        base = f"📌 {f.get('title','')}\n{truncate_text(f.get('summary',''),800)}"
        if expand and f.get("long_summary"):
            base += f"\n🔎 Detalls: {truncate_text(f['long_summary'],1000)}"
        parts.append(base)
    return "\n\n".join(parts)

# --- FUNCIO PRINCIPAL DEL BOT ---
def ask_openai(prompt, user_id=None, strict_corpus=True, population=None):
    clean_expired_context()

    # Extreure topics
    try:
        resp_topics = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":f"Extreu fins a 5 temes principals d'aquest text per cerca semàntica, separats per comes:\n{prompt}"}],
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
            return "⚠️ No tinc informació concreta al corpus sobre això."
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
            return f"S'ha produït un error amb OpenAI: {e}"

    expand = needs_expansion(prompt)
    combined_text = summarize_fragments(fragments, expand=expand)

    try:
        emb_topic = np.array(openai.embeddings.create(input=prompt, model="text-embedding-3-small").data[0].embedding, dtype=np.float32)
        if user_id:
            if user_id not in user_context:
                user_context[user_id] = {"last_topic":prompt,"last_embedding":emb_topic,"last_time":time.time(),"topics_covered":set(topics)}
            else:
                user_context[user_id]["last_topic"]=prompt
                user_context[user_id]["last_embedding"]=emb_topic
                user_context[user_id]["last_time"]=time.time()
                user_context[user_id]["topics_covered"].update(topics)
    except Exception:
        pass

    user_prompt = f"A partir d'aquests fragments del corpus, resumeix i extreu informació rellevant.\nSi és possible, crea llistes, destaca llocs històrics o esdeveniments, i fes-ho coherent:\n\n{combined_text}\n\nResposta a la pregunta de l'usuari: {prompt}"

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
        "🤖 Hola! Sóc un bot amb coneixements de la Miscel·lània del CERE de Ginestar, Benissanet, Tivissa, Rasquera i Miravet. Com puc ajudar-te avui?"
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
    port = int(os.environ.get("PORT",8080))
    class HealthHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path=="/":
                self.send_response(200)
                self.send_header("Content-type","text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_error(404,"Not Found")
    server = HTTPServer(("0.0.0.0",port), HealthHandler)
    print(f"🌐 Healthcheck HTTP server listening on port {port}")
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# --- INICI BOT ---
if __name__ == "__main__":
    print("🤖 Bot equilibrat amb FAISS HNSW, TF-IDF/SVD, memòria temporal i truncament segur actiu...")
    app.run_polling()
