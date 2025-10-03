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
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# --- CONFIGURACIÓ ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "Ets un bot expert en la Ribera d'Ebre: Ginestar, Benissanet, Tivissa, Rasquera i Miravet. "
    "Respón amablement i també com ChatGPT. "
    "Si l'usuari et diu hola, saluda com ChatGPT."
)

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
TOKEN_LOG_FILE = os.path.join(DATA_DIR, "token_log.json")

TOP_K = 5
RESUME_TOKENS = 150

# --- MEMÒRIA TEMPORAL ---
user_context = {}  # clau: user_id, valor: {"last_topic": str, "last_embedding": np.array, "last_time": timestamp, "topics_covered": set()}
CONTEXT_EXPIRY = 300  # segons abans que s'esborri el context (5 minuts)

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
            print(f"⚠️ Línia descartada: {line}")

# --- CARREGAR O CALCULAR EMBEDDINGS AMB LOG ---
if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✅ Carregats {embeddings.shape[0]} embeddings existents.")
else:
    embeddings = np.zeros((0, 1536), dtype=np.float32)  # si encara no existeix

existing_count = embeddings.shape[0]
new_docs = corpus[existing_count:]

print(f"ℹ️ Total documents al corpus: {len(corpus)}")
print(f"ℹ️ Documents amb embeddings existents: {existing_count}")
print(f"ℹ️ Documents nous per processar: {len(new_docs)}")

if new_docs:
    new_embeddings = []
    for i, entry in enumerate(new_docs, start=existing_count):
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        long_summary = entry.get("long_summary", "")
        topics = entry.get("topics", [])
        population = entry.get("population", "")
        years = entry.get("years", "")

        if not isinstance(topics, list):
            topics = [str(topics)]

        input_text = (
            f"{title}. "
            f"{summary}. "
            f"{long_summary}. "
            f"Topics: {', '.join(topics)}. "
            f"Població: {population}. "
            f"Període: {years}."
        )

        try:
            resp = openai.embeddings.create(
                input=input_text,
                model="text-embedding-3-small"
            )
            new_embeddings.append(resp.data[0].embedding)
            print(f"✅ Embedding calculat per document {i}: {title[:50]}...")
        except Exception as e:
            print(f"⚠️ Error calculant embedding per document {i}: {title[:50]}... -> {e}")

    if new_embeddings:
        new_embeddings = np.array(new_embeddings, dtype=np.float32)
        embeddings = np.vstack([embeddings, new_embeddings])
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"✅ Embeddings nous afegits. Total embeddings actuals: {embeddings.shape[0]}")

# --- CARREGAR FAISS ---
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# --- FUNCIONS AUXILIARS ---
def needs_expansion(user_query: str) -> bool:
    triggers = ["detall", "detalls", "explica", "explicació", "amplia", "llarg", "més informació", "aprofund"]
    return any(t in user_query.lower() for t in triggers)

def semantic_search(query, top_k=TOP_K, topics=None, population=None):
    resp = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    D, I = index.search(q_emb, top_k * 3)
    candidates = [corpus[i] for i in I[0]]

    if population:
        pop_fragments = [f for f in candidates if population.lower() in f.get("population", "").lower()]
        other_fragments = [f for f in candidates if population.lower() not in f.get("population", "").lower()]
        candidates = pop_fragments + other_fragments

    if topics:
        filtered = [
            f for f in candidates
            if any(
                t.lower() in f.get("summary", "").lower() or
                t.lower() in f.get("long_summary", "").lower() or
                t.lower() in [x.lower() for x in f.get("topics", [])]
                for t in topics
            )
        ]
        candidates = filtered if filtered else candidates

    return [f for f in candidates if f.get("summary") or f.get("long_summary")][:top_k]

def summarize_fragments(fragments, expand=False):
    parts = []
    for f in fragments:
        base = f"📌 {f.get('title','')}\n{f.get('summary','')}"
        if expand and f.get("long_summary"):
            base += f"\n🔎 Detalls: {f['long_summary']}"
        parts.append(base)
    return "\n\n".join(parts)

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

# --- FUNCIO PRINCIPAL DEL BOT ---
def ask_openai(prompt, user_id=None, strict_corpus=True, population=None):
    if not population and user_id and user_id in user_context:
        last_topic = user_context[user_id].get("last_topic", "")
        for poble in ["Ginestar", "Benissanet", "Tivissa", "Rasquera", "Miravet"]:
            if poble.lower() in last_topic.lower():
                population = poble
                break

    try:
        resp_topics = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Extreu fins a 5 temes principals d'aquest text per fer cerca semàntica, separats per comes:\n{prompt}"
            }],
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"S'ha produït un error amb OpenAI: {e}"

    expand = needs_expansion(prompt)
    summary = summarize_fragments(fragments, expand=expand)

    try:
        emb_topic = np.array(openai.embeddings.create(
            input=prompt,
            model="text-embedding-3-small"
        ).data[0].embedding, dtype=np.float32)
        if user_id:
            if user_id not in user_context:
                user_context[user_id] = {
                    "last_topic": prompt,
                    "last_embedding": emb_topic,
                    "last_time": time.time(),
                    "topics_covered": set(topics)
                }
            else:
                user_context[user_id]["last_topic"] = prompt
                user_context[user_id]["last_embedding"] = emb_topic
                user_context[user_id]["last_time"] = time.time()
                user_context[user_id]["topics_covered"].update(topics)
    except Exception:
        pass

    user_prompt = (
        f"Utilitza només la informació següent extreta dels arxius del CERE:\n\n{summary}\n\n"
        f"Resposta a la pregunta de l'usuari: {prompt}"
    )

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
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

# --- INICI BOT ---
if __name__ == "__main__":
    print("🤖 Bot amb corpus prioritzat, memòria temporal i resums actius...")
    app.run_polling()
