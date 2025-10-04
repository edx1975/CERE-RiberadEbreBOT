# bot_ribera_improved.py
import os
import json
import time
import asyncio
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

import numpy as np
import faiss
from dotenv import load_dotenv
import openai

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

# --- CONFIGURACIÓ ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
openai.api_key = OPENAI_API_KEY

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
TOKEN_LOG_FILE = os.path.join(DATA_DIR, "token_log.json")

TOP_K = 5
CONTEXT_EXPIRY = 5 * 60  # segons (ajusta si cal)
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"  # pots canviar a gpt-4o-mini si tens accés
MAX_RESPONSE_TOKENS = 420
TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "Ets un bot expert en la Ribera d'Ebre: Ginestar, Benissanet, Tivissa, Rasquera i Miravet. "
    "RESPON només amb informació PROVADA en els fragments següents extrets del corpus del CERE. "
    "Si la resposta no pot ser suportada per cap fragment, RESPON: \"No tinc informació al corpus sobre això.\". "
    "Sigues concís i dona referència als fragments usats (títol i breu resum)."
)

# --- ESTAT I MEMÒRIA ---
user_context = {}  # user_id -> {"last_time", "topics_covered": set(), "last_query": str}

# --- CARREGAR CORPUS I TEXTOS ---
corpus = []
corpus_texts = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            # assegura camps mínims
            entry.setdefault("title", f"fragment_{i}")
            entry.setdefault("summary", "")
            entry.setdefault("long_summary", "")
            entry.setdefault("population", "")
            entry.setdefault("id", str(i))
            corpus.append(entry)
            corpus_texts.append(" ".join([entry.get("title", ""), entry.get("summary", ""), entry.get("long_summary", "")]))
        except json.JSONDecodeError:
            print(f"⚠️ Línia descartada al corpus: {line[:120]}")

# --- TF-IDF (només per booster de coincidències textuals) ---
vectorizer = TfidfVectorizer(max_features=5000, stop_words='spanish')
if corpus_texts:
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)
else:
    tfidf_matrix = None

# --- CARREGAR / CREAR EMBEDDINGS ---
if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✅ Carregats {embeddings.shape[0]} embeddings.")
else:
    # Si no tens embeddings prèviament creats, cal generar-los. Intentem calcular per tots els fragments.
    embeddings = np.zeros((0, 1536), dtype=np.float32)

# Recalcula embeddings només si len(embeddings) != len(corpus)
if embeddings.shape[0] != len(corpus):
    print("ℹ️ Generant embeddings per al corpus (pot trigar).")
    new_embs = []
    for entry in corpus:
        text = " ".join([entry.get("title",""), entry.get("summary",""), entry.get("long_summary","")])
        try:
            resp = openai.embeddings.create(input=text, model=EMBED_MODEL)
            emb = np.array(resp.data[0].embedding, dtype=np.float32)
            # normalitza per usar IndexFlatIP (cosine similarity)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            new_embs.append(emb)
        except Exception as e:
            print(f"⚠️ Error embedding: {e}; afegint vector zero.")
            new_embs.append(np.zeros((1536,), dtype=np.float32))
    embeddings = np.vstack(new_embs).astype(np.float32)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"✅ Embeddings guardats ({embeddings.shape[0]}).")

# --- FAISS INDEX per similitud cosinus (IndexFlatIP amb vectors normalitzats) ---
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)
print("✅ Index FAISS carregat i amb embeddings afegits.")

# --- FUNCIONS AUXILIARS ---
def clean_expired_context():
    now = time.time()
    expired = [uid for uid, ctx in user_context.items() if now - ctx.get("last_time", 0) > CONTEXT_EXPIRY]
    for uid in expired:
        del user_context[uid]

def needs_expansion(text: str):
    triggers = ["detall", "detalls", "explica", "aprofund", "més informació", "explica'm", "amplia"]
    t = text.lower()
    return any(x in t for x in triggers)

def extract_topics(prompt: str):
    # Extracció simple amb LLM per obtenir fins a 5 paraules-clau (fall-back ràpid)
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": f"Extreu fins a 5 temes o paraules-clau separades per coma d'aquest text:\n\n{prompt}"}],
            temperature=0,
            max_tokens=60
        )
        raw = resp.choices[0].message.content
        topics = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()][:5]
        return topics
    except Exception:
        return []

def semantic_search(query: str, top_k=TOP_K, population: str = None, topics: list = None):
    """
    Retorna una llista de fragments ordenats per (similitud cosinus + boosters).
    Cada element: (score, fragment_dict)
    """
    # expand query mínim si cal
    qtext = query
    try:
        qemb = openai.embeddings.create(input=qtext, model=EMBED_MODEL).data[0].embedding
        qemb = np.array(qemb, dtype=np.float32)
        norm = np.linalg.norm(qemb)
        if norm > 0:
            qemb = qemb / norm
    except Exception as e:
        print("⚠️ Error embedding query:", e)
        return []

    qemb = qemb.reshape(1, -1)
    D, I = index.search(qemb, top_k * 6)  # agafem més candidats i després filtrem/reordenem
    candidate_indices = [int(i) for i in I[0] if i != -1]

    results = []
    for idx in candidate_indices:
        frag = corpus[idx]
        base_score = float(D[0][candidate_indices.index(idx)]) if idx in candidate_indices else 0.0  # IP score ~ cosine
        score = base_score

        # booster si la població coincideix (prioritat alta)
        if population and population.strip():
            if population.lower() in frag.get("population", "").lower():
                score += 0.30  # tweakable

        # booster si algun topic està al títol / summary
        if topics:
            text = " ".join([frag.get("title",""), frag.get("summary",""), frag.get("long_summary","")]).lower()
            for t in topics:
                if t.lower() in text:
                    score += 0.10

        # petit booster per coincidència del títol amb paraules de la query
        title = frag.get("title", "").lower()
        for w in query.lower().split():
            if w in title and len(w) > 3:
                score += 0.02

        results.append((score, frag))

    # ordenem per score descendent i agafem únics
    results.sort(key=lambda x: x[0], reverse=True)
    # deduplicate by id preserving order
    seen = set()
    unique_results = []
    for sc, fr in results:
        if fr.get("id") in seen:
            continue
        seen.add(fr.get("id"))
        unique_results.append((sc, fr))
        if len(unique_results) >= top_k:
            break

    # Filtra fragments sense contingut útil
    unique_results = [(s, f) for (s, f) in unique_results if f.get("summary") or f.get("long_summary")]
    return unique_results[:top_k]

def format_fragments_for_prompt(frags):
    parts = []
    for score, f in frags:
        # etiqueta curta per la cita
        part = {
            "id": f.get("id"),
            "title": f.get("title"),
            "summary": f.get("summary")[:800],
            "long_summary": (f.get("long_summary") or "")[:1200],
            "population": f.get("population",""),
            "score": round(float(score), 4)
        }
        parts.append(part)
    return parts

def build_user_prompt(question: str, fragments_for_prompt: list, expand=False):
    """
    Construeix el prompt que s'enviarà al LLM: fragments + instruccions clares.
    """
    fragments_text = ""
    for i, f in enumerate(fragments_for_prompt, start=1):
        fragments_text += f"F{i}. Títol: {f['title']} (id: {f['id']})\nPoblació: {f.get('population','')}\nResum: {f['summary']}\n\n"

    instruction = (
        "INSTRUCCIONS:\n"
        "- RESPON UTILITZANT NOMÉS la informació dels fragments F1..Fn indicats a continuació.\n"
        "- Si la pregunta no es pot respondre amb aquests fragments, RESPON exactament: \"No tinc informació al corpus sobre això.\"\n"
        "- Dona una resposta breu (2-8 línies) i al final llista les fonts utilitzades amb format: F{i} (Títol).\n"
        "- No facis especulacions ni afegeixis informació no proporcionada en els fragments.\n\n"
    )

    if expand:
        instruction += "- L'usuari ha demanat detalls; si hi ha 'long_summary' utilitza fragments addicionals per afegir 1-2 frases més de context.\n\n"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n{instruction}\nFRAGMENTS:\n{fragments_text}\n\nPREGUNTA DE L'USUARI: {question}\n\nRESPONHO AIXÍ:"
    )
    return prompt

def log_tokens(user_id, tokens_used, cost):
    try:
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
    except Exception:
        pass

# --- FUNCIO PRINCIPAL ---
def ask_openai(question: str, user_id=None, population=None, strict_corpus=True):
    clean_expired_context()

    # si tenim context, intentem deduir població si no indicada
    if user_id and not population and user_id in user_context:
        last_q = user_context[user_id].get("last_query", "")
        for poble in ["Ginestar", "Benissanet", "Tivissa", "Rasquera", "Miravet"]:
            if poble.lower() in last_q.lower():
                population = poble
                break

    # extreure topics aproximats (serveix per millorar ranking)
    topics = extract_topics(question)

    # recerca semàntica
    frags = semantic_search(question, top_k=TOP_K, population=population, topics=topics)
    if not frags:
        if strict_corpus:
            return "⚠️ No tinc informació concreta al corpus sobre això."
        # fallback: deixar al model respondre lliurement (sempre amb sistema prompt)
        try:
            resp = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":question}
                ],
                temperature=0.2,
                max_tokens=MAX_RESPONSE_TOKENS
            )
            usage = getattr(resp, "usage", None)
            if usage and user_id:
                cost = (usage.total_tokens / 1000) * 0.001
                log_tokens(user_id, usage.total_tokens, cost)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error amb OpenAI: {e}"

    fragments_for_prompt = format_fragments_for_prompt(frags)
    expand = needs_expansion(question)
    user_prompt = build_user_prompt(question, fragments_for_prompt, expand=expand)

    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS
        )
        usage = getattr(resp, "usage", None)
        if usage and user_id:
            cost = (usage.total_tokens / 1000) * 0.001
            log_tokens(user_id, usage.total_tokens, cost)

        text = resp.choices[0].message.content.strip()

        # Afegim al final una secció "Fonts" basant-nos en els fragments utilitzats si el model no ha llistat fonts
        # Comprovem si hi ha alguna referència F{i} a la resposta; si no, l'afegim automàticament
        if "F1" not in text and "F2" not in text and "F3" not in text:
            # afegim fonts escollides (els 1..k disponibles)
            srcs = []
            for i, f in enumerate(fragments_for_prompt, start=1):
                srcs.append(f"F{i} ({f['title']})")
            text = f"{text}\n\nFonts: " + "; ".join(srcs[:min(3, len(srcs))])

        # actualitzem memòria d'usuari
        if user_id:
            user_context[user_id] = {
                "last_time": time.time(),
                "topics_covered": set(topics),
                "last_query": question
            }

        return text
    except Exception as e:
        return f"Error amb OpenAI: {e}"

# --- TELEGRAM HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Hola — sóc el bot de la Ribera d'Ebre. Pregunta'm sobre llocs i història local. /forget per esborrar memòria."
    )

async def forget_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_context:
        del user_context[user_id]
    await update.message.reply_text("🗑️ Memòria d'usuari esborrada.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.message.from_user.id
    # si l'usuari escriu "esborra" o similar podem interpretar com forget
    if user_text.strip().lower() in ["/forget", "esborra memòria", "esborra"]:
        if user_id in user_context:
            del user_context[user_id]
        await update.message.reply_text("🗑️ Memòria eliminada.")
        return

    # Execució en thread per no bloquejar
    resp = await asyncio.to_thread(ask_openai, user_text, user_id)
    await update.message.reply_text(resp)

# --- HEALTHCHECK HTTP SERVER ---
def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    class HealthHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_error(404, "Not Found")
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"🌐 Healthcheck HTTP server listening on port {port}")
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# --- CONFIGURACIÓ BOT ---
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("forget", forget_cmd))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

if __name__ == "__main__":
    print("🤖 Bot millorat en execució — recuperació semàntica + prompts estrictes.")
    app.run_polling()
