#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
misCEREbot.py – v4 (Ribas Edition)
- FAISS + embeddings (OpenAI si disponible; fallback local)
- Tema net i correccions toponímiques
- Filtre suau per poble
- Memòria contextual per usuari (last_tema, last_poble, last_article, memòria curta)
- Logs humans [DEBUG][User:...]
- Modes: Llista / Resum / Resum breu (local), amb recuperació contextual
- Telegram bot (python-telegram-bot v20+)

Requisits .env:
  TELEGRAM_TOKEN=xxx
  OPENAI_API_KEY=opcional
  EMB_PATH=data/embeddings_G.npy
  FAISS_INDEX_PATH=data/faiss_index_G.index
  METADATA_PATH=data/corpus_original.jsonl
"""

import os
import re
import json
import time
import faiss
import logging
import unicodedata
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# ---------- SETUP ----------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("MisCEREbot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMB_PATH = os.getenv("EMB_PATH", "data/embeddings_G.npy")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index_G.index")
METADATA_PATH = os.getenv("METADATA_PATH", "data/corpus_original.jsonl")

try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None  # treballarem en mode local si no hi és

# ---------- CONSTANTS ----------
PAGE_LIMIT = 2500
LIST_LIMIT = 3500
SESSION_TIMEOUT = 3600  # 1 hora
TELEGRAM_CHUNK = 3800  # marge de seguretat

TITLE_WEIGHT = 3.0
SUMMARY_WEIGHT = 1.0
POPULATION_BONUS = 4.0

GENERAL_MARKERS = {
    "ribera", "ribera d'ebre", "comarca", "tota la comarca",
    "tot l'arxiu", "tota la ribera", "la ribera"
}

POBLES_ADMESOS = [
    "Ascó","Benissanet","Flix","Garcia","Ginestar","Miravet",
    "Móra d'Ebre","Móra la Nova","Palma d'Ebre, la","Rasquera",
    "Riba-roja d'Ebre","Tivissa","Torre de l'Espanyol, la","Vinebre"
]

LIST_TRIGGERS = {"llista","mostra","cita","fes una llista","enumera","llista'm","llistam"}
RESUME_TRIGGERS = {"resumeix","resum","explica","parla","sintetitza","resumix"}
RESUM_BREU_TRIGGERS = {"resum curt","resum breu","resum final","conclusió curta","en poques paraules"}


STOPWORDS_TOPIC = {
    "de","del","la","el","les","los","dels","en","a","al","als",
    "un","una","uns","unes","per","sobre","amb","sense",
    "historia","histories","històries","tema","temes","coses","relats","contes",
    "parla","parlam","busca","explica","mostra","resumeix","descriu","investiga","analitza",
    "del","de la","de les","de l"
}

FUZZY_TOPONIMS = {
    "emiravet": "miravet",
    "mirabet": "miravet",
    "ginestarr": "ginestar",
    "benisanet": "benissanet",
    "asço": "ascó",
}
def normalize_text_local(t: str) -> str:
    """Normalitza text: sense accents, tot minúscules, espais simples."""
    if not t:
        return ""
    t = unicodedata.normalize("NFKD", t).encode("ASCII", "ignore").decode()
    return re.sub(r"\s+", " ", t.lower()).strip()

def extract_core_topic(text: str) -> str:
    """Extreu el tema net, sense verbs, pobles ni expressions com 'a la comarca'."""
    t = normalize_text_local(text)

    # Elimina verbs d'acció comuns i preposicions inicials
    t = re.sub(r"^(parla(r|m)?|busca|explica|mostra|resumeix|investiga|descriu|conta)\s+(de|del|la|el|les|los)?\s*", "", t)

    # Elimina preposicions freqüents
    t = re.sub(r"\b(a|al|als|a la|a les|sobre|en|de|del|dels|de la|de les)\b", " ", t)

    # Lleva paraules de context generals
    t = re.sub(r"\b(comarca|ribera|tota|tot|arxiu|general|historia|històries)\b", " ", t)

    # Neteja pobles coneguts del tema
    for p in POBLES_ADMESOS:
        t = re.sub(fr"\b{normalize_text_local(p)}\b", " ", t)

    # Correccions toponímiques
    for k, v in FUZZY_TOPONIMS.items():
        t = t.replace(k, v)

    words = [w for w in re.findall(r"[a-zà-ÿ0-9']+", t) if w not in STOPWORDS_TOPIC and len(w) > 2]
    topic = " ".join(words).strip()

    # Retalla si és massa llarg o conté fragments sobrants
    if len(topic.split()) > 6:
        topic = " ".join(topic.split()[:6])

    return topic or "(tema no identificat)"


def is_general_corpus(text: str) -> bool:
    """Detecta si es parla de la comarca o d'un àmbit general."""
    t = normalize_text_local(text)
    return any(marker in t for marker in GENERAL_MARKERS)


def detectar_poble_en_text(text: str) -> Optional[str]:
    """Detecta pobles dins del text amb tolerància màxima (preposicions, puntuació, variacions)."""
    if not text:
        return None

    norm_text = normalize_text_local(text)

    # Si parla de la comarca, no filtrem per poble
    if is_general_corpus(norm_text):
        return None

    for p in POBLES_ADMESOS:
        p_norm = normalize_text_local(p)
        # Patró robust: admet 'a', 'de', 'del', 'del poble de', 'sobre', 'per', etc.
        pattern = rf"(?:^|\s)(?:a|de|del|dels|al|en|poble de|del poble de|sobre|per)?\s*{p_norm}(?:[.,;:\s]|$)"
        if re.search(pattern, norm_text, re.IGNORECASE):
            return p
    return None



def split_for_telegram(text: str, limit: int = TELEGRAM_CHUNK) -> List[str]:
    parts = []
    s = text
    while len(s) > limit:
        cut = s.rfind("\n", 0, limit)
        if cut == -1: cut = limit
        parts.append(s[:cut])
        s = s[cut:].strip()
    if s:
        parts.append(s)
    return parts

# ---------- DADES ----------
class Doc:
    def __init__(self, data, doc_id=None):
        self.id = doc_id or data.get("id")
        self.title = data.get("title", "")
        self.summary = data.get("summary", "")
        self.summary_long = data.get("summary_long", "")
        self.population = data.get("population", "")
        self.author = data.get("author", "")
        self.topics = data.get("topics", [])
        self.years = data.get("years", "")
        self.embedding = None

documents: List[Doc] = []
id_to_doc: Dict[int, Doc] = {}

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip(): continue
        d = json.loads(line)
        doc_id = d.get("id") or i
        doc = Doc(d, doc_id)
        documents.append(doc)
        id_to_doc[doc.id] = doc

logger.info(f"[INFO] Carregats {len(documents)} documents amb IDs assegurats.")

embeddings = np.load(EMB_PATH).astype(np.float32)
if embeddings.shape[0] != len(documents):
    raise ValueError(f"Nombre d'embeddings ({embeddings.shape[0]}) no coincideix amb documents ({len(documents)})")

for i, d in enumerate(documents):
    d.embedding = embeddings[i]

try:
    vector_index = faiss.read_index(FAISS_INDEX_PATH)
    logger.info(f"[INFO] FAISS index carregat: {vector_index.ntotal} vectors.")
except Exception as e:
    logger.warning(f"[WARN] No s’ha pogut carregar FAISS: {e}. Es crea un IndexFlatIP i s’afegeixen embeddings.")
    vector_index = faiss.IndexFlatIP(embeddings.shape[1])
    # Normalitza embeddings per IP = cosinus
    emb_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    vector_index.add(emb_norm)

# ---------- SESSIONS ----------
@dataclass
class SessionObj:
    user_id: int
    mode: str = "cerca"
    topic: Optional[str] = None
    poble: Optional[str] = None
    last_tema: Optional[str] = None
    last_poble: Optional[str] = None
    last_article: Optional[int] = None
    memoria: List[Dict[str, Any]] = field(default_factory=list)
    articles_mostrats: List[int] = field(default_factory=list)
    pagina_actual: int = 1
    selected_id: Optional[int] = None
    last_update: float = field(default_factory=time.time)

class SessionsManager:
    def __init__(self, timeout: int = SESSION_TIMEOUT):
        self.sessions: Dict[int, SessionObj] = {}
        self.timeout = timeout

    def get_session(self, user_id: int) -> SessionObj:
        s = self.sessions.get(user_id)
        if s is None or (time.time() - s.last_update) > self.timeout:
            s = SessionObj(user_id=user_id)
            self.sessions[user_id] = s
        return s

    def update_session(self, user_id: int, **kwargs):
        s = self.get_session(user_id)
        for k, v in kwargs.items():
            if hasattr(s, k):
                setattr(s, k, v)
        s.last_update = time.time()
        self.sessions[user_id] = s

    def reset_session(self, user_id: int):
        if user_id in self.sessions:
            del self.sessions[user_id]

    def add_to_memory(self, user_id: int, msg: str, tema: str, poble: Optional[str], results: List[Tuple[Doc, float]]):
        s = self.get_session(user_id)
        entry = {
            "msg": msg,
            "tema": tema,
            "poble": poble,
            "articles": [d.id for d, _ in results],
            "time": time.time(),
        }
        s.memoria.append(entry)
        s.memoria = s.memoria[-6:]
        s.last_tema = tema or s.last_tema
        s.last_poble = poble if poble is not None else s.last_poble
        if results:
            s.last_article = results[0][0].id
        self.update_session(user_id,
                            memoria=s.memoria,
                            last_tema=s.last_tema,
                            last_poble=s.last_poble,
                            last_article=s.last_article)
        logger.info(f"[DEBUG][User:{user_id}] Msg: {msg}")
        logger.info(f" → Tema net: {tema} | Poble: {poble}")
        if results:
            logger.info(f" → Top 1: /{results[0][0].id} {results[0][0].title} (pes: {results[0][1]:.2f})")

sessions = SessionsManager()

# ---------- EMBEDDINGS I CERCA ----------
def embed_query(query: str) -> np.ndarray:
    """Embedding de consulta: OpenAI si hi ha API; altrament un vector determinista local."""
    base = normalize_text_local(query) or "consulta"
    if openai and OPENAI_API_KEY:
        try:
            resp = openai.embeddings.create(model="text-embedding-3-small", input=base)
            v = np.array(resp.data[0].embedding, dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            return v
        except Exception as e:
            logger.warning(f"[WARN] Embedding OpenAI ha fallat; s'usa fallback local: {e}")
    # Fallback local determinista
    rng = np.random.RandomState(abs(hash(base)) % (2**32))
    v = rng.normal(size=(embeddings.shape[1],)).astype("float32")
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def matches_poble(doc: Doc, poble_filter: Optional[str]) -> bool:
    if not poble_filter:
        return True
    return normalize_text_local(doc.population or "") == normalize_text_local(poble_filter)

def query_faiss(query_text: str,
                poble: Optional[str] = None,
                exclude_ids: Optional[List[int]] = None,
                top_k: int = 5) -> List[Tuple[Doc, float]]:
    """Cerca amb filtre suau per poble (bonus, no exclusió)."""
    q_emb = embed_query(query_text or (poble or ""))
    # utilitzem els embeddings normalitzats per IP ~ cosinus
    emb_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)

    exclude = set(exclude_ids or [])
    results: List[Tuple[Doc, float]] = []

    for i, d in enumerate(documents):
        if d.id in exclude:
            continue
        sim = float(np.dot(q_emb, emb_norm[i]))  # cosinus
        score = sim

        q_clean = normalize_text_local(query_text)
        if q_clean and q_clean in normalize_text_local(d.title or ""):
            score += TITLE_WEIGHT
        if q_clean and q_clean in normalize_text_local(d.summary or ""):
            score += SUMMARY_WEIGHT

        if poble and matches_poble(d, poble):
            score += POPULATION_BONUS * 0.8
        elif poble and normalize_text_local(poble) in normalize_text_local(d.summary_long or ""):
            score += POPULATION_BONUS * 0.4

        results.append((d, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# ---------- COMPOSICIÓ DE RESPOSTES ----------
def compose_list_response(tema: str,
                          results: List[Tuple[Doc, float]],
                          poble: Optional[str]) -> str:
    header = f"[Mode: Llista]\n[Tema: {tema or '-'} | Poble: {poble or 'Tots'}]\n\n"
    body = ""
    for i, (d, s) in enumerate(results, 1):
        short = (d.summary or d.summary_long or "").replace("\n", " ")[:280].rstrip()
        body += f"{i}. {short}\n/{d.id} {d.title} (Pes: {s:.2f})\n\n"
    return header + body.strip()

def resum_breu_local(text: str, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r'(?<=[.!?]) +', text)
    key_terms = {"castell","templer","història","arqueologia","poble","batalla","refugi","guerra","miravet","ginestar","ascó","móra"}

    selected = []
    for s in sentences:
        if any(k in s.lower() for k in key_terms) or len(selected) < 3:
            selected.append(s)
        if len(" ".join(selected)) > max_chars:
            break

    resum = " ".join(selected)
    if len(resum) > max_chars:
        resum = resum[:max_chars].rsplit('.', 1)[0] + '.'
    return resum

def compose_summary_response(tema: str,
                             results: List[Tuple[Doc, float]],
                             poble: Optional[str],
                             brief: bool = False) -> str:
    fragments = [d.summary_long or d.summary or "" for d, _ in results]
    full_text = " ".join(fragments) if fragments else "(Cap fragment disponible)."
    if brief:
        resum = resum_breu_local(full_text)
        mode = "Resum breu"
    else:
        # resum “normal” (local simple): fem-lo més llarg que el breu
        text = re.sub(r"\s+", " ", full_text.strip())
        resum = text[:1800].rsplit('.', 1)[0] + '.'
        mode = "Resum"

    header = f"[Mode: {mode}]\n[Tema: {tema or '-'} | Poble: {poble or 'Tots'}]\n\n"
    footer = "\n\nArticles:\n" + "\n".join([f"/{d.id} {d.title} (Pes: {s:.2f})" for d, s in results])
    return header + resum + footer
    
    
    # --- Funcions auxiliars per mode creua ---

# ---------- Funcions auxiliars per mode creua ----------

def intersect_topics(docA: Doc, docB: Doc) -> List[str]:
    """Retorna una llista de temes comuns entre els dos documents (basat en paraules clau)."""
    def keywords(doc: Doc) -> set:
        text = (doc.summary_long or doc.summary or "").lower()
        words = re.findall(r"[a-zà-ÿ0-9]+", text)
        return {w for w in words if len(w) > 4}
    keysA = keywords(docA)
    keysB = keywords(docB)
    common = keysA.intersection(keysB)
    return list(common)[:8]

def extract_fragment_for_topic(doc: Doc, tema: str, radius: int = 120) -> str:
    text = doc.summary_long or doc.summary or ""
    lower = text.lower()
    idx = lower.find(tema.lower())
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(text), idx + radius)
    return text[start:end].strip()

def merge_fragments(fragA: str, fragB: str, max_len: int = 300) -> str:
    if not fragA:
        return fragB
    if not fragB:
        return fragA
    if fragA in fragB:
        return fragB
    if fragB in fragA:
        return fragA
    merged = fragA.strip() + "\n…\n" + fragB.strip()
    if len(merged) > max_len:
        return merged[:max_len].rsplit(" ", 1)[0] + "…"
    return merged


# --- Afegir dins les comandes i el main ---

# Ja tens cmd_creua i handle_creua_topic com et vaig passar abans.

# Afegeix al main():
#    app.add_handler(CommandHandler("creua", cmd_creua))
#    app.add_handler(MessageHandler(filters.Regex(r"^/[a-z]$"), handle_creua_topic))

# Assegura que vinguin **abans** de:
#    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))


# ---------- MODE / ROUTER ----------
def detect_mode(text: str) -> str:
    t = normalize_text_local(text)
    if any(x in t for x in LIST_TRIGGERS):
        return "llista"
    if any(x in t for x in RESUM_BREU_TRIGGERS):
        return "resum_breu"
    if any(x in t for x in RESUME_TRIGGERS):
        return "resum"
    return "cerca"




# ---------- GESTIÓ PRINCIPAL DEL TEXT ----------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    text = (update.message.text or "").strip()
    sess = sessions.get_session(uid)

    # --- Mode ARTICLE ---
    if sess.mode == "article" and sess.selected_id:
        doc = id_to_doc.get(sess.selected_id)
        if not doc:
            await update.message.reply_text("Error: no es troba l'article seleccionat.")
            return
        text_norm = normalize_text_local(text)
        content_norm = normalize_text_local(doc.summary_long or "")
        if text_norm in {"tot", "article complet"}:
            await cmd_tot(update, context)
            return
        # Cerca simple dins del text
        idx = content_norm.find(text_norm)
        if idx == -1:
            await update.message.reply_text("No s'ha trobat aquesta paraula o tema dins l'article.")
            return
        start = max(0, idx - 120)
        end = min(len(doc.summary_long or ""), idx + 400)
        fragment = (doc.summary_long or "")[start:end]
        await update.message.reply_text(
            f"[Mode: Article]\n"
            f"{doc.title}\n"
            f"{doc.author or ''} - {doc.population or ''}\n\n"
            f"...{fragment.strip()}...\n\n"
            "Pots escriure /tot per veure l'article complet o /cerca per tornar al mode general."
        )
        return

    # --- Mode normal (cerca general) ---
    tema_net = extract_core_topic(text)
    poble_detectat = detectar_poble_en_text(text)
    context_recovered = False

    # 🔹 Detecció de continuïtat semàntica
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    words = [w for w in cleaned.split() if w not in {'i','els','les','lo','la','l','del','de','d'}]
    logger.info(f"[CTX][User:{uid}] Paraules útils: {len(words)} -> {words}")
    short_query = len(words) <= 4
    no_verbs = not any(v in text.lower() for v in ["busca","fes","parla","mostra","resumeix","cerca","llista"])

    if short_query and no_verbs and not poble_detectat and sess.last_tema:
        tema_original = sess.last_tema
        tema_net = f"{sess.last_tema}, {tema_net}" if tema_net else sess.last_tema
        context_recovered = True
        logger.info(f"[CTX][Continuïtat][User:{uid}] Afegint '{text}' al tema anterior '{tema_original}'")

    elif poble_detectat and (not tema_net or tema_net == "(tema no identificat)") and sess.last_tema:
        tema_net = sess.last_tema
        sess.poble = poble_detectat
        context_recovered = True
        logger.info(f"[CTX][Poble][User:{uid}] Reutilitzant tema anterior '{tema_net}' per nou poble: {poble_detectat}")

    elif tema_net and not poble_detectat and sess.last_poble:
        sess.poble = sess.last_poble
        context_recovered = True
        logger.info(f"[CTX][Tema][User:{uid}] Reutilitzant poble anterior '{sess.poble}' per nou tema: '{tema_net}'")

    elif (not tema_net or tema_net == "(tema no identificat)") and not poble_detectat:
        tema_net = sess.last_tema or ""
        sess.poble = sess.last_poble or None
        context_recovered = True
        logger.info(f"[CTX][Recuperat][User:{uid}] Reutilitzant context complet: Tema '{tema_net}' | Poble '{sess.poble or 'Tots'}'")

    if poble_detectat is None and is_general_corpus(text):
        sess.poble = None
        logger.info(f"[CTX][General][User:{uid}] Cerca general: Ribera d'Ebre (sense filtre de poble)")

    sess.last_tema = tema_net
    sess.last_poble = sess.poble

    logger.info(f"[DEBUG][User:{uid}] Tema definitiu: '{tema_net}' | Poble: {sess.poble or 'Tots'}")
    if context_recovered:
        logger.info(f"[CTX][User:{uid}] Context actiu → Tema: '{tema_net}' | Poble: '{sess.poble or 'Tots'}'")

    if not tema_net:
        await update.message.reply_text("Dona'm un tema o un poble per començar la cerca.")
        return

    # --- Cerca FAISS ---
    results = query_faiss(query_text=tema_net, poble=sess.poble, exclude_ids=sess.articles_mostrats, top_k=5)
    if not results:
        await update.message.reply_text("No s'han trobat resultats rellevants per a la consulta.")
        return

    sessions.add_to_memory(uid, text, tema_net, sess.poble, results)

    # --- Mode de resposta ---
    mode = detect_mode(text)
    if mode == "llista":
        resp = compose_list_response(tema_net, results, sess.poble)
    elif mode == "resum_breu":
        resp = compose_summary_response(tema_net, results, sess.poble, brief=True)
    else:
        resp = compose_summary_response(tema_net, results, sess.poble, brief=False)

    if context_recovered:
        header = f"[Context recuperat] Tema: {tema_net} | Poble: {sess.poble or 'Tots'}\n\n"
        resp = header + resp

    for part in split_for_telegram(resp, limit=TELEGRAM_CHUNK):
        await update.message.reply_text(part, disable_web_page_preview=True)

# ---------- TELEGRAM HANDLERS ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sessions.reset_session(update.effective_user.id)
    await update.message.reply_text("Benvingut a MisCEREbot! 🤖\n Pregunta'm sobre la Ribera !\n /ajuda per més detalls.")

async def ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open("data/ajuda.txt", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Error llegint ajuda.txt: {e}")
        text = ("Error intern: no s'ha pogut carregar l'ajuda. "
                "Pots provar més tard.")
    # Si és massa llarg, fragmenta
    for part in split_for_telegram(text, limit=TELEGRAM_CHUNK):
        await update.message.reply_text(part, disable_web_page_preview=True)

async def expert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open("data/expert.txt", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Error llegint expert.txt: {e}")
        text = ("Error intern: no s'ha pogut carregar l'ajuda avançada. "
                "Pots provar més tard.")
    for part in split_for_telegram(text, limit=TELEGRAM_CHUNK):
        await update.message.reply_text(part, disable_web_page_preview=True)


async def cmd_nou(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sessions.reset_session(update.effective_user.id)
    await update.message.reply_text("Context reiniciat.")

async def cmd_poble(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) != 2:
        await update.message.reply_text("Ús: /poble NomDelPoble | /poble tots")
        return
    arg = parts[1].strip()
    norm = normalize_text_local(arg)
    if norm in GENERAL_MARKERS or norm in {"tots","tot","ribera","ribera d ebre"}:
        sessions.update_session(uid, poble=None, articles_mostrats=[])
        await update.message.reply_text("Filtre de poble eliminat (Tota la Ribera d'Ebre).")
        return
    match = None
    for p in POBLES_ADMESOS:
        if normalize_text_local(p) == norm:
            match = p
            break
    if not match:
        await update.message.reply_text("Poble no reconegut. Poblacions admeses: " + ", ".join(POBLES_ADMESOS))
        return
    sessions.update_session(uid, poble=match, articles_mostrats=[])
    await update.message.reply_text(f"Poble canviat a: {match}")

import re

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra un article concret i entra al mode 'Cerca en article'."""
    uid = update.effective_user.id
    text = (update.message.text or "").strip()
    art_id = None

    logger.info(f"[CMD_ID][User:{uid}] Entrada rebuda: {text}")

    # Obté el número tant si és /id26, /id 26 o només "26"
    match = re.match(r"^/id(\d+)$", text)
    if match:
        art_id = int(match.group(1))
    elif context.args and context.args[0].isdigit():
        art_id = int(context.args[0])
    elif text.isdigit():
        art_id = int(text)

    logger.debug(f"[CMD_ID][User:{uid}] Detectat art_id={art_id}")

    if not art_id:
        await update.message.reply_text("Ús: /idN, per exemple /id26")
        return

    doc = id_to_doc.get(art_id)
    if not doc:
        logger.warning(f"[CMD_ID][User:{uid}] Article ID {art_id} no trobat")
        await update.message.reply_text(f"No s'ha trobat l'article amb ID {art_id}.")
        return

    logger.info(f"[CMD_ID][User:{uid}] Obrint article {art_id}: {doc.title}")

    # Actualitza sessió
    sess = sessions.get_session(uid)
    sess.mode = "article"
    sess.selected_id = art_id
    sess.last_article = art_id

    resum = (doc.summary or doc.summary_long or "(Sense resum)")[:2000].rsplit('.', 1)[0] + '.'

    msg = (
        f"[Cerca en article]\n"
        f"{doc.id}) {doc.title}\n"
        f"{doc.author or ''} - {doc.population or ''}\n\n"
        f"{resum}\n\n"
        "Escriu /tot per veure l'article sencer\n"
        "Surt a cerca general amb /cerca\n"
        "O demana qualsevol cosa sobre aquest article."
    )

    await update.message.reply_text(msg, disable_web_page_preview=True)




async def cmd_mes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ara mateix el contingut es mostra sense buffer de pàgines llargues. Torna a fer una cerca o /idN.")

# ---------- MODE ARTICLE ----------

async def cmd_n(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Activa el mode 'Cerca en article'"""
    uid = update.effective_user.id
    text = (update.message.text or "").strip()
    sess = sessions.get_session(uid)

    # Si porta ID directe, ex: /n23
    m = re.match(r"^/n(\d+)$", text)
    if m:
        article_id = int(m.group(1))
    else:
        article_id = sess.last_article

    if article_id is None or article_id not in id_to_doc:
        await update.message.reply_text("No hi ha cap article seleccionat. Escriu /n23 per obrir-ne un directament.")
        return

    doc = id_to_doc[article_id]
    sess.mode = "article"
    sess.selected_id = article_id
    sess.pagina_actual = 1

    resum = (doc.summary_long or doc.summary or "(sense resum)").strip()
    resum_inicial = resum[:600] + ("..." if len(resum) > 600 else "")

    text_resp = (
        f"[Cerca en article]\n"
        f"n) {doc.title}\n"
        f"{doc.author or '(Autor desconegut)'} - {doc.population or '(Poble no especificat)'}\n\n"
        f"{resum_inicial}\n\n"
        "escriu /tot per veure l'article sencer\n\n"
        "Surt a cerca general amb /cerca\n"
        "o demana qualsevol cosa sobre aquest article."
    )

    await update.message.reply_text(text_resp, disable_web_page_preview=True)


async def cmd_tot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra l'article complet, paginat"""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)

    if sess.mode != "article" or not sess.selected_id:
        await update.message.reply_text("No hi ha cap article obert. Usa /n o /n23 primer.")
        return

    doc = id_to_doc[sess.selected_id]
    full = doc.summary_long or doc.summary or "(sense contingut)"
    parts = split_for_telegram(full, limit=3000)
    sess.pagina_actual = 1
    sess.total_pagines = len(parts)
    sess.article_pages = parts

    header = f"[Cerca en article]\n n) {doc.title}\n{doc.author or ''} - {doc.population or ''}\n\n"
    footer = (
        f"pag {sess.pagina_actual}/{sess.total_pagines}. Continua amb /mes\n\n"
        "Surt a cerca general amb /cerca\n"
        "o demana qualsevol cosa sobre aquest article."
    )

    await update.message.reply_text(header + parts[0] + "\n\n" + footer)


async def cmd_mes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra la següent pàgina d'un article"""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)

    if sess.mode != "article" or not hasattr(sess, "article_pages"):
        await update.message.reply_text("No hi ha cap article obert. Usa /tot per començar.")
        return

    if sess.pagina_actual >= len(sess.article_pages):
        doc = id_to_doc[sess.selected_id]
        await update.message.reply_text(
            f"{doc.title}\n{doc.author or ''} - {doc.population or ''}\n\n"
            f"Autor de l'article: {doc.author or '(desconegut)'}\n"
            f"pag {sess.pagina_actual}/{len(sess.article_pages)}. Fi de l'article\n\n"
            "Surt a cerca general amb /cerca\n"
            "o demana qualsevol cosa sobre aquest article."
        )
        return

    sess.pagina_actual += 1
    doc = id_to_doc[sess.selected_id]
    text = sess.article_pages[sess.pagina_actual - 1]

    footer = (
        f"pag {sess.pagina_actual}/{len(sess.article_pages)}"
        + (". Continua amb /mes\n\n" if sess.pagina_actual < len(sess.article_pages) else ". Fi de l'article\n\n")
        + "Surt a cerca general amb /cerca\n"
        "o demana qualsevol cosa sobre aquest article."
    )

    await update.message.reply_text(
        f"{doc.title}\n{doc.author or ''} - {doc.population or ''}\n\n{text}\n\n{footer}"
    )


async def cmd_cerca(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Surt del mode article i torna a la cerca general"""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    sess.mode = "cerca"
    sess.selected_id = None
    sess.pagina_actual = 1
    await update.message.reply_text("Tornes a la cerca general. Pots escriure qualsevol tema o poble.")

async def handle_numeric_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Permet obrir articles escrivint o clicant /26, /7, etc."""
    text = update.message.text.strip()
    uid = update.effective_user.id
    logger.info(f"[NUMERIC_ID][User:{uid}] Entrada rebuda: {text}")

    m = re.match(r"^/(\d+)$", text)
    if not m:
        return

    art_id = int(m.group(1))
    doc = id_to_doc.get(art_id)
    if not doc:
        await update.message.reply_text(f"No s'ha trobat l'article amb ID {art_id}.")
        return

# ---------- MODE CREUA / MIX ----------

async def cmd_creua(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    text = (update.message.text or "").strip()
    sess = sessions.get_session(uid)

    parts = text.split()
    if len(parts) < 3:
        await update.message.reply_text("Ús: /creua A B [temaOpcional]")
        return

    try:
        idA = int(parts[1])
        idB = int(parts[2])
    except ValueError:
        await update.message.reply_text("Els dos primers arguments han de ser IDs vàlids.")
        return

    if idA not in id_to_doc or idB not in id_to_doc:
        await update.message.reply_text("No es troba algun dels articles indicats.")
        return

    docA = id_to_doc[idA]
    docB = id_to_doc[idB]

    tema_donat = None
    if len(parts) >= 4:
        tema_donat = " ".join(parts[3:]).strip().lower()

    topics_comuns = intersect_topics(docA, docB)
    fragments = {}
    scores = {}
    for t in topics_comuns:
        fragA = extract_fragment_for_topic(docA, t)
        fragB = extract_fragment_for_topic(docB, t)
        fragments[t] = (fragA, fragB)
        cntA = (docA.summary_long or docA.summary or "").lower().count(t.lower())
        cntB = (docB.summary_long or docB.summary or "").lower().count(t.lower())
        scores[t] = cntA + cntB

    sess.mode = "creua"
    sess.creua_pair = (idA, idB)
    sess.creua_topics = fragments
    sess.creua_scores = scores

    if tema_donat:
        if tema_donat in fragments:
            fragA, fragB = fragments[tema_donat]
            merged = merge_fragments(fragA, fragB)
            header = (
                f"[Creuament] {idA} ⟷ {idB}\n"
                f"{docA.title} – {docA.population}\n"
                f"{docB.title} – {docB.population}\n\n"
            )
            msg = header + f"Resum creuat sobre “{tema_donat}”:\n{merged}\n\nSurt a cerca general amb /cerca"
            await update.message.reply_text(msg, disable_web_page_preview=True)
        else:
            fragA = extract_fragment_for_topic(docA, tema_donat)
            fragB = extract_fragment_for_topic(docB, tema_donat)
            if fragA or fragB:
                merged = merge_fragments(fragA, fragB)
                header = (
                    f"[Creuament] {idA} ⟷ {idB}\n"
                    f"{docA.title} – {docA.population}\n"
                    f"{docB.title} – {docB.population}\n\n"
                )
                msg = header + f"Resum creuat (fallback) sobre “{tema_donat}”:\n{merged}\n\nSurt a cerca general amb /cerca"
                await update.message.reply_text(msg, disable_web_page_preview=True)
            else:
                await update.message.reply_text("Tema no trobat entre els articles indicats.")
        return

    # Presentació de llista neta clicable
    header = (
        f"[Creuament] {idA} ⟷ {idB}\n"
        f"{docA.title} – {docA.population}\n"
        f"{docB.title} – {docB.population}\n\n"
    )
    body = ""
    labels = "abcdefghijklmnopqrstuvwxyz"
    sess.creua_labels = {}
    sorted_topics = sorted(topics_comuns, key=lambda t: scores.get(t, 0), reverse=True)
    for i, t in enumerate(sorted_topics):
        label = labels[i]
        sess.creua_labels[label] = t
        fragA, fragB = fragments[t]
        snippet = merge_fragments(fragA, fragB, max_len=200)
        sc = scores.get(t, 0)
        body += f"/{label} Tema: {t} (Pes: {sc})\n{snippet}\n\n"

    footer = "Surt a cerca general amb /cerca"
    await update.message.reply_text(header + body + footer, disable_web_page_preview=True)

async def handle_creua_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    if sess.mode != "creua":
        return

    text = (update.message.text or "").strip().lower()
    if not text.startswith("/"):
        return
    parts = text[1:].split(maxsplit=1)
    label = parts[0]
    if not hasattr(sess, "creua_labels") or label not in sess.creua_labels:
        return

    tema = sess.creua_labels[label]
    fragA, fragB = sess.creua_topics.get(tema, ("", ""))
    merged = merge_fragments(fragA, fragB)

    idA, idB = sess.creua_pair
    docA = id_to_doc[idA]
    docB = id_to_doc[idB]

    header = (
        f"[Creuament] {idA} ⟷ {idB}\n"
        f"{docA.title} – {docA.population}\n"
        f"{docB.title} – {docB.population}\n\n"
        f"Detall tema “{tema}”\n\n"
    )
    msg = header + merged + "\n\nSurt a cerca general amb /cerca"
    await update.message.reply_text(msg, disable_web_page_preview=True)

# --- En el main(), registra:
# app.add_handler(CommandHandler("creua", cmd_creua))
# app.add_handler(MessageHandler(filters.Regex(r"^/[a-z]$"), handle_creua_topic))

    # Guarda la sessió com a mode article
    sess = sessions.get_session(uid)
    sess.mode = "article"
    sess.selected_id = art_id
    sess.last_article = art_id

    resum = (doc.summary or doc.summary_long or "(Sense resum)")[:2000].rsplit('.', 1)[0] + '.'

    msg = (
        f"[Cerca en article]\n"
        f"{doc.id}) {doc.title}\n"
        f"{doc.author or ''} - {doc.population or ''}\n\n"
        f"{resum}\n\n"
        "Escriu /tot per veure l'article sencer\n"
        "Surt a cerca general amb /cerca\n"
        "O demana qualsevol cosa sobre aquest article."
    )

    logger.info(f"[NUMERIC_ID][User:{uid}] Article {art_id} obert correctament.")
    await update.message.reply_text(msg, disable_web_page_preview=True)


# ---------- BOOT ----------
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Falta TELEGRAM_TOKEN al .env")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ajuda", ajuda))
    app.add_handler(CommandHandler("expert", expert))
    app.add_handler(CommandHandler("help", ajuda))
    app.add_handler(CommandHandler("nou", cmd_nou))
    app.add_handler(CommandHandler("poble", cmd_poble))
    app.add_handler(CommandHandler("n", cmd_n))
    app.add_handler(CommandHandler("tot", cmd_tot))
    app.add_handler(CommandHandler("mes", cmd_mes))
    app.add_handler(CommandHandler("cerca", cmd_cerca))
    app.add_handler(CommandHandler("creua", cmd_creua))
    app.add_handler(CommandHandler("poble", cmd_poble))
    app.add_handler(CommandHandler("id", cmd_id))
    
    app.add_handler(MessageHandler(filters.Regex(r"^/[a-z]$"), handle_creua_topic))
    app.add_handler(MessageHandler(filters.Regex(r"^/\d+$"), handle_numeric_id))
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    logger.info("Bot en marxa, esperant missatges...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
