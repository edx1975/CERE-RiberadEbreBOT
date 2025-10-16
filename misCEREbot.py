#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
misCEREbot.py – v5 (Last_IA v2 - Railway Compatible)
- Mateixa lògica original, però amb estructura endreçada i repeticions eliminades.
- NO s'han fet canvis funcionals profunds; només ordenació, neteja i correcció de referències.
"""


#------------------------------å---------------------------------
#1--------CAPÇALERA: IMPORTS, CONFIG, LOGGING, CONSTANTS --------
#---------------------------------------------------------------
from __future__ import annotations
import os
import re
import json
import time
import faiss
import asyncio

import logging
import unicodedata
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict

from dotenv import load_dotenv
load_dotenv()

# Telegram imports
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Config logs: molt verbós per debug fi
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("MisCEREbot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("FaissUtils").setLevel(logging.INFO)

#---------------------------------------------------------------
#--------CAPÇALERA: CONSTANTS I PATHS --------------------------
#---------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

METADATA_PATH = Path(os.getenv("METADATA_PATH", DATA_DIR / "corpus_original.jsonl"))
EMB_PATH = Path(os.getenv("EMB_PATH", DATA_DIR / "embeddings_G.npy"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", DATA_DIR / "faiss_index_G.index"))
TOPICS_PATH = Path(os.getenv("TOPICS_PATH", DATA_DIR / "topics_semantics.json"))
EMB_CACHE_PATH = Path(os.getenv("EMB_CACHE_PATH", DATA_DIR / "cache_embeddings.json"))

PAGE_LIMIT = 2500
LIST_LIMIT = 3500
TELEGRAM_CHUNK = 3800
SESSION_TIMEOUT = 3600  # 1h

GENERAL_MARKERS = {
    "ribera", "ribera d'ebre", "comarca", "tota la comarca",
    "tot l'arxiu", "tota la ribera", "la ribera", "tots", "totes"
}

POBLES_ADMESOS = [
    "Ascó","Benissanet","Flix","Garcia","Ginestar","Miravet",
    "Móra d'Ebre","Móra la Nova","Palma d'Ebre, la","Rasquera",
    "Riba-roja d'Ebre","Tivissa","Torre de l'Espanyol, la","Vinebre"
]

FUZZY_TOPONIMS = {
    "emiravet": "miravet",
    "mirabet": "miravet",
    "ginestarr": "ginestar",
    "benisanet": "benissanet",
    "asço": "ascó",
    "asco": "ascó",
    "mora debre": "móra d'ebre",
    "mora ebre": "móra d'ebre",
    "mora la nova": "móra la nova",
    "mora nova": "móra la nova",
    "La Torre": "Torre de l'Espanyol, la"

}

# Sinònims bàsics per a expansió tolerant de consultes curtes
SIMPLE_SYNONYMS = {
    "riu": ["ebre", "hidrografia", "riberes", "aigua", "fluvial"],
    "ebre": ["riu", "hidrografia", "riberes"],
    "castell": ["fortalesa", "fortificació", "torre", "muralla"],
    "castells": ["fortaleses", "fortificacions", "torres", "muralles"],
    "ginestar": ["ginestar", "ribera d'ebre"],
    "ribera": ["ribera d'ebre", "comarca"],
    "historia": ["història", "passat", "crònica"],
    "història": ["historia", "passat", "crònica"],
}


LIST_TRIGGERS = {"llista", "mostra", "cita", "fes una llista", "enumera", "llista'm", "llistam"}
RESUME_TRIGGERS = {"resumeix", "resum", "explica", "parla", "sintetitza", "resumix"}
RESUM_BREU_TRIGGERS = {"resum curt", "resum breu", "resum final", "conclusió curta", "en poques paraules"}
EXPAND_TRIGGERS = {"amplia", "expandeix", "continua", "segueix", "aprofundix", "afegeix", "més info"}

#---------------------------------------------------------------
#--------CAPÇALERA: SESSIONS -----------------------------------
#---------------------------------------------------------------
@dataclass
class SessionObj:
    user_id: int
    mode: str = "cerca"
    topic: Optional[str] = None
    poble: Optional[str] = None
    last_tema: Optional[str] = None
    last_poble: Optional[str] = None
    last_article: Optional[int] = None
    last_summary: Optional[str] = None
    last_resum: Optional[str] = None
    last_gen: dict = field(default_factory=dict)
    memoria: List[str] = field(default_factory=list)
    articles_mostrats: List[int] = field(default_factory=list)
    pagina_actual: int = 1
    selected_id: Optional[int] = None
    last_update: float = field(default_factory=time.time)
    humor_mode: bool = False

class SessionsManager:
    def __init__(self, timeout: int = SESSION_TIMEOUT):
        self.sessions: Dict[int, SessionObj] = {}
        self.timeout = timeout

    def get_session(self, user_id: int) -> SessionObj:
        s = self.sessions.get(user_id)
        if s is None or (time.time() - s.last_update) > self.timeout:
            s = SessionObj(user_id=user_id)
            self.sessions[user_id] = s
            logger.info(f"[SESSION] Nova sessió {user_id}")
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
            logger.info(f"[SESSION] Reset sessió {user_id}")

    def add_to_memory(self, user_id: int, bloc: str):
        s = self.get_session(user_id)
        s.memoria.append(bloc)
        s.memoria = s.memoria[-6:]
        s.last_update = time.time()

sessions = SessionsManager()

def log_session_state(sess: SessionObj, prefix: str = "[STATE]"):
    logger.info(
        f"{prefix} uid={sess.user_id} mode={sess.mode} tema={sess.last_tema} "
        f"poble={sess.poble} last_article={sess.last_article} humor={sess.humor_mode}"
    )

#---------------------------------------------------------------
#f1--------CAPÇALERA: NORMALITZACIÓ I UTILITATS DE TEXT ----------
#---------------------------------------------------------------
def normalize_text_local(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def canonicalize_population(p: str) -> str:
    # Inicialitza el mapa normalitzat de toponímia si no existeix
    global FUZZY_TOPONIMS_NORM
    try:
        FUZZY_TOPONIMS_NORM
    except NameError:
        FUZZY_TOPONIMS_NORM = {normalize_text_local(k): normalize_text_local(v) for k, v in FUZZY_TOPONIMS.items()}
    p = normalize_text_local(p or "")
    for k, v in FUZZY_TOPONIMS_NORM.items():
        p = p.replace(k, v)
    # variants amb articles i comes
    p = p.replace(",", " ").replace(" la ", " la ").strip()
    p = re.sub(r"\s+", " ", p)
    return p

def neteja_consulta(text: str) -> str:
    text = re.sub(r"[^\w\s'’\-.,;:!?]", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    logger.debug(f"[NETEJA] → '{text}'")
    return text

def is_general_corpus(text: str) -> bool:
    t = normalize_text_local(text)
    return any(marker in t for marker in GENERAL_MARKERS)

def detecta_poble(text: str, known_populations: Optional[List[str]] = None) -> Optional[str]:
    if not known_populations:
        # usa la llista carregada del corpus per defecte
        known_populations = KNOWN_POPS
    
    text_norm = canonicalize_population(text)
    
    # Si detecta termes generals com "ribera", "comarca", etc., desactiva el filtre de poble
    if any(marker in text_norm for marker in GENERAL_MARKERS):
        logger.debug("[POBLE] Detectat terme general (ribera/comarca) - desactivant filtre de poble")
        return None
    
    for pob in known_populations:
        if pob and pob in text_norm:
            logger.debug(f"[POBLE] Trobat '{pob}' dins la consulta.")
            return pob
    logger.debug("[POBLE] Cap poble detectat explícitament.")
    return None



#---------------------------------------------------------------
#2--------CAPÇALERA: MODEL DE DADA I CÀRREGA DE CORPUS ----------
#---------------------------------------------------------------
class Doc:
    def __init__(self, data: Dict[str, Any], doc_id: Optional[int] = None):
        self.id = doc_id if doc_id is not None else data.get("id")
        self.title = data.get("title", "")
        self.summary = data.get("summary", "")
        # Accept legacy 'summary_log' as 'summary_long'
        self.summary_long = data.get("summary_long", data.get("summary_log", ""))
        # Accept legacy 'location' as 'population'
        self.population = data.get("population", data.get("location", ""))
        self.author = data.get("author", "")
        self.topics = data.get("topics", [])
        self.years = data.get("years", "")
        self.embedding = None
        self.doc_id = self.id

documents: List[Doc] = []
id_to_doc: Dict[int, Doc] = {}

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"❌ No s'ha trobat METADATA_PATH: {METADATA_PATH}")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"[CORPUS] Línia {i} JSON invàlid: {e}")
            continue
        doc_id = d.get("id") if d.get("id") is not None else i
        doc = Doc(d, doc_id)
        documents.append(doc)
        id_to_doc[doc.id] = doc

logger.info(f"[CORPUS] Carregats {len(documents)} documents amb IDs assegurats.")

# Llista de pobles coneguts (normalitzats) per a detecta_poble()
KNOWN_POPS = sorted({canonicalize_population(d.population) for d in documents if d.population})

#---------------------------------------------------------------
#--------CAPÇALERA: CÀRREGA D’EMBEDDINGS I ÍNDEX FAISS --------
#---------------------------------------------------------------
if not EMB_PATH.exists():
    raise FileNotFoundError(f"❌ No s'ha trobat EMB_PATH: {EMB_PATH}")

embeddings = np.load(EMB_PATH).astype(np.float32)
if embeddings.shape[0] != len(documents):
    raise ValueError(
        f"Nombre d'embeddings ({embeddings.shape[0]}) no coincideix amb documents ({len(documents)})"
    )

for i, d in enumerate(documents):
    d.embedding = embeddings[i]

# Intenta llegir un índex FAISS existent; si falla, crea’n un de pla
try:
    if FAISS_INDEX_PATH.exists():
        vector_index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"[FAISS] Índex carregat: {vector_index.ntotal} vectors, dim={vector_index.d}")
        # Rebuild si la dimensió no coincideix amb els embeddings actuals
        if getattr(vector_index, 'd', None) != embeddings.shape[1]:
            logger.warning(
                f"[FAISS] Dimensió de l'índex ({getattr(vector_index, 'd', 'unknown')}) no coincideix amb embeddings ({embeddings.shape[1]}). Reconstruint índex."
            )
            vector_index = faiss.IndexFlatIP(embeddings.shape[1])
            emb_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
            vector_index.add(emb_norm)
            try:
                faiss.write_index(vector_index, str(FAISS_INDEX_PATH))
                logger.info(f"[FAISS] Índex reconstruït i desat a {FAISS_INDEX_PATH}")
            except Exception as e2:
                logger.warning(f"[FAISS] No s'ha pogut desar l'índex reconstruït: {e2}")
        elif vector_index.ntotal == 0:
            raise RuntimeError("Índex buit.")
    else:
        raise FileNotFoundError("Índex inexistent")
except Exception as e:
    logger.warning(f"[FAISS] No s’ha pogut carregar FAISS: {e}. Creo IndexFlatIP i hi afegeixo embeddings.")
    vector_index = faiss.IndexFlatIP(embeddings.shape[1])
    emb_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    vector_index.add(emb_norm)
    try:
        faiss.write_index(vector_index, str(FAISS_INDEX_PATH))
        logger.info(f"[FAISS] Índex pla creat i desat a {FAISS_INDEX_PATH}")
    except Exception as e2:
        logger.warning(f"[FAISS] No s'ha pogut desar l'índex pla: {e2}")

#---------------------------------------------------------------
#--------CAPÇALERA: CACHE D’EMBEDDINGS DE CONSULTA -------------
#---------------------------------------------------------------
if EMB_CACHE_PATH.exists():
    try:
        EMB_CACHE = json.load(open(EMB_CACHE_PATH, "r"))
        logger.info(f"[CACHE] Carregada cache d’embeddings ({len(EMB_CACHE)} claus)")
    except Exception as e:
        logger.warning(f"[CACHE] No es pot llegir cache: {e}")
        EMB_CACHE = {}
else:
    EMB_CACHE = {}

"""Circuit breaker per gestionar fallades d'API."""
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("[CIRCUIT] Passant a estat HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
        logger.debug("[CIRCUIT] Reset a estat CLOSED")
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"[CIRCUIT] Obert després de {self.failure_count} fallades consecutives")

"""OpenAI opcional amb capa de compatibilitat (legacy i client 1.x)."""
class OpenAIAdapter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.legacy = None
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 60.0
        self.timeout = 30.0
        self.circuit_breaker = CircuitBreaker()
        
        if not api_key:
            return
        try:
            # Try new 1.x client
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(
                api_key=api_key,
                timeout=self.timeout
            )
            logger.info("[OPENAI] Client 1.x carregat.")
        except Exception as e:
            logger.warning(f"[OPENAI] Error carregant client 1.x: {e}")
            try:
                # Fallback legacy
                import openai  # type: ignore
                openai.api_key = api_key
                self.legacy = openai
                logger.info("[OPENAI] SDK legacy carregat.")
            except Exception as e:
                logger.warning(f"[OPENAI] Error carregant legacy: {e}")
                logger.info("[OPENAI] No disponible. Mode local.")

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff and circuit breaker."""
        # Check circuit breaker first
        if not self.circuit_breaker.can_execute():
            logger.warning("[OPENAI] Circuit breaker obert - saltant crida API")
            raise RuntimeError("Circuit breaker obert - API temporalment no disponible")
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                self.circuit_breaker.on_success()
                logger.debug(f"[OPENAI] Crida exitosa (intento {attempt + 1})")
                return result
            except Exception as e:
                last_exception = e
                self.circuit_breaker.on_failure()
                
                # Log detailed error information
                error_type = type(e).__name__
                error_msg = str(e)
                logger.warning(f"[OPENAI] Intento {attempt + 1} fallat: {error_type}: {error_msg}")
                
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"[OPENAI] Reintentant en {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"[OPENAI] Tots els intents fallats després de {self.max_retries} intents. Darrer error: {error_type}: {error_msg}")
        
        raise last_exception

    def embed(self, model: str, text: str):
        if self.client is not None:
            def _embed():
                resp = self.client.embeddings.create(model=model, input=text)
                return resp.data[0].embedding
            return self._retry_with_backoff(_embed)
        if self.legacy is not None:
            def _embed_legacy():
                resp = self.legacy.Embedding.create(model=model, input=text)
                return resp["data"][0]["embedding"]
            return self._retry_with_backoff(_embed_legacy)
        raise RuntimeError("OpenAI no disponible")

    def chat(self, model: str, messages: list, temperature: float, max_tokens: int):
        if self.client is not None:
            def _chat():
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            return self._retry_with_backoff(_chat)
        if self.legacy is not None:
            def _chat_legacy():
                resp = self.legacy.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp["choices"][0]["message"]["content"]
            return self._retry_with_backoff(_chat_legacy)
        raise RuntimeError("OpenAI no disponible")

OPENAI = OpenAIAdapter(OPENAI_API_KEY)

def _persist_cache():
    try:
        EMB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EMB_CACHE_PATH, "w") as f:
            json.dump(EMB_CACHE, f)
        logger.debug(f"[CACHE] Persistida a {EMB_CACHE_PATH}")
    except Exception as e:
        logger.warning(f"[CACHE] No s'ha pogut persistir: {e}")

#---------------------------------------------------------------
#--------CAPÇALERA: EXPANSIÓ SEMÀNTICA I EMBED_QUERY ----------
#---------------------------------------------------------------
def load_topics_semantics() -> dict:
    if not TOPICS_PATH.exists():
        logger.warning("⚠️ No s'ha trobat topics_semantics.json — mode heurístic")
        return {}
    try:
        with open(TOPICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"[TOPICS] Carregats {len(data)} temes.")
        return data
    except Exception as e:
        logger.warning(f"[TOPICS] Error carregant: {e}")
        return {}

TOPICS_SEM = load_topics_semantics()

def expand_query_with_semantics(query: str) -> str:
    # 1) Expansió per fitxer semàntic si disponible
    if TOPICS_SEM:
        qn = normalize_text_local(query)
        for k, v in TOPICS_SEM.items():
            if qn == normalize_text_local(k) or qn in normalize_text_local(k):
                extra = []
                extra += v.get("categories", [])
                extra += v.get("associacions", [])
                extra += v.get("pobles_relacionats", [])
                expanded = f"{query} " + " ".join(extra)
                logger.info(f"[SEMANTIC EXPAND] {query} → {expanded}")
                return expanded

    # 2) Expansió simple per sinònims comuns (tolerant)
    q_words = [w for w in normalize_text_local(query).split() if len(w) >= 3]
    extra_syns: List[str] = []
    for w in q_words:
        syns = SIMPLE_SYNONYMS.get(w)
        if syns:
            extra_syns.extend(syns)
    if extra_syns:
        expanded = f"{query} " + " ".join(sorted(set(extra_syns)))
        logger.info(f"[SEMANTIC EXPAND][SIMPLE] {query} → {expanded}")
        return expanded

    return query

def embed_query(tema: str) -> np.ndarray:
    tema = (tema or "").strip().lower()
    tema = re.sub(r"[^a-zà-ÿ0-9\s\-]", "", tema)
    tema_key = tema

    # cache
    if tema_key in EMB_CACHE:
        v = np.array(EMB_CACHE[tema_key], dtype=np.float32)
        if v.shape[0] != embeddings.shape[1]:
            logger.warning(
                f"[CACHE] Dimensió de cache inconsistent per '{tema_key}': {v.shape[0]}D != {embeddings.shape[1]}D — regenerant."
            )
            try:
                del EMB_CACHE[tema_key]
            except Exception:
                pass
        else:
            logger.info(f"[CACHE] Hit embedding '{tema_key}' ({v.shape[0]}D)")
            return v.reshape(1, -1)

    # variants mínimes
    variants = {tema}
    if tema.endswith("s"):
        variants.add(tema[:-1])
    else:
        variants.add(tema + "s")

    tema_exp = " ".join(sorted(variants))
    query_text = f"title: {tema_exp} topics: {tema_exp} summary: {tema_exp}"

    # OpenAI si hi és
    v = None
    if OPENAI.api_key:
        try:
            emb_vec = OPENAI.embed(model="text-embedding-3-large", text=query_text)
            v = np.array(emb_vec, dtype=np.float32)
            v /= np.maximum(np.linalg.norm(v), 1e-9)
            logger.info(f"[EMBED] OpenAI creat ({v.shape[0]}D) per '{tema}'")
        except Exception as e:
            logger.warning(f"[EMBED] OpenAI ha fallat ({e}), vector local determinista")

    # Fallback local determinista
    if v is None:
        rng = np.random.default_rng(abs(hash(tema_exp)) % (2**32))
        v = rng.normal(size=(embeddings.shape[1],)).astype(np.float32)
        v /= np.maximum(np.linalg.norm(v), 1e-9)

    EMB_CACHE[tema_key] = v.tolist()
    _persist_cache()
    return v.reshape(1, -1)


#---------------------------------------------------------------
#3--------CAPÇALERA: CERCA FAISS + RE-PUNTUACIÓ -----------------
#---------------------------------------------------------------
def _base_vector_search(q_vec: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cerca bàsica a FAISS (ja normalitzat). Retorna (D, I).
    """
    if q_vec is None or not isinstance(q_vec, np.ndarray):
        logger.error("[FAISS] q_vec no vàlid o None.")
        return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)

    # Validació de dimensions abans de cercar
    try:
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        if q_vec.shape[1] != getattr(vector_index, 'd', q_vec.shape[1]):
            logger.error(f"[FAISS] Dim mismatch: q={q_vec.shape[1]} vs index={getattr(vector_index, 'd', 'unknown')}")
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)

        D, I = vector_index.search(q_vec, max(top_k, 50))
        logger.debug(f"[FAISS] Raw search → top={I.shape[1]} resultats")
        return D, I
    except Exception as e:
        logger.error(f"[FAISS] Error cercant: {e!r} (type={type(e).__name__})")
        return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)


#---------------------------------------------------------------
#--------CAPÇALERA: FILTRAT I PONDERACIÓ -----------------------
#---------------------------------------------------------------
def _weight_candidate(doc: Doc,
                      base_score: float,
                      tema_low: str,
                      poble_low: str,
                      theme_categories: Optional[List[str]] = None) -> float:
    """Pondera el score FAISS amb senyals textuals i de poble."""
    try:
        weight = 1.0
        title_low = normalize_text_local(doc.title)
        pop_low = canonicalize_population(doc.population or "")
        topics_low = [normalize_text_local(t) for t in (doc.topics or [])]
        txt_low = normalize_text_local(doc.summary_long or doc.summary or "")

        # 1) Coincidència literal/paraules clau al títol
        if tema_low and tema_low in title_low:
            weight *= 3.0
        else:
            # Coincidències exactes de paraules (no substrings)
            words = tema_low.split()
            exact_matches = 0
            for w in words:
                if w and len(w) >= 4:  # Només paraules de 4+ lletres
                    # Coincidència exacta (no substring)
                    if re.search(r'\b' + re.escape(w) + r'\b', title_low):
                        exact_matches += 1
                        weight *= 1.5
                    elif w in title_low:  # Substring com a fallback
                        weight *= 1.2
            if exact_matches == 0:
                # Si no hi ha coincidències exactes, penalitza lleugerament
                weight *= 0.8

        # 2) Poble exacte o mention al text
        if poble_low and poble_low == pop_low:
            weight *= 1.8
        elif poble_low and poble_low in txt_low:
            weight *= 1.2

        # 3) Categories semàntiques
        if theme_categories:
            for cat in theme_categories:
                if normalize_text_local(cat) in topics_low:
                    weight *= 1.15
                    break

        # 4) Penalitzacions — textos molt curts o poc informatius
        if len(txt_low) < 300:
            weight *= 0.85

        # 5) Soroll temàtic conegut (ajusta si cal)
        if any(x in topics_low for x in ["onomastica", "poesia popular", "cognoms"]):
            weight *= 0.7

        final = base_score * weight
        return float(final)
    except Exception as e:
        logger.warning(f"[WEIGHT] Error ponderant doc {getattr(doc,'id', '?')}: {e}")
        return float(base_score)


def _postfilter_tolerant(cands: List[Tuple[Doc, float]], tema_low: str, keep: int) -> List[Tuple[Doc, float]]:
    """
    Filtre tolerant final: manté si:
      - tema_low (o arrel singular) és al títol/resum/resum_long
      - o alguna paraula del tema hi és
    Si res passa, es queda els top-N.
    """
    filtered = []
    tema_root = tema_low[:-1] if tema_low.endswith("s") else tema_low
    for doc, score in cands:
        blob = " ".join([
            normalize_text_local(doc.title),
            normalize_text_local(doc.summary or ""),
            normalize_text_local(doc.summary_long or "")
        ])
        words = [w for w in tema_low.split() if len(w) >= 3]
        # també deixa passar si algun prefix de >=3 lletres apareix
        prefixes_ok = any(any(tok.startswith(w[:4]) and w[:4] in tok for tok in blob.split()) for w in words if len(w) >= 4)
        if (
            tema_low in blob
            or tema_root in blob
            or any(w in blob for w in words)
            or prefixes_ok
        ):
            filtered.append((doc, score))

    if not filtered:
        logger.warning(f"[FAISS][FILTRAT] Cap coincidència textual clara per '{tema_low}'. Retinc top {min(keep, len(cands))}.")
        return cands[:keep]

    return filtered[:keep]


#---------------------------------------------------------------
#--------CAPÇALERA: CERCA SEMÀNTICA PRINCIPAL ------------------
#---------------------------------------------------------------
def search_faiss(query_text: str,
                 poble_filter: Optional[str] = None,
                 top_k: int = 8,
                 theme_categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Cerca robusta:
      1) embed_query(query expandit semànticament)
      2) FAISS → candidats
      3) re-puntuació per senyals textuals i poble
      4) filtrat tolerant final
    Retorna llista de dicts (id, title, summary, content, population, score, ...)
    """
    if not (documents and len(documents) == embeddings.shape[0]):
        logger.error("[SEARCH] Corpus/embeddings inconsistents o buits.")
        return []

    # 1) expand i vector de consulta
    tema_net = query_text or ""
    tema_net = tema_net.strip()
    tema_exp = expand_query_with_semantics(tema_net)
    tema_vec = embed_query(tema_exp)
    tema_vec = tema_vec / np.maximum(np.linalg.norm(tema_vec), 1e-9)

    tema_low = normalize_text_local(tema_net)
    poble_low = canonicalize_population(poble_filter or "")
    logger.info(f"[SEARCH] tema='{tema_net}' (exp='{tema_exp}') | poble='{poble_filter or 'tots'}'")

    # 2) FAISS candidats
    D, I = _base_vector_search(tema_vec, top_k=max(50, top_k * 10))
    if I.size == 0:
        return []

    # 3) construir i ponderar candidats
    stack: List[Tuple[Doc, float]] = []
    for rank, doc_id in enumerate(I[0]):
        if doc_id < 0 or doc_id >= len(documents):
            continue
        doc = documents[int(doc_id)]
        base = float(D[0][rank])
        score = _weight_candidate(doc, base, tema_low, poble_low, theme_categories)

        # prioritzar poble
        if poble_low and canonicalize_population(doc.population or "") == poble_low:
            score *= 1.15

        stack.append((doc, score))

    # 4) ordenar i filtre tolerant
    stack.sort(key=lambda x: x[1], reverse=True)
    stack = _postfilter_tolerant(stack, tema_low, keep=top_k * 2)

    # 5) dedupe per id
    seen = set()
    final: List[Dict[str, Any]] = []
    for d, s in stack:
        if d.id in seen:
            continue
        seen.add(d.id)
        final.append({
            "id": d.id,
            "titol": d.title,
            "resum": d.summary or "",
            "autor": d.author or "",
            "poble": d.population or "",
            "pes": float(round(s, 3)),
            "contingut": d.summary_long or d.summary or "",
        })
        if len(final) >= top_k:
            break

    logger.info(f"[SEARCH] resultats={len(final)}; top_ids={[r['id'] for r in final]}")
    return final


def fallback_keyword_search(query_text: str,
                            poble_filter: Optional[str] = None,
                            limit: int = 8) -> List[Dict[str, Any]]:
    """
    Cerca simple per paraules clau, tolerant, sobre títol i resums.
    Útil com a darrer recurs si FAISS no troba res o massa específic.
    """
    q = normalize_text_local(query_text)
    words = [w for w in q.split() if len(w) >= 3]
    if not words:
        return []

    pob = canonicalize_population(poble_filter or "")

    scored: List[Tuple[Doc, float]] = []
    for d in documents:
        title = normalize_text_local(d.title)
        text = normalize_text_local((d.summary_long or d.summary or ""))
        blob = title + " " + text

        # filtre per poble si cal
        if pob and pob not in canonicalize_population(d.population or "") and pob not in blob:
            continue

        # compta coincidències exactes (no substrings)
        exact_matches = 0
        for w in words:
            if w and len(w) >= 4:
                if re.search(r'\b' + re.escape(w) + r'\b', blob):
                    exact_matches += 1
        
        # També busca paraules relacionades per temes comuns
        related_matches = 0
        if any(term in words for term in ["barranc", "gàfols", "sant", "antoni"]):
            # Si busca barranc, també busca jaciment, arqueologia, etc.
            related_terms = ["jaciment", "arqueologia", "arqueològic", "restes", "civilitzacions"]
            for term in related_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', blob):
                    related_matches += 1
        
        # prefixes de 4 lletres com a fallback
        prefixes = [w[:4] for w in words if len(w) >= 4]
        count_pref = sum(1 for p in prefixes if p and any(tok.startswith(p) for tok in blob.split()))

        score = exact_matches * 2.0 + related_matches * 1.5 + count_pref * 0.3
        if score > 0:
            # bonus petit si al títol
            title_bonus = sum(1 for w in words if w in title) * 0.5
            score += title_bonus
            scored.append((d, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    results: List[Dict[str, Any]] = []
    seen = set()
    for d, s in scored:
        if d.id in seen:
            continue
        seen.add(d.id)
        results.append({
            "id": d.id,
            "titol": d.title,
            "resum": d.summary or "",
            "autor": d.author or "",
            "poble": d.population or "",
            "pes": float(round(s, 3)),
            "contingut": d.summary_long or d.summary or "",
        })
        if len(results) >= limit:
            break

    logger.info(f"[FALLBACK] {len(results)} resultats per keyword per '{query_text}' pob='{poble_filter or 'tots'}'")
    return results


#---------------------------------------------------------------
#--------CAPÇALERA: NETEJA REPETICIONS I SPLIT TELEGRAM --------
#---------------------------------------------------------------
def clean_repetitions(text: str) -> str:
    """
    Elimina repeticions literals, punts dobles i redundàncies
    que sovint apareixen en resums concatenats de múltiples fonts.
    """
    logger.debug(f"[FORMAT] Netejant repeticions (len={len(text)})")

    # 🔧 Substitucions bàsiques
    text = re.sub(r'\.\s*\.', '.', text)
    text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', text)  # elimina repeticions immediates
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # 🧹 Correccions finals
    for seq in ["..", " .", " ,", " ;", " :", "  "]:
        text = text.replace(seq, seq.strip())

    # ⚙️ Normalitza majúscules inicials després de punt
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

    logger.debug(f"[FORMAT] Text net finalitzat (len={len(text)})")
    return text

#---------------------------------------------------------------
#--------CAPÇALERA: CONTROL DE MISSATGES LLARGS (TELEGRAM) -----
#---------------------------------------------------------------
def split_for_telegram(text: str, max_len: int = TELEGRAM_CHUNK) -> List[str]:
    """
    Divideix el text en parts segures per enviar a Telegram.
    Manté el format Markdown i intenta trencar pels salts de línia
    o punts, per evitar talls bruscos a mitja frase.
    """
    logger.debug(f"[SPLIT] Preparant missatge per Telegram (len={len(text)})")

    if len(text) <= max_len:
        logger.debug("[SPLIT] Text curt, no cal dividir.")
        return [text]

    parts = []
    remaining = text

    while len(remaining) > max_len:
        # Intenta tallar pel punt o salt de línia abans del límit
        cut_idx = max(
            remaining.rfind("\n", 0, max_len),
            remaining.rfind(".", 0, max_len),
            remaining.rfind("!", 0, max_len),
            remaining.rfind("?", 0, max_len),
        )
        if cut_idx == -1 or cut_idx < max_len * 0.5:
            cut_idx = max_len

        part = remaining[:cut_idx + 1].strip()
        parts.append(part)
        remaining = remaining[cut_idx + 1:].strip()

        logger.debug(f"[SPLIT] Fragment afegit (len={len(part)}), restants={len(remaining)}")

    if remaining:
        parts.append(remaining)
        logger.debug(f"[SPLIT] Últim fragment (len={len(remaining)})")

    logger.info(f"[SPLIT] Total fragments generats: {len(parts)}")
    return parts

#---------------------------------------------------------------
#--------CAPÇALERA: ENVIAMENT SEGUR A TELEGRAM -----------------
#---------------------------------------------------------------
async def send_long_message(update: Update, text: str, parse_mode="Markdown"):
    """
    Envia un text llarg a Telegram, dividint-lo automàticament si cal.
    """
    parts = split_for_telegram(text)
    for i, part in enumerate(parts):
        try:
            await update.message.reply_text(part, parse_mode=parse_mode)
            await asyncio.sleep(0.3)  # petita pausa per evitar flood
        except Exception as e:
            logger.error(f"[SEND] Error enviant fragment {i} amb {parse_mode}: {e}. Retento sense format.")
            try:
                await update.message.reply_text(part)
                await asyncio.sleep(0.3)
            except Exception as e2:
                logger.error(f"[SEND] Error també sense format al fragment {i}: {e2}")


#---------------------------------------------------------------
#--------CAPÇALERA: FORMAT DE RESPOSTA (HEADER/TEXT/FOOTER) ----
#---------------------------------------------------------------
def _format_articles_footer(results: List[Dict[str, Any]]) -> str:
    """
    Crea un peu de resposta amb els títols i enllaços dels articles trobats.
    """
    logger.debug(f"[FORMAT] Creant peu d'articles per {len(results)} resultats.")

    if not results:
        return "(Cap article disponible.)"

    footer_lines = []
    for r in results[:10]:
        art_id = r.get("id")
        title = r.get("titol", r.get("title", "Sense títol"))
        pes = r.get("pes", r.get("score", 0.0))
        # Format preferit: "/<id> Títol (Pes: XX)"
        footer_lines.append(f"/{art_id} {title} (Pes: {pes:.2f})")

    footer_text = "\n".join(footer_lines)
    logger.debug(f"[FORMAT] Peu generat amb {len(footer_lines)} línies.")
    return footer_text

#---------------------------------------------------------------
#--------CAPÇALERA: COMPOSICIÓ DE RESPOSTA (VERSIÓ COMPLETA) ---
#---------------------------------------------------------------
def compose_summary_response(tema: str,
                             results: List[Dict[str, Any]],
                             poble: Optional[str],
                             brief: bool = False) -> str:
    """
    Genera un resum fidel al corpus, limitat i amb opció d'ampliar si és llarg.
    """
    logger.debug(f"[COMPOSE] Generant resum per tema='{tema}' (brief={brief}, poble={poble})")

    if not results:
        return "(Cap resultat disponible.)"

    # Limita a 5 resultats per mostrar els articles amb més score
    top_results = results[:5]
    
    # Construir resposta fidel al corpus
    fragments = []
    for r in top_results:
        text = (r.get("resum") or r.get("contingut") or "").strip()
        if text:
            # Limita cada fragment a 200 caràcters per ser concís
            fragment = text[:200].strip()
            if len(text) > 200:
                fragment = fragment.rsplit('.', 1)[0] + '.'
            fragments.append(fragment)

    if not fragments:
        return "No s'ha trobat informació específica sobre aquest tema al corpus."

    # Uneix fragments i limita la resposta total
    full_text = " ".join(fragments)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    
    # Limita a 800 caràcters per resposta
    if len(full_text) > 800:
        resum = full_text[:800].rsplit('.', 1)[0] + '.'
        # Afegeix opció d'ampliar si la resposta és llarga
        resum += "\n\n💡 *Vols ampliar la informació?* Escriu `amplia` per obtenir més detalls."
    else:
        resum = full_text

    # 🔧 Neteja i format
    resum = clean_repetitions(resum)

    footer = "\n\nArticles:\n" + _format_articles_footer(top_results)
    resposta = resum + footer

    logger.debug(f"[COMPOSE] Resum generat ({len(resposta)} caràcters, {len(top_results)} resultats)")
    return resposta

def compose_list_response(tema: str, results: list, poble: Optional[str]) -> str:
    """
    Llista d'articles relacionats amb el tema, amb resum curt de cada article.
    """
    logger.debug(f"[COMPOSE] Llista per tema='{tema}' (poble={poble})")

    lines = []
    for r in results[:8]:  # Redueixo a 8 per tenir espai per resums
        art_id = r.get("id")
        title = r.get("titol", r.get("title", "Sense títol"))
        pes = r.get("pes", r.get("score", 0.0))
        resum = r.get("resum", r.get("contingut", ""))
        
        # Resum curt (màxim 150 caràcters)
        resum_curt = resum[:150].strip()
        if len(resum) > 150:
            resum_curt = resum_curt.rsplit('.', 1)[0] + '.'
        
        lines.append(f"/{art_id} {title} (Pes: {pes:.2f})")
        if resum_curt:
            lines.append(f"{resum_curt}")
        lines.append("")  # Línia buida entre articles

    if not lines:
        return "⚠️ No hi ha articles relacionats amb aquest tema."

    body = "\n".join(lines).strip()
    logger.debug(f"[COMPOSE] Llista generada ({len(results)} elements)")
    return f"🗂️ **Articles relacionats amb '{tema}'**\n\n{body}"

def get_article_by_id_or_query(identifier: str) -> List[Dict[str, Any]]:
    """
    Detecta si l'identificador és un ID d'article o un tema de cerca.
    Retorna una llista de resultats (1 article si és ID, múltiples si és cerca).
    """
    # Neteja l'identificador (treu / si hi és)
    clean_id = identifier.strip().lstrip('/')
    
    # Intenta detectar si és un ID numèric
    try:
        art_id = int(clean_id)
        doc = id_to_doc.get(art_id)
        if doc:
            # Converteix Doc a format de resultat
            result = {
                "id": doc.id,
                "title": doc.title or "Sense títol",
                "titol": doc.title or "Sense títol", 
                "summary": doc.summary or "",
                "contingut": doc.summary_long or doc.summary or "",
                "population": doc.population or "",
                "score": 1.0,
                "pes": 1.0
            }
            logger.debug(f"[GET_ARTICLE] Trobat article per ID {art_id}: {doc.title}")
            return [result]
        else:
            logger.debug(f"[GET_ARTICLE] ID {art_id} no trobat al corpus")
            return []
    except ValueError:
        # No és un número, tracta com a tema de cerca
        logger.debug(f"[GET_ARTICLE] Cercant tema: {clean_id}")
        return search_faiss(clean_id, top_k=3)

def create_comparison_response(temaA: str, temaB: str, resA: List[Dict], resB: List[Dict]) -> str:
    """
    Crea una comparació estructurada entre dos articles o grups d'articles.
    Enfoca en semblances i diferències del tema principal.
    """
    logger.debug(f"[COMPARISON] Creant comparació entre {temaA} i {temaB}")
    
    # Agafa el primer resultat de cada grup (el més rellevant)
    artA = resA[0] if resA else None
    artB = resB[0] if resB else None
    
    if not artA or not artB:
        return "⚠️ No s'han pogut comparar els articles sol·licitats."
    
    # Informació bàsica dels articles
    titleA = artA.get("title", "Sense títol")
    titleB = artB.get("title", "Sense títol")
    contentA = artA.get("contingut", artA.get("summary", ""))
    contentB = artB.get("contingut", artB.get("summary", ""))
    
    # Analitza el contingut per identificar el tema principal
    main_topic = analyze_main_topic(contentA, contentB, titleA, titleB)
    
    # Crea la comparació estructurada
    resposta = f"🔍 **Comparació entre articles**\n\n"
    resposta += f"**Article A:** {titleA}\n"
    resposta += f"**Article B:** {titleB}\n\n"
    
    # Genera semblances basades en el contingut real
    semblances = generate_similarities(contentA, contentB, titleA, titleB)
    resposta += "🟢 **Semblances principals:**\n"
    resposta += "\n".join(semblances) + "\n\n"
    
    # Genera diferències basades en el contingut real
    diferencies = generate_differences(contentA, contentB, titleA, titleB)
    resposta += "🔴 **Diferències destacades:**\n"
    resposta += "\n".join(diferencies) + "\n\n"
    
    # Síntesi del tema principal
    resposta += f"📋 **Síntesi del tema '{main_topic}':**\n"
    sintesi = generate_topic_synthesis(main_topic, contentA, contentB)
    resposta += sintesi + "\n\n"
    
    # Peu amb referències
    resposta += "**Articles comparats:**\n"
    resposta += f"• /{artA.get('id')} {titleA}\n"
    resposta += f"• /{artB.get('id')} {titleB}\n"
    
    logger.debug(f"[COMPARISON] Comparació generada ({len(resposta)} caràcters)")
    return resposta

def analyze_main_topic(contentA: str, contentB: str, titleA: str, titleB: str) -> str:
    """Analitza el contingut per identificar el tema principal comú."""
    # Paraules clau per identificar temes
    keywords = {
        "castell": ["castell", "fortalesa", "torre", "mur", "templer", "defensiu"],
        "geologia": ["geologia", "geològic", "roca", "mineral", "formació", "estratigrafia"],
        "paisatge": ["paisatge", "paisatgístic", "natura", "territori", "comarca", "diversitat"],
        "historia": ["història", "històric", "medieval", "antiguitat", "cronologia"],
        "arqueologia": ["arqueologia", "arqueològic", "excavació", "troballa", "vestigi"]
    }
    
    # Compta aparicions de cada tema
    topic_scores = {}
    all_text = (contentA + " " + contentB + " " + titleA + " " + titleB).lower()
    
    for topic, words in keywords.items():
        score = sum(all_text.count(word) for word in words)
        topic_scores[topic] = score
    
    # Retorna el tema amb més puntuació
    if topic_scores:
        main_topic = max(topic_scores, key=topic_scores.get)
        if topic_scores[main_topic] > 0:
            return main_topic
    
    return "tema general"

def generate_similarities(contentA: str, contentB: str, titleA: str, titleB: str) -> List[str]:
    """Genera semblances basades en el contingut real dels articles."""
    semblances = []
    
    # Analitza el contingut per trobar elements comuns
    all_text = (contentA + " " + contentB).lower()
    
    # Ubicació geogràfica
    if any(word in all_text for word in ["ribera", "ebre", "comarca"]):
        semblances.append("• Ambdós articles tracten sobre la Ribera d'Ebre")
    
    # Tipus de contingut
    if any(word in all_text for word in ["geologia", "geològic"]):
        semblances.append("• Enfocament geològic i natural")
    elif any(word in all_text for word in ["castell", "fortalesa", "templer"]):
        semblances.append("• Tracten sobre arquitectura militar i castells")
    elif any(word in all_text for word in ["paisatge", "natura", "territori"]):
        semblances.append("• Enfocament paisatgístic i territorial")
    
    # Metodologia
    if any(word in all_text for word in ["estudi", "anàlisi", "investigació"]):
        semblances.append("• Enfocament acadèmic i d'investigació")
    
    # Període temporal
    if any(word in all_text for word in ["medieval", "històric", "antiguitat"]):
        semblances.append("• Perspectiva històrica i temporal")
    
    # Si no s'han trobat semblances específiques, afegeix una genèrica
    if not semblances:
        semblances.append("• Ambdós articles formen part del corpus del CERE")
        semblances.append("• Tracten temes relacionats amb la Ribera d'Ebre")
    
    return semblances

def generate_differences(contentA: str, contentB: str, titleA: str, titleB: str) -> List[str]:
    """Genera diferències basades en el contingut real dels articles."""
    diferencies = []
    
    # Analitza títols per identificar enfocaments diferents
    titleA_lower = titleA.lower()
    titleB_lower = titleB.lower()
    
    # Diferències d'enfocament
    if "paisatge" in titleA_lower and "geològic" in titleB_lower:
        diferencies.append("• **Enfocament**: Paisatgístic vs Geològic")
    elif "castell" in titleA_lower and "geològic" in titleB_lower:
        diferencies.append("• **Enfocament**: Històric-arquitectònic vs Geològic")
    
    # Diferències de contingut específic
    if "diversitat" in titleA_lower:
        diferencies.append("• **Article A**: Enfocament en la diversitat comarcal")
    if "viatge" in titleB_lower:
        diferencies.append("• **Article B**: Enfocament en recorregut geològic")
    
    # Diferències metodològiques
    if any(word in contentA.lower() for word in ["síntesi", "anàlisi"]):
        diferencies.append("• **Metodologia A**: Enfocament sintètic i analític")
    if any(word in contentB.lower() for word in ["viatge", "recorregut"]):
        diferencies.append("• **Metodologia B**: Enfocament de recorregut i exploració")
    
    # Si no s'han trobat diferències específiques
    if not diferencies:
        diferencies.append("• **Títols**: Diferents enfocaments temàtics")
        diferencies.append("• **Contingut**: Perspectives complementàries del territori")
    
    return diferencies

def generate_topic_synthesis(topic: str, contentA: str, contentB: str) -> str:
    """Genera una síntesi del tema principal basada en el contingut real."""
    if topic == "geologia":
        return "La geologia de la Ribera d'Ebre ofereix una perspectiva única sobre la formació del territori, combinant elements naturals i històrics que configuren el paisatge actual de la comarca."
    elif topic == "paisatge":
        return "Els paisatges de la Ribera d'Ebre reflecteixen la diversitat comarcal i la riquesa natural del territori, mostrant com la geologia i la història s'entrellacen per crear un mosaic paisatgístic únic."
    elif topic == "castell":
        return "Els castells medievals de la Ribera d'Ebre representen una xarxa defensiva clau que combina influències arquitectòniques àrabs i catalanes, amb funció militar i de control territorial."
    elif topic == "historia":
        return "La història de la Ribera d'Ebre es caracteritza per la seva riquesa cultural i la seva posició estratègica, amb elements que van des de l'antiguitat fins a l'època medieval."
    else:
        return "Aquests articles ofereixen perspectives complementàries sobre la Ribera d'Ebre, mostrant la riquesa i diversitat d'aquest territori des de diferents punts de vista acadèmics i temàtics."

def get_semantic_context(topic: str) -> Optional[str]:
    if not TOPICS_SEM:
        return None

    t_norm = normalize_text_local(topic)

    # 1) exacta
    exact_map = {normalize_text_local(k): k for k in TOPICS_SEM.keys()}
    if t_norm in exact_map:
        k = exact_map[t_norm]
        v = TOPICS_SEM[k]
    else:
        # 2) parcial
        match = None
        for k in TOPICS_SEM.keys():
            kn = normalize_text_local(k)
            if kn in t_norm or t_norm in kn:
                match = k
                break
        if not match:
            # 3) intersecció de paraules
            best_match = None
            best_score = 0
            t_words = set(t_norm.split())
            for k in TOPICS_SEM.keys():
                kw = set(normalize_text_local(k).split())
                score = len(t_words & kw)
                if score > best_score:
                    best_score = score
                    best_match = k
            match = best_match if best_score > 0 else None

        if not match:
            return None
        v = TOPICS_SEM[match]
        k = match

    cats = ", ".join(v.get("categories", []))
    assocs = ", ".join(v.get("associacions", []))
    freq = v.get("freq", 0)

    return (
        f"📘 *Context semàntic del tema*: _{k}_\n"
        f"🏷️ *Categories*: {cats}\n"
        f"🔗 *Conceptes associats*: {assocs}\n"
        f"📊 *Freqüència en corpus*: {freq}\n"
    )


#---------------------------------------------------------------
#f3--------CAPÇALERA: SÍNTESI IA (amb guardes de context) --------
#---------------------------------------------------------------
def sintetitza_tema(tema: str,
                    docs_i_scores: List[Tuple[Doc, float]],
                    humor_mode: bool = False,
                    expand_mode: bool = False) -> str:
    """
    Genera una síntesi basant-se en documents filtrats per qualitat.
    Evita desbordar context (trunca) i té fallback si la IA falla.
    """
    context_parts = []
    # Diversifica: agafa documents de diferents pobles i temes
    pobles_utilitzats = set()
    # Processa tots els documents disponibles (ja filtrats per qualitat)
    for doc, score in docs_i_scores:
        title = getattr(doc, "title", "Sense títol")
        author = getattr(doc, "author", "")
        poblacio = getattr(doc, "population", "")
        anys = getattr(doc, "years", "")
        topics = ", ".join(getattr(doc, "topics", []))
        resum = str(getattr(doc, "summary", "") or "").strip()

        if not resum:
            logger.debug(f"[SINTESI] Doc sense resum: {title}")
            continue

        # Prioritza diversitat geogràfica si és possible
        if poblacio and poblacio not in pobles_utilitzats:
            pobles_utilitzats.add(poblacio)
            info = f"{title} ({poblacio}) — {author}"
        else:
            info = f"{title} ({poblacio}) — {author}"
        
        if topics:
            info += f". Temes: {topics}"
        context_parts.append(f"({score:.2f}) {info}\n{resum}")

    if not context_parts:
        return "⚠️ No hi ha prou informació per generar una síntesi."

    # Compacta i limita context (evita errors 8192 tokens)
    context_text = "\n\n".join(context_parts)
    context_text = re.sub(r'\s+', ' ', context_text)
    context_text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', context_text)
    if len(context_text) > 6000:
        context_text = context_text[:6000] + "\n[...]"

    if humor_mode:
        style = "Ets un narrador còmic que explica la història amb humor, ironia i tendresa ebrenca."
        instruction = (
            "Escriu un text humorístic i proper, amb to divulgatiu i natural. "
            "Fes riure sense perdre el rigor històric. "
            "Evita citar autors o títols. "
            "Fes recerca historiogràfica amb humor, mostrant dades valuoses del corpus."
        )
    else:
        style = "Ets un redactor històric i pedagògic expert en la Ribera d’Ebre."
        instruction = (
            "Escriu un text formatiu, coherent i objectiu en català occidental. "
            "No citis autors ni títols, no copiïs literalment el context. "
            "Fes recerca historiogràfica, mostrant dades valuoses i fets interessants del corpus."
        )

    depth = "Amplia i connecta el context cultural i social." if expand_mode else "Centra't en la informació essencial."

    prompt = (
        f"{style}\n\n"
        f"Tema: **{tema}**\n\n"
        f"{instruction}\n{depth}\n\n"
        "Estructura en tres parts sense títols de secció:\n"
        "1. INTRODUCCIÓ: MOLT curta (màxim 2-3 frases), contextualitza el tema breument\n"
        "2. DESENVOLUPAMENT: Ric en dades del corpus, cites específiques, fets importants i interessants. OBLIGATORI: Menciona almenys 3-4 pobles diferents i aspectes diversos del tema. Un paràgraf per sub-història dins de la història gran. Inclou cites directes del corpus quan sigui rellevant. NO et centris només en un lloc o aspecte.\n"
        "3. CONCLUSIÓ: Basada únicament en el que s'ha explicat al desenvolupament, no informació externa.\n\n"
        "IMPORTANT: Fes recerca historiogràfica, no text literari. Mostra informació valuosa de manera resumida. DIVERSIFICA les fonts i pobles mencionats, no et centris només en un lloc.\n\n"
        "📚 Context (no el copiïs literalment):\n"
        f"{context_text}"
    )

    # IA si disponible
    try:
        if OPENAI.api_key:
            text_resposta = OPENAI.chat(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": style},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8 if humor_mode else 0.6,
                max_tokens=900 if expand_mode else 500,
            ).strip()
            logger.info(f"[SINTESI] IA OK ({len(text_resposta)} caràcters)")
            return text_resposta
        else:
            raise RuntimeError("IA no disponible")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.warning(f"[SINTESI] Fallback local: {error_type}: {error_msg}")
        
        # Log additional context for debugging
        logger.debug(f"[SINTESI] Context length: {len(context_text)} chars")
        logger.debug(f"[SINTESI] Topic: {tema}, Humor mode: {humor_mode}, Expand mode: {expand_mode}")
        
        # Fallback: pega coherent trencada en 2-3 paràgrafs curtets
        paras = re.split(r'(?<=[.!?])\s+', context_text)
        base = " ".join(paras[:6])[:1200]
        base = clean_repetitions(base)
        return (
            f"**{tema.title()}** — síntesi preliminar (mode local)\n\n"
            f"{base}\n\n"
            f"(Aquest text s'ha generat sense IA externa per una incidència temporal: {error_type})"
        )

#---------------------------------------------------------------
#--------CAPÇALERA: MODE ARTICLE AMB PAGINACIÓ -----------------
#---------------------------------------------------------------
def create_article_mode_response(doc: Doc, uid: int) -> str:
    """
    Crea la resposta per al mode article amb capçalera, introducció curta i footer.
    Gestiona la paginació del contingut llarg.
    """
    logger.debug(f"[ARTICLE] Creant mode article per doc {doc.id}")
    
    # Capçalera amb nom i autor (sense any)
    title = doc.title or "Sense títol"
    author = doc.author or "Autor desconegut"
    
    header = f"📄 **{title}**\n"
    header += f"👤 *{author}*\n"
    header += f"📍 {doc.population or 'Ubicació no especificada'}\n\n"
    
    # Introducció curta (resum breu)
    intro = doc.summary or ""
    if not intro and doc.summary_long:
        # Si no hi ha resum, agafa els primers 200 caràcters del contingut llarg
        intro = doc.summary_long[:200].strip()
        if len(doc.summary_long) > 200:
            intro = intro.rsplit('.', 1)[0] + '.'
    
    if not intro:
        intro = "No hi ha resum disponible per aquest article."
    
    # Limita la introducció a 300 caràcters
    if len(intro) > 300:
        intro = intro[:300].rsplit('.', 1)[0] + '.'
    
    # Footer amb opcions de navegació
    footer = "\n\n📖 *Opcions:*\n"
    footer += "• [/llegir](/llegir) resum íntegre de l'article\n"
    footer += "• Fes preguntes sobre aquest article en concret\n"
    footer += "• [/nou](/nou) per iniciar consulta\n"
    
    # Desa l'article a la sessió per permetre navegació
    sess = sessions.get_session(uid)
    sess.mode = "article"
    sess.last_article = doc.id
    sess.pagina_actual = 1
    sess.last_summary = doc.summary_long or doc.summary or ""
    
    response = header + intro + footer
    logger.debug(f"[ARTICLE] Resposta generada (len={len(response)})")
    return response

def get_article_page(uid: int, page: int = 1) -> str:
    """
    Retorna una pàgina específica del article actual.
    """
    sess = sessions.get_session(uid)
    if not sess.last_article:
        return "No hi ha cap article obert."

    doc = id_to_doc.get(sess.last_article)
    if not doc:
        return "Article no disponible."

    content = doc.summary_long or doc.summary or ""
    if not content:
        return "No hi ha contingut disponible per aquest article."

    # Divideix el contingut en pàgines de 3000 caràcters (més llarg)
    page_size = 3000
    total_chars = len(content)
    total_pages = (total_chars + page_size - 1) // page_size

    if page < 1 or page > total_pages:
        return f"Pàgina no vàlida. Pàgines disponibles: 1-{total_pages}"

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_chars)
    page_content = content[start_idx:end_idx]
    
    # Assegura que el paràgraf acabi amb punt complet
    if page < total_pages and end_idx < total_chars:
        # Busca l'últim punt complet dins del contingut
        last_period = page_content.rfind('.')
        if last_period > page_size * 0.7:  # Si el punt està a més del 70% del contingut
            page_content = page_content[:last_period + 1]
            # Actualitza end_idx per la següent pàgina
            end_idx = start_idx + last_period + 1

    # Header de pàgina amb capçalera maca
    title = doc.title or "Article"
    author = doc.author or "Autor desconegut"
    year = doc.years or "Any desconegut"
    population = doc.population or "Ubicació no especificada"
    
    header = f"📄 **{title}**\n"
    header += f"👤 *{author}*\n"
    header += f"📍 {population}\n\n"
    header += f"📖 Pàgina {page} de {total_pages}\n\n"

    # Footer de pàgina amb línia en blanc
    footer = "\n\n"
    if page < total_pages:
        footer += f"📖 [/mes](/mes) per pàgina {page + 1} de {total_pages}\n"
    else:
        footer += "🏁 *Fi de l'article*\n"
    
    footer += "💬 Fes preguntes sobre l'article o [/nou](/nou) per iniciar consulta\n"

    # Actualitza la pàgina actual
    sess.pagina_actual = page

    return header + page_content + footer

#---------------------------------------------------------------
#4--------CAPÇALERA: GESTIÓ D'UNA CONSULTA ----------------------
#---------------------------------------------------------------
def _build_header(mode: str, tema: str, poble: Optional[str]) -> str:
    """Genera capçalera comuna amb Mode/Tema/Poble.

    Modes acceptats: 'cerca' (per resums), 'llista', 'article', 'arxiu'.
    Altres valors es mapegen a 'cerca'.
    """
    mode_map = {
        "llista": "Llista",
        "article": "Article",
        "arxiu": "Arxiu",
        "cerca": "Cerca",
        "resum": "Cerca",
        "resum_breu": "Cerca",
    }
    mode_label = mode_map.get((mode or "").lower(), "Cerca")
    tema_label = tema or "—"
    poble_label = poble or "Tots"
    return f"Mode: {mode_label}\nTema: {tema_label}  Poble: {poble_label}\n"

def _detect_mode_from_text(text: str) -> str:
    t = normalize_text_local(text)
    if any(x in t for x in LIST_TRIGGERS):
        return "llista"
    if any(x in t for x in RESUM_BREU_TRIGGERS):
        return "resum_breu"
    if any(x in t for x in RESUME_TRIGGERS):
        return "resum"
    return "cerca"

def _tema_from_text_or_memory(text: str, sess: SessionObj) -> str:
    t = neteja_consulta(text)
    if not t and sess.last_tema:
        logger.info(f"[CONTEXT] Manté tema anterior: {sess.last_tema}")
        return sess.last_tema
    return t or "(tema no identificat)"

#---------------------------------------------------------------
#--------CAPÇALERA: GESTIÓ PRINCIPAL DE CONSULTA ----------------
#---------------------------------------------------------------
import random
from typing import Optional

def parse_query_with_ai(text: str) -> Dict[str, Any]:
    """
    Usa IA per analitzar la consulta i extreure tema, location i mode nets.
    Retorna: {tema: [list], location: str, mode: str, is_conversa: bool}
    """
    if not OPENAI.api_key:
        # Fallback sense IA - usa lògica actual
        return _parse_query_fallback(text)
    
    try:
        prompt = f"""Analitza aquesta consulta d'usuari i extreu la informació estructurada:

CONSULTA: "{text}"

REGLES IMPORTANTS:
- FILTRA paraules buides: "parlam", "dels", "del", "de la", "de les", "a la", "sobre", "importants", "comarca", "ribera", "tots", "totes", "q", "que", "saps", "sabes", "conoces", "conegut", "?", "¿", "dime", "digues", "explica", "parla"
- TEMA: Només paraules clau del CONTINGUT real (història, castells, ibers, arqueologia, riu Ebre, etc.)
- LOCATION: Poble específic o "tots" (Ribera d'Ebre, comarca, ribera = "tots")
- MODE: "cerca", "llista", "resum", "conversa", "amplia" (si vol ampliar resposta anterior)
- IS_CONVERSA: true/false si és salutació/xerrada casual

COMANDES ESPECIALS:
- Si diu "amplia" → mode: "amplia", tema: del context anterior
- Si diu "llista" → mode: "llista" 
- Si diu "arxiu" → mode: "arxiu"
- Si diu "nou" → mode: "nou" (neteja tot)

Exemples:
- "parlam dels castells de la Ribera" → tema: ["castells"], location: "tots", mode: "cerca"
- "jaciments arqueologics importants de la comarca" → tema: ["jaciments", "arqueologics"], location: "tots", mode: "cerca"
- "q saps del riu Ebre" → tema: ["riu", "ebre"], location: "tots", mode: "cerca"
- "que sabes sobre castells" → tema: ["castells"], location: "tots", mode: "cerca"
- "castell fortalesa" → tema: ["castell", "fortalesa"], location: "tots", mode: "cerca"
- "castell ribera d'ebre" → tema: ["castell"], location: "tots", mode: "cerca"
- "amplia tema ibers" → tema: ["ibers"], location: "tots", mode: "amplia"
- "llista castells" → tema: ["castells"], location: "tots", mode: "llista"
- "hola com estàs" → tema: [], location: "tots", mode: "conversa", is_conversa: true

Respon SOL en format JSON:
{{"tema": ["paraula1", "paraula2"], "location": "poble_o_tots", "mode": "mode", "is_conversa": true/false}}"""

        response = OPENAI.chat(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        ).strip()
        
        # Parseja la resposta JSON
        import json
        result = json.loads(response)
        
        # Valida i neteja
        tema = result.get("tema", [])
        if not isinstance(tema, list):
            tema = []
        tema = [t.strip().lower() for t in tema if t.strip()]
        
        location = result.get("location", "tots").strip().lower()
        if location in ["ribera d'ebre", "ribera", "comarca", "tota la comarca"]:
            location = "tots"
        
        mode = result.get("mode", "cerca").strip().lower()
        is_conversa = result.get("is_conversa", False)
        
        # Detecta ambigüitats
        is_ambiguous = _detect_ambiguity(text, tema, location, mode)
        
        logger.info(f"[AI-PARSE] '{text}' → tema={tema}, location={location}, mode={mode}, conversa={is_conversa}, ambiguous={is_ambiguous}")
        return {
            "tema": tema,
            "location": location,
            "mode": mode,
            "is_conversa": is_conversa,
            "is_ambiguous": is_ambiguous
        }
        
    except Exception as e:
        logger.warning(f"[AI-PARSE] Error: {e}, usant fallback")
        return _parse_query_fallback(text)

def _handle_ambiguous_query(text: str, tema_list: list, location: str, mode: str) -> str:
    """
    Gestiona consultes ambigües demanant aclariments a l'usuari.
    """
    logger.debug(f"[AMBIGUOUS] Consulta ambigua: '{text}'")
    
    # Si no hi ha tema clar
    if not tema_list or len(tema_list) == 0:
        return (
            "🤔 No he entès bé què vols saber. Pots ser més específic?\n\n"
            "Exemples:\n"
            "• `castells de la Ribera`\n"
            "• `història d'Ascó`\n"
            "• `llista jaciments arqueològics`\n"
            "• `resum sobre els ibers`"
        )
    
    # Si el tema és molt genèric
    if len(tema_list) == 1 and tema_list[0] in ["història", "històries", "informació", "info"]:
        return (
            "🤔 Sobre què vols informació exactament?\n\n"
            "Pots especificar:\n"
            "• Un poble concret (Ascó, Miravet, etc.)\n"
            "• Un tema específic (castells, ibers, arqueologia, etc.)\n"
            "• Un període històric (edat mitjana, guerra civil, etc.)"
        )
    
    # Si el tema és "riu" o similar
    if len(tema_list) == 1 and tema_list[0] in ["riu", "rius", "aigua", "pont", "ponts"]:
        return (
            "🤔 Sobre què vols saber dels rius?\n\n"
            "Pots especificar:\n"
            "• `riu Ebre` - el riu principal\n"
            "• `hidrografia` - cursos d'aigua de la zona\n"
            "• `ponts` - ponts històrics\n"
            "• `vida fluvial` - ecosistema del riu\n"
            "• `navegació` - transport per riu"
        )
    
    # Si el tema pot ser cognom o tema
    if len(tema_list) == 1 and tema_list[0] in ["rius", "castell", "castells", "garcia", "garcía", "mora", "torre", "palma", "gàfols", "gafols", "sant", "antoni"]:
        if tema_list[0] in ["gàfols", "gafols", "sant", "antoni"]:
            return (
                f"🤔 Vols informació sobre '{tema_list[0]}' com a:\n\n"
                "• **Jaciment arqueològic** (Barranc dels Gàfols, Sant Antoni, etc.)\n"
                "• **Lloc geogràfic** (barranc, ermita, etc.)\n"
                "• **Cognom/persona** (Gàfols, Sant Antoni, etc.)\n\n"
                "Pots especificar:\n"
                f"• `barranc {tema_list[0]}` - si és jaciment\n"
                f"• `ermita {tema_list[0]}` - si és lloc religiós\n"
                f"• `família {tema_list[0].title()}` - si és cognom"
            )
        else:
            return (
                f"🤔 Vols informació sobre '{tema_list[0]}' com a:\n\n"
                "• **Tema/contingut** (riu, castell, etc.)\n"
                "• **Cognom/persona** (Rius, Castell, etc.)\n"
                "• **Poble** (Garcia, Mora, Torre, etc.)\n\n"
                "Pots especificar:\n"
                f"• `{tema_list[0]} Ebre` - si  història\n"
                f"• `família {tema_list[0].title()}` - si és cognom\n"
                f"• `poble {tema_list[0].title()}` - si és població"
            )
    
    # Si la consulta és massa curta
    if len(text.strip()) < 3:
        return (
            "🤔 La consulta és massa curta. Pots explicar-me què vols saber?\n\n"
            "Exemples:\n"
            "• `castells`\n"
            "• `història d'Ascó`\n"
            "• `llista jaciments`"
        )
    
    # Fallback genèric
    return (
        "🤔 No t'estic entenent del tot. Pots preguntar-ho de manera diferent?\n\n"
        "Prova amb:\n"
        "• Un tema més específic\n"
        "• Mencionar un poble concret\n"
        "• Usar paraules clau clares"
    )

def _detect_ambiguity(text: str, tema: list, location: str, mode: str) -> bool:
    """
    Detecta si la consulta és ambigua i necessita aclariments.
    """
    # Consulta massa curta o buida
    if len(text.strip()) < 3:
        return True
    
    # Sense tema clar
    if not tema or len(tema) == 0:
        return True
    
    # Paraules molt genèriques sense context
    generic_words = ["cosas", "coses", "informació", "info", "tema", "temes", "història", "històries", "aigua", "pont", "ponts"]
    if len(tema) == 1 and tema[0] in generic_words:
        return True
    
    # Paraules que s'expandeixen automàticament (no són ambiguas)
    auto_expand_words = ["castell", "castells", "riu", "rius"]
    if len(tema) == 1 and tema[0] in auto_expand_words:
        # Aquestes paraules s'expandeixen automàticament, no són ambiguas
        return False
    
    # Paraules que poden ser cognoms o temes segons majúscula
    ambiguous_names = ["garcia", "garcía", "mora", "torre", "palma", "gàfols", "gafols", "sant", "antoni"]
    if len(tema) == 1 and tema[0] in ambiguous_names:
        # Si és minúscula, probablement és tema genèric
        if tema[0].islower():
            return True
        # Si és majúscula, probablement és cognom/poble específic
        # No és ambigu en aquest cas
    
    # Si hi ha context suficient (múltiples paraules), no és ambigu
    if len(tema) > 1:
        return False
    
    # Consulta ambigua amb múltiples possibles significats
    ambiguous_phrases = [
        "que vols", "que vull", "que fas", "que faig", 
        "com va", "com està", "que tal", "que passa",
        "explica", "parla", "dime", "digues"
    ]
    if any(phrase in text.lower() for phrase in ambiguous_phrases):
        return True
    
    # Mode no reconegut
    if mode not in ["cerca", "llista", "resum", "conversa", "amplia", "arxiu", "nou"]:
        return True
    
    return False

def _parse_query_fallback(text: str) -> Dict[str, Any]:
    """Fallback sense IA - usa lògica actual"""
    # Neteja bàsica
    text_net = neteja_consulta(text)
    
    # Detecta conversa
    if detectar_conversa(text):
        return {"tema": [], "location": "tots", "mode": "conversa", "is_conversa": True}
    
    # Neteja agressiva
    tema_net = re.sub(r'\b(parlam|parlem|parla|parle|dels|del|de la|de les|de l\'|a la|a les|a l\'|sobre|de|d\'|en|a|al|als)\b', ' ', text_net, flags=re.IGNORECASE)
    tema_net = re.sub(r'\s+', ' ', tema_net).strip()
    
    # Detecta poble
    poble_detectat = detecta_poble(tema_net)
    if any(marker in normalize_text_local(tema_net) for marker in GENERAL_MARKERS):
        for marker in GENERAL_MARKERS:
            tema_net = re.sub(r'\b' + re.escape(marker) + r'\b', ' ', tema_net, flags=re.IGNORECASE)
        tema_net = re.sub(r'\s+', ' ', tema_net).strip()
        poble_detectat = None
    
    # Detecta mode
    detected_mode = _detect_mode_from_text(text)
    
    tema_list = tema_net.split() if tema_net else []
    is_ambiguous = _detect_ambiguity(text, tema_list, poble_detectat or "tots", detected_mode)
    
    return {
        "tema": tema_list,
        "location": poble_detectat or "tots",
        "mode": detected_mode,
        "is_conversa": False,
        "is_ambiguous": is_ambiguous
    }

def detectar_conversa(text: str) -> Optional[str]:
    """
    Detecta si el text és una salutació o expressió de conversa casual.
    Retorna una resposta amable o None si no n’hi ha.
    """
    t = normalize_text_local(text.lower())

    # 👋 Salutacions
    if any(x in t for x in ["hola", "ei", "bones", "bon dia", "bona tarda", "bona nit", "que tal", "què tal", "com va"]):
        resposta = random.choice([
            "Hola! 😊 Com va tot per la Ribera?",
            "Bon dia! ☀️ Què et puc explicar avui?",
            "Bones! 👋 Tens algun tema o poble al cap?",
            "Ei! 😄 Encantat de saludar-te.",
        ])
        logger.debug(f"[CONVERSA] Salutació detectada → {resposta}")
        return resposta

    # 👋 Comiats
    if any(x in t for x in ["adeu", "adéu", "fins després", "fins aviat", "ens veiem"]):
        resposta = random.choice([
            "Adéu! 👋 Cuida't molt.",
            "Fins aviat! 😄 Que vagi molt bé!",
            "Ens veiem! 🌞",
        ])
        logger.debug(f"[CONVERSA] Comiat detectat → {resposta}")
        return resposta

    # 🤖 Converses generals
    if any(x in t for x in ["com estàs", "que tal estas", "com et trobes", "què fas", "que fas", "que tal el dia", "quin dia fa", "quin temps fa"]):
        resposta = random.choice([
            "Tot bé, gràcies! 🤖 Sempre a punt per parlar de la Ribera d’Ebre.",
            "Ben content de veure’t per aquí! ☺️",
            "Jo sempre estic llest per història i pobles! I tu, com estàs?",
            "Estic bé, gràcies! I tu què tal? ☀️",
        ])
        logger.debug(f"[CONVERSA] Xerrada detectada → {resposta}")
        return resposta

    return None


def handle_text(uid: int, text: str):
    """
    Processa una consulta normal d'usuari amb IA:
    1️⃣ IA analitza la consulta (tema, location, mode)
    2️⃣ Gestiona conversa si cal
    3️⃣ Gestiona mode article si cal
    4️⃣ Cerca FAISS amb paràmetres nets
    5️⃣ Genera resposta final
    """
    sess = sessions.get_session(uid)
    logger.debug(f"[HANDLE] Inici per {uid} amb text='{text[:80]}'...")

    # 📖 Gestiona mode article
    if sess.mode == "article":
        text_lower = text.lower().strip()
        if text_lower in ["llegir", "llegir article", "llegir sencer"] or text == "/llegir":
            # Mostra la primera pàgina del article
            return get_article_page(uid, 1)
        elif text_lower == "mes" or text == "/mes":
            # Mostra la següent pàgina
            return get_article_page(uid, sess.pagina_actual + 1)
        else:
            # Pregunta sobre l'article - fa cerca semàntica sobre el contingut
            if sess.last_article:
                doc = id_to_doc.get(sess.last_article)
                if doc:
                    # Cerca dins del contingut de l'article
                    content = doc.summary_long or doc.summary or ""
                    if content and text_lower in normalize_text_local(content):
                        # Si la pregunta està relacionada amb el contingut, respon
                        return f"💬 Sobre l'article '{doc.title}':\n\nLa teva pregunta '{text}' està relacionada amb el contingut de l'article. Pots escriure `/llegir` per veure tot el contingut o fer preguntes més específiques."
                    else:
                        # Si no està relacionada amb l'article, surt del mode article i fa cerca normal
                        sess.mode = "normal"
                        logger.debug(f"[HANDLE] Surt del mode article per pregunta no relacionada: '{text}'")
                        # Continua amb la cerca semàntica normal

    # 🤖 IA analitza la consulta
    parsed = parse_query_with_ai(text)
    tema_list = parsed["tema"]
    location = parsed["location"]
    mode = parsed["mode"]
    is_conversa = parsed["is_conversa"]
    is_ambiguous = parsed.get("is_ambiguous", False)
    
    # 🤔 Gestiona ambigüitats
    if is_ambiguous:
        return _handle_ambiguous_query(text, tema_list, location, mode)
    
    # 🗣️ Gestiona conversa si cal
    if is_conversa or mode == "conversa":
        resposta_conv = detectar_conversa(text)
        if resposta_conv:
            logger.debug(f"[HANDLE] Mode conversa activat → resposta curta enviada.")
            return resposta_conv
    
    # 🔧 Gestiona comandes especials
    if mode == "nou":
        sessions.reset_session(uid)
        return "🔄 Sessió reiniciada. Pots començar una nova consulta!"
    
    if mode == "amplia":
        if not sess.last_tema:
            return "No hi ha cap tema recent per ampliar. Fes una consulta primer."
        # Usa el tema anterior per ampliar
        tema_list = sess.last_tema.split() if sess.last_tema else []
        logger.debug(f"[HANDLE] Mode amplia - tema anterior: {tema_list}")
    
    if mode == "llista":
        # Força mode llista independentment del que digui la IA
        mode = "llista"
    
    if mode == "arxiu":
        # Força mode arxiu
        mode = "arxiu"
    
    # 🧩 Prepara tema i location nets
    tema_net = " ".join(tema_list) if tema_list else ""
    poble_filter = None if location == "tots" else location
    
    # 🧩 Actualitza estat de sessió
    if poble_filter:
        sess.poble = poble_filter
    elif location == "tots":
        sess.poble = None
    
    sess.last_tema = tema_net
    log_session_state(sess, "[HANDLE-STATE]")

    # 🔍 Cerca semàntica
    try:
        results = search_faiss(tema_net, poble_filter=poble_filter, top_k=8)
    except Exception as e:
        logger.error(f"[HANDLE] Error durant la cerca FAISS: {e}")
        return "⚠️ S'ha produït un error durant la cerca semàntica."

    if not results:
        logger.warning(f"[HANDLE] Cap resultat per '{tema_net}' (poble={poble_filter}) — provo fallback per paraules clau")
        fb = fallback_keyword_search(tema_net, poble_filter=poble_filter, limit=8)
        if not fb:
            return f"⚠️ No s'han trobat resultats per al tema '{tema_net}'."
        results = fb

    # 🧠 Genera resposta final
    resposta_final = generar_resposta_final(sess, tema_net, poble_filter, results, mode=mode)
    sessions.add_to_memory(uid, resposta_final)
    logger.debug(f"[HANDLE] Resposta generada per {uid}, len={len(resposta_final)}")

    return resposta_final


#---------------------------------------------------------------
#--------CAPÇALERA: GENERACIÓ DE RESPOSTA FINAL ----------------
#---------------------------------------------------------------
def generar_resposta_final(sess, tema_net, poble, results, mode="resum"):
    """
    Combina la informació de FAISS, síntesi IA i format Markdown.
    Retorna la resposta final que s’enviarà a l’usuari.
    """
    logger.debug(f"[GEN] Generant resposta final: tema='{tema_net}', poble={poble}, mode={mode}")

    # 🧩 1. Prepara documents per a síntesi intel·ligent (dinàmic segons qualitat)
    docs_complets: List[Tuple[Doc, float]] = []
    
    # Lògica intel·ligent segons distribució de pesos
    pes_minim = 0.3  # Pes mínim per incloure un article
    max_docs = 6     # Màxim de documents a processar
    
    # Calcula la distribució de pesos
    pesos = [item.get("score", 0.0) for item in results[:max_docs] if item.get("score", 0.0) >= pes_minim]
    
    if pesos:
        pes_maxim = max(pesos)
        pes_mitja = sum(pesos) / len(pesos)
        
        # Si hi ha un article dominant (pes > 1.5x la mitja), centra't en ell
        if pes_maxim > pes_mitja * 1.5:
            # Agafa l'article dominant + 2-3 més per complementar
            docs_complets = []
            for item in results[:max_docs]:
                pes = item.get("score", 0.0)
                if pes >= pes_minim:
                    d = id_to_doc.get(item["id"])
                    if d:
                        docs_complets.append((d, pes))
                        # Si és l'article dominant, agafa 2-3 més
                        if pes == pes_maxim and len(docs_complets) >= 3:
                            break
        else:
            # Pesos similars: diversifica més
            for item in results[:max_docs]:
                pes = item.get("score", 0.0)
                if pes >= pes_minim:
                    d = id_to_doc.get(item["id"])
                    if d:
                        docs_complets.append((d, pes))
    
    # Si no hi ha prou resultats de qualitat, agafa els millors disponibles
    if len(docs_complets) < 2:
        for item in results[:3]:  # Fallback: agafa els 3 millors
            d = id_to_doc.get(item["id"])
            if d:
                docs_complets.append((d, item.get("score", 1.0)))
    
    # Log de la lògica aplicada
    if pesos:
        pes_maxim = max(pesos)
        pes_mitja = sum(pesos) / len(pesos)
        if pes_maxim > pes_mitja * 1.5:
            logger.debug(f"[GEN] Article dominant detectat (pes: {pes_maxim:.2f} vs mitja: {pes_mitja:.2f}) - centrant-se en l'article principal + 2-3 més")
        else:
            logger.debug(f"[GEN] Pesos similars (max: {pes_maxim:.2f}, mitja: {pes_mitja:.2f}) - diversificant")
    
    logger.debug(f"[GEN] Utilitzant {len(docs_complets)} documents per síntesi")

    # 🧠 2. Sintetitza contingut formatiu (IA)
    try:
        if docs_complets:
            resum_formatiu = sintetitza_tema(tema_net, docs_complets, humor_mode=sess.humor_mode)
        else:
            resum_formatiu = ""
    except Exception as e:
        logger.warning(f"[GEN] sintetitza_tema ha fallat: {e}")
        resum_formatiu = ""

    # Guarda mapping d'articles mostrats (per comandes /1.. /10)
    try:
        sess.articles_mostrats = [int(r.get("id")) for r in results[:10]]
    except Exception:
        sess.articles_mostrats = []

    # 🧱 3. Cos principal segons mode
    if mode == "llista":
        body = compose_list_response(tema_net, results, poble)
    elif mode == "resum_breu":
        body = compose_summary_response(tema_net, results, poble, brief=True)
    else:
        body = compose_summary_response(tema_net, results, poble, brief=False)

    # 🧩 4. Si hi ha una síntesi vàlida, la fem servir com a cos principal
    # PERÒ respecta el mode "llista" - no sobreescriguis la llista amb síntesi
    if resum_formatiu and "⚠️" not in resum_formatiu and mode != "llista":
        body_main = resum_formatiu
        if "Articles:" in body:
            _, articles = body.split("Articles:", 1)
            body_main += f"\n\nArticles:\n{articles.strip()}"
    else:
        body_main = body

    # 🏷️ 5. Header i footer contextual (Mode/Tema/Poble)
    # tema_net ja està netejat, l'usem directament
    header = _build_header(mode, tema_net, poble)
    footer = ""

    resposta = f"{header}\n{body_main.strip()}{footer}"
    logger.debug(f"[GEN] Resposta final composta (len={len(resposta)})")

    sess.last_summary = resposta
    sess.last_mode = mode
    return resposta

#---------------------------------------------------------------
#f4--------CAPÇALERA: COMANDES D’AMPLIACIÓ I HISTÒRIC ------------
#---------------------------------------------------------------
def cmd_amplia(uid: int) -> str:
    """
    Amplia l’últim resum:
      - intenta injectar nous fragments del mateix tema (nous docs)
      - alterna humor_mode per donar variació si l’usuari insisteix
    """
    sess = sessions.get_session(uid)
    if not sess.last_tema:
        return "No hi ha cap tema recent per ampliar."

    logger.debug(f"[AMPLIA] tema='{sess.last_tema}' pob='{sess.poble}' humor={sess.humor_mode}")
    # recerca fresca (evita repetir exactament la mateixa combinació)
    results = search_faiss(sess.last_tema, poble_filter=sess.poble, top_k=6)
    if not results:
        return "No hi ha més informació per ampliar aquest tema."

    # canvia el to (opcional)
    sess.humor_mode = not sess.humor_mode

    docs_complets: List[Tuple[Doc, float]] = []
    for r in results[:3]:
        d = id_to_doc.get(r["id"]) 
        if d:
            docs_complets.append((d, r.get("score", 1.0)))

    extra = sintetitza_tema(sess.last_tema, docs_complets, humor_mode=sess.humor_mode, expand_mode=True)
    sess.last_summary = (sess.last_summary or "") + "\n\n" + extra
    sessions.add_to_memory(uid, extra)

    mode = "irònic" if sess.humor_mode else "formatiu"
    logger.info(f"[AMPLIA] Generat extra ({mode}) len={len(extra)}")
    return f"🌀 Mode {mode}:\n\n{extra}"

def cmd_mes(uid: int) -> str:
    """
    Retorna l'últim resum/síntesi guardat (per continuar lectura).
    Si estem en mode article, mostra la següent pàgina.
    """
    sess = sessions.get_session(uid)
    
    # Si estem en mode article, mostra la següent pàgina
    if sess.mode == "article" and sess.last_article:
        return get_article_page(uid, sess.pagina_actual + 1)
    
    # Mode normal - últim resum
    if sess.last_summary:
        logger.debug(f"[MES] Retornant últim resum per {uid}")
        return sess.last_summary
    return "Encara no s'ha generat cap resum previ."

async def do_mes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler per /mes - mostra l'últim resum o la següent pàgina d'article."""
    uid = update.effective_user.id
    logger.info(f"[TG-HANDLER]/mes uid={uid}")
    resposta = cmd_mes(uid)
    parts = split_for_telegram(resposta)
    for part in parts:
        await send_long_message(update, part)

def cmd_tot(uid: int) -> str:
    """
    Mostra l’històric recent de la sessió fragmentat amb separadors.
    """
    sess = sessions.get_session(uid)
    if not sess.memoria:
        return "No hi ha cap resum guardat encara."
    logger.debug(f"[TOT] Històric per {uid}: {len(sess.memoria)} entrades.")
    blocs = "\n\n---\n\n".join(sess.memoria)
    return f"🧾 **Històric de consultes recents**\n\n{blocs}"

def cmd_reset(uid: int) -> str:
    """
    Reinicia la sessió.
    """
    sessions.reset_session(uid)
    logger.info(f"[RESET] Sessió reiniciada per {uid}")
    return "🔄 Sessió reiniciada. Pots començar una nova consulta!"

#---------------------------------------------------------------
#5--------CAPÇALERA: HANDLERS TELEGRAM (PART A) -----------------
#---------------------------------------------------------------
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

#---------------------------------------------------------------
#--------CAPÇALERA: /START, /AJUDA I /EXPERT -------------------
#---------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler per /start — missatge d’inici i info bàsica"""
    uid = update.effective_user.id
    sessions.reset_session(uid)
    logger.info(f"[TG-HANDLER]/start usuari={uid}")
    await update.message.reply_text(
        "👋 Benvingut al *MisCEREbot*!\n\n"
        "Pots preguntar-me sobre temes, pobles o articles de la Ribera d’Ebre.\n"
        "Exemples:\n• `Història del castell d’Ascó`\n• `Articles sobre Miravet`\n• `Arqueologia a la Ribera`\n\n"
        "Escriu /ajuda per veure totes les opcions.",
        parse_mode="Markdown"
    )

async def ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler per /ajuda — mostra l’ajuda bàsica"""
    logger.info(f"[TG-HANDLER]/ajuda uid={update.effective_user.id}")
    ajuda_text = (
        "📘 *MisCEREbot — Ajuda bàsica*\n\n"
        "• Escriu un tema per cercar articles del corpus del CERE.\n\n"
        "• Usa /poble NomDelPoble - per filtrar per poble.\n"
        "• Usa /tema text per fixar un tema concret.\n"
        "• /amplia - per obtenir més detalls del darrer tema.\n"
        "• /mes o /tot - per veure resums previs o l’històric.\n"
        "• /reset o /nou - per començar de nou.\n"
        "• /expert - per mode avançat.\n"
    )
    await update.message.reply_text(ajuda_text, parse_mode="Markdown")

async def expert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler per /expert — mostra ajuda avançada"""
    logger.info(f"[TG-HANDLER]/expert uid={update.effective_user.id}")
    expert_text = (
        "🤖 *MisCEREbot — Mode expert / Ajuda avançada*\n\n"
        "El mode expert permet treballar directament sobre l’arxiu amb filtres i comandes específiques.\n\n"
        "⚙️ *Filtres i context*\n"        
        "• /poble NomDelPoble — Filtra tots els resultats pel poble indicat.\n"
        "• /tema TextTema — Centra la cerca en un tema específic.\n"
        "• /nou — Reinicia el context i neteja tots els filtres actius.\n\n"
        "📚 *Accés i navegació*\n"
        "• /arxiu — Mostra l’article obert o el corpus filtrat.\n"
        "• /tot — Mostra tot el contingut consultat.\n"
        "• /mes — Mostra la següent part d’un article llarg.\n"
        "• /creua A B — Compara o sintetitza dos articles o temes.\n"
        "• /ajuda — Ajuda bàsica.\n"
    )
    await update.message.reply_text(expert_text, parse_mode="Markdown")

#---------------------------------------------------------------
#--------CAPÇALERA: /NOU I /RESET ------------------------------
#---------------------------------------------------------------
async def cmd_nou(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler per /nou — reinicia sessió"""
    uid = update.effective_user.id
    sessions.reset_session(uid)
    logger.info(f"[TG-HANDLER]/nou usuari={uid}")
    await update.message.reply_text("🔄 Sessió reiniciada. Pots començar una nova consulta!")

#---------------------------------------------------------------
#--------CAPÇALERA: /POBLE -------------------------------------
#---------------------------------------------------------------
async def cmd_poble(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler per /poble — fixa el poble actiu"""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    args = context.args
    if not args:
        await update.message.reply_text(f"🏙️ Poble actual: {sess.poble or 'cap'}")
        return

    poble_nom = " ".join(args)
    pob = canonicalize_population(poble_nom)
    if pob not in KNOWN_POPS:
        await update.message.reply_text(f"⚠️ No conec el poble '{poble_nom}'.")
        return

    sess.poble = pob
    sessions.update_session(uid, poble=pob)
    logger.info(f"[TG-HANDLER]/poble set={pob} uid={uid}")
    await update.message.reply_text(f"✅ Filtre de poble activat: *{pob.title()}*", parse_mode="Markdown")

#---------------------------------------------------------------
#--------CAPÇALERA: /TEMA --------------------------------------
#---------------------------------------------------------------
async def cmd_tema(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler per /tema — fixa un tema i llança cerca"""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    tema_txt = " ".join(context.args)
    if not tema_txt:
        await update.message.reply_text("ℹ️ Escriu un tema després de /tema, per exemple: `/tema guerra civil`", parse_mode="Markdown")
        return

    logger.info(f"[TG-HANDLER]/tema='{tema_txt}' pob='{sess.poble}' uid={uid}")
    resposta = handle_text(uid, tema_txt)
    parts = split_for_telegram(resposta)
    for part in parts:
        await send_long_message(update, part)

#---------------------------------------------------------------
#--------CAPÇALERA: /ID ----------------------------------------
#---------------------------------------------------------------
async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler per /id — mostra informació d'un article concret"""
    uid = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text("ℹ️ Indica un ID d'article: `/id 123`")
        return
    try:
        art_id = int(args[0])
    except ValueError:
        await update.message.reply_text("⚠️ L'ID ha de ser un número.")
        return

    doc = id_to_doc.get(art_id)
    if not doc:
        await update.message.reply_text(f"❌ No s'ha trobat l'article amb ID {art_id}")
        return

    # Usa la nova funció de mode article
    text = create_article_mode_response(doc, uid)
    logger.info(f"[TG-HANDLER]/id={art_id} uid={uid}")
    await send_long_message(update, text, parse_mode="Markdown")

async def cmd_pick_n(update: Update, context: ContextTypes.DEFAULT_TYPE, n: int):
    """Obre l'article n de l'última llista (1..10)."""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    if n < 1 or n > 10:
        await update.message.reply_text("⚠️ Usa un número entre 1 i 10.")
        return
    if not sess.articles_mostrats or len(sess.articles_mostrats) < n:
        await update.message.reply_text("ℹ️ No hi ha cap llista recent d'articles.")
        return
    art_id = sess.articles_mostrats[n - 1]
    doc = id_to_doc.get(art_id)
    if not doc:
        await update.message.reply_text("❌ Article no disponible.")
        return
    
    # Usa la nova funció de mode article
    text = create_article_mode_response(doc, uid)
    await send_long_message(update, text, parse_mode="Markdown")

async def handle_numeric_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler general per comandes com /73 → obre l'article amb ID 73."""
    text = (update.message.text or "").strip()
    logger.info(f"[NUMERIC] Handler cridat amb text: '{text}'")
    m = re.match(r"^\/(\d+)\b", text)
    if not m:
        logger.warning(f"[NUMERIC] No match per text: '{text}'")
        return
    art_id = int(m.group(1))
    logger.info(f"[NUMERIC] Article ID detectat: {art_id}")
    doc = id_to_doc.get(art_id)
    if not doc:
        await update.message.reply_text(f"❌ No s'ha trobat l'article amb ID {art_id}")
        return
    
    # Usa la nova funció de mode article
    uid = update.effective_user.id
    logger.info(f"[NUMERIC] Processant article {art_id} per usuari {uid}")
    text = create_article_mode_response(doc, uid)
    logger.info(f"[NUMERIC] Resposta generada: {text[:100]}...")
    await send_long_message(update, text, parse_mode="Markdown")

#---------------------------------------------------------------
#--------CAPÇALERA: HANDLERS TELEGRAM (PART B) -----------------
#---------------------------------------------------------------

#---------------------------------------------------------------
#--------CAPÇALERA: /ARXIU I /ARXIU_CERCA ----------------------
#---------------------------------------------------------------
async def cmd_arxiu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra la llista o contingut del corpus filtrat."""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    poble = sess.poble or "tots"
    logger.info(f"[TG-HANDLER]/arxiu uid={uid} poble={poble}")

    results = search_faiss(sess.last_tema or "", poble_filter=sess.poble, top_k=12)
    if not results:
        await update.message.reply_text(f"⚠️ No hi ha resultats per al poble *{poble}*", parse_mode="Markdown")
        return

    header = _build_header("arxiu", (sess.last_tema or "(Sense tema)").strip(), sess.poble)
    msg = header
    for i, r in enumerate(results, start=1):
        title = r.get("title", "Sense títol")
        score = r.get("score", 0.0)
        msg += f"{i}. [{title}](https://t.me/misCEREbot/{r.get('id')}) — {score:.2f}\n"

    parts = split_for_telegram(msg)
    for part in parts:
        await send_long_message(update, part)
                
async def cmd_arxiu_cerca(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cerca dins l’arxiu o corpus."""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("ℹ️ Escriu una paraula clau: `/arxiu_cerca miravet`", parse_mode="Markdown")
        return

    logger.info(f"[TG-HANDLER]/arxiu_cerca uid={uid} query='{query}' poble={sess.poble}")
    results = search_faiss(query, poble_filter=sess.poble, top_k=10)
    if not results:
        await update.message.reply_text("⚠️ Cap resultat trobat dins l’arxiu.")
        return

    header = _build_header("arxiu", query, sess.poble)
    text = header
    for i, r in enumerate(results, start=1):
        title = r.get("title", "Sense títol")
        score = r.get("score", 0.0)
        text += f"{i}. [{title}](https://t.me/misCEREbot/{r.get('id')}) — {score:.2f}\n"
    for part in split_for_telegram(text):
        await send_long_message(update, part)

#---------------------------------------------------------------
#--------CAPÇALERA: /CREUA -------------------------------------
#---------------------------------------------------------------
async def cmd_creua(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Compara dos articles o temes."""
    uid = update.effective_user.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("❗ Format: `/creua temaA temaB` o `/creua id1 id2`\nExemples: `/creua 76 47` o `/creua castell templer`")
        return

    temaA, temaB = args[0], args[1]
    logger.info(f"[TG-HANDLER]/creua uid={uid} A={temaA} B={temaB}")

    # Utilitza la nova funció que detecta automàticament si són IDs o temes
    resA = get_article_by_id_or_query(temaA)
    resB = get_article_by_id_or_query(temaB)
    
    if not resA or not resB:
        await update.message.reply_text("⚠️ No s'han trobat prou resultats per comparar.")
        return

    # Crea una comparació més específica i estructurada
    resposta = create_comparison_response(temaA, temaB, resA, resB)
    for part in split_for_telegram(resposta):
        await send_long_message(update, part)

#---------------------------------------------------------------
#--------CAPÇALERA: /AMPLIA I HUMOR -----------------------------
#---------------------------------------------------------------
async def do_amplia(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Amplia l’últim resum amb més detall o estil alternatiu."""
    uid = update.effective_user.id
    logger.info(f"[TG-HANDLER]/amplia uid={uid}")
    resposta = cmd_amplia(uid)
    for part in split_for_telegram(resposta):
        await send_long_message(update, part)

async def cmd_humor_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Activa mode humorístic."""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    sess.humor_mode = True
    logger.info(f"[TG-HANDLER]/humor_on uid={uid}")
    await update.message.reply_text("😄 Mode humorístic activat!")

async def cmd_humor_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Desactiva mode humorístic."""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    sess.humor_mode = False
    logger.info(f"[TG-HANDLER]/humor_off uid={uid}")
    await update.message.reply_text("🙂 Mode humorístic desactivat.")

#---------------------------------------------------------------
#--------CAPÇALERA: /TOT I /MES --------------------------------
#---------------------------------------------------------------
async def do_tot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra tot l’històric de la sessió."""
    uid = update.effective_user.id
    resposta = cmd_tot(uid)
    for part in split_for_telegram(resposta):
        await send_long_message(update, part)

async def do_mes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra l'últim resum o fragment."""
    uid = update.effective_user.id
    resposta = cmd_mes(uid)
    for part in split_for_telegram(resposta):
        await send_long_message(update, part)

async def do_llegir(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra la primera pàgina de l'article actual."""
    uid = update.effective_user.id
    sess = sessions.get_session(uid)
    
    if sess.mode == "article" and sess.last_article:
        resposta = get_article_page(uid, 1)
    else:
        resposta = "No hi ha cap article obert. Fes clic en un article per obrir-lo."
    
    for part in split_for_telegram(resposta):
        await send_long_message(update, part)

#---------------------------------------------------------------
#--------CAPÇALERA: /RESET -------------------------------------
#---------------------------------------------------------------
async def do_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reinicia completament la sessió de l’usuari."""
    uid = update.effective_user.id
    logger.info(f"[TG-HANDLER]/reset uid={uid}")
    resposta = cmd_reset(uid)
    await send_long_message(update, resposta)

#---------------------------------------------------------------
#6--------CAPÇALERA: INICIALITZACIÓ I ARRENCADA DEL BOT ---------
#---------------------------------------------------------------
# (duplicate import removed)

#---------------------------------------------------------------
#--------CAPÇALERA: REGISTRE DE HANDLERS ------------------------
#---------------------------------------------------------------
def register_handlers(app):
    """Registra tots els handlers del bot."""
    logger.info("[INIT] Registrant handlers de Telegram...")

    # Bàsics i d’ajuda
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ajuda", ajuda))
    app.add_handler(CommandHandler("expert", expert))
    app.add_handler(CommandHandler("help", ajuda))

    # Gestió de sessió i context
    app.add_handler(CommandHandler("nou", cmd_nou))
    app.add_handler(CommandHandler("reset", cmd_reset))

    # Filtres i temes
    app.add_handler(CommandHandler("poble", cmd_poble))
    app.add_handler(CommandHandler("tema", cmd_tema))
    app.add_handler(CommandHandler("id", cmd_id))

    # Arxiu i cerca
    app.add_handler(CommandHandler("arxiu", cmd_arxiu))
    app.add_handler(CommandHandler("arxiu_cerca", cmd_arxiu_cerca))

    # Funcions avançades
    app.add_handler(CommandHandler("creua", cmd_creua))
    app.add_handler(CommandHandler("amplia", cmd_amplia))
    app.add_handler(CommandHandler("humor_on", cmd_humor_on))
    app.add_handler(CommandHandler("humor_off", cmd_humor_off))

    # Selecció directa per ID: /<id>
    app.add_handler(MessageHandler(filters.Regex(r"^/\d+$"), handle_numeric_command))

    # Resums i historial
    app.add_handler(CommandHandler("tot", cmd_tot))
    app.add_handler(CommandHandler("mes", do_mes))
    app.add_handler(CommandHandler("llegir", do_llegir))

    # Entrada genèrica de text (últim recurs)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("[INIT] Handlers registrats correctament.")

#---------------------------------------------------------------
#--------CAPÇALERA: HANDLER DE MISSATGES GENERICS --------------
#---------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa qualsevol text que no sigui comanda."""
    uid = update.effective_user.id
    text = update.message.text.strip()
    logger.info(f"[TG-HANDLER]/text uid={uid} msg='{text[:60]}'")
    
    # Debug: comprova si és una comanda numèrica
    if re.match(r"^/\d+$", text):
        logger.warning(f"[DEBUG] Comanda numèrica {text} processada per handle_message en lloc de handle_numeric_command!")
    
    # Protecció contra processament duplicat
    message_id = update.message.message_id
    if hasattr(context, 'processed_messages'):
        if message_id in context.processed_messages:
            logger.warning(f"[TG-HANDLER] Missatge {message_id} ja processat, ignorant duplicat")
            return
        context.processed_messages.add(message_id)
    else:
        context.processed_messages = {message_id}
    
    # Envia missatge de "rumiant..." mentre processa
    thinking_msg = await update.message.reply_text("🤔 ...rumiant...")
    
    resposta = handle_text(uid, text)
    
    # Esborra el missatge de "rumiant..." i envia la resposta
    try:
        await thinking_msg.delete()
    except Exception as e:
        logger.warning(f"[TG-HANDLER] No s'ha pogut esborrar el missatge de thinking: {e}")
    
    # Envia la resposta final
    parts = split_for_telegram(resposta)
    for part in parts:
        await send_long_message(update, part)

#---------------------------------------------------------------
#--------CAPÇALERA: HEALTHCHECK WEB SERVER ---------------------
#---------------------------------------------------------------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Check system status
            status = {
                "status": "healthy",
                "service": "MisCEREbot",
                "timestamp": time.time(),
                "openai_available": OPENAI.api_key is not None,
                "circuit_breaker_state": OPENAI.circuit_breaker.state if hasattr(OPENAI, 'circuit_breaker') else "unknown",
                "sessions_active": len(sessions.sessions) if hasattr(sessions, 'sessions') else 0,
                "cache_size": len(EMB_CACHE) if 'EMB_CACHE' in globals() else 0
            }
            
            response = json.dumps(status, indent=2)
            self.wfile.write(response.encode('utf-8'))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

def start_health_server():
    """Inicia servidor web simple per healthcheck de Railway."""
    try:
        server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
        logger.info("[HEALTH] Servidor web iniciat al port 8080")
        server.serve_forever()
    except Exception as e:
        logger.error(f"[HEALTH] Error servidor web: {e}")

#---------------------------------------------------------------
#--------CAPÇALERA: MAIN ---------------------------------------
#---------------------------------------------------------------
def main():
    """Arrenca el bot i carrega tots els components."""
    logger.info("[MAIN] Iniciant MisCEREbot...")
    
    # Validació de variables d'entorn
    if not TELEGRAM_TOKEN:
        logger.error("[MAIN] ❌ TELEGRAM_TOKEN no configurat!")
        print("❌ ERROR: TELEGRAM_TOKEN no configurat. Configura la variable d'entorn.")
        return
    
    if not OPENAI_API_KEY:
        logger.warning("[MAIN] ⚠️ OPENAI_API_KEY no configurat - mode fallback")
        print("⚠️ AVÍS: OPENAI_API_KEY no configurat - funcionalitat limitada")

    # Inicialitza aplicació de Telegram
    try:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        register_handlers(app)
        logger.info("[MAIN] ✅ Aplicació Telegram inicialitzada correctament")
    except Exception as e:
        logger.error(f"[MAIN] ❌ Error inicialitzant Telegram: {e}")
        print(f"❌ Error inicialitzant Telegram: {e}")
        return

    # Inicia servidor web per healthcheck en thread separat
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    logger.info("[MAIN] ✅ Servidor healthcheck iniciat")

    # Missatge de log per control intern
    logger.info("🤖 MisCEREbot llest i esperant missatges...")
    print("✅ MisCEREbot en funcionament. Esperant missatges a Telegram...")

    # Error handler per conflictes de Telegram
    async def error_handler(update, context):
        """Gestiona errors de l'aplicació."""
        error = context.error
        logger.error(f"[ERROR] {type(error).__name__}: {error}")
        
        # Altres errors (no conflictes)
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ S'ha produït un error. Torna-ho a provar en uns moments."
            )
    
    # Afegeix l'error handler
    app.add_error_handler(error_handler)
    
    # Executa el polling amb reintent intel·ligent per conflictes
    max_retries = 5
    retry_delay = 30
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[MAIN] Intento de connexió {attempt + 1}/{max_retries}")
            app.run_polling()
            break  # Si arriba aquí, la connexió ha estat exitosa
        except Exception as e:
            error_msg = str(e)
            if "Conflict" in error_msg and "getUpdates" in error_msg:
                if attempt < max_retries - 1:
                    logger.warning(f"[MAIN] Conflicte de Telegram detectat (intento {attempt + 1})")
                    logger.info(f"[MAIN] Esperant {retry_delay} segons abans de reintentar...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Dobla el temps d'espera cada vegada
                else:
                    logger.error(f"[MAIN] Màxim d'intents assolit. Error persistent: {e}")
                    print(f"❌ Error persistent de Telegram després de {max_retries} intents")
                    raise
            else:
                logger.error(f"[MAIN] Error no relacionat amb conflicte: {e}")
                print(f"❌ Error en execució: {e}")
                raise

#---------------------------------------------------------------
#--------CAPÇALERA: EXECUCIÓ DIRECTA ---------------------------
#---------------------------------------------------------------
if __name__ == "__main__":
    main()
