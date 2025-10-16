import os
import numpy as np
from openai import OpenAI
import json
import faiss
from tqdm import tqdm

# ===============================
# 🔑 Configuració
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Falta la variable d'entorn OPENAI_API_KEY")

CORPUS_PATH = "../Miscerebot_backup/data/corpus_original.jsonl"
EMB_PATH = "../Miscerebot_backup/data/embeddings_G_pro_large.npy"
INDEX_PATH = "../Miscerebot_backup/data/faiss_index_G_pro_large.index"
CHECKPOINT_INTERVAL = 10   # cada 10 documents desa un checkpoint

client = OpenAI(api_key=OPENAI_API_KEY)


# ===============================
# 🧾 Carrega els documents
# ===============================
with open(CORPUS_PATH, "r") as f:
    docs = [json.loads(line) for line in f if line.strip()]

print(f"✅ Carregats {len(docs)} documents de {CORPUS_PATH}")


# ===============================
# ⚙️ Funcions d’embeddings
# ===============================
def embed_text(text: str) -> np.ndarray:
    """Genera embedding normalitzat per un text, amb control de mida."""
    if not text:
        return np.zeros(3072, dtype=np.float32)

    emb = client.embeddings.create(model="text-embedding-3-large", input=text)
    v = np.array(emb.data[0].embedding, dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v), 1e-9)
    return v


# ----------------------------------------------------------------------
# Quan crees l’índex FAISS
# ----------------------------------------------------------------------

dim = 3072  # nova dimensió correcta pel model text-embedding-3-large
index = faiss.IndexFlatIP(dim)
print(f"✅ Índex FAISS creat amb {dim} dimensions")


def build_weighted_embedding(doc):
    """Combina títol, temes i resums amb pesos."""
    title_vec = embed_text(doc.get("title", ""))
    topics_vec = embed_text(" ".join(doc.get("topics", [])))
    summary_vec = embed_text(doc.get("summary", ""))
    long_vec = embed_text(doc.get("summary_long", ""))

    combined = (
        0.45 * title_vec +
        0.25 * topics_vec +
        0.20 * summary_vec +
        0.10 * long_vec
    )
    return combined / np.maximum(np.linalg.norm(combined), 1e-9)


# ===============================
# 💾 Comprova si hi ha checkpoint
# ===============================
if os.path.exists(EMB_PATH):
    old_embeddings = np.load(EMB_PATH)
    start_idx = len(old_embeddings)
    print(f"🔁 Reprenent des del document {start_idx} (checkpoint trobat)")
    embeddings = list(old_embeddings)
else:
    embeddings = []
    start_idx = 0


# ===============================
# 🚀 Generació amb barra i checkpoints
# ===============================
for i, doc in enumerate(tqdm(docs[start_idx:], desc="Generant embeddings", unit="doc", initial=start_idx, total=len(docs))):
    try:
        emb = build_weighted_embedding(doc)
        embeddings.append(emb)
    except Exception as e:
        print(f"⚠️ Error al doc {i + start_idx}: {doc.get('title', 'Sense títol')} → {e}")
        embeddings.append(np.zeros(3072, dtype=np.float32))
    

    # ⏸️ Desa checkpoint cada N docs
    if (i + 1) % CHECKPOINT_INTERVAL == 0 or (i + start_idx + 1) == len(docs):
        np.save(EMB_PATH, np.array(embeddings, dtype=np.float32))
        print(f"💾 Checkpoint guardat: {len(embeddings)} embeddings")

print(f"✅ Embeddings finals guardats: {len(embeddings)} vectors → {EMB_PATH}")


# ===============================
# 🧮 Crea índex FAISS normalitzat
# ===============================
embeddings = np.array(embeddings, dtype=np.float32)
embeddings_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)

index = faiss.IndexFlatIP(3072)
index.add(embeddings_norm)
faiss.write_index(index, INDEX_PATH)

print(f"✅ FAISS index creat i guardat a {INDEX_PATH}")
