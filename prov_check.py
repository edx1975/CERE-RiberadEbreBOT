import os
import numpy as np
import faiss
import json
from tqdm import tqdm

# ===============================
# 🔑 Configuració
# ===============================
CORPUS_PATH = "../Miscerebot_backup/data/corpus_original.jsonl"
EMB_PATH = "../Miscerebot_backup/data/embeddings_G_pro_large.npy"
INDEX_PATH = "../Miscerebot_backup/data/faiss_index_G_pro_large.index"

# ===============================
# 🧾 Carrega els documents
# ===============================
with open(CORPUS_PATH, "r") as f:
    docs = [json.loads(line) for line in f if line.strip()]
print(f"✅ Carregats {len(docs)} documents de {CORPUS_PATH}")

# ===============================
# 💾 Carrega embeddings existents si hi ha
# ===============================
if os.path.exists(EMB_PATH):
    embeddings = np.load(EMB_PATH)
    print(f"🔁 Checkpoint trobat: {embeddings.shape[0]} vectors amb dimensió {embeddings.shape[1]}")
else:
    raise ValueError("❌ No hi ha embeddings; genera’ls abans.")

# ===============================
# 🔧 Comprova dimensió embeddings
# ===============================
expected_dim = 3072  # nova dimensió correcta pel model text-embedding-3-large
if embeddings.shape[1] != expected_dim:
    raise ValueError(f"❌ Dimensió incorrecta: {embeddings.shape[1]}, esperada {expected_dim}")

# ===============================
# 🧮 Normalitza embeddings
# ===============================
embeddings_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
print("✅ Embeddings normalitzats")

# ===============================
# ⚙️ Crea índex FAISS amb dimensió correcta
# ===============================
index = faiss.IndexFlatIP(expected_dim)
index.add(embeddings_norm)
print(f"✅ Índex FAISS creat amb {index.ntotal} vectors")

# ===============================
# 💾 Desa índex FAISS
# ===============================
faiss.write_index(index, INDEX_PATH)
print(f"✅ FAISS index guardat a {INDEX_PATH}")

