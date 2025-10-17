#!/usr/bin/env python3
"""
Script per regenerar l'índex FAISS amb embeddings de 3072 dimensions.
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv

# Carrega variables d'entorn
load_dotenv()

# Configuració
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
METADATA_PATH = DATA_DIR / "corpus_original.jsonl"
EMB_PATH = DATA_DIR / "embeddings_G.npy"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index_G.index"

def regenerate_index():
    """Regenera l'índex FAISS amb embeddings de 3072 dimensions."""
    print("🔄 Regenerant índex FAISS amb embeddings de 3072 dimensions...")
    
    # Carrega documents
    print("📚 Carregant documents...")
    docs = []
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line.strip()))
    
    print(f"✅ Carregats {len(docs)} documents")
    
    # Carrega embeddings existents
    print("🧮 Carregant embeddings existents...")
    if EMB_PATH.exists():
        embeddings = np.load(EMB_PATH)
        print(f"✅ Carregats embeddings: {embeddings.shape}")
    else:
        print("❌ No s'han trobat embeddings. Executa primer el script de generació d'embeddings.")
        return
    
    # Verifica dimensions
    if embeddings.shape[1] != 3072:
        print(f"⚠️  Els embeddings existents tenen {embeddings.shape[1]} dimensions, no 3072.")
        print("   Necessites regenerar els embeddings amb text-embedding-3-large.")
        return
    
    # Crea nou índex FAISS
    print("🔍 Creant índex FAISS...")
    vector_index = faiss.IndexFlatIP(embeddings.shape[1])  # 3072 dimensions
    
    # Normalitza embeddings
    emb_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    
    # Afegeix embeddings a l'índex
    vector_index.add(emb_norm)
    
    # Guarda l'índex
    print("💾 Guardant índex FAISS...")
    faiss.write_index(vector_index, str(FAISS_INDEX_PATH))
    
    print(f"✅ Índex FAISS regenerat amb {vector_index.ntotal} vectors de {embeddings.shape[1]} dimensions")
    print(f"📁 Guardat a: {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    regenerate_index()
