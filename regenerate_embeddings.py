#!/usr/bin/env python3
"""
Script per regenerar els embeddings amb text-embedding-3-large (3072D).
"""

import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Carrega variables d'entorn
load_dotenv()

# Configuració
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
METADATA_PATH = DATA_DIR / "corpus_original.jsonl"
EMB_PATH = DATA_DIR / "embeddings_G.npy"

# Simula l'OpenAI adapter (per no dependre del bot principal)
class MockOpenAI:
    def __init__(self, api_key):
        self.api_key = api_key
        if not api_key:
            print("⚠️  OPENAI_API_KEY no configurat. Usant embeddings simulats.")
            return
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print("✅ Client OpenAI carregat")
        except Exception as e:
            print(f"❌ Error carregant OpenAI: {e}")
            self.client = None
    
    def embed(self, model, text):
        if not self.client:
            # Genera embedding simulats determinístics
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(3072).astype(np.float32)
        
        resp = self.client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

def regenerate_embeddings():
    """Regenera els embeddings amb text-embedding-3-large."""
    print("🔄 Regenerant embeddings amb text-embedding-3-large (3072D)...")
    
    # Inicialitza OpenAI
    openai = MockOpenAI(os.getenv("OPENAI_API_KEY"))
    
    # Carrega documents
    print("📚 Carregant documents...")
    docs = []
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line.strip()))
    
    print(f"✅ Carregats {len(docs)} documents")
    
    # Genera embeddings
    print("🧮 Generant embeddings...")
    embeddings = []
    
    for i, doc in enumerate(docs):
        if i % 10 == 0:
            print(f"   Processant document {i+1}/{len(docs)}...")
        
        # Crea text per embedding
        text_parts = []
        if doc.get('title'):
            text_parts.append(doc['title'])
        if doc.get('summary'):
            text_parts.append(doc['summary'])
        if doc.get('topics'):
            text_parts.extend(doc['topics'])
        
        text = ' '.join(text_parts)
        
        # Genera embedding
        try:
            emb = openai.embed(model="text-embedding-3-large", text=text)
            embeddings.append(emb)
        except Exception as e:
            print(f"⚠️  Error generant embedding per doc {i}: {e}")
            # Usa embedding zero com a fallback
            embeddings.append(np.zeros(3072, dtype=np.float32))
    
    # Converteix a array numpy
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"✅ Generats {embeddings.shape[0]} embeddings de {embeddings.shape[1]} dimensions")
    
    # Guarda embeddings
    print("💾 Guardant embeddings...")
    np.save(EMB_PATH, embeddings)
    print(f"📁 Guardat a: {EMB_PATH}")

if __name__ == "__main__":
    regenerate_embeddings()
