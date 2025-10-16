#python faiss_utils.py --check
# ó
#python faiss_utils.py --search "història de Ginestar"



import os
import json
import faiss
import numpy as np
import logging

logger = logging.getLogger("FaissUtils")

# ======================================================
# 📁 CONSTANTS DE PATHS PER DEFECTE
# ======================================================
DEFAULT_INDEX_PATH = "data/faiss_index_G_pro_large.index"
DEFAULT_METADATA_PATH = "data/corpus_original.jsonl"
DEFAULT_EMB_PATH = "data/embeddings_G_pro_large.npy"


class FaissIndex:
    def __init__(self, index: faiss.Index, documents=None):
        self.index = index
        self.documents = documents or []
        self.embeddings = None

    # ======================================================
    # 🧩 CREACIÓ / GUARDA / CÀRREGA
    # ======================================================
    @classmethod
    def load(cls,
             path: str = DEFAULT_INDEX_PATH,
             metadata_path: str = DEFAULT_METADATA_PATH,
             emb_path: str = DEFAULT_EMB_PATH,
             use_gpu: bool = False):
        """
        Carrega índex FAISS, embeddings i documents.
        Compatibilitat amb el format generat per embed_faiss.py.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ No s'ha trobat l'índex FAISS: {path}")

        # 🧠 Llegeix índex FAISS
        index = faiss.read_index(path)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info(f"[FAISS_UTILS] Índex carregat a GPU.")
            except Exception as e:
                logger.warning(f"[FAISS_UTILS] ⚠️ No s'ha pogut carregar a GPU: {e}")

        logger.info(f"[FAISS_UTILS] Índex FAISS carregat: {index.ntotal} vectors ({index.d}D)")

        # 🧩 Carrega embeddings
        embeddings = None
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path).astype(np.float32)
            logger.info(f"[FAISS_UTILS] Embeddings carregats: {embeddings.shape}")
        else:
            logger.warning(f"[FAISS_UTILS] ⚠️ No s'ha trobat {emb_path}")

        # 🗂️ Carrega metadades del corpus
        documents = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        documents.append(d)
                    except json.JSONDecodeError as e:
                        logger.warning(f"[FAISS_UTILS] ⚠️ JSON invàlid: {e}")
            logger.info(f"[FAISS_UTILS] Carregats {len(documents)} documents.")
        else:
            logger.warning(f"[FAISS_UTILS] ⚠️ No s'ha trobat {metadata_path}")

        # 🆔 Assigna IDs seqüencials als documents si no en tenen
        for i, d in enumerate(documents):
            if "id" not in d or d["id"] is None:
                d["id"] = i
        
        # Crea objecte FAISSIndex
        obj = cls(index=index, documents=documents)
        obj.embeddings = embeddings
        return obj

    def save(self, path: str = "faiss_G_pro.index"):
        """ Desa l’índex FAISS """
        faiss.write_index(self.index, path)
        logger.info(f"[FAISS_UTILS] Índex FAISS guardat a {path}")

    # ======================================================
    # ℹ️ INFO
    # ======================================================
    def count(self) -> int:
        """Retorna el nombre total de vectors a l’índex."""
        return self.index.ntotal

    # ======================================================
    # 🧮 COMPROVACIÓ DE COHERÈNCIA
    # ======================================================
    def check_consistency(self) -> dict:
        """
        Comprova si l'índex, els embeddings i els documents estan alineats.
        Retorna un diccionari amb les dades de coherència.
        """
        n_index = self.index.ntotal if self.index else 0
        n_emb = None if self.embeddings is None else self.embeddings.shape[0]
        n_docs = len(self.documents) if self.documents else 0

        logger.info("📊 Comprovant coherència FAISS...")
        logger.info(f"- Vectors a l'índex: {n_index}")
        logger.info(f"- Embeddings: {n_emb}")
        logger.info(f"- Documents: {n_docs}")

        if self.embeddings is None:
            logger.warning("⚠️  No hi ha embeddings carregats.")
            return {"index": n_index, "embeddings": n_emb, "docs": n_docs, "ok": False}

        ok = n_index == n_emb == n_docs
        if ok:
            logger.info("✅ Tot coherent: índex, embeddings i documents alineats.")
        else:
            logger.warning("⚠️ Inconsistència detectada!")
            logger.warning(f"   → index.ntotal = {n_index}")
            logger.warning(f"   → embeddings = {n_emb}")
            logger.warning(f"   → docs = {n_docs}")

        return {"index": n_index, "embeddings": n_emb, "docs": n_docs, "ok": ok}


    # ======================================================
    # 🔍 CERCA DIRECTA
    # ======================================================
    def search_text(self, query_text: str, top_k: int = 10) -> list[tuple["Doc", float]]:
        """
        Cerca semàntica: retorna [(Doc, score)]
        Utilitza el mateix embed_query que misCEREbot.
        """
        if self.embeddings is None or not self.documents:
            logger.warning("⚠️ No hi ha embeddings o documents carregats.")
            return []

        # Import intern per evitar dependències circulars
        from misCEREbot import embed_query, Doc

        qvec = embed_query(query_text).reshape(1, -1)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec /= norm

        scores, idxs = self.index.search(qvec.astype(np.float32), top_k)

        results = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0 or i >= len(self.documents):
                continue
            doc_data = self.documents[int(i)]

            # Si ja és un objecte Doc, no el tornis a crear
            if isinstance(doc_data, Doc):
                doc = doc_data
            else:
                doc_id = getattr(doc_data, "id", None) or doc_data.get("id") or int(i)
                doc = Doc(doc_data, doc_id=doc_id)

            results.append((doc, float(s)))


        return results


# ======================================================
# 🚀 EXECUCIÓ DES DE TERMINAL
# ======================================================
if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Utilitats FAISS — càrrega, comprovació i cerques."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Comprova coherència entre índex, embeddings i documents.",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Fes una cerca semàntica pel text indicat.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Nombre de resultats a retornar en una cerca.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Utilitza GPU si està disponible per carregar l'índex.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s"
)


    # 🔹 Carrega l'índex FAISS
    fi = FaissIndex.load(use_gpu=args.gpu)

    # 🧮 Comprovació de coherència
    if args.check:
        info = fi.check_consistency()
        print("\n📊 RESULTAT DE COHERÈNCIA:")
        print(info)

    # 🔍 Cerca semàntica
    if args.search:
        print(f"\n🔍 Cercant: {args.search}\n")
        results = fi.search_text(args.search, top_k=args.topk)
        if not results:
            print("⚠️  Cap resultat trobat.")
        else:
            for i, (doc, score) in enumerate(results, start=1):
                title = getattr(doc, "title", "Sense títol")
                doc_id = getattr(doc, "id", None)
                print(f"[{title}](https://t.me/misCEREbot/{doc_id}) (Pes: {score:.2f})")
