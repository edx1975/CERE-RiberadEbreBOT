import json, os, logging
logger = logging.getLogger("Semantic")

# Carrega el fitxer una sola vegada en memòria
TOPIC_PATH = "data/topics_semantics.json"
if os.path.exists(TOPIC_PATH):
    with open(TOPIC_PATH, "r", encoding="utf-8") as f:
        TOPIC_SEMANTICS = json.load(f)
else:
    TOPIC_SEMANTICS = {}
    logger.warning(f"[TOPICS] No s'ha trobat {TOPIC_PATH}")

def semantic_expand(query: str) -> list[str]:
    """Expansió semàntica amb suport de topics_semantics.json"""
    q_clean = query.strip().lower()

    # 1️⃣ Busca si el tema està al fitxer
    if q_clean in TOPIC_SEMANTICS:
        topic_data = TOPIC_SEMANTICS[q_clean]
        expanded = list(set(topic_data.get("categories", []) + topic_data.get("associacions", 
[])))
        tipus = ", ".join(topic_data.get("categories", []))
        logger.info(f"[SEMANTIC TOPIC] '{q_clean}' trobat al fitxer semàntic ({tipus})")
        logger.info(f"[SEMANTIC EXPAND JSON] {q_clean} → {', '.join(expanded)}")
        return expanded

    # 2️⃣ Si no hi és, torna al sistema automàtic (model o heurística)
    from misCEREbot import get_semantic_related
    expanded = get_semantic_related(q_clean)
    logger.info(f"[SEMANTIC EXPAND AUTO] {q_clean} → {', '.join(expanded)}")

    # Filtre bàsic per paraules massa genèriques
    noisy = {"música", "literatura", "popular", "modèstia"}
    expanded = [w for w in expanded if w not in noisy]

    logger.info(f"[SEMANTIC EXPAND FILTERED] {q_clean} → {', '.join(expanded)}")
    return expanded

