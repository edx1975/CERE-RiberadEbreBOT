import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ===========================
# CONFIGURACIÓ
# ===========================
TEMPERATURE = 0.4
MAX_TOKENS = 400

# --- Stopwords en català ---
catalan_stopwords = [
    "a", "abans", "acabar", "això", "al", "algun", "alguna", "algunes", "alguns",
    "allà", "allí", "allò", "als", "altra", "altre", "altres", "amb", "aprop",
    "aquí", "baix", "bé", "cada", "cap", "casa", "com", "contra",
    "d", "da", "dalt", "davant", "de", "del", "dels", "des", "després",
    "dins", "doncs", "durant", "el", "els", "ella", "elles", "ells",
    "em", "encara", "en", "ens", "entre", "era", "eren", "és", "esta",
    "està", "estaven", "etc", "ets", "fins", "fora", "gairebé", "hi",
    "igual", "jo", "la", "les", "li", "lo", "los", "m", "ma", "mai",
    "massa", "mateix", "me", "mentre", "més", "molt", "molta", "moltes",
    "moltíssim", "moltíssimes", "moltíssims", "molts", "nosaltres", "o",
    "on", "pel", "pels", "per", "perquè", "però", "poc", "poca", "poques",
    "pocs", "podem", "pot", "quan", "quant", "que", "qui", "quin",
    "quina", "quines", "quins", "s", "se", "segons", "sense", "ser",
    "ses", "si", "sobre", "sol", "solament", "sols", "som", "sou", "són",
    "també", "tan", "tant", "tanta", "tantes", "tants", "te", "tenim",
    "tenir", "tot", "tota", "totes", "tots", "un", "una", "unes", "uns",
    "va", "vam", "van", "vosaltres"
]

# ===========================
# MISSATGE DEL SISTEMA
# ===========================
SYSTEM_PROMPT = (
    "Ets un bot simpàtic i xerraire sobre la Ribera d'Ebre, especialment Ginestar, Benissanet, Tivissa, Rasquera i Miravet. "
    "T’agrada explicar històries i curiositats amb un to proper, humà i una mica d’humor. "
    "Sempre dónes prioritat a la informació que trobis als fragments del corpus del CERE. "
    "Si la pregunta no té resposta directa al corpus, pots donar context històric general o anècdotes conegudes, "
    "però has de deixar clar quan no hi ha dades concretes del corpus. "
    "Acaba sovint amb un comentari amistós o divertit."
)

# ===========================
# FUNCIONS
# ===========================
def load_corpus(filename="corpus.json"):
    """Carrega el corpus des d’un fitxer JSON."""
    with open(filename, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return corpus

def build_user_prompt(user_question, top_docs):
    """Crea el prompt per a l'usuari, amb els fragments rellevants."""
    instruction = (
        "INSTRUCCIONS:\n"
        "- Dona prioritat a la informació trobada als fragments F1..Fn.\n"
        "- Si no hi ha dades suficients al corpus, digues-ho clarament però pots afegir context històric o anècdotes generals.\n"
        "- Mantén un to simpàtic i proper, i si escau, una mica d’humor.\n"
        "- Quan utilitzis dades concretes, cita els fragments amb el format F{i} (Títol).\n\n"
    )

    docs_text = "\n\n".join(
        [f"F{i+1} ({doc['title']}): {doc['content']}" for i, doc in enumerate(top_docs)]
    )

    return f"{instruction}PREGUNTA:\n{user_question}\n\nFRAGMENTS DISPONIBLES:\n{docs_text}"

def retrieve_documents(question, vectorizer, tfidf_matrix, corpus_metadata, top_k=5):
    """Torna els top_k documents més semblants a la pregunta."""
    query_vec = vectorizer.transform([question])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    return [corpus_metadata[i] for i in top_indices]

def answer_question(client, user_question, top_docs):
    """Genera la resposta utilitzant el model GPT."""
    user_prompt = build_user_prompt(user_question, top_docs)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message["content"]

# ===========================
# PROGRAMA PRINCIPAL
# ===========================
if __name__ == "__main__":
    # --- Carregar corpus ---
    corpus_metadata = load_corpus("corpus.json")
    corpus_texts = [doc["content"] for doc in corpus_metadata]

    # --- Vectorització ---
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=catalan_stopwords)
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)

    # --- Client OpenAI ---
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("🤖 Bot de la Ribera d'Ebre llest! Escriu una pregunta (o 'surt' per acabar)\n")

    # --- Bucle de conversa ---
    while True:
        user_question = input("Tu: ").strip()
        if user_question.lower() in ["surt", "exit", "quit"]:
            print("Bot: Fins aviat! Ha estat un plaer xerrar 😊")
            break

        top_docs = retrieve_documents(user_question, vectorizer, tfidf_matrix, corpus_metadata)
        answer = answer_question(client, user_question, top_docs)

        print(f"\nBot: {answer}\n")
