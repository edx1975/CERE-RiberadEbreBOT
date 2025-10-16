#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
genera_topics_semantics_IA_v4_turbo.py

Versió optimitzada i paral·lela (4 fils)
- Usa ThreadPoolExecutor per processar temes en paral·lel
- Manté tolerància total a errors i respostes buides
- Guarda parcials contínuament
- Ideal per corpus grans
"""

import os, re, json, time, random
from collections import defaultdict, Counter
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ======================
# CONFIGURACIÓ
# ======================
INPUT_FILE = Path("data/corpus_original.jsonl")
OUTPUT_FILE = Path("data/topics_semantics.json")
RAW_ERRORS_DIR = Path("data/_ia_raw_errors")

MAX_TEXT_PER_TOPIC = 1800
MAX_RETRIES = 3
MAX_WORKERS = 4  # ← pots pujar-ho a 6 si tens una connexió molt estable

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======================
# FUNCIONS AUXILIARS
# ======================
def normalize_topic(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

def build_context_block(doc: dict) -> str:
    title = doc.get("title", "")
    years = str(doc.get("years", "") or "")
    topics = ", ".join(doc.get("topics", []) or [])
    summary = doc.get("summary", "") or ""
    long = doc.get("summary_long", "") or ""
    text = f"TÍTOL: {title}\nANY(S): {years}\nTOPICS: [{topics}]\nTEXT: {summary} {long}"
    return re.sub(r"\s+", " ", text).strip()

def force_json_object(s: str | None) -> dict | None:
    if not s or not isinstance(s, str) or not s.strip():
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None

def call_model_json(topic: str, fragments: list[str]) -> dict | None:
    joined = " ".join(fragments)[:MAX_TEXT_PER_TOPIC]
    system = (
        "Ets un analista cultural i històric expert en la Ribera d’Ebre. "
        "Respon EXCLUSIVAMENT amb JSON vàlid (objecte), sense cap text extra."
    )
    user = f"""
Analitza aquests fragments del corpus sobre el tema '{topic}' i retorna un JSON amb EXACTAMENT aquest esquema:

{{
  "topic": "{topic}",
  "categories": ["historia","societat","genere","patrimoni","economia","religio","memoria","paisatge", ...],
  "associacions": ["paraules o expressions relacionades"],
  "pobles_relacionats": ["noms propis de poblacions si n'hi ha"],
  "notes": "1-2 frases interpretatives, clares i breus"
}}

Fragments:
{joined}
""".strip()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            content = getattr(resp.choices[0].message, "content", None)
            if not content or not isinstance(content, str) or not content.strip():
                print(f"⚠️ [{topic}] Resposta buida (intent {attempt})")
                time.sleep(1.5 * attempt)
                continue
            parsed = force_json_object(content)
            if parsed is None:
                print(f"⚠️ [{topic}] JSON invàlid (intent {attempt}) → {repr(content[:80])}...")
                time.sleep(1.5 * attempt)
                continue
            return parsed
        except Exception as e:
            print(f"⚠️ [{topic}] Error cridant API (intent {attempt}): {e}")
            time.sleep(1.5 * attempt)
    return None

# ======================
# MAIN
# ======================
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No trobo {INPUT_FILE}")
    RAW_ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    topic_to_text = defaultdict(list)
    topic_freq = Counter()

    print(f"📖 Llegint {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            block = build_context_block(data)
            for t in data.get("topics", []) or []:
                nt = normalize_topic(t)
                topic_to_text[nt].append(block)
                topic_freq[nt] += 1

    topics_sorted = sorted(topic_freq.items(), key=lambda x: -x[1])
    print(f"✅ {len(topics_sorted)} temes únics trobats.")

    results = {}
    processed = 0

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except Exception:
                results = {}
        print(f"🔄 Reprenent des de {len(results)} temes ja processats...")

    def worker(topic):
        if topic in results:
            return (topic, results[topic])
        res = call_model_json(topic, topic_to_text[topic])
        if res is None:
            raw_path = RAW_ERRORS_DIR / f"{topic.replace(' ', '_')}_raw.txt"
            with open(raw_path, "w", encoding="utf-8") as rf:
                rf.write("\n\n--- CONTEXT ---\n\n".join(topic_to_text[topic])[:4000])
            return (topic, None)
        res["topic"] = res.get("topic", topic)
        res["freq"] = int(topic_freq[topic])
        return (topic, res)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, t): t for t, _ in topics_sorted}
        for i, fut in enumerate(as_completed(futures), 1):
            topic = futures[fut]
            try:
                topic, res = fut.result()
                if res:
                    results[topic] = res
                else:
                    print(f"⚠️ Error persistent amb '{topic}'")
            except Exception as e:
                print(f"💥 Error global amb '{topic}': {e}")

            processed += 1
            if processed % 10 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"💾 Desats {len(results)} temes (de {processed})...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Arxiu final generat: {OUTPUT_FILE} ({len(results)} temes bons)")

if __name__ == "__main__":
    main()
