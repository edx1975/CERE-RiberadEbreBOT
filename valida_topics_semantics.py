#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
valida_topics_semantics_v2.py
Valida i resumeix el fitxer topics_semantics.json generat per IA.
Exporta versions CSV, Markdown i JSON-resum per al bot (/arxiu).
"""

import json
import csv
from collections import Counter
from pathlib import Path

# === CONFIGURACIÓ ===
INPUT_FILE = Path("data/topics_semantics.json")
CSV_FILE = Path("data/report_topics.csv")
MD_FILE = Path("data/report_topics.md")
SUMMARY_FILE = Path("data/report_topics_summary.json")

# === FUNCIONS ===
def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error obrint {path}: {e}")
        return {}

def validate_entry(topic, data):
    errors = []
    required_fields = ["categories", "associacions", "notes"]
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Camp buit o inexistent: {field}")
    # Validació addicional
    if not isinstance(data.get("categories", []), list):
        errors.append("Camp 'categories' no és una llista")
    if len(data.get("notes", "")) < 15:
        errors.append("Camp 'notes' massa curt")
    return errors

# === MAIN ===
def main():
    print(f"📘 Analitzant {INPUT_FILE} ...")
    data = safe_load_json(INPUT_FILE)
    if not data:
        print("⚠️ No s'ha pogut carregar cap dada.")
        return

    total = len(data)
    print(f"✅ {total} temes carregats correctament.\n")

    errors_total = 0
    category_counter = Counter()
    association_counter = Counter()
    pobles_counter = Counter()
    rows_csv = []
    temes_erronis = []

    for topic, entry in data.items():
        errs = validate_entry(topic, entry)
        if errs:
            errors_total += 1
            temes_erronis.append({"topic": topic, "errors": errs})
            print(f"⚠️ {topic}: {', '.join(errs)}")

        cats = ", ".join(entry.get("categories", []))
        assocs = ", ".join(entry.get("associacions", []))
        notes = entry.get("notes", "")
        pobles = ", ".join(entry.get("pobles_relacionats", []))
        freq = entry.get("freq", 0)

        for c in entry.get("categories", []):
            category_counter[c] += 1
        for a in entry.get("associacions", []):
            association_counter[a] += 1
        for p in entry.get("pobles_relacionats", []):
            pobles_counter[p] += 1

        rows_csv.append({
            "topic": topic,
            "freq": freq,
            "categories": cats,
            "associacions": assocs,
            "pobles_relacionats": pobles,
            "notes": notes
        })

    print("\n📊 Resum general:")
    print(f"Temes totals: {total}")
    print(f"Temes amb errors: {errors_total}")

    print("\n🏷️ Categories més freqüents:")
    for cat, count in category_counter.most_common(10):
        print(f"  {cat:20s} {count}")

    print("\n🔗 Conceptes més associats:")
    for a, count in association_counter.most_common(10):
        print(f"  {a:20s} {count}")

    if pobles_counter:
        print("\n🏘️ Pobles més citats:")
        for p, count in pobles_counter.most_common(10):
            print(f"  {p:20s} {count}")

    print(f"\nTotal categories diferents: {len(category_counter)}")
    print(f"Total pobles diferents: {len(pobles_counter)}")

    # --- EXPORT CSV ---
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["topic", "freq", "categories", "associacions", "pobles_relacionats", "notes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_csv)
    print(f"\n💾 Arxiu CSV exportat: {CSV_FILE}")

    # --- EXPORT MARKDOWN ---
    with open(MD_FILE, "w", encoding="utf-8") as f:
        f.write(f"# 📚 Informe de temes del corpus\n\n")
        f.write(f"- **Temes totals:** {total}\n")
        f.write(f"- **Temes amb errors:** {errors_total}\n\n")

        f.write("## 🏷️ Categories més freqüents\n\n")
        f.write("| Categoria | Comptes |\n|------------|----------|\n")
        for cat, count in category_counter.most_common(10):
            f.write(f"| {cat} | {count} |\n")

        f.write("\n## 🔗 Conceptes més associats\n\n")
        f.write("| Associació | Comptes |\n|-------------|----------|\n")
        for a, count in association_counter.most_common(10):
            f.write(f"| {a} | {count} |\n")

        if pobles_counter:
            f.write("\n## 🏘️ Pobles més citats\n\n")
            f.write("| Poble | Comptes |\n|--------|----------|\n")
            for p, count in pobles_counter.most_common(10):
                f.write(f"| {p} | {count} |\n")

        f.write("\n---\nGenerat automàticament per *valida_topics_semantics_v2.py*.\n")

    print(f"🪶 Informe Markdown exportat: {MD_FILE}")

    # --- EXPORT JSON RESUM ---
    summary = {
        "temes_totals": total,
        "temes_errors": errors_total,
        "categories_top10": category_counter.most_common(10),
        "pobles_top10": pobles_counter.most_common(10),
        "temes_erronis": temes_erronis,
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📁 Resum numèric exportat: {SUMMARY_FILE}")

    print("\n✅ Validació i exportació completades correctament.")

if __name__ == "__main__":
    main()
