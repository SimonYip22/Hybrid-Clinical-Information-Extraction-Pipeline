"""
validate_symptom_rules.py

Purpose:
    Evaluate symptom extraction performance:
    - Section coverage
    - Extraction yield
    - Concept distribution
    - Negation behaviour

Usage:
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_symptom_rules.py
"""

import pandas as pd
import random
from collections import Counter

from deterministic_extraction.section_extraction import extract_sections
from deterministic_extraction.extraction_rules.symptom_rules import extract_symptoms, TARGET_SYMPTOM_SECTIONS

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CORPUS_PATH = "data/processed/icu_corpus.csv"
TEXT_COLUMN = "TEXT"
SAMPLE_SIZE = 30
RANDOM_SEED = 42

df = pd.read_csv(CORPUS_PATH)
notes = df.to_dict(orient="records")

random.seed(RANDOM_SEED)
sample = random.sample(notes, SAMPLE_SIZE)

# ---------------------------------------------------------------------
# TRACKERS
# ---------------------------------------------------------------------

total_notes = len(sample)

notes_with_any_sections = 0
notes_with_target_sections = 0
notes_with_no_target_sections = 0
notes_with_target_but_no_symptoms = 0

total_symptoms = 0

concept_counter = Counter()
negation_counter = Counter()

# ---------------------------------------------------------------------
# VALIDATION LOOP
# ---------------------------------------------------------------------

for i, note in enumerate(sample):
    text = note[TEXT_COLUMN]

    note_id = note.get("NOTE_ID", f"note_{i}")
    subject_id = note.get("SUBJECT_ID", "")
    hadm_id = note.get("HADM_ID", "")
    icustay_id = note.get("ICUSTAY_ID", "")

    sections = extract_sections(text)

    if sections:
        notes_with_any_sections += 1

    # Filter to target sections
    target_sections = {
        k: v for k, v in sections.items()
        if k.lower() in TARGET_SYMPTOM_SECTIONS
    }

    print("\n" + "-" * 80)
    print(f"NOTE {i+1}")
    print("-" * 80)

    if not target_sections:
        notes_with_no_target_sections += 1
        print("No target sections found.")
        continue

    notes_with_target_sections += 1

    print("\nSECTIONS:")
    for section, content in target_sections.items():
        print(f"\n[{section.upper()}]")
        print(content[:300])

    # ----------------------------------------------------------
    # Extraction
    # ----------------------------------------------------------

    extracted = []

    for section, content in target_sections.items():
        ents = extract_symptoms(
            note_id,
            subject_id,
            hadm_id,
            icustay_id,
            section,
            content
        )
        extracted.extend(ents)

    if not extracted:
        notes_with_target_but_no_symptoms += 1

    # ----------------------------------------------------------
    # Tracking
    # ----------------------------------------------------------

    for e in extracted:
        concept = e["entity_text"].lower()  # TEMP until concept added

        concept_counter.update([concept])
        negation_counter.update([(concept, e["negated"])])
        total_symptoms += 1

    # ----------------------------------------------------------
    # Print
    # ----------------------------------------------------------

    print("\nSYMPTOMS:")
    if extracted:
        for e in extracted:
            print(f'- {e["entity_text"]} | negated={e["negated"]} | section={e["section"]}')
    else:
        print("No symptoms extracted")

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION COVERAGE")
print("=" * 80)

print(f"Total notes: {total_notes}")
print(f"Notes with ANY sections: {notes_with_any_sections}")
print(f"Notes with TARGET sections: {notes_with_target_sections}")
print(f"Notes WITHOUT target sections: {notes_with_no_target_sections}")

print("\n" + "=" * 80)
print("EXTRACTION PERFORMANCE")
print("=" * 80)

if notes_with_target_sections > 0:
    print(f"Notes with NO symptoms (given sections): {notes_with_target_but_no_symptoms} / {notes_with_target_sections} "
          f"({notes_with_target_but_no_symptoms / notes_with_target_sections * 100:.1f}%)")

print(f"Total symptoms extracted: {total_symptoms}")

if notes_with_target_sections > 0:
    print(f"Average per note: {total_symptoms / notes_with_target_sections:.2f}")

print("\n" + "=" * 80)
print("TOP SURFACE FORMS (debug)")
print("=" * 80)

for k, v in concept_counter.most_common(20):
    print(f"{k}: {v}")

print("\n" + "=" * 80)
print("NEGATION BEHAVIOUR")
print("=" * 80)

for (concept, neg), count in negation_counter.items():
    print(f"{concept} | negated={neg}: {count}")