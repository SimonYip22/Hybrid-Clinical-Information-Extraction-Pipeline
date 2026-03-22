"""
validate_symptom_rules.py

Purpose:
    Evaluate the performance of the rule-based SYMPTOM extraction pipeline
    on a sample of ICU clinical notes. This validation script provides
    both quantitative and qualitative inspection to ensure that the
    extraction behaves as designed.

Usage:
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_symptom_rules.py

Workflow:
    1. Load the ICU corpus 
    2. Randomly sample 30 notes for validation
    3. Extract sections from each note using `extract_sections`
    4. Filter sections to target symptom-relevant sections
    5. For each section:
        a. Split into sentences
        b. Detect symptoms using `extract_symptoms`
        c. Apply simple negation logic
        d. Deduplicate concepts per sentence
    6. Track metrics:
        - Notes with any sections
        - Notes with target sections
        - Notes with no target sections
        - Notes with target sections but no symptoms
        - Total symptoms extracted
        - Concept counts
        - Negation counts
    7. Print outputs per note:
        - Sections (truncated)
        - Extracted symptom entities with negation, concept, and section
    8. Summarise:
        - Section coverage
        - Extraction performance (total, average)
        - Top concepts
        - Negation behaviour
"""

import pandas as pd
import random
from collections import Counter

from deterministic_extraction.section_extraction import extract_sections
from deterministic_extraction.extraction_rules.symptom_rules import (
    extract_symptoms,
    TARGET_SYMPTOM_SECTIONS
)

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

    

    # ----------------------------------------------------------
    # Show sections (truncated)
    # ----------------------------------------------------------

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
        concept = e["concept"]

        concept_counter.update([concept])
        negation_counter.update([e["negated"]])
        total_symptoms += 1

    # ----------------------------------------------------------
    # Print extracted entities
    # ----------------------------------------------------------

    print("\nSYMPTOMS:")

    if extracted:
        print(f"Extracted {len(extracted)} symptoms\n")

        for e in extracted:
            print(
                f'- {e["entity_text"]} '
                f'| concept={e["concept"]} '
                f'| negated={e["negated"]} '
                f'| section={e["section"]}'
            )
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
    print(
        f"Notes with NO symptoms (given sections): "
        f"{notes_with_target_but_no_symptoms} / {notes_with_target_sections} "
        f"({notes_with_target_but_no_symptoms / notes_with_target_sections * 100:.1f}%)"
    )

print(f"Total symptoms extracted: {total_symptoms}")

if notes_with_target_sections > 0:
    print(f"Average per note: {total_symptoms / notes_with_target_sections:.2f}")

print("\n" + "=" * 80)
print("TOP CONCEPTS")
print("=" * 80)

for concept, count in concept_counter.most_common(20):
    print(f"{concept}: {count}")

print("\n" + "=" * 80)
print("NEGATION BEHAVIOUR")
print("=" * 80)

total_neg = negation_counter[True]
total_pos = negation_counter[False]
total = total_neg + total_pos

if total > 0:
    print(f"Negated: {total_neg} ({total_neg / total * 100:.1f}%)")
    print(f"Not negated: {total_pos} ({total_pos / total * 100:.1f}%)")
else:
    print("No symptoms extracted → no negation statistics available")