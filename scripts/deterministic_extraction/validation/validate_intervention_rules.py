"""
validate_intervention_rules.py

Purpose:
    Evaluate the performance of the rule-based INTERVENTION extraction pipeline
    on a sample of ICU clinical notes. This validation script provides
    both quantitative and qualitative inspection to ensure that the
    extraction behaves as designed.

Usage:
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_intervention_rules.py

Workflow:
    1. Load the ICU corpus 
    2. Randomly sample 30 notes for validation
    3. Extract sections from each note using `extract_sections`
    4. Filter sections to target intervention-relevant sections
    5. For each section:
        a. Split into sentences
        b. Detect interventions using `extract_interventions`
        c. Deduplicate exact repetitions
    6. Track metrics:
        - Notes with any sections
        - Notes with target sections
        - Notes with no target sections
        - Notes with target sections but no symptoms
        - Total symptoms extracted
        - Concept counts
    7. Print outputs per note:
        - Sections (truncated)
        - Extracted symptom entities with concept and section
    8. Summarise:
        - Section coverage
        - Extraction performance (total, average)
        - Top concepts
"""

import pandas as pd
import random
from collections import Counter, defaultdict

from deterministic_extraction.section_extraction import extract_sections
from deterministic_extraction.extraction_rules.intervention_rules import (
    extract_interventions,
    TARGET_INTERVENTION_SECTIONS
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
notes_with_target_but_no_interventions = 0

total_interventions = 0

concept_counter = Counter()

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
        if k.lower() in TARGET_INTERVENTION_SECTIONS
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
        ents = extract_interventions(
            note_id,
            subject_id,
            hadm_id,
            icustay_id,
            section,
            content
        )
        extracted.extend(ents)

    if not extracted:
        notes_with_target_but_no_interventions += 1

    # ----------------------------------------------------------
    # Tracking
    # ----------------------------------------------------------

    for e in extracted:
        concept_counter.update([e["concept"]])
        total_interventions += 1

    concept_per_sentence = defaultdict(list)

    for e in extracted:
        key = (e["sentence_text"], e["concept"])
        concept_per_sentence[key].append(e["entity_text"])

    duplicate_concepts = {
        k: v for k, v in concept_per_sentence.items() if len(v) > 1
    }

    # ----------------------------------------------------------
    # Print extracted entities
    # ----------------------------------------------------------

    print(
        f'- {e["entity_text"]} '
        f'| concept={e["concept"]} '
        f'| section={e["section"]}'
    )

    if extracted:
        print(f"Extracted {len(extracted)} interventions\n")

        for e in extracted:
            print(
                f'- {e["entity_text"]} '
                f'| concept={e["concept"]} '
                f'| section={e["section"]}'
            )
    else:
        print("No interventions extracted")

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
        f"Notes with NO interventions (given sections): "
        f"{notes_with_target_but_no_interventions} / {notes_with_target_sections} "
        f"({notes_with_target_but_no_interventions / notes_with_target_sections * 100:.1f}%)"
    )

print(f"Total interventions extracted: {total_interventions}")

if notes_with_target_sections > 0:
    print(f"Average per note: {total_interventions / notes_with_target_sections:.2f}")

print("\n" + "=" * 80)
print("TOP CONCEPTS")
print("=" * 80)

for concept, count in concept_counter.most_common(20):
    print(f"{concept}: {count}")

if duplicate_concepts:
    print("\nMULTIPLE MATCHES (same concept, same sentence):")
    for (sent, concept), ents in duplicate_concepts.items():
        print(f"[{concept}] → {ents}")