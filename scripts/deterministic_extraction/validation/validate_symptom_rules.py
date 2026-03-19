"""
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_symptom_rules.py
"""
# ---------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# ---------------------------------------------------------------------

import pandas as pd
import random
from collections import Counter

from deterministic_extraction.section_detection import extract_sections
from deterministic_extraction.extraction_rules.symptom_rules import extract_symptoms

# Configuration
CORPUS_PATH = "data/processed/icu_corpus.csv"
TEXT_COLUMN = "TEXT"
SAMPLE_SIZE = 30
RANDOM_SEED = 42

# Sections to prioritise (optional but recommended)
TARGET_SECTIONS = [
    "chief complaint",
    "hpi",
    "assessment",
    "physical examination",
    "review of systems"
]

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

df = pd.read_csv(CORPUS_PATH)
notes = df[TEXT_COLUMN].dropna().tolist()

random.seed(RANDOM_SEED)
sample = random.sample(notes, SAMPLE_SIZE)

# Track frequency (useful later)
symptom_counter = Counter()

# ---------------------------------------------------------------------
# VALIDATION LOOP
# ---------------------------------------------------------------------

for i, note in enumerate(sample):

    sections = extract_sections(note)

    # -----------------------------------------------------------------
    # PRINT ORIGINAL NOTE
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"NOTE {i+1} — ORIGINAL")
    print("=" * 80)
    print(note)

    # -----------------------------------------------------------------
    # PRINT SECTIONS
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"NOTE {i+1} — EXTRACTED SECTIONS")
    print("-" * 80)

    for section, content in sections.items():
        print(f"\n[{section.upper()}]")
        print(content[:500])  # truncate long sections for readability

    # -----------------------------------------------------------------
    # SYMPTOM EXTRACTION (FOCUSED SECTIONS)
    # -----------------------------------------------------------------
    extracted_symptoms = []

    for section, content in sections.items():
        if section.lower() in TARGET_SECTIONS:
            symptoms = extract_symptoms(content)
            extracted_symptoms.extend(symptoms)

    # Deduplicate
    extracted_symptoms = list(set(extracted_symptoms))

    # Update frequency counter
    symptom_counter.update(extracted_symptoms)

    # -----------------------------------------------------------------
    # PRINT SYMPTOMS
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"NOTE {i+1} — EXTRACTED SYMPTOMS")
    print("-" * 80)

    if extracted_symptoms:
        for s in extracted_symptoms:
            print(f"- {s}")
    else:
        print("No symptoms extracted")

    # Pause between notes
    input("\nPress Enter for next note...")

# ---------------------------------------------------------------------
# SUMMARY (VERY IMPORTANT)
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY — TOP EXTRACTED SYMPTOMS")
print("=" * 80)

for symptom, count in symptom_counter.most_common(20):
    print(f"{symptom}: {count}")