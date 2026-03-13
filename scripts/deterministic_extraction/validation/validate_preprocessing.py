"""
validate_preprocessing.py

Purpose:
    Perform manual validation of the report preprocessing pipeline
    applied to ICU clinical notes. This ensures that de-identification,
    whitespace normalization, and EMR artefact removal behave as expected 
    without breaking semantic content.

Usage (terminal):
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_preprocessing.py

Workflow:
    1. Load the processed ICU corpus from a CSV file.
    2. Randomly sample a subset of notes for validation.
    3. Apply the preprocessing pipeline (`preprocess_note`) to each sampled note.
    4. Print original and cleaned versions for manual inspection.
    5. Wait for user input before proceeding to the next note.
"""

import pandas as pd
import random

from deterministic_extraction.preprocessing import preprocess_note

# Configuration
CORPUS_PATH = "data/processed/icu_corpus.csv"
TEXT_COLUMN = "TEXT"
SAMPLE_SIZE = 10

# Load corpus and extract notes
df = pd.read_csv(CORPUS_PATH)

notes = df[TEXT_COLUMN].dropna().tolist()

# Random sample notes for validation
sample = random.sample(notes, SAMPLE_SIZE)

# Apply preprocessing on sample and compare original vs cleaned outputs
for i, note in enumerate(sample):

    cleaned = preprocess_note(note)

    print("\n" + "=" * 80)
    print(f"NOTE {i+1} — ORIGINAL")
    print("=" * 80)
    print(note)

    print("\n" + "-" * 80)
    print(f"NOTE {i+1} — CLEANED")
    print("-" * 80)
    print(cleaned)

    input("\nPress Enter for next note...")