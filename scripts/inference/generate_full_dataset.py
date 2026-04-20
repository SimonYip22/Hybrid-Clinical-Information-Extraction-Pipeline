"""
generate_full_dataset.py

Purpose:
    Generate a large-scale structured dataset by applying the full pipeline
    (extraction + validation) to the ICU corpus.

Bash command:
    PYTHONPATH=src python scripts/inference/generate_full_dataset.py

Workflow:
    1. Load pretrained transformer model and tokenizer
    2. Stream ICU dataset in chunks
    3. Assign globally unique note_id per row
    4. Apply pipeline once per chunk
    5. Write entity-level outputs to JSONL

Outputs:
    outputs/datasets/icu_entities_full.jsonl (JSONL)
        - JSONL file containing one entity per line:
            - Structured entity fields
            - Validation results (confidence, is_valid)

Notes:
    - Uses chunked processing for scalability and memory safety
    - note_id is generated globally to avoid duplication across chunks
    - No filtering applied (full dataset retained)
"""

import pandas as pd
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from pipeline.pipeline import run_pipeline

# -------------------------
# CONFIG
# -------------------------

DATA_PATH = "data/processed/icu_corpus.csv"
MODEL_DIR = "models/bioclinicalbert_final"

OUTPUT_PATH = Path("outputs/datasets/icu_entities_full.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 3000
THRESHOLD = 0.549
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODEL
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.to(DEVICE)
model.eval()

# -------------------------
# PROCESS DATA IN CHUNKS
# -------------------------

global_idx = 0

with open(OUTPUT_PATH, "w") as f_out:

    for chunk in tqdm(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            usecols=["TEXT", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"] # Only load necessary columns for efficiency
        )
    ):
        
        # Create a copy of the chunk as a DataFrame so we can safely modify
        chunk = chunk.copy()

        # -------------------------
        # Assign global note_id
        # -------------------------

        note_ids = []     

        # Generate globally unique note_id for each row in the chunk
        for _ in range(len(chunk)):
            global_idx += 1
            note_ids.append(f"note_{global_idx}")

        chunk["note_id"] = note_ids

        # -------------------------
        # Run pipeline ONCE per chunk
        # -------------------------
        entities = run_pipeline(
            df=chunk,
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
            threshold=THRESHOLD,
            batch_size=BATCH_SIZE
            )

        # -------------------------
        # Write output
        # -------------------------
        for entity in entities:
            # Convert NumPy scalar floats to standard Python float for JSON-compatibility
            f_out.write(json.dumps(entity, default=float) + "\n")

print(f"\nSaved full dataset to: {OUTPUT_PATH}")