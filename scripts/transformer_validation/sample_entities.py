"""
sample_entities.py

Purpose:
    - Generate a balanced, annotation-ready dataset of extracted clinical entities for Phase 3 transformer validation.
    - Convert raw JSONL extraction outputs into a structured tabular format suitable for manual labeling.
    - Ensure equal representation of each entity type to support unbiased model training and evaluation.

Workflow:
    1. Load extraction candidates from JSONL file (1 line = 1 entity).
    2. Flatten nested JSON structure and extract only relevant fields:
    - note_id, entity_text, entity_type, sentence_text, negated, task
    3. Convert records into a pandas DataFrame for processing.
    4. Perform stratified sampling:
    - Sample N_PER_CLASS (200) entities per entity type:
        SYMPTOM, INTERVENTION, CLINICAL_CONDITION
    5. Concatenate sampled subsets into a single dataset.
    6. Shuffle dataset to mix entity types and reduce annotation bias.
    7. Add empty `is_valid` column for manual ground truth annotation.

Outputs:
    - annotation_sample_raw.csv:
        - Freshly generated sample for annotation (overwritten each run)
        - Used as the source of truth for sampling

    - annotation_sample_labeled.csv:
        - Copy of the sample intended for manual annotation
        - Preserved across runs to prevent loss of annotated labels

Notes:
    - The `task` field is extracted from the nested `validation` object using a safe fallback:
    row.get("validation", {}).get("task")
    to prevent errors if the field is missing.
    - The `negated` field is retained for later comparison with rule-based outputs but is NOT used for training labels.
    - Manual annotation should populate `is_valid` with TRUE/FALSE (binary classification target).
"""

import json
import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
INPUT_PATH = Path("data/interim/extraction_candidates.jsonl")
SAMPLE_OUTPUT_PATH = Path("data/extraction/sampling/annotation_sample_raw.csv")
ANNOTATED_OUTPUT_PATH = Path("data/extraction/sampling/annotation_sample_labeled.csv")

# Ensure output directory exists
SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ANNOTATED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load JSONL
# -------------------------

# List to hold all records
records = []

# Read JSONL file line by line (1 line = 1 entity)
with open(INPUT_PATH, "r") as f:
    for line in f:

        # Converts JSON string to Python dict
        row = json.loads(line)

        # Extract relevant fields and append to records list
        records.append({
            "note_id": row.get("note_id"),
            "section": row.get("section"),
            "concept": row.get("concept"),
            "entity_text": row.get("entity_text"),
            "entity_type": row.get("entity_type"),
            "sentence_text": row.get("sentence_text"),
            "negated": row.get("negated"),
            "task": row.get("validation", {}).get("task"), # If validation exists, extract task, else None
            "confidence": row.get("validation", {}).get("confidence", 0.0) # Optional: include confidence score if available, default to 0.0
        })

# Convert list of records to DataFrame
df = pd.DataFrame(records)

print(f"Loaded {len(df)} total entities")

# -------------------------
# Stratified sampling
# -------------------------
N_PER_CLASS = 200

# Sampled DataFrames for each class
sampled_dfs = []

# Sample for each entity type
for entity_type in ["SYMPTOM", "INTERVENTION", "CLINICAL_CONDITION"]:

    # Filter for current entity type
    subset = df[df["entity_type"] == entity_type]

    # Check if we have enough samples
    if len(subset) < N_PER_CLASS:
        raise ValueError(f"Not enough samples for {entity_type}")

    # Sample N_PER_CLASS from the subset
    sampled = subset.sample(n=N_PER_CLASS, random_state=42)
    # Append to list
    sampled_dfs.append(sampled)

# Combine sampled DataFrames and reset index
sample_df = pd.concat(sampled_dfs).reset_index(drop=True)

# Shuffle final dataset to mix entity types (optional but often desirable for annotation)
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------
# Add annotation column
# -------------------------

# Add empty column for manual annotation labels (ground truth)
sample_df["is_valid"] = ""

print(f"Final sample size: {len(sample_df)}")

# -------------------------
# Save
# -------------------------

# Save raw sample (always overwrite)
sample_df.to_csv(SAMPLE_OUTPUT_PATH, index=False)

print(f"Created raw sample file for annotation: {SAMPLE_OUTPUT_PATH}")

# Only create annotated file if it doesn't already exist
if not ANNOTATED_OUTPUT_PATH.exists():
    sample_df.to_csv(ANNOTATED_OUTPUT_PATH, index=False)
    print(f"Created annotated file: {ANNOTATED_OUTPUT_PATH}")
else:
    print(f"Annotated file already exists, not overwriting.")
