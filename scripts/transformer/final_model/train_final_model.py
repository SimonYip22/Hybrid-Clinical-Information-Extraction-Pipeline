"""
train_final_model.py

Purpose:
    - Train the final transformer validation model on the full training dataset.
    - Use the selected hyperparameters determined during tuning.
    - Produce a deployment-ready model for downstream evaluation and inference.

Workflow:
    1. Set reproducibility seeds.
    2. Load full training dataset (n = 1020).
    3. Convert dataset to Hugging Face Dataset format.
    4. Tokenise structured inputs.
    5. Configure training arguments using tuned hyperparameters.
    6. Train model on full dataset (no validation split, no CV).
    7. Save:
        - Trained model weights
        - Model configuration (architecture + label mapping)
        - Tokenizer files (vocab, merges, special tokens)
        - Training arguments (for reproducibility)
    8. Print training summary.

Outputs:
    models/bioclinicalbert_final/
    - model.safetensors → trained model weights (using safetensors format for efficiency and security)
	- config.json → model architecture + label mapping
    - special_tokens_map.json → tokenizer special tokens (e.g., [CLS], [SEP], [PAD])
    - tokenizer_config.json → tokenizer configuration
    - tokenizer.json → tokenizer vocabulary and merges (for BPE-based tokenizers)
    - training_args.bin → training arguments (used for reproducibility)
    - vocab.txt → tokenizer vocabulary file

Notes:
    - This script performs no evaluation.
    - No thresholding is applied here.
    - This model is fixed and used in Phase 4 (evaluation).
    - Represents final trained state of the validation component.
"""

# -------------------------
# Imports
# -------------------------
import pandas as pd
import numpy as np
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import random
import torch

# -------------------------
# 0. Reproducibility
# -------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------
# 1. Config
# -------------------------

TRAIN_PATH = "data/extraction/new_splits/train.csv"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

OUTPUT_DIR = Path("models/bioclinicalbert_final")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512

CONFIG = {
    "learning_rate": 3e-6,
    "batch_size": 8,
    "epochs": 5,
    "weight_decay": 0.05,
    "grad_accum": 2,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0
}

# -------------------------
# 2. Load Data
# -------------------------

train_df = pd.read_csv(TRAIN_PATH)
train_df["label"] = train_df["is_valid"].astype(int) # Ensure label column is integer type for classification

print(f"Training size: {len(train_df)}")

# -------------------------
# 3. Tokenizer
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # Load tokenizer for the specified model

def tokenize(batch):
    texts = [
        f"[SECTION] {sec} [ENTITY TYPE] {et} [ENTITY] {e} [CONCEPT] {c} [TASK] {t} [TEXT] {st}"
        for sec, et, e, c, t, st in zip( # zip is used to iterate over multiple columns simultaneously
            batch['section'],
            batch['entity_type'],
            batch['entity_text'],
            batch['concept'],
            batch['task'],
            batch['sentence_text']
        )
    ]

    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# -------------------------
# 4. Dataset Preparation
# -------------------------

cols = ["section", "entity_type", "entity_text", "concept", "task", "sentence_text", "label"]

# Convert DataFrame to Hugging Face Dataset format for compatibility with Trainer API
dataset = Dataset.from_pandas(train_df[cols])

# Remove index column if present
if "__index_level_0__" in dataset.column_names:
    dataset = dataset.remove_columns(["__index_level_0__"])

# Apply tokenization to the dataset using the defined tokenize function
dataset = dataset.map(tokenize, batched=True)

# Remove original text columns as they are no longer needed after tokenization
dataset = dataset.remove_columns(
    ["section", "entity_type", "entity_text", "concept", "task", "sentence_text"]
)

# Set the format of the dataset to PyTorch tensors for use with the Trainer API
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -------------------------
# 5. Training Configuration
# -------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="no",
    save_strategy="no",
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    num_train_epochs=CONFIG["epochs"],
    lr_scheduler_type="linear",
    warmup_ratio=CONFIG["warmup_ratio"],
    gradient_accumulation_steps=CONFIG["grad_accum"],
    weight_decay=CONFIG["weight_decay"],
    max_grad_norm=CONFIG["max_grad_norm"],
    logging_steps=10,
    report_to="none"
)

# -------------------------
# 6. Model Initialisation
# -------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# -------------------------
# 7. Training
# -------------------------

print("\n===== TRAINING FINAL MODEL =====\n")
trainer.train()

# -------------------------
# 8. Save Model
# -------------------------

trainer.save_model(OUTPUT_DIR)

print("\n===== FINAL MODEL SAVED =====")
print(f"All files: {OUTPUT_DIR}")