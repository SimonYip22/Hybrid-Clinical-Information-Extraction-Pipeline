"""
generate_oof_predictions.py

Purpose:
    - Generate out-of-fold (OOF) predicted probabilities using stratified
      K-fold cross-validation with the selected advanced configuration.
    - Provide unbiased probability estimates for every sample in the training set.
    - These probabilities are used exclusively for downstream threshold tuning.

Workflow:
    1. Set reproducibility seeds for deterministic behaviour.
    2. Load full training dataset (no train/validation split).
    3. Initialise empty storage arrays:
        - oof_probs: stores predicted probabilities for each sample
        - oof_labels: stores ground truth labels
    4. Perform stratified K-fold split (K=5):
        - Maintain class balance in each fold
    5. For each fold:
        a. Split into training and validation subsets
        b. Convert to Hugging Face Dataset format
        c. Tokenise structured inputs
        d. Train model using advanced configuration
        e. Generate predictions on validation fold:
            - Extract logits (raw model outputs)
            - Apply softmax to convert to probabilities
            - Select probability of positive class (label=1)
        f. Store predictions in oof_probs at correct indices
    6. After all folds:
        - Combine predictions into a single DataFrame
        - Each row now has:
            - True label (y_true)
            - Predicted probability (y_prob)
    7. Save OOF predictions to disk.
    8. Perform basic sanity checks.

Outputs:
    results/threshold_tuning/
        └── oof_predictions.csv

Columns:
    - y_true: ground truth binary labels (0 or 1)
    - y_prob: predicted probability of positive class (continuous [0,1])

Properties of OOF Output:
    - Size equals full training set (N samples)
    - Each prediction is out-of-sample (no leakage)
    - Suitable for:
        - Threshold tuning
        - Precision–recall analysis
        - Calibration assessment

Notes:
    - No evaluation metrics are computed in this script.
    - No model checkpoints are saved.
    - No thresholding is applied here (probabilities only).
    - This script is intentionally minimal and focused on probability generation.
"""

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
from sklearn.model_selection import StratifiedKFold

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

OUTPUT_DIR = Path("results/threshold_tuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512
NUM_FOLDS = 5

# Advanced config only
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
train_df["label"] = train_df["is_valid"].astype(int) # convert ground truth True/False to binary labels

print(f"Train size: {len(train_df)}")

# Prepare OOF storage
oof_probs = np.zeros(len(train_df)) # create an array with zeros to store predicted probabilities for the positive class
oof_labels = train_df["label"].values # create an array to store true labels (for sanity checks later)

# -------------------------
# 3. Tokenizer
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    texts = [
        f"[SECTION] {sec} [ENTITY TYPE] {et} [ENTITY] {e} [CONCEPT] {c} [TASK] {t} [TEXT] {st}"
        for sec, et, e, c, t, st in zip(
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
# 4. Cross Validation (OOF)
# -------------------------

print("\n===== GENERATING OOF PREDICTIONS =====")

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
 
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])): # train_df["label"] is the target for stratification
    print(f"\n=== Fold {fold + 1} / {NUM_FOLDS} ===")

    train_fold_df = train_df.iloc[train_idx] # get training data for this fold
    val_fold_df = train_df.iloc[val_idx] # get validation data for this fold

    cols = ["section", "entity_type", "entity_text", "concept", "task", "sentence_text", "label"]

    train_fold = Dataset.from_pandas(train_fold_df[cols])
    val_fold = Dataset.from_pandas(val_fold_df[cols])

    # Remove the default index column added by from_pandas (if it exists)
    if "__index_level_0__" in train_fold.column_names:
        train_fold = train_fold.remove_columns(["__index_level_0__"])
        val_fold = val_fold.remove_columns(["__index_level_0__"])

    train_fold = train_fold.map(tokenize, batched=True)
    val_fold = val_fold.map(tokenize, batched=True)

    # Remove original text columns to save memory (only keep tokenized inputs and labels)
    train_fold = train_fold.remove_columns(
        ["section", "entity_type", "entity_text", "concept", "task", "sentence_text"]
    )
    val_fold = val_fold.remove_columns(
        ["section", "entity_type", "entity_text", "concept", "task", "sentence_text"]
    )

    # Set format for PyTorch (only keep input_ids, attention_mask, and label)
    train_fold.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_fold.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="no",   # no need for eval metrics here
        save_strategy="no",
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        lr_scheduler_type="linear",
        warmup_ratio=CONFIG["warmup_ratio"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        weight_decay=CONFIG["weight_decay"],
        max_grad_norm=CONFIG["max_grad_norm"],
        logging_steps=10,
        report_to="none"
    )

    # Initialize model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_fold,
        tokenizer=tokenizer
    )

    trainer.train()

    # -------------------------
    # GET PROBABILITIES
    # -------------------------

    preds = trainer.predict(val_fold) # this returns a PredictionOutput object with .predictions and .label_ids
    logits = preds.predictions # take raw model outputs (logits) for each class; shape = (num_samples_in_val_fold, num_labels=2)

    # Convert logits to probabilities (for all rows of the positive class) using softmax across columns (classes)
    # Softmax is pytorch but logits is numpy, so we convert logits to a torch tensor first, apply softmax, then convert back to numpy
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    # Store OOF predictions
    oof_probs[val_idx] = probs

# -------------------------
# 5. Save OOF
# -------------------------

oof_df = pd.DataFrame({
    "y_true": oof_labels, # true labels for sanity checks (not strictly needed for threshold tuning, but useful to have)
    "y_prob": oof_probs   # predicted probabilities for the positive class (label=1)
})

oof_path = OUTPUT_DIR / "oof_predictions.csv"
oof_df.to_csv(oof_path, index=False)

print("\n===== OOF COMPLETE =====")
print(f"Saved to: {oof_path}")

# -------------------------
# 6. Basic Validation
# -------------------------

print("\nSanity checks:")
print(f"OOF shape: {oof_df.shape}") # should be (num_samples_in_train_df, 2) since we have y_true and y_prob columns
print(f"Label distribution:\n{oof_df['y_true'].value_counts(normalize=True)}") # should be similar to original distribution (but not identical due to stratification and fold splits)
print(f"Prob range: min={oof_df['y_prob'].min():.4f}, max={oof_df['y_prob'].max():.4f}") # should be between 0 and 1 due to softmax