"""
train_validate_transformer.py

Purpose:
    - Train and validate a transformer-based binary classification model (BioClinicalBERT)
      for entity validation in clinical text. 
    - The script performs standard fine-tuning using a predefined train/validation split, 
      monitors performance via validation metrics, and assesses robustness using stratified K-fold cross-validation.

Workflow:
    1. Set reproducibility seeds for deterministic training.
    2. Load pre-split training and validation datasets from disk.
    3. Convert raw data into Hugging Face Dataset format.
    4. Construct structured input sequences combining:
    entity_type, entity_text, concept, task, and sentence_text.
    5. Tokenize inputs using the pretrained BioClinicalBERT tokenizer.
    6. Load BioClinicalBERT with a randomly initialised classification head.
    7. Define evaluation metrics (accuracy, precision, recall, F1).
    8. Configure training parameters (batch size, learning rate, scheduler, etc.).
    9. Train the model using Hugging Face Trainer with epoch-level validation.
    10. Perform 5-fold stratified cross-validation to assess robustness:
        - Reset model for each fold
        - Train and evaluate independently
        - Aggregate metrics across folds
    11. Save the final trained model and tokenizer.

Outputs:
- Trained model and tokenizer:
  models/bioclinicalbert/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json

- Training logs (stdout):
  - Training and validation metrics per epoch
  - Cross-validation metrics per fold
  - Aggregated mean and standard deviation of metrics

- No intermediate artefacts are persisted beyond checkpoints
  (limited by save_total_limit=2).
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

# -------------------------
# 0. Reproducibility
# -------------------------
import random
import torch

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

CV_OUTPUT_DIR = Path("models/bioclinicalbert_v2/cross_validation")
CV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512
NUM_FOLDS = 5

HYPERPARAM_CONFIGS = [
    {
        "name": "config_stable",
        "learning_rate": 5e-6,
        "batch_size": 8,
        "epochs": 3,
        "weight_decay": 0.0,              # explicitly none
        "grad_accum": 1,                 # no accumulation
        "warmup_ratio": 0.0,              # no warmup
        "max_grad_norm": 1.0
    },
    {
        "name": "config_advanced",
        "learning_rate": 3e-6,
        "batch_size": 8,
        "epochs": 5,
        "weight_decay": 0.05,
        "grad_accum": 2,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0
    }
]

# -------------------------
# 2. Load Data
# -------------------------

train_df = pd.read_csv(TRAIN_PATH)
train_df["label"] = train_df["is_valid"].astype(int)

print(f"Train size: {len(train_df)}")

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
# 4. Metrics
# -------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# -------------------------
# 5. Cross Validation
# -------------------------

print("\n===== RUNNING CROSS-VALIDATION =====")

all_config_results = []

for config in HYPERPARAM_CONFIGS:
    print(f"\n\n########## {config['name']} ##########")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f"\n=== Fold {fold + 1} / {NUM_FOLDS} ===")

        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]

        cols = ["section", "entity_type", "entity_text", "concept", "task", "sentence_text", "label"]

        train_fold = Dataset.from_pandas(train_fold_df[cols])
        val_fold = Dataset.from_pandas(val_fold_df[cols])

        # Remove index column if added
        if "__index_level_0__" in train_fold.column_names:
            train_fold = train_fold.remove_columns(["__index_level_0__"])
            val_fold = val_fold.remove_columns(["__index_level_0__"])

        train_fold = train_fold.map(tokenize, batched=True)
        val_fold = val_fold.map(tokenize, batched=True)

        train_fold = train_fold.remove_columns(
            ["section", "entity_type", "entity_text", "concept", "task", "sentence_text"]
        )
        val_fold = val_fold.remove_columns(
            ["section", "entity_type", "entity_text", "concept", "task", "sentence_text"]
        )

        train_fold.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_fold.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # -------------------------
        # Dynamic TrainingArguments
        # -------------------------

        training_args = TrainingArguments(
            output_dir=str(CV_OUTPUT_DIR / config['name'] / f"fold_{fold}"),
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["epochs"],
            lr_scheduler_type="linear",
            warmup_ratio=config["warmup_ratio"],
            gradient_accumulation_steps=config["grad_accum"],
            weight_decay=config["weight_decay"],
            max_grad_norm=config["max_grad_norm"],
            logging_steps=10,
            report_to="none"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate()

        fold_metrics.append(metrics)
        print(f"Fold {fold + 1} metrics: {metrics}")

    # -------------------------
    # Aggregate Results
    # -------------------------

    metrics_df = pd.DataFrame(fold_metrics)

    # Save raw fold results
    metrics_df.to_csv(CV_OUTPUT_DIR / f"{config['name']}_folds.csv", index=False)

    eval_cols = [c for c in metrics_df.columns if c.startswith("eval_")]

    mean_metrics = metrics_df[eval_cols].mean()
    std_metrics = metrics_df[eval_cols].std()

    print("\n--- SUMMARY ---")
    print("Mean:\n", mean_metrics)
    print("Std:\n", std_metrics)

    all_config_results.append({
        "config": config["name"],
        "mean_accuracy": mean_metrics["eval_accuracy"],
        "std_accuracy": std_metrics["eval_accuracy"],
        "mean_f1": mean_metrics["eval_f1"],
        "std_f1": std_metrics["eval_f1"],
        "mean_precision": mean_metrics["eval_precision"],
        "std_precision": std_metrics["eval_precision"],
        "mean_recall": mean_metrics["eval_recall"],
        "std_recall": std_metrics["eval_recall"],
    })

# -------------------------
# Final Comparison
# -------------------------

results_df = pd.DataFrame(all_config_results)

# Save summary + comparison
results_df.to_csv(CV_OUTPUT_DIR / "final_comparison.csv", index=False)

print("\n===== FINAL COMPARISON =====")
print(results_df.sort_values(by="mean_f1", ascending=False))