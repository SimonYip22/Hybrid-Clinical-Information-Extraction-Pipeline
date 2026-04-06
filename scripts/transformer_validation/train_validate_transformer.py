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

TRAIN_PATH = "data/extraction/splits/train.csv"
VAL_PATH = "data/extraction/splits/val.csv"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
OUTPUT_DIR = "models/bioclinicalbert"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 3e-6

# -------------------------
# 2. Load Data
# -------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")

# Convert boolean → int labels (required for model training)
train_df["label"] = train_df["is_valid"].astype(int)
val_df["label"] = val_df["is_valid"].astype(int)

# -------------------------
# 3. Create HF Datasets
# -------------------------

# Convert pandas DataFrames to Hugging Face Dataset for training with the Transformers library
train_dataset = Dataset.from_pandas(
    train_df[["sentence_text", "entity_type", "entity_text", "concept", "task", "label"]]
)
val_dataset = Dataset.from_pandas(
    val_df[["sentence_text", "entity_type", "entity_text", "concept", "task", "label"]]
)

# -------------------------
# 4. Tokenization
# -------------------------

# Load the tokenizer for the pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to format input text for the model
def format_input(example):
    return (
        f"[ENTITY TYPE] {example['entity_type']} "
        f"[ENTITY] {example['entity_text']} "
        f"[CONCEPT] {example['concept']} "
        f"[TASK] {example['task']}"
        f"[TEXT] {example['sentence_text']}"
    )

# Tokenization function applied to every example in the dataset
def tokenize(batch):
    # Combine relevant fields into a single input string for the model using the specified format.
    texts = [
        f"[ENTITY TYPE] {et} [ENTITY] {e} [CONCEPT] {c} [TASK] {t} [TEXT] {s}"
        for et, e, c, t, s in zip( # Zip combines the fields from the batch into a single iterable of tuples
            batch['entity_type'],
            batch['entity_text'],
            batch['concept'],
            batch['task'],
            batch['sentence_text']
        )
    ]

    return tokenizer(
        texts,
        truncation=True, # Truncate sentences longer than MAX_LENGTH tokens
        padding="max_length", # Pad sentences shorter than MAX_LENGTH tokens
        max_length=MAX_LENGTH # Fix the input length to MAX_LENGTH for consistent input size to the model
    )

# Apply tokenization to the entire dataset. `batched=True` allows processing multiple examples at once for efficiency.
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Convert datasets into Pytorch tensors to allow model training.
# We specify the columns we need for training: `input_ids` (token IDs), `attention_mask` (mask to ignore padding tokens), and `label` (the target variable).
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -------------------------
# 5. Load Model
# -------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2 # Binary classification heads (valid vs invalid)
)

# -------------------------
# 6. Metrics (Validation Only)
# -------------------------

# Define a function to compute evaluation metrics during validation
def compute_metrics(eval_pred):
    # logits = raw model outputs, labels = ground truth labels
    logits, labels = eval_pred # `eval_pred` is a tuple of (logits, labels) returned by the model during evaluation
    preds = np.argmax(logits, axis=1) # Converts raw logits to predicted class labels (0 or 1) by taking the index of the highest logit value

    # Compute precision, recall, F1-score using sklearn's utility function.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary" # `average="binary"` is used for binary classification to compute metrics for the positive class (label=1).
    )

    acc = accuracy_score(labels, preds) # Compute accuracy as the proportion of correct predictions (both true positives and true negatives) out of all predictions.

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# -------------------------
# 7. Training Config
# -------------------------

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch", # Validate every epoch
    save_strategy="epoch", # Save model checkpoint every epoch
    learning_rate=LEARNING_RATE, 
    per_device_train_batch_size=BATCH_SIZE, # Batch size for training (number of examples processed together in one forward/backward pass)
    per_device_eval_batch_size=BATCH_SIZE, # Batch size for evaluation
    num_train_epochs=EPOCHS,
    warmup_ratio=0.1, # Warm up the learning rate for the first 10% of training steps to help stabilize training in the early stages
    lr_scheduler_type="linear", # Use a learning rate scheduler that warms up the learning rate linearly for the first few steps and then decays it linearly
    gradient_accumulation_steps=2,
    max_grad_norm=1.0, # Gradient clipping to prevent exploding gradients by capping the maximum norm of the gradients during backpropagation
    weight_decay=0.05, # L2 regularization to prevent overfitting by penalizing large weights
    load_best_model_at_end=True, # After training, load the model checkpoint that performed best on the validation set according to the specified metric
    metric_for_best_model="f1", # Use F1-score to determine the best model checkpoint during training (since we care about both precision and recall in this binary classification task)
    logging_steps=10, # Log training metrics every 10 steps
    save_total_limit=2 # Limit the total number of saved checkpoints to 2
)

# -------------------------
# 8. Trainer
# -------------------------

# Trainer is a high-level API provided by Hugging Face Transformers that abstracts away the training loop and evaluation logic.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -------------------------
# 9. K-Fold Cross Validation (NEW)
# -------------------------
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy

NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
all_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    print(f"\n=== Fold {fold + 1} / {NUM_FOLDS} ===")
    
    train_fold = Dataset.from_pandas(
        train_df.iloc[train_idx][["sentence_text", "entity_type", "entity_text", "concept", "task", "label"]]
    )
    val_fold = Dataset.from_pandas(
        train_df.iloc[val_idx][["sentence_text", "entity_type", "entity_text", "concept", "task", "label"]]
    )
    
    train_fold = train_fold.map(tokenize, batched=True)
    val_fold = val_fold.map(tokenize, batched=True)
    
    train_fold.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_fold.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    fold_model = deepcopy(model)  # reset model weights for this fold
    
    trainer = Trainer(
        model=fold_model,
        args=training_args,
        train_dataset=train_fold,
        eval_dataset=val_fold,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    metrics = trainer.evaluate()
    all_metrics.append(metrics)
    print(f"Fold {fold + 1} metrics: {metrics}")

metrics_df = pd.DataFrame(all_metrics)
print("CV mean metrics:\n", metrics_df.mean())
print("CV std metrics:\n", metrics_df.std())

# -------------------------
# 10. Save Model
# -------------------------

trainer.save_model(OUTPUT_DIR) # Saves the fine-tuned model weights and configuration to the specified output directory
tokenizer.save_pretrained(OUTPUT_DIR) # Saves the tokenizer configuration and vocabulary to the same directory so it can be loaded later for inference

print("Training complete. Model saved.")