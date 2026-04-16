"""
evaluation_metrics.py

Purpose:
    Compute core evaluation metrics for pipeline-level comparison between:
    - Rule-based extraction (baseline)
    - Transformer validation (final system)
    Allows for a direct, quantitative comparison of system performance on the unified evaluation dataset

Workflow:
    1. Load unified evaluation dataset (pipeline_predictions.csv)
    2. Extract:
        - Ground truth labels (y_true)
        - Rule-based predictions (rule_pred)
        - Transformer predictions (model_pred)
    3. Compute metrics separately for each system:
        - Accuracy, Precision, Recall, F1 score
        - Confusion matrix (TP, FP, TN, FN)
    4. Combine results into a structured comparison table:
        - Rows represent systems (Rule-Based, Transformer)
        - Columns represent metrics
    5. Save results for downstream analysis and reporting

Outputs:
    outputs/evaluation/core_metrics.csv
    Structure:
        - Rows: Rule-Based, Transformer
        - Columns: accuracy, precision, recall, f1_score, true_negatives, false_positives, false_negatives, true_positives
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import pathlib as Path

# -------------------------
# 0. Configuration
# -------------------------
INPUT_DIR = "outputs/evaluation/pipeline_predictions.csv"
OUTPUT_DIR = Path("outputs/evaluation/core_metrics.csv")

OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1. Load Data
# -------------------------
df = pd.read_csv(INPUT_DIR)

y_true = df["y_true"]
rule_pred = df["rule_pred"]
model_pred = df["model_pred"]

# -------------------------
# 2. Function to Compute Metrics
# -------------------------
def compute_metrics(y_true, y_pred):
    """
    Compute core classification metrics and confusion matrix components.

    Args:
        y_true (array-like):
            Ground truth binary labels (0 or 1)

        y_pred (array-like):
            Predicted binary labels (0 or 1)

    Returns:
        dict:
            Dictionary containing:
                - accuracy (float): Overall correctness
                - precision (float): TP / (TP + FP)
                - recall (float): TP / (TP + FN)
                - f1_score (float): Harmonic mean of precision and recall
                - true_negatives (int): Correctly predicted negatives
                - false_positives (int): Incorrectly predicted positives
                - false_negatives (int): Missed positives
                - true_positives (int): Correctly predicted positives

    Notes:
        - Confusion matrix is computed using sklearn:
            [[TN, FP],
             [FN, TP]]
        - Values are flattened using `.ravel()` into:
            tn, fp, fn, tp
        - These components provide insight into error types,
          which complements aggregate metrics such as precision and recall
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    }

# -------------------------
# 3. Compute Metrics
# -------------------------
rule_metrics = compute_metrics(y_true, rule_pred)
model_metrics = compute_metrics(y_true, model_pred)

# -------------------------
# 4. Save Results
# -------------------------
results = pd.DataFrame(
    [rule_metrics, model_metrics], # Each dict corresponds to a row, aligned by keys as columns
    index=["Rule-Based", "Transformer"] # Index to assign names to rows
)

results.to_csv("outputs/evaluation/core_metrics.csv")