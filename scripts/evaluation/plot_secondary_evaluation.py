"""
plot_secondary_evaluation.py

Purpose:
    Generate Layer 2 (Stratified & Diagnostic) evaluation plots to explain
    model and pipeline behaviour beyond aggregate metrics.
    This script produces:
    - Precision–Recall curve (model behaviour across thresholds)
    - Stratified performance by entity type (pipeline behaviour)
    - Probability distribution of model outputs (class separation)

Workflow:
    1. Load unified evaluation dataset (pipeline_predictions.csv)
    2. Extract:
        - Ground truth labels (y_true)
        - Model probabilities (model_prob)
    3. Generate Precision–Recall curve:
        - Plot precision vs recall across thresholds
        - Mark chosen operating threshold (0.549)
    4. Compute stratified metrics:
        - Precision, recall, F1 per entity type
        - For both rule-based and transformer systems
    5. Generate stratified bar plots:
        - Compare rule-based vs transformer across entity types
    6. Generate probability distribution plot:
        - Show separation between positive and negative classes
        - Mark decision threshold

Outputs:
    outputs/evaluation/
        - stratified_metrics.csv

    outputs/evaluation/secondary_plots/
        - pr_curve.png
        - stratified_metrics.png
        - probability_distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from pathlib import Path

# -------------------------
# 1. Load Data
# -------------------------

df = pd.read_csv("outputs/evaluation/pipeline_predictions.csv")

y_true = df["y_true"]
model_prob = df["model_prob"]

# Output directory
OUT_DIR = Path("outputs/evaluation/secondary_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 2. Precision–Recall Curve
# -------------------------

def plot_pr_curve(y_true, y_prob, save_path):
    """
    Plot Precision–Recall curve and highlight selected threshold.

    Args:
        y_true (array-like):
            Ground truth binary labels

        y_prob (array-like):
            Predicted probabilities (model_prob)

        save_path (Path):
            File path to save the plot

    Purpose:
        - Visualises model performance across all thresholds
        - Shows precision–recall trade-off
        - Validates chosen threshold behaviour (0.549)

    Notes:
        - precision_recall_curve returns:
            precision, recall (length n+1)
            thresholds (length n)
        - Threshold index must be aligned carefully with precision/recall
    """
    threshold = 0.549

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Find closest threshold index
    idx = np.argmin(np.abs(thresholds - threshold))

    plt.figure()

    # PR curve
    plt.plot(recall, precision, label="PR Curve")

    # Mark chosen threshold (use idx+1 for alignment)
    plt.scatter(recall[idx + 1], precision[idx + 1],
                label=f"Threshold = {threshold}",
                zorder=5)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()

    plt.savefig(save_path)
    plt.close()


plot_pr_curve(
    y_true,
    model_prob,
    OUT_DIR / "pr_curve.png"
)

# -------------------------
# 3. Stratified Metrics
# -------------------------

def compute_basic_metrics(y_true, y_pred):
    """
    Compute precision, recall, and F1 score.

    Args:
        y_true (array-like):
            Ground truth labels

        y_pred (array-like):
            Predicted binary labels

    Returns:
        dict:
            precision, recall, f1_score

    Notes:
        - zero_division=0 prevents errors when no positive predictions exist
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


def compute_stratified_metrics(df):
    """
    Compute metrics per entity type for both systems.

    Args:
        df (DataFrame):
            Must contain:
                - entity_type
                - y_true
                - rule_pred
                - model_pred

    Returns:
        DataFrame:
            Columns:
                entity_type, system, precision, recall, f1_score

    Purpose:
        - Identify where performance varies across entity types
        - Compare rule-based vs transformer behaviour
        - Explain where improvements or degradations occur
    """
    results = []

    for entity in df["entity_type"].unique():
        subset = df[df["entity_type"] == entity]

        # Rule-based
        rule_metrics = compute_basic_metrics(subset["y_true"], subset["rule_pred"])
        results.append({
            "entity_type": entity,
            "system": "Rule-Based",
            **rule_metrics
        })

        # Transformer
        model_metrics = compute_basic_metrics(subset["y_true"], subset["model_pred"])
        results.append({
            "entity_type": entity,
            "system": "Transformer",
            **model_metrics
        })

    return pd.DataFrame(results)


stratified_df = compute_stratified_metrics(df)
stratified_df.to_csv("outputs/evaluation/stratified_metrics.csv", index=False)

# -------------------------
# 4. Stratified Bar Plot
# -------------------------

def plot_stratified_metrics(df, save_path):
    """
    Plot comparison of precision, recall, and F1 across entity types.

    Args:
        df (DataFrame):
            Stratified metrics dataframe

        save_path (Path):
            Output file path

    Purpose:
        - Visualise performance differences across entity types
        - Identify where transformer improves or degrades performance
        - Support interpretation of pipeline behaviour
    """
    metrics = ["precision", "recall", "f1_score"]
    entity_types = df["entity_type"].unique()

    x = np.arange(len(entity_types))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        rule_vals = []
        model_vals = []

        for entity in entity_types:
            rule_vals.append(
                df[(df["entity_type"] == entity) & (df["system"] == "Rule-Based")][metric].values[0]
            )
            model_vals.append(
                df[(df["entity_type"] == entity) & (df["system"] == "Transformer")][metric].values[0]
            )

        ax.bar(x - width/2, rule_vals, width, label="Rule-Based")
        ax.bar(x + width/2, model_vals, width, label="Transformer")

        ax.set_title(metric.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(entity_types)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


plot_stratified_metrics(
    stratified_df,
    OUT_DIR / "stratified_metrics.png"
)

# -------------------------
# 5. Probability Distribution
# -------------------------

def plot_probability_distribution(y_true, y_prob, save_path):
    """
    Plot distribution of predicted probabilities by true class.

    Args:
        y_true (array-like):
            Ground truth labels

        y_prob (array-like):
            Predicted probabilities

        save_path (Path):
            Output file path

    Purpose:
        - Visualise separation between positive and negative classes
        - Explain false positive / false negative behaviour
        - Show impact of chosen decision threshold
    """
    plt.figure()

    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.5, label="True Class = 1")
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.5, label="True Class = 0")

    plt.axvline(x=0.549, linestyle='--', label="Threshold")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution by True Class")
    plt.legend()

    plt.savefig(save_path)
    plt.close()


plot_probability_distribution(
    y_true,
    model_prob,
    OUT_DIR / "probability_distribution.png"
)