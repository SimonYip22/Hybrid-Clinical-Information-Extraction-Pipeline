# End-to-End Pipeline Plan (Post-CV)

## Overview
This document outlines the remaining steps after cross-validation (CV) and model selection. The goal is to produce a valid, reproducible, and deployment-ready pipeline without data leakage.

---

# Phase 1 — Model Selection (Completed)
- Performed 5-fold CV
- Compared configurations
- Selected: `config_advanced`

No further CV required.

---

# Phase 2 — Threshold Tuning (Calibration)

## Objective
Convert the transformer into a **precision-first model**.

## Inputs
- CV validation outputs:
  - Predicted probabilities
  - True labels

## Steps
1. Aggregate predictions across all CV folds
2. Generate:
   - Precision–Recall curve
   - Threshold vs Precision/Recall table
3. Select threshold:
   - Example: choose threshold where **precision ≥ 0.80**

## Important Constraints
- DO NOT use test set (180 samples)
- Use only CV validation predictions

## Output
- Fixed decision threshold (e.g. 0.78)

---

# Phase 3 — Final Model Training

## Objective
Train a single final model using best configuration

## Data
- Full training set: **1020 samples**

## Steps
- Train using `config_advanced`
- Same hyperparameters as CV
- No validation splitting required here

## Output
- Final trained model
- Fixed threshold (from Phase 2)

---

# Phase 4 — Final Evaluation (Held-Out Test Set)

## Objective
Obtain unbiased final performance metrics

## Data
- Test set: **180 samples (never seen before)**

## Steps
1. Input test data (features only)
2. Model outputs probabilities
3. Apply threshold: prediction = probability > threshold

4. Compare predictions vs ground truth labels

## Metrics to Compute
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Output
- These are the **final reported metrics**

---

# Phase 5 — Visualisations

## Required
1. Pipeline diagram
- Rule-based → Transformer → Output

2. Precision–Recall curve (from CV)
- Show chosen threshold

3. Confusion matrix (test set)

4. Model comparison bar chart:
- Rule-based vs Transformer vs Combined

## Optional
- Training loss curves
- ROC curve

---

# Phase 6 — Error Analysis

## Objective
Understand model limitations

## Steps
1. Extract:
- False positives
- False negatives

2. Categorise errors:
- Negation handling errors
- Ambiguous phrasing
- Domain-specific cases
- Annotation inconsistencies

## Output
- Structured error table
- Key insights

---

# Phase 7 — Ablation Study (Optional)

## Only include if simple

Examples:
- Rule-based only vs Transformer
- Transformer without threshold tuning
- Different threshold values

## Skip if time-constrained

---

# Phase 8 — Full Dataset Inference (160,000 Reports)

## Objective
Demonstrate real-world scalability

## Pipeline
160k reports
↓
rule-based extraction
↓
transformer validation (with threshold)
↓
final entities (~500k expected)

## Notes
- No labels used here
- This is NOT evaluation

## Output
- Final extracted dataset
- Aggregate statistics

---

# Phase 9 — Deployment

## Objective
Provide usable inference pipeline

## Minimum Viable Setup

### Option 1 — CLI
python run_pipeline.py input.txt
### Option 2 — Simple API (optional)
- FastAPI endpoint

## Hosting (optional)
- Lightweight platforms:
  - Render / Railway / HuggingFace Spaces

---

# Phase 10 — Batch vs Individual Inference

## Clarification
Same pipeline, different usage modes:

### Individual
- 1 report → output
- Used for demo/UI

### Batch
- N reports → outputs
- Used for 160k dataset

## Implementation
- Single pipeline wrapped in loop

---

# Phase 11 — CI/CD (Optional)

## Minimal Version
- GitHub repository
- README with usage instructions

## Optional
- Basic GitHub Actions for:
  - Linting
  - Script testing

## Not required for project success

---

# Final Execution Order

1. Threshold tuning (CV predictions)
2. Train final model (1020 samples)
3. Evaluate on test set (180 samples)
4. Generate visualisations
5. Perform error analysis
6. (Optional) ablation study
7. Run full 160k dataset
8. Build simple deployment interface

---

# Key Rules (Critical)

- Never use test set during tuning
- Threshold is chosen BEFORE final evaluation
- Final metrics come ONLY from test set
- Full dataset run is separate from evaluation

---

# Estimated Timeline

- Threshold tuning + plots: 1–2 days
- Final training + evaluation: 1 day
- Error analysis + write-up: 3–5 days
- Full dataset + deployment: 2–3 days

---

# Final Output Summary

## You will produce:
- Final trained model
- Fixed decision threshold
- Test set metrics (primary results)
- Visualisations
- Error analysis
- Large-scale dataset (160k processed)
- Simple deployment pipeline

---