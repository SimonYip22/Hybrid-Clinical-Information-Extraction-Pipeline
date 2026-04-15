# Phase 4 – Model Evaluation Decisions

## Objective

This phase defines the evaluation framework used to assess both the trained transformer model and the overall extraction pipeline on a fully held-out test set (n = 180). This phase evaluates:

- Baseline rule-based extraction vs ground truth  
- Transformer validation vs ground truth  
- Net improvement introduced by the transformer layer  
- Precision–recall trade-offs under fixed thresholding  
- Performance variation across entity types  

The goal is to verify that the transformer operates as a precision-oriented validation layer, improving correctness over a high-recall extraction baseline, ready for deployment in the full ICU corpus.

---

## Evaluation Metrics Dataset 

### 1. Objective

Evaluation is based on constructing a single unified dataset where all prediction sources are aligned at the entity level. Each row corresponds to one candidate entity and contains:

- Ground truth label
- Rule-based prediction
- Transformer prediction (probability + thresholded output)

This dataset is the foundation for all evaluation, enabling:

- Metric computation (precision, recall, F1, accuracy)
- Confusion matrix analysis
- Rule vs transformer comparison
- Threshold-dependent analysis (PR/ROC curves)
- Stratified analysis by entity type

No metrics are computed during this stage, this step strictly produces the evaluation-ready dataset which can derive metrics downstream. This separation ensures reproducibility and flexibility in analysis.

---

### 2. Evaluation Components

#### 2.1 Ground Truth

- Source: manually annotated test set (n = 180)
- Label: `is_valid → y_true ∈ {0,1}`

This represents the only unbiased reference and is used to evaluate all systems.

---

#### 2.2 Rule-Based Predictions

The rule-based system represents the high-recall extraction baseline.

- All extracted entities are assumed valid (1)  
- Exception: **SYMPTOM entities**
  - `negated = True → 0 (invalid)`
  - `negated = False → 1 (valid)`

Rationale:

- The extraction stage prioritises recall (capture everything) as rule-based methoda are limited in precision due to semantic ambiguity and lack of context understanding
- Minimal precision logic is applied only where trivial and high-impact (negation of symptoms)
- This creates a weak but acceptable baseline, allowing measurement of transformer improvement

---

#### 2.3 Transformer Predictions

The transformer acts as a validation layer applied after extraction.

For each entity:

- Outputs logits → converted to probability: `model_prob = p(y = 1)`
- Applies tuned threshold: `model_pred = (model_prob ≥ 0.549)`

Key properties:

- This is the first use of the tuned threshold
- These predictions represent the final pipeline output and are directly compared to ground truth for performance evaluation
- The model is explicitly precision-oriented to improve correctness compared to the rule-based baseline, so we expect:
  - Precision ↑ (fewer false positives)
  - Recall ↓ (some true positives removed)

---

### 3. Dataset Design

Each row represents one extracted entity and aligns:

| Column        | Description |
|--------------|-------------|
| `entity_type` | Entity category (for stratified analysis) |
| `y_true`      | Ground truth label (manual annotation) |
| `rule_pred`   | Rule-based prediction |
| `model_prob`  | Transformer probability (p(y=1)) |
| `model_pred`  | Final transformer prediction (thresholded) |

Design rationale:

- **Single-table design:** Ensures all systems are evaluated against the same reference `y_true` without recomputation
- **model_prob retained:** Required for PR curves, ROC analysis, and threshold sensitivity (these cannot be derived from binary outputs)
- **model_pred included:** Represents the actual deployed decision rule as the final output of the pipeline
- **entity_type included:** Enables analysis of where the pipeline performs well or poorly (entity-type stratification)
- **Separation of concerns:** This dataset is purely for evaluation. No metrics are computed here, which is reserved for the next step (plots and analysis). This allows for flexible analysis and reproducibility.

---

### 4. Workflow Implementation

All script code and logic is implemented in `run_evaluation.py` with these steps:

1. **Initialise environment**
  - Select computation device:
    - GPU if available, otherwise CPU
  - Defines execution context for inference

2. **Load test dataset**
  - Read held-out test set (n = 180)
  - Define ground truth:
    - `y_true = is_valid` (binary labels)

3. **Compute rule-based predictions**
  - Apply deterministic extraction logic:
    - `SYMPTOM`: validity determined by `negated` (0 if negated, 1 if not)
    - All other entity types: assumed valid (1)
  - Output baseline:
    - `rule_pred ∈ {0,1}`

4. **Load trained model and tokenizer**
  - Load final model weights from `bioclinicalbert_final/`
  - Load corresponding tokenizer (ensures identical token mapping)
  - Move model to selected device
  - Set `model.eval()`:
    - Disables dropout
    - Ensures deterministic inference

5. **Reconstruct model inputs**
  - Concatenate structured fields into a single string:
    - `[SECTION] ... [ENTITY TYPE] ... [ENTITY] ... [CONCEPT] ... [TASK] ... [TEXT] ...`
  - Matches training-time input format exactly

6. **Tokenise inputs**
  - Convert text → token IDs + attention masks
  - Apply:
    - truncation (`max_length = 512`)
	- dynamic padding (batch-level)
  - Output: tensors aligned with model input requirements

7. **Run batched inference**
  - Iterate over dataset in fixed-size batches
  - Move inputs to same device as model
  - Forward pass only (`torch.no_grad()`): inputs → model → logits
  - Convert logits → probabilities (retain only class 1): `model_prob = softmax(logits, dim=1)[:, 1]`

8. **Apply threshold**
  - Convert probabilities to binary predictions:
    - `model_pred = 1 if model_prob ≥ 0.549 else 0`
  - Threshold fixed from prior tuning phase

9. **Construct evaluation dataset**
  - Combine all components into a single table:
    - `entity_type`
    - `y_true`
    - `rule_pred`
    - `model_prob`
    - `model_pred`

10. **Save output**
   - Write dataset to: `outputs/evaluation/pipeline_predictions.csv`
   - This file serves as the single source for all downstream evaluation

---

Next step (do NOT jump ahead)

Do not compute plots or stratification yet.

Immediate next step:

Create a metrics + confusion matrix script using this file.

⸻

Why this order matters

You structured Phase 4 into two layers:

1. Core system evaluation (FIRST)
	•	Overall performance
	•	Rule vs Transformer comparison
	•	Confusion matrices
	•	Precision / Recall / F1

2. Deeper analysis (SECOND)
	•	Entity-type breakdown
	•	Calibration / distributions
	•	PR / ROC curves

You must validate global behaviour first before deeper analysis.


---

## 1. Evaluation Setup

### 1.1 Ground Truth

- Source: **manually annotated test set (n = 180)**
- Each entity contains:
  - `is_valid` (ground truth label)

This is the **only unbiased dataset** used for final evaluation.

---

### 1.2 Predictions

For each test entity:

- **Rule-based prediction**
  - `SYMPTOM` → derived from `negated` field:
    - `negated = false` → valid
    - `negated = true` → invalid
  - `INTERVENTION` / `CLINICAL_CONDITION`:
    - All extracted entities assumed **valid**

- **Transformer prediction**
  - `is_valid` (model output)
  - `confidence` (probability score)

---

## 2. Comparisons

You are evaluating **two systems against ground truth**:

### 2.1 Rule-Based vs Ground Truth
- Measures baseline extraction quality

### 2.2 Transformer vs Ground Truth
- Measures final system performance

### 2.3 Improvement Analysis
- Compare rule vs transformer performance directly

---

## 3. Metrics

### 3.1 Core Metrics

For each comparison:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Definitions:

- Precision → correctness of positive predictions  
- Recall → ability to detect true positives  
- F1 → balance of precision and recall  

---

## 4. Confusion Matrix Analysis

For both rule-based and transformer:

- True Positive (TP)
- False Positive (FP)
- True Negative (TN)
- False Negative (FN)

Purpose:
- Understand **types of errors**, not just overall score

---

## 5. Improvement Analysis

Explicitly compare:

- Rule vs Transformer:
  - Δ Precision
  - Δ Recall
  - Δ F1

Expected pattern:

| Entity Type         | Expected Outcome |
|--------------------|----------------|
| SYMPTOM            | Small improvement |
| INTERVENTION       | Moderate improvement |
| CLINICAL_CONDITION | Large improvement |

---

✔ REQUIRED:
	•	Scalar metrics (precision, recall, F1)
	•	Precision–Recall curve
	•	Confusion matrix (at selected threshold)
	•	Baseline vs final comparison (selected threshold vs default threshold)

✔ STRONGLY RECOMMENDED:
	•	Calibration check
    •	Histogram of predicted probabilities
    or
    •	Reliability curve (if possible)
	•	ROC curve

Pipeline-level (critical):
	•	Rule-based vs Transformer comparison
	•	3-way framing with ground truth

Outputs:
	•	Save:
	•	y_true
	•	y_prob
	•	y_pred

Pipeline performance (THIS is your real objective)

This is what you were originally aiming for.

You are not just evaluating a model, you are evaluating a system:

Ground truth
vs
Rule-based extraction
vs
Transformer validation (final output)

“we need the model to output label + confidence”

✔ Yes. You need:

For each test sample:
	•	y_true
	•	y_prob (softmax output)
	•	y_pred (after threshold = 0.549)

⸻

Why this is necessary

Because you need to compute:

1. Transformer performance
	•	compare y_pred vs y_true

2. Rule-based extraction performance
	•	compare extraction labels vs y_true

3. Pipeline improvement
	•	compare:
	•	rule-based vs ground truth
	•	transformer vs ground truth

Overall Pipeline Comparison (CORE)

You need a single evaluation table where each row represents one candidate entity, with:
  - entity_type (e.g. symptom, intervention, condition)
	•	y_true → ground truth label (0/1)
	•	rule_pred → rule-based output (0/1)
    If entity is extracted → rule_pred = 1
    If entity is not extracted → rule_pred = 0
    If negated → rule_pred = 0
    Else → rule_pred = 1
    rule_pred = is_extracted AND NOT negated
	•	model_prob → transformer probability
	•	model_pred → transformer prediction (after threshold = 0.549)

You need:

Table: Pipeline Comparison

Metrics table + FP and FN table

You compute metrics twice, using the same y_true:

rule-based performance:

precision_score(y_true, rule_pred)
recall_score(y_true, rule_pred)
f1_score(y_true, rule_pred)

Transformer performance:

precision_score(y_true, model_pred)
recall_score(y_true, model_pred)
f1_score(y_true, model_pred)


That gives your 3-way comparison

You are not comparing systems directly to each other.

You are comparing both against ground truth.

Ground truth (reference)

→ Rule-based predictions vs ground truth
→ Transformer predictions vs ground truth

What about comparing rule-based vs transformer?

You do not compute TP/FP between them.

Instead, you interpret:

Change
Meaning
Precision ↑
fewer false positives
Recall ↓
some true positives removed

That implicitly shows improvement.


Entity-Type Stratified Performance (SECONDARY)
deeper analysis


“Where does the pipeline actually help the most?”

It is useful because:
	•	Different entity types have different linguistic structure
	•	Rule-based extraction performance will vary by type
	•	Transformer gains will likely be non-uniform


Entity Type
Rule-based
Transformer effect
Structured / explicit (e.g. symptoms)
High precision already
Small improvement
Ambiguous / contextual (e.g. interventions)
Lower precision
Larger improvement
Complex / semantic (e.g. conditions)
Variable
Depends on model understanding

table:

Entity Type
Stage
Precision
Recall
F1
SYMPTOM
Rule-based
…
…
…
SYMPTOM
Transformer
…
…
…
INTERVENTION
Rule-based
…
…
…
INTERVENTION
Transformer
…
…
…
CLINICAL_CONDITION
Rule-based
…
…
…
CLINICAL_CONDITION
Transformer
…
…
…

This lets you answer:
	•	Where is rule-based weakest?
	•	Where does transformer add most value?
	•	Is improvement consistent across entity types?

That is real insight, not just metrics.


What each table tells you

Table 1:
	•	Overall system behaviour
	•	Confirms:
	•	precision ↑
	•	recall ↓ (controlled)

⸻

Table 2:
	•	Where improvements occur
	•	Tests your hypothesis:
“Does the transformer help weaker entity types more?”


	•	Correctness is already measured in Table 1
	•	Table 2 shows variation in performance across entity types

So it answers:
	•	“Where is the model strong/weak?”
	•	“Where does the pipeline add value?”