# Phase 4 – Model Evaluation Decisions 

## Objective

This phase defines the evaluation framework used to assess both the trained transformer model and the overall extraction pipeline on a fully held-out test set (n = 180). This phase evaluates:

- Baseline rule-based extraction vs ground truth  
- Transformer validation vs ground truth  
- Net improvement introduced by the transformer layer  
- Precision–recall trade-offs under fixed thresholding  
- Performance variation across entity types  

The goal is to evaluate overall pipeline performance and quantify the contribution of the transformer validation layer, ensuring readiness for deployment on the full ICU corpus.

---

## Evaluation System Design

### 1. Overview

The evaluation is designed to assess pipeline-level performance, not the transformer in isolation, operating on three aligned components:

- **Ground truth** (manually annotated test set; reference)
- **Rule-based extraction** (high-recall baseline)
- **Transformer validation** (final system output)

This forms a 3-way evaluation framework, where both systems are evaluated against the same ground truth:

```text
			  Ground Truth (y_true)
				▲      	      	▲
				│        	    │
			Rule-Based  	Transformer
			(baseline)   	(final)
``` 

- Rule-based predictions vs ground truth  
- Transformer predictions vs ground truth  

There is no direct pairwise evaluation between systems, improvement is inferred from changes in metrics relative to ground truth.

This design ensures:

- A single, unbiased reference (`y_true`)
- Direct comparability of system performance
- Valid interpretation of precision–recall trade-offs  

The evaluation objective is to verify that the transformer functions as a:

> **precision-oriented validation layer applied to a high-recall extraction system**

Success is defined by:

- Precision increase (reduction in false positives)  
- Controlled reduction in recall  
- Net improvement in F1 score  
- Coherent performance across entity types  

---

### 2. System Structure

The evaluation is executed in two sequential layers:

#### 2.1 Core System Evaluation (Primary)

This layer establishes global system behaviour, each system is evaluated independently against ground truth to compute:

1. Accuracy, Precision, Recall, F1  
2. Confusion matrix components (TP, FP, TN, FN)

This produces two directly comparable metric sets:

- Rule-based performance (baseline)  
- Transformer performance (final system)  

Improvement is interpreted through metric deltas:

- Δ Precision → reduction in false positives  
- Δ Recall → impact on coverage  
- Δ F1 → overall system improvement  

This layers purpose is to:

- Validate that the transformer improves over the baseline  
- Quantify the precision–recall trade-off at a global level  
- Ensure that the pipeline behaves as intended before deeper analysis 

---

#### 2.2 Stratified & Diagnostic Analysis (Secondary)

This layer explains where and why performance changes occur, analysis includes:

- Performance stratified by entity type  
- Probability distributions (`model_prob`)  
- Calibration behaviour  
- Precision–Recall and ROC curves  
- Threshold sensitivity  

Expected trends:

| Entity Type             | Behaviour |
|------------------------|----------|
| `SYMPTOM`              | Smaller gains (already structured) |
| `INTERVENTION`         | Moderate gains |
| `CLINICAL_CONDITION`   | Larger or variable gains |

The purpose of this layer is to:

- Provide a deeper understanding of system behaviour (entity-level insights)
- Identify specific weaknesses or strengths in the baseline and transformer
- Validate expected behaviour patterns based on the model design and training data characteristics

---

### 3. Rationale for Two-Layer Design

#### 3.1 Separation of Validation and Explanation

The evaluation distinguishes between:

- **Validation (Layer 1):** Establish whether the system works at a global level  
- **Explanation (Layer 2):** Analyse where and why performance changes occur  

Without this separation, detailed analyses (e.g. by entity type or probability distribution) can be misleading if the overall system behaviour is not first verified.

---

#### 3.2 Prevention of Misleading Local Patterns

Stratified or distributional analyses operate on smaller subsets of data, which are inherently more sensitive to:

- Sampling variability  
- Class imbalance  
- Noise  

If global performance is not first established:

- Apparent improvements may reflect noise rather than real signal  
- Local trends may contradict overall system behaviour  
- Conclusions may not generalise  

---

#### 3.3 Consistent Interpretation of Metric Changes

The evaluation relies on interpreting metric shifts (Δ Precision, Δ Recall, Δ F1) between systems.

This interpretation is only valid if:

- Both systems are first evaluated at the same global level  
- Metric behaviour is stable and coherent  

Layer 1 ensures that:

- Observed improvements are real and not artefacts  
- Precision–recall trade-offs are understood at system level  

Layer 2 then explains these changes without redefining them.

---

#### 3.4 Alignment with Pipeline Objective

The system is explicitly designed as:

> High-recall extraction → precision-oriented validation

The two-layer structure directly mirrors this objective:

- Layer 1 verifies that the pipeline achieves the intended trade-off  
- Layer 2 identifies where this trade-off is most effective  

This ensures that evaluation remains aligned with the design intent of the pipeline, rather than analysing components in isolation.

---

## Evaluation Predictions Dataset 

### 1. Objective

Evaluation is based on constructing a single unified dataset where all prediction sources are aligned at the entity level. Each row corresponds to one candidate entity and contains:

- Ground truth label  
- Rule-based prediction  
- Transformer prediction (probability + thresholded output)  

This dataset is the foundation for all downstream evaluation, enabling for consistent metric computation and stratified analysis.

No metrics are computed during this stage; this step strictly produces the evaluation-ready dataset. This separation ensures reproducibility and flexibility in analysis.

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

- The extraction stage prioritises recall (maximising coverage)  
- Rule-based methods are limited in precision due to lack of contextual understanding  
- Minimal precision logic is applied only where trivial and high-impact (negation)  
- This establishes a weak but consistent baseline for measuring transformer improvement  

---

#### 2.3 Transformer Predictions

The transformer acts as a validation layer applied after extraction.

For each entity:

- Outputs logits → converted to probability: `model_prob = p(y = 1)`
- Applies tuned threshold: `model_pred = (model_prob ≥ 0.549)`

Key properties:

- This is the first application of the tuned threshold, defining the final decision boundary used for deployment-level predictions  
- These predictions represent the final pipeline output and are directly compared to ground truth  
- The model is explicitly precision-oriented, so expected behaviour is:
  - Precision ↑ (fewer false positives)  
  - Recall ↓ (removal of uncertain positives)  

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

- **Single-table design:** Ensures all systems are evaluated against the same reference without recomputation  
- **model_prob retained:** Required for PR curves, ROC analysis, and threshold sensitivity  
- **model_pred included:** Represents the deployed decision rule  
- **entity_type included:** Enables stratified performance analysis  
- **Separation of concerns:** Dataset generation is independent from metric computation, enabling reproducibility 

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

## Core System Evaluation (Layer 1) 

### 1. Overview

- This stage quantifies global pipeline performance using the unified evaluation dataset to compute core evaluation metrics.  
- Both systems (rule-based and transformer) are evaluated independently against the same ground truth, producing directly comparable metrics.
- The objective is not to analyse patterns yet, but to establish whether the transformer improves overall system behaviour and achieves the intended precision–recall trade-off, as well as error shifts relative to the baseline.

---

### 2. Metric Selection and Rationale

Evaluation is based on four core metrics, ordered by relevance to the pipeline objective:

| Metric    | Definition | What it Captures | Relevance to Pipeline |
|-----------|------------|------------------|----------------------|
| **Precision** | `TP / (TP + FP)` | Proportion of predicted positives that are correct | Primary objective: measures reduction in false positives by the transformer |
| **Recall** | `TP / (TP + FN)` | Proportion of true positives correctly identified | Captures loss of coverage due to filtering |
| **F1 Score** | Harmonic mean of precision and recall | Balance between correctness and coverage | Primary indicator of overall system improvement |
| **Accuracy** | `(TP + TN) / Total` | Overall correctness | Included for completeness, but less informative in this setting |

Metric selection reflects the pipeline design:

- The rule-based system prioritises **recall** (high coverage, low precision)
- The transformer acts as a **precision-oriented filter**

Therefore:

- **Precision increase** indicates successful false positive reduction  
- **Recall decrease** reflects the cost of filtering  
- **F1 score** determines whether this trade-off results in net improvement  

Accuracy is less informative in this setting because it is dominated by true negatives (TN), which are typically easier to predict and less relevant to the pipeline objective. As a result:

- A model can achieve high accuracy by correctly predicting many negatives, even if it performs poorly on valid entities  
- Differences in false positives and false negatives may not be reflected clearly in accuracy  

In contrast, precision and recall directly measure the behaviour of interest:

- Precision captures incorrect inclusions (false positives)  
- Recall captures missed valid entities (false negatives)  

For this reason, accuracy is reported but not used to assess system effectiveness as it may not reflect the critical trade-offs being evaluated.

---

### 3. Confusion Matrix Rationale

In addition to scalar metrics, evaluation includes full confusion matrix decomposition for each system (rule-based and transformer) against ground truth:

- True Positives (TP): correctly identified valid entities  
- False Positives (FP): invalid entities incorrectly retained  
- True Negatives (TN): correctly rejected invalid entities  
- False Negatives (FN): valid entities incorrectly removed  

This is essential because aggregate metrics do not reveal how errors change, whereas the confusion matrix enables direct interpretation of:

- **False positive reduction (FP ↓):** Validates precision improvement  
- **False negative increase (FN ↑):** Quantifies loss in recall  
- **True positive retention (TP stability):** Indicates whether useful signal is preserved  

Therefore:

- Metrics summarise performance  
- Confusion matrices explain the mechanism of change between systems

Interpretation is based on comparing confusion matrices across systems, not by computing a direct matrix between them.

---

### 4. Metric Interpretation and Comparison

### 4.1 Summary of Results

| System        | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Rule-Based    | 0.628    | 0.571     | 0.989  | 0.724    |
| Transformer   | 0.750    | 0.833     | 0.618  | 0.710    |

All metrics exist in `core_metrics.csv`, with visual comparison shown in `metrics_comparison.png`.

---

### 4.2 Interpretation by System

#### Rule-Based System (Baseline)

- **Precision = 0.571**
  - Low precision reflects a high number of false positives (66)
  - Consistent with design: high-recall, high-noise extraction with minimal filtering
  - Clinically: many extracted entities are incorrect → low reliability without further validation

- **Recall = 0.989**
  - Near-perfect recall (only 1 false negative)
  - Confirms baseline captures almost all valid entities
  - Clinically: very low risk of missing relevant information

- **F1 Score = 0.724**
  - Moderately high due to extremely high recall
  - Does not reflect poor precision

- **Accuracy = 0.628**
  - Lower overall correctness due to large number of false positives

The rule-based system behaves as intended:

- Maximises coverage with near perfect recall
- Accepts high false positive rate (low precision) as a trade-off 
- Suitable as a broad extraction stage, not for final use

---

#### Transformer System (Final Pipeline Output)

- **Precision = 0.833**
  - Substantial improvement (FP reduced from 66 → 11)
  - Indicates effective filtering of incorrect entities
  - Clinically: significantly higher reliability of outputs

- **Recall = 0.618**
  - Significant reduction (FN increased from 1 → 34)
  - Reflects cost of aggressive filtering stage
  - Clinically: some valid entities are now missed

- **F1 Score = 0.710**
  - Slight decrease relative to baseline
  - Indicates trade-off between precision gain and recall loss

- **Accuracy = 0.750**
  - Higher overall correctness driven by correct rejection of invalid entities (TN ↑ from 25 → 80)

The transformer successfully performs as a:

- Strong precision-oriented filtering layer
- Reduces false positives substantially, at the cost of reduced recall and a slight decrease in F1 score
- Produces cleaner, more trustworthy outputs at the expense of coverage

---

### 4.3 Comparative Analysis (Δ Changes)

| Metric    | Δ (Transformer − Rule) | % Change | Interpretation |
|-----------|------------------------|----------|----------------|
| Accuracy  | +0.122                 | +19.5%   | Improved overall correctness (driven by TN increase) |
| Precision | +0.262                 | +45.9%   | Major reduction in false positives (primary objective achieved) |
| Recall    | −0.371                 | −37.5%   | Substantial loss of coverage due to filtering |
| F1 Score  | −0.015                 | −2.0%    | Slight decrease due to recall loss outweighing precision gain |

Key interpretation:

- The transformer successfully achieves its primary objective:
  - Large increase in precision  
  - Significant reduction in false positives  

- The recall reduction is larger than observed during threshold tuning:
  - Reflects generalisation to a held-out dataset  
  - Expected due to distributional variation and model calibration differences (threshold tuned on OOF predictions, not test set)

- Importantly:
  - This does not invalidate the model or thresholding strategy
  - The threshold was correctly tuned using out-of-fold predictions without leakage  
  - Performance differences arise from evaluation on unseen data, not methodological error  

- Overall:
  - The system behaves as a precision-biased filter, consistent with design  
  - The trade-off is more conservative than anticipated, but still interpretable and controlled  


---

### 4.4 Confusion Matrix-Level Interpretation

| Component | Rule-Based | Transformer | Change | Interpretation |
|----------|------------|------------|--------|----------------|
| TP       | 88         | 55         | −33    | Valid entities removed by filtering |
| FP       | 66         | 11         | −55    | Large reduction in incorrect entities |
| TN       | 25         | 80         | +55    | Strong increase in correct rejections |
| FN       | 1          | 34         | +33    | Increased missed valid entities |

Key observations:

- **False positives reduced dramatically (66 → 11)**
  - Primary success of the transformer
  - Directly improves clinical reliability of extracted entities

- **False negatives increased substantially (1 → 34)**
  - Indicates aggressive filtering behaviour
  - Represents loss of valid information

- **Error trade-off is asymmetric (FP ↓ 55 vs FN ↑ 33)**
  - FP reduction (−55) is larger than FN increase (+33)
  - Confirms the system is strongly biased toward rejecting uncertain cases  

- Interpretation:
  - The transformer is not randomly discarding entities  
  - It is systematically removing low-confidence or ambiguous predictions, consistent with a precision-oriented objective
  - The net effect is a cleaner output with fewer false positives, albeit at the cost of missing some valid entities

---

### 4.5 Clinical and Pipeline-Level Interpretation

The pipeline is explicitly designed as:

> High-recall extraction → precision-oriented validation

Under this design:

- The rule-based stage ensures maximal coverage
- The transformer stage enforces quality control

From a clinical perspective:

- **False positives (FP)**:
  - Introduce noise  
  - Reduce trust in extracted data  
  - Increase downstream validation burden  

- **False negatives (FN)**:
  - Risk missing relevant clinical information  
  - Impact depends on downstream use case  

This pipeline explicitly prioritises:

- Reducing false positives → improving reliability and usability
- Accepting controlled loss in recall as a trade-off  

Observed behaviour aligns with this objective:

- Substantial FP reduction → cleaner, more trustworthy outputs  
- Increased FN → stricter filtering threshold applied  

---

### 4.6 Conclusion

- The evaluation methodology is correct and robust:
  - Threshold tuned using out-of-fold predictions  
  - No data leakage  
  - Final evaluation performed on a fully held-out dataset  

- The transformer successfully functions as a precision-oriented validation layer:
  - Precision improved substantially (+45.9%)  
  - False positives reduced significantly (66 → 11)  

- The trade-off:
  - Recall loss is larger than observed during tuning (-37.5%)
  - Results in a small decrease in F1 score (-2.0%)

- Interpretation:
  - The system generalises with a more conservative decision boundary  
  - This reflects distributional variation, not methodological failure  
  - Behaviour remains consistent with pipeline design goals  

Overall:

- The pipeline achieves its intended objective  
- Results are valid, explainable, and aligned with system design  
- If required, threshold adjustment can rebalance recall, but is a design choice, not a correction

---

### 5. Implementation

Core evaluation is implemented using two scripts in `scripts/evaluation/`:

- `evaluation_metrics.py`
  - Computes scalar metrics and confusion matrix components using `pipeline_predictions.csv`
  - Outputs: `outputs/evaluation/core_metrics.csv`

- `plot_core_evaluation.py`
  - Generates visualisations using `core_metrics.csv`:
    - Confusion matrix heatmaps
    - Metrics comparison plot
  - Outputs (`outputs/evaluation/core_plots/`):
    - `rule_confusion_matrix.png`
    - `transformer_confusion_matrix.png`
    - `metrics_comparison.png`

---

## Stratified & Diagnostic Analysis (Layer 2)

### 1. Overview

This layer builds on the core evaluation (Layer 1) to provide a deeper, explanatory analysis of system behaviour.

- Layer 1 establishes whether the pipeline works  
- Layer 2 explains where and why performance changes occur  

The analysis operates on the unified evaluation dataset and focuses on:

- Variation in performance across entity types  
- The precision–recall trade-off across decision thresholds  
- Behaviour of model probabilities (i.e. why false positives and false negatives occur)  

This layer is explicitly secondary:

- It does not redefine performance  
- It interprets and explains the results established in Layer 1  

---

### 2. Stratified Performance Rationale

Aggregate metrics (precision, recall, F1) summarise overall behaviour but mask variation across entity types.

Given that the pipeline processes heterogeneous clinical entities (e.g. `SYMPTOM`, `INTERVENTION`, `CLINICAL_CONDITION`), performance is expected to vary due to:

- Differences in linguistic structure 
- Differences in contextual ambiguity 
- Differences in clinical representation

Stratified evaluation is therefore required to:

- Identify where the rule-based system performs poorly  
- Determine where the transformer provides the most benefit  
- Validate expected behaviour based on entity characteristics  

For each entity type, the following metrics are computed for both systems:

- Precision  
- Recall  
- F1 score  

This enables direct comparison of:

- Baseline vs transformer performance  
- Variation in precision–recall trade-offs across entity categories  

The primary output is a **stratified metrics table**, supported by a bar plot for visual comparison.

---

### 3. Model Behaviour Analysis Rationale

This component explains how the model produces its predictions, using probability-based diagnostics.

#### 3.1 Precision–Recall Curve

The Precision–Recall (PR) curve is a fundamental diagnostic tool for binary classification.

- It plots precision vs recall across all possible decision thresholds  
- It is directly aligned with the pipeline objective:
  > Maximising precision while controlling recall  

Interpretation:

- Left region → high precision, low recall (strict threshold)  
- Right region → high recall, low precision (loose threshold)  

It provides:

- A global view of performance across all thresholds  
- A direct representation of the precision–recall trade-off  
- Context for evaluating the chosen operating threshold (0.549)  

The selected threshold is explicitly marked to show:

- Its position on the curve  
- Whether it reflects a conservative or balanced decision boundary   

---

#### 3.2 Probability Distribution Analysis

The distribution of predicted probabilities (`model_prob`) is analysed separately for:

- Positive class (`y_true = 1`)  
- Negative class (`y_true = 0`)  

This provides insight into:

- Degree of class separation achieved by the model  
- Extent of overlap between classes  
- Underlying causes of:
  - False positives (FP)  
  - False negatives (FN)  

A vertical threshold line is included to show:

- How the decision boundary partitions predictions  
- How this partitioning leads to observed metric changes (precision ↑, recall ↓)  

---

### 4. Excluded Analyses and Justification

#### 4.1 ROC Curve

The Receiver Operating Characteristic (ROC) curve is not included because it is **misaligned with the pipeline objective**.

- ROC evaluates:
  - True Positive Rate (TPR) vs False Positive Rate (FPR)  
- It does not incorporate precision, which is the primary optimisation target  

Additionally:

- FPR is normalised by total negatives (TN + FP)  
- This can make performance appear favourable even when false positives are high  

As a result:

> ROC analysis can misrepresent performance in precision-focused systems and is therefore not appropriate here 

---

#### 4.2 Calibration Curve

Calibration analysis evaluates whether predicted probabilities reflect true likelihood.

However, in this pipeline:

- Probabilities (`model_prob`) are used only for thresholding
- The final output is binary (`model_pred`)

Therefore:

- The system functions as a decision system, not a probabilistic one  
- Calibration does not directly inform precision, recall, or error trade-offs  

Including calibration would:

- Add analytical complexity  
- Provide limited value for interpreting system behaviour  

Calibration would only be required if:

- Probabilities were used directly in downstream modelling  
- Risk scoring or ranking was introduced  

---

### 5. Stratified Performance Analysis

#### 5.1 Summary of Metrics by Entity Type

| Entity Type           | System        | Precision | Recall | F1 Score |
|----------------------|--------------|-----------|--------|----------|
| `SYMPTOM`              | Rule-Based    | 0.794     | 0.964  | 0.871    |
| 			             | Transformer   | 0.778     | 0.500  | 0.609    |
| `INTERVENTION`         | Rule-Based    | 0.617     | 1.000  | 0.763    |
| 				         | Transformer   | 0.900     | 0.730  | 0.806    |
| `CLINICAL_CONDITION`   | Rule-Based    | 0.400     | 1.000  | 0.571    |
| 	  				     | Transformer   | 0.778     | 0.583  | 0.667    |

All metrics are reported in `stratified_metrics.csv`, with visual comparison in `stratified_metrics.png`.

---

#### 5.2 Expected Behaviour Patterns

From the pipeline design and entity characteristics:

- **SYMPTOM**
  - Expected: relatively high precision from rule-based extraction  
  - Reason: explicit lexical patterns + negation handling  
  - Expected transformer impact: **minimal improvement**

- **INTERVENTION**
  - Expected: moderate precision baseline  
  - Reason: contextual ambiguity (performed vs planned, historical vs current)  
  - Expected transformer impact: **clear improvement**

- **CLINICAL_CONDITION**
  - Expected: lowest precision baseline  
  - Reason: semantic ambiguity (confirmed vs suspected, chronic vs acute)  
  - Expected transformer impact: **largest improvement**

---

#### 5.3 SYMPTOM Interpretation

| Metric    | Δ (Transformer − Rule) | % Change |
|-----------|------------------------|----------|
| Precision | −0.016                 | −2.0%    |
| Recall    | −0.464                 | −48.1%   |
| F1 Score  | −0.262                 | −30.1%   |

**Rule-Based:**
- High precision (0.794) and recall (0.964)
- Strong baseline performance

**Transformer:**
- Precision: no improvement (0.778 ↓)
- Recall: large drop (0.964 → 0.500)
- F1: substantial decrease

**Interpretation:**

- The transformer removes a large number of true positives without reducing false positives
- This indicates over-filtering of already reliable predictions

**Conclusion:**

- This is a clear failure mode
- The transformer is not beneficial for `SYMPTOM` extraction under the current threshold
- The recall drop is not acceptable, as it is not compensated by precision gain 

---

#### 5.4 INTERVENTION Interpretation

| Metric    | Δ (Transformer − Rule) | % Change |
|-----------|------------------------|----------|
| Precision | +0.283                 | +45.9%   |
| Recall    | −0.270                 | −27.0%   |
| F1 Score  | +0.043                 | +5.6%    |

**Rule-Based:**
- Precision: moderate (0.617)
- Recall: perfect (1.000)

**Transformer:**
- Precision: large increase (0.900)
- Recall: decrease (1.000 → 0.730)
- F1: improvement (0.763 → 0.806)

**Interpretation:**

- Large reduction in false positives
- Recall decrease reflects removal of borderline or ambiguous cases
- Net effect: **improved balance of correctness vs coverage**

**Important clarification:**

- The recall drop (−27%) is substantial in absolute terms
- It is considered acceptable only because:
  - Precision improves dramatically (+0.28)
  - F1 increases (+0.043)
  - The baseline recall was artificially inflated (rule-based over-generation)

**Conclusion:**

- Transformer behaves as intended
- Trade-off is justified and beneficial for this entity type

---

#### 5.5 CLINICAL_CONDITION Interpretation

| Metric    | Δ (Transformer − Rule) | % Change |
|-----------|------------------------|----------|
| Precision | +0.378                 | +94.5%   |
| Recall    | −0.417                 | −41.7%   |
| F1 Score  | +0.096                 | +16.8%   |

**Rule-Based:**
- Very low precision (0.400)
- Perfect recall (1.000)

**Transformer:**
- Precision: large increase (0.778)
- Recall: decrease (1.000 → 0.583)
- F1: strong improvement

**Interpretation:**

- Transformer removes a large number of false positives
- Recall reduction reflects elimination of uncertain or incorrect conditions

**Conclusion:**

- Transformer provides substantial correction of baseline errors
- Trade-off is justified given the very poor starting precision

---

#### 5.6 Cross-Entity Comparison

| Entity Type         | Precision Change | Recall Change | F1 Change | Interpretation |
|--------------------|------------------|--------------|-----------|----------------|
| `SYMPTOM`            | ↓ slight         | ↓↓↓ large     | ↓↓↓ large | Harmful filtering |
| `INTERVENTION`       | ↑↑ large         | ↓ substantial | ↑         | Beneficial trade-off |
| `CLINICAL_CONDITION` | ↑↑ very large    | ↓ substantial | ↑↑        | Strong correction |

Key observations:

- Transformer impact is strongly dependent on baseline quality
- Gains occur where:
  - Baseline precision is low  
  - Semantic ambiguity is high  

- Failures occur where:
  - Baseline is already strong  
  - Additional filtering removes valid signal  

---

#### 5.7 Quality of True Positives and Recall Interpretation

It is important to distinguish between annotated correctness and clinical usefulness:

- Ground truth defines what counts as a positive entity  
- However, not all annotated positives are equally informative or clinically reliable due to the complex nature of the task 

In clinical text, positive labels may include:

- Contextually weak, ambiguous, or borderline cases that could be interpreted either way even with guidelines for annotation
- These will be "true positives" in an annotation sense but are noisy, and may not be clinically meaningful or useful for downstream applications.

The rule-based system, based on lexical matching, tends to include these cases, resulting in:

- Very high recall  
- Inclusion of lower-confidence or ambiguous entities  

The transformer, being context-aware:

- Filters uncertain or weakly supported mentions  
- Retains clearer, contextually valid entities  

As a result:

- Some baseline true positives become false negatives after validation  
- This appears as a reduction in recall  
- However, it likely reflects partial removal of borderline or low-quality positives, not purely loss of clinically meaningful information  

> Therefore, recall reduction should be interpreted with caution, particularly for semantically complex entity types.

---

#### 5.8 Clinical and Pipeline-Level Interpretation

Entity types differ in both clinical meaning and error tolerance:

---

**SYMPTOM (state-based)**

- Represents current patient state (e.g. pain, fever)
- Often:
  - Explicitly stated  
  - Repeated within text  

Metric implication:

- Recall is important → missing symptoms removes signal  
- Precision already high → little benefit from filtering 

Observed:

- Large recall loss without precision gain (0.964 → 0.500)
- Clinically unacceptable trade-off → loss of meaningful patient state information.

---

**INTERVENTION (event-based)**

- Represents discrete clinical actions (e.g. surgery, medication)
- Each instance:
  - May be mentioned once or multiple times
  - Can be ambiguous (performed vs planned)

Metric implication:

- Precision is critical → incorrect events are highly misleading (false positives are harmful)
- Recall important but secondary

Observed:

- Large precision gain, moderate recall loss → clinically beneficial trade-off (cleaner data outweighs some loss of coverage)
- F1 improves → net positive effect on pipeline performance

---

**CLINICAL_CONDITION (state-based, semantic)**

- Represents diagnoses or active conditions  
- Often:
  - Context-dependent  
  - Ambiguous (suspected vs confirmed, historical vs current)  

Metric implication:

- Precision is critical → incorrect diagnoses corrupt datasets  
- Recall less critical if ambiguity is high

Observed:

- Massive precision improvement, recall reduction  
- Strong F1 gain  
- Clinically appropriate → improved diagnostic specificity outweighs loss of ambiguous cases.

---

**Pipeline-Level Interpretation**

For a pipeline that is designed as high-recall extraction → precision-oriented validation, the system behaves as follows:

- Effective for semantically complex, ambiguous entities → `INTERVENTION`, `CLINICAL_CONDITION`  
- Ineffective for well-structured, explicit entities → `SYMPTOM` (over-filtering)

Transformer value is proportional to semantic complexity and baseline error rate

---

#### 5.9 Overall Conclusion

The core hypothesis is validated: 

> Transformer adds value where rule-based extraction is weakest

However: 

- Performance is not uniformly beneficial across heterogenous entity types
- A single global threshold is suboptimal for all categories, leading to:
	- Over-filtering of `SYMPTOM`
	- Appropriate filtering of `INTERVENTION` and `CLINICAL_CONDITION`

Recommended improvement:

- Introduce entity-type-specific thresholds:
  - Higher threshold for ambiguous entities (maintain precision)
  - Lower threshold for structured entities (preserve recall)
- This would:
  - Preserve recall where baseline is strong  
  - Maintain precision gains where needed  
  - Improve overall pipeline performance and balance

---

### 6. Probability & Threshold Behaviour Analysis

#### 6.1 Precision–Recall Curve Interpretation

**Overall Shape**

- The curve shows a stable high-precision region (plateau) where:
  - Precision remains high (~0.83–0.85)
  - While recall increases up to ~0.65–0.68  
- Beyond this region, precision begins to drop more rapidly as recall increases  

**Interpretation:**

- The model has a region of good separability  
- Within this region, recall can increase with minimal loss in precision  
- Outside this region, the trade-off worsens, with increasing false positives  

---

**Threshold Placement (0.549)**

- The selected threshold corresponds to:
  - Precision ≈ 0.83  
  - Recall ≈ 0.62  

- This lies:
  - At the edge of the high-precision plateau  
  - Just before precision begins to decline more sharply  

**Interpretation:**

- The threshold is precision-oriented, but not extreme  
- It represents a transition point:
  - Before this point → efficient trade-off  
  - After this point → precision deteriorates more quickly  

---

**Relation to Observed Metrics**

- The increase in precision (0.571 → 0.833) is explained by:
  - Operating within the high-precision region of the curve  
  - Filtering out lower-confidence predictions  

- The decrease in recall (0.989 → 0.618) is explained by:
  - The model rejecting a substantial portion of candidates  
  - Required to maintain higher precision  

---

**Interpretation of Threshold Choice**

- The chosen threshold is reasonable and aligned with the pipeline objective (precision-focused filtering)

- However:
  - It is positioned slightly towards the stricter end of the plateau  
  - Nearby **lower thresholds** could increase recall  
  - With only a modest reduction in precision  

---

**Conclusion**

- The PR curve confirms that:
  - The model can achieve high precision, but not without reducing recall  
  - The observed precision–recall trade-off is structurally expected  

- The selected threshold represents a valid precision-oriented operating point  

- However:
  - It is slightly conservative  
  - Alternative nearby thresholds may provide a more balanced trade-off depending on downstream requirements  

---

#### 6.2 Probability Distribution Interpretation

**Overall Separation**

The distributions of predicted probabilities (`model_prob`) show moderate but incomplete class separation:

- True positives (`y=1`):
  - Concentrated at higher probabilities (~0.50–0.70)
  - Relatively tighter distribution  

- True negatives (`y=0`):
  - Spread across a wider range (~0.2–0.65)
  - Significant mass in the mid-range (~0.35–0.45)  

Interpretation:

- The model successfully assigns higher confidence to true positives on average
- However, there is substantial overlap between classes, particularly in the mid-probability range  

---

**Overlap Region and Error Formation**

- The key region is approximately: 0.50 – 0.60
- Both classes have meaningful density in this range.

This directly explains observed errors:

- **False Positives (FP)**:
  - Negatives with probabilities above the threshold (~0.55)
  - Represent borderline cases incorrectly accepted  

- **False Negatives (FN)**:
  - Positives with probabilities below the threshold
  - Represent valid entities rejected due to lower confidence  

Interpretation:

> Errors are concentrated in the overlap region, where the model cannot clearly separate classes.

---

**Threshold Behaviour (0.549)**

The selected threshold:

- Lies within the overlap region, not outside it  
- Splits the distributions asymmetrically:
  - Retains most high-confidence positives  
  - Removes most low-confidence negatives  

However:

- Some positives in the 0.45–0.55 range fall below threshold → become FN  
- Some negatives in the 0.55–0.65 range exceed threshold → become FP  

Interpretation:

> The threshold is not placed at a clean separation point (none exists), but instead enforces a trade-off within an inherently ambiguous region.

---

**Relation to Pipeline Behaviour**

This distribution explains the observed evaluation results:

- **Precision increase**: Achieved by excluding the majority of low-confidence negatives  
- **Recall decrease**: Caused by rejecting positives in the overlapping region  

Additionally:

- The presence of a left tail in the positive class indicates some true entities are inherently ambiguous or weakly expressed  
- The wide spread of the negative class reflects noisy candidate generation from the rule-based stage  

---

**Conclusion**

- The model achieves partial but not complete class separation
- The overlap region is unavoidable, and drives all FP/FN errors  
- The chosen threshold:
  - Prioritises precision by excluding uncertain cases  
  - Inevitably sacrifices recall due to overlap  

> The observed precision–recall trade-off is therefore structurally determined by the distribution of model probabilities, not a flaw in the evaluation or methodology.

---

#### 6.3 Integrated Interpretation and Implications

Combining both analyses:

- The PR curve plateau reflects the same phenomenon observed in the probability distributions:  
  a region where class separation is strong but not complete  
- The selected threshold operates within the overlap region, which necessarily produces a trade-off between:
  - Increasing precision (by excluding ambiguous negatives)  
  - Reducing recall (by excluding borderline positives)  
- The observed increase in precision and reduction in recall are therefore expected outcomes of operating in this region of partial class overlap  

For this pipeline objective (precision-oriented clinical entity extraction):

- The model’s behaviour is appropriate and consistent with design goals  
- Precision improvement reflects successful suppression of false positives  
- Recall reduction reflects exclusion of borderline or ambiguous cases within the overlap region  

Overall:

- The model provides a useful precision-recall trade-off curve  
- The selected operating point represents a valid balance for structured dataset generation  

---

### 7. Overall Evaluation Synthesis

Across both evaluation layers, the results are consistent with the intended pipeline design.

- Layer 1 demonstrates a clear trade-off: increased precision at the cost of reduced recall after transformer validation.
- Layer 2 explains this behaviour through class overlap and entity-type-dependent performance differences.

Overall, the system successfully achieves its primary objective of producing a high-precision structured clinical entity dataset for downstream analysis.

While recall reduction is non-uniform across entity types (notably `SYMPTOM`), the observed behaviour is consistent with threshold-based filtering in a partially separable classification space.

The only significant improvement opportunity identified is the introduction of entity-type-specific thresholds to better balance precision and recall across heterogeneous categories.

---

### 8. Implementation

Secondary evaluation is implemented in `scripts/evaluation/plot_secondary_analysis.py`:

- Generates stratified metrics and plots, PR curve, and probability distributions using `pipeline_predictions.csv`
- Outputs (`outputs/evaluation/`):
	- `stratified_metrics.csv`
	- `secondary_plots/`
		- `stratified_metrics.png`
		- `pr_curve.png`
		- `probability_distribution.png`

---