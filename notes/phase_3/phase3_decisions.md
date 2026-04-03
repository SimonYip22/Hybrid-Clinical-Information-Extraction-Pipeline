# Phase 3 - Transformer Contextual Validation Decisions

## Objective

- This document outlines the key design decisions for Phase 3: transformer-based validation of extracted clinical entities.
- Phase 3 builds on Phase 2, where deterministic rules generate high-recall, span-aligned candidate entities.
- The transformer layer converts these candidates into contextually valid, task-specific outputs via semantic classification.
- The goal is a scalable, reproducible, and clinically meaningful validation layer.

---

## Model Selection

### 1. Objective

- Select a model capable of sentence-level, context-aware classification of extracted entities
- Ensure compatibility with Phase 2 schema (`sentence_text`, `entity_text`)
- Balance:
  - Clinical performance  
  - Computational efficiency  
  - Scalability  
  - Reproducibility    

This serves as the formal reproducibility specification for Phase 3 model selection.

---

### 2. Design Context (From Phase 2)

Phase 2 produces:

- High-recall candidate entities:
  - `SYMPTOM`
  - `INTERVENTION`
  - `CLINICAL_CONDITION`
- Structured schema with:
  - Span (`entity_text`)
  - Context (`sentence_text`)
  - Validation placeholders (`is_valid`, `confidence`, `task`)

Input–Output Requirements:

- **Input:** `(sentence_text, entity_text)`  
- **Output:**
  - `is_valid` → binary classification  
  - `confidence` → probability score  
  - `task` → entity-specific interpretation  

The task is entity-level contextual classification, not sequence labelling or text generation

---

### 3. Nature of the Task (Critical Clarification)

The task requires contextual reasoning specific to each entity type:

| Entity Type         | Required Reasoning |
|--------------------|------------------|
| `SYMPTOM`           | Negation (present vs absent) |
| `INTERVENTION`      | Intent vs execution (planned vs performed) |
| `CLINICAL_CONDITION`| Temporality, certainty, diagnostic status |

Examples:
- “no chest pain” → symptom = **invalid (negated)**
- “plan to start antibiotics” → intervention = **invalid (not performed)**
- “history of MI” → condition = **invalid (not active)**

Key Requirement:
- Task is to interpret meaning of entity within full sentence context
- Be able to handle ICU-specific language:
  - Clinical shorthand
  - Multi-clause sentences
  - Ambiguity and implicit meaning
- With constraints of reproducibility, efficiency, and structured outputs

---

### 4. Approaches Considered

#### 4.1 Model Types

| Approach                     | Strengths                              | Limitations |
|-----------------------------|----------------------------------------|-------------|
| Rule-based systems          | Interpretable, deterministic           | Cannot model complex context, brittle to variation |
| Classical ML (e.g. LR, SVM) | Simple, efficient                    | Requires manual feature engineering, weak semantic understanding |
| CNNs (text classification)  | Capture local n-gram patterns          | Limited long-range context, weak for negation/temporal scope |
| RNNs / LSTMs                | Sequential modelling                   | Struggle with long-range dependencies, less efficient than transformers |
| Transformers (BERT-style)   | Full-context attention, strong semantics | Higher computational cost |

---

#### 4.2 Key Limitations of Non-Transformer Models

**1. Classical Machine Learning (e.g. Logistic Regression, SVM)**

These models rely on explicit feature engineering, typically using:
- Bag-of-words / TF-IDF representations  
- Manually engineered indicators (e.g. negation flags, keyword presence)

Limitations:
- No inherent understanding of word order or context  
- Inability to model relationships between tokens within a sentence  
- Heavy reliance on manually defined features such as:
  - Negation rules (“no”, “denies”)  
  - Temporal indicators (“history of”, “previous”)  

Implication:
- Reintroduces the same brittleness as rule-based systems  
- Poor generalisation to:
  - Unseen phrasing  
  - Clinical shorthand  
  - Complex sentence structures  


**2. Convolutional Neural Networks (CNNs for Text)**

CNNs operate by learning local n-gram patterns through sliding convolutional filters.

Strengths:
- Effective at capturing short phrases (e.g. “chest pain”, “respiratory failure”)  
- Computationally efficient  

Limitations:
- Restricted receptive field (limited context window)  
- Weak modelling of long-range dependencies  

Failure cases:
- Negation scope: “no evidence of chest pain”  
- Multi-clause sentences: “patient denies chest pain but reports shortness of breath”  

Implication:
- Cannot reliably determine whether an entity is valid within full sentence context  


**3. Recurrent Neural Networks (RNNs / LSTMs)**

RNN-based models process text sequentially, maintaining a hidden state across tokens.

Strengths:
- Designed for sequential data  
- Can, in theory, capture contextual flow  

Limitations:
- Difficulty capturing long-range dependencies:
  - Early tokens (e.g. negation) may not effectively influence later tokens  
- Gradient degradation over long sequences (even with LSTM/GRU variants)  
- Limited parallelisation → slower training and inference  

Failure cases:
- Long clinical sentences with multiple clauses  
- Context-dependent meaning spanning distant tokens  

Implication:
- Inconsistent performance for:
  - Negation handling  
  - Temporal reasoning  
  - Complex clinical phrasing 

---

#### 4.3 Conclusion

Across all non-transformer approaches:

- Context is either not modelled or only partially captured through manual features or local patterns
- Long range dependencies and complex sentence structures are not effectively handled
- Robust handling of negation, temporality, and intent is not achievable without extensive manual engineering

These limitations directly conflict with the requirements of Phase 3, where:

- Entity validity depends on full-sentence contextual interpretation
- Clinical meaning is often distributed across multiple tokens and clauses  

As a result, non-transformer models are not suitable for this task, motivating the selection of transformer-based architectures.

---

#### 4.3 Why Transformers Are Selected

Transformers address these limitations through:

- **Self-attention mechanism:** Directly model relationships between all tokens in a sentence  
- **Context-aware representations:** Meaning of a word depends on surrounding words (contextualized embeddings)  
- **Implicit feature engineering:** Negation, temporality, and intent are learned from data, no manual rule encoding

This is critical for distinguishing:

- **Negation:** “no chest pain” vs “chest pain”
- **Intent vs execution:** “planned intubation” vs “intubated”  
- **Temporality:** “history of stroke” vs “acute stroke”  

Transformer-based models are the minimal class of models capable of reliably performing the required contextual validation task.

---

### 5. Transformer Classes Considered

#### 5.1 Transformer Variants

| Model Type              | Strengths                                  | Weaknesses |
|------------------------|---------------------------------------------|------------|
| **General BERT**           | Efficient, well-supported                   | Weak clinical understanding |
| **Clinical BERT variants** | Domain-specific language understanding      | Slightly heavier |
| **Large LLMs (GPT, Gemma)**| Strong reasoning, flexible                  | Expensive, non-deterministic, poor structure control, unstable outputs |
| **Custom-trained**   | Fully tailored                             | Requires large labelled dataset |

---

#### 5.2 Encoder vs Generative Models

**Encoder Models (BERT-style):**

- Input → full sequence processed simultaneously  
- Output → fixed contextual representation → classification head  
- Designed for:
  - Classification
  - Token labelling
  - Sentence-level understanding  

Properties:
- Bidirectional context (full sentence considered at once)
- Outputs structured probabilities (e.g. logits → softmax)
- Deterministic at inference:
  - Given fixed weights and no stochastic layers (e.g. dropout disabled), the same input produces identical output
- Strong alignment with structured prediction tasks

**Generative Models (LLMs):**

- Input → sequential token generation  
- Output → generated text (token-by-token)  
- Designed for:
  - Open-ended reasoning
  - Text generation
  - Instruction following  

Properties:
- Autoregressive decoding (predict next token repeatedly)
- Output depends on:
  - Prompt phrasing  
  - Decoding strategy (temperature, top-k, top-p)  
- Inherently probabilistic:
  - Even with temperature = 0, outputs may vary due to:
    - Implementation-level nondeterminism
    - Tie-breaking between tokens
- Produces unstructured text rather than fixed schema outputs  

---

#### 5.3 Why LLMs Are Not Used

Although LLMs are powerful, they are fundamentally misaligned with the requirements of this pipeline.

**1. Task mismatch**

- Required task: Binary classification (`is_valid`) with structured outputs  
- LLM capability: Free-form text generation  
- Implication:
    - Requires prompt engineering to simulate classification  
    - Adds unnecessary abstraction and failure modes 

**2. Output instability and lack of strict determinism**

- LLM outputs are probabilistic by design:
  - Token generation involves probability distributions  
- Even with constrained decoding, outputs can vary across runs, hardware, or implementations  
- Problems:
  - Cannot guarantee strict JSON schema compliance  
  - Requires post-processing, output validation, and error handling layers  
- In contrast, encoder models produce:
  - Fixed-dimensional outputs
  - Stable probabilities
  - Fully reproducible predictions 

**3. Scalability and efficiency constraints**

- LLMs have high computational cost per inference and slow sequential decoding  
- Pipeline requirement:
  - High-throughput batch classification over thousands of entities  
- Implication:
  - LLMs are inefficient and costly at scale 

**4. Reproducibility and auditability**

- Clinical pipelines require:
  - Stable outputs  
  - Reproducible behaviour  
  - Auditability of decisions  
- LLMs are sensitive to:
  - Model updates  
  - Prompt changes  
  - Sampling behaviour  
- Encoders provide:
  - Deterministic forward pass 
  - Consistent outputs across runs  

LLMs are better suited for exploration, prototyping, and complex reasoning tasks, but not for high-throughput, structured, deterministic validation pipelines in clinical contexts.

---

### 6. Transformer Decision

#### 6.1 Final Model Class

Chosen approach:

1. Clinical-domain pretrained **encoder transformer**
2. Supervised **fine-tuning for classification**

Candidate models:
- BioClinicalBERT  
- PubMedBERT  
- ClinicalBERT 

---

#### 6.2 Why Clinical Pretrained Models

Clinical text differs significantly from general language:

- Abbreviations: “SOB”, “PRBC”, “NGT”  
- Domain-specific terminology  
- Fragmented, shorthand-heavy syntax  
- Non-standard grammar  

General-domain models:

- Trained on Wikipedia / BooksCorpus  
- Limited exposure to clinical language  
- Misinterpret:
  - Abbreviations  
  - ICU shorthand  
  - Domain-specific phrasing  

Clinical-domain models:

- Pretrained on:
  - MIMIC clinical notes (BioClinicalBERT)  
  - PubMed abstracts (PubMedBERT)  
- They learn:
  - Clinical vocabulary  
  - Abbreviation usage  
  - Real-world documentation patterns  

Outcome:

- Better contextual understanding  
- Higher classification accuracy  
- Reduced need for large fine-tuning datasets  

---

#### 6.3 Why Fine-Tuning vs Training from Scratch

**A. Model Training**

Model training from scratch involves:

- Initialising model weights randomly  
- Training on very large corpora (millions–billions of tokens)  
- Learning:
  - Language structure  
  - Grammar  
  - Domain knowledge  
  - Semantic relationships  
- Requirements:
  - Massive labelled or unlabeled datasets  
  - Significant compute (GPUs/TPUs)  
  - Long training time  

Why training from scratch is not feasible:

- Phase 2 does not provide:
  - Large-scale labelled datasets  
  - Sufficient data diversity  
- Resource constraints:
  - Compute cost is prohibitive  
  - Time-to-iteration is too slow  
- Most importantly:
  - The task does not require learning language from scratch  
  - Only requires adapting existing language understanding  

---

**B. Fine-Tuning**

Fine-tuning involves:

- Starting from a pretrained model  
- Adding a task-specific classification head  
- Training on a smaller labelled dataset  

Process:

1. Input: `(sentence_text, entity_text)`  
2. Encode full sentence context  
3. Extract contextual representation  
4. Pass through classification head  
5. Optimise for task-specific objective (e.g. binary classification loss)  

Why fine-tuning is appropriate:

- Leverages pretrained knowledge as language understanding is already learned  
- Requires much less data and less compute  
- Adapts model to:
  - Task-specific definitions (e.g. “valid intervention”)  
  - Entity-specific semantics  

Fine-tuning transforms a general clinical language model into a task-specific clinical reasoning model without needing large-scale training.

---

#### 6.4 Candidate Model Comparison

| Model              | Pretraining Data        | Domain Focus        | Strengths | Limitations |
|------------------|------------------------|---------------------|----------|------------|
| **BioClinicalBERT** | MIMIC clinical notes    | ICU / clinical text | Best match to ICU language, strong shorthand handling | Slightly domain-specific |
| **PubMedBERT**     | PubMed abstracts        | Biomedical research | Strong biomedical knowledge, robust terminology | Less exposure to clinical note structure |
| **ClinicalBERT**   | Mixed clinical corpora  | General clinical    | Earlier clinical adaptation | Older, less optimised pretraining |

- **BioClinicalBERT:** best for real-world ICU notes (matches dataset distribution)  
- **PubMedBERT:** best for formal biomedical literature  
- **ClinicalBERT:** earlier, less specialised variant  

---

#### 6.5 Final Model Choice

Selected: **BioClinicalBERT**

**Rationale:**

1. **Data alignment**
  - Pretrained on MIMIC notes → closest match to ICU dataset  

2. **Language compatibility**
  - Strong handling of:
    - Abbreviations  
    - Clinical shorthand  
    - Irregular sentence structures  

3. **Task alignment**
  - Designed for clinical NLP tasks  
  - Proven performance on:
    - Clinical classification  
    - Entity understanding
  - Fine-tuning suitable for binary classification of entity validity  

4. **Efficiency**
  - Encoder-based → fast, parallelisable implementation  
  - Compute efficient
  - Suitable for batch inference  

**Trade-offs:**

| Aspect        | Outcome |
|--------------|--------|
| Performance  | High (domain-aligned) |
| Efficiency   | Moderate (acceptable for batch processing) |
| Determinism  | Strong (stable inference outputs) |
| Flexibility  | Lower than LLMs, but sufficient for classification |

---

### 7. Final Design Decision

Final design:
- Use BioClinicalBERT (clinical pretrained encoder)
- Apply supervised fine-tuning for entity-level classification
- Input: `(sentence_text, entity_text)`
- Output: `is_valid`, `confidence`, `task`

Key architectural decisions:
- Deterministic encoder-based inference  
- No generative modelling  
- No expansion of rule-based logic  
- No reliance on prompt-based systems 

The result is a robust, scalable, and reproducible validation layer that:
- Accurately interprets clinical context  
- Resolves ambiguity (negation, temporality, intent)  
- Integrates seamlessly with Phase 2 outputs  
- Produces structured, audit-ready predictions for downstream use  

---

## Dataset Construction and Annotation Preparation

### 1. Objective

To construct a high-quality, annotation-ready dataset for training and evaluating a transformer-based validation model. The dataset must:

- Be sufficient for fine-tuning a pretrained model
- Be efficient to manually annotate
- Maintain balanced representation across entity types
- Support reproducible downstream training, validation, and testing

---

### 2. Sampling Strategy

#### 2.1 Dataset Size

A total of 600 entities were selected for annotation. This size was chosen because:

- It is sufficient for fine-tuning a pretrained transformer (e.g., ClinicalBERT)
- It supports binary classification at sentence level
- It balances annotation effort vs performance gains

The datasset size is too small for training from scratch, but appropriate for fine-tuning pretrained models: 
 
- This is because the model already encodes language structure and clinical semantics. 
- The dataset is used to learn task-specific definitions and decision boundaries  

Increasing beyond 600:

- Produces diminishing returns
- Increases annotation cost significantly
- Is unnecessary for this project scope  

Dataset expansion (800–1000) is only justified if:

- Model performance is unstable  
- Clear underfitting is observed  

---

#### 2.2 Balanced Sampling by Entity Type

Sampling is performed equally across entity types using `entity_type`-based sampling:

- `SYMPTOM`: 200  
- `INTERVENTION`: 200  
- `CLINICAL_CONDITION`: 200  

This ensures:

- Balanced semantic coverage  
- Reduced bias toward any entity type  
- Improved generalisation across tasks  

---

### 3. Field Selection and Data Structure

The dataset is constructed from the JSONL extraction outputs (`extraction_candidates.jsonl`) and flattened into tabular csv format.

#### 3.1 Retained Fields

- `note_id` → traceability and debugging  
- `section` → contextual location in note  
- `concept` → normalized concept label  
- `entity_text` → extracted entity span  
- `entity_type` → classification category (SYMPTOM, INTERVENTION, CLINICAL_CONDITION)
- `sentence_text` → classification context (critical input)  
- `negated` → retained for rule-based comparison (not used in training)  
- `task` → defines classification objective  
- `confidence` → placeholder (0.0), used later for model outputs  

---

#### 3.2 Annotation Field

`is_valid` is the only manually annotated field:

- Binary label: `True` / `False`
- Represents ground truth for model training  

All other fields remain unchanged to preserve consistency with upstream extraction to ensure:

- Low annotation complexity  
- High consistency  
- Clear separation between:
  - Extraction (rule-based)
  - Validation (model-based)

---

#### 3.3 Confidence Score Retention

The `confidence` field is retained in the dataset but is not used during manual annotation or training input.

Its purpose is to support post-training evaluation and calibration, where it will store model-generated confidence scores. This enables:

- Threshold tuning (optimising decision boundaries)
- Prediction ranking (prioritising high-confidence outputs)
- Error analysis (identifying overconfident incorrect predictions)

The field is initialised during sampling (default `0.0`) to maintain schema consistency and allow seamless integration of model outputs later in the pipeline.

---

#### 3.4 Negation Flag Retention

The `negated` field is retained to enable comparison between rule-based extraction outputs and transformer-based validation, as its role differs by entity type. 

For `SYMPTOM`:

- The `negated` flag is populated during rule-based extraction (Phase 2)
- It directly determines rule-based validity:
  - `negated = True` → invalid (`False`)
  - `negated = False` → valid (`True`)
- This provides a strong baseline, as symptom negation is reliably captured using deterministic rules

For `INTERVENTION` and `CLINICAL_CONDITION`:

- The `negated` field is not populated (null) as no rule-based validity filtering is applied
- All extracted entities are treated as valid under the rule-based system
- This reflects an intentional high-recall design:
  - Maximise capture of candidate entities
  - Defer validity determination to the transformer model
- Validity for these entity types depends on temporality and intent, which cannot be reliably captured using deterministic rules alone

The `negated` field is therefore critical for `SYMPTOM` validation:

- Establishes a strong rule-based baseline where transformer improvements are expected to be more modest
- Enables direct comparison between deterministic negation logic and transformer predictions

For `INTERVENTION` and `CLINICAL_CONDITION` however:

- Model performance improvements are evaluated independently of `negated`, as no rule-based filtering is applied to these entity types.
- This establishes a weak rule-based baseline, allowing the transformer to provide substantial performance gains by learning complex contextual cues that determine validity.

---

### 4. Output Design

Two files are generated for reproducibility and data integrity:

`annotation_sample_raw.csv`:
- Always overwritten  
- Contains freshly sampled dataset  
- Used for traceability and reproducibility  

`annotation_sample_labeled.csv`:
- Created only if it does not exist  
- Used for manual annotation  
- Preserved to prevent accidental data loss  

This design ensures:
- Original sample is always recoverable  
- Manual annotations are never overwritten  

---

### 5. Implementation Workflow 

The dataset construction process is implemented in `sample_entities.py` and follows a deterministic, reproducible pipeline:

1. **Load extraction candidates**
  - Read JSONL file (`extraction_candidates.jsonl`)
  - Each line corresponds to a single extracted entity
  - Parse each line into a Python dictionary

2. **Field extraction and flattening**
  - Extract relevant fields:
    - `note_id`, `section`, `concept`, `entity_text`, `entity_type`, `sentence_text`, `negated`
  - Extract nested validation fields safely:
    - `task` via `row.get("validation", {}).get("task")`
    - `confidence` with default `0.0`
  - Append all records into a list

3. **DataFrame construction**
  - Convert records into a pandas DataFrame
  - This forms the full pool of candidate entities (~40k+)

4. **Per-entity-type filtering**
  - Split DataFrame into subsets by:
    - `SYMPTOM`
    - `INTERVENTION`
    - `CLINICAL_CONDITION`

5. **Balanced sampling**
  - For each entity type:
    - Verify sufficient sample size (≥200), else raise error
    - Randomly sample **200 rows** using fixed `random_state=42`
  - Ensures reproducibility and equal class representation

6. **Dataset assembly**
  - Concatenate sampled subsets into a single DataFrame (600 rows total)
  - Reset index to ensure clean ordering

7. **Global shuffling**
  - Shuffle entire dataset (`frac=1`, `random_state=42`)
  - Prevents ordering bias during manual annotation

8. **Annotation column initialization**
  - Add `is_valid` column initialised to `None`
  - Serves as the ground truth label to be manually populated

9. **Output handling**
  - Save `annotation_sample_raw.csv`:
    - Always overwritten
    - Acts as reproducible source sample
  - Save `annotation_sample_labeled.csv`:
    - Created only if it does not already exist
    - Prevents accidental overwrite of manual annotations

This pipeline ensures:
- Deterministic sampling (via fixed seed)
- Balanced representation across entity types
- Preservation of annotation work
- Full reproducibility of dataset construction

---

## Manual Annotation

### 1. Objective

To define a clear, consistent, and reproducible framework for assigning binary labels (`is_valid = True/False`) to extracted clinical entities.

The goal is to:
- Ensure high-quality ground truth labels for transformer training  
- Minimise annotation variability and drift
- Standardise interpretation of clinical language across entity types  
- Align task labels with clinically meaningful and model-relevant definitions 

Annotation is performed strictly based on sentence-level context, ensuring that labels reflect the information explicitly present in the text rather than inferred assumptions.

---

### 2. Annotation Framework and Principles

- **Sentence-level grounding**  
  - Labels are assigned using the full `sentence_text`  
  - The `entity_text` is interpreted only within this context

- **Context over metadata**  
  - `section` may assist interpretation in ambiguous cases, but is not determinative
  - `concept` can be used to verify whether the extracted entity aligns with the intended clinical meaning
  - If the entity is clearly mis-extracted (i.e. does not match the intended concept in context), label as `False`

- **Explicit evidence over inference**  
  - Labels must be based only on clearly stated information 
  - Do not infer presence, action, or activity from typical clinical patterns

- **Consistency over intuition**  
  - Apply rules systematically, even if clinically imperfect  
  - Prioritise reproducibility over subjective clinical judgement
  - This ensures learnable and reproducible patterns for the model  

- **Ambiguity handling (conservative bias)**  
  - Default to conservative interpretation  
  - Uncertain or implied mentions are typically labeled `False` unless clearly active/performed  

- **Temporal consistency**
	-	Interpret timing strictly based on wording in the sentence
	-	Do not assume recency or continuity unless explicitly stated
	-	This is critical for:
	  -	State-based tasks (current vs past)
	  -	Event-based tasks (performed vs planned)

---

### 3. State vs Event-Based Labeling

A key design decision is the use of state-based vs event-based labeling, depending on entity type which defines the task type. This dual approach is standard, defensible, and consistent with real-world clinical use cases and broad downstream applications.

| Entity Type                | Labeling Type | Interpretation |
|---------------------------|--------------|----------------|
| `symptom_presence`          | State-based  | Is it present now? |
| `intervention_performed`    | Event-based  | Has it occurred during admission? |
| `clinical_condition_active` | State-based  | Is it currently active? |

State-based:
- Reflects the patient’s current status at the time of the note
- Past, resolved, or uncertain mentions → `False`

Event-based:
- Reflects whether an action has occurred at any point during admission
- Includes current, completed, or past interventions → `True` if performed

This distinction is critical and must be applied consistently.

---

### 4. Entity-Specific Task Guidelines

#### 4.1 `symptom_presence`

**Overview:**
- Definition: Does the patient currently exhibit the symptom?
- Type: State-based

**Justification:** 
- Symptoms are inherently real-time
- Downstream tasks like monitoring, alerts, or clinical decision support (CDS) require the current status, not historical mentions.

**Rules:**
- `True` → symptom is currently present  
- `False` → negated, historical, chronic baseline, or not actively occurring  

**Examples:**
- “Patient is complaining of nausea” → `True`  
- “Patient denies pain” → `False` 

**Edge Cases:**
- Historical or chronic baseline symptoms → `False` 
- Provoked only (e.g. “asked to cough”) → `False`  
- “PRN nausea” → `False` (indication only, not current symptom)  
- Ambiguous phrasing → default `True` only if likely current  

---

#### 4.2 `intervention_performed`

**Overview:**
- Definition: Has the intervention been performed at any point during the admission up to this note?
- Type: Event-based

**Justification:**
- Interventions are often documented retrospectively. 
- Even if completed by the time of the note, they still count as having occurred. 
- This is useful for retrospective analyses, modeling treatment patterns, or cohort studies.

**Rules:**
- `True` → intervention has occurred (past or ongoing)  
- `False` → planned, conditional, hypothetical, or not confirmed  

**Examples:**
- “Received 2 units PRBCs” → `True`  
- “Plan to start heparin” → `False`  

**Edge Cases:**
- **PRN medications** → `False` unless explicitly administered  
- **Weaning/continuation** → `True` (active process)  
- **“Post antibiotics” / “received prior”** → `True` (event occurred)  
- **Prophylaxis headings** → `False` unless explicitly started (e.g. “Started heparin subq for DVT prophylaxis” → `True`)
- **Implicit presence (e.g. ETT in place)** → `True` (interventions like tubes and lines are typically documented in a way that implies they have been performed)  

---

#### 4.3 `clinical_condition_active`

**Overview:**
- Definition: Is the condition currently active and affecting the patient?
- Type: State-based

**Justification:**
- Acute conditions need to reflect current patient status. 
- This is aligned with real-time decision-making, risk assessment, and condition tracking.

**Rules:**
- `True` → explicitly active and clinically relevant  
- `False` → historical, resolved, chronic baseline, negated, or uncertain  

**Examples:**
- “Worsening ARDS” → `True`  
- “Resolved pneumonia” → `False`  

**Edge Cases:**
- **Implicit context:** “Hx diabetes admitted for sepsis” → diabetes `False`, sepsis `True`  
- **Uncertainty:** “Possible pneumonia” → `False`  
- **Causal mentions:** “Lactate elevated due to sepsis physiology” → `False` unless explicitly active  
- **Headers / lists:** ALL CAPS or diagnostic lists → `False` unless sentence confirms activity or sentence context indicates active relevance 
- **Single-term mentions:** “Sepsis” alone → `False` without context (e.g. the `section` field, or the entity itself)

---

### 3. Manual Annotation Validation 

#### 3.1 Purpose

Dataset sampling in general can be subject to errors such as incorrect filtering, mislabeling, or data corruption. Manual annotation also introduces potential risks such as incomplete labels, formatting inconsistencies, and distributional imbalances.

Validation is therefore required to:

- Ensure all 600 samples are fully annotated (`is_valid` complete)
- Confirm labels are strictly binary (`True` / `False`)
- Verify dataset structure matches expectations (no missing critical fields)
- Ensure balanced task representation (200 per task)
- Check label distribution among tasks for consistency and absence of severe imbalance
- Detect errors early before dataset splitting and model training

This ensures the dataset is:

- Structurally complete, label-consistent, and statistically balanced
- Ready for reproducible train/validation/test splitting

This prevents propagation of annotation errors into training, which would otherwise degrade model performance and invalidate evaluation results.

---

#### 3.2 Validation Workflow

The validation logic is implemented in `validate_manual_annotations.py` and follows a structured sequence of checks:

1. **Load annotated dataset**  
  - Read `annotation_sample_labeled.csv` into a DataFrame  
  - Establish dataset as the single source of truth for downstream steps  

2. **Structural integrity checks**  
  - Confirm total row count (expected: 600)  
  - Verify all expected columns are present  
  - Detect any global missing values  

3. **Critical field validation**  
  - Ensure no missing values in:
    - `is_valid` (must be fully annotated)
    - `task` (required for stratification)
    - `sentence_text` (required for model input)  

4. **Label validation (`is_valid`)**  
  - Confirm only valid binary labels are present (`True`, `False`)  
  - Identify and flag any invalid or unexpected values  
  - Compute overall label distribution to assess balance  

5. **Task distribution checks**  
  - Verify exact balance across tasks:
    - 200 `symptom_presence`
    - 200 `intervention_performed`
    - 200 `clinical_condition_active`  
  - Flag deviations from expected counts  

6. **Task–label cross-distribution analysis**  
  - Generate cross-tabulation of `task` vs `is_valid`  
  - Inspect for:
    - Severe imbalance within a task
    - Potential annotation bias or systematic errors  

7. **Diagnostic output and review**  
  - Print all validation metrics and warnings  
  - Require manual review before proceeding  
  - Only proceed to stratified splitting if no critical issues are detected  

---

#### 3.3 Validation Output Interpretation

**A. Dataset Structure**

- Total rows: **600** → matches expected sample size  
- All required columns present  
- No structural inconsistencies detected  

Dataset structure is correct and complete.

---

**B. Missing Values**

- No missing values in critical fields:
  - `task`: 0  
  - `sentence_text`: 0  
  - `is_valid`: 0  

- `negated`: 400 missing → expected and correct  
  - Only populated for `SYMPTOM` entities  
  - Not used for `INTERVENTION` or `CLINICAL_CONDITION`  

Missingness is expected and does not affect training.

---

**C. Label Integrity**

- Unique values: `[True, False]` → valid binary labels  
- Invalid label rows: **0**  

Labels are clean, consistent, and usable.

---

**D. Label Distribution**

| Label | Count |
|------|-------|
| True | 305   |
| False | 295  |

- Near-balanced distribution (~51% / 49%)
- No significant class imbalance detected

Dataset is suitable for training without label rebalancing

---

**E. Task Distribution**

| Task                        | Count |
|-----------------------------|-------|
| `symptom_presence`            | 200   |
| `clinical_condition_active`   | 200   |
| `intervention_performed`      | 200   |

- Exact balance across all tasks  

Stratified sampling was correctly implemented.

---

**F. Task vs Label Distribution**

| Task                        | `False` | `True` |
|-----------------------------|---------|--------|
| `clinical_condition_active`   | 122     | 78     |
| `intervention_performed`      | 71      | 129    |
| `symptom_presence`            | 102     | 98     |

- **symptom_presence:** Balanced (~50/50); maybe due to straightforward nature of symptom mentions
- **intervention_performed**: Skewed toward `True`; expected due to frequent documentation of completed interventions
- **clinical_condition_active** → skewed toward `False`; expected due to historical conditions, uncertain diagnoses, and header-like extractions  

Distribution patterns are clinically and methodologically consistent. No correction required.

---

**Final Assessment**

- All structural checks passed  
- All labels valid and complete  
- Balanced dataset across tasks  
- Acceptable and explainable label distributions  

The dataset is validated and ready for stratified train/validation/test splitting.

---

## Dataset Splitting


⸻

Split Strategy

Use:
	•	Train: 420 (70%)
	•	Validation: 90 (15%)
	•	Test: 90 (15%)

⸻

Per-Entity Stratified Distribution

Total per entity type:
	•	SYMPTOM: 200
	•	INTERVENTION: 200
	•	CLINICAL_CONDITION: 200

Split per entity type:

Split
Per Type
Total
Train
140
420
Val
30
90
Test
30
90


Why This Split Is Correct

1. Train (420)
	•	Large enough to learn:
	•	Negation patterns
	•	Clinical phring
	•	Entity-type differences
	•	Still feasible to annotate

⸻

2. Validation (90)

Used during training to:
	•	Monitor performance across epochs
	•	Prevent overfitting
	•	Select best model checkpoint

Without validation:
	•	Overfitting risk increases
	•	No reliable stopping criterion

⸻

3. Test (90)

Used once at the end

Purpose:
	•	Final unbiased evaluation

Metrics reported:
	•	Accuracy
	•	Precision / Recall / F1
	•	Improvement over rule-based extraction

⸻

Stratification Requirement

Stratification is mandatory.

Without it:
	•	Model may overfit to one entity type
	•	Metrics become misleading
	•	Entity-level comparisons break


Correct approach (must follow this)

You split once, not three times

Step 1 — Sample 600 total (balanced)

From your 40,000+ entities:
	•	Randomly sample:
	•	200 SYMPTOM
	•	200 INTERVENTION
	•	200 CLINICAL_CONDITION

This is your annotation dataset

⸻

Step 2 — Annotate all 600

Add:

"is_valid": true / false

Step 3 — Perform stratified split (2-stage)

You now split this single annotated dataset

⸻

Split process (correct)

First split:
	•	Train: 70% (420)
	•	Temp: 30% (180)

Stratified by entity_type

⸻

Second split:

Split the 180 into:
	•	Validation: 90
	•	Test: 90

Again stratified

⸻

Why this works
	•	No overlap between sets
	•	Maintains class balance
	•	Proper statistical separation
	•	Clean, reproducible



	•	Use validation for early stopping only
	•	Evaluate once on test set

Overall the entity-distribution 600 sample dataset, manual annotation, and subsequent stratified split is the correct approach because it ensures:
	•	Statistically valid
	•	Efficient
	•	Fully aligned with your pipeline design
	•	Minimal time for maximum outcome


⸻

Expected Outcome (with 600)

You should observe:
	•	Clear improvement over rule-based extraction
	•	Performance pattern:

Entity Type
Expected Performance
SYMPTOM
Highest
INTERVENTION
Moderate
CLINICAL_CONDITION
Lowest

Reason:
	•	Increasing semantic complexity across entity types

⸻










