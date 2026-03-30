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



Dataset Size
	•	Total annotated: 600 entities

This is sufficient for:
	•	Binary classification
	•	Sentence-level contextual tasks
	•	Fine-tuning a pretrained clinical transformer

Increasing beyond 600:
	•	Yields diminishing returns
	•	Increases annotation time significantly
	•	Not necessary for this project scope

⸻

Is 600 “too small”?

Objectively:
	•	Small for training from scratch → Yes
	•	Appropriate for fine-tuning BERT → No

Why it works:
	•	The model already understands language and clinical patterns
	•	You are only teaching:
	•	Task definition
	•	Decision boundaries

⸻

When to increase beyond 600

Only if:
	•	Test performance is unstable
	•	Clear underfitting is observed
	•	Additional time is available

Then increase to:
	•	800–1000

Otherwise:
	•	600 is the optimal trade-off

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





Annotation Scope (Minimal)

Only annotate:

"is_valid": true / false

Do NOT modify:
	•	task
	•	confidence
	•	Any other fields

⸻

Confidence Scores
	•	Produced by the model (softmax output)
	•	Not manually annotated

Used later for:
	•	Threshold tuning
	•	Ranking predictions
	•	Error analysis

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

Final Summary
	•	Use 600 annotated samples
	•	Ensure balanced entity distribution
	•	Apply stratified train/val/test split
	•	Keep annotation minimal (is_valid only)
	•	Use validation for early stopping only
	•	Evaluate once on test set

This is:
	•	Statistically valid
	•	Efficient
	•	Fully aligned with your pipeline design
	•	Minimal time for maximum outcome






Manual annotation

CLINICAL_CONDITION

TRUE  → actively contributing to current admission
FALSE → PMHx, chronic, resolved, historical, negated, or uncertain

Where errors will happen (important)

The model will struggle with:

1. Implicit context

“patient with diabetes admitted for sepsis”

	•	diabetes = FALSE
	•	sepsis = TRUE

⸻

2. Mixed clauses (your example type)

“with HTN … found to have NSTEMI”

Requires:
	•	ignoring earlier entities
	•	focusing on later clause

⸻

3. Ambiguous phrasing

“possible pneumonia”

	•	depends on your rule:
	•	usually FALSE (uncertain)

⸻

7. Why your design still works

Because:
	•	You already separated extraction from validation
	•	The transformer only needs to:
	•	filter candidates
	•	not discover them

This reduces difficulty significantly.

⸻

8. Will 600 samples be enough for this?

Yes, because:
	•	Patterns are repetitive in ICU notes
	•	Clinical phrasing is structured
	•	You are not learning language from scratch

But:
	•	CLINICAL_CONDITION will be your weakest class
	•	Expect:
	•	lower precision
	•	more edge-case errors


Edge Case Example – Implicit vs Active Conditions
	•	Sentence: "Trop bump likely demand ischemia from anemia and sepsis physiology."
	•	Entity: sepsis
	•	Decision: is_valid = False
	•	Reasoning:
	•	The mention of “sepsis physiology” describes an effect of sepsis, not an explicit active diagnosis.
	•	Phrases like “likely from X” or “effects of X” should not be labeled as active conditions.
	•	Only explicitly stated, currently active conditions are labeled True (e.g., “patient admitted with sepsis”, “currently septic”).

Takeaway: Transformers need to distinguish referential mentions or causal explanations from true active clinical conditions. Rule-based extraction often overcalls these cases.


“…start ceftriaxone for presumptive UTI…”

Analysis:
	•	“presumptive UTI” → uncertain, not confirmed
	•	No statement that the UTI is actively diagnosed or causing current admission issues

“Goal to have improved rate control in the peri-MI period.”

Analysis:
	•	This is discussing management/goals for a past MI, not an active MI causing the current admission.
	•	It’s a resolved/historical condition, not contributing to current admission.

“Likely SBP but also known UTI and possible PNA.”

Analysis:
	•	“Known UTI” → this refers to a pre-existing or previously diagnosed UTI, not necessarily active or currently contributing.


Edge Case: Header-like Extractions
	•	Issue: Sometimes the extraction picks up text that is a section header rather than part of a sentence describing the patient’s active condition.
	•	Impact: Even if the entity is a real clinical concept (e.g., MYOCARDIAL INFARCTION), the transformer should not label it as active unless the sentence text confirms current occurrence.
	•	Rule:
	•	FALSE if the extracted entity appears to be a header (all caps, followed by a colon, comma, or line break) and the sentence does not explicitly describe an active condition.
	•	TRUE only if the sentence text clearly confirms that the condition is currently active or contributing to the admission.
	•	Example:
	•	Entity: MYOCARDIAL INFARCTION
	•	Sentence: "MYOCARDIAL INFARCTION, ACUTE (AMI, STEMI, NSTEMI) 80 y/o woman with hx alzheimers transfered with dynamic EKG changes/mild troponin leak equivocal for NSTEMI with decision for medical management."
	•	Label: FALSE → header + sentence ambiguous

Edge Case – “Underlying Infection” or Cause-Linked Mentions
	•	Scenario: The entity mentions an infection as the underlying cause of an acute physiological effect (e.g., hypoperfusion, elevated lactate).
	•	Decision Rule:
	•	TRUE → If the infection is actively causing a current abnormality or clinical issue.
	•	FALSE → If the infection is only historical, chronic, or mentioned as a baseline without causing acute derangement.
	•	Rationale: The transformer needs to distinguish between causal references and explicit active conditions. Words like “underlying” can appear in either context, so annotation must consider the linked effect in the sentence.
	•	Example:
	•	Sentence: “in setting of hypoperfusion d/t underlying infection (lactate previously normal even in the setting of his liver disease).”
	•	Entity: infection → TRUE because it is causing current hypoperfusion.
	•	Notes: Treat phrases like “likely from X” or “effects of X” differently: these are usually FALSE unless they clearly indicate a current active condition.

Edge Case – Header with Differential:

If the extracted entity appears in a header or section title followed by a differential list (e.g., “# Encephalopathy: Differential includes …”), do not label as active, even if the patient might clinically have the condition. Only explicitly stated, currently active diagnoses are TRUE.


Edge Case – Single-Term / No Context Mentions:
If the sentence contains only the entity (e.g., “Sepsis”) with no contextual information, do not assume it is active. Label as FALSE unless the sentence explicitly confirms current presence.





For SYMPTOM entities:
- Count symptom as present (TRUE) if:
  - Mentioned as currently occurring
  - Mentioned as contributing factor to another symptom
- Count as absent (FALSE) if:
  - Explicitly denied
  - Historical or resolved
  - Only used as a provocation (e.g., "patient asked to cough")

In ambiguous phrasing like "increased with cough" without "asked to cough", you can default to TRUE, but note it as a potential edge case.

"nothing out of ordinary from usual chronic abdominal pain" → this indicates the symptom is not new or actively contributing to the current admission, it’s just part of their baseline or chronic history.
- For SYMPTOM, TRUE only applies to symptoms that are currently occurring and clinically relevant, not chronic or baseline complaints.


Edge Case Example – Medication Indications vs Active Symptoms
	•	The note is listing current medications and their indications, not current symptoms.
	•	“PRN nausea” and “q6H prn nausea/anxiety” indicate the patient takes these medications as needed, but there is no evidence the patient is currently experiencing nausea.
	•	Per your symptom rules:
	•	TRUE: symptom is currently occurring or contributing to admission.
	•	FALSE: symptom is historical, resolved, negated, or only mentioned as a medication indication or provocation.

✅ Conclusion: is_valid = False.

This is exactly the type of edge case where the transformer might overcall if it just looks for the keyword “nausea” without context.


INTERVENTION

TRUE: Action is actively performed on the patient (medication administered, procedure performed, imaging done, weaning off meds meaning theyre currently performaing it).
FALSE: Intervention is planned, historical, conditional, or mentioned as monitoring/guideline/consideration only.

	•	If the text describes ongoing, real actions the patient is currently receiving, label TRUE.
	•	If it’s planned, hypothetical, or past, label FALSE.

So for weaning vasopressors, even though the intervention started earlier, it is still active and being performed, so is_valid = True.

The transformer will learn that phrases like "wean", "continue", "administered", "received" indicate actual performed interventions, whereas "plan to", "if tolerated", "consider" indicate not performed.

“- will start treatment for ESBL E. coli bacteremia with meropenem desensitization protocol …”

	•	The wording “will start” explicitly indicates future action.
	•	According to your INTERVENTION rules, only interventions that have actually been carried out are True. Planned/future interventions are False.

“Elevated lactate at 3 on ABG and bloody secretions from mouth and ETT.”

Interpretation:
	•	The presence of secretions from the ETT implies that the ETT has already been placed, because secretions only come after insertion.
    In this note, the ETT is present during the documented ICU stay, as evidenced by “secretions from … ETT.” That implies it is actively in place now, not just a past event.
	•	Even if it was placed earlier in the admission, the intervention is currently affecting the patient, so it counts as TRUE per your rule: “action is actively performed on the patient.”

The text says: “24 hrs post antibiotics” — this explicitly indicates the antibiotics have already been administered.
	•	According to your intervention rules:
	•	TRUE: intervention is actively performed (medication administered, procedure done, imaging done, or weaning currently happening).
	•	FALSE: planned, historical, conditional, or guideline-only.
	•	Even though the note is reflecting retrospectively (“post antibiotics”), it refers to an intervention that was actually performed during this admission, so it counts as TRUE.


	•	Entity: Morphine
	•	Sentence: "Neurologic: Neuro checks Q: 2 hr, Morphine prn."
	•	Analysis:
	•	prn (pro re nata) indicates as needed, so the medication may or may not have been administered yet.
	•	According to your INTERVENTION rules:
	•	TRUE → actively performed
	•	FALSE → planned, historical, conditional, guideline/consideration only
	•	Since prn does not guarantee the drug was given, this counts as planned/conditional, not actively performed.
	•	Label: False for intervention_performed

Rule for prn medications:
	•	If the note says medication prn and there’s no confirmation it was actually administered, mark FALSE.
	•	If the note confirms administration (e.g., “Morphine given 2 mg IV”), mark TRUE.

Edge Case – “Additional” Without Confirmation:
The word “additional” implies prior intervention but does not confirm it within the sentence. Treat as FALSE unless prior or ongoing administration is explicitly stated.

edge case - prophylaxis vs treatment
    •	“Prophylaxis: Subutaneous heparin” → this is False because prophylaxis doesnt explicitly mean the patient is recieving it, it’s more of a guideline or consideration or plan. If the note said “started on heparin sub q for DVT prophylaxis” then it would be True because it confirms the intervention is being performed.