# Hybrid Clinical Notes Extraction Pipeline

***Precision-First Natural Language Processing System Using Deterministic Regex-Based Entity Extraction with BioClinicalBERT Classifier Validation for Structuring Clinical Notes***

---

# Executive Summary

**Tech stack:** ***Python, PyTorch, HuggingFace Transformers, scikit-learn***

This repository presents a dual-architecture natural language processing (NLP) pipeline combining deterministic rule-based entity extraction with context-aware transformer (ClinicalBERT) validation to produce structured, schema-aligned JSON outputs from 162,296 unstructured ICU progress reports over 32,910 unique ICU stays, suitable for downstream machine learning applications.

This a data filtering system for downstream ML.

Extracted 3 entity types from 10,000 reports to generate 47,487 entities for transformer validation.

Rule-based extraction ensures high-precision identification of clinically relevant patterns, while the fine-tuned ClinicalBERT classifier provides negation handling, entity validation, and contextual disambiguation.

This focused on classical clinical NLP pipeline development rather than ontology mapping, interoperability standards, or production deployment. Model training was performed using credentialed access to the MIMIC-III dataset via PhysioNet, ensuring realistic clinical data handling.

This project implements a precision-first clinical NLP pipeline for extracting structured information from ICU clinical notes.

The system combines:

- Deterministic rule-based extraction (high-precision candidate generation)

- Transformer-based validation (ClinicalBERT) for contextual filtering

The output is structured, auditable JSON designed for downstream machine learning tasks.

The focus is on reliable extraction under realistic clinical constraints, rather than maximising model complexity or benchmark performance.

---

# Table of Contents

---

# 1. Clinical Background & Motivation

## 1.1 Clinical Data in EHR Systems

Modern Electronic Health Record (EHR) systems store patient information across two primary data modalities:

| Modality | Characteristics | Examples | Usability |
|----------|----------------|----------|-----------|
| **Structured data** | Standardised format, controlled vocabularies, consistent schema | Vital signs, laboratory results, medication records, diagnosis codes | Stored in tabular or time-series formats; directly queryable and readily usable in statistical analysis and machine learning |
| **Unstructured data** | Free-text, variable structure, context-dependent language | Progress notes, admission notes, discharge summaries, radiology reports | Rich in clinical detail but not directly machine-readable or easily usable for computational analysis |

Unstructured clinical text captures aspects of patient care that are not fully represented in structured fields, including:

- Symptom descriptions, including onset, severity, and progression  
- Clinical reasoning, including differential diagnoses, impressions, and plans  
- Negation, uncertainty, and ruled-out diagnoses  
- Temporal narratives, including history, progression, and response to treatment  
- Contextual factors, including social history, functional status, and baseline state  

However, this information presents significant challenges:

- Lack of standardisation across clinicians, institutions, and time  
- High variability in terminology, abbreviations, and phrasing  
- Strong dependence on context for correct interpretation  
- Difficulty querying, aggregating, or converting into machine-learning features at scale  

As a result, unstructured clinical text contains substantial clinical signal but cannot be directly operationalised for computational use without transformation.

This creates a fundamental disconnect between **data availability** and **data usability**, motivating structured clinical NLP approaches.

##
## 1.2 The Role of Clinical NLP

Clinical natural language processing (NLP) addresses this gap by transforming free-text medical data into structured, machine-readable representations.

In practice, clinical NLP systems may extract:

- Named entities, such as symptoms, diagnoses, medications, procedures, or interventions  
- Attributes, such as negation, temporality, certainty, severity, and status  
- Relationships between clinical concepts  

These outputs can be normalised into structured schemas, enabling integration into downstream workflows such as:

- Cohort identification  
- Clinical audit and population-level analysis  
- Feature generation for predictive modelling  
- Clinical decision support and information retrieval  

This transformation allows narrative clinical data to be queried, aggregated, validated, and integrated into computational clinical systems.

##
## 1.3 Current System Paradigms in Clinical NLP

Clinical NLP systems can be implemented using several methodological paradigms. In practice, systems often combine multiple approaches depending on the task, data availability, annotation burden, and required level of auditability.

| Paradigm | Typical Use | Strengths | Limitations |
|----------|-------------|-----------|-------------|
| **Rule-based systems** | Pattern matching, section parsing, negation rules, dictionary-based extraction | Deterministic, interpretable, auditable, strong for well-defined patterns | Limited linguistic coverage; requires manual engineering |
| **Classical ML / sequence models** | CRF-based NER, feature-engineered classifiers | Historically important for token-level extraction and structured prediction | Requires engineered features; less competitive than transformers for many contextual tasks |
| **Transformer-based models** | NER, classification, attribute detection, contextual validation | Strong contextual representations; effective for negation, temporality, and ambiguity | Requires labelled data; probabilistic outputs require evaluation and thresholding |
| **Generative LLMs** | Summarisation, question answering, report generation, prompt-based extraction | Flexible, expressive, useful for generative and reasoning-heavy workflows | Requires additional controls for schema consistency, reproducibility, output validation, and cost management |

For structured clinical extraction, there is no single universally optimal architecture. The appropriate design depends on whether the priority is span traceability, recall, precision, interpretability, scalability, or flexible language understanding.

This project adopts a hybrid design because it requires:

- Deterministic span extraction  
- Fixed schema-controlled outputs  
- Sentence-level contextual validation  
- Reproducible and auditable behaviour  
- Efficient use of a limited manually annotated dataset  

##
## 1.4 Design Motivation and Project Positioning

### Design Constraints

Clinical NLP system design is shaped by practical constraints:

- **Limited labelled data:** high-quality annotation requires clinical understanding and is expensive to scale  
- **High precision requirements:** false positives can corrupt downstream features, cohorts, or analyses  
- **Need for interpretability and auditability:** outputs should be inspectable and traceable to source text  
- **Reproducibility:** behaviour should remain stable across runs and environments  
- **Context dependence:** clinical validity often depends on negation, temporality, intent, uncertainty, and section context  

These constraints do not make supervised NER or end-to-end extraction invalid in general. Rather, they shape the design choice for this project: a controlled hybrid pipeline is more appropriate than unconstrained model-based extraction or purely generative extraction.

##
### Architectural Implications

The system therefore separates extraction from validation:

| Component | Role |
|----------|------|
| **Rule-based extraction** | Generates deterministic, span-aligned candidate entities within a fixed schema |
| **Transformer validation** | Classifies whether each candidate is clinically valid in sentence context |

This creates a two-stage pipeline:

> High-recall candidate generation → precision-oriented contextual validation

This architecture allows the system to preserve exact span provenance while using machine learning only where contextual interpretation is required.

Accordingly, the project prioritises:

- Pipeline-centric design over model-centric optimisation
- Clear separation between candidate extraction and contextual validation
- Structured outputs suitable for downstream modelling
- Controlled use of learned models as validation components
- Reproducible and auditable system behaviour 

##
### Broader Positioning of This Work

Within a broader clinical data pipeline, this work addresses the transformation from unstructured narrative data into structured features.

It complements prior work on structured physiological modelling by representing an earlier stage in the clinical data processing workflow and by operating on a different data modality:

- Unstructured text → structured feature extraction (this NLP project)
- Structured physiological data → predictive modelling ([Time-Series ICU Deterioration Predictor](https://github.com/SimonYip22/time-series-icu-deterioration-predictor))  

Together, these projects represent connected but distinct components of a clinical data system: extracting structured signal from narrative documentation, then using structured data for downstream modelling.

---

# 2. Project Goals & Contributions

## 2.1 Primary Objectives

1. Develop a clinically grounded NLP pipeline to transform unstructured ICU clinical notes into structured, machine-readable representations suitable for downstream analysis and modelling.
2. Design a hybrid extraction framework that combines deterministic rule-based methods with transformer-based validation to balance recall and precision.
3. Implement section-aware extraction to capture clinically meaningful entities (symptoms, interventions, clinical conditions) from heterogeneous clinical narratives.
4. Produce precision-oriented structured outputs by incorporating contextual validation and classification mechanisms suitable for downstream use.
5. Evaluate pipeline performance against a manually annotated ground truth dataset to assess extraction accuracy and validation effectiveness.
6. Generate a schema-aligned structured dataset from the full corpus of clinical notes, demonstrating scalability beyond single-note inference.
7. Develop an end-to-end inference system that transitions the pipeline from local experimentation to a reproducible, cloud-hosted API.

##
## 2.2 Key Technical Contributions

- **ICU Corpus Generation:** Constructed a filtered corpus of adult ICU progress notes (≥24-hour stay, within 24 hours of admission), enabling consistent and clinically relevant input data for pipeline development.

- **Clinical Text Processing Pipeline:** Implemented preprocessing including text normalisation, section header detection, sentence segmentation, and tokenisation to support structured downstream extraction.

- **Deterministic Entity Extraction System:** Designed a rule-based extraction framework targeting three clinically relevant entity types, incorporating regex-based patterns, section-aware logic, and negation handling to ensure high-recall candidate generation with controlled precision.

- **Hybrid Pipeline Architecture:** Developed a modular system separating candidate generation (rule-based extraction) and contextual validation (transformer-based classification), enforcing controlled interaction between deterministic and probabilistic components.

- **Transformer-Based Validation Layer:** Fine-tuned a domain-specific transformer model (BioClinicalBERT) on 1000+ manually annotated samples for sentence-level classification, enabling filtering of false positives and improving overall precision.

- **Precision-Oriented Threshold Tuning:** Implemented threshold optimisation using out-of-fold (OOF) predictions to calibrate the transformer validation layer, enabling controlled trade-offs between precision and recall and aligning model behaviour with the pipeline’s precision-first design objective.

- **Structured Output Schema Design:** Defined a consistent JSON schema capturing entity spans, entity types, section context, negation status, and validation outputs (classification and confidence scores), ensuring compatibility with downstream ML pipelines.

- **Evaluation Framework:** Evaluated extraction and validation components against a manually annotated ground truth dataset using precision, recall, F1-score, and confusion matrices to quantify performance and validate expected behaviour.

- **Full-Corpus Execution Pipeline:** Executed the complete pipeline across the full dataset to generate large-scale structured outputs, demonstrating system scalability and robustness.

- **Deployment Inference System:** Built and deployed a production-style API using FastAPI (serving layer), Uvicorn (ASGI server), Docker (containerisation), and Cloud Run (serverless hosting) to enable real-time inference.

- **CI/CD Automation Pipeline:** Implemented GitHub Actions–based deployment integrating Git LFS model handling, Cloud Build image creation, and automated Cloud Run deployment, ensuring reproducible and version-controlled updates.

---

## 3. Pipeline Overview

* How pipeline works
* Components and flow
* What each part does
* Output format

No justification

### 3.1 End-to-End Pipeline

### 3.2 Hybrid Pipeline Structure

The system follows a hybrid extraction architecture:

```text
Raw Clinical Note
   ↓
Preprocessing
        ↓
Structural parsing
(Section Detection + Sentence Segmentation)
        ↓
Rule-based candidate generation
        ↓
Transformer-based validation
(threshold tuning for precision)
        ↓
Structured JSON entity output
```

The two core intelligence layers have separate responsibilities:

Component

Role

Rule-based extraction

Defines the search space and extracts candidate spans deterministically

Transformer validation

Defines the decision space and classifies whether candidates are clinically valid in context

Rules generate candidate spans, not final truths. The transformer does not perform extraction; it validates candidates using sentence-level context.

---

### 2. System Overview (High-Level Architecture)

## 1. Pipeline Objective and Structure

- The pipeline is designed to transform unstructured clinical text into a structured, entity-level dataset suitable for downstream analysis and modelling.
- This requires extracting clinically relevant entities and validating them to produce a dataset that is both usable and reliable.

The system follows a two-stage architecture:

> High-recall extraction → precision-oriented validation

- The rule-based stage generates a broad set of candidate entities (maximising coverage)
- The transformer stage filters these candidates to improve correctness

This design reflects a deliberate separation between coverage (recall) and quality control (precision).

#### 2.1 Pipeline Structure and Design Philosophy

**Pipeline Overview**
1. Section-aware preprocessing and segmentation  
2. Rule-based candidate generation (deterministic extraction)  
3. Transformer-based validation (contextual classification)  

**Core Design Philosophy**

The system is explicitly designed as a hybrid pipeline, separating candidate generation from contextual interpretation.

- **Rule-based extraction**
  - Defines the search space (what can be extracted)
  - Generates candidate spans, not final truths
  - Ensures deterministic, auditable, and schema-constrained outputs
  - Behaviour varies by entity type:
    - `SYMPTOM`: high-precision extraction  
    - `INTERVENTION`: moderate-precision, recall-aware candidate generation  
    - `CLINICAL_CONDITION`: high-recall candidate generation  

- **Transformer-based validation**
  - Defines the decision space (what is clinically valid)
  - Performs contextual classification of candidate validity  
  - Resolves ambiguity in:
    - Intent (performed vs planned)
    - Temporality (acute vs historical)
    - Context (current vs background)
  - Does not perform extraction, only validation

**Separation of Responsibilities**

- Rules → *where to look* (bounded, deterministic candidate generation)  
- Transformer → *what it means* (contextual clinical interpretation)  

This separation ensures:
- Deterministic and reproducible extraction  
- Strict adherence to schema boundaries  
- Clear separation of failure modes:
  - Rule failures → recall limitations (missed candidates)  
  - Transformer failures → precision limitations (misclassification)  

**Key Design Principle**

The system is not uniformly precision-first at the rule level (like traditional rule-based NLP), it is designed to produce high-precision final outputs (hybrid pipeline)

- Precision is enforced at the system level, not necessarily at the extraction stage  
- For entities with high linguistic variability (`INTERVENTION`, `CLINICAL_CONDITION`):
  - Broader candidate generation is required  
  - Precision is recovered downstream via transformer validation  

Entity-wise Responsibility Split

| Entity Type   | Rule Strength | Transformer Role        |
|---------------|--------------|------------------------|
| `SYMPTOM`       | Strong       | Refinement             |
| `INTERVENTION`  | Moderate     | Filtering              |
| `CLINICAL_CONDITION`  | Weak         | Primary classification |

**Key Interpretation**

- **`SYMPTOM`**
  - Rules capture most valid cases  
  - Transformer corrects negation and contextual ambiguity  

- **`INTERVENTION`**
  - Rules act as broad candidate generators  
  - Transformer determines whether an action was actually performed  
  - Handles intent vs execution  

- **`CLINICAL_CONDITION`**
  - Rules prioritise recall over precision  
  - Transformer performs primary classification  
  - Distinguishes acute vs historical vs resolved conditions  
---

---

# 4. Data Layer: Corpus Construction & Structural Validation

## 4.1 Data Source: MIMIC-IV (v3.1)

- **Overview:** Medical Information Mart for Intensive Care (MIMIC)-IV database is comprised of deidentified patient electronic health records, taken from PhysioNet.org.
- **Contents:** structured data and free-text clinical notes split into `hosp` module (hospital admissions data) and `icu` module (ICU admissions data)
- **Patients:** `icu` module contains 65,366 unique patients over 94,458 ICU stays
- **Datasets used:** `ICUSTAYS.csv`, `NOTEEVENTS.csv`, `PATIENTS.csv`

##
## 4.2 Pipeline Overview 

Constructed a filtered ICU clinical note corpus using deterministic filtering logic suitable for downstream NLP extraction.

```text
                 Raw MIMIC-IV Data
         (PATIENTS, ICUSTAYS, NOTEEVENTS)
                        │
                  NOTEEVENTS.csv 
         (2,083,180 reports, 61,532 stays)
                        │
                 build_corpus.py
    (Filtering logic using ICUSTAYS and PATIENTS)
                        │
                        ▼
                 icu_corpus.csv 
        (162,296 reports, 32,910 ICU stays)
```

##
## 4.3 Cohort Definition & Filtering

### Data Sources and Columns

| Data Source | Key Columns |
------------| ------------
| `PATIENTS` | `SUBJECT_ID`, `DOB`, `GENDER` |
| `ICUSTAYS` | `SUBJECT_ID`, `HADM_ID`, `ICUSTAY_ID`, `FIRST_CAREUNIT`, `INTIME`, `OUTTIME` |
| `NOTEEVENTS` | `SUBJECT_ID`, `HADM_ID`, `CHARTTIME`, `CATEGORY`, `ISERROR`, `TEXT` |

##
### Rule-Based Filtering Logic

Cohort Enforcement:

- ICU stay (`ICUSTAY_ID`) is used as the cohort anchor. 
- Notes are linked to valid ICU stays via `SUBJECT_ID` + `HADM_ID`.

Population Constraints:

- **Adult patients:** `AGE ≥ 18`
- **ICU types:** `MICU`, `SICU`, `CCU`, `TSICU`, `CSRU`
- **Minimum ICU stay:** `≥ 24 hours`

Note Selection:

- **Included categories:** physician, nursing, nursing/other
- **Excluded:** Radiology, ECG, discharge summaries, administrative notes

Temporal Filtering:

- Notes restricted to `INTIME ≤ CHARTTIME ≤ INTIME + 24h`
- **Early window:** within 24 hours post-ICU admission
- Ensures early ICU documentation relevant to acute clinical state.

Data Quality Filtering:

- Excluded rows where `ISERROR` = 1
- Removed notes with explicit error flags to ensure data integrity

##
## 4.4 Final Corpus Output

Filtered corpus contains:

- **Total notes:** 162,296
- **ICU stays:** 32,910
- **Mean notes per stay:** ~4.9
- ~72.7% of adult ICU stays contain ≥1 early qualifying note

Each row represents a single clinical note with 10 columns:

- **Patient identifiers:** `SUBJECT_ID`, `HADM_ID`, `ICUSTAY_ID`
- **Demographics:** `AGE`, `GENDER`
- **ICU metadata:** `FIRST_CAREUNIT`, `LOS_HOURS`, `CATEGORY`
- **Timestamp:** `CHARTTIME`
- **Raw clinical text:** `TEXT`

Saved to `data/processed/icu_corpus.csv` for downstream NLP processing.

##
## 4.5 Structural Profiling (Feasibility Validation)

A combination of manual inspection (n=30) and quantitative sampling (n=500), including extreme boundary inspection (n=45) was performed to verify that the corpus supports deterministic NLP extraction:

- Notes exhibit consistent section-based structure (e.g. colon headers, system-based sections)
- Numeric clinical data (e.g. vitals, labs) is highly prevalent
- Artefacts follow predictable patterns ( e.g. `[** ... **]` de-identification markers, EMR artefacts, Javascript/link fragments)
- Structural variability exists but remains bounded

These findings provided positive structural confirmation that rule-based candidate extraction was feasible and informed the design of the downstream NLP pipeline.

---

# 5. Preprocessing Layer

## 5.1 Preprocessing Overview

The preprocessing stage performs minimal, deterministic normalization of ICU clinical notes to stabilise text for downstream structural parsing and rule-based extraction.
Transformations are strictly limited to removing artefacts identified in Phase 1 that interfere with parsing, while preserving clinical meaning, numeric content, and document structure.

##
## 5.2 Implementation

The preprocessing pipeline `preprocessing.py` applies the following steps:

1. **Newline Normalisation**  
   Standardises line breaks (`\r`, `\r\n` → `\n`) to ensure consistent line-based parsing for section detection.

2. **De-identification Removal**  
   Removes all `[** ... **]` tokens. These are systematic artefacts introduced during de-identification and do not contribute to clinical content. Minor sentence disruption may occur but does not affect downstream extraction.

3. **Whitespace Normalisation**  
   Collapses multiple spaces and tabs into a single space to stabilise token alignment and prevent inconsistencies during rule-based matching.

4. **Removal of EMR Trailing Artefacts**  
   Strips non-clinical end-of-document content (e.g., `References` sections, JavaScript fragments, or EMR metadata) while preserving all preceding clinical text.

##
## 5.3 Design Rationale

- **Minimality:** Only transformations that improve parsing stability are applied to avoid altering clinical semantics 
- **Structural Preservation:** Section headers, numeric data, and narrative flow remain intact  
- **Determinism:** Output is fully reproducible and consistent across runs  
- **Extraction Compatibility:** Output format is optimised for downstream segmentation and rule-based extraction  

Preprocessing was manually validated on a representative sample to confirm that artefacts are removed while preserving structural integrity and extractable clinical content.

---

# 6. Structural Parsing Layer

## 6.1 Section Detection & Extraction

### Overview

Section detection converts raw ICU clinical notes into structured narrative sections using deterministic header-based parsing. Clinical documentation follows semi-structured formats (e.g., *HPI*, *Assessment*, *Plan*), and isolating these sections enables more precise and context-aware downstream extraction.

##
### Approach

Section detection is implemented as a deterministic, line-based parsing process (`section_extraction.py`) with the following design:

1. **Canonical Header Set**  
   A curated set of 13 high-frequency narrative section headers is used to define valid section boundaries.
    ```text
    Plan
    Assessment
    Action
    Response
    Assessment and Plan
    Chief Complaint
    HPI
    Past medical history
    Family history
    Social History
    Review of systems
    Physical Examination
    Disposition
    ```
2. **Canonical-Only Detection**  
   Only headers in the predefined set are treated as structural boundaries.  
   Non-canonical header-like patterns (e.g., vitals, labs, system labels such as `HR`, `Cardiovascular`) are ignored and treated as normal text.

3. **Flexible Header Matching**  
    - Colon-terminated headers (`Plan:`)
    - Inline headers with content (`Chief Complaint: Chest pain`)
    - Standalone headers (`HPI`)

4. **Case Normalisation**  
   All headers are matched case-insensitively and stored in canonical lowercase form.

##
### Extraction Logic

The section extraction algorithm implemented in `section_extraction.py` processes each clinical note sequentially using a deterministic line-based parsing strategy with two functions: `match_canonical_header()` and `extract_sections()`.

1. Notes are processed line-by-line using newline separation
2. When a canonical header is detected:
   - A new section is started
   - Any inline content is captured
3. All subsequent lines are assigned to the current section until the next header or end of document is reached
4. Structured output is a dictionary:
   - **Keys:** canonical section names (lowercase)
   - **Values:** concatenated section text

##
### Design Rationale

Initial broad header detection led to over-segmentation due to the presence of subsection labels and embedded structured data within notes.

Restricting boundaries to a curated canonical header set ensures preservation of complete narrative sections and robustness to noisy, real-world clinical formatting.
This approach prioritises structural reliability and downstream extraction performance over exhaustive header coverage.

Section detection was validated on a representative sample to confirm accurate boundary identification, preservation of narrative content, and zero-extraction rate.

##
## 6.2 Sentence Segmentation

### Overview

Sentence segmentation operates on extracted section text to produce sentence-level units with precise character offsets, enabling deterministic span-based entity extraction while preserving structural context.

##
### Approach

Sentence segmentation is implemented using a deterministic pipeline (`sentence_segmentation.py`) with the following design:

- **Post-section segmentation**  
  Applied after section extraction to ensure sentence boundaries are defined within meaningful clinical contexts.

- **Deterministic tokenization (NLTK Punkt)**  
  The NLTK Punkt tokenizer is used to split text into sentences. This approach is robust to clinical text characteristics such as abbreviations, irregular punctuation, and dense numeric content, while remaining lightweight and reproducible.

- **Offset-preserving mapping**  
  Each sentence is mapped back to its original position within the section text using a cursor-based search, ensuring exact character-level alignment.

##
### Extraction Logic

The sentence segmentation algorithm `sentence_segmentation.py` calls `extract_sections()` and `sent_tokenize()` in this workflow:

1. `extract_sections()` is applied to obtain section-level text. Each section is a key-value pair (`header` → `text`).

2. Text is split into sentences using NLTK's `sent_tokenize()` (Punkt tokenizer), which returns a list of sentence strings per section.

3. A cursor-based approach identifies the start and end position of each sentence within the original section text

4. Output is a list of sentence objects:

```json
{
  "sentence": "string",
  "start": 0,
  "end": 0
}
```

Overall this process:

  - Works per section to maintain context
  - Does not modify original text
  - Supports deterministic span alignment for regex-based entity extraction
  - Offsets are relative to section text, not the full note

##
### Design Rationale

Clinical notes contain irregular punctuation, abbreviations, and embedded numeric data, making naive rule-based splitting unreliable. 
A lightweight statistical tokenizer (Punkt) provides stable and sufficiently accurate segmentation without introducing heavy dependencies or requiring domain-specific training.

Sentence segmentation was validated on a representative sample to confirm accurate boundary detection and offset alignment.

---

# 7. Entity Schema Design 

## 7.1 Entity Schema Overview

The entity schema defines the output contract of the pipeline. It constrains extraction to three clinically meaningful entity types and ensures every output contains provenance, contextual information, and transformer validation results.

The schema is designed to support:

- Clear entity boundaries  
- Auditable span-level outputs  
- Compatibility with downstream analysis and modelling  
- Separation between deterministic extraction outputs and transformer validation outputs  

##
## 7.2 Entity Scope

Extraction is limited to three entity types that represent core components of ICU clinical reasoning:

| Entity Type | Clinical Role | Definition | Examples |
|------------|---------------|------------|----------|
| `SYMPTOM` | Patient state | Patient-reported complaints or clinician-observed manifestations | pain, nausea, confusion, agitation |
| `INTERVENTION` | Clinical action | Therapeutic or procedural actions performed on the patient | intubation, line insertion, treatment initiation |
| `CLINICAL_CONDITION` | Disease state | Acute or active pathological conditions during the ICU stay | AKI, sepsis, pneumothorax |

This constrained scope prevents uncontrolled expansion of entity types and keeps the extraction task clinically meaningful, auditable, and tractable.

Extending the schema to additional entity types (e.g. medications, vitals, labs) would require more complex and brittle regex-based extraction logic or ontology mapping, so is intentionally deferred to future work that may incorporate ontology-based methods or more advanced NLP techniques.

##
## 7.3 Entity Boundary Definitions

These definitions ensure that entity outputs remain clinically interpretable and consistent across notes.

| Entity Type | Include | Exclude | Boundary Examples |
|------------|---------|---------|-------------------|
| `SYMPTOM` | Subjective symptoms; observable clinical states | Laboratory values; imaging findings; diagnoses or disease states | `delirium` → `SYMPTOM`; `agitation` → `SYMPTOM`; `hypotension` / `tachycardia` → excluded as vital sign abnormalities |
| `INTERVENTION` | Procedures; treatments initiated during ICU context; continuing, titrating, weaning, stopping, or holding active treatments | Planned or hypothetical actions; recommendations; chronic/background treatments not initiated in ICU context | `started antibiotics` → `INTERVENTION`; `placed on vasopressors` → `INTERVENTION`; `may require intubation` → candidate only, validated downstream |
| `CLINICAL_CONDITION` | New or ongoing clinically significant conditions; acute complications; reasons for ICU admission | Historical conditions; chronic baseline diagnoses without acute change; resolved conditions | `AKI` → `CLINICAL_CONDITION`; `sepsis` → `CLINICAL_CONDITION`; chronic conditions excluded unless acute worsening is indicated |

##
## 7.3 Excluded Entity Types

The following entity types are intentionally excluded from the current schema:

| Excluded Type | Reason for Exclusion |
|--------------|----------------------|
| Medications | Large heterogeneous category requiring ontology support, dose/formulation normalisation, and handling of chronic medications, allergies, and plans |
| Vital signs | Highly variable formatting in text and already available in structured EHR data |
| Laboratory values | Complex value/unit/reference-range structure and already captured in structured EHR datasets |

These exclusions keep the schema focused on narrative clinical information that benefits from text extraction, rather than duplicating data already available in structured form.

##
## 7.4 JSON Output Schema

Each extracted entity is represented as one JSON object.

```json
{
  "note_id": "string",
  "subject_id": "string",
  "hadm_id": "string",
  "icustay_id": "string",

  "entity_text": "string",
  "concept": "string",
  "entity_type": "SYMPTOM | INTERVENTION | CLINICAL_CONDITION",

  "char_start": 0,
  "char_end": 0,
  "sentence_text": "string",
  "section": "string",

  "negated": true | false | null,

  "validation": {
    "is_valid": true | false,
    "confidence": 0.0,
    "task": "symptom_presence | intervention_performed | clinical_condition_active"
  }
}
```

| Group | Field | Purpose |
|------|-------|---------|
| Metadata | `note_id`, `subject_id`, `hadm_id`, `icustay_id` | Links each entity to the source note, hospital admission, and ICU stay |
|Extraction | `entity_text`, `concept`, `entity_type` | Captures the extracted surface span, normalised clinical concept, and entity category |
| Provenance | `char_start`, `char_end`, `sentence_text`, `section` | Preserves exact text span (source location) and contextual clinical information for auditability and downstream analysis |
| Rule-Derived Signal | `negated` | Captures whether the entity is negated in the text, where applicable |
| Transformer Validation | `validation` (`is_valid`, `confidence`, `task`) | Stores trnsformer-based contextual validation, including binary validity judgement, confidence score, and task type |

##
## 7.5 Schema Design Decisions

- One JSON object is generated per entity; a single note may contain multiple entities.
- `entity_text` preserves the exact extracted surface form from the note.
- `concept` stores the normalised clinical meaning mapped from rule-based extraction.
- `char_start` and `char_end` preserve exact span-level provenance.
- `sentence_text` provides local context for auditing and transformer validation.
- `section` records the structural region of the note where the entity was found.
- Validation outputs are nested separately from rule-derived extraction fields to preserve separation between candidate generation and contextual validation.

---

# 8. Rule-Based Extraction Layer

## 8.1 Extraction Overview

The rule-based extraction layer performs deterministic candidate generation. It identifies clinically relevant text spans, maps surface forms to normalised clinical concepts, and attaches sentence/section provenance for downstream transformer validation.

This layer does not produce final clinical truth labels. Instead, its high-recall approach produces schema-aligned candidate entities that are subsequently validated by the transformer layer.

The extraction layer is responsible for:

- Identifying exact text spans using regex-based rules  
- Mapping lexical variants to normalised clinical concepts  
- Preserving character offsets for traceability  
- Attaching sentence and section context  
- Applying lightweight rule-derived signals where appropriate  
- Producing structured candidate entities compatible with the JSON schema  

The layer is intentionally constrained. It focuses on high-yield, clinically meaningful patterns rather than exhaustive ontology coverage or full linguistic modelling.

| Entity Type | Rule Behaviour | Validation Requirement |
|------------|----------------|------------------------|
| `SYMPTOM` | Strong pattern coverage with section restriction and negation handling | Refinement of presence/absence |
| `INTERVENTION` | Broader candidate generation for clinical actions | Filtering planned, hypothetical, or non-performed actions |
| `CLINICAL_CONDITION` | Broad candidate generation for disease-state mentions | Classification of active/current vs historical, resolved, uncertain, or ruled-out conditions |

Contextual false positives are expected at this stage and are handled downstream by the transformer validation layer to restore precision.

Candidate generation explicitly avoids:

- Full linguistic modelling  
- Encoding complex contextual rules  
- Exhaustive ontology or hierarchy construction  
- Dataset-specific over-optimisation  

Each extractor function was manually validated on representative samples before integration to confirm section filtering, span alignment, concept mapping, and expected candidate-generation behaviour.

##
## 8.2 Negation Handling

Negation is implemented as a lightweight rule-derived signal, but it is only used where it directly aligns with the entity-level validation task.

| Entity Type | Validation Question | `negated` Field | Rationale |
|------------|---------------------|-----------------|-----------|
| `SYMPTOM` | Is the symptom present? | `true` / `false` | Negation directly indicates symptom absence or presence |
| `INTERVENTION` | Was the intervention performed? | `null` | Negation alone does not capture planned, suggested, withheld, or hypothetical actions |
| `CLINICAL_CONDITION` | Is the condition active/current? | `null` | Negation alone does not capture historical, resolved, suspected, or chronic conditions |

For symptoms, negation provides a simple and clinically useful signal:

```text
"no chest pain"      → concept = pain, negated = true
"chest pain"         → concept = pain, negated = false
"denies nausea"      → concept = nausea_vomiting, negated = true
"reports dyspnoea"   → concept = dyspnoea, negated = false
```
For interventions and clinical conditions, the same approach is insufficient because the key decision depends on context rather than simple negation:

```text
"intubation planned"     ≠ performed intervention
"may require intubation" ≠ performed intervention
"history of sepsis"      ≠ active condition
"resolved sepsis"        ≠ active condition
```
For this reason, the rule-based layer only stores negation for `SYMPTOM` entities. More complex contextual interpretation, including intent, temporality, uncertainty, and active/current status, is delegated to the transformer validation layer.

##
## 8.3 Symptom Extraction

Symptom extraction identifies patient-reported or clinician-observed manifestations using deterministic, concept-level regex patterns.

### Scope

Symptom extraction is restricted to clinically relevant narrative sections:

- `chief complaint`
- `hpi`
- `review of systems`

These sections contain most subjective symptom language and reduce false positives from assessment, plans, flowsheets, or unrelated structured content.

##
### Concept-Based Pattern Matching

A constrained symptom concept set of 17 common concepts was used to prioritise precision, interpretability, and stable outputs without expanding into broad symptom ontology construction.

```text
pain
headache
chest_discomfort
palpitations
dyspnoea
syncope
nausea_vomiting
fatigue
dizziness
fever
cough
diarrhoea
confusion
bleeding
weakness
seizure
anorexia
```

Symptoms are represented as normalised concepts, where each concept maps to multiple lexical variants captured by regex patterns. The patterns are designed to capture common ICU phrasing rather than all possible symptom expressions.

Example:

```python
SYMPTOM_PATTERNS = {
    "dyspnoea": [
        r"\b(short(ness)? of br(eath)?|sob|dyspn(o)?ea|breathless(ness)?|diff(iculty)? breath(ing|e)?)\b"
    ],
    "syncope": [
        r"\b(syncop(e|al)|faint(ing|ed|s)?|pass(ing|ed)? out|loss of consc(iousness)?|loc)\b"
    ]
}
```

This allows the system to preserve the exact extracted surface text while storing a standardised clinical concept label.

## 
### Negation Handling

A lightweight token-based negation rule is applied to symptom candidates. The rule scans preceding tokens within the same sentence for negation triggers:

```python
NEGATION_TERMS = {"no", "denies", "denied", "without", "not", "negative"}
```
Examples:

```text
"no chest pain" → concept = pain, negated = true
"chest pain" → concept = pain, negated = false
"denies nausea" → concept = nausea_vomiting, negated = true
```

This captures common high-yield negation patterns while avoiding complex syntactic modelling. More complex contextual interpretation is handled later by the transformer validation layer.

## 
### Span Alignment and Provenance

Regex matching is performed at sentence level, but entity outputs must remain traceable to the source text. The extraction logic therefore preserves two alignment steps:

- **Token–character alignment:** matched character spans are mapped to token indices so token-based negation can be applied correctly
- **Global span alignment:** sentence-relative match offsets are converted into section-level character offsets for JSON output

This ensures each extracted entity retains exact span provenance while remaining compatible with negation detection and downstream validation.

##
### Deduplication (Per-Sentence)

A maximum of one instance per concept is extracted per sentence. This prevents duplicate overlapping matches from repeated regex patterns within the same sentence.

The same concept may still appear across different sentences, preserving document-level signal while ensuring sentence-level uniqueness.

##
### Implementation

Symptom extraction is implemented in `symptom_rules.py`, which defines the functions `map_char_to_token()`, `is_negated_simple()`, and `extract_symptoms()`.

The workflow is as follows:

1. **Section Filtering:**
    Input text only processed if in symptom-relevant sections
2. **Sentence Segmentation:**
    Segment section text into sentences using `split_into_sentences()`, preserving start/end character offsets relative to original text
3. **Concept-Level Regex Matching:**
    Each sentence is scanned for regex patterns corresponding to symptom concepts, generating candidate spans with associated normalised concepts
4. **Span Alignment:**
    Convert sentence-level match offsets into section-level character offsets and extract the exact source span.
5. **Token Alignment for Negation:**
    Map matched spans to token indices to support local negation detection.
6. **Negation Detection:**
    Apply local negation logic using `is_negated_simple()` to determine if the symptom is negated within the sentence context assigning `negated = True / False` 
7. **Per-sentence deduplication:**
    Track extracted concepts within each sentence to avoid duplicate candidate entities.
8. **Entity Construction and Output:**
    Return schema-aligned SYMPTOM candidate entities for downstream validation.

Each output includes:

- Extracted span
- Normalised concept
- Character offsets
- Sentence and section context
- Negation status
- Validation task: `symptom_presence`

---

## 8.4 Intervention Extraction

Intervention extraction identifies therapeutic and procedural actions using deterministic, concept-level regex patterns. Unlike symptom extraction, this component is designed primarily as a recall-oriented candidate generation layer: it identifies plausible intervention mentions, while contextual interpretation is deferred to the transformer validation layer.

### Scope

Intervention extraction is restricted to intervention-dense clinical sections:

- `action`
- `assessment`
- `assessment and plan`

These sections were selected because they frequently contain management decisions, active treatments, and clinical actions. Their semantics differ: `action` sections often describe performed interventions, while `assessment` and `assessment and plan` may contain a mixture of performed, planned, historical, or hypothetical interventions.

The rule layer therefore does not determine whether an intervention was actually performed. It generates candidate spans for downstream validation.

##
### Concept-Based Pattern Matching

The intervention concept set contains 19 ICU-focused categories selected for clinical relevance, frequency in ICU documentation, and downstream aggregation utility.

```text
airway_management
oxygen_therapy
mechanical_ventilation
fluid_therapy
vasopressor_inotrope
analgesia
sedation
paralysis
antibiotic_therapy
anticoagulation
blood_product
renal_replacement_therapy
procedure_general
surgical_procedure
nutrition
cardiovascular_support
cardiovascular_drugs
electrolyte_replacement
resuscitation
```

Interventions are represented as normalised clinical action concepts, where each concept maps to multiple lexical variants.

Examples:

```python
INTERVENTION_PATTERNS = {
    "airway_management": [
        r"\b(intubated|intubation|reintubated|extubated|endotracheal tube(s)?|ett(s)?|et tube(s)?|tracheostomy|trach(eostomy)?|trachy|airway secured)\b"
    ],
    "oxygen_therapy": [
        r"\b(oxygen therapy|supplemental oxygen|o2 therapy|nasal cannula(s)?|nc(s)?|non[- ]rebreather(s)?|nrb(s)?|face mask oxygen|venturi(s)?|high[- ]flow oxygen|hfno|hfnc(s)?)\b"
    ],
    "mechanical_ventilation": [
        r"\b(mechanical vent(ilation)?|mv|ventilated|on ventilator|niv|non[- ]invasive vent(ilation)?|cpap|bipap|psv|pressure supp(ort)?|peep)\b"
    ]
}
```

Patterns capture heterogeneous forms of intervention language, including:

- Abbreviations (NC, NGT, IVF)
- Drug or device names (propofol, ETT)
- Procedure terms (intubated, central line)
- Treatment phrases (bolus given, transfused)

This allows the extraction layer to capture common ICU shorthand and varied intervention phrasing without expanding into medication-level ontology construction.

##
### No Trigger-Word Dependency

Intervention extraction does not require action triggers such as given, administered, or performed.

This is intentional because many valid intervention mentions occur without explicit trigger words:

```text
"on propofol"
"intubated"
"ETT in place"
"NC 2L"
```

Trigger-word rules may increase precision but would substantially reduce recall and introduce brittle phrase-level logic. Instead, the rule layer preserves plausible intervention candidates and leaves performed/planned/hypothetical interpretation to the transformer validation layer.

##
### No Semantic Filtering at Rule Stage

The intervention rules do not attempt to determine whether an intervention is:

- Performed
- Planned
- Recommended
- Historical
- Hypothetical
- Negated

These distinctions require contextual reasoning about intent and temporality. The rule-based layer therefore keeps intervention extraction lexical and span-focused, while assigning the validation task `intervention_performed`.

##
### Span Alignment and Deduplication

Extraction is performed at sentence level. For each match:

- Sentence-relative offsets are converted to section-level character offsets
- Exact matched text is preserved as `entity_text`
- Section and sentence context are attached for downstream validation

Deduplication is limited to exact duplicate spans with the same:

- Start offset
- End offset
- Concept label

Unlike symptoms, there is no concept-level per-sentence deduplication. Multiple intervention mentions may be clinically meaningful and are preserved as separate candidate entities.

Example:

```text
"on propofol, fentanyl, and midazolam"
```

This may generate multiple intervention candidates because each span represents a distinct lexical signal.

##
### Implementation

Intervention extraction is implemented in `intervention_rules.py`, using the main function`extract_interventions()`:

The workflow is:

1. **Section Filtering:**
    Process only intervention-relevant sections: action, assessment, and assessment and plan.
2. **Sentence Segmentation:**
    Split section text into sentence-level units while preserving character offsets.
3. **Concept-Level Regex Matching:**
    Apply regex patterns from `INTERVENTION_PATTERNS` using `re.finditer()` to capture all non-overlapping matches.
4. **Span Extraction and Alignment:**
    Convert sentence-level match offsets into section-level character offsets and preserve the exact source span.
5. **Exact Span Deduplication:**
    Remove only identical duplicate matches with the same start offset, end offset, and concept.
6. **Entity Construction:**
    Return schema-aligned `INTERVENTION` candidate entities for downstream validation.

Each output includes:

- Extracted span
- Normalised concept
- Character offsets
- Sentence and section context
- `negated = null`
- Validation task: `intervention_performed`

---

## 8.5 Clinical Condition Extraction

Clinical condition extraction identifies documented diagnoses, pathological states, and ICU complications using deterministic, concept-level regex patterns. This component is designed as a recall-oriented candidate generation layer: it captures plausible condition mentions while deferring contextual interpretation to the transformer validation layer.

### Scope

Clinical condition extraction is restricted to high-yield diagnostic sections:

- `assessment and plan`
- `assessment`
- `hpi`
- `chief complaint`

These sections frequently contain diagnoses, admitting problems, active complications, and clinical summaries. `HPI` and `chief complaint` may introduce additional noise from historical or uncertain conditions, but excluding them would reduce recall for early ICU diagnostic context.

##
### Concept-Based Pattern Matching

Clinical conditions are represented as high-level diagnostic concepts rather than individual diseases. Each concept maps to multiple lexical variants, abbreviations, and shorthand expressions.

The clinical condition concept set contains 13 ICU-relevant categories:

```text
infection
shock
respiratory
cardiovascular
arrhythmia
renal_failure
neurological
bleeding
gastrointestinal
metabolic
hepatic_failure
cardiac_arrest
vascular
```

Examples:

```python
CLINICAL_CONDITION_PATTERNS = {
    "infection": [
        r"\b(sep(sis|tic)|infect(ed|ion)|bacter(a)?emia|pneumonia(s)?|urinary tract infection|uti|(endo|myo|peri)carditis|meningitis)\b"
    ],
    "shock": [
        r"\b((septic|cardiogenic|hypovol(a)?emic|distributive|hypotensive|neurogenic) shock|anaphyla(xis|ctic))\b"
    ],
    "respiratory": [
        r"\b(resp(iratory)? failure|acute respiratory distress syndrome|ards|hypox(a)?emi(a|c)|hypercapni(a|c)|pneumothorax|h(a)?emothorax|pleural effusion|pulmonary (o)?edema|aspiration pneumonitis)\b"
    ]
}
```

These concepts are intentionally broad diagnostic categories. They capture clinically meaningful disease-state signals without attempting exhaustive ontology construction or fine-grained diagnosis coding.

##
### No Contextual Filtering at Rule Stage

The clinical condition rules do not attempt to determine:

- **Temporality:** current vs historical
- **Certainty:** confirmed vs suspected
- **Resolution:** active vs resolved
- **Negation:** present vs absent
- **Attribution:** reason for admission vs background comorbidity

These distinctions require semantic interpretation beyond reliable deterministic rules.

Examples:

```text
"history of sepsis"      → candidate condition, but not necessarily active
"resolved pneumonia"     → candidate condition, but not active
"?PE"                    → candidate condition, but uncertain
"AKI improving"          → candidate condition, likely active/recent
```

The rule layer therefore preserves condition candidates, while the transformer validation layer determines whether the mention represents an active/current clinical condition, where its validation task is `clinical_condition_active`.

##
### Span Alignment and Deduplication

Extraction is performed at sentence level and deduplication is limited to exact duplicate spans, same as interventions.

There is no concept-level or sentence-level deduplication. Multiple mentions of the same condition concept are preserved because they may represent distinct lexical signals, repeated documentation, or clinically relevant emphasis.

Example:

```text
"respiratory failure with ARDS"
```

##
### Implementation

Clinical condition extraction is implemented in `clinical_condition_rules.py`, using the main function `extract_clinical_conditions()`.

The workflow is:

1. **Section Filtering:**
    Process only clinical-condition-relevant sections: assessment and plan, assessment, hpi, and chief complaint.
2. **Sentence Segmentation:**
    Split section text into sentence-level units while preserving character offsets.
3. **Concept-Level Regex Matching:**
    Apply regex patterns from `CLINICAL_CONDITION_PATTERNS` using `re.finditer()` to capture all non-overlapping matches.
4. **Span Extraction and Alignment:**
    Convert sentence-level match offsets into section-level character offsets and preserve the exact source span.
5. **Exact Span Deduplication:**
    Remove only identical duplicate matches with the same start offset, end offset, and concept.
6. **Entity Construction:**
    Return schema-aligned `CLINICAL_CONDITION` candidate entities for downstream validation.

Each output includes:

- Extracted span
- Normalised concept
- Character offsets
- Sentence and section context
- `negated = null`
- Validation task: `clinical_condition_active`

##
## 8.6 Integrated Extraction Pipeline

After the three rule-based entity extractors were implemented, they were combined into a single deterministic extraction pipeline.

The integrated pipeline is implemented in `run_extraction_pipeline.py` and orchestrates the full candidate generation flow:

```text
             ICU corpus random sample 
                (10,000 reports)
                        │
                  Preprocessing
                        │
               Section extraction
                        │
              Sentence segmentation
                        │
               SYMPTOM extraction
            INTERVENTION extraction
          CLINICAL_CONDITION extraction
                        │
                        ▼
          Flat JSONL candidate dataset 
                (47,487 entities)
```

For each note, the script:

1. Loads note metadata and raw text, create a unique `note_id` for provenance
2. Applies preprocessing and structural parsing
3. Runs all three entity-specific extraction functions
4. Aggregates candidate entities across sections and sentences
5. Writes one JSON object per candidate entity to a flat .jsonl file

The pipeline was run on a reproducible sample of 10,000 ICU notes from `icu_corpus.csv` to generate `data/interim/extraction_candidates.jsonl`.

This run served two purposes:

- Confirm that the rule-based extraction components worked together end-to-end
- Generate a candidate entity dataset suitable for downstream manual annotation, dataset splitting, and transformer validation

At the end of this layer, the system produces a flat JSONL file of span-aligned candidate entities. Each candidate contains:

- Metadata (`note_id`, `subject_id`, `hadm_id`, `icustay_id`)
- Entity text (`entity_text`)
- Normalised concept (`concept`)
- Entity type (`entity_type`)
- Section/sentence provenance (`sentence_text`, `section`)
- Character offsets (`char_start`, `char_end`)
- Rule-derived negation signal (`negated`)
- Task-specific `validation` placeholders:
    - Binary classification (`is_valid = null`)
    - Confidence score (`confidence = 0.0`)
    - Entity specific task (`task`)

The output at this stage contains candidate entities only. Final clinical validity is determined later by the transformer validation layer.

---

# 9. ML Modelling Strategy

## 9.1 Validation Task Formulation

The rule-based extraction layer generates candidate entities, but these candidates are not treated as final clinical outputs. Many extracted spans require contextual interpretation before they can be considered valid.

The machine learning task is therefore **entity-level contextual validation**.

Input to the model:

- `sentence_text` → local clinical context  
- `entity_text` → extracted candidate span  
- `entity_type` / `task` → defines what validity means  

Output from the model:

- `is_valid` → binary decision  
- `confidence` → probability score  
- `task` → entity-specific interpretation  

This is not a sequence-labelling or generative extraction task. The model does not identify new spans. It validates candidate entities already generated by the deterministic extraction layer.

| Entity Type | Validation Task | Positive Label | Negative Label |
|------------|-----------------|----------------|----------------|
| `SYMPTOM` | `symptom_presence` | Symptom is present/current in context | Negated, absent, historical/background, or not currently present |
| `INTERVENTION` | `intervention_performed` | Intervention was performed or is active/currently in use | Planned, suggested, hypothetical, withheld, or not performed |
| `CLINICAL_CONDITION` | `clinical_condition_active` | Condition is active/current in context | Historical, resolved, uncertain, ruled out, or background-only |

Examples:

```text
"no chest pain"              → SYMPTOM candidate invalid
"plan to start antibiotics"  → INTERVENTION candidate invalid
"history of MI"              → CLINICAL_CONDITION candidate invalid
"intubated overnight"        → INTERVENTION candidate valid
"acute renal failure"        → CLINICAL_CONDITION candidate valid
```

The validation task is therefore focused on contextual correctness rather than span detection.

##
## 9.2 Why Validation Is Separate from Extraction

The pipeline deliberately separates extraction from validation:

| Layer | Responsibility |
|-------|----------------|
| Rule-based extraction | Finds candidate spans deterministically |
| Transformer validation | Determines whether candidates are clinically valid in context |

This separation is important because clinical entity mentions often require contextual reasoning that is difficult to encode safely using rules alone.

Rule-based extraction can identify candidate spans such as:

```text
"intubation"
"sepsis"
"chest pain"
```

But it cannot reliably determine contextual validity for all candidate entities. In particular, it cannot reliably distinguish:

- **Intent:** performed or planned interventions
- **Uncertainty:** confirmed or suspected diagnoses
- **Negation:** present or absent symptoms / conditions
- **Temporality:** clinically active or resolved conditions
- **Attribution:** reason for admission vs background comorbidity

For example:

```text
"intubation planned"     → candidate intervention, but not performed
"history of sepsis"      → candidate condition, but not active
"no chest pain"          → candidate symptom, but negated
"possible pneumonia"     → candidate condition, but uncertain
```

Attempting to encode these distinctions entirely through regex rules would require increasingly complex logic. This would make the system brittle, difficult to maintain, and prone to dataset-specific overfitting.

The validation model therefore acts as a second-stage semantic filter:

> High-recall candidate generation → precision-oriented contextual validation

This allows the pipeline to retain deterministic span extraction while using machine learning only where sentence-level interpretation is required.

##
## 9.3 Why Not End-to-End Model Extraction

A fully model-based extraction system, such as supervised named entity recognition (NER), could in principle perform span detection and entity classification together. This was not selected for this project because the system requires controlled, auditable, schema-aligned outputs from clinical text under limited annotation constraints.

| Requirement | Constraint for End-to-End Model Extraction |
|-------------|---------------------------------------------|
| Exact span traceability | Predicted spans may vary in boundary selection and require additional alignment checks |
| Schema control | The model must learn entity boundaries and entity definitions simultaneously |
| Auditability | It is harder to inspect why a span was extracted, missed, or assigned a specific label |
| Debuggability | Span detection errors and contextual validity errors become entangled |
| Labelled data requirement | Supervised NER usually requires larger token-level annotated datasets |
| Clinical reliability | Unconstrained or weakly constrained extraction is less suitable for audit-sensitive clinical pipelines |

The hybrid approach is more appropriate for this project because:

- Rules constrain the extraction space  
- Entity spans remain deterministic and auditable  
- The output schema remains fixed  
- The transformer is used only for contextual validation  
- Failure modes are easier to separate and debug  

This design does not reject supervised NER as a general clinical NLP method. Instead, it reflects the project’s specific constraints: limited labelled data, need for traceable span-level outputs, and a precision-oriented downstream validation workflow.

##
## 9.4 Model Classes Considered

The validation model must classify candidate entities using full sentence context while remaining efficient, reproducible, and suitable for batch inference.

| Approach | Use Case | Strengths | Limitations |
|----------|-----------|-----------|--------------|
| Rule-based validation | Add more regex/context rules to classify candidates | Interpretable and deterministic | Brittle for temporality, intent, uncertainty, and complex negation |
| Classical ML (LR/SVM) | Classify candidates using engineered lexical features | Simple, fast, low compute | Requires manual feature engineering; weak semantic/contextual understanding |
| CNN text classifier | Learn local phrase patterns around the entity | Captures local phrase patterns, efficient; good for local n-grams | Limited modelling of long-range context and scope |
| RNN/LSTM | Process sentence sequentially | Sequential modelling (captures sequence order) | Slower, less parallelisable, weaker than transformers for long-range context |
| Transformer encoder | Classify using contextual sentence representations | Full-context attention; structured probabilities; strong sentence-level semantics | Higher computational cost than classical models, but manageable |
| Generative LLM | Prompt model to decide validity | Flexible reasoning and prompting, expressive | Unnecessary for binary validation; higher cost; slower inference; requires extra output parsing/schema enforcement |

The selected model class must handle clinical sentences such as:

```text
"Patient denies chest pain but reports worsening shortness of breath."
"Will consider intubation if respiratory status deteriorates."
"History of sepsis, now admitted with acute renal failure."
```

These require interpretation across the sentence rather than simple keyword matching.

##
## 9.5 Why Transformer Encoders

Transformer encoders are selected because the validation task depends on contextual meaning.

Encoder models use self-attention to model relationships between tokens across the full input sequence. This is useful for clinical validation, where the same extracted span can be valid or invalid depending on surrounding words.

| Contextual Challenge | Example | Why Context Matters |
|---------------------|---------|--------------------|
| Negation | `no chest pain` | Entity span exists but symptom is absent |
| Mixed polarity | `denies chest pain but reports SOB` | Different entities in the same sentence have different validity |
| Intent vs execution | `planned intubation` vs `intubated` | Same intervention concept, different clinical status |
| Temporality | `history of stroke` vs `acute stroke` | Same condition mention, different activity status |
| Uncertainty | `?PE` or `possible sepsis` | Candidate may not represent confirmed active disease |
| Attribution | `admitted with pneumonia` vs `PMHx of pneumonia` | Same condition, different relevance to current admission |

Compared with classical ML or manually engineered rules, transformer encoders reduce the need to explicitly encode every contextual pattern. They learn task-specific decision boundaries from labelled examples while still producing structured binary outputs.

##
## 9.6 Encoder-Based Models vs Generative LLMs

The validation task is a structured classification problem, not a free-text generation problem. The required output is a binary validity decision with a confidence score, which aligns more naturally with encoder-based classifiers than generative LLMs.

| Requirement | Encoder Model (BERT-style) | Generative LLM |
|-------------|----------------------------|----------------|
| Output format | Fixed logits/probabilities from a classification head | Generated text tokens requiring parsing |
| Task alignment | Directly optimised for binary classification | Requires prompting to simulate classification behaviour |
| Inference efficiency | Supports efficient batch inference | Slower due to sequential token generation |
| Schema control | Output structure is fixed by the model head | Requires additional schema enforcement and output validation |
| Threshold tuning | Directly supports probability-based threshold tuning | Less straightforward because outputs are generated text or prompt-dependent scores |
| Reproducibility | Stable fixed-model inference under controlled settings | More sensitive to prompt design, decoding configuration, and model/version changes |
| Evaluation | Standard classification metrics apply directly | Requires additional parsing logic before classification metrics can be applied |
| Operational complexity | Simpler deployment and validation pathway | Adds prompting, parsing, validation, and error-handling layers |

LLMs are powerful for summarisation, question answering, and generative clinical text tasks. For this pipeline, however, they introduce unnecessary complexity because the required output is a high-throughput, schema-constrained validity decision.

An encoder classifier is therefore simpler, faster, easier to evaluate, and better aligned with structured validation of extracted entities.

##
## 9.7 Why Fine-Tuning Rather Than Training From Scratch

The objective is not to learn clinical language from first principles, but to adapt an existing clinical language model to a specific validation task using a limited manually annotated dataset. Fine-tuning is therefore more appropriate than training from scratch.

| Strategy | Data Requirement | Compute Requirement | Suitability |
|---------|------------------|---------------------|---------|
| Training from scratch | Very high | Very high | Not appropriate for this project |
| Fine-tuning pretrained model | Moderate | Manageable | Appropriate for task-specific validation |

Training a transformer from scratch would require large text corpora for self-supervised pretraining, substantial computational resources, and long training time. It would also require careful model architecture design, hyperparameter tuning, and validation to achieve competitive performance.

Fine-tuning a pretrained clinical model is more appropriate because it allows the model to leverage existing clinical language understanding while learning task-specific decision boundaries for entity validation. This is more efficient, requires less data, and is better aligned with the project constraint of limited annotated training data.

##
## 9.8 Candidate Encoder Models and Domain Alignment

Several pretrained encoder models were considered.

| Model | Pretraining Data | Domain Alignment | Strengths | Limitations |
|-------|------------------|------------------|-----------|-------------|
| General BERT | Wikipedia / BooksCorpus | General-domain | Efficient, well-supported | Weak clinical language understanding |
| PubMedBERT | PubMed abstracts / biomedical literature | Biomedical literature | Strong biomedical terminology | Less aligned with clinical note structure and ICU shorthand |
| ClinicalBERT variants | Clinical notes / clinical corpora | General clinical text | Clinical-domain adaptation | Variant-dependent; less specifically matched than BioClinicalBERT selected here |
| BioClinicalBERT | BioBERT initialisation + MIMIC-III clinical notes | Clinical notes / ICU documentation | Strong alignment with clinical notes, abbreviations, and shorthand | More domain-specific than general biomedical models |

Clinical notes differ substantially from general-domain text. They contain:

- Abbreviations: `SOB`, `PRBC`, `NGT`, `AKI`, `ETT`, `HFNC`
- Shorthand syntax: `c/o`, `s/p`, `abx`
- Fragmented sentence structure
- Domain-specific vocabulary: `intubation`, `vasopressor`, `sepsis`, `ARDS`
- ICU-specific phrasing
- Non-standard grammar and punctuation

General-domain language models are trained primarily on non-clinical corpora and may not represent these patterns well. A clinical-domain pretrained model is better aligned with the input distribution because it has already learned representations of clinical vocabulary, abbreviations, and documentation style.

This reduces the amount of labelled data required for downstream fine-tuning because the model already encodes relevant language structure. Therefore, closer alignment between the pretrained model domain and target clinical notes is expected to improve performance under limited annotation conditions.

##
## 9.9 Final Model Choice: BioClinicalBERT

The final selected model is [**BioClinicalBERT**](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), fine-tuned as a supervised binary classifier.

BioClinicalBERT was initialized from BioBERT and further pretrained on all notes from the MIMIC-III `NOTEEVENTS` table (~880M words), making it closely aligned with the clinical-note style used in this project.

| Requirement | BioClinicalBERT Alignment |
|-------------|-------------------------|
| ICU note language | Further pretrained on MIMIC-III clinical notes |
| Clinical abbreviations | Strong exposure to clinical shorthand and documentation style |
| Sentence-level classification | Encoder architecture supports contextual classification |
| Efficient inference | Suitable for batch classification of candidate entities |
| Reproducibility | Produces stable probability outputs through a classification head |
| Limited labelled data | Clinical pretraining reduces task-specific annotation burden |

Final modelling design:

```text
Input:
  "sentence_text" + "entity_text" + other contextual features

Model:
  BioClinicalBERT encoder
  + binary classification head

Output:
  "is_valid"
  "confidence"
  "task"
```

The final modelling strategy preserves deterministic extraction for span control and uses BioClinicalBERT only for a contextual validation layer. This allows the system to interpret sentence-level context, filter false positives, and convert broad, high-recall candidate extraction into precision-oriented structured outputs while retaining traceability and reproducibility.

---

# 10. Validation Layer: Transformer-Based Classifier

## 10.1 Validation Layer Overview

The transformer validation layer receives candidate entities from the rule-based extraction layer and classifies whether each candidate is clinically valid within its sentence context. This converts the pipeline from deterministic candidate generation into context-aware validated extraction.

The validation model does not extract new spans. It only evaluates candidates already produced by the rule-based layer.

Transformer reliance varies by entity type because the rule-based extractor is not equally reliable across all extraction tasks:

| Entity Type | Rule Strength | Transformer Role |
|------------|---------------|------------------|
| `SYMPTOM` | Strong | Refinement of present vs negated/absent mentions |
| `INTERVENTION` | Moderate | Filtering of planned, hypothetical, or non-performed actions |
| `CLINICAL_CONDITION` | Weak | Primary contextual classification of active/current vs historical/resolved/uncertain conditions |

Interpretation:

- `SYMPTOM`: rules capture many valid cases; the transformer mainly refines negation and contextual ambiguity.
- `INTERVENTION`: rules generate broad action candidates; the transformer determines whether the action was actually performed.
- `CLINICAL_CONDITION`: rules capture broad disease-state mentions; the transformer distinguishes active/current conditions from historical, resolved, uncertain, or ruled-out mentions.

This asymmetry reflects the differing contextual complexity of each entity type and justifies using the transformer as a validation layer rather than as a uniform extractor.

The validation layer was developed through the following workflow:

```text
                  Rule-generated candidate entities
                                  │
                                  ▼
                     Balanced manual annotation
                      1,200 labelled candidates
                                  │
                                  ▼
                     Stratified train/test split
                      Train: 1,020 | Test: 180
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │ Model selection on training set │
                 │   Stratified 5-fold cross-val   │
                 └────────────────┬────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            Stable baseline config     Advanced tuned config
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                      Select best configuration
                                  │
                                  ▼
                  Generate out-of-fold probabilities
                     using selected configuration
                                  │
              Threshold optimisation on OOF predictions
                    precision-biased decision rule
                                  │
                                  ▼
               Final BioClinicalBERT training on full
                      1,020-sample training set
                                  │
                                  ▼
                    Save final model + tokenizer
```

##
## 10.2 Annotation Dataset Construction

### Implementation Overview

The transformer validation model was trained using candidate entities generated by the integrated rule-based extraction pipeline. These candidates were sampled into an annotation-ready dataset for supervised binary classification.

The final labelled dataset contained **1,200 manually annotated candidate entities**, sampled evenly across the three entity types:

| Entity Type | Number of Candidates | Validation Task |
|------------|----------------------|-----------------|
| `SYMPTOM` | 400 | `symptom_presence` |
| `INTERVENTION` | 400 | `intervention_performed` |
| `CLINICAL_CONDITION` | 400 | `clinical_condition_active` |

The dataset construction process is implemented in `sample_entities.py` and `sample_additional_entities.py`.

```text
    extraction_candidates.jsonl
                │
    Flatten candidate entities
                │
  Balanced sampling by entity type
      (400 per entity type)
                │
        Annotation-ready CSV
                │
                ▼
     Manual `is_valid` labels
```

Balanced sampling prevents the validation model from being dominated by the most frequent entity type in the extraction output and ensures that each validation task contributes meaningfully to training and evaluation.

The expanded dataset was constructed by combining the original annotation sample with an additional non-overlapping sample. Previously sampled candidates were excluded before additional sampling using the fields used by the transformer input: `sentence_text`, `entity_text`, `entity_type`, `concept`, and `task`.

##
### Retained Fields

Candidate entities were flattened from JSONL into tabular format while preserving the fields required for annotation, training, evaluation, and downstream reintegration.

| Field | Purpose |
|------|---------|
| `note_id` | Links candidate back to source note for traceability |
| `section` | Records section-level context |
| `concept` | Stores normalised clinical concept |
| `entity_text` | Exact extracted candidate span |
| `entity_type` | Defines candidate category |
| `sentence_text` | Provides sentence-level context for validation |
| `negated` | Retains rule-derived symptom negation signal for comparison |
| `task` | Defines entity-specific validation objective |
| `confidence` | Placeholder for model-generated probability scores | 
| `is_valid` | Manually annotated binary ground-truth label |

Only `is_valid` was manually annotated. All other fields were preserved from the upstream extraction pipeline to maintain provenance and schema consistency.

##
### Annotation Output Files

Two annotation file types were generated:

| File Type | Purpose |
|-----------|---------|
| Raw sampled annotation files | Reproducible sampled datasets before annotation; no changes made |
| Labelled annotation files | Sampled datasets containing manually annotated `is_valid` labels |

This separation preserved the original sampled entities while protecting manual annotation work from accidental overwrite.

##
## 10.3 Manual Annotation Framework

### Annotation Objective

Manual annotation was performed to create binary ground-truth labels for transformer fine-tuning. The annotated field was:

| Field | Meaning |
|------|---------|
| `is_valid` | Whether the extracted candidate entity is clinically valid in its sentence context |

Each candidate was labelled as `True` or `False` using the full `sentence_text`, with the extracted `entity_text`, `entity_type`, `concept`, and `task` used to guide interpretation.

The annotation objective was not to judge whether the regex match existed, but whether the extracted candidate was clinically valid for its task.

##
### Annotation Principles

The annotation protocol followed several rules to improve consistency and reduce subjective drift:

| Principle | Application |
|----------|-------------|
| **Sentence-level grounding** | Labels were assigned using the full `sentence_text`; the entity was interpreted only within that local context |
| **Explicit evidence over inference** | Labels were based on what was stated, not what might be clinically inferred |
| **Context over metadata** | `section` could support interpretation but did not override sentence-level meaning |
| **Conservative ambiguity handling** | Uncertain, implied, or weakly supported candidates were labelled `False` unless clearly valid |
| **Task-specific interpretation** | Validity was defined differently for symptoms, interventions, and clinical conditions |
| **Consistency over intuition** | Rules were applied systematically, even where clinical judgement could plausibly vary |

This conservative approach was chosen because the validation model requires learnable and reproducible label definitions rather than highly subjective clinical interpretation.

##
### State-Based vs Event-Based Labelling

A key annotation decision was that not all entity types represent the same kind of clinical target.

| Validation Task | Labelling Type | Interpretation |
|----------------|----------------|----------------|
| `symptom_presence` | State-based | Is the symptom currently present in context? |
| `intervention_performed` | Event-based | Has the intervention occurred or is it actively in use? |
| `clinical_condition_active` | State-based | Is the condition currently active or clinically relevant? |

State-based labels reflect the patient’s current status at the time of the note. Historical, resolved, negated, or uncertain mentions are labelled `False`.

Event-based labels reflect whether an intervention has occurred during the admission or is actively in place. Completed or ongoing interventions are labelled `True`, while planned, suggested, hypothetical, or conditional interventions are labelled `False`.

##
### Entity-Specific Annotation Rules

| Task | Label `True` | Label `False` | Examples |
|------|--------------|---------------|----------|
| `symptom_presence` | Symptom is currently present | Negated, historical, baseline-only, provoked, or not actively occurring | “complaining of nausea” → `True`; “denies pain” → `False` |
| `intervention_performed` | Intervention has occurred, is ongoing, or is actively in use | Planned, recommended, hypothetical, PRN-only, or not confirmed as performed | “received 2 units PRBCs” → `True`; “plan to start heparin” → `False` |
| `clinical_condition_active` | Condition is active/current and clinically relevant | Historical, resolved, chronic baseline-only, negated, uncertain, or ruled out | “worsening ARDS” → `True`; “resolved pneumonia” → `False` |

##
### Annotation Edge Cases

Several recurring edge cases required explicit handling:

| Entity Type | Edge Case | Annotation Rule |
|------------|-----------|-----------------|
| `SYMPTOM` | PRN medication indication, e.g. “PRN nausea” | `False` unless symptom is explicitly present |
| `SYMPTOM` | Historical or chronic baseline symptom | `False` unless current presence is stated |
| `INTERVENTION` | Weaning, continuation, or active device/treatment | `True` if the intervention is ongoing or already performed |
| `INTERVENTION` | “Plan to”, “consider”, “may require” | `False` unless performed |
| `INTERVENTION` | Device or line “in place” | `True` because the intervention has occurred |
| `CLINICAL_CONDITION` | “history of”, “resolved”, “ruled out” | `False` |
| `CLINICAL_CONDITION` | “possible”, “?”, “concern for” | `False` unless active/confirmed in context |
| `CLINICAL_CONDITION` | Single diagnostic term without context | `False` unless sentence context supports active relevance |

##
### Annotation Difficulty

Clinical entity validation is inherently ambiguous. The same extracted span can change label depending on temporality, certainty, section context, and clinical phrasing.

For example:

```text
"sepsis"                    → may be active, historical, or part of a problem list
"possible pneumonia"        → condition candidate, but uncertain
"post antibiotics"          → intervention occurred, even if not currently being administered
"PRN nausea medication"     → nausea mention, but not necessarily current nausea
```

Because of this, the ground truth labels should be interpreted as a reproducible annotation standard for this project rather than an absolute clinical truth. Different annotators could reasonably disagree in some borderline cases, especially for uncertain diagnoses, implicit interventions, and problem-list style documentation.

The protocol therefore prioritised consistency, explicit evidence, and conservative labelling to produce a stable supervised training signal.

##
### Annotation Validation

The completed annotation dataset was validated before model training to confirm:

- All `is_valid` labels were complete
- Labels were strictly binary (`True` / `False`)
- Required fields were present
- Task representation remained balanced across entity types
- Label distributions were clinically plausible

This ensured the annotation dataset was structurally complete and suitable for downstream splitting, fine-tuning, and evaluation.

##
## 10.4 Dataset Splitting Strategy

The final labelled dataset was split into training and held-out test sets using stratified sampling. The merged dataset and final train/test split were produced using `stratified_resplit.py`.

Because model selection was performed using cross-validation, a separate validation set was not used in the final workflow. Instead, cross-validation on the training set provided validation folds for model comparison and robustness assessment, while the test set remained untouched for final evaluation.

| Split | Count | Purpose |
|------|---------|--------|
| Training set | 1020 (85%) | Used for cross-validation, model selection, and final model fitting |
| Test set | 180 (15%) | Held out for final unbiased evaluation |

Stratification preserved the joint distribution of validation task and binary label using a combined key:

```python
stratify_key = task + "_" + is_valid
```

This ensured that each split preserved both:

- Entity-task representation (`symptom_presence`, `intervention_performed`, `clinical_condition_active`)
- Binary label distribution (`True` / `False`)

A fixed random seed was used to ensure reproducible partitioning.

Final split distribution:

| Split | `SYMPTOM` | `INTERVENTION` | `CLINICAL_CONDITION` | Total |
|------|----------:|---------------:|---------------------:|------:|
| Train | 340 | 340 | 340 | 1,020 |
| Test | 60 | 60 | 60 | 180 |

Binary label distribution was also preserved:

| Split | `True` | `False` |
|------|------:|--------:|
| Train | 510 | 510 |
| Test | 89 | 91 |


##
## 10.5 Fine-Tuning Setup

BioClinicalBERT was fine-tuned as a binary sequence classification model using the Hugging Face `Trainer` API.

### Input Formatting

Each candidate entity was converted into a structured text input containing the entity, task, section context, and sentence context:

```text
[SECTION] {section}
[ENTITY TYPE] {entity_type}
[ENTITY] {entity_text}
[CONCEPT] {concept}
[TASK] {task}
[TEXT] {sentence_text}
```

This representation makes the validation target explicit. The model is not asked to infer which entity is being evaluated from the sentence alone; it receives the extracted span, its normalised concept, the entity type, the validation task, and the sentence context.

The `section` field was included because clinical sentences are often short, fragmented, or underspecified. Section context provides additional document-level signal, helping distinguish similar entity mentions in different parts of the note, such as symptoms in `chief complaint` versus diagnoses in `assessment and plan`.

Fields derived from labels or downstream predictions, such as `is_valid`, `confidence`, and `negated`, were excluded from model input to avoid label leakage.

##
### Tokenisation

Inputs were tokenised using the BioClinicalBERT tokenizer with WordPiece tokenisation, padding, truncation, and attention masks.

| Tokenisation Output | Purpose |
|-------------------|---------|
| `input_ids` | Numerical token sequence passed to the model |
| `attention_mask` | Distinguishes real tokens from padding tokens |

A maximum sequence length of 512 tokens was used for compatibility with BERT-style encoder models.

##
### Model Architecture

The model was instantiated as a sequence classifier:

```python
AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    num_labels=2
)
```
 
| Component | Role |
|-----------|------|
| BioClinicalBERT encoder | Produces contextual representation of the structured input |
| [CLS] representation | Sequence-level representation used for classification |
| Classification head | Maps representation to binary validity logits |

The model learns the function:

```text
f(section + entity + task + sentence context) → valid / invalid
```

##
## 10.6 Cross-Validation and Model Selection

### Cross-Validation Strategy

Model selection was performed using stratified 5-fold cross-validation on the 1,020-sample training set. Cross-validation was used instead of a fixed validation split because the labelled dataset was relatively small and model performance could vary depending on the sampled partition.

```text
             Training set
            (1,020 samples)
                   │
                   ▼
          Stratified 5-fold CV
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
 Stable baseline          Advanced tuned
 configuration            configuration
      │                         │
      ▼                         ▼
  Fold metrics             Fold metrics
  mean ± SD                mean ± SD
      └────────────┬────────────┘
                   ▼
        Select final configuration
                   │
                   ▼
      Final model training and
       threshold optimisation
```

For each fold:

1. A fresh BioClinicalBERT classifier was initialised from the pretrained checkpoint.
2. The model was trained on the fold-specific training subset.
3. Performance was evaluated on the fold-specific validation subset.
4. Accuracy, precision, recall, F1-score, and validation loss were recorded.
5. Fold metrics were aggregated as mean ± standard deviation.

Models trained during cross-validation were used for model selection and out-of-fold prediction generation. They were not used as the final deployed model.

##
### Configurations Compared

Two hyperparameter configurations were compared. Both were derived from earlier development iterations and re-evaluated on the expanded 1,200-sample annotation dataset.

| Configuration | Description |
|---------------|-------------|
| Stable baseline | Conservative fine-tuning setup prioritising stable convergence and lower overfitting risk |
| Advanced tuning | More regularised configuration using additional optimisation controls |

| Hyperparameters | Stable Baseline | Advanced Tuning |
|-----------------|-----------------|-----------------|
| Learning rate | 5e-6 | 3e-6 |
| Batch size | 8 | 8 |
| Epochs | 3 | 5 |
| Gradient clipping (max norm) | 1.0 | 1.0 |
| Weight decay | None | 0.05 |
| Gradient accumulation steps | None | 2 |
| Warmup ratio | None | 0.1 |

The stable configuration represented the strongest simpler setup from earlier experiments. The advanced configuration tested whether additional regularisation and optimisation controls improved generalisation once more labelled data was available.

##
### Performance Comparison

| Metric | Stable Baseline | Advanced Tuning | Interpretation |
|--------|----------------:|----------------:|----------------|
| F1 Score      |  0.7011  |  **0.7105**  | The advanced configuration provides a modest improvement in balanced performance, indicating better overall classification quality |
| Accuracy      |  **0.7088**  |  0.7049  | Difference is negligible (<0.5%), confirming both models perform similarly in aggregate correctness |
| Precision     |  **0.7220**  |  0.6996  | Stable was more conservative and produced fewer false positives |
| Recall        |  0.6824  |  **0.7235**  | Advanced retained more valid entities and produced fewer false negatives |

The stable configuration favoured precision over recall, whereas the advanced configuration favoured recall while maintaining similar overall accuracy and slightly higher F1-score.

##
### Stability and Variance

Both configurations showed stable cross-validation behaviour, with low variance across folds. The advanced configuration had slightly lower F1 variance and substantially lower loss variance, suggesting more consistent optimisation.

| Metric        | Stable (Std Dev) | Advanced (Std Dev) | Interpretation |
|---------------|------------------|--------------------|----------------|
| F1 Score      | 0.0315           | **0.0247**         | Advanced showed slightly more stable balanced performance |
| Accuracy      | 0.0342           | **0.0309**         | Similar, with marginally lower variance for advanced |
| Precision     | 0.0428           | **0.0388**         | Advanced precision varied slightly less |
| Recall        | **0.0337**       | 0.0342             | Recall variance was effectively equivalent |
| Loss          | 0.0286           | **0.0147**         | Advanced shows more consistent optimisation |

Overall, both configurations generalised reasonably across folds. The advanced configuration provided a modest F1 improvement, higher recall, and slightly better stability.

<details>
<summary><strong>Fold-level cross-validation results</strong></summary>

### Stable Baseline Configuration

| Fold | Accuracy | F1 Score | Precision | Recall | Loss |
|------|----------|----------|-----------|--------|------|
| 1 | 0.7353 | 0.7245 | 0.7553 | 0.6961 | 0.5651 |
| 2 | 0.6520 | 0.6537 | 0.6505 | 0.6569 | 0.6346 |
| 3 | 0.7157 | 0.7157 | 0.7157 | 0.7157 | 0.5694 |
| 4 | 0.7353 | 0.7273 | 0.7500 | 0.7059 | 0.5748 |
| 5 | 0.7059 | 0.6842 | 0.7386 | 0.6373 | 0.5768 |

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | 0.7088 | 0.0342 |
| F1 Score | 0.7011 | 0.0315 |
| Precision | 0.7220 | 0.0428 |
| Recall | 0.6824 | 0.0337 |
| Loss | 0.5841 | 0.0286 |

The stable baseline achieved moderate and consistent performance. Precision exceeded recall, indicating a conservative error profile with fewer false positives but more missed valid entities.

##
### Advanced Configuration

| Fold | Accuracy | F1 Score | Precision | Recall | Loss |
|------|----------|----------|-----------|--------|------|
| 1 | 0.7304 | 0.7343 | 0.7238 | 0.7451 | 0.5967 |
| 2 | 0.6520 | 0.6758 | 0.6325 | 0.7255 | 0.6332 |
| 3 | 0.7157 | 0.7264 | 0.7000 | 0.7549 | 0.5989 |
| 4 | 0.7206 | 0.7220 | 0.7184 | 0.7255 | 0.6105 |
| 5 | 0.7059 | 0.6939 | 0.7234 | 0.6667 | 0.6038 |

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | 0.7049 | 0.0309 |
| F1 Score | 0.7105 | 0.0247 |
| Precision | 0.6996 | 0.0388 |
| Recall | 0.7235 | 0.0342 |
| Loss | 0.6086 | 0.0147 |

The advanced configuration achieved slightly higher F1 and stronger recall while maintaining low fold-level variance. Its error profile was more recall-oriented, meaning it retained more true valid entities for downstream threshold optimisation.

</details>


##
### Model Selection Interpretation

The selected model was not chosen solely by default-threshold precision. The validation model produces continuous probability scores, and the final operating point is selected later through threshold optimisation.

This creates a separation between model selection and deployment-time thresholding:

| Stage | Objective |
|-------|-----------|
| Model selection | Select the model that ranks valid candidates reliably and preserves true positives |
| Threshold tuning | Select the probability threshold that achieves the desired precision–recall balance |

In this pipeline, false negatives and false positives have different downstream consequences:

| Error Type | Effect | Recoverability |
|------------|--------|----------------|
| False negative | A valid clinical entity is missed or assigned a low score | Difficult to recover later if true positives receive low probabilities |
| False positive | An invalid candidate is retained | Can often be reduced by increasing the probability threshold |

For this reason, recall is important during model selection. If true valid entities receive low model scores, threshold tuning cannot easily recover them without also admitting many false positives. By contrast, if the model assigns higher scores to more true positives, the decision threshold can later be increased to improve precision while retaining useful clinical signal.

The advanced tuned configuration was therefore selected because it provided:

- Higher mean F1-score
- Higher recall
- Comparable accuracy
- Slightly lower F1 variance
- More consistent optimisation behaviour
- A more useful score distribution for downstream threshold tuning

Although the stable baseline had higher precision at the default threshold, it also missed more valid entities. This lower recall would reduce the maximum recoverable signal available to the final pipeline.

Final decision:

> The advanced tuned configuration was selected for final model training because it better preserved valid clinical entities during model selection, while threshold tuning was used later to enforce the pipeline’s precision-oriented operating point.

##
### Implementation Summary

Cross-validation and model selection are implemented in `cross_validation.py`.

The workflow:

1. Load the 1,020-sample training set.
2. Convert `is_valid` labels to binary targets.
3. Format each candidate using `section`, `entity_type`, `entity_text`, `concept`, `task`, and `sentence_text`.
4. Tokenise inputs using the BioClinicalBERT tokenizer.
5. Compare the stable baseline and advanced tuned configurations using stratified 5-fold cross-validation.
6. Initialise a fresh BioClinicalBERT classifier for each fold.
7. Train and evaluate each fold independently.
8. Aggregate accuracy, precision, recall, F1-score, and loss as mean ± standard deviation.
9. Save fold-level results and configuration comparison outputs to `results/cross_validation/`.
10. Select the best configuration for final model training.

The models trained during cross-validation are discarded after evaluation. The selected configuration is then used to train the final BioClinicalBERT validation model on the full training set.

##
## 10.7 Threshold Optimisation Using Out-of-Fold (OOF) Predictions

### Thresholding Rationale

The BioClinicalBERT validation model outputs a probability score for each candidate entity. A decision threshold is then required to convert this probability into a binary validation decision:

```text
is_valid = True   if probability ≥ threshold
is_valid = False  if probability < threshold
```

The default threshold of 0.5 was not assumed to be optimal. Because the full pipeline is designed as:

> High-recall rule-based extraction → precision-oriented transformer validation

threshold tuning was used to calibrate the validation layer toward higher precision while preserving sufficient recall.

This is important because the rule-based extraction stage intentionally generates broad candidate coverage. The transformer layer therefore acts as a filtering mechanism, removing false positives while retaining clinically meaningful entities.

##
### Out-of-Fold Prediction Strategy

Threshold tuning was performed using out-of-fold (OOF) predictions generated during stratified 5-fold cross-validation.

OOF predictions ensure that each training example is predicted only by a model that was not trained on that example. This provides an unbiased probability estimate for every sample in the training set.

```text
                  Training set
                 (1,020 samples)
                        │
               Stratified 5-fold CV
                        │
  Each fold model predicts only its held-out fold
                        │
     OOF probability for every training sample
                        │
                        ▼
     Threshold tuning on unbiased probabilities
```

This avoids two common issues:

| Approach | Limitation |
|----------|------------|
| Training-set predictions | Optimistically biased because the model has already seen the samples |
| Single validation split | More unstable because threshold selection depends on one partition |
| OOF predictions | Out-of-sample predictions for all training samples, reducing leakage and split dependence |

OOF predictions were saved as `results/threshold_tuning/oof_predictions.csv` with:

| Field | Meaning |
|-------|---------|
| `y_true` | Ground-truth binary label |
| `y_prob` | Out-of-fold predicted probability for the positive class |

The OOF dataset contained predictions for all 1,020 training examples, with no missing samples and balanced class representation.

##
### Threshold Selection Objective

Threshold selection was treated as a constrained optimisation problem.

Rather than maximising F1-score alone, the objective was to maximise precision while preserving acceptable recall.

This matches the role of the validation layer as a precision-biased filter. A purely F1-maximising threshold would optimise balanced performance, but the final pipeline requires higher confidence in retained entities. Conversely, unconstrained precision maximisation could discard too many true positives.

The final selection rule was:

```text
Select the threshold with maximum precision
subject to recall ≥ 0.85 × baseline_recall
```

where baseline_recall is the recall at the default threshold (`t = 0.5`), representing the model’s reference operating sensitivity.

The `0.85` factor defines an allowable recall degradation tolerance:

- Recall may decrease when increasing precision, but only within controlled limits
- Maximum permitted recall degradation is 15% relative to baseline recall
- This prevents degenerate high-precision / very-low-recall behaviour

##
### Threshold Metrics and Visualisation

Threshold-dependent performance was evaluated using OOF predictions across 1,001 thresholds from `0.000` to `1.000` in increments of `0.001`.

For each threshold, probabilities were converted into binary predictions, and precision, recall, and F1-score were computed. The resulting metric table was saved to:

The resulting metric table was saved to `results/threshold_tuning/threshold_metrics.csv` containing: `threshold`, `precision`, `recall`, `f1`.

Three diagnostic plots were generated to inspect the precision–recall trade-off and confirm that the selected threshold lay within a stable operating region.

![Precision–Recall Curve](results/threshold_tuning/plots/pr_curve.png)

The precision–recall curve showed a smooth trade-off, indicating that the model produced meaningful probability rankings rather than unstable or collapsed outputs.

![F1 vs Threshold](results/threshold_tuning/plots/f1_vs_threshold.png)

F1-score peaked at approximately `t ≈ 0.44`, with `F1 ≈ 0.735`. This represents the best balanced operating point, but it was used as a reference rather than the final threshold because the pipeline objective is precision-oriented validation rather than balanced classification.

![Precision and Recall vs Threshold](results/threshold_tuning/plots/precision_recall_vs_threshold.png)

Precision increased as the threshold increased, while recall decreased. The curves showed a usable operating region where precision could be improved without abrupt recall collapse.

Overall, the visualisations confirmed that the model exhibits a well-behaved precision–recall trade-off and supports precision-oriented threshold selection.

##
### Final Threshold Selection

The selected threshold was: 

```text
threshold = 0.549
```

| Metric | Baseline Threshold (t=0.5) | Selected Threshold (t=0.549) |
|--------|-----------------------------|------------------------------|
| Precision | 0.7025 | **0.7242** |
| Recall | **0.7039** | 0.6333 |
| F1-score | **0.7032** | 0.6757 |

Recall constraint:

```text
minimum recall = 0.85 × baseline_recall
               = 0.85 × 0.7039
               = 0.5983
```

The selected threshold satisfies the recall constraint:

```text
selected recall = 0.6333 ≥ 0.5983
```

Compared with the default threshold, increasing the threshold from 0.5 to 0.549 produced:

- **Precision increase**:
  - +0.0217 (~3.1% relative improvement)   
  - Indicates improved filtering of false positives  
- **Recall decrease**:
  - −0.0706 (~10.0% relative reduction)  
  - Within the allowable degradation (≥ 0.5983)  
- **F1 decrease**:
  - −0.0275 (~3.9% relative reduction)
  - Expected, as the system moves away from balanced optimisation  

This confirms that the final validation layer behaves as intended: stricter than the default classifier threshold, but not so strict that it removes excessive true-positive signal.

##
### Pipeline-Level Interpretation

The selected threshold changes the final system behaviour from generic binary classification to precision-oriented validation:

- Removes a larger proportion of false positives
- Reduces noise in the retained entity dataset
- Accepts a controlled loss of true positives to improve output reliability
- Preserves signal for downstreamn use

The final validation policy is therefore:

```text
Accept entity only if BioClinicalBERT confidence ≥ 0.549
```

| Stage | Behaviour |
|-------|----------|
| Rule-based extraction | Broad candidate generation with high recall |
| Transformer validation | Contextual classification of candidate validity |
| Tuned threshold | Precision-biased decision boundary for accepting final entities |

The resulting validated entity dataset is therefore more precision-oriented than the default `0.5` operating point, while still preserving sufficient recall for downstream use.

##
### Implementation Summary

Threshold tuning was implemented using three scripts and one plotting script:

| Script | Purpose |
|--------|---------|
| `generate_oof_predictions.py` | Generates out-of-fold predicted probabilities using stratified 5-fold CV |
| `tune_threshold.py` | Computes precision, recall, and F1 across threshold values |
| `select_threshold.py` | Selects the final threshold based on the defined precision-recall constraint |
| `tune_threshold_plots.py` | Generates diagnostic plots for threshold-dependent metrics |

Final outputs:

| Output | Purpose |
|--------|---------|
| `oof_predictions.csv` | Out-of-fold probabilities and ground-truth labels |
| `threshold_metrics.csv` | Precision, recall, and F1 across all thresholds |
| `best_threshold.json` | Selected threshold and corresponding metrics |
| `plots/` | Precision–recall and threshold diagnostic figures |


##
## 10.8 Final Model Training and Saved Artefacts




## 10.9 Final Validation Output

After thresholding, the validation layer writes model-derived outputs into the nested `validation` field for each candidate entity.

Output:

```json
{
  "is_valid": true,
  "confidence": 0.567,
  "task": "symptom_presence"
}
```

| Field | Meaning |
|------|---------|
| `validation.is_valid` | Final binary validity decision after thresholding |
| `validation.confidence` | Model probability for the valid class |
| `validation.task` | Entity-specific validation task |

The validated output preserves the original rule-based provenance fields while adding contextual model judgement. This allows downstream users to inspect both the extracted span and the model’s confidence in its clinical validity.

- `is_valid` is the primary decision variable
- `confidence` supports thresholding, ranking, and error analysis
- `task` ensures interpretation is aligned with entity semantics



##
## 10.10 Development Refinement

Initial model development used a smaller 600-entity annotation set with a conventional train/validation/test split. This was sufficient to validate the data pipeline, tokenisation, model training loop, and metric computation.

Several training iterations were used to test input formatting, learning-rate stability, regularisation, partial freezing, and cross-validation. These experiments showed that the pipeline was functional and that BioClinicalBERT could learn the validation task, but performance remained sensitive to data splits and did not improve reliably with additional hyperparameter complexity.

This suggested that the main limitation was labelled data volume rather than model architecture alone. The validation task also proved more ambiguous than expected, particularly for `INTERVENTION` and `CLINICAL_CONDITION`, where labels depend on temporality, intent, uncertainty, and active/current status. The smaller dataset did not provide enough examples of these contextual patterns for stable decision boundaries.

The annotation dataset was therefore expanded from 600 to 1,200 candidates using the same balanced sampling protocol. The final workflow was revised to use an 85/15 train/test split with 5-fold cross-validation on the training set.

The final README-reported methodology reflects this expanded dataset and cross-validated training strategy.



---

## 9. Full Dataset Generation

The full dataset was generated only after validation confirmed system reliability.

---

## 10. Evaluation

Evaluation is designed to directly reflect the pipeline objective:

- **Precision** → primary metric (correctness of features)
- **Recall** → controlled trade-off (coverage loss)
- **F1 score** → overall balance of system behaviour
- **Confusion matrix** → explicit analysis of FP vs FN trade-offs  

The goal of evaluation is therefore not generic model assessment, but:

- To verify that precision improves over the baseline  
- To quantify the cost in recall  
- To ensure that the trade-off aligns with the intended downstream use case  

---

## 11. Comparative Analysis: Rule-Based vs Transformer

---

## 14. Methodological Rationale and Design Reflection

defense of design choices and alignment with project goals

* Why hybrid > alternatives
* Why precision-first
* Why no NER / CRF / LLM
* Trade-offs

The system follows a hybrid architecture combining deterministic rules with a transformer-based validation model.

- Rule-based layer performs high-precision candidate extraction using section-aware logic  
- Transformer layer performs sentence-level validation to reduce false positives  

This separation reflects practical constraints:

- Limited annotated data  
- Need for precision in clinical extraction tasks  
- Requirement for interpretable and auditable outputs  

The pipeline is intentionally designed as:

- Precision-first rather than recall-maximised  
- Pipeline-centric rather than model-centric  
- Focused on structured outputs for downstream use 

---

## 12. Methodological Rationale & Design Reflection

### 12.1 Why Hybrid Rather Than Fully Model-Based Extraction

A fully model-based extraction approach was avoided because it reduces deterministic control over span extraction, schema adherence, and output traceability.

The hybrid architecture separates the task into:

- **Candidate generation:** deterministic, auditable, schema-bounded extraction
- **Contextual validation:** learned classification using sentence-level context

This avoids using rules for complex contextual reasoning while avoiding unrestricted model-based extraction.

### 11.2 Hybrid Architecture as a Constraint-Driven Decision

The hybrid design is deliberate and role-separated.

**Deterministic rule layer:**
- High-precision candidate generation  
- Transparent regex-based logic  
- Section-aware extraction  
- Auditable and reproducible behaviour  

**Transformer layer (ClinicalBERT):**
- Sentence-level binary validation  
- False-positive filtering  
- Limited category disambiguation  
- No sequence tagging or generative use  

The transformer does not perform primary extraction.  
It operates strictly as a controlled validation component.

This separation reflects:

- Awareness of precision requirements in clinical pipelines  
- Understanding of transformer brittleness in low-resource settings  
- Practical handling of limited annotation budgets  
- Avoidance of unnecessary sequence-tagging complexity  

The architecture mirrors production-oriented clinical NLP systems where deterministic logic constrains probabilistic components.

### 12.2 Candidate Generation vs Validation

The system is not uniformly precision-first at the rule level. Instead, precision is enforced at the system level after transformer validation.

For linguistically variable entity types such as `INTERVENTION` and `CLINICAL_CONDITION`, broader candidate generation is required. The validation layer then filters ambiguous or invalid candidates.

### 12.3 Failure Mode Separation

The modular design makes failure modes easier to inspect:

| Component | Failure Mode |
|----------|--------------|
| Rule layer | Missed candidates → recall limitation |
| Validation layer | Incorrect filtering/classification → precision limitation |

This separation improves debugging, maintainability, and auditability compared with a single end-to-end extraction model.

## 3. Selected Use Case and Design Consequences

This pipeline is explicitly optimised for structured dataset generation for downstream modelling and analysis. In this setting:

- Extracted entities act as model features
- Data quality directly determines model validity

Error impact is therefore asymmetric:

- **False Positives (FP)**  
  - Introduce incorrect features  
  - Corrupt models and analyses  
  - Difficult to detect downstream  
  → **High cost**

- **False Negatives (FN)**  
  - Represent missing features  
  - Reduce completeness but do not introduce noise  
  → **Lower cost (within limits)**  

This leads to the core design principle:

> Precision is prioritised over recall

The system therefore accepts controlled loss in recall to ensure that retained entities are reliable.

## 12. Methodological Rationale & Design Reflection

### 12.X Design Trade-offs

The system prioritises control, auditability, and structured outputs over maximal end-to-end optimisation.

Key trade-offs:

| Trade-off | Benefit | Limitation |
|----------|---------|------------|
| Candidate generation vs precision | Broader rules improve candidate coverage | Final precision depends on transformer filtering |
| Determinism vs linguistic coverage | Exact span traceability and reproducibility | Recall is limited by predefined patterns |
| Modular separation vs error propagation | Clear failure attribution and easier debugging | Missed rule candidates cannot be recovered downstream |
| Auditability vs end-to-end optimisation | Transparent, inspectable outputs | No joint optimisation across extraction and validation |

The final design is therefore optimised for controlled clinical information extraction rather than benchmark-maximising NER performance.

## Scope

### In Scope

- Rule-based entity extraction  
- Transformer-based validation (ClinicalBERT)  
- Structured JSON output  
- Lightweight evaluation  
- Minimal API-based deployment (FastAPI + Docker + Cloud Run + CI/CD)

### Out of Scope

- Ontology mapping (e.g. SNOMED CT)  
- Interoperability standards (e.g. FHIR)  
- Large-scale annotation  
- Full production infrastructure or scaling optimisation  

---

## 10. Deliberate Scope Reductions and Rationale

### 10.1 SNOMED Mapping — Excluded

- High manual and cognitive overhead
- Ontology engineering rather than NLP signal
- Clinical knowledge already implied by background

Referenced only as a logical extension.

---

### 10.2 FHIR Output — Excluded

- Adds format complexity without improving ML signal
- More relevant to interoperability or product roles

Structured JSON preserves flexibility and clarity.

---

### 10.3 ICU Predictor Integration — Excluded

- Already demonstrated in the ICU predictor project
- Would duplicate signal
- Increases reviewer cognitive load if predictor was integrated

Referenced conceptually, not implemented.

---

## 4. Entity-Type Behaviour and Error Tolerance

Error tolerance is not uniform across entity types:

**State-Based Entities (Symptoms, Clinical Conditions)**

- Often repeated or contextually redundant  
- Signal can be preserved even if some mentions are missed  
- False negatives are more tolerable

**Event-Based Entities (Interventions)**

- Discrete, non-redundant clinical events  
- Each instance carries specific meaning  
- False negatives represent true information loss

Implication:

- A single global threshold may disproportionately affect entity types  
- Precision-oriented filtering is likely to impact event-based entities more strongly

This is an inherent limitation of a uniform decision boundary.

---

---

## 5. Alignment with Pipeline Design Choices

The pipeline components directly reflect the selected objective:

- **Rule-based extraction**
  - High recall candidate generation  
  - Minimal filtering  

- **Transformer validation**
  - Precision-oriented filtering  
  - Removes incorrect or ambiguous entities  

- **Threshold tuning**
  - Optimised using out-of-fold predictions  
  - Maximises precision under a recall constraint  
  - Defines the operating point of the system  

- **Output structure**
  - Binary predictions (`model_pred`)
  - Probabilities (`model_prob`)  

This design allows for:

- Flexible threshold adjustment  
- Downstream modelling  
- Calibration and uncertainty analysis  

This corresponds to a standard and well-established pattern:

> Candidate generation → learned validation

---

---

#### 9.3 Modular Separation vs Error Propagation

- Separation of extraction and validation improves:
  - Debuggability (clear attribution of failure source)  
  - Maintainability (independent component tuning)  

- However:
  - Errors propagate across stages:
    - Missed candidates → unrecoverable recall loss  
    - Misclassification → precision loss  

**Implication**
- Pipeline performance is constrained by weakest stage  
- No mechanism for downstream recovery of missed entities  

---

#### 9.4 Auditability vs End-to-End Optimisation

- Hybrid design enables:
  - Full traceability from output to source span  
  - Transparent decision boundaries  
  - Structured, inspectable failure modes  

- Compared to end-to-end ML systems:
  - May underperform on benchmark extraction metrics  
  - Cannot leverage joint optimisation across tasks  

**Implication**
- System is optimised for interpretability and clinical safety  
- Not for maximal benchmark performance  

---

#### 9.5 System-Level Design Position

- Prioritises:
  - Deterministic behaviour  
  - Schema control  
  - Traceability and auditability  

- Accepts:
  - Bounded recall  
  - Dependence on validation layer  
  - Lack of global optimisation  

**Conclusion**
The architecture reflects a deliberate bias toward controlled, explainable extraction suitable for clinical and audit-sensitive environments, rather than maximising raw extraction performance.



---


## 15. Limitations

Early validation experiments indicated that the initial annotation set was too small for stable transformer fine-tuning. The dataset was therefore expanded before final cross-validation and model selection. This reflects the main practical constraint of supervised clinical NLP: model quality is limited by annotation quality and coverage.
Although it was doubled, the set still remained relatively small for trasnformer training, and so ould be a source of i=suboptimal mdoel performace. however the more likely source of performance limitation is the inherent difficulty of the validation task, which requires complex contextual interpretation that may be beyond the capabilities of a sentence-level transformer classifier. where even the manual annotation task is challenging and subjective, model performance is likely to be limited by the inherent ambiguity of the task rather than the size of the training set. This reflects a fundamental limitation of using transformer-based validation for complex contextual classification in clinical text, where even human annotators may struggle to achieve high agreement.

---

## 16. Future Work

---

## 17. Potential Clinical Integration

## 2. Downstream Applications and Requirements

Structured clinical entities are used in several types of downstream tasks, each with different performance requirements:

- **Structured dataset generation (primary use case)**  
  - Entities are converted into tabular features for:
    - Machine learning models (e.g. risk prediction)
    - Cohort selection
    - Epidemiological analysis  
  - Requires high precision  
  - Incorrect entities become incorrect features and corrupt downstream outputs  
- **Clinical summarisation**  
  - Aggregation of key findings into structured summaries  
  - Requires balanced precision and recall  
- **Information retrieval**  
  - Searching or indexing clinical concepts  
  - Requires high recall (missing entities reduces retrievability)  
- **Clinical decision support**   
  - Triggering alerts or interventions
  - Requires extremely high recall (missing signals is unsafe)  

---

## 12. Cloud Deployment


---

## 13. API Usage 

---

## 18. Repository Structure

---

## 19. How to Run

---

## 20. Requirements & Dependencies

---

## 21. License

This project is licensed under the MIT License; see the [LICENSE](LICENSE) file for details

---

## 22. Copyright

---

## 23. Citation

---

## 24. Acknowledgments

**ChatGPT**
- Provided guidance throughout the project, including code explanations, debugging, project structure and architectural design

All other components, including Python scripts, preprocessing, model training, and visualisations, were developed by the author. No additional proprietary datasets, papers, or external tutorials were required beyond those cited above

---

**Project Status:** Core Development Complete  
**Last Updated:** 3rd February 2026  
**Author & Maintainer:** Simon Yip - simon.yip@city.ac.uk

---

