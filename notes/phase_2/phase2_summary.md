# Phase 2 Summary: Deterministic Rule-Based Extraction

---

## Purpose

Phase 2 constructs a deterministic information-extraction pipeline that converts ICU clinical notes into structured, span-aligned JSON entities.

- The system transforms raw narrative text into reproducible structured representations using rule-based methods. 
- No probabilistic models or machine learning systems are used in this phase.
- All outputs must be fully deterministic: identical input text must always produce identical structured output.

The goal of Phase 2 is therefore to establish a reproducible, auditable, and deterministic extraction backbone upon which all later phases (transformer validation, modelling, and deployment) depend.

---

## System Architecture

The extraction system is implemented as a sequential pipeline composed of four core processing layers.

1. **Preprocessing (Span-Preserving Normalisation)**
2. **Section Segmentation**
3. **Rule-Based Entity Extraction**
4. **JSON Schema Mapping**

Each layer performs a controlled transformation on the note while preserving the ability to trace structured outputs back to the original source text.

---

## Operational Pipeline Summary

Phase 2 is implemented as a sequential pipeline of deterministic processing stages.  
Each stage prepares the data required by the next stage.

| Step | Component | Purpose |
|-----|-----|-----|
| 1 | Schema Operationalisation | Define the exact clinical entities to extract and freeze scope to prevent uncontrolled expansion. |
| 2 | Preprocessing Layer | Normalize artefacts (e.g. headers, formatting, encoding issues) while preserving original character offsets. |
| 3 | Section Detection | Identify structural sections within clinical notes (e.g. assessment, plan, vitals) and map text spans to section labels. |
| 4 | Rule-Based Extraction Engine | Apply deterministic pattern rules to identify candidate entities belonging to the predefined schema. |
| 5 | Negation Detection | Detect negation cues (e.g. "no", "denies", "without") affecting extracted entities and attach negation flags. |
| 6 | JSON Output Construction | Convert extracted entities into a standardized schema-aligned JSON representation. |
| 7 | Deterministic Stability Testing | Verify reproducibility and structural correctness of the full pipeline before downstream modelling. |

This ordering is intentional.

Extraction rules depend on clean text and known section boundaries, while negation detection requires identified entities to operate on. JSON output construction occurs only after the extraction and negation steps are complete.

The final step ensures that the entire pipeline is deterministic, stable, and structurally valid before any downstream evaluation or modelling occurs.

---

### Step 1 — Schema Operationalisation

#### 1.1 Entity Scope

Extraction is strictly limited to four entity types:

- **SYMPTOM**
- **INTERVENTION**
- **COMPLICATION**
- **VITAL_MENTION**

No schema expansion is permitted during Phase 2.

Restricting scope prevents uncontrolled rule proliferation and ensures deterministic rule development remains feasible.

---

#### 1.2 Operational Definitions

Each entity type must define:

- Inclusion criteria
- Exclusion criteria
- Trigger phrases
- Common lexical patterns
- Boundary conditions
- Ambiguous examples
- Edge-case handling

Definitions must prevent overlap between entity classes.

Example template:

```text
ENTITY: SYMPTOM

Inclusions
- Patient-reported complaints
- Clinician-observed symptoms

Exclusions
- Diagnoses
- Laboratory values
- Imaging findings

Negation cues
- no
- denies
- without
```

---

### Step 2 — Preprocessing Layer (Span-Preserving)

#### Core Constraint

All character offsets must remain valid relative to the original raw text.

If offsets break, entity span validation fails.

---

#### Allowed Transformations

- Whitespace normalization (non-destructive)
- Standardized newline handling
- `[** ... **]` de-identification span removal
- Trailing EMR artefact trimming
- Header boundary detection

---

#### Prohibited Transformations

- Full lowercasing of documents
- Punctuation stripping
- Destructive tokenization
- Semantic rewriting

All transformations must preserve or carefully remap character offsets.

---

#### Preprocessing Output

The preprocessing module produces:

- Cleaned note text
- Section boundary metadata
- Offset-preserving character mapping

---

### Step 3 — Section Detection (Structural Segmentation)

#### Purpose

Partition clinical notes into structural sections prior to extraction.

---

#### Header Identification Rules

Supported patterns:

**Colon-terminated headers**

Assessment:
Plan:
NEURO:

**Numbered headers**

1. Cardiovascular:
2. Respiratory:

**Uppercase block headers**

RESPIRATORY:
CARDIOVASCULAR:

Indented headers are permitted.

---

#### Section Assignment

Every extracted entity must inherit:

- `section_title`
- Section character span

Constraints:

- Section spans must not overlap
- Header detection must tolerate indentation
- Inline abbreviations must not trigger headers (e.g. `CV:` inside narrative)

Entire notes must be covered by either:

- Detected header sections
- Or a default section label (`UNLABELED`)

---

### Step 4 — Rule-Based Entity Extraction

#### Architecture

Extraction rules are modularized by entity type.

```text
extraction_rules/
	symptom_rules.py
	intervention_rules.py
	complication_rules.py
	vital_rules.py
```

---

#### Expected Rule Volume

Estimated rule counts:

| Entity | Expected Patterns |
|------|------|
| `VITAL_MENTION` | 15–25 |
| `INTERVENTION` | 20–40 |
| `COMPLICATION` | 15–25 |
| `SYMPTOM` | 15–30 |

Total expected rule count: **65–120 rules**.

Initial design prioritizes high precision over maximal recall.

---

#### Rule Design Constraints

Each rule must:

- Accept raw section text
- Return matched text
- Return `entity_type`
- Return `char_start`
- Return `char_end`
- Inherit `section_title`
- Avoid overlapping span duplication

Negation detection is not handled during this stage.

---

### Step 5 — Negation Detection

#### Design Principle

Negation is evaluated after entity extraction.

---

#### Detection Method

A fixed token window preceding the entity span is evaluated.

Window size: 

- 5–7 tokens preceding the entity

Negation cues include:

- no
- denies
- without
- not
- negative for

No syntactic parsing or dependency analysis is used.

---

#### Output

Each entity record must include: 

`negated: True | False`

---

### Step 6 — JSON Output Construction

#### Extraction Interface

`extract(note_id, raw_text) → List[EntityRecord]`

---

#### Required Output Fields

Each entity record must contain:

- `note_id`
- `entity_type`
- `matched_text`
- `section_title`
- `char_start`
- `char_end`
- `negated`

Optional fields:

- `rule_id`
- `confidence` (deterministic in Phase 2)

---

#### Output Constraints

- No duplicate spans
- No overlapping entity spans
- Stable deterministic ordering
- Valid JSON schema

---

### Step 7 — Deterministic Stability Testing

#### Validation Sample

Manually inspect 20–30 representative ICU notes including:

- Long notes
- Numeric-dense notes
- Sparse narrative notes
- Addendum-heavy notes

---

#### Validation Checks

- Span alignment correctness
- Section assignment correctness
- Negation flag correctness
- Absence of crashes
- Valid JSON structure
- No uncontrolled rule triggering

---

## Phase Boundary

Phase 2 is complete when:

- Deterministic extraction works on representative notes
- Span alignment is verified
- All four entity types are extractable
- JSON output is stable and reproducible

---

## Exclusions

Phase 2 does not include:

- Transformer validation
- Corpus-wide execution
- Statistical evaluation
- Clinical outcome modelling
- Deployment or CI/CD

These belong to later project phases.

---