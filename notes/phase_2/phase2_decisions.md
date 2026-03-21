# Phase 2 — Deterministic Rule-Based Extraction Decisions

## Objective

- This document outlines the key decisions made regarding entity schema and extraction rules for Phase 2 of the project, which focuses on deterministic rule-based extraction from ICU progress notes.
- The decisions here define the scope and approach for Phase 2, ensuring a focused and well scoped extraction process.

---

## Report Preprocessing

### 1. Objective

- The preprocessing layer performs minimal normalization of ICU clinical notes to stabilize the text for deterministic rule-based extraction while preserving traceability to the original source text.
- This serves as the formal reproducibility and audit specification for `preprocessing.py`.
- All logic described here reflects the final, validated implementation.

Preprocessing therefore functions solely as a stabilisation step applied to raw clinical text prior to structural segmentation and rule-based entity extraction.

---

### 2. Preprocessing Decisions

Preprocessing decisions are derived directly from the structural analysis conducted in Phase 1. Key findings include:

- Systematic de-identification tokens (`[** ... **]`) are widespread
- Notes contain inconsistent whitespace and line formatting
- Core structural signals (e.g., colon-delimited headers and numeric expressions) remain stable

Based on these findings, preprocessing is restricted to correcting artefacts that would interfere with rule matching or section segmentation. The preprocessing stage therefore:

- Removes systematic formatting artefacts identified in Phase 1
- Preserves the semantic content and structural organization of the note
- Maintains compatibility with downstream deterministic extraction

---

### 3. Implementation Details

The preprocessing layer performs only the minimal transformations required to stabilize the text for deterministic parsing.

1. **Normalise Newlines**  
   Standardises line breaks across the corpus (`\r`, `\r\n` → `\n`) to ensure consistent line boundaries for section parsing.

2. **Remove De-identification Tokens**  
   Strips `[** ... **]` tokens to protect patient privacy. While this occasionally breaks sentences, it is necessary for de-identification and does not impede downstream rule-based extraction.

3. **Normalise Whitespace**  
   Collapses multiple spaces or tabs into a single space. This stabilises token offsets and prevents misalignment during entity extraction, without altering section or sentence structure.

4. **Remove EMR Trailing References**  
   Eliminates end-of-document artefacts starting from the `References` header, which typically contain JavaScript popups or EMR metadata irrelevant to clinical content. This step preserves all clinical sections.

---

### 4. Preprocessing Manual Validation

#### 4.1 Overview

- Validation implemented via `validate_preprocessing.py`, which applies the preprocessing function to a random sample of 10 ICU notes, and compares original vs preprocessed outputs.
- A random sample of 10 ICU clinical notes was manually inspected to evaluate the effectiveness of the Phase 2 preprocessing function. 
- The validation focused on confirming that de-identification tokens were removed, structural elements were preserved, and that the resulting text remained suitable for deterministic rule-based extraction.

---

#### 4.2 Findings

- **Artefact Removal:**  
  All `[** ... **]` blocks, including names, dates, hospitals, and identifiers, were successfully removed across all notes.  

- **Structural Preservation:**  
  - Section headers (e.g., `S:`, `O:`, `Assessment:`, `Plan:`) remained intact.  
  - Numeric values, vitals, lab results, and medication dosages were unaffected.  
  - Colon-delimited headers and other section delimiters are preserved for downstream segmentation.

- **Sentence Integrity:**  
  - Removal of de-identification tokens occasionally produced broken or incomplete sentences.  
  - This is expected and acceptable, as Phase 2 extraction relies on section and span context rather than perfect sentence syntax.  

- **Additional Observations:**  
  - Minimal extra whitespace or dangling punctuation remains in some cleaned notes.  
  - These cosmetic issues do not compromise rule-based extraction.

---

#### 4.3 Conclusion

The preprocessing function achieves its objective: 

- It stabilises the raw clinical text by removing artefacts while maintaining semantic content, numeric density, and section structure. 
- Sentence breaks and minor cosmetic issues are acceptable within the deterministic extraction pipeline and do not require further preprocessing at this stage.

---

## Section Detection and Extraction

### 1. Objective

- Section extraction identifies structural narrative sections within clinical notes.  
- Clinical documentation typically follows semi-structured formats where major components of the note are introduced by headers
- Detecting these sections allows the pipeline to:
  1. Segment notes into semantically meaningful regions
  2. Restrict downstream extraction to relevant clinical contexts
  3. Reduce noise from structured flowsheet artifacts embedded in the notes
  4. Improve determinism of rule-based extraction by providing section-specific context for entity extraction.

This stage therefore converts a flat clinical note into a structured representation of sections and their contents.

---

### 2. Section Extraction Decision

From the Phase 1 manual inspection of the dataset, several consistent formatting patterns were observed:

- Headers almost always occur at the start of a new line
- Headers frequently end with a colon
- Some headers appear as standalone capitalised or non-capitalised phrases
- Many lines contain leading whitespace or indentation

Based on these observations, the section extraction component is designed to identify headers using a combination of:

- Line-based parsing to detect potential headers at the start of lines
- Pattern matching to identify common header formats (e.g., capitalised words followed by a colon)
- A predefined list of common headers derived from the dataset

---

### 3. Header Pattern Exploration

- The notebook `header_pattern_exploration.ipynb` was used to apply various broad regex patterns to the entire corpus of ICU notes to extract and count repetition of all potential headers.
- This process identified a comprehensive list of 300 candidate headers sorted by count number, which were then manually reviewed to determine which should be included in the final section extraction rules.

#### 3.1 Regex Logic

To identify potential headers across ICU notes, we applied two general regex patterns designed to capture most narrative section headers without being overly specific.

**Colon Terminated Headers**

`colon_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()\-'&]{0,80})\s*:\s*")`

- Matches lines that start with optional whitespace.
- Captures a leading alphanumeric phrase (letters, numbers, spaces, /, (, ), -, ', &).
- Requires a colon somewhere after the phrase (allowing optional spaces before and after).

**Standalone Headers**

`standalone_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()\-\']{1,80})\s*:?\s*$")`

- Matches lines that start and end with optional whitespace.
- Captures phrases that may or may not have a colon at the end.
- Ensures the header appears alone on a line ($ anchor), to avoid picking up inline text.

**Implementation Notes**

- Each note is split line-by-line.
- Lines matching either pattern are counted using a Counter.
- This approach captures both standard colon-terminated narrative sections and headers that appear on their own line without a colon.
- The resulting counts are used to prioritise the most frequent header candidates for canonical mapping downstream.

---

#### 3.2 Header List Manual Validation

Inspection of the 300 most frequent header candidates revealed that the detected patterns fall into several distinct structural categories. These categories reflect how clinical notes mix narrative documentation, structured monitoring data, and administrative metadata within the same free-text document.

**Observed Header Categories**

Only narrative clinical headers (e.g., SOAP-style sections) are relevant for downstream section extraction. Subsections, physiological monitoring fields, laboratory variables, and administrative metadata are captured by the regex but will be ignored in the canonical mapping.

**A. Narrative clinical sections**

These represent the core narrative structure of clinical documentation.  
They introduce sections where clinicians describe patient history, examination findings, and clinical reasoning.

Examples observed in the corpus:

- `Plan`
- `Assessment`
- `Chief Complaint`
- `HPI`

---

**B. Examination or system-based subsections**

These represent organ-system subsections, commonly appearing within a physical examination or assessment section.

Examples observed in the corpus:

- `Neurologic`
- `Cardiovascular`
- `Respiratory / Chest`
- `Abdominal`

---

**C. ICU monitoring and physiological fields**

A large proportion of detected headers correspond to monitoring variables or device parameters.  
These fields appear frequently because ICU documentation often embeds flowsheet-style monitoring data directly inside clinical notes.

Examples observed in the corpus:

- `HR`
- `BP`
- `SpO2`
- `FiO2`

These are structured measurements rather than narrative sections and therefore should not be interpreted as document section boundaries.

---

**D. Laboratory or diagnostic variables**

Another large group of matches corresponds to laboratory values or diagnostic measurements that appear as structured fields.

Examples observed in the corpus:

- `WBC`
- `Glucose`
- `Creatinine`
- `Hct`

These represent individual test values rather than narrative text blocks.

---

**E. Administrative or documentation metadata**

Some detected headers correspond to administrative or documentation-tracking fields that appear in admission templates or electronic health record exports.

Examples observed in the corpus:

- `Attending MD`
- `Admit diagnosis`
- `Transferred from`
- `Transferred to`

These fields describe metadata about the encounter rather than clinical narrative content.

---

**Empirical Insight**

The ranked frequency list of the 300 most common header candidates provided an empirical view of the header distribution across the dataset. Manual inspection revealed that the detected headers fall into several structural categories, but only a subset corresponds to true narrative clinical sections, which are relevant for section extraction. Key observations include:

1. True narrative clinical sections, such as SOAP-style headers, appear very frequently, often tens of thousands of times across the corpus. Examples include:
2. Many detected headers correspond to structured data rather than document structure, including:
   - ICU monitoring fields (e.g., `HR`, `BP`, `SpO2`, `FiO2`)
   - Laboratory or diagnostic variables (e.g., `WBC`, `Glucose`, `Creatinine`, `Hct`)
   - Administrative or documentation metadata (e.g., `Attending MD`, `Admit diagnosis`, `Transferred from`)
3. The header vocabulary stabilises quickly: the most common narrative sections appear within the top few hundred candidates. Beyond this range, additional matches consist almost entirely of flowsheet variables, lab fields, abbreviations, or other non-structural artifacts.

Thus, we can be confident that mapping canonical headers using only the frequent, manually validated narrative headers is sufficient for robust section detection.

---

**Implication for Section Detection**

The frequency analysis confirms that regex-based detection alone is insufficient, because regex patterns capture both narrative headers and structured non-narrative fields. To ensure accurate section boundaries:

1. Candidate headers are detected using generalised regex patterns.
2. Only narrative clinical headers are retained for downstream mapping.
3. Headers are normalised and mapped to a curated canonical section set.

Structured monitoring fields, lab values, and administrative metadata are ignored, even if they match regex patterns, to prevent misidentification of section boundaries.

This approach ensures that section extraction captures meaningful narrative blocks while excluding non-narrative or structured artifacts embedded within clinical notes.

---

#### 3.3 Final Headers

The following 13 headers were retained as top-level narrative section headers within the clinical notes:

- `Plan`
- `Assessment`
- `Action`
- `Response`
- `Assessment and Plan`
- `Chief Complaint`
- `HPI`
- `Past medical history`
- `Family history`
- `Social History`
- `Review of systems`
- `Physical Examination`
- `Disposition`

These represent narrative clinical sections in which clinicians write extended free-text descriptions, reasoning, or summaries. Such sections typically contain the main clinical narrative (e.g., history, examination findings, clinical reasoning, and management plans).

Because these sections introduce substantial blocks of free-text content, they provide reliable boundaries for segmenting clinical documents into meaningful narrative units for downstream processing.

---

### 4. Section Detection Decisions

#### 4.1 Canonical-Only Header Detection

Based on empirical validation and observed failure modes, the section detection strategy adopts a canonical-only detection approach:

1. Only headers present in the predefined canonical header set are detected.
2. No broad regex-based detection of non-canonical headers is performed.
3. Section boundaries are defined exclusively by canonical headers.

This ensures that only clinically meaningful narrative sections are used to structure the document, avoiding interference from subsection headers, monitoring variables, or administrative fields.

---

#### 4.2 Simplified Header Matching Strategy

Header detection is implemented using a deterministic string-based approach rather than general regex pattern matching.

The function `match_canonical_header()` supports two formats:

1. **Colon-terminated headers with optional inline content**

   - Matches lines where a canonical header appears before a colon.
   - Any text after the first colon is treated as inline section content.

   Examples:
   - `Plan:`
   - `Chief Complaint: Chest pain`

2. **Standalone headers**

   - Matches lines that exactly correspond to a canonical header (case-insensitive).
   - No inline content is present.

   Examples:
   - `Chief Complaint`
   - `HPI`

This approach ensures precise and deterministic header recognition while avoiding false positives from non-canonical patterns.

---

#### 4.3 Canonical Header Normalisation

Section headers in clinical notes frequently vary in capitalisation and formatting (e.g., `Assessment`, `ASSESSMENT`, `assessment`). To ensure consistent representation across notes:

- All detected headers are normalised to lowercase before comparison.
- A predefined canonical header set (`CANONICAL_HEADER_SET`) is stored entirely in lowercase.
- Extracted sections are stored using the canonical lowercase representation.

This guarantees that the same section is represented consistently across all documents.

---

#### 4.4 Structural Boundary Rule (Canonical-Only)

Section boundaries are defined exclusively by canonical headers:

1. When a canonical header is detected:
   - The current section (if one is active) is finalised and stored.
   - A new section is started.

2. If a line does not match a canonical header:
   - It is treated as normal content.
   - It does not terminate the current section.

Non-canonical header-like patterns (e.g., `HR`, `Cardiovascular`, `HEENT`) are therefore treated as plain text and retained within the current section.

This prevents premature section termination and preserves full narrative continuity.

---

#### 4.5 Inline Header Content Handling

Clinical documentation frequently places section content on the same line as the header.

Example: `Chief Complaint: Chest pain for two days`

- The line is split on the first colon.
- The portion after the colon is treated as inline content.
- This content is added as the first entry in the section buffer.

This ensures that no clinically relevant information is lost during extraction.

---

### 5. Method Refinement

#### 5.1 Initial Design and Rationale

The initial section detection strategy aimed to maximise structural accuracy by leveraging broad header detection. Using generalised regex patterns, all header-like lines (both canonical and non-canonical) were identified. The design followed a broad detection, narrow storage principle:

- All detected headers acted as structural boundaries
- Only canonical headers were stored as section keys
- Non-canonical headers terminated sections but did not create new ones

This approach was motivated by the assumption that:

- Clinical notes contain diverse header formats
- Treating all headers as boundaries would prevent unrelated structured content (e.g., monitoring data, labs) from being included within narrative sections

---

#### 5.2 Observed Failure: Over-Segmentation

Manual validation revealed that this approach systematically failed due to over-segmentation.

Specifically:
- Non-canonical headers such as `HEENT`, `Cardiovascular`, and `Respiratory` were incorrectly treated as section boundaries
- These headers frequently occur within true narrative sections (e.g., `Physical Examination`, `Assessment and Plan`)

This resulted in:
- Premature termination of valid sections
- Empty or severely truncated outputs (e.g., missing `Physical Examination`)
- Fragmentation of continuous clinical narratives
- Loss of clinically relevant information

---

#### 5.3 Root Cause

The failure arises from a fundamental structural property of ICU notes, clinical notes contain multiple hierarchical levels that are not distinguishable using simple regex patterns:

1. **Top-level narrative sections**  
   e.g., `HPI`, `Physical Examination`, `Assessment and Plan`

2. **Nested subsections**  
   e.g., `Neurologic`, `Cardiovascular`, `Respiratory`

3. **Structured flowsheet and measurement data**  
   e.g., `HR`, `BP`, `SpO2`, laboratory values

These elements often share identical surface patterns (e.g., colon-terminated phrases), making it impossible to reliably differentiate them using rule-based pattern matching alone.

As a result, the assumption that any detected header represents a structural boundary is invalid in real-world clinical text.

---

#### 5.4 Final Design Decision

To address this, the section detection strategy was simplified to a canonical-only boundary approach:

- Only headers in the predefined canonical set are detected
- Only canonical headers define section boundaries
- All non-canonical header-like lines are treated as normal text
- Section content is defined as all text between consecutive canonical headers

---

#### 5.5 Justification of Final Approach

This refinement directly resolves the observed failure mode:

- Prevents subsection headers from prematurely terminating sections
- Preserves complete narrative blocks (e.g., full `Physical Examination`)
- Eliminates dependence on unreliable header-like pattern distinctions

This approach intentionally prioritises:

- **High recall**: ensuring clinically relevant content is not lost
- **Structural robustness**: avoiding fragmentation in noisy, real-world data

At the same time, it accepts a known trade-off:

- **Moderate precision**: some non-relevant content (e.g., flowsheet data) may be included within sections

This trade-off is appropriate because:
- Perfect structural parsing of clinical notes is not achievable with deterministic rules
- Downstream processing can tolerate or filter excess content
- The primary objective is accurate recovery of core narrative sections, not perfect boundary precision

---

#### 5.6 Final Position

The refined method reflects a realistic and methodologically sound approach to section detection in clinical text:

- It aligns with the inherent limitations of rule-based systems
- It is robust to the structural variability of ICU notes
- It achieves the project goal of reasonably accurate section extraction, rather than unattainable perfect parsing

This establishes a reliable foundation for downstream analysis and modelling.

---

### 6. Section Extraction Workflow

The section extraction algorithm implemented in `section_extraction.py` processes each clinical note sequentially using a deterministic line-based parsing strategy with two functions: `match_canonical_header()` and `extract_sections()`.

The workflow is as follows:

1. **Split the clinical note into lines**

   - The note is divided into individual lines using newline separation.

2. **Detect canonical headers**

   - Function: `match_canonical_header()`
   - Each line is checked for an exact match against the canonical header set.
   - Supports:
     - Colon-terminated headers with optional inline content
     - Standalone headers

3. **Handle canonical header detection**

   - If a canonical header is detected:
     - The current section (if active) is finalised:
       - Buffered lines are joined into a single string (`content`)
       - If the current header already exists in the dictionary:
         - The new content is appended using a space separator
       - Otherwise:
         - A new dictionary entry is created
     - A new section is initialised
     - The buffer is reset

4. **Capture inline header content**

   - If inline text exists after the colon:
     - It is added immediately to the buffer as the first line of the new section

5. **Accumulate section content**

   - If no header is detected:
     - The line is treated as content
     - Leading/trailing whitespace is stripped
     - Empty lines are ignored
     - Non-empty lines are appended to the buffer

6. **Repeat until end of note**

   - Continue processing line-by-line, maintaining the current section context

7. **Finalise the last section**

   - After processing all lines:
     - Any remaining buffered content is finalised:
       - Joined into a single string (`content`)
       - If the current header already exists in the dictionary:
         - The content is appended using a space separator
       - Otherwise:
         - A new dictionary entry is created

8. **Return structured output**

   - Output is a dictionary:
     - **Keys**: canonical headers (lowercase)
     - **Values**: concatenated section text

This workflow reflects a strictly deterministic, canonical-boundary approach that prioritises structural robustness and preservation of narrative content.

---

### 7. Section Extraction Manual Validation

#### 7.1 Overview

Validation was performed using `validate_section_extraction.py`, which applies the section extraction function to a reproducible random sample of ICU clinical notes.

The validation combines:
- Qualitative inspection (manual review of extracted sections)
- Quantitative diagnostics (basic statistics on extraction behaviour): 

A sample of 30 ICU notes was evaluated.

---

#### 7.2 Key Findings

- **Notes with no detected sections**: 11 / 30 (≈37%)  
  - Notes without any canonical headers correctly resulted in zero extractions.  
  - This is consistent with the intended behaviour: zero extraction only occurs when no relevant headers are present.

- **Empty sections detected**: 13  
  - Certain canonical headers were retained even when no text followed them, as required by the extraction rules.

- **Extraction behaviour matches expectations**:
  - Canonical headers are correctly retained, even if empty.
  - Notes with no canonical headers yield zero extraction, which is appropriate.
  - All extracted sections align with their source content; there are no missing headers where they exist.

---

#### 7.3 Interpretation

- The observed zero-extraction rate (~37%) reflects natural variability in ICU notes and does not indicate a failure of the extraction script.
- This validation confirms that the section extraction is accurate, complete, and consistent with the defined canonical headers.

---

## Sentence Segmentation

### 1. Objective

Segment section-level clinical text into sentence-level units to enable:
  
- Precise entity span alignment for deterministic extraction
- Sentence-level context for downstream validation (Phase 3)

Bridges structural parsing (sections) and semantic extraction (entities) and preserves character offsets relative to original section text.

---

### 2. Sentence Segmentation Decisions

#### 2.1 Design Decisions

- **Post-section segmentation:** Applied after section extraction to preserve structural context
- **Deterministic & reproducible:** Sentence spans and offsets do not change across runs
- **Section-level granularity:** Enables targeted entity extraction per section

#### 2.2 Rationale for NLTK

Regex-based splitting (on periods or newlines) is unreliable in ICU notes due to:

- Abbreviations (Pt., Dr., numeric units)
- Dense numeric and procedural data
- Inconsistent punctuation or missing spaces

SpaCy offers high-accuracy segmentation but:

-	Requires heavier dependency and memory overhead
- Designed for general text and may not perform well on noisy clinical notes without custom training
- NLTK’s Punkt tokenizer provides a deterministic, rule-based approach that is sufficient for rule-based deterministic extraction
-	NLTK spans integrate seamlessly with current offset-based pipeline

---

### 3. Workflow & Implementation

1. **Section Extraction:**  
  Apply `extract_sections()` to obtain section-level text. Each section is a key-value pair (`header` → `text`).

2. **Sentence Tokenization:**  
  Use NLTK’s `sent_tokenize()` (Punkt tokenizer) to split each section independently into sentences.

3. **Offset Mapping:**  
  - Initialize a `cursor = 0` pointer at the start of the section.  
  - Loop through tokenized sentences:
    - Locate sentence in original section starting from `cursor` (`start = text.find(sent, cursor)`)
    - Compute `end = start + len(sent)`
    - Append `{ "sentence": sent, "start": start, "end": end }` to output
    - Move `cursor = end` to prevent duplicate matches

4. **Output:**  
  List of dictionaries for each sentence:
   ```json
   {
     "sentence": "string",
     "start": 0,
     "end": 0
   }

5. **Notes on implementation:**
  - Works per section to maintain context
  - Does not modify original text
  - Supports deterministic span alignment for regex-based entity extraction
  - Offsets are relative to section text, not the full note

---

### 4. Validation 

Validation was implemented via `validate_sentence_segmentation.py`, which applies the section extraction and sentence segmentation function to a random sample of 10 ICU notes and manually verifies the correctness of sentence splitting and offset alignment.

- Print sections and sentence spans with start:end -> sentence
- Verify correct sentence splitting
- Check alignment of offsets with original text
- Ensure robustness to long sentences and dense numeric/clinical data

Observed characteristics:

- Long sentences (1000+ characters) occur in structured sections (e.g., [PHYSICAL EXAMINATION])
- NLTK handles typical clinical sentence boundaries accurately
- Optional post-processing can split excessively long sentences if needed

Conclusion:

- NLTK-based sentence segmentation is deterministic, accurate, and lightweight
- Preserves both text content and offset integrity
- Provides sentence-level context needed for downstream entity extraction and validation

---


## Entity Schema Operationalisation

### 1. Purpose

- Define the three clinically meaningful entity types to extract from ICU notes, finalising scope for Phase 2 deterministic extraction and preventing uncontrolled expansion.
- These entities form the foundation of structured JSON outputs for downstream use.

---

### 2. JSON Schema Definition

The extraction pipeline outputs one JSON object per entity. This design preserves auditability, traceability, and downstream compatibility.

Each extracted entity follows a **two-layer schema**:

```json
{
  "note_id": "string",
  "subject_id": "string",
  "hadm_id": "string",
  "icustay_id": "string",

  "entity_text": "string",
  "concept": "string",
  "entity_type": "SYMPTOM | INTERVENTION | COMPLICATION",

  "char_start": 0,
  "char_end": 0,
  "sentence_text": "string",
  "section": "string",

  "negated": true,

  "validation": {
    "is_valid": true,
    "confidence": 0.0,
    "task": "symptom_presence | intervention_performed | complication_active"
  }
}
```

#### 2.1 Considerations and Decisions

1. Multiple entities per report: 

- A single ICU note may generate multiple JSON objects corresponding to each entity detected across all three entity types.
- `note_id`, `subject_id`, and other identifiers ensure entities can be grouped back to the originating note or patient.

2. Concept vs surface form:

- `entity_text` is the precise term identified in the note.
- `concept` is the high-level normalised category to which the entity belongs.

3. Section awareness:

- Extraction is section-specific: each entity is mapped to its section (HPI, Chief Complaint, etc.).
- Sections allow filtering, prioritisation, and downstream analyses that depend on note structure.

4. Negation handling:

- `negated` is primarily meaningful for `SYMPTOM`, but it is still included for all entities for schema consistency
- Negation does exist and can be detected in all 3
- negation isn tthe goal for all 3, Because your goal is not detecting negation.

Entity
Real question
SYMPTOM
Is the symptom present?
INTERVENTION
Was it performed?
COMPLICATION
Is it an active condition?



- **Interpretation by entity type:** negation is critical for `SYMPTOM` (e.g., "denies chest pain" is clinically meaningful), not that improtant for `INTERVENTION` (e.g., "no intubation" weak signal as doesnt cover the other patterns we want, Negation is incomplete and unreliable), and not as improtant for `COMPLICATION` (e.g., "no evidence of sepsis" weak signal as it doesnt cover other patterns we want, Negation captures only a small fraction of reality).
- Schema is uniform but only one entity type (SYMPTOM) has a strong signal from negation, so we will populate with boolean for all entities but only interpret it for SYMPTOM. For the other entity types, it will always be  null, and we will ignore it in downstream analyses.

For SYMPTOM:
	•	It is high value, low cost, high accuracy, reliable
	•	It solves a major part of the problem
  - since for symptoms this is basically the criticalsignal and sort of ground truth for the entity, it is worth including even if it is not perfect, because it provides a strong signal for the presence or absence of symptoms, which is essential for transformer validation.
  - therefore we populate with boolean

For INTERVENTION / COMPLICATION:
	•	It is insufficient alone. negation can be accurate but is not the primary signal, it is only partial, it misses acute vs chronic or planned vs performed distinctions, which are more important for these entity types
	•	so still partially informative but not covering the main problem. 
  - therefore there is no point to include it, we will always have null for these entity types, and it may cause confusion or misinterpretation if we include it.
  - However, including it in the schema allows for future expansion if we later develop more sophisticated negation handling for these entity types, and it maintains a consistent schema across all entities.

Key insight

For SYMPTOM:

Negation directly answers the question
	•	“no chest pain” → absent ✔
	•	“chest pain” → present ✔

So:

Negation ≈ ground truth (most of the time)

⸻

For INTERVENTION:

Negation only answers one narrow case
	•	“no intubation” → not performed ✔

But:
	•	“intubation planned” → not performed ❌ (negation fails)
	•	“may require intubation” → not performed ❌
	•	“intubated” → performed ✔

So:

Negation is incomplete and unreliable

⸻

For COMPLICATION:

Same problem:
	•	“no sepsis” → not active ✔

But:
	•	“history of sepsis” → not active ❌ (negation fails)
	•	“resolved sepsis” → not active ❌
	•	“?sepsis” → uncertain ❌

So:

Negation captures only a small fraction of reality

Correct approach:
	•	SYMPTOM → negated = true/false
	•	INTERVENTION / COMPLICATION → negated = null

Why this is better:
	•	Avoids implying it is meaningful when it is not
	•	Prevents downstream misuse
	•	Makes schema semantically correct

You suggested:

“Why not also detect planned/history/resolved?”

Because:
	•	It becomes:
	•	complex
	•	brittle
	•	incomplete
	•	And still:
	•	fails on unseen phrasing

And critically:

The transformer already sees the full sentence and does this better


5. Transformer Validation:

Candidate generation (rules) + candidate filtering (model)

- `validation` does not mean the same thing for all entities, it is for contextual validation of the entity's clinical relevance and correctness rather than a pure confidence score.
- we now have 

  "validation": {
    "is_valid": true,
    "confidence": 0.0,
    "task": "symptom_presence | intervention_performed | complication_active"
  }

- is_valid is the model's binary judgement on whether the entity is clinically valid in context (e.g., true symptom, performed intervention, active complication).
-  confidence is the model's confidence score for that judgement, which can be used for thresholding or prioritisation in downstream analyses, metrics computation or error analysis.
- task indicates the specific validation task relevant to the entity type, which guides interpretation of the is_valid and confidence fields.
- we have made validation more explicit and structured to reflect its critical role in confirming the clinical relevance of extracted entities, especially given the limitations of deterministic rules and negation handling.
- this is specifically like this to explicitly avoid the misconception that the validation is a general confidence score for the entity extraction, when in reality it is a specific judgement on the clinical validity of the entity in context, which is essential for filtering out false positives and ensuring high-quality structured data for downstream use. 

- For `SYMPTOM`, it reflects the likelihood that the entity is a true symptom based on context. Presence + context + negation correctness
- For `INTERVENTION`, it reflects the likelihood that the action was actually performed or occured rather than planned or hypothetical.
- For `COMPLICATION`, it reflects the likelihood that the entity represents a true current/acute complication rather than a historical diagnosis or resolved issue.



6. Provenance and downstream use:

- Character offsets (`char_start`, `char_end`) preserve exact positioning in the raw note, supporting traceability, auditing, or text alignment.
- `sentence_text` provides surrounding context to support transformer-based validation and auditability.
- `section` allows for structural filtering and analysis based on note organization.

---

### 3. Entity Types 

Extraction in Phase 2 is strictly limited to three entity types which seperate clinical reasoning components and provide a clear, clinically relevant structure for downstream analysis:

- **SYMPTOM** → subjective patient complaints or observed clinical signs → whats happening to the patient
- **INTERVENTION** → actions performed to manage or treat the patient → what we are doing to the patient
- **COMPLICATION** → acute adverse or pathological events occurring during ICU stay → what has gone wrong with the patient

Limiting to these three ensures:

-	High precision and manageable rule creation
-	Separation between overlapping entity types
-	Clinically meaningful coverage without chasing unstructured data

---

#### 3.1 SYMPTOM

**Purpose:** Capture patient-reported complaints or clinician-observed manifestations.  
**Rationale:** Symptoms represent the subjective or observable clinical state that informs interventions and complications.

**Operational Decisions:**

- **Inclusions:** Patient complaints (e.g., “chest pain”) and observed signs (e.g., “confused”).
- **Exclusions:** Labs, imaging, baseline diagnoses, or interventions.
- **Negation Handling:** Deterministic rules using “no”, “denies”, “without”, “not”. 
- **Transformer Role:** Contextual validation to confirm the entity is a true symptom rather than a historical diagnosis or non-symptom mention, and fix negation errors.

**Positive Extraction Examples**

The following phrases should produce a SYMPTOM extraction:

- "patient reports chest pain"
- "complaining of nausea overnight"
- "severe headache this morning"
- "patient feeling dizzy"
- "persistent shortness of breath"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce a SYMPTOM extraction:

- "history of migraine"
- "CT shows intracranial hemorrhage"
- "labs notable for elevated troponin"
- "scheduled for CT scan"

These represent diagnoses, tests, or history rather than symptoms.

**Boundary Resolution**

Ambiguous terms are resolved as follows:

- tachycardia → `VITAL_MENTION` (not `SYMPTOM`)
- hypotension → `VITAL_MENTION`
- delirium → `SYMPTOM`
- agitation → `SYMPTOM`

---

#### 3.2 INTERVENTION

**Purpose:** Capture therapeutic or procedural actions performed.  
**Rationale:** Interventions document treatments and procedures, critical for understanding patient management.

**Operational Decisions:**

- **Inclusions:** Medications started during ICU stay (e.g., “started norepinephrine”) and procedures (e.g., “central line inserted”). Rule-based, but strict pattern matching to ensure high precision (e.g., "started X", "placed on Y", "administered Z").
- **Exclusions:** Hypothetical plans, suggestions, or chronic medications not initiated due to ICU admission.
- **Negation Handling:** Not reliable, not primary signal. Validate performed vs planned, which is a classification problem rather than pure negation.
- **Transformer Role:** Contextual validation to confirm the entity represents an actual performed intervention or action rather than a plan, hypothetical, or recommendation. This is where rules-based negation handling is weakest, so transformer validation is critical to filter out non-interventions.

**Positive Extraction Examples**

The following phrases should produce an `INTERVENTION` extraction:

- "started norepinephrine"
- "intubated for airway protection"
- "central line inserted"
- "patient placed on ventilator"
- "administered antibiotics"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce an `INTERVENTION` extraction:

- "plan to start antibiotics"
- "consider dialysis"
- "may require intubation"
- "recommend central line placement"

These represent future plans or recommendations rather than completed interventions.

**Boundary Resolution**

Ambiguous phrases are resolved as follows:

- "started antibiotics" → `INTERVENTION`
- "on antibiotics" → `INTERVENTION`
- "intubation planned" → NOT extracted

---

#### 3.3 COMPLICATION

**Purpose:** Capture adverse or pathological developments during ICU stay.  
**Rationale:** Complications indicate negative outcomes or new pathological events, essential for downstream analysis and evaluation.

**Operational Decisions:**

- **Inclusions:** Acute clinically significant pathological conditions (e.g., "AKI", "sepsis", "pneumothorax"), both reason for ICU admission and new complications arising during stay. Rule-based candidate generation with broad patterns (e.g., "developed X", "new X", "patient with X"), but transformer validation is critical to confirm clinical relevance and acuity.
- **Exclusions:** Chronic baseline diagnoses without acute worsening, past medical history, or resolved issues.
- **Negation Handling:** Not reliable, not primary signal. Validation is a fully contextual and temporal classification problem rather than pure negation. 
- **Transformer Validation:** Critical role, distinguishes between acute vs historical conditions, and active vs resolved conditions. This is where rules-based negation handling is weakest, so transformer validation is essential to filter out historical or non-complication mentions.

**Positive Extraction Examples**

The following phrases should produce a `COMPLICATION` extraction:

- "developed sepsis overnight"
- "new acute kidney injury"
- "patient with pneumothorax"
- "episode of ventricular arrhythmia"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce a `COMPLICATION` extraction:

- "history of chronic kidney disease"
- "prior stroke"
- "family history of cancer"

These represent baseline or historical conditions.

**Boundary Resolution**

Ambiguous terms are resolved as follows:

- AKI → `COMPLICATION`
- sepsis → `COMPLICATION`
- infection → `COMPLICATION`
- chronic heart failure → NOT extracted unless acute worsening is described

---

### 4. Excluded Entity Types

**Medications**

- Extremely large and complex category with high variability in naming, dosing, and context.
- High risk of false positives due to mentions of chronic medications, medication allergies, or medication plans.
- Not suitable for deterministic rule-based extraction without extensive lexicon and context handling, which is beyond the scope of Phase 2.

**Vital Signs**

- Highly inconsistent mentions of vital signs in clinical notes, often appearing as structured flowsheet data embedded within the text.
- Difficult to reliably extract with deterministic rules due to variability in formatting, units, and context.

**Laboratory Values**

- Similar to vital signs, lab values are often embedded as structured data within notes, with high variability in formatting and context.
- Extracting lab values would require complex rules to handle units, normal ranges, and contextual interpretation, which is beyond the scope of Phase 2.
- Already captured in structured EHR data, so extraction from notes is not essential for downstream analyses.

---

### 5. Transformer Validation Role

You are doing:
	•	inference-only classification

The model:
	•	already understands language
	•	already understands context (to some extent)

You are just asking it:
	•	“Is this real?”
	•	“Is this present?”
	•	“Was this done?”

A classifier over (sentence + entity)

No training required.

Deterministic layer (rules)

High precision, low ambition:
	•	SYMPTOM → yes (strong)
	•	INTERVENTION → yes (restricted patterns only)
	•	COMPLICATION → only as candidate generation (weak)

Transformer layer

Context understanding:
	•	Validates symptoms (negation + context)
	•	Filters interventions (performed vs planned)
	•	Classifies complications/diagnoses (primary responsibility)

Rules are used for:
	•	high precision anchors (symptoms, meds, procedures)
	•	candidate generation

Models are used for:
	•	disambiguation
	•	context understanding
	•	temporal reasoning
	•	classification

Phase 2 (rules)
	•	extract:
	•	symptoms (strong)
	•	interventions (strict)
	•	complication candidates (broad)

Phase 3 (transformer)
	•	validate:
	•	negation (symptoms)
	•	action vs plan (interventions)
	•	diagnosis classification (complications)

What that means for your entities

SYMPTOM
	•	Rules are strong → good coverage
	•	Transformer = refinement

INTERVENTION
	•	Rules are moderate → decent candidates
	•	Transformer = heavy filtering

COMPLICATION
	•	Rules are weak → broad candidates
	•	Transformer = primary filter

You are NOT doing ground-truth validation.

You are doing:

Post-extraction filtering using a model

So “validation” =

“Given this candidate, is it clinically valid in context?”

Is it okay that transformer does most of the work for 2 entities?

Yes.

This is actually expected:

Entity
Rules strength
Transformer reliance
SYMPTOM
Strong
Medium
INTERVENTION
Medium
High
COMPLICATION
Weak (candidate only)
Very High

That is a valid hybrid system

---

### 5. Summary

Phase 2 extraction will only target `SYMPTOM`, `INTERVENTION`, and `COMPLICATION` to:

- Ensure manageable rule creation and high precision
- Prevent overlap and ambiguity between entity types
- Provide structured, span-aligned JSON outputs ready for Phase 3 transformer validation

This finalises the scope of entity extraction for Phase 2. We are doing:

	•	deterministic extraction system
	•	section-aware parsing
	•	negation handling
	•	candidate generation
	•	transformer validation layer

You are building:
	•	A candidate generator (rules)
	•	A context classifier (transformer)

The JSON reflects BOTH:
	•	Raw extraction (entity_text, negated)
	•	Model judgement (validation)


---

## Rule Based Extraction

### 1. Objective

Rule-based extraction applies deterministic pattern rules to identify entities belonging to the predefined schema (**SYMPTOM, INTERVENTION, COMPLICATION**) within sectioned clinical notes.

This component is not intended to be a complete clinical NLP system. Instead, it serves as a controlled, deterministic reference system that produces stable, reproducible, and span-aligned outputs for downstream validation.

The system is defined by four core properties:
- Deterministic: identical input produces identical output  
- High precision: extracted entities are expected to be correct  
- Span-aligned: all entities map to original character offsets  
- Schema-constrained: limited strictly to the three predefined entity types  

Rule design prioritises prototypical ICU expressions and clear lexical patterns, favouring precision over recall. It does not attempt to capture all linguistic variation or achieve exhaustive coverage. Instead, it extracts representative, high-confidence instances of each entity type, forming a reliable baseline for comparison with transformer-based methods.

---

### 2. Rule Based Extraction Decision

Rule-based extraction is grounded in the Phase 2 schema operationalisation, which defines three entity types to extract:

- **SYMPTOM**
- **INTERVENTION**
- **COMPLICATION**

Each entity is governed by explicit inclusion/exclusion criteria, negation cues, and prototypical trigger patterns. The extraction engine applies deterministic pattern matching to identify candidate entities within sectioned note text, strictly adhering to these predefined definitions.

Also, the predefined JSON schema ensures that all extracted entities are represented in a consistent, structured format, facilitating traceability and downstream analysis.

This structure also allows the logic to have already been defined for what we are extracting, and espeically the types of rules we need to implement or not need to implement, and also allows us to extract where we know we can do a good job with rules, and leave the harder cases for the transformer to validate.

The json schema also allows us to have a clear separation between the deterministic extraction layer (rules) and the contextual validation layer (transformer), which is critical for ensuring that we are not overfitting rules to the data, and that we are providing a stable, reproducible output for the transformer to work with.

Rule development was guided by targeted inspection of a small sample of ICU notes from section extraction validation (`validate_section_extraction.py`). High-frequency, clinically meaningful patterns were identified and iteratively translated into deterministic rules, prioritising precision and reproducibility over exhaustive coverage.

This constrained approach prevents scope creep, ensures span-aligned and clinically coherent outputs, and provides a stable foundation for downstream transformer validation in Phase 3.

---

Rule-Based Symptom Extraction
	•	A rule-based approach was developed to extract symptom entities from clinical notes.
	•	Initial rules were derived from observed linguistic patterns (e.g. complaint expressions, descriptive phrases).
	•	The system was iteratively refined through evaluation on a validation subset of notes.
	•	Rules were designed to prioritise:
	•	high recall through broad pattern matching
	•	precision through exclusion filters and negation handling
	•	Development was stopped when additional rules produced minimal improvement in extraction performance.

  The approach focuses on pattern generalisation rather than exhaustive vocabulary enumeration, improving scalability across unseen data.

  we are starting with general ICU convention, and then light validation on notes, we do not iteratively fit rules to the dataset, as this would lead to overfitting and loss of generalisability. this is theory driven rule design with light validation, not data-driven rule fitting.

---

Layer
What it adds
Section filtering
removes irrelevant context
Concept grouping
maps many phrases → one clinical meaning
Negation handling
prevents false positives
Context rules
e.g. ROS vs exam vs plan
Pipeline integration
feeds downstream ML

Is it okay that you only have ~10 symptoms?

Yes — if they are chosen correctly.

You are not building:
	•	a clinical ontology
	•	a full medical NER system

You are building:
	•	a demonstration of pipeline design + engineering judgment
  a minimal, concept-level clinical IE system that produces stable, reproducible outputs for downstream validation.

What matters is NOT coverage

It is:
	•	clarity of logic
	•	correctness of extraction
	•	explainability
	•	integration with ML

You want:

“representative coverage of common ICU symptom patterns”

NOT:

“complete medical coverage”

key concept is at the core all rule-based clinical NLP reduces to controlled lexical matching, tehcnically at a crude basic level its keywrod extraction, however :

	•	constraint layering
	•	pipeline integration
	•	output structure
	•	evaluation

is what makes rule based extraction a non-trivial engineering task that requires careful design and judgment. this is constrained, concept-level candidate generation in a clinical IE pipeline, not a full medical NER system or ontology development.

You are NOT:
	•	oversimplifying incorrectly
	•	misunderstanding rule-based NLP
	•	missing some “hidden complexity”

You ARE:
	•	implementing the core mechanism correctly
	•	at a controlled scale
	•	with proper engineering structure

  ---

  Why deduplication is REQUIRED

Because regex cannot understand:
	•	overlap
	•	semantics
	•	hierarchy

So you must enforce it manually.


Characteristics of Short-stay Critical Care Admissions From Emergency Departments in Maryland


- The 17 symptoms included in this extraction were selected based on clinical heuristics and expert opinion, anchored to the most common presenting features of ICU admissions as reported in the Maryland short-stay critical care study (Characteristics of Short-stay Critical Care Admissions From Emergency Departments in Maryland PMID: 28323374). 
- Only patient-reported features (subjective symptoms) were included; objective signs such as vital parameters or laboratory abnormalities are captured separately by the vitals/entity extraction pipeline. 
- This approach ensures relevance to ICU presentations while maintaining clear distinction between symptoms and signs.