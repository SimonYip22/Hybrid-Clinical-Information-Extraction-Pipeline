The 4 Architectural Layers (System View)

 These describe what the pipeline is composed of.
	1.	Preprocessing (span-preserving)
	2.	Section segmentation
	3.	Rule-based entity extraction
	4.	JSON schema mapping

---

1. Preprocessing Layer (Span-Preserving Normalisation)

Purpose

Prepare raw notes for deterministic parsing while preserving structural integrity.

Decisions
	•	Normalize newline handling
	•	Normalize whitespace collapse where safe
	•	Standardize line boundaries
	•	Preserve character offsets
	•	Do not destructively re-tokenize
	•	Do not lowercase entire notes

Artefact Handling
	•	Remove [** ... **] de-identification spans
	•	Trim trailing EMR reference or JavaScript artefacts if detected
	•	Preserve clinically relevant bracketed content

This layer prepares text for segmentation without altering semantic structure.

⸻

2. Structural Segmentation Layer

Purpose

Partition notes into meaningful clinical sections.

Header Logic

Supported patterns:
	•	Colon-terminated headers
	•	Assessment:
	•	Plan:
	•	NEURO:
	•	Optional numeric prefixes
	•	1. Cardiovascular:
	•	Indented headers permitted

Uppercase-only headers supported if followed by colon.

Design Constraints
	•	Header detection must tolerate indentation
	•	Header detection must avoid matching inline abbreviations (e.g., CV: inside narrative)
	•	Section spans must not overlap
	•	Entire note must be covered by either a labeled section or default section

This produces structured section blocks.

⸻

3. Deterministic Extraction Layer

Purpose

Extract structured elements from segmented text.

Extraction Categories
	•	Numeric values (labs, vitals, dosing)
	•	Blood pressure patterns
	•	Physiologic clusters
	•	Inline numeric tables
	•	Infusion rates and medication dosing expressions

All extraction is regex-based.

Regex design principles:
	•	Avoid over-restrictive assumptions
	•	Permit optional ranges
	•	Allow flexible spacing
	•	Avoid catastrophic backtracking
	•	Avoid capturing partial numeric fragments

Each extraction must return:
	•	Extracted value
	•	Character span
	•	Section label
	•	Extraction category

No probabilistic filtering occurs here.

⸻

4. Schema Mapping Layer

Purpose

Convert extracted elements into stable JSON output.

Required Fields
	•	note_id
	•	section_title
	•	section_text
	•	numeric_values
	•	bp_values
	•	extraction_category
	•	char_start
	•	char_end

Constraints
	•	No duplication of spans
	•	No overlapping extracted entities
	•	Stable ordering
	•	Fully reproducible output

This produces deterministic JSON per note.