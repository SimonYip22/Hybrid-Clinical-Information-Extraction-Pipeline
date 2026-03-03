Step 1 — Schema Operationalisation

1.1 Entity Scope

Extraction is strictly limited to four entity types:
	•	SYMPTOM
	•	INTERVENTION
	•	COMPLICATION
	•	VITAL_MENTION

No schema expansion is permitted during Phase 2.

⸻

1.2 Operational Definitions

For each entity type, the following must be explicitly defined:
	•	Inclusion criteria
	•	Exclusion criteria
	•	Trigger phrases
	•	Common lexical patterns
	•	Boundary conditions
	•	Ambiguous examples
	•	Edge-case handling

Each definition must prevent overlap between entity classes.

Example structural template:

ENTITY: SYMPTOM

Inclusions:
- Patient-reported complaints
- Observed clinical symptoms

Exclusions:
- Diagnoses
- Lab values
- Imaging findings

Negation cues:
- no
- denies
- without

This prevents uncontrolled rule drift and ensures extraction consistency.

Step 2 — Preprocessing Layer (Span-Preserving)

2.1 Core Constraint

All character offsets must remain valid relative to the original raw text.

If offsets break, downstream validation fails.

⸻

2.2 Allowed Transformations
	•	Whitespace normalization (non-destructive)
	•	Standardized newline handling
	•	[** ... **] de-identification token removal
	•	Trailing EMR artefact trimming (including JavaScript fragments)
	•	Header boundary detection

⸻

2.3 Prohibited Transformations
	•	Full lowercasing of document
	•	Punctuation stripping
	•	Destructive re-tokenization
	•	Semantic rewriting

All transformations must preserve alignment or carefully adjust offsets.

⸻

2.4 Preprocessing Output

The preprocessing module must output:
	•	Cleaned text
	•	Section boundary metadata
	•	Offset-preserved character mapping

⸻

Step 3 — Section Detection Logic

3.1 Header Identification Rules

Supported patterns:
	•	Colon-terminated headers
Assessment:
Plan:
NEURO:
	•	Numbered headers
1. Cardiovascular:
	•	Uppercase block headers followed by colon

⸻

3.2 Section Assignment

Every extracted entity must inherit:
	•	section_title
	•	Section character span

No overlapping section spans permitted.

Entire note must be covered by either:
	•	A detected header section
	•	A default section label (e.g., UNLABELED)

⸻

Step 4 — Rule Extraction Engine

4.1 Architecture

Rules are modularized by entity type:

rules/
  vital_rules.py
  intervention_rules.py
  complication_rules.py
  symptom_rules.py

4.2 Expected Rule Volume
	•	VITAL_MENTION: 15–25 patterns
	•	INTERVENTION: 20–40 patterns
	•	COMPLICATION: 15–25 patterns
	•	SYMPTOM: 15–30 patterns

Total estimated: 65–120 regex rules.

⸻

4.3 Rule Design Constraints

Each rule must:
	•	Accept raw section text
	•	Return matched text
	•	Return entity_type
	•	Return char_start
	•	Return char_end
	•	Inherit section label
	•	Avoid overlapping span duplication

Initial priority is high precision, not maximum recall.

Negation is not handled at this stage.

⸻

Step 5 — Negation Detection

5.1 Design Principle

Negation is implemented after entity span extraction.

⸻

5.2 Method
	•	Fixed token window: 5–7 tokens preceding entity
	•	Cue-based detection:
	•	no
	•	denies
	•	without
	•	not
	•	negative for

No syntactic parsing.
No dependency models.
No full NegEx implementation.

⸻

5.3 Output

Each entity record must include:
	•	negated: True | False

⸻

Step 6 — JSON Output Construction

6.1 Final Interface

extract(note_id, raw_text) → List[EntityRecord]

6.2 Required Output Fields

Each entity record must include:
	•	note_id
	•	entity_type
	•	matched_text
	•	section_title
	•	char_start
	•	char_end
	•	negated

Optional fields may include:
	•	rule_id
	•	confidence (always deterministic in Phase 2)

⸻

6.3 Constraints
	•	No duplicate spans
	•	No overlapping entity spans
	•	Stable deterministic ordering
	•	JSON schema compliance

⸻

Step 7 — Deterministic Stability Testing

7.1 Test Sample

Manually inspect 20–30 representative ICU notes.

Include:
	•	Long notes
	•	Dense numeric notes
	•	Sparse narrative notes
	•	Addendum-heavy notes

⸻

7.2 Validation Checks
	•	Span alignment correctness
	•	Section assignment correctness
	•	Negation flag correctness
	•	No crashes
	•	No malformed JSON
	•	No uncontrolled rule triggering

⸻

Phase Boundary

Phase 2 is complete when:
	•	Deterministic extraction works reliably on sample notes
	•	Span alignment is verified
	•	All four entity types are extractable
	•	JSON output is stable and reproducible

Phase 2 does not include:
	•	Transformer validation
	•	Corpus-wide execution
	•	Statistical evaluation
	•	Clinical outcome modelling
	•	Deployment or CI/CD

Those belong to subsequent phases.