"""
clinical_condition_rules.py

Purpose:
    Deterministic (rule-based) extraction of CLINICAL_CONDITION entities from clinical note text.
    Uses regex pattern matching to generate candidate clinical condition mentions without contextual interpretation.

Pipeline Role (Phase 2):
    - Recall-oriented, lower-precision candidate generator
    - Produces span-aligned clinical condition candidates for downstream transformer validation

Workflow:
    1. Filter relevant sections (clinical-condition-dense regions)
    2. Split text into sentences
    3. For each sentence:
        - Apply concept-specific regex patterns to each sentence
        - Extract all matching spans (no concept-level deduplication)
        - Remove exact duplicate spans (same start, end, concept)
        - Output structured candidate entities

Output:
    List of structured CLINICAL_CONDITION entities, each containing:
        - Metadata (note_id, subject_id, etc.)
        - Extracted span (entity_text, concept)
        - Provenance (character offsets, sentence, section)
        - Signal (negated)
        - Placeholder validation fields for transformer classification

Design Principles:
    - Prioritises recall over precision at extraction stage
    - Intentionally broad regex patterns to over-generate candidates (recall-first strategy)
    - Allows multiple candidates per concept per sentence
    - Does NOT collapse semantically similar mentions
    - Maintains exact span traceability for auditability
    - Delegates all contextual interpretation to transformer layer
        - Filtering false positives
        - Determining whether clinical condition is present, absent, historical, or active
        - Resolving ambiguity in context and temporal references
"""

from typing import Dict
import re

# Import for sentence splitting
from deterministic_extraction.sentence_segmentation import split_into_sentences

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------

# Target sections for clinical condition extraction
TARGET_CLINICAL_CONDITION_SECTIONS = {
    "assessment and plan",
    "assessment",
    "hpi",
    "chief complaint",
}

# ------------------------------------------------------------
# 2. CLINICAL_CONDITION PATTERNS
# ------------------------------------------------------------

# Concept-level clinical condition candidates mapped to regex patterns for detection
CLINICAL_CONDITION_PATTERNS = {

    "infection": [
        r"\b(sep(sis|tic)|infect(ed|ion)|bacter(a)?emia|pneumonia(s)?|urinary tract infection|uti|(endo|myo|peri)carditis|meningitis)\b"
    ],
    "shock": [
        r"\b((septic|cardiogenic|hypovol(a)?emic|distributive|hypotensive|neurogenic) shock|anaphyla(xis|ctic))\b"
    ],
    "respiratory": [
        r"\b(resp(iratory)? failure|acute respiratory distress syndrome|ards|hypox(a)?emi(a|c)|hypercapni(a|c)|pneumothorax|h(a)?emothorax|pleural effusion|pulmonary (o)?edema|aspiration pneumonitis)\b"
    ],
    "cardiovascular": [
        r"\b(myocardial infarct(ion)?|mi|acute coronary syndrome|acs|unstable angina|heart failure|hf|acute ventricular failure|avf|hypertensive crisis|cardiomyopathy)\b"
    ],
    "arrhythmia": [
        r"\b(arrhythmia(s)?|a(trial)?[- ]fib(rillation)?|af|v(entricular)? tachy(cardia)?|v[- ]tach(y)?||supraventricular tachycardia|svt|a(trial)? tachy(cardia)?)\b"
    ],
    "renal_failure": [
        r"\b(renal failure|acute kidney injury|aki)\b"
    ],
    "neurological": [
        r"\b(stroke|cerebrovascular accident|cva|transient ischemic attack|tia|seizure(s)?|epilep(sy|tic)|status epilepticus|encephalopath(y|ic))\b"
    ],
    "bleeding": [
        r"\b(h(a)?emorrhag(e|es|ing)|bleed(s|ing)?|h(a)?ematoma|sah)\b"
    ],
    "gastrointestinal": [
        r"\b((bowel|intestinal|gastrointestinal|gastric|gi|colon(ic)?) (perforat(ion|ed)|obstruct(ion|ed)|isch(a)?emia|infarct(ion)?|ulcer(ated|ation)?)|sbo|cholecystitis)\b"
    ],
    "metabolic": [
        r"\b(diabetic ketoacidosis|dka|(hypo|hyper)glyc(a)?emi(a|c)|(hypo|hyper)natr(a)?emi(a|c)|(hypo|hyper)kal(a)?emi(a|c)|(hypo|hyper)calc(a)?emi(a|c))\b"
    ],
    "hepatic_failure": [
        r"\b((hepatic|liver) failure|acute liver (injury|failure)|alf|ali)\b"
    ],
    "cardiac_arrest": [
        r"\b((cardiac|heart|sinus) arrest|asystole|ventricular fibrillation|vf(ib)?|pulseless v(entricular)? tachy(cardia)?|pulseless vt)\b"
    ],
    "vascular": [
        r"\b(aortic dissection|aortic (aneurysm|rupture)|aaa|deep vein thrombosis|dvt|pulmonary embol(ism|us)|pe|thromboembolism|vte)\b"
    ],
}

# ------------------------------------------------------------
# 3. EXTRACTION FUNCTION
# ------------------------------------------------------------

def extract_clinical_conditions(note_id: str, subject_id: str, hadm_id: str, icustay_id:str,
                                section: str, text:str) -> list[Dict]:
    """
    Extract CLINICAL_CONDITION entities from a section of clinical text using deterministic rules.

    Inputs:
        note_id, subject_id, hadm_id, icustay_id:
            Identifiers for traceability and downstream grouping

        section: str
            Section name of the note (used to filter relevant content)

        text: str
            Raw section text to process

    Processing Steps:
        1. Section filtering: 
            Only process sections likely to contain clinical condition information
        2. Sentence segmentation: 
            Split section text into sentences for more precise pattern matching
        3. Pattern matching: 
            For each sentence, apply regex patterns for each clinical condition concept
        4. Span extraction: 
            For each match, extract the exact text span and character offsets
        5. Deduplication:
            Only exact duplicate spans (same start, end, concept) are removed
            No concept-level or semantic deduplication is performed
        6. Result structuring: 
            Compile results into a structured format with metadata and placeholders

    Output:
        List[Dict]:
            Each dict represents one CLINICAL_CONDITION entity with:
                - Span-aligned extraction (entity_text, char_start, char_end)
                - Concept label
                - Context (sentence, section)
                - Negation signal
                - Validation scaffold
    """

    # Initialise results list
    results = []

    # 1. Section filtering 
    if section.lower() not in TARGET_CLINICAL_CONDITION_SECTIONS:
        return results
    
    # 2. Sentence segmentation
    sentence_spans = split_into_sentences(text)

    # Loop through each sentence
    for s in sentence_spans:

        # Extract sentence text and its starting character offset in the section
        sent_text = s["sentence"]
        sent_start = s["start"]

        lowered_sent = sent_text.lower()

        # To avoid exact duplicate extractions
        seen_spans = set()

        # 3. Pattern matching
        for concept, patterns in CLINICAL_CONDITION_PATTERNS.items():

            # Apply each regex pattern associated with the current concept
            for pattern in patterns:

                # 4. Span extraction
                # Use re.finditer to find all non-overlapping matches of the pattern in the sentence
                for match in re.finditer(pattern, lowered_sent): 
                    
                    # Calculate character offsets of the match in the context of the entire section text
                    start_idx = sent_start + match.start()
                    end_idx = sent_start + match.end()

                    # 5. Deduplication
                    # Avoid duplicate spans (same start, same end, same concept) within the same sentence
                    span_key = (start_idx, end_idx, concept)
                    if span_key in seen_spans:
                        continue
                    
                    # Add new span to set toprevent duplicates
                    seen_spans.add(span_key)

                    # Extract the exact text span from the original sentence text using the character offsets
                    span_text = text[start_idx:end_idx]

                    # 6. Result structuring
                    results.append({
                        "note_id": note_id,
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "icustay_id": icustay_id,

                        "entity_text": span_text,
                        "concept": concept,
                        "entity_type": "CLINICAL_CONDITION",

                        "char_start": start_idx,
                        "char_end": end_idx,
                        "sentence_text": sent_text,
                        "section": section,

                        "negated": None, # Negation detection doesn't exist for this entity

                        "validation": {
                            "is_valid": None,
                            "confidence": 0.0,
                            "task": "clinical_condition_active"
                        }
                    })
    
    return results