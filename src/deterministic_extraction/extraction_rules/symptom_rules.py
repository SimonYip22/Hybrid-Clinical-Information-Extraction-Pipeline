import re
from typing import List, Dict

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------

SYMPTOM_SECTIONS = {
    "chief complaint",
    "hpi",
    "review of systems",
}

NEGATION_TERMS = {
    "no", "denies", "denied", "without", "not", "negative for"
}

NEGATION_WINDOW = 5


# ------------------------------------------------------------
# 2. SYMPTOM CONCEPT MAP (NOT KEYWORDS)
# ------------------------------------------------------------

SYMPTOM_PATTERNS = {
    "pain": [
        r"\b(chest pain|abdominal pain|abd pain|back pain|neck pain)\b",
        r"\bpain\b"
    ],
    "dyspnoea": [
        r"\b(short of breath|shortness of breath|sob|dyspnea|dyspnoea)\b"
    ],
    "syncope": [
        r"\b(syncope|fainting|passing out|collapse|collapsed|syncopal episode|loc|loss of consciousness)\b"
    ],
    "nausea_vomiting": [
        r"\b(nausea|vomiting|vomit|n/v|n+v)\b"
    ],
    "fatigue": [
        r"\b(fatigue|tiredness|lethargy|lethargic)\b"
    ],
    "dizziness": [
        r"\b(dizziness|lightheadedness|lightheaded|dizzy)\b"
    ],
    "fever": [
        r"\b(fever|febrile|pyrexia|temperature|high temperature)\b"
    ],
    "cough": [
        r"\b(cough|productive cough|dry cough|wet cough|coughing|nonproductive cough)\b"
    ],
    "diarrhoea": [
        r"\b(diarrhea|diarrhoea|loose stools|watery stools)\b"
    ],
    "confusion": [
        r"\b(confusion|confused|altered mental state|altered mental status|altered consciousness|disorientation|disoriented)\b"
    ],
}


# ------------------------------------------------------------
# 3. NEGATION DETECTION
# ------------------------------------------------------------

def is_negated(text: str, start_idx: int) -> bool:
    tokens = text[:start_idx].split()
    window = tokens[-NEGATION_WINDOW:]

    return any(term in window for term in NEGATION_TERMS)


# ------------------------------------------------------------
# 4. MAIN EXTRACTION FUNCTION
# ------------------------------------------------------------

def extract_symptoms(note_id: str, text: str, section: str) -> List[Dict]:
    results = []

    if section.lower() not in SYMPTOM_SECTIONS:
        return results

    lowered_text = text.lower()

    for concept, patterns in SYMPTOM_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, lowered_text):

                start, end = match.span()
                span_text = text[start:end]

                negated = is_negated(lowered_text, start)

                results.append({
                    "note_id": note_id,
                    "entity_text": span_text,
                    "entity_type": "SYMPTOM",
                    "concept": concept,
                    "char_start": start,
                    "char_end": end,
                    "section": section,
                    "negated": negated
                })

    return results