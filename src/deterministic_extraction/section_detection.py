"""
section_detection.py

Purpose:
    Extract clinically relevant sections from unstructured clinical notes
    using a deterministic, canonical-header-based approach.

    Only predefined canonical headers are recognised. These headers define
    section boundaries. All other text, including header-like patterns
    (e.g., subsections, vitals, labs), is treated as normal content.

Workflow:
    1. Split the clinical note into lines.
    2. Identify canonical headers only.
    3. Start a new section when a canonical header is encountered.
    4. Accumulate all subsequent text until the next canonical header.
    5. Include inline content after header colons where present.
    6. Return extracted sections as a dictionary.

Output:
    Dictionary mapping canonical section headers (lowercase) to text.

    Example:
    {
        "chief complaint": "...",
        "hpi": "...",
        "assessment": "...",
        "plan": "..."
    }

    All section headers are stored in lowercase to ensure consistent canonical representation.
"""

# ---------------------------------------------------------------------
# CANONICAL SECTION DEFINITIONS
# ---------------------------------------------------------------------

# Canonical section headers to extract, based on empirical analysis of the dataset, lowercased for case-insensitive matching
CANONICAL_HEADERS = [
    "plan",
    "assessment",
    "action",
    "response",
    "assessment and plan",
    "chief complaint",
    "hpi",
    "past medical history",
    "family history",
    "social history",
    "review of systems",
    "physical examination",
    "disposition"
]

CANONICAL_HEADER_SET = set(CANONICAL_HEADERS)


# ---------------------------------------------------------------------
# CANONICAL HEADER DETECTION FUNCTION
# ---------------------------------------------------------------------

def match_canonical_header(line):
    """
    Check if a line matches a canonical header.

    Supports:
    - "Header:"
    - "Header"
    - "Header: content"

    Returns:
        (header_lower, inline_text) or (None, None)
    """

    stripped = line.strip()

    # Empty line case
    if not stripped:
        return None, None # No header, no content

    # Split on first colon (if exists)
    if ":" in stripped:
        # Split into header and inline content
        header_part, rest = stripped.split(":", 1) # Split only on first colon
        # Clean header for matching
        header_clean = header_part.strip().lower()

        # Check if cleaned header is in canonical set
        if header_clean in CANONICAL_HEADER_SET:
            # Return header and inline content
            return header_clean, rest.strip()

    # No colon case (standalone header)
    header_clean = stripped.lower()
    if header_clean in CANONICAL_HEADER_SET:
        return header_clean, None

    # Not a canonical header
    return None, None


# ---------------------------------------------------------------------
# SECTION EXTRACTION FUNCTION
# ---------------------------------------------------------------------
def extract_sections(report):
    """
    Extract canonical sections using canonical-only boundaries.

    Rules:
    - Only canonical headers define section boundaries
    - Non-canonical header-like text is treated as normal content
    - Inline content after header colon is included
    """

    sections = {}
    current_header = None
    buffer = []

    for line in report.split("\n"):

        header, inline_text = match_canonical_header(line)

        if header:
            # Save previous section
            if current_header is not None:
                sections[current_header] = " ".join(buffer).strip()

            # Start new section
            current_header = header
            buffer = []

            # Add inline text if present
            if inline_text:
                buffer.append(inline_text)

            continue

        # Accumulate content
        if current_header is not None:
            clean = line.strip()
            if clean:
                buffer.append(clean)

    # Save final section
    if current_header is not None:
        sections[current_header] = " ".join(buffer).strip()

    return sections