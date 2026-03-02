# Phase 2 — Deterministic Rule-Based Extraction

## Objective

Phase 2 implements the deterministic, rule-based extraction of structured information from the frozen ICU early-note corpus (162,296 notes across 32,910 ICU stays). 

This phase leverages the structural insights and validation from Phase 1 to:

- Define precise section segmentation rules (headers, numeric blocks, flowsheets, addenda)
- Normalize de-identification and EMR artefacts (including `[** ... **]` masking and trailing reference/JavaScript blocks)
- Extract numeric, physiologic, and intervention-related content using robust regex patterns
- Produce structured JSON output that preserves all clinically relevant sections while separating metadata, narrative, and tabular elements

Phase 1 established that:

- Colon-based headers are prevalent, consistent, and reliable as segmentation anchors  
- Uppercase section blocks provide optional structural reinforcement  
- Numeric patterns (labs, vitals, BP) are dense, interpretable, and structurally bounded  
- De-identification tokens and EMR artefacts are systematic and separable  
- Macro-level variability (note length, line count, token density) is large but structurally regular  

Without Phase 1, deterministic extraction would risk misaligned parsing, incomplete section capture, and failure on extreme or rare note structures. Phase 1 provides empirical justification that rule-based segmentation is feasible, robust, and scalable across the full corpus.

---

## Scope

Phase 2 covers:

1. **Header Segmentation**
   - Colon-terminated headers (`Assessment:`, `Plan:`, `NEURO:`)  
   - Uppercase system blocks with optional numbering  
   - Addendum and protected section isolation  

2. **Numeric and Flowsheet Extraction**
   - BP/MAP values, vitals, labs, infusion rates, medication dosing  
   - ASCII-style vertical tables and inline numeric clusters  
   - Regex-based extraction ensures coverage across extreme density and alignment variants  

3. **Artefact Normalization**
   - `[** ... **]` de-identification token removal  
   - Trailing EMR reference blocks and JavaScript/link fragments trimmed  
   - Line wrapping, whitespace, and minor formatting inconsistencies handled deterministically  

4. **Structured JSON Schema Population**
   - Section-level hierarchy preserved  
   - Numeric and physiologic measurements linked to contextual section  
   - Addenda and attestations separately labeled  
   - All extracted elements mapped to schema fields (e.g., `section_title`, `section_text`, `numeric_values`)  

5. **Deterministic Output**
   - Fully reproducible JSON per note  
   - No ML-based inference or transformer-based validation applied at this stage  

---

## Rationale

Phase 2 is strictly rule-based to ensure:

- Extraction reproducibility  
- Full auditability of segmentation logic  
- Robust handling of both sparse and extremely dense notes  
- Isolation of artefacts and non-clinical appended blocks  
- Compatibility with later Phase 3 transformer-based validation or downstream feature engineering  

All extraction rules are informed by Phase 1 structural profiling, extreme-note inspection, and quantitative validation. This guarantees that deterministic segmentation operates within empirically observed structural boundaries and that edge cases identified in Phase 1 are accounted for.

---

## Deliverables

- JSON-formatted notes with clearly delineated sections, numeric content, and artefact-free narrative  
- Full audit log of all applied extraction rules and preprocessing transformations  
- Section-specific counts for headers, numeric tokens, and artefacts per note  
- Documentation of fallback logic for zero-header or minimal notes  

Phase 2 output forms the foundation for Phase 3 transformer-based validation, downstream NLP feature extraction, and structured dataset generation for modeling or clinical analytics.