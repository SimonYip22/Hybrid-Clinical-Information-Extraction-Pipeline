# Phase 1 — Corpus Profiling Summary

## Objective

Phase 1 establishes structural feasibility for deterministic extraction from the adult ICU early-note corpus. Its scope is to assess whether the corpus contains consistent, predictable patterns in section headers, numeric content, and artefacts, sufficient to support rule-based candidate generation in Phase 2.

Specifically, Phase 1:

- Defines corpus boundaries and filtering criteria to isolate relevant ICU early notes.
- Performs manual inspection and extreme-boundary analysis to identify dominant structural archetypes.
- Quantitatively profiles structural signals (header prevalence, numeric density, de-identification artefacts, note length) to validate consistency across the corpus.
- Confirms the presence of templated sections, flowsheets, and repeatable patterns suitable for deterministic parsing.
- Assesses artefact patterns (de-identification, EMR reference blocks, JavaScript fragments) for predictable preprocessing.
- Determines structural heterogeneity and extreme variability without compromising parsing assumptions.

---

## Scope and Limitations

Phase 1 does not implement extraction logic, JSON schema, or NLP-based entity generation. Its output provides evidence that deterministic rule-based extraction is structurally justified and informs:

- Header and section segmentation rules
- Numeric and physiologic pattern extraction feasibility
- Artefact normalisation requirements
- Scope for robust JSON schema design

Completion of Phase 1 ensures that Phase 2 (rule-based deterministic extraction) can proceed with high confidence in architectural validity and minimal risk of structural failure.

---

## Files Generated

src/data_processing:

- build_corpus.py: Implements filtering logic to construct the early ICU note corpus based on defined criteria.
- manual_sample.py: Script for manual inspection of a representative sample of notes to identify structural archetypes and patterns.
- quant_profiling.py: Script for quantitative profiling of structural signals across a larger sample of notes to confirm consistency and identify extreme cases.

notes/phase_1:

- summary.md: This document summarises the objectives, scope, methods, and outcomes of Phase 1 corpus profiling.   
- decisions.md: Documents key decisions and analysis made during Phase 1 regarding corpus definition, structural assumptions, and profiling outcomes.
- profiling_boundary_extremes.ipynb: Jupyter notebook containing script to print notes of metric structural extremes for manual inspection.

data/raw:
- ICUSTAYS.csv: Contains ICU stay records with relevant columns for filtering and linking.
- PATIENTS.csv: Contains patient demographic information for filtering and linking.
- NOTEEVENTS.csv: Contains clinical notes with relevant columns for filtering and analysis.

data/processed:
- icu_corpus.csv: Output of build_corpus.py containing the filtered early ICU notes with necessary columns for profiling and downstream extraction.

data/sample:
- manual_sample_30.csv: Sample of 30 notes selected for manual inspection to identify structural archetypes and patterns.
- profiling_sample_500.csv: Sample of 500 notes selected for quantitative profiling to confirm consistency and identify extreme cases.
- profiling_per_note.csv: Output of quant_profiling.py containing structural signal metrics for each note in the profiling sample.
- profiling_summary.csv: Summary statistics of structural signals across the profiling sample.


---

## Adult ICU Early Report Corpus Construction

### 1. Purpose

- Formal specification of cohort definition, filtering logic, and implementation reproducibility for early ICU notes.  
- Defines anchors (`ICUSTAY_ID`) and ensures proper linkage to patient and hospital admission data (`SUBJECT_ID`, `HADM_ID`).  
- Serves as structural foundation for profiling and downstream extraction.

---

### 2. Data Sources and Columns

- **PATIENTS:** `SUBJECT_ID`, `DOB`, `GENDER`  
- **ICUSTAYS:** `SUBJECT_ID`, `HADM_ID`, `ICUSTAY_ID`, `FIRST_CAREUNIT`, `INTIME`, `OUTTIME`  
- **NOTEEVENTS:** `SUBJECT_ID`, `HADM_ID`, `CHARTTIME`, `CATEGORY`, `ISERROR`, `TEXT`  

Minimal columns prevent memory overhead and accidental feature leakage.

---

### 3. Cohort Definition

- Adult ICU stays (`AGE ≥ 18`, `LOS_HOURS ≥ 24`)  
- Excluded neonatal units (`NICU`)  
- Early ICU notes: `INTIME ≤ CHARTTIME ≤ INTIME + 24h`  
- Allowed note categories: `physician`, `nursing`, `nursing/other`  
- Exclude explicit errors: `ISERROR != 1`  

---

### 4. Phase 1 Structural Analysis Scope

Phase 1 focuses exclusively on **structural feasibility**, not extraction:

- Manual inspection of representative sample (30 notes) to identify archetypes, headers, flowsheets, numeric patterns, and artefacts.  
- Quantitative profiling of 500 randomly sampled notes to confirm prevalence, density, and extreme boundary cases.  
- Boundary inspection of extreme notes (45 unique structural extremes) to confirm parser robustness.  

Phase 1 outputs:

1. Confirmed header formats suitable for deterministic segmentation.  
2. Validated numeric patterns for physiologic and lab extraction.  
3. Documented de-identification tokens and EMR artefacts (including reference and JS/link fragments).  
4. Verified macro-level structural consistency across short and long notes.  

---

### 5. Phase 1 Outcomes

- Structural archetypes are **predictable and repetitive**, supporting rule-based extraction.  
- Colon-terminated headers present in >80% of notes; uppercase headers in ~50%.  
- Numeric and physiologic signals (labs, vitals, BP patterns) are sufficiently dense and consistently formatted.  
- Artefacts (de-identification, EMR references, JS fragments) are systematic and separable.  
- Boundary notes demonstrate parser stability under extremes; no structural failure modes observed.  

**Conclusion:** Phase 1 validates that deterministic rule-based extraction is feasible and justifiable. Phase 2 can safely implement rules for section segmentation, numeric extraction, and preprocessing, using Phase 1 findings as the architectural foundation.