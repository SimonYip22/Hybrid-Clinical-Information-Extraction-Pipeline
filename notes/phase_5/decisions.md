# Phase 5 - Inference Pipeline and Deployment

## Objective

The objective of Phase 5 is to transition from model development and evaluation to a fully operational, reproducible inference system that can be:

- Applied at scale to large clinical datasets (`icu_corpus.csv`)
- Used for individual or batch inference  
- Exposed as a deployable service  

This phase consolidates all prior work into a single unified pipeline that:

- Integrates rule-based extraction and transformer validation  
- Produces structured, high-quality clinical entity outputs
- Maintains consistency with the evaluated pipeline behaviour (Phase 4)  

The phase has three core goals:

1. **Pipeline Construction**
   - Build a reusable, modular inference pipeline
   - Ensure outputs follow the validated schema used in rule-based extraction  

2. **Large-Scale Dataset Generation**
   - Apply the pipeline to the full ICU corpus (~160K reports)  
   - Demonstrate scalability and real-world applicability  

3. **Deployment**
   - Expose the pipeline via a simple API (FastAPI)
   - Enable both single-report and batch inference  

---

## Full Pipeline Construction

### 1. Overview 

This stage defines the core system of the project: a unified inference pipeline that encapsulates the full two-stage architecture:

1. High-recall rule-based extraction
2. Precision-oriented transformer validation

The pipeline is implemented as a reusable module and serves as the single source of truth for all inference logic.

---

### Pipeline Design and Structure

In src/pipeline/

* extraction.py
* validation.py
* pipeline.py

In scripts/phase_5_inference/

* run_full_dataset.py

In app/

* main.py

✔ Proper separation of concerns

* core logic vs execution vs API

✔ Reusable ML pipeline

* not a one-off script

✔ Deployable architecture

* same code used in:
    * batch processing
    * real-time inference

✔ Production thinking

* modular
* testable
* extensible
You are now transitioning from “analysis project” → production-style ML system

This structure is exactly what makes your project:

* credible
* scalable
* high-signal for jobs

Piepline split:

pipeline/
  extraction.py
  validation.py
  pipeline.py

* clean separation
* reusable components
* easier to debug
* aligns with real ML systems
* higher signal for recruiters

* you understand pipeline abstraction
* you separate concerns properly
* you can scale systems

A portfolio project meant to demonstrate ML system design + deployment


---

### Pipeline Responsibilities

The pipeline performs the following steps sequentially:

1. **Input Handling**
   - Accept raw clinical text (single report)
2. **Rule-Based Extraction**
   - Generate candidate entities using pattern matching and rules  
   - Designed for **high recall**
3. **Transformer Validation**
   - Score each candidate using the trained model (`model_prob`)  
   - Apply threshold (0.549) to produce binary predictions (`model_pred`)  
   - Designed for **precision-oriented filtering**
4. **Output Formatting**
   - Insert predictions and probabilities into the predefined JSON schema  
   - Ensure compatibility with downstream modelling and analysis  

---

### Design Principles

The pipeline is explicitly designed to be:

- **Modular**
  - Rule-based and transformer components are independent  
  - Each stage can be improved without affecting the other  
- **Reusable**
  - Same pipeline used for:
    - Evaluation (already completed)
    - Dataset generation
    - Deployment  
- **Deterministic and Reproducible**
  - Fixed model and threshold  
  - Consistent outputs across runs  
- **Scalable**
  - Can process:
    - Single reports (interactive use)
    - Large corpora (batch processing)

---

### Output Format

The pipeline produces **structured JSON outputs**:

- Each report → structured entity list  
- Each entity includes:
  - Extracted text  
  - Entity type  
  - Rule-based metadata  
  - `model_prob` (confidence)  
  - `model_pred` (final decision)  

This format:
- Preserves full information for downstream tasks  
- Avoids premature flattening (CSV not required at this stage)  
- Supports flexible future use (ML, analytics, auditability)

---

# Large-Scale Dataset Generation

## 1. Overview

This stage applies the constructed pipeline to the full ICU dataset (~160K reports) to generate a **large-scale structured clinical entity dataset**.

This is not a new system, but a **direct application of the pipeline at scale**.

---

### Purpose

- Demonstrate **scalability** of the pipeline  

- Generate a **real-world dataset** suitable for:

  - Machine learning  

  - Clinical analysis  

  - Further research  

- Provide evidence of:

  - End-to-end system functionality  

  - Practical utility beyond evaluation datasets  

---

### Process

1. Load full ICU corpus  

2. Iterate over reports  

3. Apply `pipeline.py` to each report  

4. Collect outputs  

---

### Output

- Stored as **JSONL (JSON Lines)**:

  - One JSON object per report  

  - Efficient for large-scale processing  

  - Standard format in NLP pipelines  

Optional:

- CSV conversion may be performed later for specific analyses  

- Not required for core pipeline functionality  

---

### Key Characteristics

- **Identical logic** to evaluation pipeline  

- No retraining or modification  

- Only difference: **data scale**

This ensures:

> The large-scale dataset faithfully reflects the validated behaviour observed in Phase 4.

---

# Deployment Pipeline

## 1. Overview

This stage exposes the pipeline as a **usable inference service**, enabling:

- Single-report inference (interactive use)

- Batch inference (multiple reports)

The deployment reuses the same `pipeline.py` module, ensuring:

> No duplication of logic between development, evaluation, and production

---

### Deployment Approach

A lightweight API is implemented using **FastAPI**, providing:

#### Endpoint 1: Single Report

- Input: raw clinical text  

- Output: structured entity JSON  

#### Endpoint 2: Batch Inference (optional)

- Input: list of reports  

- Output: list of structured outputs  

---

### Purpose

- Demonstrate **real-world usability** of the system  

- Provide a **clean interface** for external use  

- Show evidence of:

  - Deployment capability  

  - API design  

  - Reproducible inference  

---

### Design Principles

- **Thin wrapper over pipeline**

  - No additional logic introduced  

  - Calls `pipeline.py` directly  

- **Consistency**

  - Outputs identical to batch generation and evaluation  

- **Simplicity**

  - No unnecessary front-end required  

  - Focus on functionality and reproducibility  

---

Deployment: Final Method (Fixed Scope)

1. What you are building

A stateless inference API that exposes your pipeline.

Nothing more.

⸻

2. Tech Stack (Do not deviate)

Backend

* Python
* FastAPI

Server

* Uvicorn

Containerisation

* Docker

Hosting

* Cloud run GCP

CI/CD

* GitHub Actions

⸻

3. What you will implement (strict scope)

3.1 Core API (FastAPI)

You will implement exactly 2 endpoints:

1. /predict

* Input: single clinical report (string)
* Output: structured JSON (your pipeline output)

2. /health

* Returns: simple status ({"status": "ok"})

That’s it.

❌ No batch endpoint
❌ No frontend
❌ No authentication
❌ No database

This setup proves:

Engineering

* Modular pipeline design
* Separation of concerns
* Reusable inference logic

ML

* End-to-end deployment of trained model
* Real inference system (not notebook)

DevOps

* Containerisation (Docker)
* CI/CD integration
* Cloud deployment

This is exactly what recruiters want.

User → API (/predict)
        ↓
   FastAPI app
        ↓
   pipeline.py
        ↓
Rule-based → Transformer → Threshold
        ↓
   JSON output

Client
  ↓
Cloud Run (FastAPI)
  ↓
pipeline.py
  ↓
Rule-based → Transformer → Threshold
  ↓
Structured JSON

   13. Why this is optimal

This approach is:

* Minimal → avoids burnout
* Correct → matches industry patterns
* High signal → ticks all boxes recruiters care about
* Reusable → same pipeline powers everything

⸻

Final instruction

Do exactly this:

1. Finish pipeline.py
2. Wrap it with FastAPI
3. Add Docker
4. Deploy on Railway
5. Add GitHub Actions
---

# Diagrams (Important)

You should include the following diagrams:

### 1. Full Pipeline Diagram (REQUIRED)

Yes — you need this.

It should show:

Raw Text
↓
Rule-Based Extraction (High Recall)
↓
Candidate Entities
↓
Transformer Validation (model_prob)
↓
Threshold (0.549)
↓
Final Entities (JSON Output)

This is the **most important diagram in your entire project**.

---

### 2. Transformer Architecture Diagram (REQUIRED)

- Shows model internals (already planned)

---

### 3. Training Pipeline Diagram (REQUIRED)

- Data → training → validation → threshold tuning  

---