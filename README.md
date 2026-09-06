# Municipal Food-Borne Illness Complaint Clustering & Investigation Tool

> **A production-style municipal analytics platform for public health departments responding to heatwaves among vulnerable residents.**

---

## 🌟 Key Features

- **Synthetic Outbreak Dataset Generator:** Generates 5,000 anonymized complaint records with realistic spatial-temporal patterns, ground-truth clusters, and duplicate bursts.
- **Complaint Deduplication Engine:** Combines exact hash matching with TF-IDF cosine similarity near-duplicate detection to eliminate complaint bursts from skewing outbreak counts.
- **Multi-Signal Similarity & Clustering:** Computes weighted pairwise distance matrices incorporating Text TF-IDF (30%), Venue Match (20%), Temporal Proximity (20%), Symptom Overlap (15%), Food Item Match (10%), and Health Inspection History (5%).
- **Transparent Risk & Common-Source Scoring:** Classifies potential common-source incidents into `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL` priority levels with explicit evidence factors.
- **Resilient AI Investigation Assistant:** Integrates Google Gemini API (`google-genai` SDK) to generate structured epidemiologist briefings, with automatic rule-based fallback when offline.
- **Persistent Action Tracker & Escalation Manager:** SQLite database managing investigation lifecycles (`OPEN` -> `DUE SOON` -> `OVERDUE` -> `ESCALATED` -> `RESOLVED`) with supervisor escalation triggers.
- **Interactive Streamlit Dashboard:** 5-tab web application for real-time municipal investigation and decision support.
- **Reproducible Evaluation & Error Analysis:** Automated benchmark script evaluating Baseline vs. Proposed clustering methods into `results/evaluation_results.csv` and `results/evaluation_report.json`.

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
Clone the repository and install dependencies in Python 3.11+:

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 2. Configure Gemini API (Optional)
Add your Google Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*Note: If `GEMINI_API_KEY` is not provided, the application automatically uses a structured rule-based fallback generator.*

### 3. Generate Synthetic Dataset
Generate 5,000 synthetic complaints with ground-truth clusters:

```bash
python -m src.data_generator
```

### 4. Run Evaluation Benchmark & Error Analysis
Compare the proposed system against the baseline method:

```bash
python -m src.evaluation
```
Outputs:
- `results/evaluation_results.csv`
- `results/evaluation_report.json`

### 5. Launch Interactive Streamlit Dashboard
Start the web dashboard:

```bash
streamlit run app.py
```

### 6. Run Automated Test Suite
Execute the full pytest suite:

```bash
python -m pytest tests/ -v
```

---

## 📁 Project Architecture

```
foodborne-investigation/
├── app.py                         # Streamlit Interactive Dashboard
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation & Quick Start
├── .env.example                   # Environment configuration template
├── data/                          # Datasets (synthetic raw & deduplicated)
│   └── synthetic/
│       └── synthetic_complaints.csv
├── results/                       # Benchmark outputs
│   ├── evaluation_results.csv
│   └── evaluation_report.json
├── database/                      # SQLite persistence layer
│   ├── db.py                      # Connection session factory
│   └── models.py                  # SQLAlchemy ORM models
├── src/                           # Business logic & analytics
│   ├── data_generator.py          # 5,000 synthetic dataset generator
│   ├── preprocessing.py           # Text cleaning, date sanity checks
│   ├── duplicate_detection.py     # Exact & near-duplicate matching
│   ├── feature_engineering.py     # Pairwise multi-signal distance matrix
│   ├── baseline.py                # Simple baseline matching rule
│   ├── clustering.py              # Agglomerative & DBSCAN algorithms
│   ├── common_source.py           # Common-source evidence evaluator
│   ├── risk_scoring.py            # Priority risk model
│   ├── gemini_service.py          # Gemini API client with fallback
│   ├── investigation.py           # Master pipeline orchestrator
│   ├── followup.py                 # Action status lifecycle & escalation
│   └── evaluation.py              # Benchmark execution script
├── tests/                         # Pytest unit tests
│   ├── test_duplicates.py
│   ├── test_clustering.py
│   ├── test_edge_cases.py
│   └── test_followup.py
└── docs/                          # Professional documentation
    ├── architecture.md
    ├── methodology.md
    ├── ethics.md
    ├── evaluation.md
    ├── deployment_checklist.md
    └── user_validation.md
```
