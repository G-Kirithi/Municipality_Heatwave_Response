# System Architecture Documentation

## Overview
The **Food-Borne Illness Complaint Clustering and Investigation Tool** is a production-style municipal platform designed for public health departments responding to heatwave conditions among vulnerable residents.

```
                  +-----------------------------------+
                  |   5,000 Synthetic Complaints CSV  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Preprocessing & Validation Engine|
                  |  - Clean Text / Missing Flags     |
                  |  - Timestamp Sanity Auditing      |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Duplicate Detection Engine      |
                  |  - Exact Hash Grouping            |
                  |  - Near-Duplicate TF-IDF Similarity|
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Multi-Signal Similarity Engine  |
                  |  - Text (30%), Venue (20%)        |
                  |  - Time (20%), Symptoms (15%)     |
                  |  - Food (10%), Inspection (5%)    |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     Clustering Engine             |
                  |  - Agglomerative / DBSCAN         |
                  |  - Giant Cluster Protection       |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
| Common-Source Scoring |                       | Priority Risk Model   |
| - Spatial / Temporal  |                       | - Volume / Severity   |
| - Food Concentration  |                       | - Vulnerable Resident |
+-----------------------+                       +-----------------------+
            \                                               /
             \----------------------v----------------------/
                                    |
                                    v
                  +-----------------------------------+
                  |   Resilient Gemini AI Service     |
                  |  - Google Gemini API (gemini-2.5) |
                  |  - Structured Rule Fallback       |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Streamlit Interactive Dashboard & |
                  | SQLite Action Escalation Manager |
                  +-----------------------------------+
```

## Core Modules
1. `src/data_generator.py`: Generates realistic spatial-temporal datasets with ground-truth clusters and edge cases.
2. `src/preprocessing.py`: Cleans complaints and flags missing data / timestamp anomalies.
3. `src/duplicate_detection.py`: Deduplicates complaint bursts before clustering.
4. `src/feature_engineering.py`: Computes custom pairwise distance matrix $D = 1 - S$.
5. `src/clustering.py`: Clusters deduplicated complaints and builds cluster summaries.
6. `src/common_source.py` & `src/risk_scoring.py`: Evaluates outbreak likelihood and prioritizes investigation urgency.
7. `src/gemini_service.py`: Generates structured epidemiologist briefings with fallback support.
8. `src/followup.py` & `database/`: Tracks investigation lifecycle (`OPEN` -> `DUE SOON` -> `OVERDUE` -> `ESCALATED` -> `RESOLVED`).
