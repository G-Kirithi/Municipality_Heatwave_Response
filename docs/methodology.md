# Methodology Documentation

## Analytical Framework

### 1. Preprocessing & Duplicate Detection
- **Exact Matching:** Hashing normalized text combined with `venue_id` and purchase date.
- **Near-Duplicate Matching:** TF-IDF feature vectors evaluated via cosine similarity ($\ge 0.85$) within a 12-hour reporting window. Canonical complaint retention prevents artificial outbreak inflation.

### 2. Multi-Signal Pairwise Distance Representation
The similarity between complaints $i$ and $j$ is calculated as:

$$S(i, j) = w_1 S_{\text{text}}(i,j) + w_2 S_{\text{venue}}(i,j) + w_3 S_{\text{time}}(i,j) + w_4 S_{\text{symptoms}}(i,j) + w_5 S_{\text{food}}(i,j) + w_6 S_{\text{inspection}}(i,j)$$

Where:
- $S_{\text{text}}$: TF-IDF cosine similarity of self-reported symptom text ($w_1 = 0.30$).
- $S_{\text{venue}}$: 1.0 for identical venue, 0.5 for same zone ($w_2 = 0.20$).
- $S_{\text{time}}$: Exponential decay over time difference in hours: $\exp(-\Delta t / 24.0)$ ($w_3 = 0.20$).
- $S_{\text{symptoms}}$: Jaccard similarity of symptom sets ($w_4 = 0.15$).
- $S_{\text{food}}$: Jaccard/substring match on reported food items ($w_5 = 0.10$).
- $S_{\text{inspection}}$: Prior health inspection risk factor ($w_6 = 0.05$).

Pairwise distance is defined as:

$$D(i, j) = 1.0 - S(i, j)$$

### 3. Clustering & Safeguards
- **Algorithm:** Agglomerative Clustering (average linkage) or DBSCAN over the precomputed distance matrix.
- **Giant Cluster Safeguard:** Clusters containing more than 200 items or below 3 items are pruned/flagged to prevent over-merging.

### 4. Risk Prioritization Scoring
Priority score (0-100) combines Volume (25%), Symptom Severity (25%), Vulnerable Resident Flag (20%), Common-Source Likelihood (15%), and Inspection Risk (15%), classifying actions into LOW, MEDIUM, HIGH, and CRITICAL.
