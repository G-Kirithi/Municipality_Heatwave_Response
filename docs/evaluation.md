# Evaluation Framework & Benchmark Results

## Comparative Benchmarking Overview
The proposed multi-signal system was evaluated against a traditional simple baseline rule:
- **Baseline Rule:** Two complaints are linked if they share identical `venue_id`, fall within a 24-hour window, and share $\ge 1$ symptom.

## Metrics
- **Adjusted Rand Index (ARI):** Measures cluster similarity against ground truth annotations, adjusted for chance.
- **Normalized Mutual Information (NMI):** Quantifies mutual information shared between predicted and ground-truth clusters.
- **Precision, Recall, and F1-Score:** Evaluates duplicate matching accuracy and outbreak cluster member detection.

## Output Files
Full experimental execution outputs are automatically persisted to:
- `results/evaluation_results.csv`
- `results/evaluation_report.json`
