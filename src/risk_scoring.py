import pandas as pd
import numpy as np

def calculate_cluster_risk_priority(cluster_df, cluster_summary, common_source_res):
    """
    Computes a transparent priority risk score (0-100) for a detected cluster.
    
    Factors:
    - Volume & Velocity (25%)
    - Symptom Severity (25%)
    - Heatwave Vulnerable Area Flag (20%)
    - Common-Source Likelihood (15%)
    - Venue Inspection Concerns (15%)
    
    Outputs: LOW, MEDIUM, HIGH, CRITICAL.
    """
    n = len(cluster_df)
    
    # 1. Volume Score (0-100)
    volume_score = min(100.0, (n / 30.0) * 100.0)
    
    # 2. Symptom Severity Score (0-100)
    severe_keywords = ["severe", "fever", "dehydration", "chills", "hospital", "violent"]
    sym_text = " ".join([str(s).lower() for s in cluster_df["symptoms"].tolist()])
    severity_hits = sum(sym_text.count(kw) for kw in severe_keywords)
    severity_score = min(100.0, (severity_hits / (n + 1)) * 120.0 + 30.0)

    # 3. Vulnerable Area Heatwave Flag Score (0-100)
    vuln_count = cluster_df["vulnerable_area_flag"].sum()
    vuln_pct = (vuln_count / float(n)) if n > 0 else 0
    vuln_score = 100.0 if vuln_pct > 0.4 else (50.0 if vuln_count > 0 else 10.0)

    # 4. Common Source Likelihood Score (0-100)
    common_score = common_source_res.get("common_source_score", 0.5) * 100.0

    # 5. Inspection Concern Score (0-100)
    insp_scores = cluster_df["inspection_score"].dropna()
    avg_insp = insp_scores.mean() if not insp_scores.empty else 85
    failures = sum(cluster_df["inspection_status"] == "FAIL")
    insp_risk_score = min(100.0, (100.0 - avg_insp) + (failures * 15.0))

    # Overall Priority Score Calculation
    total_priority = (
        0.25 * volume_score +
        0.25 * severity_score +
        0.20 * vuln_score +
        0.15 * common_score +
        0.15 * insp_risk_score
    )
    total_priority = round(min(100.0, max(0.0, total_priority)), 1)

    if total_priority >= 75.0:
        priority_label = "CRITICAL"
        due_hours = 12
    elif total_priority >= 55.0:
        priority_label = "HIGH"
        due_hours = 24
    elif total_priority >= 35.0:
        priority_label = "MEDIUM"
        due_hours = 48
    else:
        priority_label = "LOW"
        due_hours = 72

    factor_breakdown = {
        "Volume & Scale": f"{volume_score:.1f} / 100 (Complaints: {n})",
        "Symptom Severity": f"{severity_score:.1f} / 100",
        "Vulnerable Resident Risk": f"{vuln_score:.1f} / 100 (Vulnerable flag count: {int(vuln_count)})",
        "Common-Source Signal": f"{common_score:.1f} / 100",
        "Inspection Risk": f"{insp_risk_score:.1f} / 100 (Avg Venue Score: {avg_insp:.1f})"
    }

    return {
        "priority_score": total_priority,
        "priority_label": priority_label,
        "recommended_due_hours": due_hours,
        "factor_breakdown": factor_breakdown
    }
