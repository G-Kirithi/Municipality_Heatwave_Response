import pandas as pd
import numpy as np

def evaluate_common_source(cluster_df, cluster_summary):
    """
    Evaluates whether a given cluster of complaints represents a potential common-source outbreak.
    
    Produces structured evidence, likelihood scores, and clear non-medical wording.
    """
    n = len(cluster_df)
    evidence_points = []
    
    # 1. Volume & Unique Complaints Evidence
    unique_count = cluster_summary.get("unique_complaint_count", n)
    evidence_points.append(f"Received {n} total complaints ({unique_count} unique independent incidents).")

    # 2. Venue Concentration Signal
    venues = cluster_df["venue_name"].value_counts()
    primary_venue = venues.index[0] if not venues.empty else "UNKNOWN"
    venue_pct = (venues.iloc[0] / float(n)) * 100 if not venues.empty else 0.0
    
    if primary_venue != "UNKNOWN" and venue_pct >= 70.0:
        venue_signal = 0.9
        evidence_points.append(f"Strong venue concentration: {venue_pct:.1f}% of complaints trace directly to '{primary_venue}'.")
    elif primary_venue != "UNKNOWN" and venue_pct >= 40.0:
        venue_signal = 0.6
        evidence_points.append(f"Moderate venue concentration: {venue_pct:.1f}% linked to '{primary_venue}'.")
    else:
        venue_signal = 0.3
        evidence_points.append("Multi-venue distribution across shared neighborhood zone.")

    # 3. Temporal Window Concentration Signal
    dts = cluster_df["parsed_complaint_dt"].dropna()
    if len(dts) > 1:
        span_hours = (max(dts) - min(dts)).total_seconds() / 3600.0
        if span_hours <= 36.0:
            time_signal = 0.95
            evidence_points.append(f"Tight temporal window: Complaints reported within a {span_hours:.1f}-hour span during heatwave.")
        elif span_hours <= 72.0:
            time_signal = 0.7
            evidence_points.append(f"Moderate temporal concentration: Complaints spanned {span_hours:.1f} hours.")
        else:
            time_signal = 0.4
            evidence_points.append(f"Diffused temporal spread across {span_hours:.1f} hours.")
    else:
        time_signal = 0.5
        evidence_points.append("Single timestamp cluster record.")

    # 4. Food Item & Symptom Overlap
    foods = cluster_df["food_item"].value_counts()
    top_food = foods.index[0] if not foods.empty else "Various"
    food_pct = (foods.iloc[0] / float(n)) * 100 if not foods.empty else 0
    if food_pct >= 50.0:
        food_signal = 0.85
        evidence_points.append(f"High food item overlap: {food_pct:.1f}% consumed '{top_food}'.")
    else:
        food_signal = 0.4
        evidence_points.append(f"Common food mentions: {cluster_summary.get('common_foods', 'N/A')}.")

    # 5. Inspection History Evidence
    insp_scores = cluster_df["inspection_score"].dropna()
    avg_score = insp_scores.mean() if not insp_scores.empty else 100
    violations = [str(v) for v in cluster_df["previous_violation"].dropna().unique() if str(v) and str(v) not in ["None", "nan"]]
    if avg_score < 70 or len(violations) > 0:
        insp_signal = 0.8
        v_str = ", ".join(violations[:2]) if violations else "Substandard inspection score"
        evidence_points.append(f"Supporting inspection evidence: Venue score {avg_score:.1f}. Active violations: {v_str}.")
    else:
        insp_signal = 0.3
        evidence_points.append("Inspection records show prior satisfactory compliance.")

    # Calculate overall Common-Source Likelihood Score
    likelihood_score = (
        0.30 * venue_signal +
        0.30 * time_signal +
        0.20 * food_signal +
        0.20 * insp_signal
    )
    likelihood_score = min(1.0, max(0.0, likelihood_score))

    if likelihood_score >= 0.75:
        likelihood_label = "High Likelihood Common-Source Cluster"
    elif likelihood_score >= 0.50:
        likelihood_label = "Moderate Likelihood Common-Source Cluster"
    else:
        likelihood_label = "Low Likelihood / Diffused Cluster"

    return {
        "common_source_score": round(float(likelihood_score), 2),
        "common_source_label": likelihood_label,
        "evidence_points": evidence_points,
        "primary_venue": primary_venue,
        "primary_food": top_food,
        "confidence": round(min(0.98, likelihood_score + 0.1), 2)
    }
