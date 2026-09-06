import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN

def perform_clustering(df, distance_matrix, method="agglomerative", distance_threshold=0.45, min_samples=3, max_cluster_size=200):
    """
    Performs clustering on preprocessed complaints given a precomputed distance matrix.
    
    Prevents giant cluster formation (Edge Case 6) and handles no clusters / single sample cases (Edge Case 5).
    """
    df = df.copy()
    n = len(df)
    
    if n < 2:
        df["cluster_id"] = "NOISE"
        return df, []

    if method == "agglomerative":
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold
        )
        labels = clustering_model.fit_predict(distance_matrix)
    elif method == "dbscan":
        clustering_model = DBSCAN(
            eps=distance_threshold,
            min_samples=min_samples,
            metric="precomputed"
        )
        labels = clustering_model.fit_predict(distance_matrix)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Process Labels & Enforce Cluster Safeguards
    cluster_counts = pd.Series(labels).value_counts()
    
    final_labels = []
    cluster_id_mapping = {}
    cluster_counter = 1
    
    for lbl in labels:
        if lbl == -1:
            final_labels.append("NOISE")
        else:
            cnt = cluster_counts.get(lbl, 0)
            # Edge Case 6 safeguard: If cluster exceeds max_cluster_size, treat as unclustered noise/overflow
            if cnt < min_samples or cnt > max_cluster_size:
                final_labels.append("NOISE")
            else:
                if lbl not in cluster_id_mapping:
                    cluster_id_mapping[lbl] = f"CLUSTER-{cluster_counter:03d}"
                    cluster_counter += 1
                final_labels.append(cluster_id_mapping[lbl])

    df["cluster_id"] = final_labels

    # Build Cluster Summaries
    cluster_summaries = summarize_clusters(df)
    return df, cluster_summaries

def summarize_clusters(df):
    """
    Extracts structured summary objects for every detected cluster.
    """
    clustered = df[df["cluster_id"] != "NOISE"]
    if len(clustered) == 0:
        return []

    summaries = []
    grouped = clustered.groupby("cluster_id")
    
    for cid, group in grouped:
        complaint_count = len(group)
        unique_complaint_count = group["duplicate_group_id"].apply(lambda x: 1 if str(x) == "NONE" else 0).sum()
        if unique_complaint_count == 0:
            unique_complaint_count = complaint_count
            
        venues = [str(v) for v in group["venue_name"].dropna().unique() if str(v) != "UNKNOWN"]
        venue_str = ", ".join(venues) if venues else "Unknown Venue"
        
        # Time Range
        dts = group["parsed_complaint_dt"].dropna()
        if len(dts) > 0:
            min_t = min(dts).strftime("%Y-%m-%d %H:%M")
            max_t = max(dts).strftime("%Y-%m-%d %H:%M")
            time_range = f"{min_t} to {max_t}"
        else:
            time_range = "N/A"
            
        # Common Foods
        foods = group["food_item"].dropna().value_counts()
        common_foods = ", ".join(foods.head(3).index.tolist()) if not foods.empty else "Various Foods"
        
        # Dominant Symptoms
        all_syms = []
        for slist in group["symptoms_list"]:
            all_syms.extend(slist)
        sym_counts = pd.Series(all_syms).value_counts()
        dominant_symptoms = ", ".join([str(s).title() for s in sym_counts.head(4).index.tolist()]) if not sym_counts.empty else "N/A"
        
        # Inspection Summary
        insp_scores = group["inspection_score"].dropna()
        avg_insp = round(insp_scores.mean(), 1) if not insp_scores.empty else "N/A"
        violations = [str(v) for v in group["previous_violation"].dropna().unique() if str(v) and str(v) not in ["None", "nan"]]
        insp_summary = f"Avg Score: {avg_insp}. Violations: {', '.join(violations[:2]) if violations else 'None'}"
        
        # Vulnerable Area Check
        vuln_count = int(group["vulnerable_area_flag"].sum())
        
        summaries.append({
            "cluster_id": cid,
            "complaint_count": complaint_count,
            "unique_complaint_count": unique_complaint_count,
            "venues": venue_str,
            "time_range": time_range,
            "common_foods": common_foods,
            "dominant_symptoms": dominant_symptoms,
            "inspection_summary": insp_summary,
            "vulnerable_area_count": vuln_count,
            "location_zone": str(group["location_zone"].mode()[0]) if not group["location_zone"].empty else "Zone-1"
        })
        
    return summaries
