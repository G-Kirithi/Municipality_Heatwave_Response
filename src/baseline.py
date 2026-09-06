import pandas as pd
import numpy as np

def run_baseline_deduplication(df):
    """
    Simple Baseline Exact Duplicate Matching:
    Marks complaints as duplicates if they share identical normalized text + venue_id.
    """
    df = df.copy()
    df["baseline_is_duplicate"] = False
    df["baseline_dup_group"] = "NONE"
    
    seen = {}
    dup_counter = 1
    
    for idx, row in df.iterrows():
        text = row["clean_text"]
        venue = row["venue_id"]
        if not text or venue == "UNKNOWN":
            continue
        key = f"{venue}|{text}"
        if key in seen:
            df.loc[idx, "baseline_is_duplicate"] = True
            df.loc[idx, "baseline_dup_group"] = seen[key]
        else:
            group_id = f"BASE-DUP-{dup_counter:04d}"
            dup_counter += 1
            seen[key] = group_id
            df.loc[idx, "baseline_dup_group"] = group_id
            
    return df

def run_baseline_clustering(df, time_window_hours=24):
    """
    Fast Optimized Baseline Clustering Rule:
    Two complaints belong to the same cluster if:
    1. venue_id is identical (and not UNKNOWN)
    2. complaint/purchase timestamp within 24 hours
    3. at least 1 symptom overlaps
    """
    df = df.copy()
    n = len(df)
    baseline_cluster_ids = ["NOISE"] * n
    cluster_counter = 1
    
    # Group by venue_id to avoid N^2 loop across unrelated venues
    grouped_venues = df.groupby("venue_id")
    
    for venue_id, group in grouped_venues:
        if venue_id == "UNKNOWN" or len(group) < 3:
            continue
            
        indices = group.index.tolist()
        m = len(indices)
        adj = np.zeros((m, m), dtype=bool)
        
        timestamps = group["parsed_complaint_dt"].tolist()
        symptom_lists = group["symptoms_list"].tolist()
        
        for i in range(m):
            t_i = timestamps[i]
            sym_i = set(symptom_lists[i])
            if not t_i or not sym_i:
                continue
                
            for j in range(i + 1, m):
                t_j = timestamps[j]
                sym_j = set(symptom_lists[j])
                if not t_j or not sym_j:
                    continue
                    
                diff_hours = abs((t_i - t_j).total_seconds()) / 3600.0
                if diff_hours <= time_window_hours and bool(sym_i.intersection(sym_j)):
                    adj[i, j] = True
                    adj[j, i] = True
                    
        # Connected components
        visited = np.zeros(m, dtype=bool)
        for i in range(m):
            if not visited[i]:
                component = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in np.where(adj[curr])[0]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                            
                if len(component) >= 3:
                    cid = f"BASE-CLUSTER-{cluster_counter:03d}"
                    cluster_counter += 1
                    for idx_in_comp in component:
                        original_idx = indices[idx_in_comp]
                        baseline_cluster_ids[original_idx] = cid
                        
    df["baseline_cluster_id"] = baseline_cluster_ids
    return df
