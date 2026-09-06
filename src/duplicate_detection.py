import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def detect_duplicates(df, similarity_threshold=0.85, time_window_hours=12):
    """
    Fast Vectorized Duplicate Detection (Exact Hash + Sparse Matrix TF-IDF Cosine Thresholding).
    Scales efficiently for 5,000+ records.
    
    Returns:
    - deduplicated_df: DataFrame with unique/canonical complaints only.
    - full_with_dup_info_df: Full DataFrame annotated with duplicate_group_id,
      is_duplicate, canonical_complaint_id, duplicate_confidence.
    """
    df = df.copy()
    n = len(df)
    
    df["duplicate_group_id"] = "NONE"
    df["is_duplicate"] = False
    df["canonical_complaint_id"] = df["complaint_id"]
    df["duplicate_confidence"] = 0.0
    
    # 1. Exact Duplicate Matching via Exact Clean Text + Venue ID + Purchase Date
    exact_groups = {}
    dup_group_counter = 1
    
    for idx, row in df.iterrows():
        clean_text = row["clean_text"]
        venue_id = row["venue_id"]
        purch_date = str(row["purchase_date"])
        
        if not clean_text or venue_id == "UNKNOWN":
            continue
            
        exact_key = f"{venue_id}|{purch_date}|{clean_text}"
        if exact_key not in exact_groups:
            exact_groups[exact_key] = []
        exact_groups[exact_key].append(idx)
        
    for key, indices in exact_groups.items():
        if len(indices) > 1:
            group_id = f"DUP-EXACT-{dup_group_counter:04d}"
            dup_group_counter += 1
            canonical_idx = indices[0]
            canonical_id = df.loc[canonical_idx, "complaint_id"]
            
            for idx in indices:
                df.loc[idx, "duplicate_group_id"] = group_id
                df.loc[idx, "canonical_complaint_id"] = canonical_id
                df.loc[idx, "duplicate_confidence"] = 1.0
                if idx != canonical_idx:
                    df.loc[idx, "is_duplicate"] = True

    # 2. Fast Vectorized Near-Duplicate Matching using TF-IDF Matrix Operations
    non_empty_mask = (df["clean_text"].str.len() > 5) & (~df["is_duplicate"])
    non_empty_indices = df[non_empty_mask].index.values
    
    if len(non_empty_indices) > 1:
        texts = df.loc[non_empty_indices, "clean_text"].tolist()
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        # Upper triangle pairs above threshold
        row_pairs, col_pairs = np.where(np.triu(sim_matrix, k=1) >= similarity_threshold)
        
        venues = df.loc[non_empty_indices, "venue_id"].values
        dts = df.loc[non_empty_indices, "parsed_complaint_dt"].values
        
        for r, c in zip(row_pairs, col_pairs):
            idx_i = non_empty_indices[r]
            idx_j = non_empty_indices[c]
            
            if df.loc[idx_j, "is_duplicate"]:
                continue
                
            v_i, v_j = venues[r], venues[c]
            if v_i != "UNKNOWN" and v_j != "UNKNOWN" and v_i != v_j:
                continue
                
            dt_i, dt_j = dts[r], dts[c]
            if dt_i is not None and dt_j is not None:
                diff_hours = abs((dt_i - dt_j) / np.timedelta64(1, 'h'))
                if diff_hours > time_window_hours:
                    continue
            
            # Record duplicate match
            if df.loc[idx_i, "duplicate_group_id"] == "NONE":
                group_id = f"DUP-NEAR-{dup_group_counter:04d}"
                dup_group_counter += 1
                df.loc[idx_i, "duplicate_group_id"] = group_id
            else:
                group_id = df.loc[idx_i, "duplicate_group_id"]
                
            canonical_id = df.loc[idx_i, "canonical_complaint_id"]
            df.loc[idx_j, "duplicate_group_id"] = group_id
            df.loc[idx_j, "canonical_complaint_id"] = canonical_id
            df.loc[idx_j, "duplicate_confidence"] = round(float(sim_matrix[r, c]), 3)
            df.loc[idx_j, "is_duplicate"] = True

    deduplicated_df = df[~df["is_duplicate"]].copy()
    return deduplicated_df, df
