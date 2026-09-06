import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_WEIGHTS = {
    "text": 0.30,
    "venue": 0.20,
    "time": 0.20,
    "symptoms": 0.15,
    "food": 0.10,
    "inspection": 0.05
}

def build_pairwise_similarity_matrix(df, weights=None):
    """
    Computes vectorized multi-signal pairwise similarity and distance matrices for a dataset of complaints.
    Fast & scalable for 5,000+ records.
    
    Returns:
    - total_sim_matrix: (N, N) similarity matrix with values in [0.0, 1.0]
    - distance_matrix: (N, N) distance matrix = 1.0 - total_sim_matrix
    - component_matrices: Dictionary of component similarity matrices
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
        
    n = len(df)
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), {}

    # 1. Text Similarity (TF-IDF + Cosine Similarity)
    texts = df["clean_text"].tolist()
    if any(len(t) > 0 for t in texts):
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_mat = vectorizer.fit_transform(texts)
        text_sim = cosine_similarity(tfidf_mat)
    else:
        text_sim = np.zeros((n, n))

    # 2. Vectorized Venue & Zone Similarity
    venue_arr = np.array(df["venue_id"].fillna("UNKNOWN").astype(str).tolist())
    zone_arr = np.array(df["location_zone"].fillna("UNKNOWN").astype(str).tolist())
    
    venue_match = (venue_arr[:, None] == venue_arr[None, :]) & (venue_arr[:, None] != "UNKNOWN")
    zone_match = (zone_arr[:, None] == zone_arr[None, :]) & (zone_arr[:, None] != "UNKNOWN") & (~venue_match)
    
    venue_sim = np.where(venue_match, 1.0, np.where(zone_match, 0.5, 0.0))
    np.fill_diagonal(venue_sim, 1.0)

    # 3. Vectorized Purchase / Complaint Time Similarity (Exponential Decay)
    dts = df["parsed_complaint_dt"].tolist()
    # Convert timestamps to float hours from epoch
    timestamps_sec = np.array([dt.timestamp() if dt is not None else np.nan for dt in dts])
    
    valid_mask = ~np.isnan(timestamps_sec)
    time_diff_sec = np.abs(timestamps_sec[:, None] - timestamps_sec[None, :])
    time_diff_hours = time_diff_sec / 3600.0
    
    time_sim = np.exp(-time_diff_hours / 24.0)
    # Zero out pairs where timestamp is missing
    missing_time_mask = np.isnan(time_diff_hours)
    time_sim[missing_time_mask] = 0.0
    np.fill_diagonal(time_sim, 1.0)

    # 4. Vectorized Symptom Similarity (CountVectorizer + Cosine/Jaccard)
    sym_texts = df["symptoms"].fillna("").astype(str).tolist()
    if any(len(s) > 0 for s in sym_texts):
        count_vec = CountVectorizer(tokenizer=lambda x: [item.strip().lower() for item in x.split(",") if item.strip()], token_pattern=None)
        sym_mat = count_vec.fit_transform(sym_texts)
        # Cosine similarity between symptom vectors
        symptom_sim = cosine_similarity(sym_mat)
    else:
        symptom_sim = np.zeros((n, n))
    np.fill_diagonal(symptom_sim, 1.0)

    # 5. Vectorized Food Similarity
    food_texts = df["food_item"].fillna("").astype(str).tolist()
    if any(len(f) > 0 for f in food_texts):
        food_vec = CountVectorizer(stop_words="english")
        try:
            food_mat = food_vec.fit_transform(food_texts)
            food_sim = cosine_similarity(food_mat)
        except ValueError:
            food_sim = np.zeros((n, n))
    else:
        food_sim = np.zeros((n, n))
    np.fill_diagonal(food_sim, 1.0)

    # 6. Vectorized Inspection Risk Signal Matrix
    insp_scores = df["inspection_score"].fillna(100.0).values
    avg_risk = ((100.0 - insp_scores[:, None]) + (100.0 - insp_scores[None, :])) / 200.0
    inspection_sim = np.clip(avg_risk, 0.0, 1.0)
    np.fill_diagonal(inspection_sim, 1.0)

    # Weighted Combination Matrix
    total_sim = (
        weights["text"] * text_sim +
        weights["venue"] * venue_sim +
        weights["time"] * time_sim +
        weights["symptoms"] * symptom_sim +
        weights["food"] * food_sim +
        weights["inspection"] * inspection_sim
    )
    
    # Clip to [0, 1] range
    total_sim = np.clip(total_sim, 0.0, 1.0)
    distance_matrix = np.clip(1.0 - total_sim, 0.0, 1.0)

    component_matrices = {
        "text": text_sim,
        "venue": venue_sim,
        "time": time_sim,
        "symptoms": symptom_sim,
        "food": food_sim,
        "inspection": inspection_sim
    }

    return total_sim, distance_matrix, component_matrices
