import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_complaints
from src.duplicate_detection import detect_duplicates
from src.feature_engineering import build_pairwise_similarity_matrix
from src.clustering import perform_clustering
from src.gemini_service import generate_investigation_summary

def test_edge_case_missing_text():
    # Edge Case 1: Missing complaint text
    data = [{
        "complaint_id": "CMP-901",
        "complaint_timestamp": "2026-07-15 12:00:00",
        "purchase_date": "2026-07-15",
        "purchase_time": "10:00:00",
        "venue_id": "VEN-001",
        "venue_name": "Test Eatery",
        "anonymised_complaint_text": None, # Missing text
        "symptoms": "Nausea, Vomiting",
        "food_item": "Pizza"
    }]
    df = preprocess_complaints(pd.DataFrame(data))
    assert df.loc[0, "has_missing_text"] == 1
    assert df.loc[0, "clean_text"] == ""

def test_edge_case_missing_venue():
    # Edge Case 2: Unknown venue
    data = [{
        "complaint_id": "CMP-902",
        "complaint_timestamp": "2026-07-15 12:00:00",
        "purchase_date": "2026-07-15",
        "venue_id": np.nan, # Unknown venue
        "anonymised_complaint_text": "Felt sick after dining",
        "symptoms": "Diarrhea"
    }]
    df = preprocess_complaints(pd.DataFrame(data))
    assert df.loc[0, "venue_id"] == "UNKNOWN"
    assert df.loc[0, "has_missing_venue"] == 1

def test_edge_case_conflicting_timestamps():
    # Edge Case 3: Purchase timestamp occurs after complaint timestamp
    data = [{
        "complaint_id": "CMP-903",
        "complaint_timestamp": "2026-07-15 10:00:00",
        "purchase_date": "2026-07-15",
        "purchase_time": "14:00:00", # 14:00 > 10:00
        "venue_id": "VEN-001",
        "anonymised_complaint_text": "Test food poisoning text."
    }]
    df = preprocess_complaints(pd.DataFrame(data))
    assert df.loc[0, "is_suspicious_timestamp"] == 1

def test_edge_case_large_duplicate_burst():
    # Edge Case 4: Massive duplicate burst (50 identical complaints)
    data = []
    for i in range(50):
        data.append({
            "complaint_id": f"CMP-BURST-{i:03d}",
            "complaint_timestamp": "2026-07-15 12:00:00",
            "purchase_date": "2026-07-15",
            "purchase_time": "10:00:00",
            "venue_id": "VEN-005",
            "venue_name": "Burst Diner",
            "anonymised_complaint_text": "Identical complaint burst text for burst testing.",
            "symptoms": "Vomiting"
        })
    df = preprocess_complaints(pd.DataFrame(data))
    dedup_df, full_df = detect_duplicates(df)
    assert len(dedup_df) == 1
    assert full_df["is_duplicate"].sum() == 49

def test_edge_case_no_clusters():
    # Edge Case 5: All complaints completely unrelated across different dates & venues
    data = [
        {"complaint_id": "CMP-1", "complaint_timestamp": "2026-07-01 10:00:00", "venue_id": "VEN-1", "clean_text": "text a", "symptoms": "Fever", "food_item": "Apple"},
        {"complaint_id": "CMP-2", "complaint_timestamp": "2026-07-20 18:00:00", "venue_id": "VEN-2", "clean_text": "text b", "symptoms": "Headache", "food_item": "Bread"}
    ]
    df = preprocess_complaints(pd.DataFrame(data))
    tot_sim, dist_mat, _ = build_pairwise_similarity_matrix(df)
    clust_df, summaries = perform_clustering(df, dist_mat, min_samples=3)
    assert len(summaries) == 0
    assert (clust_df["cluster_id"] == "NOISE").all()

def test_edge_case_gemini_fallback():
    # Edge Case 8 & 9: Gemini API failure / missing key fallback
    payload = {
        "cluster_id": "CLUSTER-99",
        "venues": "Fallback Diner",
        "complaint_count": 5,
        "unique_complaint_count": 5,
        "common_foods": "Burger",
        "dominant_symptoms": "Vomiting",
        "priority_label": "HIGH"
    }
    # Call with invalid key or empty environment
    res = generate_investigation_summary(payload, {"common_source_score": 0.8, "evidence_points": []}, {"priority_score": 75, "priority_label": "HIGH"})
    assert res["status"] in ["SUCCESS", "FALLBACK"]
    assert "What Happened" in res["content"]
