import pytest
import pandas as pd
from src.preprocessing import preprocess_complaints
from src.duplicate_detection import detect_duplicates
from src.feature_engineering import build_pairwise_similarity_matrix
from src.clustering import perform_clustering

def test_clustering_cluster_detection():
    data = [
        {
            "complaint_id": "CMP-001",
            "complaint_timestamp": "2026-07-15 12:00:00",
            "purchase_date": "2026-07-15",
            "purchase_time": "10:00:00",
            "venue_id": "VEN-001",
            "venue_name": "Golden Spice Biryani",
            "anonymised_complaint_text": "Had chicken biryani for lunch at Golden Spice. Severe vomiting and diarrhea started later.",
            "symptoms": "Vomiting, Diarrhea",
            "food_item": "Chicken Biryani",
            "location_zone": "Zone-1 North",
            "inspection_score": 60,
            "vulnerable_area_flag": 1,
            "duplicate_group_id": "NONE"
        },
        {
            "complaint_id": "CMP-002",
            "complaint_timestamp": "2026-07-15 14:30:00",
            "purchase_date": "2026-07-15",
            "purchase_time": "12:15:00",
            "venue_id": "VEN-001",
            "venue_name": "Golden Spice Biryani",
            "anonymised_complaint_text": "Ate biryani rice meal at Golden Spice Biryani House. Violent stomach cramps and nausea.",
            "symptoms": "Vomiting, Nausea",
            "food_item": "Chicken Biryani",
            "location_zone": "Zone-1 North",
            "inspection_score": 60,
            "vulnerable_area_flag": 1,
            "duplicate_group_id": "NONE"
        },
        {
            "complaint_id": "CMP-003",
            "complaint_timestamp": "2026-07-15 18:00:00",
            "purchase_date": "2026-07-15",
            "purchase_time": "13:00:00",
            "venue_id": "VEN-001",
            "venue_name": "Golden Spice Biryani",
            "anonymised_complaint_text": "Ordered takeaway chicken biryani from Golden Spice. Family member experienced severe diarrhea.",
            "symptoms": "Diarrhea, Fever",
            "food_item": "Chicken Biryani",
            "location_zone": "Zone-1 North",
            "inspection_score": 60,
            "vulnerable_area_flag": 1,
            "duplicate_group_id": "NONE"
        }
    ]
    df = preprocess_complaints(pd.DataFrame(data))
    dedup_df, _ = detect_duplicates(df)
    
    tot_sim, dist_mat, _ = build_pairwise_similarity_matrix(dedup_df)
    clust_df, summaries = perform_clustering(dedup_df, dist_mat, distance_threshold=0.65, min_samples=3)
    
    assert len(summaries) == 1
    assert summaries[0]["complaint_count"] == 3
    assert "Golden Spice Biryani" in summaries[0]["venues"]
