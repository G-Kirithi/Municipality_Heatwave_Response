import pytest
import pandas as pd
from src.preprocessing import preprocess_complaints
from src.duplicate_detection import detect_duplicates

def test_exact_duplicate_detection():
    data = [
        {
            "complaint_id": "CMP-001",
            "complaint_timestamp": "2026-07-15 12:00:00",
            "purchase_date": "2026-07-15",
            "purchase_time": "10:00:00",
            "venue_id": "VEN-100",
            "venue_name": "Test Diner",
            "anonymised_complaint_text": "Ate burger and got sick with severe vomiting.",
            "symptoms": "Vomiting"
        },
        {
            "complaint_id": "CMP-002",
            "complaint_timestamp": "2026-07-15 12:30:00",
            "purchase_date": "2026-07-15",
            "purchase_time": "10:00:00",
            "venue_id": "VEN-100",
            "venue_name": "Test Diner",
            "anonymised_complaint_text": "Ate burger and got sick with severe vomiting.",
            "symptoms": "Vomiting"
        }
    ]
    df = preprocess_complaints(pd.DataFrame(data))
    dedup_df, full_df = detect_duplicates(df, similarity_threshold=0.85)
    
    assert len(dedup_df) == 1
    assert full_df.loc[1, "is_duplicate"] == True
    assert full_df.loc[1, "duplicate_confidence"] == 1.0

def test_near_duplicate_detection():
    data = [
        {
            "complaint_id": "CMP-001",
            "complaint_timestamp": "2026-07-15 12:00:00",
            "purchase_date": "2026-07-15",
            "purchase_time": "10:00:00",
            "venue_id": "VEN-100",
            "venue_name": "Test Diner",
            "anonymised_complaint_text": "chicken biryani lunch at test diner. severe vomiting and diarrhea overnight.",
            "symptoms": "Vomiting, Diarrhea"
        },
        {
            "complaint_id": "CMP-002",
            "complaint_timestamp": "2026-07-15 13:00:00",
            "purchase_date": "2026-07-15",
            "purchase_time": "10:30:00",
            "venue_id": "VEN-100",
            "venue_name": "Test Diner",
            "anonymised_complaint_text": "chicken biryani lunch at test diner. severe vomiting and diarrhea during night.",
            "symptoms": "Vomiting, Diarrhea"
        }
    ]
    df = preprocess_complaints(pd.DataFrame(data))
    dedup_df, full_df = detect_duplicates(df, similarity_threshold=0.70)
    
    assert len(dedup_df) == 1
    assert full_df.loc[1, "is_duplicate"] == True
    assert full_df.loc[1, "duplicate_confidence"] >= 0.70
