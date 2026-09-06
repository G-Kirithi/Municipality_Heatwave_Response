import re
import pandas as pd
import numpy as np
from datetime import datetime

def clean_text(text):
    """
    Cleans complaint text by lowercasing, removing punctuation, and trimming extra spaces.
    Handles None, NaN, and non-string inputs safely (Edge Case 1).
    """
    if pd.isna(text) or text is None or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_datetime(date_str, time_str=None):
    """
    Combines date and optional time strings into a datetime object safely.
    """
    if pd.isna(date_str) or not date_str:
        return None
    try:
        if time_str and not pd.isna(time_str):
            full_str = f"{str(date_str).strip()} {str(time_str).strip()}"
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(full_str, fmt)
                except ValueError:
                    pass
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                pass
    except Exception:
        pass
    return None

def preprocess_complaints(df):
    """
    Preprocesses the raw complaints dataframe, enforcing validations, flags,
    and missing data imputation.
    """
    df = df.copy()
    
    # Ensure expected columns exist with defaults if missing
    expected_defaults = {
        "anonymised_complaint_text": "",
        "venue_id": "UNKNOWN",
        "venue_name": "UNKNOWN",
        "symptoms": "",
        "food_item": "Unknown",
        "location_zone": "Zone-1",
        "vulnerable_area_flag": 0,
        "inspection_score": np.nan,
        "inspection_status": "UNINSPECTED",
        "previous_violation": "None",
        "complaint_timestamp": "",
        "purchase_date": "",
        "purchase_time": "",
        "symptom_onset_hours": 12.0
    }
    for col, default_val in expected_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # 1. Text cleaning
    df["clean_text"] = df["anonymised_complaint_text"].apply(clean_text)
    df["has_missing_text"] = df["clean_text"].apply(lambda x: 1 if len(x) == 0 else 0)
    
    # 2. Venue cleaning
    df["venue_id"] = df["venue_id"].fillna("UNKNOWN")
    df["venue_name"] = df["venue_name"].fillna("UNKNOWN")
    df["has_missing_venue"] = df["venue_id"].apply(lambda x: 1 if str(x) == "UNKNOWN" else 0)
    
    # 3. Datetime parsing & timestamp sanity check (Edge Case 3)
    parsed_complaint_dt = []
    parsed_purchase_dt = []
    suspicious_timestamp_flags = []
    
    for idx, row in df.iterrows():
        comp_dt = parse_datetime(row.get("complaint_timestamp"))
        purch_dt = parse_datetime(row.get("purchase_date"), row.get("purchase_time"))
        
        is_suspicious = 0
        if comp_dt and purch_dt:
            if purch_dt > comp_dt:
                # Purchase occurred AFTER complaint timestamp -> Invalid/Suspicious
                is_suspicious = 1
        
        parsed_complaint_dt.append(comp_dt)
        parsed_purchase_dt.append(purch_dt)
        suspicious_timestamp_flags.append(is_suspicious)
        
    df["parsed_complaint_dt"] = parsed_complaint_dt
    df["parsed_purchase_dt"] = parsed_purchase_dt
    df["is_suspicious_timestamp"] = suspicious_timestamp_flags
    
    # 4. Symptoms standardization
    def extract_symptom_list(sym_str):
        if pd.isna(sym_str) or not sym_str:
            return []
        return [s.strip().lower() for s in str(sym_str).split(",") if s.strip()]

    df["symptoms_list"] = df["symptoms"].apply(extract_symptom_list)
    
    # 5. Inspection score handling (Edge Case 7)
    df["inspection_score"] = pd.to_numeric(df["inspection_score"], errors="coerce")
    df["has_missing_inspection"] = df["inspection_score"].isna().astype(int)
    
    # 6. Default fallback for missing numerical values
    df["symptom_onset_hours"] = pd.to_numeric(df["symptom_onset_hours"], errors="coerce").fillna(12.0)
    df["vulnerable_area_flag"] = pd.to_numeric(df["vulnerable_area_flag"], errors="coerce").fillna(0).astype(int)
    
    return df
