import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_synthetic_dataset(num_records=5000, output_path="data/synthetic/synthetic_complaints.csv", seed=None):
    """
    Generates a realistic synthetic dataset of food-borne illness complaints.
    If seed is None, generates a fresh random dataset.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed()
        np.random.seed()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    base_date = datetime(2026, 7, 1, 8, 0, 0)
    
    # Venues master list
    venues = [
        {"id": "VEN-001", "name": "Golden Spice Biryani House", "type": "Restaurant", "zone": "Zone-1 North", "vulnerable": 1, "score": 62, "status": "FAIL", "violation": "Improper Refrigeration Temp"},
        {"id": "VEN-002", "name": "Oceanic Seafood Buffet", "type": "Buffet", "zone": "Zone-4 Harbor", "vulnerable": 0, "score": 58, "status": "FAIL", "violation": "Cross Contamination & Raw Storage"},
        {"id": "VEN-003", "name": "Central Mayo Deli & Cafe", "type": "Deli", "zone": "Zone-3 Central", "vulnerable": 1, "score": 74, "status": "CONDITIONAL_PASS", "violation": "Inadequate Handwashing Station"},
        {"id": "VEN-004", "name": "Metro Fresh Catering", "type": "Caterer", "zone": "Zone-3 Central", "vulnerable": 1, "score": 71, "status": "CONDITIONAL_PASS", "violation": "Temperature Control Violation"},
        {"id": "VEN-005", "name": "Sunshine Senior Care Diner", "type": "Cafeteria", "zone": "Zone-5 East", "vulnerable": 1, "score": 65, "status": "CONDITIONAL_PASS", "violation": "Hot Holding Below 135F"},
        {"id": "VEN-006", "name": "Downtown Taco Express", "type": "Food Truck", "zone": "Zone-2 South", "vulnerable": 0, "score": 88, "status": "PASS", "violation": "None"},
        {"id": "VEN-007", "name": "Green Leaf Salad Bar", "type": "Restaurant", "zone": "Zone-3 Central", "vulnerable": 0, "score": 92, "status": "PASS", "violation": "None"},
        {"id": "VEN-008", "name": "Pizzeria Roma", "type": "Restaurant", "zone": "Zone-1 North", "vulnerable": 0, "score": 85, "status": "PASS", "violation": "None"},
        {"id": "VEN-009", "name": "Burger Haven", "type": "Fast Food", "zone": "Zone-2 South", "vulnerable": 0, "score": 79, "status": "PASS", "violation": "Minor Equipment Cleaning"},
        {"id": "VEN-010", "name": "Grand Palace Hotel Kitchen", "type": "Hotel Kitchen", "zone": "Zone-4 Harbor", "vulnerable": 0, "score": 95, "status": "PASS", "violation": "None"}
    ]
    
    # Additional 40 random venue stubs for noise
    for i in range(11, 51):
        venues.append({
            "id": f"VEN-{i:03d}",
            "name": f"Neighborhood Eatery {i}",
            "type": random.choice(["Restaurant", "Bakery", "Cafe", "Food Stand", "Grocery"]),
            "zone": random.choice(["Zone-1 North", "Zone-2 South", "Zone-3 Central", "Zone-4 Harbor", "Zone-5 East"]),
            "vulnerable": random.choice([0, 0, 1]),
            "score": random.randint(70, 98),
            "status": random.choice(["PASS", "PASS", "CONDITIONAL_PASS"]),
            "violation": random.choice(["None", "Minor Sanitation", "Storage Labels Missing"])
        })

    symptom_pools = {
        "cluster_a": ["Vomiting", "Diarrhea", "Stomach Cramps", "Nausea", "Mild Fever"],
        "cluster_b": ["Severe Diarrhea", "Vomiting", "High Fever", "Chills", "Abdominal Pain"],
        "cluster_c": ["Nausea", "Vomiting", "Headache", "Watery Diarrhea"],
        "cluster_d": ["Stomach Cramps", "Diarrhea", "Nausea"],
        "cluster_e": ["Severe Vomiting", "Dehydration", "Fever", "Diarrhea"],
        "general": ["Nausea", "Stomachache", "Headache", "Vomiting", "Diarrhea", "Fever", "Dizziness", "Fatigue"]
    }

    records = []
    current_id = 10001
    
    # --- 1. GENERATE GROUND TRUTH CLUSTERS ---
    
    # Cluster A: Biryani Outbreak at VEN-001 (Heatwave Peak)
    cluster_a_start = base_date + timedelta(days=5, hours=12)
    biryani_texts = [
        "Had chicken biryani for lunch at Golden Spice. Started vomiting and had severe diarrhea later that evening.",
        "Ate chicken biryani around 1pm at Golden Spice Biryani House. Developed violent vomiting and stomach cramps by 8pm.",
        "Got take-out biryani from Golden Spice. Experienced diarrhea, nausea, and vomiting within 6 hours of eating.",
        "Ate chicken biryani meal at Golden Spice House. Sick all night with nausea and diarrhea.",
        "Ordered chicken biryani from Golden Spice Biryani. My whole family had vomiting and cramps after dinner."
    ]
    for i in range(45):
        comp_time = cluster_a_start + timedelta(hours=random.uniform(0, 36))
        purch_time = comp_time - timedelta(hours=random.uniform(3, 14))
        text = random.choice(biryani_texts)
        dup_grp = f"DUP-A-{i//5}" if i % 5 != 0 else "NONE"
        if dup_grp != "NONE" and i % 5 == 1:
            # Exact near duplicate text
            text = biryani_texts[0]
        
        records.append({
            "complaint_id": f"CMP-{current_id}",
            "complaint_timestamp": comp_time.strftime("%Y-%m-%d %H:%M:%S"),
            "purchase_date": purch_time.strftime("%Y-%m-%d"),
            "purchase_time": purch_time.strftime("%H:%M:%S"),
            "venue_id": "VEN-001",
            "venue_name": "Golden Spice Biryani House",
            "venue_type": "Restaurant",
            "anonymised_complaint_text": text,
            "symptoms": ", ".join(random.sample(symptom_pools["cluster_a"], 3)),
            "symptom_onset_hours": round(random.uniform(4.0, 8.0), 1),
            "food_item": "Chicken Biryani",
            "meal_type": random.choice(["Lunch", "Dinner"]),
            "inspection_date": "2026-06-20",
            "inspection_score": 62,
            "inspection_status": "FAIL",
            "previous_violation": "Improper Refrigeration Temp",
            "location_zone": "Zone-1 North",
            "vulnerable_area_flag": 1,
            "duplicate_group_id": dup_grp,
            "ground_truth_cluster_id": "OUTBREAK-A-BIRYANI"
        })
        current_id += 1

    # Cluster B: Seafood Buffet Bay (VEN-002)
    cluster_b_start = base_date + timedelta(days=12, hours=18)
    seafood_texts = [
        "Ate raw oysters and prawns at Oceanic Seafood Buffet. Experienced severe diarrhea, high fever, and vomiting.",
        "Had seafood dinner at Oceanic Seafood Buffet. Got extremely sick with fever, chills, and stomach cramps.",
        "Dined at Oceanic Buffet evening. Raw oysters tasted warm. Violent food poisoning symptoms followed.",
        "Oysters and crab legs at Oceanic Seafood. Severe abdominal pain, diarrhea, and high fever started overnight."
    ]
    for i in range(35):
        comp_time = cluster_b_start + timedelta(hours=random.uniform(0, 30))
        purch_time = comp_time - timedelta(hours=random.uniform(6, 20))
        records.append({
            "complaint_id": f"CMP-{current_id}",
            "complaint_timestamp": comp_time.strftime("%Y-%m-%d %H:%M:%S"),
            "purchase_date": purch_time.strftime("%Y-%m-%d"),
            "purchase_time": purch_time.strftime("%H:%M:%S"),
            "venue_id": "VEN-002",
            "venue_name": "Oceanic Seafood Buffet",
            "venue_type": "Buffet",
            "anonymised_complaint_text": random.choice(seafood_texts),
            "symptoms": ", ".join(random.sample(symptom_pools["cluster_b"], 4)),
            "symptom_onset_hours": round(random.uniform(8.0, 16.0), 1),
            "food_item": random.choice(["Raw Oysters", "Seafood Platter", "Garlic Shrimp"]),
            "meal_type": "Dinner",
            "inspection_date": "2026-06-15",
            "inspection_score": 58,
            "inspection_status": "FAIL",
            "previous_violation": "Cross Contamination & Raw Storage",
            "location_zone": "Zone-4 Harbor",
            "vulnerable_area_flag": 0,
            "duplicate_group_id": "NONE",
            "ground_truth_cluster_id": "OUTBREAK-B-SEAFOOD"
        })
        current_id += 1

    # Cluster C: Shared Supplier Mayo Salad across VEN-003 and VEN-004
    cluster_c_start = base_date + timedelta(days=20, hours=10)
    mayo_texts = [
        "Bought chicken mayo sandwich from deli. Had nausea, vomiting, and watery diarrhea within hours.",
        "Ate egg salad mayo wrap at catering event. Woke up with terrible cramps and vomiting.",
        "Had chicken mayonnaise roll for lunch. Severe nausea and diarrhea set in by afternoon."
    ]
    for i in range(50):
        comp_time = cluster_c_start + timedelta(hours=random.uniform(0, 48))
        purch_time = comp_time - timedelta(hours=random.uniform(2, 8))
        venue = random.choice([venues[2], venues[3]]) # VEN-003 or VEN-004
        records.append({
            "complaint_id": f"CMP-{current_id}",
            "complaint_timestamp": comp_time.strftime("%Y-%m-%d %H:%M:%S"),
            "purchase_date": purch_time.strftime("%Y-%m-%d"),
            "purchase_time": purch_time.strftime("%H:%M:%S"),
            "venue_id": venue["id"],
            "venue_name": venue["name"],
            "venue_type": venue["type"],
            "anonymised_complaint_text": random.choice(mayo_texts),
            "symptoms": ", ".join(random.sample(symptom_pools["cluster_c"], 3)),
            "symptom_onset_hours": round(random.uniform(3.0, 6.0), 1),
            "food_item": random.choice(["Chicken Mayo Sandwich", "Egg Salad Wrap", "Tuna Mayo Sub"]),
            "meal_type": "Lunch",
            "inspection_date": "2026-07-02",
            "inspection_score": venue["score"],
            "inspection_status": venue["status"],
            "previous_violation": venue["violation"],
            "location_zone": "Zone-3 Central",
            "vulnerable_area_flag": 1,
            "duplicate_group_id": "NONE",
            "ground_truth_cluster_id": "OUTBREAK-C-MAYO-SUPPLIER"
        })
        current_id += 1

    # Cluster D: Sunshine Senior Care Diner (Vulnerable Flag Heatwave cluster)
    cluster_d_start = base_date + timedelta(days=25, hours=11)
    senior_texts = [
        "Senior resident ate cream soup at facility cafeteria. High fever, severe vomiting, and dehydration.",
        "My elderly parent had dinner at senior diner. Developed severe vomiting and diarrhea overnight during heatwave.",
        "Custard dessert at senior care cafeteria caused severe gastroenteritis symptoms."
    ]
    for i in range(30):
        comp_time = cluster_d_start + timedelta(hours=random.uniform(0, 24))
        purch_time = comp_time - timedelta(hours=random.uniform(4, 12))
        records.append({
            "complaint_id": f"CMP-{current_id}",
            "complaint_timestamp": comp_time.strftime("%Y-%m-%d %H:%M:%S"),
            "purchase_date": purch_time.strftime("%Y-%m-%d"),
            "purchase_time": purch_time.strftime("%H:%M:%S"),
            "venue_id": "VEN-005",
            "venue_name": "Sunshine Senior Care Diner",
            "venue_type": "Cafeteria",
            "anonymised_complaint_text": random.choice(senior_texts),
            "symptoms": ", ".join(random.sample(symptom_pools["cluster_e"], 4)),
            "symptom_onset_hours": round(random.uniform(4.0, 10.0), 1),
            "food_item": random.choice(["Cream Mushroom Soup", "Egg Custard", "Poultry Casserole"]),
            "meal_type": "Dinner",
            "inspection_date": "2026-06-28",
            "inspection_score": 65,
            "inspection_status": "CONDITIONAL_PASS",
            "previous_violation": "Hot Holding Below 135F",
            "location_zone": "Zone-5 East",
            "vulnerable_area_flag": 1,
            "duplicate_group_id": "NONE",
            "ground_truth_cluster_id": "OUTBREAK-D-SENIOR-CARE"
        })
        current_id += 1

    # --- 2. GENERATE MASSIVE DUPLICATE BURST (EDGE CASE 4) ---
    burst_time = base_date + timedelta(days=15, hours=9)
    burst_text = "Spaghetti carbonara from Pizzeria Roma tasted spoiled. Continuous vomiting and nausea."
    for i in range(100):
        comp_time = burst_time + timedelta(minutes=i*2)
        purch_time = burst_time - timedelta(hours=3)
        records.append({
            "complaint_id": f"CMP-{current_id}",
            "complaint_timestamp": comp_time.strftime("%Y-%m-%d %H:%M:%S"),
            "purchase_date": purch_time.strftime("%Y-%m-%d"),
            "purchase_time": purch_time.strftime("%H:%M:%S"),
            "venue_id": "VEN-008",
            "venue_name": "Pizzeria Roma",
            "venue_type": "Restaurant",
            "anonymised_complaint_text": burst_text,
            "symptoms": "Vomiting, Nausea",
            "symptom_onset_hours": 3.0,
            "food_item": "Spaghetti Carbonara",
            "meal_type": "Dinner",
            "inspection_date": "2026-05-10",
            "inspection_score": 85,
            "inspection_status": "PASS",
            "previous_violation": "None",
            "location_zone": "Zone-1 North",
            "vulnerable_area_flag": 0,
            "duplicate_group_id": "DUP-BURST-001",
            "ground_truth_cluster_id": "BURST-DUPLICATE-GROUP"
        })
        current_id += 1

    # --- 3. GENERATE UNRELATED NOISE COMPLAINTS (REST OF 5,000) ---
    remaining = num_records - len(records)
    general_food_items = [
        "Cheese Burger", "Veggie Pizza", "Grilled Chicken Salad", "Fish Tacos", "Pasta Alfredo",
        "Steak Bowl", "Sushi Roll", "Falafel Wrap", "Ice Cream", "Fried Rice", "Club Sandwich"
    ]
    
    for i in range(remaining):
        venue = random.choice(venues)
        comp_time = base_date + timedelta(days=random.uniform(0, 60), hours=random.uniform(0, 24))
        purch_time = comp_time - timedelta(hours=random.uniform(2, 48))
        
        # Inject specific edge cases cleanly into noise
        text = f"Felt unwell after having {random.choice(general_food_items)} at {venue['name']}. Minor stomach discomfort."
        venue_id = venue["id"]
        venue_name = venue["name"]
        inspection_score = venue["score"]
        inspection_status = venue["status"]
        inspection_date = "2026-06-01"
        
        # Edge Case 1: Missing complaint text (~20 items)
        if i < 20:
            text = ""
        # Edge Case 2: Missing venue (~15 items)
        elif i >= 20 and i < 35:
            venue_id = "UNKNOWN"
            venue_name = "UNKNOWN"
        # Edge Case 3: Conflicting timestamp (purchase after complaint) (~10 items)
        elif i >= 35 and i < 45:
            purch_time = comp_time + timedelta(hours=5)
        # Edge Case 7: Missing inspection data (~50 items)
        elif i >= 45 and i < 95:
            inspection_score = None
            inspection_status = "UNINSPECTED"
            inspection_date = None

        records.append({
            "complaint_id": f"CMP-{current_id}",
            "complaint_timestamp": comp_time.strftime("%Y-%m-%d %H:%M:%S"),
            "purchase_date": purch_time.strftime("%Y-%m-%d"),
            "purchase_time": purch_time.strftime("%H:%M:%S"),
            "venue_id": venue_id,
            "venue_name": venue_name,
            "venue_type": venue["type"],
            "anonymised_complaint_text": text,
            "symptoms": ", ".join(random.sample(symptom_pools["general"], random.randint(1, 3))),
            "symptom_onset_hours": round(random.uniform(2.0, 24.0), 1),
            "food_item": random.choice(general_food_items),
            "meal_type": random.choice(["Breakfast", "Lunch", "Dinner", "Snack"]),
            "inspection_date": inspection_date if inspection_date else "",
            "inspection_score": inspection_score if inspection_score is not None else np.nan,
            "inspection_status": inspection_status,
            "previous_violation": venue["violation"],
            "location_zone": venue["zone"],
            "vulnerable_area_flag": venue["vulnerable"],
            "duplicate_group_id": "NONE",
            "ground_truth_cluster_id": "NOISE"
        })
        current_id += 1

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} records at: {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_dataset()
