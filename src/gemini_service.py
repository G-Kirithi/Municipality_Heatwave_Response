import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_investigation_summary(cluster_summary, common_source_res, risk_res, cluster_df=None):
    """
    Generates a structured investigation synthesis using Google Gemini API (google-genai SDK).
    If Gemini API key is missing, invalid, or API fails, uses a seamless rule-based fallback.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    # 1. Build Structured Context Payload
    evidence_payload = {
        "cluster_id": cluster_summary.get("cluster_id"),
        "complaint_count": cluster_summary.get("complaint_count"),
        "unique_complaint_count": cluster_summary.get("unique_complaint_count"),
        "venues": cluster_summary.get("venues"),
        "location_zone": cluster_summary.get("location_zone"),
        "time_range": cluster_summary.get("time_range"),
        "common_foods": cluster_summary.get("common_foods"),
        "dominant_symptoms": cluster_summary.get("dominant_symptoms"),
        "inspection_summary": cluster_summary.get("inspection_summary"),
        "vulnerable_area_count": cluster_summary.get("vulnerable_area_count", 0),
        "common_source_score": common_source_res.get("common_source_score"),
        "common_source_label": common_source_res.get("common_source_label"),
        "evidence_points": common_source_res.get("evidence_points"),
        "priority_score": risk_res.get("priority_score"),
        "priority_label": risk_res.get("priority_label"),
        "risk_factors": risk_res.get("factor_breakdown")
    }

    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            prompt = f"""
You are an expert public-health epidemiologist assisting a municipal food safety investigator responding during a heatwave.
You have been provided with structured algorithmic clustering evidence for a potential food-borne illness incident.

IMPORTANT CONSTRAINTS:
1. Base your response ONLY on the provided JSON data.
2. DO NOT invent symptoms, venues, inspection records, dates, or medical conclusions not in the data.
3. If information for any section is missing or ambiguous, explicitly state "Insufficient evidence."
4. Use advisory, public-health screening terms (e.g. "Potential common-source cluster") and NOT "Medically confirmed outbreak".

STRUCTURED EVIDENCE DATA:
{json.dumps(evidence_payload, indent=2)}

Please provide a clear investigation report formatted in Markdown with the following headers:
### 1. What Happened
### 2. Why Complaints Were Grouped
### 3. Evidence Supporting Common Source
### 4. Weaknesses & Uncertainties
### 5. Investigator Verification Steps
### 6. Recommended Immediate Action
### 7. Important Caveats & Disclaimers
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if response and response.text:
                return {
                    "source": "Google Gemini API (gemini-2.5-flash)",
                    "content": response.text,
                    "status": "SUCCESS"
                }
        except Exception as e:
            # Fallback on API failure
            pass

    # 2. Rule-Based Fallback Summary
    return generate_fallback_summary(evidence_payload)

def generate_fallback_summary(payload):
    """
    Structured rule-based fallback summary when Gemini API is unavailable or unconfigured.
    """
    cid = payload.get("cluster_id", "CLUSTER")
    venues = payload.get("venues", "Unknown Venue")
    count = payload.get("complaint_count", 0)
    uniq = payload.get("unique_complaint_count", 0)
    foods = payload.get("common_foods", "Various foods")
    syms = payload.get("dominant_symptoms", "Gastrointestinal symptoms")
    priority = payload.get("priority_label", "MEDIUM")
    common_label = payload.get("common_source_label", "Potential Common Source")
    zone = payload.get("location_zone", "Municipal Zone")
    ev_points = payload.get("evidence_points", [])
    vuln_cnt = payload.get("vulnerable_area_count") or 0

    ev_bullets = "\n".join([f"- {ep}" for ep in ev_points]) if ev_points else "- Multiple complaints share temporal and geographic proximity."
    vuln_str = f"- **Heatwave Vulnerable Risk:** {vuln_cnt} complaints originate from marked senior care / heatwave high-risk zones." if vuln_cnt > 0 else ""

    content = f"""
> [!NOTE]
> *Report Generated via Automated Municipal Rule-Based Analytics Service (Gemini API Offline / Fallback Mode).*

### 1. What Happened
A total of **{count} food-borne illness complaints** ({uniq} unique incidents) were detected forming a **{priority} priority cluster** in **{zone}** associated with **{venues}**. Primary consumed items reported include **{foods}**, with residents experiencing **{syms}**.

### 2. Why Complaints Were Grouped
These complaints were algorithmically grouped based on multi-signal similarity analysis:
- High similarity in purchase and complaint reporting timestamps ({payload.get('time_range', 'N/A')}).
- Overlapping symptom profiles dominated by {syms}.
- High spatial concentration linking to venue(s): {venues}.

### 3. Evidence Supporting Common Source
**Status:** {common_label} (Score: {payload.get('common_source_score', 0.5)})
{ev_bullets}
{vuln_str}

### 4. Weaknesses & Uncertainties
- Complaint text TF-IDF similarity varies across self-reported descriptions.
- Laboratory diagnostic confirmation (stool/food culture sample results) is currently **unavailable**.
- Missing or incomplete inspection logs for minor food suppliers require field verification.

### 5. Investigator Verification Steps
1. Conduct an immediate onsite inspection at **{venues}**.
2. Audit refrigerator and hot-holding temperature logs (verify compliance with < 41°F cold / > 135°F hot holding).
3. Interview food handlers regarding recent illness or handwashing protocols.
4. Collect food samples for batch testing of high-risk items: **{foods}**.

### 6. Recommended Immediate Action
- **Priority Level:** **{priority}**
- **Action:** Assign a Food Safety Inspector within **{12 if priority=='CRITICAL' else 24} hours**. Issue a temporary advisory or corrective order if temperature violations are detected.

### 7. Important Caveats & Disclaimers
*This investigation summary is an automated screening tool designed for municipal prioritization. It does NOT constitute formal epidemiological confirmation or medical diagnosis of a food-borne outbreak.*
"""
    return {
        "source": "Rule-Based Municipal Analytics (Gemini API Fallback Mode)",
        "content": content,
        "status": "FALLBACK"
    }
