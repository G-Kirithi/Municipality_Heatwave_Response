# Ethics, Privacy & Governance Guidelines

## Core Ethical Principles

### 1. Purely Synthetic & De-Identified Data
- The application operates exclusively on synthetic complaint data (5,000 synthetic records).
- Zero Personally Identifiable Information (PII) such as names, phone numbers, email addresses, exact home addresses, or social security numbers are generated or stored.

### 2. Human-in-the-Loop Oversight
- Algorithmic outputs (cluster assignments, common-source likelihoods, priority scores) serve **strictly as advisory screening tools**.
- The platform **NEVER** issues automated public health declarations or enforcement actions. A qualified human Food Safety Officer or Epidemiologist remains solely responsible for official determinations.

### 3. Non-Medical Terminology Standard
- The application explicitly uses terms such as **"Potential common-source cluster"** and **"Suspected common venue"**.
- It is strictly forbidden from labeling incidents as "Medically confirmed outbreaks" without laboratory culture confirmation.

### 4. Vulnerable Resident Protection & Heatwave Bias Mitigation
- Heatwave conditions increase vulnerability among senior residents.
- The system includes a `vulnerable_area_flag` to prioritize vulnerable senior care facilities without unfairly stereotyping or penalizing specific community sectors.

### 5. AI Safety & Structured Constraints
- Google Gemini integration relies **exclusively** on structured JSON context provided by the backend.
- Prompts prohibit hallucination of symptoms, venues, or inspection results. When data is missing, the AI output explicitly declares "Insufficient evidence."
