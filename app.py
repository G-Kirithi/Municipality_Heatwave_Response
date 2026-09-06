import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.investigation import InvestigationPipeline
from src.followup import get_all_investigations, update_investigation_action, update_automated_statuses
from src.feature_engineering import DEFAULT_WEIGHTS

# Page Configuration
st.set_page_config(
    page_title="Municipal Food-Borne Illness Investigation Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Municipal Dashboard Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Header Banner */
    .header-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 24px;
        font-weight: 700;
        color: #38BDF8;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
    }
    
    /* KPI Card Styles */
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: left;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #38BDF8;
        margin-top: 2px;
    }
    
    /* Priority Badges */
    .badge-critical { background-color: #EF4444; color: #FFFFFF; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .badge-high { background-color: #F97316; color: #FFFFFF; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .badge-medium { background-color: #F59E0B; color: #FFFFFF; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .badge-low { background-color: #10B981; color: #FFFFFF; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .badge-escalated { background-color: #DC2626; color: #FFFFFF; padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 12px; border: 1px solid #FCA5A5; }

    /* Evidence Box */
    .evidence-box {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State & Pipeline
@st.cache_resource
def get_pipeline():
    pipeline = InvestigationPipeline()
    pipeline.load_and_process_data()
    pipeline.run_clustering_pipeline()
    return pipeline

pipeline = get_pipeline()

# Header Banner
st.markdown("""
<div class="header-box">
    <div class="header-title">🛡️ Municipal Heatwave Response: Food-Borne Illness Investigation Platform</div>
    <div class="header-subtitle">Real-time complaint deduplication, multi-signal outbreak clustering, evidence synthesis, and follow-up escalation manager.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/city-hall.png", width=64)
st.sidebar.title("System Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("Dataset Configuration")
if st.sidebar.button("🔄 Regenerate 5,000 Synthetic Dataset"):
    with st.spinner("Regenerating dataset and re-running clustering..."):
        pipeline.load_and_process_data(force_generate=True)
        pipeline.run_clustering_pipeline()
        st.sidebar.success("Dataset regenerated!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Clustering Settings")
cluster_method = "agglomerative"
dist_threshold = st.sidebar.slider("Distance Threshold", 0.20, 0.80, 0.45, 0.05)

st.sidebar.subheader("Multi-Signal Similarity Weights")
w_text = st.sidebar.slider("Text Similarity (TF-IDF)", 0.0, 0.5, 0.30, 0.05)
w_venue = st.sidebar.slider("Venue Match", 0.0, 0.5, 0.20, 0.05)
w_time = st.sidebar.slider("Temporal Proximity", 0.0, 0.5, 0.20, 0.05)
w_symptoms = st.sidebar.slider("Symptom Overlap", 0.0, 0.4, 0.15, 0.05)
w_food = st.sidebar.slider("Food Item Match", 0.0, 0.3, 0.10, 0.05)
w_inspection = st.sidebar.slider("Inspection Risk Signal", 0.0, 0.2, 0.05, 0.05)

custom_weights = {
    "text": w_text, "venue": w_venue, "time": w_time,
    "symptoms": w_symptoms, "food": w_food, "inspection": w_inspection
}

if st.sidebar.button("⚡ Apply Clustering Weights"):
    pipeline.run_clustering_pipeline(weights=custom_weights, method=cluster_method, distance_threshold=dist_threshold)
    st.sidebar.success("Clustering updated!")

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Dashboard",
    "🔍 Complaint Explorer",
    "☣️ Cluster Investigation & Evidence",
    "🤖 AI Investigation Assistant",
    "📋 Action Tracking & Escalations"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# -----------------------------------------------------------------------------
with tab1:
    total_raw = len(pipeline.processed_df)
    total_dedup = len(pipeline.dedup_df)
    dups_removed = total_raw - total_dedup
    cluster_count = len(pipeline.cluster_summaries)
    
    # Calculate stats
    high_critical_count = sum(1 for c in pipeline.cluster_summaries if c.get("priority_label") in ["HIGH", "CRITICAL"])
    common_source_count = sum(1 for c in pipeline.cluster_summaries if c.get("common_source_score", 0) >= 0.6)
    
    investigations = get_all_investigations()
    overdue_count = sum(1 for inv in investigations if inv.status in ["OVERDUE", "ESCALATED"])
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Complaints</div><div class="kpi-value">{total_raw:,}</div><div class="kpi-subtext">Raw Received</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Unique Incidents</div><div class="kpi-value">{total_dedup:,}</div><div class="kpi-subtext">-{dups_removed} duplicates</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Detected Clusters</div><div class="kpi-value">{cluster_count}</div><div class="kpi-subtext">{common_source_count} common-source</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">High Priority</div><div class="kpi-value" style="color:#F97316;">{high_critical_count}</div><div class="kpi-subtext">Requires field action</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Overdue / Escalated</div><div class="kpi-value" style="color:#EF4444;">{overdue_count}</div><div class="kpi-subtext">Supervisor alerted</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Outbreak Alert Banner if high/critical exist
    if high_critical_count > 0:
        st.warning(f"⚠️ **OUTBREAK ALERT:** {high_critical_count} High/Critical Priority Potential Common-Source Outbreak Clusters detected during heatwave period requiring immediate municipal investigation.")

    # Dashboard Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📅 Complaint Velocity & Cluster Spikes")
        time_df = pipeline.dedup_df.copy()
        time_df["date"] = pd.to_datetime(time_df["parsed_complaint_dt"]).dt.date
        daily_counts = time_df.groupby(["date", "cluster_id"]).size().reset_index(name="count")
        daily_counts["Cluster Type"] = daily_counts["cluster_id"].apply(lambda x: "Outbreak Cluster" if x != "NOISE" else "Noise / Isolated")
        
        fig_time = px.histogram(
            daily_counts, x="date", y="count", color="Cluster Type",
            color_discrete_map={"Outbreak Cluster": "#F97316", "Noise / Isolated": "#475569"},
            title="Daily Complaint Volume by Cluster Type",
            template="plotly_dark"
        )
        fig_time.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
        st.plotly_chart(fig_time, use_container_width=True)

    with col_chart2:
        st.subheader("📍 Cluster Distribution by Municipal Zone")
        zone_df = pipeline.dedup_df[pipeline.dedup_df["cluster_id"] != "NOISE"]
        if not zone_df.empty:
            zone_counts = zone_df.groupby(["location_zone", "vulnerable_area_flag"]).size().reset_index(name="count")
            zone_counts["Area Risk"] = zone_counts["vulnerable_area_flag"].apply(lambda x: "Senior/Heatwave Vulnerable" if x==1 else "General Zone")
            
            fig_zone = px.bar(
                zone_counts, x="location_zone", y="count", color="Area Risk",
                color_discrete_map={"Senior/Heatwave Vulnerable": "#EF4444", "General Zone": "#38BDF8"},
                title="Clustered Complaints Across Municipal Zones",
                template="plotly_dark"
            )
            fig_zone.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
            st.plotly_chart(fig_zone, use_container_width=True)
        else:
            st.info("No active clusters detected.")

    # High Priority Summary Table
    st.subheader("🔥 Priority Outbreak Clusters Summary")
    if pipeline.cluster_summaries:
        summary_table_df = pd.DataFrame(pipeline.cluster_summaries)
        display_cols = ["cluster_id", "priority_label", "venues", "complaint_count", "unique_complaint_count", "common_foods", "dominant_symptoms", "common_source_score"]
        st.dataframe(summary_table_df[display_cols], use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: COMPLAINT EXPLORER
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🔍 Complaint Database Explorer")
    
    # Filter Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        selected_zone = st.selectbox("Municipal Zone", ["All"] + list(pipeline.processed_df["location_zone"].unique()))
    with f_col2:
        selected_venue = st.selectbox("Venue Filter", ["All"] + list(pipeline.processed_df["venue_name"].unique()))
    with f_col3:
        selected_priority = st.selectbox("Cluster Priority", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"])
    with f_col4:
        search_query = st.text_input("Search Text Keywords", "")

    filtered_df = pipeline.processed_df.copy()
    if selected_zone != "All":
        filtered_df = filtered_df[filtered_df["location_zone"] == selected_zone]
    if selected_venue != "All":
        filtered_df = filtered_df[filtered_df["venue_name"] == selected_venue]
    if search_query:
        filtered_df = filtered_df[filtered_df["anonymised_complaint_text"].str.contains(search_query, case=False, na=False)]

    st.markdown(f"Showing **{len(filtered_df):,}** matching complaints:")
    
    explore_cols = ["complaint_id", "complaint_timestamp", "venue_name", "anonymised_complaint_text", "symptoms", "food_item", "inspection_score", "inspection_status", "duplicate_group_id", "is_duplicate"]
    st.dataframe(filtered_df[explore_cols].head(500), use_container_width=True)
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Filtered Data (CSV)", data=csv_data, file_name="filtered_complaints.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# TAB 3: CLUSTER INVESTIGATION & EVIDENCE MATRIX
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("☣️ Outbreak Cluster Deep Dive & Evidence Matrix")
    
    if not pipeline.cluster_summaries:
        st.info("No active clusters detected under current parameters.")
    else:
        cluster_ids = [c["cluster_id"] for c in pipeline.cluster_summaries]
        selected_cid = st.selectbox("Select Cluster to Investigate", cluster_ids, index=0)
        
        c_details = pipeline.cluster_details.get(selected_cid)
        if c_details:
            summary = c_details["summary"]
            cs_res = c_details["common_source"]
            risk_res = c_details["risk"]
            cluster_df = c_details["dataframe"]
            
            # Overview Panel
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                st.markdown(f"**Cluster ID:** `{selected_cid}`")
                st.markdown(f"**Priority:** `{risk_res['priority_label']}` (Score: {risk_res['priority_score']}/100)")
            with meta_col2:
                st.markdown(f"**Primary Venue:** {summary['venues']}")
                st.markdown(f"**Zone:** {summary['location_zone']}")
            with meta_col3:
                st.markdown(f"**Complaints:** {summary['complaint_count']} total ({summary['unique_complaint_count']} unique)")
                st.markdown(f"**Time Span:** {summary['time_range']}")
            with meta_col4:
                st.markdown(f"**Dominant Symptoms:** {summary['dominant_symptoms']}")
                st.markdown(f"**Common Food:** {summary['common_foods']}")

            st.markdown("---")
            
            # Evidence Matrix & Risk Breakdown
            ev_col1, ev_col2 = st.columns(2)
            
            with ev_col1:
                st.markdown("### 📋 Structured Evidence Points")
                st.markdown(f"**Common-Source Status:** `{cs_res['common_source_label']}` (Score: {cs_res['common_source_score']})")
                for ep in cs_res["evidence_points"]:
                    st.markdown(f"- {ep}")
                    
                st.markdown("### 📊 Priority Factor Breakdown")
                for factor, score_str in risk_res["factor_breakdown"].items():
                    st.markdown(f"- **{factor}:** {score_str}")

            with ev_col2:
                st.markdown("### 🍽️ Inspection History & Venue Status")
                st.markdown(f"- **Inspection Summary:** {summary['inspection_summary']}")
                st.markdown(f"- **Heatwave Vulnerable Resident Flag:** {summary['vulnerable_area_count']} complaints in senior care area.")
                
                st.markdown("### 📝 Member Complaints in Cluster")
                st.dataframe(cluster_df[["complaint_id", "complaint_timestamp", "anonymised_complaint_text", "symptoms", "food_item"]], use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: AI INVESTIGATION SUMMARY ASSISTANT
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🤖 AI Investigation Assistant (Powered by Google Gemini / Resilience Engine)")
    
    if not pipeline.cluster_summaries:
        st.info("No clusters available for AI synthesis.")
    else:
        cluster_ids = [c["cluster_id"] for c in pipeline.cluster_summaries]
        ai_cid = st.selectbox("Select Cluster for AI Briefing", cluster_ids, index=0, key="ai_cluster_select")
        
        if st.button("✨ Generate AI Investigation Briefing"):
            with st.spinner("Generating epidemiological synthesis..."):
                ai_res = pipeline.get_ai_summary(ai_cid)
                
                st.info(f"**Source:** {ai_res['source']} | **Status:** {ai_res['status']}")
                st.markdown(ai_res["content"])

# -----------------------------------------------------------------------------
# TAB 5: ACTION TRACKING & ESCALATIONS
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("📋 Follow-Up Action Tracking & Escalations Manager")
    
    update_automated_statuses()
    investigations = get_all_investigations()
    
    if not investigations:
        st.info("No active follow-up actions in database.")
    else:
        inv_data = []
        for inv in investigations:
            inv_data.append({
                "Investigation ID": inv.investigation_id,
                "Cluster ID": inv.cluster_id,
                "Priority": inv.priority,
                "Status": inv.status,
                "Escalation Level": inv.escalation_level,
                "Owner": inv.owner,
                "Venue": inv.venue_name,
                "Zone": inv.location_zone,
                "Due Date": inv.due_at.strftime("%Y-%m-%d %H:%M"),
                "Notes": inv.summary_notes
            })
            
        inv_df = pd.DataFrame(inv_data)
        st.dataframe(inv_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ✏️ Manage Investigation Action")
        
        inv_ids = [i.investigation_id for i in investigations]
        sel_inv_id = st.selectbox("Select Action to Update", inv_ids)
        
        selected_inv = next(i for i in investigations if i.investigation_id == sel_inv_id)
        
        act_col1, act_col2, act_col3 = st.columns(3)
        with act_col1:
            new_owner = st.text_input("Assign Owner", selected_inv.owner)
        with act_col2:
            new_status = st.selectbox("Status", ["OPEN", "DUE SOON", "OVERDUE", "ESCALATED", "RESOLVED"], index=["OPEN", "DUE SOON", "OVERDUE", "ESCALATED", "RESOLVED"].index(selected_inv.status))
        with act_col3:
            new_escalation = st.selectbox("Escalation Level", ["NONE", "SUPERVISOR_ALERT", "DIRECTOR_ESCALATED"], index=["NONE", "SUPERVISOR_ALERT", "DIRECTOR_ESCALATED"].index(selected_inv.escalation_level))
            
        add_note = st.text_area("Add Investigation Note", "")
        
        if st.button("💾 Save Updates"):
            update_investigation_action(
                investigation_id=sel_inv_id,
                owner=new_owner,
                status=new_status,
                escalation_level=new_escalation,
                notes=add_note if add_note else None
            )
            st.success(f"Updated investigation {sel_inv_id} successfully!")
            st.rerun()
