import os
import pandas as pd
from src.data_generator import generate_synthetic_dataset
from src.preprocessing import preprocess_complaints
from src.duplicate_detection import detect_duplicates
from src.feature_engineering import build_pairwise_similarity_matrix, DEFAULT_WEIGHTS
from src.clustering import perform_clustering
from src.common_source import evaluate_common_source
from src.risk_scoring import calculate_cluster_risk_priority
from src.followup import create_or_get_investigation, update_automated_statuses
from src.gemini_service import generate_investigation_summary

class InvestigationPipeline:
    """
    Core orchestrator for the Food-Borne Illness Investigation Tool.
    """
    def __init__(self, data_path="data/synthetic/synthetic_complaints.csv"):
        self.data_path = data_path
        self.raw_df = None
        self.processed_df = None
        self.dedup_df = None
        self.cluster_summaries = []
        self.cluster_details = {}
        self.total_sim_matrix = None
        self.distance_matrix = None
        
    def load_and_process_data(self, force_generate=False):
        if force_generate or not os.path.exists(self.data_path):
            self.raw_df = generate_synthetic_dataset(num_records=5000, output_path=self.data_path)
        else:
            self.raw_df = pd.read_csv(self.data_path)
            
        self.processed_df = preprocess_complaints(self.raw_df)
        self.dedup_df, self.processed_df = detect_duplicates(
            self.processed_df, similarity_threshold=0.85, time_window_hours=12
        )
        return self.processed_df, self.dedup_df

    def run_clustering_pipeline(self, weights=None, method="agglomerative", distance_threshold=0.45):
        if self.dedup_df is None:
            self.load_and_process_data()
            
        if weights is None:
            weights = DEFAULT_WEIGHTS
            
        self.total_sim_matrix, self.distance_matrix, _ = build_pairwise_similarity_matrix(self.dedup_df, weights)
        self.dedup_df, self.cluster_summaries = perform_clustering(
            self.dedup_df, self.distance_matrix, method=method, distance_threshold=distance_threshold
        )
        
        # Enrich cluster summaries with common-source and risk priority evaluation
        self.cluster_details = {}
        for summary in self.cluster_summaries:
            cid = summary["cluster_id"]
            cluster_subset = self.dedup_df[self.dedup_df["cluster_id"] == cid]
            
            cs_res = evaluate_common_source(cluster_subset, summary)
            risk_res = calculate_cluster_risk_priority(cluster_subset, summary, cs_res)
            
            summary["common_source_score"] = cs_res["common_source_score"]
            summary["common_source_label"] = cs_res["common_source_label"]
            summary["priority_label"] = risk_res["priority_label"]
            summary["priority_score"] = risk_res["priority_score"]
            
            self.cluster_details[cid] = {
                "summary": summary,
                "common_source": cs_res,
                "risk": risk_res,
                "dataframe": cluster_subset
            }
            
            # Automatically populate high-priority cluster investigations into SQLite DB
            if risk_res["priority_label"] in ["HIGH", "CRITICAL", "MEDIUM"]:
                create_or_get_investigation(
                    cluster_id=cid,
                    venue_name=summary["venues"],
                    location_zone=summary["location_zone"],
                    priority=risk_res["priority_label"],
                    complaint_count=summary["complaint_count"],
                    due_hours=risk_res["recommended_due_hours"]
                )
                
        update_automated_statuses()
        return self.cluster_summaries, self.cluster_details

    def get_ai_summary(self, cluster_id):
        if cluster_id not in self.cluster_details:
            return {"content": "Cluster details not found.", "status": "ERROR"}
            
        details = self.cluster_details[cluster_id]
        return generate_investigation_summary(
            cluster_summary=details["summary"],
            common_source_res=details["common_source"],
            risk_res=details["risk"],
            cluster_df=details["dataframe"]
        )
