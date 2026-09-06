import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, create_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class InvestigationAction(Base):
    __tablename__ = "investigation_actions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    investigation_id = Column(String(50), unique=True, nullable=False)
    cluster_id = Column(String(50), nullable=False)
    owner = Column(String(100), default="Unassigned")
    priority = Column(String(20), default="MEDIUM")
    created_at = Column(DateTime, default=datetime.utcnow)
    due_at = Column(DateTime, nullable=False)
    status = Column(String(30), default="OPEN") # OPEN, DUE SOON, OVERDUE, ESCALATED, RESOLVED
    escalation_level = Column(String(50), default="NONE") # NONE, SUPERVISOR_ALERT, DIRECTOR_ESCALATED
    venue_name = Column(String(200), default="Unknown")
    location_zone = Column(String(100), default="Zone-1")
    complaint_count = Column(Integer, default=0)
    summary_notes = Column(Text, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
