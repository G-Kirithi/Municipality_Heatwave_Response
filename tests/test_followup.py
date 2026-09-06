import os
import uuid
import pytest
from datetime import datetime, timedelta
from database.db import init_db, get_session
from database.models import InvestigationAction
from src.followup import create_or_get_investigation, update_automated_statuses, update_investigation_action

@pytest.fixture
def test_db_path(tmp_path):
    db_file = tmp_path / f"test_{uuid.uuid4().hex}.db"
    return str(db_file)

def test_investigation_creation_and_escalation(test_db_path):
    action = create_or_get_investigation(
        cluster_id="CLUSTER-TEST",
        venue_name="Test Outbreak Spot",
        location_zone="Zone-1 North",
        priority="CRITICAL",
        due_hours=-2, # Set in past to trigger overdue/escalation
        db_path=test_db_path
    )
    assert action.status == "OPEN"
    
    # Run automated status update
    updated_count = update_automated_statuses(db_path=test_db_path)
    assert updated_count > 0
    
    session = get_session(test_db_path)
    refreshed = session.query(InvestigationAction).filter_by(cluster_id="CLUSTER-TEST").first()
    assert refreshed.status == "ESCALATED"
    assert refreshed.escalation_level == "SUPERVISOR_ALERT"
    session.close()

def test_investigation_resolution(test_db_path):
    action = create_or_get_investigation(
        cluster_id="CLUSTER-RESOLVE",
        venue_name="Resolve Spot",
        location_zone="Zone-2 South",
        priority="MEDIUM",
        due_hours=24,
        db_path=test_db_path
    )
    
    updated = update_investigation_action(
        investigation_id=action.investigation_id,
        status="RESOLVED",
        notes="Inspected venue. Refrigerator unit replaced. Corrective order issued.",
        db_path=test_db_path
    )
    assert updated.status == "RESOLVED"
    assert "Inspected venue" in updated.summary_notes
