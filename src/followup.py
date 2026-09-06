from datetime import datetime, timedelta
from database.db import get_session, init_db
from database.models import InvestigationAction

def update_automated_statuses(db_path="database/investigation.db"):
    """
    Checks all active investigation actions in SQLite and updates their statuses
    based on due dates and escalation rules.
    
    Status Lifecycle:
    OPEN -> DUE SOON -> OVERDUE -> ESCALATED -> RESOLVED
    """
    init_db(db_path)
    session = get_session(db_path)
    now = datetime.utcnow()
    
    actions = session.query(InvestigationAction).all()
    updated_count = 0
    
    for action in actions:
        if action.status == "RESOLVED":
            continue
            
        old_status = action.status
        time_left_hours = (action.due_at - now).total_seconds() / 3600.0
        
        if time_left_hours < 0:
            action.status = "OVERDUE"
            if action.priority in ["HIGH", "CRITICAL"]:
                action.status = "ESCALATED"
                action.escalation_level = "SUPERVISOR_ALERT"
        elif time_left_hours <= 6.0:
            if action.status == "OPEN":
                action.status = "DUE SOON"
                
        if action.status != old_status:
            action.updated_at = now
            updated_count += 1
            
    session.commit()
    session.close()
    return updated_count

def create_or_get_investigation(cluster_id, venue_name, location_zone, priority="HIGH", complaint_count=0, due_hours=24, db_path="database/investigation.db"):
    """
    Creates a new investigation action record if one does not already exist for the cluster.
    """
    init_db(db_path)
    session = get_session(db_path)
    
    existing = session.query(InvestigationAction).filter(InvestigationAction.cluster_id == cluster_id).first()
    if existing:
        session.close()
        return existing
        
    created_at = datetime.utcnow()
    due_at = created_at + timedelta(hours=due_hours)
    inv_id = f"INV-{created_at.strftime('%Y%m%d')}-{cluster_id}"
    
    new_action = InvestigationAction(
        investigation_id=inv_id,
        cluster_id=cluster_id,
        owner="Food Safety Officer Alpha",
        priority=priority,
        created_at=created_at,
        due_at=due_at,
        status="OPEN",
        escalation_level="NONE",
        venue_name=venue_name,
        location_zone=location_zone,
        complaint_count=complaint_count,
        summary_notes="Initial automated investigation created based on common-source risk trigger."
    )
    
    session.add(new_action)
    session.commit()
    session.refresh(new_action)
    session.close()
    return new_action

def update_investigation_action(investigation_id, owner=None, status=None, due_at=None, notes=None, escalation_level=None, db_path="database/investigation.db"):
    """
    Updates an investigation action record in SQLite.
    """
    init_db(db_path)
    session = get_session(db_path)
    
    action = session.query(InvestigationAction).filter(InvestigationAction.investigation_id == investigation_id).first()
    if not action:
        session.close()
        return None
        
    if owner is not None:
        action.owner = owner
    if status is not None:
        action.status = status
    if due_at is not None:
        action.due_at = due_at
    if notes is not None:
        if action.summary_notes:
            action.summary_notes += f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}] {notes}"
        else:
            action.summary_notes = notes
    if escalation_level is not None:
        action.escalation_level = escalation_level
        
    action.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(action)
    session.close()
    return action

def get_all_investigations(db_path="database/investigation.db"):
    """
    Retrieves all investigation actions from database.
    """
    init_db(db_path)
    session = get_session(db_path)
    actions = session.query(InvestigationAction).order_by(InvestigationAction.created_at.desc()).all()
    session.close()
    return actions
