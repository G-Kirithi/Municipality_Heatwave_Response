import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, InvestigationAction

DB_PATH = "database/investigation.db"

def get_engine(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False, connect_args={"check_same_thread": False})
    return engine

def init_db(db_path=DB_PATH):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

def get_session(db_path=DB_PATH):
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()
