from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = "sqlite:///./data/aipark.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + threading
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a database session and ensures it closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all database tables if they don't already exist."""
    from app.core import models  # noqa: F401 — import to register models
    
    # Ensure data directory exists for SQLite
    if DATABASE_URL.startswith("sqlite:///"):
        import os
        db_path = DATABASE_URL.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
    Base.metadata.create_all(bind=engine)
