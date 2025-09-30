from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session


# PUBLIC_INTERFACE
def get_database_url() -> str:
    """Return the database URL from environment variable POSTGRES_URL.

    Note: You must set POSTGRES_URL in the .env file (handled by the orchestrator).
    Example: postgresql+psycopg://user:password@host:5432/dbname
    """
    url = os.environ.get("POSTGRES_URL", "")
    return url


DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()


# PUBLIC_INTERFACE
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency to provide a database session.

    Yields a SQLAlchemy Session if POSTGRES_URL is configured; otherwise raises RuntimeError.
    """
    if SessionLocal is None:
        raise RuntimeError("Database is not configured. Set POSTGRES_URL in environment.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
