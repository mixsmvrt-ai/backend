import logging
import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger("riddimbase_backend.db")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./riddimbase_jobs.db")

# Compatible kwargs for SQLite and Postgres.
_engine_kwargs: dict[str, object] = {
    "pool_pre_ping": True,
    "future": True,
}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    # Import here to avoid circular import.
    from models import ProcessingJob  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized for job processing.")
