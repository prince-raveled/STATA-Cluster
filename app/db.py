"""Job store — SQLite, jobs only.

SPEC §20: "no curated database in this project". This table tracks run state and
nothing biological; every result lives on disk beside it, keyed by the same token.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import pickle
import re
import secrets
import shutil

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import declarative_base, sessionmaker

from . import config

Base = declarative_base()


_engine = None
_Session = None


def utcnow() -> dt.datetime:
    """Naive UTC, the shape SQLite stores. `datetime.utcnow` is deprecated."""
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


class Job(Base):
    __tablename__ = "jobs"

    token = Column(String(32), primary_key=True)
    status = Column(String(16), default="uploaded", index=True)  # uploaded|running|done|error
    mode = Column(String(16), default="quick")
    created_at = Column(DateTime, default=utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    progress = Column(Float, default=0.0)
    message = Column(Text, default="")
    error = Column(Text, default="")
    error_hint = Column(Text, default="")
    dataset_name = Column(String(255), default="")
    n_taxa = Column(Integer, default=0)
    n_samples = Column(Integer, default=0)
    n_specs = Column(Integer, default=0)
    runtime_seconds = Column(Float, default=0.0)
    summary_json = Column(Text, default="{}")

    def as_dict(self) -> dict:
        return {
            "token": self.token,
            "status": self.status,
            "mode": self.mode,
            "progress": round(float(self.progress or 0.0), 4),
            "message": self.message or "",
            "error": self.error or "",
            "error_hint": self.error_hint or "",
            "dataset_name": self.dataset_name or "",
            "n_taxa": self.n_taxa,
            "n_samples": self.n_samples,
            "n_specs": self.n_specs,
            "runtime_seconds": round(float(self.runtime_seconds or 0.0), 2),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "summary": json.loads(self.summary_json or "{}"),
        }

    @property
    def expires_at(self):
        if not self.created_at:
            return None
        return self.created_at + dt.timedelta(days=config.RETENTION_DAYS)


def init() -> None:
    global _engine, _Session
    config.ensure_directories()
    _engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)


def session():
    if _Session is None:
        init()
    return _Session()


TOKEN_PATTERN = re.compile(r"^[0-9a-f]{24}$")


def new_token() -> str:
    return secrets.token_hex(12)


def valid_token(token: str) -> bool:
    """Tokens index a directory, so the shape is checked before any path is built."""
    return bool(TOKEN_PATTERN.match(str(token or "")))


def get_job(token: str):
    if not valid_token(token):
        return None
    with session() as db:
        return db.get(Job, token)


def update_job(token: str, **fields) -> None:
    with session() as db:
        job = db.get(Job, token)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        db.commit()


# --- result payloads on disk ---------------------------------------------
def save_payload(token: str, name: str, obj) -> None:
    path = config.job_dir(token) / f"{name}.pkl.gz"
    with gzip.open(path, "wb", compresslevel=4) as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_payload(token: str, name: str):
    path = config.job_dir(token) / f"{name}.pkl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def purge_expired() -> int:
    """SPEC §16.5: results are kept for 90 days, then deleted."""
    cutoff = utcnow() - dt.timedelta(days=config.RETENTION_DAYS)
    removed = 0
    with session() as db:
        stale = db.scalars(select(Job).where(Job.created_at < cutoff)).all()
        for job in stale:
            shutil.rmtree(config.JOBS_DIR / job.token, ignore_errors=True)
            db.delete(job)
            removed += 1
        db.commit()
    return removed
