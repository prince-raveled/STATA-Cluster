"""Job store — one table, jobs only.

SPEC §20: "no curated database in this project". This table tracks run state and
nothing biological; every result lives in `storage` beside it, keyed by the same
token. Nothing here is a scientific record, which is why it can move hosts.

SQLite is the default and is what a server with a disk should keep using. A host
without one points MICROVERSE_DB at a networked database instead; the schema is
plain enough to be portable, and the two places the databases genuinely differ —
connection handling and the race in creating the table — are handled below rather
than assumed away.
"""
from __future__ import annotations

import datetime as dt
import json
import secrets

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import declarative_base, sessionmaker

from . import config, storage

Base = declarative_base()


_engine = None
_Session = None


def utcnow() -> dt.datetime:
    """Naive UTC. Both SQLite and TIMESTAMP WITHOUT TIME ZONE store exactly this,
    so the same value round-trips either way. `datetime.utcnow` is deprecated."""
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


def _engine_options(url: str) -> dict:
    """Connection settings, which differ by driver rather than by deployment.

    SQLite needs `check_same_thread=False` because the analysis runs on a different
    thread from the request that started it. A networked database needs the opposite
    kind of care: a serverless host opens a new connection per invocation and a
    scale-to-zero database drops idle ones, so connections are verified before use
    and not pooled across invocations that will never reuse them.
    """
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    options = {"pool_pre_ping": True}
    if config.is_serverless():
        from sqlalchemy.pool import NullPool
        options["poolclass"] = NullPool
    return options


def create_schema(engine) -> None:
    """Create the table, tolerating another instance creating it at the same moment.

    `create_all` looks for the table and then creates it, which is two statements with
    a gap in between. One process owning a SQLite file never sees that gap. A
    serverless host does: every cold start runs this, several can start at once, and
    two of them can both find the table absent and both try to create it. The one that
    loses gets an error about a table that now exists, which is not a failure —
    anything else still is.
    """
    try:
        Base.metadata.create_all(engine)
    except (IntegrityError, OperationalError, ProgrammingError):
        if not sa_inspect(engine).has_table(Job.__tablename__):
            raise


def check_database_url(url: str, on_vercel: bool) -> None:
    """Refuse the one database choice that cannot work on Vercel, and say why.

    SQLite is the default because a server with a disk should use it. On Vercel it is
    impossible: the project directory is read-only, and each instance's scratch disk
    is its own and disappears with it, so a job written by one invocation would be
    invisible to the next. Left alone, the failure is "unable to open database file"
    from deep inside SQLAlchemy on the first request, which names neither the setting
    nor the fix.
    """
    if on_vercel and url.startswith("sqlite"):
        raise RuntimeError(
            "MICROVERSE_DB is not set, so the job database would be SQLite, which "
            "cannot work on Vercel: each instance's disk is its own and does not "
            "outlive it. Set MICROVERSE_DB to a PostgreSQL URL (see .env.example)."
        )


def init() -> None:
    global _engine, _Session
    check_database_url(config.DATABASE_URL, config.ON_VERCEL)
    config.ensure_directories()
    _engine = create_engine(config.DATABASE_URL, **_engine_options(config.DATABASE_URL))
    create_schema(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)


def session():
    if _Session is None:
        init()
    return _Session()


#: One definition, shared with `storage`, which applies it again before any path or
#: key is built from a token.
TOKEN_PATTERN = storage.TOKEN_PATTERN


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


# --- result payloads ------------------------------------------------------
# Kept as thin wrappers rather than removed: every caller in the application and in
# the suite already speaks this pair, and where the bytes actually go is `storage`'s
# decision, not this module's.
def save_payload(token: str, name: str, obj) -> None:
    storage.put_object(token, name, obj)


def load_payload(token: str, name: str):
    return storage.get_object(token, name)


def purge_expired() -> int:
    """SPEC §16.5: results are kept for 90 days, then deleted."""
    cutoff = utcnow() - dt.timedelta(days=config.RETENTION_DAYS)
    removed = 0
    with session() as db:
        stale = db.scalars(select(Job).where(Job.created_at < cutoff)).all()
        for job in stale:
            storage.purge(job.token)
            db.delete(job)
            removed += 1
        db.commit()
    return removed
