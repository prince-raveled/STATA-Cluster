"""The job store on a networked database instead of a file.

MicroVerse keeps nothing biological in its database — one table of run state, keyed
by the same token the results are stored under — which is what makes moving it to
another host a configuration change rather than a migration. What these tests check
is that the move is actually only that: the same schema, the same values, and the
two places SQLite and PostgreSQL genuinely differ handled rather than assumed away.

Connecting to a real database is not one of the things checked here. Everything
below runs against the dialect, the compiler and the driver as installed, which is
enough to catch a type that will not map, a driver that is not there, or settings
chosen for the wrong engine. It is not enough to prove Neon accepts a connection,
and nothing here pretends otherwise.
"""
from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.schema import CreateTable

from app import config, db

NEON_URL = "postgresql+psycopg://user:password@ep-example.aws.neon.tech/microverse?sslmode=require"


# --- the driver -------------------------------------------------------------
def test_the_postgresql_driver_is_installed():
    """Without it the first query raises ModuleNotFoundError, not a clear error."""
    import psycopg

    assert psycopg.__version__.startswith("3."), "the URL scheme assumes psycopg 3"


def test_the_driver_is_declared_as_a_runtime_dependency():
    """It has to be in requirements.txt, not merely in this developer's venv."""
    requirements = (config.BASE_DIR / "requirements.txt").read_text(encoding="utf-8")
    declared = [line for line in requirements.splitlines()
                if line.strip().startswith("psycopg")]
    assert declared, "no psycopg entry in requirements.txt"
    assert "[binary]" in declared[0], (
        "psycopg[binary] ships a prebuilt libpq; the source build needs a C toolchain "
        "and system Postgres headers, which a function build image does not have"
    )


def test_a_neon_url_builds_a_working_engine():
    """Constructing the engine resolves the dialect and imports the driver."""
    engine = create_engine(NEON_URL)
    assert engine.dialect.name == "postgresql"
    assert engine.dialect.driver == "psycopg"


# --- the schema is the same schema -----------------------------------------
def test_every_column_maps_onto_postgresql():
    """A type that will not compile is a deployment that fails on first use."""
    ddl = str(CreateTable(db.Job.__table__).compile(dialect=postgresql.dialect()))
    assert "CREATE TABLE jobs" in ddl
    assert "PRIMARY KEY (token)" in ddl
    assert "TIMESTAMP WITHOUT TIME ZONE" in ddl


def test_the_two_dialects_describe_the_same_table():
    """Same columns, same order, same nullability -- only the type spellings differ."""
    pg = str(CreateTable(db.Job.__table__).compile(dialect=postgresql.dialect()))
    lite = str(CreateTable(db.Job.__table__).compile(dialect=sqlite.dialect()))

    def columns(ddl):
        return [line.strip().split()[0] for line in ddl.splitlines()
                if line.startswith("\t") and not line.strip().startswith("PRIMARY KEY")]

    assert columns(pg) == columns(lite)
    assert len(columns(pg)) == len(db.Job.__table__.columns)


def test_nothing_biological_is_stored_in_the_database():
    """SPEC 20. The table is run state; results live in storage, keyed by token."""
    names = set(db.Job.__table__.columns.keys())
    assert not names & {"counts", "taxa", "abundance", "p_values", "effects", "tiers"}
    assert "token" in names


# --- where the databases actually differ ------------------------------------
def test_sqlite_and_postgresql_get_different_connection_settings():
    sqlite_opts = db._engine_options("sqlite:///jobs.sqlite")
    assert sqlite_opts["connect_args"]["check_same_thread"] is False, (
        "the analysis runs on a different thread from the request that started it"
    )

    pg_opts = db._engine_options(NEON_URL)
    assert "connect_args" not in pg_opts, "check_same_thread is a SQLite-only argument"
    assert pg_opts["pool_pre_ping"] is True, (
        "a scale-to-zero database drops idle connections; they must be verified"
    )


def test_a_serverless_host_does_not_pool_connections(monkeypatch):
    """Each invocation is a new process, so a pooled connection is never reused."""
    from sqlalchemy.pool import NullPool

    monkeypatch.setattr(config, "STORAGE_BACKEND", "blob")
    assert config.is_serverless()
    assert db._engine_options(NEON_URL)["poolclass"] is NullPool


def test_a_server_with_a_disk_still_pools(monkeypatch):
    monkeypatch.setattr(config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(config, "JOB_BACKEND", "inline")
    assert not config.is_serverless()
    assert "poolclass" not in db._engine_options(NEON_URL)


# --- the create-table race --------------------------------------------------
def test_creating_the_schema_twice_is_not_an_error(tmp_path):
    """Every cold start runs it, so it has to be safe to run repeatedly."""
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    db.create_schema(engine)
    db.create_schema(engine)
    assert inspect(engine).has_table("jobs")


def test_losing_the_create_race_is_tolerated_when_the_table_exists(tmp_path, monkeypatch):
    """Two instances can both find the table absent and both try to create it.

    The one that loses gets an error about a table that now exists. On one process
    owning a SQLite file this never happens; on a serverless host it will.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    db.create_schema(engine)          # the instance that won

    def already_exists(*_args, **_kwargs):
        raise ProgrammingError("CREATE TABLE jobs", {}, Exception("duplicate table"))

    monkeypatch.setattr(db.Base.metadata, "create_all", already_exists)
    db.create_schema(engine)          # must not raise: the table is there


def test_a_real_failure_still_raises(tmp_path, monkeypatch):
    """Tolerating the race must not swallow a database that is genuinely broken."""
    engine = create_engine(f"sqlite:///{tmp_path / 'absent.sqlite'}")

    def cannot_connect(*_args, **_kwargs):
        raise ProgrammingError("CREATE TABLE jobs", {}, Exception("permission denied"))

    monkeypatch.setattr(db.Base.metadata, "create_all", cannot_connect)
    with pytest.raises(ProgrammingError):
        db.create_schema(engine)      # the table does not exist, so this is real


# --- the local default is untouched ----------------------------------------
def test_the_default_is_still_sqlite_on_disk():
    assert config.DATABASE_URL.startswith("sqlite:///")
    assert "jobs.sqlite" in config.DATABASE_URL


def test_the_url_alone_chooses_the_database(monkeypatch):
    """Switching hosts is configuration; no code path branches on deployment."""
    for url, expected in ((NEON_URL, "postgresql"), ("sqlite:///x.db", "sqlite")):
        assert create_engine(url).dialect.name == expected


def test_job_rows_round_trip_on_a_real_engine(tmp_path):
    """The persistence the application actually performs, end to end.

    SQLite here, because that is the engine available without credentials. What this
    pins is the shape: insert, read back, update one field, and the timestamps and
    JSON payload surviving as written.
    """
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    db.create_schema(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    created = dt.datetime(2026, 1, 2, 3, 4, 5)
    with Session() as s:
        s.add(db.Job(token="a" * 24, status="running", mode="quick",
                     created_at=created, progress=0.5,
                     summary_json='{"tiers": {"ROBUST": 7}}'))
        s.commit()

    with Session() as s:
        job = s.get(db.Job, "a" * 24)
        assert job.created_at == created, "naive UTC must survive unchanged"
        assert job.as_dict()["summary"] == {"tiers": {"ROBUST": 7}}
        job.status = "done"
        s.commit()

    with Session() as s:
        assert s.get(db.Job, "a" * 24).status == "done"
        assert s.get(db.Job, "a" * 24).progress == 0.5, "an unrelated field was lost"
