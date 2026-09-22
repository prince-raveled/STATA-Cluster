"""The two seams that let MicroVerse run on a host without a durable disk.

`storage` decides where a job's bytes go; `jobs` decides who calls the engine. Both
default to exactly what MicroVerse has always done, and the rest of the suite is the
evidence for that: it exercises the real code path, not a stub. What is pinned here is
the part the rest of the suite cannot see — that the abstraction is faithful, that the
alternative backends are selected only by configuration, and that the worker route
cannot be used to start work on the server by anyone who asks.
"""
from __future__ import annotations

import gzip
import io
import pickle

import pytest
from fastapi.testclient import TestClient

from app import config, db, jobs, storage
from app.core.models import Specification
from app.main import app

TOKEN = "0123456789abcdef01234567"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _local_backend():
    """Every test here runs against the disk backend unless it says otherwise."""
    storage.reset()
    yield
    storage.reset()


# --- storage ---------------------------------------------------------------
def test_bytes_survive_a_round_trip():
    storage.put_bytes(TOKEN, "bundle.zip", b"PK\x03\x04payload")
    assert storage.get_bytes(TOKEN, "bundle.zip") == b"PK\x03\x04payload"
    storage.purge(TOKEN)


def test_a_missing_object_reads_as_none_not_an_error():
    """The download route relies on this to fall back to regenerating the export."""
    assert storage.get_bytes(TOKEN, "never_written.zip") is None
    assert storage.get_object(TOKEN, "never_written") is None


def test_an_object_round_trips_and_stays_a_gzipped_pickle():
    """The on-disk format is unchanged, so results written before this seam still load."""
    payload = {"taxa": [1, 2, 3], "tier": "ROBUST"}
    storage.put_object(TOKEN, "summary", payload)

    raw = storage.get_bytes(TOKEN, "summary.pkl.gz")
    assert raw is not None, "the object should land under the historical filename"
    assert raw[:2] == b"\x1f\x8b", "still gzip"
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
        assert pickle.load(handle) == payload

    assert storage.get_object(TOKEN, "summary") == payload
    storage.purge(TOKEN)


def test_purge_removes_everything_for_one_token():
    storage.put_bytes(TOKEN, "a.txt", b"a")
    storage.put_text(TOKEN, "b.txt", "b")
    storage.purge(TOKEN)
    assert storage.get_bytes(TOKEN, "a.txt") is None
    assert storage.get_bytes(TOKEN, "b.txt") is None


def test_the_local_backend_asks_the_app_to_stream():
    """A None URL is what tells the download route to serve the file itself."""
    assert storage.download_url(TOKEN, "bundle.zip") is None


def test_the_blob_backend_is_only_reachable_by_configuration(monkeypatch):
    """Nothing selects Blob implicitly, and selecting it must not import the SDK early."""
    assert storage.backend().name == "local"
    monkeypatch.setattr(config, "STORAGE_BACKEND", "blob")
    storage.reset()
    assert storage.backend().name == "blob"
    assert storage.backend()._key(TOKEN, "bundle.zip") == f"microverse/{TOKEN}/bundle.zip"


# --- job dispatch -----------------------------------------------------------
class RecordingBackground:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args):
        self.tasks.append((fn, args))


def test_the_default_dispatch_is_the_original_background_task():
    from app import services

    background = RecordingBackground()
    jobs.dispatch(background, TOKEN, "quick", None, ("age",))

    assert len(background.tasks) == 1
    fn, args = background.tasks[0]
    assert fn is services.execute
    assert args == (TOKEN, "quick", None, ("age",))


def test_the_queue_backend_publishes_instead_of_scheduling(monkeypatch):
    sent = {}
    monkeypatch.setattr(config, "JOB_BACKEND", "queue")
    monkeypatch.setattr(jobs, "_publish", lambda payload: sent.update(payload))

    background = RecordingBackground()
    jobs.dispatch(background, TOKEN, "full", None, ("age", "bmi"))

    assert background.tasks == [], "the queue backend must not also schedule locally"
    assert sent["token"] == TOKEN
    assert sent["mode"] == "full"
    assert sent["covariates"] == ["age", "bmi"]


def test_a_declared_pipeline_survives_the_message(monkeypatch):
    """The message carries a reference and a spec, never the dataset itself."""
    declared = Specification(
        rarefaction="none", rare_seed=None, rank="input", prev_filter=0.10,
        transform="tss", method="wilcoxon", fdr_method="bh", fdr_threshold=0.05,
        covariates=("age", "sex"),
    )
    payload = jobs._payload(TOKEN, "quick", declared, ())
    assert "dataset" not in payload

    restored = jobs.rebuild_declared(payload["declared"])
    assert restored == declared


def test_no_declared_pipeline_stays_none():
    assert jobs.rebuild_declared(None) is None
    assert jobs.rebuild_declared({}) is None


# --- the worker route -------------------------------------------------------
def test_the_worker_refuses_everyone_when_no_secret_is_configured(client, monkeypatch):
    """An unconfigured deployment must not leave an open way to start work."""
    monkeypatch.setattr(config, "WORKER_SECRET", "")
    response = client.post("/internal/run", json={"token": TOKEN},
                           headers={"X-Microverse-Worker": ""})
    assert response.status_code == 404


def test_the_worker_refuses_a_wrong_secret(client, monkeypatch):
    monkeypatch.setattr(config, "WORKER_SECRET", "the-real-secret")
    response = client.post("/internal/run", json={"token": TOKEN},
                           headers={"X-Microverse-Worker": "not-it"})
    assert response.status_code == 404


def test_the_worker_rejects_a_malformed_token(client, monkeypatch):
    monkeypatch.setattr(config, "WORKER_SECRET", "s")
    response = client.post("/internal/run", json={"token": "../../etc/passwd"},
                           headers={"X-Microverse-Worker": "s"})
    assert response.status_code == 400


def test_a_message_for_a_vanished_job_is_accepted_not_retried(client, monkeypatch):
    """Retention expired. Redelivery will never succeed, so it must not look like one."""
    monkeypatch.setattr(config, "WORKER_SECRET", "s")
    response = client.post("/internal/run", json={"token": TOKEN},
                           headers={"X-Microverse-Worker": "s"})
    assert response.status_code == 200
    assert response.json()["status"] == "gone"


def test_a_finished_job_is_not_run_a_second_time(client, monkeypatch):
    """Delivery is at-least-once, and a rerun would overwrite a result being read."""
    monkeypatch.setattr(config, "WORKER_SECRET", "s")
    with db.session() as session:
        session.add(db.Job(token=TOKEN, status="done", mode="quick"))
        session.commit()
    try:
        response = client.post("/internal/run", json={"token": TOKEN},
                               headers={"X-Microverse-Worker": "s"})
        assert response.status_code == 200
        assert response.json()["status"] == "done"
    finally:
        with db.session() as session:
            job = session.get(db.Job, TOKEN)
            if job is not None:
                session.delete(job)
                session.commit()


def test_the_worker_route_is_not_advertised(client):
    """It is infrastructure, not API surface."""
    schema = client.get("/api/openapi.json").json()
    assert not [p for p in schema["paths"] if p.startswith("/internal")]


# --- configuration ----------------------------------------------------------
def test_the_defaults_are_the_original_behaviour():
    assert config.STORAGE_BACKEND == "local"
    assert config.JOB_BACKEND == "inline"
    assert config.is_serverless() is False


def test_sqlite_and_postgres_get_different_connection_settings():
    """SQLite needs cross-thread access; a networked database needs liveness checks."""
    sqlite = db._engine_options("sqlite:///jobs.sqlite")
    assert sqlite["connect_args"]["check_same_thread"] is False

    postgres = db._engine_options("postgresql+psycopg://u:p@host/db")
    assert "connect_args" not in postgres
    assert postgres["pool_pre_ping"] is True
