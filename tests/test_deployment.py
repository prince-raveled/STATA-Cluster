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
import re

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


# --- what the host installs -------------------------------------------------
def test_the_project_declares_its_runtime_dependencies():
    """A host that installs this project reads `dependencies`, not requirements.txt.

    With that list empty the build succeeds, installs the package and nothing else,
    and the first request fails on `import fastapi`. The build log says it installed
    dependencies, because it did — there were none. Declaring them dynamically from
    requirements.txt keeps one list; this asserts the wiring actually resolves.
    """
    import tomllib

    manifest = tomllib.loads(
        (config.BASE_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    project = manifest["project"]

    static = project.get("dependencies")
    dynamic = "dependencies" in project.get("dynamic", [])
    assert static or dynamic, (
        "pyproject.toml declares no dependencies, so installing this project "
        "installs none of them"
    )
    if dynamic:
        source = manifest["tool"]["setuptools"]["dynamic"]["dependencies"]["file"]
        assert "requirements.txt" in source, source


def test_the_declared_dependencies_are_the_runtime_ones():
    """Whatever the mechanism, fastapi and the engine's stack must come out of it."""
    import importlib.metadata as md

    try:
        declared = md.distribution("microverse").requires or []
    except md.PackageNotFoundError:
        pytest.skip("microverse is not installed as a package in this environment")

    names = {re.split(r"[<>=!;\[ ]", entry.strip())[0].lower() for entry in declared}
    for required in ("fastapi", "sqlalchemy", "numpy", "scipy", "pandas",
                     "statsmodels", "pydeseq2", "psycopg", "vercel"):
        assert required in names, f"{required} is not declared; the host will not install it"


# --- the demo data has to reach the host ------------------------------------
def test_the_demo_datasets_are_generated_at_build_time():
    """examples/*.tsv is gitignored, so a host clones a repository without them.

    Nothing can include a file that is not there, which is why this is a build step
    rather than an inclusion rule. Docker and CI already run the same script; the
    deployment now does too.
    """
    import json as _json
    import tomllib  # noqa: F401  (kept for symmetry with the other manifest tests)

    manifest = _json.loads((config.BASE_DIR / "vercel.json").read_text(encoding="utf-8"))
    service = manifest["services"]["microverse"]
    assert "make_examples" in service.get("buildCommand", ""), (
        "nothing regenerates the demo datasets, which are not in the repository"
    )

    gitignore = (config.BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    assert "examples/*_abundance.tsv" in gitignore, (
        "if the datasets became tracked files, the build step is no longer the "
        "mechanism and this test is asserting the wrong thing"
    )


def test_the_generated_datasets_are_bundled_with_the_function():
    """Generated during the build is not the same as present in the bundle."""
    import json as _json

    manifest = _json.loads((config.BASE_DIR / "vercel.json").read_text(encoding="utf-8"))
    entry = manifest["services"]["microverse"]["functions"]["app/main.py"]
    assert "examples" in entry.get("includeFiles", "")
    assert "examples" not in entry.get("excludeFiles", "")


def test_every_file_load_demo_opens_exists():
    """The three datasets, three files each. A missing one is a 500 on /demo/<name>."""
    from app.services import DEMO_DATASETS

    missing = [
        f"{name}_{kind}.tsv"
        for name in DEMO_DATASETS
        for kind in ("abundance", "metadata", "taxonomy")
        if not (config.EXAMPLES_DIR / f"{name}_{kind}.tsv").exists()
    ]
    assert not missing, f"load_demo would raise FileNotFoundError for: {missing}"


def test_the_demo_routes_answer(client):
    """What the runtime log showed failing, asserted end to end."""
    from app.services import DEMO_DATASETS

    for name in DEMO_DATASETS:
        response = client.get(f"/demo/{name}", follow_redirects=False)
        assert response.status_code == 303, f"/demo/{name} -> {response.status_code}"
        assert response.headers["location"].startswith("/configure/")


# --- which backend a deployment actually gets -------------------------------
def _reload(monkeypatch, env):
    """Re-read configuration as a fresh process would, with exactly this environment."""
    import importlib

    for name in ("VERCEL", "VERCEL_ENV", "MICROVERSE_STORAGE", "MICROVERSE_JOBS",
                 "MICROVERSE_BLOB_ACCESS", "MICROVERSE_BLOB_HANDLER"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    importlib.reload(config)
    importlib.reload(storage)
    storage.reset()
    return storage.backend().name


@pytest.fixture(autouse=True)
def _restore_config():
    """Whatever a test does to the environment, put the module back afterwards."""
    import importlib

    yield
    importlib.reload(config)
    importlib.reload(storage)
    storage.reset()


BLANK = {  # exactly what a dashboard sets when the Value fields are left empty
    "MICROVERSE_STORAGE": "",
    "MICROVERSE_JOBS": "",
    "MICROVERSE_BLOB_ACCESS": "",
    "MICROVERSE_BLOB_HANDLER": "",
}


def test_a_blank_value_is_not_a_choice(monkeypatch):
    """`os.environ.get(name, default)` answers "" for a variable set to nothing.

    That is not the default and not a valid setting; it fell past every branch into
    whichever was written last, which was the local filesystem. On Vercel that means
    writing to /var/task and failing with EROFS on the first result.
    """
    backend = _reload(monkeypatch, BLANK)
    assert config.STORAGE_BACKEND == "local", "blank must fall back, not become ''"
    assert backend == "local"
    assert config.BLOB_UPLOAD_HANDLER == "/api/blob-upload", (
        "a blank handler disabled direct upload entirely"
    )


def test_vercel_selects_blob_even_when_nothing_is_configured(monkeypatch):
    """The local backend cannot work there, so it is not the default there."""
    backend = _reload(monkeypatch, {**BLANK, "VERCEL": "1", "VERCEL_ENV": "preview"})
    assert config.ON_VERCEL
    assert config.STORAGE_BACKEND == "blob"
    assert backend == "blob"
    assert config.JOB_BACKEND == "queue", "inline needs a process that outlives the response"


def test_local_development_still_selects_the_filesystem(monkeypatch):
    """Nothing about the default off Vercel changes."""
    backend = _reload(monkeypatch, {})
    assert not config.ON_VERCEL
    assert config.STORAGE_BACKEND == "local"
    assert config.JOB_BACKEND == "inline"
    assert backend == "local"


def test_an_explicit_setting_wins_everywhere(monkeypatch):
    """Asking for something specific is an instruction, on any host."""
    backend = _reload(monkeypatch, {"VERCEL": "1", "MICROVERSE_STORAGE": "local"})
    assert config.STORAGE_BACKEND == "local" and backend == "local"

    backend = _reload(monkeypatch, {"MICROVERSE_STORAGE": "blob"})
    assert not config.ON_VERCEL
    assert config.STORAGE_BACKEND == "blob" and backend == "blob"


@pytest.mark.parametrize("value", ["  blob  ", "BLOB", "Blob	"])
def test_a_setting_is_read_past_case_and_padding(monkeypatch, value):
    assert _reload(monkeypatch, {"MICROVERSE_STORAGE": value}) == "blob"


def test_healthz_reports_the_backends_without_reporting_secrets(client):
    """A deployment running as something other than its configuration must say so."""
    body = client.get("/healthz").json()
    assert body["config"]["storage"] in ("local", "blob")
    assert body["config"]["jobs"] in ("inline", "queue")
    assert body["config"]["database"] in ("sqlite", "postgresql")

    rendered = str(body).lower()
    for leak in ("token", "password", "secret", "sslmode", "@"):
        assert leak not in rendered, f"/healthz exposed {leak!r}"


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
