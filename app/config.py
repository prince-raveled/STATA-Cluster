"""Runtime configuration. Everything is overridable by environment variable."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MICROVERSE_DATA", BASE_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
EXAMPLES_DIR = BASE_DIR / "examples"
def _with_postgres_driver(url: str) -> str:
    """Name psycopg 3 explicitly on a PostgreSQL URL that did not name a driver.

    A hosted database hands out `postgresql://...`, and SQLAlchemy reads a bare
    `postgresql://` as psycopg *2* — a different package, which this project does not
    install and does not want. The failure is a ModuleNotFoundError for psycopg2 on a
    deployment that has psycopg 3 sitting right there.

    `postgres://` is normalised too. Some providers still issue it and SQLAlchemy
    dropped it, so it does not fail over to a wrong driver; it fails to load any.

    A URL that already names a driver is left exactly as it is: asking for something
    specific is an instruction, not an oversight.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


DATABASE_URL = _with_postgres_driver(
    os.environ.get("MICROVERSE_DB", f"sqlite:///{(DATA_DIR / 'jobs.sqlite').as_posix()}"))

#: Where a job's bytes live: "local" (a directory, the default and what the tests
#: exercise) or "blob" (Vercel Blob, for hosts without a disk that outlives a request).
STORAGE_BACKEND = os.environ.get("MICROVERSE_STORAGE", "local").strip().lower()
#: How the analysis is started: "inline" (Starlette's background threadpool, the
#: default) or "queue" (a message whose delivery invokes the worker route).
JOB_BACKEND = os.environ.get("MICROVERSE_JOBS", "inline").strip().lower()
#: Lifetime of a signed download link. Long enough to click, short enough that a
#: copied URL is not a permanent public handle on someone's data.
DOWNLOAD_URL_TTL_SECONDS = int(os.environ.get("MICROVERSE_DOWNLOAD_TTL", "900"))
#: Lifetime of a browser's permission to write one uploaded object. Long enough for
#: 64 MB on a slow connection, short enough that a leaked one stops working quickly.
UPLOAD_URL_TTL_SECONDS = int(os.environ.get("MICROVERSE_UPLOAD_TTL", "1800"))
#: Whether result objects are readable by URL alone. "private" is the default and
#: keeps every read behind the store credential, which means the app streams
#: downloads and the host's response cap applies to them. "public" lets a browser
#: fetch a result directly from an unguessable URL, which is the only way past that
#: cap with this SDK -- and a different promise about the data, so it is opt-in.
BLOB_ACCESS = os.environ.get("MICROVERSE_BLOB_ACCESS", "private").strip().lower()
#: Route that mints a browser upload token. Empty disables direct upload, and the
#: browser posts the form instead. Only the JavaScript SDK can sign these, so this
#: points at api/blob-upload.js rather than anything in this application.
BLOB_UPLOAD_HANDLER = os.environ.get("MICROVERSE_BLOB_HANDLER", "/api/blob-upload").strip()
#: Shared secret the worker route requires, so only the queue can start an analysis.
WORKER_SECRET = os.environ.get("MICROVERSE_WORKER_SECRET", "")
#: How long an undelivered run request stays claimable. A run that could not start
#: because every slot was busy must still be there when one frees up, so this is set
#: well beyond QUEUE_WAIT_SECONDS rather than at the queue's 24-hour default.
QUEUE_RETENTION_SECONDS = int(os.environ.get("MICROVERSE_QUEUE_TTL", str(6 * 3600)))
#: Absolute base URL of this deployment, used to address the worker route. Vercel
#: sets VERCEL_URL to the deployment host without a scheme.
_vercel_host = os.environ.get("VERCEL_PROJECT_PRODUCTION_URL") or os.environ.get("VERCEL_URL", "")
PUBLIC_BASE_URL = os.environ.get(
    "MICROVERSE_BASE_URL", f"https://{_vercel_host}" if _vercel_host else "").rstrip("/")

#: SPEC §16.5 — permanent token URL, 90-day retention.
RETENTION_DAYS = int(os.environ.get("MICROVERSE_RETENTION_DAYS", "90"))
#: Uploads are bounded so a single request cannot exhaust a small free-tier dyno.
MAX_UPLOAD_BYTES = int(os.environ.get("MICROVERSE_MAX_UPLOAD", str(64 * 1024 * 1024)))

VERSION = "1.0.0"
SPEC_VERSION = "2.0 (frozen)"
LICENSE = "MIT"
#: The "Source" link in the header and footer. The default was a placeholder
#: (your-org/microverse), which 404s on GitHub — override it per deployment.
REPOSITORY = os.environ.get(
    "MICROVERSE_REPO", "https://github.com/prince-raveled/STATA-Cluster")

MODE_LABELS = {
    "quick": "Quick",
    "full": "Full",
    "covariate": "Covariate",
}
MODE_BLURBS = {
    "quick": "Forks 1-4 plus the four elementary methods and FDR. The default, and the "
             "mode with the best evidence behind it — Pelto et al. 2025 found elementary "
             "methods the most replicable.",
    "full": "Quick, plus ANCOM-BC, ALDEx2 and PyDESeq2 over a stratified sample of "
            "matrices. Slower; the sampling fraction is reported.",
    "covariate": "Fork 6: every subset of the covariates you choose, over a reference "
                 "sub-grid of the other forks. This is the Tierney et al. 2022 analysis.",
}


def is_serverless() -> bool:
    """True when the disk and the process both end with the request."""
    return STORAGE_BACKEND != "local" or JOB_BACKEND != "inline"


def ensure_directories() -> None:
    """Create the data directories, tolerating a host that has no writable project dir.

    A serverless filesystem is read-only apart from a scratch directory, and on that
    kind of host nothing durable lives here anyway — the database holds the job rows
    and object storage holds the bytes. Failing to create a directory nothing will be
    written to must not stop the application from importing.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        if not is_serverless():
            raise


def job_dir(token: str) -> Path:
    path = JOBS_DIR / token
    path.mkdir(parents=True, exist_ok=True)
    return path
