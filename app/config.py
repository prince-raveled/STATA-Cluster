"""Runtime configuration. Everything is overridable by environment variable."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MICROVERSE_DATA", BASE_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
EXAMPLES_DIR = BASE_DIR / "examples"
DATABASE_URL = os.environ.get("MICROVERSE_DB", f"sqlite:///{(DATA_DIR / 'jobs.sqlite').as_posix()}")

#: SPEC §16.5 — permanent token URL, 90-day retention.
RETENTION_DAYS = int(os.environ.get("MICROVERSE_RETENTION_DAYS", "90"))
#: Uploads are bounded so a single request cannot exhaust a small free-tier dyno.
MAX_UPLOAD_BYTES = int(os.environ.get("MICROVERSE_MAX_UPLOAD", str(64 * 1024 * 1024)))

VERSION = "1.0.0"
SPEC_VERSION = "2.0 (frozen)"
LICENSE = "MIT"
REPOSITORY = os.environ.get("MICROVERSE_REPO", "https://github.com/your-org/microverse")

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


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(token: str) -> Path:
    path = JOBS_DIR / token
    path.mkdir(parents=True, exist_ok=True)
    return path
