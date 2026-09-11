"""Run every validation that does not fit in the pytest suite, and report.

The normal suite (`pytest`) checks MicroVerse against itself and against statsmodels.
These scripts check it against *other people's implementations* and against analytic
ground truth, which needs environments and services pytest cannot assume.

    .venv/Scripts/python tests/reference/run_all.py

Each entry says what it needs. Anything unavailable is reported as SKIPPED with the
reason, never silently passed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

MAIN_PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SKBIO_PYTHON = os.path.join(ROOT, ".venv310", "Scripts", "python.exe")


#: The four cohorts the runner uses; the full 17 are in real_data_study.py.
STUDY_COHORTS = ["cdi_schubert", "crc_zhao", "ibd_huttenhower", "edd_singh"]


def _cohorts_available() -> bool:
    """Cached archives are enough — only reach for Zenodo when something is missing."""
    cache = os.path.join(ROOT, "data", "microbiomehd")
    missing = [c for c in STUDY_COHORTS
               if not os.path.exists(os.path.join(cache, f"{c}_results.tar.gz"))]
    if not missing:
        return True
    try:
        request = urllib.request.Request("https://zenodo.org/api/records/1146764",
                                         headers={"User-Agent": "microverse"})
        with urllib.request.urlopen(request, timeout=30):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _rscript_available() -> bool:
    """The R reference packages, not just R: a bare R install proves nothing here."""
    sys.path.insert(0, HERE)
    try:
        from compare_r import find_rscript
    except ImportError:
        return False
    rscript = find_rscript()
    if rscript is None:
        return False
    probe = subprocess.run(
        [rscript, "-e",
         "lib <- Sys.getenv('MICROVERSE_R_LIB', 'D:/Rlocal/library');"
         "if (dir.exists(lib)) .libPaths(lib);"
         "quit(status = if (all(sapply(c('edgeR','ALDEx2'), requireNamespace,"
         " quietly = TRUE))) 0 else 1)"],
        capture_output=True, text=True, check=False)
    return probe.returncode == 0


def _server_is_up(url: str = "http://127.0.0.1:8077/healthz") -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _pelto_exported() -> bool:
    return os.path.exists(os.path.join(ROOT, "data", "pelto", "export", "index.csv"))


def _tierney_exported() -> bool:
    return os.path.exists(os.path.join(ROOT, "data", "cmd", "export", "index.csv"))


CHECKS = [
    {
        # The figure the results page quotes when nothing held up. Recorded so the
        # sentence in the interface has a measurement behind it rather than a guess.
        "name": "Noise floor: what a null table reports",
        "script": "noise_floor.py",
        "python": MAIN_PYTHON,
        "requires": lambda: True,
        "reason": "",
    },
    {
        # Added after the 2026-09-11 audit. Every other reference check compares an
        # *estimate*; this one compares a p-value distribution, which is how ANCOM-BC's
        # anti-conservative tail went unnoticed at r = 1.00 agreement on the estimates.
        "name": "ANCOM-BC null calibration, against R",
        "script": "ancombc_calibration.py",
        "python": MAIN_PYTHON,
        "requires": lambda: True,
        "reason": "",
    },
    {
        "name": "scikit-bio: multi_replace, clr, ancombc",
        "script": "compare_scikit_bio.py",
        "python": SKBIO_PYTHON,
        "requires": lambda: os.path.exists(SKBIO_PYTHON),
        "reason": "needs .venv310 with scikit-bio (py -3.10 -m venv .venv310 && "
                  ".venv310/Scripts/pip install \"scikit-bio>=0.7.1\")",
    },
    {
        "name": "conorm + ground truth: TMM",
        "script": "compare_tmm.py",
        "python": MAIN_PYTHON,
        "requires": lambda: True,
        "reason": "",
    },
    {
        "name": "ground truth: ALDEx2",
        "script": "compare_aldex2.py",
        "python": MAIN_PYTHON,
        "requires": lambda: True,
        "reason": "",
    },
    {
        "name": "R reference: edgeR TMM, ALDEx2 and ANCOM-BC",
        "script": "compare_r.py",
        "python": MAIN_PYTHON,
        "requires": _rscript_available,
        "reason": "needs R with edgeR, ALDEx2 and ANCOMBC: install.packages('BiocManager') "
                  "then BiocManager::install(c('edgeR', 'ALDEx2', 'ANCOMBC', 'microbiome')). "
                  "Set MICROVERSE_RSCRIPT and MICROVERSE_R_LIB if they are not on the "
                  "default paths",
    },
    {
        "name": "concurrency and scale (live server)",
        "script": "load_test.py",
        "python": MAIN_PYTHON,
        "requires": _server_is_up,
        "reason": "needs a server on :8077 "
                  "(.venv/Scripts/python -m uvicorn app.main:app --port 8077)",
    },
    {
        "name": "real published cohorts (SPEC §23 validation 2)",
        "script": "real_data_study.py",
        "python": MAIN_PYTHON,
        "requires": _cohorts_available,
        "reason": "needs the MicrobiomeHD cohorts: cached under data/microbiomehd/, "
                  "or network access to Zenodo record 1146764",
        "args": ["--cohorts", ",".join(STUDY_COHORTS)],
    },
    {
        "name": "reproduce Tierney et al. (SPEC §23 validation 1)",
        "script": "tierney_study.py",
        "python": MAIN_PYTHON,
        "requires": _tierney_exported,
        "reason": "needs the cohorts exported from curatedMetagenomicData: "
                  "Rscript tests/reference/tierney_export.R data/cmd/export "
                  "(R with curatedMetagenomicData; set MICROVERSE_R_LIB if needed)",
    },
    {
        "name": "reproduce Pelto et al. (SPEC §23 validation 3)",
        "script": "pelto_study.py",
        "python": MAIN_PYTHON,
        "requires": _pelto_exported,
        "reason": "needs Pelto's curated cohorts: "
                  "python tests/reference/pelto_fetch.py, then "
                  "Rscript tests/reference/pelto_export.R "
                  "data/pelto/data_171023.rds data/pelto/export",
        "args": ["--max-cohorts", "8", "--max-splits", "6"],
    },
    {
        "name": "robustness tiers predict held-out replication (SPEC §16.2)",
        "script": "tier_validation.py",
        "python": MAIN_PYTHON,
        "requires": _pelto_exported,
        "reason": "needs Pelto's curated cohorts: "
                  "python tests/reference/pelto_fetch.py, then "
                  "Rscript tests/reference/pelto_export.R "
                  "data/pelto/data_171023.rds data/pelto/export",
        "args": ["--max-cohorts", "6", "--repeats", "1"],
    },
    {
        "name": "fork attribution is stable enough to report (SPEC §17)",
        "script": "attribution_sensitivity.py",
        "python": MAIN_PYTHON,
        "requires": _cohorts_available,
        "reason": "needs the MicrobiomeHD cohorts: cached under data/microbiomehd/, "
                  "or network access to Zenodo record 1146764",
        "args": ["--cohorts", ",".join(STUDY_COHORTS)],
        "expect_failure": True,
    },
    {
        "name": "compute benchmark (SPEC §23 validation 5)",
        "script": "benchmark.py",
        "python": MAIN_PYTHON,
        "requires": lambda: True,
        "reason": "",
        "args": ["--quick"],
    },
]


def main() -> int:
    results = []
    for check in CHECKS:
        print("=" * 78)
        print(check["name"])
        print("=" * 78)
        if not check["requires"]():
            print(f"  SKIPPED — {check['reason']}\n")
            results.append((check["name"], "SKIPPED"))
            continue

        command = [check["python"], os.path.join(HERE, check["script"]),
                   *check.get("args", [])]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if check.get("expect_failure"):
            # This check asks whether an estimate is stable enough to report. It
            # currently answers "no", and that answer is the documented finding
            # (SPEC §24.3). Treat the expected outcome as a pass and shout if it
            # changes, in either direction.
            status = "RECORDED" if completed.returncode != 0 else "CHANGED"
        else:
            status = "PASSED" if completed.returncode == 0 else "FAILED"
        results.append((check["name"], status))
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, status in results:
        print(f"  {status:<8} {name}")

    changed = [name for name, status in results if status == "CHANGED"]
    if changed:
        print()
        print(f"{len(changed)} check(s) CHANGED — a recorded negative result now "
              f"passes. Re-read SPEC 24.3 before celebrating; the documented claim "
              f"may need updating in the other direction.")
    failed = [name for name, status in results if status == "FAILED"]
    skipped = [name for name, status in results if status == "SKIPPED"]
    if skipped:
        print(f"\n{len(skipped)} check(s) skipped — see the reasons above.")
    if failed:
        print(f"\n{len(failed)} check(s) FAILED.")
        return 1
    print("\nEvery available check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
