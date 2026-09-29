"""The calibrated job type end to end — v3 plan §27.5.

A calibrated run is the Quick grid plus `inference.calibrate`: started from the
configure form's checkbox or the API's mode, recorded as its own job type, shown beside
the descriptive tiers and never in place of them. Attribution is stubbed and the number
of permutations reduced, because this module tests the plumbing; the engine itself is
tests/test_inference.py.
"""
from __future__ import annotations

import io
import zipfile

import pytest

pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import config, db, services, ui  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("microverse-calibrated")
    config.DATA_DIR = root
    config.JOBS_DIR = root / "jobs"
    config.DATABASE_URL = f"sqlite:///{(root / 'jobs.sqlite').as_posix()}"
    db.init()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def calibrated(client):
    """A calibrated demo run started from the configure form."""
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "CALIBRATION_PERMUTATIONS", 99)
    patch.setattr(services, "attribute", lambda run: {})
    try:
        response = client.get("/demo/ibd_genus", follow_redirects=False)
        token = response.headers["location"].rsplit("/", 1)[-1]
        assert 'name="calibrate"' in client.get(f"/configure/{token}").text
        client.post(f"/run/{token}", data={"mode": "quick", "calibrate": "1"},
                    follow_redirects=False)
    finally:
        patch.undo()
    job = db.get_job(token)
    assert job.status == "done", job.error
    return token


def test_the_checkbox_makes_a_calibrated_job(calibrated):
    job = db.get_job(calibrated)
    assert job.mode == "calibrated"
    _, run, summary, _ = services.load_results(calibrated)
    assert run.mode == "quick"                      # the grid it ran
    assert summary.calibration.n_permutations == 99
    for column in ("calibrated_q", "certified_share", "certified_robust"):
        assert column in summary.table.columns


def test_results_page_shows_calibration_beside_the_tiers(client, calibrated):
    page = client.get(f"/results/{calibrated}").text
    assert 'id="calibration"' in page
    assert "Calibrated error control" in page
    assert "calibrated · 99 permutations" in page
    assert 'title: "Calibrated q"' in page
    assert 'id="tier-cards"' not in page or "ROBUST" in page      # tiers stay
    assert "CERTIFIED ROBUST" in page


def test_curve_marks_certified_specifications(client, calibrated):
    _, _, summary, _ = services.load_results(calibrated)
    certified = summary.table.loc[summary.table["certified_robust"], "taxon_id"]
    assert len(certified), "the demo's planted effects should produce certified taxa"
    taxon = int(certified.iloc[0])
    curve = client.get(f"/results/{calibrated}/curve/{taxon}").json()
    assert len(curve["certified"]) == curve["n_plotted"]
    assert any(curve["certified"])
    api = client.get(f"/api/jobs/{calibrated}/curve/{taxon}").json()
    assert api["certified"] == curve["certified"]


def test_exports_carry_the_calibration(client, calibrated):
    manifest = client.get(f"/download/{calibrated}/manifest").json()
    calibration = manifest["calibration"]
    assert calibration["permutations"] == 99
    assert calibration["seed"] == config.CALIBRATION_SEED
    assert len(calibration["permutations_sha256"]) == 64
    assert calibration["assumptions"]
    methods = client.get(f"/download/{calibrated}/methods").text
    assert "Error control was calibrated by permutation" in methods
    rejections = client.get(f"/download/{calibrated}/calibration")
    assert rejections.status_code == 200
    assert rejections.text.startswith("spec_id,signed_z,p_within,taxon_name")
    bundle = zipfile.ZipFile(io.BytesIO(client.get(f"/download/{calibrated}/bundle").content))
    assert "calibration_rejections.csv" in bundle.namelist()
    assert "This run: calibrated" in bundle.read("README.txt").decode()


def test_api_reports_the_calibration(client, calibrated):
    body = client.get(f"/api/jobs/{calibrated}/results?limit=3").json()
    assert body["calibration"]["n_selected"] >= body["calibration"]["n_certified_robust"]
    assert "certified_robust" in body["taxa"][0]
    info = client.get("/api/info").json()
    assert "calibrated" in info["modes"]


def test_an_uncalibrated_run_has_no_calibration(client):
    patch = pytest.MonkeyPatch()
    patch.setattr(services, "attribute", lambda run: {})
    try:
        response = client.get("/demo/ibd_genus", follow_redirects=False)
        token = response.headers["location"].rsplit("/", 1)[-1]
        client.post(f"/run/{token}", data={"mode": "quick"}, follow_redirects=False)
    finally:
        patch.undo()
    assert db.get_job(token).mode == "quick"
    _, _, summary, _ = services.load_results(token)
    assert summary.calibration is None
    assert "calibrated_q" not in summary.table.columns
    assert 'id="calibration"' not in client.get(f"/results/{token}").text
    assert client.get(f"/download/{token}/calibration").status_code == 404


def test_a_calibrated_job_shows_its_extra_stage():
    job = type("Job", (), {"status": "running", "mode": "calibrated",
                           "message": "Permuting: matrix 12 of 160", "progress": 0.98})()
    stages = ui.run_stages(job)
    assert [s["key"] for s in stages][-2:] == ["calibrate", "save"]
    current = [s for s in stages if s["state"] == "current"]
    assert current[0]["key"] == "calibrate"
    assert current[0]["detail"] == "matrix 12 of 160"
    plain = type("Job", (), {"status": "running", "mode": "quick",
                             "message": "Saving results", "progress": 0.99})()
    assert "calibrate" not in [s["key"] for s in ui.run_stages(plain)]


def test_certified_robust_is_described_by_what_v9_has_shown(client, calibrated):
    """Until V9 has run, every place that presents CERTIFIED ROBUST says it has not been
    validated on real data; afterwards, what V9 found."""
    import re

    from app.core.evidence import certified_status
    sentence = certified_status()["sentence"]

    def text(html):
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        return " ".join(re.sub(r"<[^>]+>", " ", html).split()).replace("&#39;", "'")

    _, _, summary, _ = services.load_results(calibrated)
    taxon = int(summary.table["taxon_id"].iloc[0])
    for path in (f"/results/{calibrated}", f"/results/{calibrated}/taxon/{taxon}",
                 f"/configure/{calibrated}", "/about"):
        assert sentence in text(client.get(path).text), path
    assert sentence in client.get(f"/download/{calibrated}/methods").text
    manifest = client.get(f"/download/{calibrated}/manifest").json()
    assert manifest["calibration"]["validation"]["sentence"] == sentence
