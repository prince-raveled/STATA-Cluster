"""V9's script — tests/reference/v9_calibration.py — checked without its data.

V9 itself is run where the Pelto and Nearing collections are (docs/V3_DEVIATIONS.md).
These tests run every code path on simulated tables with a handful of permutations,
write only to a temporary directory, and check that the metrics mean what the script
says they mean. Nothing here is, or is reported as, a V9 result.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "reference"))
sys.path.insert(0, HERE)

import v9_calibration as v9  # noqa: E402
from conftest import make_counts  # noqa: E402

from app.core.evidence import V9_PREREGISTRATION  # noqa: E402


class Args:
    parts = "abc"
    permutations = 19
    splits_a = 1
    replicates_b = 1
    replicates_c = 1


def _template(seed: int):
    counts, _, _, _ = make_counts(n_taxa=40, n_per_group=20, n_differential=0, seed=seed)
    frame = pd.DataFrame(counts, index=[f"T{i:02d}" for i in range(40)],
                         columns=[f"S{seed}_{j:02d}" for j in range(40)])
    return v9._named(frame)


class _Cache(v9.Cache):
    def __init__(self):
        super().__init__("", None, use=False)


def test_thresholds_come_from_the_preregistration():
    assert V9_PREREGISTRATION["q"] == v9.Q
    limit = V9_PREREGISTRATION["q"] + V9_PREREGISTRATION["fdr_margin"]
    assert pytest.approx(limit) == v9.FDR_LIMIT
    assert V9_PREREGISTRATION["uniformity_ks_alpha"] == v9.KS_ALPHA


def test_planting_changes_only_group_b_in_expectation():
    rng = np.random.default_rng(1)
    counts = rng.poisson(50, size=(30, 40))
    in_b = v9.halves(40, rng)
    taxa = np.array([3, 7])
    out = v9.plant(counts, in_b, taxa, np.array([2.0, -2.0]), rng)
    assert (out.sum(axis=0) == counts.sum(axis=0)).all()          # libraries kept
    rel = out / out.sum(axis=0)
    assert rel[3, in_b].mean() > 2.5 * rel[3, ~in_b].mean()
    assert rel[7, in_b].mean() < rel[7, ~in_b].mean() / 2.5


def test_planted_taxa_follow_the_design():
    counts = _template(3).to_numpy()
    rng = np.random.default_rng(2)
    standard, folds = v9.planted_taxa(counts, "standard", rng)
    assert len(standard) <= v9.PLANTED and set(np.abs(folds)) <= {1.0, 2.0}
    top = np.argsort(-(counts / counts.sum(axis=0)).mean(axis=1))[: v9.PLANT_SKIP_TOP]
    assert not set(standard) & set(top)
    dominant, _ = v9.planted_taxa(counts, "dominant_shift", np.random.default_rng(2))
    assert set(top[: v9.DOMINANT]) <= set(dominant)


def test_metrics_mean_what_the_docstring_says():
    rows = pd.DataFrame([
        {"false_discoveries": 1, "discoveries": 4, "true_discoveries": 3, "n_planted": 6,
         "certified": 2, "certified_true": 2, "null_taxa": 30, "null_family_rejections": 3},
        {"false_discoveries": 0, "discoveries": 0, "true_discoveries": 0, "n_planted": 6,
         "certified": 0, "certified_true": 0, "null_taxa": 30, "null_family_rejections": 0},
    ])
    block = v9.metrics(rows)
    assert block["fdr"] == pytest.approx((1 / 4 + 0) / 2)
    assert block["power"] == pytest.approx((3 / 6 + 0) / 2)
    assert block["within_family_fwer"] == pytest.approx(3 / 60)
    assert block["certified_precision"] == 1.0
    assert block["any_false_discovery_rate"] == 0.5


def test_every_part_runs_end_to_end_on_simulated_templates(monkeypatch):
    monkeypatch.setattr(v9, "FAILURES", [])
    monkeypatch.setattr(v9, "SIM_N", (20,))
    monkeypatch.setattr(v9, "SIM_SPARSITY", {"low": 500.0})
    monkeypatch.setattr(v9, "SIM_FOLDS", (4.0,))
    templates = [("pelto", "sim_a", _template(11)), ("nearing", "sim_b", _template(12))]
    cache = _Cache()
    rows = (v9.run_part_a(templates, Args, cache) + v9.run_part_b(templates, Args, cache)
            + v9.run_part_c(Args, cache))
    frame = pd.DataFrame(rows)
    assert set(frame["part"]) == {"a", "b", "c"}
    assert (frame.loc[frame["part"] == "a", "n_planted"] == 0).all()
    assert (frame.loc[frame["part"] != "a", "n_planted"] > 0).all()
    parts = v9.analyse(frame, "abc")
    assert parts["a"]["n_datasets"] == 2
    assert set(parts["b"]["scenarios"]) == {"standard", "dominant_shift"}
    assert parts["c"]["n_replicates"] == 1
    result = v9.verdict(parts)
    names = [c["name"] for c in result["criteria"]]
    assert any(n.startswith("(a)") for n in names)
    assert any(n.startswith("(b)") for n in names)
    assert any(n.startswith("(c)") for n in names)
    json.dumps({"parts": parts, "verdict": result}, default=float)


def test_a_trial_run_never_writes_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr(v9, "FAILURES", [])
    monkeypatch.setattr(v9, "SIM_N", (20,))
    monkeypatch.setattr(v9, "SIM_SPARSITY", {"low": 500.0})
    monkeypatch.setattr(v9, "SIM_FOLDS", (2.0,))
    monkeypatch.setattr(v9, "CACHE", str(tmp_path / "rows.jsonl"))
    output = tmp_path / "v9.json"
    before = os.path.exists(v9.OUTPUT), os.path.exists(v9.APP_COPY)
    v9.main(["--parts", "c", "--replicates-c", "1", "--permutations", "19",
             "--output", str(output), "--no-cache"])
    record = json.loads(output.read_text())
    assert record["registered_design"] is False
    assert record["experiment"] == "V9" and "c" in record["parts"]
    assert (os.path.exists(v9.OUTPUT), os.path.exists(v9.APP_COPY)) == before


def test_without_its_data_the_script_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(v9, "nearing_templates", lambda *a: [])
    assert v9.main(["--pelto-export", str(tmp_path)]) == 0
    assert "SKIPPED" in capsys.readouterr().out


# --- what the site says about V9 and about CERTIFIED ROBUST ---------------------------------
def _synthetic_record(passed: bool) -> dict:
    """V9's shape with made-up numbers, to exercise the rendering only. Never written."""
    return {
        "registered_design": True, "code_commit": {"sha": "0" * 40},
        "parts": {
            "a": {"n_datasets": 3, "n_replicates": 12, "ks_p": 0.5,
                  "any_discovery_rate": 0.02},
            "b": {"scenarios": {
                "standard": {"fdr": 0.04, "within_family_fwer": 0.05, "power": 0.6,
                             "certified_precision": 0.9},
                "dominant_shift": {"fdr": 0.2, "within_family_fwer": 0.1, "power": 0.5,
                                   "certified_precision": None}}},
            "c": {"n_replicates": 18, "overall": {"fdr": 0.05, "power": 0.7}},
        },
        "verdict": {"criteria": [
            {"name": "(a) uniform", "value": 0.5, "threshold": "> 0.01", "passed": True},
            {"name": "(b) standard FDR", "value": 0.04, "threshold": "<= 0.06",
             "passed": passed}],
            "restricted_to": [] if passed else ["simulated settings"]},
    }


def test_certified_robust_is_unvalidated_until_v9_has_run():
    from app.core import evidence
    status = evidence.certified_status({})
    assert status["validated"] is False
    assert "not yet been validated on real data" in status["sentence"]
    assert evidence.certified_status(_synthetic_record(True))["validated"] is True
    failed = evidence.certified_status(_synthetic_record(False))
    assert failed["validated"] is False and "restricted to" in failed["sentence"]


def test_the_matrix_grades_calibration_by_the_v9_record():
    from app.core import evidence
    assert evidence._calibrated({}).grade == evidence.INTERNAL
    component = evidence._calibrated(_synthetic_record(True))
    assert component.grade == evidence.EMPIRICAL
    assert "FDR 0.040" in component.empirical


def test_validation_page_shows_v9_as_it_stands(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import app.templating as templating
    from app.core import evidence
    from app.main import app
    with TestClient(app) as client:
        page = client.get("/validation").text
        assert 'id="v9"' in page
        if not evidence.V9_RECORD:
            assert "Not yet run" in page and "not yet been validated on real data" in page
        record = _synthetic_record(True)
        monkeypatch.setitem(templating.templates.env.globals, "V9_RECORD", record)
        monkeypatch.setitem(templating.templates.env.globals, "CERTIFIED_STATUS",
                            evidence.certified_status(record))
        page = client.get("/validation").text
        assert "(b) standard FDR" in page and "dominant shift" in page


def test_the_app_reads_the_same_v9_record_as_docs():
    from app.core import evidence
    docs = os.path.join(os.path.dirname(HERE), "docs", "v9_calibration.json")
    copy = evidence.V9_RECORD_PATH
    if not os.path.exists(docs) and not os.path.exists(copy):
        pytest.skip("V9 has not been run (tests/reference/v9_calibration.py)")
    assert os.path.exists(docs) and os.path.exists(copy), "one copy of the V9 record"
    with open(docs, encoding="utf-8") as a, open(copy, encoding="utf-8") as b:
        assert a.read() == b.read(), "app/core/records/ differs from docs/"
