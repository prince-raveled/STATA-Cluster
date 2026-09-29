"""v3 Phase 0 — what a v3 run shows and exports (docs/MICROVERSE_V3_PLAN.md §26).

The engine side is in test_v3_weighting.py. This module checks the pages, exports, API
and records built on it: the grid-composition panel, the weight-stability marks, the
signed-z axis, the methods paragraph and manifest, the defensibility register, and the
rule that replication rates are only ever shown beside the labelling they were measured
on (plan §33, V8).
"""
from __future__ import annotations

import gzip
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import config, db, services  # noqa: E402
from app.core import evidence  # noqa: E402
from app.core.effects import signed_z  # noqa: E402
from app.core.report import (  # noqa: E402
    long_results_csv_gz,
    methods_paragraph,
    run_manifest,
    specification_curve,
    weighting_summary,
)
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.weighted import taxon_axes  # noqa: E402
from app.main import app  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# --- fixtures ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def ibd():
    return services.load_demo("ibd_genus")


@pytest.fixture(scope="module")
def v3(ibd):
    run = run_multiverse(ibd, mode="quick")
    return run, compute_robustness(run)


@pytest.fixture(scope="module")
def v2(ibd):
    run = run_multiverse(ibd, mode="quick", ruleset="v2")
    return run, compute_robustness(run)


@pytest.fixture(scope="module")
def covariate():
    dataset = services.load_demo("t2d_covariates")
    run = run_multiverse(dataset, mode="covariate",
                         covariate_columns=list(dataset.covariate_columns)[:2])
    return run, compute_robustness(run)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("microverse-v3")
    config.DATA_DIR = root
    config.JOBS_DIR = root / "jobs"
    config.DATABASE_URL = f"sqlite:///{(root / 'jobs.sqlite').as_posix()}"
    db.init()
    with TestClient(app) as test_client:
        yield test_client


def _store(ibd, run, summary) -> str:
    """A finished job from an existing run, without the minutes attribution takes.
    Every page tested here reads the stored run and summary; none needs attribution."""
    token = services.create_job(ibd, "ibd_genus_abundance.tsv")
    db.save_payload(token, "run", run)
    db.save_payload(token, "summary", summary)
    db.save_payload(token, "attribution", {})
    db.update_job(token, status="done", progress=1.0, n_specs=run.grid_report.n_valid,
                  runtime_seconds=run.runtime_seconds, finished_at=db.utcnow())
    return token


@pytest.fixture(scope="module")
def v3_token(client, ibd, v3):
    return _store(ibd, *v3)


@pytest.fixture(scope="module")
def v2_token(client, ibd, v2):
    return _store(ibd, *v2)


# --- grid composition ---------------------------------------------------------------
def test_weighting_summary_puts_the_primary_scheme_first(v3):
    _, summary = v3
    facts = weighting_summary(summary)
    assert [s["scheme"] for s in facts["schemes"]][0] == "decision_tree"
    shares = {s["scheme"]: s["rarefied_share"] for s in facts["schemes"]}
    assert shares["decision_tree"] == pytest.approx(0.5)
    assert shares["flat_tree"] == pytest.approx(0.8)
    assert shares["uniform"] == pytest.approx(120 / 130)
    n_eff = {s["scheme"]: s["n_effective"] for s in facts["schemes"]}
    assert n_eff["uniform"] == pytest.approx(1560)
    assert n_eff["decision_tree"] < n_eff["flat_tree"] < n_eff["uniform"]
    unstable = int((~summary.table["weight_stable"]).sum())
    assert facts["n_weight_unstable"] == unstable


def test_a_v2_summary_has_no_weighting_panel(v2):
    assert weighting_summary(v2[1]) is None


def test_per_scheme_shares_explain_each_scheme_tier(v3):
    from app.core.robustness import assign_tier
    table = v3[1].table
    for scheme in ("uniform", "flat_tree", "decision_tree"):
        tiers = [assign_tier(int(n), float(f), float(c)) for n, f, c in zip(
            table["n_specs_tested"], table[f"frac_significant_{scheme}"],
            table[f"sign_consistency_{scheme}"], strict=True)]
        assert tiers == list(table[f"tier_{scheme}"])


def test_table_records_mark_unstable_taxa(v3, v2):
    records = services.table_records(v3[1])
    unstable = [r for r in records if not r["weight_stable"]]
    assert unstable
    for record in unstable:
        assert len(set(record["tiers"].values())) > 1
    assert all("weight_stable" not in r for r in services.table_records(v2[1]))


# --- signed z ---------------------------------------------------------------------------
def test_curve_can_be_drawn_on_the_signed_z_axis(v3):
    run, _ = v3
    effect = specification_curve(run, 0)
    z = specification_curve(run, 0, axis="z")
    assert effect["axis"] == "effect" and z["axis"] == "z"
    assert z["z"] == sorted(z["z"])
    assert effect["effect"] == sorted(effect["effect"])
    assert len(z["z"]) == len(z["effect"]) == z["n_plotted"]
    # The same specifications, only in another order.
    assert sorted(z["spec_id"]) == sorted(effect["spec_id"])
    with pytest.raises(ValueError, match="Unknown axis"):
        specification_curve(run, 0, axis="best")


def test_long_export_has_signed_z_for_v3_only(v3, v2):
    frame = pd.read_csv(io.BytesIO(gzip.decompress(long_results_csv_gz(v3[0]))),
                        low_memory=False)
    expected = signed_z(frame["p_raw"].to_numpy(), frame["effect_native"].to_numpy())
    # rtol: the CSV prints float32 p-values, and at subnormal ones (~1e-45) the printed
    # digits move z in the third decimal.
    assert np.allclose(frame["signed_z"], expected, rtol=1e-3, atol=1e-4)
    header = gzip.decompress(long_results_csv_gz(v2[0])).split(b"\n", 1)[0]
    assert b"signed_z" not in header


def test_taxon_axes_report_every_scheme_and_the_pseudocount(v3):
    run, summary = v3
    axes = taxon_axes(run, summary, 0)
    assert [s["scheme"] for s in axes["schemes"]][0] == "decision_tree"
    assert {s["scheme"] for s in axes["schemes"]} == {"decision_tree", "uniform",
                                                      "flat_tree"}
    assert axes["pseudocount"]["multipliers"] == [0.1, 10.0]
    assert axes["z"]["n"] == int(summary.table.loc[
        summary.table["taxon_id"] == 0, "n_specs_tested"].iloc[0])
    assert axes["within_matrix"] is None          # no covariates in this run


def test_signed_z_spread_is_shown_where_the_effect_cannot_move(covariate):
    run, summary = covariate
    tested = summary.table.loc[summary.table["n_specs_tested"] > 0, "taxon_id"]
    axes = taxon_axes(run, summary, int(tested.iloc[0]))
    within = axes["within_matrix"]
    assert within["effect_sd"] == pytest.approx(0.0, abs=1e-6)
    assert within["z_sd"] > 0


# --- methods paragraph and manifest -----------------------------------------------------
def test_methods_paragraph_states_rules_and_weighting(v3, v2):
    run, summary = v3
    text = methods_paragraph(run, summary)
    assert "rule set v3" in text
    assert "decision tree" in text
    assert "50% of the weight was on rarefied specifications" in text
    assert "(92% with one vote per specification)" in text
    assert "Del Giudice & Gangestad 2021" in text
    legacy = methods_paragraph(*v2)
    assert "rule set" not in legacy and "weighted" not in legacy


def test_manifest_states_rules_weighting_and_whose_evidence(v3, v2):
    manifest = run_manifest(*v3)
    assert manifest["ruleset"] == "v3"
    assert manifest["rules_applied"][-2:] == ["R8", "R9"]
    assert manifest["weighting"]["scheme"] == "decision_tree"
    validation = manifest["validation"]
    labelling = validation["tier_replication_labelling"]
    assert labelling == {"ruleset": "v3", "scheme": "decision_tree",
                         "measured": evidence.labelling_measured("v3", "decision_tree")}
    if not labelling["measured"]:
        assert validation["tier_replication"] == {}
    json.dumps(manifest, allow_nan=False, default=str)       # the API serves it

    legacy = run_manifest(*v2)
    assert "ruleset" not in legacy and "weighting" not in legacy
    assert set(legacy["validation"]["tier_replication"]) == set(evidence.TIER_REPLICATION)


# --- evidence belongs to its labelling ---------------------------------------------------
def test_v2_rates_are_never_attributed_to_another_labelling():
    assert evidence.tier_sentence("ROBUST", "v2", "uniform")
    for scheme in ("uniform", "flat_tree", "decision_tree"):
        if not evidence.labelling_measured("v3", scheme):
            assert evidence.tier_evidence("ROBUST", "v3", scheme) == {}
            assert evidence.tier_sentence("ROBUST", "v3", scheme) == ""
            assert evidence.labelling_summary("v3", scheme) == {
                "measured": False, "experiment": "V8", "record": None}


def test_a_labelling_counts_as_measured_only_from_a_registered_record(monkeypatch):
    record = _synthetic_v8(registered=False)
    monkeypatch.setattr(evidence, "WEIGHTING_VALIDATION", record)
    assert set(evidence.replication_by_labelling()) == {("v2", "uniform")}
    monkeypatch.setattr(evidence, "WEIGHTING_VALIDATION", _synthetic_v8(registered=True))
    measured = evidence.replication_by_labelling()
    assert ("v3", "decision_tree") in measured
    # v2's own record stays the source for v2, whatever V8's rerun of it says.
    assert measured[("v2", "uniform")] is evidence.TIER_REPLICATION


def test_the_validation_matrix_grades_weighted_tiers_by_the_record(monkeypatch):
    monkeypatch.setattr(evidence, "WEIGHTING_VALIDATION", {})
    assert evidence._weighted_tiers().grade == evidence.INTERNAL
    monkeypatch.setattr(evidence, "WEIGHTING_VALIDATION", _synthetic_v8(registered=True))
    component = evidence._weighted_tiers()
    assert component.grade == evidence.EMPIRICAL
    assert "0.700" in component.empirical and "non-inferior" in component.empirical


# --- pages -------------------------------------------------------------------------------
def _text(html: str) -> str:
    """Rendered text, whitespace collapsed (template source wraps sentences)."""
    import re
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def test_results_page_shows_the_composition_panel(client, v3_token):
    page = client.get(f"/results/{v3_token}").text
    assert 'id="grid-composition"' in page
    assert 'id="curve-axis"' in page
    text = _text(page)
    assert "Where the weight sits" in text
    assert "changes with weighting" in text
    if not evidence.labelling_measured("v3", "decision_tree"):
        assert "replication not yet measured for these labels" in text
        for facts in evidence.TIER_REPLICATION.values():
            assert f"{facts['rate']:.0%} replicated" not in text


def test_a_v2_results_page_is_unchanged_in_kind(client, v2_token):
    page = client.get(f"/results/{v2_token}").text
    assert 'id="grid-composition"' not in page
    assert 'id="curve-axis"' not in page
    text = _text(page)
    assert f"{evidence.TIER_REPLICATION['ROBUST']['rate']:.0%} replicated" in text
    assert "not yet measured" not in text


def test_taxon_page_shows_each_scheme_and_both_axes(client, v3_token):
    page = client.get(f"/results/{v3_token}/taxon/0").text
    assert 'class="axis-toggle"' in page
    text = _text(page)
    assert "How the label depends on counting" in text
    assert "Does the pseudocount move it?" in text
    assert "signed z" in text


def test_curve_endpoints_take_an_axis(client, v3_token):
    z = client.get(f"/results/{v3_token}/curve/0?axis=z").json()
    assert z["axis"] == "z" and z["z"] == sorted(z["z"])
    assert client.get(f"/results/{v3_token}/curve/0?axis=best").status_code == 422
    api = client.get(f"/api/jobs/{v3_token}/curve/0?axis=z").json()
    assert api["z"] == z["z"]
    assert client.get(f"/api/jobs/{v3_token}/curve/0?axis=best").status_code == 422


def test_api_exposes_the_labelling_and_weighting(client, v3_token):
    body = client.get(f"/api/jobs/{v3_token}/results?limit=3").json()
    assert body["labelling"]["ruleset"] == "v3"
    assert body["labelling"]["scheme"] == "decision_tree"
    assert body["weighting"]["schemes"][0]["scheme"] == "decision_tree"
    assert "weight_stable" in body["taxa"][0]
    info = client.get("/api/info").json()
    assert info["default_ruleset"] == "v3" and info["default_weighting"] == "decision_tree"
    assert info["rulesets"]["v3"][-2:] == ["R8", "R9"]
    assert set(info["curve_axes"]) == {"effect", "z"}


def test_about_renders_the_register_and_links_the_note(client):
    page = client.get("/about").text
    assert 'id="register"' in page
    for rule in ("R1", "R8", "R9"):
        assert f">{rule}" in page
    assert "docs/v3_weighting_note.md" in page
    assert 'id="weighting"' in page


def test_validation_page_reports_v8_as_it_stands(client, monkeypatch):
    import app.templating as templating
    page = client.get("/validation").text
    assert 'id="v8"' in page
    if not evidence.WEIGHTING_VALIDATION.get("registered_design"):
        assert "Not yet run" in page
    # The results branch of the template, rendered from a synthetic record.
    monkeypatch.setitem(templating.templates.env.globals, "V8_RECORD",
                        _synthetic_v8(registered=True))
    page = client.get("/validation").text
    assert "non-inferior" in page and "0.700" in page


# --- records the site is built from --------------------------------------------------------
def test_defensibility_register_on_disk_matches_the_engine():
    sys.path.insert(0, os.path.join(HERE, "reference"))
    import make_defensibility_register

    path = os.path.join(ROOT, "docs", "defensibility_register.json")
    with open(path, encoding="utf-8") as handle:
        on_disk = handle.read()
    assert on_disk == make_defensibility_register.render(), (
        "docs/defensibility_register.json is stale: run "
        "tests/reference/make_defensibility_register.py")


def test_the_app_reads_the_same_v8_record_as_docs():
    docs = os.path.join(ROOT, "docs", "weighting_validation.json")
    copy = evidence.WEIGHTING_RECORD
    if not os.path.exists(docs) and not os.path.exists(copy):
        pytest.skip("V8 has not been run (tests/reference/weighting_validation.py)")
    assert os.path.exists(docs) and os.path.exists(copy), "one copy of the V8 record"
    with open(docs, encoding="utf-8") as a, open(copy, encoding="utf-8") as b:
        assert a.read() == b.read(), "app/core/records/ differs from docs/"


# --- V8's statistics --------------------------------------------------------------------------
def test_v8_scores_labels_exactly_as_the_v2_experiment_did():
    """V8 must measure AUC the way tier_validation.py did, or the v2 baseline it reruns
    would not be comparable with the record it is checked against."""
    sys.path.insert(0, os.path.join(HERE, "reference"))
    import tier_validation as tv
    import weighting_validation as wv

    rng = np.random.default_rng(3)
    n = 400
    tiers = rng.choice(["ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE", "NOT DETECTED"], n)
    rows = pd.DataFrame({"cohort": rng.choice(list("abcdef"), n), "seed": 101,
                         "taxon": [f"t{i}" for i in range(n)]})
    for label, _, _ in wv.LABELLINGS:
        rows[f"tier_{label}"] = tiers
        for outcome in wv.OUTCOMES:
            rows[f"{outcome}_{label}"] = rng.random(n) < np.where(tiers == "ROBUST", .8, .2)
    view = wv.view(rows, "v3_decision_tree")
    expected = tv.discrimination(view, "replicated_majority")["auc"]
    assert wv.auc(view, "replicated_majority") == pytest.approx(expected, abs=1e-12)

    # Identical labellings and outcomes: no difference, and the interval says so.
    for label, _, _ in wv.LABELLINGS:
        rows[f"replicated_majority_{label}"] = rows["replicated_majority_v3_uniform"]
    same = wv.paired_auc_difference(rows, "v3_decision_tree", "v3_uniform",
                                    "replicated_majority", draws=200)
    assert same["difference"] == 0.0 and same["ci"] == [0.0, 0.0]
    assert wv.label_changes(rows, "v3_uniform")["v3_decision_tree"]["share_changed"] == 0
    assert evidence.V8_PREREGISTRATION["margin"] == wv.MARGIN


# --- helpers ---------------------------------------------------------------------------------
def _synthetic_v8(registered: bool) -> dict:
    """A record with V8's shape and made-up numbers, for exercising code paths only.
    It is never written anywhere and never reaches a page outside this test."""
    def labelling(ruleset, scheme, auc):
        rows = [{"tier": t, "n_taxa": 10, "n_cohorts": 4, "replication_rate": 0.5,
                 "ci_low": 0.4, "ci_high": 0.6}
                for t in ("ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE", "NOT DETECTED")]
        return {"ruleset": ruleset, "scheme": scheme, "definitions": {
            "replicated_majority": {"by_tier": rows, "discrimination": {"auc": auc},
                                    "null_rate": 0.01}}}
    return {
        "registered_design": registered, "n_cohorts": 4, "n_splits": 8,
        "n_observations": 1000, "code_commit": {"sha": "0" * 40, "dirty": False},
        "labellings": {
            "v3_uniform": labelling("v3", "uniform", 0.71),
            "v3_flat_tree": labelling("v3", "flat_tree", 0.70),
            "v3_decision_tree": labelling("v3", "decision_tree", 0.70),
            "v2_uniform": labelling("v2", "uniform", 0.72),
        },
        "non_inferiority": {"replicated_majority": {
            "difference": -0.01, "ci": [-0.02, 0.0], "margin": -0.03,
            "non_inferior": True}},
        "label_changes": {"from_v3_uniform": {
            "v3_flat_tree": {"share_changed": 0.05},
            "v3_decision_tree": {"share_changed": 0.1},
            "v2_uniform": {"share_changed": 0.02}}},
    }
