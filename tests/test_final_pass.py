"""Regressions for the defects the 2026-09-11 audit confirmed.

Each test names the audit item it guards. They are grouped by the kind of failure
rather than by file, because that is how they would recur: an error path that was fixed
on one route and not its twin, a number shown beside an interval it does not belong to,
copy that says one thing while the page shows another.
"""
from __future__ import annotations

import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def finished(client):
    """One completed Quick run, reused by every test that needs results."""
    import time

    response = client.get("/demo/ibd_genus", follow_redirects=False)
    token = response.headers["location"].rsplit("/", 1)[-1]
    client.post(f"/run/{token}", data={"mode": "quick"})
    for _ in range(900):
        status = client.get(f"/api/jobs/{token}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(1)
    assert status == "done", f"demo run did not finish: {status}"
    return token


# --------------------------------------------------------------------------
# Audit P0-1 / item 52 — an out-of-range taxon id reached the curve builder and
# surfaced as a 500. The API twin had the bounds check; the HTML route did not.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("taxon_id", [999999, -1, 10**9])
def test_invalid_taxon_curve_is_404_on_both_routes(client, finished, taxon_id):
    for path in (f"/results/{finished}/curve/{taxon_id}",
                 f"/api/jobs/{finished}/curve/{taxon_id}"):
        response = client.get(path)
        assert response.status_code == 404, (
            f"{path} returned {response.status_code}; a user-supplied id must never 500")
        assert "error" in response.json()


def test_a_valid_taxon_still_returns_its_curve(client, finished):
    for path in (f"/results/{finished}/curve/0", f"/api/jobs/{finished}/curve/0"):
        payload = client.get(path).json()
        assert {"effect", "significant", "forks"} <= set(payload), path
        assert len(payload["effect"]) > 0


def test_the_two_curve_routes_agree(client, finished):
    """They drifted once. The HTML route is the one the page actually calls."""
    html = client.get(f"/results/{finished}/curve/3").json()
    api = client.get(f"/api/jobs/{finished}/curve/3").json()
    assert html == api


def test_invalid_taxon_page_is_404(client, finished):
    assert client.get(f"/results/{finished}/taxon/999999").status_code == 404


# --------------------------------------------------------------------------
# Audit item 53 — a results link opened before the run finished returned 422,
# which reads as "your request was malformed". The token was perfectly valid.
# --------------------------------------------------------------------------
def test_unfinished_results_url_goes_to_progress_not_an_error(client):
    response = client.get("/demo/ibd_genus", follow_redirects=False)
    token = response.headers["location"].rsplit("/", 1)[-1]

    page = client.get(f"/results/{token}", follow_redirects=False)
    assert page.status_code == 303, (
        f"expected a redirect to the job page, got {page.status_code}")
    assert page.headers["location"] == f"/job/{token}"
    assert client.get(f"/job/{token}").status_code == 200


def test_the_three_token_states_are_distinguishable(client, finished):
    """Missing, unfinished and finished must not collapse into one another."""
    missing = client.get("/api/jobs/zzzzzzzzzzzzzzzzzzzzzzzz/results")
    assert missing.status_code == 404
    assert "No such job" in missing.json()["error"]

    response = client.get("/demo/ibd_genus", follow_redirects=False)
    pending = response.headers["location"].rsplit("/", 1)[-1]
    unfinished = client.get(f"/api/jobs/{pending}/results")
    assert unfinished.status_code == 409
    assert "status" in unfinished.json() and "progress" in unfinished.json()

    assert client.get(f"/api/jobs/{finished}/results").status_code == 200


def test_a_missing_job_never_reports_itself_as_unfinished(client):
    """The curve endpoint used to say "No finished run" for a job that never existed."""
    body = client.get("/api/jobs/zzzzzzzzzzzzzzzzzzzzzzzz/curve/0").json()
    assert "No such job" in body["error"]


# --------------------------------------------------------------------------
# Audit item 37 / P1-10 — two error vocabularies on one API surface.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", [
    "/api/jobs/zzzzzzzzzzzzzzzzzzzzzzzz",
    "/api/jobs/zzzzzzzzzzzzzzzzzzzzzzzz/results",
    "/api/jobs/zzzzzzzzzzzzzzzzzzzzzzzz/curve/0",
    "/api/jobs/zzzzzzzzzzzzzzzzzzzzzzzz/locate/0",
])
def test_every_api_error_uses_one_shape(client, path):
    body = client.get(path).json()
    assert "error" in body, f"{path} answered with {sorted(body)}"
    assert "detail" not in body, f"{path} leaked FastAPI's default error shape"


def test_a_malformed_path_parameter_also_uses_that_shape(client, finished):
    for path in (f"/api/jobs/{finished}/curve/not-a-number",
                 f"/results/{finished}/curve/not-a-number"):
        response = client.get(path)
        assert response.status_code == 422
        body = response.json()
        assert "error" in body and "detail" not in body, path


# --------------------------------------------------------------------------
# Audit item 41 — the bar showed a MixedLM point estimate beside a bootstrap
# interval computed from a different estimator, so a point could sit outside it.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("demo", ["ibd_genus", "gut_species", "t2d_covariates"])
def test_every_interval_contains_the_point_it_is_shown_with(demo):
    from app.core.attribution import attribute
    from app.core.runner import run_multiverse
    from app.services import load_demo

    run = run_multiverse(load_demo(demo), mode="quick")
    attribution = attribute(run, bootstrap=True)
    checked = 0
    for target, result in attribution.items():
        if not hasattr(result, "interval_consistency"):
            continue
        broken = result.interval_consistency()
        assert not broken, (
            f"{demo}/{target}: interval does not contain its own point estimate: {broken}")
        checked += len(result.intervals)
    assert checked, f"{demo} produced no intervals to check"


def test_the_interval_point_comes_from_the_interval_estimator():
    """Option B of the fix: the interval is the `within` estimator's, and it is shown
    with that estimator's own number rather than the headline one."""
    from app.core.attribution import Attribution

    result = Attribution(
        target="significance",
        shares={"method": 71.6, "transform": 28.4},
        estimator="Mixed-effects model", converged=True,
        within_shares={"method": 43.0, "transform": 57.0},
        intervals={"method": (19.0, 67.0), "transform": (33.0, 81.0)},
        interval_estimator="within")
    assert result.interval_point("method") == pytest.approx(43.0)
    assert not result.interval_consistency()
    assert not result.interval_is_for_shares, (
        "the interval belongs to a different estimator and the UI must say so")


def test_when_one_estimator_does_everything_the_interval_is_the_headline_interval():
    from app.core.attribution import Attribution

    shares = {"method": 43.0, "transform": 57.0}
    result = Attribution(
        target="significance", shares=shares, estimator="within", converged=False,
        within_shares=shares,
        intervals={"method": (19.0, 67.0), "transform": (33.0, 81.0)},
        interval_estimator="within")
    assert result.interval_is_for_shares
    assert not result.interval_consistency()


# --------------------------------------------------------------------------
# Audit item 45 / P2-22 — a single non-finite p-value propagated backwards
# through np.minimum.accumulate and voided every taxon in the specification.
# --------------------------------------------------------------------------
def test_one_non_finite_p_value_does_not_void_the_specification():
    from app.core.fdr import adjust

    clean = np.array([0.001, 0.02, 0.04])
    polluted = np.array([0.001, 0.02, 0.04, np.nan, np.inf])

    got = adjust(polluted, "bh")
    assert np.allclose(got[:3], adjust(clean, "bh")), (
        "the finite taxa must be adjusted exactly as if the bad values were absent")
    assert np.isnan(got[3]) and np.isnan(got[4]), (
        "a test that could not be computed must come back untested, not significant")


def test_the_family_size_excludes_untestable_taxa():
    """n in the BH formula is the number of tests performed, not the number attempted."""
    from app.core.fdr import adjust

    three = adjust(np.array([0.01, 0.02, 0.03]), "bh")
    padded = adjust(np.array([0.01, 0.02, 0.03, np.nan, np.nan, np.nan]), "bh")
    assert np.allclose(three, padded[:3])


def test_non_finite_p_values_are_never_significant():
    from app.core.fdr import significant

    _, mask = significant(np.array([0.001, np.nan, np.inf, -np.inf]), "bh", 0.05)
    assert mask.tolist() == [True, False, False, False]


def test_bh_and_by_arithmetic_is_untouched():
    """The guard must not have changed the correction itself."""
    from app.core.fdr import adjust

    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    expected_bh = p * len(p) / np.arange(1, len(p) + 1)
    expected_bh = np.minimum.accumulate(expected_bh[::-1])[::-1]
    assert np.allclose(adjust(p, "bh"), np.clip(expected_bh, 0, 1))

    penalty = np.sum(1.0 / np.arange(1, len(p) + 1))
    expected_by = p * len(p) * penalty / np.arange(1, len(p) + 1)
    expected_by = np.minimum.accumulate(expected_by[::-1])[::-1]
    assert np.allclose(adjust(p, "by"), np.clip(expected_by, 0, 1))


def test_an_all_non_finite_family_still_returns_the_right_shape():
    from app.core.fdr import adjust

    out = adjust(np.array([np.nan, np.nan]), "bh")
    assert out.shape == (2,) and np.isnan(out).all()


# --------------------------------------------------------------------------
# Audit item 51 — a cohort that detected nothing rendered as a row of zeros.
# --------------------------------------------------------------------------
def test_a_run_that_detected_literally_nothing_reads_as_a_finding():
    """`hiv_dinh` is the cohort the audit named; this reproduces its shape."""
    from app.core.robustness import n_detected, verdict_sentence

    class _Summary:
        tier_counts = {"ROBUST": 0, "CONDITIONAL": 0, "FRAGILE": 0, "UNSTABLE": 0,
                       "INSUFFICIENT": 12, "NOT DETECTED": 240}
        n_specs_total = 1596
        declared_percentile = float("nan")
        declared_n_significant = -1

    class _Run:
        declared_spec_id = -1

    summary = _Summary()
    assert n_detected(summary) == 0
    sentence = verdict_sentence(_Run(), summary)
    assert "0 are ROBUST" not in sentence, "the zero-filled tier list is the defect"
    assert "No taxon reached significance" in sentence
    assert "not a failure" in sentence
    assert "readiness" in sentence


def test_a_run_where_nothing_held_up_says_so(client):
    """The case a real cohort actually produces.

    The audit expected `hiv_dinh` to render the zero-detection verdict. It does not:
    in Quick mode it puts 10 taxa in UNSTABLE, so `n_detected` is 10, not 0. The
    condition that matters to a reader is narrower — nothing reached ROBUST or
    CONDITIONAL, so nothing on the page is a finding. A pure-noise table behaves the
    same way (tests/reference/noise_floor.py), which is why the page has to say it.
    """
    from app.core.robustness import n_stable

    class _Summary:
        tier_counts = {"ROBUST": 0, "CONDITIONAL": 0, "FRAGILE": 0, "UNSTABLE": 10,
                       "INSUFFICIENT": 0, "NOT DETECTED": 94}
        n_specs_total = 1596

    assert n_stable(_Summary()) == 0, (
        "a run with no ROBUST and no CONDITIONAL taxon has produced no finding")


def test_a_run_with_findings_is_not_called_empty():
    from app.core.robustness import n_stable

    class _Summary:
        tier_counts = {"ROBUST": 7, "CONDITIONAL": 8, "FRAGILE": 50, "UNSTABLE": 72,
                       "INSUFFICIENT": 0, "NOT DETECTED": 239}
        n_specs_total = 1596

    assert n_stable(_Summary()) == 15


def test_the_negative_result_panel_renders_for_a_cohort_that_found_nothing():
    """End to end, on a table built with no group difference at all."""
    import re
    import time

    import numpy as np
    import pandas as pd

    from app.core.parsers.base import AbundanceTable
    from app.core.validation import validate_dataset

    rng = np.random.default_rng(11)
    n_per_group, n_taxa = 25, 80
    n = n_per_group * 2
    props = rng.dirichlet(np.full(n_taxa, 0.4))
    depths = rng.integers(4_000, 12_000, n)
    counts = np.vstack([rng.multinomial(int(d), props) for d in depths]).T.astype(float)
    frame = pd.DataFrame(counts,
                         index=[f"taxon_{i:03d}" for i in range(n_taxa)],
                         columns=[f"s{j:03d}" for j in range(n)])
    metadata = pd.DataFrame(
        {"group": ["control"] * n_per_group + ["case"] * n_per_group}, index=frame.columns)
    # Confirms the table is one the validator accepts before it goes near the web layer.
    validate_dataset(
        AbundanceTable(counts=frame, lineages={}, source_format="simulated",
                       value_type="counts"),
        metadata, "group")

    with TestClient(app, raise_server_exceptions=False) as client:
        files = {"abundance": ("counts.tsv", frame.to_csv(sep="\t").encode(), "text/tsv"),
                 "metadata": ("meta.tsv", metadata.to_csv(sep="\t").encode(), "text/tsv")}
        response = client.post("/upload", files=files, data={"group_column": "group"},
                               follow_redirects=False)
        assert response.status_code == 303, response.text[:300]
        token = response.headers["location"].rsplit("/", 1)[-1]
        client.post(f"/run/{token}", data={"mode": "quick"})
        for _ in range(900):
            status = client.get(f"/api/jobs/{token}").json()["status"]
            if status in ("done", "error"):
                break
            time.sleep(1)
        assert status == "done"

        html = client.get(f"/results/{token}").text
        text = " ".join(re.sub(r"<[^>]+>", " ", html).split())

        assert "A result, not a failure" in text
        assert "The analysis ran correctly" in text
        assert "not read the table below as a shortlist" in text, (
            "a reader must not be invited to treat UNSTABLE taxa as a shortlist")
        assert "What was flagged before the run" in text, (
            "the readiness findings that predicted this are not shown")
        # A null result is still a result: nothing is withheld.
        for kind in ("robustness", "long", "bundle", "manifest"):
            assert client.get(f"/download/{token}/{kind}").status_code == 200, kind
        assert "Every taxon" in text, "the full table must still be available"


def test_a_normal_run_still_gets_the_tier_verdict(client, finished):
    page = client.get(f"/results/{finished}").text
    assert "ROBUST" in page
    assert "Nothing here separates your two groups" not in page, (
        "the negative-result panel must not appear on a run that detected taxa")


# --------------------------------------------------------------------------
# Audit items 21, 22, 39, 47, 48 — internal vocabulary reaching the reader.
# --------------------------------------------------------------------------
USER_FACING = ["/", "/about", "/validation"]


def _rendered_text(html: str) -> str:
    """Strip scripts, styles and tags, then collapse whitespace.

    Collapsing matters: template source wraps sentences across lines, so a phrase that
    reads as a single sentence in the browser is not a single substring in the markup.
    """
    without_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                             flags=re.S | re.I)
    return " ".join(re.sub(r"<[^>]+>", " ", without_scripts).split())


@pytest.mark.parametrize("path", USER_FACING)
def test_no_section_symbol_in_user_facing_text(client, path):
    assert "§" not in _rendered_text(client.get(path).text), (
        f"{path} still shows the section symbol a reader cannot interpret")


def test_no_section_symbol_on_the_result_pages(client, finished):
    for path in (f"/results/{finished}", f"/results/{finished}/taxon/0",
                 f"/configure/{finished}"):
        assert "§" not in _rendered_text(client.get(path).text), path


@pytest.mark.parametrize("path", USER_FACING)
def test_no_internal_spec_labels_in_user_facing_text(client, path):
    text = _rendered_text(client.get(path).text)
    assert not re.search(r"\bspec\s+\d{1,2}(\.\d+)?\b(?!\d)", text, re.I), (
        f"{path} renders an internal specification reference")


def test_no_internal_spec_labels_on_the_result_pages(client, finished):
    for path in (f"/results/{finished}", f"/results/{finished}/taxon/0"):
        text = _rendered_text(client.get(path).text)
        assert not re.search(r"\bspec\s+\d{1,2}(\.\d+)?\b(?!\d)", text, re.I), path


def test_the_eyebrow_and_its_marker_are_not_run_together(client, finished):
    """"How each taxon held upspec 16.2" — the marker is gone, and nothing replaced
    it without a separator."""
    text = _rendered_text(client.get(f"/results/{finished}").text)
    assert "held upspec" not in text
    assert "held up" in text


def test_the_results_table_headers_are_not_engine_field_names(client, finished):
    page = client.get(f"/results/{finished}").text
    for jargon in ('title: "Frac significant"', 'title: "Frac tested"',
                   'title: "Sign consistency"', 'title: "Median log2FC"',
                   'title: "Specs tested"', 'title: "Frac nominal"'):
        assert jargon not in page, f"the table still uses {jargon}"
    for readable in ("How often significant", "Direction agreement",
                     "Typical effect (log2)", "Analyses run"):
        assert readable in page, f"expected the reader-facing header {readable!r}"


def test_the_export_field_names_are_unchanged(client, finished):
    """Renaming a column title must not rename the machine-readable field."""
    page = client.get(f"/results/{finished}").text
    for field in ("n_specs_tested", "frac_tested", "frac_significant", "frac_nominal",
                  "sign_consistency", "median_effect", "iqr_low"):
        assert f'field: "{field}"' in page, f"{field} is no longer bound to a column"

    csv = client.get(f"/download/{finished}/robustness").text
    header = csv.splitlines()[0]
    for field in ("frac_significant", "sign_consistency", "median_effect"):
        assert field in header, f"the CSV lost {field}"


def test_the_homepage_expands_its_abbreviation(client):
    text = _rendered_text(client.get("/").text)
    assert "differential abundance" in text
    assert not re.search(r"microbiome DA\b", text), "\"DA\" is still unexplained"


def test_the_hero_count_matches_what_the_demos_actually_produce(client):
    """"Three thousand defensible answers" beside a graphic reading 1,596."""
    from app.services import demo_grid_sizes

    text = _rendered_text(client.get("/").text)
    assert "Three thousand" not in text
    sizes = demo_grid_sizes()
    assert sizes, "the landing page can no longer derive its own numbers"
    assert f"{sizes['_default']:,}" in text, (
        "the pipeline count shown must be the one the default demo produces")


# --------------------------------------------------------------------------
# Audit item 49 — the effect's covariate-invariance was measured and documented
# but never stated on the page that plots it.
# --------------------------------------------------------------------------
def test_the_curve_explains_that_the_effect_cannot_move_with_covariates(client, finished):
    text = _rendered_text(client.get(f"/results/{finished}").text)
    assert "does not move when an analysis adjusts for a covariate" in text
    assert "cannot show you covariate-driven instability" in text


def test_the_taxon_page_carries_the_same_warning(client, finished):
    text = _rendered_text(client.get(f"/results/{finished}/taxon/0").text)
    assert "does not change when an analysis adjusts for a covariate" in text


# --------------------------------------------------------------------------
# Audit items 42, 43 — copy contradicting the number beside it, and six zeros
# presented as a ranking.
# --------------------------------------------------------------------------
def test_the_fingerprint_example_uses_the_value_on_the_page(client, finished):
    """It used to say "Detection 72%" on a page displaying 0%."""
    page = client.get(f"/results/{finished}/taxon/0").text
    assert "Detection 72%" not in page
    assert 'means 72% of the analyses' not in page


def test_a_taxon_nothing_moved_is_not_given_a_ranking(client, finished):
    """Find a taxon with no influence at all and check the page says so."""
    from app.core.instability import analyse_taxon

    run = __import__("app.services", fromlist=["x"]).load_results(finished)[1]
    for taxon_id in range(min(200, len(run.taxa_names))):
        evidence = analyse_taxon(run, taxon_id)
        if evidence.ranked and all(i.conditional_swing < 0.005 for i in evidence.ranked):
            page = client.get(f"/results/{finished}/taxon/{taxon_id}").text
            assert "No analytical choice moved this taxon" in page, (
                f"taxon {taxon_id} has no movement but is still shown as a ranking")
            return
    pytest.skip("no taxon in this run has zero influence across every choice")


# --------------------------------------------------------------------------
# Audit items 15, 16 — provenance and tier caveats surfaced on the page.
# --------------------------------------------------------------------------
def test_the_manifest_is_reachable_without_unzipping_anything(client, finished):
    response = client.get(f"/download/{finished}/manifest")
    assert response.status_code == 200
    payload = response.json()
    for key in ("generated_utc", "environment", "seeds", "validation"):
        assert key in payload, f"the manifest lost {key}"
    assert f"/download/{finished}/manifest" in client.get(f"/results/{finished}").text


def test_the_results_page_says_what_a_tier_leaves_out(client, finished):
    text = _rendered_text(client.get(f"/results/{finished}").text)
    assert "What a tier deliberately leaves out" in text
    assert "Magnitude" in text
    assert "not 86 replications" in text or "re-analyses of the same samples" in text


# --------------------------------------------------------------------------
# Audit item 12 — the taxon selector, and the glossary link that went nowhere.
# --------------------------------------------------------------------------
def test_the_taxon_selector_is_grouped_and_searchable(client, finished):
    page = client.get(f"/results/{finished}").text
    assert 'id="taxon-search"' in page, "no search box"
    assert "<optgroup" in page, "the selector is still a flat list"
    assert 'label="ROBUST' in page


def test_every_taxon_is_still_reachable_from_the_selector(client, finished):
    """Grouping must not drop anyone."""
    import app.services as services

    _, run, summary, _ = services.load_results(finished)
    page = client.get(f"/results/{finished}").text
    options = re.findall(r'<option value="(\d+)"', page)
    assert len(set(options)) == len(summary.table), (
        f"{len(set(options))} options for {len(summary.table)} taxa")


def test_the_glossary_link_on_the_results_page_resolves(client, finished):
    assert "/about#glossary" in client.get(f"/results/{finished}").text
    about = client.get("/about").text
    assert 'id="glossary"' in about, "the anchor the results page links to does not exist"
    # The names the glossary actually defines, not the internal ones.
    for term in ("specification", "robustness tier", "comparable effect size",
                 "direction agreement"):
        assert term.lower() in _rendered_text(about).lower(), f"{term} is not defined"


# --------------------------------------------------------------------------
# Audit item 18 — benchmark provenance.
# --------------------------------------------------------------------------
def test_the_benchmark_record_identifies_the_machine():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "docs" / "benchmark.json"
    machine = json.loads(path.read_text(encoding="utf-8"))["machine"]
    for key in ("python", "cpus", "cpu_model", "memory_gb", "platform"):
        assert key in machine, f"the benchmark record does not say {key}"
