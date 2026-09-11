"""Per-taxon instability — "which choice moved my answer?"

The number this module produces goes straight into a sentence a researcher reads and
acts on ("the result depends mainly on rarefaction depth"). So the tests here are mostly
about it being *right* in cases where the answer is known by construction, and about it
refusing to overclaim when it is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.instability import (
    FINGERPRINT_LABELS,
    NOTABLE_SWING,
    ForkInfluence,
    analyse_taxon,
    stability_fingerprint,
)
from app.core.robustness import compute_robustness
from app.core.runner import run_multiverse


@pytest.fixture(scope="module")
def run(dataset):
    return run_multiverse(dataset, mode="quick")


@pytest.fixture(scope="module")
def summary(run):
    return compute_robustness(run)


# --- the swing measures what it claims to measure ---------------------------
def test_a_choice_that_changes_nothing_has_zero_swing(run, summary):
    """A taxon significant everywhere cannot have a choice that flips it."""
    always = summary.table[summary.table["frac_significant"] >= 0.999]
    if always.empty:
        pytest.skip("no taxon is significant in every specification here")
    evidence = analyse_taxon(run, int(always.iloc[0]["taxon_id"]))
    for influence in evidence.influences:
        assert influence.conditional_swing == pytest.approx(0.0, abs=1e-9), (
            f"{influence.fork} claims to move a taxon that never changes")


def test_swings_are_shares_between_zero_and_one(run, summary):
    for taxon_id in summary.table["taxon_id"].head(12):
        evidence = analyse_taxon(run, int(taxon_id))
        for influence in evidence.influences:
            assert 0.0 <= influence.conditional_swing <= 1.0
            assert 0.0 <= influence.marginal_swing <= 1.0
            assert 0.0 <= influence.direction_swing <= 1.0


def test_level_rates_are_computed_over_the_analyses_that_ran(run, summary):
    """Each level's count must add up to the taxon's tested total."""
    taxon_id = int(summary.table["taxon_id"].iloc[0])
    evidence = analyse_taxon(run, taxon_id)
    for influence in evidence.influences:
        counted = sum(level["n"] for level in influence.levels)
        assert counted <= evidence.n_tested, (
            f"{influence.fork}: levels sum to more analyses than were run")


def test_conditional_and_marginal_are_both_reported(run, summary):
    """They answer different questions and the interface shows both."""
    taxon_id = int(summary.table["taxon_id"].iloc[0])
    evidence = analyse_taxon(run, taxon_id)
    assert evidence.influences
    for influence in evidence.influences:
        assert isinstance(influence.conditional_swing, float)
        assert isinstance(influence.marginal_swing, float)


# --- a case where the right answer is known -------------------------------
def _synthetic_run(driver: str):
    """Hand-built results where exactly one choice decides significance."""
    rows, specs = [], []
    spec_id = 0
    # Three inert prevalence levels keep the grid above the ten-analysis floor while
    # changing nothing, so the deciding choice stays the only thing that matters.
    for prevalence in (0.0, 0.05, 0.10):
        for rarefaction in ("none", "1000"):
            for transform in ("raw", "clr"):
                for method in ("wilcoxon", "ttest"):
                    decisive = {"rarefaction": rarefaction, "transform": transform,
                                "method": method}[driver]
                    significant = decisive in ("none", "raw", "wilcoxon")
                    specs.append({
                        "spec_id": spec_id, "rarefaction": rarefaction,
                        "rare_seed": None, "rank": "input", "prev_filter": prevalence,
                        "transform": transform, "method": method, "fdr_method": "bh",
                        "fdr_threshold": 0.05, "covariates": (),
                    })
                    rows.append({
                        "spec_id": spec_id, "taxon": 0,
                        "p_raw": 0.001 if significant else 0.4,
                        "p_adj": 0.01 if significant else 0.6,
                        "significant": significant, "effect_h": 1.5, "effect_n": 1.5,
                    })
                    spec_id += 1

    class _Spec:
        def __init__(self, row):
            self.__dict__.update(row)
            self.rarefaction_label = row["rarefaction"]

    class _Run:
        pass

    # Built outside the class body: a class body cannot see the enclosing function's
    # locals from inside a comprehension, and `specs` would shadow itself.
    run = _Run()
    run.long = pd.DataFrame(rows)
    run.specs_frame = pd.DataFrame(specs)
    run.specs = [_Spec(row) for row in specs]
    run.taxa_names = ["taxon_0"]
    run.taxa_display = ["Taxon 0"]
    return run


@pytest.mark.parametrize("driver", ["rarefaction", "transform", "method"])
def test_the_choice_that_decides_the_answer_is_ranked_first(driver):
    """Built so exactly one choice determines significance. It must come top."""
    evidence = analyse_taxon(_synthetic_run(driver), 0)
    assert evidence.ranked, "no influences computed"
    assert evidence.ranked[0].fork == driver, (
        f"{driver} decides the answer but {evidence.ranked[0].fork} was ranked first")
    assert evidence.ranked[0].conditional_swing == pytest.approx(1.0), (
        "a choice that fully determines significance must swing by 100%")


@pytest.mark.parametrize("driver", ["rarefaction", "transform", "method"])
def test_the_headline_names_the_deciding_choice(driver):
    evidence = analyse_taxon(_synthetic_run(driver), 0)
    headline = evidence.headline()
    assert evidence.ranked[0].label.lower().split()[0] in headline.lower(), headline


def test_the_headline_does_not_mangle_an_acronym():
    """FORK_LABELS holds 'DA method'; lowercasing it whole reads as a typo."""
    assert "da method" not in analyse_taxon(_synthetic_run("method"), 0).headline()


# --- refusing to overclaim --------------------------------------------------
def test_a_taxon_tested_nowhere_returns_an_explanation_not_a_crash(run):
    class _Empty:
        long = pd.DataFrame(columns=["taxon", "spec_id", "significant", "effect_h"])
        specs_frame = pd.DataFrame(columns=["spec_id"])
        specs: list = []
        taxa_names = ["ghost"]
        taxa_display = ["Ghost taxon"]

    evidence = analyse_taxon(_Empty(), 0)
    assert evidence.n_tested == 0
    assert evidence.notes
    assert "filtered out" in evidence.notes[0]


def test_too_few_analyses_produces_no_stability_claim():
    run = _synthetic_run("method")
    run.long = run.long.head(6)          # below the 10-analysis floor
    assert "Too few" in analyse_taxon(run, 0).headline()


def test_a_result_with_no_dominant_choice_says_so(run, summary):
    """When no choice explains it, the headline must not name one anyway."""
    middling = summary.table[
        (summary.table["frac_significant"] > 0.2)
        & (summary.table["frac_significant"] < 0.8)]
    if middling.empty:
        pytest.skip("no taxon in the middling range here")
    for taxon_id in middling["taxon_id"].head(8):
        evidence = analyse_taxon(run, int(taxon_id))
        leader = evidence.leading
        headline = evidence.headline()
        if leader is None or leader.conditional_swing < 0.10:
            assert "depends mainly on" not in headline, headline


def test_method_disagreement_is_flagged_when_it_is_large(run, summary):
    for taxon_id in summary.table["taxon_id"].head(25):
        evidence = analyse_taxon(run, int(taxon_id))
        if not evidence.method_agreement:
            continue
        rates = [m["frac_significant"] for m in evidence.method_agreement.values()]
        if max(rates) - min(rates) >= NOTABLE_SWING:
            assert any("disagree" in note for note in evidence.notes), (
                "tests disagree by more than the notable threshold and nothing said so")


# --- the fingerprint --------------------------------------------------------
def test_fingerprint_reports_every_labelled_dimension(summary):
    row = summary.table.iloc[0]
    fingerprint = stability_fingerprint(row)
    assert set(fingerprint) == set(FINGERPRINT_LABELS)
    for key, value in fingerprint.items():
        assert np.isnan(value) or 0.0 <= value <= 1.0, f"{key} = {value}"


def test_fingerprint_keeps_direction_and_magnitude_apart(summary):
    """The point of the fingerprint is that one label cannot say both."""
    row = summary.table.iloc[0]
    fingerprint = stability_fingerprint(row)
    assert fingerprint["direction"] == pytest.approx(float(row["sign_consistency"]))
    assert fingerprint["detection"] == pytest.approx(float(row["frac_significant"]))


def test_a_zero_effect_is_not_called_magnitude_stable():
    """No effect means no magnitude to be stable about — it must not score well."""
    row = pd.Series({"frac_significant": 0.5, "sign_consistency": 0.5,
                     "median_effect": 0.0, "iqr_low": -0.2, "iqr_high": 0.2,
                     "frac_tested": 1.0})
    assert stability_fingerprint(row)["effect"] == 0.0


def test_a_tight_spread_around_a_large_effect_scores_high():
    row = pd.Series({"frac_significant": 0.9, "sign_consistency": 1.0,
                     "median_effect": 2.0, "iqr_low": 1.9, "iqr_high": 2.1,
                     "frac_tested": 1.0})
    assert stability_fingerprint(row)["effect"] > 0.9


def test_influence_sentence_is_readable_without_the_table():
    influence = ForkInfluence(
        fork="rarefaction", label="Rarefaction depth", n_levels=2,
        conditional_swing=0.8, marginal_swing=0.7, direction_swing=0.0,
        effect_swing=0.3,
        levels=[{"label": "no rarefaction", "frac_significant": 0.95, "n": 10,
                 "frac_positive": 1.0, "median_effect": 1.2},
                {"label": "1000", "frac_significant": 0.12, "n": 10,
                 "frac_positive": 1.0, "median_effect": 0.9}])
    sentence = influence.sentence()
    assert "95%" in sentence and "12%" in sentence
    assert "no rarefaction" in sentence
