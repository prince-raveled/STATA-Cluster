"""The evidence the interface cites must match the experiment that produced it.

`app/core/evidence.py` hard-codes the tier-replication numbers so the running app never
depends on a file under `docs/`. That is a correctness hazard: the constants can drift
away from the experiment silently, and a stale number beside a tier is worse than no
number at all. These tests re-read the experiment's own output and fail on drift.
"""
from __future__ import annotations

import json
import os

import pytest

from app.core.evidence import (
    CROSS_STUDY,
    EMPIRICAL,
    EXPLORATORY,
    GRADE_BLURB,
    GRADE_LABEL,
    NEARING_STUDY,
    PUBLISHED_STUDY,
    TIER_CAVEATS,
    TIER_REPLICATION,
    TIER_VALIDATION,
    matrix_rows,
    tier_evidence,
    tier_sentence,
)
from app.core.robustness import TIER_ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "docs", "tier_validation.json")
PUBLISHED_RECORD = os.path.join(ROOT, "docs", "published_findings_study.json")
NEARING_RECORD = os.path.join(ROOT, "docs", "nearing_study.json")


def _record():
    if not os.path.exists(RECORD):
        pytest.skip(f"{RECORD} not present; run tests/reference/tier_validation.py")
    with open(RECORD, encoding="utf-8") as handle:
        return json.load(handle)


# --- internal consistency ---------------------------------------------------
def test_every_tier_the_app_can_show_has_an_entry_or_none():
    """A tier with no evidence must be absent, not silently defaulted to something."""
    for tier in TIER_ORDER:
        facts = tier_evidence(tier)
        if facts:
            assert set(facts) >= {"rate", "ci", "n", "cohorts"}
            assert 0.0 <= facts["rate"] <= 1.0
            low, high = facts["ci"]
            assert low <= facts["rate"] <= high, f"{tier}: rate outside its own interval"


def test_insufficient_has_no_replication_claim():
    """INSUFFICIENT means 'not enough specifications to judge' — it predicts nothing."""
    assert tier_evidence("INSUFFICIENT") == {}
    assert tier_sentence("INSUFFICIENT") == ""


def test_tier_sentence_quotes_the_recorded_numbers():
    sentence = tier_sentence("ROBUST")
    assert "90%" in sentence and "ROBUST" in sentence
    assert str(TIER_VALIDATION["n_cohorts"]) in sentence


def test_grades_are_described_wherever_they_are_labelled():
    for grade in GRADE_LABEL:
        assert grade in GRADE_BLURB, f"{grade} has a label but no explanation"


def test_attribution_is_not_claimed_to_be_validated():
    """§17 has no external reference and no held-out experiment. It must say so."""
    rows = {row["name"]: row for row in matrix_rows()}
    attribution = rows["Choice attribution"]
    assert attribution["grade"] == EXPLORATORY
    assert not attribution["reference"]
    assert not attribution["empirical"]


def test_tiers_are_the_component_marked_empirical():
    rows = {row["name"]: row for row in matrix_rows()}
    assert rows["Robustness tiers"]["grade"] == EMPIRICAL
    assert rows["Robustness tiers"]["empirical"]


def test_no_component_claims_evidence_it_does_not_describe():
    """A grade is a promise about the cell beneath it."""
    for row in matrix_rows():
        if row["grade"] == EMPIRICAL:
            assert row["empirical"], f"{row['name']} graded empirical with no evidence"
        if row["grade"] == "reference":
            assert row["reference"], f"{row['name']} graded reference with no evidence"
        assert row["sources"], f"{row['name']} cites no source"


# --- agreement with the experiment -----------------------------------------
def test_tier_rates_match_the_validation_record():
    record = _record()
    by_tier = {
        row["tier"]: row
        for row in record["definitions"]["replicated_majority"]["by_tier"]
    }
    for tier, facts in TIER_REPLICATION.items():
        assert tier in by_tier, f"{tier} is cited but absent from the experiment"
        actual = by_tier[tier]
        assert abs(actual["replication_rate"] - facts["rate"]) < 0.015, (
            f"{tier}: evidence.py says {facts['rate']:.3f}, the experiment produced "
            f"{actual['replication_rate']:.3f}")
        assert actual["n_taxa"] == facts["n"], f"{tier}: taxon count drifted"
        assert actual["n_cohorts"] == facts["cohorts"], f"{tier}: cohort count drifted"


def test_headline_statistics_match_the_validation_record():
    record = _record()
    block = record["definitions"]["replicated_majority"]["discrimination"]
    assert abs(block["auc"] - TIER_VALIDATION["auc"]) < 0.01
    assert abs(block["risk_ratio"] - TIER_VALIDATION["risk_ratio_robust"]) < 0.1
    assert record["n_cohorts"] == TIER_VALIDATION["n_cohorts"]
    assert record["n_observations"] == TIER_VALIDATION["n_observations"]


def test_the_recorded_ordering_is_still_monotone():
    """If a rerun breaks the ordering, the tiers stop meaning what the UI says."""
    record = _record()
    by_tier = {
        row["tier"]: row["replication_rate"]
        for row in record["definitions"]["replicated_majority"]["by_tier"]
    }
    ordered = [by_tier[t] for t in ("ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE")
               if t in by_tier]
    assert len(ordered) >= 3
    assert all(a >= b for a, b in zip(ordered, ordered[1:], strict=False)), ordered


def test_the_null_benchmark_is_near_zero():
    """Permuted labels must not replicate. If they do, the definition is broken."""
    record = _record()
    null_rate = record["definitions"]["replicated_majority"]["null_rate"]
    assert null_rate is not None, "the null benchmark did not run"
    assert null_rate < 0.05, f"label-permuted data replicated {null_rate:.1%} of the time"


# --- the published study, and replication across studies (SPEC §24.8) ----------
def _published():
    if not os.path.exists(PUBLISHED_RECORD):
        pytest.skip(f"{PUBLISHED_RECORD} not present; "
                    "run tests/reference/published_findings_study.py")
    with open(PUBLISHED_RECORD, encoding="utf-8") as handle:
        return json.load(handle)


def test_reproduction_numbers_match_the_published_study_record():
    record = _published()
    cohorts = record["cohorts"]
    level2 = [c["level2_published_pipeline"] for c in cohorts]
    level1 = [c["level1_implementation"] for c in cohorts]
    assert len(cohorts) == PUBLISHED_STUDY["n_cohorts"]
    assert len(record["refused_by_validation"]) == PUBLISHED_STUDY["n_refused"]
    assert not record["failures"], record["failures"]
    assert sum(c["n_control"] + c["n_case"] for c in cohorts) == PUBLISHED_STUDY["n_samples"]
    assert sum(x["n_sig_paper"] for x in level2) == PUBLISHED_STUDY["paper_significant"]
    assert sum(x["n_sig_both"] for x in level2) == PUBLISHED_STUDY["recovered"]
    extra = sum(x["n_sig_matched"] - x["n_sig_both"] for x in level2)
    assert extra == PUBLISHED_STUDY["extra"]
    exact = sum(x["n_sig_paper"] == x["n_sig_matched"] == x["n_sig_both"] for x in level2)
    assert exact == PUBLISHED_STUDY["n_exact"]
    identical = sum(x["agreement_same_matrix"] == 1.0 for x in level1)
    assert identical == PUBLISHED_STUDY["same_matrix_identical"]
    worst = max(x["max_abs_diff_p_vs_scipy_mannwhitney"] for x in level1)
    assert worst <= PUBLISHED_STUDY["max_p_difference"]


def test_shared_response_agreement_matches_the_record():
    by_tier = _published()["claims"]["s3_direction_by_tier"]
    for tier, (agree, total) in PUBLISHED_STUDY["shared_response_direction"].items():
        assert by_tier[tier] == {"n": total, "direction_agrees": agree}, tier


def test_cross_study_numbers_match_the_record():
    record = _published()["replication"]
    assert record["n_pairs"] == CROSS_STUDY["n_pairs"]
    assert record["n_genera"] == CROSS_STUDY["n_genera"]
    groups = record["groups"]
    for key, name in (("stable", "paper-significant and ROBUST or CONDITIONAL"),
                      ("shaky", "paper-significant and FRAGILE or UNSTABLE")):
        quoted, measured = CROSS_STUDY[key], groups[name]
        assert quoted["n"] == measured["n_pairs"], key
        assert abs(quoted["rate"] - measured["significant"]["rate"]) < 0.0015, key
        assert abs(quoted["direction"] - measured["direction"]["rate"]) < 0.0015, key
        for q, m in zip(quoted["ci"], measured["significant"]["ci95"], strict=True):
            assert abs(q - m) < 0.0015, key
    contrast = record["stable_vs_shaky_among_paper_significant"]["significant"]
    assert abs(CROSS_STUDY["difference"] - contrast["difference"]) < 0.0015
    for q, m in zip(CROSS_STUDY["difference_ci"], contrast["difference_ci95_cluster_bootstrap"],
                    strict=True):
        assert abs(q - m) < 0.0015


def test_the_interface_does_not_sell_within_study_replication_as_external():
    """The tier rates were measured on held-out halves of one study. Every sentence
    that quotes them must say so, and the caveats must carry the cross-study result."""
    for tier in TIER_REPLICATION:
        assert "same study" in tier_sentence(tier), tier
    assert any("Across independent studies" in caveat for caveat in TIER_CAVEATS)
    rows = {r["name"]: r for r in matrix_rows()}
    assert "Across studies" in rows["Robustness tiers"]["empirical"]
    assert rows["Reproducing a published analysis"]["reference"]


# --- the methods against a published benchmark (SPEC §24.9) ----------------------
def _nearing_rows():
    if not os.path.exists(NEARING_RECORD):
        pytest.skip(f"{NEARING_RECORD} not present; run tests/reference/nearing_study.py")
    with open(NEARING_RECORD, encoding="utf-8") as handle:
        return json.load(handle)["rows"]


#: The interface's method names -> the record's, and which count is compared.
_NEARING_COUNTS = {
    "Wilcoxon (rarefied)": ("Wilcoxon (rare)", "microverse"),
    "Welch's t-test (rarefied)": ("t-test (rare)", "microverse"),
    "Wilcoxon (CLR)": ("Wilcoxon (CLR)", "microverse_test_on_paper_clr"),
    "ALDEx2": ("ALDEx2", "microverse"),
    "DESeq2": ("DESeq2", "microverse"),
}


def _same(rows, method, column):
    chosen = [r for r in rows if r["method"] == method]
    return sum(r[column] == r["published"] for r in chosen), len(chosen)


def test_benchmark_numbers_match_the_record():
    rows = _nearing_rows()
    assert len(rows) == NEARING_STUDY["n_comparisons"]
    assert len({r["dataset"] for r in rows}) == NEARING_STUDY["n_datasets"]
    rebuilt = sum(r["input_features"] == r["paper_features"] for r in rows)
    assert (rebuilt, len(rows)) == NEARING_STUDY["inputs_rebuilt"]
    for entry in NEARING_STUDY["methods"]:
        method, column = _NEARING_COUNTS[entry["paper"]]
        assert _same(rows, method, column) == entry["same_count"], entry["paper"]
    assert _same(rows, "Wilcoxon (CLR)", "microverse") == NEARING_STUDY["own_clr_same_count"]


def test_benchmark_method_order_matches_the_record():
    import pandas as pd
    from scipy import stats

    frame = pd.DataFrame(_nearing_rows())
    exact = total = 0
    for _, sub in frame.groupby("variant"):
        published = sub.pivot(index="dataset", columns="method", values="published")
        mine = sub.pivot(index="dataset", columns="method", values="microverse")
        for dataset in published.dropna().index.intersection(mine.dropna().index):
            if published.loc[dataset].nunique() > 1 and mine.loc[dataset].nunique() > 1:
                total += 1
                rho = stats.spearmanr(published.loc[dataset], mine.loc[dataset]).statistic
                exact += bool(abs(rho - 1.0) < 1e-9)
    assert (exact, total) == NEARING_STUDY["method_order_exact"]


def test_the_exact_methods_really_are_exact():
    """The two rarefied tests, and Wilcoxon given the paper's CLR, share everything with
    the paper's analysis but the code. Anything short of every dataset is a defect."""
    for entry in NEARING_STUDY["methods"][:3]:
        agree, total = entry["same_count"]
        assert agree == total, entry["paper"]
