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
    EMPIRICAL,
    EXPLORATORY,
    GRADE_BLURB,
    GRADE_LABEL,
    TIER_REPLICATION,
    TIER_VALIDATION,
    matrix_rows,
    tier_evidence,
    tier_sentence,
)
from app.core.robustness import TIER_ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "docs", "tier_validation.json")


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
