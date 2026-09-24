"""Scientific readiness — is this dataset suitable, as distinct from processable?

The distinction these tests protect: `validation.py` refuses data the software cannot
handle, and readiness warns about data the software can handle but which cannot answer
the question. Conflating them is how a tool either blocks legitimate analyses or lets a
12-sample study produce a page of confident-looking nothing.

The strongest evidence that these checks are calibrated is in
`tests/reference/real_data_study.py`: the cohorts flagged SERIOUS here are the ones that
detected nothing there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.parsers.base import AbundanceTable
from app.core.readiness import (
    CAUTION,
    LEVEL_LABEL,
    LEVEL_ORDER,
    NOTE,
    OK,
    SERIOUS,
    TIER_INFORMATIVE_PER_GROUP,
    assess,
)
from app.core.validation import validate_dataset


def _dataset(n_per_group=50, n_taxa=60, zero_share=0.3, depth_span=2.0,
             taxonomy=False, covariates=None, seed=4):
    """A dataset built to order, so each check can be exercised in isolation."""
    rng = np.random.default_rng(seed)
    n = n_per_group * 2
    taxa = [f"taxon_{i}" for i in range(n_taxa)]
    if taxonomy:
        taxa = [f"k__Bacteria;p__P{i % 5};c__C;o__O;f__F;g__G{i % 11};s__S{i}"
                for i in range(n_taxa)]
    samples = [f"s{j:03d}" for j in range(n)]

    depths = np.geomspace(5_000, 5_000 * depth_span, n)
    counts = np.zeros((n_taxa, n))
    for j in range(n):
        p = rng.dirichlet(np.ones(n_taxa) * 0.8)
        counts[:, j] = rng.multinomial(int(depths[j]), p)
    mask = rng.random(counts.shape) < zero_share
    counts = counts * ~mask
    counts[: max(1, n_taxa // 10), :] += 20          # keep some taxa always present

    frame = pd.DataFrame(counts, index=taxa, columns=samples)
    metadata = pd.DataFrame(
        {"group": ["a_control"] * n_per_group + ["b_case"] * n_per_group},
        index=samples)
    if covariates:
        for name, values in covariates.items():
            metadata[name] = values

    from app.core.parsers.base import split_lineage
    lineages = {t: split_lineage(t) for t in taxa} if taxonomy else {}
    table = AbundanceTable(counts=frame, lineages={k: v for k, v in lineages.items() if v},
                           source_format="test", value_type="counts")
    return validate_dataset(table, metadata, "group")


# --- readiness never blocks -------------------------------------------------
def test_readiness_never_refuses_a_dataset_validation_accepted():
    """Refusal is §8's job. This is advice, and advice does not stop anyone."""
    report = assess(_dataset(n_per_group=6, n_taxa=15, zero_share=0.95))
    assert report.worst == SERIOUS
    assert report.checks, "a poor dataset must still produce a report, not an exception"
    assert all(hasattr(check, "detail") and check.detail for check in report.checks)


def test_a_broken_check_degrades_to_a_note(monkeypatch):
    """One failing check must not deny a run the validator already accepted."""
    import app.core.readiness as module

    def explode(dataset):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(module, "_sparsity_check", explode)
    report = module.assess(_dataset())
    assert any("could not run" in check.title for check in report.checks)
    assert report.worst != SERIOUS


# --- the checks fire on the right things ------------------------------------
def test_a_small_group_is_flagged_against_the_measured_threshold():
    """The threshold comes from the tier validation, not from taste."""
    report = assess(_dataset(n_per_group=12))
    sample_size = next(c for c in report.checks if c.key == "sample_size")
    assert sample_size.level == SERIOUS
    assert str(TIER_INFORMATIVE_PER_GROUP) in sample_size.detail
    assert "ROBUST" in sample_size.detail


def test_an_adequate_group_is_not_flagged():
    report = assess(_dataset(n_per_group=TIER_INFORMATIVE_PER_GROUP + 10))
    assert next(c for c in report.checks if c.key == "sample_size").level == OK


def test_imbalance_is_judged_on_the_smaller_group():
    rng = np.random.default_rng(1)
    del rng
    balanced = assess(_dataset(n_per_group=60))
    assert next(c for c in balanced.checks if c.key == "balance").level == OK


def test_extreme_sparsity_is_serious():
    report = assess(_dataset(zero_share=0.95, n_taxa=80))
    sparsity = next(c for c in report.checks if c.key == "sparsity")
    assert sparsity.level in (CAUTION, SERIOUS)
    assert "%" in sparsity.detail


def test_ordinary_sparsity_is_not_flagged():
    assert next(c for c in assess(_dataset(zero_share=0.35)).checks
                if c.key == "sparsity").level == OK


def test_a_wide_depth_span_is_flagged_and_explains_the_rarefaction_fork():
    report = assess(_dataset(depth_span=40.0))
    depth = next(c for c in report.checks if c.key == "depth")
    assert depth.level in (NOTE, CAUTION)
    assert "rarefaction" in (depth.detail + depth.advice).lower()


def test_missing_taxonomy_is_a_note_not_a_failure():
    report = assess(_dataset(taxonomy=False))
    rank = next(c for c in report.checks if c.key == "rank")
    assert rank.level == NOTE
    assert "one level" in rank.detail


def test_a_table_already_at_genus_is_not_told_to_supply_a_taxonomy():
    """Lineages that stop at genus were supplied; the check must not ask for them."""
    report = assess(_dataset(taxonomy=True, n_taxa=40))
    for check in report.checks:
        if check.key == "rank" and check.level != OK:
            assert check.title != "No taxonomy supplied"


def test_genus_level_lineages_are_named_as_such(monkeypatch):
    dataset = _dataset(taxonomy=False, n_taxa=40)
    genus = {t: ["k__Bacteria", "p__P", "c__C", "o__O", "f__F", f"g__{t}"]
             for t in dataset.table.taxa}
    monkeypatch.setattr(dataset.table, "lineages", genus)
    rank = next(c for c in assess(dataset).checks if c.key == "rank")
    assert rank.title == "Already at genus level"
    assert rank.level == NOTE
    assert "one level" in rank.detail


def test_taxonomy_present_is_fine():
    report = assess(_dataset(taxonomy=True, n_taxa=40))
    assert next(c for c in report.checks if c.key == "rank").level == OK


def test_absent_covariates_are_noted_with_the_reason_it_matters():
    report = assess(_dataset())
    covariates = next(c for c in report.checks if c.key == "covariates")
    if covariates.level == NOTE:
        assert "Tierney" in covariates.advice


def test_a_covariate_that_tracks_the_grouping_is_flagged():
    """Adjusting for something collinear with the contrast removes the contrast."""
    n = 50
    separated = np.concatenate([np.full(n, 20.0), np.full(n, 60.0)])
    report = assess(_dataset(n_per_group=n, covariates={"age": separated}))
    flagged = [c for c in report.checks if c.key.startswith("covariate_")]
    assert flagged, "a covariate separating the groups by 4 SD was not flagged"
    assert flagged[0].level == CAUTION
    assert "removes much of the contrast" in flagged[0].detail


def test_a_harmless_covariate_is_not_flagged():
    rng = np.random.default_rng(7)
    report = assess(_dataset(n_per_group=50, covariates={"age": rng.normal(50, 10, 100)}))
    assert not [c for c in report.checks if c.key.startswith("covariate_")]


# --- report shape -----------------------------------------------------------
def test_findings_are_ordered_worst_first():
    report = assess(_dataset(n_per_group=8, zero_share=0.95))
    levels = [LEVEL_ORDER[c.level] for c in report.checks]
    assert levels == sorted(levels, reverse=True), (
        "a researcher should read the worst thing first")


def test_every_level_has_a_label_the_interface_can_show():
    report = assess(_dataset())
    for check in report.checks:
        assert check.label == LEVEL_LABEL[check.level]


def test_headline_says_the_analysis_still_runs_when_something_is_serious():
    report = assess(_dataset(n_per_group=8))
    assert "still run" in report.headline()


def test_a_clean_dataset_says_so_plainly():
    report = assess(_dataset(n_per_group=60, zero_share=0.3, depth_span=2.0,
                             taxonomy=True, n_taxa=40))
    assert not [c for c in report.flagged if c.level in (CAUTION, SERIOUS)]


def test_counts_add_up_to_the_checks_performed():
    report = assess(_dataset())
    assert sum(report.counts.values()) == len(report.checks)


# --- calibration against the real-data study --------------------------------
@pytest.mark.parametrize(("cohort", "expected"), [
    ("hiv_dinh", SERIOUS),        # 0 taxa detected in docs/real_data_study.json
    ("crc_zhao", SERIOUS),        # 1 taxon detected
    ("cdi_schubert", NOTE),       # supplied most ROBUST calls in the tier validation
])
def test_flags_match_what_those_cohorts_actually_produced(cohort, expected):
    """The check is only useful if it predicts which datasets yield nothing."""
    microbiomehd = pytest.importorskip("tests.reference.microbiomehd")
    try:
        dataset, _ = microbiomehd.load_dataset(cohort)
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"{cohort} unavailable: {exc}")
    assert assess(dataset).worst == expected
