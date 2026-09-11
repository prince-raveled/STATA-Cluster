"""Robustness metrics and denominators — SPEC §15, §16."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.robustness import (
    TIER_ORDER,
    compute_robustness,
    locate_declared,
    verdict_sentence,
)
from app.core.runner import run_multiverse


@pytest.fixture(scope="module")
def run(dataset):
    return run_multiverse(dataset, mode="quick")


@pytest.fixture(scope="module")
def summary(run):
    return compute_robustness(run)


def test_denominators_use_n_specs_tested_not_total(run, summary):
    """§15: frac_significant and sign_consistency divide by n_specs_tested."""
    long = run.long
    for _, row in summary.table.sample(12, random_state=0).iterrows():
        rows = long[long["taxon"] == row["taxon_id"]]
        assert row["n_specs_tested"] == len(rows)
        assert np.isclose(row["frac_significant"], rows["significant"].mean())
        assert np.isclose(row["frac_nominal"], (rows["p_raw"] < 0.05).mean())


def test_frac_tested_is_relative_to_eligible_specifications(run, summary):
    """A taxon can only be tested in specifications at its own rank."""
    ranks = pd.Series([s.rank for s in run.specs])
    for rank, count in ranks.value_counts().items():
        subset = summary.table[summary.table["rank"] == rank]
        assert (subset["n_specs_eligible"] == count).all()
        assert (subset["frac_tested"] <= 1.0 + 1e-9).all()
        assert (subset["frac_tested"] > 0).all()


def test_frac_tested_never_exceeds_one(summary):
    partial = summary.table[summary.table["frac_tested"] < 1.0]
    assert (partial["n_specs_tested"] < partial["n_specs_eligible"]).all()
    full = summary.table[summary.table["frac_tested"] >= 1.0]
    assert (full["n_specs_tested"] == full["n_specs_eligible"]).all()


def test_a_rare_taxon_is_filtered_out_of_some_specifications(synthetic):
    """§15: a taxon testable in only part of the grid must report that, not hide it."""
    from app.core.parsers.base import AbundanceTable
    from app.core.validation import validate_dataset

    counts = synthetic["table"].counts.copy()
    # Present in 3 of 40 samples: survives a 0% filter, dropped by 10% and 20%.
    rare = np.zeros(counts.shape[1], dtype=int)
    rare[[0, 1, 21]] = 40
    counts.loc["Taxon_rare"] = rare
    table = AbundanceTable(counts=counts, value_type="counts")
    dataset = validate_dataset(table, synthetic["metadata"], "group")

    local = compute_robustness(run_multiverse(dataset, mode="quick"))
    row = local.table.set_index("taxon").loc["Taxon_rare"]
    assert row["frac_tested"] < 1.0
    assert row["n_specs_tested"] < row["n_specs_eligible"]


def test_sign_consistency_is_the_modal_sign_share(run, summary):
    long = run.long
    for _, row in summary.table.head(20).iterrows():
        effects = long.loc[long["taxon"] == row["taxon_id"], "effect_h"].to_numpy()
        expected = max((effects > 0).sum(), (effects < 0).sum()) / effects.size
        assert np.isclose(row["sign_consistency"], expected)
        assert 0.0 <= row["sign_consistency"] <= 1.0


def test_iqr_brackets_the_median(summary):
    table = summary.table
    assert (table["iqr_low"] <= table["median_effect"] + 1e-6).all()
    assert (table["iqr_high"] >= table["median_effect"] - 1e-6).all()


def test_every_taxon_receives_exactly_one_tier(summary):
    assert set(summary.table["robustness_tier"]) <= set(TIER_ORDER)
    assert summary.table["robustness_tier"].notna().all()
    assert sum(summary.tier_counts.values()) == len(summary.table)


def test_table_is_ordered_best_tier_first(summary):
    order = [TIER_ORDER.index(t) for t in summary.table["robustness_tier"]]
    assert order == sorted(order)


def test_verdict_sentence_reports_the_counts(run, summary):
    sentence = verdict_sentence(run, summary)
    assert str(summary.tier_counts["ROBUST"]) in sentence
    assert f"{summary.n_specs_total:,}" in sentence
    assert sentence.endswith(".")


def test_spec_summary_covers_every_specification(run, summary):
    assert len(summary.spec_summary) == len(run.specs)
    assert (summary.spec_summary["n_significant"] <= summary.spec_summary["n_taxa_tested"]).all()


# --- §16.4 locate my result ----------------------------------------------
def test_locate_declared_returns_a_percentile(dataset):
    from app.core.models import Specification

    declared = Specification("none", None, "input", 0.10, "clr", "wilcoxon", "bh", 0.05)
    run = run_multiverse(dataset, mode="quick", declared=declared)
    assert run.declared_spec_id >= 0

    summary = compute_robustness(run)
    assert 0.0 <= summary.declared_percentile <= 100.0

    taxon_id = int(summary.table.iloc[0]["taxon_id"])
    located = locate_declared(run, taxon_id)
    assert located["tested"] is True
    assert 0.0 <= located["percentile"] <= 100.0
    assert located["rank"] <= located["n_specs_tested"]


def test_locate_declared_is_empty_without_a_declaration(run):
    assert locate_declared(run, int(run.long["taxon"].iloc[0])) == {}


def test_no_best_specification_is_exposed(summary):
    """SPEC §18: there is no 'best specification' anywhere in the results object."""
    columns = set(summary.spec_summary.columns) | set(summary.table.columns)
    assert not {"best", "best_spec", "recommended", "optimal"} & columns


def test_percentiles_read_as_ordinals():
    """'51th percentile' is the kind of thing a reviewer notices."""
    from app.core.models import ordinal

    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 51, 100)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "51st", "100th"
    ]
    assert ordinal(float("nan")) == "-"
    assert ordinal(None) == "-"


def test_verdict_uses_an_ordinal_for_the_declared_percentile(dataset):
    from app.core.models import Specification
    from app.core.runner import run_multiverse

    declared = Specification("none", None, "input", 0.10, "clr", "wilcoxon", "bh", 0.05)
    run = run_multiverse(dataset, mode="quick", declared=declared)
    summary = compute_robustness(run)
    sentence = verdict_sentence(run, summary)
    assert "th percentile" in sentence or "st percentile" in sentence \
        or "nd percentile" in sentence or "rd percentile" in sentence
    for bad in ("1th", "2th", "3th", "21th", "51th"):
        assert bad not in sentence
