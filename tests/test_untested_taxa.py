"""A taxon filtered out of every specification must still be reported — SPEC §16.2.

Found by driving the ten dataset x mode combinations through the server and adding up
what came back: the tier counts did not reach the number of taxa the person uploaded.
Between 1% and 16% of the input was missing, worst in covariate mode, where 61 of 380
taxa in `ibd_genus` had no row, no tier and no mention anywhere.

The cause was that the per-taxon frame is built by grouping the long results, so a taxon
with no rows was never in the index and never reached `assign_tier`. SPEC §16.2 is
explicit -- "a taxon with n_specs_tested < 10 is reported as INSUFFICIENT, not tiered"
-- and nought is less than ten, so the answer already existed; the taxon simply had to
get there. `compute_robustness` now keeps a row for every taxon, for the same reason
`spec_summary` keeps a row for every specification.

Making those taxa reachable then exposed a second defect behind them. The curve endpoint
had a branch for exactly this case, with a note saying the taxon was filtered out of
every specification, and it had never been delivered: it carried float("nan"), and
Starlette serialises JSON with allow_nan=False, so the endpoint answered 500 instead of
the note. Both are pinned here.
"""
from __future__ import annotations

import copy
import json

import pytest

from app.core.report import specification_curve
from app.core.robustness import compute_robustness
from app.core.runner import run_multiverse


@pytest.fixture(scope="module")
def run(dataset):
    return run_multiverse(dataset, mode="quick")


@pytest.fixture(scope="module")
def starved(run):
    """A copy of the run with one taxon removed from every specification's results.

    Simulating it is the point: whether any particular demo table happens to contain a
    taxon the prevalence filter rejects everywhere is a property of that table, not of
    the code, and a regression test should not depend on it.
    """
    victim = int(run.long["taxon"].iloc[0])
    clone = copy.copy(run)
    clone.long = run.long[run.long["taxon"] != victim].reset_index(drop=True)
    return clone, victim


def test_a_taxon_filtered_from_every_specification_is_still_reported(starved):
    clone, victim = starved
    summary = compute_robustness(clone)

    row = summary.table[summary.table["taxon_id"] == victim]
    assert len(row) == 1, "the taxon vanished instead of being reported"
    assert row["n_specs_tested"].iloc[0] == 0
    assert row["robustness_tier"].iloc[0] == "INSUFFICIENT", (
        "§16.2: below ten specifications the tier is INSUFFICIENT, and nought "
        "specifications is below ten"
    )


def test_every_taxon_gets_exactly_one_tier(starved):
    """The counts have to add up to the table, or the page cannot be trusted."""
    clone, _ = starved
    summary = compute_robustness(clone)
    assert sum(summary.tier_counts.values()) == len(summary.table)
    assert len(summary.table) == len(clone.taxa_names)
    assert summary.table["taxon_id"].is_unique


def test_the_curve_for_an_untested_taxon_is_valid_json(starved):
    """Starlette renders with allow_nan=False, so NaN here is a 500, not a payload."""
    clone, victim = starved
    payload = specification_curve(clone, victim)

    assert payload["n_specs"] == 0
    assert payload["median_effect"] is None
    assert payload["frac_significant"] is None
    assert "filtered out of every specification" in payload["note"]

    # The assertion that matters: this is what JSONResponse does to it.
    json.dumps(payload, allow_nan=False)


def test_a_tested_taxon_still_reports_numbers(run):
    """The fix must not turn a real measurement into a null."""
    summary = compute_robustness(run)
    tested = summary.table[summary.table["n_specs_tested"] > 0]
    assert len(tested), "the run produced no tested taxa at all"

    payload = specification_curve(run, int(tested["taxon_id"].iloc[0]))
    assert payload["n_specs"] > 0
    assert payload["median_effect"] is not None
    assert payload["frac_significant"] is not None
    json.dumps(payload, allow_nan=False)
