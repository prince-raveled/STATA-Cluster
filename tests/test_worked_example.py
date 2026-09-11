"""SPEC §19 — the verification gate.

"Hand-verify this example against your implementation at M6. If the tiers don't match,
the engine is wrong."

The three taxa of §19 are encoded here with the exact metrics the spec publishes, and
the tier each must receive. This file is the contract: if it fails, fix the engine.

One arithmetic note, logged in SPEC §24: §19's own headline counts are inconsistent
(208 matrices x 4 elementary methods x 3 FDR settings is 2,496 enumerated, not the
4,992 quoted, and 3,104 valid cannot exceed 2,496 enumerated). 4,992 is the two-rank
figure from §12. The tiers — the stated gate — are what is asserted here; the grid
arithmetic is asserted against §12 in `test_grid_arithmetic_matches_spec_12`.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.robustness import assign_tier

# --- §19, verbatim --------------------------------------------------------
WORKED_EXAMPLE = {
    "Faecalibacterium": {
        "n_specs_tested": 3104,
        "frac_tested": 1.00,
        "frac_significant": 0.87,
        "sign_consistency": 1.00,
        "median_effect": -1.42,
        "iqr": (-1.71, -1.18),
        "tier": "ROBUST",
    },
    "Escherichia": {
        "n_specs_tested": 2328,
        "frac_tested": 0.75,
        "frac_significant": 0.41,
        "sign_consistency": 0.98,
        "median_effect": +0.88,
        "iqr": (+0.31, +1.44),
        "tier": "CONDITIONAL",
    },
    "Bacteroides": {
        "n_specs_tested": 3104,
        "frac_tested": 1.00,
        "frac_significant": 0.23,
        "sign_consistency": 0.61,
        "median_effect": 0.0,
        "iqr": (0.0, 0.0),
        "tier": "UNSTABLE",
    },
}


@pytest.mark.parametrize("taxon", sorted(WORKED_EXAMPLE))
def test_worked_example_tiers(taxon):
    case = WORKED_EXAMPLE[taxon]
    assert assign_tier(
        case["n_specs_tested"], case["frac_significant"], case["sign_consistency"]
    ) == case["tier"], f"{taxon} must be {case['tier']}"


def test_bacteroides_is_unstable_regardless_of_frac_significant():
    """§19: 'UNSTABLE (0.61 < 0.80 -> UNSTABLE regardless of frac_significant)'."""
    for frac in (0.0, 0.23, 0.55, 0.95):
        assert assign_tier(3104, frac, 0.61) == "UNSTABLE"


def test_tier_boundaries_are_inclusive_as_written():
    assert assign_tier(3104, 0.80, 0.95) == "ROBUST"        # >= on both
    assert assign_tier(3104, 0.7999, 0.95) == "CONDITIONAL"
    assert assign_tier(3104, 0.30, 0.95) == "CONDITIONAL"   # >= 0.30
    assert assign_tier(3104, 0.2999, 0.95) == "FRAGILE"
    assert assign_tier(3104, 0.10, 0.80) == "FRAGILE"       # >= 0.80
    assert assign_tier(3104, 0.10, 0.7999) == "UNSTABLE"


def test_insufficient_short_circuits_before_every_other_tier():
    """§15: 'A taxon with n_specs_tested < 10 is reported as INSUFFICIENT, not tiered.'"""
    assert assign_tier(9, 1.0, 1.0) == "INSUFFICIENT"
    assert assign_tier(9, 0.0, 0.0) == "INSUFFICIENT"
    assert assign_tier(10, 1.0, 1.0) == "ROBUST"


def test_gap_fillers_cover_the_uncovered_region():
    """The four published criteria do not partition the space; see SPEC §24."""
    # Never significant, but a perfectly consistent sign: matches no published tier.
    assert assign_tier(3104, 0.0, 1.0) == "NOT DETECTED"
    # Often significant, sign consistent but under 0.95: matches no published tier.
    assert assign_tier(3104, 0.60, 0.90) == "CONDITIONAL"
    assert assign_tier(3104, 0.90, 0.90) == "CONDITIONAL"


def test_grid_arithmetic_matches_spec_12(dataset):
    """SPEC §12: 416 matrices when the rank fork has two levels, halved when it has one.

    matrices = 13 rarefaction x rank x 4 prevalence x 4 transform
    enumerated = matrices x 4 elementary methods x 3 FDR settings
    """
    from app.core.grid import FDR_SETTINGS, enumerate_grid
    from app.core.models import ELEMENTARY_METHODS
    from app.core.preprocess import PREVALENCE_LEVELS, TRANSFORMS, MatrixBuilder

    builder = MatrixBuilder(dataset)
    depths, _ = builder.available_depths()
    assert len(depths) == 13, "13 rarefaction states: none + 4 depths x 3 seeds"

    matrices = len(depths) * len(builder.ranks) * len(PREVALENCE_LEVELS) * len(TRANSFORMS)
    assert matrices == 208 * len(builder.ranks)

    _, report = enumerate_grid(builder, mode="quick")
    assert report.n_enumerated == matrices * len(ELEMENTARY_METHODS) * len(FDR_SETTINGS)
    # §12 predicts roughly a third pruned by the validity matrix.
    assert 0.30 < report.prune_ratio < 0.42


def test_end_to_end_recovers_spiked_taxa(dataset, synthetic):
    """SPEC §23 validation 4: spiked taxa should land ROBUST, nulls should not."""
    from app.core.robustness import compute_robustness
    from app.core.runner import run_multiverse

    # Group levels sort alphabetically, so B (the direction a positive effect points
    # to) is "disease" and A is "control". The spiked group must be B, or every
    # ground-truth sign below is inverted.
    assert dataset.group_labels == ("control", "disease")

    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    table = summary.table.set_index("taxon")

    spiked = synthetic["differential"]
    robust = set(table.index[table["robustness_tier"] == "ROBUST"])
    assert robust, "no taxon reached ROBUST — the engine is not detecting known signal"
    # Every ROBUST call must be a true positive: this is the false-discovery check.
    assert robust <= spiked, f"ROBUST false positives: {sorted(robust - spiked)}"

    # And the direction of every robust call must match the spiked direction.
    for taxon in robust:
        index = synthetic["taxa"].index(taxon)
        assert np.sign(table.loc[taxon, "median_effect"]) == np.sign(
            synthetic["effect"][index]
        ), taxon
