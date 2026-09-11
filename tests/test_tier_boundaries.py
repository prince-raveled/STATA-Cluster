"""Adversarial tests on every §16.2 tier boundary and degenerate input.

`assign_tier` is a cascade of inequalities evaluated in order, first match wins. That
shape has two classic failure modes: a value exactly on a threshold falling to the wrong
side, and a combination that matches no branch at all and silently reaches whatever the
final `return` happens to be. Both change what a researcher is told about their data, and
neither shows up in a test that only uses comfortable values.

The tier is now an empirically validated claim (`tests/reference/tier_validation.py`:
ROBUST taxa replicated 90% of the time, FRAGILE 12%). That raises the cost of a boundary
bug from cosmetic to substantive — a taxon pushed from FRAGILE to ROBUST by a rounding
error inherits a 90% replication claim it has not earned.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.robustness import MIN_SPECS_FOR_TIER, TIER_ORDER, assign_tier

ENOUGH = MIN_SPECS_FOR_TIER  # specifications; below this the tier is INSUFFICIENT


# --- the INSUFFICIENT gate short-circuits everything ------------------------
@pytest.mark.parametrize("n_specs", [0, 1, 2, ENOUGH - 1])
def test_too_few_specifications_is_insufficient_whatever_else_is_true(n_specs):
    """§16.2 evaluates INSUFFICIENT first: perfect evidence over 9 specs is still nothing."""
    assert assign_tier(n_specs, 1.0, 1.0) == "INSUFFICIENT"
    assert assign_tier(n_specs, 0.0, 0.0) == "INSUFFICIENT"


def test_exactly_the_minimum_is_enough_to_be_tiered():
    """The gate is `< MIN`, so MIN itself must be tierable — an off-by-one here would
    quietly discard every taxon sitting on the boundary."""
    assert assign_tier(ENOUGH, 1.0, 1.0) == "ROBUST"
    assert assign_tier(ENOUGH - 1, 1.0, 1.0) == "INSUFFICIENT"


# --- ROBUST / CONDITIONAL boundary: frac_significant 0.80 -------------------
@pytest.mark.parametrize(("frac", "expected"), [
    (0.7999, "CONDITIONAL"),
    (0.80, "ROBUST"),
    (0.8001, "ROBUST"),
])
def test_robust_boundary_on_significance(frac, expected):
    assert assign_tier(100, frac, 1.0) == expected


@pytest.mark.parametrize(("consistency", "expected"), [
    (0.9499, "UNSTABLE"),      # sign not settled, and below 0.80 it is UNSTABLE
    (0.95, "ROBUST"),
    (0.9501, "ROBUST"),
])
def test_robust_boundary_on_sign_consistency(consistency, expected):
    """Below 0.95 a fully significant taxon is not ROBUST; below 0.80 it is UNSTABLE."""
    tier = assign_tier(100, 1.0, consistency)
    if consistency >= 0.95:
        assert tier == expected
    else:
        assert tier in {"CONDITIONAL", "UNSTABLE"}, tier


def test_the_gap_between_080_and_095_sign_consistency_is_conditional():
    """The four published criteria do not partition the space; §24 logs the gap-filler.

    A taxon significant in most specifications whose direction is settled 85% of the
    time matches no §16.2 branch. It must land CONDITIONAL — never fall through to
    something arbitrary.
    """
    for consistency in (0.80, 0.85, 0.90, 0.9499):
        assert assign_tier(100, 0.90, consistency) == "CONDITIONAL"


# --- CONDITIONAL / FRAGILE boundary: frac_significant 0.30 -----------------
@pytest.mark.parametrize(("frac", "expected"), [
    (0.2999, "FRAGILE"),
    (0.30, "CONDITIONAL"),
    (0.3001, "CONDITIONAL"),
])
def test_conditional_boundary_on_significance(frac, expected):
    assert assign_tier(100, frac, 1.0) == expected


# --- UNSTABLE: direction not determined ------------------------------------
@pytest.mark.parametrize("consistency", [0.0, 0.5, 0.7999])
def test_direction_not_settled_is_unstable(consistency):
    """Whatever the significance rate, an undetermined direction is not a finding."""
    for frac in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert assign_tier(100, frac, consistency) == "UNSTABLE"


def test_sign_consistency_below_a_half_is_impossible_but_handled():
    """`_sign_consistency` returns max(pos, neg)/n, so it cannot be below 0.5 — but the
    tier function is public and must not misbehave if it ever is."""
    assert assign_tier(100, 0.5, 0.2) == "UNSTABLE"


# --- NOT DETECTED: significant nowhere -------------------------------------
def test_never_significant_with_a_settled_direction_is_not_detected():
    """§16.2's four criteria leave this unmatched too; §24 logs the added tier."""
    assert assign_tier(100, 0.0, 1.0) == "NOT DETECTED"
    assert assign_tier(100, 0.0, 0.95) == "NOT DETECTED"


def test_never_significant_with_an_unsettled_direction_is_unstable_not_undetected():
    """Order matters: UNSTABLE is evaluated before NOT DETECTED, and should be —
    'we could not tell which way' is more informative than 'we found nothing'."""
    assert assign_tier(100, 0.0, 0.5) == "UNSTABLE"


# --- exhaustiveness: no input may fall through -----------------------------
@pytest.mark.parametrize("n_specs", [ENOUGH, 50, 3192])
def test_every_combination_lands_in_a_named_tier(n_specs):
    """A grid over the whole input space. Any gap in §16.2's cascade shows up here."""
    grid = np.linspace(0.0, 1.0, 41)
    for frac in grid:
        for consistency in grid:
            tier = assign_tier(n_specs, float(frac), float(consistency))
            assert tier in TIER_ORDER, (n_specs, frac, consistency, tier)


def test_tier_is_monotone_in_significance_at_a_settled_direction():
    """With the direction settled, more significance must never mean a weaker tier."""
    strength = {t: i for i, t in enumerate(reversed(TIER_ORDER))}
    previous = -1
    for frac in np.linspace(0.0, 1.0, 51):
        tier = assign_tier(100, float(frac), 1.0)
        # NOT DETECTED at zero is the floor; from there the tier only strengthens.
        assert strength[tier] >= previous or frac == 0.0, (frac, tier)
        previous = max(previous, strength[tier])


# --- degenerate numeric inputs ---------------------------------------------
@pytest.mark.parametrize(("frac", "consistency"), [
    (float("nan"), 1.0),
    (1.0, float("nan")),
    (float("nan"), float("nan")),
])
def test_non_finite_inputs_are_insufficient_not_a_claim(frac, consistency):
    """NaN compares false against every threshold, so an unguarded cascade falls through
    to its final branch — which was CONDITIONAL, a tier now carrying a measured 51%
    replication rate. Undefined evidence must earn no tier at all."""
    assert assign_tier(100, frac, consistency) == "INSUFFICIENT"


@pytest.mark.parametrize(("frac", "consistency"), [(-0.1, 1.0), (1.1, 1.0), (1.0, 1.5)])
def test_out_of_range_inputs_still_return_a_named_tier(frac, consistency):
    assert assign_tier(100, frac, consistency) in TIER_ORDER


# --- the empirical claim attached to each tier ------------------------------
def test_only_tiers_with_held_out_evidence_carry_a_replication_number():
    """A tier the validation experiment never observed must not quote a rate."""
    from app.core.evidence import TIER_REPLICATION

    for tier in TIER_REPLICATION:
        assert tier in TIER_ORDER, f"{tier} has evidence but is not a tier the app assigns"
    # INSUFFICIENT is a statement about the grid, not about the taxon: it predicts
    # nothing and must never be given a replication rate.
    assert "INSUFFICIENT" not in TIER_REPLICATION
