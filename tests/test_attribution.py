"""Choice attribution — SPEC §17, on synthetic data with a known driver."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.attribution import (
    EFFECT_FORKS,
    attribute,
    fallback_attribution,
    mixedlm_attribution,
    variance_of_group_means,
)
from app.core.runner import run_multiverse


def synthetic_long(driver: str, n_taxa: int = 40, seed: int = 0) -> pd.DataFrame:
    """Rows where exactly one fork moves the outcome, by construction."""
    rng = np.random.default_rng(seed)
    levels = {
        "rarefaction": ["none", "1000 (seed 1)", "5000 (seed 1)", "10000 (seed 1)"],
        "rank": ["input", "genus"],
        "prev_filter": ["0%", "5%", "10%", "20%"],
        "transform": ["TSS", "CLR", "RAW", "TMM"],
    }
    rows = []
    shifts = {level: (i - 1.5) * 3.0 for i, level in enumerate(levels[driver])}
    for taxon in range(n_taxa):
        baseline = rng.normal(0, 1.0)
        for rarefaction in levels["rarefaction"]:
            for rank in levels["rank"]:
                for prevalence in levels["prev_filter"]:
                    for transform in levels["transform"]:
                        active = {"rarefaction": rarefaction, "rank": rank,
                                  "prev_filter": prevalence, "transform": transform}
                        value = baseline + shifts[active[driver]] + rng.normal(0, 0.15)
                        rows.append({**active, "taxon": taxon, "effect_h": value})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("driver", ["rarefaction", "prev_filter", "transform", "rank"])
def test_fallback_finds_the_known_driver(driver):
    frame = synthetic_long(driver)
    shares = fallback_attribution(frame, EFFECT_FORKS, "effect_h")
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    assert ranked[0][0] == driver
    assert ranked[0][1] > 90.0
    assert abs(sum(shares.values()) - 100.0) < 1e-6


@pytest.mark.parametrize("driver", ["rarefaction", "transform"])
def test_mixedlm_finds_the_same_driver(driver):
    frame = synthetic_long(driver, n_taxa=25)
    shares, converged, note, _ = mixedlm_attribution(
        frame, EFFECT_FORKS, "effect_h")
    assert shares, note
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    assert ranked[0][0] == driver


def test_variance_of_group_means_is_zero_for_a_constant_fork():
    frame = synthetic_long("rarefaction", n_taxa=10)
    frame["rank"] = "input"  # a single level
    assert variance_of_group_means(frame, "rank", "effect_h") == 0.0


def test_shares_are_normalised_to_one_hundred():
    frame = synthetic_long("transform")
    shares = fallback_attribution(frame, EFFECT_FORKS, "effect_h")
    assert abs(sum(shares.values()) - 100.0) < 1e-9
    assert all(v >= 0 for v in shares.values())


# --- integration with a real run -----------------------------------------
@pytest.fixture(scope="module")
def attribution(dataset):
    return attribute(run_multiverse(dataset, mode="quick"))


def test_both_decompositions_are_produced(attribution):
    assert set(attribution) == {"effect", "significance", "rarefaction_draw"}
    for target in ("effect", "significance"):
        result = attribution[target]
        assert abs(sum(result.shares.values()) - 100.0) < 1e-6
        assert result.fallback_shares, "the §17 fallback is mandatory and must always run"
        assert abs(sum(result.fallback_shares.values()) - 100.0) < 1e-6
        assert result.estimator


def test_method_is_excluded_from_the_effect_decomposition(attribution):
    """§14 makes the effect estimator method-independent; see SPEC §24."""
    assert "method" not in attribution["effect"].shares
    assert "method" in attribution["significance"].shares
    assert "left out of this chart" in attribution["effect"].note


def test_fdr_forks_only_appear_in_the_significance_decomposition(attribution):
    assert not {"fdr_method", "fdr_threshold"} & set(attribution["effect"].shares)
    assert {"fdr_method", "fdr_threshold"} <= set(attribution["significance"].shares)


def test_estimators_rank_the_forks_similarly(attribution):
    """§17: 'Both should rank the forks similarly — if they disagree wildly,
    something is wrong and you want to know.'

    Ranking, not leader identity: the top two forks are often within a few points of
    each other, and which one edges ahead is not a property the engine should promise.
    """
    for target in ("effect", "significance"):
        result = attribution[target]
        agreement = result.agreement()
        if np.isfinite(agreement):
            assert agreement >= 0.5, f"{target}: rank agreement {agreement:.2f}"
        leader = max(result.shares, key=result.shares.get)
        top_two = sorted(result.fallback_shares, key=result.fallback_shares.get,
                         reverse=True)[:2]
        assert leader in top_two, f"{target}: {leader} not in fallback top two {top_two}"


def test_da_method_is_a_leading_driver_of_significance(attribution):
    """Nearing 2022 and §19 both put DA method near the top of the significance
    decomposition — it is the fork the field argues about most."""
    shares = attribution["significance"].shares
    ranked = sorted(shares, key=shares.get, reverse=True)
    assert "method" in ranked[:3]
    assert shares["method"] > 10.0


def test_rarefaction_draw_diagnostic_splits_seed_from_depth(attribution):
    draw = attribution["rarefaction_draw"]
    assert set(draw) == {"seed_share", "depth_share"}
    assert abs(draw["seed_share"] + draw["depth_share"] - 100.0) < 1e-6
    assert draw["seed_share"] > 0, "seeds are separate specifications, so this is measurable"


def test_attribution_survives_a_mixedlm_failure(dataset, monkeypatch):
    """§17 requires the run to survive MixedLM failing, using the fallback."""
    import app.core.attribution as module

    monkeypatch.setattr(module, "mixedlm_attribution",
                        lambda *a, **k: ({}, False, "forced failure", {}))
    result = attribute(run_multiverse(dataset, mode="quick"), bootstrap=False)
    for target in ("effect", "significance"):
        assert result[target].converged is False
        # The ladder drops to a per-taxon adjusted comparison, which is exact and
        # cannot fail to fit; the backup variance comparison is still reported.
        assert "Per-taxon adjusted" in result[target].estimator
        assert result[target].fallback_shares
        assert abs(sum(result[target].shares.values()) - 100.0) < 1e-6
        assert result[target].evidence == "exploratory", (
            "a run whose named estimator failed must not be graded consistent")


def test_attribution_falls_all_the_way_to_the_group_means_estimator(dataset, monkeypatch):
    """Both model-based estimators gone: §17's mandatory fallback still carries the run."""
    import app.core.attribution as module

    monkeypatch.setattr(module, "mixedlm_attribution",
                        lambda *a, **k: ({}, False, "forced failure", {}))
    monkeypatch.setattr(module, "within_attribution",
                        lambda *a, **k: ({}, float("nan"), {"reason": "forced"}))
    result = attribute(run_multiverse(dataset, mode="quick"), bootstrap=False)
    for target in ("effect", "significance"):
        assert "Simple variance comparison" in result[target].estimator
        assert abs(sum(result[target].shares.values()) - 100.0) < 1e-6
