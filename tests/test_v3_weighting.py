"""v3 Phase 0 — weighting, pruning R8-R9, signed z (docs/MICROVERSE_V3_PLAN.md §26.5).

Each test is one acceptance criterion of §26.5, named for it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.effects import signed_z
from app.core.models import Specification
from app.core.robustness import compute_robustness
from app.core.runner import run_multiverse
from app.core.validity import (
    DEFENSIBILITY,
    R8_REASON,
    apply_rules,
    invalid_reason,
    needs_depth_adjustment,
    rule_id,
)
from app.core.weighted import per_taxon_metrics, weighted_quantiles
from app.core.weights import (
    TREE_NODES,
    WeightError,
    effective_number,
    node_value,
    tree_weights,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def spec(rarefaction="none", seed=None, rank="input", prev=0.0, transform="tss",
         method="wilcoxon", fdr="bh", threshold=0.05, covariates=()):
    return Specification(rarefaction, seed, rank, prev, transform, method, fdr, threshold,
                         covariates)


# A pruned toy grid: two unrarefied method pairs, two seeds at one depth, one at another.
TOY = [
    spec("none", None, transform="tss"),
    spec("none", None, transform="tmm"),
    spec("min", 1),
    spec("min", 2),
    spec("1000", 1),
]


# --- weights --------------------------------------------------------------
def test_toy_tree_weights_match_hand_computation():
    # decision tree: rarefy no/yes 1/2 each; no -> 2 method pairs; yes -> depths min
    # and 1000 at 1/4 each; min -> two seeds at 1/8 each.
    assert np.allclose(tree_weights(TOY, "decision_tree"), [0.25, 0.25, 0.125, 0.125, 0.25])
    # flat tree: none, min, 1000 at 1/3 each.
    assert np.allclose(tree_weights(TOY, "flat_tree"), [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 3])
    assert np.allclose(tree_weights(TOY, "uniform"), 0.2)


@pytest.mark.parametrize("scheme", ["decision_tree", "flat_tree", "uniform"])
def test_total_weight_is_one(scheme):
    assert tree_weights(TOY, scheme).sum() == pytest.approx(1.0, abs=1e-12)


def test_conditional_probabilities_sum_to_one_at_every_node(ibd_v3):
    """Re-derive P(child | parent) from the weights: it must sum to 1 over the valid
    children of every node, which is what makes the weights feasible under pruning."""
    specs = ibd_v3.specs
    weights = tree_weights(specs, "decision_tree")
    paths = [tuple(node_value(s, n) for n in TREE_NODES) for s in specs]
    for depth in range(len(TREE_NODES)):
        parents: dict = {}
        for path, w in zip(paths, weights, strict=True):
            parent, child = path[:depth], path[depth]
            parents.setdefault(parent, {}).setdefault(child, 0.0)
            parents[parent][child] += w
        for children in parents.values():
            total = sum(children.values())
            conditional = [v / total for v in children.values()]
            assert sum(conditional) == pytest.approx(1.0, abs=1e-12)
            # Equal over the valid children, at every node of the decision tree.
            assert np.allclose(conditional, 1.0 / len(children))


def test_seeds_share_their_depth_equally(ibd_v3):
    weights = tree_weights(ibd_v3.specs, "decision_tree")
    by_rest: dict = {}
    for s, w in zip(ibd_v3.specs, weights, strict=True):
        if s.rarefaction == "none":
            continue
        rest = (s.rarefaction, s.rank, s.prev_filter, s.transform, s.method, s.fdr_method,
                s.fdr_threshold)
        by_rest.setdefault(rest, []).append(w)
    assert by_rest
    for seeds in by_rest.values():
        assert len(seeds) == 3
        assert np.allclose(seeds, seeds[0])


def test_custom_scheme_is_strict():
    custom = {"rarefy": {"yes": 3, "no": 1}}
    assert np.allclose(tree_weights(TOY, "custom", custom)[:2].sum(), 0.25)
    with pytest.raises(WeightError, match="no probability"):
        tree_weights(TOY, "custom", {"rarefy": {"yes": 1}})
    with pytest.raises(WeightError, match="probability 0"):
        tree_weights(TOY, "custom", {"rarefy": {"yes": 0, "no": 0}})
    with pytest.raises(WeightError, match="practice review"):
        tree_weights(TOY, "practice")


def test_effective_number():
    assert effective_number(np.full(10, 0.1)) == pytest.approx(10.0)
    assert effective_number(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


# --- per-taxon conditioning -------------------------------------------------
def test_per_taxon_conditioning_renormalises_over_tested_specifications():
    """A taxon filtered out of some specifications is scored over the rest only."""
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    long = pd.DataFrame({
        "spec_id": [0, 1, 2, 3, 0, 2],
        "taxon": [0, 0, 0, 0, 1, 1],       # taxon 1 is missing from specs 1 and 3
        "significant": [True, True, False, False, True, False],
        "p_raw": [0.01, 0.01, 0.2, 0.3, 0.01, 0.2],
        "effect_h": [1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
    })
    metrics = per_taxon_metrics(long, weights, n_taxa=3)
    assert metrics["frac_significant"][0] == pytest.approx(0.7)
    assert metrics["frac_significant"][1] == pytest.approx(0.4 / 0.6)
    assert metrics["sign_consistency"][1] == pytest.approx(0.4 / 0.6)
    assert np.isnan(metrics["frac_significant"][2])          # never tested


def test_weighted_quantiles():
    values = np.array([1.0, 2.0, 3.0, 10.0])
    groups = np.zeros(4, dtype=int)
    equal = weighted_quantiles(values, np.ones(4), groups, 1, quantiles=(0.5,))
    assert equal[0, 0] == pytest.approx(2.5)
    heavy = weighted_quantiles(values, np.array([1, 1, 1, 97.0]), groups, 1, (0.5,))
    assert heavy[0, 0] > 3.0


# --- pruning R8, R9 ---------------------------------------------------------
@pytest.mark.parametrize("method", ["wilcoxon", "ttest", "linear"])
def test_r8_removes_depth_naive_tests_on_unrarefied_raw_counts(method):
    target = spec("none", None, transform="raw", method=method)
    assert invalid_reason(target, ruleset="v3") == R8_REASON
    assert invalid_reason(target, ruleset="v2") == ""
    # Rarefied raw counts, and unrarefied normalised ones, are untouched.
    assert invalid_reason(spec("min", 1, transform="raw", method=method)) == ""
    assert invalid_reason(spec("none", None, transform="tss", method=method)) == ""


def test_r9_adjusts_unrarefied_presence_absence_for_depth():
    unrarefied = spec("none", None, transform="raw", method="logistic")
    assert needs_depth_adjustment(unrarefied, "v3")
    assert not needs_depth_adjustment(unrarefied, "v2")
    assert not needs_depth_adjustment(spec("min", 1, transform="raw", method="logistic"))
    assert apply_rules(unrarefied, "v3").depth_adjusted
    assert not apply_rules(unrarefied, "v2").depth_adjusted


def test_r9_changes_the_fit(ibd_v3, ibd_v2):
    """The depth-adjusted logistic model is a different fit from the 2x2 test."""
    def logistic_p(run):
        ids = [i for i, s in enumerate(run.specs)
               if s.method == "logistic" and s.rarefaction == "none"
               and s.prev_filter == 0.10 and s.fdr_method == "bh"
               and s.fdr_threshold == 0.05]
        assert len(ids) == 1
        return run.long[run.long["spec_id"] == ids[0]].set_index("taxon")["p_raw"]

    assert all(s.depth_adjusted for s in ibd_v3.specs
               if s.method == "logistic" and s.rarefaction == "none")
    adjusted, plain = logistic_p(ibd_v3), logistic_p(ibd_v2)
    assert not np.allclose(adjusted.to_numpy(), plain.loc[adjusted.index].to_numpy())


def test_rule_ids_and_register_cover_every_rule():
    ids = [r["id"] for r in DEFENSIBILITY["rules"]]
    assert ids == [f"R{i}" for i in range(1, 10)]
    for rule in DEFENSIBILITY["rules"] + DEFENSIBILITY["forks"]:
        assert rule["type"] in ("E", "N", "U")
        assert rule["reason"]
    assert rule_id(R8_REASON) == "R8"
    assert rule_id("TMM + rarefaction is double library-size correction") == "R5"
    assert rule_id("PyDESeq2 requires integer counts; CLR input is not defensible") == "R1"


def test_declared_pipeline_outside_the_grid_is_explained(ibd_dataset):
    declared = spec("none", None, prev=0.10, transform="raw", method="wilcoxon")
    run = run_multiverse(ibd_dataset, mode="quick", declared=declared)
    assert run.declared_spec_id == -1
    assert any("outside the defensible grid" in note for note in run.grid_report.notes)


# --- the IBD demo -------------------------------------------------------------
def test_ibd_demo_counts_and_rarefied_shares(ibd_v3):
    assert ibd_v3.grid_report.n_valid == 1560
    summary = compute_robustness(ibd_v3)
    shares = {k: c.rarefied_share for k, c in summary.composition.items()}
    assert shares["uniform"] == pytest.approx(0.923, abs=0.001)
    assert shares["flat_tree"] == pytest.approx(0.800, abs=1e-9)
    assert shares["decision_tree"] == pytest.approx(0.500, abs=1e-9)


def test_primary_tier_is_the_decision_tree_and_every_scheme_is_reported(ibd_v3):
    summary = compute_robustness(ibd_v3)
    table = summary.table
    assert summary.scheme == "decision_tree"
    assert (table["robustness_tier"] == table["tier_decision_tree"]).all()
    stable = table[["tier_uniform", "tier_flat_tree", "tier_decision_tree"]].nunique(axis=1)
    assert (table["weight_stable"] == stable.eq(1)).all()
    assert summary.tier_changes["uniform"] == int(
        (table["tier_uniform"] != table["robustness_tier"]).sum())


def test_uniform_scheme_on_a_v3_run_counts_one_vote_each(ibd_v3):
    summary = compute_robustness(ibd_v3, scheme="uniform")
    table = summary.table
    assert np.allclose(table["frac_significant"], table["frac_significant_unweighted"])


def test_planted_truth_holds_under_the_decision_tree(ibd_v3):
    """The demo's ROBUST calls stay planted effects when the weights change."""
    truth_path = os.path.join(os.path.dirname(HERE), "examples", "ibd_genus_truth.json")
    with open(truth_path, encoding="utf-8") as handle:
        truth = json.load(handle)
    planted = set(truth["differential_taxa"])
    assert planted, "the truth file lists the planted taxa"
    table = compute_robustness(ibd_v3).table
    robust = set(table.loc[table["robustness_tier"] == "ROBUST", "taxon"])
    assert robust and robust <= planted


# --- signed z and pseudocount sensitivity -------------------------------------
def test_signed_z():
    z = signed_z(np.array([1.0, 0.05, 0.05, 1e-400]), np.array([2.0, 1.0, -1.0, 3.0]))
    assert z[0] == 0.0
    assert z[1] == pytest.approx(1.959964, abs=1e-6)
    assert z[2] == pytest.approx(-1.959964, abs=1e-6)
    assert np.isfinite(z[3]) and z[3] > 30


def test_signed_z_moves_where_the_harmonised_effect_cannot(t2d_covariate_v3):
    run = t2d_covariate_v3
    long = run.long.assign(matrix=run.spec_matrix[run.long["spec_id"].to_numpy()])
    long["z"] = signed_z(long["p_raw"].to_numpy(), long["effect_n"].to_numpy())
    spread = long.groupby(["matrix", "taxon"]).agg(effect_sd=("effect_h", "std"),
                                                    z_sd=("z", "std"))
    assert spread["effect_sd"].max() == pytest.approx(0.0, abs=1e-6)
    assert (spread["z_sd"] > 0).mean() > 0.5


def test_pseudocount_sensitivity_is_reported(ibd_v3):
    table = compute_robustness(ibd_v3).table
    for column in ("median_effect_pc_low", "median_effect_pc_high", "pseudocount_sign_flip"):
        assert column in table.columns
    tested = table["n_specs_tested"] > 0
    assert table.loc[tested, "median_effect_pc_low"].notna().all()


# --- v2 is still v2 -------------------------------------------------------------
@pytest.mark.parametrize("case", ["ibd_genus_quick", "gut_species_quick",
                                  "t2d_covariates_covariate"])
def test_v2_outputs_are_byte_identical_under_v2_rules(case):
    """§26.5: with the v2 rule set, every v2 output matches files v2 itself wrote."""
    sys.path.insert(0, os.path.join(HERE, "fixtures"))
    import make_v2_golden

    demo, _, mode = case.rpartition("_")
    # Per platform: the same v2 engine writes different bytes on Windows and Linux
    # (line separators; last-bit linear algebra in covariate mode). See make_v2_golden.
    expected = make_v2_golden.expected_hashes(sys.platform)
    if not expected:
        pytest.skip(f"no v2 golden record for {sys.platform}: generate one from 8d1cd9f "
                    "with tests/fixtures/make_v2_golden.py")
    for name, data in make_v2_golden.outputs(demo, mode, ruleset="v2").items():
        assert hashlib.sha256(data).hexdigest() == expected[f"{case}/{name}"], name


def test_a_run_stored_before_v3_is_summarised_as_v2(ibd_v2):
    """Stored runs unpickle without the v3 attributes; they must read as v2."""
    legacy = SimpleNamespace(**{k: v for k, v in vars(ibd_v2).items()
                                if k not in ("ruleset", "effect_sensitivity", "spec_matrix",
                                             "pseudocount")})
    summary = compute_robustness(legacy)
    assert summary.scheme == "uniform" and not summary.weighted
    assert "tier_uniform" not in summary.table.columns


# --- fixtures -------------------------------------------------------------------
@pytest.fixture(scope="module")
def ibd_dataset():
    from app.services import load_demo
    return load_demo("ibd_genus")


@pytest.fixture(scope="module")
def ibd_v3(ibd_dataset):
    return run_multiverse(ibd_dataset, mode="quick")


@pytest.fixture(scope="module")
def ibd_v2(ibd_dataset):
    return run_multiverse(ibd_dataset, mode="quick", ruleset="v2")


@pytest.fixture(scope="module")
def t2d_covariate_v3():
    from app.services import load_demo
    dataset = load_demo("t2d_covariates")
    return run_multiverse(dataset, mode="covariate",
                          covariate_columns=list(dataset.covariate_columns))
