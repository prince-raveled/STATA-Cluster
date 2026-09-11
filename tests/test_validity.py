"""Validity matrix — SPEC §11. Every rule, in both directions."""
from __future__ import annotations

import pytest

from app.core.grid import FDR_SETTINGS, enumerate_grid
from app.core.models import Specification
from app.core.preprocess import PREVALENCE_LEVELS, TRANSFORMS, MatrixBuilder
from app.core.validity import INCOMPATIBLE, RULES, Capabilities, invalid_reason, is_valid


def spec(method="wilcoxon", transform="raw", rarefaction="none", seed=None,
         rank="input", prevalence=0.10, fdr="bh", threshold=0.05, covariates=()):
    return Specification(rarefaction, seed, rank, prevalence, transform, method,
                         fdr, threshold, covariates)


# --- INCOMPATIBLE ---------------------------------------------------------
@pytest.mark.parametrize("method,transform", [
    ("pydeseq2", "clr"), ("pydeseq2", "tss"), ("pydeseq2", "tmm"),
    ("aldex2", "clr"),
    ("ancombc", "clr"), ("ancombc", "tss"), ("ancombc", "tmm"),
])
def test_incompatible_transforms_are_pruned(method, transform):
    assert not is_valid(spec(method=method, transform=transform))


@pytest.mark.parametrize("method", ["wilcoxon", "ttest", "linear"])
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_transform_agnostic_methods_accept_every_transform(method, transform):
    assert is_valid(spec(method=method, transform=transform))


def test_incompatible_table_matches_the_spec():
    assert INCOMPATIBLE["pydeseq2"] == {"clr", "tss", "tmm"}
    assert INCOMPATIBLE["aldex2"] == {"clr"}
    assert INCOMPATIBLE["ancombc"] == {"clr", "tss", "tmm"}
    for method in ("wilcoxon", "ttest", "logistic", "linear"):
        assert INCOMPATIBLE[method] == set()


# --- RULES ----------------------------------------------------------------
def test_ancombc_rejects_rarefaction():
    assert "double-corrects" in invalid_reason(spec(method="ancombc", rarefaction="1000", seed=1))
    assert is_valid(spec(method="ancombc", rarefaction="none"))


def test_tmm_rejects_rarefaction():
    assert "double library-size correction" in invalid_reason(
        spec(transform="tmm", rarefaction="5000", seed=2)
    )
    assert is_valid(spec(transform="tmm", rarefaction="none"))


def test_logistic_keeps_only_the_raw_canonical_specification():
    assert is_valid(spec(method="logistic", transform="raw"))
    for transform in ("tss", "clr", "tmm"):
        assert "transform-invariant" in invalid_reason(
            spec(method="logistic", transform=transform, rarefaction="none")
        )


def test_pydeseq2_rejects_rarefaction():
    assert "models library size internally" in invalid_reason(
        spec(method="pydeseq2", rarefaction="min", seed=1)
    )
    assert is_valid(spec(method="pydeseq2", rarefaction="none"))


def test_every_rule_is_reachable():
    """Each published rule must actually prune something, or it is dead code."""
    reasons = set()
    for method in INCOMPATIBLE:
        for transform in TRANSFORMS:
            for rarefaction, seed in (("none", None), ("1000", 1)):
                reason = invalid_reason(spec(method=method, transform=transform,
                                             rarefaction=rarefaction, seed=seed))
                if reason:
                    reasons.add(reason)
    for _, reason in RULES:
        assert reason in reasons, f"rule never fires: {reason}"


# --- dataset capabilities -------------------------------------------------
def test_capabilities_prune_impossible_specifications():
    caps = Capabilities(integer_counts=False, can_rarefy=False, can_collapse_to_genus=False)
    assert "not integers" in invalid_reason(spec(rarefaction="1000", seed=1), caps)
    assert "integer counts" in invalid_reason(spec(method="pydeseq2"), caps)
    assert "integer counts" in invalid_reason(spec(transform="tmm"), caps)
    assert "cannot be collapsed to genus" in invalid_reason(spec(rank="genus"), caps)
    assert is_valid(spec(), caps)


# --- grid counts ----------------------------------------------------------
def test_quick_grid_reports_both_counts(dataset):
    builder = MatrixBuilder(dataset)
    specs, report = enumerate_grid(builder, mode="quick")
    depths, _ = builder.available_depths()

    expected = (len(depths) * len(builder.ranks) * len(PREVALENCE_LEVELS)
                * len(TRANSFORMS) * 4 * len(FDR_SETTINGS))
    assert report.n_enumerated == expected
    assert report.n_valid == len(specs)
    assert report.n_pruned == report.n_enumerated - report.n_valid
    assert 0 < report.prune_ratio < 1
    assert sum(report.pruned_reasons.values()) == report.n_pruned


def test_grid_has_no_duplicate_specifications(dataset):
    specs, _ = enumerate_grid(MatrixBuilder(dataset), mode="quick")
    assert len(set(specs)) == len(specs)


def test_matrix_key_ignores_method_and_fdr():
    base = spec()
    assert base.matrix_key == spec(method="ttest", fdr="by", threshold=0.10).matrix_key
    assert base.fit_key != spec(method="ttest").fit_key
    assert base.fit_key == spec(fdr="by", threshold=0.10).fit_key


def test_full_mode_adds_the_sophisticated_methods(dataset):
    quick, _ = enumerate_grid(MatrixBuilder(dataset), mode="quick")
    full, report = enumerate_grid(MatrixBuilder(dataset), mode="full")
    quick_methods = {s.method for s in quick}
    full_methods = {s.method for s in full}
    assert quick_methods == {"wilcoxon", "ttest", "logistic", "linear"}
    assert full_methods > quick_methods
    assert len(full) > len(quick)
    assert any("sampled" in note for note in report.notes)


def test_covariate_mode_enumerates_subsets(dataset):
    specs, report = enumerate_grid(
        MatrixBuilder(dataset), mode="covariate",
        covariate_columns=["age", "sex"],
    )
    subsets = {s.covariates for s in specs}
    assert () in subsets
    assert ("age", "sex") in subsets
    assert len(subsets) == 4  # 2^2
    assert report.n_matrices <= 12  # §12 reference sub-grid


def test_declared_specification_is_added_when_absent(dataset):
    declared = spec(method="wilcoxon", transform="clr", rarefaction="none", prevalence=0.10)
    specs, report = enumerate_grid(MatrixBuilder(dataset), mode="quick", declared=declared)
    assert declared in specs
