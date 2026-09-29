"""Specification weights — v3 plan §26.2.

v2 gave every valid specification one vote. On the IBD demo 480 of the 532 specifications
per FDR setting are rarefied, because rarefaction multiplies into 4 depths x 3 seeds
while "do not rarefy" is a single level. One vote each therefore puts about 90% of the
weight behind rarefying, and a taxon significant only after rarefying can be ROBUST
while one significant only without it cannot rise above FRAGILE. The tiers took a side
in the rarefaction debate that MicroVerse says it does not take.

A weight here is the probability that an analyst who walks the pipeline in the SPEC §9
order ends at a given specification, choosing among the options that are still valid
given the choices already made. Because it is built from the valid specifications
themselves, a weighting can never be infeasible however much pruning removed, and it is
a product of conditional probabilities that can be written down and audited.

    rarefy? -> depth -> seed -> rank -> prevalence -> (transform, test) -> FDR -> covariates

Transformation and test are one node because pruning acts on their combination (TMM
exists only without rarefaction; logistic only on raw counts); splitting them would
create structural zeros that no per-fork balancing could satisfy.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: The schemes a run can report tiers under. `practice` arrives with the practice
#: review (plan §28) and is refused until its frequencies exist.
SCHEMES = ("decision_tree", "uniform", "flat_tree")
DEFAULT_SCHEME = "decision_tree"

SCHEME_LABELS = {
    "uniform": "one vote per specification (v2)",
    "decision_tree": "decision tree: rarefy or not 50/50, then equal within each choice",
    "flat_tree": "flat tree: the five rarefaction levels 1/5 each, then equal",
    "custom": "user-declared conditional probabilities",
    "practice": "frequencies of published practice",
}

#: Node order for the tree schemes. `flat_tree` merges the first two nodes.
TREE_NODES = ("rarefy", "depth", "seed", "rank", "prevalence", "method_pair", "fdr",
              "covariates")
FLAT_NODES = ("rarefaction", "seed", "rank", "prevalence", "method_pair", "fdr",
              "covariates")

NODE_LABELS = {
    "rarefy": "Rarefy or not",
    "depth": "Rarefaction depth",
    "rarefaction": "Rarefaction level",
    "seed": "Rarefaction seed",
    "rank": "Taxonomic rank",
    "prevalence": "Prevalence filter",
    "method_pair": "Transformation and test",
    "fdr": "FDR setting",
    "covariates": "Covariate adjustment",
}


def node_value(spec, node: str) -> str:
    """The level a specification takes at one decision node, as a stable string.

    Strings, not the raw values, because custom schemes are written by people and
    sealed as JSON (plan §31.2): "0.1" must mean the same thing on both sides.
    """
    if node == "rarefy":
        return "no" if spec.rarefaction == "none" else "yes"
    if node in ("depth", "rarefaction"):
        return str(spec.rarefaction)
    if node == "seed":
        return "none" if spec.rare_seed is None else str(int(spec.rare_seed))
    if node == "rank":
        return str(spec.rank)
    if node == "prevalence":
        return f"{float(spec.prev_filter):g}"
    if node == "method_pair":
        pair = f"{spec.transform}/{spec.method}"
        return pair + "+depth" if getattr(spec, "depth_adjusted", False) else pair
    if node == "fdr":
        return f"{spec.fdr_method}@{float(spec.fdr_threshold):g}"
    if node == "covariates":
        return "|".join(spec.covariates) if spec.covariates else "none"
    raise ValueError(f"Unknown decision node: {node}")


class WeightError(ValueError):
    """A custom scheme that cannot be applied to this grid, with the reason."""


def _conditional(node: str, children: Sequence[str], scheme: str, custom: dict) -> dict:
    """P(child | path so far) over the children that are valid at this point."""
    if scheme == "custom" and node in (custom or {}):
        declared = custom[node]
        missing = [c for c in children if c not in declared]
        if missing:
            raise WeightError(
                f"The custom scheme gives no probability for {NODE_LABELS[node].lower()} "
                f"level(s) {', '.join(missing)}, which this grid contains. Give every "
                f"level a value (0 is allowed).")
        raw = {c: float(declared[c]) for c in children}
        if any(v < 0 or not np.isfinite(v) for v in raw.values()):
            raise WeightError(f"Probabilities for {NODE_LABELS[node].lower()} must be "
                              "finite and non-negative.")
        total = sum(raw.values())
        if total <= 0:
            raise WeightError(f"Every valid level of {NODE_LABELS[node].lower()} has "
                              "probability 0, so no specification could be reached.")
        return {c: v / total for c, v in raw.items()}
    # Equal over the valid children. For `decision_tree` the first node is the binary
    # "rarefy or not", so each side gets half however many depths and seeds sit behind
    # it (plan §26.1); with only one side valid, that side gets everything.
    return {c: 1.0 / len(children) for c in children}


def tree_weights(specs: Sequence, scheme: str = DEFAULT_SCHEME, custom: dict = None
                 ) -> np.ndarray:
    """One weight per specification, summing to 1.

    `uniform` is 1/N each. The tree schemes walk the nodes in order: at every node the
    specifications still in play are grouped by their level there, each group receives
    its conditional probability of the parent's mass, and the recursion continues
    inside it. The walk visits only levels that some valid specification takes, so the
    conditional probabilities at every node sum to 1 over valid children.
    """
    n = len(specs)
    if n == 0:
        return np.zeros(0, dtype=float)
    if scheme == "uniform":
        return np.full(n, 1.0 / n, dtype=float)
    if scheme == "practice":
        raise WeightError("Practice weights need the practice review (plan §28), which "
                          "has not been published yet.")
    if scheme not in ("decision_tree", "flat_tree", "custom"):
        raise WeightError(f"Unknown weighting scheme: {scheme}")
    if scheme == "custom" and not isinstance(custom, dict):
        raise WeightError("A custom scheme needs its conditional probabilities.")

    nodes = FLAT_NODES if scheme == "flat_tree" else TREE_NODES
    if scheme == "custom" and "rarefaction" in custom:
        nodes = FLAT_NODES
    paths = [tuple(node_value(s, node) for node in nodes) for s in specs]
    weights = np.zeros(n, dtype=float)

    # Iterative rather than recursive: a covariate-mode grid is deep enough in places
    # that Python's recursion limit is not a comfortable margin.
    stack = [(list(range(n)), 0, 1.0)]
    while stack:
        members, depth, mass = stack.pop()
        if depth == len(nodes):
            # Distinct specifications never share a full path; if a grid ever holds
            # duplicates, they split the leaf rather than each taking all of it.
            for i in members:
                weights[i] = mass / len(members)
            continue
        groups: dict = {}
        for i in members:
            groups.setdefault(paths[i][depth], []).append(i)
        probabilities = _conditional(nodes[depth], sorted(groups), scheme, custom)
        for level, group in groups.items():
            stack.append((group, depth + 1, mass * probabilities[level]))
    return weights


def all_scheme_weights(specs: Sequence, custom: dict = None) -> dict:
    """Weights under every built-in scheme (and `custom`, when one is declared)."""
    out = {scheme: tree_weights(specs, scheme) for scheme in SCHEMES}
    if custom:
        out["custom"] = tree_weights(specs, "custom", custom)
    return out


def effective_number(weights: np.ndarray) -> float:
    """Kish's effective number of specifications, (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    denominator = float((w ** 2).sum())
    return float(w.sum() ** 2 / denominator) if denominator > 0 else 0.0


@dataclass
class GridComposition:
    """Where the weight sits, for the panel shown on every run (plan §26.2)."""

    scheme: str
    n_specs: int
    n_effective: float
    rarefied_share: float
    by_method: dict = field(default_factory=dict)
    by_transform: dict = field(default_factory=dict)
    by_rank: dict = field(default_factory=dict)
    by_prevalence: dict = field(default_factory=dict)


def _shares(specs, weights, attribute) -> dict:
    out: dict = {}
    for spec, w in zip(specs, weights, strict=True):
        key = attribute(spec)
        out[key] = out.get(key, 0.0) + float(w)
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def composition(specs: Sequence, weights: np.ndarray, scheme: str) -> GridComposition:
    w = np.asarray(weights, dtype=float)
    total = float(w.sum()) or 1.0
    rarefied = sum(float(x) for s, x in zip(specs, w, strict=True) if s.rarefaction != "none")
    return GridComposition(
        scheme=scheme,
        n_specs=len(specs),
        n_effective=effective_number(w),
        rarefied_share=rarefied / total,
        by_method=_shares(specs, w, lambda s: s.method),
        by_transform=_shares(specs, w, lambda s: s.transform),
        by_rank=_shares(specs, w, lambda s: s.rank),
        by_prevalence=_shares(specs, w, lambda s: f"{s.prev_filter:.0%}"),
    )
