"""V9 — does calibrated inference control error on real data, and with what power?

Pre-registered in docs/MICROVERSE_V3_PLAN.md §33:

    Question   Does §27 control error, and with what power?
    Data       (a) implementation check: mock comparisons of random halves of control
                   samples in 25 Pelto and 24 Nearing datasets (exact by construction,
                   so this verifies the code rather than the method);
               (b) spiked real templates with planted effects on known taxa, including
                   large shifts in dominant taxa, so the remaining taxa carry
                   compositional side-effects;
               (c) MIDASim-style simulations over sample size, sparsity and effect size.
    Metrics    FDR among non-planted taxa; within-family FWER; power; precision of
               CERTIFIED ROBUST.
    Success    uniform p-values in (a); FDR <= q + 0.01 in (b) and (c) without
               dominant-taxon shifts; results with dominant-taxon shifts reported
               explicitly, whatever they are.
    If it fails
               document the failure mode; restrict claims to the settings that pass.

The thresholds are read from `app/core/evidence.py::V9_PREREGISTRATION`, the same
object the Evidence page states them from.

Every replicate is what a calibrated web run does: the Quick grid under rule set v3,
decision-tree weights, then `inference.calibrate` with BH at q = 0.05. What is scored:

  discoveries      taxa selected by BH on the family p-values;
  FDR              mean over replicates of V / max(R, 1), V the discoveries among
                   non-planted taxa (in (a) every taxon is non-planted);
  within-family    for a non-planted taxon, the family test is single-step maxT over its
  FWER             specifications, so "some specification of this taxon rejected at
                   alpha" is exactly p_family <= alpha. Reported as the share of
                   non-planted taxa with p_family <= 0.05;
  power            mean over replicates of the share of planted taxa discovered;
  precision        of CERTIFIED ROBUST: planted taxa among all taxa certified, pooled
                   over replicates.

Design choices fixed here before any data were seen (logged in docs/V3_DEVIATIONS.md):

  * Datasets. Pelto: the 25 cohorts of the v2 tier validation (at least 40 samples in
    the smaller group), their control samples. Nearing: every dataset of the paper's
    filtered run with at most 1,500 features (the web limit) and at least 20 samples in
    its reference group, the grouping level with more samples (Nearing's groupings are
    not all case/control, and a mock comparison needs one homogeneous group).
  * (a) 4 random 50/50 splits of each dataset's reference samples. Uniformity is tested
    with Kolmogorov-Smirnov on one family p-value per split — the most prevalent
    taxon's — so the values are independent.
  * (b) 2 replicates per dataset and scenario. Each replicate splits the reference
    samples at random and, in one half, multiplies the planted taxa's proportions by
    their fold change, renormalises, and redraws every sample's counts as a multinomial
    at its own library size (both halves are redrawn, so they carry the same noise).
    "standard": 10 taxa planted (fewer when there are not 30 candidates), drawn from
    taxa present in at least 25% of samples outside the 5 most abundant, fold change 2
    or 4 up or down at random. "dominant_shift": the same, plus the 3 most abundant
    taxa at fold change 4 up or down.
  * (c) A Dirichlet-multinomial simulator: 100 taxa with log-normal base proportions,
    library sizes log-normal(9.5, 0.5); sparsity "low" (Dirichlet concentration 500) or
    "high" (50); n per group 20, 40 or 80; 10 planted taxa at fold change 1.5, 2 or 4;
    10 replicates per cell. The plan's MIDASim port (§31.1, `app/core/sim.py`) does not
    exist yet; when it does, (c) is to be rerun with it and the record says which
    simulator produced it.
  * B = 2,000 permutations, the web setting that is being validated (the plan's
    20,000 is for offline Atlas runs).
  * Intervals: 95% bootstrap over datasets for (a) and (b), over replicates for (c).

    python tests/reference/v9_calibration.py                      # the registered design
    python tests/reference/v9_calibration.py --parts c --replicates-c 2 --permutations 199
                                                                    # a look, not the record

Only the full registered design writes docs/v9_calibration.json and the copy in
app/core/records/ the site reads (docs/ is not deployed). Anything smaller writes under
the git-ignored data/, so a trial run can never become the record. Replicates are cached
in data/v9_calibration_rows.jsonl with the code commit, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from app.core import inference  # noqa: E402
from app.core.evidence import V9_PREREGISTRATION  # noqa: E402
from app.core.parsers.base import AbundanceTable  # noqa: E402
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.validation import validate_dataset  # noqa: E402

OUTPUT = os.path.join(ROOT, "docs", "v9_calibration.json")
APP_COPY = os.path.join(ROOT, "app", "core", "records", "v9_calibration.json")
TRIAL_OUTPUT = os.path.join(ROOT, "data", "v9_calibration_trial.json")
CACHE = os.path.join(ROOT, "data", "v9_calibration_rows.jsonl")
PELTO_EXPORT = os.path.join(ROOT, "data", "pelto", "export")

Q = V9_PREREGISTRATION["q"]
FDR_LIMIT = Q + V9_PREREGISTRATION["fdr_margin"]
KS_ALPHA = V9_PREREGISTRATION["uniformity_ks_alpha"]
FAMILY_ALPHA = V9_PREREGISTRATION["family_alpha"]

#: The registered design. A run with anything else is a trial.
REGISTERED = {
    "parts": "abc", "permutations": 2000, "splits_a": 4, "replicates_b": 2,
    "replicates_c": 10, "pelto_min_per_group": 40, "nearing_min_reference": 20,
    "max_features": 1500,
}
PLANTED = 10
PLANT_MIN_PREVALENCE = 0.25
PLANT_SKIP_TOP = 5
DOMINANT = 3
SIM_TAXA = 100
SIM_N = (20, 40, 80)
SIM_SPARSITY = {"low": 500.0, "high": 50.0}
SIM_FOLDS = (1.5, 2.0, 4.0)

FAILURES: list = []


def _shown(path) -> str:
    """A path for the console: relative to the repository where it can be, as given
    otherwise (Windows has no relative path between two drives)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


# ---------------------------------------------------------------------------------------
# Data: templates are (source, name, reference samples as a taxa × samples frame)
# ---------------------------------------------------------------------------------------
def _named(table: pd.DataFrame) -> pd.DataFrame:
    """Taxa present in the reference samples, with string names (planted taxa are
    matched by name)."""
    table = table.loc[table.sum(axis=1) > 0].copy()
    table.index = [str(i) for i in table.index]
    table.columns = [str(c) for c in table.columns]
    return table


def pelto_templates(export: str, min_per_group: int) -> list:
    import tier_validation as tv
    index_path = os.path.join(export, "index.csv")
    if not os.path.exists(index_path):
        return []
    tv.EXPORT = export
    index = pd.read_csv(index_path)
    whole = index[index["kind"] == "whole"].copy()
    whole["min_group"] = whole[["n_case", "n_control"]].min(axis=1)
    eligible = whole[whole["min_group"] >= min_per_group].sort_values("n_samples",
                                                                    ascending=False)
    templates = []
    for _, row in eligible.iterrows():
        counts, metadata = tv.read_cohort(row["slug"])
        controls = metadata.index[metadata["group"] == "a_control"]
        table = counts[list(controls)]
        templates.append(("pelto", str(row["study"]), _named(table)))
    return templates


def nearing_templates(min_reference: int, max_features: int) -> list:
    """The paper's filtered tables, restricted to one grouping level each."""
    import tarfile

    import nearing_study as ns
    archive_file = os.path.join(ns.CACHE, ns.ARCHIVE)
    if not os.path.exists(archive_file):
        return []
    info = ns.characteristics()
    paper_n = ns.paper_feature_counts("filtered")
    templates = []
    with tarfile.open(ns.archive_path()) as archive:
        for name in info.index:
            n_paper = paper_n.get(name)
            if n_paper is not None and n_paper > max_features:
                continue
            files = info.loc[name]
            try:
                groups = ns.read_groupings(ns.study_file(archive, name, files,
                                                         "Metadata.File.Name"))
                table = ns.filtered_samples(
                    ns.read_table_prefiltered(ns.study_file(archive, name, files,
                                                            "ASV.Table.File.Name")),
                    groups, ns._depth(info.loc[name, "Rarefaction.depth"]))
            except (KeyError, ValueError) as exc:
                print(f"  nearing {name}: unreadable — {exc}")
                continue
            if table.shape[0] > max_features:
                continue
            labels = groups.loc[table.columns].astype(str)
            counts = labels.value_counts()
            reference = sorted(counts[counts == counts.max()].index)[0]
            if counts[reference] < min_reference:
                continue
            ref = table[list(labels.index[labels == reference])]
            templates.append(("nearing", str(name), _named(ref)))
    return templates


# ---------------------------------------------------------------------------------------
# One calibrated run
# ---------------------------------------------------------------------------------------
def dataset_from(counts: pd.DataFrame, in_b: np.ndarray):
    metadata = pd.DataFrame({"group": np.where(in_b, "b", "a")}, index=counts.columns)
    metadata.index.name = "sample_id"
    table = AbundanceTable(counts=counts.round().astype(float), lineages={},
                           source_format="v9", value_type="counts")
    return validate_dataset(table, metadata, "group")


def calibrated(counts: pd.DataFrame, in_b: np.ndarray, permutations: int, seed: int):
    """What a calibrated web run reports, per taxon name."""
    dataset = dataset_from(counts, in_b)
    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    weights = summary.spec_summary.sort_values("spec_id")["weight"].to_numpy()
    result = inference.calibrate(run, dataset, weights, n_permutations=permutations,
                                 seed=seed, q=Q)
    frame = result.taxa.copy()
    frame["taxon"] = [run.taxa_names[int(j)] for j in frame["taxon_id"]]
    return frame.set_index("taxon")


def score(frame: pd.DataFrame, planted: set, prevalence: pd.Series) -> dict:
    """Everything a replicate contributes to the metrics."""
    is_planted = frame.index.isin(list(planted))
    selected = frame["selected"].to_numpy(dtype=bool)
    certified = frame["certified_robust"].to_numpy(dtype=bool)
    null_p = frame.loc[~is_planted, "p_family"].to_numpy(dtype=float)
    top = prevalence.reindex(frame.index).fillna(-1).idxmax()
    n_planted_tested = int(is_planted.sum())
    return {
        "n_taxa": int(len(frame)),
        "n_planted": n_planted_tested,
        "discoveries": int(selected.sum()),
        "false_discoveries": int((selected & ~is_planted).sum()),
        "true_discoveries": int((selected & is_planted).sum()),
        "certified": int(certified.sum()),
        "certified_true": int((certified & is_planted).sum()),
        "null_taxa": int((~is_planted).sum()),
        "null_family_rejections": int((null_p <= FAMILY_ALPHA).sum()),
        "primary_p": float(frame.loc[top, "p_family"]),
    }


# ---------------------------------------------------------------------------------------
# The three parts
# ---------------------------------------------------------------------------------------
def halves(n: int, rng) -> np.ndarray:
    in_b = np.zeros(n, dtype=bool)
    in_b[rng.permutation(n)[: n // 2]] = True
    return in_b


def plant(counts: np.ndarray, in_b: np.ndarray, taxa: np.ndarray, log2_fold: np.ndarray,
          rng) -> np.ndarray:
    """Planted proportions in group B, renormalised; every sample redrawn at its own
    library size so both groups carry the same multinomial noise."""
    libraries = counts.sum(axis=0).astype(np.int64)
    proportions = counts / np.maximum(libraries, 1)
    shifted = proportions.copy()
    factor = np.ones(counts.shape[0])
    factor[taxa] = 2.0 ** log2_fold
    shifted[:, in_b] = proportions[:, in_b] * factor[:, None]
    shifted /= shifted.sum(axis=0, keepdims=True)
    out = np.empty_like(counts, dtype=np.int64)
    for j in range(counts.shape[1]):
        p = np.clip(shifted[:, j], 0.0, None)
        out[:, j] = rng.multinomial(libraries[j], p / p.sum())
    return out


def planted_taxa(counts: np.ndarray, scenario: str, rng):
    """(taxon rows, log2 fold changes) for one replicate of part (b)."""
    rel = counts / np.maximum(counts.sum(axis=0), 1)
    abundance = rel.mean(axis=1)
    order = np.argsort(-abundance)
    top = order[:PLANT_SKIP_TOP]
    prevalence = (counts > 0).mean(axis=1)
    candidates = np.setdiff1d(np.flatnonzero(prevalence >= PLANT_MIN_PREVALENCE), top)
    k = min(PLANTED, len(candidates) // 3)
    chosen = rng.choice(candidates, size=k, replace=False) if k else np.array([], int)
    folds = rng.choice([1.0, 2.0], size=k) * rng.choice([-1.0, 1.0], size=k)
    if scenario == "dominant_shift":
        dominant = order[:DOMINANT]
        chosen = np.concatenate([chosen, dominant])
        folds = np.concatenate([folds, 2.0 * rng.choice([-1.0, 1.0], size=DOMINANT)])
    return chosen.astype(int), folds


def simulate(n_per_group: int, concentration: float, fold: float, seed: int):
    """Part (c): a Dirichlet-multinomial table with planted fold changes."""
    rng = np.random.default_rng(seed)
    base = np.sort(rng.lognormal(0.0, 1.5, SIM_TAXA))[::-1]
    base /= base.sum()
    planted = rng.choice(np.arange(10, 60), size=PLANTED, replace=False)
    shifted = base.copy()
    shifted[planted] *= fold ** rng.choice([-1.0, 1.0], size=PLANTED)
    shifted /= shifted.sum()
    n = 2 * n_per_group
    in_b = np.arange(n) >= n_per_group
    libraries = np.rint(rng.lognormal(9.5, 0.5, n)).astype(np.int64)
    counts = np.empty((SIM_TAXA, n), dtype=np.int64)
    for j in range(n):
        alpha = np.maximum((shifted if in_b[j] else base) * concentration, 1e-6)
        counts[:, j] = rng.multinomial(libraries[j], rng.dirichlet(alpha))
    names = [f"T{i:03d}" for i in range(SIM_TAXA)]
    frame = pd.DataFrame(counts, index=names, columns=[f"S{j:03d}" for j in range(n)])
    keep = frame.sum(axis=1) > 0
    return frame.loc[keep], in_b, {names[i] for i in planted if keep.iloc[i]}


# ---------------------------------------------------------------------------------------
# Running replicates, with a resumable cache
# ---------------------------------------------------------------------------------------
def commit() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"sha": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


class Cache:
    """Replicate rows keyed by id and code commit, appended as they finish."""

    def __init__(self, path: str, sha: str | None, use: bool):
        self.path, self.sha, self.rows = path, sha, {}
        if use and sha and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("code") == sha:
                        self.rows[row["id"]] = row

    def get(self, key: str):
        return self.rows.get(key)

    def put(self, row: dict) -> None:
        self.rows[row["id"]] = row
        if self.sha:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({**row, "code": self.sha}, default=float) + "\n")


def replicate(cache: Cache, key: str, meta: dict, make, permutations: int, seed: int):
    cached = cache.get(key)
    if cached is not None:
        return cached
    counts, in_b, planted = make()
    prevalence = (counts > 0).mean(axis=1)
    frame = calibrated(counts, in_b, permutations, seed)
    row = {"id": key, **meta, **score(frame, planted, prevalence)}
    cache.put(row)
    return row


def run_part_a(templates, args, cache) -> list:
    rows = []
    for t, (source, name, table) in enumerate(templates):
        for split in range(args.splits_a):
            seed = 9000 + 97 * t + split
            rng = np.random.default_rng(seed)

            def make(table=table, rng=rng):
                return table, halves(table.shape[1], rng), set()
            try:
                rows.append(replicate(cache, f"a|{source}|{name}|{split}",
                                      {"part": "a", "source": source, "dataset": name,
                                       "split": split},
                                      make, args.permutations, seed))
            except Exception as exc:                     # noqa: BLE001 — report, continue
                print(f"  (a) {source} {name} split {split}: SKIPPED — "
                      f"{type(exc).__name__}: {exc}")
        print(f"  (a) {source:8s} {name:32s} done")
    return rows


def run_part_b(templates, args, cache) -> list:
    rows = []
    for t, (source, name, table) in enumerate(templates):
        for scenario in ("standard", "dominant_shift"):
            for rep in range(args.replicates_b):
                seed = 19000 + 97 * t + 7 * rep + (0 if scenario == "standard" else 3)
                rng = np.random.default_rng(seed)

                def make(table=table, rng=rng, scenario=scenario):
                    values = table.to_numpy()
                    in_b = halves(values.shape[1], rng)
                    taxa, folds = planted_taxa(values, scenario, rng)
                    counts = plant(values, in_b, taxa, folds, rng)
                    frame = pd.DataFrame(counts, index=table.index, columns=table.columns)
                    keep = frame.sum(axis=1) > 0
                    planted = {str(table.index[i]) for i in taxa if keep.iloc[i]}
                    return frame.loc[keep], in_b, planted
                try:
                    rows.append(replicate(
                        cache, f"b|{scenario}|{source}|{name}|{rep}",
                        {"part": "b", "scenario": scenario, "source": source,
                         "dataset": name, "replicate": rep},
                        make, args.permutations, seed))
                except Exception as exc:                 # noqa: BLE001
                    print(f"  (b) {scenario} {source} {name} {rep}: SKIPPED — "
                          f"{type(exc).__name__}: {exc}")
        print(f"  (b) {source:8s} {name:32s} done")
    return rows


def run_part_c(args, cache) -> list:
    rows = []
    for n in SIM_N:
        for sparsity, concentration in SIM_SPARSITY.items():
            for fold in SIM_FOLDS:
                for rep in range(args.replicates_c):
                    seed = 29000 + 1000 * n + int(10 * fold) * 10 + rep + (
                        500 if sparsity == "high" else 0)
                    rows.append(replicate(
                        cache, f"c|{n}|{sparsity}|{fold}|{rep}",
                        {"part": "c", "n_per_group": n, "sparsity": sparsity,
                         "fold_change": fold, "replicate": rep},
                        lambda n=n, c=concentration, f=fold, s=seed: simulate(n, c, f, s),
                        args.permutations, seed))
                print(f"  (c) n={n:<3} sparsity={sparsity:<4} fold={fold:<3} done")
    return rows


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def metrics(rows: pd.DataFrame) -> dict:
    """FDR, within-family FWER, power and CERTIFIED ROBUST precision for some rows."""
    if rows.empty:
        return {"n_replicates": 0}
    fdp = rows["false_discoveries"] / rows["discoveries"].clip(lower=1)
    planted = rows["n_planted"] > 0
    power = (rows.loc[planted, "true_discoveries"] / rows.loc[planted, "n_planted"])
    certified = int(rows["certified"].sum())
    return {
        "n_replicates": int(len(rows)),
        "fdr": float(fdp.mean()),
        "any_false_discovery_rate": float((rows["false_discoveries"] > 0).mean()),
        "within_family_fwer": float(rows["null_family_rejections"].sum()
                                    / max(1, rows["null_taxa"].sum())),
        "power": float(power.mean()) if len(power) else None,
        "certified": certified,
        "certified_precision": (float(rows["certified_true"].sum() / certified)
                                if certified else None),
        "mean_discoveries": float(rows["discoveries"].mean()),
    }


def bootstrap(rows: pd.DataFrame, key: str, cluster: str | None, draws: int = 2000,
              seed: int = 5) -> list:
    """95% interval for one metric, resampling clusters (datasets) or replicates."""
    if rows.empty:
        return [None, None]
    groups = ([g for _, g in rows.groupby(cluster)] if cluster
              else [rows.iloc[[i]] for i in range(len(rows))])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        value = metrics(sample).get(key)
        if value is not None:
            values.append(value)
    if not values:
        return [None, None]
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def with_intervals(rows: pd.DataFrame, cluster: str | None) -> dict:
    block = metrics(rows)
    if block["n_replicates"]:
        block["intervals_95"] = {key: bootstrap(rows, key, cluster)
                                 for key in ("fdr", "within_family_fwer", "power")}
    return block


def analyse(rows: pd.DataFrame, parts: str) -> dict:
    out: dict = {}
    if "a" in parts:
        a = rows[rows["part"] == "a"]
        if len(a):
            ks = stats.kstest(a["primary_p"].to_numpy(dtype=float), "uniform")
            a = a.assign(cluster=a["source"] + "|" + a["dataset"])
            out["a"] = {**with_intervals(a, "cluster"),
                        "n_datasets": int(a["cluster"].nunique()),
                        "n_datasets_by_source": a.groupby("source")["dataset"].nunique()
                                                 .astype(int).to_dict(),
                        "ks_statistic": float(ks.statistic), "ks_p": float(ks.pvalue),
                        "any_discovery_rate": float((a["discoveries"] > 0).mean()),
                        "primary": "the most prevalent taxon's family p-value, one per "
                                   "split"}
    if "b" in parts:
        b = rows[rows["part"] == "b"]
        if len(b):
            b = b.assign(cluster=b["source"] + "|" + b["dataset"])
            out["b"] = {"n_datasets": int(b["cluster"].nunique()), "scenarios": {
                scenario: with_intervals(block, "cluster")
                for scenario, block in b.groupby("scenario")}}
    if "c" in parts:
        c = rows[rows["part"] == "c"]
        if len(c):
            cells = [{"n_per_group": int(n), "sparsity": s, "fold_change": float(f),
                      **metrics(block)}
                     for (n, s, f), block in c.groupby(["n_per_group", "sparsity",
                                                        "fold_change"])]
            out["c"] = {"n_replicates": int(len(c)), "simulator": "dirichlet-multinomial "
                        "(the MIDASim port of plan §31.1 does not exist yet)",
                        "overall": with_intervals(c, None), "cells": cells}
    return out


def verdict(parts: dict) -> dict:
    """The pre-registered criteria, each with the value it was judged on."""
    criteria, restricted = [], []

    def add(name, value, threshold, passed):
        criteria.append({"name": name, "value": value, "threshold": threshold,
                         "passed": bool(passed)})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} — {value:.4g} "
              f"against {threshold}")
        if not passed:
            FAILURES.append(name)

    if "a" in parts:
        add("(a) family p-values uniform under mock comparisons (KS p)",
            parts["a"]["ks_p"], f"> {KS_ALPHA}", parts["a"]["ks_p"] > KS_ALPHA)
    if "b" in parts and "standard" in parts["b"]["scenarios"]:
        fdr = parts["b"]["scenarios"]["standard"]["fdr"]
        add("(b) FDR, spiked real templates, no dominant-taxon shift", fdr,
            f"<= {FDR_LIMIT:g}", fdr <= FDR_LIMIT)
        if fdr <= FDR_LIMIT:
            restricted.append("real templates without dominant-taxon shifts")
    if "b" in parts and "dominant_shift" in parts["b"]["scenarios"]:
        block = parts["b"]["scenarios"]["dominant_shift"]
        report("(b) FDR with dominant-taxon shifts (reported, no criterion)",
               f"{block['fdr']:.4g}")
    if "c" in parts:
        fdr = parts["c"]["overall"]["fdr"]
        add("(c) FDR, simulations", fdr, f"<= {FDR_LIMIT:g}", fdr <= FDR_LIMIT)
        passing = [cell for cell in parts["c"]["cells"] if cell["fdr"] <= FDR_LIMIT]
        if fdr <= FDR_LIMIT or passing:
            restricted.append("simulated settings with FDR within the limit: "
                              + ", ".join(f"n={c['n_per_group']} {c['sparsity']} sparsity "
                                          f"FC {c['fold_change']:g}" for c in passing))
    return {"criteria": criteria, "restricted_to": restricted if FAILURES else [],
            "failures": list(FAILURES)}


# ---------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", default=REGISTERED["parts"])
    parser.add_argument("--permutations", type=int, default=REGISTERED["permutations"])
    parser.add_argument("--splits-a", type=int, default=REGISTERED["splits_a"])
    parser.add_argument("--replicates-b", type=int, default=REGISTERED["replicates_b"])
    parser.add_argument("--replicates-c", type=int, default=REGISTERED["replicates_c"])
    parser.add_argument("--pelto-export", default=PELTO_EXPORT)
    parser.add_argument("--max-datasets", type=int, default=0, help="0 = every eligible")
    parser.add_argument("--osf", default="", help="OSF registration URL (a human input)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", default="", help="write the record here (trial runs)")
    args = parser.parse_args(argv)

    templates = []
    if set(args.parts) & {"a", "b"}:
        templates = (pelto_templates(args.pelto_export, REGISTERED["pelto_min_per_group"])
                     + nearing_templates(REGISTERED["nearing_min_reference"],
                                         REGISTERED["max_features"]))
        sources = {t[0] for t in templates}
        if not {"pelto", "nearing"} <= sources:
            missing = {"pelto", "nearing"} - sources
            print("SKIPPED — parts (a) and (b) need both collections; missing: "
                  + ", ".join(sorted(missing)) + ".")
            print("  Pelto:   python tests/reference/pelto_fetch.py, then Rscript "
                  "tests/reference/pelto_export.R data/pelto/data_171023.rds "
                  "data/pelto/export")
            print("  Nearing: python tests/reference/nearing_study.py fetches its archive "
                  "into data/nearing/")
            return 0
        if args.max_datasets:
            templates = templates[: args.max_datasets]

    registered = (args.parts == REGISTERED["parts"]
                  and args.permutations == REGISTERED["permutations"]
                  and args.splits_a == REGISTERED["splits_a"]
                  and args.replicates_b == REGISTERED["replicates_b"]
                  and args.replicates_c == REGISTERED["replicates_c"]
                  and args.pelto_export == PELTO_EXPORT and not args.max_datasets
                  and not args.output)
    code = commit()
    cache = Cache(CACHE if registered else CACHE + ".trial", code["sha"],
                  use=not args.no_cache)

    print("V9: does calibrated inference control error, and with what power?")
    print(f"parts {args.parts}; {len(templates)} templates; B = {args.permutations}\n")
    started = time.perf_counter()
    rows = []
    if "a" in args.parts:
        rows += run_part_a(templates, args, cache)
    if "b" in args.parts:
        rows += run_part_b(templates, args, cache)
    if "c" in args.parts:
        rows += run_part_c(args, cache)
    frame = pd.DataFrame(rows)
    parts = analyse(frame, args.parts)
    print()
    result = verdict(parts)

    payload = {
        "experiment": "V9",
        "script": "tests/reference/v9_calibration.py",
        "preregistration": {"plan": "docs/MICROVERSE_V3_PLAN.md §33, V9",
                            "osf": args.osf or None, **V9_PREREGISTRATION},
        "code_commit": code,
        "registered_design": registered,
        "design": {**REGISTERED, "parts": args.parts, "permutations": args.permutations,
                   "splits_a": args.splits_a, "replicates_b": args.replicates_b,
                   "replicates_c": args.replicates_c, "q": Q, "discovery": "bh",
                   "weights": "decision_tree", "mode": "quick", "ruleset": "v3",
                   "planted": PLANTED, "dominant": DOMINANT},
        "sources": {"pelto": "Pelto et al., Zenodo 10.5281/zenodo.15047338 (control "
                             "samples)",
                    "nearing": "Nearing et al., figshare 14531724 (reference-group "
                               "samples)"},
        "parts": parts,
        "verdict": result,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    output = args.output or (OUTPUT if registered else TRIAL_OUTPUT)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, default=float)
        handle.write("\n")
    written = [output]
    if registered:
        os.makedirs(os.path.dirname(APP_COPY), exist_ok=True)
        shutil.copyfile(output, APP_COPY)
        written.append(APP_COPY)
    print("\nwrote " + ", ".join(_shown(p) for p in written))
    print("FAILED: " + ", ".join(FAILURES) if FAILURES else "All criteria met.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
