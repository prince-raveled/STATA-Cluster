"""SPEC §23 validation 3 — reproduce Pelto et al. on Pelto et al.'s own cohorts.

    "Reproduce Pelto et al. `github.com/jepelt/DAA_replicability`
     (Zenodo 10.5281/zenodo.15047338). Check elementary methods show tighter
     specification distributions in your framework too."  — SPEC §23.3

Pelto et al. ("Elementary methods provide more replicable results in microbial
differential abundance analysis", arXiv:2404.02691) compared 14 DA methods over 61
curated case-control cohorts — 32 16S (MicrobiomeHD-derived) and 29 shotgun (from
curatedMetagenomicData) — and split each eligible study in half five times to measure
replicability directly. Their conclusion: the elementary methods (Wilcoxon, t-test,
linear/logistic regression) replicate better than the sophisticated ones.

Their curated data ships in the archive and is read by `pelto_export.R`. This script
runs MicroVerse over it and asks two questions:

  A. Do elementary methods produce *tighter specification distributions*? — the literal
     §23.3 check. This has to be done carefully, because SPEC §11 does not give every
     method the same forks: ANCOM-BC and DESeq2 refuse rarefaction and every transform
     but raw, so a naive comparison would score them "tight" purely because they were
     asked fewer questions. Everything here is therefore computed on a *matched*
     sub-grid: the same matrices, the same taxa, the same FDR settings for every method
     being compared.

  B. Do they replicate better across Pelto's own split halves? — their actual design,
     which is immune to the fork-coverage problem because each method is compared only
     with itself on the other half of the same study.

    .venv/Scripts/python tests/reference/pelto_study.py
    .venv/Scripts/python tests/reference/pelto_study.py --max-cohorts 6 --no-splits
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import zip_longest

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import app.core.grid as grid  # noqa: E402

# Full mode samples matrices for the sophisticated methods to keep the web app inside
# its compute budget (§8). For a matched comparison the pool must be complete, or the
# methods would differ by how much of the grid they were given.
grid.FULL_MODE_MATRIX_FRACTION = 1.0

import app.core.runner as runner  # noqa: E402
from app.core.parsers.base import AbundanceTable  # noqa: E402
from app.core.validation import validate_dataset  # noqa: E402

EXPORT = os.path.join(ROOT, "data", "pelto", "export")
OUTPUT = os.path.join(ROOT, "docs", "pelto_study.json")

#: Pelto's split, mapped onto SPEC §10's seven methods.
ELEMENTARY = ("wilcoxon", "ttest", "linear", "logistic")
SOPHISTICATED = ("ancombc", "aldex2", "pydeseq2")

#: Sub-grid A1: raw counts, no rarefaction — the only cell where all seven methods are
#: valid under §11, so prevalence filter, rank and FDR vary identically for each.
#: Sub-grid A2: raw counts, rarefaction free — five methods admit it (§11 forbids it for
#: ANCOM-BC and DESeq2), and it adds the fork that dominated the MicrobiomeHD study.
A2_METHODS = ("wilcoxon", "ttest", "linear", "logistic", "aldex2")

#: Split-half runs only ever use the unrarefied sub-grid, so building the rarefied
#: matrices for them is pure waste. Toggled around `study_split`.
UNRAREFIED_ONLY = False

#: Size of the top list compared between halves — see study_split for why a fixed-size
#: list is used instead of an FDR-significant set.
TOP_K = 20

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


# ---------------------------------------------------------------------------
def _matched_enumeration(original):
    """Enumerate the full grid, then keep only untransformed matrices.

    Everything downstream of `enumerate_grid` — matrix building, fitting, effect
    harmonisation — is the app's own code path, unmodified.
    """

    def enumerate_grid(builder, mode="quick", **kwargs):
        specs, grid_report = original(builder, mode="full", **kwargs)
        keep = [s for s in specs if s.transform == "raw"
                and not (UNRAREFIED_ONLY and s.rarefaction != "none")]
        grid_report.n_valid = len(keep)
        return keep, grid_report

    return enumerate_grid


runner.enumerate_grid = _matched_enumeration(runner.enumerate_grid)


def load_cohort(slug: str):
    """Read one exported cohort into a validated Dataset."""
    folder = os.path.join(EXPORT, slug)
    counts = pd.read_csv(os.path.join(folder, "counts.tsv"), sep="\t", index_col=0)
    metadata = pd.read_csv(os.path.join(folder, "meta.tsv"), sep="\t", index_col=0)
    # Sort so "case" is group B: a positive effect then means "higher in cases".
    metadata["group"] = metadata["group"].map({"control": "a_control", "case": "b_case"})
    metadata = metadata.dropna(subset=["group"])
    counts = counts[[c for c in counts.columns if c in set(metadata.index)]]
    counts = counts.loc[counts.sum(axis=1) > 0]
    table = AbundanceTable(counts=counts, lineages={}, source_format="pelto",
                           value_type="counts")
    return validate_dataset(table, metadata.loc[counts.columns, ["group"]], "group")


def _per_taxon(long: pd.DataFrame, specs: pd.DataFrame, methods, taxa_names,
               min_specs: int = 4) -> dict:
    """Per method: one row per taxon summarising its specification distribution.

    Note what is *not* used here. SPEC §14 harmonises the effect size deliberately:
    significance comes from the method, the effect from one method-independent
    estimator. `effect_h` is therefore identical for every method on a given matrix and
    cannot distinguish them — measuring its spread compares nothing. What varies by
    method is the evidence: the p-value, the resulting call, and the method's own
    native effect. Those are what is summarised.
    """
    joined = long.merge(specs[["spec_id", "method"]], on="spec_id", how="inner")
    joined = joined.assign(neglog10p=-np.log10(
        np.clip(joined["p_raw"].to_numpy(dtype=float), 1e-300, 1.0)))
    out = {}
    for method in methods:
        part = joined[joined["method"] == method]
        if part.empty:
            continue
        grouped = part.groupby("taxon")
        frame = pd.DataFrame({
            "n_specs": grouped["effect_n"].size(),
            "sd_neglog10p": grouped["neglog10p"].std(ddof=1),
            "mean_neglog10p": grouped["neglog10p"].mean(),
            "sd_effect_native": grouped["effect_n"].std(ddof=1),
            "mean_effect_native": grouped["effect_n"].mean(),
            "frac_sig": grouped["significant"].mean(),
            "frac_positive": grouped["effect_n"].apply(lambda v: float((v > 0).mean())),
        })
        frame = frame[frame["n_specs"] >= min_specs]
        frame["sign_consistency"] = np.maximum(frame["frac_positive"],
                                               1.0 - frame["frac_positive"])
        # Native effects are on seven different scales (§10), so only a scale-free
        # ratio is comparable between methods.
        frame["cv_effect_native"] = (
            frame["sd_effect_native"]
            / frame["mean_effect_native"].abs().clip(lower=1e-6))
        frame["name"] = [taxa_names[i] for i in frame.index]
        out[method] = frame
    return out


def _summarise(per_method: dict, methods) -> dict:
    """Collapse to one number per method, over the taxa every method reported."""
    present = [m for m in methods if m in per_method]
    if len(present) < 2:
        return {}
    shared = set(per_method[present[0]].index)
    for m in present[1:]:
        shared &= set(per_method[m].index)
    if not shared:
        return {}
    shared = sorted(shared)

    summary = {"n_taxa_matched": len(shared), "methods": {}}
    for m in present:
        frame = per_method[m].loc[shared]
        unstable = frame["frac_sig"].between(0.0, 1.0, inclusive="neither")
        detected = frame["frac_sig"] > 0
        summary["methods"][m] = {
            "n_specs_per_taxon": int(frame["n_specs"].median()),
            "median_sd_neglog10p": float(frame["sd_neglog10p"].median()),
            "median_cv_effect_native": float(frame["cv_effect_native"].median()),
            "mean_frac_significant": float(frame["frac_sig"].mean()),
            "mean_sign_consistency": float(frame["sign_consistency"].mean()),
            "frac_taxa_detected": float(detected.mean()),
            "frac_detected_unstable": (
                float(unstable[detected].mean()) if detected.any() else 0.0),
        }
    return summary


def study_whole(slug: str, row) -> dict:
    dataset = load_cohort(slug)
    started = time.perf_counter()
    run = runner.run_multiverse(dataset, mode="full")
    elapsed = time.perf_counter() - started

    specs = run.specs_frame
    long = run.long
    unrarefied = set(specs.loc[specs["rarefaction"] == "none", "spec_id"])

    a1 = _summarise(
        _per_taxon(long[long["spec_id"].isin(unrarefied)], specs,
                   ELEMENTARY + SOPHISTICATED, run.taxa_names, min_specs=4),
        ELEMENTARY + SOPHISTICATED,
    )
    a2 = _summarise(
        _per_taxon(long, specs, A2_METHODS, run.taxa_names, min_specs=8),
        A2_METHODS,
    )
    return {
        "slug": slug,
        "study": row["study"],
        "disease": row["disease"],
        "type": row["type"],
        "n_samples": int(dataset.n_samples),
        "n_taxa": int(dataset.n_taxa),
        "n_case": int(row["n_case"]),
        "n_control": int(row["n_control"]),
        "n_specifications": int(run.grid_report.n_valid),
        "runtime_seconds": round(elapsed, 2),
        "matched_all_methods": a1,
        "matched_rarefiable": a2,
    }


def study_split(slug_a: str, slug_b: str, row) -> dict:
    """Pelto's own design: does a method's answer survive re-running on the other half?"""
    global UNRAREFIED_ONLY
    results = {}
    UNRAREFIED_ONLY = True
    try:
        for label, slug in (("a", slug_a), ("b", slug_b)):
            dataset = load_cohort(slug)
            run = runner.run_multiverse(dataset, mode="full")
            results[label] = _per_taxon(run.long, run.specs_frame,
                                        ELEMENTARY + SOPHISTICATED,
                                        run.taxa_names, min_specs=4)
    finally:
        UNRAREFIED_ONLY = False

    out = {"study": row["study"], "disease": row["disease"], "type": row["type"],
           "iter": int(row["iter"]), "methods": {}}
    for method in ELEMENTARY + SOPHISTICATED:
        if method not in results["a"] or method not in results["b"]:
            continue
        left = results["a"][method].set_index("name")
        right = results["b"][method].set_index("name")
        shared = sorted(set(left.index) & set(right.index))
        if len(shared) < 10:
            continue
        left, right = left.loc[shared], right.loc[shared]

        # A taxon "called" by the multiverse: significant in most specifications.
        called_a = left["frac_sig"] >= 0.5
        called_b = right["frac_sig"] >= 0.5
        same_sign = np.sign(left["mean_effect_native"]) == np.sign(
            right["mean_effect_native"])

        rho = float(pd.Series(left["mean_effect_native"].to_numpy()).corr(
            pd.Series(right["mean_effect_native"].to_numpy()), method="spearman"))
        p_rho = float(pd.Series(left["mean_neglog10p"].to_numpy()).corr(
            pd.Series(right["mean_neglog10p"].to_numpy()), method="spearman"))
        replicated = (
            float((called_b & same_sign)[called_a].mean()) if called_a.any() else np.nan)
        either = called_a | called_b
        jaccard = (
            float((called_a & called_b & same_sign).sum() / either.sum())
            if either.any() else np.nan)

        # Halves are small — 25 to 60 samples — and at BH 0.05 most methods call
        # nothing at all in most halves, so a rate conditioned on "called in half A"
        # is computed on a different, signal-selected subset of pairs for every
        # method and is not comparable between them. A fixed-size top list is:
        # every method ranks all the same taxa, so every method is scored on every
        # pair, whatever its calibration.
        top_k = min(TOP_K, max(5, len(shared) // 5))
        rank_a = left["mean_neglog10p"].rank(ascending=False, method="first")
        rank_b = right["mean_neglog10p"].rank(ascending=False, method="first")
        top_a = set(left.index[rank_a <= top_k])
        top_b = set(right.index[rank_b <= top_k])
        shared_top = top_a & top_b
        top_overlap = len(shared_top) / top_k
        top_same_sign = (
            float(same_sign.loc[sorted(shared_top)].mean()) if shared_top else np.nan)

        out["methods"][method] = {
            "n_taxa": len(shared),
            "n_called_a": int(called_a.sum()),
            "n_called_b": int(called_b.sum()),
            "top_k": int(top_k),
            "top_overlap": top_overlap,
            "top_same_sign": top_same_sign,
            "effect_spearman": rho,
            "evidence_spearman": p_rho,
            "replication_rate": replicated,
            "called_jaccard": jaccard,
        }
    return out


# ---------------------------------------------------------------------------
def _pool(records: list, block: str, key: str, methods) -> dict:
    values = {m: [] for m in methods}
    for rec in records:
        summary = rec.get(block) or {}
        for m, stats_ in (summary.get("methods") or {}).items():
            if m in values and np.isfinite(stats_.get(key, np.nan)):
                values[m].append(stats_[key])
    return {m: v for m, v in values.items() if v}


def _group_means(pooled: dict, group) -> float:
    present = [pooled[m] for m in group if m in pooled]
    return float(np.mean([np.mean(v) for v in present])) if present else float("nan")


def _paired_test(records: list, block: str, key: str) -> tuple:
    """Elementary vs sophisticated, paired within cohort — cohorts differ wildly."""
    left, right = [], []
    for rec in records:
        methods = ((rec.get(block) or {}).get("methods") or {})
        e = [methods[m][key] for m in ELEMENTARY
             if m in methods and np.isfinite(methods[m].get(key, np.nan))]
        s = [methods[m][key] for m in SOPHISTICATED
             if m in methods and np.isfinite(methods[m].get(key, np.nan))]
        if e and s:
            left.append(np.mean(e))
            right.append(np.mean(s))
    if len(left) < 6:
        return float("nan"), float("nan"), len(left), float("nan")
    _, p = stats.wilcoxon(left, right)
    return float(np.mean(left)), float(np.mean(right)), len(left), float(p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cohorts", type=int, default=0,
                        help="limit the whole-study cohorts (0 = all)")
    parser.add_argument("--max-splits", type=int, default=40,
                        help="limit the split-half pairs (0 = all)")
    parser.add_argument("--no-splits", action="store_true")
    parser.add_argument("--iters", default="1",
                        help="comma-separated split iterations to use")
    parser.add_argument("--output", default=OUTPUT,
                        help="where to write the record (default docs/pelto_study.json)")
    args = parser.parse_args()

    index_path = os.path.join(EXPORT, "index.csv")
    if not os.path.exists(index_path):
        print("SKIPPED — Pelto cohorts not exported. Run:")
        print("  python tests/reference/pelto_fetch.py")
        print("  Rscript tests/reference/pelto_export.R "
              "data/pelto/data_171023.rds data/pelto/export")
        return 0
    index = pd.read_csv(index_path)

    whole = index[index["kind"] == "whole"].reset_index(drop=True)
    if args.max_cohorts:
        whole = whole.head(args.max_cohorts)

    print(f"Pelto et al. cohorts: {len(whole)} whole studies "
          f"({(whole['type'] == '16S').sum()} 16S, {(whole['type'] != '16S').sum()} shotgun)")
    print("Matched sub-grids only: same matrices, same taxa, same FDR settings per method.")
    print()

    records, started = [], time.perf_counter()
    for i, row in whole.iterrows():
        try:
            record = study_whole(row["slug"], row)
        except Exception as exc:                        # noqa: BLE001 — report, continue
            print(f"  [{i + 1}/{len(whole)}] {row['study']}: SKIPPED — {exc}")
            continue
        records.append(record)
        a1 = record["matched_all_methods"].get("methods", {})
        elem = np.mean([a1[m]["median_sd_neglog10p"] for m in ELEMENTARY if m in a1] or [np.nan])
        soph = np.mean([a1[m]["median_sd_neglog10p"] for m in SOPHISTICATED if m in a1] or [np.nan])
        print(f"  [{i + 1}/{len(whole)}] {row['study']:<28} {record['n_specifications']:>5} specs "
              f"{record['runtime_seconds']:>6.1f}s   sd(-log10 p) elementary {elem:.3f} "
              f"vs sophisticated {soph:.3f}")

    splits = []
    if not args.no_splits:
        wanted = {int(x) for x in args.iters.split(",") if x.strip()}
        halves = index[(index["kind"] == "half") & (index["iter"].isin(wanted))]
        pairs = []
        for _, block in halves.groupby(["study", "iter"]):
            if len(block) == 2:
                first, second = block.sort_values("half")["slug"].tolist()
                pairs.append((first, second, block.iloc[0]))
        # groupby sorts by study name, and Pelto's shotgun studies are capitalised
        # while the 16S ones are not — so a plain head() of the pairs is 100% shotgun.
        # Interleave the two data types instead, so --max-splits samples both.
        by_type = {}
        for pair in pairs:
            by_type.setdefault(pair[2]["type"], []).append(pair)
        interleaved = []
        for group in zip_longest(*by_type.values()):
            interleaved.extend(x for x in group if x is not None)
        pairs = interleaved
        if args.max_splits:
            pairs = pairs[: args.max_splits]
        print(f"\nsplit-half replicability: {len(pairs)} study/iteration pairs")
        for i, (first, second, row) in enumerate(pairs):
            try:
                splits.append(study_split(first, second, row))
            except Exception as exc:                    # noqa: BLE001
                print(f"  [{i + 1}/{len(pairs)}] {row['study']}: SKIPPED — {exc}")
                continue
            got = splits[-1]["methods"]
            elem = np.mean([got[m]["replication_rate"] for m in ELEMENTARY
                            if m in got and np.isfinite(got[m]["replication_rate"])] or [np.nan])
            soph = np.mean([got[m]["replication_rate"] for m in SOPHISTICATED
                            if m in got and np.isfinite(got[m]["replication_rate"])] or [np.nan])
            print(f"  [{i + 1}/{len(pairs)}] {row['study']:<28} "
                  f"replication elementary {elem:.2f} vs sophisticated {soph:.2f}")

    elapsed = time.perf_counter() - started
    print(f"\n{len(records)} whole cohorts + {len(splits)} split pairs in {elapsed:.0f}s")

    # --- A. tighter specification distributions? --------------------------
    print("\nA. Specification dispersion, matched sub-grid (all seven methods,")
    print("   raw counts, no rarefaction — the only cell §11 gives every method)")
    pooled = _pool(records, "matched_all_methods", "median_sd_neglog10p",
                   ELEMENTARY + SOPHISTICATED)
    for method in ELEMENTARY + SOPHISTICATED:
        if method in pooled:
            group = "elementary" if method in ELEMENTARY else "sophisticated"
            report(f"{method:<9} ({group})",
                   f"median SD of -log10 p across specs = {np.mean(pooled[method]):.4f} "
                   f"over {len(pooled[method])} cohorts")
    elem, soph, n_paired, p_value = _paired_test(
        records, "matched_all_methods", "median_sd_neglog10p")
    if np.isfinite(elem) and np.isfinite(soph):
        check("elementary methods give tighter specification distributions",
              elem < soph,
              f"mean SD {elem:.4f} vs {soph:.4f} over {n_paired} paired cohorts "
              f"(Wilcoxon signed-rank p = {p_value:.3g})")
    else:
        report("elementary vs sophisticated dispersion",
               f"only {n_paired} cohorts had both groups; not tested")

    print("\n   the same, on the sub-grid that includes rarefaction (five methods)")
    pooled2 = _pool(records, "matched_rarefiable", "median_sd_neglog10p", A2_METHODS)
    for method in A2_METHODS:
        if method in pooled2:
            group = "elementary" if method in ELEMENTARY else "sophisticated"
            report(f"{method:<9} ({group})",
                   f"median SD of -log10 p = {np.mean(pooled2[method]):.4f}")

    # --- B. replicability across Pelto's own splits -----------------------
    if splits:
        print()
        print("B. Split-half replicability (Pelto's own design, their splits)")
        coverage = {}
        for key, label in (("top_overlap", "top-20 overlap between halves"),
                           ("top_same_sign", "of those, same direction"),
                           ("evidence_spearman", "evidence rank correlation"),
                           ("effect_spearman", "effect rank correlation"),
                           ("replication_rate", "replication rate (FDR-called)"),
                           ("called_jaccard", "called-set Jaccard (FDR-called)")):
            values = {m: [] for m in ELEMENTARY + SOPHISTICATED}
            for rec in splits:
                for m, got in rec["methods"].items():
                    if m in values and np.isfinite(got.get(key, np.nan)):
                        values[m].append(got[key])
            values = {m: v for m, v in values.items() if v}
            if not values:
                continue
            coverage[key] = {m: len(v) for m, v in values.items()}
            elem = _group_means(values, ELEMENTARY)
            soph = _group_means(values, SOPHISTICATED)
            detail = "  ".join(f"{m}={np.mean(v):.3f}" for m, v in values.items())
            report(label, f"elementary {elem:.3f} vs sophisticated {soph:.3f}")
            report("", detail)

        # The FDR-based measures are only defined on pairs where the method called
        # something, and at these half-sizes that is a minority of pairs — a different
        # minority for each method. Say so rather than averaging across them silently.
        for key in ("replication_rate", "called_jaccard"):
            if key in coverage:
                spread = coverage[key]
                report(f"{key} was computable on",
                       ", ".join(f"{m} {n}/{len(splits)}"
                                 for m, n in sorted(spread.items())))

        overlap = {m: [] for m in ELEMENTARY + SOPHISTICATED}
        for rec in splits:
            for m, got in rec["methods"].items():
                if m in overlap and np.isfinite(got.get("top_overlap", np.nan)):
                    overlap[m].append(got["top_overlap"])
        overlap = {m: v for m, v in overlap.items() if v}
        elem = _group_means(overlap, ELEMENTARY)
        soph = _group_means(overlap, SOPHISTICATED)
        if np.isfinite(elem) and np.isfinite(soph):
            check("elementary methods replicate better across halves", elem > soph,
                  f"top-20 overlap {elem:.3f} vs {soph:.3f} over {len(splits)} pairs "
                  f"(every method scored on every pair)")

    payload = {
        "source": "Pelto et al., arXiv:2404.02691; Zenodo 10.5281/zenodo.15047338",
        "n_whole_cohorts": len(records),
        "n_split_pairs": len(splits),
        "elementary": list(ELEMENTARY),
        "sophisticated": list(SOPHISTICATED),
        "cohorts": records,
        "splits": splits,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print()
    print(f"wrote {args.output}")

    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "All checks passed."))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
