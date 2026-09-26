"""Validation against published findings — MicrobiomeHD (Duvallet et al. 2017).

The other studies here ask whether MicroVerse's *statistics* behave (instability as
Tierney measured it, tiers across discovery/validation splits of one cohort). This one
asks the questions a reader of a paper would ask:

1. **Reproduction.** Given a paper's data, does MicroVerse's specification that matches
   the paper's pipeline find what the paper found? The pipeline is re-implemented from
   the authors' own code (`src/data/clean_otu_and_metadata.py`, `src/util/util.py` in
   github.com/cduvallet/microbiomeHD): samples with <=100 reads, OTUs with <10 reads and
   OTUs in <=1% of samples dropped; relative abundance; collapse to genus, discarding
   OTUs without one; Kruskal-Wallis (`scipy.stats.mstats`); Benjamini-Hochberg, q < 0.05.
   Compared at two levels: the same test on MicroVerse's own matrix (implementation), and
   the paper's full pipeline (published analysis).

2. **Published claims.** Supplementary File 3 of the paper (`file-S3.nonspecific_genera
   .txt`, in the Zenodo deposit) labels 24 genera health-associated and 20 disease-
   associated across diseases; the text names the genera enriched in colorectal cancer.
   Does MicroVerse agree on direction, and how robust does it find them?

3. **Replication.** For each disease with three or more cohorts, a finding in one cohort
   is checked in every other cohort of that disease with the paper's own pipeline. Does
   a MicroVerse tier in the discovery cohort predict replication elsewhere — and better
   than "significant under the pipeline the paper used"?

Data: Zenodo 1146764 (CC-BY-NC-4.0), downloaded on demand into the git-ignored cache.
Nothing from MicrobiomeHD is committed; the record written is `docs/published_findings_
study.json`.

    .venv/Scripts/python tests/reference/published_findings_study.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tarfile
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mstats
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from app.core.parsers.base import AbundanceTable, split_lineage  # noqa: E402
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.validation import DatasetError, validate_dataset  # noqa: E402
from tests.reference.microbiomehd import CACHE, download  # noqa: E402

OUTPUT = os.path.join(ROOT, "docs", "published_findings_study.json")

#: Diseases with three or more MicrobiomeHD cohorts, so every finding has at least two
#: independent cohorts to replicate in. Names are the Zenodo folder stems.
DISEASES = {
    "CRC": ["crc_baxter", "crc_zeller", "crc_zhao", "crc_xiang", "crc_zackular"],
    "CDI": ["cdi_schubert", "cdi_vincent_v3v5", "cdi_youngster"],
    "IBD": ["ibd_gevers_2014", "ibd_huttenhower", "ibd_alm", "ibd_engstrand_maxee"],
    "HIV": ["hiv_dinh", "hiv_lozupone", "hiv_noguerajulian"],
    "OB": ["ob_goodrich", "ob_ross", "ob_zupancic", "ob_escobar", "ob_gordon_2008_v2"],
}

#: `FileIO.get_classes` in the authors' code: every listed disease label is a case,
#: pooled; `H` and `nonIBD` are controls; anything else is left out.
CONTROLS = {"H", "nonIBD"}
CASES = {"ASD", "CD", "CDI", "nonCDI", "CIRR", "CRC", "EDD", "HIV", "MHE", "NASH", "OB",
         "PAR", "PSA", "RA", "T1D", "T2D", "UC"}
#: `clean_otu_and_metadata.fix_cdi_schubert` relabels nonCDI so only CDI is a case there.
RELABEL = {"cdi_schubert": {"nonCDI": "ignore-nonCDI"}}

#: The MicroVerse specification that matches the paper's pipeline.
MATCHED = {"rarefaction": "none", "prev_filter": 0.0, "transform": "tss",
           "method": "wilcoxon", "fdr_method": "bh", "fdr_threshold": 0.05}

#: Named in the paper's results as enriched in colorectal cancer.
CRC_ENRICHED = ["Fusobacterium", "Porphyromonas", "Peptostreptococcus", "Parvimonas",
                "Enterobacter"]

TIERED = ["ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE"]
Q = 0.05


# ---------------------------------------------------------------------------- inputs
def conditions() -> dict:
    """Per-folder sample conditions from the deposit's dataset_info.yaml.

    Read with a small parser rather than PyYAML, which is not a MicroVerse dependency;
    every `condition:` in the file is one key with a list of accepted values.
    """
    download_file("dataset_info.yaml")
    with open(os.path.join(CACHE, "dataset_info.yaml"), encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    found, folder = {}, None
    for i, line in enumerate(lines):
        if re.match(r"^[A-Za-z0-9_]+:\s*$", line):
            folder = None
        match = re.match(r"^\s+folder:\s*(\S+)", line)
        if match:
            folder = match.group(1).removesuffix("_results")
        if re.match(r"^\s+condition:\s*$", line) and folder:
            key, values = lines[i + 1].split(":", 1)
            accepted = [v.strip().strip("'\"") for v in values.strip().strip("[]").split(",")]
            found[folder] = (key.strip().strip("'\""), accepted)
    return found


def download_file(name: str) -> str:
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        import urllib.request
        url = f"https://zenodo.org/api/records/1146764/files/{name}/content"
        request = urllib.request.Request(url, headers={"User-Agent": "microverse-validation"})
        with urllib.request.urlopen(request, timeout=120) as response, open(path, "wb") as out:
            out.write(response.read())
    return path


def load_raw(cohort: str):
    """(samples x OTUs counts, metadata) straight from the archive."""
    with tarfile.open(download(cohort)) as archive:
        # The archives were made on a Mac and carry AppleDouble `._` twins of each file,
        # which end in the same suffix but hold resource-fork bytes, not the table.
        members = {m.name: m for m in archive.getmembers()
                   if not os.path.basename(m.name).startswith("._")}
        table_name = next(n for n in members if n.endswith("rdp_assigned"))
        meta_name = next(n for n in members if n.endswith("metadata.txt"))
        table = pd.read_csv(archive.extractfile(members[table_name]), sep="\t", index_col=0,
                            encoding="utf-8", encoding_errors="replace")
        meta = pd.read_csv(io.BytesIO(archive.extractfile(members[meta_name]).read()), sep="\t",
                           index_col=0, encoding="utf-8", encoding_errors="replace",
                           low_memory=False, dtype=str)
    meta.index = [str(i).strip() for i in meta.index]
    meta.columns = [str(c).strip() for c in meta.columns]
    table.columns = [str(c).strip() for c in table.columns]
    counts = table.apply(pd.to_numeric, errors="coerce").fillna(0.0).T
    return counts, meta


# ------------------------------------------------------------- the paper's pipeline
def paper_clean(counts: pd.DataFrame, meta: pd.DataFrame, cohort: str, condition):
    """`clean_up_samples` then `clean_up_tables`, in the authors' order."""
    if condition:
        column, accepted = condition
        meta = meta[meta[column].map(lambda v: _same_value(v, accepted))]
    df = counts.loc[[s for s in counts.index if s in meta.index]]
    df = df.loc[df.sum(axis=1) > 100]                              # remove_shallow_smpls
    df = df.loc[:, df.sum(axis=0) >= 10]                           # OTUs with < 10 reads
    df = df.loc[:, (df > 0).sum(axis=0) / df.shape[0] > 0.01]      # OTUs in <= 1% samples
    df = df.loc[df.sum(axis=1) > 100]
    meta = meta.loc[df.index]
    states = meta["DiseaseState"].astype(str).str.strip().replace(RELABEL.get(cohort, {}))
    return df, states


def _same_value(value, accepted) -> bool:
    """YAML `[0]` must match a metadata cell reading `0.0`, as pandas matched it."""
    text = str(value).strip()
    for target in accepted:
        if text == target:
            return True
        try:
            if float(text) == float(target):
                return True
        except ValueError:
            pass
    return False


def collapse_genus(df: pd.DataFrame) -> pd.DataFrame:
    """`collapse_taxonomic_contents_df(df, 'genus')`: first six ranks, unannotated dropped."""
    keys = [";".join(str(c).split(";")[:6]) for c in df.columns]
    keep = [not k.endswith("__") and len(k.split(";")) == 6 for k in keys]
    sub = df.loc[:, keep].copy()
    sub.columns = [k for k, ok in zip(keys, keep, strict=True) if ok]
    return sub.T.groupby(level=0, sort=False).sum().T


def kruskal_bh(abundance: pd.DataFrame, controls, cases) -> pd.DataFrame:
    """`compare_otus_teststat(method='kruskal-wallis', multi_comp='fdr')`."""
    p = []
    for genus in abundance.columns:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, value = mstats.kruskalwallis(abundance.loc[controls, genus].to_numpy(),
                                                abundance.loc[cases, genus].to_numpy())
            value = float(value)
            if not np.isfinite(value):
                value = 1.0
        except Exception:  # the authors' code sets p = 1 whenever the test fails
            value = 1.0
        p.append(value)
    frame = pd.DataFrame({"p": p}, index=abundance.columns)
    frame["q"] = multipletests(frame["p"], method="fdr_bh")[1]
    frame["effect"] = abundance.loc[cases].mean() - abundance.loc[controls].mean()
    return frame


def genus_name(lineage: str) -> str:
    return lineage.split(";")[-1].replace("g__", "")


# ------------------------------------------------------------------------- per cohort
def study_cohort(cohort: str, condition) -> dict:
    counts, meta = load_raw(cohort)
    df, states = paper_clean(counts, meta, cohort, condition)
    controls = list(states[states.isin(CONTROLS)].index)
    cases = list(states[states.isin(CASES)].index)
    case_labels = sorted(set(states[states.isin(CASES)]))

    # The paper's full pipeline: relative abundance over every read, then genus.
    paper = kruskal_bh(collapse_genus(df.div(df.sum(axis=1), axis=0)), controls, cases)

    # MicroVerse gets the same cleaned table, as genus counts: what an author would upload.
    genus_counts = collapse_genus(df.loc[controls + cases])
    genus_counts = genus_counts.loc[:, genus_counts.sum(axis=0) > 0]
    metadata = pd.DataFrame(
        {"group": ["a_control"] * len(controls) + ["b_case"] * len(cases)},
        index=controls + cases,
    )
    abundance = AbundanceTable(
        counts=genus_counts.T,
        lineages={t: split_lineage(t) for t in genus_counts.columns},
        source_format="microbiomehd",
        value_type="counts",
    )
    dataset = validate_dataset(abundance, metadata, "group")

    started = time.perf_counter()
    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    runtime = time.perf_counter() - started

    matched = run.specs_frame
    for key, value in MATCHED.items():
        column = matched[key]
        matched = matched[np.isclose(column, value)] if isinstance(value, float) \
            else matched[column.astype(str) == value]
    if len(matched) != 1:
        raise RuntimeError(f"{cohort}: {len(matched)} specifications match the paper's")
    matched_id = int(matched["spec_id"].iloc[0])
    rows = run.long[run.long["spec_id"] == matched_id]
    names = np.asarray(run.taxa_names, dtype=object)
    mv = pd.DataFrame({
        "p": rows["p_raw"].to_numpy(dtype=float),
        "q": rows["p_adj"].to_numpy(dtype=float),
        "significant": rows["significant"].to_numpy(dtype=bool),
        "effect": rows["effect_h"].to_numpy(dtype=float),
    }, index=names[rows["taxon"].to_numpy()])

    table = summary.table.set_index("taxon")

    # Level 1 — implementation: the paper's test and FDR on MicroVerse's own matrix.
    tss = genus_counts.div(genus_counts.sum(axis=1), axis=0)[mv.index]
    same_matrix = kruskal_bh(tss, controls, cases)
    with np.errstate(invalid="ignore"):
        mw = stats.mannwhitneyu(tss.loc[cases].to_numpy(), tss.loc[controls].to_numpy(),
                                axis=0, alternative="two-sided", method="asymptotic")
    level1 = {
        "max_abs_diff_p_vs_scipy_mannwhitney": float(
            np.nanmax(np.abs(mw.pvalue - mv["p"].to_numpy()))),
        "n_sig_kruskal_same_matrix": int((same_matrix["q"] < Q).sum()),
        "n_sig_microverse": int(mv["significant"].sum()),
        "agreement_same_matrix": float(((same_matrix["q"] < Q) == mv["significant"]).mean()),
    }

    # Level 2 — published pipeline vs MicroVerse's matched specification.
    common = paper.index.intersection(mv.index)
    a = paper.loc[common, "q"] < Q
    b = mv.loc[common, "significant"]
    both = a & b
    level2 = {
        "n_genera_paper": int(len(paper)),
        "n_genera_microverse": int(len(mv)),
        "n_common": int(len(common)),
        "n_sig_paper": int((paper["q"] < Q).sum()),
        "n_sig_matched": int(b.sum()),
        "n_sig_both": int(both.sum()),
        "jaccard": float(both.sum() / max(1, (a | b).sum())),
        "agreement": float((a == b).mean()) if len(common) else float("nan"),
        "direction_agreement_on_shared": float(
            (np.sign(paper.loc[common[both], "effect"])
             == np.sign(mv.loc[common[both], "effect"])).mean()
        ) if both.any() else float("nan"),
        "spearman_p": float(stats.spearmanr(paper.loc[common, "p"], mv.loc[common, "p"]).statistic),
    }

    record = pd.DataFrame(index=paper.index.union(mv.index))
    record["genus"] = [genus_name(g) for g in record.index]
    record["paper_q"] = paper["q"]
    record["paper_effect"] = paper["effect"]
    record["matched_significant"] = mv["significant"]
    record["tier"] = table["robustness_tier"].reindex(record.index)
    record["frac_significant"] = table["frac_significant"].reindex(record.index)
    record["median_effect"] = table["median_effect"].reindex(record.index)
    record["cohort"] = cohort

    return {
        "summary": {
            "cohort": cohort,
            "cases": case_labels,
            "condition": list(condition) if condition else None,
            "n_control": len(controls),
            "n_case": len(cases),
            "n_specifications": int(run.grid_report.n_valid),
            "runtime_seconds": round(runtime, 1),
            "tiers": {k: int(v) for k, v in summary.tier_counts.items()},
            "level1_implementation": level1,
            "level2_published_pipeline": level2,
            "paper_significant_by_tier": {
                str(k): int(v) for k, v in
                record.loc[record["paper_q"] < Q, "tier"]
                .fillna("NOT TESTED").value_counts().items()
            },
        },
        "rows": record,
    }


# ------------------------------------------------------------------ published claims
def claims(rows: pd.DataFrame) -> dict:
    s3 = pd.read_csv(download_file("file-S3.nonspecific_genera.txt"), sep="\t", index_col=0)
    label = s3.iloc[:, 0].dropna()
    expected = label.map({"health": -1.0, "disease": 1.0}).dropna()

    merged = rows.join(expected.rename("expected"), how="inner")
    merged = merged[merged["median_effect"].notna()]
    merged["agrees"] = np.sign(merged["median_effect"]) == merged["expected"]

    by_tier = {}
    for tier in TIERED:
        sub = merged[merged["tier"] == tier]
        by_tier[tier] = {"n": int(len(sub)), "direction_agrees": int(sub["agrees"].sum())}
    paper_sig = merged[merged["paper_q"] < Q]
    paper_sig_agrees = int((np.sign(paper_sig["paper_effect"]) == paper_sig["expected"]).sum())

    crc = rows[rows["cohort"].str.startswith("crc_") & rows["genus"].isin(CRC_ENRICHED)]
    crc_table = [
        {"cohort": r.cohort, "genus": r.genus,
         "tier": None if pd.isna(r.tier) else r.tier,
         "frac_significant": (None if pd.isna(r.frac_significant)
                              else round(float(r.frac_significant), 3)),
         "median_effect_sign": None if pd.isna(r.median_effect) else int(np.sign(r.median_effect)),
         "paper_q": None if pd.isna(r.paper_q) else float(r.paper_q)}
        for r in crc.sort_values(["genus", "cohort"]).itertuples()
    ]
    return {
        "s3_labelled_genera": {"health": int((label == "health").sum()),
                               "disease": int((label == "disease").sum())},
        "s3_pairs_evaluated": int(len(merged)),
        "s3_direction_by_tier": by_tier,
        "s3_paper_significant_pairs": int(len(paper_sig)),
        "s3_paper_significant_direction_agrees": paper_sig_agrees,
        "crc_named_genera": crc_table,
    }


# ------------------------------------------------------------------------ replication
def _sign(value) -> float:
    return float(np.sign(value)) if pd.notna(value) else 0.0


def _mean(values: pd.Series):
    return round(float(values.mean()), 4) if len(values) else None


def replication(rows: pd.DataFrame, n_boot: int = 4000, seed: int = 20260926) -> dict:
    """Discovery in cohort i, replication in every other cohort j of the same disease.

    The yardstick is the paper's own pipeline run on cohort j, never MicroVerse, so the
    comparison cannot favour MicroVerse by construction. Two outcomes:

    * `significant`: q < 0.05 in cohort j, in the discovery direction (strict; limited by
      cohort j's power — several cohorts find nothing at all);
    * `direction`: cohort j's effect points the discovery way, significant or not.

    A genus recurs across cohort pairs, so pairs are not independent: intervals come
    from a cluster bootstrap that resamples genera.
    """
    pairs = []
    for disease, cohorts in DISEASES.items():
        present = set(rows["cohort"])
        for i in cohorts:
            if i not in present:
                continue
            disc = rows[rows["cohort"] == i]
            for j in cohorts:
                if i == j or j not in present:
                    continue
                val = rows[rows["cohort"] == j][["paper_q", "paper_effect"]]
                shared = disc.join(val, rsuffix="_val", how="inner")
                shared = shared[shared["paper_q_val"].notna()]
                for genus, r in shared.iterrows():
                    pairs.append({
                        "disease": disease, "discovery": i, "validation": j, "genus": genus,
                        "tier": r["tier"] if pd.notna(r["tier"]) else "NOT TESTED",
                        "paper_significant": bool(pd.notna(r["paper_q"]) and r["paper_q"] < Q),
                        "sign_mv": _sign(r["median_effect"]),
                        "sign_paper": _sign(r["paper_effect"]),
                        "val_significant": bool(r["paper_q_val"] < Q),
                        "val_sign": float(np.sign(r["paper_effect_val"])),
                    })
    f = pd.DataFrame(pairs)
    for who in ("mv", "paper"):
        sign = f[f"sign_{who}"]
        f[f"sig_{who}"] = f["val_significant"] & (sign != 0) & (f["val_sign"] == sign)
        f[f"dir_{who}"] = (sign != 0) & (f["val_sign"] != 0) & (f["val_sign"] == sign)

    sig = f["paper_significant"]
    stable = f["tier"].isin(["ROBUST", "CONDITIONAL"])
    shaky = f["tier"].isin(["FRAGILE", "UNSTABLE"])
    # (mask, whose direction is the discovery direction)
    groups = {
        "all genera: ROBUST": (f["tier"] == "ROBUST", "mv"),
        "all genera: CONDITIONAL": (f["tier"] == "CONDITIONAL", "mv"),
        "all genera: FRAGILE": (f["tier"] == "FRAGILE", "mv"),
        "all genera: UNSTABLE": (f["tier"] == "UNSTABLE", "mv"),
        "all genera: NOT DETECTED": (f["tier"] == "NOT DETECTED", "mv"),
        "paper-significant (what a single-pipeline paper reports)": (sig, "paper"),
        "paper-significant and ROBUST or CONDITIONAL": (sig & stable, "paper"),
        "paper-significant and FRAGILE or UNSTABLE": (sig & shaky, "paper"),
        "paper-significant and ROBUST": (sig & (f["tier"] == "ROBUST"), "paper"),
        "paper-significant and CONDITIONAL": (sig & (f["tier"] == "CONDITIONAL"), "paper"),
        "paper-significant and FRAGILE": (sig & (f["tier"] == "FRAGILE"), "paper"),
        "paper-significant and UNSTABLE": (sig & (f["tier"] == "UNSTABLE"), "paper"),
    }

    codes, genera = pd.factorize(f["genus"])
    rng = np.random.default_rng(seed)
    weights = [np.ones(len(genera))] + [
        np.bincount(rng.integers(0, len(genera), len(genera)), minlength=len(genera)).astype(float)
        for _ in range(n_boot)
    ]

    def rates(mask, outcome):
        m = mask.to_numpy().astype(float)
        o = outcome.to_numpy().astype(float)
        out = []
        for w in weights:
            row_w = w[codes] * m
            total = row_w.sum()
            out.append((row_w * o).sum() / total if total else np.nan)
        return np.asarray(out)

    def summarise(values, n):
        low, high = np.nanpercentile(values[1:], [2.5, 97.5]) if n else (np.nan, np.nan)
        return {"n_pairs": int(n), "rate": round(float(values[0]), 4),
                "ci95": [round(float(low), 4), round(float(high), 4)]}

    table, draws = {}, {}
    for name, (mask, who) in groups.items():
        entry = {"n_pairs": int(mask.sum())}
        for kind in ("sig", "dir"):
            values = rates(mask, f[f"{kind}_{who}"])
            draws[(name, kind)] = values
            entry["significant" if kind == "sig" else "direction"] = summarise(values, mask.sum())
        table[name] = entry

    contrasts = {}
    a_name = "paper-significant and ROBUST or CONDITIONAL"
    b_name = "paper-significant and FRAGILE or UNSTABLE"
    for kind, label in (("sig", "significant"), ("dir", "direction")):
        a, b = groups[a_name][0], groups[b_name][0]
        outcome = f[f"{kind}_paper"]
        counts = [[int(outcome[a].sum()), int((~outcome[a]).sum())],
                  [int(outcome[b].sum()), int((~outcome[b]).sum())]]
        odds, p = stats.fisher_exact(counts)
        diff = draws[(a_name, kind)] - draws[(b_name, kind)]
        low, high = np.nanpercentile(diff[1:], [2.5, 97.5])
        contrasts[label] = {
            "table_stable_then_shaky_[replicated, not]": counts,
            "difference": round(float(diff[0]), 4),
            "difference_ci95_cluster_bootstrap": [round(float(low), 4), round(float(high), 4)],
            "fisher_odds_ratio": round(float(odds), 3),
            "fisher_p_naive": float(p),
            "bootstrap_share_of_draws_with_difference_le_0":
                round(float(np.mean(diff[1:] <= 0)), 4),
        }

    per_disease = {}
    for disease in DISEASES:
        d = f["disease"] == disease
        per_disease[disease] = {
            name: {"n": int((mask & d).sum()),
                   "significant": _mean(f.loc[mask & d, "sig_paper"]),
                   "direction": _mean(f.loc[mask & d, "dir_paper"])}
            for name, mask in (("paper-significant and ROBUST or CONDITIONAL", sig & stable),
                               ("paper-significant and FRAGILE or UNSTABLE", sig & shaky))
        }

    return {
        "n_pairs": int(len(f)),
        "n_genera": int(len(genera)),
        "n_bootstrap": n_boot,
        "definitions": {
            "significant": "q < 0.05 under the paper's pipeline in another cohort of the "
                           "same disease, same direction",
            "direction": "the other cohort's effect has the same sign, significant or not",
        },
        "groups": table,
        "stable_vs_shaky_among_paper_significant": contrasts,
        "per_disease": per_disease,
    }


# ------------------------------------------------------------------------------ main
def main() -> int:
    found = conditions()
    records, all_rows, failures, refused = [], [], [], []
    for disease, cohorts in DISEASES.items():
        for cohort in cohorts:
            try:
                out = study_cohort(cohort, found.get(cohort))
            except DatasetError as exc:  # MicroVerse's own input rules said no: a result
                refused.append({"cohort": cohort, "reason": str(exc)})
                print(f"  REFUSED {cohort}: {exc}")
                continue
            except Exception as exc:  # recorded, not hidden
                failures.append({"cohort": cohort, "error": f"{type(exc).__name__}: {exc}"})
                print(f"  FAILED {cohort}: {exc}")
                continue
            out["summary"]["disease"] = disease
            records.append(out["summary"])
            all_rows.append(out["rows"])
            s = out["summary"]
            l1, l2 = s["level1_implementation"], s["level2_published_pipeline"]
            print(f"  {cohort:22s} n={s['n_control']:>3}+{s['n_case']:<3} "
                  f"specs={s['n_specifications']:>5} paper sig={l2['n_sig_paper']:>3} "
                  f"matched={l2['n_sig_matched']:>3} both={l2['n_sig_both']:>3} "
                  f"jaccard={l2['jaccard']:.2f} rho={l2['spearman_p']:.3f} | "
                  f"same-matrix agree={l1['agreement_same_matrix']:.3f} "
                  f"|dp|max={l1['max_abs_diff_p_vs_scipy_mannwhitney']:.1e} "
                  f"({s['runtime_seconds']}s)")

    rows = pd.concat(all_rows)
    result = {
        "source": "MicrobiomeHD, Duvallet et al. 2017, Nat Commun 8:1784; "
                  "Zenodo 1146764 (CC-BY-NC-4.0)",
        "paper_pipeline": "re-implemented from github.com/cduvallet/microbiomeHD "
                          "(clean_otu_and_metadata.py, util.py, get_qvalues.py)",
        "matched_specification": MATCHED,
        "mode": "quick",
        "cohorts": records,
        "refused_by_validation": refused,
        "failures": failures,
        "claims": claims(rows),
        "replication": replication(rows),
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=float)
    rep = result["replication"]
    print(json.dumps({"s3_direction_by_tier": result["claims"]["s3_direction_by_tier"],
                      "crc_named_genera": result["claims"]["crc_named_genera"]}, indent=1))
    print(f"replication: {rep['n_pairs']} cohort-pair x genus observations, "
          f"{rep['n_genera']} genera")
    for name, entry in rep["groups"].items():
        sg, dr = entry["significant"], entry["direction"]
        print(f"  {name:58s} n={entry['n_pairs']:>5}  significant {sg['rate']:.3f} {sg['ci95']}  "
              f"direction {dr['rate']:.3f} {dr['ci95']}")
    print(json.dumps(rep["stable_vs_shaky_among_paper_significant"], indent=1))
    print(f"wrote {OUTPUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
