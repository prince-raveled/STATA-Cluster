"""Validation against a paper's published method results — Nearing et al. 2022.

Nearing et al. (Nat Commun 2022;13:342) ran 14 differential abundance methods on 38
public 16S datasets and published, for every dataset and method, how many ASVs came out
significant (their Figure 1A unfiltered, 1B with a 10% prevalence filter;
`Plotting_data/Main_Figures/Figure1*_feat_count.csv.gz` in their repository). That
paper is why fork 5 exists. This study asks whether MicroVerse's implementations of the
five methods it shares with the paper give the paper's numbers on the paper's data:

    paper                 MicroVerse                                 expected agreement
    Wilcoxon (rare)       wilcoxon, raw, on the paper's rarefied     exact: same table,
                          table                                      same test, same BH
    t-test (rare)         welch, raw, on the paper's rarefied table  exact
    Wilcoxon (CLR)        wilcoxon, clr                              not exact: the paper
                                                                     adds 1 before the CLR,
                                                                     MicroVerse uses
                                                                     multiplicative
                                                                     replacement
    ALDEx2                aldex2                                     approximate: separate
                                                                     implementation
    DESeq2                pydeseq2                                   approximate: Python
                                                                     port of the R package

For the CLR row the paper's own transform is also run through MicroVerse's test and FDR,
which separates "the test is implemented correctly" from "the zero handling differs".

Inputs are reconstructed exactly from the authors' figshare archive and their scripts
(`Filter_samples_of_non_rare_table.R` for the unfiltered run, which uses the rarefied
tables shipped in the archive; `Filter_samples_and_features.R` for the filtered run,
whose feature filter and depth-based sample exclusion are deterministic — its rarefied
tables are not, being drawn with R's RNG, so the two rarefied methods are compared on
the unfiltered run only). MicroVerse accepts up to 1,500 features, so the datasets used
are those at or under that size.

Data: figshare 14531724 (CC BY 4.0), cached under the git-ignored data/nearing/.

    .venv/Scripts/python tests/reference/nearing_study.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from app.core import fdr  # noqa: E402
from app.core.methods import run_method  # noqa: E402
from app.core.parsers.base import AbundanceTable  # noqa: E402
from app.core.preprocess import MatrixBuilder  # noqa: E402
from app.core.validation import DatasetError, validate_dataset  # noqa: E402

CACHE = os.path.join(ROOT, "data", "nearing")
OUTPUT = os.path.join(ROOT, "docs", "nearing_study.json")
ARCHIVE = "DA_COMPARE_DATA_21_03_08.tar.gz"
ARCHIVE_URL = "https://ndownloader.figshare.com/files/27857901"
ARCHIVE_MD5 = "3bf04686142ba000b3b6f7825cc1a606"
REPO = "https://raw.githubusercontent.com/nearinj/Comparison_of_DA_microbiome_methods/master"
PUBLISHED = {
    "unfiltered": "Plotting_data/Main_Figures/Figure1A_feat_count.csv.gz",
    "filtered": "Plotting_data/Main_Figures/Figure1B_feat_count.csv.gz",
}
CHARACTERISTICS = "Plotting_data/Main_Figures/Figure1A_dataset_characteristics.csv.gz"
#: The authors' own feature counts per table — a check that the inputs are rebuilt exactly.
FEATURE_COUNTS = {
    "unfiltered": "Misc_datafiles/ASV_nonfilt_nums.txt",
    "filtered": "Misc_datafiles/ASV_filt_nums.txt",
}
MAX_FEATURES = 1500
#: PyDESeq2 on one CPU did not finish in 30 minutes on Office's filtered table (836
#: samples by 185 features); above this many samples the DESeq2 comparison is skipped,
#: and recorded as skipped.
DESEQ_MAX_SAMPLES = 500
Q = 0.05

#: Paper method -> how MicroVerse runs it.
METHODS = {
    "Wilcoxon (rare)": {"method": "wilcoxon", "transform": "raw", "table": "rare",
                        "agreement": "exact"},
    "t-test (rare)": {"method": "ttest", "transform": "raw", "table": "rare",
                      "agreement": "exact"},
    "Wilcoxon (CLR)": {"method": "wilcoxon", "transform": "clr", "table": "nonrare",
                       "agreement": "zero handling differs"},
    "ALDEx2": {"method": "aldex2", "transform": "raw", "table": "nonrare",
               "agreement": "separate implementation"},
    "DESeq2": {"method": "pydeseq2", "transform": "raw", "table": "nonrare",
               "agreement": "separate implementation"},
}


# ---------------------------------------------------------------------------- inputs
def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "microverse-validation"})
        with urllib.request.urlopen(request, timeout=1800) as response, open(path, "wb") as out:
            out.write(response.read())
    return path


def archive_path() -> str:
    path = fetch(ARCHIVE_URL, os.path.join(CACHE, ARCHIVE))
    import hashlib
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise RuntimeError(f"{ARCHIVE} does not match figshare's md5")
    return path


def published(variant: str) -> pd.DataFrame:
    path = fetch(f"{REPO}/{PUBLISHED[variant]}",
                 os.path.join(CACHE, os.path.basename(PUBLISHED[variant])))
    return pd.read_csv(path).set_index("Unnamed: 0")


def characteristics() -> pd.DataFrame:
    path = fetch(f"{REPO}/{CHARACTERISTICS}",
                 os.path.join(CACHE, os.path.basename(CHARACTERISTICS)))
    table = pd.read_csv(path, encoding="latin-1")
    return table.set_index("Dataset.Name")


def paper_feature_counts(variant: str) -> dict:
    path = fetch(f"{REPO}/{FEATURE_COUNTS[variant]}",
                 os.path.join(CACHE, os.path.basename(FEATURE_COUNTS[variant])))
    counts = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            n, where = line.split()
            counts[where.split("/")[0]] = int(n) - 1   # `wc -l`: the header line is counted
    return counts


def read_table(raw: bytes) -> pd.DataFrame:
    """The authors' read: skip a biom-constructed first line, drop a taxonomy column."""
    skip = 1 if raw.startswith(b"# Constructed from biom file") else 0
    # Parsed as numbers directly: read as text first, the 168,728-ASV table ran the
    # machine out of memory.
    frame = pd.read_csv(io.BytesIO(raw), sep="\t", skiprows=skip, index_col=0,
                        quoting=3, comment=None, low_memory=False,
                        encoding="utf-8", encoding_errors="replace")
    frame = frame.drop(columns=[c for c in frame.columns if c == "taxonomy"])
    frame.columns = [str(c) for c in frame.columns]
    frame.index = [str(i) for i in frame.index]
    text_columns = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if text_columns:
        frame[text_columns] = frame[text_columns].apply(pd.to_numeric, errors="coerce")
    return frame.fillna(0.0)


def read_groupings(raw: bytes) -> pd.Series:
    frame = pd.read_csv(io.BytesIO(raw), sep="\t", index_col=0, dtype=str, quoting=3)
    if frame.shape[1] == 0:  # one metadata file is named .csv and is comma-separated
        frame = pd.read_csv(io.BytesIO(raw), sep=",", index_col=0, dtype=str)
    frame.index = [str(i) for i in frame.index]
    return frame.iloc[:, 0]


def study_file(archive: tarfile.TarFile, name: str, files: pd.Series, column: str) -> bytes:
    """One of the files the authors list for this dataset in their characteristics table
    (the names vary: `_meta.tsv`, `_metadata.csv`, ...)."""
    folder = f"Hackathon/Studies/{name}"
    member = archive.getmember(f"{folder}/{files[column]}")
    if member.issym():
        # One link in the archive points at the authors' own disk
        # (/home/shared/.../Exercise_ASVs_rare_table.tsv); the file it names is
        # packed beside it, so read that.
        member = archive.getmember(f"{folder}/{os.path.basename(member.linkname)}")
    return archive.extractfile(member).read()


def load_study(archive: tarfile.TarFile, name: str, files: pd.Series):
    def get(column: str) -> bytes:
        return study_file(archive, name, files, column)

    return (read_table(get("ASV.Table.File.Name")), read_table(get("ASV.Table.Rarified.Name")),
            read_groupings(get("Metadata.File.Name")))


# ------------------------------------------------------- the authors' sample handling
def unfiltered_tables(table, rare, groups):
    """`Filter_samples_of_non_rare_table.R`: the rarefied table sets the samples."""
    rare = rare[[s for s in rare.columns if s in set(groups.index)]]
    rare = rare.loc[rare.sum(axis=1) > 0]
    table = table[list(rare.columns)]
    table = table.loc[table.sum(axis=1) > 0]
    return table, rare


def filtered_table(table, groups, depth):
    """`Filter_samples_and_features.R` with cutoff 0.1, as far as it is deterministic:
    the feature filter counts every sample in the table, then GUniFrac::Rarefy discards
    samples under the depth (their draw is R's RNG and is not reproduced), then samples
    without a grouping go, then empty features."""
    return filtered_samples(feature_filter(table), groups, depth)


def feature_filter(table: pd.DataFrame, n_samples: int | None = None) -> pd.DataFrame:
    """`remove_rare_features(table, 0.1)`: present in more than ceiling(10%) of every
    sample in the table. `n_samples` lets a chunk of rows use the whole table's width."""
    cutoff = math.ceil(0.1 * (n_samples if n_samples is not None else table.shape[1]))
    return table.loc[(table > 0).sum(axis=1) > cutoff]


def filtered_samples(table: pd.DataFrame, groups: pd.Series, depth: float) -> pd.DataFrame:
    """The rest of the filtered run: samples under the depth, then samples without a
    grouping, then features left empty."""
    table = table.loc[:, table.sum(axis=0) >= depth]
    table = table[[s for s in table.columns if s in set(groups.index)]]
    return table.loc[table.sum(axis=1) > 0]


def read_table_prefiltered(raw: bytes) -> pd.DataFrame:
    """`read_table` then `feature_filter`, a slice of rows at a time. ArcticFreshwaters is
    936 samples by 168,728 ASVs and does not fit in memory whole; only 344 ASVs survive
    the filter, and the filter looks at one row at a time, so nothing is lost."""
    skip = 1 if raw.startswith(b"# Constructed from biom file") else 0
    options = {"sep": "\t", "skiprows": skip, "index_col": 0, "quoting": 3, "comment": None,
               "encoding": "utf-8", "encoding_errors": "replace"}
    header = pd.read_csv(io.BytesIO(raw), nrows=0, **options)
    n_samples = sum(1 for c in header.columns if c != "taxonomy")
    kept = []
    for chunk in pd.read_csv(io.BytesIO(raw), chunksize=5000, low_memory=False, **options):
        chunk = chunk.drop(columns=[c for c in chunk.columns if c == "taxonomy"])
        chunk.columns = [str(c) for c in chunk.columns]
        chunk.index = [str(i) for i in chunk.index]
        chunk = chunk.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        kept.append(feature_filter(chunk, n_samples))
    return pd.concat(kept)


# -------------------------------------------------------------------- the MicroVerse side
def dataset_for(table: pd.DataFrame, groups: pd.Series):
    labels = groups.loc[table.columns].astype(str)
    levels = sorted(labels.unique())
    if len(levels) != 2:
        raise DatasetError(f"{len(levels)} groups", "")
    metadata = pd.DataFrame({"group": labels.to_numpy()}, index=table.columns)
    abundance = AbundanceTable(counts=table.round().astype(float), lineages={},
                               source_format="nearing", value_type="counts")
    return validate_dataset(abundance, metadata, "group")


def count_significant(dataset, method: str, transform: str) -> int:
    matrix = MatrixBuilder(dataset).build("none", None, "input", 0.0, transform)
    fit = run_method(method, matrix, aldex_instances=128 if method == "aldex2" else None)
    adjusted = fdr.adjust(fit.p_raw, "bh")
    return int(np.sum(adjusted <= Q))


def count_pydeseq2_paper_settings(table: pd.DataFrame, groups: pd.Series) -> int:
    """`Run_DESeq2.R` as the paper ran it: poscounts size factors, and DESeq2's own
    adjusted p-values (independent filtering and Cook's cutoff, alpha 0.1 as in R's
    `results()`), counted at 0.05. MicroVerse instead applies its FDR fork to the raw
    p-values of every method alike, so this isolates the implementation from that
    design choice."""
    import warnings

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    from app.core.methods.deseq import _run_in_one_process

    _run_in_one_process()
    labels = groups.loc[table.columns].astype(str)
    counts = pd.DataFrame(np.rint(table.to_numpy(dtype=float)).astype(int).T,
                          index=table.columns, columns=table.index)
    metadata = pd.DataFrame({"group": labels.to_numpy()}, index=table.columns)
    first, second = sorted(labels.unique())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=metadata, design="~group",
                           size_factors_fit_type="poscounts", refit_cooks=True,
                           quiet=True, n_cpus=1)
        dds.deseq2()
        result = DeseqStats(dds, contrast=["group", second, first], alpha=0.1, quiet=True)
        result.summary()
    return int((result.results_df["padj"] < Q).sum())


def count_paper_clr(table: pd.DataFrame, groups: pd.Series) -> int:
    """`Run_Wilcox_CLR.R`: CLR of counts + 1, wilcox.test (normal approximation with
    continuity correction), then BH — here with scipy, on the same table."""
    logged = np.log(table.to_numpy(dtype=float) + 1.0)
    clr = logged - logged.mean(axis=0, keepdims=True)
    labels = groups.loc[table.columns].to_numpy()
    first, second = sorted(set(labels))
    result = stats.mannwhitneyu(clr[:, labels == first], clr[:, labels == second], axis=1,
                                alternative="two-sided", method="asymptotic",
                                use_continuity=True)
    p = np.where(np.isfinite(result.pvalue), result.pvalue, 1.0)
    return int(np.sum(fdr.adjust(p, "bh") <= Q))


# ---------------------------------------------------------------------------- the study
def _spearman(a: pd.Series, b: pd.Series):
    if len(a) <= 2 or a.nunique() <= 1:
        return None
    return round(float(stats.spearmanr(a, b).statistic), 4)


def _mean_pct(sub: pd.DataFrame, column: str) -> dict:
    """Mean share of features called significant, per method, in percent."""
    out = {}
    for method in METHODS:
        rows = sub[sub["method"] == method]
        if len(rows):
            out[method] = round(float((rows[column] / rows["n_features"]).mean() * 100), 2)
    return out


def _depth(value) -> float:
    """One depth in the characteristics table is written with a thousands separator."""
    return float(str(value).replace(",", ""))


def _code_fingerprint() -> str:
    """The study script and every piece of the engine it exercises. A cached dataset is
    reused only while this is unchanged, so a result can never come from older code."""
    import glob
    import hashlib

    digest = hashlib.sha256()
    paths = [os.path.abspath(__file__),
             *sorted(glob.glob(os.path.join(ROOT, "app", "core", "methods", "*.py"))),
             *(os.path.join(ROOT, "app", "core", f) for f in
               ("preprocess.py", "fdr.py", "validation.py"))]
    for path in paths:
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:16]


def study_dataset(archive, name, info, paper_n, pubs):
    """Every comparison for one dataset: (rows, skipped)."""
    rows, skipped = [], []
    sizes = {variant: paper_n[variant].get(name) for variant in paper_n}
    if all(n is not None and n > MAX_FEATURES for n in sizes.values()):
        # The authors' own counts already rule it out; do not load a table of up to
        # 168,728 features only to discard it.
        return rows, [{"dataset": name, "variant": variant,
                       "reason": f"{n} features, over {MAX_FEATURES}"}
                      for variant, n in sizes.items()]
    depth = _depth(info.loc[name, "Rarefaction.depth"])
    try:
        if sizes["unfiltered"] is not None and sizes["unfiltered"] > MAX_FEATURES:
            # Only the filtered run can be used: filter while reading, never hold the
            # whole table.
            skipped.append({"dataset": name, "variant": "unfiltered",
                            "reason": f"{sizes['unfiltered']} features, over {MAX_FEATURES}"})
            files = info.loc[name]
            groups = read_groupings(study_file(archive, name, files, "Metadata.File.Name"))
            prefiltered = read_table_prefiltered(
                study_file(archive, name, files, "ASV.Table.File.Name"))
            variants = {"filtered": (filtered_samples(prefiltered, groups, depth), None)}
        else:
            table, rare, groups = load_study(archive, name, info.loc[name])
            variants = {"unfiltered": unfiltered_tables(table, rare, groups),
                        "filtered": (filtered_table(table, groups, depth), None)}
    except KeyError as exc:
        return rows, [{"dataset": name, "reason": f"file missing from the archive: {exc}"}]
    label = info.loc[name, "Unnamed: 0"]
    for variant, (nonrare, rare_table) in variants.items():
        if nonrare.shape[0] > MAX_FEATURES:
            skipped.append({"dataset": name, "variant": variant,
                            "reason": f"{nonrare.shape[0]} features, over {MAX_FEATURES}"})
            continue
        try:
            datasets = {"nonrare": dataset_for(nonrare, groups)}
            if rare_table is not None:
                datasets["rare"] = dataset_for(rare_table, groups)
        except DatasetError as exc:
            skipped.append({"dataset": name, "variant": variant,
                            "reason": f"refused by validation: {exc}"})
            continue
        for paper_method, how in METHODS.items():
            if how["table"] == "rare" and "rare" not in datasets:
                continue
            ds = datasets[how["table"]]
            if how["method"] == "pydeseq2" and ds.n_samples > DESEQ_MAX_SAMPLES:
                skipped.append({"dataset": name, "variant": variant, "method": paper_method,
                                "reason": f"{ds.n_samples} samples, over {DESEQ_MAX_SAMPLES}: "
                                          "PyDESeq2 on one CPU did not finish in 30 "
                                          "minutes on Office's 836"})
                continue
            try:
                mine = count_significant(ds, how["method"], how["transform"])
            except Exception as exc:  # recorded, not hidden
                skipped.append({"dataset": name, "variant": variant, "method": paper_method,
                                "reason": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            row = {
                "dataset": name, "label": label, "variant": variant,
                "method": paper_method, "agreement_expected": how["agreement"],
                "n_features": int(ds.n_taxa), "n_samples": int(ds.n_samples),
                "input_features": int(nonrare.shape[0]),
                "paper_features": paper_n[variant].get(name),
                "published": int(pubs[variant].loc[paper_method, label]),
                "microverse": mine,
            }
            extra = ""
            if paper_method == "Wilcoxon (CLR)":
                row["microverse_test_on_paper_clr"] = count_paper_clr(nonrare, groups)
                extra = (f"  (paper's CLR through MicroVerse's test: "
                         f"{row['microverse_test_on_paper_clr']})")
            if paper_method == "DESeq2":
                row["pydeseq2_paper_settings"] = count_pydeseq2_paper_settings(nonrare, groups)
                extra = f"  (PyDESeq2 with the paper's settings: {row['pydeseq2_paper_settings']})"
            rows.append(row)
            print(f"  {variant:10s} {name:28s} {paper_method:16s} "
                  f"published {row['published']:>5}  MicroVerse {mine:>5}{extra}", flush=True)
    return rows, skipped


def main() -> int:
    only = None
    if "--datasets" in sys.argv:
        only = set(sys.argv[sys.argv.index("--datasets") + 1].split(","))
    info = characteristics()
    paper_n = {variant: paper_feature_counts(variant) for variant in FEATURE_COUNTS}
    pubs = {variant: published(variant) for variant in PUBLISHED}
    fingerprint = _code_fingerprint()
    cache_path = os.path.join(CACHE, "study_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cache = json.load(handle)
    rows, skipped = [], []
    with tarfile.open(archive_path()) as archive:
        for name in info.index:
            if only and name not in only:
                continue
            cached = cache.get(name)
            if cached and cached.get("code") == fingerprint:
                rows += cached["rows"]
                skipped += cached["skipped"]
                print(f"  (reused {name}: {len(cached['rows'])} comparisons from this code)")
                continue
            new_rows, new_skipped = study_dataset(archive, name, info, paper_n, pubs)
            rows += new_rows
            skipped += new_skipped
            cache[name] = {"code": fingerprint, "rows": new_rows, "skipped": new_skipped}
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(cache, handle, default=float)

    frame = pd.DataFrame(rows)
    summary = {}
    for (variant, method), sub in frame.groupby(["variant", "method"], sort=False):
        pct_pub = sub["published"] / sub["n_features"] * 100
        pct_mv = sub["microverse"] / sub["n_features"] * 100
        entry = {
            "n_datasets": int(len(sub)),
            "exact_matches": int((sub["published"] == sub["microverse"]).sum()),
            "mean_abs_difference_pct_points": round(float((pct_mv - pct_pub).abs().mean()), 3),
            "mean_pct_published": round(float(pct_pub.mean()), 2),
            "mean_pct_microverse": round(float(pct_mv.mean()), 2),
            "spearman_counts": _spearman(sub["published"], sub["microverse"]),
        }
        if method == "Wilcoxon (CLR)":
            entry["exact_matches_with_paper_clr"] = int(
                (sub["published"] == sub["microverse_test_on_paper_clr"]).sum())
        if method == "DESeq2":
            alt = sub["pydeseq2_paper_settings"]
            entry["paper_settings_exact_matches"] = int((sub["published"] == alt).sum())
            entry["paper_settings_mean_abs_difference_pct_points"] = round(
                float(((alt - sub["published"]) / sub["n_features"] * 100).abs().mean()), 3)
            entry["paper_settings_spearman_counts"] = _spearman(sub["published"], alt)
        summary[f"{variant} | {method}"] = entry

    # The paper's central finding is the spread between methods. Does MicroVerse put the
    # five shared methods in the same order, dataset by dataset?
    ordering = {}
    for variant, sub in frame.groupby("variant"):
        pub = sub.pivot(index="dataset", columns="method", values="published")
        mv = sub.pivot(index="dataset", columns="method", values="microverse")
        complete = pub.dropna().index.intersection(mv.dropna().index)
        rhos = [stats.spearmanr(pub.loc[d], mv.loc[d]).statistic for d in complete
                if pub.loc[d].nunique() > 1 and mv.loc[d].nunique() > 1]
        ordering[variant] = {
            "n_datasets": int(len(rhos)),
            "median_within_dataset_spearman": round(float(np.median(rhos)), 3) if rhos else None,
            "mean_pct_by_method_published": _mean_pct(sub, "published"),
            "mean_pct_by_method_microverse": _mean_pct(sub, "microverse"),
        }

    result = {
        "source": "Nearing JT et al. Microbiome differential abundance methods produce "
                  "different results across 38 datasets. Nat Commun 2022;13:342",
        "data": "figshare 14531724 (CC BY 4.0), archive md5 " + ARCHIVE_MD5,
        "published_results": {v: f"{REPO}/{p}" for v, p in PUBLISHED.items()},
        "methods": METHODS,
        "max_features": MAX_FEATURES,
        "rows": rows,
        "summary": summary,
        "method_ordering": ordering,
        "skipped": skipped,
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=float)
    print(json.dumps({"summary": summary, "method_ordering": ordering}, indent=1))
    print(f"skipped {len(skipped)}; wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
