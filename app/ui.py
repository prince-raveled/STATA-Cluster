"""Presentation helpers for the web interface.

Nothing here computes a result. Every number these functions return is either read
from the engine (a GridReport, a job row, a readiness report) or recounted from the
same grid definitions the engine enumerates from, so the interface can show *how* a
count arises without holding a second copy of the rules. Where a recount is shown,
it is checked against the engine's own figure and the page says whether they agree.
"""
from __future__ import annotations

import re

from .core.glossary import term as core_term

# --------------------------------------------------------------------------
# Help terms the core glossary does not define. The interface needs them for its
# [?] explanations; they are definitions of standard statistics and of what this
# build does, not new rules.
# --------------------------------------------------------------------------
UI_TERMS: dict[str, dict] = {
    "p_value": {
        "term": "p-value",
        "short": "How surprising the observed difference would be if the two groups "
                 "did not differ. Each statistical test produces its own.",
        "long": "The probability, under the test's assumptions and a true difference of "
                "zero, of seeing a difference at least this large. MicroVerse keeps each "
                "method's own raw p-value (p_raw in the exports). A small p-value is "
                "evidence against 'no difference' under that one analysis; it is not the "
                "probability that the difference is real.",
    },
    "q_value": {
        "term": "adjusted p-value (q)",
        "short": "The p-value after multiple-testing correction. A taxon is called "
                 "significant when it is at or below the threshold.",
        "long": "Testing hundreds of taxa at once inflates false positives, so each "
                "specification's p-values are adjusted with Benjamini-Hochberg or "
                "Benjamini-Yekutieli (p_adj in the exports). Some papers call this a "
                "q-value; MicroVerse does not compute Storey's q-value, only these "
                "adjusted p-values.",
    },
    "clr": {
        "term": "CLR (centred log-ratio)",
        "short": "Log of each taxon's abundance relative to the sample's geometric mean.",
        "long": "A transformation that respects the compositional nature of sequencing "
                "data: only ratios between taxa are meaningful. Zeros are replaced "
                "multiplicatively before the log is taken, rather than with a fixed "
                "pseudocount.",
    },
    "tmm": {
        "term": "TMM (trimmed mean of M-values)",
        "short": "edgeR's library-size normalisation, robust to a few very abundant taxa.",
        "long": "Scales each sample by a factor estimated from the taxa that do not change "
                "much between samples. It needs integer counts, and it is never combined "
                "with rarefaction (that would correct library size twice).",
    },
    "tss": {
        "term": "TSS (total-sum scaling)",
        "short": "Each count divided by its sample's total: relative abundance.",
        "long": "The simplest normalisation. It turns counts into proportions, so every "
                "sample sums to one.",
    },
    "taxonomic_rank": {
        "term": "taxonomic rank",
        "short": "The level at which taxa are counted: the table as uploaded, or "
                 "collapsed to genus.",
        "long": "Collapsing sums every feature in the same genus into one row. It trades "
                "resolution for power, and results can change with it. With no taxonomy, "
                "or a table already at genus, this choice has one level.",
    },
    "robustness": {
        "term": "robust to analytical choice",
        "short": "Significant in at least 80% of the analyses that tested the taxon, "
                 "with the direction agreeing in at least 95%.",
        "long": "A statement about analytical stability: the conclusion did not depend on "
                "which defensible pipeline was chosen. It is not a statement that the "
                "difference is biologically real or causal.",
    },
}

UI_GLOSSARY_HEADING = "Statistics and preprocessing"


def help_term(key: str) -> dict:
    """A definition from the interface's own terms, else from the core glossary."""
    if key in UI_TERMS:
        return UI_TERMS[key]
    return core_term(key)


def glossary_with_ui_terms(sections: list) -> list:
    """The core glossary plus the interface's statistics terms, as one list."""
    extra = {"heading": UI_GLOSSARY_HEADING,
             "entries": [dict(entry, key=key) for key, entry in UI_TERMS.items()]}
    return list(sections) + [extra]


# --------------------------------------------------------------------------
# Tier names, in reader-facing words. The engine's labels stay the labels; these
# are the one-line meaning printed beside them.
# --------------------------------------------------------------------------
TIER_HEADLINES = {
    "ROBUST": "Robust to analytical choice",
    "CONDITIONAL": "Depends on the analytical choice",
    "FRAGILE": "Rarely significant",
    "UNSTABLE": "Direction not determined",
    "INSUFFICIENT": "Too few analyses to judge",
    "NOT DETECTED": "Never significant",
}


# --------------------------------------------------------------------------
# The workflow stepper.
# --------------------------------------------------------------------------
WORKFLOW = (
    ("dataset", "Dataset"),
    ("validate", "Validate"),
    ("configure", "Configure"),
    ("run", "Run"),
    ("results", "Results"),
)


def workflow_steps(current: str, token: str = "", status: str = "") -> list:
    """The five steps, each with a state and, where it is safe, a link.

    Backward navigation is offered only where it cannot lose or corrupt anything:
    validation is read-only, configuration only starts something when submitted, and
    while a run is in progress the configure step is not linked, because submitting it
    again would start a second run over the first.
    """
    order = [key for key, _ in WORKFLOW]
    here = order.index(current) if current in order else 0
    finished = status == "done"
    running = status == "running"

    steps = []
    for index, (key, label) in enumerate(WORKFLOW):
        if index < here:
            state = "done"
        elif index == here:
            state = "current"
        elif finished and key in ("run", "results"):
            state = "done"
        else:
            state = "upcoming"

        href = ""
        if key == "dataset" and current != "dataset":
            href = "/#start"
        elif token and key == "validate":
            href = f"/validate/{token}"
        elif token and key == "configure" and not running:
            href = f"/configure/{token}"
        elif token and key == "run" and status in ("running", "error"):
            href = f"/job/{token}"
        elif token and key == "results" and finished:
            href = f"/results/{token}"
        if index == here:
            href = ""

        hint = {"done": "done", "current": "you are here", "upcoming": ""}[state]
        if key == "dataset" and state == "done":
            hint = "uploaded"
        if key == "run" and status == "error":
            hint = "stopped"
        steps.append({"key": key, "label": label, "number": index + 1,
                      "state": state, "href": href, "hint": hint})
    return steps


# --------------------------------------------------------------------------
# Dataset status on the validate page.
# --------------------------------------------------------------------------
def dataset_status(readiness) -> dict:
    """READY unless a check limits or undermines the analysis; never blocking.

    The files already passed validation (that is how a job exists at all). What is
    left is whether the design can answer the question, which the engine reports as
    readiness levels; "caution" and "serious" mean a person should read them first.
    """
    worst = getattr(readiness, "worst", "ok")
    if worst == "serious":
        return {"key": "review", "label": "Review required", "tone": "red", "glyph": "!",
                "sentence": "The files were read correctly, but the study design will make "
                            "the results hard to interpret. Read the findings below "
                            "before you continue. Nothing here blocks the analysis."}
    if worst == "caution":
        return {"key": "review", "label": "Review required", "tone": "amber", "glyph": "!",
                "sentence": "The files were read correctly. Some properties of the data "
                            "will limit what the analysis can conclude; they are listed "
                            "below. Nothing here blocks the analysis."}
    return {"key": "ready", "label": "Dataset ready", "tone": "green", "glyph": "✓",
            "sentence": "The files were read correctly and nothing about the design "
                        "limits what the analysis can conclude."}


# --------------------------------------------------------------------------
# Run stages, read from the job's own progress message.
# --------------------------------------------------------------------------
RUN_STAGES = (
    ("queue", "Waiting for an analysis slot"),
    ("grid", "Building the specification grid"),
    ("fit", "Fitting every specification"),
    ("assemble", "Assembling the results table"),
    ("tiers", "Computing robustness tiers"),
    ("attribution", "Attributing variation to the choices"),
    ("save", "Saving results and exports"),
)

_MATRIX = re.compile(r"matrix\s+(\d+)\s+of\s+(\d+)", re.I)


def _stage_index(message: str, progress: float) -> int:
    text = (message or "").lower()
    if text.startswith("queued"):
        return 0
    if text.startswith("starting") or text.startswith("enumerating"):
        return 1
    if text.startswith("fitting"):
        return 2
    if text.startswith("assembling"):
        return 3
    if text.startswith("computing robustness"):
        return 4
    if text.startswith("attributing"):
        return 5
    if text.startswith("saving"):
        return 6
    # A message this list does not know: place it by the fraction the engine reports.
    if progress >= 0.99:
        return 6
    if progress >= 0.97:
        return 5
    if progress >= 0.95:
        return 4
    if progress >= 0.92:
        return 3
    if progress >= 0.05:
        return 2
    return 1


def run_stages(job) -> list:
    """Each stage as done / current / pending (or error), with matrix k of N if known."""
    status = getattr(job, "status", "")
    message = getattr(job, "message", "") or ""
    progress = float(getattr(job, "progress", 0.0) or 0.0)
    current = len(RUN_STAGES) if status == "done" else _stage_index(message, progress)

    matrix = _MATRIX.search(message)
    stages = []
    for index, (key, label) in enumerate(RUN_STAGES):
        if index < current:
            state = "done"
        elif index == current:
            state = "error" if status == "error" else "current"
        else:
            state = "pending"
        detail = ""
        if key == "fit" and matrix and state == "current":
            detail = f"matrix {int(matrix.group(1)):,} of {int(matrix.group(2)):,}"
        stages.append({"key": key, "label": label, "state": state, "detail": detail})
    return stages


# --------------------------------------------------------------------------
# The specification space, fork by fork, for the configure page.
# --------------------------------------------------------------------------
FORK_QUESTIONS = {
    "rarefaction": "Subsample every sample to one depth first, and if so how deep?",
    "rank": "Count taxa as uploaded, or collapse them to genus?",
    "prev_filter": "How rare can a taxon be and still be tested?",
    "transform": "Which scale are the counts put on before testing?",
    "method": "Which statistical test compares the two groups?",
    "fdr": "Which multiple-testing correction, at which threshold?",
    "covariates": "Which covariates is the comparison adjusted for?",
}

#: The three modes in reader-facing words. `config.MODE_BLURBS` says the same for the
#: API, where its wording is part of the published response and is left alone.
MODE_SUMMARIES = {
    "quick": "Rarefaction, prevalence, transformation and rank, crossed with the four "
             "elementary tests and three corrections. The default, and the mode with the "
             "best evidence behind it: Pelto et al. 2025 found the elementary methods the "
             "most replicable.",
    "full": "Quick, plus ANCOM-BC, ALDEx2 and PyDESeq2, each on a stratified sample of the "
            "matrices it accepts. Slower; the sampling fraction is reported.",
    "covariate": "Every subset of the covariates you choose, over a reference sub-grid of "
                 "the other choices. This is the Tierney et al. 2022 analysis.",
}

TRANSFORM_NAMES = {"tss": "TSS", "clr": "CLR", "raw": "raw counts", "tmm": "TMM"}


def depth_labels(depth_levels) -> list:
    """('min', 1), ('min', 2), ... -> ['none', 'min depth × 3 seeds', ...]."""
    seeds: dict = {}
    for name, seed in depth_levels:
        seeds.setdefault(name, []).append(seed)
    labels = []
    for name, values in seeds.items():
        if name == "none":
            labels.append("none")
            continue
        shown = "min depth" if name == "min" else f"{int(name):,} reads"
        n = len([v for v in values if v is not None])
        labels.append(f"{shown} × {n} seed{'s' if n != 1 else ''}")
    return labels


def specification_space(builder, capabilities, covariate_columns, previews) -> dict:
    """Fork cards and a checked derivation of the enumerated count, per mode."""
    from .core.grid import (
        COVARIATE_REFERENCE_PREVALENCE,
        COVARIATE_REFERENCE_RAREFACTION,
        COVARIATE_REFERENCE_TRANSFORMS,
        FDR_SETTINGS,
        FULL_MODE_MATRIX_FRACTION,
        _matrix_keys,
        _stratified_matrix_sample,
        covariate_subsets,
    )
    from .core.models import (
        ELEMENTARY_METHODS,
        METHOD_LABELS,
        SOPHISTICATED_METHODS,
        Specification,
    )
    from .core.preprocess import PREVALENCE_LEVELS, TRANSFORMS
    from .core.validity import invalid_reason

    depth_levels, _ = builder.available_depths()
    ranks = list(builder.ranks)
    fdr_labels = [f"{method.upper()} at {threshold:g}" for method, threshold in FDR_SETTINGS]
    elementary = [METHOD_LABELS[m] for m in ELEMENTARY_METHODS]
    subsets = covariate_subsets(covariate_columns) if covariate_columns else [()]

    def card(key, number, label, levels, count, fixed=False, note=""):
        return {"key": key, "number": number, "label": label, "question": FORK_QUESTIONS[key],
                "levels": levels, "count": count, "fixed": fixed, "note": note}

    spaces = {}

    # ---- quick and full share forks 1-4 and the elementary methods ----
    matrix_keys = _matrix_keys(depth_levels, ranks, list(PREVALENCE_LEVELS), list(TRANSFORMS))
    # Numbered as the specification and the Method page number them (fork 1 to 7);
    # the order they are *applied* in is shown separately on the configure page.
    base_cards = [
        card("rarefaction", 1, "Rarefaction depth", depth_labels(depth_levels), len(depth_levels),
             note="Each random subsampling draw counts as its own level."),
        card("prev_filter", 2, "Prevalence filter", [f"{p:.0%}" for p in PREVALENCE_LEVELS],
             len(PREVALENCE_LEVELS)),
        card("transform", 3, "Transformation", [TRANSFORM_NAMES[t] for t in TRANSFORMS],
             len(TRANSFORMS)),
        card("rank", 4, "Taxonomic rank", ranks, len(ranks), fixed=len(ranks) == 1,
             note="" if len(ranks) > 1 else "One level: no taxonomy to collapse with, "
                                             "or the table is already at genus."),
    ]
    factors = [("depth draws", len(depth_levels)),
               ("prevalence filters", len(PREVALENCE_LEVELS)),
               ("transformations", len(TRANSFORMS)), ("ranks", len(ranks)),
               ("tests", len(ELEMENTARY_METHODS)), ("corrections", len(FDR_SETTINGS))]
    product = 1
    for _, n in factors:
        product *= n

    for mode in ("quick", "full"):
        report = previews.get(mode)
        cards = list(base_cards)
        extra = []
        if mode == "quick":
            cards.append(card("method", 5, "Statistical test", elementary, len(elementary),
                              note="Quick uses the four elementary tests only."))
        else:
            sampled = []
            for method in SOPHISTICATED_METHODS:
                pool = [key for key in matrix_keys if not invalid_reason(
                    Specification(key[0][0], key[0][1], key[1], key[2], key[3],
                                  method, "bh", 0.05), capabilities)]
                sample = _stratified_matrix_sample(pool, FULL_MODE_MATRIX_FRACTION,
                                                   n_total=len(matrix_keys))
                if pool:
                    sampled.append((METHOD_LABELS[method], len(sample), len(pool)))
                    extra.append((f"{METHOD_LABELS[method]}: {len(sample)} sampled matrices",
                                  len(sample) * len(FDR_SETTINGS)))
            cards.append(card("method", 5, "Statistical test",
                              elementary + [name for name, _, _ in sampled],
                              len(elementary) + len(sampled),
                              note="The three model-based methods run on a stratified "
                                   "sample of the matrices they accept."))
        cards.append(card("fdr", 6, "Multiple-testing correction", fdr_labels, len(FDR_SETTINGS),
                          note="Applied afterwards to stored p-values, so it costs no refits."))
        cards.append(card("covariates", 7, "Covariate adjustment", ["none"], 1, fixed=True,
                          note="Varied only in Covariate mode."))
        total = product + sum(n for _, n in extra)
        spaces[mode] = _space(mode, report, cards, factors, product, extra, total)

    # ---- covariate mode: a reference sub-grid crossed with covariate subsets ----
    report = previews.get("covariate")
    reference = [(name, seed) for name, seed in depth_levels
                 if name in COVARIATE_REFERENCE_RAREFACTION and (seed is None or seed == 1)]
    if not reference:
        reference = depth_levels[:1]
    cov_cards = [
        card("rarefaction", 1, "Rarefaction depth", depth_labels(reference), len(reference),
             note="A reference subset of depths, first seed only."),
        card("prev_filter", 2, "Prevalence filter", [f"{COVARIATE_REFERENCE_PREVALENCE:.0%}"], 1,
             fixed=True, note="Held at one level so the covariates are what varies."),
        card("transform", 3, "Transformation",
             [TRANSFORM_NAMES[t] for t in COVARIATE_REFERENCE_TRANSFORMS],
             len(COVARIATE_REFERENCE_TRANSFORMS)),
        card("rank", 4, "Taxonomic rank", ranks, len(ranks), fixed=len(ranks) == 1),
        card("method", 5, "Statistical test", elementary, len(elementary)),
        card("fdr", 6, "Multiple-testing correction", fdr_labels, len(FDR_SETTINGS)),
        card("covariates", 7, "Covariate adjustment",
             ["+".join(s) if s else "none" for s in subsets[:8]]
             + ([f"… {len(subsets) - 8} more"] if len(subsets) > 8 else []),
             len(subsets),
             note="Every subset of the covariates you tick: all 2^k when k ≤ 6, "
                  "otherwise 64 sampled by size."),
    ]
    cov_factors = [("depths", len(reference)), ("prevalence", 1),
                   ("transformations", len(COVARIATE_REFERENCE_TRANSFORMS)),
                   ("ranks", len(ranks)),
                   ("tests", len(ELEMENTARY_METHODS)), ("covariate sets", len(subsets)),
                   ("corrections", len(FDR_SETTINGS))]
    cov_product = 1
    for _, n in cov_factors:
        cov_product *= n
    spaces["covariate"] = _space("covariate", report, cov_cards, cov_factors, cov_product,
                                 [], cov_product)
    spaces["covariate"]["n_subsets"] = len(subsets)
    return spaces


def _space(mode, report, cards, factors, product, extra, total) -> dict:
    enumerated = report.n_enumerated if report is not None else None
    valid = report.n_valid if report is not None else None
    fits = report.n_fits if report is not None else None
    per_fit = (valid // fits) if (valid and fits and valid % fits == 0) else None
    return {
        "mode": mode, "report": report, "cards": cards,
        "factors": factors, "product": product, "extra": extra, "total": total,
        "matches": enumerated is not None and total == enumerated,
        "per_fit": per_fit,
    }


def pruning_rules() -> list:
    """The validity rules, as the engine states them, for the Method page."""
    from .core.models import METHOD_LABELS
    from .core.validity import INCOMPATIBLE, INCOMPATIBLE_REASON, RULES

    rules = []
    for method, transforms in INCOMPATIBLE.items():
        if transforms:
            names = sorted(TRANSFORM_NAMES.get(t, t) for t in transforms)
            listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
            rules.append({"rule": INCOMPATIBLE_REASON[method],
                          "excludes": f"{METHOD_LABELS[method]} given {listed} input"})
    for _, reason in RULES:
        rules.append({"rule": reason, "excludes": RULE_EXCLUDES.get(reason, "")})
    return rules


#: What each extra validity rule removes, in words. Keyed by the engine's own reason
#: text, so a rule this map does not know still appears, just without the gloss.
RULE_EXCLUDES = {
    "ANCOM-BC estimates sampling fractions; rarefaction double-corrects":
        "ANCOM-BC on rarefied data",
    "TMM + rarefaction is double library-size correction": "TMM on rarefied data",
    "presence/absence is transform-invariant; keep one canonical combination":
        "logistic regression on anything but raw counts",
    "DESeq2 models library size internally": "PyDESeq2 on rarefied data",
}
