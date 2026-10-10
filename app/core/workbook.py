"""The Excel workbook — every table and every explanation of a run, in one file.

The results page shows figures and one-line captions; the reasoning behind them lives
here, so nothing a researcher needs is lost when it leaves the page. One sheet per
subject, a README sheet first that says what each one holds and how the run was made.

The specification x taxon table is not a sheet: on larger runs it passes Excel's
1,048,576-row limit (the species demo alone is about 1.25 million rows), so it stays the
compressed CSV in the downloads and the bundle, and the README says so.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

#: Read before reporting anything from a run. The page shows `short`; the workbook,
#: the bundle README and anyone asking for the reasoning get `full`.
INTERPRETATION = [
    {"key": "biology", "short": "Robust is not biologically real.",
     "full": "Robust to analytical choice is not biologically real. Confounding, "
             "contamination and batch effects survive every analysis intact, and nothing "
             "here is causal. A difference that survives every analytical choice is still "
             "an association, and statistical stability is not importance."},
    {"key": "probability", "short": "Percentages are not probabilities.",
     "full": "The percentages are not probabilities. The analyses are re-analyses of the "
             "same samples: “significant in 86%” measures how much the conclusion "
             "depends on your choices, not 86 replications, not a p-value and not a chance "
             "that the difference is real."},
    {"key": "magnitude", "short": "A label ignores effect size.",
     "full": "A label ignores magnitude. A tier is built from two numbers, how often a "
             "taxon was significant and how often the direction agreed. Effect size is not "
             "one of them, so a ROBUST taxon can have a difference far too small to matter. "
             "Read the typical effect beside it."},
    {"key": "responsibility", "short": "A label does not say which choice moved it.",
     "full": "A tier says the answer moved, not what moved it. Open a taxon, or the "
             "specification curve, to see which analytical choice is responsible."},
    {"key": "transport", "short": "Labels describe this study, not other populations.",
     "full": "Labels measure dependence on analytical choices in these data. Across "
             "independent studies of the same disease they did not predict which findings "
             "replicated (13% for ROBUST or CONDITIONAL against 14% for the rest, 19 "
             "published cohorts; see the Evidence page)."},
]

#: Every file a run offers, described once for the page, the README sheet and the zip.
FILES = [
    ("microverse_results.xlsx", "This workbook: every table and explanation, one sheet each."),
    ("microverse_results.zip", "Everything, including the full specification x taxon table "
                               "and the manifest."),
    ("taxa_robustness.csv", "One row per taxon: label, how often significant, direction "
                            "agreement, typical effect and range."),
    ("results_long.csv.gz", "One row per taxon per specification: raw and adjusted p-value, "
                            "significance, the common effect and each test's native "
                            "statistic. Too long for Excel on larger runs."),
    ("specifications.csv", "One row per specification: its choices, weights, taxa tested and "
                           "taxa called significant."),
    ("attribution.csv", "The share of variation attributed to each choice, with both "
                        "estimators and their intervals."),
    ("calibration_rejections.csv", "Calibrated runs: every specification rejected with error "
                                   "control for every discovered taxon."),
    ("methods.txt", "A description of the multiverse for a paper, with citations."),
    ("manifest.json", "Versions, every random seed, the grid accounting and a fingerprint of "
                      "the input: what is needed to reproduce the run."),
]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Excel cells cannot hold NaN or infinity as numbers; leave them empty."""
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].where(np.isfinite(frame[column]), None)
    return frame


def _kv(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Item", "Value"])


def readme_sheet(run, summary) -> pd.DataFrame:
    from .report import dependency_versions

    dataset = run.dataset_summary
    grid = run.grid_report
    calibration = getattr(summary, "calibration", None)
    rows = [
        ("What this is", "MicroVerse results: every defensible analysis of one dataset, "
                         "and how each taxon held up across them."),
        ("Dataset", f"{dataset.get('n_samples')} samples ({dataset.get('n_group_a')} "
                    f"{dataset.get('group_a')}, {dataset.get('n_group_b')} "
                    f"{dataset.get('group_b')}), {dataset.get('n_taxa')} taxa"),
        ("Mode", run.mode),
        ("Pruning rules", getattr(run, "ruleset", "v2")),
        ("Weighting of the analyses", getattr(summary, "scheme", "uniform")),
        ("Specifications run", f"{grid.n_valid:,} of {grid.n_enumerated:,} enumerated"),
        ("Model fits", f"{grid.n_fits:,} on {grid.n_matrices:,} preprocessed matrices"),
        ("Calibrated error control",
         (f"{calibration.n_permutations:,} permutations, seed {calibration.seed}, "
          f"FDR {calibration.q:g} across taxa") if calibration is not None else "not run"),
        ("Run time (engine)", f"{run.runtime_seconds:.1f} s"),
        ("Python", dependency_versions()["python"]),
        ("", ""),
        ("Sheet", "Contents"),
        ("Summary", "The verdict and the count of taxa in each label."),
        ("Taxa", "One row per taxon: every metric, its label under each weighting, and "
                 "calibration results where run."),
        ("Specifications", "One row per analysis: its choices, its weight under each scheme, "
                           "taxa tested and called significant."),
        ("Grid", "How the analyses were counted: enumerated, ruled out (with the rule and "
                 "reason), run."),
        ("Attribution", "Which analytical choice the variation is attributed to (exploratory)."),
        ("Calibration", "Calibrated runs only: the settings and the guarantee's assumptions."),
        ("Methods", "The methods paragraph and its references."),
        ("Interpretation", "What a label does and does not mean. Read before reporting."),
        ("Glossary", "Every term used on the results page."),
        ("", ""),
        ("Not in this workbook", "The specification x taxon table (results_long.csv.gz): it "
                                 "exceeds Excel's row limit on larger runs. It is in the "
                                 "bundle and the downloads."),
    ]
    return _kv(rows)


def summary_sheet(run, summary) -> pd.DataFrame:
    from .robustness import TIER_ORDER, verdict_sentence

    rows = [("Verdict", verdict_sentence(run, summary)), ("", "")]
    rows += [(f"Taxa {tier}", int(summary.tier_counts.get(tier, 0))) for tier in TIER_ORDER]
    calibration = getattr(summary, "calibration", None)
    if calibration is not None:
        rows += [("Taxa tested with calibration", calibration.n_taxa_tested),
                 ("Discoveries (FDR-controlled)", calibration.n_selected),
                 ("CERTIFIED ROBUST", calibration.n_certified)]
    changes = getattr(summary, "tier_changes", None) or {}
    for scheme, n in changes.items():
        rows.append((f"Taxa labelled differently under {scheme}", int(n)))
    return _kv(rows)


def grid_sheet(run) -> pd.DataFrame:
    from .validity import rule_id

    report = run.grid_report
    rows = [{"Step": "Enumerated", "Rule": "", "Specifications": report.n_enumerated,
             "Reason": "every combination of the choices"}]
    for reason, n in report.pruned_reasons.items():
        rows.append({"Step": "Ruled out", "Rule": rule_id(reason), "Specifications": -int(n),
                     "Reason": reason})
    rows.append({"Step": "Run", "Rule": "", "Specifications": report.n_valid,
                 "Reason": "statistically coherent and executed"})
    rows.append({"Step": "Model fits", "Rule": "", "Specifications": report.n_fits,
                 "Reason": "distinct models (FDR settings share one fit)"})
    for note in report.notes:
        rows.append({"Step": "Note", "Rule": "", "Specifications": None, "Reason": note})
    return pd.DataFrame(rows)


def specifications_sheet(run, summary) -> pd.DataFrame:
    frame = summary.spec_summary.copy()
    frame.insert(1, "description", [run.specs[i].describe() for i in frame["spec_id"]])
    return frame


def attribution_sheet(attribution) -> pd.DataFrame:
    from .report import attribution_csv

    if not attribution:
        return pd.DataFrame({"Note": ["Attribution was not computed for this run."]})
    return pd.read_csv(io.BytesIO(attribution_csv(attribution)))


def calibration_sheet(summary) -> pd.DataFrame:
    calibration = getattr(summary, "calibration", None)
    if calibration is None:
        return pd.DataFrame({"Note": ["This run was not calibrated. Tick “Also calibrate "
                                      "the error rate” on the configure page to add it."]})
    from .report import calibration_manifest

    manifest = calibration_manifest(calibration)
    rows = []
    for key, value in manifest.items():
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value, 1):
                rows.append((f"{key} {i}", str(item)))
        elif isinstance(value, dict):
            for sub, item in value.items():
                rows.append((f"{key}: {sub}", str(item)))
        else:
            rows.append((key, "" if value is None else str(value)))
    return _kv(rows)


def methods_sheet(run, summary, attribution=None) -> pd.DataFrame:
    from .report import citation_list, methods_paragraph

    rows = [("Methods paragraph", p) for p in
            methods_paragraph(run, summary, attribution).split("\n\n")]
    rows += [("", "")] + [(f"Reference {i}", c) for i, c in enumerate(citation_list(), 1)]
    return _kv(rows)


def interpretation_sheet() -> pd.DataFrame:
    from .evidence import TIER_CAVEATS

    rows = [(item["short"], item["full"]) for item in INTERPRETATION]
    rows += [("", "")] + [(f"Label caveat {i}", c) for i, c in enumerate(TIER_CAVEATS, 1)]
    return pd.DataFrame(rows, columns=["In one line", "In full"])


def glossary_sheet() -> pd.DataFrame:
    from .glossary import glossary_sections

    rows = []
    for section in glossary_sections():
        for entry in section["entries"]:
            rows.append({"Section": section["heading"], "Term": entry.get("term", ""),
                         "Definition": entry.get("short", ""),
                         "More": entry.get("long", "")})
    return pd.DataFrame(rows)


def sheets(run, summary, attribution=None) -> dict:
    """Sheet name -> DataFrame, in the order the workbook shows them."""
    return {
        "README": readme_sheet(run, summary),
        "Summary": summary_sheet(run, summary),
        "Taxa": summary.table,
        "Specifications": specifications_sheet(run, summary),
        "Grid": grid_sheet(run),
        "Attribution": attribution_sheet(attribution),
        "Calibration": calibration_sheet(summary),
        "Methods": methods_sheet(run, summary, attribution),
        "Interpretation": interpretation_sheet(),
        "Glossary": glossary_sheet(),
    }


#: Column widths by content: long text wraps, identifiers stay narrow.
_TEXT_COLUMNS = {"Value": 100, "In full": 100, "Reason": 80, "Definition": 60, "More": 80,
                 "description": 70, "Note": 90}


def build_workbook(run, summary, attribution=None) -> bytes:
    """The workbook as bytes (.xlsx)."""
    from openpyxl.styles import Alignment, Font, PatternFill

    buffer = io.BytesIO()
    header_fill = PatternFill("solid", fgColor="E6EEEC")
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets(run, summary, attribution).items():
            _clean(frame).to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.font = Font(bold=True)
                cell.fill = header_fill
            for column_cells in sheet.columns:
                header = str(column_cells[0].value or "")
                width = _TEXT_COLUMNS.get(header)
                if width is None:
                    longest = max((len(str(c.value)) for c in column_cells[:200]
                                   if c.value is not None), default=8)
                    width = min(max(10, longest + 2), 40)
                sheet.column_dimensions[column_cells[0].column_letter].width = width
                if header in _TEXT_COLUMNS:
                    for cell in column_cells[1:]:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
    return buffer.getvalue()
