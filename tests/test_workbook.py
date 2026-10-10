"""The Excel workbook: every table and every explanation the results page moved off it."""
from __future__ import annotations

import io

import pytest

openpyxl = pytest.importorskip("openpyxl")

from app.core.attribution import attribute  # noqa: E402
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.workbook import INTERPRETATION, build_workbook, sheets  # noqa: E402

SHEETS = ["README", "Summary", "Taxa", "Specifications", "Grid", "Attribution",
          "Calibration", "Methods", "Interpretation", "Glossary"]


@pytest.fixture(scope="module")
def outputs(dataset):
    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    return run, summary, attribute(run, bootstrap=False)


def test_every_sheet_is_written_in_order(outputs):
    run, summary, attribution = outputs
    book = openpyxl.load_workbook(io.BytesIO(build_workbook(run, summary, attribution)),
                                  read_only=True)
    assert book.sheetnames == SHEETS


def test_tables_are_complete(outputs):
    run, summary, attribution = outputs
    frames = sheets(run, summary, attribution)
    assert len(frames["Taxa"]) == len(summary.table)
    assert len(frames["Specifications"]) == len(run.specs)
    grid = frames["Grid"]
    assert int(grid.loc[grid["Step"] == "Run", "Specifications"].iloc[0]) == run.grid_report.n_valid


def test_the_interpretation_moved_off_the_page_is_kept_in_full(outputs):
    run, summary, attribution = outputs
    sheet = sheets(run, summary, attribution)["Interpretation"]
    for item in INTERPRETATION:
        assert item["full"] in set(sheet["In full"])


def test_an_uncalibrated_run_says_so(outputs):
    run, summary, attribution = outputs
    note = sheets(run, summary, attribution)["Calibration"]["Note"].iloc[0]
    assert "not calibrated" in note
