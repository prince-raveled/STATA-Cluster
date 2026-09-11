"""Results dashboard, specification curve, and downloads (SPEC §16, §18)."""
from __future__ import annotations

import re

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .. import config, services
from ..core.instability import (
    FINGERPRINT_LABELS,
    analyse_taxon,
    stability_fingerprint,
)
from ..core.report import (
    attribution_csv,
    build_zip,
    citation_list,
    long_results_csv_gz,
    methods_paragraph,
    robustness_csv,
    specification_curve,
    specifications_csv,
)
from ..core.robustness import locate_declared, verdict_sentence
from ..core.validation import DatasetError, NotFoundError
from ..templating import templates

router = APIRouter()

#: Files written once at run time and streamed from disk rather than rebuilt per request.
ON_DISK = {"bundle": "bundle.zip", "long": "results_long.csv.gz"}

DOWNLOADS = {
    "robustness": ("taxa_robustness.csv", "text/csv"),
    "specifications": ("specifications.csv", "text/csv"),
    "long": ("results_long.csv.gz", "application/gzip"),
    "attribution": ("attribution.csv", "text/csv"),
    "methods": ("methods.txt", "text/plain; charset=utf-8"),
    "bundle": ("microverse_results.zip", "application/zip"),
}


#: Anything outside this set is stripped from a download filename. The name comes
#: from an uploaded file, and it is interpolated into a response header.
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(name: str) -> str:
    stem = _UNSAFE_IN_FILENAME.sub("_", (name or "microverse").split(".")[0])
    return stem.strip("._-")[:40] or "microverse"


def _require(token: str):
    job, run, summary, attribution = services.load_results(token)
    if job is None:
        raise NotFoundError("That job no longer exists.",
                           f"Results are kept for {config.RETENTION_DAYS} days.")
    if run is None:
        raise DatasetError(
            "That run has not finished yet." if job.status != "error"
            else (job.error or "That run failed."),
            job.error_hint or f"Check progress at /job/{token}.",
        )
    return job, run, summary, attribution


@router.get("/results/{token}", response_class=HTMLResponse, include_in_schema=False)
def results_page(request: Request, token: str):
    job, run, summary, attribution = _require(token)

    spec_counts = summary.spec_summary["n_significant"].to_numpy()
    histogram, edges = np.histogram(spec_counts, bins=min(40, max(5, len(set(spec_counts)))))

    return templates.TemplateResponse(
        request, "results.html",
        {
            "job": job, "token": token, "run": run, "summary": summary,
            "attribution": attribution,
            "verdict": verdict_sentence(run, summary),
            "methods_text": methods_paragraph(run, summary, attribution),
            "grid": run.grid_report,
            "dataset": run.dataset_summary,
            "taxa": services.taxon_options(summary),
            "records": services.table_records(summary),
            "spec_histogram": {"counts": histogram.tolist(),
                               "edges": [float(e) for e in edges]},
            "spec_stats": {
                "min": int(spec_counts.min()), "max": int(spec_counts.max()),
                "median": float(np.median(spec_counts)),
                "declared": summary.declared_n_significant,
            },
            "declared_spec": (run.specs[run.declared_spec_id]
                              if run.declared_spec_id >= 0 else None),
            "citations": citation_list(),
        },
    )


@router.get("/results/{token}/taxon/{taxon_id}", response_class=HTMLResponse,
            include_in_schema=False)
def taxon_evidence(request: Request, token: str, taxon_id: int):
    """Everything known about one taxon, in one place.

    The robustness table gives a tier and a few numbers; this answers the question a
    researcher actually has next — why did this taxon's answer move, and which choice
    moved it — without making them open a CSV.
    """
    job, run, summary, attribution = _require(token)
    if not 0 <= taxon_id < len(run.taxa_names):
        raise NotFoundError(
            f"No taxon {taxon_id} in this run.",
            f"This run has {len(run.taxa_names)} taxa.")

    rows = summary.table[summary.table["taxon_id"] == taxon_id]
    if rows.empty:
        raise NotFoundError(
            "That taxon was filtered out of every specification, so it has no results.",
            "Lower the prevalence filter, or pick a taxon from the table.")
    row = rows.iloc[0]

    evidence = analyse_taxon(run, taxon_id)
    ranked = list(summary.table["taxon_id"]).index(taxon_id)
    return templates.TemplateResponse(
        request, "taxon.html",
        {
            "job": job, "token": token, "run": run, "summary": summary,
            "row": row, "evidence": evidence,
            "fingerprint": stability_fingerprint(row),
            "fingerprint_labels": FINGERPRINT_LABELS,
            "declared": locate_declared(run, taxon_id),
            "rank_in_table": ranked + 1,
            "n_taxa": len(summary.table),
            "neighbours": services.taxon_neighbours(summary, taxon_id),
        },
    )


@router.get("/results/{token}/curve/{taxon_id}", include_in_schema=False)
def curve(token: str, taxon_id: int):
    _, run, _, _ = _require(token)
    return JSONResponse(specification_curve(run, taxon_id))


@router.get("/download/{token}/{kind}", include_in_schema=False)
def download(token: str, kind: str):
    if kind not in DOWNLOADS:
        raise NotFoundError(
            f"Unknown download '{kind}'.",
            "Available: " + ", ".join(DOWNLOADS) +
            ". There is no 'best specification' export, by design (SPEC §18).",
        )
    job, run, summary, attribution = _require(token)
    filename, media_type = DOWNLOADS[kind]
    disposition = f'attachment; filename="{_safe_stem(job.dataset_name)}_{filename}"'

    # The two large artefacts are written once when the run finishes; streaming them
    # from disk keeps a download off the heap for a table with thousands of taxa.
    if kind in ON_DISK:
        path = config.job_dir(token) / ON_DISK[kind]
        if path.exists():
            return FileResponse(path, media_type=media_type,
                                headers={"Content-Disposition": disposition})

    if kind == "robustness":
        payload = robustness_csv(summary)
    elif kind == "specifications":
        payload = specifications_csv(run, summary)
    elif kind == "long":
        payload = long_results_csv_gz(run)
    elif kind == "attribution":
        payload = attribution_csv(attribution or {})
    elif kind == "bundle":
        payload = build_zip(run, summary, attribution)
    else:
        payload = methods_paragraph(run, summary, attribution).encode()

    return Response(content=payload, media_type=media_type,
                    headers={"Content-Disposition": disposition})
