"""Results dashboard, specification curve, and downloads (SPEC §16, §18)."""
from __future__ import annotations

import re

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)

from .. import config, db, services, storage
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
    run_manifest,
    specification_curve,
    specifications_csv,
)
from ..core.robustness import (
    locate_declared,
    n_detected,
    n_stable,
    verdict_sentence,
)
from ..core.validation import DatasetError, NotFoundError, RunNotReadyError
from ..templating import templates

router = APIRouter()

#: Files written once at run time and served from where they were stored, never rebuilt
#: per request. On object storage they exceed a function's response cap, so the
#: browser is sent to the store for them (`_stored_download`).
ON_DISK = {"bundle": "bundle.zip", "long": "results_long.csv.gz"}

DOWNLOADS = {
    "manifest": ("manifest.json", "application/json"),
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
    """Load a finished run, or raise the error that says precisely what is missing.

    Three outcomes a caller must be able to tell apart: no such token (404), a valid
    token whose run is still going or has failed (409), and results (200). Collapsing
    the middle case into 422 reads as "your request was malformed", which it was not.
    """
    job, run, summary, attribution = services.load_results(token)
    if job is None:
        raise NotFoundError("That job no longer exists.",
                           f"Results are kept for {config.RETENTION_DAYS} days.")
    if run is None:
        raise RunNotReadyError(
            "That run has not finished yet." if job.status != "error"
            else (job.error or "That run failed."),
            job.error_hint or f"Check progress at /job/{token}.",
        )
    return job, run, summary, attribution


def _require_taxon(run, taxon_id: int) -> None:
    """Bounds-check a user-supplied taxon id. Shared so the HTML and API routes
    cannot drift apart — they did, and one of them returned a 500."""
    if not 0 <= taxon_id < len(run.taxa_names):
        raise NotFoundError(
            f"No taxon {taxon_id} in this run.",
            f"This run has {len(run.taxa_names)} taxa, numbered 0 to "
            f"{len(run.taxa_names) - 1}.")


@router.get("/results/{token}", response_class=HTMLResponse, include_in_schema=False)
def results_page(request: Request, token: str):
    # A shared results link opened while the run is still going belongs on the progress
    # page, not on an error page. Only a genuinely missing token is an error here.
    try:
        job, run, summary, attribution = _require(token)
    except RunNotReadyError:
        return RedirectResponse(f"/job/{token}", status_code=303)

    spec_counts = summary.spec_summary["n_significant"].to_numpy()
    histogram, edges = np.histogram(spec_counts, bins=min(40, max(5, len(set(spec_counts)))))

    # A run that detected nothing is a valid scientific answer, and the page says so
    # rather than rendering a row of zeros. The readiness checks usually already
    # predicted it on the configure page, so they are re-run here and shown alongside —
    # they are pure arithmetic over the stored table and cost nothing measurable.
    detected = n_detected(summary)
    stable = n_stable(summary)
    readiness = None
    if stable == 0:
        try:
            from ..core.readiness import assess
            dataset = db.load_payload(token, "dataset")
            if dataset is not None:
                readiness = assess(dataset)
        except Exception:                                   # noqa: BLE001
            readiness = None       # never let a diagnostic block the results page

    return templates.TemplateResponse(
        request, "results.html",
        {
            "job": job, "token": token, "run": run, "summary": summary,
            "attribution": attribution,
            "verdict": verdict_sentence(run, summary),
            "n_detected": detected,
            "n_stable": stable,
            "readiness": readiness,
            "methods_text": methods_paragraph(run, summary, attribution),
            "grid": run.grid_report,
            "dataset": run.dataset_summary,
            "taxa": services.taxon_options(summary),
            "taxa_listed_all": len(summary.table) <= services.TAXON_OPTION_LIMIT,
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
    _require_taxon(run, taxon_id)

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
    """The curve the results page fetches. Always JSON, including its errors.

    This is consumed by `fetch`, so an HTML error page here is useless to the caller.
    The bounds check is the same one the public API applies — without it an
    out-of-range id reached `specification_curve` and surfaced as a 500.
    """
    try:
        _, run, _, _ = _require(token)
        _require_taxon(run, taxon_id)
    except DatasetError as exc:
        return JSONResponse(status_code=getattr(exc, "status_code", 422),
                            content={"error": exc.message, "hint": exc.hint})
    return JSONResponse(specification_curve(run, taxon_id))


def _require_finished_job(token: str):
    """The job row alone, with `_require`'s answers, for a download that never needs
    the run itself: loading three pickles to serve a file that is already stored is
    work a large export can do without."""
    job = db.get_job(token)
    if job is None:
        raise NotFoundError("That job no longer exists.",
                            f"Results are kept for {config.RETENTION_DAYS} days.")
    if job.status != "done":
        raise RunNotReadyError(
            "That run has not finished yet." if job.status != "error"
            else (job.error or "That run failed."),
            job.error_hint or f"Check progress at /job/{token}.",
        )
    return job


def _stored_download(token: str, kind: str):
    """Serve one of the two large exports from wherever the run stored it.

    They are written once when the run finishes, and on object storage they are
    larger than a function may return (4.5 MB on Vercel; the bundle for a demo is
    8-11 MB). So on disk they are handed over as a file, and in the store the browser
    is sent to the object itself -- never streamed through here, and never rebuilt
    here, which would produce the same oversized response.
    """
    job = _require_finished_job(token)
    stored = ON_DISK[kind]
    filename, media_type = DOWNLOADS[kind]

    if storage.is_local():
        # A read creates nothing: the path is only looked at.
        path = storage.LocalBackend.path(token, stored)
        if path.exists():
            disposition = f'attachment; filename="{_safe_stem(job.dataset_name)}_{filename}"'
            return FileResponse(path, media_type=media_type,
                                headers={"Content-Disposition": disposition})
        return None       # a run from before these were stored: rebuilt below, as ever

    size = storage.size(token, stored)
    if size is None:
        raise NotFoundError(
            "This export is no longer stored.",
            f"Results are kept for {config.RETENTION_DAYS} days. The robustness table, "
            "specifications and manifest can still be downloaded individually.",
        )
    if size <= config.APP_RESPONSE_LIMIT_BYTES:
        # Small enough to fit a function response, so it comes from here: one request
        # fewer, and it works where a network blocks the store's own domain.
        data = storage.get_bytes(token, stored)
        if data is not None:
            disposition = f'attachment; filename="{_safe_stem(job.dataset_name)}_{filename}"'
            return Response(content=data, media_type=media_type,
                            headers={"Content-Disposition": disposition})
    direct = storage.download_url(token, stored)
    if not direct:
        error = DatasetError(
            "This deployment cannot produce a download link for this file.",
            "The file is too large to send through the application. This is a "
            "configuration problem on the server, not a problem with your results.",
        )
        error.status_code = 503
        raise error
    return RedirectResponse(direct, status_code=307)


@router.get("/download/{token}/{kind}", include_in_schema=False)
def download(token: str, kind: str):
    if kind not in DOWNLOADS:
        raise NotFoundError(
            f"Unknown download '{kind}'.",
            "Available: " + ", ".join(DOWNLOADS) +
            ". There is no 'best specification' export, by design (SPEC §18).",
        )
    if kind in ON_DISK:
        served = _stored_download(token, kind)
        if served is not None:
            return served

    job, run, summary, attribution = _require(token)
    filename, media_type = DOWNLOADS[kind]
    disposition = f'attachment; filename="{_safe_stem(job.dataset_name)}_{filename}"'

    if kind == "manifest":
        import json
        payload = json.dumps(run_manifest(run, summary, attribution),
                             indent=2, default=str).encode("utf-8")
    elif kind == "robustness":
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

