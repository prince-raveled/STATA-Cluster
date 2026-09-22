"""JSON API — SPEC O7 ("public URL, API docs").

Everything the web interface shows is reachable programmatically. Nothing here returns
a single preferred specification (SPEC §18).
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from .. import config, db, jobs, services
from ..core.methods import available_methods
from ..core.parsers import SUPPORTED_FORMATS
from ..core.report import run_manifest, specification_curve
from ..core.robustness import locate_declared, verdict_sentence
from ..core.validation import DatasetError
from ..limits import enforce_rate_limit, run_queue

router = APIRouter(prefix="/api", tags=["multiverse"])


#: Every API error is {"error", "hint"} — the same shape the DatasetError handler
#: produces — so a client never has to branch on which layer rejected it. A missing
#: token and an unfinished run are different answers and must not share a message.
def _no_such_job():
    return JSONResponse(status_code=404, content={
        "error": "No such job.",
        "hint": "The token is unknown or its results have expired.",
    })


def _not_ready(job):
    """404 when the job does not exist, 409 when it exists but has no results yet."""
    if job is None:
        return _no_such_job()
    return JSONResponse(status_code=409, content={
        "error": job.error or "This run has not finished.",
        "hint": job.error_hint or "Poll /api/jobs/{token} until status is 'done'.",
        "status": job.status,
        "progress": round(float(job.progress or 0), 3),
    })


@router.get("/info", summary="Server capabilities")
def info():
    """What this instance supports: formats, modes, methods, and the forks varied."""
    return {
        "version": config.VERSION,
        "spec_version": config.SPEC_VERSION,
        "license": config.LICENSE,
        "retention_days": config.RETENTION_DAYS,
        "formats": list(SUPPORTED_FORMATS),
        "modes": {name: config.MODE_BLURBS[name] for name in config.MODE_LABELS},
        "methods": available_methods(),
        "forks": {
            "1_rarefaction": ["none", "min_depth", "1000", "5000", "10000"],
            "1_seeds": [1, 2, 3],
            "2_prevalence_filter": [0.0, 0.05, 0.10, 0.20],
            "3_transform": ["tss", "clr", "raw", "tmm"],
            "4_rank": ["input", "genus"],
            "5_method": list(available_methods()),
            "6_fdr": ["bh@0.05", "bh@0.10", "by@0.05"],
            "7_covariates": "covariate mode only",
        },
        "policy": {
            "best_specification_export": "never — see SPEC §18",
            "two_group_only": True,
        },
    }


@router.post("/jobs", summary="Upload a dataset and start a run", status_code=202)
async def create_job(
    request: Request,
    background: BackgroundTasks,
    abundance: UploadFile = File(..., description="Abundance table (CSV/TSV/BIOM/QZA)"),
    metadata: UploadFile = File(..., description="Sample metadata with a binary group"),
    taxonomy: UploadFile | None = File(None, description="Optional feature-ID to lineage map"),
    group_column: str = Form("", description="Metadata column to compare; inferred if blank"),
    mode: str = Form("quick", description="quick | full | covariate"),
    covariates: str = Form("", description="Comma-separated columns, covariate mode only"),
):
    enforce_rate_limit(request)
    if not run_queue.has_room():
        raise DatasetError("MicroVerse is at capacity right now.",
                           "Every analysis slot and the queue behind them are full. "
                           "Retry in a minute.")
    if mode not in config.MODE_LABELS:
        raise DatasetError(f"Unknown mode '{mode}'.",
                           "Available: " + ", ".join(config.MODE_LABELS))
    abundance_bytes = await abundance.read()
    metadata_bytes = await metadata.read()
    taxonomy_bytes = await taxonomy.read() if taxonomy and taxonomy.filename else b""
    for payload, name in ((abundance_bytes, abundance.filename),
                          (metadata_bytes, metadata.filename)):
        if len(payload) > config.MAX_UPLOAD_BYTES:
            raise DatasetError(
                f"'{name}' exceeds the {config.MAX_UPLOAD_BYTES / 1e6:.0f} MB upload limit.",
                "Collapse to genus, or run MicroVerse locally with Docker.",
            )

    columns = [c.strip() for c in covariates.split(",") if c.strip()]
    dataset = services.build_dataset(
        abundance_bytes, abundance.filename, metadata_bytes, metadata.filename,
        taxonomy_bytes, taxonomy.filename if taxonomy else "",
        group_column=group_column.strip(),
        covariates=columns or None,
    )
    token = services.create_job(dataset, abundance.filename)
    db.update_job(token, status="running", mode=mode, message="Queued")
    jobs.dispatch(background, token, mode, None,
                  tuple(columns or dataset.covariate_columns))
    return {
        "token": token,
        "status_url": f"/api/jobs/{token}",
        "results_url": f"/api/jobs/{token}/results",
        "web_url": f"/results/{token}",
        "dataset": dataset.summary(),
    }


@router.get("/jobs/{token}", summary="Job status")
def job_status(token: str):
    job = db.get_job(token)
    if job is None:
        return _no_such_job()
    payload = job.as_dict()
    payload["expires_at"] = job.expires_at.isoformat() if job.expires_at else None
    return payload


@router.get("/jobs/{token}/results", summary="Robustness table and grid counts")
def job_results(
    token: str,
    tier: str = Query("", description="Filter to one robustness tier"),
    limit: int = Query(500, ge=1, le=5000),
):
    job, run, summary, attribution = services.load_results(token)
    if run is None:
        return _not_ready(job)

    table = summary.table
    if tier:
        table = table[table["robustness_tier"] == tier.upper()]
    # to_dict hands NaN straight through, and this is rendered as JSON.
    records = services.json_safe(table.head(limit).to_dict(orient="records"))
    return {
        "token": token,
        "verdict": verdict_sentence(run, summary),
        "manifest": run_manifest(run, summary, attribution),
        "n_taxa_returned": len(records),
        "n_taxa_total": len(summary.table),
        "taxa": records,
    }


@router.get("/jobs/{token}/curve/{taxon_id}", summary="Specification curve for one taxon")
def job_curve(token: str, taxon_id: int):
    job, run, _, _ = services.load_results(token)
    if run is None:
        return _not_ready(job)
    if not 0 <= taxon_id < len(run.taxa_names):
        return JSONResponse(status_code=404, content={
            "error": f"No taxon {taxon_id} in this run.",
            "hint": f"This run has {len(run.taxa_names)} taxa, numbered 0 to "
                    f"{len(run.taxa_names) - 1}.",
        })
    return specification_curve(run, taxon_id)


@router.get("/jobs/{token}/locate/{taxon_id}", summary="Where the declared pipeline sits")
def job_locate(token: str, taxon_id: int):
    job, run, _, _ = services.load_results(token)
    if run is None:
        return _not_ready(job)
    located = locate_declared(run, taxon_id)
    if not located:
        return JSONResponse(status_code=404, content={
            "error": "No pipeline was declared for this run, or the taxon was not tested.",
        })
    return located
