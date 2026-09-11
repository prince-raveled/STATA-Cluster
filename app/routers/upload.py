"""Landing page, upload, demo datasets, run configuration."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import config, db, services
from ..core.parsers import SUPPORTED_FORMATS
from ..core.validation import DatasetError, NotFoundError
from ..limits import enforce_rate_limit, run_queue
from ..templating import templates

router = APIRouter()


async def _read(upload: UploadFile | None) -> bytes:
    if upload is None or not upload.filename:
        return b""
    payload = await upload.read()
    if len(payload) > config.MAX_UPLOAD_BYTES:
        raise DatasetError(
            f"'{upload.filename}' is {len(payload) / 1e6:.0f} MB, above the "
            f"{config.MAX_UPLOAD_BYTES / 1e6:.0f} MB upload limit.",
            "Collapse to genus before uploading, or run MicroVerse locally with Docker "
            "where the limit does not apply.",
        )
    return payload


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing(request: Request):
    return templates.TemplateResponse(
        request, "index.html",
        {"demos": services.DEMO_DATASETS, "formats": SUPPORTED_FORMATS},
    )


@router.post("/upload", include_in_schema=False)
async def upload(
    request: Request,
    abundance: UploadFile = File(...),
    metadata: UploadFile = File(...),
    taxonomy: UploadFile | None = File(None),
    group_column: str = Form(""),
):
    enforce_rate_limit(request)
    abundance_bytes = await _read(abundance)
    metadata_bytes = await _read(metadata)
    taxonomy_bytes = await _read(taxonomy)
    if not abundance_bytes:
        raise DatasetError("No abundance table was uploaded.",
                           "Supported: " + ", ".join(SUPPORTED_FORMATS))
    if not metadata_bytes:
        raise DatasetError("No sample metadata was uploaded.",
                           "MicroVerse needs a table of sample IDs and a binary grouping "
                           "column to compare two groups.")

    dataset = services.build_dataset(
        abundance_bytes, abundance.filename,
        metadata_bytes, metadata.filename,
        taxonomy_bytes, taxonomy.filename if taxonomy else "",
        group_column=group_column.strip(),
    )
    token = services.create_job(dataset, abundance.filename)
    return RedirectResponse(f"/configure/{token}", status_code=303)


@router.get("/demo/{name}", include_in_schema=False)
def demo(name: str):
    dataset = services.load_demo(name)
    token = services.create_job(dataset, services.DEMO_DATASETS[name]["title"])
    return RedirectResponse(f"/configure/{token}", status_code=303)


@router.get("/configure/{token}", response_class=HTMLResponse, include_in_schema=False)
def configure(request: Request, token: str):
    job = db.get_job(token)
    dataset = db.load_payload(token, "dataset")
    if job is None or dataset is None:
        raise NotFoundError(
            "That job no longer exists.",
            f"Results and uploads are kept for {config.RETENTION_DAYS} days.",
        )
    from ..core.grid import capabilities_for, enumerate_grid
    from ..core.preprocess import MatrixBuilder
    from ..core.readiness import assess

    builder = MatrixBuilder(dataset)
    capabilities = capabilities_for(dataset)
    depth_levels, dropped_depths = builder.available_depths()
    previews = {}
    for mode in ("quick", "full", "covariate"):
        try:
            specs, report = enumerate_grid(
                builder, mode=mode, capabilities=capabilities,
                covariate_columns=dataset.covariate_columns,
            )
            previews[mode] = report
        except Exception:
            previews[mode] = None

    return templates.TemplateResponse(
        request, "configure.html",
        {
            "job": job, "token": token, "dataset": dataset,
            "summary": dataset.summary(), "previews": previews,
            "readiness": assess(dataset),
            "ranks": builder.ranks,
            "depths": list(dict.fromkeys(name for name, _ in depth_levels)),
            "dropped_depths": dropped_depths,
        },
    )


@router.post("/run/{token}", include_in_schema=False)
async def start_run(request: Request, token: str, background: BackgroundTasks):
    enforce_rate_limit(request)
    if not run_queue.has_room():
        raise DatasetError(
            "MicroVerse is at capacity right now.",
            "Every analysis slot and the queue behind them are full. Your upload is "
            "still here — wait a minute and start the run again.",
        )
    form = await request.form()
    mode = str(form.get("mode") or "quick")
    if mode not in config.MODE_LABELS:
        mode = "quick"
    # getlist, not dict(): the covariate checkboxes repeat one key, and collapsing the
    # form to a dict keeps only the last of them.
    covariates = [str(v) for v in form.getlist("covariates")]
    declared = services.parse_declared({k: form.get(k) for k in form})

    job = db.get_job(token)
    if job is None:
        raise NotFoundError("That job no longer exists.",
                           f"Uploads are kept for {config.RETENTION_DAYS} days.")

    db.update_job(token, status="running", mode=mode, progress=0.0,
                  message="Queued", error="", error_hint="")
    background.add_task(services.execute, token, mode, declared, tuple(covariates))
    return RedirectResponse(f"/job/{token}", status_code=303)
