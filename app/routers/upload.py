"""Landing page, upload, demo datasets, run configuration."""
from __future__ import annotations

import hmac

from fastapi import APIRouter, BackgroundTasks, File, Form, Header, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .. import config, db, jobs, services, storage, ui, uploads
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
        {"demos": services.DEMO_DATASETS, "formats": SUPPORTED_FORMATS,
         "grid_sizes": services.demo_grid_sizes()},
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
    return RedirectResponse(f"/validate/{token}", status_code=303)


# --- direct upload ---------------------------------------------------------
# Three steps instead of one, for hosts whose request-body cap is smaller than
# MicroVerse's upload limit. The form above still works and is still what a browser
# without JavaScript posts; this path exists so the bytes can go to storage directly
# instead of through the application. Validation is unchanged: `complete` hands the
# staged bytes to the same `build_dataset` the form post uses.
@router.post("/upload/authorize", include_in_schema=False)
async def authorize(request: Request):
    """Say where these files may be written. Grants nothing beyond those objects."""
    enforce_rate_limit(request)
    try:
        body = await request.json()
    except Exception:                                        # noqa: BLE001
        raise DatasetError("The upload request was malformed.",
                           "Reload the page and choose your files again.") from None

    files = uploads.validate_request(body.get("files") or {})
    ticket = uploads.issue(files)
    staging = uploads.staging_token(uploads.verify(ticket)["id"])
    try:
        targets = {
            field: storage.authorize_upload(staging, field, entry["size"])
            for field, entry in files.items()
        }
    except storage.DirectUploadUnavailable:
        # This backend cannot grant the browser permission to write. Say so plainly
        # rather than improvising a credential: the script falls back to posting the
        # form, which works up to whatever the host caps a request body at.
        return JSONResponse(
            status_code=501,
            content={"error": "Direct upload is not available on this deployment.",
                     "hint": "The file will be sent through the application instead.",
                     "fallback": "form"},
        )
    return {"ticket": ticket, "uploads": targets}


@router.post("/upload/ticket", include_in_schema=False)
async def ticket(request: Request, x_microverse_worker: str = Header(default="")):
    """Decide whether a ticket authorises one pathname. Called by the token minter.

    The minting endpoint holds the store credential but none of the rules. This is
    where the rules are, and it answers one question: may the browser holding this
    ticket write to exactly this pathname, and under what limits. Keeping the answer
    here means the size limit, the expiry and the field scoping have one
    implementation rather than two that must be kept in agreement.

    The pathname is the part that matters. `upload()` sends whatever the browser
    passed it, so it is checked against the one the ticket implies rather than
    trusted — otherwise a valid ticket would authorise writing anywhere in the store.
    """
    if not config.WORKER_SECRET or not hmac.compare_digest(
            str(x_microverse_worker or ""), config.WORKER_SECRET):
        return JSONResponse({"error": "Not authorised."}, status_code=403)

    try:
        body = await request.json()
    except Exception:                                        # noqa: BLE001
        return JSONResponse({"error": "Malformed request."}, status_code=400)

    payload = uploads.verify(body.get("ticket"))
    if payload is None:
        return JSONResponse({"error": "That upload has expired."}, status_code=403)

    presented = str(body.get("pathname") or "")
    staging = uploads.staging_token(payload["id"])
    allowed = {
        storage.backend().authorize(staging, field, entry["size"])["pathname"]: entry
        for field, entry in payload["files"].items()
    }
    entry = allowed.get(presented)
    if entry is None:
        return JSONResponse(
            {"error": "That upload is not authorised for this location."},
            status_code=403,
        )

    return {
        # The store enforces this, so a browser that lied about the size at
        # /upload/authorize still cannot write more than the application allows.
        "maximum_size_in_bytes": config.MAX_UPLOAD_BYTES,
        "allowed_content_types": list(uploads.ALLOWED_CONTENT_TYPES),
        # Milliseconds, and never beyond the life of the ticket that authorised it.
        "valid_until": int(payload["exp"]) * 1000,
    }


@router.put("/upload/staged/{token}/{field}", include_in_schema=False)
async def staged(request: Request, token: str, field: str,
                 x_microverse_ticket: str = Header(default="")):
    """Accept one staged file. Only reachable when storage is a local directory.

    A host with real object storage never routes here — the browser writes straight to
    the store — so this is the development and test path, and it is held to the same
    rules: a valid ticket, a field that ticket names, a key the server derived, and a
    body no larger than the limit that was checked before the ticket was issued.
    """
    payload = uploads.verify(x_microverse_ticket)
    if payload is None or uploads.staging_token(payload["id"]) != token:
        raise NotFoundError("That upload has expired.",
                            "Reload the page and choose your files again.")
    if field not in payload["files"]:
        raise NotFoundError("That upload has expired.",
                            "Reload the page and choose your files again.")

    data = await request.body()
    uploads.check_size(payload["files"][field]["filename"], len(data))
    storage.put_bytes(token, field, data)
    return {"field": field, "bytes": len(data)}


@router.post("/upload/complete", include_in_schema=False)
async def complete(request: Request):
    """Turn staged bytes into a job, using exactly the form post's validation."""
    enforce_rate_limit(request)
    try:
        body = await request.json()
    except Exception:                                        # noqa: BLE001
        raise DatasetError("The upload request was malformed.",
                           "Reload the page and choose your files again.") from None

    payload = uploads.verify(body.get("ticket"))
    if payload is None:
        raise DatasetError(
            "That upload has expired.",
            "Uploads must be completed within "
            f"{uploads.TICKET_TTL_SECONDS // 60} minutes. Choose your files again.")

    staging = uploads.staging_token(payload["id"])
    names = payload["files"]
    try:
        staged_bytes = {field: storage.get_bytes(staging, field) for field in names}
        if not staged_bytes.get("abundance"):
            raise DatasetError("No abundance table was uploaded.",
                               "Supported: " + ", ".join(SUPPORTED_FORMATS))
        if not staged_bytes.get("metadata"):
            raise DatasetError("No sample metadata was uploaded.",
                               "MicroVerse needs a table of sample IDs and a binary "
                               "grouping column to compare two groups.")

        dataset = services.build_dataset(
            staged_bytes["abundance"], names["abundance"]["filename"],
            staged_bytes["metadata"], names["metadata"]["filename"],
            staged_bytes.get("taxonomy"),
            names.get("taxonomy", {}).get("filename", ""),
            group_column=str(body.get("group_column") or "").strip(),
        )
        token = services.create_job(dataset, names["abundance"]["filename"])
    finally:
        # The staged copies have either become a dataset or failed validation. Either
        # way nothing reads them again, and leaving them costs storage quota.
        storage.purge(staging)

    return {"token": token, "next": f"/validate/{token}"}


@router.post("/upload/abandon", include_in_schema=False)
async def abandon(request: Request):
    """Drop whatever a cancelled upload already staged. Best effort, always 200."""
    try:
        body = await request.json()
        payload = uploads.verify(body.get("ticket"))
    except Exception:                                        # noqa: BLE001
        payload = None
    if payload is not None:
        storage.purge(uploads.staging_token(payload["id"]))
    return {"ok": True}


@router.get("/demo/{name}", include_in_schema=False)
def demo(name: str):
    dataset = services.load_demo(name)
    token = services.create_job(dataset, services.DEMO_DATASETS[name]["title"])
    return RedirectResponse(f"/validate/{token}", status_code=303)


def _setup(token: str) -> dict:
    """Everything the validate and configure pages show, read from the stored dataset.

    Both pages describe the same upload, so they load it the same way. The job comes
    first: it is what validates the token, and storage is never handed one that has
    not been validated.
    """
    job = db.get_job(token)
    dataset = db.load_payload(token, "dataset") if job is not None else None
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

    readiness = assess(dataset)
    return {
        "job": job, "token": token, "dataset": dataset,
        "summary": dataset.summary(), "previews": previews,
        "readiness": readiness,
        "dataset_status": ui.dataset_status(readiness),
        "capabilities": capabilities,
        "ranks": builder.ranks,
        "depths": list(dict.fromkeys(name for name, _ in depth_levels)),
        "depth_labels": ui.depth_labels(depth_levels),
        "dropped_depths": dropped_depths,
        "builder": builder,
    }


@router.get("/validate/{token}", response_class=HTMLResponse, include_in_schema=False)
def validate(request: Request, token: str):
    """How the files were read, before anything is configured or run. Read-only."""
    context = _setup(token)
    context.pop("builder")
    return templates.TemplateResponse(request, "validate.html", context)


@router.get("/configure/{token}", response_class=HTMLResponse, include_in_schema=False)
def configure(request: Request, token: str):
    context = _setup(token)
    builder = context.pop("builder")
    context["space"] = ui.specification_space(
        builder, context["capabilities"], context["dataset"].covariate_columns,
        context["previews"])
    return templates.TemplateResponse(request, "configure.html", context)


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
    jobs.dispatch(background, token, mode, declared, tuple(covariates))
    return RedirectResponse(f"/job/{token}", status_code=303)
