"""Job progress page and its HTMX polling fragment."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import config, db
from ..core.validation import NotFoundError
from ..templating import templates

router = APIRouter()


@router.get("/job/{token}", response_class=HTMLResponse, include_in_schema=False)
def job_page(request: Request, token: str):
    job = db.get_job(token)
    if job is None:
        raise NotFoundError("That job no longer exists.",
                           f"Results are kept for {config.RETENTION_DAYS} days.")
    if job.status == "done":
        return RedirectResponse(f"/results/{token}", status_code=303)
    return templates.TemplateResponse(request, "job.html", {"job": job, "token": token})


@router.get("/job/{token}/progress", response_class=HTMLResponse, include_in_schema=False)
def job_progress(request: Request, token: str):
    """HTMX polls this every second; it swaps itself out and stops when the run ends."""
    job = db.get_job(token)
    if job is None:
        return HTMLResponse('<p class="error">This job has expired.</p>')
    return templates.TemplateResponse(
        request, "_progress.html", {"job": job, "token": token}
    )
