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


#: htmx swaps a 286 response like any success and then cancels the polling trigger.
STOP_POLLING = 286


@router.get("/job/{token}/progress", response_class=HTMLResponse, include_in_schema=False)
def job_progress(request: Request, token: str):
    """HTMX polls this every second while the run is going, and not after it ends.

    It used to answer 200 forever: a failed run's page asked for its progress every
    second for as long as it stayed open, and a finished one kept polling while the
    browser navigated to the results, so a late swap landed in a page being torn down
    (htmx: "Cannot read properties of null (reading 'insertBefore')"). Once the run has
    an outcome the answer is final, so it says so with htmx's stop-polling status.
    """
    job = db.get_job(token)
    if job is None:
        return HTMLResponse('<p class="error">This job has expired.</p>',
                            status_code=STOP_POLLING)
    final = job.status in ("done", "error")
    return templates.TemplateResponse(
        request, "_progress.html", {"job": job, "token": token},
        status_code=STOP_POLLING if final else 200,
    )
