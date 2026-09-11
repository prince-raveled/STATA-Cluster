"""MicroVerse — a web server for multiverse analysis of microbiome differential abundance.

SPEC §20: FastAPI, Jinja2 + HTMX + Alpine, SQLite for jobs only, no build step.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .core.parsers.base import ParseError
from .core.validation import DatasetError
from .db import utcnow
from .limits import queue_state
from .routers import api, job, results, upload
from .templating import templates

DESCRIPTION = """
Run every defensible analytical pipeline on your microbiome differential abundance
data, and report the distribution of answers instead of one p-value.

A differential abundance result depends on at least six contested analytical choices:
rarefaction depth, prevalence filter, transformation, taxonomic rank, DA method and
covariate adjustment. MicroVerse enumerates that space, executes it, and quantifies
how much of your result is the data and how much is the choices.

**There is no "export best specification" endpoint, and there never will be.**
"""

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    db.purge_expired()  # §16.5 retention, enforced on every boot
    yield


app = FastAPI(
    lifespan=lifespan,
    title="MicroVerse",
    version=config.VERSION,
    description=DESCRIPTION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    contact={"name": "MicroVerse", "url": config.REPOSITORY},
    license_info={"name": "MIT"},
)

app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "app" / "static")),
          name="static")

app.include_router(upload.router)
app.include_router(job.router)
app.include_router(results.router)
app.include_router(api.router)


@app.exception_handler(DatasetError)
async def dataset_error_handler(request: Request, exc: DatasetError):
    """SPEC §8: reject with a clear message, never a stack trace."""
    status = getattr(exc, "status_code", 422)
    headers = {}
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        headers["Retry-After"] = str(int(retry_after))
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=status, headers=headers,
                            content={"error": exc.message, "hint": exc.hint})
    title = {404: "Not found", 429: "Slow down"}.get(status, "That dataset cannot be run")
    return templates.TemplateResponse(
        request, "error.html",
        {"message": exc.message, "hint": exc.hint, "title": title},
        status_code=status, headers=headers,
    )


@app.exception_handler(ParseError)
async def parse_error_handler(request: Request, exc: ParseError):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=422, content={"error": str(exc), "hint": ""})
    return templates.TemplateResponse(
        request, "error.html",
        {"message": str(exc), "hint": "Check the file format and try again.",
         "title": "That file could not be read"},
        status_code=422,
    )


@app.get("/about", response_class=HTMLResponse, include_in_schema=False)
def about(request: Request):
    from .core.report import citation_list

    return templates.TemplateResponse(
        request, "about.html",
        {"citations": citation_list(), "retention_days": config.RETENTION_DAYS},
    )


@app.get("/validation", response_class=HTMLResponse, include_in_schema=False)
def validation(request: Request):
    """What has actually been validated, and how — SPEC §24.2, rendered.

    A separate page rather than a paragraph in the README: a researcher deciding
    whether to trust a tier should not have to find a repository to do it.
    """
    return templates.TemplateResponse(request, "validation.html", {})


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok", "version": config.VERSION,
            "time": utcnow().isoformat(), "queue": queue_state()}
