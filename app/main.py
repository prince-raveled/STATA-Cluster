"""MicroVerse — a web server for multiverse analysis of microbiome differential abundance.

SPEC §20: FastAPI, Jinja2 + HTMX + Alpine, SQLite for jobs only, no build step.
"""
from __future__ import annotations

import platform
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
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


#: Routes that answer in JSON even though they are not mounted under /api. The results
#: page fetches these with `fetch`, so an HTML error page would be unreadable to the
#: caller. Kept as a predicate rather than a list so a new curve-like route is covered.
def _wants_json(path: str) -> bool:
    # "/upload/" with the slash, so the plain form post to "/upload" still gets the
    # error page a browser navigation should get. The direct-upload steps behind it
    # are all called with `fetch`.
    return path.startswith(("/api/", "/upload/")) or "/curve/" in path


@app.exception_handler(DatasetError)
async def dataset_error_handler(request: Request, exc: DatasetError):
    """SPEC §8: reject with a clear message, never a stack trace."""
    status = getattr(exc, "status_code", 422)
    headers = {}
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        headers["Retry-After"] = str(int(retry_after))
    if _wants_json(request.url.path):
        return JSONResponse(status_code=status, headers=headers,
                            content={"error": exc.message, "hint": exc.hint})
    title = {404: "Not found", 429: "Slow down"}.get(status, "That dataset cannot be run")
    return templates.TemplateResponse(
        request, "error.html",
        {"message": exc.message, "hint": exc.hint, "title": title},
        status_code=status, headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """One error shape across the whole surface.

    FastAPI's default is `{"detail": [...]}`, which is a second vocabulary on an API
    whose own errors are `{"error": ..., "hint": ...}`. A client should not have to
    branch on which layer rejected it.
    """
    first = (exc.errors() or [{}])[0]
    location = ".".join(str(part) for part in first.get("loc", ())[1:]) or "request"
    message = first.get("msg", "That request could not be understood.")
    if not _wants_json(request.url.path):
        return templates.TemplateResponse(
            request, "error.html",
            {"message": f"{location}: {message}", "hint": "Check the address and try again.",
             "title": "That request could not be understood"},
            status_code=422,
        )
    return JSONResponse(status_code=422, content={
        "error": f"{location}: {message}",
        "hint": "Check the request against /api/docs.",
    })


@app.exception_handler(ParseError)
async def parse_error_handler(request: Request, exc: ParseError):
    if _wants_json(request.url.path):
        return JSONResponse(status_code=422, content={"error": str(exc), "hint": ""})
    return templates.TemplateResponse(
        request, "error.html",
        {"message": str(exc), "hint": "Check the file format and try again.",
         "title": "That file could not be read"},
        status_code=422,
    )


@app.get("/about", response_class=HTMLResponse, include_in_schema=False)
def about(request: Request):
    from .core.glossary import glossary_sections
    from .core.report import citation_list

    # The results page sends beginners to /about#glossary. The anchor has to exist.
    return templates.TemplateResponse(
        request, "about.html",
        {"citations": citation_list(), "retention_days": config.RETENTION_DAYS,
         "glossary": glossary_sections()},
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
    """Liveness, plus which backends this process actually chose.

    A deployment configured for one thing and running as another looks healthy until
    the first write, then fails deep inside a request with an error about a read-only
    filesystem — which says nothing about why the wrong backend was picked. These are
    names and a scheme: no token, no credential, no connection string.
    """
    from . import storage

    return {"status": "ok", "version": config.VERSION,
            "time": utcnow().isoformat(), "queue": queue_state(),
            "config": {
                "storage": storage.backend().name,
                "jobs": config.JOB_BACKEND,
                "database": config.DATABASE_URL.split("://", 1)[0],
                "on_vercel": config.ON_VERCEL,
                # The host picks the interpreter from more than one file, and says
                # which it chose only in a build log; this is the one that is running.
                "python": platform.python_version(),
            }}
