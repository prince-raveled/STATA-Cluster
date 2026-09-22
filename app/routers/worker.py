"""The route that runs an analysis when something other than a request asks for it.

On a host that freezes the process once a response is sent, the work cannot happen in
the request that started it. `jobs.dispatch` publishes the token instead, and delivery
of that message calls this route, which does exactly what the inline backend does:
`services.execute`, same arguments, same engine.

This route is not part of the public API. It is not in the OpenAPI schema, it answers
only to a caller holding the shared secret, and it exists solely so that the queue has
something to invoke. A deployment that has not configured a secret refuses it outright
rather than leaving an unauthenticated way to start work on the server.
"""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from .. import config, db, jobs, services

router = APIRouter(prefix="/internal", include_in_schema=False)


def _authorised(presented: str | None) -> bool:
    """Constant-time comparison, and no secret configured means no access at all."""
    if not config.WORKER_SECRET:
        return False
    return hmac.compare_digest(str(presented or ""), config.WORKER_SECRET)


@router.post("/run")
async def run(request: Request, x_microverse_worker: str = Header(default="")):
    if not _authorised(x_microverse_worker):
        # Deliberately uninformative: this endpoint's existence is not a secret, but
        # whether a guess was close is not something a caller needs to learn.
        return JSONResponse({"detail": "Not found"}, status_code=404)

    try:
        message = await request.json()
    except Exception:                                        # noqa: BLE001
        return JSONResponse({"detail": "Malformed message"}, status_code=400)

    token = str(message.get("token", ""))
    if not db.valid_token(token):
        return JSONResponse({"detail": "Malformed token"}, status_code=400)

    job = db.get_job(token)
    if job is None:
        # The message outlived its job — retention expired, or the row was purged.
        # Acknowledge it: redelivering will never succeed.
        return JSONResponse({"token": token, "status": "gone"}, status_code=200)
    if job.status == "done":
        # At-least-once delivery means this can legitimately arrive twice, and
        # re-running a finished analysis would overwrite a result the user may
        # already be reading.
        return JSONResponse({"token": token, "status": "done"}, status_code=200)

    services.execute(
        token,
        str(message.get("mode") or job.mode or "quick"),
        jobs.rebuild_declared(message.get("declared")),
        tuple(message.get("covariates") or ()),
    )
    finished = db.get_job(token)
    return {"token": token, "status": finished.status if finished else "unknown"}
