"""How an analysis gets started — the second seam between MicroVerse and its host.

`services.execute` is the analysis, and it does not change. What changes is who calls
it. On a server that keeps running after it answers a request, Starlette's background
threadpool calls it and the request returns immediately. On a host that freezes the
process the moment the response is sent, that call never happens, so the work has to
be handed to something that outlives the request.

Both backends preserve the lifecycle the frontend already polls:

    POST /run/{token}  ->  job row marked queued  ->  token returned  ->  worker runs
                                                                      ->  progress written
                                                                      ->  frontend polls

`inline` is the default and is exactly what MicroVerse has always done. `queue`
publishes the token to a Vercel Queues topic; delivery invokes the worker route,
which runs the same `services.execute`. The queue's own per-consumer concurrency cap
is what keeps the global limit global once more than one instance exists — an
in-process semaphore cannot do that job when the process is not the only one.

Nothing here decides *what* runs. It decides *where the call comes from*.
"""
from __future__ import annotations

from . import config

#: Topic that carries analysis requests. One topic, one consumer group, so instances
#: compete for work instead of each running every job.
TOPIC = "microverse-runs"
CONSUMER = "worker"

# One operational hazard worth stating plainly, because it is the failure this module
# is supposed to make impossible. Messages are partitioned by deployment: a run
# published by one deployment is delivered to that deployment's consumer. A run
# published in the seconds before a redeploy can therefore be left with no consumer
# and will sit until its retention expires. The window is small — a run is dispatched
# and picked up within seconds — and the alternative, sending across all deployments,
# means an old build can pick up a job intended for a new one, which for an analysis
# engine is the worse trade. Redeploy when nothing is queued if it matters.


def _payload(token: str, mode: str, declared, covariates) -> dict:
    """What the worker needs to reconstruct the call. Small, and never the dataset.

    The dataset already lives in storage under the same token, so the message carries
    a reference rather than megabytes of counts.
    """
    return {
        "token": token,
        "mode": mode,
        "declared": declared.as_row() if declared is not None else None,
        "covariates": list(covariates or ()),
    }


def dispatch(background, token: str, mode: str, declared=None, covariates=()) -> None:
    """Arrange for `services.execute(token, mode, declared, covariates)` to run.

    `background` is the request's BackgroundTasks, used only by the inline backend.
    """
    if config.JOB_BACKEND == "queue":
        _publish(_payload(token, mode, declared, covariates))
        return
    from . import services
    background.add_task(services.execute, token, mode, declared, tuple(covariates or ()))


def _publish(payload: dict) -> None:
    """Send one message, keyed by token so a double submit cannot run twice.

    The synchronous client is used on purpose: the routes that dispatch are ordinary
    request handlers and there is nothing to overlap this call with. Imported lazily,
    so a deployment with a disk and a background thread never loads the SDK.
    """
    from vercel.queue.sync import send

    send(
        TOPIC,
        payload,
        # The run must still be claimable if the first delivery is lost.
        retention=config.QUEUE_RETENTION_SECONDS,
        # Re-submitting the same token is a no-op rather than a second run.
        idempotency_key=payload["token"],
    )


def rebuild_declared(row):
    """Turn the message's declared-pipeline dict back into a Specification."""
    if not row:
        return None
    from .core.models import Specification
    fields = dict(row)
    covariates = fields.pop("covariates", "") or ""
    return Specification(
        **fields,
        covariates=tuple(c for c in str(covariates).split("|") if c),
    )
