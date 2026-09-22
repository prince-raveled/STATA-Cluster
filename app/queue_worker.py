"""The consumer. Vercel compiles this module into a queue-triggered function.

`jobs.dispatch` publishes a token; something has to receive it. On Vercel that
something is generated at build time from the subscription registered below, which is
why `pyproject.toml` names this module under `[[tool.vercel.subscribers]]`. Without
that entry the messages would be published successfully and never delivered — which
is exactly the failure this module exists to prevent, and why the test suite asserts
the entry is present rather than trusting it.

The handler does one thing: call `services.execute` with the arguments the publisher
put in the message. It is the same function the inline backend hands to Starlette's
background threadpool, so the analysis is the same analysis.

Concurrency lives here too. `max_concurrency` is the push dispatcher's cap, which is
global across every instance — the thing an in-process semaphore cannot be once more
than one instance exists. `services.execute` therefore does not take the local
semaphore in queue mode; the queue is the limit.

This module imports the SDK at module scope, deliberately: it is only ever imported
by the generated consumer and by tests, never by `app.main`, so a deployment with a
disk and a background thread never loads it.
"""
from __future__ import annotations

from vercel.queue import Message, subscribe

from . import db, jobs, limits, services

#: Matches the topic `jobs._publish` sends to. One consumer group, so instances
#: compete for work rather than each receiving a copy.
TOPIC = jobs.TOPIC
CONSUMER_GROUP = jobs.CONSUMER


@subscribe(topic=TOPIC, consumer_group=CONSUMER_GROUP,
           max_concurrency=limits.MAX_CONCURRENT_RUNS)
async def run_analysis(message: Message[dict[str, object]]) -> None:
    """Execute one queued run.

    Raising is how a delivery is retried, so the two cases that will never succeed —
    a job whose row is gone, and one that has already finished — return quietly
    instead. Delivery is at-least-once, so the second is not hypothetical: a lease
    that expires while the analysis is still running produces a redelivery of work
    that is about to be written, and re-running it would overwrite a result someone
    may already be reading.
    """
    payload = message.payload or {}
    token = str(payload.get("token", ""))
    if not db.valid_token(token):
        return

    job = db.get_job(token)
    if job is None or job.status == "done":
        return

    services.execute(
        token,
        str(payload.get("mode") or job.mode or "quick"),
        jobs.rebuild_declared(payload.get("declared")),
        tuple(payload.get("covariates") or ()),
    )
