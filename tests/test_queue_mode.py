"""Queue mode: a published run has to actually reach the engine.

The failure this file exists to prevent is specific and silent. `send` succeeding
proves a message was stored, not that anything will ever consume it — and a
deployment where every run publishes cleanly and never executes looks healthy from
the server's side while the user watches a progress bar forever.

So the round trips here are real. `vercel.queue`'s embedded service runs an actual
queue in-process, and the tests publish through the same `jobs._publish` the routes
call, poll the message back out, and hand it to the same subscriber Vercel compiles
into its consumer. What cannot be exercised locally is Vercel's own build step
reading `[[tool.vercel.subscribers]]`, so the declaration it depends on is asserted
directly instead of assumed.
"""
from __future__ import annotations

import asyncio
import tomllib

import pytest

from app import config, db, jobs, limits, queue_worker, services

TOKEN = "0123456789abcdef01234567"
ROOT = config.BASE_DIR


@pytest.fixture
def queue_server(monkeypatch):
    """A real queue, in this process, that the default client publishes to."""
    from vercel.queue.devserver import embedded_queue_dev_server
    from vercel.queue.testing import reset_default_queue_clients

    with embedded_queue_dev_server() as server:
        monkeypatch.setenv("VERCEL_QUEUE_BASE_URL", server.base_url)
        monkeypatch.setenv("VERCEL_QUEUE_TOKEN", "local-token")
        monkeypatch.setenv("VERCEL_REGION", "iad1")
        # Messages are partitioned by deployment unless told otherwise, and the SDK
        # refuses to send without one. Vercel sets this; a test has to supply it.
        monkeypatch.setenv("VERCEL_DEPLOYMENT_ID", "dpl_test")
        reset_default_queue_clients()
        try:
            yield server
        finally:
            reset_default_queue_clients()


def drain(server, limit=10):
    """Take whatever is waiting on the topic, as the consumer group would."""
    client = server.get_sync_client()
    return list(client.poll(jobs.TOPIC, jobs.CONSUMER, limit=limit))


# --- publishing -------------------------------------------------------------
def test_a_published_run_is_really_on_the_topic(queue_server):
    """The whole point: `send` succeeding must mean a consumer can find it."""
    jobs._publish(jobs._payload(TOKEN, "full", None, ("age",)))

    deliveries = drain(queue_server)
    assert len(deliveries) == 1, "the message was published but nothing could receive it"

    payload = deliveries[0].accept().payload
    assert payload["token"] == TOKEN
    assert payload["mode"] == "full"
    assert payload["covariates"] == ["age"]
    assert "dataset" not in payload, "the dataset stays in storage, not in the message"


def test_the_same_token_published_twice_runs_once(queue_server):
    """Idempotency key is the token, so a double submit is not a second analysis."""
    payload = jobs._payload(TOKEN, "quick", None, ())
    jobs._publish(payload)
    jobs._publish(payload)

    assert len(drain(queue_server)) == 1


def test_two_different_runs_both_arrive(queue_server):
    """The deduplication must be per token, not a blanket one-message rule."""
    jobs._publish(jobs._payload(TOKEN, "quick", None, ()))
    jobs._publish(jobs._payload("f" * 24, "quick", None, ()))

    assert len(drain(queue_server)) == 2


def _run_again(finished_at):
    """What POST /run does to a job that has already run once, then dispatch."""
    db.update_job(TOKEN, status="running", finished_at=finished_at)
    jobs.dispatch(None, TOKEN, "quick", None, ())


def test_running_a_finished_job_again_is_delivered(queue_server, monkeypatch):
    """Found in production: the second run of a job was silently dropped.

    Vercel Queues discards a repeated idempotency key for as long as the original
    message is retained, and `send` still succeeds. With the bare token as the key, a
    finished job run a second time -- another mode, or a retry after an error -- was
    never delivered, stayed on "Queued", and its results page redirected there.
    """
    import datetime as dt

    monkeypatch.setattr(config, "JOB_BACKEND", "queue")
    with db.session() as session:
        session.add(db.Job(token=TOKEN, status="running", mode="quick"))
        session.commit()
    try:
        jobs.dispatch(None, TOKEN, "quick", None, ())
        drain(queue_server)[0].accept()                     # the first run, consumed

        _run_again(dt.datetime(2026, 9, 24, 8, 0, 0))
        assert len(drain(queue_server)) == 1, "the second run was dropped as a duplicate"
    finally:
        _drop(TOKEN)


def test_a_double_submit_of_a_second_run_is_still_one_message(queue_server, monkeypatch):
    import datetime as dt

    monkeypatch.setattr(config, "JOB_BACKEND", "queue")
    with db.session() as session:
        session.add(db.Job(token=TOKEN, status="done", mode="quick",
                           finished_at=dt.datetime(2026, 9, 24, 8, 0, 0)))
        session.commit()
    try:
        _run_again(dt.datetime(2026, 9, 24, 8, 0, 0))
        _run_again(dt.datetime(2026, 9, 24, 8, 0, 0))
        assert len(drain(queue_server)) == 1
    finally:
        _drop(TOKEN)


def test_a_first_run_keeps_the_bare_token_as_its_key():
    """Unchanged for every job that has never run: the key is still the token."""
    assert jobs.run_key("0" * 24) == "0" * 24


def test_dispatch_publishes_in_queue_mode_and_never_schedules_locally(monkeypatch):
    sent = {}
    monkeypatch.setattr(config, "JOB_BACKEND", "queue")
    monkeypatch.setattr(jobs, "_publish", lambda p, key=None: sent.update(p))

    class Background:
        def __init__(self):
            self.tasks = []

        def add_task(self, fn, *args):
            self.tasks.append((fn, args))

    background = Background()
    jobs.dispatch(background, TOKEN, "quick", None, ())

    assert sent["token"] == TOKEN
    assert background.tasks == []


def test_an_unrecognised_backend_behaves_like_inline(monkeypatch):
    """A typo in the environment must not silently stop running anything."""
    published = []
    monkeypatch.setattr(config, "JOB_BACKEND", "qeueu")
    monkeypatch.setattr(jobs, "_publish", lambda p, key=None: published.append(p))

    class Background:
        def __init__(self):
            self.tasks = []

        def add_task(self, fn, *args):
            self.tasks.append((fn, args))

    background = Background()
    jobs.dispatch(background, TOKEN, "quick", None, ())

    assert published == []
    assert len(background.tasks) == 1
    assert background.tasks[0][0] is services.execute


# --- the consumer -----------------------------------------------------------
def test_the_subscriber_is_registered_for_the_topic_that_is_published_to():
    """A subscriber on the wrong topic is the same outage as no subscriber."""
    from vercel.queue import get_subscriptions

    mine = [s for s in get_subscriptions() if s.topic == jobs.TOPIC]
    assert mine, f"nothing subscribes to {jobs.TOPIC!r}"
    assert mine[0].consumer_group == jobs.CONSUMER


def test_the_subscriber_carries_a_global_concurrency_cap():
    """This is what replaces the in-process semaphore once instances multiply."""
    from vercel.queue import get_subscriptions

    subscription = next(s for s in get_subscriptions() if s.topic == jobs.TOPIC)
    assert subscription.max_concurrency == limits.MAX_CONCURRENT_RUNS
    assert subscription.max_concurrency >= 1


def test_the_subscriber_module_is_declared_to_the_host():
    """Vercel builds the consumer from this entry. Without it, nothing consumes.

    Not testable by running the app — it is read by the host's build — so the
    declaration itself is the assertion.
    """
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = manifest.get("tool", {}).get("vercel", {}).get("subscribers", [])
    assert entries, "pyproject.toml declares no queue subscribers"
    assert any(e.get("entrypoint") == "app.queue_worker" for e in entries), entries


def test_a_delivered_message_runs_the_analysis(queue_server, monkeypatch):
    """Publish, receive, and confirm the engine is called with what was sent."""
    calls = []
    monkeypatch.setattr(services, "execute", lambda *a: calls.append(a))
    with db.session() as session:
        session.add(db.Job(token=TOKEN, status="running", mode="quick"))
        session.commit()
    try:
        jobs._publish(jobs._payload(TOKEN, "full", None, ("age", "bmi")))
        message = drain(queue_server)[0].accept()
        asyncio.run(queue_worker.run_analysis(message))

        assert calls == [(TOKEN, "full", None, ("age", "bmi"))]
    finally:
        _drop(TOKEN)


def test_a_failing_analysis_propagates_so_the_queue_redelivers(queue_server, monkeypatch):
    """Swallowing the error would silently drop the run instead of retrying it."""
    def boom(*_args):
        raise RuntimeError("the engine fell over")

    monkeypatch.setattr(services, "execute", boom)
    with db.session() as session:
        session.add(db.Job(token=TOKEN, status="running", mode="quick"))
        session.commit()
    try:
        jobs._publish(jobs._payload(TOKEN, "quick", None, ()))
        message = drain(queue_server)[0].accept()

        with pytest.raises(RuntimeError, match="fell over"):
            asyncio.run(queue_worker.run_analysis(message))
    finally:
        _drop(TOKEN)


def test_a_redelivered_message_for_a_finished_job_does_not_rerun_it(queue_server,
                                                                    monkeypatch):
    """At-least-once delivery is not hypothetical: a lease can expire mid-run."""
    calls = []
    monkeypatch.setattr(services, "execute", lambda *a: calls.append(a))
    with db.session() as session:
        session.add(db.Job(token=TOKEN, status="done", mode="quick"))
        session.commit()
    try:
        jobs._publish(jobs._payload(TOKEN, "quick", None, ()))
        message = drain(queue_server)[0].accept()
        asyncio.run(queue_worker.run_analysis(message))

        assert calls == [], "a finished result would have been overwritten"
    finally:
        _drop(TOKEN)


def test_a_message_for_a_vanished_job_is_accepted_quietly(queue_server, monkeypatch):
    """Retention expired. Retrying forever would be worse than dropping it."""
    calls = []
    monkeypatch.setattr(services, "execute", lambda *a: calls.append(a))
    jobs._publish(jobs._payload(TOKEN, "quick", None, ()))
    message = drain(queue_server)[0].accept()

    asyncio.run(queue_worker.run_analysis(message))       # must not raise
    assert calls == []


def test_a_malformed_token_in_a_message_is_refused(queue_server, monkeypatch):
    calls = []
    monkeypatch.setattr(services, "execute", lambda *a: calls.append(a))
    jobs._publish({"token": "../../etc/passwd", "mode": "quick",
                   "declared": None, "covariates": []})
    message = drain(queue_server)[0].accept()

    asyncio.run(queue_worker.run_analysis(message))
    assert calls == []


# --- who bounds concurrency -------------------------------------------------
def _drop(token):
    with db.session() as session:
        job = session.get(db.Job, token)
        if job is not None:
            session.delete(job)
            session.commit()


class RecordingQueue:
    """Stands in for the process-wide semaphore so the calls can be counted."""

    def __init__(self):
        self.acquired = 0
        self.released = 0

    def acquire(self, timeout=None):                          # noqa: ARG002
        self.acquired += 1
        return True

    def release(self):
        self.released += 1


def test_queue_mode_does_not_take_the_local_semaphore(monkeypatch):
    """The dispatcher already decided this run may proceed, and its cap is global."""
    recorder = RecordingQueue()
    monkeypatch.setattr(config, "JOB_BACKEND", "queue")
    monkeypatch.setattr(services, "run_queue", recorder)

    services.execute(TOKEN, "quick")            # no job row: fails fast, still runs both ends

    assert recorder.acquired == 0
    assert recorder.released == 0


def test_inline_mode_still_takes_and_returns_the_local_semaphore(monkeypatch):
    """The original behaviour, unchanged, on the backend that still needs it."""
    recorder = RecordingQueue()
    monkeypatch.setattr(config, "JOB_BACKEND", "inline")
    monkeypatch.setattr(services, "run_queue", recorder)

    services.execute(TOKEN, "quick")

    assert recorder.acquired == 1
    assert recorder.released == 1, "a slot taken and not returned leaks capacity"


def test_inline_mode_reports_a_full_queue_instead_of_running(monkeypatch):
    """The busy path must still be reachable, and must still be an error the user sees."""
    class Full(RecordingQueue):
        def acquire(self, timeout=None):                      # noqa: ARG002
            self.acquired += 1
            return False

    recorder = Full()
    monkeypatch.setattr(config, "JOB_BACKEND", "inline")
    monkeypatch.setattr(services, "run_queue", recorder)
    with db.session() as session:
        session.add(db.Job(token=TOKEN, status="running", mode="quick"))
        session.commit()
    try:
        services.execute(TOKEN, "quick")
        job = db.get_job(TOKEN)
        assert job.status == "error"
        assert "busy" in job.error
        assert recorder.released == 0, "nothing was acquired, so nothing may be released"
    finally:
        _drop(TOKEN)
