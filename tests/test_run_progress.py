"""Where a run's time goes, and what the progress page says while it goes there.

A run reports the engine's own runtime, and on a hosted deployment the engine is the
short part: in production a Quick run on a demo took 81 s end to end against 3 s of
engine. The rest was invisible -- persistence to object storage ran under the message
"Attributing variance to the forks", and nothing recorded how long any phase took.
These pin the two things that make it visible: a progress step of its own for saving,
and per-phase timings stored with the result.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import db, services, storage


@pytest.fixture(scope="module")
def run(dataset):
    """One real Quick run, with every progress message it reported."""
    storage.reset()
    messages = []
    real = db.update_job

    def spy(token, **fields):
        if "message" in fields:
            messages.append(fields["message"])
        return real(token, **fields)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(db, "update_job", spy)
        token = services.create_job(dataset, "timed.tsv")
        services.execute(token, "quick")
    job = db.get_job(token)
    assert job.status == "done", job.error
    yield SimpleNamespace(token=token, messages=messages, job=job)
    storage.purge(token)
    with db.session() as session:
        session.delete(session.get(db.Job, token))
        session.commit()


def test_a_run_records_where_its_time_went(run):
    timings = run.job.as_dict()["summary"]["timings"]
    assert set(timings) == {"analysis", "robustness", "attribution", "saving"}
    assert all(isinstance(value, float) and value >= 0 for value in timings.values())


def test_the_engine_runtime_is_still_the_engine_alone(run):
    """runtime_seconds keeps its meaning; the timings sit beside it, not in it."""
    timings = run.job.as_dict()["summary"]["timings"]
    assert run.job.runtime_seconds <= timings["analysis"] + 0.5


def test_saving_is_its_own_step_in_the_progress(run):
    """Persistence used to run under "Attributing variance to the forks"."""
    messages = run.messages
    assert messages.index("Attributing variance to the forks") \
        < messages.index("Saving results") < messages.index("Complete")
