"""Where a run's time goes, and what the progress page says while it goes there.

A run reports the engine's own runtime, and on a hosted deployment the engine is the
short part: in production a Quick run on a demo took 81 s end to end against 3 s of
engine. The rest was invisible -- persistence to object storage ran under the message
"Attributing variance to the forks", and nothing recorded how long any phase took.
These pin the two things that make it visible: a progress step of its own for saving,
and per-phase timings stored with the result.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db, services, storage, ui
from app.main import app
from app.templating import duration


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
    assert set(timings) == {"starting", "analysis", "robustness", "attribution", "saving"}
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


# --- what the pages say it took ------------------------------------------------
def test_the_progress_page_reports_the_whole_run_not_the_engine(run):
    """It said "Finished in 3.6 s" after a two-minute wait: that was the fitting alone."""
    took = ui.run_time(run.job)
    assert took["total"] > run.job.runtime_seconds
    page = " ".join(TestClient(app).get(f"/job/{run.token}/progress").text.split())
    assert f"Finished in {duration(took['total'])}" in page
    assert f"Fitting them took {duration(run.job.runtime_seconds)}" in page


def test_the_results_page_reports_the_whole_run_too(run):
    page = TestClient(app).get(f"/results/{run.token}").text
    took = ui.run_time(run.job)
    assert f"ran in {duration(took['total'])} (fitting {duration(took['fitting'])})" in page
    assert "analysis time" not in page




def test_a_run_is_timed_from_the_click_including_the_wait(dataset):
    """On Vercel a run waits for the queue and a cold function; the researcher waits too."""
    token = services.create_job(dataset, "waited.tsv")
    try:
        services.request_run(token, "quick")
        clicked = db.utcnow() - dt.timedelta(seconds=30)
        db.update_job(token, started_at=clicked)          # as if the queue took 30 s
        services.execute(token, "quick")
        job = db.get_job(token)
        assert job.status == "done", job.error
        assert job.started_at == clicked, "the click time must survive the run"
        timings = job.as_dict()["summary"]["timings"]
        assert timings["starting"] >= 30
        took = ui.run_time(job)
        assert took["total"] >= 30 + timings["analysis"]
        assert took["others"][0]["key"] == "starting" or took["others"][0]["seconds"] >= 30
    finally:
        storage.purge(token)
        with db.session() as session:
            session.delete(session.get(db.Job, token))
            session.commit()


def test_a_run_started_without_a_click_is_not_timed_from_an_old_one(dataset):
    """A script re-running a finished job must not count the time since the last run."""
    token = services.create_job(dataset, "rerun.tsv")
    try:
        long_ago = db.utcnow() - dt.timedelta(days=2)
        db.update_job(token, started_at=long_ago, finished_at=long_ago + dt.timedelta(minutes=2))
        services.execute(token, "quick")
        job = db.get_job(token)
        assert job.status == "done", job.error
        assert job.as_dict()["summary"]["timings"]["starting"] < 60
        assert job.started_at > long_ago + dt.timedelta(days=1)
    finally:
        storage.purge(token)
        with db.session() as session:
            session.delete(session.get(db.Job, token))
            session.commit()


def _job(started=None, finished=None, runtime=3.6, timings=None):
    import json
    return SimpleNamespace(started_at=started, finished_at=finished, runtime_seconds=runtime,
                           summary_json=json.dumps({"timings": timings} if timings else {}))


def test_the_total_is_click_to_finish_and_the_breakdown_is_largest_first():
    start = dt.datetime(2026, 9, 25, 10, 0, 0)
    job = _job(start, start + dt.timedelta(seconds=144), timings={
        "starting": 24.0, "analysis": 3.3, "robustness": 0.2,
        "attribution": 86.2, "saving": 30.7})
    took = ui.run_time(job)
    assert took["total"] == 144
    assert took["fitting"] == 3.6
    assert [p["key"] for p in took["others"]] == ["attribution", "saving", "starting"]


def test_a_run_from_before_timings_were_recorded_still_reads_sensibly():
    assert ui.run_time(_job())["total"] == 3.6
    start = dt.datetime(2026, 9, 25, 10, 0, 0)
    assert ui.run_time(_job(start, start + dt.timedelta(seconds=90)))["total"] == 90
    job = _job(timings={"analysis": 3.0, "attribution": 50.0, "saving": 20.0})
    assert ui.run_time(job)["total"] == 73.0
