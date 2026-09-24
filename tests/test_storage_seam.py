"""The one place a token or a name becomes a path or a key.

Tokens come from URLs. Before this seam checked them, `storage.get_bytes('..',
'jobs.sqlite')` returned the job database, and a read of a token that was never issued
created its directory. Neither was reachable over HTTP -- every route validated first
-- but a seam that trusts every caller to have validated is one refactor from a
traversal, so it now refuses on its own.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config, db, storage
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_backend():
    storage.reset()
    yield
    storage.reset()


@pytest.mark.parametrize("token", ["..", ".", "", "not-a-token", "A" * 24, "0" * 23,
                                   "../" + "0" * 21, "0" * 24 + "/.."])
def test_storage_refuses_a_token_that_could_leave_its_directory(token):
    """`get_bytes('..', 'jobs.sqlite')` used to return the job database."""
    with pytest.raises(ValueError):
        storage.get_bytes(token, "jobs.sqlite")
    with pytest.raises(ValueError):
        storage.put_bytes(token, "x", b"")


@pytest.mark.parametrize("name", ["../jobs.sqlite", "..", ".hidden", "a/b", "", "a\\b"])
def test_storage_refuses_a_name_that_is_not_one_segment(name):
    with pytest.raises(ValueError):
        storage.get_bytes("0" * 24, name)


def test_the_blob_backend_refuses_the_same(monkeypatch):
    monkeypatch.setattr(config, "STORAGE_BACKEND", "blob")
    storage.reset()
    with pytest.raises(ValueError):
        storage.backend()._key("..", "bundle.zip")


def test_reading_an_unknown_token_creates_nothing_on_disk(client):
    """A read used to create the job's directory, so every guess left one behind."""
    token = db.new_token()
    for path in (f"/download/{token}/bundle", f"/download/{token}/long",
                 f"/download/{token}/manifest", f"/configure/{token}",
                 f"/results/{token}"):
        assert client.get(path).status_code == 404, path
    assert storage.get_bytes(token, "dataset.pkl.gz") is None
    assert not (config.JOBS_DIR / token).exists()


@pytest.mark.parametrize("token", ["not-a-token", "A" * 24, "0" * 23])
def test_configure_checks_the_token_before_storage_sees_it(client, token):
    """It read the dataset before looking the job up; a bad token must stay a 404."""
    assert client.get(f"/configure/{token}").status_code == 404
