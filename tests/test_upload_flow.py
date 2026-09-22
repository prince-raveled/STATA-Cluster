"""Uploading a file without sending it through the application.

The three-step flow exists so a host whose request-body cap is smaller than
MicroVerse's upload limit can still accept a real abundance table. What it must not do
is become a second, laxer way in: the same size limit, the same accepted formats and
the same rejection messages apply, and the only thing the browser is trusted with is
a signed statement about files the server already agreed to.

The local backend implements the same protocol as object storage, so everything below
exercises the real route rather than a description of it.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app import config, limits, storage, uploads
from app.main import app

EXAMPLES = config.EXAMPLES_DIR
ABUNDANCE = (EXAMPLES / "ibd_genus_abundance.tsv").read_bytes()
METADATA = (EXAMPLES / "ibd_genus_metadata.tsv").read_bytes()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _local_backend():
    storage.reset()
    yield
    storage.reset()


@pytest.fixture(autouse=True)
def _fresh_rate_limit():
    """Isolate tests from each other's budget, without changing the budget.

    Twenty uploads per five minutes is the real limit and stays the real limit; a file
    of upload tests would otherwise spend it on itself and start measuring the limiter
    instead of the flow. `test_limits.py` is where the limit itself is pinned.
    """
    limits.rate_limiter._hits.clear()
    yield
    limits.rate_limiter._hits.clear()


def declare(taxonomy=False, drop=(), replace=None):
    """The JSON a browser sends to /upload/authorize, with the demo table's real sizes."""
    files = {
        "abundance": {"filename": "counts.tsv", "size": len(ABUNDANCE)},
        "metadata": {"filename": "meta.tsv", "size": len(METADATA)},
    }
    if taxonomy:
        files["taxonomy"] = {"filename": "tax.tsv", "size": 100}
    for field in drop:
        files.pop(field, None)
    if replace:
        files.update(replace)
    return {"files": files}


def walk_through(client, abundance=ABUNDANCE, metadata=METADATA, group_column=""):
    """The whole flow, as the browser performs it. Returns the final response."""
    granted = client.post("/upload/authorize", json=declare())
    assert granted.status_code == 200, granted.text
    body = granted.json()
    ticket = body["ticket"]

    for field, payload in (("abundance", abundance), ("metadata", metadata)):
        target = body["uploads"][field]
        put = client.request(
            target["method"], target["url"], content=payload,
            headers={**target.get("headers", {}), "X-Microverse-Ticket": ticket},
        )
        assert put.status_code == 200, put.text

    return client.post("/upload/complete",
                       json={"ticket": ticket, "group_column": group_column})


# --- the happy path ---------------------------------------------------------
def test_a_real_dataset_arrives_as_a_job(client):
    """The point of the whole exercise: a 110 KB table, never through the function."""
    response = walk_through(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["next"] == f"/configure/{body['token']}"

    page = client.get(body["next"])
    assert page.status_code == 200
    assert "configure" in page.text.lower()


def test_the_staged_copies_are_deleted_once_they_are_a_dataset(client):
    """They cost quota and nothing reads them again."""
    granted = client.post("/upload/authorize", json=declare()).json()
    ticket = granted["ticket"]
    staging = uploads.staging_token(uploads.verify(ticket)["id"])

    for field, payload in (("abundance", ABUNDANCE), ("metadata", METADATA)):
        target = granted["uploads"][field]
        client.request(target["method"], target["url"], content=payload,
                       headers={"X-Microverse-Ticket": ticket})
    assert storage.get_bytes(staging, "abundance") is not None

    client.post("/upload/complete", json={"ticket": ticket})
    assert storage.get_bytes(staging, "abundance") is None
    assert storage.get_bytes(staging, "metadata") is None


def test_an_abandoned_upload_can_be_swept(client):
    granted = client.post("/upload/authorize", json=declare()).json()
    ticket = granted["ticket"]
    staging = uploads.staging_token(uploads.verify(ticket)["id"])
    target = granted["uploads"]["abundance"]
    client.request(target["method"], target["url"], content=ABUNDANCE,
                   headers={"X-Microverse-Ticket": ticket})
    assert storage.get_bytes(staging, "abundance") is not None

    assert client.post("/upload/abandon", json={"ticket": ticket}).status_code == 200
    assert storage.get_bytes(staging, "abundance") is None


def test_abandoning_a_nonsense_ticket_is_still_fine(client):
    """Called from pagehide, where nothing can act on a failure anyway."""
    assert client.post("/upload/abandon", json={"ticket": "rubbish"}).status_code == 200
    assert client.post("/upload/abandon", json={}).status_code == 200


# --- the limits are the application's, not the host's -----------------------
def test_the_existing_64_mb_limit_still_applies(client):
    """Declared before a byte is written, so an oversized file is never staged."""
    over = config.MAX_UPLOAD_BYTES + 1
    response = client.post("/upload/authorize", json=declare(
        replace={"abundance": {"filename": "huge.tsv", "size": over}}))
    assert response.status_code == 422
    assert "upload limit" in response.json()["error"]


def test_a_declared_size_is_not_taken_on_trust(client):
    """A small declaration followed by a large body must still be refused."""
    granted = client.post("/upload/authorize", json=declare()).json()
    ticket = granted["ticket"]
    target = granted["uploads"]["abundance"]

    response = client.request(
        target["method"], target["url"],
        content=b"x" * (config.MAX_UPLOAD_BYTES + 1),
        headers={"X-Microverse-Ticket": ticket})
    assert response.status_code == 422
    assert "upload limit" in response.json()["error"]


@pytest.mark.parametrize("filename", ["notes.pdf", "archive.rar", "script.exe", "x"])
def test_an_unsupported_extension_is_refused_before_anything_is_staged(client, filename):
    response = client.post("/upload/authorize", json=declare(
        replace={"abundance": {"filename": filename, "size": 10}}))
    assert response.status_code == 422
    assert "not a file type" in response.json()["error"]


@pytest.mark.parametrize("missing", ["abundance", "metadata"])
def test_a_missing_required_file_is_refused(client, missing):
    response = client.post("/upload/authorize", json=declare(drop=[missing]))
    assert response.status_code == 422
    assert "was uploaded" in response.json()["error"]


def test_a_malformed_file_fails_exactly_as_it_does_through_the_form(client):
    """Same parsers, same validation, so the same words come back."""
    rubbish = b"this is not a table at all\n"
    direct = walk_through(client, abundance=rubbish)
    assert direct.status_code == 422

    form = client.post(
        "/upload",
        files={"abundance": ("counts.tsv", rubbish, "text/tab-separated-values"),
               "metadata": ("meta.tsv", METADATA, "text/tab-separated-values")},
        data={"group_column": ""},
    )
    assert form.status_code == 422
    assert direct.json()["error"] in form.text


# --- the ticket is the only thing the browser is trusted with ---------------
def test_a_ticket_round_trips():
    files = {"abundance": {"filename": "a.tsv", "size": 10}}
    payload = uploads.verify(uploads.issue(files))
    assert payload is not None
    assert payload["files"] == files
    assert len(payload["id"]) == 24


@pytest.mark.parametrize("forged", [
    "", "rubbish", "a.b", "....", "eyJhIjoxfQ.deadbeef",
])
def test_a_forged_ticket_is_rejected(forged):
    assert uploads.verify(forged) is None


def test_a_tampered_ticket_is_rejected():
    """Changing the declared size must not survive the signature."""
    ticket = uploads.issue({"abundance": {"filename": "a.tsv", "size": 10}})
    body, signature = ticket.split(".", 1)
    assert uploads.verify(body[:-2] + "AA" + "." + signature) is None
    assert uploads.verify(body + "." + signature[:-2] + "AA") is None


def test_an_expired_ticket_is_rejected(monkeypatch):
    ticket = uploads.issue({"abundance": {"filename": "a.tsv", "size": 10}})
    assert uploads.verify(ticket) is not None

    later = time.time() + uploads.TICKET_TTL_SECONDS + 5
    monkeypatch.setattr(uploads.time, "time", lambda: later)
    assert uploads.verify(ticket) is None


def test_completing_with_a_forged_ticket_is_refused(client):
    response = client.post("/upload/complete", json={"ticket": "forged.ticket"})
    assert response.status_code == 422
    assert "expired" in response.json()["error"]


def test_staging_needs_the_matching_ticket(client):
    """A valid ticket for one upload must not write into another's staging area."""
    mine = client.post("/upload/authorize", json=declare()).json()
    theirs = client.post("/upload/authorize", json=declare()).json()
    their_token = uploads.staging_token(uploads.verify(theirs["ticket"])["id"])

    response = client.put(f"/upload/staged/{their_token}/abundance",
                          content=b"data",
                          headers={"X-Microverse-Ticket": mine["ticket"]})
    assert response.status_code == 404


def test_staging_refuses_a_field_the_ticket_does_not_name(client):
    """The ticket named no taxonomy file, so there is no permission to write one."""
    granted = client.post("/upload/authorize", json=declare()).json()
    token = uploads.staging_token(uploads.verify(granted["ticket"])["id"])
    response = client.put(f"/upload/staged/{token}/taxonomy", content=b"data",
                          headers={"X-Microverse-Ticket": granted["ticket"]})
    assert response.status_code == 404


def test_staging_refuses_a_missing_ticket(client):
    granted = client.post("/upload/authorize", json=declare()).json()
    token = uploads.staging_token(uploads.verify(granted["ticket"])["id"])
    assert client.put(f"/upload/staged/{token}/abundance",
                      content=b"data").status_code == 404


@pytest.mark.parametrize("token", ["../../etc", "..%2f..%2fetc", "not-hex", ""])
def test_a_staging_key_cannot_be_steered_out_of_its_directory(client, token):
    granted = client.post("/upload/authorize", json=declare()).json()
    response = client.put(f"/upload/staged/{token}/abundance", content=b"data",
                          headers={"X-Microverse-Ticket": granted["ticket"]})
    assert response.status_code in (404, 307, 405)


# --- what the browser is handed --------------------------------------------
def test_the_local_backend_hands_out_no_credential_at_all(client):
    granted = client.post("/upload/authorize", json=declare()).json()
    for target in granted["uploads"].values():
        assert target["url"].startswith("/upload/staged/")
        assert target["headers"] == {}


def test_the_blob_backend_scopes_a_key_to_one_object(monkeypatch):
    """The store credential never leaves the server; the key names one pathname."""
    monkeypatch.setattr(config, "STORAGE_BACKEND", "blob")
    storage.reset()
    key = storage.backend()._key("0" * 24, "abundance")
    assert key == "microverse/" + "0" * 24 + "/abundance"


def test_authorize_names_only_the_files_that_were_declared(client):
    granted = client.post("/upload/authorize", json=declare(taxonomy=True)).json()
    assert set(granted["uploads"]) == {"abundance", "metadata", "taxonomy"}

    granted = client.post("/upload/authorize", json=declare()).json()
    assert set(granted["uploads"]) == {"abundance", "metadata"}


def test_a_malformed_authorize_body_is_a_clean_rejection(client):
    response = client.post("/upload/authorize", content=b"{not json",
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "error" in response.json()


# --- the original path is untouched ----------------------------------------
def test_the_plain_form_post_still_works(client):
    """No JavaScript, no direct upload — the original route, unchanged."""
    response = client.post(
        "/upload",
        files={"abundance": ("counts.tsv", ABUNDANCE, "text/tab-separated-values"),
               "metadata": ("meta.tsv", METADATA, "text/tab-separated-values")},
        data={"group_column": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/configure/")


def test_the_form_post_still_gets_an_html_error_page(client):
    """`/upload` is a browser navigation; only the steps behind it answer in JSON."""
    response = client.post(
        "/upload",
        files={"abundance": ("counts.tsv", b"rubbish", "text/tab-separated-values"),
               "metadata": ("meta.tsv", METADATA, "text/tab-separated-values")},
        data={"group_column": ""},
    )
    assert response.status_code == 422
    assert "text/html" in response.headers["content-type"]


def test_the_upload_form_markup_is_unchanged(client):
    """The enhancement is a script tag. The form it enhances must still stand alone."""
    page = client.get("/").text
    assert 'action="/upload"' in page
    assert 'enctype="multipart/form-data"' in page
    assert 'name="abundance" required' in page
    assert 'name="metadata" required' in page
    assert "/static/js/upload.js" in page


def test_none_of_the_new_routes_are_advertised(client):
    schema = client.get("/api/openapi.json").json()
    assert not [p for p in schema["paths"] if p.startswith("/upload")]
