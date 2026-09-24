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
#: The token minter. Its own service, laid out the way Vercel finds file-based
#: functions: <service root>/api/<name>.js
MINTER = config.BASE_DIR / "blob" / "api" / "blob-upload.js"
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
    # The first page after an upload shows how the files were read, and leads on to
    # configuring the run.
    assert body["next"] == f"/validate/{body['token']}"

    page = client.get(body["next"])
    assert page.status_code == 200
    assert f"/configure/{body['token']}" in page.text


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


def test_a_backend_that_cannot_delegate_says_so_instead_of_improvising(client, monkeypatch):
    """The Python SDK cannot mint a browser upload token, so Blob must refuse.

    The wrong answers here are both dangerous: handing over BLOB_READ_WRITE_TOKEN,
    which writes anything in the store, or pretending the upload was authorised and
    failing later. A 501 tells the script to post the form instead.
    """
    class Refusing:
        name = "refusing"

        def authorize(self, token, name, size):  # noqa: ARG002
            raise storage.DirectUploadUnavailable("no")

    monkeypatch.setattr(storage, "_backend", Refusing())
    response = client.post("/upload/authorize", json=declare())

    assert response.status_code == 501
    assert response.json()["fallback"] == "form"
    body = response.text.lower()
    assert "blob_read_write_token" not in body and "bearer" not in body


def test_the_blob_backend_delegates_minting_instead_of_refusing(monkeypatch):
    """Python cannot sign a client token, so it names the endpoint that can.

    The wrong answers are handing over BLOB_READ_WRITE_TOKEN, which writes anything
    in the store, or letting the browser choose where to write. It does neither: the
    pathname is the server's and /upload/ticket checks it again.
    """
    import vercel.blob

    assert not hasattr(vercel.blob, "generate_client_token"), (
        "the Python SDK grew a token minter; the JS endpoint may no longer be needed"
    )
    monkeypatch.setattr(config, "STORAGE_BACKEND", "blob")
    storage.reset()

    target = storage.backend().authorize("0" * 24, "abundance", 10)
    assert target["strategy"] == "vercel-blob"
    assert target["handler"] == config.BLOB_UPLOAD_HANDLER
    assert target["pathname"] == "microverse/" + "0" * 24 + "/abundance"
    assert "token" not in str(target).lower()


def test_a_deployment_without_a_minting_endpoint_refuses(monkeypatch):
    monkeypatch.setattr(config, "STORAGE_BACKEND", "blob")
    monkeypatch.setattr(config, "BLOB_UPLOAD_HANDLER", "")
    storage.reset()
    with pytest.raises(storage.DirectUploadUnavailable):
        storage.backend().authorize("0" * 24, "abundance", 10)


def test_the_local_backend_names_its_strategy_too(client):
    """One field for the browser to branch on, whatever the host provides."""
    granted = client.post("/upload/authorize", json=declare()).json()
    for target in granted["uploads"].values():
        assert target["strategy"] == "staged"


# --- /upload/ticket: the rules stay in Python -------------------------------
def _blob_mode(monkeypatch):
    monkeypatch.setattr(config, "STORAGE_BACKEND", "blob")
    monkeypatch.setattr(config, "WORKER_SECRET", "s3cret")
    storage.reset()


def test_the_minter_is_told_the_limits_for_an_authorised_pathname(client, monkeypatch):
    _blob_mode(monkeypatch)
    granted = client.post("/upload/authorize", json=declare()).json()
    pathname = granted["uploads"]["abundance"]["pathname"]

    response = client.post("/upload/ticket",
                           json={"ticket": granted["ticket"], "pathname": pathname},
                           headers={"X-Microverse-Worker": "s3cret"})
    assert response.status_code == 200
    body = response.json()
    assert body["maximum_size_in_bytes"] == config.MAX_UPLOAD_BYTES
    assert "text/*" in body["allowed_content_types"]
    assert body["valid_until"] > 0


def test_the_minter_must_present_the_shared_secret(client, monkeypatch):
    """Otherwise the token minter is an open door into the store."""
    _blob_mode(monkeypatch)
    granted = client.post("/upload/authorize", json=declare()).json()
    pathname = granted["uploads"]["abundance"]["pathname"]

    for headers in ({}, {"X-Microverse-Worker": "wrong"}):
        response = client.post("/upload/ticket",
                               json={"ticket": granted["ticket"], "pathname": pathname},
                               headers=headers)
        assert response.status_code == 403


def test_an_unconfigured_deployment_authorises_nobody(client, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_BACKEND", "blob")
    monkeypatch.setattr(config, "WORKER_SECRET", "")
    storage.reset()
    response = client.post("/upload/ticket", json={"ticket": "x", "pathname": "y"},
                           headers={"X-Microverse-Worker": ""})
    assert response.status_code == 403


def test_a_ticket_cannot_authorise_a_pathname_it_does_not_own(client, monkeypatch):
    """The browser supplies the pathname to upload(), so it is checked, not trusted."""
    _blob_mode(monkeypatch)
    mine = client.post("/upload/authorize", json=declare()).json()
    theirs = client.post("/upload/authorize", json=declare()).json()
    their_path = theirs["uploads"]["abundance"]["pathname"]

    response = client.post("/upload/ticket",
                           json={"ticket": mine["ticket"], "pathname": their_path},
                           headers={"X-Microverse-Worker": "s3cret"})
    assert response.status_code == 403


@pytest.mark.parametrize("pathname", [
    "", "microverse/../../etc/passwd", "other-prefix/x/abundance",
    "microverse/0000/taxonomy", "/etc/passwd",
])
def test_an_arbitrary_pathname_is_refused(client, monkeypatch, pathname):
    _blob_mode(monkeypatch)
    granted = client.post("/upload/authorize", json=declare()).json()
    response = client.post("/upload/ticket",
                           json={"ticket": granted["ticket"], "pathname": pathname},
                           headers={"X-Microverse-Worker": "s3cret"})
    assert response.status_code == 403


def test_a_field_the_ticket_never_named_has_no_pathname(client, monkeypatch):
    """No taxonomy was declared, so no token may be minted for one."""
    _blob_mode(monkeypatch)
    granted = client.post("/upload/authorize", json=declare()).json()
    staging = uploads.staging_token(uploads.verify(granted["ticket"])["id"])

    response = client.post(
        "/upload/ticket",
        json={"ticket": granted["ticket"],
              "pathname": f"microverse/{staging}/taxonomy"},
        headers={"X-Microverse-Worker": "s3cret"})
    assert response.status_code == 403


def test_an_expired_ticket_mints_nothing(client, monkeypatch):
    _blob_mode(monkeypatch)
    granted = client.post("/upload/authorize", json=declare()).json()
    pathname = granted["uploads"]["abundance"]["pathname"]

    later = time.time() + uploads.TICKET_TTL_SECONDS + 5
    monkeypatch.setattr(uploads.time, "time", lambda: later)
    response = client.post("/upload/ticket",
                           json={"ticket": granted["ticket"], "pathname": pathname},
                           headers={"X-Microverse-Worker": "s3cret"})
    assert response.status_code == 403


def test_the_minting_endpoint_is_declared_to_the_host():
    """Vercel must route /api/blob-upload to the JS service, not to FastAPI."""
    import json as _json

    manifest = _json.loads((config.BASE_DIR / "vercel.json").read_text(encoding="utf-8"))
    assert "blob_upload" in manifest["services"]
    routed = [r for r in manifest["rewrites"]
              if r["source"] == config.BLOB_UPLOAD_HANDLER]
    assert routed, f"nothing routes {config.BLOB_UPLOAD_HANDLER}"
    assert routed[0]["destination"]["service"] == "blob_upload"
    # The catch-all must come after it, or FastAPI would swallow the route.
    sources = [r["source"] for r in manifest["rewrites"]]
    assert sources.index(config.BLOB_UPLOAD_HANDLER) < sources.index("/(.*)")


def test_the_two_languages_agree_on_what_upload_ticket_returns():
    """The minter is JavaScript and the authority is Python, so the contract drifts
    silently unless something checks it. Renaming a field on either side fails here
    rather than at the first upload on a deployment nobody has tested yet.
    """
    import re

    js = MINTER.read_text(encoding="utf-8")
    reads = set(re.findall(r"granted\.([a-z_]+)", js))

    router = (config.BASE_DIR / "app" / "routers" / "upload.py").read_text(encoding="utf-8")
    block = router.split("async def ticket(")[1].split("@router")[0]
    returns = set(re.findall(r'"([a-z_]+)":', block)) - {"error"}

    assert reads, "the JavaScript reads nothing from the authorisation response"
    assert reads <= returns, f"JavaScript reads fields Python never sends: {reads - returns}"


def test_the_minter_holds_no_rules_of_its_own():
    """Every limit belongs to app/uploads.py. A number here is a second source of truth."""
    import re

    js = MINTER.read_text(encoding="utf-8")
    # The whole file, not only the upload handler: the download handler and the
    # helpers both share it. Comments explain the rules; only executable lines may
    # not restate them.
    code = " ".join(line for line in js.splitlines()
                    if not line.strip().startswith("//"))

    # The HTTP statuses the route answers with are vocabulary, not limits. Named one
    # by one, so a size or a lifetime that happens to have three digits still fails.
    statuses = {"302", "400", "403", "404", "409", "500", "502"}
    numbers = set(re.findall(r"\b\d+\b", code)) - statuses
    assert not numbers, (
        f"the minter contains its own numeric limits {numbers}; every limit must "
        "come from /upload/ticket so there is one source of truth"
    )
    assert ".tsv" not in code and ".biom" not in code, (
        "filename rules look duplicated in the minter"
    )


def test_the_minter_sits_where_vercel_finds_a_function():
    """A bare .js at a service root is detected as static content and never runs.

    Verified against the real builder: with the file under <root>/api/ and an
    `entrypoint`, `vercel dev` reports the service as [node]; without, it reports
    [@vercel/static], which would serve the source instead of executing it.
    """
    import json as _json

    manifest = _json.loads((config.BASE_DIR / "vercel.json").read_text(encoding="utf-8"))
    service = manifest["services"]["blob_upload"]
    assert MINTER.exists(), f"the minter is not at {MINTER}"
    assert MINTER.relative_to(config.BASE_DIR / service["root"]).as_posix() == service["entrypoint"]
    assert service["entrypoint"].startswith("api/")


def test_the_python_bundle_excludes_the_javascript_service():
    """Function settings belong under the service's `functions`, not the service.

    The older `experimentalServices` model took `maxDuration` and `excludeFiles`
    directly on a service; the current `services` model does not, and a config that
    puts them there is rejected by the published schema. The location is asserted
    here as well as the content, so the two cannot drift apart again.
    """
    import json as _json

    manifest = _json.loads((config.BASE_DIR / "vercel.json").read_text(encoding="utf-8"))
    service = manifest["services"]["microverse"]
    assert "excludeFiles" not in service and "maxDuration" not in service

    entry = service["functions"]["app/main.py"]
    assert "blob/**" in entry["excludeFiles"]
    assert entry["maxDuration"] == 300
    assert service["framework"] == "fastapi"


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
    assert response.headers["location"].startswith("/validate/")


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
