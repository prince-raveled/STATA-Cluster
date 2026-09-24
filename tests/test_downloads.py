"""Downloads, on both storage backends, and the rules about who may read what.

The failure this file pins down happened in production and nowhere else. With results
in private object storage there was no direct link to give the browser, so the route
fell back to a local path -- and building that path created a directory under the
read-only deployment root. Every bundle and long-results download answered 500. The
fallback could not have worked anyway: those files are 8-11 MB for a demo, and a
function response on Vercel stops at 4.5 MB.

So in object storage the two large exports are never streamed and never rebuilt: the
browser is sent to the signing route with a grant this application signed for exactly
one object (app/grants.py), and the route redirects to a short-lived URL for it.
Everything here runs a real analysis; the Blob store is an in-memory stand-in for
`vercel.blob`, and in Blob mode the disk under the deployment root is made unwritable,
so a write that should not happen fails loudly.
"""
from __future__ import annotations

import gzip
import io
import time
import zipfile
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from app import config, db, grants, services, storage
from app.main import app
from tests.fakes import FakeBlob, use_blob

#: The largest body a Vercel Function may return.
RESPONSE_CAP = 4_500_000
SMALL = ("manifest", "robustness", "specifications", "attribution", "methods")
LARGE = {"bundle": "bundle.zip", "long": "results_long.csv.gz"}


def finish(dataset, name="synthetic counts.tsv") -> str:
    """Create and run a real Quick analysis, synchronously. Returns the token."""
    token = services.create_job(dataset, name)
    services.execute(token, "quick")
    job = db.get_job(token)
    assert job.status == "done", job.error
    return token


def drop(token):
    with db.session() as session:
        job = session.get(db.Job, token)
        if job is not None:
            session.delete(job)
            session.commit()


def add_job(status="running", created_at=None) -> str:
    token = db.new_token()
    with db.session() as session:
        session.add(db.Job(token=token, status=status, mode="quick",
                           dataset_name="x.tsv", created_at=created_at or db.utcnow()))
        session.commit()
    return token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_backend():
    storage.reset()
    yield
    storage.reset()


# --- one real run per backend ------------------------------------------------
@pytest.fixture(scope="module")
def local_run(dataset):
    storage.reset()
    token = finish(dataset)
    yield SimpleNamespace(token=token)
    storage.purge(token)
    drop(token)


@pytest.fixture(scope="module")
def blob_run(dataset, tmp_path_factory):
    """A real analysis written to the fake store with the deployment root read-only.

    Returns the token and a snapshot of what the store held afterwards; each test
    gets its own copy of that snapshot, so none can disturb another.
    """
    root = tmp_path_factory.mktemp("vercel") / "var-task"
    fake = FakeBlob()
    with pytest.MonkeyPatch.context() as mp:
        use_blob(mp, fake, root)
        token = finish(dataset)
    storage.reset()
    assert not root.exists(), "the run wrote to the read-only deployment root"
    yield SimpleNamespace(token=token, objects=dict(fake.objects), puts=list(fake.puts))
    drop(token)


@pytest.fixture
def blob(blob_run, tmp_path, monkeypatch):
    """Blob mode, a fresh copy of the finished run's store, and a read-only root.

    The exports of this small test run fit in a function response, which the app would
    now hand over itself; the limit is set to nothing here so these tests keep
    exercising the path a real 8-60 MB export takes. `small_blob` tests the other one.
    """
    root = tmp_path / "var-task"
    fake = FakeBlob(blob_run.objects)
    use_blob(monkeypatch, fake, root)
    monkeypatch.setattr(config, "APP_RESPONSE_LIMIT_BYTES", 0)
    yield SimpleNamespace(fake=fake, token=blob_run.token, root=root)
    assert not root.exists(), "a request wrote to the read-only deployment root"


@pytest.fixture
def small_blob(blob_run, tmp_path, monkeypatch):
    """Blob mode with the real response limit."""
    root = tmp_path / "var-task"
    fake = FakeBlob(blob_run.objects)
    use_blob(monkeypatch, fake, root)
    yield SimpleNamespace(fake=fake, token=blob_run.token, root=root)
    assert not root.exists(), "a request wrote to the read-only deployment root"


# --- local storage: what a server with a disk does ------------------------------
@pytest.mark.parametrize("kind", sorted(LARGE))
def test_local_large_exports_are_the_stored_files(client, local_run, kind):
    response = client.get(f"/download/{local_run.token}/{kind}")
    assert response.status_code == 200
    stored = (config.JOBS_DIR / local_run.token / LARGE[kind]).read_bytes()
    assert response.content == stored, "the download is not the file the run wrote"
    assert 'filename="synthetic_' in response.headers["content-disposition"]
    expected = "application/zip" if kind == "bundle" else "application/gzip"
    assert response.headers["content-type"].startswith(expected)
    assert int(response.headers["content-length"]) == len(stored)


def test_local_bundle_is_a_complete_archive(client, local_run):
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/download/{local_run.token}/bundle").content))
    assert archive.testzip() is None, "the bundle is corrupt"
    assert "results_long.csv.gz" in archive.namelist()


def test_local_long_results_decompress(client, local_run):
    text = gzip.decompress(client.get(f"/download/{local_run.token}/long").content).decode()
    header = text.splitlines()[0]
    assert "spec_id" in header and "taxon" in header


@pytest.mark.parametrize("kind", SMALL)
def test_local_small_exports_still_stream(client, local_run, kind):
    response = client.get(f"/download/{local_run.token}/{kind}")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert len(response.content) > 100


def test_a_local_run_from_before_exports_were_stored_is_rebuilt(client, local_run):
    """The original behaviour for an old job whose file is missing, kept on purpose."""
    path = config.JOBS_DIR / local_run.token / "bundle.zip"
    original = path.read_bytes()
    path.unlink()
    try:
        response = client.get(f"/download/{local_run.token}/bundle")
        assert response.status_code == 200
        assert zipfile.ZipFile(io.BytesIO(response.content)).testzip() is None
    finally:
        path.write_bytes(original)


# --- object storage: what Vercel does -----------------------------------------------
def test_a_blob_run_stores_every_export_and_writes_nothing_to_disk(blob_run):
    names = {key.rsplit("/", 1)[-1] for key in blob_run.objects
             if key.startswith(f"microverse/{blob_run.token}/")}
    assert names >= {"dataset.pkl.gz", "run.pkl.gz", "summary.pkl.gz",
                     "attribution.pkl.gz", "manifest.json", "methods.txt",
                     "results_long.csv.gz", "bundle.zip"}
    assert {access for _, access, _ in blob_run.puts} == {"private"}


@pytest.mark.parametrize("kind", sorted(LARGE))
def test_a_large_blob_export_is_a_redirect_not_a_body(client, blob, kind):
    response = client.get(f"/download/{blob.token}/{kind}", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"].startswith(f"{config.BLOB_DOWNLOAD_HANDLER}?grant=")
    assert granted(response)["pathname"] == f"microverse/{blob.token}/{LARGE[kind]}"
    assert len(response.content) < 1024, "the file was streamed through the function"


@pytest.mark.parametrize("kind", sorted(LARGE))
def test_the_redirect_is_to_a_same_origin_route(client, blob, kind):
    """Relative, so a preview deployment sends the browser to its own signer."""
    location = client.get(f"/download/{blob.token}/{kind}",
                          follow_redirects=False).headers["location"]
    assert location.startswith("/api/blob-download?")


@pytest.mark.parametrize("kind", sorted(LARGE))
def test_a_stored_export_that_fits_a_response_comes_from_the_app(client, small_blob, kind):
    """Not through the store's own domain, which some networks block outright: a
    campus firewall answered every *.blob.vercel-storage.com request with a block page,
    so every bundle download failed there although the site itself worked."""
    stored = small_blob.fake.objects[f"microverse/{small_blob.token}/{LARGE[kind]}"]
    assert len(stored) <= config.APP_RESPONSE_LIMIT_BYTES
    response = client.get(f"/download/{small_blob.token}/{kind}", follow_redirects=False)
    assert response.status_code == 200
    assert response.content == stored, "the download is not the file the run wrote"
    assert "attachment" in response.headers["content-disposition"]


def test_a_stored_export_over_the_response_limit_is_still_signed(client, small_blob, monkeypatch):
    stored = small_blob.fake.objects[f"microverse/{small_blob.token}/bundle.zip"]
    monkeypatch.setattr(config, "APP_RESPONSE_LIMIT_BYTES", len(stored) - 1)
    response = client.get(f"/download/{small_blob.token}/bundle", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"].startswith(f"{config.BLOB_DOWNLOAD_HANDLER}?grant=")


def test_the_response_limit_leaves_room_under_vercels_cap():
    assert 0 < config.APP_RESPONSE_LIMIT_BYTES < RESPONSE_CAP


@pytest.mark.parametrize("kind", SMALL)
def test_small_blob_exports_stream_under_the_response_cap(client, blob, kind):
    response = client.get(f"/download/{blob.token}/{kind}")
    assert response.status_code == 200
    assert 100 < len(response.content) < RESPONSE_CAP
    assert "attachment" in response.headers["content-disposition"]


def test_the_stored_bundle_is_intact(blob):
    raw = blob.fake.objects[f"microverse/{blob.token}/bundle.zip"]
    archive = zipfile.ZipFile(io.BytesIO(raw))
    assert archive.testzip() is None
    assert {"results_long.csv.gz", "manifest.json", "README.txt"} <= set(archive.namelist())


def test_a_blob_export_that_is_gone_is_a_clean_404(client, blob):
    del blob.fake.objects[f"microverse/{blob.token}/bundle.zip"]
    response = client.get(f"/download/{blob.token}/bundle", follow_redirects=False)
    assert response.status_code == 404
    assert "no longer stored" in response.text


def test_a_store_that_fails_is_not_reported_as_an_expired_file(blob, monkeypatch):
    """Only "not found" is absence. A refused credential must surface as an error."""
    def broken(_pathname):
        raise RuntimeError("the store refused the credential")

    monkeypatch.setattr(blob.fake, "head", broken)
    response = TestClient(app, raise_server_exceptions=False).get(
        f"/download/{blob.token}/bundle", follow_redirects=False)
    assert response.status_code == 500


def test_public_objects_are_linked_directly(client, blob, monkeypatch):
    monkeypatch.setattr(config, "BLOB_ACCESS", "public")
    response = client.get(f"/download/{blob.token}/bundle", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == (
        f"https://teststore.public.blob.vercel-storage.com/microverse/{blob.token}"
        "/bundle.zip?download=1")


@pytest.mark.parametrize("kind", ["bundle", "long", "manifest"])
def test_an_unknown_job_is_404_in_blob_mode(client, blob, kind):
    assert client.get(f"/download/{db.new_token()}/{kind}").status_code == 404


@pytest.mark.parametrize("kind", ["bundle", "long", "manifest"])
def test_an_unfinished_job_is_409_not_404(client, blob, kind):
    token = add_job(status="running")
    try:
        assert client.get(f"/download/{token}/{kind}").status_code == 409
    finally:
        drop(token)


def test_an_expired_job_is_gone_everywhere(client, blob):
    """SPEC 16.5: retention removes the row and the objects, and the link says so."""
    import datetime as dt

    token = add_job(status="done",
                    created_at=db.utcnow() - dt.timedelta(days=config.RETENTION_DAYS + 1))
    blob.fake.objects[f"microverse/{token}/bundle.zip"] = b"PK\x05\x06" + b"\0" * 18
    assert db.purge_expired() >= 1
    assert f"microverse/{token}/bundle.zip" not in blob.fake.objects
    assert client.get(f"/download/{token}/bundle").status_code == 404


@pytest.mark.parametrize("token", ["not-a-token", "A" * 24, "0" * 23, "0" * 25])
def test_a_malformed_token_is_404_in_blob_mode(client, blob, token):
    assert client.get(f"/download/{token}/bundle").status_code == 404


# --- the grant the signing route acts on ------------------------------------------
def granted(response):
    """The claims of the grant a download redirect carries, or None if it is invalid."""
    raw = parse_qs(urlparse(response.headers["location"]).query)["grant"][0]
    return grants.verify(raw, "download")


def test_a_grant_names_one_object_for_a_short_time(client, blob):
    before = time.time()
    response = client.get(f"/download/{blob.token}/bundle", follow_redirects=False)
    grant = granted(response)
    assert grant["pathname"] == f"microverse/{blob.token}/bundle.zip"
    assert grant["access"] == "private"
    ttl_ms = config.DOWNLOAD_URL_TTL_SECONDS * 1000
    assert before * 1000 + ttl_ms - 5000 <= grant["valid_until"] <= time.time() * 1000 + ttl_ms


def test_only_the_large_exports_are_ever_signed(client, blob):
    """The pickled run, the dataset and staged uploads are not downloads, and nothing
    else a reader can ask for is sent to the signer."""
    import app.routers.results as results

    signed = set()
    for kind in results.DOWNLOADS:
        response = client.get(f"/download/{blob.token}/{kind}", follow_redirects=False)
        if response.status_code == 307:
            signed.add(granted(response)["pathname"].rsplit("/", 1)[-1])
    assert signed == set(LARGE.values())


def test_a_grant_expires(client, blob, monkeypatch):
    response = client.get(f"/download/{blob.token}/bundle", follow_redirects=False)
    later = time.time() + config.DOWNLOAD_URL_TTL_SECONDS + 5
    monkeypatch.setattr(grants.time, "time", lambda: later)
    assert granted(response) is None


def test_a_download_grant_cannot_be_used_to_upload(client, blob):
    response = client.get(f"/download/{blob.token}/bundle", follow_redirects=False)
    raw = parse_qs(urlparse(response.headers["location"]).query)["grant"][0]
    assert grants.verify(raw, "upload") is None


def test_a_grant_from_another_deployment_is_refused(client, blob, monkeypatch):
    response = client.get(f"/download/{blob.token}/bundle", follow_redirects=False)
    monkeypatch.setattr(config, "WORKER_SECRET", "another-deployments-secret")
    assert granted(response) is None


def test_no_grant_is_made_for_an_unfinished_job(client, blob):
    token = add_job(status="running")
    try:
        response = client.get(f"/download/{token}/bundle", follow_redirects=False)
        assert response.status_code == 409
        assert "grant" not in response.headers.get("location", "")
    finally:
        drop(token)


def test_the_old_grant_route_is_gone(client):
    assert client.post("/download/grant", json={}).status_code in (404, 405)
    paths = client.get("/api/openapi.json").json()["paths"]
    assert "/download/grant" not in paths


def test_the_signing_route_is_declared_to_the_host():
    import json

    manifest = json.loads((config.BASE_DIR / "vercel.json").read_text(encoding="utf-8"))
    sources = [rule["source"] for rule in manifest["rewrites"]]
    routed = [rule for rule in manifest["rewrites"]
              if rule["source"] == config.BLOB_DOWNLOAD_HANDLER]
    assert routed and routed[0]["destination"]["service"] == "blob_upload"
    assert sources.index(config.BLOB_DOWNLOAD_HANDLER) < sources.index("/(.*)"), (
        "the catch-all would send the signing route to FastAPI")
