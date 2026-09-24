"""The whole production path, in production's configuration, end to end.

upload -> ticket -> store -> dataset -> job -> queue -> worker -> analysis -> results
-> downloads, with results in (fake) Blob storage, runs delivered by a real
`vercel.queue` in this process, and the deployment root read-only. Nothing here
calls `services.execute` directly: the run happens because a message was published
by the route and consumed by the subscriber Vercel compiles, as it does in production.

The second half runs `blob/api/blob-upload.js` in Node against this application over
HTTP, so the contract between the two languages is exercised by both sides rather
than described by either. It needs Node and `blob/node_modules`; CI provides both and
sets MICROVERSE_REQUIRE_NODE=1, so there it fails rather than skips.
"""
from __future__ import annotations

import asyncio
import gzip
import http.server
import io
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import config, db, jobs, limits, queue_worker, storage
from app.main import app
from tests.fakes import SECRET, FakeBlob, use_blob

BLOB_ROOT = config.BASE_DIR / "blob"
NODE = shutil.which("node")
NODE_READY = bool(NODE) and (BLOB_ROOT / "node_modules" / "@vercel" / "blob").exists()
REQUIRE_NODE = os.environ.get("MICROVERSE_REQUIRE_NODE") == "1"


def _tsv(frame, index_name):
    frame = frame.copy()
    frame.index.name = index_name
    return frame.to_csv(sep="\t").encode()


@pytest.fixture(scope="module")
def flow(synthetic, tmp_path_factory):
    """Run the whole path once, in blob + queue mode. Every test reads its record."""
    from vercel.queue.devserver import embedded_queue_dev_server
    from vercel.queue.testing import reset_default_queue_clients

    root = tmp_path_factory.mktemp("vercel") / "var-task"
    fake = FakeBlob()
    record = {"fake": fake, "root": root}
    abundance = _tsv(synthetic["table"].counts, "taxon")
    metadata = _tsv(synthetic["metadata"], "sample_id")

    with pytest.MonkeyPatch.context() as mp, embedded_queue_dev_server() as server:
        use_blob(mp, fake, root)
        mp.setattr(config, "JOB_BACKEND", "queue")
        mp.setenv("VERCEL_QUEUE_BASE_URL", server.base_url)
        mp.setenv("VERCEL_QUEUE_TOKEN", "local-token")
        mp.setenv("VERCEL_REGION", "iad1")
        mp.setenv("VERCEL_DEPLOYMENT_ID", "dpl_test")
        reset_default_queue_clients()
        limits.rate_limiter.reset()
        client = TestClient(app)

        # 1. The browser declares its files and is told where each may go.
        granted = client.post("/upload/authorize", json={"files": {
            "abundance": {"filename": "counts.tsv", "size": len(abundance)},
            "metadata": {"filename": "meta.tsv", "size": len(metadata)},
        }})
        assert granted.status_code == 200, granted.text
        ticket = granted.json()["ticket"]
        targets = granted.json()["uploads"]
        record["targets"] = targets

        # 2. For each file: the minter asks MicroVerse, then the store takes the bytes.
        for field, payload in (("abundance", abundance), ("metadata", metadata)):
            target = targets[field]
            assert target["strategy"] == "vercel-blob"
            check = client.post("/upload/ticket",
                                json={"ticket": ticket, "pathname": target["pathname"]},
                                headers={"X-Microverse-Worker": SECRET})
            assert check.status_code == 200, check.text
            assert check.json()["maximum_size_in_bytes"] == config.MAX_UPLOAD_BYTES
            fake.objects[target["pathname"]] = payload

        # 3. The staged bytes become a dataset and a job, through the usual validation.
        completed = client.post("/upload/complete", json={"ticket": ticket})
        assert completed.status_code == 200, completed.text
        token = completed.json()["token"]
        record["token"] = token
        record["staged_left"] = [key for key in fake.objects
                                 if key.split("/")[1] != token]
        record["configure"] = client.get(f"/configure/{token}").status_code

        # 4. Starting the run publishes; it does not run anything here.
        started = client.post(f"/run/{token}", data={"mode": "quick"},
                              follow_redirects=False)
        record["run_status"] = started.status_code
        record["job_before"] = db.get_job(token).status

        # 5. The consumer Vercel builds from [[tool.vercel.subscribers]] takes it.
        deliveries = list(server.get_sync_client().poll(jobs.TOPIC, jobs.CONSUMER, limit=5))
        record["deliveries"] = len(deliveries)
        started_at = time.perf_counter()
        asyncio.run(queue_worker.run_analysis(deliveries[0].accept()))
        record["worker_seconds"] = time.perf_counter() - started_at

        # 6. Everything a reader asks for afterwards.
        record["job"] = client.get(f"/api/jobs/{token}").json()
        record["results"] = client.get(f"/results/{token}")
        record["curve"] = client.get(f"/results/{token}/curve/0")
        record["api_results"] = client.get(f"/api/jobs/{token}/results")
        record["downloads"] = {
            kind: client.get(f"/download/{token}/{kind}", follow_redirects=False)
            for kind in ("manifest", "robustness", "specifications", "attribution",
                         "methods", "bundle", "long")}
        record["grants"] = {
            name: client.post("/download/grant", json={"token": token, "name": name},
                              headers={"X-Microverse-Worker": SECRET})
            for name in ("bundle.zip", "results_long.csv.gz")}

        yield record
        reset_default_queue_clients()
    storage.reset()
    with db.session() as session:
        job = session.get(db.Job, record.get("token"))
        if job is not None:
            session.delete(job)
            session.commit()


def test_the_upload_becomes_a_job(flow):
    assert db.valid_token(flow["token"])
    assert flow["configure"] == 200


def test_every_upload_target_is_a_pathname_the_server_chose(flow):
    for field, target in flow["targets"].items():
        assert target["pathname"].startswith("microverse/")
        assert target["pathname"].endswith("/" + field)
        assert target["handler"] == config.BLOB_UPLOAD_HANDLER


def test_the_staged_copies_are_gone_once_they_are_a_dataset(flow):
    assert flow["staged_left"] == []


def test_starting_a_run_publishes_and_returns(flow):
    assert flow["run_status"] == 303
    assert flow["job_before"] == "running"
    assert flow["deliveries"] == 1, "the run was published but nothing could receive it"


def test_the_worker_finishes_the_run(flow):
    job = flow["job"]
    assert job["status"] == "done", job["error"]
    assert job["n_specs"] > 0
    assert set(job["summary"]["timings"]) == {"analysis", "robustness", "attribution",
                                              "saving"}


def test_the_results_are_readable(flow):
    assert flow["results"].status_code == 200
    assert "ROBUST" in flow["results"].text
    assert flow["curve"].status_code == 200 and flow["curve"].json()
    assert flow["api_results"].status_code == 200
    assert flow["api_results"].json()["n_taxa_total"] > 0


def test_small_exports_stream_and_large_ones_redirect(flow):
    downloads = flow["downloads"]
    for kind in ("manifest", "robustness", "specifications", "attribution", "methods"):
        assert downloads[kind].status_code == 200, kind
        assert len(downloads[kind].content) < 4_500_000
    for kind, name in (("bundle", "bundle.zip"), ("long", "results_long.csv.gz")):
        assert downloads[kind].status_code == 307, kind
        assert downloads[kind].headers["location"] == (
            f"/api/blob-download?token={flow['token']}&name={name}")


def test_what_a_grant_names_is_the_file_the_run_wrote(flow):
    bundle = flow["grants"]["bundle.zip"]
    assert bundle.status_code == 200
    raw = flow["fake"].objects[bundle.json()["pathname"]]
    assert zipfile.ZipFile(io.BytesIO(raw)).testzip() is None

    long = flow["grants"]["results_long.csv.gz"]
    assert long.status_code == 200
    text = gzip.decompress(flow["fake"].objects[long.json()["pathname"]]).decode()
    assert text.splitlines()[0].startswith("spec_id")


def test_nothing_was_written_to_the_deployment_root(flow):
    assert not flow["root"].exists()


# --- the JavaScript service against this application ---------------------------------
class _FakeBlobApi(http.server.BaseHTTPRequestHandler):
    """The Blob control API's one signing endpoint, for `issueSignedToken`."""

    seen: list = []

    def do_POST(self):                                        # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)))
        type(self).seen.append((self.path, body))
        scope = {"storeId": "teststore", "pathname": body["pathname"],
                 "operations": body["operations"], "validUntil": body["validUntil"]}
        import base64
        encoded = base64.urlsafe_b64encode(json.dumps(scope).encode()).decode().rstrip("=")
        payload = json.dumps({"delegationToken": f"{encoded}.sig",
                              "clientSigningToken": "fake-signing-token",
                              "validUntil": body["validUntil"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        return


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def live(flow):
    """This application on a real port, and a fake Blob control API beside it."""
    if not NODE_READY:
        if REQUIRE_NODE:
            pytest.fail("Node and blob/node_modules are required (MICROVERSE_REQUIRE_NODE=1)")
        pytest.skip("needs node and `npm ci` in blob/")
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    blob_api = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeBlobApi)
    threading.Thread(target=blob_api.serve_forever, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "the application did not start"
    yield {"app": f"http://127.0.0.1:{port}",
           "blob_api": f"http://127.0.0.1:{blob_api.server_address[1]}"}
    server.should_exit = True
    blob_api.shutdown()
    thread.join(timeout=10)


NODE_SCRIPT = r"""
const { POST, GET } = await import(process.env.HANDLER_URL);
const call = async (response) => {
  const text = await response.text();
  return { status: response.status, location: response.headers.get('location'), text };
};
const upload = (clientPayload, pathname) => POST(new Request('https://x.test/api/blob-upload', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ type: 'blob.generate-client-token',
                         payload: { pathname, clientPayload, multipart: false } }),
}));
const download = (token, name) => GET(new Request(
  `https://x.test/api/blob-download?token=${token}&name=${encodeURIComponent(name)}`));
const input = JSON.parse(process.env.CASES);
const out = {
  minted: await call(await upload(input.ticket, input.pathname)),
  forged: await call(await upload('not.a-ticket', input.pathname)),
  elsewhere: await call(await upload(input.ticket, 'microverse/' + 'f'.repeat(24) + '/abundance')),
  bundle: await call(await download(input.token, 'bundle.zip')),
  pickle: await call(await download(input.token, 'run.pkl.gz')),
  unknown: await call(await download('0'.repeat(24), 'bundle.zip')),
};
console.log(JSON.stringify(out));
"""


def test_the_javascript_service_and_this_application_agree(flow, live, tmp_path):
    client = TestClient(app)
    limits.rate_limiter.reset()
    granted = client.post("/upload/authorize", json={"files": {
        "abundance": {"filename": "counts.tsv", "size": 1000},
        "metadata": {"filename": "meta.tsv", "size": 100}}}).json()
    pathname = granted["uploads"]["abundance"]["pathname"]

    script = tmp_path / "drive.mjs"
    script.write_text(NODE_SCRIPT, encoding="utf-8")
    env = {**os.environ,
           "HANDLER_URL": (BLOB_ROOT / "api" / "blob-upload.js").as_uri(),
           "MICROVERSE_BACKEND_URL": live["app"],
           "MICROVERSE_WORKER_SECRET": SECRET,
           "BLOB_READ_WRITE_TOKEN": "vercel_blob_rw_teststore_notarealsecret",
           "VERCEL_BLOB_API_URL": live["blob_api"],
           "CASES": json.dumps({"ticket": granted["ticket"], "pathname": pathname,
                                "token": flow["token"]})}
    done = subprocess.run([NODE, str(script)], env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])

    # Upload: a real ticket for its own pathname mints; anything else does not.
    assert out["minted"]["status"] == 200, out["minted"]["text"]
    assert json.loads(out["minted"]["text"])["clientToken"].startswith(
        "vercel_blob_client_teststore_")
    assert out["forged"]["status"] == 403
    assert out["elsewhere"]["status"] == 403

    # Download: the finished run's bundle is signed for its own pathname, read-only.
    assert out["bundle"]["status"] == 302, out["bundle"]["text"]
    assert out["bundle"]["location"].startswith(
        f"https://teststore.private.blob.vercel-storage.com/microverse/{flow['token']}"
        "/bundle.zip?")
    assert _FakeBlobApi.seen[-1][1]["operations"] == ["get"]
    assert out["pickle"]["status"] == 404
    assert out["unknown"]["status"] == 404
    for case in out.values():
        assert SECRET not in json.dumps(case)
