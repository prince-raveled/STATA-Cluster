"""Where a job's bytes live — the one seam between MicroVerse and its host.

The engine writes four exports and three pickled payloads per run, and the results
pages read them back. On a server with a disk that is just a directory; on Vercel
there is no disk that survives the request, so the same seven objects have to go to
object storage instead.

Both backends implement the same five operations and nothing else:

    put(token, name, data)   store bytes
    get(token, name)         read them back, or None
    purge(token)             delete everything for one job
    url(token, name)         a direct download link, or None to stream via the app
    authorize(token, name)   where the browser may write one object, and how

`local` is the default and is byte-for-byte what MicroVerse has always done — same
directory, same filenames — so the existing suite exercises the real code path
rather than a stub. `blob` is selected only by environment variable, and its SDK is
imported inside the backend so that neither local development nor the tests pay for
a dependency they never call.

Retention (SPEC §16.5) is the database's job; this module only deletes when told.
"""
from __future__ import annotations

import gzip
import io
import pickle
import shutil
import time
from typing import Protocol

from . import config

#: Content types for the four exports, so a browser downloading straight from
#: object storage gets the same headers the app would have sent.
CONTENT_TYPES = {
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".gz": "application/gzip",
    ".zip": "application/zip",
    ".pkl": "application/octet-stream",
}


def _content_type(name: str) -> str:
    for suffix, value in CONTENT_TYPES.items():
        if name.endswith(suffix):
            return value
    return "application/octet-stream"


class Backend(Protocol):
    def put(self, token: str, name: str, data: bytes) -> None: ...
    def get(self, token: str, name: str) -> bytes | None: ...
    def purge(self, token: str) -> None: ...
    def url(self, token: str, name: str) -> str | None: ...
    def authorize(self, token: str, name: str, size: int) -> dict: ...


# --- the disk, as it has always been ---------------------------------------
class LocalBackend:
    """A directory per token. Unchanged from the original implementation."""

    name = "local"

    def put(self, token: str, name: str, data: bytes) -> None:
        (config.job_dir(token) / name).write_bytes(data)

    def get(self, token: str, name: str) -> bytes | None:
        path = config.job_dir(token) / name
        return path.read_bytes() if path.exists() else None

    def purge(self, token: str) -> None:
        shutil.rmtree(config.JOBS_DIR / token, ignore_errors=True)

    def url(self, token: str, name: str) -> str | None:  # noqa: ARG002
        return None          # the app streams it, as it does today

    def authorize(self, token: str, name: str, size: int) -> dict:  # noqa: ARG002
        """Point the browser back at this application.

        There is nothing to delegate to when storage is a directory, so the staging
        route accepts the bytes itself. The browser follows the same two-step protocol
        either way, which is what lets the suite exercise the real upload flow.
        """
        return {"url": f"/upload/staged/{token}/{name}", "method": "PUT", "headers": {}}


# --- Vercel Blob ------------------------------------------------------------
class BlobBackend:
    """Vercel Blob, one prefix per token.

    Blobs are created private: a MicroVerse token is unguessable, but a result URL
    that is public forever is a different promise from one the app can stop serving,
    and microbiome data is not ours to make public by default. Downloads are handed
    out as short-lived signed URLs instead, which also keeps result bundles off the
    4.5 MB function response limit.
    """

    name = "blob"

    def __init__(self, prefix: str = "microverse"):
        self.prefix = prefix.strip("/")

    def _key(self, token: str, name: str) -> str:
        return f"{self.prefix}/{token}/{name}"

    @staticmethod
    def _blob():
        # Imported here so local runs and the test suite never load the SDK.
        from vercel import blob
        return blob

    def put(self, token: str, name: str, data: bytes) -> None:
        self._blob().put(
            self._key(token, name), data,
            options={
                "access": "private",
                "addRandomSuffix": False,
                "contentType": _content_type(name),
                "allowOverwrite": True,
            },
        )

    def get(self, token: str, name: str) -> bytes | None:
        blob = self._blob()
        try:
            return blob.get(self._key(token, name)).content
        except Exception:                                     # noqa: BLE001
            return None          # absent or expired reads the same as absent on disk

    def purge(self, token: str) -> None:
        blob = self._blob()
        prefix = f"{self.prefix}/{token}/"
        try:
            listing = blob.list(options={"prefix": prefix})
            keys = [item.pathname for item in getattr(listing, "blobs", [])]
            if keys:
                blob.delete(keys)
        except Exception:                                     # noqa: BLE001
            return           # purge is best-effort; the database row is authoritative

    def url(self, token: str, name: str) -> str | None:
        try:
            return self._blob().get_download_url(
                self._key(token, name),
                options={"expiresIn": config.DOWNLOAD_URL_TTL_SECONDS},
            )
        except Exception:                                     # noqa: BLE001
            return None          # fall back to streaming through the app

    def authorize(self, token: str, name: str, size: int) -> dict:
        """A client token for one object, and nothing else.

        `BLOB_READ_WRITE_TOKEN` writes anything in the store and never leaves the
        server. What the browser receives is minted from it and scoped to this single
        pathname with its own expiry, so the worst a stolen one can do is overwrite
        the upload it was issued for.
        """
        client_token = self._blob().generate_client_token(
            self._key(token, name),
            options={
                "access": "private",
                "addRandomSuffix": False,
                "allowOverwrite": True,
                "maximumSizeInBytes": size,
                "validUntil": int(time.time() + config.UPLOAD_URL_TTL_SECONDS) * 1000,
            },
        )
        return {
            "url": f"https://blob.vercel-storage.com/{self._key(token, name)}",
            "method": "PUT",
            "headers": {"Authorization": f"Bearer {client_token}"},
        }


_backend: Backend | None = None


def backend() -> Backend:
    global _backend
    if _backend is None:
        _backend = BlobBackend() if config.STORAGE_BACKEND == "blob" else LocalBackend()
    return _backend


def reset() -> None:
    """Forget the cached backend. Tests switch backends; nothing else needs this."""
    global _backend
    _backend = None


# --- the API the rest of MicroVerse uses -----------------------------------
def put_bytes(token: str, name: str, data: bytes) -> None:
    backend().put(token, name, data)


def get_bytes(token: str, name: str) -> bytes | None:
    return backend().get(token, name)


def put_text(token: str, name: str, text: str) -> None:
    backend().put(token, name, text.encode("utf-8"))


def put_object(token: str, name: str, obj) -> None:
    """Gzipped pickle, the format results have always been stored in."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=4, mtime=0) as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    backend().put(token, f"{name}.pkl.gz", buffer.getvalue())


def get_object(token: str, name: str):
    raw = backend().get(token, f"{name}.pkl.gz")
    if raw is None:
        return None
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
        return pickle.load(handle)


def purge(token: str) -> None:
    backend().purge(token)


def download_url(token: str, name: str) -> str | None:
    """A direct link for the browser, or None when the app should stream it."""
    return backend().url(token, name)


def authorize_upload(token: str, name: str, size: int) -> dict:
    """Where the browser may write this one object, and what to send with it."""
    return backend().authorize(token, name, size)
