"""Where a job's bytes live — the one seam between MicroVerse and its host.

The engine writes four exports and three pickled payloads per run, and the results
pages read them back. On a server with a disk that is just a directory; on Vercel
there is no disk that survives the request, so the same seven objects have to go to
object storage instead.

Both backends implement the same operations and nothing else:

    put(token, name, data)   store bytes
    get(token, name)         read them back, or None
    exists(token, name)      whether an object is stored, without reading it
    purge(token)             delete everything for one job
    url(token, name)         a link the browser can follow, or None to stream via the app
    authorize(token, name)   where the browser may write one object, and how

`local` is the default and is byte-for-byte what MicroVerse has always done — same
directory, same filenames — so the existing suite exercises the real code path
rather than a stub. `blob` is selected only by environment variable, and its SDK is
imported inside the backend so that neither local development nor the tests pay for
a dependency they never call.

Every token and name is checked here, at the one place that turns them into a path
or a key, rather than trusted from each caller. A token is 24 lowercase hex
characters and a name is one plain path segment; nothing else reaches a filesystem
or the store.

Retention (SPEC §16.5) is the database's job; this module only deletes when told.
"""
from __future__ import annotations

import gzip
import io
import pickle
import re
import shutil
import time
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode

from . import config, grants

#: A job or staging token: what `db.new_token` and `uploads.issue` generate.
TOKEN_PATTERN = re.compile(r"^[0-9a-f]{24}$")
#: One path segment: a stored export, a pickled payload, or an upload field.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def checked(token: str, name: str | None = None) -> str:
    """Refuse anything that could address outside one job's own objects.

    Raised rather than answered with "absent": every legitimate caller has already
    validated the token or generated it, so reaching this with a bad one is a bug,
    and a bug that builds a path from a URL segment must not fail quietly.
    """
    if not TOKEN_PATTERN.match(str(token or "")):
        raise ValueError(f"not a MicroVerse token: {token!r}")
    if name is not None and (not _NAME_PATTERN.match(str(name)) or ".." in str(name)):
        raise ValueError(f"not a storable name: {name!r}")
    return token

#: Content types for the four exports, so a browser downloading straight from
#: object storage gets the same headers the app would have sent.
CONTENT_TYPES = {
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".gz": "application/gzip",
    ".zip": "application/zip",
    ".pkl": "application/octet-stream",
}


class DirectUploadUnavailable(RuntimeError):
    """Raised when a backend cannot grant a browser permission to upload."""


def _content_type(name: str) -> str:
    for suffix, value in CONTENT_TYPES.items():
        if name.endswith(suffix):
            return value
    return "application/octet-stream"


class Backend(Protocol):
    def put(self, token: str, name: str, data: bytes) -> None: ...
    def get(self, token: str, name: str) -> bytes | None: ...
    def exists(self, token: str, name: str) -> bool: ...
    def purge(self, token: str) -> None: ...
    def url(self, token: str, name: str) -> str | None: ...
    def authorize(self, token: str, name: str, size: int) -> dict: ...


# --- the disk, as it has always been ---------------------------------------
class LocalBackend:
    """A directory per token, as it has always been.

    Only `put` creates the directory. A read used to create it too, so looking up a
    token that was never issued left an empty directory behind for every guess.
    """

    name = "local"

    @staticmethod
    def path(token: str, name: str) -> Path:
        """Where an object lives. Creates nothing."""
        return config.JOBS_DIR / checked(token, name) / name

    def put(self, token: str, name: str, data: bytes) -> None:
        (config.job_dir(checked(token, name)) / name).write_bytes(data)

    def get(self, token: str, name: str) -> bytes | None:
        path = self.path(token, name)
        return path.read_bytes() if path.exists() else None

    def exists(self, token: str, name: str) -> bool:
        return self.path(token, name).exists()

    def purge(self, token: str) -> None:
        shutil.rmtree(config.JOBS_DIR / checked(token), ignore_errors=True)

    def url(self, token: str, name: str) -> str | None:  # noqa: ARG002
        return None          # the app streams it, as it does today

    def authorize(self, token: str, name: str, size: int) -> dict:  # noqa: ARG002
        """Point the browser back at this application.

        There is nothing to delegate to when storage is a directory, so the staging
        route accepts the bytes itself. The browser follows the same two-step protocol
        either way, which is what lets the suite exercise the real upload flow.
        """
        return {"strategy": "staged", "method": "PUT", "headers": {},
                "url": f"/upload/staged/{token}/{name}"}


# --- Vercel Blob ------------------------------------------------------------
class BlobBackend:
    """Vercel Blob, one prefix per token.

    Blobs are created private by default: a MicroVerse token is unguessable, but a
    result URL that is public forever is a different promise from one the app can
    stop serving, and microbiome data is not ours to make public by default.

    The Python SDK cannot sign a URL, so a private object is read here with the store
    credential, or -- for the exports too large for a function's response -- through
    a short-lived URL the JavaScript service signs (see `url`).
    `MICROVERSE_BLOB_ACCESS=public` serves objects from permanent public URLs instead,
    a different promise about the data, so it is never chosen silently.
    """

    name = "blob"

    def __init__(self, prefix: str = "microverse"):
        self.prefix = prefix.strip("/")

    def _key(self, token: str, name: str) -> str:
        return f"{self.prefix}/{checked(token, name)}/{name}"

    @staticmethod
    def _blob():
        # Imported here so local runs and the test suite never load the SDK.
        from vercel import blob
        return blob

    def put(self, token: str, name: str, data: bytes) -> None:
        self._blob().put(
            self._key(token, name), data,
            access=config.BLOB_ACCESS,
            content_type=_content_type(name),
            add_random_suffix=False,
            overwrite=True,
        )

    def get(self, token: str, name: str) -> bytes | None:
        blob = self._blob()
        try:
            return blob.get(self._key(token, name), access=config.BLOB_ACCESS).content
        except Exception:                                     # noqa: BLE001
            return None          # absent or expired reads the same as absent on disk

    def exists(self, token: str, name: str) -> bool:
        """Whether the object is stored. Only "not found" means no.

        Anything else -- a credential the store refuses, the store unreachable -- is
        raised, because answering "absent" would turn a broken deployment into a
        stream of "your results have expired" pages.
        """
        blob = self._blob()
        try:
            blob.head(self._key(token, name))
        except blob.BlobNotFoundError:
            return False
        return True

    def purge(self, token: str) -> None:
        blob = self._blob()
        try:
            listing = blob.list_objects(prefix=f"{self.prefix}/{checked(token)}/")
            keys = [item.pathname for item in listing.blobs]
            if keys:
                blob.delete(keys)
        except Exception:                                     # noqa: BLE001
            return           # purge is best-effort; the database row is authoritative

    def url(self, token: str, name: str) -> str | None:
        """A link the browser can follow, so the bytes never pass through a function.

        A public blob has one already: `get_download_url` appends `?download=1` to
        the blob's own URL. A private blob needs a signed, short-lived URL, and only
        the JavaScript SDK can sign one (`issueSignedToken` + `presignUrl`), so the
        link is to the route that does: `api/blob-upload.js` answers
        GET `BLOB_DOWNLOAD_HANDLER` by checking the grant in the link and redirecting
        to the URL it signs for exactly that object. Who may read what is decided by
        the caller before it asks for this link (results.py serves only a finished
        job's stored exports); the grant carries that decision, and nothing else.
        """
        if config.BLOB_ACCESS != "public":
            grant = grants.sign(
                "download",
                pathname=self._key(token, name),
                access=config.BLOB_ACCESS,
                # Long enough to start the download, short enough that a copied link
                # is not a lasting handle on someone's data.
                valid_until=int((time.time() + config.DOWNLOAD_URL_TTL_SECONDS) * 1000),
            )
            return f"{config.BLOB_DOWNLOAD_HANDLER}?{urlencode({'grant': grant})}"
        blob = self._blob()
        try:
            return blob.get_download_url(blob.head(self._key(token, name)).url)
        except Exception:                                     # noqa: BLE001
            return None          # fall back to streaming through the app

    def authorize(self, token: str, name: str, size: int) -> dict:  # noqa: ARG002
        """Send the browser to the one endpoint that can mint it a token.

        Minting is `generateClientTokenFromReadWriteToken`, which exists only in the
        JavaScript SDK — `vercel.blob` in Python can use a client token but not issue
        one. Rather than hand a browser `BLOB_READ_WRITE_TOKEN`, which writes
        anything in the store, the browser is pointed at `api/blob-upload.js`, which
        mints a token only for a pathname this application signed a grant for
        (`uploads.upload_grant`, added by the route). The rules stay here; only the
        signing happens there.

        The pathname is returned so the browser has nothing to invent, and the
        signing route checks the one it presents against the grant anyway.
        """
        if not config.BLOB_UPLOAD_HANDLER:
            raise DirectUploadUnavailable(
                "No client-upload endpoint is configured for this deployment."
            )
        return {
            "strategy": "vercel-blob",
            "handler": config.BLOB_UPLOAD_HANDLER,
            "pathname": self._key(token, name),
            "access": config.BLOB_ACCESS,
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


def exists(token: str, name: str) -> bool:
    return backend().exists(token, name)


def is_local() -> bool:
    """True when this process can hand a stored file straight from disk."""
    return backend().name == "local"


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
