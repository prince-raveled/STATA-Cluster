"""Signed approvals that the Blob signing service checks without calling back.

`blob/api/blob-upload.js` holds the store credential and signs what MicroVerse
approved: an upload token for one pathname, a read URL for one object. It used to
learn what was approved by calling this application over HTTP, which needed the
application's public URL and a second shared secret configured on every deployment --
and on a preview deployment behind Vercel's login, a call from a function to its own
deployment's URL is stopped at the login wall before it reaches FastAPI. So the
approval now travels with the request: this module signs it, the service checks the
signature, and nothing crosses the network between the two.

The rules stay here. A grant carries every limit the service applies -- the pathname,
the size cap, the media types, the expiry -- and the service adds none of its own.

Both sides derive the same key: MICROVERSE_WORKER_SECRET when a deployment sets one,
otherwise a key derived from BLOB_READ_WRITE_TOKEN, which every deployment with a Blob
store already gives both services. A process with neither (development, Docker) has
no signing service to talk to and uses a random per-process key.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from . import config

#: Labels that keep the derived keys apart. Changing one invalidates every grant in
#: flight, and must be matched in blob/api/blob-upload.js.
_BASE_LABEL = b"microverse/key/v1"
_GRANT_LABEL = b"microverse/grant/v1"

#: Upload and download grants are not interchangeable.
USES = ("upload", "download")

_PROCESS_KEY = secrets.token_bytes(32)


def base_key() -> bytes:
    """The key shared with the signing service. Also signs upload tickets.

    Read per call, so every instance of a deployment derives the same key from the
    same environment -- a ticket issued by one instance verifies on another.
    """
    if config.WORKER_SECRET:
        return config.WORKER_SECRET.encode("utf-8")
    store = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
    if store:
        return hmac.new(store.encode("utf-8"), _BASE_LABEL, hashlib.sha256).digest()
    return _PROCESS_KEY


def _grant_key() -> bytes:
    return hmac.new(base_key(), _GRANT_LABEL, hashlib.sha256).digest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign(use: str, *, pathname: str, valid_until: int, **claims) -> str:
    """Approve one operation on one pathname until `valid_until` (milliseconds)."""
    if use not in USES:
        raise ValueError(f"unknown grant use {use!r}")
    payload = {"use": use, "pathname": pathname, "valid_until": int(valid_until), **claims}
    body = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(_grant_key(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64(signature)}"


def verify(raw: str, use: str) -> dict | None:
    """The grant's claims, or None if it is forged, for another use, or expired.

    The signing service performs the same checks in JavaScript; this is the reference
    the tests hold it to.
    """
    try:
        body, signature = str(raw or "").split(".")
        expected = hmac.new(_grant_key(), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(signature), expected):
            return None
        claims = json.loads(_unb64(body))
    except Exception:                                        # noqa: BLE001
        return None
    if not isinstance(claims, dict) or claims.get("use") != use:
        return None
    if not isinstance(claims.get("pathname"), str):
        return None
    if int(claims.get("valid_until", 0)) <= time.time() * 1000:
        return None
    return claims
