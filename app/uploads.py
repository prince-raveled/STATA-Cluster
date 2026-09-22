"""Uploading a file without sending it through the application.

A host that caps a request body well below MicroVerse's upload limit cannot receive a
120-sample BIOM table the ordinary way. The file has to go straight from the browser to
storage, which means the browser needs permission to write one specific object — and
must never be given the credential that would let it write any object.

The permission is a *ticket*: a signed statement that this browser may stage these
named files, of these sizes, for the next few minutes, under a key the server chose.
It is signed rather than stored, so there is no table of pending uploads to clean up,
and the thing the browser cannot do is mint one for a key it picked itself.

What the ticket deliberately does not do is decide whether the data is any good. Every
upload still reaches `services.build_dataset` and the same parsers, the same validation
and the same error messages as a form post. The only question answered here is "may
these bytes be written", never "are these bytes a usable dataset".
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from . import config
from .core.validation import DatasetError

#: The form's three file inputs. A ticket may name these and nothing else.
FIELDS = ("abundance", "metadata", "taxonomy")
REQUIRED_FIELDS = ("abundance", "metadata")

#: Mirrors the `accept` attributes on the upload form. This is a gate on what may be
#: staged, not a claim about what will parse — the parsers remain the authority.
ALLOWED_SUFFIXES = (".csv", ".tsv", ".txt", ".biom", ".qza", ".gz")

#: How long a browser has to finish uploading. Generous enough for 64 MB on a slow
#: connection, short enough that a leaked ticket stops working quickly.
TICKET_TTL_SECONDS = 1800

#: Signing key. A deployment that sets a worker secret gets tickets that survive a
#: restart; one that does not gets a per-process key, which is correct for development
#: and means an abandoned ticket cannot be replayed against a new process.
_PROCESS_KEY = secrets.token_urlsafe(32)


def _signing_key() -> bytes:
    return (config.WORKER_SECRET or _PROCESS_KEY).encode("utf-8")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def check_name(field: str, filename: str) -> None:
    """Reject a filename before anything is staged under it."""
    name = (filename or "").strip()
    if not name:
        raise DatasetError(
            f"No file was chosen for {field}.",
            "Supported: " + ", ".join(ALLOWED_SUFFIXES) + ".",
        )
    if not name.lower().endswith(ALLOWED_SUFFIXES):
        raise DatasetError(
            f"'{name}' is not a file type MicroVerse reads.",
            "Supported: " + ", ".join(ALLOWED_SUFFIXES) + ". The abundance table may "
            "also be a BIOM or QIIME 2 artifact.",
        )


def check_size(filename: str, size: int) -> None:
    """The application's own limit, applied before a byte is written.

    Checking the declared size here is what keeps an oversized file from being staged
    at all; the staging route checks the real length again, because a declaration is
    not evidence.
    """
    if size < 0:
        raise DatasetError(f"'{filename}' reports a negative size.",
                           "The upload was not completed. Try again.")
    if size > config.MAX_UPLOAD_BYTES:
        raise DatasetError(
            f"'{filename}' is {size / 1e6:.0f} MB, above the "
            f"{config.MAX_UPLOAD_BYTES / 1e6:.0f} MB upload limit.",
            "Collapse to genus before uploading, or run MicroVerse locally with Docker "
            "where the limit does not apply.",
        )


def validate_request(files: dict) -> dict:
    """Check a browser's declared file list. Returns the cleaned version.

    `files` maps field name to {"filename": str, "size": int}.
    """
    cleaned = {}
    for field in FIELDS:
        entry = files.get(field)
        if not entry:
            continue
        filename = str(entry.get("filename") or "").strip()
        if not filename:
            continue
        size = int(entry.get("size") or 0)
        check_name(field, filename)
        check_size(filename, size)
        cleaned[field] = {"filename": filename, "size": size}

    for field in REQUIRED_FIELDS:
        if field not in cleaned:
            raise DatasetError(
                f"No {'abundance table' if field == 'abundance' else 'sample metadata'} "
                "was uploaded.",
                "MicroVerse needs an abundance table and a metadata table with a binary "
                "grouping column.",
            )
    return cleaned


def staging_token(ticket_id: str) -> str:
    """The storage token staged bytes live under.

    Derived from the ticket, never from anything the browser sent, and shaped like a
    job token so the existing path check applies to it unchanged.
    """
    return ticket_id


def issue(files: dict) -> str:
    """Mint a signed ticket for an already-validated file list."""
    payload = {
        "id": secrets.token_hex(12),
        "exp": int(time.time()) + TICKET_TTL_SECONDS,
        "files": {f: {"filename": e["filename"], "size": e["size"]}
                  for f, e in files.items()},
    }
    body = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    signature = hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64(signature)}"


def verify(raw: str) -> dict | None:
    """Return the ticket's payload, or None if it is forged, malformed or expired."""
    try:
        body, signature = str(raw or "").split(".", 1)
        expected = hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(signature), expected):
            return None
        payload = json.loads(_unb64(body))
    except Exception:                                        # noqa: BLE001
        return None

    if not isinstance(payload, dict) or int(payload.get("exp", 0)) < time.time():
        return None
    if not str(payload.get("id", "")).isalnum():
        return None
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) - set(FIELDS):
        return None
    return payload
