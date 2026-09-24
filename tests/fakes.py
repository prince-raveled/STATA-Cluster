"""Stand-ins for the parts of Vercel a test cannot reach.

`FakeBlob` is `vercel.blob` reduced to the calls MicroVerse makes, and `use_blob`
configures the application as Vercel runs it: results in Blob storage, and the
deployment root read-only, so a write that should not happen fails loudly.
"""
from __future__ import annotations

import errno
import pathlib
from types import SimpleNamespace

from app import config, storage

SECRET = "download-test-secret"


class FakeBlob:
    """`vercel.blob`, reduced to the calls MicroVerse makes, keyed by pathname."""

    class BlobNotFoundError(Exception):
        """What the SDK raises when the store answers 404."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = []

    def put(self, pathname, body, *, access, content_type=None, add_random_suffix=False,
            overwrite=False):
        assert access in ("private", "public")
        assert add_random_suffix is False and overwrite is True
        self.objects[pathname] = bytes(body)
        self.puts.append((pathname, access, content_type))

    def get(self, pathname, *, access):
        assert access in ("private", "public")
        if pathname not in self.objects:
            raise self.BlobNotFoundError(pathname)
        return SimpleNamespace(content=self.objects[pathname])

    def head(self, pathname):
        if pathname not in self.objects:
            raise self.BlobNotFoundError(pathname)
        return SimpleNamespace(
            pathname=pathname, size=len(self.objects[pathname]),
            url=f"https://teststore.public.blob.vercel-storage.com/{pathname}")

    @staticmethod
    def get_download_url(url):
        return url + "?download=1"

    def list_objects(self, prefix):
        return SimpleNamespace(blobs=[SimpleNamespace(pathname=key)
                                      for key in self.objects if key.startswith(prefix)])

    def delete(self, keys):
        for key in keys:
            self.objects.pop(key, None)


def use_blob(mp, fake, root: pathlib.Path, access="private"):
    """Point MicroVerse at `fake` as Vercel does, with `root` standing in for /var/task.

    Nothing may be created under `root`: `job_dir` raises the error Vercel raises,
    and any other attempt to make a directory there does too.
    """
    mp.setattr(config, "STORAGE_BACKEND", "blob")
    mp.setattr(config, "BLOB_ACCESS", access)
    mp.setattr(config, "WORKER_SECRET", SECRET)
    mp.setattr(config, "DATA_DIR", root / "data")
    mp.setattr(config, "JOBS_DIR", root / "data" / "jobs")
    mp.setattr(storage.BlobBackend, "_blob", staticmethod(lambda: fake))

    def read_only(*_args, **_kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    mp.setattr(config, "job_dir", read_only)
    real_mkdir = pathlib.Path.mkdir

    def guarded_mkdir(self, *args, **kwargs):
        if str(self).startswith(str(root)):
            read_only()
        return real_mkdir(self, *args, **kwargs)

    mp.setattr(pathlib.Path, "mkdir", guarded_mkdir)
    storage.reset()
