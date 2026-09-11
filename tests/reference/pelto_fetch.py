"""Download Pelto et al.'s replication archive and unpack the curated data.

SPEC §23.3 points at `github.com/jepelt/DAA_replicability`, archived at Zenodo
10.5281/zenodo.15047338 (CC-BY-4.0). The archive is ~48 MB and holds, among the
analysis scripts, the curated cohorts used in the paper. Nothing from it is committed:
it lands in the git-ignored `data/` cache, like the MicrobiomeHD archives.

    .venv/Scripts/python tests/reference/pelto_fetch.py

Then export to TSV with R (base R only — the .rds is a plain list, not a Bioconductor
object) and run the study:

    Rscript tests/reference/pelto_export.R data/pelto/data_171023.rds data/pelto/export
    .venv/Scripts/python tests/reference/pelto_study.py
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "data", "pelto")
RECORD = "15047338"
ARCHIVE = "DAA_replicability-v1.0.0.zip"
URL = (f"https://zenodo.org/api/records/{RECORD}/files/"
       f"jepelt/{ARCHIVE}/content")

#: The curated data the analysis scripts load. Despite the extension these are `save()`
#: images, so R reads them with load(), and each holds a plain list of
#: list(meta = data.frame, counts = matrix[samples, taxa]).
WANTED = ("data_171023.rds", "data_meta_041023.rds")


def download(destination: str) -> None:
    request = urllib.request.Request(URL, headers={"User-Agent": "microverse"})
    with urllib.request.urlopen(request, timeout=600) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(destination + ".part", "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
    print()
    os.replace(destination + ".part", destination)


def main() -> int:
    os.makedirs(CACHE, exist_ok=True)
    archive = os.path.join(CACHE, ARCHIVE)

    if not os.path.exists(archive):
        print(f"downloading Zenodo record {RECORD} ({ARCHIVE})")
        try:
            download(archive)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"SKIPPED — could not reach Zenodo: {exc}")
            return 0
    print(f"archive: {archive} ({os.path.getsize(archive) / 1e6:.1f} MB)")

    with zipfile.ZipFile(archive) as bundle:
        members = {os.path.basename(name): name for name in bundle.namelist()
                   if os.path.basename(name) in WANTED}
        missing = [name for name in WANTED if name not in members]
        if missing:
            print(f"archive does not contain {', '.join(missing)} — layout changed?")
            return 1
        for name, member in members.items():
            target = os.path.join(CACHE, name)
            if os.path.exists(target):
                print(f"  {name} already unpacked")
                continue
            with bundle.open(member) as source, open(target, "wb") as handle:
                handle.write(source.read())
            print(f"  wrote {name} ({os.path.getsize(target) / 1e6:.1f} MB)")

    print("\nnext:")
    print("  Rscript tests/reference/pelto_export.R "
          "data/pelto/data_171023.rds data/pelto/export")
    print("  python tests/reference/pelto_study.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
