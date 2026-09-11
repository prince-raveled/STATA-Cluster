#!/bin/sh
# Container entrypoint — validate the environment, then hand PID 1 to uvicorn.
#
# Startup validation exists because the alternative is a container that reports healthy
# and then fails on the first upload. Everything checked here is something that has
# actually gone wrong in deployment: an unwritable data volume, a missing demo dataset,
# an engine whose imports do not resolve.
set -eu

PORT="${PORT:-8000}"
DATA_DIR="${MICROVERSE_DATA:-/data}"
WORKERS="${WEB_CONCURRENCY:-1}"

fail() {
    echo "microverse: startup check failed — $1" >&2
    exit 1
}

# 1. The data directory must exist and be writable by the running user. A read-only
#    mount is the single most common deployment mistake and produces a confusing
#    SQLite error several seconds after the first request rather than at boot.
[ -d "$DATA_DIR" ] || fail "MICROVERSE_DATA=$DATA_DIR does not exist"
if ! touch "$DATA_DIR/.write-test" 2>/dev/null; then
    fail "MICROVERSE_DATA=$DATA_DIR is not writable by uid $(id -u)"
fi
rm -f "$DATA_DIR/.write-test"

# 2. The engine must import and the demo data must be present. This catches a broken
#    layer copy before the port opens, so an orchestrator sees a failed start rather
#    than a healthy container serving errors.
python - <<'PY' || fail "the application did not import cleanly"
import sys
from app.main import app
from app.core.runner import run_multiverse          # noqa: F401  - import check
from app.core.evidence import TIER_REPLICATION
assert app.routes, "no routes registered"
assert TIER_REPLICATION, "validation evidence missing from the image"
print(f"microverse: engine imports cleanly on Python {sys.version.split()[0]}")
PY

python - <<'PY' || fail "the bundled demo datasets are missing or unreadable"
import pathlib
examples = pathlib.Path("examples")
tables = sorted(examples.glob("*_abundance.tsv"))
assert tables, "no demo abundance tables found in examples/"
print(f"microverse: {len(tables)} demo datasets present")
PY

echo "microverse: starting on :${PORT} with ${WORKERS} worker(s), data at ${DATA_DIR}"

# exec so uvicorn replaces this shell as PID 1 and receives SIGTERM directly.
# --timeout-graceful-shutdown gives an in-flight analysis a window to finish rather
# than being cut off mid-run.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --timeout-graceful-shutdown 30 \
    --proxy-headers \
    --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}" \
    --access-log
