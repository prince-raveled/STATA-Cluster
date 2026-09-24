# Deploying MicroVerse

The public instance runs on Vercel (below). Everywhere else MicroVerse is one process
with a disk: the Docker image, Fly.io or Render. The image is built and exercised by CI
on every push (`.github/workflows/ci.yml`), so a green build means the container starts,
serves a whole analysis and returns a bundle.

## Vercel (the public instance)

<https://stata-cluster.vercel.app>, on the free plan, from the `vercel-migration`
branch. `vercel.json` declares two services behind one domain:

| Service | What it is |
|---|---|
| `microverse` | the FastAPI app as a Python function (`app.main:app`, `maxDuration` 300 s) |
| `blob_upload` | `blob/api/blob-upload.js`: signs browser upload tokens (`POST /api/blob-upload`) and short-lived download URLs (`GET /api/blob-download`) — the only things the Python Blob SDK cannot do |

A run is published to Vercel Queues and executed by the subscriber declared under
`[[tool.vercel.subscribers]]` in `pyproject.toml` (`app/queue_worker.py`). The build runs
`python examples/make_examples.py`, because the demo datasets are generated, not
committed. The Python version comes from `.python-version`.

On Vercel storage defaults to `blob` and jobs to `queue` (see `app/config.py`). The
project needs:

| Variable | Set by | Meaning |
|---|---|---|
| `MICROVERSE_DB` | you | PostgreSQL URL (Neon); SQLite is refused on Vercel, with a message saying so |
| `BLOB_READ_WRITE_TOKEN` | the Blob store, when attached | the store credential; never leaves the server |
| `MICROVERSE_WORKER_SECRET` | you | long random value shared by the app and the signing service; also signs upload tickets |
| `MICROVERSE_BACKEND_URL` | you | where the signing service reaches the app: the production URL |

Without the last two the signing service answers 500 and names what is missing, and
direct upload and the large downloads do not work. Queue credentials are provided by
the platform.

Releasing: a push to `vercel-migration` builds a **preview**, which sits behind Vercel
login; production changes only when a deployment is promoted in the dashboard.
Environment variables apply to deployments made after they are set, so redeploy after
changing one. A preview's `MICROVERSE_BACKEND_URL` still points at production, so test
uploads and large downloads on production after promoting.

## Sizing

MicroVerse is compute-hungry in bursts rather than concurrent-request-hungry. From
`docs/benchmark.json`, Quick mode at the §8 ceiling of 1,500 taxa takes about 25 s of
CPU and holds roughly 300 MB while assembling exports.

| Setting | Why |
|---|---|
| **2 vCPU, 2 GB** minimum | one run at the taxon ceiling needs ~300 MB and saturates a core |
| `MICROVERSE_MAX_CONCURRENT` = vCPU count | beyond that, runs only contend — measured at 6x slowdown when unbounded |
| A persistent volume at `/data` | jobs and results live there for `MICROVERSE_RETENTION_DAYS` (90) |

Free tiers with 512 MB will fail on large tables. That is a hosting choice, not a bug,
and the §8 limits are what keep it bounded.

## Fly.io

```bash
fly launch --no-deploy          # reads fly.toml
fly volumes create microverse_data --size 10
fly deploy
fly open /healthz
```

## Render

New → Blueprint → point at the repository. `render.yaml` declares the service, the
health check and the disk.

## Anywhere with Docker

```bash
docker build -t microverse .
docker run -d -p 8000:8000 -v microverse_data:/data microverse
```

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `MICROVERSE_DATA` | `./data` | where jobs and results live (local storage) |
| `MICROVERSE_DB` | SQLite in `MICROVERSE_DATA` | job database; any SQLAlchemy URL |
| `MICROVERSE_STORAGE` | `local` (`blob` on Vercel) | where a job's bytes live |
| `MICROVERSE_JOBS` | `inline` (`queue` on Vercel) | how a run is started |
| `MICROVERSE_RETENTION_DAYS` | `90` | SPEC §16.5 retention, purged on boot |
| `MICROVERSE_MAX_UPLOAD` | `67108864` | per-file upload ceiling in bytes |
| `MICROVERSE_MAX_CONCURRENT` | `min(4, cpus)` | simultaneous runs |
| `MICROVERSE_MAX_QUEUED` | `24` | runs allowed to wait for a slot |
| `MICROVERSE_RATE_REQUESTS` | `20` | uploads/runs per client per window |
| `MICROVERSE_RATE_WINDOW` | `300` | the window, in seconds |

## After deploying

Check `/healthz`. Besides liveness it reports the queue depth and which backends and
interpreter the process is actually running with — names only, never a credential:

```json
{"status": "ok", "version": "1.0.0", "queue": {"running": 0, "waiting": 0,
 "max_concurrent": 2, "max_queued": 24},
 "config": {"storage": "blob", "jobs": "queue", "database": "postgresql+psycopg",
            "on_vercel": true, "python": "3.14.7"}}
```

Then run `tests/reference/load_test.py` against the deployed URL to confirm the host's
CPU actually meets the O6 budget, which SPEC §20 asks to be verified on the real host
rather than assumed:

```bash
MICROVERSE_URL=https://your-app.fly.dev python tests/reference/load_test.py
```
