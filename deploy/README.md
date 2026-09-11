# Deploying MicroVerse

The image is built and exercised by CI on every push (`.github/workflows/ci.yml`), so a
green build means the container starts, serves a whole analysis and returns a bundle.

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
| `MICROVERSE_DATA` | `./data` | where jobs and results live |
| `MICROVERSE_RETENTION_DAYS` | `90` | SPEC §16.5 retention, purged on boot |
| `MICROVERSE_MAX_UPLOAD` | `67108864` | per-file upload ceiling in bytes |
| `MICROVERSE_MAX_CONCURRENT` | `min(4, cpus)` | simultaneous runs |
| `MICROVERSE_MAX_QUEUED` | `24` | runs allowed to wait for a slot |
| `MICROVERSE_RATE_REQUESTS` | `20` | uploads/runs per client per window |
| `MICROVERSE_RATE_WINDOW` | `300` | the window, in seconds |

## After deploying

Check `/healthz` — it reports the queue depth as well as liveness:

```json
{"status": "ok", "version": "1.0.0", "queue": {"running": 0, "waiting": 0,
 "max_concurrent": 2, "max_queued": 24}}
```

Then run `tests/reference/load_test.py` against the deployed URL to confirm the host's
CPU actually meets the O6 budget, which SPEC §20 asks to be verified on the real host
rather than assumed:

```bash
MICROVERSE_URL=https://your-app.fly.dev python tests/reference/load_test.py
```
