"""Job orchestration between the HTTP layer and the engine."""
from __future__ import annotations

import json
import traceback

import pandas as pd

from . import config, db
from .core.attribution import attribute
from .core.models import Specification
from .core.parsers import parse_abundance, parse_metadata, parse_taxonomy_map
from .core.report import build_zip, long_results_csv_gz, methods_paragraph, run_manifest
from .core.robustness import compute_robustness, verdict_sentence
from .core.runner import run_multiverse
from .core.validation import DatasetError, NotFoundError, validate_dataset
from .db import utcnow
from .limits import run_queue

#: How long a queued run waits for a slot before giving up and saying so.
QUEUE_WAIT_SECONDS = 900

DEMO_DATASETS = {
    "ibd_genus": {
        "title": "IBD vs control, genus level",
        "blurb": "120 samples (60/60), 380 genera, 24 taxa spiked at known effect sizes. "
                 "The rank fork collapses to one level, as in SPEC §19.",
        "shape": "380 taxa x 120 samples",
    },
    "gut_species": {
        "title": "Species-level gut cohort",
        "blurb": "80 samples, 300 species with full lineages, so fork 4 (taxonomic rank) "
                 "has two levels and the grid is twice the size.",
        "shape": "300 taxa x 80 samples",
    },
    "t2d_covariates": {
        "title": "T2D cohort with covariates",
        "blurb": "90 samples with age, BMI, sex, smoking status and read depth — built "
                 "for covariate mode, the Tierney et al. 2022 analysis.",
        "shape": "220 taxa x 90 samples",
    },
}


def build_dataset(abundance_bytes, abundance_name, metadata_bytes, metadata_name,
                  taxonomy_bytes=None, taxonomy_name="", group_column="", covariates=None):
    """Parse, attach taxonomy, and validate. Raises DatasetError / ParseError only."""
    metadata = parse_metadata(metadata_bytes, metadata_name)
    table = parse_abundance(abundance_bytes, abundance_name,
                            sample_ids=[str(i) for i in metadata.index])
    if taxonomy_bytes:
        mapping = parse_taxonomy_map(taxonomy_bytes, taxonomy_name)
        matched = table.attach_taxonomy(mapping)
        if matched == 0:
            raise DatasetError(
                "None of the feature IDs in the taxonomy file match the abundance table.",
                "Taxonomy IDs look like: " + ", ".join(list(mapping)[:4]) +
                " — table IDs look like: " + ", ".join(table.taxa[:4]) + ".",
            )
    return validate_dataset(table, metadata, group_column, covariates=covariates)


def load_demo(name: str):
    if name not in DEMO_DATASETS:
        # A demo that does not exist is a missing resource, not a malformed request.
        raise NotFoundError(f"There is no demo dataset called '{name}'.",
                            "Available: " + ", ".join(DEMO_DATASETS))
    folder = config.EXAMPLES_DIR
    return build_dataset(
        (folder / f"{name}_abundance.tsv").read_bytes(), f"{name}_abundance.tsv",
        (folder / f"{name}_metadata.tsv").read_bytes(), f"{name}_metadata.tsv",
        (folder / f"{name}_taxonomy.tsv").read_bytes(), f"{name}_taxonomy.tsv",
        group_column="group",
    )


def create_job(dataset, dataset_name: str) -> str:
    token = db.new_token()
    with db.session() as session:
        session.add(db.Job(
            token=token,
            status="uploaded",
            dataset_name=dataset_name,
            n_taxa=dataset.n_taxa,
            n_samples=dataset.n_samples,
            summary_json=json.dumps(dataset.summary(), default=str),
        ))
        session.commit()
    db.save_payload(token, "dataset", dataset)
    return token


def parse_declared(form: dict):
    """Build the user's declared pipeline from the configure form, if they gave one."""
    if not form.get("declare"):
        return None
    rarefaction = form.get("declared_rarefaction") or "none"
    seed = None if rarefaction == "none" else 1
    try:
        return Specification(
            rarefaction=rarefaction,
            rare_seed=seed,
            rank=form.get("declared_rank") or "input",
            prev_filter=float(form.get("declared_prev_filter") or 0.10),
            transform=form.get("declared_transform") or "tss",
            method=form.get("declared_method") or "wilcoxon",
            fdr_method=form.get("declared_fdr_method") or "bh",
            fdr_threshold=float(form.get("declared_fdr_threshold") or 0.05),
            covariates=(),
        )
    except (TypeError, ValueError):
        return None


def execute(token: str, mode: str, declared=None, covariates=()) -> None:
    """Run the multiverse and persist everything the results pages need.

    Runs in Starlette's background threadpool, so it must never raise into the caller.
    """
    # Wait for a slot rather than piling on. Unbounded concurrency made six
    # simultaneous runs take 63 s each against ~10 s alone, without finishing any of
    # them sooner; bounding it makes latency predictable. Measured in
    # tests/reference/load_test.py.
    db.update_job(token, status="running", progress=0.0, mode=mode,
                  message="Queued — waiting for a free analysis slot",
                  error="", error_hint="")
    if not run_queue.acquire(timeout=QUEUE_WAIT_SECONDS):
        db.update_job(
            token, status="error", finished_at=utcnow(),
            error="The server is busy and this run could not start.",
            error_hint=f"Every analysis slot stayed occupied for "
                       f"{QUEUE_WAIT_SECONDS // 60} minutes. Your upload is still here — "
                       f"reload this page to try again.",
        )
        return

    db.update_job(token, status="running", started_at=utcnow(),
                  progress=0.0, message="Starting", mode=mode, error="", error_hint="")
    try:
        dataset = db.load_payload(token, "dataset")
        if dataset is None:
            raise DatasetError("This job's dataset has expired or was never stored.",
                               f"Results are kept for {config.RETENTION_DAYS} days.")

        def progress(fraction: float, message: str):
            db.update_job(token, progress=fraction, message=message)

        run = run_multiverse(
            dataset,
            mode=mode,
            covariate_columns=list(covariates) if mode == "covariate" else (),
            declared=declared,
            progress=progress,
        )
        progress(0.95, "Computing robustness tiers")
        summary = compute_robustness(run)
        progress(0.97, "Attributing variance to the forks")
        attribution = attribute(run)

        db.save_payload(token, "run", run)
        db.save_payload(token, "summary", summary)
        db.save_payload(token, "attribution", attribution)

        manifest = run_manifest(run, summary, attribution)
        (config.job_dir(token) / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        (config.job_dir(token) / "methods.txt").write_text(
            methods_paragraph(run, summary, attribution), encoding="utf-8"
        )
        # Compressed once and reused: this is the most expensive export by far.
        long_csv_gz = long_results_csv_gz(run)
        (config.job_dir(token) / "results_long.csv.gz").write_bytes(long_csv_gz)
        (config.job_dir(token) / "bundle.zip").write_bytes(
            build_zip(run, summary, attribution, long_csv_gz=long_csv_gz)
        )

        db.update_job(
            token, status="done", progress=1.0, message="Complete",
            finished_at=utcnow(), n_specs=run.grid_report.n_valid,
            runtime_seconds=run.runtime_seconds,
            summary_json=json.dumps(
                {**run.dataset_summary,
                 "verdict": verdict_sentence(run, summary),
                 "tiers": summary.tier_counts,
                 "grid": manifest["grid"]},
                default=str,
            ),
        )
    except DatasetError as exc:
        db.update_job(token, status="error", error=exc.message, error_hint=exc.hint,
                      finished_at=utcnow())
    except Exception as exc:  # never surface a traceback to the user (§8)
        db.update_job(
            token, status="error",
            error=f"The run failed: {type(exc).__name__}: {exc}",
            error_hint="This is a bug in MicroVerse, not in your data. The details have "
                       "been written to the server log.",
            finished_at=utcnow(),
        )
        print(f"[microverse] job {token} failed\n{traceback.format_exc()}")
    finally:
        run_queue.release()


def load_results(token: str):
    """Return (job, run, summary, attribution) or (job, None, None, None)."""
    job = db.get_job(token)
    if job is None or job.status != "done":
        return job, None, None, None
    return (job, db.load_payload(token, "run"), db.load_payload(token, "summary"),
            db.load_payload(token, "attribution"))


def taxon_options(summary, limit: int = 400) -> list:
    """Taxa for the curve selector, best tier first."""
    table = summary.table.head(limit)
    return [
        {"id": int(row.taxon_id), "label": row.label, "tier": row.robustness_tier,
         "frac_significant": float(row.frac_significant),
         "median_effect": float(row.median_effect)}
        for row in table.itertuples()
    ]


def table_records(summary) -> list:
    frame: pd.DataFrame = summary.table
    records = []
    for row in frame.itertuples():
        records.append({
            "taxon_id": int(row.taxon_id),
            "label": row.label,
            "taxon": row.taxon,
            "rank": row.rank,
            "tier": row.robustness_tier,
            "n_specs_tested": int(row.n_specs_tested),
            "frac_tested": round(float(row.frac_tested), 4),
            "frac_significant": round(float(row.frac_significant), 4),
            "frac_nominal": round(float(row.frac_nominal), 4),
            "sign_consistency": round(float(row.sign_consistency), 4),
            "median_effect": round(float(row.median_effect), 4),
            "iqr_low": round(float(row.iqr_low), 4),
            "iqr_high": round(float(row.iqr_high), 4),
            "direction": row.direction,
        })
    return records

def taxon_neighbours(summary, taxon_id: int) -> dict:
    """The taxa either side of this one in the ranked table, for previous/next links."""
    ids = list(summary.table["taxon_id"])
    labels = list(summary.table["label"])
    try:
        position = ids.index(taxon_id)
    except ValueError:
        return {}
    out = {}
    if position > 0:
        out["previous"] = {"id": int(ids[position - 1]), "label": labels[position - 1]}
    if position + 1 < len(ids):
        out["next"] = {"id": int(ids[position + 1]), "label": labels[position + 1]}
    return out
