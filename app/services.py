"""Job orchestration between the HTTP layer and the engine."""
from __future__ import annotations

import json
import traceback
from functools import lru_cache

import pandas as pd

from . import config, db, storage
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
                 "No taxonomy file is supplied, so the taxonomic-rank choice has "
                 "only one level.",
        "shape": "380 taxa x 120 samples",
    },
    "gut_species": {
        "title": "Species-level gut cohort",
        "blurb": "80 samples and 300 species with full lineages, so the analysis can be "
                 "repeated at genus level as well — twice as many analyses.",
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


@lru_cache(maxsize=1)
def demo_grid_sizes() -> dict:
    """How many valid specifications each demo actually produces, per mode.

    The landing page used to promise "three thousand defensible answers" while the
    default demo produced 1,596 and the graphic beside the headline said so. Numbers
    shown to a reader are derived from the same enumeration the run uses, so the copy
    cannot drift away from the product again. Enumeration is pure combinatorics —
    roughly 0.05s per demo — and the result is cached for the process.
    """
    from .core.grid import capabilities_for, enumerate_grid
    from .core.preprocess import MatrixBuilder

    sizes = {}
    for name in DEMO_DATASETS:
        try:
            dataset = load_demo(name)
            builder = MatrixBuilder(dataset)
            capabilities = capabilities_for(dataset)
            per_mode = {}
            for mode in ("quick", "full"):
                _, report = enumerate_grid(
                    builder, mode=mode, capabilities=capabilities,
                    covariate_columns=dataset.covariate_columns)
                per_mode[mode] = int(report.n_valid)
            sizes[name] = per_mode
        except Exception:                                    # noqa: BLE001
            continue          # a demo failing to enumerate must not break the homepage
    if not sizes:
        return {}
    every = [v for per_mode in sizes.values() for v in per_mode.values()]
    sizes["_range"] = {"low": min(every), "high": max(every)}
    sizes["_default"] = sizes.get("ibd_genus", {}).get("quick", min(every))
    return sizes


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
    # Who bounds concurrency depends on who called. Under the inline backend this
    # process is the only one there is, so the semaphore is the limit and a run waits
    # for a slot. Under the queue backend the dispatcher has already decided this run
    # may proceed — its cap is global, across every instance, which a per-process
    # semaphore cannot be. Taking the local one there would be worse than redundant:
    # a busy instance would fail a job the queue had every intention of running.
    local_slot = config.JOB_BACKEND != "queue"

    db.update_job(token, status="running", progress=0.0, mode=mode,
                  message="Queued — waiting for a free analysis slot",
                  error="", error_hint="")
    if local_slot and not run_queue.acquire(timeout=QUEUE_WAIT_SECONDS):
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
        storage.put_text(token, "manifest.json",
                         json.dumps(manifest, indent=2, default=str))
        storage.put_text(token, "methods.txt",
                         methods_paragraph(run, summary, attribution))
        # Compressed once and reused: this is the most expensive export by far.
        long_csv_gz = long_results_csv_gz(run)
        storage.put_bytes(token, "results_long.csv.gz", long_csv_gz)
        storage.put_bytes(token, "bundle.zip",
                          build_zip(run, summary, attribution, long_csv_gz=long_csv_gz))

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
        if local_slot:
            run_queue.release()


def load_results(token: str):
    """Return (job, run, summary, attribution) or (job, None, None, None)."""
    job = db.get_job(token)
    if job is None or job.status != "done":
        return job, None, None, None
    return (job, db.load_payload(token, "run"), db.load_payload(token, "summary"),
            db.load_payload(token, "attribution"))


#: The curve selector holds one <option> per taxon. This is a DOM-size bound, not an
#: editorial one — above it the page gets slow to build. When it bites, the interface
#: says so and points at the table, which is always complete. It must never look like
#: the run only found this many taxa.
TAXON_OPTION_LIMIT = 2000


def json_safe(value):
    """Recursively replace values JSON cannot carry with None.

    The robustness table now has a row for every taxon, including those no
    specification could test, and their measurements are genuinely absent. pandas
    spells absent as NaN; JSON has no spelling for it at all, and both Starlette and
    Jinja's tojson refuse to emit one. Anywhere the frame becomes JSON has to make
    that conversion, so it is written once here rather than at each call site --
    there are four, and the one that was missed answered 500.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    return value


def _number(value, digits: int | None = None):
    """A float the browser can be given, or None where there is no number.

    These records are handed to the page as JSON, and JSON has no NaN. A taxon that
    no specification could test has no median effect and no significance rate, and
    float("nan") for either one makes the whole payload unserialisable -- Starlette
    and Jinja's tojson both refuse it -- so the results table 500s over a single
    absent value. None is also the truthful answer: 0.0 would claim an effect of
    zero was measured, when nothing was measured at all.
    """
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, digits) if digits is not None else number


def taxon_options(summary, limit: int = TAXON_OPTION_LIMIT) -> list:
    """Taxa for the curve selector, best tier first."""
    table = summary.table.head(limit)
    return [
        {"id": int(row.taxon_id), "label": row.label, "tier": row.robustness_tier,
         "frac_significant": _number(row.frac_significant),
         "median_effect": _number(row.median_effect)}
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
            "frac_tested": _number(row.frac_tested, 4),
            "frac_significant": _number(row.frac_significant, 4),
            "frac_nominal": _number(row.frac_nominal, 4),
            "sign_consistency": _number(row.sign_consistency, 4),
            "median_effect": _number(row.median_effect, 4),
            "iqr_low": _number(row.iqr_low, 4),
            "iqr_high": _number(row.iqr_high, 4),
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
