"""Write real BIOM and QIIME 2 files with the official libraries, as test fixtures.

The in-tree readers (`app/core/parsers/biom.py`) exist because `biom-format` has no
wheel on the Python the server runs (SPEC §24.1 B1). Testing them against strings we
wrote ourselves would only prove we are self-consistent, so the fixtures are produced
by `biom-format` itself, in a separate Python 3.10 environment where it does install:

    .venv310/Scripts/pip install "scikit-bio>=0.7.1"     # pulls in biom-format
    .venv310/Scripts/python tests/reference/make_format_fixtures.py

The files land in `tests/fixtures/` and are then read by `tests/test_real_formats.py`
in the normal suite. Re-run this only to regenerate them.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(os.path.dirname(HERE), "fixtures")

LINEAGES = [
    ["k__Bacteria", "p__Firmicutes", "c__Clostridia", "o__Clostridiales",
     "f__Lachnospiraceae", "g__Blautia", "s__wexlerae"],
    ["k__Bacteria", "p__Firmicutes", "c__Clostridia", "o__Clostridiales",
     "f__Ruminococcaceae", "g__Faecalibacterium", "s__prausnitzii"],
    ["k__Bacteria", "p__Bacteroidetes", "c__Bacteroidia", "o__Bacteroidales",
     "f__Bacteroidaceae", "g__Bacteroides", "s__fragilis"],
    ["k__Bacteria", "p__Proteobacteria", "c__Gammaproteobacteria",
     "o__Enterobacterales", "f__Enterobacteriaceae", "g__Escherichia", "s__coli"],
    ["k__Bacteria", "p__Actinobacteria", "c__Actinobacteria", "o__Bifidobacteriales",
     "f__Bifidobacteriaceae", "g__Bifidobacterium", "s__longum"],
]


def build_table(n_taxa=24, n_samples=20, seed=99):
    """A sparse count table with lineages, shaped like a real feature table."""
    from biom.table import Table

    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(3, 0.02, size=(n_taxa, n_samples)).astype(float)
    counts[rng.random(counts.shape) < 0.35] = 0.0
    counts[0] += 500                     # guarantee a prevalent taxon
    counts[:, 0] += 5                    # and no empty sample

    observation_ids = [f"ASV{i:03d}" for i in range(n_taxa)]
    sample_ids = [f"S{j:03d}" for j in range(n_samples)]
    observation_metadata = [
        {"taxonomy": LINEAGES[i % len(LINEAGES)][: 5 + (i % 3)]} for i in range(n_taxa)
    ]
    sample_metadata = [
        {"group": "control" if j < n_samples // 2 else "disease"} for j in range(n_samples)
    ]
    table = Table(counts, observation_ids, sample_ids,
                  observation_metadata, sample_metadata)
    return table, counts, observation_ids, sample_ids


def main() -> None:
    import biom
    from biom.util import biom_open

    os.makedirs(FIXTURES, exist_ok=True)
    table, counts, observation_ids, sample_ids = build_table()
    print(f"biom-format {biom.__version__} on Python {sys.version.split()[0]}")
    print(f"table: {len(observation_ids)} observations x {len(sample_ids)} samples, "
          f"{(counts == 0).mean():.0%} zeros")

    # --- BIOM v1 (JSON), written by biom-format ---------------------------
    v1_path = os.path.join(FIXTURES, "real_table_v1.biom")
    with open(v1_path, "w", encoding="utf-8") as handle:
        handle.write(table.to_json(generated_by="microverse-fixture"))
    print(f"  wrote {os.path.relpath(v1_path)}  ({os.path.getsize(v1_path):,} bytes)")

    # --- BIOM v2 (HDF5), written by biom-format ---------------------------
    v2_path = os.path.join(FIXTURES, "real_table_v2.biom")
    with biom_open(v2_path, "w") as handle:
        table.to_hdf5(handle, generated_by="microverse-fixture")
    print(f"  wrote {os.path.relpath(v2_path)}  ({os.path.getsize(v2_path):,} bytes)")

    # --- QIIME 2 artifact: the real on-disk layout -------------------------
    # A .qza is a zip of <uuid>/{VERSION, metadata.yaml, data/<payload>}.
    artifact_id = str(uuid.uuid4())
    qza_path = os.path.join(FIXTURES, "real_table.qza")
    with zipfile.ZipFile(qza_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{artifact_id}/VERSION",
            "QIIME 2\narchive: 5\nframework: 2024.2.0\n",
        )
        archive.writestr(
            f"{artifact_id}/metadata.yaml",
            f"uuid: {artifact_id}\ntype: FeatureTable[Frequency]\nformat: BIOMV210DirFmt\n",
        )
        archive.write(v2_path, f"{artifact_id}/data/feature-table.biom")
    print(f"  wrote {os.path.relpath(qza_path)}  ({os.path.getsize(qza_path):,} bytes)")

    # --- the expected values, so the reader can be checked without biom ----
    expected = {
        "n_taxa": int(counts.shape[0]),
        "n_samples": int(counts.shape[1]),
        "observation_ids": observation_ids,
        "sample_ids": sample_ids,
        "column_sums": [float(v) for v in counts.sum(axis=0)],
        "row_sums": [float(v) for v in counts.sum(axis=1)],
        "total": float(counts.sum()),
        "n_zeros": int((counts == 0).sum()),
        "lineage_of_first": LINEAGES[0][:5],
        "qza_uuid": artifact_id,
    }
    expected_path = os.path.join(FIXTURES, "real_table_expected.json")
    with open(expected_path, "w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2)
    print(f"  wrote {os.path.relpath(expected_path)}")

    # A plain TSV export too, so the tabular reader is checked against the same data.
    tsv_path = os.path.join(FIXTURES, "real_table.tsv")
    with open(tsv_path, "w", encoding="utf-8") as handle:
        handle.write(table.to_tsv(header_key="taxonomy", header_value="taxonomy"))
    print(f"  wrote {os.path.relpath(tsv_path)}  ({os.path.getsize(tsv_path):,} bytes)")

    shutil.rmtree(os.path.join(FIXTURES, "__pycache__"), ignore_errors=True)


if __name__ == "__main__":
    main()
