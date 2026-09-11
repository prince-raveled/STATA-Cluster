"""Shared fixtures. Datasets are generated, not committed, so the suite is standalone."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.parsers.base import AbundanceTable  # noqa: E402
from app.core.validation import validate_dataset  # noqa: E402


def make_counts(n_taxa=60, n_per_group=20, n_differential=8, seed=11):
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    groups = np.array([0] * n_per_group + [1] * n_per_group)
    base = np.sort(rng.lognormal(0.0, 1.3, size=n_taxa))[::-1]
    base = base / base.sum()
    differential = rng.choice(min(n_taxa, 25), size=n_differential, replace=False)
    spike = np.zeros(n_taxa)
    magnitudes = rng.uniform(1.4, 2.4, n_differential)
    signs = rng.choice([-1, 1], n_differential)
    spike[differential] = magnitudes * signs
    mean_b = base * 2.0 ** spike
    mean_b /= mean_b.sum()
    # The ground truth for a relative-abundance estimator is the fold change that
    # survives closure, not the nominal spike: pushing a few abundant taxa down
    # rescales every other taxon's share. This is the compositional problem itself,
    # and `harmonized_effect` targets the post-closure quantity.
    effect = np.log2(mean_b / base)
    libraries = np.clip(np.rint(rng.lognormal(9.3, 0.4, n)).astype(int), 1500, 200_000)
    counts = np.zeros((n_taxa, n), dtype=int)
    for j in range(n):
        composition = rng.dirichlet(np.maximum((mean_b if groups[j] else base) * 400.0, 1e-4))
        counts[:, j] = rng.multinomial(libraries[j], composition)
    return counts, groups, differential, effect


@pytest.fixture(scope="session")
def synthetic():
    counts, groups, differential, effect = make_counts()
    taxa = [f"Taxon_{i:03d}" for i in range(counts.shape[0])]
    samples = [f"S{j:03d}" for j in range(counts.shape[1])]
    table = AbundanceTable(
        counts=pd.DataFrame(counts, index=taxa, columns=samples),
        source_format="test",
        value_type="counts",
    )
    metadata = pd.DataFrame(
        {"group": np.where(groups == 1, "disease", "control"),
         "age": np.linspace(30, 70, len(samples)).round(0),
         "sex": ["F", "M"] * (len(samples) // 2)},
        index=samples,
    )
    metadata.index.name = "sample_id"
    return {
        "table": table,
        "metadata": metadata,
        "groups": groups,
        "differential": {taxa[i] for i in differential},
        "effect": effect,
        "taxa": taxa,
    }


@pytest.fixture(scope="session")
def dataset(synthetic):
    return validate_dataset(synthetic["table"], synthetic["metadata"], "group")


@pytest.fixture()
def tsv_bytes(synthetic):
    frame = synthetic["table"].counts.copy()
    frame.index.name = "taxon"
    return frame.to_csv(sep="\t").encode()


@pytest.fixture()
def metadata_bytes(synthetic):
    return synthetic["metadata"].to_csv(sep="\t").encode()
