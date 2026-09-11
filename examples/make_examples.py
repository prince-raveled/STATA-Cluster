"""Generate the three demo datasets shipped with MicroVerse.

Counts are drawn from a Dirichlet-multinomial with log-normal library sizes, which is
the usual stand-in for 16S/shotgun count data. Known differential taxa are spiked at
declared effect sizes so validation 4 of SPEC §23 (simulated ground truth) has a
target: true positives should land ROBUST, nulls FRAGILE / UNSTABLE / NOT DETECTED.

    python examples/make_examples.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

GENERA = [
    "Faecalibacterium", "Bacteroides", "Escherichia", "Blautia", "Roseburia",
    "Ruminococcus", "Akkermansia", "Bifidobacterium", "Prevotella", "Alistipes",
    "Parabacteroides", "Dorea", "Coprococcus", "Eubacterium", "Streptococcus",
    "Veillonella", "Lactobacillus", "Clostridium", "Collinsella", "Anaerostipes",
]
FAMILIES = {
    "Faecalibacterium": ("Firmicutes", "Clostridia", "Clostridiales", "Ruminococcaceae"),
    "Ruminococcus": ("Firmicutes", "Clostridia", "Clostridiales", "Ruminococcaceae"),
    "Blautia": ("Firmicutes", "Clostridia", "Clostridiales", "Lachnospiraceae"),
    "Roseburia": ("Firmicutes", "Clostridia", "Clostridiales", "Lachnospiraceae"),
    "Dorea": ("Firmicutes", "Clostridia", "Clostridiales", "Lachnospiraceae"),
    "Coprococcus": ("Firmicutes", "Clostridia", "Clostridiales", "Lachnospiraceae"),
    "Anaerostipes": ("Firmicutes", "Clostridia", "Clostridiales", "Lachnospiraceae"),
    "Eubacterium": ("Firmicutes", "Clostridia", "Clostridiales", "Eubacteriaceae"),
    "Clostridium": ("Firmicutes", "Clostridia", "Clostridiales", "Clostridiaceae"),
    "Streptococcus": ("Firmicutes", "Bacilli", "Lactobacillales", "Streptococcaceae"),
    "Lactobacillus": ("Firmicutes", "Bacilli", "Lactobacillales", "Lactobacillaceae"),
    "Veillonella": ("Firmicutes", "Negativicutes", "Veillonellales", "Veillonellaceae"),
    "Bacteroides": ("Bacteroidetes", "Bacteroidia", "Bacteroidales", "Bacteroidaceae"),
    "Prevotella": ("Bacteroidetes", "Bacteroidia", "Bacteroidales", "Prevotellaceae"),
    "Alistipes": ("Bacteroidetes", "Bacteroidia", "Bacteroidales", "Rikenellaceae"),
    "Parabacteroides": ("Bacteroidetes", "Bacteroidia", "Bacteroidales", "Tannerellaceae"),
    "Escherichia": ("Proteobacteria", "Gammaproteobacteria", "Enterobacterales",
                    "Enterobacteriaceae"),
    "Akkermansia": ("Verrucomicrobia", "Verrucomicrobiae", "Verrucomicrobiales",
                    "Akkermansiaceae"),
    "Bifidobacterium": ("Actinobacteria", "Actinobacteria", "Bifidobacteriales",
                        "Bifidobacteriaceae"),
    "Collinsella": ("Actinobacteria", "Coriobacteriia", "Coriobacteriales",
                    "Coriobacteriaceae"),
}


def genus_name(i: int) -> str:
    if i < len(GENERA):
        return GENERA[i]
    return f"Genus_{i - len(GENERA) + 1:03d}"


def lineage_for(name: str) -> str:
    phylum, klass, order, family = FAMILIES.get(
        name, ("Firmicutes", "Clostridia", "Clostridiales", "Lachnospiraceae")
    )
    return f"k__Bacteria;p__{phylum};c__{klass};o__{order};f__{family};g__{name}"


def simulate(
    n_taxa: int = 380,
    n_per_group: int = 60,
    n_differential: int = 24,
    depth_mean: float = 9.6,
    depth_sd: float = 0.55,
    depth_shift: float = 0.0,
    seed: int = 20260907,
):
    """Dirichlet-multinomial counts with a known set of differential taxa."""
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_per_group
    groups = np.array([0] * n_per_group + [1] * n_per_group)

    # A realistic, heavy-tailed community mean.
    base = rng.lognormal(mean=0.0, sigma=1.7, size=n_taxa)
    base = np.sort(base)[::-1]
    base = base / base.sum()

    # Spike differential taxa among the taxa that are actually observable — a spiked
    # effect on a taxon seen in three samples measures nothing.
    observable = int(n_taxa * 0.45)
    differential = rng.choice(observable, size=n_differential, replace=False)
    log2_effect = np.zeros(n_taxa)
    # A spread of effect sizes, so the run produces every robustness tier rather than
    # one of them: strong, moderate and marginal signals.
    strengths = np.concatenate([
        rng.uniform(1.6, 2.6, size=n_differential // 3),
        rng.uniform(0.9, 1.6, size=n_differential // 3),
        rng.uniform(0.35, 0.9, size=n_differential - 2 * (n_differential // 3)),
    ])
    log2_effect[differential] = strengths * rng.choice([-1, 1], n_differential)

    fold = 2.0 ** log2_effect
    mean_a = base
    mean_b = base * fold
    mean_b = mean_b / mean_b.sum()

    # Overdispersion: each sample draws its own composition around the group mean.
    # Concentration sets between-subject variability; ~300 gives the sparsity and
    # dispersion of a real genus-level stool cohort.
    concentration = 300.0
    libraries = np.rint(
        rng.lognormal(mean=depth_mean, sigma=depth_sd, size=n_samples)
        * np.where(groups == 1, 2.0 ** depth_shift, 1.0)
    ).astype(int)
    libraries = np.clip(libraries, 1200, 400_000)

    counts = np.zeros((n_taxa, n_samples), dtype=int)
    for j in range(n_samples):
        mean = mean_b if groups[j] == 1 else mean_a
        composition = rng.dirichlet(np.maximum(mean * concentration, 1e-4))
        counts[:, j] = rng.multinomial(libraries[j], composition)

    return counts, groups, libraries, differential, log2_effect


def write_dataset(name: str, counts, groups, libraries, differential, log2_effect,
                  taxa_names, lineages, metadata_extra=None, sample_prefix="S"):
    sample_ids = [f"{sample_prefix}{i + 1:03d}" for i in range(counts.shape[1])]
    table = pd.DataFrame(counts, index=taxa_names, columns=sample_ids)
    table.index.name = "taxon"

    metadata = pd.DataFrame(
        {"group": np.where(groups == 1, "disease", "control"), "read_depth": libraries},
        index=sample_ids,
    )
    metadata.index.name = "sample_id"
    if metadata_extra is not None:
        for key, values in metadata_extra.items():
            metadata[key] = values

    taxonomy = pd.DataFrame({"taxon": [lineages[t] for t in taxa_names]}, index=taxa_names)
    taxonomy.index.name = "feature_id"

    table.to_csv(os.path.join(HERE, f"{name}_abundance.tsv"), sep="\t")
    metadata.to_csv(os.path.join(HERE, f"{name}_metadata.tsv"), sep="\t")
    taxonomy.to_csv(os.path.join(HERE, f"{name}_taxonomy.tsv"), sep="\t")

    truth = {
        "differential_taxa": sorted(taxa_names[i] for i in differential),
        "log2_effect": {taxa_names[i]: round(float(log2_effect[i]), 3) for i in differential},
        "n_taxa": int(counts.shape[0]),
        "n_samples": int(counts.shape[1]),
    }
    with open(os.path.join(HERE, f"{name}_truth.json"), "w", encoding="utf-8") as handle:
        json.dump(truth, handle, indent=2)
    return truth


def build_ibd():
    counts, groups, libraries, differential, effect = simulate(seed=20260907)
    names = [genus_name(i) for i in range(counts.shape[0])]
    lineages = {n: lineage_for(n) for n in names}
    return write_dataset("ibd_genus", counts, groups, libraries, differential, effect,
                         names, lineages)


def build_t2d():
    """Smaller cohort with covariates, for covariate mode."""
    counts, groups, libraries, differential, effect = simulate(
        n_taxa=220, n_per_group=45, n_differential=14, depth_shift=0.35, seed=31415
    )
    rng = np.random.default_rng(99)
    n = counts.shape[1]
    age = np.rint(rng.normal(52 + 6 * groups, 11)).astype(int)
    bmi = np.round(rng.normal(26 + 2.4 * groups, 4.1), 1)
    sex = rng.choice(["F", "M"], size=n)
    smoker = rng.choice(["no", "yes"], size=n, p=[0.78, 0.22])
    names = [genus_name(i) for i in range(counts.shape[0])]
    lineages = {n_: lineage_for(n_) for n_ in names}
    return write_dataset(
        "t2d_covariates", counts, groups, libraries, differential, effect, names, lineages,
        metadata_extra={"age": age, "bmi": bmi, "sex": sex, "smoker": smoker},
        sample_prefix="P",
    )


def build_species():
    """Species-level table so fork 4 (rank) has two levels."""
    counts, groups, libraries, differential, effect = simulate(
        n_taxa=300, n_per_group=40, n_differential=18, seed=2718
    )
    rng = np.random.default_rng(7)
    names, lineages = [], {}
    for i in range(counts.shape[0]):
        genus = genus_name(i % 40)
        species = f"{genus.lower()}_sp{rng.integers(1, 40):02d}_{i:03d}"
        full = f"{lineage_for(genus)};s__{species}"
        names.append(full)
        lineages[full] = full
    return write_dataset("gut_species", counts, groups, libraries, differential, effect,
                         names, lineages, sample_prefix="R")


if __name__ == "__main__":
    for builder in (build_ibd, build_t2d, build_species):
        truth = builder()
        print(f"{builder.__name__}: {truth['n_taxa']} taxa x {truth['n_samples']} samples, "
              f"{len(truth['differential_taxa'])} spiked")
