"""Load real case-control cohorts from MicrobiomeHD.

MicrobiomeHD (Duvallet et al. 2017, *Nat Commun* 8:1784; Zenodo 1146764, CC-BY-NC-4.0)
is the collection SPEC §27 lists as public test data: 28 published 16S case-control
studies, processed through one pipeline, with OTU tables whose row labels carry the
full RDP lineage.

This module downloads a cohort on demand, caches the tarball under `data/microbiomehd/`,
and returns a validated MicroVerse `Dataset`. Nothing here is committed — the archive is
someone else's data under a non-commercial licence, and the cache is git-ignored.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import urllib.request

import pandas as pd

ZENODO_RECORD = "1146764"
CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "microbiomehd",
)

#: Cohort -> (case label, control label) in the `DiseaseState` column. Restricted to
#: two groups because MicroVerse is two-group only (SPEC §21); studies with more arms
#: are subset to the case/control contrast the original paper reported.
COHORTS = {
    # cohort:            (case, control, description)
    "cdi_schubert":      ("CDI",  "H",       "C. difficile infection — Schubert 2014"),
    "cdi_vincent_v3v5":  ("CDI",  "H",       "C. difficile infection — Vincent 2013"),
    "crc_zackular":      ("CRC",  "H",       "Colorectal cancer — Zackular 2014"),
    "crc_zhao":          ("CRC",  "H",       "Colorectal cancer — Wang 2012"),
    "crc_xiang":         ("CRC",  "H",       "Colorectal cancer — Chen 2012"),
    "ibd_huttenhower":   ("CD",   "H",       "Crohn's disease — Papa 2012"),
    "ob_escobar":        ("OB",   "H",       "Obesity — Escobar 2014"),
    "ob_ross":           ("OB",   "H",       "Obesity — Ross 2015"),
    "t1d_mejialeon":     ("T1D",  "H",       "Type 1 diabetes — Mejia-Leon 2014"),
    "autism_kb":         ("ASD",  "H",       "Autism — Kang 2013"),
    "asd_son":           ("ASD",  "H",       "Autism — Son 2015"),
    "mhe_zhang":         ("MHE",  "H",       "Hepatic encephalopathy — Zhang 2013"),
    "hiv_dinh":          ("HIV",  "H",       "HIV — Dinh 2015"),
    "edd_singh":         ("EDD",  "H",       "Diarrhoeal illness — Singh 2015"),
    "nash_chan":         ("NASH", "H",       "NASH — Wong 2013"),
    "ra_littman":        ("RA",   "H",       "Rheumatoid arthritis — Scher 2013"),
    "par_scheperjans":   ("PAR",  "H",       "Parkinson's — Scheperjans 2015"),
}


def _file_links() -> dict:
    """Zenodo file listing, cached so repeated runs do not re-query."""
    path = os.path.join(CACHE, "_zenodo_files.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    os.makedirs(CACHE, exist_ok=True)
    request = urllib.request.Request(
        f"https://zenodo.org/api/records/{ZENODO_RECORD}",
        headers={"User-Agent": "microverse-validation"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        record = json.load(response)
    links = {f["key"]: f["links"]["self"] for f in record["files"]}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(links, handle, indent=2)
    return links


def download(cohort: str) -> str:
    """Fetch and cache one cohort's tarball. Returns the local path."""
    name = f"{cohort}_results.tar.gz"
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    links = _file_links()
    if name not in links:
        raise KeyError(f"{name} is not in Zenodo record {ZENODO_RECORD}")
    os.makedirs(CACHE, exist_ok=True)
    request = urllib.request.Request(links[name], headers={"User-Agent": "microverse-validation"})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read()
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def _read_member(archive: tarfile.TarFile, suffix: str) -> bytes:
    for member in archive.getmembers():
        if member.name.endswith(suffix):
            return archive.extractfile(member).read()
    raise KeyError(f"no member ending in {suffix!r}")


def load_raw(cohort: str):
    """Return (otu_table, metadata) as DataFrames, straight from the archive."""
    with tarfile.open(download(cohort)) as archive:
        table_bytes = _read_member(archive, "rdp_assigned")
        metadata_bytes = _read_member(archive, "metadata.txt")

    table = pd.read_csv(io.BytesIO(table_bytes), sep="\t", index_col=0,
                        encoding="utf-8", encoding_errors="replace")
    metadata = pd.read_csv(io.BytesIO(metadata_bytes), sep="\t", index_col=0,
                           encoding="utf-8", encoding_errors="replace", low_memory=False)
    metadata.index = [str(i).strip() for i in metadata.index]
    return table, metadata


def load_dataset(cohort: str, max_taxa: int = 1500, keep_otu_level: bool = False):
    """Download, subset to the two-group contrast, and validate.

    Returns (Dataset, description). Raises `DatasetError` if the cohort does not meet
    the §8 constraints, which is itself informative.

    `keep_otu_level` trims to the §8 ceiling directly instead of leaving four times as
    many OTUs for the validator to collapse. That matters for fork 4: a table already
    at genus has one rank level, so the taxonomic-rank fork does nothing, and every
    MicrobiomeHD result carries that caveat. Trimming to the ceiling keeps the OTU
    table intact, so the fork has both OTU and genus to move between — at the cost of
    dropping the least prevalent OTUs, which is itself an analytical choice and is
    reported as one.
    """
    from app.core.parsers.base import AbundanceTable, split_lineage
    from app.core.validation import validate_dataset

    case, control, description = COHORTS[cohort]
    table, metadata = load_raw(cohort)

    column = next((c for c in metadata.columns if c.strip() == "DiseaseState"), None)
    if column is None:
        raise KeyError(f"{cohort}: no DiseaseState column")
    states = metadata[column].astype(str).str.strip()
    keep = states.isin({case, control})
    metadata = metadata.loc[keep, [column]].copy()
    metadata.columns = ["group"]
    # Sort so the case label is group B and a positive effect means "higher in cases".
    metadata["group"] = metadata["group"].map({control: f"a_{control}", case: f"b_{case}"})

    shared = [s for s in table.columns if s in set(metadata.index)]
    table = table[shared]
    metadata = metadata.loc[shared]

    counts = table.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    counts = counts.loc[counts.sum(axis=1) > 0]

    # MicroVerse refuses above 1,500 taxa without a taxonomy to collapse (SPEC §8);
    # these tables carry lineages, so keep the most prevalent OTUs and let the
    # validator collapse to genus if it still needs to.
    ceiling = max_taxa if keep_otu_level else max_taxa * 4
    if counts.shape[0] > ceiling:
        prevalence = (counts > 0).sum(axis=1)
        counts = counts.loc[prevalence.sort_values(ascending=False).index[:ceiling]]

    lineages = {str(t): split_lineage(t) for t in counts.index}
    abundance = AbundanceTable(
        counts=counts,
        lineages={k: v for k, v in lineages.items() if v},
        source_format="microbiomehd",
        value_type="counts",
    )
    dataset = validate_dataset(abundance, metadata, "group")
    return dataset, description
