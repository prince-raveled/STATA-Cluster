"""AbundanceTable — the single object every parser produces (SPEC §8, §10 fork 4)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Greengenes / SILVA style rank prefixes, in order.
RANK_PREFIXES = ["k__", "p__", "c__", "o__", "f__", "g__", "s__"]
RANK_NAMES = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# --- resource limits for untrusted input -----------------------------------
# Uploaded files are untrusted. The upload cap (§8, 64 MB) bounds the *compressed*
# bytes, which is not the same as bounding the work: a 200 KB BIOM file can declare a
# million rows by a million columns, and a 1 MB .qza can expand to gigabytes. Both
# allocate before any §8 check can run, so the refusal has to happen in the parser.
#
# These are deliberately far above any real table — MicroVerse refuses above 1,500 taxa
# anyway once the data is in hand — and exist only to turn an out-of-memory kill into a
# clear message.
MAX_PARSE_CELLS = 50_000_000        # 400 MB as float64
MAX_PARSE_DIMENSION = 5_000_000     # taxa or samples, individually
MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024


def check_declared_shape(n_taxa: int, n_samples: int, source: str) -> None:
    """Refuse an impossible table before allocating it.

    Called with the dimensions the *file claims*, before any array of that size is
    created. A file that declares more than this is either corrupt or hostile; either
    way the honest response is a message, not a memory error.
    """
    if n_taxa < 0 or n_samples < 0:
        raise ParseError(f"{source} declares a negative dimension "
                         f"({n_taxa} x {n_samples}).")
    if n_taxa > MAX_PARSE_DIMENSION or n_samples > MAX_PARSE_DIMENSION:
        raise ParseError(
            f"{source} declares {n_taxa:,} taxa by {n_samples:,} samples. "
            f"No dimension may exceed {MAX_PARSE_DIMENSION:,}.")
    if n_taxa * n_samples > MAX_PARSE_CELLS:
        raise ParseError(
            f"{source} declares {n_taxa:,} x {n_samples:,} = "
            f"{n_taxa * n_samples:,} values, above the {MAX_PARSE_CELLS:,}-cell parse "
            f"limit. Filter the table before uploading.")
#: A leading domain marker means kingdom. `d__` is only a domain in first position:
#: MicrobiomeHD and other de novo OTU pipelines append `d__denovo1234` as the *last*
#: token, and reading that as a kingdom collapses the whole lineage to rank 0.
LEADING_ALIASES = {"d__": "k__", "sk__": "k__"}
#: Sub-species markers. `t__` is MetaPhlAn's strain; `d__` in a trailing position is a
#: de novo OTU identifier. Both sit below species.
SUBSPECIES_PREFIXES = ("t__", "d__", "otu__")

GENUS_INDEX = RANK_NAMES.index("genus")
SPECIES_INDEX = RANK_NAMES.index("species")


class ParseError(ValueError):
    """Raised for input we can explain to the user (§8: never a stack trace)."""


def _normalise_prefix(token: str, position: int) -> str:
    """Map a rank prefix to its canonical form, using the position to disambiguate."""
    if position == 0:
        for alias, canonical in LEADING_ALIASES.items():
            if token.startswith(alias):
                return canonical + token[len(alias):]
    return token


def split_lineage(lineage) -> list:
    """Split a lineage string on ';' or '|' and strip empty/placeholder tails."""
    if lineage is None:
        return []
    if isinstance(lineage, float) and np.isnan(lineage):
        return []
    text = str(lineage).strip()
    if not text:
        return []
    sep = "|" if "|" in text else ";"
    parts = [_normalise_prefix(p.strip(), i) for i, p in enumerate(text.split(sep))]
    cleaned: list = []
    for part in parts:
        body = part
        for pref in (*RANK_PREFIXES, *SUBSPECIES_PREFIXES):
            if body.startswith(pref):
                body = body[len(pref):]
                break
        if body.lower() in {"", "unclassified", "unassigned", "na", "none", "__"}:
            body = ""
        cleaned.append(part if body else "")
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def lineage_rank_index(parts: list) -> int:
    """Deepest rank index a lineage reaches. Prefix-aware, falls back to depth."""
    if not parts:
        return -1
    last = parts[-1]
    # A strain or de novo OTU token is finer than species, so it caps at species
    # rather than being mistaken for whatever prefix it happens to share.
    if last.startswith(SUBSPECIES_PREFIXES):
        return SPECIES_INDEX
    for i, pref in enumerate(RANK_PREFIXES):
        if last.startswith(pref):
            return i
    return min(len(parts) - 1, len(RANK_NAMES) - 1)


def strip_prefix(token: str) -> str:
    for pref in RANK_PREFIXES:
        if token.startswith(pref):
            return token[len(pref):]
    return token


@dataclass
class AbundanceTable:
    """Taxa (rows) x samples (columns), plus optional lineages.

    ``counts`` always has taxa on the index. Orientation is resolved at parse time
    so nothing downstream has to guess.
    """

    counts: pd.DataFrame
    lineages: dict = field(default_factory=dict)
    source_format: str = "unknown"
    value_type: str = "counts"  # counts | relative
    input_rank: str = "input"
    notes: list = field(default_factory=list)

    # ---- shape ----------------------------------------------------------
    @property
    def n_taxa(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[1])

    @property
    def sample_ids(self) -> list:
        return [str(c) for c in self.counts.columns]

    @property
    def taxa(self) -> list:
        return [str(i) for i in self.counts.index]

    @property
    def is_integer(self) -> bool:
        values = self.counts.to_numpy(dtype=float)
        return bool(np.all(np.isfinite(values)) and np.allclose(values, np.round(values)))

    @property
    def library_sizes(self) -> np.ndarray:
        return self.counts.to_numpy(dtype=float).sum(axis=0)

    @property
    def has_negative(self) -> bool:
        return bool((self.counts.to_numpy(dtype=float) < 0).any())

    # ---- taxonomy -------------------------------------------------------
    @property
    def has_taxonomy(self) -> bool:
        return any(len(v) > 0 for v in self.lineages.values())

    @property
    def deepest_rank_index(self) -> int:
        if not self.has_taxonomy:
            return -1
        idx = [lineage_rank_index(v) for v in self.lineages.values() if v]
        return int(np.median(idx)) if idx else -1

    @property
    def can_collapse_to_genus(self) -> bool:
        """Fork 4 has two levels only when the input is finer than genus (§10)."""
        return self.has_taxonomy and self.deepest_rank_index > GENUS_INDEX

    def genus_label(self, taxon: str) -> str:
        parts = self.lineages.get(taxon) or []
        if len(parts) > GENUS_INDEX and parts[GENUS_INDEX]:
            return ";".join(parts[: GENUS_INDEX + 1])
        if parts:
            return ";".join(parts[: min(len(parts), GENUS_INDEX + 1)])
        return taxon

    def collapse_to_genus(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Sum counts of taxa sharing a genus lineage. Unassigned taxa pass through."""
        if not self.can_collapse_to_genus:
            return matrix
        groups = pd.Index([self.genus_label(str(t)) for t in matrix.index], name="taxon")
        return matrix.groupby(groups, sort=False).sum()

    def display_name(self, taxon: str) -> str:
        """Shortest informative label: the deepest non-empty lineage token."""
        parts = self.lineages.get(taxon) or split_lineage(taxon)
        for token in reversed(parts):
            body = strip_prefix(token)
            if body:
                return body
        return taxon

    # ---- helpers --------------------------------------------------------
    def attach_taxonomy(self, mapping: dict) -> int:
        """Apply an external feature-ID -> lineage map and re-derive the input rank.

        Returns the number of taxa matched, so the caller can warn about a map that
        does not line up with the abundance table.
        """
        matched = 0
        for taxon in self.taxa:
            parts = mapping.get(taxon)
            if parts:
                self.lineages[taxon] = parts
                matched += 1
        if matched:
            depths = [lineage_rank_index(v) for v in self.lineages.values() if v]
            if depths:
                index = int(np.median(depths))
                if 0 <= index < len(RANK_NAMES):
                    self.input_rank = RANK_NAMES[index]
            self.notes.append(
                f"Taxonomy applied to {matched} of {self.n_taxa} features "
                f"(input rank: {self.input_rank})."
            )
        return matched

    def align_to(self, sample_ids: list) -> AbundanceTable:
        return AbundanceTable(
            counts=self.counts.loc[:, sample_ids].copy(),
            lineages=self.lineages,
            source_format=self.source_format,
            value_type=self.value_type,
            input_rank=self.input_rank,
            notes=list(self.notes),
        )

    def summary(self) -> dict:
        libs = self.library_sizes
        return {
            "n_taxa": self.n_taxa,
            "n_samples": self.n_samples,
            "format": self.source_format,
            "value_type": self.value_type,
            "input_rank": self.input_rank,
            "integer_values": self.is_integer,
            "has_taxonomy": self.has_taxonomy,
            "can_collapse_to_genus": self.can_collapse_to_genus,
            "min_library": float(libs.min()) if libs.size else 0.0,
            "median_library": float(np.median(libs)) if libs.size else 0.0,
            "max_library": float(libs.max()) if libs.size else 0.0,
            "sparsity": float((self.counts.to_numpy(dtype=float) == 0).mean()),
            "notes": list(self.notes),
        }


TAXON_HINT = re.compile(r"(k__|p__|c__|o__|f__|g__|s__|d__|otu|asv|taxon|taxa|feature|clade)", re.I)


def looks_like_taxonomy(labels) -> float:
    """Fraction of labels that look like taxa rather than sample identifiers."""
    labels = [str(x) for x in labels]
    if not labels:
        return 0.0
    hits = sum(1 for x in labels if TAXON_HINT.search(x) or ";" in x or "|" in x)
    return hits / len(labels)


def resolve_orientation(frame: pd.DataFrame, sample_ids: list | None = None):
    """Return (taxa x samples, note). Metadata IDs win; heuristics otherwise."""
    rows = [str(x) for x in frame.index]
    cols = [str(x) for x in frame.columns]

    if sample_ids:
        wanted = {str(s) for s in sample_ids}
        row_hit = len(wanted & set(rows))
        col_hit = len(wanted & set(cols))
        if col_hit > row_hit:
            return frame, "orientation: taxa x samples (columns matched metadata IDs)"
        if row_hit > col_hit:
            return frame.T, "orientation: samples x taxa, transposed (rows matched metadata IDs)"

    row_tax = looks_like_taxonomy(rows)
    col_tax = looks_like_taxonomy(cols)
    if col_tax > row_tax + 0.2:
        return frame.T, "orientation: samples x taxa, transposed (column labels look taxonomic)"
    if row_tax > col_tax + 0.2:
        return frame, "orientation: taxa x samples (row labels look taxonomic)"

    # Microbiome tables are near-universally wider in taxa than in samples.
    if frame.shape[1] > frame.shape[0]:
        return frame.T, "orientation: assumed samples x taxa from shape, transposed"
    return frame, "orientation: assumed taxa x samples from shape"


def detect_value_type(matrix: pd.DataFrame) -> str:
    values = matrix.to_numpy(dtype=float)
    if values.size == 0:
        return "counts"
    if np.allclose(values, np.round(values)) and values.max() > 1.5:
        return "counts"
    sums = values.sum(axis=0)
    if np.allclose(sums, 1.0, atol=0.02) or np.allclose(sums, 100.0, atol=1.0):
        return "relative"
    if values.max() <= 1.0:
        return "relative"
    return "counts"
