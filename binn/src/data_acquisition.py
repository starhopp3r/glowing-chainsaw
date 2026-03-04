"""
GEO data acquisition, probe-to-gene collapse, and label binarization
for the BINN-HPV project (GSE40774).
"""
from __future__ import annotations

import os
import sys
import logging
import re

import numpy as np
import pandas as pd

# Allow `python src/data_acquisition.py` from anywhere inside binn_hpv/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Download ──────────────────────────────────────────────────────────────────

def download_geo(accession: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download a GEO dataset via GEOparse and return three DataFrames.

    Returns
    -------
    expr : pd.DataFrame
        Samples × probes, float32.
    meta : pd.DataFrame
        Samples × metadata columns.  Column names follow GEOparse's
        ``phenotype_data`` convention, e.g. ``characteristics_ch1.3.hpv status``.
    platform : pd.DataFrame
        Probes × annotation columns (includes GENE_SYMBOL / Gene Symbol).
    """
    try:
        import GEOparse
    except ImportError:
        raise ImportError(
            "GEOparse is not installed.  Run:  pip install GEOparse"
        )

    log.info(f"Downloading {accession} from GEO → {config.DATA_RAW}")
    gse = GEOparse.get_GEO(geo=accession, destdir=config.DATA_RAW, silent=False)

    # ── Expression matrix (samples × probes) ─────────────────────────
    # pivot_samples returns probes × samples; transpose to samples × probes.
    expr_wide = gse.pivot_samples("VALUE")
    expr = expr_wide.T.astype("float32")
    expr.index.name = "sample_id"

    # ── Sample metadata ───────────────────────────────────────────────
    # phenotype_data gives a clean DataFrame with dotted column names like
    # "characteristics_ch1.3.hpv status" and pre-parsed values ("positive").
    try:
        meta = gse.phenotype_data.copy()
        meta.index.name = "sample_id"
    except Exception:
        # Fallback: build from raw gsm.metadata dicts
        log.warning("phenotype_data unavailable; building metadata manually.")
        records = []
        for gsm_name, gsm in gse.gsms.items():
            row: dict = {"sample_id": gsm_name}
            for key, vals in gsm.metadata.items():
                row[key] = vals[0] if (isinstance(vals, list) and len(vals) == 1) else vals
            records.append(row)
        meta = pd.DataFrame(records).set_index("sample_id")

    # ── Platform annotation ───────────────────────────────────────────
    gpl_name = list(gse.gpls.keys())[0]
    gpl = gse.gpls[gpl_name]
    platform = gpl.table.copy()
    # First column is the probe ID; use it as the index.
    platform.index = platform.iloc[:, 0].astype(str)

    log.info(
        f"{accession}: {expr.shape[0]} samples, {expr.shape[1]} probes, "
        f"{platform.shape[0]} platform rows"
    )
    return expr, meta, platform


# ── Gene symbol utilities ─────────────────────────────────────────────────────

def standardize_gene_symbols(symbols: pd.Series) -> pd.Series:
    """
    Clean Affymetrix-style gene symbol annotations.

    Rules
    -----
    - Convert to uppercase.
    - For probes mapping to multiple genes (`` /// `` delimiter), keep only
      the first gene.
    - Strip trailing isoform / alias suffixes after ``//``.
    - Replace empty strings, "---", NaN, "N/A", "NONE" with ``NaN``.

    The input index (probe IDs) is preserved.
    """
    s = symbols.astype(str).str.strip()

    # Multiple genes: "GENE1 /// GENE2 /// GENE3" → "GENE1"
    s = s.str.split(r"\s*///\s*").str[0]

    # Isoform suffix: "GENE1 // isoform_info" → "GENE1"
    s = s.str.split(r"\s*//\s*").str[0]

    s = s.str.upper().str.strip()

    bad = s.isin({"", "NAN", "---", "N/A", "NONE", "NULL"}) | s.str.fullmatch(r"-+")
    s = s.where(~bad, other=np.nan)

    return s


# ── Probe collapse ────────────────────────────────────────────────────────────

def collapse_probes_maxmean(
    expr: pd.DataFrame,
    probe_to_gene: pd.Series,
) -> pd.DataFrame:
    """
    MaxMean probe collapse: for each gene, retain the probe whose mean
    expression across all samples is highest.

    Parameters
    ----------
    expr : pd.DataFrame
        Samples × probes (float32).
    probe_to_gene : pd.Series
        Index = probe ID, values = gene symbol (NaN for unmapped probes).

    Returns
    -------
    pd.DataFrame
        Samples × genes, float32.
    """
    # Restrict to probes present in expr and with a valid gene symbol.
    gene_map = probe_to_gene.reindex(expr.columns).dropna()

    # Mean expression per probe (scalar per column).
    probe_means = expr[gene_map.index].mean(axis=0)

    # For each gene pick the probe with the highest mean.
    mapping_df = pd.DataFrame({"gene": gene_map, "mean_expr": probe_means})
    best_probes = mapping_df.groupby("gene")["mean_expr"].idxmax()

    # Assemble gene-level expression matrix.
    gene_expr = expr[best_probes.values].copy()
    gene_expr.columns = best_probes.index          # gene symbols
    gene_expr.index.name = "sample_id"
    gene_expr = gene_expr.astype("float32")

    log.info(
        f"Probe collapse: {len(gene_map)} mapped probes → "
        f"{gene_expr.shape[1]} unique genes"
    )
    return gene_expr


# ── Label binarization ────────────────────────────────────────────────────────

def binarize_labels(
    metadata: pd.DataFrame,
    column: str,
    positive_values: set,
    negative_values: set | None = None,
) -> pd.Series:
    """
    Map HPV status column to binary 0/1 labels.

    Lookup order
    ------------
    1. ``column`` exists verbatim in metadata (GEOparse phenotype_data format).
       Values may be plain ("positive") or colon-delimited ("hpv status: positive").
    2. Any column whose name contains "hpv" (case-insensitive).
    3. Parse each entry in ``characteristics_ch1`` (a list) for "hpv status:".

    Returns
    -------
    pd.Series
        Integer 0/1 labels indexed by sample_id; rows with missing labels
        are dropped.
    """
    pos_set = {str(v).strip().lower() for v in positive_values}
    neg_set = {
        str(v).strip().lower()
        for v in (
            negative_values
            if negative_values is not None
            else {"negative", "neg", "hpv-", "hpv negative", "no", "0", "false"}
        )
    }

    def _extract_value(series: pd.Series) -> pd.Series:
        """Flatten lists, lowercase, strip, and take the part after the last ':'."""
        out = series.apply(lambda v: v[0] if isinstance(v, list) else v)
        out = out.astype(str).str.lower().str.strip()
        # Handle "hpv status: positive" → "positive"
        out = out.str.split(r":\s*").str[-1].str.strip()
        return out

    raw: pd.Series | None = None

    if column in metadata.columns:
        raw = _extract_value(metadata[column])

    else:
        # Try any column whose name contains "hpv"
        hpv_cols = [c for c in metadata.columns if "hpv" in c.lower()]
        if hpv_cols:
            log.info(
                f"Column '{column}' not found; using '{hpv_cols[0]}' instead."
            )
            raw = _extract_value(metadata[hpv_cols[0]])

        else:
            # Last resort: search characteristics_ch1 list entries
            char_col = "characteristics_ch1"
            if char_col not in metadata.columns:
                raise KeyError(
                    f"Cannot find HPV status in metadata. "
                    f"Tried '{column}', any 'hpv*' column, and '{char_col}'. "
                    f"Available columns: {list(metadata.columns[:30])}"
                )
            log.warning(
                f"Column '{column}' not found; searching '{char_col}' list entries."
            )

            def _extract_from_chars(chars: object) -> object:
                entries = chars if isinstance(chars, list) else [chars]
                for item in entries:
                    s = str(item).lower().strip()
                    if "hpv" in s:
                        return s.split(":")[-1].strip()
                return np.nan

            raw = metadata[char_col].apply(_extract_from_chars)

    # Map to 1 / 0 / NaN
    def _to_binary(v: object) -> object:
        val = str(v).strip().lower()
        if val in {"nan", "", "none", "na"}:
            return np.nan

        # Prefer exact matches first.
        if val in pos_set:
            return 1
        if val in neg_set:
            return 0

        # Fallback for compact encodings used by GEO (e.g. "pos"/"neg").
        if re.search(r"\bpos(itive)?\b", val):
            return 1
        if re.search(r"\bneg(ative)?\b", val):
            return 0

        # Unknown category: keep missing instead of forcing to negative.
        return np.nan

    mapped = raw.map(_to_binary)
    unknown_count = int(mapped.isna().sum())
    labels = mapped.dropna().astype(int)
    labels.name = "label"

    log.info(
        f"Labels: {labels.sum()} positive, {(labels == 0).sum()} negative "
        f"({len(labels)} total, {unknown_count} unknown dropped)"
    )
    return labels


# ── MAD utilities ─────────────────────────────────────────────────────────────

def compute_mad(expr: pd.DataFrame) -> pd.Series:
    """
    Compute the Median Absolute Deviation (MAD) for each gene (column).

    Returns
    -------
    pd.Series
        MAD values indexed by gene name.
    """
    median = expr.median(axis=0)
    mad = (expr - median).abs().median(axis=0)
    return mad


def apply_mad_filter(
    expr: pd.DataFrame,
    mad_values: pd.Series,
    percentile: float,
) -> pd.DataFrame:
    """
    Remove genes whose MAD falls strictly below the given percentile.

    Parameters
    ----------
    percentile : float
        Genes with MAD < np.percentile(mad_values, percentile) are dropped.
        E.g. 50 → discard the bottom half.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix (same row order, fewer columns).
    """
    threshold = float(np.percentile(mad_values, percentile))
    keep = mad_values[mad_values >= threshold].index
    log.info(
        f"MAD filter (p{percentile:.0f}): "
        f"{expr.shape[1]} genes → {len(keep)} genes retained"
    )
    return expr[keep]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing() -> None:
    """End-to-end preprocessing pipeline for GSE40774."""

    # 1. Download
    expr_raw, meta, platform = download_geo(config.GEO_ACCESSION)

    # 2. Identify the GENE_SYMBOL column in the platform annotation.
    #    Affymetrix platforms use 'Gene Symbol' or 'GENE_SYMBOL'.
    cols_upper_map = {c.upper(): c for c in platform.columns}
    gs_col = cols_upper_map.get(
        config.GENE_SYMBOL_COLUMN.upper(),
        cols_upper_map.get("GENE SYMBOL", None),
    )
    if gs_col is None:
        raise KeyError(
            f"Cannot find a gene symbol column in the platform table. "
            f"Available columns: {list(platform.columns)}"
        )
    log.info(f"Using platform column '{gs_col}' for gene symbols.")

    # 3. Standardize gene symbols (preserves probe-ID index from platform).
    probe_to_gene = standardize_gene_symbols(platform[gs_col])

    # 4. Collapse probes → genes (MaxMean).
    expr_gene = collapse_probes_maxmean(expr_raw, probe_to_gene)

    # 5. Binarize HPV status labels.
    labels = binarize_labels(
        meta,
        config.METADATA_COLUMN,
        config.POSITIVE_LABELS,
        config.NEGATIVE_LABELS,
    )

    # 6. Align samples (intersection of expression index and labels index).
    common = expr_gene.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError(
            "No samples overlap between expression matrix and labels. "
            "Check sample ID formats in GEO and metadata."
        )
    expr_aligned = expr_gene.loc[common]
    labels_aligned = labels.loc[common]

    log.info(
        f"Sample alignment: {len(common)} samples with both "
        f"expression data and valid HPV labels "
        f"(dropped {len(expr_gene) - len(common)} unmatched)."
    )

    # 7. Save outputs.
    expr_aligned.to_parquet(
        config.EXPRESSION_MATRIX_FILE, engine="pyarrow", compression="snappy"
    )

    labels_df = labels_aligned.reset_index()
    labels_df.columns = ["sample_id", "label"]
    labels_df.to_csv(config.LABELS_FILE, index=False)

    with open(config.GENE_LIST_FILE, "w") as fh:
        fh.write("\n".join(expr_aligned.columns.tolist()))

    # 8. Print summary.
    n_samples = len(expr_aligned)
    n_pos = int(labels_aligned.sum())
    n_neg = n_samples - n_pos
    n_genes = expr_aligned.shape[1]

    print()
    print("GSE40774 Preprocessing Summary")
    print("================================")
    print(f"Total samples:        {n_samples}")
    print(f"HPV-positive:         {n_pos} ({100 * n_pos / n_samples:.1f}%)")
    print(f"HPV-negative:         {n_neg} ({100 * n_neg / n_samples:.1f}%)")
    print(f"Genes after collapse: {n_genes}")
    print(f"Expression shape:     {expr_aligned.shape}")
    print(f"Saved to:             {config.DATA_PROCESSED}/")


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load preprocessed expression matrix and labels from canonical paths.

    Returns
    -------
    (expression_df, labels)
        expression_df: samples × genes DataFrame
        labels: binary 0/1 Series indexed by sample_id
    """
    if not os.path.exists(config.EXPRESSION_MATRIX_FILE):
        raise FileNotFoundError(
            f"Expression matrix not found at {config.EXPRESSION_MATRIX_FILE}.\n"
            "Run data preprocessing first:  uv run python src/data_acquisition.py"
        )
    if not os.path.exists(config.LABELS_FILE):
        raise FileNotFoundError(
            f"Labels not found at {config.LABELS_FILE}.\n"
            "Run data preprocessing first:  uv run python src/data_acquisition.py"
        )

    expr_df = pd.read_parquet(config.EXPRESSION_MATRIX_FILE)
    labels_df = pd.read_csv(config.LABELS_FILE)
    required_cols = {"sample_id", "label"}
    if not required_cols.issubset(labels_df.columns):
        raise ValueError(
            f"{config.LABELS_FILE} must contain columns {sorted(required_cols)}. "
            f"Found: {list(labels_df.columns)}"
        )

    labels = labels_df.set_index("sample_id")["label"].astype(int)
    labels.index = labels.index.astype(str)
    expr_df.index = expr_df.index.astype(str)

    common = expr_df.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping sample IDs between preprocessed expression and labels."
        )

    expr_df = expr_df.loc[common]
    labels = labels.loc[common]
    return expr_df, labels


if __name__ == "__main__":
    run_preprocessing()
