"""
Query peaks associated with a list of marker genes.

Usage (Python):
    from marker_gene_peaks import peaks_for_genes

    df = peaks_for_genes(
        genes=["foxd3", "sox10", "tbxta"],
        min_z=0.0,           # filter: keep only peaks with max_z >= min_z
        include_linked=True,  # include peaks where linked_gene matches
        include_nearest=True, # include peaks where nearest_gene matches
        wide=False,           # set True to include all 31 z_<celltype> columns
    )

Usage (CLI):
    python marker_gene_peaks.py --genes foxd3,sox10,tbxta -o peaks.csv
    python marker_gene_peaks.py --genes-file markers.txt --min-z 5 --wide

Cache:
    Reads from notebooks/EDA_peak_parts_list/outputs/V3/peak_metadata_cache.parquet
    Build it once via scripts/utils/build_peak_metadata_cache.py
"""

import os
import sys
import argparse
import pandas as pd

REPO        = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
CACHE_PATH  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3/peak_metadata_cache.parquet"

# Cached parquet load (lazy, module-level)
_CACHE = None

def _load_cache(path: str = CACHE_PATH) -> pd.DataFrame:
    """Load the peak metadata cache (memoized on module level)."""
    global _CACHE
    if _CACHE is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Peak metadata cache not found: {path}\n"
                f"Build it first: python {REPO}/scripts/utils/build_peak_metadata_cache.py"
            )
        _CACHE = pd.read_parquet(path)
    return _CACHE


def peaks_for_genes(
    genes,
    min_z: float = 0.0,
    include_linked: bool = True,
    include_nearest: bool = True,
    wide: bool = False,
    cache_path: str = CACHE_PATH,
) -> pd.DataFrame:
    """
    Return peaks associated with the given marker genes.

    Parameters
    ----------
    genes : list[str] or str
        Marker gene names (case-sensitive — match the master annotation).
    min_z : float
        Only return peaks with max_z >= min_z. Default 0 (no filter).
    include_linked : bool
        Match peaks whose linked_gene is in `genes`.
    include_nearest : bool
        Match peaks whose nearest_gene is in `genes`.
    wide : bool
        If True, output keeps the 31 z_<celltype> columns; default False
        keeps only the compact summary (top1/top2/top3 + max_z + tau).
    cache_path : str
        Path to the parquet cache.

    Returns
    -------
    pd.DataFrame
        One row per peak (peak_id × gene_match). Sorted by max_z descending.
        Columns:
          peak_id, chrom, start, end, length, peak_type, distance_to_tss,
          linked_gene, nearest_gene, query_gene, via_linked, via_nearest,
          top1_celltype, top1_z, top2_celltype, top2_z, top3_celltype, top3_z,
          max_z, tau, gini, max_accessibility
        plus z_<celltype1>, z_<celltype2>, ... if wide=True
    """
    if isinstance(genes, str):
        genes = [g.strip() for g in genes.split(",") if g.strip()]
    genes = list(genes)
    if not genes:
        raise ValueError("`genes` is empty")
    if not (include_linked or include_nearest):
        raise ValueError("At least one of include_linked / include_nearest must be True")

    df = _load_cache(cache_path)
    gene_set = set(genes)

    # Match by linked / nearest separately so we can flag the source
    via_linked  = df["linked_gene"].isin(gene_set)  if include_linked  else pd.Series(False, index=df.index)
    via_nearest = df["nearest_gene"].isin(gene_set) if include_nearest else pd.Series(False, index=df.index)
    matched = via_linked | via_nearest

    if not matched.any():
        return df.iloc[:0].copy()   # empty result with correct schema

    out = df.loc[matched].copy()
    out["via_linked"]  = via_linked.loc[matched].values
    out["via_nearest"] = via_nearest.loc[matched].values

    # Determine which input gene matched (prefer linked over nearest)
    def pick_query_gene(row):
        lg = row.get("linked_gene", "")
        ng = row.get("nearest_gene", "")
        if row["via_linked"] and lg in gene_set:
            return lg
        if row["via_nearest"] and ng in gene_set:
            return ng
        return ""
    out["query_gene"] = out.apply(pick_query_gene, axis=1)

    # min_z filter
    if min_z > 0:
        out = out[out["max_z"] >= min_z]

    # Reorder columns: identifiers first, then summary, then optionally wide
    z_cols = [c for c in out.columns if c.startswith("z_")]
    base_cols = [
        "peak_id", "chrom", "start", "end", "length",
        "peak_type", "distance_to_tss",
        "linked_gene", "nearest_gene", "query_gene",
        "via_linked", "via_nearest",
        "top1_celltype", "top1_z",
        "top2_celltype", "top2_z",
        "top3_celltype", "top3_z",
        "max_z", "tau", "gini", "max_accessibility",
    ]
    base_cols = [c for c in base_cols if c in out.columns]
    if wide:
        cols = base_cols + sorted(z_cols)
    else:
        cols = base_cols
    out = out[cols].sort_values("max_z", ascending=False).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Query peaks associated with a list of marker genes."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--genes", type=str,
                     help="Comma-separated gene names (e.g., foxd3,sox10,tbxta)")
    src.add_argument("--genes-file", type=str,
                     help="Text file with one gene name per line")

    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path. If omitted, prints to stdout.")
    parser.add_argument("--min-z", type=float, default=0.0,
                        help="Filter: keep only peaks with max_z >= this value (default 0)")
    parser.add_argument("--no-linked",  action="store_true",
                        help="Disable matching via linked_gene")
    parser.add_argument("--no-nearest", action="store_true",
                        help="Disable matching via nearest_gene")
    parser.add_argument("--wide", action="store_true",
                        help="Include all 31 z_<celltype> columns")
    parser.add_argument("--cache", type=str, default=CACHE_PATH,
                        help="Path to peak metadata cache parquet")
    args = parser.parse_args()

    if args.genes:
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        with open(args.genes_file) as f:
            genes = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    df = peaks_for_genes(
        genes,
        min_z=args.min_z,
        include_linked=not args.no_linked,
        include_nearest=not args.no_nearest,
        wide=args.wide,
        cache_path=args.cache,
    )

    print(f"# Query genes: {', '.join(genes)}", file=sys.stderr)
    print(f"# Matched peaks: {len(df)}", file=sys.stderr)
    if len(df) > 0:
        per_gene = df["query_gene"].value_counts().to_dict()
        for g in genes:
            print(f"#   {g}: {per_gene.get(g, 0)} peaks", file=sys.stderr)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"# Wrote: {args.output}", file=sys.stderr)
    else:
        df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
