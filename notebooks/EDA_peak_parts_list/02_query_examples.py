# %% [markdown]
# # Step 02: Parts List Query Examples
#
# Demonstrates the query function: given (celltype, timepoint), return top peaks
# ranked by leave-one-out specificity z-score. Five biologically motivated
# example queries validate the approach against known biology.
#
# Inputs:
#   outputs/specificity_matrix.h5ad
#   outputs/specificity_summary.csv
#
# Outputs:
#   outputs/query_{condition}_top50.csv  — for each example query

# %% Imports
import os, re
import numpy as np
import pandas as pd
import anndata as ad

print(f"anndata {ad.__version__}")

# %% Paths
BASE   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO   = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR = f"{REPO}/notebooks/EDA_peak_parts_list/outputs"

SPEC_H5AD = f"{OUTDIR}/specificity_matrix.h5ad"
SPEC_CSV  = f"{OUTDIR}/specificity_summary.csv"

# %% Load specificity matrix
print(f"Loading specificity matrix ...", flush=True)
import time; t0 = time.time()
Z_adata = ad.read_h5ad(SPEC_H5AD)
print(f"  Shape: {Z_adata.shape}  ({time.time()-t0:.1f}s)")

# Build condition→index lookup (fast)
cond_to_idx = {name: i for i, name in enumerate(Z_adata.var_names)}

# Available conditions
print(f"\nAvailable conditions ({Z_adata.n_vars}):")
# Parse celltype and timepoint
_tp_re = re.compile(r'(\d+somites)$')
celltypes = sorted(set(_tp_re.sub('', c).rstrip('_') for c in Z_adata.var_names))
timepoints = sorted(set(m.group(1) for c in Z_adata.var_names
                        if (m := _tp_re.search(c))),
                    key=lambda t: int(t.replace('somites','')))
print(f"  Celltypes ({len(celltypes)}): {celltypes[:5]} ...")
print(f"  Timepoints ({len(timepoints)}): {timepoints}")

# Reliable mask
reliable = Z_adata.var["reliable"].values

# %% Core query function
def query_parts_list(
    Z_adata,
    celltype: str,
    timepoint: str,
    top_n: int = 50,
    min_zscore: float = 2.0,
    peak_type_filter: str = None,
    reliable_only: bool = True,
) -> pd.DataFrame:
    """
    Return top peaks for (celltype, timepoint) ranked by specificity z-score.

    Parameters
    ----------
    Z_adata       : AnnData with specificity z-scores as .X
    celltype      : e.g. "heart_myocardium"
    timepoint     : e.g. "20somites"
    top_n         : number of peaks to return
    min_zscore    : minimum z-score threshold
    peak_type_filter : optional filter ("promoter", "intronic", "intergenic", "exonic")
    reliable_only : if True, skip conditions flagged as low-confidence (n_cells < 20)
    """
    condition = f"{celltype}_{timepoint}"
    if condition not in cond_to_idx:
        available = [c for c in Z_adata.var_names if c.startswith(celltype)]
        raise ValueError(f"Condition '{condition}' not found.\n"
                         f"Available for '{celltype}': {available}")

    col_idx = cond_to_idx[condition]

    if reliable_only and not Z_adata.var.loc[condition, "reliable"]:
        print(f"  WARNING: '{condition}' flagged as low-confidence (n_cells < 20)")

    zscores = np.array(Z_adata.X[:, col_idx]).ravel()

    obs = Z_adata.obs.copy()
    obs["specificity_zscore"] = zscores
    obs["condition"]          = condition
    obs["celltype_query"]     = celltype
    obs["timepoint_query"]    = timepoint

    # Filters
    result = obs[obs["specificity_zscore"] >= min_zscore].copy()
    if peak_type_filter:
        result = result[result["peak_type"] == peak_type_filter]

    result = result.nlargest(top_n, "specificity_zscore")

    # Select display columns
    keep_cols = ["specificity_zscore", "condition", "peak_type",
                 "chrom", "start", "end",
                 "associated_gene", "linked_gene", "nearest_gene", "distance_to_tss",
                 "leiden_coarse", "celltype", "celltype_contrast",
                 "timepoint", "timepoint_contrast"]
    keep_cols = [c for c in keep_cols if c in result.columns]
    result = result[keep_cols]
    result.index.name = "peak_id"

    return result


def list_conditions(Z_adata) -> pd.DataFrame:
    """Return a table of all available conditions with metadata."""
    var = Z_adata.var[["annotation_ML_coarse", "dev_stage", "n_cells", "reliable"]].copy()
    var.index.name = "condition"
    return var

# %% Print available conditions
print("\nCondition reliability summary:")
cond_df = list_conditions(Z_adata)
print(f"  Reliable: {cond_df['reliable'].sum()}/{len(cond_df)}")
print(f"  Low-confidence: {(~cond_df['reliable']).sum()}")
print("\n  Low-confidence conditions:")
print(cond_df[~cond_df['reliable']][['n_cells']].to_string())

# %% Example queries
QUERIES = [
    ("heart_myocardium",   "20somites", "Cardiac regulatory elements — late somitogenesis"),
    ("neural",             "10somites", "Early neural specification"),
    ("neural_crest",       "15somites", "Neural crest differentiation"),
    ("somites",            "5somites",  "Early somite specification"),
    ("endoderm",           "20somites", "Late endoderm maturation"),
]

os.makedirs(OUTDIR, exist_ok=True)

for celltype, timepoint, description in QUERIES:
    print(f"\n{'='*60}")
    print(f"Query: {celltype} × {timepoint}")
    print(f"  {description}")
    print(f"{'='*60}")

    try:
        result = query_parts_list(Z_adata, celltype, timepoint, top_n=50)
    except ValueError as e:
        print(f"  SKIP: {e}")
        continue

    condition = f"{celltype}_{timepoint}"
    n_total = (np.array(Z_adata.X[:, cond_to_idx[condition]]) >= 2.0).sum()
    print(f"  Peaks with z ≥ 2: {n_total:,}")
    print(f"  Top 20 results:")
    print(result.head(20)[["specificity_zscore", "peak_type",
                            "chrom", "start", "end",
                            "associated_gene", "leiden_coarse"]].to_string())

    # Save
    out_path = f"{OUTDIR}/query_{condition}_top50.csv"
    result.to_csv(out_path)
    print(f"\n  Saved: {out_path}")

print("\nDone.")
