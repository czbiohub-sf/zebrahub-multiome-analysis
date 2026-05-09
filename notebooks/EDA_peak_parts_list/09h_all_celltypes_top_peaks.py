# %% Script 09h: Top-50 peaks for ALL 31 celltypes
#
# Extracts top-50 V3-specific peaks per celltype for all 31 reliable celltypes
# (excluding primordial_germ_cells). Produces a single CSV for the web portal.
#
# Env: single-cell-base

import os, time
import numpy as np
import pandas as pd
import anndata as ad

print("=== Script 09h: All-31-celltypes Top-50 Peaks ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
V3_ZMAT     = f"{OUTDIR}/V3_specificity_matrix_celltype_level.h5ad"

TOP_N = 200

# %% Load data
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
master = ad.read_h5ad(MASTER_H5AD)
obs = master.obs.copy()
print(f"  {master.shape}  ({time.time()-t0:.1f}s)")

print("Loading V3 z-score matrix ...", flush=True)
z_adata = ad.read_h5ad(V3_ZMAT)
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)
print(f"  {Z_ct.shape}  ({len(ct_names)} celltypes)")

# %% Extract top-50 per celltype
print(f"\nExtracting top-{TOP_N} peaks for all {len(ct_names)} celltypes ...", flush=True)

all_rows = []
for ct in ct_names:
    ct_col = ct_names.index(ct)
    z_col = Z_ct[:, ct_col]
    top_idx = np.argsort(z_col)[::-1][:TOP_N]

    for rank, idx in enumerate(top_idx, 1):
        peak_obs = obs.iloc[idx]
        peak_id = obs.index[idx]

        # Gene annotation: linked > nearest > coords
        linked  = str(peak_obs.get("linked_gene", ""))
        nearest = str(peak_obs.get("nearest_gene", ""))
        associated = str(peak_obs.get("associated_gene", ""))
        if linked in ("", "nan", "None"):
            linked = ""
        if nearest in ("", "nan", "None"):
            nearest = ""

        all_rows.append({
            "celltype": ct,
            "rank": rank,
            "peak_id": peak_id,
            "chrom": str(peak_obs["chrom"]),
            "start": int(peak_obs["start"]),
            "end": int(peak_obs["end"]),
            "V3_zscore": float(z_col[idx]),
            "peak_type": str(peak_obs.get("peak_type", "")),
            "length": int(peak_obs.get("length", 0)),
            "linked_gene": linked,
            "link_score": float(peak_obs.get("link_score", 0)) if pd.notna(peak_obs.get("link_score")) else 0.0,
            "nearest_gene": nearest,
            "distance_to_tss": float(peak_obs.get("distance_to_tss", 0)) if pd.notna(peak_obs.get("distance_to_tss")) else 0.0,
            "associated_gene": str(peak_obs.get("associated_gene", "")),
            "association_type": str(peak_obs.get("association_type", "")),
            "leiden_coarse": str(peak_obs.get("leiden_coarse", "")),
        })

    print(f"  {ct}: z-range [{z_col[top_idx[-1]]:.1f}, {z_col[top_idx[0]]:.1f}]")

df = pd.DataFrame(all_rows)
out_csv = f"{OUTDIR}/V3_all_celltypes_top200_peaks.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(f"  {len(df)} rows ({len(ct_names)} celltypes × {TOP_N} peaks)")
print(f"  Columns: {list(df.columns)}")

print(f"\nDone. End: {time.strftime('%c')}")
