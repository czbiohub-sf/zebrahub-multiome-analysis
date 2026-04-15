# %% Script 09k: Temporal accessibility profiles for ALL 31 celltypes (top-200)
#
# For each celltype's top-200 peaks, extract the 6-timepoint accessibility
# vector from the master pseudobulk matrix.
#
# Output: outputs/V3/V3_all_celltypes_top200_temporal_profiles.csv
#
# Env: single-cell-base

import os, re, time
import numpy as np
import pandas as pd
import anndata as ad

print("=== Script 09k: All-celltypes Temporal Profiles (top-200) ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO   = f"{BASE}/zebrahub-multiome-analysis"
V3_DIR = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
TOP200_CSV  = f"{V3_DIR}/V3_all_celltypes_top200_peaks.csv"

MIN_CELLS = 20
TIMEPOINT_ORDER = ["0somites", "5somites", "10somites", "15somites", "20somites", "30somites"]
TP_INT = {tp: int(tp.replace("somites", "")) for tp in TIMEPOINT_ORDER}

# %% Load data
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
M = np.array(adata.X, dtype=np.float64)
obs = adata.obs
print(f"  {adata.shape}  ({time.time()-t0:.1f}s)")

# Parse conditions
def parse_condition(cond):
    m = re.search(r"(\d+somites)$", cond)
    if not m:
        return cond, ""
    tp = m.group(1)
    ct = cond[:-(len(tp)+1)]
    return ct, tp

cond_meta = pd.DataFrame(
    [parse_condition(c) for c in adata.var_names],
    columns=["celltype", "timepoint"],
    index=adata.var_names,
)
cond_meta["n_cells"] = adata.var["n_cells"].values
cond_meta["reliable"] = cond_meta["n_cells"] >= MIN_CELLS
reliable_groups = cond_meta[cond_meta["reliable"]].index.tolist()

# Build (celltype, timepoint) → column index lookup
ct_tp_col = {}
for col in adata.var_names:
    ct, tp = parse_condition(col)
    if col in reliable_groups:
        ct_tp_col[(ct, tp)] = list(adata.var_names).index(col)

# %% Load top-200 peaks
print("Loading top-200 peaks ...", flush=True)
top200 = pd.read_csv(TOP200_CSV)
print(f"  {len(top200)} rows, {top200['celltype'].nunique()} celltypes")

# %% Extract temporal profiles
print("Extracting temporal profiles ...", flush=True)

temporal_rows = []
for _, row in top200.iterrows():
    ct = row["celltype"]
    peak_id = row["peak_id"]
    peak_iloc = obs.index.get_loc(peak_id)
    peak_row = M[peak_iloc]

    for tp in TIMEPOINT_ORDER:
        key = (ct, tp)
        if key in ct_tp_col:
            accessibility = peak_row[ct_tp_col[key]]
            n_cells = cond_meta.loc[f"{ct}_{tp}", "n_cells"] if f"{ct}_{tp}" in cond_meta.index else 0
        else:
            accessibility = np.nan
            n_cells = 0

        temporal_rows.append({
            "celltype": ct,
            "peak_id": peak_id,
            "rank": int(row["rank"]),
            "V3_zscore": row["V3_zscore"],
            "linked_gene": str(row.get("linked_gene", "")),
            "nearest_gene": str(row.get("nearest_gene", "")),
            "timepoint": tp,
            "tp_int": TP_INT[tp],
            "accessibility": accessibility,
            "n_cells": int(n_cells),
            "reliable": n_cells >= MIN_CELLS,
        })

temporal_df = pd.DataFrame(temporal_rows)
out_csv = f"{V3_DIR}/V3_all_celltypes_top200_temporal_profiles.csv"
temporal_df.to_csv(out_csv, index=False)

print(f"\nSaved: {out_csv}")
print(f"  {len(temporal_df)} rows ({top200['celltype'].nunique()} celltypes × "
      f"{len(top200) // top200['celltype'].nunique()} peaks × {len(TIMEPOINT_ORDER)} timepoints)")
print(f"  Columns: {list(temporal_df.columns)}")
print(f"\nDone. End: {time.strftime('%c')}")
