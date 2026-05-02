"""
Build a slim peak metadata + V3 z-score cache for fast marker-gene queries.

One-time script: joins master h5ad obs + V3 specificity matrix + tau metrics
into a single parquet file that the query utility (marker_gene_peaks.py)
loads in ~1 second.

Output: notebooks/EDA_peak_parts_list/outputs/V3/peak_metadata_cache.parquet

Env: any single-cell-base / gReLu env with anndata + pandas.
"""

import os
import time
import numpy as np
import pandas as pd
import anndata as ad

BASE   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO   = f"{BASE}/zebrahub-multiome-analysis"
V3_DIR = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
V3_ZMAT     = f"{V3_DIR}/V3_specificity_matrix_celltype_level.h5ad"
V3_METRICS  = f"{V3_DIR}/V3_peak_specificity_metrics.csv"
OUT_PARQUET = f"{V3_DIR}/peak_metadata_cache.parquet"

print("=== Build peak metadata cache ===")
print(f"Start: {time.strftime('%c')}")

# 1. Master h5ad — only obs (skip the big .X)
print("\nLoading master h5ad obs ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD, backed="r")
obs = adata.obs.copy()
print(f"  obs shape: {obs.shape}  ({time.time()-t0:.1f}s)")
print(f"  obs columns: {list(obs.columns)}")
adata.file.close()

# Slim down obs to columns we care about
keep_cols = [c for c in [
    "chrom", "start", "end", "peak_type",
    "nearest_gene", "distance_to_tss", "linked_gene", "associated_gene",
    "leiden_coarse",
] if c in obs.columns]

obs_slim = obs[keep_cols].copy()
obs_slim.index.name = "peak_id"

# Stringify gene columns (drop pandas Categorical)
for col in ["nearest_gene", "linked_gene", "associated_gene"]:
    if col in obs_slim.columns:
        obs_slim[col] = obs_slim[col].astype(str).replace({"nan": "", "None": ""})

# Cast coords
if "chrom" in obs_slim.columns:
    obs_slim["chrom"] = obs_slim["chrom"].astype(str)
if "start" in obs_slim.columns:
    obs_slim["start"] = obs_slim["start"].astype(int)
if "end" in obs_slim.columns:
    obs_slim["end"] = obs_slim["end"].astype(int)
    obs_slim["length"] = obs_slim["end"] - obs_slim["start"]

# 2. V3 z-score matrix
print("\nLoading V3 z-score matrix ...", flush=True)
t0 = time.time()
z = ad.read_h5ad(V3_ZMAT)
Z = np.array(z.X)        # (640830, 31)
ct_names = list(z.var_names)
print(f"  Z shape: {Z.shape}, celltypes: {len(ct_names)}  ({time.time()-t0:.1f}s)")

# Z is indexed by peak — z.obs.index should match obs.index. Sanity check.
assert list(z.obs.index) == list(obs.index), "Index mismatch between master and V3 matrix"

# Per-peak top1, top2, top3 by z-score (vectorized)
print("Computing top-3 celltype per peak ...", flush=True)
t0 = time.time()
order = np.argsort(-Z, axis=1)            # descending, (640830, 31)
top1_idx = order[:, 0]
top2_idx = order[:, 1]
top3_idx = order[:, 2]

ct_arr = np.array(ct_names)
top1_z = Z[np.arange(Z.shape[0]), top1_idx]
top2_z = Z[np.arange(Z.shape[0]), top2_idx]
top3_z = Z[np.arange(Z.shape[0]), top3_idx]
print(f"  Done  ({time.time()-t0:.1f}s)")

obs_slim["top1_celltype"] = ct_arr[top1_idx]
obs_slim["top1_z"]        = top1_z.round(3)
obs_slim["top2_celltype"] = ct_arr[top2_idx]
obs_slim["top2_z"]        = top2_z.round(3)
obs_slim["top3_celltype"] = ct_arr[top3_idx]
obs_slim["top3_z"]        = top3_z.round(3)
obs_slim["max_z"]         = top1_z.round(3)   # alias

# Add 31 wide z-score columns (z_<celltype>) for full matrix queries
for i, ct in enumerate(ct_names):
    obs_slim[f"z_{ct}"] = Z[:, i].round(3)

# 3. Tau / gini / max_accessibility metrics
print("\nLoading V3 peak specificity metrics ...", flush=True)
metrics = pd.read_csv(V3_METRICS, index_col=0)
metrics.index.name = "peak_id"
print(f"  metrics shape: {metrics.shape}, cols: {list(metrics.columns)}")
obs_slim = obs_slim.join(metrics, how="left")

# Save
print(f"\nWriting parquet → {OUT_PARQUET}", flush=True)
t0 = time.time()
obs_slim.reset_index().to_parquet(OUT_PARQUET, index=False, compression="snappy")
print(f"  Wrote {os.path.getsize(OUT_PARQUET)/1e6:.1f} MB  ({time.time()-t0:.1f}s)")

# Quick summary
print("\nCache columns (first 20):")
print(list(obs_slim.reset_index().columns)[:20])
print("\nN peaks with linked_gene:", (obs_slim["linked_gene"] != "").sum())
print("N peaks with nearest_gene:", (obs_slim["nearest_gene"] != "").sum())
print(f"\nDone. End: {time.strftime('%c')}")
