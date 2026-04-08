# %% [markdown]
# # Step 01: Compute Per-Condition Specificity Matrix
#
# For each peak and each (celltype × timepoint) condition, compute a
# leave-one-out z-score measuring how specifically accessible that peak
# is in the queried condition compared to all others:
#
#   Z[peak, condition] = (M[i,j] - mean(M[i, ~j])) / std(M[i, ~j])
#
# This generalizes the existing `celltype_contrast` scalar (notebook 09,
# lines 311-322) to all 190 conditions simultaneously, enabling ranked
# "parts list" queries for any (celltype, timepoint).
#
# Input:
#   peaks_by_ct_tp_master_anno.h5ad  (640,830 peaks × 190 conditions)
#     .X  = log-transformed, median-scaled accessibility (log_norm layer)
#     .var has: annotation_ML_coarse, dev_stage, n_cells
#     .obs has: chrom, start, end, peak_type, leiden_coarse, associated_gene, etc.
#
# Outputs:
#   outputs/specificity_matrix.h5ad  — Z as .X (float32), same obs/var as input
#   outputs/specificity_summary.csv  — per-peak summary stats

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

print(f"anndata {ad.__version__}, numpy {np.__version__}")

# %% Paths
BASE   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO   = f"{BASE}/zebrahub-multiome-analysis"
INPUT  = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
OUTDIR = f"{REPO}/notebooks/EDA_peak_parts_list/outputs"
os.makedirs(OUTDIR, exist_ok=True)

MIN_CELLS = 20  # flag conditions with fewer cells as low-confidence

# %% Load h5ad
print(f"\nLoading {os.path.basename(INPUT)} ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(INPUT)
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")
print(f"  obs columns: {list(adata.obs.columns[:10])} ...")
print(f"  var sample: {adata.var_names[:5].tolist()}")

# %% Extract matrix — use log_norm layer (log-transformed, median-scaled)
# This is identical to .X in the master_anno object but we're explicit about it.
if "log_norm" in adata.layers:
    M = adata.layers["log_norm"]
    print("\nUsing log_norm layer")
else:
    M = adata.X
    print("\nUsing .X (log_norm layer not found)")

# Convert sparse to dense if needed
if sp.issparse(M):
    print("Converting sparse to dense ...", flush=True)
    M = M.toarray()
else:
    M = np.array(M)

print(f"Matrix shape: {M.shape}, dtype: {M.dtype}")
print(f"Value range: {M.min():.3f} – {M.max():.3f}")

# %% Flag unreliable conditions (n_cells < MIN_CELLS)
n_cells = adata.var["n_cells"].values
reliable_mask = n_cells >= MIN_CELLS
n_reliable = reliable_mask.sum()
print(f"\nReliable conditions (n_cells >= {MIN_CELLS}): {n_reliable}/{adata.n_vars}")
print(f"Flagged as low-confidence: {(~reliable_mask).sum()}")
flagged = adata.var_names[~reliable_mask].tolist()
if flagged:
    print(f"  Flagged: {flagged[:10]}{'...' if len(flagged) > 10 else ''}")

# %% Compute leave-one-out z-score (fully vectorized)
print("\nComputing leave-one-out specificity z-scores ...", flush=True)
t0 = time.time()

n = M.shape[1]  # 190 conditions

# Precompute row statistics (one pass over data)
row_sum    = M.sum(axis=1)           # (640830,)
row_sq_sum = (M ** 2).sum(axis=1)    # (640830,)

# Leave-one-out mean and variance (broadcast over all conditions simultaneously)
mean_other = (row_sum[:, None] - M) / (n - 1)           # (640830, 190)
var_other  = (row_sq_sum[:, None] - M**2) / (n - 1) - mean_other**2  # (640830, 190)
var_other  = np.maximum(var_other, 0.0)                  # clip small negatives from float arithmetic
std_other  = np.sqrt(var_other + 1e-10)                  # (640830, 190)

Z = (M - mean_other) / std_other                         # (640830, 190)

# Cast to float32 before saving
Z = Z.astype(np.float32)

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")
print(f"  Z range: {Z.min():.2f} – {Z.max():.2f}")

# Free intermediates
del row_sum, row_sq_sum, mean_other, var_other, std_other, M
gc.collect()

# %% Sanity check: z-score argmax should match accessibility argmax for well-characterized peaks
# The best condition by z-score should typically agree with the best condition by raw accessibility
print("\nSanity check: comparing argmax(Z) vs argmax(log_norm) ...")
Z_argmax = Z.argmax(axis=1)
if "log_norm" in adata.layers:
    raw_layer = adata.layers["log_norm"]
    if sp.issparse(raw_layer):
        raw_layer = raw_layer.toarray()
    raw_argmax = raw_layer.argmax(axis=1)
    agreement = (Z_argmax == raw_argmax).mean()
    print(f"  argmax agreement: {agreement:.1%}")

# %% Build summary CSV
print("\nBuilding per-peak specificity summary ...", flush=True)
max_z_idx = Z.argmax(axis=1)
max_z_val = Z.max(axis=1)
max_condition = adata.var_names[max_z_idx]
n_cond_z2 = (Z >= 2.0).sum(axis=1)
n_cond_z4 = (Z >= 4.0).sum(axis=1)

summary = pd.DataFrame({
    "peak_id":           adata.obs_names,
    "max_condition":     max_condition,
    "max_zscore":        max_z_val.astype(np.float32),
    "n_conditions_z2":   n_cond_z2.astype(np.int16),
    "n_conditions_z4":   n_cond_z4.astype(np.int16),
})

# Merge key obs metadata
for col in ["chrom", "start", "end", "peak_type", "celltype", "celltype_contrast",
            "timepoint", "timepoint_contrast", "leiden_coarse", "associated_gene"]:
    if col in adata.obs.columns:
        summary[col] = adata.obs[col].values

summary_path = f"{OUTDIR}/specificity_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"  Saved: {summary_path}  ({len(summary):,} rows)")

# Quick distribution report
print(f"\n  Peaks with max z ≥ 2:  {(summary['max_zscore'] >= 2).sum():,} ({(summary['max_zscore'] >= 2).mean():.1%})")
print(f"  Peaks with max z ≥ 4:  {(summary['max_zscore'] >= 4).sum():,} ({(summary['max_zscore'] >= 4).mean():.1%})")
print(f"  Peaks with max z ≥ 8:  {(summary['max_zscore'] >= 8).sum():,} ({(summary['max_zscore'] >= 8).mean():.1%})")

# %% Save specificity matrix as h5ad
print("\nSaving specificity matrix h5ad ...", flush=True)
t0 = time.time()

# Build new AnnData with same obs/var metadata
Z_adata = ad.AnnData(
    X   = Z,
    obs = adata.obs.copy(),
    var = adata.var.copy(),
)

# Add reliable flag to var
Z_adata.var["reliable"] = reliable_mask

# Copy UMAP coordinates for downstream visualization
for key in ["X_umap", "X_umap_2D", "X_umap_3D", "X_pca"]:
    if key in adata.obsm:
        Z_adata.obsm[key] = adata.obsm[key].copy()

out_path = f"{OUTDIR}/specificity_matrix.h5ad"
Z_adata.write_h5ad(out_path, compression="gzip")
print(f"  Saved: {out_path}  ({os.path.getsize(out_path)/1e9:.2f} GB)  ({time.time()-t0:.1f}s)")

print("\nDone.")
