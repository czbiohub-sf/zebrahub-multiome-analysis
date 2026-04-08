# %% [markdown]
# # Step 01b: Compute Per-Condition Specificity Matrix — V2 (shrinkage-regularized)
#
# Same leave-one-out z-score as V1, but with shrinkage regularization applied to
# unreliable conditions (n_cells < MIN_CELLS = 20) BEFORE computing z-scores.
#
# Shrinkage formula (empirical Bayes, per condition j):
#
#   M_shrunk[i, j] = (n_j * M[i, j] + alpha * global_mean[i]) / (n_j + alpha)
#
#   where alpha = 20  (pseudo-cell-count, equal to reliability threshold)
#         global_mean[i] = mean(M[i, :]) across all 190 conditions
#
# Effect by cell count:
#   n =  6 cells  → 23% original, 77% prior  (e.g. epidermis 30somites)
#   n = 12 cells  → 37% original, 63% prior  (e.g. PGC 0somites)
#   n = 19 cells  → 49% original, 51% prior  (e.g. PGC 5somites)
#   n = 20 cells  → 50% original, 50% prior  (reliability threshold)
#
# Why this helps:
#   With n < 20 cells, the pseudobulk mean is noisy (SE ∝ 1/√n). Shrinkage
#   pulls the noisy mean toward the global average, reducing false-positive
#   z-scores while preserving strong, genuine signals like PGC germ-cell loci.
#
# Reliable conditions (n ≥ 20) are untouched; their z-scores change only
# negligibly via the shared row_sum (effect: Δz < 0.01 per peak).
#
# Outputs:
#   outputs/specificity_matrix_v2.h5ad   — shrinkage-corrected Z (float32)
#   outputs/specificity_summary_v2.csv   — per-peak summary stats
#   outputs/v1_v2_comparison.csv         — change in max_zscore per peak

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

MIN_CELLS = 20
ALPHA     = 20   # shrinkage pseudo-cell-count (equal to reliability threshold)

# %% Load h5ad
print(f"\nLoading {os.path.basename(INPUT)} ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(INPUT)
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")

# %% Extract log_norm matrix
if "log_norm" in adata.layers:
    M = adata.layers["log_norm"]
    print("Using log_norm layer")
else:
    M = adata.X
    print("Using .X")

if sp.issparse(M):
    print("Converting sparse to dense ...", flush=True)
    M = M.toarray()
else:
    M = np.array(M, dtype=np.float64)

print(f"Matrix shape: {M.shape}, dtype: {M.dtype}")
print(f"Value range : {M.min():.3f} – {M.max():.3f}")

# %% Identify unreliable conditions
n_cells      = adata.var["n_cells"].values          # (190,)
reliable_mask = n_cells >= MIN_CELLS                 # (190,) bool
n_reliable   = reliable_mask.sum()
unreliable_idx = np.where(~reliable_mask)[0]

print(f"\nReliable conditions (n ≥ {MIN_CELLS}): {n_reliable}/{adata.n_vars}")
print(f"Unreliable conditions to shrink  : {len(unreliable_idx)}")
for j in unreliable_idx:
    print(f"  [{j:3d}]  {adata.var_names[j]:40s}  n_cells={n_cells[j]}")

# %% Apply shrinkage to unreliable conditions
print(f"\nApplying shrinkage (alpha={ALPHA}) to {len(unreliable_idx)} conditions ...", flush=True)
t0 = time.time()

# Global mean per peak across all 190 conditions (the "prior")
global_mean = M.mean(axis=1)   # (640830,)

M_shrunk = M.copy()            # leave reliable columns untouched
for j in unreliable_idx:
    nc = float(n_cells[j])
    weight_data  = nc / (nc + ALPHA)
    weight_prior = ALPHA / (nc + ALPHA)
    M_shrunk[:, j] = weight_data * M[:, j] + weight_prior * global_mean
    print(f"  {adata.var_names[j]:40s}  n={int(nc):2d}  "
          f"data_weight={weight_data:.3f}  prior_weight={weight_prior:.3f}")

print(f"  Shrinkage applied in {time.time()-t0:.1f}s")

# Diagnostic: how much did the unreliable columns change?
delta = M_shrunk[:, unreliable_idx] - M[:, unreliable_idx]
print(f"\n  Mean |delta| across shrunk conditions: {np.abs(delta).mean():.4f}")
print(f"  Max  |delta| across shrunk conditions: {np.abs(delta).max():.4f}")

del delta, global_mean
gc.collect()

# %% Compute leave-one-out z-score on M_shrunk (fully vectorized)
print("\nComputing leave-one-out specificity z-scores (V2) ...", flush=True)
t0 = time.time()

n = M_shrunk.shape[1]   # 190

row_sum    = M_shrunk.sum(axis=1)
row_sq_sum = (M_shrunk ** 2).sum(axis=1)

mean_other = (row_sum[:, None] - M_shrunk) / (n - 1)
var_other  = (row_sq_sum[:, None] - M_shrunk**2) / (n - 1) - mean_other**2
var_other  = np.maximum(var_other, 0.0)
std_other  = np.sqrt(var_other + 1e-10)

Z = (M_shrunk - mean_other) / std_other    # (640830, 190)
Z = Z.astype(np.float32)

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")
print(f"  Z range: {Z.min():.2f} – {Z.max():.2f}")

# %% Sanity check: argmax agreement with raw accessibility
print("\nSanity check: argmax(Z_v2) vs argmax(log_norm) ...")
Z_argmax = Z.argmax(axis=1)
if "log_norm" in adata.layers:
    raw = adata.layers["log_norm"]
    if sp.issparse(raw):
        raw = raw.toarray()
    raw_argmax = np.array(raw).argmax(axis=1)
    agreement = (Z_argmax == raw_argmax).mean()
    print(f"  argmax agreement: {agreement:.1%}")

del row_sum, row_sq_sum, mean_other, var_other, std_other, M_shrunk
gc.collect()

# %% Build V2 summary CSV
print("\nBuilding per-peak specificity summary (V2) ...", flush=True)

max_z_idx    = Z.argmax(axis=1)
max_z_val    = Z.max(axis=1)
max_cond     = adata.var_names[max_z_idx]
n_cond_z2    = (Z >= 2.0).sum(axis=1)
n_cond_z4    = (Z >= 4.0).sum(axis=1)

summary_v2 = pd.DataFrame({
    "peak_id":          adata.obs_names,
    "max_condition":    max_cond,
    "max_zscore":       max_z_val.astype(np.float32),
    "n_conditions_z2":  n_cond_z2.astype(np.int16),
    "n_conditions_z4":  n_cond_z4.astype(np.int16),
})
for col in ["chrom", "start", "end", "peak_type", "celltype", "celltype_contrast",
            "timepoint", "timepoint_contrast", "leiden_coarse", "associated_gene"]:
    if col in adata.obs.columns:
        summary_v2[col] = adata.obs[col].values

summary_v2_path = f"{OUTDIR}/specificity_summary_v2.csv"
summary_v2.to_csv(summary_v2_path, index=False)
print(f"  Saved: {summary_v2_path}  ({len(summary_v2):,} rows)")

print(f"\n  Peaks with max z ≥ 2: {(summary_v2['max_zscore'] >= 2).sum():,} ({(summary_v2['max_zscore'] >= 2).mean():.1%})")
print(f"  Peaks with max z ≥ 4: {(summary_v2['max_zscore'] >= 4).sum():,} ({(summary_v2['max_zscore'] >= 4).mean():.1%})")
print(f"  Peaks with max z ≥ 8: {(summary_v2['max_zscore'] >= 8).sum():,} ({(summary_v2['max_zscore'] >= 8).mean():.1%})")

# %% Compare V1 vs V2
print("\nComparing V1 vs V2 ...", flush=True)
v1_path = f"{OUTDIR}/specificity_summary_v1.csv"
if os.path.exists(v1_path):
    summary_v1 = pd.read_csv(v1_path, usecols=["peak_id", "max_zscore", "max_condition"])
    comp = summary_v1.rename(columns={"max_zscore": "max_z_v1", "max_condition": "max_cond_v1"}).merge(
        summary_v2[["peak_id", "max_zscore", "max_condition"]].rename(
            columns={"max_zscore": "max_z_v2", "max_condition": "max_cond_v2"}),
        on="peak_id"
    )
    comp["delta_z"]      = comp["max_z_v2"] - comp["max_z_v1"]
    comp["cond_changed"] = comp["max_cond_v1"] != comp["max_cond_v2"]

    comp_path = f"{OUTDIR}/v1_v2_comparison.csv"
    comp.to_csv(comp_path, index=False)
    print(f"  Saved: {comp_path}")

    print(f"\n  Peaks where best condition changed: {comp['cond_changed'].sum():,} "
          f"({comp['cond_changed'].mean():.2%})")
    print(f"  Mean Δz (V2 − V1): {comp['delta_z'].mean():+.4f}")
    print(f"  Max Δz (V2 − V1) : {comp['delta_z'].max():+.4f}  "
          f"(peak={comp.loc[comp['delta_z'].idxmax(),'peak_id']})")
    print(f"  Min Δz (V2 − V1) : {comp['delta_z'].min():+.4f}  "
          f"(peak={comp.loc[comp['delta_z'].idxmin(),'peak_id']})")

    # Summarize by condition: how much did each unreliable condition's peaks change?
    print("\n  V1 → V2 change focused on unreliable conditions:")
    for j in unreliable_idx:
        cond = adata.var_names[j]
        was_best_v1 = comp["max_cond_v1"] == cond
        was_best_v2 = comp["max_cond_v2"] == cond
        print(f"  {cond:40s}  "
              f"peaks_best_v1={was_best_v1.sum():5,}  "
              f"peaks_best_v2={was_best_v2.sum():5,}  "
              f"Δ={was_best_v2.sum()-was_best_v1.sum():+d}")
else:
    print("  V1 summary not found, skipping comparison")

# %% Save V2 specificity matrix as h5ad
print("\nSaving V2 specificity matrix h5ad ...", flush=True)
t0 = time.time()

Z_adata = ad.AnnData(
    X   = Z,
    obs = adata.obs.copy(),
    var = adata.var.copy(),
)
Z_adata.var["reliable"] = reliable_mask
Z_adata.uns["shrinkage_alpha"]      = ALPHA
Z_adata.uns["shrinkage_min_cells"]  = MIN_CELLS
Z_adata.uns["shrinkage_conditions"] = list(adata.var_names[unreliable_idx])

for key in ["X_umap", "X_umap_2D", "X_umap_3D", "X_pca"]:
    if key in adata.obsm:
        Z_adata.obsm[key] = adata.obsm[key].copy()

out_path = f"{OUTDIR}/specificity_matrix_v2.h5ad"
Z_adata.write_h5ad(out_path, compression="gzip")
print(f"  Saved: {out_path}  ({os.path.getsize(out_path)/1e9:.2f} GB)  ({time.time()-t0:.1f}s)")

# %% Also update the "current" symlink so downstream scripts pick up V2
# (we don't overwrite the original; keep specificity_matrix.h5ad as V1)
print("\nNote: V1 is preserved at specificity_matrix_v1.h5ad")
print("      V2 is at                specificity_matrix_v2.h5ad")
print("      Downstream scripts still point to specificity_matrix.h5ad (= V1)")
print("      Update SPEC_H5AD in downstream scripts to use V2 when ready.")
print("\nDone.")
