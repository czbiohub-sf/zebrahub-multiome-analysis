# %% Script 09e-pre: RNA pseudobulk — same pipeline as ATAC
#
# Builds a genes × conditions pseudobulk matrix from single-cell RNA counts,
# using the EXACT same pipeline as the ATAC pseudobulk (peaks_pb_preprocessing.py):
#   1. Sum raw counts per (celltype × timepoint) group
#   2. Compute per-group total coverage (sum of per-cell nCount across all cells in group)
#   3. Median-scale: multiply by median(all_group_coverages) / this_group_coverage
#   4. log1p
#
# Then computes celltype-level leave-one-out z-scores (same as ATAC V3).
#
# Outputs (outputs/V3/):
#   rna_by_ct_tp_pseudobulked.h5ad           — genes × conditions (log_norm + layers)
#   rna_specificity_matrix_celltype_level.h5ad — genes × celltypes (z-scores)
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu

import os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib
matplotlib.use("Agg")

# ── Publication figure settings ──
import matplotlib as _mpl
_mpl.rcParams.update(_mpl.rcParamsDefault)
_mpl.rcParams['font.family'] = 'Arial'
_mpl.rcParams["pdf.fonttype"] = 42
_mpl.rcParams["ps.fonttype"]  = 42
import seaborn as _sns
_sns.set(style="whitegrid", context="paper")
_mpl.rcParams["savefig.dpi"]  = 300
# ────────────────────────────────────────────────────────────────────────────────

print("=== Script 09e-pre: RNA Pseudobulk Pipeline ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUT_DIR = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
os.makedirs(OUT_DIR, exist_ok=True)

RNA_SC_H5AD = (f"{BASE}/data/processed_data/01_Signac_processed/"
               "integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")

# %% Load single-cell RNA
print("\nLoading single-cell RNA ...", flush=True)
t0 = time.time()
rna = ad.read_h5ad(RNA_SC_H5AD)
print(f"  Shape: {rna.shape}  ({time.time()-t0:.1f}s)")
print(f"  Layers: {list(rna.layers.keys())}")

# Raw counts — shape: (n_cells, n_genes)
X_raw = rna.layers["counts"]
if sp.issparse(X_raw):
    X_raw = X_raw.toarray()
X_raw = X_raw.astype(np.float64)

n_cells, n_genes = X_raw.shape
gene_names = list(rna.var_names)
celltype_col = rna.obs["annotation_ML_coarse"].astype(str)
timepoint_col = rna.obs["dev_stage"].astype(str)

# Per-cell total counts (needed for coverage calculation)
per_cell_total = X_raw.sum(axis=1)   # (n_cells,)
print(f"  Per-cell total counts: median={np.median(per_cell_total):.0f}, "
      f"range=[{per_cell_total.min():.0f}, {per_cell_total.max():.0f}]")

# %% Build groups: (celltype × timepoint)
# Use integer positional indices for numpy indexing
timepoint_order = ["0somites", "5somites", "10somites", "15somites", "20somites", "30somites"]

def tp_rank(tp):
    try: return timepoint_order.index(tp)
    except ValueError: return 99

# Build (ct, tp) → array of integer positions
obs_ct  = rna.obs["annotation_ML_coarse"].astype(str).values
obs_tp  = rna.obs["dev_stage"].astype(str).values
all_cts = sorted(set(obs_ct))
all_tps = sorted(set(obs_tp), key=tp_rank)

group_keys_all = []
for ct in all_cts:
    for tp in all_tps:
        mask = (obs_ct == ct) & (obs_tp == tp)
        pos  = np.where(mask)[0]
        if len(pos) > 0:
            group_keys_all.append(((ct, tp), pos))

print(f"\n  Total groups: {len(group_keys_all)}")

# %% Pseudobulk: sum + coverage per group
print("Computing pseudobulk sums ...", flush=True)

group_labels = []      # (celltype, timepoint) tuples
group_ncells = []      # number of cells per group
group_coverage = []    # sum of per-cell totals in group
sum_matrix_rows = []   # one row per group, shape (n_genes,)

MIN_CELLS = 20

for (ct, tp), cell_positions in group_keys_all:
    n = len(cell_positions)
    if n < MIN_CELLS:
        print(f"  Skipping {ct}_{tp}: n_cells={n} < {MIN_CELLS}")
        continue

    row_sum = X_raw[cell_positions, :].sum(axis=0)   # (n_genes,)
    coverage = per_cell_total[cell_positions].sum()   # scalar

    group_labels.append((ct, tp))
    group_ncells.append(n)
    group_coverage.append(coverage)
    sum_matrix_rows.append(row_sum)

n_groups = len(group_labels)
print(f"  Reliable groups (n_cells >= {MIN_CELLS}): {n_groups}")

# Stack into matrix: (n_groups, n_genes)
sum_matrix = np.vstack(sum_matrix_rows)   # (n_groups, n_genes)
group_coverage = np.array(group_coverage, dtype=np.float64)
group_ncells   = np.array(group_ncells,   dtype=np.int32)

# %% Median-scale normalization (identical to ATAC pipeline)
print("Applying median-scale normalization ...", flush=True)
common_scale = np.median(group_coverage)
print(f"  Median coverage: {common_scale:.0f}")
scale_factors = common_scale / group_coverage    # (n_groups,)

# Multiply each group's sum by its scale factor
normalized = sum_matrix * scale_factors[:, None]  # (n_groups, n_genes)
print(f"  Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

# %% log1p
log_norm = np.log1p(normalized)
print(f"  log_norm range: [{log_norm.min():.2f}, {log_norm.max():.2f}]")

# %% Build AnnData: genes × conditions (transpose to match ATAC convention)
# obs = genes, var = conditions
var_meta = pd.DataFrame({
    "celltype":  [ct for ct, tp in group_labels],
    "timepoint": [tp for ct, tp in group_labels],
    "n_cells":   group_ncells,
    "coverage":  group_coverage,
    "scale_factor": scale_factors,
}, index=[f"{ct}_{tp}" for ct, tp in group_labels])

rna_pb = ad.AnnData(
    X = log_norm.T,                        # (n_genes, n_groups)
    obs = pd.DataFrame(index=gene_names),
    var = var_meta,
)
rna_pb.layers["sum"]        = sum_matrix.T
rna_pb.layers["normalized"] = normalized.T
rna_pb.layers["log_norm"]   = log_norm.T

out_pb = f"{OUT_DIR}/rna_by_ct_tp_pseudobulked.h5ad"
rna_pb.write_h5ad(out_pb, compression="gzip")
print(f"\nSaved RNA pseudobulk: {out_pb}")
print(f"  Shape: {rna_pb.shape}  (genes × conditions)")
print(f"  Conditions: {list(rna_pb.var_names[:5])} ...")

# %% Celltype-level z-scores
# Step 1: average log_norm across reliable timepoints per celltype
print("\nComputing celltype-level averages ...", flush=True)
celltypes = sorted(var_meta["celltype"].unique())
ct_mean_mat = np.zeros((n_genes, len(celltypes)), dtype=np.float64)

for ci, ct in enumerate(celltypes):
    col_mask = var_meta["celltype"] == ct
    col_idx  = np.where(col_mask)[0]
    ct_mean_mat[:, ci] = log_norm.T[:, col_idx].mean(axis=1)

print(f"  Celltype-mean matrix: {ct_mean_mat.shape}  ({len(celltypes)} celltypes)")

# Step 2: leave-one-out z-score across celltypes (vectorized, same as ATAC V3)
# NOTE: RNA uses a robust std floor of 0.5 (on log1p scale, ~2-fold variability).
# This is necessary because many RNA genes are zero across most celltypes, which
# would cause the 1e-10 ATAC floor to produce astronomically high z-scores.
# A floor of 0.5 caps max z-score at ~25 (= max log1p ~12.8 / 0.5), giving a
# scale comparable to ATAC z-scores for the concordance scatter plot.
RNA_STD_FLOOR = 0.5   # robust variance floor on log1p scale
C = ct_mean_mat.shape[1]
row_sum = ct_mean_mat.sum(axis=1, keepdims=True)
row_sq  = (ct_mean_mat ** 2).sum(axis=1, keepdims=True)
mean_other = (row_sum - ct_mean_mat) / (C - 1)
var_other  = (row_sq  - ct_mean_mat ** 2) / (C - 1) - mean_other ** 2
std_other  = np.sqrt(np.maximum(var_other, RNA_STD_FLOOR ** 2))
Z_rna = (ct_mean_mat - mean_other) / std_other   # (n_genes, n_celltypes)

print(f"  RNA z-score range: [{Z_rna.min():.1f}, {Z_rna.max():.1f}]")

# Save as AnnData: genes × celltypes
rna_z_adata = ad.AnnData(
    X = Z_rna.astype(np.float32),
    obs = pd.DataFrame(index=gene_names),
    var = pd.DataFrame(index=celltypes),
)
rna_z_adata.layers["ct_mean"] = ct_mean_mat.astype(np.float32)

out_z = f"{OUT_DIR}/rna_specificity_matrix_celltype_level.h5ad"
rna_z_adata.write_h5ad(out_z, compression="gzip")
print(f"\nSaved RNA z-score matrix: {out_z}")
print(f"  Shape: {rna_z_adata.shape}  (genes × celltypes)")

# %% Validation: check known marker genes
print("\n" + "="*60)
print("VALIDATION: top-5 genes per celltype by RNA z-score")
print("="*60)
gene_arr = np.array(gene_names)
for ct in ["heart_myocardium", "fast_muscle", "epidermis", "neural_crest", "hemangioblasts"]:
    if ct not in celltypes:
        continue
    ci = celltypes.index(ct)
    top5_idx = np.argsort(Z_rna[:, ci])[::-1][:5]
    top5_genes = gene_arr[top5_idx]
    top5_z = Z_rna[top5_idx, ci]
    print(f"\n{ct}:")
    for g, z in zip(top5_genes, top5_z):
        print(f"  {g:<20}  z={z:.2f}")

print(f"\nDone. End: {time.strftime('%c')}")
