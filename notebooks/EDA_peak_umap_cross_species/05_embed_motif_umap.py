# %% [markdown]
# # Step 5: Cross-Species Peak UMAP — TF-IDF (within-species) → LSI → UMAP
#
# Pipeline (isomorphic to scATAC-seq cells × peaks → here peaks × motifs):
#   1. Within-species Signac-style TF-IDF normalization
#      - Removes species-specific genome composition bias
#      - TF = row_norm, IDF = log(1 + N_sp / col_sum_sp), scale × log1p
#   2. TruncatedSVD / LSI (sklearn, CPU — does NOT center, preserving sparsity)
#      - Drop first component (captures total motif count, like seq depth)
#   3. GPU: cosine-distance neighbor graph + UMAP
#   4. Save h5ad + lightweight CSV + quick-check figures (PDF + PNG)
#
# Leiden clustering: separate job.
#
# Env: sc_rapids (GPU for steps 3-4)
#   conda run -p /hpc/user_apps/data.science/conda_envs/sc_rapids

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad

import cupy as cp
import rapids_singlecell as rsc

from sklearn.decomposition import TruncatedSVD

print("Libraries loaded.")

# %% Paths & parameters
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
INPUT_H5AD  = f"{SCRATCH}/cross_species_motif_scores_FPR_0.010_binarized.h5ad"
OUTPUT_H5AD = f"{SCRATCH}/cross_species_motif_embedded.h5ad"
OUTPUT_CSV  = f"{SCRATCH}/cross_species_umap_coords.csv.gz"

FIG_DIR = ("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/"
           "zebrahub-multiome-analysis/figures/cross_species_motif_umap")
os.makedirs(FIG_DIR, exist_ok=True)

MIN_MOTIFS    = 2      # filter peaks with fewer hits (removes zero- and single-motif outliers)
SCALE_FACTOR  = 1e4   # Signac TF-IDF scale factor before log1p
N_SVD         = 51    # compute 51 SVD components, drop first, keep 50
N_LSI_USE     = 50    # LSI components fed to neighbors
N_NEIGHBORS   = 30
RANDOM_STATE  = 42

# %% ── Step 1: Load ──────────────────────────────────────────────────────────
t0 = time.time()
print(f"Loading {INPUT_H5AD} ...")
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}")
print(f"  Species counts:\n{adata.obs['species'].value_counts().to_string()}")

# Ensure CSR + binary layer
if not sp.isspmatrix_csr(adata.X):
    adata.X = adata.X.tocsr()
adata.layers["binary"] = adata.X.copy()

# %% ── Step 2: Filter low-motif peaks ───────────────────────────────────────
row_sums = np.asarray(adata.layers["binary"].sum(axis=1)).ravel()
adata.obs["n_motifs"] = row_sums.astype(int)

print(f"\nMotifs/peak: min={row_sums.min()}, p1={np.percentile(row_sums,1):.0f}, "
      f"median={np.median(row_sums):.0f}, max={row_sums.max()}")
print(f"  Peaks with <{MIN_MOTIFS} motifs: {(row_sums < MIN_MOTIFS).sum():,} → filtering")

adata = adata[row_sums >= MIN_MOTIFS].copy()
print(f"  Shape after filter: {adata.shape}")
print(f"  Species after filter:\n{adata.obs['species'].value_counts().to_string()}")

# %% ── Step 3: Within-species Signac TF-IDF ─────────────────────────────────
# Apply TF-IDF separately per species so each species' IDF is computed from
# its own peak population, removing genome-composition and GC-content biases.
#
# TF(i,j)    = X(i,j) / sum_j X(i,:)         [motifs per peak, row-normalized]
# IDF(j)     = log(1 + N_sp / sum_i X(:,j))   [within-species rarity]
# TF-IDF(i,j)= log1p(TF × IDF × scale_factor)
print("\nStep 3: Within-species Signac TF-IDF ...")
t1 = time.time()

X_binary = adata.layers["binary"]
species_arr = adata.obs["species"].values
species_list = sorted(np.unique(species_arr))

# Build TF-IDF via COO accumulation (avoids dense intermediate)
all_rows, all_cols, all_data = [], [], []

for sp_name in species_list:
    mask = np.where(species_arr == sp_name)[0]
    X_sp = X_binary[mask, :].astype(np.float64)
    n_sp = len(mask)

    # TF: row-normalize (each peak sums to 1)
    rs = np.asarray(X_sp.sum(axis=1)).ravel()
    rs[rs == 0] = 1
    X_tf = sp.diags(1.0 / rs) @ X_sp

    # IDF: within-species log(1 + N / col_sum)
    cs = np.asarray(X_sp.sum(axis=0)).ravel()
    cs[cs == 0] = 1
    idf = np.log1p(n_sp / cs)

    # TF-IDF: scale + log1p
    X_tfidf_sp = (X_tf @ sp.diags(idf)).multiply(SCALE_FACTOR)
    X_tfidf_sp = X_tfidf_sp.tocoo()
    X_tfidf_sp.data = np.log1p(X_tfidf_sp.data)

    # Map local row indices back to global indices
    all_rows.append(mask[X_tfidf_sp.row])
    all_cols.append(X_tfidf_sp.col)
    all_data.append(X_tfidf_sp.data.astype(np.float32))

    vmin, vmax = X_tfidf_sp.data.min(), X_tfidf_sp.data.max()
    print(f"  [{sp_name}] n={n_sp:,}  IDF range [{idf.min():.2f}, {idf.max():.2f}]  "
          f"TF-IDF range [{vmin:.2f}, {vmax:.2f}]")

# Assemble full sparse matrix
X_tfidf = sp.csr_matrix(
    (np.concatenate(all_data),
     (np.concatenate(all_rows), np.concatenate(all_cols))),
    shape=X_binary.shape, dtype=np.float32,
)
adata.X = X_tfidf
print(f"  TF-IDF done in {time.time()-t1:.1f}s  nnz={X_tfidf.nnz:,}")

# %% ── Step 4: Filter all-zero motif columns before SVD ─────────────────────
motif_totals = np.asarray(adata.X.sum(axis=0)).ravel()
nonzero_mask = motif_totals > 0
n_zero_motifs = (~nonzero_mask).sum()
if n_zero_motifs > 0:
    print(f"\nFiltering {n_zero_motifs} all-zero motif columns before SVD ...")
    adata = adata[:, nonzero_mask].copy()
    print(f"  Shape after column filter: {adata.shape}")

# %% ── Step 5: TruncatedSVD / LSI (CPU sklearn, sparse-aware) ───────────────
# TruncatedSVD does NOT center the data → sparsity preserved.
# Drop first component: typically captures total motif count (depth), not biology.
print(f"\nStep 5: TruncatedSVD (n_components={N_SVD}, CPU) ...")
t2 = time.time()

svd = TruncatedSVD(n_components=N_SVD, algorithm="randomized",
                   n_iter=5, random_state=RANDOM_STATE)
X_lsi_full = svd.fit_transform(adata.X)   # (n_peaks, N_SVD)
print(f"  SVD done in {time.time()-t2:.1f}s")

# Inspect first component — should correlate with total motifs per peak
n_motifs_arr = np.asarray(adata.layers["binary"].sum(axis=1)).ravel()
r_lsi1 = np.corrcoef(X_lsi_full[:, 0], n_motifs_arr)[0, 1]
print(f"  LSI1 correlation with n_motifs/peak: r={r_lsi1:.3f}  → dropping")

# Drop first component
X_lsi = X_lsi_full[:, 1:].astype(np.float32)   # (n_peaks, N_SVD-1 = 50)
var_ratio_full = svd.explained_variance_ratio_
var_ratio = var_ratio_full[1:]
print(f"  Retained {X_lsi.shape[1]} LSI components  "
      f"cumvar={np.cumsum(var_ratio)[-1]*100:.1f}%")

adata.obsm["X_lsi"] = X_lsi
adata.uns["lsi"] = {
    "variance_ratio": var_ratio,
    "variance_ratio_lsi1": float(var_ratio_full[0]),
    "lsi1_motif_corr": float(r_lsi1),
    "components": svd.components_[1:],    # motif loadings (50 × n_motifs)
}

# Scree plot (LSI components 2–51)
cumvar = np.cumsum(var_ratio)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(np.arange(1, len(var_ratio)+1), var_ratio, color="steelblue", alpha=0.8)
axes[0].set_xlabel("LSI component (after dropping LSI1)")
axes[0].set_ylabel("Variance ratio")
axes[0].set_title("Per-LSI variance explained")
axes[1].plot(np.arange(1, len(cumvar)+1), cumvar, "o-", ms=3)
axes[1].axhline(0.9, color="red", ls="--", label="90%")
axes[1].axvline(N_LSI_USE, color="orange", ls="--", label=f"LSI {N_LSI_USE}")
axes[1].set_xlabel("LSI components"); axes[1].set_ylabel("Cumulative variance ratio")
axes[1].set_title("Cumulative variance explained"); axes[1].legend()
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/lsi_variance_explained.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Scree plot saved. Cumvar @ LSI{N_LSI_USE}: {cumvar[N_LSI_USE-1]:.3f}")

# %% ── Step 6: GPU — neighbors + UMAP ────────────────────────────────────────
print(f"\nStep 6: GPU neighbors + UMAP ...")
rsc.get.anndata_to_GPU(adata)

print(f"  Neighbors (n={N_NEIGHBORS}, cosine) ...")
t3 = time.time()
rsc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_lsi",
                 metric="cosine", n_pcs=N_LSI_USE)
print(f"  Neighbors done in {time.time()-t3:.1f}s")

print("  UMAP (min_dist=0.1, spread=1.0) ...")
t4 = time.time()
rsc.tl.umap(adata, min_dist=0.1, spread=1.0)
print(f"  UMAP done in {time.time()-t4:.1f}s")

rsc.get.anndata_to_CPU(adata)

umap_coords = adata.obsm["X_umap"]
assert np.all(np.isfinite(umap_coords)), "UMAP contains non-finite values!"
print(f"  UMAP range: x=[{umap_coords[:,0].min():.2f}, {umap_coords[:,0].max():.2f}]  "
      f"y=[{umap_coords[:,1].min():.2f}, {umap_coords[:,1].max():.2f}]")

# %% ── Verification ──────────────────────────────────────────────────────────
print("\n=== Verification ===")
print(f"  Shape: {adata.shape}  ({1874537 - adata.shape[0]:,} peaks removed)")
for sp_name in ["zebrafish", "mouse", "human"]:
    n = (adata.obs["species"] == sp_name).sum()
    print(f"  {sp_name}: {n:,} peaks")
print("All checks passed.")

# %% ── Step 7: Quick-check species UMAP ─────────────────────────────────────
print("\nGenerating species UMAP quick-check ...")
SPECIES_COLORS = {"zebrafish": "#1f77b4", "mouse": "#ff7f0e", "human": "#2ca02c"}
x, y = umap_coords[:, 0], umap_coords[:, 1]
species_arr_out = adata.obs["species"].values

fig, ax = plt.subplots(figsize=(8, 8))
for sp_name, col in SPECIES_COLORS.items():
    mask = species_arr_out == sp_name
    ax.scatter(x[mask], y[mask], c=col, s=0.3, alpha=0.5,
               rasterized=True, linewidths=0,
               label=f"{sp_name} ({mask.sum():,})")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("Cross-species motif UMAP — colored by species\n"
             "(within-species TF-IDF → LSI → cosine neighbors)")
ax.legend(markerscale=10, framealpha=0.8)
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/umap_species_quickcheck.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved: umap_species_quickcheck.{ext}")
plt.close(fig)

# %% ── Step 8: Save ──────────────────────────────────────────────────────────
print(f"\nSaving h5ad → {OUTPUT_H5AD} ...")
t5 = time.time()
adata.layers["tfidf"] = adata.X.copy()
adata.X = adata.layers["binary"]   # revert .X to interpretable binary
adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
print(f"  Saved in {time.time()-t5:.1f}s")

print(f"Saving CSV → {OUTPUT_CSV} ...")
coords_df = pd.DataFrame(
    {"umap_1": x, "umap_2": y, "species": adata.obs["species"].values,
     "n_motifs": adata.obs["n_motifs"].values},
    index=adata.obs_names,
)
for col in ["celltype", "top_celltype", "timepoint", "top_timepoint"]:
    if col in adata.obs.columns:
        coords_df[col] = adata.obs[col].values
coords_df.to_csv(OUTPUT_CSV, compression="gzip")
print(f"  CSV saved: {len(coords_df):,} rows")

print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("Done. Leiden clustering → separate job.")
