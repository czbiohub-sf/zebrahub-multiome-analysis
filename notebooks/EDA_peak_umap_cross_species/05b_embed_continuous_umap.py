# %% [markdown]
# # Step 5b: Cross-Species Peak UMAP — Continuous PWM Scores → z-score → PCA → UMAP
#
# Alternative to the binary TF-IDF approach. Uses continuous PWM best_score values
# which carry richer signal (graded motif affinity, not just presence/absence).
#
# Pipeline:
#   1. Load continuous PWM scores h5ad (~5.96 GB, 89% non-zero)
#   2. Within-species z-score normalization per motif
#      - Removes species-specific GC/genome background score distributions
#      - z = (score - species_mean) / species_std, clipped at ±5
#   3. GPU PCA (100 comps, rsc.pp.pca) with CPU fallback if OOM
#   4. GPU cosine neighbors (n=30, 50 PCs) + UMAP
#   5. Save h5ad + CSV + figures (*_continuous_PWM_score.{pdf,png})
#
# Env: sc_rapids
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

print("Libraries loaded.")

# %% Paths & parameters
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
INPUT_H5AD  = f"{SCRATCH}/cross_species_motif_scores.h5ad"          # continuous PWM scores
OUTPUT_H5AD = f"{SCRATCH}/cross_species_motif_embedded_continuous.h5ad"
OUTPUT_CSV  = f"{SCRATCH}/cross_species_umap_coords_continuous.csv.gz"

FIG_DIR = ("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/"
           "zebrahub-multiome-analysis/figures/cross_species_motif_umap")
os.makedirs(FIG_DIR, exist_ok=True)

Z_CLIP      = 5.0   # clip z-scores at ±5 to suppress outlier motif scores
N_PCS       = 100   # PCA components to compute
N_PCS_USE   = 50    # PCs fed to neighbors
N_NEIGHBORS = 30
RANDOM_STATE = 42

# %% ── Step 1: Load ──────────────────────────────────────────────────────────
t0 = time.time()
print(f"Loading {INPUT_H5AD} ...")
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}")
print(f"  X dtype: {adata.X.dtype}, sparse: {sp.issparse(adata.X)}")
print(f"  X nnz: {adata.X.nnz if sp.issparse(adata.X) else 'dense':,}")
print(f"  obs cols: {list(adata.obs.columns)}")
print(f"  Species counts:\n{adata.obs['species'].value_counts().to_string()}")

# Ensure CSR
if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
    adata.X = adata.X.tocsr()

# Store raw scores as layer
adata.layers["raw_pwm"] = adata.X.copy()

# %% ── Step 2: Within-species z-score normalization ─────────────────────────
# Convert to dense float32 for in-place z-score.
# Memory: ~1.87M × 879 × 4 bytes = ~6.6 GB
print(f"\nStep 2: Within-species z-score normalization (clip ±{Z_CLIP}) ...")
print("  Converting to dense float32 ...")
t1 = time.time()

species_arr = adata.obs["species"].values

if sp.issparse(adata.X):
    X_dense = adata.X.toarray().astype(np.float32)
else:
    X_dense = np.asarray(adata.X, dtype=np.float32)

# Free sparse matrix to recover RAM
del adata.X
gc.collect()
print(f"  Dense matrix: {X_dense.shape}, {X_dense.nbytes / 1e9:.2f} GB")

# Within-species z-score per motif column
for sp_name in sorted(np.unique(species_arr)):
    mask = species_arr == sp_name
    X_sp = X_dense[mask, :]                          # (n_sp, n_motifs)
    mean_sp = X_sp.mean(axis=0, keepdims=True)       # (1, n_motifs)
    std_sp  = X_sp.std(axis=0, keepdims=True)
    std_sp[std_sp == 0] = 1.0                        # avoid div-by-zero
    X_dense[mask, :] = np.clip((X_sp - mean_sp) / std_sp, -Z_CLIP, Z_CLIP)
    vmin, vmax = X_dense[mask, :].min(), X_dense[mask, :].max()
    print(f"  [{sp_name}] n={mask.sum():,}  z-score range: [{vmin:.3f}, {vmax:.3f}]")

print(f"  Z-score done in {time.time()-t1:.1f}s")

# Assign back to adata
adata.X = X_dense
print(f"  RAM after z-score: X shape={adata.X.shape}")

# %% ── Step 3: PCA (GPU with CPU fallback) ───────────────────────────────────
print(f"\nStep 3: PCA (n_comps={N_PCS}) ...")
t2 = time.time()

try:
    print("  Attempting GPU PCA (rsc.pp.pca) ...")
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.pca(adata, n_comps=N_PCS, use_highly_variable=False)
    rsc.get.anndata_to_CPU(adata)
    print(f"  GPU PCA done in {time.time()-t2:.1f}s")
    pca_backend = "GPU (cuML)"

except Exception as e:
    print(f"  GPU PCA failed ({type(e).__name__}: {e})")
    print("  Falling back to CPU PCA (sklearn) ...")
    try:
        rsc.get.anndata_to_CPU(adata)
    except Exception:
        pass
    gc.collect()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=N_PCS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(adata.X).astype(np.float32)
    adata.obsm["X_pca"] = X_pca
    adata.uns["pca"] = {"variance_ratio": pca.explained_variance_ratio_}
    print(f"  CPU PCA done in {time.time()-t2:.1f}s")
    pca_backend = "CPU (sklearn)"

print(f"  PCA backend: {pca_backend}")
print(f"  X_pca shape: {adata.obsm['X_pca'].shape}")

# Scree plot
variance_ratio = np.array(adata.uns["pca"]["variance_ratio"])
cumvar = np.cumsum(variance_ratio)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(np.arange(1, len(variance_ratio)+1), variance_ratio, "o-", ms=3)
axes[0].set_xlabel("PC"); axes[0].set_ylabel("Variance ratio")
axes[0].set_title(f"Per-PC variance explained ({pca_backend})")
axes[1].plot(np.arange(1, len(cumvar)+1), cumvar, "o-", ms=3)
axes[1].axhline(0.9, color="red", ls="--", label="90%")
axes[1].axvline(N_PCS_USE, color="orange", ls="--", label=f"PC {N_PCS_USE}")
axes[1].set_xlabel("PC"); axes[1].set_ylabel("Cumulative variance ratio")
axes[1].set_title("Cumulative variance explained"); axes[1].legend()
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/pca_variance_explained_continuous_PWM_score.{ext}",
                dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Scree plot saved. Cumvar @ PC{N_PCS_USE}: {cumvar[N_PCS_USE-1]:.3f}")

# %% ── Step 4: GPU neighbors + UMAP ─────────────────────────────────────────
print(f"\nStep 4: GPU neighbors (n={N_NEIGHBORS}, cosine, {N_PCS_USE} PCs) + UMAP ...")

rsc.get.anndata_to_GPU(adata)

t3 = time.time()
rsc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca",
                 metric="cosine", n_pcs=N_PCS_USE)
print(f"  Neighbors done in {time.time()-t3:.1f}s")

t4 = time.time()
rsc.tl.umap(adata, min_dist=0.1, spread=1.0)
print(f"  UMAP done in {time.time()-t4:.1f}s")

rsc.get.anndata_to_CPU(adata)

umap_coords = adata.obsm["X_umap"]
assert np.all(np.isfinite(umap_coords)), "UMAP contains non-finite values!"
print(f"  UMAP range: x=[{umap_coords[:,0].min():.2f}, {umap_coords[:,0].max():.2f}]  "
      f"y=[{umap_coords[:,1].min():.2f}, {umap_coords[:,1].max():.2f}]")

# Quick distance check (sample 2K peaks)
from sklearn.metrics.pairwise import cosine_distances
rng = np.random.default_rng(42)
idx = rng.choice(len(adata), min(2000, len(adata)), replace=False)
D = cosine_distances(adata.obsm["X_pca"][idx, :N_PCS_USE])
upper = D[np.triu_indices(len(idx), k=1)]
print(f"  Cosine dist (2K sample): mean={upper.mean():.3f}  std={upper.std():.3f}  "
      f"p5={np.percentile(upper,5):.3f}  p95={np.percentile(upper,95):.3f}")

# %% ── Step 5: Species UMAP plot ─────────────────────────────────────────────
print("\nGenerating species UMAP quick-check ...")
SPECIES_COLORS = {"zebrafish": "#1f77b4", "mouse": "#ff7f0e", "human": "#2ca02c"}
x, y = umap_coords[:, 0], umap_coords[:, 1]
sp_vals = adata.obs["species"].values

fig, ax = plt.subplots(figsize=(8, 8))
for sp_name, col in SPECIES_COLORS.items():
    mask = sp_vals == sp_name
    ax.scatter(x[mask], y[mask], c=col, s=0.3, alpha=0.5,
               rasterized=True, linewidths=0,
               label=f"{sp_name} ({mask.sum():,})")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("Cross-species motif UMAP — continuous PWM scores\n"
             f"(within-species z-score → PCA {N_PCS_USE}PCs → cosine neighbors)")
ax.legend(markerscale=10, framealpha=0.8)
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/umap_species_quickcheck_continuous_PWM_score.{ext}",
                dpi=150, bbox_inches="tight")
    print(f"  Saved: umap_species_quickcheck_continuous_PWM_score.{ext}")
plt.close(fig)

# %% ── Step 6: Save ──────────────────────────────────────────────────────────
print(f"\nSaving h5ad → {OUTPUT_H5AD} ...")
t5 = time.time()
# Store z-scored matrix as layer; revert .X to raw PWM for interpretability
adata.layers["zscore_pwm"] = adata.X.copy()
adata.X = adata.layers["raw_pwm"]
adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
print(f"  Saved in {time.time()-t5:.1f}s")

print(f"Saving CSV → {OUTPUT_CSV} ...")
coords_df = pd.DataFrame(
    {"umap_1": x, "umap_2": y, "species": sp_vals},
    index=adata.obs_names,
)
for col in ["celltype", "top_celltype", "timepoint", "top_timepoint"]:
    if col in adata.obs.columns:
        coords_df[col] = adata.obs[col].values
coords_df.to_csv(OUTPUT_CSV, compression="gzip")
print(f"  CSV saved: {len(coords_df):,} rows")

print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("Done. Leiden clustering → separate job.")
