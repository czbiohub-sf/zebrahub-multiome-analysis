# %% [markdown]
# # Step 12: Procrustes Alignment + Aligned UMAP
#
# Aligns per-species PCA embeddings into a shared coordinate space using
# anchor peak triplets, then recomputes UMAP on the aligned representation.
#
# Algorithm:
#   1. Load annotated h5ad + root_anchors + branch_anchors
#   2. Split X_pca (50 PCs) by species
#   3. Build anchor coordinate matrices per species
#   4. Pairwise Procrustes: mouse→zebrafish, human→zebrafish
#      using scipy.linalg.orthogonal_procrustes
#   5. Stack aligned PCA → obsm["X_pca_aligned"]
#   6. GPU neighbors (n=30, cosine) + UMAP + Leiden (res 0.3/0.5/0.7/1.0)
#   7. Validation metrics + save
#
# Input:
#   - {SCRATCH}/cross_species_motif_embedded_annotated.h5ad
#   - {SCRATCH}/anchors/root_anchors.csv
#   - {SCRATCH}/anchors/branch_anchors.csv
#
# Output:
#   - {SCRATCH}/cross_species_anchor_aligned.h5ad
#   - {SCRATCH}/cross_species_anchor_aligned_coords.csv.gz
#   - Diagnostic figures
#
# Env: sc_rapids (GPU)  — 1 GPU, 8 CPUs, 128G, 4h
#   conda run -p /hpc/user_apps/data.science/conda_envs/sc_rapids python -u 12_procrustes_alignment.py

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

# GPU imports — fall back to CPU if unavailable
USE_GPU = False
try:
    import cupy as cp
    import rapids_singlecell as rsc
    USE_GPU = True
    print("GPU (rapids_singlecell) available.")
except ImportError:
    print("WARNING: rapids_singlecell not available; falling back to CPU UMAP.")

print(f"scanpy {sc.__version__}")

# %% Paths
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"

INPUT_H5AD     = f"{SCRATCH}/cross_species_motif_embedded_annotated.h5ad"
ROOT_CSV       = f"{SCRATCH}/anchors/root_anchors.csv"
BRANCH_CSV     = f"{SCRATCH}/anchors/branch_anchors.csv"

OUTPUT_H5AD    = f"{SCRATCH}/cross_species_anchor_aligned.h5ad"
OUTPUT_COORDS  = f"{SCRATCH}/cross_species_anchor_aligned_coords.csv.gz"

FIG_DIR = (f"{BASE}/zebrahub-multiome-analysis/figures/cross_species_motif_umap")
os.makedirs(FIG_DIR, exist_ok=True)

N_PCS            = 50
N_NEIGHBORS      = 30
LEIDEN_RESOLUTIONS = [0.3, 0.5, 0.7, 1.0]
SPECIES_COLORS   = {"zebrafish": "#1f77b4", "mouse": "#ff7f0e", "human": "#2ca02c"}
PT_SIZE = 0.3


# %% ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_H5AD} ...")
t0 = time.time()
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")
print(f"  obsm keys: {list(adata.obsm.keys())}")

root_anch   = pd.read_csv(ROOT_CSV)
branch_anch = pd.read_csv(BRANCH_CSV)
print(f"  Root anchors:   {len(root_anch):,}")
print(f"  Branch anchors: {len(branch_anch):,}")

# Verify X_pca present
if "X_pca" not in adata.obsm:
    raise ValueError("X_pca not found in obsm. Run earlier embedding steps first.")

X_pca = adata.obsm["X_pca"][:, :N_PCS].astype(np.float32)
print(f"  X_pca shape: {X_pca.shape}")


# %% ── Build anchor index maps ────────────────────────────────────────────────────
# Combine root + branch anchors (use all for Procrustes)
def combine_anchors(root_df, branch_df):
    """Stack root and branch anchor peak IDs into a unified list of triplets."""
    cols = ["peak_id_zf", "peak_id_mm", "peak_id_hs"]
    r = root_df[cols].dropna(subset=["peak_id_zf"]).copy()
    b = branch_df[cols].dropna(subset=["peak_id_zf"]).copy() if len(branch_df) else pd.DataFrame(columns=cols)
    return pd.concat([r, b], ignore_index=True).drop_duplicates()


all_anchors = combine_anchors(root_anch, branch_anch)
print(f"\nAll anchor triplets (root+branch): {len(all_anchors):,}")

# Build obs_name → integer index maps
obs_names = adata.obs_names.astype(str)
obs_idx_map = {name: i for i, name in enumerate(obs_names)}

species_arr = adata.obs["species"].values

def get_species_indices(species_label):
    return np.where(species_arr == species_label)[0]

zf_idx = get_species_indices("zebrafish")
mm_idx = get_species_indices("mouse")
hs_idx = get_species_indices("human")

print(f"  Species counts: ZF={len(zf_idx)}, MM={len(mm_idx)}, HS={len(hs_idx)}")


# %% ── Extract anchor coordinate matrices ─────────────────────────────────────────
def build_anchor_matrix(anchors_df, col_src, obs_idx_map, X_pca, ref_col=None, ref_idx_map=None):
    """
    Build (N_anchors × N_PCs) matrices for source and (optionally) reference species.
    Returns (src_matrix, ref_matrix, valid_mask) where valid_mask selects rows
    where both src and ref peak IDs are found in obs.
    """
    src_ids = anchors_df[col_src].astype(str).values
    src_found = np.array([obs_idx_map.get(p, -1) for p in src_ids])
    valid = src_found >= 0

    if ref_col is not None and ref_idx_map is not None:
        ref_ids   = anchors_df[ref_col].astype(str).values
        ref_found = np.array([ref_idx_map.get(p, -1) for p in ref_ids])
        valid = valid & (ref_found >= 0)

        src_mat = X_pca[src_found[valid]]
        ref_mat = X_pca[ref_found[valid]]
        return src_mat, ref_mat, valid
    else:
        src_mat = X_pca[src_found[valid]]
        return src_mat, None, valid


# ZF is the reference; align MM → ZF, HS → ZF
# Build per-species obs_idx_maps (subset)
zf_obs_idx = {name: i for i, name in enumerate(obs_names) if species_arr[i] == "zebrafish"}
mm_obs_idx = {name: i for i, name in enumerate(obs_names) if species_arr[i] == "mouse"}
hs_obs_idx = {name: i for i, name in enumerate(obs_names) if species_arr[i] == "human"}

# For global indexing
zf_global = {name: obs_idx_map[name] for name in zf_obs_idx}
mm_global = {name: obs_idx_map[name] for name in mm_obs_idx}
hs_global = {name: obs_idx_map[name] for name in hs_obs_idx}

print("\nBuilding anchor matrices ...")

# MM → ZF
mm_src_mat, zf_ref_mat_mm, valid_mm = build_anchor_matrix(
    all_anchors, "peak_id_mm", mm_global, X_pca,
    ref_col="peak_id_zf", ref_idx_map=zf_global
)
print(f"  MM→ZF: {valid_mm.sum()} / {len(all_anchors)} anchor pairs valid")

# HS → ZF
hs_src_mat, zf_ref_mat_hs, valid_hs = build_anchor_matrix(
    all_anchors, "peak_id_hs", hs_global, X_pca,
    ref_col="peak_id_zf", ref_idx_map=zf_global
)
print(f"  HS→ZF: {valid_hs.sum()} / {len(all_anchors)} anchor pairs valid")

if mm_src_mat.shape[0] < 30:
    raise ValueError(f"Too few MM→ZF anchor pairs: {mm_src_mat.shape[0]}. "
                     "Increase anchor yield (relax distance_to_tss thresholds).")
if hs_src_mat.shape[0] < 30:
    raise ValueError(f"Too few HS→ZF anchor pairs: {hs_src_mat.shape[0]}. "
                     "Increase anchor yield.")


# %% ── Procrustes alignment ───────────────────────────────────────────────────────
def procrustes_align(X_src_anchors, X_ref_anchors, X_src_all):
    """
    Center, rotate, and scale X_src_all to align with X_ref using anchor pairs.
    Returns aligned X_src_all (same shape as input).
    """
    mu_src = X_src_anchors.mean(axis=0)
    mu_ref = X_ref_anchors.mean(axis=0)

    A = X_src_anchors - mu_src   # centred source anchors
    B = X_ref_anchors - mu_ref   # centred reference anchors

    # Orthogonal Procrustes: min ||A R - B||_F  s.t. R^T R = I
    R, _ = scipy.linalg.orthogonal_procrustes(A, B)

    # Apply to all source points
    X_aligned = (X_src_all - mu_src) @ R + mu_ref
    return X_aligned.astype(np.float32), R, mu_src, mu_ref


print("\nProcrustes: MM → ZF ...")
mm_all_idx = np.array([obs_idx_map[n] for n in obs_names if species_arr[obs_idx_map[n]] == "mouse"])
# Reindex correctly: get all MM peak global indices
mm_global_indices = np.array(list(mm_global.values()))
X_mm_all = X_pca[mm_global_indices]
X_mm_aligned, R_mm, mu_mm, mu_zf_mm = procrustes_align(mm_src_mat, zf_ref_mat_mm, X_mm_all)
print(f"  MM alignment residual: {np.mean((mm_src_mat - mu_mm) @ R_mm + mu_zf_mm - zf_ref_mat_mm) ** 2:.4f}")

print("Procrustes: HS → ZF ...")
hs_global_indices = np.array(list(hs_global.values()))
X_hs_all = X_pca[hs_global_indices]
X_hs_aligned, R_hs, mu_hs, mu_zf_hs = procrustes_align(hs_src_mat, zf_ref_mat_hs, X_hs_all)
print(f"  HS alignment residual: {np.mean((hs_src_mat - mu_hs) @ R_hs + mu_zf_hs - zf_ref_mat_hs) ** 2:.4f}")

# ZF stays as-is
zf_global_indices = np.array(list(zf_global.values()))
X_zf_all = X_pca[zf_global_indices]


# %% ── Assemble aligned PCA matrix ────────────────────────────────────────────────
print("\nAssembling X_pca_aligned ...")
X_pca_aligned = np.empty_like(X_pca)
X_pca_aligned[zf_global_indices] = X_zf_all
X_pca_aligned[mm_global_indices] = X_mm_aligned
X_pca_aligned[hs_global_indices] = X_hs_aligned

adata.obsm["X_pca_aligned"] = X_pca_aligned
print(f"  X_pca_aligned shape: {X_pca_aligned.shape}")


# %% ── Validation: anchor co-localization ────────────────────────────────────────
print("\nValidating anchor co-localization ...")

def anchor_colocalization(anchors_df, obs_idx_map, X_emb, label=""):
    """
    Compute mean pairwise distance between aligned anchor triplets
    vs. random pairs of the same size.
    """
    valid_rows = []
    for _, row in anchors_df.iterrows():
        ids = [str(row.get(c, "")) for c in ["peak_id_zf", "peak_id_mm", "peak_id_hs"]]
        idxs = [obs_idx_map.get(i, -1) for i in ids if i and i != "nan"]
        if all(ix >= 0 for ix in idxs) and len(idxs) >= 2:
            valid_rows.append(idxs)
    if not valid_rows:
        print(f"  [{label}] No valid triplets to evaluate.")
        return

    dists = []
    for triplet in valid_rows:
        coords = X_emb[triplet]
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dists.append(np.linalg.norm(coords[i] - coords[j]))

    N_random = len(dists)
    rand_idx = np.random.choice(len(X_emb), size=(N_random * 2,), replace=False)
    rand_dists = [np.linalg.norm(X_emb[rand_idx[2*k]] - X_emb[rand_idx[2*k+1]])
                  for k in range(N_random)]

    print(f"  [{label}] Anchor mean dist: {np.mean(dists):.4f}  |  "
          f"Random mean dist: {np.mean(rand_dists):.4f}  "
          f"(n_triplets={len(valid_rows)})")
    return np.array(dists), np.array(rand_dists)

umap_aligned_check = adata.obsm.get("X_umap", None)  # will be recomputed below

# Validate on PCA aligned (UMAP not yet computed)
anchor_colocalization(all_anchors, obs_idx_map, X_pca_aligned, "PCA aligned")


# %% ── GPU/CPU neighbors + UMAP on aligned PCA ────────────────────────────────────
print(f"\nComputing neighbors on X_pca_aligned (n={N_NEIGHBORS}, cosine) ...")
adata.obsm["X_pca"] = X_pca_aligned   # temporarily replace so sc.pp.neighbors uses it

if USE_GPU:
    try:
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca",
                         metric="cosine", n_pcs=N_PCS)
        rsc.tl.umap(adata, min_dist=0.3)
        rsc.get.anndata_to_CPU(adata)
        print("  GPU UMAP done.")
    except Exception as e:
        print(f"  GPU failed ({e}); falling back to CPU ...")
        USE_GPU = False

if not USE_GPU:
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca",
                    metric="cosine", n_pcs=N_PCS)
    sc.tl.umap(adata, min_dist=0.3)
    print("  CPU UMAP done.")

# Store aligned UMAP
adata.obsm["X_umap_aligned"] = adata.obsm["X_umap"].copy()

# Restore original X_pca_aligned name
adata.obsm["X_pca_aligned"] = adata.obsm.pop("X_pca")
adata.obsm["X_pca"] = X_pca


# %% ── Leiden clustering on aligned UMAP ─────────────────────────────────────────
print("\nLeiden clustering ...")
for res in LEIDEN_RESOLUTIONS:
    sc.tl.leiden(adata, resolution=res, key_added=f"leiden_aligned_{res}")
    n_cl = adata.obs[f"leiden_aligned_{res}"].nunique()
    print(f"  res={res}: {n_cl} clusters")


# %% ── Quick diagnostic figure: unaligned vs aligned UMAP ────────────────────────
print("\nGenerating comparison figure ...")
umap_orig    = adata.obsm.get("X_umap", None)
umap_aligned = adata.obsm["X_umap_aligned"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, coords, title in [
    (axes[0], umap_orig,    "Unaligned UMAP"),
    (axes[1], umap_aligned, "Procrustes-Aligned UMAP"),
]:
    if coords is None:
        ax.set_title(f"{title} (N/A)")
        continue
    order = np.random.permutation(len(coords))
    for sp, c in SPECIES_COLORS.items():
        mask = species_arr == sp
        idxs = np.where(mask)[0]
        ax.scatter(coords[idxs, 0], coords[idxs, 1],
                   c=c, s=PT_SIZE, alpha=0.5, rasterized=True, label=sp)
    ax.set_title(title)
    ax.legend(loc="upper right", markerscale=5, fontsize=8)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/umap_aligned_vs_unaligned.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/umap_aligned_vs_unaligned.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved comparison figure.")


# %% ── Silhouette score (species mixing) ─────────────────────────────────────────
print("\nComputing silhouette score (species label) ...")
from sklearn.metrics import silhouette_score

try:
    # Subsample if too large
    n_sub = min(50_000, len(umap_aligned))
    sub_idx = np.random.choice(len(umap_aligned), n_sub, replace=False)
    sp_labels = species_arr[sub_idx]

    sil_aligned = silhouette_score(umap_aligned[sub_idx], sp_labels, metric="euclidean")
    print(f"  Silhouette(species) on aligned UMAP: {sil_aligned:.4f}")

    if umap_orig is not None:
        sil_orig = silhouette_score(umap_orig[sub_idx], sp_labels, metric="euclidean")
        print(f"  Silhouette(species) on original UMAP: {sil_orig:.4f}")
        print(f"  Improvement (lower is better mixing): {sil_orig - sil_aligned:.4f}")
except Exception as e:
    print(f"  Silhouette failed: {e}")


# %% ── Save h5ad ─────────────────────────────────────────────────────────────────
print(f"\nSaving {OUTPUT_H5AD} ...")
t_save = time.time()
adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
print(f"  Saved in {time.time()-t_save:.1f}s")

# Save coords CSV
print(f"Saving coords to {OUTPUT_COORDS} ...")
coords_df = pd.DataFrame({
    "obs_name":  adata.obs_names,
    "species":   adata.obs["species"].values,
    "umap1":     umap_aligned[:, 0],
    "umap2":     umap_aligned[:, 1],
    "leiden_0.5": adata.obs.get("leiden_aligned_0.5", pd.Series(dtype=str)).values,
})
coords_df.to_csv(OUTPUT_COORDS, index=False)
print(f"  Saved coords ({len(coords_df):,} rows).")

print("\nDone.")
