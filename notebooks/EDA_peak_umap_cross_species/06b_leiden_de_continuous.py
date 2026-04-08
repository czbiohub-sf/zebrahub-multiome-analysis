# %% [markdown]
# # Step 6b: Leiden Clustering + Shuffled Species Plot + Differential Motif Analysis
#
# Input: cross_species_motif_embedded_continuous.h5ad
#
# Tasks:
#   1. Leiden clustering at resolutions 0.3, 0.5, 0.7, 1.0 → UMAP plots (PNG)
#   2. Shuffled species UMAP (alpha=0.7, randomized draw order)
#   3. Wilcoxon rank-sum DE motif analysis per Leiden cluster
#      → heatmap of top motifs per cluster
#
# Env: single-cell-base (CPU)
# Note: rsc.tl.leiden has a persistent CUDA_ERROR_INVALID_CONTEXT on this cluster.
# sc.tl.leiden (igraph, CPU) is used instead — the neighbor graph is already
# precomputed and stored in .obsp, so Leiden is just graph partitioning (~seconds/res).
#   conda run -p /hpc/user_apps/data.science/conda_envs/single-cell-base

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

print(f"scanpy {sc.__version__}")

# %% Paths
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
INPUT_H5AD = f"{SCRATCH}/cross_species_motif_embedded_continuous.h5ad"

FIG_DIR = ("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/"
           "zebrahub-multiome-analysis/figures/cross_species_motif_umap")
os.makedirs(FIG_DIR, exist_ok=True)

LEIDEN_RESOLUTIONS = [0.3, 0.5, 0.7, 1.0]
SPECIES_COLORS = {"zebrafish": "#1f77b4", "mouse": "#ff7f0e", "human": "#2ca02c"}

PT_SIZE  = 0.3
RASTERIZED = True

# %% ── Load ──────────────────────────────────────────────────────────────────
t0 = time.time()
print(f"Loading {INPUT_H5AD} ...")
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  obsp keys: {list(adata.obsp.keys())}")
print(f"  layers: {list(adata.layers.keys())}")

umap_coords = adata.obsm["X_umap"]
x, y = umap_coords[:, 0], umap_coords[:, 1]
species_arr = adata.obs["species"].values
print(f"  Load done in {time.time()-t0:.1f}s")

# %% ── Task 1: Leiden clustering (CPU, sc.tl.leiden) ────────────────────────
# Neighbor graph already in .obsp["connectivities"] — just graph partitioning.
print("\n── Task 1: Leiden clustering (CPU, sc.tl.leiden) ──")
t1 = time.time()

for res in LEIDEN_RESOLUTIONS:
    key = f"leiden_res{res}"
    sc.tl.leiden(adata, resolution=res, key_added=key, flavor="igraph", n_iterations=2)
    n_cl = adata.obs[key].nunique()
    print(f"  res={res}: {n_cl} clusters  ({time.time()-t1:.1f}s elapsed)")

print(f"  Leiden done in {time.time()-t1:.1f}s")

# Plot UMAP colored by Leiden for each resolution
for res in LEIDEN_RESOLUTIONS:
    key = f"leiden_res{res}"
    labels = adata.obs[key].values
    unique_labels = sorted(adata.obs[key].unique(), key=lambda c: int(c))
    n_cl = len(unique_labels)

    # Colormap: tab20 for ≤20, nipy_spectral for more
    cmap = plt.cm.get_cmap("tab20" if n_cl <= 20 else "nipy_spectral", n_cl)
    color_map = {cl: cmap(i) for i, cl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(9, 8))
    for cl in unique_labels:
        mask = labels == cl
        ax.scatter(x[mask], y[mask], c=[color_map[cl]], s=PT_SIZE,
                   alpha=0.6, rasterized=RASTERIZED, linewidths=0, label=cl)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Leiden clusters (res={res}, n={n_cl})\ncontinuous PWM scores")
    # Compact legend for many clusters
    ncol = max(1, n_cl // 20)
    ax.legend(markerscale=6, fontsize=6, ncol=ncol,
              loc="upper right", framealpha=0.7,
              title="Cluster", title_fontsize=7)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/umap_leiden_res{res}_continuous_PWM_score.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: umap_leiden_res{res}_continuous_PWM_score.png")

# %% ── Task 2: Shuffled species UMAP ─────────────────────────────────────────
print("\n── Task 2: Shuffled species UMAP ──")

rng = np.random.default_rng(42)
shuffle_idx = rng.permutation(len(adata))

x_sh = x[shuffle_idx]
y_sh = y[shuffle_idx]
sp_sh = species_arr[shuffle_idx]

fig, ax = plt.subplots(figsize=(9, 8))
for sp_name, col in SPECIES_COLORS.items():
    mask = sp_sh == sp_name
    ax.scatter(x_sh[mask], y_sh[mask], c=col, s=PT_SIZE, alpha=0.7,
               rasterized=RASTERIZED, linewidths=0,
               label=f"{sp_name} ({mask.sum():,})")

ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("Cross-species motif UMAP — shuffled draw order\n"
             "(continuous PWM z-score → PCA → cosine neighbors)")
ax.legend(markerscale=10, framealpha=0.8)
plt.tight_layout()

for ext in ["png", "pdf"]:
    fig.savefig(f"{FIG_DIR}/umap_species_shuffled_continuous_PWM_score.{ext}",
                dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: umap_species_shuffled_continuous_PWM_score.{png,pdf}")

# Per-species highlight panels (shuffled background)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, sp_name in zip(axes, ["zebrafish", "mouse", "human"]):
    fg = sp_sh == sp_name
    bg = ~fg
    ax.scatter(x_sh[bg], y_sh[bg], c="#cccccc", s=PT_SIZE * 0.4, alpha=0.2,
               rasterized=RASTERIZED, linewidths=0)
    ax.scatter(x_sh[fg], y_sh[fg], c=SPECIES_COLORS[sp_name], s=PT_SIZE,
               alpha=0.7, rasterized=RASTERIZED, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{sp_name.capitalize()} ({fg.sum():,} peaks)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

plt.suptitle("Per-species highlight — shuffled draw order", fontsize=12)
plt.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(f"{FIG_DIR}/umap_species_per_panel_shuffled_continuous_PWM_score.{ext}",
                dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: umap_species_per_panel_shuffled_continuous_PWM_score.{png,pdf}")

# %% ── Task 3: Differential motif analysis (Wilcoxon rank-sum) ───────────────
# Use leiden_res0.5 as the primary grouping (a good balance of resolution).
# Run rank_genes_groups on the z-scored PWM layer.
print("\n── Task 3: Differential motif analysis (Wilcoxon) ──")

DE_LEIDEN_KEY = "leiden_res0.5"
N_TOP_MOTIFS  = 10   # top motifs per cluster for visualization

print(f"  Running rank_genes_groups (groupby={DE_LEIDEN_KEY}, method=wilcoxon) ...")
print(f"  n_clusters={adata.obs[DE_LEIDEN_KEY].nunique()}, n_motifs={adata.n_vars}")

# Temporarily point .X to z-scored layer for DE testing
adata.X = adata.layers["zscore_pwm"]

t_de = time.time()
sc.tl.rank_genes_groups(
    adata,
    groupby=DE_LEIDEN_KEY,
    method="wilcoxon",
    use_raw=False,
    n_genes=N_TOP_MOTIFS,
    pts=True,           # compute fraction of peaks with motif
    key_added="rank_genes_wilcoxon",
)
print(f"  DE done in {time.time()-t_de:.1f}s")

# Restore .X to raw PWM
adata.X = adata.layers["raw_pwm"]

# ── Extract top motifs and build enrichment matrix ──
de_result = adata.uns["rank_genes_wilcoxon"]
clusters = sorted(adata.obs[DE_LEIDEN_KEY].unique(), key=lambda c: int(c))

# Build (n_clusters × N_TOP_MOTIFS) table of top motif names + scores
top_motifs_per_cluster = {}
for cl in clusters:
    names  = de_result["names"][cl]
    scores = de_result["scores"][cl]
    top_motifs_per_cluster[cl] = list(zip(names, scores))

# Unique motifs appearing in any top-N list
all_top_motifs = list(dict.fromkeys(
    m for cl in clusters for m, _ in top_motifs_per_cluster[cl]
))
print(f"  Unique top motifs across all clusters: {len(all_top_motifs)}")

# Mean z-score per cluster for these motifs
adata.X = adata.layers["zscore_pwm"]
motif_idx = {m: list(adata.var_names).index(m) for m in all_top_motifs if m in adata.var_names}

mean_zscore = pd.DataFrame(index=clusters, columns=list(motif_idx.keys()), dtype=float)
for cl in clusters:
    mask = adata.obs[DE_LEIDEN_KEY].values == cl
    for motif, col_i in motif_idx.items():
        vals = adata.X[mask, col_i]
        if sp.issparse(vals):
            vals = vals.toarray().ravel()
        mean_zscore.loc[cl, motif] = float(vals.mean())

adata.X = adata.layers["raw_pwm"]

# Shorten motif names for readability (take last part after '::' or '.')
def short_name(n):
    n = n.split("::")[-1]
    parts = n.split(".")
    return parts[-1] if len(parts) > 2 else n

mean_zscore.columns = [short_name(m) for m in mean_zscore.columns]
mean_zscore.index = [f"Cluster {c}" for c in mean_zscore.index]

# ── Heatmap ──
vmax = min(mean_zscore.abs().values.max(), 3.0)
fig_h = max(5, len(clusters) * 0.4)
fig_w = max(10, len(all_top_motifs) * 0.55)

g = sns.clustermap(
    mean_zscore.astype(float),
    cmap="RdBu_r", vmin=-vmax, vmax=vmax, center=0,
    figsize=(fig_w, fig_h),
    linewidths=0.3, linecolor="white",
    xticklabels=True, yticklabels=True,
    cbar_kws={"label": "Mean z-score (PWM)"},
)
g.fig.suptitle(
    f"Top {N_TOP_MOTIFS} Wilcoxon motifs per Leiden cluster ({DE_LEIDEN_KEY})\n"
    "continuous PWM z-score",
    fontsize=11, y=1.01,
)
for ext in ["png", "pdf"]:
    g.fig.savefig(
        f"{FIG_DIR}/leiden_top_motifs_heatmap_{DE_LEIDEN_KEY}_continuous_PWM_score.{ext}",
        dpi=150, bbox_inches="tight",
    )
plt.close(g.fig)
print(f"  Saved: leiden_top_motifs_heatmap_{DE_LEIDEN_KEY}_continuous_PWM_score.{{png,pdf}}")

# ── Summary: top 3 motifs per cluster printed ──
print("\n  Top 3 motifs per cluster (by Wilcoxon score):")
for cl in clusters:
    top3 = [m for m, _ in top_motifs_per_cluster[cl][:3]]
    print(f"    Cluster {cl}: {', '.join(top3)}")

# %% ── Save updated h5ad with Leiden annotations ────────────────────────────
print(f"\nSaving updated h5ad → {INPUT_H5AD} ...")
t_save = time.time()
adata.write_h5ad(INPUT_H5AD, compression="gzip")
print(f"  Saved in {time.time()-t_save:.1f}s")

print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("Done.")
