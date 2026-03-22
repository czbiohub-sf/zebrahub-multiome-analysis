# %% [markdown]
# # Step 13: Visualize Anchor-Aligned Cross-Species UMAP
#
# Generates the full figure panel for the anchor-aligned UMAP:
#   1. Species overlay (shuffled draw order)
#   2. Per-species highlight panels (3 panels)
#   3. Anchor peaks highlighted (root=red, branch=blue, non-anchor=gray)
#   4. Lineage labels per species
#   5. Leiden clusters + species composition bar charts
#   6. Side-by-side: unaligned vs aligned UMAP
#   7. Top DE motifs per aligned Leiden cluster (heatmap)
#   8. Key ortholog TF families on UMAP (GATA, SOX, PAX, MYOD)
#
# Input:
#   - {SCRATCH}/cross_species_anchor_aligned.h5ad
#   - {SCRATCH}/anchors/root_anchors.csv
#   - {SCRATCH}/anchors/branch_anchors.csv
#
# Env: single-cell-base (CPU)  — 4 CPUs, 64G, 1h
#   conda run -p /hpc/user_apps/data.science/conda_envs/single-cell-base python -u 13_visualize_aligned_umap.py

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import seaborn as sns
import scanpy as sc

print(f"scanpy {sc.__version__}")

# %% Paths
SCRATCH  = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
BASE     = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
UNALIGNED_H5AD = f"{SCRATCH}/cross_species_motif_embedded_continuous.h5ad"

INPUT_H5AD  = f"{SCRATCH}/cross_species_anchor_aligned.h5ad"
ROOT_CSV    = f"{SCRATCH}/anchors/root_anchors.csv"
BRANCH_CSV  = f"{SCRATCH}/anchors/branch_anchors.csv"

FIG_DIR = (f"{BASE}/zebrahub-multiome-analysis/figures/cross_species_motif_umap")
os.makedirs(FIG_DIR, exist_ok=True)

SPECIES_COLORS = {"zebrafish": "#1f77b4", "mouse": "#ff7f0e", "human": "#2ca02c"}
LINEAGE_COLORS = {
    "neural_cns":        "#9467bd",
    "paraxial_mesoderm": "#d62728",
    "lateral_mesoderm":  "#e377c2",
    "endoderm":          "#8c564b",
    "ectoderm":          "#bcbd22",
    "neural_crest":      "#17becf",
}

PT_SIZE    = 0.3
RASTERIZED = True

# TF families to highlight
TF_FAMILIES = {
    "GATA": ["gata1", "gata2", "gata3", "gata4", "gata5", "gata6"],
    "SOX":  ["sox2", "sox9", "sox17", "sox10", "sox1"],
    "PAX":  ["pax2", "pax3", "pax6", "pax7"],
    "MYOD": ["myod1", "myf5", "myog"],
    "FOXA": ["foxa1", "foxa2", "foxa3"],
}


# %% ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading aligned h5ad: {INPUT_H5AD} ...")
t0 = time.time()
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")

root_anch   = pd.read_csv(ROOT_CSV)
branch_anch = pd.read_csv(BRANCH_CSV)
print(f"  Root anchors: {len(root_anch)}, Branch anchors: {len(branch_anch)}")

umap_aligned = adata.obsm["X_umap_aligned"]
umap_orig    = adata.obsm.get("X_umap", None)
species_arr  = adata.obs["species"].values
obs_names    = adata.obs_names.astype(str)
obs_idx_map  = {name: i for i, name in enumerate(obs_names)}

# Find the Leiden cluster key at res 0.5
leiden_key = None
for res in ["0.5", "0.7", "0.3", "1.0"]:
    k = f"leiden_aligned_{res}"
    if k in adata.obs.columns:
        leiden_key = k
        leiden_res = res
        break
if leiden_key is None:
    for col in adata.obs.columns:
        if col.startswith("leiden"):
            leiden_key = col
            leiden_res = col.split("_")[-1]
            break

print(f"  Using leiden key: {leiden_key}")


# %% ── Helper: scatter plot ───────────────────────────────────────────────────────
def scatter(ax, coords, colors, sizes=PT_SIZE, alpha=0.5, **kw):
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes,
               alpha=alpha, rasterized=RASTERIZED, linewidths=0, **kw)


# %% ── Figure 1: Species overlay (shuffled draw order) ───────────────────────────
print("\nFig 1: species overlay ...")
fig, ax = plt.subplots(figsize=(8, 7))
order = np.random.permutation(len(umap_aligned))
c_arr = np.array([SPECIES_COLORS.get(s, "gray") for s in species_arr])
scatter(ax, umap_aligned[order], c_arr[order])
handles = [mpatches.Patch(color=c, label=sp) for sp, c in SPECIES_COLORS.items()]
ax.legend(handles=handles, markerscale=5, fontsize=10, loc="upper right")
ax.set_title("Cross-species UMAP — anchor-aligned")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/aligned_umap_species_overlay.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/aligned_umap_species_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Done.")


# %% ── Figure 2: Per-species highlight panels ─────────────────────────────────────
print("Fig 2: per-species panels ...")
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, sp in zip(axes, ["zebrafish", "mouse", "human"]):
    mask = species_arr == sp
    other_mask = ~mask
    # Background: other species gray
    scatter(ax, umap_aligned[other_mask], "lightgray", alpha=0.15)
    # Foreground: this species
    scatter(ax, umap_aligned[mask], SPECIES_COLORS[sp], alpha=0.7)
    ax.set_title(sp)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/aligned_umap_per_species.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/aligned_umap_per_species.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Done.")


# %% ── Figure 3: Anchor peaks highlighted ────────────────────────────────────────
print("Fig 3: anchor highlights ...")
root_ids   = set(root_anch[["peak_id_zf","peak_id_mm","peak_id_hs"]]
                 .values.flatten().astype(str).tolist())
branch_ids = set(branch_anch[["peak_id_zf","peak_id_mm","peak_id_hs"]]
                 .values.flatten().astype(str).tolist())

peak_color = np.full(len(obs_names), "lightgray", dtype=object)
for i, name in enumerate(obs_names):
    if name in root_ids:
        peak_color[i] = "red"
    elif name in branch_ids:
        peak_color[i] = "steelblue"

fig, ax = plt.subplots(figsize=(9, 7))
# Draw in order: gray → branch → root
for color, label, size in [("lightgray", "Non-anchor", 0.2),
                            ("steelblue", "Branch anchor", 3),
                            ("red",       "Root anchor",   4)]:
    mask = peak_color == color
    if mask.sum() > 0:
        scatter(ax, umap_aligned[mask], color, sizes=size, alpha=0.8 if color != "lightgray" else 0.2)
        # dummy patch for legend
        ax.plot([], [], "o", color=color, label=f"{label} (n={mask.sum():,})", ms=5)
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Anchor peaks on aligned UMAP")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/aligned_umap_anchors.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/aligned_umap_anchors.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Done.")


# %% ── Figure 4: Lineage labels per species ───────────────────────────────────────
print("Fig 4: lineage labels ...")
lineage_col = None
for col in ["lineage", "zf_lineage", "lineage_label"]:
    if col in adata.obs.columns:
        lineage_col = col
        break

if lineage_col:
    lineage_arr = adata.obs[lineage_col].fillna("unknown").astype(str).values
    unique_lineages = sorted(set(lineage_arr) - {"unknown", ""})
    n_lin = len(unique_lineages)
    palette = plt.cm.tab20(np.linspace(0, 1, max(n_lin, 1)))
    lin_colors = {l: palette[i] for i, l in enumerate(unique_lineages)}
    lin_colors["unknown"] = (0.8, 0.8, 0.8, 0.3)

    c_lin = np.array([lin_colors.get(l, (0.8,0.8,0.8,0.3)) for l in lineage_arr])
    fig, ax = plt.subplots(figsize=(9, 7))
    order = np.random.permutation(len(umap_aligned))
    scatter(ax, umap_aligned[order], c_lin[order])
    handles = [mpatches.Patch(color=lin_colors[l], label=l) for l in unique_lineages]
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    ax.set_title("Aligned UMAP — lineage labels")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/aligned_umap_lineage.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/aligned_umap_lineage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Done.")
else:
    print("  No lineage column found; skipping.")


# %% ── Figure 5: Leiden clusters ──────────────────────────────────────────────────
print("Fig 5: Leiden clusters ...")
if leiden_key:
    leiden_arr = adata.obs[leiden_key].astype(str).values
    uniq_cl    = sorted(set(leiden_arr), key=lambda x: int(x) if x.isdigit() else 999)
    n_cl = len(uniq_cl)
    pal  = plt.cm.tab20(np.linspace(0, 1, max(n_cl, 1)))
    cl_colors = {cl: pal[i] for i, cl in enumerate(uniq_cl)}
    c_cl = np.array([cl_colors.get(cl, (0.5,0.5,0.5,1)) for cl in leiden_arr])

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter(ax, umap_aligned, c_cl, alpha=0.6)
    handles = [mpatches.Patch(color=cl_colors[cl], label=f"C{cl}") for cl in uniq_cl[:20]]
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    ax.set_title(f"Aligned UMAP — Leiden (res={leiden_res})")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/aligned_umap_leiden.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/aligned_umap_leiden.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Species composition bar chart per cluster
    fig, ax = plt.subplots(figsize=(max(8, n_cl*0.5), 4))
    comp = pd.crosstab(leiden_arr, species_arr, normalize="index")
    comp = comp.reindex(columns=list(SPECIES_COLORS.keys()), fill_value=0)
    comp.plot(kind="bar", stacked=True, color=list(SPECIES_COLORS.values()), ax=ax)
    ax.set_xlabel("Leiden cluster")
    ax.set_ylabel("Fraction of peaks")
    ax.set_title(f"Species composition per Leiden cluster (res={leiden_res})")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/aligned_leiden_species_composition.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/aligned_leiden_species_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Done (Leiden + composition).")


# %% ── Figure 6: Side-by-side unaligned vs aligned ───────────────────────────────
print("Fig 6: side-by-side ...")
if umap_orig is not None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    c_arr = np.array([SPECIES_COLORS.get(s, "gray") for s in species_arr])
    for ax, coords, title in [(axes[0], umap_orig, "Unaligned UMAP"),
                               (axes[1], umap_aligned, "Aligned UMAP")]:
        order = np.random.permutation(len(coords))
        scatter(ax, coords[order], c_arr[order])
        handles = [mpatches.Patch(color=c, label=sp) for sp, c in SPECIES_COLORS.items()]
        ax.legend(handles=handles, fontsize=8, loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/aligned_vs_unaligned_sidebyside.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/aligned_vs_unaligned_sidebyside.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Done.")
else:
    print("  No unaligned UMAP stored; skipping side-by-side.")


# %% ── Figure 7: Top DE motifs per aligned Leiden cluster (heatmap) ──────────────
print("Fig 7: DE motif heatmap ...")
if leiden_key and adata.n_vars <= 2000:
    try:
        sc.tl.rank_genes_groups(adata, groupby=leiden_key, method="wilcoxon",
                                 use_raw=False, key_added="rank_genes_aligned")
        # Top N motifs per cluster
        N_TOP = 5
        motif_names = adata.var_names.tolist()
        top_motifs = []
        for cl in uniq_cl[:20]:
            top = sc.get.rank_genes_groups_df(adata, group=cl, key="rank_genes_aligned")
            top_motifs.extend(top["names"].head(N_TOP).tolist())
        top_motifs = list(dict.fromkeys(top_motifs))  # unique, ordered

        # Mean expression per cluster
        mean_expr = pd.DataFrame(
            {cl: adata[adata.obs[leiden_key] == cl, top_motifs].X.mean(axis=0).A1
                  if sp.issparse(adata[adata.obs[leiden_key] == cl, top_motifs].X)
                  else adata[adata.obs[leiden_key] == cl, top_motifs].X.mean(axis=0)
             for cl in uniq_cl[:20]},
            index=top_motifs
        )

        # Z-score rows
        mean_z = mean_expr.subtract(mean_expr.mean(axis=1), axis=0)
        mean_z = mean_z.divide(mean_expr.std(axis=1).clip(lower=1e-6), axis=0)

        fig, ax = plt.subplots(figsize=(max(10, len(uniq_cl)*0.6), max(8, len(top_motifs)*0.25)))
        sns.heatmap(mean_z, cmap="RdBu_r", center=0, ax=ax,
                    xticklabels=True, yticklabels=True, vmin=-2, vmax=2)
        ax.set_title(f"Top DE motifs per Leiden cluster (res={leiden_res})")
        ax.set_xlabel("Leiden cluster"); ax.set_ylabel("Motif")
        plt.tight_layout()
        fig.savefig(f"{FIG_DIR}/aligned_leiden_motif_heatmap.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(f"{FIG_DIR}/aligned_leiden_motif_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Heatmap saved.")
    except Exception as e:
        print(f"  Heatmap failed: {e}")
else:
    print("  Skipping heatmap (too many vars or no leiden_key).")


# %% ── Figure 8: Key ortholog TF families on UMAP ────────────────────────────────
print("Fig 8: TF family highlights ...")
gene_col = "nearest_gene_symbol" if "nearest_gene_symbol" in adata.obs.columns else "nearest_gene"
if gene_col in adata.obs.columns:
    gene_arr = adata.obs[gene_col].fillna("").str.lower().values
    for tf_family, members in TF_FAMILIES.items():
        members_lower = [m.lower() for m in members]
        tf_mask = np.array([g in members_lower for g in gene_arr])
        if tf_mask.sum() < 5:
            print(f"  {tf_family}: only {tf_mask.sum()} peaks; skipping")
            continue

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter(ax, umap_aligned[~tf_mask], "lightgray", sizes=0.2, alpha=0.2)
        scatter(ax, umap_aligned[tf_mask],  "crimson",   sizes=6,   alpha=0.9)
        ax.set_title(f"{tf_family} family peaks  (n={tf_mask.sum():,})")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.plot([], [], "o", color="crimson", label=f"{tf_family} genes (n={tf_mask.sum()})", ms=4)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(f"{FIG_DIR}/aligned_umap_TF_{tf_family}.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(f"{FIG_DIR}/aligned_umap_TF_{tf_family}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {tf_family}: {tf_mask.sum()} peaks highlighted.")
else:
    print(f"  No gene column found ({gene_col}); skipping TF highlights.")


# %% ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("All figures saved to:")
print(f"  {FIG_DIR}")
print("\nFiles generated:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith("aligned"):
        print(f"  {f}")
print("Done.")
