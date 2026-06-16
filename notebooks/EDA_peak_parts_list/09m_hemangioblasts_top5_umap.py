# %% Script 09m: Peak UMAP highlighting only top-5 hemangioblast peaks
#
# All ~640K peaks in gray; top-5 hemangioblast peaks (by V3 z-score) in dark yellow.
# Large, visible dot size for the 5 highlighted peaks.
#
# Output: V3/peak_umap/
#   V3_peak_umap_hemangioblasts_top5.{png,pdf}
#
# Env: single-cell-base

import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %% Load figure-style helpers
_helpers_dirs = [
    *Path.home().glob(".claude/plugins/marketplaces/*/plugins/figure-style/scripts"),
    *Path.home().glob(".claude/plugins/cache/*/figure-style/*/scripts"),
]
for d in _helpers_dirs:
    if (d / "figure_helpers.py").exists():
        sys.path.insert(0, str(d))
        break
from figure_helpers import apply_style, save_figure, strip_embedding_axes
apply_style()

print("=== Script 09m: top-5 hemangioblast peak UMAP ===")
print(f"Start: {time.strftime('%c')}")

BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
OUTDIR  = f"{REPO}/figures/peak_parts_list/V3/peak_umap"
os.makedirs(OUTDIR, exist_ok=True)

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
TOP200_CSV  = f"{V3_DIR}/V3_all_celltypes_top200_peaks.csv"

TARGET_CT  = "hemangioblasts"
DARK_YELLOW = "#c8a800"   # darkened from pale-yellow canonical #ffed6f
TOP_N = 5

# %% Load data
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
umap_coords = adata.obsm["X_umap_2D"]
obs = adata.obs
print(f"  {adata.shape}, UMAP {umap_coords.shape}  ({time.time()-t0:.1f}s)")

top200 = pd.read_csv(TOP200_CSV)
top5 = top200[top200["celltype"] == TARGET_CT].nsmallest(TOP_N, "rank")
top5_ilocs = [obs.index.get_loc(pid) for pid in top5["peak_id"]]

print(f"\nTop-5 {TARGET_CT} peaks:")
for _, r in top5.iterrows():
    gene = str(r.get("linked_gene", ""))
    if gene in ("", "nan", "None"):
        gene = str(r.get("nearest_gene", ""))
    print(f"  rank {int(r['rank'])}: {gene} (z={r['V3_zscore']:.1f})")

# %% Subsample background (rasterized) for speed + file size
np.random.seed(42)
n_bg = min(150_000, len(umap_coords))
bg_idx = np.random.choice(len(umap_coords), n_bg, replace=False)

# %% Build figure
fig, ax = plt.subplots(figsize=(5, 5))

# Background: all peaks in gray, small + transparent + rasterized
ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
           s=0.5, c="#cccccc", alpha=0.3, rasterized=True,
           linewidths=0)

# Top 5 peaks: large dark-yellow dots with black edge for visibility
ax.scatter(umap_coords[top5_ilocs, 0], umap_coords[top5_ilocs, 1],
           s=140, c=DARK_YELLOW, edgecolors="black",
           linewidths=0.8, zorder=10, alpha=1.0)

ax.set_title(f"top-5 {TARGET_CT.replace('_', ' ')} peaks (V3 z-score)")
strip_embedding_axes(ax)

# Legend
legend_elements = [
    mpatches.Patch(facecolor="#cccccc", edgecolor="none",
                   label=f"all peaks (n = {len(umap_coords):,})"),
    mpatches.Patch(facecolor=DARK_YELLOW, edgecolor="black",
                   label=f"top-5 {TARGET_CT.replace('_', ' ')}"),
]
ax.legend(handles=legend_elements, loc="upper right",
          frameon=False, fontsize=6)

save_figure(fig, f"{OUTDIR}/V3_peak_umap_hemangioblasts_top5.png")

# %% Labeled variant — gene name annotated at each top-5 dot
print("\nGenerating labeled variant ...", flush=True)

# Resolve gene labels (linked_gene > nearest_gene > "None")
def resolve_gene(row):
    for col in ("linked_gene", "nearest_gene"):
        val = str(row.get(col, ""))
        if val not in ("", "nan", "None", "NaN"):
            return val
    return "None"

gene_labels = [resolve_gene(r) for _, r in top5.iterrows()]

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
           s=0.5, c="#cccccc", alpha=0.3, rasterized=True,
           linewidths=0)
ax.scatter(umap_coords[top5_ilocs, 0], umap_coords[top5_ilocs, 1],
           s=140, c=DARK_YELLOW, edgecolors="black",
           linewidths=0.8, zorder=10, alpha=1.0)

# Gene labels with leader lines — fan outward from each dot
# Using offset_x/y pattern so labels don't overlap each other or the dot
offsets = [(18, 18), (-22, 20), (22, -18), (-22, -18), (28, 0)]
for (iloc, gene), (dx, dy) in zip(zip(top5_ilocs, gene_labels), offsets):
    x, y = umap_coords[iloc]
    ax.annotate(
        gene,
        xy=(x, y),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=6, color="black", fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec=DARK_YELLOW, lw=0.6, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5,
                        connectionstyle="arc3,rad=0.1"),
        zorder=11,
    )

ax.set_title(f"top-5 {TARGET_CT.replace('_', ' ')} peaks (V3 z-score)")
strip_embedding_axes(ax)

legend_elements = [
    mpatches.Patch(facecolor="#cccccc", edgecolor="none",
                   label=f"all peaks (n = {len(umap_coords):,})"),
    mpatches.Patch(facecolor=DARK_YELLOW, edgecolor="black",
                   label=f"top-5 {TARGET_CT.replace('_', ' ')}"),
]
ax.legend(handles=legend_elements, loc="upper right",
          frameon=False, fontsize=6)

save_figure(fig, f"{OUTDIR}/V3_peak_umap_hemangioblasts_top5_labeled.png")

print(f"\nDone. Outputs:")
print(f"  {OUTDIR}/V3_peak_umap_hemangioblasts_top5.{{png,pdf}}")
print(f"  {OUTDIR}/V3_peak_umap_hemangioblasts_top5_labeled.{{png,pdf}}")
print(f"End: {time.strftime('%c')}")
