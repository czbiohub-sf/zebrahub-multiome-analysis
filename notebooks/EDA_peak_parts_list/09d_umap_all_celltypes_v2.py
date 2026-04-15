# %% Script 09d-v2: Peak UMAPs for all 31 celltypes (top-50 + top-200)
#
# Flat color (no z-score gradient), no grid, no colorbar.
# Generates labeled + nolabel versions for both top-50 and top-200.
#
# Output: V3/peak_umap/all_celltypes_V2/
#   {celltype}_top50_umap.{pdf,png}
#   {celltype}_top50_umap_nolabel.{pdf,png}
#   {celltype}_top200_umap.{pdf,png}
#   {celltype}_top200_umap_nolabel.{pdf,png}
#
# Env: single-cell-base

import os, time, sys
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

print("=== Script 09d-v2: All-celltypes UMAPs (top-50 + top-200) ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/figures/peak_parts_list/V3/peak_umap/all_celltypes_V2"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
os.makedirs(OUTDIR, exist_ok=True)

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"

sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

umap_color_overrides = {
    'heart_myocardium':      '#2e8b4a',
    'hemangioblasts':        '#c8a800',
    'notochord':             '#c45ab3',
    'floor_plate':           '#888888',
    'neural_telencephalon':  '#c97a5a',
    'pharyngeal_arches':     '#b09a00',
    'neurons':               '#7a8fcc',
    'pronephros':            '#777777',
    'primordial_germ_cells': '#c07080',
}

def get_umap_color(ct):
    return umap_color_overrides.get(ct, cell_type_color_dict.get(ct, "#ff0000"))

# %% Load data
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
obs = adata.obs.copy()
umap_coords = adata.obsm["X_umap_2D"]
obs["linked_gene_str"]  = obs["linked_gene"].astype(str)
obs["nearest_gene_str"] = obs["nearest_gene"].astype(str)
obs["chrom_str"]        = obs["chrom"].astype(str)
obs["start_int"]        = obs["start"].astype(int)
n_total = len(umap_coords)
print(f"  {adata.shape}  ({time.time()-t0:.1f}s)")

print("Loading V3 z-score matrix ...", flush=True)
z_adata = ad.read_h5ad(f"{V3_DIR}/V3_specificity_matrix_celltype_level.h5ad")
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)
print(f"  {len(ct_names)} celltypes")

# Background points
np.random.seed(42)
bg_idx = np.random.choice(n_total, min(100_000, n_total), replace=False)

def best_gene_label(idx):
    linked  = obs["linked_gene_str"].iloc[idx]
    nearest = obs["nearest_gene_str"].iloc[idx]
    if linked not in ("nan", "", "None"):
        return linked
    if nearest not in ("nan", "", "None"):
        return nearest
    return f"chr{obs['chrom_str'].iloc[idx]}:{obs['start_int'].iloc[idx]:,}"

# %% Generate UMAPs
print(f"\nGenerating UMAPs for {len(ct_names)} celltypes ...\n", flush=True)

for ct in ct_names:
    ct_col = ct_names.index(ct)
    z_col  = Z_ct[:, ct_col]
    color  = get_umap_color(ct)

    # Get top-200 indices (top-50 is a subset)
    top200_idx = np.argsort(z_col)[::-1][:200]
    top50_idx  = top200_idx[:50]

    for top_n, top_idx in [(50, top50_idx), (200, top200_idx)]:
        for labeled in [True, False]:
            suffix = f"top{top_n}_umap" + ("" if labeled else "_nolabel")

            fig, ax = plt.subplots(figsize=(8, 8))

            # Grey background
            ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
                       s=0.3, c="#e0e0e0", alpha=0.3, rasterized=True)

            # Top-N peaks in flat celltype color
            ax.scatter(umap_coords[top_idx, 0], umap_coords[top_idx, 1],
                       s=30, color=color, edgecolors="black", linewidths=0.4,
                       zorder=5, alpha=0.92)

            # Gene labels (top 8, labeled version only)
            if labeled:
                for i, idx in enumerate(top_idx[:8]):
                    label = best_gene_label(idx)
                    ax.annotate(label,
                                xy=(umap_coords[idx, 0], umap_coords[idx, 1]),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=6,
                                bbox=dict(boxstyle="round,pad=0.15",
                                          fc="white", ec="gray", alpha=0.7))

            # Legend
            legend = [
                mpatches.Patch(facecolor="#e0e0e0",
                               label=f"All peaks (n={n_total:,})"),
                mpatches.Patch(facecolor=color, edgecolor="black",
                               label=f"Top {top_n} {ct.replace('_', ' ')}"),
            ]
            ax.legend(handles=legend, loc="upper right", fontsize=9)

            ax.set_title(f"V3 top-{top_n} peaks: {ct.replace('_', ' ')}",
                         fontsize=11)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal")

            # Clean: no grid, no ticks, no spines
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            fig.tight_layout()
            fig.savefig(f"{OUTDIR}/{ct}_{suffix}.pdf", bbox_inches="tight")
            fig.savefig(f"{OUTDIR}/{ct}_{suffix}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"  {ct}: done (4 versions × pdf+png)")

print(f"\nDone. {len(ct_names)} celltypes × 4 versions × 2 formats = "
      f"{len(ct_names) * 4 * 2} files")
print(f"Output: {OUTDIR}/")
print(f"End: {time.strftime('%c')}")
