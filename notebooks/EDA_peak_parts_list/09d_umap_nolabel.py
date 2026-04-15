# %% Generate no-label variants of V3 peak UMAP overlays
# Same z-score gradient coloring, no gene annotations.
# Output: {celltype}_umap_overlay_nolabel.{pdf,png}
# Env: /home/yang-joon.kim/.conda/envs/gReLu

import os, re, time
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


# ── Publication figure settings (exact pattern from 01_EDA_annotate_peak_umap.py) ──
import matplotlib as _mpl
_mpl.rcParams.update(_mpl.rcParamsDefault)   # 1. reset all rcParams to defaults
_mpl.rcParams['font.family'] = 'Arial'      # 2. explicit Arial font
_mpl.rcParams["pdf.fonttype"] = 42          # 3. editable text in Illustrator
_mpl.rcParams["ps.fonttype"]  = 42
import seaborn as _sns
_sns.set(style="whitegrid", context="paper") # 4. seaborn (after fonttype)
_mpl.rcParams["savefig.dpi"]  = 300         # 5. DPI re-set after sns.set()
# ────────────────────────────────────────────────────────────────────────────────

BASE   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO   = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR = f"{REPO}/figures/peak_parts_list/V3/peak_umap/per_celltype_nolabel"
V3_DIR = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"

import sys as _sys
_sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

# UMAP-visible overrides for pale canonical colors
umap_color_overrides = {
    'heart_myocardium':         '#2e8b4a',   # pale mint → forest green
    'hemangioblasts':           '#c8a800',   # pale yellow → golden
    'notochord':                '#c45ab3',   # pale pink → magenta
    'floor_plate':              '#888888',   # very light gray → dark gray
    'neural_telencephalon':     '#c97a5a',   # pale peach → dark salmon
    'pharyngeal_arches':        '#b09a00',   # pale yellow → dark yellow
    'neurons':                  '#7a8fcc',   # pale blue → medium blue
    'pronephros':               '#777777',   # light gray → dark gray
    'primordial_germ_cells':    '#c07080',   # pale cream → dusty rose
}

def get_umap_color(ct):
    return umap_color_overrides.get(ct, cell_type_color_dict.get(ct, "#ff0000"))

def make_single_hue_cmap(hex_color):
    r, g, b = mcolors.to_rgb(hex_color)
    start = (0.92 + 0.08*r, 0.92 + 0.08*g, 0.92 + 0.08*b)
    end   = (max(r*0.6, 0), max(g*0.6, 0), max(b*0.6, 0))
    return mcolors.LinearSegmentedColormap.from_list("single_hue",
                                                      [start, hex_color, end])

FOCAL_CELLTYPES = [
    "fast_muscle", "heart_myocardium", "neural_crest",
    "PSM", "notochord", "epidermis", "hemangioblasts",
]
TOP_N = 50

os.makedirs(OUTDIR, exist_ok=True)
print("Loading data ...", flush=True)
adata = ad.read_h5ad(MASTER_H5AD)
umap_coords = adata.obsm["X_umap_2D"]

z_adata = ad.read_h5ad(f"{V3_DIR}/V3_specificity_matrix_celltype_level.h5ad")
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)

np.random.seed(42)
bg_idx = np.random.choice(len(umap_coords), min(100_000, len(umap_coords)), replace=False)

for ct in FOCAL_CELLTYPES:
    if ct not in ct_names:
        continue
    z_col = Z_ct[:, ct_names.index(ct)]
    top_idx = np.argsort(z_col)[::-1][:TOP_N]
    z_top = z_col[top_idx]

    umap_color = get_umap_color(ct)
    cmap = make_single_hue_cmap(umap_color)
    norm = plt.Normalize(vmin=z_top.min(), vmax=z_top.max())
    dot_colors = cmap(norm(z_top))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
               s=0.3, c="#e0e0e0", alpha=0.3, rasterized=True)
    ax.scatter(umap_coords[top_idx, 0], umap_coords[top_idx, 1],
               s=35, c=dot_colors, edgecolors="black", linewidths=0.4,
               zorder=5, alpha=0.92)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.4)
    cbar.set_label("V3 z-score", fontsize=8)

    legend_elements = [
        mpatches.Patch(facecolor="#e0e0e0", label=f"All peaks (n={len(umap_coords):,})"),
        mpatches.Patch(facecolor=umap_color, edgecolor="black",
                       label=f"Top {TOP_N} {ct}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.set_title(f"V3 top-{TOP_N} peaks: {ct}\n(z-score: {z_top[-1]:.1f}–{z_top[0]:.1f})",
                 fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{ct}_umap_overlay_nolabel.pdf")
    fig.savefig(f"{OUTDIR}/{ct}_umap_overlay_nolabel.png", dpi=300)
    plt.close(fig)
    print(f"  {ct}: saved")

print("Done.")
