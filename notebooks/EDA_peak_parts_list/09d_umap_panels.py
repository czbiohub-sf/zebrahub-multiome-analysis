# %% Generate all-celltypes individual UMAPs + multi-celltype panel figures
#
# Outputs (all in figures/peak_parts_list/V3/peak_umap/):
#   all_celltypes/{ct}_umap_overlay.{pdf,png}            — labeled
#   all_celltypes/{ct}_umap_overlay_nolabel.{pdf,png}    — no labels
#   V3_peak_umap_merged5.{pdf,png}                       — 5 celltypes, one UMAP
#   V3_peak_umap_merged5_nolabel.{pdf,png}
#   V3_peak_umap_highlight6.{pdf,png}                    — 2×3 panel
#   V3_peak_umap_highlight6_nolabel.{pdf,png}
#
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
FIG_DIR = f"{REPO}/figures/peak_parts_list"
OUT_ALL = f"{FIG_DIR}/V3/peak_umap/all_celltypes"
OUT_PANEL = f"{FIG_DIR}/V3/peak_umap"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"

os.makedirs(OUT_ALL, exist_ok=True)

# --- Color palette (canonical for this paper) ---
import sys as _sys
_sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

# UMAP-visible overrides: darken pale canonical colors
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
    return mcolors.LinearSegmentedColormap.from_list("single_hue", [start, hex_color, end])

TOP_N = 50

# --- Load data ---
print("Loading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
obs = adata.obs.copy()
umap_coords = adata.obsm["X_umap_2D"]
obs["linked_gene_str"]  = obs["linked_gene"].astype(str)
obs["nearest_gene_str"] = obs["nearest_gene"].astype(str)
obs["chrom_str"]  = obs["chrom"].astype(str)
obs["start_int"]  = obs["start"].astype(int)
obs["end_int"]    = obs["end"].astype(int)
print(f"  {adata.shape}, UMAP {umap_coords.shape}  ({time.time()-t0:.1f}s)")

print("Loading V3 specificity matrix ...", flush=True)
z_adata = ad.read_h5ad(f"{V3_DIR}/V3_specificity_matrix_celltype_level.h5ad")
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)
print(f"  Z_ct: {Z_ct.shape}, celltypes: {ct_names}")

np.random.seed(42)
bg_idx = np.random.choice(len(umap_coords), min(100_000, len(umap_coords)), replace=False)

def best_gene_label(idx):
    linked  = obs["linked_gene_str"].iloc[idx]
    nearest = obs["nearest_gene_str"].iloc[idx]
    if linked not in ("nan", "", "None"):
        return linked
    if nearest not in ("nan", "", "None"):
        return nearest
    return f"chr{obs['chrom_str'].iloc[idx]}:{obs['start_int'].iloc[idx]:,}"

# ══════════════════════════════════════════════════════════════════
# 1. Individual UMAPs for ALL celltypes (labeled + nolabel)
# ══════════════════════════════════════════════════════════════════
print("\n--- All-celltypes individual UMAPs ---")

for ct in ct_names:
    ct_idx = ct_names.index(ct)
    z_col  = Z_ct[:, ct_idx]
    top_idx = np.argsort(z_col)[::-1][:TOP_N]
    z_top   = z_col[top_idx]

    umap_color = get_umap_color(ct)
    cmap = make_single_hue_cmap(umap_color)
    norm = plt.Normalize(vmin=z_top.min(), vmax=z_top.max())
    dot_colors = cmap(norm(z_top))

    # --- Labeled version ---
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
    for rank, i in enumerate(top_idx[:8]):
        label = best_gene_label(i)
        offset_x = 6 + (rank % 2) * 2
        offset_y = 6 - (rank // 2) * 4
        ax.annotate(label,
                    xy=(umap_coords[i, 0], umap_coords[i, 1]),
                    xytext=(offset_x, offset_y), textcoords="offset points",
                    fontsize=6.5, color="black", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    legend_elements = [
        mpatches.Patch(facecolor="#e0e0e0", label=f"All peaks (n={len(umap_coords):,})"),
        mpatches.Patch(facecolor=umap_color, edgecolor="black", label=f"Top {TOP_N} {ct}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.set_title(f"V3 top-{TOP_N} peaks: {ct}\n(z-score: {z_top[-1]:.1f}–{z_top[0]:.1f})", fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{OUT_ALL}/{ct}_umap_overlay.pdf")
    fig.savefig(f"{OUT_ALL}/{ct}_umap_overlay.png", dpi=300)
    plt.close(fig)

    # --- No-label version ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
               s=0.3, c="#e0e0e0", alpha=0.3, rasterized=True)
    ax.scatter(umap_coords[top_idx, 0], umap_coords[top_idx, 1],
               s=35, c=dot_colors, edgecolors="black", linewidths=0.4,
               zorder=5, alpha=0.92)
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=ax, fraction=0.025, pad=0.01, shrink=0.4)
    cbar2.set_label("V3 z-score", fontsize=8)
    legend_elements2 = [
        mpatches.Patch(facecolor="#e0e0e0", label=f"All peaks (n={len(umap_coords):,})"),
        mpatches.Patch(facecolor=umap_color, edgecolor="black", label=f"Top {TOP_N} {ct}"),
    ]
    ax.legend(handles=legend_elements2, loc="upper right", fontsize=9)
    ax.set_title(f"V3 top-{TOP_N} peaks: {ct}\n(z-score: {z_top[-1]:.1f}–{z_top[0]:.1f})", fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{OUT_ALL}/{ct}_umap_overlay_nolabel.pdf")
    fig.savefig(f"{OUT_ALL}/{ct}_umap_overlay_nolabel.png", dpi=300)
    plt.close(fig)

    print(f"  {ct}: saved")

# ══════════════════════════════════════════════════════════════════
# 2. Merged 5-celltype UMAP (heart_myocardium, neural_crest,
#    notochord, epidermis, hemangioblasts)
# ══════════════════════════════════════════════════════════════════
print("\n--- Merged 5-celltype UMAP ---")

MERGED_CTS = [
    "heart_myocardium",
    "neural_crest",
    "notochord",
    "epidermis",
    "hemangioblasts",
]

def make_merged_umap(include_labels, outfile_base):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
               s=0.3, c="#e0e0e0", alpha=0.3, rasterized=True)

    legend_elements = [mpatches.Patch(facecolor="#e0e0e0",
                                       label=f"All peaks (n={len(umap_coords):,})")]
    for ct in MERGED_CTS:
        if ct not in ct_names:
            continue
        ct_idx = ct_names.index(ct)
        z_col  = Z_ct[:, ct_idx]
        top_idx = np.argsort(z_col)[::-1][:TOP_N]
        z_top   = z_col[top_idx]
        umap_color = get_umap_color(ct)
        cmap = make_single_hue_cmap(umap_color)
        norm = plt.Normalize(vmin=z_top.min(), vmax=z_top.max())
        dot_colors = cmap(norm(z_top))
        ax.scatter(umap_coords[top_idx, 0], umap_coords[top_idx, 1],
                   s=40, c=dot_colors, edgecolors="black", linewidths=0.4,
                   zorder=5, alpha=0.92)
        if include_labels:
            for rank, i in enumerate(top_idx[:3]):
                label = best_gene_label(i)
                offset_x = 5 + (rank % 2) * 3
                offset_y = 5 - (rank // 2) * 5
                ax.annotate(label,
                            xy=(umap_coords[i, 0], umap_coords[i, 1]),
                            xytext=(offset_x, offset_y), textcoords="offset points",
                            fontsize=5.5, color="black", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
                            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
        legend_elements.append(
            mpatches.Patch(facecolor=umap_color, edgecolor="black",
                           label=f"Top {TOP_N} {ct}"))

    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9, borderpad=0.5)
    ax.set_title("V3 top-50 peaks per celltype (5 celltypes)", fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{outfile_base}.pdf")
    fig.savefig(f"{outfile_base}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {outfile_base}")

make_merged_umap(include_labels=True,  outfile_base=f"{OUT_PANEL}/V3_peak_umap_merged5")
make_merged_umap(include_labels=False, outfile_base=f"{OUT_PANEL}/V3_peak_umap_merged5_nolabel")

# ══════════════════════════════════════════════════════════════════
# 3. Highlight 6-celltype 2×3 panel
#    (fast_muscle, heart_myocardium, neural_crest,
#     notochord, epidermis, hemangioblasts)
# ══════════════════════════════════════════════════════════════════
print("\n--- Highlight 6-celltype 2×3 panel ---")

HIGHLIGHT_CTS = [
    "fast_muscle",
    "heart_myocardium",
    "neural_crest",
    "notochord",
    "epidermis",
    "hemangioblasts",
]

def make_highlight6(include_labels, outfile_base):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax_i, ct in enumerate(HIGHLIGHT_CTS):
        ax = axes[ax_i]
        if ct not in ct_names:
            ax.set_visible(False)
            continue
        ct_idx = ct_names.index(ct)
        z_col  = Z_ct[:, ct_idx]
        top_idx = np.argsort(z_col)[::-1][:TOP_N]
        z_top   = z_col[top_idx]
        umap_color = get_umap_color(ct)
        cmap = make_single_hue_cmap(umap_color)
        norm = plt.Normalize(vmin=z_top.min(), vmax=z_top.max())
        dot_colors = cmap(norm(z_top))

        ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
                   s=0.15, c="#e0e0e0", alpha=0.25, rasterized=True)
        ax.scatter(umap_coords[top_idx, 0], umap_coords[top_idx, 1],
                   s=30, c=dot_colors, edgecolors="black", linewidths=0.4,
                   zorder=5, alpha=0.92)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.5)
        cbar.set_label("V3 z-score", fontsize=7)

        if include_labels:
            for rank, i in enumerate(top_idx[:5]):
                label = best_gene_label(i)
                offset_x = 5 + (rank % 2) * 3
                offset_y = 5 - (rank // 2) * 5
                ax.annotate(label,
                            xy=(umap_coords[i, 0], umap_coords[i, 1]),
                            xytext=(offset_x, offset_y), textcoords="offset points",
                            fontsize=5.5, color="black", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
                            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))

        ax.set_title(ct.replace("_", " "), fontsize=11, fontweight="bold",
                     color=umap_color if ct not in umap_color_overrides else umap_color_overrides[ct])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("V3 top-50 celltype-specific peaks on UMAP", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{outfile_base}.pdf")
    fig.savefig(f"{outfile_base}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {outfile_base}")

make_highlight6(include_labels=True,  outfile_base=f"{OUT_PANEL}/V3_peak_umap_highlight6")
make_highlight6(include_labels=False, outfile_base=f"{OUT_PANEL}/V3_peak_umap_highlight6_nolabel")

print("\nDone.")
