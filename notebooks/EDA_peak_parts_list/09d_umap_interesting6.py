# %% Combined UMAP panels for 6 high-interest celltypes
#
# Celltypes: epidermis, neural_crest, hemangioblasts,
#            hindbrain, optic_cup, hatching_gland
#
# Outputs (figures/peak_parts_list/V3/peak_umap/):
#   V3_peak_umap_interesting6_merged.{pdf,png}   — all 6 on one UMAP
#   V3_peak_umap_interesting6_panel.{pdf,png}    — 2×3 grid
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

# Publication-quality defaults
matplotlib.rcParams['pdf.fonttype'] = 42   # editable text in Illustrator
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['savefig.dpi']  = 300


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

BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUT_DIR = f"{REPO}/figures/peak_parts_list/V3/peak_umap"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"

import sys as _sys
_sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

umap_color_overrides = {
    'heart_myocardium':         '#2e8b4a',
    'hemangioblasts':           '#c8a800',
    'notochord':                '#c45ab3',
    'floor_plate':              '#888888',
    'neural_telencephalon':     '#c97a5a',
    'pharyngeal_arches':        '#b09a00',
    'neurons':                  '#7a8fcc',
    'pronephros':               '#777777',
    'primordial_germ_cells':    '#c07080',
}

def get_umap_color(ct):
    return umap_color_overrides.get(ct, cell_type_color_dict.get(ct, "#ff0000"))

def make_single_hue_cmap(hex_color):
    r, g, b = mcolors.to_rgb(hex_color)
    start = (0.92 + 0.08*r, 0.92 + 0.08*g, 0.92 + 0.08*b)
    end   = (max(r*0.6, 0), max(g*0.6, 0), max(b*0.6, 0))
    return mcolors.LinearSegmentedColormap.from_list("single_hue", [start, hex_color, end])

CELLTYPES = [
    "epidermis",
    "neural_crest",
    "hemangioblasts",
    "hindbrain",
    "optic_cup",
    "hatching_gland",
]
TOP_N = 50

# --- Load data ---
print("Loading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
obs = adata.obs.copy()
umap_coords = adata.obsm["X_umap_2D"]
obs["linked_gene_str"]  = obs["linked_gene"].astype(str)
obs["nearest_gene_str"] = obs["nearest_gene"].astype(str)
obs["chrom_str"] = obs["chrom"].astype(str)
obs["start_int"] = obs["start"].astype(int)
print(f"  {adata.shape}  ({time.time()-t0:.1f}s)")

print("Loading V3 specificity matrix ...", flush=True)
z_adata = ad.read_h5ad(f"{V3_DIR}/V3_specificity_matrix_celltype_level.h5ad")
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)

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

# Pre-compute per celltype
ct_data = {}
for ct in CELLTYPES:
    if ct not in ct_names:
        print(f"  WARNING: {ct} not in ct_names — skipping")
        continue
    ct_idx  = ct_names.index(ct)
    z_col   = Z_ct[:, ct_idx]
    top_idx = np.argsort(z_col)[::-1][:TOP_N]
    z_top   = z_col[top_idx]
    umap_color = get_umap_color(ct)
    cmap = make_single_hue_cmap(umap_color)
    norm = plt.Normalize(vmin=z_top.min(), vmax=z_top.max())
    dot_colors = cmap(norm(z_top))
    ct_data[ct] = dict(top_idx=top_idx, z_top=z_top,
                       umap_color=umap_color, cmap=cmap,
                       norm=norm, dot_colors=dot_colors)

# ══════════════════════════════════════════════════════════════════
# 1. Merged single-panel UMAP (all 6 celltypes)
# ══════════════════════════════════════════════════════════════════
print("\n--- Merged single-panel ---")

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
           s=0.3, c="#e0e0e0", alpha=0.3, rasterized=True)

legend_elements = [mpatches.Patch(facecolor="#e0e0e0",
                                   label=f"All peaks (n={len(umap_coords):,})")]
for ct, d in ct_data.items():
    ax.scatter(umap_coords[d["top_idx"], 0], umap_coords[d["top_idx"], 1],
               s=40, c=d["dot_colors"], edgecolors="black", linewidths=0.4,
               zorder=5, alpha=0.92)
    legend_elements.append(
        mpatches.Patch(facecolor=d["umap_color"], edgecolor="black",
                       label=f"Top {TOP_N} {ct.replace('_', ' ')}"))

ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5,
          framealpha=0.92, borderpad=0.6)
ax.set_title("V3 top-50 celltype-specific peaks\n"
             "(epidermis · neural crest · hemangioblasts · hindbrain · optic cup · hatching gland)",
             fontsize=11)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_merged.pdf")
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_merged.png", dpi=300)
plt.close(fig)
print("  Saved: V3_peak_umap_interesting6_merged")

# ══════════════════════════════════════════════════════════════════
# 2. 2×3 panel — each celltype in its own subplot
# ══════════════════════════════════════════════════════════════════
print("\n--- 2×3 panel ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for ax_i, (ct, d) in enumerate(ct_data.items()):
    ax = axes[ax_i]
    ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
               s=0.15, c="#e0e0e0", alpha=0.25, rasterized=True)
    ax.scatter(umap_coords[d["top_idx"], 0], umap_coords[d["top_idx"], 1],
               s=30, c=d["dot_colors"], edgecolors="black", linewidths=0.4,
               zorder=5, alpha=0.92)

    sm = plt.cm.ScalarMappable(cmap=d["cmap"], norm=d["norm"])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.5)
    cbar.set_label("V3 z-score", fontsize=7)

    z_top = d["z_top"]
    ax.set_title(f"{ct.replace('_', ' ')}\n"
                 f"z-score: {z_top[-1]:.1f}–{z_top[0]:.1f}",
                 fontsize=11, fontweight="bold", color=d["umap_color"])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

# hide unused axes (if any)
for j in range(len(ct_data), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("V3 top-50 celltype-specific peaks on peak UMAP", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_panel.pdf")
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_panel.png", dpi=300)
plt.close(fig)
print("  Saved: V3_peak_umap_interesting6_panel")

# ══════════════════════════════════════════════════════════════════
# 3. No-hue variants — flat color per celltype (no z-score gradient)
#    Shows spatial clustering without encoding z-score in color.
# ══════════════════════════════════════════════════════════════════
print("\n--- No-hue variants (flat colors) ---")

# 3a. Merged no-hue
fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
           s=0.3, c="#e0e0e0", alpha=0.3, rasterized=True)

legend_elements = [mpatches.Patch(facecolor="#e0e0e0",
                                   label=f"All peaks (n={len(umap_coords):,})")]
for ct, d in ct_data.items():
    ax.scatter(umap_coords[d["top_idx"], 0], umap_coords[d["top_idx"], 1],
               s=40, color=d["umap_color"], edgecolors="black", linewidths=0.4,
               zorder=5, alpha=0.92)
    legend_elements.append(
        mpatches.Patch(facecolor=d["umap_color"], edgecolor="black",
                       label=f"Top {TOP_N} {ct.replace('_', ' ')}"))

ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5,
          framealpha=0.92, borderpad=0.6)
ax.set_title("V3 top-50 celltype-specific peaks\n"
             "(epidermis · neural crest · hemangioblasts · hindbrain · optic cup · hatching gland)",
             fontsize=11)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_merged_nohue.pdf")
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_merged_nohue.png", dpi=300)
plt.close(fig)
print("  Saved: V3_peak_umap_interesting6_merged_nohue")

# 3b. 2×3 panel no-hue
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for ax_i, (ct, d) in enumerate(ct_data.items()):
    ax = axes[ax_i]
    ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
               s=0.15, c="#e0e0e0", alpha=0.25, rasterized=True)
    ax.scatter(umap_coords[d["top_idx"], 0], umap_coords[d["top_idx"], 1],
               s=30, color=d["umap_color"], edgecolors="black", linewidths=0.4,
               zorder=5, alpha=0.92)
    ax.set_title(f"{ct.replace('_', ' ')}\n(top {TOP_N} peaks)",
                 fontsize=11, fontweight="bold", color=d["umap_color"])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

for j in range(len(ct_data), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("V3 top-50 celltype-specific peaks on peak UMAP", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_panel_nohue.pdf")
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_panel_nohue.png", dpi=300)
plt.close(fig)
print("  Saved: V3_peak_umap_interesting6_panel_nohue")

print("\nDone.")
