# %% Standalone colorbar legend for the interesting6 UMAP panels
#
# Shows one gradient bar per celltype (light → dark = low → high z-score).
# Saved as a separate PDF so size can be adjusted independently.
#
# Output: figures/peak_parts_list/V3/peak_umap/V3_peak_umap_interesting6_legend.pdf
#         figures/peak_parts_list/V3/peak_umap/V3_peak_umap_interesting6_legend.png
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

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

REPO    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
OUT_DIR = f"{REPO}/figures/peak_parts_list/V3/peak_umap"

import sys as _sys
_sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

umap_color_overrides = {
    'hemangioblasts': '#c8a800',
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

# --- Figure layout ---
# One thin axes per celltype, stacked vertically.
# Width: narrow (legend-sized). Height: scales with n_celltypes.

n = len(CELLTYPES)
bar_h   = 0.28   # height of each gradient bar (inches)
pad     = 0.18   # vertical padding between bars
label_w = 1.6    # left margin for celltype labels (inches)
bar_w   = 2.4    # width of gradient bar (inches)
margin_b = 0.55  # bottom margin for x-axis label
margin_t = 0.25  # top margin

fig_w = label_w + bar_w + 0.3
fig_h = margin_b + margin_t + n * bar_h + (n - 1) * pad

fig = plt.figure(figsize=(fig_w, fig_h))

# Gradient data (0→1 across 256 steps)
grad = np.linspace(0, 1, 256).reshape(1, -1)

axes = []
for i, ct in enumerate(CELLTYPES):
    # Position from bottom: bottom-most celltype is index 0
    row = (n - 1 - i)   # flip so first celltype is at top
    bottom = (margin_b + row * (bar_h + pad)) / fig_h
    left   = label_w / fig_w
    width  = bar_w / fig_w
    height = bar_h / fig_h

    ax = fig.add_axes([left, bottom, width, height])

    umap_color = get_umap_color(ct)
    cmap = make_single_hue_cmap(umap_color)
    ax.imshow(grad, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#888888")

    # Celltype label to the left
    fig.text(
        (label_w - 0.08) / fig_w,
        bottom + height / 2,
        ct.replace("_", " "),
        ha="right", va="center",
        fontsize=8.5, fontstyle="italic",
        color="#222222",
    )
    axes.append(ax)

# Low / High tick labels below the bottom bar and above the top bar
bottom_ax = axes[-1]   # visually bottom (last celltype)
ax_pos = bottom_ax.get_position()

# "low" and "high" labels under the bottom bar
fig.text(ax_pos.x0, ax_pos.y0 - 0.04 / fig_h,
         "low", ha="left", va="top", fontsize=7.5, color="#666666")
fig.text(ax_pos.x0 + ax_pos.width, ax_pos.y0 - 0.04 / fig_h,
         "high", ha="right", va="top", fontsize=7.5, color="#222222")

# Shared x-axis label
fig.text(
    ax_pos.x0 + ax_pos.width / 2,
    0.01,
    "V3 specificity z-score",
    ha="center", va="bottom",
    fontsize=8, color="#333333",
)

# Arrow under bottom bar spanning full bar width (low → high)
ax_arrow = fig.add_axes([ax_pos.x0, margin_b * 0.38 / fig_h,
                          ax_pos.width, 0.001])
ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
ax_arrow.annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="#555555",
                                  lw=0.8, mutation_scale=8))
ax_arrow.axis("off")

fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_legend.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/V3_peak_umap_interesting6_legend.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved legend to {OUT_DIR}/V3_peak_umap_interesting6_legend.{{pdf,png}}")
