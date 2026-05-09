# %% [markdown]
# # Script 09d: Advanced V3 Visualizations (refined)
#
# 1. Peak UMAP overlay — top-50 V3-specific peaks on 640K peak UMAP
#    - UMAP-visible color overrides for pale celltypes
#    - Z-score gradient coloring (single-hue, white→dark)
#    - Gene label fallback to genomic coords; annotate top 8
# 2. Temporal heatmap per celltype — two versions:
#    - Absolute log-norm (cividis) with outlier clipping
#    - Row-normalized z-score (coolwarm) to reveal temporal patterns
#    - V3 z-score side bar; unreliable timepoints marked with *
# 3. Gene-centric locus view — refined:
#    - Simplified legend (celltypes with ≥2 peaks only)
#    - Human-readable x-axis (Mb/kb)
#    - Minimum peak width for visibility
#    - Simplified gene body track (line + TSS arrow)
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu

# %% Imports
import os, re, time
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from matplotlib.collections import PatchCollection


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

print("=== Script 09d: Advanced V3 Visualizations (refined) ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
FIG_DIR = f"{REPO}/figures/peak_parts_list"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"

MASTER_H5AD = (f"{BASE}/data/annotated_data/objects_v2/"
               "peaks_by_ct_tp_master_anno.h5ad")

# %% Color palettes — from scripts/utils/module_dict_colors.py (canonical for this paper)
import sys as _sys
_sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

# UMAP-visible overrides: several canonical colors are too pale to see
# against the gray UMAP background. Darken only those, preserving hue identity.
umap_color_overrides = {
    'heart_myocardium':         '#2e8b4a',   # #ccebc5 pale mint → forest green
    'hemangioblasts':           '#c8a800',   # #ffed6f pale yellow → golden
    'notochord':                '#c45ab3',   # #f4cae4 pale pink → magenta
    'floor_plate':              '#888888',   # #d9d9d9 very light gray → dark gray
    'neural_telencephalon':     '#c97a5a',   # #fdcdac pale peach → dark salmon
    'pharyngeal_arches':        '#b09a00',   # #fff2ae pale yellow → dark yellow
    'neurons':                  '#7a8fcc',   # #cbd5e8 pale blue → medium blue
    'pronephros':               '#777777',   # #cccccc light gray → dark gray
    'primordial_germ_cells':    '#c07080',   # #f1e2cc pale cream → dusty rose
}

def get_umap_color(ct):
    return umap_color_overrides.get(ct, cell_type_color_dict.get(ct, "#ff0000"))

FOCAL_CELLTYPES = [
    "fast_muscle", "heart_myocardium", "neural_crest",
    "PSM", "notochord", "epidermis", "hemangioblasts",
]

TIMEPOINT_ORDER = ['0somites', '5somites', '10somites',
                   '15somites', '20somites', '30somites']
TP_INT = {tp: int(tp.replace('somites', '')) for tp in TIMEPOINT_ORDER}
MIN_CELLS = 20
TOP_N = 50

# %% Load data
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
M = np.array(adata.X, dtype=np.float64)
obs = adata.obs.copy()
umap_coords = adata.obsm["X_umap_2D"]  # (640830, 2)
print(f"  Shape: {adata.shape}, UMAP: {umap_coords.shape}  ({time.time()-t0:.1f}s)")

# %% Parse conditions
def parse_condition(cond):
    m = re.search(r"(\d+somites)$", cond)
    if not m:
        return cond, ""
    tp = m.group(1)
    ct = cond[:-(len(tp)+1)]
    return ct, tp

cond_meta = pd.DataFrame(
    [parse_condition(c) for c in adata.var_names],
    columns=["celltype", "timepoint"], index=adata.var_names,
)
cond_meta["n_cells"] = adata.var["n_cells"].values
cond_meta["reliable"] = cond_meta["n_cells"] >= MIN_CELLS
reliable_groups = cond_meta[cond_meta["reliable"]].index.tolist()
celltype_mapping = {col: parse_condition(col)[0] for col in adata.var_names}

# Column indices per (celltype, timepoint)
ct_tp_col = {}
for col in adata.var_names:
    ct, tp = parse_condition(col)
    if col in reliable_groups:
        ct_tp_col[(ct, tp)] = list(adata.var_names).index(col)

# %% Load V3 specificity matrix
print("Loading V3 specificity matrix ...", flush=True)
z_adata = ad.read_h5ad(f"{V3_DIR}/V3_specificity_matrix_celltype_level.h5ad")
Z_ct = np.array(z_adata.X)  # (640830, ~31)
ct_names = list(z_adata.var_names)
print(f"  Z_ct: {Z_ct.shape}")

# Pre-compute obs string columns needed by multiple sections
obs["nearest_gene_str"]    = obs["nearest_gene"].astype(str)
obs["linked_gene_str"]     = obs["linked_gene"].astype(str)
obs["associated_gene_str"] = obs["associated_gene"].astype(str)
obs["chrom_str"]           = obs["chrom"].astype(str)
obs["start_int"]           = obs["start"].astype(int)
obs["end_int"]             = obs["end"].astype(int)

def best_gene_label(idx):
    """Return best gene label for a peak: linked_gene > nearest_gene > coords."""
    linked = obs["linked_gene_str"].iloc[idx]
    nearest = obs["nearest_gene_str"].iloc[idx]
    if linked not in ("nan", "", "None"):
        return linked
    if nearest not in ("nan", "", "None"):
        return nearest
    chrom = obs["chrom_str"].iloc[idx]
    start = obs["start_int"].iloc[idx]
    end   = obs["end_int"].iloc[idx]
    return f"chr{chrom}:{start:,}-{end:,}"

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 1: Peak UMAP overlay (refined)
# ══════════════════════════════════════════════════════════════════════════════

print("\n--- Viz 1: Peak UMAP overlay ---")
out_umap = f"{FIG_DIR}/V3/peak_umap/per_celltype"
os.makedirs(out_umap, exist_ok=True)

# Subsample background for faster plotting
np.random.seed(42)
n_bg = min(100_000, len(umap_coords))
bg_idx = np.random.choice(len(umap_coords), n_bg, replace=False)

def make_single_hue_cmap(hex_color):
    """Create a colormap from near-white -> hex_color."""
    r, g, b = mcolors.to_rgb(hex_color)
    # lighten start color (80% white blend)
    start = (0.92 + 0.08*r, 0.92 + 0.08*g, 0.92 + 0.08*b)
    end   = (max(r * 0.6, 0), max(g * 0.6, 0), max(b * 0.6, 0))  # darken end
    return mcolors.LinearSegmentedColormap.from_list(
        "single_hue", [start, hex_color, end])

for ct in FOCAL_CELLTYPES:
    if ct not in ct_names:
        continue
    ct_idx = ct_names.index(ct)
    z_col = Z_ct[:, ct_idx]

    # Top N peaks
    top_idx = np.argsort(z_col)[::-1][:TOP_N]
    z_top = z_col[top_idx]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Background (gray)
    ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
               s=0.3, c="#e0e0e0", alpha=0.3, rasterized=True)

    # Top peaks colored by z-score (single-hue gradient)
    umap_color = get_umap_color(ct)
    cmap = make_single_hue_cmap(umap_color)
    norm = plt.Normalize(vmin=z_top.min(), vmax=z_top.max())
    dot_colors = cmap(norm(z_top))

    sc = ax.scatter(umap_coords[top_idx, 0], umap_coords[top_idx, 1],
                    s=35, c=dot_colors, edgecolors="black", linewidths=0.4,
                    zorder=5, alpha=0.92)

    # Colorbar inset (z-score)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.4)
    cbar.set_label("V3 z-score", fontsize=8)

    # Annotate top 8 with gene names (fallback to coords)
    for rank, i in enumerate(top_idx[:8]):
        label = best_gene_label(i)
        offset_x = 6 + (rank % 2) * 2
        offset_y = 6 - (rank // 2) * 4
        ax.annotate(label,
                    xy=(umap_coords[i, 0], umap_coords[i, 1]),
                    xytext=(offset_x, offset_y), textcoords="offset points",
                    fontsize=6.5, color="black", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                              alpha=0.8),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.set_title(f"V3 top-{TOP_N} peaks: {ct}\n(z-score range: "
                 f"{z_top[-1]:.1f}–{z_top[0]:.1f})",
                 fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#e0e0e0", label=f"All peaks (n={len(umap_coords):,})"),
        mpatches.Patch(facecolor=umap_color, edgecolor="black",
                       label=f"Top {TOP_N} {ct}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{out_umap}/{ct}_umap_overlay.pdf")
    fig.savefig(f"{out_umap}/{ct}_umap_overlay.png", dpi=300)
    plt.close(fig)
    print(f"  {ct}: saved")

# Combined panel
fig, axes = plt.subplots(2, 4, figsize=(28, 14))
axes = axes.flatten()

for ax_i, ct in enumerate(FOCAL_CELLTYPES):
    if ct not in ct_names:
        continue
    ax = axes[ax_i]
    ct_idx = ct_names.index(ct)
    z_col = Z_ct[:, ct_idx]
    top_idx = np.argsort(z_col)[::-1][:TOP_N]
    z_top = z_col[top_idx]

    umap_color = get_umap_color(ct)
    cmap = make_single_hue_cmap(umap_color)
    norm = plt.Normalize(vmin=z_top.min(), vmax=z_top.max())
    dot_colors = cmap(norm(z_top))

    ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
               s=0.1, c="#e0e0e0", alpha=0.2, rasterized=True)
    ax.scatter(umap_coords[top_idx, 0], umap_coords[top_idx, 1],
               s=20, c=dot_colors, edgecolors="black", linewidths=0.35,
               zorder=5, alpha=0.92)
    ax.set_title(ct, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

for j in range(len(FOCAL_CELLTYPES), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("V3 Celltype-Specific Peaks on Peak UMAP", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/V3/peak_umap/V3_peak_umap_overlay_combined.pdf")
fig.savefig(f"{FIG_DIR}/V3/peak_umap/V3_peak_umap_overlay_combined.png", dpi=300)
plt.close(fig)
print("  Combined panel saved")

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 2: Temporal heatmap per celltype (refined — two versions per celltype)
# ══════════════════════════════════════════════════════════════════════════════

print("\n--- Viz 2: Temporal heatmaps ---")
out_heatmap = f"{FIG_DIR}/V3/temporal/heatmaps"
os.makedirs(out_heatmap, exist_ok=True)

for ct in FOCAL_CELLTYPES:
    if ct not in ct_names:
        continue
    ct_idx_z = ct_names.index(ct)
    z_col = Z_ct[:, ct_idx_z]

    # Top N peaks
    top_idx = np.argsort(z_col)[::-1][:TOP_N]

    # Build temporal matrix: top_N × timepoints
    # Include ALL 6 timepoints; mark unreliable ones
    tp_labels_all = []
    col_indices_all = []
    tp_reliable = []
    for tp in TIMEPOINT_ORDER:
        if (ct, tp) in ct_tp_col:
            tp_labels_all.append(tp.replace("somites", "s"))
            col_indices_all.append(ct_tp_col[(ct, tp)])
            tp_reliable.append(True)
        else:
            # Check if this timepoint exists at all (unreliable)
            raw_col = f"{ct}_{tp}"
            if raw_col in adata.var_names:
                tp_labels_all.append(tp.replace("somites", "s") + "*")
                col_indices_all.append(list(adata.var_names).index(raw_col))
                tp_reliable.append(False)
            # else: timepoint doesn't exist at all for this celltype — skip

    temp_mat = M[top_idx][:, col_indices_all]  # (top_n, n_timepoints)

    # Row labels: gene name + z-score (with fallback to coords)
    row_labels = []
    for i in top_idx:
        gene = best_gene_label(i)
        z = z_col[i]
        row_labels.append(f"{gene} (z={z:.1f})")

    # Hierarchical clustering on rows (using reliable columns only for distance)
    reliable_cols = [j for j, r in enumerate(tp_reliable) if r]
    cluster_mat = temp_mat[:, reliable_cols] if reliable_cols else temp_mat
    if cluster_mat.shape[0] > 2 and cluster_mat.shape[1] > 1:
        row_linkage = hierarchy.linkage(pdist(cluster_mat, metric="euclidean"),
                                         method="ward")
        row_order = hierarchy.leaves_list(row_linkage)
    else:
        row_order = np.arange(temp_mat.shape[0])

    temp_mat_ordered = temp_mat[row_order]
    row_labels_ordered = [row_labels[i] for i in row_order]
    z_ordered = z_col[top_idx][row_order]

    # Row-normalized version (z-score across timepoints)
    row_mean = temp_mat_ordered.mean(axis=1, keepdims=True)
    row_std  = temp_mat_ordered.std(axis=1, keepdims=True)
    row_std  = np.maximum(row_std, 1e-6)
    temp_mat_norm = (temp_mat_ordered - row_mean) / row_std

    # ── Version A: Absolute accessibility (cividis, clipped at 95th percentile) ──
    vmax_abs = np.percentile(temp_mat_ordered, 95)
    fig_h = max(8, TOP_N * 0.22)

    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h),
                              gridspec_kw={"width_ratios": [0.04, 1]})
    ax_zscore, ax_main = axes

    # Left strip: V3 z-score per peak
    ax_zscore.imshow(z_ordered[:, None], aspect="auto", cmap="Reds",
                     vmin=0, vmax=z_ordered.max())
    ax_zscore.set_xticks([0])
    ax_zscore.set_xticklabels(["z"], fontsize=7)
    ax_zscore.set_yticks([])
    ax_zscore.set_title("", fontsize=1)

    # Main heatmap
    im = ax_main.imshow(temp_mat_ordered, aspect="auto", cmap="cividis",
                         vmin=0, vmax=vmax_abs, interpolation="nearest")

    ax_main.set_xticks(range(len(tp_labels_all)))
    ax_main.set_xticklabels(tp_labels_all, fontsize=9)
    ax_main.set_yticks(range(len(row_labels_ordered)))
    ax_main.set_yticklabels(row_labels_ordered, fontsize=6)
    ax_main.set_xlabel("Developmental timepoint (* = unreliable, n_cells < 20)")
    ax_main.set_title(f"{ct} — top {TOP_N} V3-specific peaks\n"
                      f"Absolute log-norm accessibility (cividis, clipped at 95th pct)",
                      fontsize=10)

    # Mark unreliable columns with hatching
    for j, reliable in enumerate(tp_reliable):
        if not reliable:
            ax_main.add_patch(plt.Rectangle(
                (j - 0.5, -0.5), 1, TOP_N,
                fill=True, facecolor="white", alpha=0.3,
                hatch="//", edgecolor="gray", linewidth=0))

    cbar = fig.colorbar(im, ax=ax_main, shrink=0.4, label="Log-norm accessibility")

    fig.tight_layout()
    fig.savefig(f"{out_heatmap}/{ct}_temporal_heatmap_abs.pdf")
    fig.savefig(f"{out_heatmap}/{ct}_temporal_heatmap_abs.png", dpi=300)
    plt.close(fig)

    # ── Version B: Row-normalized (coolwarm, centered at 0) ──
    vmax_norm = max(abs(temp_mat_norm.min()), abs(temp_mat_norm.max()))
    vmax_norm = min(vmax_norm, 3.0)  # cap at ±3 SD

    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h),
                              gridspec_kw={"width_ratios": [0.04, 1]})
    ax_zscore2, ax_main2 = axes

    ax_zscore2.imshow(z_ordered[:, None], aspect="auto", cmap="Reds",
                      vmin=0, vmax=z_ordered.max())
    ax_zscore2.set_xticks([0])
    ax_zscore2.set_xticklabels(["z"], fontsize=7)
    ax_zscore2.set_yticks([])

    im2 = ax_main2.imshow(temp_mat_norm, aspect="auto", cmap="coolwarm",
                           vmin=-vmax_norm, vmax=vmax_norm, interpolation="nearest")

    ax_main2.set_xticks(range(len(tp_labels_all)))
    ax_main2.set_xticklabels(tp_labels_all, fontsize=9)
    ax_main2.set_yticks(range(len(row_labels_ordered)))
    ax_main2.set_yticklabels(row_labels_ordered, fontsize=6)
    ax_main2.set_xlabel("Developmental timepoint (* = unreliable, n_cells < 20)")
    ax_main2.set_title(f"{ct} — top {TOP_N} V3-specific peaks\n"
                       f"Row-normalized (z-score across timepoints, coolwarm ±{vmax_norm:.1f})",
                       fontsize=10)

    # Mark unreliable columns
    for j, reliable in enumerate(tp_reliable):
        if not reliable:
            ax_main2.add_patch(plt.Rectangle(
                (j - 0.5, -0.5), 1, TOP_N,
                fill=True, facecolor="white", alpha=0.3,
                hatch="//", edgecolor="gray", linewidth=0))

    cbar2 = fig.colorbar(im2, ax=ax_main2, shrink=0.4,
                          label="Row z-score (across timepoints)")

    fig.tight_layout()
    fig.savefig(f"{out_heatmap}/{ct}_temporal_heatmap_norm.pdf")
    fig.savefig(f"{out_heatmap}/{ct}_temporal_heatmap_norm.png", dpi=300)
    plt.close(fig)

    # Keep the original filename pointing to normalized version (as the preferred one)
    import shutil
    shutil.copy(f"{out_heatmap}/{ct}_temporal_heatmap_norm.pdf",
                f"{out_heatmap}/{ct}_temporal_heatmap.pdf")
    shutil.copy(f"{out_heatmap}/{ct}_temporal_heatmap_norm.png",
                f"{out_heatmap}/{ct}_temporal_heatmap.png")

    print(f"  {ct}: abs + norm saved ({len(tp_labels_all)} timepoints, "
          f"{sum(tp_reliable)} reliable)")

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 3: Gene-centric locus view (refined)
# ══════════════════════════════════════════════════════════════════════════════

print("\n--- Viz 3: Gene-centric locus views ---")
out_locus = f"{FIG_DIR}/V3/gene_locus_views/per_gene"
os.makedirs(out_locus, exist_ok=True)

MARKER_GENES = [
    ("myod1",   "fast_muscle"),
    ("myhz1.3", "fast_muscle"),
    ("gata4",   "heart_myocardium"),
    ("tnnc1a",  "heart_myocardium"),
    ("sox10",   "neural_crest"),
    ("foxd3",   "neural_crest"),
    ("tbxta",   "notochord"),
    ("noto",    "notochord"),
    ("tp63",    "epidermis"),
    ("krt4",    "epidermis"),
    ("gata1a",  "hemangioblasts"),
    ("tal1",    "hemangioblasts"),
]

def format_genomic_pos(x, pos, locus_start, locus_len):
    """Format genomic position as Mb or kb."""
    if locus_len >= 500_000:
        return f"{x/1e6:.2f}"
    elif locus_len >= 50_000:
        return f"{x/1e6:.3f}"
    else:
        return f"{x/1e3:.1f}"

for gene_name, target_ct in MARKER_GENES:
    mask = ((obs["nearest_gene_str"] == gene_name) |
            (obs["linked_gene_str"] == gene_name) |
            (obs["associated_gene_str"] == gene_name))
    gene_peaks = obs[mask].copy()

    if len(gene_peaks) == 0:
        print(f"  {gene_name}: no peaks found — skipping")
        continue

    # V3 z-scores for target celltype
    ct_idx_z = ct_names.index(target_ct) if target_ct in ct_names else None
    if ct_idx_z is not None:
        peak_ilocs = [obs.index.get_loc(pid) for pid in gene_peaks.index]
        gene_peaks = gene_peaks.copy()
        gene_peaks["V3_zscore"] = Z_ct[peak_ilocs, ct_idx_z]
    else:
        peak_ilocs = [obs.index.get_loc(pid) for pid in gene_peaks.index]
        gene_peaks = gene_peaks.copy()
        gene_peaks["V3_zscore"] = 0.0

    # Max celltype per peak
    gene_peaks["max_ct"] = ""
    gene_peaks["max_ct_zscore"] = 0.0
    for pid, iloc in zip(gene_peaks.index, peak_ilocs):
        z_row = Z_ct[iloc]
        best = np.argmax(z_row)
        gene_peaks.loc[pid, "max_ct"] = ct_names[best]
        gene_peaks.loc[pid, "max_ct_zscore"] = z_row[best]

    gene_peaks = gene_peaks.sort_values("start_int")
    chrom = gene_peaks["chrom_str"].iloc[0]

    locus_start = gene_peaks["start_int"].min() - 5000
    locus_end   = gene_peaks["end_int"].max() + 5000
    locus_len   = locus_end - locus_start

    # Minimum peak width: 0.4% of locus width for visibility
    min_width = max(locus_len * 0.004, 200)

    # TSS from closest peak
    closest_peak = gene_peaks.loc[
        gene_peaks["distance_to_tss"].astype(float).abs().idxmin()]
    tss_pos = int(closest_peak["start_int"]) + int(float(closest_peak["distance_to_tss"]))

    # Simplified legend: only show celltypes with >= 2 peaks;
    # group singletons as "other (n=X)"
    ct_counts = gene_peaks["max_ct"].value_counts()
    main_cts  = set(ct_counts[ct_counts >= 2].index)
    singleton_cts = set(ct_counts[ct_counts == 1].index)
    n_other = len(singleton_cts)

    # X-axis formatter
    unit = "Mb" if locus_len >= 500_000 else ("Mb" if locus_len >= 50_000 else "kb")
    def make_formatter(ls, ll):
        def fmt(x, pos):
            return format_genomic_pos(x, pos, ls, ll)
        return fmt

    # ── Plot: 3 panels (z-score | gene body | max-celltype) ──
    fig, (ax_z, ax_gene, ax_ct) = plt.subplots(
        3, 1, figsize=(14, 6),
        gridspec_kw={"height_ratios": [3, 0.6, 1]},
        sharex=True)

    # Panel 1: V3 z-score (target celltype)
    target_color = cell_type_color_dict.get(target_ct, "#888")
    for _, pk in gene_peaks.iterrows():
        s = int(pk["start_int"])
        e = int(pk["end_int"])
        w = max(e - s, min_width)
        z = pk["V3_zscore"]
        rect = plt.Rectangle((s, 0), w, z,
                              facecolor=target_color,
                              edgecolor="black", linewidth=0.5, alpha=0.85)
        ax_z.add_patch(rect)

    ymax_z = max(gene_peaks["V3_zscore"].max() * 1.15, 2.5)
    ax_z.set_ylim(0, ymax_z)
    ax_z.set_ylabel(f"V3 z-score\n({target_ct})", fontsize=9)
    ax_z.set_title(f"{gene_name} locus  —  chr{chrom} "
                    f"(~{locus_len/1e3:.0f} kb)\n"
                    f"{len(gene_peaks)} peaks annotated",
                    fontsize=11)
    ax_z.axvline(tss_pos, color="red", ls="--", lw=1, alpha=0.7)
    ax_z.axhline(2.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax_z.axhline(4.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax_z.text(locus_end, 2.0, " z=2", fontsize=7, color="gray", va="bottom")
    ax_z.text(locus_end, 4.0, " z=4", fontsize=7, color="gray", va="bottom")

    # Panel 2: Simplified gene body track
    # Draw intron line across the locus + TSS arrow + gene name
    gene_mid = (locus_start + locus_end) / 2
    ax_gene.hlines(0.5, locus_start + locus_len * 0.02, locus_end - locus_len * 0.02,
                   colors="black", lw=1.5)
    # TSS arrow (right-facing = positive strand assumption)
    ax_gene.annotate("", xy=(tss_pos + locus_len * 0.02, 0.5),
                      xytext=(tss_pos, 0.5),
                      arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax_gene.text(tss_pos, 0.75, f" {gene_name}", fontsize=9,
                  color="red", fontweight="bold", va="bottom")
    ax_gene.set_ylim(0, 1)
    ax_gene.set_yticks([])
    ax_gene.set_ylabel("Gene", fontsize=8)
    ax_gene.axvline(tss_pos, color="red", ls="--", lw=1, alpha=0.5)
    ax_gene.spines["top"].set_visible(False)
    ax_gene.spines["right"].set_visible(False)
    ax_gene.spines["left"].set_visible(False)

    # Panel 3: Max celltype per peak
    for _, pk in gene_peaks.iterrows():
        s = int(pk["start_int"])
        e = int(pk["end_int"])
        w = max(e - s, min_width)
        max_ct = pk["max_ct"]
        color = cell_type_color_dict.get(max_ct, "#cccccc")
        rect = plt.Rectangle((s, 0), w, 1,
                              facecolor=color, edgecolor="black", linewidth=0.5)
        ax_ct.add_patch(rect)

    ax_ct.set_ylim(0, 1.3)
    ax_ct.set_yticks([])
    ax_ct.set_ylabel("Max\ncelltype", fontsize=9)
    ax_ct.axvline(tss_pos, color="red", ls="--", lw=1, alpha=0.7)

    # Simplified legend
    legend_patches = []
    for c in sorted(main_cts):
        legend_patches.append(
            mpatches.Patch(facecolor=cell_type_color_dict.get(c, "#ccc"),
                           edgecolor="black", label=c))
    if n_other > 0:
        legend_patches.append(
            mpatches.Patch(facecolor="#dddddd", edgecolor="black",
                           label=f"other ({n_other} celltypes, 1 peak each)"))

    ax_ct.legend(handles=legend_patches, loc="upper right", fontsize=7,
                 ncol=min(4, len(legend_patches)), framealpha=0.9)

    # X-axis: human-readable position labels
    ax_ct.set_xlim(locus_start, locus_end)
    formatter = make_formatter(locus_start, locus_len)
    ax_ct.xaxis.set_major_formatter(FuncFormatter(formatter))
    ax_ct.set_xlabel(f"Genomic position (chr{chrom}, {unit})", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{out_locus}/{gene_name}_{target_ct}_locus.pdf")
    fig.savefig(f"{out_locus}/{gene_name}_{target_ct}_locus.png", dpi=300)
    plt.close(fig)

    n_specific = (gene_peaks["V3_zscore"] >= 2.0).sum()
    print(f"  {gene_name} ({target_ct}): {len(gene_peaks)} peaks, "
          f"{n_specific} with z>=2, max_z={gene_peaks['V3_zscore'].max():.1f}, "
          f"legend: {len(main_cts)} main + {n_other} other celltypes")

print(f"\nDone.")
print(f"End: {time.strftime('%c')}")
