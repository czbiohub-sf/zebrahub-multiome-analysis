# %% Script 09c-marker: Peak profiles for known marker genes (reverse lookup)
#
# For each marker gene, finds the best peak (highest V3 z-score in its primary
# celltype) linked/nearest to that gene, then generates 2-panel bar plots:
#   Panel A: Celltype accessibility (mean across timepoints, all 31 celltypes)
#   Panel B: Temporal accessibility (within the primary celltype)
#
# Env: single-cell-base

import os, re, time
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

print("=== Script 09c-marker: Marker Gene Peak Profiles ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/figures/peak_parts_list/V3/peak_profiles/marker_genes"
os.makedirs(OUTDIR, exist_ok=True)

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
V3_ZMAT     = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_specificity_matrix_celltype_level.h5ad"

# %% Marker genes to profile
# (gene_name, primary_celltype, description)
MARKER_GENES = [
    ("myf5",   "muscle",           "Broadly accessible across muscle lineage"),
    ("myod1",  "fast_muscle",      "Committed myogenic — narrower than myf5"),
    ("myog",   "fast_muscle",      "Terminal muscle differentiation"),
    ("nkx2.5", "heart_myocardium", "Shared cardiac/vascular progenitor"),
    ("hand2",  "heart_myocardium", "Lateral plate mesoderm derivatives"),
    ("gata4",  "heart_myocardium", "Committed cardiac specification"),
]

MIN_CELLS = 20

# %% Color palette
import sys as _sys
_sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

CELLTYPE_ORDER = [
    'neural', 'neural_optic', 'neural_posterior', 'neural_telencephalon',
    'neurons', 'hindbrain', 'midbrain_hindbrain_boundary', 'optic_cup',
    'spinal_cord', 'differentiating_neurons', 'floor_plate', 'neural_floor_plate',
    'enteric_neurons', 'neural_crest',
    'somites', 'fast_muscle', 'muscle', 'PSM', 'NMPs', 'tail_bud', 'notochord',
    'lateral_plate_mesoderm', 'heart_myocardium', 'hematopoietic_vasculature',
    'hemangioblasts', 'pharyngeal_arches', 'pronephros', 'hatching_gland',
    'endoderm', 'endocrine_pancreas',
    'epidermis',
]
LINEAGE_BOUNDARIES = [14, 21, 28, 30]

TIMEPOINT_ORDER = ['0somites', '5somites', '10somites',
                   '15somites', '20somites', '30somites']
TP_INT = {tp: int(tp.replace('somites', '')) for tp in TIMEPOINT_ORDER}
n_tp = len(TIMEPOINT_ORDER)
_viridis = plt.cm.viridis(np.linspace(0, 1, n_tp))
timepoint_colors = dict(zip(TIMEPOINT_ORDER, _viridis))

# %% Load data
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
M = np.array(adata.X, dtype=np.float64)
obs = adata.obs.copy()
obs["linked_gene_str"]  = obs["linked_gene"].astype(str)
obs["nearest_gene_str"] = obs["nearest_gene"].astype(str)
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")

print("Loading V3 z-score matrix ...", flush=True)
z_adata = ad.read_h5ad(V3_ZMAT)
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)

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
    columns=["celltype", "timepoint"],
    index=adata.var_names,
)
cond_meta["n_cells"] = adata.var["n_cells"].values
cond_meta["reliable"] = cond_meta["n_cells"] >= MIN_CELLS

reliable_groups = cond_meta[cond_meta["reliable"]].index.tolist()
celltype_mapping = {col: parse_condition(col)[0] for col in adata.var_names}

reliable_celltypes = sorted(set(
    celltype_mapping[col] for col in reliable_groups
    if celltype_mapping[col] != "primordial_germ_cells"
))

# Precompute column indices per celltype
ct_col_indices = {}
for ct in reliable_celltypes:
    cols = [col for col, c in celltype_mapping.items()
            if c == ct and col in reliable_groups]
    ct_col_indices[ct] = [list(adata.var_names).index(c) for c in cols]

# Precompute column indices per (celltype, timepoint)
ct_tp_col = {}
for col in adata.var_names:
    ct, tp = parse_condition(col)
    if col in reliable_groups:
        ct_tp_col[(ct, tp)] = list(adata.var_names).index(col)

# %% Temporal trend classifier
def classify_temporal_trend(temporal_vals):
    tps = sorted(temporal_vals.keys(), key=lambda x: TP_INT[x])
    vals = [temporal_vals[tp] for tp in tps]
    if len(vals) < 3:
        return "insufficient_data", 0.0
    xs = np.array([TP_INT[tp] for tp in tps], dtype=float)
    ys = np.array(vals, dtype=float)
    mean_x, mean_y = xs.mean(), ys.mean()
    ss_xy = ((xs - mean_x) * (ys - mean_y)).sum()
    ss_xx = ((xs - mean_x) ** 2).sum()
    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    ss_yy = ((ys - mean_y) ** 2).sum()
    r_sq = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0
    cv = ys.std() / ys.mean() if ys.mean() > 0 else 0
    if cv < 0.15:
        pattern = "constitutive"
    elif slope > 0 and r_sq > 0.5:
        pattern = "increasing"
    elif slope < 0 and r_sq > 0.5:
        pattern = "decreasing"
    elif ys.argmax() > 0 and ys.argmax() < len(ys) - 1:
        pattern = "transient_peak"
    else:
        pattern = "variable"
    return pattern, r_sq

# %% Generate profiles for each marker gene
print("\nGenerating marker gene profiles ...\n", flush=True)

for gene, primary_ct, description in MARKER_GENES:
    # Find peaks linked/nearest to this gene
    mask_linked  = obs["linked_gene_str"] == gene
    mask_nearest = obs["nearest_gene_str"] == gene
    mask = mask_linked | mask_nearest
    n_peaks = mask.sum()

    if n_peaks == 0:
        print(f"  {gene}: NOT FOUND — skipping")
        continue

    # Select best peak: highest z-score in primary celltype
    peak_indices = np.where(mask.values)[0]
    ct_col_idx = ct_names.index(primary_ct)
    z_vals = Z_ct[peak_indices, ct_col_idx]
    best_local = np.argmax(z_vals)
    best_peak_idx = peak_indices[best_local]
    best_z = z_vals[best_local]
    link_type = "linked" if obs.iloc[best_peak_idx]["linked_gene_str"] == gene else "nearest"

    chrom = str(obs.iloc[best_peak_idx].get("chrom", ""))
    start = int(obs.iloc[best_peak_idx].get("start", 0))
    end   = int(obs.iloc[best_peak_idx].get("end", 0))

    print(f"  {gene} → {n_peaks} peaks, best: chr{chrom}:{start}-{end} "
          f"(z={best_z:.1f}, {link_type})")

    # Get celltype-level mean accessibility
    row = M[best_peak_idx]
    ct_vals = {}
    for ct in reliable_celltypes:
        ct_vals[ct] = np.mean(row[ct_col_indices[ct]])

    # Get temporal profile in primary celltype
    tp_vals = {}
    for tp in TIMEPOINT_ORDER:
        key = (primary_ct, tp)
        if key in ct_tp_col:
            tp_vals[tp] = row[ct_tp_col[key]]

    pattern, r_sq = classify_temporal_trend(tp_vals)

    # ── Two-panel figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5),
                                    gridspec_kw={"width_ratios": [3, 1.2]})

    # Panel A: Celltype bar plot
    ct_order_present = [ct for ct in CELLTYPE_ORDER if ct in ct_vals]
    x_vals = [ct_vals[ct] for ct in ct_order_present]
    colors = [cell_type_color_dict.get(ct, "#cccccc") for ct in ct_order_present]
    edgecolors = ["black" if ct == primary_ct else "none" for ct in ct_order_present]
    linewidths = [2.0 if ct == primary_ct else 0.5 for ct in ct_order_present]

    ax1.bar(range(len(ct_order_present)), x_vals, color=colors,
            edgecolor=edgecolors, linewidth=linewidths)
    ax1.set_xticks(range(len(ct_order_present)))
    ax1.set_xticklabels(ct_order_present, rotation=90, fontsize=7)
    ax1.set_ylabel("Mean log-norm accessibility")
    ax1.set_title(f"Celltype profile — {gene}\n"
                  f"chr{chrom}:{start}-{end}  |  V3 z={best_z:.1f} ({primary_ct})  |  "
                  f"{link_type} gene  |  {description}",
                  fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    for boundary in LINEAGE_BOUNDARIES:
        if boundary < len(ct_order_present):
            ax1.axvline(boundary - 0.5, color="gray", ls="--", lw=0.7, alpha=0.5)

    # Panel B: Temporal bar plot
    tp_present = [tp for tp in TIMEPOINT_ORDER if tp in tp_vals]
    tp_x = list(range(len(tp_present)))
    tp_y = [tp_vals[tp] for tp in tp_present]
    tp_colors = [timepoint_colors[tp] for tp in tp_present]
    tp_labels = [tp.replace("somites", "s") for tp in tp_present]

    ax2.bar(tp_x, tp_y, color=tp_colors, edgecolor="none", width=0.7)
    ax2.set_xticks(tp_x)
    ax2.set_xticklabels(tp_labels, fontsize=9)
    ax2.set_ylabel("Log-norm accessibility")
    ax2.set_xlabel(f"{primary_ct} timepoints")
    ax2.set_title(f"Temporal profile — {pattern}\n(R²={r_sq:.2f})", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fname = f"{gene}_{primary_ct}"
    fig.savefig(f"{OUTDIR}/{fname}.pdf")
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=300)
    plt.close(fig)

print(f"\nDone. Figures: {OUTDIR}/")
print(f"End: {time.strftime('%c')}")
