# %% Script 09l: 3-panel figures for top-5 hemangioblast peaks (figure-style)
#
# For each of the top-5 hemangioblast peaks (by V3 z-score), generate one figure
# with three vertically stacked panels:
#   A: Celltype accessibility bar plot (all 31 celltypes, lineage-grouped)
#   B: Temporal accessibility bar plot (6 timepoints within hemangioblasts)
#   C: TF motif position map (FDR < 0.05 enriched TFs only)
#
# Uses figure-style helpers: apply_style() + save_figure() for verified
# publication-quality output (Avenir Light, 8/6pt, editable PDF).
#
# Output: V3/combined_panels/hemangioblasts/
#   hemangioblasts_rank{N}_{gene}_combined_panel.{pdf,png}
#
# Env: single-cell-base

import os, re, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# %% Load figure-style helpers
_helpers_dirs = [
    *Path.home().glob(".claude/plugins/marketplaces/*/plugins/figure-style/scripts"),
    *Path.home().glob(".claude/plugins/cache/*/figure-style/*/scripts"),
]
for d in _helpers_dirs:
    if (d / "figure_helpers.py").exists():
        sys.path.insert(0, str(d))
        break

from figure_helpers import apply_style, save_figure
apply_style()

print("=== Script 09l: Hemangioblasts top-5 combined panels (figure-style) ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs"
OUTDIR  = f"{REPO}/figures/peak_parts_list/V3/combined_panels/hemangioblasts"
os.makedirs(OUTDIR, exist_ok=True)

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
V3_ZMAT     = f"{V3_DIR}/V3_specificity_matrix_celltype_level.h5ad"
TOP200_CSV  = f"{V3_DIR}/V3_all_celltypes_top200_peaks.csv"
POSITIONS_CSV  = f"{SCRATCH}/batches/hemangioblasts_hits.csv"
ENRICH_CSV  = f"{SCRATCH}/V3_top200_motif_enrichment_all31.csv"

TARGET_CT = "hemangioblasts"
TOP_N = 5
MIN_CELLS = 20

# %% Canonical color palette (intentional named colors per skill conventions)
sys.path.insert(0, f"{REPO}/scripts/utils")
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

# Distinct TF track colors (colorblind-safe)
TF_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]

# %% Load data
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
M = np.array(adata.X, dtype=np.float64)
obs = adata.obs.copy()
print(f"  {adata.shape}  ({time.time()-t0:.1f}s)")

print("Loading V3 z-score matrix ...", flush=True)
z_adata = ad.read_h5ad(V3_ZMAT)
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)

print("Loading top-200 peaks + FIMO data ...", flush=True)
top200 = pd.read_csv(TOP200_CSV)
positions = pd.read_csv(POSITIONS_CSV)
enrichment = pd.read_csv(ENRICH_CSV)

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

ct_col_indices = {}
for ct in reliable_celltypes:
    cols = [col for col, c in celltype_mapping.items()
            if c == ct and col in reliable_groups]
    ct_col_indices[ct] = [list(adata.var_names).index(c) for c in cols]

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
        return "insufficient data", 0.0
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
        pattern = "transient peak"
    else:
        pattern = "variable"
    return pattern, r_sq

# %% Enriched TFs for hemangioblasts (FDR < 0.05)
ct_enrich = enrichment[(enrichment["celltype"] == TARGET_CT) & (enrichment["fdr"] < 0.05)]
ct_enrich = ct_enrich.sort_values("enrichment_zscore", ascending=False)
enriched_tfs = ct_enrich["tf"].tolist()
print(f"\n{TARGET_CT} enriched TFs (FDR < 0.05): {len(enriched_tfs)}")

# %% Top 5 peaks
ct_peaks = top200[top200["celltype"] == TARGET_CT].nsmallest(TOP_N, "rank")
print(f"\nGenerating combined panels for top {TOP_N} peaks ...\n", flush=True)

for _, peak_row in ct_peaks.iterrows():
    pid = peak_row["peak_id"]
    rank = int(peak_row["rank"])
    zscore = peak_row["V3_zscore"]
    chrom = str(peak_row["chrom"])
    start = int(peak_row["start"])
    end = int(peak_row["end"])
    peak_len = end - start

    linked = str(peak_row.get("linked_gene", ""))
    nearest = str(peak_row.get("nearest_gene", ""))
    if linked in ("", "nan", "None"):
        linked = ""
    if nearest in ("", "nan", "None"):
        nearest = ""
    gene = linked if linked else nearest if nearest else f"chr{chrom}:{start}"

    peak_iloc = obs.index.get_loc(pid)
    peak_row_vals = M[peak_iloc]

    ct_vals = {}
    for ct in reliable_celltypes:
        ct_vals[ct] = np.mean(peak_row_vals[ct_col_indices[ct]])

    tp_vals = {}
    for tp in TIMEPOINT_ORDER:
        key = (TARGET_CT, tp)
        if key in ct_tp_col:
            tp_vals[tp] = peak_row_vals[ct_tp_col[key]]
    pattern, r_sq = classify_temporal_trend(tp_vals)

    peak_hits = positions[positions["peak_id"] == pid]
    peak_hits = peak_hits[peak_hits["tf"].isin(enriched_tfs)]
    tf_hits = {}
    for tf in enriched_tfs:
        tf_rows = peak_hits[peak_hits["tf"] == tf]
        if len(tf_rows) > 0:
            tf_hits[tf] = tf_rows
        if len(tf_hits) >= 10:
            break
    tf_order = list(tf_hits.keys())
    n_tracks = len(tf_order)

    # ── Build 3-panel figure ──
    # Sized for paper SI: ~7 inches wide. Panel heights proportional.
    motif_height = max(1.5, 0.6 + n_tracks * 0.22)
    fig_height = 2.3 + 1.3 + motif_height

    fig = plt.figure(figsize=(7.5, fig_height))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.3, 1.3, motif_height],
                          width_ratios=[3, 1.2], hspace=0.7, wspace=0.25)

    # Panel A: celltype bar
    ax1 = fig.add_subplot(gs[0, :])
    ct_order_present = [ct for ct in CELLTYPE_ORDER if ct in ct_vals]
    x_vals = [ct_vals[ct] for ct in ct_order_present]
    colors = [cell_type_color_dict.get(ct, "#cccccc") for ct in ct_order_present]
    edgecolors = ["black" if ct == TARGET_CT else "none" for ct in ct_order_present]
    linewidths = [1.2 if ct == TARGET_CT else 0.3 for ct in ct_order_present]

    ax1.bar(range(len(ct_order_present)), x_vals, color=colors,
            edgecolor=edgecolors, linewidth=linewidths)
    ax1.set_xticks(range(len(ct_order_present)))
    ax1.set_xticklabels([ct.replace("_", " ") for ct in ct_order_present],
                        rotation=90, fontsize=5)
    ax1.set_ylabel("mean log-norm accessibility")
    ax1.set_title(f"a. celltype profile — {gene}  |  chr{chrom}:{start:,}-{end:,}  |  "
                  f"V3 z={zscore:.1f}  |  rank {rank}",
                  loc="left")
    for boundary in LINEAGE_BOUNDARIES:
        if boundary < len(ct_order_present):
            ax1.axvline(boundary - 0.5, color="gray", ls="--", lw=0.4, alpha=0.5)

    # Panel B (left): temporal bar
    ax2 = fig.add_subplot(gs[1, 0])
    tp_present = [tp for tp in TIMEPOINT_ORDER if tp in tp_vals]
    tp_x = list(range(len(tp_present)))
    tp_y = [tp_vals[tp] for tp in tp_present]
    tp_colors = [timepoint_colors[tp] for tp in tp_present]
    tp_labels = [tp.replace("somites", "s") for tp in tp_present]

    ax2.bar(tp_x, tp_y, color=tp_colors, edgecolor="none", width=0.7)
    ax2.set_xticks(tp_x)
    ax2.set_xticklabels(tp_labels)
    ax2.set_ylabel("log-norm accessibility")
    ax2.set_xlabel(f"{TARGET_CT.replace('_', ' ')} timepoints (somites)")
    ax2.set_title(f"b. temporal profile — {pattern} (R²={r_sq:.2f})", loc="left")

    # Panel B (right): stats
    ax2b = fig.add_subplot(gs[1, 1])
    ax2b.axis("off")
    stats_text = (
        f"peak length: {peak_len} bp\n"
        f"V3 z-score: {zscore:.1f}\n"
        f"rank: {rank} / 200\n"
        f"enriched TFs (FDR<0.05): {n_tracks}\n"
        f"total motif hits: {sum(len(v) for v in tf_hits.values())}"
    )
    ax2b.text(0.05, 0.95, stats_text, transform=ax2b.transAxes,
              fontsize=6, verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="gray",
                        alpha=0.8, lw=0.4))

    # Panel C: motif position map
    ax3 = fig.add_subplot(gs[2, :])
    if n_tracks > 0:
        ax3.barh(n_tracks, peak_len, left=0, height=0.6,
                 color="#e0e0e0", edgecolor="#999999", linewidth=0.3)
        ax3.text(peak_len / 2, n_tracks, f"{peak_len} bp",
                 ha="center", va="center", fontsize=5, color="#555555")

        tf_color_map = {tf: TF_COLORS[i % len(TF_COLORS)]
                        for i, tf in enumerate(tf_order)}
        for track_i, tf in enumerate(tf_order):
            color = tf_color_map[tf]
            for _, h in tf_hits[tf].iterrows():
                width = h["hit_end"] - h["hit_start"]
                ax3.barh(track_i, width, left=h["hit_start"],
                         height=0.6, color=color, edgecolor="black",
                         linewidth=0.2, alpha=0.85)

        ax3.set_yticks(list(range(n_tracks)) + [n_tracks])
        ax3.set_yticklabels(tf_order + ["peak"])
        ax3.set_xlabel("position within peak (bp)")
        ax3.set_xlim(-5, peak_len + 5)
        ax3.set_ylim(-0.5, n_tracks + 1)
    else:
        ax3.text(0.5, 0.5,
                 "no enriched TF motifs (FDR < 0.05) in this peak",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=7, color="gray", style="italic")
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.set_xticks([]); ax3.set_yticks([])

    ax3.set_title(f"c. TF motif position map — {n_tracks} enriched TFs, "
                  f"{sum(len(v) for v in tf_hits.values())} hits",
                  loc="left")

    # Save via figure-style (PNG + PDF + verification)
    fname = f"{TARGET_CT}_rank{rank}_{gene}_combined_panel".replace("/", "_").replace(":", "_")
    save_figure(fig, f"{OUTDIR}/{fname}.png")

    print(f"  rank {rank}: {gene} ({peak_len}bp, z={zscore:.1f}, {n_tracks} TFs) → saved")

print(f"\nDone. Output: {OUTDIR}/")
print(f"End: {time.strftime('%c')}")
