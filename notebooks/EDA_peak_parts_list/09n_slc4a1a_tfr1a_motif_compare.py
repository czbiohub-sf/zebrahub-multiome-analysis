# %% Script 09n: Motif position comparison — slc4a1a (rank1) vs tfr1a (rank5)
#
# Two peaks stacked vertically, each with top-5 enriched TF motif tracks.
# TFs from the same family share a color across both panels (easy cross-comparison).
#
# Output: V3/combined_panels/hemangioblasts/
#   slc4a1a_tfr1a_motif_compare.{png,pdf}
#
# Env: single-cell-base

import os, sys, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %% figure-style helpers
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

print("=== Script 09n: slc4a1a vs tfr1a motif comparison ===")
print(f"Start: {time.strftime('%c')}")

BASE   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO   = f"{BASE}/zebrahub-multiome-analysis"
V3_DIR = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs"
OUTDIR = f"{REPO}/figures/peak_parts_list/V3/combined_panels/hemangioblasts"
os.makedirs(OUTDIR, exist_ok=True)

TOP200 = f"{V3_DIR}/V3_all_celltypes_top200_peaks.csv"
HITS   = f"{SCRATCH}/batches/hemangioblasts_hits.csv"
ENRICH = f"{SCRATCH}/V3_top200_motif_enrichment_all31.csv"

TARGET_CT = "hemangioblasts"
TOP_N_TFS = 5

# %% Load data
top200 = pd.read_csv(TOP200)
hits   = pd.read_csv(HITS)
enr    = pd.read_csv(ENRICH)

hem_enr = enr[(enr["celltype"] == TARGET_CT) & (enr["fdr"] < 0.05)]\
            .sort_values("enrichment_zscore", ascending=False)

# Pick slc4a1a and tfr1a rows
peaks_of_interest = []
for gene in ("slc4a1a", "tfr1a"):
    row = top200[(top200["celltype"] == TARGET_CT) &
                 (top200["linked_gene"] == gene)].iloc[0]
    peaks_of_interest.append({
        "gene":     gene,
        "rank":     int(row["rank"]),
        "peak_id":  row["peak_id"],
        "chrom":    str(row["chrom"]),
        "start":    int(row["start"]),
        "end":      int(row["end"]),
        "zscore":   float(row["V3_zscore"]),
        "length":   int(row["end"]) - int(row["start"]),
    })

# %% Resolve top 5 enriched TFs present in each peak
def top5_tfs_for_peak(peak_id):
    peak_hits = hits[hits["peak_id"] == peak_id]
    tfs_with_hits = set(peak_hits["tf"])
    return hem_enr[hem_enr["tf"].isin(tfs_with_hits)].head(TOP_N_TFS)["tf"].tolist()

for p in peaks_of_interest:
    p["top5_tfs"] = top5_tfs_for_peak(p["peak_id"])
    print(f"  {p['gene']} (rank {p['rank']}, {p['length']}bp): {p['top5_tfs']}")

# %% Family assignment (strip trailing digits to get family prefix)
def tf_family(tf):
    m = re.match(r"^([A-Z]+?)\d+[A-Z]?$", tf)
    return m.group(1) if m else tf

all_tfs = sorted({tf for p in peaks_of_interest for tf in p["top5_tfs"]})
family_map = {tf: tf_family(tf) for tf in all_tfs}
families_ordered = []
for tf in all_tfs:
    fam = family_map[tf]
    if fam not in families_ordered:
        families_ordered.append(fam)

print(f"\nUnique families across both peaks: {families_ordered}")

# Color per family (colorblind-safe qualitative palette, distinct hues)
FAMILY_COLORS = {
    "GATA": "#e41a1c",   # red
    "KLF":  "#377eb8",   # blue
    "TRPS": "#4daf4a",   # green
    # fallback cycle for other families if needed
}
_fallback_palette = ["#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999",
                     "#66c2a5", "#fc8d62"]
for i, fam in enumerate(families_ordered):
    if fam not in FAMILY_COLORS:
        FAMILY_COLORS[fam] = _fallback_palette[i % len(_fallback_palette)]

tf_color = {tf: FAMILY_COLORS[family_map[tf]] for tf in all_tfs}

# %% Build figure — 2 panels stacked vertically, shared x-axis
max_len = max(p["length"] for p in peaks_of_interest)
per_panel_h = 1.6   # inches per panel
fig_h = 0.8 + 2 * per_panel_h + 0.4   # title + 2 panels + legend
fig, axes = plt.subplots(2, 1, figsize=(6.5, fig_h), sharex=True)

for ax, p in zip(axes, peaks_of_interest):
    tf_order = p["top5_tfs"]
    n_tracks = len(tf_order)

    # Peak bar on top row of each panel
    ax.barh(n_tracks, p["length"], left=0, height=0.55,
            color="#e0e0e0", edgecolor="#999999", linewidth=0.3)
    ax.text(p["length"] / 2, n_tracks, f"{p['length']} bp",
            ha="center", va="center", fontsize=6, color="#555555")

    # Motif tracks
    peak_hits = hits[hits["peak_id"] == p["peak_id"]]
    for track_i, tf in enumerate(tf_order):
        color = tf_color[tf]
        tf_hit_rows = peak_hits[peak_hits["tf"] == tf]
        for _, h in tf_hit_rows.iterrows():
            width = h["hit_end"] - h["hit_start"]
            ax.barh(track_i, width, left=h["hit_start"], height=0.6,
                    color=color, edgecolor="black", linewidth=0.2, alpha=0.9)

    ax.set_yticks(list(range(n_tracks)) + [n_tracks])
    ax.set_yticklabels(tf_order + ["peak"])
    ax.set_ylabel("TF motif tracks")
    ax.set_xlim(-5, max_len + 5)
    ax.set_ylim(-0.6, n_tracks + 1)
    ax.set_title(
        f"{p['gene']} (rank {p['rank']}) — chr{p['chrom']}:{p['start']:,}-{p['end']:,} "
        f"| V3 z={p['zscore']:.1f}",
        loc="left",
    )

axes[-1].set_xlabel("position within peak (bp)")

# %% Legend: TF family → color (shared, placed at figure bottom right of bottom panel)
legend_handles = [
    mpatches.Patch(facecolor=FAMILY_COLORS[fam], edgecolor="black",
                   linewidth=0.3, label=f"{fam} family")
    for fam in families_ordered
]
axes[-1].legend(handles=legend_handles, loc="lower right",
                bbox_to_anchor=(1.0, -0.55), ncol=len(families_ordered),
                frameon=False, fontsize=6, handlelength=1.5,
                columnspacing=1.0)

fig.suptitle(f"top-5 enriched TF motifs ({TARGET_CT.replace('_',' ')}, FDR<0.05) — "
             f"colors shared by family",
             y=0.995, fontsize=8)

save_figure(fig, f"{OUTDIR}/slc4a1a_tfr1a_motif_compare.png")

print(f"\nDone. Output: {OUTDIR}/slc4a1a_tfr1a_motif_compare.{{png,pdf}}")
print(f"End: {time.strftime('%c')}")
