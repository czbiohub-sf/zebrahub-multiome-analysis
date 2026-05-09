# %% Script 09i-v2: Motif position maps from precomputed top-200 FIMO data
#
# Reads the precomputed FIMO positions from scratch and generates motif
# position maps for the top 5 peaks per celltype (all 31 celltypes).
# Uses only TFs that are significantly enriched (FDR < 0.05) in that celltype.
#
# Env: single-cell-base (no pysam/pymemesuite needed — reads precomputed CSVs)

import os, time
import numpy as np
import pandas as pd
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

print("=== Script 09i-v2: Motif Position Maps (from precomputed top-200 FIMO) ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
REPO    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs"
FIG_DIR = f"{REPO}/figures/peak_parts_list/V3/motif_position_maps_V2_top200"
os.makedirs(FIG_DIR, exist_ok=True)

TOP_N_PEAKS = 5   # top peaks per celltype to visualize
TOP_N_TFS   = 10  # max TFs to show per peak

# %% Load precomputed data
print("Loading precomputed FIMO data ...", flush=True)
positions = pd.read_csv(f"{SCRATCH}/V3_top200_motif_positions.csv")
enrichment = pd.read_csv(f"{SCRATCH}/V3_top200_motif_enrichment_all31.csv")
top200 = pd.read_csv(f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_all_celltypes_top200_peaks.csv")

print(f"  Positions: {len(positions)} hits")
print(f"  Enrichment: {len(enrichment)} rows")
print(f"  Top-200: {len(top200)} peaks, {top200['celltype'].nunique()} celltypes")

all_celltypes = sorted(top200["celltype"].unique())

# %% Distinct colors for TF tracks
TF_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]

# %% Generate motif maps
print(f"\nGenerating motif maps for {len(all_celltypes)} celltypes × top {TOP_N_PEAKS} peaks ...\n",
      flush=True)

for ct in all_celltypes:
    # Get top enriched TFs for this celltype (FDR < 0.05)
    ct_enrich = enrichment[(enrichment["celltype"] == ct) & (enrichment["fdr"] < 0.05)]
    ct_enrich = ct_enrich.sort_values("enrichment_zscore", ascending=False)
    enriched_tfs = ct_enrich["tf"].tolist()

    if len(enriched_tfs) == 0:
        print(f"  {ct}: no significant TFs (FDR < 0.05) — skipping")
        continue

    # Get top N peaks for this celltype
    ct_peaks = top200[top200["celltype"] == ct].nsmallest(TOP_N_PEAKS, "rank")

    ct_fig_dir = f"{FIG_DIR}/{ct}"
    os.makedirs(ct_fig_dir, exist_ok=True)

    for _, peak_row in ct_peaks.iterrows():
        pid = peak_row["peak_id"]
        rank = int(peak_row["rank"])
        zscore = peak_row["V3_zscore"]
        chrom = str(peak_row["chrom"])
        start = int(peak_row["start"])
        end = int(peak_row["end"])
        peak_len = end - start

        # Gene label
        linked = str(peak_row.get("linked_gene", ""))
        nearest = str(peak_row.get("nearest_gene", ""))
        if linked in ("", "nan", "None"):
            linked = ""
        if nearest in ("", "nan", "None"):
            nearest = ""
        gene = linked if linked else nearest if nearest else f"chr{chrom}:{start}"

        # Get motif hits for this peak, filtered to enriched TFs
        peak_hits = positions[positions["peak_id"] == pid]
        peak_hits = peak_hits[peak_hits["tf"].isin(enriched_tfs)]

        if len(peak_hits) == 0:
            continue

        # Group by TF, order by enrichment rank
        tf_hits = {}
        for tf in enriched_tfs:
            tf_rows = peak_hits[peak_hits["tf"] == tf]
            if len(tf_rows) > 0:
                tf_hits[tf] = tf_rows
            if len(tf_hits) >= TOP_N_TFS:
                break

        if len(tf_hits) == 0:
            continue

        n_tfs_found = len(tf_hits)
        total_hits = sum(len(v) for v in tf_hits.values())

        # Assign colors
        tf_order = list(tf_hits.keys())
        tf_color_map = {tf: TF_COLORS[i % len(TF_COLORS)] for i, tf in enumerate(tf_order)}

        # Plot
        n_tracks = len(tf_order)
        fig_height = max(2.5, 1.0 + n_tracks * 0.4)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Peak bar at top
        ax.barh(n_tracks, peak_len, left=0, height=0.6,
                color="#e0e0e0", edgecolor="#999999", linewidth=0.5)
        ax.text(peak_len / 2, n_tracks, f"{peak_len} bp",
                ha="center", va="center", fontsize=7, color="#555555")

        # Motif hits on tracks
        for track_i, tf in enumerate(tf_order):
            color = tf_color_map[tf]
            for _, h in tf_hits[tf].iterrows():
                width = h["hit_end"] - h["hit_start"]
                ax.barh(track_i, width, left=h["hit_start"],
                        height=0.6, color=color, edgecolor="black",
                        linewidth=0.3, alpha=0.85)

        ax.set_yticks(list(range(n_tracks)) + [n_tracks])
        ax.set_yticklabels(tf_order + ["Peak"], fontsize=8)
        ax.set_xlabel("Position within peak (bp)", fontsize=9)
        ax.set_xlim(-5, peak_len + 5)
        ax.set_ylim(-0.5, n_tracks + 1)
        ax.set_title(f"{ct.replace('_', ' ')} — rank {rank}: {gene}\n"
                     f"chr{chrom}:{start:,}-{end:,}  |  V3 z={zscore:.1f}  |  "
                     f"{n_tfs_found} enriched TFs, {total_hits} hits",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.grid(axis="y", visible=False)

        fig.tight_layout()
        fname = f"{ct}_rank{rank}_{gene}".replace("/", "_").replace(":", "_")
        fig.savefig(f"{ct_fig_dir}/{fname}_motif_map.pdf", bbox_inches="tight")
        fig.savefig(f"{ct_fig_dir}/{fname}_motif_map.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    n_generated = len([f for f in os.listdir(ct_fig_dir) if f.endswith(".png")])
    print(f"  {ct}: {n_generated} maps (from {len(enriched_tfs)} enriched TFs)")

print(f"\nDone. Figures: {FIG_DIR}/")
print(f"End: {time.strftime('%c')}")
