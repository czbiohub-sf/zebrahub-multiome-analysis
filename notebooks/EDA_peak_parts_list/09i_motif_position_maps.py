# %% Script 09i: Motif position maps for top peaks
#
# For the 6 interesting celltypes, takes top 5 peaks each, runs FIMO to find
# WHERE in the DNA sequence the enriched TF motifs are, then generates
# position maps showing motif locations within each peak.
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu (needs pysam, pymemesuite)

import os, time
import numpy as np
import pandas as pd
import anndata as ad
import pysam
from pymemesuite.common import (MotifFile, Sequence as MemeSequence,
                                 Background, Array as MemeArray)
from pymemesuite.fimo import FIMO as FIMOScanner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

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

print("=== Script 09i: Motif Position Maps ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
FIG_DIR = f"{REPO}/figures/peak_parts_list/V3/motif_position_maps"
os.makedirs(FIG_DIR, exist_ok=True)

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
FASTA_PATH  = "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa"
MEME_PATH   = ("/hpc/projects/data.science/yangjoon.kim/github_repos/"
               "gReLU/src/grelu/resources/meme/H12CORE_meme_format.meme")
V3_ZMAT     = f"{OUTDIR}/V3_specificity_matrix_celltype_level.h5ad"
ENRICH_CSV  = f"{OUTDIR}/V3_all31_motif_enrichment.csv"

INTERESTING_6 = [
    "epidermis", "neural_crest", "hemangioblasts",
    "hindbrain", "optic_cup", "hatching_gland",
]
TOP_N_PEAKS = 5
TOP_N_TFS   = 10   # top enriched TFs per celltype to scan for
PVAL_THRESH = 1e-4

# %% Load data
print("\nLoading data ...", flush=True)
t0 = time.time()
master = ad.read_h5ad(MASTER_H5AD)
obs = master.obs.copy()
obs["linked_gene_str"]  = obs["linked_gene"].astype(str)
obs["nearest_gene_str"] = obs["nearest_gene"].astype(str)
print(f"  Master: {master.shape}  ({time.time()-t0:.1f}s)")

z_adata = ad.read_h5ad(V3_ZMAT)
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)

enrich_df = pd.read_csv(ENRICH_CSV)
print(f"  Enrichment: {len(enrich_df)} rows")

# %% Load JASPAR motifs
print("Loading JASPAR motifs ...", flush=True)
motif_file = MotifFile(MEME_PATH)
motif_list = list(motif_file)
# Name is in .accession
motif_names = [m.accession.decode() if isinstance(m.accession, bytes) else str(m.accession)
               for m in motif_list]
motif_tf_names = [n.split(".")[0] for n in motif_names]
n_motifs = len(motif_list)
print(f"  {n_motifs} motifs loaded")

_bg = Background(motif_list[0].alphabet, MemeArray([0.25, 0.25, 0.25, 0.25]))

# Build TF name → motif indices mapping
tf_to_motif_idx = {}
for j, tf in enumerate(motif_tf_names):
    tf_to_motif_idx.setdefault(tf, []).append(j)

# %% Distinct colors for TF motifs
TF_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]

# %% Process each celltype
print(f"\nProcessing {len(INTERESTING_6)} celltypes × top {TOP_N_PEAKS} peaks ...\n",
      flush=True)

fa = pysam.FastaFile(FASTA_PATH)
fimo_scanner = FIMOScanner(both_strands=True, threshold=PVAL_THRESH)

all_peak_motif_data = []  # for CSV export

for ct in INTERESTING_6:
    ct_col = ct_names.index(ct)
    z_col = Z_ct[:, ct_col]
    top_idx = np.argsort(z_col)[::-1][:TOP_N_PEAKS]

    # Get top enriched TFs for this celltype
    ct_enrich = enrich_df[enrich_df["celltype"] == ct].sort_values(
        "enrichment_zscore", ascending=False)
    top_tfs = ct_enrich.head(TOP_N_TFS)["tf"].tolist()
    # Get all motif indices for these TFs
    scan_motif_indices = []
    for tf in top_tfs:
        scan_motif_indices.extend(tf_to_motif_idx.get(tf, []))

    print(f"=== {ct} === (scanning {len(scan_motif_indices)} motifs for {len(top_tfs)} TFs)")

    ct_fig_dir = f"{FIG_DIR}/{ct}"
    os.makedirs(ct_fig_dir, exist_ok=True)

    for rank, peak_iloc in enumerate(top_idx, 1):
        peak_obs = obs.iloc[peak_iloc]
        chrom = str(peak_obs["chrom"])
        start = int(peak_obs["start"])
        end   = int(peak_obs["end"])
        peak_len = end - start

        # Gene label
        linked  = str(peak_obs.get("linked_gene", ""))
        nearest = str(peak_obs.get("nearest_gene", ""))
        if linked in ("", "nan", "None"):
            linked = ""
        if nearest in ("", "nan", "None"):
            nearest = ""
        gene = linked if linked else nearest if nearest else f"chr{chrom}:{start}"

        zscore = z_col[peak_iloc]

        # Extract sequence
        try:
            seq = fa.fetch(f"chr{chrom}", start, end).upper()
        except:
            print(f"  rank{rank}: could not fetch chr{chrom}:{start}-{end}")
            continue

        if not seq or len(seq) < 10:
            continue

        # Run FIMO for the top TF motifs only
        meme_seq = MemeSequence(seq, b"peak")

        motif_hits = []  # (tf_name, motif_start, motif_end, strand, score)
        for j in scan_motif_indices:
            tf = motif_tf_names[j]
            pattern = fimo_scanner.score_motif(motif_list[j], [meme_seq], _bg)
            if pattern is not None:
                for me in pattern.matched_elements:
                    hit_start = me.start
                    hit_end   = me.stop
                    strand    = "+" if me.strand == 1 else "-"
                    score     = me.score if hasattr(me, 'score') else 0
                    motif_hits.append({
                        "tf": tf,
                        "hit_start": hit_start,
                        "hit_end": hit_end,
                        "strand": strand,
                    })

        # Deduplicate by TF (keep unique TF hits, merge overlapping positions)
        tf_hits = {}
        for h in motif_hits:
            tf = h["tf"]
            if tf not in tf_hits:
                tf_hits[tf] = []
            tf_hits[tf].append(h)

        # Only keep TFs from top_tfs list (already filtered, but just in case)
        tf_hits = {tf: hits for tf, hits in tf_hits.items() if tf in top_tfs}

        n_tfs_found = len(tf_hits)
        total_hits = sum(len(v) for v in tf_hits.values())

        print(f"  rank{rank}: {gene} chr{chrom}:{start}-{end} ({peak_len}bp) "
              f"z={zscore:.1f}  {n_tfs_found} TFs, {total_hits} hits")

        # Save to CSV data
        for tf, hits in tf_hits.items():
            for h in hits:
                all_peak_motif_data.append({
                    "celltype": ct,
                    "rank": rank,
                    "gene": gene,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "peak_length": peak_len,
                    "V3_zscore": zscore,
                    "tf": tf,
                    "hit_start": h["hit_start"],
                    "hit_end": h["hit_end"],
                    "strand": h["strand"],
                })

        # ── Generate motif position map ──
        if n_tfs_found == 0:
            continue

        # Assign colors to TFs (ordered by enrichment rank)
        tf_order = [tf for tf in top_tfs if tf in tf_hits]
        tf_color_map = {tf: TF_COLORS[i % len(TF_COLORS)]
                        for i, tf in enumerate(tf_order)}

        n_tracks = len(tf_order)
        fig_height = max(2.5, 1.0 + n_tracks * 0.4)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Draw the peak as a grey bar at the top
        ax.barh(n_tracks, peak_len, left=0, height=0.6,
                color="#e0e0e0", edgecolor="#999999", linewidth=0.5)
        ax.text(peak_len / 2, n_tracks, f"{peak_len} bp",
                ha="center", va="center", fontsize=7, color="#555555")

        # Draw motif hits as colored blocks on separate tracks
        for track_i, tf in enumerate(tf_order):
            color = tf_color_map[tf]
            hits = tf_hits[tf]
            for h in hits:
                width = h["hit_end"] - h["hit_start"]
                ax.barh(track_i, width, left=h["hit_start"],
                        height=0.6, color=color, edgecolor="black",
                        linewidth=0.3, alpha=0.85)
                # Strand arrow
                if h["strand"] == "+":
                    ax.annotate("", xy=(h["hit_end"], track_i),
                                xytext=(h["hit_start"], track_i),
                                arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

        # Y-axis labels = TF names
        ax.set_yticks(list(range(n_tracks)) + [n_tracks])
        ax.set_yticklabels(tf_order + ["Peak"], fontsize=8)
        ax.set_xlabel("Position within peak (bp)", fontsize=9)
        ax.set_xlim(-5, peak_len + 5)
        ax.set_ylim(-0.5, n_tracks + 1)
        ax.set_title(f"{ct.replace('_', ' ')} — rank {rank}: {gene}\n"
                     f"chr{chrom}:{start:,}-{end:,}  |  V3 z={zscore:.1f}  |  "
                     f"{n_tfs_found} TFs, {total_hits} motif hits",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.grid(axis="y", visible=False)

        fig.tight_layout()
        fname = f"{ct}_rank{rank}_{gene}".replace("/", "_").replace(":", "_")
        fig.savefig(f"{ct_fig_dir}/{fname}_motif_map.pdf", bbox_inches="tight")
        fig.savefig(f"{ct_fig_dir}/{fname}_motif_map.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

fa.close()

# %% Save motif positions CSV
motif_csv = f"{OUTDIR}/V3_interesting6_top5_motif_positions.csv"
pd.DataFrame(all_peak_motif_data).to_csv(motif_csv, index=False)
print(f"\nSaved: {motif_csv} ({len(all_peak_motif_data)} rows)")

print(f"\nDone. Figures: {FIG_DIR}/")
print(f"End: {time.strftime('%c')}")
