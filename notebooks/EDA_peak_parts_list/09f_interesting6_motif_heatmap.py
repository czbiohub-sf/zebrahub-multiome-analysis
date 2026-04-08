# %% Script 09f: Motif enrichment heatmap for 6 high-interest celltypes
#
# Runs FIMO on top-50 peaks for ALL 31 celltypes, then computes Fisher's exact
# test with a COMMON BACKGROUND (all other celltypes pooled). This ensures
# consistent statistical power across celltypes.
#
# Generates focused heatmaps for the 6 interesting celltypes:
#   Panel A: enrichment z-score (top 5 TFs per celltype)
#   Panel B: raw hit rate
#   Panel C: log2 fold enrichment
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu  (needs pysam, pymemesuite)

import os, re, time, warnings
import numpy as np
import pandas as pd
import anndata as ad
import pysam
from pymemesuite.common import (MotifFile, Sequence as MemeSequence,
                                 Background, Array as MemeArray)
from pymemesuite.fimo import FIMO as FIMOScanner
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

warnings.filterwarnings("ignore", category=FutureWarning)
print("=== Script 09f: Interesting-6 Motif Enrichment (all-31 background) ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
FIG_V3  = f"{REPO}/figures/peak_parts_list/V3/motif_enrichment"
os.makedirs(FIG_V3, exist_ok=True)

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
FASTA_PATH  = "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa"
MEME_PATH   = ("/hpc/projects/data.science/yangjoon.kim/github_repos/"
               "gReLU/src/grelu/resources/meme/H12CORE_meme_format.meme")

INTERESTING_6 = [
    "epidermis", "neural_crest", "hemangioblasts",
    "hindbrain", "optic_cup", "hatching_gland",
]
TOP_N = 50
PVAL_THRESH = 1e-4
TOP_TF_N = 5   # top TFs per celltype for the focused heatmap

# %% Color palette
import sys as _sys
_sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

# %% Load data
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
obs = adata.obs.copy()
print(f"  {adata.shape}  ({time.time()-t0:.1f}s)")

print("Loading V3 z-score matrix ...", flush=True)
z_adata = ad.read_h5ad(f"{OUTDIR}/V3_specificity_matrix_celltype_level.h5ad")
Z_ct = np.array(z_adata.X)
ct_names = list(z_adata.var_names)
print(f"  {len(ct_names)} celltypes")

# %% Load JASPAR motifs
print("Loading JASPAR motifs ...", flush=True)
motif_file = MotifFile(MEME_PATH)
motif_list = list(motif_file)
# Name is in .accession (pymemesuite: .name is empty for H12CORE)
motif_names = [m.accession.decode() if isinstance(m.accession, bytes) else str(m.accession)
               for m in motif_list]
motif_tf_names = [n.split(".")[0] for n in motif_names]
n_motifs = len(motif_list)
print(f"  {n_motifs} motifs loaded")
print(f"  Example: {motif_names[0]} -> TF: {motif_tf_names[0]}")

_bg = Background(motif_list[0].alphabet, MemeArray([0.25, 0.25, 0.25, 0.25]))

# Deduplicate TF names
unique_tfs = sorted(set(motif_tf_names))
tf_to_motif_indices = {}
for tf in unique_tfs:
    tf_to_motif_indices[tf] = [j for j, n in enumerate(motif_tf_names) if n == tf]
print(f"  {len(unique_tfs)} unique TFs after deduplication")

# %% Extract top-50 peaks and sequences for ALL 31 celltypes
print(f"\nExtracting sequences for all {len(ct_names)} celltypes ...", flush=True)
fa = pysam.FastaFile(FASTA_PATH)

celltype_seqs = {}    # ct → list of sequences
celltype_peak_idx = {}  # ct → array of peak indices

for ct in ct_names:
    ct_col = ct_names.index(ct)
    z_col = Z_ct[:, ct_col]
    top_idx = np.argsort(z_col)[::-1][:TOP_N]

    seqs = []
    for idx in top_idx:
        chrom = str(obs.iloc[idx].get("chrom", ""))
        start = int(obs.iloc[idx].get("start", 0))
        end   = int(obs.iloc[idx].get("end", 0))
        try:
            seq = fa.fetch(f"chr{chrom}", start, end).upper()
        except (KeyError, ValueError):
            try:
                seq = fa.fetch(str(chrom), start, end).upper()
            except:
                seq = None
        if seq and len(seq) >= 10 and set(seq) <= set("ACGT"):
            seqs.append(seq)

    celltype_seqs[ct] = seqs
    celltype_peak_idx[ct] = top_idx
    print(f"  {ct}: {len(seqs)}/{TOP_N} valid sequences")

fa.close()

# %% Run FIMO for ALL 31 celltypes
print(f"\nRunning FIMO (p < {PVAL_THRESH:.0e}) for all {len(ct_names)} celltypes ...",
      flush=True)
fimo_scanner = FIMOScanner(both_strands=True, threshold=PVAL_THRESH)

hit_binary = {}  # ct → (n_peaks, n_motifs) bool array

for ct in ct_names:
    seqs = celltype_seqs[ct]
    if not seqs:
        hit_binary[ct] = np.zeros((0, n_motifs), dtype=bool)
        continue

    print(f"  {ct}: {len(seqs)} seqs x {n_motifs} motifs ...", end=" ", flush=True)
    t0 = time.time()

    meme_seqs = [MemeSequence(s, f"seq_{i}".encode()) for i, s in enumerate(seqs)]
    peak_motif_hits = np.zeros((len(seqs), n_motifs), dtype=bool)

    for j, motif in enumerate(motif_list):
        pattern = fimo_scanner.score_motif(motif, meme_seqs, _bg)
        if pattern is not None:
            for me in pattern.matched_elements:
                seq_name = me.source.name.decode() if hasattr(me.source, 'name') else ""
                try:
                    seq_idx = int(seq_name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                peak_motif_hits[seq_idx, j] = True

    hit_binary[ct] = peak_motif_hits
    n_active = (peak_motif_hits.any(axis=0)).sum()
    print(f"{time.time()-t0:.1f}s  (active motifs: {n_active})")

# %% Fisher's exact test: each celltype vs ALL OTHER 30 celltypes pooled
print(f"\nRunning Fisher's exact tests (common background: all other celltypes) ...",
      flush=True)
t0 = time.time()

ct_with_data = [ct for ct in ct_names if hit_binary[ct].shape[0] > 0]
fisher_results = []

for ct in ct_with_data:
    n_fg = hit_binary[ct].shape[0]

    # Background: pool ALL other celltypes (not just the 6 interesting ones)
    bg_arrays = [hit_binary[other] for other in ct_with_data if other != ct]
    bg_hits = np.vstack(bg_arrays)
    n_bg = bg_hits.shape[0]

    for tf in unique_tfs:
        mot_idx = tf_to_motif_indices[tf]

        fg_any_hit = hit_binary[ct][:, mot_idx].any(axis=1)
        a = fg_any_hit.sum()
        b = n_fg - a

        bg_any_hit = bg_hits[:, mot_idx].any(axis=1)
        c = bg_any_hit.sum()
        d = n_bg - c

        if a + c == 0:
            continue

        odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative="greater")

        fisher_results.append({
            "celltype": ct, "tf": tf,
            "hits_fg": int(a), "total_fg": int(n_fg),
            "hit_rate_fg": a / n_fg,
            "hits_bg": int(c), "total_bg": int(n_bg),
            "hit_rate_bg": c / n_bg if n_bg > 0 else 0,
            "odds_ratio": odds_ratio, "pvalue": pval,
        })

fisher_df = pd.DataFrame(fisher_results)
print(f"  Fisher's tests done ({time.time()-t0:.1f}s), {len(fisher_df)} rows")

# FDR per celltype
fisher_df["fdr"] = np.nan
for ct in ct_with_data:
    mask = fisher_df["celltype"] == ct
    if mask.sum() == 0:
        continue
    _, fdr_vals, _, _ = multipletests(fisher_df.loc[mask, "pvalue"].values, method="fdr_bh")
    fisher_df.loc[mask, "fdr"] = fdr_vals

# Enrichment z-score: across ALL 31 celltypes (robust denominator)
fisher_df["enrichment_zscore"] = np.nan
for tf in fisher_df["tf"].unique():
    mask = fisher_df["tf"] == tf
    rates = fisher_df.loc[mask, "hit_rate_fg"].values
    mean_r = rates.mean()
    std_r  = rates.std()
    fisher_df.loc[mask, "enrichment_zscore"] = (rates - mean_r) / (std_r + 0.02)

# Log2 fold enrichment (vs background hit rate)
fisher_df["log2_fold"] = np.log2(
    (fisher_df["hit_rate_fg"] + 0.01) / (fisher_df["hit_rate_bg"] + 0.01)
)

# Save full results
out_csv = f"{OUTDIR}/V3_all31_motif_enrichment.csv"
fisher_df.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}  ({len(fisher_df)} rows)")

# Also save interesting-6 subset
i6_df = fisher_df[fisher_df["celltype"].isin(INTERESTING_6)].copy()
i6_csv = f"{OUTDIR}/V3_interesting6_motif_enrichment.csv"
i6_df.to_csv(i6_csv, index=False)
print(f"  Saved: {i6_csv}  ({len(i6_df)} rows)")

# Print top enriched TFs for the 6
print("\nTop enriched TFs (interesting 6):")
for ct in INTERESTING_6:
    ct_df = i6_df[i6_df["celltype"] == ct].sort_values("enrichment_zscore", ascending=False)
    top5 = ct_df.head(5)
    tfs = ", ".join(f"{r['tf']}(z={r['enrichment_zscore']:.1f},rate={r['hit_rate_fg']:.0%})"
                    for _, r in top5.iterrows())
    n_sig = (ct_df["fdr"] < 0.05).sum()
    print(f"  {ct}: {tfs}  (FDR<0.05: {n_sig})")

# ══════════════════════════════════════════════════════════════════════════════
# HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

print("\n--- Generating heatmaps ---", flush=True)

# Select top 5 enriched TFs per interesting celltype (union)
selected_tfs = []
for ct in INTERESTING_6:
    ct_df = i6_df[i6_df["celltype"] == ct].sort_values("enrichment_zscore", ascending=False)
    selected_tfs.extend(ct_df.head(TOP_TF_N)["tf"].tolist())

# Deduplicate preserving order
seen = set()
unique_selected = []
for tf in selected_tfs:
    if tf not in seen:
        unique_selected.append(tf)
        seen.add(tf)

print(f"  Union of top-{TOP_TF_N} TFs per celltype: {len(unique_selected)} unique TFs")

# Build matrices for heatmap
n_ct = len(INTERESTING_6)
n_tf = len(unique_selected)
ct_labels = [ct.replace("_", " ") for ct in INTERESTING_6]

zscore_mat  = np.zeros((n_ct, n_tf))
hitrate_mat = np.zeros((n_ct, n_tf))
log2f_mat   = np.zeros((n_ct, n_tf))
sig_mat     = np.full((n_ct, n_tf), "", dtype=object)

for i, ct in enumerate(INTERESTING_6):
    ct_df = i6_df[i6_df["celltype"] == ct]
    for j, tf in enumerate(unique_selected):
        tf_row = ct_df[ct_df["tf"] == tf]
        if len(tf_row) > 0:
            r = tf_row.iloc[0]
            zscore_mat[i, j]  = r["enrichment_zscore"]
            hitrate_mat[i, j] = r["hit_rate_fg"]
            log2f_mat[i, j]   = r["log2_fold"]
            if r["fdr"] < 0.01:
                sig_mat[i, j] = "**"
            elif r["fdr"] < 0.05:
                sig_mat[i, j] = "*"

zscore_df  = pd.DataFrame(zscore_mat,  index=ct_labels, columns=unique_selected)
hitrate_df = pd.DataFrame(hitrate_mat, index=ct_labels, columns=unique_selected)
log2f_df   = pd.DataFrame(log2f_mat,   index=ct_labels, columns=unique_selected)


# ── Panel A+B: z-score + hit rate side by side ──
print("  Generating A+B panel (z-score + hit rate) ...", flush=True)

fig, (ax_z, ax_h) = plt.subplots(1, 2, figsize=(max(10, n_tf * 0.5 + 4), 5),
                                  gridspec_kw={"wspace": 0.4})

vmax_z = min(max(abs(zscore_mat.min()), abs(zscore_mat.max()), 2.0), 5.0)

sns.heatmap(zscore_df, ax=ax_z, cmap="RdBu_r", center=0, vmin=-vmax_z, vmax=vmax_z,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Enrichment z-score", "shrink": 0.6})
for i in range(n_ct):
    for j in range(n_tf):
        if sig_mat[i, j]:
            ax_z.text(j + 0.5, i + 0.5, sig_mat[i, j],
                      ha="center", va="center", fontsize=7, color="black", fontweight="bold")
ax_z.set_title("A. Enrichment z-score\n(across 31 celltypes)", fontsize=10)
ax_z.set_xlabel("Transcription factor")
ax_z.set_ylabel("Cell type")
ax_z.set_xticklabels(ax_z.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax_z.set_yticklabels(ax_z.get_yticklabels(), fontsize=9)

sns.heatmap(hitrate_df, ax=ax_h, cmap="YlOrRd", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white", annot=True, fmt=".0%", annot_kws={"fontsize": 6},
            cbar_kws={"label": "Hit rate (fraction of top-50 peaks)", "shrink": 0.6})
ax_h.set_title("B. Raw hit rate\n(fraction of peaks with motif)", fontsize=10)
ax_h.set_xlabel("Transcription factor")
ax_h.set_ylabel("")
ax_h.set_xticklabels(ax_h.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax_h.set_yticklabels(ax_h.get_yticklabels(), fontsize=9)

fig.suptitle(f"TF motif enrichment: top-{TOP_TF_N} per celltype\n"
             f"(Fisher's exact vs. all-31-celltype background, * FDR<0.05, ** FDR<0.01)",
             fontsize=10, y=1.03)
fig.tight_layout()
fig.savefig(f"{FIG_V3}/V3_interesting6_motif_zscore_hitrate.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_V3}/V3_interesting6_motif_zscore_hitrate.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: V3_interesting6_motif_zscore_hitrate.{pdf,png}")


# ── Panel C: log2 fold enrichment ──
print("  Generating log2 fold enrichment panel ...", flush=True)

fig, ax = plt.subplots(figsize=(max(8, n_tf * 0.45 + 2), 5))
vmax_l2 = min(max(abs(log2f_mat.min()), abs(log2f_mat.max()), 2.0), 5.0)

sns.heatmap(log2f_df, ax=ax, cmap="PiYG", center=0, vmin=-vmax_l2, vmax=vmax_l2,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "log2(fold enrichment vs background)", "shrink": 0.6})
for i in range(n_ct):
    for j in range(n_tf):
        if sig_mat[i, j]:
            ax.text(j + 0.5, i + 0.5, sig_mat[i, j],
                    ha="center", va="center", fontsize=7, color="black", fontweight="bold")
ax.set_title(f"TF motif enrichment: log2 fold (top-{TOP_TF_N} per celltype)\n"
             f"(vs. all-31-celltype background, * FDR<0.05, ** FDR<0.01)", fontsize=10)
ax.set_xlabel("Transcription factor")
ax.set_ylabel("Cell type")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG_V3}/V3_interesting6_motif_log2fold.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_V3}/V3_interesting6_motif_log2fold.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: V3_interesting6_motif_log2fold.{pdf,png}")


# ── Single z-score heatmap with celltype color bar (for manuscript) ──
print("  Generating z-score heatmap with color bar ...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(max(9, n_tf * 0.45 + 3), 4.5),
                         gridspec_kw={"width_ratios": [0.03, 1], "wspace": 0.02})

ax_color = axes[0]
for i, ct in enumerate(INTERESTING_6):
    color = cell_type_color_dict.get(ct, "#888888")
    ax_color.add_patch(plt.Rectangle((0, i), 1, 1, facecolor=color, edgecolor="white", lw=0.5))
ax_color.set_xlim(0, 1)
ax_color.set_ylim(0, n_ct)
ax_color.invert_yaxis()
ax_color.set_xticks([])
ax_color.set_yticks([])

ax_hm = axes[1]
sns.heatmap(zscore_df, ax=ax_hm, cmap="RdBu_r", center=0, vmin=-vmax_z, vmax=vmax_z,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Enrichment z-score", "shrink": 0.7})
for i in range(n_ct):
    for j in range(n_tf):
        if sig_mat[i, j]:
            ax_hm.text(j + 0.5, i + 0.5, sig_mat[i, j],
                       ha="center", va="center", fontsize=7, color="black", fontweight="bold")
ax_hm.set_title(f"TF motif enrichment (top-{TOP_TF_N} per celltype, all-31 background)\n"
                f"* FDR<0.05, ** FDR<0.01", fontsize=10)
ax_hm.set_xlabel("Transcription factor")
ax_hm.set_ylabel("")
plt.sca(ax_hm)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG_V3}/V3_interesting6_motif_heatmap_colorbar.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_V3}/V3_interesting6_motif_heatmap_colorbar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: V3_interesting6_motif_heatmap_colorbar.{pdf,png}")

print(f"\nDone. End: {time.strftime('%c')}")
