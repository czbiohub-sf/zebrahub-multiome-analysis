# %% [markdown]
# # Step 08: Cross-Celltype JASPAR Motif Enrichment
#
# For each focal cell type, take the top 50 most-specific peaks
# (by V2 leave-one-out z-score at best reliable timepoint) and scan their
# ATAC-seq sequences against the JASPAR2022 CORE vertebrates motif DB
# (H12CORE, 1443 PWMs) using tangermeme's FIMO implementation.
#
# For each (celltype, motif) pair compute the **motif hit rate**
# = fraction of the celltype's top peaks containing ≥1 FIMO hit at p<1e-4.
#
# Produces:
#   figures/peak_parts_list/motif_enrichment_celltypes_heatmap.pdf/.png
#   figures/peak_parts_list/motif_enrichment_celltypes_barplots.pdf/.png
#   outputs/motif_hit_rate_per_celltype.csv     (celltypes × TFs)
#   outputs/motif_enrichment_top_tfs.csv        (top 15 TFs per celltype)
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu  (has grelu, tangermeme, pysam, anndata)

# %% Imports
import os, re, gc, time
import numpy as np
import pandas as pd
import anndata as ad
import pysam
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

print(f"anndata {ad.__version__}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs"
FIG_DIR = f"{REPO}/figures/peak_parts_list"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

FASTA_PATH = "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa"
MEME_PATH  = ("/hpc/projects/data.science/yangjoon.kim/github_repos/"
              "gReLU/src/grelu/resources/meme/H12CORE_meme_format.meme")

SPEC_V2   = f"{OUTDIR}/specificity_matrix_v2.h5ad"
META_H5AD = (f"{BASE}/data/annotated_data/objects_v2/"
             "peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")

# %% Configuration
FOCAL_CELLTYPES = [
    "fast_muscle",
    "heart_myocardium",
    "neural_crest",
    "PSM",
    "notochord",
    "epidermis",
    "hemangioblasts",
]

TOP_N_PEAKS = 50    # top peaks per celltype (by z-score at best timepoint)
PVAL_THRESH = 1e-4  # FIMO p-value threshold for motif hit
MIN_ZSCORE  = 2.0   # minimum z-score to include a peak in the foreground

# TF family annotation (for colour-coding in figure)
TF_FAMILY_MAP = {
    "GATA":         ["GATA"],
    "TBX":          ["TBX", "BRACHYURY"],
    "NKX":          ["NKX"],
    "MEF2":         ["MEF2"],
    "MyoD/bHLH":    ["MYOD", "MYF", "MYOG", "ASCL"],
    "SOX":          ["SOX"],
    "TFAP2":        ["TFAP2"],
    "PAX":          ["PAX"],
    "FOX":          ["FOXA", "FOXD", "FOXC", "FOXF", "FOXH", "FOXO"],
    "ETS":          ["ETV", "ERG", "FLI", "GABPA", "ELK", "ETS1", "ETS2"],
    "TEAD":         ["TEAD"],
    "HAND":         ["HAND"],
    "RUNX":         ["RUNX"],
    "KLF/SP":       ["KLF", "SP1", "SP2", "SP3", "SP4"],
    "HOX":          ["HOX"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Load motif database (once)
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading JASPAR H12CORE motifs from {MEME_PATH.split('/')[-1]} ...", flush=True)
from tangermeme.tools.fimo import read_meme, fimo as tangermeme_fimo
t0 = time.time()
motifs_db  = read_meme(MEME_PATH)
motif_names = list(motifs_db.keys())   # e.g. "GATA4.H12CORE.0.P.B"
# Clean name: "GATA4.H12CORE.0.P.B" → "GATA4"
motif_tf_names = [n.split(".")[0] for n in motif_names]
print(f"  {len(motif_names)} motifs loaded  ({time.time()-t0:.1f}s)", flush=True)
print(f"  Example: {motif_names[0]} → TF: {motif_tf_names[0]}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: one-hot encode a list of DNA sequences (pad to max length)
# ──────────────────────────────────────────────────────────────────────────────
BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def one_hot_encode(seqs):
    """Return float32 tensor (N, 4, L) with N-padded sequences."""
    L = max(len(s) for s in seqs)
    arr = torch.zeros(len(seqs), 4, L, dtype=torch.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            idx = BASE2IDX.get(c, None)
            if idx is not None:
                arr[i, idx, j] = 1.0
    return arr

# ──────────────────────────────────────────────────────────────────────────────
# Load specificity matrix (var = conditions, obs = peaks)
# ──────────────────────────────────────────────────────────────────────────────
print("\nLoading V2 specificity matrix ...", flush=True)
t0 = time.time()
zad = ad.read_h5ad(SPEC_V2)
print(f"  Shape: {zad.shape}  ({time.time()-t0:.1f}s)", flush=True)

# Parse condition columns → celltype, timepoint
def parse_condition(cond):
    m = re.search(r"(\d+somites)$", cond)
    if not m:
        return cond, ""
    tp = m.group(1)
    ct = cond[:-(len(tp)+1)]
    return ct, tp

cond_meta = pd.DataFrame(
    [parse_condition(c) for c in zad.var_names],
    columns=["celltype", "timepoint"],
    index=zad.var_names,
)
tp_int = {"0somites":0,"5somites":5,"10somites":10,
          "15somites":15,"20somites":20,"30somites":30}
cond_meta["tp_int"] = cond_meta["timepoint"].map(tp_int)

# n_cells per condition
n_cells_col = [c for c in zad.var.columns if "n_cells" in c.lower()]
if n_cells_col:
    cond_meta["n_cells"] = zad.var[n_cells_col[0]].values
    cond_meta["reliable"] = cond_meta["n_cells"] >= 20
else:
    cond_meta["n_cells"] = 999
    cond_meta["reliable"] = True

# ──────────────────────────────────────────────────────────────────────────────
# Load peak metadata (coordinates + gene annotations)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading peak metadata ...", flush=True)
t0 = time.time()
meta = ad.read_h5ad(META_H5AD, backed="r")
obs  = meta.obs.copy()
meta.file.close()
# obs has 'chrom' (Categorical, no chr prefix), 'start', 'end' (int64)
obs["chrom_str"] = "chr" + obs["chrom"].astype(str)
print(f"  obs shape: {obs.shape}  ({time.time()-t0:.1f}s)", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# For each focal celltype: select top N peaks at best reliable timepoint
# ──────────────────────────────────────────────────────────────────────────────
print("\nSelecting top peaks and extracting sequences ...", flush=True)
fa = pysam.FastaFile(FASTA_PATH)

celltype_seqs    = {}   # ct → list of DNA sequences
celltype_n_peaks = {}   # ct → actual count of peaks used

for ct in FOCAL_CELLTYPES:
    ct_conds = cond_meta[cond_meta["celltype"] == ct]
    if ct_conds.empty:
        print(f"  {ct}: no conditions found — skipping", flush=True)
        continue

    rel_conds = ct_conds[ct_conds["reliable"]]
    if rel_conds.empty:
        rel_conds = ct_conds
        print(f"  WARNING: {ct} — all timepoints unreliable, using all", flush=True)

    # Best timepoint = most n_cells among reliable
    best_cond = rel_conds.sort_values("n_cells", ascending=False).index[0]
    col_idx   = list(zad.var_names).index(best_cond)
    z_vec     = np.asarray(zad.X[:, col_idx]).ravel()

    # Top N peaks with z ≥ MIN_ZSCORE
    n_pass  = (z_vec >= MIN_ZSCORE).sum()
    top_idx = np.argsort(z_vec)[::-1]
    top_idx = top_idx[z_vec[top_idx] >= MIN_ZSCORE][:TOP_N_PEAKS]

    # Extract sequences using pysam
    seqs = []
    for i in top_idx:
        row    = obs.iloc[i]
        chrom  = str(row["chrom_str"])
        start  = int(row["start"])
        end    = int(row["end"])
        try:
            seq = fa.fetch(chrom, start, end).upper()
            if len(seq) > 0 and seq.count("N") / len(seq) < 0.5:
                seqs.append(seq)
        except Exception:
            pass

    celltype_seqs[ct]    = seqs
    celltype_n_peaks[ct] = len(seqs)
    print(f"  {ct}: {len(seqs)} seqs from {best_cond}  (z≥{MIN_ZSCORE}: {n_pass:,})", flush=True)

fa.close()
del zad
gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# Run FIMO for each celltype and compute per-motif hit rate
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nRunning FIMO (tangermeme, p<{PVAL_THRESH:.0e}) ...", flush=True)

n_motifs   = len(motif_names)
ct_list    = list(celltype_seqs.keys())
# hit_rate[ct][j] = fraction of ct's peaks with ≥1 hit for motif j
hit_rate_mat = np.zeros((len(ct_list), n_motifs), dtype=np.float32)

for ct_idx, ct in enumerate(ct_list):
    seqs   = celltype_seqs[ct]
    n_seqs = len(seqs)
    if n_seqs == 0:
        continue

    print(f"  {ct}: encoding {n_seqs} seqs ...", end=" ", flush=True)
    t0 = time.time()
    seq_tensor = one_hot_encode(seqs)   # (N, 4, L)

    hits_list = tangermeme_fimo(motifs_db, seq_tensor, threshold=PVAL_THRESH)
    # hits_list[j] = DataFrame with columns: motif_name, sequence_name (int seq idx), ...
    for j, h_df in enumerate(hits_list):
        if len(h_df) > 0:
            n_peaks_with_hit = h_df["sequence_name"].nunique()
            hit_rate_mat[ct_idx, j] = n_peaks_with_hit / n_seqs

    print(f"{time.time()-t0:.1f}s  "
          f"(motifs in ≥1 peak: {(hit_rate_mat[ct_idx] > 0).sum()})", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Build result DataFrames
# ──────────────────────────────────────────────────────────────────────────────
# Use TF display names (deduplicated by summing hit rates for same TF)
hit_rate_df = pd.DataFrame(hit_rate_mat, index=ct_list, columns=motif_tf_names)
# Multiple motifs may map to same TF name → take max across duplicates
hit_rate_tf = hit_rate_df.T.groupby(level=0).max().T  # celltypes × unique TFs

hit_rate_tf.to_csv(f"{OUTDIR}/motif_hit_rate_per_celltype.csv")
print(f"\nSaved hit rate matrix: {hit_rate_tf.shape}", flush=True)
print(f"  Saved: {OUTDIR}/motif_hit_rate_per_celltype.csv", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Compute RELATIVE enrichment: log2 fold-change vs. mean across all celltypes
# This removes ubiquitous motifs (zinc fingers that hit everything) and
# highlights celltype-specific TF signatures.
# ──────────────────────────────────────────────────────────────────────────────
global_mean  = hit_rate_tf.mean(axis=0)            # mean hit rate per TF across cts
global_std   = hit_rate_tf.std(axis=0)             # std across cts
# z-score enrichment: (ct_rate - global_mean) / (global_std + eps)
enrich_tf    = (hit_rate_tf - global_mean) / (global_std + 0.02)
# Also compute log2 fold-enrichment: log2((ct_rate + eps) / (global_mean + eps))
log2fc_tf    = np.log2((hit_rate_tf + 0.02) / (global_mean + 0.02))

enrich_tf.to_csv(f"{OUTDIR}/motif_enrichment_zscore_per_celltype.csv")
log2fc_tf.to_csv(f"{OUTDIR}/motif_enrichment_log2fc_per_celltype.csv")
print(f"  Saved: enrichment z-score and log2FC tables", flush=True)

# Top TFs per celltype (by enrichment z-score, not raw hit rate)
top_tfs_records = []
for ct in ct_list:
    top = enrich_tf.loc[ct].nlargest(15)
    for tf, zscore in top.items():
        top_tfs_records.append({
            "celltype": ct, "tf_name": tf,
            "enrichment_zscore": zscore,
            "hit_rate": float(hit_rate_tf.loc[ct, tf]),
            "log2fc": float(log2fc_tf.loc[ct, tf]),
        })
top_tfs_df = pd.DataFrame(top_tfs_records)
top_tfs_df.to_csv(f"{OUTDIR}/motif_enrichment_top_tfs.csv", index=False)

print("\nTop enriched TFs per celltype (by z-score enrichment):")
for ct in ct_list:
    top5 = enrich_tf.loc[ct].nlargest(5)
    top5_str = ", ".join([
        f"{tf}(z={z:.1f},r={hit_rate_tf.loc[ct,tf]:.0%})"
        for tf, z in top5.items()
    ])
    print(f"  {ct}: {top5_str}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Select informative motifs for heatmap (use enrichment z-scores)
# ──────────────────────────────────────────────────────────────────────────────
all_tfs = list(enrich_tf.columns)

# Top 8 per celltype by enrichment z-score
informative_tfs = set()
for ct in ct_list:
    informative_tfs.update(enrich_tf.loc[ct].nlargest(8).index.tolist())

# Also include TF family members that are enriched somewhere
for family, patterns in TF_FAMILY_MAP.items():
    for tf in all_tfs:
        if any(p.upper() in tf.upper() for p in patterns):
            if enrich_tf[tf].max() > 0.5:  # enriched by ≥0.5 std in at least one ct
                informative_tfs.add(tf)

informative_tfs = [tf for tf in informative_tfs if tf in enrich_tf.columns]
# Use enrichment z-scores for the heatmap (not raw hit rates)
hm_data = enrich_tf[informative_tfs].T   # TFs × celltypes
# Remove TFs with very low enrichment range (not discriminating)
hm_data = hm_data.loc[hm_data.max(axis=1) > 0.5]

print(f"\nInformative TFs for heatmap: {len(hm_data)}", flush=True)

if len(hm_data) < 2:
    print("  WARNING: too few informative TFs — skipping heatmap", flush=True)
else:
    # Cluster TFs (rows)
    tf_link  = hierarchy.linkage(pdist(hm_data.values, "euclidean"), method="ward")
    tf_order = hierarchy.leaves_list(tf_link)
    # Cluster celltypes (cols)
    ct_link  = hierarchy.linkage(pdist(hm_data.T.values, "euclidean"), method="ward")
    ct_order = hierarchy.leaves_list(ct_link)
    hm_plot  = hm_data.iloc[tf_order, ct_order]

    # ──── Figure 1: Annotated Heatmap ────────────────────────────────────────
    print("Plotting heatmap ...", flush=True)

    def get_tf_family(tf_name):
        for family, patterns in TF_FAMILY_MAP.items():
            if any(p.upper() in tf_name.upper() for p in patterns):
                return family
        return "Other"

    family_colors = {
        "GATA":      "#e74c3c",
        "TBX":       "#8e44ad",
        "NKX":       "#9b59b6",
        "MEF2":      "#e67e22",
        "MyoD/bHLH": "#d35400",
        "SOX":       "#2980b9",
        "TFAP2":     "#16a085",
        "PAX":       "#1abc9c",
        "FOX":       "#27ae60",
        "ETS":       "#f39c12",
        "TEAD":      "#c0392b",
        "HAND":      "#e91e63",
        "RUNX":      "#795548",
        "KLF/SP":    "#607d8b",
        "HOX":       "#ff9800",
        "Other":     "#cccccc",
    }

    tf_families    = [get_tf_family(tf) for tf in hm_plot.index]
    fam_colors_row = [family_colors.get(f, "#cccccc") for f in tf_families]

    n_tfs = len(hm_plot)
    n_cts = len(hm_plot.columns)
    fig_h = max(8, n_tfs * 0.28 + 3)
    fig_w = max(8, n_cts * 1.4 + 3)

    fig  = plt.figure(figsize=(fig_w, fig_h))
    gs   = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.1, n_cts * 1.4, 2.0],
                             wspace=0.03)

    # Left: TF family colour strip
    ax_fam = fig.add_subplot(gs[0])
    for i, col in enumerate(fam_colors_row):
        ax_fam.barh(n_tfs - i - 1, 1, color=col, edgecolor="white", lw=0.3)
    ax_fam.set_xlim(0, 1)
    ax_fam.set_ylim(-0.5, max(n_tfs - 0.5, 0.5))
    ax_fam.axis("off")

    # Centre: heatmap (enrichment z-scores — diverging colormap)
    ax_hm  = fig.add_subplot(gs[1])
    abs_max = min(4.0, float(np.abs(hm_plot.values).max()))
    im = ax_hm.imshow(hm_plot.values, aspect="auto", cmap="RdBu_r",
                      vmin=-abs_max, vmax=abs_max, interpolation="nearest")
    ax_hm.set_xticks(range(n_cts))
    ax_hm.set_xticklabels(
        [ct.replace("_", "\n") for ct in hm_plot.columns],
        fontsize=8, ha="center"
    )
    ax_hm.set_yticks(range(n_tfs))
    ax_hm.set_yticklabels(hm_plot.index, fontsize=7)
    ax_hm.xaxis.tick_top()
    ax_hm.xaxis.set_label_position("top")
    ax_hm.set_xlabel("Cell type", fontsize=9, labelpad=8)
    plt.colorbar(im, ax=ax_hm, label="Enrichment z-score\nvs. mean across celltypes",
                 fraction=0.04, pad=0.03, shrink=0.4, location="bottom")

    # Right: TF family legend
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis("off")
    seen_fam, handles = [], []
    for fam, col in family_colors.items():
        if fam in tf_families and fam not in seen_fam:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=col, label=fam))
            seen_fam.append(fam)
    ax_leg.legend(handles=handles, loc="upper left", fontsize=8,
                  title="TF family", frameon=False,
                  handlelength=1.2, handleheight=1.0)

    fig.suptitle(
        f"JASPAR CORE vertebrates motif enrichment across cell types\n"
        f"Top {TOP_N_PEAKS} peaks per celltype  |  FIMO p<{PVAL_THRESH:.0e}  |  "
        f"H12CORE, {len(motif_names)} motifs",
        fontsize=9, y=1.01
    )

    out_base = f"{FIG_DIR}/motif_enrichment_celltypes_heatmap"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(f"{out_base}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_base}.{{pdf,png}}", flush=True)

# ──── Figure 2: Per-celltype barplots ────────────────────────────────────────
print("Plotting per-celltype TF barplots ...", flush=True)
n_cols = min(4, len(ct_list))
n_rows = (len(ct_list) + n_cols - 1) // n_cols
fig2, axes = plt.subplots(n_rows, n_cols,
                           figsize=(n_cols * 4, n_rows * 3.5),
                           squeeze=False)

for ax_idx, ct in enumerate(ct_list):
    ax = axes[ax_idx // n_cols][ax_idx % n_cols]
    # Top 10 by enrichment z-score; annotate with raw hit rate
    top10_z = enrich_tf.loc[ct].nlargest(10)
    if len(top10_z) == 0:
        ax.text(0.5, 0.5, "no hits", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title(ct.replace("_", " "), fontsize=9)
        continue
    bar_colors = [family_colors.get(get_tf_family(tf), "#cccccc")
                  for tf in top10_z.index]
    ax.barh(range(len(top10_z))[::-1], top10_z.values,
            color=bar_colors, edgecolor="white", lw=0.4)
    # Annotate with hit rate
    for i, (tf, z) in enumerate(top10_z.items()):
        rate = hit_rate_tf.loc[ct, tf] if tf in hit_rate_tf.columns else 0
        ax.text(max(z + 0.05, 0.1), len(top10_z)-1-i,
                f"{rate:.0%}", va="center", fontsize=6, color="gray")
    ax.set_yticks(range(len(top10_z)))
    ax.set_yticklabels(top10_z.index[::-1], fontsize=7)
    ax.set_xlabel("Enrichment z-score", fontsize=8)
    ax.set_title(ct.replace("_", " "), fontsize=9, fontweight="bold")
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-1, max(4, top10_z.values.max() + 0.5))

for ax_idx in range(len(ct_list), n_rows * n_cols):
    axes[ax_idx // n_cols][ax_idx % n_cols].axis("off")

fig2.suptitle(
    f"Top 10 enriched JASPAR motifs per cell type  (enrichment z-score vs. mean)\n"
    f"Top {TOP_N_PEAKS} peaks per celltype  |  H12CORE {len(motif_names)} motifs",
    fontsize=10
)
plt.tight_layout()
out2_base = f"{FIG_DIR}/motif_enrichment_celltypes_barplots"
fig2.savefig(f"{out2_base}.pdf", bbox_inches="tight", dpi=150)
fig2.savefig(f"{out2_base}.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {out2_base}.{{pdf,png}}", flush=True)

print("\nDone.", flush=True)
