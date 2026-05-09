# %% Script 09j-merge: Merge FIMO batches + Fisher's exact test
#
# Merges the N batch outputs from 09j_fimo_batch.py, then computes:
#   1. Fisher's exact test (each celltype's 200 peaks vs all other ~6000 pooled)
#   2. FDR correction per celltype
#   3. Enrichment z-score across all 31 celltypes
#   4. Per-peak motif summary for the portal table
#
# Env: single-cell-base (no pysam/pymemesuite needed)

import os, time
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

print("=== Script 09j-merge: Merge FIMO Batches + Fisher's Exact Test ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
SCRATCH     = "/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs"
BATCH_DIR   = f"{SCRATCH}/batches"
TOP200_CSV  = f"{V3_DIR}/V3_all_celltypes_top200_peaks.csv"

# %% Discover celltypes from batch directory
print("\nDiscovering batch files ...", flush=True)
top200 = pd.read_csv(TOP200_CSV)
all_celltypes = sorted(top200["celltype"].unique())
print(f"  Expected: {len(all_celltypes)} celltypes")

# %% Merge position tables from per-celltype batches
all_positions = []
all_peak_ids = []
celltype_peak_map = {}  # celltype → list of peak_ids

for ct in all_celltypes:
    hits_file = f"{BATCH_DIR}/{ct}_hits.csv"
    peaks_file = f"{BATCH_DIR}/{ct}_peaks.csv"

    if not os.path.exists(hits_file):
        print(f"  {ct}: MISSING — skipping")
        continue

    hits = pd.read_csv(hits_file)
    peaks = pd.read_csv(peaks_file)
    all_positions.append(hits)
    peak_list = peaks["peak_id"].tolist()
    all_peak_ids.extend(peak_list)
    celltype_peak_map[ct] = peak_list
    print(f"  {ct}: {len(peak_list)} peaks, {len(hits)} hits")

pos_df = pd.concat(all_positions, ignore_index=True)
print(f"  Total: {len(all_peak_ids)} peaks, {len(pos_df)} motif hits")

# Save merged positions
pos_df.to_csv(f"{SCRATCH}/V3_top200_motif_positions.csv", index=False)
print(f"  Saved: V3_top200_motif_positions.csv")

# %% Build binary hit matrix: peaks × unique TFs
print("\nBuilding binary hit matrix (peaks × TFs) ...", flush=True)
unique_tfs = sorted(pos_df["tf"].unique())
print(f"  {len(unique_tfs)} unique TFs with at least 1 hit")

# For each peak, which TFs have at least one hit?
peak_tf_hits = pos_df.groupby(["peak_id", "tf"]).size().reset_index(name="n_hits")
peak_ids_ordered = sorted(set(all_peak_ids))

# Build boolean matrix
tf_to_col = {tf: i for i, tf in enumerate(unique_tfs)}
pid_to_row = {pid: i for i, pid in enumerate(peak_ids_ordered)}
hit_mat = np.zeros((len(peak_ids_ordered), len(unique_tfs)), dtype=bool)

for _, row in peak_tf_hits.iterrows():
    pid = row["peak_id"]
    tf  = row["tf"]
    if pid in pid_to_row and tf in tf_to_col:
        hit_mat[pid_to_row[pid], tf_to_col[tf]] = True

hit_df = pd.DataFrame(hit_mat, index=peak_ids_ordered, columns=unique_tfs)
hit_df.to_csv(f"{SCRATCH}/V3_top200_motif_hit_matrix.csv")
print(f"  Saved: V3_top200_motif_hit_matrix.csv  ({hit_df.shape})")

# %% Celltype → peak mapping (from batch discovery above)
ct_to_peaks = celltype_peak_map
print(f"\n  {len(ct_to_peaks)} celltypes with FIMO data, {len(peak_ids_ordered)} unique peaks")

# %% Fisher's exact test: each celltype's 200 peaks vs ALL other peaks pooled
print(f"\nRunning Fisher's exact test ({len(all_celltypes)} celltypes × {len(unique_tfs)} TFs) ...",
      flush=True)
t0 = time.time()

fisher_results = []

for ct in all_celltypes:
    fg_peaks = [p for p in ct_to_peaks[ct] if p in pid_to_row]
    bg_peaks = [p for p in peak_ids_ordered if p not in set(fg_peaks)]

    n_fg = len(fg_peaks)
    n_bg = len(bg_peaks)
    if n_fg == 0 or n_bg == 0:
        continue

    fg_idx = [pid_to_row[p] for p in fg_peaks]
    bg_idx = [pid_to_row[p] for p in bg_peaks]

    fg_hits = hit_mat[fg_idx, :]  # (n_fg, n_tfs)
    bg_hits = hit_mat[bg_idx, :]  # (n_bg, n_tfs)

    for j, tf in enumerate(unique_tfs):
        a = fg_hits[:, j].sum()   # fg with hit
        b = n_fg - a              # fg without hit
        c = bg_hits[:, j].sum()   # bg with hit
        d = n_bg - c              # bg without hit

        if a + c == 0:
            continue

        odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative="greater")

        fisher_results.append({
            "celltype": ct,
            "tf": tf,
            "hits_fg": int(a),
            "total_fg": int(n_fg),
            "hit_rate_fg": a / n_fg,
            "hits_bg": int(c),
            "total_bg": int(n_bg),
            "hit_rate_bg": c / n_bg,
            "odds_ratio": odds_ratio,
            "pvalue": pval,
        })

fisher_df = pd.DataFrame(fisher_results)
print(f"  {len(fisher_df)} tests in {time.time()-t0:.0f}s")

# FDR per celltype
fisher_df["fdr"] = np.nan
for ct in all_celltypes:
    mask = fisher_df["celltype"] == ct
    if mask.sum() == 0:
        continue
    _, fdr_vals, _, _ = multipletests(fisher_df.loc[mask, "pvalue"].values, method="fdr_bh")
    fisher_df.loc[mask, "fdr"] = fdr_vals

# Enrichment z-score across all 31 celltypes
fisher_df["enrichment_zscore"] = np.nan
for tf in fisher_df["tf"].unique():
    mask = fisher_df["tf"] == tf
    rates = fisher_df.loc[mask, "hit_rate_fg"].values
    if len(rates) < 2:
        fisher_df.loc[mask, "enrichment_zscore"] = 0.0
        continue
    mean_r = rates.mean()
    std_r  = rates.std()
    fisher_df.loc[mask, "enrichment_zscore"] = (rates - mean_r) / (std_r + 0.02)

# Log2 fold enrichment
fisher_df["log2_fold"] = np.log2(
    (fisher_df["hit_rate_fg"] + 0.01) / (fisher_df["hit_rate_bg"] + 0.01)
)

# Save enrichment results
enrich_path = f"{SCRATCH}/V3_top200_motif_enrichment_all31.csv"
fisher_df.to_csv(enrich_path, index=False)
print(f"  Saved: {enrich_path}  ({len(fisher_df)} rows)")

# %% Per-peak motif summary
print("\nComputing per-peak summary ...", flush=True)
peak_summary = pd.DataFrame(index=peak_ids_ordered)
peak_summary["n_tfs_with_hit"] = hit_df.sum(axis=1)

# Total hits (from position table)
hits_per_peak = pos_df.groupby("peak_id").size()
peak_summary["n_total_hits"] = hits_per_peak.reindex(peak_ids_ordered).fillna(0).astype(int)

# Top 5 TFs per peak
tf_counts = pos_df.groupby(["peak_id", "tf"]).size().reset_index(name="n_hits")
top_tfs_per_peak = {}
for pid in peak_ids_ordered:
    pid_tfs = tf_counts[tf_counts["peak_id"] == pid].nlargest(5, "n_hits")
    if len(pid_tfs) > 0:
        top_tfs_per_peak[pid] = ", ".join(
            f"{r['tf']}({r['n_hits']})" for _, r in pid_tfs.iterrows())
    else:
        top_tfs_per_peak[pid] = ""

peak_summary["top_tfs"] = pd.Series(top_tfs_per_peak)
peak_summary["has_motif_support"] = peak_summary["n_tfs_with_hit"] >= 3

summary_path = f"{SCRATCH}/V3_top200_peak_motif_summary.csv"
peak_summary.to_csv(summary_path)
print(f"  Saved: {summary_path}")

# %% Portal-ready table: top-200 + motif columns
print("Building portal table ...", flush=True)
portal_table = top200.merge(
    peak_summary[["n_tfs_with_hit", "n_total_hits", "top_tfs", "has_motif_support"]],
    left_on="peak_id", right_index=True, how="left"
)
portal_path = f"{SCRATCH}/V3_all_celltypes_top200_peaks_with_motifs.csv"
portal_table.to_csv(portal_path, index=False)
print(f"  Saved: {portal_path}  ({len(portal_table)} rows)")

# %% Summary stats
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Peaks scanned: {len(peak_ids_ordered)}")
print(f"TFs with any hit: {len(unique_tfs)}")
print(f"Total motif hits: {len(pos_df)}")
print(f"Peaks with motif support (>=3 TFs): {peak_summary['has_motif_support'].sum()} "
      f"({100*peak_summary['has_motif_support'].mean():.0f}%)")
print(f"Peaks with 0 TF hits: {(peak_summary['n_tfs_with_hit'] == 0).sum()}")

print(f"\nTop enriched TFs per celltype (FDR < 0.05):")
for ct in all_celltypes:
    sig = fisher_df[(fisher_df["celltype"] == ct) & (fisher_df["fdr"] < 0.05)]
    n_sig = len(sig)
    top3 = sig.nlargest(3, "enrichment_zscore")["tf"].tolist()
    print(f"  {ct:<30} {n_sig:>3} sig TFs  top: {', '.join(top3)}")

print(f"\nOutput files:")
print(f"  {SCRATCH}/V3_top200_motif_positions.csv")
print(f"  {SCRATCH}/V3_top200_motif_hit_matrix.csv")
print(f"  {SCRATCH}/V3_top200_motif_enrichment_all31.csv")
print(f"  {SCRATCH}/V3_top200_peak_motif_summary.csv")
print(f"  {SCRATCH}/V3_all_celltypes_top200_peaks_with_motifs.csv")

print(f"\nDone. End: {time.strftime('%c')}")
