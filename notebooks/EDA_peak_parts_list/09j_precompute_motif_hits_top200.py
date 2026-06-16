# %% Script 09j: Precompute FIMO motif hits for top-200 peaks (all 31 celltypes)
#
# Runs FIMO on all 6,200 peaks from V3_all_celltypes_top200_peaks.csv.
# Saves results to scratch drive for portal consumption:
#   1. Binary hit matrix (peaks × TFs) as sparse parquet
#   2. Motif position table (peak_id, tf, start, end, strand, pvalue)
#   3. Per-peak summary (n_motif_hits, top_tfs)
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu (needs pysam, pymemesuite)

import os, sys, time, argparse
import numpy as np
import pandas as pd
import pysam
from pymemesuite.common import (MotifFile, Sequence as MemeSequence,
                                 Background, Array as MemeArray)
from pymemesuite.fimo import FIMO as FIMOScanner

# %% Motif-database registry
#
# Each entry:
#   path      — MEME-format motif file
#   suffix    — appended to every output filename (empty = default layout)
#   tf_parser — given (accession, name) from pymemesuite, return a TF symbol
#               used for TF-level deduplication / output columns.
#
# H12CORE entries look like `MOTIF AHR.H12CORE.0.P.B`       → TF = AHR
# JASPAR2024 entries look like `MOTIF C001:HES_SREBF:bHLH`  → TF = HES_SREBF (cluster)
# CIS-BP v2 entries look like `MOTIF M00008_2.00 hmga1a`    → TF = hmga1a (from `name`)

def _parse_h12core(acc, name):
    return (acc or "").split(".")[0]

def _parse_jaspar2024(acc, name):
    parts = (acc or "").split(":")
    return parts[1] if len(parts) >= 2 else (acc or "")

def _parse_cisbpv2(acc, name):
    return name or (acc or "").split(".")[0]

GRELU_MEME = ("/hpc/projects/data.science/yangjoon.kim/github_repos/"
              "gReLU/src/grelu/resources/meme")
SCRATCH_ROOT = "/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs"

MOTIF_DBS = {
    "h12core": {
        "path":      f"{GRELU_MEME}/H12CORE_meme_format.meme",
        "suffix":    "",
        "tf_parser": _parse_h12core,
    },
    "jaspar2024": {
        "path":      f"{GRELU_MEME}/jaspar_2024_consensus.meme",
        "suffix":    "_jaspar2024",
        "tf_parser": _parse_jaspar2024,
    },
    "cisbpv2_danrer": {
        "path":      f"{SCRATCH_ROOT}/motif_dbs/cisbpv2_danrer.meme",
        "suffix":    "_cisbpv2_danrer",
        "tf_parser": _parse_cisbpv2,
    },
}

# %% CLI
parser = argparse.ArgumentParser(description="Precompute FIMO motif hits for top-200 peaks.")
parser.add_argument("--motif-db", choices=list(MOTIF_DBS.keys()), default="h12core",
                    help="Motif database to scan against (default: h12core = HOCOMOCO v12 CORE)")
args = parser.parse_args()

db_cfg    = MOTIF_DBS[args.motif_db]
MEME_PATH = db_cfg["path"]
SUFFIX    = db_cfg["suffix"]
tf_parser = db_cfg["tf_parser"]

print("=== Script 09j: Precompute FIMO Motif Hits (Top-200 × 31 Celltypes) ===")
print(f"Start:    {time.strftime('%c')}")
print(f"Motif DB: {args.motif_db}")
print(f"MEME:     {MEME_PATH}")
print(f"Suffix:   '{SUFFIX}'")

if not os.path.exists(MEME_PATH):
    sys.exit(f"ERROR: MEME file not found: {MEME_PATH}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"

SCRATCH = SCRATCH_ROOT
os.makedirs(SCRATCH, exist_ok=True)

TOP200_CSV  = f"{V3_DIR}/V3_all_celltypes_top200_peaks.csv"
FASTA_PATH  = "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa"

PVAL_THRESH = 1e-4

# %% Load top-200 peaks
print("\nLoading top-200 peaks ...", flush=True)
top200 = pd.read_csv(TOP200_CSV)
print(f"  {len(top200)} rows, {top200['celltype'].nunique()} celltypes")

# Deduplicate peaks (same peak can appear in multiple celltypes)
unique_peaks = top200[["peak_id", "chrom", "start", "end"]].drop_duplicates("peak_id")
print(f"  Unique peaks: {len(unique_peaks)}")

# %% Extract sequences
print("Extracting sequences ...", flush=True)
fa = pysam.FastaFile(FASTA_PATH)

peak_seqs = {}  # peak_id → sequence
for _, row in unique_peaks.iterrows():
    chrom = str(row["chrom"])
    start = int(row["start"])
    end   = int(row["end"])
    pid   = row["peak_id"]
    try:
        seq = fa.fetch(f"chr{chrom}", start, end).upper()
        if seq and len(seq) >= 10 and set(seq) <= set("ACGT"):
            peak_seqs[pid] = seq
    except:
        pass

fa.close()
print(f"  Valid sequences: {len(peak_seqs)} / {len(unique_peaks)}")

# %% Load motifs (database-dependent TF name extraction)
print(f"Loading motifs from {args.motif_db} ...", flush=True)
motif_file = MotifFile(MEME_PATH)
motif_list = list(motif_file)

def _decode(x):
    if x is None:
        return ""
    return x.decode() if isinstance(x, bytes) else str(x)

motif_accessions = [_decode(m.accession) for m in motif_list]
motif_display    = [_decode(getattr(m, "name", "")) for m in motif_list]
motif_tf_names   = [tf_parser(acc, name) for acc, name in zip(motif_accessions, motif_display)]
motif_names      = motif_accessions  # preserve original variable name used downstream

n_motifs = len(motif_list)
print(f"  {n_motifs} motifs, {len(set(motif_tf_names))} unique TFs")

_bg = Background(motif_list[0].alphabet, MemeArray([0.25, 0.25, 0.25, 0.25]))

# TF name deduplication
unique_tfs = sorted(set(motif_tf_names))
tf_to_motif_idx = {}
for j, tf in enumerate(motif_tf_names):
    tf_to_motif_idx.setdefault(tf, []).append(j)

# %% Run FIMO on all unique peaks
print(f"\nRunning FIMO on {len(peak_seqs)} peaks × {n_motifs} motifs ...", flush=True)
fimo_scanner = FIMOScanner(both_strands=True, threshold=PVAL_THRESH)

peak_ids_ordered = sorted(peak_seqs.keys())
n_peaks = len(peak_ids_ordered)

# Output 1: binary hit matrix (peaks × motifs)
hit_matrix = np.zeros((n_peaks, n_motifs), dtype=bool)

# Output 2: position table
position_rows = []

t0 = time.time()
for pi, pid in enumerate(peak_ids_ordered):
    seq = peak_seqs[pid]
    meme_seq = MemeSequence(seq, f"peak_{pi}".encode())

    for j, motif in enumerate(motif_list):
        pattern = fimo_scanner.score_motif(motif, [meme_seq], _bg)
        if pattern is not None:
            for me in pattern.matched_elements:
                hit_matrix[pi, j] = True
                position_rows.append({
                    "peak_id": pid,
                    "tf": motif_tf_names[j],
                    "motif_accession": motif_names[j],
                    "hit_start": me.start,
                    "hit_end": me.stop,
                    "strand": "+" if me.strand == 1 else "-",
                    "score": me.score,
                    "pvalue": me.pvalue,
                })

    if (pi + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (pi + 1) / elapsed
        eta = (n_peaks - pi - 1) / rate
        print(f"  {pi+1}/{n_peaks} peaks  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
              flush=True)

elapsed = time.time() - t0
print(f"  Done: {n_peaks} peaks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# %% Deduplicate hit matrix by TF name (any variant counts)
print("\nDeduplicating by TF name ...", flush=True)
hit_df = pd.DataFrame(hit_matrix, index=peak_ids_ordered, columns=motif_tf_names)
hit_by_tf = hit_df.T.groupby(level=0).any().T  # peaks × unique_TFs (bool)
print(f"  Hit matrix: {hit_by_tf.shape} (peaks × unique TFs)")

# %% Save outputs to scratch

# 1. Binary hit matrix (peaks × TFs)
hit_path = f"{SCRATCH}/V3_top200_motif_hit_matrix{SUFFIX}.csv"
hit_by_tf.to_csv(hit_path)
print(f"\nSaved hit matrix: {hit_path}")

# 2. Full position table
pos_df = pd.DataFrame(position_rows)
pos_path = f"{SCRATCH}/V3_top200_motif_positions{SUFFIX}.csv"
pos_df.to_csv(pos_path, index=False)
print(f"Saved positions: {pos_path}  ({len(pos_df)} rows)")

# 3. Per-peak summary
print("Computing per-peak summary ...", flush=True)
peak_summary = pd.DataFrame(index=peak_ids_ordered)
peak_summary["n_tfs_with_hit"] = hit_by_tf.sum(axis=1)
peak_summary["n_total_hits"] = pd.DataFrame(position_rows).groupby("peak_id").size().reindex(peak_ids_ordered).fillna(0).astype(int)

# Top 5 TFs per peak (by number of hits)
top_tfs_per_peak = {}
if len(pos_df) > 0:
    tf_counts = pos_df.groupby(["peak_id", "tf"]).size().reset_index(name="n_hits")
    for pid in peak_ids_ordered:
        pid_tfs = tf_counts[tf_counts["peak_id"] == pid].nlargest(5, "n_hits")
        top_tfs_per_peak[pid] = ", ".join(
            f"{r['tf']}({r['n_hits']})" for _, r in pid_tfs.iterrows())

peak_summary["top_tfs"] = pd.Series(top_tfs_per_peak)
peak_summary["has_motif_support"] = peak_summary["n_tfs_with_hit"] >= 3

summary_path = f"{SCRATCH}/V3_top200_peak_motif_summary{SUFFIX}.csv"
peak_summary.to_csv(summary_path)
print(f"Saved summary: {summary_path}")

# 4. Also save a version merged with the top-200 table for direct portal use
print("Merging with top-200 table ...", flush=True)
portal_table = top200.merge(
    peak_summary[["n_tfs_with_hit", "n_total_hits", "top_tfs", "has_motif_support"]],
    left_on="peak_id", right_index=True, how="left"
)
portal_path = f"{SCRATCH}/V3_all_celltypes_top200_peaks_with_motifs{SUFFIX}.csv"
portal_table.to_csv(portal_path, index=False)
print(f"Saved portal table: {portal_path}  ({len(portal_table)} rows)")

# %% Summary stats
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Unique peaks scanned: {n_peaks}")
print(f"Total motif hits: {len(pos_df)}")
print(f"Peaks with >= 1 TF hit: {(peak_summary['n_tfs_with_hit'] > 0).sum()}")
print(f"Peaks with >= 3 TF hits (motif-supported): {peak_summary['has_motif_support'].sum()}")
print(f"Peaks with 0 TF hits: {(peak_summary['n_tfs_with_hit'] == 0).sum()}")
print(f"\nMedian TFs per peak: {peak_summary['n_tfs_with_hit'].median():.0f}")
print(f"Mean TFs per peak: {peak_summary['n_tfs_with_hit'].mean():.1f}")

print(f"\nOutput files (scratch):")
print(f"  {hit_path}")
print(f"  {pos_path}")
print(f"  {summary_path}")
print(f"  {portal_path}")

print(f"\nDone. End: {time.strftime('%c')}")
