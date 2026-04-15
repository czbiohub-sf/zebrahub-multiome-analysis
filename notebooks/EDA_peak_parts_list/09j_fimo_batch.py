# %% Script 09j-batch: FIMO scanning for one celltype's top-200 peaks
#
# Designed for SLURM array: each task scans one celltype's 200 peaks
# against all 1,443 JASPAR motifs.
#
# Usage: python 09j_fimo_batch.py --celltype-idx 0
#        (index into the sorted list of 31 celltypes)
#
# Output (to scratch):
#   V3_fimo_{celltype}_hits.csv     — motif hit positions
#   V3_fimo_{celltype}_binary.npz   — binary hit matrix (200 peaks × 1443 motifs)
#   V3_fimo_{celltype}_peaks.csv    — peak IDs + metadata for this celltype
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu

import os, sys, time, argparse
import numpy as np
import pandas as pd
import pysam
from pymemesuite.common import (MotifFile, Sequence as MemeSequence,
                                 Background, Array as MemeArray)
from pymemesuite.fimo import FIMO as FIMOScanner

parser = argparse.ArgumentParser()
parser.add_argument("--celltype-idx", type=int, required=True,
                    help="Index into sorted list of 31 celltypes")
args = parser.parse_args()

CT_IDX = args.celltype_idx

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/batches"
os.makedirs(SCRATCH, exist_ok=True)

TOP200_CSV = f"{V3_DIR}/V3_all_celltypes_top200_peaks.csv"
FASTA_PATH = "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa"
MEME_PATH  = ("/hpc/projects/data.science/yangjoon.kim/github_repos/"
              "gReLU/src/grelu/resources/meme/H12CORE_meme_format.meme")

PVAL_THRESH = 1e-4

# %% Load peaks for this celltype
print("Loading top-200 peaks ...", flush=True)
top200 = pd.read_csv(TOP200_CSV)
all_celltypes = sorted(top200["celltype"].unique())

if CT_IDX >= len(all_celltypes):
    print(f"ERROR: celltype-idx {CT_IDX} >= {len(all_celltypes)} celltypes")
    sys.exit(1)

celltype = all_celltypes[CT_IDX]
ct_peaks = top200[top200["celltype"] == celltype].copy()
n_peaks = len(ct_peaks)

print(f"=== FIMO: {celltype} (idx {CT_IDX}/{len(all_celltypes)}) — {n_peaks} peaks ===")
print(f"Start: {time.strftime('%c')}")

# %% Extract sequences
print("Extracting sequences ...", flush=True)
fa = pysam.FastaFile(FASTA_PATH)

peak_ids = []
peak_seqs = []
for _, row in ct_peaks.iterrows():
    chrom = str(row["chrom"])
    start = int(row["start"])
    end   = int(row["end"])
    pid   = row["peak_id"]
    try:
        seq = fa.fetch(f"chr{chrom}", start, end).upper()
        if seq and len(seq) >= 10 and set(seq) <= set("ACGT"):
            peak_ids.append(pid)
            peak_seqs.append(seq)
    except:
        pass

fa.close()
n_valid = len(peak_ids)
print(f"  Valid sequences: {n_valid}/{n_peaks}")

# %% Load motifs
print("Loading JASPAR motifs ...", flush=True)
motif_file = MotifFile(MEME_PATH)
motif_list = list(motif_file)
motif_names = [m.accession.decode() if isinstance(m.accession, bytes) else str(m.accession)
               for m in motif_list]
motif_tf_names = [n.split(".")[0] for n in motif_names]
n_motifs = len(motif_list)
print(f"  {n_motifs} motifs")

_bg = Background(motif_list[0].alphabet, MemeArray([0.25, 0.25, 0.25, 0.25]))

# %% Run FIMO
print(f"\nScanning {n_valid} peaks × {n_motifs} motifs ...", flush=True)
fimo_scanner = FIMOScanner(both_strands=True, threshold=PVAL_THRESH)

hit_matrix = np.zeros((n_valid, n_motifs), dtype=bool)
position_rows = []

t0 = time.time()
for pi, (pid, seq) in enumerate(zip(peak_ids, peak_seqs)):
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

    if (pi + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (pi + 1) / elapsed
        eta = (n_valid - pi - 1) / rate
        print(f"  {pi+1}/{n_valid}  ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

elapsed = time.time() - t0
print(f"  Done: {n_valid} peaks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# %% Save outputs
prefix = f"{SCRATCH}/{celltype}"

pd.DataFrame({"peak_id": peak_ids}).to_csv(f"{prefix}_peaks.csv", index=False)
np.savez_compressed(f"{prefix}_binary.npz",
                    hit_matrix=hit_matrix,
                    motif_tf_names=motif_tf_names)
pos_df = pd.DataFrame(position_rows)
pos_df.to_csv(f"{prefix}_hits.csv", index=False)

print(f"\nSaved: {prefix}_peaks.csv  ({n_valid} peaks)")
print(f"Saved: {prefix}_binary.npz  ({hit_matrix.shape})")
print(f"Saved: {prefix}_hits.csv  ({len(pos_df)} hits)")
print(f"\nDone. End: {time.strftime('%c')}")
