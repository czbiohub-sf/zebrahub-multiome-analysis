#!/usr/bin/env python
"""
Per-peak binary motif scanning using GimmeMotifs Scanner with FPR threshold.

Uses Scanner.count() with genome-calibrated FPR cutoffs (background estimated
from random genomic sequences, 500bp, matching our peak width).

Output per species:
    motif_scores_fpr/{species}_motif_binary.npz      -- sparse binary matrix (0/1)
    motif_scores_fpr/{species}_motif_binary_meta.npz -- peak_names, motif_names arrays

Usage:
    python 03b_scan_motifs_fpr.py --species zebrafish --ncpus 32 --fpr 0.01
    python 03b_scan_motifs_fpr.py --species zebrafish --test --ncpus 4  # 1000 peaks
"""

import argparse
import time
import numpy as np
from pathlib import Path
from scipy import sparse

BASE_DIR   = Path("/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap")
MOTIF_FILE = BASE_DIR / "motif_database" / "jaspar_fixed.meme.motif"
SEQ_DIR    = BASE_DIR / "peak_sequences"
OUT_DIR    = BASE_DIR / "motif_scores_fpr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-species genome FASTA for background calibration
GENOME_FASTA = {
    "zebrafish": "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa",
    "mouse":     "/hpc/reference/sequencing_alignment/fasta_references/mm10.fa",
    "human":     "/hpc/reference/sequencing_alignment/fasta_references/hg19.fa",
}

PEAK_WIDTH = 500  # must match set_background(size=...) for correct FPR calibration


def scan_species(fasta_path: Path, motif_file: Path, genome_fasta: str,
                 n_cpus: int, fpr: float, out_prefix: str):
    from gimmemotifs.motif import read_motifs
    from gimmemotifs.scanner import Scanner

    print(f"  Loading motifs from {motif_file}...")
    motifs = read_motifs(str(motif_file), fmt="meme")
    motif_names = np.array([m.id for m in motifs])
    print(f"  {len(motif_names)} motifs loaded")

    print(f"  Setting up scanner ({n_cpus} CPUs)...")
    s = Scanner(ncpus=n_cpus)
    s.set_motifs(motifs)

    print(f"  Calibrating background: genome={genome_fasta}, size={PEAK_WIDTH}, fpr={fpr}")
    print("  (First run per genome/fpr/size combo will compute background; subsequent runs use cache)")
    s.set_background(genome=genome_fasta, size=PEAK_WIDTH)
    s.set_threshold(fpr=fpr)
    print("  Background calibration complete.")

    # Read peak names from FASTA headers (format: >peak_name::chr:start-end)
    with open(fasta_path) as f:
        peak_names = [line[1:].split("::")[0] for line in f if line.startswith(">")]
    n_peaks = len(peak_names)
    print(f"  {n_peaks} sequences to scan")

    print(f"  Scanning {fasta_path} ...")
    t0 = time.time()
    rows, cols = [], []

    for i, counts in enumerate(s.count(str(fasta_path), nreport=1, scan_rc=True)):
        # counts: list of ints, one per motif (0 or positive count)
        for j, c in enumerate(counts):
            if c > 0:
                rows.append(i)
                cols.append(j)
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            nnz = len(rows)
            print(f"    {i+1}/{n_peaks} peaks  ({elapsed:.0f}s)  nnz={nnz}", flush=True)

    total = time.time() - t0
    n_motifs = len(motif_names)
    nnz = len(rows)
    print(f"  Done: {n_peaks} peaks x {n_motifs} motifs in {total:.1f}s  nnz={nnz}")
    print(f"  Mean motifs/peak: {nnz/n_peaks:.2f}  (FPR={fpr} baseline: {fpr*n_motifs:.1f})")

    # Build sparse binary matrix
    data = np.ones(nnz, dtype=np.uint8)
    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_peaks, n_motifs),
        dtype=np.uint8,
    )

    # Save sparse matrix
    npz_path = OUT_DIR / f"{out_prefix}_motif_binary.npz"
    sparse.save_npz(str(npz_path), mat)
    print(f"  Sparse binary matrix saved: {npz_path}")

    # Save metadata
    meta_path = OUT_DIR / f"{out_prefix}_motif_binary_meta.npz"
    np.savez(str(meta_path),
             peak_names=np.array(peak_names),
             motif_names=motif_names,
             fpr=np.float32(fpr))
    print(f"  Metadata saved: {meta_path}")

    return mat.shape, nnz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True,
                        choices=["zebrafish", "mouse", "human"])
    parser.add_argument("--fasta", default=None,
                        help="Override FASTA path")
    parser.add_argument("--ncpus", type=int, default=32)
    parser.add_argument("--fpr", type=float, default=0.01,
                        help="False positive rate for motif threshold (default: 0.01)")
    parser.add_argument("--test", action="store_true",
                        help="Use zebrafish_test1000.fa for quick benchmarking")
    args = parser.parse_args()

    if args.test:
        fasta = SEQ_DIR / "zebrafish_test1000.fa"
        out_prefix = "zebrafish_test1000_fpr"
        genome_fasta = GENOME_FASTA["zebrafish"]
    elif args.fasta:
        fasta = Path(args.fasta)
        out_prefix = f"{args.species}_fpr"
        genome_fasta = GENOME_FASTA[args.species]
    else:
        fasta = SEQ_DIR / f"{args.species}_peaks.fa"
        out_prefix = args.species
        genome_fasta = GENOME_FASTA[args.species]

    print(f"FPR Binary Motif Scan: {args.species}")
    print(f"  FASTA:   {fasta}")
    print(f"  Genome:  {genome_fasta}")
    print(f"  Motifs:  {MOTIF_FILE}")
    print(f"  CPUs:    {args.ncpus}")
    print(f"  FPR:     {args.fpr}")
    print(f"  Output:  {OUT_DIR}/{out_prefix}_*\n")

    shape, nnz = scan_species(
        fasta_path=fasta,
        motif_file=MOTIF_FILE,
        genome_fasta=genome_fasta,
        n_cpus=args.ncpus,
        fpr=args.fpr,
        out_prefix=out_prefix,
    )
    density = nnz / (shape[0] * shape[1])
    print(f"\nDone. Matrix: {shape[0]} peaks x {shape[1]} motifs  nnz={nnz}  density={density:.4f}")


if __name__ == "__main__":
    main()
