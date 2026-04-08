#!/usr/bin/env python
"""
Per-peak motif scoring using GimmeMotifs Scanner.

Reads FASTA sequences for one species and scans against JASPAR MEME motifs,
producing a peaks x motifs score matrix saved as .npz (sparse) + metadata .npz.

Usage:
    python 03_scan_motifs.py --species zebrafish [--fasta path] [--ncpus 32] [--test]

Output:
    motif_scores/{species}_motif_scores.npz         -- sparse score matrix
    motif_scores/{species}_motif_scores_meta.npz    -- peak_names, motif_names arrays
"""

import argparse
import time
import numpy as np
from pathlib import Path
from scipy import sparse

BASE_DIR = Path("/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap")
MOTIF_FILE = BASE_DIR / "motif_database" / "jaspar_fixed.meme.motif"  # whitespace-fixed MEME
SEQ_DIR    = BASE_DIR / "peak_sequences"
OUT_DIR    = BASE_DIR / "motif_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def scan_species(fasta_path: Path, motif_file: Path, n_cpus: int, out_prefix: str):
    from gimmemotifs.motif import read_motifs
    from gimmemotifs.scanner import Scanner

    print(f"  Loading motifs from {motif_file}...")
    motifs = read_motifs(str(motif_file), fmt="meme")
    motif_names = np.array([m.id for m in motifs])
    print(f"  {len(motif_names)} motifs loaded")

    print(f"  Setting up scanner ({n_cpus} CPUs)...")
    s = Scanner(ncpus=n_cpus)
    s.set_motifs(motifs)  # pass list of Motif objects directly
    s.set_threshold(threshold=0.0)  # keep all best scores

    # Read peak names from FASTA headers (format: >peak_name::chr:start-end)
    with open(fasta_path) as f:
        peak_names = [line[1:].split("::")[0] for line in f if line.startswith(">")]
    print(f"  {len(peak_names)} sequences to scan")

    print(f"  Scanning {fasta_path} ...")
    t0 = time.time()
    scores_list = []
    for i, scores in enumerate(s.best_score(str(fasta_path), zscore=False, gc=False)):
        scores_list.append(scores)
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1} peaks scanned  ({elapsed:.0f}s)", flush=True)

    total = time.time() - t0
    print(f"  Done scanning: {len(scores_list)} peaks in {total:.1f}s")

    score_matrix = np.array(scores_list, dtype=np.float32)  # (n_peaks, n_motifs)
    print(f"  Score matrix shape: {score_matrix.shape}")

    # Save sparse matrix
    npz_path = OUT_DIR / f"{out_prefix}_motif_scores.npz"
    sparse.save_npz(str(npz_path), sparse.csr_matrix(score_matrix))
    print(f"  Sparse matrix saved: {npz_path}")

    # Save metadata (peak names + motif names)
    meta_path = OUT_DIR / f"{out_prefix}_motif_scores_meta.npz"
    np.savez(str(meta_path),
             peak_names=np.array(peak_names),
             motif_names=motif_names)
    print(f"  Metadata saved: {meta_path}")

    return score_matrix.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True,
                        choices=["zebrafish", "mouse", "human"],
                        help="Species to scan")
    parser.add_argument("--fasta", default=None,
                        help="Override FASTA path (default: peak_sequences/{species}_peaks.fa)")
    parser.add_argument("--ncpus", type=int, default=32,
                        help="Number of CPUs for GimmeMotifs Scanner")
    parser.add_argument("--test", action="store_true",
                        help="Use zebrafish_test1000.fa for a quick benchmark")
    args = parser.parse_args()

    if args.test:
        fasta = SEQ_DIR / "zebrafish_test1000.fa"
        out_prefix = "zebrafish_test1000"
    elif args.fasta:
        fasta = Path(args.fasta)
        out_prefix = args.species
    else:
        fasta = SEQ_DIR / f"{args.species}_peaks.fa"
        out_prefix = args.species

    print(f"Motif Scanning: {args.species}")
    print(f"  FASTA:  {fasta}")
    print(f"  Motifs: {MOTIF_FILE}")
    print(f"  CPUs:   {args.ncpus}")
    print(f"  Output: {OUT_DIR}/{out_prefix}_*")

    shape = scan_species(fasta, MOTIF_FILE, args.ncpus, out_prefix)
    print(f"\nDone. Final matrix: {shape[0]} peaks x {shape[1]} motifs")


if __name__ == "__main__":
    main()
