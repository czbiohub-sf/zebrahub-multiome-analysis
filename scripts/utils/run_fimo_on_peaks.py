"""
Generic FIMO TF-motif scanner for any peak list (CSV with chrom/start/end).

Reuses the motif-database registry from notebooks/EDA_peak_parts_list/
09j_fimo_batch.py (H12CORE / JASPAR2024 / CIS-BP v2 danRer).

Usage:
    python run_fimo_on_peaks.py \\
        --peaks-csv path/to/peaks.csv \\
        --label pax2a \\
        --output-dir path/to/output_dir \\
        [--motif-db h12core | jaspar2024 | cisbpv2_danrer]

Output (in --output-dir):
    {label}_fimo_hits.csv          # one row per (peak, motif) hit
    {label}_fimo_binary.npz        # boolean (n_peaks × n_motifs) matrix
    {label}_fimo_peaks.csv         # peak ids in scan order
    {label}_fimo_tf_summary.csv    # per-TF: n peaks with at least one hit

Env: must have pymemesuite + pysam. Tested with single-cell-base env.
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import pysam

from pymemesuite.common import (MotifFile, Sequence as MemeSequence,
                                 Background, Array as MemeArray)
from pymemesuite.fimo import FIMO as FIMOScanner


# ── Motif DB registry (matches 09j_fimo_batch.py) ────────────────────────────
def _parse_h12core(acc, name):
    return (acc or "").split(".")[0]

def _parse_jaspar2024(acc, name):
    parts = (acc or "").split(":")
    return parts[1] if len(parts) >= 2 else (acc or "")

def _parse_cisbpv2(acc, name):
    return name or (acc or "").split(".")[0]

GRELU_MEME    = ("/hpc/projects/data.science/yangjoon.kim/github_repos/"
                 "gReLU/src/grelu/resources/meme")
SCRATCH_MOTIF = ("/hpc/scratch/group.data.science/yang-joon.kim/"
                 "peak-parts-list-motifs/motif_dbs")

MOTIF_DBS = {
    "h12core":        {"path": f"{GRELU_MEME}/H12CORE_meme_format.meme",
                       "tf_parser": _parse_h12core},
    "jaspar2024":     {"path": f"{GRELU_MEME}/jaspar_2024_consensus.meme",
                       "tf_parser": _parse_jaspar2024},
    "cisbpv2_danrer": {"path": f"{SCRATCH_MOTIF}/cisbpv2_danrer.meme",
                       "tf_parser": _parse_cisbpv2},
}

DEFAULT_FASTA = "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa"
DEFAULT_PVAL  = 1e-4


def _decode(x):
    if x is None:
        return ""
    return x.decode() if isinstance(x, bytes) else str(x)


def run_fimo(peaks_csv: str,
             label: str,
             output_dir: str,
             motif_db: str = "h12core",
             fasta: str = DEFAULT_FASTA,
             pval_thresh: float = DEFAULT_PVAL):
    """Run FIMO on a peak list and write hits + binary matrix to output_dir."""

    db_cfg    = MOTIF_DBS[motif_db]
    meme_path = db_cfg["path"]
    tf_parser = db_cfg["tf_parser"]

    if not os.path.exists(meme_path):
        sys.exit(f"ERROR: MEME file not found: {meme_path}")
    if not os.path.exists(peaks_csv):
        sys.exit(f"ERROR: peaks CSV not found: {peaks_csv}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"=== FIMO scan: {label} ({motif_db}) ===")
    print(f"Start: {time.strftime('%c')}")

    # Load peaks
    peaks = pd.read_csv(peaks_csv)
    needed = {"chrom", "start", "end"}
    if not needed.issubset(peaks.columns):
        sys.exit(f"ERROR: peaks CSV must contain columns {needed}; got {list(peaks.columns)}")
    if "peak_id" not in peaks.columns:
        peaks["peak_id"] = (peaks["chrom"].astype(str) + "-"
                            + peaks["start"].astype(str) + "-"
                            + peaks["end"].astype(str))
    n_peaks = len(peaks)
    print(f"Loaded {n_peaks} peaks from {peaks_csv}")

    # Extract sequences
    print("Extracting sequences ...", flush=True)
    fa = pysam.FastaFile(fasta)
    peak_ids, peak_seqs = [], []
    for _, row in peaks.iterrows():
        chrom = str(row["chrom"])
        start = int(row["start"])
        end   = int(row["end"])
        pid   = row["peak_id"]
        try:
            # try with chr prefix first; fall back to bare
            try:
                seq = fa.fetch(f"chr{chrom}", start, end).upper()
            except Exception:
                seq = fa.fetch(chrom, start, end).upper()
            if seq and len(seq) >= 10 and set(seq) <= set("ACGT"):
                peak_ids.append(pid)
                peak_seqs.append(seq)
        except Exception:
            pass
    fa.close()
    n_valid = len(peak_ids)
    print(f"  Valid sequences: {n_valid}/{n_peaks}")

    # Load motifs
    print(f"Loading motifs from {motif_db} ({meme_path}) ...", flush=True)
    motif_file = MotifFile(meme_path)
    motif_list = list(motif_file)
    motif_accessions = [_decode(m.accession) for m in motif_list]
    motif_display    = [_decode(getattr(m, "name", "")) for m in motif_list]
    motif_tf_names   = [tf_parser(acc, name) for acc, name in zip(motif_accessions, motif_display)]
    n_motifs = len(motif_list)
    print(f"  {n_motifs} motifs, {len(set(motif_tf_names))} unique TFs")

    _bg = Background(motif_list[0].alphabet, MemeArray([0.25, 0.25, 0.25, 0.25]))

    # Run FIMO
    print(f"\nScanning {n_valid} peaks × {n_motifs} motifs (p < {pval_thresh}) ...", flush=True)
    fimo_scanner = FIMOScanner(both_strands=True, threshold=pval_thresh)
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
                        "peak_id":         pid,
                        "tf":              motif_tf_names[j],
                        "motif_accession": motif_accessions[j],
                        "hit_start":       me.start,
                        "hit_end":         me.stop,
                        "strand":          "+" if me.strand == 1 else "-",
                        "score":           me.score,
                        "pvalue":          me.pvalue,
                    })
        if (pi + 1) % 5 == 0 or pi + 1 == n_valid:
            elapsed = time.time() - t0
            rate = (pi + 1) / elapsed
            eta  = (n_valid - pi - 1) / rate
            print(f"  {pi+1}/{n_valid}  ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_valid} peaks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save outputs
    prefix = f"{output_dir}/{label}_fimo"
    pd.DataFrame({"peak_id": peak_ids}).to_csv(f"{prefix}_peaks.csv", index=False)
    np.savez_compressed(f"{prefix}_binary.npz",
                        hit_matrix=hit_matrix,
                        motif_tf_names=np.array(motif_tf_names),
                        motif_accessions=np.array(motif_accessions))
    pos_df = pd.DataFrame(position_rows)
    pos_df.to_csv(f"{prefix}_hits.csv", index=False)

    # Per-TF summary: how many peaks have at least one hit for each TF
    tf_hits = pos_df.groupby("tf")["peak_id"].nunique().sort_values(ascending=False).reset_index()
    tf_hits.columns = ["tf", "n_peaks_with_hit"]
    tf_hits["fraction"] = (tf_hits["n_peaks_with_hit"] / n_valid).round(3)
    tf_hits.to_csv(f"{prefix}_tf_summary.csv", index=False)

    print(f"\nSaved:")
    print(f"  {prefix}_peaks.csv         ({n_valid} peaks)")
    print(f"  {prefix}_binary.npz        ({hit_matrix.shape})")
    print(f"  {prefix}_hits.csv          ({len(pos_df)} hits)")
    print(f"  {prefix}_tf_summary.csv    ({len(tf_hits)} TFs)")
    print(f"\nDone. End: {time.strftime('%c')}")
    return prefix


def main():
    parser = argparse.ArgumentParser(description="Run FIMO on a peak list (CSV with chrom/start/end columns).")
    parser.add_argument("--peaks-csv",  required=True,  help="Path to peaks CSV (must have chrom, start, end columns)")
    parser.add_argument("--label",      required=True,  help="Output filename prefix (e.g., 'pax2a')")
    parser.add_argument("--output-dir", required=True,  help="Directory for output files")
    parser.add_argument("--motif-db",   choices=list(MOTIF_DBS.keys()), default="h12core",
                        help="Motif database (default: h12core)")
    parser.add_argument("--fasta",      default=DEFAULT_FASTA, help=f"Genome FASTA (default: danRer11)")
    parser.add_argument("--pval",       type=float, default=DEFAULT_PVAL,
                        help=f"FIMO p-value threshold (default: {DEFAULT_PVAL})")
    args = parser.parse_args()
    run_fimo(args.peaks_csv, args.label, args.output_dir,
             motif_db=args.motif_db, fasta=args.fasta, pval_thresh=args.pval)


if __name__ == "__main__":
    main()
