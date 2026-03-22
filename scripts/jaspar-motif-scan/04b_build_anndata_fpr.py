#!/usr/bin/env python
"""
Build binarized cross-species peaks × motifs AnnData from FPR-thresholded scans.

Reads per-species sparse binary matrices from motif_scores_fpr/ and stacks
them into a single h5ad. Matrix values are 0 or 1 (uint8).

Output:
    {BASE_DIR}/cross_species_motif_scores_FPR_0.01_binarized.h5ad
        .X     — sparse CSR matrix (n_peaks × 879), binary (0/1, uint8)
        .obs   — peak metadata: species, chr, start, end, original_peak_id, etc.
        .var   — motif metadata: motif_id, tf_name, consensus
        .uns["fpr"]         — 0.01
        .uns["description"] — description string

Usage:
    conda run -p .../gimme python 04b_build_anndata_fpr.py [--fpr 0.01]
"""

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from pathlib import Path

BASE_DIR   = Path("/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap")
SEQ_DIR    = BASE_DIR / "peak_sequences"
SCORE_DIR  = BASE_DIR / "motif_scores_fpr"
DB_DIR     = BASE_DIR / "motif_database"
OUT_DIR    = BASE_DIR

SPECIES = ["zebrafish", "mouse", "human"]

OBS_COLS = [
    "species",
    "original_peak_id",
    "chr", "start", "end",
    "peak_name",
    "celltype",
    "timepoint",
    "peak_lineage",
    "leiden_coarse",
    "peak_type",
    "lineage",     # mouse-only
]


def load_species(species: str):
    print(f"  Loading {species}...")

    mat = sp.load_npz(str(SCORE_DIR / f"{species}_motif_binary.npz"))
    print(f"    Matrix: {mat.shape}  dtype={mat.dtype}  nnz={mat.nnz}")

    meta = np.load(str(SCORE_DIR / f"{species}_motif_binary_meta.npz"), allow_pickle=True)
    peak_names  = meta["peak_names"].tolist()
    motif_names = meta["motif_names"].tolist()
    fpr_val     = float(meta["fpr"])
    print(f"    FPR stored in metadata: {fpr_val}")

    obs = pd.read_csv(SEQ_DIR / f"{species}_peak_metadata.csv", dtype=str)
    obs = obs.set_index("peak_name").loc[peak_names].reset_index()

    return mat, obs, motif_names, fpr_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpr", type=float, default=0.01)
    args = parser.parse_args()

    fpr_str = f"{args.fpr:.2f}".replace(".", "p")  # e.g. "0p01"
    out_name = f"cross_species_motif_scores_FPR_{args.fpr:.3f}_binarized.h5ad"

    print("Building binarized cross-species motif AnnData")
    print(f"  FPR threshold: {args.fpr}")
    print(f"  Input:  {SCORE_DIR}")
    print(f"  Output: {OUT_DIR}/{out_name}\n")

    matrices, obs_dfs, motif_name_sets, fpr_vals = [], [], [], []
    for species in SPECIES:
        mat, obs, motif_names, fpr_val = load_species(species)
        matrices.append(mat)
        obs_dfs.append(obs)
        motif_name_sets.append(motif_names)
        fpr_vals.append(fpr_val)

    # Verify all FPR values match
    assert all(f == fpr_vals[0] for f in fpr_vals), f"FPR mismatch across species: {fpr_vals}"
    print(f"  FPR verified across species: {fpr_vals[0]}")

    # Verify motif order is identical across species
    ref_motifs = motif_name_sets[0]
    for sp_name, mn in zip(SPECIES[1:], motif_name_sets[1:]):
        assert mn == ref_motifs, f"Motif order mismatch for {sp_name}!"
    print(f"  Motif order verified: {len(ref_motifs)} motifs consistent across species")

    # Stack matrices
    print("\n  Stacking sparse matrices...")
    X = sp.vstack(matrices, format="csr")
    print(f"  Combined matrix: {X.shape}  dtype={X.dtype}  nnz={X.nnz}")

    # Sanity check: values must be 0 or 1
    assert X.data.max() == 1, f"Unexpected max value: {X.data.max()}"
    assert X.data.min() == 1, f"Unexpected non-zero min value: {X.data.min()}"
    density = X.nnz / (X.shape[0] * X.shape[1])
    print(f"  Matrix density: {density:.4f}  (expected > FPR={fpr_vals[0]})")

    # Build .obs
    print("  Building .obs...")
    obs_all = pd.concat(obs_dfs, ignore_index=True)
    for col in OBS_COLS:
        if col not in obs_all.columns:
            obs_all[col] = np.nan
    obs_all = obs_all[OBS_COLS].copy()
    obs_all["start"] = pd.to_numeric(obs_all["start"])
    obs_all["end"]   = pd.to_numeric(obs_all["end"])
    obs_all["leiden_coarse"] = pd.to_numeric(obs_all["leiden_coarse"], errors="coerce")
    obs_all.index = obs_all["peak_name"].values
    print(f"  .obs shape: {obs_all.shape}")
    print(f"  Species counts:\n{obs_all['species'].value_counts().to_string()}")

    # Build .var
    print("\n  Building .var...")
    var = pd.read_csv(DB_DIR / "motif_metadata.csv")
    var = var.set_index("motif_id").loc[ref_motifs]
    print(f"  .var shape: {var.shape}")

    # Create AnnData
    print("\n  Creating AnnData...")
    adata = ad.AnnData(X=X, obs=obs_all, var=var)
    adata.uns["fpr"] = fpr_vals[0]
    adata.uns["description"] = (
        f"Cross-species peaks x JASPAR2024 motif binary presence matrix. "
        f"Values are 0 (no significant match) or 1 (motif match above FPR={fpr_vals[0]} threshold). "
        f"Threshold calibrated from random genomic sequences (size=500bp) per species. "
        f"Species: zebrafish (danRer11), mouse (mm10/GRCm38), human (hg19/GRCh37)."
    )
    adata.uns["motif_file"] = str(DB_DIR / "jaspar_fixed.meme.motif")
    adata.uns["peak_width_bp"] = 500
    adata.uns["scanner_method"] = "GimmeMotifs Scanner.count() with set_background(size=500)"

    print(f"\n  AnnData summary:\n{adata}")

    # Save
    out_path = OUT_DIR / out_name
    print(f"\n  Saving to {out_path} ...")
    adata.write_h5ad(str(out_path), compression="gzip")
    print(f"  Done. File size: {out_path.stat().st_size / 1e9:.2f} GB")

    # Quick report
    print("\nVerification summary:")
    print(f"  .X.max() = {X.data.max()}  (expected 1)")
    print(f"  .X.min() nonzero = {X.data.min()}  (expected 1)")
    print(f"  Overall density: {density:.4f}")
    for species in SPECIES:
        mask = obs_all["species"] == species
        n = mask.sum()
        sp_nnz = X[mask.values].nnz
        sp_density = sp_nnz / (n * len(ref_motifs))
        print(f"  {species}: {n} peaks  density={sp_density:.4f}  mean_motifs/peak={sp_nnz/n:.2f}")


if __name__ == "__main__":
    main()
