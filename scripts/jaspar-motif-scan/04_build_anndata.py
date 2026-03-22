#!/usr/bin/env python
"""
Harmonize per-species motif score matrices into a single AnnData h5ad.

Loads sparse peaks × motifs matrices for zebrafish, mouse, and human,
stacks them vertically, and saves as h5ad with peak metadata in .obs
and JASPAR motif metadata in .var.

Output:
    {OUT_DIR}/cross_species_motif_scores.h5ad
        .X     — sparse CSR matrix (n_peaks × 879 motifs), raw PWM best-scores
        .obs   — peak metadata: species, chr, start, end, original_peak_id, etc.
        .var   — motif metadata: motif_id (JASPAR), tf_name, consensus

Usage:
    conda run -p .../gimme python 04_build_anndata.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from pathlib import Path

BASE_DIR  = Path("/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap")
SEQ_DIR   = BASE_DIR / "peak_sequences"
SCORE_DIR = BASE_DIR / "motif_scores"
DB_DIR    = BASE_DIR / "motif_database"
OUT_DIR   = BASE_DIR

SPECIES = ["zebrafish", "mouse", "human"]

# Columns to keep in .obs — species with missing columns will get NaN
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

    # --- Sparse score matrix ---
    mat = sp.load_npz(str(SCORE_DIR / f"{species}_motif_scores.npz"))
    print(f"    Matrix: {mat.shape}")

    # --- Motif names (var axis) ---
    meta = np.load(str(SCORE_DIR / f"{species}_motif_scores_meta.npz"), allow_pickle=True)
    peak_names  = meta["peak_names"].tolist()
    motif_names = meta["motif_names"].tolist()

    # --- Peak metadata (obs) ---
    obs = pd.read_csv(SEQ_DIR / f"{species}_peak_metadata.csv", dtype=str)

    # Align obs rows to the order in the score matrix (peak_names from FASTA scan)
    obs = obs.set_index("peak_name").loc[peak_names].reset_index()

    return mat, obs, motif_names


def main():
    print("Building cross-species motif AnnData")
    print(f"  Input:  {SCORE_DIR}")
    print(f"  Output: {OUT_DIR}/cross_species_motif_scores.h5ad\n")

    # Load all species
    matrices, obs_dfs, motif_name_sets = [], [], []
    for species in SPECIES:
        mat, obs, motif_names = load_species(species)
        matrices.append(mat)
        obs_dfs.append(obs)
        motif_name_sets.append(motif_names)

    # Verify motif order is identical across species
    ref_motifs = motif_name_sets[0]
    for sp_name, mn in zip(SPECIES[1:], motif_name_sets[1:]):
        assert mn == ref_motifs, f"Motif order mismatch for {sp_name}!"
    print(f"  Motif order verified: {len(ref_motifs)} motifs consistent across species")

    # Stack matrices
    print("\n  Stacking sparse matrices...")
    X = sp.vstack(matrices, format="csr")
    print(f"  Combined matrix: {X.shape}")

    # Concatenate obs
    print("  Building .obs...")
    obs_all = pd.concat(obs_dfs, ignore_index=True)

    # Ensure all OBS_COLS exist (fill missing with NaN)
    for col in OBS_COLS:
        if col not in obs_all.columns:
            obs_all[col] = np.nan
    obs_all = obs_all[OBS_COLS].copy()

    # Convert numeric columns
    obs_all["start"] = pd.to_numeric(obs_all["start"])
    obs_all["end"]   = pd.to_numeric(obs_all["end"])
    obs_all["leiden_coarse"] = pd.to_numeric(obs_all["leiden_coarse"], errors="coerce")

    obs_all.index = obs_all["peak_name"].values
    print(f"  .obs shape: {obs_all.shape}")
    print(f"  Species counts:\n{obs_all['species'].value_counts().to_string()}")

    # Build .var from motif metadata
    print("\n  Building .var...")
    var = pd.read_csv(DB_DIR / "motif_metadata.csv")
    var = var.set_index("motif_id")
    # Reorder to match matrix column order
    var = var.loc[ref_motifs]
    print(f"  .var shape: {var.shape}")

    # Create AnnData
    print("\n  Creating AnnData...")
    adata = ad.AnnData(X=X, obs=obs_all, var=var)
    adata.uns["description"] = (
        "Cross-species peaks x JASPAR2024 motif scores. "
        "Values are GimmeMotifs best_score() PWM log-odds scores (continuous, raw). "
        "Species: zebrafish (danRer11), mouse (mm10/GRCm38), human (hg19/GRCh37)."
    )
    adata.uns["motif_file"] = str(DB_DIR / "jaspar_fixed.meme.motif")
    adata.uns["peak_width_bp"] = 500

    print(f"\n  AnnData summary:\n{adata}")

    # Save
    out_path = OUT_DIR / "cross_species_motif_scores.h5ad"
    print(f"\n  Saving to {out_path} ...")
    adata.write_h5ad(str(out_path), compression="gzip")
    print(f"  Done. File size: {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
