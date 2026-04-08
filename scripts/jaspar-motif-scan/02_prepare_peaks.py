#!/usr/bin/env python
"""
Extract peak coordinates from h5ad files for all three species,
resize to 500bp centered, write BED files, extract FASTA sequences,
and save peak metadata CSVs.

Species-specific coordinate parsing:
  Zebrafish: obs index '1-32-526'  -> chr=chr1, start=32, end=526 (prepend 'chr')
  Mouse:     obs columns chr, start, end (Categorical -> int)
  Human:     same as mouse (genome = hg19)

Output (per species):
  peak_sequences/{species}_peaks.bed
  peak_sequences/{species}_peaks.fa
  peak_sequences/{species}_peak_metadata.csv
"""

import os
import subprocess
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

# =============================================================================
# Config
# =============================================================================

OUTPUT_DIR = Path("/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap/peak_sequences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PEAK_WIDTH = 500  # bp, centered on peak midpoint

SPECIES_CONFIG = {
    "zebrafish": {
        "h5ad":   "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad",
        "genome": "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa",
        "celltype_col": "celltype",
        "timepoint_col": "timepoint",
    },
    "mouse": {
        "h5ad":   "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad",
        "genome": "/hpc/reference/sequencing_alignment/fasta_references/mm10.fa",
        "celltype_col": "top_celltype",
        "timepoint_col": "top_timepoint",
    },
    "human": {
        "h5ad":   "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/human_domcke_2020/peaks_by_pb_celltype_stage_annotated.h5ad",
        "genome": "/hpc/reference/sequencing_alignment/fasta_references/hg19.fa",
        "celltype_col": "top_celltype",
        "timepoint_col": "top_timepoint",
    },
}


# =============================================================================
# Coordinate parsing
# =============================================================================

def parse_peak_coordinates(adata, species):
    """
    Extract chr, start, end from h5ad obs, handling species-specific formats.
    Returns a DataFrame with columns: original_peak_id, chr, start, end.
    """
    if species == "zebrafish":
        # Index format: '1-32-526' (no chr prefix)
        index_list = list(adata.obs_names)
        split = [idx.split("-") for idx in index_list]
        df = pd.DataFrame({
            "original_peak_id": index_list,
            "chr":   ["chr" + p[0] for p in split],
            "start": [int(p[1]) for p in split],
            "end":   [int(p[2]) for p in split],
        })
    else:
        # Mouse / Human: separate columns (stored as Categorical)
        df = pd.DataFrame({
            "original_peak_id": adata.obs_names,
            "chr":   adata.obs["chr"].astype(str),
            "start": adata.obs["start"].astype(int),
            "end":   adata.obs["end"].astype(int),
        })
    return df.reset_index(drop=True)


def resize_to_fixed_width(df, width):
    """Resize peaks to fixed width centered on midpoint.
    If center < width//2 (near chrom start), shift window right to ensure exactly width bp."""
    center = (df["start"] + df["end"]) // 2
    df = df.copy()
    df["start"] = (center - width // 2).clip(lower=0)
    df["end"]   = df["start"] + width  # ensures exactly width bp even when start is clipped
    return df


# =============================================================================
# Main per-species processing
# =============================================================================

def process_species(species, config):
    print(f"\n{'='*60}")
    print(f"Processing: {species}")
    print(f"{'='*60}")

    # --- Load h5ad (backed mode to save memory) ---
    print(f"  Loading h5ad...")
    adata = ad.read_h5ad(config["h5ad"], backed="r")
    print(f"  {adata.n_obs} peaks x {adata.n_vars} pseudobulk groups")

    # --- Parse coordinates ---
    coords = parse_peak_coordinates(adata, species)
    coords = resize_to_fixed_width(coords, PEAK_WIDTH)

    # Verify widths
    widths = coords["end"] - coords["start"]
    assert widths.eq(PEAK_WIDTH).all(), f"Not all peaks are {PEAK_WIDTH}bp!"
    print(f"  All {len(coords)} peaks resized to {PEAK_WIDTH}bp")

    # --- Add peak name and metadata ---
    coords["peak_name"] = [f"{species}_peak_{i}" for i in range(len(coords))]

    # Celltype / timepoint of max accessibility
    celltype_col  = config["celltype_col"]
    timepoint_col = config["timepoint_col"]
    if celltype_col in adata.obs.columns:
        coords["celltype"] = adata.obs[celltype_col].values
    if timepoint_col in adata.obs.columns:
        coords["timepoint"] = adata.obs[timepoint_col].values
    coords["species"] = species

    # Additional annotation columns if present
    for col in ["lineage", "peak_lineage", "leiden_coarse", "peak_type"]:
        if col in adata.obs.columns:
            coords[col] = adata.obs[col].values

    adata.file.close()

    # --- Write BED file ---
    bed_path = OUTPUT_DIR / f"{species}_peaks.bed"
    bed = coords[["chr", "start", "end", "peak_name"]].copy()
    bed["score"] = 0
    bed["strand"] = "."
    bed.to_csv(bed_path, sep="\t", header=False, index=False)
    print(f"  BED written: {bed_path}")

    # --- Extract FASTA with bedtools ---
    fa_path = OUTPUT_DIR / f"{species}_peaks.fa"
    genome  = config["genome"]
    cmd = [
        "/hpc/user_apps/data.science/conda_envs/gimme/bin/bedtools", "getfasta",
        "-fi", genome,
        "-bed", str(bed_path),
        "-fo", str(fa_path),
        "-name",
    ]
    print(f"  Running bedtools getfasta...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"bedtools failed:\n{result.stderr}")
    print(f"  FASTA written: {fa_path}")

    # Parse FASTA headers to get successfully extracted peak names
    # Header format: >peak_name::chr:start-end
    with open(fa_path) as f:
        fasta_peaks = set(
            line[1:].split("::")[0]  # strip '>' and take name before '::'
            for line in f if line.startswith(">")
        )
    n_seqs = len(fasta_peaks)
    n_dropped = len(coords) - n_seqs
    if n_dropped > 0:
        print(f"  WARNING: {n_dropped} peaks dropped by bedtools (out-of-bounds coords)")
        coords = coords[coords["peak_name"].isin(fasta_peaks)].reset_index(drop=True)
    print(f"  Verified: {n_seqs} sequences in FASTA")

    # --- Save metadata CSV ---
    meta_path = OUTPUT_DIR / f"{species}_peak_metadata.csv"
    coords.to_csv(meta_path, index=False)
    print(f"  Metadata written: {meta_path}")

    return coords


# =============================================================================
# Main
# =============================================================================

def main():
    print("Cross-Species Peak Sequence Extraction")
    print(f"Peak width: {PEAK_WIDTH}bp centered")
    print(f"Output dir: {OUTPUT_DIR}")

    summary = {}
    for species, config in SPECIES_CONFIG.items():
        coords = process_species(species, config)
        summary[species] = len(coords)

    print(f"\n{'='*60}")
    print("DONE — Summary:")
    for species, n in summary.items():
        print(f"  {species}: {n} peaks")
    print(f"  Total: {sum(summary.values())} peaks")


if __name__ == "__main__":
    main()
