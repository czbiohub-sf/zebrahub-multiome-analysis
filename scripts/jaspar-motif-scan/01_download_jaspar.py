#!/usr/bin/env python
"""
Download JASPAR 2024 CORE vertebrates non-redundant motifs in MEME format
and convert to HOMER format.

Adapted from Ben Iovino's get_jaspar.py (af3-tf-motif repo).
Key difference: uses non-redundant collection (~900 motifs vs ~2000+).

Output:
    motif_database/jaspar.meme.motif   -- raw MEME download
    motif_database/jaspar.homer.motif  -- converted to HOMER format
    motif_database/motif_metadata.csv  -- motif_id, tf_name, consensus
"""

import re
import requests
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap/motif_database")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JASPAR_URL = (
    "https://jaspar.elixir.no/download/data/2024/CORE/"
    "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
)


def download_jaspar(output_file: Path):
    print(f"Downloading JASPAR 2024 CORE vertebrates non-redundant (MEME format)...")
    response = requests.get(JASPAR_URL)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"  Saved to: {output_file}")
    else:
        raise RuntimeError(f"Download failed. Status code: {response.status_code}")


def generate_consensus(matrix: list) -> str:
    """Generate consensus sequence from position probability matrix (A,C,G,T)."""
    consensus = ""
    alphabet = "ACGT"
    for position_probs in matrix:
        consensus += alphabet[position_probs.index(max(position_probs))]
    return consensus


def convert_meme_to_homer(input_file: Path, output_file: Path) -> list:
    """
    Convert MEME format to HOMER motif format.
    Returns list of metadata dicts (motif_id, tf_name, consensus).
    """
    metadata = []

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        current_motif_id = None
        current_motif_name = None
        matrix_width = 0
        matrix_lines_read = 0
        matrix = []

        def write_motif(motif_id, motif_name, matrix, outfile, metadata):
            consensus = generate_consensus(matrix)
            outfile.write(f">{consensus}\t{motif_name}\t6.0\t0\t0\n")
            for row in matrix:
                outfile.write("\t".join(f"{p:.6f}" for p in row) + "\n")
            metadata.append({
                "motif_id":  motif_id,
                "tf_name":   motif_name,
                "consensus": consensus,
            })

        for line in infile:
            line = line.strip()
            if not line:
                continue

            if line.startswith("MOTIF"):
                if current_motif_id and matrix:
                    write_motif(current_motif_id, current_motif_name, matrix, outfile, metadata)
                    matrix = []
                parts = line.split()
                current_motif_id = parts[1]
                current_motif_name = parts[2] if len(parts) >= 3 else parts[1]

            elif line.startswith("letter-probability matrix"):
                match = re.search(r"w=\s*(\d+)", line)
                if match:
                    matrix_width = int(match.group(1))
                    matrix_lines_read = 0
                    matrix = []

            elif current_motif_id and matrix_width > 0 and matrix_lines_read < matrix_width:
                probs = [float(p) for p in line.split()]
                if len(probs) == 4:
                    matrix.append(probs)
                    matrix_lines_read += 1

        # Write last motif
        if current_motif_id and matrix:
            write_motif(current_motif_id, current_motif_name, matrix, outfile, metadata)

    return metadata


def main():
    meme_path  = OUTPUT_DIR / "jaspar.meme.motif"
    homer_path = OUTPUT_DIR / "jaspar.homer.motif"
    meta_path  = OUTPUT_DIR / "motif_metadata.csv"

    # Step 1: Download
    download_jaspar(meme_path)

    # Step 2: Convert to HOMER format
    print("Converting MEME → HOMER format...")
    metadata = convert_meme_to_homer(meme_path, homer_path)
    print(f"  Converted {len(metadata)} motifs → {homer_path}")

    # Step 3: Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(meta_path, index=False)
    print(f"  Metadata saved to: {meta_path}")

    # Summary
    print(f"\nDone!")
    print(f"  MEME file:     {meme_path}  ({meme_path.stat().st_size / 1024:.1f} KB)")
    print(f"  HOMER file:    {homer_path}  ({homer_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Motifs:        {len(metadata)}")
    print(f"\nSample motifs:")
    print(meta_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
