# A script for computing de novo motifs using gimmemotifs
# this script is submitted as a slurm job
# reference: https://gimmemotifs.readthedocs.io/en/latest/
# reference: https://github.com/morris-lab/CellOracle/blob/master/celloracle/motif_analysis/process_bed_file.py

# conda env: celloracle_env

# import scanpy as sc
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import os
from gimmemotifs.denovo import gimme_motifs
import celloracle as co
from celloracle import motif_analysis as ma

# Import the custom module
import sys
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs")
from atac_seq_motif_analysis import ATACSeqMotifAnalysis

# Initialize the analysis class
# genomes_dir = "/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/fasta/"  # Replace with your genome data directory
# ref_genome = "" # "danRer11" for zebrafish, GRCz11
# atac_analysis = ATACSeqMotifAnalysis(ref_genome, genomes_dir)

# # import the peak data
# adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated.h5ad")
# # let's take all the peaks
# peaks = adata_peaks.var_names.to_list()

# # Step 1: Convert peak strings to DataFrame
# peaks_df = atac_analysis.list_peakstr_to_df(peaks)
# print("Peaks DataFrame:")
# print(peaks_df.head())

# # Step 2: Validate peaks
# valid_peaks = atac_analysis.check_peak_format(peaks_df)
# print("\nValid Peaks DataFrame:")
# print(valid_peaks)

# # Step 3: Convert valid peaks to FASTA
# fasta = atac_analysis.peak_to_fasta(valid_peaks["chr"] + "-" + valid_peaks["start"].astype(str) + "-" + valid_peaks["end"].astype(str))

# # Step 4: Remove zero-length sequences
# filtered_fasta = atac_analysis.remove_zero_seq(fasta)

# # Step 5: Save FASTA file
# output_fasta_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/12_peak_umap_motif_analysis/peaks_integrated.fasta"

# with open(output_fasta_path, "w") as f:
#     for name, seq in zip(filtered_fasta.ids, filtered_fasta.seqs):
#         f.write(f">{name}\n{seq}\n")

# Part 2. Run gimmemotifs
ref_genome = "danRer11"
genome_installation = ma.is_genome_installed(ref_genome=ref_genome,
                                             genomes_dir=None)
print(genome_installation)

from gimmemotifs.config import MotifConfig

config = MotifConfig()
config.set_motif_dir("/hpc/mydata/yang-joon.kim/genomes/danRer11/")
config.set_bg_dir("/hpc/mydata/yang-joon.kim/genomes/danRer11/")
config.write(open(config.user_config, "w"))

print(f"Motif database directory: {config.get_motif_dir()}")
print(f"Background directory: {config.get_bg_dir()}")


# Add the directory containing bedtools to PATH
os.environ["PATH"] += ":/home/yang-joon.kim/.conda/envs/celloracle_env/bin"
# Verify if BEDTools is now accessible
bedtools_path = os.popen("which bedtools").read().strip()
if bedtools_path:
    print(f"BEDTools found at: {bedtools_path}")
else:
    print("BEDTools is still not in PATH. Double-check the installation.")

# check if bedtools is properly installed
import subprocess
try:
    subprocess.run(['bedtools', '--version'], check=True, capture_output=True)
    print("BEDTools is properly installed")
except subprocess.CalledProcessError as e:
    print(f"Error running bedtools: {e}")
except FileNotFoundError:
    print("BEDTools is not found in PATH")

# Paths to input and output
# peaks_fasta = "/hpc/mydata/yang-joon.kim/genomes/danRer11/danRer11.fa"
# output_dir = "/path/to/output_dir"
peaks_fasta = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/12_peak_umap_motif_analysis/peaks_integrated.fasta"
output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/12_peak_umap_motif_analysis/"

# Parameters
params = {
    "genome": "danRer11",  # Specify the genome name
    "tools": "Homer,BioProspector,MEME",
    "size": 200,
}

# Run GimmeMotifs
motifs = gimme_motifs(peaks_fasta, output_dir, params=params)
print("GimmeMotifs completed successfully.")