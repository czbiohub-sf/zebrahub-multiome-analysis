# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: celloracle_env
#     language: python
#     name: celloracle_env
# ---

# %% [markdown]
# ## Use GimmeMotifs to find de novo motifs from scATAC-seq datasets
#
# - last updated: 1/21/2025
#

# %%
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the custom module
import sys
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs")

from atac_seq_motif_analysis import ATACSeqMotifAnalysis

# %%
# Initialize the analysis class
genomes_dir = "/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/fasta/"  # Replace with your genome data directory
ref_genome = "" # "danRer11" for zebrafish, GRCz11

atac_analysis = ATACSeqMotifAnalysis(ref_genome, genomes_dir)

# %%
# import the peak data
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated.h5ad")
adata_peaks

# %%
adata_peaks.var_names

# %%
# Example list of peaks
peaks = ["1-32-526", "25-37492035-37493157"]

# Step 1: Convert peak strings to DataFrame
peaks_df = atac_analysis.list_peakstr_to_df(peaks)
print("Peaks DataFrame:")
print(peaks_df)



# %%
# let's take all the peaks
peaks = adata_peaks.var_names.to_list()

# Step 1: Convert peak strings to DataFrame
peaks_df = atac_analysis.list_peakstr_to_df(peaks)
print("Peaks DataFrame:")
print(peaks_df.head())


# %%
def check_peak_format(self, peaks_df):
    """
    Validate peak format and filter invalid peaks based on genome information.

    Args:
        peaks_df (pd.DataFrame): DataFrame with columns ["chr", "start", "end"].

    Returns:
        pd.DataFrame: Filtered DataFrame with valid peaks.
    """
    n_peaks_before = peaks_df.shape[0]
    all_chr_list = list(self.genome_data.keys())

    # Check chromosome names and peak lengths
    lengths = np.abs(peaks_df["end"] - peaks_df["start"])
    n_threshold = 5
    valid_peaks = peaks_df[(lengths >= n_threshold) & peaks_df["chr"].isin(all_chr_list)]

    # Check if peaks exceed chromosome lengths
    for idx, row in peaks_df.iterrows():
        chr_length = len(self.genome_data[row["chr"]])  # Fixed here
        if row["end"] > chr_length:
            valid_peaks = valid_peaks.drop(idx)

    # Print summary
    n_invalid_length = len(lengths[lengths < n_threshold])
    n_invalid_chr = n_peaks_before - peaks_df["chr"].isin(all_chr_list).sum()
    n_invalid_end = n_peaks_before - valid_peaks.shape[0]
    print(f"Peaks before filtering: {n_peaks_before}")
    print(f"Invalid chromosome names: {n_invalid_chr}")
    print(f"Invalid lengths (< {n_threshold} bp): {n_invalid_length}")
    print(f"Peaks exceeding chromosome lengths: {n_invalid_end}")
    print(f"Peaks after filtering: {valid_peaks.shape[0]}")

    return valid_peaks


# %%
# Step 2: Validate peaks
valid_peaks = check_peak_format(atac_analysis, peaks_df)
print("\nValid Peaks DataFrame:")
print(valid_peaks)

# %%
# Step 3: Convert valid peaks to FASTA
fasta = atac_analysis.peak_to_fasta(valid_peaks["chr"] + "-" + valid_peaks["start"].astype(str) + "-" + valid_peaks["end"].astype(str))
fasta

# %%
# Step 4: Remove zero-length sequences
filtered_fasta = atac_analysis.remove_zero_seq(fasta)


# %%
# Step 5: Save FASTA file
output_fasta_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/12_peak_umap_motif_analysis/peaks_integrated.fasta"

# save
with open(output_fasta_path, "w") as f:
    for name, seq in zip(filtered_fasta.ids, filtered_fasta.seqs):
        f.write(f">{name}\n{seq}\n")

# %%

# %% [markdown]
# ## Use GimmeMotifs to find de novo motifs

# %%
# from gimmemotifs.scanner import Scanner
# from gimmemotifs.fasta import Fasta
# from gimmemotifs.confg import DIRECT_NAME, INDIRECT_NAME


# %%
from gimmemotifs.denovo import gimme_motifs
help(gimme_motifs)

# %%
import celloracle as co
from celloracle import motif_analysis as ma
ref_genome = "danRer11"

genome_installation = ma.is_genome_installed(ref_genome=ref_genome,
                                             genomes_dir=None)
genome_installation

# %%
ma.SUPPORTED_REF_GENOME

# %%
from gimmemotifs.config import MotifConfig

config = MotifConfig()
config.set_motif_dir("/hpc/mydata/yang-joon.kim/genomes/danRer11/")
config.set_bg_dir("/hpc/mydata/yang-joon.kim/genomes/danRer11/")
config.write(open(config.user_config, "w"))

print(f"Motif database directory: {config.get_motif_dir()}")
print(f"Background directory: {config.get_bg_dir()}")

# %%
import os
# Add the directory containing bedtools to PATH
os.environ["PATH"] += ":/home/yang-joon.kim/.conda/envs/celloracle_env/bin"
# Verify if BEDTools is now accessible
bedtools_path = os.popen("which bedtools").read().strip()
if bedtools_path:
    print(f"BEDTools found at: {bedtools_path}")
else:
    print("BEDTools is still not in PATH. Double-check the installation.")

# %%
import subprocess

try:
    subprocess.run(['bedtools', '--version'], check=True, capture_output=True)
    print("BEDTools is properly installed")
except subprocess.CalledProcessError as e:
    print(f"Error running bedtools: {e}")
except FileNotFoundError:
    print("BEDTools is not found in PATH")

# %%
# from gimmemotifs.denovo import gimme_motifs

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

# %%

# %%

# %% [markdown]
# ## Testing the gimmemotifs package with the tutorial/sample dataset
#

# %%
# Or from a consensus sequence
from gimmemotifs.motif import motif_from_consensus
ap1 = motif_from_consensus("TGASTCA")
print(ap1.to_ppm())

# %%
# MEME
print(ap1.to_meme())

# %%

# %%

# %%
