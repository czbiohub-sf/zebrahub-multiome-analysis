#!/bin/bash

#SBATCH --job-name=gimme_motifs_de_novo      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/de_novo_motifs_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/de_novo_motifs_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=48:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=256G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=12                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Load necessary modules
module load anaconda
module load R/4.3
conda activate celloracle_env

# Paths to input and output
peaks_fasta="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/12_peak_umap_motif_analysis/peaks_integrated.fasta"
output_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/12_peak_umap_motif_analysis/gimmemotifs_output/"
# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# GimmeMotifs command to find the de novo motifs
gimme motifs "$peaks_fasta" "$output_dir" --denovo -g /hpc/mydata/yang-joon.kim/genomes/danRer11/danRer11.fa -N 12

# Deactivate the conda environment
conda deactivate