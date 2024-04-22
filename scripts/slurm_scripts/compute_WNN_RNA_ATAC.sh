#!/bin/bash

#SBATCH --job-name=compute_WNN_RNA_ATAC      # Job name
#SBATCH --partition=cpu,gpu                     # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/compute_WNN_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/compute_WNN_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=48:00:00                     # Runtime in HH:MM:SS            
#SBATCH --mem=256G                          # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Create the output directory if it doesn't exist
# mkdir -p "$output_filepath"

# Load necessary modules
module load R/4.3

# Navigate to the directory containing the R script
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/R_scripts/

# Run the R script
Rscript compute_WNN_embeddings_RNA_ATAC.R