#!/bin/bash

#SBATCH --job-name=compute_corr_RNA_ATAC      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/metacell_rna_atac_corr_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/metacell_rna_atac_corr_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=24:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=128G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Usage check
# Input Arguments:

# Load necessary modules
module load anaconda
# module load R/4.3
conda activate seacells

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/SEACells_metacell/


# Run the Python script with command-line arguments
python compute_corr_RNA_gene_activity_metacells.py

# Deactivate the conda environment
conda deactivate


