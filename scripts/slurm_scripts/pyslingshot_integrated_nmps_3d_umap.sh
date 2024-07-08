#!/bin/bash

#SBATCH --job-name=compute_pseudotime_pyslingshot      # Job name
#SBATCH --partition=cpu                     # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/pyslingshot_3d_umap_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/pyslingshot_3d_umap_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=48:00:00                     # Runtime in HH:MM:SS            
#SBATCH --mem=800G                          # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Create the output directory if it doesn't exist
# mkdir -p "$output_filepath"

# Load necessary modules
module load anaconda
module load R/4.3
module load data.science
conda activate single-cell-base

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/utils/

# Run the Python script with command-line arguments
python pyslingshot_integrated_nmp_subsets_3d_umap.py

# Deactivate the conda environment
conda deactivate