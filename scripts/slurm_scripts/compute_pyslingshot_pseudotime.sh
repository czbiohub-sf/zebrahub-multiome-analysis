#!/bin/bash

#SBATCH --job-name=pyslingshot_pseudotime      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/pyslingshot_pseudotime_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/pyslingshot_pseudotime_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=24:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=2G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=1                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Usage check
if [ "$#" -ne 6 ]; then
    echo "Usage: sbatch $0 filepath data_id annotation progenitor_cluster embedding_key figpath"
    exit 1
fi

# Input Arguments:
# 1) filepath: AnnData object with the preprocessed data.
# 2) data_id: data identifier for the output files.
# 3) annotation: annotation class for celltypes (clusters)
# 4) progenitor_cluster: progenitor cluster to be used as the root for the pseudotime calculation.
# 5) embedding_key: key for the embedding to be used for the pseudotime calculation.
# 6) figpath: path to save the figures/plots for diagnostics purpose.

# Assign command-line arguments to variables
# Define arguments for the first script
filepath=$1 # input filepath
data_id=$2 # data_id
annotation=$3 # annotation class (cell-type annotation)
progenitor_cluster=$4 # progenitor cluster
embedding_key=$5 # embedding key
figpath=$6 # output figure path

# # Create the figure directory if it doesn't exist
# mkdir -p "$figpath"

# Load necessary modules
module load anaconda
module load data.science
module load R/4.3
conda activate single-cell-basics

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/

# Run the Python script with command-line arguments
python run_06_pyslingshot_compute_pseudotime.py $filepath $data_id $annotation $progenitor_cluster $embedding_key $figpath

# Deactivate the conda environment
conda deactivate


