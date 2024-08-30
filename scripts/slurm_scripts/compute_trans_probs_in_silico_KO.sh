#!/bin/bash

#SBATCH --job-name=in_silico_KO      # Job name
#SBATCH --partition=cpu                 # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_insilico_KO_%j.out # STDOUT file
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_insilico_KO_%j.err  # STDERR file
#SBATCH --time=24:00:00                 # Runtime in HH:MM:SS
#SBATCH --mem=32G                       # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL            # Type of email notification
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email for notifications

# Usage check
if [ "$#" -lt 5 ]; then
    echo "Usage: sbatch $0 adata_path oracle_path data_id figpath list_KO_genes"
    exit 1
fi

# Assign command-line arguments to variables
adata_path=$1
# metacell_path=$2
oracle_path=$2
data_id=$3
figpath=$4
list_KO_genes=$5

# Create the output directory if it doesn't exist
mkdir -p "$figpath"

# Load necessary modules
module load anaconda
module load R/4.3
conda activate celloracle_env

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/

# Run the Python script with command-line arguments 
python run_07_co_in_silico_KO_systematic.py \
    --adata_path "$adata_path" \
    --oracle_path "$oracle_path" \
    --data_id "$data_id" \
    --figpath "$figpath" \
    --list_KO_genes "$list_KO_genes"

# Deactivate the conda environment
conda deactivate