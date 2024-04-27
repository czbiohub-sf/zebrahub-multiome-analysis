#!/bin/bash

#SBATCH --job-name=co_in_silico_KO      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_insilico_KO_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_insilico_KO_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=24:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=16G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=2                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Usage check
if [ "$#" -ne 5 ]; then
    echo "Usage: sbatch $0 oracle_path data_id annotation figpath list_KO_genes"
    exit 1
fi

# Input Arguments:
# 1) oracle_path: filepath for the output
# 2) data_id: data_id
# 3) annotation: annotation class (cell-type annotation)
# 4) figpath: filepath for the figures
# 5) list_KO_genes: a comma-separated list of KO genes

# Assign command-line arguments to variables
# Define arguments for the first script
oracle_path=$1 # input/output filepath
data_id=$2 # data_id
annotation=$3 # annotation class
figpath=$4 # output figure path
list_KO_genes=$5 # a comma-separated list of KO genes

# Create the output directory if it doesn't exist
# mkdir -p "$input_path"

# Load necessary modules
module load anaconda
module load R/4.3
conda activate celloracle_env

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/


# Run the Python script with command-line arguments
python run_07_co_KO_simulation.py $oracle_path $data_id $annotation $figpath $list_KO_genes

# Deactivate the conda environment
conda deactivate