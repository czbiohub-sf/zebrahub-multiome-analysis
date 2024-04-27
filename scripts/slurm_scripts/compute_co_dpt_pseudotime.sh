#!/bin/bash

#SBATCH --job-name=compute_dpt_pseudotime      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_dpt_pseudotime_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_dpt_pseudotime_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=24:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=8G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=2                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Usage check
if [ "$#" -ne 4 ]; then
    echo "Usage: sbatch $0 input_path data_id dim_reduce figpath"
    exit 1
fi
# Input Arguments:
# 1) output_path: filepath for the output
# 2) RNAdata_path: filepath for the adata (RNA)
# 3) baseGRN_path: filepath for the base GRN (parquet)
# 4) data_id: data_id
# 5) annotation: annotation class (cell-type annotation)
# 6) dim_reduce: dimensionality reduction embedding name

# Assign command-line arguments to variables
# Define arguments for the first script
input_path=$1 # output filepath
data_id=$2 # data_id
dim_reduce=$3 # Let's just use one dimensionality reduction embedding name. (i.e. "umap.joint")
figpath=$4 # output figure path

# Create the output directory if it doesn't exist
# mkdir -p "$input_path"

# Load necessary modules
module load anaconda
module load R/4.3
conda activate celloracle_env

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/


# Run the Python script with command-line arguments
python run_06_co_compute_pseudotime.py $input_path $data_id $dim_reduce $figpath

# Deactivate the conda environment
conda deactivate


