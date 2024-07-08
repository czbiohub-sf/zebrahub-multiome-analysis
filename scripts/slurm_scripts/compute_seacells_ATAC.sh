#!/bin/bash

#SBATCH --job-name=seacells_ATAC      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/seacells_ATAC_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/seacells_ATAC_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=24:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=256G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Usage check
if [ "$#" -ne 5 ]; then
    echo "Usage: sbatch $0 input_path output_path data_id annotation figpath "
    exit 1
fi
# 1) input_path: a filepath to an input anndata object with scATAC-seq data (h5ad format)
# 2) output_path: a filepath to save the output
# 3) data_id: name of the output file.
# 4) annotation_class: annotation class for the celltype assignment
# 5) figpath: path for the plots/figures

# Assign command-line arguments to variables
# Define arguments for the first script
input_path=$1 # input filepath
output_path=$2 # output filepath
data_id=$3 # data_id
annotation=$4 # annotation class
figpath=$5 # output figure path

# Create the output directory if it doesn't exist
# mkdir -p "$input_path"

# Load necessary modules
module load anaconda
# module load R/4.3
conda activate seacells

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/SEACells_metacell/

# Run the Python script with command-line arguments
python compute_seacells_atac.py $input_path $output_path $data_id $annotation $figpath

# Deactivate the conda environment
conda deactivate