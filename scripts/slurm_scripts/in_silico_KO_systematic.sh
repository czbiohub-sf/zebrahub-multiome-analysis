#!/bin/bash

#SBATCH --job-name=in_silico_KO_systematic      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_in_silico_KO_systematic_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_in_silico_KO_systematic_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=24:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=64G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Usage check
if [ "$#" -ne 8 ]; then
    echo "Usage: sbatch $0 oracle_path data_id annotation figpath list_KO_genes use_pseudotime pseudotime_path systematic_KO"
    exit 1
fi
# Input Arguments:
# 1) oracle_path: output filepath
# 2) data_id: data_id
# 3) annotation: annotation class (cell-type annotation)
# 4) figpath: filepath for the figures
# 5) list_KO_genes: list of KO genes
# 6) use_pseudotime: whether to use pseudotime or not
# 7) pseudotime_path: filepath for the pseudotime
# 8) systematic_KO: whether to perform systematic KO or not

# Assign command-line arguments to variables
# Define arguments for the first script
oracle_path=$1 # output filepath
data_id=$2 # data_id
annotation=$3 # Let's just use one annotation_class. (i.e. "global_annotation")
figpath=$4 # filepath for the figures
list_KO_genes=$5 # list of KO genes
use_pseudotime=$6 # whether to use pseudotime or not
pseudotime_path=$7 # filepath for the pseudotime
systematic_KO=$8 # whether to perform systematic KO or not

# Create the figure directory if it doesn't exist
mkdir -p "$figpath"

# Load necessary modules
module load anaconda
module load R/4.3
conda activate celloracle_env

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/


# Run the Python script with command-line arguments
python run_07_co_KO_simulation.py $oracle_path $data_id $annotation $figpath $list_KO_genes --use_pseudotime=$use_pseudotime --pseudotime_path=$pseudotime_path --systematic_KO=$systematic_KO

# Deactivate the conda environment
conda deactivate