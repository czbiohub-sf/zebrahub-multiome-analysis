#!/bin/bash

#SBATCH --job-name=co_in_silico_KO      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_insilico_KO_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_insilico_KO_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=24:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=32G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# # Usage check
# if [ "$#" -ne 5 ]; then
#     echo "Usage: sbatch $0 oracle_path data_id annotation figpath list_KO_genes [--use_pseudotime true/false] [--pseudotime_path path] [--systematic_KO true/false]"
#     exit 1
# fi
# Usage check
if [ "$#" -lt 5 ]; then
    echo "Usage: sbatch $0 oracle_path data_id annotation figpath list_KO_genes [--use_pseudotime true/false] [--pseudotime_path path] [--systematic_KO true/false]"
    exit 1
fi

# Input Arguments:
# 1) oracle_path: filepath for the output
# 2) data_id: data_id
# 3) annotation: annotation class (cell-type annotation)
# 4) figpath: filepath for the figures
# 5) list_KO_genes: a comma-separated list of KO genes
# 6) (optional) --use_pseudotime: use different pseudotime method other than DPT (true/false)
# 7) (optional) --pseudotime_path: pseudotime df filepath
# 8) (optional) --systematic_KO: perform in silico KO for all TFs/genes (true/false)


# Assign command-line arguments to variables
# Define arguments for the first script
oracle_path=$1 # input/output filepath
data_id=$2 # data_id
annotation=$3 # annotation class
figpath=$4 # output figure path
list_KO_genes=$5 # a comma-separated list of KO genes
shift 5

# Additional optional arguments
use_pseudotime=""
pseudotime_path=""
systematic_KO=""

while (( "$#" )); do
  case "$1" in
    --use_pseudotime)
      use_pseudotime="--use_pseudotime $2"
      shift 2
      ;;
    --pseudotime_path)
      pseudotime_path="--pseudotime_path $2"
      shift 2
      ;;
    --systematic_KO)
      systematic_KO="--systematic_KO $2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create the output directory if it doesn't exist
mkdir -p "$figpath"

# Load necessary modules
module load anaconda
module load R/4.3
conda activate celloracle_env

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/


# Run the Python script with command-line arguments 
python run_07_co_KO_simulation.py $oracle_path $data_id $annotation $figpath $list_KO_genes $use_pseudotime $pseudotime_path $systematic_KO

# Deactivate the conda environment
conda deactivate