#!/bin/bash
#SBATCH --job-name=cicero_assembly_individual_chroms
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_slurm_outputs/cicero_assembly_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_slurm_outputs/cicero_assembly_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

# Usage check
if [ "$#" -ne 4 ]; then
    echo "Usage: sbatch $0 OUTPUT_PATH DATA_ID PEAKTYPE CICERO_MODEL_DIR"
    exit 1
fi

# Assign command-line arguments to variables
OUTPUT_PATH=$1
DATA_ID=$2
PEAKTYPE=$3
CICERO_MODEL_DIR=$4

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"
mkdir -p "$CICERO_MODEL_DIR"

# Load R module (if needed)
module load R/4.3

# Run the R script
Rscript /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/run_cicero_join_chroms.R $OUTPUT_PATH $DATA_ID $PEAKTYPE $CICERO_MODEL_DIR