#!/bin/bash
#SBATCH --job-name=cicero_analysis
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_shell_scripts/cicero_analysis_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_shell_scripts/cicero_analysis_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=1250G
#SBATCH --cpus-per-task=25
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

# Usage check
if [ "$#" -ne 7 ]; then
    echo "Usage: sbatch $0 SEURAT_OBJ_PATH ASSAY DIM_REDUCED OUTPUT_PATH DATA_ID PEAKTYPE SHELL_SCRIPT_DIR"
    exit 1
fi

# Assign command-line arguments to variables
SEURAT_OBJ_PATH=$1
ASSAY=$2
DIM_REDUCED=$3
OUTPUT_PATH=$4
DATA_ID=$5
PEAKTYPE=$6
SHELL_SCRIPT_DIR=$7

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"
mkdir -p "$SHELL_SCRIPT_DIR"

# Load R module (if needed)
module load R/4.3

# Run the R script
Rscript /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/run_02_compute_CCANs_cicero_parallelized.R $SEURAT_OBJ_PATH $ASSAY $DIM_REDUCED $OUTPUT_PATH $DATA_ID $PEAKTYPE $SHELL_SCRIPT_DIR