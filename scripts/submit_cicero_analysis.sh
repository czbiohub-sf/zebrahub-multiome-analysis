#!/bin/bash
#SBATCH --job-name=cicero_main_analysis
#SBATCH --output=cicero_main_analysis_%j.out
#SBATCH --error=cicero_main_analysis_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

# Load required modules
module load R/4.3

# Set script directory (adjust path as needed)
SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts"

# Default parameters - MODIFY THESE AS NEEDED
SEURAT_OBJECT_PATH="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/your_seurat_object.rds"
ASSAY="ATAC"
DIM_REDUCED="UMAP.ATAC"
OUTPUT_PATH="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/"
DATA_ID="TDR118"
PEAKTYPE="peaks_merged"
SHELL_SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_slurm_outputs/"

# Print job information
echo "=== CICERO ANALYSIS JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo ""

echo "=== PARAMETERS ==="
echo "Seurat object: $SEURAT_OBJECT_PATH"
echo "Assay: $ASSAY"
echo "Dimensionality reduction: $DIM_REDUCED"
echo "Output path: $OUTPUT_PATH"
echo "Data ID: $DATA_ID"
echo "Peak type: $PEAKTYPE"
echo "Shell script directory: $SHELL_SCRIPT_DIR"
echo ""

# Check if input file exists
if [ ! -f "$SEURAT_OBJECT_PATH" ]; then
    echo "ERROR: Seurat object file not found: $SEURAT_OBJECT_PATH"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p "$OUTPUT_PATH"
mkdir -p "$SHELL_SCRIPT_DIR"

echo "=== STARTING CICERO ANALYSIS ==="
echo "Running R script..."

# Run the R script with parameters
Rscript "$SCRIPT_DIR/run_02_compute_CCANs_cicero_parallelized_improved.R" \
    "$SEURAT_OBJECT_PATH" \
    "$ASSAY" \
    "$DIM_REDUCED" \
    "$OUTPUT_PATH" \
    "$DATA_ID" \
    "$PEAKTYPE" \
    "$SHELL_SCRIPT_DIR"

# Check if the R script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== CICERO ANALYSIS COMPLETED SUCCESSFULLY ==="
    echo "Job completed at: $(date)"
    echo "Check the following locations for results:"
    echo "  - Main output: $OUTPUT_PATH"
    echo "  - Job logs: $SHELL_SCRIPT_DIR/cicero_integrated_peaks/"
    echo ""
    echo "Monitor chromosome jobs with:"
    echo "  squeue -u $USER"
    echo "  tail -f $SHELL_SCRIPT_DIR/cicero_integrated_peaks/collect_results_*.out"
else
    echo ""
    echo "=== ERROR: CICERO ANALYSIS FAILED ==="
    echo "Job failed at: $(date)"
    echo "Check the error log for details: cicero_main_analysis_${SLURM_JOB_ID}.err"
    exit 1
fi 