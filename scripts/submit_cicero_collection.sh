#!/bin/bash
#SBATCH --job-name=cicero_collect_results
#SBATCH --output=cicero_collect_results_%j.out
#SBATCH --error=cicero_collect_results_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

# Load required modules
module load R/4.3

# Parameters - MODIFY THESE TO MATCH YOUR PREVIOUS RUN
SEURAT_OBJECT_PATH="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC.rds"
ASSAY="peaks_integrated"
OUTPUT_PATH="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/"
DATA_ID="integrated_ATAC_v2"
PEAKTYPE="peaks_integrated"
SHELL_SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_slurm_outputs/cicero_integrated_peaks_v2/"

# Derived paths
SCRIPT_DIR="${SHELL_SCRIPT_DIR}/cicero_integrated_peaks"
COLLECTION_R_SCRIPT="${SCRIPT_DIR}/collect_results.R"

# Print job information
echo "=== CICERO COLLECTION JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Available memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo ""

echo "=== PARAMETERS ==="
echo "Seurat object: $SEURAT_OBJECT_PATH"
echo "Assay: $ASSAY"
echo "Output path: $OUTPUT_PATH"
echo "Data ID: $DATA_ID"
echo "Peak type: $PEAKTYPE"
echo "Script directory: $SCRIPT_DIR"
echo "Collection R script: $COLLECTION_R_SCRIPT"
echo ""

# Check if required files exist
echo "=== CHECKING PREREQUISITES ==="
if [ ! -f "$SEURAT_OBJECT_PATH" ]; then
    echo "ERROR: Seurat object file not found: $SEURAT_OBJECT_PATH"
    exit 1
fi

if [ ! -f "$COLLECTION_R_SCRIPT" ]; then
    echo "ERROR: Collection R script not found: $COLLECTION_R_SCRIPT"
    echo "This suggests the main cicero analysis hasn't been run yet or failed."
    echo "Expected location: $COLLECTION_R_SCRIPT"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR" ]; then
    echo "ERROR: Script directory not found: $SCRIPT_DIR"
    echo "This suggests the main cicero analysis hasn't been run yet."
    exit 1
fi

# Check for chromosome model files
echo "Checking for chromosome model files..."
MODEL_FILES=($(find "$SCRIPT_DIR" -name "cicero_model_*.rds" 2>/dev/null))
if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "ERROR: No cicero model files found in $SCRIPT_DIR"
    echo "Individual chromosome jobs may not have completed successfully."
    echo "Expected files: cicero_model_1.rds, cicero_model_2.rds, etc."
    exit 1
fi

echo "Found ${#MODEL_FILES[@]} chromosome model files:"
for file in "${MODEL_FILES[@]}"; do
    filename=$(basename "$file")
    size=$(du -h "$file" | cut -f1)
    echo "  - $filename ($size)"
done
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

echo "=== STARTING CICERO COLLECTION ==="
echo "Running collection R script..."
echo "This may take a while depending on the number of connections..."
echo ""

# Run the collection R script
cd "$SCRIPT_DIR"
Rscript "$COLLECTION_R_SCRIPT"

# Check if the R script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== CICERO COLLECTION COMPLETED SUCCESSFULLY ==="
    echo "Job completed at: $(date)"
    echo ""
    echo "Expected output files:"
    echo "  1. Peaks: ${OUTPUT_PATH}01_${DATA_ID}_${PEAKTYPE}_peaks.csv"
    echo "  2. Connections: ${OUTPUT_PATH}02_${DATA_ID}_cicero_connections_${PEAKTYPE}_peaks.csv"
    echo "  3. Statistics: ${OUTPUT_PATH}03_${DATA_ID}_cicero_stats_${PEAKTYPE}_peaks.csv"
    echo ""
    echo "Checking output files..."
    
    # Check if output files were created
    PEAKS_FILE="${OUTPUT_PATH}01_${DATA_ID}_${PEAKTYPE}_peaks.csv"
    CONNECTIONS_FILE="${OUTPUT_PATH}02_${DATA_ID}_cicero_connections_${PEAKTYPE}_peaks.csv"
    STATS_FILE="${OUTPUT_PATH}03_${DATA_ID}_cicero_stats_${PEAKTYPE}_peaks.csv"
    
    for file in "$PEAKS_FILE" "$CONNECTIONS_FILE" "$STATS_FILE"; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            lines=$(wc -l < "$file")
            echo "  ✓ $(basename "$file") - $size ($lines lines)"
        else
            echo "  ✗ $(basename "$file") - NOT FOUND"
        fi
    done
    
    echo ""
    echo "Analysis complete! Check the output files above."
    
else
    echo ""
    echo "=== ERROR: CICERO COLLECTION FAILED ==="
    echo "Job failed at: $(date)"
    echo "Check the error log for details: cicero_collect_results_${SLURM_JOB_ID}.err"
    echo "Check the R script output above for specific error messages."
    
    # Show some debugging information
    echo ""
    echo "=== DEBUGGING INFORMATION ==="
    echo "Available model files:"
    ls -la "$SCRIPT_DIR"/cicero_model_*.rds 2>/dev/null || echo "No model files found"
    echo ""
    echo "Script directory contents:"
    ls -la "$SCRIPT_DIR" 2>/dev/null || echo "Directory not accessible"
    echo ""
    echo "Available memory at failure:"
    free -h
    
    exit 1
fi 