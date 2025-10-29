#!/bin/bash

# Submit script for collecting Cicero results from individual chromosome jobs
# Usage: ./submit_collect_results.sh

# Default parameters - MODIFY THESE TO MATCH YOUR ANALYSIS
SEURAT_OBJECT_PATH="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC.rds"
ASSAY="peaks_integrated"
OUTPUT_PATH="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/"
DATA_ID="integrated_ATAC_v2"
PEAKTYPE="peaks_integrated"
SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_slurm_outputs/cicero_integrated_peaks_v2/"

# Script paths - collect_results.R should be in the script directory
COLLECT_R_SCRIPT="${SCRIPT_DIR}/collect_results.R"
CURRENT_DIR="$(pwd)"

# Print usage if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [seurat_object_path] [assay] [output_path] [data_id] [peaktype] [script_dir]"
    echo ""
    echo "Default parameters:"
    echo "  seurat_object_path: $SEURAT_OBJECT_PATH"
    echo "  assay: $ASSAY"
    echo "  output_path: $OUTPUT_PATH"
    echo "  data_id: $DATA_ID"
    echo "  peaktype: $PEAKTYPE"
    echo "  script_dir: $SCRIPT_DIR"
    echo ""
    echo "Or modify the default parameters in this script."
    exit 0
fi

# Override defaults with command line arguments if provided
if [ $# -ge 1 ]; then SEURAT_OBJECT_PATH="$1"; fi
if [ $# -ge 2 ]; then ASSAY="$2"; fi
if [ $# -ge 3 ]; then OUTPUT_PATH="$3"; fi
if [ $# -ge 4 ]; then DATA_ID="$4"; fi
if [ $# -ge 5 ]; then PEAKTYPE="$5"; fi
if [ $# -ge 6 ]; then SCRIPT_DIR="$6"; fi

# Update collect script path based on final script_dir
COLLECT_R_SCRIPT="${SCRIPT_DIR}/collect_results.R"

echo "=== CICERO RESULTS COLLECTION SUBMISSION ==="
echo "Submission started at: $(date)"
echo ""

echo "=== PARAMETERS ==="
echo "Seurat object: $SEURAT_OBJECT_PATH"
echo "Assay: $ASSAY"
echo "Output path: $OUTPUT_PATH"
echo "Data ID: $DATA_ID"
echo "Peak type: $PEAKTYPE"
echo "Script directory: $SCRIPT_DIR"
echo "Collection R script: $COLLECT_R_SCRIPT"
echo ""

# Validate inputs
echo "=== VALIDATING INPUTS ==="
if [ ! -f "$SEURAT_OBJECT_PATH" ]; then
    echo "ERROR: Seurat object file not found: $SEURAT_OBJECT_PATH"
    exit 1
fi

if [ ! -f "$COLLECT_R_SCRIPT" ]; then
    echo "ERROR: Collection R script not found: $COLLECT_R_SCRIPT"
    echo "Copying collect_results.R to script directory..."
    
    # Try to copy from the same directory as this script
    SCRIPT_SOURCE="$(dirname "$0")/collect_results.R"
    if [ -f "$SCRIPT_SOURCE" ]; then
        cp "$SCRIPT_SOURCE" "$COLLECT_R_SCRIPT"
        echo "Successfully copied collect_results.R to $COLLECT_R_SCRIPT"
    else
        echo "ERROR: Could not find collect_results.R in $(dirname "$0")"
        echo "Please ensure collect_results.R is in the same directory as this script or in $SCRIPT_DIR"
        exit 1
    fi
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
    size=$(du -h "$file" 2>/dev/null | cut -f1)
    echo "  - $filename ($size)"
done
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Create the SLURM submission script
SLURM_SCRIPT="$(mktemp).sh"
cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=cicero_collect_results
#SBATCH --output=${SCRIPT_DIR}/cicero_collect_results_%j.out
#SBATCH --error=${SCRIPT_DIR}/cicero_collect_results_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

# Load required modules
module load R/4.3

# Print job information
echo "=== CICERO COLLECTION JOB STARTED ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job started at: \$(date)"
echo "Running on node: \$SLURM_NODELIST"
echo "Available memory: \$SLURM_MEM_PER_NODE MB"
echo "Working directory: \$(pwd)"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"
echo "Changed to script directory: \$(pwd)"
echo ""

# Run the collection R script
echo "Running collection R script..."
Rscript "$COLLECT_R_SCRIPT" \\
    "$SEURAT_OBJECT_PATH" \\
    "$ASSAY" \\
    "$OUTPUT_PATH" \\
    "$DATA_ID" \\
    "$PEAKTYPE" \\
    "$SCRIPT_DIR"

# Check if the R script completed successfully
if [ \$? -eq 0 ]; then
    echo ""
    echo "=== CICERO COLLECTION COMPLETED SUCCESSFULLY ==="
    echo "Job completed at: \$(date)"
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
    
    for file in "\$PEAKS_FILE" "\$CONNECTIONS_FILE" "\$STATS_FILE"; do
        if [ -f "\$file" ]; then
            size=\$(du -h "\$file" | cut -f1)
            lines=\$(wc -l < "\$file")
            echo "  ✓ \$(basename "\$file") - \$size (\$lines lines)"
        else
            echo "  ✗ \$(basename "\$file") - NOT FOUND"
        fi
    done
    
    echo ""
    echo "Analysis complete! Check the output files above."
    
else
    echo ""
    echo "=== ERROR: CICERO COLLECTION FAILED ==="
    echo "Job failed at: \$(date)"
    echo "Check the error log for details: ${SCRIPT_DIR}/cicero_collect_results_\${SLURM_JOB_ID}.err"
    echo "Check the R script output above for specific error messages."
    
    # Show some debugging information
    echo ""
    echo "=== DEBUGGING INFORMATION ==="
    echo "Available model files:"
    ls -la "$SCRIPT_DIR"/cicero_model_*.rds 2>/dev/null || echo "No model files found"
    echo ""
    echo "Available memory at failure:"
    free -h
    
    exit 1
fi
EOF

# Submit the job
echo "=== SUBMITTING JOB ==="
echo "Submitting SLURM job..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | sed 's/Submitted batch job //')

if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor progress with:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f ${SCRIPT_DIR}/cicero_collect_results_${JOB_ID}.out"
    echo ""
    echo "Log files will be created in:"
    echo "  Output: ${SCRIPT_DIR}/cicero_collect_results_${JOB_ID}.out"
    echo "  Error: ${SCRIPT_DIR}/cicero_collect_results_${JOB_ID}.err"
    echo ""
    echo "Expected output files:"
    echo "  1. ${OUTPUT_PATH}01_${DATA_ID}_${PEAKTYPE}_peaks.csv"
    echo "  2. ${OUTPUT_PATH}02_${DATA_ID}_cicero_connections_${PEAKTYPE}_peaks.csv"
    echo "  3. ${OUTPUT_PATH}03_${DATA_ID}_cicero_stats_${PEAKTYPE}_peaks.csv"
else
    echo "ERROR: Failed to submit job"
    exit 1
fi

# Clean up temporary file
rm -f "$SLURM_SCRIPT"

echo ""
echo "Submission completed at: $(date)" 