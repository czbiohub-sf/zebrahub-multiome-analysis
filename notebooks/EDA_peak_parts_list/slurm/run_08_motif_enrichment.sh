#!/bin/bash
#SBATCH --job-name=parts_list_08
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/08_motif_enrichment_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/08_motif_enrichment_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"

# Use gReLu env (has grelu, pysam, anndata — NOT single-cell-base)
GRELU_PYTHON="/home/yang-joon.kim/.conda/envs/gReLu/bin/python"

echo "=== Script 08: Cross-Celltype JASPAR Motif Enrichment ==="
echo "Start: $(date)"; echo "Host: $(hostname)"
echo "Python: ${GRELU_PYTHON}"

${GRELU_PYTHON} -u "${SCRIPT_DIR}/08_motif_enrichment_celltypes.py"

echo "Exit code: $?"; echo "End: $(date)"
