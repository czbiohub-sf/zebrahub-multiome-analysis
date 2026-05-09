#!/bin/bash
#SBATCH --job-name=parts_list_09
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09_celltype_level_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09_celltype_level_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"

# Use gReLu env (has tangermeme, pysam, anndata, torch, scipy, statsmodels)
GRELU_PYTHON="/home/yang-joon.kim/.conda/envs/gReLu/bin/python"

echo "=== Script 09: V3 Celltype-Level Parts List ==="
echo "Start: $(date)"; echo "Host: $(hostname)"
echo "Python: ${GRELU_PYTHON}"

${GRELU_PYTHON} -u "${SCRIPT_DIR}/09_celltype_level_parts_list.py"

echo "Exit code: $?"; echo "End: $(date)"
