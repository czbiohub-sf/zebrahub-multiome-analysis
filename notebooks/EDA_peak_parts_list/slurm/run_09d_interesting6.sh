#!/bin/bash
#SBATCH --job-name=parts_09d_i6
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09d_interesting6_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09d_interesting6_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
SCB_PYTHON="/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python"

echo "=== Script 09d: Interesting-6 UMAP panels + standalone legend ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${SCB_PYTHON} -u "${SCRIPT_DIR}/09d_umap_interesting6.py"
echo "09d_umap_interesting6 exit: $?"

${SCB_PYTHON} -u "${SCRIPT_DIR}/09d_umap_legend.py"
echo "09d_umap_legend exit: $?"

echo "End: $(date)"
