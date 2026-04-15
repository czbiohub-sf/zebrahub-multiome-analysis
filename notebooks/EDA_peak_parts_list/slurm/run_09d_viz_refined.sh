#!/bin/bash
#SBATCH --job-name=parts_09d_viz
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09d_viz_refined_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09d_viz_refined_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
SCB_PYTHON="/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python"

echo "=== Script 09d: Advanced V3 Visualizations (refined) ==="
echo "Start: $(date)"; echo "Host: $(hostname)"
echo "Python: ${SCB_PYTHON}"

${SCB_PYTHON} -u "${SCRIPT_DIR}/09d_advanced_visualizations.py"
echo "09d_advanced_visualizations exit: $?"

echo "--- Running 09d_umap_nolabel (focal 7 celltypes, no-label variants) ---"
${SCB_PYTHON} -u "${SCRIPT_DIR}/09d_umap_nolabel.py"
echo "09d_umap_nolabel exit: $?"

echo "End: $(date)"
