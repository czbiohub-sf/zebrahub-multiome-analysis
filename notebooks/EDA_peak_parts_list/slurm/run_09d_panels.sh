#!/bin/bash
#SBATCH --job-name=parts_09d_panels
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09d_panels_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09d_panels_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
SCB_PYTHON="/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python"

echo "=== Script 09d-panels: All-celltypes + merged5 + highlight6 UMAPs ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${SCB_PYTHON} -u "${SCRIPT_DIR}/09d_umap_panels.py"

echo "Exit code: $?"; echo "End: $(date)"
