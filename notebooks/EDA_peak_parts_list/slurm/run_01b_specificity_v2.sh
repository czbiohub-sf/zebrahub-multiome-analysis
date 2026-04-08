#!/bin/bash
#SBATCH --job-name=parts_list_01b
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/01b_specificity_v2_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/01b_specificity_v2_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"

source /hpc/user_apps/data.science/data.science.bashrc 2>/dev/null || true
module load anaconda 2>/dev/null || true
module load data.science 2>/dev/null || true

echo "=== Script 01b: Specificity Matrix V2 (shrinkage-regularized) ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

conda run --no-capture-output \
    -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "${SCRIPT_DIR}/01b_compute_specificity_matrix_v2.py"

echo "Exit code: $?"; echo "End: $(date)"
