#!/bin/bash
#SBATCH --job-name=parts_05b_v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/05b_detail_v2_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/05b_detail_v2_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
source /hpc/user_apps/data.science/data.science.bashrc 2>/dev/null || true
module load anaconda 2>/dev/null || true; module load data.science 2>/dev/null || true
echo "=== Script 05b: Celltype Detail Report V2 ==="; echo "Start: $(date)"; echo "Host: $(hostname)"
conda run --no-capture-output -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "${SCRIPT_DIR}/05b_celltype_detail_report_v2.py"
echo "Exit code: $?"; echo "End: $(date)"
