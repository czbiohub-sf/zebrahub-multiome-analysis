#!/bin/bash
#SBATCH --job-name=motif_maps_v2
#SBATCH --partition=preempted
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09i_motif_maps_v2_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09i_motif_maps_v2_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
SCB_PYTHON="/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python"

echo "=== 09i-v2: Motif position maps (from precomputed top-200) ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${SCB_PYTHON} -u "${SCRIPT_DIR}/09i_motif_position_maps_v2.py"
echo "Exit code: $?"

echo "End: $(date)"
