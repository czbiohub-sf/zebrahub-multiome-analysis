#!/bin/bash
#SBATCH --job-name=mhb_panels
#SBATCH --partition=preempted
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09o_mhb_panels_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09o_mhb_panels_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
SCB_PYTHON="/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python"

echo "=== 09o: Midbrain-hindbrain boundary top-5 combined panels ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${SCB_PYTHON} -u "${SCRIPT_DIR}/09o_mhb_combined_panels.py"
echo "Exit code: $?"
echo "End: $(date)"
