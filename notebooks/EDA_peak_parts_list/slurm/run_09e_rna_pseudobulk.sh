#!/bin/bash
#SBATCH --job-name=parts_09e_rna_pb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09e_rna_pseudobulk_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09e_rna_pseudobulk_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
SCB_PYTHON="/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python"

echo "=== Script 09e-pre: RNA pseudobulk pipeline ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${SCB_PYTHON} -u "${SCRIPT_DIR}/09e_rna_pseudobulk.py"
echo "Exit code: $?"

echo "End: $(date)"
