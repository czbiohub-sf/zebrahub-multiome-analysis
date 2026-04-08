#!/bin/bash
#SBATCH --job-name=parts_09b_logos
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09b_logos_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09b_logos_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
GRELU_PYTHON="/home/yang-joon.kim/.conda/envs/gReLu/bin/python"

echo "=== Script 09b: TF sequence logos ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${GRELU_PYTHON} -u "${SCRIPT_DIR}/09b_generate_TF_logos.py"
echo "Exit code: $?"

echo "End: $(date)"
