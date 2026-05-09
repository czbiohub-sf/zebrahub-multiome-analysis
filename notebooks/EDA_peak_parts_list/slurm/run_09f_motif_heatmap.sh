#!/bin/bash
#SBATCH --job-name=parts_09f_motif
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09f_motif_heatmap_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09f_motif_heatmap_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
GRELU_PYTHON="/home/yang-joon.kim/.conda/envs/gReLu/bin/python"

echo "=== Script 09f: Interesting-6 motif enrichment heatmap ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${GRELU_PYTHON} -u "${SCRIPT_DIR}/09f_interesting6_motif_heatmap.py"
echo "Exit code: $?"

echo "End: $(date)"
