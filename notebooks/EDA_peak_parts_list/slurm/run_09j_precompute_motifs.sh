#!/bin/bash
#SBATCH --job-name=parts_09j_fimo
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09j_precompute_motifs_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09j_precompute_motifs_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
GRELU_PYTHON="/home/yang-joon.kim/.conda/envs/gReLu/bin/python"

echo "=== Script 09j: Precompute FIMO motif hits (top-200) ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${GRELU_PYTHON} -u "${SCRIPT_DIR}/09j_precompute_motif_hits_top200.py"
echo "Exit code: $?"

echo "End: $(date)"
