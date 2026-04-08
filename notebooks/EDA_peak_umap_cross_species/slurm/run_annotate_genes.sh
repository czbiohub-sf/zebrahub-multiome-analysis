#!/bin/bash
#SBATCH --job-name=annotate_genes_07
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species/logs/annotate_genes_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species/logs/annotate_genes_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species"
mkdir -p "${SCRIPT_DIR}/logs"

source /hpc/user_apps/data.science/data.science.bashrc 2>/dev/null || true
module load anaconda 2>/dev/null || true
module load data.science 2>/dev/null || true

echo "=== Script 07: Annotate Mouse/Human Peaks ==="
echo "Start: $(date)"
echo "Host:  $(hostname)"

conda run --no-capture-output \
    -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "${SCRIPT_DIR}/07_annotate_genes_mouse_human.py"

echo "End: $(date)"
echo "Exit: $?"
