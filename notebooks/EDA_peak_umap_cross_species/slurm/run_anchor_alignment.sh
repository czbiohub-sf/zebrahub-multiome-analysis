#!/bin/bash
#SBATCH --job-name=procrustes_12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species/logs/procrustes_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species/logs/procrustes_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species"
mkdir -p "${SCRIPT_DIR}/logs"

source /hpc/user_apps/data.science/data.science.bashrc 2>/dev/null || true
module load anaconda 2>/dev/null || true
module load data.science 2>/dev/null || true

echo "=== Script 12: Procrustes Alignment ==="
echo "Start: $(date)"
echo "Host:  $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

SCRATCH="/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
for f in "${SCRATCH}/cross_species_motif_embedded_annotated.h5ad" \
          "${SCRATCH}/anchors/root_anchors.csv" \
          "${SCRATCH}/anchors/branch_anchors.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required input missing: $f"
        exit 1
    fi
done

conda run --no-capture-output \
    -p /hpc/user_apps/data.science/conda_envs/sc_rapids \
    python -u "${SCRIPT_DIR}/12_procrustes_alignment.py"

echo "End: $(date)"
