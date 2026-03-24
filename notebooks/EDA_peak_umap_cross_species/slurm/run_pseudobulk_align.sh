#!/bin/bash
#SBATCH --job-name=pb_align_14
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species/logs/pb_align_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species/logs/pb_align_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_umap_cross_species"
mkdir -p "${SCRIPT_DIR}/logs"

source /hpc/user_apps/data.science/data.science.bashrc 2>/dev/null || true
module load anaconda 2>/dev/null || true
module load data.science 2>/dev/null || true

echo "=== Script 14: Align Pseudobulk UMAPs ==="
echo "Start: $(date)"
echo "Host:  $(hostname)"

SCRATCH="/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
for f in "${SCRATCH}/anchors/root_anchors.csv" \
          "${SCRATCH}/anchors/branch_anchors.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required input missing: $f"
        exit 1
    fi
done

conda run --no-capture-output \
    -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "${SCRIPT_DIR}/14_align_pseudobulk_umap.py"

echo "End: $(date)"
