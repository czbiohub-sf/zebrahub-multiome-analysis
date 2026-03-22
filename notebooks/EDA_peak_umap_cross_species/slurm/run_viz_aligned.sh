#!/bin/bash
#SBATCH --job-name=viz_aligned_13
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/viz_aligned_%j.out
#SBATCH --error=logs/viz_aligned_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

module load anaconda data.science

echo "=== Script 13: Visualize Aligned UMAP ==="
echo "Start: $(date)"
echo "Host:  $(hostname)"

SCRATCH="/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
for f in "${SCRATCH}/cross_species_anchor_aligned.h5ad" \
          "${SCRATCH}/anchors/root_anchors.csv" \
          "${SCRATCH}/anchors/branch_anchors.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required input missing: $f"
        exit 1
    fi
done

conda run --no-capture-output \
    -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "${SCRIPT_DIR}/13_visualize_aligned_umap.py"

echo "End: $(date)"
