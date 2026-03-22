#!/bin/bash
#SBATCH --job-name=branch_anchors_11
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/branch_anchors_%j.out
#SBATCH --error=logs/branch_anchors_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

module load anaconda data.science

echo "=== Script 11: Select Branch Anchors ==="
echo "Start: $(date)"
echo "Host:  $(hostname)"

SCRATCH="/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
for f in "${SCRATCH}/cross_species_motif_embedded_annotated.h5ad" \
          "${SCRATCH}/orthologs/zebrafish_mouse_human_1to1_orthologs.csv" \
          "${SCRATCH}/anchors/root_anchors.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required input missing: $f"
        exit 1
    fi
done

conda run --no-capture-output \
    -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "${SCRIPT_DIR}/11_select_branch_anchors.py"

echo "End: $(date)"
