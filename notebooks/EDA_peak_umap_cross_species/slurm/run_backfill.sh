#!/bin/bash
#SBATCH --job-name=backfill_09
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/backfill_%j.out
#SBATCH --error=logs/backfill_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

module load anaconda data.science

echo "=== Script 09: Backfill Annotations ==="
echo "Start: $(date)"
echo "Host:  $(hostname)"

# Dependency: scripts 07 (gene annotations) and 08 (orthologs) must be complete
SCRATCH="/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
for f in "${SCRATCH}/gene_annotations/mouse_peaks_gene_annotated.csv" \
          "${SCRATCH}/gene_annotations/human_peaks_gene_annotated.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required input missing: $f"
        echo "  Run script 07 first."
        exit 1
    fi
done

conda run --no-capture-output \
    -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "${SCRIPT_DIR}/09_backfill_annotations.py"

echo "End: $(date)"
