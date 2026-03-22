#!/bin/bash
#SBATCH --job-name=annotate_genes_07
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/annotate_genes_%j.out
#SBATCH --error=logs/annotate_genes_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

module load anaconda data.science

echo "=== Script 07: Annotate Mouse/Human Peaks ==="
echo "Start: $(date)"
echo "Host:  $(hostname)"

conda run --no-capture-output \
    -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "${SCRIPT_DIR}/07_annotate_genes_mouse_human.py"

echo "End: $(date)"
