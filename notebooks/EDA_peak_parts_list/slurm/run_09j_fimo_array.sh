#!/bin/bash
#SBATCH --job-name=fimo_ct
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:30:00
#SBATCH --array=0-30
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09j_fimo_ct%a_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09j_fimo_ct%a_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
GRELU_PYTHON="/home/yang-joon.kim/.conda/envs/gReLu/bin/python"

echo "=== FIMO: celltype ${SLURM_ARRAY_TASK_ID}/31 ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${GRELU_PYTHON} -u "${SCRIPT_DIR}/09j_fimo_batch.py" \
    --celltype-idx ${SLURM_ARRAY_TASK_ID}

echo "Exit code: $?"
echo "End: $(date)"
