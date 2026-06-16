#!/bin/bash
#SBATCH --job-name=fimo_ct_cisbp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --array=0-30
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09j_fimo_cisbpv2_ct%a_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/logs/09j_fimo_cisbpv2_ct%a_%j.err

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list"
GRELU_PYTHON="/home/yang-joon.kim/.conda/envs/gReLu/bin/python"

# CIS-BP has ~5,300 motifs (vs 1,443 for H12CORE) — walltime bumped to 3h per task
echo "=== FIMO (CIS-BP v2 Danio rerio): celltype ${SLURM_ARRAY_TASK_ID}/31 ==="
echo "Start: $(date)"; echo "Host: $(hostname)"

${GRELU_PYTHON} -u "${SCRIPT_DIR}/09j_fimo_batch.py" \
    --celltype-idx ${SLURM_ARRAY_TASK_ID} \
    --motif-db cisbpv2_danrer

echo "Exit code: $?"
echo "End: $(date)"
