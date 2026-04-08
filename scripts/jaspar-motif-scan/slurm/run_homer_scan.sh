#!/bin/bash
#SBATCH --job-name=jaspar_scan
#SBATCH --array=0-2                  # 0=zebrafish, 1=mouse, 2=human
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/jaspar_scan_%a_%j.out
#SBATCH --error=logs/jaspar_scan_%a_%j.err

SPECIES_ARR=(zebrafish mouse human)
SPECIES=${SPECIES_ARR[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/jaspar-motif-scan"

echo "========================================"
echo "JASPAR motif scan: $SPECIES"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

module load anaconda
module load data.science

conda run -p /hpc/user_apps/data.science/conda_envs/gimme \
    python "${SCRIPT_DIR}/03_scan_motifs.py" \
    --species "$SPECIES" \
    --ncpus "$SLURM_CPUS_PER_TASK"

echo "Finished: $(date)"
