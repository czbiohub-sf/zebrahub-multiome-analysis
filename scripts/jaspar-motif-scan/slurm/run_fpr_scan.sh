#!/bin/bash
#SBATCH --job-name=jaspar_fpr_scan
#SBATCH --array=0-2                  # 0=zebrafish, 1=mouse, 2=human
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/jaspar_fpr_scan_%a_%j.out
#SBATCH --error=logs/jaspar_fpr_scan_%a_%j.err

SPECIES_ARR=(zebrafish mouse human)
SPECIES=${SPECIES_ARR[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/jaspar-motif-scan"

echo "========================================"
echo "JASPAR FPR binary scan: $SPECIES"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Mem: $SLURM_MEM_PER_NODE MB"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

module load anaconda
module load data.science

# GimmeMotifs uses SQLite for caching background thresholds.
# Parallel array jobs corrupt the shared cache — give each task its own dir.
GIMME_CACHE="/hpc/scratch/group.data.science/yang-joon.kim/gimme_cache/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$GIMME_CACHE"
export XDG_CACHE_HOME="$GIMME_CACHE"
echo "GimmeMotifs cache: $GIMME_CACHE"

conda run -p /hpc/user_apps/data.science/conda_envs/gimme \
    python "${SCRIPT_DIR}/03b_scan_motifs_fpr.py" \
    --species "$SPECIES" \
    --ncpus "$SLURM_CPUS_PER_TASK" \
    --fpr 0.01

echo "Finished: $(date)"
