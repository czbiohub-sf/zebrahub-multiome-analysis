#!/bin/bash
#SBATCH --job-name=leiden_de_continuous
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/leiden_de_continuous_%j.out
#SBATCH --error=logs/leiden_de_continuous_%j.err

echo "========================================"
echo "Leiden clustering + DE motif analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Mem: $SLURM_MEM_PER_NODE MB"
echo "Started: $(date)"
echo "========================================"

REPO="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
SCRIPT="${REPO}/notebooks/EDA_peak_umap_cross_species/06b_leiden_de_continuous.py"

module load anaconda
module load data.science

conda run --no-capture-output -p /hpc/user_apps/data.science/conda_envs/single-cell-base \
    python -u "$SCRIPT"

EXIT_CODE=$?
echo "========================================"
echo "Finished: $(date)  exit_code=$EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
