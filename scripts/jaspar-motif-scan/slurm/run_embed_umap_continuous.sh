#!/bin/bash
#SBATCH --job-name=motif_embed_continuous
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/motif_embed_continuous_%j.out
#SBATCH --error=logs/motif_embed_continuous_%j.err

echo "========================================"
echo "Cross-species motif UMAP — continuous PWM scores"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Mem: $SLURM_MEM_PER_NODE MB"
echo "Started: $(date)"
echo "========================================"

REPO="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
SCRIPT="${REPO}/notebooks/EDA_peak_umap_cross_species/05b_embed_continuous_umap.py"

module load anaconda
module load data.science

conda run --no-capture-output -p /hpc/user_apps/data.science/conda_envs/sc_rapids \
    python "$SCRIPT"

EXIT_CODE=$?
echo "========================================"
echo "Finished: $(date)  exit_code=$EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
