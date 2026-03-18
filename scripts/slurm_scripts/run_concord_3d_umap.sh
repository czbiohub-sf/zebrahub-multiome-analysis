#!/bin/bash

#SBATCH --job-name=concord_3d_umap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/concord_3d_umap_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/concord_3d_umap_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "=========================================="
echo "CONCORD 3D UMAP Job Started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

nvidia-smi

# Load modules and activate conda environment
module load anaconda
module load data.science
conda activate sc_rapids

echo "Python: $(which python)"

# Install torch and concord-sc if not already installed
pip install torch --quiet
pip install concord-sc --quiet

cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/
python scripts/run_concord_3d_umap.py

echo "=========================================="
echo "Job Completed: $(date)"
echo "=========================================="
