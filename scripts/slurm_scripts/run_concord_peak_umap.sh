#!/bin/bash

#SBATCH --job-name=concord_peak_umap           # Job name
#SBATCH --partition=gpu                        # GPU partition
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/concord_peak_umap_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/concord_peak_umap_%j.err
#SBATCH --time=4:00:00                         # Runtime (4 hours should be enough)
#SBATCH --mem=128G                             # Memory
#SBATCH --cpus-per-task=8                      # CPU cores
#SBATCH --mail-type=END,FAIL                   # Email notifications
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

echo "=========================================="
echo "CONCORD Peak UMAP Job Started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Show GPU info
nvidia-smi

# Load modules and activate conda environment
module load anaconda
module load data.science
conda activate sc_rapids

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Install torch and concord-sc if not already installed
pip install torch --quiet
pip install concord-sc --quiet

# Navigate to scripts directory
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/

# Run the Python script
python run_concord_peak_umap.py

echo "=========================================="
echo "Job Completed: $(date)"
echo "=========================================="
