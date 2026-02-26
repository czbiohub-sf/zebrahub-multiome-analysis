#!/bin/bash

#SBATCH --job-name=plot_concord
#SBATCH --partition=cpu
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/plot_concord_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/plot_concord_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module load anaconda
module load data.science
conda activate sc_rapids

cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/
python scripts/plot_concord_comparison.py
