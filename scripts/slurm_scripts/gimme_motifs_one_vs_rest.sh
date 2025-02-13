#!/bin/bash
#SBATCH --job-name=gimme_motifs_one_vs_rest
#SBATCH --partition=cpu
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/gimme_motifs_one_vs_rest_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/gimme_motifs_one_vs_rest_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

# Create a new cache directory 
NEW_CACHE="/hpc/scratch/group.data.science/yangjoon.kim/gimmemotifs_tmp/"
mkdir -p "$NEW_CACHE"  
# If XDG_CACHE_HOME is not set, default to the scratch directory
if [ -z "$XDG_CACHE_HOME" ]; then     
    XDG_CACHE_HOME="$NEW_CACHE"
fi  
# Copy existing gimmemotifs cache to the new cache location
cp -r "$HOME/.cache/gimmemotifs" "$NEW_CACHE/gimmemotifs" 2>/dev/null || true

# Set the XDG_CACHE_HOME environment variable to point to the new cache
export XDG_CACHE_HOME="$NEW_CACHE"

# Print a message indicating the cache location
echo "Using $XDG_CACHE_HOME for cache"

# Load mamba and activate gimme module
module load mamba
mamba activate gimme

# Change directory to the folder with FASTA files
# cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/one_vs_rest/
# 1. Set working directory to the folder with FASTA files
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/one_vs_rest/

# Base directory for input and output
BASE_DIR="$PWD"

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_fasta_file>"
    exit 1
fi
# Get the input file name (full path)
input_file="$BASE_DIR/$1"
# Extract cluster ID from the filename (assumes format like cluster_0.fasta)
# cluster_id=$(basename "$input_file" .fasta)
cluster_id=$(echo "$1" | sed -E 's/(.+)\.(fa|fasta)/\1/')

# Define base output directory
base_output_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/motifs_result_one_vs_rest/"
# Create cluster-specific output directory
output_dir="${base_output_dir}/${cluster_id}/"
mkdir -p "$output_dir"

# Construct background file name
background_file="${BASE_DIR}/${cluster_id}_rest.fasta"

# ref genome directory
genome_dir="/hpc/mydata/yang-joon.kim/genomes/danRer11/"

# Print out detailed logging information
echo "Job Information:"
echo "---------------"
echo "Input FASTA file: $input_file"
echo "Cluster ID: $cluster_id"
echo "Output Directory: $output_dir"
echo "Background File: $background_file"
echo "Genome Directory: $genome_dir"
echo "---------------"

# Run gimme motifs (NOTE that the -s 0 is to use the original region size)
gimme motifs "$input_file" "$output_dir" -b "$background_file" -g "$genome_dir" -s 0

# Deactivate the mamba environment
source deactivate