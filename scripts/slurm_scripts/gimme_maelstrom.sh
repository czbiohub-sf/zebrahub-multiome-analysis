#!/bin/bash

#SBATCH --job-name=gimme_maelstrom      # Job name
#SBATCH --partition=cpu                       # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/gimme_maelstrom_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/gimme_maelstrom_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=48:00:00                       # Runtime in HH:MM:SS
#SBATCH --mem=1024G                             # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=12                     # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                  # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Load necessary modules
module load mamba
mamba activate gimme

# Set up a job-specific cache directory in scratch to avoid SQLite conflicts
SCRATCH="/hpc/scratch/group.data.science/yangjoon.kim"
export XDG_CACHE_HOME="$SCRATCH/gimme_cache_$SLURM_JOB_ID"
mkdir -p "$XDG_CACHE_HOME"

# Fallback in case original XDG_CACHE_HOME isn't set
if [ -z "$XDG_CACHE_HOME_ORIG" ]; then
    XDG_CACHE_HOME_ORIG="$HOME/.cache"
fi
# Copy over existing GimmeMotifs cache to local scratch if available
if [ -d "$XDG_CACHE_HOME_ORIG/gimmemotifs" ]; then
    cp -r "$XDG_CACHE_HOME_ORIG/gimmemotifs" "$XDG_CACHE_HOME/"
fi
# Set the XDG_CACHE_HOME environment variable to the new cache directory
echo "Using XDG_CACHE_HOME=$XDG_CACHE_HOME"

# input arguments (input, ref_genome, output_dir, and optionally pfmfile)
#input="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/motif_enrich_analysis_leiden_1.5/peaks_640830_leiden_1.5.txt"
input="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_fine_cisBP_v2_danio_rerio_output/peaks_leiden_coarse_fine_unified.txt"
# ref_genome="/hpc/mydata/yang-joon.kim/genomes/danRer11/danRer11.fa"
ref_genome="danRer11"
output_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_fine_cisBP_v2_danio_rerio_output"
#output_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden1.5_GM.vertebrate.v5_output"
mkdir -p $output_dir
pfmfile="CisBP_ver2_Danio_rerio"

gimme maelstrom $input $ref_genome $output_dir --pfmfile $pfmfile

