#!/bin/bash

#SBATCH --job-name=multiome_processing      # Job name
#SBATCH --partition=cpu                     # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/Signac_processing/signac_processing_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/Signac_processing/signac_processing_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=48:00:00                     # Runtime in HH:MM:SS            
#SBATCH --mem=256G                          # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=20                  # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Usage check
if [ "$#" -ne 6 ]; then
    echo "Usage: sbatch $0 raw_data_path gref_path reference annotation_class output_filepath data_id"
    exit 1
fi

# Assign command-line arguments to variables
raw_data_path=$1 # raw data path (10xCellranger-arc output files)
gref_path=$2 # genome reference path (GTF files)
reference=$3 # reference dataset with cell-type annotation
annotation_class=$4 # Let's just use one annotation_class. (i.e. "global_annotation")
output_filepath=$5 # filepath for the output
data_id=$6 # data_id for the output

# Create the output directory if it doesn't exist
mkdir -p "$output_filepath"

# Load necessary modules
module load mamba # for snakemake conda environment
module load R/4.3
conda activate snakemake # activate the conda environment (with macs2)

# Navigate to the directory containing the R script
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/

# Run the R script
Rscript run_01_preprocess_multiome_object_signac.R $raw_data_path $gref_path $reference $annotation_class $output_filepath $data_id