#!/bin/bash

# Directory containing the FASTA files
FASTA_DIR="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/one_vs_rest/"

# Path to the gimme motifs script
GIMME_SCRIPT="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/slurm_scripts/gimme_motifs_one_vs_rest.sh"

# Loop through clusters 1-38, skipping 6
for cluster in {1..38}; do
    if [ $cluster -ne 6 ]; then
        # Try both .fasta and .fa extensions
        if [ -f "${FASTA_DIR}/cluster_${cluster}.fasta" ]; then
            input_file="cluster_${cluster}.fasta"
        elif [ -f "${FASTA_DIR}/cluster_${cluster}.fa" ]; then
            input_file="cluster_${cluster}.fa"
        else
            echo "Warning: No file found for cluster $cluster"
            continue
        fi
        
        echo "Submitting job for cluster $cluster"
        sbatch "$GIMME_SCRIPT" "$input_file"
    fi
done