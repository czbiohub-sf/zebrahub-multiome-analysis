#!/bin/bash
#SBATCH --job-name=merge_fragments      # Job name
#SBATCH --partition=cpu                     # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/merge_fragments_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/merge_fragments_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=48:00:00                     # Runtime in HH:MM:SS            
#SBATCH --mem=4G                          # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

module load samtools

cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/

# decompress files and add the same cell prefix as was added to the Seurat object
gzip -dc TDR118reseq/outs/atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,"TDR118_"$4,$5}' - > TDR118_fragments.tsv
gzip -dc TDR119reseq/outs/atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,"TDR119_"$4,$5}' - > TDR119_fragments.tsv
gzip -dc TDR124reseq/outs/atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,"TDR124_"$4,$5}' - > TDR124_fragments.tsv
gzip -dc TDR125reseq/outs/atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,"TDR125_"$4,$5}' - > TDR125_fragments.tsv
gzip -dc TDR126/outs/atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,"TDR126_"$4,$5}' - > TDR126_fragments.tsv
gzip -dc TDR127/outs/atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,"TDR127_"$4,$5}' - > TDR127_fragments.tsv
gzip -dc TDR128/outs/atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,"TDR128_"$4,$5}' - > TDR128_fragments.tsv

# merge files (avoids having to re-sort)
sort -m -k 1,1V -k2,2n TDR118_fragments.tsv TDR119_fragments.tsv TDR124_fragments.tsv TDR125_fragments.tsv TDR126_fragments.tsv TDR127_fragments.tsv TDR128_fragments.tsv > merged_fragments.tsv

# block gzip compress the merged file
bgzip -@ 4 merged_fragments.tsv # -@ 4 uses 4 threads

# index the bgzipped file
tabix -p bed merged_fragments.tsv.gz

# remove intermediate files
rm TDR118_fragments.tsv TDR119_fragments.tsv TDR124_fragments.tsv TDR125_fragments.tsv TDR126_fragments.tsv TDR127_fragments.tsv TDR128_fragments.tsv