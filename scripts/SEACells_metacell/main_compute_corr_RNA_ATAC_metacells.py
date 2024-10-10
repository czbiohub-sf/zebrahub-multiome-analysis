# A python script to compute the correlation between GEX(RNA) and peak accessibility(ATAC) at the metacell level
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

# parallel processing
import os
from multiprocessing import Pool

# pyensembl library to map the genes to chromosomes where they belong to.
from pyensembl import EnsemblRelease

# Use the appropriate Ensembl release
data = EnsemblRelease(111, species="danio_rerio")
data.download()
data.index()

# define the slurm log paths
script_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/corr_metacells_TDR118/"
output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/corr_metacells_TDR118/"

# Step 1. Load the RNA and ATAC metacell data (aggregated)
# This should be the input arguments to generalize this script.
# Load the RNA metacell data
rna_meta_ad = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/TDR118reseq/TDR118_RNA_seacells_aggre_chr_annotated.h5ad")
rna_meta_ad

# # Get chromosomes for each gene in the list
# chromosome_list = [get_chromosome(gene) for gene in rna_meta_ad.var_names]
# chromosome_list
# rna_meta_ad.var["chromosome"] = chromosome_list

# # save the metacell adata (with chromosomes annotated for each gene)
# rna_meta_ad.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/TDR118reseq/TDR118_RNA_seacells_aggre_chr_annotated.h5ad")

# Load the ATAC metacell data
atac_meta_ad = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/TDR118reseq/TDR118_ATAC_seacells_aggre.h5ad")
atac_meta_ad

# Extract chromosome number
def extract_chromosome(peak_str):
    # Split the string by '-' and return the first element (chromosome number)
    return peak_str.split('-')[0]

atac_meta_ad.var['chromosome'] = atac_meta_ad.var_names.map(extract_chromosome)
atac_meta_ad.var

# Initialize an empty DataFrame to store gene-peak correlations
# gene_peak_correlations = pd.DataFrame()

# recover the raw counts
rna_meta_ad.X = rna_meta_ad.raw.X.copy()
atac_meta_ad.X = atac_meta_ad.raw.X.copy()

# log-normalize the counts
sc.pp.normalize_total(rna_meta_ad, target_sum=1e4)
sc.pp.log1p(rna_meta_ad)

sc.pp.normalize_total(atac_meta_ad, target_sum=1e4)
sc.pp.log1p(atac_meta_ad)

# Step 2. Subset the data (RNA and ATAC) for each chromosome,
# then compute the correlation between the (gene, peak) pairs for each chromosome.

for chrom in rna_meta_ad.var["chromosome"].unique():

    # chrom = "1"

    # subset the adata_rna for one chromosome
    rna_subset = rna_meta_ad[:,rna_meta_ad.var["chromosome"]==chrom]

    # subset the adata_atac for one chromosome
    atac_subset = atac_meta_ad[:,atac_meta_ad.var["chromosome"]==chrom]

    # save the subsetted adata (RNA and ATAC) for each chromosome
    rna_chr_path = f"{output_dir}/rna_chr_{chrom}.h5ad"
    atac_chr_path = f"{output_dir}/atac_chr_{chrom}.h5ad"

    # Create a shell script for the SLURM job
    script_name = os.path.join(script_dir, f"corr_job_{chrom}.sh")
    script_content = f"""#!/bin/bash
        #SBATCH --output={script_dir}/corr_{chrom}_%j.out
        #SBATCH --error={script_dir}/corr_{chrom}_%j.err
        #SBATCH --time=24:00:00
        #SBATCH --mem=64G
        #SBATCH --mail-type=FAIL
        #SBATCH --mail-user=yang-joon.kim@czbiohub.org
        module load anaconda
        conda activate celloracle_env

        cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/
        python compute_corr_RNA_ATAC_metacells.py --rna_data_path {rna_chr_path} --atac_data_path {atac_chr_path} --chromosome {chrom} --output_dir {output_dir}
        """

    #script_name = os.path.join(script_dir, f"corr_job_{chrom}.sh")
    with open(script_name, 'w') as file:
        file.write(script_content)
    os.system(f"sbatch {script_name}")

#     # extract the count matrices (log-norm)
#     rna_matrix = rna_subset.X
#     atac_matrix = atac_subset.X

#     # Number of genes and peaks
#     num_genes = rna_matrix.shape[1]
#     num_peaks = atac_matrix.shape[1]

#     # Initialize an empty DataFrame to store gene-peak correlations
#     # gene_peak_correlation = pd.DataFrame(index=rna_subset.var_names, columns=atac_subset.var_names)

#     # # Compute correlations
#     # for i in range(num_genes):
#     #     for j in range(num_peaks):
#     #         gene_vector = rna_matrix[:, i]
#     #         peak_vector = atac_matrix[:, j]
#     #         correlation = compute_correlation(gene_vector, peak_vector)
#     #         gene_peak_correlation.iloc[i, j] = correlation

#     # Generate all combinations of gene and peak indices
#     indices_combinations = [(i, j) for i in range(rna_matrix.shape[1]) for j in range(atac_matrix.shape[1])]

#     # Number of processes to run in parallel
#     num_processes = 32  # Adjust this based on your machine's capability

#     # Perform parallel computation
#     with Pool(processes=num_processes) as pool:
#         correlation_values = pool.map(compute_correlation, indices_combinations)

#     # Converting the correlation values to a DataFrame
#     correlation_matrix = np.array(correlation_values).reshape((rna_matrix.shape[1], atac_matrix.shape[1]))
#     gene_peak_correlations = pd.DataFrame(correlation_matrix, index=rna_subset.var_names, columns=atac_subset.var_names)

# # merge the gene_peak_correlations dataframe from all chromosomes
# gene_peak_correlations_all = pd.concat(gene_peak_correlations, axis=1)




# Additional functions
# Function to compute correlation coefficient
# def compute_correlation(x, y):
#     x_dense = x.A.flatten()  # Convert to dense format and flatten
#     y_dense = y.A.flatten()
#     corr, _ = pearsonr(x_dense, y_dense)
#     return corr

# # Function to compute correlation coefficient
# def compute_correlation(args):
#     i, j = args
#     gene_vector = rna_matrix[:, i].A.flatten()  # Convert to dense format and flatten
#     peak_vector = atac_matrix[:, j].A.flatten()
#     return pearsonr(gene_vector, peak_vector)[0]