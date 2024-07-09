# A python script to compute the correlation between RNA(gene) and ATAC(gene activity) across metacells.
# NOTE. This script is designed to be run on the cluster environment.
# NOTE. This script takes RNA and ATAC adata objects as input.
# This script takes RNA and ATAC adata objects from all timepoints as input, subset them for each timepoint/dataset, 
# then computing the correlation between RNA and ATAC (gene activity) for each gene across metacells.

# NOTE. conda environment: seacells
# conda activate seacells

# Import libraries
import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
from scipy.stats import pearsonr
import ray # for parallelization

import os
import sys
# Add the directory containing the utility functions to the Python path
sys.path.append('/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/')
# Import the functions from the utils
from utils.module_rna_atac_correlations import compute_gene_correlations  # Import the function

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# # Some plotting aesthetics
# %matplotlib inline
# sns.set_style('ticks')
# matplotlib.rcParams['figure.figsize'] = [4, 4]
# matplotlib.rcParams['figure.dpi'] = 100

# Step 0. Load the adata objects (RNA and ATAC)
# Load the RNA and ATAC(gene.activity) master object, then subset for each timepoint/dataset
filepath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/'

# RNA master object (all timepoints/datasets)
adata_RNA = sc.read_h5ad(filepath + "integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
print(adata_RNA)

# ATAC (gene.acitivitiy) master object (all timepoints/datasets)
adata_ATAC = sc.read_h5ad(filepath + "integrated_RNA_ATAC_counts_gene_activity_raw_counts_master_filtered.h5ad")
print(adata_ATAC)

# (Optional) transfer some annotations
adata_ATAC.obs["annotation_ML_coarse"] = adata_RNA.obs["annotation_ML_coarse"]
adata_ATAC.uns['annotation_ML_coarse_colors'] = adata_RNA.uns['annotation_ML_coarse_colors']

# Step 1. subset for individual dataset
list_datasets = ['TDR126', 'TDR127', 'TDR128',
                'TDR118reseq', 'TDR119reseq', 'TDR125reseq', 'TDR124reseq']

# define an empty dataframe to save the genes-by-corr.coeffs (for each dataset)
combined_df = pd.DataFrame()

# First, get the intersection of all genes across all datasets
all_genes = set(adata_RNA.var_names).intersection(set(adata_ATAC.var_names))

# A for loop to subset the RNA and ATAC objects for each dataset
for data_id in list_datasets:
    # trim the "reseq" from the data_id to match the dataset annotation
    sample_id = data_id.replace("reseq","")
    print(f"Subsetting for {data_id}")

    # subset for each "dataset"
    rna_ad = adata_RNA[adata_RNA.obs.dataset==sample_id]
    atac_ad = adata_ATAC[adata_ATAC.obs.dataset==sample_id]
    
    # reformat the adata.obs_names (to remove the additional index from f"XXXX_{index}")
    rna_ad.obs_names = rna_ad.obs_names.str.rsplit('_', n=1).str[0]
    atac_ad.obs_names = atac_ad.obs_names.str.rsplit('_', n=1).str[0]

    # import the individual ATAC data to copy "X_lsi" ("X_svd") in adata.obsm field
    # since SEACell aggregation requires "X_svd" embedding, we will add it (it's basically the same as X_lsi in our pipeline)
    adata_temp = sc.read_h5ad(filepath + f"{data_id}/{sample_id}_processed_peaks_merged.h5ad")
    adata_temp = adata_temp[adata_temp.obs_names.isin(atac_ad.obs_names)]
    atac_ad.obsm["X_svd"] = adata_temp.obsm["X_lsi"]

    del adata_temp

    # make sure that the "adata.X" is the raw counts
    rna_ad.X = rna_ad.layers["counts"].copy()
    atac_ad.X = atac_ad.layers["counts"].copy()

    # Step 2. Load the SEACells (adata.obs) and copy over the "SEACell" annotation for the aggregation of the counts
    # Load the SEACells (adata.obs)
    seacellpath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/'
    # data_id = "TDR118reseq"
    # sample_id = data_id.replace("reseq","")

    # import the seacell metadata (dataframe)
    df_seacells = pd.read_csv(seacellpath + f"{data_id}_seacells_obs.csv", index_col=0)
    df_seacells

    # create a dictionary of {"cell_id":"SEACell"}
    dict_seacells = df_seacells["SEACell"].to_dict()

    # transfer the "SEACell" information to the RNA and ATAC adata objects
    rna_ad.obs["SEACell"] = rna_ad.obs_names.map(dict_seacells)
    atac_ad.obs["SEACell"] = atac_ad.obs_names.map(dict_seacells)
    
    # add a placeholder value of 0.5 for atac_ad_seacells.var["GC"] field 
    # this is for SEACElls operations
    atac_ad.var["GC"] = 0.5

    # Aggregate the counts across SEACells
    # NOTE. the aggregated "raw" counts are saved in adata.raw slot, and the adata.X contains log-normalized counts
    # NOTE. the normalization was done using using sc.pp.normalize_total, which takes the median # of UMIs/cell across all cells in that dataset as the denominator
    atac_meta_ad, rna_meta_ad = SEACells.genescores.prepare_multiome_anndata(atac_ad, rna_ad, SEACells_label='SEACell')

    # save the aggregated adata objects
    os.makedirs(seacellpath + f"{data_id}/", exist_ok=True)
    atac_meta_ad.write_h5ad(seacellpath + f"{data_id}/{sample_id}_ATAC_seacells_aggre.h5ad")
    rna_meta_ad.write_h5ad(seacellpath + f"{data_id}/{sample_id}_RNA_seacells_aggre.h5ad")

    # First, we'll subset the features that are shared between RNA and ATAC (gene names)
    shared_genes = np.intersect1d(rna_meta_ad.var_names, atac_meta_ad.var_names)
    print(f"Number of shared genes: {len(shared_genes)}")

    # subset the RNA and ATAC objects for the shared genes
    rna_meta_ad = rna_meta_ad[:, shared_genes]
    atac_meta_ad = atac_meta_ad[:, shared_genes]

    # Step 3. Compute the correlation between RNA and ATAC (gene activity) across metacells
    # Call the compute_gene_correlations function with the prepared AnnData objects
    print(f"Computing gene correlations for {data_id}")
    correlation_df, fig = compute_gene_correlations(rna_meta_ad, atac_meta_ad, data_id)

    # Concatenate the results into the combined DataFrame
    if combined_df.empty:
        combined_df = correlation_df
    else:
        # combined_df = pd.concat([combined_df, correlation_df], axis=1)
        combined_df[data_id] = correlation_df[data_id]
    # # Save the correlation DataFrame
    combined_df.to_csv(f'{seacellpath}/gene_rna_gene_activity_corr_coeffs_alltimepoints.csv', index=False)

    # Save the figure
    fig.savefig(f'{seacellpath}/{data_id}_correlation_histogram.png')
    plt.close(fig)

# At the end of your script, shut down Ray gracefully
ray.shutdown()