# This is a module for a collection of functions to compute MetaCell for scRNA-seq/scATAC-seq data using SEACells (Persad et al., 2023)

import numpy as np
import pandas as pd
import scanpy as sc
import SEACells

# NOTE 1. We're using the ATAC modality for the SEACell computation.

# loading the datasets
filepath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/'
# RNA data
rna_ad = sc.read(filepath + "TDR118_processed_RNA.h5ad")
# ATAC data (post-SEACell computation)
atac_ad = sc.read(filepath + 'TDR118_processed_peaks_merged.h5ad')

def aggregate_counts_multiome(adata_rna, adata_atac):
    """
    This function aggregates counts over SEACells
    """
    # Aggregate counts over SEACells
    SEACell_ad = SEACells.aggregate_counts(adata, groupby=groupby)
    # Add celltype information
    SEACell_ad.obs["celltype"] = celltype
    # Add the aggregated counts to the original adata object
    adata.obs["SEACell"] = SEACell_ad.obs["SEACell"]
    adata.obs["celltype"] = celltype
    return adata, SEACell_ad