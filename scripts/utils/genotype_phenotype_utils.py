# Utility functions to compute genotype-by-phenotype (embryos-by-celltype) matrices

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# inputs:
# 1) adata
# 2) genotype_df: a dataframe with columns ['cell_id', 'genotype']

# function to aggregate the celltypes(phenotypes) per embryo(genotypes)
def compute_genotype_by_phenotype_matrix(adata, key_genotype='embryo',
                                         key_phenotype='cell_type_broad'):
    """
    Compute genotype-by-phenotype matrix from the adata object.
    This function is useful to compute the genotype-by-phenotype matrix
    for the multiome data, where the genotype information is stored in the
    adata.obs['embryo'] column.
    """

    # Extract the necessary metadata
    embryos = adata.obs[key_genotype]
    cell_types = adata.obs[key_phenotype]

    # Create a DataFrame from the metadata
    metadata_df = pd.DataFrame({'embryo': embryos, 'cell_type': cell_types})

    # Group by embryo and cell type, and count the occurrences
    grouped = metadata_df.groupby(['embryo', 'cell_type']).size().unstack(fill_value=0)

    # Display the resulting aggregated counts
    grouped.head()

    # Convert the DataFrame to a sparse matrix
    X_sparse = csr_matrix(grouped.values)

    # Create a new AnnData object with the sparse matrix
    adata_agg = sc.AnnData(X=X_sparse)

    # Ensure that the index and columns are correctly named
    adata_agg.obs_names = grouped.index
    adata_agg.var_names = grouped.columns

    # # Optionally, save the new AnnData object to a file
    # adata_agg.write("adata_agg.h5ad")

    return adata_agg

# function to preprocess the aggregated embryos-by-celltype matrix (UMAP)
def preprocess_aggregated_adata(adata_agg):
    """
    Preprocess the aggregated adata object for visualization.
    """

    # save the raw counts (number of cells per celltype per embryo)
    adata_agg.layers["counts"] = adata_agg.X.copy()

    # Normalize the data
    sc.pp.normalize_total(adata_agg, target_sum=1e4)

    # Log-transform the data
    sc.pp.log1p(adata_agg)
    adata_agg.raw = adata_agg

    # Scale the data
    sc.pp.scale(adata_agg)

    # Perform PCA
    sc.tl.pca(adata_agg, svd_solver="arpack")

    # Perform UMAP
    sc.pp.neighbors(adata_agg, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata_agg, min_dist=0.05)

    return adata_agg




