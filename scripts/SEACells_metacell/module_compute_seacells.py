# This is a module for a collection of functions to compute MetaCell for scRNA-seq/scATAC-seq data using SEACells (Persad et al., 2023)
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import SEACells
import matplotlib.pyplot as plt
import scipy.sparse as sp


# NOTE 1. We're using the ATAC modality for the SEACell computation.
# # loading the datasets
# filepath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/'
# # RNA data
# rna_ad = sc.read(filepath + "TDR118_processed_RNA.h5ad")
# # ATAC data (post-SEACell computation)
# atac_ad = sc.read(filepath + 'TDR118_processed_peaks_merged.h5ad')

def aggregate_counts_multiome(adata_rna, adata_atac, groupby='SEACell', celltype=None):
    """
    This function aggregates counts over SEACells for multiome data
    Parameters:
    -----------
    adata_rna : AnnData
        RNA expression data
    adata_atac : AnnData
        ATAC data (gene expression in adata_atac.X)
    groupby : str, optional
        Column name in obs containing SEACell assignments (default: 'SEACell')
    celltype : array-like, optional
        Cell type annotations for each cell (default: None)
    
    Returns:
    --------
    tuple
        (adata_rna, SEACell_rna_ad, adata_atac, SEACell_atac_ad)
    """
    # Aggregate RNA counts over SEACells
    SEACell_rna_ad = SEACells.aggregate_counts(adata_rna, groupby=groupby)
    
    # Aggregate ATAC counts over SEACells
    SEACell_atac_ad = SEACells.aggregate_counts(adata_atac, groupby=groupby)
    
    # Add celltype information if provided
    if celltype is not None:
        SEACell_rna_ad.obs["celltype"] = celltype
        SEACell_atac_ad.obs["celltype"] = celltype
        adata_rna.obs["celltype"] = celltype
        adata_atac.obs["celltype"] = celltype
    
    # Add the SEACell assignments to the original adata objects
    adata_rna.obs["SEACell"] = SEACell_rna_ad.obs["SEACell"]
    adata_atac.obs["SEACell"] = SEACell_atac_ad.obs["SEACell"]
    
    return adata_rna, SEACell_rna_ad, adata_atac, SEACell_atac_ad

# define utility functions
def assert_raw_counts(adata):
    # Check if the data is stored in a sparse matrix format
    if sp.issparse(adata.X):
        # Convert to dense format to work with numpy
        data = adata.X.todense()
    else:
        data = adata.X

    # Check if all values in the data are integers
    if not np.all(np.rint(data) == data):
        raise ValueError("adata.X does not contain raw counts (integer values).")

# SEACells.plot.plot_2D modified
def plot_2D_modified(
    ad,
    ax,
    key="X_umap",
    colour_metacells=True,
    title="Metacell Assignments",
    palette="Set2",
    SEACell_size=20,
    cell_size=10,
):
    """Plot 2D visualization of metacells using the embedding provided in 'key'.

    :param ad: annData containing 'Metacells' label in .obs
    :param ax: Axes object where the plot will be drawn
    :param key: (str) 2D embedding of data. Default: 'X_umap'
    :param colour_metacells: (bool) whether to colour cells by metacell assignment. Default: True
    :param title: (str) title for figure
    :param palette: (str) matplotlib colormap for metacells. Default: 'Set2'
    :param SEACell_size: (int) size of SEACell points
    :param cell_size: (int) size of cell points
    """
    umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs["SEACell"])
    umap["SEACell"] = umap["SEACell"].astype("category")
    mcs = umap.groupby("SEACell").mean().reset_index()

    if colour_metacells:
        sns.scatterplot(
            x=0, y=1, hue="SEACell", data=umap, s=cell_size, palette=sns.color_palette(palette), legend=None, ax=ax
        )
        sns.scatterplot(
            x=0,
            y=1,
            s=SEACell_size,
            hue="SEACell",
            data=mcs,
            palette=sns.color_palette(palette),
            edgecolor="black",
            linewidth=1.25,
            legend=None,
            ax=ax
        )
    else:
        sns.scatterplot(
            x=0, y=1, color="grey", data=umap, s=cell_size, legend=None, ax=ax
        )
        sns.scatterplot(
            x=0,
            y=1,
            s=SEACell_size,
            color="red",
            data=mcs,
            edgecolor="black",
            linewidth=1.25,
            legend=None,
            ax=ax
        )

    ax.set_xlabel(f"{key}-0")
    ax.set_ylabel(f"{key}-1")
    ax.set_title(title)
    ax.set_axis_off()

# SEACells.plot.plot_SEACell_sizes modified
def plot_SEACell_sizes_modified(
    ad,
    ax,
    save_as=None,
    show=True,
    title="Distribution of Metacell Sizes",
    bins=None,
    figsize=(5, 5),
):
    """Plot distribution of number of cells contained per metacell.

    :param ad: annData containing 'Metacells' label in .obs
    :param ax: Axes object where the plot will be drawn
    :param save_as: (str) path to which figure is saved. If None, figure is not saved.
    :param show: (bool) whether to show the plot
    :param title: (str) title of figure.
    :param bins: (int) number of bins for histogram
    :param figsize: (int,int) tuple of integers representing figure size
    :return: None.
    """
    assert "SEACell" in ad.obs, 'AnnData must contain "SEACell" in obs DataFrame.'
    label_df = ad.obs[["SEACell"]].reset_index()
    
    sns.histplot(label_df.groupby("SEACell").count().iloc[:, 0], bins=bins, ax=ax)
    sns.despine()
    ax.set_xlabel("Number of Cells per SEACell")
    ax.set_title(title)

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()
    return pd.DataFrame(label_df.groupby("SEACell").count().iloc[:, 0]).rename(
        columns={"index": "size"}
    )