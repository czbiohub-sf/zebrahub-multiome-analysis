# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Global single-cell-base
#     language: python
#     name: global-single-cell-base
# ---

# %% [markdown]
# ## chromatin accessbility and RNA dynamics (correlation)
#
# - Last updated: 11/18/2024
# - Author: Yang-Joon Kim
#
# - How the RNA expression/chromatin accessibility change over time (0 somites to 30 somites)
# - There are multiple ways to look at this. we'll summarize our attempts in EDA here.
#
#
# ### feature level
# - **[RNA] gene expression (log-norm)
# - **[ATAC] gene.activity** (peaks were linked to each gene based on proximity to the promoter/TSS, or cicero-scores): there are a couple of choices - (1) Signac, (2) cicero, and (3) SEACells gene score. We'll choose the Signac-generated gene activity score, as we can compute these metrics per dataset (without integrating/merging peaks).
#
# ### cell grouping level (pseudo-bulk)
# - **Metacells (SEAcells)**
# - **all cells (per dataset)**
# - single cells
#
# We will try out a couple of combinations. Ultimately, we'd like to automate this process for any count matrices (however they were computed - metacells-by-gene.activity, for example).
#
# - First, we decide on which count matrice (cells-by-features) we'll use. 
# - Second, we will prepare the count matrices by aggregating over either all cells or metaclles. 
# - Third, we will generate features-by-time matrix (there can be the third dimension of Metacells).
# - EDA1: create UMAP where each point is a gene, and the vectors are the concatenation of the gene expression and gene activity (6 numbers with some basic normalization). The different classes will be clusters in the UMAP.
#
# - EDA2: 
#
#

# %%
# 0. Import
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# %%
# %matplotlib inline
from scipy import sparse
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import zscore

# %%
# figure parameter setting
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')

# Plotting style function (run this before plotting the final figure)
def set_plotting_style():
    plt.style.use('seaborn-paper')
    plt.rc('axes', labelsize=12)
    plt.rc('axes', titlesize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=2)
    plt.rc('mathtext', fontset='stixsans', sf='sansserif')
    plt.rc('figure', figsize=[10,9])
    plt.rc('svg', fonttype='none')



# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/RNA_ATAC_dynamics_EDA_v1/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %% [markdown]
# ## Step 0. Import datasets 
#
# - RNA
# - ATAC (Gene.Activity)

# %%
# import the adata to find N highly variable genes
adata_RNA = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
adata_RNA

# %%
adata_ATAC = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_gene_activity_raw_counts_master_filtered.h5ad")
adata_ATAC

# %%
# filter for the gene_names that are shared between "RNA" and "gene.activity"
adata_RNA_filtered = adata_RNA[:,adata_RNA.var_names.isin(adata_ATAC.var_names)]
adata_RNA_filtered

# %%
# the adata object whose genes are not present in gene.activity feature space
adata_RNA_filtered_out = adata_RNA[:,~adata_RNA.var_names.isin(adata_ATAC.var_names)]
adata_RNA_filtered_out

# %% [markdown]
# ## Step 1. RNA dynamics along the real-time (developmental stages)
#
# - Following the Calderon,...,Trapnell, Science, 2022's approach (Figure 1F), we will average the "log-norm" counts across cells for both RNA and ATAC.
#

# %%
# adata_RNA.X = adata_RNA.layers["counts"].copy()

# sc.pp.normalize_total(adata_RNA, target_sum=1e4)
# sc.pp.log1p(adata_RNA)

# %% [markdown]
# ### choosing the HVGs

# %%
# compute N highly variable genes
N_top_genes = 3000
sc.pp.highly_variable_genes(adata_RNA_filtered, layer="counts", 
                            n_top_genes=N_top_genes, flavor="seurat_v3")

# %%
# extract the list of highly variable genes
list_hvg_RNA = adata_RNA_filtered.var_names[adata_RNA_filtered.var.highly_variable==True]

# check if some of the marker genes are present in the list of highly variable genes
print("myf5" in list_hvg_RNA)
print("meox1" in list_hvg_RNA)
print("sox2" in list_hvg_RNA)
print("tbxta" in list_hvg_RNA)

# %%
"sox2" in adata_RNA_filtered.var_names

# %% [markdown]
# ## Step 1. computing the count matrix of genes-by-timepoints (RNA/ATAC)

# %%
# Group by 'timepoint' and calculate the mean of each gene across cells within each timepoint
df_avg_RNA = pd.DataFrame(adata_RNA.to_df()).groupby(adata_RNA.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_RNA.index.name = 'timepoint'

# %%
# # copy over the "dev_stage" from RNA to ATAC object
adata_ATAC.obs["dev_stage"] = adata_ATAC.obs_names.map(adata_RNA.obs["dev_stage"])
adata_ATAC

# %%
# Group by 'timepoint' and calculate the mean of each gene activity across cells within each timepoint
df_avg_ATAC = pd.DataFrame(adata_ATAC.to_df()).groupby(adata_ATAC.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_ATAC.index.name = 'timepoint'
print(df_avg_ATAC)

# %%
# Find intersection of genes between RNA and ATAC data
common_genes = df_avg_RNA.columns.intersection(df_avg_ATAC.columns)

# Subset both dataframes for common genes
df_RNA_common = df_avg_RNA[common_genes]
df_ATAC_common = df_avg_ATAC[common_genes]

# %%
df_RNA_common

# %%
df_ATAC_common


# %%
def process_matrices(df_RNA, df_ATAC):
    """
    Process RNA and ATAC matrices by transposing, renaming columns, and concatenating.
    
    Parameters:
    df_RNA (pd.DataFrame): RNA expression matrix
    df_ATAC (pd.DataFrame): ATAC expression matrix
    
    Returns:
    pd.DataFrame: Combined genes-by-features matrix
    """
    # 1. Transpose both matrices
    df_RNA_t = df_RNA.transpose()
    df_ATAC_t = df_ATAC.transpose()
    
    # 2. Rename columns to include data type
    df_RNA_t.columns = [f'{col}-RNA' for col in df_RNA_t.columns]
    df_ATAC_t.columns = [f'{col}-ATAC' for col in df_ATAC_t.columns]
    
    # 3. Concatenate the matrices horizontally
    result = pd.concat([df_RNA_t, df_ATAC_t], axis=1)
    
    return result

# Example usage:
# Assuming your data is already in DataFrames df_RNA_common and df_ATAC_common
# result_df = process_matrices(df_RNA_common, df_ATAC_common)

# To verify the results:
def print_matrix_info(df):
    """Helper function to print information about the matrix"""
    print("Shape:", df.shape)
    print("\nFirst few column names:")
    print(df.columns[:6])
    print("\nFirst few rows:")
    print(df.head(3))


# %%
# concatenate the count matrices to make a genes-by-features matrix
result_df = process_matrices(df_RNA_common, df_ATAC_common)

# Check the results
print_matrix_info(result_df)

# %%
result_df

# %%
# define an adata object 
adata = sc.AnnData(X=result_df)
adata.obs_names = result_df.index
adata.var_names = result_df.columns

adata

# %%
# Create observation metadata
adata.var['timepoint'] = [col.split('-')[0] for col in adata.var_names]
adata.var['data_type'] = [col.split('-')[1] for col in adata.var_names]

# %%
# save the averaged raw counts
adata.layers["raw"] = adata.X.copy()


# %%
# utility functions
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import anndata as ad


def create_adata_and_umap(result_df, norm_method='zscore', n_pcs=50, n_neighbors=15, min_dist=0.5):
    """
    Create AnnData object with genes as observations and generate UMAP.
    
    Parameters:
    result_df (pd.DataFrame): DataFrame with genes as rows and timepoint-datatypes as columns
    norm_method (str): Normalization method ('zscore', 'robust', 'minmax', 'log', or 'none')
    n_pcs (int): Number of principal components to use
    n_neighbors (int): Number of neighbors for UMAP
    min_dist (float): Minimum distance for UMAP
    
    Returns:
    AnnData: Processed AnnData object with UMAP coordinates
    """
    # Create AnnData object (no transpose needed now)
    adata = ad.AnnData(X=result_df)
    
    # Set observations (genes) and variables (timepoints)
    adata.obs_names = result_df.index  # Genes as observations
    adata.var_names = result_df.columns  # Timepoints as variables
    
    # Add variable metadata
    adata.var['timepoint'] = [col.split('-')[0] for col in adata.var_names]
    adata.var['data_type'] = [col.split('-')[1] for col in adata.var_names]
    
    # Store raw data
    adata.layers['raw'] = adata.X.copy()
    
    # Preprocess based on selected method
    if norm_method == 'zscore':
        # Z-score normalization (standardization) across timepoints
        scaler = StandardScaler()
        adata.X = scaler.fit_transform(adata.X)
        
    elif norm_method == 'robust':
        # Robust scaling (less sensitive to outliers)
        scaler = RobustScaler()
        adata.X = scaler.fit_transform(adata.X)
        
    elif norm_method == 'minmax':
        # MinMax scaling to [0,1] range
        scaler = MinMaxScaler()
        adata.X = scaler.fit_transform(adata.X)
        
    elif norm_method == 'log':
        # Log transformation (adding small constant to handle zeros)
        adata.X = np.log1p(adata.X)
        
    elif norm_method != 'none':
        raise ValueError("Invalid normalization method")
    
    # Run PCA
    sc.tl.pca(adata, n_comps=min(n_pcs, min(adata.X.shape)-1))
    
    # Generate UMAP
    sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, len(adata.obs_names)-1))
    sc.tl.umap(adata, min_dist=min_dist)
    
    return adata

def plot_umap(adata, save_path=None, gene_list=None):
    """
    Create UMAP plot with gene highlights option
    
    Parameters:
    adata (AnnData): Processed AnnData object with UMAP coordinates
    save_path (str, optional): Path to save the plot
    gene_list (list, optional): List of genes to highlight
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Base plot with all genes
    ax.scatter(adata.obsm['X_umap'][:,0], 
              adata.obsm['X_umap'][:,1], 
              c='gray', 
              alpha=0.5,
              s=10)
    
    # Highlight specific genes if provided
    if gene_list:
        mask = adata.obs_names.isin(gene_list)
        highlight = ax.scatter(adata.obsm['X_umap'][mask,0], 
                             adata.obsm['X_umap'][mask,1], 
                             c='red', 
                             s=50)
        
        # Add labels for highlighted genes
        for gene in gene_list:
            if gene in adata.obs_names:
                idx = adata.obs_names.get_loc(gene)
                ax.annotate(gene, 
                          (adata.obsm['X_umap'][idx,0], 
                           adata.obsm['X_umap'][idx,1]))
    
    ax.set_title('UMAP of Genes')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_patterns(adata, gene_list=None):
    """
    Analyze expression patterns for specific genes
    
    Parameters:
    adata (AnnData): Processed AnnData object
    gene_list (list, optional): List of genes to analyze
    """
    if gene_list is None:
        # If no genes specified, take a few random genes
        gene_list = list(np.random.choice(adata.obs_names, 5))
    
    # Get the raw data for these genes
    data = pd.DataFrame(adata.layers['raw'][adata.obs_names.isin(gene_list)],
                       index=gene_list,
                       columns=adata.var_names)
    
    # Create separate plots for RNA and ATAC
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot RNA patterns
    rna_data = data.filter(like='RNA')
    rna_data.columns = [col.split('-')[0] for col in rna_data.columns]
    rna_data.T.plot(ax=ax1, marker='o')
    ax1.set_title('RNA Expression Patterns')
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('Expression')
    
    # Plot ATAC patterns
    atac_data = data.filter(like='ATAC')
    atac_data.columns = [col.split('-')[0] for col in atac_data.columns]
    atac_data.T.plot(ax=ax2, marker='o')
    ax2.set_title('ATAC Signal Patterns')
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel('Signal')
    
    plt.tight_layout()
    plt.show()
    
    return data

# Example usage
def process_data(result_df, genes_of_interest=None):
    """
    Process the data and create visualizations
    
    Parameters:
    result_df (pd.DataFrame): Combined dataframe
    genes_of_interest (list, optional): List of genes to highlight
    """
    # Create AnnData object and generate UMAP
    adata = create_adata_and_umap(result_df, norm_method='robust')
    
    # Create visualization
    plot_umap(adata, gene_list=genes_of_interest)
    
    # Analyze patterns for specific genes
    if genes_of_interest:
        patterns = analyze_patterns(adata, genes_of_interest)
    
    return adata


# %%
def compute_gene_zscores(adata, copy=False):
    """
    Compute z-scores for each gene (row) across timepoints (columns).
    
    Parameters:
    adata (AnnData): AnnData object with genes as rows and timepoints as columns
    copy (bool): Whether to return a new AnnData object or modify in place
    
    Returns:
    AnnData: AnnData object with z-scored data in .X and original data in .layers['raw']
    """
    if copy:
        adata = adata.copy()
    
    # Store raw data
    adata.layers['raw'] = adata.X.copy()
    
    # Compute z-scores for each gene (row)
    means = np.mean(adata.X, axis=1, keepdims=True)
    stds = np.std(adata.X, axis=1, keepdims=True, ddof=1)  # ddof=1 for sample standard deviation
    
    # Handle cases where std might be 0
    stds[stds == 0] = 1.0
    
    # Compute z-scores
    adata.X = (adata.X - means) / stds
    
    # Add layer with z-scores
    adata.layers['z_scored'] = adata.X.copy()
    
    return adata


# %%
adata_zscore = create_adata_and_umap(result_df, norm_method='zscore')
plot_umap(adata_zscore)

# %%
# If you want to highlight specific genes
genes_of_interest = ['myf5', 'tbx16', 'myl1','hbbe3']
adata = process_data(result_df, genes_of_interest=genes_of_interest)


# %%

# %%
adata.var_names

# %%
adata_zscore = compute_gene_zscores(adata)
adata_zscore

# %%
sc.tl.pca(adata, svd_solver="arpack")
sc.pl.pca(adata, color=["0somites-RNA","30somites-RNA"])

# %%
sc.pl.pca_variance_ratio(adata, log=True)

# %%
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# %%
# sc.tl.paga(adata)
# sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata, min_dist=0.3)

# %%
sc.pl.umap(adata, color=["0somites-RNA","30somites-RNA"])

# %%
# subset the adata for the top 3000 highly variable genes
adata_sub = adata[adata.obs_names.isin(list_hvg_RNA)]
adata_sub

# %%
sc.tl.pca(adata_sub, svd_solver="arpack")
sc.pl.pca(adata_sub, color=["0somites-RNA","30somites-RNA"])
sc.pl.pca_variance_ratio(adata_sub, log=True)

sc.pp.neighbors(adata_sub, n_neighbors=10, n_pcs=10)
sc.tl.umap(adata_sub, min_dist=0.1)


# %%
sc.pl.umap(adata_sub, color=["0somites-RNA","30somites-RNA"])

# %%

# %%

# %%

# %%

# %%

# %%
# Create a new DataFrame to store correlation coefficients for each gene
correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# Compute Pearson correlation coefficient for each gene across timepoints
for gene in common_genes:
    correlation_df.loc[gene, 'correlation'] = df_RNA_common[gene].corr(df_ATAC_common[gene])

print(correlation_df)

# %%
# Drop NaN values from correlation dataframe, if any
correlation_df = correlation_df.dropna()

# Determine the top 5% and bottom 5% thresholds
top_threshold = correlation_df['correlation'].quantile(0.95)
bottom_threshold = correlation_df['correlation'].quantile(0.05)

# Get the top 5% and bottom 5% of the correlation coefficients
top_5_percent = correlation_df[correlation_df['correlation'] >= top_threshold]
bottom_5_percent = correlation_df[correlation_df['correlation'] <= bottom_threshold]

# Display the results
print("Top 5% Correlations:")
print(top_5_percent)

print("\nBottom 5% Correlations:")
print(bottom_5_percent)

# %%

# %%

# %% [markdown]
# ## check the normalization methods (across genes)

# %%
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import seaborn as sns

def normalize_genes(df, method='robust', plot=True, genes_list=None):
    """
    Normalize gene expression data using various methods.
    
    Parameters:
    df (pd.DataFrame): DataFrame with genes as columns and timepoints as rows
    method (str): Normalization method ('robust', 'minmax', 'percent_max', 'zscore', 'log')
    plot (bool): Whether to plot comparison of original vs normalized values
    genes_list (list, optional): List of specific genes to plot. If None, random genes will be selected
    
    Returns:
    pd.DataFrame: Normalized data
    """
    # Store original data
    df_orig = df.copy()
    
    if method == 'robust':
        scaler = RobustScaler()
        df_norm = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns
        )
        
    elif method == 'minmax':
        scaler = MinMaxScaler()
        df_norm = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns
        )
        
    elif method == 'percent_max':
        df_norm = df.apply(lambda x: (x / x.max()) * 100)
        
    elif method == 'zscore':
        df_norm = df.apply(lambda x: (x - x.mean()) / x.std())
        
    elif method == 'log':
        df_norm = np.log1p(df)
        
    else:
        raise ValueError("Invalid normalization method")
    
    if plot:
        plot_normalization_comparison(df_orig, df_norm, method, genes_list=genes_list)
    
    return df_norm

def plot_normalization_comparison(df_orig, df_norm, method, genes_list=None, n_genes=5, figsize=(15, 10)):
    """
    Plot comparison of original vs normalized values.
    
    Parameters:
    df_orig (pd.DataFrame): Original data
    df_norm (pd.DataFrame): Normalized data
    method (str): Normalization method used
    genes_list (list, optional): List of specific genes to plot
    n_genes (int): Number of random genes to plot if genes_list is None
    figsize (tuple): Figure size (width, height)
    """
    # Select genes to plot
    if genes_list is not None:
        # Verify all genes exist in the data
        missing_genes = [gene for gene in genes_list if gene not in df_orig.columns]
        if missing_genes:
            print(f"Warning: The following genes were not found: {missing_genes}")
        # Use only genes that exist in the data
        genes = [gene for gene in genes_list if gene in df_orig.columns]
        if not genes:
            raise ValueError("None of the specified genes were found in the data")
    else:
        genes = np.random.choice(df_orig.columns, n_genes)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Color palette for consistent colors across plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(genes)))
    
    # Plot original values
    for gene, color in zip(genes, colors):
        axes[0].plot(df_orig.index, df_orig[gene], 'o-', label=gene, color=color)
    axes[0].set_title('Original Values')
    axes[0].set_xlabel('Timepoint')
    axes[0].set_ylabel('Expression')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot normalized values
    for gene, color in zip(genes, colors):
        axes[1].plot(df_norm.index, df_norm[gene], 'o-', label=gene, color=color)
    axes[1].set_title(f'Normalized Values ({method})')
    axes[1].set_xlabel('Timepoint')
    axes[1].set_ylabel('Normalized Expression')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def compare_normalizations(df, genes_list=None):
    """
    Compare different normalization methods on the same data.
    
    Parameters:
    df (pd.DataFrame): Original data
    genes_list (list, optional): List of specific genes to compare
    """
    methods = ['robust', 'minmax', 'percent_max', 'zscore', 'log']
    normalized_dfs = {}
    
    # Apply each normalization method
    for method in methods:
        normalized_dfs[method] = normalize_genes(df, method, plot=False)
    
    # Select genes for comparison
    if genes_list is not None and len(genes_list) > 0:
        genes = [gene for gene in genes_list if gene in df.columns]
        if not genes:
            raise ValueError("None of the specified genes were found in the data")
    else:
        genes = [np.random.choice(df.columns)]
    
    # Plot comparison for each gene
    for gene in genes:
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes = axes.flatten()
        
        # Plot original data
        axes[0].plot(df.index, df[gene], 'o-')
        axes[0].set_title('Original')
        axes[0].set_ylabel(gene)
        
        # Plot each normalization
        for i, method in enumerate(methods, 1):
            axes[i].plot(df.index, normalized_dfs[method][gene], 'o-')
            axes[i].set_title(method)
            axes[i].set_ylabel(gene)
        
        plt.suptitle(f'Comparison of Normalization Methods for Gene: {gene}')
        plt.tight_layout()
        plt.show()


# %%
def compare_normalizations_single_plot(df, genes_list=None, n_random_genes=5, figsize=(15, 5)):
    """
    Compare different normalization methods for each gene in a single plot.
    
    Parameters:
    df (pd.DataFrame): Original data
    genes_list (list, optional): List of specific genes to compare
    n_random_genes (int): Number of random genes to plot if genes_list is None
    figsize (tuple): Base figure size (will be adjusted based on number of genes)
    """
    # Define normalization methods
    methods = {
        'original': lambda x: x,
        'robust': lambda x: RobustScaler().fit_transform(x.reshape(-1, 1)).flatten(),
        'minmax': lambda x: MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten(),
        'percent_max': lambda x: (x / x.max()) * 100,
        'zscore': lambda x: (x - x.mean()) / x.std(),
        'log': lambda x: np.log1p(x)
    }
    
    # Select genes to plot
    if genes_list is not None:
        genes = [gene for gene in genes_list if gene in df.columns]
        if not genes:
            raise ValueError("None of the specified genes were found in the data")
    else:
        genes = np.random.choice(df.columns, n_random_genes).tolist()
    
    # Calculate number of rows needed
    n_rows = (len(genes) + 2) // 3  # 3 genes per row
    n_cols = min(3, len(genes))
    
    # Adjust figure size based on number of genes
    fig_height = figsize[1] * n_rows
    fig_width = figsize[0] * (n_cols/3)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Color palette for methods
    colors = sns.color_palette("husl", len(methods))
    
    # Plot each gene
    for i, gene in enumerate(genes):
        if i < len(axes):  # Make sure we don't exceed number of subplots
            ax = axes[i]
            gene_data = df[gene]
            
            # Plot each normalization method
            for (method_name, method_func), color in zip(methods.items(), colors):
                normalized_data = method_func(gene_data.values)
                ax.plot(df.index, normalized_data, 'o-', label=method_name, color=color)
            
            ax.set_title(f'{gene}', pad=10)
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Expression')
            
            # Rotate x-axis labels if they're strings
            if isinstance(df.index[0], str):
                ax.tick_params(axis='x', rotation=45)
            
            # Add legend to first plot only
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    return {gene: {method: method_func(df[gene].values) 
                  for method, method_func in methods.items()}
            for gene in genes}

def calculate_summary_stats(normalized_data):
    """
    Calculate summary statistics for each normalization method.
    
    Parameters:
    normalized_data (dict): Dictionary of normalized values
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    stats = {}
    for gene in normalized_data:
        for method in normalized_data[gene]:
            data = normalized_data[gene][method]
            stats[f"{gene}_{method}"] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data)
            }
    
    return pd.DataFrame(stats).T

# Function to get normalized values for further analysis
def get_normalized_values(df, genes_list, method='robust'):
    """
    Get normalized values for specific genes using a chosen method.
    
    Parameters:
    df (pd.DataFrame): Original data
    genes_list (list): List of genes to normalize
    method (str): Normalization method
    
    Returns:
    pd.DataFrame: Normalized values for specified genes
    """
    methods = {
        'robust': lambda x: RobustScaler().fit_transform(x.reshape(-1, 1)).flatten(),
        'minmax': lambda x: MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten(),
        'percent_max': lambda x: (x / x.max()) * 100,
        'zscore': lambda x: (x - x.mean()) / x.std(),
        'log': lambda x: np.log1p(x)
    }
    
    if method not in methods:
        raise ValueError(f"Method must be one of {list(methods.keys())}")
    
    normalized_df = pd.DataFrame(index=df.index)
    for gene in genes_list:
        if gene in df.columns:
            normalized_df[gene] = methods[method](df[gene].values)
    
    return normalized_df


# %%
# For ATAC data
# df_ATAC_normalized = normalize_genes(df_ATAC_common, method='robust')

# Compare all methods for a better understanding
compare_normalizations(df_ATAC_common, genes_list = list_genes)

# pick a list of genes (of interest)
list_genes = ["myf5","tbx16","myl1","hbbe3"]

# Normalize and plot specific genes
df_normalized = normalize_genes(df_ATAC_common, 
                              method='zscore', 
                              plot=True, 
                              genes_list=list_genes)

# Compare all normalization methods for specific genes
# compare_normalizations(df_ATAC_common, genes_list=list_genes)


# %%
df_ATAC_common

# %%
# Group by 'timepoint' and calculate the mean of each gene activity across cells within each timepoint
df_avg_ATAC = pd.DataFrame(adata_ATAC.to_df()).groupby(adata_ATAC.obs["dev_stage"]).mean()
df_sem_ATAC = pd.DataFrame(adata_ATAC.to_df()).groupby(adata_ATAC.obs["dev_stage"]).sem()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_ATAC.index.name = 'timepoint'
df_sem_ATAC.index.name = 'timepoint'

print(df_avg_ATAC)
# print(df_sem_ATAC)

# %%
# Group by 'timepoint' and calculate the mean of each gene activity across cells within each timepoint
df_avg_RNA = pd.DataFrame(adata_RNA.to_df()).groupby(adata_RNA.obs["dev_stage"]).mean()
df_sem_RNA = pd.DataFrame(adata_RNA.to_df()).groupby(adata_RNA.obs["dev_stage"]).sem()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_RNA.index.name = 'timepoint'
df_sem_RNA.index.name = 'timepoint'

print(df_avg_RNA)
print(df_sem_RNA)

# %%

# %%
# Assume df_RNA and df_RNA_sem are your dataframes

# Calculate weights
weights = 1 / (df_RNA_sem ** 2)

# Calculate weighted mean and std for each gene
weighted_mean = (df_RNA * weights).sum(axis=1) / weights.sum(axis=1)
weighted_std = np.sqrt(((weights * (df_RNA.T - weighted_mean).T ** 2).sum(axis=1)) / weights.sum(axis=1))

# Perform weighted z-score normalization
df_normalized = ((df_RNA.T - weighted_mean) / weighted_std).T

# %%

# %%

# %%

# %% [markdown]
# ## Plot the average gene expression dynamics (RNA) over real-time (averaged across all cells)

# %%
# genes_to_plot = ['tbxta', 'sox2', 'myf5', 'pax6a', 'meox1']
# define the gene
list_genes = ["myf5","sox2","tbxta","myog",
              "meox1","tbx16","hes6",'pax6a',"en2a",
              "hbbe1.1","hbae3","hbbe3","myl1"]

# %%
# subset the adata_RNA for the list of genes (this is to reduce the computing resource/time)
adata_RNA_sub = adata_RNA[:, adata_RNA.var_names.isin(list_genes)]
adata_RNA_sub

# %%
# create a dataframe of cells-by-gene.activity (Signac)
count_matrice = pd.DataFrame(adata_RNA_sub.X.todense(),
                             index=adata_RNA_sub.obs.index,
                             columns=adata_RNA_sub.var_names)
count_matrice.head()

# %%
# transfer the "dataset" category labels to count_matrice df
count_matrice["dataset"] = adata_RNA_sub.obs["dataset"]
count_matrice["annotation_ML_coarse"] = adata_RNA_sub.obs["annotation_ML_coarse"]
count_matrice["dev_stage"] = adata_RNA_sub.obs["dev_stage"]

# %%
# add the "timepoints" category for the column

# define the dictionary for mapping
dict_timepoints = {"TDR126":"0somites",
                   "TDR127":"5somites",
                   "TDR128":"10somites",
                   "TDR118":"15somites",
                   "TDR119":"15somites",
                   "TDR125":"20somites",
                   "TDR124":"30somites"}

# map the "dataset" to "timepoints"
count_matrice["timepoints"] = count_matrice["dataset"].map(dict_timepoints)
count_matrice.head()

# %%
count_matrice.timepoints.value_counts()

# %%
# compute the averaged gene expression

# Define numeric columns (gene expression columns) by selecting columns with float/int data types
numeric_columns = count_matrice.select_dtypes(include=[np.number]).columns.tolist()

# Compute the mean gene expression level across all cells per 'timepoints'
timepoints_by_genes = count_matrice.groupby('timepoints')[numeric_columns].mean()

# Display the result
print(timepoints_by_genes)

# %%
# compute the "standard errors" gene.activity across all cells per "timepoint" (TDR118 and TDR119 will be merged here)
timepoints_by_genes_sem = count_matrice.groupby('timepoints')[numeric_columns].sem()
timepoints_by_genes_sem.head()


# %%
# numeric_timepoints to re-order the rows (timepoints_by_genes)
timepoints_by_genes['numeric_timepoints'] = timepoints_by_genes.index.str.extract('(\d+)').astype(int).values
timepoints_by_genes['numeric_timepoints']

# Sort by the numeric timepoints to ensure correct order in plot
timepoints_by_genes_sorted = timepoints_by_genes.sort_values('numeric_timepoints')
timepoints_by_genes_sorted

# %%
# numeric_timepoints to re-order the rows (timepoints_by_genes_sem)
timepoints_by_genes_sem['numeric_timepoints'] = timepoints_by_genes_sem.index.str.extract('(\d+)').astype(int).values
timepoints_by_genes_sem['numeric_timepoints']

# Sort by the numeric timepoints to ensure correct order in plot
timepoints_by_genes_sem_sorted = timepoints_by_genes_sem.sort_values('numeric_timepoints')
timepoints_by_genes_sem_sorted

# %%
timepoints_by_genes_sorted.index

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/RNA_ATAC_dynamics/"

# %%
for gene in list_genes:
    # generate a plot
    plt.figure(figsize=(4, 3))
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 timepoints_by_genes_sorted[gene], 
                 timepoints_by_genes_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    # plt.title('Scatter Plot of log-normalized counts over Timepoints')
    plt.xlabel('time (somites)')
    plt.ylabel('averaged gene expression (RNA, log-norm)')
    plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(False)
    plt.savefig(figpath + f"RNA_timepoints_{gene}.pdf")
    plt.show()

# %%
# Plotting the gene.activity dynamics (0-30 somites)
# define the gene
gene = "myf5"

# generate a plot
plt.figure(figsize=(4, 3))
plt.errorbar(timepoints_by_genes_sorted.index, 
             timepoints_by_genes_sorted[gene], 
             timepoints_by_genes_sem_sorted[gene],
             fmt='o', capsize=2, linestyle='None', 
             label=gene, markersize=2, elinewidth=1, capthick=1)
# plt.title('Scatter Plot of log-normalized counts over Timepoints')
plt.xlabel('time (somites)')
plt.ylabel('averaged gene expression (RNA, log-norm)')
plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
plt.legend()
plt.grid(False)
# plt.savefig(figpath + "gene_activity_score_myf5.pdf")
plt.show()

# %%
# # Plotting the gene.activity dynamics (0-30 somites)
# # define the gene
# gene = "myf5"

# # generate a plot
# plt.figure(figsize=(4, 3))
# plt.errorbar([0, 5, 10, 15, 20, 30], 
#              timepoints_by_genes_sorted[gene], 
#              timepoints_by_genes_sem_sorted[gene],
#              fmt='o', capsize=5, linestyle='None', 
#              label=gene, markersize=8, elinewidth=2, capthick=2)
# # plt.title('Scatter Plot of log-normalized counts over Timepoints')
# plt.xlabel('time (somites)')
# plt.ylabel('averaged gene expression (RNA, log-norm)')
# # plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
# plt.xticks([0, 5, 10, 15, 20, 30])
# plt.legend()
# plt.grid(False)
# plt.savefig(figpath + "RNA_timepoints_myf5.pdf")
# plt.show()

# %%
# Plotting the gene.activity dynamics (0-30 somites)
# define the gene
list_genes = ["myf5","sox2","tbxta","myog",
              "meox1","tbx16","hes6",
              "hbbe1.1","hbae3","hbbe3"]

# generate a plot
plt.figure(figsize=(10, 6))

for gene in list_genes:
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 timepoints_by_genes_sorted[gene], 
                 timepoints_by_genes_sem_sorted[gene],
                 fmt='o', capsize=5, linestyle='None', 
                 label=gene, markersize=8, elinewidth=2, capthick=2)
    
# plt.title('Scatter Plot of log-normalized counts over Timepoints')
plt.xlabel('time (somites)')
plt.ylabel('averaged gene expression (RNA, log-norm)')
plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
# Move the legend outside of the plot to the right
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect parameter to make room for the legend
plt.grid(False)
# plt.savefig(figpath + "RNA_timepoints_genesets.pdf")
plt.show()

# %%
# Plotting the gene.activity dynamics (0-30 somites)
# define the gene
list_genes = ["myf5","sox2","tbxta","myog",
              "meox1","tbx16","hes6",
              "hbbe1.1","hbae3","hbbe3"]

# generate a plot
plt.figure(figsize=(10, 6))

for gene in list_genes:
    # Normalization step: divide by the max value to scale to 1
    normalized_values = timepoints_by_genes_sorted[gene] / timepoints_by_genes_sorted[gene].max()
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 normalized_values, 
                 yerr=timepoints_by_genes_sem_sorted[gene] / timepoints_by_genes_sorted[gene].max(), 
                 fmt='o', capsize=5, linestyle='None', 
                 label=gene, markersize=8, elinewidth=2, capthick=2)
    
# plt.title('Scatter Plot of log-normalized counts over Timepoints')
plt.xlabel('time (somites)')
plt.ylabel('averaged gene expression (RNA, normalized)')
plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
# Move the legend outside of the plot to the right
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect parameter to make room for the legend
plt.grid(False)
# plt.savefig(figpath + "gene_activity_score_genesets_normalized.pdf")
plt.show()

# %% [markdown]
# ## Step 2. ATAC dynamics using cells-by-gene.activity (Signac)
#
# - The gene.activity scores were computed by Signac for individual timepoints  (**using the "peaks_merged" for individual timepoints**, not "peaks_integrated" after the merge/integration over timepoints)
# - We created a concatenated adata (all cells-by-gene.activity, with adata.obs having "dataset" identifiers)
#

# %% [markdown]
# ### averaged gene.activity dynamics along the real time

# %%
# genes_to_plot = ['tbxta', 'sox2', 'myf5', 'pax6a', 'meox1']
# define the gene
list_genes = ["myf5","sox2","tbxta","myog",
              "meox1","tbx16","hes6",'pax6a',"en2a",
              "hbbe1.1","hbae3","hbbe3","myl1"]

# %%
# subset the adata_RNA for the list of genes (this is to reduce the computing resource/time)
adata_ATAC_sub = adata_ATAC[:, adata_ATAC.var_names.isin(list_genes)]
adata_ATAC_sub

# %%
# create a dataframe of cells-by-gene.activity (Signac)
count_matrice_atac = pd.DataFrame(adata_ATAC_sub.X.todense(),
                             index=adata_ATAC_sub.obs.index,
                             columns=adata_ATAC_sub.var_names)
count_matrice_atac.head()

# %%
# transfer the "dataset" category labels to count_matrice df
count_matrice_atac["dataset"] = adata_ATAC_sub.obs["dataset"]
count_matrice_atac["annotation_ML_coarse"] = adata_RNA_sub.obs["annotation_ML_coarse"]
count_matrice_atac["dev_stage"] = adata_RNA_sub.obs["dev_stage"]

# %%
# add the "timepoints" category for the column

# define the dictionary for mapping
dict_timepoints = {"TDR126":"0somites",
                   "TDR127":"5somites",
                   "TDR128":"10somites",
                   "TDR118":"15somites",
                   "TDR119":"15somites",
                   "TDR125":"20somites",
                   "TDR124":"30somites"}

# map the "dataset" to "timepoints"
count_matrice_atac["timepoints"] = count_matrice_atac["dataset"].map(dict_timepoints)
count_matrice_atac.head()

# %%
# compute the averaged gene expression

# Define numeric columns (gene expression columns) by selecting columns with float/int data types
numeric_columns = count_matrice_atac.select_dtypes(include=[np.number]).columns.tolist()

# Compute the mean gene expression level across all cells per 'timepoints'
timepoints_by_gene_activity = count_matrice_atac.groupby('timepoints')[numeric_columns].mean()

# Display the result
print(timepoints_by_gene_activity)

# %%
# compute the "standard errors" gene.activity across all cells per "timepoint" (TDR118 and TDR119 will be merged here)
timepoints_by_gene_activity_sem = count_matrice_atac.groupby('timepoints')[numeric_columns].sem()
timepoints_by_gene_activity_sem.head()


# %%
# numeric_timepoints to re-order the rows (timepoints_by_genes)
timepoints_by_gene_activity['numeric_timepoints'] = timepoints_by_gene_activity.index.str.extract('(\d+)').astype(int).values
timepoints_by_gene_activity['numeric_timepoints']

# Sort by the numeric timepoints to ensure correct order in plot
timepoints_by_gene_activity_sorted = timepoints_by_gene_activity.sort_values('numeric_timepoints')
timepoints_by_gene_activity_sorted

# %%
# numeric_timepoints to re-order the rows (timepoints_by_genes_sem)
timepoints_by_gene_activity_sem['numeric_timepoints'] = timepoints_by_gene_activity_sem.index.str.extract('(\d+)').astype(int).values
timepoints_by_gene_activity_sem['numeric_timepoints']

# Sort by the numeric timepoints to ensure correct order in plot
timepoints_by_gene_activity_sem_sorted = timepoints_by_gene_activity_sem.sort_values('numeric_timepoints')
timepoints_by_gene_activity_sem_sorted

# %%
for gene in list_genes:
    # generate a plot
    plt.figure(figsize=(4, 3))
    plt.errorbar(timepoints_by_gene_activity_sorted.index, 
                 timepoints_by_gene_activity_sorted[gene], 
                 timepoints_by_gene_activity_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    # plt.title('Scatter Plot of log-normalized counts over Timepoints')
    plt.xlabel('time (somites)')
    plt.ylabel('averaged gene expression (gene activity, log-norm)')
    plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(False)
    plt.savefig(figpath + f"ATAC_timepoints_{gene}.pdf")
    plt.show()

# %% [markdown]
# ## Step 3. visualization
#
# - plotting the RNA and ATAC (gene.activity) in the same scatter plot (for each gene)
#
# - plotting the correlation between RNA and ATAC (gene.activity) for each gene (color the dots with the viridis color palette for timepoints).

# %%
for gene in list_genes:
    # generate a plot
    plt.figure(figsize=(4, 3))
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 timepoints_by_genes_sorted[gene], 
                 timepoints_by_genes_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    plt.errorbar(timepoints_by_gene_activity_sorted.index, 
                 timepoints_by_gene_activity_sorted[gene], 
                 timepoints_by_gene_activity_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    plt.title(f'{gene}')
    plt.xlabel('time (somites)')
    plt.ylabel('averaged gene expression (log-norm)')
    plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    plt.xticks(rotation=90)
    plt.legend(["RNA", "ATAC"])
    plt.grid(False)
    plt.savefig(figpath + f"RNA_ATAC_timepoints_{gene}.pdf")
    plt.show()


# %%
# Just to monitor the trend over time, we can normalize each modality

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

for gene in list_genes:
    
#     # normalize the RNA
#     max_rna = np.max(timepoints_by_genes_sorted[gene])
#     rna_norm = timepoints_by_genes_sorted[gene]/max_rna
#     rna_sem_norm = timepoints_by_genes_sem_sorted[gene]/max_rna
    
#     # normalize the ATAC
#     max_atac = np.max(timepoints_by_gene_activity_sorted[gene])
#     atac_norm = timepoints_by_gene_activity_sorted[gene]/max_atac
#     atac_sem_norm = timepoints_by_gene_activity_sem_sorted[gene]/max_atac
    rna_norm = normalize(timepoints_by_genes_sorted[gene])
    rna_sem_norm = timepoints_by_genes_sem_sorted[gene] / (timepoints_by_genes_sorted[gene].max() - timepoints_by_genes_sorted[gene].min())
    atac_norm = normalize(timepoints_by_gene_activity_sorted[gene])
    atac_sem_norm = timepoints_by_gene_activity_sem_sorted[gene] / (timepoints_by_gene_activity_sorted[gene].max() - timepoints_by_gene_activity_sorted[gene].min())


    
    # generate a plot
    plt.figure(figsize=(4, 3))
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 rna_norm, 
                 rna_sem_norm,
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    plt.errorbar(timepoints_by_gene_activity_sorted.index, 
                 atac_norm, 
                 atac_sem_norm,
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    plt.title(f'{gene}')
    plt.xlabel('time (somites)')
    plt.ylabel('averaged gene expression (log-norm)')
    plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    plt.xticks(rotation=90)
    plt.legend(["RNA", "ATAC"])
    plt.grid(False)
    plt.savefig(figpath + f"RNA_ATAC_timepoints_{gene}_norm.pdf")
    plt.show()

# %% [markdown]
# ### correlation between RNA and ATAC

# %%
color_dict

timepoints_by_gene_activity_sorted.index.map(color_dict)

# %%
from scipy.stats import pearsonr

# plot the correlation between RNA and ATAC
for gene in list_genes:
    
    # define the color palette using the viridis
    colors = timepoints_by_gene_activity_sorted.index.map(color_dict)
    
    # define the x_vals and y_vals
    # Collect data points for correlation calculation
    x_vals = timepoints_by_gene_activity_sorted[gene]
    y_vals = timepoints_by_genes_sorted[gene]
    
    # Calculate correlation coefficient
    correlation, _ = pearsonr(x_vals, y_vals)
    
    # generate a plot
    plt.figure(figsize=(3, 3))
    plt.errorbar(x = timepoints_by_gene_activity_sorted[gene],
                 y = timepoints_by_genes_sorted[gene],
                 xerr=timepoints_by_gene_activity_sem_sorted[gene],
                 yerr=timepoints_by_genes_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1,)
                 # ecolor=colors, color=colors)
        
    # Add the correlation coefficient text to the plot
    plt.text(0.05, 0.95, f'r = {correlation:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.title(f'{gene}')
    plt.xlabel('gene activity (ATAC)')
    plt.ylabel('gene expression (RNA)')
    # plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    # plt.xticks(rotation=90)
    # plt.legend(["RNA", "ATAC"])
    plt.grid(False)
    plt.savefig(figpath + f"RNA_ATAC_corr_scatter_{gene}.pdf")
    plt.show()

# %%

# %% [markdown]
# ## [HOLD] Step 3 . Find patterns in the RNA/ATAC dynamics (real-time)
#
# - First, we will normalize the timepoints_by_genes dataframe for each gene's maximum gene.activity score (averaged)
# - Second, we will generate a heatmap for genes-by-timepoints
# - Lastly, we will perform hirarchical clustering on the heatmap to find the gene clusters that have similar ATAC dynamics (either increasing or decreasing over time).

# %%
# first, exclude the "numeric_timepoints" column before normalization
timepoints_by_genes_sorted.drop('numeric_timepoints', axis=1, inplace=True)

# Normalize each gene's expression levels to the maximum value for that gene
normalized_timepoints_by_genes = timepoints_by_genes_sorted.divide(timepoints_by_genes_sorted.max())
normalized_timepoints_by_genes.head()

# %%
# Replace any infinite values with NaN
timepoints_by_genes_sorted.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any rows or columns with NaN values if they can't be imputed
# normalized_timepoints_by_genes.dropna(axis=0, inplace=True)  # Drop rows with NaN
timepoints_by_genes_sorted.dropna(axis=1, inplace=True)  # Drop columns with NaN

# %%
# Replace any infinite values with NaN
normalized_timepoints_by_genes.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any rows or columns with NaN values if they can't be imputed
# normalized_timepoints_by_genes.dropna(axis=0, inplace=True)  # Drop rows with NaN
normalized_timepoints_by_genes.dropna(axis=1, inplace=True)  # Drop columns with NaN

# %%
normalized_timepoints_by_genes.T.head()

# %%
# Create a heatmap with hierarchical clustering
plt.figure(figsize=(12, 8))
sns.clustermap(normalized_timepoints_by_genes.T, method='average', metric='euclidean', cmap='viridis', standard_scale=0,
               row_cluster=True, col_cluster=False)
plt.show()

# %%
# Create a heatmap with hierarchical clustering
plt.figure(figsize=(12, 8))
g = sns.clustermap(normalized_timepoints_by_genes.T, method='average', metric='euclidean', cmap='viridis', standard_scale=0,
               row_cluster=True, col_cluster=False, yticklabels=False)

reordered_labels = normalized_timepoints_by_genes.T.index[g.dendrogram_row.reordered_ind].tolist()
use_labels = ["myf5","sox2","tbxta",
              "meox1","tbx16","hes6",
              "hbbe1.1","hbae3","hbbe3","rps16",
              "lrrc24","krt96"]
use_ticks = [reordered_labels.index(label) + .5 for label in use_labels]
g.ax_heatmap.set(yticks=use_ticks, yticklabels=use_labels)
g.savefig(figpath + "clustered_heatmap_gene_activity_score_timepoints_norm.png")
g.savefig(figpath + "clustered_heatmap_gene_activity_score_timepoints_norm.pdf")
plt.show()

# %% [markdown]
# ## Step 4. correlation between ATAC and RNA

# %%
# Group by 'timepoint' and calculate the mean of each gene across cells within each timepoint
df_avg_RNA = pd.DataFrame(adata_RNA.to_df()).groupby(adata_RNA.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_RNA.index.name = 'timepoint'
print(df_avg_RNA)

# %%
# # copy over the "dev_stage" from RNA to ATAC object
adata_ATAC.obs["dev_stage"] = adata_ATAC.obs_names.map(adata_RNA.obs["dev_stage"])
adata_ATAC

# %%
# Group by 'timepoint' and calculate the mean of each gene activity across cells within each timepoint
df_avg_ATAC = pd.DataFrame(adata_ATAC.to_df()).groupby(adata_ATAC.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_ATAC.index.name = 'timepoint'
print(df_avg_ATAC)

# %%
# Find intersection of genes between RNA and ATAC data
common_genes = df_avg_RNA.columns.intersection(df_avg_ATAC.columns)

# Subset both dataframes for common genes
df_RNA_common = df_avg_RNA[common_genes]
df_ATAC_common = df_avg_ATAC[common_genes]

# %%
# Create a new DataFrame to store correlation coefficients for each gene
correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# Compute Pearson correlation coefficient for each gene across timepoints
for gene in common_genes:
    correlation_df.loc[gene, 'correlation'] = df_RNA_common[gene].corr(df_ATAC_common[gene])

print(correlation_df)

# %%
# Drop NaN values from correlation dataframe, if any
correlation_df = correlation_df.dropna()

# Determine the top 5% and bottom 5% thresholds
top_threshold = correlation_df['correlation'].quantile(0.95)
bottom_threshold = correlation_df['correlation'].quantile(0.05)

# Get the top 5% and bottom 5% of the correlation coefficients
top_5_percent = correlation_df[correlation_df['correlation'] >= top_threshold]
bottom_5_percent = correlation_df[correlation_df['correlation'] <= bottom_threshold]

# Display the results
print("Top 5% Correlations:")
print(top_5_percent)

print("\nBottom 5% Correlations:")
print(bottom_5_percent)

# %%
# Convert the index to a list and save as a text file
top_genes_list = top_genes.tolist()

# Save the gene names as a text file using to_csv
with open('top_5_percent_genes.txt', 'w') as f:
    for gene in top_genes_list:
        f.write(f"{gene}\n")

# %%
# Convert the index to a list and save as a text file

bottom_genes_list = bottom_5_percent.index.tolist()

# Save the gene names as a text file using to_csv
with open('bottom_5_percent_genes.txt', 'w') as f:
    for gene in bottom_genes_list:
        f.write(f"{gene}\n")


# %%
def generate_permutation_without_self_match(genes):
    """Generate a permutation of gene names such that no gene is paired with itself."""
    while True:
        permuted_genes = np.random.permutation(genes)
        # Check if any gene is paired with itself; if not, return
        if not np.any(permuted_genes == genes):
            return permuted_genes


# Generate a valid permutation without any gene being paired with itself
shuffled_columns = generate_permutation_without_self_match(common_genes)
df_ATAC_shuffled = df_ATAC_common[shuffled_columns]

# Create a DataFrame to store correlation coefficients for non-pairs
null_correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# Compute Pearson correlation coefficient for each "non-pair" gene
for gene_rna, gene_atac in zip(common_genes, shuffled_columns):
    null_correlation_df.loc[gene_rna, 'correlation'] = df_RNA_common[gene_rna].corr(df_ATAC_shuffled[gene_atac])

# Drop NaN values if any were generated
null_correlation_df = null_correlation_df.dropna()

print(null_correlation_df)

# %%
plt.hist(correlation_df.correlation, bins=50, density=True, alpha=0.5)
plt.hist(null_correlation_df.correlation, bins=50, density=True, alpha=0.5)
plt.legend(["gene pairs", "null"])
plt.xlabel("pearson correlation")
plt.ylabel("density")
plt.grid(False)
plt.savefig("pearson_corr_hist.pdf")
plt.show()

# %% [markdown]
# ## highly variable genes (RNA expression level)

# %%
# Calculate highly variable genes in the RNA data
sc.pp.highly_variable_genes(adata_RNA, flavor='seurat', n_top_genes=2000)

# Subset the adata_RNA object to only include highly variable genes
adata_RNA_hvg = adata_RNA[:, adata_RNA.var['highly_variable']]

# %%
# Group by 'timepoint' and calculate the mean of each gene across cells within each timepoint
df_avg_RNA = pd.DataFrame(adata_RNA_hvg.to_df()).groupby(adata_RNA_hvg.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_RNA.index.name = 'timepoint'
print(df_avg_RNA)

# %%
# # copy over the "dev_stage" from RNA to ATAC object
adata_ATAC.obs["dev_stage"] = adata_ATAC.obs_names.map(adata_RNA.obs["dev_stage"])
adata_ATAC

# %%
# Group by 'timepoint' and calculate the mean of each gene activity across cells within each timepoint
df_avg_ATAC = pd.DataFrame(adata_ATAC.to_df()).groupby(adata_ATAC.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_ATAC.index.name = 'timepoint'
print(df_avg_ATAC)

# %%
# Find intersection of genes between RNA and ATAC data
common_genes = df_avg_RNA.columns.intersection(df_avg_ATAC.columns)

# Subset both dataframes for common genes
df_RNA_common = df_avg_RNA[common_genes]
df_ATAC_common = df_avg_ATAC[common_genes]

# %%
# Create a new DataFrame to store correlation coefficients for each gene
correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# Compute Pearson correlation coefficient for each gene across timepoints
for gene in common_genes:
    correlation_df.loc[gene, 'correlation'] = df_RNA_common[gene].corr(df_ATAC_common[gene])

print(correlation_df)

# %%
plt.hist(correlation_df.correlation, bins=50, density=True)
plt.grid(False)
plt.show()

# %%
# def generate_permutation_without_self_match(genes):
#     """Generate a permutation of gene names such that no gene is paired with itself."""
#     while True:
#         permuted_genes = np.random.permutation(genes)
#         # Check if any gene is paired with itself; if not, return
#         if not np.any(permuted_genes == genes):
#             return permuted_genes


# # Generate a valid permutation without any gene being paired with itself
# shuffled_columns = generate_permutation_without_self_match(common_genes)
# df_ATAC_shuffled = df_ATAC_common[shuffled_columns]

# # Create a DataFrame to store correlation coefficients for non-pairs
# null_correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# # Compute Pearson correlation coefficient for each "non-pair" gene
# for gene_rna, gene_atac in zip(common_genes, shuffled_columns):
#     null_correlation_df.loc[gene_rna, 'correlation'] = df_RNA_common[gene_rna].corr(df_ATAC_shuffled[gene_atac])

# # Drop NaN values if any were generated
# null_correlation_df = null_correlation_df.dropna()

# print(null_correlation_df)

# %%
import itertools
import pandas as pd

# Generate all possible non-pair combinations using itertools.product
all_pairs = list(itertools.product(common_genes, common_genes))

# Filter out self-pairs (gene_rna == gene_atac)
non_pairs = [(gene_rna, gene_atac) for gene_rna, gene_atac in all_pairs if gene_rna != gene_atac]

# # Create a DataFrame to store correlation coefficients for non-pairs
# null_correlation_df = pd.DataFrame(columns=['gene_rna', 'gene_atac', 'correlation'])

# # Compute Pearson correlation coefficient for each "non-pair" gene
# for gene_rna, gene_atac in non_pairs:
#     correlation = df_RNA_common[gene_rna].corr(df_ATAC_common[gene_atac])
#     null_correlation_df = null_correlation_df.append({
#         'gene_rna': gene_rna,
#         'gene_atac': gene_atac,
#         'correlation': correlation
#     }, ignore_index=True)

# # Drop NaN values if any were generated
# null_correlation_df = null_correlation_df.dropna()

# print(null_correlation_df)
# Create a list to collect correlation results
correlation_results = []

# Compute Pearson correlation coefficient for each "non-pair" gene
for gene_rna, gene_atac in non_pairs:
    correlation = df_RNA_common[gene_rna].corr(df_ATAC_common[gene_atac])
    correlation_results.append({
        'gene_rna': gene_rna,
        'gene_atac': gene_atac,
        'correlation': correlation
    })

# Convert the results list to a DataFrame
null_correlation_df = pd.DataFrame(correlation_results)

# Drop NaN values if any were generated
null_correlation_df = null_correlation_df.dropna()

print(null_correlation_df)


# %%
# Function to generate a permutation of gene names such that no gene is paired with itself
def generate_permutation_without_self_match(genes):
    """Generate a permutation of gene names such that no gene is paired with itself."""
    while True:
        permuted_genes = np.random.permutation(genes)
        # Check if any gene is paired with itself; if not, return
        if not np.any(permuted_genes == genes):
            return permuted_genes

# Number of repetitions
n_repeats = 5

# Create a DataFrame to accumulate correlation values for each gene across repetitions
correlation_accumulator = pd.DataFrame(index=common_genes, columns=range(n_repeats), dtype=float)

# Repeat the permutation and correlation calculation multiple times
for i in range(n_repeats):
    # Generate a valid permutation without any gene being paired with itself
    shuffled_columns = generate_permutation_without_self_match(common_genes)
    df_ATAC_shuffled = df_ATAC_common[shuffled_columns]

    # Compute Pearson correlation coefficient for each "non-pair" gene
    for gene_rna, gene_atac in zip(common_genes, shuffled_columns):
        correlation_accumulator.loc[gene_rna, i] = df_RNA_common[gene_rna].corr(df_ATAC_shuffled[gene_atac])

# Compute the average correlation per gene across repetitions
average_correlation_df = correlation_accumulator.mean(axis=1)

# Drop NaN values if any were generated
average_correlation_df = average_correlation_df.dropna()

# Convert to a DataFrame for better formatting if needed
null_correlation_df = pd.DataFrame(average_correlation_df, columns=['average_correlation'])

print(null_correlation_df)

# %%
null_correlation_df.average_correlation.mean()

# %%
plt.hist(correlation_df.correlation, bins=50, density=True, alpha=0.5, range=(-1, 1), rwidth=0.9)
plt.hist(null_correlation_df.average_correlation, bins=50, density=True, alpha=0.5, range=(-1, 1), rwidth=0.9)

plt.legend(["gene pairs", "null"])
# Draw vertical lines for the cutoffs
plt.axvline(x=top_threshold, color='r', linestyle='--', linewidth=2, label='Top 5% cutoff')
plt.axvline(x=mid_threshold, color='r', linestyle='--', linewidth=2, label='Top 75% cutoff')
plt.axvline(x=bottom_threshold, color='r', linestyle='--', linewidth=2, label='Top 50% cutoff')
plt.xlabel("pearson correlation")
plt.ylabel("density")
plt.grid(False)
plt.savefig("pearson_corr_hist_hvgs.pdf")
plt.show()

# %%
# Calculate the thresholds for the top 5% and bottom 5% correlations
top_threshold = correlation_df['correlation'].quantile(0.95)
mid_threshold = correlation_df['correlation'].quantile(0.75)
bottom_threshold = correlation_df['correlation'].quantile(0.60)

# Plot the histogram of correlation coefficients
plt.figure(figsize=(8, 6))
plt.hist(correlation_df['correlation'], bins=50, edgecolor='k', alpha=0.7)

# Draw vertical lines for the cutoffs
plt.axvline(x=top_threshold, color='r', linestyle='--', linewidth=2, label='Top 5% cutoff')
plt.axvline(x=mid_threshold, color='r', linestyle='--', linewidth=2, label='Top 75% cutoff')
plt.axvline(x=bottom_threshold, color='b', linestyle='--', linewidth=2, label='Top 50% cutoff')

# Add labels and legend
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.legend()
plt.grid(False)
plt.show()

# %%
# Drop NaN values from correlation dataframe, if any
correlation_df = correlation_df.dropna()

# Determine the top 5% and bottom 5% thresholds
top_threshold = correlation_df['correlation'].quantile(0.95)
mid_threshold = correlation_df['correlation'].quantile(0.75)
bottom_threshold = correlation_df['correlation'].quantile(0.50)

# Get the top 5% and bottom 5% of the correlation coefficients
top_5_percent = correlation_df[correlation_df['correlation'] >= top_threshold]
# Get the middle 50-75% of the correlation coefficients
mid_50_75_percent = correlation_df[(correlation_df['correlation'] <= mid_threshold) & (correlation_df['correlation'] >= bottom_threshold)]

# Display the results
print("Top 5% Correlations:")
print(top_5_percent)

print("\nTop 50-75% Correlations:")
print(mid_50_75_percent)

# %%
# Convert the index to a list and save as a text file
top_genes_list = top_5_percent.index.tolist()

# Save the gene names as a text file using to_csv
with open('top_5_percent_genes.txt', 'w') as f:
    for gene in top_genes_list:
        f.write(f"{gene}\n")

# %%
# Convert the index to a list and save as a text file
mid_50_75_genes_list = mid_50_75_percent.index.tolist()

# Save the gene names as a text file using to_csv
with open('mid_50_75_genes_list.txt', 'w') as f:
    for gene in mid_50_75_genes_list:
        f.write(f"{gene}\n")

# %%
correlation_df.correlation.mean()
