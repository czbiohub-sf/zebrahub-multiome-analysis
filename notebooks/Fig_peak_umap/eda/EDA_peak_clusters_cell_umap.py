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
#     display_name: sc_rapids
#     language: python
#     name: sc_rapids
# ---

# %% [markdown]
# ## EDA on peak clusters mapped back to cell UMAP
#
# - last updated: 7/4/2025
# - Goals:
# - For each peak cluster, check their enrichment profiles in cell UMAP.
#     - this is to check whether the peak clusters mark specific cell types or timepoints.
#     - make a module/functions to repeat this systematically (potentially in the web portal)
#     - write a function/script that can systematically "score" the specificity/localization (gini index?) of peak clusters enrichment in cell UMAP.

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
import os

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# %%
# figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')
# Set default DPI for saved figures
mpl.rcParams['savefig.dpi'] = 600

# %%
import logging
# Suppress INFO-level logs for the entire logger
logging.getLogger().setLevel(logging.WARNING)

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_cluster_projection_cell_umap/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad")
adata_peaks

# %%
# import the cells-by-peaks object
adata_atac = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad")
adata_atac

# %%
# filter out the 4 peaks that go beyond the chromosome end, or MT chromosomes
adata_atac = adata_atac[:, adata_atac.var_names.isin(adata_peaks.obs_names)]
adata_atac

# %%
rsc.get.anndata_to_GPU(adata_atac)

# %%
# normalize the counts in adata_atac
# adata_atac.X = adata_atac.layers["counts"] # already raw counts in the adata_atac.X
rsc.pp.normalize_total(adata_atac, target_sum=1e4)
adata_atac.layers["normalized"] = adata_atac.X.copy()


# %%
# a function that takes in the followin inputs and returns an output:
# inputs: adata_peaks, "leiden_unified", 
# output(s): adata_atac.obs[f"access_leiden_{clust_id}"] 

# %%
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_peak_cluster_scores(adata_ATAC, adata_peaks, cluster_key='leiden_unified', 
                               use_raw=False, copy=False):
    """
    Compute peak cluster scores for each cell by averaging accessibility 
    across peaks within each cluster.
    
    Parameters:
    -----------
    adata_ATAC : AnnData
        cell-by-peaks object with cell UMAP
    adata_peaks : AnnData  
        peaks-by-pseudobulk object with peak clusters
    cluster_key : str
        Key in adata_peaks.obs containing cluster labels
    use_raw : bool
        Whether to use raw or normalized data
    copy : bool
        Whether to return a copy
    
    Returns:
    --------
    adata_ATAC : AnnData
        Updated with peak cluster scores in .obs
    """
    
    if copy:
        adata_ATAC = adata_ATAC.copy()
    
    # Get the data matrix to use
    if use_raw and adata_ATAC.raw is not None:
        X = adata_ATAC.raw.X
    else:
        X = adata_ATAC.X
    
    # Ensure peak names match between objects
    peak_names_cells = adata_ATAC.var_names
    peak_names_clusters = adata_peaks.obs_names
    
    # Find common peaks
    common_peaks = peak_names_cells.intersection(peak_names_clusters)
    print(f"Found {len(common_peaks)} common peaks between objects")
    
    if len(common_peaks) == 0:
        raise ValueError("No common peaks found between adata_ATAC and adata_peaks")
    
    # Get indices for common peaks in both objects
    cell_peak_indices = [i for i, peak in enumerate(peak_names_cells) if peak in common_peaks]
    cluster_peak_indices = [i for i, peak in enumerate(peak_names_clusters) if peak in common_peaks]
    
    # Get cluster labels for common peaks
    cluster_labels = adata_peaks.obs[cluster_key].iloc[cluster_peak_indices]
    unique_clusters = cluster_labels.unique()
    
    print(f"Computing scores for {len(unique_clusters)} peak clusters...")
    
    # Compute cluster scores for each cell
    cluster_scores = {}
    
    for cluster in tqdm(unique_clusters, desc="Processing clusters"):
        # Get peak indices for this cluster
        cluster_mask = cluster_labels == cluster
        cluster_peak_idx = np.array(cell_peak_indices)[cluster_mask]
        
        if len(cluster_peak_idx) == 0:
            continue
            
        # Compute mean accessibility across cluster peaks for each cell
        if hasattr(X, 'toarray'):  # sparse matrix
            cluster_accessibility = X[:, cluster_peak_idx].toarray()
        else:  # dense matrix
            cluster_accessibility = X[:, cluster_peak_idx]
        
        # Average across peaks in the cluster
        mean_accessibility = np.mean(cluster_accessibility, axis=1)
        cluster_scores[f'peak_cluster_{cluster}'] = mean_accessibility
    
    # Add scores to adata_ATAC.obs
    for cluster_name, scores in cluster_scores.items():
        adata_ATAC.obs[cluster_name] = scores
    
    print(f"Added {len(cluster_scores)} peak cluster scores to adata_ATAC.obs")
    
    return adata_ATAC if copy else None

def plot_peak_cluster_umaps(adata_ATAC, cluster_prefix='peak_cluster_', 
                           n_cols=4, figsize_per_plot=(4, 3), 
                           save_prefix=None, cmap='viridis'):
    """
    Plot UMAP for all peak cluster scores
    
    Parameters:
    -----------
    adata_ATAC : AnnData
        Cell data with peak cluster scores
    cluster_prefix : str
        Prefix for peak cluster score columns
    n_cols : int
        Number of columns in the plot grid
    figsize_per_plot : tuple
        Figure size for each subplot
    save_prefix : str
        Prefix for saving plots (optional)
    cmap : str
        Colormap for plots
    """
    
    # Get all peak cluster score columns
    cluster_cols = [col for col in adata_ATAC.obs.columns if col.startswith(cluster_prefix)]
    
    if len(cluster_cols) == 0:
        print(f"No columns found with prefix '{cluster_prefix}'")
        return
    
    print(f"Plotting {len(cluster_cols)} peak cluster UMAPs...")
    
    # Calculate grid dimensions
    n_rows = (len(cluster_cols) + n_cols - 1) // n_cols
    
    # Create figure
    fig_width = n_cols * figsize_per_plot[0]
    fig_height = n_rows * figsize_per_plot[1]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each cluster
    for i, cluster_col in enumerate(cluster_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Extract cluster number/name for title
        cluster_name = cluster_col.replace(cluster_prefix, '')
        
        # Plot UMAP
        sc.pl.umap(adata_ATAC, color=cluster_col, ax=ax, show=False, 
                  cmap=cmap, title=f'Peak Cluster {cluster_name}')
        
    # Hide empty subplots
    for i in range(len(cluster_cols), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_peak_cluster_umaps.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_top_peak_clusters(adata_ATAC, cluster_prefix='peak_cluster_', 
                          n_top=20, metric='max', figsize=(15, 10)):
    """
    Plot top peak clusters by maximum or variance
    
    Parameters:
    -----------
    adata_ATAC : AnnData
        Cell data with peak cluster scores
    cluster_prefix : str
        Prefix for peak cluster score columns
    n_top : int
        Number of top clusters to plot
    metric : str
        'max' or 'var' - how to rank clusters
    figsize : tuple
        Figure size
    """
    
    # Get all peak cluster score columns
    cluster_cols = [col for col in adata_ATAC.obs.columns if col.startswith(cluster_prefix)]
    
    if len(cluster_cols) == 0:
        print(f"No columns found with prefix '{cluster_prefix}'")
        return
    
    # Calculate ranking metric
    if metric == 'max':
        scores = {col: adata_ATAC.obs[col].max() for col in cluster_cols}
    elif metric == 'var':
        scores = {col: adata_ATAC.obs[col].var() for col in cluster_cols}
    else:
        raise ValueError("metric must be 'max' or 'var'")
    
    # Sort and select top clusters
    top_clusters = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_top]
    top_cluster_names = [cluster[0] for cluster in top_clusters]
    
    print(f"Plotting top {n_top} peak clusters by {metric}...")
    
    # Calculate grid dimensions
    n_cols = 5
    n_rows = (len(top_cluster_names) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each cluster
    for i, cluster_col in enumerate(top_cluster_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Extract cluster number/name for title
        cluster_name = cluster_col.replace(cluster_prefix, '')
        score_val = scores[cluster_col]
        
        # Plot UMAP
        sc.pl.umap(adata_ATAC, color=cluster_col, ax=ax, show=False, 
                  cmap='viridis', title=f'Cluster {cluster_name}\n({metric}={score_val:.3f})')
        
    # Hide empty subplots
    for i in range(len(top_cluster_names), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def analyze_peak_cluster_celltype_enrichment(adata_ATAC, cluster_prefix='peak_cluster_', 
                                           celltype_key='cell_type', 
                                           timepoint_key='timepoint'):
    """
    Analyze which cell types/timepoints are enriched for each peak cluster
    
    Parameters:
    -----------
    adata_ATAC : AnnData
        Cell data with peak cluster scores
    cluster_prefix : str
        Prefix for peak cluster score columns
    celltype_key : str
        Key in adata_ATAC.obs with cell type annotations
    timepoint_key : str
        Key in adata_ATAC.obs with timepoint information
    
    Returns:
    --------
    enrichment_df : DataFrame
        Enrichment analysis results
    """
    
    # Get all peak cluster score columns
    cluster_cols = [col for col in adata_ATAC.obs.columns if col.startswith(cluster_prefix)]
    
    if len(cluster_cols) == 0:
        print(f"No columns found with prefix '{cluster_prefix}'")
        return None
    
    enrichment_results = []
    
    for cluster_col in tqdm(cluster_cols, desc="Analyzing enrichment"):
        cluster_name = cluster_col.replace(cluster_prefix, '')
        
        # Get top 10% of cells for this cluster
        threshold = np.percentile(adata_ATAC.obs[cluster_col], 90)
        top_cells = adata_ATAC.obs[cluster_col] >= threshold
        
        if celltype_key in adata_ATAC.obs.columns:
            # Cell type enrichment
            celltype_counts = adata_ATAC.obs[celltype_key].value_counts()
            top_celltype_counts = adata_ATAC.obs.loc[top_cells, celltype_key].value_counts()
            
            for celltype in celltype_counts.index:
                enrichment = (top_celltype_counts.get(celltype, 0) / top_cells.sum()) / \
                           (celltype_counts[celltype] / len(adata_ATAC.obs))
                
                enrichment_results.append({
                    'peak_cluster': cluster_name,
                    'category': 'cell_type',
                    'value': celltype,
                    'enrichment': enrichment,
                    'n_cells': top_celltype_counts.get(celltype, 0)
                })
        
        if timepoint_key in adata_ATAC.obs.columns:
            # Timepoint enrichment
            timepoint_counts = adata_ATAC.obs[timepoint_key].value_counts()
            top_timepoint_counts = adata_ATAC.obs.loc[top_cells, timepoint_key].value_counts()
            
            for timepoint in timepoint_counts.index:
                enrichment = (top_timepoint_counts.get(timepoint, 0) / top_cells.sum()) / \
                           (timepoint_counts[timepoint] / len(adata_ATAC.obs))
                
                enrichment_results.append({
                    'peak_cluster': cluster_name,
                    'category': 'timepoint',
                    'value': timepoint,
                    'enrichment': enrichment,
                    'n_cells': top_timepoint_counts.get(timepoint, 0)
                })
    
    enrichment_df = pd.DataFrame(enrichment_results)
    return enrichment_df


# %%

# %%
# Step 1: Compute peak cluster scores
adata_ATAC = compute_peak_cluster_scores(
    adata_ATAC, 
    adata_peaks, 
    cluster_key='leiden_unified'
)

# Step 2: Plot all peak cluster UMAPs
plot_peak_cluster_umaps(adata_ATAC, save_prefix='peak_clusters')

# Step 3: Plot top 20 clusters by maximum signal
plot_top_peak_clusters(adata_ATAC, n_top=20, metric='max')

# Step 4: Analyze cell type/timepoint enrichment
enrichment_df = analyze_peak_cluster_celltype_enrichment(
    adata_ATAC, 
    celltype_key='cell_type',  # adjust to your cell type column name
    timepoint_key='timepoint'   # adjust to your timepoint column name
)

# Display top enrichments
print("Top cell type enrichments:")
print(enrichment_df[enrichment_df['category'] == 'cell_type'].nlargest(10, 'enrichment'))

print("\nTop timepoint enrichments:")
print(enrichment_df[enrichment_df['category'] == 'timepoint'].nlargest(10, 'enrichment'))
"""

# %%

# %%

# %%
# move the .X to GPU (for faster operation)
rsc.get.anndata_to_GPU(adata_peaks)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
