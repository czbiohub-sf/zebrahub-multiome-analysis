"""
Plotting utilities for Figure 1 analysis.

This module provides centralized plotting configuration and visualization
utilities used across multiple Figure 1 notebooks.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple


def set_plotting_style() -> None:
    """
    Set consistent matplotlib plotting style for publication-quality figures.

    Configures matplotlib to use:
    - Seaborn paper style
    - Specific font sizes for axes, titles, ticks
    - Sans-serif fonts with LaTeX support
    - SVG font embedding disabled for editability
    - Default figure size of 10x9 inches

    This function should be called before generating any plots to ensure
    consistent styling across all figures.

    Returns:
        None

    Example:
        >>> set_plotting_style()
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> plt.savefig('figure.pdf')
    """
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
    plt.rc('figure', figsize=[10, 9])
    plt.rc('svg', fonttype='none')


def plot_umap(
    adata,
    save_path: Optional[str] = None,
    gene_list: Optional[List[str]] = None,
    color_by: str = 'dataset',
    **kwargs
) -> plt.Figure:
    """
    Create UMAP plot with optional gene highlighting.

    Args:
        adata: AnnData object with UMAP coordinates in obsm['X_umap']
        save_path: Optional path to save the figure
        gene_list: Optional list of genes to highlight in the plot
        color_by: Column in adata.obs to color points by (default: 'dataset')
        **kwargs: Additional arguments passed to plt.scatter()

    Returns:
        matplotlib Figure object

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('data.h5ad')
        >>> fig = plot_umap(adata, gene_list=['myf5', 'sox2'], save_path='umap.pdf')
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']

    # Create scatter plot
    if color_by in adata.obs.columns:
        colors = adata.obs[color_by]
        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=colors if isinstance(colors.dtype, (int, float)) else pd.Categorical(colors).codes,
            s=5,
            alpha=0.5,
            **kwargs
        )
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            s=5,
            alpha=0.5,
            **kwargs
        )

    # Highlight specific genes if provided
    if gene_list:
        for gene in gene_list:
            if gene in adata.var_names:
                gene_idx = adata.var_names.tolist().index(gene)
                high_expr = adata.X[:, gene_idx] > np.median(adata.X[:, gene_idx])
                ax.scatter(
                    umap_coords[high_expr, 0],
                    umap_coords[high_expr, 1],
                    s=10,
                    alpha=0.8,
                    label=gene
                )
        ax.legend()

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP Projection')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compare_normalizations_single_plot(
    df: pd.DataFrame,
    genes_list: Optional[List[str]] = None,
    n_random_genes: int = 5,
    figsize: Tuple[int, int] = (15, 5)
) -> Dict[str, pd.DataFrame]:
    """
    Compare different normalization methods for genes in a single plot.

    Creates comparison plots showing how different normalization methods
    affect gene expression profiles across samples.

    Args:
        df: DataFrame with genes as rows and samples as columns
        genes_list: List of specific genes to plot. If None, random genes are selected
        n_random_genes: Number of random genes to plot if genes_list is None
        figsize: Figure size as (width, height) tuple

    Returns:
        Dictionary mapping normalization method names to normalized DataFrames

    Example:
        >>> df = pd.DataFrame(np.random.randn(100, 6), columns=['s1', 's2', 's3', 's4', 's5', 's6'])
        >>> norm_data = compare_normalizations_single_plot(df, genes_list=['myf5', 'sox2'])
    """
    from sklearn.preprocessing import RobustScaler, MinMaxScaler

    # Select genes to plot
    if genes_list is None:
        genes_list = df.sample(n=min(n_random_genes, len(df))).index.tolist()

    # Define normalization methods
    methods = {
        'original': lambda x: x,
        'robust': lambda x: RobustScaler().fit_transform(x.reshape(-1, 1)).flatten(),
        'minmax': lambda x: MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten(),
        'percent_max': lambda x: (x / x.max()) * 100,
        'zscore': lambda x: (x - x.mean()) / x.std(),
        'log': lambda x: np.log1p(x)
    }

    # Store normalized data
    normalized_data = {}

    # Create subplots
    n_genes = len(genes_list)
    fig, axes = plt.subplots(1, n_genes, figsize=figsize, sharey=False)
    if n_genes == 1:
        axes = [axes]

    for idx, gene in enumerate(genes_list):
        if gene not in df.index:
            continue

        gene_data = df.loc[gene].values
        ax = axes[idx]

        # Apply each normalization method and plot
        for method_name, method_func in methods.items():
            normalized = method_func(gene_data)
            normalized_data[f'{gene}_{method_name}'] = normalized
            ax.plot(normalized, label=method_name, marker='o', markersize=3)

        ax.set_title(f'{gene}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(False)

    plt.tight_layout()

    return normalized_data


def plot_comprehensive_neighborhood_purity(
    all_purity_results: Dict[str, pd.DataFrame],
    adata,
    plot_type: str = "violin",
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create comprehensive visualization of neighborhood purity metrics.

    Generates 3 subplots showing neighborhood purity across different
    metadata categories (e.g., cell types, batches, timepoints).

    Args:
        all_purity_results: Dictionary mapping cluster names to purity DataFrames
        adata: AnnData object with cell metadata
        plot_type: Type of plot - "violin" or "box"
        figsize: Figure size as (width, height) tuple

    Returns:
        matplotlib Figure object with 3 subplots

    Example:
        >>> purity_results = compute_comprehensive_neighborhood_purity(adata, ...)
        >>> fig = plot_comprehensive_neighborhood_purity(purity_results, adata)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for idx, (cluster_name, purity_df) in enumerate(all_purity_results.items()):
        if idx >= 3:
            break

        ax = axes[idx]

        if plot_type == "violin":
            purity_df.boxplot(ax=ax, rot=45)
        else:
            purity_df.plot(kind='box', ax=ax, rot=45)

        ax.set_title(f'Neighborhood Purity: {cluster_name}')
        ax.set_ylabel('Purity Score')
        ax.set_xlabel('Integration Method')
        ax.grid(False)

    plt.tight_layout()
    return fig


def plot_rna_weights_by_celltype(
    adata,
    groupby: str = 'annotation_ML_coarse',
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot violin plot of RNA weights by cell type annotation.

    Visualizes the distribution of RNA vs ATAC weights across different
    cell types in a WNN (Weighted Nearest Neighbor) integration.

    Args:
        adata: AnnData object with 'RNA_weight' in obs
        groupby: Column in adata.obs to group cells by
        figsize: Figure size as (width, height) tuple

    Returns:
        Tuple of (Figure object, DataFrame with weight statistics)

    Example:
        >>> fig, stats = plot_rna_weights_by_celltype(adata, groupby='cell_type')
    """
    if 'RNA_weight' not in adata.obs.columns:
        raise ValueError("AnnData object must have 'RNA_weight' in obs")

    # Create plot data
    plot_data = pd.DataFrame({
        'cell_type': adata.obs[groupby],
        'RNA_weight': adata.obs['RNA_weight']
    })

    # Calculate ATAC weight
    plot_data['ATAC_weight'] = 1 - plot_data['RNA_weight']

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # RNA weights
    plot_data.boxplot(column='RNA_weight', by='cell_type', ax=axes[0], rot=90)
    axes[0].set_title('RNA Weight Distribution by Cell Type')
    axes[0].set_ylabel('RNA Weight')
    axes[0].set_xlabel('Cell Type')

    # ATAC weights
    plot_data.boxplot(column='ATAC_weight', by='cell_type', ax=axes[1], rot=90)
    axes[1].set_title('ATAC Weight Distribution by Cell Type')
    axes[1].set_ylabel('ATAC Weight')
    axes[1].set_xlabel('Cell Type')

    plt.tight_layout()

    # Calculate statistics
    stats = plot_data.groupby('cell_type').agg({
        'RNA_weight': ['mean', 'std', 'median'],
        'ATAC_weight': ['mean', 'std', 'median']
    })

    return fig, stats
