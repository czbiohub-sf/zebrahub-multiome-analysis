"""
Data normalization utilities for gene expression and chromatin accessibility analysis.

This module provides various normalization methods commonly used in single-cell
multiome data analysis, including z-score, robust scaling, min-max scaling, and
logarithmic transformations.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from typing import Union, Optional, Literal
import matplotlib.pyplot as plt


def compute_gene_zscores(adata, copy: bool = False):
    """
    Compute z-scores for each gene (row) across timepoints (columns).

    Z-scores standardize gene expression values by subtracting the mean and
    dividing by the standard deviation for each gene independently. This
    makes expression values comparable across genes with different scales.

    Args:
        adata: AnnData object with genes as rows and timepoints as columns
        copy: Whether to return a new AnnData object or modify in place

    Returns:
        AnnData object with z-scored data in .X and original data in .layers['raw']

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('expression_data.h5ad')
        >>> adata_zscore = compute_gene_zscores(adata, copy=True)
        >>> # Access z-scored data
        >>> zscore_matrix = adata_zscore.X
        >>> # Access original data
        >>> raw_matrix = adata_zscore.layers['raw']
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


def normalize_genes(
    df: pd.DataFrame,
    method: Literal['robust', 'minmax', 'percent_max', 'zscore', 'log', 'standard'] = 'robust',
    plot: bool = False,
    genes_list: Optional[list] = None
) -> pd.DataFrame:
    """
    Normalize gene expression data using various methods.

    Provides 6 different normalization strategies suitable for different
    analysis scenarios:
    - robust: Robust scaling (median and IQR), resistant to outliers
    - minmax: Min-max scaling to [0, 1] range
    - percent_max: Scale to percentage of maximum value (0-100%)
    - zscore: Z-score standardization (mean=0, std=1)
    - log: Log1p transformation (log(x + 1))
    - standard: Standard scaling (sklearn StandardScaler)

    Args:
        df: DataFrame with genes as columns and timepoints/samples as rows
        method: Normalization method to apply
        plot: Whether to plot comparison of original vs normalized values
        genes_list: Optional list of specific genes to plot if plot=True

    Returns:
        Normalized DataFrame with same shape as input

    Raises:
        ValueError: If an invalid normalization method is specified

    Example:
        >>> df = pd.DataFrame({
        ...     'gene1': [10, 20, 30, 40],
        ...     'gene2': [100, 200, 300, 400]
        ... })
        >>> df_norm = normalize_genes(df, method='zscore')
        >>> df_robust = normalize_genes(df, method='robust', plot=True)
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
        # Normalize each column (gene) to percentage of its maximum value
        df_norm = df.apply(lambda x: (x / x.max()) * 100)

    elif method == 'zscore':
        # Z-score normalization: (x - mean) / std
        df_norm = df.apply(lambda x: (x - x.mean()) / x.std())

    elif method == 'log':
        # Log1p transformation: log(1 + x)
        df_norm = np.log1p(df)

    elif method == 'standard':
        # Standard scaling (similar to zscore but uses sklearn)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_norm = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns
        )

    else:
        raise ValueError(
            f"Invalid normalization method: {method}. "
            f"Choose from: 'robust', 'minmax', 'percent_max', 'zscore', 'log', 'standard'"
        )

    if plot:
        plot_normalization_comparison(df_orig, df_norm, method, genes_list=genes_list)

    return df_norm


def normalize_minmax(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series to [0, 1] range using min-max scaling.

    Formula: (x - min) / (max - min)

    Args:
        series: Pandas Series to normalize

    Returns:
        Normalized Series with values in [0, 1] range

    Example:
        >>> s = pd.Series([10, 20, 30, 40, 50])
        >>> normalized = normalize_minmax(s)
        >>> normalized.min(), normalized.max()
        (0.0, 1.0)
    """
    return (series - series.min()) / (series.max() - series.min())


def plot_normalization_comparison(
    df_orig: pd.DataFrame,
    df_norm: pd.DataFrame,
    method: str,
    genes_list: Optional[list] = None,
    n_genes: int = 5,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Plot comparison of original vs normalized gene expression values.

    Creates side-by-side plots showing how normalization affects
    the expression profiles of selected genes.

    Args:
        df_orig: Original (unnormalized) DataFrame
        df_norm: Normalized DataFrame
        method: Name of normalization method used
        genes_list: List of specific genes to plot. If None, random genes selected
        n_genes: Number of random genes to plot if genes_list is None
        figsize: Figure size as (width, height) tuple

    Returns:
        matplotlib Figure object

    Example:
        >>> df_orig = pd.DataFrame(np.random.randn(10, 100))
        >>> df_norm = normalize_genes(df_orig, method='zscore', plot=False)
        >>> fig = plot_normalization_comparison(df_orig, df_norm, 'zscore')
    """
    # Select genes to plot
    if genes_list is None:
        available_genes = df_orig.columns.tolist()
        genes_list = np.random.choice(
            available_genes,
            size=min(n_genes, len(available_genes)),
            replace=False
        )

    n_plot_genes = len(genes_list)
    fig, axes = plt.subplots(n_plot_genes, 2, figsize=figsize, squeeze=False)

    for idx, gene in enumerate(genes_list):
        if gene not in df_orig.columns:
            continue

        # Plot original values
        ax_orig = axes[idx, 0]
        ax_orig.plot(df_orig[gene].values, marker='o', linewidth=2)
        ax_orig.set_title(f'{gene} - Original')
        ax_orig.set_ylabel('Expression')
        ax_orig.grid(True, alpha=0.3)

        # Plot normalized values
        ax_norm = axes[idx, 1]
        ax_norm.plot(df_norm[gene].values, marker='o', linewidth=2, color='orange')
        ax_norm.set_title(f'{gene} - {method.capitalize()} Normalized')
        ax_norm.set_ylabel('Normalized Expression')
        ax_norm.grid(True, alpha=0.3)

        if idx == n_plot_genes - 1:
            ax_orig.set_xlabel('Sample')
            ax_norm.set_xlabel('Sample')

    plt.tight_layout()
    return fig


def normalize_by_max_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each gene to percentage of its maximum value.

    Each gene is scaled so that its maximum value becomes 100%
    and all other values are proportional percentages.

    Args:
        df: DataFrame with genes as columns and samples as rows

    Returns:
        DataFrame with values normalized to 0-100% scale

    Example:
        >>> df = pd.DataFrame({'gene1': [1, 2, 3, 4, 5]})
        >>> normalized = normalize_by_max_percent(df)
        >>> normalized['gene1'].max()
        100.0
    """
    return df.apply(lambda x: (x / x.max()) * 100, axis=0)


def robust_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute robust z-scores using median and MAD (Median Absolute Deviation).

    More robust to outliers than standard z-score normalization.
    Formula: (x - median) / MAD

    Args:
        df: DataFrame with features as columns

    Returns:
        DataFrame with robust z-scores

    Example:
        >>> df = pd.DataFrame(np.random.randn(100, 10))
        >>> df_robust = robust_zscore(df)
    """
    def mad(x):
        """Calculate Median Absolute Deviation"""
        return np.median(np.abs(x - np.median(x)))

    return df.apply(lambda x: (x - np.median(x)) / (mad(x) + 1e-10), axis=0)
