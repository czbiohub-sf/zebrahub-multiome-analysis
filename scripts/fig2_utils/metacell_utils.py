"""
Metacell computation and visualization utilities for SEACells analysis.

This module provides functions for:
- Computing SEACells metacells from single-cell RNA+ATAC data
- Aggregating count matrices over metacells
- Computing cell type purity and prevalence within metacells
- Visualizing metacells on 2D embeddings (UMAP, etc.)

These utilities support the RNA-ATAC correlation analysis via metacells,
enabling dimensionality reduction while preserving biological structure.

Dependencies:
    - SEACells: For metacell aggregation
    - scanpy: For AnnData object manipulation
    - pandas, numpy: For data processing
    - matplotlib, seaborn: For visualization
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sci
import scipy.sparse as sp
import SEACells
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_counts_multiome(
    adata_rna: sc.AnnData,
    adata_atac: sc.AnnData,
    groupby: str = "SEACell",
    celltype_key: str = "celltype"
) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Aggregate counts over SEACells for RNA and ATAC modalities.

    This function uses SEACells.aggregate_counts() to sum counts within
    each metacell, enabling downstream metacell-level analysis.

    Args:
        adata_rna: AnnData object with RNA counts and SEACell assignments
        adata_atac: AnnData object with ATAC counts and SEACell assignments
        groupby: Column name in obs containing SEACell assignments
        celltype_key: Column name in obs containing cell type annotations

    Returns:
        Tuple of (aggregated_rna, aggregated_atac) AnnData objects

    Example:
        >>> rna_meta, atac_meta = aggregate_counts_multiome(
        ...     adata_rna, adata_atac, groupby="SEACell"
        ... )
        >>> print(f"Reduced from {adata_rna.n_obs} cells to {rna_meta.n_obs} metacells")

    Notes:
        - Input adata objects must have raw counts in .X layer
        - SEACell assignments must exist in .obs[groupby]
        - Cell type information is transferred to metacells based on prevalence
    """
    # Aggregate counts over SEACells for both modalities
    rna_metacell = SEACells.core.summarize_by_SEACell(
        adata_rna,
        SEACells_label=groupby,
        summarize_layer="raw"
    )

    atac_metacell = SEACells.core.summarize_by_SEACell(
        adata_atac,
        SEACells_label=groupby,
        summarize_layer="raw"
    )

    return rna_metacell, atac_metacell


def compute_prevalent_celltype_per_metacell(
    adata: sc.AnnData,
    celltype_key: str = "annotation_ML_coarse",
    metacell_key: str = "SEACell"
) -> pd.Series:
    """
    Compute the most prevalent cell type in each metacell.

    For each metacell, finds the cell type that appears most frequently
    among its constituent single cells. This provides a cell type label
    for metacell-level analysis.

    Args:
        adata: AnnData object with cell type and metacell info in .obs
        celltype_key: Column name for cell types (e.g., 'annotation_ML_coarse')
        metacell_key: Column name for metacells (e.g., 'SEACell')

    Returns:
        Series mapping metacell ID -> most prevalent cell type

    Example:
        >>> prevalent = compute_prevalent_celltype_per_metacell(
        ...     adata, celltype_key="annotation_ML_coarse"
        ... )
        >>> adata_meta.obs["celltype"] = adata_meta.obs_names.map(prevalent)

    Notes:
        - If there's a tie, the first cell type (alphabetically) is returned
        - Metacells with no assigned cells will not appear in the output
    """
    # Extract relevant columns
    df = adata.obs[[celltype_key, metacell_key]].copy()

    # Group by metacell and find most common cell type
    prevalent_celltypes = df.groupby(metacell_key)[celltype_key] \
                            .apply(lambda x: x.value_counts().idxmax())

    return prevalent_celltypes


def compute_top_two_celltypes_by_combined_expression(
    rna_meta_ad: sc.AnnData,
    atac_meta_ad: sc.AnnData,
    celltype_key: str = "celltype"
) -> pd.Series:
    """
    Find top 2 cell types per gene based on combined RNA + ATAC expression.

    For each gene, computes mean expression across cell types for both
    RNA and ATAC modalities, sums them, and returns the top 2 cell types
    with highest combined expression. Useful for identifying which cell
    types drive gene activity.

    Args:
        rna_meta_ad: AnnData with RNA metacell data (metacells × genes)
        atac_meta_ad: AnnData with ATAC metacell data (metacells × genes)
        celltype_key: Column in .obs with cell type labels

    Returns:
        Series mapping gene -> list of top 2 cell types

    Example:
        >>> top_celltypes = compute_top_two_celltypes_by_combined_expression(
        ...     rna_meta, atac_meta
        ... )
        >>> print(f"Gene cdx1a top celltypes: {top_celltypes['cdx1a']}")

    Notes:
        - Handles both sparse and dense matrices
        - Genes must be shared between RNA and ATAC objects
        - Cell type must be present in .obs for both objects
    """
    # Convert sparse matrices to dense if necessary
    if sci.sparse.issparse(rna_meta_ad.X):
        rna_expr_values = pd.DataFrame(
            rna_meta_ad.X.toarray(),
            index=rna_meta_ad.obs.index,
            columns=rna_meta_ad.var.index
        )
    else:
        rna_expr_values = pd.DataFrame(
            rna_meta_ad.X,
            index=rna_meta_ad.obs.index,
            columns=rna_meta_ad.var.index
        )

    if sci.sparse.issparse(atac_meta_ad.X):
        atac_expr_values = pd.DataFrame(
            atac_meta_ad.X.toarray(),
            index=atac_meta_ad.obs.index,
            columns=atac_meta_ad.var.index
        )
    else:
        atac_expr_values = pd.DataFrame(
            atac_meta_ad.X,
            index=atac_meta_ad.obs.index,
            columns=atac_meta_ad.var.index
        )

    # Add RNA and ATAC expression values for each gene
    combined_expr_values = rna_expr_values + atac_expr_values

    # Add cell type information
    combined_expr_values[celltype_key] = rna_meta_ad.obs[celltype_key]

    # Compute mean expression of combined values by cell type
    mean_combined_expr_by_celltype = combined_expr_values.groupby(celltype_key).mean().T

    # For each gene, find top 2 cell types by combined expression
    top_two_celltypes_per_gene = mean_combined_expr_by_celltype.apply(
        lambda row: row.nlargest(2).index.tolist(), axis=1
    )

    return top_two_celltypes_per_gene


def plot_2D_modified(
    ad: sc.AnnData,
    ax: plt.Axes,
    key: str = "X_umap",
    colour_metacells: bool = True,
    title: str = "Metacell Assignments",
    palette: str = "Set2",
    SEACell_size: int = 20,
    cell_size: int = 10
) -> None:
    """
    Plot 2D visualization of metacells on an embedding.

    Modified version of SEACells.plot.plot_2D() that works with provided axes.
    Plots single cells as small points and metacell centroids as larger points
    with black borders.

    Args:
        ad: AnnData with 'SEACell' labels in .obs
        ax: Matplotlib axes object for plotting
        key: Key in .obsm for 2D embedding (default: 'X_umap')
        colour_metacells: Whether to color by metacell assignment
        title: Plot title
        palette: Matplotlib colormap for metacells
        SEACell_size: Point size for metacell centroids
        cell_size: Point size for individual cells

    Returns:
        None (modifies ax in place)

    Example:
        >>> fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        >>> plot_2D_modified(
        ...     adata, ax, key="X_umap.joint",
        ...     colour_metacells=True
        ... )
        >>> plt.savefig("metacells_umap.png")

    Notes:
        - Requires 'SEACell' column in adata.obs
        - Embedding must be 2D (first 2 components used)
    """
    # Prepare data
    umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs["SEACell"])
    umap["SEACell"] = umap["SEACell"].astype("category")

    # Compute metacell centroids
    mcs = umap.groupby("SEACell").mean().reset_index()

    if colour_metacells:
        # Plot cells colored by metacell
        sns.scatterplot(
            x=0, y=1, hue="SEACell", data=umap, s=cell_size,
            palette=sns.color_palette(palette), legend=None, ax=ax
        )
        # Plot metacell centroids
        sns.scatterplot(
            x=0, y=1, s=SEACell_size, hue="SEACell", data=mcs,
            palette=sns.color_palette(palette),
            edgecolor="black", linewidth=1.25, legend=None, ax=ax
        )
    else:
        # Plot cells in grey
        sns.scatterplot(
            x=0, y=1, color="grey", data=umap, s=cell_size, legend=None, ax=ax
        )
        # Plot metacell centroids in red
        sns.scatterplot(
            x=0, y=1, s=SEACell_size, color="red", data=mcs,
            edgecolor="black", linewidth=1.25, legend=None, ax=ax
        )

    ax.set_xlabel(f"{key}-0")
    ax.set_ylabel(f"{key}-1")
    ax.set_title(title)
    ax.set_axis_off()


def plot_SEACell_sizes_modified(
    ad: sc.AnnData,
    ax: plt.Axes,
    show: bool = True,
    title: str = "Distribution of Metacell Sizes",
    bins: Optional[int] = None,
    figsize: Tuple[int, int] = (5, 5)
) -> pd.DataFrame:
    """
    Plot distribution of cells per metacell.

    Modified version of SEACells.plot.plot_SEACell_sizes() that works with
    provided axes. Shows histogram of metacell sizes to assess quality.

    Args:
        ad: AnnData with 'SEACell' labels in .obs
        ax: Matplotlib axes object for plotting
        show: Whether to display the plot
        title: Plot title
        bins: Number of histogram bins (default: auto)
        figsize: Figure size (unused when ax is provided)

    Returns:
        DataFrame with metacell sizes (index=SEACell, column='size')

    Example:
        >>> fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        >>> sizes = plot_SEACell_sizes_modified(adata, ax, bins=10)
        >>> print(f"Mean cells per metacell: {sizes['size'].mean():.1f}")

    Notes:
        - Requires 'SEACell' column in adata.obs
        - Returns size distribution for downstream QC analysis
    """
    assert "SEACell" in ad.obs, 'AnnData must contain "SEACell" in obs DataFrame.'

    label_df = ad.obs[["SEACell"]].reset_index()

    sns.histplot(label_df.groupby("SEACell").count().iloc[:, 0], bins=bins, ax=ax)
    sns.despine()
    ax.set_xlabel("Number of Cells per SEACell")
    ax.set_title(title)

    return pd.DataFrame(
        label_df.groupby("SEACell").count().iloc[:, 0]
    ).rename(columns={"index": "size"})


def plot_2D_with_metacells(
    ad: sc.AnnData,
    key: str = "X_umap.joint",
    hue: str = "annotation_ML_coarse",
    metacell_key: str = "SEACell",
    title: str = "UMAP with Metacells",
    palette: Optional[Dict] = None,
    SEACell_size: int = 50,
    cell_size: int = 10
) -> plt.Figure:
    """
    Plot cells colored by cell type with metacell centroids overlaid.

    Creates a 2D embedding visualization where cells are colored by their
    biological annotation and metacell centroids are shown as larger points
    with black borders. The metacell centroids are colored by the most
    prevalent cell type within that metacell.

    Args:
        ad: AnnData object with embedding, cell types, and metacell labels
        key: Key in .obsm for 2D embedding (e.g., 'X_umap.joint')
        hue: Column in .obs for cell type coloring
        metacell_key: Column in .obs for metacell assignments
        title: Plot title
        palette: Color palette dict mapping cell types to colors
        SEACell_size: Point size for metacell centroids
        cell_size: Point size for individual cells

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_2D_with_metacells(
        ...     adata,
        ...     key="X_umap.joint",
        ...     hue="annotation_ML_coarse",
        ...     palette=cell_type_color_dict
        ... )
        >>> fig.savefig("umap_with_metacells.png")

    Notes:
        - Automatically computes most prevalent cell type per metacell
        - Metacells inherit color from their dominant cell type
        - Useful for visualizing metacell quality and cell type preservation
    """
    # Prepare data
    umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(
        ad.obs[[hue, metacell_key]]
    )
    umap[hue] = umap[hue].astype("category")
    umap[metacell_key] = umap[metacell_key].astype("category")

    # Get metacell centroids
    mcs = umap.groupby(metacell_key).mean().reset_index()

    # Compute most prevalent cell type for each metacell
    prevalent_celltypes = compute_prevalent_celltype_per_metacell(
        ad, celltype_key=hue, metacell_key=metacell_key
    )

    # Add most prevalent cell type to metacell dataframe
    mcs[hue] = mcs[metacell_key].map(prevalent_celltypes)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot cells colored by cell type
    sns.scatterplot(
        x=0, y=1, hue=hue, data=umap, s=cell_size,
        palette=palette, legend=None, ax=ax
    )

    # Overlay metacells on UMAP
    sns.scatterplot(
        x=0, y=1, hue=hue, data=mcs, s=SEACell_size,
        palette=palette, edgecolor="black", linewidth=1.25, legend=None, ax=ax
    )

    # Adjust plot
    ax.set_xlabel(f"{key}-0")
    ax.set_ylabel(f"{key}-1")
    ax.set_axis_off()

    return fig
