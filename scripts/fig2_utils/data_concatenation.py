"""
Data concatenation and aggregation utilities for metacell analysis.

This module provides functions for:
- Finding shared genes across multiple datasets
- Concatenating RNA and ATAC data across timepoints
- Computing group averages (by cell type and timepoint)
- Calculating gene-level statistics (mean, variance, etc.)

These utilities enable construction of genes-by-(celltype×timepoint) matrices
for downstream temporal dynamics analysis.

Dependencies:
    - scanpy: For AnnData concatenation and manipulation
    - pandas, numpy: For data aggregation
    - scipy: For statistical computations
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
from scipy.stats import median_abs_deviation
from typing import Dict, List, Optional, Tuple


def find_shared_genes(
    list_datasets: List[str],
    metacell_path: str,
    rna_suffix: str = "_RNA_seacells_aggre.h5ad",
    atac_suffix: str = "_ATAC_seacells_aggre.h5ad"
) -> np.ndarray:
    """
    Find genes shared across all RNA and ATAC datasets.

    Computes the intersection of gene sets across multiple timepoints/datasets,
    ensuring all downstream analysis uses a consistent feature space.

    Args:
        list_datasets: List of dataset IDs (e.g., ['TDR126', 'TDR127', ...])
        metacell_path: Base path to metacell aggregated data files
        rna_suffix: Filename suffix for RNA files (default: "_RNA_seacells_aggre.h5ad")
        atac_suffix: Filename suffix for ATAC files (default: "_ATAC_seacells_aggre.h5ad")

    Returns:
        Array of gene names present in all datasets (both RNA and ATAC)

    Example:
        >>> list_datasets = ['TDR126', 'TDR127', 'TDR128']
        >>> shared_genes = find_shared_genes(
        ...     list_datasets,
        ...     "/path/to/metacells/"
        ... )
        >>> print(f"Found {len(shared_genes)} shared genes")

    Notes:
        - Handles "reseq" suffix in dataset names
        - Returns intersection: genes present in ALL datasets
        - Use this before concatenate_data() to ensure consistency
    """
    all_shared_genes = None

    for data_id in list_datasets:
        data_name = data_id.strip("reseq")

        # Read RNA and ATAC data
        rna_meta_ad = sc.read_h5ad(metacell_path + f"{data_id}/{data_name}{rna_suffix}")
        atac_meta_ad = sc.read_h5ad(metacell_path + f"{data_id}/{data_name}{atac_suffix}")

        # Find shared genes between RNA and ATAC for this dataset
        shared_genes = np.intersect1d(rna_meta_ad.var_names, atac_meta_ad.var_names)

        # Update set of genes shared across ALL datasets
        if all_shared_genes is None:
            all_shared_genes = set(shared_genes)
        else:
            all_shared_genes = all_shared_genes.intersection(set(shared_genes))

    all_shared_genes = np.array(list(all_shared_genes))
    print(f"Number of genes shared across ALL datasets: {len(all_shared_genes)}")

    return all_shared_genes


def concatenate_data(
    list_datasets: List[str],
    list_timepoints: List[str],
    metacell_path: str,
    shared_genes: np.ndarray,
    rna_suffix: str = "_RNA_seacells_aggre.h5ad",
    atac_suffix: str = "_ATAC_seacells_aggre.h5ad"
) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Concatenate RNA and ATAC data across timepoints with shared genes.

    Loads metacell-aggregated data for multiple timepoints, subsets to shared
    genes, adds timepoint metadata, and concatenates into single AnnData objects
    for cross-timepoint analysis.

    Args:
        list_datasets: List of dataset IDs (e.g., ['TDR126', 'TDR127'])
        list_timepoints: List of timepoint labels (e.g., ['10hpf', '12hpf'])
        metacell_path: Base path to metacell data
        shared_genes: Array of genes to use (from find_shared_genes())
        rna_suffix: RNA file suffix
        atac_suffix: ATAC file suffix

    Returns:
        Tuple of (combined_rna, combined_atac) AnnData objects

    Example:
        >>> shared_genes = find_shared_genes(datasets, path)
        >>> combined_rna, combined_atac = concatenate_data(
        ...     datasets, timepoints, path, shared_genes
        ... )
        >>> print(f"RNA: {combined_rna.shape} (metacells × genes)")
        >>> print(f"ATAC: {combined_atac.shape} (metacells × genes)")

    Notes:
        - Adds 'timepoint' column to .obs
        - Makes obs_names unique by appending timepoint
        - Uses outer join (fills missing with NaN)
        - Handles "reseq" suffix in dataset names
    """
    rna_matrices = []
    atac_matrices = []

    for i, data_id in enumerate(list_datasets):
        data_name = data_id.strip("reseq")

        # Read data
        rna_meta_ad = sc.read_h5ad(metacell_path + f"{data_name}{rna_suffix}")
        atac_meta_ad = sc.read_h5ad(metacell_path + f"{data_name}{atac_suffix}")

        # Subset to shared genes
        rna_meta_ad = rna_meta_ad[:, shared_genes]
        atac_meta_ad = atac_meta_ad[:, shared_genes]

        # Add timepoint information
        timepoint = list_timepoints[i]
        rna_meta_ad.obs['timepoint'] = timepoint
        atac_meta_ad.obs['timepoint'] = timepoint

        # Make unique indices by adding timepoint
        rna_meta_ad.obs_names = [f"{idx}_{timepoint}" for idx in rna_meta_ad.obs_names]
        atac_meta_ad.obs_names = [f"{idx}_{timepoint}" for idx in atac_meta_ad.obs_names]

        # Store
        rna_matrices.append(rna_meta_ad)
        atac_matrices.append(atac_meta_ad)

    # Concatenate all matrices
    combined_rna = sc.concat(rna_matrices, join='outer')
    combined_atac = sc.concat(atac_matrices, join='outer')

    # Print summary
    print("\nFinal data shapes:")
    print(f"RNA: {combined_rna.shape} (metacells × genes)")
    print(f"ATAC: {combined_atac.shape} (metacells × genes)")

    return combined_rna, combined_atac


def compute_group_averages(
    adata: sc.AnnData,
    group_by: List[str] = ['celltype', 'timepoint']
) -> sc.AnnData:
    """
    Compute average expression grouped by metadata columns.

    Aggregates expression values by cell type and timepoint (or other grouping
    variables), filling NaN values with 0. Creates a new AnnData object with
    averaged values where observations are group combinations.

    Args:
        adata: Input AnnData with observations containing group_by columns
        group_by: List of column names to group by (default: ['celltype', 'timepoint'])

    Returns:
        New AnnData object with:
            - Rows: Unique combinations of grouping variables
            - Columns: Same genes as input
            - X: Averaged expression values
            - obs: Grouping variable values
            - obs_names: Formatted as "{group1}_{group2}_..." (e.g., "NMPs_10hpf")

    Example:
        >>> # Average over cell types and timepoints
        >>> rna_grouped = compute_group_averages(
        ...     combined_rna, group_by=['celltype', 'timepoint']
        ... )
        >>> print(rna_grouped.shape)  # (n_celltype×timepoint combos, n_genes)
        >>> print(rna_grouped.obs_names[:3])  # ['NMPs_10hpf', 'NMPs_12hpf', ...]

    Notes:
        - Converts sparse matrices to dense
        - Fills NaN values with 0 (occurs when group has no data)
        - Preserves original .var information
        - Useful for temporal analysis at cell type resolution
    """
    # Convert sparse matrix to dense if needed
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

    # Create DataFrame with expression values
    exp_df = pd.DataFrame(
        X,
        index=adata.obs.index,
        columns=adata.var_names
    )

    # Add metadata columns for grouping
    for col in group_by:
        exp_df[col] = adata.obs[col]

    # Compute averages and fill NaNs with 0
    grouped_means = exp_df.groupby(group_by)[adata.var_names].mean().fillna(0)

    # Create new AnnData object
    adata_grouped = sc.AnnData(
        X=grouped_means.values,
        obs=grouped_means.index.to_frame(index=True),
        var=adata.var.copy()
    )

    # Create formatted obs_names
    adata_grouped.obs_names = [
        '_'.join(map(str, idx)) for idx in grouped_means.index
    ]

    # Print summary
    print("\nGrouping summary:")
    print(f"Original shape: {adata.shape}")
    print(f"Grouped shape: {adata_grouped.shape}")
    print(f"Number of zeros from NaN filling: {(adata_grouped.X == 0).sum()}")

    return adata_grouped


def compute_gene_stats(adata: sc.AnnData) -> pd.DataFrame:
    """
    Compute gene-level statistics across observations.

    Calculates mean, median, variance, robust variance (MAD²), and derived
    metrics (CV, variance-to-mean ratio) for each gene. Useful for feature
    selection based on temporal variability.

    Args:
        adata: AnnData object (typically genes × (celltype×timepoint) matrix)

    Returns:
        DataFrame with rows=genes and columns:
            - mean: Mean expression across observations
            - median: Median expression
            - variance: Variance across observations
            - robust_variance: Median absolute deviation squared
            - var_mean_ratio: Variance / mean (dispersion)
            - var_median_ratio: Variance / median
            - robust_var_median_ratio: Robust variance / median
            - cv: Coefficient of variation (std / mean)

    Example:
        >>> # Compute stats for RNA data
        >>> rna_stats = compute_gene_stats(rna_meta_grouped)
        >>> # Select highly variable genes
        >>> hvg = rna_stats.nlargest(2000, 'var_mean_ratio').index
        >>> rna_filtered = rna_meta_grouped[:, hvg]

    Notes:
        - Handles sparse matrices (converts to dense for median/MAD)
        - Robust metrics less sensitive to outlier observations
        - var_mean_ratio useful for identifying overdispersed genes
        - cv = coefficient of variation (normalized variability)
    """
    # Compute mean and variance (works on sparse or dense)
    means = adata.X.mean(axis=0)  # across observations
    vars = adata.X.var(axis=0)

    # Convert to 1D array if needed (for sparse matrices)
    if hasattr(means, 'A1'):
        means = means.A1
    if hasattr(vars, 'A1'):
        vars = vars.A1

    # Convert to dense for median/MAD (required by these functions)
    X_dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X

    # Compute medians
    medians = np.median(X_dense, axis=0)

    # Compute robust variance (MAD²)
    mads = median_abs_deviation(X_dense, axis=0)
    robust_variance = mads ** 2

    # Create DataFrame with statistics
    stats_df = pd.DataFrame({
        'mean': means,
        'median': medians,
        'variance': vars,
        'robust_variance': robust_variance,
        'var_mean_ratio': vars / means,
        'var_median_ratio': vars / medians,
        'robust_var_median_ratio': robust_variance / medians,
        'cv': np.sqrt(vars) / means  # coefficient of variation
    }, index=adata.var_names)

    return stats_df


def aggregate_celltype_timepoint_data(
    combined_rna: sc.AnnData,
    combined_atac: sc.AnnData,
    group_by: List[str] = ['celltype', 'timepoint']
) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Aggregate RNA and ATAC data by cell type and timepoint.

    Convenience wrapper that applies compute_group_averages() to both
    modalities and adds modality suffix to obs_names.

    Args:
        combined_rna: Concatenated RNA metacell data
        combined_atac: Concatenated ATAC metacell data
        group_by: Grouping variables (default: ['celltype', 'timepoint'])

    Returns:
        Tuple of (rna_grouped, atac_grouped) with:
            - Averaged expression by group
            - obs_names suffixed with "_rna" or "_atac"
            - "raw" layer containing original averaged values

    Example:
        >>> rna_grouped, atac_grouped = aggregate_celltype_timepoint_data(
        ...     combined_rna, combined_atac
        ... )
        >>> # obs_names format: "NMPs_10hpf_rna", "PSM_12hpf_atac", etc.

    Notes:
        - Adds "_rna" and "_atac" suffixes for easy identification
        - Stores original values in .layers["raw"]
        - Both outputs have identical gene sets and group structure
    """
    # Compute averages
    rna_grouped = compute_group_averages(combined_rna, group_by=group_by)
    atac_grouped = compute_group_averages(combined_atac, group_by=group_by)

    # Add modality suffix to obs_names
    rna_grouped.obs_names = rna_grouped.obs_names + "_rna"
    atac_grouped.obs_names = atac_grouped.obs_names + "_atac"

    # Store raw layer
    rna_grouped.layers["raw"] = rna_grouped.X.copy()
    atac_grouped.layers["raw"] = atac_grouped.X.copy()

    print("\nAggregated data:")
    print(f"RNA: {rna_grouped.shape}")
    print(f"ATAC: {atac_grouped.shape}")
    print(f"Example obs_names: {rna_grouped.obs_names[:2].tolist()}")

    return rna_grouped, atac_grouped


def filter_by_gene_variance(
    rna_meta: sc.AnnData,
    atac_meta: sc.AnnData,
    n_top: int = 2000,
    metric: str = 'var_mean_ratio',
    union: bool = True
) -> Tuple[sc.AnnData, sc.AnnData, np.ndarray]:
    """
    Filter genes by temporal variance using specified metric.

    Computes gene statistics and selects top N genes with highest variability
    across timepoints/celltypes. Can return union or intersection of top genes
    from both modalities.

    Args:
        rna_meta: RNA AnnData (genes × groups)
        atac_meta: ATAC AnnData (genes × groups)
        n_top: Number of top genes to select per modality
        metric: Statistic to use for ranking ('var_mean_ratio', 'robust_var_median_ratio', etc.)
        union: If True, return union of top genes; if False, return intersection

    Returns:
        Tuple of:
            - rna_filtered: RNA data with top genes
            - atac_filtered: ATAC data with top genes
            - top_genes: Array of selected gene names

    Example:
        >>> # Select top 3000 genes by variance/mean ratio
        >>> rna_filt, atac_filt, genes = filter_by_gene_variance(
        ...     rna_grouped, atac_grouped,
        ...     n_top=3000,
        ...     metric='var_mean_ratio'
        ... )
        >>> print(f"Selected {len(genes)} genes")

    Notes:
        - Union mode (default): More inclusive, captures modality-specific patterns
        - Intersection mode: More stringent, only highly variable in both
        - Common metrics: 'var_mean_ratio', 'robust_var_median_ratio', 'cv'
    """
    # Compute statistics
    rna_stats = compute_gene_stats(rna_meta)
    atac_stats = compute_gene_stats(atac_meta)

    # Get top genes for each modality
    rna_top = set(rna_stats.nlargest(n_top, metric).index)
    atac_top = set(atac_stats.nlargest(n_top, metric).index)

    # Combine
    if union:
        top_genes = set.union(rna_top, atac_top)
    else:
        top_genes = set.intersection(rna_top, atac_top)

    top_genes = np.array(list(top_genes))

    print(f"Selected {len(top_genes)} genes (n_top={n_top}, metric={metric}, union={union})")
    print(f"  RNA-specific: {len(rna_top - atac_top)}")
    print(f"  ATAC-specific: {len(atac_top - rna_top)}")
    print(f"  Shared: {len(rna_top.intersection(atac_top))}")

    # Filter
    rna_filtered = rna_meta[:, rna_meta.var_names.isin(top_genes)]
    atac_filtered = atac_meta[:, atac_meta.var_names.isin(top_genes)]

    return rna_filtered, atac_filtered, top_genes
