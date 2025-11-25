"""
Data processing utilities for RNA-ATAC multiome analysis.

This module provides functions for matrix manipulation, AnnData object creation,
and data formatting operations commonly used in multiome data analysis workflows.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional, Literal, Tuple


def process_matrices(df_RNA: pd.DataFrame, df_ATAC: pd.DataFrame) -> pd.DataFrame:
    """
    Process RNA and ATAC matrices by transposing, renaming columns, and concatenating.

    Combines RNA and ATAC expression matrices into a single genes-by-features matrix
    where each gene has both RNA and ATAC measurements across timepoints/conditions.

    Args:
        df_RNA: RNA expression matrix (timepoints × genes)
        df_ATAC: ATAC expression matrix (timepoints × genes)

    Returns:
        Combined genes-by-features DataFrame with columns labeled as:
        - '{timepoint}-RNA' for RNA measurements
        - '{timepoint}-ATAC' for ATAC measurements

    Example:
        >>> df_RNA = pd.DataFrame(
        ...     [[1, 2], [3, 4], [5, 6]],
        ...     index=['0somites', '15somites', '30somites'],
        ...     columns=['gene1', 'gene2']
        ... )
        >>> df_ATAC = pd.DataFrame(
        ...     [[10, 20], [30, 40], [50, 60]],
        ...     index=['0somites', '15somites', '30somites'],
        ...     columns=['gene1', 'gene2']
        ... )
        >>> result = process_matrices(df_RNA, df_ATAC)
        >>> result.shape  # (2 genes, 6 features: 3 RNA + 3 ATAC)
        (2, 6)
        >>> '0somites-RNA' in result.columns
        True
    """
    # 1. Transpose both matrices (timepoints become columns, genes become rows)
    df_RNA_t = df_RNA.transpose()
    df_ATAC_t = df_ATAC.transpose()

    # 2. Rename columns to include data type
    df_RNA_t.columns = [f'{col}-RNA' for col in df_RNA_t.columns]
    df_ATAC_t.columns = [f'{col}-ATAC' for col in df_ATAC_t.columns]

    # 3. Concatenate the matrices horizontally
    result = pd.concat([df_RNA_t, df_ATAC_t], axis=1)

    return result


def create_adata_and_umap(
    result_df: pd.DataFrame,
    norm_method: Literal['zscore', 'robust', 'minmax', 'log', 'none'] = 'zscore',
    n_pcs: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.5
) -> ad.AnnData:
    """
    Create AnnData object with genes as observations and generate UMAP embedding.

    Takes a genes-by-features DataFrame (output of process_matrices), applies
    normalization, computes PCA, and generates UMAP coordinates for visualization.

    Args:
        result_df: DataFrame with genes as rows and timepoint-datatypes as columns
        norm_method: Normalization method to apply:
            - 'zscore': Z-score standardization (mean=0, std=1)
            - 'robust': Robust scaling using median and IQR
            - 'minmax': Min-max scaling to [0, 1] range
            - 'log': Log1p transformation
            - 'none': No normalization
        n_pcs: Number of principal components for PCA
        n_neighbors: Number of neighbors for UMAP computation
        min_dist: Minimum distance parameter for UMAP

    Returns:
        AnnData object with:
        - .X: Normalized expression matrix
        - .layers['raw']: Original unnormalized data
        - .obsm['X_pca']: PCA coordinates
        - .obsm['X_umap']: UMAP coordinates
        - .var['timepoint']: Timepoint annotation for each feature
        - .var['data_type']: Data type ('RNA' or 'ATAC') for each feature

    Example:
        >>> result_df = process_matrices(df_RNA, df_ATAC)
        >>> adata = create_adata_and_umap(result_df, norm_method='zscore', n_pcs=30)
        >>> # Access UMAP coordinates
        >>> umap_coords = adata.obsm['X_umap']
        >>> # Access original data
        >>> raw_data = adata.layers['raw']
    """
    # Create AnnData object (genes as observations)
    adata = ad.AnnData(X=result_df.values)

    # Set observations (genes) and variables (timepoints)
    adata.obs_names = result_df.index  # Genes as observations
    adata.var_names = result_df.columns  # Timepoints as variables

    # Add variable metadata by parsing column names (format: 'timepoint-datatype')
    adata.var['timepoint'] = [col.split('-')[0] for col in adata.var_names]
    adata.var['data_type'] = [col.split('-')[1] for col in adata.var_names]

    # Store raw data
    adata.layers['raw'] = adata.X.copy()

    # Apply normalization based on selected method
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
        # Log transformation (log1p handles zeros)
        adata.X = np.log1p(adata.X)

    elif norm_method == 'none':
        # No normalization
        pass

    else:
        raise ValueError(
            f"Invalid normalization method: {norm_method}. "
            f"Choose from: 'zscore', 'robust', 'minmax', 'log', 'none'"
        )

    # Run PCA (ensure n_pcs doesn't exceed data dimensions)
    n_pcs_safe = min(n_pcs, min(adata.X.shape) - 1)
    sc.tl.pca(adata, n_comps=n_pcs_safe)

    # Generate UMAP (ensure n_neighbors doesn't exceed number of observations)
    n_neighbors_safe = min(n_neighbors, len(adata.obs_names) - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors_safe)
    sc.tl.umap(adata, min_dist=min_dist)

    return adata


def replace_periods_with_underscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace periods with underscores in DataFrame column names.

    Useful for preparing data for tools like CellxGene that have restrictions
    on special characters in column names.

    Args:
        df: DataFrame with potentially period-containing column names

    Returns:
        DataFrame with periods replaced by underscores in column names

    Example:
        >>> df = pd.DataFrame({'col.1': [1, 2], 'col.2': [3, 4]})
        >>> df_clean = replace_periods_with_underscores(df)
        >>> list(df_clean.columns)
        ['col_1', 'col_2']
    """
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.replace('.', '_', regex=False)
    return df_copy


def compute_pseudobulk_by_timepoint(
    adata: ad.AnnData,
    groupby_col: str = 'dev_stage',
    layer: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute pseudobulk expression by aggregating cells within each timepoint.

    Calculates mean expression for each gene across all cells within each
    timepoint/group, effectively creating bulk-like profiles.

    Args:
        adata: AnnData object with single-cell data
        groupby_col: Column in adata.obs to group cells by (e.g., 'dev_stage', 'timepoint')
        layer: Optional layer to use for expression values. If None, uses .X

    Returns:
        DataFrame with genes as columns and timepoints as rows, containing
        mean expression values

    Example:
        >>> # Assuming adata has 'dev_stage' annotation
        >>> pseudobulk = compute_pseudobulk_by_timepoint(adata, groupby_col='dev_stage')
        >>> # Result: timepoints x genes matrix with mean expression
        >>> pseudobulk.shape
        (6, 25000)  # 6 timepoints, 25k genes
    """
    if groupby_col not in adata.obs.columns:
        raise ValueError(f"Column '{groupby_col}' not found in adata.obs")

    # Get expression matrix
    if layer is not None:
        expr_df = pd.DataFrame(
            adata.layers[layer],
            index=adata.obs_names,
            columns=adata.var_names
        )
    else:
        expr_df = adata.to_df()

    # Add grouping column
    expr_df[groupby_col] = adata.obs[groupby_col]

    # Compute mean expression for each group
    pseudobulk = expr_df.groupby(groupby_col).mean()

    return pseudobulk


def split_by_modality(
    adata: ad.AnnData,
    modality_col: str = 'feature_types'
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Split a multiome AnnData object into separate RNA and ATAC objects.

    Args:
        adata: AnnData object containing both RNA and ATAC features
        modality_col: Column in adata.var specifying feature type
                     (e.g., 'Gene Expression', 'Peaks')

    Returns:
        Tuple of (adata_RNA, adata_ATAC)

    Example:
        >>> adata = sc.read_h5ad('multiome_data.h5ad')
        >>> adata_RNA, adata_ATAC = split_by_modality(adata)
        >>> print(f"RNA: {adata_RNA.n_vars} genes")
        >>> print(f"ATAC: {adata_ATAC.n_vars} peaks")
    """
    if modality_col not in adata.var.columns:
        raise ValueError(f"Column '{modality_col}' not found in adata.var")

    # Identify RNA and ATAC features
    rna_mask = adata.var[modality_col].isin(['Gene Expression', 'RNA', 'GEX'])
    atac_mask = adata.var[modality_col].isin(['Peaks', 'ATAC', 'Chromatin Accessibility'])

    # Subset for each modality
    adata_RNA = adata[:, rna_mask].copy()
    adata_ATAC = adata[:, atac_mask].copy()

    return adata_RNA, adata_ATAC


def print_matrix_info(df: pd.DataFrame, name: str = "Matrix") -> None:
    """
    Print summary information about a DataFrame/matrix.

    Utility function for debugging and data exploration.

    Args:
        df: DataFrame to summarize
        name: Name to display in output

    Example:
        >>> df = pd.DataFrame(np.random.randn(100, 50))
        >>> print_matrix_info(df, "Expression Matrix")
    """
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few column names:")
    print(df.columns[:6].tolist())
    print(f"\nFirst few row names:")
    print(df.index[:6].tolist())
    print(f"\nFirst few values:")
    print(df.iloc[:3, :3])
    print(f"\nData type: {df.dtypes[0] if len(df.dtypes) > 0 else 'N/A'}")
    print(f"Contains NaN: {df.isna().any().any()}")
