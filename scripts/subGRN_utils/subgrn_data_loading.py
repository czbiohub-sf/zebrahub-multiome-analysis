"""
Data loading and preprocessing utilities for subGRN analysis

This module provides functions to load Gene Regulatory Networks (GRNs),
peak cluster data, motif enrichment scores, and linked genes from various
file formats.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_grn_dict_pathlib(base_dir: str, grn_type: str = "filtered") -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Load GRN dictionary using pathlib (more robust than os.path)

    This function loads cell-type and timepoint-specific GRN DataFrames from
    a directory structure: base_dir/grn_type/timepoint_XX/celltype.csv

    Parameters
    ----------
    base_dir : str
        Base directory containing GRN exports (e.g., "grn_exported/")
    grn_type : str, default="filtered"
        Type of GRN to load (subdirectory name, e.g., "filtered", "unfiltered")

    Returns
    -------
    dict
        Dictionary keyed by (celltype, timepoint) tuples, values are GRN DataFrames
        Example: {('neural_crest', '15'): DataFrame, ('PSM', '20'): DataFrame, ...}

    Notes
    -----
    - GRN DataFrames are expected to have columns: 'source', 'target', 'coef_mean', etc.
    - Timepoint is extracted from parent directory name (e.g., "timepoint_15" → "15")
    - Celltype is extracted from filename (e.g., "neural_crest.csv" → "neural_crest")

    Examples
    --------
    >>> grn_dict = load_grn_dict_pathlib(
    ...     base_dir="/path/to/grn_exported/",
    ...     grn_type="filtered"
    ... )
    >>> print(len(grn_dict))  # Number of celltype-timepoint combinations
    189
    >>> grn_df = grn_dict[('neural_crest', '15')]
    >>> print(grn_df.columns)
    Index(['source', 'target', 'coef_mean', 'coef_std', ...])
    """
    grn_dict = {}
    base_path = Path(base_dir) / grn_type

    if not base_path.exists():
        raise FileNotFoundError(f"GRN directory not found: {base_path}")

    # Find all CSV files recursively
    csv_files = list(base_path.glob("*/*.csv"))

    if len(csv_files) == 0:
        logger.warning(f"No CSV files found in {base_path}")
        return grn_dict

    logger.info(f"Loading {len(csv_files)} GRN files from {base_path}")

    for csv_file in csv_files:
        # Extract timepoint from parent directory
        timepoint_dir = csv_file.parent.name
        timepoint = timepoint_dir.split('_')[1] if 'timepoint_' in timepoint_dir else timepoint_dir

        # Extract celltype from filename
        celltype = csv_file.stem  # filename without extension

        # Load GRN DataFrame
        try:
            grn_df = pd.read_csv(csv_file)
            grn_dict[(celltype, timepoint)] = grn_df
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")
            continue

    logger.info(f"Successfully loaded {len(grn_dict)} GRN combinations")
    return grn_dict


def load_peak_adata(file_path: str) -> sc.AnnData:
    """
    Load peaks-by-pseudobulk AnnData object

    Parameters
    ----------
    file_path : str
        Path to H5AD file containing peak accessibility data

    Returns
    -------
    sc.AnnData
        AnnData object with peak accessibility matrix

    Examples
    --------
    >>> adata_peaks = load_peak_adata(
    ...     "/path/to/peaks_by_pb_annotated_master.h5ad"
    ... )
    >>> print(adata_peaks.shape)
    (n_peaks, n_celltypes_x_timepoints)
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Peak AnnData file not found: {file_path}")

    logger.info(f"Loading peak AnnData from {file_path}")
    adata_peaks = sc.read_h5ad(file_path)
    logger.info(f"Loaded AnnData with shape {adata_peaks.shape}")

    return adata_peaks


def load_motif_enrichment(file_path: str, index_col: int = 0) -> pd.DataFrame:
    """
    Load cluster-by-motifs enrichment matrix from GimmeMotifs maelstrom output

    Parameters
    ----------
    file_path : str
        Path to CSV file containing motif enrichment scores
    index_col : int, default=0
        Column to use as row index

    Returns
    -------
    pd.DataFrame
        Matrix of enrichment scores (clusters × motifs)

    Examples
    --------
    >>> clust_by_motifs = load_motif_enrichment(
    ...     "/path/to/leiden_fine_motifs_maelstrom.csv"
    ... )
    >>> print(clust_by_motifs.shape)
    (n_clusters, n_motifs)
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Motif enrichment file not found: {file_path}")

    logger.info(f"Loading motif enrichment matrix from {file_path}")
    clust_by_motifs = pd.read_csv(file_path, index_col=index_col)
    logger.info(f"Loaded matrix with shape {clust_by_motifs.shape}")

    return clust_by_motifs


def load_cluster_pseudobulk_accessibility(file_path: str, index_col: int = 0) -> pd.DataFrame:
    """
    Load cluster-by-pseudobulk accessibility matrix

    Parameters
    ----------
    file_path : str
        Path to CSV file containing cluster accessibility across celltypes/timepoints
    index_col : int, default=0
        Column to use as row index

    Returns
    -------
    pd.DataFrame
        Matrix of accessibility values (clusters × celltypes_timepoints)

    Examples
    --------
    >>> df_clusters_groups = load_cluster_pseudobulk_accessibility(
    ...     "/path/to/leiden_fine_by_pseudobulk.csv"
    ... )
    >>> print(df_clusters_groups.shape)
    (n_clusters, n_celltype_timepoint_combinations)
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Cluster accessibility file not found: {file_path}")

    logger.info(f"Loading cluster accessibility matrix from {file_path}")
    df_clusters_groups = pd.read_csv(file_path, index_col=index_col)
    logger.info(f"Loaded matrix with shape {df_clusters_groups.shape}")

    return df_clusters_groups


def validate_grn_dataframe(grn_df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validate that a GRN DataFrame has required columns

    Parameters
    ----------
    grn_df : pd.DataFrame
        GRN DataFrame to validate
    required_columns : list, optional
        List of required column names. Defaults to ['source', 'target']

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise

    Raises
    ------
    ValueError
        If required columns are missing

    Examples
    --------
    >>> grn_df = pd.DataFrame({'source': ['gata1'], 'target': ['tal1']})
    >>> validate_grn_dataframe(grn_df)
    True
    """
    if required_columns is None:
        required_columns = ['source', 'target']

    missing_cols = [col for col in required_columns if col not in grn_df.columns]

    if missing_cols:
        raise ValueError(f"GRN DataFrame missing required columns: {missing_cols}")

    if len(grn_df) == 0:
        logger.warning("GRN DataFrame is empty")

    return True


# Utility function for consistent file path handling
def get_data_path(relative_path: str, base_dir: Optional[str] = None) -> Path:
    """
    Get absolute path to data file with optional base directory

    Parameters
    ----------
    relative_path : str
        Relative path to data file
    base_dir : str, optional
        Base directory. If None, uses relative path as-is

    Returns
    -------
    Path
        Absolute Path object

    Examples
    --------
    >>> path = get_data_path(
    ...     "processed_data/grn_exported/",
    ...     base_dir="/hpc/projects/data.science/user/zebrahub_multiome/data/"
    ... )
    """
    if base_dir is None:
        return Path(relative_path).resolve()
    else:
        return (Path(base_dir) / relative_path).resolve()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Example: Load GRN dictionary
    grn_dict = load_grn_dict_pathlib(
        base_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/grn_exported/",
        grn_type="filtered"
    )
    print(f"Loaded {len(grn_dict)} GRN combinations")
    print(f"Example keys: {list(grn_dict.keys())[:5]}")
