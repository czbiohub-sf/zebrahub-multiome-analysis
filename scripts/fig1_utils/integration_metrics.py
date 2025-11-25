"""
Integration quality metrics for multimodal single-cell analysis.

This module provides utilities for evaluating the quality of multimodal integration
in single-cell RNA+ATAC (multiome) data. It includes functions for:

- Loading Weighted Nearest Neighbor (WNN) graphs from R/Seurat exports
- Extracting RNA and ATAC modality weights from WNN integration
- Computing neighborhood purity metrics across modalities
- Evaluating integration quality across RNA, ATAC, and WNN embeddings

These metrics help assess how well the multimodal integration preserves biological
structure and whether different modalities provide complementary information.

Dependencies:
    - Core purity computations use scripts/neighborhood_purity_analysis.py
    - WNN graph loading requires scipy.io for Matrix Market files
"""

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
from typing import Dict, List, Optional, Tuple
import warnings

# Import existing neighborhood purity functions
try:
    from scripts.neighborhood_purity_analysis import (
        compute_multimodal_knn_purity,
        summarize_purity_scores
    )
    PURITY_ANALYSIS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "scripts.neighborhood_purity_analysis not available. "
        "Neighborhood purity functions will not work."
    )
    PURITY_ANALYSIS_AVAILABLE = False


def load_wnn_graph(
    mtx_file: str,
    cell_names_file: str,
    adata,
    verbose: bool = True
) -> scipy.sparse.csr_matrix:
    """
    Load Weighted Nearest Neighbor (WNN) graph from R/Seurat export.

    Reads a WNN connectivity matrix exported from Seurat/R and aligns it with
    an AnnData object. Handles cell ordering differences between Seurat and
    AnnData representations.

    Args:
        mtx_file: Path to Matrix Market (.mtx) file containing WNN connectivities
        cell_names_file: Path to text file with cell names (one per line)
        adata: AnnData object to align the WNN graph with
        verbose: Print alignment diagnostics

    Returns:
        Sparse CSR matrix with WNN connectivities aligned to adata cell order

    Raises:
        ValueError: If cell alignment fails or critical cells are missing

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('multiome_data.h5ad')
        >>> wnn = load_wnn_graph(
        ...     'wsnn_matrix.mtx',
        ...     'cell_names.txt',
        ...     adata
        ... )
        >>> adata.obsp['connectivities_wnn'] = wnn

    Notes:
        - Seurat cell names may differ in format from AnnData (e.g., hyphens vs periods)
        - Function performs robust alignment checking first 10 and last 10 cells
        - If alignment fails, performs full reordering which may be slow for large datasets
    """
    # Read the sparse matrix from Matrix Market format
    wsnn_matrix = scipy.io.mmread(mtx_file)

    # Convert to CSR format (more efficient for row operations)
    wsnn_matrix = wsnn_matrix.tocsr()

    # Read cell names from text file
    with open(cell_names_file, 'r') as f:
        seurat_cell_names = [line.strip() for line in f.readlines()]

    if verbose:
        print(f"WNN matrix shape: {wsnn_matrix.shape}")
        print(f"Number of cell names from Seurat: {len(seurat_cell_names)}")
        print(f"AnnData shape: {adata.shape}")

    # Get AnnData cell names
    adata_cells = adata.obs_names.tolist()

    # Quick check: if lengths match, verify if order is identical
    if len(seurat_cell_names) == len(adata_cells):
        if verbose:
            print("Same number of cells - checking if order matches...")

        # Check first 10 and last 10 cells for quick verification
        n_check = min(10, len(seurat_cell_names))
        first_match = all(seurat_cell_names[i] == adata_cells[i] for i in range(n_check))
        last_match = all(seurat_cell_names[-(i+1)] == adata_cells[-(i+1)] for i in range(n_check))

        if first_match and last_match:
            if verbose:
                print("✓ Cell orders appear to match! Skipping reordering.")
            return wsnn_matrix
        else:
            if verbose:
                print("✗ Cell orders don't match - will need reordering")
    else:
        if verbose:
            print("Different number of cells - will need subsetting and reordering")

    # Fallback: perform cell alignment
    if verbose:
        print("Proceeding with cell alignment...")

    # Check which cells are present in both datasets
    adata_cells_set = set(adata_cells)
    seurat_cells_set = set(seurat_cell_names)

    missing_in_seurat = adata_cells_set - seurat_cells_set
    extra_in_seurat = seurat_cells_set - adata_cells_set

    if verbose:
        print(f"Cells in AnnData: {len(adata_cells_set)}")
        print(f"Cells in Seurat: {len(seurat_cells_set)}")
        print(f"Missing in Seurat: {len(missing_in_seurat)}")
        print(f"Extra in Seurat: {len(extra_in_seurat)}")

    # Create mapping from Seurat order to AnnData order
    seurat_to_adata_idx = {}
    for i, cell in enumerate(seurat_cell_names):
        if cell in adata_cells_set:
            adata_idx = adata_cells.index(cell)
            seurat_to_adata_idx[i] = adata_idx

    # Get indices of cells that exist in both
    keep_seurat_idx = list(seurat_to_adata_idx.keys())
    keep_adata_idx = list(seurat_to_adata_idx.values())

    if len(keep_seurat_idx) == 0:
        raise ValueError("No overlapping cells found between Seurat and AnnData!")

    # Subset and reorder the matrix to match AnnData cell order
    # First subset to common cells
    wsnn_subset = wsnn_matrix[keep_seurat_idx, :][:, keep_seurat_idx]

    # Then reorder to match AnnData cell ordering
    reorder_idx = np.argsort(keep_adata_idx)
    final_seurat_idx = [keep_seurat_idx[i] for i in reorder_idx]
    wsnn_reordered = wsnn_matrix[final_seurat_idx, :][:, final_seurat_idx]

    if verbose:
        print(f"✓ Successfully aligned {wsnn_reordered.shape[0]} cells")

    return wsnn_reordered


def load_rna_atac_weights(
    rna_weights_file: str,
    adata,
    rna_weight_col: str = 'rna_weights',
    verbose: bool = True
) -> None:
    """
    Load RNA and ATAC modality weights from WNN integration.

    Reads modality weights exported from Seurat's WNN integration and adds them
    to the AnnData object. ATAC weights are computed as (1 - RNA_weight).

    Args:
        rna_weights_file: Path to CSV file with RNA weights (index=cell names)
        adata: AnnData object to add weights to
        rna_weight_col: Column name for RNA weights in the CSV file
        verbose: Print loading diagnostics

    Returns:
        None (modifies adata.obs in place, adding 'rna_weights' and 'atac_weights' columns)

    Example:
        >>> load_rna_atac_weights('RNA_weights.csv', adata)
        >>> # Access weights
        >>> rna_wt = adata.obs['rna_weights']
        >>> atac_wt = adata.obs['atac_weights']
        >>> # Cells with high RNA weight (RNA-dominant integration)
        >>> rna_dominant = adata[adata.obs['rna_weights'] > 0.7]

    Notes:
        - RNA weight > 0.5: RNA modality more informative for this cell
        - ATAC weight > 0.5: ATAC modality more informative for this cell
        - Weights close to 0.5: Both modalities equally informative
    """
    # Read RNA weights from CSV
    rna_weights = pd.read_csv(rna_weights_file, index_col=0)

    if verbose:
        print(f"Loaded RNA weights for {len(rna_weights)} cells")

    # Align with AnnData cells
    if rna_weight_col not in rna_weights.columns:
        raise ValueError(f"Column '{rna_weight_col}' not found in RNA weights file")

    # Reindex to match AnnData order, filling missing with NaN
    aligned_weights = rna_weights.reindex(adata.obs_names)[rna_weight_col]

    # Check for missing cells
    n_missing = aligned_weights.isna().sum()
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} cells in AnnData do not have RNA weights. "
            f"These will be set to NaN."
        )

    # Add to AnnData
    adata.obs['rna_weights'] = aligned_weights.values
    adata.obs['atac_weights'] = 1 - aligned_weights.values

    if verbose:
        print(f"✓ Added 'rna_weights' and 'atac_weights' to adata.obs")
        print(f"Mean RNA weight: {adata.obs['rna_weights'].mean():.3f}")
        print(f"Mean ATAC weight: {adata.obs['atac_weights'].mean():.3f}")


def compute_comprehensive_neighborhood_purity_wrapper(
    adata,
    connectivity_keys: Optional[Dict[str, str]] = None,
    cluster_keys: Optional[Dict[str, str]] = None,
    k_neighbors: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    Compute neighborhood purity across all modalities and clustering methods.

    Wrapper around scripts/neighborhood_purity_analysis.compute_multimodal_knn_purity()
    that processes multiple clustering annotations and modalities systematically.

    Args:
        adata: AnnData object with precomputed connectivity matrices in obsp
        connectivity_keys: Dict mapping modality names to connectivity matrix keys.
                          Default: {'RNA': 'RNA_connectivities',
                                   'ATAC': 'ATAC_connectivities',
                                   'WNN': 'connectivities_wnn'}
        cluster_keys: Dict mapping clustering names to metadata column names.
                     Default: {'RNA': 'RNA_leiden_0.8_merged',
                              'ATAC': 'ATAC_leiden_0.5_merged',
                              'WNN': 'WNN_leiden_0.35_merged'}
        k_neighbors: Number of nearest neighbors to evaluate purity over

    Returns:
        Dictionary mapping cluster names to purity result DataFrames.
        Each DataFrame contains purity scores per cell for each modality.

    Example:
        >>> # Compute purity across all modalities
        >>> results = compute_comprehensive_neighborhood_purity_wrapper(adata)
        >>>
        >>> # Access RNA clustering results
        >>> rna_purity = results['RNA']
        >>>
        >>> # Plot distribution of WNN purity for WNN clusters
        >>> import seaborn as sns
        >>> sns.violinplot(data=results['WNN'], x='cluster', y='WNN_purity')

    Notes:
        - Requires connectivity matrices precomputed in adata.obsp
        - Cluster keys must exist in adata.obs
        - Uses existing neighborhood_purity_analysis module for core computation
    """
    if not PURITY_ANALYSIS_AVAILABLE:
        raise ImportError(
            "scripts.neighborhood_purity_analysis module not available. "
            "Cannot compute neighborhood purity."
        )

    # Set default connectivity keys
    if connectivity_keys is None:
        connectivity_keys = {
            'RNA': 'RNA_connectivities',
            'ATAC': 'ATAC_connectivities',
            'WNN': 'connectivities_wnn'
        }

    # Set default cluster keys
    if cluster_keys is None:
        cluster_keys = {
            'RNA': 'RNA_leiden_0.8_merged',
            'ATAC': 'ATAC_leiden_0.5_merged',
            'WNN': 'WNN_leiden_0.35_merged'
        }

    print("Computing comprehensive neighborhood purity analysis...")

    # Store all results
    all_purity_results = {}

    # For each clustering metadata
    for cluster_name, cluster_key in cluster_keys.items():
        print(f"\nAnalyzing {cluster_name} clusters ({cluster_key})...")

        # Check if cluster key exists
        if cluster_key not in adata.obs.columns:
            print(f"Warning: {cluster_key} not found in adata.obs, skipping")
            continue

        # Compute purity scores using pre-computed connectivities
        purity_results = compute_multimodal_knn_purity(
            adata=adata,
            connectivity_keys=connectivity_keys,
            metadata_key=cluster_key,
            k=k_neighbors
        )

        # Store results
        all_purity_results[cluster_name] = purity_results

        # Quick summary
        summary_purity = summarize_purity_scores(purity_results, adata, cluster_key)
        overall_summary = summary_purity[summary_purity['Metadata'] == 'Overall'][
            ['Modality', 'Mean_Purity', 'Std_Purity']
        ]
        print(f"Purity Summary for {cluster_name} clusters:")
        print(overall_summary.to_string(index=False))

    return all_purity_results
