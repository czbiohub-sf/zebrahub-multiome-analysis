# Peak Cluster to GRN Pipeline - GRN Extraction Module
# Author: YangJoon Kim
# Date: 2025-06-25
# Description: Extract meaningful sub-GRNs from large GRN using putative relationships

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import glob
import pickle
import warnings

def load_celloracle_grn(
    grn_path: str, 
    cell_type: str = None, 
    timepoint: str = None,
    file_pattern: str = "*.csv"
) -> pd.DataFrame:
    """
    Load CellOracle GRN for specific cell type and timepoint.
    
    Parameters:
    -----------
    grn_path : str
        Path to GRN files directory or specific file
    cell_type : str, optional
        Cell type of interest
    timepoint : str, optional
        Timepoint of interest
    file_pattern : str
        File pattern to match GRN files
        
    Returns:
    --------
    grn_df : pd.DataFrame
        TFs x genes matrix with edge strengths
    """
    if os.path.isfile(grn_path):
        # Single file provided
        grn_df = pd.read_csv(grn_path, index_col=0)
        print(f"Loaded GRN from {grn_path}: {grn_df.shape}")
        return grn_df
    
    elif os.path.isdir(grn_path):
        # Directory provided, search for matching files
        search_patterns = []
        
        if cell_type and timepoint:
            search_patterns.extend([
                f"*{cell_type}*{timepoint}*{file_pattern}",
                f"*{timepoint}*{cell_type}*{file_pattern}",
                f"{cell_type}_{timepoint}*{file_pattern}",
                f"{timepoint}_{cell_type}*{file_pattern}"
            ])
        elif cell_type:
            search_patterns.append(f"*{cell_type}*{file_pattern}")
        elif timepoint:
            search_patterns.append(f"*{timepoint}*{file_pattern}")
        else:
            search_patterns.append(file_pattern)
        
        # Find matching files
        matching_files = []
        for pattern in search_patterns:
            full_pattern = os.path.join(grn_path, pattern)
            matching_files.extend(glob.glob(full_pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"No GRN files found matching criteria in {grn_path}")
        
        # Use the first matching file
        grn_file = matching_files[0]
        if len(matching_files) > 1:
            print(f"Multiple files found, using: {grn_file}")
        
        grn_df = pd.read_csv(grn_file, index_col=0)
        print(f"Loaded GRN from {grn_file}: {grn_df.shape}")
        return grn_df
    
    else:
        raise FileNotFoundError(f"GRN path not found: {grn_path}")

def load_celloracle_grn_from_pickle(
    pickle_path: str,
    cell_type: str = None,
    timepoint: str = None
) -> pd.DataFrame:
    """
    Load CellOracle GRN from pickle file.
    
    Parameters:
    -----------
    pickle_path : str
        Path to pickle file containing GRN data
    cell_type : str, optional
        Cell type to extract
    timepoint : str, optional
        Timepoint to extract
        
    Returns:
    --------
    grn_df : pd.DataFrame
        TFs x genes matrix with edge strengths
    """
    with open(pickle_path, 'rb') as f:
        grn_data = pickle.load(f)
    
    # Handle different pickle formats
    if isinstance(grn_data, dict):
        # If data is a dictionary, try to find the relevant GRN
        if cell_type and cell_type in grn_data:
            grn_df = grn_data[cell_type]
        elif timepoint and timepoint in grn_data:
            grn_df = grn_data[timepoint]
        else:
            # Use the first available GRN
            grn_df = list(grn_data.values())[0]
    else:
        # Assume it's a DataFrame
        grn_df = grn_data
    
    print(f"Loaded GRN from pickle {pickle_path}: {grn_df.shape}")
    return grn_df

def extract_subgrn_from_putative(
    full_grn_df: pd.DataFrame,
    putative_tf_target_matrix: pd.DataFrame,
    edge_strength_threshold: float = 0.1,
    keep_only_putative: bool = True
) -> pd.DataFrame:
    """
    Extract sub-GRN based on putative TF-target relationships.
    
    Parameters:
    -----------
    full_grn_df : pd.DataFrame
        Full CellOracle GRN (TFs x genes with edge strengths)
    putative_tf_target_matrix : pd.DataFrame
        Binary matrix of putative relationships (TFs x genes)
    edge_strength_threshold : float
        Minimum edge strength to keep from full GRN
    keep_only_putative : bool
        If True, only keep edges that are in putative matrix
        
    Returns:
    --------
    subgrn_df : pd.DataFrame
        Filtered sub-GRN with only putative relationships above threshold
    """
    # Find common TFs and genes
    common_tfs = full_grn_df.index.intersection(putative_tf_target_matrix.index)
    common_genes = full_grn_df.columns.intersection(putative_tf_target_matrix.columns)
    
    print(f"Common TFs: {len(common_tfs)}/{len(putative_tf_target_matrix.index)}")
    print(f"Common genes: {len(common_genes)}/{len(putative_tf_target_matrix.columns)}")
    
    if len(common_tfs) == 0 or len(common_genes) == 0:
        warnings.warn("No common TFs or genes found between GRN and putative matrix")
        return pd.DataFrame()
    
    # Subset both matrices to common elements
    full_grn_subset = full_grn_df.loc[common_tfs, common_genes]
    putative_subset = putative_tf_target_matrix.loc[common_tfs, common_genes]
    
    # Apply edge strength threshold
    strong_edges_mask = np.abs(full_grn_subset) >= edge_strength_threshold
    
    if keep_only_putative:
        # Keep only edges that are both strong and in putative matrix
        putative_mask = putative_subset > 0
        final_mask = strong_edges_mask & putative_mask
    else:
        # Keep all strong edges, but weight by putative relationships
        final_mask = strong_edges_mask
    
    # Create sub-GRN
    subgrn_df = full_grn_subset.copy()
    subgrn_df[~final_mask] = 0
    
    # Remove TFs and genes with no connections
    subgrn_df = remove_empty_nodes(subgrn_df)
    
    print(f"Extracted sub-GRN: {subgrn_df.shape}")
    print(f"Non-zero edges: {(subgrn_df != 0).sum().sum()}")
    
    return subgrn_df

def remove_empty_nodes(grn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove TFs and genes with no connections from GRN.
    
    Parameters:
    -----------
    grn_df : pd.DataFrame
        GRN matrix (TFs x genes)
        
    Returns:
    --------
    pd.DataFrame
        GRN with empty nodes removed
    """
    # Remove TFs with no outgoing edges
    active_tfs = (grn_df != 0).any(axis=1)
    grn_df = grn_df.loc[active_tfs]
    
    # Remove genes with no incoming edges
    active_genes = (grn_df != 0).any(axis=0)
    grn_df = grn_df.loc[:, active_genes]
    
    return grn_df

def validate_subgrn_enrichment(
    subgrn_edges: pd.DataFrame,
    full_grn_edges: pd.DataFrame,
    putative_tf_target_matrix: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Validate that extracted sub-GRN is enriched for strong edges.
    
    Parameters:
    -----------
    subgrn_edges : pd.DataFrame
        Sub-GRN edges
    full_grn_edges : pd.DataFrame
        Full GRN edges
    putative_tf_target_matrix : pd.DataFrame, optional
        Putative TF-target relationships for additional validation
        
    Returns:
    --------
    validation_metrics : Dict[str, float]
        {metric_name: value} - e.g., mean edge strength comparison
    """
    # Find common elements for comparison
    common_tfs = subgrn_edges.index.intersection(full_grn_edges.index)
    common_genes = subgrn_edges.columns.intersection(full_grn_edges.columns)
    
    if len(common_tfs) == 0 or len(common_genes) == 0:
        return {'error': 'No common elements for validation'}
    
    # Subset to common elements
    subgrn_subset = subgrn_edges.loc[common_tfs, common_genes]
    full_grn_subset = full_grn_edges.loc[common_tfs, common_genes]
    
    # Calculate metrics
    metrics = {}
    
    # Edge strength comparison
    subgrn_nonzero = subgrn_subset[subgrn_subset != 0]
    full_grn_nonzero = full_grn_subset[full_grn_subset != 0]
    
    if len(subgrn_nonzero) > 0 and len(full_grn_nonzero) > 0:
        metrics['subgrn_mean_edge_strength'] = np.abs(subgrn_nonzero).mean().mean()
        metrics['full_grn_mean_edge_strength'] = np.abs(full_grn_nonzero).mean().mean()
        metrics['edge_strength_enrichment'] = (
            metrics['subgrn_mean_edge_strength'] / metrics['full_grn_mean_edge_strength']
        )
    
    # Edge density comparison
    metrics['subgrn_edge_density'] = (subgrn_subset != 0).sum().sum() / subgrn_subset.size
    metrics['full_grn_edge_density'] = (full_grn_subset != 0).sum().sum() / full_grn_subset.size
    
    # Coverage metrics
    metrics['tf_coverage'] = len(common_tfs) / len(subgrn_edges.index)
    metrics['gene_coverage'] = len(common_genes) / len(subgrn_edges.columns)
    
    # Putative relationship validation
    if putative_tf_target_matrix is not None:
        putative_common_tfs = common_tfs.intersection(putative_tf_target_matrix.index)
        putative_common_genes = common_genes.intersection(putative_tf_target_matrix.columns)
        
        if len(putative_common_tfs) > 0 and len(putative_common_genes) > 0:
            putative_subset = putative_tf_target_matrix.loc[putative_common_tfs, putative_common_genes]
            subgrn_putative_subset = subgrn_subset.loc[putative_common_tfs, putative_common_genes]
            
            # Calculate overlap with putative relationships
            putative_edges = (putative_subset > 0).sum().sum()
            subgrn_edges_in_putative = ((subgrn_putative_subset != 0) & (putative_subset > 0)).sum().sum()
            
            if putative_edges > 0:
                metrics['putative_edge_recovery'] = subgrn_edges_in_putative / putative_edges
    
    return metrics

def merge_cluster_subgrns(
    cluster_subgrns: Dict[str, pd.DataFrame],
    method: str = "union",
    weight_by_cluster_size: bool = False,
    cluster_sizes: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Merge sub-GRNs from multiple clusters.
    
    Parameters:
    -----------
    cluster_subgrns : Dict[str, pd.DataFrame]
        Sub-GRNs for each cluster
    method : str
        Merging method: "union", "intersection", or "average"
    weight_by_cluster_size : bool
        Whether to weight edges by cluster size
    cluster_sizes : Dict[str, int], optional
        Number of peaks/cells per cluster for weighting
        
    Returns:
    --------
    pd.DataFrame
        Merged sub-GRN
    """
    if not cluster_subgrns:
        return pd.DataFrame()
    
    # Get all TFs and genes
    all_tfs = set()
    all_genes = set()
    for subgrn in cluster_subgrns.values():
        all_tfs.update(subgrn.index)
        all_genes.update(subgrn.columns)
    
    all_tfs = sorted(list(all_tfs))
    all_genes = sorted(list(all_genes))
    
    # Initialize merged matrix
    merged_grn = pd.DataFrame(0.0, index=all_tfs, columns=all_genes)
    
    if method == "union":
        # Take maximum edge weight across clusters
        for cluster, subgrn in cluster_subgrns.items():
            # Align subgrn to merged matrix
            for tf in subgrn.index:
                for gene in subgrn.columns:
                    if tf in merged_grn.index and gene in merged_grn.columns:
                        merged_grn.loc[tf, gene] = max(
                            merged_grn.loc[tf, gene], 
                            abs(subgrn.loc[tf, gene])
                        )
    
    elif method == "intersection":
        # Only keep edges present in all clusters
        edge_counts = pd.DataFrame(0, index=all_tfs, columns=all_genes)
        edge_sums = pd.DataFrame(0.0, index=all_tfs, columns=all_genes)
        
        for cluster, subgrn in cluster_subgrns.items():
            for tf in subgrn.index:
                for gene in subgrn.columns:
                    if tf in merged_grn.index and gene in merged_grn.columns:
                        if subgrn.loc[tf, gene] != 0:
                            edge_counts.loc[tf, gene] += 1
                            edge_sums.loc[tf, gene] += abs(subgrn.loc[tf, gene])
        
        # Keep only edges present in all clusters
        n_clusters = len(cluster_subgrns)
        intersection_mask = edge_counts == n_clusters
        merged_grn[intersection_mask] = edge_sums[intersection_mask] / n_clusters
    
    elif method == "average":
        # Average edge weights across clusters
        edge_counts = pd.DataFrame(0, index=all_tfs, columns=all_genes)
        edge_sums = pd.DataFrame(0.0, index=all_tfs, columns=all_genes)
        
        for cluster, subgrn in cluster_subgrns.items():
            weight = 1.0
            if weight_by_cluster_size and cluster_sizes and cluster in cluster_sizes:
                weight = cluster_sizes[cluster]
            
            for tf in subgrn.index:
                for gene in subgrn.columns:
                    if tf in merged_grn.index and gene in merged_grn.columns:
                        edge_counts.loc[tf, gene] += weight
                        edge_sums.loc[tf, gene] += abs(subgrn.loc[tf, gene]) * weight
        
        # Calculate weighted average
        nonzero_mask = edge_counts > 0
        merged_grn[nonzero_mask] = edge_sums[nonzero_mask] / edge_counts[nonzero_mask]
    
    else:
        raise ValueError(f"Unknown merging method: {method}")
    
    # Remove empty nodes
    merged_grn = remove_empty_nodes(merged_grn)
    
    print(f"Merged {len(cluster_subgrns)} cluster sub-GRNs using {method} method")
    print(f"Final merged GRN: {merged_grn.shape}")
    
    return merged_grn

def filter_subgrn_by_degree(
    grn_df: pd.DataFrame,
    min_out_degree: int = 1,
    min_in_degree: int = 1
) -> pd.DataFrame:
    """
    Filter sub-GRN by node degree requirements.
    
    Parameters:
    -----------
    grn_df : pd.DataFrame
        GRN matrix (TFs x genes)
    min_out_degree : int
        Minimum out-degree for TFs
    min_in_degree : int
        Minimum in-degree for genes
        
    Returns:
    --------
    pd.DataFrame
        Filtered GRN
    """
    # Calculate degrees
    out_degrees = (grn_df != 0).sum(axis=1)  # TF out-degrees
    in_degrees = (grn_df != 0).sum(axis=0)   # Gene in-degrees
    
    # Filter by degree requirements
    valid_tfs = out_degrees[out_degrees >= min_out_degree].index
    valid_genes = in_degrees[in_degrees >= min_in_degree].index
    
    filtered_grn = grn_df.loc[valid_tfs, valid_genes]
    
    print(f"Filtered GRN by degree: {grn_df.shape} -> {filtered_grn.shape}")
    
    return filtered_grn

def get_subgrn_statistics(grn_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Compute statistics for a sub-GRN.
    
    Parameters:
    -----------
    grn_df : pd.DataFrame
        GRN matrix (TFs x genes)
        
    Returns:
    --------
    Dict[str, Union[int, float]]
        Statistics dictionary
    """
    stats = {
        'n_tfs': len(grn_df.index),
        'n_genes': len(grn_df.columns),
        'n_edges': (grn_df != 0).sum().sum(),
        'edge_density': (grn_df != 0).sum().sum() / grn_df.size,
        'mean_edge_strength': np.abs(grn_df[grn_df != 0]).mean().mean() if (grn_df != 0).any().any() else 0,
        'max_edge_strength': np.abs(grn_df).max().max(),
        'mean_tf_out_degree': (grn_df != 0).sum(axis=1).mean(),
        'mean_gene_in_degree': (grn_df != 0).sum(axis=0).mean()
    }
    
    return stats

def save_subgrn_results(
    cluster_subgrns: Dict[str, pd.DataFrame],
    output_dir: str,
    file_prefix: str = "subgrn"
):
    """
    Save sub-GRN results to files.
    
    Parameters:
    -----------
    cluster_subgrns : Dict[str, pd.DataFrame]
        Sub-GRNs for each cluster
    output_dir : str
        Output directory
    file_prefix : str
        Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for cluster, subgrn in cluster_subgrns.items():
        filename = f"{file_prefix}_cluster_{cluster}.csv"
        filepath = os.path.join(output_dir, filename)
        subgrn.to_csv(filepath)
        print(f"Saved sub-GRN for cluster {cluster} to {filepath}")
    
    # Save summary statistics
    summary_stats = []
    for cluster, subgrn in cluster_subgrns.items():
        stats = get_subgrn_statistics(subgrn)
        stats['cluster'] = cluster
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, f"{file_prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to {summary_path}")