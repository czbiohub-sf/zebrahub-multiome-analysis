# Peak Cluster to GRN Pipeline - Data Processing Module
# Author: YangJoon Kim
# Date: 2025-06-25
# Description: Handle input data and cluster-level aggregation for peak cluster to GRN analysis

import pandas as pd
import numpy as np
import scanpy as sc
from typing import Tuple, Dict, List, Optional
import warnings

def aggregate_peaks_by_clusters(
    peaks_motifs_adata, 
    peaks_genes_adata, 
    cluster_resolution: str = "coarse"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate peak-level matrices to cluster-level matrices.
    
    Parameters:
    -----------
    peaks_motifs_adata : AnnData
        Peaks x motifs matrix with cluster labels in .obs
    peaks_genes_adata : AnnData  
        Peaks x genes matrix with cluster labels in .obs
    cluster_resolution : str
        "coarse" or "fine" - determines which leiden cluster labels to use
        
    Returns:
    --------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs aggregated matrix
    clusters_genes_df : pd.DataFrame
        Clusters x genes aggregated matrix
    """
    # Validate inputs
    if not validate_input_data(peaks_motifs_adata, peaks_genes_adata):
        raise ValueError("Input data validation failed")
    
    # Determine cluster column name
    if cluster_resolution == "coarse":
        cluster_col = "leiden_coarse"
    elif cluster_resolution == "fine":
        cluster_col = "leiden"
    else:
        raise ValueError(f"cluster_resolution must be 'coarse' or 'fine', got {cluster_resolution}")
    
    # Check if cluster column exists
    if cluster_col not in peaks_motifs_adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in peaks_motifs_adata.obs")
    if cluster_col not in peaks_genes_adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in peaks_genes_adata.obs")
    
    # Get cluster labels
    motifs_clusters = peaks_motifs_adata.obs[cluster_col].astype(str)
    genes_clusters = peaks_genes_adata.obs[cluster_col].astype(str)
    
    # Aggregate motifs data by clusters
    motifs_df = pd.DataFrame(
        peaks_motifs_adata.X.toarray() if hasattr(peaks_motifs_adata.X, 'toarray') else peaks_motifs_adata.X,
        index=peaks_motifs_adata.obs_names,
        columns=peaks_motifs_adata.var_names
    )
    motifs_df['cluster'] = motifs_clusters
    clusters_motifs_df = motifs_df.groupby('cluster').mean()
    
    # Aggregate genes data by clusters
    genes_df = pd.DataFrame(
        peaks_genes_adata.X.toarray() if hasattr(peaks_genes_adata.X, 'toarray') else peaks_genes_adata.X,
        index=peaks_genes_adata.obs_names,
        columns=peaks_genes_adata.var_names
    )
    genes_df['cluster'] = genes_clusters
    clusters_genes_df = genes_df.groupby('cluster').mean()
    
    print(f"Aggregated {len(motifs_df)} peaks into {len(clusters_motifs_df)} clusters")
    print(f"Motifs matrix shape: {clusters_motifs_df.shape}")
    print(f"Genes matrix shape: {clusters_genes_df.shape}")
    
    return clusters_motifs_df, clusters_genes_df

def validate_input_data(peaks_motifs_adata, peaks_genes_adata) -> bool:
    """
    Validate that input adata objects have required structure.
    
    Parameters:
    -----------
    peaks_motifs_adata : AnnData
        Peaks x motifs matrix
    peaks_genes_adata : AnnData
        Peaks x genes matrix
        
    Returns:
    --------
    bool
        True if validation passes, False otherwise
    """
    try:
        # Check if objects are AnnData
        if not isinstance(peaks_motifs_adata, sc.AnnData):
            print("Error: peaks_motifs_adata is not an AnnData object")
            return False
        if not isinstance(peaks_genes_adata, sc.AnnData):
            print("Error: peaks_genes_adata is not an AnnData object")
            return False
        
        # Check if observations match
        if not np.array_equal(peaks_motifs_adata.obs_names, peaks_genes_adata.obs_names):
            print("Error: Peak names do not match between motifs and genes matrices")
            return False
        
        # Check for required cluster columns
        required_cols = ['leiden', 'leiden_coarse']
        for col in required_cols:
            if col not in peaks_motifs_adata.obs.columns:
                print(f"Warning: {col} not found in peaks_motifs_adata.obs")
            if col not in peaks_genes_adata.obs.columns:
                print(f"Warning: {col} not found in peaks_genes_adata.obs")
        
        # Check data shapes
        print(f"Motifs data shape: {peaks_motifs_adata.shape}")
        print(f"Genes data shape: {peaks_genes_adata.shape}")
        
        if peaks_motifs_adata.shape[0] == 0 or peaks_genes_adata.shape[0] == 0:
            print("Error: Empty data matrices")
            return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False

def get_cluster_peak_counts(adata, cluster_col: str) -> pd.Series:
    """
    Get number of peaks per cluster for quality control.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with cluster labels
    cluster_col : str
        Column name containing cluster labels
        
    Returns:
    --------
    pd.Series
        Series with cluster IDs as index and peak counts as values
    """
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in adata.obs")
    
    cluster_counts = adata.obs[cluster_col].value_counts().sort_index()
    return cluster_counts

def filter_low_count_clusters(
    clusters_df: pd.DataFrame, 
    min_peaks: int = 10
) -> pd.DataFrame:
    """
    Filter out clusters with too few peaks.
    
    Parameters:
    -----------
    clusters_df : pd.DataFrame
        Cluster-level aggregated data
    min_peaks : int
        Minimum number of peaks required per cluster
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    # This function would need the original peak counts to filter properly
    # For now, return as-is with a warning
    warnings.warn("filter_low_count_clusters requires original peak count information")
    return clusters_df

def compute_cluster_statistics(
    clusters_motifs_df: pd.DataFrame,
    clusters_genes_df: pd.DataFrame
) -> Dict[str, Dict]:
    """
    Compute basic statistics for cluster-level data.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    clusters_genes_df : pd.DataFrame
        Clusters x genes matrix
        
    Returns:
    --------
    Dict[str, Dict]
        Statistics for motifs and genes data
    """
    stats = {
        'motifs': {
            'n_clusters': len(clusters_motifs_df),
            'n_motifs': len(clusters_motifs_df.columns),
            'mean_motif_score': clusters_motifs_df.values.mean(),
            'std_motif_score': clusters_motifs_df.values.std(),
            'sparsity': (clusters_motifs_df == 0).sum().sum() / clusters_motifs_df.size
        },
        'genes': {
            'n_clusters': len(clusters_genes_df),
            'n_genes': len(clusters_genes_df.columns),
            'mean_gene_score': clusters_genes_df.values.mean(),
            'std_gene_score': clusters_genes_df.values.std(),
            'sparsity': (clusters_genes_df == 0).sum().sum() / clusters_genes_df.size
        }
    }
    
    return stats

def normalize_cluster_data(
    clusters_df: pd.DataFrame,
    method: str = "zscore"
) -> pd.DataFrame:
    """
    Normalize cluster-level data.
    
    Parameters:
    -----------
    clusters_df : pd.DataFrame
        Cluster-level data to normalize
    method : str
        Normalization method: "zscore", "minmax", or "none"
        
    Returns:
    --------
    pd.DataFrame
        Normalized data
    """
    if method == "zscore":
        return (clusters_df - clusters_df.mean()) / clusters_df.std()
    elif method == "minmax":
        return (clusters_df - clusters_df.min()) / (clusters_df.max() - clusters_df.min())
    elif method == "none":
        return clusters_df.copy()
    else:
        raise ValueError(f"Unknown normalization method: {method}")