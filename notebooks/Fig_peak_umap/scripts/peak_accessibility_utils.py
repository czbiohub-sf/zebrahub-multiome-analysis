# Collection of functions for analyzing peak accessibility patterns across celltypes and timepoints
import pandas as pd
import numpy as np
import scanpy as sc


def create_cluster_celltype_profiles(adata, cluster_col='leiden_coarse', min_cells=0, verbose=True):
    """
    For each cluster, compute the mean accessibility across celltypes.
    
    Parameters
    ----------
    adata : AnnData
        peaks-by-pseudobulk matrix with clustering in .obs[cluster_col]
        and celltype/timepoint info in .var
    cluster_col : str
        Column name for clusters
    min_cells : int
        Minimum number of peaks per cluster (usually set to 0 for peaks)
    
    Returns
    -------
    cluster_celltype_profiles : dict
        {cluster_id: {celltype: mean_accessibility}}
    """
    if verbose:
        print(f"\n=== Creating celltype profiles for {cluster_col} ===")
    
    # Get normalized counts
    X = adata.layers["normalized"]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    cluster_profiles = {}
    clusters = adata.obs[cluster_col].unique()
    
    for cluster_id in sorted(clusters):
        # Get peaks in this cluster
        cluster_mask = adata.obs[cluster_col] == cluster_id
        n_peaks = cluster_mask.sum()
        
        if n_peaks < min_cells:
            if verbose:
                print(f"Skipping cluster {cluster_id}: only {n_peaks} peaks")
            continue
        
        # Get accessibility matrix for this cluster
        cluster_X = X[cluster_mask, :]
        
        # Average across celltypes
        celltype_profiles = {}
        for celltype in adata.var['celltype'].unique():
            celltype_mask = adata.var['celltype'] == celltype
            if celltype_mask.sum() > 0:
                # Mean across pseudobulk groups of this celltype
                celltype_accessibility = cluster_X[:, celltype_mask].mean()
                celltype_profiles[celltype] = celltype_accessibility
        
        cluster_profiles[cluster_id] = celltype_profiles
        
        if verbose and len(cluster_profiles) % 5 == 0:
            print(f"Processed {len(cluster_profiles)} clusters...")
    
    if verbose:
        print(f"\nCompleted! Created profiles for {len(cluster_profiles)} clusters")
    
    return cluster_profiles


def create_cluster_timepoint_profiles(adata, cluster_col='leiden_coarse', min_cells=0, verbose=True):
    """
    For each cluster, compute the mean accessibility across timepoints.
    
    Parameters
    ----------
    adata : AnnData
        peaks-by-pseudobulk matrix with clustering in .obs[cluster_col]
        and celltype/timepoint info in .var
    cluster_col : str
        Column name for clusters
    min_cells : int
        Minimum number of peaks per cluster (usually set to 0 for peaks)
    
    Returns
    -------
    cluster_timepoint_profiles : dict
        {cluster_id: {timepoint: mean_accessibility}}
    """
    if verbose:
        print(f"\n=== Creating timepoint profiles for {cluster_col} ===")
    
    # Get normalized counts
    X = adata.layers["normalized"]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    cluster_profiles = {}
    clusters = adata.obs[cluster_col].unique()
    
    for cluster_id in sorted(clusters):
        # Get peaks in this cluster
        cluster_mask = adata.obs[cluster_col] == cluster_id
        n_peaks = cluster_mask.sum()
        
        if n_peaks < min_cells:
            if verbose:
                print(f"Skipping cluster {cluster_id}: only {n_peaks} peaks")
            continue
        
        # Get accessibility matrix for this cluster
        cluster_X = X[cluster_mask, :]
        
        # Average across timepoints
        timepoint_profiles = {}
        for timepoint in adata.var['timepoint'].unique():
            timepoint_mask = adata.var['timepoint'] == timepoint
            if timepoint_mask.sum() > 0:
                # Mean across pseudobulk groups of this timepoint
                timepoint_accessibility = cluster_X[:, timepoint_mask].mean()
                timepoint_profiles[timepoint] = timepoint_accessibility
        
        cluster_profiles[cluster_id] = timepoint_profiles
        
        if verbose and len(cluster_profiles) % 5 == 0:
            print(f"Processed {len(cluster_profiles)} clusters...")
    
    if verbose:
        print(f"\nCompleted! Created profiles for {len(cluster_profiles)} clusters")
    
    return cluster_profiles


def get_top_annotations(cluster_profiles, top_n=3):
    """
    For each cluster, identify the top N most accessible celltypes/timepoints.
    
    Parameters
    ----------
    cluster_profiles : dict
        Dictionary of {cluster_id: {annotation: accessibility_value}}
    top_n : int
        Number of top annotations to return per cluster
    
    Returns
    -------
    top_annotations : dict
        {cluster_id: [(annotation, accessibility_value), ...]}
    """
    top_annotations = {}
    
    for cluster_id, profile in cluster_profiles.items():
        # Sort by accessibility value
        sorted_items = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        top_annotations[cluster_id] = sorted_items[:top_n]
    
    return top_annotations

