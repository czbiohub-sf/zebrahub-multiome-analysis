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


def compute_accessibility_entropy(values, normalize=True):
    """
    Compute Shannon entropy of accessibility values.
    
    High entropy = more uniform distribution (ubiquitous)
    Low entropy = concentrated in few celltypes (specific)
    
    Parameters
    ----------
    values : array-like
        Accessibility values across celltypes/timepoints
    normalize : bool
        If True, normalize entropy to [0, 1] range
    
    Returns
    -------
    entropy : float
        Shannon entropy (normalized if normalize=True)
    """
    from scipy.stats import entropy as shannon_entropy
    
    values = np.array(values)
    
    # Avoid log(0) by adding small constant
    values = values + 1e-10
    
    # Normalize to sum to 1 (probability distribution)
    p = values / values.sum()
    
    # Calculate Shannon entropy
    H = shannon_entropy(p, base=2)
    
    if normalize:
        # Normalize by maximum possible entropy (log2(n))
        max_entropy = np.log2(len(values))
        if max_entropy > 0:
            H = H / max_entropy
    
    return H


def classify_cluster_specificity(
    cluster_celltype_profiles,
    entropy_threshold=0.75,
    dominance_threshold=0.5,
    n_specific_min=2,
    n_specific_max=5,
    accessibility_threshold_percentile=50,
    verbose=True
):
    """
    Classify each cluster as 'ubiquitous', 'specific', or 'moderate'.
    
    Parameters
    ----------
    cluster_celltype_profiles : dict
        {cluster_id: {celltype: accessibility_value}}
    entropy_threshold : float
        Normalized entropy above which cluster is considered ubiquitous (default 0.75)
    dominance_threshold : float
        Fraction of total accessibility in top celltype for specific clusters (default 0.5)
    n_specific_min : int
        Minimum number of highly accessible celltypes for specific pattern (default 2)
    n_specific_max : int
        Maximum number of highly accessible celltypes for specific pattern (default 5)
    accessibility_threshold_percentile : float
        Percentile threshold for determining "high" accessibility (default 50)
    verbose : bool
        Print classification summary
    
    Returns
    -------
    classifications : dict
        {cluster_id: {
            'pattern': 'ubiquitous'|'specific'|'moderate',
            'entropy': float,
            'n_high_accessible': int,
            'top_celltype_fraction': float,
            'top_celltypes': list
        }}
    """
    classifications = {}
    
    for cluster_id, profile in cluster_celltype_profiles.items():
        celltypes = list(profile.keys())
        values = np.array(list(profile.values()))
        
        # Calculate metrics
        entropy = compute_accessibility_entropy(values, normalize=True)
        
        # Determine threshold for "high" accessibility
        threshold = np.percentile(values, accessibility_threshold_percentile)
        n_high_accessible = (values >= threshold).sum()
        
        # Calculate dominance (fraction in top celltype)
        total_accessibility = values.sum()
        top_celltype_fraction = values.max() / total_accessibility if total_accessibility > 0 else 0
        
        # Get top celltypes
        sorted_indices = np.argsort(values)[::-1]
        top_celltypes = [celltypes[i] for i in sorted_indices[:5]]
        
        # Classification logic
        if entropy >= entropy_threshold:
            pattern = 'ubiquitous'
        elif (n_specific_min <= n_high_accessible <= n_specific_max) or \
             (top_celltype_fraction >= dominance_threshold):
            pattern = 'specific'
        else:
            pattern = 'moderate'
        
        classifications[cluster_id] = {
            'pattern': pattern,
            'entropy': entropy,
            'n_high_accessible': n_high_accessible,
            'top_celltype_fraction': top_celltype_fraction,
            'top_celltypes': top_celltypes
        }
    
    if verbose:
        print("\n=== Cluster Specificity Classification ===")
        
        # Count patterns
        pattern_counts = {}
        for cluster_id, info in classifications.items():
            pattern = info['pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        print(f"\nTotal clusters: {len(classifications)}")
        for pattern, count in sorted(pattern_counts.items()):
            print(f"  {pattern.capitalize()}: {count} clusters ({count/len(classifications)*100:.1f}%)")
        
        # Show examples
        print("\n--- Example Classifications ---")
        for pattern_type in ['ubiquitous', 'specific', 'moderate']:
            examples = [cid for cid, info in classifications.items() 
                       if info['pattern'] == pattern_type][:3]
            
            if examples:
                print(f"\n{pattern_type.upper()} clusters (examples):")
                for cluster_id in examples:
                    info = classifications[cluster_id]
                    print(f"  Cluster {cluster_id}:")
                    print(f"    Entropy: {info['entropy']:.3f}")
                    print(f"    N high accessible: {info['n_high_accessible']}")
                    print(f"    Top celltype fraction: {info['top_celltype_fraction']:.3f}")
                    print(f"    Top celltypes: {', '.join(info['top_celltypes'][:3])}")
    
    return classifications


def add_specificity_to_adata(adata, classifications, cluster_col='leiden_coarse'):
    """
    Add specificity classifications to AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        peaks-by-pseudobulk matrix
    classifications : dict
        Output from classify_cluster_specificity
    cluster_col : str
        Column name for clusters in adata.obs
    
    Returns
    -------
    None (modifies adata in place)
    """
    # Create mapping from cluster to pattern
    pattern_map = {cid: info['pattern'] for cid, info in classifications.items()}
    entropy_map = {cid: info['entropy'] for cid, info in classifications.items()}
    
    # Add to obs
    adata.obs['accessibility_pattern'] = adata.obs[cluster_col].map(pattern_map)
    adata.obs['accessibility_entropy'] = adata.obs[cluster_col].map(entropy_map)
    
    print(f"\nAdded 'accessibility_pattern' and 'accessibility_entropy' to adata.obs")
    print(f"Pattern distribution:")
    print(adata.obs['accessibility_pattern'].value_counts())

