# Peak Cluster to GRN Pipeline - Differential Analysis Module
# Author: YangJoon Kim
# Date: 2025-06-25
# Description: Identify differentially enriched motifs per cluster

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import hypergeom, mannwhitneyu, false_discovery_control
import warnings

def compute_differential_motifs(
    clusters_motifs_df: pd.DataFrame,
    method: str = "top_n",
    top_n: int = 10,
    fold_change_threshold: float = 2.0,
    pvalue_threshold: float = 0.001,
    fdr_correction: bool = True
) -> Dict[str, List[str]]:
    """
    Identify differentially enriched motifs for each cluster.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    method : str
        "top_n", "threshold", or "statistical"
    top_n : int
        Number of top motifs to keep per cluster
    fold_change_threshold : float
        Minimum fold change for motif enrichment
    pvalue_threshold : float
        P-value threshold for statistical testing
    fdr_correction : bool
        Whether to apply FDR correction for multiple testing
        
    Returns:
    --------
    cluster_differential_motifs : Dict[str, List[str]]
        {cluster_id: [list_of_differential_motifs]}
    """
    if method == "top_n":
        return top_n_motifs(clusters_motifs_df, top_n)
    elif method == "threshold":
        return threshold_based_motifs(clusters_motifs_df, fold_change_threshold)
    elif method == "statistical":
        return statistical_differential_motifs(
            clusters_motifs_df, pvalue_threshold, fdr_correction
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'top_n', 'threshold', or 'statistical'")

def top_n_motifs(
    clusters_motifs_df: pd.DataFrame, 
    top_n: int = 10
) -> Dict[str, List[str]]:
    """
    Select top N motifs for each cluster based on highest values.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    top_n : int
        Number of top motifs to select per cluster
        
    Returns:
    --------
    Dict[str, List[str]]
        {cluster_id: [list_of_top_motifs]}
    """
    cluster_motifs = {}
    
    for cluster in clusters_motifs_df.index:
        cluster_data = clusters_motifs_df.loc[cluster]
        top_motifs = cluster_data.nlargest(top_n).index.tolist()
        cluster_motifs[str(cluster)] = top_motifs
    
    return cluster_motifs

def threshold_based_motifs(
    clusters_motifs_df: pd.DataFrame, 
    fold_change_threshold: float = 2.0
) -> Dict[str, List[str]]:
    """
    Use fold-change thresholding for differential motifs.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    fold_change_threshold : float
        Minimum fold change for motif enrichment
        
    Returns:
    --------
    Dict[str, List[str]]
        {cluster_id: [list_of_differential_motifs]}
    """
    cluster_motifs = {}
    
    # Compute mean expression across all clusters for each motif
    global_mean = clusters_motifs_df.mean(axis=0)
    
    for cluster in clusters_motifs_df.index:
        cluster_data = clusters_motifs_df.loc[cluster]
        
        # Calculate fold change relative to global mean
        # Add small pseudocount to avoid division by zero
        pseudocount = 1e-6
        fold_change = (cluster_data + pseudocount) / (global_mean + pseudocount)
        
        # Select motifs above threshold
        enriched_motifs = fold_change[fold_change >= fold_change_threshold].index.tolist()
        cluster_motifs[str(cluster)] = enriched_motifs
    
    return cluster_motifs

def statistical_differential_motifs(
    clusters_motifs_df: pd.DataFrame,
    pvalue_threshold: float = 0.001,
    fdr_correction: bool = True
) -> Dict[str, List[str]]:
    """
    Use statistical testing (Mann-Whitney U test) for differential motifs.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    pvalue_threshold : float
        P-value threshold for significance
    fdr_correction : bool
        Whether to apply FDR correction
        
    Returns:
    --------
    Dict[str, List[str]]
        {cluster_id: [list_of_differential_motifs]}
    """
    cluster_motifs = {}
    
    for cluster in clusters_motifs_df.index:
        cluster_data = clusters_motifs_df.loc[cluster]
        other_clusters_data = clusters_motifs_df.drop(cluster)
        
        pvalues = []
        motif_names = []
        
        for motif in clusters_motifs_df.columns:
            # Test if motif is significantly higher in this cluster vs others
            cluster_values = [cluster_data[motif]]  # Single value per cluster
            other_values = other_clusters_data[motif].values
            
            # Skip if not enough data points
            if len(other_values) < 2:
                pvalues.append(1.0)
                motif_names.append(motif)
                continue
            
            try:
                # Use Mann-Whitney U test (one-sided, testing if cluster > others)
                # Since we have single values, we'll compare cluster value to others
                if cluster_data[motif] > np.median(other_values):
                    # Simple z-score based test for single value
                    z_score = (cluster_data[motif] - np.mean(other_values)) / (np.std(other_values) + 1e-6)
                    # Convert z-score to approximate p-value (one-sided)
                    from scipy.stats import norm
                    pval = 1 - norm.cdf(z_score)
                else:
                    pval = 1.0
                    
                pvalues.append(pval)
                motif_names.append(motif)
                
            except Exception as e:
                warnings.warn(f"Statistical test failed for motif {motif}: {str(e)}")
                pvalues.append(1.0)
                motif_names.append(motif)
        
        # Apply FDR correction if requested
        if fdr_correction and len(pvalues) > 0:
            try:
                corrected_pvalues = false_discovery_control(pvalues, method='bh')
                significant_motifs = [motif for motif, pval in zip(motif_names, corrected_pvalues) 
                                    if pval < pvalue_threshold]
            except:
                # Fallback to uncorrected p-values
                significant_motifs = [motif for motif, pval in zip(motif_names, pvalues) 
                                    if pval < pvalue_threshold]
        else:
            significant_motifs = [motif for motif, pval in zip(motif_names, pvalues) 
                                if pval < pvalue_threshold]
        
        cluster_motifs[str(cluster)] = significant_motifs
    
    return cluster_motifs

def compute_motif_enrichment_scores(
    clusters_motifs_df: pd.DataFrame,
    method: str = "fold_change"
) -> pd.DataFrame:
    """
    Compute enrichment scores for all motifs across clusters.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    method : str
        Method for computing enrichment: "fold_change", "zscore", or "rank"
        
    Returns:
    --------
    pd.DataFrame
        Enrichment scores matrix (same shape as input)
    """
    if method == "fold_change":
        global_mean = clusters_motifs_df.mean(axis=0)
        pseudocount = 1e-6
        enrichment_scores = (clusters_motifs_df + pseudocount) / (global_mean + pseudocount)
        
    elif method == "zscore":
        # Z-score normalization across clusters for each motif
        enrichment_scores = clusters_motifs_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        
    elif method == "rank":
        # Rank-based enrichment (higher rank = more enriched)
        enrichment_scores = clusters_motifs_df.rank(axis=0, ascending=False)
        
    else:
        raise ValueError(f"Unknown enrichment method: {method}")
    
    return enrichment_scores

def filter_motifs_by_variance(
    clusters_motifs_df: pd.DataFrame,
    min_variance: float = 0.01
) -> pd.DataFrame:
    """
    Filter out motifs with low variance across clusters.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    min_variance : float
        Minimum variance threshold
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    motif_variances = clusters_motifs_df.var(axis=0)
    high_var_motifs = motif_variances[motif_variances >= min_variance].index
    
    print(f"Filtered {len(clusters_motifs_df.columns)} motifs to {len(high_var_motifs)} high-variance motifs")
    
    return clusters_motifs_df[high_var_motifs]

def get_motif_cluster_specificity(
    clusters_motifs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute specificity scores for each motif-cluster pair.
    Specificity = (cluster_value) / (sum of all cluster values)
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
        
    Returns:
    --------
    pd.DataFrame
        Specificity scores matrix
    """
    # Compute specificity: each cell / sum of column
    specificity_scores = clusters_motifs_df.div(clusters_motifs_df.sum(axis=0), axis=1)
    
    # Handle division by zero
    specificity_scores = specificity_scores.fillna(0)
    
    return specificity_scores

def summarize_differential_results(
    cluster_differential_motifs: Dict[str, List[str]],
    clusters_motifs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a summary table of differential motif results.
    
    Parameters:
    -----------
    cluster_differential_motifs : Dict[str, List[str]]
        Results from differential analysis
    clusters_motifs_df : pd.DataFrame
        Original clusters x motifs matrix
        
    Returns:
    --------
    pd.DataFrame
        Summary table with cluster, motif, and enrichment information
    """
    summary_data = []
    
    for cluster, motifs in cluster_differential_motifs.items():
        for motif in motifs:
            if motif in clusters_motifs_df.columns:
                motif_value = clusters_motifs_df.loc[cluster, motif]
                global_mean = clusters_motifs_df[motif].mean()
                fold_change = motif_value / (global_mean + 1e-6)
                
                summary_data.append({
                    'cluster': cluster,
                    'motif': motif,
                    'cluster_value': motif_value,
                    'global_mean': global_mean,
                    'fold_change': fold_change
                })
    
    return pd.DataFrame(summary_data)

def plot_motif_heatmap(
    clusters_motifs_df: pd.DataFrame,
    selected_motifs: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot heatmap of motif enrichment across clusters.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    selected_motifs : List[str], optional
        Specific motifs to plot. If None, plots all motifs
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if selected_motifs is not None:
        plot_data = clusters_motifs_df[selected_motifs]
    else:
        plot_data = clusters_motifs_df
    
    plt.figure(figsize=(max(8, len(plot_data.columns) * 0.3), 
                       max(6, len(plot_data.index) * 0.3)))
    
    sns.heatmap(plot_data, 
                cmap='viridis', 
                cbar=True,
                xticklabels=True, 
                yticklabels=True)
    
    plt.title('Motif Enrichment Across Clusters')
    plt.ylabel('Clusters')
    plt.xlabel('Motifs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()