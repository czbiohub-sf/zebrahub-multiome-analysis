"""
Neighborhood Purity Analysis Module

This module computes neighborhood purity scores for different embedding modalities
(RNA, ATAC, weighted nearest neighbors) to evaluate embedding quality based on
how well cells of the same metadata category cluster together in the embedding space.

For each cell, the purity score measures the proportion of k-nearest neighbors
that share the same metadata label (e.g., cell type, leiden cluster).

Author: Zebrahub-Multiome Analysis Pipeline
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Tuple, Dict, Optional
import warnings

# Try to import scIB metrics - if not available, provide fallback implementations
try:
    import scib
    SCIB_AVAILABLE = True
    print("✓ scIB package available for comprehensive integration metrics")
except ImportError:
    SCIB_AVAILABLE = False
    warnings.warn("scIB package not available. Using fallback implementations for integration metrics.")


def compute_knn_purity_from_connectivities(
    adata: sc.AnnData,
    connectivity_key: str,
    metadata_key: str,
    k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbor purity scores using pre-computed connectivities.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object containing connectivities and metadata
    connectivity_key : str
        Key for the connectivity matrix in adata.obsp
        (e.g., 'connectivities_RNA', 'connectivities_ATAC', 'connectivities_wnn')
    metadata_key : str
        Key in adata.obs for the metadata category to evaluate purity
        (e.g., 'celltype', 'leiden', 'annotation_ML_coarse')
    k : int, optional
        Number of nearest neighbors to consider. If None, uses all non-zero
        connections from the connectivity matrix
        
    Returns:
    --------
    purity_scores : np.ndarray
        Purity score for each cell (proportion of k-NN with same metadata)
    neighbor_indices : np.ndarray
        Indices of k-nearest neighbors for each cell
    """
    
    # Get connectivity matrix
    if connectivity_key not in adata.obsp.keys():
        raise ValueError(f"Connectivity matrix {connectivity_key} not found in adata.obsp")
    
    connectivities = adata.obsp[connectivity_key]
    
    # Get metadata labels
    if metadata_key not in adata.obs.columns:
        raise ValueError(f"Metadata {metadata_key} not found in adata.obs")
    
    labels = adata.obs[metadata_key].values
    n_cells = len(labels)
    
    # Convert to CSR format for efficient row access
    if not sparse.isspmatrix_csr(connectivities):
        connectivities = connectivities.tocsr()
    
    purity_scores = np.zeros(n_cells)
    neighbor_indices = []
    
    for i in range(n_cells):
        # Get non-zero connections (neighbors)
        neighbors = connectivities[i].nonzero()[1]
        
        # Remove self-connection if present
        neighbors = neighbors[neighbors != i]
        
        if len(neighbors) == 0:
            purity_scores[i] = 0.0
            neighbor_indices.append(np.array([]))
            continue
        
        # If k is specified, take top k neighbors based on connectivity weights
        if k is not None and len(neighbors) > k:
            # Get connectivity weights for these neighbors
            weights = connectivities[i, neighbors].toarray().flatten()
            # Sort by weight (descending) and take top k
            top_k_idx = np.argsort(weights)[::-1][:k]
            neighbors = neighbors[top_k_idx]
        
        neighbor_indices.append(neighbors)
        
        # Calculate purity
        cell_label = labels[i]
        neighbor_labels = labels[neighbors]
        purity_scores[i] = np.sum(neighbor_labels == cell_label) / len(neighbors)
    
    return purity_scores, neighbor_indices


def compute_knn_purity_single_modality(
    adata: sc.AnnData,
    modality_key: str,
    metadata_key: str,
    k: int = 30,
    metric: str = 'euclidean',
    use_rep: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbor purity scores for a single modality.
    
    DEPRECATED: Use compute_knn_purity_from_connectivities() instead for better performance.
    
    This function computes neighbors on-the-fly from embeddings.
    For better performance, use pre-computed connectivities with 
    compute_knn_purity_from_connectivities().
    """
    warnings.warn(
        "This function is deprecated. Use compute_knn_purity_from_connectivities() "
        "with pre-computed connectivities for better performance.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Get embedding
    if modality_key in adata.obsm.keys():
        embedding = adata.obsm[modality_key]
    elif use_rep and use_rep in adata.obsm.keys():
        embedding = adata.obsm[use_rep]
    else:
        raise ValueError(f"Embedding {modality_key} not found in adata.obsm")
    
    # Get metadata labels
    if metadata_key not in adata.obs.columns:
        raise ValueError(f"Metadata {metadata_key} not found in adata.obs")
    
    labels = adata.obs[metadata_key].values
    
    # Handle sparse matrices
    if sparse.issparse(embedding):
        embedding = embedding.toarray()
    
    # Compute k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nbrs.fit(embedding)
    
    # Get neighbor indices (exclude self - first neighbor)
    distances, neighbor_indices = nbrs.kneighbors(embedding)
    neighbor_indices = neighbor_indices[:, 1:]  # Remove self
    
    # Compute purity scores
    n_cells = len(labels)
    purity_scores = np.zeros(n_cells)
    
    for i in range(n_cells):
        cell_label = labels[i]
        neighbor_labels = labels[neighbor_indices[i]]
        
        # Calculate proportion of neighbors with same label
        purity_scores[i] = np.sum(neighbor_labels == cell_label) / k
    
    return purity_scores, neighbor_indices


def compute_multimodal_knn_purity(
    adata: sc.AnnData,
    connectivity_keys: Dict[str, str],
    metadata_key: str,
    k: Optional[int] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute k-nearest neighbor purity scores for multiple modalities using connectivities.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object
    connectivity_keys : Dict[str, str]
        Dictionary mapping modality names to connectivity matrix keys
        e.g., {'RNA': 'connectivities_RNA', 'ATAC': 'connectivities_ATAC', 'WNN': 'connectivities_wnn'}
    metadata_key : str
        Metadata category to evaluate purity against
    k : int, optional
        Number of nearest neighbors to consider. If None, uses all neighbors from connectivities
        
    Returns:
    --------
    results : Dict
        Dictionary with modality names as keys and (purity_scores, neighbor_indices) tuples
    """
    
    results = {}
    
    for modality_name, connectivity_key in connectivity_keys.items():
        try:
            purity_scores, neighbor_indices = compute_knn_purity_from_connectivities(
                adata, connectivity_key, metadata_key, k=k
            )
            results[modality_name] = (purity_scores, neighbor_indices)
            print(f"✓ Computed purity for {modality_name} modality using {connectivity_key}")
            
        except Exception as e:
            print(f"✗ Failed to compute purity for {modality_name}: {str(e)}")
            results[modality_name] = (None, None)
    
    return results


def compute_multimodal_knn_purity_legacy(
    adata: sc.AnnData,
    modality_keys: Dict[str, str],
    metadata_key: str,
    k: int = 30,
    metric: str = 'euclidean'
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    DEPRECATED: Compute k-nearest neighbor purity scores for multiple modalities from embeddings.
    
    Use compute_multimodal_knn_purity() with connectivity matrices instead for better performance.
    """
    warnings.warn(
        "This function is deprecated. Use compute_multimodal_knn_purity() "
        "with connectivity matrices for better performance.",
        DeprecationWarning,
        stacklevel=2
    )
    
    results = {}
    
    for modality_name, embedding_key in modality_keys.items():
        try:
            purity_scores, neighbor_indices = compute_knn_purity_single_modality(
                adata, embedding_key, metadata_key, k=k, metric=metric
            )
            results[modality_name] = (purity_scores, neighbor_indices)
            print(f"✓ Computed purity for {modality_name} modality")
            
        except Exception as e:
            print(f"✗ Failed to compute purity for {modality_name}: {str(e)}")
            results[modality_name] = (None, None)
    
    return results


def summarize_purity_scores(
    purity_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    adata: sc.AnnData,
    metadata_key: str
) -> pd.DataFrame:
    """
    Summarize purity scores across modalities and metadata categories.
    
    Parameters:
    -----------
    purity_results : Dict
        Results from compute_multimodal_knn_purity
    adata : sc.AnnData
        Annotated data object
    metadata_key : str
        Metadata category used for purity calculation
        
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary statistics for purity scores
    """
    
    summary_data = []
    
    for modality_name, (purity_scores, _) in purity_results.items():
        if purity_scores is None:
            continue
            
        # Overall statistics
        summary_data.append({
            'Modality': modality_name,
            'Metadata': 'Overall',
            'Mean_Purity': np.mean(purity_scores),
            'Median_Purity': np.median(purity_scores),
            'Std_Purity': np.std(purity_scores),
            'Min_Purity': np.min(purity_scores),
            'Max_Purity': np.max(purity_scores),
            'N_cells': len(purity_scores)
        })
        
        # Per-category statistics
        categories = adata.obs[metadata_key].unique()
        for category in categories:
            mask = adata.obs[metadata_key] == category
            cat_purity = purity_scores[mask]
            
            if len(cat_purity) > 0:
                summary_data.append({
                    'Modality': modality_name,
                    'Metadata': str(category),
                    'Mean_Purity': np.mean(cat_purity),
                    'Median_Purity': np.median(cat_purity),
                    'Std_Purity': np.std(cat_purity),
                    'Min_Purity': np.min(cat_purity),
                    'Max_Purity': np.max(cat_purity),
                    'N_cells': len(cat_purity)
                })
    
    return pd.DataFrame(summary_data)


def plot_purity_comparison(
    purity_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    adata: sc.AnnData,
    metadata_key: str,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison plots for neighborhood purity across modalities.
    
    Parameters:
    -----------
    purity_results : Dict
        Results from compute_multimodal_knn_purity
    adata : sc.AnnData
        Annotated data object
    metadata_key : str
        Metadata category used for purity calculation
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.Figure
        The created figure
    """
    
    # Prepare data for plotting
    plot_data = []
    for modality_name, (purity_scores, _) in purity_results.items():
        if purity_scores is None:
            continue
        
        for i, score in enumerate(purity_scores):
            plot_data.append({
                'Modality': modality_name,
                'Purity_Score': score,
                'Metadata': adata.obs[metadata_key].iloc[i]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Overall distribution comparison
    sns.boxplot(data=plot_df, x='Modality', y='Purity_Score', ax=axes[0])
    axes[0].set_title('Overall Purity Distribution')
    axes[0].set_ylabel('Neighborhood Purity Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Violin plot
    sns.violinplot(data=plot_df, x='Modality', y='Purity_Score', ax=axes[1])
    axes[1].set_title('Purity Score Distribution')
    axes[1].set_ylabel('Neighborhood Purity Score')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Mean purity by metadata category
    mean_purity = plot_df.groupby(['Modality', 'Metadata'])['Purity_Score'].mean().reset_index()
    mean_purity_pivot = mean_purity.pivot(index='Metadata', columns='Modality', values='Purity_Score')
    
    sns.heatmap(mean_purity_pivot, annot=True, cmap='viridis', ax=axes[2])
    axes[2].set_title(f'Mean Purity by {metadata_key}')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def add_purity_to_adata(
    adata: sc.AnnData,
    purity_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    metadata_key: str
) -> None:
    """
    Add purity scores to AnnData object for further analysis.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object to modify
    purity_results : Dict
        Results from compute_multimodal_knn_purity
    metadata_key : str
        Metadata category used for purity calculation
    """
    
    for modality_name, (purity_scores, _) in purity_results.items():
        if purity_scores is None:
            continue
            
        col_name = f'purity_{metadata_key}_{modality_name}'
        adata.obs[col_name] = purity_scores
        
    print(f"✓ Added purity scores to adata.obs for metadata: {metadata_key}")


# Example usage function
def example_neighborhood_purity_analysis():
    """
    Example function showing how to use the neighborhood purity analysis.
    This would typically be called from a Jupyter notebook.
    """
    
    # This is a template - actual usage would load real data
    print("Example usage of neighborhood purity analysis:")
    print("""
    # Load your AnnData object
    adata = sc.read_h5ad('your_multiome_data.h5ad')
    
    # Define connectivity matrices (preferred method)
    connectivity_keys = {
        'RNA': 'connectivities_RNA',      # RNA neighborhood graph
        'ATAC': 'connectivities_ATAC',    # ATAC neighborhood graph  
        'WNN': 'connectivities_wnn'       # Weighted nearest neighbor graph
    }
    
    # Compute purity scores using pre-computed connectivities
    purity_results = compute_multimodal_knn_purity(
        adata=adata,
        connectivity_keys=connectivity_keys,
        metadata_key='celltype',  # or 'leiden', 'annotation_ML_coarse', etc.
        k=30  # Optional: limit to top k neighbors, or None for all neighbors
    )
    
    # Alternative: compute from single connectivity matrix
    rna_purity, rna_neighbors = compute_knn_purity_from_connectivities(
        adata, 'connectivities_RNA', 'celltype', k=30
    )
    
    # Summarize results
    summary_df = summarize_purity_scores(purity_results, adata, 'celltype')
    print(summary_df)
    
    # Create comparison plots
    fig = plot_purity_comparison(purity_results, adata, 'celltype')
    plt.show()
    
    # Add to AnnData object
    add_purity_to_adata(adata, purity_results, 'celltype')
    
    # Check available connectivity matrices
    print("Available connectivity matrices:")
    for key in adata.obsp.keys():
        if 'connectivities' in key:
            print(f"  - {key}: {adata.obsp[key].shape}")
    """)


# Cross-Modality Validation Functions

def perform_modality_specific_clustering(
    adata: sc.AnnData,
    embedding_key: str,
    n_clusters: int = None,
    leiden_resolution: float = 0.5,
    method: str = 'leiden'
) -> np.ndarray:
    """
    Perform clustering on a specific modality embedding.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object
    embedding_key : str
        Key for embedding in adata.obsm (e.g., 'X_pca_rna', 'X_lsi_atac')
    n_clusters : int, optional
        Number of clusters for K-means. If None, uses leiden clustering
    leiden_resolution : float, default=0.5
        Resolution for leiden clustering
    method : str, default='leiden'
        Clustering method: 'leiden' or 'kmeans'
        
    Returns:
    --------
    cluster_labels : np.ndarray
        Cluster labels for each cell
    """
    
    if embedding_key not in adata.obsm.keys():
        raise ValueError(f"Embedding {embedding_key} not found in adata.obsm")
    
    embedding = adata.obsm[embedding_key]
    
    if sparse.issparse(embedding):
        embedding = embedding.toarray()
    
    if method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for K-means clustering")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)
        
    elif method == 'leiden':
        # Create temporary AnnData for leiden clustering
        temp_adata = sc.AnnData(X=embedding)
        temp_adata.obsm['X_embedding'] = embedding
        
        # Compute neighbors and perform leiden clustering
        sc.pp.neighbors(temp_adata, use_rep='X_embedding', n_neighbors=15)
        sc.tl.leiden(temp_adata, resolution=leiden_resolution)
        
        cluster_labels = temp_adata.obs['leiden'].astype(int).values
        
    else:
        raise ValueError("Method must be 'leiden' or 'kmeans'")
    
    return cluster_labels


def compute_cross_modality_preservation(
    adata: sc.AnnData,
    cluster_keys: Dict[str, str],
    reference_modality: str
) -> Dict[str, Dict[str, float]]:
    """
    Compute cross-modality cluster preservation using ARI and NMI with pre-computed clusters.
    
    This function:
    1. Uses existing cluster labels for the reference modality
    2. Measures how well these clusters are preserved in other modalities
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object
    cluster_keys : Dict[str, str]
        Dictionary mapping modality names to cluster label keys in adata.obs
        e.g., {'RNA': 'RNA_leiden_08', 'ATAC': 'ATAC_leiden_08', 'WNN': 'wsnn_res.0.8'}
    reference_modality : str
        Modality to use for reference clustering (key in cluster_keys)
        
    Returns:
    --------
    preservation_scores : Dict[str, Dict[str, float]]
        Nested dictionary with modality names and metrics (ARI, NMI)
    """
    
    if reference_modality not in cluster_keys:
        raise ValueError(f"Reference modality {reference_modality} not found in cluster_keys")
    
    # Get reference clustering labels
    reference_cluster_key = cluster_keys[reference_modality]
    if reference_cluster_key not in adata.obs.columns:
        raise ValueError(f"Reference cluster key {reference_cluster_key} not found in adata.obs")
    
    reference_clusters = adata.obs[reference_cluster_key].astype(str).values
    print(f"Using reference clusters from {reference_modality}: {reference_cluster_key}")
    print(f"Found {len(np.unique(reference_clusters))} unique clusters")
    
    preservation_scores = {}
    
    # Test preservation in other modalities
    for modality_name, target_cluster_key in cluster_keys.items():
        if modality_name == reference_modality:
            # Perfect preservation of self
            preservation_scores[modality_name] = {'ARI': 1.0, 'NMI': 1.0}
            continue
        
        print(f"Testing preservation in {modality_name} modality using {target_cluster_key}")
        
        try:
            # Get target clustering labels
            if target_cluster_key not in adata.obs.columns:
                print(f"✗ Target cluster key {target_cluster_key} not found in adata.obs")
                preservation_scores[modality_name] = {'ARI': np.nan, 'NMI': np.nan}
                continue
            
            target_clusters = adata.obs[target_cluster_key].astype(str).values
            print(f"Found {len(np.unique(target_clusters))} unique clusters in {modality_name}")
            
            # Compute preservation metrics
            ari = adjusted_rand_score(reference_clusters, target_clusters)
            nmi = normalized_mutual_info_score(reference_clusters, target_clusters)
            
            preservation_scores[modality_name] = {'ARI': ari, 'NMI': nmi}
            print(f"  ARI: {ari:.3f}, NMI: {nmi:.3f}")
            
        except Exception as e:
            print(f"✗ Failed to compute preservation for {modality_name}: {str(e)}")
            preservation_scores[modality_name] = {'ARI': np.nan, 'NMI': np.nan}
    
    return preservation_scores


def compute_bidirectional_cross_modality_validation(
    adata: sc.AnnData,
    cluster_keys: Dict[str, str]
) -> pd.DataFrame:
    """
    Perform bidirectional cross-modality validation using pre-computed clusters.
    
    Tests cluster preservation in both directions:
    - RNA → ATAC, WNN
    - ATAC → RNA, WNN  
    - WNN → RNA, ATAC
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object
    cluster_keys : Dict[str, str]
        Dictionary mapping modality names to cluster label keys in adata.obs
        e.g., {'RNA': 'RNA_leiden_08', 'ATAC': 'ATAC_leiden_08', 'WNN': 'wsnn_res.0.8'}
        
    Returns:
    --------
    validation_df : pd.DataFrame
        DataFrame with preservation scores for all modality pairs
    """
    
    all_results = []
    
    for reference_modality in cluster_keys.keys():
        print(f"\n=== Using {reference_modality} as reference modality ===")
        
        preservation_scores = compute_cross_modality_preservation(
            adata, cluster_keys, reference_modality
        )
        
        # Convert to DataFrame format
        for target_modality, scores in preservation_scores.items():
            all_results.append({
                'Reference_Modality': reference_modality,
                'Target_Modality': target_modality,
                'ARI': scores['ARI'],
                'NMI': scores['NMI'],
                'Is_Self': reference_modality == target_modality
            })
    
    validation_df = pd.DataFrame(all_results)
    return validation_df


def plot_cross_modality_validation(
    validation_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create visualization for cross-modality validation results.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        Results from compute_bidirectional_cross_modality_validation
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.Figure
        The created figure
    """
    
    # Remove self-comparisons for plotting
    plot_data = validation_df[~validation_df['Is_Self']].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ARI heatmap
    ari_pivot = plot_data.pivot(
        index='Reference_Modality', 
        columns='Target_Modality', 
        values='ARI'
    )
    
    sns.heatmap(
        ari_pivot, 
        annot=True, 
        cmap='viridis', 
        ax=axes[0],
        vmin=0, 
        vmax=1,
        fmt='.3f'
    )
    axes[0].set_title('Adjusted Rand Index (ARI)')
    axes[0].set_xlabel('Target Modality')
    axes[0].set_ylabel('Reference Modality')
    
    # NMI heatmap
    nmi_pivot = plot_data.pivot(
        index='Reference_Modality', 
        columns='Target_Modality', 
        values='NMI'
    )
    
    sns.heatmap(
        nmi_pivot, 
        annot=True, 
        cmap='viridis', 
        ax=axes[1],
        vmin=0, 
        vmax=1,
        fmt='.3f'
    )
    axes[1].set_title('Normalized Mutual Information (NMI)')
    axes[1].set_xlabel('Target Modality')
    axes[1].set_ylabel('Reference Modality')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def summarize_cross_modality_validation(
    validation_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize cross-modality validation results.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        Results from compute_bidirectional_cross_modality_validation
        
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary statistics for cross-modality preservation
    """
    
    # Exclude self-comparisons
    cross_modal_data = validation_df[~validation_df['Is_Self']].copy()
    
    summary_stats = []
    
    # Overall statistics
    summary_stats.extend([
        {
            'Category': 'Overall',
            'Reference': 'All',
            'Target': 'All',
            'Metric': 'ARI',
            'Mean': cross_modal_data['ARI'].mean(),
            'Std': cross_modal_data['ARI'].std(),
            'Min': cross_modal_data['ARI'].min(),
            'Max': cross_modal_data['ARI'].max()
        },
        {
            'Category': 'Overall',
            'Reference': 'All',
            'Target': 'All', 
            'Metric': 'NMI',
            'Mean': cross_modal_data['NMI'].mean(),
            'Std': cross_modal_data['NMI'].std(),
            'Min': cross_modal_data['NMI'].min(),
            'Max': cross_modal_data['NMI'].max()
        }
    ])
    
    # By reference modality
    for ref_mod in cross_modal_data['Reference_Modality'].unique():
        ref_data = cross_modal_data[cross_modal_data['Reference_Modality'] == ref_mod]
        
        summary_stats.extend([
            {
                'Category': 'By_Reference',
                'Reference': ref_mod,
                'Target': 'All',
                'Metric': 'ARI',
                'Mean': ref_data['ARI'].mean(),
                'Std': ref_data['ARI'].std(),
                'Min': ref_data['ARI'].min(),
                'Max': ref_data['ARI'].max()
            },
            {
                'Category': 'By_Reference',
                'Reference': ref_mod,
                'Target': 'All',
                'Metric': 'NMI',
                'Mean': ref_data['NMI'].mean(),
                'Std': ref_data['NMI'].std(),
                'Min': ref_data['NMI'].min(),
                'Max': ref_data['NMI'].max()
            }
        ])
    
    # By target modality
    for target_mod in cross_modal_data['Target_Modality'].unique():
        target_data = cross_modal_data[cross_modal_data['Target_Modality'] == target_mod]
        
        summary_stats.extend([
            {
                'Category': 'By_Target',
                'Reference': 'All',
                'Target': target_mod,
                'Metric': 'ARI', 
                'Mean': target_data['ARI'].mean(),
                'Std': target_data['ARI'].std(),
                'Min': target_data['ARI'].min(),
                'Max': target_data['ARI'].max()
            },
            {
                'Category': 'By_Target',
                'Reference': 'All',
                'Target': target_mod,
                'Metric': 'NMI',
                'Mean': target_data['NMI'].mean(),
                'Std': target_data['NMI'].std(),
                'Min': target_data['NMI'].min(),
                'Max': target_data['NMI'].max()
            }
        ])
    
    return pd.DataFrame(summary_stats)


# Convenience function for specific data structure
def analyze_zebrahub_multiome_neighborhoods(
    adata: sc.AnnData,
    metadata_key: str = 'global_annotation',
    k: Optional[int] = 30,
    figsize: Tuple[int, int] = (15, 8),
    save_plots: bool = False,
    output_dir: str = './figures/'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete neighborhood purity and cross-modality analysis for Zebrahub multiome data.
    
    This is a convenience function tailored to the specific data structure described.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Zebrahub multiome AnnData object
    metadata_key : str, default='global_annotation'
        Metadata category for purity analysis
    k : int, optional, default=30
        Number of neighbors for purity calculation
    figsize : Tuple[int, int]
        Figure size for plots
    save_plots : bool, default=False
        Whether to save plots to disk
    output_dir : str, default='./figures/'
        Directory for saved plots
        
    Returns:
    --------
    purity_summary : pd.DataFrame
        Neighborhood purity analysis summary
    validation_summary : pd.DataFrame
        Cross-modality validation summary
    all_results : Dict
        Dictionary containing all results and intermediate data
    """
    
    print("🔬 Starting Zebrahub Multiome Neighborhood Analysis")
    print(f"Data shape: {adata.shape}")
    print(f"Analyzing purity for: {metadata_key}")
    
    # Define connectivity matrices matching the data structure
    connectivity_keys = {
        'RNA': 'RNA_connectivities',
        'ATAC': 'ATAC_connectivities', 
        'WNN': 'connectivities_wnn'
    }
    
    # Define cluster labels matching the data structure
    cluster_keys = {
        'RNA': 'RNA_leiden_08',
        'ATAC': 'ATAC_leiden_08',
        'WNN': 'wsnn_res.0.8'
    }
    
    all_results = {}
    
    # 1. NEIGHBORHOOD PURITY ANALYSIS
    print("\n📊 Computing neighborhood purity scores...")
    purity_results = compute_multimodal_knn_purity(
        adata=adata,
        connectivity_keys=connectivity_keys,
        metadata_key=metadata_key,
        k=k
    )
    
    purity_summary = summarize_purity_scores(purity_results, adata, metadata_key)
    all_results['purity_results'] = purity_results
    all_results['purity_summary'] = purity_summary
    
    # Add purity scores to AnnData
    add_purity_to_adata(adata, purity_results, metadata_key)
    
    # Plot purity comparison
    fig_purity = plot_purity_comparison(
        purity_results, adata, metadata_key, figsize=figsize
    )
    if save_plots:
        fig_purity.savefig(f"{output_dir}neighborhood_purity_{metadata_key}.pdf")
        fig_purity.savefig(f"{output_dir}neighborhood_purity_{metadata_key}.png", dpi=300)
    
    all_results['purity_plot'] = fig_purity
    
    # 2. CROSS-MODALITY VALIDATION
    print("\n🔄 Computing cross-modality cluster preservation...")
    validation_df = compute_bidirectional_cross_modality_validation(
        adata=adata,
        cluster_keys=cluster_keys
    )
    
    validation_summary = summarize_cross_modality_validation(validation_df)
    all_results['validation_results'] = validation_df
    all_results['validation_summary'] = validation_summary
    
    # Plot cross-modality validation
    fig_cross = plot_cross_modality_validation(
        validation_df, figsize=(12, 5)
    )
    if save_plots:
        fig_cross.savefig(f"{output_dir}cross_modality_validation.pdf")
        fig_cross.savefig(f"{output_dir}cross_modality_validation.png", dpi=300)
    
    all_results['cross_validation_plot'] = fig_cross
    
    # 3. SUMMARY REPORT
    print("\n📋 Analysis Summary:")
    print("=" * 50)
    print("\n🎯 Neighborhood Purity (Higher = Better):")
    overall_purity = purity_summary[purity_summary['Metadata'] == 'Overall']
    for _, row in overall_purity.iterrows():
        print(f"  {row['Modality']}: {row['Mean_Purity']:.3f} ± {row['Std_Purity']:.3f}")
    
    print("\n🔄 Cross-Modality Validation (ARI/NMI, Higher = Better):")
    overall_cross = validation_summary[validation_summary['Category'] == 'Overall']
    for _, row in overall_cross.iterrows():
        print(f"  {row['Metric']}: {row['Mean']:.3f} ± {row['Std']:.3f}")
    
    print("\n✅ Analysis complete!")
    
    return purity_summary, validation_summary, all_results


# Updated example usage
def example_neighborhood_purity_analysis():
    """
    Example function showing how to use the neighborhood purity analysis.
    This would typically be called from a Jupyter notebook.
    """
    
    # This is a template - actual usage would load real data
    print("Example usage of neighborhood purity analysis:")
    print("""
    # Load your AnnData object
    adata = sc.read_h5ad('your_multiome_data.h5ad')
    
    # Define connectivity matrices (matching your data structure)
    connectivity_keys = {
        'RNA': 'RNA_connectivities',      # RNA neighborhood graph
        'ATAC': 'ATAC_connectivities',    # ATAC neighborhood graph  
        'WNN': 'connectivities_wnn'       # Weighted nearest neighbor graph
    }
    
    # 1. NEIGHBORHOOD PURITY ANALYSIS
    # Compute purity scores using pre-computed connectivities
    purity_results = compute_multimodal_knn_purity(
        adata=adata,
        connectivity_keys=connectivity_keys,
        metadata_key='global_annotation',  # or 'RNA_leiden_08', 'ATAC_leiden_08', etc.
        k=30  # Optional: limit to top k neighbors, or None for all neighbors
    )
    
    # Summarize and visualize purity results
    summary_df = summarize_purity_scores(purity_results, adata, 'global_annotation')
    fig = plot_purity_comparison(purity_results, adata, 'global_annotation')
    add_purity_to_adata(adata, purity_results, 'global_annotation')
    
    # 2. CROSS-MODALITY VALIDATION (using existing leiden clusters)
    # Define cluster label keys (matching your data structure)
    cluster_keys = {
        'RNA': 'RNA_leiden_08',      # RNA leiden clusters at resolution 0.8
        'ATAC': 'ATAC_leiden_08',    # ATAC leiden clusters at resolution 0.8
        'WNN': 'wsnn_res.0.8'        # WNN clusters at resolution 0.8
    }
    
    # Perform bidirectional cross-modality validation using existing clusters
    validation_df = compute_bidirectional_cross_modality_validation(
        adata=adata,
        cluster_keys=cluster_keys
    )
    
    # Summarize and visualize cross-modality results
    summary_cross_df = summarize_cross_modality_validation(validation_df)
    fig_cross = plot_cross_modality_validation(validation_df)
    
    print("Cross-modality validation summary:")
    print(summary_cross_df)
    
    # Check available connectivity matrices and embeddings
    print("\\nAvailable connectivity matrices:")
    for key in adata.obsp.keys():
        if 'connectivities' in key or 'connectivities' in key.lower():
            print(f"  - {key}: {adata.obsp[key].shape}")
            
    print("\\nAvailable embeddings:")
    for key in adata.obsm.keys():
        print(f"  - {key}: {adata.obsm[key].shape}")
        
    print("\\nAvailable cluster labels:")
    cluster_cols = ['RNA_leiden_08', 'ATAC_leiden_08', 'wsnn_res.0.8', 'global_annotation']
    for col in cluster_cols:
        if col in adata.obs.columns:
            n_unique = adata.obs[col].nunique()
            print(f"  - {col}: {n_unique} unique values")
    """)


# scIB-Based Integration Quality Assessment Functions

def compute_scIB_leiden_clusters_per_modality(
    adata: sc.AnnData,
    embedding_keys: Dict[str, str],
    resolution: float = 0.5,
    n_neighbors: int = 15
) -> Dict[str, np.ndarray]:
    """
    Compute leiden clusters independently for each modality using scanpy.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object
    embedding_keys : Dict[str, str]
        Dictionary mapping modality names to embedding keys
    resolution : float, default=0.5
        Leiden clustering resolution
    n_neighbors : int, default=15
        Number of neighbors for graph construction
        
    Returns:
    --------
    modality_clusters : Dict[str, np.ndarray]
        Dictionary mapping modality names to cluster labels
    """
    
    modality_clusters = {}
    
    for modality_name, embedding_key in embedding_keys.items():
        print(f"Computing leiden clusters for {modality_name} modality...")
        
        if embedding_key not in adata.obsm.keys():
            print(f"✗ Embedding {embedding_key} not found, skipping {modality_name}")
            continue
        
        # Create temporary AnnData for clustering this modality
        temp_adata = adata.copy()
        temp_adata.obsm['X_embedding'] = adata.obsm[embedding_key]
        
        # Compute neighbors and leiden clustering using scanpy
        sc.pp.neighbors(temp_adata, use_rep='X_embedding', n_neighbors=n_neighbors)
        sc.tl.leiden(temp_adata, resolution=resolution, key_added=f'leiden_{modality_name}')
        
        # Store cluster labels
        cluster_labels = temp_adata.obs[f'leiden_{modality_name}'].astype(int).values
        modality_clusters[modality_name] = cluster_labels
        
        print(f"✓ Found {len(np.unique(cluster_labels))} clusters in {modality_name}")
    
    return modality_clusters


def compute_scIB_cross_modality_metrics(
    adata: sc.AnnData,
    embedding_keys: Dict[str, str],
    modality_clusters: Dict[str, np.ndarray],
    use_scib: bool = True
) -> pd.DataFrame:
    """
    Compute comprehensive scIB-style cross-modality integration metrics.
    
    This function implements the key insight from scIB: use each modality as an
    independent validator of others to avoid circularity.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object
    embedding_keys : Dict[str, str]
        Dictionary mapping modality names to embedding keys
    modality_clusters : Dict[str, np.ndarray]
        Pre-computed clusters for each modality
    use_scib : bool, default=True
        Whether to use scIB package functions if available
        
    Returns:
    --------
    metrics_df : pd.DataFrame
        Comprehensive cross-modality validation metrics
    """
    
    results = []
    
    for ref_modality, ref_clusters in modality_clusters.items():
        ref_embedding = adata.obsm[embedding_keys[ref_modality]]
        
        for target_modality, target_clusters in modality_clusters.items():
            if ref_modality == target_modality:
                continue  # Skip self-comparison
            
            target_embedding = adata.obsm[embedding_keys[target_modality]]
            
            print(f"Computing metrics: {ref_modality} → {target_modality}")
            
            # Core metrics (always computed)
            ari = adjusted_rand_score(ref_clusters, target_clusters)
            nmi = normalized_mutual_info_score(ref_clusters, target_clusters)
            
            # Silhouette scores - how well reference clusters separate in target embedding
            if sparse.issparse(target_embedding):
                target_embedding_dense = target_embedding.toarray()
            else:
                target_embedding_dense = target_embedding
            
            try:
                asw_label = silhouette_score(target_embedding_dense, ref_clusters)
            except:
                asw_label = np.nan
            
            # Initialize result
            result = {
                'Reference_Modality': ref_modality,
                'Target_Modality': target_modality,
                'ARI_cluster': ari,
                'NMI_cluster': nmi,
                'ASW_label': asw_label,
            }
            
            # scIB-specific metrics if available
            if use_scib and SCIB_AVAILABLE:
                try:
                    # Create temporary AnnData for scIB metrics
                    temp_adata = sc.AnnData(X=target_embedding_dense)
                    temp_adata.obs['ref_clusters'] = ref_clusters.astype(str)
                    temp_adata.obs['target_clusters'] = target_clusters.astype(str)
                    temp_adata.obsm['X_emb'] = target_embedding_dense
                    
                    # Graph connectivity preservation
                    graph_conn = scib.me.graph_connectivity(temp_adata, label_key='ref_clusters')
                    result['Graph_connectivity'] = graph_conn
                    
                except Exception as e:
                    print(f"Warning: scIB graph connectivity failed for {ref_modality}→{target_modality}: {e}")
                    result['Graph_connectivity'] = np.nan
            else:
                result['Graph_connectivity'] = np.nan
            
            results.append(result)
    
    return pd.DataFrame(results)


def compute_scIB_integration_quality_comprehensive(
    adata: sc.AnnData,
    embedding_keys: Dict[str, str],
    leiden_resolution: float = 0.5,
    n_neighbors: int = 15,
    use_scib: bool = True
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Comprehensive scIB-based integration quality assessment.
    
    This function implements the complete scIB validation strategy:
    1. Independent clustering on each modality
    2. Cross-modality validation using multiple metrics
    3. Breaking circularity by using each modality as validator
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object
    embedding_keys : Dict[str, str]
        Dictionary mapping modality names to embedding keys
        e.g., {'RNA': 'X_pca_rna', 'ATAC': 'X_lsi_atac', 'WNN': 'X_pca_wnn'}
    leiden_resolution : float, default=0.5
        Resolution for leiden clustering
    n_neighbors : int, default=15
        Number of neighbors for graph construction
    use_scib : bool, default=True
        Whether to use scIB package functions if available
        
    Returns:
    --------
    metrics_df : pd.DataFrame
        Comprehensive cross-modality metrics
    modality_clusters : Dict[str, np.ndarray]
        Cluster assignments for each modality
    """
    
    print("=== scIB-Based Integration Quality Assessment ===")
    
    # Step 1: Independent clustering on each modality
    print("\n1. Computing independent leiden clusters for each modality...")
    modality_clusters = compute_scIB_leiden_clusters_per_modality(
        adata, embedding_keys, resolution=leiden_resolution, n_neighbors=n_neighbors
    )
    
    # Step 2: Cross-modality validation
    print("\n2. Computing cross-modality integration metrics...")
    metrics_df = compute_scIB_cross_modality_metrics(
        adata, embedding_keys, modality_clusters, use_scib=use_scib
    )
    
    print("✓ scIB integration quality assessment completed")
    
    return metrics_df, modality_clusters


def summarize_scIB_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize scIB cross-modality integration metrics.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        Results from compute_scIB_integration_quality_comprehensive
        
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary statistics for scIB integration metrics
    """
    
    summary_stats = []
    
    # Metrics to summarize
    metric_columns = ['ARI_cluster', 'NMI_cluster', 'ASW_label', 'Graph_connectivity']
    available_metrics = [col for col in metric_columns if col in metrics_df.columns]
    
    # Overall statistics
    for metric in available_metrics:
        if metrics_df[metric].notna().sum() > 0:  # Only if we have valid data
            summary_stats.append({
                'Category': 'Overall',
                'Reference': 'All',
                'Target': 'All',
                'Metric': metric,
                'Mean': metrics_df[metric].mean(),
                'Std': metrics_df[metric].std(),
                'Min': metrics_df[metric].min(),
                'Max': metrics_df[metric].max(),
                'N_valid': metrics_df[metric].notna().sum()
            })
    
    # By reference modality (how well each modality preserves others)
    for ref_mod in metrics_df['Reference_Modality'].unique():
        ref_data = metrics_df[metrics_df['Reference_Modality'] == ref_mod]
        
        for metric in available_metrics:
            if ref_data[metric].notna().sum() > 0:
                summary_stats.append({
                    'Category': 'By_Reference',
                    'Reference': ref_mod,
                    'Target': 'All',
                    'Metric': metric,
                    'Mean': ref_data[metric].mean(),
                    'Std': ref_data[metric].std(),
                    'Min': ref_data[metric].min(),
                    'Max': ref_data[metric].max(),
                    'N_valid': ref_data[metric].notna().sum()
                })
    
    # By target modality (how well each modality is preserved by others)
    for target_mod in metrics_df['Target_Modality'].unique():
        target_data = metrics_df[metrics_df['Target_Modality'] == target_mod]
        
        for metric in available_metrics:
            if target_data[metric].notna().sum() > 0:
                summary_stats.append({
                    'Category': 'By_Target',
                    'Reference': 'All',
                    'Target': target_mod,
                    'Metric': metric,
                    'Mean': target_data[metric].mean(),
                    'Std': target_data[metric].std(),
                    'Min': target_data[metric].min(),
                    'Max': target_data[metric].max(),
                    'N_valid': target_data[metric].notna().sum()
                })
    
    return pd.DataFrame(summary_stats)


def plot_scIB_integration_metrics(
    metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive visualization for scIB integration metrics.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        Results from compute_scIB_integration_quality_comprehensive
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.Figure
        The created figure
    """
    
    # Available metrics
    metric_columns = ['ARI_cluster', 'NMI_cluster', 'ASW_label', 'Graph_connectivity']
    available_metrics = [col for col in metric_columns if col in metrics_df.columns and metrics_df[col].notna().sum() > 0]
    
    n_metrics = len(available_metrics)
    if n_metrics == 0:
        print("No valid metrics to plot")
        return None
    
    # Create subplots
    fig, axes = plt.subplots(2, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = axes.reshape(2, 1)
    
    for i, metric in enumerate(available_metrics):
        # Top row: Heatmaps
        pivot_data = metrics_df.pivot(
            index='Reference_Modality',
            columns='Target_Modality', 
            values=metric
        )
        
        # Determine colormap range
        vmin = metrics_df[metric].min() if not np.isnan(metrics_df[metric].min()) else 0
        vmax = metrics_df[metric].max() if not np.isnan(metrics_df[metric].max()) else 1
        
        sns.heatmap(
            pivot_data,
            annot=True,
            cmap='viridis',
            ax=axes[0, i],
            vmin=vmin,
            vmax=vmax,
            fmt='.3f',
            cbar_kws={'shrink': 0.8}
        )
        axes[0, i].set_title(f'{metric} (Cross-Modality)')
        axes[0, i].set_xlabel('Target Modality')
        axes[0, i].set_ylabel('Reference Modality')
        
        # Bottom row: Bar plots showing preservation scores
        ref_means = metrics_df.groupby('Reference_Modality')[metric].mean()
        target_means = metrics_df.groupby('Target_Modality')[metric].mean()
        
        x_pos = np.arange(len(ref_means))
        width = 0.35
        
        axes[1, i].bar(x_pos - width/2, ref_means.values, width, 
                      label='As Reference', alpha=0.8)
        axes[1, i].bar(x_pos + width/2, target_means.values, width, 
                      label='As Target', alpha=0.8)
        
        axes[1, i].set_title(f'{metric} by Modality Role')
        axes[1, i].set_xlabel('Modality')
        axes[1, i].set_ylabel(f'Mean {metric}')
        axes[1, i].set_xticks(x_pos)
        axes[1, i].set_xticklabels(ref_means.index, rotation=45)
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def add_scIB_clusters_to_adata(
    adata: sc.AnnData,
    modality_clusters: Dict[str, np.ndarray],
    cluster_key_prefix: str = 'scIB_leiden'
) -> None:
    """
    Add scIB-computed cluster assignments to AnnData object.
    
    Parameters:
    -----------
    adata : sc.AnnData
        Annotated data object to modify
    modality_clusters : Dict[str, np.ndarray]
        Cluster assignments from compute_scIB_leiden_clusters_per_modality
    cluster_key_prefix : str, default='scIB_leiden'
        Prefix for cluster keys in adata.obs
    """
    
    for modality_name, clusters in modality_clusters.items():
        key_name = f'{cluster_key_prefix}_{modality_name}'
        adata.obs[key_name] = clusters.astype(str)
        print(f"✓ Added {key_name} to adata.obs")


# Updated example usage with scIB functions
def example_neighborhood_purity_analysis():
    """
    Example function showing how to use the neighborhood purity analysis.
    This would typically be called from a Jupyter notebook.
    """
    
    # This is a template - actual usage would load real data
    print("Example usage of neighborhood purity analysis:")
    print("""
    # Load your AnnData object
    adata = sc.read_h5ad('your_multiome_data.h5ad')
    
    # Define connectivity matrices (for neighborhood purity)
    connectivity_keys = {
        'RNA': 'connectivities_RNA',      # RNA neighborhood graph
        'ATAC': 'connectivities_ATAC',    # ATAC neighborhood graph  
        'WNN': 'connectivities_wnn'       # Weighted nearest neighbor graph
    }
    
    # Define embedding keys (for clustering and scIB metrics)
    embedding_keys = {
        'RNA': 'X_pca_rna',      # RNA PCA embedding
        'ATAC': 'X_lsi_atac',    # ATAC LSI embedding  
        'WNN': 'X_pca_wnn'       # Weighted nearest neighbor embedding
    }
    
    # 1. NEIGHBORHOOD PURITY ANALYSIS
    print("=== Neighborhood Purity Analysis ===")
    purity_results = compute_multimodal_knn_purity(
        adata=adata,
        connectivity_keys=connectivity_keys,
        metadata_key='celltype',
        k=30
    )
    
    summary_purity = summarize_purity_scores(purity_results, adata, 'celltype')
    fig_purity = plot_purity_comparison(purity_results, adata, 'celltype')
    add_purity_to_adata(adata, purity_results, 'celltype')
    
    # 2. STANDARD CROSS-MODALITY VALIDATION
    print("\\n=== Standard Cross-Modality Validation ===")
    validation_df = compute_bidirectional_cross_modality_validation(
        adata=adata,
        embedding_keys=embedding_keys,
        leiden_resolution=0.5
    )
    
    summary_cross = summarize_cross_modality_validation(validation_df)
    fig_cross = plot_cross_modality_validation(validation_df)
    
    # 3. scIB-BASED COMPREHENSIVE EVALUATION
    print("\\n=== scIB-Based Integration Quality Assessment ===")
    scib_metrics_df, modality_clusters = compute_scIB_integration_quality_comprehensive(
        adata=adata,
        embedding_keys=embedding_keys,
        leiden_resolution=0.5,
        use_scib=True  # Use scIB package if available
    )
    
    # Summarize and visualize scIB results
    scib_summary = summarize_scIB_metrics(scib_metrics_df)
    fig_scib = plot_scIB_integration_metrics(scib_metrics_df)
    add_scIB_clusters_to_adata(adata, modality_clusters)
    
    print("\\nscIB Integration Quality Summary:")
    print(scib_summary)
    
    # 4. INTERPRETATION GUIDANCE
    print("\\n=== Results Interpretation ===")
    print("For a good joint (WNN) embedding, you should see:")
    print("- High ARI/NMI when WNN clusters are compared to RNA and ATAC clusters")
    print("- Lower ARI/NMI between RNA-only and ATAC-only clusters")
    print("- High ASW_label scores (good separation of biological groups)")
    print("- High Graph_connectivity scores (preserved neighborhood structure)")
    
    # Check available data
    print("\\nAvailable connectivity matrices:")
    for key in adata.obsp.keys():
        if 'connectivities' in key:
            print(f"  - {key}: {adata.obsp[key].shape}")
            
    print("\\nAvailable embeddings:")
    for key in adata.obsm.keys():
        print(f"  - {key}: {adata.obsm[key].shape}")
    """)


if __name__ == "__main__":
    example_neighborhood_purity_analysis()