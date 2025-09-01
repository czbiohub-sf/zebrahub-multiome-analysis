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
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Tuple, Dict, Optional
import warnings


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


if __name__ == "__main__":
    example_neighborhood_purity_analysis()