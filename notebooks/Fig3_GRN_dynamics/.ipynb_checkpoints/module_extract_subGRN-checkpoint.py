# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Dict, List, Tuple
# import pickle
# import os

import pandas as pd
import numpy as np

# =============================================================================
# STEP 1: Create clusters-by-motifs dict
# =============================================================================

# def get_top_motifs_per_cluster(clusters_motifs_df, percentile_threshold=99):
#     """
#     Step 1: Extract top motifs for each cluster above percentile threshold.
    
#     Parameters:
#     -----------
#     clusters_motifs_df : pd.DataFrame
#         Clusters x motifs with enrichment scores
#     percentile_threshold : float
#         Percentile threshold (e.g., 99 for 99th percentile)
        
#     Returns:
#     --------
#     clusters_motifs_dict : dict
#         {cluster_id: [list_of_top_motifs]}
#     """
    
#     clusters_motifs_dict = {}
    
#     for cluster_id in clusters_motifs_df.index:
#         scores = clusters_motifs_df.loc[cluster_id]
#         threshold = np.percentile(scores, percentile_threshold)
        
#         top_motifs = scores[scores >= threshold].sort_values(ascending=False)
#         clusters_motifs_dict[cluster_id] = top_motifs.index.tolist()
        
#         # print(f"Cluster {cluster_id}: {len(top_motifs)} motifs above {threshold:.3f}")
    
#     return clusters_motifs_dict
def get_top_motifs_per_cluster(clusters_motifs_df, method="threshold", threshold_value=2):
    """
    Step 1: Extract top motifs for each cluster using percentile or z-score threshold.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs with enrichment scores
    method : str
        Either "percentile" or "threshold" (z-score)
    threshold_value : float
        - If method="percentile": percentile threshold (e.g., 99 for 99th percentile)
        - If method="threshold": z-score threshold (e.g., 2.0 for z > 2.0)
        
    Returns:
    --------
    clusters_motifs_dict : dict
        {cluster_id: [list_of_top_motifs]}
    """
    
    clusters_motifs_dict = {}
    
    print(f"Using {method} method with threshold: {threshold_value}")
    
    for cluster_id in clusters_motifs_df.index:
        scores = clusters_motifs_df.loc[cluster_id]
        
        # default is "threshold" using z-score values
        if method == "threshold":
            # Use direct z-score threshold
            threshold = threshold_value
            top_motifs = scores[scores >= threshold].sort_values(ascending=False)
        elif method == "percentile":
            # Use percentile-based threshold
            threshold = np.percentile(scores, threshold_value)
            top_motifs = scores[scores >= threshold].sort_values(ascending=False)
            
        else:
            raise ValueError("method must be either 'percentile' or 'threshold'")
        
        clusters_motifs_dict[cluster_id] = top_motifs.index.tolist()
        
        # Optional: print details for verification
        # print(f"Cluster {cluster_id}: {len(top_motifs)} motifs above {threshold:.3f}")
    
    return clusters_motifs_dict

# =============================================================================
# STEP 2: Create clusters-by-TFs using info_motifs
# =============================================================================

def get_tfs_from_motifs(clusters_motifs_dict, info_motifs_df):
    """
    Step 2: Map motifs to TFs for each cluster using info_motifs DataFrame.
    
    Parameters:
    -----------
    clusters_motifs_dict : dict
        {cluster_id: [list_of_motifs]}
    info_motifs_df : pd.DataFrame
        Your motif info with 'indirect' column containing TF names
        
    Returns:
    --------
    clusters_tfs_dict : dict
        {cluster_id: [list_of_TFs]}
    """
    
    clusters_tfs_dict = {}
    
    for cluster_id, motifs in clusters_motifs_dict.items():
        cluster_tfs = []
        
        for motif in motifs:
            if motif in info_motifs_df.index:
                # Get TFs from 'indirect' column
                tf_string = info_motifs_df.loc[motif, 'indirect']
                
                if pd.notna(tf_string) and str(tf_string).strip() != 'NaN':
                    # Split by comma and clean up
                    tfs = [tf.strip() for tf in str(tf_string).split(',') if tf.strip()]
                    cluster_tfs.extend(tfs)
        
        # Remove duplicates while preserving order
        unique_tfs = list(dict.fromkeys(cluster_tfs))
        clusters_tfs_dict[cluster_id] = unique_tfs
        
        print(f"Cluster {cluster_id}: {len(motifs)} motifs → {len(unique_tfs)} unique TFs")
    
    return clusters_tfs_dict

# =============================================================================
# STEP 3: Get associated genes per cluster
# =============================================================================

def get_associated_genes_per_cluster(adata_peaks, cluster_col="leiden_unified", gene_col="associated_gene"):
    """
    Step 3: Extract associated genes for each cluster from adata_peaks.
    
    Parameters:
    -----------
    adata_peaks : AnnData
        AnnData object with peaks data
    cluster_col : str
        Column name in adata_peaks.obs containing cluster labels (default: "leiden_unified")
    gene_col : str
        Column name in adata_peaks.obs containing gene names associated with each peak (default: "associated_gene")
        
    Returns:
    --------
    clusters_genes_dict : dict
        {cluster_id: [list_of_associated_genes]}
    """
    
    clusters_genes_dict = {}
    
    # Get cluster labels and associated genes
    cluster_labels = adata_peaks.obs[cluster_col]
    associated_genes = adata_peaks.obs[gene_col]
    
    # Get unique clusters
    unique_clusters = cluster_labels.unique()
    
    for cluster_id in unique_clusters:
        # Get peaks belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_genes = associated_genes[cluster_mask]
        
        # Remove NaN values and get unique genes
        cluster_genes_clean = cluster_genes.dropna().unique().tolist()
        
        # Filter out empty strings if any
        cluster_genes_clean = [gene for gene in cluster_genes_clean if gene and str(gene).strip()]
        
        clusters_genes_dict[cluster_id] = cluster_genes_clean
        
        print(f"Cluster {cluster_id}: {len(cluster_genes_clean)} associated genes")
    
    return clusters_genes_dict

# =============================================================================
# STEP 4: Create TFs-by-genes matrix for each cluster
# =============================================================================

def create_tf_gene_matrix_per_cluster(clusters_tfs_dict, clusters_genes_dict):
    """
    Step 4: Create binary TFs x genes matrix for each cluster.
    
    Parameters:
    -----------
    clusters_tfs_dict : dict
        {cluster_id: [list_of_TFs]}
    clusters_genes_dict : dict
        {cluster_id: [list_of_genes]}
        
    Returns:
    --------
    cluster_tf_gene_matrices : dict
        {cluster_id: pd.DataFrame (TFs x genes, binary matrix)}
    """
    
    cluster_tf_gene_matrices = {}
    
    for cluster_id in clusters_tfs_dict.keys():
        if cluster_id in clusters_genes_dict:
            
            tfs = clusters_tfs_dict[cluster_id]
            genes = clusters_genes_dict[cluster_id]
            
            if len(tfs) > 0 and len(genes) > 0:
                # Create binary matrix (all 1s for potential relationships)
                tf_gene_matrix = pd.DataFrame(
                    1, 
                    index=tfs,
                    columns=genes
                )
                cluster_tf_gene_matrices[cluster_id] = tf_gene_matrix
                
                print(f"Cluster {cluster_id}: {len(tfs)} TFs x {len(genes)} genes = {len(tfs) * len(genes)} potential edges")
            else:
                print(f"Cluster {cluster_id}: Skipped (no TFs or genes)")
    
    return cluster_tf_gene_matrices

# =============================================================================
# COMBINED FUNCTION: Run all 4 steps
# =============================================================================

def run_core_pipeline(clusters_motifs_df, info_motifs_df, adata_peaks, 
                     motif_percentile=99, cluster_col="leiden_unified", gene_col="associated_gene"):
    """
    Run all 4 core steps in sequence.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs with scores
    info_motifs_df : pd.DataFrame  
        Motif info with 'indirect' TF column
    adata_peaks : AnnData
        AnnData object with peaks data
    motif_percentile : float
        Percentile threshold for motifs (default 99)
    cluster_col : str
        Column name in adata_peaks.obs containing cluster labels (default: "leiden_unified")
    gene_col : str
        Column name in adata_peaks.obs containing gene names associated with each peak (default: "associated_gene")
        
    Returns:
    --------
    results : dict
        All results from the 4 steps
    """
    
    print("RUNNING CORE 4-STEP PIPELINE")
    print("="*50)
    
    # Step 1: Get top motifs per cluster
    print("\nSTEP 1: Getting top motifs per cluster...")
    clusters_motifs_dict = get_top_motifs_per_cluster(clusters_motifs_df, motif_percentile)
    
    # Step 2: Map motifs to TFs
    print("\nSTEP 2: Mapping motifs to TFs...")
    clusters_tfs_dict = get_tfs_from_motifs(clusters_motifs_dict, info_motifs_df)
    
    # Step 3: Get associated genes
    print("\nSTEP 3: Getting associated genes per cluster...")
    clusters_genes_dict = get_associated_genes_per_cluster(adata_peaks, cluster_col, gene_col)
    
    # Step 4: Create TF-gene matrices
    print("\nSTEP 4: Creating TF-gene matrices...")
    cluster_tf_gene_matrices = create_tf_gene_matrix_per_cluster(clusters_tfs_dict, clusters_genes_dict)
    
    print(f"\nCOMPLETE! Created matrices for {len(cluster_tf_gene_matrices)} clusters")
    
    return {
        'clusters_motifs_dict': clusters_motifs_dict,
        'clusters_tfs_dict': clusters_tfs_dict, 
        'clusters_genes_dict': clusters_genes_dict,
        'cluster_tf_gene_matrices': cluster_tf_gene_matrices
    }

# =============================================================================
# HELPER FUNCTIONS: Inspect results
# =============================================================================

def inspect_cluster_results(results, cluster_id):
    """
    Inspect results for a specific cluster.
    """
    
    print(f"\nINSPECTING CLUSTER {cluster_id}")
    print("="*40)
    
    if cluster_id in results['clusters_motifs_dict']:
        motifs = results['clusters_motifs_dict'][cluster_id]
        print(f"Top motifs ({len(motifs)}): {motifs[:5]}...")
    
    if cluster_id in results['clusters_tfs_dict']:
        tfs = results['clusters_tfs_dict'][cluster_id]
        print(f"Associated TFs ({len(tfs)}): {tfs[:10]}...")
    
    if cluster_id in results['clusters_genes_dict']:
        genes = results['clusters_genes_dict'][cluster_id]
        print(f"Associated genes ({len(genes)}): {genes[:10]}...")
    
    if cluster_id in results['cluster_tf_gene_matrices']:
        matrix = results['cluster_tf_gene_matrices'][cluster_id]
        print(f"TF-gene matrix: {matrix.shape}")
        print("Sample matrix:")
        print(matrix.iloc[:3, :5])

def summary_all_clusters(results):
    """
    Print summary for all clusters.
    """
    
    print("\nSUMMARY FOR ALL CLUSTERS")
    print("="*50)
    
    cluster_summary = []
    
    for cluster_id in results['clusters_motifs_dict'].keys():
        n_motifs = len(results['clusters_motifs_dict'].get(cluster_id, []))
        n_tfs = len(results['clusters_tfs_dict'].get(cluster_id, []))
        n_genes = len(results['clusters_genes_dict'].get(cluster_id, []))
        
        has_matrix = cluster_id in results['cluster_tf_gene_matrices']
        matrix_size = 0
        if has_matrix:
            matrix = results['cluster_tf_gene_matrices'][cluster_id]
            matrix_size = matrix.shape[0] * matrix.shape[1]
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'n_motifs': n_motifs,
            'n_tfs': n_tfs,
            'n_genes': n_genes,
            'has_matrix': has_matrix,
            'matrix_size': matrix_size
        })
    
    summary_df = pd.DataFrame(cluster_summary)
    print(summary_df)
    
    return summary_df

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
# SIMPLE USAGE:

# Run all 4 steps at once
results = run_core_pipeline(
    clusters_motifs_df=clusters_motifs_df,
    info_motifs_df=info_motifs,
    adata_peaks=adata_peaks,  # AnnData object with peaks data
    motif_percentile=99,
    cluster_col="leiden_unified",  # Column name for cluster labels
    gene_col="associated_gene"     # Column name for associated genes
)

# Inspect results
summary_all_clusters(results)
inspect_cluster_results(results, '22_0')

# Get TF-gene matrix for a specific cluster
tf_gene_matrix_22_0 = results['cluster_tf_gene_matrices']['22_0']
print(tf_gene_matrix_22_0)

# OR run step by step:

# Step 1
clusters_motifs_dict = get_top_motifs_per_cluster(clusters_motifs_df, 99)

# Step 2  
clusters_tfs_dict = get_tfs_from_motifs(clusters_motifs_dict, info_motifs)

# Step 3
clusters_genes_dict = get_associated_genes_per_cluster(adata_peaks, "leiden_unified", "associated_gene")

# Step 4
cluster_tf_gene_matrices = create_tf_gene_matrix_per_cluster(clusters_tfs_dict, clusters_genes_dict)
"""

print("Simple 4-step pipeline ready!")
print("Use: run_core_pipeline(clusters_motifs_df, info_motifs, adata_peaks, cluster_col='leiden_unified', gene_col='associated_gene')")

# =============================================================================
# STEP 5: Filter with CellOracle GRN to get active sub-GRNs
# =============================================================================

def filter_with_celloracle_grn(cluster_tf_gene_matrices, celloracle_grn_df, 
                              edge_strength_threshold=0.1, return_edge_weights=True):
    """
    Filter cluster TF-gene matrices with CellOracle GRN to extract active sub-GRNs.
    
    Parameters:
    -----------
    cluster_tf_gene_matrices : dict
        {cluster_id: pd.DataFrame (TFs x genes, binary matrix)}
    celloracle_grn_df : pd.DataFrame
        TFs x genes matrix with edge strengths from CellOracle
    edge_strength_threshold : float
        Minimum edge strength to consider active (default 0.1)
    return_edge_weights : bool
        If True, return actual edge weights; if False, return binary matrix
        
    Returns:
    --------
    active_subgrns : dict
        {cluster_id: pd.DataFrame (active sub-GRN with edge weights or binary)}
    """
    
    active_subgrns = {}
    
    print(f"Filtering cluster matrices with CellOracle GRN...")
    print(f"CellOracle GRN shape: {celloracle_grn_df.shape}")
    print(f"Edge strength threshold: {edge_strength_threshold}")
    print("="*60)
    
    for cluster_id, cluster_matrix in cluster_tf_gene_matrices.items():
        print(f"\nProcessing Cluster {cluster_id}...")
        
        # Find overlapping TFs and genes
        common_tfs = list(set(cluster_matrix.index) & set(celloracle_grn_df.index))
        common_genes = list(set(cluster_matrix.columns) & set(celloracle_grn_df.columns))
        
        if len(common_tfs) == 0 or len(common_genes) == 0:
            print(f"  No overlap found (TFs: {len(common_tfs)}, Genes: {len(common_genes)})")
            active_subgrns[cluster_id] = pd.DataFrame()
            continue
        
        print(f"  Overlap: {len(common_tfs)} TFs, {len(common_genes)} genes")
        
        # Extract overlapping subsets
        cluster_subset = cluster_matrix.loc[common_tfs, common_genes]
        grn_subset = celloracle_grn_df.loc[common_tfs, common_genes]
        
        # Create masks
        cluster_mask = cluster_subset == 1  # Potential relationships from cluster
        strength_mask = grn_subset.abs() >= edge_strength_threshold  # Strong edges from GRN
        
        # Final mask: both conditions must be true
        active_mask = cluster_mask & strength_mask
        
        # Create output matrix
        if return_edge_weights:
            # Return actual edge weights where active
            subgrn = grn_subset.copy()
            subgrn[~active_mask] = 0
        else:
            # Return binary matrix
            subgrn = active_mask.astype(int)
        
        # Remove empty rows/columns
        subgrn = subgrn.loc[(subgrn != 0).any(axis=1), (subgrn != 0).any(axis=0)]
        
        active_subgrns[cluster_id] = subgrn
        
        n_active_edges = (subgrn != 0).sum().sum()
        n_potential_edges = (cluster_subset == 1).sum().sum()
        
        print(f"  Result: {subgrn.shape[0]} TFs x {subgrn.shape[1]} genes")
        print(f"  Active edges: {n_active_edges}/{n_potential_edges} ({n_active_edges/n_potential_edges*100:.1f}%)")
    
    print(f"\nCompleted filtering for {len(active_subgrns)} clusters")
    return active_subgrns

def analyze_subgrn_statistics(active_subgrns):
    """
    Analyze statistics of the active sub-GRNs.
    
    Parameters:
    -----------
    active_subgrns : dict
        {cluster_id: pd.DataFrame (active sub-GRN)}
        
    Returns:
    --------
    stats_df : pd.DataFrame
        Summary statistics for each cluster
    """
    
    stats_data = []
    
    for cluster_id, subgrn in active_subgrns.items():
        if subgrn.empty:
            stats_data.append({
                'cluster_id': cluster_id,
                'n_tfs': 0,
                'n_genes': 0,
                'n_edges': 0,
                'density': 0,
                'avg_edge_strength': 0,
                'max_edge_strength': 0
            })
        else:
            n_tfs = subgrn.shape[0]
            n_genes = subgrn.shape[1]
            n_edges = (subgrn != 0).sum().sum()
            density = n_edges / (n_tfs * n_genes) if (n_tfs * n_genes) > 0 else 0
            
            # Calculate edge strength statistics (if not binary)
            non_zero_values = subgrn.values[subgrn.values != 0]
            avg_strength = np.mean(np.abs(non_zero_values)) if len(non_zero_values) > 0 else 0
            max_strength = np.max(np.abs(non_zero_values)) if len(non_zero_values) > 0 else 0
            
            stats_data.append({
                'cluster_id': cluster_id,
                'n_tfs': n_tfs,
                'n_genes': n_genes,
                'n_edges': n_edges,
                'density': density,
                'avg_edge_strength': avg_strength,
                'max_edge_strength': max_strength
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    print("\nSub-GRN Statistics:")
    print("="*50)
    print(stats_df.round(3))
    
    return stats_df

def get_top_regulators_per_cluster(active_subgrns, top_n=10):
    """
    Get top regulators (TFs) for each cluster based on number of targets or edge strength.
    
    Parameters:
    -----------
    active_subgrns : dict
        {cluster_id: pd.DataFrame (active sub-GRN)}
    top_n : int
        Number of top regulators to return per cluster
        
    Returns:
    --------
    top_regulators : dict
        {cluster_id: [(tf_name, n_targets, avg_strength), ...]}
    """
    
    top_regulators = {}
    
    for cluster_id, subgrn in active_subgrns.items():
        if subgrn.empty:
            top_regulators[cluster_id] = []
            continue
        
        regulator_stats = []
        
        for tf in subgrn.index:
            tf_row = subgrn.loc[tf]
            targets = tf_row[tf_row != 0]
            
            n_targets = len(targets)
            avg_strength = np.mean(np.abs(targets.values)) if n_targets > 0 else 0
            
            regulator_stats.append((tf, n_targets, avg_strength))
        
        # Sort by number of targets (primary) and average strength (secondary)
        regulator_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        top_regulators[cluster_id] = regulator_stats[:top_n]
        
        print(f"\nCluster {cluster_id} - Top {min(top_n, len(regulator_stats))} Regulators:")
        for i, (tf, n_targets, avg_strength) in enumerate(top_regulators[cluster_id]):
            print(f"  {i+1}. {tf}: {n_targets} targets (avg strength: {avg_strength:.3f})")
    
    return top_regulators

# # =============================================================================
# # EXTENDED PIPELINE: Include CellOracle filtering
# # =============================================================================

# def run_extended_pipeline(clusters_motifs_df, info_motifs_df, adata_peaks, celloracle_grn_df,
#                          motif_percentile=99, cluster_col="leiden_unified", gene_col="associated_gene",
#                          edge_strength_threshold=0.1, return_edge_weights=True):
#     """
#     Run complete pipeline including CellOracle GRN filtering.
    
#     Parameters:
#     -----------
#     clusters_motifs_df : pd.DataFrame
#         Clusters x motifs with scores
#     info_motifs_df : pd.DataFrame  
#         Motif info with 'indirect' TF column
#     adata_peaks : AnnData
#         AnnData object with peaks data
#     celloracle_grn_df : pd.DataFrame
#         TFs x genes matrix with edge strengths from CellOracle
#     motif_percentile : float
#         Percentile threshold for motifs (default 99)
#     cluster_col : str
#         Column name for cluster labels (default: "leiden_unified")
#     gene_col : str
#         Column name for associated genes (default: "associated_gene")
#     edge_strength_threshold : float
#         Minimum edge strength for CellOracle filtering (default 0.1)
#     return_edge_weights : bool
#         Return edge weights vs binary matrix (default True)
        
#     Returns:
#     --------
#     results : dict
#         All results including active sub-GRNs
#     """
    
#     print("RUNNING EXTENDED PIPELINE WITH CELLORACLE FILTERING")
#     print("="*70)
    
#     # Run core 4-step pipeline
#     core_results = run_core_pipeline(
#         clusters_motifs_df, info_motifs_df, adata_peaks,
#         motif_percentile, cluster_col, gene_col
#     )
    
#     # Step 5: Filter with CellOracle GRN
#     print("\nSTEP 5: Filtering with CellOracle GRN...")
#     active_subgrns = filter_with_celloracle_grn(
#         core_results['cluster_tf_gene_matrices'], 
#         celloracle_grn_df,
#         edge_strength_threshold,
#         return_edge_weights
#     )
    
#     # Analyze results
#     print("\nSTEP 6: Analyzing sub-GRN statistics...")
#     stats_df = analyze_subgrn_statistics(active_subgrns)
    
#     print("\nSTEP 7: Identifying top regulators...")
#     top_regulators = get_top_regulators_per_cluster(active_subgrns)
    
#     # Combine all results
#     extended_results = core_results.copy()
#     extended_results.update({
#         'celloracle_grn_df': celloracle_grn_df,
#         'active_subgrns': active_subgrns,
#         'subgrn_stats': stats_df,
#         'top_regulators': top_regulators
#     })
    
#     print(f"\nEXTENDED PIPELINE COMPLETE!")
#     print(f"Active sub-GRNs created for {len([k for k, v in active_subgrns.items() if not v.empty])} clusters")
    
#     return extended_results

# =============================================================================
# USAGE EXAMPLES FOR CELLORACLE FILTERING
# =============================================================================

"""
# EXAMPLE 1: Complete extended pipeline

# Load your CellOracle GRN data (TFs x genes matrix)
celloracle_grn_df = pd.read_csv("path/to/celloracle_grn_celltype_timepoint.csv", index_col=0)

# Run complete extended pipeline
extended_results = run_extended_pipeline(
    clusters_motifs_df=clusters_motifs_df,
    info_motifs_df=info_motifs,
    adata_peaks=adata_peaks,
    celloracle_grn_df=celloracle_grn_df,
    motif_percentile=99,
    cluster_col="leiden_unified",
    gene_col="associated_gene",
    edge_strength_threshold=0.1,  # Minimum edge strength from CellOracle
    return_edge_weights=True      # Return actual edge weights vs binary
)

# Access results
active_subgrns = extended_results['active_subgrns']
subgrn_stats = extended_results['subgrn_stats']
top_regulators = extended_results['top_regulators']

# Get sub-GRN for specific cluster
cluster_subgrn = active_subgrns['22_0']
print(f"Cluster 22_0 sub-GRN shape: {cluster_subgrn.shape}")
print(cluster_subgrn.head())

# EXAMPLE 2: Step-by-step with CellOracle filtering

# First run core pipeline
core_results = run_core_pipeline(
    clusters_motifs_df, info_motifs, adata_peaks,
    cluster_col="leiden_unified", gene_col="associated_gene"
)

# Then filter with CellOracle GRN
active_subgrns = filter_with_celloracle_grn(
    core_results['cluster_tf_gene_matrices'],
    celloracle_grn_df,
    edge_strength_threshold=0.1,
    return_edge_weights=True
)

# Analyze results
stats_df = analyze_subgrn_statistics(active_subgrns)
top_regulators = get_top_regulators_per_cluster(active_subgrns, top_n=5)

# EXAMPLE 3: Working with multiple cell types/timepoints

cell_types = ['spinal_cord', 'neural_crest', 'neural_tube']
timepoints = ['12hpf', '14hpf', '16hpf']

all_subgrns = {}

for cell_type in cell_types:
    for timepoint in timepoints:
        # Load cell-type/timepoint specific GRN
        grn_file = f"celloracle_grn_{cell_type}_{timepoint}.csv"
        celloracle_grn = pd.read_csv(grn_file, index_col=0)
        
        # Run pipeline
        results = run_extended_pipeline(
            clusters_motifs_df, info_motifs, adata_peaks, celloracle_grn,
            edge_strength_threshold=0.1
        )
        
        all_subgrns[f"{cell_type}_{timepoint}"] = results['active_subgrns']

# Compare sub-GRNs across conditions
for condition, subgrns in all_subgrns.items():
    print(f"\n{condition}:")
    for cluster_id, subgrn in subgrns.items():
        if not subgrn.empty:
            n_edges = (subgrn != 0).sum().sum()
            print(f"  Cluster {cluster_id}: {subgrn.shape} ({n_edges} edges)")
"""

# print("\nExtended pipeline with CellOracle filtering ready!")
# print("Use: run_extended_pipeline(clusters_motifs_df, info_motifs, adata_peaks, celloracle_grn_df)")
# print("Or step-by-step: filter_with_celloracle_grn(cluster_tf_gene_matrices, celloracle_grn_df)")

def extract_subGRN_from_cluster(grn_df, cluster_tf_gene_matrix, cluster_id):
    """
    Extract subGRN based on TF-gene relationships from peak cluster
    
    Parameters:
    - grn_df: GRN dataframe with 'source', 'target', coefficients, etc.
    - cluster_tf_gene_matrix: TF-by-genes binarized matrix (pandas DataFrame)
    - cluster_id: identifier for the cluster
    
    Returns:
    - filtered GRN dataframe containing only edges predicted by the cluster
    """
    
    # Get all TF-target pairs where matrix = 1
    tf_target_pairs = []
    for tf in cluster_tf_gene_matrix.index:
        for gene in cluster_tf_gene_matrix.columns:
            if cluster_tf_gene_matrix.loc[tf, gene] == 1:
                tf_target_pairs.append((tf, gene))
    
    # Convert to set for faster lookup
    predicted_pairs = set(tf_target_pairs)
    
    # Filter GRN to only include predicted pairs
    mask = grn_df.apply(lambda row: (row['source'], row['target']) in predicted_pairs, axis=1)
    subgrn = grn_df[mask].copy()
    
    # Add cluster information
    subgrn['cluster_id'] = cluster_id
    
    return subgrn

# Apply to all clusters
def extract_all_cluster_subGRNs(grn_df, cluster_dict):
    """
    Extract subGRNs for all clusters
    """
    all_subgrns = []
    
    for cluster_id, tf_gene_matrix in cluster_dict.items():
        subgrn = extract_subGRN_from_cluster(grn_df, tf_gene_matrix, cluster_id)
        if len(subgrn) > 0:  # Only keep non-empty subGRNs
            all_subgrns.append(subgrn)
            print(f"Cluster {cluster_id}: {len(subgrn)} edges found")
    
    return all_subgrns