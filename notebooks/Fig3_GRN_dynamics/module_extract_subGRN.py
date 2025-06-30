# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Dict, List, Tuple
# import pickle
# import os

# # =============================================================================
# # STEP 1: Extract top motifs above 99th percentile
# # =============================================================================

# def extract_top_motifs_per_cluster(clusters_motifs_df, percentile_threshold=99):
#     """
#     Extract motifs above specified percentile for each cluster.
    
#     Parameters:
#     -----------
#     clusters_motifs_df : pd.DataFrame
#         Clusters x motifs with enrichment scores
#     percentile_threshold : float
#         Percentile threshold (e.g., 99 for 99th percentile)
        
#     Returns:
#     --------
#     cluster_top_motifs : dict
#         {cluster_id: [list_of_top_motifs]}
#     """
    
#     cluster_top_motifs = {}
    
#     print(f"Extracting motifs above {percentile_threshold}th percentile...")
#     print("="*60)
    
#     for cluster_id in clusters_motifs_df.index:
#         scores = clusters_motifs_df.loc[cluster_id]
#         threshold = np.percentile(scores, percentile_threshold)
        
#         top_motifs = scores[scores >= threshold].sort_values(ascending=False)
#         cluster_top_motifs[cluster_id] = top_motifs.index.tolist()
        
#         print(f"Cluster {cluster_id}: {len(top_motifs)} motifs above {threshold:.3f}")
    
#     return cluster_top_motifs

# # =============================================================================
# # STEP 2: Map motifs to TFs
# # =============================================================================

# def create_motif_tf_mapping(info_motifs_df=None, motif_database_path=None):
#     """
#     Create mapping from motif IDs to transcription factors.
    
#     Parameters:
#     -----------
#     info_motifs_df : pd.DataFrame or None
#         DataFrame with motif info (your format with 'indirect' column)
#     motif_database_path : str or None
#         Alternative: path to motif-TF mapping file
        
#     Returns:
#     --------
#     motif_tf_dict : dict
#         {motif_id: [list_of_tfs]}
#     """
    
#     if info_motifs_df is not None:
#         # Use your motif info dataframe
#         print("Creating motif-TF mapping from info_motifs DataFrame...")
#         motif_tf_dict = parse_motif_info_df(info_motifs_df)
        
#     elif motif_database_path and os.path.exists(motif_database_path):
#         # Load from file (assume format: motif_id \t tf_name)
#         motif_tf_df = pd.read_csv(motif_database_path, sep='\t', 
#                                  names=['motif_id', 'tf_name'])
#         motif_tf_dict = motif_tf_df.groupby('motif_id')['tf_name'].apply(list).to_dict()
#     else:
#         # Create simple mapping based on motif names (for zebrafish)
#         # This is a placeholder - replace with your actual database
#         print("Using built-in motif-TF mapping...")
#         motif_tf_dict = create_builtin_motif_mapping()
    
#     print(f"Loaded mapping for {len(motif_tf_dict)} motifs")
#     return motif_tf_dict

# def parse_motif_info_df(info_motifs_df):
#     """
#     Parse your motif info DataFrame to create motif-TF mapping.
    
#     Parameters:
#     -----------
#     info_motifs_df : pd.DataFrame
#         DataFrame with columns: motif (index), direct, indirect
        
#     Returns:
#     --------
#     motif_tf_dict : dict
#         {motif_id: [list_of_tfs]}
#     """
    
#     motif_tf_dict = {}
    
#     for motif_id in info_motifs_df.index:
#         tfs = []
        
#         # Check 'direct' column first
#         if 'direct' in info_motifs_df.columns:
#             direct_tfs = info_motifs_df.loc[motif_id, 'direct']
#             if pd.notna(direct_tfs) and str(direct_tfs).strip() != 'NaN':
#                 direct_list = [tf.strip() for tf in str(direct_tfs).split(',') if tf.strip()]
#                 tfs.extend(direct_list)
        
#         # Check 'indirect' column
#         if 'indirect' in info_motifs_df.columns:
#             indirect_tfs = info_motifs_df.loc[motif_id, 'indirect']
#             if pd.notna(indirect_tfs) and str(indirect_tfs).strip() != 'NaN':
#                 indirect_list = [tf.strip() for tf in str(indirect_tfs).split(',') if tf.strip()]
#                 tfs.extend(indirect_list)
        
#         # Remove duplicates while preserving order
#         unique_tfs = list(dict.fromkeys(tfs))
        
#         # Only add if we found TFs
#         if unique_tfs:
#             motif_tf_dict[motif_id] = unique_tfs
    
#     print(f"Parsed {len(motif_tf_dict)} motifs with TF mappings")
#     print(f"Example mappings:")
#     for i, (motif, tfs) in enumerate(list(motif_tf_dict.items())[:3]):
#         print(f"  {motif}: {tfs[:5]}{'...' if len(tfs) > 5 else ''}")
    
#     return motif_tf_dict

# def create_builtin_motif_mapping():
#     """
#     Create a built-in motif to TF mapping for common zebrafish TFs.
#     Replace this with your actual motif database.
#     """
    
#     # Example mapping - you'll need to replace this with your actual data
#     mapping = {
#         'M00008_2.00': ['sox2'],
#         'M00045_2.00': ['pax6a', 'pax6b'],
#         'M00056_2.00': ['gata1', 'gata2'],
#         'M00066_2.00': ['tbx16', 'tbx6'],
#         'M00070_2.00': ['foxp4', 'foxp1'],
#         'M00111_2.00': ['myf5', 'myod1'],
#         'M00112_2.00': ['mef2c', 'mef2d'],
#         'M00113_2.00': ['nkx2.2a', 'nkx2.2b'],
#         'M00114_2.00': ['lmo2', 'tal1'],
#         # Add more mappings based on your actual motif database
#     }
    
#     print("WARNING: Using example motif-TF mapping. Replace with your actual database!")
#     return mapping

# def extract_cluster_associated_tfs(cluster_top_motifs, motif_tf_mapping):
#     """
#     Map top motifs to associated TFs for each cluster.
    
#     Parameters:
#     -----------
#     cluster_top_motifs : dict
#         {cluster_id: [list_of_motifs]}
#     motif_tf_mapping : dict
#         {motif_id: [list_of_tfs]}
        
#     Returns:
#     --------
#     cluster_tfs_df : pd.DataFrame
#         DataFrame with cluster_id and associated_TFs columns
#     """
    
#     cluster_tf_data = []
    
#     print("\nMapping motifs to TFs for each cluster...")
#     print("="*50)
    
#     for cluster_id, motifs in cluster_top_motifs.items():
#         # Get all TFs associated with top motifs in this cluster
#         cluster_tfs = []
#         motifs_with_tfs = 0
        
#         for motif in motifs:
#             if motif in motif_tf_mapping:
#                 cluster_tfs.extend(motif_tf_mapping[motif])
#                 motifs_with_tfs += 1
        
#         # Remove duplicates while preserving order
#         unique_tfs = list(dict.fromkeys(cluster_tfs))
        
#         cluster_tf_data.append({
#             'cluster_id': cluster_id,
#             'n_top_motifs': len(motifs),
#             'n_motifs_with_tfs': motifs_with_tfs,
#             'associated_TFs': unique_tfs,
#             'n_associated_TFs': len(unique_tfs)
#         })
        
#         print(f"Cluster {cluster_id}: {len(motifs)} motifs → {motifs_with_tfs} with TFs → {len(unique_tfs)} unique TFs")
    
#     cluster_tfs_df = pd.DataFrame(cluster_tf_data)
    
#     print(f"\nSummary: {cluster_tfs_df['n_associated_TFs'].sum()} total TF associations")
    
#     return cluster_tfs_df

# # =============================================================================
# # STEP 3: Extract associated genes per cluster
# # =============================================================================

# def extract_cluster_associated_genes(clusters_genes_df, correlation_threshold=0.5):
#     """
#     Extract genes associated with each cluster based on correlation threshold.
    
#     Parameters:
#     -----------
#     clusters_genes_df : pd.DataFrame
#         Clusters x genes with correlation/association scores
#     correlation_threshold : float
#         Minimum correlation to consider gene as associated
        
#     Returns:
#     --------
#     cluster_genes_dict : dict
#         {cluster_id: [list_of_associated_genes]}
#     """
    
#     cluster_genes_dict = {}
    
#     print(f"Extracting genes with correlation >= {correlation_threshold}...")
#     print("="*60)
    
#     for cluster_id in clusters_genes_df.index:
#         scores = clusters_genes_df.loc[cluster_id]
#         associated_genes = scores[scores >= correlation_threshold].sort_values(ascending=False)
#         cluster_genes_dict[cluster_id] = associated_genes.index.tolist()
        
#         print(f"Cluster {cluster_id}: {len(associated_genes)} associated genes")
    
#     return cluster_genes_dict

# def add_genes_to_cluster_df(cluster_tfs_df, cluster_genes_dict):
#     """
#     Add associated genes to the cluster TFs dataframe.
#     """
    
#     cluster_tfs_df['associated_genes'] = cluster_tfs_df['cluster_id'].map(cluster_genes_dict)
#     cluster_tfs_df['n_associated_genes'] = cluster_tfs_df['associated_genes'].apply(len)
    
#     print("\nUpdated Cluster Summary:")
#     print(cluster_tfs_df[['cluster_id', 'n_associated_TFs', 'n_associated_genes']])
    
#     return cluster_tfs_df

# # =============================================================================
# # STEP 4: Create binary TF-target matrix (potential subGRN)
# # =============================================================================

# def create_cluster_tf_target_matrix(cluster_tfs_df, cluster_id):
#     """
#     Create binary TF x target gene matrix for a specific cluster.
    
#     Parameters:
#     -----------
#     cluster_tfs_df : pd.DataFrame
#         DataFrame with TFs and genes per cluster
#     cluster_id : str
#         Specific cluster to create matrix for
        
#     Returns:
#     --------
#     tf_target_matrix : pd.DataFrame
#         Binary matrix (TFs x genes) with 1s for potential relationships
#     """
    
#     cluster_row = cluster_tfs_df[cluster_tfs_df['cluster_id'] == cluster_id].iloc[0]
    
#     tfs = cluster_row['associated_TFs']
#     genes = cluster_row['associated_genes']
    
#     if len(tfs) == 0 or len(genes) == 0:
#         print(f"Warning: Cluster {cluster_id} has no TFs or genes")
#         return pd.DataFrame()
    
#     # Create binary matrix (all 1s for now - these are potential relationships)
#     tf_target_matrix = pd.DataFrame(
#         1, 
#         index=tfs,
#         columns=genes
#     )
    
#     print(f"Created {len(tfs)} x {len(genes)} potential TF-target matrix for cluster {cluster_id}")
    
#     return tf_target_matrix

# # =============================================================================
# # STEP 5 & 6: Load GRN and extract active subGRN
# # =============================================================================

# def load_grn_data(grn_path, cell_type=None, timepoint=None):
#     """
#     Load GRN data from file.
    
#     Parameters:
#     -----------
#     grn_path : str
#         Path to GRN file or directory
#     cell_type : str
#         Cell type identifier
#     timepoint : str  
#         Timepoint identifier
        
#     Returns:
#     --------
#     grn_df : pd.DataFrame
#         TFs x genes matrix with edge strengths
#     """
    
#     # This depends on your GRN file format - adjust as needed
#     if cell_type and timepoint:
#         full_path = f"{grn_path}/grn_{cell_type}_{timepoint}.csv"
#     else:
#         full_path = grn_path
    
#     try:
#         if full_path.endswith('.pkl'):
#             grn_df = pd.read_pickle(full_path)
#         elif full_path.endswith('.csv'):
#             grn_df = pd.read_csv(full_path, index_col=0)
#         else:
#             raise ValueError("Unsupported file format")
            
#         print(f"Loaded GRN: {grn_df.shape} (TFs x genes)")
#         print(f"TFs: {len(grn_df.index)}, Genes: {len(grn_df.columns)}")
        
#     except Exception as e:
#         print(f"Error loading GRN: {e}")
#         print("Creating mock GRN for demonstration...")
#         grn_df = create_mock_grn()
    
#     return grn_df

# def create_mock_grn():
#     """Create a mock GRN for testing purposes."""
    
#     tfs = ['sox2', 'pax6a', 'gata1', 'tbx16', 'foxp4', 'myf5', 'mef2c', 'nkx2.2a']
#     genes = [f'gene_{i}' for i in range(50)]
    
#     # Random edge strengths between -3 and 3
#     grn_matrix = np.random.normal(0, 1, (len(tfs), len(genes)))
#     grn_df = pd.DataFrame(grn_matrix, index=tfs, columns=genes)
    
#     print("Created mock GRN for demonstration")
#     return grn_df

# def extract_active_subgrn(grn_df, tf_target_matrix, edge_strength_threshold=0.1):
#     """
#     Extract active subGRN based on potential TF-target relationships.
    
#     Parameters:
#     -----------
#     grn_df : pd.DataFrame
#         Full GRN (TFs x genes with edge strengths)
#     tf_target_matrix : pd.DataFrame
#         Binary potential relationships (TFs x genes)
#     edge_strength_threshold : float
#         Minimum edge strength to keep
        
#     Returns:
#     --------
#     subgrn_df : pd.DataFrame
#         Active subGRN with only significant edges
#     """
    
#     # Find overlapping TFs and genes
#     common_tfs = list(set(grn_df.index) & set(tf_target_matrix.index))
#     common_genes = list(set(grn_df.columns) & set(tf_target_matrix.columns))
    
#     if len(common_tfs) == 0 or len(common_genes) == 0:
#         print("Warning: No overlap between GRN and potential relationships")
#         return pd.DataFrame()
    
#     print(f"Overlap: {len(common_tfs)} TFs, {len(common_genes)} genes")
    
#     # Extract subsets
#     grn_subset = grn_df.loc[common_tfs, common_genes]
#     potential_subset = tf_target_matrix.loc[common_tfs, common_genes]
    
#     # Create mask: predicted relationships AND strong edges
#     strength_mask = grn_subset.abs() >= edge_strength_threshold
#     potential_mask = potential_subset == 1
    
#     final_mask = strength_mask & potential_mask
    
#     # Apply mask
#     subgrn_df = grn_subset.copy()
#     subgrn_df[~final_mask] = 0
    
#     # Remove empty rows/columns
#     subgrn_df = subgrn_df.loc[(subgrn_df != 0).any(axis=1), (subgrn_df != 0).any(axis=0)]
    
#     n_edges = (subgrn_df != 0).sum().sum()
#     print(f"Extracted subGRN: {subgrn_df.shape}, {n_edges} active edges")
    
#     return subgrn_df

# def analyze_cluster_grn(cluster_tfs_df, cluster_id, grn_df, edge_strength_threshold=0.1):
#     """
#     Complete analysis pipeline for one cluster.
    
#     Parameters:
#     -----------
#     cluster_tfs_df : pd.DataFrame
#         DataFrame with cluster TFs and genes
#     cluster_id : str
#         Cluster to analyze
#     grn_df : pd.DataFrame
#         Full GRN data
#     edge_strength_threshold : float
#         Minimum edge strength threshold
        
#     Returns:
#     --------
#     subgrn_df : pd.DataFrame
#         Active subGRN for this cluster
#     """
    
#     print(f"\nAnalyzing Cluster {cluster_id}")
#     print("="*50)
    
#     # Step 1: Create potential TF-target matrix
#     tf_target_matrix = create_cluster_tf_target_matrix(cluster_tfs_df, cluster_id)
    
#     if tf_target_matrix.empty:
#         return pd.DataFrame()
    
#     # Step 2: Extract active subGRN
#     subgrn_df = extract_active_subgrn(grn_df, tf_target_matrix, edge_strength_threshold)
    
#     return subgrn_df

# # =============================================================================
# # STEP 7: Visualization with NetworkX
# # =============================================================================

# def visualize_subgrn(subgrn_df, cluster_id, layout='spring', figsize=(12, 10)):
#     """
#     Visualize subGRN using NetworkX.
    
#     Parameters:
#     -----------
#     subgrn_df : pd.DataFrame
#         SubGRN (TFs x genes with edge strengths)
#     cluster_id : str
#         Cluster identifier for title
#     layout : str
#         Network layout ('spring', 'circular', 'shell')
#     figsize : tuple
#         Figure size
#     """
    
#     if subgrn_df.empty:
#         print(f"No active subGRN for cluster {cluster_id}")
#         return None
    
#     # Create directed graph
#     G = nx.DiGraph()
    
#     # Add nodes
#     tfs = subgrn_df.index.tolist()
#     genes = subgrn_df.columns.tolist()
    
#     G.add_nodes_from(tfs, node_type='TF')
#     G.add_nodes_from(genes, node_type='gene')
    
#     # Add edges with weights
#     edges = []
#     edge_weights = []
#     edge_colors = []
    
#     for tf in tfs:
#         for gene in genes:
#             weight = subgrn_df.loc[tf, gene]
#             if weight != 0:
#                 G.add_edge(tf, gene, weight=weight)
#                 edges.append((tf, gene))
#                 edge_weights.append(abs(weight))
#                 edge_colors.append('red' if weight > 0 else 'blue')
    
#     if len(edges) == 0:
#         print(f"No edges to plot for cluster {cluster_id}")
#         return None
    
#     # Set up plot
#     plt.figure(figsize=figsize)
    
#     # Choose layout
#     if layout == 'spring':
#         pos = nx.spring_layout(G, k=3, iterations=50)
#     elif layout == 'circular':
#         pos = nx.circular_layout(G)
#     elif layout == 'shell':
#         pos = nx.shell_layout(G, nlist=[tfs, genes])
#     else:
#         pos = nx.spring_layout(G)
    
#     # Draw nodes
#     tf_nodes = [n for n in G.nodes() if n in tfs]
#     gene_nodes = [n for n in G.nodes() if n in genes]
    
#     nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, 
#                           node_color='lightcoral', node_size=800, 
#                           label='TFs', alpha=0.8)
#     nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, 
#                           node_color='lightblue', node_size=600, 
#                           label='Genes', alpha=0.8)
    
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, edgelist=edges,
#                           edge_color=edge_colors, 
#                           width=[w*2 for w in edge_weights],
#                           alpha=0.6, arrows=True, arrowsize=20)
    
#     # Add labels
#     nx.draw_networkx_labels(G, pos, font_size=8)
    
#     plt.title(f'Cluster {cluster_id} SubGRN\n'
#               f'{len(tfs)} TFs → {len(genes)} genes ({len(edges)} edges)')
#     plt.legend()
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
#     # Print network statistics
#     print(f"\nNetwork Statistics for Cluster {cluster_id}:")
#     print(f"  Nodes: {G.number_of_nodes()} ({len(tfs)} TFs, {len(genes)} genes)")
#     print(f"  Edges: {G.number_of_edges()}")
#     print(f"  Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
    
#     return G

# # =============================================================================
# # COMPLETE PIPELINE FUNCTION
# # =============================================================================

# def complete_cluster_grn_pipeline(
#     clusters_motifs_df,
#     clusters_genes_df, 
#     grn_path,
#     info_motifs_df=None,
#     cell_type=None,
#     timepoint=None,
#     motif_database_path=None,
#     percentile_threshold=99,
#     correlation_threshold=0.5,
#     edge_strength_threshold=0.1,
#     clusters_to_analyze=None
# ):
#     """
#     Complete pipeline from peak clusters to subGRN analysis.
    
#     Parameters:
#     -----------
#     clusters_motifs_df : pd.DataFrame
#         Clusters x motifs with enrichment scores
#     clusters_genes_df : pd.DataFrame
#         Clusters x genes with correlation scores
#     grn_path : str
#         Path to GRN data
#     info_motifs_df : pd.DataFrame
#         Your motif info DataFrame with 'indirect' column
#     cell_type : str
#         Cell type for GRN
#     timepoint : str
#         Timepoint for GRN
#     motif_database_path : str
#         Alternative path to motif-TF mapping
#     percentile_threshold : float
#         Percentile for top motifs (default 99)
#     correlation_threshold : float
#         Minimum gene correlation (default 0.5)
#     edge_strength_threshold : float
#         Minimum GRN edge strength (default 0.1)
#     clusters_to_analyze : list
#         Specific clusters to analyze, or None for all
        
#     Returns:
#     --------
#     results : dict
#         Complete analysis results
#     """
    
#     print("COMPLETE CLUSTER-GRN ANALYSIS PIPELINE")
#     print("="*70)
    
#     # Step 1: Extract top motifs
#     print("\n1. EXTRACTING TOP MOTIFS")
#     cluster_top_motifs = extract_top_motifs_per_cluster(
#         clusters_motifs_df, percentile_threshold
#     )
    
#     # Step 2: Map motifs to TFs
#     print("\n2. MAPPING MOTIFS TO TFs")
#     motif_tf_mapping = create_motif_tf_mapping(
#         info_motifs_df=info_motifs_df, 
#         motif_database_path=motif_database_path
#     )
#     cluster_tfs_df = extract_cluster_associated_tfs(cluster_top_motifs, motif_tf_mapping)
    
#     # Step 3: Add associated genes
#     print("\n3. EXTRACTING ASSOCIATED GENES")
#     cluster_genes_dict = extract_cluster_associated_genes(
#         clusters_genes_df, correlation_threshold
#     )
#     cluster_tfs_df = add_genes_to_cluster_df(cluster_tfs_df, cluster_genes_dict)
    
#     # Step 4: Load GRN
#     print("\n4. LOADING GRN DATA")
#     grn_df = load_grn_data(grn_path, cell_type, timepoint)
    
#     # Step 5: Analyze clusters
#     print("\n5. ANALYZING CLUSTER SubGRNs")
    
#     if clusters_to_analyze is None:
#         clusters_to_analyze = cluster_tfs_df['cluster_id'].tolist()
    
#     subgrns = {}
    
#     for cluster_id in clusters_to_analyze[:5]:  # Limit to first 5 for demo
#         subgrn = analyze_cluster_grn(
#             cluster_tfs_df, cluster_id, grn_df, edge_strength_threshold
#         )
#         subgrns[cluster_id] = subgrn
        
#         # Visualize if subGRN exists and is not too large
#         if not subgrn.empty and subgrn.shape[0] <= 20 and subgrn.shape[1] <= 30:
#             visualize_subgrn(subgrn, cluster_id)
    
#     return {
#         'cluster_top_motifs': cluster_top_motifs,
#         'motif_tf_mapping': motif_tf_mapping,
#         'cluster_tfs_df': cluster_tfs_df,
#         'grn_df': grn_df,
#         'subgrns': subgrns
#     }

# # =============================================================================
# # USAGE EXAMPLE
# # =============================================================================

# """
# # Example usage:

# results = complete_cluster_grn_pipeline(
#     clusters_motifs_df=clusters_motifs_df,
#     clusters_genes_df=clusters_genes_df,  # You need this - cluster x gene correlations
#     grn_path="path/to/your/grn_spinal_cord_14hpf.csv",
#     cell_type="spinal_cord", 
#     timepoint="14hpf",
#     motif_database_path="path/to/motif_tf_mapping.txt",  # Optional
#     percentile_threshold=99,
#     correlation_threshold=0.5,
#     edge_strength_threshold=0.1,
#     clusters_to_analyze=['22_0', '22_1', '22_2']  # Your specific clusters
# )

# # Access results:
# cluster_tfs_df = results['cluster_tfs_df']
# subgrns = results['subgrns']

# # Analyze specific cluster
# subgrn_22_0 = subgrns['22_0']
# print(subgrn_22_0)
# """

# print("Complete pipeline ready!")
# print("You'll need: clusters_motifs_df, clusters_genes_df, and GRN data")