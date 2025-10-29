# Module for GRN visualization
# Author: YangJoon Kim
# Date: 2024-12-03
# Description: This module contains functions for visualizing GRNs.

# Load libraries
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scanpy as sc

# Define functions

# Assuming that the GRN is a dataframe with TFs-by-genes

# Compute UMAP and Leiden clustering
def compute_umap_clustering(df, 
                          n_neighbors=10, 
                          n_pcs=30, 
                          min_dist=0.1, 
                          resolution=0.1,
                          random_state=42):
    """
    Compute UMAP and Leiden clustering on a genes-by-TFs dataframe.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with genes as rows and TFs as columns
    n_neighbors : int
        Number of neighbors for UMAP computation
    n_pcs : int
        Number of principal components to use
    min_dist : float
        Minimum distance parameter for UMAP
    resolution : float
        Resolution parameter for Leiden clustering
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    adata : AnnData
        Annotated data object with UMAP and clustering results
    """
    # Create AnnData object (transpose to get TFs-by-genes)
    adata = sc.AnnData(X=df.T.values)
    adata.obs_names = df.columns
    adata.var_names = df.index
    
    # Compute PCA
    sc.tl.pca(adata, random_state=random_state)
    # Compute neighbors
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
    # Compute UMAP
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    # Perform Leiden clustering
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)

    return adata






# Create GRN network
def create_grn_network(edges_df, threshold=0.5):
    """
    Create a NetworkX graph from an edge list DataFrame.
    
    Parameters:
    -----------
    edges_df : pandas DataFrame
        DataFrame containing edges with columns 'source', 'target', and 'weight'
    threshold : float
        Minimum weight threshold for including edges
        
    Returns:
    --------
    networkx.Graph
        Network representation of the GRN
    """
    G = nx.Graph()
    
    # Filter edges based on threshold
    filtered_edges = edges_df[edges_df['weight'].abs() >= threshold]
    
    # Add edges to the graph
    for _, row in filtered_edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    return G

# Plot GRN
def plot_grn(G, node_colors=None, node_size=1000, figsize=(12, 12)):
    """
    Plot a GRN using NetworkX and Matplotlib.
    
    Parameters:
    -----------
    G : networkx.Graph
        Network to visualize
    node_colors : dict or None
        Dictionary mapping node names to colors
    node_size : int
        Size of nodes in the visualization
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Set up layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Draw nodes
    if node_colors:
        colors = [node_colors.get(node, 'gray') for node in G.nodes()]
    else:
        colors = 'lightblue'
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    plt.axis('off')
    plt.tight_layout()

# Plot edge weight distribution (histogram)
def plot_edge_weight_distribution(edges_df, bins=50):
    """
    Plot the distribution of edge weights in the GRN.
    
    Parameters:
    -----------
    edges_df : pandas DataFrame
        DataFrame containing edges with a 'weight' column
    bins : int
        Number of bins for the histogram
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=edges_df, x='weight', bins=bins)
    plt.title('Distribution of Edge Weights')
    plt.xlabel('Weight')
    plt.ylabel('Count')

# Get hub genes
def get_hub_genes(G, top_n=10):
    """
    Identify hub genes based on degree centrality.
    
    Parameters:
    -----------
    G : networkx.Graph
        Network to analyze
    top_n : int
        Number of top hub genes to return
        
    Returns:
    --------
    dict
        Dictionary of top hub genes and their degree centrality scores
    """
    degree_centrality = nx.degree_centrality(G)
    sorted_genes = dict(sorted(degree_centrality.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:top_n])
    return sorted_genes

# Plot subnetwork
def plot_subnetwork(G, center_gene, radius=1, figsize=(8, 8)):
    """
    Plot a subnetwork centered on a specific gene.
    
    Parameters:
    -----------
    G : networkx.Graph
        Full network
    center_gene : str
        Gene to center the subnetwork on
    radius : int
        Number of edges to traverse from center gene
    figsize : tuple
        Figure size (width, height)
    """
    # Extract subgraph
    nodes = {center_gene}
    current_nodes = {center_gene}
    
    for _ in range(radius):
        next_nodes = set()
        for node in current_nodes:
            next_nodes.update(G.neighbors(node))
        nodes.update(next_nodes)
        current_nodes = next_nodes
    
    subgraph = G.subgraph(nodes)
    
    # Plot
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(subgraph)
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2)
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color=['red' if node == center_gene else 'lightblue' 
                                    for node in subgraph.nodes()],
                          node_size=1000)
    
    # Add labels
    nx.draw_networkx_labels(subgraph, pos)
    
    plt.title(f"Subnetwork centered on {center_gene} (radius={radius})")
    plt.axis('off')
    plt.tight_layout()