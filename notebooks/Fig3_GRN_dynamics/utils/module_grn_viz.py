import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_subGRN_networkx(subgrn_df, cluster_id, figsize=(12, 8)):
    """
    Visualize subGRN using NetworkX
    """
    # Create directed graph
    G = nx.from_pandas_edgelist(
        subgrn_df, 
        source='source', 
        target='target', 
        edge_attr=['coef_mean', 'coef_abs', 'p', '-logp'],
        create_using=nx.DiGraph()
    )
    
    # Get unique nodes and categorize them
    all_sources = set(subgrn_df['source'])
    all_targets = set(subgrn_df['target'])
    tf_nodes = all_sources - all_targets  # TFs that don't appear as targets
    target_nodes = all_targets - all_sources  # Targets that don't appear as sources
    tf_target_nodes = all_sources & all_targets  # Nodes that are both TF and target
    
    # Set up the plot
    plt.figure(figsize=figsize)
    
    # Choose layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    # Alternative layouts:
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.circular_layout(G)
    
    # Draw nodes with different colors for TFs vs targets
    nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, 
                          node_color='lightcoral', node_size=800, 
                          label='Transcription Factors')
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, 
                          node_color='lightblue', node_size=600,
                          label='Target Genes')
    nx.draw_networkx_nodes(G, pos, nodelist=tf_target_nodes, 
                          node_color='orange', node_size=700,
                          label='TF & Target')
    
    # Draw edges with thickness proportional to coefficient strength
    edge_weights = [G[u][v]['coef_abs'] * 20 for u, v in G.edges()]  # Scale for visibility
    nx.draw_networkx_edges(G, pos, width=edge_weights, 
                          edge_color='gray', alpha=0.7,
                          arrowsize=20, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(f'Gene Regulatory Network - Cluster {cluster_id}\n({len(G.edges())} edges, {len(G.nodes())} nodes)')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G

# Usage
# G = visualize_subGRN_networkx(test_subGRN, "35_8")

def visualize_subGRN_networkx_styled(subgrn_df, cluster_id, figsize=(12, 8)):
    """
    Visualize subGRN using NetworkX - matching your coding style
    """
    # Construct a graph object in networkx and use the graph dictionary to add edges
    G = nx.DiGraph()  # Using DiGraph since GRN is directed
    
    # Add edges from the subGRN dataframe
    for _, row in subgrn_df.iterrows():
        if len(row['source']) >= 1 and len(row['target']) >= 1:  # Following your length check pattern
            G.add_edge(row['source'], row['target'], 
                      weight=round(row['coef_abs'], 4),  # Round like your style
                      coef_mean=round(row['coef_mean'], 4),
                      pvalue=row['p'])
    
    # Compute graph layout
    pos = nx.spring_layout(G)  # Use spring_layout for layout
    
    # Save the positions as node attributes (following your pattern)
    for node, position in pos.items():
        G.nodes[node]['pos'] = position
    
    # Categorize nodes for visualization
    all_sources = set(subgrn_df['source'])
    all_targets = set(subgrn_df['target'])
    tf_nodes = all_sources - all_targets
    target_nodes = all_targets - all_sources  
    tf_target_nodes = all_sources & all_targets
    
    # Set up the plot
    plt.figure(figsize=figsize)
    
    # Draw nodes with different colors
    nx.draw_networkx_nodes(G, pos, nodelist=list(tf_nodes), 
                          node_color='lightcoral', node_size=800, 
                          label='Transcription Factors')
    nx.draw_networkx_nodes(G, pos, nodelist=list(target_nodes), 
                          node_color='lightblue', node_size=600,
                          label='Target Genes')
    nx.draw_networkx_nodes(G, pos, nodelist=list(tf_target_nodes), 
                          node_color='orange', node_size=700,
                          label='TF & Target')
    
    # Draw edges with thickness proportional to weight
    edge_weights = []
    for u, v in G.edges():
        edge_weights.append(G[u][v]['weight'] * 20)  # Scale for visibility
    
    nx.draw_networkx_edges(G, pos, width=edge_weights, 
                          edge_color='gray', alpha=0.7,
                          arrowsize=20, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(f'Gene Regulatory Network - Cluster {cluster_id}\n({len(G.edges())} edges, {len(G.nodes())} nodes)')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G

# Usage following your style
# G = visualize_subGRN_networkx_styled(test_subGRN, "35_8")