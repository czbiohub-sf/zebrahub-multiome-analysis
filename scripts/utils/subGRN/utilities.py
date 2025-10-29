# Peak Cluster to GRN Pipeline - Utilities Module
# Author: YangJoon Kim
# Date: 2025-06-25
# Description: Helper functions and visualization utilities for sub-GRN analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import json
import os
from pathlib import Path
import warnings

# =============================================================================
# File I/O Utilities
# =============================================================================

def save_results(
    data: Any, 
    filepath: str, 
    format: str = "pickle"
):
    """
    Save analysis results in various formats.
    
    Parameters:
    -----------
    data : Any
        Data to save
    filepath : str
        Output file path
    format : str
        Format: "pickle", "json", "csv", "excel"
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == "json":
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath)
        else:
            raise ValueError("CSV format requires pandas DataFrame")
    elif format == "excel":
        if isinstance(data, pd.DataFrame):
            data.to_excel(filepath)
        elif isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
            with pd.ExcelWriter(filepath) as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name)
        else:
            raise ValueError("Excel format requires pandas DataFrame or dict of DataFrames")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved data to {filepath} in {format} format")

def load_results(
    filepath: str, 
    format: str = "pickle"
) -> Any:
    """
    Load analysis results from various formats.
    
    Parameters:
    -----------
    filepath : str
        Input file path
    format : str
        Format: "pickle", "json", "csv", "excel"
        
    Returns:
    --------
    Any
        Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == "pickle":
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    elif format == "json":
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif format == "csv":
        data = pd.read_csv(filepath, index_col=0)
    elif format == "excel":
        data = pd.read_excel(filepath, index_col=0)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Loaded data from {filepath}")
    return data

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_cluster_motif_heatmap(
    clusters_motifs_df: pd.DataFrame, 
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis',
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    selected_motifs: List[str] = None,
    title: str = "Cluster-Motif Enrichment Heatmap"
):
    """
    Plot heatmap of cluster-motif associations.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix
    output_path : str, optional
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap to use
    cluster_rows : bool
        Whether to cluster rows
    cluster_cols : bool
        Whether to cluster columns
    selected_motifs : List[str], optional
        Specific motifs to plot
    title : str
        Plot title
    """
    # Select subset of motifs if specified
    if selected_motifs:
        available_motifs = [m for m in selected_motifs if m in clusters_motifs_df.columns]
        if not available_motifs:
            warnings.warn("No selected motifs found in data")
            return
        plot_data = clusters_motifs_df[available_motifs]
    else:
        plot_data = clusters_motifs_df
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create clustermap
    g = sns.clustermap(
        plot_data,
        cmap=cmap,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        cbar_kws={'label': 'Enrichment Score'},
        figsize=figsize
    )
    
    # Set title
    g.fig.suptitle(title, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    
    plt.show()

def plot_tf_target_network(
    tf_target_matrix: pd.DataFrame, 
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 10),
    node_size_tf: int = 300,
    node_size_gene: int = 100,
    edge_width_scale: float = 3.0,
    layout: str = "spring",
    min_edge_weight: float = 0.1,
    max_nodes: int = 100
):
    """
    Plot network visualization of TF-target relationships.
    
    Parameters:
    -----------
    tf_target_matrix : pd.DataFrame
        TF-target matrix with edge weights
    output_path : str, optional
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    node_size_tf : int
        Size of TF nodes
    node_size_gene : int
        Size of gene nodes
    edge_width_scale : float
        Scaling factor for edge widths
    layout : str
        Network layout algorithm
    min_edge_weight : float
        Minimum edge weight to display
    max_nodes : int
        Maximum number of nodes to display
    """
    # Filter edges by minimum weight
    filtered_matrix = tf_target_matrix.copy()
    filtered_matrix[np.abs(filtered_matrix) < min_edge_weight] = 0
    
    # Remove empty nodes
    active_tfs = (filtered_matrix != 0).any(axis=1)
    active_genes = (filtered_matrix != 0).any(axis=0)
    filtered_matrix = filtered_matrix.loc[active_tfs, active_genes]
    
    # Limit number of nodes if necessary
    if len(filtered_matrix.index) + len(filtered_matrix.columns) > max_nodes:
        # Keep top TFs and genes by connectivity
        tf_connectivity = (filtered_matrix != 0).sum(axis=1).sort_values(ascending=False)
        gene_connectivity = (filtered_matrix != 0).sum(axis=0).sort_values(ascending=False)
        
        n_tfs = min(len(tf_connectivity), max_nodes // 2)
        n_genes = min(len(gene_connectivity), max_nodes - n_tfs)
        
        top_tfs = tf_connectivity.head(n_tfs).index
        top_genes = gene_connectivity.head(n_genes).index
        
        filtered_matrix = filtered_matrix.loc[top_tfs, top_genes]
        print(f"Limited network to {len(top_tfs)} TFs and {len(top_genes)} genes")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    tfs = list(filtered_matrix.index)
    genes = list(filtered_matrix.columns)
    
    G.add_nodes_from(tfs, node_type='TF')
    G.add_nodes_from(genes, node_type='gene')
    
    # Add edges
    for tf in tfs:
        for gene in genes:
            weight = filtered_matrix.loc[tf, gene]
            if weight != 0:
                G.add_edge(tf, gene, weight=abs(weight))
    
    if len(G.edges()) == 0:
        print("No edges to plot")
        return
    
    # Set up layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Define node colors and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'TF':
            node_colors.append('lightcoral')
            node_sizes.append(node_size_tf)
        else:
            node_colors.append('lightblue')
            node_sizes.append(node_size_gene)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=node_sizes,
        alpha=0.7
    )
    
    # Draw edges with varying widths
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [w * edge_width_scale / max_weight for w in weights]
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=10
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("TF-Target Gene Network")
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Network plot saved to {output_path}")
    
    plt.show()

def plot_grn_comparison(
    grn1: pd.DataFrame,
    grn2: pd.DataFrame,
    grn1_label: str = "GRN 1",
    grn2_label: str = "GRN 2",
    output_path: str = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Compare two GRNs side by side.
    
    Parameters:
    -----------
    grn1, grn2 : pd.DataFrame
        GRN matrices to compare
    grn1_label, grn2_label : str
        Labels for the GRNs
    output_path : str, optional
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Find common elements
    common_tfs = grn1.index.intersection(grn2.index)
    common_genes = grn1.columns.intersection(grn2.columns)
    
    if len(common_tfs) == 0 or len(common_genes) == 0:
        print("No common elements for comparison")
        return
    
    grn1_common = grn1.loc[common_tfs, common_genes]
    grn2_common = grn2.loc[common_tfs, common_genes]
    
    # Plot GRN 1
    im1 = axes[0].imshow(grn1_common.values, cmap='RdBu_r', aspect='auto')
    axes[0].set_title(grn1_label)
    axes[0].set_xlabel('Genes')
    axes[0].set_ylabel('TFs')
    
    # Plot GRN 2
    im2 = axes[1].imshow(grn2_common.values, cmap='RdBu_r', aspect='auto')
    axes[1].set_title(grn2_label)
    axes[1].set_xlabel('Genes')
    axes[1].set_ylabel('TFs')
    
    # Plot difference
    diff = grn1_common - grn2_common
    im3 = axes[2].imshow(diff.values, cmap='RdBu_r', aspect='auto')
    axes[2].set_title(f'{grn1_label} - {grn2_label}')
    axes[2].set_xlabel('Genes')
    axes[2].set_ylabel('TFs')
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0])
    plt.colorbar(im2, ax=axes[1])
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
    
    plt.show()

def plot_cluster_statistics(
    cluster_stats: pd.DataFrame,
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot summary statistics for clusters.
    
    Parameters:
    -----------
    cluster_stats : pd.DataFrame
        Statistics per cluster
    output_path : str, optional
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    """
    n_plots = len([col for col in cluster_stats.columns if col != 'cluster'])
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for col in cluster_stats.columns:
        if col == 'cluster':
            continue
        
        ax = axes[plot_idx]
        cluster_stats.plot(x='cluster', y=col, kind='bar', ax=ax)
        ax.set_title(f'{col} by Cluster')
        ax.set_xlabel('Cluster')
        ax.tick_params(axis='x', rotation=45)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to {output_path}")
    
    plt.show()

# =============================================================================
# Analysis Utilities
# =============================================================================

def compute_overlap_statistics(
    predicted_edges: List[Tuple], 
    actual_edges: List[Tuple]
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for predicted vs actual edges.
    
    Parameters:
    -----------
    predicted_edges : List[Tuple]
        List of predicted (TF, gene) edges
    actual_edges : List[Tuple]
        List of actual (TF, gene) edges
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with precision, recall, F1 scores
    """
    pred_set = set(predicted_edges)
    actual_set = set(actual_edges)
    
    # Calculate overlap
    true_positives = len(pred_set.intersection(actual_set))
    false_positives = len(pred_set - actual_set)
    false_negatives = len(actual_set - pred_set)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def extract_edges_from_matrix(
    matrix: pd.DataFrame,
    threshold: float = 0.0
) -> List[Tuple[str, str]]:
    """
    Extract edge list from adjacency matrix.
    
    Parameters:
    -----------
    matrix : pd.DataFrame
        Adjacency matrix (TFs x genes)
    threshold : float
        Minimum edge weight threshold
        
    Returns:
    --------
    List[Tuple[str, str]]
        List of (TF, gene) edges
    """
    edges = []
    for tf in matrix.index:
        for gene in matrix.columns:
            if abs(matrix.loc[tf, gene]) > threshold:
                edges.append((tf, gene))
    return edges

def compute_network_metrics(grn_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute network topology metrics for a GRN.
    
    Parameters:
    -----------
    grn_df : pd.DataFrame
        GRN matrix (TFs x genes)
        
    Returns:
    --------
    Dict[str, float]
        Network metrics
    """
    # Convert to NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    tfs = list(grn_df.index)
    genes = list(grn_df.columns)
    G.add_nodes_from(tfs, node_type='TF')
    G.add_nodes_from(genes, node_type='gene')
    
    # Add edges
    for tf in tfs:
        for gene in genes:
            weight = grn_df.loc[tf, gene]
            if weight != 0:
                G.add_edge(tf, gene, weight=abs(weight))
    
    # Compute metrics
    metrics = {}
    
    if len(G.nodes()) > 0:
        metrics['n_nodes'] = len(G.nodes())
        metrics['n_edges'] = len(G.edges())
        metrics['density'] = nx.density(G)
        
        if len(G.edges()) > 0:
            try:
                # Degree metrics
                in_degrees = [G.in_degree(n) for n in G.nodes()]
                out_degrees = [G.out_degree(n) for n in G.nodes()]
                
                metrics['mean_in_degree'] = np.mean(in_degrees)
                metrics['mean_out_degree'] = np.mean(out_degrees)
                metrics['max_in_degree'] = max(in_degrees)
                metrics['max_out_degree'] = max(out_degrees)
                
                # Connectivity
                if not nx.is_strongly_connected(G):
                    largest_cc = max(nx.weakly_connected_components(G), key=len)
                    metrics['largest_cc_size'] = len(largest_cc)
                    metrics['n_connected_components'] = nx.number_weakly_connected_components(G)
                else:
                    metrics['largest_cc_size'] = len(G.nodes())
                    metrics['n_connected_components'] = 1
                
            except Exception as e:
                warnings.warn(f"Error computing network metrics: {str(e)}")
    
    return metrics

def create_summary_report(
    results: Dict[str, Any],
    output_path: str
):
    """
    Create a summary report of the analysis results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Dictionary containing analysis results
    output_path : str
        Path to save the report
    """
    report_lines = []
    report_lines.append("# Peak Cluster to GRN Analysis Report")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Add timestamp
    from datetime import datetime
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Add sections based on available results
    if 'data_processing' in results:
        report_lines.append("## Data Processing")
        report_lines.append("-" * 20)
        dp = results['data_processing']
        if 'clusters_motifs_shape' in dp:
            report_lines.append(f"Clusters-motifs matrix shape: {dp['clusters_motifs_shape']}")
        if 'clusters_genes_shape' in dp:
            report_lines.append(f"Clusters-genes matrix shape: {dp['clusters_genes_shape']}")
        report_lines.append("")
    
    if 'differential_analysis' in results:
        report_lines.append("## Differential Analysis")
        report_lines.append("-" * 25)
        da = results['differential_analysis']
        if 'n_clusters' in da:
            report_lines.append(f"Number of clusters analyzed: {da['n_clusters']}")
        if 'total_differential_motifs' in da:
            report_lines.append(f"Total differential motifs found: {da['total_differential_motifs']}")
        report_lines.append("")
    
    if 'tf_target_construction' in results:
        report_lines.append("## TF-Target Construction")
        report_lines.append("-" * 28)
        ttc = results['tf_target_construction']
        if 'n_tf_target_matrices' in ttc:
            report_lines.append(f"TF-target matrices created: {ttc['n_tf_target_matrices']}")
        report_lines.append("")
    
    if 'grn_extraction' in results:
        report_lines.append("## GRN Extraction")
        report_lines.append("-" * 18)
        ge = results['grn_extraction']
        if 'n_subgrns' in ge:
            report_lines.append(f"Sub-GRNs extracted: {ge['n_subgrns']}")
        report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to {output_path}")

# =============================================================================
# Configuration Utilities
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    required_sections = ['input_data', 'differential_analysis', 'tf_target_construction', 'grn_extraction']
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required section: {section}")
            return False
    
    # Validate specific parameters
    input_data = config['input_data']
    if 'peaks_motifs_path' not in input_data or 'peaks_genes_path' not in input_data:
        print("Missing required input data paths")
        return False
    
    return True

def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary.
    
    Returns:
    --------
    Dict[str, Any]
        Default configuration
    """
    config = {
        'input_data': {
            'peaks_motifs_path': 'data/peaks_motifs.h5ad',
            'peaks_genes_path': 'data/peaks_genes.h5ad',
            'cluster_resolution': 'coarse'
        },
        'differential_analysis': {
            'method': 'top_n',
            'top_n': 10,
            'fold_change_threshold': 2.0,
            'pvalue_threshold': 0.001,
            'fdr_correction': True
        },
        'tf_target_construction': {
            'motif_database_path': None,
            'gene_association_method': 'correlation',
            'correlation_threshold': 0.5,
            'top_n_genes': None
        },
        'grn_extraction': {
            'grn_path': 'data/celloracle_grns/',
            'edge_strength_threshold': 0.1,
            'keep_only_putative': True
        },
        'output': {
            'results_dir': 'results/peak_cluster_grn_analysis/',
            'save_intermediate': True,
            'file_format': 'pickle'
        }
    }
    
    return config