import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from typing import Dict, List, Tuple, Optional, Union, Mapping, Hashable, Sequence


def cluster_similarity_analysis(clust_by_motifs, threshold):
    """
    Analyze similarity between clusters based on shared motifs using Jaccard similarity
    
    Parameters:
    -----------
    clust_by_motifs : pd.DataFrame
        Clusters x motifs matrix with enrichment scores
    threshold : float
        Threshold for considering motifs as significant
        
    Returns:
    --------
    tuple
        (similarities, cluster_names) - Jaccard similarities and cluster pair names
    """
    # Get binary matrix of significant motifs
    sig_matrix = (clust_by_motifs >= threshold).astype(int)
    
    # Calculate Jaccard similarity between clusters
    # (intersection / union)
    similarities = []
    cluster_names = []
    
    for i, cluster1 in enumerate(sig_matrix.index):
        for j, cluster2 in enumerate(sig_matrix.index):
            if i < j:  # Only upper triangle
                motifs1 = set(sig_matrix.columns[sig_matrix.loc[cluster1] == 1])
                motifs2 = set(sig_matrix.columns[sig_matrix.loc[cluster2] == 1])
                
                if len(motifs1) == 0 and len(motifs2) == 0:
                    jaccard = 1.0  # Both empty
                elif len(motifs1 | motifs2) == 0:
                    jaccard = 0.0
                else:
                    jaccard = len(motifs1 & motifs2) / len(motifs1 | motifs2)
                
                similarities.append(jaccard)
                cluster_names.append(f"{cluster1}-{cluster2}")
    
    return similarities, cluster_names


def cluster_dict_to_df(d: Mapping[Hashable, Sequence], col_name: str) -> pd.DataFrame:
    """
    Convert a dictionary of the form {cluster_id: list_of_items}
    into a DataFrame whose index is the cluster IDs and whose single
    column contains the length of each list.

    Parameters:
    -----------
    d : dict
        Keys are cluster identifiers (e.g. "35_8"); values are list-like
        collections (e.g. TF names or genes).
    col_name : str
        Name for the output column.

    Returns:
    --------
    pd.DataFrame
        Index = dictionary keys; one column with the list lengths.
    """
    lengths = {k: len(v) for k, v in d.items()}
    return pd.DataFrame.from_dict(lengths, orient="index", columns=[col_name])


def build_master_df(
    dict_map: Mapping[str, Mapping[Hashable, Sequence]],
    *,
    prefix: str = "n_",
    fill_value: int = 0
) -> pd.DataFrame:
    """
    Build a master DataFrame combining multiple cluster dictionaries
    
    Parameters:
    -----------
    dict_map : dict
        Keys = short labels ("tfs", "linked_genes", …);
        Values = the actual {cluster → list} dictionaries.
    prefix : str
        Prepended to each column name (default "n_" → "n_tfs", …).
    fill_value : int
        Value used to fill clusters that are missing from some dictionaries.
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all cluster statistics
    """
    dfs = [
        cluster_dict_to_df(d, f"{prefix}{label}")
        for label, d in dict_map.items()
    ]
    master = pd.concat(dfs, axis=1)          # outer join on the index (cluster IDs)
    return master.fillna(fill_value).astype(int)


def create_cluster_tf_matrix(clusters_tfs_dict):
    """
    Create a binary clusters-by-TFs matrix
    
    Parameters:
    -----------
    clusters_tfs_dict : dict
        Dictionary mapping cluster_id -> list of TFs
        
    Returns:
    --------
    tuple
        (cluster_tf_matrix, all_tfs) - Binary matrix and list of all unique TFs
    """
    print("Creating clusters-by-TFs matrix...")
    
    # Get all unique TFs across all clusters
    all_tfs = set()
    for tfs in clusters_tfs_dict.values():
        all_tfs.update(tfs)
    
    all_tfs = sorted(list(all_tfs))  # Sort for consistency
    cluster_ids = sorted(list(clusters_tfs_dict.keys()))
    
    print(f"Total unique TFs across all clusters: {len(all_tfs)}")
    print(f"Total clusters: {len(cluster_ids)}")
    
    # Create binary matrix
    cluster_tf_matrix = pd.DataFrame(0, index=cluster_ids, columns=all_tfs)
    
    for cluster_id, tfs in clusters_tfs_dict.items():
        for tf in tfs:
            cluster_tf_matrix.loc[cluster_id, tf] = 1
    
    return cluster_tf_matrix, all_tfs


def analyze_tf_sharing(cluster_tf_matrix, figpath="", savefig=False, filename="dist_TFs_across_peak_clusts.pdf"):
    """
    Analyze how TFs are shared across clusters
    
    Parameters:
    -----------
    cluster_tf_matrix : pd.DataFrame
        Binary clusters-by-TFs matrix
    figpath : str
        Path for saving figures
    savefig : bool
        Whether to save the figure
    filename : str
        Filename for saved figure
        
    Returns:
    --------
    tuple
        (tf_cluster_counts, most_shared) - TF sharing statistics and most shared TFs
    """
    print("\n=== TF SHARING ANALYSIS ===")
    
    # TF frequency across clusters
    tf_cluster_counts = cluster_tf_matrix.sum(axis=0)  # How many clusters each TF appears in
    
    # Cluster statistics
    print(f"TF sharing statistics:")
    print(f"  TFs appearing in only 1 cluster: {(tf_cluster_counts == 1).sum()}")
    print(f"  TFs appearing in 2-5 clusters: {((tf_cluster_counts >= 2) & (tf_cluster_counts <= 5)).sum()}")
    print(f"  TFs appearing in 6-10 clusters: {((tf_cluster_counts >= 6) & (tf_cluster_counts <= 10)).sum()}")
    print(f"  TFs appearing in >10 clusters: {(tf_cluster_counts > 10).sum()}")
    
    # Most shared TFs
    most_shared = tf_cluster_counts.nlargest(20)
    print(f"\nTop 20 most shared TFs:")
    for tf, count in most_shared.items():
        print(f"  {tf}: {count} clusters")
    
    # Plot TF sharing distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(tf_cluster_counts, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Clusters Containing TF')
    plt.ylabel('Number of TFs')
    plt.title('Distribution of TF Sharing Across Clusters')
    plt.grid(False)
    
    plt.subplot(1, 2, 2)
    # Show cumulative distribution
    sorted_counts = np.sort(tf_cluster_counts)[::-1]  # Descending order
    plt.plot(range(1, len(sorted_counts) + 1), sorted_counts)
    plt.xlabel('TF Rank')
    plt.ylabel('Number of Clusters')
    plt.title('TF Sharing: Rank vs Frequency')
    plt.grid(False)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(figpath + filename)
    plt.show()
    
    return tf_cluster_counts, most_shared


def create_cluster_similarity_heatmap(cluster_feature_matrix,
                                      top_n_clusters=50,
                                      feature_type="TFs",
                                      savefig=False,
                                      filename=None,
                                      linkage_info=None,
                                      return_linkage=True,
                                      hide_axis_labels=True,
                                      similarity_cutoff=0.80,
                                      min_box_size=3,
                                      highlight_blocks=True,
                                      return_block_clusters=True):
    """
    Create a heatmap of cluster-to-cluster Jaccard similarity with optional block highlighting
    
    Parameters:
    -----------
    cluster_feature_matrix : pd.DataFrame
        Binary matrix of clusters x features (TFs/genes)
    top_n_clusters : int
        Maximum number of clusters to include
    feature_type : str
        Type of features being analyzed ("TFs", "genes", etc.)
    savefig : bool
        Whether to save the figure
    filename : str, optional
        Filename for saved figure
    linkage_info : dict, optional
        Pre-computed linkage information for consistent ordering
    return_linkage : bool
        Whether to return linkage information
    hide_axis_labels : bool
        Whether to hide messy axis labels
    similarity_cutoff : float
        Minimum similarity for defining blocks
    min_box_size : int
        Minimum size for highlighting blocks
    highlight_blocks : bool
        Whether to highlight similarity blocks
    return_block_clusters : bool
        Whether to return lists of clusters in each block
        
    Returns:
    --------
    tuple
        Various outputs based on return parameters: (similarity, clusters, [linkage_info], [block_lists])
    """
    print(f"\n=== CLUSTER SIMILARITY HEATMAP ({feature_type}) ===")

    # Down-sample clusters if needed
    if len(cluster_feature_matrix) > top_n_clusters:
        counts = cluster_feature_matrix.sum(axis=1)
        matrix_subset = cluster_feature_matrix.loc[counts.nlargest(top_n_clusters).index]
        print(f"Using top {top_n_clusters} clusters by {feature_type} count")
    else:
        matrix_subset = cluster_feature_matrix
        print(f"Using all {len(cluster_feature_matrix)} clusters")

    # Compute pairwise Jaccard similarity
    clusters = matrix_subset.index.tolist()
    n = len(clusters)
    similarity = np.eye(n)

    for i in range(n):
        set_i = set(matrix_subset.columns[matrix_subset.iloc[i] == 1])
        for j in range(i + 1, n):
            set_j = set(matrix_subset.columns[matrix_subset.iloc[j] == 1])
            denom = len(set_i | set_j)
            sim = 0 if denom == 0 else len(set_i & set_j) / denom
            similarity[i, j] = similarity[j, i] = sim

    # Hierarchical ordering (reuse if given)
    if linkage_info is None:
        print("Computing new hierarchical clustering")
        dist_vec = squareform(1 - similarity, checks=False)
        linkage_matrix = linkage(dist_vec, method='average')
        order = dendrogram(linkage_matrix, no_plot=True)['leaves']
    else:
        print("Using provided linkage information")
        linkage_matrix = linkage_info['linkage_matrix']
        ref_names = linkage_info['cluster_names']
        order = [ref_names.index(c) for c in clusters]  # map subset to reference order

    sim_ord = similarity[np.ix_(order, order)]
    name_ord = [clusters[i] for i in order]

    # Create plot
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(sim_ord, cmap='Blues', vmin=0, vmax=1,
                     xticklabels=name_ord, yticklabels=name_ord,
                     square=True, cbar_kws={'label': 'Jaccard Similarity'})

    # Remove messy tick labels if requested
    if hide_axis_labels:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Highlight dense blocks & list their members
    block_lists = []
    if highlight_blocks or return_block_clusters:
        # Cut the dendrogram at distance = 1-similarity_cutoff
        labels = fcluster(linkage_matrix, t=1 - similarity_cutoff, criterion='distance')
        groups = {}
        for idx, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(idx)

        # Keep only "big" blocks
        dense_groups = [idxs for idxs in groups.values() if len(idxs) >= min_box_size]

        for g in dense_groups:
            i0, i1 = min(g), max(g)
            size = i1 - i0 + 1
            block_lists.append([name_ord[i] for i in g])

            if highlight_blocks:
                ax.add_patch(Rectangle((i0, i0), size, size,
                                     edgecolor='red', linewidth=2, fill=False))

    title = f'Cluster Similarity Based on Shared {feature_type}\n' \
            f'({"Hierarchically Clustered" if linkage_info is None else "Consistent Ordering"})'
    plt.title(title, pad=20)
    plt.tight_layout()

    if savefig and filename:
        plt.savefig(filename)
    plt.show()

    # Return values
    out = [similarity, clusters]
    if return_linkage:
        out.append({'linkage_matrix': linkage_matrix,
                    'cluster_order': order,
                    'cluster_names': clusters})
    if return_block_clusters:
        out.append(block_lists)

    return tuple(out)


def create_scatter_with_histograms(df_master, x_col, y_col, figsize=(6, 6)):
    """
    Create scatter plot with marginal histograms for cluster analysis
    
    Parameters:
    -----------
    df_master : pd.DataFrame
        Master DataFrame with cluster statistics
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    x = df_master[x_col].values
    y = df_master[y_col].values

    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2, 
        width_ratios=[4, 1], 
        height_ratios=[1, 4],
        hspace=0.05, 
        wspace=0.05
    )

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.scatter(x, y, alpha=0.6, s=30)
    ax_main.set_xlabel(x_col.replace("_", " ").title())
    ax_main.set_ylabel(y_col.replace("_", " ").title())
    ax_main.grid(True, alpha=0.3)

    # Top histogram (x-axis marginal)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.hist(x, bins=20, alpha=0.7, edgecolor='black')
    ax_top.set_ylabel('Count')
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Right histogram (y-axis marginal)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.hist(y, bins=20, alpha=0.7, edgecolor='black', orientation='horizontal')
    ax_right.set_xlabel('Count')
    plt.setp(ax_right.get_yticklabels(), visible=False)

    plt.tight_layout()
    return fig


def analyze_cluster_relationships(df_master, feature_pairs=None):
    """
    Analyze relationships between different cluster features
    
    Parameters:
    -----------
    df_master : pd.DataFrame
        Master DataFrame with cluster statistics
    feature_pairs : list of tuples, optional
        Pairs of features to analyze. If None, uses default pairs.
        
    Returns:
    --------
    dict
        Analysis results including correlations and statistics
    """
    if feature_pairs is None:
        feature_pairs = [
            ("n_tfs", "n_linked_genes"),
            ("n_tfs", "mesh_size") if "mesh_size" in df_master.columns else ("n_tfs", "n_linked_genes")
        ]
    
    results = {}
    
    for x_col, y_col in feature_pairs:
        if x_col in df_master.columns and y_col in df_master.columns:
            x = df_master[x_col]
            y = df_master[y_col]
            
            # Calculate correlation
            correlation = x.corr(y)
            
            # Basic statistics
            stats = {
                'correlation': correlation,
                'x_mean': x.mean(),
                'x_std': x.std(),
                'y_mean': y.mean(),
                'y_std': y.std(),
                'n_points': len(x)
            }
            
            results[f"{x_col}_vs_{y_col}"] = stats
            
            print(f"\n{x_col} vs {y_col}:")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  {x_col}: {x.mean():.1f} ± {x.std():.1f}")
            print(f"  {y_col}: {y.mean():.1f} ± {y.std():.1f}")
    
    return results 