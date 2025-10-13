"""
Cluster similarity analysis based on shared TFs and linked genes

This module provides functions for analyzing similarity between peak clusters
based on their enriched transcription factors and linked genes, identifying
dense similarity regions and characterizing TF sharing patterns.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional, Set, Union
import logging

logger = logging.getLogger(__name__)


def cluster_similarity_analysis(clust_by_motifs: pd.DataFrame,
                                threshold: float) -> Tuple[List[float], List[str]]:
    """
    Calculate pairwise Jaccard similarity between clusters based on motifs

    Parameters
    ----------
    clust_by_motifs : pd.DataFrame
        Motif enrichment scores (clusters × motifs)
    threshold : float
        Z-score threshold for calling motif enrichment significant

    Returns
    -------
    similarities : List[float]
        Pairwise Jaccard similarities (upper triangle only)
    cluster_names : List[str]
        Cluster pair names (e.g., "0_0-0_1")

    Examples
    --------
    >>> enrichment = pd.DataFrame(np.random.randn(10, 50) * 2)
    >>> similarities, names = cluster_similarity_analysis(enrichment, threshold=2.0)
    >>> print(f"Mean similarity: {np.mean(similarities):.3f}")
    """
    # Get binary matrix of significant motifs
    sig_matrix = (clust_by_motifs >= threshold).astype(int)

    # Calculate Jaccard similarity between clusters
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


def analyze_tf_sharing(cluster_tf_matrix: pd.DataFrame,
                       savefig: bool = False,
                       filename: str = "dist_TFs_across_peak_clusts.pdf") -> Tuple[pd.Series, pd.Series]:
    """
    Analyze how TFs are shared across clusters

    Parameters
    ----------
    cluster_tf_matrix : pd.DataFrame
        Binary matrix (clusters × TFs)
    savefig : bool, default=False
        Whether to save figure
    filename : str
        Output filename for figure

    Returns
    -------
    tf_cluster_counts : pd.Series
        Number of clusters each TF appears in
    most_shared : pd.Series
        Top 20 most shared TFs

    Examples
    --------
    >>> matrix = pd.DataFrame(np.random.randint(0, 2, (100, 50)))
    >>> counts, top = analyze_tf_sharing(matrix)
    >>> print(f"Most shared TF appears in {counts.max()} clusters")
    """
    logger.info("=== TF SHARING ANALYSIS ===")

    # TF frequency across clusters
    tf_cluster_counts = cluster_tf_matrix.sum(axis=0)  # How many clusters each TF appears in

    # Cluster statistics
    logger.info("TF sharing statistics:")
    logger.info(f"  TFs appearing in only 1 cluster: {(tf_cluster_counts == 1).sum()}")
    logger.info(f"  TFs appearing in 2-5 clusters: {((tf_cluster_counts >= 2) & (tf_cluster_counts <= 5)).sum()}")
    logger.info(f"  TFs appearing in 6-10 clusters: {((tf_cluster_counts >= 6) & (tf_cluster_counts <= 10)).sum()}")
    logger.info(f"  TFs appearing in >10 clusters: {(tf_cluster_counts > 10).sum()}")

    # Most shared TFs
    most_shared = tf_cluster_counts.nlargest(20)
    logger.info("Top 20 most shared TFs:")
    for tf, count in most_shared.items():
        logger.info(f"  {tf}: {count} clusters")

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
        plt.savefig(filename)
    plt.show()

    return tf_cluster_counts, most_shared


def create_cluster_similarity_heatmap(cluster_feature_matrix: pd.DataFrame,
                                      top_n_clusters: int = 50,
                                      feature_type: str = "TFs",
                                      savefig: bool = False,
                                      filename: Optional[str] = None,
                                      linkage_info: Optional[Dict] = None,
                                      return_linkage: bool = True,
                                      hide_axis_labels: bool = True,
                                      similarity_cutoff: float = 0.80,
                                      min_box_size: int = 3,
                                      highlight_blocks: bool = True,
                                      return_block_clusters: bool = True) -> Tuple:
    """
    Create heatmap of cluster-to-cluster Jaccard similarity with optional block detection

    Parameters
    ----------
    cluster_feature_matrix : pd.DataFrame
        Binary matrix (clusters × features)
    top_n_clusters : int, default=50
        Number of top clusters to include
    feature_type : str, default="TFs"
        Type of features (for labels)
    savefig : bool, default=False
        Whether to save figure
    filename : str, optional
        Output filename
    linkage_info : dict, optional
        Pre-computed linkage information
    return_linkage : bool, default=True
        Whether to return linkage information
    hide_axis_labels : bool, default=True
        Whether to hide axis tick labels
    similarity_cutoff : float, default=0.80
        Jaccard threshold for block detection
    min_box_size : int, default=3
        Minimum cluster count for block
    highlight_blocks : bool, default=True
        Whether to draw rectangles around blocks
    return_block_clusters : bool, default=True
        Whether to return cluster lists per block

    Returns
    -------
    similarity : np.ndarray
        Similarity matrix
    clusters : List[str]
        Cluster IDs
    linkage_info : dict (optional)
        Linkage matrix and ordering
    block_lists : List[List[str]] (optional)
        Cluster IDs in each block

    Examples
    --------
    >>> matrix = pd.DataFrame(np.random.randint(0, 2, (100, 50)))
    >>> sim, clusters, linkage, blocks = create_cluster_similarity_heatmap(
    ...     matrix, top_n_clusters=50
    ... )
    >>> print(f"Found {len(blocks)} similarity blocks")
    """
    logger.info(f"=== CLUSTER SIMILARITY HEATMAP ({feature_type}) ===")

    # Down-sample clusters if needed
    if len(cluster_feature_matrix) > top_n_clusters:
        counts = cluster_feature_matrix.sum(axis=1)
        matrix_subset = cluster_feature_matrix.loc[counts.nlargest(top_n_clusters).index]
        logger.info(f"Using top {top_n_clusters} clusters by {feature_type} count")
    else:
        matrix_subset = cluster_feature_matrix
        logger.info(f"Using all {len(cluster_feature_matrix)} clusters")

    # Calculate pairwise Jaccard similarity
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

    # Hierarchical ordering
    if linkage_info is None:
        logger.info("Computing new hierarchical clustering")
        dist_vec = squareform(1 - similarity, checks=False)
        linkage_matrix = linkage(dist_vec, method='average')
        order = dendrogram(linkage_matrix, no_plot=True)['leaves']
    else:
        logger.info("Using provided linkage information")
        linkage_matrix = linkage_info['linkage_matrix']
        ref_names = linkage_info['cluster_names']
        order = [ref_names.index(c) for c in clusters]

    sim_ord = similarity[np.ix_(order, order)]
    name_ord = [clusters[i] for i in order]

    # Plot
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(sim_ord, cmap='Blues', vmin=0, vmax=1,
                     xticklabels=name_ord, yticklabels=name_ord,
                     square=True, cbar_kws={'label': 'Jaccard Similarity'})

    # Remove axis labels if requested
    if hide_axis_labels:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Highlight dense blocks
    block_lists = []
    if highlight_blocks or return_block_clusters:
        labels = fcluster(linkage_matrix, t=1 - similarity_cutoff, criterion='distance')
        groups = {}
        for idx, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(idx)

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


def analyze_similarity_distribution(cluster_tf_matrix: pd.DataFrame) -> Tuple[Dict, float, float]:
    """
    Analyze similarity distribution to set realistic thresholds

    Parameters
    ----------
    cluster_tf_matrix : pd.DataFrame
        Binary cluster-by-TF matrix

    Returns
    -------
    stats : dict
        Distribution statistics (mean, median, percentiles)
    recommended_primary : float
        Recommended primary cutoff
    recommended_internal : float
        Recommended internal similarity minimum

    Examples
    --------
    >>> matrix = pd.DataFrame(np.random.randint(0, 2, (100, 50)))
    >>> stats, primary, internal = analyze_similarity_distribution(matrix)
    >>> print(f"Recommended primary cutoff: {primary:.3f}")
    """
    logger.info("=== ANALYZING SIMILARITY DISTRIBUTION ===")

    # Calculate similarity matrix
    matrix_subset = cluster_tf_matrix
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

    # Get upper triangle (excluding diagonal)
    upper_triangle = similarity[np.triu_indices_from(similarity, k=1)]

    # Calculate statistics
    stats = {
        'mean': np.mean(upper_triangle),
        'median': np.median(upper_triangle),
        'std': np.std(upper_triangle),
        'p75': np.percentile(upper_triangle, 75),
        'p90': np.percentile(upper_triangle, 90),
        'p95': np.percentile(upper_triangle, 95),
        'p99': np.percentile(upper_triangle, 99),
        'max': np.max(upper_triangle)
    }

    logger.info("Similarity Statistics:")
    logger.info(f"  Mean: {stats['mean']:.3f}")
    logger.info(f"  Median: {stats['median']:.3f}")
    logger.info(f"  75th percentile: {stats['p75']:.3f}")
    logger.info(f"  90th percentile: {stats['p90']:.3f}")
    logger.info(f"  95th percentile: {stats['p95']:.3f}")
    logger.info(f"  99th percentile: {stats['p99']:.3f}")
    logger.info(f"  Maximum: {stats['max']:.3f}")

    # Plot distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(upper_triangle, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.3f}')
    ax1.axvline(stats['p90'], color='orange', linestyle='--', label=f'90th %ile: {stats["p90"]:.3f}')
    ax1.axvline(stats['p95'], color='green', linestyle='--', label=f'95th %ile: {stats["p95"]:.3f}')
    ax1.set_xlabel('Jaccard Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Pairwise Similarities')
    ax1.legend()

    # Cumulative distribution
    sorted_sims = np.sort(upper_triangle)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax2.plot(sorted_sims, cumulative)
    ax2.axvline(stats['p90'], color='orange', linestyle='--', label=f'90th %ile: {stats["p90"]:.3f}')
    ax2.axvline(stats['p95'], color='green', linestyle='--', label=f'95th %ile: {stats["p95"]:.3f}')
    ax2.set_xlabel('Jaccard Similarity')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Recommend thresholds
    recommended_internal = max(stats['p90'], stats['mean'] + stats['std'])
    recommended_primary = max(stats['p75'], stats['mean'])

    logger.info("=== RECOMMENDED THRESHOLDS ===")
    logger.info(f"Primary cutoff: {recommended_primary:.3f} (for initial detection)")
    logger.info(f"Internal similarity minimum: {recommended_internal:.3f} (for quality filtering)")

    return stats, recommended_primary, recommended_internal


def find_dense_similarity_regions(cluster_feature_matrix: pd.DataFrame,
                                   top_n_clusters: int = 50,
                                   feature_type: str = "TFs",
                                   savefig: bool = False,
                                   filename: Optional[str] = None,
                                   min_similarity_threshold: float = 0.15,
                                   average_similarity_threshold: Optional[float] = None,
                                   min_block_size: int = 4,
                                   max_block_size: int = 30,
                                   hide_axis_labels: bool = True,
                                   cmap: Union[str, mcolors.Colormap] = "Blues",
                                   gamma: Optional[float] = None,
                                   show_blocks: bool = True) -> Tuple:
    """
    Find dense similarity regions using seed-and-grow approach

    Parameters
    ----------
    cluster_feature_matrix : pd.DataFrame
        Binary cluster-by-feature matrix
    top_n_clusters : int, default=50
        Number of top clusters to analyze
    feature_type : str, default="TFs"
        Type of features (for labels)
    savefig : bool, default=False
        Whether to save figure
    filename : str, optional
        Output filename
    min_similarity_threshold : float, default=0.15
        Minimum pairwise similarity within block
    average_similarity_threshold : float, optional
        Minimum average similarity for block quality filtering
    min_block_size : int, default=4
        Minimum clusters per block
    max_block_size : int, default=30
        Maximum clusters per block
    hide_axis_labels : bool, default=True
        Whether to hide axis labels
    cmap : str or Colormap, default="Blues"
        Colormap for heatmap
    gamma : float, optional
        Gamma correction for colormap
    show_blocks : bool, default=True
        Whether to draw block rectangles

    Returns
    -------
    sim_ord : np.ndarray
        Ordered similarity matrix
    name_ord : List[str]
        Ordered cluster names
    linkage_info : dict
        Linkage information
    dense_block_clusters : List[List[str]]
        Cluster IDs per block
    dense_blocks : List[dict]
        Block details with metrics

    Examples
    --------
    >>> matrix = pd.DataFrame(np.random.randint(0, 2, (100, 50)))
    >>> sim, names, link, clusters, details = find_dense_similarity_regions(
    ...     matrix, min_similarity_threshold=0.20
    ... )
    >>> print(f"Found {len(clusters)} dense regions")
    """
    logger.info(f"=== DENSE REGION DETECTION ({feature_type}) ===")

    # Preprocessing
    if len(cluster_feature_matrix) > top_n_clusters:
        counts = cluster_feature_matrix.sum(axis=1)
        matrix_subset = cluster_feature_matrix.loc[counts.nlargest(top_n_clusters).index]
        logger.info(f"Using top {top_n_clusters} clusters by {feature_type} count")
    else:
        matrix_subset = cluster_feature_matrix
        logger.info(f"Using all {len(cluster_feature_matrix)} clusters")

    # Calculate Jaccard similarity
    clusters = matrix_subset.index.tolist()
    n = len(clusters)
    similarity = np.eye(n)

    logger.info("Computing pairwise similarities...")
    for i in range(n):
        set_i = set(matrix_subset.columns[matrix_subset.iloc[i] == 1])
        for j in range(i + 1, n):
            set_j = set(matrix_subset.columns[matrix_subset.iloc[j] == 1])
            denom = len(set_i | set_j)
            sim = 0 if denom == 0 else len(set_i & set_j) / denom
            similarity[i, j] = similarity[j, i] = sim

    # Hierarchical ordering for visualization
    dist_vec = squareform(1 - similarity, checks=False)
    linkage_matrix = linkage(dist_vec, method='average')
    order = dendrogram(linkage_matrix, no_plot=True)['leaves']

    sim_ord = similarity[np.ix_(order, order)]
    name_ord = [clusters[i] for i in order]

    logger.info("Detection parameters:")
    logger.info(f"  Minimum similarity threshold: {min_similarity_threshold:.3f}")
    if average_similarity_threshold is not None:
        logger.info(f"  Average similarity threshold: {average_similarity_threshold:.3f}")
    else:
        logger.info("  Average similarity threshold: None (no filtering)")
    logger.info(f"  Block size range: {min_block_size}-{max_block_size}")
    logger.info(f"  Colormap: {cmap if isinstance(cmap, str) else 'Custom'}")
    if gamma is not None:
        logger.info(f"  Gamma correction: {gamma}")
    else:
        logger.info("  Gamma correction: None")

    # Seed-and-grow approach
    logger.info("Using seed-and-grow approach...")

    used_clusters = set()
    dense_blocks = []

    # Start with highest similarity pairs as seeds
    similarity_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= min_similarity_threshold:
                similarity_pairs.append((similarity[i, j], i, j))

    similarity_pairs.sort(reverse=True)

    logger.info(f"Found {len(similarity_pairs)} pairs above threshold {min_similarity_threshold:.3f}")

    for sim_value, seed_i, seed_j in similarity_pairs:
        if seed_i in used_clusters or seed_j in used_clusters:
            continue

        # Start growing a block
        current_block = {seed_i, seed_j}

        # Keep adding clusters
        improved = True
        while improved and len(current_block) < max_block_size:
            improved = False
            best_candidate = None
            best_min_sim = 0

            for candidate in range(n):
                if candidate in current_block or candidate in used_clusters:
                    continue

                min_sim_to_block = min(similarity[candidate, block_member]
                                      for block_member in current_block)

                if min_sim_to_block >= min_similarity_threshold and min_sim_to_block > best_min_sim:
                    best_candidate = candidate
                    best_min_sim = min_sim_to_block

            if best_candidate is not None:
                current_block.add(best_candidate)
                improved = True

        # Accept block if meets size requirements
        if len(current_block) >= min_block_size:
            all_pairs_valid = True
            internal_similarities = []

            for i in current_block:
                for j in current_block:
                    if i != j:
                        sim_ij = similarity[i, j]
                        internal_similarities.append(sim_ij)
                        if sim_ij < min_similarity_threshold:
                            all_pairs_valid = False
                            break
                if not all_pairs_valid:
                    break

            if all_pairs_valid:
                block_clusters = [clusters[i] for i in current_block]
                avg_sim = np.mean(internal_similarities)
                min_sim = np.min(internal_similarities)

                dense_blocks.append({
                    'indices': list(current_block),
                    'clusters': block_clusters,
                    'avg_similarity': avg_sim,
                    'min_similarity': min_sim,
                    'size': len(current_block)
                })

                used_clusters.update(current_block)
                logger.info(f"  ✓ Found block: {len(current_block)} clusters, avg_sim={avg_sim:.3f}, min_sim={min_sim:.3f}")
            else:
                logger.info("  ✗ Block validation failed: not all pairs meet threshold")

    logger.info(f"Seed-and-grow result: {len(dense_blocks)} dense blocks")

    # Apply average similarity filtering if threshold provided
    if average_similarity_threshold is not None:
        logger.info(f"Applying average similarity filter (≥ {average_similarity_threshold:.3f})...")

        high_quality_blocks = []
        rejected_blocks = []

        for i, block in enumerate(dense_blocks):
            avg_sim = block['avg_similarity']

            if avg_sim >= average_similarity_threshold:
                high_quality_blocks.append(block)
                logger.info(f"  ✓ Block {i+1}: KEPT - size={block['size']}, avg_sim={avg_sim:.3f}")
            else:
                rejected_blocks.append(block)
                logger.info(f"  ✗ Block {i+1}: REJECTED - size={block['size']}, avg_sim={avg_sim:.3f} < {average_similarity_threshold:.3f}")

        logger.info("Filtering summary:")
        logger.info(f"  Original blocks: {len(dense_blocks)}")
        logger.info(f"  High-quality blocks: {len(high_quality_blocks)}")
        logger.info(f"  Rejected blocks: {len(rejected_blocks)}")

        dense_blocks = high_quality_blocks

    # Convert to ordered positions
    ordered_blocks = []
    for block in dense_blocks:
        ordered_indices = []
        for orig_idx in block['indices']:
            cluster_name = clusters[orig_idx]
            if cluster_name in name_ord:
                ordered_pos = name_ord.index(cluster_name)
                ordered_indices.append(ordered_pos)

        if ordered_indices:
            ordered_blocks.append({
                **block,
                'ordered_indices': ordered_indices
            })

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 12))

    # Apply gamma correction if specified
    if gamma is not None and isinstance(cmap, str):
        logger.info(f"Applying gamma correction (γ={gamma}) to {cmap} colormap")
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, 256))
        colors_gamma = colors.copy()
        colors_gamma[:, :3] = np.power(colors[:, :3], gamma)
        plot_cmap = mcolors.ListedColormap(colors_gamma)
    else:
        plot_cmap = cmap

    # Plot heatmap
    im = ax.imshow(sim_ord, cmap=plot_cmap, aspect='auto', vmin=0, vmax=1)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Jaccard Similarity', rotation=270, labelpad=20, fontsize=12)

    # Draw dense blocks
    colors = ['darkred', 'gold', 'green', 'purple', 'indigo', 'pink', 'cyan', 'lime', 'magenta', 'yellow']
    if show_blocks:
        for i, block in enumerate(ordered_blocks):
            indices = block['ordered_indices']
            if not indices:
                continue

            min_pos = min(indices)
            max_pos = max(indices)
            size = max_pos - min_pos + 1

            color = colors[i % len(colors)]

            rect = Rectangle((min_pos, min_pos), size, size,
                            linewidth=3, edgecolor=color, facecolor='none', alpha=0.9)
            ax.add_patch(rect)

            avg_sim = block['avg_similarity']
            min_sim = block['min_similarity']
            block_size = block['size']

            label_prefix = 'HQ' if average_similarity_threshold is not None else 'B'
            label_text = f'{label_prefix}{i+1}\n({block_size})\navg:{avg_sim:.3f}\nmin:{min_sim:.3f}'
            ax.text(min_pos + size/2, min_pos - 25, label_text,
                   ha='center', va='bottom', fontweight='bold',
                   color=color, fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

            logger.info(f"Block {i+1} visualization: positions {min_pos}-{max_pos}, actual size {block_size}")

    # Title
    if average_similarity_threshold is not None:
        title = f'High-Quality Dense Similarity Regions ({feature_type})\n' \
                f'{len(ordered_blocks)} blocks (all pairs ≥ {min_similarity_threshold:.3f}, avg ≥ {average_similarity_threshold:.3f})'
    else:
        title = f'Dense Similarity Regions ({feature_type})\n' \
                f'{len(ordered_blocks)} blocks (all pairs ≥ {min_similarity_threshold:.3f})'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)

    if hide_axis_labels:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()

    if savefig and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

    # Return results
    dense_block_clusters = [block['clusters'] for block in dense_blocks]
    linkage_info = {
        'linkage_matrix': linkage_matrix,
        'cluster_order': order,
        'cluster_names': clusters
    }

    return sim_ord, name_ord, linkage_info, dense_block_clusters, dense_blocks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Module: subgrn_similarity_analysis.py")
    print("Contains functions for cluster similarity analysis and dense region detection")
