"""
Cluster specificity analysis for peak accessibility

This module provides functions for calculating and analyzing peak cluster
specificity based on chromatin accessibility patterns across cell types
and developmental timepoints.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_cluster_specificity(df_clusters_groups: pd.DataFrame,
                                  top_n: int = 2) -> pd.DataFrame:
    """
    Calculate specificity metrics for each cluster to identify those with
    strong signal in only 1-2 celltypes/timepoints

    Parameters
    ----------
    df_clusters_groups : pd.DataFrame
        Clusters (rows) by pseudobulk groups (columns) accessibility matrix
    top_n : int, default=2
        Number of top groups to consider

    Returns
    -------
    pd.DataFrame
        Specificity metrics for each cluster with columns:
        - top1_group: Most accessible group
        - top1_value: Accessibility value
        - top2_group: Second most accessible group
        - top2_value: Second accessibility value
        - specificity_score: Fraction of signal in top N groups
        - fold_enrichment: Ratio of top mean to rest mean
        - normalized_entropy: Entropy-based specificity (0=specific, 1=uniform)
        - total_signal: Total accessibility signal

    Notes
    -----
    Specificity score: Higher values (closer to 1) indicate more specific clusters
    that have signal concentrated in top N groups.

    Normalized entropy: Lower values indicate higher specificity. Calculated as
    Shannon entropy normalized by maximum possible entropy.

    Fold enrichment: Ratio of mean accessibility in top N groups vs remaining groups.
    Higher values indicate stronger specificity.

    Examples
    --------
    >>> access = pd.DataFrame(np.random.rand(100, 50))
    >>> specificity = calculate_cluster_specificity(access, top_n=2)
    >>> print(f"Most specific cluster: {specificity['specificity_score'].idxmax()}")
    >>> high_spec = specificity[specificity['specificity_score'] > 0.5]
    >>> print(f"Found {len(high_spec)} highly specific clusters")
    """
    specificity_metrics = []

    for cluster_id in df_clusters_groups.index:
        values = df_clusters_groups.loc[cluster_id].sort_values(ascending=False)

        # Calculate metrics
        top_values = values.iloc[:top_n]
        rest_values = values.iloc[top_n:]

        total_signal = values.sum()
        top_signal = top_values.sum()

        # Specificity score: fraction of signal in top N groups
        specificity_score = top_signal / total_signal if total_signal > 0 else 0

        # Signal concentration: ratio of top mean to rest mean
        top_mean = top_values.mean()
        rest_mean = rest_values.mean() if len(rest_values) > 0 else 0
        fold_enrichment = top_mean / rest_mean if rest_mean > 0 else top_mean

        # Entropy-based specificity (lower = more specific)
        normalized_values = values / values.sum() if values.sum() > 0 else values
        entropy = -np.sum(normalized_values * np.log2(normalized_values + 1e-10))
        max_entropy = np.log2(len(values))
        normalized_entropy = entropy / max_entropy

        specificity_metrics.append({
            'cluster_id': cluster_id,
            'top1_group': values.index[0],
            'top1_value': values.iloc[0],
            'top2_group': values.index[1] if len(values) > 1 else None,
            'top2_value': values.iloc[1] if len(values) > 1 else 0,
            'specificity_score': specificity_score,
            'fold_enrichment': fold_enrichment,
            'normalized_entropy': normalized_entropy,
            'total_signal': total_signal
        })

    df_specificity = pd.DataFrame(specificity_metrics).set_index('cluster_id')

    logger.info(f"Calculated specificity for {len(df_specificity)} clusters")
    logger.info(f"Mean specificity score: {df_specificity['specificity_score'].mean():.3f}")
    logger.info(f"Median specificity score: {df_specificity['specificity_score'].median():.3f}")

    return df_specificity


def visualize_specificity_distribution(df_specificity: pd.DataFrame,
                                       savefig: bool = False,
                                       filename: str = "cluster_specificity_distribution.pdf") -> None:
    """
    Visualize specificity distribution across clusters

    Parameters
    ----------
    df_specificity : pd.DataFrame
        Output from calculate_cluster_specificity()
    savefig : bool, default=False
        Whether to save figure
    filename : str
        Output filename

    Examples
    --------
    >>> specificity = calculate_cluster_specificity(access_matrix)
    >>> visualize_specificity_distribution(specificity, savefig=True)
    """
    logger.info("Creating specificity distribution plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Specificity score distribution
    ax = axes[0, 0]
    ax.hist(df_specificity['specificity_score'], bins=50, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', label='50% threshold')
    ax.set_xlabel('Specificity Score\n(Fraction in top 2 groups)')
    ax.set_ylabel('Number of clusters')
    ax.set_title('Distribution of Cluster Specificity')
    ax.legend()

    # 2. Fold enrichment vs specificity
    ax = axes[0, 1]
    scatter = ax.scatter(df_specificity['specificity_score'],
                        np.log2(df_specificity['fold_enrichment'] + 1),
                        c=df_specificity['total_signal'],
                        cmap='viridis', alpha=0.6)
    ax.set_xlabel('Specificity Score')
    ax.set_ylabel('Log2(Fold Enrichment + 1)')
    ax.set_title('Specificity vs Fold Enrichment')
    plt.colorbar(scatter, ax=ax, label='Total Signal')

    # 3. Normalized entropy distribution
    ax = axes[1, 0]
    ax.hist(df_specificity['normalized_entropy'], bins=50, edgecolor='black', color='orange')
    ax.axvline(0.5, color='red', linestyle='--', label='0.5 threshold')
    ax.set_xlabel('Normalized Entropy\n(Lower = More Specific)')
    ax.set_ylabel('Number of clusters')
    ax.set_title('Entropy-Based Specificity')
    ax.legend()

    # 4. Entropy vs specificity score
    ax = axes[1, 1]
    ax.scatter(df_specificity['specificity_score'],
              df_specificity['normalized_entropy'],
              alpha=0.5)
    ax.set_xlabel('Specificity Score')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Specificity Score vs Entropy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if savefig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to: {filename}")

    plt.show()


def identify_highly_specific_clusters(df_specificity: pd.DataFrame,
                                      specificity_threshold: float = 0.6,
                                      fold_enrichment_threshold: float = 5.0,
                                      min_signal: float = 0.01) -> pd.DataFrame:
    """
    Identify highly specific clusters based on multiple criteria

    Parameters
    ----------
    df_specificity : pd.DataFrame
        Output from calculate_cluster_specificity()
    specificity_threshold : float, default=0.6
        Minimum specificity score
    fold_enrichment_threshold : float, default=5.0
        Minimum fold enrichment
    min_signal : float, default=0.01
        Minimum total signal threshold

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with highly specific clusters

    Examples
    --------
    >>> specificity = calculate_cluster_specificity(access_matrix)
    >>> high_spec = identify_highly_specific_clusters(
    ...     specificity, specificity_threshold=0.7
    ... )
    >>> print(f"Found {len(high_spec)} highly specific clusters")
    """
    logger.info("=== IDENTIFYING HIGHLY SPECIFIC CLUSTERS ===")
    logger.info(f"Criteria:")
    logger.info(f"  Specificity score ≥ {specificity_threshold}")
    logger.info(f"  Fold enrichment ≥ {fold_enrichment_threshold}")
    logger.info(f"  Total signal ≥ {min_signal}")

    highly_specific = df_specificity[
        (df_specificity['specificity_score'] >= specificity_threshold) &
        (df_specificity['fold_enrichment'] >= fold_enrichment_threshold) &
        (df_specificity['total_signal'] >= min_signal)
    ].copy()

    # Sort by specificity score
    highly_specific = highly_specific.sort_values('specificity_score', ascending=False)

    logger.info(f"\nFound {len(highly_specific)} highly specific clusters "
               f"({len(highly_specific)/len(df_specificity)*100:.1f}% of total)")

    if len(highly_specific) > 0:
        logger.info("\nTop 10 most specific clusters:")
        for idx, (cluster_id, row) in enumerate(highly_specific.head(10).iterrows()):
            logger.info(f"  {idx+1:2d}. {cluster_id}: "
                       f"specificity={row['specificity_score']:.3f}, "
                       f"fold_enrich={row['fold_enrichment']:.1f}, "
                       f"top_group={row['top1_group']}")

    return highly_specific


def annotate_block_clusters(block_name: str,
                            cluster_list: List[str],
                            additional_info: str = "") -> str:
    """
    Helper function to create annotation text for a block

    Parameters
    ----------
    block_name : str
        Name of the similarity block
    cluster_list : List[str]
        List of cluster IDs in the block
    additional_info : str, default=""
        Additional information to include

    Returns
    -------
    str
        Formatted annotation text

    Examples
    --------
    >>> clusters = ['0_0', '0_1', '0_2', '1_0']
    >>> annotation = annotate_block_clusters('Block1', clusters, 'avg_sim=0.85')
    >>> print(annotation)
    Block1: 4 clusters (avg_sim=0.85)
    Sample: 0_0, 0_1, 0_2...
    """
    annotation = f"{block_name}: {len(cluster_list)} clusters"
    if additional_info:
        annotation += f" ({additional_info})"

    # Add sample clusters
    if len(cluster_list) <= 3:
        annotation += f"\nClusters: {', '.join(cluster_list)}"
    else:
        annotation += f"\nSample: {', '.join(cluster_list[:3])}..."

    return annotation


def analyze_specificity_by_celltype(df_clusters_groups: pd.DataFrame,
                                    df_specificity: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which cell types have the most specific peak clusters

    Parameters
    ----------
    df_clusters_groups : pd.DataFrame
        Cluster accessibility matrix
    df_specificity : pd.DataFrame
        Output from calculate_cluster_specificity()

    Returns
    -------
    pd.DataFrame
        Summary statistics per cell type

    Examples
    --------
    >>> specificity = calculate_cluster_specificity(access_matrix)
    >>> celltype_summary = analyze_specificity_by_celltype(
    ...     access_matrix, specificity
    ... )
    >>> print(celltype_summary.sort_values('mean_specificity', ascending=False).head())
    """
    logger.info("=== ANALYZING SPECIFICITY BY CELL TYPE ===")

    # Group clusters by their top cell type
    celltype_stats = []

    for group in df_clusters_groups.columns:
        # Find clusters where this is the top group
        top_clusters = df_specificity[df_specificity['top1_group'] == group]

        if len(top_clusters) > 0:
            celltype_stats.append({
                'celltype_timepoint': group,
                'n_clusters': len(top_clusters),
                'mean_specificity': top_clusters['specificity_score'].mean(),
                'median_specificity': top_clusters['specificity_score'].median(),
                'mean_fold_enrichment': top_clusters['fold_enrichment'].mean(),
                'n_highly_specific': (top_clusters['specificity_score'] > 0.6).sum()
            })

    df_celltype_stats = pd.DataFrame(celltype_stats)

    if len(df_celltype_stats) > 0:
        df_celltype_stats = df_celltype_stats.sort_values('n_clusters', ascending=False)

        logger.info("\nTop 10 cell type/timepoint groups by cluster count:")
        for idx, (_, row) in enumerate(df_celltype_stats.head(10).iterrows()):
            logger.info(f"  {idx+1:2d}. {row['celltype_timepoint']}: "
                       f"{row['n_clusters']} clusters, "
                       f"mean_spec={row['mean_specificity']:.3f}, "
                       f"{row['n_highly_specific']} highly specific")

    return df_celltype_stats


def compare_specificity_across_blocks(df_specificity: pd.DataFrame,
                                      blocks_data: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compare specificity metrics across similarity blocks

    Parameters
    ----------
    df_specificity : pd.DataFrame
        Output from calculate_cluster_specificity()
    blocks_data : Dict[str, List[str]]
        Dictionary with block_name -> list of cluster IDs

    Returns
    -------
    pd.DataFrame
        Summary statistics per block

    Examples
    --------
    >>> specificity = calculate_cluster_specificity(access_matrix)
    >>> blocks = {'Block1': ['0_0', '0_1'], 'Block2': ['1_0', '1_1']}
    >>> block_comparison = compare_specificity_across_blocks(
    ...     specificity, blocks
    ... )
    >>> print(block_comparison)
    """
    logger.info("=== COMPARING SPECIFICITY ACROSS BLOCKS ===")

    block_stats = []

    for block_name, cluster_list in blocks_data.items():
        # Get specificity for clusters in this block
        valid_clusters = [c for c in cluster_list if c in df_specificity.index]

        if len(valid_clusters) > 0:
            block_spec = df_specificity.loc[valid_clusters]

            block_stats.append({
                'block_name': block_name,
                'n_clusters': len(valid_clusters),
                'mean_specificity': block_spec['specificity_score'].mean(),
                'median_specificity': block_spec['specificity_score'].median(),
                'mean_fold_enrichment': block_spec['fold_enrichment'].mean(),
                'mean_entropy': block_spec['normalized_entropy'].mean(),
                'n_highly_specific': (block_spec['specificity_score'] > 0.6).sum(),
                'pct_highly_specific': (block_spec['specificity_score'] > 0.6).sum() / len(block_spec) * 100
            })

    df_block_stats = pd.DataFrame(block_stats)

    if len(df_block_stats) > 0:
        logger.info("\nBlock specificity summary:")
        for _, row in df_block_stats.iterrows():
            logger.info(f"  {row['block_name']}: "
                       f"mean_spec={row['mean_specificity']:.3f}, "
                       f"{row['pct_highly_specific']:.1f}% highly specific "
                       f"({row['n_highly_specific']}/{row['n_clusters']} clusters)")

    return df_block_stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Module: subgrn_cluster_specificity.py")
    print("Contains functions for analyzing peak cluster specificity")
