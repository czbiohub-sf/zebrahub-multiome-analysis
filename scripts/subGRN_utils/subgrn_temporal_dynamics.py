"""
Temporal dynamics scoring and ranking for subGRNs

This module provides functions for computing temporal dynamics scores,
identifying biologically interesting subGRNs based on their evolution
across developmental timepoints.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient to measure concentration of accessibility

    The Gini coefficient measures inequality in a distribution. For peak cluster
    accessibility, it indicates how concentrated activity is in specific cell types.

    Parameters
    ----------
    values : array-like
        Accessibility values across all pseudobulk groups

    Returns
    -------
    float
        Gini coefficient (0=equal distribution, 1=concentrated in few groups)

    Examples
    --------
    >>> values = np.array([0.1, 0.1, 0.8])  # Highly concentrated
    >>> print(f"{gini_coefficient(values):.2f}")
    0.47
    >>> values = np.array([0.33, 0.33, 0.34])  # Evenly distributed
    >>> print(f"{gini_coefficient(values):.2f}")
    0.00
    """
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


def find_most_accessible_celltype(cluster_id: str,
                                  df_clusters_groups: pd.DataFrame,
                                  min_accessibility: float = 0.01) -> Optional[Dict]:
    """
    Find the celltype×timepoint with highest accessibility for a peak cluster

    Parameters
    ----------
    cluster_id : str
        Peak cluster ID (e.g., "26_11")
    df_clusters_groups : pd.DataFrame
        Cluster × pseudobulk accessibility matrix (clusters as rows)
    min_accessibility : float, default=0.01
        Minimum threshold to consider (filter noise)

    Returns
    -------
    dict or None
        Dictionary containing:
        - cluster_id: str
        - best_group: str (e.g., "hemangioblasts_15")
        - celltype: str (e.g., "hemangioblasts")
        - timepoint: str (e.g., "15")
        - accessibility: float
        - gini_coefficient: float (concentration score)
        - top_5_groups: list of (group, accessibility) tuples
        Returns None if no signal above threshold

    Examples
    --------
    >>> df = pd.DataFrame([[0.05, 0.85, 0.10]],
    ...                   index=['26_11'],
    ...                   columns=['neural_05', 'hemangioblasts_15', 'PSM_20'])
    >>> result = find_most_accessible_celltype('26_11', df)
    >>> print(result['celltype'])
    hemangioblasts
    >>> print(result['timepoint'])
    15
    """
    # Get accessibility profile for this cluster
    values = df_clusters_groups.loc[cluster_id]

    # Filter by minimum threshold
    values_filtered = values[values >= min_accessibility]

    if len(values_filtered) == 0:
        logger.warning(f"No accessibility signal above {min_accessibility} for cluster {cluster_id}")
        return None

    # Compute Gini coefficient (concentration)
    gini = gini_coefficient(values.values)

    # Find top group
    best_group = values_filtered.idxmax()
    best_value = values_filtered.max()

    # Parse celltype and timepoint from group name
    # Format: "celltype_timepoint" (e.g., "hemangioblasts_15")
    parts = best_group.rsplit('_', 1)
    if len(parts) == 2:
        celltype, timepoint = parts
    else:
        celltype = best_group
        timepoint = None

    # Get top 5 for context
    top_5 = [(group, val) for group, val in
             values_filtered.sort_values(ascending=False).head(5).items()]

    return {
        'cluster_id': cluster_id,
        'best_group': best_group,
        'celltype': celltype,
        'timepoint': timepoint,
        'accessibility': best_value,
        'gini_coefficient': gini,
        'top_5_groups': top_5
    }


def compute_temporal_dynamics_score(cluster_id: str,
                                   celltype: str,
                                   grn_dict: Dict[Tuple[str, str], pd.DataFrame],
                                   cluster_tf_gene_matrices: Dict[str, pd.DataFrame],
                                   min_edges: int = 5,
                                   min_timepoints: int = 3) -> Optional[Dict]:
    """
    Compute temporal dynamics score for a cluster-celltype combination

    This score quantifies how dynamically a subGRN changes over developmental time,
    considering TF turnover, edge turnover, and developmental TF enrichment.

    Parameters
    ----------
    cluster_id : str
        Peak cluster ID
    celltype : str
        Cell type of interest
    grn_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary of GRNs
    cluster_tf_gene_matrices : Dict[str, pd.DataFrame]
        TF-gene mesh matrices
    min_edges : int, default=5
        Minimum edges required at any timepoint
    min_timepoints : int, default=3
        Minimum timepoints required

    Returns
    -------
    dict or None
        Dictionary with:
        - cluster_id, celltype, dynamics_score
        - component scores: tf_turnover, edge_turnover, dev_tf_turnover, temporal_variance
        - developmental_tfs_list: list of TFs
        - timepoints_with_edges: list
        - max_edges, min_edges, mean_edges
        Returns None if filters not met

    Notes
    -----
    Dynamics score formula:
    score = 0.4 * dev_tf_turnover + 0.3 * edge_turnover +
            0.2 * tf_turnover + 0.1 * temporal_variance

    Examples
    --------
    >>> score_dict = compute_temporal_dynamics_score(
    ...     '26_11', 'hemangioblasts', grn_dict, cluster_meshes
    ... )
    >>> if score_dict:
    ...     print(f"Dynamics score: {score_dict['dynamics_score']:.3f}")
    """
    # Get predicted pairs
    if cluster_id not in cluster_tf_gene_matrices:
        return None

    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = []
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.append((tf, gene))

    predicted_pairs = set(predicted_pairs)

    # Get timepoints for this celltype
    timepoints = sorted([tp for (ct, tp) in grn_dict.keys() if ct == celltype])

    if len(timepoints) < min_timepoints:
        return None

    # Extract subGRNs across timepoints
    subgrns = {}
    all_tfs = set()
    all_edges_across_time = set()

    for timepoint in timepoints:
        if (celltype, timepoint) not in grn_dict:
            continue

        grn_df = grn_dict[(celltype, timepoint)]
        grn_pairs = set(zip(grn_df['source'], grn_df['target']))
        found_pairs = predicted_pairs & grn_pairs

        mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
        subgrn = grn_df[mask].copy()

        if len(subgrn) > 0:
            subgrns[timepoint] = subgrn
            all_tfs.update(subgrn['source'].unique())
            all_edges_across_time.update(zip(subgrn['source'], subgrn['target']))

    if len(subgrns) < min_timepoints:
        return None

    # Check minimum edge requirement
    max_edges = max(len(subgrn) for subgrn in subgrns.values())
    if max_edges < min_edges:
        return None

    # Calculate component scores
    edge_counts = [len(subgrn) for subgrn in subgrns.values()]

    # 1. TF turnover: how many TFs appear/disappear
    tf_sets_by_timepoint = [set(subgrn['source'].unique()) for subgrn in subgrns.values()]
    tf_changes = sum(len(tf_sets_by_timepoint[i] ^ tf_sets_by_timepoint[i-1])
                     for i in range(1, len(tf_sets_by_timepoint)))
    tf_turnover = tf_changes / max(len(all_tfs), 1)

    # 2. Edge turnover: how many edges change
    edge_sets_by_timepoint = [set(zip(subgrn['source'], subgrn['target']))
                              for subgrn in subgrns.values()]
    edge_changes = sum(len(edge_sets_by_timepoint[i] ^ edge_sets_by_timepoint[i-1])
                      for i in range(1, len(edge_sets_by_timepoint)))
    edge_turnover = edge_changes / max(len(all_edges_across_time), 1)

    # 3. Developmental TF turnover: appearance of new TFs over time
    developmental_tfs = set()
    for i in range(1, len(tf_sets_by_timepoint)):
        new_tfs = tf_sets_by_timepoint[i] - tf_sets_by_timepoint[i-1]
        developmental_tfs.update(new_tfs)
    dev_tf_turnover = len(developmental_tfs) / max(len(all_tfs), 1)

    # 4. Temporal variance in edge count
    temporal_variance = np.std(edge_counts) / max(np.mean(edge_counts), 1)

    # Combined score (weighted)
    dynamics_score = (0.4 * dev_tf_turnover +
                     0.3 * edge_turnover +
                     0.2 * tf_turnover +
                     0.1 * temporal_variance)

    return {
        'cluster_id': cluster_id,
        'celltype': celltype,
        'dynamics_score': dynamics_score,
        'tf_turnover': tf_turnover,
        'edge_turnover': edge_turnover,
        'dev_tf_turnover': dev_tf_turnover,
        'temporal_variance': temporal_variance,
        'developmental_tfs_list': sorted(list(developmental_tfs)),
        'timepoints_with_edges': sorted(subgrns.keys()),
        'n_timepoints': len(subgrns),
        'max_edges': max_edges,
        'min_edges': min(edge_counts),
        'mean_edges': np.mean(edge_counts)
    }


def rank_clusters_by_temporal_dynamics(df_clusters_groups: pd.DataFrame,
                                       grn_dict: Dict[Tuple[str, str], pd.DataFrame],
                                       cluster_tf_gene_matrices: Dict[str, pd.DataFrame],
                                       min_edges: int = 5,
                                       min_timepoints: int = 3,
                                       top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Rank all cluster-celltype combinations by temporal dynamics

    Parameters
    ----------
    df_clusters_groups : pd.DataFrame
        Cluster accessibility matrix
    grn_dict : Dict[Tuple[str, str], pd.DataFrame]
        GRN dictionary
    cluster_tf_gene_matrices : Dict[str, pd.DataFrame]
        TF-gene mesh matrices
    min_edges : int, default=5
        Minimum edge threshold
    min_timepoints : int, default=3
        Minimum timepoint threshold
    top_n : int, optional
        Return only top N results

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame with columns: cluster_id, celltype, dynamics_score, etc.

    Examples
    --------
    >>> df_ranked = rank_clusters_by_temporal_dynamics(
    ...     df_access, grn_dict, cluster_meshes, top_n=10
    ... )
    >>> print(df_ranked.head())
    """
    logger.info("Ranking clusters by temporal dynamics...")

    results = []

    for cluster_id in cluster_tf_gene_matrices.keys():
        # Find most accessible celltype
        access_info = find_most_accessible_celltype(cluster_id, df_clusters_groups)

        if access_info is None:
            continue

        celltype = access_info['celltype']

        # Compute dynamics score
        dynamics_info = compute_temporal_dynamics_score(
            cluster_id, celltype, grn_dict, cluster_tf_gene_matrices,
            min_edges=min_edges, min_timepoints=min_timepoints
        )

        if dynamics_info is not None:
            # Merge information
            result = {**access_info, **dynamics_info}
            results.append(result)

    df_results = pd.DataFrame(results)

    if len(df_results) == 0:
        logger.warning("No clusters met the criteria")
        return df_results

    # Sort by dynamics score
    df_results = df_results.sort_values('dynamics_score', ascending=False)

    logger.info(f"Ranked {len(df_results)} cluster-celltype combinations")

    if top_n is not None:
        df_results = df_results.head(top_n)
        logger.info(f"Returning top {top_n} results")

    return df_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Gini coefficient
    values = np.array([0.1, 0.1, 0.8])
    print(f"Gini coefficient: {gini_coefficient(values):.3f}")

    print("\nModule: subgrn_temporal_dynamics.py")
    print("Contains functions for temporal dynamics scoring")
