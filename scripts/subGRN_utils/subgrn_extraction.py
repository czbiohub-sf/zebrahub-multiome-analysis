"""
SubGRN extraction from full GRNs using TF-gene mesh predictions

This module extracts sub-Gene Regulatory Networks (subGRNs) by intersecting
predicted TF-gene relationships (from peak cluster meshes) with inferred
GRNs from CellOracle.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)


def extract_subGRN_from_cluster(grn_df: pd.DataFrame,
                                cluster_tf_gene_matrix: pd.DataFrame,
                                cluster_id: str) -> pd.DataFrame:
    """
    Extract subGRN based on TF-gene relationships from peak cluster

    This function filters a full GRN to only include edges that are predicted
    by the cluster's TF-gene mesh.

    Parameters
    ----------
    grn_df : pd.DataFrame
        GRN dataframe with columns ['source', 'target', 'coef_mean', ...]
    cluster_tf_gene_matrix : pd.DataFrame
        Binary TF-by-genes matrix (from mesh construction)
    cluster_id : str
        Identifier for the cluster

    Returns
    -------
    pd.DataFrame
        Filtered GRN dataframe containing only predicted edges, with added
        'cluster_id' column

    Examples
    --------
    >>> grn = pd.DataFrame({
    ...     'source': ['gata1', 'gata1', 'sox2'],
    ...     'target': ['cd34', 'tal1', 'nes'],
    ...     'coef_mean': [0.5, 0.3, -0.2]
    ... })
    >>> mesh = pd.DataFrame(1, index=['gata1'], columns=['cd34', 'tal1'])
    >>> subgrn = extract_subGRN_from_cluster(grn, mesh, '0_0')
    >>> print(len(subgrn))
    2
    >>> print(subgrn['cluster_id'].unique())
    ['0_0']
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

    logger.debug(f"Cluster {cluster_id}: {len(subgrn)} edges found from {len(predicted_pairs)} predictions")

    return subgrn


def extract_all_cluster_subGRNs(grn_df: pd.DataFrame,
                                cluster_dict: Dict[str, pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Extract subGRNs for all clusters

    Parameters
    ----------
    grn_df : pd.DataFrame
        Full GRN dataframe
    cluster_dict : Dict[str, pd.DataFrame]
        Dictionary mapping cluster IDs to TF-gene mesh matrices

    Returns
    -------
    List[pd.DataFrame]
        List of subGRN dataframes (one per cluster with edges found)

    Examples
    --------
    >>> grn = pd.DataFrame({
    ...     'source': ['gata1', 'sox2'],
    ...     'target': ['cd34', 'nes'],
    ...     'coef_mean': [0.5, -0.2]
    ... })
    >>> meshes = {
    ...     '0_0': pd.DataFrame(1, index=['gata1'], columns=['cd34']),
    ...     '0_1': pd.DataFrame(1, index=['sox2'], columns=['nes'])
    ... }
    >>> subgrns = extract_all_cluster_subGRNs(grn, meshes)
    >>> print(len(subgrns))
    2
    """
    all_subgrns = []

    for cluster_id, tf_gene_matrix in cluster_dict.items():
        subgrn = extract_subGRN_from_cluster(grn_df, tf_gene_matrix, cluster_id)
        if len(subgrn) > 0:  # Only keep non-empty subGRNs
            all_subgrns.append(subgrn)
            logger.info(f"Cluster {cluster_id}: {len(subgrn)} edges found")

    logger.info(f"Extracted {len(all_subgrns)} non-empty subGRNs from {len(cluster_dict)} clusters")

    return all_subgrns


def extract_subgrn_for_celltype_timepoint(grn_dict: Dict[Tuple[str, str], pd.DataFrame],
                                          celltype: str,
                                          timepoint: str,
                                          predicted_pairs: Set[Tuple[str, str]]) -> pd.DataFrame:
    """
    Extract subGRN for specific celltype and timepoint

    Parameters
    ----------
    grn_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary of GRNs keyed by (celltype, timepoint)
    celltype : str
        Cell type of interest
    timepoint : str
        Timepoint of interest (e.g., '15')
    predicted_pairs : Set[Tuple[str, str]]
        Set of predicted (TF, target) tuples from mesh

    Returns
    -------
    pd.DataFrame
        Filtered GRN with only predicted edges

    Examples
    --------
    >>> grn_dict = {
    ...     ('neural_crest', '15'): pd.DataFrame({
    ...         'source': ['sox2', 'pax3'],
    ...         'target': ['nes', 'twist']
    ...     })
    ... }
    >>> predicted = {('sox2', 'nes')}
    >>> subgrn = extract_subgrn_for_celltype_timepoint(
    ...     grn_dict, 'neural_crest', '15', predicted
    ... )
    >>> print(len(subgrn))
    1
    """
    if (celltype, timepoint) not in grn_dict:
        logger.warning(f"GRN not found for ({celltype}, {timepoint})")
        return pd.DataFrame()

    grn_df = grn_dict[(celltype, timepoint)]

    # Find intersection with predicted pairs
    grn_pairs = set(zip(grn_df['source'], grn_df['target']))
    found_pairs = predicted_pairs & grn_pairs

    # Extract matching edges
    mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
    subgrn = grn_df[mask].copy()

    return subgrn


def extract_subgrn_metrics(cluster_id: str,
                          celltype_of_interest: str,
                          grn_dict: Dict[Tuple[str, str], pd.DataFrame],
                          cluster_tf_gene_matrices: Dict[str, pd.DataFrame],
                          predicted_pairs: Set[Tuple[str, str]]) -> Dict:
    """
    Extract comprehensive metrics for subGRN analysis

    This function computes key statistics about a subGRN including node counts,
    edge counts, TF/target classifications, and complexity reduction metrics.

    Parameters
    ----------
    cluster_id : str
        The cluster ID to analyze (e.g., "26_11")
    celltype_of_interest : str
        The celltype to focus on (e.g., "hemangioblasts")
    grn_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary of GRNs keyed by (celltype, timepoint)
    cluster_tf_gene_matrices : Dict[str, pd.DataFrame]
        Dictionary of TF-gene matrices for each cluster
    predicted_pairs : Set[Tuple[str, str]]
        Set of predicted TF-target pairs from the mesh

    Returns
    -------
    dict
        Dictionary containing:
        - total_nodes: Total unique nodes (NN)
        - total_tfs: Total transcription factors (N)
        - total_targets: Total target genes (M)
        - total_edges: Total edges
        - tf_only_nodes: Set of TF-only nodes
        - target_only_nodes: Set of target-only nodes
        - dual_nodes: Set of nodes that are both TF and target
        - complexity_reduction: Percentage reduction vs original mesh
        - subgrns_by_timepoint: Dict of subGRNs per timepoint

    Examples
    --------
    >>> metrics = extract_subgrn_metrics(
    ...     '26_11', 'hemangioblasts', grn_dict,
    ...     cluster_meshes, predicted_pairs
    ... )
    >>> print(f"Total nodes: {metrics['total_nodes']}")
    >>> print(f"TFs: {metrics['total_tfs']}, Targets: {metrics['total_targets']}")
    """
    logger.info(f"=== Extracting metrics for cluster {cluster_id} in {celltype_of_interest} ===")

    # Get all timepoints for this celltype
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)
    timepoints = sorted(timepoints)

    # Extract subGRNs for each timepoint
    all_subgrn_nodes = set()
    all_subgrn_edges = set()
    subgrns_by_timepoint = {}

    for timepoint in timepoints:
        subgrn = extract_subgrn_for_celltype_timepoint(
            grn_dict, celltype_of_interest, timepoint, predicted_pairs
        )
        subgrns_by_timepoint[timepoint] = subgrn

        if len(subgrn) > 0:
            # Collect nodes and edges
            all_subgrn_nodes.update(subgrn['source'])
            all_subgrn_nodes.update(subgrn['target'])
            all_subgrn_edges.update(zip(subgrn['source'], subgrn['target']))

    # Calculate node classifications
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns_by_timepoint.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])

    tf_only_nodes = all_sources - all_targets  # TFs that are never targets
    target_only_nodes = all_targets - all_sources  # Targets that are never TFs
    dual_tf_target_nodes = all_sources & all_targets  # Both TF and target

    # Calculate metrics
    total_nodes = len(all_subgrn_nodes)
    total_tfs = len(all_sources)
    total_targets = len(all_targets)
    total_edges = len(all_subgrn_edges)
    dual_nodes = len(dual_tf_target_nodes)

    # Calculate complexity reduction
    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    original_tf_count = len(cluster_matrix.index)
    original_gene_count = len(cluster_matrix.columns)
    original_possible_edges = (cluster_matrix == 1).sum().sum()

    edge_reduction = (1 - total_edges / original_possible_edges) * 100 if original_possible_edges > 0 else 0

    # Print detailed breakdown
    logger.info(f"\n=== SUBGRN COMPOSITION ===")
    logger.info(f"Total nodes (NN): {total_nodes}")
    logger.info(f"  - TF-only nodes: {len(tf_only_nodes)}")
    logger.info(f"  - Target-only nodes: {len(target_only_nodes)}")
    logger.info(f"  - Dual TF/Target nodes: {dual_nodes}")
    logger.info(f"Total transcription factors (N): {total_tfs}")
    logger.info(f"Total target genes (M): {total_targets}")
    logger.info(f"Total edges: {total_edges}")

    logger.info(f"\n=== COMPLEXITY REDUCTION ===")
    logger.info(f"Original mesh: {original_tf_count} TFs × {original_gene_count} genes = {original_possible_edges} possible edges")
    logger.info(f"SubGRN: {total_edges} edges ({edge_reduction:.1f}% reduction)")

    return {
        'cluster_id': cluster_id,
        'celltype': celltype_of_interest,
        'total_nodes': total_nodes,
        'total_tfs': total_tfs,
        'total_targets': total_targets,
        'total_edges': total_edges,
        'tf_only_nodes': tf_only_nodes,
        'target_only_nodes': target_only_nodes,
        'dual_nodes': dual_tf_target_nodes,
        'n_dual_nodes': dual_nodes,
        'complexity_reduction': edge_reduction,
        'original_mesh_size': original_possible_edges,
        'timepoints': timepoints,
        'subgrns_by_timepoint': subgrns_by_timepoint
    }


def get_predicted_pairs_from_mesh(cluster_tf_gene_matrix: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Extract all predicted TF-target pairs from mesh matrix

    Parameters
    ----------
    cluster_tf_gene_matrix : pd.DataFrame
        Binary TF-by-gene matrix

    Returns
    -------
    Set[Tuple[str, str]]
        Set of (TF, target) tuples where matrix value is 1

    Examples
    --------
    >>> mesh = pd.DataFrame([[1, 0], [1, 1]],
    ...                     index=['gata1', 'tal1'],
    ...                     columns=['cd34', 'cd45'])
    >>> pairs = get_predicted_pairs_from_mesh(mesh)
    >>> print(sorted(pairs))
    [('gata1', 'cd34'), ('tal1', 'cd34'), ('tal1', 'cd45')]
    """
    predicted_pairs = []
    for tf in cluster_tf_gene_matrix.index:
        for gene in cluster_tf_gene_matrix.columns:
            if cluster_tf_gene_matrix.loc[tf, gene] == 1:
                predicted_pairs.append((tf, gene))

    return set(predicted_pairs)


def count_subgrn_edges_per_timepoint(subgrns_by_timepoint: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Count edges in subGRNs across timepoints

    Parameters
    ----------
    subgrns_by_timepoint : Dict[str, pd.DataFrame]
        Dictionary mapping timepoints to subGRN dataframes

    Returns
    -------
    pd.DataFrame
        DataFrame with timepoint and edge_count columns

    Examples
    --------
    >>> subgrns = {
    ...     '05': pd.DataFrame({'source': [], 'target': []}),
    ...     '10': pd.DataFrame({'source': ['gata1', 'sox2'], 'target': ['cd34', 'nes']})
    ... }
    >>> counts = count_subgrn_edges_per_timepoint(subgrns)
    >>> print(counts)
       timepoint  edge_count
    0         05           0
    1         10           2
    """
    counts = []
    for timepoint, subgrn in sorted(subgrns_by_timepoint.items()):
        counts.append({
            'timepoint': timepoint,
            'edge_count': len(subgrn)
        })

    return pd.DataFrame(counts)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Example: Extract subGRN
    grn = pd.DataFrame({
        'source': ['gata1', 'gata1', 'tal1', 'sox2'],
        'target': ['cd34', 'tal1', 'cd45', 'nes'],
        'coef_mean': [0.5, 0.3, 0.4, -0.2]
    })

    mesh = pd.DataFrame(1, index=['gata1', 'tal1'], columns=['cd34', 'cd45', 'tal1'])

    subgrn = extract_subGRN_from_cluster(grn, mesh, '0_0')
    print(f"\nExtracted subGRN:\n{subgrn}")
    print(f"\nEdges found: {len(subgrn)}")
