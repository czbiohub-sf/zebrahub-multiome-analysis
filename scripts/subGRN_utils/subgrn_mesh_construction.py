"""
TF-gene mesh network construction from peak clusters

This module constructs "mesh" networks representing predicted TF-gene regulatory
relationships based on peak cluster motif enrichment and linked genes.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
from typing import Mapping, Hashable, Sequence, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def cluster_dict_to_df(d: Mapping[Hashable, Sequence], col_name: str) -> pd.DataFrame:
    """
    Convert a {cluster: list} dictionary into a single-column DataFrame

    Parameters
    ----------
    d : Mapping[Hashable, Sequence]
        Dictionary where keys are cluster IDs and values are list-like collections
    col_name : str
        Name for the output column

    Returns
    -------
    pd.DataFrame
        DataFrame with cluster IDs as index and list lengths as values

    Examples
    --------
    >>> clusters_tfs = {'0_0': ['gata1', 'tal1'], '0_1': ['sox2']}
    >>> df = cluster_dict_to_df(clusters_tfs, 'n_TFs')
    >>> print(df)
         n_TFs
    0_0      2
    0_1      1
    """
    return pd.DataFrame({col_name: [len(v) for v in d.values()]}, index=d.keys())


def build_master_df(
    dict_map: Mapping[str, Mapping[Hashable, Sequence]],
    *,
    prefix: str = "n_",
    fill_value: int = 0
) -> pd.DataFrame:
    """
    Build master DataFrame combining multiple cluster dictionaries

    Parameters
    ----------
    dict_map : Mapping[str, Mapping[Hashable, Sequence]]
        Dictionary of dictionaries. Keys are labels (e.g., "tfs", "genes"),
        values are {cluster → list} dictionaries
    prefix : str, default="n_"
        Prefix prepended to column names (e.g., "n_tfs", "n_genes")
    fill_value : int, default=0
        Value to fill for missing clusters

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all cluster statistics

    Examples
    --------
    >>> dict_map = {
    ...     "tfs": {'0_0': ['gata1', 'tal1'], '0_1': ['sox2']},
    ...     "genes": {'0_0': ['cd34', 'cd45']}
    ... }
    >>> df = build_master_df(dict_map, prefix="n_")
    >>> print(df.columns)
    Index(['n_tfs', 'n_genes'], dtype='object')
    """
    dfs = [
        cluster_dict_to_df(d, f"{prefix}{label}")
        for label, d in dict_map.items()
    ]
    master = pd.concat(dfs, axis=1)  # outer join on index
    return master.fillna(fill_value).astype(int)


def create_cluster_tf_matrix(clusters_tfs_dict: Dict[str, List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a binary cluster-by-TFs matrix

    Parameters
    ----------
    clusters_tfs_dict : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of TF names

    Returns
    -------
    cluster_tf_matrix : pd.DataFrame
        Binary matrix (clusters × TFs) where 1 indicates TF is enriched in cluster
    all_tfs : List[str]
        Sorted list of all unique TFs across all clusters

    Examples
    --------
    >>> clusters_tfs = {'0_0': ['gata1', 'tal1'], '0_1': ['sox2', 'gata1']}
    >>> matrix, tfs = create_cluster_tf_matrix(clusters_tfs)
    >>> print(matrix)
         gata1  sox2  tal1
    0_0      1     0     1
    0_1      1     1     0
    >>> print(tfs)
    ['gata1', 'sox2', 'tal1']
    """
    logger.info("Creating clusters-by-TFs matrix...")

    # Get all unique TFs across all clusters
    all_tfs = set()
    for tfs in clusters_tfs_dict.values():
        all_tfs.update(tfs)

    all_tfs = sorted(list(all_tfs))  # Sort for consistency
    cluster_ids = sorted(list(clusters_tfs_dict.keys()))

    logger.info(f"Total unique TFs across all clusters: {len(all_tfs)}")
    logger.info(f"Total clusters: {len(cluster_ids)}")

    # Create binary matrix
    cluster_tf_matrix = pd.DataFrame(0, index=cluster_ids, columns=all_tfs)

    for cluster_id, tfs in clusters_tfs_dict.items():
        for tf in tfs:
            cluster_tf_matrix.loc[cluster_id, tf] = 1

    logger.info(f"Matrix shape: {cluster_tf_matrix.shape}")
    logger.info(f"Total 1s in matrix: {cluster_tf_matrix.sum().sum()}")
    logger.info(f"Sparsity: {(1 - cluster_tf_matrix.sum().sum() / cluster_tf_matrix.size) * 100:.1f}%")

    return cluster_tf_matrix, all_tfs


def create_cluster_gene_matrix(clusters_genes_dict: Dict[str, List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a binary cluster-by-genes matrix

    Parameters
    ----------
    clusters_genes_dict : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of gene names

    Returns
    -------
    cluster_gene_matrix : pd.DataFrame
        Binary matrix (clusters × genes) where 1 indicates gene is linked to cluster
    all_genes : List[str]
        Sorted list of all unique genes across all clusters

    Examples
    --------
    >>> clusters_genes = {'0_0': ['cd34', 'cd45'], '0_1': ['sox2', 'cd34']}
    >>> matrix, genes = create_cluster_gene_matrix(clusters_genes)
    >>> print(matrix.shape)
    (2, 3)
    """
    logger.info("Creating clusters-by-genes matrix...")

    # Get all unique genes across all clusters
    all_genes = set()
    for genes in clusters_genes_dict.values():
        all_genes.update(genes)

    all_genes = sorted(list(all_genes))
    cluster_ids = sorted(list(clusters_genes_dict.keys()))

    logger.info(f"Total unique genes across all clusters: {len(all_genes)}")
    logger.info(f"Total clusters: {len(cluster_ids)}")

    # Create binary matrix
    cluster_gene_matrix = pd.DataFrame(0, index=cluster_ids, columns=all_genes)

    for cluster_id, genes in clusters_genes_dict.items():
        for gene in genes:
            cluster_gene_matrix.loc[cluster_id, gene] = 1

    return cluster_gene_matrix, all_genes


def create_tf_gene_mesh(cluster_id: str,
                       tfs: List[str],
                       genes: List[str]) -> pd.DataFrame:
    """
    Create TF-by-gene mesh matrix for a single cluster

    This creates a binarized matrix where TF-gene pairs represent
    potential regulatory relationships to be validated against GRNs.

    Parameters
    ----------
    cluster_id : str
        Cluster identifier
    tfs : List[str]
        List of transcription factors enriched in this cluster
    genes : List[str]
        List of linked genes in this cluster

    Returns
    -------
    pd.DataFrame
        Binary matrix (TFs × genes) initialized to 1 for all pairs

    Notes
    -----
    All TF-gene pairs are initially set to 1 (predicted), then filtered
    by actual GRN edges during subGRN extraction.

    Examples
    --------
    >>> mesh = create_tf_gene_mesh('0_0', ['gata1', 'tal1'], ['cd34', 'cd45'])
    >>> print(mesh.shape)
    (2, 2)
    >>> print((mesh == 1).all().all())
    True
    """
    # Create all-ones matrix (all TF-gene pairs are potential interactions)
    mesh = pd.DataFrame(1, index=tfs, columns=genes)

    logger.debug(f"Created mesh for cluster {cluster_id}: {len(tfs)} TFs × {len(genes)} genes")

    return mesh


def create_all_cluster_meshes(clusters_tfs_dict: Dict[str, List[str]],
                              clusters_genes_dict: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Create TF-gene mesh matrices for all clusters

    Parameters
    ----------
    clusters_tfs_dict : Dict[str, List[str]]
        Dictionary mapping cluster IDs to TF lists
    clusters_genes_dict : Dict[str, List[str]]
        Dictionary mapping cluster IDs to gene lists

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping cluster IDs to TF-gene mesh matrices

    Examples
    --------
    >>> clusters_tfs = {'0_0': ['gata1'], '0_1': ['sox2']}
    >>> clusters_genes = {'0_0': ['cd34'], '0_1': ['nes']}
    >>> meshes = create_all_cluster_meshes(clusters_tfs, clusters_genes)
    >>> print(len(meshes))
    2
    """
    cluster_meshes = {}

    # Get all cluster IDs that have both TFs and genes
    common_clusters = set(clusters_tfs_dict.keys()) & set(clusters_genes_dict.keys())

    logger.info(f"Creating meshes for {len(common_clusters)} clusters...")

    for cluster_id in common_clusters:
        tfs = clusters_tfs_dict[cluster_id]
        genes = clusters_genes_dict[cluster_id]

        if len(tfs) > 0 and len(genes) > 0:
            mesh = create_tf_gene_mesh(cluster_id, tfs, genes)
            cluster_meshes[cluster_id] = mesh

    logger.info(f"Created {len(cluster_meshes)} cluster meshes")

    return cluster_meshes


def filter_motifs_by_threshold(clust_by_motifs: pd.DataFrame,
                               threshold: float = 2.0) -> pd.DataFrame:
    """
    Binarize motif enrichment matrix by threshold

    Parameters
    ----------
    clust_by_motifs : pd.DataFrame
        Matrix of motif enrichment scores (clusters × motifs)
    threshold : float, default=2.0
        Z-score threshold for calling motif enrichment significant

    Returns
    -------
    pd.DataFrame
        Binary matrix where 1 indicates enrichment above threshold

    Examples
    --------
    >>> enrichment = pd.DataFrame([[1.5, 2.5], [3.0, 0.5]],
    ...                          columns=['motif1', 'motif2'])
    >>> binary = filter_motifs_by_threshold(enrichment, threshold=2.0)
    >>> print(binary.values)
    [[0 1]
     [1 0]]
    """
    logger.info(f"Applying threshold {threshold} to motif enrichment matrix")

    binary_matrix = (clust_by_motifs >= threshold).astype(int)

    sig_counts = binary_matrix.sum(axis=1)
    logger.info(f"Mean significant motifs per cluster: {sig_counts.mean():.1f}")
    logger.info(f"Median significant motifs per cluster: {sig_counts.median():.1f}")
    logger.info(f"Clusters with no significant motifs: {(sig_counts == 0).sum()}")

    return binary_matrix


def compute_mesh_statistics(cluster_meshes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute summary statistics for all cluster meshes

    Parameters
    ----------
    cluster_meshes : Dict[str, pd.DataFrame]
        Dictionary of TF-gene mesh matrices

    Returns
    -------
    pd.DataFrame
        Statistics including n_TFs, n_genes, n_edges for each cluster

    Examples
    --------
    >>> meshes = {'0_0': pd.DataFrame(1, index=['gata1'], columns=['cd34'])}
    >>> stats = compute_mesh_statistics(meshes)
    >>> print(stats.columns)
    Index(['n_TFs', 'n_genes', 'n_edges'], dtype='object')
    """
    stats = []

    for cluster_id, mesh in cluster_meshes.items():
        n_tfs = mesh.shape[0]
        n_genes = mesh.shape[1]
        n_edges = (mesh == 1).sum().sum()

        stats.append({
            'cluster_id': cluster_id,
            'n_TFs': n_tfs,
            'n_genes': n_genes,
            'n_edges': n_edges
        })

    df_stats = pd.DataFrame(stats).set_index('cluster_id')
    return df_stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Example: Create cluster TF matrix
    clusters_tfs = {
        '0_0': ['gata1', 'tal1', 'fli1'],
        '0_1': ['sox2', 'pax6'],
        '1_0': ['gata1', 'sox2']
    }

    matrix, all_tfs = create_cluster_tf_matrix(clusters_tfs)
    print(f"\nCluster-TF Matrix:\n{matrix}")
    print(f"\nAll TFs: {all_tfs}")

    # Example: Create mesh
    clusters_genes = {
        '0_0': ['cd34', 'cd45'],
        '0_1': ['nes', 'sox9'],
        '1_0': ['cd34', 'sox9']
    }

    meshes = create_all_cluster_meshes(clusters_tfs, clusters_genes)
    print(f"\nCreated {len(meshes)} meshes")
    print(f"\nExample mesh for cluster 0_0:\n{meshes['0_0']}")
