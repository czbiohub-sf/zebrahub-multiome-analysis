"""
Similarity Metrics for Perturbation Analysis

Functions for quantifying how much knockout perturbations change
transition probabilities using cosine similarity and Euclidean distance.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean


def compute_row_cosine_similarities(df_wt, df_ko):
    """
    Compute cosine similarities between corresponding rows of WT and KO dataframes.

    Cosine similarity measures the angle between two vectors, ranging from -1 to 1.
    For transition probabilities (non-negative), values closer to 1 indicate
    similar transition patterns, while lower values indicate perturbation.

    Parameters
    ----------
    df_wt : pd.DataFrame
        Transition probability matrix for wildtype (WT)
    df_ko : pd.DataFrame
        Transition probability matrix for knockout (KO)

    Returns
    -------
    pd.Series
        Cosine similarities for each row (cell type or metacell),
        with name "cos_sim"

    Raises
    ------
    AssertionError
        If dataframes have different indices or columns

    Examples
    --------
    >>> # Compare cell type transitions for WT vs gene knockout
    >>> trans_wt = compute_celltype_transitions(adata, "T_fwd_WT")
    >>> trans_ko = compute_celltype_transitions(adata, "T_fwd_meox1_KO")
    >>> cosine_sims = compute_row_cosine_similarities(trans_wt, trans_ko)
    >>> print(cosine_sims)
    NMPs              0.87
    PSM               0.92
    fast_muscle       0.45  # <-- Most perturbed
    ...

    >>> # Compute for all genes systematically
    >>> cosine_sim_df = pd.DataFrame(index=trans_wt.index)
    >>> for gene in oracle.active_regulatory_genes:
    ...     trans_ko = compute_celltype_transitions(
    ...         adata, f"T_fwd_{gene}_KO"
    ...     )
    ...     cosine_sim_df[gene] = compute_row_cosine_similarities(
    ...         trans_wt, trans_ko
    ...     )

    Notes
    -----
    - Cosine similarity = 1.0 means no change from WT
    - Lower values indicate greater perturbation
    - This metric is insensitive to magnitude, only direction
    """
    # Ensure both dataframes have the same index and columns
    assert df_wt.index.equals(df_ko.index), "Dataframes must have the same index"
    assert df_wt.columns.equals(df_ko.columns), "Dataframes must have the same columns"

    similarities = {}
    for idx in df_wt.index:
        wt_row = df_wt.loc[idx].values
        ko_row = df_ko.loc[idx].values

        # Compute cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(wt_row, ko_row)
        similarities[idx] = similarity

    return pd.Series(similarities, name="cos_sim")


def compute_row_euclidean_dist(df_wt, df_ko):
    """
    Compute Euclidean distances between corresponding rows of WT and KO dataframes.

    Euclidean distance measures the straight-line distance between two vectors.
    Unlike cosine similarity, this metric is sensitive to both direction and magnitude
    of changes in transition probabilities.

    Parameters
    ----------
    df_wt : pd.DataFrame
        Transition probability matrix for wildtype (WT)
    df_ko : pd.DataFrame
        Transition probability matrix for knockout (KO)

    Returns
    -------
    pd.Series
        Euclidean distances for each row (cell type or metacell),
        with name "euclid_dist"

    Raises
    ------
    AssertionError
        If dataframes have different indices or columns

    Examples
    --------
    >>> # Compare cell type transitions for WT vs gene knockout
    >>> trans_wt = compute_celltype_transitions(adata, "T_fwd_WT")
    >>> trans_ko = compute_celltype_transitions(adata, "T_fwd_tbx16_KO")
    >>> euclidean_dists = compute_row_euclidean_dist(trans_wt, trans_ko)
    >>> print(euclidean_dists)
    NMPs              0.12
    PSM               0.08
    somites           0.45  # <-- Most perturbed
    ...

    >>> # Compute for all genes and timepoints
    >>> dict_euclid_dist = {}
    >>> for data_id in list_datasets:
    ...     oracle = co.load_hdf5(f"{data_id}_oracle.celloracle.oracle")
    ...     trans_wt = compute_celltype_transitions(oracle.adata, "T_fwd_WT")
    ...
    ...     euclid_df = pd.DataFrame(index=trans_wt.index)
    ...     for gene in oracle.active_regulatory_genes:
    ...         trans_ko = compute_celltype_transitions(
    ...             oracle.adata, f"T_fwd_{gene}_KO"
    ...         )
    ...         euclid_df[gene] = compute_row_euclidean_dist(trans_wt, trans_ko)
    ...
    ...     dict_euclid_dist[data_id] = euclid_df

    Notes
    -----
    - Distance = 0.0 means no change from WT
    - Higher values indicate greater perturbation
    - This metric captures both directional and magnitude differences
    - For normalized transition probabilities, values typically range 0-2
    """
    # Ensure both dataframes have the same index and columns
    assert df_wt.index.equals(df_ko.index), "Dataframes must have the same index"
    assert df_wt.columns.equals(df_ko.columns), "Dataframes must have the same columns"

    euclidean_distance = {}
    for idx in df_wt.index:
        wt_row = df_wt.loc[idx].values
        ko_row = df_ko.loc[idx].values

        # Compute euclidean distance
        euclid_dist = euclidean(wt_row, ko_row)
        euclidean_distance[idx] = euclid_dist

    return pd.Series(euclidean_distance, name="euclid_dist")
