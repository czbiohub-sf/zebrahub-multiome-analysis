"""
Temporal and spatial analysis of subGRNs

This module provides functions for analyzing how subGRNs change across
timepoints and cell types, computing similarity metrics, and tracking
temporal dynamics.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional
from sklearn.metrics import jaccard_score
import logging

logger = logging.getLogger(__name__)


def analyze_single_timepoint(grn_dict: Dict[Tuple[str, str], pd.DataFrame],
                             timepoint: str,
                             predicted_pairs: Set[Tuple[str, str]],
                             cluster_id: str) -> Dict:
    """
    Analyze how a regulatory program manifests across celltypes at a single timepoint

    Parameters
    ----------
    grn_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary of GRNs keyed by (celltype, timepoint)
    timepoint : str
        Timepoint to analyze (e.g., '15')
    predicted_pairs : Set[Tuple[str, str]]
        Set of predicted TF-target pairs from mesh
    cluster_id : str
        Cluster identifier for logging

    Returns
    -------
    Dict
        Dictionary mapping celltypes to their subGRN analysis:
        {celltype: {'subgrn': DataFrame, 'n_edges': int,
                    'implementation_rate': float, 'mean_strength': float,
                    'implemented_pairs': Set}}

    Examples
    --------
    >>> grn_dict = {('neural_crest', '15'): grn_df1, ('PSM', '15'): grn_df2}
    >>> predicted = {('sox2', 'nes'), ('pax3', 'twist')}
    >>> results = analyze_single_timepoint(grn_dict, '15', predicted, '0_0')
    >>> print(results['neural_crest']['n_edges'])
    2
    """
    logger.info(f"\n=== ANALYZING TIMEPOINT: {timepoint} ===")

    # Get all celltypes at this timepoint
    celltypes_at_tp = [ct for (ct, tp) in grn_dict.keys() if tp == timepoint]
    logger.info(f"Available celltypes: {celltypes_at_tp}")

    # Extract subGRNs for each celltype
    celltype_subgrns = {}
    for celltype in celltypes_at_tp:
        if (celltype, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype, timepoint)]

            # Find which predicted pairs exist in this GRN
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = predicted_pairs & grn_pairs

            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()

            # Compute metrics
            implementation_rate = len(subgrn) / len(predicted_pairs) if len(predicted_pairs) > 0 else 0
            mean_strength = subgrn['coef_abs'].mean() if len(subgrn) > 0 and 'coef_abs' in subgrn.columns else 0

            celltype_subgrns[celltype] = {
                'subgrn': subgrn,
                'n_edges': len(subgrn),
                'implementation_rate': implementation_rate,
                'mean_strength': mean_strength,
                'implemented_pairs': found_pairs
            }

            logger.info(f"{celltype}: {len(subgrn)}/{len(predicted_pairs)} edges ({implementation_rate:.2%})")

    return celltype_subgrns


def compare_celltypes_similarity(celltype_subgrns: Dict,
                                 predicted_pairs: Set[Tuple[str, str]],
                                 timepoint: str,
                                 cluster_id: Optional[str] = None,
                                 plot: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compare how similar celltypes are in implementing a regulatory program

    Uses Jaccard similarity to compare which edges are implemented across celltypes.

    Parameters
    ----------
    celltype_subgrns : Dict
        Output from analyze_single_timepoint()
    predicted_pairs : Set[Tuple[str, str]]
        Set of predicted TF-target pairs
    timepoint : str
        Timepoint being analyzed
    cluster_id : str, optional
        Cluster identifier for plot title
    plot : bool, default=True
        Whether to generate similarity heatmap

    Returns
    -------
    similarity_matrix : np.ndarray
        Pairwise Jaccard similarity matrix (n_celltypes × n_celltypes)
    similarities : List[Dict]
        List of pairwise similarity scores sorted in descending order

    Examples
    --------
    >>> results = analyze_single_timepoint(grn_dict, '15', predicted, '0_0')
    >>> sim_matrix, sim_list = compare_celltypes_similarity(
    ...     results, predicted, '15', '0_0'
    ... )
    >>> print(sim_matrix.shape)
    (n_celltypes, n_celltypes)
    """
    logger.info(f"\n--- Celltype Similarity Analysis at {timepoint} ---")

    celltypes = list(celltype_subgrns.keys())
    n_celltypes = len(celltypes)

    # Create binary implementation matrix
    binary_matrix = []
    for celltype in celltypes:
        implemented_pairs = celltype_subgrns[celltype]['implemented_pairs']
        binary_row = [1 if pair in implemented_pairs else 0 for pair in predicted_pairs]
        binary_matrix.append(binary_row)

    # Compute pairwise similarities
    similarity_matrix = np.zeros((n_celltypes, n_celltypes))
    for i in range(n_celltypes):
        for j in range(n_celltypes):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Jaccard similarity
                similarity_matrix[i, j] = jaccard_score(binary_matrix[i], binary_matrix[j])

    # Plot similarity heatmap
    if plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix,
                   xticklabels=celltypes,
                   yticklabels=celltypes,
                   annot=True, fmt='.2f', cmap='Blues')
        title = f'Celltype Similarity at {timepoint}'
        if cluster_id:
            title += f' - Cluster {cluster_id}'
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # Find most and least similar pairs
    similarities = []
    for i in range(n_celltypes):
        for j in range(i + 1, n_celltypes):
            similarities.append({
                'celltype1': celltypes[i],
                'celltype2': celltypes[j],
                'similarity': similarity_matrix[i, j]
            })

    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

    logger.info("Most similar celltype pairs:")
    for sim in similarities[:3]:
        logger.info(f"  {sim['celltype1']} vs {sim['celltype2']}: {sim['similarity']:.3f}")

    logger.info("Least similar celltype pairs:")
    for sim in similarities[-3:]:
        logger.info(f"  {sim['celltype1']} vs {sim['celltype2']}: {sim['similarity']:.3f}")

    return similarity_matrix, similarities


def compare_across_timepoints(grn_dict: Dict[Tuple[str, str], pd.DataFrame],
                              predicted_pairs: Set[Tuple[str, str]],
                              cluster_id: str) -> Dict[str, Dict]:
    """
    Compare how a regulatory program changes across all available timepoints

    Parameters
    ----------
    grn_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary of GRNs keyed by (celltype, timepoint)
    predicted_pairs : Set[Tuple[str, str]]
        Set of predicted TF-target pairs from mesh
    cluster_id : str
        Cluster identifier

    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping timepoints to celltype analysis results

    Examples
    --------
    >>> results = compare_across_timepoints(grn_dict, predicted, '0_0')
    >>> print(results.keys())
    dict_keys(['05', '10', '15', '20', '30'])
    """
    logger.info(f"\n=== MULTI-TIMEPOINT ANALYSIS ===")

    # Get all available timepoints
    all_timepoints = sorted(set([tp for (ct, tp) in grn_dict.keys()]))
    logger.info(f"Available timepoints: {all_timepoints}")

    # Store results for each timepoint
    timepoint_results = {}

    for timepoint in all_timepoints:
        logger.info(f"\nProcessing timepoint {timepoint}...")
        celltype_subgrns = analyze_single_timepoint(grn_dict, timepoint, predicted_pairs, cluster_id)
        timepoint_results[timepoint] = celltype_subgrns

    return timepoint_results


def track_celltype_across_time(timepoint_results: Dict[str, Dict],
                               cluster_id: str,
                               plot: bool = True) -> Dict[str, List[Dict]]:
    """
    Track how specific celltypes implement a program over developmental time

    Parameters
    ----------
    timepoint_results : Dict[str, Dict]
        Output from compare_across_timepoints()
    cluster_id : str
        Cluster identifier for plot title
    plot : bool, default=True
        Whether to generate temporal evolution plots

    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary mapping celltypes to temporal metrics:
        {celltype: [{'timepoint': str, 'implementation_rate': float,
                     'mean_strength': float, 'n_edges': int}, ...]}

    Examples
    --------
    >>> tp_results = compare_across_timepoints(grn_dict, predicted, '0_0')
    >>> tracking = track_celltype_across_time(tp_results, '0_0')
    >>> print(tracking['neural_crest'][0])
    {'timepoint': '05', 'implementation_rate': 0.15, ...}
    """
    logger.info(f"\n--- Temporal Tracking ---")

    # Get celltypes that appear in multiple timepoints
    all_celltypes = set()
    for tp_results in timepoint_results.values():
        all_celltypes.update(tp_results.keys())

    # Track each celltype across time
    temporal_tracking = {}
    for celltype in all_celltypes:
        temporal_tracking[celltype] = []
        for timepoint in sorted(timepoint_results.keys()):
            if celltype in timepoint_results[timepoint]:
                result = timepoint_results[timepoint][celltype]
                temporal_tracking[celltype].append({
                    'timepoint': timepoint,
                    'implementation_rate': result['implementation_rate'],
                    'mean_strength': result['mean_strength'],
                    'n_edges': result['n_edges']
                })

    # Plot temporal evolution
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Implementation rate over time
        for celltype, data in temporal_tracking.items():
            if len(data) > 1:  # Only plot if celltype appears in multiple timepoints
                timepoints = [d['timepoint'] for d in data]
                impl_rates = [d['implementation_rate'] for d in data]
                ax1.plot(timepoints, impl_rates, marker='o', label=celltype)

        ax1.set_xlabel('Timepoint')
        ax1.set_ylabel('Implementation Rate')
        ax1.set_title(f'Regulatory Program Implementation - Cluster {cluster_id}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Number of edges over time
        for celltype, data in temporal_tracking.items():
            if len(data) > 1:
                timepoints = [d['timepoint'] for d in data]
                n_edges = [d['n_edges'] for d in data]
                ax2.plot(timepoints, n_edges, marker='s', label=celltype)

        ax2.set_xlabel('Timepoint')
        ax2.set_ylabel('Number of Edges')
        ax2.set_title(f'SubGRN Edge Count - Cluster {cluster_id}')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return temporal_tracking


def summarize_analysis(timepoint_results: Dict[str, Dict],
                       temporal_tracking: Dict[str, List[Dict]],
                       cluster_id: str) -> pd.DataFrame:
    """
    Create summary statistics for temporal subGRN analysis

    Parameters
    ----------
    timepoint_results : Dict[str, Dict]
        Output from compare_across_timepoints()
    temporal_tracking : Dict[str, List[Dict]]
        Output from track_celltype_across_time()
    cluster_id : str
        Cluster identifier

    Returns
    -------
    pd.DataFrame
        Summary table with columns: celltype, n_timepoints, mean_impl_rate,
        max_edges, timepoints_present

    Examples
    --------
    >>> summary = summarize_analysis(tp_results, tracking, '0_0')
    >>> print(summary.head())
                        celltype  n_timepoints  mean_impl_rate  max_edges
    0               neural_crest             5           0.234         12
    """
    logger.info(f"\n=== SUMMARY STATISTICS ===")

    summary_data = []
    for celltype, data in temporal_tracking.items():
        if len(data) > 0:
            summary_data.append({
                'celltype': celltype,
                'n_timepoints': len(data),
                'mean_impl_rate': np.mean([d['implementation_rate'] for d in data]),
                'max_impl_rate': np.max([d['implementation_rate'] for d in data]),
                'mean_edges': np.mean([d['n_edges'] for d in data]),
                'max_edges': np.max([d['n_edges'] for d in data]),
                'timepoints_present': [d['timepoint'] for d in data]
            })

    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('mean_impl_rate', ascending=False)

    logger.info(f"\nTop celltypes by mean implementation rate:")
    for idx, row in df_summary.head(5).iterrows():
        logger.info(f"  {row['celltype']}: {row['mean_impl_rate']:.3f} (present in {row['n_timepoints']} timepoints)")

    return df_summary


def analyze_edge_types(grn_dict: Dict[Tuple[str, str], pd.DataFrame],
                       predicted_pairs: Set[Tuple[str, str]],
                       celltype_of_interest: str) -> Dict:
    """
    Analyze activation vs repression edges in subGRN across timepoints

    Parameters
    ----------
    grn_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary of GRNs
    predicted_pairs : Set[Tuple[str, str]]
        Predicted TF-target pairs
    celltype_of_interest : str
        Celltype to analyze

    Returns
    -------
    Dict
        Dictionary with edge type statistics per timepoint

    Examples
    --------
    >>> edge_stats = analyze_edge_types(grn_dict, predicted, 'neural_crest')
    >>> print(edge_stats['15'])
    {'n_activation': 8, 'n_repression': 4, 'total': 12}
    """
    logger.info(f"\n=== Edge Type Analysis for {celltype_of_interest} ===")

    timepoints = sorted(set([tp for (ct, tp) in grn_dict.keys() if ct == celltype_of_interest]))

    edge_type_stats = {}
    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]

            # Extract subGRN
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = predicted_pairs & grn_pairs
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()

            # Count edge types based on coefficient sign
            if len(subgrn) > 0 and 'coef_mean' in subgrn.columns:
                n_activation = (subgrn['coef_mean'] > 0).sum()
                n_repression = (subgrn['coef_mean'] < 0).sum()
            else:
                n_activation = 0
                n_repression = 0

            edge_type_stats[timepoint] = {
                'n_activation': n_activation,
                'n_repression': n_repression,
                'total': len(subgrn),
                'ratio_activation': n_activation / len(subgrn) if len(subgrn) > 0 else 0
            }

            logger.info(f"Timepoint {timepoint}: {n_activation} activation, {n_repression} repression")

    return edge_type_stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Module: subgrn_analysis.py")
    print("Contains functions for temporal and spatial subGRN analysis")
