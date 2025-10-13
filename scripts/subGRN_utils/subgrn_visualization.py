"""
Visualization functions for subGRN network plots

This module provides comprehensive NetworkX-based visualization of subGRNs
across timepoints with consistent layouts, edge coloring, and node classification.

Note: The main plotting function plot_subgrns_over_time() is ~500 lines and should
be imported directly from the notebook until fully refactored. This module provides
helper functions and simplified plotting utilities.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)

# Configure matplotlib for editable text in PDFs
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def classify_nodes(subgrns: Dict[str, pd.DataFrame]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Classify nodes as TF-only, Target-only, or dual TF/Target

    Parameters
    ----------
    subgrns : Dict[str, pd.DataFrame]
        Dictionary mapping timepoints to subGRN dataframes

    Returns
    -------
    tf_only : Set[str]
        Nodes that only appear as sources (TFs)
    target_only : Set[str]
        Nodes that only appear as targets
    tf_and_target : Set[str]
        Nodes that appear as both sources and targets

    Examples
    --------
    >>> subgrns = {'10': pd.DataFrame({'source': ['gata1'], 'target': ['fli1b']}),
    ...           '15': pd.DataFrame({'source': ['fli1b'], 'target': ['tal1']})}
    >>> tf_only, target_only, dual = classify_nodes(subgrns)
    >>> print(dual)
    {'fli1b'}
    """
    all_sources = set()
    all_targets = set()

    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])

    tf_only = all_sources - all_targets
    target_only = all_targets - all_sources
    tf_and_target = all_sources & all_targets

    return tf_only, target_only, tf_and_target


def get_node_colors(nodes: List[str],
                   tf_only: Set[str],
                   target_only: Set[str],
                   tf_and_target: Set[str]) -> List[str]:
    """
    Assign colors to nodes based on classification

    Parameters
    ----------
    nodes : List[str]
        List of node names
    tf_only : Set[str]
        TF-only nodes
    target_only : Set[str]
        Target-only nodes
    tf_and_target : Set[str]
        Dual role nodes

    Returns
    -------
    List[str]
        List of colors matching node order

    Examples
    --------
    >>> nodes = ['gata1', 'fli1b', 'tal1']
    >>> tf_only = {'gata1'}
    >>> target_only = {'tal1'}
    >>> dual = {'fli1b'}
    >>> colors = get_node_colors(nodes, tf_only, target_only, dual)
    >>> print(colors)
    ['lightcoral', 'orange', 'lightblue']
    """
    colors = []
    for node in nodes:
        if node in tf_only:
            colors.append('lightcoral')
        elif node in target_only:
            colors.append('lightblue')
        elif node in tf_and_target:
            colors.append('orange')
        else:
            colors.append('gray')
    return colors


def compute_edge_widths(edges: List[Tuple[str, str]],
                       edge_weights: Dict[Tuple[str, str], float],
                       max_edge_width: float = 2.0,
                       min_edge_width: float = 0.3) -> List[float]:
    """
    Scale edge widths based on regulatory strength

    Parameters
    ----------
    edges : List[Tuple[str, str]]
        List of (source, target) tuples
    edge_weights : Dict[Tuple[str, str], float]
        Dictionary mapping edges to absolute coefficient values
    max_edge_width : float, default=2.0
        Maximum edge width
    min_edge_width : float, default=0.3
        Minimum edge width

    Returns
    -------
    List[float]
        Scaled widths for each edge

    Examples
    --------
    >>> edges = [('gata1', 'tal1'), ('sox2', 'nes')]
    >>> weights = {('gata1', 'tal1'): 0.5, ('sox2', 'nes'): 0.1}
    >>> widths = compute_edge_widths(edges, weights)
    >>> print(len(widths))
    2
    """
    all_weights = [edge_weights.get(e, 0.1) for e in edges]

    if len(all_weights) == 0:
        return []

    max_weight = max(all_weights)
    min_weight = min(all_weights)

    if max_weight == min_weight:
        return [max_edge_width * 0.6] * len(edges)

    # Scale to [min_edge_width, max_edge_width]
    scaled_widths = []
    for weight in all_weights:
        normalized = (weight - min_weight) / (max_weight - min_weight)
        width = min_edge_width + normalized * (max_edge_width - min_edge_width)
        scaled_widths.append(width)

    return scaled_widths


def separate_edges_by_sign(subgrn: pd.DataFrame,
                           coef_column: str = 'coef_mean') -> Tuple[List, List, Dict]:
    """
    Separate edges into activation (positive) and repression (negative)

    Parameters
    ----------
    subgrn : pd.DataFrame
        SubGRN dataframe with source, target, and coefficient columns
    coef_column : str, default='coef_mean'
        Column name containing coefficient values

    Returns
    -------
    positive_edges : List[Tuple[str, str]]
        Activation edges (positive coefficient)
    negative_edges : List[Tuple[str, str]]
        Repression edges (negative coefficient)
    edge_weights : Dict[Tuple[str, str], float]
        Dictionary mapping edges to absolute coefficient values

    Examples
    --------
    >>> subgrn = pd.DataFrame({
    ...     'source': ['gata1', 'sox2'],
    ...     'target': ['tal1', 'nes'],
    ...     'coef_mean': [0.5, -0.3]
    ... })
    >>> pos, neg, weights = separate_edges_by_sign(subgrn)
    >>> print(len(pos), len(neg))
    1 1
    """
    positive_edges = []
    negative_edges = []
    edge_weights = {}

    for _, row in subgrn.iterrows():
        edge = (row['source'], row['target'])

        if coef_column in row and pd.notna(row[coef_column]):
            coef_value = row[coef_column]
            edge_weights[edge] = abs(coef_value)

            if coef_value > 0:
                positive_edges.append(edge)
            else:
                negative_edges.append(edge)
        else:
            # Default to positive if no coefficient
            edge_weights[edge] = 0.1
            positive_edges.append(edge)

    return positive_edges, negative_edges, edge_weights


def create_legend_elements() -> List:
    """
    Create standard legend elements for subGRN plots

    Returns
    -------
    List
        List of matplotlib Line2D objects for legend

    Examples
    --------
    >>> legend = create_legend_elements()
    >>> print(len(legend))
    6
    """
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
               markersize=10, label='TFs (Active)', alpha=0.9),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
               markersize=8, label='Targets (Active)', alpha=0.9),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=9, label='TF&Target (Active)', alpha=0.9),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='Inactive', alpha=0.3),
        Line2D([0], [0], color='darkred', linewidth=2, label='Activation', alpha=0.8),
        Line2D([0], [0], color='darkblue', linewidth=2, label='Repression', alpha=0.8)
    ]

    return legend_elements


def save_figure_publication_quality(fig: plt.Figure,
                                    filename: str,
                                    dpi: int = 600,
                                    formats: List[str] = ['png', 'pdf']) -> None:
    """
    Save figure in multiple formats with publication-quality settings

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    dpi : int, default=600
        DPI for raster formats
    formats : List[str], default=['png', 'pdf']
        File formats to save

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> save_figure_publication_quality(fig, 'my_plot', formats=['png', 'pdf'])
    """
    from pathlib import Path

    base_path = Path(filename)
    base_name = base_path.stem
    output_dir = base_path.parent

    for fmt in formats:
        output_file = output_dir / f"{base_name}.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight',
                   facecolor='white', format=fmt)
        logger.info(f"Saved: {output_file}")


# NOTE: The full plot_subgrns_over_time() function (~500 lines) from the notebook
# should be imported directly when needed. It's located at line 5169 in
# EDA_extract_subGRN_reg_programs_Take2.py
#
# Usage:
# from notebooks.Fig3_GRN_dynamics.EDA_extract_subGRN_reg_programs_Take2 import plot_subgrns_over_time
#
# Or copy the full function here once finalized.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SubGRN Visualization Module")
    print("Contains helper functions for network visualization")
    print("\nFor full plotting function, import plot_subgrns_over_time from notebook")
