"""
Pathway enrichment visualization for gene set analysis.

This module provides plotting functions for FishEnrichR pathway enrichment results,
enabling visualization of enriched biological pathways for gene clusters.

Dependencies:
    - pandas: For reading enrichment results
    - matplotlib: For plotting
    - seaborn: For aesthetics (optional)
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def plot_pathway_enrichment(
    file_path: str,
    output_path: str,
    top_n: int = 10,
    p_threshold: float = 0.05,
    sig_color: str = 'purple',
    nonsig_color: str = 'grey'
) -> None:
    """
    Plot pathway enrichment results as horizontal bar chart.

    Creates a horizontal bar plot showing the top N enriched pathways,
    colored by statistical significance. Higher combined scores indicate
    stronger enrichment.

    Args:
        file_path: Path to tab-separated enrichment results file
        output_path: Path to save output plot
        top_n: Number of top pathways to display (default: 10)
        p_threshold: P-value threshold for significance (default: 0.05)
        sig_color: Color for significant pathways (default: 'purple')
        nonsig_color: Color for non-significant pathways (default: 'grey')

    Returns:
        None (saves plot to output_path)

    Example:
        >>> plot_pathway_enrichment(
        ...     'cluster_0_enrichment.txt',
        ...     'cluster_0_pathways.png',
        ...     top_n=15
        ... )

    Notes:
        - Input file format: tab-separated with 'Term', 'Combined Score', 'P-value' columns
        - Pathway terms are expected in format "{Pathway}_WP{ID}"
        - Combined Score = -log10(p-value) × z-score
        - Plots are saved with high DPI (300) for publication quality
    """
    # Read tab-separated enrichment results
    df = pd.read_csv(file_path, sep='\t')

    # Sort by Combined Score and get top N pathways
    df = df.sort_values('Combined Score', ascending=True).tail(top_n)

    # Assign colors based on p-value significance
    colors = [
        nonsig_color if p > p_threshold else sig_color
        for p in df['P-value']
    ]

    # Create horizontal bar plot
    plt.figure(figsize=(12, 8))
    plt.barh(df['Term'], df['Combined Score'], color=colors, alpha=0.6)

    # Customize plot
    plt.title('WikiPathways 2018 Enrichment', fontsize=14, pad=20)
    plt.xlabel('Combined Score', fontsize=12)

    # Clean pathway names (remove WP suffix)
    cleaned_terms = [term.split('_WP')[0] for term in df['Term']]
    plt.yticks(range(len(df['Term'])), cleaned_terms, fontsize=10)

    # Add legend for significance
    from matplotlib.patches import Rectangle
    legend_handles = [
        Rectangle((0, 0), 1, 1, fc=sig_color, alpha=0.6),
        Rectangle((0, 0), 1, 1, fc=nonsig_color, alpha=0.6)
    ]
    plt.legend(
        legend_handles,
        [f'p < {p_threshold}', f'p > {p_threshold}'],
        loc='best'
    )

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
