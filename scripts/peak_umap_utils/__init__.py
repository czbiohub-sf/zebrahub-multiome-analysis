"""
Peak UMAP Analysis Utilities

This package contains utilities for analyzing and visualizing peak-level
chromatin accessibility data projected onto UMAP embeddings.

Modules:
- annotation_analysis: Pseudobulk profiling, entropy analysis, accessibility metrics
- visualization: Plotting functions for cluster profiles, grids, and distributions
- color_palettes: Color scheme generators for chromosomes, cell types, and timepoints

Usage:
    from scripts.peak_umap_utils.annotation_analysis import run_metadata_entropy_analysis
    from scripts.peak_umap_utils.visualization import plot_cluster_grid
    from scripts.peak_umap_utils.color_palettes import create_custom_chromosome_palette
"""

from . import annotation_analysis
from . import visualization
from . import color_palettes

__all__ = [
    'annotation_analysis',
    'visualization',
    'color_palettes',
]
