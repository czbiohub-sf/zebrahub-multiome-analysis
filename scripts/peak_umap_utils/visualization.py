"""
Peak UMAP Visualization Utilities

This module provides visualization functions for analyzing peak clustering results,
including accessibility profiles across cell types and timepoints, distribution plots,
grid layouts, and UMAP overlays.

Functions are organized into the following categories:
- Distribution plots: Chromosome and metadata distributions
- Single cluster plots: Individual cluster accessibility profiles
- Grid layouts: Multi-panel cluster comparisons
- UMAP visualization: Highlighting specific clusters
- Enrichment heatmaps: Statistical enrichment analysis results
- Helper functions: Sorting, color palettes, and temporal regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Optional, List, Dict, Tuple, Union

# Import helper functions from annotation_analysis module
from .annotation_analysis import (
    compute_accessibility_entropy,
    compute_comprehensive_accessibility_metrics,
    _compute_accessibility_metrics
)


# ============================================================================
# Helper Functions
# ============================================================================

def sort_cluster_ids_numerically(cluster_ids: List[Union[str, int]]) -> List[Union[str, int]]:
    """
    Sort cluster IDs numerically instead of lexicographically.

    Handles patterns like 'cluster_0', 'cluster_1', ..., 'cluster_10' to ensure
    proper numerical ordering rather than alphabetical ('0', '1', '10', '2', ...).

    Parameters
    ----------
    cluster_ids : list of str or int
        List of cluster identifiers to sort

    Returns
    -------
    list
        Sorted cluster IDs in numerical order

    Examples
    --------
    >>> cluster_ids = ['cluster_10', 'cluster_2', 'cluster_1']
    >>> sort_cluster_ids_numerically(cluster_ids)
    ['cluster_1', 'cluster_2', 'cluster_10']

    >>> cluster_ids = ['0', '10', '2', '20', '3']
    >>> sort_cluster_ids_numerically(cluster_ids)
    ['0', '2', '3', '10', '20']
    """
    def extract_number(cluster_id):
        # Try to extract number from cluster ID (e.g., 'cluster_10' -> 10)
        match = re.search(r'(\d+)', str(cluster_id))
        return int(match.group(1)) if match else float('inf')

    return sorted(cluster_ids, key=extract_number)


def make_timepoint_palette(timepoints: List[str], cmap_name: str = 'viridis') -> Dict[str, tuple]:
    """
    Build a timepoint-to-color mapping dictionary using a Matplotlib colormap.

    Parameters
    ----------
    timepoints : list of str
        List of timepoint identifiers
    cmap_name : str, default='viridis'
        Name of matplotlib colormap to use

    Returns
    -------
    dict
        Dictionary mapping each timepoint to an RGBA color tuple

    Examples
    --------
    >>> timepoints = ['3somites', '6somites', '12somites', '18somites']
    >>> colors = make_timepoint_palette(timepoints, cmap_name='viridis')
    >>> print(colors['3somites'])
    (0.267004, 0.004874, 0.329415, 1.0)

    Notes
    -----
    - Colors are evenly distributed across the colormap range
    - For single timepoint, uses middle of colormap (0.5)
    """
    cmap = plt.get_cmap(cmap_name)
    n = max(1, len(timepoints))
    colors = [cmap(i/(n-1) if n > 1 else 0.5) for i in range(n)]
    return {tp: col for tp, col in zip(timepoints, colors)}


def fit_temporal_regression(timepoint_order: List[str],
                           accessibility_values: np.ndarray) -> Dict[str, float]:
    """
    Fit linear regression to temporal accessibility data.

    Uses sklearn if available, otherwise falls back to numpy polyfit.

    Parameters
    ----------
    timepoint_order : list of str
        Ordered list of timepoints (used as x-axis indices)
    accessibility_values : array-like
        Accessibility values corresponding to timepoints (y-axis)

    Returns
    -------
    dict
        Dictionary containing:
        - 'slope': Regression slope (change in accessibility per timepoint)
        - 'r_squared': R-squared goodness of fit
        - 'rmsd': Root mean squared deviation
        - 'y_pred': Predicted y-values from regression

    Examples
    --------
    >>> timepoints = ['3somites', '6somites', '12somites', '18somites']
    >>> accessibility = [10.5, 15.2, 22.8, 28.1]
    >>> stats = fit_temporal_regression(timepoints, accessibility)
    >>> print(f"Slope: {stats['slope']:.2f}, R²: {stats['r_squared']:.2f}")
    Slope: 5.88, R²: 0.98

    Notes
    -----
    - Timepoints are converted to numeric indices (0, 1, 2, ...)
    - Requires sklearn for full functionality; numpy fallback available
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # Convert timepoints to numeric (assume they're in chronological order)
        X = np.arange(len(timepoint_order)).reshape(-1, 1)  # Use indices as x-values
        y = np.array(accessibility_values)

        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)

        # Predictions
        y_pred = reg.predict(X)

        # Calculate metrics
        slope = reg.coef_[0]  # slope per timepoint unit
        r_squared = r2_score(y, y_pred)
        rmsd = np.sqrt(np.mean((y - y_pred) ** 2))

        return {
            'slope': slope,
            'r_squared': r_squared,
            'rmsd': rmsd,
            'y_pred': y_pred
        }

    except ImportError:
        print("Warning: sklearn not available. Using numpy fallback.")
        # Fallback to numpy polyfit
        X = np.arange(len(timepoint_order))
        y = np.array(accessibility_values)

        # Fit linear regression using numpy
        coeffs = np.polyfit(X, y, 1)  # 1st degree polynomial (linear)
        slope = coeffs[0]
        y_pred = np.polyval(coeffs, X)

        # Calculate R-squared manually
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmsd = np.sqrt(np.mean((y - y_pred) ** 2))

        return {
            'slope': slope,
            'r_squared': r_squared,
            'rmsd': rmsd,
            'y_pred': y_pred
        }


# ============================================================================
# Distribution Plots
# ============================================================================

def plot_chromosome_distribution_stacked(adata_peaks,
                                        cluster_col: str = 'leiden_coarse',
                                        chrom_col: str = 'chrom',
                                        normalize: bool = True) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create stacked bar plot showing chromosome distribution across peak clusters.

    Visualizes how peaks from different chromosomes are distributed across
    clustering results, useful for detecting chromosome-specific artifacts or
    biological patterns.

    Parameters
    ----------
    adata_peaks : AnnData
        AnnData object containing peak data
    cluster_col : str, default='leiden_coarse'
        Column name in adata_peaks.obs containing cluster assignments
    chrom_col : str, default='chrom'
        Column name in adata_peaks.obs containing chromosome information
    normalize : bool, default=True
        If True, shows proportions (each bar sums to 1)
        If False, shows absolute peak counts

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    plot_data : pd.DataFrame
        Cluster-by-chromosome contingency table used for plotting

    Examples
    --------
    >>> fig, data = plot_chromosome_distribution_stacked(
    ...     adata_peaks,
    ...     cluster_col='leiden_coarse',
    ...     normalize=True
    ... )
    >>> plt.savefig('chrom_distribution.pdf')

    Notes
    -----
    - Expected uniform line (red dashed) shown for normalized plots
    - Uses tab20 + Set3 colormaps for up to 25 chromosomes
    - Useful for QC: clusters enriched for specific chromosomes may indicate artifacts
    """
    # Create contingency table
    cluster_chrom_counts = pd.crosstab(adata_peaks.obs[cluster_col],
                                      adata_peaks.obs[chrom_col])

    if normalize:
        # Convert to proportions (each row sums to 1)
        plot_data = cluster_chrom_counts.div(cluster_chrom_counts.sum(axis=1), axis=0)
        ylabel = 'Proportion of peaks'
        title_suffix = '(Normalized)'
    else:
        plot_data = cluster_chrom_counts
        ylabel = 'Number of peaks'
        title_suffix = '(Absolute counts)'

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create color palette - using tab20 for 25 chromosomes
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(plot_data.columns))))
    if len(plot_data.columns) > 20:
        # Add more colors for chromosomes 21-25
        extra_colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data.columns) - 20))
        colors = np.vstack([colors, extra_colors])

    # Create stacked bar plot
    bottom = np.zeros(len(plot_data.index))

    for i, chrom in enumerate(plot_data.columns):
        ax.bar(plot_data.index, plot_data[chrom],
               bottom=bottom, label=f'Chr {chrom}',
               color=colors[i % len(colors)], alpha=0.8)
        bottom += plot_data[chrom]

    ax.set_xlabel('Peak Cluster (Leiden)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Chromosome Distribution Across Peak Clusters {title_suffix}')

    # Add expected line for normalized plot
    if normalize:
        expected_prop = 1.0 / len(plot_data.columns)  # 1/25 = 0.04 for uniform
        ax.axhline(expected_prop, color='red', linestyle='--', alpha=0.7,
                   label=f'Expected uniform ({expected_prop:.3f})')

    # Customize legend
    if len(plot_data.columns) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, plot_data


# ============================================================================
# Single Cluster Plots
# ============================================================================

def plot_cluster_accessibility_profiles(cluster_metadata_profiles: pd.DataFrame,
                                       clusters_to_plot: Optional[List] = None,
                                       metadata_type: str = 'celltype',
                                       figsize: Tuple[int, int] = (15, 10),
                                       celltype_order: Optional[List[str]] = None,
                                       save_name: Optional[str] = None):
    """
    Plot accessibility profiles for multiple clusters in a grid (minimal, clean version).

    Creates a multi-panel figure showing how peak accessibility varies across
    metadata categories (e.g., cell types) for each cluster.

    Parameters
    ----------
    cluster_metadata_profiles : pd.DataFrame
        Cluster-by-metadata matrix (rows=clusters, columns=metadata categories)
    clusters_to_plot : list or None, default=None
        List of cluster IDs to plot. If None, plots first 6 clusters.
    metadata_type : str, default='celltype'
        Type of metadata being plotted (for axis labels)
    figsize : tuple, default=(15, 10)
        Overall figure size (width, height) in inches
    celltype_order : list or None, default=None
        Optional order for metadata categories on x-axis
    save_name : str or None, default=None
        Filename to save plot (e.g., 'cluster_profiles.pdf')

    Examples
    --------
    >>> # Plot top 6 clusters with default ordering
    >>> plot_cluster_accessibility_profiles(cluster_celltype_profiles)

    >>> # Plot specific clusters in custom order
    >>> clusters = ['0', '5', '10', '15', '20', '25']
    >>> celltype_order = ['Neural', 'Mesoderm', 'Endoderm', 'Epiderm']
    >>> plot_cluster_accessibility_profiles(
    ...     cluster_celltype_profiles,
    ...     clusters_to_plot=clusters,
    ...     celltype_order=celltype_order,
    ...     save_name='selected_clusters.pdf'
    ... )

    Notes
    -----
    - Entropy shown in title quantifies specificity (low) vs. ubiquity (high)
    - Y-axis scaled adaptively based on median accessibility across all clusters
    - Only top 3 values labeled to reduce clutter
    - Grid removes unused subplots automatically
    """
    if clusters_to_plot is None:
        clusters_to_plot = list(cluster_metadata_profiles.index)[:6]

    # Ensure clusters exist in data
    clusters_to_plot = [c for c in clusters_to_plot if c in cluster_metadata_profiles.index]

    if len(clusters_to_plot) == 0:
        print("No valid clusters found to plot")
        return

    # Compute median accessibility for y-axis scaling
    median_accessibility = cluster_metadata_profiles.values.flatten()
    median_accessibility = np.median(median_accessibility[median_accessibility > 0])

    # Create subplots
    n_clusters = len(clusters_to_plot)
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, cluster in enumerate(clusters_to_plot):
        if i >= len(axes):
            break

        ax = axes[i]

        # Get accessibility profile for this cluster
        profile = cluster_metadata_profiles.loc[cluster]

        # Order celltypes if specified, otherwise sort by accessibility
        if celltype_order is not None:
            # Use specified order, fill missing with 0
            ordered_profile = pd.Series(index=celltype_order, dtype=float)
            for celltype in celltype_order:
                if celltype in profile.index:
                    ordered_profile[celltype] = profile[celltype]
                else:
                    ordered_profile[celltype] = 0.0
            profile_to_plot = ordered_profile
        else:
            # Sort by accessibility (descending)
            profile_to_plot = profile.sort_values(ascending=False)

        # Create simple bar plot (single color, no grid)
        ax.bar(range(len(profile_to_plot)), profile_to_plot.values,
               color='steelblue', alpha=0.7)

        # Compute entropy for title
        entropy = compute_accessibility_entropy(profile, normalize=True)

        # Determine y-axis limit
        max_accessibility = profile_to_plot.max()
        if max_accessibility > median_accessibility * 3:  # High accessibility cluster
            y_limit = max_accessibility * 1.1
        else:  # Normal/low accessibility cluster
            y_limit = median_accessibility * 2

        # Formatting
        ax.set_title(f'Cluster {cluster} (entropy={entropy:.3f})',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{metadata_type.capitalize()}', fontsize=10)
        ax.set_ylabel('Mean Accessibility', fontsize=10)
        ax.set_ylim(0, y_limit)

        # Set x-tick labels
        ax.set_xticklabels([])

        # Remove grid
        ax.grid(False)

        # Add value labels on highest bars only
        for j, (cat, val) in enumerate(profile_to_plot.head(3).items()):
            if val > median_accessibility * 0.5:  # Only label significant bars
                ax.text(j, val + y_limit*0.02, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # Hide unused subplots
    for i in range(len(clusters_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_name}")

    plt.show()


def plot_single_cluster(cluster_metadata_profiles: pd.DataFrame,
                       cluster_id: str,
                       celltype_order: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (12, 6),
                       save_path: Optional[str] = None,
                       use_celltype_colors: bool = True,
                       cell_type_color_dict: Optional[Dict[str, str]] = None):
    """
    Plot accessibility profile for a single cluster with comprehensive metrics.

    Creates a standalone figure for one cluster showing accessibility across
    cell types, with entropy, dominance, and CV statistics.

    Parameters
    ----------
    cluster_metadata_profiles : pd.DataFrame
        Cluster-by-celltype matrix
    cluster_id : str
        Cluster ID to plot
    celltype_order : list or None, default=None
        Optional order for cell types on x-axis
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
    save_path : str or None, default=None
        Full path to save figure (e.g., '/path/to/cluster_25_profile.png')
    use_celltype_colors : bool, default=True
        Whether to use cell type-specific colors
    cell_type_color_dict : dict or None, default=None
        Dictionary mapping cell types to colors

    Examples
    --------
    >>> # Plot cluster with default colors
    >>> plot_single_cluster(cluster_celltype_profiles, '25')

    >>> # Plot with custom colors and save
    >>> colors = {'Neural': '#FF0000', 'Mesoderm': '#00FF00'}
    >>> plot_single_cluster(
    ...     cluster_celltype_profiles,
    ...     '25',
    ...     use_celltype_colors=True,
    ...     cell_type_color_dict=colors,
    ...     save_path='figures/cluster_25.pdf'
    ... )

    Notes
    -----
    - Dashed horizontal line shows median accessibility across all clusters
    - Entropy: low = specific, high = ubiquitous
    - Dominance: high = one category dominates
    - CV: coefficient of variation (SD/mean)
    """
    if cluster_id not in cluster_metadata_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return

    # Get profile
    profile = cluster_metadata_profiles.loc[cluster_id]

    # Order celltypes
    if celltype_order is not None:
        ordered_profile = pd.Series(index=celltype_order, dtype=float)
        for celltype in celltype_order:
            ordered_profile[celltype] = profile.get(celltype, 0.0)
        profile_to_plot = ordered_profile
    else:
        profile_to_plot = profile.sort_values(ascending=False)

    # Compute y-limit based on median of all profiles vs current profile max
    overall_median = cluster_metadata_profiles.values.flatten()
    overall_median = pd.Series(overall_median).median()
    profile_max = profile_to_plot.max()
    y_limit = max(overall_median, profile_max) + 10

    # Determine colors
    if use_celltype_colors and cell_type_color_dict is not None:
        colors = [cell_type_color_dict.get(celltype, 'steelblue')
                 for celltype in profile_to_plot.index]
    else:
        colors = 'steelblue'

    # Simple bar plot
    plt.figure(figsize=figsize)
    plt.bar(range(len(profile_to_plot)), profile_to_plot.values,
            color=colors, alpha=0.7, edgecolor="none")

    # Set y-limit
    plt.ylim(0, y_limit)
    # plot the horizontal line for the median value
    plt.axhline(y=overall_median, linestyle="--")

    # Compute comprehensive metrics
    metrics = compute_comprehensive_accessibility_metrics(profile)
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    cv = metrics['cv']

    plt.title(f'Cluster {cluster_id} Celltype Accessibility\n'
              f'entropy={entropy:.3f}, dominance={dominance:.3f}, CV={cv:.1f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Celltype', fontsize=12)
    plt.ylabel('Mean Accessibility', fontsize=12)

    plt.xticks(range(len(profile_to_plot)), profile_to_plot.index,
               rotation=45, ha='right')
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_single_cluster_subplot(cluster_metadata_profiles: pd.DataFrame,
                                cluster_id: str,
                                ax: plt.Axes,
                                celltype_order: Optional[List[str]] = None,
                                use_celltype_colors: bool = True,
                                cell_type_color_dict: Optional[Dict[str, str]] = None):
    """
    Plot accessibility profile for a single cluster on a given axis (for subplot grids).

    Compact version of plot_single_cluster designed for use in multi-panel figures.

    Parameters
    ----------
    cluster_metadata_profiles : pd.DataFrame
        Cluster-by-celltype matrix
    cluster_id : str
        Cluster ID to plot
    ax : matplotlib.axes.Axes
        Axis object to plot on
    celltype_order : list or None, default=None
        Optional order for cell types
    use_celltype_colors : bool, default=True
        Whether to use celltype-specific colors
    cell_type_color_dict : dict or None, default=None
        Dictionary mapping cell types to colors

    Examples
    --------
    >>> fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    >>> for i, cluster_id in enumerate(['0', '1', '2', '3', '4', '5']):
    ...     plot_single_cluster_subplot(
    ...         cluster_profiles, cluster_id, axes.flatten()[i],
    ...         use_celltype_colors=True, cell_type_color_dict=colors
    ...     )
    >>> plt.tight_layout()
    >>> plt.show()

    Notes
    -----
    - X-tick labels hidden to reduce clutter in grids
    - Abbreviated metrics in title (ent, dom, CV) to save space
    - Dashed line shows overall median accessibility
    """
    if cluster_id not in cluster_metadata_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return

    # Get profile
    profile = cluster_metadata_profiles.loc[cluster_id]

    # Order celltypes
    if celltype_order is not None:
        ordered_profile = pd.Series(index=celltype_order, dtype=float)
        for celltype in celltype_order:
            ordered_profile[celltype] = profile.get(celltype, 0.0)
        profile_to_plot = ordered_profile
    else:
        profile_to_plot = profile.sort_values(ascending=False)

    # Compute y-limit based on median of all profiles vs current profile max
    overall_median = cluster_metadata_profiles.values.flatten()
    overall_median = pd.Series(overall_median).median()
    profile_max = profile_to_plot.max()
    y_limit = max(overall_median, profile_max) + 10

    # Determine colors
    if use_celltype_colors and cell_type_color_dict is not None:
        colors = [cell_type_color_dict.get(celltype, 'steelblue')
                 for celltype in profile_to_plot.index]
    else:
        colors = 'steelblue'

    # Simple bar plot on the given axis
    ax.bar(range(len(profile_to_plot)), profile_to_plot.values,
           color=colors, alpha=0.7, edgecolor="none")

    # Set y-limit
    ax.set_ylim(0, y_limit)
    # plot the horizontal line for the median value
    ax.axhline(y=overall_median, linestyle="--", alpha=0.5)

    # Compute comprehensive metrics
    metrics = compute_comprehensive_accessibility_metrics(profile)
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    cv = metrics['cv']

    ax.set_title(f'Cluster {cluster_id}\nent={entropy:.2f}, dom={dominance:.2f}, CV={cv:.1f}',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Celltype', fontsize=8)
    ax.set_ylabel('Mean Accessibility', fontsize=8)

    ax.set_xticklabels([])
    ax.grid(False)


def plot_single_cluster_timepoint_subplot(cluster_timepoint_profiles: pd.DataFrame,
                                         cluster_id: str,
                                         ax: plt.Axes,
                                         timepoint_order: Optional[List[str]] = None,
                                         *,
                                         color_by_timepoint: bool = True,
                                         timepoint_colors: Optional[Dict[str, str]] = None,
                                         default_color: str = "#B0B0B0",
                                         show_regression: bool = True):
    """
    Plot accessibility profile for one cluster across timepoints on a given axis.

    Shows temporal dynamics of peak accessibility with optional linear regression
    to quantify trends.

    Parameters
    ----------
    cluster_timepoint_profiles : pd.DataFrame
        Cluster-by-timepoint matrix
    cluster_id : str
        Cluster ID to plot
    ax : matplotlib.axes.Axes
        Axis object to plot on
    timepoint_order : list or None, default=None
        Optional order for timepoints
    color_by_timepoint : bool, default=True
        Whether to use timepoint-specific colors
    timepoint_colors : dict or None, default=None
        Dictionary mapping timepoints to colors (auto-generated if None)
    default_color : str, default='#B0B0B0'
        Default color if timepoint colors not provided
    show_regression : bool, default=True
        Whether to show linear regression line and temporal statistics

    Examples
    --------
    >>> fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    >>> timepoint_order = ['3somites', '6somites', '12somites', '18somites']
    >>> for i, cluster_id in enumerate(['0', '5', '10', '15']):
    ...     plot_single_cluster_timepoint_subplot(
    ...         cluster_timepoint_profiles,
    ...         cluster_id,
    ...         axes.flatten()[i],
    ...         timepoint_order=timepoint_order,
    ...         show_regression=True
    ...     )
    >>> plt.tight_layout()
    >>> plt.show()

    Notes
    -----
    - Red dashed line shows linear regression fit
    - Trend symbols: ↗ (increasing), ↘ (decreasing), steady (<0.1 slope)
    - RMSD quantifies deviation from linear trend
    - Auto-generates viridis color palette if timepoint_colors not provided
    """
    if cluster_id not in cluster_timepoint_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return

    profile = cluster_timepoint_profiles.loc[cluster_id]

    # Order timepoints
    if timepoint_order is not None:
        ordered = pd.Series(index=timepoint_order, dtype=float)
        for tp in timepoint_order:
            ordered[tp] = profile.get(tp, 0.0)
        profile_to_plot = ordered
    else:
        profile_to_plot = profile.sort_values(ascending=False)

    # y-limit based on overall median vs this cluster's max
    overall_median = pd.Series(cluster_timepoint_profiles.values.flatten()).median()
    profile_max = profile_to_plot.max()
    y_limit = max(overall_median, profile_max) + 10

    # Colors
    if color_by_timepoint:
        if timepoint_colors is None:
            timepoint_colors = make_timepoint_palette(list(profile_to_plot.index))
        colors = [timepoint_colors.get(tp, default_color) for tp in profile_to_plot.index]
    else:
        colors = ["steelblue"] * len(profile_to_plot)

    # Plot bars on the given axis
    x_pos = range(len(profile_to_plot))
    ax.bar(x_pos, profile_to_plot.values, color=colors, alpha=0.85, edgecolor="none")

    # Fit and plot regression line if requested
    regression_stats = None
    if show_regression and len(profile_to_plot) > 1:
        regression_stats = fit_temporal_regression(
            list(profile_to_plot.index),
            profile_to_plot.values
        )
        # Plot regression line
        ax.plot(x_pos, regression_stats['y_pred'], 'r--', linewidth=2, alpha=0.8)

    ax.set_ylim(0, y_limit)
    ax.axhline(y=overall_median, linestyle="--", alpha=0.5, color='gray')

    # Create title with metrics
    metrics = compute_comprehensive_accessibility_metrics(profile)
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    cv = metrics['cv']

    title = f'Cluster {cluster_id}\nent={entropy:.2f}, dom={dominance:.2f}, CV={cv:.1f}'

    if regression_stats and show_regression:
        slope = regression_stats['slope']
        r_sq = regression_stats['r_squared']
        rmsd = regression_stats['rmsd']

        # Interpret slope direction
        if abs(slope) < 0.1:
            trend = "steady"
        elif slope > 0:
            trend = "↗"  # increasing
        else:
            trend = "↘"  # decreasing

        title += f'\nslope={slope:.2f} {trend}, R²={r_sq:.2f}, RMSD={rmsd:.2f}'

    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=8)
    ax.set_ylabel('Mean Accessibility', fontsize=8)

    ax.set_xticklabels([])
    ax.grid(False)


# ============================================================================
# Grid Layouts
# ============================================================================

def plot_cluster_grid(cluster_metadata_profiles: pd.DataFrame,
                     cluster_ids: Optional[List] = None,
                     celltype_order: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (24, 24),
                     use_celltype_colors: bool = True,
                     cell_type_color_dict: Optional[Dict[str, str]] = None,
                     save_path: Optional[str] = None,
                     ncols: int = 6,
                     nrows: int = 6):
    """
    Plot accessibility profiles for multiple clusters in a grid layout.

    Creates a comprehensive multi-panel figure showing celltype accessibility
    patterns for many clusters simultaneously.

    Parameters
    ----------
    cluster_metadata_profiles : pd.DataFrame
        Cluster-by-celltype matrix
    cluster_ids : list or None, default=None
        List of cluster IDs to plot. If None, plots all clusters
    celltype_order : list or None, default=None
        Optional order for cell types
    figsize : tuple, default=(24, 24)
        Overall figure size (width, height) in inches
    use_celltype_colors : bool, default=True
        Whether to use celltype-specific colors
    cell_type_color_dict : dict or None, default=None
        Dictionary mapping cell types to colors
    save_path : str or None, default=None
        Full path to save the figure (e.g., 'figures/all_clusters_grid.pdf')
    ncols : int, default=6
        Number of columns in the grid
    nrows : int, default=6
        Number of rows in the grid

    Examples
    --------
    >>> # Plot all clusters in 6x6 grid
    >>> plot_cluster_grid(cluster_celltype_profiles, save_path='all_clusters.pdf')

    >>> # Plot specific clusters with custom colors
    >>> selected_clusters = [str(i) for i in range(20)]
    >>> celltype_colors = {'Neural': '#FF0000', 'Mesoderm': '#00FF00'}
    >>> plot_cluster_grid(
    ...     cluster_celltype_profiles,
    ...     cluster_ids=selected_clusters,
    ...     use_celltype_colors=True,
    ...     cell_type_color_dict=celltype_colors,
    ...     ncols=5, nrows=4,
    ...     figsize=(20, 16)
    ... )

    Notes
    -----
    - Clusters sorted numerically (e.g., 0, 1, 2, ..., 10, not 0, 1, 10, 2)
    - Maximum plots = ncols × nrows
    - Unused subplots automatically hidden
    - High DPI (300) recommended for publication
    """
    # Get cluster IDs to plot
    if cluster_ids is None:
        cluster_ids = cluster_metadata_profiles.index.tolist()

    # Sort cluster IDs numerically
    cluster_ids = sort_cluster_ids_numerically(cluster_ids)

    # Limit to the number of subplots available
    max_plots = ncols * nrows
    cluster_ids = cluster_ids[:max_plots]

    # Create the subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Make it easier to index

    # Plot each cluster
    for i, cluster_id in enumerate(cluster_ids):
        plot_single_cluster_subplot(
            cluster_metadata_profiles, cluster_id, axes[i],
            celltype_order=celltype_order,
            use_celltype_colors=use_celltype_colors,
            cell_type_color_dict=cell_type_color_dict
        )

    # Hide any unused subplots
    for j in range(len(cluster_ids), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to: {save_path}")

    plt.show()


def plot_timepoint_grid(cluster_timepoint_profiles: pd.DataFrame,
                       cluster_ids: Optional[List] = None,
                       timepoint_order: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (24, 24),
                       color_by_timepoint: bool = True,
                       timepoint_colors: Optional[Dict[str, str]] = None,
                       default_color: str = "#B0B0B0",
                       save_path: Optional[str] = None,
                       ncols: int = 6,
                       nrows: int = 6,
                       show_regression: bool = True):
    """
    Plot timepoint accessibility profiles for multiple clusters in a grid layout.

    Creates a comprehensive view of temporal dynamics across many clusters,
    with optional regression analysis for each cluster.

    Parameters
    ----------
    cluster_timepoint_profiles : pd.DataFrame
        Cluster-by-timepoint matrix
    cluster_ids : list or None, default=None
        List of cluster IDs to plot. If None, plots all clusters
    timepoint_order : list or None, default=None
        Optional order for timepoints (e.g., ['3somites', '6somites', ...])
    figsize : tuple, default=(24, 24)
        Overall figure size (width, height) in inches
    color_by_timepoint : bool, default=True
        Whether to use timepoint-specific colors
    timepoint_colors : dict or None, default=None
        Dictionary mapping timepoints to colors (auto-generated if None)
    default_color : str, default='#B0B0B0'
        Default color if timepoint colors not provided
    save_path : str or None, default=None
        Full path to save the figure
    ncols : int, default=6
        Number of columns in the grid
    nrows : int, default=6
        Number of rows in the grid
    show_regression : bool, default=True
        Whether to show linear regression lines and temporal statistics

    Examples
    --------
    >>> # Plot all clusters with temporal regression
    >>> timepoint_order = ['3somites', '6somites', '12somites', '18somites', '24somites']
    >>> plot_timepoint_grid(
    ...     cluster_timepoint_profiles,
    ...     timepoint_order=timepoint_order,
    ...     show_regression=True,
    ...     save_path='temporal_dynamics_grid.pdf'
    ... )

    >>> # Plot subset without regression
    >>> selected = [str(i) for i in range(12)]
    >>> plot_timepoint_grid(
    ...     cluster_timepoint_profiles,
    ...     cluster_ids=selected,
    ...     ncols=4, nrows=3,
    ...     show_regression=False,
    ...     figsize=(16, 12)
    ... )

    Notes
    -----
    - Red dashed lines show linear regression fits when show_regression=True
    - Colors auto-generated using viridis colormap if not provided
    - Useful for identifying clusters with strong temporal trends
    - Trend symbols: ↗ (activation), ↘ (repression), steady (constitutive)
    """
    # Get cluster IDs to plot
    if cluster_ids is None:
        cluster_ids = cluster_timepoint_profiles.index.tolist()

    # Sort cluster IDs numerically
    cluster_ids = sort_cluster_ids_numerically(cluster_ids)

    # Limit to the number of subplots available
    max_plots = ncols * nrows
    cluster_ids = cluster_ids[:max_plots]

    # Create timepoint colors if needed and not provided
    if color_by_timepoint and timepoint_colors is None:
        all_timepoints = cluster_timepoint_profiles.columns.tolist()
        if timepoint_order is not None:
            all_timepoints = timepoint_order
        timepoint_colors = make_timepoint_palette(all_timepoints)

    # Create the subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Make it easier to index

    # Plot each cluster
    for i, cluster_id in enumerate(cluster_ids):
        plot_single_cluster_timepoint_subplot(
            cluster_timepoint_profiles,
            cluster_id,
            axes[i],
            timepoint_order=timepoint_order,
            color_by_timepoint=color_by_timepoint,
            timepoint_colors=timepoint_colors,
            default_color=default_color,
            show_regression=show_regression
        )

    # Hide any unused subplots
    for j in range(len(cluster_ids), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timepoint grid plot saved to: {save_path}")

    plt.show()


# ============================================================================
# Profile Plots with Error Bars
# ============================================================================

def plot_cluster_profile(cluster_metadata_profiles: pd.DataFrame,
                         cluster_id: str,
                         metadata_info: Dict,
                         cluster_metadata_sems: Optional[pd.DataFrame] = None,
                         figsize: Tuple[int, int] = (8, 5),
                         show_metrics: bool = True,
                         show_error_bars: bool = True,
                         return_fig: bool = True) -> Optional[plt.Figure]:
    """
    Plot accessibility profile for a single cluster with optional error bars.

    Enhanced version with standard error of the mean (SEM) visualization for
    statistical rigor.

    Parameters
    ----------
    cluster_metadata_profiles : pd.DataFrame
        Cluster-by-metadata matrix (mean accessibility values)
    cluster_id : str
        Cluster ID to plot
    metadata_info : dict
        Dictionary containing 'colors' (color mapping) and optionally 'order' (category order)
    cluster_metadata_sems : pd.DataFrame or None, default=None
        Cluster-by-metadata matrix (SEM values)
    figsize : tuple, default=(8, 5)
        Figure size (width, height) in inches
    show_metrics : bool, default=True
        Whether to show entropy/dominance in title
    show_error_bars : bool, default=True
        Whether to show error bars (requires cluster_metadata_sems)
    return_fig : bool, default=True
        If True, returns figure object; if False, displays and returns None

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if return_fig=True, otherwise None

    Examples
    --------
    >>> metadata_info = {
    ...     'colors': {'Neural': '#FF0000', 'Mesoderm': '#00FF00'},
    ...     'order': ['Neural', 'Mesoderm', 'Endoderm', 'Epiderm']
    ... }
    >>> fig = plot_cluster_profile(
    ...     cluster_profiles, '25', metadata_info,
    ...     cluster_metadata_sems=sems, show_error_bars=True
    ... )
    >>> fig.savefig('cluster_25_with_sem.pdf')

    Notes
    -----
    - Error bars represent ±1 SEM
    - Dashed line shows median accessibility across all clusters
    - Cap size of 3 points for error bar ends
    """
    if cluster_id not in cluster_metadata_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return None

    # Get and order profile
    profile = cluster_metadata_profiles.loc[cluster_id]
    if 'order' in metadata_info and metadata_info['order']:
        ordered_profile = profile.reindex(metadata_info['order'], fill_value=0)
    else:
        ordered_profile = profile.sort_values(ascending=False)

    # Get corresponding SEMs if available
    sems = None
    if show_error_bars and cluster_metadata_sems is not None:
        if cluster_id in cluster_metadata_sems.index:
            cluster_sems = cluster_metadata_sems.loc[cluster_id]
            if 'order' in metadata_info and metadata_info['order']:
                sems = cluster_sems.reindex(metadata_info['order'], fill_value=0)
            else:
                sems = cluster_sems.reindex(ordered_profile.index, fill_value=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    colors = [metadata_info['colors'].get(cat, 'steelblue')
              for cat in ordered_profile.index]

    # Plot bars with error bars
    x_pos = range(len(ordered_profile))
    bars = ax.bar(x_pos, ordered_profile.values, color=colors, alpha=0.8,
                  yerr=sems.values if sems is not None else None,
                  capsize=3, error_kw={'alpha': 0.7})

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ordered_profile.index, rotation=45, ha='right')
    ax.set_ylabel('Mean Accessibility')

    # Title with metrics
    if show_metrics:
        metrics = _compute_accessibility_metrics(profile)
        title = f'Cluster {cluster_id}\nEntropy: {metrics["entropy"]:.3f}, Dominance: {metrics["dominance"]:.3f}'
    else:
        title = f'Cluster {cluster_id}'

    ax.set_title(title, fontweight='bold')
    ax.grid(False)

    # Add horizontal line at median
    median_val = cluster_metadata_profiles.values.flatten()
    median_val = np.median(median_val[median_val > 0])
    ax.axhline(y=median_val, linestyle='--', alpha=0.5, color='gray')

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
        return None


def _plot_on_axis(cluster_metadata_profiles: pd.DataFrame,
                 cluster_id: str,
                 metadata_info: Dict,
                 ax: plt.Axes,
                 cluster_metadata_sems: Optional[pd.DataFrame] = None,
                 show_error_bars: bool = True):
    """
    Plot single cluster profile on given axis with optional error bars.

    Helper function for creating multi-panel figures with error bars.

    Parameters
    ----------
    cluster_metadata_profiles : pd.DataFrame
        Cluster-by-metadata matrix (means)
    cluster_id : str
        Cluster ID to plot
    metadata_info : dict
        Dictionary with 'colors' and optionally 'order'
    ax : matplotlib.axes.Axes
        Axis to plot on
    cluster_metadata_sems : pd.DataFrame or None, default=None
        Cluster-by-metadata matrix (SEMs)
    show_error_bars : bool, default=True
        Whether to show error bars

    Examples
    --------
    >>> fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    >>> metadata_info = {'colors': celltype_colors, 'order': celltype_order}
    >>> for i, cluster_id in enumerate(['0', '5', '10', '15']):
    ...     _plot_on_axis(
    ...         cluster_profiles, cluster_id, metadata_info,
    ...         axes.flatten()[i], cluster_sems, show_error_bars=True
    ...     )
    """
    if cluster_id not in cluster_metadata_profiles.index:
        ax.set_visible(False)
        return

    profile = cluster_metadata_profiles.loc[cluster_id]

    # Order profile
    if 'order' in metadata_info and metadata_info['order']:
        ordered_profile = profile.reindex(metadata_info['order'], fill_value=0)
    else:
        ordered_profile = profile.sort_values(ascending=False)

    # Get corresponding SEMs if available
    sems = None
    if show_error_bars and cluster_metadata_sems is not None:
        if cluster_id in cluster_metadata_sems.index:
            cluster_sems = cluster_metadata_sems.loc[cluster_id]
            if 'order' in metadata_info and metadata_info['order']:
                sems = cluster_sems.reindex(metadata_info['order'], fill_value=0)
            else:
                sems = cluster_sems.reindex(ordered_profile.index, fill_value=0)

    # Colors
    colors = [metadata_info['colors'].get(cat, 'steelblue')
              for cat in ordered_profile.index]

    # Plot with error bars
    x_pos = range(len(ordered_profile))
    ax.bar(x_pos, ordered_profile.values, color=colors, alpha=0.8,
           yerr=sems.values if sems is not None else None,
           capsize=2, error_kw={'alpha': 0.6, 'linewidth': 1})

    # Metrics for title
    metrics = _compute_accessibility_metrics(profile)
    title = f'Cluster {cluster_id}\nE={metrics["entropy"]:.2f}, D={metrics["dominance"]:.2f}'

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ordered_profile.index, rotation=45, ha='right', fontsize=8)
    ax.grid(False)


# ============================================================================
# Heatmap Visualizations
# ============================================================================

def plot_cluster_heatmap(cluster_profiles: pd.DataFrame,
                         cluster_id: Union[str, int],
                         celltype_orders: List[str],
                         cell_type_color_dict: Dict[str, str],
                         figsize: Tuple[int, int] = (12, 8),
                         cmap: str = 'RdBu_r',
                         save_path: Optional[str] = None,
                         show_values: bool = False,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None) -> Optional[plt.Figure]:
    """
    Generate a timepoint × celltype heatmap for a specific cluster.

    Creates a 2D visualization showing how accessibility varies across both
    cell types and developmental timepoints.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        Cluster-by-pseudobulk matrix where pseudobulk groups are named as
        'celltype_timepoint' (e.g., 'Neural_6somites')
    cluster_id : str or int
        Cluster ID to plot
    celltype_orders : list of str
        List of cell types in desired display order
    cell_type_color_dict : dict
        Dictionary mapping cell types to colors (shown as colored bars)
    figsize : tuple, default=(12, 8)
        Figure size (width, height) in inches
    cmap : str, default='RdBu_r'
        Colormap name (diverging maps recommended)
    save_path : str or None, default=None
        Path to save figure
    show_values : bool, default=False
        Whether to show numerical values in heatmap cells
    vmin, vmax : float or None, default=None
        Colormap limits (auto-scaled if None)

    Returns
    -------
    matplotlib.figure.Figure or None
        Heatmap figure if successful, None if cluster not found

    Examples
    --------
    >>> celltype_order = ['Neural', 'Mesoderm', 'Endoderm', 'Epiderm']
    >>> celltype_colors = {'Neural': '#FF0000', 'Mesoderm': '#00FF00'}
    >>> fig = plot_cluster_heatmap(
    ...     cluster_profiles, '25', celltype_order, celltype_colors,
    ...     cmap='RdBu_r', save_path='cluster_25_heatmap.pdf'
    ... )

    Notes
    -----
    - Timepoints sorted numerically (e.g., 3somites, 6somites, 12somites)
    - Colored bars at bottom indicate cell type colors
    - Useful for identifying cell type-specific temporal patterns
    - Pseudobulk groups must contain '_' and end with 'somites'
    """
    from matplotlib.patches import Rectangle

    # Convert cluster_id to string for consistency
    cluster_id = str(cluster_id)

    if cluster_id not in cluster_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return None

    # Get accessibility values for this cluster
    cluster_data = cluster_profiles.loc[cluster_id]

    # Parse pseudobulk group names into timepoint and celltype
    timepoint_celltype_data = []

    for group, accessibility in cluster_data.items():
        # Split by last underscore to separate celltype from timepoint
        if '_' in group and 'somites' in group:
            parts = group.rsplit('_', 1)  # Split from right, only once
            if len(parts) == 2 and parts[1].endswith('somites'):
                celltype = parts[0]
                timepoint = parts[1]
                timepoint_celltype_data.append({
                    'celltype': celltype,
                    'timepoint': timepoint,
                    'accessibility': accessibility
                })

    if not timepoint_celltype_data:
        print(f"No valid pseudobulk groups found for cluster {cluster_id}")
        return None

    # Convert to DataFrame and pivot
    heatmap_df = pd.DataFrame(timepoint_celltype_data)
    heatmap_matrix = heatmap_df.pivot(index='timepoint', columns='celltype', values='accessibility')
    heatmap_matrix = heatmap_matrix.fillna(0)  # Fill missing combinations with 0

    # Sort timepoints numerically
    timepoint_order = sorted(heatmap_matrix.index,
                           key=lambda x: int(x.replace('somites', '')) if x.replace('somites', '').isdigit() else 0)
    heatmap_matrix = heatmap_matrix.reindex(timepoint_order)

    # Order celltypes according to biological order
    available_celltypes = [ct for ct in celltype_orders if ct in heatmap_matrix.columns]
    # Add any celltypes not in the predefined order
    remaining_celltypes = [ct for ct in heatmap_matrix.columns if ct not in celltype_orders]
    final_celltype_order = available_celltypes + sorted(remaining_celltypes)
    heatmap_matrix = heatmap_matrix[final_celltype_order]

    # Create figure with extra space at bottom for color blocks
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap using imshow
    im = ax.imshow(heatmap_matrix.values, cmap=cmap, aspect='auto',
                   interpolation='nearest', vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_xticks(range(len(heatmap_matrix.columns)))
    ax.set_xticklabels(heatmap_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_matrix.index)))
    ax.set_yticklabels(heatmap_matrix.index)

    # Add colored blocks for celltypes at bottom
    for i, celltype in enumerate(heatmap_matrix.columns):
        if celltype in cell_type_color_dict:
            color = cell_type_color_dict[celltype]
            # Add rectangle at bottom of plot
            rect = Rectangle((i-0.4, -0.8), 0.8, 0.3,
                           facecolor=color, edgecolor='black', linewidth=0.5,
                           transform=ax.transData, clip_on=False)
            ax.add_patch(rect)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accessibility', rotation=270, labelpad=20)

    # Add values to cells if requested
    if show_values:
        for i in range(len(heatmap_matrix.index)):
            for j in range(len(heatmap_matrix.columns)):
                value = heatmap_matrix.iloc[i, j]
                if value > 0:  # Only show non-zero values
                    # Choose text color based on background
                    matrix_range = heatmap_matrix.values.max() - heatmap_matrix.values.min()
                    threshold = heatmap_matrix.values.min() + matrix_range * 0.5
                    text_color = 'white' if value < threshold else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                           color=text_color, fontsize=8)

    ax.set_title(f'Cluster {cluster_id}: Timepoint × Celltype Accessibility',
                fontweight='bold', pad=20)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Timepoint')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    return fig


# ============================================================================
# UMAP Visualization
# ============================================================================

def plot_umap_variable_size(adata,
                           cluster_col: str,
                           cluster_id: Union[str, int],
                           true_color: str = "steelblue",
                           false_color: str = 'lightgrey',
                           true_size: float = 1,
                           false_size: float = 0.2,
                           alpha: float = 0.7,
                           figsize: Tuple[int, int] = (8, 6),
                           save_path: Optional[str] = None):
    """
    Plot UMAP with different point sizes for highlighted vs background clusters.

    Creates a UMAP visualization emphasizing a specific cluster while showing
    context of other clusters.

    Parameters
    ----------
    adata : AnnData
        AnnData object with UMAP coordinates in obsm['X_umap']
    cluster_col : str
        Column name in adata.obs containing cluster assignments
    cluster_id : int or str
        Cluster to highlight
    true_color : str, default='steelblue'
        Color for highlighted points (use adata.uns[cluster_col + '_colors'][i]
        for original scanpy color)
    false_color : str, default='lightgrey'
        Color for background points
    true_size : float, default=1
        Point size for highlighted cluster
    false_size : float, default=0.2
        Point size for background clusters
    alpha : float, default=0.7
        Point transparency (0-1)
    figsize : tuple, default=(8, 6)
        Figure size (width, height) in inches
    save_path : str or None, default=None
        Path to save figure

    Examples
    --------
    >>> # Basic usage with default colors
    >>> plot_umap_variable_size(adata, 'leiden_coarse', cluster_id=25)

    >>> # Use original scanpy color for cluster
    >>> cluster_idx = 25
    >>> original_color = adata.uns['leiden_coarse_colors'][cluster_idx]
    >>> plot_umap_variable_size(
    ...     adata, 'leiden_coarse', cluster_id=25,
    ...     true_color=original_color,
    ...     true_size=2, false_size=0.1,
    ...     save_path='cluster_25_umap.pdf'
    ... )

    Notes
    -----
    - Background points rendered with 60% of specified alpha for further de-emphasis
    - Axes and spines removed for cleaner scanpy-style appearance
    - Legend shows point counts for both highlighted and background
    """
    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']

    # Create boolean mask
    mask = adata.obs[cluster_col] == cluster_id

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot background points (smaller, grey)
    if np.sum(~mask) > 0:  # If there are background points
        ax.scatter(
            umap_coords[~mask, 0],
            umap_coords[~mask, 1],
            c=false_color,
            s=false_size,
            alpha=alpha * 0.6,  # Make background even more transparent
            edgecolors='none',
            label=f'Other clusters (n={np.sum(~mask)})'
        )

    # Plot highlighted points (larger, colored)
    if np.sum(mask) > 0:  # If there are highlighted points
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=true_color,
            s=true_size,
            alpha=alpha,
            edgecolors='none',
            label=f'Cluster {cluster_id} (n={np.sum(mask)})'
        )

    # Styling
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'Leiden Cluster {cluster_id} Highlighted')
    ax.legend()

    # Remove ticks for cleaner look (scanpy style)
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Variable size plot saved to: {save_path}")

    plt.show()


# ============================================================================
# Enrichment Analysis Visualization
# ============================================================================

def plot_enrichment_heatmap(enrichment_df: pd.DataFrame,
                           metric: str = 'log2_odds_ratio',
                           significance_threshold: float = 0.05,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot heatmap of celltype enrichment across clusters.

    Visualizes statistical enrichment analysis results showing which cell types
    are over/under-represented in each cluster.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Enrichment analysis results with columns: 'celltype', 'cluster', metric, 'p_value_adj'
    metric : str, default='log2_odds_ratio'
        Column name for the metric to visualize (e.g., 'log2_odds_ratio', 'odds_ratio')
    significance_threshold : float, default=0.05
        Adjusted p-value threshold for masking non-significant values
    figsize : tuple, default=(12, 8)
        Figure size (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
        Heatmap figure

    Examples
    --------
    >>> # Basic enrichment heatmap
    >>> fig = plot_enrichment_heatmap(enrichment_df)
    >>> plt.savefig('enrichment_heatmap.pdf')

    >>> # Use odds ratio instead of log2 odds ratio
    >>> fig = plot_enrichment_heatmap(
    ...     enrichment_df,
    ...     metric='odds_ratio',
    ...     significance_threshold=0.01,
    ...     figsize=(14, 10)
    ... )

    Notes
    -----
    - Non-significant values (p_adj > threshold) are masked (shown as white/gray)
    - Red indicates enrichment (positive log2 odds ratio)
    - Blue indicates depletion (negative log2 odds ratio)
    - Values annotated in cells (formatted to 2 decimal places)
    """
    # Create pivot table
    pivot_data = enrichment_df.pivot(index='celltype', columns='cluster', values=metric)
    pivot_pval = enrichment_df.pivot(index='celltype', columns='cluster', values='p_value_adj')

    # Create significance mask
    sig_mask = pivot_pval > significance_threshold

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(pivot_data,
                mask=sig_mask,  # Mask non-significant values
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': f'{metric}'},
                ax=ax)

    ax.set_title(f'Celltype Enrichment Across Leiden Clusters\n({metric}, adj. p < {significance_threshold})')
    ax.set_xlabel('Leiden Cluster')
    ax.set_ylabel('Cell Type')

    plt.tight_layout()
    return fig
