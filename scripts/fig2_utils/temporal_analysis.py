"""
Temporal dynamics analysis for gene expression across developmental timepoints.

This module provides functions for analyzing how gene expression and chromatin
accessibility change over developmental time. Key capabilities include:

- Computing temporal trends (increasing, decreasing, no trend)
- Identifying peak timepoints for each gene
- Calculating dynamic ranges and variance metrics
- Normalizing values for alpha transparency in plots
- Analyzing peak contrast and temporal patterns

These functions support the analysis of RNA-ATAC correlation dynamics across
zebrafish developmental timepoints.

Dependencies:
    - scipy.stats: For linear regression and statistical tests
    - pandas, numpy: For data manipulation
    - scanpy: For AnnData object handling
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats import linregress
from typing import Dict, List, Optional, Tuple, Literal


def compute_temporal_trends(
    row: pd.Series,
    timepoints: Optional[np.ndarray] = None,
    p_threshold: Optional[float] = None,
    classify_trend: bool = True
) -> pd.Series:
    """
    Compute temporal trends via linear regression across timepoints.

    Fits a linear model to gene expression/accessibility across developmental
    timepoints. Returns slope, R-squared, p-value, and trend classification.

    Args:
        row: Series of expression values indexed by timepoints
        timepoints: Numeric timepoint values (default: [10, 12, 14, 16, 19, 24])
        p_threshold: P-value threshold for significance (None = no filtering)
        classify_trend: Whether to classify as increasing/decreasing/no_trend

    Returns:
        Series with keys:
            - slope: Rate of change over time
            - p_value: Statistical significance
            - r_squared: Goodness of fit
            - dynamic_range: Max - min expression
            - std_err: Standard error of slope
            - trend: 'increasing', 'decreasing', or 'no_trend'

    Example:
        >>> rna_trends = rna_celltype_avg.apply(
        ...     compute_temporal_trends, axis=1
        ... )
        >>> increasing_genes = rna_trends[rna_trends['trend'] == 'increasing']
        >>> print(f"Found {len(increasing_genes)} increasing genes")

    Notes:
        - Positive slope = expression increases with time
        - Negative slope = expression decreases with time
        - If p_threshold is None, trend classification ignores p-value
        - Dynamic range is max - min across all timepoints
    """
    if timepoints is None:
        timepoints = np.array([10, 12, 14, 16, 19, 24])

    expression = row.values

    # Compute linear regression
    slope, intercept, r_value, p_value, std_err = linregress(timepoints, expression)

    # Compute R-squared
    r_squared = r_value**2

    # Dynamic range
    dynamic_range = row.max() - row.min()

    # Classify trend
    if classify_trend:
        if p_threshold is not None:
            # Use p-value threshold for classification
            if slope > 0 and p_value < p_threshold:
                trend = 'increasing'
            elif slope < 0 and p_value < p_threshold:
                trend = 'decreasing'
            else:
                trend = 'no_trend'
        else:
            # Classify based on slope direction only
            if slope > 0:
                trend = 'increasing'
            elif slope < 0:
                trend = 'decreasing'
            else:
                trend = 'zero'
    else:
        trend = None

    return pd.Series({
        'slope': slope,
        'p_value': p_value,
        'r_squared': r_squared,
        'dynamic_range': dynamic_range,
        'std_err': std_err,
        'trend': trend
    })


def compute_peaks_and_range(row: pd.Series) -> pd.Series:
    """
    Find peak timepoint and compute expression dynamics for a gene.

    Identifies when gene expression/accessibility is maximal and computes
    summary statistics (mean, variance, dynamic range) across timepoints.

    Args:
        row: Series of expression values indexed by timepoints

    Returns:
        Series with keys:
            - peak_timepoint: Timepoint with maximum expression
            - std_dev: Standard deviation across timepoints
            - variance: Variance across timepoints
            - dynamic_range: Max - min expression
            - mean: Mean expression across timepoints
            - max: Maximum expression value

    Example:
        >>> rna_dynamics = rna_celltype_avg.apply(
        ...     compute_peaks_and_range, axis=1
        ... )
        >>> print(rna_dynamics['peak_timepoint'].value_counts())

    Notes:
        - If multiple timepoints have same max, first is returned
        - Dynamic range captures magnitude of temporal variation
        - Variance captures overall variability across time
    """
    # Find peak timepoint
    peak_timepoint = row.idxmax()

    # Compute metrics
    std_dev = row.std()
    variance = row.var()
    dynamic_range = row.max() - row.min()
    mean = row.mean()
    max_value = row.max()

    return pd.Series({
        'peak_timepoint': peak_timepoint,
        'std_dev': std_dev,
        'variance': variance,
        'dynamic_range': dynamic_range,
        'mean': mean,
        'max': max_value
    })


def compute_peak_metrics(row: pd.Series) -> pd.Series:
    """
    Compute peak contrast metrics for temporal expression patterns.

    Calculates how distinct the peak timepoint is relative to other timepoints
    using peak contrast: (peak_value - mean_others) / std_others. High values
    indicate sharp, well-defined peaks; low values indicate flat profiles.

    Args:
        row: Series of expression values indexed by timepoints (e.g., "10hpf", "12hpf")

    Returns:
        Series with keys:
            - peak_timepoint: Timepoint string with maximum expression
            - peak_time_numeric: Numeric hours post fertilization
            - peak_value: Expression value at peak
            - peak_contrast: (peak - mean_others) / std_others
            - mean_others: Mean of non-peak timepoints
            - std_others: Std dev of non-peak timepoints

    Example:
        >>> rna_peaks = rna_celltype_avg.apply(compute_peak_metrics, axis=1)
        >>> # Genes with sharp peaks
        >>> sharp_peaks = rna_peaks[rna_peaks['peak_contrast'] > 2]
        >>> print(f"Found {len(sharp_peaks)} genes with sharp peaks")

    Notes:
        - Peak contrast is undefined if std_others = 0 (returns 0)
        - Assumes timepoint format like "10hpf", "12hpf", etc.
        - Useful for filtering genes with dynamic temporal patterns
    """
    # Find peak timepoint and value
    peak_timepoint = row.idxmax()
    peak_value = row.max()

    # Mean and std of non-peak points
    other_points = row.drop(peak_timepoint)
    mean_others = other_points.mean()
    std_others = other_points.std()

    # Peak contrast (avoid division by zero)
    peak_contrast = (peak_value - mean_others) / std_others if std_others > 0 else 0

    # Extract numeric timepoint (assumes format like "10hpf")
    peak_time_numeric = float(peak_timepoint.split('hpf')[0])

    return pd.Series({
        'peak_timepoint': peak_timepoint,
        'peak_time_numeric': peak_time_numeric,
        'peak_value': peak_value,
        'peak_contrast': peak_contrast,
        'mean_others': mean_others,
        'std_others': std_others
    })


def normalize_for_alpha(
    values: pd.Series,
    min_alpha: float = 0.2,
    max_alpha: float = 1.0
) -> pd.Series:
    """
    Normalize values to alpha transparency range [min_alpha, max_alpha].

    Linear normalization mapping min(values) -> min_alpha and max(values) -> max_alpha.
    Useful for encoding dynamic range or variance as point transparency in plots.

    Args:
        values: Series or array of values to normalize
        min_alpha: Minimum alpha value (default: 0.2, semi-transparent)
        max_alpha: Maximum alpha value (default: 1.0, fully opaque)

    Returns:
        Normalized values in range [min_alpha, max_alpha]

    Example:
        >>> adata.obs['alpha_rna'] = normalize_for_alpha(
        ...     adata.obs['dynamic_range_rna'], min_alpha=0.3, max_alpha=1.0
        ... )
        >>> sc.pl.umap(adata, color='peak_timepoint', alpha=adata.obs['alpha_rna'])

    Notes:
        - If all values are identical, returns min_alpha for all
        - Preserves relative ordering of input values
        - Commonly used for dynamic range or variance visualization
    """
    min_val = values.min()
    max_val = values.max()

    # Handle case where all values are the same
    if max_val == min_val:
        return pd.Series([min_alpha] * len(values), index=values.index)

    # Linear normalization
    normalized = (values - min_val) / (max_val - min_val)
    return normalized * (max_alpha - min_alpha) + min_alpha


def normalize_for_alpha_robust(
    values: pd.Series,
    min_alpha: float = 0.2,
    max_alpha: float = 1.0,
    lower_percentile: float = 1,
    upper_percentile: float = 99
) -> pd.Series:
    """
    Robust normalization to alpha range using percentiles to handle outliers.

    Clips values to percentile range before normalizing, preventing extreme
    outliers from compressing the majority of the data into a narrow range.

    Args:
        values: Series or array of values to normalize
        min_alpha: Minimum alpha value (default: 0.2)
        max_alpha: Maximum alpha value (default: 1.0)
        lower_percentile: Lower percentile for clipping (default: 1st percentile)
        upper_percentile: Upper percentile for clipping (default: 99th percentile)

    Returns:
        Normalized values in range [min_alpha, max_alpha]

    Example:
        >>> # Use robust normalization for variance with outliers
        >>> adata.obs['alpha_rna'] = normalize_for_alpha_robust(
        ...     adata.obs['variance_rna'],
        ...     lower_percentile=5,
        ...     upper_percentile=95
        ... )

    Notes:
        - Outliers beyond percentile range are clipped to range endpoints
        - More stable than linear normalization when outliers are present
        - Common settings: (1, 99) or (5, 95) for percentiles
    """
    # Get percentile values
    min_val = np.percentile(values, lower_percentile)
    max_val = np.percentile(values, upper_percentile)

    # Clip values to percentile range
    clipped = np.clip(values, min_val, max_val)

    # Handle case where percentiles are identical
    if max_val == min_val:
        return pd.Series([min_alpha] * len(values), index=values.index)

    # Normalize
    normalized = (clipped - min_val) / (max_val - min_val)
    return normalized * (max_alpha - min_alpha) + min_alpha


def analyze_temporal_patterns(
    rna_avg: pd.DataFrame,
    atac_avg: pd.DataFrame,
    timepoints: Optional[np.ndarray] = None,
    p_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Comprehensive temporal pattern analysis for RNA and ATAC data.

    Computes trends, peaks, and dynamics for both modalities and returns
    a combined DataFrame with all metrics.

    Args:
        rna_avg: DataFrame of RNA expression (genes × timepoints)
        atac_avg: DataFrame of ATAC accessibility (genes × timepoints)
        timepoints: Numeric timepoint values (default: [10, 12, 14, 16, 19, 24])
        p_threshold: Significance threshold for trend classification

    Returns:
        DataFrame with columns for RNA and ATAC:
            - {modality}_slope, {modality}_p_value, {modality}_r_squared
            - {modality}_trend ('increasing', 'decreasing', 'no_trend')
            - {modality}_peak_timepoint, {modality}_dynamic_range
            - {modality}_variance, {modality}_mean, etc.

    Example:
        >>> dynamics = analyze_temporal_patterns(
        ...     rna_celltype_avg, atac_celltype_avg
        ... )
        >>> # Find genes with concordant RNA/ATAC trends
        >>> concordant = dynamics[
        ...     (dynamics['rna_trend'] == 'increasing') &
        ...     (dynamics['atac_trend'] == 'increasing')
        ... ]

    Notes:
        - Combines multiple analysis functions into one call
        - All genes must be present in both RNA and ATAC dataframes
        - Prefixes all metric names with 'rna_' or 'atac_'
    """
    if timepoints is None:
        timepoints = np.array([10, 12, 14, 16, 19, 24])

    # Compute trends
    rna_trends = rna_avg.apply(
        lambda row: compute_temporal_trends(row, timepoints, p_threshold),
        axis=1
    )
    atac_trends = atac_avg.apply(
        lambda row: compute_temporal_trends(row, timepoints, p_threshold),
        axis=1
    )

    # Compute peaks and ranges
    rna_peaks = rna_avg.apply(compute_peaks_and_range, axis=1)
    atac_peaks = atac_avg.apply(compute_peaks_and_range, axis=1)

    # Rename columns with modality prefix
    rna_trends = rna_trends.add_prefix('rna_')
    atac_trends = atac_trends.add_prefix('atac_')
    rna_peaks = rna_peaks.add_prefix('rna_')
    atac_peaks = atac_peaks.add_prefix('atac_')

    # Combine all results
    dynamics_summary = pd.concat(
        [rna_trends, atac_trends, rna_peaks, atac_peaks],
        axis=1
    )

    return dynamics_summary


def split_by_modality(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    modality: Literal['rna', 'atac']
) -> pd.DataFrame:
    """
    Extract columns for a specific modality from concatenated data.

    Helper function for separating RNA and ATAC columns when working with
    genes-by-(celltype_timepoint_modality) matrices.

    Args:
        df: DataFrame with columns labeled by celltype_timepoint_modality
        df_info: DataFrame mapping column names to parsed components
        modality: 'rna' or 'atac'

    Returns:
        Subset of df containing only specified modality columns

    Example:
        >>> # Parse column names
        >>> df_info = pd.DataFrame([
        ...     parse_var_names(col) for col in genes_df.columns
        ... ], columns=['celltype', 'timepoint', 'modality'])
        >>>
        >>> rna_df = split_by_modality(genes_df, df_info, 'rna')
        >>> atac_df = split_by_modality(genes_df, df_info, 'atac')

    Notes:
        - Assumes df_info has 'modality' column with values 'rna' or 'atac'
        - Returns view of original dataframe (not a copy)
    """
    cols = df_info[df_info['modality'] == modality].index
    return df[cols]


def average_over_celltypes(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    modality: Literal['rna', 'atac']
) -> pd.DataFrame:
    """
    Average gene expression across cell types for each timepoint.

    For temporal analysis, compute mean expression at each timepoint
    by averaging over all cell types. Reduces celltype_timepoint_modality
    dimension to just timepoints.

    Args:
        df: DataFrame with genes as rows, celltype_timepoint combos as columns
        df_info: DataFrame mapping columns to celltype/timepoint/modality
        modality: 'rna' or 'atac'

    Returns:
        DataFrame with genes as rows, timepoints as columns (e.g., '10hpf', '12hpf')

    Example:
        >>> rna_avg = average_over_celltypes(rna_df, df_info, 'rna')
        >>> print(rna_avg.columns)  # ['10hpf', '12hpf', '14hpf', ...]
        >>> # Now compute temporal trends
        >>> rna_trends = rna_avg.apply(compute_temporal_trends, axis=1)

    Notes:
        - Uses mean by default (consider median_over_celltypes for robustness)
        - Timepoints are sorted in output
        - Assumes df_info has 'timepoint' and 'modality' columns
    """
    unique_timepoints = df_info['timepoint'].unique()
    result_dict = {}

    for tp in sorted(unique_timepoints):
        # Get columns for this timepoint and modality
        mask = (df_info['timepoint'] == tp) & (df_info['modality'] == modality)
        tp_cols = df_info[mask].index

        # Average across cell types
        result_dict[tp] = df[tp_cols].mean(axis=1)

    return pd.DataFrame(result_dict, index=df.index)


def median_over_celltypes(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    modality: Literal['rna', 'atac']
) -> pd.DataFrame:
    """
    Compute median gene expression across cell types for each timepoint.

    Robust alternative to average_over_celltypes() that is less sensitive
    to outlier cell types with extreme expression.

    Args:
        df: DataFrame with genes as rows, celltype_timepoint combos as columns
        df_info: DataFrame mapping columns to celltype/timepoint/modality
        modality: 'rna' or 'atac'

    Returns:
        DataFrame with genes as rows, timepoints as columns

    Example:
        >>> # Use median for robust temporal analysis
        >>> rna_median = median_over_celltypes(rna_df, df_info, 'rna')
        >>> rna_dynamics = rna_median.apply(compute_peaks_and_range, axis=1)

    Notes:
        - More robust than mean when cell type expression varies widely
        - Timepoints are sorted in output
        - Returns numpy median (not pandas Series.median)
    """
    unique_timepoints = df_info['timepoint'].unique()
    result_dict = {}

    for tp in sorted(unique_timepoints):
        # Get columns for this timepoint and modality
        mask = (df_info['timepoint'] == tp) & (df_info['modality'] == modality)
        tp_cols = df_info[mask].index

        # Compute median across cell types
        result_dict[tp] = np.median(df[tp_cols], axis=1)

    return pd.DataFrame(result_dict, index=df.index)


def parse_var_names(var_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse concatenated variable name into components.

    Expected format: {celltype}_{timepoint}_{modality}
    Example: "NMPs_10hpf_rna" -> ("NMPs", "10hpf", "rna")

    Args:
        var_name: Column name in format celltype_timepoint_modality

    Returns:
        Tuple of (celltype, timepoint, modality) or (None, None, None) if invalid

    Example:
        >>> parse_var_names("neural_crest_12hpf_atac")
        ('neural_crest', '12hpf', 'atac')
        >>>
        >>> # Parse all columns
        >>> parsed = [parse_var_names(col) for col in genes_df.columns]
        >>> df_info = pd.DataFrame(parsed, columns=['celltype', 'timepoint', 'modality'])

    Notes:
        - Handles cell types with underscores (e.g., "neural_crest")
        - Modality must be 'rna' or 'atac' as last component
        - Returns (None, None, None) for invalid formats
    """
    parts = var_name.split('_')

    # Check if last part is a valid modality
    if len(parts) >= 3 and parts[-1] in ['rna', 'atac']:
        modality = parts[-1]
        timepoint = parts[-2]  # e.g., '10hpf'
        celltype = '_'.join(parts[:-2])  # Rejoin in case celltype has underscores
        return celltype, timepoint, modality

    return None, None, None
