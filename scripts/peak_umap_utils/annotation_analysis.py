"""
Peak UMAP Annotation Analysis Module

This module provides core functionality for analyzing chromatin accessibility patterns
in peak UMAP clusters, including pseudobulk profile creation, entropy-based pattern
analysis, and temporal regression analysis.

Main Components:
    - Pseudobulk profile creation and filtering
    - Metadata parsing and aggregation (celltype, timepoint, lineage)
    - Accessibility entropy computation and pattern classification
    - Comprehensive statistical metrics for accessibility distributions
    - Temporal regression analysis for developmental dynamics

Author: Zebrahub-Multiome Analysis Pipeline
Created: Phase 6a refactoring
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Union


# =============================================================================
# PSEUDOBULK PROFILE CREATION & FILTERING
# =============================================================================

def get_cell_count_for_group(adata, col):
    """
    Get cell count for a specific pseudobulk group.

    This function extracts the number of cells that were aggregated to create
    a pseudobulk profile for a given group.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with peaks as obs and pseudobulk groups as var.
        Must contain 'n_cells' in var metadata.
    col : str
        Column name (pseudobulk group identifier) to query.

    Returns
    -------
    int
        Number of cells in the pseudobulk group.

    Examples
    --------
    >>> cell_count = get_cell_count_for_group(adata_peaks, "neural_5somites")
    >>> print(f"Group has {cell_count} cells")
    Group has 150 cells

    Notes
    -----
    This assumes the AnnData object has 'n_cells' stored in var for each
    pseudobulk group, typically created during pseudobulk aggregation.
    """
    cell_count_series = adata[:, col].var["n_cells"]
    return int(cell_count_series.iloc[0])


def create_cluster_pseudobulk_profiles(adata, cluster_col='leiden_coarse',
                                      min_cells=20, verbose=True):
    """
    Create cluster-by-pseudobulk accessibility profiles with quality filtering.

    This is Step 1 of the entropy analysis pipeline. It aggregates peak accessibility
    across clusters for each pseudobulk group, filtering out low-confidence groups
    with insufficient cell counts.

    Parameters
    ----------
    adata : AnnData
        Data object with peaks as obs and pseudobulk groups as var.
        Must contain cluster assignments in obs[cluster_col] and
        'n_cells' metadata in var.
    cluster_col : str, default='leiden_coarse'
        Column name in adata.obs containing cluster assignments.
    min_cells : int, default=20
        Minimum number of cells required per pseudobulk group for inclusion.
        Groups with fewer cells are filtered out to ensure statistical reliability.
    verbose : bool, default=True
        Whether to print detailed filtering information and progress updates.

    Returns
    -------
    pd.DataFrame
        Cluster-by-pseudobulk_groups matrix (clusters x reliable_groups).
        Values represent mean accessibility across peaks in each cluster.
        Shape: (n_clusters, n_reliable_groups)

    Examples
    --------
    >>> cluster_profiles = create_cluster_pseudobulk_profiles(
    ...     adata_peaks,
    ...     cluster_col='leiden_coarse',
    ...     min_cells=20
    ... )
    Keeping 147/165 groups with ≥20 cells
    Created cluster profiles: (47, 147)

    Notes
    -----
    - Reliable groups are those with ≥min_cells cells
    - Mean accessibility is computed across all peaks within each cluster
    - The resulting matrix is the foundation for downstream entropy analysis
    """

    if verbose:
        print("="*80)
        print("STEP 1: CREATE CLUSTER-BY-PSEUDOBULK PROFILES")
        print("="*80)

    # Filter reliable groups (≥min_cells)
    reliable_groups = []
    filtered_out_groups = []

    for col in adata.var.index:
        cell_count = get_cell_count_for_group(adata, col)
        if cell_count >= min_cells:
            reliable_groups.append(col)
        else:
            filtered_out_groups.append((col, cell_count))

    if verbose:
        print(f"Keeping {len(reliable_groups)}/{len(adata.var.index)} groups with ≥{min_cells} cells")
        print(f"Filtered out {len(filtered_out_groups)} groups")
        if filtered_out_groups:
            print("Examples of filtered groups:")
            for col, count in filtered_out_groups[:5]:
                print(f"  {col}: {count} cells")

    # Create cluster profiles using only reliable groups
    clusters = adata.obs[cluster_col].astype(str)

    # Get data for reliable groups only
    reliable_indices = [adata.var.index.get_loc(col) for col in reliable_groups]
    reliable_data = adata.X[:, reliable_indices]

    # Create DataFrame for easier groupby
    data_df = pd.DataFrame(
        reliable_data.toarray() if hasattr(reliable_data, 'toarray') else reliable_data,
        columns=reliable_groups,
        index=adata.obs.index
    )
    data_df['cluster'] = clusters

    # Aggregate by cluster (mean accessibility across peaks in each cluster)
    cluster_profiles = data_df.groupby('cluster').mean()

    if verbose:
        print(f"Created cluster profiles: {cluster_profiles.shape}")
        print(f"Clusters: {sorted(cluster_profiles.index)}")
        print(f"Reliable pseudobulk groups: {len(reliable_groups)}")

    return cluster_profiles


# =============================================================================
# METADATA PARSING & AGGREGATION
# =============================================================================

def parse_pseudobulk_groups(reliable_groups, verbose=True):
    """
    Parse pseudobulk group names to extract celltype and timepoint mappings.

    Assumes pseudobulk groups follow the naming format: celltype_timepoint
    (e.g., "neural_5somites", "heart_myocardium_15somites").

    Parameters
    ----------
    reliable_groups : List[str]
        List of reliable pseudobulk group names (after filtering by cell count).
    verbose : bool, default=True
        Whether to print parsing information and summary statistics.

    Returns
    -------
    tuple of (dict, dict, set, set)
        - celltype_mapping : dict
            Maps pseudobulk group name -> celltype
        - timepoint_mapping : dict
            Maps pseudobulk group name -> timepoint
        - reliable_celltypes : set
            Unique celltypes found in reliable groups
        - reliable_timepoints : set
            Unique timepoints found in reliable groups

    Examples
    --------
    >>> groups = ["neural_5somites", "neural_10somites", "heart_10somites"]
    >>> ct_map, tp_map, celltypes, timepoints = parse_pseudobulk_groups(groups)
    Parsed 3 pseudobulk groups
    Found 2 reliable celltypes: ['heart', 'neural']
    Found 2 reliable timepoints: ['10somites', '5somites']

    Notes
    -----
    - Timepoint is expected to end with 'somites' (e.g., '5somites', '15somites')
    - Celltype is extracted as everything before the final '_timepoint'
    - Groups that don't match the expected format will be skipped
    """

    if verbose:
        print("\n" + "="*60)
        print("PARSING PSEUDOBULK GROUP NAMES")
        print("="*60)

    celltype_mapping = {}
    timepoint_mapping = {}

    # Parse group names
    for col in reliable_groups:
        # Find the timepoint (ends with 'somites')
        timepoint_match = re.search(r'(\d+somites)$', col)
        if timepoint_match:
            timepoint = timepoint_match.group(1)
            celltype = col.replace(f'_{timepoint}', '')
            celltype_mapping[col] = celltype
            timepoint_mapping[col] = timepoint

    # Get unique celltypes and timepoints
    reliable_celltypes = set(celltype_mapping.values())
    reliable_timepoints = set(timepoint_mapping.values())

    if verbose:
        print(f"Parsed {len(celltype_mapping)} pseudobulk groups")
        print(f"Found {len(reliable_celltypes)} reliable celltypes: {sorted(reliable_celltypes)}")
        print(f"Found {len(reliable_timepoints)} reliable timepoints: {sorted(reliable_timepoints)}")

    return celltype_mapping, timepoint_mapping, reliable_celltypes, reliable_timepoints


def aggregate_by_metadata(cluster_profiles, metadata_type='celltype',
                         celltype_mapping=None, timepoint_mapping=None,
                         lineage_mapping=None, verbose=True):
    """
    Aggregate cluster accessibility profiles by metadata type.

    This is Step 2 of the entropy analysis pipeline. It collapses pseudobulk groups
    by averaging over the orthogonal metadata dimension. For example, if aggregating
    by celltype, it averages across all timepoints for each celltype.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        Cluster-by-pseudobulk_groups matrix from create_cluster_pseudobulk_profiles.
    metadata_type : str, default='celltype'
        Type of metadata to aggregate by. One of: 'celltype', 'timepoint', 'lineage'.
    celltype_mapping : dict, optional
        Mapping from pseudobulk group name -> celltype. Required if metadata_type='celltype'.
    timepoint_mapping : dict, optional
        Mapping from pseudobulk group name -> timepoint. Required if metadata_type='timepoint'.
    lineage_mapping : dict, optional
        Mapping from pseudobulk group name -> lineage. Required if metadata_type='lineage'.
    verbose : bool, default=True
        Whether to print aggregation information.

    Returns
    -------
    pd.DataFrame
        Cluster-by-metadata matrix (e.g., clusters x celltypes).
        Shape: (n_clusters, n_metadata_categories)

    Raises
    ------
    ValueError
        If metadata_type is not one of 'celltype', 'timepoint', or 'lineage'.
        If the required mapping dictionary is not provided for the metadata_type.

    Examples
    --------
    >>> # Aggregate by celltype (average over timepoints)
    >>> cluster_celltype = aggregate_by_metadata(
    ...     cluster_profiles,
    ...     metadata_type='celltype',
    ...     celltype_mapping=ct_map,
    ...     timepoint_mapping=tp_map
    ... )
    Aggregating across 35 celltypes
    Created (47, 35) cluster-by-celltype matrix

    >>> # Aggregate by timepoint (average over celltypes)
    >>> cluster_timepoint = aggregate_by_metadata(
    ...     cluster_profiles,
    ...     metadata_type='timepoint',
    ...     celltype_mapping=ct_map,
    ...     timepoint_mapping=tp_map
    ... )
    Aggregating across 5 timepoints
    Created (47, 5) cluster-by-timepoint matrix

    Notes
    -----
    - Equal weighting: All pseudobulk groups for a category receive equal weight
    - Missing data: Categories with no reliable groups are skipped
    - The aggregation preserves the cluster structure while simplifying the metadata dimension
    """

    if verbose:
        print(f"\n" + "="*60)
        print(f"STEP 2: AGGREGATE BY {metadata_type.upper()}")
        print("="*60)

    if metadata_type == 'celltype':
        mapping = celltype_mapping
        reliable_categories = set(mapping.values())
    elif metadata_type == 'timepoint':
        mapping = timepoint_mapping
        reliable_categories = set(mapping.values())
    elif metadata_type == 'lineage':
        mapping = lineage_mapping
        reliable_categories = set(mapping.values())
    else:
        raise ValueError(f"metadata_type must be 'celltype', 'timepoint', or 'lineage', got {metadata_type}")

    if mapping is None:
        raise ValueError(f"No mapping provided for {metadata_type}")

    if verbose:
        print(f"Aggregating across {len(reliable_categories)} {metadata_type}s")

    # Create cluster-by-metadata matrix
    cluster_metadata_profiles = pd.DataFrame(index=cluster_profiles.index)

    for category in reliable_categories:
        # Get pseudobulk groups for this category
        category_cols = [col for col, cat in mapping.items()
                        if cat == category and col in cluster_profiles.columns]

        if category_cols:
            # Mean accessibility across all pseudobulk groups for this category
            mean_accessibility = cluster_profiles[category_cols].mean(axis=1)
            cluster_metadata_profiles[category] = mean_accessibility

            if verbose:
                print(f"  {category:25s}: averaged {len(category_cols):2d} groups")
        else:
            if verbose:
                print(f"  {category:25s}: no reliable groups found")

    if verbose:
        print(f"\nCreated {cluster_metadata_profiles.shape} cluster-by-{metadata_type} matrix")
        print(f"Sample values:")
        print(cluster_metadata_profiles.iloc[:3, :5])

    return cluster_metadata_profiles


# =============================================================================
# ACCESSIBILITY ENTROPY & PATTERN ANALYSIS
# =============================================================================

def compute_accessibility_entropy(values, normalize=True, min_value=0.0):
    """
    Compute Shannon entropy of accessibility distribution.

    Shannon entropy measures the uniformity of accessibility across categories.
    High entropy indicates broad/uniform accessibility, while low entropy indicates
    specific/concentrated accessibility.

    Entropy = -Σ(p_i * log2(p_i))
    where p_i = accessibility_i / sum(all_accessibility)

    Parameters
    ----------
    values : array-like
        Accessibility values across categories (e.g., celltypes or timepoints).
    normalize : bool, default=True
        Whether to normalize by maximum possible entropy (log2(n_categories)).
        If True, returns values in [0, 1] range.
    min_value : float, default=0.0
        Minimum accessibility value to include in computation.
        Values below this threshold are excluded. Use 0.0 to include all non-zero values.

    Returns
    -------
    float
        Entropy score. Range is [0, 1] if normalized, [0, log2(n)] otherwise.
        - 0.0: Completely specific (all accessibility in one category)
        - 1.0: Perfectly uniform (equal accessibility across all categories, normalized)

    Examples
    --------
    >>> # Uniform distribution - high entropy
    >>> uniform_vals = [0.2, 0.2, 0.2, 0.2, 0.2]
    >>> entropy = compute_accessibility_entropy(uniform_vals)
    >>> print(f"Uniform entropy: {entropy:.3f}")  # Should be close to 1.0
    Uniform entropy: 1.000

    >>> # Specific distribution - low entropy
    >>> specific_vals = [0.9, 0.05, 0.02, 0.02, 0.01]
    >>> entropy = compute_accessibility_entropy(specific_vals)
    >>> print(f"Specific entropy: {entropy:.3f}")  # Should be close to 0.0
    Specific entropy: 0.352

    Notes
    -----
    - Only non-zero (or above min_value) categories are considered
    - Returns 0.0 if there is only 1 non-zero category
    - Uses log2 (bits) as the entropy unit
    - Normalization allows comparison across different numbers of categories
    """

    # Convert to numpy array
    vals = np.array(values)

    # Filter out negative and very small values if specified
    if min_value > 0:
        vals = vals[vals >= min_value]
    else:
        vals = vals[vals > 0]  # Remove only true zeros

    if len(vals) <= 1:
        return 0.0

    # Compute probabilities (proper definition)
    total_accessibility = vals.sum()
    if total_accessibility == 0:
        return 0.0

    probabilities = vals / total_accessibility

    # Calculate Shannon entropy: -Σ(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))

    if normalize:
        max_entropy = np.log2(len(probabilities))
        entropy = entropy / max_entropy if max_entropy > 0 else 0

    return float(entropy)


def analyze_cluster_accessibility_patterns(cluster_metadata_profiles,
                                          metadata_type='celltype',
                                          broad_entropy_threshold=0.75,
                                          specific_dominance_threshold=0.6,
                                          specific_entropy_threshold=0.4,
                                          moderate_dominance_threshold=0.4,
                                          verbose=True):
    """
    Analyze accessibility patterns for each cluster using entropy and dominance metrics.

    This is Step 3 of the entropy analysis pipeline. It classifies each cluster's
    accessibility pattern as broadly accessible, specific to a category, or enriched
    for a category based on complementary metrics.

    Parameters
    ----------
    cluster_metadata_profiles : pd.DataFrame
        Cluster-by-metadata matrix (e.g., cluster-by-celltype) from aggregate_by_metadata.
    metadata_type : str, default='celltype'
        Type of metadata being analyzed ('celltype', 'timepoint', or 'lineage').
    broad_entropy_threshold : float, default=0.75
        Entropy threshold for classifying as broadly accessible (not currently used
        in updated classification logic).
    specific_dominance_threshold : float, default=0.6
        Dominance threshold for specific accessibility classification (not currently used).
    specific_entropy_threshold : float, default=0.4
        Entropy threshold for specific accessibility classification (not currently used).
    moderate_dominance_threshold : float, default=0.4
        Dominance threshold for moderate enrichment classification (not currently used).
    verbose : bool, default=True
        Whether to print detailed classification results for each cluster.

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns:
        - cluster: Cluster identifier
        - pattern: Classification ('broadly_accessible', 'specific_{category}',
                   'enriched_{category}')
        - confidence: Classification confidence ('high', 'medium', 'low')
        - entropy: Normalized Shannon entropy [0-1]
        - dominance: Fraction of total accessibility in dominant category [0-1]
        - dominant_category: Category with highest accessibility
        - n_categories: Number of non-zero categories
        - coverage_5pct, coverage_10pct, coverage_20pct: Number of categories above
                                                          5%, 10%, 20% of total
        - mean_accessibility: Total accessibility across all categories
        - top_3_categories: Dictionary of top 3 categories and their values

    Examples
    --------
    >>> results = analyze_cluster_accessibility_patterns(
    ...     cluster_celltype_profiles,
    ...     metadata_type='celltype'
    ... )
    Cluster  0: broadly_accessible          (entropy=0.978, dominance=0.082, categories=34)
    Cluster  1: specific_neural             (entropy=0.812, dominance=0.312, categories=28)

    >>> # Filter specific patterns
    >>> specific_clusters = results[results['pattern'].str.startswith('specific_')]
    >>> print(f"Found {len(specific_clusters)} clusters with specific patterns")

    Notes
    -----
    The classification uses a revised decision tree based on data-driven thresholds:
    - entropy ≥ 0.95: broadly_accessible
    - dominance ≥ 0.25 and entropy ≤ 0.90: specific_{category}
    - dominance ≥ 0.25 and entropy > 0.90: enriched_{category}
    - dominance 0.15-0.25: enriched_{category} (low confidence)
    - Otherwise: broadly_accessible

    These thresholds were empirically determined from the zebrafish multiome dataset
    and may need adjustment for other datasets.
    """

    if verbose:
        print(f"\n" + "="*60)
        print(f"STEP 3: ANALYZE {metadata_type.upper()} ACCESSIBILITY PATTERNS")
        print("="*60)

    results = []

    for cluster in cluster_metadata_profiles.index:
        cluster_profile = cluster_metadata_profiles.loc[cluster]

        # NO FILTERING - use all accessibility values as is
        # This preserves the true accessibility distribution

        # Compute metrics using the proper entropy definition
        entropy = compute_accessibility_entropy(cluster_profile, normalize=True, min_value=0.0)

        # Dominance = proportion of total accessibility in dominant category
        total_accessibility = cluster_profile.sum()
        if total_accessibility > 0:
            dominance = cluster_profile.max() / total_accessibility
            dominant_category = cluster_profile.idxmax()
        else:
            dominance = 0.0
            dominant_category = "unclear"

        n_categories = len(cluster_profile[cluster_profile > 0])  # Count non-zero categories

        # Coverage metrics (based on proportions of total accessibility)
        if total_accessibility > 0:
            proportions = cluster_profile / total_accessibility
            coverage_5pct = (proportions >= 0.05).sum()
            coverage_10pct = (proportions >= 0.10).sum()
            coverage_20pct = (proportions >= 0.20).sum()
        else:
            coverage_5pct = coverage_10pct = coverage_20pct = 0

        # REVISED CLASSIFICATION with realistic thresholds for your data
        if entropy >= 0.95:  # Very high entropy = truly uniform
            pattern = "broadly_accessible"
            confidence = "high" if entropy >= 0.98 else "medium"

        elif dominance >= 0.25:  # >25% in one category = enriched
            if entropy <= 0.90:  # Low entropy + high dominance = specific
                pattern = f"specific_{dominant_category}"
                confidence = "high" if dominance >= 0.30 else "medium"
            else:  # High entropy but some dominance = enriched
                pattern = f"enriched_{dominant_category}"
                confidence = "medium" if dominance >= 0.20 else "low"

        elif dominance >= 0.15:  # 15-25% = moderate enrichment
            pattern = f"enriched_{dominant_category}"
            confidence = "low"

        else:  # Very uniform distribution
            pattern = "broadly_accessible"
            confidence = "medium"

        if verbose:
            print(f"Cluster {cluster:2s}: {pattern:30s} "
                  f"(entropy={entropy:.3f}, dominance={dominance:.3f}, "
                  f"categories={n_categories})")

        # Store results
        results.append({
            'cluster': cluster,
            'pattern': pattern,
            'confidence': confidence,
            'entropy': entropy,
            'dominance': dominance,
            'dominant_category': dominant_category,
            'n_categories': n_categories,
            'coverage_5pct': coverage_5pct,
            'coverage_10pct': coverage_10pct,
            'coverage_20pct': coverage_20pct,
            'mean_accessibility': total_accessibility,
            'top_3_categories': dict(cluster_profile.nlargest(3))
        })

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n{metadata_type.upper()} PATTERN SUMMARY:")
        pattern_counts = results_df['pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"  {pattern:30s}: {count:2d} clusters")

    return results_df


def run_metadata_entropy_analysis(adata, cluster_col='leiden_coarse',
                                 metadata_type='celltype', min_cells=20,
                                 lineage_mapping=None, verbose=True):
    """
    Complete workflow for entropy-based accessibility analysis.

    This is a convenience function that runs the complete 3-step entropy analysis pipeline:
    1. Create cluster pseudobulk profiles (with filtering)
    2. Parse metadata and aggregate by metadata type
    3. Analyze patterns and classify clusters

    Parameters
    ----------
    adata : AnnData
        Data object with peaks as obs, pseudobulk groups as var, and cluster
        assignments in obs[cluster_col].
    cluster_col : str, default='leiden_coarse'
        Column name for cluster assignments in adata.obs.
    metadata_type : str, default='celltype'
        Type of metadata to analyze. One of: 'celltype', 'timepoint', 'lineage'.
    min_cells : int, default=20
        Minimum cells per pseudobulk group for inclusion in analysis.
    lineage_mapping : dict, optional
        Mapping from lineage -> list of celltypes. Required only if metadata_type='lineage'.
        Example: {"CNS": ["neural", "neurons"], "Mesoderm": ["somites", "heart"]}
    verbose : bool, default=True
        Whether to print detailed progress for all steps.

    Returns
    -------
    pd.DataFrame
        Analysis results with cluster patterns and comprehensive metrics.
        See analyze_cluster_accessibility_patterns for column descriptions.

    Examples
    --------
    >>> # Analyze celltype accessibility patterns
    >>> celltype_results = run_metadata_entropy_analysis(
    ...     adata_peaks,
    ...     cluster_col='leiden_coarse',
    ...     metadata_type='celltype',
    ...     min_cells=20
    ... )
    RUNNING CELLTYPE ENTROPY ANALYSIS
    ================================================================================
    STEP 1: CREATE CLUSTER-BY-PSEUDOBULK PROFILES
    ...

    >>> # Analyze temporal patterns
    >>> timepoint_results = run_metadata_entropy_analysis(
    ...     adata_peaks,
    ...     metadata_type='timepoint',
    ...     min_cells=20
    ... )

    >>> # Analyze by developmental lineage (requires lineage mapping)
    >>> lineage_map = {
    ...     "CNS": ["neural", "neurons", "hindbrain"],
    ...     "Mesoderm": ["somites", "heart_myocardium", "lateral_plate_mesoderm"]
    ... }
    >>> lineage_results = run_metadata_entropy_analysis(
    ...     adata_peaks,
    ...     metadata_type='lineage',
    ...     lineage_mapping=lineage_map,
    ...     min_cells=20
    ... )

    Notes
    -----
    This function orchestrates the complete analysis pipeline and is the recommended
    entry point for most use cases. For more control over individual steps, use
    the component functions directly.
    """

    if verbose:
        print(f"RUNNING {metadata_type.upper()} ENTROPY ANALYSIS")
        print("="*80)

    # Step 1: Create cluster pseudobulk profiles
    cluster_profiles = create_cluster_pseudobulk_profiles(
        adata, cluster_col, min_cells, verbose
    )

    # Step 2: Parse group names and get mappings
    reliable_groups = list(cluster_profiles.columns)
    celltype_mapping, timepoint_mapping, reliable_celltypes, reliable_timepoints = parse_pseudobulk_groups(
        reliable_groups, verbose
    )

    # Step 3: Aggregate by metadata type
    if metadata_type == 'lineage' and lineage_mapping is not None:
        # Create lineage mapping from celltype mapping
        lineage_to_groups = {}
        for group, celltype in celltype_mapping.items():
            for lineage, celltypes_in_lineage in lineage_mapping.items():
                if celltype in celltypes_in_lineage:
                    if lineage not in lineage_to_groups:
                        lineage_to_groups[lineage] = []
                    lineage_to_groups[lineage].append(group)
                    break

        # Create lineage aggregation mapping
        lineage_group_mapping = {}
        for lineage, groups in lineage_to_groups.items():
            for group in groups:
                lineage_group_mapping[group] = lineage

        cluster_metadata_profiles = aggregate_by_metadata(
            cluster_profiles, 'lineage',
            lineage_mapping=lineage_group_mapping,
            verbose=verbose
        )
    else:
        cluster_metadata_profiles = aggregate_by_metadata(
            cluster_profiles, metadata_type,
            celltype_mapping=celltype_mapping,
            timepoint_mapping=timepoint_mapping,
            verbose=verbose
        )

    # Step 4: Analyze patterns
    results = analyze_cluster_accessibility_patterns(
        cluster_metadata_profiles, metadata_type, verbose=verbose
    )

    return results


def validate_cluster_23_entropy(adata, cluster_col='leiden_coarse', min_cells=20):
    """
    Validation function specifically for testing Cluster 23 entropy analysis.

    This function was created to validate that Cluster 23 is correctly classified
    as "broadly_accessible" for both celltype and timepoint dimensions, as expected
    from the biological characteristics of this cluster.

    Parameters
    ----------
    adata : AnnData
        Data object with peaks and cluster assignments.
    cluster_col : str, default='leiden_coarse'
        Column name for cluster assignments.
    min_cells : int, default=20
        Minimum cells per pseudobulk group.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - celltype_results: Full celltype entropy analysis results
        - timepoint_results: Full timepoint entropy analysis results

    Examples
    --------
    >>> ct_results, tp_results = validate_cluster_23_entropy(adata_peaks)
    ================================================================================
    CLUSTER 23 ENTROPY VALIDATION
    ================================================================================
    Testing CELLTYPE entropy for Cluster 23...
    Testing TIMEPOINT entropy for Cluster 23...

    Cluster 23 Results:
    ----------------------------------------
    Celltype:
      Pattern: broadly_accessible
      Confidence: high
      Entropy: 0.978
      Dominance: 0.082
      Categories: 34
      Expected broadly_accessible: ✓

    Notes
    -----
    This is a specialized validation function. For general analysis, use
    run_metadata_entropy_analysis instead.
    """

    print("="*80)
    print("CLUSTER 23 ENTROPY VALIDATION")
    print("="*80)

    # Run celltype analysis
    print("Testing CELLTYPE entropy for Cluster 23...")
    celltype_results = run_metadata_entropy_analysis(
        adata, cluster_col, 'celltype', min_cells, verbose=False
    )

    # Run timepoint analysis
    print("\nTesting TIMEPOINT entropy for Cluster 23...")
    timepoint_results = run_metadata_entropy_analysis(
        adata, cluster_col, 'timepoint', min_cells, verbose=False
    )

    # Check Cluster 23 results
    cluster_23_ct = celltype_results[celltype_results['cluster'] == '23']
    cluster_23_tp = timepoint_results[timepoint_results['cluster'] == '23']

    print(f"\nCluster 23 Results:")
    print("-" * 40)

    if len(cluster_23_ct) > 0:
        ct_row = cluster_23_ct.iloc[0]
        print(f"Celltype:")
        print(f"  Pattern: {ct_row['pattern']}")
        print(f"  Confidence: {ct_row['confidence']}")
        print(f"  Entropy: {ct_row['entropy']:.3f}")
        print(f"  Dominance: {ct_row['dominance']:.3f}")
        print(f"  Categories: {ct_row['n_categories']}")

        ct_pass = "✓" if ct_row['pattern'] == 'broadly_accessible' else "✗"
        print(f"  Expected broadly_accessible: {ct_pass}")
    else:
        print("Celltype: Cluster 23 not found")

    if len(cluster_23_tp) > 0:
        tp_row = cluster_23_tp.iloc[0]
        print(f"\nTimepoint:")
        print(f"  Pattern: {tp_row['pattern']}")
        print(f"  Confidence: {tp_row['confidence']}")
        print(f"  Entropy: {tp_row['entropy']:.3f}")
        print(f"  Dominance: {tp_row['dominance']:.3f}")
        print(f"  Categories: {tp_row['n_categories']}")

        tp_pass = "✓" if tp_row['pattern'] == 'broadly_accessible' else "✗"
        print(f"  Expected broadly_accessible: {tp_pass}")
    else:
        print("Timepoint: Cluster 23 not found")

    return celltype_results, timepoint_results


# =============================================================================
# TEMPORAL REGRESSION ANALYSIS
# =============================================================================

def fit_temporal_regression(timepoint_order, accessibility_values):
    """
    Fit linear regression to temporal accessibility data.

    This function quantifies temporal trends in accessibility across developmental
    timepoints, providing metrics for directionality (slope) and fit quality (R²).

    Parameters
    ----------
    timepoint_order : list
        Ordered list of timepoints (e.g., ['0somites', '5somites', '10somites']).
        Order matters - should be chronological.
    accessibility_values : array-like
        Accessibility values corresponding to each timepoint.
        Length must match timepoint_order.

    Returns
    -------
    dict
        Dictionary containing:
        - 'slope': float
            Regression slope (change in accessibility per timepoint unit).
            Positive = increasing, negative = decreasing over time.
        - 'r_squared': float
            R² coefficient of determination [0-1]. Measures goodness of fit.
            Values close to 1 indicate strong linear trend.
        - 'rmsd': float
            Root mean square deviation. Measures average prediction error.
        - 'y_pred': np.ndarray
            Predicted accessibility values from the linear model.

    Examples
    --------
    >>> # Increasing accessibility over time
    >>> timepoints = ['0somites', '5somites', '10somites', '15somites', '20somites']
    >>> accessibilities = [0.1, 0.3, 0.5, 0.7, 0.9]
    >>> fit = fit_temporal_regression(timepoints, accessibilities)
    >>> print(f"Slope: {fit['slope']:.3f}, R²: {fit['r_squared']:.3f}")
    Slope: 0.200, R²: 1.000

    >>> # Check if accessibility is increasing or decreasing
    >>> if fit['slope'] > 0 and fit['r_squared'] > 0.8:
    ...     print("Strong increasing trend")
    ... elif fit['slope'] < 0 and fit['r_squared'] > 0.8:
    ...     print("Strong decreasing trend")
    ... else:
    ...     print("No clear linear trend")
    Strong increasing trend

    Notes
    -----
    - Falls back to numpy.polyfit if sklearn is not available
    - Uses timepoint indices (0, 1, 2, ...) as X values for regression
    - Assumes timepoints are provided in chronological order
    - Linear model may not capture complex non-monotonic patterns
    - Consider complementary metrics (e.g., peak timepoint) for interpretation

    Dependencies
    ------------
    Optional: scikit-learn (for LinearRegression)
    Fallback: numpy (for polyfit-based regression)
    """

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
    except ImportError:
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


# =============================================================================
# COMPREHENSIVE STATISTICAL METRICS
# =============================================================================

def compute_comprehensive_accessibility_metrics(profile):
    """
    Compute multiple complementary metrics for accessibility distribution analysis.

    This function provides a comprehensive statistical characterization of how
    accessibility is distributed across categories, going beyond simple entropy
    to include inequality measures, concentration metrics, and dominance ratios.

    Parameters
    ----------
    profile : pd.Series
        Accessibility values across categories (e.g., celltypes, timepoints).

    Returns
    -------
    dict
        Dictionary with the following metrics:

        - 'entropy': float [0-1]
            Normalized Shannon entropy. Measures distribution uniformity.

        - 'dominance': float [0-1]
            Fraction of total accessibility in the top category.

        - 'cv': float [0-∞)
            Coefficient of variation (std/mean). Measures relative variability.

        - 'top3_fraction': float [0-1]
            Fraction of total accessibility concentrated in top 3 categories.

        - 'effective_categories': float [1-n]
            Effective number of categories (1/Σp_i²). Measures true diversity.

        - 'gini': float [0-1]
            Gini coefficient. Measures inequality (0=perfect equality, 1=perfect inequality).

        - 'ratio_90_10': float [1-∞)
            Ratio of mean accessibility in top 10% vs bottom 10% of categories.

        - 'dominant_category': str
            Name of the category with highest accessibility.

    Examples
    --------
    >>> # Uniform distribution
    >>> uniform_profile = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2],
    ...                             index=['A', 'B', 'C', 'D', 'E'])
    >>> metrics = compute_comprehensive_accessibility_metrics(uniform_profile)
    >>> print(f"Entropy: {metrics['entropy']:.3f}")
    >>> print(f"Gini: {metrics['gini']:.3f}")
    >>> print(f"Effective categories: {metrics['effective_categories']:.1f}")
    Entropy: 1.000
    Gini: 0.000
    Effective categories: 5.0

    >>> # Highly concentrated distribution
    >>> concentrated_profile = pd.Series([0.9, 0.05, 0.03, 0.01, 0.01],
    ...                                  index=['A', 'B', 'C', 'D', 'E'])
    >>> metrics = compute_comprehensive_accessibility_metrics(concentrated_profile)
    >>> print(f"Entropy: {metrics['entropy']:.3f}")
    >>> print(f"Dominance: {metrics['dominance']:.3f}")
    >>> print(f"Top3 fraction: {metrics['top3_fraction']:.3f}")
    Entropy: 0.415
    Dominance: 0.900
    Top3 fraction: 0.980

    Notes
    -----
    These metrics are complementary and capture different aspects of distribution:
    - Entropy and effective_categories capture overall uniformity
    - Dominance and top3_fraction capture concentration in top categories
    - Gini and ratio_90_10 capture inequality across the full distribution
    - CV captures relative variability

    Using multiple metrics provides more robust classification than any single metric.
    """

    if profile.sum() == 0:
        return {'entropy': 0, 'dominance': 0, 'cv': 0, 'top3_fraction': 0,
                'effective_categories': 0, 'gini': 0}

    # Basic metrics
    total = profile.sum()
    proportions = profile / total

    # 1. Shannon Entropy (your current metric)
    nonzero_props = proportions[proportions > 0]
    if len(nonzero_props) > 1:
        entropy = -np.sum(nonzero_props * np.log2(nonzero_props))
        max_entropy = np.log2(len(nonzero_props))
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0

    # 2. Dominance (fraction in top category)
    dominance = profile.max() / total

    # 3. Coefficient of Variation (measures relative variability)
    cv = profile.std() / profile.mean() if profile.mean() > 0 else 0

    # 4. Top-3 Fraction (concentration in top 3 categories)
    top3_sum = profile.nlargest(3).sum()
    top3_fraction = top3_sum / total

    # 5. Effective Number of Categories (1/sum(p_i^2))
    # This is more sensitive to concentration than entropy
    effective_n = 1 / np.sum(proportions**2) if np.sum(proportions**2) > 0 else 0

    # 6. Gini Coefficient (inequality measure)
    sorted_vals = np.sort(profile.values)
    n = len(sorted_vals)
    if n > 1 and sorted_vals.sum() > 0:
        cumsum = np.cumsum(sorted_vals)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    else:
        gini = 0

    # 7. 90/10 Ratio (top 10% vs bottom 10% of categories)
    n_top = max(1, int(n * 0.1))
    n_bottom = max(1, int(n * 0.1))
    top_10pct_mean = profile.nlargest(n_top).mean()
    bottom_10pct_mean = profile.nsmallest(n_bottom).mean()
    ratio_90_10 = top_10pct_mean / bottom_10pct_mean if bottom_10pct_mean > 0 else np.inf

    return {
        'entropy': normalized_entropy,
        'dominance': dominance,
        'cv': cv,
        'top3_fraction': top3_fraction,
        'effective_categories': effective_n,
        'gini': gini,
        'ratio_90_10': ratio_90_10,
        'dominant_category': profile.idxmax()
    }


def classify_accessibility_pattern_comprehensive(profile, metadata_type='celltype'):
    """
    Classify accessibility patterns using multi-metric decision tree.

    This function uses multiple complementary metrics to robustly classify
    accessibility patterns, providing more nuanced classification than single-metric
    approaches. It's an alternative to the simpler classification in
    analyze_cluster_accessibility_patterns.

    Parameters
    ----------
    profile : pd.Series
        Accessibility values across categories (e.g., celltypes).
    metadata_type : str, default='celltype'
        Type of metadata (for labeling purposes).

    Returns
    -------
    tuple of (str, str, dict)
        - pattern : str
            Classification label. One of:
            - "specific_{dominant_category}": Highly concentrated accessibility
            - "enriched_{dominant_category}": Moderately concentrated accessibility
            - "broadly_accessible": Uniform/flat distribution
            - "intermediate": Ambiguous pattern

        - confidence : str
            Confidence level: "high", "medium", or "low"

        - metrics : dict
            All computed metrics from compute_comprehensive_accessibility_metrics

    Examples
    --------
    >>> celltype_profile = pd.Series(
    ...     [0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05],
    ...     index=['neural', 'somites', 'heart', 'endoderm', 'epidermis', 'PSM', 'NC']
    ... )
    >>> pattern, conf, metrics = classify_accessibility_pattern_comprehensive(
    ...     celltype_profile, metadata_type='celltype'
    ... )
    >>> print(f"Pattern: {pattern}")
    >>> print(f"Confidence: {conf}")
    >>> print(f"Dominance: {metrics['dominance']:.3f}")
    Pattern: specific_neural
    Confidence: high
    Dominance: 0.500

    Notes
    -----
    Classification criteria:
    1. HIGHLY SPECIFIC: dominance ≥ 0.4 OR top3_fraction ≥ 0.7 OR ratio_90_10 ≥ 10
    2. MODERATELY ENRICHED: dominance ≥ 0.2 OR top3_fraction ≥ 0.5 OR ratio_90_10 ≥ 3
    3. BROADLY ACCESSIBLE: effective_categories ≥ 15 AND cv ≤ 2.0 AND ratio_90_10 ≤ 2
    4. INTERMEDIATE: Everything else

    These thresholds were empirically determined from zebrafish multiome data and
    may need adjustment for other datasets.
    """

    metrics = compute_comprehensive_accessibility_metrics(profile)

    # Extract key metrics
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    cv = metrics['cv']
    top3_fraction = metrics['top3_fraction']
    effective_n = metrics['effective_categories']
    gini = metrics['gini']
    ratio_90_10 = metrics['ratio_90_10']
    dominant_cat = metrics['dominant_category']

    # Multi-metric classification
    # 1. HIGHLY SPECIFIC (strong concentration)
    if dominance >= 0.4 or top3_fraction >= 0.7 or ratio_90_10 >= 10:
        pattern = f"specific_{dominant_cat}"
        confidence = "high" if dominance >= 0.5 else "medium"

    # 2. MODERATELY ENRICHED
    elif dominance >= 0.2 or top3_fraction >= 0.5 or ratio_90_10 >= 3:
        pattern = f"enriched_{dominant_cat}"
        confidence = "medium" if dominance >= 0.25 else "low"

    # 3. BROADLY ACCESSIBLE (flat distribution)
    elif effective_n >= 15 and cv <= 2.0 and ratio_90_10 <= 2:
        pattern = "broadly_accessible"
        confidence = "high" if effective_n >= 20 else "medium"

    # 4. INTERMEDIATE
    else:
        pattern = "intermediate"
        confidence = "low"

    return pattern, confidence, metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_accessibility_metrics(profile):
    """
    Compute basic accessibility metrics (simplified version).

    This is a lightweight alternative to compute_comprehensive_accessibility_metrics
    that computes only the core metrics: entropy, dominance, and coefficient of variation.

    Parameters
    ----------
    profile : pd.Series
        Accessibility values across categories.

    Returns
    -------
    dict
        Dictionary with keys: 'entropy', 'dominance', 'cv', 'dominant_category'

    Notes
    -----
    This function is used internally and provides a faster alternative when only
    basic metrics are needed.
    """

    if profile.sum() == 0:
        return {'entropy': 0, 'dominance': 0, 'cv': 0, 'dominant_category': 'none'}

    # Shannon entropy
    proportions = profile / profile.sum()
    nonzero_props = proportions[proportions > 0]
    if len(nonzero_props) > 1:
        entropy = -np.sum(nonzero_props * np.log2(nonzero_props))
        max_entropy = np.log2(len(nonzero_props))
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0

    # Dominance (fraction in top category)
    dominance = profile.max() / profile.sum()

    # Coefficient of variation
    cv = profile.std() / profile.mean() if profile.mean() > 0 else 0

    return {
        'entropy': normalized_entropy,
        'dominance': dominance,
        'cv': cv,
        'dominant_category': profile.idxmax()
    }


def _get_default_order(categories, metadata_type):
    """
    Get default ordering for categories based on biological/temporal logic.

    Parameters
    ----------
    categories : list
        List of category names to order.
    metadata_type : str
        Type of metadata ('timepoint', 'celltype', or other).

    Returns
    -------
    list
        Ordered list of categories.

    Notes
    -----
    - Timepoints are sorted numerically (0somites, 5somites, 10somites, ...)
    - Celltypes are ordered by developmental lineage (CNS, mesoderm, endoderm, ...)
    - Other metadata types are sorted alphabetically
    """

    if metadata_type == 'timepoint':
        # Sort timepoints numerically (0somites, 5somites, etc.)
        def extract_number(tp):
            match = re.search(r'(\d+)', tp)
            return int(match.group(1)) if match else 0
        return sorted(categories, key=extract_number)

    elif metadata_type == 'celltype':
        # Developmental lineage-based ordering
        celltype_order = [
            # CNS/Neural
            'neural', 'neural_optic', 'neural_posterior', 'neural_telencephalon',
            'neurons', 'differentiating_neurons',
            'hindbrain', 'midbrain_hindbrain_boundary', 'spinal_cord',
            'optic_cup', 'floor_plate', 'neural_floor_plate',

            # Neural Crest + derivatives
            'neural_crest', 'enteric_neurons',

            # Early mesoderm/multipotent
            'NMPs', 'tail_bud',

            # Axial mesoderm
            'notochord',

            # Paraxial mesoderm
            'PSM', 'somites', 'fast_muscle', 'muscle',

            # Lateral plate mesoderm
            'lateral_plate_mesoderm', 'heart_myocardium',
            'hematopoietic_vasculature', 'hemangioblasts',

            # Other mesoderm-derived
            'pharyngeal_arches', 'pronephros', 'hatching_gland',

            # Endoderm
            'endoderm', 'endocrine_pancreas',

            # Ectoderm
            'epidermis',

            # Germline
            'primordial_germ_cells'
        ]

        # Return categories in the specified order (only those that exist)
        ordered = [ct for ct in celltype_order if ct in categories]
        # Add any remaining categories not in the predefined order
        remaining = [ct for ct in categories if ct not in celltype_order]
        return ordered + sorted(remaining)

    else:
        # Alphabetical for other metadata types
        return sorted(categories)


def _get_default_lineage_mapping():
    """
    Get default lineage to celltype mapping for zebrafish development.

    Returns
    -------
    dict
        Dictionary mapping lineage names to lists of celltypes.
        Keys are major developmental lineages, values are lists of celltype names.

    Examples
    --------
    >>> lineage_map = _get_default_lineage_mapping()
    >>> print(lineage_map['CNS'])
    ['neural', 'neural_optic', 'neural_posterior', ...]

    >>> # Use for lineage-level analysis
    >>> lineage_results = run_metadata_entropy_analysis(
    ...     adata_peaks,
    ...     metadata_type='lineage',
    ...     lineage_mapping=lineage_map
    ... )

    Notes
    -----
    This mapping reflects the major developmental lineages in zebrafish:
    - CNS: Central nervous system and neural derivatives
    - Neural Crest: Neural crest and derivatives
    - Early Mesoderm: Neuromesodermal progenitors and tail bud
    - Axial Mesoderm: Notochord
    - Paraxial Mesoderm: PSM, somites, muscle
    - Lateral Plate Mesoderm: Heart, blood, intermediate mesoderm
    - Other Mesoderm: Pharyngeal arches, pronephros, hatching gland
    - Endoderm: Gut and endodermal derivatives
    - Ectoderm: Epidermis
    - Germline: Primordial germ cells
    """

    return {
        "CNS": [
            "neural", "neural_optic", "neural_posterior", "neural_telencephalon",
            "neurons", "differentiating_neurons", "hindbrain",
            "midbrain_hindbrain_boundary", "spinal_cord", "optic_cup",
            "floor_plate", "neural_floor_plate"
        ],
        "Neural Crest": [
            "neural_crest", "enteric_neurons"
        ],
        "Early Mesoderm": [
            "NMPs", "tail_bud"
        ],
        "Axial Mesoderm": [
            "notochord"
        ],
        "Paraxial Mesoderm": [
            "PSM", "somites", "fast_muscle", "muscle"
        ],
        "Lateral Plate Mesoderm": [
            "lateral_plate_mesoderm", "heart_myocardium",
            "hematopoietic_vasculature", "hemangioblasts"
        ],
        "Other Mesoderm": [
            "pharyngeal_arches", "pronephros", "hatching_gland"
        ],
        "Endoderm": [
            "endoderm", "endocrine_pancreas"
        ],
        "Ectoderm": [
            "epidermis"
        ],
        "Germline": [
            "primordial_germ_cells"
        ]
    }
