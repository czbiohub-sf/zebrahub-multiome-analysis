"""
GRN Comparison and QC Utilities

Functions for comparing gene regulatory networks across conditions,
computing similarity metrics, and performing quality control analyses.
"""

import pandas as pd
import numpy as np
from difflib import get_close_matches


def extract_tf_gene_pairs(grn_matrix):
    """
    Extract TF-gene regulatory pairs from a GRN matrix.

    Converts a GRN adjacency matrix into a set of TF-gene pair strings
    for easier comparison and overlap analysis.

    Parameters
    ----------
    grn_matrix : pd.DataFrame or None
        GRN adjacency matrix with targets as rows, TFs as columns
        Values are regulatory weights (can be positive/negative)

    Returns
    -------
    set of str
        Set of regulatory pairs in format 'target_TF'
        Empty set if input is None or empty

    Examples
    --------
    >>> # Extract pairs from a GRN matrix
    >>> grn = pd.DataFrame({
    ...     'tbx16': [0.5, 0, -0.3],
    ...     'meox1': [0, 0.8, 0]
    ... }, index=['gene1', 'gene2', 'gene3'])
    >>>
    >>> pairs = extract_tf_gene_pairs(grn)
    >>> print(pairs)
    {'gene1_tbx16', 'gene1_meox1', 'gene2_meox1', 'gene3_tbx16'}

    Notes
    -----
    - Only includes pairs with non-zero regulatory weights
    - Useful for computing Jaccard similarity between GRNs
    - Can be used to track regulatory edge conservation across conditions
    """
    if grn_matrix is None or grn_matrix.empty:
        return set()

    # Stack the matrix to get (target, TF) pairs with non-zero values
    grn_stacked = grn_matrix.stack()
    pairs = set(f"{target}_{tf}" for (target, tf), val in grn_stacked.items() if val != 0)
    return pairs


def recommend_threshold(fractions, analysis_type):
    """
    Recommend presence fraction threshold for filtering regulatory pairs.

    Analyzes the distribution of regulatory pair presence across conditions
    and recommends a threshold based on data retention.

    Parameters
    ----------
    fractions : list of float
        Presence fractions (0-1) for each regulatory pair
        e.g., [0.5, 0.8, 0.3] means pairs present in 50%, 80%, 30% of conditions
    analysis_type : str
        Type of analysis for labeling ('Timepoint' or 'Celltype')

    Returns
    -------
    recommended_threshold : float
        Recommended threshold value (0.6, 0.7, or 0.8)
    reason : str
        Explanation of recommendation

    Examples
    --------
    >>> # Analyze presence fractions across timepoints
    >>> fractions = [0.5, 0.6, 0.8, 0.9, 0.4, 0.7, 0.85]
    >>> threshold, reason = recommend_threshold(fractions, "Timepoint")
    >>> print(f"Recommended: {threshold} - {reason}")
    Recommended: 0.7 - Moderate (recommended)

    Notes
    -----
    - 0.8 threshold: Conservative, retains ≥20% of pairs
    - 0.7 threshold: Moderate, retains ≥30% of pairs
    - 0.6 threshold: Liberal, used when data is sparse
    - Prints detailed retention statistics at multiple thresholds
    """
    percentiles = [50, 70, 80, 90]
    threshold_retention = []

    print(f"\n{analysis_type} Analysis - Data retention at different thresholds:")
    for pct in [50, 60, 70, 80, 90]:
        threshold = pct / 100
        retained = sum(1 for f in fractions if f >= threshold)
        retention_pct = retained / len(fractions) * 100
        threshold_retention.append(retention_pct)
        print(f"  {threshold:.1f} threshold: {retention_pct:.1f}% of pairs retained ({retained:,} pairs)")

    # Recommend based on retention
    if threshold_retention[3] >= 20:  # 80% threshold retains ≥20% of data
        return 0.8, "Conservative (recommended)"
    elif threshold_retention[2] >= 30:  # 70% threshold retains ≥30% of data
        return 0.7, "Moderate (recommended)"
    else:
        return 0.6, "Liberal (recommended)"


def compute_corr_betwn_GRNs(df_GRN1, df_GRN2, celltype1, celltype2, network_metric):
    """
    Compute Pearson correlation of a network metric between two GRNs.

    Compares network topology metrics (e.g., degree centrality) for genes
    across two GRN datasets, useful for QC and reproducibility assessment.

    Parameters
    ----------
    df_GRN1 : pd.DataFrame
        First GRN dataset with network metrics
        Must have 'cluster' column and network metric column
    df_GRN2 : pd.DataFrame
        Second GRN dataset with network metrics
    celltype1 : str
        Cell type to analyze from df_GRN1
    celltype2 : str
        Cell type to analyze from df_GRN2 (often same as celltype1)
    network_metric : str
        Column name of metric to compare
        e.g., 'degree_centrality_all', 'betweenness_centrality'

    Returns
    -------
    float
        Pearson correlation coefficient between metrics
        Range: -1 to 1 (typically 0.5-0.95 for replicate GRNs)

    Examples
    --------
    >>> # Compare degree centrality between technical replicates
    >>> df_grn1 = pd.read_csv('TDR118_grn_metrics.csv', index_col=0)
    >>> df_grn2 = pd.read_csv('TDR119_grn_metrics.csv', index_col=0)
    >>>
    >>> # Same cell type comparison (technical replicate QC)
    >>> corr_psm = compute_corr_betwn_GRNs(
    ...     df_grn1, df_grn2, 'PSM', 'PSM', 'degree_centrality_all'
    ... )
    >>> print(f"PSM correlation: {corr_psm:.3f}")
    PSM correlation: 0.876
    >>>
    >>> # Different cell type comparison (biological variation)
    >>> corr_cross = compute_corr_betwn_GRNs(
    ...     df_grn1, df_grn2, 'PSM', 'NMPs', 'degree_centrality_all'
    ... )

    Notes
    -----
    - Takes union of gene sets from both GRNs
    - Missing values handled via reindexing (NaN for genes not in both)
    - Higher correlation indicates more similar network topology
    - Useful for assessing: technical reproducibility, biological similarity,
      batch effects
    """
    df1 = df_GRN1[df_GRN1.cluster == celltype1]
    df2 = df_GRN2[df_GRN2.cluster == celltype2]

    # Step 1. Get a union of gene_names
    gene_names = set(df1.index).union(df2.index)

    # Step 2. Create a new dataframe with matching indices
    new_df1 = df1[df1.index.isin(gene_names)]
    new_df2 = df2[df2.index.isin(gene_names)]

    # Step 3. Fill missing values with NaNs
    new_df1 = new_df1.reindex(gene_names)
    new_df2 = new_df2.reindex(gene_names)

    # Step 4. Create the zipped DataFrame
    zipped_df = pd.DataFrame({
        'metric_df1': new_df1[network_metric],
        'metric_df2': new_df2[network_metric]
    })

    # Step 5. Compute Pearson correlation
    corr = zipped_df.metric_df1.corr(zipped_df.metric_df2, method="pearson")
    return corr


def find_celltype_lineage(celltype_name, celltype_to_lineage):
    """
    Find the best matching tissue lineage for a cell type using fuzzy matching.

    Maps cell type annotations to broader tissue lineage categories,
    handling naming variations and typos through fuzzy string matching.

    Parameters
    ----------
    celltype_name : str
        Cell type name to classify (e.g., 'PSM', 'neural_posterior')
    celltype_to_lineage : dict
        Mapping from lineage names to lists of cell types
        e.g., {'Paraxial Mesoderm': ['PSM', 'somites', 'NMPs'],
               'CNS': ['neural', 'neural_posterior', 'spinal_cord']}

    Returns
    -------
    str
        Matched lineage name
        Returns 'Other' if no match found

    Examples
    --------
    >>> celltype_to_lineage = {
    ...     'Paraxial Mesoderm': ['PSM', 'somites', 'NMPs'],
    ...     'CNS': ['neural', 'neural_posterior', 'spinal_cord'],
    ...     'Endoderm': ['endoderm', 'gut']
    ... }
    >>>
    >>> # Exact match
    >>> lineage = find_celltype_lineage('PSM', celltype_to_lineage)
    >>> print(lineage)
    Paraxial Mesoderm
    >>>
    >>> # Fuzzy match (handles typos)
    >>> lineage = find_celltype_lineage('nueral_posterior', celltype_to_lineage)
      Fuzzy match: 'nueral_posterior' → 'neural_posterior' (CNS)
    >>> print(lineage)
    CNS
    >>>
    >>> # Partial match (substring)
    >>> lineage = find_celltype_lineage('neural_plate', celltype_to_lineage)
      Partial match: 'neural_plate' → 'neural' (CNS)
    >>> print(lineage)
    CNS

    Notes
    -----
    - Three-tier matching strategy:
      1. Exact string match (fastest)
      2. Fuzzy string matching (handles typos, minimum 60% similarity)
      3. Partial string matching (substring contains)
    - Prints match type for transparency
    - Useful for harmonizing annotations across datasets
    """
    # First try exact match
    for lineage, celltypes in celltype_to_lineage.items():
        if celltype_name in celltypes:
            return lineage

    # If no exact match, try fuzzy matching
    all_reference_celltypes = []
    lineage_lookup = {}

    for lineage, celltypes in celltype_to_lineage.items():
        for ct in celltypes:
            all_reference_celltypes.append(ct)
            lineage_lookup[ct] = lineage

    # Find closest matches (top 3, minimum 60% similarity)
    closest_matches = get_close_matches(celltype_name, all_reference_celltypes,
                                      n=3, cutoff=0.6)

    if closest_matches:
        best_match = closest_matches[0]
        matched_lineage = lineage_lookup[best_match]
        print(f"  Fuzzy match: '{celltype_name}' → '{best_match}' ({matched_lineage})")
        return matched_lineage

    # If still no match, try partial string matching
    for lineage, celltypes in celltype_to_lineage.items():
        for ref_ct in celltypes:
            if ref_ct in celltype_name or celltype_name in ref_ct:
                print(f"  Partial match: '{celltype_name}' → '{ref_ct}' ({lineage})")
                return lineage

    print(f"  ⚠️ No match found for: '{celltype_name}' → using 'Other'")
    return 'Other'
