"""
Enhanced Peak Cluster Annotation Module

This module provides comprehensive methods for annotating peak clusters using appropriate
statistical approaches for different data types:

CONTINUOUS DATA (accessibility values):
- Uses entropy-based analysis to detect accessibility patterns
- Detects broadly accessible, specifically accessible, and intermediate patterns
- Direct analysis of mean accessibility profiles per cluster

CATEGORICAL DATA (peak_type, gene_type, etc.):
- Uses Fisher's exact test for enrichment analysis  
- Detects highly enriched, enriched, evenly distributed patterns
- Proper statistical testing for categorical assignments

MAIN FUNCTIONS:
- run_unified_cluster_analysis(): Unified interface that routes data types automatically
- compute_simple_cluster_accessibility_entropy(): Direct entropy analysis of accessibility
- analyze_categorical_fisher_enrichment(): Fisher's exact test for categorical data
- analyze_mixed_data_types(): Original mixed analysis interface

Author: Yang-Joon Kim  
Date: 2025-08-14
"""
import scanpy as sc
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import fisher_exact


def create_cluster_pseudobulk_profiles(adata, 
                                      cluster_col: str = "leiden_coarse") -> pd.DataFrame:
    """
    Create cluster-level pseudobulk profiles by aggregating peaks within each cluster.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object where:
        - rows are peaks
        - columns are celltype_timepoint combinations
        - adata.obs[cluster_col] contains cluster labels
        
    cluster_col : str
        Column name for peak cluster assignments in adata.obs
        
    Returns:
    --------
    pd.DataFrame
        Cluster-by-celltype_timepoint pseudobulk profiles 
        (rows=clusters, cols=celltype_timepoint combinations)
    """
    import pandas as pd
    
    # Get cluster labels
    clusters = adata.obs[cluster_col].astype(str)
    
    # Convert to DataFrame for easier manipulation
    data_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        columns=adata.var_names,
        index=adata.obs_names
    )
    
    # Add cluster labels
    data_df['cluster'] = clusters
    
    # Aggregate by cluster (mean accessibility across peaks in each cluster)
    pseudobulk_profiles = data_df.groupby('cluster').mean()
    
    return pseudobulk_profiles


def compute_cluster_entropy_by_metadata(pseudobulk_profiles: pd.DataFrame,
                                       metadata_type: str = "celltype",
                                       separator: str = "_") -> pd.Series:
    """
    Compute entropy for each cluster across different metadata categories.
    
    Assumes column names are in format: celltype_timepoint, and we want to
    aggregate by celltype or timepoint to compute entropy.
    
    Parameters:
    -----------
    pseudobulk_profiles : pd.DataFrame
        Cluster-by-celltype_timepoint profiles (rows=clusters, cols=combinations)
    metadata_type : str
        Which metadata to extract: "celltype" (first part) or "timepoint" (second part)
    separator : str
        Separator used in column names
        
    Returns:
    --------
    pd.Series
        Entropy values for each cluster indexed by cluster name
    """
    
    # Extract metadata from column names
    if metadata_type == "celltype":
        # Extract first part before separator
        metadata_labels = [col.split(separator)[0] for col in pseudobulk_profiles.columns]
    elif metadata_type == "timepoint":
        # Extract second part after separator
        metadata_labels = [col.split(separator)[1] if separator in col else col 
                          for col in pseudobulk_profiles.columns]
    else:
        raise ValueError(f"metadata_type must be 'celltype' or 'timepoint', got {metadata_type}")
    
    # Aggregate pseudobulk profiles by metadata type
    metadata_df = pd.DataFrame(pseudobulk_profiles.values, 
                              index=pseudobulk_profiles.index,
                              columns=metadata_labels)
    aggregated_profiles = metadata_df.groupby(metadata_df.columns, axis=1).mean()
    
    # Compute entropy for each cluster
    entropy_values = []
    for cluster in aggregated_profiles.index:
        cluster_profile = aggregated_profiles.loc[cluster]
        entropy = compute_accessibility_entropy(cluster_profile, normalize=True)
        entropy_values.append(entropy)
    
    return pd.Series(entropy_values, index=aggregated_profiles.index, name=f"{metadata_type}_entropy")


def compute_accessibility_entropy(composition: pd.Series, normalize: bool = True) -> float:
    """
    Compute Shannon entropy of accessibility pattern across categories.
    
    Higher entropy indicates broader accessibility across categories.
    Lower entropy indicates more specific accessibility patterns.
    
    Parameters:
    -----------
    composition : pd.Series
        Counts of peaks in each category (e.g., cell types)
    normalize : bool
        Whether to normalize by maximum possible entropy
        
    Returns:
    --------
    float
        Entropy score (0-1 if normalized, 0-log2(n_categories) if not)
    """
    if composition.sum() == 0:
        return 0.0
        
    # Convert to proportions
    proportions = composition / composition.sum()
    
    # Remove zero proportions for log calculation
    proportions = proportions[proportions > 0]
    
    # Calculate Shannon entropy
    entropy = -np.sum(proportions * np.log2(proportions))
    
    if normalize and len(composition) > 1:
        max_entropy = np.log2(len(composition))  # Uniform distribution
        entropy = entropy / max_entropy
        
    return float(entropy)


def compute_gini_coefficient(composition: pd.Series) -> float:
    """
    Compute Gini coefficient of accessibility distribution.
    
    Gini coefficient measures inequality in the distribution:
    - 0: Perfect equality (uniform accessibility)
    - 1: Perfect inequality (all peaks in one category)
    
    Parameters:
    -----------
    composition : pd.Series
        Counts of peaks in each category
        
    Returns:
    --------
    float
        Gini coefficient (0-1)
    """
    if composition.sum() == 0 or len(composition) <= 1:
        return 0.0
        
    # Convert to proportions and sort
    proportions = composition / composition.sum()
    sorted_props = np.sort(proportions.values)
    
    n = len(sorted_props)
    cumsum = np.cumsum(sorted_props)
    
    # Gini coefficient formula
    gini = (n + 1 - 2 * np.sum(cumsum)) / n
    
    return float(gini)


def compute_coverage_metrics(composition: pd.Series, 
                           thresholds: List[float] = [0.05, 0.10, 0.20]) -> Dict[str, int]:
    """
    Compute coverage metrics: number of categories with >= threshold fraction.
    
    Parameters:
    -----------
    composition : pd.Series
        Counts of peaks in each category
    thresholds : List[float]
        Fraction thresholds to test
        
    Returns:
    --------
    Dict[str, int]
        Coverage counts for each threshold
    """
    if composition.sum() == 0:
        return {f"coverage_{int(t*100)}pct": 0 for t in thresholds}
    
    proportions = composition / composition.sum()
    coverage = {}
    
    for threshold in thresholds:
        count = (proportions >= threshold).sum()
        coverage[f"coverage_{int(threshold*100)}pct"] = int(count)
    
    return coverage


def classify_accessibility_pattern(composition: pd.Series,
                                 category_name: str,
                                 # Entropy thresholds
                                 broad_entropy_threshold: float = 0.75,
                                 intermediate_entropy_threshold: float = 0.40,
                                 # Dominance thresholds  
                                 high_dominance_threshold: float = 0.60,
                                 medium_dominance_threshold: float = 0.40,
                                 # Coverage thresholds
                                 broad_coverage_threshold: int = 3,
                                 min_peak_count: int = 10) -> Tuple[str, str, Dict[str, float]]:
    """
    Classify accessibility pattern using multi-metric approach.
    
    Parameters:
    -----------
    composition : pd.Series
        Counts of peaks in each category
    category_name : str
        Name of the category type (celltype, timepoint, etc.)
    broad_entropy_threshold : float
        Entropy threshold for "broad" classification
    intermediate_entropy_threshold : float
        Entropy threshold between specific and intermediate
    high_dominance_threshold : float
        Dominance threshold for "highly specific"
    medium_dominance_threshold : float
        Dominance threshold for "moderately specific"
    broad_coverage_threshold : int
        Minimum categories for "broad" classification
    min_peak_count : int
        Minimum peaks required for classification
        
    Returns:
    --------
    Tuple[str, str, Dict[str, float]]
        (annotation, confidence, metrics_dict)
    """
    
    # Initialize metrics
    metrics = {}
    
    # Check if we have enough data
    total_peaks = composition.sum()
    if total_peaks < min_peak_count:
        return f"insufficient_data", "none", {"total_peaks": total_peaks}
    
    # Core metrics
    entropy = compute_accessibility_entropy(composition, normalize=True)
    gini = compute_gini_coefficient(composition)
    coverage = compute_coverage_metrics(composition)
    
    # Dominance metrics
    max_count = composition.max()
    max_category = composition.idxmax()
    dominance = max_count / total_peaks
    
    # Store metrics
    metrics.update({
        "entropy": entropy,
        "gini": gini,  
        "dominance": dominance,
        "dominant_category": max_category,
        "total_peaks": total_peaks,
        **coverage
    })
    
    # Classification logic
    n_categories = len(composition)
    broad_coverage = coverage.get("coverage_5pct", 0) >= broad_coverage_threshold
    
    # 1. BROAD ACCESSIBILITY
    if entropy >= broad_entropy_threshold and broad_coverage:
        confidence = "high" if entropy >= 0.85 else "medium"
        return f"broadly_accessible", confidence, metrics
    
    # 2. HIGHLY SPECIFIC 
    elif dominance >= high_dominance_threshold and entropy <= intermediate_entropy_threshold:
        confidence = "high" if dominance >= 0.75 else "medium"
        return f"specific_{max_category}", confidence, metrics
    
    # 3. MODERATELY SPECIFIC
    elif dominance >= medium_dominance_threshold and entropy <= broad_entropy_threshold:
        confidence = "medium" if dominance >= 0.50 else "low"
        return f"enriched_{max_category}", confidence, metrics
        
    # 4. INTERMEDIATE/MIXED
    elif entropy >= intermediate_entropy_threshold and entropy < broad_entropy_threshold:
        # Check if there are 2-3 dominant categories
        top_categories = composition.nlargest(3)
        top_3_fraction = top_categories.sum() / total_peaks
        
        if top_3_fraction >= 0.70 and len(top_categories[top_categories/total_peaks >= 0.15]) >= 2:
            top_2_cats = "_".join(top_categories.head(2).index.astype(str))
            return f"mixed_{top_2_cats}", "medium", metrics
        else:
            return f"intermediate", "low", metrics
    
    # 5. UNCLEAR/EDGE CASES
    else:
        return f"unclear", "very_low", metrics


def annotate_peak_clusters(adata, 
                          cluster_col: str = "leiden_coarse",
                          category_col: str = "celltype", 
                          category_name: str = "celltype",
                          min_peaks_per_cluster: int = 100,
                          min_peaks_per_annotation: int = 10,
                          # Classification parameters
                          broad_entropy_threshold: float = 0.75,
                          intermediate_entropy_threshold: float = 0.40,
                          high_dominance_threshold: float = 0.60,
                          medium_dominance_threshold: float = 0.40,
                          broad_coverage_threshold: int = 3,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Annotate peak clusters using entropy-based accessibility patterns.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with peak cluster and category annotations
    cluster_col : str
        Column name for peak cluster assignments
    category_col : str  
        Column name for category to analyze (celltype, timepoint, lineage, peak_type)
    category_name : str
        Display name for the category type
    min_peaks_per_cluster : int
        Minimum peaks required per cluster for analysis
    min_peaks_per_annotation : int
        Minimum peaks required for pattern classification
    broad_entropy_threshold : float
        Entropy threshold for broad accessibility (0-1)
    intermediate_entropy_threshold : float
        Entropy threshold between specific and intermediate
    high_dominance_threshold : float
        Dominance threshold for highly specific patterns
    medium_dominance_threshold : float
        Dominance threshold for moderately specific patterns  
    broad_coverage_threshold : int
        Minimum number of categories for broad classification
    verbose : bool
        Whether to print detailed analysis
        
    Returns:
    --------
    pd.DataFrame
        Results with cluster annotations and metrics
    """
    
    if verbose:
        print(f"="*80)
        print(f"ENTROPY-BASED PEAK CLUSTER ANNOTATION")
        print(f"Analyzing {category_name} accessibility patterns")
        print(f"="*80)
    
    # Get data
    clusters = adata.obs[cluster_col].astype(str)
    categories = adata.obs[category_col].astype(str)
    
    # Filter clusters by size
    cluster_counts = clusters.value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_peaks_per_cluster].index
    valid_clusters = sorted(valid_clusters, key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    if verbose:
        print(f"Analyzing {len(valid_clusters)} clusters with ≥{min_peaks_per_cluster} peaks")
        print(f"Categories found: {sorted(categories.unique())}")
        print(f"Thresholds: broad_entropy≥{broad_entropy_threshold}, dominance≥{high_dominance_threshold}")
        print()
    
    results = []
    
    for cluster in valid_clusters:
        if verbose:
            print(f"=== CLUSTER {cluster} ===")
            
        # Get peaks in this cluster
        cluster_mask = clusters == cluster
        cluster_size = cluster_mask.sum()
        cluster_categories = categories[cluster_mask]
        
        # Get category composition
        composition = cluster_categories.value_counts()
        
        if verbose:
            print(f"Cluster size: {cluster_size:,} peaks")
            print(f"Category distribution:")
            for cat, count in composition.head(5).items():
                pct = count / cluster_size * 100
                print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")
            if len(composition) > 5:
                print(f"  ... and {len(composition)-5} others")
        
        # Classify accessibility pattern
        annotation, confidence, metrics = classify_accessibility_pattern(
            composition=composition,
            category_name=category_name,
            broad_entropy_threshold=broad_entropy_threshold,
            intermediate_entropy_threshold=intermediate_entropy_threshold,
            high_dominance_threshold=high_dominance_threshold,
            medium_dominance_threshold=medium_dominance_threshold,
            broad_coverage_threshold=broad_coverage_threshold,
            min_peak_count=min_peaks_per_annotation
        )
        
        if verbose:
            print(f"Metrics:")
            print(f"  Entropy: {metrics.get('entropy', 0):.3f} (1.0 = perfectly broad)")
            print(f"  Gini: {metrics.get('gini', 0):.3f} (1.0 = perfectly specific)")
            print(f"  Dominance: {metrics.get('dominance', 0):.3f} (dominant: {metrics.get('dominant_category', 'N/A')})")
            print(f"  Coverage 5%+: {metrics.get('coverage_5pct', 0)}/{len(composition)} categories")
            print(f"")
            print(f"→ ANNOTATION: {annotation} ({confidence} confidence)")
            print()
        
        # Store results
        result = {
            'cluster': cluster,
            f'{category_name}_annotation': annotation,
            f'{category_name}_confidence': confidence,
            'cluster_size': cluster_size,
            'entropy': metrics.get('entropy', np.nan),
            'gini': metrics.get('gini', np.nan),
            'dominance': metrics.get('dominance', np.nan),
            'dominant_category': metrics.get('dominant_category', None),
            'coverage_5pct': metrics.get('coverage_5pct', 0),
            'coverage_10pct': metrics.get('coverage_10pct', 0), 
            'coverage_20pct': metrics.get('coverage_20pct', 0),
            'n_categories': len(composition)
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"="*80)
        print(f"ANNOTATION SUMMARY")
        print(f"="*80)
        
        # Summary by pattern type
        annotation_counts = results_df[f'{category_name}_annotation'].value_counts()
        print(f"\nPattern distribution:")
        for pattern, count in annotation_counts.items():
            print(f"  {pattern:30s}: {count:2d} clusters")
            
        # Summary by confidence
        confidence_counts = results_df[f'{category_name}_confidence'].value_counts()
        print(f"\nConfidence distribution:")
        for conf, count in confidence_counts.items():
            print(f"  {conf:15s}: {count:2d} clusters")
    
    return results_df


def add_annotations_to_adata(adata, results_df, category_name: str = "celltype", 
                           cluster_col: str = "leiden_coarse"):
    """
    Add entropy-based annotations back to AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object to modify
    results_df : pd.DataFrame
        Results from annotate_peak_clusters
    category_name : str
        Category name for column naming
    cluster_col : str
        Cluster column name
    """
    
    # Create mappings
    cluster_to_annotation = dict(zip(results_df['cluster'], results_df[f'{category_name}_annotation']))
    cluster_to_confidence = dict(zip(results_df['cluster'], results_df[f'{category_name}_confidence']))
    cluster_to_entropy = dict(zip(results_df['cluster'], results_df['entropy']))
    cluster_to_gini = dict(zip(results_df['cluster'], results_df['gini']))
    cluster_to_dominance = dict(zip(results_df['cluster'], results_df['dominance']))
    
    # Map to adata
    clusters = adata.obs[cluster_col].astype(str)
    adata.obs[f'{cluster_col}_{category_name}_entropy_annotation'] = clusters.map(cluster_to_annotation).fillna('unknown')
    adata.obs[f'{cluster_col}_{category_name}_entropy_confidence'] = clusters.map(cluster_to_confidence).fillna('none')
    adata.obs[f'{cluster_col}_{category_name}_entropy'] = clusters.map(cluster_to_entropy).fillna(np.nan)
    adata.obs[f'{cluster_col}_{category_name}_gini'] = clusters.map(cluster_to_gini).fillna(np.nan)
    adata.obs[f'{cluster_col}_{category_name}_dominance'] = clusters.map(cluster_to_dominance).fillna(np.nan)
    
    print(f"Added entropy-based {category_name} annotation columns to adata.obs:")
    print(f"  - '{cluster_col}_{category_name}_entropy_annotation'")
    print(f"  - '{cluster_col}_{category_name}_entropy_confidence'")
    print(f"  - '{cluster_col}_{category_name}_entropy'")
    print(f"  - '{cluster_col}_{category_name}_gini'")
    print(f"  - '{cluster_col}_{category_name}_dominance'")


def run_pseudobulk_entropy_analysis(adata,
                                   cluster_col: str = "leiden_coarse",
                                   separator: str = "_",
                                   metadata_types: List[str] = ["celltype", "timepoint"],
                                   verbose: bool = True) -> Dict[str, pd.Series]:
    """
    Run pseudobulk-based entropy analysis for peak clusters.
    
    First creates cluster-level pseudobulk profiles by averaging accessibility
    over peaks within each cluster. Then computes entropy for different metadata
    types extracted from column names.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object where:
        - rows are peaks with cluster labels in .obs[cluster_col]
        - columns are celltype_timepoint combinations
    cluster_col : str
        Column name for peak cluster assignments
    separator : str
        Separator used in column names (e.g., "_" for "celltype_timepoint")
    metadata_types : List[str]
        List of metadata types to analyze (e.g., ["celltype", "timepoint"])
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    Dict[str, pd.Series]
        Dictionary with entropy values for each metadata type
    """
    
    if verbose:
        print(f"="*80)
        print(f"PSEUDOBULK-BASED ENTROPY ANALYSIS")
        print(f"="*80)
        print(f"Creating cluster-level pseudobulk profiles...")
    
    # Step 1: Create cluster-level pseudobulk profiles
    pseudobulk_profiles = create_cluster_pseudobulk_profiles(adata, cluster_col)
    
    if verbose:
        print(f"Created profiles for {len(pseudobulk_profiles)} clusters")
        print(f"Across {len(pseudobulk_profiles.columns)} celltype_timepoint combinations")
        print()
    
    # Step 2: Compute entropy for each metadata type
    entropy_results = {}
    
    for metadata_type in metadata_types:
        if verbose:
            print(f"Computing {metadata_type} entropy for each cluster...")
        
        try:
            entropy_values = compute_cluster_entropy_by_metadata(
                pseudobulk_profiles, 
                metadata_type=metadata_type,
                separator=separator
            )
            entropy_results[metadata_type] = entropy_values
            
            if verbose:
                print(f"  Mean {metadata_type} entropy: {entropy_values.mean():.3f}")
                print(f"  Std {metadata_type} entropy: {entropy_values.std():.3f}")
                print(f"  Range: {entropy_values.min():.3f} - {entropy_values.max():.3f}")
                print()
                
        except Exception as e:
            print(f"Error computing {metadata_type} entropy: {e}")
            continue
    
    if verbose:
        print(f"="*80)
        print(f"ENTROPY ANALYSIS SUMMARY")
        print(f"="*80)
        
        for metadata_type, entropy_vals in entropy_results.items():
            print(f"\n{metadata_type.upper()} ENTROPY DISTRIBUTION:")
            print(f"{'Cluster':>8} {metadata_type + '_entropy':>15}")
            print("-" * 25)
            
            # Show top/bottom clusters by entropy
            sorted_entropy = entropy_vals.sort_values(ascending=False)
            
            print("Top 5 (highest entropy - most broad):")
            for cluster, entropy in sorted_entropy.head(5).items():
                print(f"{cluster:>8} {entropy:>15.3f}")
            
            print("\nBottom 5 (lowest entropy - most specific):")
            for cluster, entropy in sorted_entropy.tail(5).items():
                print(f"{cluster:>8} {entropy:>15.3f}")
            print()
    
    return entropy_results


def run_all_entropy_analyses(adata, 
                           cluster_col: str = "leiden_coarse",
                           celltype_col: str = "celltype",
                           timepoint_col: str = "timepoint", 
                           lineage_col: str = "lineage",
                           peak_type_col: str = "peak_type",
                           min_peaks_per_cluster: int = 100,
                           verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run entropy-based annotation for all category types.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    cluster_col : str
        Cluster column name
    celltype_col, timepoint_col, lineage_col, peak_type_col : str
        Category column names
    min_peaks_per_cluster : int
        Minimum peaks per cluster
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Results for each category type
    """
    
    results = {}
    category_configs = [
        (celltype_col, "celltype"),
        (timepoint_col, "timepoint"),
        (lineage_col, "lineage"),
        (peak_type_col, "peak_type")
    ]
    
    for col_name, category_name in category_configs:
        if col_name not in adata.obs.columns:
            if verbose:
                print(f"Skipping {category_name}: column '{col_name}' not found")
            continue
            
        if verbose:
            print(f"\n{'='*80}")
            print(f"ANALYZING {category_name.upper()} ACCESSIBILITY PATTERNS")
            print(f"{'='*80}")
        
        results[category_name] = annotate_peak_clusters(
            adata=adata,
            cluster_col=cluster_col,
            category_col=col_name,
            category_name=category_name,
            min_peaks_per_cluster=min_peaks_per_cluster,
            verbose=verbose
        )
        
        # Add to adata
        add_annotations_to_adata(adata, results[category_name], category_name, cluster_col)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY - ENTROPY-BASED ANNOTATIONS")
        print(f"{'='*80}")
        
        for category_name in results.keys():
            col_name = f'{cluster_col}_{category_name}_entropy_annotation'
            if col_name in adata.obs.columns:
                print(f"\n{category_name.capitalize()} patterns:")
                pattern_counts = adata.obs[col_name].value_counts().head(10)
                for pattern, count in pattern_counts.items():
                    print(f"  {pattern:35s}: {count:6,} peaks")
    
    return results


def compare_methods(adata, entropy_results, fisher_results, category_name: str = "celltype"):
    """
    Compare entropy-based vs Fisher's test annotations.
    
    Parameters:
    -----------
    adata : AnnData
        Data object with both annotation types
    entropy_results : pd.DataFrame
        Results from entropy-based method
    fisher_results : pd.DataFrame  
        Results from Fisher's test method
    category_name : str
        Category type to compare
    """
    
    print(f"="*80)
    print(f"METHOD COMPARISON: Entropy vs Fisher's Test ({category_name})")
    print(f"="*80)
    
    # Merge results
    comparison = entropy_results.merge(
        fisher_results[['cluster', f'{category_name}_annotation']], 
        on='cluster', 
        suffixes=('_entropy', '_fisher')
    )
    
    print(f"Cluster-level comparison:")
    print(f"{'Cluster':>7} {'Entropy Method':>25} {'Fisher Method':>25} {'Match':>8}")
    print("-" * 70)
    
    matches = 0
    for _, row in comparison.iterrows():
        entropy_ann = row[f'{category_name}_annotation_entropy']
        fisher_ann = row[f'{category_name}_annotation_fisher']
        
        # Simplify comparison
        entropy_simple = entropy_ann.split('_')[0] if '_' in entropy_ann else entropy_ann
        fisher_simple = fisher_ann.split('_')[0] if '_' in fisher_ann else fisher_ann
        
        match = "✓" if entropy_simple == fisher_simple else "✗"
        if match == "✓":
            matches += 1
            
        print(f"{row['cluster']:>7} {entropy_ann[:24]:>25} {fisher_ann[:24]:>25} {match:>8}")
    
    print(f"\nAgreement: {matches}/{len(comparison)} clusters ({matches/len(comparison)*100:.1f}%)")
    
    # Pattern type analysis
    print(f"\nPattern detection capabilities:")
    entropy_patterns = entropy_results[f'{category_name}_annotation'].apply(
        lambda x: x.split('_')[0]).value_counts()
    
    for pattern, count in entropy_patterns.items():
        print(f"  {pattern:20s}: {count:2d} clusters (entropy method only)")


def analyze_mixed_data_types(adata, cluster_col='leiden_coarse'):
    """
    Analyze both continuous accessibility and categorical metadata.
    """
    
    # Continuous accessibility data → Entropy analysis
    continuous_results = run_simple_cluster_entropy_analysis(adata, cluster_col)
    
    # Categorical metadata → Fisher's exact test
    categorical_results = {}
    
    if 'peak_type' in adata.obs.columns:
        categorical_results['peak_type'] = analyze_categorical_fisher_enrichment(
            adata, cluster_col, 'peak_type', 'peak_type'
        )
    
    return {
        'continuous': continuous_results,
        'categorical': categorical_results
    }


def analyze_categorical_fisher_enrichment(adata, cluster_col, category_col, category_name,
                                        min_peaks_per_cluster=100,
                                        # Fisher's test thresholds (reuse our calibrated values)
                                        high_or_threshold=5.0,
                                        medium_or_threshold=2.0,
                                        high_representation_threshold=30.0,
                                        medium_representation_threshold=15.0,
                                        verbose=True):
    """
    Analyze categorical enrichment using Fisher's exact test (our original approach).
    
    This is perfect for peak_type, where each peak has a definitive categorical label.
    """
    
    # Input validation
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in adata.obs")
    
    if category_col not in adata.obs.columns:
        raise ValueError(f"Category column '{category_col}' not found in adata.obs")
    
    if len(adata.obs) == 0:
        raise ValueError("Empty adata object - no peaks found")
    
    if verbose:
        print(f"="*80)
        print(f"CATEGORICAL ENRICHMENT ANALYSIS: {category_name.upper()}")
        print(f"Using Fisher's exact test for categorical data")
        print(f"="*80)
    
    clusters = adata.obs[cluster_col].astype(str)
    categories = adata.obs[category_col].astype(str)
    
    # Check for missing values
    if clusters.isna().sum() > 0:
        if verbose:
            print(f"Warning: {clusters.isna().sum()} missing cluster assignments")
        clusters = clusters.fillna('unknown')
    
    if categories.isna().sum() > 0:
        if verbose:
            print(f"Warning: {categories.isna().sum()} missing category assignments") 
        categories = categories.fillna('unknown')
    
    # Get global distribution
    global_dist = categories.value_counts()
    
    cluster_results = []
    
    for cluster in sorted(set(clusters), key=lambda x: int(x) if x.isdigit() else float('inf')):
        cluster_mask = clusters == cluster
        cluster_size = cluster_mask.sum()
        
        if cluster_size < min_peaks_per_cluster:
            continue
            
        if verbose:
            print(f"\n=== CLUSTER {cluster} ===")
            print(f"Cluster size: {cluster_size} peaks")
        
        cluster_categories = categories[cluster_mask].value_counts()
        
        # Test each category for enrichment
        enrichment_scores = []
        
        for category, count_in_cluster in cluster_categories.items():
            if count_in_cluster < 5:  # Skip small counts
                continue
                
            category_mask = categories == category
            
            # Fisher's exact test (our proven approach)
            a = count_in_cluster
            b = category_mask.sum() - a  
            c = cluster_size - a
            d = len(adata.obs) - cluster_size - b
            
            # Ensure non-negative values for Fisher's test
            if any(x < 0 for x in [a, b, c, d]):
                if verbose:
                    print(f"    Warning: Invalid Fisher's test values for {category}: a={a}, b={b}, c={c}, d={d}")
                continue
            
            try:
                odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
            except ValueError as e:
                if verbose:
                    print(f"    Warning: Fisher's test failed for {category}: {e}")
                odds_ratio, p_value = 1.0, 1.0  # Neutral values as fallback
            
            pct_cluster = (a / cluster_size) * 100
            expected_count = (cluster_size * category_mask.sum()) / len(adata.obs)
            fold_enrichment = a / expected_count if expected_count > 0 else np.inf
            
            enrichment_scores.append({
                'category': category,
                'count': count_in_cluster,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'pct_of_cluster': pct_cluster,
                'fold_enrichment': fold_enrichment
            })
            
            if verbose:
                print(f"  {category:15s}: {count_in_cluster:4d} ({pct_cluster:5.1f}%) "
                      f"OR={odds_ratio:5.2f} FE={fold_enrichment:4.1f}x p={p_value:.2e}")
        
        # Sort by odds ratio
        enrichment_scores.sort(key=lambda x: -x['odds_ratio'])
        
        if enrichment_scores:
            top_category = enrichment_scores[0]
            
            # Apply our calibrated Fisher's thresholds
            if (top_category['odds_ratio'] >= high_or_threshold and 
                top_category['pct_of_cluster'] >= high_representation_threshold):
                pattern = f"highly_enriched_{top_category['category']}"
                confidence = "high"
                
            elif (top_category['odds_ratio'] >= medium_or_threshold and 
                  top_category['pct_of_cluster'] >= medium_representation_threshold):
                pattern = f"enriched_{top_category['category']}"
                confidence = "medium"
                
            elif len(enrichment_scores) >= 3 and top_category['odds_ratio'] < medium_or_threshold:
                # Multiple categories, none strongly enriched = evenly distributed
                pattern = "evenly_distributed"
                confidence = "medium"
                
            else:
                pattern = "mixed_categorical"
                confidence = "low"
            
            if verbose:
                print(f"→ PATTERN: {pattern} ({confidence} confidence)")
        
        else:
            pattern = "unclear"
            confidence = "none"
        
        cluster_results.append({
            'cluster': cluster,
            'cluster_size': cluster_size,
            'pattern': pattern,
            'confidence': confidence,
            'top_category': top_category['category'] if enrichment_scores else None,
            'top_odds_ratio': top_category['odds_ratio'] if enrichment_scores else None,
            'top_pct_cluster': top_category['pct_of_cluster'] if enrichment_scores else None,
            'n_categories': len(cluster_categories)
        })
    
    return pd.DataFrame(cluster_results)


def compute_simple_cluster_accessibility_entropy(adata, 
                                               cluster_col='leiden_coarse',
                                               accessibility_cols=None,
                                               min_peaks_per_cluster=100,
                                               verbose=True):
    """
    SIMPLE approach: Compute entropy from mean accessibility profile per cluster.
    
    This function computes entropy directly from the accessibility values in the data matrix,
    treating accessibility as continuous values rather than categorical assignments.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with accessibility data
    cluster_col : str
        Column name for peak cluster assignments
    accessibility_cols : list or None
        List of accessibility column names. If None, auto-detects.
    min_peaks_per_cluster : int
        Minimum peaks required per cluster for analysis
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    pd.DataFrame
        Results with cluster patterns and metrics
    """
    
    # Input validation
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in adata.obs")
    
    if adata.X is None:
        raise ValueError("adata.X is None - no data matrix found")
    
    if len(adata.obs) == 0:
        raise ValueError("Empty adata object - no peaks found")
    
    # Auto-detect accessibility columns
    if accessibility_cols is None:
        accessibility_cols = [col for col in adata.var.index if 'accessibility' in col.lower()]
        
    if len(accessibility_cols) == 0:
        if verbose:
            print("Warning: No accessibility columns found. Using all columns.")
        accessibility_cols = list(adata.var.index)
        if len(accessibility_cols) == 0:
            raise ValueError("No columns found in adata.var.index")
    
    # Validate accessibility columns exist
    missing_cols = [col for col in accessibility_cols if col not in adata.var.index]
    if missing_cols:
        if verbose:
            print(f"Warning: {len(missing_cols)} accessibility columns not found: {missing_cols[:3]}...")
        accessibility_cols = [col for col in accessibility_cols if col in adata.var.index]
        
    if len(accessibility_cols) == 0:
        raise ValueError("No valid accessibility columns found after filtering")
    
    clusters = adata.obs[cluster_col].astype(str)
    cluster_results = []
    
    if verbose:
        print(f"Computing simple accessibility entropy for {len(set(clusters))} clusters")
        print(f"Using {len(accessibility_cols)} accessibility columns")
    
    for cluster in sorted(set(clusters), key=lambda x: int(x) if x.isdigit() else float('inf')):
        cluster_mask = clusters == cluster
        cluster_size = cluster_mask.sum()
        
        if cluster_size < min_peaks_per_cluster:
            continue
            
        # Get mean accessibility profile for this cluster  
        if hasattr(adata.X, 'toarray'):
            cluster_data = adata.X[cluster_mask, :].toarray()
        else:
            cluster_data = adata.X[cluster_mask, :]
            
        cluster_accessibility = cluster_data.mean(axis=0)
        
        # Create accessibility series
        accessibility_profile = pd.Series(
            cluster_accessibility,
            index=adata.var.index
        )
        
        # Filter to accessibility columns only
        if accessibility_cols:
            accessibility_profile = accessibility_profile[accessibility_cols]
        
        # Remove very low values to focus on meaningful accessibility
        accessibility_profile = accessibility_profile[accessibility_profile >= 0.1]
        
        if len(accessibility_profile) == 0:
            if verbose:
                print(f"Cluster {cluster}: No significant accessibility values found (all < 0.1)")
            continue
        
        # Safety check for edge cases
        if accessibility_profile.sum() == 0:
            if verbose:
                print(f"Cluster {cluster}: Sum of accessibility is zero")
            continue
        
        # Compute entropy and other metrics with error handling
        try:
            entropy = compute_accessibility_entropy(accessibility_profile, normalize=True)
            dominance = accessibility_profile.max() / accessibility_profile.sum()
            dominant_category = accessibility_profile.idxmax()
        except Exception as e:
            if verbose:
                print(f"Cluster {cluster}: Error computing metrics: {e}")
            continue
        
        # Validate computed values
        if not np.isfinite(entropy) or not np.isfinite(dominance):
            if verbose:
                print(f"Cluster {cluster}: Invalid entropy ({entropy}) or dominance ({dominance})")
            continue
        
        # Classify pattern based on entropy and dominance
        if entropy >= 0.75 and len(accessibility_profile) >= 4:
            pattern = "broadly_accessible"
            confidence = "high"
        elif dominance >= 0.6 and entropy <= 0.4:
            pattern = f"specific_{dominant_category}"
            confidence = "high"
        elif dominance >= 0.4 and entropy <= 0.6:
            pattern = f"enriched_{dominant_category}"
            confidence = "medium"
        else:
            pattern = f"intermediate_{dominant_category}"
            confidence = "low"
        
        cluster_results.append({
            'cluster': cluster,
            'pattern': pattern,
            'confidence': confidence,
            'entropy': entropy,
            'dominance': dominance,
            'dominant_category': dominant_category,
            'cluster_size': cluster_size,
            'n_accessible_categories': len(accessibility_profile)
        })
        
        if verbose:
            print(f"Cluster {cluster}: {pattern} (entropy={entropy:.3f}, dominance={dominance:.3f})")
    
    return pd.DataFrame(cluster_results)


def run_simple_cluster_entropy_analysis(adata, cluster_col='leiden_coarse'):
    """
    Simple wrapper for continuous entropy analysis that matches the expected interface.
    
    This function provides a simplified interface to the new simple entropy analysis functionality
    for use in mixed data type analysis.
    """
    return compute_simple_cluster_accessibility_entropy(
        adata=adata,
        cluster_col=cluster_col,
        accessibility_cols=None,  # Auto-detect
        verbose=False
    )


def run_unified_cluster_analysis(adata, cluster_col='leiden_coarse', verbose=True):
    """
    Run appropriate analysis for each data type:
    - Continuous accessibility → Simple entropy
    - Categorical metadata → Fisher's exact test
    
    This is the main entry point for comprehensive cluster analysis that automatically
    routes different data types to their most appropriate analysis methods.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with cluster assignments and various data types
    cluster_col : str
        Column name for peak cluster assignments
    verbose : bool
        Whether to print detailed analysis results
        
    Returns:
    --------
    dict
        Results dictionary with analysis results for each data type
    """
    
    results = {}
    
    # 1. Continuous accessibility data (celltype, timepoint, lineage)
    if verbose:
        print("="*80)
        print("CONTINUOUS ACCESSIBILITY ANALYSIS")
        print("="*80)
    
    # Detect available accessibility data types
    all_cols = list(adata.var.index)
    
    for metadata_type in ['celltype', 'timepoint', 'lineage']:
        accessibility_cols = None
        
        if metadata_type == 'celltype':
            # Look for general accessibility columns (not timepoint or lineage specific)
            # More flexible patterns: accessibility_*, acc_*, *accessibility*
            accessibility_cols = [col for col in all_cols 
                                 if (('accessibility' in col.lower() or col.lower().startswith('acc_')) and 
                                     'somites' not in col.lower() and 
                                     'timepoint' not in col.lower() and
                                     'lineage' not in col.lower())]
        elif metadata_type == 'timepoint':
            # Look for timepoint-specific columns  
            # Patterns: *somites*, *timepoint*, *tp*, *time*
            accessibility_cols = [col for col in all_cols 
                                 if (('accessibility' in col.lower() or col.lower().startswith('acc_')) and 
                                     ('somites' in col.lower() or 'timepoint' in col.lower() or 
                                      '_tp' in col.lower() or 'time' in col.lower()))]
        elif metadata_type == 'lineage':
            # Look for lineage-specific columns
            # Patterns: *lineage*, *lin*, *linage*  
            accessibility_cols = [col for col in all_cols 
                                 if (('accessibility' in col.lower() or col.lower().startswith('acc_')) and 
                                     ('lineage' in col.lower() or '_lin' in col.lower() or 
                                      'linage' in col.lower()))]
        
        if accessibility_cols and len(accessibility_cols) > 0:
            if verbose:
                print(f"\n--- {metadata_type.upper()} ANALYSIS ---")
                print(f"Found {len(accessibility_cols)} accessibility columns")
            
            try:
                result_df = compute_simple_cluster_accessibility_entropy(
                    adata, cluster_col, accessibility_cols, verbose=verbose
                )
                if len(result_df) > 0:
                    results[metadata_type] = result_df
                else:
                    if verbose:
                        print(f"No clusters met minimum requirements for {metadata_type} analysis")
            except Exception as e:
                if verbose:
                    print(f"Error in {metadata_type} analysis: {e}")
                continue
        else:
            if verbose:
                print(f"\n--- {metadata_type.upper()} ANALYSIS ---")
                print(f"No accessibility columns found for {metadata_type}")
    
    # 2. Categorical data (peak_type and other categorical columns)
    if verbose:
        print("\n" + "="*80)
        print("CATEGORICAL DATA ANALYSIS")
        print("="*80)
    
    categorical_columns = ['peak_type', 'gene_type', 'regulatory_element']
    
    for col_name in categorical_columns:
        if col_name in adata.obs.columns:
            if verbose:
                print(f"\n--- {col_name.upper()} ANALYSIS ---")
            
            try:
                result_df = analyze_categorical_fisher_enrichment(
                    adata, cluster_col, col_name, col_name, verbose=verbose
                )
                if len(result_df) > 0:
                    results[col_name] = result_df
                else:
                    if verbose:
                        print(f"No clusters met minimum requirements for {col_name} analysis")
            except Exception as e:
                if verbose:
                    print(f"Error in {col_name} analysis: {e}")
                continue
        else:
            if verbose:
                print(f"\n--- {col_name.upper()} ANALYSIS ---")
                print(f"Column '{col_name}' not found in adata.obs")
    
    # Summary
    if verbose:
        print("\n" + "="*80)
        print("UNIFIED ANALYSIS SUMMARY")
        print("="*80)
        
        for data_type, result_df in results.items():
            if len(result_df) > 0:
                print(f"\n{data_type.capitalize()}:")
                if 'pattern' in result_df.columns:
                    pattern_counts = result_df['pattern'].value_counts()
                    for pattern, count in pattern_counts.head(5).items():
                        print(f"  {pattern:30s}: {count:2d} clusters")
                else:
                    print(f"  Analyzed {len(result_df)} clusters")
    
    return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

"""
# Example 1: Basic pseudobulk entropy analysis
import scanpy as sc
from module_annotate_peak_clusters import run_pseudobulk_entropy_analysis

# Load your data
adata_peaks_ct_tp = sc.read_h5ad("path/to/your/adata_peaks_ct_tp.h5ad")

# Run pseudobulk-based entropy analysis
entropy_results = run_pseudobulk_entropy_analysis(
    adata_peaks_ct_tp,
    cluster_col="leiden_coarse",  # Column with cluster labels
    separator="_",                # Separator in column names (celltype_timepoint)
    metadata_types=["celltype", "timepoint"],
    verbose=True
)

# Access results
celltype_entropy = entropy_results["celltype"]
timepoint_entropy = entropy_results["timepoint"]

# Print cluster rankings
print("Clusters with highest celltype entropy (most broad):")
print(celltype_entropy.sort_values(ascending=False).head(10))

print("Clusters with lowest celltype entropy (most specific):")
print(celltype_entropy.sort_values(ascending=True).head(10))


# Example 2: Create and inspect pseudobulk profiles manually
from module_annotate_peak_clusters import create_cluster_pseudobulk_profiles, compute_cluster_entropy_by_metadata

# Create pseudobulk profiles
pseudobulk_profiles = create_cluster_pseudobulk_profiles(
    adata_peaks_ct_tp, 
    cluster_col="leiden_coarse"
)

print(f"Pseudobulk profiles shape: {pseudobulk_profiles.shape}")
print("First few clusters and celltype_timepoint combinations:")
print(pseudobulk_profiles.iloc[:5, :5])

# Compute entropy for specific metadata type
celltype_entropy = compute_cluster_entropy_by_metadata(
    pseudobulk_profiles,
    metadata_type="celltype",
    separator="_"
)

timepoint_entropy = compute_cluster_entropy_by_metadata(
    pseudobulk_profiles,
    metadata_type="timepoint", 
    separator="_"
)

# Create comparison DataFrame
import pandas as pd
entropy_comparison = pd.DataFrame({
    'celltype_entropy': celltype_entropy,
    'timepoint_entropy': timepoint_entropy
})

print("Entropy comparison:")
print(entropy_comparison.head(10))


# Example 3: Visualize entropy results
import matplotlib.pyplot as plt
import seaborn as sns

# Plot entropy distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Celltype entropy distribution
axes[0].hist(celltype_entropy.values, bins=20, alpha=0.7, color='blue')
axes[0].set_xlabel('Celltype Entropy')
axes[0].set_ylabel('Number of Clusters')
axes[0].set_title('Distribution of Celltype Entropy')
axes[0].axvline(celltype_entropy.mean(), color='red', linestyle='--', 
                label=f'Mean: {celltype_entropy.mean():.3f}')
axes[0].legend()

# Timepoint entropy distribution
axes[1].hist(timepoint_entropy.values, bins=20, alpha=0.7, color='green')
axes[1].set_xlabel('Timepoint Entropy')
axes[1].set_ylabel('Number of Clusters')
axes[1].set_title('Distribution of Timepoint Entropy')
axes[1].axvline(timepoint_entropy.mean(), color='red', linestyle='--',
                label=f'Mean: {timepoint_entropy.mean():.3f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('entropy_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot comparing celltype vs timepoint entropy
plt.figure(figsize=(8, 6))
plt.scatter(celltype_entropy, timepoint_entropy, alpha=0.7)
plt.xlabel('Celltype Entropy')
plt.ylabel('Timepoint Entropy') 
plt.title('Celltype vs Timepoint Entropy by Cluster')

# Add diagonal line
max_val = max(celltype_entropy.max(), timepoint_entropy.max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='x=y')
plt.legend()

# Annotate some interesting points
for cluster in entropy_comparison.index:
    ct_ent = celltype_entropy[cluster]
    tp_ent = timepoint_entropy[cluster]
    
    # Annotate clusters with very different entropies
    if abs(ct_ent - tp_ent) > 0.3:
        plt.annotate(f'C{cluster}', (ct_ent, tp_ent), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('entropy_comparison_scatter.png', dpi=300, bbox_inches='tight')
plt.show()


# Example 4: Add entropy results back to AnnData
# Add entropy values as cluster-level annotations
cluster_to_celltype_entropy = dict(celltype_entropy)
cluster_to_timepoint_entropy = dict(timepoint_entropy)

# Map to individual peaks
clusters = adata_peaks_ct_tp.obs["leiden_coarse"].astype(str)
adata_peaks_ct_tp.obs['cluster_celltype_entropy'] = clusters.map(cluster_to_celltype_entropy).fillna(np.nan)
adata_peaks_ct_tp.obs['cluster_timepoint_entropy'] = clusters.map(cluster_to_timepoint_entropy).fillna(np.nan)

print("Added entropy annotations to adata.obs:")
print("- 'cluster_celltype_entropy'")
print("- 'cluster_timepoint_entropy'")


# Example 5: Identify interesting cluster types
# High celltype entropy (celltype-broad)
celltype_broad = celltype_entropy[celltype_entropy > 0.8].index.tolist()
print(f"Celltype-broad clusters (entropy > 0.8): {celltype_broad}")

# Low celltype entropy (celltype-specific)  
celltype_specific = celltype_entropy[celltype_entropy < 0.3].index.tolist()
print(f"Celltype-specific clusters (entropy < 0.3): {celltype_specific}")

# High timepoint entropy (timepoint-broad)
timepoint_broad = timepoint_entropy[timepoint_entropy > 0.8].index.tolist()
print(f"Timepoint-broad clusters (entropy > 0.8): {timepoint_broad}")

# Low timepoint entropy (timepoint-specific)
timepoint_specific = timepoint_entropy[timepoint_entropy < 0.3].index.tolist()  
print(f"Timepoint-specific clusters (entropy < 0.3): {timepoint_specific}")

# Clusters that are celltype-specific but timepoint-broad
ct_specific_tp_broad = [c for c in celltype_specific if c in timepoint_broad]
print(f"Celltype-specific + timepoint-broad: {ct_specific_tp_broad}")

# Clusters that are timepoint-specific but celltype-broad
tp_specific_ct_broad = [c for c in timepoint_specific if c in celltype_broad]
print(f"Timepoint-specific + celltype-broad: {tp_specific_ct_broad}")
"""