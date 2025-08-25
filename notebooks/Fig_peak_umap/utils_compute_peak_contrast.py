"""
A module for computing cluster enrichment and contrast metrics
This module provides functions for analyzing peak cluster enrichment across different 
metadata types (celltype, timepoint, lineage, peak_type) using contrast-weighted 
Fisher's exact tests.

Author: Yang-Joon Kim
Date: 2025-08-14
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact


def _cluster_sort_key(value):
	"""Sort clusters numerically when possible; fallback to lexicographic.

	Returns a tuple so that numeric labels come first in true numeric order,
	followed by any non-numeric labels in lexicographic order.
	"""
	s = str(value)
	if s.isdigit():
		return (0, int(s))
	return (1, s)


def _compute_annotation_metrics(annotation,
								 count_in_cluster,
								 annotations,
								 contrasts,
								 cluster_mask,
								 cluster_size,
								 total_size,
								 annotation_type):
	"""Compute enrichment and contrast metrics for a single annotation within a cluster."""
	annotation_mask = (annotations == annotation)
	# Fisher's exact test components
	a = count_in_cluster
	b = int(annotation_mask.sum()) - a
	c = int(cluster_size) - a
	d = int(total_size) - int(cluster_size) - b
	odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
	# Contrast metrics
	cluster_annotation_contrasts = contrasts[cluster_mask & annotation_mask]
	mean_contrast = float(cluster_annotation_contrasts.mean()) if cluster_annotation_contrasts.size else 0.0
	median_contrast = float(np.median(cluster_annotation_contrasts)) if cluster_annotation_contrasts.size else 0.0
	high_contrast_count = int((cluster_annotation_contrasts >= 4).sum())
	moderate_contrast_count = int((cluster_annotation_contrasts >= 2).sum())
	broad_contrast_count = int((cluster_annotation_contrasts < 2).sum())
	# Combined score with saturation on contrast bonus
	contrast_bonus = 1 + float(np.tanh(mean_contrast / 5.0))
	combined_score = float(odds_ratio) * contrast_bonus
	# Interpretability helpers
	pct_cluster_is_annotation = (a / cluster_size) * 100 if cluster_size else 0.0
	pct_annotation_in_cluster = (a / (a + b)) * 100 if (a + b) else 0.0
	expected_count = (cluster_size * int(annotation_mask.sum())) / total_size if total_size else 0.0
	fold_enrichment = (a / expected_count) if expected_count > 0 else np.inf
	return {
		f"{annotation_type}": annotation,
		'count': a,
		'odds_ratio': float(odds_ratio),
		'p_value': float(p_value),
		'fold_enrichment': float(fold_enrichment),
		'mean_contrast': mean_contrast,
		'median_contrast': median_contrast,
		'high_contrast_count': high_contrast_count,
		'moderate_contrast_count': moderate_contrast_count,
		'broad_contrast_count': broad_contrast_count,
		'contrast_bonus': contrast_bonus,
		'combined_score': combined_score,
		'pct_of_cluster': float(pct_cluster_is_annotation),
		'pct_of_annotation_total': float(pct_annotation_in_cluster),
		'expected_count': float(expected_count)
	}


def _determine_confidence(top_annotation_row,
							high_p_threshold,
							high_or_threshold,
							medium_p_threshold,
							medium_or_threshold,
							high_contrast_threshold,
							medium_contrast_threshold,
							good_representation_threshold,
							fair_representation_threshold,
							annotation_type):
	"""Return (best_annotation_label, confidence_level)."""
	high_enrichment = (top_annotation_row['p_value'] < high_p_threshold and
					 top_annotation_row['odds_ratio'] > high_or_threshold)
	high_contrast = (top_annotation_row['mean_contrast'] > high_contrast_threshold)
	good_representation = (top_annotation_row['pct_of_cluster'] > good_representation_threshold)
	medium_enrichment = (top_annotation_row['p_value'] < medium_p_threshold and
					   top_annotation_row['odds_ratio'] > medium_or_threshold)
	medium_contrast = (top_annotation_row['mean_contrast'] > medium_contrast_threshold)
	fair_representation = (top_annotation_row['pct_of_cluster'] > fair_representation_threshold)
	if high_enrichment and high_contrast and good_representation:
		return top_annotation_row[f'{annotation_type}'], 'high'
	if (high_enrichment and medium_contrast) or (medium_enrichment and high_contrast):
		return top_annotation_row[f'{annotation_type}'], 'medium'
	if medium_enrichment and medium_contrast and fair_representation:
		return top_annotation_row[f'{annotation_type}'], 'low'
	return 'mixed', 'very_low'


def enhanced_cluster_enrichment_analysis(adata, 
                                        cluster_col="leiden_coarse", 
                                        annotation_col="celltype", 
                                        contrast_col="celltype_contrast",
                                        annotation_type="celltype",
                                        min_peaks_per_cluster=100,
                                        # Enrichment thresholds
                                        high_p_threshold=0.001,
                                        high_or_threshold=5.0,
                                        medium_p_threshold=0.05,
                                        medium_or_threshold=2.0,
                                        # Contrast thresholds (set to 0 for constant contrast)
                                        high_contrast_threshold=3.0,
                                        medium_contrast_threshold=2.0,
                                        # Representation thresholds
                                        good_representation_threshold=30.0,
                                        fair_representation_threshold=15.0,
                                        verbose=True):
    """
    Analyze cluster enrichment for any annotation type with contrast weighting.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object
    cluster_col : str
        Column name for cluster assignments  
    annotation_col : str
        Column name for annotations to test (celltype, timepoint, lineage, etc.)
    contrast_col : str
        Column name for contrast scores to weight by
    annotation_type : str
        Type of annotation for display purposes
    min_peaks_per_cluster : int
        Minimum number of peaks required per cluster
    high_p_threshold : float
        P-value threshold for high confidence
    high_or_threshold : float
        Odds ratio threshold for high confidence
    medium_p_threshold : float
        P-value threshold for medium confidence  
    medium_or_threshold : float
        Odds ratio threshold for medium confidence
    high_contrast_threshold : float
        Contrast threshold for high confidence (set to 0 for constant contrast)
    medium_contrast_threshold : float
        Contrast threshold for medium confidence (set to 0 for constant contrast)
    good_representation_threshold : float
        Percentage threshold for good representation in cluster
    fair_representation_threshold : float
        Percentage threshold for fair representation in cluster
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    pd.DataFrame
        Results dataframe with cluster annotations and statistics
    """
    
	clusters = adata.obs[cluster_col].astype(str)
    annotations = adata.obs[annotation_col].astype(str)
    contrasts = adata.obs[contrast_col].values
    
	# Filter clusters with sufficient peaks, and iterate in numeric order when possible
	cluster_counts = clusters.value_counts()
	valid_clusters = cluster_counts[cluster_counts >= min_peaks_per_cluster].index
	valid_clusters = sorted(valid_clusters, key=_cluster_sort_key)
    
    if verbose:
        print(f"Annotating {len(valid_clusters)} clusters with ≥{min_peaks_per_cluster} peaks")
        print(f"Using {annotation_type} annotations from '{annotation_col}' column")
        print(f"Using contrast weighting from '{contrast_col}' column")
        print(f"Thresholds: high_OR≥{high_or_threshold}, medium_OR≥{medium_or_threshold}")
    
    cluster_annotations = []
    
	for cluster in valid_clusters:
        if verbose:
            print(f"\n=== CLUSTER {cluster} ===")
        
        cluster_mask = clusters == cluster
        cluster_size = cluster_mask.sum()
        
		# Get annotation composition of this cluster
		cluster_annotation_counts = annotations[cluster_mask].value_counts()
        
        if verbose:
            print(f"Cluster size: {cluster_size} peaks")
            print(f"Top {annotation_type}s with contrast-enhanced scoring:")
        
		annotation_scores = []
        
		# Analyze each annotation in this cluster
		for annotation, count_in_cluster in cluster_annotation_counts.items():
			# Skip very small groups
			if count_in_cluster < 5:
				continue
			metrics = _compute_annotation_metrics(
				annotation=annotation,
				count_in_cluster=int(count_in_cluster),
				annotations=annotations,
				contrasts=contrasts,
				cluster_mask=cluster_mask,
				cluster_size=int(cluster_size),
				total_size=len(adata.obs),
				annotation_type=annotation_type,
			)
			annotation_scores.append(metrics)
			if verbose:
				print(
					f"  {annotation:25} {int(count_in_cluster):5d} ({metrics['pct_of_cluster']:5.1f}%) "
					f"OR={metrics['odds_ratio']:5.2f} contrast={metrics['mean_contrast']:4.1f} "
					f"combined={metrics['combined_score']:5.2f} p={metrics['p_value']:.2e}"
				)
        
        # Sort by combined score (enrichment × contrast quality)
        annotation_scores.sort(key=lambda x: (-x['combined_score'], x['p_value']))
        
		if annotation_scores:
			# Get the top annotation based on combined score
			top_annotation = annotation_scores[0]
			best_annotation, confidence = _determine_confidence(
				top_annotation,
				high_p_threshold,
				high_or_threshold,
				medium_p_threshold,
				medium_or_threshold,
				high_contrast_threshold,
				medium_contrast_threshold,
				good_representation_threshold,
				fair_representation_threshold,
				annotation_type,
			)
            
            if verbose:
                print(f"\n→ {annotation_type.upper()} ANNOTATION: {best_annotation} ({confidence} confidence)")
                print(f"  Top {annotation_type}: {top_annotation[f'{annotation_type}']}")
                print(f"  Traditional: {top_annotation['fold_enrichment']:.1f}x enriched, OR={top_annotation['odds_ratio']:.2f}")
                print(f"  Contrast: mean={top_annotation['mean_contrast']:.1f}, "
                      f"high-spec peaks={top_annotation['high_contrast_count']}")
                print(f"  Combined score: {top_annotation['combined_score']:.2f} "
                      f"({top_annotation['pct_of_cluster']:.1f}% of cluster)")
            
            cluster_annotations.append({
                'cluster': cluster,
                f'{annotation_type}_annotation': best_annotation,
                f'{annotation_type}_confidence': confidence,
                f'top_{annotation_type}': top_annotation[f'{annotation_type}'],
                'odds_ratio': top_annotation['odds_ratio'],
                'p_value': top_annotation['p_value'],
                'fold_enrichment': top_annotation['fold_enrichment'],
                'mean_contrast': top_annotation['mean_contrast'],
                'median_contrast': top_annotation['median_contrast'],
                'high_contrast_count': top_annotation['high_contrast_count'],
                'moderate_contrast_count': top_annotation['moderate_contrast_count'],
                'broad_contrast_count': top_annotation['broad_contrast_count'],
                'contrast_bonus': top_annotation['contrast_bonus'],
                'combined_score': top_annotation['combined_score'],
                'pct_of_cluster': top_annotation['pct_of_cluster'],
                'cluster_size': cluster_size
            })
        else:
            if verbose:
                print(f"\n→ {annotation_type.upper()} ANNOTATION: unclear (no significant enrichments)")
            
            cluster_annotations.append({
                'cluster': cluster,
                f'{annotation_type}_annotation': 'unclear',
                f'{annotation_type}_confidence': 'none',
                f'top_{annotation_type}': None,
                'odds_ratio': None,
                'p_value': None,
                'fold_enrichment': None,
                'mean_contrast': None,
                'median_contrast': None,
                'high_contrast_count': None,
                'moderate_contrast_count': None,
                'broad_contrast_count': None,
                'contrast_bonus': None,
                'combined_score': None,
                'pct_of_cluster': None,
                'cluster_size': cluster_size
            })
    
    return pd.DataFrame(cluster_annotations)


def create_enhanced_summary(annotation_df, annotation_type="celltype", verbose=True):
    """
    Create enhanced summary showing both enrichment and contrast metrics.
    
    Parameters:
    -----------
    annotation_df : pd.DataFrame
        Results from enhanced_cluster_enrichment_analysis
    annotation_type : str
        Type of annotation for display
    verbose : bool
        Whether to print detailed summary
        
    Returns:
    --------
    pd.DataFrame
        Sorted summary dataframe
    """
    if not verbose:
        return annotation_df.sort_values('cluster')
    
    print("\n" + "="*100)
    print(f"ENHANCED CLUSTER {annotation_type.upper()} ENRICHMENT SUMMARY (with Contrast Weighting)")
    print("="*100)
    
    summary = annotation_df.sort_values('cluster').copy()
    
    for _, row in summary.iterrows():
        confidence_symbol = {
            'high': '***',
            'medium': '**', 
            'low': '*',
            'very_low': '·',
            'none': ''
        }.get(row[f'{annotation_type}_confidence'], '')
        
        annotation_col = f'{annotation_type}_annotation'
        if row[annotation_col] not in ['mixed', 'unclear']:
            print(f"Cluster {row['cluster']:2s}: {row[annotation_col]:25s} {confidence_symbol:3s} "
                  f"(OR={row['odds_ratio']:4.1f}, contrast={row['mean_contrast']:4.1f}, "
                  f"combined={row['combined_score']:4.1f}, "
                  f"{row['pct_of_cluster']:4.1f}% of cluster, n={row['cluster_size']})")
            
            # Show contrast composition
            if pd.notna(row['high_contrast_count']):
                print(f"           {'':25s}     "
                      f"Specificity: {row['high_contrast_count']} high, "
                      f"{row['moderate_contrast_count']} moderate, "
                      f"{row['broad_contrast_count']} broad peaks")
        else:
            print(f"Cluster {row['cluster']:2s}: {row[annotation_col]:25s} {confidence_symbol:3s} "
                  f"(n={row['cluster_size']})")
    
    print(f"\nLegend: *** = high confidence, ** = medium, * = low, · = very low")
    print(f"OR = Odds Ratio, contrast = mean {annotation_type} contrast, combined = OR × contrast bonus")
    
    return summary


def add_annotations_to_adata(adata, annotation_df, annotation_type="celltype", cluster_col="leiden_coarse"):
    """
    Add cluster annotations back to the AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object to modify
    annotation_df : pd.DataFrame
        Results from enhanced_cluster_enrichment_analysis
    annotation_type : str
        Type of annotation (celltype, timepoint, lineage, peak_type)
    cluster_col : str
        Column name for cluster assignments
    """
    # Create mapping from cluster to annotation
    annotation_col = f'{annotation_type}_annotation'
    confidence_col = f'{annotation_type}_confidence'
    
    cluster_to_annotation = dict(zip(annotation_df['cluster'], annotation_df[annotation_col]))
    cluster_to_confidence = dict(zip(annotation_df['cluster'], annotation_df[confidence_col]))
    cluster_to_combined_score = dict(zip(annotation_df['cluster'], annotation_df['combined_score']))
    cluster_to_mean_contrast = dict(zip(annotation_df['cluster'], annotation_df['mean_contrast']))
    
    # Map annotations
    clusters = adata.obs[cluster_col].astype(str)
    adata.obs[f'{cluster_col}_{annotation_type}_annotation'] = clusters.map(cluster_to_annotation).fillna('unknown')
    adata.obs[f'{cluster_col}_{annotation_type}_confidence'] = clusters.map(cluster_to_confidence).fillna('none')
    adata.obs[f'{cluster_col}_{annotation_type}_combined_score'] = clusters.map(cluster_to_combined_score).fillna(0)
    adata.obs[f'{cluster_col}_{annotation_type}_mean_contrast'] = clusters.map(cluster_to_mean_contrast).fillna(0)
    
    print(f"Added enhanced {annotation_type} annotation columns to adata.obs:")
    print(f"  - '{cluster_col}_{annotation_type}_annotation'")
    print(f"  - '{cluster_col}_{annotation_type}_confidence'") 
    print(f"  - '{cluster_col}_{annotation_type}_combined_score'")
    print(f"  - '{cluster_col}_{annotation_type}_mean_contrast'")


def run_all_enrichment_analyses(adata, cluster_col="leiden_coarse", 
                               celltype_col="celltype", timepoint_col="timepoint", 
                               lineage_col="lineage", peak_type_col="peak_type",
                               celltype_contrast_col="celltype_contrast",
                               timepoint_contrast_col="timepoint_contrast",
                               lineage_contrast_col="lineage_contrast",
                               peak_type_contrast_col="constant_contrast",
                               # Universal thresholds
                               high_or_threshold=5.0,
                               medium_or_threshold=2.0,
                               good_representation_threshold=30.0,
                               fair_representation_threshold=15.0,
                               verbose=True):
    """
    Run enrichment analysis for all annotation types with universal thresholds.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object
    cluster_col : str
        Column name for cluster assignments
    celltype_col, timepoint_col, lineage_col, peak_type_col : str
        Column names for different annotation types
    celltype_contrast_col, timepoint_contrast_col, constant_contrast_col : str
        Column names for contrast scores
    high_or_threshold, medium_or_threshold : float
        Universal odds ratio thresholds
    good_representation_threshold, fair_representation_threshold : float
        Universal representation thresholds
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    dict
        Dictionary containing results for all annotation types
    """
    
    results = {}
    
    # 1. CELLTYPE ENRICHMENT
    if verbose:
        print("="*80)
        print("CELLTYPE ENRICHMENT ANALYSIS")
        print("="*80)
    
    results['celltype'] = enhanced_cluster_enrichment_analysis(
        adata, 
        cluster_col=cluster_col,
        annotation_col=celltype_col, 
        contrast_col=celltype_contrast_col,
        annotation_type="celltype",
        high_or_threshold=high_or_threshold,
        medium_or_threshold=medium_or_threshold,
        good_representation_threshold=good_representation_threshold,
        fair_representation_threshold=fair_representation_threshold,
        verbose=verbose
    )
    
    celltype_summary = create_enhanced_summary(results['celltype'], "celltype", verbose)
    add_annotations_to_adata(adata, results['celltype'], "celltype", cluster_col)

    # 2. TIMEPOINT ENRICHMENT  
    if verbose:
        print("\n" + "="*80)
        print("TIMEPOINT ENRICHMENT ANALYSIS")
        print("="*80)
    
    results['timepoint'] = enhanced_cluster_enrichment_analysis(
        adata,
        cluster_col=cluster_col, 
        annotation_col=timepoint_col,
        contrast_col=timepoint_contrast_col,
        annotation_type="timepoint",
        high_or_threshold=high_or_threshold,
        medium_or_threshold=medium_or_threshold,
        good_representation_threshold=good_representation_threshold,
        fair_representation_threshold=fair_representation_threshold,
        verbose=verbose
    )
    
    timepoint_summary = create_enhanced_summary(results['timepoint'], "timepoint", verbose)
    add_annotations_to_adata(adata, results['timepoint'], "timepoint", cluster_col)

    # 3. LINEAGE ENRICHMENT
    if verbose:
        print("\n" + "="*80)
        print("LINEAGE ENRICHMENT ANALYSIS") 
        print("="*80)
    
    results['lineage'] = enhanced_cluster_enrichment_analysis(
        adata,
        cluster_col=cluster_col,
        annotation_col=lineage_col, 
        contrast_col=lineage_contrast_col,  # Using timepoint contrast as proxy
        annotation_type="lineage",
        high_or_threshold=high_or_threshold,
        medium_or_threshold=medium_or_threshold,
        good_representation_threshold=good_representation_threshold,
        fair_representation_threshold=fair_representation_threshold,
        verbose=verbose
    )
    
    lineage_summary = create_enhanced_summary(results['lineage'], "lineage", verbose)
    add_annotations_to_adata(adata, results['lineage'], "lineage", cluster_col)

    # 4. PEAK TYPE ENRICHMENT (with disabled contrast thresholds)
    if verbose:
        print("\n" + "="*80)
        print("PEAK TYPE ENRICHMENT ANALYSIS")
        print("="*80)
    
    results['peak_type'] = enhanced_cluster_enrichment_analysis(
        adata,
        cluster_col=cluster_col,
        annotation_col=peak_type_col,
        contrast_col=peak_type_contrast_col,  # Equal contrast for all peaks
        annotation_type="peak_type",
        high_or_threshold=high_or_threshold,
        medium_or_threshold=medium_or_threshold,
        high_contrast_threshold=0.0,  # Disable contrast thresholds
        medium_contrast_threshold=0.0,
        good_representation_threshold=good_representation_threshold,
        fair_representation_threshold=fair_representation_threshold,
        verbose=verbose
    )
    
    peak_type_summary = create_enhanced_summary(results['peak_type'], "peak_type", verbose)
    add_annotations_to_adata(adata, results['peak_type'], "peak_type", cluster_col)

    # Final summary
    if verbose:
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        for ann_type in ['celltype', 'timepoint', 'lineage', 'peak_type']:
            print(f"\n{ann_type.capitalize()} annotation distribution:")
            col_name = f'{cluster_col}_{ann_type}_annotation'
            if col_name in adata.obs.columns:
                print(adata.obs[col_name].value_counts().head(10))
    
    return results


def create_constant_contrast_column(adata, value=1.0):
    """
    Create a constant contrast column for peak_type analysis.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object to modify
    value : float
        Constant value to assign to all peaks
    """
    adata.obs['constant_contrast'] = value
    print(f"Created 'constant_contrast' column with value {value}")


def print_cluster_validation(adata, cluster_id, cluster_col="leiden_coarse",
                             annotation_types=['celltype', 'timepoint', 'lineage', 'peak_type']):
    """
    Print validation statistics for a specific cluster across all annotation types.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object
    cluster_id : str or int
        Cluster ID to validate
    cluster_col : str
        Column name for cluster assignments
    """
    
    cluster_mask = adata.obs[cluster_col].astype(str) == str(cluster_id)
    cluster_size = cluster_mask.sum()
    
    print(f"\n=== CLUSTER {cluster_id} VALIDATION ===")
    print(f"Cluster size: {cluster_size:,} peaks")
    
    # Check all annotation types
    # annotation_types = ['celltype', 'timepoint', 'lineage', 'peak_type']
    
    for ann_type in annotation_types:
        ann_col = f'{cluster_col}_{ann_type}_annotation'
        conf_col = f'{cluster_col}_{ann_type}_confidence'
        
        if ann_col in adata.obs.columns:
            annotation = adata.obs.loc[cluster_mask, ann_col].iloc[0]
            confidence = adata.obs.loc[cluster_mask, conf_col].iloc[0]
            
            # Get composition
            original_col = {'celltype': 'celltype', 'timepoint': 'timepoint', 
                           'lineage': 'lineage', 'peak_type': 'peak_type'}[ann_type]
            
            if original_col in adata.obs.columns:
                composition = adata.obs.loc[cluster_mask, original_col].value_counts()
                top_3 = composition.head(3)
                
                print(f"\n{ann_type.capitalize():12}: {annotation} ({confidence})")
                print(f"  Composition: ", end="")
                for i, (item, count) in enumerate(top_3.items()):
                    pct = count / cluster_size * 100
                    print(f"{item}={count}({pct:.1f}%)", end="")
                    if i < len(top_3) - 1:
                        print(", ", end="")
                print()


# Example usage and testing functions
def test_cluster_23_calibration(adata, cluster_col="leiden_coarse"):
    """
    Test that Cluster 23 gets the expected annotations with current thresholds.
    """
    print("="*60)
    print("CLUSTER 23 CALIBRATION TEST")
    print("="*60)
    
    print_cluster_validation(adata, "23", cluster_col)
    
    expected_results = {
        'celltype': 'hemangioblasts (high confidence)',
        'timepoint': 'mixed (very_low confidence)', 
        'peak_type': 'promoter (high confidence)'
    }
    
    print(f"\nExpected results:")
    for ann_type, expected in expected_results.items():
        print(f"  {ann_type:12}: {expected}")