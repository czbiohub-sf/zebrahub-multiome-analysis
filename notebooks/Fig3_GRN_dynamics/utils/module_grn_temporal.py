import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score
from typing import Dict, List, Tuple, Optional, Union, Set


def analyze_single_timepoint(grn_dict, timepoint, predicted_pairs, cluster_id):
    """
    Analyze how the mesh manifests across celltypes at a single timepoint
    
    Parameters:
    -----------
    grn_dict : dict
        Dictionary mapping (celltype, timepoint) -> GRN DataFrame
    timepoint : str
        Timepoint to analyze
    predicted_pairs : list
        List of (TF, target) pairs predicted by peak cluster
    cluster_id : str
        Identifier for the peak cluster
        
    Returns:
    --------
    dict
        Dictionary mapping celltype -> analysis results
    """
    print(f"\n=== ANALYZING TIMEPOINT: {timepoint} ===")
    
    # Get all celltypes at this timepoint
    celltypes_at_tp = [ct for (ct, tp) in grn_dict.keys() if tp == timepoint]
    print(f"Available celltypes: {celltypes_at_tp}")
    
    # Extract subGRNs for each celltype
    celltype_subgrns = {}
    for celltype in celltypes_at_tp:
        if (celltype, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype, timepoint)]
            
            # Find which predicted pairs exist in this GRN
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            celltype_subgrns[celltype] = {
                'subgrn': subgrn,
                'n_edges': len(subgrn),
                'implementation_rate': len(subgrn) / len(predicted_pairs),
                'mean_strength': subgrn['coef_abs'].mean() if len(subgrn) > 0 else 0,
                'implemented_pairs': found_pairs
            }
            
            print(f"{celltype}: {len(subgrn)}/{len(predicted_pairs)} edges ({celltype_subgrns[celltype]['implementation_rate']:.2%})")
    
    return celltype_subgrns


def compare_celltypes_similarity(celltype_subgrns, predicted_pairs, timepoint, cluster_id=""):
    """
    Compare how similar celltypes are in implementing the regulatory program
    
    Parameters:
    -----------
    celltype_subgrns : dict
        Results from analyze_single_timepoint
    predicted_pairs : list
        List of (TF, target) pairs predicted by peak cluster
    timepoint : str
        Timepoint being analyzed
    cluster_id : str
        Identifier for the peak cluster
        
    Returns:
    --------
    tuple
        (similarity_matrix, similarities) - Similarity matrix and ranked pairs
    """
    print(f"\n--- Celltype Similarity Analysis at {timepoint} ---")
    
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
                similarity_matrix[i,j] = 1.0
            else:
                # Jaccard similarity
                similarity_matrix[i,j] = jaccard_score(binary_matrix[i], binary_matrix[j])
    
    # Plot similarity heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, 
                xticklabels=celltypes, 
                yticklabels=celltypes,
                annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Celltype Similarity - Cluster {cluster_id} at {timepoint}')
    plt.tight_layout()
    plt.show()
    
    # Find most and least similar pairs
    similarities = []
    for i in range(n_celltypes):
        for j in range(i+1, n_celltypes):
            similarities.append({
                'celltype1': celltypes[i],
                'celltype2': celltypes[j], 
                'similarity': similarity_matrix[i,j]
            })
    
    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    print("Most similar celltype pairs:")
    for sim in similarities[:3]:
        print(f"  {sim['celltype1']} vs {sim['celltype2']}: {sim['similarity']:.3f}")
    
    print("Least similar celltype pairs:")
    for sim in similarities[-3:]:
        print(f"  {sim['celltype1']} vs {sim['celltype2']}: {sim['similarity']:.3f}")
    
    return similarity_matrix, similarities


def compare_across_timepoints(grn_dict, predicted_pairs, cluster_id):
    """
    Compare how the regulatory program changes across timepoints
    
    Parameters:
    -----------
    grn_dict : dict
        Dictionary mapping (celltype, timepoint) -> GRN DataFrame
    predicted_pairs : list
        List of (TF, target) pairs predicted by peak cluster
    cluster_id : str
        Identifier for the peak cluster
        
    Returns:
    --------
    dict
        Dictionary mapping timepoint -> celltype analysis results
    """
    print(f"\n=== MULTI-TIMEPOINT ANALYSIS ===")
    
    # Get all available timepoints
    all_timepoints = sorted(set([tp for (ct, tp) in grn_dict.keys()]))
    print(f"Available timepoints: {all_timepoints}")
    
    # Store results for each timepoint
    timepoint_results = {}
    
    for timepoint in all_timepoints:
        print(f"\nProcessing timepoint {timepoint}...")
        celltype_subgrns = analyze_single_timepoint(grn_dict, timepoint, predicted_pairs, cluster_id)
        timepoint_results[timepoint] = celltype_subgrns
    
    return timepoint_results


def track_celltype_across_time(timepoint_results, cluster_id):
    """
    Track how specific celltypes implement the program over time
    
    Parameters:
    -----------
    timepoint_results : dict
        Results from compare_across_timepoints
    cluster_id : str
        Identifier for the peak cluster
        
    Returns:
    --------
    dict
        Dictionary mapping celltype -> temporal tracking data
    """
    print(f"\n--- Temporal Tracking ---")
    
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Implementation rate over time
    for celltype, tracking in temporal_tracking.items():
        if len(tracking) > 1:  # Only plot celltypes with multiple timepoints
            timepoints = [t['timepoint'] for t in tracking]
            impl_rates = [t['implementation_rate'] for t in tracking]
            ax1.plot(timepoints, impl_rates, marker='o', label=celltype)
    
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('Implementation Rate')
    ax1.set_title(f'Implementation Rate Over Time - Cluster {cluster_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mean strength over time
    for celltype, tracking in temporal_tracking.items():
        if len(tracking) > 1:
            timepoints = [t['timepoint'] for t in tracking]
            strengths = [t['mean_strength'] for t in tracking]
            ax2.plot(timepoints, strengths, marker='s', label=celltype)
    
    ax2.set_xlabel('Timepoint') 
    ax2.set_ylabel('Mean Edge Strength')
    ax2.set_title(f'Mean Strength Over Time - Cluster {cluster_id}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return temporal_tracking


def summarize_analysis(timepoint_results, temporal_tracking, cluster_id):
    """
    Provide summary statistics of the analysis
    
    Parameters:
    -----------
    timepoint_results : dict
        Results from compare_across_timepoints
    temporal_tracking : dict
        Results from track_celltype_across_time
    cluster_id : str
        Identifier for the peak cluster
        
    Returns:
    --------
    dict
        Summary statistics and top implementers
    """
    print(f"\n=== SUMMARY FOR CLUSTER {cluster_id} ===")
    
    # Overall implementation statistics
    all_impl_rates = []
    all_strengths = []
    for tp_results in timepoint_results.values():
        for ct_result in tp_results.values():
            all_impl_rates.append(ct_result['implementation_rate'])
            all_strengths.append(ct_result['mean_strength'])
    
    print(f"Implementation rate: {np.mean(all_impl_rates):.2%} ± {np.std(all_impl_rates):.2%}")
    print(f"Mean edge strength: {np.mean(all_strengths):.4f} ± {np.std(all_strengths):.4f}")
    
    # Best implementers
    best_implementers = []
    for timepoint, tp_results in timepoint_results.items():
        for celltype, result in tp_results.items():
            best_implementers.append({
                'celltype': celltype,
                'timepoint': timepoint,
                'implementation_rate': result['implementation_rate'],
                'mean_strength': result['mean_strength']
            })
    
    best_implementers = sorted(best_implementers, key=lambda x: x['implementation_rate'], reverse=True)
    
    print("\nTop 5 implementers:")
    for impl in best_implementers[:5]:
        print(f"  {impl['celltype']} at {impl['timepoint']}: {impl['implementation_rate']:.2%} (strength: {impl['mean_strength']:.4f})")
    
    # Temporal trends
    print(f"\nCelltypes tracked across time: {len([ct for ct, track in temporal_tracking.items() if len(track) > 1])}")
    
    return {
        'overall_stats': {
            'mean_implementation_rate': np.mean(all_impl_rates),
            'std_implementation_rate': np.std(all_impl_rates),
            'mean_strength': np.mean(all_strengths),
            'std_strength': np.std(all_strengths)
        },
        'best_implementers': best_implementers,
        'temporal_celltypes': len([ct for ct, track in temporal_tracking.items() if len(track) > 1])
    }


def run_complete_temporal_analysis(grn_dict, predicted_pairs, cluster_id):
    """
    Run the complete temporal analysis workflow
    
    Parameters:
    -----------
    grn_dict : dict
        Dictionary mapping (celltype, timepoint) -> GRN DataFrame
    predicted_pairs : list
        List of (TF, target) pairs predicted by peak cluster
    cluster_id : str
        Identifier for the peak cluster
        
    Returns:
    --------
    dict
        Complete analysis results
    """
    print(f"\n🔬 RUNNING COMPLETE TEMPORAL ANALYSIS FOR CLUSTER {cluster_id}")
    print("=" * 60)
    
    # Step 1: Multi-timepoint comparison
    timepoint_results = compare_across_timepoints(grn_dict, predicted_pairs, cluster_id)
    
    # Step 2: Temporal tracking
    temporal_tracking = track_celltype_across_time(timepoint_results, cluster_id)
    
    # Step 3: Summary
    summary = summarize_analysis(timepoint_results, temporal_tracking, cluster_id)
    
    return {
        'timepoint_results': timepoint_results,
        'temporal_tracking': temporal_tracking, 
        'summary': summary,
        'cluster_id': cluster_id
    }


def analyze_temporal_patterns(temporal_tracking, min_timepoints=2):
    """
    Analyze temporal patterns in celltype behavior
    
    Parameters:
    -----------
    temporal_tracking : dict
        Results from track_celltype_across_time
    min_timepoints : int
        Minimum number of timepoints required for analysis
        
    Returns:
    --------
    dict
        Analysis of temporal patterns
    """
    patterns = {
        'increasing': [],
        'decreasing': [],
        'stable': [],
        'fluctuating': []
    }
    
    for celltype, tracking in temporal_tracking.items():
        if len(tracking) >= min_timepoints:
            impl_rates = [t['implementation_rate'] for t in tracking]
            
            # Calculate trend
            x = np.arange(len(impl_rates))
            slope = np.polyfit(x, impl_rates, 1)[0]
            
            # Calculate variance for stability measure
            variance = np.var(impl_rates)
            
            if variance < 0.01:  # Low variance = stable
                patterns['stable'].append(celltype)
            elif slope > 0.05:  # Positive slope = increasing
                patterns['increasing'].append(celltype)
            elif slope < -0.05:  # Negative slope = decreasing
                patterns['decreasing'].append(celltype)
            else:  # High variance but no clear trend = fluctuating
                patterns['fluctuating'].append(celltype)
    
    print("\n=== TEMPORAL PATTERNS ===")
    for pattern, celltypes in patterns.items():
        print(f"{pattern.title()}: {len(celltypes)} celltypes")
        for ct in celltypes[:5]:  # Show first 5
            print(f"  - {ct}")
        if len(celltypes) > 5:
            print(f"  ... and {len(celltypes) - 5} more")
    
    return patterns


def compare_clusters_temporal_similarity(analysis_results_dict):
    """
    Compare temporal patterns between different clusters
    
    Parameters:
    -----------
    analysis_results_dict : dict
        Dictionary mapping cluster_id -> complete analysis results
        
    Returns:
    --------
    dict
        Cross-cluster comparison results
    """
    print("\n=== CROSS-CLUSTER TEMPORAL COMPARISON ===")
    
    cluster_ids = list(analysis_results_dict.keys())
    n_clusters = len(cluster_ids)
    
    # Compare implementation patterns
    for i, cluster1 in enumerate(cluster_ids):
        for j, cluster2 in enumerate(cluster_ids):
            if i < j:
                summary1 = analysis_results_dict[cluster1]['summary']
                summary2 = analysis_results_dict[cluster2]['summary']
                
                # Compare mean implementation rates
                rate_diff = abs(summary1['overall_stats']['mean_implementation_rate'] - 
                               summary2['overall_stats']['mean_implementation_rate'])
                
                print(f"{cluster1} vs {cluster2}: Implementation rate difference = {rate_diff:.2%}")
    
    return cluster_ids


def plot_cluster_comparison(analysis_results_dict, metric='implementation_rate'):
    """
    Plot comparison of temporal patterns across clusters
    
    Parameters:
    -----------
    analysis_results_dict : dict
        Dictionary mapping cluster_id -> complete analysis results
    metric : str
        Metric to compare ('implementation_rate' or 'mean_strength')
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created comparison plot
    """
    fig, axes = plt.subplots(1, len(analysis_results_dict), figsize=(15, 5))
    if len(analysis_results_dict) == 1:
        axes = [axes]
    
    for i, (cluster_id, results) in enumerate(analysis_results_dict.items()):
        ax = axes[i]
        temporal_tracking = results['temporal_tracking']
        
        for celltype, tracking in temporal_tracking.items():
            if len(tracking) > 1:
                timepoints = [t['timepoint'] for t in tracking]
                values = [t[metric] for t in tracking]
                ax.plot(timepoints, values, marker='o', label=celltype, alpha=0.7)
        
        ax.set_xlabel('Timepoint')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Cluster {cluster_id}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig 