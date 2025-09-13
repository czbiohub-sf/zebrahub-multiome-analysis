"""
GRN Overlap Quantification Module

This module provides functions to analyze the overlap and presence fractions 
of TF-gene pairs across Gene Regulatory Networks (GRNs) in different 
timepoints and cell types.

Author: Generated for GRN similarity analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def extract_tf_gene_pairs(grn_df):
    """
    Extract TF-gene pairs from a GRN DataFrame in long format
    
    Parameters:
    -----------
    grn_df : pd.DataFrame or None
        GRN DataFrame with columns: source, target, coef_mean, coef_abs, p, -logp
        
    Returns:
    --------
    set : Set of TF-gene pairs in format "source_target"
    """
    if grn_df is None or grn_df.empty:
        return set()
    
    # Create pairs from source (TF) and target columns
    pairs = set(f"{row['source']}_{row['target']}" for _, row in grn_df.iterrows())
    return pairs

def compute_timepoint_presence_fractions(dict_filtered_grns, exclude_groups=None):
    """
    Compute presence fractions of TF-gene pairs across celltypes within each timepoint
    
    Parameters:
    -----------
    dict_filtered_grns : dict
        Dictionary with structure {timepoint: {celltype: grn_dataframe}}
        where grn_dataframe has columns: source, target, coef_mean, coef_abs, p, -logp
    exclude_groups : list, optional
        List of strings in format "{timepoint}_{celltype}" to exclude from analysis
        (e.g., groups with too few cells)
        
    Returns:
    --------
    dict : Dictionary containing presence fraction analysis for each timepoint
    """
    timepoints = list(dict_filtered_grns.keys())
    celltypes = list(dict_filtered_grns[timepoints[0]].keys())
    
    timepoint_presence = {}
    
    for tp in timepoints:
        print(f"Analyzing timepoint: {tp}")
        
        # Get all valid GRNs for this timepoint
        tp_grns = []
        valid_celltypes = []
        
        for ct in celltypes:
            # Skip if this timepoint-celltype combination should be excluded
            if exclude_groups and f"{tp}_{ct}" in exclude_groups:
                continue
                
            grn = dict_filtered_grns[tp].get(ct)
            if grn is not None and not grn.empty:
                tp_grns.append(grn)
                valid_celltypes.append(ct)
        
        print(f"  Valid celltypes: {len(valid_celltypes)}/{len(celltypes)}")
        
        if len(tp_grns) < 2:
            print(f"  Skipping {tp} - insufficient data")
            continue
        
        # Count pair occurrences
        pair_counts = Counter()
        total_grns = len(tp_grns)
        
        for grn in tp_grns:
            pairs = extract_tf_gene_pairs(grn)
            for pair in pairs:
                pair_counts[pair] += 1
        
        # Calculate presence fractions
        presence_fractions = {pair: count/total_grns for pair, count in pair_counts.items()}
        timepoint_presence[tp] = {
            'presence_fractions': presence_fractions,
            'total_pairs': len(pair_counts),
            'total_grns': total_grns,
            'valid_celltypes': valid_celltypes
        }
        
        print(f"  Total unique pairs: {len(pair_counts)}")
        print(f"  Mean presence fraction: {np.mean(list(presence_fractions.values())):.3f}")
    
    return timepoint_presence

def compute_celltype_presence_fractions(dict_filtered_grns, exclude_groups=None):
    """
    Compute presence fractions of TF-gene pairs across timepoints within each celltype
    
    Parameters:
    -----------
    dict_filtered_grns : dict
        Dictionary with structure {timepoint: {celltype: grn_dataframe}}
        where grn_dataframe has columns: source, target, coef_mean, coef_abs, p, -logp
    exclude_groups : list, optional
        List of strings in format "{timepoint}_{celltype}" to exclude from analysis
        (e.g., groups with too few cells)
        
    Returns:
    --------
    dict : Dictionary containing presence fraction analysis for each celltype
    """
    timepoints = list(dict_filtered_grns.keys())
    celltypes = list(dict_filtered_grns[timepoints[0]].keys())
    
    celltype_presence = {}
    
    for ct in celltypes:
        print(f"Analyzing celltype: {ct}")
        
        # Get all valid GRNs for this celltype
        ct_grns = []
        valid_timepoints = []
        
        for tp in timepoints:
            # Skip if this timepoint-celltype combination should be excluded
            if exclude_groups and f"{tp}_{ct}" in exclude_groups:
                continue
                
            grn = dict_filtered_grns[tp].get(ct)
            if grn is not None and not grn.empty:
                ct_grns.append(grn)
                valid_timepoints.append(tp)
        
        print(f"  Valid timepoints: {len(valid_timepoints)}/{len(timepoints)}")
        
        if len(ct_grns) < 2:
            print(f"  Skipping {ct} - insufficient data")
            continue
        
        # Count pair occurrences
        pair_counts = Counter()
        total_grns = len(ct_grns)
        
        for grn in ct_grns:
            pairs = extract_tf_gene_pairs(grn)
            for pair in pairs:
                pair_counts[pair] += 1
        
        # Calculate presence fractions
        presence_fractions = {pair: count/total_grns for pair, count in pair_counts.items()}
        celltype_presence[ct] = {
            'presence_fractions': presence_fractions,
            'total_pairs': len(pair_counts),
            'total_grns': total_grns,
            'valid_timepoints': valid_timepoints
        }
        
        print(f"  Total unique pairs: {len(pair_counts)}")
        print(f"  Mean presence fraction: {np.mean(list(presence_fractions.values())):.3f}")
    
    return celltype_presence

def create_summary_statistics(presence_data, analysis_type='timepoint'):
    """
    Create summary statistics from presence fraction data
    
    Parameters:
    -----------
    presence_data : dict
        Output from compute_timepoint_presence_fractions or compute_celltype_presence_fractions
    analysis_type : str
        Either 'timepoint' or 'celltype'
        
    Returns:
    --------
    pd.DataFrame : Summary statistics
    """
    summary = []
    
    for key, data in presence_data.items():
        fractions = list(data['presence_fractions'].values())
        if fractions:
            n_fractions = len(fractions)
            mean_presence = np.mean(fractions)
            std_presence = np.std(fractions)
            sem_presence = std_presence / np.sqrt(n_fractions) if n_fractions > 0 else 0
            
            summary.append({
                analysis_type: key,
                'mean_presence': mean_presence,
                'median_presence': np.median(fractions),
                'std_presence': std_presence,
                'sem_presence': sem_presence,  # Add SEM
                'total_pairs': len(fractions),
                'pairs_50pct': sum(1 for f in fractions if f >= 0.5),
                'pairs_80pct': sum(1 for f in fractions if f >= 0.8),
                'pairs_90pct': sum(1 for f in fractions if f >= 0.9)
            })
    
    return pd.DataFrame(summary)

def analyze_grn_overlap(dict_filtered_grns, exclude_groups=None, verbose=True):
    """
    Complete analysis of GRN overlap including both timepoint and celltype analyses
    
    Parameters:
    -----------
    dict_filtered_grns : dict
        Dictionary with structure {timepoint: {celltype: grn_dataframe}}
        where grn_dataframe has columns: source, target, coef_mean, coef_abs, p, -logp
    exclude_groups : list, optional
        List of strings in format "{timepoint}_{celltype}" to exclude from analysis
        (e.g., groups with too few cells)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict : Dictionary containing all analysis results
    """
    if verbose:
        timepoints = list(dict_filtered_grns.keys())
        celltypes = list(dict_filtered_grns[timepoints[0]].keys())
        print(f"Found {len(timepoints)} timepoints: {timepoints}")
        print(f"Found {len(celltypes)} celltypes")
        if exclude_groups:
            print(f"Excluding {len(exclude_groups)} timepoint-celltype combinations: {exclude_groups[:5]}{'...' if len(exclude_groups) > 5 else ''}")
        print("\n" + "="*60)
        print("COMPUTING PRESENCE FRACTIONS...")
        print("="*60)
    
    # Compute presence fractions
    timepoint_presence = compute_timepoint_presence_fractions(dict_filtered_grns, exclude_groups)
    celltype_presence = compute_celltype_presence_fractions(dict_filtered_grns, exclude_groups)
    
    # Create summary statistics
    tp_summary_df = create_summary_statistics(timepoint_presence, 'timepoint')
    ct_summary_df = create_summary_statistics(celltype_presence, 'celltype')
    
    # Collect all fractions for overall analysis
    all_tp_fractions = []
    for data in timepoint_presence.values():
        all_tp_fractions.extend(list(data['presence_fractions'].values()))
    
    all_ct_fractions = []
    for data in celltype_presence.values():
        all_ct_fractions.extend(list(data['presence_fractions'].values()))
    
    if verbose:
        print("\n" + "="*60)
        print("ANALYZING PRESENCE FRACTION DISTRIBUTIONS...")
        print("="*60)
        
        print("\nTIMEPOINT ANALYSIS SUMMARY:")
        print(tp_summary_df.round(3))
        
        print("\nCELLTYPE ANALYSIS SUMMARY:")
        print(ct_summary_df.round(3))
    
    return {
        'timepoint_presence': timepoint_presence,
        'celltype_presence': celltype_presence,
        'tp_summary_df': tp_summary_df,
        'ct_summary_df': ct_summary_df,
        'all_tp_fractions': all_tp_fractions,
        'all_ct_fractions': all_ct_fractions
    }

def recommend_thresholds(all_tp_fractions, all_ct_fractions, verbose=True):
    """
    Recommend thresholds based on data distribution
    
    Parameters:
    -----------
    all_tp_fractions : list
        List of all presence fractions from timepoint analysis
    all_ct_fractions : list
        List of all presence fractions from celltype analysis
    verbose : bool
        Whether to print recommendations
        
    Returns:
    --------
    dict : Dictionary with recommended thresholds and statistics
    """
    # Calculate overall statistics
    overall_tp_mean = np.mean(all_tp_fractions)
    overall_ct_mean = np.mean(all_ct_fractions)
    
    def _recommend_threshold(fractions, analysis_type):
        """Helper function to recommend threshold for a given analysis"""
        retention_data = []
        
        for pct in [50, 60, 70, 80, 90]:
            threshold = pct / 100
            retained = sum(1 for f in fractions if f >= threshold)
            retention_pct = retained / len(fractions) * 100 if fractions else 0
            retention_data.append({
                'threshold': threshold,
                'retention_pct': retention_pct,
                'n_pairs': retained
            })
        
        # Recommend based on retention
        retention_80 = retention_data[3]['retention_pct']  # 80% threshold
        retention_70 = retention_data[2]['retention_pct']  # 70% threshold
        
        if retention_80 >= 20:
            return 0.8, "Conservative (recommended)", retention_data
        elif retention_70 >= 30:
            return 0.7, "Moderate (recommended)", retention_data
        else:
            return 0.6, "Liberal (recommended)", retention_data
    
    tp_rec_threshold, tp_rec_reason, tp_retention = _recommend_threshold(all_tp_fractions, "Timepoint")
    ct_rec_threshold, ct_rec_reason, ct_retention = _recommend_threshold(all_ct_fractions, "Celltype")
    
    if verbose:
        print("\n" + "="*60)
        print("RECOMMENDATIONS BASED ON YOUR DATA:")
        print("="*60)
        
        print(f"\nOverall Statistics:")
        print(f"  Timepoint analysis - Mean presence fraction: {overall_tp_mean:.3f}")
        print(f"  Celltype analysis - Mean presence fraction: {overall_ct_mean:.3f}")
        
        print(f"\nTimepoint Analysis - Data retention at different thresholds:")
        for data in tp_retention:
            print(f"  {data['threshold']:.1f} threshold: {data['retention_pct']:.1f}% of pairs retained ({data['n_pairs']:,} pairs)")
        
        print(f"\nCelltype Analysis - Data retention at different thresholds:")
        for data in ct_retention:
            print(f"  {data['threshold']:.1f} threshold: {data['retention_pct']:.1f}% of pairs retained ({data['n_pairs']:,} pairs)")
        
        print(f"\nRECOMMENDED THRESHOLDS:")
        print(f"  Timepoint analysis: {tp_rec_threshold} - {tp_rec_reason}")
        print(f"  Celltype analysis: {ct_rec_threshold} - {ct_rec_reason}")
        
        # Check for potential issues
        print(f"\nPOTENTIAL ISSUES TO WATCH:")
        if overall_tp_mean < 0.3:
            print("  ⚠️  Low overall presence fractions in timepoint analysis - data may be very sparse")
        if overall_ct_mean < 0.3:
            print("  ⚠️  Low overall presence fractions in celltype analysis - data may be very sparse")
        if len(all_tp_fractions) < 100:
            print("  ⚠️  Very few TF-gene pairs in timepoint analysis - limited statistical power")
        if len(all_ct_fractions) < 100:
            print("  ⚠️  Very few TF-gene pairs in celltype analysis - limited statistical power")
        
        print(f"\nSUGGESTED NEXT STEPS:")
        print(f"1. Use the recommended thresholds above for your main analysis")
        print(f"2. Run sensitivity analysis with ±0.1 threshold variation")
        print(f"3. Report the number of pairs retained at your chosen threshold")
        print(f"4. Consider biological relevance of excluded pairs (are they important regulators?)")
    
    return {
        'overall_tp_mean': overall_tp_mean,
        'overall_ct_mean': overall_ct_mean,
        'tp_rec_threshold': tp_rec_threshold,
        'tp_rec_reason': tp_rec_reason,
        'ct_rec_threshold': ct_rec_threshold,
        'ct_rec_reason': ct_rec_reason,
        'tp_retention_data': tp_retention,
        'ct_retention_data': ct_retention
    }

def plot_grn_overlap_analysis(analysis_results, save_path=None, show_plot=True):
    """
    Create comprehensive visualization of GRN overlap analysis
    
    Parameters:
    -----------
    analysis_results : dict
        Output from analyze_grn_overlap()
    save_path : str, optional
        Path to save the HTML plot
    show_plot : bool
        Whether to display the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure object
    """
    tp_summary_df = analysis_results['tp_summary_df']
    ct_summary_df = analysis_results['ct_summary_df']
    all_tp_fractions = analysis_results['all_tp_fractions']
    all_ct_fractions = analysis_results['all_ct_fractions']
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Presence Fraction Distribution (Timepoint Analysis)',
            'Presence Fraction Distribution (Celltype Analysis)',
            'Data Retention at Different Thresholds (Timepoints)',
            'Data Retention at Different Thresholds (Celltypes)',
            'Mean Presence Fraction by Timepoint',
            'Mean Presence Fraction by Celltype'
        ),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Plot 1: Histogram of presence fractions for timepoint analysis
    fig.add_trace(
        go.Histogram(
            x=all_tp_fractions,
            nbinsx=20,
            name='Timepoint Analysis',
            opacity=0.7,
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    # Plot 2: Histogram of presence fractions for celltype analysis
    fig.add_trace(
        go.Histogram(
            x=all_ct_fractions,
            nbinsx=20,
            name='Celltype Analysis',
            opacity=0.7,
            marker_color='red'
        ),
        row=1, col=2
    )
    
    # Plot 3: Data retention at different thresholds (timepoints)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    tp_retention = []
    for threshold in thresholds:
        retained = sum(1 for f in all_tp_fractions if f >= threshold)
        tp_retention.append(retained / len(all_tp_fractions) * 100 if all_tp_fractions else 0)
    
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=tp_retention,
            mode='lines+markers',
            name='Timepoint Retention',
            marker=dict(size=10, color='blue'),
            line=dict(width=3)
        ),
        row=2, col=1
    )
    
    # Plot 4: Data retention at different thresholds (celltypes)
    ct_retention = []
    for threshold in thresholds:
        retained = sum(1 for f in all_ct_fractions if f >= threshold)
        ct_retention.append(retained / len(all_ct_fractions) * 100 if all_ct_fractions else 0)
    
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=ct_retention,
            mode='lines+markers',
            name='Celltype Retention',
            marker=dict(size=10, color='red'),
            line=dict(width=3)
        ),
        row=2, col=2
    )
    
    # Plot 5: Mean presence fraction by timepoint
    if not tp_summary_df.empty:
        fig.add_trace(
            go.Bar(
                x=tp_summary_df['timepoint'],
                y=tp_summary_df['mean_presence'],
                name='Timepoint Means',
                marker_color='lightblue',
                error_y=dict(type='data', array=tp_summary_df['sem_presence'])  # Use SEM instead of std
            ),
            row=3, col=1
        )
    
    # Plot 6: Mean presence fraction by celltype
    if not ct_summary_df.empty:
        fig.add_trace(
            go.Bar(
                x=list(range(len(ct_summary_df))),
                y=ct_summary_df['mean_presence'],
                name='Celltype Means',
                marker_color='lightcoral',
                error_y=dict(type='data', array=ct_summary_df['sem_presence']),  # Use SEM instead of std
                text=ct_summary_df['celltype'],
                hovertemplate='%{text}<br>Mean: %{y:.3f} ± %{error_y.array:.4f}<extra></extra>'
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Presence Fraction", row=1, col=1)
    fig.update_xaxes(title_text="Presence Fraction", row=1, col=2)
    fig.update_xaxes(title_text="Min Presence Threshold", row=2, col=1)
    fig.update_xaxes(title_text="Min Presence Threshold", row=2, col=2)
    fig.update_xaxes(title_text="Timepoints", row=3, col=1, tickangle=45)
    fig.update_xaxes(title_text="Celltypes", row=3, col=2, tickangle=45)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="% Pairs Retained", row=2, col=1)
    fig.update_yaxes(title_text="% Pairs Retained", row=2, col=2)
    fig.update_yaxes(title_text="Mean Presence Fraction", row=3, col=1)
    fig.update_yaxes(title_text="Mean Presence Fraction", row=3, col=2)
    
    fig.update_layout(
        height=1200,
        width=1400,
        title_text="TF-Gene Pair Presence Fraction Analysis<br><sub>Error bars show Standard Error of the Mean (SEM)</sub>",
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
    
    if show_plot:
        fig.show()
    
    return fig

def save_results_to_csv(analysis_results, prefix="grn_overlap"):
    """
    Save analysis results to CSV files
    
    Parameters:
    -----------
    analysis_results : dict
        Output from analyze_grn_overlap()
    prefix : str
        Prefix for output file names
        
    Returns:
    --------
    dict : Dictionary with saved file paths
    """
    tp_summary_df = analysis_results['tp_summary_df']
    ct_summary_df = analysis_results['ct_summary_df']
    
    # Save summary tables
    tp_file = f'{prefix}_timepoint_summary.csv'
    ct_file = f'{prefix}_celltype_summary.csv'
    
    tp_summary_df.to_csv(tp_file, index=False)
    ct_summary_df.to_csv(ct_file, index=False)
    
    print(f"Summary tables saved:")
    print(f"  Timepoint analysis: {tp_file}")
    print(f"  Celltype analysis: {ct_file}")
    
    return {
        'timepoint_file': tp_file,
        'celltype_file': ct_file
    }

# Convenience function for complete analysis workflow
def complete_grn_overlap_analysis(dict_filtered_grns, exclude_groups=None, save_prefix="grn_overlap", 
                                 save_plot_path=None, show_plot=True, save_csv=True, verbose=True):
    """
    Complete workflow for GRN overlap analysis
    
    Parameters:
    -----------
    dict_filtered_grns : dict
        Dictionary with structure {timepoint: {celltype: grn_dataframe}}
        where grn_dataframe has columns: source, target, coef_mean, coef_abs, p, -logp
    exclude_groups : list, optional
        List of strings in format "{timepoint}_{celltype}" to exclude from analysis
        (e.g., groups with too few cells)
    save_prefix : str
        Prefix for saved files
    save_plot_path : str, optional
        Path to save HTML plot (if None, uses save_prefix)
    show_plot : bool
        Whether to display the plot
    save_csv : bool
        Whether to save results to CSV
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    dict : Dictionary containing all analysis results and recommendations
    """
    # Run main analysis
    analysis_results = analyze_grn_overlap(dict_filtered_grns, exclude_groups, verbose=verbose)
    
    # Get threshold recommendations
    recommendations = recommend_thresholds(
        analysis_results['all_tp_fractions'], 
        analysis_results['all_ct_fractions'], 
        verbose=verbose
    )
    
    # Create plot
    if save_plot_path is None:
        save_plot_path = f"{save_prefix}_analysis.html"
    
    fig = plot_grn_overlap_analysis(analysis_results, save_plot_path, show_plot)
    
    # Save CSV files
    if save_csv:
        file_paths = save_results_to_csv(analysis_results, save_prefix)
    else:
        file_paths = None
    
    # Combine all results
    complete_results = {
        **analysis_results,
        'recommendations': recommendations,
        'figure': fig,
        'saved_files': file_paths
    }
    
    return complete_results