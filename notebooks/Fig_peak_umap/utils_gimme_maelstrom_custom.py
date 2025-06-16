
#!/usr/bin/env python3
"""
Custom script to run GimmeMotifs rank aggregation on pre-computed motif scores
Bypasses the expensive motif scanning step for single-cell ATAC-seq analysis
"""

import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
import logging
from scipy.stats import pearsonr

# Import the rank aggregation function from GimmeMotifs
# You'll need: pip install gimmemotifs
from gimmemotifs.maelstrom.rank import rankagg
from gimmemotifs.maelstrom.moap import Moap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_custom_maelstrom(
    peaks_motifs_matrix,      # Your pre-computed peaks × motifs score matrix
    peak_clusters,            # Peak cluster assignments (leiden clusters)
    methods=['hypergeom', 'mwu', 'rf'],  # Activity methods to use
    aggregation='int_stouffer',  # Rank aggregation method
    ncpus=16,
    random_state=42
):
    """
    Run maelstrom rank aggregation on pre-computed motif scores
    
    Parameters
    ----------
    peaks_motifs_matrix : pd.DataFrame
        Pre-computed peaks × motifs matrix (your existing data)
        Index: peak IDs, Columns: motif names
        
    peak_clusters : pd.DataFrame or pd.Series
        Peak cluster assignments. If DataFrame, should have 1 column with cluster labels
        Index must match peaks_motifs_matrix index
        
    methods : list
        Activity prediction methods to use
        
    aggregation : str
        'int_stouffer' or 'stuart'
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'final_scores': aggregated motif activities
        - 'individual_activities': individual method results
        - 'correlations': motif-cluster correlations (if applicable)
    """
    
    logger.info("Starting custom maelstrom analysis")
    
    # Ensure peak_clusters is a DataFrame with one column
    if isinstance(peak_clusters, pd.Series):
        cluster_df = peak_clusters.to_frame('cluster')
    else:
        cluster_df = peak_clusters.copy()
        if cluster_df.shape[1] != 1:
            raise ValueError("peak_clusters should have exactly 1 column")
    
    # Align indices
    common_peaks = peaks_motifs_matrix.index.intersection(cluster_df.index)
    logger.info(f"Using {len(common_peaks)} peaks with both motif scores and cluster assignments")
    
    motif_scores = peaks_motifs_matrix.loc[common_peaks]
    clusters = cluster_df.loc[common_peaks]
    
    # Convert motif scores to counts for count-based methods
    # Simple thresholding - you might want to adjust this
    motif_counts = (motif_scores > motif_scores.quantile(0.7, axis=0)).astype(int)
    
    # Run activity prediction for each method
    logger.info("Running activity prediction methods...")
    activity_results = {}
    
    for method in methods:
        logger.info(f"Running {method}")
        try:
            # Create the predictor
            predictor = Moap.create(method, ncpus=ncpus, random_state=random_state)
            
            # Choose appropriate input (scores vs counts)
            if predictor.pref_table == 'count':
                input_data = motif_counts
            else:
                input_data = motif_scores
                
            # Fit the predictor
            predictor.fit(input_data, clusters)
            
            # Store results
            activity_results[f"{method}.{predictor.pref_table}"] = predictor.act_
            
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            continue
    
    if len(activity_results) == 0:
        raise ValueError("No activity prediction methods succeeded")
    
    # Run rank aggregation if we have multiple methods
    if len(activity_results) > 1:
        logger.info("Running rank aggregation")
        final_scores = _df_rank_aggregation(
            clusters, activity_results, method=aggregation, ncpus=ncpus
        )
        
        # Add motif frequency information
        cluster_names = clusters.iloc[:, 0].unique()
        for cluster_name in cluster_names:
            cluster_mask = clusters.iloc[:, 0] == cluster_name
            cluster_peaks = motif_counts[cluster_mask]
            freq = (cluster_peaks > 0).sum() / len(cluster_peaks) * 100
            final_scores[f"{cluster_name} % with motif"] = freq
            
        # Add correlations between motif scores and cluster assignments
        logger.info("Computing correlations")
        for cluster_name in cluster_names:
            final_scores[f"corr {cluster_name}"] = 0
            cluster_indicator = (clusters.iloc[:, 0] == cluster_name).astype(int)
            
            for motif in final_scores.index:
                if motif in motif_scores.columns:
                    corr_val = pearsonr(motif_scores[motif], cluster_indicator)[0]
                    final_scores.loc[motif, f"corr {cluster_name}"] = corr_val
    else:
        # Only one method, use its results directly
        final_scores = list(activity_results.values())[0]
    
    logger.info("Analysis complete!")
    
    return {
        'final_scores': final_scores,
        'individual_activities': activity_results,
        'motif_scores': motif_scores,
        'motif_counts': motif_counts,
        'clusters': clusters
    }


def _df_rank_aggregation(df, dfs, method="int_stouffer", ncpus=4):
    """
    Rank aggregation function extracted from GimmeMotifs
    """
    df_p = pd.DataFrame(index=list(dfs.values())[0].index)
    names = list(dfs.values())[0].columns
    dfs_list = [
        pd.concat([v[col].rename(k) for k, v in dfs.items()], axis=1)
        for col in names
    ]

    pool = Pool(processes=ncpus)
    func = partial(rankagg, method=method)
    ret = pool.map(func, dfs_list)
    pool.close()
    pool.join()

    for name, result in zip(names, ret):
        df_p[name] = result.iloc[:, 0]  # rankagg returns DataFrame, we want the series

    if df.shape[1] != 1:
        df_p = df_p[df.columns]

    if method == "int_stouffer":
        df_p.columns = ["z-score " + str(c) for c in df_p.columns]
    else:
        df_p.columns = ["activity " + str(c) for c in df_p.columns]
    
    return df_p


def save_results(results, output_dir):
    """Save results in GimmeMotifs-compatible format"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save final aggregated results
    results['final_scores'].to_csv(
        os.path.join(output_dir, 'custom_maelstrom_final.txt'), 
        sep='\t'
    )
    
    # Save individual method results
    for method_name, activity_df in results['individual_activities'].items():
        activity_df.to_csv(
            os.path.join(output_dir, f'activity_{method_name}.txt'),
            sep='\t'
        )
    
    logger.info(f"Results saved to {output_dir}")