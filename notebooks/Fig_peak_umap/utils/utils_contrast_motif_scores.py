# NOTE. This script is designed to process the output from gimmemotifs maelstrom
# NOTE. this script is designed for quantifying the contrast metrics for a TF across clusters
# for the purpose of identifying the most informative TFS for the clustering
# the input adata should be the following format:
# cluster-by-motifs matrix (with aggregated motif scores in the element)
# - index: 'leiden', the cluster assignments
# - columns: motifs - the TF motif names
# - elements:the aggregated TF enrichment scores for that cluser-motif pair

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
# silhouette score
from sklearn.metrics import silhouette_score
# mutual information
from sklearn.feature_selection import mutual_info_classif
# ANOVA
from scipy import stats

def motif_total_information(pwm,
                            background=np.array([0.30, 0.20, 0.20, 0.30]),
                            eps=1e-6):
    """
    Total information content (bits) of a PWM.

    Parameters
    ----------
    pwm : array-like, shape (L, 4)
        Probability matrix for A, C, G, T at each position.
    background : length-4 array-like
        Background probs [q_A, q_C, q_G, q_T].  Default = zebrafish genome-wide.
    eps : float
        Small constant to avoid log2(0).

    Returns
    -------
    float
        Total information content of the motif in bits.
    """
    pwm = np.clip(np.asarray(pwm, dtype=float), eps, 1.0)
    bg  = np.clip(np.asarray(background, dtype=float), eps, 1.0)

    kl_cell  = pwm * (np.log2(pwm) - np.log2(bg))  # p * log2(p/q)
    total_ic = kl_cell.sum()                       # sum over all rows & cols
    return total_ic

def max_deviation_ratio(scores):
    """
    Calculate how much the most extreme value deviates from the mean
    
    Parameters:
    -----------
    scores : array-like
        Array of cluster-level scores for a motif
    
    Returns:
    --------
    float
        Maximum deviation ratio
    """
    mean_val = np.mean(scores)
    max_dev = np.max(np.abs(scores - mean_val))
    
    # Normalize by the overall range to make it comparable across TFs
    score_range = np.max(scores) - np.min(scores)
    if score_range > 0:
        return max_dev / score_range
    else:
        return 0

def peak_to_median_ratio(scores):
    """
    Calculate ratio of peak value to median value
    
    Parameters:
    -----------
    scores : array-like
        Array of cluster-level scores for a motif
    
    Returns:
    --------
    float
        Peak-to-average ratio
    """
    # Find peak (highest absolute value, positive or negative)
    peak_idx = np.argmax(np.abs(scores))
    peak_val = scores[peak_idx]
    
    # Calculate mean excluding the peak
    other_scores = np.concatenate([scores[:peak_idx], scores[peak_idx+1:]])
    median_others = np.median(np.abs(other_scores)) if len(other_scores) > 0 else 1
    
    # Calculate ratio
    if median_others > 0:
        return np.abs(peak_val) / median_others
    else:
        return np.abs(peak_val)

def percentile_ratio(scores, percentile=90):
    """
    Calculate ratio of top percentile to median
    
    Parameters:
    -----------
    scores : array-like
        Array of cluster-level scores for a motif
    percentile : int, default=90
        Percentile to use (0-100)
    
    Returns:
    --------
    float
        Percentile ratio
    """
    top_val = np.percentile(np.abs(scores), percentile)
    median_val = np.median(np.abs(scores))
    
    if median_val > 0:
        return top_val / median_val
    else:
        return top_val

def binarity_score(scores, threshold=0.5):
    """
    Calculate how binary the pattern is
    
    Parameters:
    -----------
    scores : array-like
        Array of cluster-level scores for a motif
    threshold : float, default=0.5
        Threshold for determining high vs. low values (0-1)
    
    Returns:
    --------
    float
        Binarity score (0-1, higher = more binary)
    """
    # Normalize scores to [0,1]
    norm_scores = np.abs(scores)
    if np.max(norm_scores) > 0:
        norm_scores = norm_scores / np.max(norm_scores)
    
    # Count clusters above and below threshold
    high = np.sum(norm_scores >= threshold)
    low = np.sum(norm_scores < threshold)
    total = len(norm_scores)
    
    # Calculate score (1 = perfect binary distribution, 0 = uniform distribution)
    return 1 - (4 * high * low) / (total * total)

def calculate_gini_index(scores):
    """
    Calculate Gini index for a distribution of values
    
    Parameters:
    -----------
    scores : array-like
        Array of cluster-level scores for a motif
    
    Returns:
    --------
    float
        Gini index (0-1, higher = more unequal distribution)
    """
    # Convert to absolute values to handle negative scores
    abs_scores = np.abs(scores)
    
    # Sort values in ascending order
    sorted_values = np.sort(abs_scores)
    n = len(sorted_values)
    
    # Calculate Gini index
    if np.sum(sorted_values) > 0:
        # This formula computes the Gini coefficient directly
        numerator = np.sum((2 * np.arange(1, n+1) - n - 1) * sorted_values)
        denominator = n * np.sum(sorted_values)
        gini = numerator / denominator
        return gini
    else:
        return 0

def calculate_discrete_mutual_information(scores):
    """
    Calculate an approximation of mutual information by discretizing scores
    
    Parameters:
    -----------
    scores : array-like
        Array of cluster-level scores for a motif
    
    Returns:
    --------
    float
        Approximate MI score
    """
    from sklearn.metrics import mutual_info_score
    
    # Discretize scores into bins
    n_bins = min(10, len(scores) // 2)  # Use fewer bins for small number of clusters
    if n_bins < 2:
        n_bins = 2
    
    # Create bins for scores
    abs_scores = np.abs(scores)
    bins = np.linspace(np.min(abs_scores), np.max(abs_scores), n_bins + 1)
    discretized_scores = np.digitize(abs_scores, bins)
    
    # Create "target" variable (just the indices of the clusters)
    # This effectively treats each cluster as a unique category
    target = np.arange(len(scores))
    
    # Calculate mutual information
    mi = mutual_info_score(discretized_scores, target)
    
    # Normalize by entropy of target
    from math import log2
    target_entropy = log2(len(target))
    normalized_mi = mi / target_entropy if target_entropy > 0 else 0
    
    return normalized_mi