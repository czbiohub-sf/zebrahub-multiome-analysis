# a module for sub-GRN analysis
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

# aggregate (average) the motif scores per cluster
def aggregate_motifs_by_clusters(adata_motifs, cluster_col="leiden_unified"):
    """
    Simple aggregation of motif scores by cluster.
    """
    
    print(f"Input shape: {adata_motifs.shape}")
    
    # Convert to DataFrame
    if hasattr(adata_motifs.X, 'toarray'):
        motif_matrix = adata_motifs.X.toarray()
    else:
        motif_matrix = adata_motifs.X
    
    motif_df = pd.DataFrame(
        motif_matrix,
        index=adata_motifs.obs_names,
        columns=adata_motifs.var_names
    )
    
    # Add cluster labels
    motif_df['cluster'] = adata_motifs.obs[cluster_col].astype(str)
    
    # Check cluster sizes
    cluster_counts = motif_df['cluster'].value_counts()
    print(f"Number of clusters: {len(cluster_counts)}")
    print(f"Cluster sizes - min: {cluster_counts.min()}, max: {cluster_counts.max()}, mean: {cluster_counts.mean():.1f}")
    
    # Aggregate by mean (since these are z-scores from GimmeMotifs)
    clusters_motifs_df = motif_df.groupby('cluster').mean()
    
    print(f"Aggregated shape: {clusters_motifs_df.shape}")
    print(f"Score range: [{clusters_motifs_df.values.min():.3f}, {clusters_motifs_df.values.max():.3f}]")
    
    return clusters_motifs_df, cluster_counts

def find_top_motifs_per_cluster(clusters_motifs_df, top_n=10):
    """
    Find top N motifs for each cluster.
    """
    top_motifs = {}
    
    for cluster_id in clusters_motifs_df.index:
        cluster_scores = clusters_motifs_df.loc[cluster_id]
        top_cluster_motifs = cluster_scores.nlargest(top_n).index.tolist()
        top_motifs[cluster_id] = top_cluster_motifs
    
    return top_motifs

def plot_motif_heatmap(clusters_motifs_df, top_n_variable_motifs=50):
    """
    Plot heatmap of most variable motifs across clusters.
    """
    # Find most variable motifs
    motif_variance = clusters_motifs_df.var(axis=0)
    top_variable_motifs = motif_variance.nlargest(top_n_variable_motifs).index
    
    # Subset for plotting
    plot_df = clusters_motifs_df[top_variable_motifs]
    
    # Create heatmap
    plt.figure(figsize=(12, max(6, len(plot_df) * 0.3)))
    sns.heatmap(plot_df, 
                cmap='RdBu_r', 
                center=0,
                cbar_kws={'label': 'Motif Enrichment Score'},
                xticklabels=False)  # Too many motifs to show labels
    plt.title(f'Top {top_n_variable_motifs} Variable Motifs Across Clusters')
    plt.xlabel('Motifs')
    plt.ylabel('Clusters')
    plt.tight_layout()
    plt.show()
    
    return plot_df

def analyze_cluster_motif_distributions(clusters_motifs_df, n_example_clusters=5):
    """
    Analyze motif score distributions within clusters to identify natural thresholds.
    """
    print("CLUSTER-MOTIF DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Basic stats
    print(f"Number of clusters: {len(clusters_motifs_df)}")
    print(f"Number of motifs: {len(clusters_motifs_df.columns)}")
    print(f"Overall mean score: {clusters_motifs_df.values.mean():.3f}")
    print(f"Overall score std: {clusters_motifs_df.values.std():.3f}")
    
    # Analyze distribution characteristics for each cluster
    cluster_stats = []
    
    for cluster_id in clusters_motifs_df.index:
        cluster_scores = clusters_motifs_df.loc[cluster_id].values
        
        stats = {
            'cluster_id': cluster_id,
            'mean': np.mean(cluster_scores),
            'std': np.std(cluster_scores),
            'min': np.min(cluster_scores),
            'max': np.max(cluster_scores),
            'q25': np.percentile(cluster_scores, 25),
            'q50': np.percentile(cluster_scores, 50),
            'q75': np.percentile(cluster_scores, 75),
            'q90': np.percentile(cluster_scores, 90),
            'q95': np.percentile(cluster_scores, 95),
            'n_positive': np.sum(cluster_scores > 0),
            'n_high_positive': np.sum(cluster_scores > 1),
            'n_very_high': np.sum(cluster_scores > 2),
        }
        cluster_stats.append(stats)
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    # Show top motifs for first few clusters
    print(f"\nTOP MOTIFS PER CLUSTER (first {n_example_clusters} clusters):")
    print("-" * 60)
    
    for i, cluster_id in enumerate(clusters_motifs_df.index[:n_example_clusters]):
        cluster_scores = clusters_motifs_df.loc[cluster_id]
        top_10 = cluster_scores.nlargest(10)
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Distribution: mean={cluster_stats_df.iloc[i]['mean']:.3f}, "
              f"std={cluster_stats_df.iloc[i]['std']:.3f}, "
              f"95th percentile={cluster_stats_df.iloc[i]['q95']:.3f}")
        print(f"  Positive scores: {cluster_stats_df.iloc[i]['n_positive']}/{len(cluster_scores)} "
              f"({100*cluster_stats_df.iloc[i]['n_positive']/len(cluster_scores):.1f}%)")
        
        print("  Top 10 motifs:")
        for motif, score in top_10.items():
            print(f"    {motif}: {score:.3f}")
    
    return cluster_stats_df

def plot_motif_distribution_analysis(clusters_motifs_df, n_example_clusters=6):
    """
    Plot distribution of motif scores within clusters to identify natural thresholds.
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot distributions for first n_example_clusters
    example_clusters = clusters_motifs_df.index[:n_example_clusters]
    
    for i, cluster_id in enumerate(example_clusters):
        cluster_scores = clusters_motifs_df.loc[cluster_id].values
        
        # Histogram
        axes[i].hist(cluster_scores, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add vertical lines for percentiles
        q75 = np.percentile(cluster_scores, 75)
        q90 = np.percentile(cluster_scores, 90)
        q95 = np.percentile(cluster_scores, 95)
        
        axes[i].axvline(q75, color='orange', linestyle='--', alpha=0.8, label=f'75th: {q75:.2f}')
        axes[i].axvline(q90, color='red', linestyle='--', alpha=0.8, label=f'90th: {q90:.2f}')
        axes[i].axvline(q95, color='darkred', linestyle='--', alpha=0.8, label=f'95th: {q95:.2f}')
        axes[i].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero')
        
        axes[i].set_title(f'Cluster {cluster_id}\nMean: {np.mean(cluster_scores):.3f}, Std: {np.std(cluster_scores):.3f}')
        axes[i].set_xlabel('Motif Score')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def detect_motif_enrichment_threshold(clusters_motifs_df, method='percentile', percentile=90):
    """
    Detect optimal threshold for motif enrichment based on distribution analysis.
    """
    print(f"\nMOTIF ENRICHMENT THRESHOLD DETECTION")
    print("="*50)
    
    thresholds_per_cluster = {}
    enriched_motifs_per_cluster = {}
    
    if method == 'percentile':
        print(f"Using {percentile}th percentile as threshold per cluster")
        
        for cluster_id in clusters_motifs_df.index:
            cluster_scores = clusters_motifs_df.loc[cluster_id]
            threshold = np.percentile(cluster_scores.values, percentile)
            enriched_motifs = cluster_scores[cluster_scores >= threshold]
            
            thresholds_per_cluster[cluster_id] = threshold
            enriched_motifs_per_cluster[cluster_id] = enriched_motifs.index.tolist()
    
    elif method == 'zscore':
        print("Using z-score > 1.5 within each cluster as threshold")
        
        for cluster_id in clusters_motifs_df.index:
            cluster_scores = clusters_motifs_df.loc[cluster_id]
            cluster_mean = cluster_scores.mean()
            cluster_std = cluster_scores.std()
            threshold = cluster_mean + 1.5 * cluster_std
            enriched_motifs = cluster_scores[cluster_scores >= threshold]
            
            thresholds_per_cluster[cluster_id] = threshold
            enriched_motifs_per_cluster[cluster_id] = enriched_motifs.index.tolist()
    
    elif method == 'mixture':
        print("Using Gaussian mixture model to find natural breakpoint")
        from sklearn.mixture import GaussianMixture
        
        for cluster_id in clusters_motifs_df.index:
            cluster_scores = clusters_motifs_df.loc[cluster_id].values.reshape(-1, 1)
            
            # Try fitting 2-component mixture
            try:
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(cluster_scores)
                
                # Find intersection point between two Gaussians as threshold
                means = gmm.means_.flatten()
                stds = np.sqrt(gmm.covariances_.flatten())
                
                # Simple approximation: midpoint between means
                threshold = np.mean(means)
                enriched_motifs = clusters_motifs_df.loc[cluster_id][clusters_motifs_df.loc[cluster_id] >= threshold]
                
                thresholds_per_cluster[cluster_id] = threshold
                enriched_motifs_per_cluster[cluster_id] = enriched_motifs.index.tolist()
                
            except:
                # Fall back to percentile if mixture model fails
                threshold = np.percentile(cluster_scores, 90)
                enriched_motifs = clusters_motifs_df.loc[cluster_id][clusters_motifs_df.loc[cluster_id] >= threshold]
                thresholds_per_cluster[cluster_id] = threshold
                enriched_motifs_per_cluster[cluster_id] = enriched_motifs.index.tolist()
    
    # Summary statistics
    threshold_values = list(thresholds_per_cluster.values())
    enriched_counts = [len(motifs) for motifs in enriched_motifs_per_cluster.values()]
    
    print(f"\nThreshold statistics:")
    print(f"  Mean threshold: {np.mean(threshold_values):.3f}")
    print(f"  Threshold range: [{np.min(threshold_values):.3f}, {np.max(threshold_values):.3f}]")
    print(f"  Mean enriched motifs per cluster: {np.mean(enriched_counts):.1f}")
    print(f"  Range of enriched motifs: {np.min(enriched_counts)} - {np.max(enriched_counts)}")
    
    return {
        'thresholds': thresholds_per_cluster,
        'enriched_motifs': enriched_motifs_per_cluster,
        'threshold_stats': {
            'mean_threshold': np.mean(threshold_values),
            'mean_enriched_count': np.mean(enriched_counts)
        }
    }

def comprehensive_motif_analysis(clusters_motifs_df):
    """
    Complete motif distribution analysis workflow.
    """
    print("COMPREHENSIVE MOTIF ANALYSIS")
    print("="*70)
    
    # Step 1: Basic distribution analysis
    cluster_stats = analyze_cluster_motif_distributions(clusters_motifs_df, n_example_clusters=8)
    
    # Step 2: Plot distributions
    print("\n" + "="*70)
    print("PLOTTING MOTIF SCORE DISTRIBUTIONS")
    print("="*70)
    fig = plot_motif_distribution_analysis(clusters_motifs_df, n_example_clusters=6)
    
    # Step 3: Try different threshold methods
    print("\n" + "="*70)
    print("TESTING DIFFERENT THRESHOLD METHODS")
    print("="*70)
    
    methods_to_try = [
        ('percentile', {'percentile': 90}),
        ('percentile', {'percentile': 95}), 
        ('zscore', {}),
        ('mixture', {})
    ]
    
    threshold_results = {}
    
    for method, kwargs in methods_to_try:
        print(f"\n--- Testing {method} method ---")
        try:
            result = detect_motif_enrichment_threshold(clusters_motifs_df, method=method, **kwargs)
            threshold_results[f"{method}_{kwargs}"] = result
        except Exception as e:
            print(f"Method {method} failed: {e}")
    
    # Step 4: Recommend best method
    print("\n" + "="*70)
    print("THRESHOLD METHOD COMPARISON")
    print("="*70)
    
    for method_name, result in threshold_results.items():
        stats = result['threshold_stats']
        print(f"{method_name}: avg_threshold={stats['mean_threshold']:.3f}, "
              f"avg_enriched={stats['mean_enriched_count']:.1f}")
    
    return {
        'cluster_stats': cluster_stats,
        'threshold_results': threshold_results,
        'plot': fig
    }

def save_cluster_motif_results(clusters_motifs_df, top_motifs_dict, output_dir="results"):
    """
    Save all results to files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated scores
    clusters_motifs_df.to_csv(f"{output_dir}/clusters_motifs_aggregated.csv")
    print(f"Saved aggregated motif scores to {output_dir}/clusters_motifs_aggregated.csv")
    
    # Save top motifs per cluster
    top_motifs_df = pd.DataFrame.from_dict(top_motifs_dict, orient='index')
    top_motifs_df.to_csv(f"{output_dir}/top_motifs_per_cluster.csv")
    print(f"Saved top motifs per cluster to {output_dir}/top_motifs_per_cluster.csv")
    
    # Save cluster statistics
    cluster_stats = analyze_cluster_motif_patterns(clusters_motifs_df)
    cluster_stats['cluster_means'].to_csv(f"{output_dir}/cluster_mean_scores.csv")
    cluster_stats['motif_variance'].to_csv(f"{output_dir}/motif_variance_across_clusters.csv")
    
    print(f"All results saved to {output_dir}/")
    
    return cluster_stats

def analyze_cluster_motif_patterns(clusters_motifs_df):
    """
    Analyze patterns in cluster-motif relationships.
    """
    print("CLUSTER-MOTIF ANALYSIS")
    print("="*50)
    
    # Basic stats
    print(f"Number of clusters: {len(clusters_motifs_df)}")
    print(f"Number of motifs: {len(clusters_motifs_df.columns)}")
    print(f"Mean score: {clusters_motifs_df.values.mean():.3f}")
    print(f"Score std: {clusters_motifs_df.values.std():.3f}")
    
    # Find clusters with highest/lowest mean scores
    cluster_means = clusters_motifs_df.mean(axis=1)
    print(f"\nClusters with highest mean motif scores:")
    print(cluster_means.nlargest(5))
    
    print(f"\nClusters with lowest mean motif scores:")
    print(cluster_means.nsmallest(5))
    
    # Find most/least variable motifs
    motif_variance = clusters_motifs_df.var(axis=0)
    print(f"\nMost variable motifs across clusters:")
    print(motif_variance.nlargest(10))
    
    # Correlation between clusters
    cluster_corr = clusters_motifs_df.T.corr()
    print(f"\nMean correlation between clusters: {cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].mean():.3f}")
    
    return {
        'cluster_means': cluster_means,
        'motif_variance': motif_variance,
        'cluster_correlation': cluster_corr
    }

def plot_motif_distributions_grid(clusters_motifs_df, max_clusters=16, figsize=(20, 16)):
    """
    Plot histogram of motif scores for multiple clusters in a grid layout.
    
    Parameters:
    -----------
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs dataframe with motif scores
    max_clusters : int
        Maximum number of clusters to plot
    figsize : tuple
        Figure size (width, height)
    """
    
    # Select clusters to plot
    clusters_to_plot = clusters_motifs_df.index[:max_clusters]
    n_clusters = len(clusters_to_plot)
    
    # Calculate grid dimensions
    n_cols = 4
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_clusters > 1 else [axes]
    
    for i, cluster_id in enumerate(clusters_to_plot):
        cluster_scores = clusters_motifs_df.loc[cluster_id].values
        
        # Plot histogram
        axes[i].hist(cluster_scores, bins=40, alpha=0.7, color='skyblue', 
                    edgecolor='black', density=True)
        
        # Add reference lines
        mean_score = np.mean(cluster_scores)
        median_score = np.median(cluster_scores)
        q75 = np.percentile(cluster_scores, 75)
        q90 = np.percentile(cluster_scores, 90)
        q95 = np.percentile(cluster_scores, 95)
        
        axes[i].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero')
        axes[i].axvline(mean_score, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_score:.2f}')
        axes[i].axvline(q90, color='orange', linestyle='--', alpha=0.8, label=f'90th: {q90:.2f}')
        axes[i].axvline(q95, color='darkred', linestyle='--', alpha=0.8, label=f'95th: {q95:.2f}')
        
        # Title and labels
        axes[i].set_title(f'Cluster {cluster_id}\n'
                         f'Mean: {mean_score:.2f}, Std: {np.std(cluster_scores):.2f}')
        axes[i].set_xlabel('Motif Score')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Motif Score Distributions by Cluster', fontsize=16, y=1.02)
    plt.show()
    
    return fig

def plot_single_cluster_detailed(clusters_motifs_df, cluster_id, figsize=(12, 8)):
    """
    Detailed distribution plot for a single cluster with multiple visualizations.
    """
    cluster_scores = clusters_motifs_df.loc[cluster_id].values
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Histogram with reference lines
    axes[0,0].hist(cluster_scores, bins=50, alpha=0.7, color='lightblue', 
                   edgecolor='black', density=True)
    
    # Add percentile lines
    percentiles = [75, 90, 95, 99]
    colors = ['green', 'orange', 'red', 'darkred']
    for p, color in zip(percentiles, colors):
        value = np.percentile(cluster_scores, p)
        axes[0,0].axvline(value, color=color, linestyle='--', alpha=0.8, 
                         label=f'{p}th: {value:.2f}')
    
    axes[0,0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero')
    axes[0,0].set_title(f'Histogram - Cluster {cluster_id}')
    axes[0,0].set_xlabel('Motif Score')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Box plot
    axes[0,1].boxplot(cluster_scores, vert=True)
    axes[0,1].set_title(f'Box Plot - Cluster {cluster_id}')
    axes[0,1].set_ylabel('Motif Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot (test for normality)
    stats.probplot(cluster_scores, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_scores = np.sort(cluster_scores)
    y_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1,1].plot(sorted_scores, y_vals, linewidth=2)
    
    # Add percentile markers
    for p, color in zip([90, 95, 99], ['orange', 'red', 'darkred']):
        value = np.percentile(cluster_scores, p)
        axes[1,1].axvline(value, color=color, linestyle='--', alpha=0.8, 
                         label=f'{p}th percentile')
    
    axes[1,1].set_title('Cumulative Distribution')
    axes[1,1].set_xlabel('Motif Score')
    axes[1,1].set_ylabel('Cumulative Probability')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Detailed Distribution Analysis - Cluster {cluster_id}', 
                 fontsize=14, y=1.02)
    plt.show()
    
    # Print summary statistics
    print(f"\nSUMMARY STATISTICS - Cluster {cluster_id}")
    print("="*50)
    print(f"Total motifs: {len(cluster_scores)}")
    print(f"Mean: {np.mean(cluster_scores):.3f}")
    print(f"Std: {np.std(cluster_scores):.3f}")
    print(f"Min: {np.min(cluster_scores):.3f}")
    print(f"Max: {np.max(cluster_scores):.3f}")
    print(f"Median: {np.median(cluster_scores):.3f}")
    
    print(f"\nPercentiles:")
    for p in [75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(cluster_scores, p):.3f}")
    
    print(f"\nMotifs above thresholds:")
    for thresh in [0, 1, 2, 3]:
        count = np.sum(cluster_scores > thresh)
        pct = 100 * count / len(cluster_scores)
        print(f"  > {thresh}: {count} motifs ({pct:.1f}%)")
    
    return fig

def plot_clusters_comparison(clusters_motifs_df, cluster_ids, figsize=(15, 10)):
    """
    Compare distributions of multiple specific clusters side by side.
    """
    n_clusters = len(cluster_ids)
    
    fig, axes = plt.subplots(2, n_clusters, figsize=figsize)
    if n_clusters == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i, cluster_id in enumerate(cluster_ids):
        cluster_scores = clusters_motifs_df.loc[cluster_id].values
        
        # Histogram
        axes[0, i].hist(cluster_scores, bins=40, alpha=0.7, color=colors[i], 
                       edgecolor='black', density=True)
        
        # Add key percentiles
        q95 = np.percentile(cluster_scores, 95)
        axes[0, i].axvline(q95, color='red', linestyle='--', alpha=0.8, 
                          label=f'95th: {q95:.2f}')
        axes[0, i].axvline(0, color='black', linestyle='-', alpha=0.5)
        
        axes[0, i].set_title(f'Cluster {cluster_id}')
        axes[0, i].set_xlabel('Motif Score')
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[1, i].boxplot(cluster_scores, vert=True)
        axes[1, i].set_title(f'Box Plot - {cluster_id}')
        axes[1, i].set_ylabel('Motif Score')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Cluster Distribution Comparison', fontsize=14, y=1.02)
    plt.show()
    
    return fig

def plot_all_clusters_overlay(clusters_motifs_df, max_clusters=10, figsize=(12, 8)):
    """
    Overlay distributions of multiple clusters to see overall patterns.
    """
    plt.figure(figsize=figsize)
    
    clusters_to_plot = clusters_motifs_df.index[:max_clusters]
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters_to_plot)))
    
    for i, cluster_id in enumerate(clusters_to_plot):
        cluster_scores = clusters_motifs_df.loc[cluster_id].values
        
        plt.hist(cluster_scores, bins=40, alpha=0.3, color=colors[i], 
                density=True, label=f'Cluster {cluster_id}')
    
    plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero')
    plt.xlabel('Motif Score')
    plt.ylabel('Density')
    plt.title(f'Overlaid Distributions - First {len(clusters_to_plot)} Clusters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def explore_threshold_effects(clusters_motifs_df, cluster_id, thresholds=[1, 1.5, 2, 2.5], figsize=(12, 6)):
    """
    Visualize how different thresholds would affect motif selection for a cluster.
    """
    cluster_scores = clusters_motifs_df.loc[cluster_id].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with threshold lines
    ax1.hist(cluster_scores, bins=50, alpha=0.7, color='lightblue', 
             edgecolor='black', density=True)
    
    colors = ['green', 'orange', 'red', 'darkred']
    threshold_counts = []
    
    for thresh, color in zip(thresholds, colors):
        ax1.axvline(thresh, color=color, linestyle='--', linewidth=2, 
                   label=f'Threshold {thresh}')
        count = np.sum(cluster_scores >= thresh)
        threshold_counts.append(count)
    
    ax1.set_title(f'Cluster {cluster_id} - Threshold Effects')
    ax1.set_xlabel('Motif Score')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar plot of motif counts at different thresholds
    ax2.bar(range(len(thresholds)), threshold_counts, color=colors, alpha=0.7)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Number of Enriched Motifs')
    ax2.set_title('Enriched Motifs vs Threshold')
    ax2.set_xticks(range(len(thresholds)))
    ax2.set_xticklabels([str(t) for t in thresholds])
    ax2.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(threshold_counts):
        ax2.text(i, count + max(threshold_counts)*0.01, str(count), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print threshold summary
    print(f"\nTHRESHOLD ANALYSIS - Cluster {cluster_id}")
    print("="*50)
    total_motifs = len(cluster_scores)
    for thresh, count in zip(thresholds, threshold_counts):
        pct = 100 * count / total_motifs
        print(f"Threshold >= {thresh}: {count} motifs ({pct:.1f}%)")
    
    return fig


def quick_plot_cluster_distributions(clusters_motifs_df, n_clusters=12):
    """
    Simple function to quickly plot motif distributions for multiple clusters.
    Shows both 95th and 99th percentiles for outlier identification.
    """
    
    # Select first n_clusters or all if fewer
    clusters_to_plot = clusters_motifs_df.index[:min(n_clusters, len(clusters_motifs_df))]
    
    # Set up grid
    n_cols = 3
    n_rows = int(np.ceil(len(clusters_to_plot) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if len(clusters_to_plot) > 1 else [axes]
    
    for i, cluster_id in enumerate(clusters_to_plot):
        scores = clusters_motifs_df.loc[cluster_id].values
        
        # Plot histogram
        axes[i].hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add key reference lines
        q95 = np.percentile(scores, 95)
        q99 = np.percentile(scores, 99)
        mean_score = np.mean(scores)
        
        # Count outliers for each threshold
        n_above_95 = np.sum(scores >= q95)
        n_above_99 = np.sum(scores >= q99)
        
        axes[i].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero')
        axes[i].axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.2f}')
        axes[i].axvline(q95, color='orange', linestyle='--', label=f'95th: {q95:.2f} ({n_above_95})')
        axes[i].axvline(q99, color='darkred', linestyle='--', linewidth=2, label=f'99th: {q99:.2f} ({n_above_99})')
        
        # Labels and title
        axes[i].set_title(f'Cluster {cluster_id}')
        axes[i].set_xlabel('Motif Score')
        axes[i].set_ylabel('Count')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused plots
    for j in range(len(clusters_to_plot), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Motif Score Distributions by Cluster (95th & 99th Percentiles)', fontsize=16, y=1.02)
    plt.show()

def quick_single_cluster(clusters_motifs_df, cluster_id):
    """
    Quick detailed plot for a single cluster.
    """
    scores = clusters_motifs_df.loc[cluster_id].values
    
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(scores, bins=40, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Reference lines
    percentiles = [75, 90, 95, 99]
    colors = ['green', 'orange', 'red', 'darkred']
    
    for p, color in zip(percentiles, colors):
        value = np.percentile(scores, p)
        plt.axvline(value, color=color, linestyle='--', alpha=0.8, 
                   label=f'{p}th: {value:.2f}')
    
    plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero')
    
    # Labels
    plt.title(f'Motif Score Distribution - Cluster {cluster_id}')
    plt.xlabel('Motif Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Print stats
    print(f"Cluster {cluster_id} Statistics:")
    print(f"  Total motifs: {len(scores)}")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std: {np.std(scores):.3f}")
    print(f"  95th percentile: {np.percentile(scores, 95):.3f}")
    print(f"  Motifs > 0: {np.sum(scores > 0)} ({100*np.sum(scores > 0)/len(scores):.1f}%)")
    print(f"  Motifs > 2: {np.sum(scores > 2)} ({100*np.sum(scores > 2)/len(scores):.1f}%)")
    
    plt.show()
