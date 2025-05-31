import os
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# plot the pathway enrichment results for subclusters within a coarse cluster
# the input is the target coarse cluster, the dictionary of sub-clustered adata objects, and the key for the subclusters in the adata.obs
# the output is the figure and the summary statistics about the pathways found
def plot_subcluster_pathways(dict_adata_sub, target_coarse_cluster, 
                           subcluster_key="leiden_sub_0.7_merged_renumbered",
                           base_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis",
                           top_n=10,
                           nrows=4, 
                           ncols=5,
                           save_plot=True,
                           show_plot=True,
                           figsize_width_scale=0.70,
                           use_linked_overlap=False,
                           nonsig_color='#F5F5DC',
                           title_size=12,
                           label_size=10,
                           tick_size=9,
                           legend_size=10):
    """
    Plot pathway enrichment results for subclusters within a coarse cluster.
    
    Parameters:
    -----------
    dict_adata_sub : dict
        Dictionary containing sub-clustered adata objects
    target_coarse_cluster : int
        The coarse cluster ID to analyze
    subcluster_key : str
        Column name for subclusters in adata.obs
    base_dir : str
        Base directory for data files
    top_n : int
        Number of top pathways to show per subcluster
    nrows : int
        Number of rows in subplot grid
    ncols : int
        Number of columns in subplot grid
    save_plot : bool
        Whether to save the plot
    show_plot : bool
        Whether to display the plot
    figsize_width_scale : float
        Width scaling factor for figure size
    use_linked_overlap : bool
        Whether to use linked_overlap directory
    nonsig_color : str
        Color for non-significant pathways
    title_size : int
        Font size for subplot titles
    label_size : int
        Font size for axis labels
    tick_size : int
        Font size for tick labels and pathway names
    legend_size : int
        Font size for legend
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    pathway_summary : dict
        Summary statistics about pathways found
    """
    
    print(f"Plotting pathway enrichment for coarse cluster {target_coarse_cluster}")
    print("=" * 60)
    
    # Get the adata_sub for this coarse cluster
    if target_coarse_cluster not in dict_adata_sub:
        raise ValueError(f"Coarse cluster {target_coarse_cluster} not found in dict_adata_sub")
    
    adata_sub = dict_adata_sub[target_coarse_cluster]
    
    # Get unique subclusters and force consecutive ordering: 0, 1, 2, 3, 4, 5, etc.
    unique_subclusters_raw = adata_sub.obs[subcluster_key].unique()
    unique_subclusters = [str(i) for i in np.arange(0, len(unique_subclusters_raw))]
    
    print(f"Analyzing coarse cluster {target_coarse_cluster}")
    print(f"Found {len(unique_subclusters)} sub-clusters: {unique_subclusters}")
    
    # Get sub-cluster colors from the AnnData object (matching UMAP colors)
    color_key = f"{subcluster_key}_colors"
    print(f"Looking for color key: {color_key}")
    print(f"Available keys in uns: {list(adata_sub.uns.keys())}")
    
    # Try different possible color keys
    possible_color_keys = [
        f"{subcluster_key}_colors",
        "leiden_sub_0.7_merged_renumbered_colors", 
        "leiden_sub_0.7_merged_colors",
        "leiden_sub_0.7_colors"
    ]
    
    subcluster_colors_list = None
    color_key_used = None
    
    for key in possible_color_keys:
        if key in adata_sub.uns:
            subcluster_colors_list = adata_sub.uns[key]
            color_key_used = key
            print(f"Found colors using key: {key}")
            break
    
    # If no colors found, generate them
    if subcluster_colors_list is None:
        print("No colors found in uns, generating new colors...")
        import matplotlib.cm as cm
        n_clusters = len(unique_subclusters)
        colors = cm.tab20(np.linspace(0, 1, n_clusters))
        subcluster_colors_list = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r,g,b,a in colors]
        color_key_used = "generated"
    
    # Create color mapping
    subcluster_labels = adata_sub.obs[subcluster_key].cat.categories
    subcluster_colors = {str(cluster): color for cluster, color in zip(subcluster_labels, subcluster_colors_list)}
    
    print(f"Color mapping: {subcluster_colors}")
    
    # Determine input directory
    if use_linked_overlap:
        input_dir = os.path.join(base_dir, f"FishEnrichR_leiden_coarse_cluster_{target_coarse_cluster}_linked_overlap")
    else:
        input_dir = os.path.join(base_dir, f"FishEnrichR_leiden_coarse_cluster_{target_coarse_cluster}")
    
    # Fallback to original directory if linked_overlap doesn't exist
    if not os.path.exists(input_dir):
        fallback_dir = os.path.join(base_dir, f"FishEnrichR_leiden_coarse_cluster_{target_coarse_cluster}")
        if os.path.exists(fallback_dir):
            input_dir = fallback_dir
            print(f"Fallback directory used: {input_dir}")
        else:
            raise ValueError(f"Neither primary nor fallback directory exists")
    else:
        print(f"Using input directory: {input_dir}")
    
    # Set up output path
    output_fig = os.path.join(input_dir, f"leiden_coarse_{target_coarse_cluster}_subclusts_pathways_colored.pdf")
    
    # Set up the figure
    fig_width = 6.5 * ncols * figsize_width_scale
    fig_height = 3.5 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), constrained_layout=True)
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    # Track statistics
    files_found = 0
    files_missing = 0
    total_significant = 0
    total_pathways = 0
    
    # Loop through clusters explicitly
    for i, cluster in enumerate(unique_subclusters):
        # Try different file extensions
        possible_files = [
            os.path.join(input_dir, f"cluster_{cluster}_enrichment_WikiPathways_2018.tsv"),
            os.path.join(input_dir, f"cluster_{cluster}_enrichment_WikiPathways_2018.tsv.txt")
        ]
        
        input_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                input_file = file_path
                break
        
        if input_file is None:
            print(f"File not found for cluster {cluster}, skipping...")
            files_missing += 1
            # Create empty subplot
            if i < len(axes):
                ax = axes[i]
                ax.set_title(f'Cluster {cluster} (No data)', fontsize=title_size, pad=5)
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            continue
        
        files_found += 1
        
        try:
            df = pd.read_csv(input_file, sep='\t')
            df = df.sort_values('Combined Score', ascending=True).tail(top_n)
            
            # Get the color for this specific cluster
            cluster_color = subcluster_colors.get(str(cluster), '#999999')
            
            # Determine colors based on p-value
            colors = [
                cluster_color if p <= 0.05 else nonsig_color
                for p in df['P-value']
            ]
            
            # Count significant pathways
            significant_count = (df['P-value'] <= 0.05).sum()
            total_significant += significant_count
            total_pathways += len(df)
            
            # Select the correct subplot
            if i < len(axes):
                ax = axes[i]
                
                # Remove grid lines inside the bar plots
                ax.grid(False)
                
                # Create horizontal bars
                bars = ax.barh(range(len(df)), df['Combined Score'], color=colors, alpha=0.7)
                
                # Title
                ax.set_title(f'Cluster {cluster}', fontsize=title_size, pad=5)
                
                # Remove y-ticks and y-axis labels completely
                ax.set_yticks([])
                ax.set_yticklabels([])
                
                # Add labels **inside** the bars
                for bar, label in zip(bars, df['Term']):
                    ax.text(
                        bar.get_width() * 0.02,  # Small left margin inside the bar
                        bar.get_y() + bar.get_height() / 2,
                        label.split('_WP')[0],  # Remove '_WP' suffix
                        va='center', ha='left', fontsize=tick_size, color='black', fontweight='bold'
                    )
                
                # Format x-axis
                ax.tick_params(axis='x', labelsize=tick_size)
                ax.set_xlabel('Combined Score', fontsize=label_size)
            
        except Exception as e:
            print(f"Error processing cluster {cluster}: {e}")
            files_missing += 1
    
    print(f"\nFiles found: {files_found}, Files missing: {files_missing}")
    
    # Remove unused subplots if there are fewer than nrows * ncols
    for j in range(len(unique_subclusters), len(axes)):
        fig.delaxes(axes[j])
    
    # Add a legend with a sample of actual colors used
    legend_elements = [
        mpatches.Patch(facecolor=list(subcluster_colors.values())[0], alpha=0.7, label='p ≤ 0.05 (cluster color)'),
        mpatches.Patch(facecolor=nonsig_color, alpha=0.7, label='p > 0.05')
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=legend_size)
    
    # Save plot
    if save_plot:
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_fig}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    # Don't close if user wants to return the figure
    if not show_plot:
        plt.close()
    
    # Create summary
    pathway_summary = {
        'coarse_cluster': target_coarse_cluster,
        'subclusters_total': len(unique_subclusters),
        'subclusters_processed': files_found,
        'subclusters_missing': files_missing,
        'total_pathways': total_pathways,
        'significant_pathways': total_significant,
        'significance_rate': total_significant/total_pathways*100 if total_pathways > 0 else 0,
        'color_key_used': color_key_used,
        'input_directory': input_dir,
        'output_file': output_fig if save_plot else None
    }
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  Subclusters total: {len(unique_subclusters)}")
    print(f"  Subclusters processed: {files_found}")
    print(f"  Subclusters missing: {files_missing}")
    print(f"  Total pathways: {total_pathways}")
    print(f"  Significant pathways: {total_significant}")
    print(f"  Significance rate: {pathway_summary['significance_rate']:.1f}%")
    
    return fig, pathway_summary

# Convenience function to plot multiple coarse clusters
def plot_all_coarse_clusters(coarse_clusts, dict_adata_sub, **kwargs):
    """
    Plot pathway enrichment for multiple coarse clusters.
    
    Parameters:
    -----------
    coarse_clusts : list
        List of coarse cluster IDs to process
    dict_adata_sub : dict
        Dictionary containing sub-clustered adata objects
    **kwargs
        Additional arguments passed to plot_subcluster_pathways
        
    Returns:
    --------
    results : dict
        Dictionary with results for each coarse cluster
    """
    results = {}
    
    for clust in coarse_clusts:
        print(f"\n{'='*80}")
        try:
            fig, summary = plot_subcluster_pathways(clust, dict_adata_sub, **kwargs)
            results[clust] = {
                'figure': fig,
                'summary': summary,
                'success': True
            }
        except Exception as e:
            print(f"Error processing coarse cluster {clust}: {e}")
            results[clust] = {
                'figure': None,
                'summary': None,
                'success': False,
                'error': str(e)
            }
    
    return results

# Usage examples:
"""
# Single coarse cluster
fig, summary = plot_subcluster_pathways(
    target_coarse_cluster=1, 
    dict_adata_sub=dict_adata_sub
)

# Multiple coarse clusters with custom parameters
results = plot_all_coarse_clusters(
    coarse_clusts=[1, 7, 13, 22], 
    dict_adata_sub=dict_adata_sub,
    top_n=15,
    show_plot=False,
    save_plot=True
)

# Access results
for clust_id, result in results.items():
    if result['success']:
        print(f"Cluster {clust_id}: {result['summary']['significance_rate']:.1f}% significant pathways")
"""