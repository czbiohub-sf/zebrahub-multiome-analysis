"""
Refactored Peak Cluster Entropy Analysis Module

Streamlined implementation for analyzing peak cluster accessibility patterns.
Core concept: Aggregate across metadata, then across peaks within clusters.

Author: Yang-Joon Kim
Date: 2025-08-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import re


def create_cluster_metadata_profiles(adata, cluster_col='leiden_coarse', 
                                   metadata_type='celltype', min_cells=20,
                                   compute_sem=True, verbose=True):
    """
    Unified function to create cluster-by-metadata profiles with standard error.
    
    Workflow:
    1. Filter pseudobulk groups with <min_cells
    2. Parse metadata from group names (celltype_timepoint format)
    3. Aggregate across the OTHER metadata (e.g., if analyzing celltype, average across timepoints)
    4. Aggregate across peaks within each cluster (compute mean and SEM)
    
    Parameters:
    -----------
    adata : AnnData
        Data with peaks as obs, pseudobulk groups as var
    cluster_col : str
        Cluster column name in adata.obs
    metadata_type : str
        'celltype', 'timepoint', or 'lineage'
    min_cells : int
        Minimum cells per pseudobulk group
    compute_sem : bool
        Whether to compute standard error of the mean
    verbose : bool
        Print progress information
        
    Returns:
    --------
    pd.DataFrame
        cluster-by-metadata matrix (means)
    pd.DataFrame or None
        cluster-by-metadata matrix (SEMs) if compute_sem=True
    dict
        metadata info (colors, order, etc.)
    """
    
    if verbose:
        print(f"Creating cluster-by-{metadata_type} profiles...")
        print(f"Input data: {adata.shape[0]} peaks × {adata.shape[1]} pseudobulk groups")
    
    # Step 1: Filter reliable groups
    reliable_groups = []
    for col in adata.var.index:
        cell_count = int(adata[:, col].var["n_cells"].iloc[0])
        if cell_count >= min_cells:
            reliable_groups.append(col)
    
    if verbose:
        print(f"Kept {len(reliable_groups)}/{adata.shape[1]} groups with ≥{min_cells} cells")
    
    # Step 2: Parse metadata from group names
    metadata_mapping = _parse_group_metadata(reliable_groups, verbose=verbose)
    
    # Step 3: Get data for reliable groups only
    reliable_indices = [adata.var.index.get_loc(col) for col in reliable_groups]
    reliable_data = adata.X[:, reliable_indices]
    if hasattr(reliable_data, 'toarray'):
        reliable_data = reliable_data.toarray()
    
    # Create DataFrame with cluster assignments
    data_df = pd.DataFrame(
        reliable_data,
        columns=reliable_groups,
        index=adata.obs.index
    )
    data_df['cluster'] = adata.obs[cluster_col].astype(str)
    
    # Step 4: First aggregation - average across OTHER metadata dimension
    if metadata_type == 'lineage':
        # Use default lineage mapping if not provided
        lineage_mapping = _get_default_lineage_mapping()
        group_to_metadata = {}
        for group in reliable_groups:
            celltype = metadata_mapping['celltype_mapping'].get(group)
            if celltype:
                for lineage, celltypes in lineage_mapping.items():
                    if celltype in celltypes:
                        group_to_metadata[group] = lineage
                        break
    else:
        group_to_metadata = metadata_mapping[f'{metadata_type}_mapping']
    
    # Create peak-by-metadata matrix (first aggregation)
    peak_metadata_data = pd.DataFrame(index=data_df.index)
    
    for metadata_category in set(group_to_metadata.values()):
        # Get groups for this metadata category
        category_groups = [g for g, m in group_to_metadata.items() if m == metadata_category]
        
        if category_groups:
            # Average across groups in this category (e.g., average across timepoints for each celltype)
            category_mean = data_df[category_groups].mean(axis=1)
            peak_metadata_data[metadata_category] = category_mean
    
    # Step 5: Second aggregation - aggregate across peaks within each cluster (with SEM)
    cluster_metadata_profiles = pd.DataFrame(index=data_df['cluster'].unique())
    cluster_metadata_sems = pd.DataFrame(index=data_df['cluster'].unique()) if compute_sem else None
    
    for metadata_category in peak_metadata_data.columns:
        category_data = pd.DataFrame({
            'accessibility': peak_metadata_data[metadata_category],
            'cluster': data_df['cluster']
        })
        
        # Group by cluster and compute mean and SEM
        cluster_stats = category_data.groupby('cluster')['accessibility'].agg(['mean', 'std', 'count'])
        
        # Store means
        cluster_metadata_profiles[metadata_category] = cluster_stats['mean']
        
        # Compute and store SEM if requested
        if compute_sem:
            sem_values = cluster_stats['std'] / np.sqrt(cluster_stats['count'])
            cluster_metadata_sems[metadata_category] = sem_values.fillna(0)  # Handle single-peak clusters
    
    # Remove any clusters with all-zero profiles
    cluster_metadata_profiles = cluster_metadata_profiles.fillna(0)
    valid_clusters = (cluster_metadata_profiles > 0).any(axis=1)
    cluster_metadata_profiles = cluster_metadata_profiles.loc[valid_clusters]
    
    if compute_sem:
        cluster_metadata_sems = cluster_metadata_sems.fillna(0)
        cluster_metadata_sems = cluster_metadata_sems.loc[valid_clusters]
    
    # Create metadata info
    metadata_info = {
        'categories': list(cluster_metadata_profiles.columns),
        'colors': _create_color_palette(list(cluster_metadata_profiles.columns), metadata_type),
        'order': _get_default_order(list(cluster_metadata_profiles.columns), metadata_type)
    }
    
    if verbose:
        print(f"Created {cluster_metadata_profiles.shape} cluster-by-{metadata_type} matrix")
        if compute_sem:
            print(f"Computed standard errors for {cluster_metadata_sems.shape[0]} clusters")
        print(f"Categories: {metadata_info['categories']}")
    
    if compute_sem:
        return cluster_metadata_profiles, cluster_metadata_sems, metadata_info
    else:
        return cluster_metadata_profiles, None, metadata_info


def analyze_accessibility_patterns(cluster_metadata_profiles, metadata_type='celltype',
                                 entropy_threshold=0.8, dominance_threshold=0.3):
    """
    Analyze accessibility patterns using entropy and dominance.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-metadata matrix
    metadata_type : str
        Type of metadata being analyzed
    entropy_threshold : float
        Threshold for broad vs specific classification
    dominance_threshold : float
        Threshold for specific accessibility
        
    Returns:
    --------
    pd.DataFrame
        Results with patterns and metrics
    """
    
    results = []
    
    for cluster in cluster_metadata_profiles.index:
        profile = cluster_metadata_profiles.loc[cluster]
        
        # Compute metrics
        metrics = _compute_accessibility_metrics(profile)
        
        # Classify pattern
        pattern, confidence = _classify_pattern(metrics, entropy_threshold, dominance_threshold)
        
        results.append({
            'cluster': cluster,
            'pattern': pattern,
            'confidence': confidence,
            'dominant_category': metrics['dominant_category'],
            **metrics  # Include all metrics
        })
    
    return pd.DataFrame(results)


def plot_cluster_profile(cluster_metadata_profiles, cluster_id, metadata_info, 
                        cluster_metadata_sems=None, figsize=(8, 5), 
                        show_metrics=True, show_error_bars=True, return_fig=True):
    """
    Plot accessibility profile for a single cluster with optional error bars.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-metadata matrix (means)
    cluster_id : str
        Cluster ID to plot
    metadata_info : dict
        Metadata information (colors, order, etc.)
    cluster_metadata_sems : pd.DataFrame or None
        cluster-by-metadata matrix (SEMs)
    figsize : tuple
        Figure size
    show_metrics : bool
        Whether to show entropy/dominance in title
    show_error_bars : bool
        Whether to show error bars (requires cluster_metadata_sems)
    return_fig : bool
        Whether to return figure object instead of showing
        
    Returns:
    --------
    matplotlib.figure.Figure or None
        Figure object if return_fig=True
    """
    
    if cluster_id not in cluster_metadata_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return None
    
    # Get and order profile
    profile = cluster_metadata_profiles.loc[cluster_id]
    if 'order' in metadata_info and metadata_info['order']:
        ordered_profile = profile.reindex(metadata_info['order'], fill_value=0)
    else:
        ordered_profile = profile.sort_values(ascending=False)
    
    # Get corresponding SEMs if available
    sems = None
    if show_error_bars and cluster_metadata_sems is not None:
        if cluster_id in cluster_metadata_sems.index:
            cluster_sems = cluster_metadata_sems.loc[cluster_id]
            if 'order' in metadata_info and metadata_info['order']:
                sems = cluster_sems.reindex(metadata_info['order'], fill_value=0)
            else:
                sems = cluster_sems.reindex(ordered_profile.index, fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors
    colors = [metadata_info['colors'].get(cat, 'steelblue') 
              for cat in ordered_profile.index]
    
    # Plot bars with error bars
    x_pos = range(len(ordered_profile))
    bars = ax.bar(x_pos, ordered_profile.values, color=colors, alpha=0.8,
                  yerr=sems.values if sems is not None else None,
                  capsize=3, error_kw={'alpha': 0.7})
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ordered_profile.index, rotation=45, ha='right')
    ax.set_ylabel('Mean Accessibility')
    
    # Title with metrics
    if show_metrics:
        metrics = _compute_accessibility_metrics(profile)
        title = f'Cluster {cluster_id}\nEntropy: {metrics["entropy"]:.3f}, Dominance: {metrics["dominance"]:.3f}'
    else:
        title = f'Cluster {cluster_id}'
    
    ax.set_title(title, fontweight='bold')
    ax.grid(False)
    
    # Add horizontal line at median
    median_val = cluster_metadata_profiles.values.flatten()
    median_val = np.median(median_val[median_val > 0])
    ax.axhline(y=median_val, linestyle='--', alpha=0.5, color='gray')
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def create_profile_grid(cluster_metadata_profiles, metadata_info, 
                       cluster_metadata_sems=None, cluster_ids=None, 
                       ncols=6, nrows=6, figsize=(24, 24), 
                       show_error_bars=True, save_path=None):
    """
    Create grid of cluster profiles with optional error bars.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-metadata matrix (means)
    metadata_info : dict
        Metadata information
    cluster_metadata_sems : pd.DataFrame or None
        cluster-by-metadata matrix (SEMs)
    cluster_ids : list or None
        Specific clusters to plot
    ncols, nrows : int
        Grid dimensions
    figsize : tuple
        Overall figure size
    show_error_bars : bool
        Whether to show error bars
    save_path : str or None
        Path to save figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Grid figure object
    """
    
    if cluster_ids is None:
        cluster_ids = _sort_cluster_ids_numerically(cluster_metadata_profiles.index)
    
    cluster_ids = cluster_ids[:ncols * nrows]  # Limit to grid size
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, cluster_id in enumerate(cluster_ids):
        if i >= len(axes):
            break
            
        ax = axes[i]
        _plot_on_axis(cluster_metadata_profiles, cluster_id, metadata_info, ax,
                     cluster_metadata_sems=cluster_metadata_sems,
                     show_error_bars=show_error_bars)
    
    # Hide unused subplots
    for j in range(len(cluster_ids), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid saved to: {save_path}")
    
    return fig


def run_complete_analysis(adata, cluster_col='leiden_coarse', 
                         metadata_type='celltype', min_cells=20, compute_sem=True):
    """
    Run complete analysis workflow with optional SEM calculation.
    
    Parameters:
    -----------
    adata : AnnData
        Input data
    cluster_col : str
        Cluster column
    metadata_type : str
        Type of metadata to analyze
    min_cells : int
        Minimum cells per group
    compute_sem : bool
        Whether to compute standard errors
        
    Returns:
    --------
    tuple
        (profiles_df, results_df, metadata_info, sems_df)
        sems_df is None if compute_sem=False
    """
    
    # Create profiles
    if compute_sem:
        profiles_df, sems_df, metadata_info = create_cluster_metadata_profiles(
            adata, cluster_col, metadata_type, min_cells, compute_sem=True
        )
    else:
        profiles_df, _, metadata_info = create_cluster_metadata_profiles(
            adata, cluster_col, metadata_type, min_cells, compute_sem=False
        )
        sems_df = None
    
    # Analyze patterns
    results_df = analyze_accessibility_patterns(profiles_df, metadata_type)
    
    return profiles_df, results_df, metadata_info, sems_df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_group_metadata(groups, verbose=True):
    """Parse celltype and timepoint from group names (celltype_timepoint format)."""
    
    celltype_mapping = {}
    timepoint_mapping = {}
    
    for group in groups:
        # Match pattern: celltype_timepoint (e.g., neural_15somites)
        match = re.search(r'^(.+)_(\d+somites)$', group)
        if match:
            celltype, timepoint = match.groups()
            celltype_mapping[group] = celltype
            timepoint_mapping[group] = timepoint
    
    if verbose:
        unique_celltypes = set(celltype_mapping.values())
        unique_timepoints = set(timepoint_mapping.values())
        print(f"Parsed {len(celltype_mapping)} groups: "
              f"{len(unique_celltypes)} celltypes × {len(unique_timepoints)} timepoints")
    
    return {
        'celltype_mapping': celltype_mapping,
        'timepoint_mapping': timepoint_mapping
    }


def _compute_accessibility_metrics(profile):
    """Compute comprehensive accessibility metrics."""
    
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


def _classify_pattern(metrics, entropy_threshold=0.8, dominance_threshold=0.3):
    """Classify accessibility pattern based on metrics."""
    
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    dominant_cat = metrics['dominant_category']
    
    if entropy >= entropy_threshold:
        return "broadly_accessible", "high"
    elif dominance >= dominance_threshold:
        return f"specific_{dominant_cat}", "high" if dominance >= 0.4 else "medium"
    else:
        return f"enriched_{dominant_cat}", "medium"


def _create_color_palette(categories, metadata_type):
    """Create color palette for categories."""
    
    if metadata_type == 'celltype':
        # Create biologically-informed color palette
        color_scheme = {
            # CNS/Neural - blues
            'neural': '#1f77b4', 'neural_optic': '#aec7e8', 'neural_posterior': '#4682b4',
            'neural_telencephalon': '#6495ed', 'neurons': '#0000cd', 'differentiating_neurons': '#4169e1',
            'hindbrain': '#1e90ff', 'midbrain_hindbrain_boundary': '#87ceeb', 'spinal_cord': '#00bfff',
            'optic_cup': '#87cefa', 'floor_plate': '#b0e0e6', 'neural_floor_plate': '#add8e6',
            
            # Neural Crest - purples
            'neural_crest': '#9467bd', 'enteric_neurons': '#c5b0d5',
            
            # Early Mesoderm - dark greens
            'NMPs': '#2ca02c', 'tail_bud': '#98df8a',
            
            # Axial Mesoderm - brown
            'notochord': '#8c564b',
            
            # Paraxial Mesoderm - greens
            'PSM': '#2ca02c', 'somites': '#98df8a', 'fast_muscle': '#c5b0d5', 'muscle': '#bcbd22',
            
            # Lateral Plate Mesoderm - reds
            'lateral_plate_mesoderm': '#d62728', 'heart_myocardium': '#ff7f0e',
            'hematopoietic_vasculature': '#ff9896', 'hemangioblasts': '#ffbb78',
            
            # Other Mesoderm - oranges
            'pharyngeal_arches': '#ff7f0e', 'pronephros': '#ffbb78', 'hatching_gland': '#ffd700',
            
            # Endoderm - yellows
            'endoderm': '#bcbd22', 'endocrine_pancreas': '#dbdb8d',
            
            # Ectoderm - grays
            'epidermis': '#7f7f7f',
            
            # Germline - pink
            'primordial_germ_cells': '#e377c2'
        }
        
        # Use predefined colors if available, otherwise generate
        palette = {}
        for cat in categories:
            if cat in color_scheme:
                palette[cat] = color_scheme[cat]
            else:
                # Generate color for unknown categories
                palette[cat] = plt.cm.Set3(hash(cat) % 12 / 12)
                
        return palette
        
    elif metadata_type == 'timepoint':
        # Use sequential palette for timepoints (temporal progression)
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        return {cat: colors[i] for i, cat in enumerate(categories)}
    else:
        # Default qualitative palette for other metadata types
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        return {cat: colors[i] for i, cat in enumerate(categories)}


def _get_default_order(categories, metadata_type):
    """Get default ordering for categories."""
    
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
    """Default lineage to celltype mapping (updated to match celltype ordering)."""
    
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


def _sort_cluster_ids_numerically(cluster_ids):
    """Sort cluster IDs numerically."""
    
    def extract_number(cluster_id):
        match = re.search(r'(\d+)', str(cluster_id))
        return int(match.group(1)) if match else float('inf')
    
    return sorted(cluster_ids, key=extract_number)


def _plot_on_axis(cluster_metadata_profiles, cluster_id, metadata_info, ax,
                 cluster_metadata_sems=None, show_error_bars=True):
    """Plot single cluster profile on given axis with optional error bars."""
    
    if cluster_id not in cluster_metadata_profiles.index:
        ax.set_visible(False)
        return
    
    profile = cluster_metadata_profiles.loc[cluster_id]
    
    # Order profile
    if 'order' in metadata_info and metadata_info['order']:
        ordered_profile = profile.reindex(metadata_info['order'], fill_value=0)
    else:
        ordered_profile = profile.sort_values(ascending=False)
    
    # Get corresponding SEMs if available
    sems = None
    if show_error_bars and cluster_metadata_sems is not None:
        if cluster_id in cluster_metadata_sems.index:
            cluster_sems = cluster_metadata_sems.loc[cluster_id]
            if 'order' in metadata_info and metadata_info['order']:
                sems = cluster_sems.reindex(metadata_info['order'], fill_value=0)
            else:
                sems = cluster_sems.reindex(ordered_profile.index, fill_value=0)
    
    # Colors
    colors = [metadata_info['colors'].get(cat, 'steelblue') 
              for cat in ordered_profile.index]
    
    # Plot with error bars
    x_pos = range(len(ordered_profile))
    ax.bar(x_pos, ordered_profile.values, color=colors, alpha=0.8,
           yerr=sems.values if sems is not None else None,
           capsize=2, error_kw={'alpha': 0.6, 'linewidth': 1})
    
    # Metrics for title
    metrics = _compute_accessibility_metrics(profile)
    title = f'Cluster {cluster_id}\nE={metrics["entropy"]:.2f}, D={metrics["dominance"]:.2f}'
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ordered_profile.index, rotation=45, ha='right', fontsize=8)
    ax.grid(False)

# A function to plot heatmap for cluster-by-pseudobulk, subsetted by "cluster"
def plot_cluster_heatmap(cluster_profiles, cluster_id, celltype_orders, cell_type_color_dict,
                        figsize=(12, 8), cmap='RdBu_r', save_path=None, show_values=False,
                        vmin=None, vmax=None):
    """
    Generate a timepoint × celltype heatmap for a specific cluster.
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        cluster-by-pseudobulk matrix 
    cluster_id : str or int
        Cluster ID to plot
    celltype_orders : list
        List of celltypes in desired order
    cell_type_color_dict : dict
        Dictionary mapping celltypes to colors
    figsize : tuple
        Figure size
    cmap : str
        Colormap name (default: 'RdBu_r')
    save_path : str or None
        Path to save figure
    show_values : bool
        Whether to show values in cells
    vmin, vmax : float or None
        Colormap limits
        
    Returns:
    --------
    matplotlib.figure.Figure
        Heatmap figure
    """
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    
    # Convert cluster_id to string for consistency
    cluster_id = str(cluster_id)
    
    if cluster_id not in cluster_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return None
    
    # Get accessibility values for this cluster
    cluster_data = cluster_profiles.loc[cluster_id]
    
    # Parse pseudobulk group names into timepoint and celltype
    timepoint_celltype_data = []
    
    for group, accessibility in cluster_data.items():
        # Split by last underscore to separate celltype from timepoint
        if '_' in group and 'somites' in group:
            parts = group.rsplit('_', 1)  # Split from right, only once
            if len(parts) == 2 and parts[1].endswith('somites'):
                celltype = parts[0]
                timepoint = parts[1]
                timepoint_celltype_data.append({
                    'celltype': celltype,
                    'timepoint': timepoint,
                    'accessibility': accessibility
                })
    
    if not timepoint_celltype_data:
        print(f"No valid pseudobulk groups found for cluster {cluster_id}")
        return None
    
    # Convert to DataFrame and pivot
    heatmap_df = pd.DataFrame(timepoint_celltype_data)
    heatmap_matrix = heatmap_df.pivot(index='timepoint', columns='celltype', values='accessibility')
    heatmap_matrix = heatmap_matrix.fillna(0)  # Fill missing combinations with 0
    
    # Sort timepoints numerically
    timepoint_order = sorted(heatmap_matrix.index, 
                           key=lambda x: int(x.replace('somites', '')) if x.replace('somites', '').isdigit() else 0)
    heatmap_matrix = heatmap_matrix.reindex(timepoint_order)
    
    # Order celltypes according to biological order
    available_celltypes = [ct for ct in celltype_orders if ct in heatmap_matrix.columns]
    # Add any celltypes not in the predefined order
    remaining_celltypes = [ct for ct in heatmap_matrix.columns if ct not in celltype_orders]
    final_celltype_order = available_celltypes + sorted(remaining_celltypes)
    heatmap_matrix = heatmap_matrix[final_celltype_order]
    
    # Create figure with extra space at bottom for color blocks
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap using imshow
    im = ax.imshow(heatmap_matrix.values, cmap=cmap, aspect='auto', 
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Set ticks and labels
    ax.set_xticks(range(len(heatmap_matrix.columns)))
    ax.set_xticklabels(heatmap_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_matrix.index)))
    ax.set_yticklabels(heatmap_matrix.index)
    
    # Add colored blocks for celltypes at bottom
    for i, celltype in enumerate(heatmap_matrix.columns):
        if celltype in cell_type_color_dict:
            color = cell_type_color_dict[celltype]
            # Add rectangle at bottom of plot
            rect = Rectangle((i-0.4, -0.8), 0.8, 0.3, 
                           facecolor=color, edgecolor='black', linewidth=0.5,
                           transform=ax.transData, clip_on=False)
            ax.add_patch(rect)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accessibility', rotation=270, labelpad=20)
    
    # Add values to cells if requested
    if show_values:
        for i in range(len(heatmap_matrix.index)):
            for j in range(len(heatmap_matrix.columns)):
                value = heatmap_matrix.iloc[i, j]
                if value > 0:  # Only show non-zero values
                    # Choose text color based on background
                    matrix_range = heatmap_matrix.values.max() - heatmap_matrix.values.min()
                    threshold = heatmap_matrix.values.min() + matrix_range * 0.5
                    text_color = 'white' if value > threshold else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                           color=text_color, fontsize=8)
    
    # Title and labels
    ax.set_title(f'Cluster {cluster_id} - Timepoint × Celltype Accessibility', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Celltype', fontweight='bold')
    ax.set_ylabel('Timepoint', fontweight='bold')
    
    # Adjust layout to accommodate color blocks
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    return fig


# Example usage:
# Define your ordering and colors outside the function
# celltype_orders = ['neural', 'neural_optic', ...]
# cell_type_color_dict = {'neural': '#e6ab02', 'NMPs': '#8dd3c7', ...}
#
# fig = plot_cluster_heatmap(cluster_profiles, cluster_id=23, 
#                           celltype_orders=celltype_orders,
#                           cell_type_color_dict=cell_type_color_dict,
#                           figsize=(15, 8), cmap='RdBu_r',
#                           vmin=0, vmax=100,
#                           save_path='cluster_23_heatmap.pdf')