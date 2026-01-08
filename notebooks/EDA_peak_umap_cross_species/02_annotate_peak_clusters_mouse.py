# %% [markdown]
# # Annotate Peak Clusters with Most Accessible Celltypes and Timepoints - Mouse Dataset
# 
# This notebook annotates peak UMAP clusters with the celltypes and developmental timepoints
# that show the highest accessibility for each cluster.
# 
# **Input**: Preprocessed mouse peak object with UMAP computed
# **Output**: Annotated peak object with cluster-specific celltype/timepoint labels

# %% Load the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# %% Define paths
# TODO: Update these paths for your mouse dataset
input_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated.h5ad"
output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/"
figure_path = output_dir + "figures/"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)
sc.settings.figdir = figure_path

# %% Load the preprocessed mouse peak object
print("="*60)
print("Loading mouse peak object...")
print("="*60)

peaks_by_pseudobulk = sc.read_h5ad(input_path)

print(f"\nLoaded object:")
print(f"  Shape: {peaks_by_pseudobulk.shape}")
print(f"  Peaks (obs): {peaks_by_pseudobulk.n_obs:,}")
print(f"  Pseudobulk groups (vars): {peaks_by_pseudobulk.n_vars}")

# Check if UMAP exists
if 'X_umap' not in peaks_by_pseudobulk.obsm:
    raise ValueError("UMAP not found! Please compute UMAP first.")

print("\nExisting UMAP found ✓")

# %% [markdown]
# ## Import utility functions for accessibility analysis

# %% Import necessary functions
import re
import sys
import importlib

# Add path to utility functions
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig_peak_umap/scripts/")

# Import and reload to get latest version
import peak_accessibility_utils
importlib.reload(peak_accessibility_utils)
from peak_accessibility_utils import (
    create_cluster_celltype_profiles,
    create_cluster_timepoint_profiles,
    get_top_annotations,
    classify_cluster_specificity,
    add_specificity_to_adata
)

print("Utility functions imported successfully ✓")

# %% [markdown]
# ## Perform Leiden clustering on the peak UMAP

# %% Leiden clustering
print("\n" + "="*60)
print("Performing leiden clustering on peaks...")
print("="*60)

# Compute leiden clustering at two resolutions
sc.tl.leiden(peaks_by_pseudobulk, resolution=0.4, key_added='leiden_coarse')
sc.tl.leiden(peaks_by_pseudobulk, resolution=1.0, key_added='leiden_fine')

print(f"\nCoarse clustering: {len(peaks_by_pseudobulk.obs['leiden_coarse'].unique())} clusters")
print(f"Fine clustering: {len(peaks_by_pseudobulk.obs['leiden_fine'].unique())} clusters")

# Visualize the clusters
sc.pl.umap(peaks_by_pseudobulk, color=['leiden_coarse', 'leiden_fine'], 
           legend_loc='on data', legend_fontsize=8,
           save='_mouse_leiden_clusters.png')

# %% [markdown]
# ## Parse celltype and timepoint information from pseudobulk groups

# %% Parse pseudobulk group names
print("\n" + "="*60)
print("Parsing celltype and timepoint information...")
print("="*60)

# The var_names should be in format: celltype_timepoint (e.g., "Astro_E7.5")
celltype_list = []
timepoint_list = []

for var_name in peaks_by_pseudobulk.var_names:
    # Split by last underscore to separate celltype from timepoint
    parts = var_name.rsplit('_', 1)
    if len(parts) == 2:
        celltype, timepoint = parts
        celltype_list.append(celltype)
        timepoint_list.append(timepoint)
    else:
        # If no underscore, treat entire name as celltype
        celltype_list.append(var_name)
        timepoint_list.append('unknown')

# Store in var
peaks_by_pseudobulk.var['celltype'] = celltype_list
peaks_by_pseudobulk.var['timepoint'] = timepoint_list

print(f"\nFound {len(set(celltype_list))} unique celltypes")
print(f"Found {len(set(timepoint_list))} unique timepoints")
print(f"\nCelltypes: {sorted(set(celltype_list))[:10]}...")  # Show first 10
print(f"Timepoints: {sorted(set(timepoint_list))}")

# %% [markdown]
# ## Compute accessibility profiles for each cluster

# %% Compute celltype accessibility profiles
print("\n" + "="*60)
print("Computing celltype accessibility profiles...")
print("="*60)

cluster_celltype_profiles_coarse = create_cluster_celltype_profiles(
    peaks_by_pseudobulk, 
    cluster_col='leiden_coarse',
    verbose=True
)

cluster_celltype_profiles_fine = create_cluster_celltype_profiles(
    peaks_by_pseudobulk,
    cluster_col='leiden_fine',
    verbose=True
)

# %% Compute timepoint accessibility profiles
print("\n" + "="*60)
print("Computing timepoint accessibility profiles...")
print("="*60)

cluster_timepoint_profiles_coarse = create_cluster_timepoint_profiles(
    peaks_by_pseudobulk,
    cluster_col='leiden_coarse',
    verbose=True
)

cluster_timepoint_profiles_fine = create_cluster_timepoint_profiles(
    peaks_by_pseudobulk,
    cluster_col='leiden_fine',
    verbose=True
)

# %% [markdown]
# ## Classify cluster specificity patterns

# %% Classify specificity
print("\n" + "="*60)
print("Classifying cluster specificity patterns...")
print("="*60)

# Classify coarse clusters based on accessibility patterns
specificity_classifications = classify_cluster_specificity(
    cluster_celltype_profiles_coarse,
    entropy_threshold=0.75,          # High entropy = ubiquitous
    dominance_threshold=0.5,         # Top celltype has >50% = specific
    n_specific_min=2,                # At least 2 celltypes
    n_specific_max=5,                # At most 5 celltypes for "specific"
    accessibility_threshold_percentile=50,  # Above median = "high"
    verbose=True
)

# Add classifications to adata
add_specificity_to_adata(
    peaks_by_pseudobulk,
    specificity_classifications,
    cluster_col='leiden_coarse'
)

print("\nSpecificity classification complete ✓")

# %% [markdown]
# ## Identify top accessible celltypes and timepoints for each cluster

# %% Get top annotations
print("\n" + "="*60)
print("Identifying top accessible celltypes and timepoints...")
print("="*60)

top_celltypes_coarse = get_top_annotations(cluster_celltype_profiles_coarse, top_n=3)
top_timepoints_coarse = get_top_annotations(cluster_timepoint_profiles_coarse, top_n=3)

# Print summary
print("\n=== Coarse Clustering Summary ===")
for cluster_id in sorted(top_celltypes_coarse.keys()):
    print(f"\nCluster {cluster_id}:")
    print(f"  Top celltypes: {', '.join([f'{ct} ({acc:.2f})' for ct, acc in top_celltypes_coarse[cluster_id]])}")
    print(f"  Top timepoints: {', '.join([f'{tp} ({acc:.2f})' for tp, acc in top_timepoints_coarse[cluster_id]])}")

# %% [markdown]
# ## Annotate individual peaks with cluster-level annotations

# %% Annotate peaks
print("\n" + "="*60)
print("Annotating individual peaks with cluster assignments...")
print("="*60)

# Add top celltype for each peak (based on its cluster)
peaks_by_pseudobulk.obs['top_celltype'] = peaks_by_pseudobulk.obs['leiden_coarse'].map(
    lambda x: top_celltypes_coarse[x][0][0] if x in top_celltypes_coarse else 'unknown'
)

# Add top timepoint for each peak (based on its cluster)
peaks_by_pseudobulk.obs['top_timepoint'] = peaks_by_pseudobulk.obs['leiden_coarse'].map(
    lambda x: top_timepoints_coarse[x][0][0] if x in top_timepoints_coarse else 'unknown'
)

# Create combined annotation
peaks_by_pseudobulk.obs['cluster_annotation'] = (
    'C' + peaks_by_pseudobulk.obs['leiden_coarse'].astype(str) + 
    ': ' + peaks_by_pseudobulk.obs['top_celltype']
)

# Convert top_timepoint to numeric for continuous colormap visualization
# Extract numeric part from mouse embryonic stages (e.g., "E7.5" -> 7.5)
def extract_numeric_timepoint(tp_str):
    """Extract numeric value from timepoint strings like 'E7.5', '8-10 somites', etc."""
    try:
        # Handle E-stage format (E7.5, E8.5, etc.)
        if str(tp_str).startswith('E'):
            return float(str(tp_str)[1:])
        # Handle somites format
        elif 'somites' in str(tp_str):
            # Extract first number from ranges like "8-10 somites"
            import re
            match = re.search(r'(\d+)', str(tp_str))
            if match:
                return float(match.group(1))
        # Try direct numeric conversion
        return float(tp_str)
    except:
        return np.nan

peaks_by_pseudobulk.obs['top_timepoint_numeric'] = peaks_by_pseudobulk.obs['top_timepoint'].apply(
    extract_numeric_timepoint
)

print("\nAnnotation complete!")
print(f"Added columns: top_celltype, top_timepoint, top_timepoint_numeric, cluster_annotation")
if peaks_by_pseudobulk.obs['top_timepoint_numeric'].notna().any():
    print(f"Timepoint range: {peaks_by_pseudobulk.obs['top_timepoint_numeric'].min():.1f} - {peaks_by_pseudobulk.obs['top_timepoint_numeric'].max():.1f}")

# %% [markdown]
# ## Visualize annotated UMAPs

# %% Individual UMAP plots
print("\n" + "="*60)
print("Creating UMAP visualizations...")
print("="*60)

# Plot 1: Top celltype
sc.pl.umap(peaks_by_pseudobulk, color='top_celltype',
           title='Most Accessible Celltype (Mouse)',
           save='_mouse_most_access_ct.png')

# Plot 2: Top timepoint (continuous colormap if numeric, categorical otherwise)
if peaks_by_pseudobulk.obs['top_timepoint_numeric'].notna().any():
    sc.pl.umap(peaks_by_pseudobulk, color='top_timepoint_numeric',
               title='Most Accessible Timepoint (Mouse Embryonic Day)',
               cmap='viridis',
               save='_mouse_most_access_tp.png')
else:
    sc.pl.umap(peaks_by_pseudobulk, color='top_timepoint',
               title='Most Accessible Timepoint (Mouse)',
               save='_mouse_most_access_tp.png')

# %% Save intermediate annotated object
peaks_by_pseudobulk.write_h5ad(output_dir + "peaks_by_pb_celltype_stage_annotated.h5ad")
print(f"\nSaved annotated object to: {output_dir}peaks_by_pb_celltype_stage_annotated.h5ad")

# %% 3-panel combined visualization
print("\nCreating 3-panel combined visualization...")

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Get UMAP coordinates
umap_coords = peaks_by_pseudobulk.obsm['X_umap']

# Plot 1: Leiden clusters (categorical)
sc.pl.umap(peaks_by_pseudobulk, color='leiden_coarse', 
           ax=axes[0], show=False, title='Leiden Clusters (coarse)')

# Plot 2: Top celltype (categorical)
sc.pl.umap(peaks_by_pseudobulk, color='top_celltype',
           ax=axes[1], show=False, title='Most Accessible Celltype')

# Plot 3: Top timepoint (continuous with viridis if numeric)
if peaks_by_pseudobulk.obs['top_timepoint_numeric'].notna().any():
    timepoint_numeric = peaks_by_pseudobulk.obs['top_timepoint_numeric'].values
    valid_mask = ~pd.isna(timepoint_numeric)
    
    scatter = axes[2].scatter(
        umap_coords[valid_mask, 0], 
        umap_coords[valid_mask, 1],
        c=timepoint_numeric[valid_mask],
        cmap='viridis',
        s=1,
        alpha=0.8,
        rasterized=True
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label('Embryonic Day', rotation=270, labelpad=20)
    
    axes[2].set_title('Most Accessible Timepoint\n(early = blue, late = yellow)', fontsize=12)
    axes[2].set_xlabel('UMAP1')
    axes[2].set_ylabel('UMAP2')
    axes[2].set_aspect('equal')
else:
    # Fallback to categorical
    sc.pl.umap(peaks_by_pseudobulk, color='top_timepoint',
               ax=axes[2], show=False, title='Most Accessible Timepoint')

plt.tight_layout()
plt.savefig(figure_path + 'mouse_peak_umap_annotated_most_accessible_ct_tp.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_peak_umap_annotated_most_accessible_ct_tp.pdf', 
            bbox_inches='tight')
plt.show()

print("3-panel visualization saved ✓")

# %% [markdown]
# ## Summary statistics

# %% Summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

print("\nCelltype distribution across clusters:")
print(peaks_by_pseudobulk.obs.groupby('leiden_coarse')['top_celltype'].value_counts().head(20))

print("\n\nTimepoint distribution across clusters:")
print(peaks_by_pseudobulk.obs.groupby('leiden_coarse')['top_timepoint'].value_counts().head(20))

print("\n\nOverall top celltype distribution:")
print(peaks_by_pseudobulk.obs['top_celltype'].value_counts())

print("\n\nOverall top timepoint distribution:")
print(peaks_by_pseudobulk.obs['top_timepoint'].value_counts())

# %% [markdown]
# ## Visualize accessibility specificity patterns

# %% Specificity patterns
print("\n" + "="*60)
print("Visualizing specificity patterns...")
print("="*60)

# Create color palette for specificity patterns
specificity_colors = {
    'ubiquitous': '#e41a1c',     # Red
    'specific': '#377eb8',       # Blue
    'moderate': '#4daf4a'        # Green
}

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Plot 1: Accessibility pattern
sc.pl.umap(peaks_by_pseudobulk, color='accessibility_pattern',
           ax=axes[0], show=False, 
           title='Accessibility Specificity Pattern',
           palette=specificity_colors)

# Plot 2: Entropy (continuous)
sc.pl.umap(peaks_by_pseudobulk, color='accessibility_entropy',
           ax=axes[1], show=False,
           title='Accessibility Entropy\n(high = ubiquitous, low = specific)',
           cmap='RdYlBu_r', vmin=0, vmax=1)

# Plot 3: Top celltype for context
sc.pl.umap(peaks_by_pseudobulk, color='top_celltype',
           ax=axes[2], show=False,
           title='Most Accessible Celltype')

plt.tight_layout()
plt.savefig(figure_path + 'mouse_peak_umap_specificity_patterns.png',
            dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_peak_umap_specificity_patterns.pdf',
            bbox_inches='tight')
plt.show()

# Print detailed breakdown
print("\n=== Specificity Pattern Details ===")
for pattern in ['ubiquitous', 'specific', 'moderate']:
    pattern_peaks = peaks_by_pseudobulk.obs[peaks_by_pseudobulk.obs['accessibility_pattern'] == pattern]
    n_peaks = len(pattern_peaks)
    if n_peaks > 0:
        print(f"\n{pattern.upper()} peaks (n={n_peaks:,}):")
        print(f"  Mean entropy: {pattern_peaks['accessibility_entropy'].mean():.3f}")
        print(f"  Median entropy: {pattern_peaks['accessibility_entropy'].median():.3f}")
        print(f"  Top celltypes:")
        print(pattern_peaks['top_celltype'].value_counts().head(5))

# %% [markdown]
# ## Plot accessibility profiles for example clusters

# %% Define plotting function
def plot_cluster_accessibility_profile(cluster_id, celltype_profiles, timepoint_profiles,
                                       figsize=(14, 5), save_path=None):
    """
    Plot celltype and timepoint accessibility profiles for a specific cluster.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Celltype profile
    celltypes = list(celltype_profiles[cluster_id].keys())
    celltype_values = list(celltype_profiles[cluster_id].values())
    
    axes[0].bar(range(len(celltypes)), celltype_values, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(celltypes)))
    axes[0].set_xticklabels(celltypes, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Accessibility')
    axes[0].set_title(f'Cluster {cluster_id}: Celltype Accessibility')
    axes[0].grid(False)
    
    # Timepoint profile
    timepoints = list(timepoint_profiles[cluster_id].keys())
    timepoint_values = list(timepoint_profiles[cluster_id].values())
    
    # Sort timepoints by numeric value (E7.5, E8.5, etc.)
    try:
        sorted_indices = sorted(range(len(timepoints)), 
                              key=lambda i: float(timepoints[i].replace('E', '').replace('somites', '')))
        timepoints_sorted = [timepoints[i] for i in sorted_indices]
        values_sorted = [timepoint_values[i] for i in sorted_indices]
    except:
        # If conversion fails, keep original order
        timepoints_sorted = timepoints
        values_sorted = timepoint_values
    
    axes[1].bar(range(len(timepoints_sorted)), values_sorted, color='coral', alpha=0.7)
    axes[1].set_xticks(range(len(timepoints_sorted)))
    axes[1].set_xticklabels(timepoints_sorted, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Accessibility')
    axes[1].set_title(f'Cluster {cluster_id}: Timepoint Accessibility')
    axes[1].grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# %% Plot example cluster profiles
print("\n" + "="*60)
print("Plotting example cluster profiles...")
print("="*60)

# Plot first 3 clusters as examples
for cluster_id in sorted(list(cluster_celltype_profiles_coarse.keys()))[:3]:
    print(f"\nPlotting cluster {cluster_id}...")
    plot_cluster_accessibility_profile(
        cluster_id,
        cluster_celltype_profiles_coarse,
        cluster_timepoint_profiles_coarse,
        save_path=f"{figure_path}mouse_cluster_{cluster_id}_accessibility_profile.png"
    )

# %% [markdown]
# ## Save final annotated object

# %% Save final output
print("\n" + "="*60)
print("Saving final annotated peak object...")
print("="*60)

final_output_path = output_dir + "peaks_by_pb_celltype_stage_annotated_with_clusters.h5ad"
peaks_by_pseudobulk.write_h5ad(final_output_path)

print(f"Saved to: {final_output_path}")

# Print summary
print("\n" + "="*60)
print("ANNOTATION PIPELINE COMPLETE!")
print("="*60)
print(f"\nFinal object shape: {peaks_by_pseudobulk.shape}")
print(f"Total annotations added:")
print(f"  - leiden_coarse: {len(peaks_by_pseudobulk.obs['leiden_coarse'].unique())} clusters")
print(f"  - leiden_fine: {len(peaks_by_pseudobulk.obs['leiden_fine'].unique())} clusters")
print(f"  - top_celltype: {len(peaks_by_pseudobulk.obs['top_celltype'].unique())} celltypes")
print(f"  - top_timepoint: {len(peaks_by_pseudobulk.obs['top_timepoint'].unique())} timepoints")
print(f"  - accessibility_pattern: {peaks_by_pseudobulk.obs['accessibility_pattern'].value_counts().to_dict()}")
print(f"\nOutput files saved to: {output_dir}")
print(f"Figures saved to: {figure_path}")

