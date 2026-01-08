# %% [markdown]
# # Preprocess peak objects - Domcke 2020 Human Dataset
# 
# This notebook preprocesses the peak objects for the EDA_peak_umap_cross_species notebook.
# sc_rapids jupyter kernel is used. (with GPU acceleration)
# 
# %% Load the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

import cupy as cp
import rapids_singlecell as rsc

# %% Dataset 1. Domcke 2020 human peak objects
# 1) file paths for Domcke 2020 human peak objects
peak_objects_path = "/hpc/scratch/group.data.science/yang-joon.kim/domcke_shendure_2020/Domcke-2020.h5ad"

# 2) load the peak objects
peak_objects = sc.read_h5ad(peak_objects_path)

# %% 
# 3) inspect the peak objects
print(peak_objects.shape)
print(peak_objects.var.shape)
print(peak_objects.obs.shape)

# 4) inspect the peak objects celltype and stage
print(peak_objects.obs["cell_type"].value_counts())
print(peak_objects.obs["tissue"].value_counts())
print(peak_objects.obs["day_of_pregnancy"].value_counts())

# 5) inspect the peak objects
print(peak_objects.var.head())
print(peak_objects.obs.head())

# 6) add the UMAP coordinates to the peak_objects(cell UMAP)
peak_umap_coords = peak_objects.obs[['tissue_umap_1', 'tissue_umap_2']].values
peak_objects.obsm['X_umap'] = peak_umap_coords


# %% Clean up cell_type column
cell_type_mapping = {
    "ENS neurons?": "ENS neurons",
    "Inhibitory interneurons?": "Inhibitory neurons",
    "Megakaryocytes?": "Megakaryocytes",
    "Mesangial cells?": "Mesangial cells",
    "Skeletal muscle cells?": "Skeletal muscle cells",
    "Stromal cells?": "Stromal cells",
    "Vascular endothelial cells?": "Vascular endothelial cells",
    "Horizontal cells/Amacrine cells?": "Horizontal/Amacrine cells",
    "Syncytiotrophoblast and villous cytotrophoblasts?": "Syncytiotrophoblasts and villous cytotrophoblasts",
    "ELF3_AGBL2 positive cells?": "ELF3_AGBL2 positive cells",
    "Lymphoid and Myeloid cells": "Lymphoid/Myeloid cells",
    "Cardiomyocytes/Vascular endothelial cells": "Cardiomyocytes",
    "Cerebrum_Unknown.3": "Cerebrum_Unknown",
    "Eye_Unknown.6": "Eye_Unknown", 
    "Heart_Unknown.10": "Heart_Unknown",
    "Intestine_Unknown.4": "Intestine_Unknown_1",
    "Intestine_Unknown.8": "Intestine_Unknown_2",
    "Kidney_Unknown.7": "Kidney_Unknown_1",
    "Kidney_Unknown.14": "Kidney_Unknown_2",
    "Muscle_Unknown.7": "Muscle_Unknown",
    "Pancreas_Unknown.1": "Pancreas_Unknown",
}

peak_objects.obs['cell_type_clean'] = peak_objects.obs['cell_type'].replace(cell_type_mapping)

# Remove unknowns
unknown_mask = peak_objects.obs['cell_type_clean'].str.contains('Unknown', na=False)
peak_objects = peak_objects[~unknown_mask].copy()

# %% pseudobulk the peak objects to create peaks-by-pseudobulk (celltype-AND-stage) matrix
# Define the analyze_peaks_with_normalization function locally with fixes for this dataset
# (This version includes dtype conversion for numeric timepoint keys)
def analyze_peaks_with_normalization(
    adata, 
    celltype_key='annotation_ML_coarse', 
    timepoint_key='dev_stage'
):
    """
    1) Compute each cell's total_counts (sum of peaks/reads).
    2) For each (celltype, timepoint) group, compute the total_coverage 
       = sum of total_counts from all cells in that group.
    3) Create pseudobulk by summing (func='sum') each group's cells for the peaks matrix.
    4) The common_scale_factor = median of all group_total_coverage.
    5) For each group g, normalized_pseudobulk = raw_pseudobulk * (common_scale_factor / group_total_coverage[g]).

    Returns
    -------
    adata_pseudo : an AnnData with:
        - .X = raw pseudobulk counts
        - layers['normalized'] = scaled pseudobulk counts
        - obs['total_coverage'] = group's raw coverage
        - obs['scale_factor'] = how much that group's coverage was scaled
        - obs['n_cells'] and obs['mean_depth'] optionally stored as well
        - uns['common_scale_factor'] = the median coverage used for scaling
    """

    # 1) total_counts per cell
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1))

    # 2) total_coverage per group (sum of total_counts)
    group_total_coverage = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].sum()

    # 3) Pseudobulk by summing group cells
    ident_cols = [celltype_key, timepoint_key]
    adata_pseudo = sc.get.aggregate(adata, ident_cols, func='sum')
    # Copy the summed counts into .X
    adata_pseudo.X = adata_pseudo.layers["sum"].copy()

    # Split the new obs index (e.g. "Astro_dev_stage1") back into celltype/timepoint
    celltype_timepoint = pd.DataFrame({
        'celltype': ['_'.join(x.split('_')[:-1]) for x in adata_pseudo.obs.index],
        'timepoint': [x.split('_')[-1] for x in adata_pseudo.obs.index]
    }, index=adata_pseudo.obs.index)
    
    # Convert timepoint to match the original data type
    # Get the original dtype from the input adata
    original_timepoint_dtype = adata.obs[timepoint_key].dtype
    if pd.api.types.is_numeric_dtype(original_timepoint_dtype):
        # Convert to numeric if the original was numeric
        celltype_timepoint['timepoint'] = pd.to_numeric(celltype_timepoint['timepoint'])

    # Prepare for normalized counts
    X = adata_pseudo.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # 4) common_scale_factor = median of group_total_coverage
    common_scale_factor = np.mean(group_total_coverage.values)

    # 5) Rescale each group's pseudobulk
    normalized_counts = np.zeros_like(X)
    coverage_list = []
    scale_factor_list = []

    for i, idx in enumerate(adata_pseudo.obs.index):
        ct = celltype_timepoint.loc[idx, 'celltype']
        tp = celltype_timepoint.loc[idx, 'timepoint']
        coverage_g = group_total_coverage[(ct, tp)]

        # Scale factor = common_scale_factor / group's total coverage
        scale_g = common_scale_factor / coverage_g
        normalized_counts[i, :] = X[i, :] * scale_g
        
        coverage_list.append(coverage_g)
        scale_factor_list.append(scale_g)

    # Store normalized counts in a new layer
    adata_pseudo.layers['normalized'] = normalized_counts

    # Record coverage and scaling info in .obs
    adata_pseudo.obs['total_coverage'] = coverage_list
    adata_pseudo.obs['scale_factor'] = scale_factor_list

    # Optionally, also store #cells and mean_depth
    group_ncells = adata.obs.groupby([celltype_key, timepoint_key]).size()
    group_mean_depth = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].mean()

    n_cells_list = []
    mean_depth_list = []
    for idx in adata_pseudo.obs.index:
        ct = celltype_timepoint.loc[idx, 'celltype']
        tp = celltype_timepoint.loc[idx, 'timepoint']
        n_cells_list.append(group_ncells[(ct, tp)])
        mean_depth_list.append(group_mean_depth[(ct, tp)])
    adata_pseudo.obs['n_cells'] = n_cells_list
    adata_pseudo.obs['mean_depth'] = mean_depth_list

    # Save the "common" scale factor in .uns
    adata_pseudo.uns['common_scale_factor'] = common_scale_factor

    return adata_pseudo

# %%
# Run the pseudobulking
adata_pseudo = analyze_peaks_with_normalization(
    peak_objects,
    celltype_key='cell_type_clean',
    timepoint_key='day_of_pregnancy'
)

print(adata_pseudo)
print(f"Shape: {adata_pseudo.shape}")
print(f"Common scale factor: {adata_pseudo.uns['common_scale_factor']}")

# %% Inspect the results
print(adata_pseudo.obs[['total_coverage', 'scale_factor', 'n_cells', 'mean_depth']].head(5))

# %% Define the figure path
figure_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/human_domcke_2020/figures/"
os.makedirs(figure_path, exist_ok=True)
sc.settings.figdir = figure_path

# %% Transpose to create peaks-by-pseudobulk matrix
# Note: After pseudobulking, we have pseudobulk_groups x peaks
# We need to transpose to get peaks x pseudobulk_groups for peak-level analysis
peaks_by_pseudobulk = adata_pseudo.T.copy()
print(f"\nTransposed shape: {peaks_by_pseudobulk.shape}")
print(f"Now: peaks (obs) x pseudobulk_groups (vars)")

# %% Compute PCA once (will reuse for different UMAP parameters)
# move the data to the GPU
rsc.get.anndata_to_GPU(peaks_by_pseudobulk)
# compute the PCA
rsc.pp.pca(peaks_by_pseudobulk, n_comps=100, use_highly_variable=False)
print(f"PCA computed: {peaks_by_pseudobulk.obsm['X_pca'].shape}")

# %% [markdown]
# ## Parameter sweep for UMAP optimization

# %% Parameter sweep: test different n_neighbors and min_dist combinations
import copy

# Define parameter ranges
n_neighbors_range = [15, 30, 50, 100]
min_dist_range = [0.1, 0.2, 0.3, 0.5]

# Create figure for parameter sweep results
n_rows = len(min_dist_range)
n_cols = len(n_neighbors_range)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

print(f"Testing {n_rows * n_cols} parameter combinations...")
print("="*60)

# Store all UMAP embeddings
umap_results = {}

for i, min_dist in enumerate(min_dist_range):
    for j, n_neighbors in enumerate(n_neighbors_range):
        print(f"\nComputing UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        # Create a temporary copy to avoid modifying the original
        adata_temp = peaks_by_pseudobulk.copy()
        
        # Compute neighbors and UMAP with current parameters
        rsc.pp.neighbors(adata_temp, n_neighbors=n_neighbors, n_pcs=40, use_rep='X_pca')
        rsc.tl.umap(adata_temp, min_dist=min_dist, random_state=42)
        
        # Store results
        param_key = f"nn{n_neighbors}_md{min_dist}"
        umap_results[param_key] = adata_temp.obsm['X_umap'].copy()
        
        # Plot on the grid
        ax = axes[i, j] if n_rows > 1 else axes[j]
        umap_coords = adata_temp.obsm['X_umap']
        
        # Plot with gray points
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                   c='gray', s=0.5, alpha=0.5, rasterized=True)
        
        ax.set_title(f'n_neighbors={n_neighbors}\nmin_dist={min_dist}', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_aspect('equal')
        
        del adata_temp

plt.tight_layout()
plt.savefig(figure_path + 'human_umap_parameter_sweep.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'human_umap_parameter_sweep.pdf', bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Parameter sweep complete!")
print(f"Saved visualization to: {figure_path}human_umap_parameter_sweep.png")

# %% [markdown]
# ## Visualize parameter sweep colored by chromosome (to see structure)

# %% Create colored version of parameter sweep (colored by chromosome)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

print("\nCreating colored visualization of parameter sweep...")

for i, min_dist in enumerate(min_dist_range):
    for j, n_neighbors in enumerate(n_neighbors_range):
        param_key = f"nn{n_neighbors}_md{min_dist}"
        umap_coords = umap_results[param_key]
        
        # Plot on the grid
        ax = axes[i, j] if n_rows > 1 else axes[j]
        
        # Color by chromosome
        chr_colors = peaks_by_pseudobulk.obs['Chromosome'].astype('category').cat.codes
        scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                            c=chr_colors, s=0.5, alpha=0.6, 
                            cmap='tab20', rasterized=True)
        
        ax.set_title(f'n_neighbors={n_neighbors}\nmin_dist={min_dist}', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(figure_path + 'human_umap_parameter_sweep_colored.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'human_umap_parameter_sweep_colored.pdf', bbox_inches='tight')
plt.show()

print(f"Saved colored visualization to: {figure_path}human_umap_parameter_sweep_colored.png")

# %% Quantitative analysis: compute connectivity metrics for each parameter set
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import connected_components

# connectivity_metrics = {}

# print("\nComputing connectivity metrics for each parameter combination...")
# print("="*60)

# for param_key in umap_results.keys():
#     # Parse parameters from key
#     parts = param_key.split('_')
#     n_neighbors = int(parts[0].replace('nn', ''))
#     min_dist = float(parts[1].replace('md', ''))
    
#     # Create temporary adata to get the connectivity graph
#     adata_temp = peaks_by_pseudobulk.copy()
#     rsc.pp.neighbors(adata_temp, n_neighbors=n_neighbors, n_pcs=40, use_rep='X_pca')
    
#     # Get connectivity graph
#     connectivities = adata_temp.obsp['connectivities']
    
#     # Compute number of connected components
#     n_components, labels = connected_components(connectivities, directed=False)
    
#     # Compute average connectivity (mean number of edges per node)
#     avg_connectivity = connectivities.sum() / connectivities.shape[0]
    
#     # Store metrics
#     connectivity_metrics[param_key] = {
#         'n_neighbors': n_neighbors,
#         'min_dist': min_dist,
#         'n_components': n_components,
#         'avg_connectivity': avg_connectivity
#     }
    
#     print(f"{param_key}: {n_components} components, avg_connectivity={avg_connectivity:.2f}")
    
#     del adata_temp

# # Create summary table
# import pandas as pd
# metrics_df = pd.DataFrame(connectivity_metrics).T
# metrics_df = metrics_df.sort_values(['n_components', 'avg_connectivity'])

# print("\n" + "="*60)
# print("CONNECTIVITY METRICS SUMMARY")
# print("="*60)
# print("(Sorted by fewest components, then highest connectivity)")
# print(metrics_df)
# print("\nBest parameters (fewest disconnected components):")
# best_params = metrics_df.iloc[0]
# print(f"  n_neighbors = {int(best_params['n_neighbors'])}")
# print(f"  min_dist = {best_params['min_dist']}")
# print(f"  n_components = {int(best_params['n_components'])}")
# print(f"  avg_connectivity = {best_params['avg_connectivity']:.2f}")

# %% Choose optimal parameters and compute final UMAP
# Based on connectivity metrics and visual inspection
optimal_n_neighbors = 30  # Use best from metrics
optimal_min_dist = 0.5              # Use best from metrics

# Or manually override based on visual inspection:
# optimal_n_neighbors = 50  
# optimal_min_dist = 0.2

print(f"\nComputing final UMAP with optimal parameters:")
print(f"  n_neighbors = {optimal_n_neighbors}")
print(f"  min_dist = {optimal_min_dist}")

rsc.pp.neighbors(peaks_by_pseudobulk, n_neighbors=optimal_n_neighbors, n_pcs=40, use_rep='X_pca')
rsc.tl.umap(peaks_by_pseudobulk, min_dist=optimal_min_dist, random_state=42)

# plot the final UMAP
sc.pl.umap(peaks_by_pseudobulk, 
           title=f'Peak UMAP (n_neighbors={optimal_n_neighbors}, min_dist={optimal_min_dist})',
           save='_human_peaks_umap_basic.png')

# %% Save the adata_pseudo and peaks_by_pseudobulk objects
output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/human_domcke_2020/"
os.makedirs(output_dir, exist_ok=True)
#adata_pseudo.write_h5ad(output_dir + "Domcke-2020_pb_by_celltype_dayofpregnancy.h5ad")
#print(f"Saved pseudobulk object to: {output_dir}Domcke-2020_pb_by_celltype_dayofpregnancy.h5ad")



# %% Annotate the peaks_by_pseudobulk object
# %% 1) Extract chromosome info from peak names (format: chr1-752336-752980)
peak_names = peaks_by_pseudobulk.obs_names
parts = [x.split('-') for x in peak_names]
peaks_by_pseudobulk.obs['Chromosome'] = [p[0] for p in parts]
peaks_by_pseudobulk.obs['Start'] = [int(p[1]) for p in parts]
peaks_by_pseudobulk.obs['End'] = [int(p[2]) for p in parts]
# %% 2) Calculate total accessibility (sum across all pseudobulk groups)
# For peaks_by_pseudobulk: rows=peaks, columns=pseudobulk groups
# Sum across axis=1 to get total accessibility per peak
print("Calculating total accessibility per peak...")

# Get the data matrix
X = peaks_by_pseudobulk.layers["normalized"]
if hasattr(X, 'toarray'):
    # If sparse matrix, convert to dense (or use sparse operations)
    total_accessibility = np.array(X.sum(axis=1)).flatten()
else:
    # Already dense
    total_accessibility = X.sum(axis=1)

# Add to obs
peaks_by_pseudobulk.obs['total_accessibility'] = total_accessibility

# Calculate log-transformed accessibility (log1p to handle zeros)
peaks_by_pseudobulk.obs['log_total_accessibility'] = np.log1p(total_accessibility)

print(f"Total accessibility range: {total_accessibility.min():.2f} - {total_accessibility.max():.2f}")
print(f"Log total accessibility range: {peaks_by_pseudobulk.obs['log_total_accessibility'].min():.2f} - {peaks_by_pseudobulk.obs['log_total_accessibility'].max():.2f}")
# %% Import the utility functions
import pyranges as pr
import sys
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig_peak_umap/scripts/")
from peak_annotation_utils import annotate_peak_types
help(annotate_peak_types)

# %% Prepare peaks DataFrame for annotation
# Extract peak coordinates from the obs columns
peaks_df = peaks_by_pseudobulk.obs[['Chromosome', 'Start', 'End']].copy()

# Ensure proper data types
peaks_df['Chromosome'] = peaks_df['Chromosome'].astype(str)
peaks_df['Start'] = peaks_df['Start'].astype(int)
peaks_df['End'] = peaks_df['End'].astype(int)

# Check the reformatted data
print(peaks_df.head())
print(f"Shape: {peaks_df.shape}")
print(f"Data types:\n{peaks_df.dtypes}")

# %% save the annotated peaks_by_pseudobulk object
peaks_by_pseudobulk.write_h5ad(output_dir + "peaks_by_pb_annotated.h5ad")
print(f"Saved annotated peaks_by_pseudobulk object to: {output_dir}peaks_by_pb_annotated.h5ad")

# %% checkpoint 2.
output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/human_domcke_2020/"
peaks_by_pseudobulk = sc.read_h5ad(output_dir + "peaks_by_pb_annotated.h5ad")
peaks_by_pseudobulk
# %% Annotate peaks using human GTF file
# Path to human GTF file
human_gtf_file = '/hpc/reference/sequencing_alignment/alignment_references/GRCH38.gencode.v47.primary_assembly_Cellranger_20250321/genes/genes.gtf.gz'

# Annotate peaks (using Argelaguet 2022 definition: 500bp upstream, 200bp downstream)
annotated_peaks = annotate_peak_types(
    peaks_df, 
    human_gtf_file, 
    upstream_promoter=500, 
    downstream_promoter=200
)

print("\nAnnotation complete!")
print(annotated_peaks.head())
print(f"\nPeak type distribution:")
print(annotated_peaks['peak_type'].value_counts())

# %% Add annotations back to the AnnData object
peaks_by_pseudobulk.obs['peak_type'] = annotated_peaks['peak_type'].values

# Verify
print(peaks_by_pseudobulk.obs[['Chromosome', 'Start', 'End', 'peak_type']].head())

# %% Visualize peak types on UMAP
sc.pl.umap(peaks_by_pseudobulk, color="peak_type", 
           title='Human peaks colored by genomic annotation',
           save='_human_peak_type.png')

# %% checkpoint 1.
output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/human_domcke_2020/"
peaks_by_pseudobulk = sc.read_h5ad(output_dir + "peaks_by_pb_annotated.h5ad")
peaks_by_pseudobulk
# %% Import the associate_peaks_to_genes function
import importlib
import peak_annotation_utils
importlib.reload(peak_annotation_utils)
from peak_annotation_utils import associate_peaks_to_genes

# %% Associate peaks to nearest genes and calculate TSS distances
print("\n" + "="*50)
print("Starting full peak-to-gene association...")
print("="*50)

peaks_with_genes = associate_peaks_to_genes(
    peaks_by_pseudobulk,
    human_gtf_file,
    max_distance=50000,  # Cap at 50kb
    chunk_size=1000
)

print("\nGene association complete!")
print(peaks_with_genes.head())

# %% Add gene associations and TSS distances to AnnData object
peaks_by_pseudobulk.obs['gene_body_overlaps'] = peaks_with_genes['gene_body_overlaps'].values
peaks_by_pseudobulk.obs['nearest_gene'] = peaks_with_genes['nearest_gene'].values
peaks_by_pseudobulk.obs['distance_to_tss'] = peaks_with_genes['distance_to_tss'].values

# %% Summary statistics for TSS distances
print("\n=== TSS Distance Statistics ===")
print(f"Total peaks: {len(peaks_by_pseudobulk.obs):,}")
print(f"Peaks with nearest gene within 50kb: {peaks_by_pseudobulk.obs['nearest_gene'].notna().sum():,}")
print(f"Peaks with gene body overlaps: {(peaks_by_pseudobulk.obs['gene_body_overlaps'] != '').sum():,}")

# Distance statistics
valid_distances = peaks_by_pseudobulk.obs['distance_to_tss'].dropna()
if len(valid_distances) > 0:
    print(f"\nDistance to TSS statistics (for peaks with nearest gene):")
    print(f"  Min: {valid_distances.min():.0f} bp")
    print(f"  Max: {valid_distances.max():.0f} bp")
    print(f"  Mean: {valid_distances.mean():.0f} bp")
    print(f"  Median: {valid_distances.median():.0f} bp")

# %% Visualize distance to TSS on UMAP
sc.pl.umap(peaks_by_pseudobulk, color='distance_to_tss', 
           title='Distance to nearest TSS (bp)',
           cmap='magma',
           vmin=0, vmax=10000,
           save='_human_distance_to_tss.png')

# %% Create binned distance categories
bins = [0, 1000, 5000, 10000, 20000, 50000]
labels = ['0-1kb', '1-5kb', '5-10kb', '10-20kb', '20-50kb']

peaks_by_pseudobulk.obs['distance_to_tss_binned'] = pd.cut(
    peaks_by_pseudobulk.obs['distance_to_tss'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Plot binned distances
sc.pl.umap(peaks_by_pseudobulk, color='distance_to_tss_binned',
           title='Binned distance to nearest TSS',
           save='_human_distance_to_tss_binned.png')

# %% Distribution of TSS distances
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of distances
valid_distances = peaks_by_pseudobulk.obs['distance_to_tss'].dropna()
axes[0].hist(valid_distances, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Distance to TSS (bp)')
axes[0].set_ylabel('Number of peaks')
axes[0].set_title(f'Distribution of TSS distances (n={len(valid_distances):,})')
axes[0].axvline(valid_distances.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {valid_distances.median():.0f}bp')
axes[0].legend()
axes[0].grid(False)

# Box plot by peak type
peaks_with_dist = peaks_by_pseudobulk.obs[peaks_by_pseudobulk.obs['distance_to_tss'].notna()]
sns.boxplot(data=peaks_with_dist, x='peak_type', y='distance_to_tss', ax=axes[1])
axes[1].set_xlabel('Peak Type')
axes[1].set_ylabel('Distance to TSS (bp)')
axes[1].set_title('TSS Distance by Peak Type')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(False)

plt.tight_layout()
plt.savefig(figure_path + 'human_tss_distance_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'human_tss_distance_distributions.pdf', bbox_inches='tight')
plt.show()

# %% TSS distance statistics by peak type
print("\n=== TSS Distance by Peak Type ===")
tss_by_type = peaks_by_pseudobulk.obs.groupby('peak_type')['distance_to_tss'].describe()
print(tss_by_type)

# %% [markdown]
# ## Additional accessibility visualizations on UMAP

# %% Visualize log(total accessibility) on UMAP with different colormaps
sc.pl.umap(peaks_by_pseudobulk, color='log_total_accessibility', 
           title='Log(Total Accessibility) per peak',
           cmap='viridis',
           save='_human_log_total_accessibility.png')

# %% Also visualize with different color maps
sc.pl.umap(peaks_by_pseudobulk, color='log_total_accessibility', 
           title='Log(Total Accessibility) - magma colormap',
           cmap='magma', save='_human_log_total_accessibility_magma.png')

# %% Distribution of log total accessibility
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(peaks_by_pseudobulk.obs['log_total_accessibility'], bins=50, edgecolor='black')
axes[0].set_xlabel('Log(Total Accessibility)')
axes[0].set_ylabel('Number of peaks')
axes[0].set_title('Distribution of Log(Total Accessibility)')
axes[0].grid(False)

# Violin plot by peak type
sns.violinplot(data=peaks_by_pseudobulk.obs, x='peak_type', y='log_total_accessibility', ax=axes[1])
axes[1].set_xlabel('Peak Type')
axes[1].set_ylabel('Log(Total Accessibility)')
axes[1].set_title('Accessibility by Peak Type')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(False)

plt.tight_layout()
plt.savefig(figure_path + 'human_log_total_accessibility_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# %% Statistics of accessibility by peak type
print("\n=== Accessibility Statistics by Peak Type ===")
accessibility_by_type = peaks_by_pseudobulk.obs.groupby('peak_type')['log_total_accessibility'].describe()
print(accessibility_by_type)

# %% Overlaid histograms for all peak types
peak_types = peaks_by_pseudobulk.obs['peak_type'].unique()

# Create color palette
colors = {'promoter': '#e41a1c', 'exonic': '#377eb8', 'intronic': '#4daf4a', 'intergenic': '#984ea3'}

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each peak type on the same axes
for peak_type in sorted(peak_types):
    # Get data for this peak type
    peak_data = peaks_by_pseudobulk.obs[peaks_by_pseudobulk.obs['peak_type'] == peak_type]
    n_peaks = len(peak_data)
    
    # Create histogram
    ax.hist(peak_data['log_total_accessibility'], 
            bins=50, 
            alpha=0.5,
            color=colors.get(peak_type, 'gray'),
            label=f'{peak_type.capitalize()} (n={n_peaks:,})',
            edgecolor='black', density=True,
            linewidth=0.5)

ax.set_xlabel('Log(Total Accessibility)', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Distribution of Log(Total Accessibility) by Peak Type', 
            fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(False)

plt.tight_layout()
plt.savefig(figure_path + 'human_log_total_accessibility_distributions_density.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'human_log_total_accessibility_distributions_density.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Highlight peak types on UMAP

# %% Create binary column for promoter vs non-promoter
peaks_by_pseudobulk.obs['is_promoter'] = peaks_by_pseudobulk.obs['peak_type'] == 'promoter'

# Visualize with scanpy (categorical)
sc.pl.umap(peaks_by_pseudobulk, color='is_promoter', 
           title='Promoter peaks highlighted',
           palette=['lightgray', 'red'],
           save='_human_promoters_highlighted.png')

# %% Create separate plots for each peak type
# Get UMAP coordinates
umap_coords = peaks_by_pseudobulk.obsm['X_umap']

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

peak_types_list = ['promoter', 'exonic', 'intronic', 'intergenic']
colors_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # Red, Blue, Green, Purple

for idx, (peak_type, color) in enumerate(zip(peak_types_list, colors_list)):
    ax = axes[idx]
    
    # Create mask for current peak type
    mask = peaks_by_pseudobulk.obs['peak_type'] == peak_type
    
    # Plot all peaks in light gray (background)
    ax.scatter(umap_coords[:, 0], 
               umap_coords[:, 1],
               c='lightgray', 
               s=0.5, 
               alpha=0.2)
    
    # Plot current peak type in color (foreground)
    ax.scatter(umap_coords[mask, 0], 
               umap_coords[mask, 1],
               c=color, 
               s=1.5, 
               alpha=0.7)
    
    n_peaks = mask.sum()
    pct_peaks = (n_peaks / len(mask)) * 100
    ax.set_title(f'{peak_type.capitalize()} (n={n_peaks:,}, {pct_peaks:.1f}%)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(figure_path + 'human_peak_types_individual_highlighted.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'human_peak_types_individual_highlighted.pdf', bbox_inches='tight')
plt.show()

# %% Display detailed peak type statistics
print("\n=== Peak Type Statistics ===")
peak_counts = peaks_by_pseudobulk.obs['peak_type'].value_counts()
peak_props = peaks_by_pseudobulk.obs['peak_type'].value_counts(normalize=True) * 100

for peak_type in peak_counts.index:
    print(f"{peak_type}: {peak_counts[peak_type]:,} ({peak_props[peak_type]:.2f}%)")

# %% Visualize peak type distribution as bar plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot of counts
sns.barplot(x=peak_counts.index, y=peak_counts.values, ax=ax1)
ax1.set_title('Number of Peaks by Type (Human Domcke 2020)')
ax1.set_ylabel('Number of peaks')
ax1.set_xlabel('Peak Type')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(False)

# Bar plot of proportions
sns.barplot(x=peak_props.index, y=peak_props.values, ax=ax2)
ax2.set_title('Proportion of Peak Types')
ax2.set_ylabel('Percentage (%)')
ax2.set_xlabel('Peak Type')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(False)

plt.tight_layout()
plt.savefig(figure_path + 'human_peak_type_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'human_peak_type_distribution.pdf', bbox_inches='tight')
plt.show()

# %% Summary of the annotated object
print("\n=== Summary of annotated peaks_by_pseudobulk ===")
print(f"Shape: {peaks_by_pseudobulk.shape}")
print(f"Total peaks (obs): {peaks_by_pseudobulk.n_obs}")
print(f"Total pseudobulk groups (vars): {peaks_by_pseudobulk.n_vars}")
print(f"\nAnnotations in .obs:")
print(peaks_by_pseudobulk.obs.columns.tolist())

# %% Save intermediate annotated object
intermediate_output_path = output_dir + "peaks_by_pb_celltype_stage_annotated.h5ad"
peaks_by_pseudobulk.write_h5ad(intermediate_output_path)
print(f"\nSaved intermediate annotated object to: {intermediate_output_path}")

# %% [markdown]
# # Part 2: Annotate peak UMAP with most accessible celltypes and timepoints
# 
# Use functions to identify which celltypes and timepoints show highest accessibility for each cluster

# %% Import necessary functions from the annotation script
import re
import sys
import importlib
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

# %% Perform leiden clustering on the peak UMAP
print("\n" + "="*60)
print("Performing leiden clustering on the peak object...")
print("="*60)

# Use the existing neighbors graph to compute leiden clustering
sc.tl.leiden(peaks_by_pseudobulk, resolution=0.4, key_added='leiden_coarse')
sc.tl.leiden(peaks_by_pseudobulk, resolution=1.0, key_added='leiden_fine')

# Visualize the clusters
sc.pl.umap(peaks_by_pseudobulk, color=['leiden_coarse', 'leiden_fine'], 
           legend_loc='on data', legend_fontsize=8,
           save='_human_leiden_clusters.png')

print(f"Coarse clustering: {len(peaks_by_pseudobulk.obs['leiden_coarse'].unique())} clusters")
print(f"Fine clustering: {len(peaks_by_pseudobulk.obs['leiden_fine'].unique())} clusters")

# %% Parse celltype and timepoint information from pseudobulk group names
print("\n" + "="*60)
print("Parsing celltype and timepoint information from pseudobulk groups...")
print("="*60)

# The var_names should be in format: celltype_timepoint (e.g., "Astrocytes_52.0")
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
        celltype_list.append(var_name)
        timepoint_list.append('unknown')

peaks_by_pseudobulk.var['celltype'] = celltype_list
peaks_by_pseudobulk.var['timepoint'] = timepoint_list

print(f"Found {len(set(celltype_list))} unique celltypes")
print(f"Found {len(set(timepoint_list))} unique timepoints")
print(f"Celltypes: {sorted(set(celltype_list))}")
print(f"Timepoints: {sorted(set(timepoint_list))}")

# %% Compute profiles for both clustering resolutions
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

# %% Classify clusters by accessibility specificity
print("\n" + "="*60)
print("Classifying cluster specificity patterns...")
print("="*60)

# Classify coarse clusters
specificity_classifications = classify_cluster_specificity(
    cluster_celltype_profiles_coarse,
    entropy_threshold=0.75,          # High entropy = ubiquitous
    dominance_threshold=0.5,         # Top celltype has >50% of accessibility = specific
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

# %% Get top annotations for each cluster
print("\n" + "="*60)
print("Identifying top accessible celltypes and timepoints...")
print("="*60)

top_celltypes_coarse = get_top_annotations(cluster_celltype_profiles_coarse, top_n=3)
top_timepoints_coarse = get_top_annotations(cluster_timepoint_profiles_coarse, top_n=3)

# Print summary for coarse clustering
print("\n=== Coarse Clustering Summary ===")
for cluster_id in sorted(top_celltypes_coarse.keys()):
    print(f"\nCluster {cluster_id}:")
    print(f"  Top celltypes: {', '.join([f'{ct} ({acc:.2f})' for ct, acc in top_celltypes_coarse[cluster_id]])}")
    print(f"  Top timepoints: {', '.join([f'{tp} ({acc:.2f})' for tp, acc in top_timepoints_coarse[cluster_id]])}")

# %% Annotate peaks with their cluster's top celltype and timepoint
print("\n" + "="*60)
print("Annotating individual peaks with cluster assignments...")
print("="*60)

# For coarse clustering
peaks_by_pseudobulk.obs['top_celltype'] = peaks_by_pseudobulk.obs['leiden_coarse'].map(
    lambda x: top_celltypes_coarse[x][0][0] if x in top_celltypes_coarse else 'unknown'
)

peaks_by_pseudobulk.obs['top_timepoint'] = peaks_by_pseudobulk.obs['leiden_coarse'].map(
    lambda x: top_timepoints_coarse[x][0][0] if x in top_timepoints_coarse else 'unknown'
)

# Create combined annotation
peaks_by_pseudobulk.obs['cluster_annotation'] = (
    'C' + peaks_by_pseudobulk.obs['leiden_coarse'].astype(str) + 
    ': ' + peaks_by_pseudobulk.obs['top_celltype']
)

# Convert top_timepoint to numeric for continuous colormap visualization
peaks_by_pseudobulk.obs['top_timepoint_numeric'] = pd.to_numeric(
    peaks_by_pseudobulk.obs['top_timepoint'], 
    errors='coerce'
)

print("\nAnnotation complete!")
print(f"Added columns: top_celltype, top_timepoint, top_timepoint_numeric, cluster_annotation")
print(f"\nTimepoint range: {peaks_by_pseudobulk.obs['top_timepoint_numeric'].min():.1f} - {peaks_by_pseudobulk.obs['top_timepoint_numeric'].max():.1f} days")

# %% Visualize annotated UMAP

# Plot 1: Top celltype
sc.pl.umap(peaks_by_pseudobulk, color='top_celltype',
           title='Most Accessible Celltype', save='_most_access_ct.png')

# Plot 2: Top timepoint (continuous colormap)
sc.pl.umap(peaks_by_pseudobulk, color='top_timepoint_numeric',
           title='Most Accessible Timepoint (days of pregnancy)',
           cmap='viridis',
           save='_most_access_tp.png')


# %% Visualize annotated UMAP (3-panel figure)
print("\n" + "="*60)
print("Creating annotated UMAP visualizations...")
print("="*60)

# Create 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Get UMAP coordinates
umap_coords = peaks_by_pseudobulk.obsm['X_umap']

# Plot 1: Leiden clusters (categorical)
sc.pl.umap(peaks_by_pseudobulk, color='leiden_coarse', 
           ax=axes[0], show=False, title='Leiden Clusters (coarse)')

# Plot 2: Top celltype (categorical)
sc.pl.umap(peaks_by_pseudobulk, color='top_celltype',
           ax=axes[1], show=False, title='Most Accessible Celltype')

# Plot 3: Top timepoint (continuous with viridis)
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
cbar.set_label('Day of Pregnancy', rotation=270, labelpad=20)

axes[2].set_title('Most Accessible Timepoint\n(early = blue, late = yellow)', fontsize=12)
axes[2].set_xlabel('UMAP1')
axes[2].set_ylabel('UMAP2')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig(figure_path + 'human_peak_umap_annotated_most_accessible_ct_tp.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'human_peak_umap_annotated_most_accessible_ct_tp.pdf', 
            bbox_inches='tight')
plt.show()

# %% Create summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

print("\nCelltype distribution across clusters:")
print(peaks_by_pseudobulk.obs.groupby('leiden_coarse')['top_celltype'].value_counts())

print("\nTimepoint distribution across clusters:")
print(peaks_by_pseudobulk.obs.groupby('leiden_coarse')['top_timepoint'].value_counts())

print("\nOverall top celltype distribution:")
print(peaks_by_pseudobulk.obs['top_celltype'].value_counts())

print("\nOverall top timepoint distribution:")
print(peaks_by_pseudobulk.obs['top_timepoint'].value_counts())

# %% Visualize accessibility specificity patterns on UMAP
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
plt.savefig(figure_path + 'human_peak_umap_specificity_patterns.png',
            dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'human_peak_umap_specificity_patterns.pdf',
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

# %% Plot accessibility profiles for selected clusters
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
    
    # Sort timepoints by numeric value
    try:
        sorted_indices = sorted(range(len(timepoints)), 
                              key=lambda i: float(timepoints[i]))
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

# %% Plot profiles for a few example clusters
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
        save_path=f"{figure_path}human_cluster_{cluster_id}_accessibility_profile.png"
    )

# %% Save the final annotated object with cluster annotations
print("\n" + "="*60")
print("Saving final annotated peak object...")
print("="*60")

final_output_path = output_dir + "peaks_by_pb_celltype_stage_annotated_with_clusters.h5ad"
peaks_by_pseudobulk.write_h5ad(final_output_path)
print(f"Saved to: {final_output_path}")

print("\n" + "="*60)
print("Annotation pipeline complete!")
print("="*60)
