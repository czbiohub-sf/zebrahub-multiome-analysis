# %% [markdown]
# # Preprocess peak objects
# 
# This notebook preprocesses the peak objects for the EDA_peak_umap_cross_species notebook.
# sc_rapids jupyter kernel is used. (with GPU acceleration)
# 
# %% Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

import cupy as cp
import rapids_singlecell as rsc

# %% Dataset 1. Argelaguet 2022 mouse peak objects
# 1) file paths for Argelaguet 2022 mouse peak objects
peak_objects_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/PeakMatrix_anndata.h5ad"

# 2) load the peak objects
peak_objects = sc.read_h5ad(peak_objects_path)

# %% 
# 3) inspect the peak objects
print(peak_objects.shape)
print(peak_objects.var.shape)
print(peak_objects.obs.shape)

# 4) inspect the peak objects celltype and stage
print(peak_objects.obs["celltype.mapped"].value_counts())
print(peak_objects.obs["stage"].value_counts())

# 5) inspect the peak objects
print(peak_objects.var.head())
print(peak_objects.obs.head())

# %% pseudobulk the peak objects to create peaks-by-pseudobulk (celltype-AND-stage) matrix
# %% pseudobulk the peak objects to create peaks-by-pseudobulk (celltype-AND-stage) matrix
from Fig_peak_umap.scripts.analyze_peaks_with_normalization import analyze_peaks_with_normalization

# Run the pseudobulking
adata_pseudo = analyze_peaks_with_normalization(
    peak_objects,
    celltype_key='celltype.mapped',
    timepoint_key='stage'
)

print(adata_pseudo)
print(f"Shape: {adata_pseudo.shape}")
print(f"Common scale factor: {adata_pseudo.uns['common_scale_factor']}")

# %% Inspect the results
print(adata_pseudo.obs[['total_coverage', 'scale_factor', 'n_cells', 'mean_depth']].head(10))

# %% Optional: Transpose to get peaks-by-pseudobulk for UMAP analysis
# This is useful if you want to embed peaks instead of pseudobulk groups
adata_pseudo.X = adata_pseudo.layers["log_norm"]
peaks_by_pseudobulk = adata_pseudo.copy().T
print(f"Transposed shape (peaks-by-pseudobulk): {peaks_by_pseudobulk.shape}")


## save the adata objects
adata_pseudo.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/pb_by_celltype_stage_peaks.h5ad")
peaks_by_pseudobulk.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage.h5ad")

# %% load the adata objects
adata_pseudo = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/pb_by_celltype_stage_peaks.h5ad")
peaks_by_pseudobulk = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage.h5ad")

# %% computing UMAP
# moves `.X` to the GPU
rsc.get.anndata_to_GPU(peaks_by_pseudobulk)

# Compute UMAP
# rsc.pp.scale(peaks_pb_norm) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_by_pseudobulk, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_by_pseudobulk, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_by_pseudobulk, min_dist=0.2, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_by_pseudobulk, color="chr")

# %% 
# define the figure path
figure_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/figures/"
os.makedirs(figure_path, exist_ok=True)
sc.settings.figdir = figure_path
sc.pl.umap(peaks_by_pseudobulk, color="chr", save='_chr.png')
# %% 
# import the utility functions
import pyranges as pr
import sys
sys.path.append("../Fig_peak_umap/scripts/")
from peak_annotation_utils import annotate_peak_types
help(annotate_peak_types)

# %% Prepare peaks DataFrame for annotation
# Extract peak coordinates from the obs_names or existing columns
peaks_df = peaks_by_pseudobulk.obs[['chr', 'start', 'end']].copy()

# Rename columns to match the expected format
peaks_df = peaks_df.rename(columns={
    'chr': 'Chromosome',
    'start': 'Start',
    'end': 'End'
})

# Ensure proper data types
peaks_df['Chromosome'] = peaks_df['Chromosome'].astype(str)
peaks_df['Start'] = peaks_df['Start'].astype(int)
peaks_df['End'] = peaks_df['End'].astype(int)

# Check the reformatted data
print(peaks_df.head())
print(f"Shape: {peaks_df.shape}")
print(f"Data types:\n{peaks_df.dtypes}")

# %% Annotate peaks using mouse GTF file
# Path to mouse GTF file
mouse_gtf_file = '/hpc/reference/sequencing_alignment/alignment_references/mouse_gencode_M31_GRCm39_cellranger/genes/genes.gtf.gz'  # Update this path

# Annotate peaks (using Argelaguet 2022 definition: 500bp upstream, 100bp downstream)
annotated_peaks = annotate_peak_types(
    peaks_df, 
    mouse_gtf_file, 
    upstream_promoter=2000, 
    downstream_promoter=200
)

print("\nAnnotation complete!")
print(annotated_peaks.head())
print(f"\nPeak type distribution:")
print(annotated_peaks['peak_type'].value_counts())

# %% Add annotations back to the AnnData object
peaks_by_pseudobulk.obs['Chromosome'] = annotated_peaks['Chromosome'].values
peaks_by_pseudobulk.obs['Start'] = annotated_peaks['Start'].values
peaks_by_pseudobulk.obs['End'] = annotated_peaks['End'].values
peaks_by_pseudobulk.obs['peak_type'] = annotated_peaks['peak_type'].values

# Verify
print(peaks_by_pseudobulk.obs[['Chromosome', 'Start', 'End', 'peak_type']].head())

# %% 
sc.pl.umap(peaks_by_pseudobulk, color="peak_type", save='_peak_type.png')

# %% Reformat peak names to match expected format (chr-start-end instead of chr:start-end)
print("Reformatting peak names...")
# Create new index with chr-start-end format
new_index = [f"{row['chr']}-{row['start']}-{row['end']}" 
             for _, row in peaks_by_pseudobulk.obs.iterrows()]
peaks_by_pseudobulk.obs_names = new_index

print(f"Peak name format updated. Example: {peaks_by_pseudobulk.obs_names[0]}")

# %% Import the associate_peaks_to_genes function (with reload to get latest changes)
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
    mouse_gtf_file,
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
# Only show peaks with valid TSS distances
peaks_by_pseudobulk.obs['distance_to_tss_capped'] = peaks_by_pseudobulk.obs['distance_to_tss'].copy()

sc.pl.umap(peaks_by_pseudobulk, color='distance_to_tss', 
           title='Distance to nearest TSS (bp)',
           cmap='magma',  # Reverse so closer = darker
           vmin=0, vmax=10000,
           save='_distance_to_tss.png')

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
           save=figure_path + 'mouse_distance_to_tss_binned.png')

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
plt.savefig(figure_path + 'mouse_tss_distance_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_tss_distance_distributions.pdf', bbox_inches='tight')
plt.show()

# %% TSS distance statistics by peak type
print("\n=== TSS Distance by Peak Type ===")
tss_by_type = peaks_by_pseudobulk.obs.groupby('peak_type')['distance_to_tss'].describe()
print(tss_by_type)

# %% [markdown]
# ## Calculate total accessibility per peak and visualize on UMAP

# %% Calculate total accessibility (sum across all pseudobulk groups)
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

# %% Visualize log(total accessibility) on UMAP
# Set the figure path for scanpy
# figure_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/figures/"
# os.makedirs(figure_path, exist_ok=True)
# sc.settings.figdir = figure_path

# Note: scanpy's save parameter only accepts a suffix, not a full path
# It will save to sc.settings.figdir + save
sc.pl.umap(peaks_by_pseudobulk, color='log_total_accessibility', 
           title='Log(Total Accessibility) per peak',
           cmap='viridis',
           save='_mouse_log_total_accessibility.png')

# %% Also visualize with different color maps
sc.pl.umap(peaks_by_pseudobulk, color='log_total_accessibility', 
           title='Log(Total Accessibility) - magma colormap',
           cmap='magma', save='_mouse_log_total_accessibility_magma.png')

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
plt.savefig(figure_path + 'mouse_log_total_accessibility_distributions.png', dpi=300, bbox_inches='tight')
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
plt.savefig(figure_path + 'mouse_log_total_accessibility_distributions_density.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_log_total_accessibility_distributions_density.pdf', bbox_inches='tight')
plt.show()

# %% Print detailed summary statistics for each peak type
# print("\n=== Log(Total Accessibility) Statistics by Peak Type ===")
# for peak_type in sorted(peak_types):
#     peak_data = peaks_by_pseudobulk.obs[peaks_by_pseudobulk.obs['peak_type'] == peak_type]
#     print(f"\n{peak_type.upper()}:")
#     print(f"  N peaks: {len(peak_data):,}")
#     print(f"  Mean: {peak_data['log_total_accessibility'].mean():.3f}")
#     print(f"  Median: {peak_data['log_total_accessibility'].median():.3f}")
#     print(f"  Std: {peak_data['log_total_accessibility'].std():.3f}")
#     print(f"  Min: {peak_data['log_total_accessibility'].min():.3f}")
#     print(f"  Max: {peak_data['log_total_accessibility'].max():.3f}")

# %% Visualize peak types on UMAP
sc.pl.umap(peaks_by_pseudobulk, color="peak_type", 
           title='Mouse peaks colored by genomic annotation')

# %% [markdown]
# ## Highlight only promoter peaks on UMAP

# %% Create binary column for promoter vs non-promoter
peaks_by_pseudobulk.obs['is_promoter'] = peaks_by_pseudobulk.obs['peak_type'] == 'promoter'

# Visualize with scanpy (categorical)
sc.pl.umap(peaks_by_pseudobulk, color='is_promoter', 
           title='Promoter peaks highlighted',
           palette=['lightgray', 'red'],
           save='_promoters_highlighted.png')

# %% Create separate plots for each peak type
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

peak_types = ['promoter', 'exonic', 'intronic', 'intergenic']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # Red, Blue, Green, Purple

for idx, (peak_type, color) in enumerate(zip(peak_types, colors)):
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
plt.savefig(figure_path + 'mouse_peak_types_individual_highlighted.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_peak_types_individual_highlighted.pdf', bbox_inches='tight')
plt.show()

# %% Print summary statistics
print("\n=== Promoter Peak Summary ===")
print(f"Total peaks: {len(peaks_by_pseudobulk.obs):,}")
print(f"Promoter peaks: {promoter_mask.sum():,} ({(promoter_mask.sum()/len(peaks_by_pseudobulk.obs))*100:.2f}%)")
print(f"Non-promoter peaks: {(~promoter_mask).sum():,} ({((~promoter_mask).sum()/len(peaks_by_pseudobulk.obs))*100:.2f}%)")

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
ax1.set_title('Number of Peaks by Type (Mouse Argelaguet 2022)')
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
plt.show()

# %% Verify coordinate consistency
print("\nVerifying coordinate consistency:")
print("\nData types:")
print(f"  chr: {peaks_by_pseudobulk.obs['chr'].dtype}")
print(f"  Chromosome: {peaks_by_pseudobulk.obs['Chromosome'].dtype}")
print(f"  start: {peaks_by_pseudobulk.obs['start'].dtype}")
print(f"  Start: {peaks_by_pseudobulk.obs['Start'].dtype}")
print(f"  end: {peaks_by_pseudobulk.obs['end'].dtype}")
print(f"  End: {peaks_by_pseudobulk.obs['End'].dtype}")

print("\nFirst few values:")
print(peaks_by_pseudobulk.obs[['chr', 'Chromosome', 'start', 'Start', 'end', 'End']].head())

# Convert to same types for comparison
chr_match = (peaks_by_pseudobulk.obs['chr'].astype(str) == peaks_by_pseudobulk.obs['Chromosome'].astype(str))
start_match = (peaks_by_pseudobulk.obs['start'].astype(int) == peaks_by_pseudobulk.obs['Start'].astype(int))
end_match = (peaks_by_pseudobulk.obs['end'].astype(int) == peaks_by_pseudobulk.obs['End'].astype(int))

coord_match = (chr_match & start_match & end_match).all()

print(f"\nChromosome match: {chr_match.all()}")
print(f"Start match: {start_match.all()}")
print(f"End match: {end_match.all()}")
print(f"All coordinates match: {coord_match}")

# Show any mismatches
if not coord_match:
    mismatches = peaks_by_pseudobulk.obs[~(chr_match & start_match & end_match)]
    print(f"\nNumber of mismatches: {len(mismatches)}")
    if len(mismatches) > 0:
        print("\nFirst few mismatches:")
        print(mismatches[['chr', 'Chromosome', 'start', 'Start', 'end', 'End']].head())

# %% Summary of the annotated object
print("\n=== Summary of annotated peaks_by_pseudobulk ===")
print(f"Shape: {peaks_by_pseudobulk.shape}")
print(f"Total peaks (obs): {peaks_by_pseudobulk.n_obs}")
print(f"Total pseudobulk groups (vars): {peaks_by_pseudobulk.n_vars}")
print(f"\nAnnotations in .obs:")
print(peaks_by_pseudobulk.obs.columns.tolist())

# %% Optional: Save the annotated pseudobulked object
peaks_by_pseudobulk.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated.h5ad")
# adata_pseudo.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/pseudobulk_by_celltype_stage.h5ad")

# %% Part 2. Annotate peak UMAP with most accessible celltypes and timepoints
# Use functions from 09_annotate_peak_umap_celltype_timepoints.py to identify
# which celltypes and timepoints show highest accessibility for each cluster

# load the adata object (peaks_by_pseudobulk)
peaks_by_pseudobulk = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated.h5ad")

# %% Import necessary functions from the annotation script
import re
import sys
sys.path.append("../Fig_peak_umap/scripts/")
from peak_accessibility_utils import (
    create_cluster_celltype_profiles,
    create_cluster_timepoint_profiles,
    get_top_annotations
)

# %% Perform leiden clustering on the peak UMAP
print("Performing leiden clustering on the peak object...")

# Use the existing neighbors graph to compute leiden clustering
sc.tl.leiden(peaks_by_pseudobulk, resolution=0.4, key_added='leiden_coarse')
sc.tl.leiden(peaks_by_pseudobulk, resolution=1.0, key_added='leiden_fine')

# Visualize the clusters
sc.pl.umap(peaks_by_pseudobulk, color=['leiden_coarse', 'leiden_fine'], 
           legend_loc='on data', legend_fontsize=8)

print(f"Coarse clustering: {len(peaks_by_pseudobulk.obs['leiden_coarse'].unique())} clusters")
print(f"Fine clustering: {len(peaks_by_pseudobulk.obs['leiden_fine'].unique())} clusters")

# %% Parse celltype and timepoint information from pseudobulk group names
print("\nParsing celltype and timepoint information from pseudobulk groups...")

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

print("\nAnnotation complete!")
print(f"Added columns: top_celltype, top_timepoint, cluster_annotation")

# %% Visualize annotated UMAP
print("\n" + "="*60)
print("Creating annotated UMAP visualizations...")
print("="*60)

# Plot leiden clusters with top celltype annotation
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Plot 1: Leiden clusters
sc.pl.umap(peaks_by_pseudobulk, color='leiden_coarse', 
           ax=axes[0], show=False, title='Leiden Clusters (coarse)')

# Plot 2: Top celltype
sc.pl.umap(peaks_by_pseudobulk, color='top_celltype',
           ax=axes[1], show=False, title='Most Accessible Celltype')

# Plot 3: Top timepoint  
sc.pl.umap(peaks_by_pseudobulk, color='top_timepoint',
           ax=axes[2], show=False, title='Most Accessible Timepoint')

plt.tight_layout()
plt.savefig(figure_path + 'mouse_peak_umap_annotated_celltype_timepoint.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_peak_umap_annotated_celltype_timepoint.pdf', 
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
    
    # Sort timepoints naturally
    sorted_indices = sorted(range(len(timepoints)), 
                          key=lambda i: float(timepoints[i].replace('E', '').replace('somites', '')))
    timepoints_sorted = [timepoints[i] for i in sorted_indices]
    values_sorted = [timepoint_values[i] for i in sorted_indices]
    
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
        save_path=f"{figure_path}mouse_cluster_{cluster_id}_accessibility_profile.png"
    )

# %% Save the annotated object
print("\n" + "="*60)
print("Saving annotated peak object...")
print("="*60)

output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_with_clusters.h5ad"
peaks_by_pseudobulk.write_h5ad(output_path)
print(f"Saved to: {output_path}")

print("\n" + "="*60)
print("Annotation pipeline complete!")
print("="*60)