# %% [markdown]
# # Preprocess peak objects
# 
# This notebook preprocesses the peak objects for the EDA_peak_umap_cross_species notebook.
# sc_rapids jupyter kernel is used. (with GPU acceleration)
# 
# %% Load the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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
import sys
sys.path.append("../")
from Fig_peak_umap.utils.utils_pseudobulk import analyze_peaks_with_normalization

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
adata_pseudo.X = adata_pseudo.layers["normalized"]
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
#rsc.get.anndata_to_GPU(peaks_by_pseudobulk)

# Compute UMAP
# rsc.pp.scale(peaks_pb_norm) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_by_pseudobulk, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_by_pseudobulk, n_neighbors=50, n_pcs=40)
rsc.tl.umap(peaks_by_pseudobulk, min_dist=0.4, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_by_pseudobulk, color="chr")

# %% 
# define the figure path
figure_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/figures_v2/"
os.makedirs(figure_path, exist_ok=True)
sc.settings.figdir = figure_path
sc.pl.umap(peaks_by_pseudobulk, color="chr", save='_chromosome.png')
# %% 
# import the utility functions
import pyranges as pr
import sys
sys.path.append("../")
from Fig_peak_umap.scripts.peak_annotation_utils import annotate_peak_types
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
from Fig_peak_umap.scripts import peak_annotation_utils
importlib.reload(peak_annotation_utils)
from Fig_peak_umap.scripts.peak_annotation_utils import associate_peaks_to_genes

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
           title='Log(Total Accessibility) - cividis colormap',
           cmap='cividis', save='_mouse_log_total_accessibility_cividis.png')

# %% Export just the colorbar as a standalone PDF
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# Get the data range for the colorbar
vmin = peaks_by_pseudobulk.obs['log_total_accessibility'].min()
vmax = peaks_by_pseudobulk.obs['log_total_accessibility'].max()

# Create a figure with just the colorbar
fig, ax = plt.subplots(figsize=(1.5, 6))
norm = Normalize(vmin=vmin, vmax=vmax)
cbar = ColorbarBase(ax, cmap=cm.cividis, norm=norm, orientation='vertical')
cbar.set_label('Log(Total Accessibility)', fontsize=12)

# Save as PDF
plt.savefig(figure_path + 'mouse_log_total_accessibility_cividis_colorbar.pdf', 
            bbox_inches='tight', dpi=300)
plt.show()
print(f"Colorbar saved to: {figure_path}mouse_log_total_accessibility_cividis_colorbar.pdf")

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
peaks_by_pseudobulk.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad")
# adata_pseudo.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/pseudobulk_by_celltype_stage.h5ad")

# %% Part 2. Annotate peak UMAP with most accessible celltypes and timepoints
# Use functions from 09_annotate_peak_umap_celltype_timepoints.py to identify
# which celltypes and timepoints show highest accessibility for each cluster

# load the adata object (peaks_by_pseudobulk)
peaks_by_pseudobulk = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad")
peaks_by_pseudobulk
# %% Import necessary functions from the annotation script
import re
import sys
import importlib
sys.path.append("../")

# Import and reload to get latest version
from Fig_peak_umap.scripts import peak_accessibility_utils
importlib.reload(peak_accessibility_utils)
from Fig_peak_umap.scripts.peak_accessibility_utils import (
    create_cluster_celltype_profiles,
    create_cluster_timepoint_profiles,
    get_top_annotations,
    classify_cluster_specificity,
    add_specificity_to_adata
)

# %% Perform leiden clustering on the peak UMAP
print("Performing leiden clustering on the peak object...")

# Use the existing neighbors graph to compute leiden clustering
sc.tl.leiden(peaks_by_pseudobulk, resolution=0.4, key_added='leiden_coarse')
#sc.tl.leiden(peaks_by_pseudobulk, resolution=1.0, key_added='leiden_fine')

# Visualize the clusters
sc.pl.umap(peaks_by_pseudobulk, color=['leiden_coarse'], 
           legend_loc='on data', legend_fontsize=8)

print(f"Coarse clustering: {len(peaks_by_pseudobulk.obs['leiden_coarse'].unique())} clusters")
#print(f"Fine clustering: {len(peaks_by_pseudobulk.obs['leiden_fine'].unique())} clusters")




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

# cluster_celltype_profiles_fine = create_cluster_celltype_profiles(
#     peaks_by_pseudobulk,
#     cluster_col='leiden_fine',
#     verbose=True
# )

print("\n" + "="*60)
print("Computing timepoint accessibility profiles...")
print("="*60)

cluster_timepoint_profiles_coarse = create_cluster_timepoint_profiles(
    peaks_by_pseudobulk,
    cluster_col='leiden_coarse',
    verbose=True
)

# cluster_timepoint_profiles_fine = create_cluster_timepoint_profiles(
#     peaks_by_pseudobulk,
#     cluster_col='leiden_fine',
#     verbose=True
# )

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

print("\nAnnotation complete!")
print(f"Added columns: top_celltype, top_timepoint, cluster_annotation")

# %% Compute peak contrast for celltype and timepoint accessibility
print("\n" + "="*60)
print("Computing peak contrast metrics...")
print("="*60)

def compute_peak_contrast(accessibility_vector):
    """
    Compute peak contrast: (max - mean_of_others) / std_of_others
    
    This measures how strongly a peak stands out from the background.
    High values = specific signal, Low values = ubiquitous/weak signal
    """
    if len(accessibility_vector) < 2:
        return np.nan
    
    max_val = np.max(accessibility_vector)
    max_idx = np.argmax(accessibility_vector)
    
    # Get all values except the maximum
    other_values = np.delete(accessibility_vector, max_idx)
    
    if len(other_values) == 0:
        return np.nan
    
    mean_others = np.mean(other_values)
    std_others = np.std(other_values)
    
    # Avoid division by zero
    if std_others == 0:
        # If all other values are the same, return a large value if max differs
        if max_val > mean_others:
            return 100.0  # Arbitrarily large
        else:
            return 0.0
    
    contrast = (max_val - mean_others) / std_others
    return contrast

# Compute celltype contrast and most accessible celltype for each peak
print("\nComputing celltype peak contrast and most accessible celltype...")
celltype_contrast_list = []
top_celltype_list = []
top_celltype_accessibility_list = []

for peak_idx in range(peaks_by_pseudobulk.n_obs):
    # Get accessibility across all pseudobulk groups for this peak
    peak_accessibility = peaks_by_pseudobulk.layers['normalized'][peak_idx, :].toarray().flatten() if hasattr(peaks_by_pseudobulk.layers['normalized'], 'toarray') else peaks_by_pseudobulk.layers['normalized'][peak_idx, :]
    
    # Group by celltype and compute mean accessibility per celltype
    celltype_accessibility = {}
    for i, pb_group in enumerate(peaks_by_pseudobulk.var_names):
        celltype = peaks_by_pseudobulk.var.loc[pb_group, 'celltype']
        if celltype not in celltype_accessibility:
            celltype_accessibility[celltype] = []
        celltype_accessibility[celltype].append(peak_accessibility[i])
    
    # Average across timepoints for each celltype
    celltype_means = {ct: np.mean(vals) for ct, vals in celltype_accessibility.items()}
    celltype_means_array = np.array(list(celltype_means.values()))
    celltype_names = list(celltype_means.keys())
    
    # Find most accessible celltype
    max_idx = np.argmax(celltype_means_array)
    top_celltype = celltype_names[max_idx]
    top_accessibility = celltype_means_array[max_idx]
    
    # Compute contrast
    contrast = compute_peak_contrast(celltype_means_array)
    
    celltype_contrast_list.append(contrast)
    top_celltype_list.append(top_celltype)
    top_celltype_accessibility_list.append(top_accessibility)

peaks_by_pseudobulk.obs['celltype_peak_contrast'] = celltype_contrast_list
peaks_by_pseudobulk.obs['peak_top_celltype'] = top_celltype_list
peaks_by_pseudobulk.obs['peak_top_celltype_accessibility'] = top_celltype_accessibility_list

# Compute timepoint contrast and most accessible timepoint for each peak
print("Computing timepoint peak contrast and most accessible timepoint...")
timepoint_contrast_list = []
top_timepoint_list = []
top_timepoint_accessibility_list = []

for peak_idx in range(peaks_by_pseudobulk.n_obs):
    # Get accessibility across all pseudobulk groups for this peak
    peak_accessibility = peaks_by_pseudobulk.layers['normalized'][peak_idx, :].toarray().flatten() if hasattr(peaks_by_pseudobulk.layers['normalized'], 'toarray') else peaks_by_pseudobulk.layers['normalized'][peak_idx, :]
    
    # Group by timepoint and compute mean accessibility per timepoint
    timepoint_accessibility = {}
    for i, pb_group in enumerate(peaks_by_pseudobulk.var_names):
        timepoint = peaks_by_pseudobulk.var.loc[pb_group, 'timepoint']
        if timepoint not in timepoint_accessibility:
            timepoint_accessibility[timepoint] = []
        timepoint_accessibility[timepoint].append(peak_accessibility[i])
    
    # Average across celltypes for each timepoint
    timepoint_means = {tp: np.mean(vals) for tp, vals in timepoint_accessibility.items()}
    timepoint_means_array = np.array(list(timepoint_means.values()))
    timepoint_names = list(timepoint_means.keys())
    
    # Find most accessible timepoint
    max_idx = np.argmax(timepoint_means_array)
    top_timepoint = timepoint_names[max_idx]
    top_accessibility = timepoint_means_array[max_idx]
    
    # Compute contrast
    contrast = compute_peak_contrast(timepoint_means_array)
    
    timepoint_contrast_list.append(contrast)
    top_timepoint_list.append(top_timepoint)
    top_timepoint_accessibility_list.append(top_accessibility)

peaks_by_pseudobulk.obs['timepoint_peak_contrast'] = timepoint_contrast_list
peaks_by_pseudobulk.obs['peak_top_timepoint'] = top_timepoint_list
peaks_by_pseudobulk.obs['peak_top_timepoint_accessibility'] = top_timepoint_accessibility_list

print("\nPeak-level analysis complete!")
print(f"Celltype contrast range: {np.nanmin(celltype_contrast_list):.2f} - {np.nanmax(celltype_contrast_list):.2f}")
print(f"Timepoint contrast range: {np.nanmin(timepoint_contrast_list):.2f} - {np.nanmax(timepoint_contrast_list):.2f}")
print(f"Mean celltype contrast: {np.nanmean(celltype_contrast_list):.2f}")
print(f"Mean timepoint contrast: {np.nanmean(timepoint_contrast_list):.2f}")

print(f"\nTop celltypes per peak:")
print(pd.Series(top_celltype_list).value_counts().head(10))
print(f"\nTop timepoints per peak:")
print(pd.Series(top_timepoint_list).value_counts().head(10))

# %% Convert timepoints to numeric values for continuous colormap visualization
def timepoint_to_numeric(timepoint_str):
    """Convert timepoint strings like 'E7.5' to numeric values like 7.5"""
    if timepoint_str == 'unknown' or pd.isna(timepoint_str):
        return np.nan
    # Remove 'E' prefix and convert to float
    try:
        return float(str(timepoint_str).replace('E', ''))
    except:
        return np.nan

# Convert categorical to string first, then apply the numeric conversion
# Convert both cluster-based and peak-based timepoint annotations
peaks_by_pseudobulk.obs['top_timepoint_numeric'] = peaks_by_pseudobulk.obs['top_timepoint'].astype(str).apply(timepoint_to_numeric)
peaks_by_pseudobulk.obs['peak_top_timepoint_numeric'] = peaks_by_pseudobulk.obs['peak_top_timepoint'].astype(str).apply(timepoint_to_numeric)

print("\nTimepoint conversion complete!")
valid_timepoints = peaks_by_pseudobulk.obs['top_timepoint_numeric'].dropna()
print(f"Cluster-based numeric timepoint range: {valid_timepoints.min():.1f} - {valid_timepoints.max():.1f}")
print(f"Unique timepoints: {sorted(valid_timepoints.unique())}")

valid_peak_timepoints = peaks_by_pseudobulk.obs['peak_top_timepoint_numeric'].dropna()
print(f"\nPeak-based numeric timepoint range: {valid_peak_timepoints.min():.1f} - {valid_peak_timepoints.max():.1f}")
print(f"Unique timepoints: {sorted(valid_peak_timepoints.unique())}")

# %% Save the annotated object
peaks_by_pseudobulk.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad")

# %% create individual plots for the UMAP
# Plot 1: Leiden clusters
# sc.pl.umap(peaks_by_pseudobulk, color='leiden_coarse',
#         title='Leiden Clusters (coarse)', save='_leiden_clusters_coarse.png')

def normalize_for_alpha_robust(values, min_alpha=0.1, max_alpha=0.9):
    """Normalize values to alpha range using robust percentile clipping."""
    min_val = np.percentile(values, 5)
    max_val = np.percentile(values, 95)
    clipped = np.clip(values, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    return normalized * (max_alpha - min_alpha) + min_alpha

# Compute alpha values from contrast
peaks_by_pseudobulk.obs['alpha_celltype'] = normalize_for_alpha_robust(
    peaks_by_pseudobulk.obs['celltype_peak_contrast']
)
peaks_by_pseudobulk.obs['alpha_timepoint'] = normalize_for_alpha_robust(
    peaks_by_pseudobulk.obs['timepoint_peak_contrast']
)

# Plot: Top celltype with alpha = specificity
sc.pl.umap(
    peaks_by_pseudobulk, 
    color='top_celltype',
    alpha=peaks_by_pseudobulk.obs['alpha_celltype'].values,
    title='Top Celltype (α = specificity)',
    frameon=False,
    save='_top_celltype_alpha_contrast.png'
)

# Plot: Top timepoint with alpha = specificity  
sc.pl.umap(
    peaks_by_pseudobulk,
    color='top_timepoint', 
    alpha=peaks_by_pseudobulk.obs['alpha_timepoint'].values,
    title='Top Timepoint (α = specificity)',
    frameon=False,
    save='_top_timepoint_alpha_contrast.png'
)

# %% mapping cell types to broader tissues (or lineages)
mouse_celltype_to_lineage = {
    # ECTODERM / CNS
    "Ectoderm": [
        "Epiblast",
        "Rostral_neurectoderm",
        "Caudal_neurectoderm", 
        "Forebrain_Midbrain_Hindbrain",
        "Spinal_cord",
        "Neural_crest",
        "Surface_ectoderm",
    ],
    
    # EXTRAEMBRYONIC
    "Extraembryonic": [
        "ExE_ectoderm",
        "ExE_endoderm", 
        "ExE_mesoderm",
        "Visceral_endoderm",
        "Parietal_endoderm",
    ],
    
    # ENDODERM
    "Endoderm": [
        "Def._endoderm",
        "Gut",
    ],
    
    # PARAXIAL MESODERM
    "Paraxial Mesoderm": [
        "Paraxial_mesoderm",
        "Somitic_mesoderm",
        "NMP",  # Neuromesodermal progenitors
        "Caudal_Mesoderm",
        "Caudal_epiblast",
    ],
    
    # LATERAL PLATE / CARDIAC MESODERM
    "Lateral Mesoderm": [
        "Mesenchyme",
        "Mixed_mesoderm",
        "Intermediate_mesoderm",
        "Cardiomyocytes",
        "Pharyngeal_mesoderm",
        "Allantois",
    ],
    
    # HEMATOPOIETIC / BLOOD
    "Hematopoietic": [
        "Blood_progenitors_1",
        "Blood_progenitors_2",
        "Erythroid1",
        "Erythroid2",
        "Erythroid3",
        "Haematoendothelial_progenitors",
        "Endothelium",
    ],
    
    # PRIMITIVE STREAK / EARLY MESODERM
    "Primitive Streak": [
        "Primitive_Streak",
        "Anterior_Primitive_Streak",
        "Nascent_mesoderm",
        "Notochord",
    ],
    
    # GERMLINE
    "Germline": [
        "PGC",
    ],
}

# %% map the celltype to the lineage
import numpy as np
import re
import scanpy as sc

# Step 1: Strip the stage suffix (E7.5, E8.0, etc.) to get base celltype
def strip_stage_suffix(celltype_stage):
    """Remove _E7.5, _E8.0, _E8.75 etc. from celltype names."""
    # Match pattern: _E followed by numbers and optional decimal
    return re.sub(r'_E\d+\.?\d*$', '', str(celltype_stage))

# Apply to create clean celltype column
peaks_by_pseudobulk.obs['celltype_clean'] = peaks_by_pseudobulk.obs['top_celltype'].apply(strip_stage_suffix)

# Verify the cleaning worked
print("Original celltypes (sample):")
print(peaks_by_pseudobulk.obs['top_celltype'].value_counts().head(10))
print("\nCleaned celltypes:")
print(peaks_by_pseudobulk.obs['celltype_clean'].value_counts())

# Step 2: Map to lineages using the dictionary
mouse_celltype_to_lineage = {
    "Ectoderm": [
        "Epiblast",
        "Rostral_neurectoderm",
        "Caudal_neurectoderm", 
        "Forebrain_Midbrain_Hindbrain",
        "Spinal_cord",
        "Neural_crest",
        "Surface_ectoderm",
    ],
    "Extraembryonic": [
        "ExE_ectoderm",
        "ExE_endoderm", 
        "ExE_mesoderm",
        "Visceral_endoderm",
        "Parietal_endoderm",
    ],
    "Endoderm": [
        "Def._endoderm",
        "Gut",
    ],
    "Paraxial Mesoderm": [
        "Paraxial_mesoderm",
        "Somitic_mesoderm",
        "NMP",
        "Caudal_Mesoderm",
        "Caudal_epiblast",
    ],
    "Lateral Mesoderm": [
        "Mesenchyme",
        "Mixed_mesoderm",
        "Intermediate_mesoderm",
        "Cardiomyocytes",
        "Pharyngeal_mesoderm",
        "Allantois",
    ],
    "Hematopoietic": [
        "Blood_progenitors_1",
        "Blood_progenitors_2",
        "Erythroid1",
        "Erythroid2",
        "Erythroid3",
        "Haematoendothelial_progenitors",
        "Endothelium",
    ],
    "Primitive Streak": [
        "Primitive_Streak",
        "Anterior_Primitive_Streak",
        "Nascent_mesoderm",
        "Notochord",
    ],
    "Germline": [
        "PGC",
    ],
}

# Create reverse mapping
celltype_to_lineage_map = {}
for lineage, celltypes in mouse_celltype_to_lineage.items():
    for celltype in celltypes:
        celltype_to_lineage_map[celltype] = lineage

# Step 3: Map cleaned celltypes to lineages
peaks_by_pseudobulk.obs['lineage'] = peaks_by_pseudobulk.obs['celltype_clean'].map(celltype_to_lineage_map)

# Check for unmapped celltypes
unmapped = peaks_by_pseudobulk.obs[peaks_by_pseudobulk.obs['lineage'].isna()]['celltype_clean'].unique()
if len(unmapped) > 0:
    print(f"\nWarning: {len(unmapped)} unmapped celltypes: {list(unmapped)}")
    peaks_by_pseudobulk.obs['lineage'] = peaks_by_pseudobulk.obs['lineage'].fillna('Unknown')

print(f"\nLineage distribution:")
print(peaks_by_pseudobulk.obs['lineage'].value_counts())

# Step 4: Plot
mouse_lineage_colors = {
    'Ectoderm': '#DAA520',
    'Endoderm': '#6A5ACD',
    'Paraxial Mesoderm': '#4169E1',
    'Lateral Mesoderm': '#228B22',
    'Hematopoietic': '#DC143C',
    'Primitive Streak': '#FF8C00',
    'Extraembryonic': '#808080',
    'Germline': '#DA70D6',
    'Unknown': '#D3D3D3',
}

sc.pl.umap(
    peaks_by_pseudobulk,
    color='lineage',
    palette=mouse_lineage_colors,
    title='Mouse Peak UMAP - Lineage',
    frameon=False,
    save='_mouse_lineage.png'
)
  
# %% Plot PEAK-LEVEL most accessible celltype and timepoint
print("\nPlotting peak-level most accessible annotations...")

# Plot 4: Peak-level top celltype
sc.pl.umap(peaks_by_pseudobulk, color='peak_top_celltype',
           title='Most Accessible Celltype (Per Peak)',
           save='_peak_top_celltype.png')

# Plot 5: Peak-level top timepoint (numeric with viridis colormap)
sc.pl.umap(peaks_by_pseudobulk, color='peak_top_timepoint_numeric',
           title='Most Accessible Timepoint (Per Peak)',
           cmap='viridis',
           save='_peak_top_timepoint_numeric.png')


# %% 
# Step 1: Check the structure of your data
print(f"peaks_by_pseudobulk shape: {peaks_by_pseudobulk.shape}")
print(f"var (pseudobulk groups) sample: {peaks_by_pseudobulk.var_names[:10].tolist()}")

# Step 2: Parse pseudobulk column names to get celltype and stage
import re

def parse_celltype_stage(name):
    """Extract celltype and stage from 'Celltype_E8.0' format."""
    match = re.match(r'^(.+)_(E\d+\.?\d*)$', name)
    if match:
        return match.group(1), match.group(2)
    return name, None

# Create mappings
celltype_mapping = {}
stage_mapping = {}

for col in peaks_by_pseudobulk.var_names:
    celltype, stage = parse_celltype_stage(col)
    celltype_mapping[col] = celltype
    stage_mapping[col] = stage

unique_celltypes = sorted(set(celltype_mapping.values()))
unique_stages = sorted(set(s for s in stage_mapping.values() if s is not None))

print(f"\nFound {len(unique_celltypes)} unique celltypes")
print(f"Found {len(unique_stages)} unique stages: {unique_stages}")

# Step 3: Compute mean accessibility per celltype (averaging across stages)
# Get the dense matrix (if sparse)
X = peaks_by_pseudobulk.X
if hasattr(X, 'toarray'):
    X = X.toarray()

for celltype in unique_celltypes:
    # Get columns for this celltype (all stages)
    celltype_cols = [col for col, ct in celltype_mapping.items() if ct == celltype]
    col_indices = [peaks_by_pseudobulk.var_names.get_loc(col) for col in celltype_cols]
    
    if col_indices:
        # Mean accessibility across stages
        mean_accessibility = np.mean(X[:, col_indices], axis=1)
        peaks_by_pseudobulk.obs[f'accessibility_{celltype}'] = mean_accessibility

print(f"\nCreated {len(unique_celltypes)} accessibility columns")

# Step 4: Compute mean accessibility per lineage (averaging across celltypes)
lineage_accessibility = {}

for lineage, celltypes_in_lineage in mouse_celltype_to_lineage.items():
    lineage_cols = [f'accessibility_{ct}' for ct in celltypes_in_lineage 
                    if f'accessibility_{ct}' in peaks_by_pseudobulk.obs.columns]
    
    if lineage_cols:
        lineage_vals = peaks_by_pseudobulk.obs[lineage_cols].values
        peaks_by_pseudobulk.obs[f'accessibility_lineage_{lineage}'] = np.mean(lineage_vals, axis=1)
        lineage_accessibility[lineage] = True
        print(f"Created accessibility_lineage_{lineage} from {len(lineage_cols)} celltypes")

# Step 5: Compute top lineage and lineage contrast
lineage_acc_cols = [f'accessibility_lineage_{lin}' for lin in lineage_accessibility.keys()]
lineage_vals = peaks_by_pseudobulk.obs[lineage_acc_cols].values
lineage_names = list(lineage_accessibility.keys())

# Find max lineage for each peak
max_lineage_idx = np.argmax(lineage_vals, axis=1)
peaks_by_pseudobulk.obs['top_lineage'] = [lineage_names[i] for i in max_lineage_idx]

# Compute contrast
max_vals = np.max(lineage_vals, axis=1)
other_vals_mean = np.array([
    np.mean(np.delete(row, max_idx)) 
    for row, max_idx in zip(lineage_vals, max_lineage_idx)
])
other_vals_std = np.array([
    np.std(np.delete(row, max_idx)) 
    for row, max_idx in zip(lineage_vals, max_lineage_idx)
])

peaks_by_pseudobulk.obs['lineage_contrast'] = np.where(
    other_vals_std > 1e-10,
    (max_vals - other_vals_mean) / other_vals_std,
    0
)

print(f"\nLineage contrast stats: mean={peaks_by_pseudobulk.obs['lineage_contrast'].mean():.2f}, "
      f"std={peaks_by_pseudobulk.obs['lineage_contrast'].std():.2f}")

# Step 6: Compute alpha and plot
peaks_by_pseudobulk.obs['alpha_lineage'] = normalize_for_alpha_robust(
    peaks_by_pseudobulk.obs['lineage_contrast']
)

sc.pl.umap(
    peaks_by_pseudobulk,
    color='top_lineage',
    palette=mouse_lineage_colors,
    alpha=peaks_by_pseudobulk.obs['alpha_lineage'].values,
    title='Mouse Peak UMAP - Lineage (α = specificity)',
    frameon=False,
    save='_mouse_lineage_alpha_contrast.png'
)

print(f"\nTop lineage distribution:")
print(peaks_by_pseudobulk.obs['top_lineage'].value_counts())

# %% repeat this for the timepoints

# %% Export viridis colorbar for timepoint as standalone PDF
# Get the data range for the timepoint colorbar
vmin_tp = peaks_by_pseudobulk.obs['top_timepoint_numeric'].min()
vmax_tp = peaks_by_pseudobulk.obs['top_timepoint_numeric'].max()

# Create a figure with just the colorbar
fig, ax = plt.subplots(figsize=(1.5, 6))
norm_tp = Normalize(vmin=vmin_tp, vmax=vmax_tp)
cbar_tp = ColorbarBase(ax, cmap=cm.viridis, norm=norm_tp, orientation='vertical')
cbar_tp.set_label('Embryonic Day', fontsize=12)

# Save as PDF
plt.savefig(figure_path + 'mouse_timepoint_viridis_colorbar.pdf', 
            bbox_inches='tight', dpi=300)
plt.show()
print(f"Colorbar saved to: {figure_path}mouse_timepoint_viridis_colorbar.pdf")

# %% Normalize peak contrast values to 0-1 range for dot size scaling
print("\n" + "="*60)
print("Normalizing peak contrast values for size scaling...")
print("="*60)

# Min-max normalization to 0-1 range
def normalize_to_01(values):
    """Normalize values to 0-1 range using min-max scaling"""
    values_clean = np.array(values)
    values_clean = values_clean[~np.isnan(values_clean)]
    
    if len(values_clean) == 0:
        return np.zeros_like(values)
    
    min_val = np.min(values_clean)
    max_val = np.max(values_clean)
    
    if max_val == min_val:
        return np.ones_like(values) * 0.5  # If all same, return 0.5
    
    normalized = (np.array(values) - min_val) / (max_val - min_val)
    # Replace any NaN with 0
    normalized[np.isnan(normalized)] = 0
    
    return normalized

# Normalize both contrast metrics
celltype_contrast_normalized = normalize_to_01(peaks_by_pseudobulk.obs['celltype_peak_contrast'])
timepoint_contrast_normalized = normalize_to_01(peaks_by_pseudobulk.obs['timepoint_peak_contrast'])

peaks_by_pseudobulk.obs['celltype_peak_contrast_normalized'] = celltype_contrast_normalized
peaks_by_pseudobulk.obs['timepoint_peak_contrast_normalized'] = timepoint_contrast_normalized

print(f"Celltype contrast normalized range: {np.min(celltype_contrast_normalized):.3f} - {np.max(celltype_contrast_normalized):.3f}")
print(f"Timepoint contrast normalized range: {np.min(timepoint_contrast_normalized):.3f} - {np.max(timepoint_contrast_normalized):.3f}")

# %% Create UMAP plots with dot size scaled by peak contrast
print("\n" + "="*60)
print("Creating UMAP plots with contrast-scaled dot sizes...")
print("="*60)

# Get UMAP coordinates
umap_coords = peaks_by_pseudobulk.obsm['X_umap']

# Scale the normalized contrast to reasonable dot sizes
# Base size + contrast-based size
min_size = 0.5
max_size = 20.0
celltype_sizes = min_size + celltype_contrast_normalized * (max_size - min_size)
timepoint_sizes = min_size + timepoint_contrast_normalized * (max_size - min_size)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Plot 1: Celltype - colored by top celltype, sized by contrast
ax = axes[0]
# Get unique celltypes and assign colors
celltypes = peaks_by_pseudobulk.obs['peak_top_celltype'].unique()
celltype_colors = plt.cm.tab20(np.linspace(0, 1, len(celltypes)))
celltype_to_color = {ct: celltype_colors[i] for i, ct in enumerate(celltypes)}

# Map colors to each peak
colors_ct = [celltype_to_color[ct] for ct in peaks_by_pseudobulk.obs['peak_top_celltype']]

scatter1 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                     c=colors_ct, s=celltype_sizes, alpha=0.6, edgecolors='none')
ax.set_xlabel('UMAP1', fontsize=14)
ax.set_ylabel('UMAP2', fontsize=14)
ax.set_title('Celltype: Size = Contrast\n(Larger dots = more cell-type specific)', 
             fontsize=14, fontweight='bold')
ax.set_aspect('equal')

# Plot 2: Timepoint - colored by timepoint, sized by contrast
ax = axes[1]
# Use viridis colormap for timepoint
timepoint_numeric = peaks_by_pseudobulk.obs['peak_top_timepoint_numeric'].values
norm_tp_scatter = Normalize(vmin=np.nanmin(timepoint_numeric), vmax=np.nanmax(timepoint_numeric))
colors_tp = cm.viridis(norm_tp_scatter(timepoint_numeric))

scatter2 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                     c=colors_tp, s=timepoint_sizes, alpha=0.6, edgecolors='none')
ax.set_xlabel('UMAP1', fontsize=14)
ax.set_ylabel('UMAP2', fontsize=14)
ax.set_title('Timepoint: Size = Contrast\n(Larger dots = more stage-specific)', 
             fontsize=14, fontweight='bold')
ax.set_aspect('equal')

# Add colorbar for timepoint
cbar = plt.colorbar(cm.ScalarMappable(norm=norm_tp_scatter, cmap='viridis'), 
                    ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Embryonic Day', fontsize=12)

plt.tight_layout()
plt.savefig(figure_path + 'mouse_umap_contrast_sized_dots.png', dpi=150, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_umap_contrast_sized_dots.pdf', bbox_inches='tight')
plt.show()

print(f"Saved contrast-sized UMAP to: {figure_path}mouse_umap_contrast_sized_dots.png/pdf")

# %% Create separate plots for high contrast peaks only
print("\nCreating plots highlighting high-contrast peaks...")

# Define high contrast threshold (e.g., top 25%)
ct_threshold = np.percentile(celltype_contrast_normalized, 75)
tp_threshold = np.percentile(timepoint_contrast_normalized, 75)

high_ct_mask = celltype_contrast_normalized >= ct_threshold
high_tp_mask = timepoint_contrast_normalized >= tp_threshold

print(f"High celltype contrast peaks (top 25%): {high_ct_mask.sum():,} / {len(high_ct_mask):,}")
print(f"High timepoint contrast peaks (top 25%): {high_tp_mask.sum():,} / {len(high_tp_mask):,}")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Plot 1: Highlight high celltype contrast peaks
ax = axes[0]
# Background: all peaks in gray
ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
          c='lightgray', s=1, alpha=0.3, edgecolors='none')
# Foreground: high contrast peaks colored by celltype
colors_ct_high = [celltype_to_color[ct] for ct in peaks_by_pseudobulk.obs['peak_top_celltype'][high_ct_mask]]
ax.scatter(umap_coords[high_ct_mask, 0], umap_coords[high_ct_mask, 1],
          c=colors_ct_high, s=celltype_sizes[high_ct_mask], alpha=0.7, edgecolors='none')
ax.set_xlabel('UMAP1', fontsize=14)
ax.set_ylabel('UMAP2', fontsize=14)
ax.set_title(f'High Celltype Contrast Peaks (Top 25%)\nn={high_ct_mask.sum():,}', 
            fontsize=14, fontweight='bold')
ax.set_aspect('equal')

# Plot 2: Highlight high timepoint contrast peaks
ax = axes[1]
# Background: all peaks in gray
ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
          c='lightgray', s=1, alpha=0.3, edgecolors='none')
# Foreground: high contrast peaks colored by timepoint
colors_tp_high = cm.viridis(norm_tp_scatter(timepoint_numeric[high_tp_mask]))
ax.scatter(umap_coords[high_tp_mask, 0], umap_coords[high_tp_mask, 1],
          c=colors_tp_high, s=timepoint_sizes[high_tp_mask], alpha=0.7, edgecolors='none')
ax.set_xlabel('UMAP1', fontsize=14)
ax.set_ylabel('UMAP2', fontsize=14)
ax.set_title(f'High Timepoint Contrast Peaks (Top 25%)\nn={high_tp_mask.sum():,}', 
            fontsize=14, fontweight='bold')
ax.set_aspect('equal')

# Add colorbar for timepoint
cbar = plt.colorbar(cm.ScalarMappable(norm=norm_tp_scatter, cmap='viridis'), 
                    ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Embryonic Day', fontsize=12)

plt.tight_layout()
plt.savefig(figure_path + 'mouse_umap_high_contrast_peaks.png', dpi=150, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_umap_high_contrast_peaks.pdf', bbox_inches='tight')
plt.show()

print(f"Saved high-contrast peaks UMAP to: {figure_path}mouse_umap_high_contrast_peaks.png/pdf")

# %% Visualize peak contrast on UMAP
print("\n" + "="*60)
print("Visualizing peak contrast metrics on UMAP...")
print("="*60)

# Create a 2x2 grid for contrast visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Celltype peak contrast
sc.pl.umap(peaks_by_pseudobulk, color='celltype_peak_contrast',
           ax=axes[0, 0], show=False,
           title='Celltype Peak Contrast\n(Higher = more cell-type specific)',
           cmap='magma', vmin=0)

# Plot 2: Timepoint peak contrast
sc.pl.umap(peaks_by_pseudobulk, color='timepoint_peak_contrast',
           ax=axes[0, 1], show=False,
           title='Timepoint Peak Contrast\n(Higher = more stage-specific)',
           cmap='magma', vmin=0)

# Plot 3: Peak-level top celltype (for reference)
sc.pl.umap(peaks_by_pseudobulk, color='peak_top_celltype',
           ax=axes[1, 0], show=False,
           title='Most Accessible Celltype (Per Peak)')

# Plot 4: Peak-level top timepoint numeric (for reference)
sc.pl.umap(peaks_by_pseudobulk, color='peak_top_timepoint_numeric',
           ax=axes[1, 1], show=False,
           title='Most Accessible Timepoint (Per Peak)',
           cmap='viridis')

plt.tight_layout()
plt.savefig(figure_path + 'mouse_peak_contrast_umaps.png', dpi=150, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_peak_contrast_umaps.pdf', bbox_inches='tight')
plt.show()

# %% Individual peak contrast plots
# Celltype contrast
sc.pl.umap(peaks_by_pseudobulk, color='celltype_peak_contrast',
           title='Celltype Peak Contrast',
           cmap='magma', vmin=0,
           save='_celltype_peak_contrast.png')

# Timepoint contrast
sc.pl.umap(peaks_by_pseudobulk, color='timepoint_peak_contrast',
           title='Timepoint Peak Contrast',
           cmap='magma', vmin=0,
           save='_timepoint_peak_contrast.png')

# %% Distribution of peak contrast values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Celltype contrast histogram
axes[0].hist(peaks_by_pseudobulk.obs['celltype_peak_contrast'].dropna(), 
             bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Celltype Peak Contrast', fontsize=12)
axes[0].set_ylabel('Number of peaks', fontsize=12)
axes[0].set_title('Distribution of Celltype Peak Contrast', fontsize=14, fontweight='bold')
axes[0].axvline(peaks_by_pseudobulk.obs['celltype_peak_contrast'].median(), 
                color='red', linestyle='--', linewidth=2, 
                label=f'Median: {peaks_by_pseudobulk.obs["celltype_peak_contrast"].median():.2f}')
axes[0].legend()
axes[0].grid(False)

# Timepoint contrast histogram
axes[1].hist(peaks_by_pseudobulk.obs['timepoint_peak_contrast'].dropna(), 
             bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1].set_xlabel('Timepoint Peak Contrast', fontsize=12)
axes[1].set_ylabel('Number of peaks', fontsize=12)
axes[1].set_title('Distribution of Timepoint Peak Contrast', fontsize=14, fontweight='bold')
axes[1].axvline(peaks_by_pseudobulk.obs['timepoint_peak_contrast'].median(), 
                color='red', linestyle='--', linewidth=2,
                label=f'Median: {peaks_by_pseudobulk.obs["timepoint_peak_contrast"].median():.2f}')
axes[1].legend()
axes[1].grid(False)

plt.tight_layout()
plt.savefig(figure_path + 'mouse_peak_contrast_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_peak_contrast_distributions.pdf', bbox_inches='tight')
plt.show()

# %% Peak contrast by peak type
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Celltype contrast by peak type
sns.violinplot(data=peaks_by_pseudobulk.obs, x='peak_type', y='celltype_peak_contrast', ax=axes[0])
axes[0].set_xlabel('Peak Type', fontsize=12)
axes[0].set_ylabel('Celltype Peak Contrast', fontsize=12)
axes[0].set_title('Celltype Specificity by Peak Type', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(False)

# Timepoint contrast by peak type
sns.violinplot(data=peaks_by_pseudobulk.obs, x='peak_type', y='timepoint_peak_contrast', ax=axes[1])
axes[1].set_xlabel('Peak Type', fontsize=12)
axes[1].set_ylabel('Timepoint Peak Contrast', fontsize=12)
axes[1].set_title('Temporal Specificity by Peak Type', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(False)

plt.tight_layout()
plt.savefig(figure_path + 'mouse_peak_contrast_by_type.png', dpi=300, bbox_inches='tight')
plt.savefig(figure_path + 'mouse_peak_contrast_by_type.pdf', bbox_inches='tight')
plt.show()

print("\n=== Peak Contrast Summary Statistics ===")
print("\nCelltype Peak Contrast by Peak Type:")
print(peaks_by_pseudobulk.obs.groupby('peak_type')['celltype_peak_contrast'].describe())
print("\nTimepoint Peak Contrast by Peak Type:")
print(peaks_by_pseudobulk.obs.groupby('peak_type')['timepoint_peak_contrast'].describe())

# %% Visualize annotated UMAP (a master plot)
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
            dpi=150, bbox_inches='tight')
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

# %% Visualize accessibility specificity patterns on UMAP
print("\n" + "="*60)
print("Visualizing specificity patterns...")
print("="*60)

# Create color palette for specificity patterns
specificity_colors = {
    'ubiquitous': '#e41a1c',     # Green
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

# %% Part 3. Plot the most accessible celltypes and timepoints for each peak (using contrast as the opacity)
# use "celltype.mapped" for celltype, and "stage" for timepoints

# plot the most accessible celltypes and timepoints for each peak (using contrast as the opacity)
# fig, axes = plt.subplots(1, 2, figsize=(14, 7))
# ax = axes[0]
# ax.scatter(peaks_by_pseudobulk.obs['celltype_peak_contrast'], peaks_by_pseudobulk.obs['peak_top_celltype_accessibility'], alpha=0.5)
# ax.set_xlabel('Celltype Peak Contrast', fontsize=12)
# ax.set_ylabel('Peak Top Celltype Accessibility', fontsize=12)
# ax.set_title('Celltype Peak Contrast vs. Peak Top Celltype Accessibility', fontsize=14, fontweight='bold')
# ax.grid(False)

# %% plot the most accessible celltypes and timepoints for each peak (using contrast as the opacity)
# 1) compute the contrast for each peak (celltype and timepoint, respectively)
celltype_contrast_list = []
timepoint_contrast_list = []
for peak_idx in range(peaks_by_pseudobulk.n_obs):
    # Get accessibility across all pseudobulk groups for this peak
    peak_accessibility = peaks_by_pseudobulk.layers['normalized'][peak_idx, :].toarray().flatten() if hasattr(peaks_by_pseudobulk.layers['normalized'], 'toarray') else peaks_by_pseudobulk.layers['normalized'][peak_idx, :]
    
    # Group by celltype and compute mean accessibility per celltype
    celltype_accessibility = {}
    for i, pb_group in enumerate(peaks_by_pseudobulk.var_names):
        celltype = peaks_by_pseudobulk.var.loc[pb_group, 'celltype']
        if celltype not in celltype_accessibility:
            celltype_accessibility[celltype] = []
        celltype_accessibility[celltype].append(peak_accessibility[i])
    
    # Average across timepoints for each celltype
    celltype_means = {ct: np.mean(vals) for ct, vals in celltype_accessibility.items()}
    celltype_means_array = np.array(list(celltype_means.values()))
    celltype_names = list(celltype_means.keys())
    
    # Find most accessible celltype
    max_idx = np.argmax(celltype_means_array)
    top_celltype = celltype_names[max_idx]
    top_accessibility = celltype_means_array[max_idx]
    
    # Compute contrast
    contrast = compute_peak_contrast(celltype_means_array)
    
    celltype_contrast_list.append(contrast)
    top_celltype_list.append(top_celltype)
    top_celltype_accessibility_list.append(top_accessibility)

peaks_by_pseudobulk.obs['celltype_peak_contrast'] = celltype_contrast_list
peaks_by_pseudobulk.obs['top_celltype'] = top_celltype_list
# 
# Repeat this for the timepoint contrast (stage)
timepoint_contrast_list = []
top_timepoint_list = []
top_timepoint_accessibility_list = []
for peak_idx in range(peaks_by_pseudobulk.n_obs):
    # Get accessibility across all pseudobulk groups for this peak
    peak_accessibility = peaks_by_pseudobulk.layers['normalized'][peak_idx, :].toarray().flatten() if hasattr(peaks_by_pseudobulk.layers['normalized'], 'toarray') else peaks_by_pseudobulk.layers['normalized'][peak_idx, :]
    
    # Group by timepoint and compute mean accessibility per timepoint
    timepoint_accessibility = {}
    for i, pb_group in enumerate(peaks_by_pseudobulk.var_names):
        timepoint = peaks_by_pseudobulk.var.loc[pb_group, 'timepoint']
        if timepoint not in timepoint_accessibility:
            timepoint_accessibility[timepoint] = []
        timepoint_accessibility[timepoint].append(peak_accessibility[i])
    
    # Average across celltypes for each timepoint
    timepoint_means = {tp: np.mean(vals) for tp, vals in timepoint_accessibility.items()}
    timepoint_means_array = np.array(list(timepoint_means.values()))
    timepoint_names = list(timepoint_means.keys())
    
    # Find most accessible timepoint
    max_idx = np.argmax(timepoint_means_array)
    top_timepoint = timepoint_names[max_idx]
    top_accessibility = timepoint_means_array[max_idx]
    
    # Compute contrast
    contrast = compute_peak_contrast(timepoint_means_array)
    timepoint_contrast_list.append(contrast)
    top_timepoint_list.append(top_timepoint)
    top_timepoint_accessibility_list.append(top_accessibility)

peaks_by_pseudobulk.obs['timepoint_peak_contrast'] = timepoint_contrast_list
peaks_by_pseudobulk.obs['peak_top_timepoint'] = top_timepoint_list
#peaks_by_pseudobulk.obs['peak_top_timepoint_accessibility'] = top_timepoint_accessibility_list


# %% Save the annotated object
print("\n" + "="*60)
print("Saving annotated peak object...")
print("="*60)

print("\nFinal annotations included in .obs:")
print("  - Peak genomic annotations: peak_type, chr, start, end")
print("  - Gene associations: gene_body_overlaps, nearest_gene, distance_to_tss")
print("  - Accessibility metrics: total_accessibility, log_total_accessibility")
print("  - Cluster assignments: leiden_coarse")
print("  - Cluster-based annotations: top_celltype, top_timepoint, top_timepoint_numeric")
print("  - Peak-level most accessible: peak_top_celltype, peak_top_timepoint, peak_top_timepoint_numeric")
print("  - Peak-level accessibility values: peak_top_celltype_accessibility, peak_top_timepoint_accessibility")
print("  - Peak contrast metrics (raw): celltype_peak_contrast, timepoint_peak_contrast")
print("  - Peak contrast metrics (normalized 0-1): celltype_peak_contrast_normalized, timepoint_peak_contrast_normalized")
print("  - Specificity patterns: accessibility_pattern, accessibility_entropy")
print("  - Combined annotation: cluster_annotation")

output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_with_clusters.h5ad"
peaks_by_pseudobulk.write_h5ad(output_path)
print(f"\nSaved to: {output_path}")

print("\n" + "="*60)
print("Annotation pipeline complete!")
print("="*60)