# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Global single-cell-base
#     language: python
#     name: global-single-cell-base
# ---

# %% [markdown]
# ## Annotate the peaks-by-celltype&timepoint (pseudobulked) object
#
# - EDA on the peaks-by-celltype&timepoint
# - Annotate the peaks by the peak types (refer to ArchR)
# - Annotate the peaks by the gene names (based on their proximity, or co-accessibility to TSS)
# - Annotate the peaks by the TFs whose motifs were found within the peaks
#
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

# import celloracle as co
# # Import celloracle function
# from celloracle import motif_analysis as ma

# %%
# figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')
# Set default DPI for saved figures
mpl.rcParams['savefig.dpi'] = 600

# Plotting style function (run this before plotting the final figure)
def set_plotting_style():
    plt.style.use('seaborn-paper')
    plt.rc('axes', labelsize=12)
    plt.rc('axes', titlesize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=2)
    plt.rc('mathtext', fontset='stixsans', sf='sansserif')
    plt.rc('figure', figsize=[10,9])
    plt.rc('svg', fonttype='none')



# %%
import logging
# Suppress INFO-level logs for the entire logger
logging.getLogger().setLevel(logging.WARNING)

# import logging
# logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_umap_annotated/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged.h5ad")
adata_peaks

# %%
# Plot UMAP without colors
sc.pl.umap(adata_peaks, save="_peaks_grey.png")

# %% [markdown]
# ## Annotate the peaks using genomic annotation
#
# - We will refer to the Argelaguet 2022 annotation - promoter, exonic, intronic, and intergenic
#

# %%
# Look into the ArchR on how to annotate the peaks using genomic annotation
import pyranges as pr
def annotate_peaks(peaks, gtf_file, upstream_promoter=2000, downstream_promoter=100):
    """
    Annotate peaks as promoter, exonic, intronic, or intergenic
    
    Parameters:
    peaks: DataFrame with columns 'Chromosome', 'Start', 'End'
    gtf_file: Path to GTF annotation file
    upstream_promoter: bases upstream of TSS to consider as promoter (default 2000)
    downstream_promoter: bases downstream of TSS to consider as promoter (default 100)
    """
    # Create PyRanges object for peaks
    peaks_gr = pr.PyRanges(peaks)
    
    # Read GTF file
    gtf = pr.read_gtf(gtf_file)
    
    # Get genes
    genes = gtf[gtf.Feature == 'gene']
    
    # Create promoter regions based on strand
    plus_promoters = genes[genes.Strand == '+'].copy()
    minus_promoters = genes[genes.Strand == '-'].copy()
    
    # Adjust coordinates for plus strand
    plus_promoters.Start = plus_promoters.Start - upstream_promoter
    plus_promoters.End = plus_promoters.Start + downstream_promoter + upstream_promoter
    
    # Adjust coordinates for minus strand
    minus_promoters.End = minus_promoters.End + upstream_promoter
    minus_promoters.Start = minus_promoters.End - (downstream_promoter + upstream_promoter)
    
    # Combine promoters
    promoters = pr.concat([plus_promoters, minus_promoters])
    
    # Get exons
    exons = gtf[gtf.Feature == 'exon']
    
    # Initialize peak types
    peaks_df = peaks.copy()
    peaks_df['peak_type'] = 'intergenic'
    
    # Find overlaps
    promoter_peaks = peaks_gr.overlap(promoters).as_df()
    exon_peaks = peaks_gr.overlap(exons).as_df()
    gene_peaks = peaks_gr.overlap(genes).as_df()
    
    # Create sets of overlapping peaks for efficient lookup
    promoter_set = set(zip(promoter_peaks.Chromosome, promoter_peaks.Start, promoter_peaks.End))
    exon_set = set(zip(exon_peaks.Chromosome, exon_peaks.Start, exon_peaks.End))
    gene_set = set(zip(gene_peaks.Chromosome, gene_peaks.Start, gene_peaks.End))
    
    # Annotate peaks
    for idx, row in peaks_df.iterrows():
        peak_tuple = (row.Chromosome, row.Start, row.End)
        if peak_tuple in promoter_set:
            peaks_df.at[idx, 'peak_type'] = 'promoter'
        elif peak_tuple in exon_set:
            peaks_df.at[idx, 'peak_type'] = 'exonic'
        elif peak_tuple in gene_set:
            peaks_df.at[idx, 'peak_type'] = 'intronic'
    
    return peaks_df


# %%
# Convert peak names to dataframe
peaks_df = pd.DataFrame([
    x.split('-') for x in peaks_pb_hvp_50k.obs_names
], columns=['chrom', 'start', 'end'])

# Convert to numeric and fix column names for PyRanges
peaks_df = peaks_df.rename(columns={
    'chrom': 'Chromosome',
    'start': 'Start',
    'end': 'End'
})

# Convert to numeric
peaks_df['Chromosome'] = peaks_df['Chromosome'].astype(str)
peaks_df['Start'] = peaks_df['Start'].astype(int)
peaks_df['End'] = peaks_df['End'].astype(int)

# %%
# Annotate peaks
annotated_peaks = annotate_peaks(peaks_df, '/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/genes/genes.gtf.gz')


# %%
annotated_peaks.head()

# %%
peaks_pb_hvp_50k.obs.head()

# %%
peaks_pb_hvp_50k.obs['peak_type'].values

# %%
# Create DataFrame from obs_names (which contain the coordinates)
coords = [x.split('-') for x in peaks_pb_hvp_50k.obs.index]
peaks_df = pd.DataFrame(coords, columns=['Chromosome', 'Start', 'End'])

# Convert Start and End to integers
peaks_df['Start'] = peaks_df['Start'].astype(int)
peaks_df['End'] = peaks_df['End'].astype(int)

# Add peak_type from obs
peaks_df.index = peaks_pb_hvp_50k.obs_names
peaks_df['peak_type'] = peaks_pb_hvp_50k.obs['peak_type']

# Look at the result
print(peaks_df.head())

# %%
annotated_peaks = peaks_df

# %%
# Add annotations to adata object
# Create a proper index for annotated_peaks using original peak names
annotated_peaks.index = peaks_pb_hvp_50k.obs.index
peaks_pb_hvp_50k.obs['peak_type'] = annotated_peaks['peak_type']

# Calculate proportions
props = peaks_pb_hvp_50k.obs['peak_type'].value_counts(normalize=True) * 100
print("\nPeak type proportions:")
for peak_type, prop in props.items():
    print(f"{peak_type}: {prop:.2f}%")

# Plot UMAP colored by peak type
sc.pl.umap(peaks_pb_hvp_50k, color='peak_type', save="_peaks_50k_peak_type.png")

# %%
# Plot UMAP colored by peak type
sc.pl.umap(peaks_pb_hvp_50k, color='peak_type', save="_peaks_50k_peak_type.pdf")

# %%
# Get the colors used for peak types in the UMAP
peak_type_colors = peaks_pb_hvp_50k.uns['peak_type_colors']
peak_type_names = ['exonic', 'intergenic', 'intronic', 'promoter']  # Match the order in UMAP
color_dict = dict(zip(peak_type_names, peak_type_colors))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot absolute counts with matching colors
sns.barplot(x=peak_counts.index, y=peak_counts.values, ax=ax1, 
           palette=color_dict)
ax1.set_title('Number of Peaks by Type')
ax1.set_ylabel('Number of peaks')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(False)

# Plot proportions with matching colors
sns.barplot(x=peak_props.index, y=peak_props.values, ax=ax2, 
           palette=color_dict)
ax2.set_title('Proportion of Peak Types')
ax2.set_ylabel('Percentage (%)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(False)

# Add value labels on top of bars
# for i, v in enumerate(peak_counts):
#     ax1.text(i, v, f'{v:,}', ha='center', va='bottom')
# for i, v in enumerate(peak_props):
#     ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(figpath+'peak_type_distributions.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
# sc.pl.umap(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["peak_type"]=="promoter"], color='peak_type', save="_promoter_peaks_50k_peak_type.png")

# %%
peaks_pb_hvp_50k.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k.h5ad")

# %%
sc.pl.umap(peaks_pb_hvp_50k, color="chrom", save="_chromosome.pdf")

# %%
sc.pl.umap(peaks_pb_hvp_50k, color="chrom", save="_chromosome.png")

# %% [markdown]
# ## Add manual annotation based on celltype&timepoint enrichment
#
# - used exCellxgene
#

# %%
annotation = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig1_atlas_QC/umap_obs.txt", index_col=0, sep="\t")
annotation.head()


# %%
# transfer the annotation
peaks_pb_hvp_50k.obs["manual_annotation_peaks"] = peaks_pb_hvp_50k.obs_names.map(annotation["manual_annotation_peaks"].astype('category'))
peaks_pb_hvp_50k.obs["leiden_r1"] = peaks_pb_hvp_50k.obs_names.map(annotation["leiden_v1_r1"].astype('category'))

# %%
# Get unique categories
categories = peaks_pb_hvp_50k.obs['manual_annotation_peaks'].cat.categories

# Create a custom color palette
palette = plt.get_cmap('tab20').colors  # Use a distinguishable palette
custom_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

# Set 'unassigned' to light grey
custom_colors['unassigned'] = 'lightgrey'

# Update the Scanpy settings
sc.pl.umap(
    peaks_pb_hvp_50k, color='manual_annotation_peaks', palette=[custom_colors[cat] for cat in categories],
    save="_peaks_50K_manual_annotation_v1_celltypes.png"
)

# %%
sc.pl.umap(peaks_pb_hvp_50k, color=["leiden_r1"])

# %%
# save the h5ad object with updated annotation
peaks_pb_hvp_50k.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k.h5ad")


# %% [markdown]
# ## annotate the peaks based on the genes in proximity
#
#

# %%
def associate_peaks_to_genes(peaks_df, gtf_file, max_distance=50000, chunk_size=1000):
    """
    Associate peaks with genes using chunked processing to reduce memory usage
    """
    # Create PyRanges object for peaks
    print("Initializing...")
    peaks_gr = pr.PyRanges(peaks_df)
    
    # Read GTF file and get genes
    print("Reading GTF file...")
    gtf = pr.read_gtf(gtf_file)
    genes = gtf[gtf.Feature == 'gene']
    
    print("Processing TSS coordinates...")
    # Create TSS coordinates for + and - strands separately
    plus_tss = genes[genes.Strand == '+'].copy()
    plus_tss.End = plus_tss.Start + 1
    
    minus_tss = genes[genes.Strand == '-'].copy()
    minus_tss.Start = minus_tss.End - 1
    
    # Combine TSS coordinates
    tss = pr.concat([plus_tss, minus_tss])
    
    # Initialize results DataFrame
    result_df = peaks_df.copy()
    result_df['gene_body_overlaps'] = ''
    result_df['nearest_gene'] = ''
    result_df['distance_to_tss'] = np.nan
    
    # Process in chunks
    total_chunks = len(peaks_df) // chunk_size + (1 if len(peaks_df) % chunk_size else 0)
    
    for chunk_idx in tqdm(range(total_chunks), desc="Processing peaks in chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(peaks_df))
        
        # Get chunk of peaks
        chunk_peaks = peaks_df.iloc[start_idx:end_idx]
        chunk_gr = pr.PyRanges(chunk_peaks)
        
        # Find overlaps for this chunk
        gene_overlaps = chunk_gr.join(genes, suffix='_gene', apply_strand_suffix=False)
        nearest_tss = chunk_gr.nearest(tss, suffix='_tss', apply_strand_suffix=False)
        
        # Process gene body overlaps
        if not gene_overlaps.empty:
            overlaps_df = gene_overlaps.as_df()
            overlaps_df['gene_name'] = overlaps_df['gene_name'].astype(str)
            
            # Group by peak coordinates
            for _, peak_group in overlaps_df.groupby(['Chromosome', 'Start', 'End']):
                peak_genes = ','.join(set(peak_group['gene_name']))
                idx = result_df[
                    (result_df['Chromosome'] == peak_group['Chromosome'].iloc[0]) &
                    (result_df['Start'] == peak_group['Start'].iloc[0]) &
                    (result_df['End'] == peak_group['End'].iloc[0])
                ].index
                result_df.loc[idx, 'gene_body_overlaps'] = peak_genes
        
        # Process nearest TSS
        if not nearest_tss.empty:
            nearest_df = nearest_tss.as_df()
            nearest_df['gene_name'] = nearest_df['gene_name'].astype(str)
            
            for _, row in nearest_df.iterrows():
                peak_center = (row['Start'] + row['End']) // 2
                # Use the regular Strand column instead of Strand_tss
                tss_pos = row['Start_tss'] if row['Strand'] == '+' else row['End_tss']
                distance = abs(peak_center - tss_pos)
                
                if distance <= max_distance:
                    idx = result_df[
                        (result_df['Chromosome'] == row['Chromosome']) &
                        (result_df['Start'] == row['Start']) &
                        (result_df['End'] == row['End'])
                    ].index
                    result_df.loc[idx, 'nearest_gene'] = row['gene_name']
                    result_df.loc[idx, 'distance_to_tss'] = distance
    
    return result_df

# Try the association with chunked processing
try:
    print("Starting peak to gene association...")
    peaks_with_genes = associate_peaks_to_genes(
        annotated_peaks,
        '/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/genes/genes.gtf.gz',
        chunk_size=1000
    )
    
    print("\nProcessing results...")
    with tqdm(total=3, desc="Adding annotations") as pbar:
        # Print statistics
        print("\nPeaks overlapping gene bodies:", 
              (peaks_with_genes['gene_body_overlaps'] != '').sum())
        print("Peaks with nearest gene within 50kb:", 
              (peaks_with_genes['nearest_gene'] != '').sum())
        pbar.update(1)
        
        # Add gene body overlaps
        peaks_pb_hvp_50k.obs['gene_body_overlaps'] = peaks_with_genes['gene_body_overlaps']
        pbar.update(1)
        
        # Add nearest gene and distance
        peaks_pb_hvp_50k.obs['nearest_gene'] = peaks_with_genes['nearest_gene']
        peaks_pb_hvp_50k.obs['distance_to_tss'] = peaks_with_genes['distance_to_tss']
        pbar.update(1)
        
except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("\nFull error details:")
    import traceback
    traceback.print_exc()

# %%
peaks_pb_hvp_50k.obs.head()

# %%
# Basic statistics for gene body overlaps
print("Gene Body Overlap Statistics:")
print("-" * 30)
# Count peaks with no gene body overlaps
no_body_overlaps = (peaks_pb_hvp_50k.obs['gene_body_overlaps'] == '').sum()
total_peaks = len(peaks_pb_hvp_50k.obs)
print(f"Peaks with no gene body overlaps: {no_body_overlaps} ({(no_body_overlaps/total_peaks)*100:.2f}%)")

# Count peaks with gene body overlaps
has_body_overlaps = (peaks_pb_hvp_50k.obs['gene_body_overlaps'] != '').sum()
print(f"Peaks with gene body overlaps: {has_body_overlaps} ({(has_body_overlaps/total_peaks)*100:.2f}%)")

# Distribution of number of overlapping genes per peak
overlapping_gene_counts = peaks_pb_hvp_50k.obs['gene_body_overlaps'].apply(
    lambda x: len(x.split(',')) if isinstance(x, str) and x != '' else 0
)
print("\nDistribution of number of overlapping genes per peak:")
print(overlapping_gene_counts.value_counts().sort_index())

print("\nNearest Gene Statistics:")
print("-" * 30)
# Count peaks with no nearest gene
no_nearest = peaks_pb_hvp_50k.obs['nearest_gene'].isna().sum()
print(f"Peaks with no nearest gene: {no_nearest} ({(no_nearest/total_peaks)*100:.2f}%)")
print(f"Peaks with nearest gene assigned: {total_peaks - no_nearest} ({((total_peaks-no_nearest)/total_peaks)*100:.2f}%)")

# Distance to TSS statistics
print("\nDistance to TSS Statistics (for peaks with nearest gene):")
print(peaks_pb_hvp_50k.obs['distance_to_tss'].describe())

# Overlap between methods
both_methods = ((peaks_pb_hvp_50k.obs['gene_body_overlaps'] != '') & 
                (~peaks_pb_hvp_50k.obs['nearest_gene'].isna())).sum()
only_body = ((peaks_pb_hvp_50k.obs['gene_body_overlaps'] != '') & 
             (peaks_pb_hvp_50k.obs['nearest_gene'].isna())).sum()
only_nearest = ((peaks_pb_hvp_50k.obs['gene_body_overlaps'] == '') & 
                (~peaks_pb_hvp_50k.obs['nearest_gene'].isna())).sum()
no_association = ((peaks_pb_hvp_50k.obs['gene_body_overlaps'] == '') & 
                 (peaks_pb_hvp_50k.obs['nearest_gene'].isna())).sum()

# %%
# check how many peaks were annotated by associated "genes"
peaks_pb_hvp_50k.obs["nearest_gene"].value_counts()

# %%
peaks_pb_hvp_50k.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k.h5ad")

# %% [markdown]
# ### generating UMAPs

# %%
sc.pl.umap(peaks_pb_hvp_50k, color=["distance_to_tss"], 
           vmin=0, vmax=50000,
           save="_peaks_50k_distance_to_tss.png")

# %%
# save as pdf as well
sc.pl.umap(peaks_pb_hvp_50k, color=["distance_to_tss"], vmin=0, vmax=50000, save="_peaks_50k_distance_to_tss.pdf")

# %%
# Create bins every 10000bp
bins = np.arange(0, 60000, 10000)  # Going up to 50000 to ensure we capture all values
labels = [f'{bins[i]}-{bins[i+1]}bp' for i in range(len(bins)-1)]

# Create binned column
peaks_pb_hvp_50k.obs['distance_to_tss_binned'] = pd.cut(
    peaks_pb_hvp_50k.obs['distance_to_tss'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Plot UMAP with binned distances
sc.pl.umap(peaks_pb_hvp_50k, 
           color='distance_to_tss_binned',
           save='_peaks_50k_distance_binned.png')

# %%
# sanity check using the "myf5" locus as an example
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names.str.startswith("4-2174")].obs

# %% [markdown]
# NOTE that the gene_body_overlaps doesn't necessarily mean that the "nearest_gene" would be the same.

# %%
# Create histogram with no gaps and KDE
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram
peaks_pb_hvp_50k.obs.distance_to_tss.hist(bins=30, density=True, 
                                         histtype='bar', 
                                         edgecolor='none',
                                         alpha=0.6,  # Make bars slightly transparent
                                         rwidth=1.0)  # Remove gaps between bars

# Add KDE
sns.kdeplot(data=peaks_pb_hvp_50k.obs.distance_to_tss, 
            color='red', 
            linewidth=2)

plt.xlabel("distance to TSS (bp)")
plt.ylabel("density")
plt.grid(False)
plt.savefig(figpath + "hist_peaks_50k_distance_to_tss.pdf")
plt.show()

# %%
peaks_pb_hvp_50k.obs["distance_to_tss_binned"].value_counts()

# %%

# %%

# %% [markdown]
# ## Aggreage over "celltype" or "timepoint"

# %%
# Create dictionaries to map columns to celltype and timepoint
celltype_mapping = {}
timepoint_mapping = {}

# Parse var names
for col in peaks_pb_hvp_50k.var.index:
    parts = col.rsplit('_', 1)
    if len(parts) == 2 and 'somites' in parts[1]:
        celltype = parts[0]
        timepoint = parts[1]
        celltype_mapping[col] = celltype
        timepoint_mapping[col] = timepoint

# Get unique celltypes and timepoints
unique_celltypes = set(celltype_mapping.values())
unique_timepoints = set(timepoint_mapping.values())

# Create new obs columns for each celltype and timepoint
for celltype in unique_celltypes:
    # Get columns for this celltype
    celltype_cols = [col for col, ct in celltype_mapping.items() if ct == celltype]
    # Sum accessibility across all timepoints for this celltype
    peaks_pb_hvp_50k.obs[f'accessibility_{celltype}'] = peaks_pb_hvp_50k.X[:, [peaks_pb_hvp_50k.var.index.get_loc(col) for col in celltype_cols]].sum(axis=1)

for timepoint in unique_timepoints:
    # Get columns for this timepoint
    timepoint_cols = [col for col, tp in timepoint_mapping.items() if tp == timepoint]
    # Sum accessibility across all celltypes for this timepoint
    peaks_pb_hvp_50k.obs[f'accessibility_{timepoint}'] = peaks_pb_hvp_50k.X[:, [peaks_pb_hvp_50k.var.index.get_loc(col) for col in timepoint_cols]].sum(axis=1)


# %%
# For timepoints
timepoint_cols = ['accessibility_0somites', 'accessibility_5somites', 'accessibility_10somites', 
                 'accessibility_15somites', 'accessibility_20somites', 'accessibility_30somites']
timepoint_vals = np.array([peaks_pb_hvp_50k.obs[col] for col in timepoint_cols]).T

# Find max timepoint for each peak
max_timepoint_idx = np.argmax(timepoint_vals, axis=1)
timepoint_names = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
peaks_pb_hvp_50k.obs['timepoint'] = [timepoint_names[i] for i in max_timepoint_idx]

# Calculate corrected timepoint contrast
max_vals = np.max(timepoint_vals, axis=1)
# Calculate mean and std excluding max value for each peak
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
peaks_pb_hvp_50k.obs['timepoint_contrast'] = (max_vals - other_vals_mean) / other_vals_std

# For celltypes
celltype_cols = [col for col in peaks_pb_hvp_50k.obs.columns 
                if col.startswith('accessibility_') and 'somites' not in col]
celltype_vals = np.array([peaks_pb_hvp_50k.obs[col] for col in celltype_cols]).T

# Find max celltype for each peak
max_celltype_idx = np.argmax(celltype_vals, axis=1)
celltype_names = [col.replace('accessibility_', '') for col in celltype_cols]
peaks_pb_hvp_50k.obs['celltype'] = [celltype_names[i] for i in max_celltype_idx]

# Calculate corrected celltype contrast
max_vals = np.max(celltype_vals, axis=1)
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(celltype_vals, max_celltype_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(celltype_vals, max_celltype_idx)])
peaks_pb_hvp_50k.obs['celltype_contrast'] = (max_vals - other_vals_mean) / other_vals_std

# %%
peaks_pb_hvp_50k

# %%
# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))
peaks_pb_hvp_50k.uns['timepoint_colors'] = [timepoint_colors[t] for t in timepoint_order]

# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k, 
           color='timepoint',
           size=peaks_pb_hvp_50k.obs['timepoint_contrast'],
           palette=timepoint_colors,
           save='_timepoint_scaled_viridis.png')


# %%
# Normalize peak contrast for alpha values
def normalize_for_alpha_robust(values, min_alpha=0.1, max_alpha=0.9):
    min_val = np.percentile(values, 5)
    max_val = np.percentile(values, 95)
    clipped = np.clip(values, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    return normalized * (max_alpha - min_alpha) + min_alpha

# Create alpha values from peak contrast
peaks_pb_hvp_50k.obs['alpha_timepoint'] = normalize_for_alpha_robust(peaks_pb_hvp_50k.obs['timepoint_contrast'])
peaks_pb_hvp_50k.obs['alpha_celltype'] = normalize_for_alpha_robust(peaks_pb_hvp_50k.obs['celltype_contrast'])


# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k, 
           color='timepoint',
           alpha=peaks_pb_hvp_50k.obs['alpha_timepoint'],
           palette=timepoint_colors,
           save='_timepoint_alpha_scaled_viridis.png')

# %%
# Use "Set3" color palette
sc.pl.umap(peaks_pb_hvp_50k, 
           color='timepoint',
           size=peaks_pb_hvp_50k.obs['timepoint_contrast'],
           palette="Set3",
           save='_timepoint_scaled.png')

# %%
# plot for the celltype with the celltype color palette
# A module to define the color palettes used in this paper
import matplotlib.pyplot as plt
import seaborn as sns

# a color palette for the "coarse" grained celltype annotation ("annotation_ML_coarse")
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'differentiating_neurons': '#bebada',
    'endocrine_pancreas': '#fb8072',
    'endoderm': '#80b1d3',
    'enteric_neurons': '#fdb462',
    'epidermis': '#b3de69',
    'fast_muscle': '#df4b9b',
    'floor_plate': '#d9d9d9',
    'hatching_gland': '#bc80bd',
    'heart_myocardium': '#ccebc5',
    'hemangioblasts': '#ffed6f',
    'hematopoietic_vasculature': '#e41a1c',
    'hindbrain': '#377eb8',
    'lateral_plate_mesoderm': '#4daf4a',
    'midbrain_hindbrain_boundary': '#984ea3',
    'muscle': '#ff7f00',
    'neural': '#e6ab02',
    'neural_crest': '#a65628',
    'neural_floor_plate': '#66a61e',
    'neural_optic': '#999999',
    'neural_posterior': '#393b7f',
    'neural_telencephalon': '#fdcdac',
    'neurons': '#cbd5e8',
    'notochord': '#f4cae4',
    'optic_cup': '#c0c000',
    'pharyngeal_arches': '#fff2ae',
    'primordial_germ_cells': '#f1e2cc',
    'pronephros': '#cccccc',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}



# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k, 
           color='celltype',
           size=peaks_pb_hvp_50k.obs['celltype_contrast'],
           palette=cell_type_color_dict,
           save='_celltype_scaled.png')

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k, 
           color='celltype',
           alpha=peaks_pb_hvp_50k.obs['alpha_celltype'],
           palette=cell_type_color_dict,
           save='_celltype_alpha_scaled.png')

# %%
# Create a shuffled copy of the indices
shuffled_indices = np.random.permutation(peaks_pb_hvp_50k.obs_names)

# Reorder the AnnData object
peaks_pb_hvp_50k_shuffled = peaks_pb_hvp_50k[shuffled_indices]

# Plot with shuffled order
sc.pl.umap(peaks_pb_hvp_50k_shuffled, 
           color='celltype',
           save='_celltype_shuffled.png')

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k_shuffled, 
           color='timepoint',
           alpha=peaks_pb_hvp_50k_shuffled.obs['alpha_timepoint'],
           palette=timepoint_colors,
           save='_timepoint_alpha_scaled_viridis_shuffled.png')

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k_shuffled, 
           color='celltype',
           alpha=peaks_pb_hvp_50k_shuffled.obs['alpha_celltype'],
           palette=cell_type_color_dict,
           save='_celltype_alpha_scaled_shuffled.png')

# %%
peaks_pb_hvp_50k.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k.h5ad")

# %%

# %%

# %% [markdown]
# # ------------------ HOLD ----------------------

# %% [markdown]
# ## EDA on the "marker genes/peaks" for specific cluster (using manual annotation on the peaks)

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["manual_annotation_peaks"]=="hatching_gland"].obs["gene_body_overlaps"].unique()

# %%
# query which genes were associated with the enriched peaks for each cluster
celltype = "muscle"
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["manual_annotation_peaks"]==celltype].obs["gene_body_overlaps"].unique()

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["manual_annotation_peaks"]=="NMPs"].obs["gene_body_overlaps"].value_counts()[0:20]

# %%
# query which genes were associated with the enriched peaks for each cluster
celltype = "tail_bud_late"
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["manual_annotation_peaks"]==celltype].obs["gene_body_overlaps"].value_counts()[0:20]


# %%
# 1. Calculate gene enrichment per cluster
def analyze_peak_clusters(adata, cluster_key='leiden_r1'):
    """
    Analyze gene associations for each peak cluster
    """
    # Group peaks by cluster
    cluster_genes = {}
    cluster_stats = {}
    
    # For each cluster
    for cluster in adata.obs[cluster_key].unique():
        # Get peaks in this cluster
        cluster_peaks = adata.obs[adata.obs[cluster_key] == cluster]
        
        # Collect genes (both overlapping and nearest)
        all_genes = set()
        if 'gene_body_overlaps' in cluster_peaks.columns:
            genes = [g for genes in cluster_peaks['gene_body_overlaps'] if genes 
                    for g in genes.split(',') if g]
            all_genes.update(genes)
            
        if 'nearest_gene' in cluster_peaks.columns:
            all_genes.update(cluster_peaks['nearest_gene'].dropna())
        
        # Calculate statistics
        cluster_stats[cluster] = {
            'n_peaks': len(cluster_peaks),
            'n_genes': len(all_genes),
            'genes_per_peak': len(all_genes) / len(cluster_peaks) if len(cluster_peaks) > 0 else 0,
            'median_tss_distance': cluster_peaks['distance_to_tss'].median()
        }
        
        cluster_genes[cluster] = list(all_genes)
    
    return pd.DataFrame(cluster_stats).T, cluster_genes

# 2. Analyze peak type distributions per cluster
def analyze_peak_types(adata, cluster_key='leiden_r1'):
    """
    Analyze peak type distributions for each cluster
    """
    return pd.crosstab(
        adata.obs[cluster_key], 
        adata.obs['peak_type'], 
        normalize='index'
    )


# %%
# Example usage:
cluster_stats, cluster_genes = analyze_peak_clusters(peaks_pb_hvp_50k, 
                                                     cluster_key="manual_annotation_peaks")
peak_type_dist = analyze_peak_types(peaks_pb_hvp_50k, cluster_key="manual_annotation_peaks")

print("Cluster Statistics:")
print(cluster_stats)

print("\nPeak Type Distribution:")
print(peak_type_dist)


# %%

# %%
def analyze_specific_genes(adata, genes_of_interest, celltype_key='manual_annotation_peaks'):
    """
    Analyze distribution of specific genes across cell types
    """
    results = {}
    
    for gene in genes_of_interest:
        # Find all peaks associated with this gene
        cell_types = []
        for celltype in adata.obs[celltype_key].unique():
            genes = adata[adata.obs[celltype_key] == celltype].obs['gene_body_overlaps']
            # Check if gene appears in any peaks for this cell type
            if genes.str.contains(gene, na=False).any():
                cell_types.append(celltype)
        
        results[gene] = cell_types
    
    # Convert to DataFrame
    result_df = pd.DataFrame({
        'gene': list(results.keys()),
        'n_celltypes': [len(types) for types in results.values()],
        'celltypes': [','.join(types) for types in results.values()]
    })
    
    return result_df


# %%
genes_of_interest = ['myf5', 'tbxta', 'sox2']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print("\nDistribution of specific genes:")
print(gene_analysis)

# %%
genes_of_interest = ['myf5', 'msgn1', 'meox1']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print("\nDistribution of specific genes:")
print(gene_analysis)

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names=="19-14185848-14186763"].obs["nearest_gene"]

# %%
peaks_pb_hvp_50k


# %%
def analyze_specific_genes(adata, genes_of_interest, celltype_key='manual_annotation_peaks'):
    """
    Analyze distribution of specific genes across cell types using both gene_body_overlaps 
    and nearest_gene information
    """
    results = {}
    
    for gene in genes_of_interest:
        # Find all peaks associated with this gene
        cell_types = []
        gene_associations = {}  # To track how the gene was associated (overlap or nearest)
        
        for celltype in adata.obs[celltype_key].unique():
            celltype_peaks = adata[adata.obs[celltype_key] == celltype]
            
            # Check gene_body_overlaps first
            overlaps = celltype_peaks.obs['gene_body_overlaps']
            if overlaps.str.contains(gene, na=False).any():
                cell_types.append(celltype)
                gene_associations[celltype] = 'overlap'
                continue
                
            # If not found in overlaps, check nearest_gene
            nearest = celltype_peaks.obs['nearest_gene']
            if nearest.str.contains(gene, na=False).any():
                cell_types.append(celltype)
                gene_associations[celltype] = 'nearest'
        
        # Store results with association type
        results[gene] = {
            'celltypes': cell_types,
            'associations': gene_associations
        }
    
    # Convert to DataFrame with additional information
    result_df = pd.DataFrame({
        'gene': list(results.keys()),
        'n_celltypes': [len(res['celltypes']) for res in results.values()],
        'celltypes': [','.join(res['celltypes']) for res in results.values()],
        'association_details': [
            '; '.join([f"{ct}({assoc})" for ct, assoc in res['associations'].items()]) 
            for res in results.values()
        ]
    })
    
    return result_df

# Example usage:
"""
genes_of_interest = ['myf5', 'msgn1', 'meox1']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print(gene_analysis)
"""

# %%
genes_of_interest = ['myf5', 'msgn1', 'meox1']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print(gene_analysis)

# %%
genes_of_interest = ['myf5', 'tbxta', 'sox2']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print(gene_analysis)


# %%
# compute the n_celltypes and celltyeps for all genes for systematic analysis
def analyze_all_genes(adata, celltype_key='manual_annotation_peaks'):
    """
    Analyze distribution of all genes across cell types
    Returns a DataFrame with gene statistics
    """
    # Get all unique genes from gene_body_overlaps
    all_genes = set()
    for genes_str in adata.obs['gene_body_overlaps']:
        if isinstance(genes_str, str):
            genes = genes_str.split(',')
            all_genes.update([g.strip() for g in genes])
    
    # Remove empty strings if any
    all_genes = {g for g in all_genes if g}
    
    results = []
    
    print(f"Analyzing {len(all_genes)} genes...")
    for gene in tqdm(all_genes):
        cell_types = []
        # Get counts per celltype
        celltype_counts = {}
        
        for celltype in adata.obs[celltype_key].unique():
            # Get peaks for this celltype
            celltype_peaks = adata[adata.obs[celltype_key] == celltype]
            # Check which peaks contain this gene
            gene_peaks = celltype_peaks.obs['gene_body_overlaps'].str.contains(gene, na=False)
            if gene_peaks.any():
                cell_types.append(celltype)
                celltype_counts[celltype] = gene_peaks.sum()
        
        results.append({
            'gene': gene,
            'n_celltypes': len(cell_types),
            'celltypes': ','.join(cell_types),
            'total_peaks': sum(celltype_counts.values()),
            'celltype_counts': celltype_counts
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Add columns for peak counts per celltype
    for celltype in adata.obs[celltype_key].unique():
        result_df[f'peaks_in_{celltype}'] = result_df['celltype_counts'].apply(
            lambda x: x.get(celltype, 0)
        )
    
    # Drop the dictionary column
    result_df = result_df.drop('celltype_counts', axis=1)
    
    # Sort by number of celltypes and total peaks
    result_df = result_df.sort_values(['n_celltypes', 'total_peaks'], ascending=[False, False])
    
    return result_df


# %%
# Run analysis for all genes
gene_stats = analyze_all_genes(peaks_pb_hvp_50k)

# Look at top genes by number of cell types
print("\nGenes present in most cell types:")
print(gene_stats.head())

# Look at cell-type specific genes
print("\nCell-type specific genes:")
print(gene_stats[gene_stats['n_celltypes'] == 1].head())

# Save results
gene_stats.to_csv('peak_gene_analysis.csv')

# Get summary statistics
print("\nDistribution of genes across cell types:")
print(gene_stats['n_celltypes'].value_counts().sort_index())

# %%
gene_stats = pd.read_csv('peak_gene_analysis.csv', index_col=0)
gene_stats

# %%
gene_stats[gene_stats.gene=="myf5"]

# %%
gene_stats[gene_stats.gene=="meox1"]

# %%
celltype_specific_genes = gene_stats[gene_stats['n_celltypes'] == 1][gene_stats["celltypes"]!="unassigned"]
celltype_specific_genes

# %%
celltype_specific_genes[celltype_specific_genes.celltypes=="muscle"].gene.to_list()

# %%

# %%
peaks_pb_hvp_50k.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k_EDA.h5ad")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
