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
# ## Associate peaks in the peak UMAP with "genes"
#
# - Annotate the peaks by the gene names (based on their proximity, or co-accessibility to TSS)
# - compute the peak clustering from the peak UMAP (celltype & timepoint)
# - compute the gene UMAP using the gene-by-"peak cluster" count matrix, and annotate an d perform EDA.
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

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

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
peaks_pb_hvp_50k = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k.h5ad")
peaks_pb_hvp_50k

# %%
# Plot UMAP without colors
sc.pl.umap(peaks_pb_hvp_50k, save="_peaks_50k_grey.png")

# %%
sc.pl.umap(peaks_pb_hvp_50k, color="dispersions_norm")


# %%

# %% [markdown]
# ## annotate the peaks based on the genes in proximity
#
#

# %%
# original function to associate the peaks to genes based on (1) overlap with the gene body, and (2) the nearest TSS
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
# sanity check for the peak with 17 overlapping gene bodies
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names=="10-21806886-21809893"].obs["gene_body_overlaps"].to_list()

# %% [markdown]
# #### NOTE: the genes that are associated with this "10-21806886-21809893" peak are all one gene, pcdh1g's isoforms

# %% [markdown]
# ## Associate the peaks to genes using "the most likely gene"

# %%
# Count peaks with no gene body overlaps
no_body_overlaps = (peaks_pb_hvp_50k.obs['gene_body_overlaps'] == '').sum()
total_peaks = len(peaks_pb_hvp_50k.obs)

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
peaks_pb_hvp_50k.obs.columns


# %%
# Associate the peaks to genes by associating peaks to (1) gene whose body is overlapping with the peak
# and (2) gene whose closest to the peak if (1) is not the case (within 50kb distance). 
# We will develop this into more sophisticated approach later on...(i.e., using chromatin co-accessibility data, etc.)
def get_primary_gene_associations(adata):
    """
    Associate each peak with a single most likely gene based on:
    1. Gene body overlap (if unique)
    2. If multiple overlaps, use the nearest TSS among overlapping genes
    3. If no overlaps, use nearest gene
    """
    peak_to_gene = {}
    
    for idx in adata.obs_names:
        # Get overlapping genes
        overlaps = adata.obs.loc[idx, 'gene_body_overlaps']
        nearest = adata.obs.loc[idx, 'nearest_gene']
        distance = adata.obs.loc[idx, 'distance_to_tss']
        
        if isinstance(overlaps, str) and overlaps != '':
            overlapping_genes = overlaps.split(',')
            if len(overlapping_genes) == 1:
                # Single overlap case
                peak_to_gene[idx] = overlapping_genes[0]
            else:
                # Multiple overlaps - use nearest TSS among overlapping genes
                if nearest in overlapping_genes:
                    peak_to_gene[idx] = nearest
                else:
                    peak_to_gene[idx] = overlapping_genes[0]  # Could be refined
        else:
            # No overlaps - use nearest gene
            peak_to_gene[idx] = nearest
            
    return peak_to_gene


# %%
# associate the peaks to genes using the above function
peak_to_gene = get_primary_gene_associations(peaks_pb_hvp_50k)

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names=="1-803100-803299"].obs["gene_body_overlaps"]


# %%
# First, let's analyze why some peaks have empty associations
def analyze_empty_associations(adata):
    """
    Analyze peaks that have empty gene associations
    """
    # Look at peaks with empty nearest_gene
    empty_nearest = adata.obs[adata.obs['nearest_gene'] == '']
    print(f"Peaks with empty nearest_gene: {len(empty_nearest)}")
    
    # Look at peaks with empty gene_body_overlaps
    empty_overlaps = adata.obs[adata.obs['gene_body_overlaps'] == '']
    print(f"Peaks with empty gene_body_overlaps: {len(empty_overlaps)}")
    
    # Look at peaks with both empty
    both_empty = adata.obs[(adata.obs['nearest_gene'] == '') & 
                          (adata.obs['gene_body_overlaps'] == '')]
    print(f"Peaks with both empty: {len(both_empty)}")
    
    # Sample a few cases
    print("\nExample peaks with empty associations:")
    for idx in both_empty.index[:5]:
        print(f"\nPeak: {idx}")
        print(f"Distance to TSS: {adata.obs.loc[idx, 'distance_to_tss']}")
        print(f"Chromosome: {idx.split('-')[0]}")
        print(f"Start: {idx.split('-')[1]}")
        print(f"End: {idx.split('-')[2]}")
        
    return both_empty

# Now let's modify the function to handle empty cases and return a DataFrame
def get_primary_gene_associations(adata):
    """
    Create a DataFrame of peak-gene associations with additional metadata
    """
    associations = []
    
    for idx in adata.obs_names:
        overlaps = adata.obs.loc[idx, 'gene_body_overlaps']
        nearest = adata.obs.loc[idx, 'nearest_gene']
        distance = adata.obs.loc[idx, 'distance_to_tss']
        
        # Initialize association info
        association = {
            'peak_id': idx,
            'chromosome': idx.split('-')[0],
            'start': int(idx.split('-')[1]),
            'end': int(idx.split('-')[2]),
            'associated_gene': None,
            'association_type': None,
            'distance_to_tss': distance,
            'n_overlapping_genes': 0,
            'all_overlapping_genes': '',
            'nearest_gene': nearest
        }
        
        # Process overlapping genes
        if isinstance(overlaps, str) and overlaps != '':
            overlapping_genes = [g for g in overlaps.split(',') if g]
            association['n_overlapping_genes'] = len(overlapping_genes)
            association['all_overlapping_genes'] = overlaps
            
            if len(overlapping_genes) == 1:
                association['associated_gene'] = overlapping_genes[0]
                association['association_type'] = 'single_overlap'
            elif len(overlapping_genes) > 1:
                # If nearest gene is among overlapping genes, use it
                if nearest in overlapping_genes:
                    association['associated_gene'] = nearest
                    association['association_type'] = 'nearest_among_overlaps'
                else:
                    association['associated_gene'] = overlapping_genes[0]
                    association['association_type'] = 'first_of_multiple_overlaps'
        
        # If no overlaps, use nearest gene
        if not association['associated_gene'] and nearest:
            association['associated_gene'] = nearest
            association['association_type'] = 'nearest_only'
            
        associations.append(association)
    
    # Convert to DataFrame
    df = pd.DataFrame(associations)
    
    # Add some summary statistics
    print("\nAssociation type distribution:")
    print(df['association_type'].value_counts())
    print("\nNumber of overlapping genes distribution:")
    print(df['n_overlapping_genes'].value_counts())
    
    return df

# Let's run both analyses
analyze_empty_associations(peaks_pb_hvp_50k)
associations_df = get_primary_gene_associations(peaks_pb_hvp_50k)

# %%
associations_df = associations_df.reset_index(drop=True).set_index('peak_id')
associations_df

# %%
null_associations = associations_df[associations_df['associated_gene'].isna()]
null_associations.head()

# %%
associations_df[associations_df.associated_gene==None]

# %%
# map the associated gene/type, and n_overlapping genes to the main adata object
peaks_pb_hvp_50k.obs['associated_gene'] = peaks_pb_hvp_50k.obs.index.map(associations_df['associated_gene'])
peaks_pb_hvp_50k.obs['association_type'] = peaks_pb_hvp_50k.obs.index.map(associations_df['association_type'])
peaks_pb_hvp_50k.obs['n_overlapping_genes'] = peaks_pb_hvp_50k.obs.index.map(associations_df['n_overlapping_genes'])

# %%

# %%
peaks_pb_hvp_50k.obs.columns

# %%
# Convert categorical to string before combining
peaks_pb_hvp_50k.obs['celltype_timepoint'] = (peaks_pb_hvp_50k.obs['celltype'].astype(str) + 
                                             '_' + 
                                             peaks_pb_hvp_50k.obs['timepoint'].astype(str))

# Convert back to categorical if desired
peaks_pb_hvp_50k.obs['celltype_timepoint'] = pd.Categorical(peaks_pb_hvp_50k.obs['celltype_timepoint'])

# Verify the new categories
print("Number of unique celltype_timepoint combinations:", 
      peaks_pb_hvp_50k.obs['celltype_timepoint'].nunique())

print("\nSample of celltype_timepoint categories:")
print(peaks_pb_hvp_50k.obs['celltype_timepoint'].value_counts().head())

# %%
from tqdm import tqdm


# %%
# First, let's create a mapping of peaks to their associated genes and celltype_timepoints
def create_gene_celltype_matrix(adata):
    """
    Create a gene-by-celltype_timepoint matrix showing peak counts
    """
    # Get unique genes and celltype_timepoints
    genes = set(adata.obs['associated_gene'].dropna())
    celltype_timepoints = adata.obs['celltype_timepoint'].unique()
    
    # Initialize the matrix
    gene_ct_matrix = pd.DataFrame(0, 
                                index=sorted(genes), 
                                columns=sorted(celltype_timepoints))
    
    # Fill the matrix by counting peaks
    for gene in tqdm(genes, desc="Processing genes"):
        # Get peaks associated with this gene
        gene_peaks = adata[adata.obs['associated_gene'] == gene]
        
        # Count peaks per celltype_timepoint
        peak_counts = gene_peaks.obs['celltype_timepoint'].value_counts()
        
        # Add to matrix
        gene_ct_matrix.loc[gene, peak_counts.index] = peak_counts
    
    return gene_ct_matrix



# %%
# Create the matrix
gene_celltype_matrix = create_gene_celltype_matrix(peaks_pb_hvp_50k)

# Print some statistics
print("Matrix shape:", gene_celltype_matrix.shape)
print("\nSample of the matrix:")
print(gene_celltype_matrix.iloc[:5, :5])

# Get genes with most peaks
print("\nGenes with most peaks:")
print(gene_celltype_matrix.sum(axis=1).sort_values(ascending=False).head())

# Get celltype_timepoints with most peaks
print("\nCelltype_timepoints with most peaks:")
print(gene_celltype_matrix.sum(axis=0).sort_values(ascending=False).head())

# %%
gene_celltype_matrix

# %%

# %%
from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix(gene_celltype_matrix.values)

# %%
# make an adata object (genes-by-{celltype_timepoint})
adata_assoc_genes = sc.AnnData(X=sparse_matrix)

# Assign obs_names (samples) and var_names (genes)
adata_assoc_genes.obs_names = gene_celltype_matrix.index
adata_assoc_genes.var_names = gene_celltype_matrix.columns
adata_assoc_genes

# %%
adata_assoc_genes.X

# %%
adata_assoc_genes.layers["counts"] = adata_assoc_genes.X.copy()

# %%
sc.pp.log1p(adata_assoc_genes)
adata_assoc_genes.layers["log_transformed"] = adata_assoc_genes.X.copy()

# %%

# %%
# normalize and scale the counts
adata_assoc_genes.X = adata_assoc_genes.layers["counts"]
sc.pp.normalize_total(adata_assoc_genes, target_sum=1e4)
sc.pp.log1p(adata_assoc_genes)
sc.pp.scale(adata_assoc_genes)
sc.pp.pca(adata_assoc_genes, n_comps=100, use_highly_variable=False)


# %%
sc.pp.neighbors(adata_assoc_genes, n_neighbors=100, n_pcs=40)
sc.tl.umap(adata_assoc_genes, min_dist=0.5, random_state=42)
sc.pl.umap(adata_assoc_genes)

# %%
# Create aggregated scores for timepoints
timepoint_cols = [col for col in adata_assoc_genes.var_names if col.endswith(('0somites', '5somites', '10somites', '15somites', '20somites', '30somites'))]

adata_assoc_genes.X = adata_assoc_genes.layers["counts"].copy()
sc.pp.normalize_total(adata_assoc_genes, target_sum=1e4)
sc.pp.log1p(adata_assoc_genes)

# Sum counts for each timepoint
for timepoint in ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']:
    relevant_cols = [col for col in adata_assoc_genes.var_names if col.endswith(timepoint)]
    adata_assoc_genes.obs[f'timepoint_{timepoint}'] = adata_assoc_genes.X[:, [adata_assoc_genes.var_names.get_loc(col) for col in relevant_cols]].sum(axis=1)

# Create aggregated scores for celltypes
# First get unique celltypes (removing timepoint suffix)
celltypes = set('_'.join(col.split('_')[:-1]) for col in adata_assoc_genes.var_names)

# Sum counts for each celltype
for celltype in celltypes:
    relevant_cols = [col for col in adata_assoc_genes.var_names if col.startswith(celltype + '_')]
    adata_assoc_genes.obs[f'celltype_{celltype}'] = adata_assoc_genes.X[:, [adata_assoc_genes.var_names.get_loc(col) for col in relevant_cols]].sum(axis=1)

# Plot UMAP with aggregated scores
# For timepoints
sc.pl.umap(adata_assoc_genes, 
           color=['timepoint_0somites', 'timepoint_5somites', 
                  'timepoint_10somites', 'timepoint_15somites',
                  'timepoint_20somites', 'timepoint_30somites'],
           ncols=3,
           save='_gene_timepoints.png')

# For celltypes (showing a few examples)
celltype_cols = [col for col in adata_assoc_genes.obs.columns if col.startswith('celltype_')]
sc.pl.umap(adata_assoc_genes, 
           color=celltype_cols[:3],  # Show first 3 celltypes
           save='_gene_celltypes.png')

# %%
# For each gene, count number of unique celltype_timepoint combinations with non-zero peaks
gene_program_counts = (gene_celltype_matrix > 0).sum(axis=1)

# Add this information to the adata_assoc_genes object
adata_assoc_genes.obs['n_regulatory_programs'] = gene_program_counts[adata_assoc_genes.obs_names]

# Plot UMAP colored by number of regulatory programs
sc.pl.umap(adata_assoc_genes, 
           color='n_regulatory_programs',
           title='Number of regulatory programs per gene',
           color_map='viridis',
           save='_num_regulatory_programs.png', vmin=1, vmax=10)

# %%
adata_assoc_genes

# %%
# Get all celltype columns
celltype_cols = [col for col in adata_assoc_genes.obs.columns if col.startswith('celltype_')]

# Count number of celltypes with non-zero counts for each gene
adata_assoc_genes.obs['n_celltypes'] = (adata_assoc_genes.obs[celltype_cols] > 0).sum(axis=1)

# Plot UMAP colored by number of celltypes
sc.pl.umap(adata_assoc_genes, 
           color='n_celltypes',
           title='Number of cell types per gene',
           color_map='viridis',
           save='_n_celltypes.png')

# Print some statistics
print("Distribution of number of cell types per gene:")
print(adata_assoc_genes.obs['n_celltypes'].value_counts().sort_index())

print("\nGenes with most cell types:")
print(adata_assoc_genes.obs['n_celltypes'].nlargest(10))

print("\nGenes with single cell type:")
print(adata_assoc_genes.obs[adata_assoc_genes.obs['n_celltypes'] == 1].index)

# %%
# First, find which celltype is active for genes with n_celltypes == 1
celltype_cols = [col for col in adata_assoc_genes.obs.columns if col.startswith('celltype_')]

# Initialize program_celltype column as 'multiple'
adata_assoc_genes.obs['program_celltype'] = 'multiple'

# For genes with single celltype, find which celltype it is
single_celltype_mask = adata_assoc_genes.obs['n_celltypes'] == 1
for gene_idx in adata_assoc_genes.obs[single_celltype_mask].index:
    # Get the active celltype (where count > 0)
    active_celltypes = [col.replace('celltype_', '') 
                       for col in celltype_cols 
                       if adata_assoc_genes.obs.loc[gene_idx, col] > 0]
    if len(active_celltypes) == 1:
        adata_assoc_genes.obs.loc[gene_idx, 'program_celltype'] = active_celltypes[0]

# Plot UMAP with program_celltype
sc.pl.umap(adata_assoc_genes, 
           color='program_celltype',
           title='Gene regulatory programs',
           save='_program_celltype.png')

# Print statistics
print("\nDistribution of program_celltype:")
print(adata_assoc_genes.obs['program_celltype'].value_counts())

# %%
# Plot UMAP with program_celltype
sc.pl.umap(adata_assoc_genes[adata_assoc_genes.obs["program_celltype"]!="multiple"], 
           color='program_celltype',
           title='Gene regulatory programs',
           save='_program_celltype_unique.png')


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
# Add 'multiple' to the color dictionary
cell_type_color_dict['multiple'] = '#808080'  # grey

# For program_celltype, prepare colors maintaining the order of categories
program_celltype_categories = sorted(adata_assoc_genes.obs['program_celltype'].unique())
program_celltype_colors = [cell_type_color_dict[cat] for cat in program_celltype_categories]

# Add to adata.uns
adata_assoc_genes.uns['program_celltype_colors'] = program_celltype_colors

# Plot UMAP
sc.pl.umap(adata_assoc_genes, 
           color='program_celltype',
           title='Gene regulatory programs by cell type',
           palette=cell_type_color_dict,
           save='_celltype_analysis_custom.png')

# %%
# # For program_celltype, prepare colors maintaining the order of categories
# program_celltype_categories = sorted(adata_assoc_genes.obs['program_celltype'].unique())
# program_celltype_colors = [cell_type_color_dict[cat] for cat in program_celltype_categories]

# # If you want to set different alpha values for different categories:
# alphas = {cat: 1.0 if cat != 'multiple' else 0.2 for cat in program_celltype_categories}
# sc.pl.umap(adata_assoc_genes, 
#            color='program_celltype',
#            title='Gene regulatory programs by cell type',
#            palette=cell_type_color_dict,
#            alpha=alphas,
#            save='_celltype_analysis_custom_transparent.png')

# %%
print(adata_assoc_genes[adata_assoc_genes.obs_names=="meox1"].obs["program_celltype"])
print(adata_assoc_genes[adata_assoc_genes.obs_names=="myf5"].obs["program_celltype"])
print(adata_assoc_genes[adata_assoc_genes.obs_names=="msgn1"].obs["program_celltype"])
print(adata_assoc_genes[adata_assoc_genes.obs_names=="myog"].obs["program_celltype"])

# %%

# %%
# First identify unique timepoints
timepoints = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']

# Count timepoints per gene
timepoint_counts = np.zeros(len(adata_assoc_genes.obs_names))

for gene_idx in range(len(adata_assoc_genes.obs_names)):
    active_timepoints = set()
    for col in adata_assoc_genes.var_names:
        if adata_assoc_genes.X[gene_idx, adata_assoc_genes.var_names.get_loc(col)] > 0:
            timepoint = col.split('_')[-1]
            if timepoint in timepoints:
                active_timepoints.add(timepoint)
    timepoint_counts[gene_idx] = len(active_timepoints)

# Add number of timepoints to obs
adata_assoc_genes.obs['n_timepoints'] = timepoint_counts

# Create program_timepoint field
adata_assoc_genes.obs['program_timepoint'] = 'multiple'

# For genes with single timepoint, identify which timepoint
single_timepoint_mask = adata_assoc_genes.obs['n_timepoints'] == 1
for gene_idx in np.where(single_timepoint_mask)[0]:
    active_timepoints = set()
    for col in adata_assoc_genes.var_names:
        if adata_assoc_genes.X[gene_idx, adata_assoc_genes.var_names.get_loc(col)] > 0:
            timepoint = col.split('_')[-1]
            if timepoint in timepoints:
                active_timepoints.add(timepoint)
    if len(active_timepoints) == 1:
        adata_assoc_genes.obs.iloc[gene_idx, adata_assoc_genes.obs.columns.get_loc('program_timepoint')] = list(active_timepoints)[0]

# Print statistics
print("Distribution of number of timepoints per gene:")
print(adata_assoc_genes.obs['n_timepoints'].value_counts().sort_index())

print("\nDistribution of program_timepoint:")
print(adata_assoc_genes.obs['program_timepoint'].value_counts())

# Plot UMAPs
sc.pl.umap(adata_assoc_genes, 
           color=['n_timepoints', 'program_timepoint'],
           title=['Number of timepoints per gene', 'Gene temporal programs'],
           save='_genes_timepoint_analysis.png')

# %%
# Create colors from viridis for timepoints
timepoints = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(timepoints)))
colors = {timepoint: viridis_colors[i] for i, timepoint in enumerate(timepoints)}
colors['multiple'] = '#808080'  # Add grey for 'multiple'

# Create the color palette
adata_assoc_genes.uns['program_timepoint_colors'] = [colors[x] for x in 
    sorted(adata_assoc_genes.obs['program_timepoint'].unique())]

# Plot UMAP
sc.pl.umap(adata_assoc_genes, 
           color='program_timepoint',
           title='Gene temporal programs',
           palette=colors,
           save='_timepoint_analysis_viridis.png')

# %%

# %%
adata_assoc_genes.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k.h5ad")

# %%

# %%

# %% [markdown]
# ## EDA2: Check the leiden clusters of peaks and map the "genes"

# %%
sc.tl.leiden(peaks_pb_hvp_50k, resolution = 0.7, key_added="leiden_0.7")
sc.tl.leiden(peaks_pb_hvp_50k, resolution = 0.5, key_added="leiden_0.5")
sc.tl.leiden(peaks_pb_hvp_50k, resolution = 0.3, key_added="leiden_0.3")
sc.tl.leiden(peaks_pb_hvp_50k, resolution = 1, key_added="leiden_1")


# %%
peaks_pb_hvp_50k

# %%
sc.pl.umap(peaks_pb_hvp_50k, color=['leiden_0.3', 'leiden_0.5', 
                                    'leiden_0.7','leiden_1'], ncols=2)

# %%
sc.pl.umap(peaks_pb_hvp_50k, color=['leiden_0.3',"celltype"], ncols=1)

# %%
peaks_pb_hvp_50k.obs['associated_gene']

# %%
# subset for a specific peak cluster, and ask which genes are associated with those peaks
cluster_id = "1"

genes_assoc_peak_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.3"]==cluster_id].obs["associated_gene"].unique()
genes_assoc_background_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.3"]!=cluster_id].obs["associated_gene"].unique()

# %%
len(genes_assoc_peak_cluster)

# %%
len(set(genes_assoc_peak_cluster) - set(genes_assoc_background_cluster))

# %%
genes_assoc_peak_cluster

# %%
genes_assoc_background_cluster

# %%
# subset for a specific peak cluster, and ask which genes are associated with those peaks
cluster_id = "0" # optic_cup (10hpf stage)

genes_assoc_peak_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.5"]==cluster_id].obs["associated_gene"].unique()
genes_assoc_background_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.5"]!=cluster_id].obs["associated_gene"].unique()

len(set(genes_assoc_peak_cluster) - set(genes_assoc_background_cluster))

# %%
# subset for a specific peak cluster, and ask which genes are associated with those peaks
cluster_id = "8" #

genes_assoc_peak_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_1"]==cluster_id].obs["associated_gene"].unique()
genes_assoc_background_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_1"]!=cluster_id].obs["associated_gene"].unique()

len(set(genes_assoc_peak_cluster) - set(genes_assoc_background_cluster))

# %%
goi = "meox1"

peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == goi].obs

# %%
goi = "myf5"

peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == goi].obs

# %%
goi="myf5"

peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == goi].obs["leiden_1"].unique()

len(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == goi].obs["leiden_1"].unique())

# %%
# df_genes_peaks = pd.DataFrame()
list_genes_unique_reg_programs = []
list_genes_multiple_reg_programs = []

# A for loop going through genes and ask whether peaks belong to multiple peak clusters or single
for gene in peaks_pb_hvp_50k.obs["associated_gene"].unique():
    if len(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == goi].obs["leiden_0.3"].unique())==1:
        list_genes_unique_reg_programs.append(gene)
    else:
        list_genes_multiple_reg_programs.append(gene)
        


# %%
list_genes_test = ["meox1","raraa","rxraa",
                   "pax6a","pax6b","msgn1"]

for gene in list_genes_test:
    num_clusters = len(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == gene].obs["leiden_0.3"].unique())
    n_peaks = len(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == gene])
    print(f"{gene} has {n_peaks} peaks over {num_clusters} cluster(s)")

# %%
dev_biologically_interesting_genes = [
    "tfap2b",
    "pax5",
    "prdm10",
    "foxp1b",
    "notch1b",
    "wnt7bb",
    "fgf14",
    "erbb4b",
    "epha4l",
    "neurod2",
    "disc1",
    "cux2b",
    "gria3a",
    "gabra4",
    "cdh23",
    "cdh7a",
    "cdh18a",
    "nid2a",
    "mmp13a",
    "lrrfip1b",
    "igfbp2b",
    "jakmip2",
    "pak1",
    "bach2a",
    "osr1",
    "ryr3",
    "mbnl1",
    "zdhhc23b",
]

for gene in dev_biologically_interesting_genes:
    num_clusters = len(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == gene].obs["leiden_0.3"].unique())
    n_peaks = len(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == gene])
    print(f"{gene} has {n_peaks} peaks over {num_clusters} cluster(s)")

# %%

# %%

# %%
len(peaks_pb_hvp_50k.obs["associated_gene"].unique())

# %%
len(list_genes_unique_reg_programs)

# %%
list_genes_multiple_reg_programs

# %%
len(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"] == goi].obs["leiden_0.3"].unique())

# %%

# %%
# Extract just the columns we need from .obs:
# - 'associated_gene': the gene nearest/overlapping each peak
# - 'leiden_0.3': or whichever leiden resolution you want to examine
df = peaks_pb_hvp_50k.obs[['associated_gene', 'leiden_0.3']].copy()

# Ensure 'associated_gene' is a string (in case of missing values or mixed types)
df['associated_gene'] = df['associated_gene'].astype(str)

# %%
# Group by gene and count the number of *unique* clusters it appears in
gene_cluster_counts = df.groupby('associated_gene')['leiden_0.3'].nunique()

# Look for genes that appear in more than 1 cluster
genes_in_multiple_clusters = gene_cluster_counts[gene_cluster_counts > 1]

print(f"Total genes: {len(gene_cluster_counts)}")
print(f"Genes that appear in multiple clusters: {len(genes_in_multiple_clusters)}")

# If you want to see which genes specifically:
print(genes_in_multiple_clusters)

# %%
# Group by cluster and count the number of *unique* genes
cluster_gene_counts = df.groupby('leiden_0.3')['associated_gene'].nunique()

print(cluster_gene_counts.sort_values(ascending=False))

# %%
# Subset only genes that appear in multiple clusters
df_multi = df[df['associated_gene'].isin(genes_in_multiple_clusters.index)]

# Check the distribution:
counts_per_cluster_for_genes_in_multi = (
    df_multi.groupby(['associated_gene', 'leiden_0.3'])
    .size()
    .reset_index(name='peak_count')
)

# Optionally sort to see which gene is split across which clusters
counts_per_cluster_for_genes_in_multi.sort_values(
    by=['associated_gene', 'peak_count'], 
    ascending=[True, False],
    inplace=True
)

counts_per_cluster_for_genes_in_multi.head(30)  # see top 30# Subset only genes that appear in multiple clusters
df_multi = df[df['associated_gene'].isin(genes_in_multiple_clusters.index)]

# Check the distribution:
counts_per_cluster_for_genes_in_multi = (
    df_multi.groupby(['associated_gene', 'leiden_0.3'])
    .size()
    .reset_index(name='peak_count')
)

# Optionally sort to see which gene is split across which clusters
counts_per_cluster_for_genes_in_multi.sort_values(
    by=['associated_gene', 'peak_count'], 
    ascending=[True, False],
    inplace=True
)

counts_per_cluster_for_genes_in_multi.head(30)  # see top 30

# %%
goi = "ABCA7"

peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["associated_gene"]==goi].obs_names

# %%
counts_per_cluster_for_genes_in_multi.associated_gene.unique()

# %%
# Save unique gene names to a CSV file
counts_per_cluster_for_genes_in_multi['associated_gene'].unique().tofile('unique_associated_genes.csv', sep='\n')

# %%
len(counts_per_cluster_for_genes_in_multi.associated_gene.unique())

# %%
goi = "bmp4"
# counts_per_cluster_for_genes_in_multi[counts_per_cluster_for_genes_in_multi.associated_gene==goi]

# check how many clusters the gene's peaks were distributed
counts_per_cluster_for_genes_in_multi[(counts_per_cluster_for_genes_in_multi.associated_gene==goi) & (counts_per_cluster_for_genes_in_multi.peak_count!=0)]

# %%
selected_genes = ["ascl1b","bmp4","fgf8a",
    "foxa2","gata6","her9","mef2ca",
    "mef2d","nog1","nog2","notch1a",
    "notch1b","pax2a","pax5","tbx20","tfap2b"
]

for goi in selected_genes:
    print(counts_per_cluster_for_genes_in_multi[(counts_per_cluster_for_genes_in_multi.associated_gene==goi) & (counts_per_cluster_for_genes_in_multi.peak_count!=0)])

# %%
goi = "pax6a"
print(counts_per_cluster_for_genes_in_multi[(counts_per_cluster_for_genes_in_multi.associated_gene==goi) & (counts_per_cluster_for_genes_in_multi.peak_count!=0)])

# %%

# %%

# %% [markdown]
# ## EDA on the clusters with distinct "peak type"
#
# - cluster 15 (leiden_0.3), where the tail end is enriched with "promoters"
# - cluster 

# %%
peaks_pb_hvp_50k[(peaks_pb_hvp_50k.obs["leiden_0.3"]=="15") & (peaks_pb_hvp_50k.obs["peak_type"]=="promoter")].obs

# %%
peaks_pb_hvp_50k.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k_EDA.h5ad")


# %% [markdown]
# ## EDA on some specific peak clusters (cluster 0 in leiden resolution of 0.3)
# - this peak cluster is highly accessible in optic_cup (10hpf)
# - manually sub-clustered the cluster 0
#

# %%
manual_anno = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig1_atlas_QC/peak_umap_subcluster.txt",
                          sep='\t',  # Specify tab separator
                         header=None,  # No header in the file
                         names=["index",'subcluster'], index_col = "index",  skiprows=1)
manual_anno

# %%
peaks_pb_hvp_50k.obs["optic_cup_subclust"] = peaks_pb_hvp_50k.obs_names.map(manual_anno.subcluster)
peaks_pb_hvp_50k.obs["optic_cup_subclust"]

# %%
sc.pl.umap(peaks_pb_hvp_50k, color="optic_cup_subclust")

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["optic_cup_subclust"]=="sub_clust_0"].obs["associated_gene"].unique()

# %%

# %%
## export the 50K peaks
# Extract the list of filtered peaks (from adata)
filtered_peaks = peaks_pb_hvp_50k.obs_names.tolist()

# Save to a text file
with open("peaks_hvp_50k.txt", "w") as f:
    for peak in filtered_peaks:
        f.write(peak + "\n")

# %%
peaks_stretched_over_chr_bounds = ["3-62628283-62628504",
                                   "10-45419551-45420917"]

for peak in peaks_stretched_over_chr_bounds:
    print(peak in filtered_peaks)

# %%

# %%
# load the master object
adata = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
adata

# %%
adata.obs["annotation_ML"]
adata.obs["dev_stage"]

# %%
# Create a new column 'celltype_timepoint' by combining 'annotation_ML' and 'dev_stage'
adata.obs['celltype_timepoint'] = adata.obs['annotation_ML'].astype(str) + "_" + adata.obs['dev_stage'].astype(str)
adata.obs['celltype_timepoint'] = adata.obs['celltype_timepoint'].astype('category')
adata.obs['celltype_timepoint'].unique()

# %%
adata.obs[['celltype_timepoint']].to_csv("celltype_timepoint_metadata.csv")

# %%

# %%

# %% [markdown]
# ## UMAP exploration (EDA)
# - look at the peaks around the hematopoetic system

# %%
peaks_pb_hvp_50k

# %%
# subset for a specific peak cluster, and ask which genes are associated with those peaks
cluster_id = "9"

genes_assoc_peak_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"]==cluster_id].obs["associated_gene"].unique()
genes_assoc_background_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"]!=cluster_id].obs["associated_gene"].unique()

# %%
peaks_hemato_sub = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"]==cluster_id]
peaks_hemato_sub

# %%
sc.pl.umap(peaks_hemato_sub,
           color = "timepoint")

# %%
manual_anno = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig1_atlas_QC/peaks_anno_hemangioblasts.txt",
                          sep='\t',  # Specify tab separator
                         header=None,  # No header in the file
                         names=["index",'subcluster'], index_col = "index",  skiprows=1)
manual_anno

# %%
manual_anno[manual_anno.subcluster=="hemangioblasts_late"]

# %%
peaks_hemato_sub[peaks_hemato_sub.obs_names.isin(manual_anno[manual_anno.subcluster=="hemangioblasts_late"].index)].obs["associated_gene"].unique().to_list()

# %%
peaks_hemato_sub[peaks_hemato_sub.obs_names.isin(manual_anno[manual_anno.subcluster=="hemangioblasts_early"].index)].obs["associated_gene"].unique().to_list()
