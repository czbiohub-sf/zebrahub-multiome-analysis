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
# - Annotate the peaks by the gene names (based on their correlation with gene expression("SCT" assay, using Signac's LinkPeaks function)
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
peaks_pb_hvp_50k = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k_EDA.h5ad")
peaks_pb_hvp_50k

# %%
linked_peaks = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/peak_gene_links.csv", index_col="peak")
linked_peaks.head()

# %%
linked_peaks

# %%
plt.hist(linked_peaks["score"].values)
plt.grid(False)
plt.show()

# %%
linked_peaks[linked_peaks["zscore"]<0]

# %%
# First create a new column in adata.obs to store gene linkages
peaks_pb_hvp_50k.obs['linked_gene'] = ""
peaks_pb_hvp_50k.obs['link_score'] = np.nan
peaks_pb_hvp_50k.obs['link_zscore'] = np.nan
peaks_pb_hvp_50k.obs['link_pvalue'] = np.nan

# Map values from linked_peaks to adata.obs
for idx, row in linked_peaks.iterrows():
   peaks_pb_hvp_50k.obs.loc[idx, 'linked_gene'] = row['gene']
   peaks_pb_hvp_50k.obs.loc[idx, 'link_score'] = row['score'] 
   peaks_pb_hvp_50k.obs.loc[idx, 'link_zscore'] = row['zscore']
   peaks_pb_hvp_50k.obs.loc[idx, 'link_pvalue'] = row['pvalue']
    
# Create a categorical column for whether peak is linked
peaks_pb_hvp_50k.obs['is_linked'] = peaks_pb_hvp_50k.obs['linked_gene'].apply(lambda x: 'Linked' if x != '' else 'Not linked')

# Print summary statistics
print(f"Total linked peaks: {(peaks_pb_hvp_50k.obs['is_linked'] == 'Linked').sum()}")
print(f"Total unlinked peaks: {(peaks_pb_hvp_50k.obs['is_linked'] == 'Not linked').sum()}")

# Visualize on UMAP
sc.pl.umap(peaks_pb_hvp_50k, color='is_linked')

# %%
# Create a shuffled copy of the indices
shuffled_indices = np.random.permutation(peaks_pb_hvp_50k.obs_names)

# Reorder the AnnData object
peaks_pb_hvp_50k_shuffled = peaks_pb_hvp_50k[shuffled_indices]

# Plot with shuffled order
sc.pl.umap(peaks_pb_hvp_50k_shuffled, 
           color='is_linked',
           save='_peaks_linked_to_genes.png')

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["is_linked"]=="Linked"]

# %%
linked_peaks["gene"].value_counts().head(30)

# %%
linked_peaks["gene"].value_counts().tail(30)


# %%
# 1. Analyze peak cluster distribution for each gene
def analyze_gene_cluster_distribution(adata):
   # Get cluster assignments for each peak
   peak_clusters = adata.obs['leiden_0.3']
   linked_genes = adata.obs['linked_gene']
   
   # Create dictionary to store clusters for each gene
   gene_clusters = {}
   for gene in linked_genes.unique():
       if gene == '':  # Skip unlinked peaks
           continue
       # Get clusters where this gene has peaks
       gene_peaks = peak_clusters[linked_genes == gene]
       gene_clusters[gene] = {
           'clusters': set(gene_peaks),
           'n_clusters': len(set(gene_peaks)),
           'cluster_counts': gene_peaks.value_counts().to_dict()
       }
   
   # Create DataFrame with results
   results = pd.DataFrame({
       'gene': list(gene_clusters.keys()),
       'n_clusters': [d['n_clusters'] for d in gene_clusters.values()],
       'clusters': [','.join(map(str, d['clusters'])) for d in gene_clusters.values()],
       'cluster_counts': [str(d['cluster_counts']) for d in gene_clusters.values()]
   })
   
   return results, gene_clusters


# %%
results, gene_clusters = analyze_gene_cluster_distribution(peaks_pb_hvp_50k)

# %%

# %%
results.sort_values("n_clusters", ascending=False).head(20)

# %%
results.sort_values("n_clusters", ascending=False).tail(20)


# %%
# 2. Create gene-by-cluster matrix
def create_gene_cluster_matrix(adata):
   peak_clusters = adata.obs['leiden_0.3']
   linked_genes = adata.obs['linked_gene']
   unique_clusters = sorted(peak_clusters.unique())
   
   # Initialize matrix
   genes = sorted(set(linked_genes.unique()) - {''})
   gene_cluster_matrix = pd.DataFrame(0, 
                                    index=genes,
                                    columns=unique_clusters)
   
   # Fill matrix with counts
   for gene in genes:
       gene_peaks = peak_clusters[linked_genes == gene]
       counts = gene_peaks.value_counts()
       gene_cluster_matrix.loc[gene, counts.index] = counts
   
   # Create AnnData object for gene UMAP
   gene_adata = sc.AnnData(X=gene_cluster_matrix)
   
   # Compute UMAP
   sc.pp.normalize_total(gene_adata)
   sc.pp.log1p(gene_adata)
   sc.pp.pca(gene_adata)
   sc.pp.neighbors(gene_adata)
   sc.tl.umap(gene_adata)
   
   return gene_adata


# %%
adata_gene = create_gene_cluster_matrix(peaks_pb_hvp_50k)

# %%
adata_gene

# %%
sc.tl.leiden(adata_gene, resolution=0.3, key_added="leiden_0.3")
sc.tl.leiden(adata_gene, resolution=0.5, key_added="leiden_0.5")
sc.tl.leiden(adata_gene, resolution=0.7, key_added="leiden_0.7")
sc.tl.leiden(adata_gene, resolution=1, key_added="leiden_1")

# %%
sc.pl.umap(adata_gene, color=["leiden_0.3", "leiden_0.5",
                              "leiden_0.7", "leiden_1"], ncols=2)

# %%
linked_peaks[linked_peaks.gene=="sox6"]
linked_peaks[linked_peaks.gene=="myog"]

# %%
adata_gene[adata_gene.obs_names=="myog"].obs

# %%
adata_gene[adata_gene.obs["leiden_0.3"]=="5"].obs

# %%
for leiden_clust in adata_gene.obs["leiden_0.3"].unique():
    print(f"leiden cluster {leiden_clust}")
    print(adata_gene[adata_gene.obs["leiden_0.3"]==leiden_clust].obs_names)


# %%
def get_cluster_celltype_distribution(adata):
   # Get distribution of celltypes in each cluster
   cluster_celltype = {}
   
   for cluster in adata.obs['leiden_0.3'].unique():
       # Get peaks in this cluster
       cluster_mask = adata.obs['leiden_0.3'] == cluster
       # Get celltype distribution
       celltype_counts = adata.obs.loc[cluster_mask, 'celltype'].value_counts()
       # Store most common celltype and its percentage
       most_common = celltype_counts.index[0]
       percentage = (celltype_counts[0] / celltype_counts.sum()) * 100
       
       cluster_celltype[cluster] = {
           'dominant_celltype': most_common,
           'percentage': percentage,
           'distribution': celltype_counts
       }
   
   # Create mapping dictionary and results DataFrame
   cluster_to_celltype = {k: v['dominant_celltype'] for k, v in cluster_celltype.items()}
   
   results = pd.DataFrame({
       'cluster': list(cluster_celltype.keys()),
       'dominant_celltype': [v['dominant_celltype'] for v in cluster_celltype.values()],
       'percentage': [v['percentage'] for v in cluster_celltype.values()]
   })
   
   # # Add to adata.obs
   # adata.obs['cluster_celltype'] = adata.obs['leiden_0.3'].map(cluster_to_celltype)
   
   return results, cluster_celltype


# %%
results_cluster, cluster_celltype = get_cluster_celltype_distribution(peaks_pb_hvp_50k)
results_cluster.head()
# cluster_celltype

# %%
results

# %%
# First check data types
print("Results index type:", results.index.dtype)
print("AnnData index type:", adata_gene.obs.index.dtype)
print("Sample of results genes:", results['gene'].head())
print("Sample of AnnData index:", adata_gene.obs.index[:5])

# %%
results.set_index("gene", inplace=True)
results

# %%
# Add n_clusters and clusters to adata_gene
adata_gene.obs['n_clusters'] = adata_gene.obs_names.map(results["n_clusters"])
adata_gene.obs['cluster_ids'] = adata_gene.obs_names.map(results["clusters"])
adata_gene.obs.head()

# %%
sc.pl.umap(adata_gene, color="n_clusters", save='_genes_n_clusters_for_peaks.png', )


# %%
def map_clusters_to_celltypes(x):
    if pd.isna(x): return ''
    clusters = str(x).split(',')
    return ','.join([cluster_to_celltype.get(c.strip(), '') for c in clusters])

# Map clusters to celltypes
cluster_to_celltype = dict(zip(results_cluster['cluster'], results_cluster['dominant_celltype']))
cluster_to_celltype

# %%
adata_gene.obs['cluster_celltypes'] = adata_gene.obs['cluster_ids'].apply(map_clusters_to_celltypes)
adata_gene.obs.head()

# %%
sc.pl.umap(adata_gene)

# %%
sc.pl.umap(adata_gene, color=["cluster_celltypes"], save='_genes_peak_clusters_ids.pdf')

# %%
adata_gene[adata_gene.obs["cluster_celltypes"]=="optic_cup"].obs

# %%
adata_gene[adata_gene.obs["leiden_0.3"]=="8"].obs

# %%
adata_gene.obs["cluster_celltypes"].value_counts().head(10)

# %%
adata_gene[adata_gene.obs["cluster_celltypes"]=="fast_muscle,hemangioblasts"].obs_names

# %%
adata_gene

# %%
pathway_categories = {
    'endothelial_markers': ['cdh5', 'flt1', 'flt4', 'kdr', 'tek'],
    'vessel_formation': ['clec14a', 'esama', 'egfl7'],
    'hematopoietic_factors': ['gata1a', 'lmo2', 'tal1', 'spi1b'],
    'hemoglobin': ['hbae3', 'hbae5', 'hbbe2'],
    'cardiac_development': ['hand2', 'gata4', 'myh6', 'tnnt2a', 'tbx2a']
}

# Create pathway field with default 'NA'
adata_gene.obs['pathway'] = 'NA'

# Assign categories
for pathway, genes in pathway_categories.items():
    adata_gene.obs.loc[genes, 'pathway'] = pathway

# %%
sc.pl.umap(adata_gene, color="pathway", alpha=)

# %%
sc.pl.umap(adata_gene[adata_gene.obs["cluster_celltypes"]=="hemangioblasts"], color="pathway")

# %%
adata_gene[adata_gene.obs["leiden_0.3"]=="5"].obs

# %%

# %%

# %%

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
peaks_pb_hvp_50k.obs["hemato_manual"] = peaks_pb_hvp_50k.obs_names.map(manual_anno["subcluster"])
peaks_pb_hvp_50k

# %%
peaks_pb_hvp_50k.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k_EDA2.h5ad")
