# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: sc_rapids
#     language: python
#     name: sc_rapids
# ---

# %% [markdown]
# ## Finer sub-clustering of peak UMAP (sub-cluster "leiden_coarse" to create "leiden_fine" mapping)
#
# - last updated: 06/20/2025
# - The goal is to pick the large clusters from "leiden_coarse", and sub-cluster for each leaflet
#
# - 1) For each "leiden_coarse" cluster, check the number of peaks.
# - 2) For a selected group of coarse clusters, subset, and perform fine-clustering to 
#
# - Inspired by Cytoself and SubCell papers, we will perform hierarchical clustering on the peak UMAP to find the meaningful "resolution" of peak groups
# - tissue-level (or dev stages)
# - biological pathways
# - gene-gene interaction? (ligand-receptor interaction?)
# - TF motif enrichments
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
import os

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# %%
# figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# %%
import logging
# Suppress INFO-level logs for the entire logger
logging.getLogger().setLevel(logging.WARNING)

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_umap_sub_cluster_640K_peaks/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# define the filepath to save the subsetted adata objects
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/adata_peak_clusters_subset/"
filepath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")
adata_peaks

# %%
# move the .X to GPU (for faster operation)
rsc.get.anndata_to_GPU(adata_peaks)

# %%
# Add the linked_genes from the Signac's LinkGenes2Peaks function
df_linked_genes = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/all_peaks_annotated.csv", index_col=0)
df_linked_genes.head()

# %%
# # copy over the linked genes
fields_to_copy = ["link_pvalue", "link_score", "link_zscore", "linked_gene"]
for field in fields_to_copy:
    adata_peaks.obs[field] = df_linked_genes[field]
    
adata_peaks.obs.columns

# %%
adata_peaks.obs["linked_gene"].unique()

# %%
# import the functions to annotate the peaks with "associated genes"
# this function uses "linked_genes" from Signac and "overlapping with gene body" based on GTF file
from utils_gene_annotate import *

# %%
# annotate the peaks with associated genes
adata_peaks = create_gene_associations(adata_peaks)
adata_peaks

# %% [markdown]
# ## Step 1. Subset the leiden_coarse clusters, re-embedding and re-clustering
#
# - Here, we manually selected several coarse clusters to sub-cluster: 1, 7, 13, and 22, based on their "sub-branches" and also their celltypes

# %%
adata_peaks.X = adata_peaks.layers["normalized"].copy()
adata_peaks

# %%
rsc.get.anndata_to_GPU(adata_peaks)

# %%
# choose a cluster, subset, then sub-cluster, and save the resulting adata_sub (or adata_sub.obs)
coarse_clusts = [1,7,13,22]

# create an empty dict to save the subsetted adata for each "leiden_coarse" cluster
dict_adata_sub = {}

for clust in coarse_clusts:
    adata_sub = adata_peaks[adata_peaks.obs["leiden_coarse"]==clust].copy()
    # move the global UMAP coordinates
    adata_sub.obsm["X_umap_global"] = adata_sub.obsm["X_umap"]

    # recompute the UMAP
    rsc.pp.scale(adata_sub)
    rsc.pp.pca(adata_sub, n_comps=50, use_highly_variable=False)
    rsc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=40)
    rsc.tl.umap(adata_sub, random_state=42, min_dist=0.3)

    # define the resolution and compute the leiden clustering (sub-clustering)
    list_leiden_sub_res = []

    for res in [0.3, 0.5, 0.7, 1, 1.2]:
        # sub-cluster using the res
        rsc.tl.leiden(adata_sub, resolution=res, key_added=f"leiden_sub_{res}")
        # save the key_added in a list
        list_leiden_sub_res.append(f"leiden_sub_{res}")
        num_clusts = len(adata_sub.obs[f"leiden_sub_{res}"].unique())
        print(f"leiden resolution:{res} gave {num_clusts} clusters")
    
    print(adata_sub.obs["leiden_sub_0.5"].value_counts())
    # create a UMAP colored by different leiden resolutions
    print(f"re-computed UMAP with sub-clustering for leiden_coarse={clust}")
    sc.pl.umap(adata_sub, color=list_leiden_sub_res)
    
    # save the adata_sub
    dict_adata_sub[clust] = adata_sub

# %% [markdown]
# ### We decided to use the leiden resolution of 0.7, then merge the sub-clusters that are too small (less than 100 peaks)

# %%
sys.path.append("../../scripts/utils/")
from clustering import *

# %%
adata_peaks.obs.leiden_coarse.unique()
np.arange(0,36)

# %%
# for loop to subset the adata for each "leiden_coarse" cluster
# then, do the sub-clustering, and merging the small clusters (<100 peaks)
# define an empty dictionary to save the subsetted adata objects
dict_adata_sub = {}

for clust in np.arange(0,36):
    # adata_sub = dict_adata_sub[clust]
    adata_sub = adata_peaks[adata_peaks.obs["leiden_coarse"]==clust].copy()
    # move the global UMAP coordinates
    adata_sub.obsm["X_umap_global"] = adata_sub.obsm["X_umap"]

    # recompute the UMAP
    rsc.pp.scale(adata_sub)
    rsc.pp.pca(adata_sub, n_comps=50, use_highly_variable=False)
    rsc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=40)
    rsc.tl.umap(adata_sub, random_state=42, min_dist=0.3)

    # define the resolution and compute the leiden clustering (sub-clustering)
    list_leiden_sub_res = []

    # for res in [0.3, 0.5, 0.7, 1, 1.2]:
    res = 0.7
    # sub-cluster using the res
    rsc.tl.leiden(adata_sub, resolution=res, key_added=f"leiden_sub_{res}")
    # save the key_added in a list
    # list_leiden_sub_res.append(f"leiden_sub_{res}")
    # num_clusts = len(adata_sub.obs[f"leiden_sub_{res}"].unique())
    # print(f"leiden resolution:{res} gave {num_clusts} clusters")
#     print(adata_sub.obs["leiden_sub_0.5"].value_counts())
    # create a UMAP colored by different leiden resolutions
    # print(f"re-computed UMAP with sub-clustering for leiden_coarse={clust}")
    # sc.pl.umap(adata_sub, color=f"leiden_sub_{res}")

    # Part 2: merging sub-clusters that are too mall
    # we will choose the 0.7 as the default resolution
    # res = 0.7
    # merge the small clusters than have fewer than 100 entries (peaks)
    adata_sub = merge_small_clusters(
        adata_sub,
        leiden_key= f"leiden_sub_{res}",
        adjacency_key= "connectivities",
        merged_key_suffix = "merged",
        threshold = 100,
        merge_small_clusters= True,
        random_seed= 1)
    
    # Renumber clusters to be consecutive (0, 1, 2, 3, ...)
    merged_key = f"leiden_sub_{res}_merged"
    
    # Get unique cluster labels and sort them
    unique_clusters = sorted(adata_sub.obs[merged_key].unique())
    
    # Create mapping from old labels to new consecutive labels
    cluster_mapping = {old_label: str(i) for i, old_label in enumerate(unique_clusters)}
    
    print(f"Cluster renaming mapping: {cluster_mapping}")
    
    # Apply the renumbering
    adata_sub.obs[f"{merged_key}_renumbered"] = adata_sub.obs[merged_key].map(cluster_mapping).astype('category')
    
    print(f"After renumbering:")
    print(adata_sub.obs[f"{merged_key}_renumbered"].value_counts().sort_index())
    
    # Update the dictionary with the modified adata_sub
    dict_adata_sub[clust] = adata_sub
    
    # create a UMAP colored by different leiden resolutions
    print(f"re-computed UMAP with sub-clustering for leiden_coarse={clust}")
    
    # Plot both original merged and renumbered versions
    sc.pl.umap(adata_sub, color=[f"{merged_key}_renumbered"], 
               save=f"_peaks_leiden_coarse_{clust}_subclust.png")
    
    # save the subsetted h5ad objects
    adata_sub.write_h5ad(filepath + f"peaks_leiden_coarse_cluster_{clust}_subset.h5ad")
    # print(adata_sub.obs[f"leiden_sub_{res}_merged"].value_counts())
    # # create a UMAP colored by different leiden resolutions
    # print(f"re-computed UMAP with sub-clustering for leiden_coarse={clust}")
    # sc.pl.umap(adata_sub, color=f"leiden_sub_{res}_merged")
    
print("ALL clusters are sub-clustered now")

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Step 2. Subcluste 4 "leiden_coarse" clusters - deep dive into these for the Figures

# %%
for clust in coarse_clusts:
    adata_sub = dict_adata_sub[clust]
    res = 0.7
    sc.pl.umap(adata_sub, color=f"leiden_sub_{res}_merged_renumbered",
               save=f"_peaks_leiden_coarse_{clust}_subclust.png")

# %%
adata_sub = dict_adata_sub[13]
sc.pl.umap(adata_sub, color="leiden_sub_0.7_merged_renumbered", legend_loc="on data")
sc.pl.embedding(adata_sub, basis ="X_umap_global", 
                color="leiden_sub_0.7_merged_renumbered", legend_loc="on data")

# %%
for clust in coarse_clusts:
    adata_sub = dict_adata_sub[clust]
    res = 0.7
    sc.pl.embedding(adata_sub, basis="X_umap_global", color=f"leiden_sub_{res}_merged_renumbered",
                    save=f"_peaks_leiden_coarse_{clust}_subclust.png")

# %%
for clust in coarse_clusts:
    adata_sub = dict_adata_sub[clust]
    res = 0.7
    sc.pl.umap(adata_sub, color="timepoint", save=f"_peaks_leiden_coarse_{clust}_timepoint.png")

# %%
# check the top 5 enriched celltypes for each "leiden_coarse" cluster
for clust in coarse_clusts:
    adata_sub = dict_adata_sub[clust]
    print(f"leiden_coarse cluster {clust}")
    print(adata_sub.obs["celltype"].value_counts()[0:5])


# %%

# %%
adata_sub = dict_adata_sub[1]

sc.pl.umap(adata_sub, color="leiden_sub_0.7_merged_renumbered", legend_loc="on data")

for subclust in adata_sub.obs["leiden_sub_0.7_merged_renumbered"].unique():
    print(f"subcluster {subclust}")
    print(adata_sub[adata_sub.obs["leiden_sub_0.7_merged_renumbered"]==subclust].obs["celltype"].value_counts()[0:3])

# %%
adata_sub = dict_adata_sub[13]

sc.pl.umap(adata_sub, color="leiden_sub_0.7_merged_renumbered", legend_loc="on data")

for subclust in adata_sub.obs["leiden_sub_0.7_merged_renumbered"].unique():
    print(f"subcluster {subclust}")
    print(adata_sub[adata_sub.obs["leiden_sub_0.7_merged_renumbered"]==subclust].obs["celltype"].value_counts()[0:3])

# %%
for clust in coarse_clusts:
    adata_sub = dict_adata_sub[clust]
    res = 0.7
    sc.pl.umap(adata_sub, color="celltype", save=f"_peaks_leiden_coarse_{clust}_celltype.png")

# %%
# save the adata_sub objects
for clust in coarse_clusts:
    adata_sub = dict_adata_sub[clust]
    adata_sub.write_h5ad(filepath + f"peaks_leiden_coarse_cluster_{clust}_subset.h5ad")

# %%
# EDA on clusters enriched with specific celltypes
sc.pl.umap(adata_peaks, color="leiden_coarse", legend_loc="on data")

# %%
# Quick check for highly enriched clusters
result = adata_peaks.obs.groupby('leiden_coarse')['celltype'].apply(
    lambda x: (x == 'hematopoietic_vasculature').mean() * 100
).sort_values(ascending=False)

print("Hematopoietic vasculature percentage by cluster:")
print(result)

highly_enriched = result[result > 50]
if len(highly_enriched) > 0:
    print(f"\nClusters with >50%: {list(highly_enriched.index)}")
else:
    print(f"\nNo clusters have >50%. Highest: {result.iloc[0]:.2f}% in cluster {result.index[0]}")

# %%
# Quick check for highly enriched clusters
celltype = "tail_bud" # "hematopoetic_vasculature"
result = adata_peaks.obs.groupby('leiden_coarse')['celltype'].apply(
    lambda x: (x == celltype).mean() * 100
).sort_values(ascending=False)

print(f"{celltype} percentage by cluster:")
print(result)

highly_enriched = result[result > 50]
if len(highly_enriched) > 0:
    print(f"\nClusters with >50%: {list(highly_enriched.index)}")
else:
    print(f"\nNo clusters have >50%. Highest: {result.iloc[0]:.2f}% in cluster {result.index[0]}")

# %%
sc.pl.umap(adata_peaks[adata_peaks.obs["leiden_coarse"].isin([29,31])], color=["celltype", "timepoint"], ncols=1)

# %%
sc.pl.umap(adata_peaks[adata_peaks.obs["leiden_coarse"]==25], color=["celltype", "timepoint"], ncols=1)

# %%
# subset and re-cluster the hemangioblast & hematopoetic vasculature cluster
clust = 25
# subset first
adata_sub = adata_peaks[adata_peaks.obs["leiden_coarse"]==clust].copy()
# move the global UMAP coordinates
adata_sub.obsm["X_umap_global"] = adata_sub.obsm["X_umap"]

# recompute the UMAP
rsc.pp.scale(adata_sub)
rsc.pp.pca(adata_sub, n_comps=50, use_highly_variable=False)
rsc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=40)
rsc.tl.umap(adata_sub, random_state=42, min_dist=0.3)

# define the resolution and compute the leiden clustering (sub-clustering)
list_leiden_sub_res = []

res = 0.7
# sub-cluster using the res
rsc.tl.leiden(adata_sub, resolution=res, key_added=f"leiden_sub_{res}")
# save the key_added in a list
# num_clusts = len(adata_sub.obs[f"leiden_sub_{res}"].unique())
# print(f"leiden resolution:{res} gave {num_clusts} clusters")
# merge the small clusters than have fewer than 100 entries (peaks)
adata_sub = merge_small_clusters(
                adata_sub,
                leiden_key= f"leiden_sub_{res}",
                adjacency_key= "connectivities",
                merged_key_suffix = "merged",
                threshold = 100,
                merge_small_clusters= True,
                random_seed= 1)
    
# Renumber clusters to be consecutive (0, 1, 2, 3, ...)
merged_key = f"leiden_sub_{res}_merged"

# Get unique cluster labels and sort them
unique_clusters = sorted(adata_sub.obs[merged_key].unique())

# Create mapping from old labels to new consecutive labels
cluster_mapping = {old_label: str(i) for i, old_label in enumerate(unique_clusters)}

print(f"Cluster renaming mapping: {cluster_mapping}")

# Apply the renumbering
adata_sub.obs[f"{merged_key}_renumbered"] = adata_sub.obs[merged_key].map(cluster_mapping).astype('category')

print(adata_sub.obs[f"{merged_key}_renumbered"].value_counts())
# create a UMAP colored by different leiden resolutions
print(f"re-computed UMAP with sub-clustering for leiden_coarse={clust}")
sc.pl.umap(adata_sub, color=[f"{merged_key}_renumbered", "celltype","timepoint"])

# # save the adata_sub
# dict_adata_sub[clust] = adata_sub

# %% [markdown]
# ### REPEAT the above procedure for all "leiden_coarse" clusters
#
#

# %%

# %% [markdown]
# ## RESUMPTION FROM HERE

# %%
# RESUMPTION FROM HERE

# filepath for the adata_sub objects
filepath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/adata_peak_clusters_subset/'
# "leiden_coarse" clusters
coarse_clusts = [1,7,13,22]
# define an empty dictionary
dict_adata_sub = {}

# load the adata_sub objects and add them into the dictionary
for clust in coarse_clusts:
    adata_sub = sc.read_h5ad(filepath + f"peaks_leiden_coarse_cluster_{clust}_subset.h5ad")
    dict_adata_sub[clust] = adata_sub
    print(f"imported cluster {clust}")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## [gene association strategy]
# - "linked_gene", "overlap", "nearest_gene"

# %%
# compute the associated genes for peaks to all your sub-clustered data (using linked and overlap)
for clust in coarse_clusts:
    print(f"Processing gene associations for coarse cluster {clust}...")
    
    adata_sub = dict_adata_sub[clust]
    adata_sub = create_gene_associations(adata_sub)
    
    # Update the dictionary
    dict_adata_sub[clust] = adata_sub
    
    # Print summary statistics
    print(f"  Association type distribution:")
    print(f"    {adata_sub.obs['association_type'].value_counts()}")
    
    # Show some examples
    print(f"  Examples of associations:")
    sample_data = adata_sub.obs[['linked_gene', 'gene_body_overlaps', 'nearest_gene', 
                                'associated_gene', 'association_type']].head(10)
    print(sample_data)
    print("-" * 80)

print("Gene association processing complete!")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_gene_associations(adata_sub, subclust_key):
    """
    Analyze different gene association strategies for all subclusters
    """
    results = []
    
    for subclust in sorted(adata_sub.obs[subclust_key].unique()):
        peaks_sub = adata_sub[adata_sub.obs[subclust_key] == subclust]
        n_peaks = len(peaks_sub)
        
        # Strategy 1: Linked genes only
        linked_genes = peaks_sub.obs['linked_gene'].dropna().unique()
        linked_clean = [g for g in linked_genes if isinstance(g, str) and g.strip() != '' and '/' not in g]
        
        # Strategy 2: Linked + Overlap genes
        linked_overlap_mask = peaks_sub.obs['association_type'].isin(['linked', 'overlap'])
        linked_overlap_peaks = peaks_sub[linked_overlap_mask]
        linked_overlap_genes = linked_overlap_peaks.obs['associated_gene'].dropna().unique()
        linked_overlap_clean = [g for g in linked_overlap_genes if isinstance(g, str) and g.strip() != '' and '/' not in g]
        
        # Strategy 3: All associations (linked + overlap + nearest)
        all_genes = peaks_sub.obs['associated_gene'].dropna().unique()
        all_clean = [g for g in all_genes if isinstance(g, str) and g.strip() != '' and '/' not in g]
        
        # Count peaks with each association type
        assoc_counts = peaks_sub.obs['association_type'].value_counts()
        
        results.append({
            'subcluster': subclust,
            'n_peaks': n_peaks,
            'linked_genes': len(linked_clean),
            'linked_overlap_genes': len(linked_overlap_clean),
            'all_genes': len(all_clean),
            'linked_peaks': assoc_counts.get('linked', 0),
            'overlap_peaks': assoc_counts.get('overlap', 0),
            'nearest_peaks': assoc_counts.get('nearest', 0),
            'none_peaks': assoc_counts.get('none', 0),
            'linked_gene_list': linked_clean,
            'linked_overlap_gene_list': linked_overlap_clean,
            'all_gene_list': all_clean
        })
    
    return pd.DataFrame(results)

# Analyze all coarse clusters
print("GENE ASSOCIATION STRATEGY ANALYSIS")
print("=" * 80)

all_results = {}
summary_stats = []

for clust in coarse_clusts:
    print(f"\nCOARSE CLUSTER {clust}")
    print("-" * 50)
    
    adata_sub = dict_adata_sub[clust]
    subclust_key = "leiden_sub_0.7_merged_renumbered"
    
    # Analyze this coarse cluster
    df = analyze_gene_associations(adata_sub, subclust_key)
    all_results[clust] = df
    
    # Print detailed results
    print(f"{'Sub':>3} {'Peaks':>6} {'Linked':>7} {'L+O':>7} {'All':>7} {'L%':>6} {'O%':>6} {'N%':>6} {'None%':>6}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        linked_pct = (row['linked_peaks'] / row['n_peaks'] * 100) if row['n_peaks'] > 0 else 0
        overlap_pct = (row['overlap_peaks'] / row['n_peaks'] * 100) if row['n_peaks'] > 0 else 0
        nearest_pct = (row['nearest_peaks'] / row['n_peaks'] * 100) if row['n_peaks'] > 0 else 0
        none_pct = (row['none_peaks'] / row['n_peaks'] * 100) if row['n_peaks'] > 0 else 0
        
        print(f"{str(row['subcluster']):>3} {row['n_peaks']:6d} {row['linked_genes']:7d} {row['linked_overlap_genes']:7d} {row['all_genes']:7d} "
              f"{linked_pct:5.1f} {overlap_pct:5.1f} {nearest_pct:5.1f} {none_pct:5.1f}")
        
        # Add to summary
        summary_stats.append({
            'coarse_cluster': clust,
            'subcluster': row['subcluster'],
            'n_peaks': row['n_peaks'],
            'linked_genes': row['linked_genes'],
            'linked_overlap_genes': row['linked_overlap_genes'],
            'all_genes': row['all_genes'],
            'linked_pct': linked_pct,
            'overlap_pct': overlap_pct,
            'nearest_pct': nearest_pct,
            'none_pct': none_pct
        })

# Overall summary statistics
summary_df = pd.DataFrame(summary_stats)

print(f"\n" + "=" * 80)
print("OVERALL SUMMARY STATISTICS")
print("=" * 80)

print(f"Total subclusters analyzed: {len(summary_df)}")

# Gene count statistics
strategies = ['linked_genes', 'linked_overlap_genes', 'all_genes']
strategy_names = ['Linked Only', 'Linked + Overlap', 'All Associations']

print(f"\nGENE COUNT STATISTICS:")
print(f"{'Strategy':<20} {'Mean':<8} {'Median':<8} {'Min':<6} {'Max':<8} {'Zero%':<8}")
print("-" * 70)

for strategy, name in zip(strategies, strategy_names):
    mean_val = summary_df[strategy].mean()
    median_val = summary_df[strategy].median()
    min_val = summary_df[strategy].min()
    max_val = summary_df[strategy].max()
    zero_pct = (summary_df[strategy] == 0).mean() * 100
    
    print(f"{name:<20} {mean_val:<8.1f} {median_val:<8.1f} {min_val:<6d} {max_val:<8d} {zero_pct:<8.1f}")

# Peak association statistics
print(f"\nPEAK ASSOCIATION TYPE DISTRIBUTION:")
print(f"{'Type':<15} {'Mean %':<8} {'Median %':<10} {'Min %':<8} {'Max %':<8}")
print("-" * 60)

assoc_types = ['linked_pct', 'overlap_pct', 'nearest_pct', 'none_pct']
assoc_names = ['Linked', 'Overlap', 'Nearest', 'None']

for assoc_type, name in zip(assoc_types, assoc_names):
    mean_pct = summary_df[assoc_type].mean()
    median_pct = summary_df[assoc_type].median()
    min_pct = summary_df[assoc_type].min()
    max_pct = summary_df[assoc_type].max()
    
    print(f"{name:<15} {mean_pct:<8.1f} {median_pct:<10.1f} {min_pct:<8.1f} {max_pct:<8.1f}")

# Identify problematic cases
print(f"\n" + "=" * 80)
print("PROBLEMATIC CASES ANALYSIS")
print("=" * 80)

# Clusters with no linked genes
no_linked = summary_df[summary_df['linked_genes'] == 0]
print(f"Subclusters with NO linked genes: {len(no_linked)}/{len(summary_df)} ({len(no_linked)/len(summary_df)*100:.1f}%)")
if len(no_linked) > 0:
    print("Examples:")
    for _, row in no_linked.head(10).iterrows():
        print(f"  Coarse {row['coarse_cluster']}, Sub {row['subcluster']}: {row['n_peaks']} peaks, "
              f"{row['linked_overlap_genes']} L+O genes, {row['all_genes']} all genes")

# Clusters with very few linked genes but many total genes
few_linked_many_total = summary_df[(summary_df['linked_genes'] < 10) & (summary_df['all_genes'] > 1000)]
print(f"\nSubclusters with <10 linked but >1000 total genes: {len(few_linked_many_total)}")
if len(few_linked_many_total) > 0:
    print("Examples:")
    for _, row in few_linked_many_total.head(5).iterrows():
        print(f"  Coarse {row['coarse_cluster']}, Sub {row['subcluster']}: "
              f"{row['linked_genes']} linked, {row['all_genes']} total")

# Show the problematic cluster (7,4) specifically
prob_cluster = summary_df[(summary_df['coarse_cluster'] == 7) & (summary_df['subcluster'] == 4)]
if not prob_cluster.empty:
    row = prob_cluster.iloc[0]
    print(f"\nProblematic cluster (Coarse 7, Sub 4):")
    print(f"  Peaks: {row['n_peaks']}")
    print(f"  Linked genes: {row['linked_genes']}")
    print(f"  Linked + Overlap genes: {row['linked_overlap_genes']}")
    print(f"  All genes: {row['all_genes']}")
    print(f"  Association distribution: {row['linked_pct']:.1f}% linked, {row['overlap_pct']:.1f}% overlap, "
          f"{row['nearest_pct']:.1f}% nearest, {row['none_pct']:.1f}% none")

print(f"\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("1. LINKED ONLY:")
print(f"   - Pros: Highest confidence associations (correlation-based)")
print(f"   - Cons: {(summary_df['linked_genes'] == 0).sum()} clusters have no genes")
print(f"   - Cons: Very small gene lists (median: {summary_df['linked_genes'].median():.0f} genes)")

print(f"\n2. LINKED + OVERLAP:")
print(f"   - Pros: Good compromise between quality and quantity")
print(f"   - Pros: Only {(summary_df['linked_overlap_genes'] == 0).sum()} clusters have no genes") 
print(f"   - Pros: Reasonable gene lists (median: {summary_df['linked_overlap_genes'].median():.0f} genes)")

print(f"\n3. ALL ASSOCIATIONS:")
print(f"   - Pros: No clusters without genes")
print(f"   - Cons: Very large gene lists (median: {summary_df['all_genes'].median():.0f} genes)")
print(f"   - Cons: Many 'nearest' associations may be spurious")

print(f"\nSUGGESTED APPROACH:")
print(f"Use LINKED + OVERLAP strategy with adaptive filtering:")
print(f"- Prioritize linked genes when available")
print(f"- Include overlap genes to maintain reasonable power")
print(f"- Skip extremely large clusters (>2000 genes)")
print(f"- This gives you biological relevance with statistical power")

print("=" * 80)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Step 2-1. exploratory visualization + pathway enrichment
#
# - sub-clustered UMAPs - both global and local UMAP coordinates
# - export the gene lists from these clusters for FishEnrichR (use **"associated_gene"**)

# %%
# Load the FishEnrichR utility module
sys.path.append('/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/utils/')
from FishEnrichR import *


# %%
# a function to filter out genes with non-canonical names (which will error for FishEnrichR)
# genes with "/" will cause an error
def clean_and_validate_genes(gene_list, max_genes=10000):
    """
    Clean and validate gene list for EnrichR submission
    """
    # Remove nulls and convert to strings
    cleaned = [str(g).strip() for g in gene_list if pd.notna(g) and str(g).strip() != '']
    
    # Remove common problematic patterns
    valid_genes = []
    for gene in cleaned:
        # Skip obviously invalid entries
        if gene.lower() in ['nan', 'none', 'null', '', 'na']:
            continue
        
        # Remove genes with problematic characters
        if any(char in gene for char in [',', ';', '\t', '\n', '|', '/']):
            continue
            
        # Skip very long names (likely not real gene names)
        if len(gene) > 50:
            continue
            
        # Basic gene name validation (adjust pattern for your organism)
        if re.match(r'^[A-Za-z0-9][A-Za-z0-9\-_.:\(\)]*[A-Za-z0-9]$|^[A-Za-z0-9]$', gene):
            valid_genes.append(gene)
    
    # Remove duplicates while preserving order
    unique_genes = list(dict.fromkeys(valid_genes))
    
    # Limit size if too large (EnrichR might have limits)
    if len(unique_genes) > max_genes:
        print(f"    Warning: Gene list too large ({len(unique_genes)}), truncating to {max_genes}")
        unique_genes = unique_genes[:max_genes]
    
    return unique_genes


# %%
# For each leiden_coarse cluster, we will initiliaze and compute the FishEnrichR object
for clust in coarse_clusts:
    print(f"Processing coarse cluster {clust}")
    
    # create the output directory for the GSEA results
    output_dir = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/FishEnrichR_leiden_coarse_cluster_{clust}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the FishEnrichR client
    enrichr = FishEnrichR()
    
    # load the adata_sub
    adata_sub = dict_adata_sub[clust]
    
    # pick the sub-clusters and compute the enriched pathways
    # Loop over unique cluster labels
    subclust_key = "leiden_sub_0.7_merged_renumbered"
    for subclust in adata_sub.obs[subclust_key].unique():
        print(f"Processing subcluster {subclust}...")

        # Subset the AnnData to the cluster of interest
        peaks_sub = adata_sub[adata_sub.obs[subclust_key] == subclust]

        # Extract the list of genes from obs["associated_gene"] in this cluster
        # gene_annotation_method = "associated_gene"
        # print(f"Using {gene_annotation_method} to annotate the peaks")
        list_genes_sub = peaks_sub.obs["associated_gene"].unique().tolist()
        # Remove any NaN or non-string entries
        list_genes_sub = [g for g in list_genes_sub if isinstance(g, str)]
        # Clean up non-canonical gene names
        list_genes_sub_cleaned = clean_and_validate_genes(list_genes_sub)

        # If no genes, skip
        if not list_genes_sub:
            print(f"No genes found for cluster {subclust}; skipping.")
            continue

        # Analyze the gene list with FishEnrichR (you can change the "description" string)
        result = enrichr.analyze_genes(list_genes_sub_cleaned, description=f"Cluster_{subclust}_genes")

        # Retrieve the userListId
        user_list_id = result["userListId"]

        # Build an output file path (e.g. cluster_2_enrichment_WikiPathways_2018.tsv)
        output_filename = f"cluster_{subclust}_enrichment_WikiPathways_2018.tsv"
        output_path = os.path.join(output_dir, output_filename)

        # Download the WikiPathways_2018 results
        enrichr.download_results(user_list_id, output_path, 'WikiPathways_2018')

        print(f"  Saved WikiPathways_2018 enrichment for cluster {subclust} to:\n  {output_path}")
        
    print(f"coarse cluster {clust} is processed for enriched pathways")


# %%
# Debug the problematic subcluster
clust = 7
subclust = "4"

print(f"Debugging coarse cluster {clust}, subcluster {subclust}")
print("=" * 60)

# Get the data
adata_sub = dict_adata_sub[clust]
subclust_key = "leiden_sub_0.7_merged_renumbered"

# Subset to the problematic cluster
peaks_sub = adata_sub[adata_sub.obs[subclust_key] == subclust]

print(f"Number of peaks in subcluster {subclust}: {len(peaks_sub)}")

# Examine the associated genes
associated_genes = peaks_sub.obs["associated_gene"]
print(f"Raw associated_gene values:")
print(f"  Total entries: {len(associated_genes)}")
print(f"  Non-null entries: {associated_genes.notna().sum()}")
print(f"  Null entries: {associated_genes.isna().sum()}")

# Get unique genes (before cleaning)
print(f"\nUnique raw values (first 20):")
unique_raw = associated_genes.unique()
for i, gene in enumerate(unique_raw[:20]):
    print(f"  {i+1:2d}. '{gene}' (type: {type(gene)})")

# Apply the same cleaning as in your script
list_genes_sub = peaks_sub.obs["associated_gene"].dropna().unique().tolist()
list_genes_sub = [g for g in list_genes_sub if isinstance(g, str) and g.strip() != '']

print(f"\nAfter cleaning:")
print(f"  Clean gene count: {len(list_genes_sub)}")
print(f"  First 20 clean genes: {list_genes_sub[:20]}")

# Check for potential problematic characters or patterns
problematic_genes = []
for gene in list_genes_sub:
    # Check for various issues
    issues = []
    if ',' in gene:
        issues.append("contains comma")
    if ';' in gene:
        issues.append("contains semicolon")
    if '\t' in gene:
        issues.append("contains tab")
    if '\n' in gene:
        issues.append("contains newline")
    if len(gene) > 50:
        issues.append("very long name")
    if not gene.replace('-', '').replace('_', '').replace('.', '').replace(':', '').isalnum():
        issues.append("contains special characters")
    if gene.lower() in ['nan', 'none', 'null', '']:
        issues.append("null-like value")
    
    if issues:
        problematic_genes.append((gene, issues))

if problematic_genes:
    print(f"\nPotentially problematic genes ({len(problematic_genes)} found):")
    for gene, issues in problematic_genes[:10]:  # Show first 10
        print(f"  '{gene}' - Issues: {', '.join(issues)}")
else:
    print(f"\nNo obviously problematic genes detected")

# Check association types
print(f"\nAssociation type distribution for this subcluster:")
print(peaks_sub.obs['association_type'].value_counts())

# Check for very long gene lists (EnrichR might have limits)
print(f"\nGene list statistics:")
print(f"  Total unique genes: {len(list_genes_sub)}")
print(f"  Longest gene name: {max(len(g) for g in list_genes_sub) if list_genes_sub else 0}")
print(f"  Shortest gene name: {min(len(g) for g in list_genes_sub) if list_genes_sub else 0}")

# Try a smaller subset to see if it's a size issue
if len(list_genes_sub) > 100:
    print(f"\nTesting with smaller gene list (first 50 genes)...")
    small_gene_list = list_genes_sub[:50]
    print(f"Small list: {small_gene_list}")

# Show some examples of the original data for this subcluster
print(f"\nOriginal data examples for problematic subcluster:")
example_data = peaks_sub.obs[['linked_gene', 'gene_body_overlaps', 'nearest_gene', 
                             'associated_gene', 'association_type']].head(10)
print(example_data)

print("\n" + "=" * 60)
print("Debug complete. Check the output above for issues.")

# Additional check: try to identify genes that might not be valid
print(f"\nChecking for non-standard gene names...")
import re

# Common gene name patterns (adjust for your organism)
valid_gene_pattern = re.compile(r'^[A-Za-z0-9][A-Za-z0-9\-_.]*[A-Za-z0-9]$|^[A-Za-z]$')
invalid_genes = [gene for gene in list_genes_sub if not valid_gene_pattern.match(gene)]

if invalid_genes:
    print(f"Potentially invalid gene names ({len(invalid_genes)} found):")
    for gene in invalid_genes[:10]:
        print(f"  '{gene}'")
else:
    print("All genes appear to have valid naming patterns")

# Check for duplicates (shouldn't happen with unique(), but worth checking)
if len(list_genes_sub) != len(set(list_genes_sub)):
    print(f"Warning: Found duplicate genes after cleaning!")
    from collections import Counter
    gene_counts = Counter(list_genes_sub)
    duplicates = {gene: count for gene, count in gene_counts.items() if count > 1}
    print(f"Duplicates: {duplicates}")
else:
    print("No duplicate genes found")

# %% [markdown]
# ### plotting the enrichment result using a bar plot

# %%
subcluster_colors

# %%
adata_sub.obs["leiden_sub_0.7_merged"].value_counts()

# %%
adata_sub = dict_adata_sub[1]
sc.pl.umap(adata_sub, color="leiden_sub_0.7_merged_renumbered", legend_loc="on data")

# %%
# Choose which coarse cluster to analyze
target_coarse_cluster = 1  # Change this to analyze different coarse clusters

# Get the adata_sub for this coarse cluster
adata_sub = dict_adata_sub[target_coarse_cluster]

# Use the renumbered clustering for consistency
subcluster_key = "leiden_sub_0.7_merged_renumbered"  # Use renumbered version
# unique_subclusters = sorted(adata_sub.obs[subcluster_key].unique())
# Get unique subclusters and sort them numerically
unique_subclusters = adata_sub.obs[subcluster_key].unique()
# Force consecutive ordering: 0, 1, 2, 3, 4, 5, etc.
unique_subclusters = [str(i) for i in np.arange(0, len(unique_subclusters))]

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
for key in possible_color_keys:
    if key in adata_sub.uns:
        subcluster_colors_list = adata_sub.uns[key]
        print(f"Found colors using key: {key}")
        break

# If no colors found, generate them
if subcluster_colors_list is None:
    print("No colors found in uns, generating new colors...")
    import matplotlib.cm as cm
    import numpy as np
    n_clusters = len(unique_subclusters)
    colors = cm.tab20(np.linspace(0, 1, n_clusters))
    subcluster_colors_list = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r,g,b,a in colors]

# Create color mapping
subcluster_labels = adata_sub.obs[subcluster_key].cat.categories
subcluster_colors = {str(cluster): color for cluster, color in zip(subcluster_labels, subcluster_colors_list)}

print(f"Color mapping: {subcluster_colors}")

# Plot parameters
TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 10

nrows = 4
ncols = 5
top_n = 10

# Define base directory and output path
# Use the linked_overlap directory if that's where your new results are
input_dir = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/FishEnrichR_leiden_coarse_cluster_{target_coarse_cluster}_linked_overlap/"

# Fallback to original directory if linked_overlap doesn't exist
if not os.path.exists(input_dir):
    input_dir = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/FishEnrichR_leiden_coarse_cluster_{target_coarse_cluster}/"

print(f"Using input directory: {input_dir}")

output_fig = os.path.join(input_dir, f"leiden_coarse_{target_coarse_cluster}_subclusts_pathways_colored.pdf")

# Define a "non-significant" color
nonsig_color = '#F5F5DC'

# Set up the figure
width_scale = 0.70
fig_width = 6.5 * ncols * width_scale
fig_height = 3.5 * nrows
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), constrained_layout=True)

# Flatten axes array for easy indexing
axes = axes.flatten()

# Track which files are found
files_found = 0
files_missing = 0

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
        ax = axes[i]
        ax.set_title(f'Cluster {cluster} (No data)', fontsize=TITLE_SIZE, pad=5)
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
        # print(f"Cluster {cluster}: using color {cluster_color}")

        # Determine colors based on p-value
        colors = [
            cluster_color if p <= 0.05 else nonsig_color
            for p in df['P-value']
        ]

        # Select the correct subplot
        ax = axes[i]

        # Remove grid lines inside the bar plots
        ax.grid(False)

        # Create horizontal bars
        bars = ax.barh(range(len(df)), df['Combined Score'], color=colors, alpha=0.7)

        # Title
        ax.set_title(f'Cluster {cluster}', fontsize=TITLE_SIZE, pad=5)

        # Remove y-ticks and y-axis labels completely
        ax.set_yticks([])
        ax.set_yticklabels([])

        # Add labels **inside** the bars
        for bar, label in zip(bars, df['Term']):
            ax.text(
                bar.get_width() * 0.02,  # Small left margin inside the bar
                bar.get_y() + bar.get_height() / 2,
                label.split('_WP')[0],  # Remove '_WP' suffix
                va='center', ha='left', fontsize=TICK_SIZE, color='black', fontweight='bold'
            )

        # Format x-axis
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.set_xlabel('Combined Score', fontsize=LABEL_SIZE)
        
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
fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=LEGEND_SIZE)

plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"Plot saved to: {output_fig}")

# # Debug: Show the actual color mapping being used
# print(f"\nDEBUG: Color mapping being used:")
# for cluster in unique_subclusters:
#     color = subcluster_colors.get(str(cluster), 'NOT FOUND')
#     print(f"  Cluster {cluster}: {color}")

# %% [markdown]
# ### Now, let's use the module/function from the above exercise

# %%
# import the plotting module for the pathway enrichment analysis results
from module_pathway_plotting import *

# %%
for coarse_clust in coarse_clusts:
    plot_subcluster_pathways(dict_adata_sub, target_coarse_cluster=coarse_clust,
                            subcluster_key="leiden_sub_0.7_merged_renumbered",
                            base_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis",
                            use_linked_overlap=False)

# %%

# %%

# %% [markdown]
# ## Step 3. Repeat the process for all "leiden_coarse" clusters
# - 1) sub-clusterin for each leiden_coarse cluster, save the adata_sub object
# - 2) Run FishEnrichR for subclusters, generating the summary plot
# - 3) Export the sub-cluster:enriched pathways as a dataframe, such that we can feed into the LLM (ChatGPT/Gemini)

# %%

# %% [markdown]
# ### export peaks with cluster labels as txt format

# %%
# load the utility functions for exporting the peaks:clusters df to txt format
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs/")
from utils_maelstrom import export_peaks_for_gimmemotifs

# %%
import subprocess

# %%
clust_label = f"leiden_sub_0.7_merged_renumbered"

for clust in np.arange(0,36):
    # extract the subsetted adata object
    adata_sub = dict_adata_sub[clust]
    
    # create a directory
    # os.makedirs(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/adata_peak_clusters_subset/", exist_ok=True)
    # export the peaks with the labels
    export_peaks_for_gimmemotifs(adata_sub, cluster=clust_label, 
                                 out_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/adata_peak_clusters_subset/",
                                 out_name= f"leiden_coarse_{clust}_subclust")

# %% [markdown]
# ### Run GimmeMotifs maelstrom for differential motif computation
# - This is done on HPC using Slurm
# - reference: https://gimmemotifs.readthedocs.io/en/master/tutorials.html#find-differential-motifs
#
# - We will run gimme maelstrom for each "leiden_coarse" cluster, for their sub-clusters.

# %%
for clust in np.arange(0,36):
    

# %%
# submit a batch slurm job for all "leiden_coarse" clusters
SLURM_DIR = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/" \
            "zebrahub-multiome-analysis/scripts/slurm_scripts"
SCRIPT = f"{SLURM_DIR}/gimme_maelstrom_modular.sh"

for clust in range(36):
    input_file = ("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/"
                  "data/processed_data/13_peak_umap_analysis/adata_peak_clusters_subset/"
                  f"peaks_leiden_coarse_{clust}_subclust.txt")
    output_dir = ("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/"
                  "data/processed_data/13_peak_umap_analysis/"
                  f"maelstrom_leiden_coarse_{clust}_cisBP_v2_output/")

    # !sbatch {SCRIPT} \
#         --input {input_file} \
#         --ref_genome danRer11 \
#         --output_dir {output_dir} \
#         --pfmfile CisBP_ver2_Danio_rerio

# %%

# %%

# %%

# %%
