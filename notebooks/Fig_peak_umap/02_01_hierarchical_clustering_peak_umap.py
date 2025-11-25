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
# ## Hirarchical_clustering of peak UMAP
#
# - Last updated: 04/21/2025
# - Inspired by Cytoself and SubCell papers, we will perform hierarchical clustering on the peak UMAP to find the meaningful "resolution" of peak groups
# - tissue-level (or dev stages)
# - biological pathways
# - gene-gene interaction? (ligand-receptor interaction?)
# - TF motif enrichments
#
# ## EDA1: Leiden clustering
# - For each leiden cluster (leiden resolution of XX), we wanted to know how we can annotate these.
# - They are NOT clustered by the chromosome, peak type, etc., rather the set of peaks that are co-regulated for specific celltype&timepoint.
# - 1) So, what we can do is, for each leiden cluster, get the list of "associated genes", and compute the enriched pathways
# - 2) For each peak cluster, compute the over-represented "motifs", and see if any of those match with known TF's PWM/PFM
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
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_pseudobulk_leiden_0.4/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated.h5ad")
adata_peaks

# %%
adata_peaks.obs.head()

# %%
# move the .X to GPU (for faster operation)
rsc.get.anndata_to_GPU(adata_peaks)

# %%
# compute ladata_peaksen clustering with different resolutions
list_res_leiden = [0.2, 0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.5, 5]
for res in list_res_leiden:
    rsc.tl.leiden(adata_peaks, resolution=res, key_added=f"leiden_{res}")
    # check if there's any singletons
    # Count how many cells are in each cluster
    cluster_counts = adata_peaks.obs[f"leiden_{res}"].value_counts()
    # Check if there's any singleton (a cluster with exactly one cell)
    singletons = cluster_counts[cluster_counts == 1]
    
    if len(singletons) > 0:
        print(f"Resolution {res}: Found singleton cluster(s) -> {singletons.index.tolist()}")
    else:
        print(f"Resolution {res}: No singletons found.")
    
    # group_singletons_seurat_style(adata_peaks_filt, leiden_key = f"leiden_{res}")
    print(f"leiden clustering for {res} is done")

# %%
list_res_leiden = [0.2, 0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.5, 5]
for res in list_res_leiden:
    n_clusters = len(adata_peaks.obs[f"leiden_{res}"].unique())
    print(f"the number of clusters in {res} is: {n_clusters}")

# %%
# plot the peak UMAPs colored by the leiden clustering
sc.pl.umap(adata_peaks, color=["leiden_0.2", "leiden_0.5", 
                                    "leiden_0.7","leiden_1",
                                    "leiden_1.5","leiden_5"], ncols=3, legend_loc=None)

# %%
## plot the histogram for the number of peaks per cluster distribution
list_res_leiden = [0.2, 0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.5, 5]
bin_edges = np.arange(0, 210000, 1000)

for res in list_res_leiden:
    cluster_counts = adata_peaks.obs[f"leiden_{res}"].value_counts()
    # adata_peaks.obs[f"leiden_{res}"].value_counts().hist(alpha=0.5, bins=40, density=True)
    # Plot the histogram + KDE together
    sns.histplot(
        cluster_counts,
        bins=40,
        kde=True,         # Enables KDE
        kde_kws={"cut": 0},  # Keeps the KDE from extending beyond data
        stat="density",   # Normalizes the histogram
        alpha=0.3,
        label=str(res),
    )
plt.xlabel("number of peaks (per cluster)")
plt.ylabel("density")
plt.legend(list_res_leiden)
plt.grid(False)
plt.show()


# %%
adata_peaks.obs["leiden_1.5"].value_counts()
adata_peaks.obs["leiden_1.5"].value_counts().median()

# %% [markdown]
# ### we will choose the leiden clustering resolution of 1.5 as the "coarse" clustering resolution
#

# %% [markdown]
# ## computing the 3D UMAP - to visualize the leiden clusters
#

# %%
# first, copy over the 2D UMAP
adata_peaks.obsm["X_umap_2D"] = adata_peaks.obsm["X_umap"]


# %%
rsc.tl.umap(adata_peaks, min_dist=0.3, n_components=3)

# %%
umap_3d_array = adata_peaks.obsm["X_umap"]
umap_3d_array

# %%
adata_peaks.obsm["X_umap_3D"] = umap_3d_array
adata_peaks

# %%
# create a dataframe for plotting
umap_3d = pd.DataFrame(umap_3d_array, 
                       index=adata_peaks.obs_names,
                       columns=["UMAP_1","UMAP_2","UMAP_3"])
umap_3d.head()

# %%
# remove the blacklisted peaks (peaks that extends beyond the chromosome ends, and also MT chromosomes)
peaks_blacklisted = ["3-62628283-62628504", "10-45419551-45420917",
                     "MT-22-3567", "MT-13233-16532"]

# filter the adata object
adata_peaks_filt = adata_peaks[~adata_peaks.obs_names.isin(peaks_blacklisted)].copy()
adata_peaks_filt

# %%
# filter out the low_quality cells (using the index matching with adata.obs_names)
umap_3d = umap_3d[umap_3d.index.isin(adata_peaks_filt.obs_names)]
umap_3d

# %%
umap_3d["celltype"] = umap_3d.index.map(adata_peaks_filt.obs["celltype"])
umap_3d["timepoint"] = umap_3d.index.map(adata_peaks_filt.obs["timepoint"])
umap_3d["leiden_1.5"] = umap_3d.index.map(adata_peaks_filt.obs["leiden_1.5"])
umap_3d.head()

# %%
# Assuming your dataframe is named df
# umap_3d.rename(columns={'wnnUMAP3D_1': 'UMAP_1', 'wnnUMAP3D_2': 'UMAP_2', 'wnnUMAP3D_3': 'UMAP_3'}, inplace=True)

# Check the renamed dataframe
print(umap_3d.head())

# %%
# visualize the 3D UMAP using plotly
import plotly.express as px
fig = px.scatter_3d(umap_3d, x='UMAP_1', y='UMAP_2', z='UMAP_3', color='celltype', hover_data=['timepoint'])

# Show the figure
fig.update_traces(marker=dict(size=3))  # Change the size to 3 for all points
fig.show()

# %%
px.scatter_3d(umap_3d, x='UMAP_1', y='UMAP_2', z='UMAP_3', color='leiden_1.5', hover_data=['celltype','timepoint'])

# Show the figure
fig.update_traces(marker=dict(size=1))  # Change the size to 3 for all points
fig.show()

# %%
# Show the figure
fig.update_traces(marker=dict(size=0.1))  # Change the size to 3 for all points
fig.show()

# %% [markdown]
# ## save the object

# %%
# save the 3D UMAP as a csv file
umap_3d.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peak_umap_3d_annotated_v1.csv")

# %%
# save the h5ad objects
adata_peaks.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated.h5ad")
adata_peaks_filt.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")

# %%

# %% [markdown]
# ## Step 2. manual curation of clustering
# - 1) create a merge table for over-clustered branches
# - 2) sub-cluster each "branch" for exploration
#

# %%
sc.pl.umap(adata_peaks_filt, color="leiden_1.5", legend_loc="on data")

# %%
# visualize the subset of the clusters
list_leiden_clusts = ["48","9","40","32","24","12","18","14","9","38"]
sc.pl.umap(adata_peaks_filt[adata_peaks_filt.obs["leiden_1.5"].isin(list_leiden_clusts)], color="leiden_1.5",
           legend_loc="on data")

# %%
list_leiden_clusts = ["9","41"]
sc.pl.umap(adata_peaks_filt[adata_peaks_filt.obs["leiden_1.5"].isin(list_leiden_clusts)], color="leiden_1.5",
           legend_loc="on data")

# %%
# create a merge table - by manually looking at the clusters in both 2D and 3D
merge_table = {"34":"34",
               "37":"34",
               "10":"10",
               "43":"10",
               "6":"6",
               "35":"6",
               "1":"1",
               # "12":"1",
               "21":"1",
               # "26":"1",
               "47":"1",
               "3":"3",
               "31":"3",
               "9":"9",
               "41":"9",
               "15":"15",
               "25":"15",
               "28":"15",
               "29":"15",
               "30":"15",
               "45":"15",
               "46":"15"}

# %%
# map the clusters in "leiden/hpc/projects/ "leiden_coarse" using the merge_table
# Create a leiden_coarse column based on the merge table
adata_peaks_filt.obs['leiden_coarse'] = adata_peaks_filt.obs['leiden_1.5'].astype(str).map(merge_table)

# For any clusters not in the merge table, keep their original ID
# This ensures we don't lose any clusters that weren't explicitly defined in the merge table
unmapped_clusters = set(adata_peaks_filt.obs['leiden_1.5'].astype(str)) - set(merge_table.keys())
for cluster in unmapped_clusters:
    # You can choose to either:
    # 1. Keep original IDs for unmapped clusters
    adata_peaks_filt.obs.loc[adata_peaks_filt.obs['leiden_1.5'].astype(str) == cluster, 'leiden_coarse'] = cluster

# Now renumber the clusters sequentially (no gaps)
# First, get the unique coarse cluster IDs
unique_coarse_clusters = sorted(adata_peaks_filt.obs['leiden_coarse'].unique())

# Create a mapping from the current coarse IDs to sequential numbers
sequential_mapping = {cluster: str(i) for i, cluster in enumerate(unique_coarse_clusters)}

# Apply the sequential mapping
adata_peaks_filt.obs['leiden_coarse_renumbered'] = adata_peaks_filt.obs['leiden_coarse'].map(sequential_mapping)

# Convert to categorical for better visualization and handling in scanpy
adata_peaks_filt.obs['leiden_coarse_renumbered'] = adata_peaks_filt.obs['leiden_coarse_renumbered'].astype('category')

# Print a summary of the new clustering
print(f"Original clusters: {len(adata_peaks_filt.obs['leiden_1.5'].unique())}")
print(f"Coarse clusters before renumbering: {len(adata_peaks_filt.obs['leiden_coarse'].unique())}")
print(f"Coarse clusters after renumbering: {len(adata_peaks_filt.obs['leiden_coarse_renumbered'].unique())}")

# Create a reference table showing the mapping from original to coarse to renumbered
mapping_df = pd.DataFrame({
    'original_cluster': list(merge_table.keys()),
    'coarse_cluster': list(merge_table.values()),
    'renumbered_cluster': [sequential_mapping[merge_table[k]] for k in merge_table.keys()]
}).drop_duplicates()

print("\nCluster mapping reference:")
print(mapping_df)

# You might want to save this mapping for future reference
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/"
mapping_df.to_csv('leiden_clusters_merged.csv', index=False)

# Optionally visualize the new clustering
import scanpy as sc
sc.pl.umap(adata_peaks_filt, color=['leiden_1.5', 'leiden_coarse', 'leiden_coarse_renumbered'], 
           ncols=3, title=['Original', 'Coarse', 'Renumbered'])


# %%
adata_peaks_filt.obs.head()

# %%
# export the csv file
adata_peaks_filt.obs.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/all_peaks_leiden_clusters.csv")
