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
# # Differential motif enrichment between peak clusters - fine sub-clusters
#
# - We want to analyze the TF motif enrichment profiles between the peak clusters in the peak UMAP.
# - Although we have not landed in the clustering resolutions for the sub-clusters, we'll start with the leiden resolution of 1.5 as the starting point.
#
# ## the gimme maelstrom workflow is the following:
# - 1) export the peaks:clusters dataframe as a txt file
# - 2) Run gimme maelstrom (this includes finding enrichment scores for the known TF motifs, then motif activity computation per "cluster"
# - NOTE. To repeat the above process, we'd need to repeat from the step 1.
#

# %% [markdown]
# ## Step 1. Load the adata objects and modules
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom
import sys
import os

from gimmemotifs import maelstrom

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# # GPU-acceleration (numpy, pandas, and sklearn)
# import cudf
# import cupy as cp
# # from cupyx.scipy.stats import hypergeom as cupy_hypergeom
# import cuml
# from cuml.ensemble import RandomForestClassifier as cuRF
# from statsmodels.stats.multitest import multipletests

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
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_umap_motifs_640K_peaks_leiden_coarse_subclust/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")
adata_peaks

# %% [markdown]
# ### Load the subsetted data objects (adata_sub)

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

# %% [markdown]
# ## Sanity check the data

# %%
# move the counts to GPU
rsc.get.anndata_to_GPU(adata_peaks)

# %%
# plot the coarse leiden clusters
sc.pl.embedding(adata_peaks, basis="X_umap_2D", color="leiden_coarse", save="_peaks_leiden_coarse.png")

# %%
# plot the coarse leiden clusters (with labels on top)
sc.pl.embedding(adata_peaks, basis="X_umap_2D", color="leiden_coarse", save="_peaks_leiden_coarse_labels_on_data.png", legend_loc="on data")

# %%

# %%

# %% [markdown]
# ## Step 2. Motif enrichment analysis (gimme maelstrom)
# - export the peaks as a txt file for gimmemotifs maelstrom
# - Run gimme maelstrom using slurm (on HPC)

# %% [markdown]
# ## export peaks with cluster labels as txt format

# %%
# load the utility functions for exporting the peaks:clusters df to txt format
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs/")
from utils_maelstrom import export_peaks_for_gimmemotifs

# %%
clust_label = f"leiden_sub_0.7_merged_renumbered"

for clust in coarse_clusts:
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
# !sbatch gimme_maelstrom_modular.sh --input /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/adata_peak_clusters_subset/peaks_leiden_coarse_1_subclust.txt --ref_genome danRer11 --output_dir /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_leiden_coarse_1_cisBP_v2_output/ --pfmfile CisBP_ver2_Danio_rerio

# %%

# %%
coarse_clusts

# %%
# ## Run the GimmeMotifs maelstrom using Slurm on HPC
# # !sbatch /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/slurm_scripts/gimme_maelstrom.sh

# %% [markdown]
# ## Step 3. Check the Maelstrom output:
#

# %%
# import the result of the maelstrom
from gimmemotifs.maelstrom import MaelstromResult

# %% [markdown]
# ### "leiden_coarse" cluster 13

# %%
# peaks-by-motifs count matrix
mr_13 = MaelstromResult("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_leiden_coarse_13_cisBP_v2_output/")
mr_13.scores.head()

# %%
# 1) Filter based on threshold
threshold = 3.5 # for example
df = mr_13.result.copy()

# 2) Keep only those rows where at least one absolute value >= threshold
mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]
# Example: remove "z-score " prefix
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]

# 3) RENAME Columns to factor names (similar to 'plot_heatmap(name=True)')
# ---------------------------------------------------------
df_filt_named = df_filt.copy()
factors = []
for motif_id in df_filt_named.index:
    # The same logic that mr.plot_heatmap() uses:
    # mr.motifs[motif_id].format_factors(...)
    motif_obj = mr.motifs.get(motif_id, None)
    if motif_obj is not None:
        # Example parameters (tweak as needed)
        factor_str = motif_obj.format_factors(
            max_length=3,        # how many factors to show
            html=False,
            include_indirect=True,
            extra_str="",
        )
    else:
        factor_str = motif_id  # fallback if motif is missing
    factors.append(factor_str)

# Put these factors in a new column, then set as index
df_filt_named["factors"] = factors
df_filt_named = df_filt_named.set_index("factors")


# Optional: check how many motifs remain
print(f"Number of motifs passing threshold: {df_filt.shape[0]}")

# transpose the matrix
df_filt_trans = df_filt_named.transpose()


# 4) CLUSTER + PLOT using seaborn.clustermap
# ---------------------------------------------------------
g = sns.clustermap(
    df_filt_trans,
    metric="euclidean",   # or "correlation"
    method="ward",        # or "average", "complete", etc.
    cmap="RdBu_r",        # diverging colormap for +/- z-scores
    center=0,             # center colormap on zero
    linewidths=0.5,
    figsize=(9, 10),      # adjust as you like
    xticklabels=True,
    yticklabels=True
)

# Rotate x-axis tick labels if desired
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
# # 4) Use clustermap to cluster both rows and columns
# g = sns.clustermap(
#     df_filt,
#     metric="euclidean",    # or "correlation", "cosine", etc.
#     method="ward",         # or "average", "complete", etc.
#     cmap="RdBu_r",         # typical diverging colormap
#     center=0,              # center on zero for z-scores
#     linewidths=0.5,        # helps separate cells visually
#     figsize=(8,6)        # adjust figure size as needed
# )
plt.savefig(figpath + "leiden_coarse_13_motifs_heatmap_thresh_3.5_clustered_transposed.png")
plt.savefig(figpath + "leiden_coarse_13_motifs_heatmap_thresh_3.5_clustered_transposed.pdf")
plt.show()

# %%
# Define threshold
threshold = 3.5

# Filter rows where max or min value exceeds the threshold
mr = mr_13
filtered_tfs = mr.result[(mr.result.max(axis=1) > threshold) | (mr.result.min(axis=1) < -threshold)].index.tolist()

# Print the list of TFs
print(filtered_tfs)
print(f"the number of TFs with high contrast of enrichment: {len(filtered_tfs)}")

# %%
# check the TFs associated with the highly differentially enriched motifs
for motif_id in filtered_tfs:
    motif_obj = mr.motifs.get(motif_id, None)
    if motif_obj is not None:
        # Example parameters (tweak as needed)
        factor_str = motif_obj.format_factors(
            max_length=3,        # how many factors to show
            html=False,
            include_indirect=True,
            extra_str="",
        )
        print(f"{motif_id}: ", factor_str)

# %%
# take the adata_sub for one "leiden_coarse" cluster
clust = 13
adata_sub = dict_adata_sub[clust]
# # copy the adata object so that we can annotate the peaks by their motif enrichment scores
adata_sub_motifs = adata_sub.copy()

# Convert index from 'chr1:start-end' to '1-start-end'
mr_scores = mr_13.scores
mr_scores.index = mr_scores.index.str.replace(r'chr(\d+):', r'\1-', regex=True)

# map the columns to the adata obs field
for col in mr_scores.columns:
    adata_sub_motifs.obs[col] = adata_sub_motifs.obs_names.map(mr_scores[col])
    


# %%
# define the empty list to save the vmin and vmax
list_vmin=[]
list_vmax=[]

# compute the 5, 95th percentile for the vmin and vmax
for i, tf in enumerate(filtered_tfs):
    # Get data and compute percentiles for robust scaling
    data = adata_sub_motifs.obs[tf]
    
    # Use robust percentile-based scaling to handle outliers
    vmin, vmax = np.percentile(data, [5, 95])
    #vmin=0
    # add the vmin, vmax to the lists
    list_vmin.append(vmin)
    list_vmax.append(vmax)
    
# Plot with custom settings
sc.pl.umap(adata_sub_motifs, color=filtered_tfs, 
           vmin=-2, vmax=2,
           # vmin=list_vmin, vmax=list_vmax, 
           cmap="plasma", ncols=4)
           # ax=axs[i], 
           # title=f"{tf} (range: {vmin:.2f} to {vmax:.2f})")

plt.tight_layout()
plt.savefig(figpath + f"peaks_leiden_coarse_{clust}_umap_motifs_thresh_3.5.png")
plt.show()


# %%
# import scipy.sparse as sp
# extract the count matrix and convert it to a sparse matrix
# sparse_matrix = sp.csr_matrix(mr_scores.values)
# Convert index from 'chr1:start-end' to '1-start-end'
mr_scores = mr_13.scores
mr_scores.index = mr_scores.index.str.replace(r'chr(\d+):', r'\1-', regex=True)

# create an adata object
peaks_by_motifs_sub = sc.AnnData(X = mr_scores.values)
peaks_by_motifs_sub.obs_names = mr_scores.index
peaks_by_motifs_sub.var_names = mr_scores.columns
rsc.get.anndata_to_GPU(peaks_by_motifs_sub)
peaks_by_motifs_sub

# %%
adata_sub

# %%
# # copy over the obs fields
fields_to_copy = ['timepoint', 'timepoint_contrast', 'celltype', 'celltype_contrast', 'leiden_sub_0.7_merged_renumbered']

# # copy over the metadata
peaks_by_motifs_sub.obs[fields_to_copy] = adata_sub.obs[fields_to_copy].loc[peaks_by_motifs_sub.obs_names]
peaks_by_motifs_sub

# %%
obsm_to_copy = ['X_pca', 'X_umap', 'X_umap_2D', 'X_umap_global']
for obsm_key in obsm_to_copy:
    peaks_by_motifs_sub.obsm[obsm_key] = adata_sub.obsm[obsm_key]
peaks_by_motifs_sub

# %%
# # # copy over the PCA/UMAP coordinates
# peaks_by_motifs_sub.obsm["X_pca_pseudobulk"] = adata_sub.obsm["X_pca"]
# peaks_by_motifs_sub.obsm["X_umap_pseudobulk"] = adata_sub.obsm["X_umap_2D"]

# %%
peaks_by_motifs_sub

# %%
# Step 1: Create a mapping from cluster ID to its row in mr.result
cluster_to_scores = {}
leiden_res = "leiden_sub_0.7_merged_renumbered"

# Iterate through columns in mr.result (each representing a leiden cluster)
for col in mr_13.result.columns:
    if col.startswith("z-score"):
        # Extract cluster ID
        cluster_id = col.split(" ")[1]
        # Store this column as the scores for this cluster
        cluster_to_scores[cluster_id] = mr.result[col]

# Step 2: Add all motif scores directly to peaks_by_motifs.obs
for motif in mr_13.result.index:
    # motif_name = motif.replace('.', '_')  # Clean up motif name for column naming
    motif_name = motif
    
    # For each peak, find its leiden cluster and assign the corresponding motif score
    peaks_by_motifs_sub.obs[f'motif_{motif_name}'] = peaks_by_motifs_sub.obs[leiden_res].apply(
        lambda x: cluster_to_scores.get(str(x), {}).get(motif, 0)
    )
    # ensure that the scores are "float64", not "category"
    peaks_by_motifs_sub.obs[f'motif_{motif_name}'] = peaks_by_motifs_sub.obs[f'motif_{motif_name}'].astype("float32")

# %%
# list_motifs = ["motif_M07867_2.00", "motif_M08057_2.00", "motif_M03376_2.00", "motif_M09367_2.00"]
for motif_id in filtered_tfs:
    motif_obj = mr_13.motifs.get(motif_id, None)
    factor_str = motif_obj.format_factors(
    max_length=3,        # how many factors to show
    html=False,
    include_indirect=True,
    extra_str="",
    )
    # motif_name
    motif_name = "motif_"+motif_id
    sc.pl.embedding(peaks_by_motifs_sub, basis="X_umap", color=[motif_name], 
                    title=factor_str,cmap="RdBu_r", vmin=-4, vmax=4, 
                    save=f"_peaks_leiden_{motif_name}.png")

# %% [markdown]
# ### "leiden_coarse" cluster 22

# %%
# peaks-by-motifs count matrix
mr_22 = MaelstromResult("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_leiden_coarse_22_cisBP_v2_output/")
mr_22.scores.head()
