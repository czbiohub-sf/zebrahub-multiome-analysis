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
# # Differential motif enrichment between peak clusters (peak UMAP)
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
# ## Step 1. import the maelstrom result (per peak, and per leiden cluster)
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom
import sys
import os

# gimmemotifs maelstrom module
from gimmemotifs import maelstrom

# # rapids-singlecell
# import cupy as cp
# import rapids_singlecell as rsc

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
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_umap_motifs_640K_peaks_leiden_coarse/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %% jp-MarkdownHeadingCollapsed=true
# import the peaks-by-celltype&timepoint pseudobulk object
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad")
adata_peaks

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### import the "leiden_coarse" annotation

# %%
# # import the cluster annotation
# clust_anno = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/all_peaks_leiden_clusters.csv", index_col=0)
# clust_anno.head()

# %%
# adata_peaks.obs["leiden_coarse"] = adata_peaks.obs_names.map(clust_anno["leiden_coarse_renumbered"])
# adata_peaks.obs.head()

# %%
# # conver the "leiden_coarse" as categorical var
# adata_peaks.obs["leiden_coarse"] = adata_peaks.obs["leiden_coarse"].astype('category')

# %%
# # save the new h5ad object (with "leiden_coarse" label)
# adata_peaks.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")

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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## export peaks with cluster labels as txt format

# %%
# load the utility functions for exporting the peaks:clusters df to txt format
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs/")
from utils_maelstrom import export_peaks_for_gimmemotifs


# %%
# create a directory
os.makedirs("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/motif_enrich_analysis_leiden_coarse/", exist_ok=True)
# export the peaks with the labels
export_peaks_for_gimmemotifs(adata_peaks, cluster="leiden_coarse", 
                             out_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/motif_enrich_analysis_leiden_coarse/")


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Step 2. Run GimmeMotifs maelstrom for differential motif computation
# - This is done on HPC using Slurm
# - reference: https://gimmemotifs.readthedocs.io/en/master/tutorials.html#find-differential-motifs

# %%
## Run the GimmeMotifs maelstrom using Slurm on HPC
# !sbatch /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/slurm_scripts/gimme_maelstrom.sh

# %% [markdown]
# ## Step 3. Check the Maelstrom output:
#

# %%
# import the result of the maelstrom
from gimmemotifs.maelstrom import MaelstromResult
mr = MaelstromResult("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/")
# mr.plot_heatmap(threshold=2)

# %%
# peaks-by-motifs count matrix
mr.scores.head()

# %%
from utils_gimmemotifs_maelstrom_gpu import *
help(compute_gpu_motif_enrichment)

# %%
from utils_gimmemotifs_maelstrom_gpu_dask import *
help(compute_dask_gpu_motif_enrichment)

# %%
# 1. Load your peaks-by-motifs matrix
# read using cudf instead of pandas
# peaks_motifs = cudf.read_csv(
#     txt_path,
#     sep="\t",
#     index_col=0,
# )

from pathlib import Path
import gzip, shutil, pandas as pd, cudf   # cpu → gpu path
import dask_cudf as dc

# define the filepaths for the txt.gz and txt files
gz_path   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden1.5_cisBP_v2_danio_rerio_output/motif.score.txt.gz"
txt_path  = Path(gz_path).with_suffix('')  # …/motif.score.txt

# Decompress on the CPU
with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)


# read the dataframe using dask_cudf
ddf = dc.read_csv(
    txt_path,                # still the decompressed file
    sep="\t",
    #index_col=0,
    #dtype="float32",         # 4-byte floats instead of 8
    chunksize="256MB",       # each partition ≈256 MB on disk
)
ddf.head()

# %%
# # set the first column as the index
# ddf = ddf.set_index('loc')
# ddf.head()

# %%
# ddf.loc["chr1:32-526"].compute()

# %%
# # 1. Load your peaks-by-motifs matrix
# peaks_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden1.5_cisBP_v2_danio_rerio_output/motif.score.txt.gz", index_col=0, sep="\t")
# peaks_motifs

# 2. Load your cluster labels
# The cluster labels should be a DataFrame with the same index as peaks_motifs
cluster_df = adata_peaks.obs["leiden_coarse"]
cluster_df.index = ['chr' + idx.split('-')[0] + ':' + idx.split('-')[1] + '-' + idx.split('-')[2] 
                    for idx in cluster_df.index]
cluster_df.head()

# %%
if isinstance(cluster_df, pd.Series):
    cluster_df = cluster_df.to_frame(name='cluster')
    
cluster_df.head()

# %%
# # Create a Dask client
# client = Client(n_workers=8)  # Adjust based on your system

# try:
#     # Compute enrichment scores
#     enrichment_scores = compute_dask_gpu_motif_enrichment(
#         ddf,
#         cluster_df,
#         methods=["hypergeom", "mwu", "rf"],  # Start with just hypergeom
#         batch_size=1000,         # Adjust based on available memory
#         client=client
#     )
    
#     # Save results
#     enrichment_scores.to_csv("motif_enrichment_scores.txt", sep="\t")
    
# finally:
#     # Close the client
#     client.close()

# %%
# # 3. Compute the enrichment scores
# enrichment_scores = compute_gpu_motif_enrichment(
#     peaks_motifs, 
#     cluster_df, 
#     methods=["hypergeom", "mwu", "rf"],  # You can use any combination of these
#     ncpus=12  # Adjust based on your system
# )

# # 4. Save the results
# enrichment_scores.to_csv("motif_enrichment_scores_leiden_coarse.txt", sep="\t")

# %%

# %%

# %%
# from gimmemotifs.maelstrom.moap import moap

# # create a dataframe with the new labels
# sub_labels = adata_peaks.obs[['leiden_coarse']].copy()
# sub_labels = sub_labels.rename(columns = {"leiden_coarse":"cluster"})  # Maelstrom expects e.g. a 'cluster' column

# # Make sure the index (peak coordinates) in sub_labels 
# # matches the row index of mr.counts. 
# # If needed, subset or reindex to ensure alignment:
# sub_labels_fixed = sub_labels.copy()
# sub_labels_fixed.index = (
#     "chr" +
#     sub_labels_fixed.index.str.replace("-", ":", n=1).str.replace("-", "-", n=1)
# )
# # sub_labels = sub_labels.reindex(mr.counts.index).dropna()
# # Now subset to shared peaks
# shared_peaks = mr.scores.index.intersection(sub_labels_fixed.index)
# scores_subset = mr.scores.loc[shared_peaks]
# labels_subset = sub_labels_fixed.loc[shared_peaks]

# # Run MOAP
# # For motif "counts", the Hypergeom or Random Forest methods are typical classification methods. 
# # Here, we demonstrate Hypergeom:

# activity = moap(
#     inputfile=labels_subset,# Our 1-column classification table
#     method='hypergeom',       # 'hypergeom' suits count data
#     scoring='count',          # Must match the motif data type
#     motiffile=mr.counts,      # Reuse your precomputed motif counts DataFrame
#     random_state=42,          # For reproducibility
#     ncpus=8                   # Adjust cores as needed
# )

# ##############################################
# # 3. Inspect the results
# ##############################################

# print(activity.head())

# %% [markdown]
# ## Plotting the heatmap

# %%
# save the cluster-by-motifs dataframe
mr.result.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/peak_clusts_by_motifs_maelstrom.csv")


# %%
# define a function to extract the direct/indirect factors for each motif, and create a dataframe of motifs
def extract_motif_factors(mr_motifs):
    """
    Extracts direct and indirect factors from mr.motifs and converts them into a DataFrame.

    Parameters:
    -----------
    mr_motifs : dict
        Dictionary of motifs (mr.motifs) where each motif contains a `.factors` dictionary.

    Returns:
    --------
    pd.DataFrame
        DataFrame with motif names as the index and two columns: "direct" and "indirect" factors.
    """
    motif_data = []

    for motif_name, motif_obj in mr_motifs.items():
        # Extract factors (handle cases where direct/indirect keys are missing)
        direct_factors = motif_obj.factors.get('direct', [])
        indirect_factors = motif_obj.factors.get('indirect\nor predicted', [])

        # Convert lists to comma-separated strings
        direct_str = ", ".join(direct_factors) if direct_factors else "None"
        indirect_str = ", ".join(indirect_factors) if indirect_factors else "None"

        # Append to list
        motif_data.append([motif_name, direct_str, indirect_str])

    # Convert to DataFrame
    df_motif_factors = pd.DataFrame(motif_data, columns=["motif", "direct", "indirect"])
    df_motif_factors.set_index("motif", inplace=True)

    return df_motif_factors

# Example usage
df_motifs = extract_motif_factors(mr.motifs)

# check the df
df_motifs.head()

# Save to CSV
# df_motifs.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors.csv")

# # Display first few rows
# print(df_motifs.head())


# %%

# %%
# plot the enrichment of motifs vs leiden clusters (with some threshold)
mr.plot_heatmap(threshold=4)
plt.savefig(figpath + "maelstrom_peak_leiden_clusts_thresh_4.png")
plt.savefig(figpath + "maelstrom_peak_leiden_clusts_thresh_4.pdf")
plt.show()

# %%
# plot the enrichment of motifs vs leiden clusters (with different threshold)
mr.plot_heatmap(threshold=3.5)
plt.savefig(figpath + "maelstrom_peak_leiden_clusts_thresh_3.5.png")
plt.savefig(figpath + "maelstrom_peak_leiden_clusts_thresh_3.5.pdf")
plt.show()

# %%
sc.pl.umap(adata_peaks, color="leiden_coarse", legend_loc="on data", save="_peaks_leiden_coarse_labels.png")

# %%
# Check each cluster for the most accessible celltypes (number of peaks)
adata_peaks[adata_peaks.obs["leiden_coarse"]==2].obs["celltype"].value_counts()[0:10]

# %% [markdown]
# ### create a heatmap of leiden cluster-by-motifs (enrichment scores)

# %%
# 1) Filter based on threshold
threshold = 3.5 # for example
df = mr.result.copy()

# 2) Keep only those rows where at least one absolute value >= threshold
mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]
# Example: remove "z-score " prefix
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]

# 3) RENAME ROWS to factor names (similar to 'plot_heatmap(name=True)')
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


# 4) CLUSTER + PLOT using seaborn.clustermap
# ---------------------------------------------------------
g = sns.clustermap(
    df_filt_named,
    metric="euclidean",   # or "correlation"
    method="ward",        # or "average", "complete", etc.
    cmap="RdBu_r",        # diverging colormap for +/- z-scores
    center=0,             # center colormap on zero
    linewidths=0.5,
    figsize=(10, 8),      # adjust as you like
    xticklabels=True,
    yticklabels=True
)
plt.xlabel("leiden clusters")
plt.ylabel("factors")

# Rotate x-axis tick labels if desired
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0)

plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered.png")
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered.pdf")
plt.show()

# %% [markdown]
# ### heatmap (tranposed - leiden clusters-by-motifs)

# %%
# 1) Filter based on threshold
threshold = 3.5 # for example
df = mr.result.copy()

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
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered_transposed.png")
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered_transposed.pdf")
plt.show()

# %% [markdown]
# ## Step 4. motif enrichment scores projected onto peak UMAP

# %%
mr_scores = mr.scores
mr_scores.head()

# %%
# # copy the adata object so that we can annotate the peaks by their motif enrichment scores
# peaks_by_pb: peaks-by-pseudobulk
peaks_by_pb_motifs = adata_peaks.copy()

# %%
# Convert index from 'chr1:start-end' to '1-start-end'
mr_scores.index = mr_scores.index.str.replace(r'chr(\d+):', r'\1-', regex=True)

# map the columns to the adata obs field
for col in mr_scores.columns:
    peaks_by_pb_motifs.obs[col] = peaks_by_pb_motifs.obs_names.map(mr_scores[col])

# %%
# Define threshold
threshold = 3

# Filter rows where max or min value exceeds the threshold
filtered_tfs = mr.result[(mr.result.max(axis=1) > threshold) | (mr.result.min(axis=1) < -threshold)].index.tolist()

# Print the list of TFs
print(filtered_tfs)


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
# check the distribution of the enrichment scores (per-peak level)
mr_scores.iloc[:,10].hist(bins=40)
plt.grid(False)
plt.show()

# %%

# %%
# define the empty list to save the vmin and vmax
list_vmin=[]
list_vmax=[]

# compute the 5, 95th percentile for the vmin and vmax
for i, tf in enumerate(filtered_tfs):
    # Get data and compute percentiles for robust scaling
    data = peaks_by_pb_motifs.obs[tf]
    
    # Use robust percentile-based scaling to handle outliers
    vmin, vmax = np.percentile(data, [5, 95])
    #vmin=0
    # add the vmin, vmax to the lists
    list_vmin.append(vmin)
    list_vmax.append(vmax)
    
# Plot with custom settings
sc.pl.umap(peaks_by_pb_motifs, color=filtered_tfs, 
           vmin=list_vmin, vmax=list_vmax, 
           cmap="plasma", ncols=3)
           # ax=axs[i], 
           # title=f"{tf} (range: {vmin:.2f} to {vmax:.2f})")

plt.tight_layout()
plt.savefig(figpath + "peaks_640K_umap_motifs_thresh_3.png")
plt.show()


# %%
# Plot with custom settings
sc.pl.umap(peaks_by_pb_motifs, color=filtered_tfs, 
           vmin=-2, vmax=2, 
           cmap="plasma", ncols=3)
           # ax=axs[i], 
           # title=f"{tf} (range: {vmin:.2f} to {vmax:.2f})")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### NOTE: the individual peak's enrichment score is "weak".

# %%
# threshold
threshold = 3.5

# Create a dictionary to store counts for each TF
tf_counts = {}

# For each filtered TF, count instances above and below threshold
for tf in filtered_tfs:
    # Get the column for this TF
    col = mr_scores.loc[:,tf]
    
    # Count values above threshold
    above_count = sum(col > threshold)
    
    # Count values below negative threshold
    below_count = sum(col < -threshold)
    
    # Store counts in dictionary
    tf_counts[tf] = {'above': above_count, 'below': below_count, 'total': above_count + below_count}

# Print results
print(f"Found {len(filtered_tfs)} TFs with values exceeding thresholds")
print("\nCounts per TF:")
for tf, counts in tf_counts.items():
    print(f"{tf}: {counts['above']} above {threshold}, {counts['below']} below -{threshold} (Total: {counts['total']})")

# %% [markdown]
# ## Step 5. peaks-by-motifs 
#
# - Create an adata object for peaks-by-motifs (differential)
# - We will create an adata object using the peaks-by-motifs count matrix - this will keep the same metadata for the peaks (obs)

# %%
mr_scores.values

# %%
# import scipy.sparse as sp
# extract the count matrix and convert it to a sparse matrix
# sparse_matrix = sp.csr_matrix(mr_scores.values)
# Convert index from 'chr1:start-end' to '1-start-end'
mr_scores.index = mr_scores.index.str.replace(r'chr(\d+):', r'\1-', regex=True)


# create an adata object
peaks_by_motifs = sc.AnnData(X = mr_scores.values)
peaks_by_motifs.obs_names = mr_scores.index
peaks_by_motifs.var_names = mr_scores.columns
rsc.get.anndata_to_GPU(peaks_by_motifs)
peaks_by_motifs

# %%
# # copy over the obs fields
fields_to_copy = ['timepoint', 'timepoint_contrast', 'celltype', 'celltype_contrast', 'leiden_coarse']

# # copy over the metadata
peaks_by_motifs.obs[fields_to_copy] = adata_peaks.obs[fields_to_copy].loc[peaks_by_motifs.obs_names]
peaks_by_motifs

# %%
obsm_to_copy = ['X_pca', 'X_umap', 'X_umap_2D', 'X_umap_3D']
for obsm_key in obsm_to_copy:
    peaks_by_motifs.obsm[obsm_key] = adata_peaks.obsm[obsm_key]
peaks_by_motifs

# %%
# # copy over the PCA/UMAP coordinates
peaks_by_motifs.obsm["X_pca_pseudobulk"] = adata_peaks.obsm["X_pca"]
peaks_by_motifs.obsm["X_umap_pseudobulk"] = adata_peaks.obsm["X_umap_2D"]

# %%
rsc.tl.pca(peaks_by_motifs)
rsc.pp.neighbors(peaks_by_motifs, n_neighbors=20, n_pcs=10, metric="cosine")

# %%
rsc.tl.umap(peaks_by_motifs, min_dist=0.1)
sc.pl.umap(peaks_by_motifs, color=["leiden_coarse", "celltype", "timepoint"])

# %% [markdown]
# #### NOTE: There's no clear structure in terms of peaks-by-motifs (other than two large structures).

# %%
# save the peaks-by-motifs adata object
# peaks_by_motifs.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden1.5_cisBP_v2_danio_rerio_output/peaks_by_motifs_640K_peaks_cisBP_v2_danio_rerio_motifs.h5ad")

# %% [markdown]
# ## plot the UMAP with aggreagated/ranked scores (per cluster)

# %%
# mr.result is a dataframe of motifs-by-peak_clusters
mr.result.head()

# %%
# Step 1: Create a mapping from cluster ID to its row in mr.result
cluster_to_scores = {}
leiden_res = "leiden_coarse"

# Iterate through columns in mr.result (each representing a leiden cluster)
for col in mr.result.columns:
    if col.startswith("z-score"):
        # Extract cluster ID
        cluster_id = col.split(" ")[1]
        # Store this column as the scores for this cluster
        cluster_to_scores[cluster_id] = mr.result[col]

# Step 2: Add all motif scores directly to peaks_by_motifs.obs
for motif in mr.result.index:
    # motif_name = motif.replace('.', '_')  # Clean up motif name for column naming
    motif_name = motif
    
    # For each peak, find its leiden cluster and assign the corresponding motif score
    peaks_by_motifs.obs[f'motif_{motif_name}'] = peaks_by_motifs.obs[leiden_res].apply(
        lambda x: cluster_to_scores.get(str(x), {}).get(motif, 0)
    )
    # ensure that the scores are "float64", not "category"
    peaks_by_motifs.obs[f'motif_{motif_name}'] = peaks_by_motifs.obs[f'motif_{motif_name}'].astype("float32")

# %%
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
# Define threshold
threshold = 3.5

# Filter rows where max or min value exceeds the threshold
filtered_tfs = mr_result[(mr_result.max(axis=1) > threshold) | (mr_result.min(axis=1) < -threshold)].index.tolist()

# Print the list of TFs
print(filtered_tfs)

# %%
# list_motifs = ["motif_M07867_2.00", "motif_M08057_2.00", "motif_M03376_2.00", "motif_M09367_2.00"]
for motif_id in filtered_tfs:
    motif_obj = mr.motifs.get(motif_id, None)
    factor_str = motif_obj.format_factors(
    max_length=3,        # how many factors to show
    html=False,
    include_indirect=True,
    extra_str="",
    )
    # motif_name
    motif_name = "motif_"+motif_id
    sc.pl.embedding(peaks_by_motifs, basis="X_umap_pseudobulk", color=[motif_name], 
                    title=factor_str,cmap="RdBu_r", vmin=-4, vmax=4, 
                    save=f"_peaks_leiden_{motif_name}.png")

# %%
# sort the motifs by their "length"
motif_cluster_df.loc[filtered_tfs].sort_values("Motif_Length")

# %%
# take the list of motifs
list_motifs = motif_cluster_df.loc[filtered_tfs].sort_values("Motif_Length").index
list_motifs = ["motif_" + m for m in list_motifs]

sc.pl.umap(peaks_by_motifs, color=list_motifs, ncols=4, 
           cmap="RdBu_r", vmin=-4, vmax=4,)

# %%

# %%
list_motifs_high_contrast = ["M06313_2.00","M09755_2.00","M08130_2.00"]

for motif in list_motifs_high_contrast:
    sc.pl.embedding(peaks_by_motifs, basis="X_umap", color="motif_"+motif, 
                    title=motif,cmap="RdBu_r", vmin=-4, vmax=4, 
                    save=f"_peaks_leiden_{motif}.png")

# %%
# List of motifs you want to plot
# motif_list = ["GM.5.0.Ets.0033", "GM.5.0.Ets.0034", "GM.5.0.Ets.0035"]  # Replace with your motif IDs
motif_list = motif_cluster_df.loc[filtered_tfs].sort_values("Motif_Length").index

# Create a directory for output
output_dir = figpath + "logos/"
os.makedirs(output_dir, exist_ok=True)

# compute the maximum bits
background = np.array([0.3, 0.2, 0.2, 0.3]) # A, C, G, T (the zebrafish genome is roughly 40% GC ratio compared to AT)
ymax = -np.log2(background.min())

for motif_name in motif_list:
    # Use PWM and normalize to information content
    selected_motif = next(m for m in motifs if m.id == motif_name)
    pwm = selected_motif.pwm  # Or use selected_motif.pfm if necessary
    ic_matrix, total_ic = pwm_to_information_matrix(pwm)
    # Convert to Pandas DataFrame for Logomaker
    df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

    # Plot with Logomaker
    plt.figure(figsize=(max(6, 0.6*df.shape[0]), 3.2))
    logo = logomaker.Logo(df)
    ax = logo.ax

    ax.set_title(f"Sequence Logo for {selected_motif.id}")
    ax.set_ylabel("bits")
    ax.set_xlabel("position")
    ax.set_xticks(np.arange(len(df)))

    ax.set_ylim(0, ymax) # DNA max with uniform bg is 2 bits; leave room if bg ≠ uniform
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_dir + f"{motif_name}.seq.logo.png")
    plt.savefig(output_dir + f"{motif_name}.seq.logo.pdf")
    plt.show()
    print(total_ic)

# %% [markdown]
# ## Resume from here - 6/10/2025
#
# - we will find the motifs that have very low contrast between the "leiden_coarse" clusters.
#

# %%
mr_result = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/peak_clusts_by_motifs_maelstrom.csv",
                        index_col=0)

mr_result.head()

# %%
peaks_by_motifs = adata_peaks.copy()

# %%
# Step 1: Create a mapping from cluster ID to its row in mr.result
cluster_to_scores = {}
leiden_res = "leiden_coarse"

# Iterate through columns in mr.result (each representing a leiden cluster)
for col in mr_result.columns:
    if col.startswith("z-score"):
        # Extract cluster ID
        cluster_id = col.split(" ")[1]
        # Store this column as the scores for this cluster
        cluster_to_scores[cluster_id] = mr_result[col]

# Step 2: Add all motif scores directly to peaks_by_motifs.obs
for motif in mr_result.index:
    # motif_name = motif.replace('.', '_')  # Clean up motif name for column naming
    motif_name = motif
    
    # For each peak, find its leiden cluster and assign the corresponding motif score
    peaks_by_motifs.obs[f'motif_{motif_name}'] = peaks_by_motifs.obs[leiden_res].apply(
        lambda x: cluster_to_scores.get(str(x), {}).get(motif, 0)
    )
    # ensure that the scores are "float64", not "category"
    peaks_by_motifs.obs[f'motif_{motif_name}'] = peaks_by_motifs.obs[f'motif_{motif_name}'].astype("float32")

# %%
# Calculate max absolute value for each motif (across all clusters)
max_abs_scores = mr_result.abs().max(axis=1)

# Calculate other simple metrics
mean_abs_scores = mr_result.abs().mean(axis=1)
std_scores = mr_result.std(axis=1)
range_scores = mr_result.max(axis=1) - mr_result.min(axis=1)

# Create a DataFrame with all scores
low_contrast_df = pd.DataFrame({
    'Max_Abs': max_abs_scores,
    'Mean_Abs': mean_abs_scores,
    'Std_Dev': std_scores,
    'Range': range_scores
})

# Sort by max absolute value (ascending - smallest first = least contrast)
low_contrast_df = low_contrast_df.sort_values('Max_Abs', ascending=True)

# Display top 20 motifs with least contrast
print("TOP 20 MOTIFS WITH LEAST CONTRAST (by Max Absolute Value):")
print("="*70)
print(f"{'Rank':<4} {'Motif':<15} {'Max|Abs|':<8} {'Mean|Abs|':<9} {'Std Dev':<8} {'Range':<8}")
print("-" * 70)

# %%
low_contrast_df.head(20)

# %%
abs(-2)

# %%
# Define threshold
threshold = 1.5

# Filter rows where max or min value exceeds the threshold
filtered_tfs = mr_result[abs(mr_result.max(axis=1) < threshold) & abs(mr_result.min(axis=1) < threshold)].index.tolist()

# Print the list of TFs
print(filtered_tfs)

# %%
# sort the motifs by their "length"
motif_cluster_df.loc[filtered_tfs].sort_values("Motif_Length")


# %%
# generate the UMAP colored by the least contrasted motifs
# filtered_tfs = low_contrast_df.head(20).index
for motif_id in filtered_tfs:
    # motif_obj = mr.motifs.get(motif_id, None)
    # factor_str = motif_obj.format_factors(
    # max_length=3,        # how many factors to show
    # html=False,
    # include_indirect=True,
    # extra_str="",
    # )
    # motif_name
    motif_name = "motif_"+motif_id
    sc.pl.embedding(peaks_by_motifs, basis="X_umap_2D", color=[motif_name], 
                    title=motif_id,cmap="RdBu_r", vmin=-4, vmax=4, 
                    save=f"_peaks_leiden_{motif_name}.png")

# %%
filtered_tfs

# %%
selected_motif.factors["indirect\nor predicted"][0:3]

# %%
# check the TFs associated with the highly differentially enriched motifs
for motif_id in filtered_tfs:
    selected_motif = next(m for m in motifs if m.id == motif_id)
    # motif_obj = mr.motifs.get(motif_id, None)
    factors_list = selected_motif.factors["indirect\nor predicted"][0:3]
    print(f"{motif_id}: {factors_list}, length is {len(selected_motif.consensus)}")
    # if motif_obj is not None:
    #     # Example parameters (tweak as needed)
    #     factor_str = motif_obj.format_factors(
    #         max_length=3,        # how many factors to show
    #         html=False,
    #         include_indirect=True,
    #         extra_str="",
    #     )
    #     print(f"{motif_id}: ", factor_str)

# %%
# Do the above in a systematic way for all "motifs"
# for motif in adata_peaks.obs.columns[adata_peaks.obs.columns.str.startswith("M")]:
#     extracted_motif = motif.split("motif_GM_5_0_")[1]
#     sc.pl.embedding(peaks_by_motifs, basis="X_umap_pseudobulk", color=motif, 
#                     cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}.png", show=False)

# %% [markdown]
# ## Step 6. GimmeMotifs to generate the sequence logo plots

# %%
from gimmemotifs.motif import Motif, read_motifs

# Load motifs from the default database
motif_file = "/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm"  # Use the correct database file
motifs = read_motifs(motif_file)

# %%
print(f"the number of motifs is: {len(motifs)}")

# %%
# Get the specific motif (e.g., "GM.5.0.Ets.0033")
motif_name = "M08572_2.00"
selected_motif = next(m for m in motifs if m.id == motif_name)

# Extract the Position Frequency Matrix (PFM) and Position Weight Matrix (PWM)
pfm = selected_motif.to_pfm()  # Transpose to make it compatible with logomaker
pwm = selected_motif.to_ppm()

# consensus sequence
print(selected_motif.to_consensus())
print(pwm)

# %%
'/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_umap_motifs_640K_peaks_leiden_coarse/logos/M03072_2.00.seq.logo.pdf'

# %%
selected_motif.id

# %%
from gimmemotifs.motif import _plotting as motif_plot_module
motif_plot_module.plot_logo(selected_motif, kind="information")
plt.grid(False)
# plt.yscale("log")
plt.savefig(figpath + f"logos/{selected_motif.id}.seq.logo.pdf")
plt.show()

# %%
# # Compute information content (IC) from PWM
# def compute_information_content(pwm):
#     # background = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform background probabilities
#     background = np.array([0.3, 0.2, 0.2, 0.3]) # A, C, G, T (the zebrafish genome is roughly 40% GC ratio compared to AT)
#     ic_matrix = pwm * np.log2(pwm / background)  # Shannon information content
#     ic_matrix = np.nan_to_num(ic_matrix)  # Replace NaNs with 0
#     return ic_matrix

# # Use PWM and normalize to information content
# pwm = selected_motif.pwm  # Or use selected_motif.pfm if necessary
# ic_matrix = compute_information_content(pwm)

# # Convert to Pandas DataFrame for Logomaker
# df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

# # Plot with Logomaker
# plt.figure(figsize=(12, 4))  # Increase figure size
# logomaker.Logo(df)

# plt.title(f"Sequence Logo for {selected_motif.id}")
# plt.ylabel("bits")
# plt.xlabel("bases")
# plt.grid(False)
# # plt.savefig(figpath + f"{motif_name}.seq.logo.png")
# # plt.savefig(figpath + f"{motif_name}.seq.logo.pdf")
# plt.show()

# %%
# define the information content using Schneider-Stephens/column-KL style
import logomaker

def pwm_to_information_matrix(pwm):
    """
    Convert a probability PWM (rows = positions, cols = A,C,G,T) to the
    Schneider–Stephens / KL-divergence information matrix that Logomaker
    expects for a classic sequence logo.

    Returns
    -------
    ic_mat : ndarray  (same shape as pwm, all entries ≥ 0)
    """
    # define the base parameters
    background = np.array([0.30, 0.20, 0.20, 0.30])   # A, C, G, T   (zebrafish genome-wide)
    eps     = 1e-6           # tiny value to avoid log(0)
    # protect zeros
    pwm = np.clip(np.asarray(pwm, dtype=float), eps, 1.0)
    bg  = np.clip(np.asarray(background, dtype=float), eps, 1.0)

    # per-cell KL term   p * (log2 p – log2 q)
    kl_cell   = pwm * (np.log2(pwm) - np.log2(bg))

    # total information per position (column height)
    ic_col    = kl_cell.sum(axis=1)    # shape (L,)
    total_ic = ic_col.sum()

    # final letter heights  P_ib  ×  IC_i
    ic_matrix = pwm * ic_col[:, None]                              # shape (L,4)
    return ic_matrix, total_ic


# %%
# test with a PWM
pwm = selected_motif.pwm
ic_matrix, total_ic = pwm_to_information_matrix(pwm)       # all non-negative

df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

plt.figure(figsize=(max(6, 0.6*df.shape[0]), 3.2))
plt.figure()
logomaker.Logo(df)

plt.title(f"Sequence Logo for {selected_motif.id}")
plt.xlabel("position")
plt.ylabel("bits")

# compute the maximum bits
background = np.array([0.3, 0.2, 0.2, 0.3]) # A, C, G, T (the zebrafish genome is roughly 40% GC ratio compared to AT)
ymax = -np.log2(background.min())
plt.ylim(0, ymax) # DNA max with uniform bg is 2 bits; leave room if bg ≠ uniform
plt.tight_layout()
plt.grid(False)
plt.show()
print(total_ic)

# %% [markdown]
# ## Use a module to generate the seq.logo plots systematically

# %%
# List of motifs you want to plot
# motif_list = ["GM.5.0.Ets.0033", "GM.5.0.Ets.0034", "GM.5.0.Ets.0035"]  # Replace with your motif IDs
motif_list = filtered_tfs

# Create a directory for output
output_dir = figpath + "logos/"
os.makedirs(output_dir, exist_ok=True)

# compute the maximum bits
background = np.array([0.3, 0.2, 0.2, 0.3]) # A, C, G, T (the zebrafish genome is roughly 40% GC ratio compared to AT)
ymax = -np.log2(background.min())

for motif_name in motif_list:
    # Use PWM and normalize to information content
    selected_motif = next(m for m in motifs if m.id == motif_name)
    pwm = selected_motif.pwm  # Or use selected_motif.pfm if necessary
    ic_matrix, total_ic = pwm_to_information_matrix(pwm)
    # Convert to Pandas DataFrame for Logomaker
    df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

    # Plot with Logomaker
    plt.figure(figsize=(max(6, 0.6*df.shape[0]), 3.2))
    logo = logomaker.Logo(df)
    ax = logo.ax

    ax.set_title(f"Sequence Logo for {selected_motif.id}")
    ax.set_ylabel("bits")
    ax.set_xlabel("position")
    ax.set_xticks(np.arange(len(df)))

    ax.set_ylim(0, ymax) # DNA max with uniform bg is 2 bits; leave room if bg ≠ uniform
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_dir + f"{motif_name}.seq.logo.png")
    plt.savefig(output_dir + f"{motif_name}.seq.logo.pdf")
    plt.show()
    print(total_ic)

# %%

# %% [markdown]
# ## Step 7. quantify the information content of TF motifs
# 1. compute the information content for each motif (short vs long motifs) -> motif complexity
# 2. compute the "variability" of motif enrichment between clusters (and correlate this with the motif complexity) -> C.V.
#

# %%
# load the maelstrom result (mr)
mr_result = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/peak_clusts_by_motifs_maelstrom.csv",
                        index_col=0)
mr_result.head()

# %%
# import the util functions
from utils_contrast_motif_scores import *

# %%
motif_id = "M04320_2.00"
selected_motif = next(m for m in motifs if m.id == motif_id)
np.sum(selected_motif.pwm, axis=1)

# %%
len(selected_motif.pwm)

# %%
selected_motif.consensus

# %%
motif_cluster_df = pd.DataFrame()

# go through each TF motif, and compute the contrast metrics (across leiden clusters), as well as IC
for motif_id in mr_result.index:
    # Get the PWM for this motif
    selected_motif = next(m for m in motifs if m.id == motif_id)
    pwm = selected_motif.pwm
    factors = selected_motif.factors
    # compute the total information content (total_ic)
    total_ic = motif_total_information(pwm)
    length_TF = len(pwm)
    print(f"{selected_motif} has information content of {total_ic}")
    print(f"{factors}")

# %%
motif_list_subclusts = ["M07944_2.00","M02899_2.00","M07485_2.00","M06254_2.00",
                        "M07907_2.00","M04471_2.00","M06514_2.00","M08452_2.00"]

# go through each TF motif, and compute the contrast metrics (across leiden clusters), as well as IC
for motif_id in motif_list_subclusts:
    # Get the PWM for this motif
    selected_motif = next(m for m in motifs if m.id == motif_id)
    # pwm = selected_motif.pwm
    factors = selected_motif.factors
    # compute the total information content (total_ic)
    # total_ic = motif_total_information(pwm)
    # length_TF = len(pwm)
    print(f"{selected_motif} is associated with {factors}")
    # print(f"{factors}")

# %%
from math import log2
# Initialize empty DataFrame to store all metrics
motif_cluster_df = pd.DataFrame()

# Initialize lists to store all metrics for each motif
motif_ids = []
total_ic_values = []
motif_lengths = []
ic_per_position = []
max_dev_ratios = []
peak_to_median_ratios = []
percentile_ratios = []
binarity_scores = []
gini_indices = []
discrete_mi_values = []
cv_values = []
std_values = []

# Go through each TF motif and compute all metrics
for motif_id in mr_result.index:
    try:
        # Get the PWM for this motif
        selected_motif = next(m for m in motifs if m.id == motif_id)
        pwm = selected_motif.pwm
        
        # Compute information content metrics
        total_ic = motif_total_information(pwm)
        length_TF = len(pwm)
        ic_per_pos = total_ic / length_TF if length_TF > 0 else 0
        
        # Get the cluster scores for this motif
        cluster_scores = mr_result.loc[motif_id].values
        
        # Compute contrast metrics across clusters
        max_dev = max_deviation_ratio(cluster_scores)
        peak_to_med = peak_to_median_ratio(cluster_scores)
        perc_ratio = percentile_ratio(cluster_scores, percentile=90)
        bin_score = binarity_score(cluster_scores, threshold=0.5)
        gini_idx = calculate_gini_index(cluster_scores)
        discrete_mi = calculate_discrete_mutual_information(cluster_scores)
        
        # Compute traditional metrics
        cv = np.std(cluster_scores) / np.abs(np.mean(cluster_scores)) if np.mean(cluster_scores) != 0 else 0
        std = np.std(cluster_scores)
        
        # Store all values
        motif_ids.append(motif_id)
        total_ic_values.append(total_ic)
        motif_lengths.append(length_TF)
        ic_per_position.append(ic_per_pos)
        max_dev_ratios.append(max_dev)
        peak_to_median_ratios.append(peak_to_med)
        percentile_ratios.append(perc_ratio)
        binarity_scores.append(bin_score)
        gini_indices.append(gini_idx)
        discrete_mi_values.append(discrete_mi)
        cv_values.append(cv)
        std_values.append(std)
        
        # print(f"{motif_id} has information content of {total_ic:.3f} bits")
        
    except (StopIteration, AttributeError, KeyError) as e:
        print(f"Could not process motif {motif_id}: {e}")
        # Skip this motif if we can't find it or process it
        continue

# Create the comprehensive DataFrame
motif_cluster_df = pd.DataFrame({
    'Total_IC': total_ic_values,
    'Motif_Length': motif_lengths,
    'IC_per_Position': ic_per_position,
    'Max_Deviation_Ratio': max_dev_ratios,
    'Peak_to_Median_Ratio': peak_to_median_ratios,
    'Percentile_90_Ratio': percentile_ratios,
    'Binarity_Score': binarity_scores,
    'Gini_Index': gini_indices,
    'Discrete_MI': discrete_mi_values,
    'CV': cv_values,
    'STD': std_values
}, index=motif_ids)

# Handle any NaN values
motif_cluster_df = motif_cluster_df.fillna(0)

# Sort by one of the contrast metrics (you can change this)
motif_cluster_df = motif_cluster_df.sort_values('Gini_Index', ascending=False)

# Display summary
print(f"\nProcessed {len(motif_cluster_df)} motifs")
print(f"DataFrame shape: {motif_cluster_df.shape}")
print("\nTop 10 motifs by Gini Index (most unequal distribution across clusters):")
print(motif_cluster_df.head(10))

# Save the comprehensive results
# motif_cluster_df.to_csv('comprehensive_motif_metrics.csv')
# print(f"\nResults saved to 'comprehensive_motif_metrics.csv'")

# Optional: Display correlation matrix between different contrast metrics
print("\nCorrelation matrix between contrast metrics:")
contrast_metrics = ['Max_Deviation_Ratio', 'Peak_to_Median_Ratio', 'Percentile_90_Ratio', 
                   'Binarity_Score', 'Gini_Index', 'Discrete_MI', 'CV', 'STD']
correlation_matrix = motif_cluster_df[contrast_metrics].corr()
print(correlation_matrix.round(3))

# %%
motif_cluster_df.tail(30)

# %%
motif_cluster_df.loc["M11368_2.00"]

# %%
# Define the contrast metrics to analyze
contrast_metrics = ['Max_Deviation_Ratio', 'Peak_to_Median_Ratio', 'Percentile_90_Ratio', 
                   'Binarity_Score', 'Gini_Index', 'Discrete_MI', 'CV', 'STD']

# Calculate correlations with Total_IC
ic_correlations = {}
ic_pvalues = {}

print("Correlation between Total_IC and contrast metrics:")
print("="*60)

for metric in contrast_metrics:
    # Calculate Pearson correlation
    corr, p_val = stats.pearsonr(motif_cluster_df['Total_IC'], motif_cluster_df[metric])
    ic_correlations[metric] = corr
    ic_pvalues[metric] = p_val
    
    # Also calculate Spearman correlation (rank-based, more robust)
    spearman_corr, spearman_p = stats.spearmanr(motif_cluster_df['Total_IC'], motif_cluster_df[metric])
    
    print(f"{metric:25s}: Pearson r = {corr:6.3f} (p = {p_val:.3e}), Spearman r = {spearman_corr:6.3f}")

# Create a DataFrame for easy sorting
correlation_df = pd.DataFrame({
    'Metric': contrast_metrics,
    'Pearson_r': [ic_correlations[m] for m in contrast_metrics],
    'P_value': [ic_pvalues[m] for m in contrast_metrics],
    'Abs_Pearson_r': [abs(ic_correlations[m]) for m in contrast_metrics]
})

# Sort by absolute correlation coefficient
correlation_df = correlation_df.sort_values('Abs_Pearson_r', ascending=False)

print("\n" + "="*60)
print("Ranking of metrics by correlation with Total_IC:")
print("="*60)
for i, row in correlation_df.iterrows():
    significance = "***" if row['P_value'] < 0.001 else "**" if row['P_value'] < 0.01 else "*" if row['P_value'] < 0.05 else ""
    print(f"{row['Metric']:25s}: r = {row['Pearson_r']:6.3f} {significance}")

# Create visualizations
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Correlation between Total Information Content and Contrast Metrics', fontsize=16)

# Plot scatter plots for each metric
for i, metric in enumerate(contrast_metrics):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    
    # Scatter plot
    ax.scatter(motif_cluster_df['Total_IC'], motif_cluster_df[metric], alpha=0.6, s=30)
    
    # Add correlation line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        motif_cluster_df['Total_IC'], motif_cluster_df[metric]
    )
    x_line = np.linspace(motif_cluster_df['Total_IC'].min(), motif_cluster_df['Total_IC'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Total Information Content')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'{metric.replace("_", " ")}\nr = {ic_correlations[metric]:.3f}')
    ax.grid(False)

plt.tight_layout()
# plt.savefig(figpath + 'ic_contrast_correlations.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Create a bar plot of correlations
plt.figure(figsize=(12, 6))
colors = ['red' if abs(r) > 0.3 else 'orange' if abs(r) > 0.2 else 'lightblue' 
          for r in correlation_df['Pearson_r']]
bars = plt.bar(range(len(correlation_df)), correlation_df['Pearson_r'], color=colors)

# Add value labels on bars
for i, (bar, r, p) in enumerate(zip(bars, correlation_df['Pearson_r'], correlation_df['P_value'])):
    height = bar.get_height()
    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
             f'{r:.3f}{significance}', ha='center', va='bottom' if height > 0 else 'top')

plt.xticks(range(len(correlation_df)), 
           [m.replace('_', '\n') for m in correlation_df['Metric']], 
           rotation=45, ha='right')
plt.ylabel('Correlation with Total IC')
plt.title('Correlation between Total Information Content and Contrast Metrics\n(* p<0.05, ** p<0.01, *** p<0.001)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.grid(False)
plt.tight_layout()
# plt.savefig(figpath + 'ic_correlation_barplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
best_metric = correlation_df.iloc[0]['Metric']
best_corr = correlation_df.iloc[0]['Pearson_r']
best_p = correlation_df.iloc[0]['P_value']

print(f"Best correlated metric: {best_metric}")
print(f"Correlation coefficient: {best_corr:.3f}")
print(f"P-value: {best_p:.3e}")
print(f"Interpretation: {'Strong' if abs(best_corr) > 0.5 else 'Moderate' if abs(best_corr) > 0.3 else 'Weak'} correlation")

# Additional analysis: Look at motifs that are outliers
print(f"\nMotifs with highest Total_IC but low {best_metric}:")
motif_cluster_df['IC_rank'] = motif_cluster_df['Total_IC'].rank(ascending=False)
motif_cluster_df['Contrast_rank'] = motif_cluster_df[best_metric].rank(ascending=False)
motif_cluster_df['Rank_diff'] = motif_cluster_df['IC_rank'] - motif_cluster_df['Contrast_rank']

# High IC but low contrast
outliers_high_ic_low_contrast = motif_cluster_df[
    (motif_cluster_df['IC_rank'] <= 20) & (motif_cluster_df['Rank_diff'] < -20)
].sort_values('Rank_diff')

if len(outliers_high_ic_low_contrast) > 0:
    print(outliers_high_ic_low_contrast[['Total_IC', best_metric, 'IC_rank', 'Contrast_rank']].head())
else:
    print("No major outliers found")

# %%
motif_cluster_df.sort_values("Gini_Index", ascending=False)

# %%
motif_cluster_df

# %%
# plt.scatter(motif_cluster_df["Motif_Length"], motif_cluster_df["Total_IC"])
# plt.xlabel("motif length (bp)")
# plt.xticks([0, 5, 10, 15, 20, 25])
# plt.ylabel("information content (bits)")
# plt.grid(False)
# plt.savefig(figpath + "motif_len_IC.pdf")
# plt.show()

from scipy.stats import pearsonr
from scipy import stats


plt.scatter(motif_cluster_df["Motif_Length"], motif_cluster_df["Total_IC"])

# Calculate linear regression (same as your code)
slope, intercept, r_value, p_value, std_err = stats.linregress(
    motif_cluster_df["Motif_Length"], motif_cluster_df["Total_IC"]
)

# Create trend line points
x_line = np.linspace(motif_cluster_df["Motif_Length"].min(), 
                     motif_cluster_df["Motif_Length"].max(), 100)
y_line = slope * x_line + intercept

# Plot trend line (red dashed line like in your code)
plt.plot(x_line, y_line, 'r--', alpha=0.8)

# Add correlation text
plt.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3e}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel("motif length (bp)")
plt.xticks([0, 5, 10, 15, 20, 25])
plt.ylabel("information content (bits)")
plt.grid(False)
plt.savefig(figpath + "motif_len_IC.pdf")
plt.show()

print(f"Linear regression: slope = {slope:.4f}, r = {r_value:.4f}, p = {p_value:.4e}")

# %%
plt.scatter(motif_cluster_df["Total_IC"], motif_cluster_df["Gini_Index"])

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    motif_cluster_df["Total_IC"], motif_cluster_df["Gini_Index"]
)

# Create trend line points
x_line = np.linspace(motif_cluster_df["Total_IC"].min(), 
                     motif_cluster_df["Total_IC"].max(), 100)
y_line = slope * x_line + intercept

# Plot trend line (red dashed line)
plt.plot(x_line, y_line, 'r--', alpha=0.8)

# Add correlation text
plt.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3e}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel("information content (bits)")
plt.ylabel("gini index")
plt.grid(False)
plt.savefig(figpath + "motif_IC_gini.pdf")
plt.show()

print(f"Linear regression: slope = {slope:.4f}, r = {r_value:.4f}, p = {p_value:.4e}")

# %%
figpath

# %%
# from scipy import stats
# def calculate_kl_divergence(row):
#     """
#     Calculate KL divergence from uniform distribution.
#     Higher value = more divergent from uniform = more cluster-specific
    
#     Parameters:
#     -----------
#     row : pandas.Series
#         A row from the motifs-by-clusters matrix
        
#     Returns:
#     --------
#     float
#         KL divergence value
#     """
#     # Convert to probability distribution
#     values = row.values
#     if np.min(values) < 0:
#         values = values - np.min(values)  # Make all values non-negative
    
#     # If all values are 0, return 0
#     if np.sum(values) == 0:
#         return 0
    
#     # Normalize to get probability distribution
#     prob_dist = values / np.sum(values)
    
#     # Uniform distribution (theoretical equal distribution)
#     uniform_dist = np.ones_like(prob_dist) / len(prob_dist)
    
#     # Calculate KL divergence
#     kl_div = 0
#     for p, q in zip(prob_dist, uniform_dist):
#         if p > 0:  # Avoid division by 0
#             kl_div += p * np.log2(p / q)
    
#     return kl_div


# kl_divergence = mr.result.apply(calculate_kl_divergence, axis=1)
# kl_divergence

# %%

# %%

# %%
