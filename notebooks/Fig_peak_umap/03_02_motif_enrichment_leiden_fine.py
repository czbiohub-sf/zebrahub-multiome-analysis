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
# - Last updated: 05/30/2025
# - We want to analyze the TF motif enrichment profiles between the peak clusters in the peak UMAP.
# -
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

# For maelstrom module
import logging
from functools import partial
from multiprocessing import Pool
from scipy.stats import pearsonr

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
# import the utility functions from gimmemotifs maelstrom
from utils_gimme_maelstrom_custom import *

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

# %%
dict_adata_sub[13]

# %% [markdown]
# ## Sanity check the data

# %%
# move the counts to GPU
rsc.get.anndata_to_GPU(adata_peaks)

# %%
# plot the coarse leiden clusters
sc.pl.embedding(adata_peaks, basis="X_umap_2D", color="leiden_coarse")#, save="_peaks_leiden_coarse.png")

# %%
## merge the fine sub-cluster annotations with the "leiden_coarse" annotations
# Initialize unified labels with coarse cluster labels
unified_labels = adata_peaks.obs['leiden_coarse'].astype(str).copy()

# For each sub-clustered coarse cluster, update with fine labels
# "leiden_coarse" clusters
coarse_clusts = [1,7,13,22]

for clust in coarse_clusts:
    if clust in dict_adata_sub:
        adata_sub = dict_adata_sub[clust]
        
        # Get the cell indices that belong to this coarse cluster
        coarse_mask = adata_peaks.obs['leiden_coarse'] == clust
        coarse_cell_indices = adata_peaks.obs.index[coarse_mask]
        
        # Check if the sub-clustered data has the fine clustering column
        if 'leiden_sub_0.7_merged_renumbered' in adata_sub.obs.columns:
            # Create mapping from cell index to fine cluster label
            for cell_idx in coarse_cell_indices:
                if cell_idx in adata_sub.obs.index:
                    fine_label = adata_sub.obs.loc[cell_idx, 'leiden_sub_0.7_merged_renumbered']
                    unified_labels.loc[cell_idx] = f"{clust}_{fine_label}"
                    
            print(f"Updated cluster {clust} with fine labels")
        else:
            print(f"Warning: 'leiden_sub_0.7_merged_renumbered' not found in cluster {clust}")

# Add to adata_peaks
adata_peaks.obs['leiden_unified'] = unified_labels

# Check the results
print("\nUnified label distribution:")
print(adata_peaks.obs['leiden_unified'].value_counts().sort_index())

# %%
sc.pl.umap(adata_peaks, color="leiden_unified", legend_loc="on data")

# %%

# %% [markdown]
# ## Step 2. Motif enrichment analysis (gimme maelstrom)
# - export the peaks as a txt file for gimmemotifs maelstrom
# - Run gimme maelstrom using slurm (on HPC)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### export peaks with cluster labels as txt format

# %%
# load the utility functions for exporting the peaks:clusters df to txt format
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs/")
from utils_maelstrom import export_peaks_for_gimmemotifs

# %%
clust_label = "leiden_unified"
out_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_fine_cisBP_v2_danio_rerio_output/"
os.makedirs(out_dir, exist_ok=True)

# export the peaks with the labels
export_peaks_for_gimmemotifs(adata_peaks, cluster=clust_label, 
                             out_dir = out_dir,
                             out_name= f"leiden_coarse_fine_unified")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Run GimmeMotifs maelstrom for differential motif computation
# - This is done on HPC using Slurm
# - reference: https://gimmemotifs.readthedocs.io/en/master/tutorials.html#find-differential-motifs
#
# - We will run gimme maelstrom for each "leiden_coarse" cluster, for their sub-clusters.

# %%
# # !sbatch gimme_maelstrom_modular.sh --input /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/adata_peak_clusters_subset/peaks_leiden_coarse_1_subclust.txt --ref_genome danRer11 --output_dir /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_leiden_coarse_1_cisBP_v2_output/ --pfmfile CisBP_ver2_Danio_rerio

# %%
coarse_clusts

# %%
## Run the GimmeMotifs maelstrom using Slurm on HPC
# !sbatch /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/slurm_scripts/gimme_maelstrom.sh

# %% [markdown]
# ## Step 3. Check the Maelstrom output:
#

# %%
# import the result of the maelstrom
from gimmemotifs.maelstrom import MaelstromResult

# %%
# import the maelstrom result (peaks-by-motifs count matrix, and clusters-by-motifs)
mr = MaelstromResult("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_fine_cisBP_v2_danio_rerio_output/")
mr.scores.head()

# %%

# %%
# 1) Filter based on threshold
threshold = 4 # for example
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
plt.savefig(figpath + "leiden_coarse_fine_motifs_heatmap_thresh_4_clustered_transposed.png")
plt.savefig(figpath + "leiden_coarse_fine_motifs_heatmap_thresh_4_clustered_transposed.pdf")
plt.show()

# %% [markdown]
# ## create mapping for the sub-clusters

# %% [markdown]
# ### example: cluster 13

# %%
# 1) Filter based on threshold
# threshold = 3.5 # for example
df = mr.result.copy()
# 2) Keep only those rows where at least one absolute value >= threshold
# mask = df.abs().ge(threshold).any(axis=1)
# df_filt = df[mask]
df_filt = df
# Example: remove "z-score " prefix
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]
# transpose the matrix
df_filt_trans = df_filt.transpose()
# Filter columns (clusters) that start with "13" or "22"
cluster_mask = df_filt_trans.index.str.startswith(('13'))
df_subset = df_filt_trans[cluster_mask]
print(f"Original clusters: {df_filt_trans.shape[0]}")
print(f"Filtered clusters: {df_subset.shape[0]}")
print(f"Clusters included: {list(df_subset.index)}")

# Additional filtering: Keep only TFs (columns) where max absolute value >= threshold
tf_threshold = 3.7  # Adjust this threshold as needed
# CORRECTED: Filter based on maximum absolute value across clusters (axis=0)
tf_mask = df_subset.abs().max(axis=0) >= tf_threshold
df_subset_filtered = df_subset.loc[:, tf_mask]  # Use .loc[:, tf_mask] to filter columns

print(f"Original TFs: {df_subset.shape[1]}")
print(f"Filtered TFs: {df_subset_filtered.shape[1]}")
print(f"TFs removed: {df_subset.shape[1] - df_subset_filtered.shape[1]}")

# Create the clustered heatmap for the subset
g_subset = sns.clustermap(
    df_subset_filtered,
    metric="euclidean",   
    method="ward",        
    cmap="RdBu_r",        
    center=0,             
    linewidths=0.5,
    figsize=(9, 6),       # Might want smaller height since fewer clusters
    xticklabels=True,
    yticklabels=True
)
# Rotate labels
plt.setp(g_subset.ax_heatmap.get_xticklabels(), rotation=90)
plt.setp(g_subset.ax_heatmap.get_yticklabels(), rotation=0)
# Save the subset heatmap
plt.savefig(figpath + "leiden_clusters_13_motifs_heatmap_thresh_3.5_clustered.png")
plt.savefig(figpath + "leiden_clusters_13_motifs_heatmap_thresh_3.5_clustered.pdf")
plt.show()

# Optional: Show which TFs were kept
print(f"\nTFs kept after filtering (threshold >= {tf_threshold}):")
print(f"Total: {len(df_subset_filtered.columns)}")
print("TF names:", list(df_subset_filtered.columns[:5]), "..." if len(df_subset_filtered.columns) > 5 else "")

# %%
# Filter adata_peaks for cluster 13 peaks
adata_cluster_mask = adata_peaks.obs['leiden_unified'].astype(str).str.startswith('13_')
adata_cluster = adata_peaks[adata_cluster_mask].copy()
adata_cluster

# Map motif enrichment scores to individual peaks
# For each TF motif, create a mapping from leiden_unified to enrichment score
for motif in df_subset_filtered.columns:
    # Create mapping dictionary: cluster -> enrichment score
    cluster_to_score = df_subset_filtered[motif].to_dict()
    
    # Map scores to peaks based on their leiden_unified label
    motif_scores = adata_cluster.obs['leiden_unified'].map(cluster_to_score)
    
    # Add to adata.obs (clean motif name for column)
    motif_clean = motif.replace('_2.00', '').replace(':', '_')
    adata_cluster.obs[f'motif_{motif_clean}'] = motif_scores

print(f"Added {len(df_subset_filtered.columns)} motif enrichment scores to adata_cluster.obs")

# %%
df_subset_filtered.columns

# %%
motif_list = ['motif_' + motif for motif in df_subset_filtered.columns]
motif_list = [motif.replace('_2.00', '') for motif in motif_list]
motif_list

# %%
# Convert motif scores from categorical to numerical
for motif in motif_list:
    if motif in adata_cluster.obs.columns:
        # Convert to numeric, handling any potential string values
        adata_cluster.obs[motif] = pd.to_numeric(adata_cluster.obs[motif], errors='coerce')
        
        # Alternatively, if they're stored as categorical:
        # adata_cluster.obs[motif] = adata_cluster.obs[motif].astype(float)

# %%
adata_cluster

# %%
cluster_coarse = 13
# Convert motif scores from categorical to numerical
for motif in motif_list:
    if motif in adata_cluster.obs.columns:
        # Convert to numeric, handling any potential string values
        adata_cluster.obs[motif] = pd.to_numeric(adata_cluster.obs[motif], errors='coerce')
        
sc.pl.umap(adata_cluster, color=["motif_M02899","motif_M06254"],
           cmap="RdBu_r", vmin=-4, vmax=4,
           save=f"_peaks_cluster_{cluster_coarse}_example_motifs.png")

# %%
from gimmemotifs.motif import Motif, read_motifs

# Load motifs from the default database
motif_file = "/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm"  # Use the correct database file
motifs = read_motifs(motif_file)

# %%
print(f"the number of motifs is: {len(motifs)}")

# %%
# Get the specific motif (e.g., "GM.5.0.Ets.0033")
motif_name = "M02899_2.00"
selected_motif = next(m for m in motifs if m.id == motif_name)

# Extract the Position Frequency Matrix (PFM) and Position Weight Matrix (PWM)
pfm = selected_motif.to_pfm()  # Transpose to make it compatible with logomaker
pwm = selected_motif.to_ppm()

# consensus sequence
print(selected_motif.to_consensus())
print(pwm)

# %%
selected_motif.factor_info


# %%
# define the information content using Schneider-Stephens/column-KL style
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
# Get the specific motif (e.g., "GM.5.0.Ets.0033")
# motif_name = "M02899_2.00"
motif_name = "M08968_2.00"
selected_motif = next(m for m in motifs if m.id == motif_name)

# Extract the Position Frequency Matrix (PFM) and Position Weight Matrix (PWM)
pfm = selected_motif.to_pfm()  # Transpose to make it compatible with logomaker
pwm = selected_motif.to_ppm()

# consensus sequence
print(selected_motif.to_consensus())
print(pwm)

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
plt.savefig(figpath + f"{motif_name}.seq.logo.png")
plt.savefig(figpath + f"{motif_name}.seq.logo.pdf")
plt.show()

# %% [markdown]
# ### cluster 22

# %%
# 1) Filter based on threshold
threshold = 4 # for example
df = mr.result.copy()

# 2) Keep only those rows where at least one absolute value >= threshold
mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]
# df_filt = df
# Example: remove "z-score " prefix
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]

# transpose the matrix
df_filt_trans = df_filt.transpose()

# Filter columns (clusters) that start with "13" or "22"
cluster_mask = df_filt_trans.index.str.startswith(('22'))
df_subset = df_filt_trans[cluster_mask]

print(f"Original clusters: {df_filt_trans.shape[0]}")
print(f"Filtered clusters: {df_subset.shape[0]}")
print(f"Clusters included: {list(df_subset.index)}")

# Additional filtering: Keep only TFs where max absolute value >= threshold
tf_threshold = 4  # Adjust this threshold as needed

# Option 1: Filter based on maximum absolute value across the subset
tf_mask = df_subset.abs().max(axis=1) >= tf_threshold
df_subset_filtered = df_subset[tf_mask]


# Create the clustered heatmap for the subset
g_subset = sns.clustermap(
    df_subset_filtered,
    metric="euclidean",   
    method="ward",        
    cmap="RdBu_r",        
    center=0,             
    linewidths=0.5,
    figsize=(9, 6),       # Might want smaller height since fewer clusters
    xticklabels=True,
    yticklabels=True
)

# Rotate labels
plt.setp(g_subset.ax_heatmap.get_xticklabels(), rotation=90)
plt.setp(g_subset.ax_heatmap.get_yticklabels(), rotation=0)

# Save the subset heatmap
plt.savefig(figpath + "leiden_clusters_22_motifs_heatmap_thresh_3.5_clustered.png")
plt.savefig(figpath + "leiden_clusters_22_motifs_heatmap_thresh_3.5_clustered.pdf")
plt.show()

# %%
# Filter adata_peaks for cluster 22 peaks
adata_cluster_mask = adata_peaks.obs['leiden_unified'].astype(str).str.startswith('22_')
adata_cluster = adata_peaks[adata_cluster_mask].copy()
adata_cluster

# Map motif enrichment scores to individual peaks
# For each TF motif, create a mapping from leiden_unified to enrichment score
for motif in df_subset_filtered.columns:
    # Create mapping dictionary: cluster -> enrichment score
    cluster_to_score = df_subset_filtered[motif].to_dict()
    
    # Map scores to peaks based on their leiden_unified label
    motif_scores = adata_cluster.obs['leiden_unified'].map(cluster_to_score)
    
    # Add to adata.obs (clean motif name for column)
    motif_clean = motif.replace('_2.00', '').replace(':', '_')
    adata_cluster.obs[f'motif_{motif_clean}'] = motif_scores

print(f"Added {len(df_subset_filtered.columns)} motif enrichment scores to adata_cluster.obs")

# %%
motif_list = ['motif_' + motif for motif in df_subset_filtered.columns]
motif_list = [motif.replace('_2.00', '') for motif in motif_list]
motif_list

# %%
# Convert motif scores from categorical to numerical
for motif in motif_list:
    if motif in adata_cluster.obs.columns:
        # Convert to numeric, handling any potential string values
        adata_cluster.obs[motif] = pd.to_numeric(adata_cluster.obs[motif], errors='coerce')
        
sc.pl.umap(adata_cluster, color=motif_list,
           cmap="RdBu_r", vmin=-4, vmax=4, ncols=3)
#            save=f"_peaks_cluster_{cluster_coarse}_example_motifs.png")

# %%

# %%
cluster_coarse = 22

sc.pl.umap(adata_cluster, color=["motif_M09419","motif_M08129"],
           cmap="RdBu_r", vmin=-4, vmax=4, 
           save=f"_peaks_cluster_{cluster_coarse}_leiden.png")

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

# %% [markdown]
# ### create an adata with adata.obs["enrichment_scores"]

# %%
# # copy the adata object so that we can annotate the peaks by their motif enrichment scores
adata_peaks_motifs = adata_peaks.copy()

# Convert index from 'chr1:start-end' to '1-start-end'
mr_scores = mr.scores
mr_scores.index = mr_scores.index.str.replace(r'chr(\d+):', r'\1-', regex=True)

# map the columns to the adata obs field
for col in mr_scores.columns:
    adata_peaks_motifs.obs[col] = adata_peaks_motifs.obs_names.map(mr_scores[col])


# %% [markdown]
# ### example: cluster 7

# %%
# 1) Filter based on threshold
# threshold = 3.5 # for example
df = mr.result.copy()
# 2) Keep only those rows where at least one absolute value >= threshold
# mask = df.abs().ge(threshold).any(axis=1)
# df_filt = df[mask]
df_filt = df
# Example: remove "z-score " prefix
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]
# transpose the matrix
df_filt_trans = df_filt.transpose()
# Filter columns (clusters) that start with "13" or "22"
cluster_mask = df_filt_trans.index.str.startswith(('7'))
df_subset = df_filt_trans[cluster_mask]
print(f"Original clusters: {df_filt_trans.shape[0]}")
print(f"Filtered clusters: {df_subset.shape[0]}")
print(f"Clusters included: {list(df_subset.index)}")

# Additional filtering: Keep only TFs (columns) where max absolute value >= threshold
tf_threshold = 3.5  # Adjust this threshold as needed
# CORRECTED: Filter based on maximum absolute value across clusters (axis=0)
tf_mask = df_subset.abs().max(axis=0) >= tf_threshold
df_subset_filtered = df_subset.loc[:, tf_mask]  # Use .loc[:, tf_mask] to filter columns

print(f"Original TFs: {df_subset.shape[1]}")
print(f"Filtered TFs: {df_subset_filtered.shape[1]}")
print(f"TFs removed: {df_subset.shape[1] - df_subset_filtered.shape[1]}")

# Create the clustered heatmap for the subset
g_subset = sns.clustermap(
    df_subset_filtered,
    metric="euclidean",   
    method="ward",        
    cmap="RdBu_r",        
    center=0,             
    linewidths=0.5,
    figsize=(9, 6),       # Might want smaller height since fewer clusters
    xticklabels=True,
    yticklabels=True
)
# Rotate labels
plt.setp(g_subset.ax_heatmap.get_xticklabels(), rotation=90)
plt.setp(g_subset.ax_heatmap.get_yticklabels(), rotation=0)
# Save the subset heatmap
plt.savefig(figpath + "leiden_clusters_7_motifs_heatmap_thresh_3.5_clustered.png")
plt.savefig(figpath + "leiden_clusters_7_motifs_heatmap_thresh_3.5_clustered.pdf")
plt.show()

# Optional: Show which TFs were kept
print(f"\nTFs kept after filtering (threshold >= {tf_threshold}):")
print(f"Total: {len(df_subset_filtered.columns)}")
print("TF names:", list(df_subset_filtered.columns[:5]), "..." if len(df_subset_filtered.columns) > 5 else "")

# %%
# Filter adata_peaks for cluster 13 peaks
adata_cluster_mask = adata_peaks.obs['leiden_unified'].astype(str).str.startswith('7_')
adata_cluster = adata_peaks[adata_cluster_mask].copy()
adata_cluster

# Map motif enrichment scores to individual peaks
# For each TF motif, create a mapping from leiden_unified to enrichment score
for motif in df_subset_filtered.columns:
    # Create mapping dictionary: cluster -> enrichment score
    cluster_to_score = df_subset_filtered[motif].to_dict()
    
    # Map scores to peaks based on their leiden_unified label
    motif_scores = adata_cluster.obs['leiden_unified'].map(cluster_to_score)
    
    # Add to adata.obs (clean motif name for column)
    motif_clean = motif.replace('_2.00', '').replace(':', '_')
    adata_cluster.obs[f'motif_{motif_clean}'] = motif_scores

print(f"Added {len(df_subset_filtered.columns)} motif enrichment scores to adata_cluster.obs")

# %%
df_subset_filtered.columns

# %%
motif_list = ['motif_' + motif for motif in df_subset_filtered.columns]
motif_list = [motif.replace('_2.00', '') for motif in motif_list]
motif_list

# %%
# Convert motif scores from categorical to numerical
for motif in motif_list:
    if motif in adata_cluster.obs.columns:
        # Convert to numeric, handling any potential string values
        adata_cluster.obs[motif] = pd.to_numeric(adata_cluster.obs[motif], errors='coerce')
        
        # Alternatively, if they're stored as categorical:
        # adata_cluster.obs[motif] = adata_cluster.obs[motif].astype(float)

# %%
adata_cluster

# %%
cluster_coarse = 7
# Convert motif scores from categorical to numerical
for motif in motif_list:
    if motif in adata_cluster.obs.columns:
        # Convert to numeric, handling any potential string values
        adata_cluster.obs[motif] = pd.to_numeric(adata_cluster.obs[motif], errors='coerce')
        
sc.pl.umap(adata_cluster, color=motif_list,
           cmap="RdBu_r", vmin=-4, vmax=4, ncols=3)
#            save=f"_peaks_cluster_{cluster_coarse}_example_motifs.png")

# %%
# Get the specific motif (e.g., "GM.5.0.Ets.0033")
motif_name = "M02899_2.00"
selected_motif = next(m for m in motifs if m.id == motif_name)

# Extract the Position Frequency Matrix (PFM) and Position Weight Matrix (PWM)
pfm = selected_motif.to_pfm()  # Transpose to make it compatible with logomaker
pwm = selected_motif.to_ppm()

# consensus sequence
print(selected_motif.to_consensus())
print(pwm)

# %%

# %% [markdown]
# ### example: cluster 1

# %%
# 1) Filter based on threshold
# threshold = 3.5 # for example
df = mr.result.copy()
# 2) Keep only those rows where at least one absolute value >= threshold
# mask = df.abs().ge(threshold).any(axis=1)
# df_filt = df[mask]
df_filt = df
# Example: remove "z-score " prefix
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]
# transpose the matrix
df_filt_trans = df_filt.transpose()
# Filter columns (clusters) that start with "13" or "22"
cluster_mask = df_filt_trans.index.str.startswith(('1_'))
df_subset = df_filt_trans[cluster_mask]
print(f"Original clusters: {df_filt_trans.shape[0]}")
print(f"Filtered clusters: {df_subset.shape[0]}")
print(f"Clusters included: {list(df_subset.index)}")

# Additional filtering: Keep only TFs (columns) where max absolute value >= threshold
tf_threshold = 3.5  # Adjust this threshold as needed
# CORRECTED: Filter based on maximum absolute value across clusters (axis=0)
tf_mask = df_subset.abs().max(axis=0) >= tf_threshold
df_subset_filtered = df_subset.loc[:, tf_mask]  # Use .loc[:, tf_mask] to filter columns

print(f"Original TFs: {df_subset.shape[1]}")
print(f"Filtered TFs: {df_subset_filtered.shape[1]}")
print(f"TFs removed: {df_subset.shape[1] - df_subset_filtered.shape[1]}")

# Create the clustered heatmap for the subset
g_subset = sns.clustermap(
    df_subset_filtered,
    metric="euclidean",   
    method="ward",        
    cmap="RdBu_r",        
    center=0,             
    linewidths=0.5,
    figsize=(9, 6),       # Might want smaller height since fewer clusters
    xticklabels=True,
    yticklabels=True
)
# Rotate labels
plt.setp(g_subset.ax_heatmap.get_xticklabels(), rotation=90)
plt.setp(g_subset.ax_heatmap.get_yticklabels(), rotation=0)
# Save the subset heatmap
plt.savefig(figpath + "leiden_clusters_1_motifs_heatmap_thresh_3.5_clustered.png")
plt.savefig(figpath + "leiden_clusters_1_motifs_heatmap_thresh_3.5_clustered.pdf")
plt.show()

# Optional: Show which TFs were kept
print(f"\nTFs kept after filtering (threshold >= {tf_threshold}):")
print(f"Total: {len(df_subset_filtered.columns)}")
print("TF names:", list(df_subset_filtered.columns[:5]), "..." if len(df_subset_filtered.columns) > 5 else "")

# %%
# Filter adata_peaks for cluster 13 peaks
adata_cluster_mask = adata_peaks.obs['leiden_unified'].astype(str).str.startswith('1_')
adata_cluster = adata_peaks[adata_cluster_mask].copy()
adata_cluster

# Map motif enrichment scores to individual peaks
# For each TF motif, create a mapping from leiden_unified to enrichment score
for motif in df_subset_filtered.columns:
    # Create mapping dictionary: cluster -> enrichment score
    cluster_to_score = df_subset_filtered[motif].to_dict()
    
    # Map scores to peaks based on their leiden_unified label
    motif_scores = adata_cluster.obs['leiden_unified'].map(cluster_to_score)
    
    # Add to adata.obs (clean motif name for column)
    motif_clean = motif.replace('_2.00', '').replace(':', '_')
    adata_cluster.obs[f'motif_{motif_clean}'] = motif_scores

print(f"Added {len(df_subset_filtered.columns)} motif enrichment scores to adata_cluster.obs")

# %%
df_subset_filtered.columns

# %%
motif_list = ['motif_' + motif for motif in df_subset_filtered.columns]
motif_list = [motif.replace('_2.00', '') for motif in motif_list]
motif_list

# %%
# Convert motif scores from categorical to numerical
for motif in motif_list:
    if motif in adata_cluster.obs.columns:
        # Convert to numeric, handling any potential string values
        adata_cluster.obs[motif] = pd.to_numeric(adata_cluster.obs[motif], errors='coerce')
        
        # Alternatively, if they're stored as categorical:
        # adata_cluster.obs[motif] = adata_cluster.obs[motif].astype(float)

# %%
cluster_coarse = 1
# Convert motif scores from categorical to numerical
for motif in motif_list:
    if motif in adata_cluster.obs.columns:
        # Convert to numeric, handling any potential string values
        adata_cluster.obs[motif] = pd.to_numeric(adata_cluster.obs[motif], errors='coerce')
        
sc.pl.umap(adata_cluster, color=motif_list,
           cmap="RdBu_r", vmin=-4, vmax=4, ncols=4)
#            save=f"_peaks_cluster_{cluster_coarse}_example_motifs.png")

# %%
# Get the specific motif (e.g., "GM.5.0.Ets.0033")
motif_name = "M02899_2.00"
selected_motif = next(m for m in motifs if m.id == motif_name)

# Extract the Position Frequency Matrix (PFM) and Position Weight Matrix (PWM)
pfm = selected_motif.to_pfm()  # Transpose to make it compatible with logomaker
pwm = selected_motif.to_ppm()

# consensus sequence
print(selected_motif.to_consensus())
print(pwm)

# %%

# %%
