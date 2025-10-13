# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: sc_rapids
#     language: python
#     name: sc_rapids
# ---

# %% [markdown]
# # EDA on extracting sub-GRNs from the regulatory programs
#
# - last updated: 8/4/2025
#
# - inputs: 
#     - fine cluster-by-motifs matrix (from GimmeMotifs scanning)
#     - fine cluster-by-linked_genes (which was computed from peaks-by-linked_genes - from Signac)
#         - From the Take 1, we decided to use "linked_genes" instead of "associated_genes" (which is a union of linked_genes and genes overlapping with the peaks).
#     - CellOracle GRNs: GRN[celltype, timepoint]
#     
#     
# - outputs (per each peak cluster):
#     - Mesh: a binarized mini-GRN for each peak cluster
#     - a list of prominent celltypes&timepoints for each peak cluster (where the reg.program is enriched)
#     - subGRN (an intersection between a GRN(celltype,timepoint) and a Mesh -> function of peak_cluster, celltype, timepoint
#     
#
# ### organization
# - 1) check the inputs: 
#     - [DONE] check the stats for "linked_genes"
#     - [DONE] check the cluster-by-motifs from gimme maelstrom run ("leiden_unified")
#         - [DONE] thresholding for the z-scored enriched scores
#
# - 2) EDA on each input type
#     - 
#
# - 3) [DONE] Establish a workflow to extract a list of TFs and associated genes for each peak cluster (a mesh)
#     - turn this into a python module
# - 4) [DONE] Compute the intersection with the mesh and the GRN to get a sub-GRN
# - 5) systematic analysis on how the subGRNs change over time and cell-types. 
#     - (1) identify biologically interesting subGRNs
#     - (2) use ChatGPT or Claude to identify biologically novel subGRNs - potentially novel, previously unknown interactions between TFs and genes.
#     - (3) turn this into a function for systematic exploration
#     - (4) brainstorm and test out "LLM Agents" to find a quantitative metric to score the subGRN dynamics over time or celltypes.
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
import os
import pickle

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
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
# NOTE. the 2 MT peaks and 2 blacklisted peaks (since they go beyond the end of the chromosome) were filtered out.
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad")
adata_peaks

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## step 1. curate the inputs
#
# - TO-DO: make sure to save all the input files in a directory for systematic querying.
#

# %%
sc.pl.umap(adata_peaks, color="leiden_coarse")

# %%
adata_peaks.obs["leiden_unified"].value_counts()

# %%
plt.hist(adata_peaks.obs["leiden_unified"].value_counts(), bins=30)
plt.xlabel("number of peaks (per cluster)")
plt.ylabel("occurences")
plt.grid(False)
plt.savefig(figpath + "hist_num_peaks_per_fine_clusters.pdf")
plt.show()

# %%
num_linked_genes = len(adata_peaks.obs["linked_gene"].unique())
adata_peaks.obs["linked_gene"].value_counts().hist(bins=30)
plt.xlabel("number of peaks per gene")
plt.ylabel("occurences")
plt.title(f"total number of linked genes: {num_linked_genes}")
plt.grid(False)
plt.show()

# %%
# check the associated_gene, linked_gene statistics for "leiden_unified" category
plt.hist(adata_peaks.obs["leiden_unified"].value_counts(), bins=30)
plt.xlabel("number of peaks (per cluster)")
plt.ylabel("occurences")
plt.grid(False)
plt.savefig(figpath + "hist_num_peaks_per_fine_clusters.pdf")
plt.show()

# %%
adata_peaks

# %% [markdown]
# ## 1. define the over-represented "motifs" per peak cluster
#
# ### Use the GimmeMotifs maelstrom result on "leiden_unified"
# - gimme maelstrom on "leiden_unified" gives motif enrichment scores per "fine" clusters.
#

# %%
# import the clusters-by-motifs matrix from gimmemotifs maelstrom output mr.result
clust_by_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_fine_motifs_maelstrom.csv",
                              index_col=0)
clust_by_motifs.head()

# %%
# check the top motifs for each cluster
plt.hist(clust_by_motifs.loc["0_0",:], bins=20)
plt.xlabel("enrichment score (from one cluster)")
plt.ylabel("occurences")
plt.grid(False)
# plt.savefig(figpath + "hist_motif_enrich_score_clust_0_0.pdf")
plt.show()

# %%
# Test different threshold values
thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

threshold_stats = []
for thresh in thresholds:
    # Count significant motifs per cluster
    sig_counts = (clust_by_motifs >= thresh).sum(axis=1)
    
    stats = {
        'threshold': thresh,
        'mean_motifs_per_cluster': sig_counts.mean(),
        'median_motifs_per_cluster': sig_counts.median(),
        'max_motifs_per_cluster': sig_counts.max(),
        'clusters_with_no_motifs': (sig_counts == 0).sum(),
        'total_significant_pairs': (clust_by_motifs >= thresh).sum().sum()
    }
    threshold_stats.append(stats)

# Convert to DataFrame for easy viewing
threshold_df = pd.DataFrame(threshold_stats)
print(threshold_df)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Motifs per cluster vs threshold
axes[0,0].plot(threshold_df['threshold'], threshold_df['mean_motifs_per_cluster'], 'o-', label='Mean')
axes[0,0].plot(threshold_df['threshold'], threshold_df['median_motifs_per_cluster'], 's-', label='Median')
axes[0,0].set_xlabel('Threshold')
axes[0,0].set_ylabel('Motifs per cluster')
axes[0,0].set_title('Average motifs per cluster')
axes[0,0].grid(False)
axes[0,0].legend()

# Plot 2: Clusters with no significant motifs
axes[0,1].plot(threshold_df['threshold'], threshold_df['clusters_with_no_motifs'], 'ro-')
axes[0,1].set_xlabel('Threshold')
axes[0,1].set_ylabel('Clusters with 0 motifs')
axes[0,1].set_title('Clusters losing all signal')
axes[0,1].grid(False)

# Plot 3: Total significant pairs
axes[1,0].plot(threshold_df['threshold'], threshold_df['total_significant_pairs'], 'go-')
axes[1,0].set_xlabel('Threshold')
axes[1,0].set_ylabel('Total significant pairs')
axes[1,0].set_title('Overall signal retained')
axes[1,0].grid(False)


# Plot 4: Distribution at different thresholds with KDE
thresh_to_plot = [1.0, 2.0, 3.0]
colors = ['blue', 'orange', 'green']

# --- decide on common bin edges once ----------------------------------------
sig_counts_all = np.concatenate([
    (clust_by_motifs >= t).sum(axis=1) for t in thresh_to_plot      # [1.0, 2.0, 3.0]
])
bin_edges = np.arange(sig_counts_all.min() - 0.5,
                      sig_counts_all.max() + 1.5, 1)                # bar width = 1

# --- panel 4: histogram + KDE -----------------------------------------------
ax = axes[1, 1]                                                     # convenience alias
for thresh, color in zip(thresh_to_plot, colors):
    sig_counts = (clust_by_motifs >= thresh).sum(axis=1)

    # histogram (same bin_edges for every series → equal bar widths)
    ax.hist(sig_counts,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            color=color,
            edgecolor=color,
            label=f'Threshold {thresh}')

    # KDE overlay
    if len(sig_counts) > 1 and sig_counts.std() > 0:
        sns.kdeplot(sig_counts,
                    ax=ax,
                    color=color,
                    linewidth=2,
                    bw_adjust=1.5)

# cosmetic details
ax.set_xlabel('Significant motifs per cluster')
ax.set_ylabel('Density')
ax.set_title('Distribution of motif counts (Histogram + KDE)')
ax.grid(False)                       # keep grid off, as in your other panels
ax.legend(frameon=False)             # match the “no box” look of the first figure


plt.tight_layout()
plt.savefig(figpath + "EDA_motif_thresholds.pdf")
plt.show()

# %%
# Calculate pairwise similarity between clusters
from scipy.spatial.distance import pdist, squareform

def cluster_similarity_analysis(threshold):
    # Get binary matrix of significant motifs
    sig_matrix = (clust_by_motifs >= threshold).astype(int)
    
    # Calculate Jaccard similarity between clusters
    # (intersection / union)
    similarities = []
    cluster_names = []
    
    for i, cluster1 in enumerate(sig_matrix.index):
        for j, cluster2 in enumerate(sig_matrix.index):
            if i < j:  # Only upper triangle
                motifs1 = set(sig_matrix.columns[sig_matrix.loc[cluster1] == 1])
                motifs2 = set(sig_matrix.columns[sig_matrix.loc[cluster2] == 1])
                
                if len(motifs1) == 0 and len(motifs2) == 0:
                    jaccard = 1.0  # Both empty
                elif len(motifs1 | motifs2) == 0:
                    jaccard = 0.0
                else:
                    jaccard = len(motifs1 & motifs2) / len(motifs1 | motifs2)
                
                similarities.append(jaccard)
                cluster_names.append(f"{cluster1}-{cluster2}")
    
    return similarities, cluster_names

# Compare cluster distinctiveness at different thresholds
thresholds_test = [1.0, 1.5, 2.0, 2.5]

for thresh in thresholds_test:
    similarities, names = cluster_similarity_analysis(thresh)
    
    print(f"\n=== THRESHOLD {thresh} ===")
    print(f"Mean cluster similarity: {np.mean(similarities):.3f}")
    print(f"Max cluster similarity: {np.max(similarities):.3f}")
    print(f"Clusters with >50% similarity: {sum([s > 0.5 for s in similarities])}")
    
    # Show most similar cluster pairs
    if len(similarities) > 0:
        most_similar_idx = np.argmax(similarities)
        print(f"Most similar pair: {names[most_similar_idx]} (similarity: {similarities[most_similar_idx]:.3f})")

# %% [markdown]
# ## Step 2. construct a TF-gene mesh for each fine peak cluster 
# - First, prototype here, then script as a workflow later.
# - 

# %%
from module_extract_subGRN import *

# %%
# How to run the workflow step-by-step
""" 
# Step 1
clusters_motifs_dict = get_top_motifs_per_cluster(clust_by_motifs, threshold_value=2)

# Step 2  
clusters_tfs_dict = get_tfs_from_motifs(clusters_motifs_dict, info_motifs)

# Step 3
clusters_genes_dict = get_associated_genes_per_cluster(clusters_genes_df, 0.5)

# Step 4
cluster_tf_gene_matrices = create_tf_gene_matrix_per_cluster(clusters_tfs_dict, clusters_genes_dict)
"""

# %%
# Step 1. extract the enriched motifs per each cluster (above some threshold z-score, default is 2)
# # output is a dict mapping the list of motifs for each peak cluster
clusters_motifs_dict = get_top_motifs_per_cluster(clust_by_motifs, threshold_value=2)

# %% [markdown]
# ### [DONE] double-check the motif:TF database for the litemind inputs
#
# ### [DEPRECATED]Import the motif:TF dataframe (CisBP_v2_danio_rerio)
# - this should be revisited and simplified later as there are too many redundant motifs (5000 motifs, where the maelstrom reduced it to 110 motifs for "leiden_coarse" clusters)

# %%
# import the motif:TF dataframe from maelstrom (CisBP_v2_danio_rerio)
# NOTE that gimmemotifs applies TFClass domain homology + PWM similarity to attach every TF
# whose DNA-binding domain can plausibly bind that motif. (hence, explosion of TFs for Hox motifs)
info_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_unified_cisBP_ver2_Danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors.csv", index_col=0)
info_motifs.head()

# %% [markdown]
# ### [USE THIS] load the master motif:TF dataframe

# %%
# import the master motif:TF dataframe (5000 motifs, with redundancy, also, note that the Motif column should be aggregated)
info_motifs_TFs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/CisBP_ver2_Danio_rerio.motif2factors.txt", 
                              sep="\t")
info_motifs_TFs.head()

# %%
from module_motif_db import *
# # convert to a dict (motifs:TFs)
# motif_dict = create_motif_to_factors_dict(info_motifs_TFs)

# %%
# convert to a dataframe (motifs:TFs)
info_motifs_df = create_motif_factors_dataframe(info_motifs_TFs)

# %%
# reformat the dataframe
info_motifs_df = info_motifs_TFs.groupby('Motif')['Factor'].apply(lambda x: ', '.join(x)).reset_index()
info_motifs_df.columns = ['motif', 'indirect']
info_motifs_df['direct'] = 'None'
info_motifs_df = info_motifs_df[['direct', 'motif', 'indirect']]  # reorder columns to match your target format
info_motifs_df.set_index("motif", inplace=True)
info_motifs_df.head()

# %% [markdown]
# ### CHECK the Motif:TF dataframes

# %%
print(set(clust_by_motifs.columns) - set(info_motifs.index))
print(set(info_motifs.index) - set(clust_by_motifs.columns))

# %%
print(set(clust_by_motifs.columns) - set(info_motifs_df.index))
print(len(set(info_motifs_df.index) - set(clust_by_motifs.columns)))

# %%
info_motifs_filtered = info_motifs[info_motifs.index.isin(info_motifs_df.index)]
info_motifs_filtered.head()

# %%
info_motifs_filtered.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/info_cisBP_v2_danio_rerio_motif_factors_fine_clusts.csv")

# %%
print(len(info_motifs.loc["M00199_2.00"]["indirect"]))
print(len(info_motifs_df.loc["M00199_2.00"]["indirect"]))

# %% [markdown]
# ## Step 2. Map the Motifs to associated TFs (cluster-by-enriched TFs)

# %%
# Step 2. map the motifs to associated TFs
clusters_tfs_dict = get_tfs_from_motifs(clusters_motifs_dict, info_motifs_df)

# %%
# Extract the number of TFs per peak cluster
cluster_tf_counts = {cluster_id: len(tfs) for cluster_id, tfs in clusters_tfs_dict.items()}

# Convert to lists for plotting
cluster_ids = list(cluster_tf_counts.keys())
tf_counts = list(cluster_tf_counts.values())

print(f"Total clusters: {len(cluster_ids)}")
print(f"TF count range: {min(tf_counts)} - {max(tf_counts)}")
print(f"Mean TFs per cluster: {np.mean(tf_counts):.1f}")
print(f"Median TFs per cluster: {np.median(tf_counts):.1f}")

# Plot histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(tf_counts, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Number of Unique TFs per Cluster')
plt.ylabel('Number of Clusters')
plt.title('Distribution of TF Counts per Cluster')
plt.grid(False)

# Add statistics
plt.axvline(np.mean(tf_counts), color='red', linestyle='--', label=f'Mean: {np.mean(tf_counts):.1f}')
plt.axvline(np.median(tf_counts), color='orange', linestyle='--', label=f'Median: {np.median(tf_counts):.1f}')
plt.legend()

# Box plot for additional perspective
plt.subplot(1, 2, 2)
plt.boxplot(tf_counts, vert=True)
plt.ylabel('Number of Unique TFs')
plt.title('TF Count Distribution (Box Plot)')
plt.grid(False)

plt.tight_layout()
# plt.savefig(figpath + "hist_n_TFs_per_peak_clust.pdf")
plt.show()

# Show some statistics
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    val = np.percentile(tf_counts, p)
    print(f"  {p}th percentile: {val:.0f} TFs")

# %% [markdown]
# ### compute the "linked_genes" per peak cluster (NOTE that we're not using the "associated_gene" from Take 1)
#
# - this is to make sure that we're focusing on the "clean" signal
#
# - "linked_gene" is computed using Signac. It's basically genes whose gene expression is highly correlated with peak accessibility. This leaves only 3000 genes for the 640K peaks.
# - "genes_overlap" is computed using the GRCz11 reference genome. For any peak that overlaps with the gene body (from GRCz11 annotation), we call the peak to that gene (the closest one). There are 30K genes overlapping with 640K peaks.

# %%
# Step 3. compute a dict of peak_cluster:linked_genes
cluster_genes_dict = get_associated_genes_per_cluster(adata_peaks, 
                                                      cluster_col="leiden_unified", 
                                                      gene_col="linked_gene")

# %%
# Step 4. construct a binarized TF-gene sub-GRN "mesh"
cluster_tf_gene_matrices = create_tf_gene_matrix_per_cluster(clusters_tfs_dict, cluster_genes_dict)


# %% [markdown]
# ## simple visualization of "meshes"
# - a scatter plot where each dot represents a peak cluster, for the number of TFs (x-axis), and the number of linked_genes (y-axis), and corresponding histogram at each axis.
#
# ### 1) Check the clusters-by-TFs/genes 
#
# - [DONE] there are multiple thresholding options for both candidate TFs and associated genes
# - (1) candidate TFs: enrichment score, percentile, etc.
# - (2) associated genes: linked genes, overlapping with gene body, etc.

# %%
def cluster_dict_to_df(cluster_dict, col_name: str = "list_len") -> pd.DataFrame:
    """
    Convert a dictionary of the form {cluster_id: list_of_items}
    into a DataFrame whose index is the cluster IDs and whose single
    column contains the length of each list.

    Parameters
    ----------
    cluster_dict : dict
        Keys are cluster identifiers (e.g. "35_8"); values are list-like
        collections (e.g. TF names or genes).
    col_name : str, optional
        Name for the output column (default: "list_len").

    Returns
    -------
    pandas.DataFrame
        Index = dictionary keys; one column with the list lengths.
    """
    lengths = {k: len(v) for k, v in cluster_dict.items()}          # map cluster → list length
    return pd.DataFrame.from_dict(lengths, orient="index", columns=[col_name])


# %%
df_clust_TFs = cluster_dict_to_df(clusters_tfs_dict, col_name="n_TFs")
df_clust_TFs.head()

# %%
# compute a dataframe for the size of the meshes
dict_meshes = {}
for clust in cluster_tf_gene_matrices.keys():
    size_mesh = (cluster_tf_gene_matrices[clust].sum().sum())
    dict_meshes[clust] = size_mesh
    
df_meshes = pd.DataFrame.from_dict(dict_meshes, orient="index", columns=["mesh_size"])
df_meshes.head()

# %%
from typing import Mapping, Hashable, Sequence

def cluster_dict_to_df(d: Mapping[Hashable, Sequence], col_name: str) -> pd.DataFrame:
    """Convert one {cluster: list} dict into a 1-column DataFrame."""
    return pd.DataFrame({col_name: [len(v) for v in d.values()]}, index=d.keys())

def build_master_df(
    dict_map: Mapping[str, Mapping[Hashable, Sequence]],
    *,
    prefix: str = "n_",
    fill_value: int = 0
) -> pd.DataFrame:
    """
    Parameters
    ----------
    dict_map : dict
        Keys = short labels (“tfs”, “linked_genes”, …);
        Values = the actual {cluster → list} dictionaries.
    prefix : str
        Prepended to each column name (default “n_” → “n_tfs”, …).
    fill_value : int
        Value used to fill clusters that are missing from some dictionaries.
    """
    dfs = [
        cluster_dict_to_df(d, f"{prefix}{label}")
        for label, d in dict_map.items()
    ]
    master = pd.concat(dfs, axis=1)          # outer join on the index (cluster IDs)
    return master.fillna(fill_value).astype(int)

# ---------- example usage ----------
dict_map = {
    "tfs":          clusters_tfs_dict,
    "linked_genes": cluster_genes_dict,
    # "genes":        cluster_genes_dict,
}

df_master = build_master_df(dict_map, prefix="n_")
print(df_master.head())

# %%
from matplotlib import gridspec
# generate scatter plots and histogram for the numbef of TFs and genes
# Choose which variables to plot
x_col = "n_tfs"
y_col = "n_linked_genes"   # or "n_genes"

# ---------- PREP ----------
x = df_master[x_col].values
y = df_master[y_col].values

# ---------- FIGURE & GRIDSPEC ----------
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    width_ratios=[4, 1],   # main scatter gets 4× width of right hist
    height_ratios=[1, 4],  # top hist gets 1× height of main scatter
    wspace=0.05,           # minimal gap between axes
    hspace=0.05,
)

ax_scatter = fig.add_subplot(gs[1, 0])
ax_histx  = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
ax_histy  = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

# ---------- TURN OFF GRIDS ----------
for ax in (ax_scatter, ax_histx, ax_histy):
    ax.grid(False)

# ---------- SCATTER ----------
ax_scatter.scatter(x, y, s=20, alpha=0.8)
ax_scatter.set_xlabel("number of TFs/cluster")
ax_scatter.set_ylabel("number of linked genes/cluster")

# ---------- MARGINAL HISTOGRAMS ----------
bins = 30
ax_histx.hist(x, bins=bins, density=True)
ax_histy.hist(y, bins=bins, density=True, orientation="horizontal")

# Hide tick labels on the marginal plots to declutter
plt.setp(ax_histx.get_xticklabels(), visible=False)
plt.setp(ax_histy.get_yticklabels(), visible=False)

# Optional: tidy up spines on the marginals
for ax in (ax_histx, ax_histy):
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
        
        
plt.savefig(figpath + "scatter_hist_num_TFs_linked_genes_per_cluster.pdf")
plt.show()

# %% [markdown]
# ## EDA: analysis on the unique/shared TFs/genes across peak clusters
# - last updated: 7/24/2025
#
#

# %%
# from sklearn.manifold import UMAP
from sklearn.metrics import jaccard_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform


# %%
# ADD THIS FUNCTION INTO THE MODULE
def create_cluster_tf_matrix(clusters_tfs_dict):
    """
    Create a binary clusters-by-TFs matrix
    """
    print("Creating clusters-by-TFs matrix...")
    
    # Get all unique TFs across all clusters
    all_tfs = set()
    for tfs in clusters_tfs_dict.values():
        all_tfs.update(tfs)
    
    all_tfs = sorted(list(all_tfs))  # Sort for consistency
    cluster_ids = sorted(list(clusters_tfs_dict.keys()))
    
    print(f"Total unique TFs across all clusters: {len(all_tfs)}")
    print(f"Total clusters: {len(cluster_ids)}")
    
    # Create binary matrix
    cluster_tf_matrix = pd.DataFrame(0, index=cluster_ids, columns=all_tfs)
    
    for cluster_id, tfs in clusters_tfs_dict.items():
        for tf in tfs:
            cluster_tf_matrix.loc[cluster_id, tf] = 1
    
    return cluster_tf_matrix, all_tfs


# %%
# Create the matrix
cluster_tf_matrix, all_unique_tfs = create_cluster_tf_matrix(clusters_tfs_dict)

print(f"Matrix shape: {cluster_tf_matrix.shape}")
print(f"Total 1s in matrix: {cluster_tf_matrix.sum().sum()}")
print(f"Sparsity: {(1 - cluster_tf_matrix.sum().sum() / cluster_tf_matrix.size) * 100:.1f}%")


# %%
def analyze_tf_sharing(cluster_tf_matrix, savefig=False, filename="dist_TFs_across_peak_clusts.pdf"):
    """
    Analyze how TFs are shared across clusters
    """
    print("\n=== TF SHARING ANALYSIS ===")
    
    # TF frequency across clusters
    tf_cluster_counts = cluster_tf_matrix.sum(axis=0)  # How many clusters each TF appears in
    
    # Cluster statistics
    print(f"TF sharing statistics:")
    print(f"  TFs appearing in only 1 cluster: {(tf_cluster_counts == 1).sum()}")
    print(f"  TFs appearing in 2-5 clusters: {((tf_cluster_counts >= 2) & (tf_cluster_counts <= 5)).sum()}")
    print(f"  TFs appearing in 6-10 clusters: {((tf_cluster_counts >= 6) & (tf_cluster_counts <= 10)).sum()}")
    print(f"  TFs appearing in >10 clusters: {(tf_cluster_counts > 10).sum()}")
    
    # Most shared TFs
    most_shared = tf_cluster_counts.nlargest(20)
    print(f"\nTop 20 most shared TFs:")
    for tf, count in most_shared.items():
        print(f"  {tf}: {count} clusters")
    
    # Plot TF sharing distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(tf_cluster_counts, bins=20, alpha=0.7, edgecolor='black') # bins=range(1, tf_cluster_counts.max() + 2)
    plt.xlabel('Number of Clusters Containing TF')
    plt.ylabel('Number of TFs')
    plt.title('Distribution of TF Sharing Across Clusters')
    # plt.yscale('log')  # Log scale because of long tail
    plt.grid(False)
    
    plt.subplot(1, 2, 2)
    # Show cumulative distribution
    sorted_counts = np.sort(tf_cluster_counts)[::-1]  # Descending order
    plt.plot(range(1, len(sorted_counts) + 1), sorted_counts)
    plt.xlabel('TF Rank')
    plt.ylabel('Number of Clusters')
    plt.title('TF Sharing: Rank vs Frequency')
    plt.grid(False)
    
    plt.tight_layout()
    if savefig==True:
        plt.savefig(figpath + filename)
    plt.show()
    
    return tf_cluster_counts, most_shared

tf_sharing_stats, most_shared_tfs = analyze_tf_sharing(cluster_tf_matrix, savefig=True,
                                                       filename="dist_TFs_across_peak_clusts.pdf")

# %%
tf_sharing_stats.sort_values(ascending=False).tail(30)

# %%
tf_sharing_stats.sort_values(ascending=False).head(30)

# %% [markdown]
# ### NOTE:
# - Dlx, Kdm, and Smad TFs are enriched in very few specific peak clusters (specific regulatory programs)
# - Sox TFs are enriched in many (140) peak clusters

# %% [markdown]
# ### 2) REPEAT this for "linked_gene": shared and unique genes across peak clusters
#
# - expectation is that the linked_gene is NOT shared across peak clusters as they are more specific than the enriched TFs

# %%
# Create the matrix
cluster_genes_matrix, all_unique_genes = create_cluster_tf_matrix(cluster_genes_dict)

print(f"Matrix shape: {cluster_genes_matrix.shape}")
print(f"Total 1s in matrix: {cluster_genes_matrix.sum().sum()}")
print(f"Sparsity: {(1 - cluster_genes_matrix.sum().sum() / cluster_genes_matrix.size) * 100:.1f}%")

# %%
gene_sharing_stats, most_shared_genes = analyze_tf_sharing(cluster_genes_matrix, \
                                                           savefig=True, filename="dist_linked_genes_across_peak_clusts.pdf")

# %%
gene_sharing_stats.sort_values(ascending=False).head(20)

# %%
gene_sharing_stats.sort_values(ascending=False).tail(20)

# %% [markdown]
# ### visualize the similarity between peak clusters based on their enriched TFs (or linked_genes)

# %%
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def create_cluster_similarity_heatmap(cluster_feature_matrix,
                                      top_n_clusters=50,
                                      feature_type="TFs",
                                      savefig=False,
                                      filename=None,
                                      linkage_info=None,
                                      return_linkage=True,
                                      # NEW ↓↓↓ ------------------------------------------------
                                      hide_axis_labels=True,        # remove messy labels?
                                      similarity_cutoff=0.80,       # Jaccard ≥ 0.80 defines a block
                                      min_box_size=3,               # ignore tiny blocks
                                      highlight_blocks=True,        # draw rectangles?
                                      return_block_clusters=True):  # hand back the lists?
    """
    Make a heat‑map of cluster‑to‑cluster Jaccard similarity and optionally
      • suppress axis tick labels
      • outline dense similarity “blocks”
      • return the list of clusters in each block
    """
    print(f"\n=== CLUSTER SIMILARITY HEATMAP ({feature_type}) ===")

    # ─────────────────── same down‑sampling of clusters ──────────────────────
    if len(cluster_feature_matrix) > top_n_clusters:
        counts = cluster_feature_matrix.sum(axis=1)
        matrix_subset = cluster_feature_matrix.loc[counts.nlargest(top_n_clusters).index]
        print(f"Using top {top_n_clusters} clusters by {feature_type} count")
    else:
        matrix_subset = cluster_feature_matrix
        print(f"Using all {len(cluster_feature_matrix)} clusters")

    # ─────────────────── pair‑wise Jaccard similarity ────────────────────────
    clusters = matrix_subset.index.tolist()
    n = len(clusters)
    similarity = np.eye(n)

    for i in range(n):
        set_i = set(matrix_subset.columns[matrix_subset.iloc[i] == 1])
        for j in range(i + 1, n):
            set_j = set(matrix_subset.columns[matrix_subset.iloc[j] == 1])
            denom = len(set_i | set_j)
            sim   = 0 if denom == 0 else len(set_i & set_j) / denom
            similarity[i, j] = similarity[j, i] = sim

    # ─────────────────── hierarchical ordering (reuse if given) ──────────────
    if linkage_info is None:
        print("Computing new hierarchical clustering")
        dist_vec      = squareform(1 - similarity, checks=False)
        linkage_matrix = linkage(dist_vec, method='average')
        order          = dendrogram(linkage_matrix, no_plot=True)['leaves']
    else:
        print("Using provided linkage information")
        linkage_matrix = linkage_info['linkage_matrix']
        ref_names      = linkage_info['cluster_names']
        order = [ref_names.index(c) for c in clusters]  # map subset to reference order

    sim_ord   = similarity[np.ix_(order, order)]
    name_ord  = [clusters[i] for i in order]

    # ─────────────────── plot ────────────────────────────────────────────────
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(sim_ord, cmap='Blues', vmin=0, vmax=1,
                     xticklabels=name_ord, yticklabels=name_ord,
                     square=True, cbar_kws={'label': 'Jaccard Similarity'})

    # 1) remove messy tick labels ------------------------------------------------
    if hide_axis_labels:
        ax.set_xticks([]); ax.set_yticks([])        # no ticks
        ax.set_xlabel(''); ax.set_ylabel('')        # no axis labels

    # 2) highlight dense blocks & list their members ----------------------------
    block_lists = []
    if highlight_blocks or return_block_clusters:
        # cut the dendrogram at distance = 1‑similarity_cutoff
        labels = fcluster(linkage_matrix, t=1 - similarity_cutoff, criterion='distance')
        groups = {}
        for idx, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(idx)

        # keep only “big” blocks
        dense_groups = [idxs for idxs in groups.values() if len(idxs) >= min_box_size]

        for g in dense_groups:
            i0, i1 = min(g), max(g)
            size   = i1 - i0 + 1
            block_lists.append([name_ord[i] for i in g])

            if highlight_blocks:
                ax.add_patch(Rectangle((i0, i0), size, size,
                                       edgecolor='red', linewidth=2, fill=False))

    title = f'Cluster Similarity Based on Shared {feature_type}\n' \
            f'({"Hierarchically Clustered" if linkage_info is None else "Consistent Ordering"})'
    plt.title(title, pad=20)
    plt.tight_layout()

    if savefig and filename:
        plt.savefig(filename)
    plt.show()

    # ─────────────────── return values ────────────────────────────────────────
    out = [similarity, clusters]
    if return_linkage:
        out.append({'linkage_matrix': linkage_matrix,
                    'cluster_order' : order,
                    'cluster_names' : clusters})
    if return_block_clusters:
        out.append(block_lists)

    return tuple(out)



# %%
# compute the similarity matrix for “enriched TFs” and plot the heat‑map
sim_matrix, cluster_names, linkage_info, dense_blocks = create_cluster_similarity_heatmap(
    cluster_feature_matrix=cluster_tf_matrix,
    top_n_clusters=402,                             # keep all 402 peak clusters
    savefig=True,
    filename="enriched_TFs_per_peak_cluster_similarity.png",
    feature_type="TFs",

    # NEW knobs ↓↓↓ -----------------------------------------------------------
    hide_axis_labels=True,          # suppress messy tick labels
    similarity_cutoff=0.85,         # clusters with Jaccard ≥ 0.85 form a “block”
    min_box_size=5,                 # only highlight blocks that have ≥ 5 clusters
    highlight_blocks=False,          # draw red rectangles around those blocks
    return_block_clusters=True,     # hand back the cluster lists
    # -------------------------------------------------------------------------

    linkage_info=None,              # compute fresh linkage this time
    return_linkage=True             # still get linkage info for reuse later
)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

def analyze_similarity_distribution(cluster_tf_matrix):
    """
    Analyze the actual similarity distribution in your data to set realistic thresholds
    """
    print("=== ANALYZING SIMILARITY DISTRIBUTION ===")
    
    # Calculate similarity matrix (same as in your function)
    matrix_subset = cluster_tf_matrix
    clusters = matrix_subset.index.tolist()
    n = len(clusters)
    similarity = np.eye(n)
    
    for i in range(n):
        set_i = set(matrix_subset.columns[matrix_subset.iloc[i] == 1])
        for j in range(i + 1, n):
            set_j = set(matrix_subset.columns[matrix_subset.iloc[j] == 1])
            denom = len(set_i | set_j)
            sim = 0 if denom == 0 else len(set_i & set_j) / denom
            similarity[i, j] = similarity[j, i] = sim
    
    # Get upper triangle (excluding diagonal)
    upper_triangle = similarity[np.triu_indices_from(similarity, k=1)]
    
    # Calculate statistics
    stats = {
        'mean': np.mean(upper_triangle),
        'median': np.median(upper_triangle),
        'std': np.std(upper_triangle),
        'p75': np.percentile(upper_triangle, 75),
        'p90': np.percentile(upper_triangle, 90),
        'p95': np.percentile(upper_triangle, 95),
        'p99': np.percentile(upper_triangle, 99),
        'max': np.max(upper_triangle)
    }
    
    print(f"Similarity Statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  75th percentile: {stats['p75']:.3f}")
    print(f"  90th percentile: {stats['p90']:.3f}")
    print(f"  95th percentile: {stats['p95']:.3f}")
    print(f"  99th percentile: {stats['p99']:.3f}")
    print(f"  Maximum: {stats['max']:.3f}")
    
    # Plot distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(upper_triangle, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.3f}')
    ax1.axvline(stats['p90'], color='orange', linestyle='--', label=f'90th %ile: {stats["p90"]:.3f}')
    ax1.axvline(stats['p95'], color='green', linestyle='--', label=f'95th %ile: {stats["p95"]:.3f}')
    ax1.set_xlabel('Jaccard Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Pairwise Similarities')
    ax1.legend()
    
    # Cumulative distribution
    sorted_sims = np.sort(upper_triangle)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax2.plot(sorted_sims, cumulative)
    ax2.axvline(stats['p90'], color='orange', linestyle='--', label=f'90th %ile: {stats["p90"]:.3f}')
    ax2.axvline(stats['p95'], color='green', linestyle='--', label=f'95th %ile: {stats["p95"]:.3f}')
    ax2.set_xlabel('Jaccard Similarity')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Recommend thresholds
    recommended_internal = max(stats['p90'], stats['mean'] + stats['std'])
    recommended_primary = max(stats['p75'], stats['mean'])
    
    print(f"\n=== RECOMMENDED THRESHOLDS ===")
    print(f"Primary cutoff: {recommended_primary:.3f} (for initial detection)")
    print(f"Internal similarity minimum: {recommended_internal:.3f} (for quality filtering)")
    
    return stats, recommended_primary, recommended_internal

# analyze the similarity distribution
analyze_similarity_distribution(cluster_tf_matrix)

# %%
# adjust the gamma value
import matplotlib.colors as mcolors

# Create gamma-corrected colormap
colors = plt.cm.Blues(np.linspace(0, 1, 256))
colors_gamma = colors.copy()
colors_gamma[:, :3] = np.power(colors[:, :3], 0.6)  # Try 0.5, 0.6, or 0.7
gamma_cmap = mcolors.ListedColormap(colors_gamma)
gamma_cmap


# %%
def find_dense_similarity_regions(cluster_feature_matrix,
                                top_n_clusters=50,
                                feature_type="TFs",
                                savefig=False,
                                filename=None,
                                # Dense region parameters
                                min_similarity_threshold=0.15,  # All pairs must exceed this
                                average_similarity_threshold=None,  # NEW: Optional average similarity filter
                                min_block_size=4,
                                max_block_size=30,
                                hide_axis_labels=True, 
                                cmap="Blues",
                                gamma=None,
                                show_blocks=True):
    """
    Find dense similarity regions using a region-growing approach
    
    Parameters:
    - average_similarity_threshold: Optional. If provided, only keep blocks with 
                                   average similarity >= this threshold. If None, no filtering.
    - cmap: Colormap to use ('Blues', 'Reds', etc.) or custom colormap object
    - gamma: Optional gamma correction (e.g., 0.5 for enhanced light regions). If None, no correction.
    """
    
    print(f"\n=== DENSE REGION DETECTION ({feature_type}) ===")
    
    # Preprocessing
    if len(cluster_feature_matrix) > top_n_clusters:
        counts = cluster_feature_matrix.sum(axis=1)
        matrix_subset = cluster_feature_matrix.loc[counts.nlargest(top_n_clusters).index]
        print(f"Using top {top_n_clusters} clusters by {feature_type} count")
    else:
        matrix_subset = cluster_feature_matrix
        print(f"Using all {len(cluster_feature_matrix)} clusters")
    
    # Calculate Jaccard similarity
    clusters = matrix_subset.index.tolist()
    n = len(clusters)
    similarity = np.eye(n)
    
    print("Computing pairwise similarities...")
    for i in range(n):
        set_i = set(matrix_subset.columns[matrix_subset.iloc[i] == 1])
        for j in range(i + 1, n):
            set_j = set(matrix_subset.columns[matrix_subset.iloc[j] == 1])
            denom = len(set_i | set_j)
            sim = 0 if denom == 0 else len(set_i & set_j) / denom
            similarity[i, j] = similarity[j, i] = sim
    
    # Hierarchical ordering for visualization (but not for block detection)
    dist_vec = squareform(1 - similarity, checks=False)
    linkage_matrix = linkage(dist_vec, method='average')
    order = dendrogram(linkage_matrix, no_plot=True)['leaves']
    
    sim_ord = similarity[np.ix_(order, order)]
    name_ord = [clusters[i] for i in order]
    
    print(f"Detection parameters:")
    print(f"  Minimum similarity threshold: {min_similarity_threshold:.3f}")
    if average_similarity_threshold is not None:
        print(f"  Average similarity threshold: {average_similarity_threshold:.3f}")
    else:
        print(f"  Average similarity threshold: None (no filtering)")
    print(f"  Block size range: {min_block_size}-{max_block_size}")
    print(f"  Colormap: {cmap if isinstance(cmap, str) else 'Custom'}")        # ← ADDED
    if gamma is not None:                                                      # ← ADDED
        print(f"  Gamma correction: {gamma}")                                  # ← ADDED
    else:                                                                      # ← ADDED
        print(f"  Gamma correction: None")                                     # ← ADDED
    
    # METHOD 1: Seed-and-grow approach
    print("\nUsing seed-and-grow approach...")
    
    used_clusters = set()
    dense_blocks = []
    
    # Start with highest similarity pairs as seeds
    similarity_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= min_similarity_threshold:
                similarity_pairs.append((similarity[i, j], i, j))
    
    # Sort by similarity (highest first)
    similarity_pairs.sort(reverse=True)
    
    print(f"Found {len(similarity_pairs)} pairs above threshold {min_similarity_threshold:.3f}")
    
    for sim_value, seed_i, seed_j in similarity_pairs:
        # Skip if either cluster is already used
        if seed_i in used_clusters or seed_j in used_clusters:
            continue
        
        # Start growing a block from this seed pair
        current_block = {seed_i, seed_j}
        
        # Keep adding clusters that are similar to ALL clusters in the current block
        improved = True
        while improved and len(current_block) < max_block_size:
            improved = False
            best_candidate = None
            best_min_sim = 0
            
            # Find the best candidate to add
            for candidate in range(n):
                if candidate in current_block or candidate in used_clusters:
                    continue
                
                # Check if this candidate is similar enough to ALL clusters in current block
                min_sim_to_block = min(similarity[candidate, block_member] for block_member in current_block)
                
                if min_sim_to_block >= min_similarity_threshold and min_sim_to_block > best_min_sim:
                    best_candidate = candidate
                    best_min_sim = min_sim_to_block
            
            # Add the best candidate if found
            if best_candidate is not None:
                current_block.add(best_candidate)
                improved = True
        
        # Accept block if it meets size requirements
        if len(current_block) >= min_block_size:
            # Verify all pairs meet threshold (double-check)
            all_pairs_valid = True
            internal_similarities = []
            
            for i in current_block:
                for j in current_block:
                    if i != j:
                        sim_ij = similarity[i, j]
                        internal_similarities.append(sim_ij)
                        if sim_ij < min_similarity_threshold:
                            all_pairs_valid = False
                            break
                if not all_pairs_valid:
                    break
            
            if all_pairs_valid:
                block_clusters = [clusters[i] for i in current_block]
                avg_sim = np.mean(internal_similarities)
                min_sim = np.min(internal_similarities)
                
                dense_blocks.append({
                    'indices': list(current_block),
                    'clusters': block_clusters,
                    'avg_similarity': avg_sim,
                    'min_similarity': min_sim,
                    'size': len(current_block)
                })
                
                # Mark these clusters as used
                used_clusters.update(current_block)
                
                print(f"  ✓ Found block: {len(current_block)} clusters, avg_sim={avg_sim:.3f}, min_sim={min_sim:.3f}")
            else:
                print(f"  ✗ Block validation failed: not all pairs meet threshold")
    
    print(f"\nSeed-and-grow result: {len(dense_blocks)} dense blocks")
    
    # NEW: Apply average similarity filtering if threshold is provided
    if average_similarity_threshold is not None:
        print(f"\nApplying average similarity filter (≥ {average_similarity_threshold:.3f})...")
        
        high_quality_blocks = []
        rejected_blocks = []
        
        for i, block in enumerate(dense_blocks):
            avg_sim = block['avg_similarity']
            
            if avg_sim >= average_similarity_threshold:
                high_quality_blocks.append(block)
                print(f"  ✓ Block {i+1}: KEPT - size={block['size']}, avg_sim={avg_sim:.3f}")
            else:
                rejected_blocks.append(block)
                print(f"  ✗ Block {i+1}: REJECTED - size={block['size']}, avg_sim={avg_sim:.3f} < {average_similarity_threshold:.3f}")
        
        print(f"\nFiltering summary:")
        print(f"  Original blocks: {len(dense_blocks)}")
        print(f"  High-quality blocks: {len(high_quality_blocks)}")
        print(f"  Rejected blocks: {len(rejected_blocks)}")
        
        # Use filtered blocks for visualization
        dense_blocks = high_quality_blocks
    
    # Convert to ordered positions for visualization
    ordered_blocks = []
    for block in dense_blocks:
        # Map original indices to ordered positions
        ordered_indices = []
        for orig_idx in block['indices']:
            cluster_name = clusters[orig_idx]
            if cluster_name in name_ord:
                ordered_pos = name_ord.index(cluster_name)
                ordered_indices.append(ordered_pos)
        
        if ordered_indices:
            ordered_blocks.append({
                **block,
                'ordered_indices': ordered_indices
            })
    
    # PLOTTING
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot heatmap
    # im = ax.imshow(sim_ord, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    # im = ax.imshow(sim_ord, cmap=gamma_cmap, aspect='auto', vmin=0, vmax=1)
    # Create gamma-corrected colormap
    # Apply gamma correction if specified                                     
    if gamma is not None and isinstance(cmap, str):                           
        print(f"Applying gamma correction (γ={gamma}) to {cmap} colormap")   
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, 256))              
        colors_gamma = colors.copy()                                          
        colors_gamma[:, :3] = np.power(colors[:, :3], gamma)                  
        plot_cmap = mcolors.ListedColormap(colors_gamma)                      
    else:                                                                     
        plot_cmap = cmap                                                      
                                                                              
    # Plot heatmap with enhanced colormap                                    
    im = ax.imshow(sim_ord, cmap=plot_cmap, aspect='auto', vmin=0, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Jaccard Similarity', rotation=270, labelpad=20, fontsize=12)
    
    # Draw dense blocks
    colors = ['darkred', 'gold', 'green', 'purple', 'indigo', 'pink', 'cyan', 'lime', 'magenta', 'yellow']
    if show_blocks ==True:
        for i, block in enumerate(ordered_blocks):
            indices = block['ordered_indices']
            if not indices:
                continue

            # Create bounding rectangle for the block
            min_pos = min(indices)
            max_pos = max(indices)
            size = max_pos - min_pos + 1

            color = colors[i % len(colors)]

            # Draw rectangle
            rect = Rectangle((min_pos, min_pos), size, size,
                            linewidth=3, edgecolor=color, facecolor='none', alpha=0.9)
            ax.add_patch(rect)

            # Enhanced labels
            avg_sim = block['avg_similarity']
            min_sim = block['min_similarity']
            block_size = block['size']

            # Add quality indicator to label if filtering was applied
            label_prefix = 'HQ' if average_similarity_threshold is not None else 'B'
            label_text = f'{label_prefix}{i+1}\n({block_size})\navg:{avg_sim:.3f}\nmin:{min_sim:.3f}'
            ax.text(min_pos + size/2, min_pos - 25, label_text, 
                   ha='center', va='bottom', fontweight='bold', 
                   color=color, fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

            print(f"Block {i+1} visualization: positions {min_pos}-{max_pos}, actual size {block_size}")
    
    # Enhanced title
    if average_similarity_threshold is not None:
        title = f'High-Quality Dense Similarity Regions ({feature_type})\n' \
                f'{len(ordered_blocks)} blocks (all pairs ≥ {min_similarity_threshold:.3f}, avg ≥ {average_similarity_threshold:.3f})'
    else:
        title = f'Dense Similarity Regions ({feature_type})\n' \
                f'{len(ordered_blocks)} blocks (all pairs ≥ {min_similarity_threshold:.3f})'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)
    
    if hide_axis_labels:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    
    if savefig and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Return results
    dense_block_clusters = [block['clusters'] for block in dense_blocks]
    linkage_info = {
        'linkage_matrix': linkage_matrix,
        'cluster_order': order,
        'cluster_names': clusters
    }
    
    return sim_ord, name_ord, linkage_info, dense_block_clusters, dense_blocks



# %%

# ====================================================================
# RUN DENSE REGION DETECTION
# ====================================================================

print("="*60)
print("RUNNING DENSE REGION DETECTION")
print("="*60)

# Try with realistic thresholds based on your data
sim_matrix, cluster_names, linkage_info, dense_blocks, block_details = find_dense_similarity_regions(
    cluster_feature_matrix=cluster_tf_matrix,
    top_n_clusters=402,
    feature_type="TFs",
    min_similarity_threshold=0.15,  # All pairs must be at least 0.15
    average_similarity_threshold=0.35,
    min_block_size=4,
    max_block_size=100,
    savefig=True,
    filename="dense_similarity_regions.png",
    hide_axis_labels=True, cmap="Blues", gamma=0.6,
)

print(f"\nFINAL RESULT: {len(dense_blocks)} dense regions detected!")

for i, (block, details) in enumerate(zip(dense_blocks, block_details)):
    print(f"Dense Region {i+1}: {len(block)} clusters")
    print(f"  Average similarity: {details['avg_similarity']:.3f}")
    print(f"  Minimum similarity: {details['min_similarity']:.3f} (guaranteed ≥ 0.15)")
    print(f"  Sample clusters: {block[:3]}{'...' if len(block) > 3 else ''}")

# # Also try DBSCAN approach
# print(f"\n" + "="*40)
# print("TRYING DBSCAN APPROACH")
# print("="*40)

# dbscan_blocks = find_blocks_with_dbscan(
#     cluster_tf_matrix,
#     similarity_threshold=0.15,
#     min_samples=4
# )

# print(f"DBSCAN found {len(dbscan_blocks)} blocks")

# %%
# plot without the boxes
sim_matrix, cluster_names, linkage_info, dense_blocks, block_details = find_dense_similarity_regions(
    cluster_feature_matrix=cluster_tf_matrix,
    top_n_clusters=402,
    feature_type="TFs",
    min_similarity_threshold=0.15,  # All pairs must be at least 0.15
    average_similarity_threshold=0.35,
    min_block_size=4,
    max_block_size=100,
    savefig=True,
    filename=figpath + "enriched_TFs_per_peak_cluster_similarity_with_blocks.png",
    hide_axis_labels=True, cmap="Blues", gamma=0.6,show_blocks=False,
)

# %%
# adjust the gamma value
import matplotlib.colors as mcolors

# Create gamma-corrected colormap
colors = plt.cm.Blues(np.linspace(0, 1, 256))
colors_gamma = colors.copy()
colors_gamma[:, :3] = np.power(colors[:, :3], 0.6)  # Try 0.5, 0.6, or 0.7
gamma_cmap = mcolors.ListedColormap(colors_gamma)

# Save your gamma colorbar as PDF
fig, ax = plt.subplots(figsize=(2, 8))
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=gamma_cmap), 
                   ax=ax, shrink=0.8)
cbar.set_label('Jaccard Similarity', rotation=270, labelpad=20, fontsize=12)
ax.remove()
plt.savefig(figpath + 'gamma_colorbar.pdf', bbox_inches='tight')
plt.show()

# %%
# Extract specific high-quality blocks
# Your function returned: sim_matrix, cluster_names, linkage_info, dense_blocks, block_details

# ====================================================================
# EXTRACT SPECIFIC BLOCKS
# ====================================================================

print("=== EXTRACTING SPECIFIC HIGH-QUALITY BLOCKS ===")

# Extract the blocks you're interested in (HQ1, HQ2, HQ3, HQ5, HQ6)
# Note: Python uses 0-based indexing, so HQ1 = index 0, HQ2 = index 1, etc.

target_blocks = {
    'HQ1': 0,  # First block
    'HQ2': 1,  # Second block  
    'HQ3': 2,  # Third block
    'HQ5': 4,  # Fifth block
    'HQ6': 5   # Sixth block
}

extracted_blocks = {}

for block_name, block_index in target_blocks.items():
    if block_index < len(dense_blocks):
        # Extract cluster list
        cluster_list = dense_blocks[block_index]
        
        # Extract detailed information
        block_info = block_details[block_index]
        
        extracted_blocks[block_name] = {
            'clusters': cluster_list,
            'size': len(cluster_list),
            'avg_similarity': block_info['avg_similarity'],
            'min_similarity': block_info['min_similarity'],
            'details': block_info
        }
        
        print(f"\n{block_name} ({len(cluster_list)} clusters):")
        print(f"  Average similarity: {block_info['avg_similarity']:.3f}")
        print(f"  Minimum similarity: {block_info['min_similarity']:.3f}")
        print(f"  Clusters: {cluster_list[:5]}{'...' if len(cluster_list) > 5 else ''}")
        print(f"  All clusters: {cluster_list}")
    else:
        print(f"\n{block_name}: Block index {block_index} not found (only {len(dense_blocks)} blocks available)")

# ====================================================================
# CREATE ANNOTATION DICTIONARY
# ====================================================================

print(f"\n=== CREATING ANNOTATION DICTIONARY ===")

# Create a clean dictionary for easy access
block_annotations = {}

for block_name, block_data in extracted_blocks.items():
    block_annotations[block_name] = {
        'clusters': block_data['clusters'],
        'description': f"{block_data['size']} clusters, avg_sim={block_data['avg_similarity']:.3f}",
        'quality_score': block_data['avg_similarity']
    }

# Display the annotation dictionary
for block_name, annotation in block_annotations.items():
    print(f"\n{block_name}:")
    print(f"  Description: {annotation['description']}")
    print(f"  Cluster count: {len(annotation['clusters'])}")
    print(f"  First 3 clusters: {annotation['clusters'][:3]}")

# ====================================================================
# EXPORT TO VARIOUS FORMATS
# ====================================================================

print(f"\n=== EXPORTING BLOCK INFORMATION ===")

# 1. Export to CSV
import pandas as pd

# Create a DataFrame with all cluster assignments
cluster_assignments = []
for block_name, annotation in block_annotations.items():
    for cluster in annotation['clusters']:
        cluster_assignments.append({
            'cluster_id': cluster,
            'block_name': block_name,
            'block_size': len(annotation['clusters']),
            'avg_similarity': annotation['quality_score']
        })

df_assignments = pd.DataFrame(cluster_assignments)
df_assignments.to_csv("selected_block_assignments.csv", index=False)
print("Saved cluster assignments to: selected_block_assignments.csv")

# 2. Export to text file with detailed info
with open("selected_blocks_detailed.txt", "w") as f:
    f.write("=== SELECTED HIGH-QUALITY SIMILARITY BLOCKS ===\n\n")
    
    for block_name, annotation in block_annotations.items():
        f.write(f"{block_name}: {annotation['description']}\n")
        f.write(f"Clusters ({len(annotation['clusters'])}): {', '.join(annotation['clusters'])}\n")
        f.write("\n" + "-"*50 + "\n\n")

print("Saved detailed block info to: selected_blocks_detailed.txt")

# 3. Create a summary table
print(f"\n=== SUMMARY TABLE ===")
summary_data = []
for block_name, annotation in block_annotations.items():
    summary_data.append({
        'Block': block_name,
        'Size': len(annotation['clusters']),
        'Avg_Similarity': f"{annotation['quality_score']:.3f}",
        'Sample_Clusters': ', '.join(annotation['clusters'][:3]) + ('...' if len(annotation['clusters']) > 3 else '')
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# ====================================================================
# QUICK ACCESS VARIABLES
# ====================================================================

print(f"\n=== QUICK ACCESS VARIABLES ===")

# Create individual variables for easy access
if 'HQ1' in extracted_blocks:
    HQ1_clusters = extracted_blocks['HQ1']['clusters']
    print(f"HQ1_clusters = {HQ1_clusters}")

if 'HQ2' in extracted_blocks:
    HQ2_clusters = extracted_blocks['HQ2']['clusters']
    print(f"HQ2_clusters = {HQ2_clusters}")

if 'HQ3' in extracted_blocks:
    HQ3_clusters = extracted_blocks['HQ3']['clusters']
    print(f"HQ3_clusters = {HQ3_clusters}")

if 'HQ5' in extracted_blocks:
    HQ5_clusters = extracted_blocks['HQ5']['clusters']
    print(f"HQ5_clusters = {HQ5_clusters}")

if 'HQ6' in extracted_blocks:
    HQ6_clusters = extracted_blocks['HQ6']['clusters']
    print(f"HQ6_clusters = {HQ6_clusters}")

# ====================================================================
# ANNOTATION HELPER FUNCTION
# ====================================================================

def annotate_block_clusters(block_name, cluster_list, additional_info=""):
    """
    Helper function to create annotation text for a block
    """
    annotation = f"{block_name}: {len(cluster_list)} clusters"
    if additional_info:
        annotation += f" ({additional_info})"
    
    # Add sample clusters
    if len(cluster_list) <= 3:
        annotation += f"\nClusters: {', '.join(cluster_list)}"
    else:
        annotation += f"\nSample: {', '.join(cluster_list[:3])}..."
    
    return annotation

# Example usage for annotations
print(f"\n=== EXAMPLE ANNOTATIONS ===")
for block_name, annotation in block_annotations.items():
    if block_name in ['HQ1', 'HQ2', 'HQ3', 'HQ5', 'HQ6']:
        sample_annotation = annotate_block_clusters(
            block_name, 
            annotation['clusters'], 
            f"avg_sim={annotation['quality_score']:.3f}"
        )
        print(f"\n{sample_annotation}")

# ====================================================================
# CREATE MAPPING FOR FURTHER ANALYSIS
# ====================================================================

print(f"\n=== CREATING CLUSTER-TO-BLOCK MAPPING ===")

# Create reverse mapping: cluster -> block
cluster_to_block_mapping = {}
for block_name, annotation in block_annotations.items():
    for cluster in annotation['clusters']:
        cluster_to_block_mapping[cluster] = block_name

print(f"Created mapping for {len(cluster_to_block_mapping)} clusters")
print("Example mappings:")
for i, (cluster, block) in enumerate(list(cluster_to_block_mapping.items())[:5]):
    print(f"  {cluster} → {block}")

# Save the mapping
with open("cluster_to_block_mapping.txt", "w") as f:
    for cluster, block in cluster_to_block_mapping.items():
        f.write(f"{cluster}\t{block}\n")

print("Saved cluster-to-block mapping to: cluster_to_block_mapping.txt")

print(f"\n=== READY FOR ANNOTATION! ===")
print("You now have:")
print("1. extracted_blocks - Dictionary with detailed block information")
print("2. block_annotations - Clean annotation dictionary")
print("3. Individual variables: HQ1_clusters, HQ2_clusters, etc.")
print("4. cluster_to_block_mapping - Reverse lookup")
print("5. CSV and text file exports for further use")

# %%
# HQ1
list_clusters = ['19_5', '0_0', '19_7', '7_3', '7_7', '7_5', '1_1', '29_9', '7_9', '1_11', '1_13', '1_2', '1_3', '7_2', '1_0', '1_12', '9_3', '1_7', '1_9', '1_8', '1_4', '2_6', '11_8', '7_6', '12_1', '30_4', '22_10', '12_13', '12_3', '22_12', '22_14', '12_5', '12_7', '7_8', '13_12', '32_14', '13_15', '24_12', '24_4', '14_4', '24_7', '25_12', '15_13', '15_14', '35_13', '35_12', '15_3', '35_2', '35_4', '35_6', '35_7', '16_2', '16_5', '4_0', '4_1', '27_4', '4_3', '4_4', '18_11', '29_14', '18_2', '5_5', '7_12', '19_1', '19_10', '19_11', '19_12', '7_0', '19_15', '7_10', '19_17', '29_17', '19_3', '19_4']

# Create boolean mask
adata_peaks.obs['HQ1'] = adata_peaks.obs['leiden_unified'].isin(list_clusters)

# Plot with custom colors
sc.pl.umap(adata_peaks, color='HQ1', palette={'True': 'darkred', 'False': 'lightgrey'}, save="_cluster_HQ1.png")

# %%
# HQ2
list_clusters = ['0_1', '10_6', '11_11', '11_9', '12_4', '12_8', '13_11', '13_16', '13_3', '13_4', '13_6', '14_0', '14_6', '15_10', '15_11', '15_12', '15_4', '17_7', '18_1', '18_10', '18_12', '18_5', '18_6', '19_13', '1_6', '21_1', '21_2', '21_3', '21_4', '21_6', '22_15', '22_7', '22_8', '22_9', '24_9', '25_4', '25_9', '26_6', '26_7', '28_0', '28_10', '28_11', '28_12', '28_2', '28_3', '28_5', '28_7', '28_8', '28_9', '29_13', '29_15', '29_4', '29_8', '2_12', '2_17', '2_18', '2_19', '2_2', '30_0', '30_6', '31_0', '32_15', '34_0', '34_4', '35_0', '35_1', '35_15', '35_3', '3_1', '3_10', '3_3', '3_4', '3_5', '3_8', '3_9', '5_1', '7_11']

# Create boolean mask
adata_peaks.obs['HQ2'] = adata_peaks.obs['leiden_unified'].isin(list_clusters)

# Plot with custom colors
sc.pl.umap(adata_peaks, color='HQ2', palette={'True': 'gold', 'False': 'lightgrey'}, save="_cluster_HQ2.png")

# %%
# HQ3
list_clusters = ['7_4', '10_10', '29_7', '10_2', '10_4', '2_0', '8_2', '10_5', '8_4', '8_5', '11_0', '2_15', '11_15', '20_4', '11_5', '11_6', '11_7', '21_0', '21_5', '12_11', '22_0', '22_1', '22_11', '12_2', '31_1', '22_16', '31_6', '22_17', '22_2', '22_5', '22_6', '32_12', '13_13', '32_13', '13_10', '32_6', '33_1', '33_2', '13_9', '14_1', '14_2', '33_4', '33_7', '14_5', '14_8', '34_3', '34_2', '25_10', '26_0', '26_10', '3_0', '16_4', '26_3', '26_5', '26_4', '26_8', '3_6', '3_7', '17_5', '17_8', '5_2', '5_3', '18_4', '28_6', '5_8', '18_9', '6_1', '6_2', '6_3', '29_12', '7_1']

# Create boolean mask
adata_peaks.obs['HQ3'] = adata_peaks.obs['leiden_unified'].isin(list_clusters)

# Plot with custom colors
sc.pl.umap(adata_peaks, color='HQ3', palette={'True': 'green', 'False': 'lightgrey'}, save="_cluster_HQ3.png")

# %%
# HQ5
list_clusters = ['33_6', '17_9', '31_3', '8_0', '8_3', '32_0', '13_0', '2_5', '15_9', '35_8', '24_1', '19_16', '13_8', '33_3', '29_19']

# Create boolean mask
adata_peaks.obs['HQ5'] = adata_peaks.obs['leiden_unified'].isin(list_clusters)

# Plot with custom colors
sc.pl.umap(adata_peaks, color='HQ5', palette={'True': 'indigo', 'False': 'lightgrey'}, save="_cluster_HQ5.png")

# %%
# HQ6
list_clusters = ['24_3', '24_5', '10_0', '4_2', '31_4', '35_10', '9_1', '6_0', '20_1', '32_3']

# Create boolean mask
adata_peaks.obs['HQ6'] = adata_peaks.obs['leiden_unified'].isin(list_clusters)

# Plot with custom colors
sc.pl.umap(adata_peaks, color='HQ6', palette={'True': 'darkorange', 'False': 'lightgrey'}, save="_cluster_HQ6.png")

# %%

# %%

# %%

# %% [markdown]
# ## What are the enriched motifs in these "cluster of clusters"?
#

# %%
from scipy.stats import hypergeom, chi2_contingency
from collections import defaultdict

def analyze_tf_enrichment_in_blocks(cluster_tf_matrix, blocks_data, 
                                   min_frequency=0.3, 
                                   min_enrichment_ratio=1.5,
                                   max_tfs_per_block=15,
                                   statistical_test='hypergeometric'):
    """
    Analyze TF enrichment in similarity blocks
    
    Parameters:
    - cluster_tf_matrix: Binary matrix (clusters × TFs)
    - blocks_data: Dictionary with block_name -> list of cluster IDs
    - min_frequency: Minimum frequency of TF within block to consider
    - min_enrichment_ratio: Minimum enrichment ratio vs background
    - statistical_test: 'hypergeometric', 'chi2', or 'none'
    """
    
    print("=== TF ENRICHMENT ANALYSIS FOR SIMILARITY BLOCKS ===")
    
    # Verify cluster availability
    available_clusters = set(cluster_tf_matrix.index)
    all_analysis_clusters = set()
    for clusters in blocks_data.values():
        all_analysis_clusters.update(clusters)
    
    missing_clusters = all_analysis_clusters - available_clusters
    if missing_clusters:
        print(f"Warning: {len(missing_clusters)} clusters not found in TF matrix")
        print(f"Missing: {list(missing_clusters)[:10]}{'...' if len(missing_clusters) > 10 else ''}")
    
    # Calculate background TF frequencies (all clusters)
    background_tf_freq = cluster_tf_matrix.mean(axis=0)
    total_clusters = len(cluster_tf_matrix)
    
    print(f"Background: {total_clusters} total clusters, {len(background_tf_freq)} TFs")
    
    # Analyze each block
    enrichment_results = {}
    
    for block_name, cluster_list in blocks_data.items():
        print(f"\n--- Analyzing {block_name} ({len(cluster_list)} clusters) ---")
        
        # Get clusters that exist in TF matrix
        valid_clusters = [c for c in cluster_list if c in available_clusters]
        print(f"Valid clusters in TF matrix: {len(valid_clusters)}/{len(cluster_list)}")
        
        if len(valid_clusters) < 2:
            print(f"Skipping {block_name}: insufficient valid clusters")
            continue
        
        # Extract TF matrix for this block
        block_tf_matrix = cluster_tf_matrix.loc[valid_clusters]
        block_size = len(valid_clusters)
        
        # Calculate TF frequencies within block
        block_tf_freq = block_tf_matrix.mean(axis=0)
        block_tf_counts = block_tf_matrix.sum(axis=0)
        
        # Calculate enrichment metrics
        enrichment_data = []
        
        for tf in cluster_tf_matrix.columns:
            # Frequencies
            freq_in_block = block_tf_freq[tf]
            freq_background = background_tf_freq[tf]
            
            # Counts for statistical testing
            count_in_block = block_tf_counts[tf]
            count_in_background = cluster_tf_matrix[tf].sum()
            
            # Skip if too infrequent in block
            if freq_in_block < min_frequency:
                continue
            
            # Calculate enrichment ratio
            enrichment_ratio = freq_in_block / freq_background if freq_background > 0 else float('inf')
            
            # Statistical testing
            p_value = 1.0
            if statistical_test == 'hypergeometric':
                # Hypergeometric test: is this TF overrepresented in the block?
                # Population: all clusters, successes in population: clusters with this TF
                # Sample: block clusters, observed successes: block clusters with this TF
                p_value = hypergeom.sf(
                    count_in_block - 1,  # observed - 1 (for survival function)
                    total_clusters,      # population size
                    count_in_background, # successes in population
                    block_size          # sample size
                )
            elif statistical_test == 'chi2':
                # Chi-square test for independence
                contingency_table = [
                    [count_in_block, block_size - count_in_block],
                    [count_in_background - count_in_block, total_clusters - block_size - (count_in_background - count_in_block)]
                ]
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                except:
                    p_value = 1.0
            
            enrichment_data.append({
                'tf': tf,
                'freq_in_block': freq_in_block,
                'freq_background': freq_background,
                'enrichment_ratio': enrichment_ratio,
                'count_in_block': count_in_block,
                'count_background': count_in_background,
                'p_value': p_value,
                'neg_log_p': -np.log10(p_value) if p_value > 0 else 10
            })
        
        # Sort by enrichment ratio and significance
        enrichment_df = pd.DataFrame(enrichment_data)
        
        if len(enrichment_df) > 0:
            # Filter by enrichment ratio
            significant_tfs = enrichment_df[
                (enrichment_df['enrichment_ratio'] >= min_enrichment_ratio) &
                (enrichment_df['freq_in_block'] >= min_frequency)
            ].copy()
            
            # Sort by combined score (enrichment × significance)
            significant_tfs['combined_score'] = significant_tfs['enrichment_ratio'] * significant_tfs['neg_log_p']
            significant_tfs = significant_tfs.sort_values('combined_score', ascending=False)
            
            # Keep top TFs
            top_tfs = significant_tfs.head(max_tfs_per_block)
            
            enrichment_results[block_name] = {
                'top_tfs': top_tfs,
                'all_significant': significant_tfs,
                'block_size': block_size,
                'valid_clusters': valid_clusters
            }
            
            print(f"Top {len(top_tfs)} enriched TFs:")
            for _, row in top_tfs.iterrows():
                print(f"  {row['tf']}: {row['freq_in_block']:.2%} (vs {row['freq_background']:.2%} bg), "
                      f"ratio={row['enrichment_ratio']:.2f}, p={row['p_value']:.2e}")
        else:
            print(f"No TFs passed enrichment criteria")
            enrichment_results[block_name] = None
    
    return enrichment_results

def visualize_tf_enrichment(enrichment_results, top_n=10):
    """
    Create comprehensive visualizations of TF enrichment
    """
    
    print(f"\n=== CREATING TF ENRICHMENT VISUALIZATIONS ===")
    
    # Collect data for visualization
    plot_data = []
    all_top_tfs = set()
    
    for block_name, result in enrichment_results.items():
        if result is not None:
            top_tfs = result['top_tfs'].head(top_n)
            for _, row in top_tfs.iterrows():
                plot_data.append({
                    'Block': block_name,
                    'TF': row['tf'],
                    'Frequency': row['freq_in_block'],
                    'Background': row['freq_background'],
                    'Enrichment_Ratio': row['enrichment_ratio'],
                    'Neg_Log_P': row['neg_log_p'],
                    'Combined_Score': row['combined_score']
                })
                all_top_tfs.add(row['tf'])
    
    if not plot_data:
        print("No enrichment data available for visualization")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Heatmap of TF frequencies across blocks
    ax1 = plt.subplot(3, 3, (1, 2))
    
    # Create matrix for heatmap
    tf_block_matrix = df_plot.pivot_table(
        index='TF', columns='Block', values='Frequency', fill_value=0
    )
    
    # Sort TFs by maximum enrichment across blocks
    tf_max_enrichment = df_plot.groupby('TF')['Enrichment_Ratio'].max()
    tf_order = tf_max_enrichment.sort_values(ascending=False).index
    tf_block_matrix = tf_block_matrix.reindex(tf_order)
    
    sns.heatmap(tf_block_matrix, annot=True, fmt='.2f', cmap='Reds', 
                ax=ax1, cbar_kws={'label': 'TF Frequency'})
    ax1.set_title('TF Frequency Across Blocks', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Similarity Block')
    ax1.set_ylabel('Transcription Factor')
    
    # 2. Enrichment ratio heatmap
    ax2 = plt.subplot(3, 3, (3, 3))
    
    enrichment_matrix = df_plot.pivot_table(
        index='TF', columns='Block', values='Enrichment_Ratio', fill_value=1
    )
    enrichment_matrix = enrichment_matrix.reindex(tf_order)
    
    sns.heatmap(enrichment_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                center=1, ax=ax2, cbar_kws={'label': 'Enrichment Ratio'})
    ax2.set_title('TF Enrichment Ratios', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Similarity Block')
    ax2.set_ylabel('')
    
    # 3. Top TFs by block (bar plots)
    for i, (block_name, result) in enumerate(enrichment_results.items()):
        if result is None:
            continue
            
        ax = plt.subplot(3, 3, 4 + i)
        
        top_tfs = result['top_tfs'].head(8)  # Top 8 for space
        
        if len(top_tfs) > 0:
            bars = ax.barh(range(len(top_tfs)), top_tfs['enrichment_ratio'], 
                          color='steelblue', alpha=0.7)
            
            # Add frequency annotations
            for j, (_, row) in enumerate(top_tfs.iterrows()):
                ax.text(row['enrichment_ratio'] + 0.1, j, 
                       f"{row['freq_in_block']:.1%}", 
                       va='center', fontsize=8)
            
            ax.set_yticks(range(len(top_tfs)))
            ax.set_yticklabels(top_tfs['tf'], fontsize=9)
            ax.set_xlabel('Enrichment Ratio')
            ax.set_title(f'{block_name} Top TFs', fontweight='bold', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No enriched TFs', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{block_name} Top TFs', fontweight='bold', fontsize=12)
        
        # Limit subplots to available blocks
        if i >= 4:  # Only show first 5 blocks
            break
    
    plt.suptitle('TF Enrichment Analysis Across Similarity Blocks', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig("tf_enrichment_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_block_tf_summary(enrichment_results, blocks_data):
    """
    Create a comprehensive summary of block-defining TFs
    """
    
    print(f"\n=== BLOCK-DEFINING TF SUMMARY ===")
    
    summary = {}
    
    for block_name, result in enrichment_results.items():
        if result is None:
            continue
        
        top_tfs = result['top_tfs']
        block_size = result['block_size']
        
        if len(top_tfs) > 0:
            # Get top 5 TFs for summary
            top_5 = top_tfs.head(5)
            
            tf_summary = []
            for _, row in top_5.iterrows():
                tf_info = f"{row['tf']} ({row['freq_in_block']:.1%}, {row['enrichment_ratio']:.1f}×)"
                tf_summary.append(tf_info)
            
            # Identify highly specific TFs (high enrichment, low background)
            highly_specific = top_tfs[
                (top_tfs['enrichment_ratio'] > 3.0) & 
                (top_tfs['freq_background'] < 0.1)
            ]
            
            summary[block_name] = {
                'size': block_size,
                'top_tfs': tf_summary,
                'n_enriched': len(top_tfs),
                'highly_specific': list(highly_specific['tf'].head(3)),
                'best_tf': top_tfs.iloc[0]['tf'] if len(top_tfs) > 0 else None,
                'best_enrichment': top_tfs.iloc[0]['enrichment_ratio'] if len(top_tfs) > 0 else 0
            }
            
            print(f"\n{block_name} ({block_size} clusters):")
            print(f"  Block-defining TFs: {', '.join(tf_summary[:3])}")
            if len(highly_specific) > 0:
                print(f"  Highly specific TFs: {', '.join(list(highly_specific['tf'].head(3)))}")
            print(f"  Total enriched TFs: {len(top_tfs)}")
        else:
            summary[block_name] = {'size': block_size, 'no_enrichment': True}
            print(f"\n{block_name}: No significantly enriched TFs")
    
    return summary

def find_shared_vs_specific_tfs(enrichment_results):
    """
    Identify TFs that are shared across blocks vs block-specific
    """
    
    print(f"\n=== SHARED vs BLOCK-SPECIFIC TF ANALYSIS ===")
    
    # Collect all enriched TFs by block
    block_tfs = {}
    all_enriched_tfs = set()
    
    for block_name, result in enrichment_results.items():
        if result is not None and len(result['top_tfs']) > 0:
            block_enriched = set(result['top_tfs']['tf'])
            block_tfs[block_name] = block_enriched
            all_enriched_tfs.update(block_enriched)
    
    # Categorize TFs
    tf_block_membership = defaultdict(list)
    for tf in all_enriched_tfs:
        for block_name, tf_set in block_tfs.items():
            if tf in tf_set:
                tf_block_membership[tf].append(block_name)
    
    # Categorize by sharing pattern
    shared_tfs = {}  # TF -> list of blocks
    specific_tfs = {}  # Block -> list of specific TFs
    
    for tf, blocks in tf_block_membership.items():
        if len(blocks) > 1:
            shared_tfs[tf] = blocks
        else:
            block_name = blocks[0]
            if block_name not in specific_tfs:
                specific_tfs[block_name] = []
            specific_tfs[block_name].append(tf)
    
    print(f"Shared TFs (enriched in multiple blocks): {len(shared_tfs)}")
    for tf, blocks in list(shared_tfs.items())[:10]:  # Show top 10
        print(f"  {tf}: {', '.join(blocks)}")
    
    print(f"\nBlock-specific TFs:")
    for block_name, tf_list in specific_tfs.items():
        print(f"  {block_name}: {len(tf_list)} specific TFs")
        print(f"    {', '.join(tf_list[:5])}{'...' if len(tf_list) > 5 else ''}")
    
    return shared_tfs, specific_tfs

def create_enrichment_ranking_table(enrichment_results, output_file="tf_enrichment_ranking.csv"):
    """
    Create a comprehensive ranking table of all enriched TFs
    """
    
    print(f"\n=== CREATING COMPREHENSIVE TF RANKING ===")
    
    all_enrichment_data = []
    
    for block_name, result in enrichment_results.items():
        if result is not None and len(result['top_tfs']) > 0:
            for _, row in result['top_tfs'].iterrows():
                all_enrichment_data.append({
                    'Block': block_name,
                    'TF': row['tf'],
                    'Frequency_in_Block': row['freq_in_block'],
                    'Background_Frequency': row['freq_background'],
                    'Enrichment_Ratio': row['enrichment_ratio'],
                    'Count_in_Block': row['count_in_block'],
                    'P_Value': row['p_value'],
                    'Neg_Log_P': row['neg_log_p'],
                    'Combined_Score': row['combined_score']
                })
    
    df_all_enrichment = pd.DataFrame(all_enrichment_data)
    
    if len(df_all_enrichment) > 0:
        # Sort by combined score
        df_ranked = df_all_enrichment.sort_values('Combined_Score', ascending=False)
        
        # Save to CSV
        df_ranked.to_csv(output_file, index=False)
        print(f"Comprehensive ranking saved to: {output_file}")
        
        # Show top overall TFs
        print(f"\nTop 10 TFs across all blocks (by combined score):")
        for i, (_, row) in enumerate(df_ranked.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['TF']} ({row['Block']}): "
                  f"{row['Frequency_in_Block']:.1%} freq, {row['Enrichment_Ratio']:.1f}× enrichment")
        
        return df_ranked
    else:
        print("No enrichment data to rank")
        return pd.DataFrame()



# %%

# ====================================================================
# RUN COMPLETE TF ENRICHMENT ANALYSIS
# ====================================================================

# Define your blocks data
blocks_data = {
    'HQ1': ['19_5', '0_0', '19_7', '7_3', '7_7', '7_5', '1_1', '29_9', '7_9', '1_11', '1_13', '1_2', '1_3', '7_2', '1_0', '1_12', '9_3', '1_7', '1_9', '1_8', '1_4', '2_6', '11_8', '7_6', '12_1', '30_4', '22_10', '12_13', '12_3', '22_12', '22_14', '12_5', '12_7', '7_8', '13_12', '32_14', '13_15', '24_12', '24_4', '14_4', '24_7', '25_12', '15_13', '15_14', '35_13', '35_12', '15_3', '35_2', '35_4', '35_6', '35_7', '16_2', '16_5', '4_0', '4_1', '27_4', '4_3', '4_4', '18_11', '29_14', '18_2', '5_5', '7_12', '19_1', '19_10', '19_11', '19_12', '7_0', '19_15', '7_10', '19_17', '29_17', '19_3', '19_4'],
    'HQ2': ['0_1', '10_6', '11_11', '11_9', '12_4', '12_8', '13_11', '13_16', '13_3', '13_4', '13_6', '14_0', '14_6', '15_10', '15_11', '15_12', '15_4', '17_7', '18_1', '18_10', '18_12', '18_5', '18_6', '19_13', '1_6', '21_1', '21_2', '21_3', '21_4', '21_6', '22_15', '22_7', '22_8', '22_9', '24_9', '25_4', '25_9', '26_6', '26_7', '28_0', '28_10', '28_11', '28_12', '28_2', '28_3', '28_5', '28_7', '28_8', '28_9', '29_13', '29_15', '29_4', '29_8', '2_12', '2_17', '2_18', '2_19', '2_2', '30_0', '30_6', '31_0', '32_15', '34_0', '34_4', '35_0', '35_1', '35_15', '35_3', '3_1', '3_10', '3_3', '3_4', '3_5', '3_8', '3_9', '5_1', '7_11'],
    'HQ3': ['7_4', '10_10', '29_7', '10_2', '10_4', '2_0', '8_2', '10_5', '8_4', '8_5', '11_0', '2_15', '11_15', '20_4', '11_5', '11_6', '11_7', '21_0', '21_5', '12_11', '22_0', '22_1', '22_11', '12_2', '31_1', '22_16', '31_6', '22_17', '22_2', '22_5', '22_6', '32_12', '13_13', '32_13', '13_10', '32_6', '33_1', '33_2', '13_9', '14_1', '14_2', '33_4', '33_7', '14_5', '14_8', '34_3', '34_2', '25_10', '26_0', '26_10', '3_0', '16_4', '26_3', '26_5', '26_4', '26_8', '3_6', '3_7', '17_5', '17_8', '5_2', '5_3', '18_4', '28_6', '5_8', '18_9', '6_1', '6_2', '6_3', '29_12', '7_1'],
    'HQ5': ['33_6', '17_9', '31_3', '8_0', '8_3', '32_0', '13_0', '2_5', '15_9', '35_8', '24_1', '19_16', '13_8', '33_3', '29_19'],
    'HQ6': ['24_3', '24_5', '10_0', '4_2', '31_4', '35_10', '9_1', '6_0', '20_1', '32_3']
}

print("Step 1: Running TF enrichment analysis...")

# Run the main enrichment analysis
enrichment_results = analyze_tf_enrichment_in_blocks(
    cluster_tf_matrix=cluster_tf_matrix,
    blocks_data=blocks_data,
    min_frequency=0.25,         # TF must be in ≥25% of block clusters
    min_enrichment_ratio=1.5,   # Must be ≥1.5× more frequent than background
    max_tfs_per_block=15,       # Show top 15 TFs per block
    statistical_test='hypergeometric'
)

print("\nStep 2: Creating visualizations...")

# Create visualizations
visualize_tf_enrichment(enrichment_results, top_n=12)

print("\nStep 3: Analyzing shared vs specific TFs...")

# Analyze shared vs specific patterns
shared_tfs, specific_tfs = find_shared_vs_specific_tfs(enrichment_results)

print("\nStep 4: Creating comprehensive ranking...")

# Create ranking table
df_ranked = create_enrichment_ranking_table(enrichment_results)

print("\nStep 5: Creating block summary...")

# Create summary
block_summary = create_block_tf_summary(enrichment_results, blocks_data)

print(f"\n=== ANALYSIS COMPLETE ===")
print("Generated files:")
print("1. tf_enrichment_analysis.png - Comprehensive visualization")
print("2. tf_enrichment_ranking.csv - Detailed TF rankings") 
print("3. Console output with block-specific TF signatures")

print(f"\nKey variables created:")
print("- enrichment_results: Detailed enrichment data per block")
print("- shared_tfs: TFs enriched in multiple blocks")
print("- specific_tfs: TFs unique to each block")
print("- block_summary: Clean summary of block-defining TFs")

# %%

# %%
# plot without the boxes
sim_matrix, cluster_names, linkage_info, dense_blocks, block_details = find_dense_similarity_regions(
    cluster_feature_matrix=cluster_genes_matrix,
    top_n_clusters=402,
    feature_type="genes",
    # min_similarity_threshold=0.15,  # All pairs must be at least 0.15
    # average_similarity_threshold=0.35,
    # min_block_size=4,
    # max_block_size=100,
    savefig=True,
    filename=figpath + "linked_genes_per_peak_cluster_similarity_blocks.png",
    hide_axis_labels=True, cmap="Blues", gamma=0.6,show_blocks=False,
)

# %%
# # compute the similarity matrix for “linked genes” and plot the heat‑map
# sim_matrix_genes, cluster_names_genes, dense_blocks_genes = create_cluster_similarity_heatmap(
#     cluster_feature_matrix=cluster_genes_matrix,
#     top_n_clusters=402,
#     savefig=True,
#     filename="linked_genes_per_peak_cluster_similarity.png",
#     feature_type="genes",

#     # reuse ordering from the TF heat‑map so rows/cols line up
#     linkage_info=linkage_info,
#     return_linkage=False,          # we don’t need the linkage back this time

#     # ── NEW OPTIONS ────────────────────────────────────────────────
#     hide_axis_labels=True,         # drop cluttered tick labels
#     similarity_cutoff=0.85,        # ≥ 0.85 Jaccard ⇒ dense block
#     min_box_size=5,                # only mark blocks with ≥ 5 clusters
#     highlight_blocks=False,         # draw red rectangles
#     return_block_clusters=True     # get the cluster lists in Python
# )

# %%
# Compute correlation between the two similarity matrices
# (only for overlapping clusters)
tf_sim = similarity_matrix
tf_names = cluster_names
gene_sim = similarity_matrix_genes
gene_names = cluster_names_genes

common_clusters = list(set(tf_names) & set(gene_names))
if len(common_clusters) > 1:
    tf_indices = [tf_names.index(c) for c in common_clusters]
    gene_indices = [gene_names.index(c) for c in common_clusters]

    tf_sim_subset = tf_sim[np.ix_(tf_indices, tf_indices)]
    gene_sim_subset = gene_sim[np.ix_(gene_indices, gene_indices)]

    # Flatten upper triangular matrices (excluding diagonal)
    mask = np.triu(np.ones_like(tf_sim_subset, dtype=bool), k=1)
    tf_vals = tf_sim_subset[mask]
    gene_vals = gene_sim_subset[mask]

    correlation = np.corrcoef(tf_vals, gene_vals)[0, 1]
    print(f"\nCorrelation between TF-based and Gene-based similarities: {correlation:.3f}")

results =  {
    'tf_similarity': tf_sim,
    'gene_similarity': gene_sim,
    'tf_names': tf_names,
    'gene_names': gene_names,
    'linkage_info': linkage_info,
    'correlation': correlation if 'correlation' in locals() else None
}

# def plot_similarity_comparison_scatter(results):
"""
Create scatter plot comparing TF-based vs Gene-based similarities
"""
tf_sim = results['tf_similarity']
gene_sim = results['gene_similarity'] 
tf_names = results['tf_names']
gene_names = results['gene_names']

# Get common clusters
common_clusters = list(set(tf_names) & set(gene_names))
if len(common_clusters) < 2:
    print("Not enough common clusters for comparison")

tf_indices = [tf_names.index(c) for c in common_clusters]
gene_indices = [gene_names.index(c) for c in common_clusters]

tf_sim_subset = tf_sim[np.ix_(tf_indices, tf_indices)]
gene_sim_subset = gene_sim[np.ix_(gene_indices, gene_indices)]

# Get upper triangular values (excluding diagonal)
mask = np.triu(np.ones_like(tf_sim_subset, dtype=bool), k=1)
tf_vals = tf_sim_subset[mask]
gene_vals = gene_sim_subset[mask]

plt.figure(figsize=(8, 6))
plt.scatter(tf_vals, gene_vals, alpha=0.6, s=30)
plt.xlabel('TF-based Similarity')
plt.ylabel('Gene-based Similarity')
plt.title('Comparison of Cluster Similarities:\nTF-based vs Gene-based')

# Add diagonal line and correlation
max_val = max(tf_vals.max(), gene_vals.max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect correlation')

correlation = np.corrcoef(tf_vals, gene_vals)[0, 1]
plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### create a mesh - TFs:linked_genes combination (binarized matrix)

# %%
# create a mesh with candidate TFs:linked genes (instead of associated genes, to capture "clean" signal)
cluster_tf_gene_matrices = create_tf_gene_matrix_per_cluster(clusters_tfs_dict, cluster_genes_dict)

# Save the cluster_tf_gene_matrices for future reference
output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl"
with open(output_path, "wb") as f:
    pickle.dump(cluster_tf_gene_matrices, f)
print(f"Saved cluster_tf_gene_matrices with {len(cluster_tf_gene_matrices)} clusters to:")
print(f"  {output_path}")

# %%
# load the cluster_tf_gene_matrices
# output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl"
# with open(output_path, "rb") as f:
#     cluster_tf_gene_matrices = pickle.load(f)
# print(f"Loaded cluster_tf_gene_matrices with {len(cluster_tf_gene_matrices)} clusters from:")
# print(f"  {output_path}")

# %%
# Function to identify clusters with strong specificity
def calculate_cluster_specificity(df_clusters_groups, top_n=2):
    """
    Calculate specificity metrics for each cluster to identify those with 
    strong signal in only 1-2 celltypes/timepoints.
    
    Parameters:
    -----------
    df_clusters_groups : pd.DataFrame
        Clusters (rows) by pseudobulk groups (columns) accessibility matrix
    top_n : int
        Number of top groups to consider (default: 2)
    
    Returns:
    --------
    pd.DataFrame
        Specificity metrics for each cluster
    """
    specificity_metrics = []
    
    for cluster_id in df_clusters_groups.index:
        values = df_clusters_groups.loc[cluster_id].sort_values(ascending=False)
        
        # Calculate metrics
        top_values = values.iloc[:top_n]
        rest_values = values.iloc[top_n:]
        
        total_signal = values.sum()
        top_signal = top_values.sum()
        
        # Specificity score: fraction of signal in top N groups
        specificity_score = top_signal / total_signal if total_signal > 0 else 0
        
        # Signal concentration: ratio of top mean to rest mean
        top_mean = top_values.mean()
        rest_mean = rest_values.mean() if len(rest_values) > 0 else 0
        fold_enrichment = top_mean / rest_mean if rest_mean > 0 else top_mean
        
        # Entropy-based specificity (lower = more specific)
        normalized_values = values / values.sum() if values.sum() > 0 else values
        entropy = -np.sum(normalized_values * np.log2(normalized_values + 1e-10))
        max_entropy = np.log2(len(values))
        normalized_entropy = entropy / max_entropy
        
        specificity_metrics.append({
            'cluster_id': cluster_id,
            'top1_group': values.index[0],
            'top1_value': values.iloc[0],
            'top2_group': values.index[1] if len(values) > 1 else None,
            'top2_value': values.iloc[1] if len(values) > 1 else 0,
            'specificity_score': specificity_score,
            'fold_enrichment': fold_enrichment,
            'normalized_entropy': normalized_entropy,
            'total_signal': total_signal
        })
    
    return pd.DataFrame(specificity_metrics).set_index('cluster_id')

# %%
# Load the clusters-by-pseudobulk accessibility matrix
df_clusters_groups = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_fine_by_pseudobulk.csv",
                                 index_col=0)
print(f"Loaded cluster accessibility matrix: {df_clusters_groups.shape}")
print(f"  {len(df_clusters_groups)} clusters x {len(df_clusters_groups.columns)} pseudobulk groups")

# %%
# Calculate specificity for all clusters
df_specificity = calculate_cluster_specificity(df_clusters_groups, top_n=2)

# Sort by specificity score (higher = more specific to top 2 groups)
df_specificity_sorted = df_specificity.sort_values('specificity_score', ascending=False)

print("Top 20 most specific clusters (high signal in 1-2 celltypes/timepoints):")
print("="*80)
display(df_specificity_sorted.head(20))

# %%
# Visualize specificity distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Specificity score distribution
ax = axes[0, 0]
ax.hist(df_specificity['specificity_score'], bins=50, edgecolor='black')
ax.axvline(0.5, color='red', linestyle='--', label='50% threshold')
ax.set_xlabel('Specificity Score\n(Fraction in top 2 groups)')
ax.set_ylabel('Number of clusters')
ax.set_title('Distribution of Cluster Specificity')
ax.legend()

# 2. Fold enrichment vs specificity
ax = axes[0, 1]
scatter = ax.scatter(df_specificity['specificity_score'], 
                     np.log2(df_specificity['fold_enrichment'] + 1),
                     c=df_specificity['total_signal'],
                     cmap='viridis', alpha=0.6, s=30)
ax.set_xlabel('Specificity Score')
ax.set_ylabel('Log2(Fold Enrichment)')
ax.set_title('Specificity vs Fold Enrichment')
plt.colorbar(scatter, ax=ax, label='Total Signal')

# 3. Entropy distribution
ax = axes[1, 0]
ax.hist(df_specificity['normalized_entropy'], bins=50, edgecolor='black')
ax.axvline(0.3, color='red', linestyle='--', label='Low entropy threshold')
ax.set_xlabel('Normalized Entropy\n(Lower = more specific)')
ax.set_ylabel('Number of clusters')
ax.set_title('Distribution of Signal Entropy')
ax.legend()

# 4. Top cluster annotations
ax = axes[1, 1]
top_clusters = df_specificity_sorted.head(15)
y_pos = np.arange(len(top_clusters))
ax.barh(y_pos, top_clusters['specificity_score'], alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{idx}" for idx in top_clusters.index])
ax.set_xlabel('Specificity Score')
ax.set_ylabel('Cluster ID')
ax.set_title('Top 15 Most Specific Clusters')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# %%
# Identify highly specific clusters (you can adjust thresholds)
specific_clusters = df_specificity[
    (df_specificity['specificity_score'] > 0.5) &  # >50% signal in top 2 groups
    (df_specificity['fold_enrichment'] > 3) &      # At least 3x enrichment
    (df_specificity['normalized_entropy'] < 0.4)   # Low entropy
].sort_values('specificity_score', ascending=False)

print(f"\nFound {len(specific_clusters)} highly specific clusters:")
print("="*80)
for cluster_id in specific_clusters.index[:10]:
    row = df_specificity.loc[cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  Top group: {row['top1_group']} (value: {row['top1_value']:.3f})")
    print(f"  2nd group: {row['top2_group']} (value: {row['top2_value']:.3f})")
    print(f"  Specificity: {row['specificity_score']:.3f} | Fold enrichment: {row['fold_enrichment']:.2f}x")

# %%

# %% [markdown]
# ## Step 3. compute the intersection with a GRN (from CellOracle)
#
# - 1) import the CellOracle GRN objects (Links object)
# - 2) for each peak cluster - find the most relevant GRN (the most accessible celltype & timepoint)
# - 3) compute the "intersection" of GRN and the "mesh" computed above.
# - 4) the resulting sub-GRN will be visualized with a module_subGRN_viz.py module (usin networkX spring layout)

# %%
# import the custom module to export/import the GRNs (dataframes)
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig_GRN_zoom_in/")
from module_grn_export import *


# %%
# Later, load specific datasets:
# Single celltype at one timepoint
base_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/grn_exported/"
nmp_5som = load_grn_by_timepoint_celltype(base_dir, 5, "NMPs", "filtered")
nmp_5som.head()

# %%
# the number of cells for the pseudobulk groups
df_n_cells_pseudobulk = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/num_cells_per_pseudobulk_group.csv", index_col=0)
df_n_cells_pseudobulk.head()

# %%
df_n_cells_pseudobulk.n_cells.sort_values(ascending=False).tail(30)

# %%
df_n_cells_pseudobulk.loc["neural_floor_plate_30somites"]

# %%
# Plot histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_n_cells_pseudobulk.n_cells, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('number of cells per pseudobulk group')
plt.ylabel('occurences')
plt.title('distribution of number of cells per pseudobulk')
plt.grid(False)

# Add statistics
# plt.axvline(np.mean(tf_counts), color='red', linestyle='--', label=f'Mean: {np.mean(tf_counts):.1f}')
# plt.axvline(np.median(tf_counts), color='orange', linestyle='--', label=f'Median: {np.median(tf_counts):.1f}')
# plt.legend()

# Box plot for additional perspective
plt.subplot(1, 2, 2)
plt.boxplot(df_n_cells_pseudobulk.n_cells, vert=True)
plt.ylabel('number of cells per pseudobulk group')
plt.title('box plot')
plt.grid(False)

plt.tight_layout()
plt.savefig(figpath + "hist_num_cells_pseudobulk_groups.pdf")
plt.show()

# Show some statistics
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    val = np.percentile(df_n_cells_pseudobulk.n_cells, p)
    print(f"  {p}th percentile: {val:.0f} cells")


# %%
# for each peak cluster - find the most relevant GRN (the most accessible celltype & timepoint)
# peak clusters-by-pseudobulk groups
df_clusters_groups = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_fine_by_pseudobulk.csv",
                                 index_col=0)
# define the cluster_id
clust_id = "26_8"

# check the pseudobulk profiles for the given cluster
df_clusters_groups.loc[clust_id].sort_values(ascending=False)[0:20]


# %%
df_clusters_groups.loc[:,"NMPs_0somites"].sort_values(ascending=False)

# %%
# check the pseudobulk profiles for the given cluster
df_clusters_groups.loc[clust_id].sort_values(ascending=False)[0:20]


# %%
def extract_subGRN_from_cluster(grn_df, cluster_tf_gene_matrix, cluster_id):
    """
    Extract subGRN based on TF-gene relationships from peak cluster
    
    Parameters:
    - grn_df: GRN dataframe with 'source', 'target', coefficients, etc.
    - cluster_tf_gene_matrix: TF-by-genes binarized matrix (pandas DataFrame)
    - cluster_id: identifier for the cluster
    
    Returns:
    - filtered GRN dataframe containing only edges predicted by the cluster
    """
    
    # Get all TF-target pairs where matrix = 1
    tf_target_pairs = []
    for tf in cluster_tf_gene_matrix.index:
        for gene in cluster_tf_gene_matrix.columns:
            if cluster_tf_gene_matrix.loc[tf, gene] == 1:
                tf_target_pairs.append((tf, gene))
    
    # Convert to set for faster lookup
    predicted_pairs = set(tf_target_pairs)
    
    # Filter GRN to only include predicted pairs
    mask = grn_df.apply(lambda row: (row['source'], row['target']) in predicted_pairs, axis=1)
    subgrn = grn_df[mask].copy()
    
    # Add cluster information
    subgrn['cluster_id'] = cluster_id
    
    return subgrn

# Apply to all clusters
def extract_all_cluster_subGRNs(grn_df, cluster_dict):
    """
    Extract subGRNs for all clusters
    """
    all_subgrns = []
    
    for cluster_id, tf_gene_matrix in cluster_dict.items():
        subgrn = extract_subGRN_from_cluster(grn_df, tf_gene_matrix, cluster_id)
        if len(subgrn) > 0:  # Only keep non-empty subGRNs
            all_subgrns.append(subgrn)
            print(f"Cluster {cluster_id}: {len(subgrn)} edges found")
    
    return all_subgrns


# %%
# test
clust_id = "26_8"
cluster_matrix = cluster_tf_gene_matrices[clust_id]
test_subGRN = extract_subGRN_from_cluster(nmp_5som, cluster_matrix, cluster_id=clust_id)
test_subGRN.head()

# %% [markdown]
# ## visualize the subGRN with NetworkX

# %%
from module_grn_viz import *

# %%
# test with the TFs:linked_genes
clust_id = "26_8"
cluster_matrix = cluster_tf_gene_matrices[clust_id]
test_subGRN = extract_subGRN_from_cluster(nmp_5som, cluster_matrix, cluster_id=clust_id)
test_subGRN.head()

# %%
visualize_subGRN_networkx_styled(test_subGRN, cluster_id = clust_id)

# %%

# %% [markdown]
# ## Systematic analysis of "dynamic" subGRN modules

# %%
from pathlib import Path
# loading the GRNs as a dict
def load_grn_dict_pathlib(base_dir="grn_exports", grn_type="filtered"):
    """
    Load GRN dictionary using pathlib (more robust)
    """
    grn_dict = {}
    base_path = Path(base_dir) / grn_type
    
    # Find all CSV files recursively
    csv_files = list(base_path.glob("*/*.csv"))
    
    for csv_file in csv_files:
        # Extract timepoint from parent directory
        timepoint_dir = csv_file.parent.name
        timepoint = timepoint_dir.split('_')[1] if 'timepoint_' in timepoint_dir else timepoint_dir
        
        # Extract celltype from filename
        celltype = csv_file.stem  # filename without extension
        
        # Load GRN
        # try:
        #     grn_df = pd.read_csv(csv_file)
        #     grn_dict[(celltype, timepoint)] = grn_df
        # except Exception as e:
        grn_df = pd.read_csv(csv_file)
        grn_dict[(celltype, timepoint)] = grn_df
    
    return grn_dict


# %%
# import the individual celltype- and timepoint-specific GRNs to create a master dictionary
grn_dict = load_grn_dict_pathlib(base_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/grn_exported/",
                                 grn_type="filtered")
# grn_dict

# %%
len(grn_dict.keys())

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score
from scipy.cluster.hierarchy import dendrogram, linkage

# STEP 1: Setup - Define your mesh and get the predicted pairs
cluster_id = "26_8"
cluster_matrix = cluster_tf_gene_matrices[cluster_id]

# Extract all predicted TF-target pairs from this mesh
predicted_pairs = []
for tf in cluster_matrix.index:
    for gene in cluster_matrix.columns:
        if cluster_matrix.loc[tf, gene] == 1:
            predicted_pairs.append((tf, gene))

print(f"Mesh {cluster_id} has {len(predicted_pairs)} predicted TF-target pairs")
print(f"TFs in mesh: {list(cluster_matrix.index)}")
print(f"Target genes in mesh: {list(cluster_matrix.columns)}")

# STEP 2: Single timepoint analysis
def analyze_single_timepoint(grn_dict, timepoint, predicted_pairs, cluster_id):
    """
    Analyze how the mesh manifests across celltypes at a single timepoint
    """
    print(f"\n=== ANALYZING TIMEPOINT: {timepoint} ===")
    
    # Get all celltypes at this timepoint
    celltypes_at_tp = [ct for (ct, tp) in grn_dict.keys() if tp == timepoint]
    print(f"Available celltypes: {celltypes_at_tp}")
    
    # Extract subGRNs for each celltype
    celltype_subgrns = {}
    for celltype in celltypes_at_tp:
        if (celltype, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype, timepoint)]
            
            # Find which predicted pairs exist in this GRN
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            celltype_subgrns[celltype] = {
                'subgrn': subgrn,
                'n_edges': len(subgrn),
                'implementation_rate': len(subgrn) / len(predicted_pairs),
                'mean_strength': subgrn['coef_abs'].mean() if len(subgrn) > 0 else 0,
                'implemented_pairs': found_pairs
            }
            
            print(f"{celltype}: {len(subgrn)}/{len(predicted_pairs)} edges ({celltype_subgrns[celltype]['implementation_rate']:.2%})")
    
    return celltype_subgrns

# STEP 3: Compare similarities between celltypes
def compare_celltypes_similarity(celltype_subgrns, predicted_pairs, timepoint):
    """
    Compare how similar celltypes are in implementing the regulatory program
    """
    print(f"\n--- Celltype Similarity Analysis at {timepoint} ---")
    
    celltypes = list(celltype_subgrns.keys())
    n_celltypes = len(celltypes)
    
    # Create binary implementation matrix
    binary_matrix = []
    for celltype in celltypes:
        implemented_pairs = celltype_subgrns[celltype]['implemented_pairs']
        binary_row = [1 if pair in implemented_pairs else 0 for pair in predicted_pairs]
        binary_matrix.append(binary_row)
    
    # Compute pairwise similarities
    similarity_matrix = np.zeros((n_celltypes, n_celltypes))
    for i in range(n_celltypes):
        for j in range(n_celltypes):
            if i == j:
                similarity_matrix[i,j] = 1.0
            else:
                # Jaccard similarity
                similarity_matrix[i,j] = jaccard_score(binary_matrix[i], binary_matrix[j])
    
    # Plot similarity heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, 
                xticklabels=celltypes, 
                yticklabels=celltypes,
                annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Celltype Similarity - Cluster {cluster_id} at {timepoint}')
    plt.tight_layout()
    plt.show()
    
    # Find most and least similar pairs
    similarities = []
    for i in range(n_celltypes):
        for j in range(i+1, n_celltypes):
            similarities.append({
                'celltype1': celltypes[i],
                'celltype2': celltypes[j], 
                'similarity': similarity_matrix[i,j]
            })
    
    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    print("Most similar celltype pairs:")
    for sim in similarities[:3]:
        print(f"  {sim['celltype1']} vs {sim['celltype2']}: {sim['similarity']:.3f}")
    
    print("Least similar celltype pairs:")
    for sim in similarities[-3:]:
        print(f"  {sim['celltype1']} vs {sim['celltype2']}: {sim['similarity']:.3f}")
    
    return similarity_matrix, similarities

# STEP 4: Run analysis for one timepoint
timepoint_to_analyze = "05"  # Adjust based on your timepoints
celltype_results = analyze_single_timepoint(grn_dict, timepoint_to_analyze, predicted_pairs, cluster_id)
similarity_matrix, similarities = compare_celltypes_similarity(celltype_results, predicted_pairs, timepoint_to_analyze)

# STEP 5: Multi-timepoint comparison
def compare_across_timepoints(grn_dict, predicted_pairs, cluster_id):
    """
    Compare how the regulatory program changes across timepoints
    """
    print(f"\n=== MULTI-TIMEPOINT ANALYSIS ===")
    
    # Get all available timepoints
    all_timepoints = sorted(set([tp for (ct, tp) in grn_dict.keys()]))
    print(f"Available timepoints: {all_timepoints}")
    
    # Store results for each timepoint
    timepoint_results = {}
    
    for timepoint in all_timepoints:
        print(f"\nProcessing timepoint {timepoint}...")
        celltype_subgrns = analyze_single_timepoint(grn_dict, timepoint, predicted_pairs, cluster_id)
        timepoint_results[timepoint] = celltype_subgrns
    
    return timepoint_results

# STEP 6: Temporal tracking of specific celltypes
def track_celltype_across_time(timepoint_results, cluster_id):
    """
    Track how specific celltypes implement the program over time
    """
    print(f"\n--- Temporal Tracking ---")
    
    # Get celltypes that appear in multiple timepoints
    all_celltypes = set()
    for tp_results in timepoint_results.values():
        all_celltypes.update(tp_results.keys())
    
    # Track each celltype across time
    temporal_tracking = {}
    for celltype in all_celltypes:
        temporal_tracking[celltype] = []
        for timepoint in sorted(timepoint_results.keys()):
            if celltype in timepoint_results[timepoint]:
                result = timepoint_results[timepoint][celltype]
                temporal_tracking[celltype].append({
                    'timepoint': timepoint,
                    'implementation_rate': result['implementation_rate'],
                    'mean_strength': result['mean_strength'],
                    'n_edges': result['n_edges']
                })
    
    # Plot temporal evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Implementation rate over time
    for celltype, tracking in temporal_tracking.items():
        if len(tracking) > 1:  # Only plot celltypes with multiple timepoints
            timepoints = [t['timepoint'] for t in tracking]
            impl_rates = [t['implementation_rate'] for t in tracking]
            ax1.plot(timepoints, impl_rates, marker='o', label=celltype)
    
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('Implementation Rate')
    ax1.set_title(f'Implementation Rate Over Time - Cluster {cluster_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mean strength over time
    for celltype, tracking in temporal_tracking.items():
        if len(tracking) > 1:
            timepoints = [t['timepoint'] for t in tracking]
            strengths = [t['mean_strength'] for t in tracking]
            ax2.plot(timepoints, strengths, marker='s', label=celltype)
    
    ax2.set_xlabel('Timepoint') 
    ax2.set_ylabel('Mean Edge Strength')
    ax2.set_title(f'Mean Strength Over Time - Cluster {cluster_id}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return temporal_tracking

# STEP 7: Summary statistics
def summarize_analysis(timepoint_results, temporal_tracking, cluster_id):
    """
    Provide summary statistics of the analysis
    """
    print(f"\n=== SUMMARY FOR CLUSTER {cluster_id} ===")
    
    # Overall implementation statistics
    all_impl_rates = []
    all_strengths = []
    for tp_results in timepoint_results.values():
        for ct_result in tp_results.values():
            all_impl_rates.append(ct_result['implementation_rate'])
            all_strengths.append(ct_result['mean_strength'])
    
    print(f"Implementation rate: {np.mean(all_impl_rates):.2%} ± {np.std(all_impl_rates):.2%}")
    print(f"Mean edge strength: {np.mean(all_strengths):.4f} ± {np.std(all_strengths):.4f}")
    
    # Best implementers
    best_implementers = []
    for timepoint, tp_results in timepoint_results.items():
        for celltype, result in tp_results.items():
            best_implementers.append({
                'celltype': celltype,
                'timepoint': timepoint,
                'implementation_rate': result['implementation_rate'],
                'mean_strength': result['mean_strength']
            })
    
    best_implementers = sorted(best_implementers, key=lambda x: x['implementation_rate'], reverse=True)
    
    print("\nTop 5 implementers:")
    for impl in best_implementers[:5]:
        print(f"  {impl['celltype']} at {impl['timepoint']}: {impl['implementation_rate']:.2%} (strength: {impl['mean_strength']:.4f})")
    
    # Temporal trends
    print(f"\nCelltypes tracked across time: {len([ct for ct, track in temporal_tracking.items() if len(track) > 1])}")

# Run the complete analysis
all_timepoint_results = compare_across_timepoints(grn_dict, predicted_pairs, cluster_id)
temporal_tracking = track_celltype_across_time(all_timepoint_results, cluster_id)
summarize_analysis(all_timepoint_results, temporal_tracking, cluster_id)

# %%
grn_dict.keys()

# %%
grn_dict[('neural_floor_plate', '00')]

# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_subgrns_over_time(grn_dict, predicted_pairs, cluster_id="26_8", celltype_of_interest="NMPs"):
    """
    Plot NetworkX diagrams for a celltype-specific subGRNs across all timepoints
    using a master GRN layout for consistent node positioning
    """
    # Get all timepoints where the celltype exists
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)
    
    timepoints = sorted(timepoints)
    print(f"Found {celltype_of_interest} at timepoints: {timepoints}")
    
    # Extract subGRNs for celltype at each timepoint
    subgrns = {}
    all_nodes = set()
    all_edges = set()
    
    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]
            
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            subgrns[timepoint] = subgrn
            print(f"Timepoint {timepoint}: {len(subgrn)} edges")
            
            # Collect all nodes and edges for master GRN
            if len(subgrn) > 0:
                all_nodes.update(subgrn['source'])
                all_nodes.update(subgrn['target'])
                all_edges.update(zip(subgrn['source'], subgrn['target']))
    
    print(f"Master GRN: {len(all_nodes)} total nodes, {len(all_edges)} total edges")
    
    # Create master GRN and compute layout
    master_G = nx.DiGraph()
    master_G.add_edges_from(all_edges)
    
    # Compute master layout based on network size
    n_master_nodes = len(master_G.nodes())
    n_master_edges = len(master_G.edges())
    
    print(f"Computing master layout for {n_master_nodes} nodes, {n_master_edges} edges...")
    
    # Choose layout algorithm based on master network properties
    if n_master_nodes < 30:
        master_pos = nx.circular_layout(master_G, scale=1.2)
    elif n_master_nodes < 80:
        master_pos = nx.spring_layout(master_G, k=0.8, iterations=200, seed=42, scale=1.3)
    else:
        try:
            master_pos = nx.kamada_kawai_layout(master_G, scale=1.3)
        except:
            master_pos = nx.spring_layout(master_G, k=1.0, iterations=250, seed=42, scale=1.3)
    
    # Get node classifications for consistent coloring
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets
    
    # Create subplot layout
    n_timepoints = len(subgrns)
    if n_timepoints <= 3:
        nrows, ncols = 1, n_timepoints
        figsize = (7*n_timepoints, 7)
    else:
        nrows, ncols = 2, 3
        figsize = (12, 8)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_timepoints == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each timepoint using master layout
    for i, (timepoint, subgrn) in enumerate(subgrns.items()):
        ax = axes[i]
        
        if len(subgrn) > 0:
            # Create timepoint-specific graph
            G = nx.DiGraph()
            edge_weights = {}
            
            # Add edges with weights
            for _, row in subgrn.iterrows():
                G.add_edge(row['source'], row['target'])
                edge_weights[(row['source'], row['target'])] = round(row['coef_mean'], 4)
            
            # Use master positions, but only for nodes present in this timepoint
            pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}
            
            # Draw all master nodes (both present and absent) for consistency
            present_nodes = set(G.nodes())
            absent_nodes = all_nodes - present_nodes
            
            # Classify nodes for this timepoint
            current_tf = present_nodes & tf_nodes
            current_targets = present_nodes & target_genes  
            current_tf_targets = present_nodes & tf_target_nodes
            
            # Draw present nodes with full opacity
            if current_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf), 
                                      node_color='lightcoral', node_size=600, 
                                      ax=ax, alpha=0.9)
            if current_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_targets), 
                                      node_color='lightblue', node_size=400,
                                      ax=ax, alpha=0.9)
            if current_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf_targets), 
                                      node_color='orange', node_size=500,
                                      ax=ax, alpha=0.9)
            
            # Draw absent nodes with low opacity (ghosted)
            absent_tf = absent_nodes & tf_nodes
            absent_targets = absent_nodes & target_genes
            absent_tf_targets = absent_nodes & tf_target_nodes
            
            if absent_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf), 
                                      node_color='lightcoral', node_size=300, 
                                      ax=ax, alpha=0.15)
            if absent_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_targets), 
                                      node_color='lightblue', node_size=200,
                                      ax=ax, alpha=0.15)
            if absent_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf_targets), 
                                      node_color='orange', node_size=250,
                                      ax=ax, alpha=0.15)
            
            # Draw present edges with full opacity
            if len(G.edges()) > 0:
                edge_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, width=edge_widths, 
                                      edge_color='darkblue', alpha=0.7,
                                      arrowsize=20, arrowstyle='->', ax=ax)
            
            # Draw absent edges with very low opacity
            absent_edges = all_edges - set(G.edges())
            if absent_edges:
                absent_G = nx.DiGraph()
                absent_G.add_edges_from(absent_edges)
                # Only draw absent edges if both nodes exist in master_pos
                valid_absent_edges = [(u, v) for u, v in absent_edges 
                                    if u in master_pos and v in master_pos]
                if valid_absent_edges:
                    absent_G_filtered = nx.DiGraph()
                    absent_G_filtered.add_edges_from(valid_absent_edges)
                    nx.draw_networkx_edges(absent_G_filtered, master_pos, 
                                          width=0.5, edge_color='gray', alpha=0.1,
                                          arrowsize=10, arrowstyle='->', ax=ax, style='dashed')
            
            # Draw labels only for present nodes
            present_pos = {node: master_pos[node] for node in present_nodes if node in master_pos}
            nx.draw_networkx_labels(G, present_pos, font_size=8, font_weight='bold', ax=ax)
            
            # Set consistent axis limits based on master layout
            if master_pos:
                x_coords = [coord[0] for coord in master_pos.values()]
                y_coords = [coord[1] for coord in master_pos.values()]
                margin = 0.15
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}\n({len(G.edges())} edges, {len(G.nodes())} nodes)', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No edges found\nfor timepoint {timepoint}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title and legend
    fig.suptitle(f'{celltype_of_interest.replace("_", " ").title()} Regulatory Network Evolution - Cluster {cluster_id}\n(Master GRN: {len(all_nodes)} nodes, {len(all_edges)} edges)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Enhanced legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Transcription Factors (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Target Genes (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=11, label='TF & Target (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Inactive Nodes', alpha=0.3),
        plt.Line2D([0], [0], color='darkblue', linewidth=3, label='Active Edges', alpha=0.7),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Inactive Edges', alpha=0.3)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    return subgrns, master_G, master_pos

# Alternative function for comparing specific timepoints
def compare_timepoints(grn_dict, predicted_pairs, timepoint1, timepoint2, 
                      cluster_id="26_8", celltype_of_interest="NMPs"):
    """
    Compare two specific timepoints side by side with master layout
    """
    subgrns, master_G, master_pos = plot_subgrns_over_time(
        grn_dict, predicted_pairs, cluster_id, celltype_of_interest)
    
    # Create focused comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot logic for each timepoint would go here...
    # (Similar to above but focused on just two timepoints)
    
    plt.tight_layout()
    plt.show()

# Run the analysis
cluster_id = "26_8"
nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
    grn_dict, predicted_pairs, cluster_id, celltype_of_interest="neural_floor_plate")

# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_subgrns_over_time(grn_dict, predicted_pairs, cluster_id="26_8", celltype_of_interest="NMPs", 
                          spring_k=1.2, layout_scale=1.5, max_labels=25, label_strategy="top_connected"):
    """
    Plot NetworkX diagrams for a celltype-specific subGRNs across all timepoints
    using a master GRN layout for consistent node positioning
    
    Edge colors: Red = Activation, Blue = Repression (based on coefficient sign)
    
    Parameters:
    - spring_k: Spring constant for layout (higher = more spread out)
    - layout_scale: Overall scale of the layout (higher = bigger)
    - max_labels: Maximum number of labels to show
    - label_strategy: "top_connected", "tf_plus_top_targets", "all_tfs_plus_dynamic", 
                     "all_tfs", "dynamic_only", "degree_threshold", or "all"
    """
    # Get all timepoints where the celltype exists
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)
    
    timepoints = sorted(timepoints)
    print(f"Found {celltype_of_interest} at timepoints: {timepoints}")
    
    # Extract subGRNs for celltype at each timepoint
    subgrns = {}
    all_nodes = set()
    all_edges = set()
    
    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]
            
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            subgrns[timepoint] = subgrn
            print(f"Timepoint {timepoint}: {len(subgrn)} edges")
            
            # Collect all nodes and edges for master GRN
            if len(subgrn) > 0:
                all_nodes.update(subgrn['source'])
                all_nodes.update(subgrn['target'])
                all_edges.update(zip(subgrn['source'], subgrn['target']))
    
    print(f"Master GRN: {len(all_nodes)} total nodes, {len(all_edges)} total edges")
    
    # Create master GRN and compute layout
    master_G = nx.DiGraph()
    master_G.add_edges_from(all_edges)
    
    # Compute master layout based on network size
    n_master_nodes = len(master_G.nodes())
    n_master_edges = len(master_G.edges())
    
    print(f"Computing master layout for {n_master_nodes} nodes, {n_master_edges} edges...")
    
    # Choose layout algorithm based on master network properties - further increased spacing
    if n_master_nodes < 30:
        master_pos = nx.circular_layout(master_G, scale=layout_scale*1.1)
    elif n_master_nodes < 80:
        master_pos = nx.spring_layout(master_G, k=spring_k*1.2, iterations=300, seed=42, scale=layout_scale*1.1)
    else:
        try:
            master_pos = nx.kamada_kawai_layout(master_G, scale=layout_scale*1.1)
        except:
            master_pos = nx.spring_layout(master_G, k=spring_k*1.3, iterations=350, seed=42, scale=layout_scale*1.1)
    
    # Calculate dynamic nodes - nodes whose edges change the most over time
    print("Calculating node dynamics across timepoints...")
    node_edge_changes = {}  # node -> total edge changes across time
    
    for node in all_nodes:
        total_changes = 0
        prev_edges = set()
        
        for timepoint in timepoints:
            if timepoint in subgrns and len(subgrns[timepoint]) > 0:
                subgrn = subgrns[timepoint]
                # Get current edges for this node (both incoming and outgoing)
                current_edges = set()
                node_edges = subgrn[(subgrn['source'] == node) | (subgrn['target'] == node)]
                for _, row in node_edges.iterrows():
                    current_edges.add((row['source'], row['target']))
                
                # Calculate changes from previous timepoint
                if prev_edges is not None:
                    gained_edges = current_edges - prev_edges
                    lost_edges = prev_edges - current_edges
                    total_changes += len(gained_edges) + len(lost_edges)
                
                prev_edges = current_edges
        
        node_edge_changes[node] = total_changes
    
    # Get most dynamic nodes
    dynamic_nodes = sorted(node_edge_changes.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 10 most dynamic nodes: {[(node, changes) for node, changes in dynamic_nodes[:10]]}")
    
    # Get node classifications for consistent coloring
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets
    
    # Create subplot layout
    n_timepoints = len(subgrns)
    if n_timepoints <= 3:
        nrows, ncols = 1, n_timepoints
        figsize = (7*n_timepoints, 7)
    else:
        nrows, ncols = 2, 3
        figsize = (21, 14)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_timepoints == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each timepoint using master layout
    for i, (timepoint, subgrn) in enumerate(subgrns.items()):
        ax = axes[i]
        
        if len(subgrn) > 0:
            # Create timepoint-specific graph
            G = nx.DiGraph()
            edge_weights = {}
            edge_signs = {}  # Track positive/negative interactions
            
            # Add edges with weights and signs
            for _, row in subgrn.iterrows():
                G.add_edge(row['source'], row['target'])
                edge_weights[(row['source'], row['target'])] = round(row['coef_abs'], 4)
                
                # Determine edge sign - check for original coefficient column
                if 'coef' in row:
                    edge_signs[(row['source'], row['target'])] = 1 if row['coef'] > 0 else -1
                    sign_column = 'coef'
                elif 'coefficient' in row:
                    edge_signs[(row['source'], row['target'])] = 1 if row['coefficient'] > 0 else -1
                    sign_column = 'coefficient'
                elif 'weight' in row:
                    edge_signs[(row['source'], row['target'])] = 1 if row['weight'] > 0 else -1
                    sign_column = 'weight'
                else:
                    # If no sign info available, assume positive (activation)
                    edge_signs[(row['source'], row['target'])] = 1
                    sign_column = 'assumed_positive'
            
            # Print edge type information
            if len(subgrn) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                print(f"Timepoint {timepoint}: {pos_count} activation, {neg_count} repression edges (using '{sign_column}' column)")
            
            # Use master positions, but only for nodes present in this timepoint
            pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}
            
            # Draw all master nodes (both present and absent) for consistency
            present_nodes = set(G.nodes())
            absent_nodes = all_nodes - present_nodes
            
            # Classify nodes for this timepoint
            current_tf = present_nodes & tf_nodes
            current_targets = present_nodes & target_genes  
            current_tf_targets = present_nodes & tf_target_nodes
            
            # Draw present nodes with full opacity
            if current_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf), 
                                      node_color='lightcoral', node_size=600, 
                                      ax=ax, alpha=0.9)
            if current_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_targets), 
                                      node_color='lightblue', node_size=400,
                                      ax=ax, alpha=0.9)
            if current_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf_targets), 
                                      node_color='orange', node_size=500,
                                      ax=ax, alpha=0.9)
            
            # Draw absent nodes with low opacity (ghosted)
            absent_tf = absent_nodes & tf_nodes
            absent_targets = absent_nodes & target_genes
            absent_tf_targets = absent_nodes & tf_target_nodes
            
            if absent_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf), 
                                      node_color='lightcoral', node_size=300, 
                                      ax=ax, alpha=0.15)
            if absent_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_targets), 
                                      node_color='lightblue', node_size=200,
                                      ax=ax, alpha=0.15)
            if absent_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf_targets), 
                                      node_color='orange', node_size=250,
                                      ax=ax, alpha=0.15)
            
            # Draw present edges with different colors for activation/repression
            if len(G.edges()) > 0:
                # Separate positive and negative edges
                positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
                negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]
                
                # Draw positive edges (activation) in red
                if positive_edges:
                    pos_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in positive_edges]
                    pos_G = nx.DiGraph()
                    pos_G.add_edges_from(positive_edges)
                    nx.draw_networkx_edges(pos_G, pos, width=pos_widths, 
                                          edge_color='red', alpha=0.7,
                                          arrowsize=20, arrowstyle='->', ax=ax)
                
                # Draw negative edges (repression) in blue
                if negative_edges:
                    neg_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in negative_edges]
                    neg_G = nx.DiGraph()
                    neg_G.add_edges_from(negative_edges)
                    nx.draw_networkx_edges(neg_G, pos, width=neg_widths, 
                                          edge_color='blue', alpha=0.7,
                                          arrowsize=20, arrowstyle='->', ax=ax)
            
            # Draw absent edges with very low opacity
            absent_edges = all_edges - set(G.edges())
            if absent_edges:
                absent_G = nx.DiGraph()
                absent_G.add_edges_from(absent_edges)
                # Only draw absent edges if both nodes exist in master_pos
                valid_absent_edges = [(u, v) for u, v in absent_edges 
                                    if u in master_pos and v in master_pos]
                if valid_absent_edges:
                    absent_G_filtered = nx.DiGraph()
                    absent_G_filtered.add_edges_from(valid_absent_edges)
                    nx.draw_networkx_edges(absent_G_filtered, master_pos, 
                                          width=0.5, edge_color='gray', alpha=0.1,
                                          arrowsize=10, arrowstyle='->', ax=ax, style='dashed')
            
            # Selective labeling - configurable strategies
            node_degrees = dict(G.degree())
            
            if label_strategy == "top_connected":
                # Show labels for top N most connected nodes
                if len(node_degrees) > max_labels:
                    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
                    nodes_to_label = [node for node, degree in sorted_nodes[:max_labels]]
                else:
                    nodes_to_label = list(present_nodes)
                    
            elif label_strategy == "tf_plus_top_targets":
                # Always label TFs and TF-targets, plus top target genes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                target_nodes_present = present_nodes & target_genes
                
                # Always label TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add top connected target genes
                target_degrees = {node: degree for node, degree in node_degrees.items() 
                                if node in target_nodes_present}
                if target_degrees:
                    n_targets_to_add = max(5, max_labels - len(nodes_to_label))
                    top_targets = sorted(target_degrees.items(), key=lambda x: x[1], reverse=True)[:n_targets_to_add]
                    nodes_to_label.extend([node for node, degree in top_targets])
                    
            elif label_strategy == "all_tfs_plus_dynamic":
                # Label ALL TFs and TF-targets, plus most dynamic nodes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                
                # Always label ALL TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add most dynamic target genes that aren't already TFs
                target_nodes_present = present_nodes & target_genes
                remaining_slots = max(5, max_labels - len(nodes_to_label))
                
                # Get dynamic target genes (excluding those already labeled as TFs)
                dynamic_targets = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in target_nodes_present and node not in nodes_to_label]
                
                # Add top dynamic target genes
                for node, changes in dynamic_targets[:remaining_slots]:
                    nodes_to_label.append(node)
                    
            elif label_strategy == "all_tfs":
                # Label ALL transcription factors and TF-targets only
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
            elif label_strategy == "dynamic_only":
                # Label only the most dynamic nodes
                dynamic_present = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in present_nodes]
                nodes_to_label = [node for node, changes in dynamic_present[:max_labels]]
                    
            elif label_strategy == "degree_threshold":
                # Label nodes with degree above threshold
                threshold = max(2, np.percentile(list(node_degrees.values()), 70))  # Top 30%
                nodes_to_label = [node for node, degree in node_degrees.items() if degree >= threshold]
                
            else:  # "all"
                nodes_to_label = list(present_nodes)
            
            # Draw labels only for selected nodes that exist in master_pos
            nodes_to_label_filtered = [node for node in nodes_to_label if node in master_pos]
            label_pos = {node: master_pos[node] for node in nodes_to_label_filtered}
            
            # Create labels dict for only the nodes we want to show
            labels_to_show = {node: node for node in nodes_to_label_filtered}
            nx.draw_networkx_labels(G, label_pos, labels=labels_to_show, font_size=8, font_weight='bold', ax=ax)
            
            print(f"Timepoint {timepoint}: Showing labels for {len(nodes_to_label_filtered)} out of {len(present_nodes)} nodes")
            
            # Count edge types for title
            if len(G.edges()) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                edge_info = f"({pos_count} activation, {neg_count} repression)"
            else:
                edge_info = ""
            
            # Set consistent axis limits based on master layout
            if master_pos:
                x_coords = [coord[0] for coord in master_pos.values()]
                y_coords = [coord[1] for coord in master_pos.values()]
                margin = 0.15
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}\n({len(G.edges())} edges, {len(G.nodes())} nodes) {edge_info}', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No edges found\nfor timepoint {timepoint}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title and legend
    fig.suptitle(f'{celltype_of_interest.replace("_", " ").title()} Regulatory Network Evolution - Cluster {cluster_id}\n(Master GRN: {len(all_nodes)} nodes, {len(all_edges)} edges)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Enhanced legend with edge types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Transcription Factors (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Target Genes (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=11, label='TF & Target (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Inactive Nodes', alpha=0.3),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Activation', alpha=0.7),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Repression', alpha=0.7),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Inactive Edges', alpha=0.3)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    return subgrns, master_G, master_pos

# Alternative function for comparing specific timepoints
def compare_timepoints(grn_dict, predicted_pairs, timepoint1, timepoint2, 
                      cluster_id="26_8", celltype_of_interest="NMPs"):
    """
    Compare two specific timepoints side by side with master layout
    """
    subgrns, master_G, master_pos = plot_subgrns_over_time(
        grn_dict, predicted_pairs, cluster_id, celltype_of_interest)
    
    # Create focused comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot logic for each timepoint would go here...
    # (Similar to above but focused on just two timepoints)
    
    plt.tight_layout()
    plt.show()


# %%
# Run the analysis with adjustable parameters
cluster_id = "26_8"

# Experiment with different settings:
# Option 1: All TFs + Dynamic nodes (RECOMMENDED)
nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
    grn_dict, predicted_pairs, cluster_id, 
    celltype_of_interest="neural_floor_plate",
    spring_k=1.8,          # Higher k = more spread out (increased)
    layout_scale=1.8,      # Larger scale = bigger overall layout (increased)
    max_labels=40,         # More labels to accommodate all TFs
    label_strategy="all_tfs_plus_dynamic"  # Show all TFs + dynamic nodes
)

# Option 2: Just all TFs (cleaner, less crowded)
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.8,
#     layout_scale=1.8,
#     max_labels=30,
#     label_strategy="all_tfs"  # Show only transcription factors
# )

# Option 3: Focus on dynamic nodes only
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.8,
#     layout_scale=1.8,
#     max_labels=25,
#     label_strategy="dynamic_only"  # Show most temporally variable nodes
# )

# Option 4: Original approach with higher spacing
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.6,
#     layout_scale=1.7,
#     max_labels=20,
#     label_strategy="top_connected"  # Show most connected nodes
# )

# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def analyze_edge_types(grn_dict, predicted_pairs, celltype_of_interest="NMPs"):
    """
    Diagnostic function to analyze edge types (activation vs repression) in the raw data
    """
    print(f"\n=== Edge Type Analysis for {celltype_of_interest} ===")
    
    for (celltype, timepoint), grn_df in grn_dict.items():
        if celltype == celltype_of_interest:
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            if len(subgrn) > 0:
                print(f"\nTimepoint {timepoint}:")
                print(f"  Available columns: {list(subgrn.columns)}")
                
                # Check different possible coefficient columns
                for col in ['coef', 'coefficient', 'weight', 'coef_abs']:
                    if col in subgrn.columns:
                        values = subgrn[col]
                        pos_count = sum(values > 0)
                        neg_count = sum(values < 0)
                        zero_count = sum(values == 0)
                        print(f"  Column '{col}': {pos_count} positive, {neg_count} negative, {zero_count} zero")
                        print(f"    Range: {values.min():.4f} to {values.max():.4f}")
                        if col != 'coef_abs':  # Don't show examples for absolute values
                            print(f"    Sample values: {list(values.head())}")

def plot_subgrns_over_time(grn_dict, predicted_pairs, cluster_id="26_8", celltype_of_interest="NMPs", 
                          spring_k=1.2, layout_scale=1.5, max_labels=25, label_strategy="top_connected",
                          debug_labels=False):
    """
    Plot NetworkX diagrams for a celltype-specific subGRNs across all timepoints
    using a master GRN layout for consistent node positioning
    
    Edge colors: Dark Red = Activation, Dark Blue = Repression (based on coefficient sign)
    
    Parameters:
    - spring_k: Spring constant for layout (higher = more spread out)
    - layout_scale: Overall scale of the layout (higher = bigger)
    - max_labels: Maximum number of labels to show
    - label_strategy: "top_connected", "tf_plus_top_targets", "all_tfs_plus_dynamic", 
                     "all_tfs", "dynamic_only", "degree_threshold", or "all"
    - debug_labels: Print debugging info for label positioning and edge types
    """
    # Get all timepoints where the celltype exists
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)
    
    timepoints = sorted(timepoints)
    print(f"Found {celltype_of_interest} at timepoints: {timepoints}")
    
    # Extract subGRNs for celltype at each timepoint
    subgrns = {}
    all_nodes = set()
    all_edges = set()
    
    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]
            
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            subgrns[timepoint] = subgrn
            print(f"Timepoint {timepoint}: {len(subgrn)} edges")
            
            # Collect all nodes and edges for master GRN
            if len(subgrn) > 0:
                all_nodes.update(subgrn['source'])
                all_nodes.update(subgrn['target'])
                all_edges.update(zip(subgrn['source'], subgrn['target']))
    
    print(f"Master GRN: {len(all_nodes)} total nodes, {len(all_edges)} total edges")
    
    # Create master GRN and compute layout
    master_G = nx.DiGraph()
    master_G.add_edges_from(all_edges)
    
    # Compute master layout based on network size
    n_master_nodes = len(master_G.nodes())
    n_master_edges = len(master_G.edges())
    
    print(f"Computing master layout for {n_master_nodes} nodes, {n_master_edges} edges...")
    
    # Choose layout algorithm based on master network properties - further increased spacing
    if n_master_nodes < 30:
        master_pos = nx.circular_layout(master_G, scale=layout_scale*1.1)
    elif n_master_nodes < 80:
        master_pos = nx.spring_layout(master_G, k=spring_k*1.2, iterations=300, seed=42, scale=layout_scale*1.1)
    else:
        try:
            master_pos = nx.kamada_kawai_layout(master_G, scale=layout_scale*1.1)
        except:
            master_pos = nx.spring_layout(master_G, k=spring_k*1.3, iterations=350, seed=42, scale=layout_scale*1.1)
    
    # Calculate dynamic nodes - nodes whose edges change the most over time
    print("Calculating node dynamics across timepoints...")
    node_edge_changes = {}  # node -> total edge changes across time
    
    for node in all_nodes:
        total_changes = 0
        prev_edges = set()
        
        for timepoint in timepoints:
            if timepoint in subgrns and len(subgrns[timepoint]) > 0:
                subgrn = subgrns[timepoint]
                # Get current edges for this node (both incoming and outgoing)
                current_edges = set()
                node_edges = subgrn[(subgrn['source'] == node) | (subgrn['target'] == node)]
                for _, row in node_edges.iterrows():
                    current_edges.add((row['source'], row['target']))
                
                # Calculate changes from previous timepoint
                if prev_edges is not None:
                    gained_edges = current_edges - prev_edges
                    lost_edges = prev_edges - current_edges
                    total_changes += len(gained_edges) + len(lost_edges)
                
                prev_edges = current_edges
        
        node_edge_changes[node] = total_changes
    
    # Get most dynamic nodes
    dynamic_nodes = sorted(node_edge_changes.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 10 most dynamic nodes: {[(node, changes) for node, changes in dynamic_nodes[:10]]}")
    
    # Get node classifications for consistent coloring
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets
    
    # Create subplot layout
    n_timepoints = len(subgrns)
    if n_timepoints <= 3:
        nrows, ncols = 1, n_timepoints
        figsize = (7*n_timepoints, 7)
    else:
        nrows, ncols = 2, 3
        figsize = (21, 14)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_timepoints == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each timepoint using master layout
    for i, (timepoint, subgrn) in enumerate(subgrns.items()):
        ax = axes[i]
        
        if len(subgrn) > 0:
            # Create timepoint-specific graph
            G = nx.DiGraph()
            edge_weights = {}
            edge_signs = {}  # Track positive/negative interactions
            
            # Add edges with weights and signs
            sign_column = None
            for _, row in subgrn.iterrows():
                G.add_edge(row['source'], row['target'])
                edge_weights[(row['source'], row['target'])] = round(row['coef_abs'], 4)
                
                # Determine edge sign - check for original coefficient column
                if 'coef' in row and pd.notna(row['coef']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['coef'] > 0 else -1
                    sign_column = 'coef'
                elif 'coefficient' in row and pd.notna(row['coefficient']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['coefficient'] > 0 else -1
                    sign_column = 'coefficient'
                elif 'weight' in row and pd.notna(row['weight']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['weight'] > 0 else -1
                    sign_column = 'weight'
                else:
                    # If no sign info available, assume positive (activation)
                    edge_signs[(row['source'], row['target'])] = 1
                    sign_column = 'assumed_positive'
            
            # Print edge type information
            if len(subgrn) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                print(f"Timepoint {timepoint}: {pos_count} activation, {neg_count} repression edges (using '{sign_column}' column)")
                
                # Debug: show some coefficient values if debugging enabled
                # if debug_labels and sign_column != 'assumed_positive':
                #     coef_col = sign_column
                #     sample_coefs = subgrn[coef_col].head(5)
                #     print(f"  Sample {coef_col} values: {list(sample_coefs)}")
            
            # Use master positions, but only for nodes present in this timepoint
            pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}
            
            # Draw all master nodes (both present and absent) for consistency
            present_nodes = set(G.nodes())
            absent_nodes = all_nodes - present_nodes
            
            # Classify nodes for this timepoint
            current_tf = present_nodes & tf_nodes
            current_targets = present_nodes & target_genes  
            current_tf_targets = present_nodes & tf_target_nodes
            
            # Draw present nodes with full opacity
            if current_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf), 
                                      node_color='lightcoral', node_size=600, 
                                      ax=ax, alpha=0.9)
            if current_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_targets), 
                                      node_color='lightblue', node_size=400,
                                      ax=ax, alpha=0.9)
            if current_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf_targets), 
                                      node_color='orange', node_size=500,
                                      ax=ax, alpha=0.9)
            
            # Draw absent nodes with low opacity (ghosted)
            absent_tf = absent_nodes & tf_nodes
            absent_targets = absent_nodes & target_genes
            absent_tf_targets = absent_nodes & tf_target_nodes
            
            if absent_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf), 
                                      node_color='lightcoral', node_size=300, 
                                      ax=ax, alpha=0.15)
            if absent_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_targets), 
                                      node_color='lightblue', node_size=200,
                                      ax=ax, alpha=0.15)
            if absent_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf_targets), 
                                      node_color='orange', node_size=250,
                                      ax=ax, alpha=0.15)
            
            # Draw present edges with different colors for activation/repression
            if len(G.edges()) > 0:
                # Separate positive and negative edges
                positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
                negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]
                
                # Draw positive edges (activation) in dark red
                if positive_edges:
                    pos_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in positive_edges]
                    pos_G = nx.DiGraph()
                    pos_G.add_edges_from(positive_edges)
                    nx.draw_networkx_edges(pos_G, pos, width=pos_widths, 
                                          edge_color='darkred', alpha=0.8,
                                          arrowsize=20, arrowstyle='->', ax=ax)
                
                # Draw negative edges (repression) in dark blue
                if negative_edges:
                    neg_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in negative_edges]
                    neg_G = nx.DiGraph()
                    neg_G.add_edges_from(negative_edges)
                    nx.draw_networkx_edges(neg_G, pos, width=neg_widths, 
                                          edge_color='darkblue', alpha=0.8,
                                          arrowsize=20, arrowstyle='->', ax=ax)
            
            # Draw absent edges with very low opacity
            absent_edges = all_edges - set(G.edges())
            if absent_edges:
                absent_G = nx.DiGraph()
                absent_G.add_edges_from(absent_edges)
                # Only draw absent edges if both nodes exist in master_pos
                valid_absent_edges = [(u, v) for u, v in absent_edges 
                                    if u in master_pos and v in master_pos]
                if valid_absent_edges:
                    absent_G_filtered = nx.DiGraph()
                    absent_G_filtered.add_edges_from(valid_absent_edges)
                    nx.draw_networkx_edges(absent_G_filtered, master_pos, 
                                          width=0.5, edge_color='gray', alpha=0.1,
                                          arrowsize=10, arrowstyle='->', ax=ax, style='dashed')
            
            # Selective labeling - configurable strategies
            node_degrees = dict(G.degree())
            
            if label_strategy == "top_connected":
                # Show labels for top N most connected nodes
                if len(node_degrees) > max_labels:
                    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
                    nodes_to_label = [node for node, degree in sorted_nodes[:max_labels]]
                else:
                    nodes_to_label = list(present_nodes)
                    
            elif label_strategy == "tf_plus_top_targets":
                # Always label TFs and TF-targets, plus top target genes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                target_nodes_present = present_nodes & target_genes
                
                # Always label TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add top connected target genes
                target_degrees = {node: degree for node, degree in node_degrees.items() 
                                if node in target_nodes_present}
                if target_degrees:
                    n_targets_to_add = max(5, max_labels - len(nodes_to_label))
                    top_targets = sorted(target_degrees.items(), key=lambda x: x[1], reverse=True)[:n_targets_to_add]
                    nodes_to_label.extend([node for node, degree in top_targets])
                    
            elif label_strategy == "all_tfs_plus_dynamic":
                # Label ALL TFs and TF-targets, plus most dynamic nodes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                
                # Always label ALL TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add most dynamic target genes that aren't already TFs
                target_nodes_present = present_nodes & target_genes
                remaining_slots = max(5, max_labels - len(nodes_to_label))
                
                # Get dynamic target genes (excluding those already labeled as TFs)
                dynamic_targets = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in target_nodes_present and node not in nodes_to_label]
                
                # Add top dynamic target genes
                for node, changes in dynamic_targets[:remaining_slots]:
                    nodes_to_label.append(node)
                    
            elif label_strategy == "all_tfs":
                # Label ALL transcription factors and TF-targets only
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
            elif label_strategy == "dynamic_only":
                # Label only the most dynamic nodes
                dynamic_present = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in present_nodes]
                nodes_to_label = [node for node, changes in dynamic_present[:max_labels]]
                    
            elif label_strategy == "degree_threshold":
                # Label nodes with degree above threshold
                threshold = max(2, np.percentile(list(node_degrees.values()), 70))  # Top 30%
                nodes_to_label = [node for node, degree in node_degrees.items() if degree >= threshold]
                
            else:  # "all"
                nodes_to_label = list(present_nodes)
            
            # Draw labels only for selected nodes that exist in master_pos
            nodes_to_label_filtered = [node for node in nodes_to_label if node in master_pos]
            label_pos = {node: master_pos[node] for node in nodes_to_label_filtered}
            
            # Debug label positioning if enabled
            # if debug_labels:
            #     print(f"  Labeling {len(nodes_to_label_filtered)} nodes: {nodes_to_label_filtered[:10]}...")
            #     # Check specific problematic nodes
            #     for problem_node in ['hnf4g', 'nr5a1a']:
            #         if problem_node in master_pos:
            #             pos = master_pos[problem_node]
            #             is_present = problem_node in present_nodes
            #             is_labeled = problem_node in nodes_to_label_filtered
            #             print(f"  {problem_node}: pos=({pos[0]:.2f},{pos[1]:.2f}), present={is_present}, labeled={is_labeled}")
            
            # Create labels dict for only the nodes we want to show
            labels_to_show = {node: node for node in nodes_to_label_filtered}
            nx.draw_networkx_labels(G, label_pos, labels=labels_to_show, font_size=8, font_weight='bold', ax=ax)
            
            print(f"Timepoint {timepoint}: Showing labels for {len(nodes_to_label_filtered)} out of {len(present_nodes)} nodes")
            
            # Count edge types for title
            if len(G.edges()) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                edge_info = f"({pos_count} activation, {neg_count} repression)"
            else:
                edge_info = ""
            
            # Set consistent axis limits based on master layout
            if master_pos:
                x_coords = [coord[0] for coord in master_pos.values()]
                y_coords = [coord[1] for coord in master_pos.values()]
                margin = 0.15
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}\n({len(G.edges())} edges, {len(G.nodes())} nodes) {edge_info}', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No edges found\nfor timepoint {timepoint}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title and legend
    fig.suptitle(f'{celltype_of_interest.replace("_", " ").title()} Regulatory Network Evolution - Cluster {cluster_id}\n(Master GRN: {len(all_nodes)} nodes, {len(all_edges)} edges)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Enhanced legend with edge types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Transcription Factors (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Target Genes (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=11, label='TF & Target (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Inactive Nodes', alpha=0.3),
        plt.Line2D([0], [0], color='darkred', linewidth=3, label='Activation', alpha=0.8),
        plt.Line2D([0], [0], color='darkblue', linewidth=3, label='Repression', alpha=0.8),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Inactive Edges', alpha=0.3)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    return subgrns, master_G, master_pos

# Alternative function for comparing specific timepoints
def compare_timepoints(grn_dict, predicted_pairs, timepoint1, timepoint2, 
                      cluster_id="26_8", celltype_of_interest="NMPs"):
    """
    Compare two specific timepoints side by side with master layout
    """
    subgrns, master_G, master_pos = plot_subgrns_over_time(
        grn_dict, predicted_pairs, cluster_id, celltype_of_interest)
    
    # Create focused comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot logic for each timepoint would go here...
    # (Similar to above but focused on just two timepoints)
    
    plt.tight_layout()
    plt.show()



# %%
# Run the analysis with adjustable parameters
cluster_id = "26_8"

# FIRST: Analyze edge types in the raw data (uncomment if needed for diagnostics)
# print("=== DIAGNOSTIC: Analyzing edge types in raw data ===")
# analyze_edge_types(grn_dict, predicted_pairs, celltype_of_interest="neural_floor_plate")

# Option 1: All TFs + Dynamic nodes (RECOMMENDED)
nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
    grn_dict, predicted_pairs, cluster_id, 
    celltype_of_interest="neural_floor_plate",
    spring_k=2,          # Higher k = more spread out
    layout_scale=1.8,      # Larger scale = bigger overall layout
    max_labels=50,         # More labels to accommodate all TFs
    label_strategy="all_tfs_plus_dynamic",  # Show all TFs + dynamic nodes
    debug_labels=False     # Clean output without debugging
)

# Option 2: Test with a simpler labeling strategy to debug positioning
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.8,
#     layout_scale=1.8,
#     max_labels=30,
#     label_strategy="all_tfs",  # Show only transcription factors
#     debug_labels=True
# )

# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def analyze_edge_types(grn_dict, predicted_pairs, celltype_of_interest="NMPs"):
    """
    Diagnostic function to analyze edge types (activation vs repression) in the raw data
    """
    print(f"\n=== Edge Type Analysis for {celltype_of_interest} ===")
    
    for (celltype, timepoint), grn_df in grn_dict.items():
        if celltype == celltype_of_interest:
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            if len(subgrn) > 0:
                print(f"\nTimepoint {timepoint}:")
                print(f"  Available columns: {list(subgrn.columns)}")
                
                # Check different possible coefficient columns
                for col in ['coef', 'coefficient', 'weight', 'coef_mean']:
                    if col in subgrn.columns:
                        values = subgrn[col]
                        pos_count = sum(values > 0)
                        neg_count = sum(values < 0)
                        zero_count = sum(values == 0)
                        print(f"  Column '{col}': {pos_count} positive, {neg_count} negative, {zero_count} zero")
                        print(f"    Range: {values.min():.4f} to {values.max():.4f}")
                        if col != 'coef_mean':  # Don't show examples for absolute values
                            print(f"    Sample values: {list(values.head())}")

def plot_subgrns_over_time(grn_dict, predicted_pairs, cluster_id="26_8", celltype_of_interest="NMPs", 
                          spring_k=1.2, layout_scale=1.5, max_labels=25, label_strategy="top_connected",
                          debug_labels=False):
    """
    Plot NetworkX diagrams for a celltype-specific subGRNs across all timepoints
    using a master GRN layout for consistent node positioning
    
    Edge colors: Dark Red = Activation, Dark Blue = Repression (based on coefficient sign)
    
    Parameters:
    - spring_k: Spring constant for layout (higher = more spread out)
    - layout_scale: Overall scale of the layout (higher = bigger)
    - max_labels: Maximum number of labels to show
    - label_strategy: "top_connected", "tf_plus_top_targets", "all_tfs_plus_dynamic", 
                     "all_tfs", "dynamic_only", "degree_threshold", or "all"
    - debug_labels: Print debugging info for label positioning and edge types
    """
    # Get all timepoints where the celltype exists
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)
    
    timepoints = sorted(timepoints)
    print(f"Found {celltype_of_interest} at timepoints: {timepoints}")
    
    # Extract subGRNs for celltype at each timepoint
    subgrns = {}
    all_nodes = set()
    all_edges = set()
    
    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]
            
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            subgrns[timepoint] = subgrn
            print(f"Timepoint {timepoint}: {len(subgrn)} edges")
            
            # Collect all nodes and edges for master GRN
            if len(subgrn) > 0:
                all_nodes.update(subgrn['source'])
                all_nodes.update(subgrn['target'])
                all_edges.update(zip(subgrn['source'], subgrn['target']))
    
    print(f"Master GRN: {len(all_nodes)} total nodes, {len(all_edges)} total edges")
    
    # Create master GRN and compute layout
    master_G = nx.DiGraph()
    master_G.add_edges_from(all_edges)
    
    # Compute master layout based on network size
    n_master_nodes = len(master_G.nodes())
    n_master_edges = len(master_G.edges())
    
    print(f"Computing master layout for {n_master_nodes} nodes, {n_master_edges} edges...")
    
    # Choose layout algorithm based on master network properties - further increased spacing
    if n_master_nodes < 30:
        master_pos = nx.circular_layout(master_G, scale=layout_scale*1.1)
    elif n_master_nodes < 80:
        master_pos = nx.spring_layout(master_G, k=spring_k*1.2, iterations=300, seed=42, scale=layout_scale*1.1)
    else:
        try:
            master_pos = nx.kamada_kawai_layout(master_G, scale=layout_scale*1.1)
        except:
            master_pos = nx.spring_layout(master_G, k=spring_k*1.3, iterations=350, seed=42, scale=layout_scale*1.1)
    
    # Calculate dynamic nodes - nodes whose edges change the most over time
    print("Calculating node dynamics across timepoints...")
    node_edge_changes = {}  # node -> total edge changes across time
    
    for node in all_nodes:
        total_changes = 0
        prev_edges = set()
        
        for timepoint in timepoints:
            if timepoint in subgrns and len(subgrns[timepoint]) > 0:
                subgrn = subgrns[timepoint]
                # Get current edges for this node (both incoming and outgoing)
                current_edges = set()
                node_edges = subgrn[(subgrn['source'] == node) | (subgrn['target'] == node)]
                for _, row in node_edges.iterrows():
                    current_edges.add((row['source'], row['target']))
                
                # Calculate changes from previous timepoint
                if prev_edges is not None:
                    gained_edges = current_edges - prev_edges
                    lost_edges = prev_edges - current_edges
                    total_changes += len(gained_edges) + len(lost_edges)
                
                prev_edges = current_edges
        
        node_edge_changes[node] = total_changes
    
    # Get most dynamic nodes
    dynamic_nodes = sorted(node_edge_changes.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 10 most dynamic nodes: {[(node, changes) for node, changes in dynamic_nodes[:10]]}")
    
    # Get node classifications for consistent coloring
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets
    
    # Create subplot layout
    n_timepoints = len(subgrns)
    if n_timepoints <= 3:
        nrows, ncols = 1, n_timepoints
        figsize = (7*n_timepoints, 7)
    else:
        nrows, ncols = 2, 3
        figsize = (21, 14)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_timepoints == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each timepoint using master layout
    for i, (timepoint, subgrn) in enumerate(subgrns.items()):
        ax = axes[i]
        
        if len(subgrn) > 0:
            # Create timepoint-specific graph
            G = nx.DiGraph()
            edge_weights = {}
            edge_signs = {}  # Track positive/negative interactions
            
            # Add edges with weights and signs
            sign_column = None
            for _, row in subgrn.iterrows():
                G.add_edge(row['source'], row['target'])
                edge_weights[(row['source'], row['target'])] = round(row['coef_abs'], 4)
                
                # Determine edge sign - check for signed coefficient columns
                if 'coef_mean' in row and pd.notna(row['coef_mean']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['coef_mean'] > 0 else -1
                    sign_column = 'coef_mean'
                elif 'coef' in row and pd.notna(row['coef']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['coef'] > 0 else -1
                    sign_column = 'coef'
                elif 'coefficient' in row and pd.notna(row['coefficient']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['coefficient'] > 0 else -1
                    sign_column = 'coefficient'
                elif 'weight' in row and pd.notna(row['weight']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['weight'] > 0 else -1
                    sign_column = 'weight'
                else:
                    # If no sign info available, assume positive (activation)
                    edge_signs[(row['source'], row['target'])] = 1
                    sign_column = 'assumed_positive'
            
            # Print edge type information
            if len(subgrn) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                print(f"Timepoint {timepoint}: {pos_count} activation, {neg_count} repression edges (using '{sign_column}' column)")
                
                # Debug: show some coefficient values if debugging enabled
                # if debug_labels and sign_column != 'assumed_positive':
                #     coef_col = sign_column
                #     sample_coefs = subgrn[coef_col].head(5)
                #     print(f"  Sample {coef_col} values: {list(sample_coefs)}")
            
            # Use master positions, but only for nodes present in this timepoint
            pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}
            
            # Draw all master nodes (both present and absent) for consistency
            present_nodes = set(G.nodes())
            absent_nodes = all_nodes - present_nodes
            
            # Classify nodes for this timepoint
            current_tf = present_nodes & tf_nodes
            current_targets = present_nodes & target_genes  
            current_tf_targets = present_nodes & tf_target_nodes
            
            # Draw present nodes with full opacity
            if current_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf), 
                                      node_color='lightcoral', node_size=600, 
                                      ax=ax, alpha=0.9)
            if current_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_targets), 
                                      node_color='lightblue', node_size=400,
                                      ax=ax, alpha=0.9)
            if current_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf_targets), 
                                      node_color='orange', node_size=500,
                                      ax=ax, alpha=0.9)
            
            # Draw absent nodes with low opacity (ghosted)
            absent_tf = absent_nodes & tf_nodes
            absent_targets = absent_nodes & target_genes
            absent_tf_targets = absent_nodes & tf_target_nodes
            
            if absent_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf), 
                                      node_color='lightcoral', node_size=300, 
                                      ax=ax, alpha=0.15)
            if absent_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_targets), 
                                      node_color='lightblue', node_size=200,
                                      ax=ax, alpha=0.15)
            if absent_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf_targets), 
                                      node_color='orange', node_size=250,
                                      ax=ax, alpha=0.15)
            
            # Draw present edges with different colors for activation/repression
            if len(G.edges()) > 0:
                # Separate positive and negative edges
                positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
                negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]
                
                # Draw positive edges (activation) in dark red
                if positive_edges:
                    pos_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in positive_edges]
                    pos_G = nx.DiGraph()
                    pos_G.add_edges_from(positive_edges)
                    nx.draw_networkx_edges(pos_G, pos, width=pos_widths, 
                                          edge_color='darkred', alpha=0.8,
                                          arrowsize=20, arrowstyle='->', ax=ax)
                
                # Draw negative edges (repression) in dark blue
                if negative_edges:
                    neg_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in negative_edges]
                    neg_G = nx.DiGraph()
                    neg_G.add_edges_from(negative_edges)
                    nx.draw_networkx_edges(neg_G, pos, width=neg_widths, 
                                          edge_color='darkblue', alpha=0.8,
                                          arrowsize=20, arrowstyle='->', ax=ax)
            
            # Draw absent edges with very low opacity
            absent_edges = all_edges - set(G.edges())
            if absent_edges:
                absent_G = nx.DiGraph()
                absent_G.add_edges_from(absent_edges)
                # Only draw absent edges if both nodes exist in master_pos
                valid_absent_edges = [(u, v) for u, v in absent_edges 
                                    if u in master_pos and v in master_pos]
                if valid_absent_edges:
                    absent_G_filtered = nx.DiGraph()
                    absent_G_filtered.add_edges_from(valid_absent_edges)
                    nx.draw_networkx_edges(absent_G_filtered, master_pos, 
                                          width=0.5, edge_color='gray', alpha=0.1,
                                          arrowsize=10, arrowstyle='->', ax=ax, style='dashed')
            
            # Selective labeling - configurable strategies
            node_degrees = dict(G.degree())
            
            if label_strategy == "top_connected":
                # Show labels for top N most connected nodes
                if len(node_degrees) > max_labels:
                    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
                    nodes_to_label = [node for node, degree in sorted_nodes[:max_labels]]
                else:
                    nodes_to_label = list(present_nodes)
                    
            elif label_strategy == "tf_plus_top_targets":
                # Always label TFs and TF-targets, plus top target genes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                target_nodes_present = present_nodes & target_genes
                
                # Always label TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add top connected target genes
                target_degrees = {node: degree for node, degree in node_degrees.items() 
                                if node in target_nodes_present}
                if target_degrees:
                    n_targets_to_add = max(5, max_labels - len(nodes_to_label))
                    top_targets = sorted(target_degrees.items(), key=lambda x: x[1], reverse=True)[:n_targets_to_add]
                    nodes_to_label.extend([node for node, degree in top_targets])
                    
            elif label_strategy == "all_tfs_plus_dynamic":
                # Label ALL TFs and TF-targets, plus most dynamic nodes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                
                # Always label ALL TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add most dynamic target genes that aren't already TFs
                target_nodes_present = present_nodes & target_genes
                remaining_slots = max(5, max_labels - len(nodes_to_label))
                
                # Get dynamic target genes (excluding those already labeled as TFs)
                dynamic_targets = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in target_nodes_present and node not in nodes_to_label]
                
                # Add top dynamic target genes
                for node, changes in dynamic_targets[:remaining_slots]:
                    nodes_to_label.append(node)
                    
            elif label_strategy == "all_tfs":
                # Label ALL transcription factors and TF-targets only
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
            elif label_strategy == "dynamic_only":
                # Label only the most dynamic nodes
                dynamic_present = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in present_nodes]
                nodes_to_label = [node for node, changes in dynamic_present[:max_labels]]
                    
            elif label_strategy == "degree_threshold":
                # Label nodes with degree above threshold
                threshold = max(2, np.percentile(list(node_degrees.values()), 70))  # Top 30%
                nodes_to_label = [node for node, degree in node_degrees.items() if degree >= threshold]
                
            else:  # "all"
                nodes_to_label = list(present_nodes)
            
            # Draw labels only for selected nodes that exist in master_pos
            nodes_to_label_filtered = [node for node in nodes_to_label if node in master_pos]
            label_pos = {node: master_pos[node] for node in nodes_to_label_filtered}
            
            # Debug label positioning if enabled
            # if debug_labels:
            #     print(f"  Labeling {len(nodes_to_label_filtered)} nodes: {nodes_to_label_filtered[:10]}...")
            #     # Check specific problematic nodes
            #     for problem_node in ['hnf4g', 'nr5a1a']:
            #         if problem_node in master_pos:
            #             pos = master_pos[problem_node]
            #             is_present = problem_node in present_nodes
            #             is_labeled = problem_node in nodes_to_label_filtered
            #             print(f"  {problem_node}: pos=({pos[0]:.2f},{pos[1]:.2f}), present={is_present}, labeled={is_labeled}")
            
            # Create labels dict for only the nodes we want to show
            labels_to_show = {node: node for node in nodes_to_label_filtered}
            nx.draw_networkx_labels(G, label_pos, labels=labels_to_show, font_size=8, font_weight='bold', ax=ax)
            
            print(f"Timepoint {timepoint}: Showing labels for {len(nodes_to_label_filtered)} out of {len(present_nodes)} nodes")
            
            # Count edge types for title
            if len(G.edges()) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                edge_info = f"({pos_count} activation, {neg_count} repression)"
            else:
                edge_info = ""
            
            # Set consistent axis limits based on master layout
            if master_pos:
                x_coords = [coord[0] for coord in master_pos.values()]
                y_coords = [coord[1] for coord in master_pos.values()]
                margin = 0.15
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}\n({len(G.edges())} edges, {len(G.nodes())} nodes) {edge_info}', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No edges found\nfor timepoint {timepoint}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title and legend
    fig.suptitle(f'{celltype_of_interest.replace("_", " ").title()} Regulatory Network Evolution - Cluster {cluster_id}\n(Master GRN: {len(all_nodes)} nodes, {len(all_edges)} edges)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Enhanced legend with edge types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Transcription Factors (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Target Genes (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=11, label='TF & Target (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Inactive Nodes', alpha=0.3),
        plt.Line2D([0], [0], color='darkred', linewidth=3, label='Activation', alpha=0.8),
        plt.Line2D([0], [0], color='darkblue', linewidth=3, label='Repression', alpha=0.8),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Inactive Edges', alpha=0.3)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    return subgrns, master_G, master_pos

# Alternative function for comparing specific timepoints
def compare_timepoints(grn_dict, predicted_pairs, timepoint1, timepoint2, 
                      cluster_id="26_8", celltype_of_interest="NMPs"):
    """
    Compare two specific timepoints side by side with master layout
    """
    subgrns, master_G, master_pos = plot_subgrns_over_time(
        grn_dict, predicted_pairs, cluster_id, celltype_of_interest)
    
    # Create focused comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot logic for each timepoint would go here...
    # (Similar to above but focused on just two timepoints)
    
    plt.tight_layout()
    plt.show()

# Run the analysis with adjustable parameters
cluster_id = "26_8"

# DIAGNOSTIC: Check your data structure (uncomment to run)
print("=== CHECKING DATA STRUCTURE ===")
# Let's see what columns are actually available
sample_timepoint = ('neural_floor_plate', '00')  # Adjust if needed
if sample_timepoint in grn_dict:
    sample_df = grn_dict[sample_timepoint]
    print(f"Available columns: {list(sample_df.columns)}")
    print(f"Sample of first few rows:")
    print(sample_df.head())
    
    # Check if there are any negative values in coef_abs (which would be wrong)
    if 'coef_abs' in sample_df.columns:
        print(f"\ncoef_abs range: {sample_df['coef_abs'].min()} to {sample_df['coef_abs'].max()}")
    
    # Look for any other columns that might contain signs
    for col in sample_df.columns:
        if sample_df[col].dtype in ['float64', 'int64'] and col != 'coef_abs':
            has_negatives = (sample_df[col] < 0).any()
            print(f"Column '{col}': has negative values = {has_negatives}")

print("\n" + "="*60)

# Option 1: All TFs + Dynamic nodes (RECOMMENDED)
nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
    grn_dict, predicted_pairs, cluster_id, 
    celltype_of_interest="neural_floor_plate",
    spring_k=1.8,          # Higher k = more spread out
    layout_scale=1.8,      # Larger scale = bigger overall layout
    max_labels=40,         # More labels to accommodate all TFs
    label_strategy="all_tfs_plus_dynamic",  # Show all TFs + dynamic nodes
    debug_labels=False     # Clean output without debugging
)

# Option 2: Test with a simpler labeling strategy to debug positioning
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.8,
#     layout_scale=1.8,
#     max_labels=30,
#     label_strategy="all_tfs",  # Show only transcription factors
#     debug_labels=True
# )

# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def analyze_edge_types(grn_dict, predicted_pairs, celltype_of_interest="NMPs"):
    """
    Diagnostic function to analyze edge types (activation vs repression) in the raw data
    """
    print(f"\n=== Edge Type Analysis for {celltype_of_interest} ===")
    
    for (celltype, timepoint), grn_df in grn_dict.items():
        if celltype == celltype_of_interest:
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            if len(subgrn) > 0:
                print(f"\nTimepoint {timepoint}:")
                print(f"  Available columns: {list(subgrn.columns)}")
                
                # Check different possible coefficient columns
                for col in ['coef', 'coefficient', 'weight', 'coef_abs']:
                    if col in subgrn.columns:
                        values = subgrn[col]
                        pos_count = sum(values > 0)
                        neg_count = sum(values < 0)
                        zero_count = sum(values == 0)
                        print(f"  Column '{col}': {pos_count} positive, {neg_count} negative, {zero_count} zero")
                        print(f"    Range: {values.min():.4f} to {values.max():.4f}")
                        if col != 'coef_abs':  # Don't show examples for absolute values
                            print(f"    Sample values: {list(values.head())}")

def plot_subgrns_over_time(grn_dict, predicted_pairs, cluster_id="26_8", celltype_of_interest="NMPs", 
                          spring_k=1.2, layout_scale=1.5, max_labels=25, label_strategy="top_connected",
                          debug_labels=False):
    """
    Plot NetworkX diagrams for a celltype-specific subGRNs across all timepoints
    using a master GRN layout for consistent node positioning
    
    Edge colors: Dark Red = Activation, Dark Blue = Repression (based on coefficient sign)
    
    Parameters:
    - spring_k: Spring constant for layout (higher = more spread out)
    - layout_scale: Overall scale of the layout (higher = bigger)
    - max_labels: Maximum number of labels to show
    - label_strategy: "top_connected", "tf_plus_top_targets", "all_tfs_plus_dynamic", 
                     "all_tfs", "dynamic_only", "degree_threshold", or "all"
    - debug_labels: Print debugging info for label positioning and edge types
    """
    # Get all timepoints where the celltype exists
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)
    
    timepoints = sorted(timepoints)
    print(f"Found {celltype_of_interest} at timepoints: {timepoints}")
    
    # Extract subGRNs for celltype at each timepoint
    subgrns = {}
    all_nodes = set()
    all_edges = set()
    
    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]
            
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            subgrns[timepoint] = subgrn
            print(f"Timepoint {timepoint}: {len(subgrn)} edges")
            
            # Collect all nodes and edges for master GRN
            if len(subgrn) > 0:
                all_nodes.update(subgrn['source'])
                all_nodes.update(subgrn['target'])
                all_edges.update(zip(subgrn['source'], subgrn['target']))
    
    print(f"Master GRN: {len(all_nodes)} total nodes, {len(all_edges)} total edges")
    
    # Create master GRN and compute layout
    master_G = nx.DiGraph()
    master_G.add_edges_from(all_edges)
    
    # Compute master layout based on network size
    n_master_nodes = len(master_G.nodes())
    n_master_edges = len(master_G.edges())
    
    print(f"Computing master layout for {n_master_nodes} nodes, {n_master_edges} edges...")
    
    # Choose layout algorithm based on master network properties - further increased spacing
    if n_master_nodes < 30:
        master_pos = nx.circular_layout(master_G, scale=layout_scale*1.1)
    elif n_master_nodes < 80:
        master_pos = nx.spring_layout(master_G, k=spring_k*1.2, iterations=300, seed=42, scale=layout_scale*1.1)
    else:
        try:
            master_pos = nx.kamada_kawai_layout(master_G, scale=layout_scale*1.1)
        except:
            master_pos = nx.spring_layout(master_G, k=spring_k*1.3, iterations=350, seed=42, scale=layout_scale*1.1)
    
    # Calculate dynamic nodes - both edge changes AND temporal presence patterns
    print("Calculating node dynamics across timepoints...")
    node_edge_changes = {}  # node -> total edge changes across time
    node_presence = {}      # node -> timepoints where node is present
    
    # Track when each node is present
    for node in all_nodes:
        presence_timepoints = []
        total_changes = 0
        prev_edges = set()
        
        for timepoint in timepoints:
            if timepoint in subgrns and len(subgrns[timepoint]) > 0:
                subgrn = subgrns[timepoint]
                # Get current edges for this node (both incoming and outgoing)
                current_edges = set()
                node_edges = subgrn[(subgrn['source'] == node) | (subgrn['target'] == node)]
                
                if len(node_edges) > 0:
                    presence_timepoints.append(timepoint)
                    for _, row in node_edges.iterrows():
                        current_edges.add((row['source'], row['target']))
                
                # Calculate changes from previous timepoint
                if prev_edges is not None:
                    gained_edges = current_edges - prev_edges
                    lost_edges = prev_edges - current_edges
                    total_changes += len(gained_edges) + len(lost_edges)
                
                prev_edges = current_edges
        
        node_edge_changes[node] = total_changes
        node_presence[node] = presence_timepoints
    
    # Calculate temporal dynamics scores
    node_temporal_dynamics = {}
    for node in all_nodes:
        presence_tp = node_presence[node]
        n_present = len(presence_tp)
        n_total = len(timepoints)
        
        # Transient score: high for nodes present in few timepoints
        transient_score = (n_total - n_present) * 10  # Weight transience highly
        
        # Discontinuous score: high for nodes with gaps in presence
        discontinuous_score = 0
        if n_present > 1:
            # Check for gaps in temporal presence
            tp_indices = [timepoints.index(tp) for tp in presence_tp]
            expected_continuous = list(range(min(tp_indices), max(tp_indices) + 1))
            gaps = len(expected_continuous) - len(tp_indices)
            discontinuous_score = gaps * 15  # Weight discontinuity very highly
        
        # Early/late appearance patterns
        pattern_score = 0
        if n_present > 0:
            first_appearance = timepoints.index(presence_tp[0])
            last_appearance = timepoints.index(presence_tp[-1])
            
            # Early disappearance (appears early, disappears)
            if first_appearance <= 1 and last_appearance < n_total - 2:
                pattern_score += 20
            
            # Late appearance (appears later in development)
            if first_appearance >= 2:
                pattern_score += 15
        
        # Combine scores
        edge_changes = node_edge_changes[node]
        total_dynamics = edge_changes + transient_score + discontinuous_score + pattern_score
        node_temporal_dynamics[node] = total_dynamics
        
    # Get most dynamic nodes (combining edge and presence dynamics)
    dynamic_nodes = sorted(node_temporal_dynamics.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 15 most dynamic nodes (edge + temporal patterns):")
    for i, (node, score) in enumerate(dynamic_nodes[:15]):
        presence_tp = node_presence[node]
        edge_changes = node_edge_changes[node]
        temporal_score = score - edge_changes
        print(f"  {i+1:2d}. {node}: total={score:.0f} (edges={edge_changes}, temporal={temporal_score:.0f}) present in {presence_tp}")
    
    # Identify specific temporal patterns for debugging
    transient_nodes = [(node, tps) for node, tps in node_presence.items() 
                      if len(tps) <= 2 and len(tps) > 0]  # Present in ≤2 timepoints
    early_disappearing = [(node, tps) for node, tps in node_presence.items()
                         if len(tps) > 0 and timepoints.index(tps[-1]) <= 1]  # Last seen in first 2 timepoints
    
    print(f"\nTransient nodes (present ≤2 timepoints): {len(transient_nodes)}")
    for node, tps in transient_nodes[:5]:  # Show top 5
        print(f"  {node}: {tps}")
    
    print(f"\nEarly disappearing nodes: {len(early_disappearing)}")
    for node, tps in early_disappearing[:5]:  # Show top 5
        print(f"  {node}: {tps}")

    
    # Get node classifications for consistent coloring
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets
    
    # Create subplot layout
    n_timepoints = len(subgrns)
    if n_timepoints <= 3:
        nrows, ncols = 1, n_timepoints
        figsize = (7*n_timepoints, 7)
    else:
        nrows, ncols = 2, 3
        figsize = (21, 14)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_timepoints == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each timepoint using master layout
    for i, (timepoint, subgrn) in enumerate(subgrns.items()):
        ax = axes[i]
        
        if len(subgrn) > 0:
            # Create timepoint-specific graph
            G = nx.DiGraph()
            edge_weights = {}
            edge_signs = {}  # Track positive/negative interactions
            
            # Add edges with weights and signs
            sign_column = None
            for _, row in subgrn.iterrows():
                G.add_edge(row['source'], row['target'])
                edge_weights[(row['source'], row['target'])] = round(row['coef_abs'], 4)
                
                # Determine edge sign - check for signed coefficient columns
                if 'coef_mean' in row and pd.notna(row['coef_mean']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['coef_mean'] > 0 else -1
                    sign_column = 'coef_mean'
                elif 'coef' in row and pd.notna(row['coef']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['coef'] > 0 else -1
                    sign_column = 'coef'
                elif 'coefficient' in row and pd.notna(row['coefficient']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['coefficient'] > 0 else -1
                    sign_column = 'coefficient'
                elif 'weight' in row and pd.notna(row['weight']):
                    edge_signs[(row['source'], row['target'])] = 1 if row['weight'] > 0 else -1
                    sign_column = 'weight'
                else:
                    # If no sign info available, assume positive (activation)
                    edge_signs[(row['source'], row['target'])] = 1
                    sign_column = 'assumed_positive'
            
            # Print edge type information
            if len(subgrn) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                print(f"Timepoint {timepoint}: {pos_count} activation, {neg_count} repression edges (using '{sign_column}' column)")
                
                # Debug: show some coefficient values if debugging enabled
                # if debug_labels and sign_column != 'assumed_positive':
                #     coef_col = sign_column
                #     sample_coefs = subgrn[coef_col].head(5)
                #     print(f"  Sample {coef_col} values: {list(sample_coefs)}")
            
            # Use master positions, but only for nodes present in this timepoint
            pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}
            
            # Draw all master nodes (both present and absent) for consistency
            present_nodes = set(G.nodes())
            absent_nodes = all_nodes - present_nodes
            
            # Classify nodes for this timepoint
            current_tf = present_nodes & tf_nodes
            current_targets = present_nodes & target_genes  
            current_tf_targets = present_nodes & tf_target_nodes
            
            # Draw present nodes with full opacity
            if current_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf), 
                                      node_color='lightcoral', node_size=600, 
                                      ax=ax, alpha=0.9)
            if current_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_targets), 
                                      node_color='lightblue', node_size=400,
                                      ax=ax, alpha=0.9)
            if current_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf_targets), 
                                      node_color='orange', node_size=500,
                                      ax=ax, alpha=0.9)
            
            # Draw absent nodes with low opacity (ghosted)
            absent_tf = absent_nodes & tf_nodes
            absent_targets = absent_nodes & target_genes
            absent_tf_targets = absent_nodes & tf_target_nodes
            
            if absent_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf), 
                                      node_color='lightcoral', node_size=300, 
                                      ax=ax, alpha=0.15)
            if absent_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_targets), 
                                      node_color='lightblue', node_size=200,
                                      ax=ax, alpha=0.15)
            if absent_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf_targets), 
                                      node_color='orange', node_size=250,
                                      ax=ax, alpha=0.15)
            
            # Draw present edges with different colors for activation/repression
            if len(G.edges()) > 0:
                # Separate positive and negative edges
                positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
                negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]
                
                # Draw positive edges (activation) in dark red
                if positive_edges:
                    pos_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in positive_edges]
                    pos_G = nx.DiGraph()
                    pos_G.add_edges_from(positive_edges)
                    nx.draw_networkx_edges(pos_G, pos, width=pos_widths, 
                                          edge_color='darkred', alpha=0.8,
                                          arrowsize=20, arrowstyle='->', ax=ax)
                
                # Draw negative edges (repression) in dark blue
                if negative_edges:
                    neg_widths = [edge_weights.get((u, v), 0.1) * 25 for u, v in negative_edges]
                    neg_G = nx.DiGraph()
                    neg_G.add_edges_from(negative_edges)
                    nx.draw_networkx_edges(neg_G, pos, width=neg_widths, 
                                          edge_color='darkblue', alpha=0.8,
                                          arrowsize=20, arrowstyle='->', ax=ax)
            
            # Draw absent edges with very low opacity
            absent_edges = all_edges - set(G.edges())
            if absent_edges:
                absent_G = nx.DiGraph()
                absent_G.add_edges_from(absent_edges)
                # Only draw absent edges if both nodes exist in master_pos
                valid_absent_edges = [(u, v) for u, v in absent_edges 
                                    if u in master_pos and v in master_pos]
                if valid_absent_edges:
                    absent_G_filtered = nx.DiGraph()
                    absent_G_filtered.add_edges_from(valid_absent_edges)
                    nx.draw_networkx_edges(absent_G_filtered, master_pos, 
                                          width=0.5, edge_color='gray', alpha=0.1,
                                          arrowsize=10, arrowstyle='->', ax=ax, style='dashed')
            
            # Selective labeling - configurable strategies
            node_degrees = dict(G.degree())
            
            if label_strategy == "top_connected":
                # Show labels for top N most connected nodes
                if len(node_degrees) > max_labels:
                    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
                    nodes_to_label = [node for node, degree in sorted_nodes[:max_labels]]
                else:
                    nodes_to_label = list(present_nodes)
                    
            elif label_strategy == "tf_plus_top_targets":
                # Always label TFs and TF-targets, plus top target genes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                target_nodes_present = present_nodes & target_genes
                
                # Always label TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add top connected target genes
                target_degrees = {node: degree for node, degree in node_degrees.items() 
                                if node in target_nodes_present}
                if target_degrees:
                    n_targets_to_add = max(5, max_labels - len(nodes_to_label))
                    top_targets = sorted(target_degrees.items(), key=lambda x: x[1], reverse=True)[:n_targets_to_add]
                    nodes_to_label.extend([node for node, degree in top_targets])
                    
            elif label_strategy == "all_tfs_plus_dynamic":
                # Label ALL TFs and TF-targets, plus most dynamic nodes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                
                # Always label ALL TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add most dynamic target genes that aren't already TFs
                target_nodes_present = present_nodes & target_genes
                remaining_slots = max(5, max_labels - len(nodes_to_label))
                
                # Get dynamic target genes (excluding those already labeled as TFs)
                dynamic_targets = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in target_nodes_present and node not in nodes_to_label]
                
                # Add top dynamic target genes
                for node, changes in dynamic_targets[:remaining_slots]:
                    nodes_to_label.append(node)
                    
            elif label_strategy == "all_tfs":
                # Label ALL transcription factors and TF-targets only
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
            elif label_strategy == "dynamic_only":
                # Label only the most dynamic nodes
                dynamic_present = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in present_nodes]
                nodes_to_label = [node for node, changes in dynamic_present[:max_labels]]
                    
            elif label_strategy == "degree_threshold":
                # Label nodes with degree above threshold
                threshold = max(2, np.percentile(list(node_degrees.values()), 70))  # Top 30%
                nodes_to_label = [node for node, degree in node_degrees.items() if degree >= threshold]
                
            else:  # "all"
                nodes_to_label = list(present_nodes)
            
            # Draw labels only for selected nodes that exist in master_pos
            nodes_to_label_filtered = [node for node in nodes_to_label if node in master_pos]
            label_pos = {node: master_pos[node] for node in nodes_to_label_filtered}
            
            # Debug label positioning if enabled
            # if debug_labels:
            #     print(f"  Labeling {len(nodes_to_label_filtered)} nodes: {nodes_to_label_filtered[:10]}...")
            #     # Check specific problematic nodes
            #     for problem_node in ['hnf4g', 'nr5a1a']:
            #         if problem_node in master_pos:
            #             pos = master_pos[problem_node]
            #             is_present = problem_node in present_nodes
            #             is_labeled = problem_node in nodes_to_label_filtered
            #             print(f"  {problem_node}: pos=({pos[0]:.2f},{pos[1]:.2f}), present={is_present}, labeled={is_labeled}")
            
            # Create labels dict for only the nodes we want to show
            labels_to_show = {node: node for node in nodes_to_label_filtered}
            nx.draw_networkx_labels(G, label_pos, labels=labels_to_show, font_size=8, font_weight='bold', ax=ax)
            
            print(f"Timepoint {timepoint}: Showing labels for {len(nodes_to_label_filtered)} out of {len(present_nodes)} nodes")
            
            # Count edge types for title
            if len(G.edges()) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                edge_info = f"({pos_count} activation, {neg_count} repression)"
            else:
                edge_info = ""
            
            # Set consistent axis limits based on master layout
            if master_pos:
                x_coords = [coord[0] for coord in master_pos.values()]
                y_coords = [coord[1] for coord in master_pos.values()]
                margin = 0.15
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}\n({len(G.edges())} edges, {len(G.nodes())} nodes) {edge_info}', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No edges found\nfor timepoint {timepoint}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title and legend
    fig.suptitle(f'{celltype_of_interest.replace("_", " ").title()} Regulatory Network Evolution - Cluster {cluster_id}\n(Master GRN: {len(all_nodes)} nodes, {len(all_edges)} edges)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Enhanced legend with edge types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Transcription Factors (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Target Genes (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=11, label='TF & Target (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Inactive Nodes', alpha=0.3),
        plt.Line2D([0], [0], color='darkred', linewidth=3, label='Activation', alpha=0.8),
        plt.Line2D([0], [0], color='darkblue', linewidth=3, label='Repression', alpha=0.8),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Inactive Edges', alpha=0.3)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    return subgrns, master_G, master_pos

# Alternative function for comparing specific timepoints
def compare_timepoints(grn_dict, predicted_pairs, timepoint1, timepoint2, 
                      cluster_id="26_8", celltype_of_interest="NMPs"):
    """
    Compare two specific timepoints side by side with master layout
    """
    subgrns, master_G, master_pos = plot_subgrns_over_time(
        grn_dict, predicted_pairs, cluster_id, celltype_of_interest)
    
    # Create focused comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot logic for each timepoint would go here...
    # (Similar to above but focused on just two timepoints)
    
    plt.tight_layout()
    plt.show()



# %%
# Run the analysis with adjustable parameters
cluster_id = "26_8"

# # DIAGNOSTIC: Check your data structure (uncomment to run)
# print("=== CHECKING DATA STRUCTURE ===")
# # Let's see what columns are actually available
# sample_timepoint = ('neural_floor_plate', '00')  # Adjust if needed
# if sample_timepoint in grn_dict:
#     sample_df = grn_dict[sample_timepoint]
#     print(f"Available columns: {list(sample_df.columns)}")
#     print(f"Sample of first few rows:")
#     print(sample_df.head())
    
#     # Check if there are any negative values in coef_abs (which would be wrong)
#     if 'coef_abs' in sample_df.columns:
#         print(f"\ncoef_abs range: {sample_df['coef_abs'].min()} to {sample_df['coef_abs'].max()}")
    
#     # Look for any other columns that might contain signs
#     for col in sample_df.columns:
#         if sample_df[col].dtype in ['float64', 'int64'] and col != 'coef_abs':
#             has_negatives = (sample_df[col] < 0).any()
#             print(f"Column '{col}': has negative values = {has_negatives}")

# print("\n" + "="*60)

# Option 1: All TFs + Dynamic nodes (RECOMMENDED)
nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
    grn_dict, predicted_pairs, cluster_id, 
    celltype_of_interest="neural_floor_plate",
    spring_k=1.8,          # Higher k = more spread out
    layout_scale=1.8,      # Larger scale = bigger overall layout
    max_labels=50,         # More labels to accommodate all TFs
    label_strategy="all_tfs_plus_dynamic",  # Show all TFs + dynamic nodes
    debug_labels=False     # Clean output without debugging
)

# Option 2: Test with a simpler labeling strategy to debug positioning
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.8,
#     layout_scale=1.8,
#     max_labels=30,
#     label_strategy="all_tfs",  # Show only transcription factors
#     debug_labels=True
# )

# %%

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def analyze_edge_types(grn_dict, predicted_pairs, celltype_of_interest="NMPs"):
    """
    Diagnostic function to analyze edge types (activation vs repression) in the raw data
    """
    print(f"\n=== Edge Type Analysis for {celltype_of_interest} ===")
    
    for (celltype, timepoint), grn_df in grn_dict.items():
        if celltype == celltype_of_interest:
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            if len(subgrn) > 0:
                print(f"\nTimepoint {timepoint}:")
                print(f"  Available columns: {list(subgrn.columns)}")
                
                # Check different possible coefficient columns
                for col in ['coef', 'coefficient', 'weight', 'coef_abs']:
                    if col in subgrn.columns:
                        values = subgrn[col]
                        pos_count = sum(values > 0)
                        neg_count = sum(values < 0)
                        zero_count = sum(values == 0)
                        print(f"  Column '{col}': {pos_count} positive, {neg_count} negative, {zero_count} zero")
                        print(f"    Range: {values.min():.4f} to {values.max():.4f}")
                        if col != 'coef_abs':  # Don't show examples for absolute values
                            print(f"    Sample values: {list(values.head())}")

def plot_subgrns_over_time(grn_dict, predicted_pairs, cluster_id="26_8", celltype_of_interest="NMPs", 
                          spring_k=1.2, layout_scale=1.5, max_labels=25, label_strategy="top_connected",
                          debug_labels=False, savefig=False, filename=None, max_edge_width=2.0,
                          node_size_scale=1.0, figsize=None):
    """
    Plot NetworkX diagrams for a celltype-specific subGRNs across all timepoints
    using a master GRN layout for consistent node positioning
    
    Edge colors: Dark Red = Activation, Dark Blue = Repression (based on coefficient sign)
    
    Parameters:
    - spring_k: Spring constant for layout (higher = more spread out)
    - layout_scale: Overall scale of the layout (higher = bigger)
    - max_labels: Maximum number of labels to show
    - label_strategy: "top_connected", "tf_plus_top_targets", "all_tfs_plus_dynamic", 
                     "all_tfs", "dynamic_only", "degree_threshold", or "all"
    - debug_labels: Print debugging info for label positioning and edge types
    - savefig: If True, save the figure to file instead of displaying
    - filename: Path/filename for saving (e.g., 'grn_temporal.png', 'grn_temporal.pdf')
    - max_edge_width: Maximum edge thickness (default: 2.0, min will be 0.3)
    - node_size_scale: Scale factor for node sizes (default: 1.0, use 0.6 for smaller figures)
    - figsize: Tuple (width, height) for figure size, None for auto-sizing
    """
    # Get all timepoints where the celltype exists
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)
    
    timepoints = sorted(timepoints)
    print(f"Found {celltype_of_interest} at timepoints: {timepoints}")
    
    # Extract subGRNs for celltype at each timepoint
    subgrns = {}
    all_nodes = set()
    all_edges = set()
    
    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]
            
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            subgrns[timepoint] = subgrn
            print(f"Timepoint {timepoint}: {len(subgrn)} edges")
            
            # Collect all nodes and edges for master GRN
            if len(subgrn) > 0:
                all_nodes.update(subgrn['source'])
                all_nodes.update(subgrn['target'])
                all_edges.update(zip(subgrn['source'], subgrn['target']))
    
    print(f"Master GRN: {len(all_nodes)} total nodes, {len(all_edges)} total edges")
    
    # Create master GRN and compute layout
    master_G = nx.DiGraph()
    master_G.add_edges_from(all_edges)
    
    # Compute master layout based on network size
    n_master_nodes = len(master_G.nodes())
    n_master_edges = len(master_G.edges())
    
    print(f"Computing master layout for {n_master_nodes} nodes, {n_master_edges} edges...")
    
    # Choose layout algorithm based on master network properties - further increased spacing
    if n_master_nodes < 30:
        master_pos = nx.circular_layout(master_G, scale=layout_scale*1.1)
    elif n_master_nodes < 80:
        master_pos = nx.spring_layout(master_G, k=spring_k*1.2, iterations=300, seed=42, scale=layout_scale*1.1)
    else:
        try:
            master_pos = nx.kamada_kawai_layout(master_G, scale=layout_scale*1.1)
        except:
            master_pos = nx.spring_layout(master_G, k=spring_k*1.3, iterations=350, seed=42, scale=layout_scale*1.1)
    
    # Calculate dynamic nodes - both edge changes AND temporal presence patterns
    print("Calculating node dynamics across timepoints...")
    node_edge_changes = {}  # node -> total edge changes across time
    node_presence = {}      # node -> timepoints where node is present
    
    # Track when each node is present
    for node in all_nodes:
        presence_timepoints = []
        total_changes = 0
        prev_edges = set()
        
        for timepoint in timepoints:
            if timepoint in subgrns and len(subgrns[timepoint]) > 0:
                subgrn = subgrns[timepoint]
                # Get current edges for this node (both incoming and outgoing)
                current_edges = set()
                node_edges = subgrn[(subgrn['source'] == node) | (subgrn['target'] == node)]
                
                if len(node_edges) > 0:
                    presence_timepoints.append(timepoint)
                    for _, row in node_edges.iterrows():
                        current_edges.add((row['source'], row['target']))
                
                # Calculate changes from previous timepoint
                if prev_edges is not None:
                    gained_edges = current_edges - prev_edges
                    lost_edges = prev_edges - current_edges
                    total_changes += len(gained_edges) + len(lost_edges)
                
                prev_edges = current_edges
        
        node_edge_changes[node] = total_changes
        node_presence[node] = presence_timepoints
    
    # Calculate temporal dynamics scores
    node_temporal_dynamics = {}
    for node in all_nodes:
        presence_tp = node_presence[node]
        n_present = len(presence_tp)
        n_total = len(timepoints)
        
        # Transient score: high for nodes present in few timepoints
        transient_score = (n_total - n_present) * 10  # Weight transience highly
        
        # Discontinuous score: high for nodes with gaps in presence
        discontinuous_score = 0
        if n_present > 1:
            # Check for gaps in temporal presence
            tp_indices = [timepoints.index(tp) for tp in presence_tp]
            expected_continuous = list(range(min(tp_indices), max(tp_indices) + 1))
            gaps = len(expected_continuous) - len(tp_indices)
            discontinuous_score = gaps * 15  # Weight discontinuity very highly
        
        # Early/late appearance patterns
        pattern_score = 0
        if n_present > 0:
            first_appearance = timepoints.index(presence_tp[0])
            last_appearance = timepoints.index(presence_tp[-1])
            
            # Early disappearance (appears early, disappears)
            if first_appearance <= 1 and last_appearance < n_total - 2:
                pattern_score += 20
            
            # Late appearance (appears later in development)
            if first_appearance >= 2:
                pattern_score += 15
        
        # Combine scores
        edge_changes = node_edge_changes[node]
        total_dynamics = edge_changes + transient_score + discontinuous_score + pattern_score
        node_temporal_dynamics[node] = total_dynamics
        
    # Get most dynamic nodes (combining edge and presence dynamics)
    dynamic_nodes = sorted(node_temporal_dynamics.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 15 most dynamic nodes (edge + temporal patterns):")
    for i, (node, score) in enumerate(dynamic_nodes[:15]):
        presence_tp = node_presence[node]
        edge_changes = node_edge_changes[node]
        temporal_score = score - edge_changes
        print(f"  {i+1:2d}. {node}: total={score:.0f} (edges={edge_changes}, temporal={temporal_score:.0f}) present in {presence_tp}")
    
    # Identify specific temporal patterns for debugging
    transient_nodes = [(node, tps) for node, tps in node_presence.items() 
                      if len(tps) <= 2 and len(tps) > 0]  # Present in ≤2 timepoints
    early_disappearing = [(node, tps) for node, tps in node_presence.items()
                         if len(tps) > 0 and timepoints.index(tps[-1]) <= 1]  # Last seen in first 2 timepoints
    
    print(f"\nTransient nodes (present ≤2 timepoints): {len(transient_nodes)}")
    for node, tps in transient_nodes[:5]:  # Show top 5
        print(f"  {node}: {tps}")
    
    print(f"\nEarly disappearing nodes: {len(early_disappearing)}")
    for node, tps in early_disappearing[:5]:  # Show top 5
        print(f"  {node}: {tps}")

    
    # Get node classifications for consistent coloring
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets
    
    # Create subplot layout with configurable figure size
    n_timepoints = len(subgrns)
    if figsize is None:
        # Default figure size calculation
        if n_timepoints <= 3:
            nrows, ncols = 1, n_timepoints
            figsize = (4*n_timepoints, 4)
        else:
            nrows, ncols = 2, 3
            figsize = (12, 8)
    else:
        # Use provided figure size and calculate grid
        if n_timepoints <= 3:
            nrows, ncols = 1, n_timepoints
        else:
            nrows, ncols = 2, 3
    
    # Calculate node sizes based on scale factor
    tf_node_size = int(400 * node_size_scale)
    target_node_size = int(250 * node_size_scale)
    tf_target_node_size = int(320 * node_size_scale)
    
    # Inactive node sizes (smaller)
    inactive_tf_size = int(200 * node_size_scale)
    inactive_target_size = int(120 * node_size_scale)
    inactive_tf_target_size = int(160 * node_size_scale)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_timepoints == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each timepoint using master layout
    for i, (timepoint, subgrn) in enumerate(subgrns.items()):
        ax = axes[i]
        
        if len(subgrn) > 0:
            # Create timepoint-specific graph
            G = nx.DiGraph()
            edge_weights = {}
            edge_signs = {}  # Track positive/negative interactions
            
            # Add edges with weights and signs
            sign_column = None
            for _, row in subgrn.iterrows():
                G.add_edge(row['source'], row['target'])
                
                # Use coef_mean for everything - absolute value for thickness, sign for color
                if 'coef_mean' in row and pd.notna(row['coef_mean']):
                    coef_value = row['coef_mean']
                    edge_weights[(row['source'], row['target'])] = round(abs(coef_value), 4)
                    edge_signs[(row['source'], row['target'])] = 1 if coef_value > 0 else -1
                    sign_column = 'coef_mean'
                elif 'coef' in row and pd.notna(row['coef']):
                    coef_value = row['coef']
                    edge_weights[(row['source'], row['target'])] = round(abs(coef_value), 4)
                    edge_signs[(row['source'], row['target'])] = 1 if coef_value > 0 else -1
                    sign_column = 'coef'
                elif 'coefficient' in row and pd.notna(row['coefficient']):
                    coef_value = row['coefficient']
                    edge_weights[(row['source'], row['target'])] = round(abs(coef_value), 4)
                    edge_signs[(row['source'], row['target'])] = 1 if coef_value > 0 else -1
                    sign_column = 'coefficient'
                else:
                    # Fallback to coef_abs if no signed coefficient available
                    edge_weights[(row['source'], row['target'])] = round(row.get('coef_abs', 0.1), 4)
                    edge_signs[(row['source'], row['target'])] = 1
                    sign_column = 'assumed_positive'
            
            # Print edge type information
            if len(subgrn) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                print(f"Timepoint {timepoint}: {pos_count} activation, {neg_count} repression edges (using '{sign_column}' column)")
                
                # Debug: show some coefficient values if debugging enabled
                # if debug_labels and sign_column != 'assumed_positive':
                #     coef_col = sign_column
                #     sample_coefs = subgrn[coef_col].head(5)
                #     print(f"  Sample {coef_col} values: {list(sample_coefs)}")
            
            # Use master positions, but only for nodes present in this timepoint
            pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}
            
            # Draw all master nodes (both present and absent) for consistency
            present_nodes = set(G.nodes())
            absent_nodes = all_nodes - present_nodes
            
            # Classify nodes for this timepoint
            current_tf = present_nodes & tf_nodes
            current_targets = present_nodes & target_genes  
            current_tf_targets = present_nodes & tf_target_nodes
            
            # Draw present nodes with full opacity
            if current_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf), 
                                      node_color='lightcoral', node_size=tf_node_size, 
                                      ax=ax, alpha=0.9)
            if current_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_targets), 
                                      node_color='lightblue', node_size=target_node_size,
                                      ax=ax, alpha=0.9)
            if current_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf_targets), 
                                      node_color='orange', node_size=tf_target_node_size,
                                      ax=ax, alpha=0.9)
            
            # Draw absent nodes with low opacity (ghosted)
            absent_tf = absent_nodes & tf_nodes
            absent_targets = absent_nodes & target_genes
            absent_tf_targets = absent_nodes & tf_target_nodes
            
            if absent_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf), 
                                      node_color='lightcoral', node_size=inactive_tf_size, 
                                      ax=ax, alpha=0.15)
            if absent_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_targets), 
                                      node_color='lightblue', node_size=inactive_target_size,
                                      ax=ax, alpha=0.15)
            if absent_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf_targets), 
                                      node_color='orange', node_size=inactive_tf_target_size,
                                      ax=ax, alpha=0.15)
            
            # Draw present edges with different colors for activation/repression
            if len(G.edges()) > 0:
                # Calculate scaled edge widths (max thickness = 2, min = 0.3)
                all_weights = [edge_weights.get((u, v), 0.1) for u, v in G.edges()]
                max_weight = max(all_weights) if all_weights else 0.1
                min_weight = min(all_weights) if all_weights else 0.1
                
                def scale_width(weight):
                    # Scale weights to range [0.3, max_edge_width]
                    if max_weight == min_weight:
                        return max_edge_width * 0.6  # Use 60% of max if all weights equal
                    normalized = (weight - min_weight) / (max_weight - min_weight)
                    min_width = 0.3
                    return min_width + normalized * (max_edge_width - min_width)
                
                # Separate positive and negative edges
                positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
                negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]
                
                # Draw positive edges (activation) in dark red
                if positive_edges:
                    pos_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in positive_edges]
                    pos_G = nx.DiGraph()
                    pos_G.add_edges_from(positive_edges)
                    nx.draw_networkx_edges(pos_G, pos, width=pos_widths, 
                                          edge_color='darkred', alpha=0.8,
                                          arrowsize=15, arrowstyle='->', ax=ax)
                
                # Draw negative edges (repression) in dark blue
                if negative_edges:
                    neg_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in negative_edges]
                    neg_G = nx.DiGraph()
                    neg_G.add_edges_from(negative_edges)
                    nx.draw_networkx_edges(neg_G, pos, width=neg_widths, 
                                          edge_color='darkblue', alpha=0.8,
                                          arrowsize=15, arrowstyle='->', ax=ax)
            
            # Draw absent edges with very low opacity
            absent_edges = all_edges - set(G.edges())
            if absent_edges:
                absent_G = nx.DiGraph()
                absent_G.add_edges_from(absent_edges)
                # Only draw absent edges if both nodes exist in master_pos
                valid_absent_edges = [(u, v) for u, v in absent_edges 
                                    if u in master_pos and v in master_pos]
                if valid_absent_edges:
                    absent_G_filtered = nx.DiGraph()
                    absent_G_filtered.add_edges_from(valid_absent_edges)
                    nx.draw_networkx_edges(absent_G_filtered, master_pos, 
                                          width=0.5, edge_color='gray', alpha=0.1,
                                          arrowsize=10, arrowstyle='->', ax=ax, style='dashed')
            
            # Selective labeling - configurable strategies
            node_degrees = dict(G.degree())
            
            if label_strategy == "top_connected":
                # Show labels for top N most connected nodes
                if len(node_degrees) > max_labels:
                    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
                    nodes_to_label = [node for node, degree in sorted_nodes[:max_labels]]
                else:
                    nodes_to_label = list(present_nodes)
                    
            elif label_strategy == "tf_plus_top_targets":
                # Always label TFs and TF-targets, plus top target genes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                target_nodes_present = present_nodes & target_genes
                
                # Always label TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add top connected target genes
                target_degrees = {node: degree for node, degree in node_degrees.items() 
                                if node in target_nodes_present}
                if target_degrees:
                    n_targets_to_add = max(5, max_labels - len(nodes_to_label))
                    top_targets = sorted(target_degrees.items(), key=lambda x: x[1], reverse=True)[:n_targets_to_add]
                    nodes_to_label.extend([node for node, degree in top_targets])
                    
            elif label_strategy == "all_tfs_plus_dynamic":
                # Label ALL TFs and TF-targets, plus most dynamic nodes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                
                # Always label ALL TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add most dynamic target genes that aren't already TFs
                target_nodes_present = present_nodes & target_genes
                remaining_slots = max(5, max_labels - len(nodes_to_label))
                
                # Get dynamic target genes (excluding those already labeled as TFs)
                dynamic_targets = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in target_nodes_present and node not in nodes_to_label]
                
                # Add top dynamic target genes
                for node, changes in dynamic_targets[:remaining_slots]:
                    nodes_to_label.append(node)
                    
            elif label_strategy == "all_tfs":
                # Label ALL transcription factors and TF-targets only
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
            elif label_strategy == "dynamic_only":
                # Label only the most dynamic nodes
                dynamic_present = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in present_nodes]
                nodes_to_label = [node for node, changes in dynamic_present[:max_labels]]
                    
            elif label_strategy == "degree_threshold":
                # Label nodes with degree above threshold
                threshold = max(2, np.percentile(list(node_degrees.values()), 70))  # Top 30%
                nodes_to_label = [node for node, degree in node_degrees.items() if degree >= threshold]
                
            else:  # "all"
                nodes_to_label = list(present_nodes)
            
            # Draw labels only for selected nodes that exist in master_pos
            nodes_to_label_filtered = [node for node in nodes_to_label if node in master_pos]
            label_pos = {node: master_pos[node] for node in nodes_to_label_filtered}
            
            # Debug label positioning if enabled
            # if debug_labels:
            #     print(f"  Labeling {len(nodes_to_label_filtered)} nodes: {nodes_to_label_filtered[:10]}...")
            #     # Check specific problematic nodes
            #     for problem_node in ['hnf4g', 'nr5a1a']:
            #         if problem_node in master_pos:
            #             pos = master_pos[problem_node]
            #             is_present = problem_node in present_nodes
            #             is_labeled = problem_node in nodes_to_label_filtered
            #             print(f"  {problem_node}: pos=({pos[0]:.2f},{pos[1]:.2f}), present={is_present}, labeled={is_labeled}")
            
            # Create labels dict for only the nodes we want to show
            labels_to_show = {node: node for node in nodes_to_label_filtered}
            nx.draw_networkx_labels(G, label_pos, labels=labels_to_show, font_size=8, font_weight='bold', ax=ax)
            
            print(f"Timepoint {timepoint}: Showing labels for {len(nodes_to_label_filtered)} out of {len(present_nodes)} nodes")
            
            # Count edge types for title
            if len(G.edges()) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                edge_info = f"({pos_count} activation, {neg_count} repression)"
            else:
                edge_info = ""
            
            # Set consistent axis limits based on master layout
            if master_pos:
                x_coords = [coord[0] for coord in master_pos.values()]
                y_coords = [coord[1] for coord in master_pos.values()]
                margin = 0.15
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}\n({len(G.edges())} edges, {len(G.nodes())} nodes) {edge_info}', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No edges found\nfor timepoint {timepoint}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title and legend
    fig.suptitle(f'{celltype_of_interest.replace("_", " ").title()} Regulatory Network Evolution - Cluster {cluster_id}\n(Master GRN: {len(all_nodes)} nodes, {len(all_edges)} edges)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Enhanced legend with edge types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Transcription Factors (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Target Genes (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=11, label='TF & Target (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Inactive Nodes', alpha=0.3),
        plt.Line2D([0], [0], color='darkred', linewidth=3, label='Activation', alpha=0.8),
        plt.Line2D([0], [0], color='darkblue', linewidth=3, label='Repression', alpha=0.8),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Inactive Edges', alpha=0.3)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure if requested
    if savefig:
        if filename is None:
            # Generate default filename
            filename = f"{celltype_of_interest}_grn_temporal_{cluster_id}.png"
        
        # Save with high DPI for publication quality
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved as: {filename}")
    
    plt.show()
    
    return subgrns, master_G, master_pos

# Alternative function for comparing specific timepoints
def compare_timepoints(grn_dict, predicted_pairs, timepoint1, timepoint2, 
                      cluster_id="26_8", celltype_of_interest="NMPs"):
    """
    Compare two specific timepoints side by side with master layout
    """
    subgrns, master_G, master_pos = plot_subgrns_over_time(
        grn_dict, predicted_pairs, cluster_id, celltype_of_interest)
    
    # Create focused comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot logic for each timepoint would go here...
    # (Similar to above but focused on just two timepoints)
    
    plt.tight_layout()
    plt.show()


# %%
# Run the analysis with adjustable parameters
cluster_id = "26_8"

# Option 1: All TFs + Dynamic nodes (RECOMMENDED)
nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
    grn_dict, predicted_pairs, cluster_id, 
    celltype_of_interest="neural_floor_plate",
    spring_k=1.8,          # Higher k = more spread out
    layout_scale=1.8,      # Larger scale = bigger overall layout
    max_labels=50,         # More labels to accommodate all TFs
    label_strategy="all_tfs_plus_dynamic",  # Show all TFs + dynamic nodes
    debug_labels=False,    # Clean output without debugging
    savefig=True,         # Set to True to save instead of display
    filename=figpath+f"subGRN_{cluster_id}_neural_floor_plate.pdf",         # Will auto-generate filename if savefig=True
    max_edge_width=2.0,     # Maximum edge thickness (adjust as needed)
    figsize=(15,10),
    node_size_scale=0.5
)

# Option 2: Even thinner edges (uncomment to try)
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.8,
#     layout_scale=1.8,
#     max_labels=40,
#     label_strategy="all_tfs_plus_dynamic",
#     debug_labels=False,
#     savefig=False,
#     filename=None,
#     max_edge_width=1.0     # Thinner maximum edge thickness
# )

# Option 3: Save to file (uncomment to use)
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.8,
#     layout_scale=1.8,
#     max_labels=40,
#     label_strategy="all_tfs_plus_dynamic",
#     debug_labels=False,
#     savefig=True,          # Save the figure
#     filename="neural_floor_plate_grn_evolution.png",  # Custom filename
#     max_edge_width=2.0     # Edge thickness
# )

# Option 2: Test with a simpler labeling strategy to debug positioning
# nmp_subgrns, master_grn, master_positions = plot_subgrns_over_time(
#     grn_dict, predicted_pairs, cluster_id, 
#     celltype_of_interest="neural_floor_plate",
#     spring_k=1.8,
#     layout_scale=1.8,
#     max_labels=30,
#     label_strategy="all_tfs",  # Show only transcription factors
#     debug_labels=True
# )

# %% [markdown]
# ## quantification of subGRN modules 
# - updated: 9/24/2025

# %%
import pandas as pd
import numpy as np

def extract_subgrn_metrics(cluster_id, celltype_of_interest, grn_dict, cluster_tf_gene_matrices, predicted_pairs):
    """
    Extract key metrics for subGRN analysis to fill in manuscript statement.
    
    Parameters:
    -----------
    cluster_id : str
        The cluster ID to analyze (e.g., "26_8")
    celltype_of_interest : str  
        The celltype to focus on (e.g., "neural_floor_plate")
    grn_dict : dict
        Dictionary of GRNs keyed by (celltype, timepoint)
    cluster_tf_gene_matrices : dict
        Dictionary of TF-gene matrices for each cluster
    predicted_pairs : list
        List of predicted TF-target pairs from the mesh
        
    Returns:
    --------
    dict : Dictionary containing all the metrics
    """
    
    print(f"=== Extracting metrics for cluster {cluster_id} in {celltype_of_interest} ===")
    
    # Get all timepoints for this celltype
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)
    timepoints = sorted(timepoints)
    
    # Extract subGRNs for each timepoint
    all_subgrn_nodes = set()
    all_subgrn_edges = set()
    subgrns_by_timepoint = {}
    
    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]
            
            # Find intersection with predicted pairs (same logic as in your notebook)
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            subgrns_by_timepoint[timepoint] = subgrn
            
            if len(subgrn) > 0:
                # Collect nodes and edges
                all_subgrn_nodes.update(subgrn['source'])
                all_subgrn_nodes.update(subgrn['target'])
                all_subgrn_edges.update(zip(subgrn['source'], subgrn['target']))
    
    # Calculate node classifications
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns_by_timepoint.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_only_nodes = all_sources - all_targets  # TFs that are never targets
    target_only_nodes = all_targets - all_sources  # Targets that are never TFs  
    dual_tf_target_nodes = all_sources & all_targets  # Both TF and target
    
    # Calculate metrics
    total_nodes = len(all_subgrn_nodes)  # NN
    total_tfs = len(all_sources)  # N (all nodes that act as TFs)
    total_targets = len(all_targets)  # M (all nodes that act as targets)
    total_edges = len(all_subgrn_edges)
    dual_nodes = len(dual_tf_target_nodes)
    
    # Calculate complexity reduction (XX%)
    # Compare subGRN size to original mesh size
    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    original_tf_count = len(cluster_matrix.index)  # TFs in mesh
    original_gene_count = len(cluster_matrix.columns)  # Genes in mesh
    original_possible_edges = (cluster_matrix == 1).sum().sum()  # Actual predicted edges in mesh
    
    # Calculate reduction percentages
    node_reduction = (1 - total_nodes / (original_tf_count + original_gene_count)) * 100
    edge_reduction = (1 - total_edges / original_possible_edges) * 100 if original_possible_edges > 0 else 0
    
    # Overall complexity reduction (using edge reduction as primary metric)
    complexity_reduction = edge_reduction
    
    # Print detailed breakdown
    print(f"\n=== SUBGRN COMPOSITION ===")
    print(f"Total nodes (NN): {total_nodes}")
    print(f"  - TF-only nodes: {len(tf_only_nodes)}")
    print(f"  - Target-only nodes: {len(target_only_nodes)}")  
    print(f"  - Dual TF/Target nodes: {dual_nodes}")
    print(f"Total transcription factors (N): {total_tfs}")
    print(f"Total target genes (M): {total_targets}")
    print(f"Total edges: {total_edges}")
    
    print(f"\n=== ORIGINAL MESH COMPARISON ===")
    print(f"Original mesh TFs: {original_tf_count}")
    print(f"Original mesh genes: {original_gene_count}")
    print(f"Original predicted edges: {original_possible_edges}")
    print(f"Node reduction: {node_reduction:.1f}%")
    print(f"Edge reduction: {edge_reduction:.1f}%")
    print(f"Overall complexity reduction (XX): {complexity_reduction:.1f}%")
    
    print(f"\n=== TEMPORAL ANALYSIS ===")
    for timepoint in timepoints:
        subgrn = subgrns_by_timepoint[timepoint]
        print(f"Timepoint {timepoint}: {len(subgrn)} edges, {len(set(subgrn['source']) | set(subgrn['target'])) if len(subgrn) > 0 else 0} nodes")
    
    # Return all metrics
    metrics = {
        'cluster_id': cluster_id,
        'celltype': celltype_of_interest,
        'total_nodes_NN': total_nodes,
        'total_tfs_N': total_tfs, 
        'total_targets_M': total_targets,
        'dual_nodes': dual_nodes,
        'total_edges': total_edges,
        'complexity_reduction_XX': round(complexity_reduction, 1),
        'node_classifications': {
            'tf_only': list(tf_only_nodes),
            'target_only': list(target_only_nodes), 
            'dual_tf_target': list(dual_tf_target_nodes)
        },
        'temporal_breakdown': {tp: len(subgrns_by_timepoint[tp]) for tp in timepoints}
    }
    
    return metrics



# %%
cluster_matrix = cluster_tf_gene_matrices[cluster_id]

# Extract all predicted TF-target pairs from this mesh
predicted_pairs = []
for tf in cluster_matrix.index:
    for gene in cluster_matrix.columns:
        if cluster_matrix.loc[tf, gene] == 1:
            predicted_pairs.append((tf, gene))

# %%
# Example usage for your specific cluster:
cluster_id = "26_8"
celltype_of_interest = "neural_floor_plate"

# Extract metrics using your existing data structures
metrics = extract_subgrn_metrics(
    cluster_id=cluster_id,
    celltype_of_interest=celltype_of_interest, 
    grn_dict=grn_dict,
    cluster_tf_gene_matrices=cluster_tf_gene_matrices,
    predicted_pairs=predicted_pairs
)

# Print the values for your manuscript
print(f"\n{'='*50}")
print(f"VALUES FOR MANUSCRIPT:")
print(f"{'='*50}")
print(f"NN (total nodes): {metrics['total_nodes_NN']}")
print(f"N (transcription factors): {metrics['total_tfs_N']}")
print(f"M (target genes): {metrics['total_targets_M']}")
print(f"XX (complexity reduction): {metrics['complexity_reduction_XX']}%")
print(f"Dual TF/Target nodes: {metrics['dual_nodes']}")

print(f"\n=== MANUSCRIPT TEXT ===")
manuscript_text = f"""This approach successfully identified modular regulatory programs with distinct temporal dynamics. Analysis of regulatory program "{cluster_id}" in neural floor plate cells revealed a {metrics['total_nodes_NN']}-node subGRN containing {metrics['total_tfs_N']} transcription factors and {metrics['total_targets_M']} target genes (Fig.~\\ref{{fig:grn}}h). The network showed clear temporal dynamics, with specific TF-gene interactions active at early stages (10-14 hpf) becoming inactive by later timepoints (19-24 hpf), while new regulatory connections emerged during mid-development. Notably, {metrics['dual_nodes']} nodes functioned as both transcription factors and target genes (highlighted in yellow), indicating hierarchical regulatory cascades within the module. This regulatory program-based approach reduced network complexity by {metrics['complexity_reduction_XX']}% while preserving biologically meaningful modules, enabling detailed analysis of how specific regulatory circuits evolve during development."""

print(manuscript_text)

# Optional: Save metrics to file
import json
with open(f'subgrn_metrics_{cluster_id}_{celltype_of_interest}.json', 'w') as f:
    # Convert sets to lists for JSON serialization
    metrics_json = metrics.copy()
    for key, value in metrics_json['node_classifications'].items():
        if isinstance(value, set):
            metrics_json['node_classifications'][key] = list(value)
    json.dump(metrics_json, f, indent=2)
    
print(f"\nMetrics saved to: subgrn_metrics_{cluster_id}_{celltype_of_interest}.json")

# %%

# %% [markdown]
# ## Systematic Discovery of Dynamic SubGRNs via Temporal Dynamics Scoring
# - last updated: 10/9/2025
# - Goal: Rank all 402 peak clusters by temporal dynamics to identify biologically interesting regulatory programs
# - Strategy:
#   1. For each cluster, find the most accessible celltype (via Gini coefficient)
#   2. Extract subGRNs for that celltype across all timepoints
#   3. Compute temporal dynamics score (focus on developmental TF turnover)
#   4. Rank and visualize top candidates

# %%
def gini_coefficient(values):
    """
    Calculate Gini coefficient to measure concentration of accessibility.

    Parameters:
    -----------
    values : array-like
        Accessibility values across all pseudobulk groups

    Returns:
    --------
    float : Gini coefficient (0=equal distribution, 1=concentrated in few groups)
    """
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


def find_most_accessible_celltype(cluster_id, df_clusters_groups, min_accessibility=0.01):
    """
    Find the celltype×timepoint with highest accessibility for a peak cluster.

    Parameters:
    -----------
    cluster_id : str
        Peak cluster ID (e.g., "26_8")
    df_clusters_groups : pd.DataFrame
        Cluster × pseudobulk accessibility matrix
    min_accessibility : float
        Minimum threshold to consider (filter noise)

    Returns:
    --------
    dict with:
        - best_group: str (e.g., "neural_floor_plate_14somites")
        - celltype: str (e.g., "neural_floor_plate")
        - timepoint: str (e.g., "14somites" or "03")
        - accessibility: float
        - gini_coefficient: float (concentration score)
        - top_5_groups: list of (group, accessibility) tuples
    """

    # Get accessibility profile for this cluster
    values = df_clusters_groups.loc[cluster_id]

    # Filter by minimum threshold
    values_filtered = values[values >= min_accessibility]

    if len(values_filtered) == 0:
        return None  # No signal

    # Compute Gini coefficient (concentration)
    gini = gini_coefficient(values.values)

    # Find top group
    best_group = values_filtered.idxmax()
    best_value = values_filtered.max()

    # Parse celltype and timepoint from group name
    # Format: "celltype_timepoint" (e.g., "neural_floor_plate_14somites")
    parts = best_group.rsplit('_', 1)
    if len(parts) == 2:
        celltype, timepoint = parts
    else:
        celltype = best_group
        timepoint = None

    # Get top 5 for context
    top_5 = [(group, val) for group, val in
             values_filtered.sort_values(ascending=False).head(5).items()]

    return {
        'cluster_id': cluster_id,
        'best_group': best_group,
        'celltype': celltype,
        'timepoint': timepoint,
        'accessibility': best_value,
        'gini_coefficient': gini,
        'top_5_groups': top_5
    }

# %%
def compute_temporal_dynamics_score(cluster_id, celltype, grn_dict,
                                    cluster_tf_gene_matrices,
                                    developmental_tfs=None):
    """
    Compute how dynamically a subGRN changes over time for a specific celltype.

    Parameters:
    -----------
    cluster_id : str
        Peak cluster ID
    celltype : str
        Celltype to analyze (e.g., "neural_floor_plate")
    grn_dict : dict
        {(celltype, timepoint): grn_df}
    cluster_tf_gene_matrices : dict
        {cluster_id: TF×gene binary matrix}
    developmental_tfs : set, optional
        Set of well-known developmental TF names to track specifically

    Returns:
    --------
    dict with dynamics metrics:
        - n_timepoints: int
        - node_turnover_rate: float (0-1, higher = more dynamic)
        - edge_turnover_rate: float (0-1, higher = more dynamic)
        - tf_turnover_rate: float (focuses on TF nodes only)
        - developmental_tf_dynamics: float (focuses on known dev TFs)
        - temporal_variance: float (std of edge counts across timepoints)
        - dynamics_score: float (composite score, 0-1)
        - subgrns_by_timepoint: dict {timepoint: subgrn_df}
    """

    if developmental_tfs is None:
        # Well-known zebrafish developmental TFs
        developmental_tfs = {
            'sox2', 'sox3', 'sox9a', 'sox9b', 'sox10', 'sox19a', 'sox19b',
            'pax6a', 'pax6b', 'pax2a', 'pax8',
            'tbx6', 'tbx16', 'tbx5a', 'tbxa',
            'myod1', 'myf5', 'myog', 'myf6',
            'neurog1', 'neurod1', 'neurod4',
            'gata1a', 'gata2a', 'gata3', 'gata4', 'gata5', 'gata6',
            'hand2', 'hoxb1b', 'hoxa2b', 'hoxb5a',
            'foxa1', 'foxa2', 'foxa3',
            'nkx2.1a', 'nkx2.5', 'nkx6.1', 'nkx6.2',
            'olig2', 'ascl1a', 'msgn1', 'meox1', 'tcf21'
        }

    # Get predicted TF-gene pairs from mesh
    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = set()
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.add((tf, gene))

    # Get all timepoints for this celltype
    timepoints = sorted([tp for (ct, tp) in grn_dict.keys() if ct == celltype])

    if len(timepoints) < 2:
        return None  # Need at least 2 timepoints for dynamics

    # Extract subGRNs for each timepoint
    subgrns_by_timepoint = {}
    nodes_by_timepoint = {}
    edges_by_timepoint = {}
    tfs_by_timepoint = {}
    dev_tfs_by_timepoint = {}

    for timepoint in timepoints:
        if (celltype, timepoint) not in grn_dict:
            continue

        grn_df = grn_dict[(celltype, timepoint)]

        # Find intersection with predicted pairs
        grn_pairs = set(zip(grn_df['source'], grn_df['target']))
        found_pairs = predicted_pairs & grn_pairs

        # Extract matching edges
        mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
        subgrn = grn_df[mask].copy()

        subgrns_by_timepoint[timepoint] = subgrn

        if len(subgrn) > 0:
            # Collect nodes, edges, TFs
            nodes = set(subgrn['source']) | set(subgrn['target'])
            edges = set(zip(subgrn['source'], subgrn['target']))
            tfs = set(subgrn['source'])
            dev_tfs = tfs & developmental_tfs

            nodes_by_timepoint[timepoint] = nodes
            edges_by_timepoint[timepoint] = edges
            tfs_by_timepoint[timepoint] = tfs
            dev_tfs_by_timepoint[timepoint] = dev_tfs
        else:
            nodes_by_timepoint[timepoint] = set()
            edges_by_timepoint[timepoint] = set()
            tfs_by_timepoint[timepoint] = set()
            dev_tfs_by_timepoint[timepoint] = set()

    # Compute turnover metrics
    all_nodes = set.union(*nodes_by_timepoint.values()) if nodes_by_timepoint else set()
    all_edges = set.union(*edges_by_timepoint.values()) if edges_by_timepoint else set()
    all_tfs = set.union(*tfs_by_timepoint.values()) if tfs_by_timepoint else set()
    all_dev_tfs = set.union(*dev_tfs_by_timepoint.values()) if dev_tfs_by_timepoint else set()

    if len(all_nodes) == 0:
        return None  # No signal at all

    # Node turnover: how many nodes are NOT present in all timepoints?
    nodes_present_count = {node: sum(1 for tp_nodes in nodes_by_timepoint.values()
                                     if node in tp_nodes)
                           for node in all_nodes}
    dynamic_nodes = sum(1 for count in nodes_present_count.values()
                        if count < len(timepoints))
    node_turnover_rate = dynamic_nodes / len(all_nodes) if len(all_nodes) > 0 else 0

    # Edge turnover
    edges_present_count = {edge: sum(1 for tp_edges in edges_by_timepoint.values()
                                     if edge in tp_edges)
                           for edge in all_edges}
    dynamic_edges = sum(1 for count in edges_present_count.values()
                        if count < len(timepoints))
    edge_turnover_rate = dynamic_edges / len(all_edges) if len(all_edges) > 0 else 0

    # TF turnover (focus on source nodes)
    tfs_present_count = {tf: sum(1 for tp_tfs in tfs_by_timepoint.values()
                                 if tf in tp_tfs)
                         for tf in all_tfs}
    dynamic_tfs = sum(1 for count in tfs_present_count.values()
                      if count < len(timepoints))
    tf_turnover_rate = dynamic_tfs / len(all_tfs) if len(all_tfs) > 0 else 0

    # Developmental TF dynamics (HIGH PRIORITY)
    dev_tfs_present_count = {tf: sum(1 for tp_dev_tfs in dev_tfs_by_timepoint.values()
                                     if tf in tp_dev_tfs)
                             for tf in all_dev_tfs}
    dynamic_dev_tfs = sum(1 for count in dev_tfs_present_count.values()
                          if count < len(timepoints))
    dev_tf_turnover_rate = (dynamic_dev_tfs / len(all_dev_tfs)
                           if len(all_dev_tfs) > 0 else 0)

    # Temporal variance in edge counts
    edge_counts = [len(edges) for edges in edges_by_timepoint.values()]
    temporal_variance = np.std(edge_counts) / np.mean(edge_counts) if np.mean(edge_counts) > 0 else 0

    # Composite dynamics score (weighted average)
    # Prioritize: developmental TF dynamics > edge turnover > TF turnover > temporal variance
    dynamics_score = (
        0.4 * dev_tf_turnover_rate +  # Highest weight: known dev TFs changing
        0.3 * edge_turnover_rate +      # Edges changing
        0.2 * tf_turnover_rate +        # TFs changing
        0.1 * temporal_variance         # Variance in activity
    )

    return {
        'cluster_id': cluster_id,
        'celltype': celltype,
        'n_timepoints': len(timepoints),
        'n_total_nodes': len(all_nodes),
        'n_total_edges': len(all_edges),
        'n_total_tfs': len(all_tfs),
        'n_developmental_tfs': len(all_dev_tfs),
        'developmental_tfs_list': sorted(list(all_dev_tfs)),
        'node_turnover_rate': node_turnover_rate,
        'edge_turnover_rate': edge_turnover_rate,
        'tf_turnover_rate': tf_turnover_rate,
        'dev_tf_turnover_rate': dev_tf_turnover_rate,
        'temporal_variance': temporal_variance,
        'dynamics_score': dynamics_score,
        'subgrns_by_timepoint': subgrns_by_timepoint,
        'timepoints': timepoints
    }

# %%
from tqdm import tqdm

def rank_clusters_by_temporal_dynamics(df_clusters_groups, grn_dict,
                                       cluster_tf_gene_matrices,
                                       min_accessibility=0.01,
                                       min_timepoints=3,
                                       output_csv="cluster_ranking_temporal_dynamics.csv"):
    """
    Full pipeline: rank all 402 clusters by temporal dynamics.

    Parameters:
    -----------
    df_clusters_groups : pd.DataFrame
        Cluster × pseudobulk accessibility matrix
    grn_dict : dict
        {(celltype, timepoint): grn_df}
    cluster_tf_gene_matrices : dict
        {cluster_id: TF×gene binary matrix}
    min_accessibility : float
        Minimum accessibility threshold
    min_timepoints : int
        Minimum timepoints required for dynamics analysis
    output_csv : str
        Path to save ranking table

    Returns:
    --------
    pd.DataFrame : Ranked clusters with all metrics
    """

    print(f"Starting analysis of {len(cluster_tf_gene_matrices)} clusters...")

    results = []

    for cluster_id in tqdm(cluster_tf_gene_matrices.keys(), desc="Processing clusters"):
        # Step 1: Find most accessible celltype
        access_info = find_most_accessible_celltype(
            cluster_id, df_clusters_groups, min_accessibility
        )

        if access_info is None:
            continue  # Skip clusters with no signal

        celltype = access_info['celltype']

        # Step 2: Compute temporal dynamics for this celltype
        dynamics_info = compute_temporal_dynamics_score(
            cluster_id, celltype, grn_dict, cluster_tf_gene_matrices
        )

        if dynamics_info is None:
            continue  # Skip if no dynamics (e.g., <2 timepoints)

        if dynamics_info['n_timepoints'] < min_timepoints:
            continue  # Skip if too few timepoints

        # Combine results
        result = {**access_info, **dynamics_info}
        results.append(result)

    # Create DataFrame
    df_ranked = pd.DataFrame(results)

    # Sort by dynamics score (highest = most dynamic)
    df_ranked = df_ranked.sort_values('dynamics_score', ascending=False)

    # Save to CSV
    # Don't save the subgrns_by_timepoint dict (too large)
    df_ranked_export = df_ranked.drop(columns=['subgrns_by_timepoint', 'top_5_groups'], errors='ignore')
    df_ranked_export.to_csv(output_csv, index=False)
    print(f"\nRanking table saved to: {output_csv}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"RANKING SUMMARY")
    print(f"{'='*60}")
    print(f"Total clusters analyzed: {len(df_ranked)}")
    print(f"Dynamics score range: {df_ranked['dynamics_score'].min():.3f} - {df_ranked['dynamics_score'].max():.3f}")
    print(f"Median dynamics score: {df_ranked['dynamics_score'].median():.3f}")
    print(f"\nTop 10 most dynamic clusters:")
    print(df_ranked[['cluster_id', 'celltype', 'dynamics_score',
                     'n_developmental_tfs', 'developmental_tfs_list']].head(10).to_string(index=False))

    return df_ranked

# %%
def visualize_top_dynamic_clusters(df_ranked, grn_dict, cluster_tf_gene_matrices,
                                   top_n=10, figpath="./"):
    """
    Generate NetworkX visualizations for top N most dynamic clusters.

    Parameters:
    -----------
    df_ranked : pd.DataFrame
        Output from rank_clusters_by_temporal_dynamics()
    grn_dict : dict
        {(celltype, timepoint): grn_df}
    cluster_tf_gene_matrices : dict
        {cluster_id: TF×gene binary matrix}
    top_n : int
        Number of top clusters to visualize
    figpath : str
        Directory to save figures

    Returns:
    --------
    None (saves figures to disk)
    """

    import os
    os.makedirs(figpath, exist_ok=True)

    for idx, row in df_ranked.head(top_n).iterrows():
        cluster_id = row['cluster_id']
        celltype = row['celltype']
        dynamics_score = row['dynamics_score']
        dev_tfs = row['developmental_tfs_list']

        print(f"\n{'='*60}")
        print(f"Visualizing rank #{idx+1}: cluster {cluster_id} in {celltype}")
        print(f"Dynamics score: {dynamics_score:.3f}")
        print(f"Developmental TFs: {', '.join(dev_tfs)}")
        print(f"{'='*60}")

        # Get predicted pairs
        cluster_matrix = cluster_tf_gene_matrices[cluster_id]
        predicted_pairs = []
        for tf in cluster_matrix.index:
            for gene in cluster_matrix.columns:
                if cluster_matrix.loc[tf, gene] == 1:
                    predicted_pairs.append((tf, gene))

        # Call your existing visualization function
        try:
            subgrns, master_grn, master_pos = plot_subgrns_over_time(
                grn_dict=grn_dict,
                predicted_pairs=predicted_pairs,
                cluster_id=cluster_id,
                celltype_of_interest=celltype,
                spring_k=1.8,
                layout_scale=1.8,
                max_labels=50,
                label_strategy="all_tfs_plus_dynamic",
                debug_labels=False,
                savefig=True,
                filename=f"{figpath}subGRN_rank{idx+1}_{cluster_id}_{celltype}.pdf",
                max_edge_width=2.0,
                figsize=(15, 10),
                node_size_scale=0.5
            )
        except Exception as e:
            print(f"Error visualizing {cluster_id}/{celltype}: {e}")
            continue

# %%
# Filter out pseudobulk groups with < 20 cells before ranking
excluded_groups = [
    'NMPs_30somites', 'epidermis_30somites', 'fast_muscle_0somites',
    'hatching_gland_30somites', 'muscle_30somites', 'neural_10somites',
    'optic_cup_0somites', 'primordial_germ_cells_0somites',
    'primordial_germ_cells_5somites', 'primordial_germ_cells_10somites',
    'primordial_germ_cells_15somites', 'primordial_germ_cells_20somites',
    'primordial_germ_cells_30somites', 'tail_bud_30somites'
]

# Create filtered version of df_clusters_groups
cols_to_drop = [col for col in excluded_groups if col in df_clusters_groups.columns]
df_clusters_groups_filtered = df_clusters_groups.drop(columns=cols_to_drop)
print(f"Filtered out {len(cols_to_drop)} low-cell groups (< 20 cells)")
print(f"Original shape: {df_clusters_groups.shape}, Filtered shape: {df_clusters_groups_filtered.shape}")

# %%
# Run the full ranking pipeline with filtered data
df_ranked = rank_clusters_by_temporal_dynamics(
    df_clusters_groups=df_clusters_groups_filtered,
    grn_dict=grn_dict,
    cluster_tf_gene_matrices=cluster_tf_gene_matrices,
    min_accessibility=0.01,
    min_timepoints=3,
    output_csv=figpath + "cluster_ranking_temporal_dynamics.csv"
)

# %%
# Inspect the top 20 candidates
print("\nTop 20 most dynamic clusters:")
print(df_ranked[['cluster_id', 'celltype', 'dynamics_score',
                 'n_developmental_tfs', 'dev_tf_turnover_rate',
                 'edge_turnover_rate', 'developmental_tfs_list']].head(20))

# %%
# Explore by celltype
print("\n=== Top clusters by celltype ===")
for celltype in df_ranked['celltype'].unique()[:10]:
    ct_clusters = df_ranked[df_ranked['celltype'] == celltype]
    if len(ct_clusters) > 0:
        print(f"\n{celltype}: {len(ct_clusters)} clusters")
        print(ct_clusters[['cluster_id', 'dynamics_score', 'n_developmental_tfs']].head(3).to_string(index=False))

# %%
# Visualize top 5 most dynamic clusters
visualize_top_dynamic_clusters(
    df_ranked=df_ranked,
    grn_dict=grn_dict,
    cluster_tf_gene_matrices=cluster_tf_gene_matrices,
    top_n=5,
    figpath=figpath
)

# %%

# %% [markdown]
# ## EDA on "linked genes" and where they are in the peak UMAP
#
# - 

# %%
# 1. Grab your UMAP coordinates and obs DataFrame:
umap_coords = adata_peaks.obsm["X_umap"]
obs = adata_peaks.obs

# 2. Make a scatterplot of *all* peaks in light gray (background):
plt.figure(figsize=(6,6))
plt.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c="lightgray",
    s=10,
    alpha=0.3,
    label="_background"  # leading underscore => won't clutter legend
)

# 3. Overlay each gene of interest in a different color:
genes_of_interest = ["msgn1","myf5","meox1","myog","myl1"]
colors = ["red","blue","green","purple","orange"]  # etc.
for gene, color in zip(genes_of_interest, colors):
    mask = (obs["linked_gene"] == gene)
    plt.scatter(
        umap_coords[mask, 0],
        umap_coords[mask, 1],
        c=color,
        s=40,
        label=gene
    )
    plt.grid(False)

plt.legend(markerscale=1.5)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("Peaks from select muscle/myotome genes")
# plt.savefig(figpath + "umap_peaks_mesoderm_subclust_pseudotime.png")
# plt.savefig(figpath + "umap_peaks_mesoderm_subclust_pseudotime.pdf")
plt.show()

# %%
# 1. Grab your UMAP coordinates and obs DataFrame:
umap_coords = adata_peaks.obsm["X_umap"]
obs = adata_peaks.obs

# 2. Make a scatterplot of *all* peaks in light gray (background):
plt.figure(figsize=(8,8))
plt.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c="lightgray",
    s=10,
    alpha=0.3,
    label="_background"  # leading underscore => won't clutter legend
)

# 3. Define gene sets and their corresponding color palettes:
gene_sets = {
    "Muscle/Mesoderm": ["msgn1","myf5","meox1","myog","myl1"],
    "Hematopoietic": ["gata1", "pu.1", "runx1", "cd41", "lyz", "rag1", "cd3e", "cd79a"],
    "Endoderm": ["sox17", "foxa2", "gsc", "cas", "mixer", "sox32", "hhex", "pdx1"],
    "Epidermis": ["krt4", "tp63", "dlx3b", "foxi1", "bmp2b", "gata3", "msx1b"],
    "Neural": ["sox2", "nestin", "neurod1", "elavl3", "gap43", "tubb3", "ascl1a", "olig2"]
}

# Distinct color palettes for each tissue type
color_palettes = {
    "Muscle/Mesoderm": ["#d62728", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"],  # reds/oranges
    "Hematopoietic": ["#1f77b4", "#aec7e8", "#17becf", "#9edae5", "#393b79", "#5254a3", "#6b6ecf", "#9c9ede"],  # blues
    "Endoderm": ["#2ca02c", "#98df8a", "#2ca02c", "#98df8a", "#bcbd22", "#dbdb8d", "#8ca252", "#b5cf6b"],  # greens
    "Epidermis": ["#ff7f0e", "#ffbb78", "#d62728", "#ff9896", "#ff7f0e", "#ffbb78", "#d62728"],  # oranges/reds
    "Neural": ["#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d3", "#7f7f7f", "#c7c7c7"]  # purples/browns
}

# 4. Plot each gene set with its distinct colors:
for tissue_type, genes in gene_sets.items():
    colors = color_palettes[tissue_type]
    for gene, color in zip(genes, colors):
        mask = (obs["linked_gene"] == gene)
        if mask.sum() > 0:  # only plot if gene has peaks
            plt.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=color,
                s=40,
                label=f"{gene} ({tissue_type})"
            )

plt.grid(False)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=1.5)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("Peaks from tissue-specific marker genes")
plt.tight_layout()
# plt.savefig(figpath + "umap_peaks_all_tissues.png", bbox_inches='tight', dpi=300)
# plt.savefig(figpath + "umap_peaks_all_tissues.pdf", bbox_inches='tight')
plt.show()

# %%

# %%

# %% [markdown]
# ## EDA on some coarse clusters with strong celltype/timepoint signals
#
# - clusters 

# %%
adata_peaks[adata_peaks.obs["leiden_unified"]=="26_8"]

# %%
# Create boolean mask
adata_peaks.obs['specific_cluster'] = adata_peaks.obs['leiden_unified'] == '26_8'

# Plot with custom colors
sc.pl.umap(adata_peaks, color='specific_cluster', palette={'True': 'red', 'False': 'lightgrey'}, save="_cluster_26_8.png")

# %%
list_clusters = ["12_13","29_19","26_6","0_0","19_13","2_4",""]

# Create boolean mask
adata_peaks.obs['specific_cluster'] = adata_peaks.obs['leiden_unified'] == '35_9'

# Plot with custom colors
sc.pl.umap(adata_peaks, color='specific_cluster', palette={'True': 'red', 'False': 'lightgrey'})

# %%
# Create boolean mask
adata_peaks.obs['specific_cluster'] = adata_peaks.obs['leiden_coarse'] == 35

# Plot with custom colors
sc.pl.umap(adata_peaks, color='specific_cluster', palette={'True': 'red', 'False': 'lightgrey'})
