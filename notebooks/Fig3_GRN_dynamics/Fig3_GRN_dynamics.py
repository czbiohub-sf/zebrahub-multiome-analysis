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
#     display_name: celloracle_env
#     language: python
#     name: celloracle_env
# ---

# %% [markdown]
# # EDA on GRN temporal dynamics
#
# - Last updated: 10/4/2024
# - Author: Yang-Joon Kim
#
# Description/notes:
# - Exploratry data analysis on GRNs from different timepoints/pseudotimes.
#
# - Analyses where we'd like to see how the GRN evolves over time/development.
#     - 1) for a trajectory of cell-types, how does the GRN evolves during the differentiation (pseudotime axis) - focus on the NMP trajectories.
#     
#     - 2) for the same cell-type (progenitor, or in intermediate fate), how does the GRN evolves over the developmental stages (real-time).
#
#     - From these analyses, can we learn a transient key driver genes/TFs that were unidentifiable from "static" GRNs?
#
# - Changes in v3 compared to v2: 
#      - we're using 2000 edges for each GRN (time,celltype), which is the default in CellOracle. Note that the v2 used 50 edges to emphasize the strongest connections within the GRNs (network).

# %%
# 0. Import
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import scipy.sparse as sp
from itertools import combinations

# %%
import celloracle as co
co.__version__

# %%
# visualization settings
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 600

# %%
import logging

# Set the logging level to WARN, filtering out informational messages
logging.getLogger().setLevel(logging.WARNING)

import matplotlib as mpl

# Import project-specific utilities
from scripts.fig2_utils.plotting_utils import set_plotting_style

mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Set the default font to Arial
mpl.rcParams['font.family'] = 'Arial'

# If Arial is not available on your system, you might need to specify an alternative or ensure Arial is installed.
# On some systems, you might need to use 'font.sans-serif' as a fallback option:
# mpl.rcParams['font.sans-serif'] = 'Arial'

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number for editable text in PDFs
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')

# Plotting style function (run this before plotting the final figure)
# %%
set_plotting_style()

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/EDA_GRN_dynamics_timepoints_v3/"
os.makedirs(figpath, exist_ok=True)

# %% [markdown]
# ## Step 0. Import the GRNs (Links object)

# %%
# We're using "TDR118" as the representative for "15-somites", and drop the "TDR119" for now.
# We'll use the "TDR119" for benchmark/comparison of biological replicates later on.
list_files = ['TDR126', 'TDR127', 'TDR128',
              'TDR118', 'TDR125', 'TDR124']

# %%
# # define the master directory for all Links objects (GRN objects from CellOracle)
# oracle_base_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"

# # extract the dataset names
# list_files = os.listdir(oracle_base_dir)
# list_files = [x for x in list_files if (not x.endswith("h5ad") and x.startswith("TDR"))]
# list_files

# %%
# import all adata objects and save as a dictionary
dict_adata = {}

for dataset in list_files:
    adata = sc.read_h5ad(oracle_base_dir + f"{dataset}_nmps_manual_annotation.h5ad")
    dict_adata[dataset] = adata
    
dict_adata

# %%
# define an empty dictionary
dict_links = {}

# for loop to import all Links objects
for dataset in list_files:
    file_name = f"{dataset}/08_{dataset}_celltype_GRNs.celloracle.links"
    file_path = os.path.join(oracle_base_dir, file_name)
    dict_links[dataset] = co.load_hdf5(file_path)
    
    print("importing ", dataset)
    
dict_links

# %% [markdown]
# ## Step 1. Further filtering of weak edges within the GRNs
# - By default, we keep 2000 edges for each GRN [celltype, time]. 
# - We'd like to filter out the weak edges by (1) edge strength, and (2) p-values

# %%
# define a new dict to save the "pruned" links
n_edges = 2000

# define an empty dict
dict_links_pruned = {}

for dataset in dict_links.keys():
    # filter for n_edges
    links = dict_links[dataset]
    links.filter_links(thread_number=n_edges)
    dict_links_pruned[dataset] = links
    
dict_links_pruned

# %%
# import the filtered_links from each GRN, and save them into another dictionary
dict_filtered_GRNs = {}

# for loop to import all filtered_GRN dataframes
for dataset in dict_links_pruned.keys():
    # extract the filtered links
    dict_filtered_GRNs[dataset] = dict_links_pruned[dataset].filtered_links
    
    print("importing filtered GRNs", dataset)
    
# dict_filtered_GRNs

# %%
# import the merged_scores from each GRN, and save them into another dictionary
dict_merged_score = {}

# for loop to import all filtered_GRN dataframes
for dataset in dict_links_pruned.keys():
    # extract the filtered links
    dict_merged_score[dataset] = dict_links_pruned[dataset].merged_score
    
    print("importing ", dataset)
    
# dict_merged_score

# %% [markdown]
# ### NOTES:
#
# - For any testing for n_edges, we'll have to re-run the above 3-cells.
#

# %% [markdown]
# ### NOTES: we have imported three dictionaries (nested with the dataset as the primary key, and the celltype as the secondary key).
#
# - dict_links (all CellOracle objects, called Links)
# - dict_filtered_GRNs (all filterd GRNs, 2000 edges per celltype, for all timepoints)
# - dict_merged_score (all network topology metrics from the filtered GRN above)
#

# %%
dict_filtered_GRNs["TDR118"].keys()

# %%
dict_merged_score["TDR118"].head()

# %%
dict_merged_score["TDR118"][dict_merged_score["TDR118"].cluster=="NMPs"].sort_values("degree_all", ascending=False)

# %% [markdown]
# ## Step 1. Visualize the GRNs using heatmap 
#
# (note that there will be lots of zeros for the source:target pair that is absent from the GRN)

# %%
df = dict_filtered_GRNs["TDR126"]["NMPs"]

df_counts = df.pivot(index="target", columns="source", values="coef_mean").fillna(0)

# mtx = df_counts.to_numpy()

# sparse_mtx = sp.csr_matrix(mtx)


# %%
df_counts.index
df_counts.columns

# %%
# Create a heatmap with hierarchical clustering
plt.figure(figsize=(12, 8))
g = sns.clustermap(df_counts, method='average', metric='euclidean', cmap='RdBu_r', standard_scale=None,
               row_cluster=True, col_cluster=True, yticklabels=False, vmax=0.2, vmin=-0.2)

# reordered_labels = df_counts.index[g.dendrogram_row.reordered_ind].tolist()
# use_labels = ["myf5","sox2","tbxta",
#               "meox1","tbx16","hes6",
#               "hbbe1.1","hbae3","hbbe3","rps16",
#               "lrrc24","krt96"]
# use_ticks = [reordered_labels.index(label) + .5 for label in use_labels]
# g.ax_heatmap.set(yticks=use_ticks, yticklabels=use_labels)
# g.savefig(figpath + "clustered_heatmap_GRN_test.png")
# g.savefig(figpath + "clustered_heatmap_GRN_test.pdf")
plt.show()

# %%
# compute the union of dataframes (GRNs) for the first and last timepoints only (this reduces noisy visualization that are only transient in intermediate timepoints)
# For loop to plot the GRNs over timepoints
# Choose one celltype
celltype = "PSM"

# for all the rest of the timepoints (0-30 somites)
timepoints = ["TDR126", "TDR127", "TDR128", "TDR118", "TDR125", "TDR124"]

# Step 1. collect all sources and targets across all timepoints
all_sources = set()
all_targets = set()

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    all_sources.update(df['source'].unique())
    all_targets.update(df['target'].unique())
    
# Step 2: Recreate each df_counts DataFrame
df_counts_union = {}

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    # Pivot the original DataFrame
    df_pivot = df.pivot(index='target', columns='source', values='coef_mean').reindex(index=all_targets, columns=all_sources).fillna(0)
    df_counts_union[timepoint] = df_pivot
    
# Assuming df_counts_union is your dictionary of adjusted count matrices
timepoints = list(df_counts_union.keys())

# %%
# Choose one celltype
celltype = "PSM"

# extract the celltype specific GRN at one timepoint
df_sample = dict_merged_score["TDR118"][dict_merged_score["TDR118"].cluster==celltype]

df_sample.sort_values("degree_centrality_all", ascending=False)

# %%
list_top_genes = df_sample.sort_values("degree_centrality_all", ascending=False).index[0:30]
list_top_genes

# %%
list_top_sources = df_sample.sort_values("degree_centrality_out", ascending=False).index[0:30]
list_top_sources

# %%
list_top_targets = df_sample.sort_values("degree_centrality_in", ascending=False).index[0:30]
list_top_targets

# %%
# first timepoint (0 somite/budstage)
df_counts = df_counts_union["TDR126"]

g1 = sns.clustermap(df_counts, method='average', metric='euclidean', cmap='RdBu_r', standard_scale=None,
                row_cluster=True, col_cluster=True, 
                xticklabels=False, yticklabels=False, 
                figsize=(10, 10),  # Adjust the figsize to fit your data
                vmax=0.1, vmin=-0.1)

# Access the axes for annotation
heatmap_ax = g1.ax_heatmap

# Determine the size of the heatmap for proper annotation placement
heatmap_size = heatmap_ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())

plt.show()

# %% [markdown]
# ## Step 1. generate the GRN heatmaps for all timepoints (for one celltype)
#
# - here, we will pick a couple of celltypes as an example. Note that we want to take a closer look at some of the GRN modules later (if they are conserved in other species or not) 
#
# - PSM (NOTE that the PSM GRN at 30-somites (TDR124) is very weak, likely because the number of PSM cells at 30-somites stage is very small...on the order of 10-20 cells.
# - spinal_cord (a good number of cells throughout the developmental stages
#
#
# ### NOTE:
# - we'll concatenate the df_counts from all timepoints for the "clustering" for rows and cols, respectively.
#
#

# %% [markdown]
# ### Fine-tuning the GRN visualization (using the linkages from concatenated GRNs along the timepoints)

# %%
# For loop to plot the GRNs over timepoints
# Choose one celltype
celltype = "PSM"

# for all the rest of the timepoints (0-30 somites)
timepoints = ["TDR126", "TDR127", "TDR128", "TDR118", "TDR125", "TDR124"]

# Step 1. collect all sources and targets across all timepoints
all_sources = set()
all_targets = set()

# for timepoint in timepoints:
#     df = dict_filtered_GRNs[timepoint][celltype]
#     all_sources.update(df['source'].unique())
#     all_targets.update(df['target'].unique())

# union for only the first and the last two timepoints (to reduce the noisy nodes)
for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    all_sources.update(df['source'].unique())
    all_targets.update(df['target'].unique())

    
# Step 2: Recreate each df_counts DataFrame
df_counts_union = {}

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    # Pivot the original DataFrame
    df_pivot = df.pivot(index='target', columns='source', values='coef_mean').reindex(index=all_targets, columns=all_sources).fillna(0)
    df_counts_union[timepoint] = df_pivot
    
# Assuming df_counts_union is your dictionary of adjusted count matrices
timepoints = list(df_counts_union.keys())

# %% [markdown]
# ### GRN visualization for PSM (over time)

# %%
# compute the linkages from the first and the last timepoints, by augmenting the "time" components
df_counts1 = df_counts_union["TDR126"]
df_counts2 = df_counts_union["TDR127"]
df_counts3 = df_counts_union["TDR128"]
df_counts4 = df_counts_union["TDR118"]
df_counts5 = df_counts_union["TDR125"]
df_counts6 = df_counts_union["TDR124"]

df = pd.concat([df_counts1, df_counts2, df_counts3,
                df_counts4, df_counts5, df_counts6], axis=1)
df

# %%
# Check the coef_abs (GRN edge strength_ distribution for GRN[PSM, 30-somites]
df_GRN_test = dict_filtered_GRNs["TDR124"]["PSM"]

plt.scatter(df_GRN_test["coef_mean"], df_GRN_test["-logp"])
plt.show()

# %% [markdown]
# ### NOTE: above scatter plot shows that the GRN [PSM, 30somites] is very weak. This is likely because the number of PSM cells at 30-somites tage is very few.

# %% [markdown]
# ### generate heatmap visualization for GRNs (PSM)

# %%
# For loop to plot the GRNs over timepoints
# Choose one celltype
celltype = "PSM"

# for all the rest of the timepoints (0-30 somites)
timepoints = ["TDR126", "TDR127", "TDR128", "TDR118", "TDR125", "TDR124"]
# define the dev stages corresponding to the timepoints list (for dataset IDs)
stages = ["0somites", "5somites", "10somites","15somites","20somites","30somites"]
stages

# Step 1. collect all sources and targets across all timepoints
all_sources = set()
all_targets = set()

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    all_sources.update(df['source'].unique())
    all_targets.update(df['target'].unique())
    
# Step 2: Recreate each df_counts DataFrame
df_counts_union = {}

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    # Pivot the original DataFrame
    df_pivot = df.pivot(index='target', columns='source', values='coef_mean').reindex(index=all_targets, columns=all_sources).fillna(0)
    df_counts_union[timepoint] = df_pivot
    
# # Assuming df_counts_union is your dictionary of adjusted count matrices
# timepoints = list(df_counts_union.keys())

# based on the histogram above, we'll define the vmax/vmin for color scale
# vmax = 0.15
# vmin = -0.15
vmax = 0.1
vmin = -0.1


# compute the linkages from the first and the last timepoints, by augmenting the "time" components
df_counts1 = df_counts_union["TDR126"]
df_counts2 = df_counts_union["TDR127"]
df_counts3 = df_counts_union["TDR128"]
df_counts4 = df_counts_union["TDR118"]
df_counts5 = df_counts_union["TDR125"]
df_counts6 = df_counts_union["TDR124"]

# concatenate over the columns
df_counts_rows = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=1)

# concatenate over the rows
df_counts_cols = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=0)

# create a clustered heatmap for the "rows"
g1 = sns.clustermap(df_counts_rows, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# create a clustered heatmap for the "cols"
g2 = sns.clustermap(df_counts_cols, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# extract the row/col indices
row_linkage = g1.dendrogram_row.linkage
col_linkage = g2.dendrogram_col.linkage

# Loop over all timepoints (using the pre-computed linkages from the first timepoint)
for idx, timepoint in enumerate(timepoints):
    # extract the df_counts at corresponding timepoint
    df_counts = df_counts_union[timepoint]
    
    # plot the clustermap
    g = sns.clustermap(df_counts, method='ward', metric='euclidean', 
                       cmap='coolwarm', standard_scale=None, 
                       row_cluster=True, col_cluster=True, 
                       xticklabels=False, yticklabels=False, 
                       vmax=vmax, vmin=vmin, 
                       row_linkage=row_linkage, col_linkage=col_linkage)
    
    # hide the dendrograms
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    
    # extract the dev stage
    stage=stages[idx]
    # save the plot
    # g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_coolwarm.pdf") # celltype is defined above where we computed df_counts_union
    # g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_coolwarm.png")

# # Display all heatmaps side by side with consistent clustering
plt.show()

# %% [markdown]
# ### Sectioning the left-top for "decreasing over time"

# %%
df_counts_union.keys()

# %%
df_counts_union["TDR126"]

# %%
# Check the GRN at the first timepoint
df_counts = df_counts_union["TDR126"]

# Assuming you have determined your cluster size and its position (last N rows and M columns)
N = 50  # The number of rows in the cluster, replace with the actual number
M = 5  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g.dendrogram_row.reordered_ind[0:N]
col_indices = g.dendrogram_col.reordered_ind[0:M]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union["TDR126"]

df_counts_0somites_upreg = df_counts.loc[row_names, col_names]
df_counts_0somites_upreg

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_0somites_upreg, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "PSM"
stage = "0somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}.png")

plt.show()

# %% [markdown]
# ### Spinal_cord (as an alternative to PSM)

# %%
# Check the coef_abs (GRN edge strength_ distribution for GRN[ct, 30-somites]
df_GRN_test = dict_filtered_GRNs["TDR127"]["spinal_cord"]

plt.scatter(df_GRN_test["coef_mean"], df_GRN_test["-logp"])
plt.grid(False)
plt.show()

# %% [markdown]
# From here, we'll use the 0.1 and -0.1 as v_max and v_min.

# %%
# For loop to plot the GRNs over timepoints
# Choose one celltype
celltype = "spinal_cord"

# for all the rest of the timepoints (0-30 somites)
timepoints = ["TDR126", "TDR127", "TDR128", "TDR118", "TDR125", "TDR124"]
# define the dev stages corresponding to the timepoints list (for dataset IDs)
stages = ["0somites", "5somites", "10somites","15somites","20somites","30somites"]
stages

# Step 1. collect all sources and targets across all timepoints
all_sources = set()
all_targets = set()

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    all_sources.update(df['source'].unique())
    all_targets.update(df['target'].unique())
    
# Step 2: Recreate each df_counts DataFrame
df_counts_union = {}

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    # Pivot the original DataFrame
    df_pivot = df.pivot(index='target', columns='source', values='coef_mean').reindex(index=all_targets, columns=all_sources).fillna(0)
    df_counts_union[timepoint] = df_pivot
    
# # Assuming df_counts_union is your dictionary of adjusted count matrices
# timepoints = list(df_counts_union.keys())

# based on the histogram above, we'll define the vmax/vmin for color scale
# vmax = 0.15
# vmin = -0.15
vmax = 0.1
vmin = -0.1


# compute the linkages from the first and the last timepoints, by augmenting the "time" components
df_counts1 = df_counts_union["TDR126"]
df_counts2 = df_counts_union["TDR127"]
df_counts3 = df_counts_union["TDR128"]
df_counts4 = df_counts_union["TDR118"]
df_counts5 = df_counts_union["TDR125"]
df_counts6 = df_counts_union["TDR124"]

# concatenate over the columns
df_counts_rows = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=1)

# concatenate over the rows
df_counts_cols = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=0)

# create a clustered heatmap for the "rows"
g1 = sns.clustermap(df_counts_rows, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# create a clustered heatmap for the "cols"
g2 = sns.clustermap(df_counts_cols, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# extract the row/col indices
row_linkage = g1.dendrogram_row.linkage
col_linkage = g2.dendrogram_col.linkage

# Loop over all timepoints (using the pre-computed linkages from the first timepoint)
for idx, timepoint in enumerate(timepoints):
    # extract the df_counts at corresponding timepoint
    df_counts = df_counts_union[timepoint]
    
    # plot the clustermap
    g = sns.clustermap(df_counts, method='ward', metric='euclidean', 
                       cmap='coolwarm', standard_scale=None, 
                       row_cluster=True, col_cluster=True, 
                       xticklabels=False, yticklabels=False, 
                       vmax=vmax, vmin=vmin, 
                       row_linkage=row_linkage, col_linkage=col_linkage)
    
    # hide the dendrograms
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    
    # extract the dev stage
    stage=stages[idx]
    # save the plot
    # g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_coolwarm.pdf") # celltype is defined above where we computed df_counts_union
    # g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_coolwarm.png")

# # Display all heatmaps side by side with consistent clustering
plt.show()

# %% [markdown]
# ### NOTES:
#
# - left top and right top sections "decrease" over developemental time (0->30 somites stages)
# - left bottom section "increases" over developemental time

# %% [markdown]
# ### Sectioning the left-top for "decreasing over time"

# %%
df_counts_union.keys()

# %%
df_counts_union["TDR126"]

# %%
# Check the GRN at the first timepoint
df_counts = df_counts_union["TDR126"]

# Assuming you have determined your cluster size and its position (last N rows and M columns)
N = 48  # The number of rows in the cluster, replace with the actual number
M = 5  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g.dendrogram_row.reordered_ind[0:N]
col_indices = g.dendrogram_col.reordered_ind[0:M]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union["TDR126"]

df_counts_0somites_upreg = df_counts.loc[row_names, col_names]
df_counts_0somites_upreg

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_0somites_upreg, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "spinal_cord"
stage = "0somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}.png")

plt.show()

# %% [markdown]
# ### Sectioning the right top (decreasing over time)

# %%
# Check the GRN at the first timepoint
df_counts = df_counts_union["TDR126"]

# Assuming you have determined your cluster size and its position (last N rows and M columns)
N = 50  # The number of rows in the cluster, replace with the actual number
M = -10  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g1.dendrogram_row.reordered_ind[20:80]
col_indices = g2.dendrogram_col.reordered_ind[-27:-15]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union["TDR126"]

df_counts_0somites_upreg2 = df_counts.loc[row_names, col_names]
df_counts_0somites_upreg2

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_0somites_upreg2, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "spinal_cord"
stage = "0somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_2.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_2.png")

plt.show()

# %% [markdown]
# ### Sectioning the left-bottom for "increasing over time"

# %%
df_counts = df_counts_union["TDR125"]

# %%
# Assuming you have determined your cluster size and its position (last N rows and M columns)
N = 1  # The number of rows in the cluster, replace with the actual number
M = 14  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g1.dendrogram_row.reordered_ind[-49:-4]
col_indices = g2.dendrogram_col.reordered_ind[5:14]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union["TDR125"]

df_counts_20somites_upreg = df_counts.loc[row_names, col_names]
df_counts_20somites_upreg

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_20somites_upreg, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "spinal_cord"
stage = "20somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg.png")

plt.show()

# %% [markdown]
# ### other block of sub-GRN that is activated until 20-somites, and disappear after 30 somites

# %%
df_counts = df_counts_union["TDR125"]

# %%
# Assuming you have determined your cluster size and its position (last N rows and M columns)
N = 1  # The number of rows in the cluster, replace with the actual number
M = 14  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g1.dendrogram_row.reordered_ind[-67:-50]
col_indices = g2.dendrogram_col.reordered_ind[5:14]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union["TDR125"]

df_counts_20somites_upreg2 = df_counts.loc[row_names, col_names]
df_counts_20somites_upreg2

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_20somites_upreg2, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "spinal_cord"
stage = "20somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg2.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg2.png")

plt.show()

# %% [markdown]
# ### sum of the first and the second blocks
#

# %%
df_counts = df_counts_union["TDR125"]

# %%
# Assuming you have determined your cluster size and its position (last N rows and M columns)
N = 1  # The number of rows in the cluster, replace with the actual number
M = 14  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g1.dendrogram_row.reordered_ind[-67:-4]
col_indices = g2.dendrogram_col.reordered_ind[5:14]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union["TDR125"]

df_counts_20somites_upreg = df_counts.loc[row_names, col_names]
df_counts_20somites_upreg

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_20somites_upreg, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "spinal_cord"
stage = "20somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg_3.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg_3.png")

plt.show()

# %% [markdown]
# ### Conclusion - "spinal_cord" GRNs over time
#

# %%

# %% [markdown]
# ### [OLD] Conclusion - "PSM" GRNs over time
#
# - "hmga1a"'s regulation is down-regulated over time. "hmga" family of TFs are small non-histone proteins that can bind to DNA and modify chromatin states - likely accessilibilty of regulatory TFs to DNA (Vignali and Marracci, Int.J.Mol.Sci, 2020).
# - We looked at "up-regulated/activated" GRNs over time - 30-somites stage is qualitatively very different from the previous stages, so we characterized two sub-GRNs: (1) activated at 30-somites, and (2) activated at 20-somites stage.

# %%

# %% [markdown]
# ## Step2. Evolution of GRNs along the celltypes
#
# - there are several approaches to this. We can either pick one timepoint, check the GRN evolution along the NMP trajectories, 
# - or, we can see how this works over certain "modules" that we identified above (temporal dynamics)
#
#

# %%
# check the list of celltypes
dict_filtered_GRNs["TDR128"].keys()

# %%
# For loop to plot the GRNs over celltypes (dev trajectories)
# Choose one timepoint and define the "stage"
timepoint = "TDR118"
# define the dev stage
stage = "15somites"
# timepoints = ["TDR126", "TDR127", "TDR128", "TDR118reseq", "TDR125reseq", "TDR124reseq"]

# for all the rest of the timepoints (0-30 somites)
# mesoderm_lineages = ["neural_posterior", "spinal_cord",
#                      "NMPs","tail_bud","PSM","somites","fast_muscle"]
NMP_lineages = ["neural_posterior", "spinal_cord",
                     "NMPs","tail_bud","PSM","somites"]

# Step 1. collect all sources and targets across all celltypes
all_sources = set()
all_targets = set()

for celltype in NMP_lineages:
    df = dict_filtered_GRNs[timepoint][celltype]
    all_sources.update(df['source'].unique())
    all_targets.update(df['target'].unique())
    
# Step 2: Recreate each df_counts DataFrame
df_counts_union_NMP_lin = {}

for celltype in NMP_lineages:
    df = dict_filtered_GRNs[timepoint][celltype]
    # Pivot the original DataFrame
    df_pivot = df.pivot(index='target', columns='source', values='coef_mean').reindex(index=all_targets, columns=all_sources).fillna(0)
    df_counts_union_NMP_lin[celltype] = df_pivot
    
# list of all celltypes
list_celltypes = list(df_counts_union_NMP_lin.keys())

# based on the histogram above, we'll define the vmax/vmin for color scale
vmax = 0.1
vmin = -0.1

# compute the linkages from the first and the last timepoints, by augmenting the "time" components
df_counts1 = df_counts_union_NMP_lin["neural_posterior"]
df_counts2 = df_counts_union_NMP_lin["spinal_cord"]
df_counts3 = df_counts_union_NMP_lin["NMPs"]
df_counts4 = df_counts_union_NMP_lin["tail_bud"]
df_counts5 = df_counts_union_NMP_lin["PSM"]
df_counts6 = df_counts_union_NMP_lin["somites"]

# concatenate over the columns
df_counts_rows = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=1)

# concatenate over the rows
df_counts_cols = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=0)

# create a clustered heatmap for the "rows"
g1 = sns.clustermap(df_counts_rows, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# create a clustered heatmap for the "cols"
g2 = sns.clustermap(df_counts_cols, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# extract the row/col indices
row_linkage = g1.dendrogram_row.linkage
col_linkage = g2.dendrogram_col.linkage

# Loop over all timepoints (using the pre-computed linkages from the first timepoint)
for idx, celltype in enumerate(NMP_lineages):

    df_counts = df_counts_union_NMP_lin[celltype]
    
    # plot the clustermap
    g = sns.clustermap(df_counts, method='ward', metric='euclidean', 
                       cmap='coolwarm', standard_scale=None, 
                       row_cluster=True, col_cluster=True, 
                       xticklabels=False, yticklabels=False, 
                       vmax=vmax, vmin=vmin, 
                       row_linkage=row_linkage, col_linkage=col_linkage)
    
    # hide the dendrograms
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    # hide the colorbar
    g.cax.set_visible(False)
    
    # save the plot
    g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_NMPtraj.pdf") # celltype is defined above where we computed df_counts_union
    g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_NMPtraj.png")
# # Since sns.clustermap uses its own figure by default, we adjust our approach slightly:
# plt.close(g1.fig)  # Close the first plot's figure to prevent it from showing twice

# # Display all heatmaps side by side with consistent clustering
plt.show()

# %% [markdown]
# ### sectioning out the sub-GRN (up-regulated along the lineages)
#
# - neuronal trajectory
# - mesoderm trajectory

# %%
df_counts = df_counts_union_NMP_lin["neural_posterior"]

# Assuming you have determined your cluster size and its position (last N rows and M columns)
# N = 1  # The number of rows in the cluster, replace with the actual number
# M = 14  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g1.dendrogram_row.reordered_ind[22:50]
col_indices = g2.dendrogram_col.reordered_ind[-19:-9]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union_NMP_lin["neural_posterior"]

df_counts_neuronal_upreg = df_counts.loc[row_names, col_names]
df_counts_neuronal_upreg

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_neuronal_upreg, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "neuronal"
stage = "15somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg.png")

plt.show()

# %% [markdown]
# ### mesodermal cells

# %%
# Assuming you have determined your cluster size and its position (last N rows and M columns)
# N = 1  # The number of rows in the cluster, replace with the actual number
# M = 14  # The number of columns in the cluster, replace with the actual number

df_counts = df_counts_union_NMP_lin["somites"]

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g1.dendrogram_row.reordered_ind[0:22]
col_indices = g2.dendrogram_col.reordered_ind[-11:-1]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union_NMP_lin["somites"]

df_counts_mesodermal_upreg = df_counts.loc[row_names, col_names]
df_counts_mesodermal_upreg

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_mesodermal_upreg, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "mesodermal"
stage = "15somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}_upreg.png")

plt.show()

# %%

# %% [markdown]
# ## [SI] GRN evolution over celltypes (lineages) at 10hpf (the earliest stage)
#
#

# %%
# For loop to plot the GRNs over celltypes (dev trajectories)
# Choose one timepoint and define the "stage"
timepoint = "TDR126"
# define the dev stage
stage = "0somites"
# timepoints = ["TDR126", "TDR127", "TDR128", "TDR118reseq", "TDR125reseq", "TDR124reseq"]
NMP_lineages = ["neural_posterior", "spinal_cord",
                     "NMPs","tail_bud","PSM","somites"]

# Step 1. collect all sources and targets across all celltypes
all_sources = set()
all_targets = set()

for celltype in NMP_lineages:
    df = dict_filtered_GRNs[timepoint][celltype]
    all_sources.update(df['source'].unique())
    all_targets.update(df['target'].unique())
    
# Step 2: Recreate each df_counts DataFrame
df_counts_union_NMP_lin = {}

for celltype in NMP_lineages:
    df = dict_filtered_GRNs[timepoint][celltype]
    # Pivot the original DataFrame
    df_pivot = df.pivot(index='target', columns='source', values='coef_mean').reindex(index=all_targets, columns=all_sources).fillna(0)
    df_counts_union_NMP_lin[celltype] = df_pivot
    
# list of all celltypes
list_celltypes = list(df_counts_union_NMP_lin.keys())

# based on the histogram above, we'll define the vmax/vmin for color scale
vmax = 0.1
vmin = -0.1

# compute the linkages from the first and the last timepoints, by augmenting the "time" components
df_counts1 = df_counts_union_NMP_lin["neural_posterior"]
df_counts2 = df_counts_union_NMP_lin["spinal_cord"]
df_counts3 = df_counts_union_NMP_lin["NMPs"]
df_counts4 = df_counts_union_NMP_lin["tail_bud"]
df_counts5 = df_counts_union_NMP_lin["PSM"]
df_counts6 = df_counts_union_NMP_lin["somites"]

# concatenate over the columns
df_counts_rows = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=1)

# concatenate over the rows
df_counts_cols = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=0)

# create a clustered heatmap for the "rows"
g1 = sns.clustermap(df_counts_rows, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# create a clustered heatmap for the "cols"
g2 = sns.clustermap(df_counts_cols, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# extract the row/col indices
row_linkage = g1.dendrogram_row.linkage
col_linkage = g2.dendrogram_col.linkage

# Loop over all timepoints (using the pre-computed linkages from the first timepoint)
for idx, celltype in enumerate(NMP_lineages):

    df_counts = df_counts_union_NMP_lin[celltype]
    
    # plot the clustermap
    g = sns.clustermap(df_counts, method='ward', metric='euclidean', 
                       cmap='coolwarm', standard_scale=None, 
                       row_cluster=True, col_cluster=True, 
                       xticklabels=False, yticklabels=False, 
                       vmax=vmax, vmin=vmin, 
                       row_linkage=row_linkage, col_linkage=col_linkage)
    
    # hide the dendrograms
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    # hide the colorbar
    g.cax.set_visible(False)
    
    # save the plot
    g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_NMPtraj.pdf") # celltype is defined above where we computed df_counts_union
    g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_NMPtraj.png")
# # Since sns.clustermap uses its own figure by default, we adjust our approach slightly:
# plt.close(g1.fig)  # Close the first plot's figure to prevent it from showing twice

# # Display all heatmaps side by side with consistent clustering
plt.show()

# %% [markdown]
# ### Sectioning the left-top for "decreasing over time"

# %%
df_counts_union_NMP_lin.keys()

# %%
# Check the GRN at the first celltype
df_counts = df_counts_union_NMP_lin["neural_posterior"]

# Assuming you have determined your cluster size and its position (last N rows and M columns)
N = 48  # The number of rows in the cluster, replace with the actual number
M = 5  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g.dendrogram_row.reordered_ind[0:N]
col_indices = g.dendrogram_col.reordered_ind[0:M]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union_NMP_lin["neural_posterior"]

df_counts_sectioned = df_counts.loc[row_names, col_names]
df_counts_sectioned

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_sectioned, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "neural_posterior"
stage = "0somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}.png")

plt.show()

# %% [markdown]
# ## Step 3. visualizing the whole-embryo level GRN (computed from the whole-embryo at each time point)
#
#

# %%
oracle_base_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/11_celloracle_whole_embryo_GRNs/"

# %%
# # import all adata objects and save as a dictionary
# dict_adata = {}

# for dataset in list_files:
#     adata = sc.read_h5ad(oracle_base_dir + f"{dataset}_nmps_manual_annotation.h5ad")
#     dict_adata[dataset] = adata
    
# dict_adata

# %%
# define an empty dictionary
dict_links = {}

# for loop to import all Links objects
for dataset in list_files:
    file_name = f"{dataset}/08_{dataset}_celltype_GRNs.celloracle.links"
    file_path = os.path.join(oracle_base_dir, file_name)
    dict_links[dataset] = co.load_hdf5(file_path)
    
    print("importing ", dataset)
    
dict_links

# %% [markdown]
# ### Further filtering of weak edges within the GRNs
# - By default, we keep 2000 edges for each GRN [celltype, time]. 
# - We'd like to filter out the weak edges by (1) edge strength, and (2) p-values

# %%
# define a new dict to save the "pruned" links
n_edges = 2000

# define an empty dict
dict_links_pruned = {}

for dataset in dict_links.keys():
    # filter for n_edges
    links = dict_links[dataset]
    links.filter_links(thread_number=n_edges)
    dict_links_pruned[dataset] = links
    
dict_links_pruned

# %%
# import the filtered_links from each GRN, and save them into another dictionary
dict_filtered_GRNs = {}

# for loop to import all filtered_GRN dataframes
for dataset in dict_links_pruned.keys():
    # extract the filtered links
    dict_filtered_GRNs[dataset] = dict_links_pruned[dataset].filtered_links
    
    print("importing filtered GRNs", dataset)
    
# dict_filtered_GRNs

# %%
# import the merged_scores from each GRN, and save them into another dictionary
dict_merged_score = {}

# for loop to import all filtered_GRN dataframes
for dataset in dict_links_pruned.keys():
    # extract the filtered links
    dict_merged_score[dataset] = dict_links_pruned[dataset].merged_score
    
    print("importing ", dataset)
    
# dict_merged_score

# %%
dict_filtered_GRNs["TDR126"]["whole_embryo"]

# %%
df_counts1

# %%
# For loop to plot the GRNs over timepoints
# Choose one celltype
celltype = "whole_embryo"

# for all the rest of the timepoints (0-30 somites)
timepoints = ["TDR126", "TDR127", "TDR128", "TDR118", "TDR125", "TDR124"]
# define the dev stages corresponding to the timepoints list (for dataset IDs)
stages = ["0somites", "5somites", "10somites","15somites","20somites","30somites"]
stages

# Step 1. collect all sources and targets across all timepoints
all_sources = set()
all_targets = set()

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    all_sources.update(df['source'].unique())
    all_targets.update(df['target'].unique())
    
# Step 2: Recreate each df_counts DataFrame
df_counts_union = {}

for timepoint in timepoints:
    df = dict_filtered_GRNs[timepoint][celltype]
    # Pivot the original DataFrame
    df_pivot = df.pivot(index='target', columns='source', values='coef_mean').reindex(index=all_targets, columns=all_sources).fillna(0)
    df_counts_union[timepoint] = df_pivot
    
# # Assuming df_counts_union is your dictionary of adjusted count matrices
# timepoints = list(df_counts_union.keys())

# based on the histogram above, we'll define the vmax/vmin for color scale
# vmax = 0.15
# vmin = -0.15
vmax = 0.1
vmin = -0.1


# compute the linkages from the first and the last timepoints, by augmenting the "time" components
df_counts1 = df_counts_union["TDR126"]
df_counts2 = df_counts_union["TDR127"]
df_counts3 = df_counts_union["TDR128"]
df_counts4 = df_counts_union["TDR118"]
df_counts5 = df_counts_union["TDR125"]
df_counts6 = df_counts_union["TDR124"]

# concatenate over the columns
df_counts_rows = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=1)

# concatenate over the rows
df_counts_cols = pd.concat([df_counts1, df_counts2, df_counts3,
                            df_counts4, df_counts5, df_counts6], axis=0)

# create a clustered heatmap for the "rows"
g1 = sns.clustermap(df_counts_rows, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# create a clustered heatmap for the "cols"
g2 = sns.clustermap(df_counts_cols, method='ward', metric='euclidean', 
                    cmap='coolwarm', standard_scale=None,
                    row_cluster=True, col_cluster=True, 
                    xticklabels=False, yticklabels=False, 
                    vmax=vmax, vmin=vmin)

# extract the row/col indices
row_linkage = g1.dendrogram_row.linkage
col_linkage = g2.dendrogram_col.linkage

# Loop over all timepoints (using the pre-computed linkages from the first timepoint)
for idx, timepoint in enumerate(timepoints):
    # extract the df_counts at corresponding timepoint
    df_counts = df_counts_union[timepoint]
    
    # plot the clustermap
    g = sns.clustermap(df_counts, method='ward', metric='euclidean', 
                       cmap='coolwarm', standard_scale=None, 
                       row_cluster=True, col_cluster=True, 
                       xticklabels=False, yticklabels=False, 
                       vmax=vmax, vmin=vmin, 
                       row_linkage=row_linkage, col_linkage=col_linkage)
    
    # hide the dendrograms
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    
    # extract the dev stage
    stage=stages[idx]
    # save the plot
    g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_coolwarm.pdf") # celltype is defined above where we computed df_counts_union
    g.savefig(figpath + f"GRN_heatmap_{celltype}_{stage}_coolwarm.png")

# # Display all heatmaps side by side with consistent clustering
plt.show()

# %% [markdown]
# ### Sectioning the left-top for "decreasing over time"

# %%
df_counts_union.keys()

# %%
df_counts_union["TDR126"]

# %%
print(dict_merged_score["TDR126"].sort_values("degree_centrality_all", ascending=False).head(10).index)
print(dict_merged_score["TDR127"].sort_values("degree_centrality_all", ascending=False).head(10).index)
print(dict_merged_score["TDR128"].sort_values("degree_centrality_all", ascending=False).head(10).index)
print(dict_merged_score["TDR118"].sort_values("degree_centrality_all", ascending=False).head(10).index)
print(dict_merged_score["TDR125"].sort_values("degree_centrality_all", ascending=False).head(10).index)
print(dict_merged_score["TDR124"].sort_values("degree_centrality_all", ascending=False).head(10).index)

# %%
dict_merged_score["TDR126"].sort_values("degree_centrality_all", ascending=False).head()

# %%
df_merged_score_sorted = dict_merged_score["TDR126"].sort_values("degree_centrality_all", ascending=False)
df_merged_score_sorted

# %%
for timepoint in timepoints:
    df_merged_score_sorted = dict_merged_score[timepoint].sort_values("degree_centrality_all", ascending=False)
    plt.scatter(df_merged_score_sorted.head().degree_centrality_all, 
            df_merged_score_sorted.head().index)
    plt.gca().invert_yaxis()
    plt.xlabel('degree centrality')
    plt.ylabel('genes')
    plt.xlim([0.07, 0.28])
    plt.grid(False)
    plt.savefig(figpath + f"degree_centrality_top5_whole_embryo_GRN_{timepoint}.pdf")
    plt.show()

# %%
plt.scatter(df_merged_score_sorted.head(5).degree_centrality_all, 
            df_merged_score_sorted.head(5).index)
plt.gca().invert_yaxis()
plt.xlabel('Degree Centrality')
plt.ylabel('Genes')
plt.grid(False)
plt.savefig(figpath + f"")
plt.show()

# %%
# Check the GRN at the first timepoint
df_counts = df_counts_union["TDR126"]

# Assuming you have determined your cluster size and its position (last N rows and M columns)
N = 50  # The number of rows in the cluster, replace with the actual number
M = 5  # The number of columns in the cluster, replace with the actual number

# The indices of the rows and columns in the original data that make up the cluster
row_indices = g.dendrogram_row.reordered_ind[20:70]
col_indices = g.dendrogram_col.reordered_ind[10:30]

# Extract the data for this cluster
cluster_data = df_counts.iloc[row_indices, col_indices]

# Print or process the cluster_data as needed
print(cluster_data)

# If you want to look at the labels of the rows and columns in the cluster
row_labels = df_counts.index[row_indices]
col_labels = df_counts.columns[col_indices]

print("Row Labels in the Cluster:", row_labels)
print("Column Labels in the Cluster:", col_labels)

# %%
col_names = col_labels
row_names = row_labels

# subset the GRN for the portion that is upregulated in Muscle compared to NMPs
df_counts = df_counts_union["TDR126"]

df_counts_0somites_upreg = df_counts.loc[row_names, col_names]
df_counts_0somites_upreg

# %%
vmax = 0.1
vmin = -0.1

# plot the clustermap
g = sns.clustermap(df_counts_0somites_upreg, method='ward', metric='euclidean', 
                   cmap='coolwarm', standard_scale=None, 
                   row_cluster=False, col_cluster=False, 
                   xticklabels=True, yticklabels=True, 
                   vmax=vmax, vmin=vmin)

# hide the dendrograms
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(True)

celltype = "PSM"
stage = "0somites"

g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}.pdf") # celltype is defined above where we computed df_counts_union
g.savefig(figpath + f"subGRN_heatmap_{celltype}_{stage}.png")

plt.show()

# %%

# %%
