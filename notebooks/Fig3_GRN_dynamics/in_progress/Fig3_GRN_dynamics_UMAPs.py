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
# # EDA/visualization on GRN temporal dynamics
#
# - Last updated: 12/3/2024
# - Author: Yang-Joon Kim
#
# Description/notes:
# - Exploratry data analysis on GRNs from different timepoints/pseudotimes.
#
# - using different data visualizations for GRNs
#     - 1) UMAP (TFs-by-genes, and plotting UMAP)
#     - 2) networkX (using spring_layout)
#     - 3) Dictys

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

    # Override any previously set font settings to ensure Arial is used
    plt.rc('font', family='Arial')


# %%
set_plotting_style()

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/viz_GRN_dynamics/"
os.makedirs(figpath, exist_ok=True)

# %% [markdown]
# ## Step 0. Import the GRNs (Links object)

# %%
# define the master directory for all Links objects (GRN objects from CellOracle)
oracle_base_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"

# We're using "TDR118" as the representative for "15-somites", and drop the "TDR119" for now.
# We'll use the "TDR119" for benchmark/comparison of biological replicates later on.
list_files = ['TDR126', 'TDR127', 'TDR128',
              'TDR118', 'TDR125', 'TDR124']

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
celltype = "spinal_cord"

# for all the rest of the timepoints (0-30 somites)
timepoints = ["TDR126", "TDR127", "TDR128", "TDR118", "TDR125", "TDR124"]

# Step 1. collect all sources and targets across all timepoints
all_sources = set()
all_targets = set()

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
# ## GRN visualization for Spinal Cord (using UMAPs)

# %%
# compute the linkages from the first and the last timepoints, by augmenting the "time" components
df_counts1 = df_counts_union["TDR126"]
# df_counts2 = df_counts_union["TDR127"]
# df_counts3 = df_counts_union["TDR128"]
# df_counts4 = df_counts_union["TDR118"]
# df_counts5 = df_counts_union["TDR125"]
# df_counts6 = df_counts_union["TDR124"]

# df = pd.concat([df_counts1, df_counts2, df_counts3,
#                 df_counts4, df_counts5, df_counts6], axis=1)
# df

# %%
df_counts1

# %%
df_counts1.sum(axis=1).sort_values(ascending=False)[0:20]

# %%
df_counts1.sum(axis=0).sort_values(ascending=False)[0:20]

# %%

# %%

# %% [markdown]
# ## computing the GRN UMAP for each timepoint for one celltype
#
# - spinal_cord

# %%
df_counts1 = df_counts_union["TDR126"]
# df_counts2 = df_counts_union["TDR127"]
# df_counts3 = df_counts_union["TDR128"]
# df_counts4 = df_counts_union["TDR118"]
# df_counts5 = df_counts_union["TDR125"]
# df_counts6 = df_counts_union["TDR124"]

# create an adata object (TFs-by-target_genes)
adata_df1 = sc.AnnData(X=df_counts1.T.values) # transpose
adata_df1.obs_names = df_counts1.columns
adata_df1.var_names = df_counts1.index

adata_df1

# computing UMAP
# let's not normalize the counts, as the counts were already normalized at the whole GRN level
# sc.pp.scale(adata_df1)
sc.tl.pca(adata_df1)
sc.pl.pca(adata_df1)

sc.pp.neighbors(adata_df1, n_neighbors=10, n_pcs=30)

sc.tl.umap(adata_df1, min_dist=0.1)
sc.pl.umap(adata_df1)

sc.tl.leiden(adata_df1, resolution=0.1)
sc.pl.umap(adata_df1, color="leiden")

# %%
# computing UMAP
# let's not normalize the counts, as the counts were already normalized at the whole GRN level
# sc.pp.scale(adata_df1)
sc.tl.pca(adata_df1)
sc.pl.pca(adata_df1)

# %%
sc.pp.neighbors(adata_df1, n_neighbors=10, n_pcs=30)

# %%
sc.tl.umap(adata_df1, min_dist=0.1)
sc.pl.umap(adata_df1)

# %%
"hmga1a" in adata_df1.var_names

# %%
sc.pl.umap(adata_df1, color=["hmga1a","msgn1"])

# %%
sc.tl.leiden(adata_df1, resolution=0.1)
sc.pl.umap(adata_df1, color="leiden")

# %%
dict_spinal_cord = {}

for timepoint in timepoints:
    df_counts = df_counts_union[timepoint]
    
    # create an adata object (TFs-by-target_genes)
    adata = sc.AnnData(X=df_counts.T.values) # transpose
    adata.obs_names = df_counts.columns
    adata.var_names = df_counts.index

    print(adata)
    
    
    # computing UMAP
    # let's not normalize the counts, as the counts were already normalized at the whole GRN level
    # sc.pp.scale(adata_df1)
    sc.tl.pca(adata)
    sc.pl.pca(adata)

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

    sc.tl.umap(adata, min_dist=0.1)

    sc.tl.leiden(adata, resolution=0.1)
    sc.pl.umap(adata, color="leiden")
    
    dict_spinal_cord[timepoint] = adata
    




# %%
dict_spinal_cord["TDR127"]

# %%
dict_spinal_cord["TDR128"]

# %%
shared_tfs = set(dict_filtered_GRNs["TDR118"]["spinal_cord"].source)

# %%

# %%
# compute the shared TFs
shared_tfs = set(dict_filtered_GRNs["TDR118"]["spinal_cord"].source)
for timepoint in timepoints:
    shared_tfs = shared_tfs.intersection(set(dict_filtered_GRNs[timepoint]["spinal_cord"].source))
# return list(shared_tfs)

len(shared_tfs)



# %%
from sklearn.neighbors import NearestNeighbors
import umap.aligned_umap
import sklearn.datasets

def get_shared_tfs(dict_spinal_cord):
    """Get TFs present across all timepoints"""
    shared_tfs = set(dict_spinal_cord[list(dict_spinal_cord.keys())[0]].obs_names)
    for adata in dict_spinal_cord.values():
        shared_tfs = shared_tfs.intersection(set(adata.obs_names))
    return list(shared_tfs)

# def compute_tf_anchors(dict_spinal_cord, timepoints, max_k=20, max_dist=0.05, use_metric='cosine'):
#     """Compute anchors between consecutive timepoints based on shared TFs"""
#     shared_tfs = get_shared_tfs(dict_spinal_cord)
#     print(f"Number of shared TFs: {len(shared_tfs)}")
    
#     anchor_dict = []
    
#     # Get embeddings for shared TFs at each timepoint
#     embedding_list = []
#     for tp in timepoints:
#         adata = dict_spinal_cord[tp]
#         # Use PCA embeddings if they exist, otherwise use raw data
#         if 'X_pca' in adata.obsm:
#             embed = adata[shared_tfs].obsm['X_pca']
#         else:
#             embed = adata[shared_tfs].X
#         embedding_list.append(embed)
    
#     # Compute anchors between consecutive timepoints
#     for i in range(len(timepoints)-1):
#         Y = embedding_list[i]  # previous timepoint
#         X = embedding_list[i+1]  # next timepoint
        
#         # Find nearest neighbors
#         nbrs = NearestNeighbors(n_neighbors=1, metric=use_metric).fit(Y)
#         distances, indices = nbrs.kneighbors(X)
        
#         # Create pairs DataFrame
#         pairs = pd.DataFrame({
#             'neighbor': np.concatenate(indices, axis=0),
#             'dist': np.concatenate(distances, axis=0)
#         })
#         pairs.reset_index(inplace=True)
#         pairs.rename(columns={'index': 'tf_target'}, inplace=True)
        
#         # Filter by distance
#         pairs = pairs[pairs['dist'] < max_dist]
        
#         # Keep top k pairs
#         pairs = pairs.nsmallest(max_k, 'dist')
        
#         # Convert to dictionary format
#         pairs_dict = dict(zip(pairs['neighbor'].values, pairs['tf_target'].values))
#         anchor_dict.append(pairs_dict)
    
#     return anchor_dict

def compute_tf_anchors(dict_spinal_cord, timepoints, shared_tfs):
    """Create anchor dictionaries for shared TFs between consecutive timepoints"""
    # # Get shared TFs across all timepoints
    # shared_tfs = set(dict_spinal_cord[list(dict_spinal_cord.keys())[0]].obs_names)
    # for adata in dict_spinal_cord.values():
    #     shared_tfs = shared_tfs.intersection(set(adata.obs_names))
    # shared_tfs = list(shared_tfs)
    print(f"Number of shared TFs: {len(shared_tfs)}")
    
    # For each consecutive pair of timepoints, create anchor dictionary
    anchor_dict = []
    for i in range(len(timepoints)-1):
        # Get indices of shared TFs in each timepoint's adata
        tf_indices_t1 = [list(dict_spinal_cord[timepoints[i]].obs_names).index(tf) for tf in shared_tfs]
        tf_indices_t2 = [list(dict_spinal_cord[timepoints[i+1]].obs_names).index(tf) for tf in shared_tfs]
        
        # Create dictionary mapping indices between consecutive timepoints
        pairs_dict = dict(zip(tf_indices_t1, tf_indices_t2))
        anchor_dict.append(pairs_dict)
    
    return anchor_dict

def align_umaps(dict_spinal_cord, timepoints, anchor_dict):
    """Align UMAPs using computed anchors"""
    # Get embeddings for each timepoint
    embedding_list = []
    for tp in timepoints:
        if 'X_pca' in dict_spinal_cord[tp].obsm:
            embed = dict_spinal_cord[tp].obsm['X_pca']
        else:
            embed = dict_spinal_cord[tp].X
        embedding_list.append(embed)
    
    # Align UMAPs
    aligned_mapper = umap.AlignedUMAP(
        metric="cosine",
        n_neighbors=10,
        alignment_regularisation=0.001,
        alignment_window_size=4,
        n_epochs=200,
        random_state=42,
    ).fit(embedding_list, relations=anchor_dict)
    
    # Collect aligned coordinates
    all_timepoints = []
    for i, tp in enumerate(timepoints):
        aligned_umap_coord = pd.DataFrame({
            'UMAP_1': aligned_mapper.embeddings_[i].T[0],
            'UMAP_2': aligned_mapper.embeddings_[i].T[1],
            'timepoint': tp,
            'leiden': dict_spinal_cord[tp].obs['leiden'].values,
            'tf_id': dict_spinal_cord[tp].obs_names.to_list()
        })
        all_timepoints.append(aligned_umap_coord)
    
    umap_coords = pd.concat(all_timepoints)
    return umap_coords, aligned_mapper


# %%
# Compute anchors
anchor_dict = compute_tf_anchors(dict_spinal_cord, timepoints, shared_tfs)

# Align UMAPs
umap_coords, aligned_mapper = align_umaps(dict_spinal_cord, timepoints, anchor_dict)

# Plot results
g = sns.relplot(
    data=umap_coords, 
    x="UMAP_1", 
    y="UMAP_2",
    col="timepoint", 
    hue="leiden",
    kind="scatter"
)

# %%
plt.show()

# %%
embedding_list = []
for tp in timepoints:
    if 'X_pca' in dict_spinal_cord[tp].obsm:
        embed = dict_spinal_cord[tp].obsm['X_pca']
    else:
        embed = dict_spinal_cord[tp].X
    embedding_list.append(embed)

# %%
# Align UMAPs
aligned_mapper = umap.AlignedUMAP(
    metric="cosine",
    n_neighbors=10,
    alignment_regularisation=0.01,
    alignment_window_size=2,
    n_epochs=200,
    random_state=42,
).fit(embedding_list, relations=anchor_dict)



# %%
# Collect aligned coordinates
all_timepoints = []
for i, tp in enumerate(timepoints):
    aligned_umap_coord = pd.DataFrame({
        'UMAP_1': aligned_mapper.embeddings_[i].T[0],
        'UMAP_2': aligned_mapper.embeddings_[i].T[1],
        'timepoint': tp,
        'leiden': dict_spinal_cord[tp].obs['leiden'].values,
        'tf_id': dict_spinal_cord[tp].obs_names.to_list()
    })
    all_timepoints.append(aligned_umap_coord)

umap_coords = pd.concat(all_timepoints)

# %%
umap_coords

# %%
# Plot results
g = sns.relplot(
    data=umap_coords, 
    x="UMAP_1", 
    y="UMAP_2",
    col="timepoint", 
    hue="leiden",
    kind="scatter",
)
plt.show()

# %%
# Create a copy to avoid modifying original
umap_coords_viz = umap_coords.copy()

# Create TF family categories
def categorize_tf(tf_name):
    tf_name = tf_name.lower()
    if tf_name.startswith('hox'):
        return 'Hox'
    elif tf_name.startswith('sox'):
        return 'Sox'
    elif tf_name.startswith('meox'):
        return 'Meox'
    elif tf_name.startswith('meis'):
        return 'Meis'
    elif tf_name.startswith('rar') or tf_name.startswith('rxr'):
        return 'RA'
    elif tf_name.startswith('tbx'):
        return 'Tbx'
    elif tf_name.startswith('hmga1a'):
        return 'Hmga1a'
    else:
        return 'Other'

# Add new column for TF families
umap_coords_viz['tf_family'] = umap_coords_viz['tf_id'].apply(categorize_tf)

# Plot with TF families highlighted
g = sns.relplot(
    data=umap_coords_viz,
    x="UMAP_1",
    y="UMAP_2",
    col="timepoint",
    hue="tf_family",
    # style="tf_family",  # Different markers for different families
    palette={"Hox": "#E41A1C", 
            "Sox": "#377EB8", 
            "Meox": "#4DAF4A", 
            "Meis": "#984EA3", 
            "RA": "#FF7F00", 
            "Tbx": "#FFFF33", 
            "Hmga1a": "#8B008B",  # Dark magenta for Hmga1a
            "Other": "#CCCCCC"},  # Light grey for Other
    kind="scatter",
    height=4,
    aspect=1.2
)


# Adjust legend
g._legend.set_title("TF Family")

plt.show()

# Print counts of each family
print("\nTF Family counts:")
print(umap_coords_viz['tf_family'].value_counts())

# %%
# Create the figure first
g = sns.FacetGrid(
    data=umap_coords_viz,
    col="timepoint",
    height=4,
    aspect=1.2
)

# Plot Others first in each subplot
for ax in g.axes.flat:
    timepoint = ax.get_title().split(' = ')[1]
    data_others = umap_coords_viz[
        (umap_coords_viz['tf_family'] == 'Other') & 
        (umap_coords_viz['timepoint'] == timepoint)
    ]
    ax.scatter(
        data_others['UMAP_1'],
        data_others['UMAP_2'],
        c="#CCCCCC",
        alpha=0.3
    )

# Plot TF families with a single legend
legend_handles = []
legend_labels = []
palette = {
    "Hox": "#E41A1C", 
    "Sox": "#377EB8", 
    "Meox": "#4DAF4A", 
    "Meis": "#984EA3", 
    "RA": "#FF7F00", 
    "Tbx": "#FFFF33", 
    "Hmga1a": "#8B008B"
}

for tf_family in palette.keys():
    for ax in g.axes.flat:
        timepoint = ax.get_title().split(' = ')[1]
        data_family = umap_coords_viz[
            (umap_coords_viz['tf_family'] == tf_family) & 
            (umap_coords_viz['timepoint'] == timepoint)
        ]
        scatter = ax.scatter(
            data_family['UMAP_1'],
            data_family['UMAP_2'],
            c=palette[tf_family],
            label=tf_family,
            alpha=1.0
        )
        if ax == g.axes.flat[0]:  # Only collect legend handles from first subplot
            legend_handles.append(scatter)
            legend_labels.append(tf_family)

# Add legend to the figure
g.fig.legend(
    legend_handles,
    legend_labels,
    title="TF Family",
    bbox_to_anchor=(1.02, 0.5),
    loc='center left'
)

# Adjust layout to make room for legend
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()

# Print counts of each family
print("\nTF Family counts:")
print(umap_coords_viz['tf_family'].value_counts())

# %%
dict_spinal_cord

# %%
# Get ordered list of timepoints
timepoints = ['TDR126', 'TDR127', 'TDR128', 
              'TDR118', 'TDR125', 'TDR124']

# Dictionary to store distances for each TF
tf_distances = {}

# For each TF, compute distances between consecutive timepoints in PCA space
for tf in dict_spinal_cord[timepoints[0]].obs_names:  # use first timepoint's TFs
    tf_distances[tf] = []
    
    # Compare consecutive timepoints
    for i in range(len(timepoints)-1):
        t1 = timepoints[i]
        t2 = timepoints[i+1]
        
        # Get PCA coordinates for this TF at both timepoints
        pca_t1 = dict_spinal_cord[t1].obsm['X_pca'][dict_spinal_cord[t1].obs_names == tf]
        pca_t2 = dict_spinal_cord[t2].obsm['X_pca'][dict_spinal_cord[t2].obs_names == tf]
        
        # Compute distance (Euclidean)
        dist_euclidean = np.sqrt(np.sum((pca_t1 - pca_t2)**2))
        
        # Or compute cosine similarity
        cos_sim = np.dot(pca_t1.flatten(), pca_t2.flatten()) / (
            np.linalg.norm(pca_t1) * np.linalg.norm(pca_t2)
        )
        
        tf_distances[tf].append({
            'timepoint_pair': f"{t1}->{t2}",
            'euclidean_dist': dist_euclidean,
            'cosine_similarity': cos_sim
        })

# Convert to DataFrame for easier analysis
distances_df = []
for tf, distances in tf_distances.items():
    for d in distances:
        distances_df.append({
            'tf': tf,
            'timepoint_pair': d['timepoint_pair'],
            'euclidean_dist': d['euclidean_dist'],
            'cosine_similarity': d['cosine_similarity']
        })
distances_df = pd.DataFrame(distances_df)

# Get TFs with largest total movement
total_movement = distances_df.groupby('tf')['euclidean_dist'].sum().sort_values(ascending=False)
print("\nTop 20 TFs with largest total movement in PCA space:")
print(total_movement.head(20))

# Or get TFs with most dramatic changes between specific timepoints
print("\nLargest changes between consecutive timepoints:")
for tp_pair in distances_df['timepoint_pair'].unique():
    print(f"\n{tp_pair}:")
    print(distances_df[distances_df['timepoint_pair'] == tp_pair]
          .nlargest(5, 'euclidean_dist')[['tf', 'euclidean_dist']])

# %%
# Create list of top TFs to label
top_movers = ['hmga1a', 'meox1', 'sox19a', 'tbx16', 'rfx4', 'foxp4', 'rxraa', 'rarga', 
              'sox3', 'meis2a', 'meis1a', 'sox13', 'sox11a', 'meis1b', 'ved', 'pax6a', 
              'pax6b', 'vox', 'sox21a', 'lef1']

# Create a copy to avoid modifying original
umap_coords_viz = umap_coords.copy()

def categorize_tf(tf_name):
    tf_name = tf_name.lower()
    if tf_name.startswith('hox'):
        return 'Hox'
    elif tf_name.startswith('sox'):
        return 'Sox'
    elif tf_name.startswith('meox'):
        return 'Meox'
    elif tf_name.startswith('meis'):
        return 'Meis'
    elif tf_name.startswith('rar') or tf_name.startswith('rxr'):
        return 'RA'
    elif tf_name.startswith('tbx'):
        return 'Tbx'
    elif tf_name.startswith('hmga1a'):
        return 'Hmga1a'
    else:
        return 'Other'

# Add new column for TF families
umap_coords_viz['tf_family'] = umap_coords_viz['tf_id'].apply(categorize_tf)

# Create the figure
g = sns.FacetGrid(
    data=umap_coords_viz,
    col="timepoint",
    height=4,
    aspect=1.2
)

# Plot Others first in each subplot
for ax in g.axes.flat:
    timepoint = ax.get_title().split(' = ')[1]
    data_others = umap_coords_viz[
        (umap_coords_viz['tf_family'] == 'Other') & 
        (umap_coords_viz['timepoint'] == timepoint)
    ]
    ax.scatter(
        data_others['UMAP_1'],
        data_others['UMAP_2'],
        c="#CCCCCC",
        alpha=0.3
    )

# Plot TF families with a single legend
legend_handles = []
legend_labels = []
palette = {
    "Hox": "#E41A1C", 
    "Sox": "#377EB8", 
    "Meox": "#4DAF4A", 
    "Meis": "#984EA3", 
    "RA": "#FF7F00", 
    "Tbx": "#FFFF33", 
    "Hmga1a": "#8B008B"
}

for tf_family in palette.keys():
    for ax in g.axes.flat:
        timepoint = ax.get_title().split(' = ')[1]
        data_family = umap_coords_viz[
            (umap_coords_viz['tf_family'] == tf_family) & 
            (umap_coords_viz['timepoint'] == timepoint)
        ]
        scatter = ax.scatter(
            data_family['UMAP_1'],
            data_family['UMAP_2'],
            c=palette[tf_family],
            label=tf_family,
            alpha=1.0
        )
        if ax == g.axes.flat[0]:
            legend_handles.append(scatter)
            legend_labels.append(tf_family)
        
        # Add labels for top movers in this family
        for _, row in data_family[data_family['tf_id'].isin(top_movers)].iterrows():
            ax.annotate(
                row['tf_id'],
                (row['UMAP_1'], row['UMAP_2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )

# Add legend to the figure
g.fig.legend(
    legend_handles,
    legend_labels,
    title="TF Family",
    bbox_to_anchor=(1.02, 0.5),
    loc='center left'
)

# Adjust layout to make room for legend
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()

# %% [markdown]
# ### A heatmap showing the Euclidean distance between timepoints (PCs)

# %%
# Create a matrix of distances for top TFs
top_tfs = total_movement.head(20).index
transitions = distances_df['timepoint_pair'].unique()

# Create matrix for heatmap
heatmap_data = []
for tf in top_tfs:
    tf_distances = distances_df[distances_df['tf'] == tf]['euclidean_dist'].values
    heatmap_data.append(tf_distances)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, 
            xticklabels=transitions,
            yticklabels=top_tfs,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f')
plt.title('PCA Distance Between Consecutive Timepoints for Top Variable TFs')
plt.xlabel('Timepoint Transition')
plt.ylabel('TF')
plt.tight_layout()
plt.show()

# %%
figpath

# %%
# Get top N most variable TFs
top_N_tfs = total_movement.head(50).index
transitions = distances_df['timepoint_pair'].unique()

# Create matrix data with peak timing information
heatmap_data = []
for tf in top_N_tfs:
    tf_distances = distances_df[distances_df['tf'] == tf]['euclidean_dist'].values
    peak_transition_idx = np.argmax(tf_distances)
    heatmap_data.append({
        'tf': tf,
        'distances': tf_distances,
        'peak_transition': peak_transition_idx,
        'peak_value': np.max(tf_distances)
    })

# Convert to DataFrame and sort by peak transition and magnitude
heatmap_df = pd.DataFrame(heatmap_data)
heatmap_df = heatmap_df.sort_values(['peak_transition', 'peak_value'], ascending=[True, False])

# Create matrix for heatmap
heatmap_matrix = np.array([row['distances'] for _, row in heatmap_df.iterrows()])

# Create heatmap
plt.figure(figsize=(10, 20))  # Adjust size for 200 TFs
sns.heatmap(heatmap_matrix, 
            xticklabels=transitions,
            yticklabels=heatmap_df['tf'],
            cmap='YlOrRd',
            annot=False,  # Remove annotations as they would be too crowded
            center=np.median(heatmap_matrix))

plt.title('PCA Distance Between Consecutive Timepoints\nTop 50 Variable TFs (clustered by peak transition)')
plt.xlabel('Timepoint Transition')
plt.ylabel('TF')
plt.tight_layout()
celltype = "spinal_cord"
plt.savefig(figpath + f"euclidean_dist_PCs_grn_{celltype}.pdf")
plt.show()

# Print summary of TFs peaking at each transition
print("\nNumber of TFs peaking at each transition:")
print(heatmap_df.groupby('peak_transition').size())

# Print top TFs for each transition group
for i, transition in enumerate(transitions):
    print(f"\nTop 10 TFs peaking at {transition}:")
    peak_tfs = heatmap_df[heatmap_df['peak_transition'] == i].head(10)
    print(peak_tfs[['tf', 'peak_value']].to_string())

# %%
# Create a matrix of distances for all TFs
all_tfs = distances_df['tf'].unique()
transitions = distances_df['timepoint_pair'].unique()

# Create DataFrame for heatmap
heatmap_data = []
for tf in all_tfs:
    tf_distances = distances_df[distances_df['tf'] == tf]['euclidean_dist'].values
    heatmap_data.append({
        'tf': tf,
        'max_dist': np.max(tf_distances),  # Store max distance for sorting
        'peak_timepoint': transitions[np.argmax(tf_distances)],  # Store peak timepoint
        'distances': tf_distances
    })

# Convert to DataFrame
heatmap_df = pd.DataFrame(heatmap_data)

# Sort by peak timepoint first, then by maximum distance within each peak timepoint
heatmap_df = heatmap_df.sort_values(['peak_timepoint', 'max_dist'], ascending=[True, False])

# Create matrix for heatmap
heatmap_matrix = np.array([row['distances'] for _, row in heatmap_df.iterrows()])

# Create heatmap
plt.figure(figsize=(12, 20))  # Adjust size as needed
sns.heatmap(heatmap_matrix, 
            xticklabels=transitions,
            yticklabels=heatmap_df['tf'],
            cmap='YlOrRd',
            center=np.median(heatmap_matrix),
            robust=True)  # Use robust scaling to handle outliers

plt.title('TF Distance Changes Between Consecutive Timepoints\n(sorted by peak transition)')
plt.xlabel('Timepoint Transition')
plt.ylabel('TF')
plt.tight_layout()
plt.show()

# Print summary of TFs peaking at each transition
print("\nNumber of TFs peaking at each transition:")
print(heatmap_df['peak_timepoint'].value_counts())

# Print top TFs for each peak timepoint
for transition in transitions:
    print(f"\nTop 10 TFs peaking at {transition}:")
    peak_tfs = heatmap_df[heatmap_df['peak_timepoint'] == transition].head(10)
    print(peak_tfs[['tf', 'max_dist']].to_string())

# %% [markdown]
# ### line plot to show the cumulative "distance" change for each TF 

# %%
# Create cumulative distance plot for top TFs
plt.figure(figsize=(12, 6))

for tf in top_tfs:
    tf_data = distances_df[distances_df['tf'] == tf]
    cum_dist = np.cumsum(tf_data['euclidean_dist'])
    plt.plot(range(len(transitions)), cum_dist, marker='o', label=tf)

plt.xticks(range(len(transitions)), transitions, rotation=45)
plt.ylabel('Cumulative Distance in PCA Space')
plt.xlabel('Timepoint Transitions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Cumulative Movement of Top Variable TFs')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))

# Show only top 5-6 most distinctive patterns
key_tfs = ['hmga1a', 'meox1', 'sox19a', 'tbx16', 'rfx4']
for tf in key_tfs:
    tf_data = distances_df[distances_df['tf'] == tf]
    cum_dist = np.cumsum(tf_data['euclidean_dist'])
    plt.plot(range(len(transitions)), cum_dist, marker='o', linewidth=2.5, label=tf)

plt.xticks(range(len(transitions)), transitions, rotation=45)
plt.ylabel('Cumulative Distance in PCA Space')
plt.xlabel('Timepoint Transitions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.title('Cumulative Movement of Key Variable TFs')
plt.tight_layout()
plt.show()

# %%
# Create two subplots: early movers vs late movers
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Early movers (significant changes in first two transitions)
early_movers = ['sox19a', 'meox1', 'hmga1a']
for tf in early_movers:
    tf_data = distances_df[distances_df['tf'] == tf]
    cum_dist = np.cumsum(tf_data['euclidean_dist'])
    ax1.plot(range(len(transitions)), cum_dist, marker='o', linewidth=2, label=tf)
ax1.set_title('Early Dynamic TFs')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Late movers (significant changes in later transitions)
late_movers = ['meis1b', 'foxp4', 'rxraa']
for tf in late_movers:
    tf_data = distances_df[distances_df['tf'] == tf]
    cum_dist = np.cumsum(tf_data['euclidean_dist'])
    ax2.plot(range(len(transitions)), cum_dist, marker='o', linewidth=2, label=tf)
ax2.set_title('Late Dynamic TFs')
ax2.legend()
ax2.grid(True, alpha=0.3)

for ax in [ax1, ax2]:
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(transitions, rotation=45)
    ax.set_ylabel('Cumulative Distance in PCA Space')

plt.xlabel('Timepoint Transitions')
plt.tight_layout()
plt.show()

# %%

# %%

# %% [markdown]
# ## Repeat the GRN UMAP analysis for "NMPs"

# %%

# %%
# subset the GRNs for "NMPs" for each timepoint
# For loop to plot the GRNs over timepoints
# Choose one celltype
celltype = "NMPs"

# for all the rest of the timepoints (0-30 somites)
timepoints = ["TDR126", "TDR127", "TDR128", "TDR118", "TDR125", "TDR124"]

# Step 1. collect all sources and targets across all timepoints
all_sources = set()
all_targets = set()

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
# ### compute the GRN UMAPs (TFs-by-genes)

# %%
dict_nmps = {}

for timepoint in timepoints:
    df_counts = df_counts_union[timepoint]
    
    # create an adata object (TFs-by-target_genes)
    adata = sc.AnnData(X=df_counts.T.values) # transpose
    adata.obs_names = df_counts.columns
    adata.var_names = df_counts.index

    print(adata)
    
    
    # computing UMAP
    # let's not normalize the counts, as the counts were already normalized at the whole GRN level
    # sc.pp.scale(adata_df1)
    sc.tl.pca(adata)
    sc.pl.pca(adata)

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

    sc.tl.umap(adata, min_dist=0.05)

    sc.tl.leiden(adata, resolution=0.1)
    sc.pl.umap(adata, color="leiden")
    
    dict_nmps[timepoint] = adata
    




# %%
# Compute anchors
anchor_dict = compute_tf_anchors(dict_nmps, timepoints, shared_tfs)

# Align UMAPs
umap_coords, aligned_mapper = align_umaps(dict_nmps, timepoints, anchor_dict)

# Plot results
g = sns.relplot(
    data=umap_coords, 
    x="UMAP_1", 
    y="UMAP_2",
    col="timepoint", 
    hue="leiden",
    kind="scatter"
)

# %%
plt.show()

# %%
embedding_list = []
for tp in timepoints:
    if 'X_pca' in dict_nmps[tp].obsm:
        embed = dict_nmps[tp].obsm['X_pca']
    else:
        embed = dict_nmps[tp].X
    embedding_list.append(embed)

# %%
# Align UMAPs
aligned_mapper = umap.AlignedUMAP(
    metric="cosine",
    n_neighbors=10,
    alignment_regularisation=0.01,
    alignment_window_size=2,
    n_epochs=200,
    random_state=42,
).fit(embedding_list, relations=anchor_dict)



# %%
# Collect aligned coordinates
all_timepoints = []
for i, tp in enumerate(timepoints):
    aligned_umap_coord = pd.DataFrame({
        'UMAP_1': aligned_mapper.embeddings_[i].T[0],
        'UMAP_2': aligned_mapper.embeddings_[i].T[1],
        'timepoint': tp,
        'leiden': dict_nmps[tp].obs['leiden'].values,
        'tf_id': dict_nmps[tp].obs_names.to_list()
    })
    all_timepoints.append(aligned_umap_coord)

umap_coords = pd.concat(all_timepoints)

# %%
umap_coords

# %%
# Plot results
g = sns.relplot(
    data=umap_coords, 
    x="UMAP_1", 
    y="UMAP_2",
    col="timepoint", 
    hue="leiden",
    kind="scatter",
)
plt.show()

# %%
# Get ordered list of timepoints
timepoints = ['TDR126', 'TDR127', 'TDR128', 
              'TDR118', 'TDR125', 'TDR124']

# Dictionary to store distances for each TF
tf_distances = {}

# For each TF, compute distances between consecutive timepoints in PCA space
for tf in dict_nmps[timepoints[0]].obs_names:  # use first timepoint's TFs
    tf_distances[tf] = []
    
    # Compare consecutive timepoints
    for i in range(len(timepoints)-1):
        t1 = timepoints[i]
        t2 = timepoints[i+1]
        
        # Get PCA coordinates for this TF at both timepoints
        pca_t1 = dict_nmps[t1].obsm['X_pca'][dict_nmps[t1].obs_names == tf]
        pca_t2 = dict_nmps[t2].obsm['X_pca'][dict_nmps[t2].obs_names == tf]
        
        # Compute distance (Euclidean)
        dist_euclidean = np.sqrt(np.sum((pca_t1 - pca_t2)**2))
        
        # Or compute cosine similarity
        cos_sim = np.dot(pca_t1.flatten(), pca_t2.flatten()) / (
            np.linalg.norm(pca_t1) * np.linalg.norm(pca_t2)
        )
        
        tf_distances[tf].append({
            'timepoint_pair': f"{t1}->{t2}",
            'euclidean_dist': dist_euclidean,
            'cosine_similarity': cos_sim
        })

# Convert to DataFrame for easier analysis
distances_df = []
for tf, distances in tf_distances.items():
    for d in distances:
        distances_df.append({
            'tf': tf,
            'timepoint_pair': d['timepoint_pair'],
            'euclidean_dist': d['euclidean_dist'],
            'cosine_similarity': d['cosine_similarity']
        })
distances_df = pd.DataFrame(distances_df)

# Get TFs with largest total movement
total_movement = distances_df.groupby('tf')['euclidean_dist'].sum().sort_values(ascending=False)
print("\nTop 20 TFs with largest total movement in PCA space:")
print(total_movement.head(20))

# Or get TFs with most dramatic changes between specific timepoints
print("\nLargest changes between consecutive timepoints:")
for tp_pair in distances_df['timepoint_pair'].unique():
    print(f"\n{tp_pair}:")
    print(distances_df[distances_df['timepoint_pair'] == tp_pair]
          .nlargest(5, 'euclidean_dist')[['tf', 'euclidean_dist']])

# %%
# Create list of top TFs to label
top_movers = total_movement.head(20).index

# Create a copy to avoid modifying original
umap_coords_viz = umap_coords.copy()

def categorize_tf(tf_name):
    tf_name = tf_name.lower()
    if tf_name.startswith('hox'):
        return 'Hox'
    elif tf_name.startswith('sox'):
        return 'Sox'
    elif tf_name.startswith('meox'):
        return 'Meox'
    elif tf_name.startswith('meis'):
        return 'Meis'
    elif tf_name.startswith('rar') or tf_name.startswith('rxr'):
        return 'RA'
    elif tf_name.startswith('tbx'):
        return 'Tbx'
    elif tf_name.startswith('hmga1a'):
        return 'Hmga1a'
    else:
        return 'Other'

# Add new column for TF families
umap_coords_viz['tf_family'] = umap_coords_viz['tf_id'].apply(categorize_tf)

# Create the figure
g = sns.FacetGrid(
    data=umap_coords_viz,
    col="timepoint",
    height=4,
    aspect=1.2
)

# Plot Others first in each subplot
for ax in g.axes.flat:
    timepoint = ax.get_title().split(' = ')[1]
    data_others = umap_coords_viz[
        (umap_coords_viz['tf_family'] == 'Other') & 
        (umap_coords_viz['timepoint'] == timepoint)
    ]
    ax.scatter(
        data_others['UMAP_1'],
        data_others['UMAP_2'],
        c="#CCCCCC",
        alpha=0.3
    )

# Plot TF families with a single legend
legend_handles = []
legend_labels = []
palette = {
    "Hox": "#E41A1C", 
    "Sox": "#377EB8", 
    "Meox": "#4DAF4A", 
    "Meis": "#984EA3", 
    "RA": "#FF7F00", 
    "Tbx": "#FFFF33", 
    "Hmga1a": "#8B008B"
}

for tf_family in palette.keys():
    for ax in g.axes.flat:
        timepoint = ax.get_title().split(' = ')[1]
        data_family = umap_coords_viz[
            (umap_coords_viz['tf_family'] == tf_family) & 
            (umap_coords_viz['timepoint'] == timepoint)
        ]
        scatter = ax.scatter(
            data_family['UMAP_1'],
            data_family['UMAP_2'],
            c=palette[tf_family],
            label=tf_family,
            alpha=1.0
        )
        if ax == g.axes.flat[0]:
            legend_handles.append(scatter)
            legend_labels.append(tf_family)
        
        # Add labels for top movers in this family
        for _, row in data_family[data_family['tf_id'].isin(top_movers)].iterrows():
            ax.annotate(
                row['tf_id'],
                (row['UMAP_1'], row['UMAP_2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )

# Add legend to the figure
g.fig.legend(
    legend_handles,
    legend_labels,
    title="TF Family",
    bbox_to_anchor=(1.02, 0.5),
    loc='center left'
)

# Adjust layout to make room for legend
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()

# %% [markdown]
# ### A heatmap showing the Euclidean distance between timepoints (PCs)

# %%
# Create a matrix of distances for top TFs
top_tfs = total_movement.head(20).index
transitions = distances_df['timepoint_pair'].unique()

# Create matrix for heatmap
heatmap_data = []
for tf in top_tfs:
    tf_distances = distances_df[distances_df['tf'] == tf]['euclidean_dist'].values
    heatmap_data.append(tf_distances)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, 
            xticklabels=transitions,
            yticklabels=top_tfs,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f')
plt.title('PCA Distance Between Consecutive Timepoints for Top Variable TFs')
plt.xlabel('Timepoint Transition')
plt.ylabel('TF')
plt.tight_layout()
plt.show()

# %%
figpath

# %%
# Get top N most variable TFs
top_N_tfs = total_movement.head(50).index
transitions = distances_df['timepoint_pair'].unique()

# Create matrix data with peak timing information
heatmap_data = []
for tf in top_N_tfs:
    tf_distances = distances_df[distances_df['tf'] == tf]['euclidean_dist'].values
    peak_transition_idx = np.argmax(tf_distances)
    heatmap_data.append({
        'tf': tf,
        'distances': tf_distances,
        'peak_transition': peak_transition_idx,
        'peak_value': np.max(tf_distances)
    })

# Convert to DataFrame and sort by peak transition and magnitude
heatmap_df = pd.DataFrame(heatmap_data)
heatmap_df = heatmap_df.sort_values(['peak_transition', 'peak_value'], ascending=[True, False])

# Create matrix for heatmap
heatmap_matrix = np.array([row['distances'] for _, row in heatmap_df.iterrows()])

# Create heatmap
plt.figure(figsize=(10, 20))  # Adjust size for 200 TFs
sns.heatmap(heatmap_matrix, 
            xticklabels=transitions,
            yticklabels=heatmap_df['tf'],
            cmap='YlOrRd',
            annot=False,  # Remove annotations as they would be too crowded
            center=np.median(heatmap_matrix))

plt.title('PCA Distance Between Consecutive Timepoints\nTop 50 Variable TFs (clustered by peak transition)')
plt.xlabel('Timepoint Transition')
plt.ylabel('TF')
plt.tight_layout()
celltype = "nmps"
plt.savefig(figpath + f"euclidean_dist_PCs_grn_{celltype}.pdf")
plt.show()

# Print summary of TFs peaking at each transition
print("\nNumber of TFs peaking at each transition:")
print(heatmap_df.groupby('peak_transition').size())

# Print top TFs for each transition group
for i, transition in enumerate(transitions):
    print(f"\nTop 10 TFs peaking at {transition}:")
    peak_tfs = heatmap_df[heatmap_df['peak_transition'] == i].head(10)
    print(peak_tfs[['tf', 'peak_value']].to_string())

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

# %%
for timepoint in timepoints:
    # subset for the timepoint
    umap_coords_sub = umap_coords[umap_coords.timepoint==timepoint]
    umap_aligned = umap_coords_sub[['UMAP_1', 'UMAP_2']].to_numpy()
    dict_spinal_cord[timepoint].obsm["X_umap_aligned"] = umap_aligned
    

# %%
for timepoint in timepoints:
    # subset for the timepoint
    umap_coords_sub = umap_coords[umap_coords.timepoint==timepoint]
    umap_aligned = umap_coords_sub[['UMAP_1', 'UMAP_2']].to_numpy()
    adata.obsm[f"X_umap_aligned_{timepoint}"] = umap_aligned
    
adata

# %%

# %%
adata.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/annotations/grn_spinal_cord_TDR126_alignedUMAPs.h5ad")

# %%

# %%
from matplotlib import animation
from scipy.interpolate import interp1d

def create_tf_animation(umap_coords_list, timepoints, n_interpolation_steps=30):
    """
    Create animation of TFs moving through UMAP space across timepoints
    
    Parameters:
    -----------
    umap_coords_list : list of numpy arrays
        List of UMAP coordinates for each timepoint, each array shape (n_tfs, 2)
    timepoints : list
        List of timepoint names
    n_interpolation_steps : int
        Number of frames to interpolate between each timepoint
    """
    # Get number of TFs
    n_tfs = umap_coords_list[0].shape[0]
    
    # Create interpolation between timepoints
    interpolated_traces = []
    
    # Create time points for interpolation
    t = np.arange(len(timepoints))
    t_interp = np.linspace(0, len(timepoints)-1, n_interpolation_steps * (len(timepoints)-1))
    
    # Interpolate for each TF
    for tf_idx in range(n_tfs):
        # Get coordinates for this TF across all timepoints
        x_coords = np.array([coords[tf_idx, 0] for coords in umap_coords_list])
        y_coords = np.array([coords[tf_idx, 1] for coords in umap_coords_list])
        
        # Create interpolation functions
        f_x = interp1d(t, x_coords, kind='cubic')
        f_y = interp1d(t, y_coords, kind='cubic')
        
        # Generate interpolated points
        x_interp = f_x(t_interp)
        y_interp = f_y(t_interp)
        
        interpolated_traces.append(np.column_stack([x_interp, y_interp]))
    
    # Convert to array and transpose to get frame-by-frame coordinates
    interpolated_traces = np.array(interpolated_traces)
    offsets = np.array(interpolated_traces).transpose(1, 0, 2)
    
    # Set up the figure
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    
    # Create scatter plot
    scat = ax.scatter([], [], s=5, alpha=0.6)
    
    # Add timepoint indicator
    text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                  verticalalignment='top')
    
    # Set axis limits
    all_coords = np.concatenate(umap_coords_list)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # Set up animation function
    def animate(i):
        scat.set_offsets(offsets[i])
        # Calculate current timepoint
        current_timepoint = timepoints[int(i / n_interpolation_steps)]
        next_timepoint = timepoints[min(int(i / n_interpolation_steps) + 1, len(timepoints)-1)]
        text.set_text(f'{current_timepoint} → {next_timepoint}')
        return scat, text

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(t_interp),
        interval=50,
        blit=True
    )
    
    return anim, fig



# %%
# Prepare list of UMAP coordinates for each timepoint
umap_coords_list = []

for timepoint in timepoints:
    # subset for the timepoint
    umap_coords_sub = umap_coords[umap_coords.timepoint==timepoint]
    umap_aligned = umap_coords_sub[['UMAP_1', 'UMAP_2']].to_numpy()
    # adata.obsm[f"X_umap_aligned_{timepoint}"] = umap_aligned
    umap_coords_list.append(umap_aligned)

# Create animation
anim, fig = create_tf_animation(umap_coords_list, timepoints)

# Save animation
anim.save('tf_movement.gif', writer='pillow')

# %%
from IPython.display import HTML
HTML(anim.to_jshtml())

# %%
dict_spinal_cord

# %%
# Prepare list of UMAP coordinates for each timepoint
umap_coords_ind_list = []

for timepoint in timepoints:
    # subset for the timepoint
    umap_coords_sub = dict_spinal_cord[timepoint].obsm["X_umap"]
    # umap_coords_sub = umap_coords[umap_coords.timepoint==timepoint]
    # umap_aligned = umap_coords_sub[['UMAP_1', 'UMAP_2']].to_numpy()
    # adata.obsm[f"X_umap_aligned_{timepoint}"] = umap_aligned
    # umap_coords_list.append(umap_aligned)
    umap_coords_ind_list.append(umap_coords_sub)

# Create animation
anim, fig = create_tf_animation(umap_coords_ind_list, timepoints)

# Save animation
anim.save('tf_movement_ind_umaps.gif', writer='pillow')

# %%

# %%

# %%

# %%

# %%
