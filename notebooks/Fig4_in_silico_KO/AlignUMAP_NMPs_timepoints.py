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
#     display_name: Global single-cell-base
#     language: python
#     name: global-single-cell-base
# ---

# %% [markdown]
# ## Align UMAPs for the NMP lineages (subsets) for different timepoints
# - Last updated: 10/18/2024
# - Author: Yang-Joon Kim
#
# ### NOTES:
# 1. Let's start with the ATAC modality to align ("X_lsi")
#
# ### Overview:
# - Start with the UMAPs from individual timepoints 
# - Find the k-nearest neighbors for each cluster in dataset t to dataset (t+1), using "X_lsi" for all timepoints
# - Aligned UMAP should reflect the time-dependence of the dataset
#
#
# ### TO-DO:
# - clean documentation on what is the required format for the input datasets

# %%
import pandas as pd 
import numpy as np 
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns 

from scipy.sparse import csr_matrix
from scipy.io import mmread
import re

from sklearn.neighbors import NearestNeighbors

import umap.aligned_umap
import sklearn.datasets

# %% [markdown]
# ## Step 1. Subset for the NMP trajectory population (annotation_ML_coarse)
#
# - mesoderm: ["NMPs", "tail_bud", "PSM", "somites","fast_muscle"]
# - neuro-ectoderm: ["NMPs", "spinal_cord", "neural_posterior"]

# %%
# load the master object
multiome = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
multiome

# %%
# import the dim.reductions
integrated_lsi = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_lsi.csv", index_col=0)
integrated_pca = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_pca.csv", index_col=0)


# %%
# filter the integrated_lsi (filter out the rows from the "low_quality_cells")
integrated_lsi = integrated_lsi[integrated_lsi.index.isin(multiome.obs_names)]
integrated_lsi.shape

# %%
metadata = multiome.obs
metadata.annotation_ML_coarse.unique()

# %%
# create a dictionary to map the timepoints to each "dataset"
dict_timepoints = {"0somites":"TDR126",
                   "5somites":"TDR127",
                   "10somites":"TDR128",
                   "15somites":"TDR118",
                   "15somites-2":"TDR119",
                   "20somites":"TDR125",
                   "30somites":"TDR124",
}
# Creating a reversed dictionary
reversed_dict_timepoints = {value: key for key, value in dict_timepoints.items()}

print(reversed_dict_timepoints)

# %%
metadata["timepoints"] = metadata.dataset.map(reversed_dict_timepoints)
metadata.head()

# %%
# First, subset for the NMP trajectories - celltypes that are in Figure 6, zebrahub
celltypes_NMPs = ["NMPs", "tail_bud", "PSM", "somites", "fast_muscle",
                  "spinal_cord", "neural_posterior"]

# subset the metadata and lsi
metadata_NMPs = metadata[metadata.annotation_ML_coarse.isin(celltypes_NMPs)]
integrated_lsi_NMPs = integrated_lsi[metadata.annotation_ML_coarse.isin(celltypes_NMPs)]

# %%
# define the timepointns to subset the metadata and lsi for individual timepoint
timepoints = ['0somites', '5somites', '10somites', '15somites','15somites-2', '20somites', '30somites']

# %%
# subset the metadata dataframe
meta_list = []
for timepoint in timepoints:
    df = metadata_NMPs[metadata_NMPs.timepoints==timepoint]
    meta_list.append(df)

# %%
metadata_NMPs.timepoints.unique()

# %%
# subset the lsi dataframe
lsi_list = []
lsi_df_list = []
n_lsis = 40

for timepoint in timepoints:
    # subset the metadata first, to use the indices for integrated_lsi subsetting (as they match)
    df_meta = metadata_NMPs[metadata_NMPs.timepoints==timepoint]
    # subset the integrated_lsi using the indices
    df_lsi = integrated_lsi_NMPs[integrated_lsi_NMPs.index.isin(df_meta.index)]
    
    print(df_meta.shape)
    print(df_lsi.shape)
    
    # add the lsi list
    lsi_df_list.append(df_lsi)
    
    # extract the lsi components
    X = df_lsi.values
    # subset for 2:n_lsis components (we exclude the first LSI as it's usually correlated to the seq.depth)
    lsi_list.append(X[:,1:n_lsis])


# %% [markdown]
# Based on the distribution of distances we can select the top % of cells to use as anchors between the two datasets. 

# %%
for m in meta_list:
    print(m.shape)

# %%
for lsi in lsi_list: 
    print(lsi.shape)

# %% [markdown]
# Merge PCA projections

# %% [markdown]
# ### Run Aligned UMAP

# %% [markdown]
# Create list of dictionaries 

# %%
anchor_dict = []
# # Parameter set 1: 
# max_k = 10
# frac_k = 0.05
# max_dist = 0.05
# use_metric = 'cosine'

# Parameter set 2: 
max_k = 20
frac_k = 0.05
max_dist = 0.05
use_metric = 'cosine'

annotation_class = "annotation_ML_coarse"

for i in range(len(timepoints)-1):
    Y = lsi_list[i] # train on previous timepoint "progenitor space"
    X = lsi_list[i+1] # for cells in next timepoint predict "progenitors"
    
    nbrs = NearestNeighbors(n_neighbors=1, #algorithm='ball_tree',
                           metric = use_metric).fit(Y)
    
    distances, indices = nbrs.kneighbors(X) # predict top progenitor for all cells
    
    neigh_distribution = np.concatenate(distances, axis = 0)
    neigh_indexes = np.concatenate(indices, axis =0)
    
    pairs = pd.DataFrame( {'neighbor':neigh_indexes ,'dist':neigh_distribution})
    pairs.reset_index(inplace = True)
    pairs.rename(columns ={'index':'cell_target'},inplace=True)
    
    # Grup by cell type (we'll find top anchors for each cell type)
    pairs['cell_type'] = meta_list[i+1][annotation_class].values
    df1 = pairs.groupby(['cell_type'])

    df2 = df1.apply(lambda x: x.sort_values(["dist"]))

    df3=df2.reset_index(drop=True)

    # keep the top neighbors for each cell type (NOTE some cells in t+1 will map to many cells in t)
    
    # For each progenitor in t keep only the cell in (t+1) with the smallest distance
    # Closest relative
    pairs_rank = df3.groupby('neighbor').head(1)
    
    #pairs_rank = df3.groupby('cell_type').head(max_k)
    
    # For each cell type we keep the top k prgenitor relations 
    pairs_rank = pairs_rank.groupby('cell_type').head(max_k)
    
    # filter any neighbor pair with distance larger than threshold
    pairs_rank = pairs_rank[pairs_rank['dist']<max_dist] 
    
    
    pairs_dict = {pairs_rank['neighbor'].values[j] :pairs_rank['cell_target'].values[j]  for j in range(pairs_rank.shape[0])}
    
    
    anchor_dict.append(pairs_dict)

# %%
aligned_mapper = umap.AlignedUMAP(metric="cosine",
                                    n_neighbors=20,
                                    alignment_regularisation=0.01, # strength of the anchors across timepoints, default 0.1
                                    alignment_window_size=3, # how far forward and backward across the datasets we look when doing alignment, defaut 5
                                    n_epochs=200,
                                    random_state=42,).fit(lsi_list, relations=anchor_dict)

# %%
all_timepoints = []
annotation_class = "annotation_ML_coarse"
for i in range(0,len(timepoints)):
    aligned_umap_coord = pd.DataFrame( {'UMAP_1':aligned_mapper.embeddings_[i].T[0], 'UMAP_2':aligned_mapper.embeddings_[i].T[1], 
                                        'timepoint' :timepoints[i], 
                                        'cell_type':meta_list[i][annotation_class].values, 
                                        'cell_id' : meta_list[i].index.to_list()})
    all_timepoints.append(aligned_umap_coord)
    
umap_coords = pd.concat(all_timepoints)

# %%
import os

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/aligned_umaps_NMPs_v2/"
os.makedirs(figpath, exist_ok=True)

# %%
sns.relplot(
    data=umap_coords, x="UMAP_1", y="UMAP_2",
    col="timepoint", hue="cell_type",
    kind="scatter"
)

# plt.savefig(figpath + "aligned_umap_NMPs_timepoints.png")
# plt.savefig(figpath + "aligned_umap_NMPs_timepoints.pdf")

# %% [markdown]
# ### try with different hyper-parameters
#

# %%
anchor_dict = []
# # Parameter set 1: 
# max_k = 10
# frac_k = 0.05
# max_dist = 0.05
# use_metric = 'cosine'

# Parameter set 2: 
max_k = 20
frac_k = 0.05
max_dist = 0.05
use_metric = 'cosine'

annotation_class = "annotation_ML_coarse"

for i in range(len(timepoints)-1):
    Y = lsi_list[i] # train on previous timepoint "progenitor space"
    X = lsi_list[i+1] # for cells in next timepoint predict "progenitors"
    
    nbrs = NearestNeighbors(n_neighbors=1, #algorithm='ball_tree',
                           metric = use_metric).fit(Y)
    
    distances, indices = nbrs.kneighbors(X) # predict top progenitor for all cells
    
    neigh_distribution = np.concatenate(distances, axis = 0)
    neigh_indexes = np.concatenate(indices, axis =0)
    
    pairs = pd.DataFrame( {'neighbor':neigh_indexes ,'dist':neigh_distribution})
    pairs.reset_index(inplace = True)
    pairs.rename(columns ={'index':'cell_target'},inplace=True)
    
    # Grup by cell type (we'll find top anchors for each cell type)
    pairs['cell_type'] = meta_list[i+1][annotation_class].values
    df1 = pairs.groupby(['cell_type'])

    df2 = df1.apply(lambda x: x.sort_values(["dist"]))

    df3=df2.reset_index(drop=True)

    # keep the top neighbors for each cell type (NOTE some cells in t+1 will map to many cells in t)
    
    # For each progenitor in t keep only the cell in (t+1) with the smallest distance
    # Closest relative
    pairs_rank = df3.groupby('neighbor').head(1)
    
    #pairs_rank = df3.groupby('cell_type').head(max_k)
    
    # For each cell type we keep the top k prgenitor relations 
    pairs_rank = pairs_rank.groupby('cell_type').head(max_k)
    
    # filter any neighbor pair with distance larger than threshold
    pairs_rank = pairs_rank[pairs_rank['dist']<max_dist] 
    
    
    pairs_dict = {pairs_rank['neighbor'].values[j] :pairs_rank['cell_target'].values[j]  for j in range(pairs_rank.shape[0])}
    
    
    anchor_dict.append(pairs_dict)

# %%
aligned_mapper = umap.AlignedUMAP(metric="cosine",
                                    n_neighbors=10,
                                    alignment_regularisation=0.001, # strength of the anchors across timepoints, default 0.1
                                    alignment_window_size=7, # how far forward and backward across the datasets we look when doing alignment, defaut 5
                                    n_epochs=200,
                                    random_state=42,).fit(lsi_list, relations=anchor_dict)

all_timepoints = []
annotation_class = "annotation_ML_coarse"
for i in range(0,len(timepoints)):
    aligned_umap_coord = pd.DataFrame( {'UMAP_1':aligned_mapper.embeddings_[i].T[0], 'UMAP_2':aligned_mapper.embeddings_[i].T[1], 
                                        'timepoint' :timepoints[i], 
                                        'cell_type':meta_list[i][annotation_class].values, 
                                        'cell_id' : meta_list[i].index.to_list()})
    all_timepoints.append(aligned_umap_coord)
    
umap_coords = pd.concat(all_timepoints)

sns.relplot(
    data=umap_coords, x="UMAP_1", y="UMAP_2",
    col="timepoint", hue="cell_type",
    kind="scatter"
)

# %%
aligned_mapper = umap.AlignedUMAP(metric="cosine",
                                    n_neighbors=10,
                                    alignment_regularisation=0.001, # strength of the anchors across timepoints, default 0.1
                                    alignment_window_size=7, # how far forward and backward across the datasets we look when doing alignment, defaut 5
                                    n_epochs=200,
                                    random_state=42,).fit(lsi_list, relations=anchor_dict)

all_timepoints = []
annotation_class = "annotation_ML_coarse"
for i in range(0,len(timepoints)):
    aligned_umap_coord = pd.DataFrame( {'UMAP_1':aligned_mapper.embeddings_[i].T[0], 'UMAP_2':aligned_mapper.embeddings_[i].T[1], 
                                        'timepoint' :timepoints[i], 
                                        'cell_type':meta_list[i][annotation_class].values, 
                                        'cell_id' : meta_list[i].index.to_list()})
    all_timepoints.append(aligned_umap_coord)
    
umap_coords = pd.concat(all_timepoints)

sns.relplot(
    data=umap_coords, x="UMAP_1", y="UMAP_2",
    col="timepoint", hue="cell_type",
    kind="scatter"
)

# %%
sns.relplot(
    data=umap_coords, x="UMAP_1", y="UMAP_2",
    col="timepoint", hue="cell_type",
    kind="scatter"
)

plt.savefig(figpath + "aligned_umap_NMPs_timepoints.png")
plt.savefig(figpath + "aligned_umap_NMPs_timepoints.pdf")

# %% [markdown]
# ### These UMAPs do look better! Now, let's see how many NMP cells are there at each timepoint
#

# %%
# save the aligned_umap coordinates for all timepoints (note that TDR119reseq was excluded)
umap_coords.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/aligned_umap_coords_NMPs_v2.csv")

# %%
aligned_umap_coord.loc[:,["UMAP_1","UMAP_2"]].values

# %% [markdown]
# ## Subsetting the alignedUMAP coordinates to each timepoint adata

# %%
aligned_umap_coords = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/aligned_umap_coords_NMPs_v2.csv", index_col=0)
aligned_umap_coords

# %%
reversed_dict_timepoints

# %%
reversed_dict_timepoints["TDR126"]

# %%
aligned_umap_coords.timepoint.unique()

# %%
# # import the h5ad object for all cells across all timepoints
# multiome = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA.h5ad")
# multiome

# %%
# First, subset the dataset into multiome 
multiome_NMPs = multiome[multiome.obs_names.isin(aligned_umap_coords.cell_id)]
multiome_NMPs

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/aligned_umaps_NMPs_v2/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
multiome_NMPs.obs.dataset.unique()

for dataset in multiome_NMPs.obs.dataset.unique():
    # subset the multiome adata object for each dataset
    adata_sub = multiome_NMPs[multiome_NMPs.obs.dataset==dataset]
    
    # check which timepoint does the dataset correspond to
    timepoint_id = reversed_dict_timepoints[dataset]
    
    # subset the umap_coords
    umap_coords_sub = aligned_umap_coords[aligned_umap_coords.timepoint==timepoint_id]
    
    # transfer the umap_coords to the adata_sub.obsm
    adata_sub.obsm["X_umap_aligned"] = umap_coords_sub.loc[:,["UMAP_1","UMAP_2"]].values
    
    # sc.pl.embedding(adata_sub, basis="X_umap_aligned", color="annotation_ML_coarse", save=f"_{dataset}_aligned_umap.pdf")
    # sc.pl.embedding(adata_sub, basis="X_umap_aligned", color="annotation_ML_coarse", save=f"_{dataset}_aligned_umap.png", show=False)

# %%
multiome_NMPs.obs.dataset.unique()

for dataset in multiome_NMPs.obs.dataset.unique():
    # subset the multiome adata object for each dataset
    adata_sub = multiome_NMPs[multiome_NMPs.obs.dataset==dataset]
    
    # check which timepoint does the dataset correspond to
    timepoint_id = reversed_dict_timepoints[dataset]
    
    # subset the umap_coords
    umap_coords_sub = aligned_umap_coords[aligned_umap_coords.timepoint==timepoint_id]
    
    # transfer the umap_coords to the adata_sub.obsm
    adata_sub.obsm["X_umap_aligned"] = umap_coords_sub.loc[:,["UMAP_1","UMAP_2"]].values
    
    adata_sub.write_h5ad(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/{dataset}_NMPs.h5ad")
    print(dataset)

# %%

# %% [markdown]
# ## After manual curation of celltype annotation - based on individual UMAPs and integrated UMAPs
#
# - 1) curated "notochord" cells mis-labelled as "NMPs". Typically, these were isolated and further apart from the rest of the NMP clusters.
#
# - 2) identified "low_quality" cells that seem to be further apart from the main NMP cluster in the "integrated" UMAP. These are likely very vague celltypes and potentially confuse the pseudotime computation.

# %%
adata_sub.shape[0]

# %%
# step 1. for each "data_id", 1) import the subsetted adata (v1), 
# and 2) the curated annotation(df).
# step 2. transfer the annotations from the "manual_annotation" dataframe
# step 3. filter out "notochord" and "low_quality" cells
# step 4. save the objects with curated annotation ("manual_annotation")

celltypes_NMPs = ["NMPs", "tail_bud", "PSM", "somites", "fast_muscle",
                  "spinal_cord", "neural_posterior"]

# define the filepaths
adata_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/subsetted_from_integrated_object/"
annotation_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/manual_annotation_exCellxgene/"
output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"

for data_id in reversed_dict_timepoints.keys():
    # import the adata (subsetted)
    subset = sc.read_h5ad(adata_path + f"{data_id}_NMPs.h5ad")
    print("n_cells: ", str(subset.shape[0]))
    # import the manual annotation df
    df_anno = pd.read_csv(annotation_path + f"{data_id}_nmps_manual_annotation.txt", sep="\t", index_col=0)
    
    # transfer the manual_annotation
    subset.obs["manual_annotation"] = df_anno["manual_annotation"]
    
    # filter out the "notochord", and "low_quality" cells
    subset_filtered = subset[subset.obs.manual_annotation.isin(celltypes_NMPs)]
    print(subset_filtered)
    print("n_cells: ", str(subset_filtered.shape[0]))
    
    # add the "raw counts" back to the adata.X layer for GRN computation
    subset_filtered.X = subset_filtered.layers["counts"].copy()
    # save the output adata
    subset_filtered.write_h5ad(output_path + f"{data_id}_nmps_manual_annotation.h5ad")
    print(f"{data_id} is saved")


# %% [markdown]
# ### generate slurm commands for computing GRNs

# %%
# dictionary for data_id and timepoints
dict_datasets = {"0somites":"TDR126",
                 "5somites":"TDR127",
                 "10somites":"TDR128",
                 "15somites":"TDR118reseq",
                 "15somites-2":"TDR119reseq",
                 "20somites":"TDR125reseq",
                 "30somites":"TDR124reseq"}
# List of identifiers for your jobs
identifiers = dict_datasets.values()  # Add more identifiers as needed
# Base paths (assuming these don't change, else they can be added to the loop)
base_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/"
# Template for the command
command_template = """
sbatch compute_co_celltype_GRNs.sh \\
    {base_path}/09_NMPs_subsetted_v2/{sample_id}/ \\
    {base_path}/09_NMPs_subsetted_v2/{sample_id}_nmps_manual_annotation.h5ad \\
    {base_path}/02_cicero_processed/{data_id}_cicero/05_{data_id}_base_GRN_dataframe.parquet \\
    {sample_id} manual_annotation X_umap_aligned
"""
# Generate and print each command
for identifier in identifiers:
    command = command_template.format(base_path=base_path, 
                                      data_id=identifier,
                                      sample_id = identifier.replace("reseq",""))
    print(command)

# %%
timepoints

# %%
metadata_NMPs.timepoints.unique()

# %%
# subset the lsi dataframe
lsi_list = []
lsi_df_list = []
n_lsis = 40

for timepoint in timepoints:
    # subset the metadata first, to use the indices for integrated_lsi subsetting (as they match)
    df_meta = metadata_NMPs[metadata_NMPs.timepoints==timepoint]
    # subset the integrated_lsi using the indices
    df_lsi = integrated_lsi_NMPs[integrated_lsi_NMPs.index.isin(df_meta.index)]
    
    # add the lsi list
    lsi_df_list.append(df_lsi)
    
    # extract the lsi components
    X = df_lsi.values
    # subset for 2:n_lsis components (we exclude the first LSI as it's usually correlated to the seq.depth)
    lsi_list.append(X[:,1:n_lsis])


# %% [markdown]
# Based on the distribution of distances we can select the top % of cells to use as anchors between the two datasets. 

# %%
for m in meta_list:
    print(m.shape)

# %%
for lsi in lsi_list: 
    print(lsi.shape)

# %% [markdown]
# Merge PCA projections

# %% [markdown]
# ### Run Aligned UMAP

# %% [markdown]
# Create list of dictionaries 

# %%
anchor_dict = []
# # Parameter set 1: 
# max_k = 10
# frac_k = 0.05
# max_dist = 0.05
# use_metric = 'cosine'

# Parameter set 2: 
max_k = 20
frac_k = 0.05
max_dist = 0.05
use_metric = 'cosine'


for i in range(len(timepoints)-1):
    Y = lsi_list[i] # train on previous timepoint "progenitor space"
    X = lsi_list[i+1] # for cells in next timepoint predict "progenitors"
    
    nbrs = NearestNeighbors(n_neighbors=1, #algorithm='ball_tree',
                           metric = use_metric).fit(Y)
    
    distances, indices = nbrs.kneighbors(X) # predict top progenitor for all cells
    
    neigh_distribution = np.concatenate(distances, axis = 0)
    neigh_indexes = np.concatenate(indices, axis =0)
    
    pairs = pd.DataFrame( {'neighbor':neigh_indexes ,'dist':neigh_distribution})
    pairs.reset_index(inplace = True)
    pairs.rename(columns ={'index':'cell_target'},inplace=True)
    
    # Grup by cell type (we'll find top anchors for each cell type)
    pairs['cell_type'] = meta_list[i+1].manual_annotation.values
    df1 = pairs.groupby(['cell_type'])

    df2 = df1.apply(lambda x: x.sort_values(["dist"]))

    df3=df2.reset_index(drop=True)

    # keep the top neighbors for each cell type (NOTE some cells in t+1 will map to many cells in t)
    
    # For each progenitor in t keep only the cell in (t+1) with the smallest distance
    # Closest relative
    pairs_rank = df3.groupby('neighbor').head(1)
    
    #pairs_rank = df3.groupby('cell_type').head(max_k)
    
    # For each cell type we keep the top k prgenitor relations 
    pairs_rank = pairs_rank.groupby('cell_type').head(max_k)
    
    # filter any neighbor pair with distance larger than threshold
    pairs_rank = pairs_rank[pairs_rank['dist']<max_dist] 
    
    
    pairs_dict = {pairs_rank['neighbor'].values[j] :pairs_rank['cell_target'].values[j]  for j in range(pairs_rank.shape[0])}
    
    
    anchor_dict.append(pairs_dict)

# %%
len(pairs_dict)

# %%
aligned_mapper = umap.AlignedUMAP(metric="cosine",
                                    n_neighbors=20,
                                    alignment_regularisation=0.1, # strength of the anchors across timepoints, default 0.1
                                    alignment_window_size=3, # how far forward and backward across the datasets we look when doing alignment, defaut 5
                                    n_epochs=200,
                                    random_state=42,).fit(lsi_list, relations=anchor_dict)

# %%
all_timepoints = []
for i in range(0,len(timepoints)):
    aligned_umap_coord = pd.DataFrame( {'UMAP_1':aligned_mapper.embeddings_[i].T[0], 'UMAP_2':aligned_mapper.embeddings_[i].T[1], 
                                        'timepoint' :timepoints[i], 
                                        'cell_type':meta_list[i]["manual_annotation"].values, 
                                        'cell_id' : meta_list[i].index.to_list()})
    all_timepoints.append(aligned_umap_coord)
    
umap_coords = pd.concat(all_timepoints)

# %%
import os

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/aligned_umaps/"
os.makedirs(figpath, exist_ok=True)

# %%
sns.relplot(
    data=umap_coords, x="UMAP_1", y="UMAP_2",
    col="timepoint", hue="cell_type",
    kind="scatter"
)

plt.savefig(figpath + "aligned_umap_NMPs_timepoints.png")
plt.savefig(figpath + "aligned_umap_NMPs_timepoints.pdf")

# %% [markdown]
# Load meta data

# %% [markdown]
# ### These UMAPs do look better! Now, let's see how many NMP cells are there at each timepoint
#

# %%
# save the aligned_umap coordinates for all timepoints (note that TDR119reseq was excluded)
umap_coords.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/aligned_umap_coords_NMPs.csv")

# %%

# %%
aligned_umap_coords = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/aligned_umap_coords_NMPs.csv", index_col=0)
aligned_umap_coords

# %%
umap_coords.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/aligned_umap_coords_all_timepoints.csv")

# %%
umap_coords

# %%
multiome

# %%
# import the h5ad object for all cells across all timepoints
multiome = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA.h5ad")
multiome

# %%
# First, subset the dataset into multiome 
multiome_NMPs = multiome[multiome.obs_names.isin(aligned_umap_coords.cell_id)]
multiome_NMPs

# %%
# transfer the UMAP, PCA, neighbors, and distances from the adata(seurat integrated) object to the adata_filtered
# # copy over the dim.reductions

# Convert to DataFrame for alignment
df_umap = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs.index)
df_pca = pd.DataFrame(adata.obsm["X_pca"], index=adata.obs.index)

# Reindex this DataFrame to match adata_filtered
adata_filtered.obsm['X_umap'] = df_umap.loc[adata_filtered.obs.index].values
adata_filtered.obsm['X_pca'] = df_pca.loc[adata_filtered.obs.index].values

# re-compute the nearest neighbors
sc.pp.neighbors(adata_filtered, n_neighbors=15, n_pcs=30)
adata_filtered

# adata_filtered.obsm["X_umap"] = adata.obsm["X_umap"]
# adata_filtered.obsm["X_pca"] = adata.obsm["X_pca"]
# adata_filtered.uns["neighbors"] = adata.uns["neighbors"]
# adata_filtered.obsp["distances"] = adata.obsp["distances"]

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
