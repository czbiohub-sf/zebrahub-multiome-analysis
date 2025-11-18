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
# # Figure 1. notebook to generate UMAP plots with different categories
#
# - last updated: 6/5/2024

# %% [markdown]
# # Set up - import libraries, datasets, and annotations

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re

# %%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
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


# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/umaps/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
multiome = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
multiome

# %%
# Generate a color palette with 33 distinct colors
# Start with the seaborn colorblind palette and extend it
base_palette = sns.color_palette("Set3", 12)
extended_palette = sns.color_palette("Set1", 9) + sns.color_palette("Pastel2", 8) + sns.color_palette("Dark2", 8)

# Combine the base and extended palettes to get 33 unique colors
combined_palette = base_palette + extended_palette

# manually swap some colors from the front with the ones from the back
teal_color = (0.0, 0.5019607843137255, 0.5019607843137255)  # RGB for teal
combined_palette[-1] = teal_color  # Replace the light yellow with teal

combined_palette[1] = combined_palette[-1]
combined_palette[17] = combined_palette[-3]
combined_palette[19] = combined_palette[-4]
combined_palette[32] = (213/256, 108/256, 85/256)
combined_palette[7] = (0.875, 0.29296875, 0.609375)
combined_palette[25] = (0.75390625, 0.75390625, 0.0)
combined_palette[21] = (0.22265625, 0.23046875, 0.49609375)

combined_palette = combined_palette[:33]  # Ensure we only take the first 33 colors

# Verify the palette length
assert len(combined_palette) == 33, "The palette must have exactly 33 colors"

# Plot the color palette
plt.figure(figsize=(10, 2))
sns.palplot(combined_palette)
plt.title("Combined Palette with 32 Unique Colors")
plt.show()

# Print the dictionary to verify
#print(cell_type_color_dict)

# %%
# extract a list of celltypes
cell_types = multiome.obs.annotation_ML_coarse.unique().tolist()
cell_types

# extract a list of colors (palette)
color_palettes = multiome.uns["annotation_ML_coarse_colors"]

# Create a dictionary mapping cell types to colors
cell_type_color_dict = {cell_type: combined_palette[i] for i, cell_type in enumerate(cell_types)}
cell_type_color_dict

# %%
# Plot the color palette
plt.figure(figsize=(10, 2))
sns.palplot(cell_type_color_dict.values())
plt.title("Combined Palette with 32 Unique Colors")
plt.show()

# %%
# visually inspect the colormap by sorting based on the HSV scale
from colorsys import rgb_to_hsv

# Convert RGB to HSV and sort by HSV values
sorted_palette = sorted(combined_palette, key=lambda color: rgb_to_hsv(color[0], color[1], color[2]))

# Visualize the sorted palette to manually check for similarity
# Plot the color palette
plt.figure(figsize=(10, 2))
sns.palplot(sorted_palette)
plt.title("Combined Palette with 32 Unique Colors")
plt.show()

# %%
# # Define different palettes to combine
# palettes = [
#     sns.color_palette("Set3", 14),
#     sns.color_palette("Paired", 10),
#     sns.color_palette("Dark2", 11),
#     sns.color_palette("Accent",5)
# ]

# # Combine and deduplicate colors
# alternative_palette = []
# seen_colors = set()

# for palette in palettes:
#     for color in palette:
#         if color not in seen_colors:
#             alternative_palette.append(color)
#             seen_colors.add(color)
#         if len(alternative_palette) >= 32:
#             break
#     if len(alternative_palette) >= 32:
#         break
        
# # Suppose the light yellow is the 10th color in the palette (9th index, as indexing is 0-based)
# # and you want to replace it with teal color
# teal_color = (0.0, 0.5019607843137255, 0.5019607843137255)  # RGB for teal
# alternative_palette[1] = teal_color  # Replace the light yellow with teal

# # Identify and replace similar light purple colors
# similar_purples_indices = [2,31]  # Example indices for similar purples
# replacement_colors = sns.color_palette("tab20", 3)

# for idx, replacement in zip(similar_purples_indices, replacement_colors):
#     alternative_palette[idx] = replacement

# # Assign colors to cell types
# # custom_palette = {cell_type: color for cell_type, color in zip(cell_types, combined_palette)}

# # Plot the color palette
# plt.figure(figsize=(10, 2))
# sns.palplot(alternative_palette)
# plt.title("Alternative Palette with 32 Unique Colors")
# plt.show()

# %% [markdown]
# ## generation of UMAPs
#
# ## ATAC modality

# %%
# UMAP (integrated_ATAC) colored by the cell types (annotation_ML_coarse)
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(multiome, basis="X_umap.atac",color=["annotation_ML_coarse"], palette=combined_palette,
               legend_fontsize=8,save="_integrated_ATAC_celltypes.pdf")

# %%
# UMAP (integrated_ATAC) colored by the cell types (annotation_ML_coarse)
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(multiome, basis="X_umap.atac",color=["annotation_ML_coarse"], palette=combined_palette,
               legend_fontsize=8,save="_integrated_ATAC_celltypes.png")

# %%
# shuffle the cell ordering in the adata (and make a copied adata object just fot plotting)
np.random.seed(42)
shuffled_indices = np.random.permutation(multiome.n_obs)

adata_shuffled = multiome[shuffled_indices].copy()
adata_shuffled.obsm["X_umap.atac"] = adata.obsm["X_umap.atac"][shuffled_indices]

# %%
# UMAP (integrated_ATAC) colored with batch
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(adata_shuffled, basis="X_umap.atac", color=["dataset"],
               legend_fontsize=8,save="_integrated_ATAC_batch.pdf")

# %%
# UMAP (integrated_ATAC) colored with batch
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(adata_shuffled, basis="X_umap.atac", color=["dataset"],
               legend_fontsize=8,save="_integrated_ATAC_batch.png")

# %%
dict_dev_stages = {
    "TDR118":"15somites",
    "TDR119":"15somites",
    "TDR124":"30somites",
    "TDR125":"20somites",
    "TDR126":"0somites",
    "TDR127":"5somites",
    "TDR128":"10somites"
}

# %%
adata_shuffled.obs["dev_stage"] = adata_shuffled.obs["dataset"].map(dict_dev_stages)
adata_shuffled.obs["dev_stage"]

# %%
# colormap
# Define the timepoints
timepoints = ["0somites", "5somites", "10somites", "15somites", "20somites", "30somites"]

# Load the "viridis" colormap
viridis = plt.cm.get_cmap('viridis', 256)

# Select a subset of the colormap to ensure that "30 somites" is yellow
# You can adjust the start and stop indices to shift the colors
start = 50
stop = 256
colors = viridis(np.linspace(start/256, stop/256, len(timepoints)))

# Create a dictionary to map timepoints to colors
color_dict = dict(zip(timepoints, colors))
color_dict

# %%
# UMAP (integrated_ATAC) colored with dev_stage
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(adata_shuffled, basis="X_umap.atac", color=["dev_stage"], palette=color_dict,
               legend_fontsize=8,save="_integrated_ATAC_dev_stage.pdf")

# %%
# UMAP (integrated_ATAC) colored with dev_stage
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(adata_shuffled, basis="X_umap.atac", color=["dev_stage"], palette=color_dict,
               legend_fontsize=8,save="_integrated_ATAC_dev_stage.png")

# %% [markdown]
# ## RNA modality

# %%
# UMAP (integrated_RNA) colored by the cell types (annotation_ML_coarse)
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(multiome, basis="X_umap.rna",color=["annotation_ML_coarse"], palette=combined_palette,
               legend_fontsize=8,save="_integrated_RNA_celltypes.pdf")

# %%
# UMAP (integrated_RNA) colored by the cell types (annotation_ML_coarse)
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(multiome, basis="X_umap.rna",color=["annotation_ML_coarse"], palette=combined_palette,
               legend_fontsize=8,save="_integrated_RNA_celltypes.png")

# %%
# UMAP (integrated_RNA) colored with batch
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(adata_shuffled, basis="X_umap.rna", color=["dataset"], sort_order=True,
               legend_fontsize=8,save="_integrated_rna_batch.pdf")

# %%
# UMAP (integrated_RNA) colored with batch
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(adata_shuffled, basis="X_umap.rna", color=["dataset"], sort_order=True,
               legend_fontsize=8,save="_integrated_rna_batch.png")

# %%
# UMAP (integrated_RNA) colored with "dev_stages"
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(adata_shuffled, basis="X_umap.rna", color=["dev_stage"], sort_order=True,
               legend_fontsize=8,save="_integrated_rna_dev_stages.pdf")

# %%
# UMAP (integrated_RNA) colored with "celltype annotation"
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(adata_shuffled, basis="X_umap.rna", color=["dev_stage"], sort_order=True,
               legend_fontsize=8,save="_integrated_rna_dev_stages.png")

# %% [markdown]
# # joint embedding (Weighted Nearest-Neighbors)

# %%
multiome

# %%
# UMAP (integrated_wnn) colored by the cell types (annotation_ML_coarse)
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(multiome, basis="X_wnn.umap",color=["annotation_ML_coarse"], palette=combined_palette,
               legend_fontsize=8,save="_integrated_wnn_celltypes.pdf")

# %%
# UMAP (integrated_wnn) colored by the cell types (annotation_ML_coarse)
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(multiome, basis="X_wnn.umap",color=["annotation_ML_coarse"], palette=combined_palette,
               legend_fontsize=8,save="_integrated_wnn_celltypes.png")

# %%
# UMAP (integrated_wnn) colored with batch
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(adata_shuffled, basis="X_wnn.umap", color=["dataset"], sort_order=True,
               legend_fontsize=8,save="_integrated_wnn_batch.pdf")

# %%
# UMAP (integrated_wnn) colored with batch
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(adata_shuffled, basis="X_wnn.umap", color=["dataset"], sort_order=True,
               legend_fontsize=8,save="_integrated_wnn_batch.png")

# %%
# UMAP (integrated_wnn) colored with "dev_stages"
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(adata_shuffled, basis="X_wnn.umap", color=["dev_stage"], sort_order=True,
               legend_fontsize=8,save="_integrated_wnn_dev_stages.pdf")

# %%
# UMAP (integrated_RNA) colored with "celltype annotation"
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
    sc.pl.embedding(adata_shuffled, basis="X_wnn.umap", color=["dev_stage"], sort_order=True,
               legend_fontsize=8,save="_integrated_wnn_dev_stages.png")

# %%
# add the combined_palette to the adata object (multiome)
multiome.uns["annotation_ML_coarse_colors"] = combined_palette

# %%
multiome.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")


# %%

# %% [markdown]
# # bar plots for the number of cells per timepoint/celltype

# %%
multiome.obs.dataset.value_counts()

# %%
dict_dev_stages = {
    "TDR118":"15somites",
    "TDR119":"15somites",
    "TDR124":"30somites",
    "TDR125":"20somites",
    "TDR126":"0somites",
    "TDR127":"5somites",
    "TDR128":"10somites"
}

# %%
color_dict

# %%
# Convert the Series to a DataFrame for seaborn
data_df = data.reset_index()
data_df.columns = ['Dataset', 'Count']

# Map the development stages to the dataset
data_df['Stage'] = data_df['Dataset'].map(dict_dev_stages)

# Ensure the dataset is in the desired order
data_df['Dataset'] = pd.Categorical(data_df['Dataset'], categories=[
    'TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR119', 'TDR125', 'TDR124'], ordered=True)

# Sort the DataFrame by the categorical order
data_df = data_df.sort_values('Dataset')

# Generate the color palette for the bar plot
palette = data_df['Stage'].map(color_dict).tolist()

# Plotting the bar plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Dataset', y='Count', data=data_df, palette=palette)

# Adding titles and labels
#plt.title('Dataset Value Counts by Development Stage')
plt.xlabel('dataset')
plt.ylabel('number of cells')

# Remove horizontal grid lines
plt.grid(False)
sns.despine()

# Display the plot
plt.xticks(rotation=0)

plt.savefig(figpath+"bar_plot_cell_numbers_datasets.png")
plt.savefig(figpath+"bar_plot_cell_numbers_datasets.pdf")

plt.show()


# %%

# %% [markdown]
# # Step 2. UMAPs from individual timepoints

# %%
dict_dev_stages = {
    "TDR118":"15somites",
    "TDR119":"15somites",
    "TDR124":"30somites",
    "TDR125":"20somites",
    "TDR126":"0somites",
    "TDR127":"5somites",
    "TDR128":"10somites"
}

# %%
# list of datasets (as they are saved in their filepaths
list_datasets = ["TDR126", "TDR127", "TDR128",
                 "TDR118reseq", "TDR119reseq",
                 "TDR125reseq", "TDR124reseq"]


# %%
plt.figure(figsize=(10, 2))
sns.palplot(multiome.uns["annotation_ML_coarse_colors"])
plt.title("Combined Palette with 32 Unique Colors")
plt.show()

# %%
# test cell
data_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"
dataset = "TDR126"
data = sc.read_h5ad(data_path + f"{dataset}/{dataset}_processed_RNA.h5ad")
data

# %%
list_datasets

# %%
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'differentiating_neurons': '#bebada',
    'endocrine_pancreas': '#fb8072',
    'endoderm': '#80b1d3',
    'enteric_neurons': '#fdb462',
    'epidermis': '#b3de69',
    'fast_muscle': '#df4b9b',
    'floor_plate': '#d9d9d9',
    'hatching_gland': '#bc80bd',
    'heart_myocardium': '#ccebc5',
    'hemangioblasts': '#ffed6f',
    'hematopoietic_vasculature': '#e41a1c',
    'hindbrain': '#377eb8',
    'lateral_plate_mesoderm': '#4daf4a',
    'midbrain_hindbrain_boundary': '#984ea3',
    'muscle': '#ff7f00',
    'neural': '#e6ab02',
    'neural_crest': '#a65628',
    'neural_floor_plate': '#66a61e',
    'neural_optic': '#999999',
    'neural_posterior': '#393b7f',
    'neural_telencephalon': '#fdcdac',
    'neurons': '#cbd5e8',
    'notochord': '#f4cae4',
    'optic_cup': '#c0c000',
    'pharyngeal_arches': '#fff2ae',
    'primordial_germ_cells': '#f1e2cc',
    'pronephros': '#cccccc',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}


# %%
sc.pl.embedding(data, basis=key, color="annotation_ML_coarse", palette=cell_type_color_dict)

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/umaps_individual_datasets/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
list_datasets

# %%
data_id

# %%
# define the master data filepath
data_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"

for dataset in list_datasets:
    # strip off "reseq" for the data_id
    data_id = dataset.replace("reseq","")
    # import the dataset
    data = sc.read_h5ad(data_path + f"{dataset}/{data_id}_processed_RNA.h5ad")
    
    # transfer the annotations
    # subset the adata
    subset = multiome[multiome.obs["dataset"]==data_id]
    # correct indices by removing the appended '_X' or any such pattern
    subset.obs_names = subset.obs_names.str.replace(r'_[^_]*$', '', regex=True)
    
    # create a mapping from the subset to the original data based on index
    annotation_mapping = subset.obs['annotation_ML_coarse']
    
    # map the annotations to the 'data' object using the corrected indices
    # first, subset for the shared cell_ids
    data = data[data.obs_names.isin(subset.obs_names)]
    # this step assumes that indices in 'data' after modification are a subset of those in 'subset'
    data.obs['annotation_ML_coarse'] = data.obs_names.map(annotation_mapping)
    
    # check the embeddings
    print(data.obsm)
    # generate the UMAPs for each modality (note that we have to be careful with the color dict)
    for key in data.obsm_keys():
        sc.pl.embedding(data, basis=key, color="annotation_ML_coarse", palette=cell_type_color_dict,
                       save=f"_{data_id}_ML_coarse.png")
        
        
    

# %%
aligned_umap_coords = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/aligned_umap_coords_all_timepoints.csv", index_col=0)
aligned_umap_coords

aligned_umap_coords.set_index("cell_id", inplace=True)
aligned_umap_coords

# %%
aligned_umap_coords.timepoint.unique()

# %%
dict_timepoints = {'TDR118': '15somites',
 'TDR119': '15somites-2',
 'TDR124': '30somites',
 'TDR125': '20somites',
 'TDR126': '0somites',
 'TDR127': '5somites',
 'TDR128': '10somites'}

# %%
# go through each dataset, and generate the UMAP by subsetting the aligned_umap_coords by the timepoint
for dataset in list_datasets:
    # strip off "reseq" for the data_id
    data_id = dataset.replace("reseq","")
    # import the dataset
    data = sc.read_h5ad(data_path + f"{dataset}/{data_id}_processed_RNA.h5ad")
    
    # transfer the annotations
    # subset the adata
    subset = multiome[multiome.obs["dataset"]==data_id]
    # correct indices by removing the appended '_X' or any such pattern
    subset.obs_names = subset.obs_names.str.replace(r'_[^_]*$', '', regex=True)
    
    # create a mapping from the subset to the original data based on index
    annotation_mapping = subset.obs['annotation_ML_coarse']
    
    # map the annotations to the 'data' object using the corrected indices
    # first, subset for the shared cell_ids
    data = data[data.obs_names.isin(subset.obs_names)]
    # this step assumes that indices in 'data' after modification are a subset of those in 'subset'
    data.obs['annotation_ML_coarse'] = data.obs_names.map(annotation_mapping)
    
    # subset the umap_coords
    time_key = data_id
    time_value = dict_timepoints[data_id]
    
    umap_coords = aligned_umap_coords[aligned_umap_coords.timepoint==time_value]
    data.obsm["X_aligned_umap"] = umap_coords[["UMAP_1","UMAP_2"]].values
    
    # check the embeddings
    print(data.obsm)
    # generate the UMAPs for each modality (note that we have to be careful with the color dict)
    # for key in data.obsm_keys():
    #     sc.pl.embedding(data, basis=key, color="annotation_ML_coarse", palette=cell_type_color_dict,
    #                    save=f"_{data_id}_ML_coarse.png")
    sc.pl.embedding(data, basis="X_aligned_umap", color="annotation_ML_coarse", palette=cell_type_color_dict,
                       save=f"_{data_id}_ML_coarse_aligned.pdf")
    sc.pl.embedding(data, basis="X_aligned_umap", color="annotation_ML_coarse", palette=cell_type_color_dict,
                       save=f"_{data_id}_ML_coarse_aligned.png")
        

# %%
time_value = "15somites-2"

umap_coords = aligned_umap_coords[aligned_umap_coords.timepoint==time_value]
umap_coords

# %%
umap_coords[["UMAP_1","UMAP_2"]].values

# %%

# %% [markdown]
# ## marker gene expression (tissue-level)
#
# - last updated: 07/09/2024
#
# The goal is to denote that the marker gene expression highlights the tissue-level annotation.
# So, we will show the "averaged" expression of a set of marker genes for tissues (endoderm, neuro-ectoderm, mesoderm, and hematopoetic vasculature)
#

# %%
# list of marker genes (from Cellxgene exploration)
markers_endo = ['fgfrl1b', 'col2a1a', 'ptprfa', 'emid1', 'nr5a2', 'ism2a', 'pawr', 'mmp15b', 'foxa3', 'onecut1']
markers_meso = ['msgn1', 'meox1', 'tbx6', 'tbxta', 'fgf8a', 'her1']
markers_neuro = ['pax7a', 'pax6a', 'pax6b', 'col18a1a', 'en2b', 'znf536', 'gpm6aa', 'gli3', 'chl1a']
markers_hemato = ['lmo2', 'etv2', 'tal1', 'sox17']

# %%
list_tissues_markers = [markers_endo, markers_meso, markers_neuro, markers_hemato]

for index, markers in enumerate(list_tissues_markers):
    print(index, markers)

# %%
list_tissues_markers = [markers_endo, markers_meso, markers_neuro, markers_hemato]
list_tissues_markers

list_tissues = ["endoderm", "mesoderm", "neuroecto", "hematopoetic"]

for index, tissue_markers in enumerate(list_tissues_markers):
    print(tissue_markers)
    # subset for the list of marker genes
    adata_sub = multiome[:,multiome.var_names.isin(tissue_markers)]

    adata_sub.X = adata_sub.layers["counts"].copy()

    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    
    # add the obs field to the main object (multiome)
    tissue = list_tissues[index]
    multiome.obs[f"avg_exp_rna_{tissue}"] = np.mean(adata_sub.X.todense(),axis=1)
    with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
        sc.pl.embedding(multiome, basis="X_wnn.umap", color=f"avg_exp_rna_{tissue}", save=f"_{tissue}_markers_rna.png", vmin=1, vmax=6.5)

# %%
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(multiome, basis="X_wnn.umap", color=f"avg_exp_rna_{tissue}", save=f"_{tissue}_markers_rna.pdf", vmin=1.5, vmax=6.5)

# %% [markdown]
# ### Gene.Activity object 

# %%
# import the multiome object where the adata.X contains the "gene.activity" - aggregated fragment counts from the scATAC-seq
multiome_ga = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_gene_activity_raw_counts_master_filtered.h5ad")
multiome_ga

# %%
# log-normalize the "counts"
sc.pp.normalize_total(multiome_ga, target_sum=1e4)
sc.pp.log1p(multiome_ga)

# %%
list_tissues_markers = [markers_endo, markers_meso, markers_neuro, markers_hemato]
list_tissues_markers

list_tissues = ["endoderm", "mesoderm", "neuroecto", "hematopoetic"]

for index, tissue_markers in enumerate(list_tissues_markers):
    print(tissue_markers)
    # subset for the list of marker genes
    adata_sub = multiome_ga[:,multiome_ga.var_names.isin(tissue_markers)]

#     adata_sub.X = adata_sub.layers["counts"].copy()

#     sc.pp.normalize_total(adata_sub, target_sum=1e4)
#     sc.pp.log1p(adata_sub)
    
    # add the obs field to the main object (multiome)
    tissue = list_tissues[index]
    multiome_ga.obs[f"avg_exp_gene_activity_{tissue}"] = np.mean(adata_sub.X.todense(),axis=1)
    with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (600)}):
        sc.pl.embedding(multiome_ga, basis="X_wnn.umap", color=f"avg_exp_gene_activity_{tissue}", save=f"_{tissue}_markers_gene_activity.png",
                        vmin=0.2, vmax=1.6)

# %%
with plt.rc_context({"figure.figsize": (2, 2), "figure.dpi": (100)}):
    sc.pl.embedding(multiome_ga, basis="X_wnn.umap", color=f"avg_exp_gene_activity_{tissue}", save=f"_{tissue}_markers_gene_activity.pdf",
                    vmin=0.2, vmax=1.6)

# %% [markdown]
# ### This is the end of the notebook

# %%
# Calculate the count of each cell-type within each dataset
cell_type_counts = adata.obs.groupby(['dataset', 'global_annotation']).size().unstack(fill_value=0)

# Calculate the fraction of each cell-type within each dataset
cell_type_fractions = cell_type_counts.div(cell_type_counts.sum(axis=1), axis=0)

# %%
cell_type_fractions

# %%
# Prepare the DataFrame for plotting
data_for_plotting = cell_type_fractions.stack().reset_index()
data_for_plotting.columns = ['dataset', 'cell_type', 'fraction']

# %%
set_plotting_style()

# %%
# order of the datasets
order_datasets = ["TDR126","TDR127","TDR128","TDR118","TDR119","TDR125","TDR124"]

# Reorder the DataFrame according to your specific dataset order
cell_type_fractions = cell_type_fractions.reindex(order_datasets)

# Unique datasets and cell-types
datasets = cell_type_fractions.index.tolist()
cell_types = cell_type_fractions.columns.tolist()

# Initialize the bottom parameter for stacking
bottom = pd.Series([0] * len(datasets), index=datasets)

# Plot each cell-type fraction
plt.figure(figsize=(10, 6))
for cell_type in cell_types:
    plt.bar(datasets, cell_type_fractions[cell_type], bottom=bottom, 
            label=cell_type, color=custom_palette[cell_type])
    bottom += cell_type_fractions[cell_type]

plt.xticks(rotation=45)
plt.ylabel('Fraction of Cell Types')
plt.xlabel('Dataset')
plt.title('Fraction of Cell Types within Each Dataset')
plt.ylim(0, 1)  # Ensure y-axis maxes at 1
# Remove grid lines
plt.grid(False)
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(figpath + "frac_celltypes_timepoints.pdf")
plt.savefig(figpath + "frac_celltypes_timepoints.png")
plt.show()

# %%
cell_type_fractions['TDR118_TDR119_avg'] = cell_type_fractions.loc[['TDR118', 'TDR119']].mean()
cell_type_fractions

# %%
# Unique datasets and cell-types
datasets = cell_type_fractions.index.tolist()
cell_types = cell_type_fractions.columns.tolist()

# Initialize the bottom parameter for stacking
bottom = pd.Series([0] * len(datasets), index=datasets)

# Plot each cell-type fraction
plt.figure(figsize=(10, 6))
for cell_type in cell_types:
    # Use the get method to provide a default color if the key doesn't exist
    color = custom_palette.get(cell_type, "grey")  # Default color set to "grey"
    plt.bar(datasets, cell_type_fractions[cell_type],
            bottom=bottom, label=cell_type, color=color)
    bottom += cell_type_fractions[cell_type]

plt.xticks(rotation=45)
plt.ylabel('fraction of cell types')
plt.xlabel('developmental stages')
plt.title('Fraction of Cell Types within Each Dataset')
plt.ylim(0, 1)  # Ensure y-axis maxes at 1
# Remove grid lines
plt.grid(False)
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
# Replace 'figpath' with the actual path where you want to save the figures
plt.savefig(figpath + "frac_celltypes_dev_stages.pdf")
plt.savefig(figpath + "frac_celltypes_dev_stages.png")
plt.show()

# %%
# Assuming you've already defined an appropriate color palette
# For this example, let's reuse the unique extended_set2_palette for coloring

order_datasets = ["TDR126","TDR127","TDR128","TDR118","TDR119","TDR125","TDR124"]

# Reorder the DataFrame according to your specific dataset order
cell_type_counts = cell_type_counts.reindex(order_datasets)

# Plot
plt.figure(figsize=(12, 6))
bottom = None
for cell_type in cell_type_counts.columns:
    # Check if bottom has been initialized
    if bottom is None:
        bottom = [0] * cell_type_counts.shape[0]
    plt.bar(cell_type_counts.index, cell_type_counts[cell_type], 
            bottom=bottom, label=cell_type, color=custom_palette[cell_type])
    # Update bottom for the next cell type
    bottom = [left + height for left, height in zip(bottom, cell_type_counts[cell_type])]

plt.xticks(rotation=45)
plt.ylabel('Number of Cells')
plt.xlabel('Dataset')
plt.title('Number of Cells per Cell-Type within Each Dataset')
# Remove grid lines
plt.grid(False)
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(figpath + "num_celltypes_timepoints.pdf")
plt.savefig(figpath + "num_celltypes_timepoints.png")
plt.show()

# %%

# %%
