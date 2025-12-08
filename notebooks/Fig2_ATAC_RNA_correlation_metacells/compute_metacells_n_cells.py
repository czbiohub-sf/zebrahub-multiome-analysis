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
#     display_name: seacells
#     language: python
#     name: seacells
# ---

# %% [markdown]
# ## re-computing SEACells with different n_cells/metacell parameters
#
# - last updated: 11/11/2024
# - source: compute_seacells_atac_n_cells.py script

# %%
# import libraries
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import SEACells
import os

# plotting modules
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# %%
# %matplotlib inline

# %%
import sys
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/SEACells_metacell/")
from module_compute_seacells import * # import all functions

# Import from fig2_utils modules (refactored utilities)
from scripts.fig2_utils.metacell_utils import (
    plot_2D_modified,
    plot_SEACell_sizes_modified,
    compute_prevalent_celltype_per_metacell,
    plot_2D_with_metacells
)

# %%
# Some plotting aesthetics
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 300

# %% [markdown]
# ### start with one object, then make a for loop

# %%
#plot_2D_modified arguments (filepaths, dim.reduction, annotation, etc.)
input_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_75cells_per_metacell_integrated_lsi/"
output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_30cells_per_metacell/"
data_id = "" # this will be looped into the for loop
annotation_class = "annotation_ML_coarse"
dim_reduction = "X_lsi"
#figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/seacells_individual_datasets_30cells/"
#os.makedirs(figpath, exist_ok=True)
# add the n_cells (for the number of cells per SEACells)
n_cells = 30

# %%
# the list of datasets
list_datasets = ["TDR126","TDR127","TDR128",
                 "TDR118reseq","TDR119reseq","TDR125reseq","TDR124reseq"]


# %%
list_objects = os.listdir(input_path)
list_objects = [file for file in list_objects if "ML_coarse.h5ad" in file]
list_objects

# %%
# # test - loading the lsi and pca dataframes
# lsi = pd.read_csv(input_path + f"{data_id}/{data_name}_lsi.csv", index_col=0)
# pca = pd.read_csv(input_path + f"{data_id}/{data_name}_pca.csv", index_col=0)

# %% [markdown]
# ## recompute the SEACells (75 cells/metacell) using "X_lsi" from individual objects (not the "integrated_lsi")

# %%
output_path

# %%
input_path

# %%
seurat_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"


# %% [markdown]
# **Note:** SEACells plotting functions (`plot_2D_modified`, `plot_SEACell_sizes_modified`, etc.) are now imported from scripts.fig2_utils.metacell_utils

# %%
list_objects = ['TDR124reseq_seacells_annotation_ML_coarse.h5ad',
 'TDR118reseq_seacells_annotation_ML_coarse.h5ad',
 'TDR126_seacells_annotation_ML_coarse.h5ad',
 'TDR119reseq_seacells_annotation_ML_coarse.h5ad',
 'TDR125reseq_seacells_annotation_ML_coarse.h5ad']

# %%
input_path

# %%
# pick an example dataset
index=0
obj_id = "TDR124reseq_seacells_annotation_ML_coarse.h5ad"

# extract the data_id
data_name = list_objects[index].split("_")[0]
data_id = data_name.strip("reseq") # without "reseq"

# Part 1. load the old adata object
# import the adata object (pre-cleaned up with annotations)
ad = sc.read_h5ad(input_path + obj_id)
# save the old SEACell annotation done by "integrated_lsi"
ad.obs["SEACell_integrated_lsi"] = ad.obs["SEACell"]

# import the lsi (individually computed for each object by Seurat)
lsi = pd.read_csv(seurat_path + f"{data_name}/{data_id}_lsi.csv", index_col=0)
# filter the lsi components
lsi = lsi[lsi.index.isin(ad.obs_names)]
# replace the "X_lsi" with the lsi from the individual objects
# ad.obsm["X_lsi"] = lsi.values

# %%
# define the filepaths
output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_75cells_per_metacell/"
os.makedirs(output_path, exist_ok=True)
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/seacells_individual_datasets_75cells"
os.makedirs(figpath, exist_ok=True)

# number of cells
n_cells=75

for index, obj_id in enumerate(list_objects):
    
    # data_id
    data_name = list_objects[index].split("_")[0]
    data_id = data_name.strip("reseq") # without "reseq"
    
    # Part 1. load the old adata object
    # import the adata object (pre-cleaned up with annotations)
    ad = sc.read_h5ad(input_path + obj_id)
    # save the old SEACell annotation done by "integrated_lsi"
    ad.obs["SEACell_integrated_lsi"] = ad.obs["SEACell"]
    
    # import the lsi (individually computed for each object by Seurat)
    lsi = pd.read_csv(seurat_path + f"{data_name}/{data_id}_lsi.csv", index_col=0)
    # filter the lsi components
    lsi = lsi[lsi.index.isin(ad.obs_names)]
    # Reorder the lsi DataFrame to match the order of ad.obs_names
    lsi = lsi.loc[ad.obs_names]
    # replace the "X_lsi" with the lsi from the individual objects
    ad.obsm["X_lsi"] = lsi.values
    
    # Compute number of metacells
    n_metacells = np.floor(ad.n_obs/n_cells)
    
    # Set up and run SEACells
    model = SEACells.core.SEACells(
        ad,
        build_kernel_on='X_lsi',
        n_SEACells=n_metacells,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5
    )
    
    model.construct_kernel_matrix()
    M = model.kernel_matrix
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=100)
    model.plot_convergence()
    
    # Create visualization plot for this dataset
    fig, axs = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle(f"SEACells Results for {data_id}")

    # # Plot metacell sizes
    # sns.histplot(ad.obs.SEACell.value_counts(), ax=axs[0,0])
    # axs[0,0].set_xlabel("Cells per metacell")
    # axs[0,0].set_title("Metacell Size Distribution")
    # a histogram of the metacell sizes (number of cells per metacell)
    axs[0, 1].hist(ad.obs.SEACell.value_counts())
    axs[0, 1].set_xlabel("cell counts/metacell")
    axs[0, 1].set_ylabel("counts")

    # SEACElls model QC metric
    sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[1, 0])
    axs[1, 0].set_title(f'Non-trivial (> 0.1) assignments per cell')
    axs[1, 0].set_xlabel('# Non-trivial SEACell Assignments')
    axs[1, 0].set_ylabel('# Cells')
    # # Plot assignments
    # sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[0,1])
    # axs[0,1].set_title('Non-trivial Assignments per Cell')

    b = np.partition(model.A_.T, -5)    
    sns.heatmap(np.sort(b[:, -5:])[:, ::-1], cmap='viridis', vmin=0, ax=axs[1, 1])
    axs[1, 1].set_title('Strength of top 5 strongest assignments')
    axs[1, 1].set_xlabel('$n^{th}$ strongest assignment')

    # Plot UMAP with metacells if available
    # if 'X_umap.joint' in adata.obsm:
    plot_2D_modified(ad, ax=axs[2, 0], key="X_umap.joint", colour_metacells=False)
    # Plot the metacell assignments with coloring metacells
    plot_2D_modified(ad, ax=axs[2, 1], key="X_umap.joint", colour_metacells=True)
    # Plot the metacell sizes
    plot_SEACell_sizes_modified(ad, ax=axs[3, 0], bins=5)

    # Step 6. Quantifying the results (celltype_purity, compactness, etc.)
    # Compute the celltype purity
    SEACell_purity = SEACells.evaluate.compute_celltype_purity(ad, annotation_class)

    sns.boxplot(data=SEACell_purity, y=f'{annotation_class}_purity', ax=axs[3, 1])
    axs[3, 1].set_title('Celltype Purity')
    sns.despine(ax=axs[3, 1])

    # compute the compactness
    compactness = SEACells.evaluate.compactness(ad, dim_reduction)

    sns.boxplot(data=compactness, y='compactness', ax=axs[4, 0])
    axs[4, 0].set_title('Compactness')
    sns.despine(ax=axs[4, 0])

    # compute the separation
    separation = SEACells.evaluate.separation(ad, dim_reduction, nth_nbr=1)

    sns.boxplot(data=separation, y='separation', ax=axs[4, 1])
    axs[4, 1].set_title('Separation')
    sns.despine(ax=axs[4, 1])
    fig.tight_layout()
    # Save results
    plt.savefig(f"{figpath}/combined_plots_{data_id}_seacells_individual_lsi.pdf")
    plt.savefig(f"{figpath}/combined_plots_{data_id}_seacells_individual_lsi.png")
    ad.obs.to_csv(f"{output_path}/{data_id}_seacells.csv")
    # adata.write_h5ad(f"{output_path}/{data_id}_seacells_individual_lsi.h5ad")
    
    # Store results in dictionary
    # results[data_id] = {'adata': ad, 'model': model}
    
    print(f"Completed processing {data_id}")
    
    # del ad
    # Clear the current figure to free memory
    plt.close()

# %%
# plt.clf()  # Clear the current figure
# plt.close('all')  # Close all figures


# # Create visualization plot for this dataset
# fig, axs = plt.subplots(5, 2, figsize=(15, 20))
# fig.suptitle(f"SEACells Results for {data_id}")

# # # Plot metacell sizes
# # sns.histplot(ad.obs.SEACell.value_counts(), ax=axs[0,0])
# # axs[0,0].set_xlabel("Cells per metacell")
# # axs[0,0].set_title("Metacell Size Distribution")
# # a histogram of the metacell sizes (number of cells per metacell)
# axs[0, 1].hist(ad.obs.SEACell.value_counts())
# axs[0, 1].set_xlabel("cell counts/metacell")
# axs[0, 1].set_ylabel("counts")

# # SEACElls model QC metric
# sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[1, 0])
# axs[1, 0].set_title(f'Non-trivial (> 0.1) assignments per cell')
# axs[1, 0].set_xlabel('# Non-trivial SEACell Assignments')
# axs[1, 0].set_ylabel('# Cells')
# # # Plot assignments
# # sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[0,1])
# # axs[0,1].set_title('Non-trivial Assignments per Cell')

# b = np.partition(model.A_.T, -5)    
# sns.heatmap(np.sort(b[:, -5:])[:, ::-1], cmap='viridis', vmin=0, ax=axs[1, 1])
# axs[1, 1].set_title('Strength of top 5 strongest assignments')
# axs[1, 1].set_xlabel('$n^{th}$ strongest assignment')

# # Plot UMAP with metacells if available
# # if 'X_umap.joint' in adata.obsm:
# plot_2D_modified(ad, ax=axs[2, 0], key="X_umap.joint", colour_metacells=False)
# # Plot the metacell assignments with coloring metacells
# plot_2D_modified(ad, ax=axs[2, 1], key="X_umap.joint", colour_metacells=True)
# # Plot the metacell sizes
# plot_SEACell_sizes_modified(ad, ax=axs[3, 0], bins=5)
# plt.tight_layout()

# # Step 6. Quantifying the results (celltype_purity, compactness, etc.)
# # Compute the celltype purity
# SEACell_purity = SEACells.evaluate.compute_celltype_purity(ad, annotation_class)

# sns.boxplot(data=SEACell_purity, y=f'{annotation_class}_purity', ax=axs[3, 1])
# axs[3, 1].set_title('Celltype Purity')
# sns.despine(ax=axs[3, 1])

# # compute the compactness
# compactness = SEACells.evaluate.compactness(ad, dim_reduction)

# sns.boxplot(data=compactness, y='compactness', ax=axs[4, 0])
# axs[4, 0].set_title('Compactness')
# sns.despine(ax=axs[4, 0])

# # compute the separation
# separation = SEACells.evaluate.separation(ad, dim_reduction, nth_nbr=1)

# sns.boxplot(data=separation, y='separation', ax=axs[4, 1])
# axs[4, 1].set_title('Separation')
# sns.despine(ax=axs[4, 1])

# fig.tight_layout()
# # # Save results
# plt.savefig(f"{figpath}/combined_plots_{data_id}_seacells_individual_lsi.pdf")
# plt.savefig(f"{figpath}/combined_plots_{data_id}_seacells_individual_lsi.png")
# # ad.obs.to_csv(f"{output_path}/{data_id}_seacells.csv")

# %% [markdown]
# ### n_cells=30

# %%
# define the filepaths
output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_30cells_per_metacell/"
os.makedirs(output_path, exist_ok=True)
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/seacells_individual_datasets_30cells"
os.makedirs(figpath, exist_ok=True)

# number of cells
n_cells=30

for index, obj_id in enumerate(list_objects):
    
    # data_id
    data_name = list_objects[index].split("_")[0]
    data_id = data_name.strip("reseq") # without "reseq"
    
    # Part 1. load the old adata object
    # import the adata object (pre-cleaned up with annotations)
    ad = sc.read_h5ad(input_path + obj_id)
    # save the old SEACell annotation done by "integrated_lsi"
    ad.obs["SEACell_integrated_lsi"] = ad.obs["SEACell"]
    
    # import the lsi (individually computed for each object by Seurat)
    lsi = pd.read_csv(seurat_path + f"{data_name}/{data_id}_lsi.csv", index_col=0)
    # filter the lsi components
    lsi = lsi[lsi.index.isin(ad.obs_names)]
    # replace the "X_lsi" with the lsi from the individual objects
    ad.obsm["X_lsi"] = lsi.values
    
    # Compute number of metacells
    n_metacells = np.floor(ad.n_obs/n_cells)
    
    # Set up and run SEACells
    model = SEACells.core.SEACells(
        ad,
        build_kernel_on='X_lsi',
        n_SEACells=n_metacells,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5
    )
    
    model.construct_kernel_matrix()
    M = model.kernel_matrix
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=100)
    model.plot_convergence()
    
    # Create visualization plot for this dataset
    fig, axs = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle(f"SEACells Results for {data_id}")

    # # Plot metacell sizes
    # sns.histplot(ad.obs.SEACell.value_counts(), ax=axs[0,0])
    # axs[0,0].set_xlabel("Cells per metacell")
    # axs[0,0].set_title("Metacell Size Distribution")
    # a histogram of the metacell sizes (number of cells per metacell)
    axs[0, 1].hist(ad.obs.SEACell.value_counts())
    axs[0, 1].set_xlabel("cell counts/metacell")
    axs[0, 1].set_ylabel("counts")

    # SEACElls model QC metric
    sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[1, 0])
    axs[1, 0].set_title(f'Non-trivial (> 0.1) assignments per cell')
    axs[1, 0].set_xlabel('# Non-trivial SEACell Assignments')
    axs[1, 0].set_ylabel('# Cells')
    # # Plot assignments
    # sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[0,1])
    # axs[0,1].set_title('Non-trivial Assignments per Cell')

    b = np.partition(model.A_.T, -5)    
    sns.heatmap(np.sort(b[:, -5:])[:, ::-1], cmap='viridis', vmin=0, ax=axs[1, 1])
    axs[1, 1].set_title('Strength of top 5 strongest assignments')
    axs[1, 1].set_xlabel('$n^{th}$ strongest assignment')

    # Plot UMAP with metacells if available
    # if 'X_umap.joint' in adata.obsm:
    plot_2D_modified(ad, ax=axs[2, 0], key="X_umap.joint", colour_metacells=False)
    # Plot the metacell assignments with coloring metacells
    plot_2D_modified(ad, ax=axs[2, 1], key="X_umap.joint", colour_metacells=True)
    # Plot the metacell sizes
    plot_SEACell_sizes_modified(ad, ax=axs[3, 0], bins=5)

    # Step 6. Quantifying the results (celltype_purity, compactness, etc.)
    # Compute the celltype purity
    SEACell_purity = SEACells.evaluate.compute_celltype_purity(ad, annotation_class)

    sns.boxplot(data=SEACell_purity, y=f'{annotation_class}_purity', ax=axs[3, 1])
    axs[3, 1].set_title('Celltype Purity')
    sns.despine(ax=axs[3, 1])

    # compute the compactness
    compactness = SEACells.evaluate.compactness(ad, dim_reduction)

    sns.boxplot(data=compactness, y='compactness', ax=axs[4, 0])
    axs[4, 0].set_title('Compactness')
    sns.despine(ax=axs[4, 0])

    # compute the separation
    separation = SEACells.evaluate.separation(ad, dim_reduction, nth_nbr=1)

    sns.boxplot(data=separation, y='separation', ax=axs[4, 1])
    axs[4, 1].set_title('Separation')
    sns.despine(ax=axs[4, 1])
    fig.tight_layout()
    # Save results
    plt.savefig(f"{figpath}/combined_plots_{data_id}_seacells_individual_lsi.pdf")
    plt.savefig(f"{figpath}/combined_plots_{data_id}_seacells_individual_lsi.png")
    ad.obs.to_csv(f"{output_path}/{data_id}_seacells.csv")
    # adata.write_h5ad(f"{output_path}/{data_id}_seacells_individual_lsi.h5ad")
    
    # Store results in dictionary
    # results[data_id] = {'adata': ad, 'model': model}
    
    print(f"Completed processing {data_id}")
    
    # del ad
    # Clear the current figure to free memory
    plt.close()

# %%
# run for the rest two objects
list_objects = ['TDR128_seacells_annotation_ML_coarse.h5ad',
                 'TDR127_seacells_annotation_ML_coarse.h5ad']

# number of cells
n_cells=30

for index, obj_id in enumerate(list_objects):
    
    # data_id
    data_name = list_objects[index].split("_")[0]
    data_id = data_name.strip("reseq") # without "reseq"
    
    # Part 1. load the old adata object
    # import the adata object (pre-cleaned up with annotations)
    ad = sc.read_h5ad(input_path + obj_id)
    # save the old SEACell annotation done by "integrated_lsi"
    ad.obs["SEACell_integrated_lsi"] = ad.obs["SEACell"]
    
    # import the lsi (individually computed for each object by Seurat)
    lsi = pd.read_csv(seurat_path + f"{data_name}/{data_id}_lsi.csv", index_col=0)
    # filter the lsi components
    lsi = lsi[lsi.index.isin(ad.obs_names)]
    # replace the "X_lsi" with the lsi from the individual objects
    ad.obsm["X_lsi"] = lsi.values
    
    # Compute number of metacells
    n_metacells = np.floor(ad.n_obs/n_cells)
    
    # Set up and run SEACells
    model = SEACells.core.SEACells(
        ad,
        build_kernel_on='X_lsi',
        n_SEACells=n_metacells,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5
    )
    
    model.construct_kernel_matrix()
    M = model.kernel_matrix
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=100)
    model.plot_convergence()
    
    # Create visualization plot for this dataset
    fig, axs = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle(f"SEACells Results for {data_id}")

    # # Plot metacell sizes
    # sns.histplot(ad.obs.SEACell.value_counts(), ax=axs[0,0])
    # axs[0,0].set_xlabel("Cells per metacell")
    # axs[0,0].set_title("Metacell Size Distribution")
    # a histogram of the metacell sizes (number of cells per metacell)
    axs[0, 1].hist(ad.obs.SEACell.value_counts())
    axs[0, 1].set_xlabel("cell counts/metacell")
    axs[0, 1].set_ylabel("counts")

    # SEACElls model QC metric
    sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[1, 0])
    axs[1, 0].set_title(f'Non-trivial (> 0.1) assignments per cell')
    axs[1, 0].set_xlabel('# Non-trivial SEACell Assignments')
    axs[1, 0].set_ylabel('# Cells')
    # # Plot assignments
    # sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[0,1])
    # axs[0,1].set_title('Non-trivial Assignments per Cell')

    b = np.partition(model.A_.T, -5)    
    sns.heatmap(np.sort(b[:, -5:])[:, ::-1], cmap='viridis', vmin=0, ax=axs[1, 1])
    axs[1, 1].set_title('Strength of top 5 strongest assignments')
    axs[1, 1].set_xlabel('$n^{th}$ strongest assignment')

    # Plot UMAP with metacells if available
    # if 'X_umap.joint' in adata.obsm:
    plot_2D_modified(ad, ax=axs[2, 0], key="X_umap.joint", colour_metacells=False)
    # Plot the metacell assignments with coloring metacells
    plot_2D_modified(ad, ax=axs[2, 1], key="X_umap.joint", colour_metacells=True)
    # Plot the metacell sizes
    plot_SEACell_sizes_modified(ad, ax=axs[3, 0], bins=5)

    # Step 6. Quantifying the results (celltype_purity, compactness, etc.)
    # Compute the celltype purity
    SEACell_purity = SEACells.evaluate.compute_celltype_purity(ad, annotation_class)

    sns.boxplot(data=SEACell_purity, y=f'{annotation_class}_purity', ax=axs[3, 1])
    axs[3, 1].set_title('Celltype Purity')
    sns.despine(ax=axs[3, 1])

    # compute the compactness
    compactness = SEACells.evaluate.compactness(ad, dim_reduction)

    sns.boxplot(data=compactness, y='compactness', ax=axs[4, 0])
    axs[4, 0].set_title('Compactness')
    sns.despine(ax=axs[4, 0])

    # compute the separation
    separation = SEACells.evaluate.separation(ad, dim_reduction, nth_nbr=1)

    sns.boxplot(data=separation, y='separation', ax=axs[4, 1])
    axs[4, 1].set_title('Separation')
    sns.despine(ax=axs[4, 1])
    fig.tight_layout()
    # Save results
    plt.savefig(f"{figpath}/combined_plots_{data_id}_seacells_individual_lsi.pdf")
    plt.savefig(f"{figpath}/combined_plots_{data_id}_seacells_individual_lsi.png")
    ad.obs.to_csv(f"{output_path}/{data_id}_seacells.csv")
    # adata.write_h5ad(f"{output_path}/{data_id}_seacells_individual_lsi.h5ad")
    
    # Store results in dictionary
    # results[data_id] = {'adata': ad, 'model': model}
    
    print(f"Completed processing {data_id}")
    
    # del ad
    # Clear the current figure to free memory
    plt.close()

# %%

# %% [markdown]
# ## Plotting the UMAPs overlaid with metacells (different metacell resolutions)

# %% [markdown]
# ### plot the metacells on top of the single-cells

# %%
# reset the colors
# a color palette for the "coarse" grained celltype annotation ("annotation_ML_coarse")
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
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/seacells_individual_datasets_75cells/"

# %%
# define the list of datasets (data_id)
list_datasets = ['TDR126', 'TDR127', 'TDR128',
                'TDR118reseq', 'TDR125reseq', 'TDR124reseq']

metacell_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/"

# individual LSI (75cells/metacell)
for i, data_id in enumerate(list_datasets):
    # data_id = "TDR126"
    # Import the RNA and ATAC adata objects (aggregated over metacells)
    data_name = data_id.strip("reseq")
    # rna_meta_ad = sc.read_h5ad(metacell_path + f"objects_75cells_per_metacell_integrated_lsi/{data_id}/{data_name}_RNA_seacells_aggre.h5ad")
    # atac_meta_ad = sc.read_h5ad(metacell_path + f"objects_75cells_per_metacell_integrated_lsi/{data_id}/{data_name}_ATAC_seacells_aggre.h5ad")

    # import the RNA adata object (single-cell)
    rna_ad = sc.read_h5ad(metacell_path + f"objects_75cells_per_metacell_integrated_lsi/{data_id}_seacells_annotation_ML_coarse.h5ad")
    
    # import the metacells (SEACells)
    metacell = pd.read_csv(metacell_path + f"objects_75cells_per_metacell/{data_name}_seacells.csv", index_col=0)
    # replace the metacells column (rna_ad.obs)
    rna_ad.obs["SEACell"] = rna_ad.obs_names.map(metacell["SEACell"])
    
    # compute the most prevalent celltype for each metacell
    prevalent_celltypes = compute_prevalent_celltype_per_metacell(rna_ad, celltype_key="annotation_ML_coarse", 
                                                              metacell_key="SEACell")

    fig = plot_2D_with_metacells(rna_ad, key="X_umap.joint", hue="annotation_ML_coarse", palette=cell_type_color_dict)
    plt.savefig(figpath + f"umap_{data_id}_metacells_75cells_ind_lsi.png")
    plt.show()

# %%
# change the figure path
figpath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/seacells_individual_datasets_30cells/'

# %%
# define the list of datasets (data_id)
list_datasets = ['TDR126', 'TDR127', 'TDR128',
                'TDR118reseq', 'TDR125reseq', 'TDR124reseq']

metacell_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/"

# individual LSI (30cells/metacell)
for i, data_id in enumerate(list_datasets):
    
    # data_id = "TDR126"
    # Import the RNA and ATAC adata objects (aggregated over metacells)
    data_name = data_id.strip("reseq")
    # import the RNA adata object (single-cell)
    rna_ad = sc.read_h5ad(metacell_path + f"objects_75cells_per_metacell/{data_id}_seacells_annotation_ML_coarse.h5ad")
    
    # import the metacells (SEACells)
    metacell = pd.read_csv(metacell_path + f"objects_30cells_per_metacell/{data_name}_seacells.csv", index_col=0)
    # replace the metacells column (rna_ad.obs)
    rna_ad.obs["SEACell"] = rna_ad.obs_names.map(metacell["SEACell"])
    
    # # compute the most prevalent celltype for each metacell
    # prevalent_celltypes = compute_prevalent_celltype_per_metacell(rna_ad, celltype_key="annotation_ML_coarse", 
    #                                                           metacell_key="SEACell")

    fig = plot_2D_with_metacells(rna_ad, key="X_umap.joint", hue="annotation_ML_coarse", palette=cell_type_color_dict)
    plt.savefig(figpath + f"umap_{data_id}_metacells_30cells_ind_lsi.png")
    plt.show()

# %%
SEACells.core.
