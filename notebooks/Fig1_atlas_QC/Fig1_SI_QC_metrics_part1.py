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
# ## Figure_QC_metrics: 
#
# This notebook is to generate plots/figures for the QC metrics from the multiome sequencing.
# Some example metrics are:
# - number of UMIs/cell
# - number of genes/cell
# - % mitochondrial reads/cell
# - grouped.by datasets (or timepoints)

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
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/QC_metrics/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
adata = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_formatted_excxg_zscape_labels.h5ad")
adata

# %% [markdown]
# ## step 0. compute the RNA QC metrics

# %%
adata.var_names

# %%
adata.var_names[adata.var_names.str.startswith("rps")]

# %%
# compute the QC metrics (scanpy)
adata.X = adata.layers["counts"].copy()

# annotate the group of mitochondrial genes as "mt"
adata.var["mt"] = adata.var_names.str.startswith("mt")
adata.var["nc"] = adata.var_names.str.startswith("nc")
adata.var["ribo"] = adata.var_names.str.startswith("rps")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","nc","ribo"], 
                           percent_top=None, log1p=False, inplace=True)

# %% [markdown]
# ### import the annotation

# %%
annotation = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/annotations/annotation_ML_06052024.txt", index_col=0, sep="\t")
annotation

# %%
adata.obs["annotation_ML"] = annotation["annotation_ML_v4"]
adata.obs["annotation_ML"]

# %%
adata.obs.annotation_ML.unique()

# %%
# filter out the low quality cells (annotated as either "nan" or "low_quality_cells")
multiome = adata[adata.obs.annotation_ML!="low_quality_cells"]
multiome

# %% [markdown]
# ## Step 1. generate RNA QC plots (grouped by timepoints/datasets)

# %%
df = multiome.obs

# Custom order for the x-axis
custom_order = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR119', 'TDR125', 'TDR124']


# %%
df.columns

# %%
# Custom order for the x-axis
custom_order = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR119', 'TDR125', 'TDR124']

# Generate colors using the viridis colormap
viridis = plt.cm.get_cmap('viridis', 6)  # Get 6 colors from the viridis colormap

# Assign colors to the datasets, with TDR118 and TDR119 sharing the same color
custom_colors = [viridis(0), viridis(1), viridis(2), viridis(3), viridis(3), viridis(4), viridis(5)]

# Set up the plot
plt.figure(figsize=(4, 5))

ax = sns.boxplot(data=df, x="dataset", y="nCount_RNA", palette=custom_colors, order=custom_order, showfliers=False)
ax.grid(False)
plt.xticks(rotation=90)


# %%
# Set up the subplots
fig, axes = plt.subplots(2, 3, figsize=(7, 5), sharey=False)

# List of columns to plot
columns_to_plot = ["total_counts","n_genes_by_counts",
                   "pct_counts_mt",#"pct_counts_ribo",
                   "TSS_enrichment","nCount_peaks_integrated", "nFeature_peaks_integrated"]

# Titles for the subplots
#titles = ["Number of Features (RNA)", "Number of Peaks Integrated", "Number of Features Peaks Integrated"]

# Y-axis limits for each subplot
y_limits = [(0, 13000), (0, 12000), (0, 20), 
            (2.5, 7.5), (0, 60000), (0, 60000)]  # Adjust these limits based on your data

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through the columns and create a boxplot for each
for ax, column, y_lim in zip(axes, columns_to_plot, y_limits):
    sns.boxplot(data=df, x="dataset", y=column, palette=custom_colors, order=custom_order, showfliers=False, ax=ax)
    # ax.set_title(title)
    ax.set_xlabel('Dataset')
    ax.set_ylabel(column)
    ax.set_ylim(y_lim)  # Set y-axis limits
    ax.grid(False)
    # ax.axhline(y=500, color='r', linestyle='--')  # Add a horizontal line at y=500

# Rotate x-axis labels
for ax in axes:
    plt.sca(ax)
    plt.xticks(rotation=90)

# Adjust layout
plt.tight_layout()

plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/QC_metrics/QC_metrics_RNA_ATAC.pdf")
plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/QC_metrics/QC_metrics_RNA_ATAC.png")


# Show the plot
plt.show()

# %%
