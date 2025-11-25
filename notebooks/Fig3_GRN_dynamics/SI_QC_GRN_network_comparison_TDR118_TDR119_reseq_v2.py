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
# ## A comparison of GRNs from biological replicates (TDR118 vs TDR119)
#
# NOTE that they are both from 15-somites stage, 16hpf.
#
# - Author: Yang-Joon Kim
# - last updated: 2/29/2024 (with re-sequenced datasets)
#
# ## Goals
# - (1) comparative analysis on the two GRNs from biological replicates - using the network topology metrics (i.e. degree centrality, etc.)
# - (2) [TBD] EDA on other metrics
# - (3) [TBD] EDA on the network motifs?

# %%
# 0. Import

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

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
# define the figure paths
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/GRN_benchmark_TDR118_TDR119_peaks_merged/"
os.makedirs(figpath, exist_ok=True)

# %% [markdown]
# ## Step 1. Import the GRN (Links object)

# %%
# import the GRNs (Links objects)
TDR118_GRN = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR118reseq/08_TDR118reseq_celltype_GRNs.celloracle.links")
TDR119_GRN = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR119reseq/08_TDR119reseq_celltype_GRNs.celloracle.links")


# %%
# syntax to explore the Links objects (GRNs)
TDR118_GRN.filtered_links["NMPs"].value_counts()

# %%
# syntax to explore the Links objects (GRNs)
TDR118_GRN.filtered_links["NMPs"].value_counts()

# %%
# extract the network topology scores (i.e. degree_centrality_all)
df_GRN1 = TDR118_GRN.merged_score
df_GRN2 = TDR119_GRN.merged_score

# %%
df_GRN1.cluster.unique()

# %%
# Check the correlation of "degree_centrality_all" metric 
# per gene between the same cell-types

# subset the dataframes for specific cell-type
celltype1 = "PSM"
celltype2 = "PSM"

df1 = df_GRN1[df_GRN1.cluster==celltype1]
df2 = df_GRN2[df_GRN2.cluster==celltype2]

# Step 1. Get a union of gene_names
gene_names = set(df1.index).union(df2.index)
len(gene_names)

# Step 2. Create a new dataframe with matching indices
new_df1 = df1[df1.index.isin(gene_names)]
new_df2 = df2[df2.index.isin(gene_names)]

# Step 3. Fill missing values with NaNs
new_df1 = new_df1.reindex(gene_names) #fill_value=0
new_df2 = new_df2.reindex(gene_names)

# Step 4. Create the zipped DataFrame
zipped_df = pd.DataFrame({'degree_centrality_all_df1': new_df1['degree_centrality_all'], 'degree_centrality_all_df2': new_df2['degree_centrality_all']})
zipped_df

# Step 5. Generate scatter plots, with Pearson correlation coeff.
plt.scatter(x=zipped_df.degree_centrality_all_df1,
            y=zipped_df.degree_centrality_all_df2)
plt.xlabel("TDR118: degree_centrality_" + celltype1)
plt.ylabel("TDR119: degree_centrality_" + celltype2)
plt.xlim([0, 0.2])
plt.ylim([0, 0.2])
plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
plt.yticks([0, 0.05, 0.1, 0.15, 0.2])
#plt.title("degree_centrality (gene)")
plt.grid(False)  # Disables gridlines for this plot


# Annotate the plot with the correlation coefficient
corr = zipped_df.degree_centrality_all_df1.corr(zipped_df.degree_centrality_all_df2,
                                                method = "pearson")
plt.annotate(f"Pearson Correlation: {corr:.2f}",
             xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12)

plt.savefig(figpath + "scatter_degree_centrality_TDR118_TDR119_PSM.pdf")
plt.savefig(figpath + "scatter_degree_centrality_TDR118_TDR119_PSM.png")

plt.show()

# %%
zipped_df.sort_values("degree_centrality_all_df1", ascending=False)

# %%
df_GRN1.cluster.unique()

# %% [markdown]
# ### Same embryo - different celltypes
#
# - this is to see the correlation between different celltypes within the same embryo
#

# %%
# Check the correlation of "degree_centrality_all" metric 
# per gene between different cell-types

# subset the dataframes for specific cell-type
celltype1 = "Muscle"
celltype2 = "Differentiating_Neurons"

# sample from the TDR118reseq (df_GRN1)
df1 = df_GRN1[df_GRN1.cluster==celltype1]
df2 = df_GRN1[df_GRN1.cluster==celltype2]

# Step 1. Get a union of gene_names
gene_names = set(df1.index).union(df2.index)
len(gene_names)

# Step 2. Create a new dataframe with matching indices
new_df1 = df1[df1.index.isin(gene_names)]
new_df2 = df2[df2.index.isin(gene_names)]

# Step 3. Fill missing values with 0
new_df1 = new_df1.reindex(gene_names) #fill_value=0
new_df2 = new_df2.reindex(gene_names)

# Step 4. Create the zipped DataFrame
zipped_df = pd.DataFrame({'degree_centrality_all_df1': new_df1['degree_centrality_all'], 'degree_centrality_all_df2': new_df2['degree_centrality_all']})
zipped_df

# Step 5. Generate scatter plots, with Pearson correlation coeff.
plt.scatter(x=zipped_df.degree_centrality_all_df1,
            y=zipped_df.degree_centrality_all_df2)
plt.xlabel("TDR118: degree_centrality_" + celltype1)
plt.ylabel("TDR118: degree_centrality_" + celltype2)
# plt.title("degree_centrality (gene)")
plt.xlim([0, 0.15])
plt.ylim([0, 0.15])
plt.xticks([0, 0.05, 0.1, 0.15])
plt.yticks([0, 0.05, 0.1, 0.15])

plt.grid(False)

# Annotate the plot with the correlation coefficient
corr = zipped_df.degree_centrality_all_df1.corr(zipped_df.degree_centrality_all_df2,
                                                method = "pearson")
plt.annotate(f"Pearson Correlation: {corr:.2f}",
             xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12)

plt.savefig(figpath + "scatter_degree_centrality_TDR118_Muscle_Diff_Neurons.pdf")
plt.savefig(figpath + "scatter_degree_centrality_TDR118_Muscle_Diff_Neurons.png")

plt.show()

# %% [markdown]
# ### Check the correlation between different celltypes from two biological replicates

# %%
# Check the correlation of "degree_centrality_all" metric 
# per gene between different cell-types

# subset the dataframes for specific cell-type
celltype1 = "Muscle"
celltype2 = "Differentiating_Neurons"

df1 = df_GRN1[df_GRN1.cluster==celltype1]
df2 = df_GRN2[df_GRN2.cluster==celltype2]

# Step 1. Get a union of gene_names
gene_names = set(df1.index).union(df2.index)
len(gene_names)

# Step 2. Create a new dataframe with matching indices
new_df1 = df1[df1.index.isin(gene_names)]
new_df2 = df2[df2.index.isin(gene_names)]

# Step 3. Fill missing values with 0
new_df1 = new_df1.reindex(gene_names) #fill_value=0
new_df2 = new_df2.reindex(gene_names)

# Step 4. Create the zipped DataFrame
zipped_df = pd.DataFrame({'degree_centrality_all_df1': new_df1['degree_centrality_all'], 'degree_centrality_all_df2': new_df2['degree_centrality_all']})
zipped_df

# Step 5. Generate scatter plots, with Pearson correlation coeff.
plt.scatter(x=zipped_df.degree_centrality_all_df1,
            y=zipped_df.degree_centrality_all_df2)
plt.xlabel("TDR118: degree_centrality_" + celltype1)
plt.ylabel("TDR119: degree_centrality_" + celltype2)
# plt.title("degree_centrality (gene)")
plt.xlim([0, 0.15])
plt.ylim([0, 0.15])
plt.xticks([0, 0.05, 0.1, 0.15])
plt.yticks([0, 0.05, 0.1, 0.15])

plt.grid(False)

# Annotate the plot with the correlation coefficient
corr = zipped_df.degree_centrality_all_df1.corr(zipped_df.degree_centrality_all_df2,
                                                method = "pearson")
plt.annotate(f"Pearson Correlation: {corr:.2f}",
             xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12)

plt.savefig(figpath + "scatter_degree_centrality_TDR118_Muscle_TDR119_Diff_Neurons.pdf")
plt.savefig(figpath + "scatter_degree_centrality_TDR118_Muscle_TDR119_Diff_Neurons.png")

plt.show()

# %%
zipped_df.sort_values("degree_centrality_all_df1", ascending=False)

# %%
zipped_df.sort_values("degree_centrality_all_df2", ascending=False)


# %% [markdown]
# ## Step 2. Compute the correlation coefficients between the same cell-types and different cell-types
#
# - correlation coefficients of different network metrics (across all genes, that are present in both GRNs - two biological replicates)

# %%
# define a function to compute the correlation of network_metrics (per gene) between two GRNs
# GRN1, GRN2: two GRNs (filtered Links object)
# celltype1, celltype2: cell-types
# network_metric: network topology metrics, i.e. degree_centrality_all

def compute_corr_betwn_GRNs(df_GRN1, df_GRN2, celltype1, celltype2, network_metric):
    df1 = df_GRN1[df_GRN1.cluster==celltype1]
    df2 = df_GRN2[df_GRN2.cluster==celltype2]

    # Step 1. Get a union of gene_names
    gene_names = set(df1.index).union(df2.index)
    len(gene_names)

    # Step 2. Create a new dataframe with matching indices
    new_df1 = df1[df1.index.isin(gene_names)]
    new_df2 = df2[df2.index.isin(gene_names)]

    # Step 3. Fill missing values with NaNs
    new_df1 = new_df1.reindex(gene_names) #fill_value=0
    new_df2 = new_df2.reindex(gene_names)

    # Step 4. Create the zipped DataFrame
    zipped_df = pd.DataFrame({'metric_df1': new_df1[network_metric], 'metric_df2': new_df2[network_metric]})
    zipped_df

#     # Step 5. Generate scatter plots, with Pearson correlation coeff.
#     plt.scatter(x=zipped_df.metric_df1,
#                 y=zipped_df.metric_df2)
#     plt.xlabel("TDR118: "+ network_metric + "_" + celltype1)
#     plt.ylabel("TDR119: "+ network_metric + "_" + celltype2)
#     plt.title(network_metric)


    # Annotate the plot with the correlation coefficient
    corr = zipped_df.metric_df1.corr(zipped_df.metric_df2,
                                    method = "pearson")
    return corr

# %% [markdown]
# ### Step 2-1. degree_centrality_all
#
# - This metric was used by Kamimoto et al., Cell Stem Cell, 2023. So, we will use this metric as our first-pass metric to compare the two GRNs.

# %%
# define the cell-types
# celltypes = ['Adaxial_Cells', 'Differentiating_Neurons', 'Endoderm',
#        'Epidermal', 'Lateral_Mesoderm', 'Muscle', 'NMPs',
#        'Neural_Anterior', 'Neural_Crest', 'Neural_Posterior', 'Notochord',
#        'PSM', 'Somites', 'unassigned']
celltypes = ['Adaxial_Cells', 'Differentiating_Neurons', 'Endoderm',
       'Epidermal', 'Lateral_Mesoderm', 'Muscle', 'NMPs',
       'Neural_Anterior', 'Neural_Crest', 'Neural_Posterior', 'Notochord',
       'PSM', 'Somites']

# define empty series to save the correlation coefficients
corr_same_celltypes =[]
corr_diff_celltypes = []


for ct1 in celltypes:
    for ct2 in celltypes:
        corr_coeff = compute_corr_betwn_GRNs(df_GRN1, df_GRN2, ct1, ct2, "degree_centrality_all")
        
        if ct1==ct2:
            corr_same_celltypes.append(corr_coeff)
        else:
            corr_diff_celltypes.append(corr_coeff)

# %%
# define the bin width
bin_width = 0.025

# Calculate the number of bins for each histogram
num_bins1 = int((max(corr_same_celltypes) - min(corr_same_celltypes)) / bin_width)
num_bins2 = int((max(corr_diff_celltypes) - min(corr_diff_celltypes)) / bin_width)

plt.figure()
sns.histplot(corr_same_celltypes, kde=True, bins=num_bins1, stat="count", label='same_celltypes', alpha=0.5)
sns.histplot(corr_diff_celltypes, kde=True, bins=num_bins2, stat="count", label='diff_celltypes', alpha=0.5)
# plt.hist(corr_same_celltypes, density=True, bins=num_bins1)
# plt.hist(corr_diff_celltypes, density=True, bins=num_bins2)
plt.xlim([0.2, 1])
plt.xlabel("Pearson correlation: GRN1/GRN2")
plt.ylabel("density")
plt.title("degree_centrality_all")
plt.legend()
plt.grid(False)

#plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/network_plots_TDR118/QC_corr_centrality_TDR118_TDR119_allCelltypes.pdf")
# plt.savefig(figpath + "QC_corr_centrality_all_TDR118_TDR119_allCelltypes.pdf")
# plt.savefig(figpath + "QC_corr_centrality_all_TDR118_TDR119_allCelltypes.png")
plt.show()

# %% [markdown]
# ### Notochord is one of the worse correlation coeff. Let's examine this

# %%
# Check the correlation of "degree_centrality_all" metric 
# per gene between different cell-types

# subset the dataframes for specific cell-type
celltype1 = "Notochord"
celltype2 = "Notochord"

df1 = df_GRN1[df_GRN1.cluster==celltype1]
df2 = df_GRN2[df_GRN2.cluster==celltype2]

# Step 1. Get a union of gene_names
gene_names = set(df1.index).union(df2.index)
len(gene_names)

# Step 2. Create a new dataframe with matching indices
new_df1 = df1[df1.index.isin(gene_names)]
new_df2 = df2[df2.index.isin(gene_names)]

# Step 3. Fill missing values with 0
new_df1 = new_df1.reindex(gene_names) #fill_value=0
new_df2 = new_df2.reindex(gene_names)

# Step 4. Create the zipped DataFrame
zipped_df = pd.DataFrame({'degree_centrality_all_df1': new_df1['degree_centrality_all'], 'degree_centrality_all_df2': new_df2['degree_centrality_all']})
zipped_df

# Step 5. Generate scatter plots, with Pearson correlation coeff.
plt.scatter(x=zipped_df.degree_centrality_all_df1,
            y=zipped_df.degree_centrality_all_df2)
plt.xlabel("TDR118: degree_centrality_" + celltype1)
plt.ylabel("TDR119: degree_centrality_" + celltype2)
# plt.title("degree_centrality (gene)")
plt.grid(False)


# Annotate the plot with the correlation coefficient
corr = zipped_df.degree_centrality_all_df1.corr(zipped_df.degree_centrality_all_df2,
                                                method = "pearson")
plt.annotate(f"Pearson Correlation: {corr:.2f}",
             xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12)

plt.show()

# %%
zipped_df.sort_values("degree_centrality_all_df1", ascending=False)

# %%
zipped_df.sort_values("degree_centrality_all_df2", ascending=False)

# %% [markdown]
# ### Step 2.2 Other degree metrics
#
# - degree_all
# - degree_in
# - degree_out
# - degree_centrality_all (used above)
# - degree_centrality_in
# - degree_centrality_out
# - eigenvector_centrality

# %%
# define a dictionary to save all the corr.coeff results
dict_corr_coeff = {}
dict_corr_coeff["degree_centrality_all_same"] = corr_same_celltypes
dict_corr_coeff['degree_centrality_all_diff'] = corr_diff_celltypes

# %%
# define the cell-types
celltypes = ['Adaxial_Cells', 'Differentiating_Neurons', 'Endoderm',
       'Epidermal', 'Lateral_Mesoderm', 'Muscle', 'NMPs',
       'Neural_Anterior', 'Neural_Crest', 'Neural_Posterior', 'Notochord',
       'PSM', 'Somites']

# define the degree_metrics
degree_metrics = ["degree_all","degree_in","degree_out",
                  "degree_centrality_all","degree_centrality_in",
                  "degree_centrality_out","eigenvector_centrality"]

# define an empty dictionary to save the list of corr.coeff (per metric)
dict_corr_coeff = {}
# dict_corr_coeff["degree_centrality_all_same"] = corr_same_celltypes
# dict_corr_coeff['degree_centrality_all_diff'] = corr_diff_celltypes

# For loop to go over all degree metrics
for metric in degree_metrics:
    # define empty series to save the correlation coefficients
    corr_same_celltypes =[]
    corr_diff_celltypes = []

    for ct1 in celltypes:
        for ct2 in celltypes:
            corr_coeff = compute_corr_betwn_GRNs(df_GRN1, df_GRN2, ct1, ct2, metric)

            if ct1==ct2:
                corr_same_celltypes.append(corr_coeff)
            else:
                corr_diff_celltypes.append(corr_coeff)
                
    # define the keys for the dictionary
    namekey_same = metric + "_same"
    namekey_diff = metric + "_diff"
    # save the corr.coeff. into the dictionary
    dict_corr_coeff[namekey_same] = corr_same_celltypes
    dict_corr_coeff[namekey_diff] = corr_diff_celltypes
    
    # generate plots (optional)
    # define the bin width
    bin_width = 0.025

    # Calculate the number of bins for each histogram
    num_bins1 = int((max(corr_same_celltypes) - min(corr_same_celltypes)) / bin_width)
    num_bins2 = int((max(corr_diff_celltypes) - min(corr_diff_celltypes)) / bin_width)

    plt.figure()
    sns.histplot(corr_same_celltypes, kde=True, bins=num_bins1, stat="density", label='same_celltypes')
    sns.histplot(corr_diff_celltypes, kde=True, bins=num_bins2, stat="density", label='diff_celltypes')
    plt.xlim([0.2, 1])
    plt.xlabel("Pearson correlation: GRN1/GRN2")
    plt.ylabel("density")
    plt.title(metric)
    plt.grid(False)
    plt.legend()

    plt.savefig(figpath + "QC_corr_" + metric + "_TDR118_TDR119_allCelltypes.pdf")
    plt.savefig(figpath + "QC_corr_" + metric + "_TDR118_TDR119_allCelltypes.png")
    plt.show()

# %% [markdown]
# ### Step 3. generate a boxplot showing the corr.coeff distribution for same/diff cell-types, for different metrics
#
#

# %%
# A dictionary that has the distributions of corr.coeff., from all metrics
data = dict_corr_coeff
metric_names = degree_metrics

# Extract metric names and categories
# metric_names = list(set(key.split('_')[0] for key in data.keys()))
categories = ['same', 'diff']

# Compute means and standard errors for each metric and category
means = {}
std_errors = {}
for metric in metric_names:
    means[metric] = [np.mean(data[f'{metric}_{category}']) for category in categories]
    std_errors[metric] = [np.std(data[f'{metric}_{category}']) / np.sqrt(len(data[f'{metric}_{category}'])) for category in categories]

# Create x-values for each metric
x_values = np.arange(len(metric_names))

# Create a scatter plot with grouped data points and error bars
fig, ax = plt.subplots()

for i, category in enumerate(categories):
    y_values = [np.mean(data[f'{metric}_{category}']) for metric in metric_names]
    error_bar = [np.std(data[f'{metric}_{category}']) / np.sqrt(len(data[f'{metric}_{category}'])) for metric in metric_names]
    ax.errorbar(x_values, y_values, yerr=error_bar, marker='o', linestyle='None', label=category)

ax.set_xlabel('degree metrics')
ax.set_ylabel('correlation coefficients \n (across genes)')
# ax.set_title('Mean Value with Standard Error (Grouped Scatter Plot with Error Bars)')
ax.set_xticks(x_values)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_ylim([0.45, 1])
ax.set_xticklabels(metric_names, rotation = 45)
ax.legend(loc="upper left")
ax.grid(False)

plt.tight_layout()
plt.savefig(figpath + "corr_coeff_all_degree_metrics_scatter.pdf")
plt.savefig(figpath + "corr_coeff_all_degree_metrics_scatter.png")
plt.show()

# %% [markdown]
# ## NOTES (updated as of 2/29/2024)
#
# - The correlation became worse than the pilot study, possibly because we are now recovering more accurate GRNs. The difference from this round of datasets (re-sequenced) and the previous round is the following:
#     - (1) since we did re-sequencing (for deeper sequencing depth), lowly expressed genes would be recovered (i.e. myf5).
#     - (2) the GRNs were re-computed from "peaks_merged", which are narrower than the Cellranger-arc called peaks.
#

# %% [markdown]
# ## Comparison between two timepoints
#
# Next, we wanted to see how variable the GRNs are across the timepoints. For this, we wanted to compare two timepoints.
# There are many ways that we can pick the pairs of (t1, t2), but we can start with the folloiwngs.
#
# - 1) (15 somites, 0 somites) = (TDR118, TDR126): mid and early timepoints. This way, we can see the variability between biological replicates and different timepoints.
# - 2) (15 somites, 30 somites) = (TDR118, TDR124): mid and late timepoints. This way, we can see the variability between biological replicates and different timepoints.
# - 3) (0 somites, 30 somites) = (TDR126, TDR124): early and late timepoints.
#

# %%
# TDR126 (10hpf, 0 somites = budstage)
TDR126_GRN = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR126/08_TDR126_celltype_GRNs.celloracle.links")

# TDR124 (24hpf, 30 somites)
TDR124_GRN = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR124reseq/08_TDR124reseq_celltype_GRNs.celloracle.links")

# %%
# First, (15 somites, 0 somites) = (TDR118, TDR126)
df_GRN1 = TDR118_GRN.merged_score
df_GRN2 = TDR126_GRN.merged_score

# %%
# define a dictionary to save all the corr.coeff results
dict_corr_coeff = {}
dict_corr_coeff["degree_centrality_all_same"] = corr_same_celltypes
dict_corr_coeff['degree_centrality_all_diff'] = corr_diff_celltypes

# %%
celltypes = set(TDR126_GRN.links_dict.keys()).intersection(TDR118_GRN.links_dict.keys())
celltypes.remove("unassigned")
celltypes

# %%
# define the cell-types
# celltypes = ['Adaxial_Cells', 'Differentiating_Neurons', 'Endoderm',
#        'Epidermal', 'Lateral_Mesoderm', 'Muscle', 'NMPs',
#        'Neural_Anterior', 'Neural_Crest', 'Neural_Posterior', 'Notochord',
#        'PSM', 'Somites']
celltypes = set(TDR126_GRN.links_dict.keys()).intersection(TDR118_GRN.links_dict.keys())
celltypes.remove("unassigned")
celltypes

# define the degree_metrics
degree_metrics = ["degree_all","degree_in","degree_out",
                  "degree_centrality_all","degree_centrality_in",
                  "degree_centrality_out","eigenvector_centrality"]

# define an empty dictionary to save the list of corr.coeff (per metric)
dict_corr_coeff = {}
# dict_corr_coeff["degree_centrality_all_same"] = corr_same_celltypes
# dict_corr_coeff['degree_centrality_all_diff'] = corr_diff_celltypes

# For loop to go over all degree metrics
for metric in degree_metrics:
    # define empty series to save the correlation coefficients
    corr_same_celltypes =[]
    corr_diff_celltypes = []

    for ct1 in celltypes:
        for ct2 in celltypes:
            corr_coeff = compute_corr_betwn_GRNs(df_GRN1, df_GRN2, ct1, ct2, metric)

            if ct1==ct2:
                corr_same_celltypes.append(corr_coeff)
            else:
                corr_diff_celltypes.append(corr_coeff)
                
    # define the keys for the dictionary
    namekey_same = metric + "_same"
    namekey_diff = metric + "_diff"
    # save the corr.coeff. into the dictionary
    dict_corr_coeff[namekey_same] = corr_same_celltypes
    dict_corr_coeff[namekey_diff] = corr_diff_celltypes
    
    # generate plots (optional)
    # define the bin width
    bin_width = 0.025

    # Calculate the number of bins for each histogram
    num_bins1 = int((max(corr_same_celltypes) - min(corr_same_celltypes)) / bin_width)
    num_bins2 = int((max(corr_diff_celltypes) - min(corr_diff_celltypes)) / bin_width)

    plt.figure()
    sns.histplot(corr_same_celltypes, kde=True, bins=num_bins1, stat="density", label='same_celltypes')
    sns.histplot(corr_diff_celltypes, kde=True, bins=num_bins2, stat="density", label='diff_celltypes')
    plt.xlim([0.2, 1])
    plt.xlabel("Pearson correlation: GRN1/GRN2")
    plt.ylabel("density")
    plt.title(metric)
    plt.legend()

    #plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/network_plots_TDR118/QC_corr_centrality_TDR118_TDR119_allCelltypes.pdf")
    plt.savefig(figpath + "QC_corr_" + metric + "_TDR118_TDR126_15somites_0somites_allCelltypes.pdf")
    plt.savefig(figpath + "QC_corr_" + metric + "_TDR118_TDR126_15somites_0somites_allCelltypes.png")
    plt.show()

# %%
# A dictionary that has the distributions of corr.coeff., from all metrics
data = dict_corr_coeff
metric_names = degree_metrics

# Extract metric names and categories
# metric_names = list(set(key.split('_')[0] for key in data.keys()))
categories = ['same', 'diff']

# Compute means and standard errors for each metric and category
means = {}
std_errors = {}
for metric in metric_names:
    means[metric] = [np.mean(data[f'{metric}_{category}']) for category in categories]
    std_errors[metric] = [np.std(data[f'{metric}_{category}']) / np.sqrt(len(data[f'{metric}_{category}'])) for category in categories]

# Create x-values for each metric
x_values = np.arange(len(metric_names))

# Create a scatter plot with grouped data points and error bars
fig, ax = plt.subplots()

for i, category in enumerate(categories):
    y_values = [np.mean(data[f'{metric}_{category}']) for metric in metric_names]
    error_bar = [np.std(data[f'{metric}_{category}']) / np.sqrt(len(data[f'{metric}_{category}'])) for metric in metric_names]
    ax.errorbar(x_values, y_values, yerr=error_bar, marker='o', linestyle='None', label=category)

ax.set_xlabel('degree metrics')
ax.set_ylabel('correlation coefficients \n (across genes)')
# ax.set_title('Mean Value with Standard Error (Grouped Scatter Plot with Error Bars)')
ax.set_xticks(x_values)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_ylim([0.3, 1])
ax.set_xticklabels(metric_names, rotation = 45)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(figpath + "corr_coeff_TDR118_TDR126_15somites_0somites_all_degree_metrics.pdf")
plt.savefig(figpath + "corr_coeff_TDR118_TDR126_15somites_0somites_all_degree_metrics.png")
plt.show()

# %% [markdown]
# ### TDR118 vs TDR124 (15 somites vs 30 somites)

# %%
# First, (15 somites, 0 somites) = (TDR118, TDR126)
df_GRN1 = TDR118_GRN.merged_score
df_GRN2 = TDR124_GRN.merged_score

# %%
# define a dictionary to save all the corr.coeff results
dict_corr_coeff = {}
dict_corr_coeff["degree_centrality_all_same"] = corr_same_celltypes
dict_corr_coeff['degree_centrality_all_diff'] = corr_diff_celltypes

# %%
# celltypes = set(TDR124_GRN.links_dict.keys()).intersection(TDR118_GRN.links_dict.keys())
# celltypes.remove("unassigned")
# celltypes

# %%

# %%
# define the cell-types
# celltypes = ['Adaxial_Cells', 'Differentiating_Neurons', 'Endoderm',
#        'Epidermal', 'Lateral_Mesoderm', 'Muscle', 'NMPs',
#        'Neural_Anterior', 'Neural_Crest', 'Neural_Posterior', 'Notochord',
#        'PSM', 'Somites']
celltypes = set(TDR124_GRN.links_dict.keys()).intersection(TDR118_GRN.links_dict.keys())
celltypes.remove("unassigned")
celltypes

# define the degree_metrics
degree_metrics = ["degree_all","degree_in","degree_out",
                  "degree_centrality_all","degree_centrality_in",
                  "degree_centrality_out","eigenvector_centrality"]

# define an empty dictionary to save the list of corr.coeff (per metric)
dict_corr_coeff = {}
# dict_corr_coeff["degree_centrality_all_same"] = corr_same_celltypes
# dict_corr_coeff['degree_centrality_all_diff'] = corr_diff_celltypes

# For loop to go over all degree metrics
for metric in degree_metrics:
    # define empty series to save the correlation coefficients
    corr_same_celltypes =[]
    corr_diff_celltypes = []

    for ct1 in celltypes:
        for ct2 in celltypes:
            corr_coeff = compute_corr_betwn_GRNs(df_GRN1, df_GRN2, ct1, ct2, metric)

            if ct1==ct2:
                corr_same_celltypes.append(corr_coeff)
            else:
                corr_diff_celltypes.append(corr_coeff)
                
    # define the keys for the dictionary
    namekey_same = metric + "_same"
    namekey_diff = metric + "_diff"
    # save the corr.coeff. into the dictionary
    dict_corr_coeff[namekey_same] = corr_same_celltypes
    dict_corr_coeff[namekey_diff] = corr_diff_celltypes
    
    # generate plots (optional)
    # define the bin width
    bin_width = 0.025

    # Calculate the number of bins for each histogram
    num_bins1 = int((max(corr_same_celltypes) - min(corr_same_celltypes)) / bin_width)
    num_bins2 = int((max(corr_diff_celltypes) - min(corr_diff_celltypes)) / bin_width)

    plt.figure()
    sns.histplot(corr_same_celltypes, kde=True, bins=num_bins1, stat="density", label='same_celltypes')
    sns.histplot(corr_diff_celltypes, kde=True, bins=num_bins2, stat="density", label='diff_celltypes')
    plt.xlim([0, 1])
    plt.xlabel("Pearson correlation: GRN1/GRN2")
    plt.ylabel("density")
    plt.title(metric)
    plt.legend()

    #plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/network_plots_TDR118/QC_corr_centrality_TDR118_TDR119_allCelltypes.pdf")
    plt.savefig(figpath + "QC_corr_" + metric + "_TDR118_TDR124_15somites_30somites_allCelltypes.pdf")
    plt.savefig(figpath + "QC_corr_" + metric + "_TDR118_TDR124_15somites_30somites_allCelltypes.png")
    plt.show()

# %%
degree_metrics

# %%
data.keys()

# %%
data["degree_all_same"]

# %%
celltypes

# %%
df_GRN1[df_GRN1.cluster=="PSM"]

# %%
# Check the correlation of "degree_centrality_all" metric 
# per gene between the same cell-types

# subset the dataframes for specific cell-type
celltype1 = "Notochord"
celltype2 = "Notochord"

df1 = df_GRN1[df_GRN1.cluster==celltype1]
df2 = df_GRN2[df_GRN2.cluster==celltype2]

# Step 1. Get a union of gene_names
gene_names = set(df1.index).union(df2.index)
len(gene_names)

# Step 2. Create a new dataframe with matching indices
new_df1 = df1[df1.index.isin(gene_names)]
new_df2 = df2[df2.index.isin(gene_names)]

# Step 3. Fill missing values with NaNs
new_df1 = new_df1.reindex(gene_names) #fill_value=0
new_df2 = new_df2.reindex(gene_names)

# Step 4. Create the zipped DataFrame
zipped_df = pd.DataFrame({'degree_centrality_all_df1': new_df1['degree_centrality_all'], 'degree_centrality_all_df2': new_df2['degree_centrality_all']})
zipped_df

# Step 5. Generate scatter plots, with Pearson correlation coeff.
plt.scatter(x=zipped_df.degree_centrality_all_df1,
            y=zipped_df.degree_centrality_all_df2)
plt.xlabel("TDR118: degree_centrality_" + celltype1)
plt.ylabel("TDR119: degree_centrality_" + celltype2)
plt.title("degree_centrality (gene)")


# Annotate the plot with the correlation coefficient
corr = zipped_df.degree_centrality_all_df1.corr(zipped_df.degree_centrality_all_df2,
                                                method = "pearson")
plt.annotate(f"Pearson Correlation: {corr:.2f}",
             xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12)

# %%
# A dictionary that has the distributions of corr.coeff., from all metrics
data = dict_corr_coeff
metric_names = degree_metrics

# Extract metric names and categories
# metric_names = list(set(key.split('_')[0] for key in data.keys()))
categories = ['same', 'diff']

# Compute means and standard errors for each metric and category
means = {}
std_errors = {}
for metric in metric_names:
    means[metric] = [np.mean(data[f'{metric}_{category}']) for category in categories]
    std_errors[metric] = [np.std(data[f'{metric}_{category}']) / np.sqrt(len(data[f'{metric}_{category}'])) for category in categories]

# Create x-values for each metric
x_values = np.arange(len(metric_names))

# Create a scatter plot with grouped data points and error bars
fig, ax = plt.subplots()

for i, category in enumerate(categories):
    y_values = [np.mean(data[f'{metric}_{category}']) for metric in metric_names]
    error_bar = [np.std(data[f'{metric}_{category}']) / np.sqrt(len(data[f'{metric}_{category}'])) for metric in metric_names]
    ax.errorbar(x_values, y_values, yerr=error_bar, marker='o', linestyle='None', label=category)

ax.set_xlabel('degree metrics')
ax.set_ylabel('correlation coefficients \n (across genes)')
# ax.set_title('Mean Value with Standard Error (Grouped Scatter Plot with Error Bars)')
ax.set_xticks(x_values)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_ylim([0.3, 1])
ax.set_xticklabels(metric_names, rotation = 45)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(figpath + "corr_coeff_TDR118_TDR124_15somites_30somites_all_degree_metrics.pdf")
plt.savefig(figpath + "corr_coeff_TDR118_TDR124_15somites_30somites_all_degree_metrics.png")
plt.show()

# %%
dict_corr_coeff

# %%

# %%

# %%
