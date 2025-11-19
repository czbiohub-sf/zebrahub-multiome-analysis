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
# ## chromatin accessbility and RNA dynamics (correlation)
#
# - Last updated: 6/14/2024
# - Author: Yang-Joon Kim
#
# - How the RNA expression/chromatin accessibility change over time (0 somites to 30 somites)
# - There are multiple ways to look at this. we'll summarize our attempts in EDA here.
#
#
# ### feature level
# - **[RNA] gene expression (log-norm)
# - **[ATAC] individual peaks (600K)**
# - **[ATAC] gene.activity** (peaks were linked to each gene based on proximity to the promoter/TSS, or cicero-scores): there are a couple of choices - (1) Signac, (2) cicero, and (3) SEACells gene score. We'll choose **(1) Signac, and (2) cicero Gene.Activity score**, as we can compute these metrics per dataset (without integrating/merging peaks).
#
# ### cell grouping level (pseudo-bulk)
# - **Metacells (SEAcells)**
# - **all cells (per dataset)**
# - single cells
#
# We will try out a couple of combinations. Ultimately, we'd like to automate this process for any count matrices (however they were computed - metacells-by-gene.activity, for example).
#
# - First, we decide on which count matrice (cells-by-features) we'll use. 
# - Second, we will prepare the count matrices by aggregating over either all cells or metaclles. 
# - Third, we will generate features-by-time matrix (there can be the third dimension of Metacells).
# - EDA1: cluster the matrix to see if there's a trend in terms of features over time (i.e. gene.activity going up, going down, etc.)
#
# - EDA2: 
#
#

# %%
# 0. Import
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# %%
# %matplotlib inline
from scipy import sparse
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import zscore

# %%
# figure parameter setting
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')

# Import utilities from refactored modules
from scripts.fig1_utils.plotting_utils import set_plotting_style
from scripts.fig1_utils.data_processing import replace_periods_with_underscores
from scripts.fig1_utils.normalization import normalize_minmax as normalize



# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/RNA_ATAC_dynamics/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %% [markdown]
# ## Step 0. Import datasets 
#
# - RNA
# - ATAC (Gene.Activity)

# %%
# import the adata to find N highly variable genes
adata_RNA = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
adata_RNA

# %%
adata_ATAC = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_gene_activity.h5ad")
adata_ATAC

# %%
# filter out the "low_quality_cells" - either using the adata_RNA cell_ids, or the latest annotation csv file
adata_ATAC = adata_ATAC[adata_ATAC.obs_names.isin(adata_RNA.obs_names)]
adata_ATAC

# %%
# remove unnecessary fields from adata_ATAC 
columns_to_drop = adata_ATAC.obs.columns[adata_ATAC.obs.columns.str.startswith("prediction")]
columns_to_drop

adata_ATAC.obs = adata_ATAC.obs.drop(columns=columns_to_drop)
adata_ATAC

# replace the "." within the strings in adata.obs.columns to "_" (this is for exCellxgene)
# Apply the function to obs
adata_ATAC.obs = replace_periods_with_underscores(adata_ATAC.obs)

# # Apply the function to var if needed
# adata_ATAC.var = replace_periods_with_underscores(adata_ATAC.var)

# # Apply the function to obsm if needed
# adata_ATAC.obsm = {replace_periods_with_underscores(pd.DataFrame(adata_ATAC.obsm[key])).columns[0]: adata.obsm[key] for key in adata.obsm.keys()}

# Verify the changes
print(adata_ATAC)

# %%
del adata_ATAC.raw

# %%
# save the reformatted and filtered adata_ATAC (gene.activity for the counts)
adata_ATAC.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_gene_activity_master_filtered.h5ad")

# %% [markdown]
# ## Step 1. RNA dynamics along the real-time (developmental stages)
#
# - Following the Calderon,...,Trapnell, Science, 2022's approach (Figure 1F), we will average the "log-norm" counts across cells for both RNA and ATAC.
#

# %%
np.sum(np.expm1(adata_RNA.X),1)

# %%
# since the adata_RNA.X layer is already log-normalized, we'll compute the average across cells for each timepoint
adata_RNA

# %%
# adata_RNA.X = adata_RNA.layers["counts"].copy()

# sc.pp.normalize_total(adata_RNA, target_sum=1e4)
# sc.pp.log1p(adata_RNA)

# %%
adata_RNA.X.shape
adata_RNA.var_names.shape

# %% [markdown]
# ### DEPRECATED - choosing the HVGs (to reduce the computational cost)

# %%
# # compute N highly variable genes
# N_top_genes = 3000
# sc.pp.highly_variable_genes(adata_RNA, layer="counts", 
#                             n_top_genes=N_top_genes, flavor="seurat_v3")

# %%
# extract the list of highly variable genes
list_hvg_RNA = adata_RNA.var_names[adata_RNA.var.highly_variable==True]

# check if some of the marker genes are present in the list of highly variable genes
print("myf5" in list_hvg_RNA)
print("meox1" in list_hvg_RNA)
print("sox2" in list_hvg_RNA)
print("tbxta" in list_hvg_RNA)

# %% [markdown]
# ## Plot the average gene expression dynamics (RNA) over real-time (averaged across all cells)

# %%
# genes_to_plot = ['tbxta', 'sox2', 'myf5', 'pax6a', 'meox1']
# define the gene
list_genes = ["myf5","sox2","tbxta","myog",
              "meox1","tbx16","hes6",'pax6a',"en2a",
              "hbbe1.1","hbae3","hbbe3","myl1"]

# %%
# subset the adata_RNA for the list of genes (this is to reduce the computing resource/time)
adata_RNA_sub = adata_RNA[:, adata_RNA.var_names.isin(list_genes)]
adata_RNA_sub

# %%
# create a dataframe of cells-by-gene.activity (Signac)
count_matrice = pd.DataFrame(adata_RNA_sub.X.todense(),
                             index=adata_RNA_sub.obs.index,
                             columns=adata_RNA_sub.var_names)
count_matrice.head()

# %%
# transfer the "dataset" category labels to count_matrice df
count_matrice["dataset"] = adata_RNA_sub.obs["dataset"]
count_matrice["annotation_ML_coarse"] = adata_RNA_sub.obs["annotation_ML_coarse"]
count_matrice["dev_stage"] = adata_RNA_sub.obs["dev_stage"]

# %%
# add the "timepoints" category for the column

# define the dictionary for mapping
dict_timepoints = {"TDR126":"0somites",
                   "TDR127":"5somites",
                   "TDR128":"10somites",
                   "TDR118":"15somites",
                   "TDR119":"15somites",
                   "TDR125":"20somites",
                   "TDR124":"30somites"}

# map the "dataset" to "timepoints"
count_matrice["timepoints"] = count_matrice["dataset"].map(dict_timepoints)
count_matrice.head()

# %%
count_matrice.timepoints.value_counts()

# %%
# compute the averaged gene expression

# Define numeric columns (gene expression columns) by selecting columns with float/int data types
numeric_columns = count_matrice.select_dtypes(include=[np.number]).columns.tolist()

# Compute the mean gene expression level across all cells per 'timepoints'
timepoints_by_genes = count_matrice.groupby('timepoints')[numeric_columns].mean()

# Display the result
print(timepoints_by_genes)

# %%
# compute the "standard errors" gene.activity across all cells per "timepoint" (TDR118 and TDR119 will be merged here)
timepoints_by_genes_sem = count_matrice.groupby('timepoints')[numeric_columns].sem()
timepoints_by_genes_sem.head()


# %%
# numeric_timepoints to re-order the rows (timepoints_by_genes)
timepoints_by_genes['numeric_timepoints'] = timepoints_by_genes.index.str.extract('(\d+)').astype(int).values
timepoints_by_genes['numeric_timepoints']

# Sort by the numeric timepoints to ensure correct order in plot
timepoints_by_genes_sorted = timepoints_by_genes.sort_values('numeric_timepoints')
timepoints_by_genes_sorted

# %%
# numeric_timepoints to re-order the rows (timepoints_by_genes_sem)
timepoints_by_genes_sem['numeric_timepoints'] = timepoints_by_genes_sem.index.str.extract('(\d+)').astype(int).values
timepoints_by_genes_sem['numeric_timepoints']

# Sort by the numeric timepoints to ensure correct order in plot
timepoints_by_genes_sem_sorted = timepoints_by_genes_sem.sort_values('numeric_timepoints')
timepoints_by_genes_sem_sorted

# %%
timepoints_by_genes_sorted.index

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/RNA_ATAC_dynamics/"

# %%
for gene in list_genes:
    # generate a plot
    plt.figure(figsize=(4, 3))
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 timepoints_by_genes_sorted[gene], 
                 timepoints_by_genes_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    # plt.title('Scatter Plot of log-normalized counts over Timepoints')
    plt.xlabel('time (somites)')
    plt.ylabel('averaged gene expression (RNA, log-norm)')
    plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(False)
    plt.savefig(figpath + f"RNA_timepoints_{gene}.pdf")
    plt.show()

# %%
# Plotting the gene.activity dynamics (0-30 somites)
# define the gene
gene = "myf5"

# generate a plot
plt.figure(figsize=(4, 3))
plt.errorbar(timepoints_by_genes_sorted.index, 
             timepoints_by_genes_sorted[gene], 
             timepoints_by_genes_sem_sorted[gene],
             fmt='o', capsize=2, linestyle='None', 
             label=gene, markersize=2, elinewidth=1, capthick=1)
# plt.title('Scatter Plot of log-normalized counts over Timepoints')
plt.xlabel('time (somites)')
plt.ylabel('averaged gene expression (RNA, log-norm)')
plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
plt.legend()
plt.grid(False)
# plt.savefig(figpath + "gene_activity_score_myf5.pdf")
plt.show()

# %%
# # Plotting the gene.activity dynamics (0-30 somites)
# # define the gene
# gene = "myf5"

# # generate a plot
# plt.figure(figsize=(4, 3))
# plt.errorbar([0, 5, 10, 15, 20, 30], 
#              timepoints_by_genes_sorted[gene], 
#              timepoints_by_genes_sem_sorted[gene],
#              fmt='o', capsize=5, linestyle='None', 
#              label=gene, markersize=8, elinewidth=2, capthick=2)
# # plt.title('Scatter Plot of log-normalized counts over Timepoints')
# plt.xlabel('time (somites)')
# plt.ylabel('averaged gene expression (RNA, log-norm)')
# # plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
# plt.xticks([0, 5, 10, 15, 20, 30])
# plt.legend()
# plt.grid(False)
# plt.savefig(figpath + "RNA_timepoints_myf5.pdf")
# plt.show()

# %%
# Plotting the gene.activity dynamics (0-30 somites)
# define the gene
list_genes = ["myf5","sox2","tbxta","myog",
              "meox1","tbx16","hes6",
              "hbbe1.1","hbae3","hbbe3"]

# generate a plot
plt.figure(figsize=(10, 6))

for gene in list_genes:
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 timepoints_by_genes_sorted[gene], 
                 timepoints_by_genes_sem_sorted[gene],
                 fmt='o', capsize=5, linestyle='None', 
                 label=gene, markersize=8, elinewidth=2, capthick=2)
    
# plt.title('Scatter Plot of log-normalized counts over Timepoints')
plt.xlabel('time (somites)')
plt.ylabel('averaged gene expression (RNA, log-norm)')
plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
# Move the legend outside of the plot to the right
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect parameter to make room for the legend
plt.grid(False)
# plt.savefig(figpath + "RNA_timepoints_genesets.pdf")
plt.show()

# %%
# Plotting the gene.activity dynamics (0-30 somites)
# define the gene
list_genes = ["myf5","sox2","tbxta","myog",
              "meox1","tbx16","hes6",
              "hbbe1.1","hbae3","hbbe3"]

# generate a plot
plt.figure(figsize=(10, 6))

for gene in list_genes:
    # Normalization step: divide by the max value to scale to 1
    normalized_values = timepoints_by_genes_sorted[gene] / timepoints_by_genes_sorted[gene].max()
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 normalized_values, 
                 yerr=timepoints_by_genes_sem_sorted[gene] / timepoints_by_genes_sorted[gene].max(), 
                 fmt='o', capsize=5, linestyle='None', 
                 label=gene, markersize=8, elinewidth=2, capthick=2)
    
# plt.title('Scatter Plot of log-normalized counts over Timepoints')
plt.xlabel('time (somites)')
plt.ylabel('averaged gene expression (RNA, normalized)')
plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
# Move the legend outside of the plot to the right
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect parameter to make room for the legend
plt.grid(False)
# plt.savefig(figpath + "gene_activity_score_genesets_normalized.pdf")
plt.show()

# %% [markdown]
# ## Step 2. ATAC dynamics using cells-by-gene.activity (Signac)
#
# - The gene.activity scores were computed by Signac for individual timepoints  (**using the "peaks_merged" for individual timepoints**, not "peaks_integrated" after the merge/integration over timepoints)
# - We created a concatenated adata (all cells-by-gene.activity, with adata.obs having "dataset" identifiers)
#

# %% [markdown]
# ### averaged gene.activity dynamics along the real time

# %%
# genes_to_plot = ['tbxta', 'sox2', 'myf5', 'pax6a', 'meox1']
# define the gene
list_genes = ["myf5","sox2","tbxta","myog",
              "meox1","tbx16","hes6",'pax6a',"en2a",
              "hbbe1.1","hbae3","hbbe3","myl1"]

# %%
# subset the adata_RNA for the list of genes (this is to reduce the computing resource/time)
adata_ATAC_sub = adata_ATAC[:, adata_ATAC.var_names.isin(list_genes)]
adata_ATAC_sub

# %%
# create a dataframe of cells-by-gene.activity (Signac)
count_matrice_atac = pd.DataFrame(adata_ATAC_sub.X.todense(),
                             index=adata_ATAC_sub.obs.index,
                             columns=adata_ATAC_sub.var_names)
count_matrice_atac.head()

# %%
# transfer the "dataset" category labels to count_matrice df
count_matrice_atac["dataset"] = adata_ATAC_sub.obs["dataset"]
count_matrice_atac["annotation_ML_coarse"] = adata_RNA_sub.obs["annotation_ML_coarse"]
count_matrice_atac["dev_stage"] = adata_RNA_sub.obs["dev_stage"]

# %%
# add the "timepoints" category for the column

# define the dictionary for mapping
dict_timepoints = {"TDR126":"0somites",
                   "TDR127":"5somites",
                   "TDR128":"10somites",
                   "TDR118":"15somites",
                   "TDR119":"15somites",
                   "TDR125":"20somites",
                   "TDR124":"30somites"}

# map the "dataset" to "timepoints"
count_matrice_atac["timepoints"] = count_matrice_atac["dataset"].map(dict_timepoints)
count_matrice_atac.head()

# %%
# compute the averaged gene expression

# Define numeric columns (gene expression columns) by selecting columns with float/int data types
numeric_columns = count_matrice_atac.select_dtypes(include=[np.number]).columns.tolist()

# Compute the mean gene expression level across all cells per 'timepoints'
timepoints_by_gene_activity = count_matrice_atac.groupby('timepoints')[numeric_columns].mean()

# Display the result
print(timepoints_by_gene_activity)

# %%
# compute the "standard errors" gene.activity across all cells per "timepoint" (TDR118 and TDR119 will be merged here)
timepoints_by_gene_activity_sem = count_matrice_atac.groupby('timepoints')[numeric_columns].sem()
timepoints_by_gene_activity_sem.head()


# %%
# numeric_timepoints to re-order the rows (timepoints_by_genes)
timepoints_by_gene_activity['numeric_timepoints'] = timepoints_by_gene_activity.index.str.extract('(\d+)').astype(int).values
timepoints_by_gene_activity['numeric_timepoints']

# Sort by the numeric timepoints to ensure correct order in plot
timepoints_by_gene_activity_sorted = timepoints_by_gene_activity.sort_values('numeric_timepoints')
timepoints_by_gene_activity_sorted

# %%
# numeric_timepoints to re-order the rows (timepoints_by_genes_sem)
timepoints_by_gene_activity_sem['numeric_timepoints'] = timepoints_by_gene_activity_sem.index.str.extract('(\d+)').astype(int).values
timepoints_by_gene_activity_sem['numeric_timepoints']

# Sort by the numeric timepoints to ensure correct order in plot
timepoints_by_gene_activity_sem_sorted = timepoints_by_gene_activity_sem.sort_values('numeric_timepoints')
timepoints_by_gene_activity_sem_sorted

# %%
for gene in list_genes:
    # generate a plot
    plt.figure(figsize=(4, 3))
    plt.errorbar(timepoints_by_gene_activity_sorted.index, 
                 timepoints_by_gene_activity_sorted[gene], 
                 timepoints_by_gene_activity_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    # plt.title('Scatter Plot of log-normalized counts over Timepoints')
    plt.xlabel('time (somites)')
    plt.ylabel('averaged gene expression (gene activity, log-norm)')
    plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(False)
    plt.savefig(figpath + f"ATAC_timepoints_{gene}.pdf")
    plt.show()

# %% [markdown]
# ## Step 3. visualization
#
# - plotting the RNA and ATAC (gene.activity) in the same scatter plot (for each gene)
#
# - plotting the correlation between RNA and ATAC (gene.activity) for each gene (color the dots with the viridis color palette for timepoints).

# %%
for gene in list_genes:
    # generate a plot
    plt.figure(figsize=(4, 3))
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 timepoints_by_genes_sorted[gene], 
                 timepoints_by_genes_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    plt.errorbar(timepoints_by_gene_activity_sorted.index, 
                 timepoints_by_gene_activity_sorted[gene], 
                 timepoints_by_gene_activity_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    plt.title(f'{gene}')
    plt.xlabel('time (somites)')
    plt.ylabel('averaged gene expression (log-norm)')
    plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    plt.xticks(rotation=90)
    plt.legend(["RNA", "ATAC"])
    plt.grid(False)
    plt.savefig(figpath + f"RNA_ATAC_timepoints_{gene}.pdf")
    plt.show()


# %%
# Just to monitor the trend over time, we can normalize each modality

for gene in list_genes:
    
#     # normalize the RNA
#     max_rna = np.max(timepoints_by_genes_sorted[gene])
#     rna_norm = timepoints_by_genes_sorted[gene]/max_rna
#     rna_sem_norm = timepoints_by_genes_sem_sorted[gene]/max_rna
    
#     # normalize the ATAC
#     max_atac = np.max(timepoints_by_gene_activity_sorted[gene])
#     atac_norm = timepoints_by_gene_activity_sorted[gene]/max_atac
#     atac_sem_norm = timepoints_by_gene_activity_sem_sorted[gene]/max_atac
    rna_norm = normalize(timepoints_by_genes_sorted[gene])
    rna_sem_norm = timepoints_by_genes_sem_sorted[gene] / (timepoints_by_genes_sorted[gene].max() - timepoints_by_genes_sorted[gene].min())
    atac_norm = normalize(timepoints_by_gene_activity_sorted[gene])
    atac_sem_norm = timepoints_by_gene_activity_sem_sorted[gene] / (timepoints_by_gene_activity_sorted[gene].max() - timepoints_by_gene_activity_sorted[gene].min())


    
    # generate a plot
    plt.figure(figsize=(4, 3))
    plt.errorbar(timepoints_by_genes_sorted.index, 
                 rna_norm, 
                 rna_sem_norm,
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    plt.errorbar(timepoints_by_gene_activity_sorted.index, 
                 atac_norm, 
                 atac_sem_norm,
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1)
    plt.title(f'{gene}')
    plt.xlabel('time (somites)')
    plt.ylabel('averaged gene expression (log-norm)')
    plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    plt.xticks(rotation=90)
    plt.legend(["RNA", "ATAC"])
    plt.grid(False)
    plt.savefig(figpath + f"RNA_ATAC_timepoints_{gene}_norm.pdf")
    plt.show()

# %% [markdown]
# ### correlation between RNA and ATAC

# %%
color_dict

timepoints_by_gene_activity_sorted.index.map(color_dict)

# %%
from scipy.stats import pearsonr

# plot the correlation between RNA and ATAC
for gene in list_genes:
    
    # define the color palette using the viridis
    colors = timepoints_by_gene_activity_sorted.index.map(color_dict)
    
    # define the x_vals and y_vals
    # Collect data points for correlation calculation
    x_vals = timepoints_by_gene_activity_sorted[gene]
    y_vals = timepoints_by_genes_sorted[gene]
    
    # Calculate correlation coefficient
    correlation, _ = pearsonr(x_vals, y_vals)
    
    # generate a plot
    plt.figure(figsize=(3, 3))
    plt.errorbar(x = timepoints_by_gene_activity_sorted[gene],
                 y = timepoints_by_genes_sorted[gene],
                 xerr=timepoints_by_gene_activity_sem_sorted[gene],
                 yerr=timepoints_by_genes_sem_sorted[gene],
                 fmt='o', capsize=2, linestyle='None', 
                 label=gene, markersize=2, elinewidth=1, capthick=1,)
                 # ecolor=colors, color=colors)
        
    # Add the correlation coefficient text to the plot
    plt.text(0.05, 0.95, f'r = {correlation:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.title(f'{gene}')
    plt.xlabel('gene activity (ATAC)')
    plt.ylabel('gene expression (RNA)')
    # plt.xticks(timepoints_by_genes_sorted.index)  # Show original timepoint labels
    # plt.xticks(rotation=90)
    # plt.legend(["RNA", "ATAC"])
    plt.grid(False)
    plt.savefig(figpath + f"RNA_ATAC_corr_scatter_{gene}.pdf")
    plt.show()

# %%

# %% [markdown]
# ## [HOLD] Step 3 . Find patterns in the RNA/ATAC dynamics (real-time)
#
# - First, we will normalize the timepoints_by_genes dataframe for each gene's maximum gene.activity score (averaged)
# - Second, we will generate a heatmap for genes-by-timepoints
# - Lastly, we will perform hirarchical clustering on the heatmap to find the gene clusters that have similar ATAC dynamics (either increasing or decreasing over time).

# %%
# first, exclude the "numeric_timepoints" column before normalization
timepoints_by_genes_sorted.drop('numeric_timepoints', axis=1, inplace=True)

# Normalize each gene's expression levels to the maximum value for that gene
normalized_timepoints_by_genes = timepoints_by_genes_sorted.divide(timepoints_by_genes_sorted.max())
normalized_timepoints_by_genes.head()

# %%
# Replace any infinite values with NaN
timepoints_by_genes_sorted.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any rows or columns with NaN values if they can't be imputed
# normalized_timepoints_by_genes.dropna(axis=0, inplace=True)  # Drop rows with NaN
timepoints_by_genes_sorted.dropna(axis=1, inplace=True)  # Drop columns with NaN

# %%
# Replace any infinite values with NaN
normalized_timepoints_by_genes.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any rows or columns with NaN values if they can't be imputed
# normalized_timepoints_by_genes.dropna(axis=0, inplace=True)  # Drop rows with NaN
normalized_timepoints_by_genes.dropna(axis=1, inplace=True)  # Drop columns with NaN

# %%
normalized_timepoints_by_genes.T.head()

# %%
# Create a heatmap with hierarchical clustering
plt.figure(figsize=(12, 8))
sns.clustermap(normalized_timepoints_by_genes.T, method='average', metric='euclidean', cmap='viridis', standard_scale=0,
               row_cluster=True, col_cluster=False)
plt.show()

# %%
# Create a heatmap with hierarchical clustering
plt.figure(figsize=(12, 8))
g = sns.clustermap(normalized_timepoints_by_genes.T, method='average', metric='euclidean', cmap='viridis', standard_scale=0,
               row_cluster=True, col_cluster=False, yticklabels=False)

reordered_labels = normalized_timepoints_by_genes.T.index[g.dendrogram_row.reordered_ind].tolist()
use_labels = ["myf5","sox2","tbxta",
              "meox1","tbx16","hes6",
              "hbbe1.1","hbae3","hbbe3","rps16",
              "lrrc24","krt96"]
use_ticks = [reordered_labels.index(label) + .5 for label in use_labels]
g.ax_heatmap.set(yticks=use_ticks, yticklabels=use_labels)
g.savefig(figpath + "clustered_heatmap_gene_activity_score_timepoints_norm.png")
g.savefig(figpath + "clustered_heatmap_gene_activity_score_timepoints_norm.pdf")
plt.show()

# %% [markdown]
# ## Step 4. correlation between ATAC and RNA

# %%
# Group by 'timepoint' and calculate the mean of each gene across cells within each timepoint
df_avg_RNA = pd.DataFrame(adata_RNA.to_df()).groupby(adata_RNA.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_RNA.index.name = 'timepoint'
print(df_avg_RNA)

# %%
# # copy over the "dev_stage" from RNA to ATAC object
adata_ATAC.obs["dev_stage"] = adata_ATAC.obs_names.map(adata_RNA.obs["dev_stage"])
adata_ATAC

# %%
# Group by 'timepoint' and calculate the mean of each gene activity across cells within each timepoint
df_avg_ATAC = pd.DataFrame(adata_ATAC.to_df()).groupby(adata_ATAC.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_ATAC.index.name = 'timepoint'
print(df_avg_ATAC)

# %%
# Find intersection of genes between RNA and ATAC data
common_genes = df_avg_RNA.columns.intersection(df_avg_ATAC.columns)

# Subset both dataframes for common genes
df_RNA_common = df_avg_RNA[common_genes]
df_ATAC_common = df_avg_ATAC[common_genes]

# %%
# Create a new DataFrame to store correlation coefficients for each gene
correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# Compute Pearson correlation coefficient for each gene across timepoints
for gene in common_genes:
    correlation_df.loc[gene, 'correlation'] = df_RNA_common[gene].corr(df_ATAC_common[gene])

print(correlation_df)

# %%
# Drop NaN values from correlation dataframe, if any
correlation_df = correlation_df.dropna()

# Determine the top 5% and bottom 5% thresholds
top_threshold = correlation_df['correlation'].quantile(0.95)
bottom_threshold = correlation_df['correlation'].quantile(0.05)

# Get the top 5% and bottom 5% of the correlation coefficients
top_5_percent = correlation_df[correlation_df['correlation'] >= top_threshold]
bottom_5_percent = correlation_df[correlation_df['correlation'] <= bottom_threshold]

# Display the results
print("Top 5% Correlations:")
print(top_5_percent)

print("\nBottom 5% Correlations:")
print(bottom_5_percent)

# %%
# Convert the index to a list and save as a text file
top_genes_list = top_genes.tolist()

# Save the gene names as a text file using to_csv
with open('top_5_percent_genes.txt', 'w') as f:
    for gene in top_genes_list:
        f.write(f"{gene}\n")

# %%
# Convert the index to a list and save as a text file

bottom_genes_list = bottom_5_percent.index.tolist()

# Save the gene names as a text file using to_csv
with open('bottom_5_percent_genes.txt', 'w') as f:
    for gene in bottom_genes_list:
        f.write(f"{gene}\n")


# %%
def generate_permutation_without_self_match(genes):
    """Generate a permutation of gene names such that no gene is paired with itself."""
    while True:
        permuted_genes = np.random.permutation(genes)
        # Check if any gene is paired with itself; if not, return
        if not np.any(permuted_genes == genes):
            return permuted_genes


# Generate a valid permutation without any gene being paired with itself
shuffled_columns = generate_permutation_without_self_match(common_genes)
df_ATAC_shuffled = df_ATAC_common[shuffled_columns]

# Create a DataFrame to store correlation coefficients for non-pairs
null_correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# Compute Pearson correlation coefficient for each "non-pair" gene
for gene_rna, gene_atac in zip(common_genes, shuffled_columns):
    null_correlation_df.loc[gene_rna, 'correlation'] = df_RNA_common[gene_rna].corr(df_ATAC_shuffled[gene_atac])

# Drop NaN values if any were generated
null_correlation_df = null_correlation_df.dropna()

print(null_correlation_df)

# %%
plt.hist(correlation_df.correlation, bins=50, density=True, alpha=0.5)
plt.hist(null_correlation_df.correlation, bins=50, density=True, alpha=0.5)
plt.legend(["gene pairs", "null"])
plt.xlabel("pearson correlation")
plt.ylabel("density")
plt.grid(False)
plt.savefig("pearson_corr_hist.pdf")
plt.show()

# %% [markdown]
# ## highly variable genes (RNA expression level)

# %%
# Calculate highly variable genes in the RNA data
sc.pp.highly_variable_genes(adata_RNA, flavor='seurat', n_top_genes=2000)

# Subset the adata_RNA object to only include highly variable genes
adata_RNA_hvg = adata_RNA[:, adata_RNA.var['highly_variable']]

# %%
# Group by 'timepoint' and calculate the mean of each gene across cells within each timepoint
df_avg_RNA = pd.DataFrame(adata_RNA_hvg.to_df()).groupby(adata_RNA_hvg.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_RNA.index.name = 'timepoint'
print(df_avg_RNA)

# %%
# # copy over the "dev_stage" from RNA to ATAC object
adata_ATAC.obs["dev_stage"] = adata_ATAC.obs_names.map(adata_RNA.obs["dev_stage"])
adata_ATAC

# %%
# Group by 'timepoint' and calculate the mean of each gene activity across cells within each timepoint
df_avg_ATAC = pd.DataFrame(adata_ATAC.to_df()).groupby(adata_ATAC.obs["dev_stage"]).mean()

# Rename the index to indicate it's timepoint and columns to indicate genes
df_avg_ATAC.index.name = 'timepoint'
print(df_avg_ATAC)

# %%
# Find intersection of genes between RNA and ATAC data
common_genes = df_avg_RNA.columns.intersection(df_avg_ATAC.columns)

# Subset both dataframes for common genes
df_RNA_common = df_avg_RNA[common_genes]
df_ATAC_common = df_avg_ATAC[common_genes]

# %%
# Create a new DataFrame to store correlation coefficients for each gene
correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# Compute Pearson correlation coefficient for each gene across timepoints
for gene in common_genes:
    correlation_df.loc[gene, 'correlation'] = df_RNA_common[gene].corr(df_ATAC_common[gene])

print(correlation_df)

# %%
plt.hist(correlation_df.correlation, bins=50, density=True)
plt.grid(False)
plt.show()

# %%
# def generate_permutation_without_self_match(genes):
#     """Generate a permutation of gene names such that no gene is paired with itself."""
#     while True:
#         permuted_genes = np.random.permutation(genes)
#         # Check if any gene is paired with itself; if not, return
#         if not np.any(permuted_genes == genes):
#             return permuted_genes


# # Generate a valid permutation without any gene being paired with itself
# shuffled_columns = generate_permutation_without_self_match(common_genes)
# df_ATAC_shuffled = df_ATAC_common[shuffled_columns]

# # Create a DataFrame to store correlation coefficients for non-pairs
# null_correlation_df = pd.DataFrame(index=common_genes, columns=['correlation'])

# # Compute Pearson correlation coefficient for each "non-pair" gene
# for gene_rna, gene_atac in zip(common_genes, shuffled_columns):
#     null_correlation_df.loc[gene_rna, 'correlation'] = df_RNA_common[gene_rna].corr(df_ATAC_shuffled[gene_atac])

# # Drop NaN values if any were generated
# null_correlation_df = null_correlation_df.dropna()

# print(null_correlation_df)

# %%
import itertools
import pandas as pd

# Generate all possible non-pair combinations using itertools.product
all_pairs = list(itertools.product(common_genes, common_genes))

# Filter out self-pairs (gene_rna == gene_atac)
non_pairs = [(gene_rna, gene_atac) for gene_rna, gene_atac in all_pairs if gene_rna != gene_atac]

# # Create a DataFrame to store correlation coefficients for non-pairs
# null_correlation_df = pd.DataFrame(columns=['gene_rna', 'gene_atac', 'correlation'])

# # Compute Pearson correlation coefficient for each "non-pair" gene
# for gene_rna, gene_atac in non_pairs:
#     correlation = df_RNA_common[gene_rna].corr(df_ATAC_common[gene_atac])
#     null_correlation_df = null_correlation_df.append({
#         'gene_rna': gene_rna,
#         'gene_atac': gene_atac,
#         'correlation': correlation
#     }, ignore_index=True)

# # Drop NaN values if any were generated
# null_correlation_df = null_correlation_df.dropna()

# print(null_correlation_df)
# Create a list to collect correlation results
correlation_results = []

# Compute Pearson correlation coefficient for each "non-pair" gene
for gene_rna, gene_atac in non_pairs:
    correlation = df_RNA_common[gene_rna].corr(df_ATAC_common[gene_atac])
    correlation_results.append({
        'gene_rna': gene_rna,
        'gene_atac': gene_atac,
        'correlation': correlation
    })

# Convert the results list to a DataFrame
null_correlation_df = pd.DataFrame(correlation_results)

# Drop NaN values if any were generated
null_correlation_df = null_correlation_df.dropna()

print(null_correlation_df)


# %%
# Function to generate a permutation of gene names such that no gene is paired with itself
def generate_permutation_without_self_match(genes):
    """Generate a permutation of gene names such that no gene is paired with itself."""
    while True:
        permuted_genes = np.random.permutation(genes)
        # Check if any gene is paired with itself; if not, return
        if not np.any(permuted_genes == genes):
            return permuted_genes

# Number of repetitions
n_repeats = 5

# Create a DataFrame to accumulate correlation values for each gene across repetitions
correlation_accumulator = pd.DataFrame(index=common_genes, columns=range(n_repeats), dtype=float)

# Repeat the permutation and correlation calculation multiple times
for i in range(n_repeats):
    # Generate a valid permutation without any gene being paired with itself
    shuffled_columns = generate_permutation_without_self_match(common_genes)
    df_ATAC_shuffled = df_ATAC_common[shuffled_columns]

    # Compute Pearson correlation coefficient for each "non-pair" gene
    for gene_rna, gene_atac in zip(common_genes, shuffled_columns):
        correlation_accumulator.loc[gene_rna, i] = df_RNA_common[gene_rna].corr(df_ATAC_shuffled[gene_atac])

# Compute the average correlation per gene across repetitions
average_correlation_df = correlation_accumulator.mean(axis=1)

# Drop NaN values if any were generated
average_correlation_df = average_correlation_df.dropna()

# Convert to a DataFrame for better formatting if needed
null_correlation_df = pd.DataFrame(average_correlation_df, columns=['average_correlation'])

print(null_correlation_df)

# %%
null_correlation_df.average_correlation.mean()

# %%
plt.hist(correlation_df.correlation, bins=50, density=True, alpha=0.5, range=(-1, 1), rwidth=0.9)
plt.hist(null_correlation_df.average_correlation, bins=50, density=True, alpha=0.5, range=(-1, 1), rwidth=0.9)

plt.legend(["gene pairs", "null"])
# Draw vertical lines for the cutoffs
plt.axvline(x=top_threshold, color='r', linestyle='--', linewidth=2, label='Top 5% cutoff')
plt.axvline(x=mid_threshold, color='r', linestyle='--', linewidth=2, label='Top 75% cutoff')
plt.axvline(x=bottom_threshold, color='r', linestyle='--', linewidth=2, label='Top 50% cutoff')
plt.xlabel("pearson correlation")
plt.ylabel("density")
plt.grid(False)
plt.savefig("pearson_corr_hist_hvgs.pdf")
plt.show()

# %%
# Calculate the thresholds for the top 5% and bottom 5% correlations
top_threshold = correlation_df['correlation'].quantile(0.95)
mid_threshold = correlation_df['correlation'].quantile(0.75)
bottom_threshold = correlation_df['correlation'].quantile(0.60)

# Plot the histogram of correlation coefficients
plt.figure(figsize=(8, 6))
plt.hist(correlation_df['correlation'], bins=50, edgecolor='k', alpha=0.7)

# Draw vertical lines for the cutoffs
plt.axvline(x=top_threshold, color='r', linestyle='--', linewidth=2, label='Top 5% cutoff')
plt.axvline(x=mid_threshold, color='r', linestyle='--', linewidth=2, label='Top 75% cutoff')
plt.axvline(x=bottom_threshold, color='b', linestyle='--', linewidth=2, label='Top 50% cutoff')

# Add labels and legend
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.legend()
plt.grid(False)
plt.show()

# %%
# Drop NaN values from correlation dataframe, if any
correlation_df = correlation_df.dropna()

# Determine the top 5% and bottom 5% thresholds
top_threshold = correlation_df['correlation'].quantile(0.95)
mid_threshold = correlation_df['correlation'].quantile(0.75)
bottom_threshold = correlation_df['correlation'].quantile(0.50)

# Get the top 5% and bottom 5% of the correlation coefficients
top_5_percent = correlation_df[correlation_df['correlation'] >= top_threshold]
# Get the middle 50-75% of the correlation coefficients
mid_50_75_percent = correlation_df[(correlation_df['correlation'] <= mid_threshold) & (correlation_df['correlation'] >= bottom_threshold)]

# Display the results
print("Top 5% Correlations:")
print(top_5_percent)

print("\nTop 50-75% Correlations:")
print(mid_50_75_percent)

# %%
# Convert the index to a list and save as a text file
top_genes_list = top_5_percent.index.tolist()

# Save the gene names as a text file using to_csv
with open('top_5_percent_genes.txt', 'w') as f:
    for gene in top_genes_list:
        f.write(f"{gene}\n")

# %%
# Convert the index to a list and save as a text file
mid_50_75_genes_list = mid_50_75_percent.index.tolist()

# Save the gene names as a text file using to_csv
with open('mid_50_75_genes_list.txt', 'w') as f:
    for gene in mid_50_75_genes_list:
        f.write(f"{gene}\n")

# %%
correlation_df.correlation.mean()
