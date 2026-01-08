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
# ## pre-processing the peak UMAP
# - last updated: 03/23/2025
# - pseudobulk the scATAC-seq dataset

# %%
# !nvidia-smi

# %%
# 0. Import
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
# import muon as mu # added muon
import seaborn as sns
import scipy.sparse
from scipy.io import mmread

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# %%
# figure parameter setting
# %matplotlib inline
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
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_v2/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %% [markdown]
# ## strategies
#
# - (1) raw counts pseudo-bulked using the initial approach (divide by the total counts per group)
# - (2) scaled counts pseudo-bulked using the new approach (divide by the total counts per group, then scaled up using the median counts)
#

# %% [markdown]
#
# ### import the adata object

# %%
# import the adata_peaks with raw counts
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad")
adata_peaks

# %%
# import the adata for cells-by-genes (RNA) to transfer some metadata
adata_RNA = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
adata_RNA

# %%
# # copy over the metadata
adata_peaks.obs = adata_RNA.obs.copy()

# %%
import scipy.sparse as sp
# If .X is a sparse matrix
adata_peaks.X = sp.csr_matrix(adata_peaks.X, dtype=np.float32)

# %%
adata_peaks

# %%
# # save the new master object (raw counts, and also sparse matrix with float32 format)
# adata_peaks.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered_v2.h5ad")


# %%
# compute the QC metrics for the peaks

# move the counts to GPU
rsc.get.anndata_to_GPU(adata_peaks) # moves `.X` to the GPU
# compute the QC metrics
rsc.pp.calculate_qc_metrics(adata_peaks, expr_type='counts', 
                            var_type='peaks', qc_vars=None, log1p=False)

# %%
# save the new master object (raw counts, and also sparse matrix with float32 format)
adata_peaks.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered_v2.h5ad")


# %% [markdown]
# ## 1. raw counts with pseudo-bulking with old strategy


# %%
# Run normalization
adata_pseudo__old = analyze_peaks_with_normalization(
    adata_peaks,
    celltype_key='annotation_ML_coarse',
    timepoint_key='dev_stage'
)

adata_pseudo__old

# %%
# log(1+p) transformation to convert the counts distribution from Poisson to Gaussian
adata_pseudo__old.X = adata_pseudo__old.layers["normalized"].copy()
sc.pp.log1p(adata_pseudo__old)
adata_pseudo__old.layers["log_norm"] = adata_pseudo__old.X.copy()

# %%
adata_pseudo__old

# %%
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo__old.X = adata_pseudo__old.layers["log_norm"]
peaks_pb_log_norm_old = adata_pseudo__old.copy().T
rsc.get.anndata_to_GPU(peaks_pb_log_norm_old) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb_log_norm_old)
rsc.pp.pca(peaks_pb_log_norm_old, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_log_norm_old, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb_log_norm_old, min_dist=0.3, random_state=42)
sc.pl.umap(peaks_pb_log_norm_old)

# %%
# annotate the highly variable peaks (50K) defined earlier
peaks_pb_log_norm_old.obs["hvps_50K"] = peaks_pb_log_norm_old.obs_names.isin(peaks_50k_obj.obs_names)

# %%
sc.pl.umap(peaks_pb_log_norm_old, color="hvps_50K")

# %%

# %%
sc.pl.pca(peaks_pb_log_norm_old)

# %%
sc.pl.pca_variance_ratio(peaks_pb_log_norm_old, log=True)

# %%

# %%
# import the 50k peaks object to make sure that we're dealing with the same set of peaks
peaks_50k_obj = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k_EDA2.h5ad")
peaks_50k_obj

# %%
adata_pseudo_filt = adata_pseudo[:, adata_pseudo.var_names.isin(peaks_50k_obj.obs_names)]
adata_pseudo_filt

# %%
# # compute the highly variable genes using "sum" counts layer
# adata_pseudo.X = adata_pseudo.layers["sum"].copy()
# # placing the sparse count matrix on the GPU
# rsc.get.anndata_to_GPU(adata_pseudo)

# # compute the highly variable genes using the raw counts after the pseudo-bulking
# rsc.pp.highly_variable_genes(adata_pseudo,n_top_genes=50000, flavor="seurat_v3")

# %%
# # subset for the highly variable peaks
# adata_peaks_pb_hvp_20000 = adata_pseudo[:,adata_pseudo.var["highly_variable"]==True]
# adata_peaks_pb_hvp_20000

# %%

# %%
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo_filt.X = adata_pseudo_filt.layers["log_norm"]
peaks_pb_hvp_50k = adata_pseudo_filt.copy().T
rsc.get.anndata_to_GPU(peaks_pb_hvp_50k) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb_hvp_50k)
rsc.pp.pca(peaks_pb_hvp_50k, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_hvp_50k, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb_hvp_50k, min_dist=0.3, random_state=42)

# %%
# rsc.tl.umap(peaks_pb_hvp_20000, min_dist=0.1)
sc.pl.umap(peaks_pb_hvp_50k)

# %%
peaks_pb_hvp_50k.obs = peaks_50k_obj.obs.copy()

# %%
sc.pl.umap(peaks_pb_hvp_50k, color=["celltype","timepoint","leiden"], ncols=1)

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 2. original adata_peaks object, re-pseudobulked using median scaling

# %%
# import the adata_peaks with raw counts
adata_peaks_orig = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_master_filtered.h5ad")
adata_peaks_orig


# %%
def analyze_peaks_with_normalization(
    adata, 
    celltype_key='annotation_ML_coarse', 
    timepoint_key='dev_stage'
):
    """
    1) Compute each cell's total_counts (sum of peaks/reads).
    2) For each (celltype, timepoint) group, compute the total_coverage 
       = sum of total_counts from all cells in that group.
    3) Create pseudobulk by summing (func='sum') each group's cells for the peaks matrix.
    4) The common_scale_factor = median of all group_total_coverage.
    5) For each group g, normalized_pseudobulk = raw_pseudobulk * (common_scale_factor / group_total_coverage[g]).

    Returns
    -------
    adata_pseudo : an AnnData with:
        - .X = raw pseudobulk counts
        - layers['normalized'] = scaled pseudobulk counts
        - obs['total_coverage'] = group's raw coverage
        - obs['scale_factor'] = how much that group's coverage was scaled
        - obs['n_cells'] and obs['mean_depth'] optionally stored as well
        - uns['common_scale_factor'] = the median coverage used for scaling
    """

    # 1) total_counts per cell
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1))

    # 2) total_coverage per group (sum of total_counts)
    group_total_coverage = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].sum()

    # 3) Pseudobulk by summing group cells
    ident_cols = [celltype_key, timepoint_key]
    adata_pseudo = sc.get.aggregate(adata, ident_cols, func='sum')
    # Copy the summed counts into .X
    adata_pseudo.X = adata_pseudo.layers["sum"].copy()

    # Split the new obs index (e.g. "Astro_dev_stage1") back into celltype/timepoint
    celltype_timepoint = pd.DataFrame({
        'celltype': ['_'.join(x.split('_')[:-1]) for x in adata_pseudo.obs.index],
        'timepoint': [x.split('_')[-1] for x in adata_pseudo.obs.index]
    }, index=adata_pseudo.obs.index)

    # Prepare for normalized counts
    X = adata_pseudo.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # 4) common_scale_factor = median of group_total_coverage
    common_scale_factor = np.median(group_total_coverage.values)

    # 5) Rescale each group's pseudobulk
    normalized_counts = np.zeros_like(X)
    coverage_list = []
    scale_factor_list = []

    for i, idx in enumerate(adata_pseudo.obs.index):
        ct = celltype_timepoint.loc[idx, 'celltype']
        tp = celltype_timepoint.loc[idx, 'timepoint']
        coverage_g = group_total_coverage[(ct, tp)]

        # Scale factor = common_scale_factor / group's total coverage
        scale_g = common_scale_factor / coverage_g
        normalized_counts[i, :] = X[i, :] * scale_g
        
        coverage_list.append(coverage_g)
        scale_factor_list.append(scale_g)

    # Store normalized counts in a new layer
    adata_pseudo.layers['normalized'] = normalized_counts

    # Record coverage and scaling info in .obs
    adata_pseudo.obs['total_coverage'] = coverage_list
    adata_pseudo.obs['scale_factor'] = scale_factor_list

    # Optionally, also store #cells and mean_depth
    group_ncells = adata.obs.groupby([celltype_key, timepoint_key]).size()
    group_mean_depth = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].mean()

    n_cells_list = []
    mean_depth_list = []
    for i, idx in enumerate(adata_pseudo.obs.index):
        ct = celltype_timepoint.loc[idx, 'celltype']
        tp = celltype_timepoint.loc[idx, 'timepoint']
        n_cells_list.append(group_ncells[(ct, tp)])
        mean_depth_list.append(group_mean_depth[(ct, tp)])
    adata_pseudo.obs['n_cells'] = n_cells_list
    adata_pseudo.obs['mean_depth'] = mean_depth_list

    # Save the "common" scale factor in .uns
    adata_pseudo.uns['common_scale_factor'] = common_scale_factor

    return adata_pseudo


# %%
# Run normalization
adata_pseudo_eda = analyze_peaks_with_normalization(
    adata_peaks_orig,
    celltype_key='annotation_ML_coarse',
    timepoint_key='dev_stage'
)

adata_pseudo_eda

# %%
# log(1+p) transformation to convert the counts distribution from Poisson to Gaussian
adata_pseudo_eda.X = adata_pseudo_eda.layers["normalized"].copy()
sc.pp.log1p(adata_pseudo_eda)
adata_pseudo_eda.layers["log_norm"] = adata_pseudo_eda.X.copy()

# %%
adata_pseudo_eda

# %%
adata_pseudo_eda_filt = adata_pseudo_eda[:, adata_pseudo_eda.var_names.isin(peaks_50k_obj.obs_names)]
adata_pseudo_eda_filt

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 2-2. computing the UMAP

# %%
# # compute the highly variable genes using "sum" counts layer
# adata_pseudo.X = adata_pseudo.layers["sum"].copy()
# # placing the sparse count matrix on the GPU
# rsc.get.anndata_to_GPU(adata_pseudo)

# # compute the highly variable genes using the raw counts after the pseudo-bulking
# rsc.pp.highly_variable_genes(adata_pseudo,n_top_genes=50000, flavor="seurat_v3")

# %%
# # subset for the highly variable peaks
# adata_peaks_pb_hvp_20000 = adata_pseudo[:,adata_pseudo.var["highly_variable"]==True]
# adata_peaks_pb_hvp_20000

# %%

# %%
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo_eda_filt.X = adata_pseudo_eda_filt.layers["log_norm"]
peaks_pb_hvp_50k_eda = adata_pseudo_eda_filt.copy().T
rsc.get.anndata_to_GPU(peaks_pb_hvp_50k_eda) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb_hvp_50k_eda)
rsc.pp.pca(peaks_pb_hvp_50k_eda, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_hvp_50k_eda, n_neighbors=10, n_pcs=40)
rsc.tl.umap(peaks_pb_hvp_50k_eda, min_dist=0.3, random_state=42)
# plot the UMAP
sc.pl.umap(peaks_pb_hvp_50k_eda)

# %%
peaks_pb_hvp_50k_eda.obs = peaks_50k_obj.obs.copy()

# %%
sc.pl.umap(peaks_pb_hvp_50k_eda, color=["celltype","timepoint","leiden"])

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 3. raw counts with median scaling (50k hvps)

# %%

# %%
# Run normalization
adata_pseudo = analyze_peaks_with_normalization(
    adata_peaks,
    celltype_key='annotation_ML_coarse',
    timepoint_key='dev_stage'
)

adata_pseudo

# %%
adata_temp = sc.pp.calculate_qc_metrics(adata_pseudo)
adata_temp

# %%
adata_pseudo

# %%
adata_pseudo_filt = adata_pseudo[:, adata_pseudo.var_names.isin(peaks_50k_obj.obs_names)]
adata_pseudo_filt

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 3-2. computing the UMAP - 50K hvps

# %%
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo_filt.X = adata_pseudo_filt.layers["log_norm"]
peaks_pb_hvp_50k_v2 = adata_pseudo_filt.copy().T
rsc.get.anndata_to_GPU(peaks_pb_hvp_50k_v2) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb_hvp_50k_v2)
rsc.pp.pca(peaks_pb_hvp_50k_v2, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_hvp_50k_v2, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb_hvp_50k_v2, min_dist=0.3)#, random_state=42)
# plot the UMAP
sc.pl.umap(peaks_pb_hvp_50k_v2)

# %%
peaks_pb_hvp_50k_v2.obs = peaks_50k_obj.obs.copy()

# %%
sc.pl.umap(peaks_pb_hvp_50k_v2, color=["celltype","timepoint","leiden"], ncols=1)

# %%
peaks_pb_hvp_50k_v2.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_hvp_50k.h5ad")

# %% [markdown]
# ## 4. pseudo-bulked with raw counts, and scaled with over-dispersion=1
#
# ### computing the UMAP - all peaks

# %%
# import the pseudobulk-by-peaks object
adata_pseudo = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/pseudobulk_ct_tp_by_peaks.h5ad")
adata_pseudo

# %%
# # log(1+p) transformation to convert the counts distribution from Poisson to Gaussian
# adata_pseudo.X = adata_pseudo.layers["normalized"].copy()
# sc.pp.log1p(adata_pseudo)
# adata_pseudo.layers["log_norm"] = adata_pseudo.X.copy()

# %%
print(adata_pseudo.obs['total_coverage'].sort_values(ascending=False))

# %%
adata_peaks.obs["annotation_ML_coarse"].value_counts()

# %%
adata_peaks[adata_peaks.obs["annotation_ML_coarse"]=="primordial_germ_cells"].obs["dev_stage"].value_counts()

# %% [markdown]
# #### So, the two outlier groups are "early" epidermis, with just a lot more cells than the rest.

# %%
# save the adata_pseudo (pseudobulk-by-peaks)
# adata_pseudo.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/pseudobulk_ct_tp_by_peaks.h5ad")

# %%
# import the pseudo-bulked data
adata_pseudo = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/pseudobulk_ct_tp_by_peaks.h5ad")
adata_pseudo

# %% [markdown]
# ## Checking the mean-variance relationship

# %%
adata_pseudo.shape

# %%
adata_pseudo.layers["normalized"].mean(axis=1).shape

# %%
# Create a figure with 1x3 subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# First subplot: Mean vs Variance scatter plot
mean_peaks = np.array(adata_pseudo.layers["sum"].mean(axis=0)).flatten()
var_peaks = np.array(adata_pseudo.layers["sum"].var(axis=0)).flatten()

# First subplot: Mean vs Variance for "sum" layer
mean_peaks_sum = np.array(adata_pseudo.layers["sum"].mean(axis=0)).flatten()
var_peaks_sum = np.array(adata_pseudo.layers["sum"].var(axis=0)).flatten()

axs[0].scatter(mean_peaks_sum, var_peaks_sum, alpha=0.5, s=10)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel("mean_peaks (sum)")
axs[0].set_ylabel("variance_peaks (sum)")
axs[0].set_title("mean vs variance (sum)")
axs[0].grid(False)

# Second subplot: Mean vs Variance for "normalized" layer
mean_peaks_norm = np.array(adata_pseudo.layers["normalized"].mean(axis=0)).flatten()
var_peaks_norm = np.array(adata_pseudo.layers["normalized"].var(axis=0)).flatten()

axs[1].scatter(mean_peaks_norm, var_peaks_norm, alpha=0.5, s=10)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlabel("mean_peaks (normalized)")
axs[1].set_ylabel("variance_peaks (normalized)")
axs[1].set_title("mean vs variance (normalized)")
axs[1].grid(False)

# Third subplot: Mean vs Variance for "log_norm" layer
mean_peaks_log = np.array(adata_pseudo.layers["log_norm"].mean(axis=0)).flatten()
var_peaks_log = np.array(adata_pseudo.layers["log_norm"].var(axis=0)).flatten()

axs[2].scatter(mean_peaks_log, var_peaks_log, alpha=0.5, s=10)
axs[2].set_xscale("log") 
axs[2].set_yscale("log")
axs[2].set_xlabel("mean_peaks (log_norm)")
axs[2].set_ylabel("variance_peaks (log_norm)")
axs[2].set_title("mean vs variance (log_norm)")
axs[2].grid(False)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# %%
# Create a figure with 1x4 subplots
fig, axs = plt.subplots(1, 4, figsize=(12, 3))

# First subplot: Mean vs Variance scatter plot
mean_peaks = np.array(adata_pseudo.layers["sum"].mean(axis=0)).flatten()
var_peaks = np.array(adata_pseudo.layers["sum"].var(axis=0)).flatten()

# First subplot: Mean vs Variance for "sum" layer
mean_peaks_sum = np.array(adata_pseudo.layers["sum"].mean(axis=0)).flatten()
var_peaks_sum = np.array(adata_pseudo.layers["sum"].var(axis=0)).flatten()

axs[0].scatter(mean_peaks_sum, var_peaks_sum, alpha=0.5, s=10)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel("mean_peaks (sum)")
axs[0].set_ylabel("variance_peaks (sum)")
axs[0].set_title("mean vs variance (sum)")
axs[0].grid(False)

# Second subplot: Mean vs Variance for "normalized" layer
mean_peaks_norm = np.array(adata_pseudo.layers["normalized"].mean(axis=0)).flatten()
var_peaks_norm = np.array(adata_pseudo.layers["normalized"].var(axis=0)).flatten()

axs[1].scatter(mean_peaks_norm, var_peaks_norm, alpha=0.5, s=10)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlabel("mean_peaks (normalized)")
axs[1].set_ylabel("variance_peaks (normalized)")
axs[1].set_title("mean vs variance (normalized)")
axs[1].grid(False)

# Third subplot: Mean vs Variance for "log_norm" layer
mean_peaks_log = np.array(adata_pseudo.layers["log_norm"].mean(axis=0)).flatten()
var_peaks_log = np.array(adata_pseudo.layers["log_norm"].var(axis=0)).flatten()

axs[2].scatter(mean_peaks_log, var_peaks_log, alpha=0.5, s=10)
axs[2].set_xscale("log") 
axs[2].set_yscale("log")
axs[2].set_xlabel("mean_peaks (log_norm)")
axs[2].set_ylabel("variance_peaks (log_norm)")
axs[2].set_title("mean vs variance (log_norm)")
axs[2].grid(False)

# Fourth subplot: Mean vs Variance for log(normalized+1)
# First, create the log(normalized+1) matrix
if sp.issparse(adata_pseudo.layers["sum"]):
    # For sparse matrices
    norm_by_coverage = adata_pseudo.layers["sum"].copy()
    for i in range(norm_by_coverage.shape[0]):
        norm_by_coverage[i] = norm_by_coverage[i] / adata_pseudo.obs["total_coverage"].iloc[i]
    log_norm_plus1 = np.log1p(norm_by_coverage)
    # log_norm_plus1.data = sc.pp.log1p(log_norm_plus1.data)
else:
    # For dense matrices
    norm_by_coverage = adata_pseudo.layers["sum"].copy()
    for i in range(norm_by_coverage.shape[0]):
        norm_by_coverage[i] = norm_by_coverage[i] / adata_pseudo.obs["total_coverage"].iloc[i]
    log_norm_plus1 = np.log1p(norm_by_coverage)

# Calculate statistics and plot
mean_peaks_log1p = np.array(log_norm_plus1.mean(axis=0)).flatten()
var_peaks_log1p = np.array(log_norm_plus1.var(axis=0)).flatten()

axs[3].scatter(mean_peaks_log1p, var_peaks_log1p, alpha=0.5, s=10)
axs[3].set_xscale("log") 
axs[3].set_yscale("log")
axs[3].set_xlabel("mean_peaks (log_norm: 1/total_counts)")
axs[3].set_ylabel("variance_peaks (log_norm: 1/total_counts)")
axs[3].set_title("mean vs variance (log_norm: 1/total_counts)")
axs[3].grid(False)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# %%
# save the log-normalized counts (normalized by the total_coutns - so the numbers are very small - 1e(-6))
adata_pseudo.layers["log_norm_division_by_total_counts"] = log_norm_plus1
adata_pseudo

# %%
adata_pseudo.var["mean_log_norm"] = mean_peaks_log
adata_pseudo.var["var_log_norm"] = var_peaks_log

# %%
sc.pp.calculate_qc_metrics(adata_pseudo, expr_type="counts", 
                                          var_type="peaks", layer="sum", 
                                          log1p=False, inplace=True)
adata_pseudo

# %%
plt.hist(adata_pseudo.obs["n_peaks_by_counts"], bins=40)
plt.grid(False)

# %%
print(adata_pseudo.obs["n_peaks_by_counts"].sort_values(ascending=False))

# %%
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo.X = adata_pseudo.layers["log_norm"]
peaks_pb = adata_pseudo.copy().T
rsc.get.anndata_to_GPU(peaks_pb) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_pb, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb, min_dist=0.3)#, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_pb)

# %%
# sc.pl.umap(peaks_pb, color=["n_cells_by_counts", "mean_log_norm","var_log_norm"])
sc.pl.umap(peaks_pb, color=["n_cells_by_counts"])

# %%
peaks_pb

# %%
# move the counts layer to CPU
rsc.get.anndata_to_CPU(peaks_pb)

# make sure that the counts matrix is "normalized"
peaks_pb.X = peaks_pb.layers["normalized"].copy()

# %%
# Create dictionaries to map columns to celltype and timepoint
celltype_mapping = {}
timepoint_mapping = {}

# Parse var names
for col in peaks_pb.var.index:
    parts = col.rsplit('_', 1)
    if len(parts) == 2 and 'somites' in parts[1]:
        celltype = parts[0]
        timepoint = parts[1]
        celltype_mapping[col] = celltype
        timepoint_mapping[col] = timepoint

# Get unique celltypes and timepoints
unique_celltypes = set(celltype_mapping.values())
unique_timepoints = set(timepoint_mapping.values())

# Create new obs columns for each celltype and timepoint
for celltype in unique_celltypes:
    # Get columns for this celltype
    celltype_cols = [col for col, ct in celltype_mapping.items() if ct == celltype]
    # Sum accessibility across all timepoints for this celltype
    peaks_pb.obs[f'accessibility_{celltype}'] = peaks_pb.X[:, [peaks_pb.var.index.get_loc(col) for col in celltype_cols]].sum(axis=1)

for timepoint in unique_timepoints:
    # Get columns for this timepoint
    timepoint_cols = [col for col, tp in timepoint_mapping.items() if tp == timepoint]
    # Sum accessibility across all celltypes for this timepoint
    peaks_pb.obs[f'accessibility_{timepoint}'] = peaks_pb.X[:, [peaks_pb.var.index.get_loc(col) for col in timepoint_cols]].sum(axis=1)


# %%
# For timepoints
timepoint_cols = ['accessibility_0somites', 'accessibility_5somites', 'accessibility_10somites', 
                 'accessibility_15somites', 'accessibility_20somites', 'accessibility_30somites']
timepoint_vals = np.array([peaks_pb.obs[col] for col in timepoint_cols]).T

# Find max timepoint for each peak
max_timepoint_idx = np.argmax(timepoint_vals, axis=1)
timepoint_names = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
peaks_pb.obs['timepoint'] = [timepoint_names[i] for i in max_timepoint_idx]

# Calculate corrected timepoint contrast
max_vals = np.max(timepoint_vals, axis=1)
# Calculate mean and std excluding max value for each peak
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
peaks_pb.obs['timepoint_contrast'] = (max_vals - other_vals_mean) / other_vals_std

# For celltypes
celltype_cols = [col for col in peaks_pb.obs.columns 
                if col.startswith('accessibility_') and 'somites' not in col]
celltype_vals = np.array([peaks_pb.obs[col] for col in celltype_cols]).T

# Find max celltype for each peak
max_celltype_idx = np.argmax(celltype_vals, axis=1)
celltype_names = [col.replace('accessibility_', '') for col in celltype_cols]
peaks_pb.obs['celltype'] = [celltype_names[i] for i in max_celltype_idx]

# Calculate corrected celltype contrast
max_vals = np.max(celltype_vals, axis=1)
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(celltype_vals, max_celltype_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(celltype_vals, max_celltype_idx)])
peaks_pb.obs['celltype_contrast'] = (max_vals - other_vals_mean) / other_vals_std

# %%
peaks_pb

# %%
# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))
peaks_pb.uns['timepoint_colors'] = [timepoint_colors[t] for t in timepoint_order]

# Plot with corrected contrast scores
sc.pl.umap(peaks_pb, 
           color='timepoint',
           # size=peaks_pb.obs['timepoint_contrast'],
           palette=timepoint_colors,
           save='_allpeaks_timepoint_viridis.png')

# %%
# plot for the celltype with the celltype color palette
# A module to define the color palettes used in this paper
import matplotlib.pyplot as plt
import seaborn as sns

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
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb, 
           color='celltype',
           # size=peaks_pb.obs['celltype_contrast']/np.max(peaks_pb.obs['celltype_contrast'])*20,
           palette=cell_type_color_dict,
           save='_celltype_allpeaks.png')

# %%
peaks_pb.obs["celltype"].value_counts()

# %%
# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))
peaks_pb.uns['timepoint_colors'] = [timepoint_colors[t] for t in timepoint_order]


# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb, 
           color='timepoint',
           palette=timepoint_colors,
           save='_timepoint_allpeaks.png')

# %%
peaks_pb.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_all_peaks.h5ad")

# %%

# %% [markdown]
# ### without log(1+p) transformation

# %%
# adata_pseudo.var["mean_norm"] = mean_peaks_norm
# adata_pseudo.var["var_norm"] = var_peaks_norm

# %%
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo.X = adata_pseudo.layers["normalized"]
peaks_pb_norm = adata_pseudo.copy().T
rsc.get.anndata_to_GPU(peaks_pb_norm) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb_norm) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_pb_norm, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_norm, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb_norm, min_dist=0.3)#, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_pb_norm)

# %%
sc.pl.umap(peaks_pb_norm, color=["n_cells_by_counts"])

# %%
peaks_pb_norm.var.head()

# %%
rsc.get.anndata_to_CPU(peaks_pb_norm) # moves `.X` to the GPU

# %%
sc.pl.umap(peaks_pb_norm, layer="log_norm", color=["optic_cup_0somites", "tail_bud_30somites"])

# %%
peaks_pb_norm.obs["mean_norm"]

# %%
peaks_pb_norm.obs = peaks_pb.obs.copy()

# %%
sc.pl.umap(peaks_pb_norm, color=["n_cells_by_counts", "mean_norm","var_norm"], vmax=[175, 100, 10000])

# %%
peaks_pb_norm.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_wo_log_all_peaks.h5ad")

# %%

# %% [markdown]
# ## 4. TF-IDF normalization (muon)
# explanation: Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).
# - TF: peak counts are normalized by total number of counts per pseudobulk. 
# - DF: total number of counts for each peak 
# - IDF: number of cells divided by DF

# %%
adata_pseudo = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/pseudobulk_ct_tp_by_peaks.h5ad")
adata_pseudo

# %%
# # convert back to the "sum" layer before the TF-IDF normalization
# adata_pseudo.X = adata_pseudo.layers["sum"].copy()
# adata_pseudo

# %%
# bring back the common scale factor (median of total counts - pseudobulks)
common_scale_factor = adata_pseudo.uns["common_scale_factor"]

# %%
# TF-IDF normalization
# mu.atac.pp.tfidf(adata_pseudo, scale_factor=common_scale_factor, from_layer="sum", to_layer="tfidf")

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ### check if sc.pp.scale changes the UMAP -> NO, it does not

# %%
# without sc.pp.scale

# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo.X = adata_pseudo.layers["normalized"]
peaks_pb_norm_v2 = adata_pseudo.copy().T
rsc.get.anndata_to_GPU(peaks_pb_norm_v2) # moves `.X` to the GPU

# Compute UMAP
# rsc.pp.scale(peaks_pb_norm) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_pb_norm_v2, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_norm_v2, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb_norm_v2, min_dist=0.3, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_pb_norm_v2)

# %%

# %% [markdown]
# ### plot the distribution of counts (pseudobulk groups)
#
# - summed counts
# - normalized counts
# - scale factors
# - log-normalized

# %%

# %% [markdown]
# ### UMAP without log(1+p)

# %%
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo.X = adata_pseudo.layers["normalized"]
peaks_pb = adata_pseudo.copy().T
rsc.get.anndata_to_GPU(peaks_pb) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_pb, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb, min_dist=0.3)#, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_pb)

# %%
adata = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_all_peaks.h5ad")
adata

# %%
peaks_pb.obs = adata.obs.copy()

# %%
sc.pl.umap(peaks_pb, color="timepoint")

# %%

# %%
adata_pseudo

# %%
adata_pseudo.layers["log_norm_division_by_total_counts"]

# %%
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo.X = adata_pseudo.layers["log_norm_division_by_total_counts"].copy()
peaks_pb_old = adata_pseudo.copy().T
rsc.get.anndata_to_GPU(peaks_pb_old) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb_old) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_pb_old, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_old, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb_old, min_dist=0.3)#, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_pb_old)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# Create dictionaries to map columns to celltype and timepoint
celltype_mapping = {}
timepoint_mapping = {}

# Parse var names
for col in peaks_pb.var.index:
    parts = col.rsplit('_', 1)
    if len(parts) == 2 and 'somites' in parts[1]:
        celltype = parts[0]
        timepoint = parts[1]
        celltype_mapping[col] = celltype
        timepoint_mapping[col] = timepoint

# Get unique celltypes and timepoints
unique_celltypes = set(celltype_mapping.values())
unique_timepoints = set(timepoint_mapping.values())

# Create new obs columns for each celltype and timepoint
for celltype in unique_celltypes:
    # Get columns for this celltype
    celltype_cols = [col for col, ct in celltype_mapping.items() if ct == celltype]
    # Sum accessibility across all timepoints for this celltype
    peaks_pb.obs[f'accessibility_{celltype}'] = peaks_pb.X[:, [peaks_pb.var.index.get_loc(col) for col in celltype_cols]].sum(axis=1)

for timepoint in unique_timepoints:
    # Get columns for this timepoint
    timepoint_cols = [col for col, tp in timepoint_mapping.items() if tp == timepoint]
    # Sum accessibility across all celltypes for this timepoint
    peaks_pb.obs[f'accessibility_{timepoint}'] = peaks_pb.X[:, [peaks_pb.var.index.get_loc(col) for col in timepoint_cols]].sum(axis=1)


# %%
# For timepoints
timepoint_cols = ['accessibility_0somites', 'accessibility_5somites', 'accessibility_10somites', 
                 'accessibility_15somites', 'accessibility_20somites', 'accessibility_30somites']
timepoint_vals = np.array([peaks_pb.obs[col] for col in timepoint_cols]).T

# Find max timepoint for each peak
max_timepoint_idx = np.argmax(timepoint_vals, axis=1)
timepoint_names = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
peaks_pb.obs['timepoint'] = [timepoint_names[i] for i in max_timepoint_idx]

# Calculate corrected timepoint contrast
max_vals = np.max(timepoint_vals, axis=1)
# Calculate mean and std excluding max value for each peak
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
peaks_pb.obs['timepoint_contrast'] = (max_vals - other_vals_mean) / other_vals_std

# For celltypes
celltype_cols = [col for col in peaks_pb.obs.columns 
                if col.startswith('accessibility_') and 'somites' not in col]
celltype_vals = np.array([peaks_pb.obs[col] for col in celltype_cols]).T

# Find max celltype for each peak
max_celltype_idx = np.argmax(celltype_vals, axis=1)
celltype_names = [col.replace('accessibility_', '') for col in celltype_cols]
peaks_pb.obs['celltype'] = [celltype_names[i] for i in max_celltype_idx]

# Calculate corrected celltype contrast
max_vals = np.max(celltype_vals, axis=1)
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(celltype_vals, max_celltype_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(celltype_vals, max_celltype_idx)])
peaks_pb.obs['celltype_contrast'] = (max_vals - other_vals_mean) / other_vals_std

# %%
peaks_pb

# %%
# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))
peaks_pb.uns['timepoint_colors'] = [timepoint_colors[t] for t in timepoint_order]

# Plot with corrected contrast scores
sc.pl.umap(peaks_pb, 
           color='timepoint',
           # size=peaks_pb.obs['timepoint_contrast'],
           palette=timepoint_colors,
           save='_allpeaks_timepoint_viridis.png')

# %%
# plot for the celltype with the celltype color palette
# A module to define the color palettes used in this paper
import matplotlib.pyplot as plt
import seaborn as sns

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
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb, 
           color='celltype',
           #size=peaks_pb.obs['celltype_contrast'],
           palette=cell_type_color_dict,
           save='_celltype_allpeaks.png')

# %%

# %%
peaks_pb.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_all_peaks.h5ad")

# %%

# %%

# %%

# %% [markdown]
# ## UPDATE: re-normalize the counts with the peak width
# - the rationale here is that the longer/wider peaks will likely have more reads captured, so there might need to be a peak width normalization.

# %%
# pseudobulk-by-peaks object
adata_pseudo

# %%
# 1) normalize the counts

# %%
peaks_pb = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_master.h5ad")
peaks_pb

# %%
# normalize the peak counts by "peak width(length)"
# Step 1: Start with normalized counts (already depth-normalized)
peaks_pb.X = peaks_pb.layers["normalized"]

# Step 2: Normalize by peak length to get "reads per kilobase" equivalent
# Divide by peak width, then multiply by median width to maintain scale
peak_widths = peaks_pb.obs["length"].values[:, np.newaxis]  # Shape: (n_peaks, 1)
median_width = np.median(peaks_pb.obs["length"])

# Perform width normalization
peaks_pb.X = peaks_pb.X / peak_widths * median_width

# Alternative: if you want to store the intermediate steps
peaks_pb.layers["width_normalized"] = peaks_pb.X.copy()


# %%
rsc.get.anndata_to_GPU(peaks_pb) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_pb, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb, min_dist=0.3)#, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_pb)

# %%
sc.pl.umap(peaks_pb, color="celltype")

# %%

# %%
len(np.sum(peaks_pb.layers["normalized"],axis=1))

# %%
# subset for the peaks <=1000 bp (most of the peaks)
peaks_pb_subset = peaks_pb[peaks_pb.obs["length"]<=1000]
plt.scatter(peaks_pb_subset.obs["length"],np.sum(peaks_pb_subset.layers["normalized"],axis=1), alpha=0.2, s=0.1)
plt.xlabel("peak width")
plt.ylabel("aggregated read counts (normalized)")
plt.ylim([0, 2000])
plt.grid(False)
plt.show()

# %%
peaks_pb_subset
# subset for the peaks <=1000 bp (most of the peaks)
peaks_pb_subset = peaks_pb[peaks_pb.obs["length"]<=1000]
plt.scatter(peaks_pb_subset.obs["length"],np.sum(peaks_pb_subset.layers["normalized"],axis=1), alpha=0.2, s=0.1)
plt.xlabel("peak width")
plt.ylabel("aggregated read counts (normalized)")
plt.ylim([0, 2000])
plt.grid(False)
plt.show()


# %%
plt.scatter(peaks_pb.obs["length"],np.sum(peaks_pb.layers["sum"],axis=1), alpha=0.05, s=0.1)
plt.xlabel("peak width")
plt.ylabel("aggregated read counts (normalized)")
plt.grid(False)
plt.show()

# %%
from scipy.interpolate import UnivariateSpline
from scipy.stats import binned_statistic

# Extract your data
peak_widths = peaks_pb_subset.obs["length"].values
aggregated_counts = np.array(np.sum(peaks_pb_subset.layers["sum"], axis=1)).flatten()

# Method 1: Quantile-based spline (RECOMMENDED)
# Bin the data and compute quantiles
n_bins = 50
bin_edges = np.linspace(peak_widths.min(), peak_widths.max(), n_bins + 1)

# Compute median and other quantiles per bin
median_vals, bin_edges, _ = binned_statistic(
    peak_widths, aggregated_counts, statistic='median', bins=bin_edges
)
q25_vals, _, _ = binned_statistic(
    peak_widths, aggregated_counts, 
    statistic=lambda x: np.percentile(x, 25), bins=bin_edges
)
q75_vals, _, _ = binned_statistic(
    peak_widths, aggregated_counts, 
    statistic=lambda x: np.percentile(x, 75), bins=bin_edges
)

# Get bin centers for x-values
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Remove any NaN values (from empty bins)
valid = ~np.isnan(median_vals)
bin_centers_valid = bin_centers[valid]
median_vals_valid = median_vals[valid]
q25_vals_valid = q25_vals[valid]
q75_vals_valid = q75_vals[valid]

# Fit splines with appropriate smoothing
# s parameter controls smoothing (higher = smoother)
median_spline = UnivariateSpline(bin_centers_valid, median_vals_valid, s=1e6, k=3)
q25_spline = UnivariateSpline(bin_centers_valid, q25_vals_valid, s=1e6, k=3)
q75_spline = UnivariateSpline(bin_centers_valid, q75_vals_valid, s=1e6, k=3)

# Generate smooth curve for plotting
x_smooth = np.linspace(peak_widths.min(), peak_widths.max(), 1000)
y_median_smooth = median_spline(x_smooth)
y_q25_smooth = q25_spline(x_smooth)
y_q75_smooth = q75_spline(x_smooth)

# Calculate correlation
from scipy.stats import pearsonr, spearmanr
pearson_r, pearson_p = pearsonr(peak_widths, aggregated_counts)
spearman_r, spearman_p = spearmanr(peak_widths, aggregated_counts)

print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.2e}")
print(f"Spearman correlation: r = {spearman_r:.4f}, p = {spearman_p:.2e}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot
ax.scatter(peak_widths, aggregated_counts, alpha=0.05, s=0.1, color='blue', 
           label='Individual peaks', rasterized=True)

# Spline fits
ax.plot(x_smooth, y_median_smooth, 'r-', linewidth=2.5, label='Median spline', zorder=10)
ax.fill_between(x_smooth, y_q25_smooth, y_q75_smooth, 
                alpha=0.3, color='red', label='25th-75th percentile', zorder=5)

# Add correlation text
ax.text(0.05, 0.95, 
        f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Peak Width (bp)', fontsize=12)
ax.set_ylabel('Aggregated Read Counts (normalized)', fontsize=12)
ax.set_title('Relationship Between Peak Width and Read Counts', fontsize=14)
ax.legend(loc='lower right')
ax.grid(False)

plt.tight_layout()
plt.savefig('peak_width_spline_fit.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
