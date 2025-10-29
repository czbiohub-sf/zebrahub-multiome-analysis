# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: sc_rapids
#     language: python
#     name: sc_rapids
# ---

# %% [markdown]
# ## pre-processing the peak UMAP
# - last updated: 04/01/2025
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
import muon as mu # added muon
import seaborn as sns
import scipy.sparse
from scipy.io import mmread

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# %%

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
#

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 1-1. [old] scaling factor = total number of counts per pseudobulk group
#
# - in this case, the normalized counts will be very very small digits, so might affect numerical roundup.
# - Also, the over-dispersion will be much smaller than 1.

# %%
# def analyze_peaks_with_normalization(adata, celltype_key='annotation_ML_coarse', 
#                                timepoint_key='dev_stage'):
#     # Calculate statistics
#     adata.obs['total_counts'] = np.array(adata.X.sum(axis=1))
#     group_means = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].mean()
#     cells_per_group = adata.obs.groupby([celltype_key, timepoint_key]).size()
#     scale_factors = cells_per_group * group_means
    
#     # Compute pseudobulk
#     ident_cols = [celltype_key, timepoint_key]
#     adata_pseudo = sc.get.aggregate(adata, ident_cols, func='sum')
#     # have the "sum" layer as the count matrix
#     adata_pseudo.X = adata_pseudo.layers["sum"].copy()

#     # Extract celltype and timepoint 
#     celltype_timepoint = pd.DataFrame({
#         'celltype': [
#             '_'.join(x.split('_')[:-1]) for x in adata_pseudo.obs.index
#         ],
#         'timepoint': [
#             x.split('_')[-1] for x in adata_pseudo.obs.index
#         ]
#     }, index=adata_pseudo.obs.index)
    
#     # Handle arrays appropriately
#     X = adata_pseudo.X
#     if isinstance(X, np.ndarray):
#         normalized_counts = np.zeros_like(X)
#     else:
#         normalized_counts = np.zeros_like(X.toarray())
    
#     # Normalize counts
#     for i, idx in enumerate(adata_pseudo.obs.index):
#         celltype = celltype_timepoint.loc[idx, 'celltype']
#         timepoint = celltype_timepoint.loc[idx, 'timepoint']
#         sf = scale_factors[(celltype, timepoint)]
#         if isinstance(X, np.ndarray):
#             normalized_counts[i, :] = X[i] / sf
#         else:
#             normalized_counts[i, :] = X[i].toarray() / sf
    
    
#     # Store results
#     adata_pseudo.layers['normalized'] = normalized_counts
#     adata_pseudo.obs['scale_factor'] = [scale_factors[(ct, tp)] 
#                                       for ct, tp in celltype_timepoint.values]
#     adata_pseudo.obs['n_cells'] = [cells_per_group[(ct, tp)] 
#                                  for ct, tp in celltype_timepoint.values]
#     adata_pseudo.obs['mean_depth'] = [group_means[(ct, tp)] 
#                                    for ct, tp in celltype_timepoint.values]
    
#     return adata_pseudo

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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 1-2. computing the UMAP - 50K highly variable peaks

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
# def analyze_peaks_with_normalization(
#     adata, 
#     celltype_key='annotation_ML_coarse', 
#     timepoint_key='dev_stage'
# ):
#     """
#     1) Compute each cell's total_counts (sum of peaks/reads).
#     2) For each (celltype, timepoint) group, compute the total_coverage 
#        = sum of total_counts from all cells in that group.
#     3) Create pseudobulk by summing (func='sum') each group's cells for the peaks matrix.
#     4) The common_scale_factor = median of all group_total_coverage.
#     5) For each group g, normalized_pseudobulk = raw_pseudobulk * (common_scale_factor / group_total_coverage[g]).

#     Returns
#     -------
#     adata_pseudo : an AnnData with:
#         - .X = raw pseudobulk counts
#         - layers['normalized'] = scaled pseudobulk counts
#         - obs['total_coverage'] = group's raw coverage
#         - obs['scale_factor'] = how much that group's coverage was scaled
#         - obs['n_cells'] and obs['mean_depth'] optionally stored as well
#         - uns['common_scale_factor'] = the median coverage used for scaling
#     """

#     # 1) total_counts per cell
#     adata.obs['total_counts'] = np.array(adata.X.sum(axis=1))

#     # 2) total_coverage per group (sum of total_counts)
#     group_total_coverage = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].sum()
#     # [Max] this should be done after the pseudo-bulk

#     # 3) Pseudobulk by summing group cells
#     ident_cols = [celltype_key, timepoint_key]
#     adata_pseudo = sc.get.aggregate(adata, ident_cols, func='sum')
#     # Copy the summed counts into .X
#     adata_pseudo.X = adata_pseudo.layers["sum"].copy()

#     # Split the new obs index (e.g. "Astro_dev_stage1") back into celltype/timepoint
#     celltype_timepoint = pd.DataFrame({
#         'celltype': ['_'.join(x.split('_')[:-1]) for x in adata_pseudo.obs.index],
#         'timepoint': [x.split('_')[-1] for x in adata_pseudo.obs.index]
#     }, index=adata_pseudo.obs.index)

#     # Prepare for normalized counts
#     X = adata_pseudo.X
#     if not isinstance(X, np.ndarray):
#         X = X.toarray()

#     # 4) common_scale_factor = median of group_total_coverage
#     common_scale_factor = np.median(group_total_coverage.values)

#     # 5) Rescale each group's pseudobulk
#     normalized_counts = np.zeros_like(X)
#     coverage_list = []
#     scale_factor_list = []

#     # [Max] vectorize this: scale_factor = group_total_coverage/common_scale_factor
#     # the scale factor should be centered around 1. 
#     # Make sure that it's a row vector 
#     # adata.X/scale_factor
#     for i, idx in enumerate(adata_pseudo.obs.index):
#         ct = celltype_timepoint.loc[idx, 'celltype']
#         tp = celltype_timepoint.loc[idx, 'timepoint']
#         coverage_g = group_total_coverage[(ct, tp)]

#         # Scale factor = common_scale_factor / group's total coverage
#         scale_g = common_scale_factor / coverage_g
#         normalized_counts[i, :] = X[i, :] * scale_g
        
#         coverage_list.append(coverage_g)
#         scale_factor_list.append(scale_g)

#     # Store normalized counts in a new layer
#     adata_pseudo.layers['normalized'] = normalized_counts

#     # Record coverage and scaling info in .obs
#     adata_pseudo.obs['total_coverage'] = coverage_list
#     adata_pseudo.obs['scale_factor'] = scale_factor_list

#     # Optionally, also store #cells and mean_depth
#     group_ncells = adata.obs.groupby([celltype_key, timepoint_key]).size()
#     group_mean_depth = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].mean()

#     n_cells_list = []
#     mean_depth_list = []
#     for i, idx in enumerate(adata_pseudo.obs.index):
#         ct = celltype_timepoint.loc[idx, 'celltype']
#         tp = celltype_timepoint.loc[idx, 'timepoint']
#         n_cells_list.append(group_ncells[(ct, tp)])
#         mean_depth_list.append(group_mean_depth[(ct, tp)])
#     adata_pseudo.obs['n_cells'] = n_cells_list
#     adata_pseudo.obs['mean_depth'] = mean_depth_list

#     # Save the "common" scale factor in .uns
#     adata_pseudo.uns['common_scale_factor'] = common_scale_factor

#     return adata_pseudo

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
# import the peaks_pb_norm
peaks_pb_norm = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_wo_log_all_peaks.h5ad")
peaks_pb_norm

# %%
sc.pl.umap(peaks_pb_norm)
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_norm, 
           color='timepoint',
           # size=peaks_pb.obs['timepoint_contrast'],
           palette=timepoint_colors,
           save='_allpeaks_norm_timepoint_viridis.png')

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_norm, 
           color='celltype',
           #size=peaks_pb.obs['celltype_contrast'],
           palette=cell_type_color_dict,
           save='_allpeaks_norm_celltype.png')


# %%
# 3D UMAP of the peaks_pb_norm
peaks_pb_norm.obsm["X_umap_2d"] = peaks_pb_norm.obsm["X_umap"]

rsc.tl.umap(peaks_pb_norm, n_components=3, min_dist=0.1, random_state=42)
peaks_pb_norm.obsm["X_umap_3d"] = peaks_pb_norm.obsm["X_umap"]

# %%
# save the peaks_pb_norm with 3D UMAP
peaks_pb_norm.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_wo_log_all_peaks_3d_umap.h5ad")
# %%
# plot the 3D UMAP
import plotly.express as px
import plotly.graph_objects as go
def plot_3d_umap(umap_array, 
                 color_array=None,
                 color_label='cluster',
                 title='3D UMAP',
                 point_size=3,
                 opacity=0.7,
                 height=800,
                 width=1000):
    """
    Create an interactive 3D UMAP visualization using plotly.
    
    Parameters:
    -----------
    umap_array : np.array
        Array of shape (n_cells, 3) containing 3D UMAP coordinates
    color_array : array-like, optional
        Array of values/categories to color the points by
    color_label : str, optional
        Label for the color legend
    title : str, optional
        Title of the plot
    point_size : int, optional
        Size of the scatter points
    opacity : float, optional
        Opacity of the points (0-1)
    height : int, optional
        Height of the plot in pixels
    width : int, optional
        Width of the plot in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    
    # Create a DataFrame with UMAP coordinates
    df = pd.DataFrame(
        umap_array,
        columns=['UMAP1', 'UMAP2', 'UMAP3']
    )
    
    if color_array is not None:
        df[color_label] = color_array.values  # aligns by position, not index
        
        # Create figure with color
        fig = px.scatter_3d(
            df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color=color_label,
            title=title,
            opacity=opacity,
            height=height,
            width=width
        )
    else:
        # Create figure without color
        fig = px.scatter_3d(
            df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            title=title,
            opacity=opacity,
            height=height,
            width=width
        )
    
    # Update marker size
    fig.update_traces(marker_size=point_size)
    
    # Update layout for better visualization
    fig.update_layout(
        scene = dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3',
            aspectmode='cube'  # This ensures equal aspect ratio
        ),
        showlegend=True
    )
    
    return fig

# %%
# Example usage with your data:
umap_coords = peaks_pb_norm.obsm['X_umap_3d']  # Your UMAP coordinates
# If you have clusters or other metadata to color by:
celltype = peaks_pb_norm.obs['celltype']  # or whatever your metadata column is
timepoint = peaks_pb_norm.obs['timepoint'] 

# %%
# First create a dataframe with your UMAP coordinates and metadata
df = pd.DataFrame(
    peaks_pb_norm.obsm['X_umap_3d'],  # Your 3D UMAP coordinates
    columns=['UMAP1', 'UMAP2', 'UMAP3']
)

# Add your celltype data as a new column
df['celltype'] = peaks_pb_norm.obs['celltype'].values
df['timepoint'] = peaks_pb_norm.obs['timepoint'].values
# %%
# Create the 3D plot
fig = px.scatter_3d(
    df,
    x='UMAP1',
    y='UMAP2', 
    z='UMAP3',
    color='celltype',  # Color by celltype
    labels={'color': 'Cell Type'},  # Legend title
    title='3D UMAP',
    opacity=0.3
)

# # Optional: update the layout for better visualization
# fig.update_layout(
#     scene=dict(
#         aspectmode='cube'  # This ensures equal aspect ratio
#     )
# )
fig.show()
# %%

plot_3d_umap(umap_coords, color_array=celltype, color_label='celltype', 
             point_size=3, opacity=0.5, title='3D UMAP')


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
print(common_scale_factor)
# %%
# TF-IDF normalization
mu.atac.pp.tfidf(adata_pseudo, scale_factor=common_scale_factor, from_layer="sum", to_layer="tfidf")
# NOTE that the tf-idf normalization is very memory-intenstive, requiring 3TB of memory in the case of 640K peaks.

# %%
from scipy.sparse import csr_matrix

X = adata_pseudo.layers["sum"]  # shape (n_groups, n_peaks), presumably sparse

# Term Frequency = row-normalized. 
# row_sums has shape (n_groups,).
row_sums = np.asarray(X.sum(axis=1)).ravel()
# Avoid division by zero
row_sums[row_sums == 0] = 1

# If X is CSR, row-sum normalization can be done with spdiags or multiply:
tf = csr_matrix(X).copy()  # ensure a copy if we want to preserve original
for i in range(tf.shape[0]):
    start = tf.indptr[i]
    end = tf.indptr[i+1]
    if end > start:
        tf.data[start:end] /= row_sums[i]

# Optionally multiply by scale_factor if you want "CP10k"
scale_factor = common_scale_factor
tf.data *= scale_factor

# Then log TF if desired
tf.data = np.log1p(tf.data)

# IDF
idf = X.shape[0] / np.asarray(X.sum(axis=0)).ravel()  # #groups / sum_of_counts_in_col
idf = np.log1p(idf)  # if log_idf

# Finally do column-wise multiply in a memory-friendly manner:
# tf is row-major (CSR), so column multiplication is less direct. If it’s CSC, it’s simpler.
# One approach: convert to CSC for efficient column multiplication:
tf = tf.tocsc()
for j in range(tf.shape[1]):
    start = tf.indptr[j]
    end = tf.indptr[j+1]
    if end > start:
        tf.data[start:end] *= idf[j]

tf = tf.tocsr()  # convert back to CSR if you prefer

# Now `tf` is your TF-IDF matrix. Assign it to a new layer:
adata_pseudo.layers["tfidf"] = tf

# %%
# transpose the adata_pseudo to make it peaks-by-cells
peaks_pb_tfidf = adata_pseudo.copy().T

# make sure the TF-IDF matrix is in .X
peaks_pb_tfidf.X = peaks_pb_tfidf.layers["tfidf"]

# compute the LSI (Latent Semantic Indexing) for the TF-IDF matrix
mu.atac.tl.lsi(peaks_pb_tfidf, n_comps=40)

# %% 
peaks_pb_tfidf
# Row sums = coverage of each peak across all pseudobulk columns.
peak_coverage = np.asarray(peaks_pb_tfidf.X.sum(axis=1)).ravel()

# (Optional) Store it in obs for convenience:
peaks_pb_tfidf.obs["peak_coverage"] = peak_coverage
# %%
# plot the LSI components
plt.scatter(peaks_pb_tfidf.obsm["X_lsi"][:,0],
peaks_pb_tfidf.obs["peak_coverage"])

# %%
# plot the LSI components
plt.scatter(peaks_pb_tfidf.obsm["X_lsi"][:,0],
            peaks_pb_tfidf.obsm["X_lsi"][:,1],
            c=peak_coverage,
            cmap="viridis")

# %%
# remove the first LSI component (as it's highly correlated with the sequencing depth)
X_lsi = peaks_pb_tfidf.obsm["X_lsi"]  # shape (n_peaks, n_comps)
X_lsi_no_first = X_lsi[:, 1:]     # keep comps 2..n_comps
peaks_pb_tfidf.obsm["X_lsi_filtered"] = X_lsi_no_first

# %%
# Use the LSI components to compute the UMAP
peaks_pb_tfidf
# move the count matrix to the GPU
rsc.get.anndata_to_GPU(peaks_pb_tfidf)
# compute the neighbors
rsc.pp.neighbors(peaks_pb_tfidf, n_neighbors=15, n_pcs=39, use_rep="X_lsi_filtered")
# compute the UMAP      
rsc.tl.umap(peaks_pb_tfidf, min_dist=0.3, random_state=42)
# plot the UMAP
sc.pl.umap(peaks_pb_tfidf)

# %%
# copy over the obs from the original peaks_pb_norm
peaks_pb_tfidf.obs = peaks_pb_norm.obs.copy()
peaks_pb_tfidf
# %%
# generate some plots
sc.pl.umap(peaks_pb_tfidf, color=["celltype", "timepoint", "total_counts"])
# %%
# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))
peaks_pb_tfidf.uns['timepoint_colors'] = [timepoint_colors[t] for t in timepoint_order]

# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_tfidf, 
           color='timepoint',
           # size=peaks_pb.obs['timepoint_contrast'],
           palette=timepoint_colors,
           save='_allpeaks_tf_idf_timepoint_viridis.png')

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_tfidf, 
           color='celltype',
           #size=peaks_pb.obs['celltype_contrast'],
           palette=cell_type_color_dict,
           save='_allpeaks_tf_idf_celltype.png')

# %%
peaks_pb_tfidf.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_tf_idf_all_peaks.h5ad")

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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Old count matrix (normalized by the 1/total_counts, then log(1+p) transformation, and sc.pp.scale)

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
