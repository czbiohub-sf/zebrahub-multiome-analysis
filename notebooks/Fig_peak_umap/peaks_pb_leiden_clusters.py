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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## pre-processing the peak UMAP
# - last updated: 04/03/2025
# - pseudobulk the scATAC-seq dataset using the leiden clusters and timepoints

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
# figure parameter setting
# %matplotlib inline
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_leiden/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %% [markdown]
# ### import the adata object
# %%
# import the adata_peaks with raw counts
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad")
adata_peaks

# %%
# import the adata for cells-by-genes (RNA) to transfer some metadata - especially the leiden clusters and celltypes
adata_RNA = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_leiden.h5ad")
adata_RNA

# %%
# filter out the "low_quality_cells" from adata_RNA, and copy over the metadata to adata_peaks
# adata_peaks was already filtered out for the "low_quality_cells"
adata_RNA = adata_RNA[adata_RNA.obs_names.isin(adata_peaks.obs_names)]

# %%
adata_RNA.obs["annotation_ML_coarse"] = adata_RNA.obs_names.map(adata_peaks.obs["annotation_ML_coarse"])
adata_RNA.obs["dev_stage"] = adata_RNA.obs_names.map(adata_peaks.obs["dev_stage"])


# %%
sc.pl.embedding(adata_RNA, "X_wnn.umap", color=["annotation_ML_coarse", "dev_stage"])

# %%
# # copy over the metadata from adata_RNA to adata_peaks
adata_peaks.obs["leiden_0.01_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.01_merged"])
adata_peaks.obs["leiden_0.03_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.03_merged"])
adata_peaks.obs["leiden_0.05_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.05_merged"])
adata_peaks.obs["leiden_0.1_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.1_merged"])
adata_peaks.obs["leiden_0.2_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.2_merged"])
adata_peaks.obs["leiden_0.3_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.3_merged"])
adata_peaks.obs["leiden_0.4_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.4_merged"])
adata_peaks.obs["leiden_0.5_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.5_merged"])
adata_peaks.obs["leiden_0.6_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.6_merged"])
adata_peaks.obs["leiden_0.7_merged"] = adata_peaks.obs_names.map(adata_RNA.obs["leiden_0.7_merged"])

adata_peaks.obsm["X_wnn.umap"] = adata_RNA.obsm["X_wnn.umap"]
sc.pl.embedding(adata_peaks, "X_wnn.umap", color=["leiden_0.05_merged", "annotation_ML_coarse", "dev_stage"])


# %%
sc.pl.embedding(adata_peaks, "X_wnn.umap", 
                color=["leiden_0.01_merged", "leiden_0.1_merged",
                       "leiden_0.8_merged", "leiden_2_merged",
                       "leiden_5_merged", "leiden_10_merged"], ncols=3)


# %%
sc.pl.embedding(adata_peaks, "X_wnn.umap", 
                color=["leiden_0.01_merged", "leiden_0.1_merged",
                       "leiden_0.8_merged", "leiden_2_merged",
                       "leiden_3_merged", "leiden_5_merged"], ncols=3, save="_cells_joint_leiden_res.png")

# %%
for res in [0.01, 0.1, 0.8, 3]:
       # plot the UMAP for the given resolution
       sc.pl.embedding(adata_RNA, "X_wnn.umap", 
                       color=f"leiden_{res}_merged",
                       save=f"_cells_joint_leiden_res_{res}.png")

# %%
adata_RNA.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_leiden_filtered.h5ad")

# %%
# # # copy over the metadata from adata_RNA to adata_peaks
# adata_peaks.obs["annotation_ML_coarse"] = adata_peaks.obs_names.map(adata_test.obs["annotation_ML_coarse"])
# adata_peaks.obs["dev_stage"] = adata_peaks.obs_names.map(adata_test.obs["dev_stage"])
# adata_peaks

# %%
# # save the new master object (raw counts, and also sparse matrix with float32 format)
# adata_peaks.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad")

# %%
# check the .X layer in the adata_peaks (make sure that these are the same as the raw counts)
adata_peaks[0:10,0:10].X.todense()

# %%
# compute the QC metrics for the peaks

# # move the counts to GPU
# rsc.get.anndata_to_GPU(adata_peaks) # moves `.X` to the GPU
# compute the QC metrics
# rsc.pp.calculate_qc_metrics(adata_peaks, expr_type='counts', 
#                             var_type='peaks', qc_vars=None, log1p=False)

# %%
# save the new master object (raw counts, and also sparse matrix with float32 format)
# adata_peaks.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered_v2.h5ad")


# %% [markdown]
# ## 1. define the function to pseudobulk across {cluster, timepoint}
# Import the custom module
# sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/utils/")
# from pseudobulk_utils import create_normalized_peak_pseudobulk

# %%
# help(create_normalized_peak_pseudobulk)
adata_peaks.obs_keys()

# %%
# import the pseudobulk function
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/utils/")
from pseudobulk_utils import create_normalized_peak_pseudobulk

# %%
# test the function
adata_pseudo = create_normalized_peak_pseudobulk(
    adata_peaks,
    cluster_key='annotation_ML_coarse',
    timepoint_key='orig.ident',
)
adata_pseudo

# %%
# test the counts from the pseudobulked object
adata_pseudo[0:10,0:10].layers["normalized"]

# %%
# compute the UMAP
# transpose for peaks-by-celltype&timepoint (pseudo-bulked)
adata_pseudo.X = adata_pseudo.layers["normalized"]
peaks_pb_norm = adata_pseudo.copy().T
rsc.get.anndata_to_GPU(peaks_pb_norm) # moves `.X` to the GPU

# Compute UMAP
rsc.pp.scale(peaks_pb_norm)
rsc.pp.pca(peaks_pb_norm, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_norm, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb_norm, min_dist=0.3, random_state=42)
# plot the UMAP
sc.pl.umap(peaks_pb_norm)

# %%
adata_peaks.obs.keys()
# Pseudobulk the adata object across {cluster, timepoint}

# %%
# Pseudobulk the adata object across {cluster, timepoint}
# First, the list of leiden clustering resolutions
list_res_leiden = [0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] #  1, 2, 3, 4, 5, 6, 7, 8, 9, 10
pseudobulk_dict = {}  # Store adata_pseudo objects
peaks_norm_dict = {}  # Store peaks_pb_norm objects

for res in list_res_leiden:
    # Create pseudobulk object
    adata_pseudo = create_normalized_peak_pseudobulk(
        adata_peaks,
        cluster_key=f'leiden_{res}_merged',
        timepoint_key='dev_stage',
    )
    
    # Create transposed normalized object
    adata_pseudo.X = adata_pseudo.layers["normalized"]
    peaks_pb_norm = adata_pseudo.copy().T
    
    # Compute UMAP for peaks_pb_norm
    rsc.get.anndata_to_GPU(peaks_pb_norm)
    rsc.pp.scale(peaks_pb_norm)
    # Check number of columns and use minimum between that and 100 for PCA
    n_comps = min(100, peaks_pb_norm.n_vars)
    rsc.pp.pca(peaks_pb_norm, n_comps=n_comps, use_highly_variable=False)
    # Use min(n_comps, 40) for neighbors to ensure we don't use more PCs than available
    n_pcs = min(40, n_comps)
    rsc.pp.neighbors(peaks_pb_norm, n_neighbors=15, n_pcs=n_pcs)
    rsc.tl.umap(peaks_pb_norm, min_dist=0.3, random_state=42)
    sc.pl.umap(peaks_pb_norm)
    
    # Store both objects in dictionaries with resolution as key
    pseudobulk_dict[f'leiden_{res}_merged'] = adata_pseudo
    peaks_norm_dict[f'leiden_{res}_merged'] = peaks_pb_norm
    print(f"leiden_{res}_merged", " done")

    # save the peaks_pb_norm object
    peaks_pb_norm.write_h5ad(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_{res}_merged.h5ad")
# Now you can access any resolution's objects like this:
# pseudobulk_dict['leiden_0.5_merged']
# peaks_norm_dict['leiden_0.5_merged']

# %%
# import the peaks_norm object with annotations
peaks_norm_ref = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_wo_log_all_peaks.h5ad")
peaks_norm_ref

# %%
## RESUME HERE
# import the peaks_norm objects computed from the leiden clustering resolutions
list_res_leiden = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
peaks_norm_dict = {}  # Store peaks_pb_norm objects

for res in list_res_leiden:
    peaks_norm = sc.read_h5ad(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_{res}_merged.h5ad")
    peaks_norm_dict[f'leiden_{res}_merged'] = peaks_norm
    print(f'leiden_{res}_merged', " done")


# %%
list_res_leiden = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]

# print the number of unique clusters for each resolution
for res in list_res_leiden:
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    print(f'leiden_{res}_merged', " has ", n_clusters, " clusters")


# %%
for res in [0.01, 0.1, 0.8, 3]:
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    # peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    print(f'leiden_{res}_merged', " has ", n_clusters, " clusters")
    sc.pl.umap(peaks_norm)


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

# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))



# %% [markdown]
# ## generate UMAP plots from different leiden clustering resolutions

# %%
list_res_leiden_plot = [
    0.01, 0.1, 0.8, 3
]
# Create a figure with 3x4 subplots
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the resolutions and plot UMAPs
for i, res in enumerate(list_res_leiden_plot):
    # Get the corresponding peaks_norm object
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # transfer the annotations from the peaks_norm_ref object to the peaks_norm object
    peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    # QC metrics
    # peaks_norm.obs["n_cells_by_counts"] = peaks_norm_ref.obs["n_cells_by_counts"]
    # peaks_norm.obs["total_counts"] = peaks_norm_ref.obs["total_counts"]
    
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    
    # # Plot UMAP on the corresponding subplot
    # sc.pl.umap(peaks_norm, 
    #            ax=axes[i],
    #            show=False,
    #            title=f'Resolution {res}\n({n_clusters} clusters)',
    #            frameon=False)
    # Plot with corrected contrast scores
    sc.pl.umap(peaks_norm, 
               color='celltype',
               palette=cell_type_color_dict,
               ax=axes[i],
               show=False,
               title=f'Resolution {res}\n({n_clusters} clusters)',
               frameon=False, legend_loc=None)
    # Remove axis labels for cleaner look
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title
fig.suptitle('peak UMAPs by leiden resolution', y=1.02, fontsize=16)

plt.tight_layout()
plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_leiden/peaks_pb_leiden_clusters_umap_4res_celltypes.png", dpi=600)
plt.show()

# %%
# Create a figure with 3x4 subplots
fig, axes = plt.subplots(3, 4, figsize=(20, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the resolutions and plot UMAPs
for i, res in enumerate(list_res_leiden):
    # Get the corresponding peaks_norm object
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # transfer the annotations from the peaks_norm_ref object to the peaks_norm object
    peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    
    # # Plot UMAP on the corresponding subplot
    # sc.pl.umap(peaks_norm, 
    #            ax=axes[i],
    #            show=False,
    #            title=f'Resolution {res}\n({n_clusters} clusters)',
    #            frameon=False)
    # Plot with corrected contrast scores
    sc.pl.umap(peaks_norm, 
               color='celltype',
               palette=cell_type_color_dict,
               ax=axes[i],
               show=False,
               title=f'Resolution {res}\n({n_clusters} clusters)',
               frameon=False, legend_loc=None)
    # Remove axis labels for cleaner look
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title
fig.suptitle('UMAP of Peak Accessibility by Leiden Resolution', y=1.02, fontsize=16)

plt.tight_layout()
plt.show()
# %%
# Create a figure with 3x4 subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the resolutions and plot UMAPs
for i, res in enumerate(list_res_leiden):
    # Get the corresponding peaks_norm object
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # transfer the annotations from the peaks_norm_ref object to the peaks_norm object
    peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    
    # # Plot UMAP on the corresponding subplot
    # sc.pl.umap(peaks_norm, 
    #            ax=axes[i],
    #            show=False,
    #            title=f'Resolution {res}\n({n_clusters} clusters)',
    #            frameon=False)
    # Plot with corrected contrast scores
    sc.pl.umap(peaks_norm, 
               color='celltype',
               palette=cell_type_color_dict,
               ax=axes[i],
               show=False,
               title=f'Resolution {res}\n({n_clusters} clusters)',
               frameon=False, legend_loc=None)
    # Remove axis labels for cleaner look
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title
fig.suptitle('UMAP of Peak Accessibility by Leiden Resolution', y=1.02, fontsize=16)

plt.tight_layout()
plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_leiden/peaks_pb_leiden_clusters_umap_low_res.png", dpi=600)
plt.show()

# %%
list_res_leiden_plot = [
    0.01,  # 6 clusters - Very broad clustering
    0.05,  # 11 clusters - Broad clustering
    0.2,   # 24 clusters - Starting to get more granular
    0.4,   # 32 clusters - Moderate clustering
    0.8,   # 43 clusters - Moderate clustering
    1,   # 51 clusters - More granular
    1.5,   # 61 clusters - More granular
    3,   # 87 clusters - Fine-grained
    5,   # 116 clusters - Very fine-grained
    7,   # 148 clusters - Extremely fine-grained
    8,   # 160 clusters - Extremely fine-grained
    9    # 172 clusters - Extremely fine-grained
]
# Create a figure with 3x4 subplots
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the resolutions and plot UMAPs
for i, res in enumerate(list_res_leiden_plot):
    # Get the corresponding peaks_norm object
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # transfer the annotations from the peaks_norm_ref object to the peaks_norm object
    peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    # QC metrics
    # peaks_norm.obs["n_cells_by_counts"] = peaks_norm_ref.obs["n_cells_by_counts"]
    # peaks_norm.obs["total_counts"] = peaks_norm_ref.obs["total_counts"]
    
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    
    # # Plot UMAP on the corresponding subplot
    # sc.pl.umap(peaks_norm, 
    #            ax=axes[i],
    #            show=False,
    #            title=f'Resolution {res}\n({n_clusters} clusters)',
    #            frameon=False)
    # Plot with corrected contrast scores
    sc.pl.umap(peaks_norm, 
               color='n_cells_by_counts',
               palette=cell_type_color_dict,
               ax=axes[i],
               show=False,
               title=f'Resolution {res}\n({n_clusters} clusters)',
               frameon=False, legend_loc=None)
    # Remove axis labels for cleaner look
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title
fig.suptitle('peak UMAPs by leiden resolution', y=1.02, fontsize=16)

plt.tight_layout()
plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_leiden/peaks_pb_leiden_clusters_umap_range_res_celltypes.png", dpi=600)
plt.show()

# %%
list_res_leiden_plot = [
    0.01,  # 6 clusters - Very broad clustering
    0.05,  # 11 clusters - Broad clustering
    0.2,   # 24 clusters - Starting to get more granular
    0.4,   # 32 clusters - Moderate clustering
    0.8,   # 43 clusters - Moderate clustering
    1,   # 51 clusters - More granular
    1.5,   # 61 clusters - More granular
    3,   # 87 clusters - Fine-grained
    5,   # 116 clusters - Very fine-grained
    7,   # 148 clusters - Extremely fine-grained
    8,   # 160 clusters - Extremely fine-grained
    9    # 172 clusters - Extremely fine-grained
]
# Create a figure with 3x4 subplots
fig, axes = plt.subplots(3, 4, figsize=(20, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the resolutions and plot UMAPs
for i, res in enumerate(list_res_leiden_plot):
    # Get the corresponding peaks_norm object
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # transfer the annotations from the peaks_norm_ref object to the peaks_norm object
    peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    # QC metrics
    peaks_norm.obs["n_cells_by_counts"] = peaks_norm_ref.obs["n_cells_by_counts"]
    peaks_norm.obs["total_counts"] = peaks_norm_ref.obs["total_counts"]
    
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    
    # # Plot UMAP on the corresponding subplot
    # sc.pl.umap(peaks_norm, 
    #            ax=axes[i],
    #            show=False,
    #            title=f'Resolution {res}\n({n_clusters} clusters)',
    #            frameon=False)
    # Plot with corrected contrast scores
    sc.pl.umap(peaks_norm, 
               color='n_cells_by_counts',
               cmap="magma",
            #    palette=cell_type_color_dict,
               ax=axes[i],
               show=False,
               title=f'Resolution {res}\n({n_clusters} clusters)',
               frameon=False, legend_loc=None)
    # Remove axis labels for cleaner look
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title
fig.suptitle('peak UMAPs by leiden resolution -n_cells_by_counts', y=1.02, fontsize=16)

plt.tight_layout()
plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_leiden/peaks_pb_leiden_clusters_umap_range_res_n_cells_by_counts.png", dpi=600)
plt.show()

# %%
rsc.pp.calculate_qc_metrics(peaks_norm, inplace=True)

# %%
peaks_norm.obs["total_counts"]

# %%
list_res_leiden_plot = [
    0.01,  # 6 clusters - Very broad clustering
    0.05,  # 11 clusters - Broad clustering
    0.2,   # 24 clusters - Starting to get more granular
    0.4,   # 32 clusters - Moderate clustering
    0.8,   # 43 clusters - Moderate clustering
    1,   # 51 clusters - More granular
    1.5,   # 61 clusters - More granular
    3,   # 87 clusters - Fine-grained
    5,   # 116 clusters - Very fine-grained
    7,   # 148 clusters - Extremely fine-grained
    8,   # 160 clusters - Extremely fine-grained
    9    # 172 clusters - Extremely fine-grained
]
# Create a figure with 3x4 subplots
fig, axes = plt.subplots(3, 4, figsize=(20, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the resolutions and plot UMAPs
for i, res in enumerate(list_res_leiden_plot):
    # Get the corresponding peaks_norm object
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # transfer the annotations from the peaks_norm_ref object to the peaks_norm object
    peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    # QC metrics
    peaks_norm.obs["n_cells_by_counts"] = peaks_norm_ref.obs["n_cells_by_counts"]
    peaks_norm.obs["total_counts"] = peaks_norm_ref.obs["total_counts"]
    
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    
    # # Plot UMAP on the corresponding subplot
    # sc.pl.umap(peaks_norm, 
    #            ax=axes[i],
    #            show=False,
    #            title=f'Resolution {res}\n({n_clusters} clusters)',
    #            frameon=False)
    # Plot with corrected contrast scores
    sc.pl.umap(peaks_norm, 
               color='total_counts',
               cmap="magma", vmax=1000,
            #    palette=cell_type_color_dict,
               ax=axes[i],
               show=False,
               title=f'Resolution {res}\n({n_clusters} clusters)',
               frameon=False, legend_loc=None)
    # Remove axis labels for cleaner look
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title
fig.suptitle('peak UMAPs by leiden resolution -total_counts', y=1.02, fontsize=16)

plt.tight_layout()
plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_leiden/peaks_pb_leiden_clusters_umap_range_res_total_counts.png", dpi=600)
plt.show()

# %%
# Create a figure with 3x4 subplots
fig, axes = plt.subplots(3, 4, figsize=(20, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the resolutions and plot UMAPs
for i, res in enumerate(list_res_leiden_plot):
    # Get the corresponding peaks_norm object
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # transfer the annotations from the peaks_norm_ref object to the peaks_norm object
    peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    
    # # Plot UMAP on the corresponding subplot
    # sc.pl.umap(peaks_norm, 
    #            ax=axes[i],
    #            show=False,
    #            title=f'Resolution {res}\n({n_clusters} clusters)',
    #            frameon=False)
    # Plot with corrected contrast scores
    sc.pl.umap(peaks_norm, 
               color='timepoint',
               palette=timepoint_colors,
               ax=axes[i],
               show=False,
               title=f'Resolution {res}\n({n_clusters} clusters)',
               frameon=False, legend_loc=None)
    # Remove axis labels for cleaner look
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title
fig.suptitle('peak UMAPs by leiden resolution', y=1.02, fontsize=16)

plt.tight_layout()
plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_leiden/peaks_pb_leiden_clusters_umap_range_res_timepoints.png", dpi=600)
plt.show()

# %%
# Create a figure with 2x4 subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the resolutions and plot UMAPs
for i, res in enumerate(list_res_leiden):
    # Get the corresponding peaks_norm object
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # transfer the annotations from the peaks_norm_ref object to the peaks_norm object
    peaks_norm.obs["celltype"] = peaks_norm_ref.obs["celltype"]
    peaks_norm.obs["timepoint"] = peaks_norm_ref.obs["timepoint"]
    
    # Get the number of unique clusters for the title
    n_clusters = len(adata_peaks.obs[f"leiden_{res}_merged"].unique())
    
    # # Plot UMAP on the corresponding subplot
    # sc.pl.umap(peaks_norm, 
    #            ax=axes[i],
    #            show=False,
    #            title=f'Resolution {res}\n({n_clusters} clusters)',
    #            frameon=False)
    # Plot with corrected contrast scores
    sc.pl.umap(peaks_norm, 
               color='timepoint',
               palette=timepoint_colors,
               ax=axes[i],
               show=False,
               title=f'Resolution {res}\n({n_clusters} clusters)',
               frameon=False, legend_loc=None)
    # Remove axis labels for cleaner look
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title
fig.suptitle('UMAP of Peak Accessibility by Leiden Resolution', y=1.02, fontsize=16)

plt.tight_layout()
plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_leiden/peaks_pb_leiden_clusters_umap_low_res_timepoints.png", dpi=600)
plt.show()

# %%
# load the newly computed leiden clusters
adata_test = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_leiden.h5ad")
adata_test

# %%
adata_peaks.obs["leiden_0.3_merged"] = adata_peaks.obs_names.map(adata_test.obs["leiden_0.3_merged"])
adata_peaks.obs["leiden_0.5_merged"] = adata_peaks.obs_names.map(adata_test.obs["leiden_0.5_merged"])
adata_peaks.obs["leiden_0.7_merged"] = adata_peaks.obs_names.map(adata_test.obs["leiden_0.7_merged"])

# %%
# Pseudobulk the adata object across {cluster, timepoint}
# First, the list of leiden clustering resolutions
list_res_leiden = [0.3, 0.5, 0.7]
pseudobulk_dict = {}  # Store adata_pseudo objects
peaks_norm_dict = {}  # Store peaks_pb_norm objects

for res in list_res_leiden:
    # Create pseudobulk object
    adata_pseudo = create_normalized_peak_pseudobulk(
        adata_peaks,
        cluster_key=f'leiden_{res}_merged',
        timepoint_key='dev_stage',
    )
    
    # Create transposed normalized object
    adata_pseudo.X = adata_pseudo.layers["normalized"]
    peaks_pb_norm = adata_pseudo.copy().T
    
    # Compute UMAP for peaks_pb_norm
    rsc.get.anndata_to_GPU(peaks_pb_norm)
    rsc.pp.scale(peaks_pb_norm)
    rsc.pp.pca(peaks_pb_norm, n_comps=100, use_highly_variable=False)
    rsc.pp.neighbors(peaks_pb_norm, n_neighbors=15, n_pcs=40)
    rsc.tl.umap(peaks_pb_norm, min_dist=0.3, random_state=42)
    sc.pl.umap(peaks_pb_norm)
    
    # Store both objects in dictionaries with resolution as key
    pseudobulk_dict[f'leiden_{res}_merged'] = adata_pseudo
    peaks_norm_dict[f'leiden_{res}_merged'] = peaks_pb_norm
    print(f"leiden_{res}_merged", " done")
# Now you can access any resolution's objects like this:
# pseudobulk_dict['leiden_0.5_merged']
# peaks_norm_dict['leiden_0.5_merged']

# %%
res = 10
adata_pseudo = create_normalized_peak_pseudobulk(
        adata_peaks,
        cluster_key=f'leiden_{res}_merged',
        timepoint_key='dev_stage',
)

# Create transposed normalized object
adata_pseudo.X = adata_pseudo.layers["normalized"]
peaks_pb_norm = adata_pseudo.copy().T

# Compute UMAP for peaks_pb_norm
rsc.get.anndata_to_GPU(peaks_pb_norm)
rsc.pp.scale(peaks_pb_norm)
rsc.pp.pca(peaks_pb_norm, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pb_norm, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pb_norm, min_dist=0.3, random_state=42)
sc.pl.umap(peaks_pb_norm)

# %%
rsc.tl.umap(peaks_pb_norm, min_dist=0.3, random_state=42)
sc.pl.umap(peaks_pb_norm)
# %%
# color the UMAP by the celltype and timepoints
peaks_pb_norm_ct_tp = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_wo_log_all_peaks.h5ad")

# copy over the obs from the peaks_pb_norm_ct_tp object
peaks_pb_norm.obs = peaks_pb_norm_ct_tp.obs.copy()

sc.pl.umap(peaks_pb_norm, color=["celltype", "timepoint"])
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

# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))



# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_norm, 
           color='celltype',
           # size=peaks_pb.obs['celltype_contrast']/np.max(peaks_pb.obs['celltype_contrast'])*20,
           palette=cell_type_color_dict,
           save='_celltype_allpeaks.png')

# %%
peaks_pb.obs["celltype"].value_counts()

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_norm, 
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
# ### plot the distribution of counts (pseudobulk groups)
#
# - summed counts
# - normalized counts
# - scale factors
# - log-normalized

