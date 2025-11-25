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

# %%

# %% [markdown]
# ## generate peak UMAP plots colored by different metadata

# %%
# import the libraries
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import entropy

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# %%
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad")
adata_peaks

# %%
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


# %%
# import the peaks_norm object with annotations
peaks_norm_ref = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_wo_log_all_peaks.h5ad")
peaks_norm_ref

# %%
# moves the .X to GPU
rsc.get.anndata_to_GPU(peaks_norm_ref)

# %%
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
# import the peak UMAP of our choice (leiden resolution of 0.8)
peaks_norm = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.8_merged.h5ad")
peaks_norm

# %%
# # copy over the metadata
peaks_norm.obs = peaks_norm_ref.obs.copy()

# %%
peaks_norm

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_norm, 
           color='celltype',
           size=peaks_norm.obs['celltype_contrast']/np.max(peaks_norm.obs['celltype_contrast'])*10,
           palette=cell_type_color_dict,
           save='_pb_leiden_0.8_celltype_allpeaks.png')

# %%
sc.pl.umap(peaks_norm, 
           color='timepoint',
           size=peaks_norm.obs['timepoint_contrast']/100,
           save="_pb_leiden_0.8_timepoint_allpeaks.png")

# %%
np.max(peaks_norm.obs['timepoint_contrast'])

# %%

# %% [markdown]
# ## For each leiden clustering resolution, compute the cell type purity for each cluster, then plot the distribution
#
# - For each leiden clustering in cell UMAP (WNN), we can compute the leiden clustering of peak UMAP, then ask the celltype purity for each cluster.
# - The mean/std of this distribution is a function of leiden clustering resolution of cell UMAP (leiden_res_cell). We will make a plot of "leiden_res_cell" vs "purity_celltype".

# %%
adata = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_leiden_filtered.h5ad")
adata

# %%
# check the number of leiden clusters for each resolution
for res in list_res_leiden:
    print(f"the number of leiden clusters for {res} is ", len(adata.obs[f"leiden_{res}_merged"].unique()))

# %%
## RESUME HERE
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/"

# import the peaks_norm objects computed from the leiden clustering resolutions
list_res_leiden = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
peaks_norm_dict = {}  # Store peaks_pb_norm objects

for res in list_res_leiden:
    # import the adata
    peaks_norm = sc.read_h5ad(filepath + f"peaks_by_pb_leiden_{res}_merged.h5ad")
    # move the .X to GPU
    rsc.get.anndata_to_GPU(peaks_norm)
    peaks_norm_dict[f'leiden_{res}_merged'] = peaks_norm
    print(f'leiden_{res}_merged', " done")

# %%
peaks_norm_ref


# %%
def find_resolution_for_target_clusters(adata, target_n_clusters, min_res=0.1, max_res=1, 
                                       max_iterations=15, tolerance=2):
    """Find the Leiden resolution that gives approximately the target number of clusters."""
    current_min = min_res
    current_max = max_res
    
    for i in range(max_iterations):
        current_res = (current_min + current_max) / 2
        rsc.tl.leiden(adata, resolution=current_res, random_state=42, key_added='temp_leiden')
        n_clusters = len(adata.obs['temp_leiden'].unique())
        
        if abs(n_clusters - target_n_clusters) <= tolerance:
            return current_res
        elif n_clusters < target_n_clusters:
            current_min = current_res
        else:
            current_max = current_res
    
    # Return the best approximation we found
    return current_res


# %%
from scipy.stats import entropy

# Function to calculate cell type purity for each peak UMAP cluster
def calculate_cluster_purity(adata, cluster_key, celltype_key):
    """Calculate purity metrics for each cluster."""
    results = []
    
    for cluster in adata.obs[cluster_key].unique():
        # Subset to this cluster
        mask = adata.obs[cluster_key] == cluster
        cluster_cells = adata.obs.loc[mask]
        
        if len(cluster_cells) == 0:
            continue
            
        # Count cell types in this cluster
        celltype_counts = cluster_cells[celltype_key].value_counts()
        total_cells = celltype_counts.sum()
        
        # Calculate proportions
        proportions = celltype_counts / total_cells
        
        # Calculate entropy (lower means more pure)
        cluster_entropy = entropy(proportions)
        
        # Calculate purity (proportion of most common cell type)
        purity = proportions.max() if len(proportions) > 0 else 0
        
        results.append({
            'cluster': cluster,
            'purity': purity,
            'entropy': cluster_entropy,
            'num_cells': total_cells
        })
    
    return pd.DataFrame(results)


# %%
# (1) compute the leiden clustering for each peak UMAP object.
# (2) then, compute the purity of celltypes per each cluster (for each object)

# Initialize storage for all resolution results
all_resolution_metrics = {}

# Process each resolution
for res in list_res_leiden:
    # Get the peaks object for this resolution
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    # copy over the metadata
    peaks_norm.obs = peaks_norm_ref.obs.copy()
    
    # compute the peak leiden clusters (resolution can vary)
    peak_clust_res = 0.5
    rsc.tl.leiden(peaks_norm, resolution=peak_clust_res, 
                 random_state=42, key_added=f'peaks_leiden_{res}')
    sc.pl.umap(peaks_norm, color=f"peaks_leiden_{res}", title = f"leiden_{res}_merged",
               save=f"_peaks_cell_clust_leiden_{res}_merged_colored_peak_clusts_0.5.png")
    
    # Calculate cell type purity for each peak cluster
    # Assuming the cell type annotations are in 'annotation_ML_coarse'
    metrics = calculate_cluster_purity(peaks_norm, 
                                      cluster_key=f'peaks_leiden_{res}', 
                                      celltype_key='celltype')
    
    # Save metrics for this resolution
    all_resolution_metrics[res] = metrics
    
    # Print summary statistics
    print(f"Resolution {res}: Mean purity = {metrics['purity'].mean():.3f} ± {metrics['purity'].std():.3f}")
    print(f"Resolution {res}: Mean entropy = {metrics['entropy'].mean():.3f} ± {metrics['entropy'].std():.3f}")

# Create a summary dataframeb
summary_metrics = []
for res, metrics in all_resolution_metrics.items():
    summary_metrics.append({
        'resolution': res,
        'mean_purity': metrics['purity'].mean(),
        'std_purity': metrics['purity'].std(),
        'mean_entropy': metrics['entropy'].mean(),
        'std_entropy': metrics['entropy'].std(),
        'num_clusters': len(metrics)
    })

summary_df = pd.DataFrame(summary_metrics)

# %%
summary_df.head()

# %%
summary_df

# %%
resolution_clusters = {
    'resolution': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
    'num_clusters': [6, 9, 11, 16, 24, 27, 32, 38, 43, 51, 61, 70, 87, 104, 116, 133, 148, 160, 172]
}
plt.scatter(resolution_clusters["num_clusters"], summary_df["mean_entropy"])
plt.xlabel("number of leiden clusters (cell UMAP)")
plt.ylabel("mean entropy")
plt.grid(False)
plt.show()

# %% [markdown]
# ## REPEAT: standardize the peak cluster counts across peak UMAP objects
# - since the more peak leiden clusters will bias the results (it'll give purer clusters), we'll standardize the number of clusters around some number.
#

# %%
# In your main loop:
target_clusters = 25  # Choose a fixed number of clusters for all resolutions

for res in list_res_leiden:
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    peaks_norm.obs = peaks_norm_ref.obs.copy()
    
    # Find the resolution that gives approximately target_clusters
    peak_clust_res = find_resolution_for_target_clusters(peaks_norm, target_clusters)
    
    print(f"Cell resolution {res}: Using peak resolution {peak_clust_res:.4f} to get ~{target_clusters} clusters")
    
    # The rest of your code remains the same...
    rsc.tl.leiden(peaks_norm, resolution=peak_clust_res, 
                 random_state=42, key_added=f'peaks_leiden_{peak_clust_res}')
    sc.pl.umap(peaks_norm, color=f"peaks_leiden_{peak_clust_res}", title = f"leiden_{res}_merged",
               save=f"_peaks_cell_clust_leiden_{res}_merged_colored_peak_clusts_{peak_clust_res}.png")
    
    # Calculate cell type purity for each peak cluster
    # Assuming the cell type annotations are in 'annotation_ML_coarse'
    metrics = calculate_cluster_purity(peaks_norm, 
                                      cluster_key=f'peaks_leiden_{peak_clust_res}', 
                                      celltype_key='celltype')
    
    # Save metrics for this resolution
    all_resolution_metrics[res] = metrics
    
    # Print summary statistics
    print(f"Resolution {res}: Mean purity = {metrics['purity'].mean():.3f} ± {metrics['purity'].std():.3f}")
    print(f"Resolution {res}: Mean entropy = {metrics['entropy'].mean():.3f} ± {metrics['entropy'].std():.3f}")

# Create a summary dataframeb
summary_metrics = []
for res, metrics in all_resolution_metrics.items():
    summary_metrics.append({
        'resolution': res,
        'mean_purity': metrics['purity'].mean(),
        'std_purity': metrics['purity'].std(),
        'mean_entropy': metrics['entropy'].mean(),
        'std_entropy': metrics['entropy'].std(),
        'num_clusters': len(metrics)
    })

summary_df = pd.DataFrame(summary_metrics)

# %%
summary_df

# %%
all_resolution_metrics[0.01].purity.median()

# %%
# Create a summary dataframe with additional statistics
summary_metrics = []
for res, metrics in all_resolution_metrics.items():
    # Calculate number of clusters
    num_clusters = len(metrics)
    
    # Calculate mean, median, std, and SEM for purity
    mean_purity = metrics['purity'].mean()
    median_purity = metrics['purity'].median()
    std_purity = metrics['purity'].std()
    sem_purity = std_purity / np.sqrt(num_clusters)
    
    # Calculate mean, median, std, and SEM for entropy
    mean_entropy = metrics['entropy'].mean()
    median_entropy = metrics['entropy'].median()
    std_entropy = metrics['entropy'].std()
    sem_entropy = std_entropy / np.sqrt(num_clusters)
    
    summary_metrics.append({
        'resolution': res,
        'mean_purity': mean_purity,
        'median_purity': median_purity,
        'std_purity': std_purity,
        'sem_purity': sem_purity,
        'mean_entropy': mean_entropy,
        'median_entropy': median_entropy,
        'std_entropy': std_entropy,
        'sem_entropy': sem_entropy,
        'num_clusters': num_clusters
    })

summary_df = pd.DataFrame(summary_metrics)
summary_df

# %%
summary_df_celltype = summary_df

# %%
plt.errorbar(resolution_clusters["num_clusters"], 
             summary_df["median_purity"],
             summary_df["sem_purity"], fmt="o",)
plt.xlabel("number of clusters in cell UMAP")
plt.ylabel("median purity")
plt.grid(False)
plt.tight_layout()
plt.savefig(figpath + "scatter_num_peak_clusters_vs_median_purity_celltypes.pdf")
plt.show()

# %%
plt.errorbar(resolution_clusters["num_clusters"], 
             summary_df["median_entropy"],
             summary_df["sem_entropy"], fmt="o",)
plt.xlabel("number of clusters in cell UMAP")
plt.ylabel("median entropy")
plt.grid(False)
plt.tight_layout()
plt.savefig(figpath + "scatter_num_peak_clusters_vs_median_entropy_celltypes.pdf")
plt.show()

# %% [markdown]
# res = 0.4 seemed tobe the optimal resolution

# %%

# %%
peaks_norm_ref

# %%
peaks_norm_ref.obs["celltype"].unique()

# %%
# theoretical limit of entropy (the maximum)
n_celltypes = 33
n_celltypes*(-1/n_celltypes*np.log(1/n_celltypes))

# %%
# Create dictionary to store the peak resolutions you already calculated
peak_resolutions = {
    0.01: 0.21250000000000002,
    0.03: 0.4375,
    0.05: 0.4375,
    0.1: 0.325,
    0.2: 0.55,
    0.3: 0.55,
    0.4: 0.55,
    0.5: 0.55,
    0.8: 0.4375,
    1: 0.325,
    1.5: 0.55,
    2: 0.55,
    3: 0.325,
    4: 0.325,
    5: 0.55,
    6: 0.22128906250000002,
    7: 0.26875000000000004,
    8: 0.26875000000000004,
    9: 0.24062500000000003
}

# Create dictionary to store timepoint metrics
all_resolution_metrics_timepoint = {}

# Iterate through resolutions
for res in list_res_leiden:
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    peaks_norm.obs = peaks_norm_ref.obs.copy()
    
    # Get the already calculated peak resolution
    peak_clust_res = peak_resolutions[res]
    
    # Use the existing leiden clusters - no need to recalculate
    cluster_key = f'peaks_leiden_{peak_clust_res}'
    
    # If you need to check that the clusters exist
    if cluster_key not in peaks_norm.obs.columns:
        print(f"Warning: {cluster_key} not found, recalculating...")
        rsc.tl.leiden(peaks_norm, resolution=peak_clust_res, 
                     random_state=42, key_added=cluster_key)
    
    # Calculate timepoint purity for each peak cluster
    metrics_timepoint = calculate_cluster_purity(peaks_norm, 
                                    cluster_key=cluster_key, 
                                    celltype_key='timepoint')
    
    # Save metrics for this resolution
    all_resolution_metrics_timepoint[res] = metrics_timepoint
    
    # Print summary statistics for timepoints
    print(f"Resolution {res} (Timepoints): Mean purity = {metrics_timepoint['purity'].mean():.3f} ± {metrics_timepoint['purity'].std():.3f}")
    print(f"Resolution {res} (Timepoints): Mean entropy = {metrics_timepoint['entropy'].mean():.3f} ± {metrics_timepoint['entropy'].std():.3f}")

# Create summary dataframe for timepoints
summary_metrics_timepoint = []
for res, metrics in all_resolution_metrics_timepoint.items():
    summary_metrics_timepoint.append({
        'resolution': res,
        'mean_purity': metrics['purity'].mean(),
        'median_purity': metrics['purity'].median(),
        'std_purity': metrics['purity'].std(),
        'sem_purity': metrics['purity'].std() / np.sqrt(len(metrics)),
        'mean_entropy': metrics['entropy'].mean(),
        'median_entropy': metrics['entropy'].median(),
        'std_entropy': metrics['entropy'].std(),
        'sem_entropy': metrics['entropy'].std() / np.sqrt(len(metrics)),
        'num_clusters': len(metrics)
    })

summary_df_timepoint = pd.DataFrame(summary_metrics_timepoint)

# %%
all_resolution_metrics_timepoint[0.01].purity.hist(alpha=0.7)
all_resolution_metrics_timepoint[0.1].purity.hist(alpha=0.7)
all_resolution_metrics_timepoint[0.4].purity.hist(alpha=0.7)
all_resolution_metrics_timepoint[4].purity.hist(alpha=0.7)
plt.legend([0.01, 0.1, 0.4, 4])
plt.grid(False)
plt.show()


# %%
all_resolution_metrics_timepoint[0.01].purity.median()

# %%
plt.errorbar(resolution_clusters["num_clusters"], 
             summary_df_timepoint["median_purity"],
             summary_df_timepoint["sem_purity"], fmt="o",)
plt.xlabel("number of clusters in cell UMAP")
plt.ylabel("median purity")
plt.grid(False)
plt.tight_layout()
plt.savefig(figpath + "scatter_num_peak_clusters_vs_median_purity_timepoints.pdf")
plt.show()

# %%
plt.errorbar(resolution_clusters["num_clusters"], 
             summary_df_timepoint["median_entropy"],
             summary_df_timepoint["sem_entropy"], fmt="o",)
plt.xlabel("number of clusters in cell UMAP")
plt.ylabel("median entropy")
plt.grid(False)
plt.tight_layout()
plt.savefig(figpath + "scatter_num_peak_clusters_vs_median_entropy_timepoints.pdf")
plt.show()


# %%
def calculate_cluster_purity_combined(adata, cluster_key, celltype_key, timepoint_key):
    """Calculate purity metrics for each cluster using combined cell type and timepoint."""
    
    # Create the combined category - convert categoricals to string first
    if 'combined_category' not in adata.obs.columns:
        # Convert categorical columns to string before concatenation
        celltype_str = adata.obs[celltype_key].astype(str)
        timepoint_str = adata.obs[timepoint_key].astype(str)
        adata.obs['combined_category'] = celltype_str + '_' + timepoint_str
    
    results = []
    
    for cluster in adata.obs[cluster_key].unique():
        # Subset to this cluster
        mask = adata.obs[cluster_key] == cluster
        cluster_cells = adata.obs.loc[mask]
        
        if len(cluster_cells) == 0:
            continue
            
        # Count combined categories in this cluster
        category_counts = cluster_cells['combined_category'].value_counts()
        total_cells = category_counts.sum()
        
        # Calculate proportions
        proportions = category_counts / total_cells
        
        # Calculate entropy (lower means more pure)
        cluster_entropy = entropy(proportions)
        
        # Calculate purity (proportion of most common category)
        purity = proportions.max() if len(proportions) > 0 else 0
        
        results.append({
            'cluster': cluster,
            'purity': purity,
            'entropy': cluster_entropy,
            'num_cells': total_cells
        })
    
    return pd.DataFrame(results)


# %%
# Create dictionary to store combined metrics
all_resolution_metrics_combined = {}

# Iterate through resolutions
for res in list_res_leiden:
    peaks_norm = peaks_norm_dict[f'leiden_{res}_merged']
    peaks_norm.obs = peaks_norm_ref.obs.copy()
    
    # Get the already calculated peak resolution
    peak_clust_res = peak_resolutions[res]
    
    # Use the existing leiden clusters
    cluster_key = f'peaks_leiden_{peak_clust_res}'
    
    # If you need to check that the clusters exist
    if cluster_key not in peaks_norm.obs.columns:
        print(f"Warning: {cluster_key} not found, recalculating...")
        rsc.tl.leiden(peaks_norm, resolution=peak_clust_res, 
                     random_state=42, key_added=cluster_key)
    
    # Calculate combined category purity for each peak cluster
    metrics_combined = calculate_cluster_purity_combined(
        peaks_norm, 
        cluster_key=cluster_key, 
        celltype_key='celltype',  # Your cell type column
        timepoint_key='timepoint'  # Your timepoint column
    )
    
    # Save metrics for this resolution
    all_resolution_metrics_combined[res] = metrics_combined
    
    # Print summary statistics
    print(f"Resolution {res} (Combined): Mean purity = {metrics_combined['purity'].mean():.3f} ± {metrics_combined['purity'].std():.3f}")
    print(f"Resolution {res} (Combined): Mean entropy = {metrics_combined['entropy'].mean():.3f} ± {metrics_combined['entropy'].std():.3f}")

# Create summary dataframe for combined metrics
summary_metrics_combined = []
for res, metrics in all_resolution_metrics_combined.items():
    summary_metrics_combined.append({
        'resolution': res,
        'mean_purity': metrics['purity'].mean(),
        'median_purity': metrics['purity'].median(),
        'std_purity': metrics['purity'].std(),
        'sem_purity': metrics['purity'].std() / np.sqrt(len(metrics)),
        'mean_entropy': metrics['entropy'].mean(),
        'median_entropy': metrics['entropy'].median(),
        'std_entropy': metrics['entropy'].std(),
        'sem_entropy': metrics['entropy'].std() / np.sqrt(len(metrics)),
        'num_clusters': len(metrics)
    })

summary_df_combined = pd.DataFrame(summary_metrics_combined)

# %%
plt.errorbar(resolution_clusters["num_clusters"], 
             summary_df_combined["median_purity"],
             summary_df_combined["sem_purity"], fmt="o",)
plt.xlabel("number of clusters in cell UMAP")
plt.ylabel("median purity")
plt.grid(False)
plt.tight_layout()
plt.savefig(figpath + "scatter_num_peak_clusters_vs_median_purity_celltype_timepoints.pdf")
plt.show()

# %%
plt.errorbar(resolution_clusters["num_clusters"], 
             summary_df_combined["median_entropy"],
             summary_df_combined["sem_entropy"], fmt="o",)
plt.xlabel("number of clusters in cell UMAP")
plt.ylabel("median_entropy")
plt.grid(False)
plt.tight_layout()
plt.savefig(figpath + "scatter_num_peak_clusters_vs_median_entropy_celltype_timepoints.pdf")
plt.show()

# %%

# %% [markdown]
# ## Optimal leiden resolution == 0.4 (cell UMAP) to compute the pseudobulked peak UMAP

# %%
# Get the peaks object for this resolution - 0.4 was the optimal resolution
peaks_norm = peaks_norm_dict[f'leiden_0.4_merged']
# # copy over the metadata
peaks_norm.obs = peaks_norm_ref.obs.copy()

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_norm, 
           color='celltype',
           # size=peaks_norm.obs['celltype_contrast']/np.max(peaks_norm.obs['celltype_contrast'])*10,
           palette=cell_type_color_dict,
           save='_pb_leiden_0.4_celltype_allpeaks.png')

# %%
sc.pl.umap(peaks_norm, 
           color='timepoint',
           # size=peaks_norm.obs['timepoint_contrast']/100,
           palette=timepoint_colors,
           save="_pb_leiden_0.4_timepoint_allpeaks.png")

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_norm, 
           color='celltype',
           size=peaks_norm.obs['celltype_contrast']/np.max(peaks_norm.obs['celltype_contrast'])*10,
           palette=cell_type_color_dict,
           save='_pb_leiden_0.4_celltype_allpeaks.png')

# %%
# Get the peaks object for this resolution - 0.4 was the optimal resolution
peaks_norm = peaks_norm_dict[f'leiden_0.4_merged']
# # copy over the metadata
peaks_norm.obs = peaks_norm_ref.obs.copy()
peaks_norm

# %%
peaks_norm

# %%
peaks_norm.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated.h5ad")

# %%
# # Convert all .obs categorical columns to strings
# for col in peaks_norm.obs.columns:
#     if peaks_norm.obs[col].dtype.name == 'category':
#         peaks_norm.obs[col] = peaks_norm.obs[col].astype(str)

# # Convert all .var categorical columns to strings
# for col in peaks_norm.var.columns:
#     if peaks_norm.var[col].dtype.name == 'category':
#         peaks_norm.var[col] = peaks_norm.var[col].astype(str)

# del peaks_norm.raw
        
# peaks_norm.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_no_cat.h5ad")

