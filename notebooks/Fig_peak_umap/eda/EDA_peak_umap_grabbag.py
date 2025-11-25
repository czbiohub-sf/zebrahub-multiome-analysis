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
# ## EDA for the peak UMAP - a collection of exploratory workflows with half-baked ideas
#
# - projection of the chromatin accessibilities from peak clusters onto the cell UMAP
# - Hox gene cluster accessibility
# - Hematopoesis cell cluster

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## EDA 2: projection of the accessibility to the cell UMAP
#
# - for the peak groups in each leiden cluster (peak UMAP), show their accessibility profiles on the cell UMAP

# %%
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_master_filtered.h5ad")
adata_peaks

# %%
# # First get peaks for each leiden cluster
# leiden_peaks = {}
# for cluster in peaks_pb_hvp_50k.obs['leiden_0.7'].unique():
#     # Get peaks belonging to this cluster
#     peaks_in_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs['leiden_0.7']==cluster].obs_names.tolist()
#     leiden_peaks[cluster] = peaks_in_cluster

# # Now compute average accessibility for each peak group
# peak_group_scores = pd.DataFrame(index=adata_peaks.obs_names)
# for cluster, peaks in leiden_peaks.items():
#     # Get subset of peaks matrix for this cluster's peaks
#     peaks_subset = adata_peaks[:, peaks].X
    
#     # Compute mean accessibility across these peaks
#     mean_acc = np.array(peaks_subset.mean(axis=1)).flatten()
    
#     # Add to dataframe
#     peak_group_scores[f'Cluster_{cluster}_accessibility'] = mean_acc

# # Add scores to original adata object
# for col in peak_group_scores.columns:
#     adata_peaks.obs[col] = peak_group_scores[col]

# # Plot scores on UMAP
# sc.pl.umap(adata_peaks, color=peak_group_scores.columns, ncols=3)

import scipy.sparse as sp

# First get peaks for each leiden cluster
leiden_peaks = {}
for cluster in peaks_pb_hvp_50k.obs['leiden_0.7'].unique():
    peaks_in_cluster = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs['leiden_0.7']==cluster].obs_names.tolist()
    leiden_peaks[cluster] = peaks_in_cluster

# Now compute average accessibility for each peak group
peak_group_scores = pd.DataFrame(index=adata_peaks.obs_names)
for cluster, peaks in leiden_peaks.items():
    # Get subset of peaks matrix for this cluster's peaks
    peaks_subset = adata_peaks[:, peaks].X
    
    # If matrix is sparse, keep it sparse for the mean calculation
    if sp.issparse(peaks_subset):
        mean_acc = np.array(peaks_subset.mean(axis=1)).flatten()
    else:
        mean_acc = peaks_subset.mean(axis=1)
    
    # Add to dataframe
    peak_group_scores[f'Cluster_{cluster}_accessibility'] = mean_acc

# Add scores to original adata object
for col in peak_group_scores.columns:
    adata_peaks.obs[col] = peak_group_scores[col]

# Plot scores on UMAP
sc.pl.umap(adata_peaks, color=peak_group_scores.columns, ncols=3)

# %%

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## EDA3: Hox gene cluster (Hoxc)

# %%
peaks_pb_hvp_50k.obs["linked_gene"].str.contains("hox")

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["linked_gene"].str.contains("hox")]

# %%
# Get UMAP coordinates of your subset
subset_coords = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names.isin(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["linked_gene"].str.contains("hoxc")].obs_names)].obsm['X_umap']

# Plot base UMAP
sc.pl.umap(peaks_pb_hvp_50k, show=False)

# Add markers for subset
plt.scatter(subset_coords[:, 0], subset_coords[:, 1], 
           c='red', 
           # s=100, 
           marker='*')
plt.show()

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names.isin(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["linked_gene"].str.contains("hoxc")].obs_names)].obs

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names.isin(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["linked_gene"].str.contains("hoxc")].obs_names)].obs["celltype"]

# %%
peaks_hox = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names.isin(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["linked_gene"].str.contains("hoxc")].obs_names)]
peaks_hox

# %%
peaks_hox.obs[['linked_gene', 'accessibility_0somites', 'accessibility_5somites','accessibility_10somites', 
               'accessibility_15somites','accessibility_20somites', 'accessibility_30somites', 'timepoint']]

# %%
# First, let's parse the obs_names to get chromosome and position info
peaks_chr23 = peaks_hox[peaks_hox.obs_names.str.startswith('23-')]

# Parse start positions from obs_names
peaks_chr23.obs['start_pos'] = peaks_chr23.obs_names.str.split('-').str[1].astype(int)

# For each peak, find the somite stage with maximum accessibility
somite_cols = ['accessibility_0somites', 'accessibility_5somites', 'accessibility_10somites',
               'accessibility_15somites', 'accessibility_20somites', 'accessibility_30somites']

# Get the somite stage where accessibility peaks for each gene
peaks_chr23.obs['peak_somite'] = peaks_chr23.obs[somite_cols].idxmax(axis=1).str.replace('accessibility_', '').str.replace('somites', '')

# Create scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(peaks_chr23.obs['start_pos'], 
           peaks_chr23.obs['peak_somite'].astype(int),
           c=peaks_chr23.obs[somite_cols].max(axis=1),  # Color by max accessibility
           s=100)

# Add gene labels
for idx, row in peaks_chr23.obs.iterrows():
    plt.annotate(row['linked_gene'], 
                (row['start_pos'], int(row['peak_somite'])),
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('Chromosome 23 Position')
plt.ylabel('Somite Stage of Peak Accessibility')
plt.title('Hoxc Gene Accessibility Peaks vs Chromosomal Position')
plt.colorbar(label='Maximum Accessibility')
plt.grid(False)
plt.show()

# %%
# First, let's parse the obs_names to get chromosome and position info
peaks_chr23 = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names.str.startswith('23-')]

# Parse start positions from obs_names
peaks_chr23.obs['start_pos'] = peaks_chr23.obs_names.str.split('-').str[1].astype(int)


# %%
peaks_chr23_transposed = peaks_chr23.T
peaks_chr23_transposed

# %%
sc.pp.log1p(peaks_chr23_transposed, layer="normalized")
peaks_chr23_transposed

# %%
np.sum(np.expm1(peaks_chr23.X.todense()),1)

# %%
peaks_chr23.X = peaks_chr23_transposed.T.X.copy()

# %%
# peaks_chr23.X = peaks_chr23.layers["log_norm"].copy()

# %%
peaks_hoxc = peaks_chr23[peaks_chr23.obs["linked_gene"].str.startswith("hoxc")]
peaks_hoxc

# %%
# Define our tissues of interest
tissues_of_interest = ['hindbrain',
    'neural_posterior', 'spinal_cord',
    'NMPs', 'tail_bud', 
]

# Define color map
import matplotlib.cm as cm
viridis = cm.get_cmap('viridis', len(stages))
# # Define color map
# viridis = plt.colormaps.get_cmap('viridis')

# Create subplots dynamically based on the number of tissues
fig, axes = plt.subplots(len(tissues_of_interest), 1, figsize=(7, 5 * len(tissues_of_interest)))
if len(tissues_of_interest) == 1:
    axes = [axes]  # Ensure axes is iterable when there is only one tissue

# Iterate through each tissue type
for ax, celltype in zip(axes, tissues_of_interest):
    for stage_idx, stage in enumerate(stages):
        var_name = f"{celltype}_{stage}"
        if var_name in peaks_hoxc.var_names:
            col_idx = peaks_hoxc.var_names.get_loc(var_name)
            accessibility = peaks_hoxc.X[:, col_idx].toarray().flatten() if hasattr(peaks_hoxc.X, 'toarray') else peaks_hoxc.X[:, col_idx]
            
            # Sort points by chromosome position to connect them properly
            positions = peaks_hoxc.obs['start_pos'].to_numpy()
            sorted_indices = np.argsort(positions)
            sorted_pos = positions[sorted_indices]
            sorted_acc = accessibility[sorted_indices]
            
            # Plot line connecting points
            ax.plot(sorted_pos, sorted_acc, '-', color=viridis(stage_idx/len(stages)), alpha=0.3)
            # Plot points on top
            ax.scatter(sorted_pos, sorted_acc,
                       c=[viridis(stage_idx/len(stages))],
                       s=100,
                       label=f'{stage}')
    
    # Add gene labels in each subplot
    for idx, row in peaks_hoxc.obs.iterrows():
        ax.annotate(row['linked_gene'], 
                    (row['start_pos'], min(accessibility)),
                    xytext=(0, -20), textcoords='offset points',
                    rotation=45, ha='right')

    ax.set_xlabel('chr 23 (bp)')
    ax.set_ylabel('chromatin accessibility')
    ax.set_title(f'Hoxc accessibility in {celltype}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(False)
    ax.set_ylim(-1, 13)  # Set y-axis range

plt.savefig(figpath + "hoxc_chr_access_dynamics_celltypes.pdf")
plt.tight_layout()
plt.show()

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## EDA4: Hematopoesis
#
# - the hematopoetic system (hemangioblast, hematopoetic vasculature) regulatory programs are clustered together strongly based on their "celltype", less than their "timepoint". However, there's a clear pattern of "temporality" in the data - that the hemangioblast and hematopoetic vasculature shares the early timepoints, then branches out from there...
#
# - This might be a good place to zoom-in and do a deeper dive into the biology. 
