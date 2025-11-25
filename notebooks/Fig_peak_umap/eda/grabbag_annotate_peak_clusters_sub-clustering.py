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

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Aggreage over "celltype" or "timepoint"

# %%
# Create dictionaries to map columns to celltype and timepoint
celltype_mapping = {}
timepoint_mapping = {}

# Parse var names
for col in peaks_pb_hvp_50k.var.index:
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
    peaks_pb_hvp_50k.obs[f'accessibility_{celltype}'] = peaks_pb_hvp_50k.X[:, [peaks_pb_hvp_50k.var.index.get_loc(col) for col in celltype_cols]].sum(axis=1)

for timepoint in unique_timepoints:
    # Get columns for this timepoint
    timepoint_cols = [col for col, tp in timepoint_mapping.items() if tp == timepoint]
    # Sum accessibility across all celltypes for this timepoint
    peaks_pb_hvp_50k.obs[f'accessibility_{timepoint}'] = peaks_pb_hvp_50k.X[:, [peaks_pb_hvp_50k.var.index.get_loc(col) for col in timepoint_cols]].sum(axis=1)


# %%
# For timepoints
timepoint_cols = ['accessibility_0somites', 'accessibility_5somites', 'accessibility_10somites', 
                 'accessibility_15somites', 'accessibility_20somites', 'accessibility_30somites']
timepoint_vals = np.array([peaks_pb_hvp_50k.obs[col] for col in timepoint_cols]).T

# Find max timepoint for each peak
max_timepoint_idx = np.argmax(timepoint_vals, axis=1)
timepoint_names = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
peaks_pb_hvp_50k.obs['timepoint'] = [timepoint_names[i] for i in max_timepoint_idx]

# Calculate corrected timepoint contrast
max_vals = np.max(timepoint_vals, axis=1)
# Calculate mean and std excluding max value for each peak
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
peaks_pb_hvp_50k.obs['timepoint_contrast'] = (max_vals - other_vals_mean) / other_vals_std

# For celltypes
celltype_cols = [col for col in peaks_pb_hvp_50k.obs.columns 
                if col.startswith('accessibility_') and 'somites' not in col]
celltype_vals = np.array([peaks_pb_hvp_50k.obs[col] for col in celltype_cols]).T

# Find max celltype for each peak
max_celltype_idx = np.argmax(celltype_vals, axis=1)
celltype_names = [col.replace('accessibility_', '') for col in celltype_cols]
peaks_pb_hvp_50k.obs['celltype'] = [celltype_names[i] for i in max_celltype_idx]

# Calculate corrected celltype contrast
max_vals = np.max(celltype_vals, axis=1)
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(celltype_vals, max_celltype_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(celltype_vals, max_celltype_idx)])
peaks_pb_hvp_50k.obs['celltype_contrast'] = (max_vals - other_vals_mean) / other_vals_std

# %%
peaks_pb_hvp_50k

# %%
# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))
peaks_pb_hvp_50k.uns['timepoint_colors'] = [timepoint_colors[t] for t in timepoint_order]

# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k, 
           color='timepoint',
           size=peaks_pb_hvp_50k.obs['timepoint_contrast'],
           palette=timepoint_colors,
           save='_timepoint_scaled_viridis.png')


# %%
# Normalize peak contrast for alpha values
def normalize_for_alpha_robust(values, min_alpha=0.1, max_alpha=0.9):
    min_val = np.percentile(values, 5)
    max_val = np.percentile(values, 95)
    clipped = np.clip(values, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    return normalized * (max_alpha - min_alpha) + min_alpha

# Create alpha values from peak contrast
peaks_pb_hvp_50k.obs['alpha_timepoint'] = normalize_for_alpha_robust(peaks_pb_hvp_50k.obs['timepoint_contrast'])
peaks_pb_hvp_50k.obs['alpha_celltype'] = normalize_for_alpha_robust(peaks_pb_hvp_50k.obs['celltype_contrast'])


# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k, 
           color='timepoint',
           alpha=peaks_pb_hvp_50k.obs['alpha_timepoint'],
           palette=timepoint_colors,
           save='_timepoint_alpha_scaled_viridis.png')

# %%
# Use "Set3" color palette
sc.pl.umap(peaks_pb_hvp_50k, 
           color='timepoint',
           size=peaks_pb_hvp_50k.obs['timepoint_contrast'],
           palette="Set3",
           save='_timepoint_scaled.png')

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
sc.pl.umap(peaks_pb_hvp_50k, 
           color='celltype',
           size=peaks_pb_hvp_50k.obs['celltype_contrast'],
           palette=cell_type_color_dict,
           save='_celltype_scaled.png')

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k, 
           color='celltype',
           alpha=peaks_pb_hvp_50k.obs['alpha_celltype'],
           palette=cell_type_color_dict,
           save='_celltype_alpha_scaled.png')

# %%
# Create a shuffled copy of the indices
shuffled_indices = np.random.permutation(peaks_pb_hvp_50k.obs_names)

# Reorder the AnnData object
peaks_pb_hvp_50k_shuffled = peaks_pb_hvp_50k[shuffled_indices]

# Plot with shuffled order
sc.pl.umap(peaks_pb_hvp_50k_shuffled, 
           color='celltype',
           save='_celltype_shuffled.png')

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k_shuffled, 
           color='timepoint',
           alpha=peaks_pb_hvp_50k_shuffled.obs['alpha_timepoint'],
           palette=timepoint_colors,
           save='_timepoint_alpha_scaled_viridis_shuffled.png')

# %%
# Plot with corrected contrast scores
sc.pl.umap(peaks_pb_hvp_50k_shuffled, 
           color='celltype',
           alpha=peaks_pb_hvp_50k_shuffled.obs['alpha_celltype'],
           palette=cell_type_color_dict,
           save='_celltype_alpha_scaled_shuffled.png')

# %%
peaks_pb_hvp_50k.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k.h5ad")

# %%

# %%

# %% [markdown]
# # ------------------ HOLD ----------------------

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## EDA1-2. Sub-clustering for specific clusters

# %%
# make sure that we have log-normalized counts in the adata.X
peaks_pb_hvp_50k.X = peaks_pb_hvp_50k.layers["log_norm"].copy()

# %%
# scale the counts
sc.pp.scale(peaks_pb_hvp_50k)

# %%
# Subset the data to only cluster 0
adata_sub = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"] == "0"].copy()

# Recompute PCA & neighbors on the subset
sc.pp.pca(adata_sub)
sc.pp.neighbors(adata_sub)
sc.tl.umap(adata_sub, random_state=42, min_dist=0.1)

# Run Leiden clustering with a new resolution
sc.tl.leiden(adata_sub, resolution=0.4)  # Increase resolution for subclustering

# Plot the new UMAP with subclusters
sc.pl.umap(adata_sub, color="leiden", frameon=False)

# %%
# Plot the new UMAP with subclusters
# sc.pl.umap(adata_sub, color="leiden",legend_loc="on data", save="_peaks_clust0_leiden.png")
sc.pl.umap(adata_sub, color="leiden",legend_loc="on data")

# %%
sc.pl.umap(adata_sub, color="leiden", save="_peaks_clust0_leiden.pdf")

# %%
sc.pl.umap(adata_sub, color="leiden", save="_peaks_clust0_leiden.png")

# %%
# Extract numeric values from categorical labels (remove "somites" and convert to int)
adata_sub.obs["timepoint_numeric"] = (
    adata_sub.obs["timepoint"].str.replace("somites", "").astype(float)
)

# Plot UMAP using a continuous colormap
sc.pl.umap(
    adata_sub, 
    color="timepoint_numeric",  # Use the new numeric column
    color_map="viridis", alpha=0.7, # Apply viridis colormap
    save="_peaks_clust0_timepoint_viridis.png"
)

# %%
# map to the global UMAP coordinates
peaks_sub = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"]=="0"]

peaks_sub.obs["leiden"] = peaks_sub.obs_names.map(adata_sub.obs["leiden"])
sc.pl.umap(peaks_sub, color="leiden", save="_peaks_clust0_leiden_global_coords.png")

# %%
# # check the genes associated with the peaks
# # clust_names = ["1", "4", "5", "7", "11"]
# clust_names = peaks_sub.obs.leiden.unique()

# for clust in clust_names:
#     print(f"cluster {clust} has genes:")
#     list_genes_sub = peaks_sub[peaks_sub.obs.leiden==clust].obs["associated_gene"].unique().to_list()
#     print(list_genes_sub)
    

# %%
# save the adata_sub
adata_sub.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/peak_cluster_tailbud_late.h5ad")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## EDA 1-2. sub-clustering for the optic_cup (early)
#
# - sub-clustering another "wing" structure from the large peak UMAP

# %%
sc.pl.umap(peaks_pb_hvp_50k, color="leiden", legend_loc="on data")

# %%
sc.pl.umap(peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden"]=="1"], color="leiden", legend_loc="on data")

# %%
# Subset the data to only cluster 1
# adata_sub2 = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"].isin(["1", "6"])]
adata_sub2 = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden"]=="1"]
# print(adata_sub2)

# Recompute PCA & neighbors on the subset
sc.pp.pca(adata_sub2)
sc.pp.neighbors(adata_sub2)
sc.tl.umap(adata_sub2, random_state=42)#, min_dist=0.1)

# Run Leiden clustering with a new resolution
sc.tl.leiden(adata_sub2, resolution=0.4)  # Increase resolution for subclustering

# Plot the new UMAP with subclusters
sc.pl.umap(adata_sub2, color="leiden", legend_loc="on data")

# %%
sc.pp.scale(peaks_pb_hvp_50k)

# %%
# Subset the data to only cluster 1
# adata_sub2 = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"].isin(["1", "6"])]
adata_sub2 = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden"]=="1"]
# print(adata_sub2)

# Recompute PCA & neighbors on the subset
sc.pp.pca(adata_sub2)
sc.pp.neighbors(adata_sub2)
sc.tl.umap(adata_sub2, min_dist=0.1)

# Run Leiden clustering with a new resolution
sc.tl.leiden(adata_sub2, resolution=0.4)  # Increase resolution for subclustering

# Plot the new UMAP with subclusters
sc.pl.umap(adata_sub2, color="leiden", legend_loc="on data")

# %%
# Plot the new UMAP with subclusters
sc.pl.umap(adata_sub2, color="leiden", save="_peaks_clust_1_scaled_counts_leiden.png")

# %%
# # Plot the new UMAP with subclusters
# sc.pl.umap(adata_sub2, color="leiden", save="_peaks_clust_1_leiden.png")

# %%
# # Plot the new UMAP with subclusters
# sc.pl.umap(adata_sub2, color="leiden", save="_peaks_clust_1_6_leiden.png")
# # sc.pl.umap(adata_sub2, color="leiden",legend_loc="on data")

# %%
# sc.pl.umap(adata_sub2, color="leiden", save="_peaks_clust_1_6_leiden.pdf")

# %%
# # Extract numeric values from categorical labels (remove "somites" and convert to int)
# adata_sub2.obs["timepoint_numeric"] = (
#     adata_sub2.obs["timepoint"].str.replace("somites", "").astype(float)
# )

# # Plot UMAP using a continuous colormap
# sc.pl.umap(
#     adata_sub2, 
#     color="timepoint_numeric",  # Use the new numeric column
#     color_map="viridis", alpha=0.7, # Apply viridis colormap
#     save="_peaks_clust_1_timepoint_viridis.png"
# )

# %%
# Extract numeric values from categorical labels (remove "somites" and convert to int)
adata_sub2.obs["timepoint_numeric"] = (
    adata_sub2.obs["timepoint"].str.replace("somites", "").astype(float)
)

# Plot UMAP using a continuous colormap
sc.pl.umap(
    adata_sub2, 
    color="timepoint_numeric",  # Use the new numeric column
    color_map="viridis", alpha=0.7, # Apply viridis colormap
    save="_peaks_clust_1_scaled_counts_timepoint_viridis.png"
)

# %%
# # Extract numeric values from categorical labels (remove "somites" and convert to int)
# adata_sub2.obs["timepoint_numeric"] = (
#     adata_sub2.obs["timepoint"].str.replace("somites", "").astype(float)
# )

# # Plot UMAP using a continuous colormap
# sc.pl.umap(
#     adata_sub2, 
#     color="timepoint_numeric",  # Use the new numeric column
#     color_map="viridis", alpha=0.7, # Apply viridis colormap
#     save="_peaks_clust_1_6_timepoint_viridis.png"
# )

# %%
# map to the global UMAP coordinates
peaks_sub = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden"].isin(["1"])]

peaks_sub.obs["leiden"] = peaks_sub.obs_names.map(adata_sub2.obs["leiden"])
sc.pl.umap(peaks_sub, color="leiden", save="_peaks_clust_1_leiden_global_coords.png")

# %%
# # check the genes associated with the peaks
# # clust_names = ["1", "4", "5", "7", "11"]
# clust_names = adata_sub2.obs.leiden.unique()

# for clust in clust_names:
#     print(f"cluster {clust} has genes:")
#     list_genes_sub = adata_sub2[adata_sub2.obs.leiden==clust].obs["associated_gene"].unique().to_list()
#     print(list_genes_sub)
    

# %%
adata_sub2.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/peak_cluster_optic_cup_early.h5ad")

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## EDA 1-3. sub-clustering for the PSM+Somites cluster
# - sub-clustering the PSM+somites structure from the large peak UMAP

# %%
# Subset the data to only cluster 0
adata_sub3 = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"].isin(["8", "11"])]

# Recompute PCA & neighbors on the subset
sc.pp.pca(adata_sub3)
sc.pp.neighbors(adata_sub3)
sc.tl.umap(adata_sub3, min_dist=0.1)

# Run Leiden clustering with a new resolution
sc.tl.leiden(adata_sub3, resolution=0.4)  # Increase resolution for subclustering

# Plot the new UMAP with subclusters
sc.pl.umap(adata_sub3, color="leiden", legend_loc="on data")

# %%
# Plot the new UMAP with subclusters
sc.pl.umap(adata_sub3, color="leiden", save="_peaks_clust_8_11_leiden.png")
# sc.pl.umap(adata_sub2, color="leiden",legend_loc="on data")

# %%
sc.pl.umap(adata_sub3, color="leiden", save="_peaks_clust_8_11_leiden.pdf")

# %%
# Extract numeric values from categorical labels (remove "somites" and convert to int)
adata_sub3.obs["timepoint_numeric"] = (
    adata_sub3.obs["timepoint"].str.replace("somites", "").astype(float)
)

# Plot UMAP using a continuous colormap
sc.pl.umap(
    adata_sub3, 
    color="timepoint_numeric",  # Use the new numeric column
    color_map="viridis", alpha=0.7, # Apply viridis colormap
    save="_peaks_clust_8_11_timepoint_viridis.png"
)

# %%
# map to the global UMAP coordinates
peaks_sub = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden_0.7"].isin(["8", "11"])]

peaks_sub.obs["leiden"] = peaks_sub.obs_names.map(adata_sub3.obs["leiden"])
sc.pl.umap(peaks_sub, color="leiden", save="_peaks_clust_8_11_leiden_global_coords.png")

# %%
# # check the genes associated with the peaks
# # clust_names = ["1", "4", "5", "7", "11"]
# clust_names = adata_sub3.obs.leiden.unique()

# for clust in clust_names:
#     print(f"cluster {clust} has genes:")
#     list_genes_sub = adata_sub3[adata_sub3.obs.leiden==clust].obs["associated_gene"].unique().to_list()
#     print(list_genes_sub)
    

# %%
adata_sub3

# %%
# 1) Subset to clusters 3,1,0
clusters_to_keep = {"3", "1", "0"}
adata_subset = adata_sub3[adata_sub3.obs["leiden"].isin(clusters_to_keep)].copy()

# 2) Group by "associated_gene" and count how many peaks (obs_names) per gene
gene_peak_counts = (
    adata_subset.obs
    .groupby("associated_gene")
    .size()
    .sort_values(ascending=False)
)

# gene_peak_counts is now a Series of "gene_name -> number_of_peaks"
print(gene_peak_counts.head(20))

# %%
# (1) Subset to clusters of interest
clusters_to_keep = {"3", "1", "0"}
adata_subset = adata_sub3[adata_sub3.obs["leiden"].isin(clusters_to_keep)].copy()

# (2) Make a contingency table:
#     Rows = genes, Columns = cluster labels, Values = #peaks
peak_cluster_counts = (
    adata_subset.obs
      .groupby(["associated_gene", "leiden"])
      .size()
      .unstack(fill_value=0)
)

# peak_cluster_counts now has columns "0", "1", "3" (each cluster)
# and an index of all genes found in the subset.

# (3) Filter for genes that have peaks in *all three* clusters.
genes_in_all_3 = peak_cluster_counts[
    (peak_cluster_counts["3"] > 0)
    & (peak_cluster_counts["1"] > 0)
    & (peak_cluster_counts["0"] > 0)
]
print("Genes with peaks in clusters 3, 1, AND 0:")
print(genes_in_all_3)

# If instead you want "genes that have peaks in multiple clusters (≥2)",
# you could do:
genes_in_multiple = peak_cluster_counts[ (peak_cluster_counts > 0).sum(axis=1) >= 2 ]
print("Genes with peaks in at least two of the three clusters:")
print(genes_in_multiple)

# %%
adata_sub3[adata_sub3.obs["associated_gene"]=="kcnma1a"].obs

# %%
adata_sub3.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/peak_cluster_mesoderm.h5ad")

# %%
adata_sub3

# %%
sc.tl.leiden(adata_sub3, resolution=1, key_added="leiden_1_sub")
sc.pl.umap(adata_sub3, color="leiden_1_sub")

# %%
sc.pl.pca(adata_sub3, components=['1,2', '2,3'], color="leiden_0.7_sub")

# %%
from pyslingshot import Slingshot

# Step 1: Create integer mapping for leiden clusters
leiden_clusters = adata_sub3.obs['leiden'].unique()
category_to_integer = {category: i for i, category in enumerate(leiden_clusters)}
print("Category mapping:", category_to_integer)

# Step 2: Create new column with integer labels
new_annotation = 'leiden_integer'
adata_sub3.obs[new_annotation] = adata_sub3.obs['leiden'].astype('category').cat.codes

# Step 3: Set up and run Slingshot
# Create visualization plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Define starting cluster (cluster_id)
progenitor_cluster = "5"  # your starting cluster
progenitor_cluster_int = category_to_integer[progenitor_cluster]
# end_clusters = ["0","3","2"]
# end_clusters_int = [category_to_integer[x] for x in end_clusters]

# Run Slingshot
slingshot = Slingshot(adata_sub3, 
                     celltype_key=new_annotation,
                     obsm_key='X_umap',  # assuming you're using UMAP coordinates
                     start_node=progenitor_cluster_int, #end_nodes=end_clusters_int,
                     debug_level='verbose')

slingshot.fit(num_epochs=1, debug_axes=axes)
plt.show()

# Step 4: Visualize pseudotime
fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
axes[0].set_title('Clusters')
axes[1].set_title('Pseudotime')

slingshot.plotter.curves(axes[0], slingshot.curves)
slingshot.plotter.clusters(axes[0], labels=np.arange(slingshot.num_clusters), s=4, alpha=0.5)
slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)
plt.show()

# Step 5: Add pseudotime to AnnData object
pseudotime = slingshot.unified_pseudotime
adata_sub3.obs["Pseudotime"] = pseudotime

# Optional: Save results
# adata_sub3.obs.to_csv("pseudotime_results.csv")

# %%
adata_sub3.obs.to_csv("leiden_mesoderm_pseudotime_pyslingshot.csv")

# %%
sc.pl.umap(adata_sub3, color="Pseudotime", save="_mesoderm_cluster_pseudotime.png")

# %%
sc.pl.umap(adata_sub3, color="Pseudotime", save="_mesoderm_cluster_pseudotime.pdf")

# %%
adata_sub3.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/peak_cluster_mesoderm.h5ad")

# %%
[peaks_pb_hvp_50k.obs["leiden_0.7"] == "1"] or [peaks_pb_hvp_50k.obs["leiden_0.7"] == "6"]

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## EDA 1-4. sub-clustering for the hematopoetic system
#
# - sub-clustering the hemangioblas/hematopoetic vasculature in the cluster "9"

# %%
# Subset the data to only cluster 0
adata_sub4 = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden"].isin(["9"])]

# Recompute PCA & neighbors on the subset
sc.pp.pca(adata_sub4)
sc.pp.neighbors(adata_sub4)
sc.tl.umap(adata_sub4, min_dist=0.1)

# Run Leiden clustering with a new resolution
sc.tl.leiden(adata_sub4, resolution=0.7)  # Increase resolution for subclustering

# Plot the new UMAP with subclusters
sc.pl.umap(adata_sub4, color="leiden", legend_loc="on data")

# %%
sc.tl.leiden(adata_sub4, resolution=0.4)

# %%
# Plot the new UMAP with subclusters
sc.pl.umap(adata_sub4, color="leiden", save="_peaks_clust_9_leiden.png")
# sc.pl.umap(adata_sub2, color="leiden",legend_loc="on data")

# %%
# save as pdf
sc.pl.umap(adata_sub4, color="leiden", save="_peaks_clust_9_leiden.pdf")

# %%
# Extract numeric values from categorical labels (remove "somites" and convert to int)
adata_sub4.obs["timepoint_numeric"] = (
    adata_sub4.obs["timepoint"].str.replace("somites", "").astype(float)
)

# Plot UMAP using a continuous colormap
sc.pl.umap(
    adata_sub4, 
    color="timepoint_numeric",  # Use the new numeric column
    color_map="viridis", alpha=0.7, # Apply viridis colormap
    save="_peaks_clust_9_timepoint_viridis.png"
)

# %%
# map to the global UMAP coordinates
peaks_sub = peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["leiden"].isin(["9"])]

peaks_sub.obs["leiden"] = peaks_sub.obs_names.map(adata_sub4.obs["leiden"])
sc.pl.umap(peaks_sub, color="leiden", save="_peaks_clust_9_leiden_global_coords.png")

# %%
adata_sub4

# %%
# plot the signal of the celltype trajectory
# Plot UMAP using a continuous colormap
sc.pl.umap(
    adata_sub4, 
    color=["accessibility_hemangioblasts", "accessibility_hematopoietic_vasculature"], alpha=0.7,
    save="_peaks_clust_9_chr_acc_celltypes.png"
)

# %%
# Plot UMAP
sc.pl.umap(
    adata_sub4, 
    color=["celltype"], alpha=0.7,
    save="_peaks_clust_9_celltypes.png"
)

# %%
# check the genes in specific leiden sub-cluster
# cluster 6: hemangioblast, late
# cluster 7: hematopoetic_vasculature, late
clust_id = "3"
adata_sub4[adata_sub4.obs["leiden"]==clust_id].obs["linked_gene"].value_counts()

# %%
# Temporal hierarchy for hematopoietic (blood cell) development
blood_development_hierarchy = [
    "tal1",     # Earliest marker, initiates hematopoietic specification
    "lmo2",     # Early hematopoietic progenitor marker
    "myb",      # Intermediate stage, committed progenitors
    "gata1a",   # Later stage, erythroid commitment
    "hbae3",    # Terminal differentiation, mature erythroid cells
    # "hbae5"     # Terminal differentiation, mature erythroid cells
]

# Temporal hierarchy for vascular development
vascular_development_hierarchy = [
    "npas4l",   # Earliest vascular specification (cloche)
    "etv2",     # Early angioblast specification
    "fli1a",    # Vascular progenitor commitment
    # "fli1b",    # Vascular progenitor commitment
    "cdh5",     # Established endothelial identity (VE-cadherin)
    "flt4"      # Later stage vascular specialization (particularly lymphatic)
]

# Additional intermediate markers for finer resolution
intermediate_blood_markers = [
    "gfi1aa",
    "gfi1b",
    "ikzf1",
    "tfr1a"
]

intermediate_vascular_markers = [
    "tie1",
    "esama", 
    "clec14a",
    "rasip1"
]

# %%
# 1. Grab your UMAP coordinates and obs DataFrame:
umap_coords = adata_sub4.obsm["X_umap"]
obs = adata_sub4.obs

# 2. Make a scatterplot of *all* peaks in light gray (background):
plt.figure(figsize=(6,6))
plt.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c="lightgray",
    s=10,
    alpha=0.3,
    label="_background"  # leading underscore => won't clutter legend
)

# 3. Overlay each gene of interest in a different color:
# blood_cell_markers = [
#     "gata1a",
#     "hbae3",
#     "hbae5",
#     "myb",
#     "gfi1b",
#     "tfr1a"
# ]
blood_cell_markers = ["spi1", "gata1a", "lyz", "hbbe3", "mmp9"]

# vasculature_markers = [
#     "cdh5",
#     "fli1a",
#     "fli1b",
#     "flt4",
#     "clec14a",
#     "esama"
# ]

genes_of_interest = blood_development_hierarchy
# genes_of_interest = ["hbbe1.1","myf5","meox1","myog","myl1"]
colors = ["red","blue","green","purple","orange"]  # etc.
for gene, color in zip(genes_of_interest, colors):
    mask = (obs["linked_gene"] == gene)
    plt.scatter(
        umap_coords[mask, 0],
        umap_coords[mask, 1],
        c=color,
        s=40,
        label=gene
    )
    plt.grid(False)

plt.legend(markerscale=1.5)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("Peaks from select muscle/myotome genes")
plt.savefig(figpath + "umap_peaks_hemato_subclust_lineage.png")
plt.savefig(figpath + "umap_peaks_hemato_subclust_lineage.pdf")
plt.show()

# %%
genes_of_interest = vascular_development_hierarchy

# 2. Make a scatterplot of *all* peaks in light gray (background):
plt.figure(figsize=(6,6))
plt.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c="lightgray",
    s=10,
    alpha=0.3,
    label="_background"  # leading underscore => won't clutter legend
)

colors = ["red","blue","green","purple","orange"]  # etc.
for gene, color in zip(genes_of_interest, colors):
    mask = (obs["linked_gene"] == gene)
    plt.scatter(
        umap_coords[mask, 0],
        umap_coords[mask, 1],
        c=color,
        s=40,
        label=gene
    )
    plt.grid(False)

plt.legend(markerscale=1.5)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("Peaks from select muscle/myotome genes")
# plt.savefig(figpath + "umap_peaks_hemato_subclust_lineage.png")
# plt.savefig(figpath + "umap_peaks_hemato_subclust_lineage.pdf")
plt.show()

# %%

# %%

# %%
# 1. Grab your UMAP coordinates and obs DataFrame:
umap_coords = adata_sub4.obsm["X_umap"]
obs = adata_sub4.obs

# 2. Make a scatterplot of *all* peaks in light gray (background):
plt.figure(figsize=(6,6))
plt.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c="lightgray",
    s=10,
    alpha=0.3,
    label="_background"  # leading underscore => won't clutter legend
)

# 3. Overlay each gene of interest in a different color:
blood_cell_markers = [
    "gata1a",
    "hbae3",
    "hbae5",
    "myb",
    "gfi1b",
    "tfr1a"
]

vasculature_markers = [
    "cdh5",
    "fli1a",
    "fli1b",
    "flt4",
    "clec14a",
    "esama"
]

genes_of_interest = vasculature_markers
# genes_of_interest = ["hbbe1.1","myf5","meox1","myog","myl1"]
colors = ["red","blue","green","purple","orange"]  # etc.
for gene, color in zip(genes_of_interest, colors):
    mask = (obs["linked_gene"] == gene)
    plt.scatter(
        umap_coords[mask, 0],
        umap_coords[mask, 1],
        c=color,
        s=40,
        label=gene
    )
    plt.grid(False)

plt.legend(markerscale=1.5)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("Peaks from select muscle/myotome genes")
plt.savefig(figpath + "umap_peaks_hemato_vasculature_subclust_lineage.png")
plt.savefig(figpath + "umap_peaks_hemato_vasculature_subclust_lineage.pdf")
plt.show()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## plot the regions along the chromosome positions

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## utilities for genomic coordinates

# %% [markdown]
# ## utilities for genomic coordinates

# %%

# %% [markdown]
# ## EDA on the "marker genes/peaks" for specific cluster (using manual annotation on the peaks)

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["manual_annotation_peaks"]=="hatching_gland"].obs["gene_body_overlaps"].unique()

# %%
# query which genes were associated with the enriched peaks for each cluster
celltype = "muscle"
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["manual_annotation_peaks"]==celltype].obs["gene_body_overlaps"].unique()

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["manual_annotation_peaks"]=="NMPs"].obs["gene_body_overlaps"].value_counts()[0:20]

# %%
# query which genes were associated with the enriched peaks for each cluster
celltype = "tail_bud_late"
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs["manual_annotation_peaks"]==celltype].obs["gene_body_overlaps"].value_counts()[0:20]


# %%
# 1. Calculate gene enrichment per cluster
def analyze_peak_clusters(adata, cluster_key='leiden_r1'):
    """
    Analyze gene associations for each peak cluster
    """
    # Group peaks by cluster
    cluster_genes = {}
    cluster_stats = {}
    
    # For each cluster
    for cluster in adata.obs[cluster_key].unique():
        # Get peaks in this cluster
        cluster_peaks = adata.obs[adata.obs[cluster_key] == cluster]
        
        # Collect genes (both overlapping and nearest)
        all_genes = set()
        if 'gene_body_overlaps' in cluster_peaks.columns:
            genes = [g for genes in cluster_peaks['gene_body_overlaps'] if genes 
                    for g in genes.split(',') if g]
            all_genes.update(genes)
            
        if 'nearest_gene' in cluster_peaks.columns:
            all_genes.update(cluster_peaks['nearest_gene'].dropna())
        
        # Calculate statistics
        cluster_stats[cluster] = {
            'n_peaks': len(cluster_peaks),
            'n_genes': len(all_genes),
            'genes_per_peak': len(all_genes) / len(cluster_peaks) if len(cluster_peaks) > 0 else 0,
            'median_tss_distance': cluster_peaks['distance_to_tss'].median()
        }
        
        cluster_genes[cluster] = list(all_genes)
    
    return pd.DataFrame(cluster_stats).T, cluster_genes

# 2. Analyze peak type distributions per cluster
def analyze_peak_types(adata, cluster_key='leiden_r1'):
    """
    Analyze peak type distributions for each cluster
    """
    return pd.crosstab(
        adata.obs[cluster_key], 
        adata.obs['peak_type'], 
        normalize='index'
    )


# %%
# Example usage:
cluster_stats, cluster_genes = analyze_peak_clusters(peaks_pb_hvp_50k, 
                                                     cluster_key="manual_annotation_peaks")
peak_type_dist = analyze_peak_types(peaks_pb_hvp_50k, cluster_key="manual_annotation_peaks")

print("Cluster Statistics:")
print(cluster_stats)

print("\nPeak Type Distribution:")
print(peak_type_dist)


# %%

# %%
def analyze_specific_genes(adata, genes_of_interest, celltype_key='manual_annotation_peaks'):
    """
    Analyze distribution of specific genes across cell types
    """
    results = {}
    
    for gene in genes_of_interest:
        # Find all peaks associated with this gene
        cell_types = []
        for celltype in adata.obs[celltype_key].unique():
            genes = adata[adata.obs[celltype_key] == celltype].obs['gene_body_overlaps']
            # Check if gene appears in any peaks for this cell type
            if genes.str.contains(gene, na=False).any():
                cell_types.append(celltype)
        
        results[gene] = cell_types
    
    # Convert to DataFrame
    result_df = pd.DataFrame({
        'gene': list(results.keys()),
        'n_celltypes': [len(types) for types in results.values()],
        'celltypes': [','.join(types) for types in results.values()]
    })
    
    return result_df


# %%
genes_of_interest = ['myf5', 'tbxta', 'sox2']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print("\nDistribution of specific genes:")
print(gene_analysis)

# %%
genes_of_interest = ['myf5', 'msgn1', 'meox1']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print("\nDistribution of specific genes:")
print(gene_analysis)

# %%
peaks_pb_hvp_50k[peaks_pb_hvp_50k.obs_names=="19-14185848-14186763"].obs["nearest_gene"]

# %%
peaks_pb_hvp_50k


# %%
def analyze_specific_genes(adata, genes_of_interest, celltype_key='manual_annotation_peaks'):
    """
    Analyze distribution of specific genes across cell types using both gene_body_overlaps 
    and nearest_gene information
    """
    results = {}
    
    for gene in genes_of_interest:
        # Find all peaks associated with this gene
        cell_types = []
        gene_associations = {}  # To track how the gene was associated (overlap or nearest)
        
        for celltype in adata.obs[celltype_key].unique():
            celltype_peaks = adata[adata.obs[celltype_key] == celltype]
            
            # Check gene_body_overlaps first
            overlaps = celltype_peaks.obs['gene_body_overlaps']
            if overlaps.str.contains(gene, na=False).any():
                cell_types.append(celltype)
                gene_associations[celltype] = 'overlap'
                continue
                
            # If not found in overlaps, check nearest_gene
            nearest = celltype_peaks.obs['nearest_gene']
            if nearest.str.contains(gene, na=False).any():
                cell_types.append(celltype)
                gene_associations[celltype] = 'nearest'
        
        # Store results with association type
        results[gene] = {
            'celltypes': cell_types,
            'associations': gene_associations
        }
    
    # Convert to DataFrame with additional information
    result_df = pd.DataFrame({
        'gene': list(results.keys()),
        'n_celltypes': [len(res['celltypes']) for res in results.values()],
        'celltypes': [','.join(res['celltypes']) for res in results.values()],
        'association_details': [
            '; '.join([f"{ct}({assoc})" for ct, assoc in res['associations'].items()]) 
            for res in results.values()
        ]
    })
    
    return result_df

# Example usage:
"""
genes_of_interest = ['myf5', 'msgn1', 'meox1']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print(gene_analysis)
"""

# %%
genes_of_interest = ['myf5', 'msgn1', 'meox1']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print(gene_analysis)

# %%
genes_of_interest = ['myf5', 'tbxta', 'sox2']
gene_analysis = analyze_specific_genes(peaks_pb_hvp_50k, genes_of_interest)
print(gene_analysis)


# %%
# compute the n_celltypes and celltyeps for all genes for systematic analysis
def analyze_all_genes(adata, celltype_key='manual_annotation_peaks'):
    """
    Analyze distribution of all genes across cell types
    Returns a DataFrame with gene statistics
    """
    # Get all unique genes from gene_body_overlaps
    all_genes = set()
    for genes_str in adata.obs['gene_body_overlaps']:
        if isinstance(genes_str, str):
            genes = genes_str.split(',')
            all_genes.update([g.strip() for g in genes])
    
    # Remove empty strings if any
    all_genes = {g for g in all_genes if g}
    
    results = []
    
    print(f"Analyzing {len(all_genes)} genes...")
    for gene in tqdm(all_genes):
        cell_types = []
        # Get counts per celltype
        celltype_counts = {}
        
        for celltype in adata.obs[celltype_key].unique():
            # Get peaks for this celltype
            celltype_peaks = adata[adata.obs[celltype_key] == celltype]
            # Check which peaks contain this gene
            gene_peaks = celltype_peaks.obs['gene_body_overlaps'].str.contains(gene, na=False)
            if gene_peaks.any():
                cell_types.append(celltype)
                celltype_counts[celltype] = gene_peaks.sum()
        
        results.append({
            'gene': gene,
            'n_celltypes': len(cell_types),
            'celltypes': ','.join(cell_types),
            'total_peaks': sum(celltype_counts.values()),
            'celltype_counts': celltype_counts
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Add columns for peak counts per celltype
    for celltype in adata.obs[celltype_key].unique():
        result_df[f'peaks_in_{celltype}'] = result_df['celltype_counts'].apply(
            lambda x: x.get(celltype, 0)
        )
    
    # Drop the dictionary column
    result_df = result_df.drop('celltype_counts', axis=1)
    
    # Sort by number of celltypes and total peaks
    result_df = result_df.sort_values(['n_celltypes', 'total_peaks'], ascending=[False, False])
    
    return result_df


# %%
# Run analysis for all genes
gene_stats = analyze_all_genes(peaks_pb_hvp_50k)

# Look at top genes by number of cell types
print("\nGenes present in most cell types:")
print(gene_stats.head())

# Look at cell-type specific genes
print("\nCell-type specific genes:")
print(gene_stats[gene_stats['n_celltypes'] == 1].head())

# Save results
gene_stats.to_csv('peak_gene_analysis.csv')

# Get summary statistics
print("\nDistribution of genes across cell types:")
print(gene_stats['n_celltypes'].value_counts().sort_index())

# %%
gene_stats = pd.read_csv('peak_gene_analysis.csv', index_col=0)
gene_stats

# %%
gene_stats[gene_stats.gene=="myf5"]

# %%
gene_stats[gene_stats.gene=="meox1"]

# %%
celltype_specific_genes = gene_stats[gene_stats['n_celltypes'] == 1][gene_stats["celltypes"]!="unassigned"]
celltype_specific_genes

# %%
celltype_specific_genes[celltype_specific_genes.celltypes=="muscle"].gene.to_list()

# %%

# %%

# %%
