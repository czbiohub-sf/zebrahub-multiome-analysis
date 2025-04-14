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
#     display_name: single-cell-base
#     language: python
#     name: single-cell-base
# ---

# %% [markdown]
# ## compute leiden clustering on the cell UMAP (integrated object with weighted nearest neighbors)

# %%
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Import the custom module
import os
import sys
sys.path.append("/hpc/projects/data.science/yangjoon.kim/excellxgene_tutorial_manuscript/celltype_annotation_tutorial/utilities/")
from sankey import sankey
# help(sankey)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %%
# figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')
# Set default DPI for saved figures
mpl.rcParams['savefig.dpi'] = 600

# %%
import logging
# Suppress INFO-level logs for the entire logger
logging.getLogger().setLevel(logging.WARNING)
%matplotlib inline
# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/cell_umap_wnn_leiden_clustering/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the object
adata = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_leiden_filtered.h5ad")
adata

# %% Import the distances and connectivities from WNN (weighted nearest neighbors) - computed using Seurat
# %%
# import the distances from WNN (weighted nearest neighbors)
dist_df = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/wnn.csv")
dist_df.head()


# %%
# Display info about the dataframe
print(f"DataFrame shape: {dist_df.shape}")
print(f"Total pairs: {len(dist_df)}")
print(f"Number of unique cells: {dist_df['cell_name'].nunique()}")
print(f"Max neighbors per cell: {dist_df.groupby('cell_name').size().max()}")

# %%
# Get unique cell names to create the mapping
unique_cells = dist_df['cell_name'].unique()
n_cells = len(unique_cells)
print(f"Will create a {n_cells} x {n_cells} sparse matrix")

# Create a dictionary mapping cell names to row/column indices
cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}

# Initialize sparse matrices for distances and connectivities
# Use LIL format for efficient construction, then convert to CSR
distances = sp.lil_matrix((n_cells, n_cells), dtype=np.float32)
# connectivities = sp.lil_matrix((n_cells, n_cells), dtype=np.float32)

# Fill the matrices
# Note: This vectorized approach is much faster than iterating through rows
cell_indices = [cell_to_idx[name] for name in dist_df['cell_name']]
neighbor_indices = [cell_to_idx[name] for name in dist_df['neighbor_name']]
dist_values = dist_df['distance'].values

# Set the values in the matrices
for i, (cell_idx, neighbor_idx, dist) in enumerate(zip(cell_indices, neighbor_indices, dist_values)):
    # Set distance value
    distances[cell_idx, neighbor_idx] = dist
    
    # Set connectivity value (1 - distance)
    # This is a simple transformation - adjust if you have a better formula
    # connectivities[cell_idx, neighbor_idx] = 1.0 - dist
    
    # Print progress for large datasets
    if i % 1000000 == 0 and i > 0:
        print(f"Processed {i:,} / {len(dist_df):,} entries")

# Convert to CSR format for efficient operations
distances = distances.tocsr()
# connectivities = connectivities.tocsr()


# %%
import scipy.io

# Load the matrix in scipy sparse format
input_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"
wsnn_matrix = scipy.io.mmread(os.path.join(input_dir, "wsnn_matrix.mtx"))
# Make sure it's CSR format
wsnn_matrix = wsnn_matrix.tocsr()

# Load cell names
with open(os.path.join(input_dir, "cell_names.txt"), "r") as f:
    mtx_cell_names = [line.strip() for line in f]

# Add the WSNN graph as connectivities
adata.obsp['connectivities'] = wsnn_matrix

# %%
# add the distances to the adata
adata.obsp['distances'] = distances
adata

# %%
# Add the necessary information to adata.uns['neighbors']
adata.uns['neighbors'] = {
    'connectivities_key': 'connectivities',
    'distances_key': 'distances',  # Even if you don't have distances, you need this key
    'params': {
        'method': 'seurat_wnn',
        'n_neighbors': -1  # -1 indicates it's from an external source
    }
}
adata
# %%
# Only check ordering if the sets have the same cells
is_same_order = all(a == b for a, b in zip(adata.obs_names, mtx_cell_names))
print(is_same_order)

# %%
# adata.var = adata.var.reset_index()
# del adata.var["level_0"]
# Remove all columns containing "prediction.score" from adata.obs
cols_to_drop = [col for col in adata.obs.columns if "prediction.score" in col]
adata.obs = adata.obs.drop(columns=cols_to_drop)
adata
# %%
# save the adata object
del adata.raw
adata.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_wnn.h5ad")

# %% [markdown]
# ## re-compute the leiden clustering
# %%
res_leiden = [0.5, 0.8, 1, 1.2, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for res in res_leiden:
    sc.tl.leiden(adata, resolution=res, 
                 key_added=f"leiden_{res}")
    print(f"the number of clusters are:", len(adata.obs[f"leiden_{res}"].unique()))
adata

# %% [markdown]
# ## merge the singletons to the nearest leiden cluster
#
# - this is basically what Seurat's "FindCluster" does.
#
# %%
def group_singletons_seurat_style(
    adata,
    leiden_key: str = "leiden",
    adjacency_key: str = "connectivities",
    merged_key_suffix: str = "merged",
    group_singletons: bool = True,
    random_seed: int = 1
):
    """
    Replicates Seurat's 'GroupSingletons' post-processing step.
    - Finds clusters of size 1 (singletons) in adata.obs[leiden_key].
    - For each singleton, measures average connectivity to each other cluster
      by summing the adjacency submatrix SNN[i_cell, j_cells] and dividing
      by (# i_cells * # j_cells).
    - Reassigns the singleton cell to whichever cluster has the highest connectivity.
    - If there's a tie, picks randomly (set by random_seed).
    - Writes the merged labels to adata.obs[f"{leiden_key}_{merged_key_suffix}"].
    - If group_singletons=False, singletons remain in a “singleton” label.

    Parameters
    ----------
    adata : AnnData
        Your annotated data matrix.
    leiden_key : str
        Column in adata.obs where the initial Leiden (or other) clustering is stored.
    adjacency_key : str
        Key in adata.obsp containing an NxN adjacency matrix (e.g. "connectivities").
        Must be the same dimension as number of cells.
    merged_key_suffix : str
        Suffix to append when creating the merged labels column. The merged labels go
        in adata.obs[f"{leiden_key}_{merged_key_suffix}"].
    group_singletons : bool
        If True, merge singletons. If False, label them all "singleton".
    random_seed : int
        RNG seed for tie-breaking among equally connected clusters.

    Returns
    -------
    None
        (Modifies adata.obs in place, adding a column with merged labels.)
    """
    # Copy cluster labels
    old_labels = adata.obs[leiden_key].astype(str).values  # ensure string
    unique_labels, counts = np.unique(old_labels, return_counts=True)

    # Identify the singleton clusters (size=1)
    singleton_labels = unique_labels[counts == 1]

    # If not grouping them, just mark them as "singleton" and return
    if not group_singletons:
        new_labels = old_labels.copy()
        for s in singleton_labels:
            new_labels[new_labels == s] = "singleton"
        adata.obs[f"{leiden_key}_{merged_key_suffix}"] = new_labels
        adata.obs[f"{leiden_key}_{merged_key_suffix}"] = adata.obs[
            f"{leiden_key}_{merged_key_suffix}"
        ].astype("category")
        return

    # Otherwise, proceed to merge each singleton
    adjacency = adata.obsp[adjacency_key]
    new_labels = old_labels.copy()
    cluster_names = [cl for cl in unique_labels if cl not in singleton_labels]

    rng = np.random.default_rng(seed=random_seed)  # for tie-breaking

    for s_label in singleton_labels:
        i_cells = np.where(new_labels == s_label)[0]
        if len(i_cells) == 0:
            # Possibly already reassigned if something changed mid-loop
            continue

        # Seurat only has 1 cell for a singleton cluster, but let's be robust:
        # We'll compute the average connectivity for all i_cells anyway.
        # Usually i_cells will be length 1.
        sub_row_count = len(i_cells)

        best_cluster = None
        best_conn = -1  # track maximum average connectivity

        for j_label in cluster_names:
            j_cells = np.where(new_labels == j_label)[0]
            if len(j_cells) == 0:
                continue
            # Extract adjacency submatrix
            # shape is (len(i_cells), len(j_cells))
            sub_snn = adjacency[i_cells[:, None], j_cells]
            avg_conn = sub_snn.sum() / (sub_snn.shape[0] * sub_snn.shape[1])
            if np.isclose(avg_conn, best_conn):
                # tie => randomly pick
                if rng.integers(2) == 0:
                    best_cluster = j_label
                    best_conn = avg_conn
            elif avg_conn > best_conn:
                best_cluster = j_label
                best_conn = avg_conn

        if best_cluster is None:
            # If the singleton has zero connectivity to everything, you could:
            # (A) leave it in its own cluster, or
            # (B) label it "disconnected_singleton"
            # We'll leave it as is for now:
            continue

        # Reassign all i_cells to the chosen cluster
        new_labels[i_cells] = best_cluster

    # Store merged labels in adata.obs
    adata.obs[f"{leiden_key}_{merged_key_suffix}"] = new_labels
    # Remove any unused categories
    adata.obs[f"{leiden_key}_{merged_key_suffix}"] = adata.obs[
        f"{leiden_key}_{merged_key_suffix}"
    ].astype("category")
    adata.obs[f"{leiden_key}_{merged_key_suffix}"].cat.remove_unused_categories()
    
    return adata

# %%
res_leiden

# %%
# compute the merged labels for all the leiden clustering resolutions
for res in res_leiden:
    adata = group_singletons_seurat_style(
            adata,
            leiden_key=f"leiden_{res}",       # your Leiden column
            adjacency_key="connectivities", # your adjacency
            merged_key_suffix="merged",
            group_singletons=True,
            random_seed=1
)
adata

# %%
adata.var.head()

# %%
# save the adata object
adata.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_leiden.h5ad")

# %%
# RESUME HERE
# import the adata object
adata = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_leiden.h5ad")
adata

# %%
# print the number of clusters for each leiden clustering resolution
for res in res_leiden:
    print(f"the number of clusters for {res} are:", len(adata.obs[f"leiden_{res}_merged"].unique()))

# %%
sc.pl.embedding(adata, basis="X_wnn.umap",
                color=["leiden_0.8_merged", "leiden_1_merged","leiden_4_merged","leiden_10_merged"], legend_loc=None)
# %%
plt.show()

# %%
# help(sankey)
# adata.uns["leiden_0.01_merged_colors"]
del colorDict
# %%
key1 = "wsnn_res.0.8"
key2 = "leiden_0.8"

sankey(adata.obs[key1], adata.obs[key2])
plt.show()

# %%
key1 = "leiden_0.01_merged"
key2 = "leiden_0.1_merged"

# Get the colors from adata.uns
key1_colors = adata.uns[f"{key1}_colors"]  # This gets the color list
categories = adata.obs[key1].unique()  # Get unique categories
# Create the color dictionary mapping categories to their assigned colors
colorDict = {cat: key1_colors[i] for i, cat in enumerate(categories)}

sankey(adata.obs[key1], adata.obs[key2])
plt.show()
# %%

# %%
key1 = "leiden_1"
key2 = "leiden_1.2"

sankey(adata.obs[key1], adata.obs[key2])
plt.show()

# %%
key1 = "celltype"
key2 = "wsnn_res.0.8"

sankey(adata.obs[key1], adata.obs[key2])
plt.show()

# %%
help(sankey)

# %%
res_key = "leiden_1"
adata.obs[res_key].value_counts()

# %%
res_leiden

# %%
res_key = "wsnn_res.0.8"
sns.histplot(adata.obs[res_key].value_counts(), bins=40, kde=True)
plt.show()

# %%
# res_leiden.append("wsnn_res.0.8")
# res_leiden.append("celltype")
list_categories = ['wsnn_res.0.8', 'celltype', 'leiden_0.5', 'leiden_0.8', 'leiden_1', 'leiden_1.2', 'leiden_1.5', 'leiden_2']

for res_key in list_categories:
    adata.obs[res_key].value_counts().hist(bins=40, alpha=0.7)
    

plt.legend(list_categories)
plt.show()

# %%



# %%
adata.obs["leiden_0.8_merged"].unique()

# %%
# use the leiden clustering resolutions to compare the seurat and scanpy
key1 = "wsnn_res.0.8"
key2 = "leiden_0.8_merged"

sankey(adata.obs[key1], adata.obs[key2], threshold=5)
plt.show()

# %%
print(len(adata.obs[key1].unique()))
print(len(adata.obs[key2].unique()))

# %%
# res_leiden.append("wsnn_res.0.8")
# res_leiden.append("celltype")
list_categoaries = ['wsnn_res.0.8',  'leiden_0.8_merged']

for res_key in list_categories:
    adata.obs[res_key].value_counts().hist(bins=40, alpha=0.7)
    
plt.legend(list_categories)
plt.grid(False)
plt.show()

# %%
# Now, clean up the 
list_leiden_res = ['leiden_0.5', 'leiden_0.8', 'leiden_1', 'leiden_1.2', 'leiden_1.5', 'leiden_2', 'celltype', 'leiden_2.5', 'leiden_3', 'leiden_4', 'leiden_5']
for res in list_leiden_res:
    adata = group_singletons_seurat_style(
            adata,
            leiden_key=res,       # your Leiden column
            adjacency_key="connectivities", # your adjacency
            merged_key_suffix="merged",
            group_singletons=True,
            random_seed=1
            )
adata

# %%
adata.obs["leiden_5_merged"].unique()

# %%

# %%
adata_filt = adata[adata.obs["celltype"]!="low_quality_cells"]
adata_filt

# %% [markdown]
# ## import the cell type annotation (by Merlin Lange)
#

# %%
annotation = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/annotations/annotation_ML_06052024.txt",
                         index_col=0, sep="\t")
annotation.head()

# %%
annotation.annotation_ML_v4.unique()

# %%
len(annotation.annotation_ML_v4.unique())

# %%
adata.obs["celltype"] = adata.obs_names.map(annotation["annotation_ML_v4"])
sc.pl.embedding(adata, basis="X_wnn.umap", color="celltype")

# %% [markdown]
### plots for basic statistics
- a histogram showing the number of cells distribution for each leiden clustering resolution
- a histogram 

# %%
# plot the number of cells distribution for each leiden clustering resolution
# with kernel density estimation
# for res in res_leiden:
#     adata.obs[f"leiden_{res}_merged"].value_counts().hist(bins=40, alpha=0.7, density=True)

# plt.legend(res_leiden)
# plt.show()
adata

# %%
plt.figure(figsize=(10, 6))

for res in res_leiden:
    # Get the cluster sizes
    cluster_sizes = adata.obs[f"leiden_{res}_merged"].value_counts()
    
    # Plot histogram
    cluster_sizes.hist(bins=40, alpha=0.3, density=True, label=f'res={res}')
    
    # Add KDE curve
    sns.kdeplot(data=cluster_sizes, linewidth=2, label=f'KDE res={res}')

plt.xlabel('Cluster Size')
plt.ylabel('Density')
plt.title('Distribution of Cluster Sizes for Different Resolutions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
1
# %%
