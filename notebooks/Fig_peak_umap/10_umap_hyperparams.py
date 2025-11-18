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
# ## tuning the hyper-parameters for peak UMAP
#
# - this notebook is to support our claim that the peak UMAP has a tree-like structure, and to see whether the structure is preserved regardless of the UMAP hyper-parameters.
#
# - The parameters that we will tune are the following. We're referencing the tutorial from the UMAP package (https://umap-learn.readthedocs.io/en/latest/parameters.html)
#     - n_neighbors
#     - min_dist
#     - metric

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
import os
# import pyranges as pr

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# %%
# A) Use PGF Backend (40x smaller files!):
import matplotlib as mpl
# mpl.use('pgf')  # Add this before importing pyplot
import matplotlib.pyplot as plt

# B) Rasterize Dense Elements:
# plt.scatter(x, y, rasterized=True)  # For UMAP scatter plots
# ax.set_rasterization_zorder(0)     # Rasterize everything below z=0

# C) Optimize Font Embedding:
plt.rcParams['pdf.fonttype'] = 3  # Reduces font embedding size

# %%
# figure parameter setting
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')  # Use this at the very start of your script
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
mpl.rcParams['savefig.dpi'] = 300


# %%
import logging
# Suppress INFO-level logs for the entire logger
logging.getLogger().setLevel(logging.WARNING)


# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_umap_hyperparams/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
# NOTE. the 2 MT peaks and 2 blacklisted peaks (since they go beyond the end of the chromosome) were filtered out.
# adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad")
adata_peaks

# %%
# peak_type annotation using "Argelaguet 2022 style" (500bp upstream and 100bp downstream)
anno = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/all_peaks_annotated.csv",
                   index_col=0)
anno.head()

# %%
adata_peaks.obs["peak_type_argelaguet"] = anno["peak_type_argelaguet"].values

# %%
adata_peaks.obs["peak_type_argelaguet"].value_counts()

# %%
# make a copy of the data 
peaks_pb = adata_peaks.copy()

peaks_pb.X = peaks_pb.layers["normalized"]
rsc.get.anndata_to_GPU(peaks_pb) # moves `.X` to the GPU
rsc.pp.scale(peaks_pb)

# %% [markdown]
# ## 1) n_neighbors
# - scanpy's default number of k-nearest neighbors for computing UMAP is 15.

# %%
# ============================================================================
# 1. EXPLORING n_neighbors (number of k-nearest neighbors)
# ============================================================================
print("1. Exploring n_neighbors parameter...")

n_neighbors_values = [5, 10, 15, 20, 30, 50, 100]

for n_neighbors in n_neighbors_values:
    print(f"Computing UMAP with n_neighbors={n_neighbors}")
    
    # Copy the original data to avoid overwriting
    peaks_temp = peaks_pb.copy()
    
    # Compute neighbors with current n_neighbors
    rsc.pp.neighbors(peaks_temp, n_neighbors=n_neighbors, n_pcs=40)
    
    # Compute UMAP with fixed other parameters
    rsc.tl.umap(peaks_temp, min_dist=0.3, random_state=42)
    
    # Plot UMAP
    plt.figure(figsize=(3, 3))
    sc.pl.umap(peaks_temp, title=f'n_neighbors={n_neighbors}', color="peak_type_argelaguet", show=True,
               save=f"_n_neighbors_{n_neighbors}.png")
    # plt.tight_layout()
    # plt.show()

print("n_neighbors exploration completed.\n")

# %%

# %%

# %% [markdown]
# ## 2) min_dist

# %%
# 2. EXPLORING min_dist (minimum distance parameter)
# ============================================================================
print("2. Exploring min_dist parameter...")

min_dist_values = [0.01, 0.1, 0.3, 0.5, 1.0]

for min_dist in min_dist_values:
    print(f"Computing UMAP with min_dist={min_dist}")
    
    # Copy the original data
    peaks_temp = peaks_pb.copy()
    
    # Compute neighbors with fixed parameters
    rsc.pp.neighbors(peaks_temp, n_neighbors=15, n_pcs=40)
    
    # Compute UMAP with current min_dist
    rsc.tl.umap(peaks_temp, min_dist=min_dist, random_state=42)
    
    # Plot UMAP
    plt.figure(figsize=(3, 3))
    sc.pl.umap(peaks_temp, title=f'min_dist={min_dist}', color="peak_type_argelaguet", show=True,
               save=f"_min_dist_{min_dist}.png")
    # plt.tight_layout()
    # plt.show()

print("min_dist exploration completed.\n")


# %%

# %%

# %% [markdown]
# ## 3) number of PCs

# %%
# 3. EXPLORING n_pcs (number of principal components)
# ============================================================================
print("3. Exploring n_pcs parameter...")

n_pcs_values = [10, 20, 40, 50, 75, 100]

for n_pcs in n_pcs_values:
    print(f"Computing UMAP with n_pcs={n_pcs}")
    
    # Copy the original data
    peaks_temp = peaks_pb.copy()
    
    # Compute neighbors with current n_pcs
    rsc.pp.neighbors(peaks_temp, n_neighbors=15, n_pcs=n_pcs)
    
    # Compute UMAP with fixed other parameters
    rsc.tl.umap(peaks_temp, min_dist=0.3, random_state=42)
    
    # Plot UMAP
    plt.figure(figsize=(3, 3))
    sc.pl.umap(peaks_temp, title=f'n_pcs={n_pcs}', color="peak_type_argelaguet", show=True,
               save=f"_n_pcs_{n_pcs}.png")
print("n_pcs exploration completed.\n")

# %%

# %%

# %% [markdown]
# ## 4) metric

# %%
# 4. EXPLORING metric (distance metric)
# ============================================================================
print("4. Exploring metric parameter...")

# Note: Available metrics depend on your rapids-singlecell version
# Common metrics: 'euclidean', 'cosine', 'manhattan', 'chebyshev'
metrics = ['euclidean', 'cosine', 'manhattan']

for metric in metrics:
    print(f"Computing UMAP with metric='{metric}'")
    
    try:
        # Copy the original data
        peaks_temp = peaks_pb.copy()
        
        # Compute neighbors with current metric
        rsc.pp.neighbors(peaks_temp, n_neighbors=15, n_pcs=40, metric=metric)
        
        # Compute UMAP with fixed other parameters
        rsc.tl.umap(peaks_temp, min_dist=0.3, random_state=42)
        
        # Plot UMAP
        plt.figure(figsize=(3, 3))
        sc.pl.umap(peaks_temp, title=f'metric={metric}', color="peak_type_argelaguet", show=True,
                  save=f"_metric_{metric}.png")
        # plt.tight_layout()
        # plt.show()
        
    except Exception as e:
        print(f"Error with metric '{metric}': {e}")
        continue

print("metric exploration completed.\n")

# %% [markdown]
# ## Different 2D projection methods such as t-SNE, PHATE, etc.

# %%
# pip install phate pacmap trimap umap-learn scikit-learn

# %%
# 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)
print("1. Computing t-SNE...")

peaks_tsne = peaks_pb.copy()

# Compute t-SNE with similar parameters
rsc.tl.tsne(peaks_tsne, n_pcs=40)

# X_tsne = tsne.fit_transform(X_pca)
# peaks_tsne.obsm['X_tsne'] = X_tsne

plt.figure(figsize=(3, 3))
sc.pl.embedding(peaks_tsne, basis='tsne', color='peak_type_argelaguet', title='t-SNE', show=True)

# %%
# save the tSNE plot
plt.figure(figsize=(3, 3))
sc.pl.embedding(peaks_tsne, basis='tsne', color='peak_type_argelaguet', title='t-SNE', show=True,
                save="_default_params.png")

# %%
# bring the .X counts layer back to CPU (for PHATE and PaCMAP methods)
rsc.get.anndata_to_CPU(peaks_pb)

# %%
# 2. PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)
print("2. Computing PHATE...")

peaks_phate = peaks_pb.copy()

# Initialize PHATE
sc.external.tl.phate(peaks_phate, n_pca=40, n_landmark=100000, n_jobs=-1)

plt.figure(figsize=(3, 3))
sc.pl.embedding(peaks_phate, basis='phate', color='peak_type_argelaguet', title='PHATE', show=True,
                save="_default_params.png")


# %%
# 3. PaCMAP
print("3. Computing PaCMAP...")
import pacmap

peaks_pacmap = peaks_pb.copy()
rsc.pp.pca(peaks_pacmap, n_comps=40, use_highly_variable=False)
X_pca = peaks_pacmap.obsm['X_pca']

# PaCMAP parameters
pacmap_embedding = pacmap.PaCMAP(
    n_components=2,
    n_neighbors=15,  # Similar to UMAP
    MN_ratio=0.5,
    FP_ratio=2.0,
    random_state=42
)

X_pacmap = pacmap_embedding.fit_transform(X_pca)
peaks_pacmap.obsm['X_pacmap'] = X_pacmap

plt.figure(figsize=(3, 3))
sc.pl.embedding(peaks_pacmap, basis='pacmap', color='peak_type', title='PaCMAP', show=True,
                save="_default_params.png")
