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
# ## Trajectory inference - pseudotime using PAGA (scanpy)
#
# - last updated: 5/20/2025
# - The goal is to infer the trajectory using PAGA for the peak UMAP
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
import os

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

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

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_umap_annotated/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")
adata_peaks

# %% [markdown]
# ## Step 1. Running PAGA (implemented in scanpy)
#
# - https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.paga.html
# - NOTE. PAGA uses neighborhoods to compute the abstracted "graph"

# %%
# check the UMAP
sc.pl.umap(adata_peaks, color="leiden_coarse")

# %%
# plot a force-directed graph
rsc.tl.draw_graph(adata_peaks, init_pos="X_umap_2D")
sc.pl.draw_graph(adata_peaks, color="leiden_coarse")

# %%
sc.tl.paga(adata_peaks, groups = "leiden_coarse")
sc.pl.paga(adata_peaks)

# %%
sc.pl.paga(adata_peaks, threshold=0.1)

# %%
# regenrate the 2D embedding using PAGA
rsc.tl.draw_graph(adata_peaks, init_pos='paga')
sc.pl.draw_graph(adata_peaks, color="leiden_coarse")

# %%
# Compute the Diffusion Pseudotime
adata_peaks.uns["iroot"] = np.flatnonzero(adata_peaks.obs["leiden_coarse"]==23)[-1]
sc.tl.dpt(adata_peaks)

# %%
adata_peaks.obsm["X_draw_graph_fa"][np.flatnonzero(adata_peaks.obs["leiden_coarse"]==23)]

# %%
np.flatnonzero(adata_peaks.obs["leiden_coarse"]==23)

# %%
sc.pl.draw_graph(adata_peaks, color=["leiden_coarse","dpt_pseudotime"], legend_loc="on data")

# %%
sc.pl.draw_graph(adata_peaks, color=["dpt_pseudotime"], vmin=0.4, vmax=0.8)

# %% [markdown]
# ### NOTE: DPT is not good at capturing the "smooth" pseudotime along the entire trajectories (good at the initial branches, though)

# %%
# subset for the initial clusters 23 and 11 and plot the pseudotime
sc.pl.draw_graph(adata_peaks[adata_peaks.obs["leiden_coarse"].isin([23, 11])], color="dpt_pseudotime")

# %%
# export the Diffusion Pseudotime to csv
adata_peaks.obs[["leiden_coarse","dpt_pseudotime"]].to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/dpt_pseudotime_640K_peaks.csv")
