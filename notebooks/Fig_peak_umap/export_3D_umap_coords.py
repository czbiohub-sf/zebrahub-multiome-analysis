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
# ## export 3D UMAP coordinates 
#
# - last updated: 9/30/2025
#
# - export the 3D UMAP coordinates and metadata for 3D UMAP visualizer by Kyle
#
# - the metadata columns are the following:
#     - chromosome
#     - peak_type (Argelaguet)
#     - (the most accessible) celltype
#     - (the most accessible) timepoint
#     - motif families?

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
# NOTE. the 2 MT peaks and 2 blacklisted peaks (since they go beyond the end of the chromosome) were filtered out.
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad")
adata_peaks

# %%
# read the annotation for the "peak_type_argelaguet"
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/all_peaks_annotated.csv"
annotated_peaks = pd.read_csv(filepath, index_col=0)
annotated_peaks.head()

# %%
adata_peaks.obs["peak_type_argelaguet"] = annotated_peaks["peak_type_argelaguet"].values
adata_peaks.obs.head()

# %%
# read the annotated peaks metadata, for celltype and timepoints (excluding the pseudobulk group with less than 20 cells)
annotated_peaks_ct_tp = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/all_peaks_annotated_ct_tp.csv",
                                    index_col=0)
annotated_peaks_ct_tp.head()

# %%
annotated_peaks_ct_tp.columns

# %%
# # copy over the metadata
adata_peaks.obs["celltype"] = adata_peaks.obs_names.map(annotated_peaks_ct_tp["celltype"])
adata_peaks.obs["celltype_contrast"] = adata_peaks.obs_names.map(annotated_peaks_ct_tp["celltype_contrast"])
adata_peaks.obs["timepoint"] = adata_peaks.obs_names.map(annotated_peaks_ct_tp["timepoint"])
adata_peaks.obs["timepoint_contrast"] = adata_peaks.obs_names.map(annotated_peaks_ct_tp["timepoint_contrast"])
adata_peaks.obs["lineage"] = adata_peaks.obs_names.map(annotated_peaks_ct_tp["lineage"])
adata_peaks.obs["lineage_contrast"] = adata_peaks.obs_names.map(annotated_peaks_ct_tp["lineage_contrast"])

# %%
# Create your DataFrame with metadata
df = pd.DataFrame(index=adata_peaks.obs_names)
# Add 3D UMAP coordinates
umap_3d = adata_peaks.obsm["X_umap_3D"]
df["UMAP_1"] = umap_3d[:, 0]
df["UMAP_2"] = umap_3d[:, 1]
df["UMAP_3"] = umap_3d[:, 2]

df["celltype"] = adata_peaks.obs["celltype"]
df["timepoint"] = adata_peaks.obs["timepoint"]
df["lineage"] = adata_peaks.obs["lineage"]
df["peak_type"] = adata_peaks.obs["peak_type_argelaguet"]
df["chromosome"] = adata_peaks.obs["chrom"]
df["leiden_coarse"] = adata_peaks.obs["leiden_coarse"]
df["leiden_fine"] = adata_peaks.obs["leiden_unified"]

# Export to CSV
df.to_csv("peak_umap_3d_annotated_v6.csv", index=True)

# Optional: Preview the DataFrame
print(f"DataFrame shape: {df.shape}")
print(df.head())

# %%
# save the master adata object for the peaks-by-pseudobulk with annotation
adata_peaks.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad")

