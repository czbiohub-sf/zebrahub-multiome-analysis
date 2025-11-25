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
# ## Subset/Zoom-into the GRNs using regulatory programs
#
# - 1) EDA on regulatory programs (peak clusters) that has interesting set of TFs and genes.
# - 2) make the EDA on 1) in a systematic way (using modules and functions)
# - 3) Use the module to subset the big GRN -> visualize (maybe in a separate notebook)

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom
import sys
import os

from gimmemotifs import maelstrom

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# For maelstrom module
import logging
from functools import partial
from multiprocessing import Pool
from scipy.stats import pearsonr

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
# import custom utils

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/reg_programs/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")
adata_peaks

# %%
# import the annotation ("associated_genes")
df_genes_anno = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/all_peaks_annotated.csv",
                            index_col=0)
df_genes_anno.head()

# %%
# # copy over the metadata ("linked_gene", etc.)
metadata_list = ['linked_gene', 'link_score', 'link_zscore', 'link_pvalue']
for col in metadata_list:
    adata_peaks.obs[col] = adata_peaks.obs_names.map(df_genes_anno[col])

adata_peaks.obs["linked_gene"]


# %%
print(adata_peaks.obs["linked_gene"].unique()[0:10])

# %%
# Check the example genes for each peak cluster
for cluster in adata_peaks.obs["leiden_coarse"].unique():
    # subset
    adata_sub = adata_peaks[adata_peaks.obs["leiden_coarse"]==cluster]
    num_genes = len(adata_sub.obs["linked_gene"].unique())
    # print out the example list of genes
    print(f"cluster {cluster} has {num_genes} associated genes:")
    print(adata_sub.obs["linked_gene"].unique()[1:11])

# %%
# import the functions to annotate the peaks with "associated genes"
# this function uses "linked_genes" from Signac and "overlapping with gene body" based on GTF file
from utils_gene_annotate import *

# %%
# associate peaks to genes
# (1) use "linked_gene" if possible
# (2) use "gene_body_overlaps" as secondary
# (3) add NaN otherwise
adata_peaks = create_gene_associations(adata_peaks)
adata_peaks.obs.head()

# %%
adata_peaks.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")
print("updated object saved")

# %%

# %%
