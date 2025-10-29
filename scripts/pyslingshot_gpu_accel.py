# A python script to compute pseudotime using pySlingshot (a python implementation of Slingshot).
# source: https://github.com/mossjacob/pyslingshot/blob/master/slingshot.ipynb
# NOTE: Run this on "single-cell-basics" conda environment. (basically a scanpy environment with pyslingshot pip-installed)

## Assumptions (to be revisited later):
# 1) we will assume that the "NMPs" will be the root cells for the pseudotime calculation.
# 2) we will assume the two lineages - mesodermal_lineages and neuroectodermal_lineages.
# mesodermal_lineages: "NMPs", "tail_bud", "PSM","somites", "fast_muscle"
# neuroectodermal_lineages: "NMPs", "Neural_Posterior", "Neural_Anterior"

# Step 0. Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyslingshot import Slingshot

import scanpy as sc
import anndata as ad
import argparse
import os

"""
# Input arguments
# 1) filepath: AnnData object with the preprocessed data.
# 2) data_id: data identifier for the output files.
# 3) annotation: annotation class for celltypes (clusters)
# 4) progenitor_cluster: progenitor cluster to be used as the root for the pseudotime calculation.
# 5) embedding_key: key for the embedding to be used for the pseudotime calculation.
# 6) figpath: path to save the figures/plots for diagnostics purpose.
"""

# Step 1. Set up argument parsing
parser = argparse.ArgumentParser(description="Run pySlingshot to compute pseudotime before in silico KO perturbation.")
parser.add_argument("filepath", type=str, help="Path to the AnnData object with the preprocessed data.")
parser.add_argument("data_id", type=str, help="Data identifier for the output files.")
parser.add_argument("annotation", type=str, help="Annotation class for celltypes (clusters).")
parser.add_argument("progenitor_cluster", type=str, help="Progenitor cluster to be used as the root for the pseudotime calculation.")
parser.add_argument("embedding_key", type=str, help="Key for the embedding to be used for the pseudotime calculation.")
parser.add_argument("figpath", type=str, help="Path to save the figures/plots for diagnostics purpose.")
args = parser.parse_args()

# Assign arguments to variables
filepath = args.filepath
data_id = args.data_id
annotation = args.annotation
progenitor_cluster = args.progenitor_cluster
embedding_key = args.embedding_key
figpath = args.figpath

# create the directory to save the diagnostic plots if it doesn't exist yet
os.makedirs(filepath + figpath, exist_ok=True)

# example for the input arguments
# filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"
# data_id = "TDR118reseq_NMPs"
# annotation = "manual_annotation"
# progenitor_cluster = "NMPs"
# embedding_key = "X_umap_aligned"
# figpath = "figs_diagnose/"

# Load the dataset
# adata = sc.read_h5ad(filepath + f"{data_id}.h5ad")
adata = sc.read_h5ad(filepath + f"{data_id}_nmps_manual_annotation.h5ad")
adata

# Assuming 'annotation' contains the categories as shown
try:
    categories = adata.obs[annotation].cat.categories
except:
    categories = adata.obs[annotation].unique()

# Create a dictionary mapping each category to an integer
category_to_integer = {category: i for i, category in enumerate(categories)}
# Print the mapping to verify
print(category_to_integer)

# Replace the categorical labels in the DataFrame with the mapped integers
new_annotation = annotation + "_integer"
adata.obs[new_annotation] = adata.obs[annotation].map(category_to_integer)

# Convert categorical labels to integer codes
try:
    adata.obs[new_annotation] = adata.obs[annotation].cat.codes
except:
    adata.obs[new_annotation] = adata.obs[annotation].astype('category').cat.codes

# Check the new column to ensure the mapping is correct and the datatype
print(adata.obs[new_annotation].head())
print(adata.obs[new_annotation].dtype)

# Run pySlingshot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
custom_xlim = (-12, 12)
custom_ylim = (-12, 12)
# plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)

# define the progenitor_cluster
progenitor_cluster_int = category_to_integer[progenitor_cluster]

slingshot = Slingshot(adata, celltype_key=new_annotation, 
                      obsm_key=embedding_key, start_node=progenitor_cluster_int, 
                      debug_level='verbose')

slingshot.fit(num_epochs=1, debug_axes=axes)

fig.savefig(filepath + figpath + f"{data_id}_pyslingshot_pseudotime_run.png")

# Additional pseudotime plots
fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
axes[0].set_title('Clusters')
axes[1].set_title('Pseudotime')
slingshot.plotter.curves(axes[0], slingshot.curves)
slingshot.plotter.clusters(axes[0], labels=np.arange(slingshot.num_clusters), s=4, alpha=0.5)
slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)
fig.savefig(filepath + figpath + f"{data_id}_pyslingshot_pseudotime.png")

# Save the pseudotime values to the dataframe
# NOTE: the Slingshot class has a property which has the pseudotime that is used to 
# color the plot above
pseudotime = slingshot.unified_pseudotime
pseudotime

# if there's already "Pseudotime" column, then replace it with the new one.
if "Pseudotime" in adata.obs.columns:
    # move the "Pseudotime" to "Pseudotime_DPT
    adata.obs["Pseudotime_DPT"] = adata.obs["Pseudotime"]
    adata.obs["Pseudotime"] = pseudotime
else:
    adata.obs["Pseudotime"] = pseudotime


# # add the columns for lineages (mesoderm/neuroectoderm) - used later in CellOracle
# Lineage_Meso = ["NMPs","tail_bud","PSM","somites","fast_muscle"]
# Lineage_NeuroEcto = ["NMPs", "spinal_cord", "neural_posterior"]

# adata.obs["Lineage_Meso"] = adata.obs[annotation].isin(Lineage_Meso)
# adata.obs["Lineage_NeuroEcto"] = adata.obs[annotation].isin(Lineage_NeuroEcto)

# # Fill in Pseudotime for Lineage_Meso where Lineage_Meso is True
# adata.obs['Pseudotime_Lineage_Meso'] = adata.obs['Pseudotime'].where(adata.obs['Lineage_Meso'], np.nan)

# # Fill in Pseudotime for Lineage_NeuroEcto where Lineage_NeuroEcto is True
# adata.obs['Pseudotime_Lineage_NeuroEcto'] = adata.obs['Pseudotime'].where(adata.obs['Lineage_NeuroEcto'], np.nan)

# save the adata.obs as a dataframe (csv file) for further analysis
adata.obs.to_csv(filepath + f"{data_id}_slingshot.csv")

# save the updated adata object
# adata.write_h5ad(filepath + f"{data_id}_pyslingshot.h5ad")