# A python script to compute pseudotime using pySlingshot (Kelly Streets, et al.,2018)
# source: https://github.com/mossjacob/pyslingshot/blob/master/slingshot.ipynb
# NOTE: Run this on "single-cell-base" conda environment.

## Assumptions (to be revisited later):
# 1) we will assume that the "NMPs" will be the root cells for the pseudotime calculation.
# 2) we will assume the two lineages - mesodermal_lineages and neuroectodermal_lineages.
# mesodermal_lineages: "NMPs","tail_bud","PSM","somites","fast_muscle
# neuroectodermal_lineages: "NMPs","spinal_cord","neural_posterior"

# Step 0. Import libraries
import numpy as np
from matplotlib import pyplot as plt
from pyslingshot import Slingshot

import scanpy as sc
import anndata as ad
import pandas as pd
import argparse

"""
# Input arguments
# 1) filepath: AnnData object with the preprocessed data.
# 2) data_id: data identifier for the output files.
# 3) annotation: annotation class for celltypes (clusters)
# 4) progenitor_cluster: progenitor cluster to be used as the root for the pseudotime calculation.
# 5) embedding_key: key for the embedding to be used for the pseudotime calculation.
"""

# # Step 1. Set up argument parsing
# example for the input arguments
# filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/08_NMPs_subsetted/"
# data_id = "TDR118reseq_NMPs"
# annotation = "manual_annotation"
# progenitor_cluster = "NMPs"
# embedding_key = "X_umap_aligned"

filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad"
data_id = "integrated_nmps"
annotation="annotation_ML_coarse"
progenitor_cluster = "NMPs"
embedding_key = "X_umap_joint_3d"
output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/07_slingshot_processed/"

# import the integrated object (with the latest annotation)
adata = sc.read_h5ad(filepath)
adata

# import 3D UMAP
umap_3d = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_umap_3d.csv", index_col=0)
umap_3d

# filter out the "low_quality_cells"
umap_3d = umap_3d[umap_3d.index.isin(adata.obs_names)]
print(umap_3d.head())

# Add the 3D UMAP to the adata's obsm slot
adata.obsm["X_umap_joint_3d"] = np.array(umap_3d.values)

# subset the adata for the mesoderm/neuroectoderm trajectores
# define the lineage cell types
lineage_celltypes = ['neural_posterior',
                    'spinal_cord',
                    'NMPs',
                    'tail_bud',
                    'PSM',
                    'somites',
                    'fast_muscle']
# subset the adata
adata = adata[adata.obs[annotation].isin(lineage_celltypes), :]

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
progenitor_cluster = category_to_integer[progenitor_cluster]

slingshot = Slingshot(adata, celltype_key=new_annotation, 
                      obsm_key=embedding_key, start_node=progenitor_cluster, 
                      debug_level='verbose')

slingshot.fit(num_epochs=1, debug_axes=axes)

fig.savefig(output_path + f"{data_id}_pyslingshot_pseudotime_run.png")

# Additional pseudotime plots
fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
axes[0].set_title('Clusters')
axes[1].set_title('Pseudotime')
slingshot.plotter.curves(axes[0], slingshot.curves)
slingshot.plotter.clusters(axes[0], labels=np.arange(slingshot.num_clusters), s=4, alpha=0.5)
slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)
fig.savefig(output_path + f"{data_id}_pyslingshot_pseudotime.png")

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


# add the columns for lineages (mesoderm/neuroectoderm) - used later in CellOracle
Lineage_Meso = ["NMPs","tail_bud","PSM","somites","fast_muscle"]
Lineage_NeuroEcto = ["NMPs", "spinal_cord", "neural_posterior"]

adata.obs["Lineage_Meso"] = adata.obs[annotation].isin(Lineage_Meso)
adata.obs["Lineage_NeuroEcto"] = adata.obs[annotation].isin(Lineage_NeuroEcto)

# Fill in Pseudotime for Lineage_Meso where Lineage_Meso is True
adata.obs['Pseudotime_Lineage_Meso'] = adata.obs['Pseudotime'].where(adata.obs['Lineage_Meso'], np.nan)

# Fill in Pseudotime for Lineage_NeuroEcto where Lineage_NeuroEcto is True
adata.obs['Pseudotime_Lineage_NeuroEcto'] = adata.obs['Pseudotime'].where(adata.obs['Lineage_NeuroEcto'], np.nan)

# save the adata.obs as a dataframe (csv file) for further analysis
adata.obs.to_csv(output_path + f"{data_id}_slingshot.csv")

print("Pseudotime calculation is done.")
# save the updated adata object
# adata.write_h5ad(filepath + f"{data_id}_pyslingshot.h5ad")

