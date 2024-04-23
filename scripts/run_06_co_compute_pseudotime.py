# A python script to compute Diffusion Pseudotime (DPT) before in silico KO perturbation.
# Modified from "02_GRN/05_in_silico_KO_simulation/01_Pseudotime_calculation_script.ipynb" Jupyter notebook from CellOracle.
# source: https://morris-lab.github.io/CellOracle.documentation/notebooks/05_simulation/Pseudotime_calculation_with_Paul_etal_2015_data.html
# NOTE: Run this on "celloracle_env" conda environment.

## Assumptions:
# 1) we will assume that the "NMPs" will be the root cells for the pseudotime calculation.
# 2) we will assume the two lineages - mesodermal_lineages and neuroectodermal_lineages.
# mesodermal_lineages: "NMPs", "PSM", "Somites", "Muscle"
# neuroectodermal_lineages: "NMPs", "Neural_Posterior", "Neural_Anterior"

# Step 0. Import libraries
import copy
import glob
import time
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm.auto import tqdm

import celloracle as co
from celloracle.applications import Pseudotime_calculator
co.__version__

import time

# This notebook takes an "Oracle" object, then outputs the updated "Oracle" object with pseudotime information.
# The pseudotime information is computed using the diffusion pseudotime (DPT) algorithm.


####### NEED TO REVIST THE INPUT ARGUMENTS #######
# Input Arguments:
# 1) input_path: filepath for the inputs (the oracle object)
# 2) data_id: data_id
# 3) dim_reduce: dimensionality reduction embedding name. i.e. 'umap.atac', 'umap.joint', 'pca'
# 4) annotation: annotation class (cell-type annotation)

# Parse command-line arguments
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Filter and map data using CellOracle")
#input_path, data_id, dim_reduce, annotation
# Add command-line arguments
parser.add_argument('input_path', type=str, help="input filepath")
parser.add_argument('data_id', type=str, help="data_id")
parser.add_argument('dim_reduce', type=str, help="dim.reduction embedding name")
#parser.add_argument('annotation', type=str, help="celltype annotation class")
#parser.add_argument('timepoints', type=list, help="a list of timepoints")

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments as attributes of the 'args' object
input_path = args.input_path
data_id = args.data_id
dim_reduce = args.dim_reduce
#annotation = args.annotation
#timepoints = args.timepoints

# Define the function here
def compute_pseudotime_oracle(input_path, data_id, dim_reduce):
    """
    A function to compute pseudotime for an Oracle object using DPT.
    
    Parameters:
        input_path: filepath for the input files (below)
        data_id: identifier for the input/output Oracle file
        dim_reduce: dim.reduction name

    
    Returns: 
        Oracle: an Oracle object with pseudotime. NOTE that the results are saved in 
        oracle.adata.obs["Pseudotime"] for cell-level pseudotime information.

    Examples:
        input_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/08_NMPs_subsetted/TDR118reseq/"
        data_id = "TDR118reseq"
        dim_reduce: "X_umap_aligned"
    """

# Step 0. load the Oracle object
oracle = co.load_hdf5(input_path + f"06_{data_id}.celloracle.oracle")
# links = co.load_hdf5(input_path + f"08_{data_id}_celltype_GRNs.celloracle.links")

# redefine the default embedding for the oracle object 
oracle.embedding = oracle.adata.obsm[dim_reduce]
oracle.embedding_name = dim_reduce

# Step 1. computing the pseudotime (DPT)
# Instantiate pseudotime object using oracle object
pt = Pseudotime_calculator(oracle_object=oracle)

# Add lineage information (pt.cluster_column_name is the adata.obs category used to compute the Oracle object, and GRNs)
print("Clustering name: ", pt.cluster_column_name)
print("Cluster list", pt.cluster_list)

# Here, clusters can be classified into either mesoderm lineage or neuro-ectoderm lineage
clusters_in_meso_lineage = ["NMPs", "PSM", "Somites","Muscle"]
clusters_in_neuroecto_lineage = ["NMPs", "Neural_Posterior", "Neural_Anterior"]

# Make a dictionary
lineage_dictionary = {"Lineage_Meso": clusters_in_meso_lineage,
           "Lineage_NeuroEcto": clusters_in_neuroecto_lineage}

# Input lineage information into pseudotime object
pt.set_lineage(lineage_dictionary=lineage_dictionary)

# Visualize lineage information
pt.plot_lineages()

# Step 1-2. define the root cells
# extract the centroid cell, and define it as the "root cell" (this logic can be revisited with better workflow)
adata = oracle.adata
# subset the NMP population
adata_nmps = adata[adata.obs.manual_annotation=="NMPs"]
# extract the NMP UMAP coordinates
nmp_coords = pd.DataFrame(adata_nmps.obsm["X_umap_aligned"], columns=["UMAP_1","UMAP_2"], index=adata_nmps.obs_names)

# calculate the centroid of the NMP population
centroid = nmp_coords[['UMAP_1', 'UMAP_2']].mean().values

# Find the closest cell to the centroid
nmp_coords['distance_to_centroid'] = np.sqrt((nmp_coords['UMAP_1'] - centroid[0]) ** 2 + (nmp_coords['UMAP_2'] - centroid[1]) ** 2)
closest_cell_index = nmp_coords['distance_to_centroid'].idxmin()

# Now you have the index of the closest cell to the centroid
root_cell_id = closest_cell_index
print(f"The root cell index for the NMP population is: {root_cell_id}")

# Estimated root cell name for each lineage (AATGGCGCAGCTAACC-1)
root_cells = {"Lineage_Meso": root_cell_id, "Lineage_NeuroEcto": root_cell_id}
pt.set_root_cells(root_cells=root_cells)

# Check root cell and lineage
pt.plot_root_cells()

# Step 1-3. compute the diffusion map
if "X_diffmap" in pt.adata.obsm:
    print("Diffmap already computed.")
else:
    sc.pp.neighbors(pt.adata, n_neighbors=20)
    sc.tl.diffmap(pt.adata)
    print("Diffmap just computed.")

# Step 1-4. compute the pseudotime
pt.get_pseudotime_per_each_lineage()

# Check results
pt.plot_pseudotime(cmap="rainbow")

# Add calculated pseudotime data to the oracle object
oracle.adata.obs = pt.adata.obs

# Save updated oracle object
input_path + f"06_{data_id}.celloracle.oracle"
oracle.to_hdf5(input_path + f"10_{data_id}_pseudotime.celloracle.oracle")

####### LINUX TERMINAL COMMANDS #######
compute_cluster_specific_GRNs(output_path, RNAdata_path, baseGRN_path, 
                                    data_id, annotation, dim_reduce)