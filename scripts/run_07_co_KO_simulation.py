# A python script to compute in silico knock-out (KO) perturbation using CellOracle.
# Modified from "02_GRN/05_in_silico_KO_simulation/02_KO_simulation.ipynb" Jupyter notebook from CellOracle.
# source: https://github.com/morris-lab/CellOracle/blob/master/docs/notebooks/05_simulation/Gata1_KO_simulation_with_Paul_etal_2015_data.ipynb
# NOTE: Run this on "celloracle_env" conda environment.

# Description:
# In this notebook, we perform two analyses.
# 1) in silico TF perturbation to simulate cell identity shifts. CellOracle uses the GRN model to simulate cell identity shifts in response to TF perturbations. For this analysis, you will need the GRN models from the previous notebook.
# 2) Compare simulation vectors with developmental vectors. In order to properly interpret the simulation results, it is also important to consider the natural direction of development. First, we will calculate a pseudotime gradient vector field to recapitulate the developmental flow. Then, we will compare the CellOracle TF perturbation vector field with the developmental vector field by calculating the inner product scores. Let's call the inner product value as perturbation score (PS). Please see the step 5.6 for detail.

# Custom data class / object

# In this notebook, CellOracle uses four custom classes, Oracle, Links, Gradient_calculator, and Oracle_development_module.

# 1) Oracle is the main class in the CellOracle package. It is responsible for most of the calculations during GRN model construction and TF perturbation simulations.
# 2) Links is a class to store GRN data.
# 3) The Gradient_calculator calculates the developmental vector field from the pseudotime results. If you do not have pseudotime data for your trajectory, please see the pseudotime notebook to calculate this information. https://morris-lab.github.io/CellOracle.documentation/tutorials/pseudotime.html
# 4) The Oracle_development_module integrates the Oracle object data and the Gradient_calculator object data to analyze how TF perturbation affects on the developmental process. It also has many visualization functions.

## Define the input arguments
# Input Arguments:
# 1) output_path: filepath for the output
# 2) RNAdata_path: filepath for the adata (RNA)
# 3) baseGRN_path: filepath for the base GRN (parquet)
# 4) data_id: data_id
# 5) annotation: annotation class (cell-type annotation)
# 6) dim_reduce: dimensionality reduction embedding name

# Parse command-line argument
import argparse
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="making Oracle/Gradients objects ready for in silico KO simulation")
#output_path, RNAdata_path, baseGRN_path, data_id, annotation, dim_reduce
# Add command-line arguments
parser.add_argument('oracle_path', type=str, help="oracle filepath")
parser.add_argument('data_id', type=str, help="data_id")
parser.add_argument('annotation', type=str, help="celltype annotation class")
# parser.add_argument('dim_reduce', type=str, help="dim.reduction embedding name")
parser.add_argument('figpath', type=str, help="figure path")
# parser.add_argument('list_KO_genes', type=list, help="a list of KO genes")
parser.add_argument('list_KO_genes', type=str, help="a comma-separated list of KO genes")

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments as attributes of the 'args' object
oracle_path = args.oracle_path
data_id = args.data_id
annotation = args.annotation
# dim_reduce = args.dim_reduce
figpath = args.figpath
# list_KO_genes = args.list_KO_genes
list_KO_genes = args.list_KO_genes.split(',')

# input arguments
# oracle_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/08_NMPs_subsetted/"
# data_id = "TDR118reseq" # data identifier
# annotation = "manual_annotation"
# figpath = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_{data_id}/"
# list_KO_genes = [] # list of KO genes ("meox1", "pax6a", etc.)

## Step 0. Load the packages
import os
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import time
import math

import celloracle as co
co.__version__

# Step 1. Load the data
# 1.1. Load processed Oracle object
oracle_path = oracle_path + f"{data_id}/"
oracle = co.load_hdf5(oracle_path + f"10_{data_id}_pseudotime.celloracle.oracle")
oracle

# set the color palettes using "Set3"
# Cell types
cell_types = [
    'Epidermal', 'Lateral_Mesoderm', 'PSM', 'Neural_Posterior',
    'Neural_Anterior', 'Neural_Crest', 'Differentiating_Neurons',
    'Adaxial_Cells', 'Muscle', 'Somites', 'Endoderm', 'Notochord',
    'NMPs'
]

# Optional: Change the color palette
# Generate "Set3" color palette
set3_palette = sns.color_palette("Set3", n_colors=len(cell_types))

# Suppose the light yellow is the 10th color in the palette (9th index, as indexing is 0-based)
# and you want to replace it with teal color
teal_color = (0.0, 0.5019607843137255, 0.5019607843137255)  # RGB for teal
set3_palette[1] = teal_color  # Replace the light yellow with teal

# Assign colors to cell types
custom_palette = {cell_type: color for cell_type, color in zip(cell_types, set3_palette)}
# Change the color for 'NMPs' to a dark blue
custom_palette['NMPs'] = (0.12941176470588237, 0.4, 0.6745098039215687)  # Dark blue color

# Get the order of categories as they appear in the dataset
categories_in_order = oracle.adata.obs[annotation].cat.categories

# Reorder your palette dictionary to match this order
ordered_palette = {cat: custom_palette[cat] for cat in categories_in_order if cat in custom_palette}

# Now assign this ordered palette back to the AnnData object
annotation_color = annotation + "_color"
oracle.adata.uns[annotation_color] = ordered_palette

# Change the color palette in the Oracle object
# Extract the cell type annotations from the AnnData object
cell_types = oracle.adata.obs[annotation].values

# Map each cell type to its corresponding color
colors_for_cells = np.array([custom_palette[cell_type] for cell_type in cell_types])

# Replace the colorandum in the oracle object
oracle.colorandum = colors_for_cells

## Step 1.2. Load inferred GRNs (Links object)
links = co.load_hdf5(oracle_path + f"08_{data_id}_celltype_GRNs.celloracle.links")


## Step 2. MAke predictive models for simulation
# Here, we will need to fit the ridge regression models again. This process will take less time than the GRN inference in the previous notebook, because we are using the filtered GRN models.
links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, 
                              use_cluster_specific_TFdict=True)

## check the basic metadata in the Oracle object
print(oracle)
# Save the oracle object (now, the Oracle object is "READY" for in silico KO simulation)
oracle.to_hdf5(oracle_path + f"10_{data_id}_pseudotime.celloracle.oracle")


## Step N-1. Visualization (Optional) - this is probably the best to leave for a Jupyter notebook, as it requires some interactive visualization
# such as picking the "scale" parameter for scaling the developmental vector field, 
# and the "n_grid" parameter as well as "min_mass" parameter for the visualization of the perturbation score.
# n_grid: number of grid points (in 2D embedding)
# min_mass: threshold value for the cell density

# we need to simulate the embedding shift (for "delta_embedding"), so we'll run this for one gene
goi = "meox1"

print(f"Simulating KO of {goi}")
oracle.simulate_shift(perturb_condition={goi: 0.0},
                  n_propagation=3)
# Get transition probability
oracle.estimate_transition_prob(n_neighbors=200,
                                knn_random=True, 
                                sampled_fraction=1)
# Calculate embedding 
oracle.calculate_embedding_shift(sigma_corr=0.05)

# n_grid = 40 is a good starting value.
n_grid = 40 
oracle.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

# compute the min_mass
# Here, we're assuming that the median value from the min_mass suggestion would be ideal.
# this might not be true, and one might need to curate this in Jupyter notebook later.
p_mass_min = oracle.total_p_mass.min()
p_mass_max = oracle.total_p_mass.max()
n_suggestion = 12

suggestions = np.linspace(p_mass_min, p_mass_max/2, n_suggestion)

# pick the 6th value from the suggestions
min_mass = math.floor(suggestions[5])

# calculate the grid points based on n_grid and min_mass
oracle.calculate_mass_filter(min_mass=min_mass, plot=True)


## Step N. computing the developmental flow (Gradient class)
# This step is necessary for the comparison of the simulation results with the developmental flow.
from celloracle.applications import Gradient_calculator

# Load the WT Oracle object
oracle_wt = co.load_hdf5(oracle_path + f"10_{data_id}_pseudotime.celloracle.oracle")

# Instantiate Gradient calculator object
gradient = Gradient_calculator(oracle_object=oracle_wt, pseudotime_key="Pseudotime")

# add the hyper-parameters for data viz/projection to 2D embeddings
gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)

# Transfer pseudotime values to the grid points
# Next, we convert the pseudotime data into grid points. For this calculation we can chose one of two methods.
# - (1) `knn`: K-Nearesr Neighbor regression. You will need to set number of neighbors.
#   Please adjust `n_knn` for best results.This will depend on the population size and density of your scRNA-seq data.

#  `gradient.transfer_data_into_grid(args={"method": "knn", "n_knn":50})`
# - (2) `polynomial`: Polynomial regression using x-axis and y-axis of dimensional reduction space.
#  In general, this method will be more robust. Please use this method if knn method does not work.
#  `n_poly` is the number of degrees for the polynomial regression model. Please try to find appropriate`n_poly` searching for best results.
 
#  `gradient.transfer_data_into_grid(args={"method": "polynomial", "n_poly":3})`
n_knn = oracle.k_knn_imputation
gradient.transfer_data_into_grid(args={"method": "knn", "n_knn":n_knn}, plot=True)

# Calculate graddient
gradient.calculate_gradient()

# # Show results
# scale_dev = 50
# gradient.visualize_results(scale=scale_dev, s=5)

# # Plot vector field with cell cluster 
# fig, ax = plt.subplots(figsize=[8, 8])

# oracle.plot_cluster_whole(ax=ax, s=10)
# gradient.plot_dev_flow_on_grid(scale=40, ax=ax, show_background=False)
# plt.savefig(save_folder + "umap_nmps_15somites_dev_flow.png")
# plt.savefig(save_folder + "umap_nmps_15somites_dev_flow.pdf")

# Save gradient object if you want.
gradient.to_hdf5(oracle_path + f"11_{data_id}_pseudotime_knn.celloracle.gradient")

## Step N+1. Perform in silico knock-out simulation
# Next, we will simulate the TF perturbation effects on cell identity to investigate its potential functions and regulatory mechanisms.

# First, create a directory to save the cell-cell transition probability matrices
# cell_trans_prob_dir = oracle_path + "in_silico_KO_trans_probs/"
# os.makedirs(cell_trans_prob_dir, exist_ok=True)

# # Make a folder to save the plots
# figpath = figpath + f"in_silico_KO_{data_id}/"
# os.makedirs(figpath, exist_ok=True)

# for gene in list_KO_genes:
#     print(f"Simulating KO of {gene}")
#     oracle.simulate_shift(perturb_condition={goi: 0.0},
#                       n_propagation=3)
#     # Get transition probability
#     oracle.estimate_transition_prob(n_neighbors=200,
#                                     knn_random=True, 
#                                     sampled_fraction=1)
#     # Calculate embedding 
#     oracle.calculate_embedding_shift(sigma_corr=0.05)

#     # [TO-DO] export the transition probabilities and save it as a csv file

#     # Plot vector field with cell cluster 
#     fig, ax = plt.subplots(figsize=[8, 8])

#     oracle.plot_cluster_whole(ax=ax, s=10)
#     oracle.plot_simulation_flow_on_grid(scale=10, ax=ax, show_background=False)
#     plt.savefig(figpath + f"umap_nmps_{data_id}_{gene}_KO.png")
#     plt.savefig(figpath + f"umap_nmps_{data_id}_{gene}_KO.pdf")