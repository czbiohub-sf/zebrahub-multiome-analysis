# A python script to compute in silico knock-out (KO) perturbation using CellOracle.
# Modified from "02_GRN/05_in_silico_KO_simulation/data_id_KO_simulation.ipynb" Jupyter notebook from CellOracle.

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

########## NEED TO WORK FROM HERE (4/17/2024)


####### NEED TO REVIST THE INPUT ARGUMENTS #######
# Input Arguments:
# 1) input_path: filepath for the inputs (oracle and links objects)
# 2) data_id: data_id
# 3) dim_reduce: dimensionality reduction embedding name. i.e. 'umap.atac', 'umap.joint', 'pca'

# 5) annotation: annotation class (cell-type annotation)
# 6) 

# Parse command-line arguments
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Filter and map data using CellOracle")
#output_path, RNAdata_path, baseGRN_path, data_id, annotation, dim_reduce
# Add command-line arguments
parser.add_argument('output_path', type=str, help="output filepath")
parser.add_argument('RNAdata_path', type=str, help="RNA data filepath")
parser.add_argument('baseGRN_path', type=str, help="base GRN filepath")
parser.add_argument('data_id', type=str, help="data_id")
parser.add_argument('annotation', type=str, help="celltype annotation class")
parser.add_argument('dim_reduce', type=str, help="dim.reduction embedding name")
#parser.add_argument('timepoints', type=list, help="a list of timepoints")

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments as attributes of the 'args' object
output_path = args.output_path
RNAdata_path = args.RNAdata_path
baseGRN_path = args.baseGRN_path
data_id = args.data_id
annotation = args.annotation
dim_reduce = args.dim_reduce
#timepoints = args.timepoints

####### NEED TO REVIST THE INPUT ARGUMENTS #######

# Define the function here

# Step 0. load the Oracle and Links objects
oracle = co.load_hdf5(input_path + f"06_{data_id}.celloracle.oracle")
links = co.load_hdf5(input_path + f"08_{data_id}_celltype_GRNs.celloracle.links")

# redefine the default embedding for the oracle object 
oracle.embedding = oracle.adata.obsm[dim_reduce]
oracle.embedding_name = dim_reduce

# Step 1. computing the pseudotime (DPT)
# Instantiate pseudotime object using oracle object
pt = Pseudotime_calculator(oracle_object=oracle)

# Add lineage information
print("Clustering name: ", pt.cluster_column_name)
print("Cluster list", pt.cluster_list)

# Define the function here


####### LINUX TERMINAL COMMANDS #######
compute_cluster_specific_GRNs(output_path, RNAdata_path, baseGRN_path, 
                                    data_id, annotation, dim_reduce)