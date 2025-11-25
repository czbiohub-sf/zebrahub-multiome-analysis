# %% [markdown]
# # Preprocess peak objects
# 
# This notebook preprocesses the peak objects for the EDA_peak_umap_cross_species notebook.
# 
# %% Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# %% Dataset 1. Argelaguet 2022 mouse peak objects
# 1) file paths for Argelaguet 2022 mouse peak objects
peak_objects_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/PeakMatrix_anndata.h5ad"

# 2) load the peak objects
peak_objects = sc.read_h5ad(peak_objects_path)

# %% 
# 3) inspect the peak objects
print(peak_objects.shape)
print(peak_objects.var.shape)
print(peak_objects.obs.shape)

# %% 
# 4) inspect the peak objects celltype and stage
print(peak_objects.obs["celltype.mapped"].value_counts())
print(peak_objects.obs["stage"].value_counts())

# %% 
# 5) inspect the peak objects
print(peak_objects.var.head())
print(peak_objects.obs.head())

# %% pseudobulk the peak objects to create peaks-by-pseudobulk (celltype-AND-stage) matrix
