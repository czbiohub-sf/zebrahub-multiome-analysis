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

import os
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import celloracle as co
co.__version__

# #plt.rcParams["font.family"] = "arial"
# plt.rcParams["figure.figsize"] = [6,6]
# %config InlineBackend.figure_format = 'retina'
# plt.rcParams["savefig.dpi"] = 600

# %matplotlib inline

# Make folder to save plots
data_id = ""
save_folder = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_{data_id}/"
os.makedirs(save_folder, exist_ok=True)