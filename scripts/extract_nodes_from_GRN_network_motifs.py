# This script is from XX.ipynb notebook from the zebrahub_multiome_analysis github repo.
# This script extracts the nodes (gene_names) corresponding to the network motifs from a GRN.

# Inputs:
# GRN
# network_motif
# 

# Outputs:
# dataframe: gene


# NOTE: Run this script within the celloracle_env conda environment
# Import libraries
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import celloracle as co
co.__version__


