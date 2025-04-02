# A jupyter notebook to compute the peak UMAP for the 10x multiome data
# Last updated: 2025-03-25
# ---
 # jupyter:
 #   jupytext:
 #     text_representation:
 #       extension: .py
 #       format_name: percent
 #       format_version: '1.3'
 #       jupytext_version: 1.16.4
 #   kernelspec:
 #     module_name: data.science
 #     display_name: sc_rapids
 #     language: python
 #     name: sc_rapids
 # NOTE: This notebook is intended to be run in the rapids-singlecell environment (GPU)
 # ---
# %%
# 0. Import
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse
from scipy.io import mmread
# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc
# interacive shell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"