## 01_create_structured_prompts.py
## Last updated: 2025-02-19
## Author: YangJoon Kim

## This script creates structured prompts for LLM queries from single-cell multiome datasets.
## The first part is to process the peaks-by-pseudobulk data to create a set of vectors that can be used to query the LLM.
## The second part is to create a set of "prompts for the LLM queries.
# export OPENAI_API_KEY=

# %%
# import libraries
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import sys
import os

# from IPython.core.interactiveshell import interactiveShell
# InteraciveShell.ast_node_interactivity = “all”

# figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# # Editable text and proper LaTeX fonts in illustrator
# # matplotlib.rcParams['ps.useafm'] = True
# # Editable fonts. 42 is the magic number
# mpl.rcParams['pdf.fonttype'] = 42
# sns.set(style='whitegrid', context='paper')
# # Set default DPI for saved figures
# mpl.rcParams['savefig.dpi'] = 600

# # Plotting style function (run this before plotting the final figure)
# def set_plotting_style():
#     plt.style.use('seaborn-paper')
#     plt.rc('axes', labelsize=12)
#     plt.rc('axes', titlesize=12)
#     plt.rc('xtick', labelsize=10)
#     plt.rc('ytick', labelsize=10)
#     plt.rc('legend', fontsize=10)
#     plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
#     plt.rc('xtick.major', pad=2)
#     plt.rc('ytick.major', pad=2)
#     plt.rc('mathtext', fontset='stixsans', sf='sansserif')
#     plt.rc('figure', figsize=[10,9])
#     plt.rc('svg', fonttype='none')


# %%

# %%

# %%
