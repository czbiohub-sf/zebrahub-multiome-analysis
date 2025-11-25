# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: celloracle_env
#     language: python
#     name: celloracle_env
# ---

# %% [markdown]
# ## GRN EDA - for GRN subsetting
#
# - last updated: 7/1/2025
#
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
import os

import celloracle as co
co.__version__

# # rapids-singlecell
# import cupy as cp
# import rapids_singlecell as rsc

# %%
# figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')
# Set default DPI for saved figures
mpl.rcParams['savefig.dpi'] = 600


# %%
import logging
# Suppress INFO-level logs for the entire logger
logging.getLogger().setLevel(logging.WARNING)


# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%

# %%

# %% [markdown]
# ## Step 0. Import the GRNs (Links object)

# %%
# define the master directory for all Links objects (GRN objects from CellOracle)
oracle_base_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/"

# We're using "TDR118" as the representative for "15-somites", and drop the "TDR119" for now.
# We'll use the "TDR119" for benchmark/comparison of biological replicates later on.
list_files = ['TDR126', 'TDR127', 'TDR128',
              'TDR118', 'TDR125', 'TDR124']

# %%
# define an empty dictionary
dict_links = {}

# for loop to import all Links objects
for dataset in list_files:
    file_name = f"{dataset}/08_{dataset}_celltype_GRNs.celloracle.links"
    file_path = os.path.join(oracle_base_dir, file_name)
    dict_links[dataset] = co.load_hdf5(file_path)
    
    print("importing ", dataset)
    
dict_links

# %%
from module_grn_export import *

# %%
# export all GRN (both filtered and unfiltered) as csv files (dataframes) for easier exploration
export_stats = export_grn_data(dict_links, base_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/")


# %%
# Later, load specific datasets:
# Single celltype at one timepoint
nmp_10som = load_grn_by_timepoint_celltype("grn_exports", 10, "NMPs", "filtered")

# All timepoints for one celltype
psm_timecourse = load_all_grns_for_celltype("grn_exports", "PSM", "filtered")

# Or load the massive combined dataset
all_filtered = pd.read_csv("grn_exports/combined/all_filtered_grns.csv")

# %%

# %%
# Debug script to find the missing filtered GRN
import pandas as pd

def find_missing_filtered_grn(base_dir):
    """Find which celltype-timepoint combination is missing from filtered data"""
    
    # Load the export statistics
    stats_df = pd.read_csv(f"{base_dir}/metadata/export_statistics.csv")
    
    # Get all unfiltered combinations
    unfiltered = stats_df[stats_df['grn_type'] == 'unfiltered'][['timepoint_code', 'somite_stage', 'celltype']].copy()
    unfiltered['combination'] = unfiltered['timepoint_code'] + "_" + unfiltered['celltype']
    
    # Get all filtered combinations  
    filtered = stats_df[stats_df['grn_type'] == 'filtered'][['timepoint_code', 'somite_stage', 'celltype']].copy()
    filtered['combination'] = filtered['timepoint_code'] + "_" + filtered['celltype']
    
    # Find missing combinations
    missing_from_filtered = set(unfiltered['combination']) - set(filtered['combination'])
    
    print(f"Total unfiltered combinations: {len(unfiltered)}")
    print(f"Total filtered combinations: {len(filtered)}")
    print(f"Missing from filtered: {len(missing_from_filtered)}")
    
    if missing_from_filtered:
        print("\nMissing combinations:")
        for combo in missing_from_filtered:
            timepoint_code, celltype = combo.split("_", 1)
            unfiltered_row = unfiltered[unfiltered['combination'] == combo].iloc[0]
            print(f"  - Timepoint: {timepoint_code} (Stage {unfiltered_row['somite_stage']}), Celltype: {celltype}")
            
            # Check the actual edge count for this combination
            unfiltered_edges = stats_df[(stats_df['timepoint_code'] == timepoint_code) & 
                                      (stats_df['celltype'] == celltype) & 
                                      (stats_df['grn_type'] == 'unfiltered')]['n_edges'].iloc[0]
            print(f"    Unfiltered edges: {unfiltered_edges}")
    
    return missing_from_filtered, stats_df

def check_original_data_for_missing(dict_links, missing_combinations):
    """Check the original CellOracle objects for the missing combinations"""
    
    for combo in missing_combinations:
        timepoint_code, celltype = combo.split("_", 1)
        
        print(f"\nChecking {timepoint_code} - {celltype}:")
        
        links_obj = dict_links[timepoint_code]
        
        # Check if it exists in unfiltered
        if hasattr(links_obj, 'links_dict') and celltype in links_obj.links_dict:
            unfiltered_df = links_obj.links_dict[celltype]
            print(f"  Unfiltered: {len(unfiltered_df) if unfiltered_df is not None else 0} edges")
        
        # Check if it exists in filtered
        if hasattr(links_obj, 'filtered_links'):
            if celltype in links_obj.filtered_links:
                filtered_df = links_obj.filtered_links[celltype]
                if filtered_df is None:
                    print(f"  Filtered: None (filtering may have removed all edges)")
                elif len(filtered_df) == 0:
                    print(f"  Filtered: 0 edges (empty DataFrame)")
                else:
                    print(f"  Filtered: {len(filtered_df)} edges")
            else:
                print(f"  Filtered: Not present in filtered_links dict")
        else:
            print(f"  Filtered: No filtered_links attribute")

# Usage:
base_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/grn_exported/"

missing_combos, stats_df = find_missing_filtered_grn(base_dir)

# %%

# %%
