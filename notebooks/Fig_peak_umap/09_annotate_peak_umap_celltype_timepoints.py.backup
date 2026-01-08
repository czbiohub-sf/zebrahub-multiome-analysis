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
#     display_name: sc_rapids
#     language: python
#     name: sc_rapids
# ---

# %% [markdown]
# ## Annotate peak UMAP with celltype and timepoints
#
# - last updated: 9/15/2025
#
# - Revisit this to remove the pseudobulk groups with too few cells (<10 cells)
#

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
import os
import re

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

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
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_umap_annotated/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# import the peaks-by-celltype&timepoint pseudobulk object
# NOTE. the 2 MT peaks and 2 blacklisted peaks (since they go beyond the end of the chromosome) were filtered out.
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")
adata_peaks

# %%
# load the pseudobulk by peaks
pb_by_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/pseudobulk_ct_tp_by_peaks.h5ad")
pb_by_peaks

# %%
pb_by_peaks.var.head()

# %%
adata_peaks.obs_names

# %%
# # copy over the variance and hvp info to the adata_peaks object (.obs)
# first, subset for the shared peaks (filtering out the 4 peaks that are either MT peaks or peaks extend beyond the chromosome end)
pb_by_peaks = pb_by_peaks[:,pb_by_peaks.var_names.isin(adata_peaks.obs_names)]

# %%
adata_peaks.obs_names

# %%
# # copy over the metadata
adata_peaks.obs["variances"] = adata_peaks.obs_names.map(pb_by_peaks.var["variances"])
adata_peaks.obs["means"] = adata_peaks.obs_names.map(pb_by_peaks.var["means"])
adata_peaks.obs["variances_norm"] = adata_peaks.obs_names.map(pb_by_peaks.var["variances_norm"])
adata_peaks.obs["highly_variable_peaks"] = adata_peaks.obs_names.map(pb_by_peaks.var["highly_variable"])
adata_peaks.obs["highly_variable_peaks_rank"] = adata_peaks.obs_names.map(pb_by_peaks.var["highly_variable_rank"])

# %%
adata_peaks.obs["log_variances"] = np.log10(adata_peaks.obs["variances"])

# %%
sc.pl.umap(adata_peaks, color=["log_variances", "variances_norm"], 
           vmin=[2,2.5], vmax=[5, 4])

# %%
sc.pl.umap(adata_peaks, color="variances_norm", vmin=1, vmax=2.5, palette="magma")

# %%
sc.pl.umap(adata_peaks, color="highly_variable_peaks")

# %%
adata_peaks.obs["hvps_50K"] = False
mask = adata_peaks.obs_names.isin(adata_peaks_50K.obs_names)
adata_peaks.obs.loc[mask, "hvps_50K"] = True

# %%
sc.pl.umap(adata_peaks, color="hvps_50K", palette=['lightgrey', 'orange'],
           save="_highly_variable_peaks_50K.png")

# %% [markdown]
# ### We needed "normalized" layer for any downstream analyses (not "scaled" counts in .X)
# - for binarized counts, we can use either "sum" or "normalized', but for thresholding for the accessibility profiles, we need "normalized".

# %% [markdown]
# ## Import the adata_peaks_ct_tp

# %%
# import the master object
adata_peaks_ct_tp = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_master.h5ad")
adata_peaks_ct_tp

# %%
# check the number of cells/pseudobulk group distribution
adata_peaks_ct_tp.var["n_cells"].hist(bins=40)
plt.xlabel("number of cells (pseudobulk group")
plt.ylabel("occurences")
plt.grid(False)
# plt.xscale("log")
plt.savefig(figpath + "hist_n_cells_pseudobulk_groups_ct_tp.pdf")
plt.show()

# %% [markdown]
# ## Step 1. compute the averaged counts for each celltype / timepoint (using normalized counts)

# %%
# Create dictionaries to map columns to celltype and timepoint
celltype_mapping = {}
timepoint_mapping = {}

# parse var names
for col in adata_peaks_ct_tp.var.index:
    # Find the timepoint (ends with 'somites')
    timepoint_match = re.search(r'(\d+somites)$', col)
    if timepoint_match:
        timepoint = timepoint_match.group(1)
        celltype = col.replace(f'_{timepoint}', '')
        celltype_mapping[col] = celltype
        timepoint_mapping[col] = timepoint

# Get unique celltypes and timepoints
unique_celltypes = set(celltype_mapping.values())
unique_timepoints = set(timepoint_mapping.values())


# %%
# Function to get cell count for a group
def get_cell_count_for_group(col):
    cell_count_series = adata_peaks_ct_tp[:, col].var["n_cells"]
    return int(cell_count_series.iloc[0])  # Extract the integer value

# Set minimum cell threshold
MIN_CELLS = 20  # or 10 (based on the data)

# Filter out low-confidence groups
reliable_groups = []
filtered_out_groups = []

for col in adata_peaks_ct_tp.var.index:
    if col in celltype_mapping:  # Only process mapped columns
        cell_count = get_cell_count_for_group(col)
        if cell_count >= MIN_CELLS:
            reliable_groups.append(col)
        else:
            filtered_out_groups.append((col, cell_count))

print(f"Keeping {len(reliable_groups)}/{len(adata_peaks_ct_tp.var.index)} groups")
print(f"Filtered out {len(filtered_out_groups)} groups with <{MIN_CELLS} cells")

# %%
filtered_out_groups

# %%
# Get unique celltypes and timepoints from reliable groups only
reliable_celltypes = set()
reliable_timepoints = set()

for col in reliable_groups:
    if col in celltype_mapping:
        reliable_celltypes.add(celltype_mapping[col])
    if col in timepoint_mapping:
        reliable_timepoints.add(timepoint_mapping[col])

print(f"Reliable celltypes: {len(reliable_celltypes)}")
print(f"Reliable timepoints: {len(reliable_timepoints)}")

# Create new obs columns for each celltype (using only reliable groups)
for celltype in reliable_celltypes:
    # Get reliable columns for this celltype
    celltype_cols = [col for col, ct in celltype_mapping.items() 
                    if ct == celltype and col in reliable_groups]
    
    if celltype_cols:  # Only proceed if we have reliable data
        # Mean accessibility across timepoints for this celltype (equal weights)
        col_indices = [adata_peaks_ct_tp.var.index.get_loc(col) for col in celltype_cols]
        mean_accessibility = np.mean(adata_peaks_ct_tp.X[:, col_indices], axis=1)
        adata_peaks_ct_tp.obs[f'accessibility_{celltype}'] = mean_accessibility
        print(f"Created accessibility_{celltype} from {len(celltype_cols)} reliable groups")
    else:
        print(f"Warning: No reliable groups for celltype {celltype}")

# Create new obs columns for each timepoint (using only reliable groups)
for timepoint in reliable_timepoints:
    # Get reliable columns for this timepoint
    timepoint_cols = [col for col, tp in timepoint_mapping.items() 
                     if tp == timepoint and col in reliable_groups]
    
    if timepoint_cols:  # Only proceed if we have reliable data
        # Mean accessibility across celltypes for this timepoint (equal weights)
        col_indices = [adata_peaks_ct_tp.var.index.get_loc(col) for col in timepoint_cols]
        mean_accessibility = np.mean(adata_peaks_ct_tp.X[:, col_indices], axis=1)
        adata_peaks_ct_tp.obs[f'accessibility_{timepoint}'] = mean_accessibility
        print(f"Created accessibility_{timepoint} from {len(timepoint_cols)} reliable groups")
    else:
        print(f"Warning: No reliable groups for timepoint {timepoint}")

# %%
# remove the accessibility profiles for primordial_germ_cells, as they are all filtered out based on cells>=20cells threshold
del adata_peaks_ct_tp.obs["accessibility_primordial_germ_cells"]

# %% [markdown]
# ### compute the maximum celltype/timepoint for each peak (and also the peak_contrast)

# %%
# For timepoints
timepoint_cols = ['accessibility_0somites', 'accessibility_5somites', 'accessibility_10somites', 
                 'accessibility_15somites', 'accessibility_20somites', 'accessibility_30somites']
timepoint_vals = np.array([adata_peaks_ct_tp.obs[col] for col in timepoint_cols]).T

# Find max timepoint for each peak
max_timepoint_idx = np.argmax(timepoint_vals, axis=1)
timepoint_names = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
adata_peaks_ct_tp.obs['timepoint'] = [timepoint_names[i] for i in max_timepoint_idx]

# Calculate corrected timepoint contrast
max_vals = np.max(timepoint_vals, axis=1)
# Calculate mean and std excluding max value for each peak
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(timepoint_vals, max_timepoint_idx)])
# adata_peaks_ct_tp.obs['timepoint_contrast'] = (max_vals - other_vals_mean) / other_vals_std
adata_peaks_ct_tp.obs['timepoint_contrast'] = np.where(
    other_vals_std > 1e-10, 
    (max_vals - other_vals_mean) / other_vals_std, 
    0
)

# For celltypes
celltype_cols = [col for col in adata_peaks_ct_tp.obs.columns 
                if col.startswith('accessibility_') and 'somites' not in col]
celltype_vals = np.array([adata_peaks_ct_tp.obs[col] for col in celltype_cols]).T

# Find max celltype for each peak
max_celltype_idx = np.argmax(celltype_vals, axis=1)
celltype_names = [col.replace('accessibility_', '') for col in celltype_cols]
adata_peaks_ct_tp.obs['celltype'] = [celltype_names[i] for i in max_celltype_idx]

# Calculate corrected celltype contrast
max_vals = np.max(celltype_vals, axis=1)
other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                           for row, max_idx in zip(celltype_vals, max_celltype_idx)])
other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                          for row, max_idx in zip(celltype_vals, max_celltype_idx)])
# adata_peaks_ct_tp.obs['celltype_contrast'] = (max_vals - other_vals_mean) / other_vals_std
adata_peaks_ct_tp.obs['celltype_contrast'] = np.where(
    other_vals_std > 1e-10, 
    (max_vals - other_vals_mean) / other_vals_std, 
    0
)


# %%
# Normalize peak contrast for alpha values
def normalize_for_alpha_robust(values, min_alpha=0.1, max_alpha=0.9):
    min_val = np.percentile(values, 5)
    max_val = np.percentile(values, 95)
    clipped = np.clip(values, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    return normalized * (max_alpha - min_alpha) + min_alpha

# Create alpha values from peak contrast
adata_peaks_ct_tp.obs['alpha_timepoint'] = normalize_for_alpha_robust(adata_peaks_ct_tp.obs['timepoint_contrast'])
adata_peaks_ct_tp.obs['alpha_celltype'] = normalize_for_alpha_robust(adata_peaks_ct_tp.obs['celltype_contrast'])

# %%
for access_time in ['accessibility_0somites', 'accessibility_5somites', 'accessibility_10somites',
                                     'accessibility_15somites', 'accessibility_20somites', 'accessibility_30somites']:
    adata_peaks_ct_tp.obs["log_"+access_time] = np.log(1+adata_peaks_ct_tp.obs[access_time])
    
    

# %%
sc.pl.umap(adata_peaks_ct_tp, color=['log_accessibility_0somites', 'log_accessibility_5somites', 'log_accessibility_10somites',
                                     'log_accessibility_15somites', 'log_accessibility_20somites', 'log_accessibility_30somites'], 
           vmin=0, vmax=7,
           ncols=3, save="_log_access_timepoints.png"
            )

# %%
# generate individual figure panels
list_metadata = ['log_accessibility_0somites', 'log_accessibility_5somites', 'log_accessibility_10somites',
                                     'log_accessibility_15somites', 'log_accessibility_20somites', 'log_accessibility_30somites']

for metadata in list_metadata:
    sc.pl.umap(adata_peaks_ct_tp, color=metadata, 
               vmin=0, vmax=7,
               save=f"_{metadata}.png", show=False)

# %%
# plot for the celltype with the celltype color palette
# A module to define the color palettes used in this paper
import matplotlib.pyplot as plt
import seaborn as sns

# a color palette for the "coarse" grained celltype annotation ("annotation_ML_coarse")
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'differentiating_neurons': '#bebada',
    'endocrine_pancreas': '#fb8072',
    'endoderm': '#80b1d3',
    'enteric_neurons': '#fdb462',
    'epidermis': '#b3de69',
    'fast_muscle': '#df4b9b',
    'floor_plate': '#d9d9d9',
    'hatching_gland': '#bc80bd',
    'heart_myocardium': '#ccebc5',
    'hemangioblasts': '#ffed6f',
    'hematopoietic_vasculature': '#e41a1c',
    'hindbrain': '#377eb8',
    'lateral_plate_mesoderm': '#4daf4a',
    'midbrain_hindbrain_boundary': '#984ea3',
    'muscle': '#ff7f00',
    'neural': '#e6ab02',
    'neural_crest': '#a65628',
    'neural_floor_plate': '#66a61e',
    'neural_optic': '#999999',
    'neural_posterior': '#393b7f',
    'neural_telencephalon': '#fdcdac',
    'neurons': '#cbd5e8',
    'notochord': '#f4cae4',
    'optic_cup': '#c0c000',
    'pharyngeal_arches': '#fff2ae',
    'primordial_germ_cells': '#f1e2cc',
    'pronephros': '#cccccc',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}



# %%
sc.pl.umap(adata_peaks_ct_tp, color=["celltype"], palette=cell_type_color_dict,
           alpha=adata_peaks_ct_tp.obs["alpha_celltype"], save="_peaks_max_celltype_peak_contrast_exclude_pb_20cells.png")

# %%
# with pseudobulk >=20 cells threshold
adata_peaks_ct_tp.obs["celltype"].value_counts()

# %%
# with pseudobulk >=10 cells threshold
adata_peaks_ct_tp.obs["celltype"].value_counts()

# %%
# without any thresholding for pseudobulk groups
adata_peaks_ct_tp.obs["celltype"].value_counts()

# %%
# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))
adata_peaks_ct_tp.uns['timepoint_colors'] = [timepoint_colors[t] for t in timepoint_order]

# %%
sc.pl.umap(adata_peaks_ct_tp, color=["timepoint"], palette=timepoint_colors,
           alpha=adata_peaks_ct_tp.obs["alpha_timepoint"])#, save="_peaks_max_timepoints_peak_contrast_exclude_pb_20cells.png")

# %%
adata_peaks_ct_tp.obs.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/all_peaks_annotated_ct_tp.csv")

# %%
# lineage-level annotation
lineage_colors = {
    'CNS': '#DAA520',                    # Golden/orange
    'Endoderm': '#6A5ACD',              # Blue/purple  
    'Epiderm': '#DC143C',               # Red
    'Germline': '#DA70D6',              # Magenta/orchid
    'Lateral Mesoderm': '#228B22',      # Forest green
    'Neural Crest': '#20B2AA',          # Light sea green/teal
    'Paraxial Mesoderm': '#4169E1'      # Royal blue
}

# %%
# Updated celltype to lineage mapping based on your actual data
celltype_to_lineage = {
    "CNS": [
        "neural", 
        "neural_optic", 
        "neural_posterior", 
        "neural_telencephalon",
        "neurons",
        "hindbrain",
        "midbrain_hindbrain_boundary",
        "optic_cup",
        "spinal_cord",
        "differentiating_neurons",
        "floor_plate",
        "neural_floor_plate",
        "enteric_neurons",
    ],
    
    "Neural Crest": [
        "neural_crest"
    ],
    
    "Paraxial Mesoderm": [
        "somites",
        "fast_muscle",
        "muscle",
        "PSM",  # Presomitic mesoderm
        "NMPs",  # Neuromesodermal progenitors
        "tail_bud", 
        "notochord",
    ],
    
    "Lateral Mesoderm": [
        "lateral_plate_mesoderm",
        "heart_myocardium",
        "hematopoietic_vasculature",
        "pharyngeal_arches",
        "pronephros",
        "hemangioblasts",
        "hatching_gland",
    ],
    
    "Endoderm": [
        "endoderm",
        "endocrine_pancreas",
    ],
    
    "Epiderm": [
        "epidermis"
    ],
    
    "Germline": [
        "primordial_germ_cells"
    ],
}

# %%
# Create reverse mapping: celltype -> lineage
celltype_to_lineage_map = {}
for lineage, celltypes in celltype_to_lineage.items():
    for celltype in celltypes:
        celltype_to_lineage_map[celltype] = lineage

print(f"Celltype to lineage mapping created for {len(celltype_to_lineage_map)} celltypes")

# Get available celltype accessibility columns
available_celltype_cols = [col for col in adata_peaks_ct_tp.obs.columns 
                          if col.startswith('accessibility_') and 'somites' not in col]

print(f"Available celltype columns: {len(available_celltype_cols)}")

# Create lineage accessibility by aggregating celltypes
lineage_accessibility = {}

for lineage, celltypes_in_lineage in celltype_to_lineage.items():
    # Find which accessibility columns correspond to this lineage
    lineage_cols = []
    for celltype in celltypes_in_lineage:
        col_name = f'accessibility_{celltype}'
        if col_name in available_celltype_cols:
            lineage_cols.append(col_name)
    
    if lineage_cols:
        # Mean accessibility across all celltypes in this lineage
        col_indices = [adata_peaks_ct_tp.obs.columns.get_loc(col) for col in lineage_cols]
        lineage_vals = np.array([adata_peaks_ct_tp.obs[col] for col in lineage_cols])
        mean_lineage_accessibility = np.mean(lineage_vals, axis=0)
        
        # Store in the obs
        adata_peaks_ct_tp.obs[f'accessibility_lineage_{lineage}'] = mean_lineage_accessibility
        lineage_accessibility[lineage] = mean_lineage_accessibility
        
        print(f"Created accessibility_lineage_{lineage} from {len(lineage_cols)} celltypes: {[col.replace('accessibility_', '') for col in lineage_cols]}")
    else:
        print(f"Warning: No available celltypes for lineage {lineage}")

# Now compute max lineage and contrast using lineage-level accessibility
if len(lineage_accessibility) > 1:
    # Get lineage accessibility columns
    lineage_cols = [f'accessibility_lineage_{lineage}' for lineage in lineage_accessibility.keys()]
    lineage_vals = np.array([adata_peaks_ct_tp.obs[col] for col in lineage_cols]).T
    
    # Find max lineage for each peak
    max_lineage_idx = np.argmax(lineage_vals, axis=1)
    lineage_names = list(lineage_accessibility.keys())
    adata_peaks_ct_tp.obs['lineage'] = [lineage_names[i] for i in max_lineage_idx]
    
    # Calculate corrected lineage contrast
    max_vals = np.max(lineage_vals, axis=1)
    other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
                               for row, max_idx in zip(lineage_vals, max_lineage_idx)])
    other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
                              for row, max_idx in zip(lineage_vals, max_lineage_idx)])
    
    # Calculate contrast with protection against division by zero
    adata_peaks_ct_tp.obs['lineage_contrast'] = np.where(
        other_vals_std > 1e-10, 
        (max_vals - other_vals_mean) / other_vals_std, 
        0
    )
    
    print(f"Computed lineage annotations using {len(lineage_accessibility)} lineages")
    print(f"Lineage contrast stats: mean={adata_peaks_ct_tp.obs['lineage_contrast'].mean():.2f}, "
          f"std={adata_peaks_ct_tp.obs['lineage_contrast'].std():.2f}")
    
    # Show distribution of peak assignments
    lineage_counts = adata_peaks_ct_tp.obs['lineage'].value_counts()
    print("\nPeaks assigned to each lineage:")
    for lineage, count in lineage_counts.items():
        print(f"  {lineage}: {count} peaks")
        
else:
    print("Warning: Less than 2 lineages available, skipping lineage analysis")

# # Optional: Keep the original celltype analysis as well for comparison
# print("\n" + "="*50)
# print("Also computing original celltype-level analysis for comparison...")

# if len(available_celltype_cols) > 1:
#     celltype_vals = np.array([adata_peaks_ct_tp.obs[col] for col in available_celltype_cols]).T
    
#     # Find max celltype for each peak
#     max_celltype_idx = np.argmax(celltype_vals, axis=1)
#     celltype_names = [col.replace('accessibility_', '') for col in available_celltype_cols]
#     adata_peaks_ct_tp.obs['celltype'] = [celltype_names[i] for i in max_celltype_idx]
    
#     # Calculate corrected celltype contrast
#     max_vals = np.max(celltype_vals, axis=1)
#     other_vals_mean = np.array([np.mean(np.delete(row, max_idx)) 
#                                for row, max_idx in zip(celltype_vals, max_celltype_idx)])
#     other_vals_std = np.array([np.std(np.delete(row, max_idx)) 
#                               for row, max_idx in zip(celltype_vals, max_celltype_idx)])
    
#     adata_peaks_ct_tp.obs['celltype_contrast'] = np.where(
#         other_vals_std > 1e-10, 
#         (max_vals - other_vals_mean) / other_vals_std, 
#         0
#     )
    
#     print(f"Computed celltype annotations using {len(available_celltype_cols)} celltypes")

# %%
# Normalize peak contrast for alpha values
# Create alpha values from peak contrast
adata_peaks_ct_tp.obs['alpha_lineage'] = normalize_for_alpha_robust(adata_peaks_ct_tp.obs['lineage_contrast'])

# %%
adata_peaks_ct_tp.obs["lineage"].value_counts()

# %%
# color for the most accessible lineage/tissue
sc.pl.umap(adata_peaks_ct_tp, color=["lineage"], palette=lineage_colors,
           alpha=adata_peaks_ct_tp.obs["alpha_lineage"], save="_peaks_max_lineage_peak_contrast_exclude_pb_20cells.png")

# %%
# export the metadata as a csv file
adata_peaks_ct_tp.obs.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/all_peaks_annotated_ct_tp.csv")

# %% [markdown]
# ### chromosomes

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

def create_distinctive_chromosome_palette(n_chromosomes=25):
    """
    Create a highly distinctive color palette for chromosomes
    """
    
    # Combine multiple qualitative palettes for maximum distinction
    colors = []
    
    # Start with tab10 (10 distinct colors)
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors.extend(tab10_colors)
    
    # Add Set1 colors (9 distinct colors, avoid overlap)
    set1_colors = plt.cm.Set1(np.linspace(0, 1, 9))
    colors.extend(set1_colors)
    
    # Add some distinct colors from Set2 and Set3
    set2_colors = plt.cm.Set2(np.linspace(0, 1, 6))
    colors.extend(set2_colors[:6])  # Take first 6 to reach 25 total
    
    # Ensure we have exactly n_chromosomes colors
    if len(colors) > n_chromosomes:
        colors = colors[:n_chromosomes]
    elif len(colors) < n_chromosomes:
        # Add more colors if needed
        additional_colors = plt.cm.tab20c(np.linspace(0, 1, n_chromosomes - len(colors)))
        colors.extend(additional_colors)
    
    return colors[:n_chromosomes]

# Alternative: Custom high-contrast palette
def create_custom_chromosome_palette():
    """
    Create a custom high-contrast palette for 25 chromosomes
    """
    
    # Define 25 highly distinctive colors
    custom_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange  
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aec7e8',  # light blue
        '#ffbb78',  # light orange
        '#98df8a',  # light green
        '#ff9896',  # light red
        '#c5b0d5',  # light purple
        '#c49c94',  # light brown
        '#f7b6d3',  # light pink
        '#c7c7c7',  # light gray
        '#dbdb8d',  # light olive
        '#9edae5',  # light cyan
        '#393b79',  # dark blue
        '#637939',  # dark green
        '#8c6d31',  # dark orange
        '#843c39',  # dark red
        '#7b4173'   # dark purple
    ]
    
    return custom_colors

# Choose your preferred palette
chromosome_colors = create_distinctive_chromosome_palette()
# OR use custom palette:
# chromosome_colors = create_custom_chromosome_palette()

# %%
sc.pl.umap(adata_peaks_ct_tp, color=["chrom"], palette=chromosome_colors, alpha=0.5,
           save="_peaks_chromosomes.png")

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_circular_chromosome_palette(chromosome_colors, save_path=None, 
                                     figsize=(10, 6), circles_per_row=5):
    """
    Create a circular color palette display for chromosomes
    """
    
    n_chromosomes = len(chromosome_colors)
    n_rows = int(np.ceil(n_chromosomes / circles_per_row))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate circle positions
    circle_radius = 0.3
    spacing_x = 2.0
    spacing_y = 1.5
    
    positions = []
    labels = []
    
    for i, color in enumerate(chromosome_colors):
        row = i // circles_per_row
        col = i % circles_per_row
        
        # Center the circles in each row
        x_offset = (circles_per_row - 1) * spacing_x / 2
        x = col * spacing_x - x_offset
        y = -row * spacing_y
        
        # Create circle
        circle = patches.Circle((x, y), circle_radius, 
                              facecolor=color, 
                              edgecolor='black', 
                              linewidth=1.5)
        ax.add_patch(circle)
        
        # Add chromosome label
        ax.text(x, y, f'Chr\n{i+1}', 
               ha='center', va='center', 
               fontsize=9, fontweight='bold',
               color='white' if _is_dark_color(color) else 'black')
        
        positions.append((x, y))
        labels.append(f'Chr {i+1}')
    
    # Add expected uniform reference
    ref_y = -n_rows * spacing_y - 0.5
    ax.text(0, ref_y, 'Expected uniform (0.040)', 
           ha='center', va='center', 
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor='red', linestyle='--', linewidth=2))
    
    # Set axis limits and properties
    max_x = max([pos[0] for pos in positions]) + circle_radius + 0.5
    min_x = min([pos[0] for pos in positions]) - circle_radius - 0.5
    max_y = max([pos[1] for pos in positions]) + circle_radius + 0.5
    min_y = min([pos[1] for pos in positions]) - circle_radius - 1.0
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title('Chromosome Color Palette', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add border
    border = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                             linewidth=2, edgecolor='gray', facecolor='none',
                             linestyle='-', alpha=0.5)
    ax.add_patch(border)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Circular palette saved to: {save_path}")
    
    plt.show()
    return fig

def _is_dark_color(color):
    """Helper function to determine if a color is dark (for text contrast)"""
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    # Calculate luminance
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return luminance < 0.5

# Use your existing chromosome_colors or define them based on your image
chromosome_colors = [
    '#1f77b4',  # Chr 1 - blue
    '#ff7f0e',  # Chr 2 - orange
    '#2ca02c',  # Chr 3 - green
    '#d62728',  # Chr 4 - red
    '#9467bd',  # Chr 5 - purple
    '#8c564b',  # Chr 6 - brown
    '#e377c2',  # Chr 7 - pink
    '#7f7f7f',  # Chr 8 - gray
    '#bcbd22',  # Chr 9 - olive
    '#17becf',  # Chr 10 - cyan
    '#d62728',  # Chr 11 - red
    '#1f77b4',  # Chr 12 - blue
    '#2ca02c',  # Chr 13 - green
    '#9467bd',  # Chr 14 - purple
    '#ff7f0e',  # Chr 15 - orange
    '#ffff00',  # Chr 16 - yellow
    '#8c564b',  # Chr 17 - brown
    '#e377c2',  # Chr 18 - pink
    '#7f7f7f',  # Chr 19 - gray
    '#98df8a',  # Chr 20 - mint green
    '#ffbb78',  # Chr 21 - light orange
    '#f7b6d3',  # Chr 22 - light pink
    '#aec7e8',  # Chr 23 - light green
    '#dbdb8d',  # Chr 24 - tan
    '#c7c7c7'   # Chr 25 - light gray
]

# Create circular palette
fig_circles = create_circular_chromosome_palette(
    chromosome_colors, 
    save_path=figpath + 'chromosome_circles_palette.png',
    figsize=(12, 8),
    circles_per_row=5
)

# %%
from scipy.stats import hypergeom, fisher_exact
from statsmodels.stats.multitest import multipletests


def test_chromosome_enrichment(adata_peaks, cluster_col='leiden_coarse', chrom_col='chromosome'):
    """
    Test for chromosome enrichment in each cluster using hypergeometric test
    """
    
    # Create contingency data
    cluster_chrom = pd.crosstab(adata_peaks.obs[cluster_col], adata_peaks.obs[chrom_col])
    
    # Total peaks and peaks per chromosome
    total_peaks = len(adata_peaks)
    chrom_totals = adata_peaks.obs[chrom_col].value_counts()
    
    results = []
    
    for cluster in cluster_chrom.index:
        cluster_size = cluster_chrom.loc[cluster].sum()
        
        for chrom in cluster_chrom.columns:
            # Observed peaks from this chromosome in this cluster
            observed = cluster_chrom.loc[cluster, chrom]
            
            # Total peaks from this chromosome across all clusters
            chrom_total = chrom_totals[chrom]
            
            # Hypergeometric test
            # Population size: total_peaks
            # Successes in population: chrom_total  
            # Sample size: cluster_size
            # Observed successes: observed
            
            # P(X >= observed) - test for enrichment
            p_enriched = 1 - hypergeom.cdf(observed - 1, total_peaks, chrom_total, cluster_size)
            
            # P(X <= observed) - test for depletion  
            p_depleted = hypergeom.cdf(observed, total_peaks, chrom_total, cluster_size)
            
            # Expected count
            expected = (cluster_size * chrom_total) / total_peaks
            
            # Fold enrichment
            fold_enrichment = observed / expected if expected > 0 else np.inf
            
            results.append({
                'cluster': cluster,
                'chromosome': chrom,
                'observed': observed,
                'expected': expected,
                'fold_enrichment': fold_enrichment,
                'p_enriched': p_enriched,
                'p_depleted': p_depleted,
                'cluster_size': cluster_size,
                'chrom_total': chrom_total
            })
    
    return pd.DataFrame(results)

# Run the analysis
enrichment_results = test_chromosome_enrichment(adata_peaks_ct_tp, cluster_col='leiden_coarse', chrom_col='chrom')
enrichment_results

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_chromosome_distribution_stacked(adata_peaks, cluster_col='leiden_coarse', 
                                       chrom_col='chrom', normalize=True):
    """
    Create stacked bar plot showing chromosome distribution across clusters
    """
    
    # Create contingency table
    cluster_chrom_counts = pd.crosstab(adata_peaks.obs[cluster_col], 
                                      adata_peaks.obs[chrom_col])
    
    if normalize:
        # Convert to proportions (each row sums to 1)
        plot_data = cluster_chrom_counts.div(cluster_chrom_counts.sum(axis=1), axis=0)
        ylabel = 'Proportion of peaks'
        title_suffix = '(Normalized)'
    else:
        plot_data = cluster_chrom_counts
        ylabel = 'Number of peaks'
        title_suffix = '(Absolute counts)'
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create color palette - using tab20 for 25 chromosomes
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(plot_data.columns))))
    if len(plot_data.columns) > 20:
        # Add more colors for chromosomes 21-25
        extra_colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data.columns) - 20))
        colors = np.vstack([colors, extra_colors])
    
    # Create stacked bar plot
    bottom = np.zeros(len(plot_data.index))
    
    for i, chrom in enumerate(plot_data.columns):
        ax.bar(plot_data.index, plot_data[chrom], 
               bottom=bottom, label=f'Chr {chrom}', 
               color=colors[i % len(colors)], alpha=0.8)
        bottom += plot_data[chrom]
    
    ax.set_xlabel('Peak Cluster (Leiden)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Chromosome Distribution Across Peak Clusters {title_suffix}')
    
    # Add expected line for normalized plot
    if normalize:
        expected_prop = 1.0 / len(plot_data.columns)  # 1/25 = 0.04 for uniform
        ax.axhline(expected_prop, color='red', linestyle='--', alpha=0.7,
                   label=f'Expected uniform ({expected_prop:.3f})')
    
    # Customize legend
    if len(plot_data.columns) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, plot_data

# Create both normalized and absolute versions
fig_norm, data_norm = plot_chromosome_distribution_stacked(
    adata_peaks_ct_tp, normalize=True
)
plt.show()

fig_abs, data_abs = plot_chromosome_distribution_stacked(
    adata_peaks_ct_tp, normalize=False
)
plt.show()

# %% [markdown]
# ## Step 2. compute the "specificity" of metadata
#
# - check the "specificity" of metadata for each peak using "peak_contrast"

# %%
# plot the distribution of peak_contrast (celltypes)
plt.hist(adata_peaks_ct_tp.obs["celltype_contrast"], bins=50)
plt.xlabel("peak contrast (celltype)")
plt.ylabel("occurences")
plt.yscale("log")
plt.grid(False)
plt.savefig(figpath + "hist_peak_contrast_celltypes.pdf")
plt.show()

# %%
# plot the distribution of peak_contrast (timepoints)
plt.hist(adata_peaks_ct_tp.obs["timepoint_contrast"], bins=50)
plt.xlabel("peak contrast (timepoint)")
plt.ylabel("occurences")
plt.yscale("log")
plt.grid(False)
plt.savefig(figpath + "hist_peak_contrast_timepoint.pdf")
plt.show()

# %%
# # Calculate key percentiles
# contrast_values = adata_peaks_ct_tp.obs['celltype_contrast'].dropna()

# print("Contrast percentiles:")
# for pct in [50, 75, 90, 95, 99]:
#     val = np.percentile(contrast_values, pct)
#     count = np.sum(contrast_values >= val)
#     print(f"{pct}th percentile: {val:.1f} ({count:,} peaks, {count/len(contrast_values)*100:.1f}%)")

# %%
# # Classify into 3 categories
# adata_peaks_ct_tp.obs['celltype_specificity'] = 'broad'
# adata_peaks_ct_tp.obs.loc[contrast_values >= np.percentile(contrast_values, 75), 'celltype_specificity'] = 'moderate'  
# adata_peaks_ct_tp.obs.loc[contrast_values >= np.percentile(contrast_values, 90), 'celltype_specificity'] = 'specific'
# 
# # using the percentile-based approach
# sc.pl.umap(adata_peaks_ct_tp, color="celltype_specificity")

# %%
# Biologically meaningful z-score thresholds
adata_peaks_ct_tp.obs['celltype_specificity'] = 'broad'  # z < 2
adata_peaks_ct_tp.obs.loc[adata_peaks_ct_tp.obs['celltype_contrast'] >= 2, 'celltype_specificity'] = 'moderate'  # 2 ≤ z < 4  
adata_peaks_ct_tp.obs.loc[adata_peaks_ct_tp.obs['celltype_contrast'] >= 4, 'celltype_specificity'] = 'specific'  # z ≥ 4

# %%
specificity_colors = {
    'broad': '#D3D3D3',      # Light grey (neutral/background)
    'moderate': '#87CEEB',    # Sky blue (moderate intensity)  
    'specific': '#191970'     # Midnight blue (strong/specific)
}
# specificity_colors = {
#     'broad': '#D3D3D3',      # Light grey
#     'moderate': '#32CD32',    # Lime green
#     'specific': '#006400'     # Dark green
# }

# %%
sc.pl.umap(adata_peaks_ct_tp, color="celltype_specificity", palette=specificity_colors,
           save="_celltype_specificity.png")

# %%

# %% [markdown]
# ## Step 3. compute the cluster-specific statistics
#
# - Average the accessibility profiles over (1) peak clusters ("leiden_coarse"), and (2) metadata (celltype or timepoints)
# - Compute the entropy to determine the 'broad" vs "specific" patterns of enrichment
#

# %%
"""
Fresh Cluster Entropy Analysis Module

Clean implementation for analyzing peak cluster accessibility patterns.

Workflow:
1. Create cluster-by-pseudobulk profiles (filter groups with <20 cells)
2. Aggregate by metadata type (celltype, timepoint, lineage) 
3. Compute entropy to classify broad vs specific accessibility patterns

Author: Yang-Joon Kim
Date: 2025-08-14
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional


def create_cluster_pseudobulk_profiles(adata, cluster_col='leiden_coarse', min_cells=20, verbose=True):
    """
    Step 1: Create cluster-by-pseudobulk profiles, filtering groups with <min_cells.
    
    Parameters:
    -----------
    adata : AnnData
        Data object with peaks as obs and pseudobulk groups as var
    cluster_col : str
        Column name for cluster assignments
    min_cells : int
        Minimum cells required per pseudobulk group
    verbose : bool
        Whether to print filtering information
        
    Returns:
    --------
    pd.DataFrame
        cluster-by-pseudobulk_groups matrix with reliable groups only
    """
    
    if verbose:
        print("="*80)
        print("STEP 1: CREATE CLUSTER-BY-PSEUDOBULK PROFILES")
        print("="*80)
    
    # Function to get cell count for a pseudobulk group
    def get_cell_count_for_group(col):
        cell_count_series = adata[:, col].var["n_cells"]
        return int(cell_count_series.iloc[0])
    
    # Filter reliable groups (≥min_cells)
    reliable_groups = []
    filtered_out_groups = []
    
    for col in adata.var.index:
        cell_count = get_cell_count_for_group(col)
        if cell_count >= min_cells:
            reliable_groups.append(col)
        else:
            filtered_out_groups.append((col, cell_count))
    
    if verbose:
        print(f"Keeping {len(reliable_groups)}/{len(adata.var.index)} groups with ≥{min_cells} cells")
        print(f"Filtered out {len(filtered_out_groups)} groups")
        if filtered_out_groups:
            print("Examples of filtered groups:")
            for col, count in filtered_out_groups[:5]:
                print(f"  {col}: {count} cells")
    
    # Create cluster profiles using only reliable groups
    clusters = adata.obs[cluster_col].astype(str)
    
    # Get data for reliable groups only
    reliable_indices = [adata.var.index.get_loc(col) for col in reliable_groups]
    reliable_data = adata.X[:, reliable_indices]
    
    # Create DataFrame for easier groupby
    data_df = pd.DataFrame(
        reliable_data.toarray() if hasattr(reliable_data, 'toarray') else reliable_data,
        columns=reliable_groups,
        index=adata.obs.index
    )
    data_df['cluster'] = clusters
    
    # Aggregate by cluster (mean accessibility across peaks in each cluster)
    cluster_profiles = data_df.groupby('cluster').mean()
    
    if verbose:
        print(f"Created cluster profiles: {cluster_profiles.shape}")
        print(f"Clusters: {sorted(cluster_profiles.index)}")
        print(f"Reliable pseudobulk groups: {len(reliable_groups)}")
    
    return cluster_profiles


def parse_pseudobulk_groups(reliable_groups, verbose=True):
    """
    Parse pseudobulk group names to extract celltype and timepoint mappings.
    
    Assumes format: celltype_timepoint (e.g., "neural_15somites")
    
    Parameters:
    -----------
    reliable_groups : List[str]
        List of reliable pseudobulk group names
    verbose : bool
        Whether to print parsing information
        
    Returns:
    --------
    tuple
        (celltype_mapping, timepoint_mapping, reliable_celltypes, reliable_timepoints)
    """
    
    if verbose:
        print("\n" + "="*60)
        print("PARSING PSEUDOBULK GROUP NAMES")
        print("="*60)
    
    celltype_mapping = {}
    timepoint_mapping = {}
    
    # Parse group names
    for col in reliable_groups:
        # Find the timepoint (ends with 'somites')
        timepoint_match = re.search(r'(\d+somites)$', col)
        if timepoint_match:
            timepoint = timepoint_match.group(1)
            celltype = col.replace(f'_{timepoint}', '')
            celltype_mapping[col] = celltype
            timepoint_mapping[col] = timepoint
    
    # Get unique celltypes and timepoints
    reliable_celltypes = set(celltype_mapping.values())
    reliable_timepoints = set(timepoint_mapping.values())
    
    if verbose:
        print(f"Parsed {len(celltype_mapping)} pseudobulk groups")
        print(f"Found {len(reliable_celltypes)} reliable celltypes: {sorted(reliable_celltypes)}")
        print(f"Found {len(reliable_timepoints)} reliable timepoints: {sorted(reliable_timepoints)}")
    
    return celltype_mapping, timepoint_mapping, reliable_celltypes, reliable_timepoints


def aggregate_by_metadata(cluster_profiles, metadata_type='celltype', 
                         celltype_mapping=None, timepoint_mapping=None,
                         lineage_mapping=None, verbose=True):
    """
    Step 2: Aggregate cluster profiles by metadata type (celltype, timepoint, or lineage) - average over the other metadata.
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        cluster-by-pseudobulk_groups matrix from step 1
    metadata_type : str
        'celltype', 'timepoint', or 'lineage'
    celltype_mapping, timepoint_mapping, lineage_mapping : dict
        Mappings from pseudobulk group to metadata category
    verbose : bool
        Whether to print aggregation information
        
    Returns:
    --------
    pd.DataFrame
        cluster-by-metadata matrix (e.g., cluster-by-celltype)
    """
    
    if verbose:
        print(f"\n" + "="*60)
        print(f"STEP 2: AGGREGATE BY {metadata_type.upper()}")
        print("="*60)
    
    if metadata_type == 'celltype':
        mapping = celltype_mapping
        reliable_categories = set(mapping.values())
    elif metadata_type == 'timepoint':
        mapping = timepoint_mapping  
        reliable_categories = set(mapping.values())
    elif metadata_type == 'lineage':
        mapping = lineage_mapping
        reliable_categories = set(mapping.values())
    else:
        raise ValueError(f"metadata_type must be 'celltype', 'timepoint', or 'lineage', got {metadata_type}")
    
    if mapping is None:
        raise ValueError(f"No mapping provided for {metadata_type}")
    
    if verbose:
        print(f"Aggregating across {len(reliable_categories)} {metadata_type}s")
    
    # Create cluster-by-metadata matrix
    cluster_metadata_profiles = pd.DataFrame(index=cluster_profiles.index)
    
    for category in reliable_categories:
        # Get pseudobulk groups for this category
        category_cols = [col for col, cat in mapping.items() 
                        if cat == category and col in cluster_profiles.columns]
        
        if category_cols:
            # Mean accessibility across all pseudobulk groups for this category
            mean_accessibility = cluster_profiles[category_cols].mean(axis=1)
            cluster_metadata_profiles[category] = mean_accessibility
            
            if verbose:
                print(f"  {category:25s}: averaged {len(category_cols):2d} groups")
        else:
            if verbose:
                print(f"  {category:25s}: no reliable groups found")
    
    if verbose:
        print(f"\nCreated {cluster_metadata_profiles.shape} cluster-by-{metadata_type} matrix")
        print(f"Sample values:")
        print(cluster_metadata_profiles.iloc[:3, :5])
    
    return cluster_metadata_profiles


def compute_accessibility_entropy(values, normalize=True, min_value=0.0):
    """
    Compute Shannon entropy of accessibility values using proper probability definition.
    
    Entropy = -Σ(p_i * log2(p_i))
    where p_i = accessibility_i / sum(all_accessibility)
    
    Parameters:
    -----------
    values : array-like
        Accessibility values across categories
    normalize : bool
        Whether to normalize by maximum possible entropy
    min_value : float
        Minimum value to include (use 0.0 to include all non-zero values)
        
    Returns:
    --------
    float
        Entropy score (0-1 if normalized)
    """
    
    # Convert to numpy array
    vals = np.array(values)
    
    # Filter out negative and very small values if specified
    if min_value > 0:
        vals = vals[vals >= min_value]
    else:
        vals = vals[vals > 0]  # Remove only true zeros
    
    if len(vals) <= 1:
        return 0.0
    
    # Compute probabilities (proper definition)
    total_accessibility = vals.sum()
    if total_accessibility == 0:
        return 0.0
        
    probabilities = vals / total_accessibility
    
    # Calculate Shannon entropy: -Σ(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    if normalize:
        max_entropy = np.log2(len(probabilities))
        entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return float(entropy)


def analyze_cluster_accessibility_patterns(cluster_metadata_profiles, metadata_type='celltype',
                                         # Classification thresholds
                                         broad_entropy_threshold=0.75,
                                         specific_dominance_threshold=0.6,
                                         specific_entropy_threshold=0.4,
                                         moderate_dominance_threshold=0.4,
                                         verbose=True):
    """
    Step 3: Analyze accessibility patterns using entropy and dominance metrics.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-metadata matrix (e.g., cluster-by-celltype)
    metadata_type : str
        Type of metadata being analyzed
    broad_entropy_threshold : float
        Entropy threshold for broad accessibility
    specific_dominance_threshold : float
        Dominance threshold for specific accessibility
    specific_entropy_threshold : float
        Entropy threshold for specific accessibility
    moderate_dominance_threshold : float
        Dominance threshold for moderate accessibility
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    pd.DataFrame
        Results with cluster annotations and metrics
    """
    
    if verbose:
        print(f"\n" + "="*60)
        print(f"STEP 3: ANALYZE {metadata_type.upper()} ACCESSIBILITY PATTERNS")
        print("="*60)
    
    results = []
    
    for cluster in cluster_metadata_profiles.index:
        cluster_profile = cluster_metadata_profiles.loc[cluster]
        
        # NO FILTERING - use all accessibility values as is
        # This preserves the true accessibility distribution
        
        # Compute metrics using the proper entropy definition
        entropy = compute_accessibility_entropy(cluster_profile, normalize=True, min_value=0.0)
        
        # Dominance = proportion of total accessibility in dominant category
        total_accessibility = cluster_profile.sum()
        if total_accessibility > 0:
            dominance = cluster_profile.max() / total_accessibility
            dominant_category = cluster_profile.idxmax()
        else:
            dominance = 0.0
            dominant_category = "unclear"
        
        n_categories = len(cluster_profile[cluster_profile > 0])  # Count non-zero categories
        
        # Coverage metrics (based on proportions of total accessibility)
        if total_accessibility > 0:
            proportions = cluster_profile / total_accessibility
            coverage_5pct = (proportions >= 0.05).sum()
            coverage_10pct = (proportions >= 0.10).sum()
            coverage_20pct = (proportions >= 0.20).sum()
        else:
            coverage_5pct = coverage_10pct = coverage_20pct = 0
        
        # REVISED CLASSIFICATION with realistic thresholds for your data
        if entropy >= 0.95:  # Very high entropy = truly uniform
            pattern = "broadly_accessible"
            confidence = "high" if entropy >= 0.98 else "medium"
            
        elif dominance >= 0.25:  # >25% in one category = enriched
            if entropy <= 0.90:  # Low entropy + high dominance = specific
                pattern = f"specific_{dominant_category}"
                confidence = "high" if dominance >= 0.30 else "medium"
            else:  # High entropy but some dominance = enriched
                pattern = f"enriched_{dominant_category}"
                confidence = "medium" if dominance >= 0.20 else "low"
                
        elif dominance >= 0.15:  # 15-25% = moderate enrichment
            pattern = f"enriched_{dominant_category}"
            confidence = "low"
            
        else:  # Very uniform distribution
            pattern = "broadly_accessible"
            confidence = "medium"
        
        if verbose:
            print(f"Cluster {cluster:2s}: {pattern:30s} "
                  f"(entropy={entropy:.3f}, dominance={dominance:.3f}, "
                  f"categories={n_categories})")
        
        # Store results
        results.append({
            'cluster': cluster,
            'pattern': pattern,
            'confidence': confidence,
            'entropy': entropy,
            'dominance': dominance,
            'dominant_category': dominant_category,
            'n_categories': n_categories,
            'coverage_5pct': coverage_5pct,
            'coverage_10pct': coverage_10pct,
            'coverage_20pct': coverage_20pct,
            'mean_accessibility': total_accessibility,
            'top_3_categories': dict(cluster_profile.nlargest(3))
        })
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n{metadata_type.upper()} PATTERN SUMMARY:")
        pattern_counts = results_df['pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"  {pattern:30s}: {count:2d} clusters")
    
    return results_df


def run_metadata_entropy_analysis(adata, cluster_col='leiden_coarse', 
                                 metadata_type='celltype', min_cells=20,
                                 lineage_mapping=None, verbose=True):
    """
    Complete workflow for entropy analysis of a specific metadata type.
    
    Parameters:
    -----------
    adata : AnnData
        Data object with peaks and cluster assignments
    cluster_col : str
        Column name for cluster assignments
    metadata_type : str
        'celltype', 'timepoint', or 'lineage'
    min_cells : int
        Minimum cells per pseudobulk group
    lineage_mapping : dict, optional
        Mapping from celltype to lineage (only needed for lineage analysis)
    verbose : bool
        Whether to print detailed progress
        
    Returns:
    --------
    pd.DataFrame
        Analysis results with cluster patterns and metrics
    """
    
    if verbose:
        print(f"RUNNING {metadata_type.upper()} ENTROPY ANALYSIS")
        print("="*80)
    
    # Step 1: Create cluster pseudobulk profiles
    cluster_profiles = create_cluster_pseudobulk_profiles(
        adata, cluster_col, min_cells, verbose
    )
    
    # Step 2: Parse group names and get mappings
    reliable_groups = list(cluster_profiles.columns)
    celltype_mapping, timepoint_mapping, reliable_celltypes, reliable_timepoints = parse_pseudobulk_groups(
        reliable_groups, verbose
    )
    
    # Step 3: Aggregate by metadata type
    if metadata_type == 'lineage' and lineage_mapping is not None:
        # Create lineage mapping from celltype mapping
        lineage_to_groups = {}
        for group, celltype in celltype_mapping.items():
            for lineage, celltypes_in_lineage in lineage_mapping.items():
                if celltype in celltypes_in_lineage:
                    if lineage not in lineage_to_groups:
                        lineage_to_groups[lineage] = []
                    lineage_to_groups[lineage].append(group)
                    break
        
        # Create lineage aggregation mapping
        lineage_group_mapping = {}
        for lineage, groups in lineage_to_groups.items():
            for group in groups:
                lineage_group_mapping[group] = lineage
        
        cluster_metadata_profiles = aggregate_by_metadata(
            cluster_profiles, 'lineage', 
            lineage_mapping=lineage_group_mapping,
            verbose=verbose
        )
    else:
        cluster_metadata_profiles = aggregate_by_metadata(
            cluster_profiles, metadata_type,
            celltype_mapping=celltype_mapping,
            timepoint_mapping=timepoint_mapping,
            verbose=verbose
        )
    
    # Step 4: Analyze patterns
    results = analyze_cluster_accessibility_patterns(
        cluster_metadata_profiles, metadata_type, verbose=verbose
    )
    
    return results


def parse_pseudobulk_groups(reliable_groups, verbose=True):
    """
    Parse pseudobulk group names to extract celltype and timepoint.
    """
    
    if verbose:
        print(f"\nParsing {len(reliable_groups)} reliable pseudobulk groups...")
    
    celltype_mapping = {}
    timepoint_mapping = {}
    
    # Parse group names  
    for col in reliable_groups:
        # Find the timepoint (ends with 'somites')
        timepoint_match = re.search(r'(\d+somites)$', col)
        if timepoint_match:
            timepoint = timepoint_match.group(1)
            celltype = col.replace(f'_{timepoint}', '')
            celltype_mapping[col] = celltype
            timepoint_mapping[col] = timepoint
    
    # Get unique categories
    reliable_celltypes = set(celltype_mapping.values())
    reliable_timepoints = set(timepoint_mapping.values())
    
    if verbose:
        print(f"Found {len(reliable_celltypes)} celltypes: {sorted(reliable_celltypes)}")
        print(f"Found {len(reliable_timepoints)} timepoints: {sorted(reliable_timepoints)}")
    
    return celltype_mapping, timepoint_mapping, reliable_celltypes, reliable_timepoints


def aggregate_by_metadata(cluster_profiles, metadata_type, 
                         celltype_mapping=None, timepoint_mapping=None, lineage_mapping=None,
                         verbose=True):
    """
    Aggregate cluster profiles by metadata type.
    """
    
    if metadata_type == 'celltype':
        mapping = celltype_mapping
        reliable_categories = set(mapping.values())
    elif metadata_type == 'timepoint':
        mapping = timepoint_mapping
        reliable_categories = set(mapping.values())
    elif metadata_type == 'lineage':
        mapping = lineage_mapping
        reliable_categories = set(mapping.values())
    else:
        raise ValueError(f"metadata_type must be 'celltype', 'timepoint', or 'lineage'")
    
    if verbose:
        print(f"Aggregating by {metadata_type} across {len(reliable_categories)} categories")
    
    # Create cluster-by-metadata matrix
    cluster_metadata_profiles = pd.DataFrame(index=cluster_profiles.index)
    
    for category in reliable_categories:
        # Get columns for this category
        category_cols = [col for col, cat in mapping.items() 
                        if cat == category and col in cluster_profiles.columns]
        
        if category_cols:
            # Mean accessibility across timepoints/celltypes for this category
            mean_accessibility = cluster_profiles[category_cols].mean(axis=1)
            cluster_metadata_profiles[category] = mean_accessibility
            
            if verbose:
                print(f"  {category:25s}: {len(category_cols):2d} groups → mean accessibility")
        else:
            if verbose:
                print(f"  {category:25s}: no reliable groups")
    
    if verbose:
        print(f"\nCreated {cluster_metadata_profiles.shape} cluster-by-{metadata_type} matrix")
    
    return cluster_metadata_profiles


def validate_cluster_23_entropy(adata, cluster_col='leiden_coarse', min_cells=20):
    """
    Test the entropy analysis specifically on Cluster 23.
    """
    
    print("="*80)
    print("CLUSTER 23 ENTROPY VALIDATION")
    print("="*80)
    
    # Run celltype analysis
    print("Testing CELLTYPE entropy for Cluster 23...")
    celltype_results = run_metadata_entropy_analysis(
        adata, cluster_col, 'celltype', min_cells, verbose=False
    )
    
    # Run timepoint analysis
    print("\nTesting TIMEPOINT entropy for Cluster 23...")
    timepoint_results = run_metadata_entropy_analysis(
        adata, cluster_col, 'timepoint', min_cells, verbose=False
    )
    
    # Check Cluster 23 results
    cluster_23_ct = celltype_results[celltype_results['cluster'] == '23']
    cluster_23_tp = timepoint_results[timepoint_results['cluster'] == '23']
    
    print(f"\nCluster 23 Results:")
    print("-" * 40)
    
    if len(cluster_23_ct) > 0:
        ct_row = cluster_23_ct.iloc[0]
        print(f"Celltype:")
        print(f"  Pattern: {ct_row['pattern']}")
        print(f"  Confidence: {ct_row['confidence']}")
        print(f"  Entropy: {ct_row['entropy']:.3f}")
        print(f"  Dominance: {ct_row['dominance']:.3f}")
        print(f"  Categories: {ct_row['n_categories']}")
        
        ct_pass = "✓" if ct_row['pattern'] == 'broadly_accessible' else "✗"
        print(f"  Expected broadly_accessible: {ct_pass}")
    else:
        print("Celltype: Cluster 23 not found")
    
    if len(cluster_23_tp) > 0:
        tp_row = cluster_23_tp.iloc[0]
        print(f"\nTimepoint:")
        print(f"  Pattern: {tp_row['pattern']}")
        print(f"  Confidence: {tp_row['confidence']}")
        print(f"  Entropy: {tp_row['entropy']:.3f}")
        print(f"  Dominance: {tp_row['dominance']:.3f}")
        print(f"  Categories: {tp_row['n_categories']}")
        
        tp_pass = "✓" if tp_row['pattern'] == 'broadly_accessible' else "✗"
        print(f"  Expected broadly_accessible: {tp_pass}")
    else:
        print("Timepoint: Cluster 23 not found")
    
    return celltype_results, timepoint_results


# =============================================================================
# LINEAGE MAPPING (for optional lineage analysis)
# =============================================================================

DEFAULT_LINEAGE_MAPPING = {
    "CNS": [
        "neural", 
        "neural_optic", 
        "neural_posterior", 
        "neural_telencephalon",
        "neurons",
        "hindbrain",
        "midbrain_hindbrain_boundary",
        "optic_cup",
        "spinal_cord",
        "differentiating_neurons",
        "floor_plate",
        "neural_floor_plate",
        "enteric_neurons",
    ],
    
    "Neural Crest": [
        "neural_crest"
    ],
    
    "Paraxial Mesoderm": [
        "somites",
        "fast_muscle",
        "muscle",
        "PSM",
        "NMPs",
        "tail_bud", 
        "notochord",
    ],
    
    "Lateral Mesoderm": [
        "lateral_plate_mesoderm",
        "heart_myocardium",
        "hematopoietic_vasculature",
        "pharyngeal_arches",
        "pronephros",
        "hemangioblasts",
        "hatching_gland",
    ],
    
    "Endoderm": [
        "endoderm",
        "endocrine_pancreas",
    ],
    
    "Epiderm": [
        "epidermis"
    ],
    
    "Germline": [
        "primordial_germ_cells"
    ],
}


def plot_cluster_accessibility_profiles(cluster_metadata_profiles, clusters_to_plot=None, 
                                       metadata_type='celltype', figsize=(15, 10), 
                                       celltype_order=None, save_name=None):
    """
    Plot accessibility profiles for specific clusters (minimal, clean version).
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-metadata matrix (e.g., cluster-by-celltype)
    clusters_to_plot : list or None
        List of cluster IDs to plot. If None, plots first 6 clusters.
    metadata_type : str
        Type of metadata being plotted
    figsize : tuple
        Figure size
    celltype_order : list or None
        Optional order for celltypes on x-axis
    save_name : str or None
        Filename to save plot
    """
    
    import matplotlib.pyplot as plt
    
    if clusters_to_plot is None:
        clusters_to_plot = list(cluster_metadata_profiles.index)[:6]
    
    # Ensure clusters exist in data
    clusters_to_plot = [c for c in clusters_to_plot if c in cluster_metadata_profiles.index]
    
    if len(clusters_to_plot) == 0:
        print("No valid clusters found to plot")
        return
    
    # Compute median accessibility for y-axis scaling
    median_accessibility = cluster_metadata_profiles.values.flatten()
    median_accessibility = np.median(median_accessibility[median_accessibility > 0])
    
    # Create subplots
    n_clusters = len(clusters_to_plot)
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, cluster in enumerate(clusters_to_plot):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get accessibility profile for this cluster
        profile = cluster_metadata_profiles.loc[cluster]
        
        # Order celltypes if specified, otherwise sort by accessibility
        if celltype_order is not None:
            # Use specified order, fill missing with 0
            ordered_profile = pd.Series(index=celltype_order, dtype=float)
            for celltype in celltype_order:
                if celltype in profile.index:
                    ordered_profile[celltype] = profile[celltype]
                else:
                    ordered_profile[celltype] = 0.0
            profile_to_plot = ordered_profile
        else:
            # Sort by accessibility (descending)
            profile_to_plot = profile.sort_values(ascending=False)
        
        # Create simple bar plot (single color, no grid)
        ax.bar(range(len(profile_to_plot)), profile_to_plot.values, 
               color='steelblue', alpha=0.7)
        
        # Compute entropy for title
        entropy = compute_accessibility_entropy(profile, normalize=True)
        
        # Determine y-axis limit
        max_accessibility = profile_to_plot.max()
        if max_accessibility > median_accessibility * 3:  # High accessibility cluster
            y_limit = max_accessibility * 1.1
        else:  # Normal/low accessibility cluster
            y_limit = median_accessibility * 2
        
        # Formatting
        ax.set_title(f'Cluster {cluster} (entropy={entropy:.3f})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{metadata_type.capitalize()}', fontsize=10)
        ax.set_ylabel('Mean Accessibility', fontsize=10)
        ax.set_ylim(0, y_limit)
        
        # Set x-tick labels
        # ax.set_xticks(range(len(profile_to_plot)))
        # ax.set_xticklabels(profile_to_plot.index, rotation=45, ha='right', fontsize=8)
        ax.set_xticklabels([])
        
        # Remove grid
        ax.grid(False)
        
        # Add value labels on highest bars only
        for j, (cat, val) in enumerate(profile_to_plot.head(3).items()):
            if val > median_accessibility * 0.5:  # Only label significant bars
                ax.text(j, val + y_limit*0.02, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(clusters_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_name}")
    
    plt.show()


def plot_single_cluster(cluster_metadata_profiles, cluster_id, 
                       celltype_order=None, figsize=(12, 6), save_path=None,
                       use_celltype_colors=True, cell_type_color_dict=None):
    """
    Plot accessibility profile for a single cluster with save option.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-celltype matrix
    cluster_id : str
        Cluster ID to plot
    celltype_order : list or None
        Optional order for celltypes
    figsize : tuple
        Figure size
    save_path : str or None
        Full path to save the figure (e.g., '/path/to/cluster_25_profile.png')
    """
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    if cluster_id not in cluster_metadata_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return
    
    # Get profile
    profile = cluster_metadata_profiles.loc[cluster_id]
    
    # Order celltypes
    if celltype_order is not None:
        ordered_profile = pd.Series(index=celltype_order, dtype=float)
        for celltype in celltype_order:
            ordered_profile[celltype] = profile.get(celltype, 0.0)
        profile_to_plot = ordered_profile
    else:
        profile_to_plot = profile.sort_values(ascending=False)
    
    # Compute y-limit based on median of all profiles vs current profile max
    overall_median = cluster_metadata_profiles.values.flatten()
    overall_median = pd.Series(overall_median).median()
    profile_max = profile_to_plot.max()
    y_limit = max(overall_median, profile_max) + 10
    
    # Determine colors
    if use_celltype_colors and cell_type_color_dict is not None:
        colors = [cell_type_color_dict.get(celltype, 'steelblue') 
                 for celltype in profile_to_plot.index]
    else:
        colors = 'steelblue'
    
    # Simple bar plot
    plt.figure(figsize=figsize)
    plt.bar(range(len(profile_to_plot)), profile_to_plot.values, 
            color=colors, alpha=0.7, edgecolor="none")
    
    # Set y-limit
    plt.ylim(0, y_limit)
    # plot the horizontal line for the median value
    plt.axhline(y=overall_median, linestyle="--")
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_accessibility_metrics(profile)
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    cv = metrics['cv']
    
    plt.title(f'Cluster {cluster_id} Celltype Accessibility\n'
              f'entropy={entropy:.3f}, dominance={dominance:.3f}, CV={cv:.1f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Celltype', fontsize=12)
    plt.ylabel('Mean Accessibility', fontsize=12)
    
    plt.xticks(range(len(profile_to_plot)), profile_to_plot.index, 
               rotation=45, ha='right')
    plt.grid(False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_single_cluster_subplot(cluster_metadata_profiles, cluster_id, ax, 
                               celltype_order=None, use_celltype_colors=True, 
                               cell_type_color_dict=None):
    """
    Plot accessibility profile for a single cluster on a given axis (for subplots).
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-celltype matrix
    cluster_id : str
        Cluster ID to plot
    ax : matplotlib axis
        Axis to plot on
    celltype_order : list or None
        Optional order for celltypes
    use_celltype_colors : bool
        Whether to use celltype-specific colors
    cell_type_color_dict : dict or None
        Dictionary mapping celltypes to colors
    """
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    if cluster_id not in cluster_metadata_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return
    
    # Get profile
    profile = cluster_metadata_profiles.loc[cluster_id]
    
    # Order celltypes
    if celltype_order is not None:
        ordered_profile = pd.Series(index=celltype_order, dtype=float)
        for celltype in celltype_order:
            ordered_profile[celltype] = profile.get(celltype, 0.0)
        profile_to_plot = ordered_profile
    else:
        profile_to_plot = profile.sort_values(ascending=False)
    
    # Compute y-limit based on median of all profiles vs current profile max
    overall_median = cluster_metadata_profiles.values.flatten()
    overall_median = pd.Series(overall_median).median()
    profile_max = profile_to_plot.max()
    y_limit = max(overall_median, profile_max) + 10
    
    # Determine colors
    if use_celltype_colors and cell_type_color_dict is not None:
        colors = [cell_type_color_dict.get(celltype, 'steelblue') 
                 for celltype in profile_to_plot.index]
    else:
        colors = 'steelblue'
    
    # Simple bar plot on the given axis
    ax.bar(range(len(profile_to_plot)), profile_to_plot.values, 
           color=colors, alpha=0.7, edgecolor="none")
    
    # Set y-limit
    ax.set_ylim(0, y_limit)
    # plot the horizontal line for the median value
    ax.axhline(y=overall_median, linestyle="--", alpha=0.5)
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_accessibility_metrics(profile)
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    cv = metrics['cv']
    
    ax.set_title(f'Cluster {cluster_id}\nent={entropy:.2f}, dom={dominance:.2f}, CV={cv:.1f}', 
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Celltype', fontsize=8)
    ax.set_ylabel('Mean Accessibility', fontsize=8)
    
    # ax.set_xticks(range(len(profile_to_plot)))
    # ax.set_xticklabels(profile_to_plot.index, rotation=45, ha='right', fontsize=6)
    ax.set_xticklabels([])
    ax.grid(False)


def sort_cluster_ids_numerically(cluster_ids):
    """
    Sort cluster IDs numerically instead of lexicographically.
    Handles patterns like 'cluster_0', 'cluster_1', etc.
    """
    import re
    
    def extract_number(cluster_id):
        # Try to extract number from cluster ID (e.g., 'cluster_10' -> 10)
        match = re.search(r'(\d+)', str(cluster_id))
        return int(match.group(1)) if match else float('inf')
    
    return sorted(cluster_ids, key=extract_number)


def plot_cluster_grid(cluster_metadata_profiles, cluster_ids=None, 
                     celltype_order=None, figsize=(24, 24), 
                     use_celltype_colors=True, cell_type_color_dict=None,
                     save_path=None, ncols=6, nrows=6):
    """
    Plot accessibility profiles for multiple clusters in a grid layout.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-celltype matrix
    cluster_ids : list or None
        List of cluster IDs to plot. If None, plots all clusters
    celltype_order : list or None
        Optional order for celltypes
    figsize : tuple
        Figure size
    use_celltype_colors : bool
        Whether to use celltype-specific colors
    cell_type_color_dict : dict or None
        Dictionary mapping celltypes to colors
    save_path : str or None
        Full path to save the figure
    ncols : int
        Number of columns in the grid
    nrows : int
        Number of rows in the grid
    """
    
    import matplotlib.pyplot as plt
    
    # Get cluster IDs to plot
    if cluster_ids is None:
        cluster_ids = cluster_metadata_profiles.index.tolist()
    
    # Sort cluster IDs numerically
    cluster_ids = sort_cluster_ids_numerically(cluster_ids)
    
    # Limit to the number of subplots available
    max_plots = ncols * nrows
    cluster_ids = cluster_ids[:max_plots]
    
    # Create the subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Make it easier to index
    
    # Plot each cluster
    for i, cluster_id in enumerate(cluster_ids):
        plot_single_cluster_subplot(
            cluster_metadata_profiles, cluster_id, axes[i],
            celltype_order=celltype_order,
            use_celltype_colors=use_celltype_colors,
            cell_type_color_dict=cell_type_color_dict
        )
    
    # Hide any unused subplots
    for j in range(len(cluster_ids), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to: {save_path}")
    
    plt.show()
    

def fit_temporal_regression(timepoint_order, accessibility_values):
    """
    Fit linear regression to temporal accessibility data.
    
    Parameters:
    -----------
    timepoint_order : list
        Ordered list of timepoints
    accessibility_values : array-like
        Accessibility values corresponding to timepoints
    
    Returns:
    --------
    dict with 'slope', 'r_squared', 'rmsd', 'y_pred'
    
    Note: Requires sklearn (pip install scikit-learn)
    """
    import numpy as np
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
    except ImportError:
        print("Warning: sklearn not available. Install with: pip install scikit-learn")
        # Fallback to numpy polyfit
        X = np.arange(len(timepoint_order))
        y = np.array(accessibility_values)
        
        # Fit linear regression using numpy
        coeffs = np.polyfit(X, y, 1)  # 1st degree polynomial (linear)
        slope = coeffs[0]
        y_pred = np.polyval(coeffs, X)
        
        # Calculate R-squared manually
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmsd = np.sqrt(np.mean((y - y_pred) ** 2))
        
        return {
            'slope': slope,
            'r_squared': r_squared,
            'rmsd': rmsd,
            'y_pred': y_pred
        }
    
    # Convert timepoints to numeric (assume they're in chronological order)
    X = np.arange(len(timepoint_order)).reshape(-1, 1)  # Use indices as x-values
    y = np.array(accessibility_values)
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Predictions
    y_pred = reg.predict(X)
    
    # Calculate metrics
    slope = reg.coef_[0]  # slope per timepoint unit
    r_squared = r2_score(y, y_pred)
    rmsd = np.sqrt(np.mean((y - y_pred) ** 2))
    
    return {
        'slope': slope,
        'r_squared': r_squared, 
        'rmsd': rmsd,
        'y_pred': y_pred
    }


def plot_single_cluster_timepoint_subplot(
    cluster_timepoint_profiles,
    cluster_id,
    ax,
    timepoint_order=None,
    *,
    color_by_timepoint=True,
    timepoint_colors=None,
    default_color="#B0B0B0",
    show_regression=True
):
    """
    Plot accessibility profile for one cluster across timepoints on a given axis (for subplots).
    
    Parameters:
    -----------
    cluster_timepoint_profiles : pd.DataFrame
        cluster-by-timepoint matrix
    cluster_id : str
        Cluster ID to plot
    ax : matplotlib axis
        Axis to plot on
    timepoint_order : list or None
        Optional order for timepoints
    color_by_timepoint : bool
        Whether to use timepoint-specific colors
    timepoint_colors : dict or None
        Dictionary mapping timepoints to colors
    default_color : str
        Default color if timepoint colors not provided
    show_regression : bool
        Whether to show linear regression line and stats
    """
    
    import pandas as pd
    
    if cluster_id not in cluster_timepoint_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return
    
    profile = cluster_timepoint_profiles.loc[cluster_id]
    
    # Order timepoints
    if timepoint_order is not None:
        ordered = pd.Series(index=timepoint_order, dtype=float)
        for tp in timepoint_order:
            ordered[tp] = profile.get(tp, 0.0)
        profile_to_plot = ordered
    else:
        profile_to_plot = profile.sort_values(ascending=False)
    
    # y-limit based on overall median vs this cluster's max
    overall_median = pd.Series(cluster_timepoint_profiles.values.flatten()).median()
    profile_max = profile_to_plot.max()
    y_limit = max(overall_median, profile_max) + 10
    
    # Colors
    if color_by_timepoint:
        if timepoint_colors is None:
            timepoint_colors = make_timepoint_palette(list(profile_to_plot.index))
        colors = [timepoint_colors.get(tp, default_color) for tp in profile_to_plot.index]
    else:
        colors = ["steelblue"] * len(profile_to_plot)
    
    # Plot bars on the given axis
    x_pos = range(len(profile_to_plot))
    ax.bar(x_pos, profile_to_plot.values, color=colors, alpha=0.85, edgecolor="none")
    
    # Fit and plot regression line if requested
    regression_stats = None
    if show_regression and len(profile_to_plot) > 1:
        regression_stats = fit_temporal_regression(
            list(profile_to_plot.index), 
            profile_to_plot.values
        )
        # Plot regression line
        ax.plot(x_pos, regression_stats['y_pred'], 'r--', linewidth=2, alpha=0.8)
    
    ax.set_ylim(0, y_limit)
    ax.axhline(y=overall_median, linestyle="--", alpha=0.5, color='gray')
    
    # Create title with metrics
    metrics = compute_comprehensive_accessibility_metrics(profile)
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    cv = metrics['cv']
    
    title = f'Cluster {cluster_id}\nent={entropy:.2f}, dom={dominance:.2f}, CV={cv:.1f}'
    
    if regression_stats and show_regression:
        slope = regression_stats['slope']
        r_sq = regression_stats['r_squared']
        rmsd = regression_stats['rmsd']
        
        # Interpret slope direction
        if abs(slope) < 0.1:
            trend = "steady"
        elif slope > 0:
            trend = "↗"  # increasing
        else:
            trend = "↘"  # decreasing
            
        title += f'\nslope={slope:.2f} {trend}, R²={r_sq:.2f}, RMSD={rmsd:.2f}'
    
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Timepoint', fontsize=8)
    ax.set_ylabel('Mean Accessibility', fontsize=8)
    
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(profile_to_plot.index, rotation=45, ha='right', fontsize=6)
    ax.set_xticklabels([])
    ax.grid(False)


def sort_cluster_ids_numerically(cluster_ids):
    """
    Sort cluster IDs numerically instead of lexicographically.
    Handles patterns like 'cluster_0', 'cluster_1', etc.
    """
    import re
    
    def extract_number(cluster_id):
        # Try to extract number from cluster ID (e.g., 'cluster_10' -> 10)
        match = re.search(r'(\d+)', str(cluster_id))
        return int(match.group(1)) if match else float('inf')
    
    return sorted(cluster_ids, key=extract_number)


def plot_timepoint_grid(
    cluster_timepoint_profiles, 
    cluster_ids=None,
    timepoint_order=None,
    figsize=(24, 24),
    color_by_timepoint=True,
    timepoint_colors=None,
    default_color="#B0B0B0",
    save_path=None,
    ncols=6,
    nrows=6
):
    """
    Plot timepoint accessibility profiles for multiple clusters in a grid layout.
    
    Parameters:
    -----------
    cluster_timepoint_profiles : pd.DataFrame
        cluster-by-timepoint matrix
    cluster_ids : list or None
        List of cluster IDs to plot. If None, plots all clusters
    timepoint_order : list or None
        Optional order for timepoints
    figsize : tuple
        Figure size
    color_by_timepoint : bool
        Whether to use timepoint-specific colors
    timepoint_colors : dict or None
        Dictionary mapping timepoints to colors
    default_color : str
        Default color if timepoint colors not provided
    save_path : str or None
        Full path to save the figure
    ncols : int
        Number of columns in the grid
    nrows : int
        Number of rows in the grid
    """
    
    import matplotlib.pyplot as plt
    
    # Get cluster IDs to plot
    if cluster_ids is None:
        cluster_ids = cluster_timepoint_profiles.index.tolist()
    
    # Sort cluster IDs numerically
    cluster_ids = sort_cluster_ids_numerically(cluster_ids)
    
    # Limit to the number of subplots available
    max_plots = ncols * nrows
    cluster_ids = cluster_ids[:max_plots]
    
    # Create timepoint colors if needed and not provided
    if color_by_timepoint and timepoint_colors is None:
        all_timepoints = cluster_timepoint_profiles.columns.tolist()
        if timepoint_order is not None:
            all_timepoints = timepoint_order
        timepoint_colors = make_timepoint_palette(all_timepoints)
    
    # Create the subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Make it easier to index
    
def plot_timepoint_grid(
    cluster_timepoint_profiles, 
    cluster_ids=None,
    timepoint_order=None,
    figsize=(24, 24),
    color_by_timepoint=True,
    timepoint_colors=None,
    default_color="#B0B0B0",
    save_path=None,
    ncols=6,
    nrows=6,
    show_regression=True
):
    """
    Plot timepoint accessibility profiles for multiple clusters in a grid layout.
    
    Parameters:
    -----------
    cluster_timepoint_profiles : pd.DataFrame
        cluster-by-timepoint matrix
    cluster_ids : list or None
        List of cluster IDs to plot. If None, plots all clusters
    timepoint_order : list or None
        Optional order for timepoints
    figsize : tuple
        Figure size
    color_by_timepoint : bool
        Whether to use timepoint-specific colors
    timepoint_colors : dict or None
        Dictionary mapping timepoints to colors
    default_color : str
        Default color if timepoint colors not provided
    save_path : str or None
        Full path to save the figure
    ncols : int
        Number of columns in the grid
    nrows : int
        Number of rows in the grid
    show_regression : bool
        Whether to show linear regression lines and temporal statistics
    """
    
    import matplotlib.pyplot as plt
    
    # Get cluster IDs to plot
    if cluster_ids is None:
        cluster_ids = cluster_timepoint_profiles.index.tolist()
    
    # Sort cluster IDs numerically
    cluster_ids = sort_cluster_ids_numerically(cluster_ids)
    
    # Limit to the number of subplots available
    max_plots = ncols * nrows
    cluster_ids = cluster_ids[:max_plots]
    
    # Create timepoint colors if needed and not provided
    if color_by_timepoint and timepoint_colors is None:
        all_timepoints = cluster_timepoint_profiles.columns.tolist()
        if timepoint_order is not None:
            all_timepoints = timepoint_order
        timepoint_colors = make_timepoint_palette(all_timepoints)
    
    # Create the subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Make it easier to index
    
    # Plot each cluster
    for i, cluster_id in enumerate(cluster_ids):
        plot_single_cluster_timepoint_subplot(
            cluster_timepoint_profiles,
            cluster_id,
            axes[i],
            timepoint_order=timepoint_order,
            color_by_timepoint=color_by_timepoint,
            timepoint_colors=timepoint_colors,
            default_color=default_color,
            show_regression=show_regression
        )
    
    # Hide any unused subplots
    for j in range(len(cluster_ids), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timepoint grid plot saved to: {save_path}")
    
    plt.show()



def create_celltype_order_from_lineages(lineage_mapping=None):
    """
    Create ordered celltype list based on lineage groupings.
    
    Parameters:
    -----------
    lineage_mapping : dict or None
        Lineage to celltype mapping. Uses DEFAULT_LINEAGE_MAPPING if None.
        
    Returns:
    --------
    list
        Ordered celltype list grouped by lineages
    """
    
    if lineage_mapping is None:
        lineage_mapping = DEFAULT_LINEAGE_MAPPING
    
    # Define lineage order (CNS → Neural Crest → Mesoderm → Endoderm → Epiderm → Germline)
    lineage_order = ["CNS", "Neural Crest", "Paraxial Mesoderm", "Lateral Mesoderm", 
                     "Endoderm", "Epiderm", "Germline"]
    
    ordered_celltypes = []
    
    for lineage in lineage_order:
        if lineage in lineage_mapping:
            ordered_celltypes.extend(lineage_mapping[lineage])
    
    return ordered_celltypes


def compute_comprehensive_accessibility_metrics(profile):
    """
    Compute multiple metrics to quantify distribution flatness vs enrichment.
    
    Parameters:
    -----------
    profile : pd.Series
        Accessibility values across categories
        
    Returns:
    --------
    dict
        Dictionary with multiple distribution metrics
    """
    
    if profile.sum() == 0:
        return {'entropy': 0, 'dominance': 0, 'cv': 0, 'top3_fraction': 0, 
                'effective_categories': 0, 'gini': 0}
    
    # Basic metrics
    total = profile.sum()
    proportions = profile / total
    
    # 1. Shannon Entropy (your current metric)
    nonzero_props = proportions[proportions > 0]
    if len(nonzero_props) > 1:
        entropy = -np.sum(nonzero_props * np.log2(nonzero_props))
        max_entropy = np.log2(len(nonzero_props))
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0
    
    # 2. Dominance (fraction in top category)
    dominance = profile.max() / total
    
    # 3. Coefficient of Variation (measures relative variability)
    cv = profile.std() / profile.mean() if profile.mean() > 0 else 0
    
    # 4. Top-3 Fraction (concentration in top 3 categories)
    top3_sum = profile.nlargest(3).sum()
    top3_fraction = top3_sum / total
    
    # 5. Effective Number of Categories (1/sum(p_i^2))
    # This is more sensitive to concentration than entropy
    effective_n = 1 / np.sum(proportions**2) if np.sum(proportions**2) > 0 else 0
    
    # 6. Gini Coefficient (inequality measure)
    sorted_vals = np.sort(profile.values)
    n = len(sorted_vals)
    if n > 1 and sorted_vals.sum() > 0:
        cumsum = np.cumsum(sorted_vals)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    else:
        gini = 0
    
    # 7. 90/10 Ratio (top 10% vs bottom 10% of categories)
    n_top = max(1, int(n * 0.1))
    n_bottom = max(1, int(n * 0.1))
    top_10pct_mean = profile.nlargest(n_top).mean()
    bottom_10pct_mean = profile.nsmallest(n_bottom).mean()
    ratio_90_10 = top_10pct_mean / bottom_10pct_mean if bottom_10pct_mean > 0 else np.inf
    
    return {
        'entropy': normalized_entropy,
        'dominance': dominance,
        'cv': cv,
        'top3_fraction': top3_fraction,
        'effective_categories': effective_n,
        'gini': gini,
        'ratio_90_10': ratio_90_10,
        'dominant_category': profile.idxmax()
    }


def classify_accessibility_pattern_comprehensive(profile, metadata_type='celltype'):
    """
    Classify accessibility patterns using multiple complementary metrics.
    """
    
    metrics = compute_comprehensive_accessibility_metrics(profile)
    
    # Extract key metrics
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    cv = metrics['cv']
    top3_fraction = metrics['top3_fraction']
    effective_n = metrics['effective_categories']
    gini = metrics['gini']
    ratio_90_10 = metrics['ratio_90_10']
    dominant_cat = metrics['dominant_category']
    
    # Multi-metric classification
    # 1. HIGHLY SPECIFIC (strong concentration)
    if dominance >= 0.4 or top3_fraction >= 0.7 or ratio_90_10 >= 10:
        pattern = f"specific_{dominant_cat}"
        confidence = "high" if dominance >= 0.5 else "medium"
        
    # 2. MODERATELY ENRICHED
    elif dominance >= 0.2 or top3_fraction >= 0.5 or ratio_90_10 >= 3:
        pattern = f"enriched_{dominant_cat}"
        confidence = "medium" if dominance >= 0.25 else "low"
        
    # 3. BROADLY ACCESSIBLE (flat distribution)
    elif effective_n >= 15 and cv <= 2.0 and ratio_90_10 <= 2:
        pattern = "broadly_accessible"
        confidence = "high" if effective_n >= 20 else "medium"
        
    # 4. INTERMEDIATE
    else:
        pattern = "intermediate"
        confidence = "low"
    
    return pattern, confidence, metrics
    """
    Plot how entropy changes with different accessibility thresholds.
    """
    
    import matplotlib.pyplot as plt
    
    print("="*80)
    print("ENTROPY THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)
    
    entropy_by_threshold = {}
    
    for threshold in thresholds:
        entropies = []
        for cluster in cluster_celltype_profiles.index:
            profile = cluster_celltype_profiles.loc[cluster]
            filtered_profile = profile[profile >= threshold]
            entropy = compute_accessibility_entropy(filtered_profile) if len(filtered_profile) > 1 else 0
            entropies.append(entropy)
        
        entropy_by_threshold[f'threshold_{threshold}'] = entropies
        print(f"Threshold ≥{threshold}: mean entropy = {np.mean(entropies):.3f}, std = {np.std(entropies):.3f}")
    
    # Plot entropy distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (label, entropies) in enumerate(entropy_by_threshold.items()):
        threshold = label.split('_')[1]
        ax.hist(entropies, bins=20, alpha=0.6, label=f'Threshold ≥{threshold}')
    
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Entropy Distribution by Accessibility Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return entropy_by_threshold
    """
    Run entropy analysis using lineage-level aggregation.
    
    Parameters:
    -----------
    adata : AnnData
        Data object
    cluster_col : str
        Cluster column name
    lineage_mapping : dict, optional
        Custom lineage mapping. Uses default if None.
    min_cells : int
        Minimum cells per group
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    pd.DataFrame
        Lineage entropy analysis results
    """
    
    if lineage_mapping is None:
        lineage_mapping = DEFAULT_LINEAGE_MAPPING
        
    return run_metadata_entropy_analysis(
        adata, cluster_col, 'lineage', min_cells, 
        lineage_mapping=lineage_mapping, verbose=verbose
    )



# %%
"""
Refactored Peak Cluster Entropy Analysis Module

Streamlined implementation for analyzing peak cluster accessibility patterns.
Core concept: Aggregate across metadata, then across peaks within clusters.

Author: Yang-Joon Kim
Date: 2025-08-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import re


def create_cluster_metadata_profiles(adata, cluster_col='leiden_coarse', 
                                   metadata_type='celltype', min_cells=20,
                                   compute_sem=True, verbose=True):
    """
    Unified function to create cluster-by-metadata profiles with standard error.
    
    Workflow:
    1. Filter pseudobulk groups with <min_cells
    2. Parse metadata from group names (celltype_timepoint format)
    3. Aggregate across the OTHER metadata (e.g., if analyzing celltype, average across timepoints)
    4. Aggregate across peaks within each cluster (compute mean and SEM)
    
    Parameters:
    -----------
    adata : AnnData
        Data with peaks as obs, pseudobulk groups as var
    cluster_col : str
        Cluster column name in adata.obs
    metadata_type : str
        'celltype', 'timepoint', or 'lineage'
    min_cells : int
        Minimum cells per pseudobulk group
    compute_sem : bool
        Whether to compute standard error of the mean
    verbose : bool
        Print progress information
        
    Returns:
    --------
    pd.DataFrame
        cluster-by-metadata matrix (means)
    pd.DataFrame or None
        cluster-by-metadata matrix (SEMs) if compute_sem=True
    dict
        metadata info (colors, order, etc.)
    """
    
    if verbose:
        print(f"Creating cluster-by-{metadata_type} profiles...")
        print(f"Input data: {adata.shape[0]} peaks × {adata.shape[1]} pseudobulk groups")
    
    # Step 1: Filter reliable groups
    reliable_groups = []
    for col in adata.var.index:
        cell_count = int(adata[:, col].var["n_cells"].iloc[0])
        if cell_count >= min_cells:
            reliable_groups.append(col)
    
    if verbose:
        print(f"Kept {len(reliable_groups)}/{adata.shape[1]} groups with ≥{min_cells} cells")
    
    # Step 2: Parse metadata from group names
    metadata_mapping = _parse_group_metadata(reliable_groups, verbose=verbose)
    
    # Step 3: Get data for reliable groups only
    reliable_indices = [adata.var.index.get_loc(col) for col in reliable_groups]
    reliable_data = adata.X[:, reliable_indices]
    if hasattr(reliable_data, 'toarray'):
        reliable_data = reliable_data.toarray()
    
    # Create DataFrame with cluster assignments
    data_df = pd.DataFrame(
        reliable_data,
        columns=reliable_groups,
        index=adata.obs.index
    )
    data_df['cluster'] = adata.obs[cluster_col].astype(str)
    
    # Step 4: First aggregation - average across OTHER metadata dimension
    if metadata_type == 'lineage':
        # Use default lineage mapping if not provided
        lineage_mapping = _get_default_lineage_mapping()
        group_to_metadata = {}
        for group in reliable_groups:
            celltype = metadata_mapping['celltype_mapping'].get(group)
            if celltype:
                for lineage, celltypes in lineage_mapping.items():
                    if celltype in celltypes:
                        group_to_metadata[group] = lineage
                        break
    else:
        group_to_metadata = metadata_mapping[f'{metadata_type}_mapping']
    
    # Create peak-by-metadata matrix (first aggregation)
    peak_metadata_data = pd.DataFrame(index=data_df.index)
    
    for metadata_category in set(group_to_metadata.values()):
        # Get groups for this metadata category
        category_groups = [g for g, m in group_to_metadata.items() if m == metadata_category]
        
        if category_groups:
            # Average across groups in this category (e.g., average across timepoints for each celltype)
            category_mean = data_df[category_groups].mean(axis=1)
            peak_metadata_data[metadata_category] = category_mean
    
    # Step 5: Second aggregation - aggregate across peaks within each cluster (with SEM)
    cluster_metadata_profiles = pd.DataFrame(index=data_df['cluster'].unique())
    cluster_metadata_sems = pd.DataFrame(index=data_df['cluster'].unique()) if compute_sem else None
    
    for metadata_category in peak_metadata_data.columns:
        category_data = pd.DataFrame({
            'accessibility': peak_metadata_data[metadata_category],
            'cluster': data_df['cluster']
        })
        
        # Group by cluster and compute mean and SEM
        cluster_stats = category_data.groupby('cluster')['accessibility'].agg(['mean', 'std', 'count'])
        
        # Store means
        cluster_metadata_profiles[metadata_category] = cluster_stats['mean']
        
        # Compute and store SEM if requested
        if compute_sem:
            sem_values = cluster_stats['std'] / np.sqrt(cluster_stats['count'])
            cluster_metadata_sems[metadata_category] = sem_values.fillna(0)  # Handle single-peak clusters
    
    # Remove any clusters with all-zero profiles
    cluster_metadata_profiles = cluster_metadata_profiles.fillna(0)
    valid_clusters = (cluster_metadata_profiles > 0).any(axis=1)
    cluster_metadata_profiles = cluster_metadata_profiles.loc[valid_clusters]
    
    if compute_sem:
        cluster_metadata_sems = cluster_metadata_sems.fillna(0)
        cluster_metadata_sems = cluster_metadata_sems.loc[valid_clusters]
    
    # Create metadata info
    metadata_info = {
        'categories': list(cluster_metadata_profiles.columns),
        'colors': _create_color_palette(list(cluster_metadata_profiles.columns), metadata_type),
        'order': _get_default_order(list(cluster_metadata_profiles.columns), metadata_type)
    }
    
    if verbose:
        print(f"Created {cluster_metadata_profiles.shape} cluster-by-{metadata_type} matrix")
        if compute_sem:
            print(f"Computed standard errors for {cluster_metadata_sems.shape[0]} clusters")
        print(f"Categories: {metadata_info['categories']}")
    
    if compute_sem:
        return cluster_metadata_profiles, cluster_metadata_sems, metadata_info
    else:
        return cluster_metadata_profiles, None, metadata_info


def analyze_accessibility_patterns(cluster_metadata_profiles, metadata_type='celltype',
                                 entropy_threshold=0.8, dominance_threshold=0.3):
    """
    Analyze accessibility patterns using entropy and dominance.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-metadata matrix
    metadata_type : str
        Type of metadata being analyzed
    entropy_threshold : float
        Threshold for broad vs specific classification
    dominance_threshold : float
        Threshold for specific accessibility
        
    Returns:
    --------
    pd.DataFrame
        Results with patterns and metrics
    """
    
    results = []
    
    for cluster in cluster_metadata_profiles.index:
        profile = cluster_metadata_profiles.loc[cluster]
        
        # Compute metrics
        metrics = _compute_accessibility_metrics(profile)
        
        # Classify pattern
        pattern, confidence = _classify_pattern(metrics, entropy_threshold, dominance_threshold)
        
        results.append({
            'cluster': cluster,
            'pattern': pattern,
            'confidence': confidence,
            'dominant_category': metrics['dominant_category'],
            **metrics  # Include all metrics
        })
    
    return pd.DataFrame(results)


def plot_cluster_profile(cluster_metadata_profiles, cluster_id, metadata_info, 
                        cluster_metadata_sems=None, figsize=(8, 5), 
                        show_metrics=True, show_error_bars=True, return_fig=True):
    """
    Plot accessibility profile for a single cluster with optional error bars.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-metadata matrix (means)
    cluster_id : str
        Cluster ID to plot
    metadata_info : dict
        Metadata information (colors, order, etc.)
    cluster_metadata_sems : pd.DataFrame or None
        cluster-by-metadata matrix (SEMs)
    figsize : tuple
        Figure size
    show_metrics : bool
        Whether to show entropy/dominance in title
    show_error_bars : bool
        Whether to show error bars (requires cluster_metadata_sems)
    return_fig : bool
        Whether to return figure object instead of showing
        
    Returns:
    --------
    matplotlib.figure.Figure or None
        Figure object if return_fig=True
    """
    
    if cluster_id not in cluster_metadata_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return None
    
    # Get and order profile
    profile = cluster_metadata_profiles.loc[cluster_id]
    if 'order' in metadata_info and metadata_info['order']:
        ordered_profile = profile.reindex(metadata_info['order'], fill_value=0)
    else:
        ordered_profile = profile.sort_values(ascending=False)
    
    # Get corresponding SEMs if available
    sems = None
    if show_error_bars and cluster_metadata_sems is not None:
        if cluster_id in cluster_metadata_sems.index:
            cluster_sems = cluster_metadata_sems.loc[cluster_id]
            if 'order' in metadata_info and metadata_info['order']:
                sems = cluster_sems.reindex(metadata_info['order'], fill_value=0)
            else:
                sems = cluster_sems.reindex(ordered_profile.index, fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors
    colors = [metadata_info['colors'].get(cat, 'steelblue') 
              for cat in ordered_profile.index]
    
    # Plot bars with error bars
    x_pos = range(len(ordered_profile))
    bars = ax.bar(x_pos, ordered_profile.values, color=colors, alpha=0.8,
                  yerr=sems.values if sems is not None else None,
                  capsize=3, error_kw={'alpha': 0.7})
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ordered_profile.index, rotation=45, ha='right')
    ax.set_ylabel('Mean Accessibility')
    
    # Title with metrics
    if show_metrics:
        metrics = _compute_accessibility_metrics(profile)
        title = f'Cluster {cluster_id}\nEntropy: {metrics["entropy"]:.3f}, Dominance: {metrics["dominance"]:.3f}'
    else:
        title = f'Cluster {cluster_id}'
    
    ax.set_title(title, fontweight='bold')
    ax.grid(False)
    
    # Add horizontal line at median
    median_val = cluster_metadata_profiles.values.flatten()
    median_val = np.median(median_val[median_val > 0])
    ax.axhline(y=median_val, linestyle='--', alpha=0.5, color='gray')
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def create_profile_grid(cluster_metadata_profiles, metadata_info, 
                       cluster_metadata_sems=None, cluster_ids=None, 
                       ncols=6, nrows=6, figsize=(24, 24), 
                       show_error_bars=True, save_path=None):
    """
    Create grid of cluster profiles with optional error bars.
    
    Parameters:
    -----------
    cluster_metadata_profiles : pd.DataFrame
        cluster-by-metadata matrix (means)
    metadata_info : dict
        Metadata information
    cluster_metadata_sems : pd.DataFrame or None
        cluster-by-metadata matrix (SEMs)
    cluster_ids : list or None
        Specific clusters to plot
    ncols, nrows : int
        Grid dimensions
    figsize : tuple
        Overall figure size
    show_error_bars : bool
        Whether to show error bars
    save_path : str or None
        Path to save figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Grid figure object
    """
    
    if cluster_ids is None:
        cluster_ids = _sort_cluster_ids_numerically(cluster_metadata_profiles.index)
    
    cluster_ids = cluster_ids[:ncols * nrows]  # Limit to grid size
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, cluster_id in enumerate(cluster_ids):
        if i >= len(axes):
            break
            
        ax = axes[i]
        _plot_on_axis(cluster_metadata_profiles, cluster_id, metadata_info, ax,
                     cluster_metadata_sems=cluster_metadata_sems,
                     show_error_bars=show_error_bars)
    
    # Hide unused subplots
    for j in range(len(cluster_ids), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid saved to: {save_path}")
    
    return fig


def run_complete_analysis(adata, cluster_col='leiden_coarse', 
                         metadata_type='celltype', min_cells=20, compute_sem=True):
    """
    Run complete analysis workflow with optional SEM calculation.
    
    Parameters:
    -----------
    adata : AnnData
        Input data
    cluster_col : str
        Cluster column
    metadata_type : str
        Type of metadata to analyze
    min_cells : int
        Minimum cells per group
    compute_sem : bool
        Whether to compute standard errors
        
    Returns:
    --------
    tuple
        (profiles_df, results_df, metadata_info, sems_df)
        sems_df is None if compute_sem=False
    """
    
    # Create profiles
    if compute_sem:
        profiles_df, sems_df, metadata_info = create_cluster_metadata_profiles(
            adata, cluster_col, metadata_type, min_cells, compute_sem=True
        )
    else:
        profiles_df, _, metadata_info = create_cluster_metadata_profiles(
            adata, cluster_col, metadata_type, min_cells, compute_sem=False
        )
        sems_df = None
    
    # Analyze patterns
    results_df = analyze_accessibility_patterns(profiles_df, metadata_type)
    
    return profiles_df, results_df, metadata_info, sems_df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_group_metadata(groups, verbose=True):
    """Parse celltype and timepoint from group names (celltype_timepoint format)."""
    
    celltype_mapping = {}
    timepoint_mapping = {}
    
    for group in groups:
        # Match pattern: celltype_timepoint (e.g., neural_15somites)
        match = re.search(r'^(.+)_(\d+somites)$', group)
        if match:
            celltype, timepoint = match.groups()
            celltype_mapping[group] = celltype
            timepoint_mapping[group] = timepoint
    
    if verbose:
        unique_celltypes = set(celltype_mapping.values())
        unique_timepoints = set(timepoint_mapping.values())
        print(f"Parsed {len(celltype_mapping)} groups: "
              f"{len(unique_celltypes)} celltypes × {len(unique_timepoints)} timepoints")
    
    return {
        'celltype_mapping': celltype_mapping,
        'timepoint_mapping': timepoint_mapping
    }


def _compute_accessibility_metrics(profile):
    """Compute comprehensive accessibility metrics."""
    
    if profile.sum() == 0:
        return {'entropy': 0, 'dominance': 0, 'cv': 0, 'dominant_category': 'none'}
    
    # Shannon entropy
    proportions = profile / profile.sum()
    nonzero_props = proportions[proportions > 0]
    if len(nonzero_props) > 1:
        entropy = -np.sum(nonzero_props * np.log2(nonzero_props))
        max_entropy = np.log2(len(nonzero_props))
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0
    
    # Dominance (fraction in top category)
    dominance = profile.max() / profile.sum()
    
    # Coefficient of variation
    cv = profile.std() / profile.mean() if profile.mean() > 0 else 0
    
    return {
        'entropy': normalized_entropy,
        'dominance': dominance,
        'cv': cv,
        'dominant_category': profile.idxmax()
    }


def _classify_pattern(metrics, entropy_threshold=0.8, dominance_threshold=0.3):
    """Classify accessibility pattern based on metrics."""
    
    entropy = metrics['entropy']
    dominance = metrics['dominance']
    dominant_cat = metrics['dominant_category']
    
    if entropy >= entropy_threshold:
        return "broadly_accessible", "high"
    elif dominance >= dominance_threshold:
        return f"specific_{dominant_cat}", "high" if dominance >= 0.4 else "medium"
    else:
        return f"enriched_{dominant_cat}", "medium"


def _create_color_palette(categories, metadata_type):
    """Create color palette for categories."""
    
    if metadata_type == 'celltype':
        # Create biologically-informed color palette
        color_scheme = {
            # CNS/Neural - blues
            'neural': '#1f77b4', 'neural_optic': '#aec7e8', 'neural_posterior': '#4682b4',
            'neural_telencephalon': '#6495ed', 'neurons': '#0000cd', 'differentiating_neurons': '#4169e1',
            'hindbrain': '#1e90ff', 'midbrain_hindbrain_boundary': '#87ceeb', 'spinal_cord': '#00bfff',
            'optic_cup': '#87cefa', 'floor_plate': '#b0e0e6', 'neural_floor_plate': '#add8e6',
            
            # Neural Crest - purples
            'neural_crest': '#9467bd', 'enteric_neurons': '#c5b0d5',
            
            # Early Mesoderm - dark greens
            'NMPs': '#2ca02c', 'tail_bud': '#98df8a',
            
            # Axial Mesoderm - brown
            'notochord': '#8c564b',
            
            # Paraxial Mesoderm - greens
            'PSM': '#2ca02c', 'somites': '#98df8a', 'fast_muscle': '#c5b0d5', 'muscle': '#bcbd22',
            
            # Lateral Plate Mesoderm - reds
            'lateral_plate_mesoderm': '#d62728', 'heart_myocardium': '#ff7f0e',
            'hematopoietic_vasculature': '#ff9896', 'hemangioblasts': '#ffbb78',
            
            # Other Mesoderm - oranges
            'pharyngeal_arches': '#ff7f0e', 'pronephros': '#ffbb78', 'hatching_gland': '#ffd700',
            
            # Endoderm - yellows
            'endoderm': '#bcbd22', 'endocrine_pancreas': '#dbdb8d',
            
            # Ectoderm - grays
            'epidermis': '#7f7f7f',
            
            # Germline - pink
            'primordial_germ_cells': '#e377c2'
        }
        
        # Use predefined colors if available, otherwise generate
        palette = {}
        for cat in categories:
            if cat in color_scheme:
                palette[cat] = color_scheme[cat]
            else:
                # Generate color for unknown categories
                palette[cat] = plt.cm.Set3(hash(cat) % 12 / 12)
                
        return palette
        
    elif metadata_type == 'timepoint':
        # Use sequential palette for timepoints (temporal progression)
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        return {cat: colors[i] for i, cat in enumerate(categories)}
    else:
        # Default qualitative palette for other metadata types
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        return {cat: colors[i] for i, cat in enumerate(categories)}


def _get_default_order(categories, metadata_type):
    """Get default ordering for categories."""
    
    if metadata_type == 'timepoint':
        # Sort timepoints numerically (0somites, 5somites, etc.)
        def extract_number(tp):
            match = re.search(r'(\d+)', tp)
            return int(match.group(1)) if match else 0
        return sorted(categories, key=extract_number)
    elif metadata_type == 'celltype':
        # Developmental lineage-based ordering
        celltype_order = [
            # CNS/Neural
            'neural', 'neural_optic', 'neural_posterior', 'neural_telencephalon', 
            'neurons', 'differentiating_neurons',
            'hindbrain', 'midbrain_hindbrain_boundary', 'spinal_cord',
            'optic_cup', 'floor_plate', 'neural_floor_plate',
            
            # Neural Crest + derivatives
            'neural_crest', 'enteric_neurons',
            
            # Early mesoderm/multipotent
            'NMPs', 'tail_bud',
            
            # Axial mesoderm
            'notochord',
            
            # Paraxial mesoderm  
            'PSM', 'somites', 'fast_muscle', 'muscle',
            
            # Lateral plate mesoderm
            'lateral_plate_mesoderm', 'heart_myocardium', 
            'hematopoietic_vasculature', 'hemangioblasts',
            
            # Other mesoderm-derived
            'pharyngeal_arches', 'pronephros', 'hatching_gland',
            
            # Endoderm
            'endoderm', 'endocrine_pancreas',
            
            # Ectoderm
            'epidermis',
            
            # Germline
            'primordial_germ_cells'
        ]
        
        # Return categories in the specified order (only those that exist)
        ordered = [ct for ct in celltype_order if ct in categories]
        # Add any remaining categories not in the predefined order
        remaining = [ct for ct in categories if ct not in celltype_order]
        return ordered + sorted(remaining)
    else:
        # Alphabetical for other metadata types
        return sorted(categories)


def _get_default_lineage_mapping():
    """Default lineage to celltype mapping (updated to match celltype ordering)."""
    
    return {
        "CNS": [
            "neural", "neural_optic", "neural_posterior", "neural_telencephalon",
            "neurons", "differentiating_neurons", "hindbrain", 
            "midbrain_hindbrain_boundary", "spinal_cord", "optic_cup", 
            "floor_plate", "neural_floor_plate"
        ],
        "Neural Crest": [
            "neural_crest", "enteric_neurons"
        ],
        "Early Mesoderm": [
            "NMPs", "tail_bud"
        ],
        "Axial Mesoderm": [
            "notochord"
        ],
        "Paraxial Mesoderm": [
            "PSM", "somites", "fast_muscle", "muscle"
        ],
        "Lateral Plate Mesoderm": [
            "lateral_plate_mesoderm", "heart_myocardium", 
            "hematopoietic_vasculature", "hemangioblasts"
        ],
        "Other Mesoderm": [
            "pharyngeal_arches", "pronephros", "hatching_gland"
        ],
        "Endoderm": [
            "endoderm", "endocrine_pancreas"
        ],
        "Ectoderm": [
            "epidermis"
        ],
        "Germline": [
            "primordial_germ_cells"
        ]
    }


def _sort_cluster_ids_numerically(cluster_ids):
    """Sort cluster IDs numerically."""
    
    def extract_number(cluster_id):
        match = re.search(r'(\d+)', str(cluster_id))
        return int(match.group(1)) if match else float('inf')
    
    return sorted(cluster_ids, key=extract_number)


def _plot_on_axis(cluster_metadata_profiles, cluster_id, metadata_info, ax,
                 cluster_metadata_sems=None, show_error_bars=True):
    """Plot single cluster profile on given axis with optional error bars."""
    
    if cluster_id not in cluster_metadata_profiles.index:
        ax.set_visible(False)
        return
    
    profile = cluster_metadata_profiles.loc[cluster_id]
    
    # Order profile
    if 'order' in metadata_info and metadata_info['order']:
        ordered_profile = profile.reindex(metadata_info['order'], fill_value=0)
    else:
        ordered_profile = profile.sort_values(ascending=False)
    
    # Get corresponding SEMs if available
    sems = None
    if show_error_bars and cluster_metadata_sems is not None:
        if cluster_id in cluster_metadata_sems.index:
            cluster_sems = cluster_metadata_sems.loc[cluster_id]
            if 'order' in metadata_info and metadata_info['order']:
                sems = cluster_sems.reindex(metadata_info['order'], fill_value=0)
            else:
                sems = cluster_sems.reindex(ordered_profile.index, fill_value=0)
    
    # Colors
    colors = [metadata_info['colors'].get(cat, 'steelblue') 
              for cat in ordered_profile.index]
    
    # Plot with error bars
    x_pos = range(len(ordered_profile))
    ax.bar(x_pos, ordered_profile.values, color=colors, alpha=0.8,
           yerr=sems.values if sems is not None else None,
           capsize=2, error_kw={'alpha': 0.6, 'linewidth': 1})
    
    # Metrics for title
    metrics = _compute_accessibility_metrics(profile)
    title = f'Cluster {cluster_id}\nE={metrics["entropy"]:.2f}, D={metrics["dominance"]:.2f}'
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ordered_profile.index, rotation=45, ha='right', fontsize=8)
    ax.grid(False)


# %%
def plot_cluster_heatmap(cluster_profiles, cluster_id, figsize=(12, 8), 
                        cmap='RdBu_r', save_path=None, show_values=False):
    """
    Generate a timepoint × celltype heatmap for a specific cluster.
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        cluster-by-pseudobulk matrix 
    cluster_id : str or int
        Cluster ID to plot
    figsize : tuple
        Figure size
    cmap : str
        Colormap name (default: 'RdBu_r')
    save_path : str or None
        Path to save figure
    show_values : bool
        Whether to show values in cells
        
    Returns:
    --------
    matplotlib.figure.Figure
        Heatmap figure
    """
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert cluster_id to string for consistency
    cluster_id = str(cluster_id)
    
    if cluster_id not in cluster_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return None
    
    # Get accessibility values for this cluster
    cluster_data = cluster_profiles.loc[cluster_id]
    
    # Parse pseudobulk group names into timepoint and celltype
    timepoint_celltype_data = []
    
    for group, accessibility in cluster_data.items():
        # Split by last underscore to separate celltype from timepoint
        if '_' in group and 'somites' in group:
            parts = group.rsplit('_', 1)  # Split from right, only once
            if len(parts) == 2 and parts[1].endswith('somites'):
                celltype = parts[0]
                timepoint = parts[1]
                timepoint_celltype_data.append({
                    'celltype': celltype,
                    'timepoint': timepoint,
                    'accessibility': accessibility
                })
    
    if not timepoint_celltype_data:
        print(f"No valid pseudobulk groups found for cluster {cluster_id}")
        return None
    
    # Convert to DataFrame and pivot
    heatmap_df = pd.DataFrame(timepoint_celltype_data)
    heatmap_matrix = heatmap_df.pivot(index='timepoint', columns='celltype', values='accessibility')
    heatmap_matrix = heatmap_matrix.fillna(0)  # Fill missing combinations with 0
    
    # Sort timepoints numerically
    timepoint_order = sorted(heatmap_matrix.index, 
                           key=lambda x: int(x.replace('somites', '')) if x.replace('somites', '').isdigit() else 0)
    heatmap_matrix = heatmap_matrix.reindex(timepoint_order)
    
    # Sort celltypes alphabetically (or you can customize this)
    celltype_order = sorted(heatmap_matrix.columns)
    heatmap_matrix = heatmap_matrix[celltype_order]
    
    # Create figure and heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap using imshow
    im = ax.imshow(heatmap_matrix.values, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(heatmap_matrix.columns)))
    ax.set_xticklabels(heatmap_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_matrix.index)))
    ax.set_yticklabels(heatmap_matrix.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accessibility', rotation=270, labelpad=20)
    
    # Add values to cells if requested
    if show_values:
        for i in range(len(heatmap_matrix.index)):
            for j in range(len(heatmap_matrix.columns)):
                value = heatmap_matrix.iloc[i, j]
                if value > 0:  # Only show non-zero values
                    # Choose text color based on background
                    text_color = 'white' if value > heatmap_matrix.values.max() * 0.5 else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                           color=text_color, fontsize=8)
    
    # Title and labels
    ax.set_title(f'Cluster {cluster_id} - Timepoint × Celltype Accessibility', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Celltype', fontweight='bold')
    ax.set_ylabel('Timepoint', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    return fig


# Example usage:
# fig = plot_cluster_heatmap(cluster_profiles, cluster_id=23, 
#                           figsize=(15, 8), cmap='RdBu_r', 
#                           save_path='cluster_23_heatmap.pdf')

# %%
def plot_cluster_heatmap(cluster_profiles, cluster_id, celltype_orders, cell_type_color_dict,
                        figsize=(12, 8), cmap='RdBu_r', save_path=None, show_values=False,
                        vmin=None, vmax=None):
    """
    Generate a timepoint × celltype heatmap for a specific cluster.
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        cluster-by-pseudobulk matrix 
    cluster_id : str or int
        Cluster ID to plot
    celltype_orders : list
        List of celltypes in desired order
    cell_type_color_dict : dict
        Dictionary mapping celltypes to colors
    figsize : tuple
        Figure size
    cmap : str
        Colormap name (default: 'RdBu_r')
    save_path : str or None
        Path to save figure
    show_values : bool
        Whether to show values in cells
    vmin, vmax : float or None
        Colormap limits
        
    Returns:
    --------
    matplotlib.figure.Figure
        Heatmap figure
    """
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    
    # Convert cluster_id to string for consistency
    cluster_id = str(cluster_id)
    
    if cluster_id not in cluster_profiles.index:
        print(f"Cluster {cluster_id} not found")
        return None
    
    # Get accessibility values for this cluster
    cluster_data = cluster_profiles.loc[cluster_id]
    
    # Parse pseudobulk group names into timepoint and celltype
    timepoint_celltype_data = []
    
    for group, accessibility in cluster_data.items():
        # Split by last underscore to separate celltype from timepoint
        if '_' in group and 'somites' in group:
            parts = group.rsplit('_', 1)  # Split from right, only once
            if len(parts) == 2 and parts[1].endswith('somites'):
                celltype = parts[0]
                timepoint = parts[1]
                timepoint_celltype_data.append({
                    'celltype': celltype,
                    'timepoint': timepoint,
                    'accessibility': accessibility
                })
    
    if not timepoint_celltype_data:
        print(f"No valid pseudobulk groups found for cluster {cluster_id}")
        return None
    
    # Convert to DataFrame and pivot
    heatmap_df = pd.DataFrame(timepoint_celltype_data)
    heatmap_matrix = heatmap_df.pivot(index='timepoint', columns='celltype', values='accessibility')
    heatmap_matrix = heatmap_matrix.fillna(0)  # Fill missing combinations with 0
    
    # Sort timepoints numerically
    timepoint_order = sorted(heatmap_matrix.index, 
                           key=lambda x: int(x.replace('somites', '')) if x.replace('somites', '').isdigit() else 0)
    heatmap_matrix = heatmap_matrix.reindex(timepoint_order)
    
    # Order celltypes according to biological order
    available_celltypes = [ct for ct in celltype_orders if ct in heatmap_matrix.columns]
    # Add any celltypes not in the predefined order
    remaining_celltypes = [ct for ct in heatmap_matrix.columns if ct not in celltype_orders]
    final_celltype_order = available_celltypes + sorted(remaining_celltypes)
    heatmap_matrix = heatmap_matrix[final_celltype_order]
    
    # Create figure with extra space at bottom for color blocks
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap using imshow
    im = ax.imshow(heatmap_matrix.values, cmap=cmap, aspect='auto', 
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Set ticks and labels
    ax.set_xticks(range(len(heatmap_matrix.columns)))
    ax.set_xticklabels(heatmap_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_matrix.index)))
    ax.set_yticklabels(heatmap_matrix.index)
    
    # Add colored blocks for celltypes at bottom
    for i, celltype in enumerate(heatmap_matrix.columns):
        if celltype in cell_type_color_dict:
            color = cell_type_color_dict[celltype]
            # Add rectangle at bottom of plot
            rect = Rectangle((i-0.4, -0.8), 0.8, 0.3, 
                           facecolor=color, edgecolor='black', linewidth=0.5,
                           transform=ax.transData, clip_on=False)
            ax.add_patch(rect)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accessibility', rotation=270, labelpad=20)
    
    # Add values to cells if requested
    if show_values:
        for i in range(len(heatmap_matrix.index)):
            for j in range(len(heatmap_matrix.columns)):
                value = heatmap_matrix.iloc[i, j]
                if value > 0:  # Only show non-zero values
                    # Choose text color based on background
                    matrix_range = heatmap_matrix.values.max() - heatmap_matrix.values.min()
                    threshold = heatmap_matrix.values.min() + matrix_range * 0.5
                    text_color = 'white' if value > threshold else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                           color=text_color, fontsize=8)
    
    # Title and labels
    ax.set_title(f'Cluster {cluster_id} - Timepoint × Celltype Accessibility', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Celltype', fontweight='bold')
    ax.set_ylabel('Timepoint', fontweight='bold')
    ax.grid(False)
    
    # Adjust layout to accommodate color blocks
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    return fig


# Example usage:
# Define your ordering and colors outside the function
# celltype_orders = ['neural', 'neural_optic', ...]
# cell_type_color_dict = {'neural': '#e6ab02', 'NMPs': '#8dd3c7', ...}
#
# fig = plot_cluster_heatmap(cluster_profiles, cluster_id=23, 
#                           celltype_orders=celltype_orders,
#                           cell_type_color_dict=cell_type_color_dict,
#                           figsize=(15, 8), cmap='RdBu_r',
#                           vmin=0, vmax=100,
#                           save_path='cluster_23_heatmap.pdf')

# %%
fig = plot_cluster_heatmap(cluster_profiles, cluster_id=25, 
                          celltype_orders=celltype_orders,
                          cell_type_color_dict=cell_type_color_dict,
                          figsize=(15, 8), cmap='cividis',
                          vmin=0, vmax=20,
                          save_path=figpath + f"heatmap_cluster_{cluster_id}.pdf")
plt.show()

# %%
fig = plot_cluster_heatmap(cluster_profiles, cluster_id=25, 
                          celltype_orders=celltype_orders,
                          cell_type_color_dict=cell_type_color_dict,
                          figsize=(15, 8), cmap='cividis',
                          vmin=0, vmax=20,
                          save_path=figpath + f"heatmap_cluster_{cluster_id}.pdf")
plt.show()

# %%
fig = plot_cluster_heatmap(cluster_profiles, cluster_id=23, 
                          celltype_orders=celltype_orders,
                          cell_type_color_dict=cell_type_color_dict,
                          figsize=(15, 8), cmap='cividis',
                          vmin=0, vmax=20,
                          save_path=figpath + f"heatmap_cluster_{cluster_id}.pdf")
plt.show()

# %%
fig = plot_cluster_heatmap(cluster_profiles, cluster_id=2, 
                          celltype_orders=celltype_orders,
                          cell_type_color_dict=cell_type_color_dict,
                          figsize=(15, 8), cmap='cividis',
                          vmin=0, vmax=20,
                          save_path=figpath + f"heatmap_cluster_{cluster_id}.pdf")
plt.show()

# %%
cluster_profiles

# %%
celltype_orders = ['neural', 'neural_optic', 'neural_posterior', 'neural_telencephalon', 'neurons', 'hindbrain',
                   'midbrain_hindbrain_boundary', 'optic_cup', 'spinal_cord', 'differentiating_neurons', 
                   'floor_plate', 'neural_floor_plate', 'enteric_neurons', 'neural_crest', 
                   'somites', 'fast_muscle', 'muscle', 'PSM', 'NMPs', 'tail_bud', 'notochord', 
                   'lateral_plate_mesoderm', 'heart_myocardium', 'hematopoietic_vasculature', 'hemangioblasts',
                   'pharyngeal_arches', 'pronephros',  'hatching_gland', 'endoderm', 'endocrine_pancreas',
                   'epidermis', 'primordial_germ_cells']

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Color palette for celltypes
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'differentiating_neurons': '#bebada',
    'endocrine_pancreas': '#fb8072',
    'endoderm': '#80b1d3',
    'enteric_neurons': '#fdb462',
    'epidermis': '#b3de69',
    'fast_muscle': '#df4b9b',
    'floor_plate': '#d9d9d9',
    'hatching_gland': '#bc80bd',
    'heart_myocardium': '#ccebc5',
    'hemangioblasts': '#ffed6f',
    'hematopoietic_vasculature': '#e41a1c',
    'hindbrain': '#377eb8',
    'lateral_plate_mesoderm': '#4daf4a',
    'midbrain_hindbrain_boundary': '#984ea3',
    'muscle': '#ff7f00',
    'neural': '#e6ab02',
    'neural_crest': '#a65628',
    'neural_floor_plate': '#66a61e',
    'neural_optic': '#999999',
    'neural_posterior': '#393b7f',
    'neural_telencephalon': '#fdcdac',
    'neurons': '#cbd5e8',
    'notochord': '#f4cae4',
    'optic_cup': '#c0c000',
    'pharyngeal_arches': '#fff2ae',
    'primordial_germ_cells': '#f1e2cc',
    'pronephros': '#cccccc',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}

# %%
# Plot Cluster 25 (should show hemangioblasts dominance)
plot_single_cluster(cluster_celltype_profiles, '2',
                    celltype_order=celltype_orders, save_path=None, 
                    use_celltype_colors=True, cell_type_color_dict=cell_type_color_dict)

# %%
plot_cluster_grid(
    cluster_celltype_profiles, 
    celltype_order=celltype_orders,
    figsize=(24, 24),  # Adjust size as needed
    use_celltype_colors=True,
    cell_type_color_dict=cell_type_color_dict,
    save_path="all_clusters_celltype_access_grid.pdf"
)

# %%
# for loop to generate the per-cluster celltype enrichment
for cluster_id in cluster_celltype_profiles.index:
    # generate the bar plot for the mean accessibility of celltypes for each cluster
    plot_single_cluster(cluster_celltype_profiles, cluster_id,
                    celltype_order=celltype_orders, save_path=figpath + f"hist_celltype_access_cluster_{cluster_id}.pdf",
                    use_celltype_colors=True, cell_type_color_dict = cell_type_color_dict)
    plt.show(False)

# %%
classify_accessibility_pattern_comprehensive(cluster_celltype_profiles.loc["23"])
classify_accessibility_pattern_comprehensive(cluster_celltype_profiles.loc["25"])

# %%
# Example 2: Test timepoint entropy analysis  
# timepoint_results = run_metadata_entropy_analysis(
#     adata_peaks_ct_tp,
#     cluster_col='leiden_coarse', 
#     metadata_type='timepoint',
#     min_cells=20,
#     verbose=True
# )


# # Step 1: Create cluster profiles
# cluster_profiles = pef.create_cluster_pseudobulk_profiles(
#     adata_peaks_ct_tp, 'leiden_coarse', min_cells=20
# )

# Step 2: Parse group names
celltype_mapping, timepoint_mapping, celltypes, timepoints = parse_pseudobulk_groups(
    list(cluster_profiles.columns)
)

# Step 3: Aggregate by timepoint
cluster_timepoint_profiles = pef.aggregate_by_metadata(
    cluster_profiles, 'timepoint', timepoint_mapping=timepoint_mapping
)

# Step 4: Analyze patterns
timepoint_analysis = analyze_cluster_accessibility_patterns(
    cluster_timepoint_profiles, 'timepoint'
)

# %%
cluster_timepoint_profiles.head()

# %%
classify_accessibility_pattern_comprehensive(cluster_timepoint_profiles.loc["23"])


# %%
def create_timepoint_order(timepoints):
    """
    Sort labels like '3somites', '10somites' numerically.
    Falls back to lexicographic if no integer is found.
    """
    def key(tp):
        m = re.match(r'^\s*(\d+)\s*somites\s*$', str(tp))
        return (0, int(m.group(1))) if m else (1, str(tp))
    return sorted(timepoints, key=key)

def make_timepoint_palette(timepoints, cmap_name='viridis'):
    """
    Build a {timepoint -> color} dict using a Matplotlib colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    n = max(1, len(timepoints))
    colors = [cmap(i/(n-1) if n > 1 else 0.5) for i in range(n)]
    return {tp: col for tp, col in zip(timepoints, colors)}


# %%
timepoint_order = create_timepoint_order(cluster_timepoint_profiles.columns)
timepoint_order

# %%
cluster_id = "23"
plot_single_cluster_timepoint(
    cluster_timepoint_profiles,
    cluster_id,
    timepoint_order=timepoint_order,
    figsize=(6, 6),
    save_path=None,
    color_by_timepoint=True,
    timepoint_colors=None,
    default_color="#B0B0B0",
    show_legend=False,
    legend_kwargs=None)

# %%
plot_timepoint_grid(
    cluster_timepoint_profiles,
    timepoint_order=timepoint_order,
    figsize=(24, 24),  # Adjust size as needed
    color_by_timepoint=True,
    timepoint_colors=None,  # Will auto-generate using your make_timepoint_palette function
    default_color="#B0B0B0",
    save_path=figpath + "all_clusters_timepoint_access_grid.pdf"
)

# %%
plot_timepoint_grid(
    cluster_timepoint_profiles,
    timepoint_order=timepoint_order,
    figsize=(24, 24),  # Adjust size as needed
    color_by_timepoint=True,
    timepoint_colors=None,  # Will auto-generate using your make_timepoint_palette function
    default_color="#B0B0B0",
    save_path=figpath + "all_clusters_timepoint_access_grid_lin_regress.pdf", show_regression=True,
)

# %%
# for loop to generate the per-cluster celltype enrichment
for cluster_id in cluster_celltype_profiles.index:
    # generate the bar plot for the mean accessibility of celltypes for each cluster
    plot_single_cluster_timepoint(
    cluster_timepoint_profiles,
    cluster_id,
    timepoint_order=timepoint_order,
    figsize=(6, 6),
    save_path=figpath + f"hist_timepoint_access_cluster_{cluster_id}.pdf",
    color_by_timepoint=True,
    timepoint_colors=None,
    default_color="#B0B0B0",
    show_legend=False,
    legend_kwargs=None)


# %% [markdown]
# ## quantify the statistical metrics from the celltype and timepoint groups

# %%
def extract_celltype_metrics(cluster_celltype_profiles):
    """
    Extract celltype accessibility metrics for all clusters.
    
    Parameters:
    -----------
    cluster_celltype_profiles : pd.DataFrame
        cluster-by-celltype matrix
    
    Returns:
    --------
    pd.DataFrame with columns: cluster_id, entropy, dominance, cv
    """
    
    metrics_list = []
    
    for cluster_id in cluster_celltype_profiles.index:
        profile = cluster_celltype_profiles.loc[cluster_id]
        
        # Use your existing comprehensive metrics function
        metrics = compute_comprehensive_accessibility_metrics(profile)
        
        metrics_list.append({
            'cluster_id': cluster_id,
            'entropy': metrics['entropy'],
            'dominance': metrics['dominance'], 
            'cv': metrics['cv']
        })
    
    return pd.DataFrame(metrics_list).set_index('cluster_id')


def extract_timepoint_metrics(cluster_timepoint_profiles, timepoint_order=None):
    """
    Extract timepoint accessibility and regression metrics for all clusters.
    
    Parameters:
    -----------
    cluster_timepoint_profiles : pd.DataFrame
        cluster-by-timepoint matrix
    timepoint_order : list or None
        Optional order for timepoints (should be chronological for meaningful slopes)
    
    Returns:
    --------
    pd.DataFrame with columns: cluster_id, slope, r_squared, rmsd, cv, entropy, dominance
    """
    
    metrics_list = []
    
    for cluster_id in cluster_timepoint_profiles.index:
        profile = cluster_timepoint_profiles.loc[cluster_id]
        
        # Order timepoints if provided
        if timepoint_order is not None:
            ordered_profile = pd.Series(index=timepoint_order, dtype=float)
            for tp in timepoint_order:
                ordered_profile[tp] = profile.get(tp, 0.0)
            profile_to_analyze = ordered_profile
        else:
            profile_to_analyze = profile
        
        # Get basic accessibility metrics (CV, entropy, dominance)
        basic_metrics = compute_comprehensive_accessibility_metrics(profile_to_analyze)
        
        # Get regression metrics
        if len(profile_to_analyze) > 1:
            regression_metrics = fit_temporal_regression(
                list(profile_to_analyze.index), 
                profile_to_analyze.values
            )
        else:
            # Handle single timepoint case
            regression_metrics = {
                'slope': 0.0,
                'r_squared': 0.0,
                'rmsd': 0.0,
                'y_pred': [profile_to_analyze.iloc[0]]
            }
        
        metrics_list.append({
            'cluster_id': cluster_id,
            'slope': regression_metrics['slope'],
            'r_squared': regression_metrics['r_squared'],
            'rmsd': regression_metrics['rmsd'],
            'cv': basic_metrics['cv'],
            'entropy': basic_metrics['entropy'],
            'dominance': basic_metrics['dominance']
        })
    
    return pd.DataFrame(metrics_list).set_index('cluster_id')


# def categorize_temporal_dynamics(timepoint_metrics_df, slope_threshold=0.1, rsquared_threshold=0.3):
#     """
#     Categorize clusters by their temporal dynamics.
    
#     Parameters:
#     -----------
#     timepoint_metrics_df : pd.DataFrame
#         Output from extract_timepoint_metrics()
#     slope_threshold : float
#         Threshold for calling a slope "steady" vs increasing/decreasing
        
#     Returns:
#     --------
#     pd.DataFrame with added 'temporal_category' column
#     """
    
#     df = timepoint_metrics_df.copy()
    
#     def categorize_slope(slope, rsquared):
#         if (abs(slope) < slope_threshold) or (rsquared < rsquared_threshold) :
#             return 'steady'
#         elif slope > 0:
#             return 'increasing'
#         else:
#             return 'decreasing'
    
#     df['temporal_category'] = df['slope'].apply(categorize_slope)
    
#     return df
def categorize_temporal_dynamics(timepoint_metrics_df, slope_threshold=0.1, rsquared_threshold=0.3):
    """
    Categorize clusters by their temporal dynamics using both slope and R-squared thresholds.
    
    Parameters:
    -----------
    timepoint_metrics_df : pd.DataFrame
        Output from extract_timepoint_metrics()
    slope_threshold : float
        Threshold for calling a slope "steady" vs increasing/decreasing
    rsquared_threshold : float
        Minimum R-squared value to consider the linear fit reliable
        
    Returns:
    --------
    pd.DataFrame with added 'temporal_category' column
    """
    
    df = timepoint_metrics_df.copy()
    
    def categorize_slope(row):
        slope = row['slope']
        rsquared = row['r_squared']
        
        # If slope is small OR R-squared is low, classify as steady
        if (abs(slope) < slope_threshold) or (rsquared < rsquared_threshold):
            return 'steady'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    # Use apply with axis=1 to pass entire row to the function
    df['temporal_category'] = df.apply(categorize_slope, axis=1)
    
    # Optional: Add additional info columns for analysis
    df['reliable_fit'] = df['r_squared'] >= rsquared_threshold
    df['significant_slope'] = abs(df['slope']) >= slope_threshold
    
    return df


def summarize_cluster_dynamics(celltype_metrics_df, timepoint_metrics_df):
    """
    Create a combined summary of cluster celltype and timepoint dynamics.
    
    Parameters:
    -----------
    celltype_metrics_df : pd.DataFrame
        Output from extract_celltype_metrics()
    timepoint_metrics_df : pd.DataFrame  
        Output from extract_timepoint_metrics()
        
    Returns:
    --------
    pd.DataFrame with combined metrics
    """
    
    # Merge the dataframes
    combined = pd.merge(
        celltype_metrics_df, 
        timepoint_metrics_df, 
        left_index=True, 
        right_index=True, 
        suffixes=('_celltype', '_timepoint')
    )
    
    # Add categorization
    combined = categorize_temporal_dynamics(combined)
    
    return combined



# %%
# Example usage:
# Extract celltype metrics
celltype_metrics = extract_celltype_metrics(cluster_celltype_profiles)
print("Celltype metrics:")
print(celltype_metrics.head())

# Extract timepoint metrics  
timepoint_metrics = extract_timepoint_metrics(
    cluster_timepoint_profiles, 
    timepoint_order=timepoint_order
)
print("\\nTimepoint metrics:")
print(timepoint_metrics.head())


# %%
plt.hist(celltype_metrics["cv"], bins=15)
plt.xlabel("coefficient of variation (CV)")
plt.ylabel("occurences")
plt.grid(False)
plt.savefig(figpath + "hist_leiden_coarse_celltype_CV.pdf")
plt.show()

# %%
plt.hist(timepoint_metrics["slope"], bins=30)
plt.xlabel("slope (temporal dynamics)")
plt.ylabel("occurences")
plt.grid(False)
plt.savefig(figpath + "hist_leiden_coarse_timepoint_slope.pdf")
plt.show()

# %%
# 1. Slope vs R-squared scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(timepoint_metrics['slope'], timepoint_metrics['r_squared'], 
           c=timepoint_metrics['cv'], cmap='viridis', alpha=0.7)
plt.colorbar(label='CV (timepoint)')
plt.xlabel('Temporal Slope')
plt.ylabel('R-squared')
plt.title('Temporal Dynamics: Slope vs Goodness of Fit')
plt.grid(False)
# plt.savefig(figpath + "")
plt.show()

# %%
# Categorize temporal dynamics
timepoint_metrics_categorized = categorize_temporal_dynamics(timepoint_metrics)
print("\\nTemporal categories:")
print(timepoint_metrics_categorized['temporal_category'].value_counts())

# Combined summary
combined_metrics = summarize_cluster_dynamics(celltype_metrics, timepoint_metrics)
print("\\nCombined metrics:")
print(combined_metrics.head())

# Save to files for downstream analysis
# celltype_metrics.to_csv(figpath + "cluster_celltype_metrics.csv")
# timepoint_metrics_categorized.to_csv(figpath + "cluster_timepoint_metrics.csv") 
# combined_metrics.to_csv(figpath + "cluster_combined_metrics.csv")

# Quick diagnostic plots you can make:

# # 1. Slope vs R-squared scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(timepoint_metrics['slope'], timepoint_metrics['r_squared'], 
#            c=timepoint_metrics['cv'], cmap='viridis', alpha=0.7)
# plt.colorbar(label='CV (timepoint)')
# plt.xlabel('Temporal Slope')
# plt.ylabel('R-squared')
# plt.title('Temporal Dynamics: Slope vs Goodness of Fit')
# plt.grid(False)
# plt.savefig(figpath + "")
# plt.show()

# 2. CV comparison: celltype vs timepoint
plt.figure(figsize=(8, 6))
plt.scatter(combined_metrics['cv_celltype'], combined_metrics['cv_timepoint'], 
           c=combined_metrics['slope'], cmap='RdBu_r', alpha=0.7, vmin=-0.5, vmax=0.5)
plt.colorbar(label='Temporal Slope')
plt.xlabel('CV (Celltype)')
plt.ylabel('CV (Timepoint)')
plt.title('Celltype vs Timepoint Variability')
plt.grid(False)
plt.savefig(figpath + "scatter_CV_ct_tp.pdf")
plt.show()

# 3. Distribution of temporal categories
temporal_counts = timepoint_metrics_categorized['temporal_category'].value_counts()
plt.figure(figsize=(8, 6))
temporal_counts.plot(kind='bar')
plt.title('Distribution of Temporal Dynamics')
plt.ylabel('Number of Clusters')
plt.xticks(rotation=45)
plt.grid(False)
plt.savefig(figpath + "bar_temporal_dynamics_leiden_coarse_classes.pdf")
plt.show()

# %%
timepoint_metrics_categorized

# %%
# Convert the metrics index to match leiden_coarse type
timepoint_metrics_fixed = timepoint_metrics_categorized.copy()
timepoint_metrics_fixed.index = timepoint_metrics_fixed.index.astype(int)

# %%
# map the temporal dynamics
adata_peaks_ct_tp.obs["temporal_dynamics"] = adata_peaks_ct_tp.obs["leiden_coarse"].map(timepoint_metrics_fixed["temporal_category"])
adata_peaks_ct_tp

# %%
manual_colors

# %%
# Manual specification with exact RdBu_r colors
manual_colors_light = {
    'increasing': '#f4a582',    # Light red from RdBu_r
    # 'steady': '#f7f7f7',        # Light grey  
    'steady':'#e0e0e0',
    'decreasing': '#92c5de'     # Light blue from RdBu_r
}

# Alternative light RdBu_r colors
manual_colors_light_alt = {
    'increasing': '#fdbf6f',    # Light salmon-red
    'steady': '#e0e0e0',        # Medium grey
    'decreasing': '#abd9e9'     # Light sky blue
}
sc.pl.umap(adata_peaks_ct_tp, color="temporal_dynamics", palette=manual_colors_light,
           save="_temporal_dynamics.png")

# %% [markdown]
# ## quantitatively classify the peak clusters based on their mean accessibility profiles
# - (1) celltypes: broad vs specific (if specific, which celltype or lineages?)
# - (2) timepoint: either increasing, decreasing, or stable (over the timecourse)
#

# %%
adata_peaks_ct_tp[adata_peaks_ct_tp.obs["leiden_coarse"]==25]

# %%
adata_peaks_ct_tp.obs["leiden_coarse"].unique()

# %%
# Create boolean mask
cluster_id = 25
adata_peaks_ct_tp.obs['specific_cluster'] = adata_peaks_ct_tp.obs['leiden_coarse'] == cluster_id

# Plot with custom colors
sc.pl.umap(adata_peaks_ct_tp, color='specific_cluster', palette={'True': 'steelblue', 'False': 'lightgrey'}, save=f"_cluster_{cluster_id}.png")



# %%
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

def plot_umap_variable_size(adata, cluster_col, cluster_id, 
                           true_color="steelblue", false_color='lightgrey',
                           true_size=1, false_size=0.2, alpha=0.7,
                           figsize=(8, 6), save_path=None):
    """
    Plot UMAP with different point sizes for highlighted vs background clusters.
    Now automatically uses the cluster's original scanpy color if true_color is None.
    
    Parameters:
    -----------
    adata : AnnData
        Your AnnData object
    cluster_col : str
        Column name for clustering (e.g., 'leiden_coarse')
    cluster_id : int or str
        Which cluster to highlight
    true_color : str or None
        Color for highlighted points. If None, uses original scanpy color
    false_color : str
        Color for background points
    true_size, false_size : float
        Point sizes for highlighted and background points
    alpha : float
        Point transparency
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure
    """
    
    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']
    
    # Create boolean mask
    mask = adata.obs[cluster_col] == cluster_id
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot background points (smaller, grey)
    if np.sum(~mask) > 0:  # If there are background points
        ax.scatter(
            umap_coords[~mask, 0], 
            umap_coords[~mask, 1],
            c=false_color,
            s=false_size,
            alpha=alpha * 0.6,  # Make background even more transparent
            edgecolors='none',
            label=f'Other clusters (n={np.sum(~mask)})'
        )
    
    # Plot highlighted points (larger, colored)  
    if np.sum(mask) > 0:  # If there are highlighted points
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1], 
            c=true_color,
            s=true_size,
            alpha=alpha,
            edgecolors='none',
            label=f'Cluster {cluster_id} (n={np.sum(mask)})'
        )
    
    # Styling
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2') 
    ax.set_title(f'Leiden Cluster {cluster_id} Highlighted')
    ax.legend()
    
    # Remove ticks for cleaner look (scanpy style)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Variable size plot saved to: {save_path}")
    
    plt.show()




# %%
# a UMAP with highlithed for the cluster of interest
cluster_id = 25
plot_umap_variable_size(
    adata_peaks_ct_tp,
    cluster_col='leiden_coarse',
    cluster_id=cluster_id,
    true_color=adata_peaks_ct_tp.uns["leiden_coarse_colors"][cluster_id],  # Automatically uses original scanpy color!
    false_color='lightgrey',
    true_size=1,
    false_size=0.2,
    save_path=figpath + f"cluster_{cluster_id}.png"
)

# %%
# define the clusters of interest to highlight with individual UMAPs
clusters_of_interest = [0, 25, #hematopoetic
                        10, # muscle
                        14, # endoderm
                        18, # hatching gland
                        21, # NMPs, tail bud, early descendents of NMPs
                        28, 12, 30, # enteric neurons
                        3, # CNS
                        5, # epidermis
                        34, # neural crest
                        ]

for cluster_id in clusters_of_interest:
    plot_umap_variable_size(
        adata_peaks_ct_tp,
        cluster_col='leiden_coarse',
        cluster_id=cluster_id,
        true_color=adata_peaks_ct_tp.uns["leiden_coarse_colors"][cluster_id],  # Automatically uses original scanpy color!
        false_color='lightgrey',
        true_size=1,
        false_size=0.2,
        save_path=figpath + f"cluster_{cluster_id}.png"
    )



# %%
sc.pl.umap(adata_peaks_ct_tp, color="leiden_coarse")

# %%
# Your cluster 0 will automatically use its original yellow color (#ffff00)
cluster_id = 0
plot_umap_variable_size(
    adata_peaks_ct_tp,
    cluster_col='leiden_coarse',
    cluster_id=cluster_id,
    true_color=adata_peaks_ct_tp.uns["leiden_coarse_colors"][cluster_id],
    false_color='lightgrey',
    true_size=1,
    false_size=0.2,
    save_path=figpath + f"cluster_{cluster_id}.png"
)

# %%

# %%

# %%

# %% [markdown]
# ## Step 4. For each cluster, compute the enriched metadata
# - metadata: celltype, timepoint, lineage, etc.
#

# %% [markdown]
# ### REPEAT this for the timepoints and lineages

# %%
# Example usage for all three annotation types:

# 1. CELLTYPE ENRICHMENT
print("="*80)
print("CELLTYPE ENRICHMENT ANALYSIS")
print("="*80)
celltype_df = enhanced_cluster_enrichment_analysis(
    adata_peaks_ct_tp, 
    cluster_col="leiden_coarse",
    annotation_col="celltype", 
    contrast_col="celltype_contrast",
    annotation_type="celltype"
)
celltype_summary = create_enhanced_summary(celltype_df, "celltype")
add_annotations_to_adata(adata_peaks_ct_tp, celltype_df, "celltype", "leiden_coarse")

# 2. TIMEPOINT ENRICHMENT  
print("\n" + "="*80)
print("TIMEPOINT ENRICHMENT ANALYSIS")
print("="*80)
timepoint_df = enhanced_cluster_enrichment_analysis(
    adata_peaks_ct_tp,
    cluster_col="leiden_coarse", 
    annotation_col="timepoint",
    contrast_col="timepoint_contrast",
    annotation_type="timepoint"
)
timepoint_summary = create_enhanced_summary(timepoint_df, "timepoint")
add_annotations_to_adata(adata_peaks_ct_tp, timepoint_df, "timepoint", "leiden_coarse")

# 3. LINEAGE ENRICHMENT
print("\n" + "="*80)
print("LINEAGE ENRICHMENT ANALYSIS") 
print("="*80)
lineage_df = enhanced_cluster_enrichment_analysis(
    adata_peaks_ct_tp,
    cluster_col="leiden_coarse",
    annotation_col="lineage", 
    contrast_col="timepoint_contrast",  # Using timepoint contrast as proxy
    annotation_type="lineage"
)
lineage_summary = create_enhanced_summary(lineage_df, "lineage")
add_annotations_to_adata(adata_peaks_ct_tp, lineage_df, "lineage", "leiden_coarse")

# Show final results
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"Celltype annotation distribution:")
print(adata_peaks_ct_tp.obs['leiden_coarse_celltype_annotation'].value_counts().head(10))

print(f"\nTimepoint annotation distribution:")
print(adata_peaks_ct_tp.obs['leiden_coarse_timepoint_annotation'].value_counts())

print(f"\nLineage annotation distribution:")
print(adata_peaks_ct_tp.obs['leiden_coarse_lineage_annotation'].value_counts())

# %%

# %%
"""
Peak Cluster Enrichment Analysis Module

This module provides functions for analyzing peak cluster enrichment across different 
metadata types (celltype, timepoint, lineage, peak_type) using contrast-weighted 
Fisher's exact tests.

Author: [Your name]
Date: [Date]
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact


def enhanced_cluster_enrichment_analysis(adata, 
                                        cluster_col="leiden_coarse", 
                                        annotation_col="celltype", 
                                        contrast_col="celltype_contrast",
                                        annotation_type="celltype",
                                        min_peaks_per_cluster=100,
                                        # Enrichment thresholds
                                        high_p_threshold=0.001,
                                        high_or_threshold=5.0,
                                        medium_p_threshold=0.05,
                                        medium_or_threshold=2.0,
                                        # Contrast thresholds (set to 0 for constant contrast)
                                        high_contrast_threshold=3.0,
                                        medium_contrast_threshold=2.0,
                                        # Representation thresholds
                                        good_representation_threshold=30.0,
                                        fair_representation_threshold=15.0,
                                        verbose=True):
    """
    Analyze cluster enrichment for any annotation type with contrast weighting.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object
    cluster_col : str
        Column name for cluster assignments  
    annotation_col : str
        Column name for annotations to test (celltype, timepoint, lineage, etc.)
    contrast_col : str
        Column name for contrast scores to weight by
    annotation_type : str
        Type of annotation for display purposes
    min_peaks_per_cluster : int
        Minimum number of peaks required per cluster
    high_p_threshold : float
        P-value threshold for high confidence
    high_or_threshold : float
        Odds ratio threshold for high confidence
    medium_p_threshold : float
        P-value threshold for medium confidence  
    medium_or_threshold : float
        Odds ratio threshold for medium confidence
    high_contrast_threshold : float
        Contrast threshold for high confidence (set to 0 for constant contrast)
    medium_contrast_threshold : float
        Contrast threshold for medium confidence (set to 0 for constant contrast)
    good_representation_threshold : float
        Percentage threshold for good representation in cluster
    fair_representation_threshold : float
        Percentage threshold for fair representation in cluster
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    pd.DataFrame
        Results dataframe with cluster annotations and statistics
    """
    
    clusters = adata.obs[cluster_col].astype(str)
    annotations = adata.obs[annotation_col].astype(str)
    contrasts = adata.obs[contrast_col].values
    
    # Filter clusters with sufficient peaks
    cluster_counts = clusters.value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_peaks_per_cluster].index
    
    # Sort clusters in numeric order for consistent processing
    try:
        # Try to sort numerically if clusters are numeric
        valid_clusters = sorted(valid_clusters, key=lambda x: int(x))
    except ValueError:
        # Fall back to string sorting if clusters contain non-numeric values
        valid_clusters = sorted(valid_clusters)
    
    if verbose:
        print(f"Annotating {len(valid_clusters)} clusters with ≥{min_peaks_per_cluster} peaks")
        print(f"Using {annotation_type} annotations from '{annotation_col}' column")
        print(f"Using contrast weighting from '{contrast_col}' column")
        print(f"Thresholds: high_OR≥{high_or_threshold}, medium_OR≥{medium_or_threshold}")
    
    cluster_annotations = []
    
    for cluster in valid_clusters:
        if verbose:
            print(f"\n=== CLUSTER {cluster} ===")
        
        cluster_mask = clusters == cluster
        cluster_size = cluster_mask.sum()
        
        # Get annotation composition of this cluster
        cluster_annotation_counts = annotations[cluster_mask].value_counts()
        
        if verbose:
            print(f"Cluster size: {cluster_size} peaks")
            print(f"Top {annotation_type}s with contrast-enhanced scoring:")
        
        annotation_scores = []
        
        # Analyze each annotation in this cluster
        for annotation, count_in_cluster in cluster_annotation_counts.items():
            # Skip very small groups
            if count_in_cluster < 5:
                continue
                
            annotation_mask = annotations == annotation
            
            # Traditional Fisher's exact test
            a = count_in_cluster  # annotation in cluster
            b = annotation_mask.sum() - a  # annotation not in cluster
            c = cluster_size - a  # not annotation in cluster  
            d = len(adata.obs) - cluster_size - b  # not annotation not in cluster
            
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
            
            # Contrast quality metrics
            cluster_annotation_contrasts = contrasts[cluster_mask & annotation_mask]
            mean_contrast = cluster_annotation_contrasts.mean()
            median_contrast = np.median(cluster_annotation_contrasts)
            
            # Count peaks by specificity level
            high_contrast_count = (cluster_annotation_contrasts >= 4).sum()  # Very specific peaks
            moderate_contrast_count = (cluster_annotation_contrasts >= 2).sum()  # Moderately specific peaks
            broad_contrast_count = (cluster_annotation_contrasts < 2).sum()  # Broad peaks
            
            # Calculate contrast bonus (enhances odds ratio based on peak specificity)
            # Higher mean contrast = higher bonus, but cap it to avoid extreme values
            contrast_bonus = 1 + np.tanh(mean_contrast / 5)  # Bonus between 1.0 and ~2.0
            
            # Combined score: traditional enrichment × contrast quality
            combined_score = odds_ratio * contrast_bonus
            
            # Calculate interpretable metrics
            pct_cluster_is_annotation = (a / cluster_size) * 100
            pct_annotation_in_cluster = (a / (a + b)) * 100
            expected_count = (cluster_size * annotation_mask.sum()) / len(adata.obs)
            fold_enrichment = a / expected_count if expected_count > 0 else np.inf
            
            annotation_scores.append({
                f'{annotation_type}': annotation,
                'count': count_in_cluster,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'fold_enrichment': fold_enrichment,
                'mean_contrast': mean_contrast,
                'median_contrast': median_contrast,
                'high_contrast_count': high_contrast_count,
                'moderate_contrast_count': moderate_contrast_count,
                'broad_contrast_count': broad_contrast_count,
                'contrast_bonus': contrast_bonus,
                'combined_score': combined_score,
                'pct_of_cluster': pct_cluster_is_annotation,
                'pct_of_annotation_total': pct_annotation_in_cluster,
                'expected_count': expected_count
            })
            
            if verbose:
                print(f"  {annotation:25} {count_in_cluster:5d} ({pct_cluster_is_annotation:5.1f}%) "
                      f"OR={odds_ratio:5.2f} contrast={mean_contrast:4.1f} "
                      f"combined={combined_score:5.2f} p={p_value:.2e}")
        
        # Sort by combined score (enrichment × contrast quality)
        annotation_scores.sort(key=lambda x: (-x['combined_score'], x['p_value']))
        
        if annotation_scores:
            # Get the top annotation based on combined score
            top_annotation = annotation_scores[0]
            
            # Enhanced confidence scoring that considers both enrichment and contrast
            high_enrichment = (top_annotation['p_value'] < high_p_threshold and 
                             top_annotation['odds_ratio'] > high_or_threshold)
            high_contrast = top_annotation['mean_contrast'] > high_contrast_threshold
            good_representation = top_annotation['pct_of_cluster'] > good_representation_threshold
            
            medium_enrichment = (top_annotation['p_value'] < medium_p_threshold and 
                               top_annotation['odds_ratio'] > medium_or_threshold)
            medium_contrast = top_annotation['mean_contrast'] > medium_contrast_threshold
            fair_representation = top_annotation['pct_of_cluster'] > fair_representation_threshold
            
            if high_enrichment and high_contrast and good_representation:
                best_annotation = top_annotation[f'{annotation_type}']
                confidence = "high"
            elif (high_enrichment and medium_contrast) or (medium_enrichment and high_contrast):
                best_annotation = top_annotation[f'{annotation_type}']
                confidence = "medium"
            elif medium_enrichment and medium_contrast and fair_representation:
                best_annotation = top_annotation[f'{annotation_type}']
                confidence = "low"
            else:
                best_annotation = "mixed"
                confidence = "very_low"
            
            if verbose:
                print(f"\n→ {annotation_type.upper()} ANNOTATION: {best_annotation} ({confidence} confidence)")
                print(f"  Top {annotation_type}: {top_annotation[f'{annotation_type}']}")
                print(f"  Traditional: {top_annotation['fold_enrichment']:.1f}x enriched, OR={top_annotation['odds_ratio']:.2f}")
                print(f"  Contrast: mean={top_annotation['mean_contrast']:.1f}, "
                      f"high-spec peaks={top_annotation['high_contrast_count']}")
                print(f"  Combined score: {top_annotation['combined_score']:.2f} "
                      f"({top_annotation['pct_of_cluster']:.1f}% of cluster)")
            
            cluster_annotations.append({
                'cluster': cluster,
                f'{annotation_type}_annotation': best_annotation,
                f'{annotation_type}_confidence': confidence,
                f'top_{annotation_type}': top_annotation[f'{annotation_type}'],
                'odds_ratio': top_annotation['odds_ratio'],
                'p_value': top_annotation['p_value'],
                'fold_enrichment': top_annotation['fold_enrichment'],
                'mean_contrast': top_annotation['mean_contrast'],
                'median_contrast': top_annotation['median_contrast'],
                'high_contrast_count': top_annotation['high_contrast_count'],
                'moderate_contrast_count': top_annotation['moderate_contrast_count'],
                'broad_contrast_count': top_annotation['broad_contrast_count'],
                'contrast_bonus': top_annotation['contrast_bonus'],
                'combined_score': top_annotation['combined_score'],
                'pct_of_cluster': top_annotation['pct_of_cluster'],
                'cluster_size': cluster_size
            })
        else:
            if verbose:
                print(f"\n→ {annotation_type.upper()} ANNOTATION: unclear (no significant enrichments)")
            
            cluster_annotations.append({
                'cluster': cluster,
                f'{annotation_type}_annotation': 'unclear',
                f'{annotation_type}_confidence': 'none',
                f'top_{annotation_type}': None,
                'odds_ratio': None,
                'p_value': None,
                'fold_enrichment': None,
                'mean_contrast': None,
                'median_contrast': None,
                'high_contrast_count': None,
                'moderate_contrast_count': None,
                'broad_contrast_count': None,
                'contrast_bonus': None,
                'combined_score': None,
                'pct_of_cluster': None,
                'cluster_size': cluster_size
            })
    
    return pd.DataFrame(cluster_annotations)


def create_enhanced_summary(annotation_df, annotation_type="celltype", verbose=True):
    """
    Create enhanced summary showing both enrichment and contrast metrics.
    
    Parameters:
    -----------
    annotation_df : pd.DataFrame
        Results from enhanced_cluster_enrichment_analysis
    annotation_type : str
        Type of annotation for display
    verbose : bool
        Whether to print detailed summary
        
    Returns:
    --------
    pd.DataFrame
        Sorted summary dataframe
    """
    if not verbose:
        return annotation_df.sort_values('cluster')
    
    print("\n" + "="*100)
    print(f"ENHANCED CLUSTER {annotation_type.upper()} ENRICHMENT SUMMARY (with Contrast Weighting)")
    print("="*100)
    
    summary = annotation_df.sort_values('cluster', key=lambda x: pd.to_numeric(x, errors='coerce')).copy()
    
    for _, row in summary.iterrows():
        confidence_symbol = {
            'high': '***',
            'medium': '**', 
            'low': '*',
            'very_low': '·',
            'none': ''
        }.get(row[f'{annotation_type}_confidence'], '')
        
        annotation_col = f'{annotation_type}_annotation'
        if row[annotation_col] not in ['mixed', 'unclear']:
            print(f"Cluster {row['cluster']:2s}: {row[annotation_col]:25s} {confidence_symbol:3s} "
                  f"(OR={row['odds_ratio']:4.1f}, contrast={row['mean_contrast']:4.1f}, "
                  f"combined={row['combined_score']:4.1f}, "
                  f"{row['pct_of_cluster']:4.1f}% of cluster, n={row['cluster_size']})")
            
            # Show contrast composition
            if pd.notna(row['high_contrast_count']):
                print(f"           {'':25s}     "
                      f"Specificity: {row['high_contrast_count']} high, "
                      f"{row['moderate_contrast_count']} moderate, "
                      f"{row['broad_contrast_count']} broad peaks")
        else:
            print(f"Cluster {row['cluster']:2s}: {row[annotation_col]:25s} {confidence_symbol:3s} "
                  f"(n={row['cluster_size']})")
    
    print(f"\nLegend: *** = high confidence, ** = medium, * = low, · = very low")
    print(f"OR = Odds Ratio, contrast = mean {annotation_type} contrast, combined = OR × contrast bonus")
    
    return summary


def add_annotations_to_adata(adata, annotation_df, annotation_type="celltype", cluster_col="leiden_coarse"):
    """
    Add cluster annotations back to the AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object to modify
    annotation_df : pd.DataFrame
        Results from enhanced_cluster_enrichment_analysis
    annotation_type : str
        Type of annotation (celltype, timepoint, lineage, peak_type)
    cluster_col : str
        Column name for cluster assignments
    """
    # Create mapping from cluster to annotation
    annotation_col = f'{annotation_type}_annotation'
    confidence_col = f'{annotation_type}_confidence'
    
    cluster_to_annotation = dict(zip(annotation_df['cluster'], annotation_df[annotation_col]))
    cluster_to_confidence = dict(zip(annotation_df['cluster'], annotation_df[confidence_col]))
    cluster_to_combined_score = dict(zip(annotation_df['cluster'], annotation_df['combined_score']))
    cluster_to_mean_contrast = dict(zip(annotation_df['cluster'], annotation_df['mean_contrast']))
    
    # Map annotations
    clusters = adata.obs[cluster_col].astype(str)
    adata.obs[f'{cluster_col}_{annotation_type}_annotation'] = clusters.map(cluster_to_annotation).fillna('unknown')
    adata.obs[f'{cluster_col}_{annotation_type}_confidence'] = clusters.map(cluster_to_confidence).fillna('none')
    adata.obs[f'{cluster_col}_{annotation_type}_combined_score'] = clusters.map(cluster_to_combined_score).fillna(0)
    adata.obs[f'{cluster_col}_{annotation_type}_mean_contrast'] = clusters.map(cluster_to_mean_contrast).fillna(0)
    
    print(f"Added enhanced {annotation_type} annotation columns to adata.obs:")
    print(f"  - '{cluster_col}_{annotation_type}_annotation'")
    print(f"  - '{cluster_col}_{annotation_type}_confidence'") 
    print(f"  - '{cluster_col}_{annotation_type}_combined_score'")
    print(f"  - '{cluster_col}_{annotation_type}_mean_contrast'")


def run_all_enrichment_analyses(adata, cluster_col="leiden_coarse", 
                               celltype_col="celltype", timepoint_col="timepoint", 
                               lineage_col="lineage", peak_type_col="peak_type",
                               celltype_contrast_col="celltype_contrast",
                               timepoint_contrast_col="timepoint_contrast",
                               constant_contrast_col="constant_contrast",
                               # Universal thresholds
                               high_or_threshold=5.0,
                               medium_or_threshold=2.0,
                               good_representation_threshold=30.0,
                               fair_representation_threshold=15.0,
                               verbose=True):
    """
    Run enrichment analysis for all annotation types with universal thresholds.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object
    cluster_col : str
        Column name for cluster assignments
    celltype_col, timepoint_col, lineage_col, peak_type_col : str
        Column names for different annotation types
    celltype_contrast_col, timepoint_contrast_col, constant_contrast_col : str
        Column names for contrast scores
    high_or_threshold, medium_or_threshold : float
        Universal odds ratio thresholds
    good_representation_threshold, fair_representation_threshold : float
        Universal representation thresholds
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    dict
        Dictionary containing results for all annotation types
    """
    
    results = {}
    
    # 1. CELLTYPE ENRICHMENT
    if verbose:
        print("="*80)
        print("CELLTYPE ENRICHMENT ANALYSIS")
        print("="*80)
    
    results['celltype'] = enhanced_cluster_enrichment_analysis(
        adata, 
        cluster_col=cluster_col,
        annotation_col=celltype_col, 
        contrast_col=celltype_contrast_col,
        annotation_type="celltype",
        high_or_threshold=high_or_threshold,
        medium_or_threshold=medium_or_threshold,
        good_representation_threshold=good_representation_threshold,
        fair_representation_threshold=fair_representation_threshold,
        verbose=verbose
    )
    
    celltype_summary = create_enhanced_summary(results['celltype'], "celltype", verbose)
    add_annotations_to_adata(adata, results['celltype'], "celltype", cluster_col)

    # 2. TIMEPOINT ENRICHMENT  
    if verbose:
        print("\n" + "="*80)
        print("TIMEPOINT ENRICHMENT ANALYSIS")
        print("="*80)
    
    results['timepoint'] = enhanced_cluster_enrichment_analysis(
        adata,
        cluster_col=cluster_col, 
        annotation_col=timepoint_col,
        contrast_col=timepoint_contrast_col,
        annotation_type="timepoint",
        high_or_threshold=high_or_threshold,
        medium_or_threshold=medium_or_threshold,
        good_representation_threshold=good_representation_threshold,
        fair_representation_threshold=fair_representation_threshold,
        verbose=verbose
    )
    
    timepoint_summary = create_enhanced_summary(results['timepoint'], "timepoint", verbose)
    add_annotations_to_adata(adata, results['timepoint'], "timepoint", cluster_col)

    # 3. LINEAGE ENRICHMENT
    if verbose:
        print("\n" + "="*80)
        print("LINEAGE ENRICHMENT ANALYSIS") 
        print("="*80)
    
    results['lineage'] = enhanced_cluster_enrichment_analysis(
        adata,
        cluster_col=cluster_col,
        annotation_col=lineage_col, 
        contrast_col=timepoint_contrast_col,  # Using timepoint contrast as proxy
        annotation_type="lineage",
        high_or_threshold=high_or_threshold,
        medium_or_threshold=medium_or_threshold,
        good_representation_threshold=good_representation_threshold,
        fair_representation_threshold=fair_representation_threshold,
        verbose=verbose
    )
    
    lineage_summary = create_enhanced_summary(results['lineage'], "lineage", verbose)
    add_annotations_to_adata(adata, results['lineage'], "lineage", cluster_col)

    # 4. PEAK TYPE ENRICHMENT (with disabled contrast thresholds)
    if verbose:
        print("\n" + "="*80)
        print("PEAK TYPE ENRICHMENT ANALYSIS")
        print("="*80)
    
    results['peak_type'] = enhanced_cluster_enrichment_analysis(
        adata,
        cluster_col=cluster_col,
        annotation_col=peak_type_col,
        contrast_col=constant_contrast_col,  # Equal contrast for all peaks
        annotation_type="peak_type",
        high_or_threshold=high_or_threshold,
        medium_or_threshold=medium_or_threshold,
        high_contrast_threshold=0.0,  # Disable contrast thresholds
        medium_contrast_threshold=0.0,
        good_representation_threshold=good_representation_threshold,
        fair_representation_threshold=fair_representation_threshold,
        verbose=verbose
    )
    
    peak_type_summary = create_enhanced_summary(results['peak_type'], "peak_type", verbose)
    add_annotations_to_adata(adata, results['peak_type'], "peak_type", cluster_col)

    # Final summary
    if verbose:
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        for ann_type in ['celltype', 'timepoint', 'lineage', 'peak_type']:
            print(f"\n{ann_type.capitalize()} annotation distribution:")
            col_name = f'{cluster_col}_{ann_type}_annotation'
            if col_name in adata.obs.columns:
                print(adata.obs[col_name].value_counts().head(10))
    
    return results


def create_constant_contrast_column(adata, value=1.0):
    """
    Create a constant contrast column for peak_type analysis.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object to modify
    value : float
        Constant value to assign to all peaks
    """
    adata.obs['constant_contrast'] = value
    print(f"Created 'constant_contrast' column with value {value}")


def print_cluster_validation(adata, cluster_id, cluster_col="leiden_coarse"):
    """
    Print validation statistics for a specific cluster across all annotation types.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data object
    cluster_id : str or int
        Cluster ID to validate
    cluster_col : str
        Column name for cluster assignments
    """
    
    cluster_mask = adata.obs[cluster_col].astype(str) == str(cluster_id)
    cluster_size = cluster_mask.sum()
    
    print(f"\n=== CLUSTER {cluster_id} VALIDATION ===")
    print(f"Cluster size: {cluster_size:,} peaks")
    
    # Check all annotation types
    annotation_types = ['celltype', 'timepoint', 'lineage', 'peak_type']
    
    for ann_type in annotation_types:
        ann_col = f'{cluster_col}_{ann_type}_annotation'
        conf_col = f'{cluster_col}_{ann_type}_confidence'
        
        if ann_col in adata.obs.columns:
            annotation = adata.obs.loc[cluster_mask, ann_col].iloc[0]
            confidence = adata.obs.loc[cluster_mask, conf_col].iloc[0]
            
            # Get composition
            original_col = {'celltype': 'celltype', 'timepoint': 'timepoint', 
                           'lineage': 'lineage', 'peak_type': 'peak_type'}[ann_type]
            
            if original_col in adata.obs.columns:
                composition = adata.obs.loc[cluster_mask, original_col].value_counts()
                top_3 = composition.head(3)
                
                print(f"\n{ann_type.capitalize():12}: {annotation} ({confidence})")
                print(f"  Composition: ", end="")
                for i, (item, count) in enumerate(top_3.items()):
                    pct = count / cluster_size * 100
                    print(f"{item}={count}({pct:.1f}%)", end="")
                    if i < len(top_3) - 1:
                        print(", ", end="")
                print()


# Example usage and testing functions
def test_cluster_23_calibration(adata, cluster_col="leiden_coarse"):
    """
    Test that Cluster 23 gets the expected annotations with current thresholds.
    """
    print("="*60)
    print("CLUSTER 23 CALIBRATION TEST")
    print("="*60)
    
    print_cluster_validation(adata, "23", cluster_col)
    
    expected_results = {
        'celltype': 'hemangioblasts (high confidence)',
        'timepoint': 'mixed (very_low confidence)', 
        'peak_type': 'promoter (high confidence)'
    }
    
    print(f"\nExpected results:")
    for ann_type, expected in expected_results.items():
        print(f"  {ann_type:12}: {expected}")


# %%
# Example usage for all three annotation types:

# # 1. CELLTYPE ENRICHMENT
# print("="*80)
# print("CELLTYPE ENRICHMENT ANALYSIS")
# print("="*80)
# celltype_df = enhanced_cluster_enrichment_analysis(
#     adata_peaks_ct_tp, 
#     cluster_col="leiden_coarse",
#     annotation_col="celltype", 
#     contrast_col="celltype_contrast",
#     annotation_type="celltype"
# )
# celltype_summary = create_enhanced_summary(celltype_df, "celltype")
# add_annotations_to_adata(adata_peaks_ct_tp, celltype_df, "celltype", "leiden_coarse")

# 2. TIMEPOINT ENRICHMENT  
print("\n" + "="*80)
print("TIMEPOINT ENRICHMENT ANALYSIS")
print("="*80)
timepoint_df = enhanced_cluster_enrichment_analysis(
    adata_peaks_ct_tp,
    cluster_col="leiden_coarse", 
    annotation_col="timepoint",
    contrast_col="timepoint_contrast",
    annotation_type="timepoint"
)
timepoint_summary = create_enhanced_summary(timepoint_df, "timepoint")
add_annotations_to_adata(adata_peaks_ct_tp, timepoint_df, "timepoint", "leiden_coarse")

# # 3. LINEAGE ENRICHMENT
# print("\n" + "="*80)
# print("LINEAGE ENRICHMENT ANALYSIS") 
# print("="*80)
# lineage_df = enhanced_cluster_enrichment_analysis(
#     adata_peaks_ct_tp,
#     cluster_col="leiden_coarse",
#     annotation_col="lineage", 
#     contrast_col="timepoint_contrast",  # Using timepoint contrast as proxy
#     annotation_type="lineage"
# )
# lineage_summary = create_enhanced_summary(lineage_df, "lineage")
# add_annotations_to_adata(adata_peaks_ct_tp, lineage_df, "lineage", "leiden_coarse")

# # Show final results
# print("\n" + "="*80)
# print("FINAL SUMMARY")
# print("="*80)

# print(f"Celltype annotation distribution:")
# print(adata_peaks_ct_tp.obs['leiden_coarse_celltype_annotation'].value_counts().head(10))

# print(f"\nTimepoint annotation distribution:")
# print(adata_peaks_ct_tp.obs['leiden_coarse_timepoint_annotation'].value_counts())

# print(f"\nLineage annotation distribution:")
# print(adata_peaks_ct_tp.obs['leiden_coarse_lineage_annotation'].value_counts())

# %%

# %%

# %%
# Calculate the default size
default_size = 120000 / len(adata_peaks_ct_tp.obs)
print(f"Default scanpy size: {default_size:.2f}")

# Define confidence levels and corresponding sizes (relative to default)
confidence_sizes = {
    'high': default_size * 1,      # 3x larger than default (~0.57)
    'medium': default_size * 0.6,    # 2x larger than default (~0.38) 
    'very_low': default_size * 0.2, # Half of default (~0.09)
    'low': default_size * 0.4,       # Same as default (~0.19)
    'none': default_size * 0.05     # Very small (~0.06)
}

# Create a sequence of sizes for each cell (same order as adata.obs)
size_sequence = adata_peaks_ct_tp.obs['leiden_coarse_confidence'].map(confidence_sizes).values

# Create a color palette for annotations
unique_annotations = adata_peaks_ct_tp.obs['leiden_coarse_annotation'].unique()
n_colors = len(unique_annotations)

# Use a diverse color palette
if n_colors <= 20:
    colors = sns.color_palette("tab20", n_colors)
else:
    colors = sns.color_palette("hsv", n_colors)

annotation_colors = dict(zip(unique_annotations, colors))

# Special handling for 'mixed' - make it gray
if 'mixed' in annotation_colors:
    annotation_colors['mixed'] = '#CCCCCC'

# Print size mapping
print("\nSize mapping:")
for conf, size in confidence_sizes.items():
    count = (adata_peaks_ct_tp.obs['leiden_coarse_confidence'] == conf).sum()
    if count > 0:
        print(f"  {conf}: {size:.3f} pts, {count:,} peaks")

# Plot using scanpy with size sequence
sc.pl.umap(adata_peaks_ct_tp, 
           color='leiden_coarse_annotation',
           size=size_sequence,  # Pass the sequence directly
           palette=cell_type_color_dict,
           alpha=0.7,
           legend_loc='right margin',
           legend_fontsize=8,
           title='Peak Clusters: Celltype Annotations & Confidence Levels',
           frameon=False,
           save="_peak_cluster_celltype_anno_confidence.png")

# %%
# Calculate the default size
default_size = 120000 / len(adata_peaks_ct_tp.obs)
print(f"Default scanpy size: {default_size:.2f}")

# Define confidence levels and corresponding sizes (relative to default)
confidence_sizes = {
    'high': default_size * 1,      # 3x larger than default (~0.57)
    'medium': default_size * 0.6,    # 2x larger than default (~0.38) 
    'very_low': default_size * 0.2, # Half of default (~0.09)
    'low': default_size * 0.4,       # Same as default (~0.19)
    'none': default_size * 0.05     # Very small (~0.06)
}


# %%

# Create a sequence of sizes for each cell (same order as adata.obs)
size_sequence = adata_peaks_ct_tp.obs['leiden_coarse_timepoint_confidence'].map(confidence_sizes).values

# Create a color palette for annotations
unique_annotations = adata_peaks_ct_tp.obs['leiden_coarse_timepoint_annotation'].unique()
n_colors = len(unique_annotations)

# Plot with viridis colormap for timepoints
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
n_timepoints = len(timepoint_order)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
timepoint_colors = dict(zip(timepoint_order, viridis_colors))
timepoint_colors["mixed"] = "#CCCCCC"
adata_peaks_ct_tp.uns['timepoint_colors'] = [timepoint_colors[t] for t in timepoint_order]

# # Use a diverse color palette
# if n_colors <= 20:
#     colors = sns.color_palette("tab20", n_colors)
# else:
#     colors = sns.color_palette("hsv", n_colors)

# annotation_colors = dict(zip(unique_annotations, colors))

# # Special handling for 'mixed' - make it gray
# if 'mixed' in annotation_colors:
#     annotation_colors['mixed'] = '#CCCCCC'

# Print size mapping
print("\nSize mapping:")
for conf, size in confidence_sizes.items():
    count = (adata_peaks_ct_tp.obs['leiden_coarse_confidence'] == conf).sum()
    if count > 0:
        print(f"  {conf}: {size:.3f} pts, {count:,} peaks")

# Plot using scanpy with size sequence
sc.pl.umap(adata_peaks_ct_tp, 
           color='leiden_coarse_timepoint_annotation',
           size=size_sequence,  # Pass the sequence directly
           palette=timepoint_colors,
           alpha=0.7,
           legend_loc='right margin',
           legend_fontsize=8,
           title='Peak Clusters: timepoint annotation',
           frameon=False,
           save="_peak_cluster_timepoint_anno_confidence.png")

# %%
# Create a sequence of sizes for each cell (same order as adata.obs)
size_sequence = adata_peaks_ct_tp.obs['leiden_coarse_lineage_confidence'].map(confidence_sizes).values

# Create a color palette for annotations
unique_annotations = adata_peaks_ct_tp.obs['leiden_coarse_lineage_annotation'].unique()
n_colors = len(unique_annotations)

# # Use a diverse color palette
# if n_colors <= 20:
#     colors = sns.color_palette("tab20", n_colors)
# else:
#     colors = sns.color_palette("hsv", n_colors)

# annotation_colors = dict(zip(unique_annotations, colors))

# # Special handling for 'mixed' - make it gray
# if 'mixed' in annotation_colors:
#     annotation_colors['mixed'] = '#CCCCCC'

# Print size mapping
print("\nSize mapping:")
for conf, size in confidence_sizes.items():
    count = (adata_peaks_ct_tp.obs['leiden_coarse_lineage_confidence'] == conf).sum()
    if count > 0:
        print(f"  {conf}: {size:.3f} pts, {count:,} peaks")

# Plot using scanpy with size sequence
sc.pl.umap(adata_peaks_ct_tp, 
           color='leiden_coarse_lineage_annotation',
           size=size_sequence,  # Pass the sequence directly
           # palette=cell_type_color_dict,
           alpha=0.7,
           legend_loc='right margin',
           legend_fontsize=8,
           title='Peak Clusters: lineage annotation',
           frameon=False,
           save="_peak_cluster_lineage_anno_confidence.png")

# %%

# %%
sc.pl.umap(adata_peaks_ct_tp, color="leiden_coarse", legend_loc="on data")

# %% [markdown]
# ### PEAK_TYPE enrichment per peak cluster

# %%
# Create a constant contrast column (all peaks have same "contrast")
adata_peaks_ct_tp.obs['constant_contrast'] = 10.0

# PEAK TYPE ENRICHMENT ANALYSIS with equal weighting
print("="*80)
print("PEAK TYPE ENRICHMENT ANALYSIS")
print("="*80)

peak_type_df = enhanced_cluster_enrichment_analysis(
    adata_peaks_ct_tp,
    cluster_col="leiden_coarse",
    annotation_col="peak_type",  # Your peak_type column
    contrast_col="constant_contrast",  # Equal contrast for all peaks
    annotation_type="peak_type"
)

peak_type_summary = create_enhanced_summary(peak_type_df, "peak_type")
add_annotations_to_adata(adata_peaks_ct_tp, peak_type_df, "peak_type", "leiden_coarse")

# Show results
print(f"\nPeak type annotation distribution:")
print(adata_peaks_ct_tp.obs['leiden_coarse_peak_type_annotation'].value_counts())

print(f"\nPeak type confidence distribution:")
print(adata_peaks_ct_tp.obs['leiden_coarse_peak_type_confidence'].value_counts())

# %%
sc.pl.umap(adata_peaks_ct_tp, color="leiden_coarse_peak_type_annotation")

# %% [markdown]
# ### AFTER REVISING THE ODDS_RATIO threshold
#
# - 

# %%
from utils_compute_peak_contrast import *

# %%
# create constant contrast column if needed (for peak_type)
create_constant_contrast_column(adata_peaks_ct_tp, value=10)

# test cluster 23 calibration first
test_cluster_23_calibration(adata_peaks_ct_tp)

# %%
# Run all analyses with universal thresholds
results = run_all_enrichment_analyses(
    adata_peaks_ct_tp,
    cluster_col="leiden_coarse",
    celltype_col="celltype",
    timepoint_col="timepoint", 
    lineage_col="lineage",
    peak_type_col="peak_type",
    celltype_contrast_col="celltype_contrast",
    timepoint_contrast_col="timepoint_contrast",
    lineage_contrast_col="lineage_contrast",
    peak_type_contrast_col="constant_contrast",
    # Universal calibrated thresholds
    high_or_threshold=5.0,
    medium_or_threshold=2.0,
    good_representation_threshold=30.0,
    fair_representation_threshold=15.0
)

# %%

# %% [markdown]
# ## Part 3. peak-cluster level analysis
# - For statistical reasons, we decided to average the pseudobulk profiles for each peak cluster, creating peak cluster-by-psueudobulk group.
# - NOTE to exclude the pseudobulk groups with too few cells (less than 20 cells)
# - For each metadata (i.e., celltype, timepoint, etc.), average the pseudobulk profiles 
# - For each peak cluster, compute the entropy across different metadata (celltype, timepoint, or other metadata)

# %%
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/")

# %%
# Quick test - just run this in your notebook
import module_annotate_peak_clusters as pem

# Test celltype entropy
celltype_results = pem.compute_simple_cluster_accessibility_entropy(
    adata_peaks_ct_tp,
    cluster_col='leiden_coarse', 
    accessibility_cols=None,  # Auto-detect celltype columns
    verbose=True
)

# Check Cluster 23
cluster_23 = celltype_results[celltype_results['cluster'] == '23']
if len(cluster_23) > 0:
    print(f"\nCluster 23 result: {cluster_23.iloc[0]['pattern']}")
    print(f"Entropy: {cluster_23.iloc[0]['entropy']:.3f}")
    print(f"Should be 'broadly_accessible': {'✓' if cluster_23.iloc[0]['pattern'] == 'broadly_accessible' else '✗'}")

# %%

# %%
# Run pseudobulk-based entropy analysis
entropy_results = pem.run_pseudobulk_entropy_analysis(
    adata_peaks_ct_tp,
    cluster_col="leiden_coarse",  # Column with cluster labels
    separator="_",                # Separator in column names (celltype_timepoint)
    metadata_types=["celltype", "timepoint"],
    verbose=True
)

# Access results
celltype_entropy = entropy_results["celltype"]
timepoint_entropy = entropy_results["timepoint"]

# %%

# %%


# Print cluster rankings
print("Clusters with highest celltype entropy (most broad):")
print(celltype_entropy.sort_values(ascending=False).head(10))

print("Clusters with lowest celltype entropy (most specific):")
print(celltype_entropy.sort_values(ascending=True).head(10))


# Example 2: Create and inspect pseudobulk profiles manually
from module_annotate_peak_clusters import create_cluster_pseudobulk_profiles, compute_cluster_entropy_by_metadata

# Create pseudobulk profiles
pseudobulk_profiles = create_cluster_pseudobulk_profiles(
    adata_peaks_ct_tp, 
    cluster_col="leiden_coarse"
)

print(f"Pseudobulk profiles shape: {pseudobulk_profiles.shape}")
print("First few clusters and celltype_timepoint combinations:")
print(pseudobulk_profiles.iloc[:5, :5])

# Compute entropy for specific metadata type
celltype_entropy = compute_cluster_entropy_by_metadata(
    pseudobulk_profiles,
    metadata_type="celltype",
    separator="_"
)

timepoint_entropy = compute_cluster_entropy_by_metadata(
    pseudobulk_profiles,
    metadata_type="timepoint", 
    separator="_"
)

# Create comparison DataFrame
import pandas as pd
entropy_comparison = pd.DataFrame({
    'celltype_entropy': celltype_entropy,
    'timepoint_entropy': timepoint_entropy
})

print("Entropy comparison:")
print(entropy_comparison.head(10))


# Example 3: Visualize entropy results
import matplotlib.pyplot as plt
import seaborn as sns

# Plot entropy distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Celltype entropy distribution
axes[0].hist(celltype_entropy.values, bins=20, alpha=0.7, color='blue')
axes[0].set_xlabel('Celltype Entropy')
axes[0].set_ylabel('Number of Clusters')
axes[0].set_title('Distribution of Celltype Entropy')
axes[0].axvline(celltype_entropy.mean(), color='red', linestyle='--', 
                label=f'Mean: {celltype_entropy.mean():.3f}')
axes[0].legend()

# Timepoint entropy distribution
axes[1].hist(timepoint_entropy.values, bins=20, alpha=0.7, color='green')
axes[1].set_xlabel('Timepoint Entropy')
axes[1].set_ylabel('Number of Clusters')
axes[1].set_title('Distribution of Timepoint Entropy')
axes[1].axvline(timepoint_entropy.mean(), color='red', linestyle='--',
                label=f'Mean: {timepoint_entropy.mean():.3f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('entropy_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot comparing celltype vs timepoint entropy
plt.figure(figsize=(8, 6))
plt.scatter(celltype_entropy, timepoint_entropy, alpha=0.7)
plt.xlabel('Celltype Entropy')
plt.ylabel('Timepoint Entropy') 
plt.title('Celltype vs Timepoint Entropy by Cluster')

# Add diagonal line
max_val = max(celltype_entropy.max(), timepoint_entropy.max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='x=y')
plt.legend()

# Annotate some interesting points
for cluster in entropy_comparison.index:
    ct_ent = celltype_entropy[cluster]
    tp_ent = timepoint_entropy[cluster]
    
    # Annotate clusters with very different entropies
    if abs(ct_ent - tp_ent) > 0.3:
        plt.annotate(f'C{cluster}', (ct_ent, tp_ent), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('entropy_comparison_scatter.png', dpi=300, bbox_inches='tight')
plt.show()


# Example 4: Add entropy results back to AnnData
# Add entropy values as cluster-level annotations
cluster_to_celltype_entropy = dict(celltype_entropy)
cluster_to_timepoint_entropy = dict(timepoint_entropy)

# Map to individual peaks
clusters = adata_peaks_ct_tp.obs["leiden_coarse"].astype(str)
adata_peaks_ct_tp.obs['cluster_celltype_entropy'] = clusters.map(cluster_to_celltype_entropy).fillna(np.nan)
adata_peaks_ct_tp.obs['cluster_timepoint_entropy'] = clusters.map(cluster_to_timepoint_entropy).fillna(np.nan)

print("Added entropy annotations to adata.obs:")
print("- 'cluster_celltype_entropy'")
print("- 'cluster_timepoint_entropy'")


# Example 5: Identify interesting cluster types
# High celltype entropy (celltype-broad)
celltype_broad = celltype_entropy[celltype_entropy > 0.8].index.tolist()
print(f"Celltype-broad clusters (entropy > 0.8): {celltype_broad}")

# Low celltype entropy (celltype-specific)  
celltype_specific = celltype_entropy[celltype_entropy < 0.3].index.tolist()
print(f"Celltype-specific clusters (entropy < 0.3): {celltype_specific}")

# High timepoint entropy (timepoint-broad)
timepoint_broad = timepoint_entropy[timepoint_entropy > 0.8].index.tolist()
print(f"Timepoint-broad clusters (entropy > 0.8): {timepoint_broad}")

# Low timepoint entropy (timepoint-specific)
timepoint_specific = timepoint_entropy[timepoint_entropy < 0.3].index.tolist()  
print(f"Timepoint-specific clusters (entropy < 0.3): {timepoint_specific}")

# Clusters that are celltype-specific but timepoint-broad
ct_specific_tp_broad = [c for c in celltype_specific if c in timepoint_broad]
print(f"Celltype-specific + timepoint-broad: {ct_specific_tp_broad}")

# Clusters that are timepoint-specific but celltype-broad
tp_specific_ct_broad = [c for c in timepoint_specific if c in celltype_broad]
print(f"Timepoint-specific + celltype-broad: {tp_specific_ct_broad}")
"""

# %%
# A module to compute the enrichment of metadata (i.e., celltype) across leiden clusters ("leiden_coarse")
from scipy.stats import fisher_exact, chi2_contingency

def celltype_enrichment_analysis(adata, cluster_col="leiden_coarse", celltype_col="celltype", 
                                min_peaks_per_cluster=100, top_n_celltypes=15):
    """
    Analyze celltype enrichment in leiden clusters
    """
    
    # Get cluster and celltype data
    clusters = adata.obs[cluster_col].astype(str)
    celltypes = adata.obs[celltype_col].astype(str)
    
    # Filter clusters with sufficient peaks
    cluster_counts = clusters.value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_peaks_per_cluster].index
    
    print(f"Analyzing {len(valid_clusters)} clusters with ≥{min_peaks_per_cluster} peaks")
    print(f"Total peaks: {len(adata.obs)}")
    
    # Get top celltypes for analysis
    top_celltypes = celltypes.value_counts().head(top_n_celltypes).index
    print(f"Analyzing top {len(top_celltypes)} celltypes")
    
    # Create contingency tables and compute enrichment
    enrichment_results = []
    
    for cluster in valid_clusters:
        cluster_mask = clusters == cluster
        cluster_size = cluster_mask.sum()
        
        for celltype in top_celltypes:
            celltype_mask = celltypes == celltype
            
            # 2x2 contingency table
            # |           | In Cluster | Not in Cluster |
            # |-----------|------------|----------------|
            # | Celltype  |     a      |       b        |
            # | Not CT    |     c      |       d        |
            
            a = (cluster_mask & celltype_mask).sum()  # celltype in cluster
            b = (celltype_mask & ~cluster_mask).sum()  # celltype not in cluster
            c = (cluster_mask & ~celltype_mask).sum()  # not celltype in cluster
            d = (~cluster_mask & ~celltype_mask).sum()  # not celltype not in cluster
            
            # Skip if no peaks of this celltype in this cluster
            if a == 0:
                continue
                
            # Fisher's exact test
            oddsratio, pvalue = fisher_exact([[a, b], [c, d]])
            
            # Calculate percentages
            pct_cluster_is_celltype = (a / cluster_size) * 100
            pct_celltype_in_cluster = (a / (a + b)) * 100
            
            enrichment_results.append({
                'cluster': cluster,
                'celltype': celltype,
                'n_peaks_celltype_in_cluster': a,
                'cluster_size': cluster_size,
                'pct_cluster_is_celltype': pct_cluster_is_celltype,
                'pct_celltype_in_cluster': pct_celltype_in_cluster,
                'odds_ratio': oddsratio,
                'p_value': pvalue,
                'log2_odds_ratio': np.log2(oddsratio) if oddsratio > 0 else np.nan
            })
    
    # Convert to DataFrame
    enrichment_df = pd.DataFrame(enrichment_results)
    
    # Multiple testing correction (Bonferroni)
    enrichment_df['p_value_adj'] = enrichment_df['p_value'] * len(enrichment_df)
    enrichment_df['p_value_adj'] = enrichment_df['p_value_adj'].clip(upper=1.0)
    
    # Add significance flags
    enrichment_df['significant'] = enrichment_df['p_value_adj'] < 0.05
    enrichment_df['highly_significant'] = enrichment_df['p_value_adj'] < 0.01
    
    return enrichment_df

def plot_enrichment_heatmap(enrichment_df, metric='log2_odds_ratio', 
                           significance_threshold=0.05, figsize=(12, 8)):
    """
    Plot heatmap of celltype enrichment across clusters
    """
    
    # Create pivot table
    pivot_data = enrichment_df.pivot(index='celltype', columns='cluster', values=metric)
    pivot_pval = enrichment_df.pivot(index='celltype', columns='cluster', values='p_value_adj')
    
    # Create significance mask
    sig_mask = pivot_pval > significance_threshold
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(pivot_data, 
                mask=sig_mask,  # Mask non-significant values
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': f'{metric}'},
                ax=ax)
    
    ax.set_title(f'Celltype Enrichment Across Leiden Clusters\n({metric}, adj. p < {significance_threshold})')
    ax.set_xlabel('Leiden Cluster')
    ax.set_ylabel('Cell Type')
    
    plt.tight_layout()
    return fig

def summarize_cluster_identity(enrichment_df, cluster, top_n=5):
    """
    Summarize the cellular identity of a specific cluster
    """
    cluster_data = enrichment_df[enrichment_df['cluster'] == cluster].copy()
    
    if len(cluster_data) == 0:
        print(f"No data for cluster {cluster}")
        return
    
    # Sort by significance and effect size
    cluster_data = cluster_data.sort_values(['p_value_adj', 'log2_odds_ratio'], 
                                          ascending=[True, False])
    
    print(f"\n=== CLUSTER {cluster} IDENTITY ===")
    print(f"Total peaks in cluster: {cluster_data.iloc[0]['cluster_size']}")
    
    print(f"\nTop {top_n} enriched celltypes:")
    for i, (_, row) in enumerate(cluster_data.head(top_n).iterrows()):
        sig_flag = "***" if row['highly_significant'] else "**" if row['significant'] else "*"
        print(f"{i+1}. {row['celltype']:25} "
              f"OR={row['odds_ratio']:.2f} "
              f"({row['pct_cluster_is_celltype']:.1f}% of cluster) "
              f"p_adj={row['p_value_adj']:.2e} {sig_flag}")



# %%
# Run the analysis
print("Performing celltype enrichment analysis...")
enrichment_df = celltype_enrichment_analysis(adata_peaks_ct_tp, cluster_col="leiden_coarse", celltype_col="celltype", 
                                min_peaks_per_cluster=100, top_n_celltypes=32)

# Show overall statistics
print(f"\nFound {enrichment_df['significant'].sum()} significant enrichments out of {len(enrichment_df)} tests")
print(f"Highly significant: {enrichment_df['highly_significant'].sum()}")

# Plot heatmap
fig = plot_enrichment_heatmap(enrichment_df)
plt.savefig(figpath + "celltype_enrichment_heatmap.pdf", dpi=600, bbox_inches='tight')
plt.show()

# Show top enrichments overall
print("\n=== TOP 10 MOST SIGNIFICANT ENRICHMENTS ===")
top_enrichments = enrichment_df.sort_values(['p_value_adj', 'log2_odds_ratio'], 
                                          ascending=[True, False]).head(10)

for i, (_, row) in enumerate(top_enrichments.iterrows()):
    print(f"{i+1}. Cluster {row['cluster']} → {row['celltype']}")
    print(f"   OR={row['odds_ratio']:.2f}, {row['pct_cluster_is_celltype']:.1f}% of cluster, p_adj={row['p_value_adj']:.2e}")

# Analyze specific clusters (you can modify these cluster IDs)
example_clusters = ['0', '1', '2', '3', '4']  # Adjust based on your actual cluster IDs
for cluster in example_clusters:
    if cluster in enrichment_df['cluster'].values:
        summarize_cluster_identity(enrichment_df, cluster)

print("\n=== ANALYSIS COMPLETE ===")
print("The heatmap shows log2(odds ratio) for significant enrichments.")
print("Red = enriched, Blue = depleted, White/Gray = not significant")

# %%

# %%

# %%

# %%
