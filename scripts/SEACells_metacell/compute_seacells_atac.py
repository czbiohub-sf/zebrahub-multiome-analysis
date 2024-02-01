# This script computes MetaCell for scATAC-seq data using SEACells (Persad et al., 2023)
# Note that this script should be run under "seacells" conda environment
# bash script to activate the conda environment
# module load anaconda
# conda activate seacells

# Input Arguments:
# input_path: a filepath to an input anndata object with scATAC-seq data (h5ad format)
# output_path: a filepath to save the output
# filename: name of the file.
# save_figure: an optional argument to save the EDA plot or not.
# figpath: path for the plots/figures

# examples:
# input_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/TDR118_processed_peaks_merged.h5ad" 
# output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/"
# filename = "TDR118_ATAC_SEACells.h5ad"

# Output(s): anndata object(s) with MetaCell scores (h5ad format)
# NOTE. We currently use only the "adata" object for downstream analyses, not the rest.
# 1) adata: original object with adata.obs["SEACell"] for SEACells assignments
# 2) (Optional) SEACell_ad: an anndata object with aggregated counts over SEACells
# 3) (Optional) SEACell_soft_ad: an anndata object with aggregated counts over soft SEACells

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 4) {
  stop("Usage: python compute_seacells_atac.py input_path output_path filename figpath")
}

input_path <- args[1]  # a filepath to an input anndata object with scATAC-seq data (h5ad format)
output_path <- args[2] # a filepath to save the output
filename <- args[3] # name of the output file.
figpath <- args[4] # path for the plots/figures

# import libraries
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import SEACells

# plotting modules
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# Some plotting aesthetics
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100

# Figure setup (a large figure containing all plots together)
# Initialize the figure and subplots
fig, axs = plt.subplots(5, 2, figsize=(15, 20))  # Adjust the size as needed

# Step 1. load the data
adata = sc.read(input_path)
print(adata)

# optional. cleaning up the adata.obs fields
# List of columns to keep (those not starting with "prediction")
columns_to_keep = [col for col in adata.obs.columns if not col.startswith("prediction")]

# Subset the adata.obs with the columns to keep
adata.obs = adata.obs[columns_to_keep]


# Step 2. pre-proecessing
# NOTE. Make sure that we have "raw" counts for SEACells (adata.X)
# NOTE. The Seurat/Signac pipeline results in a RDS file. Seurat-disk library's Convert function saves "data" to adata.X, and "counts" to adata.raw.X
# So, we will recover the raw counts to adata.X
adata.X = adata.raw.X.copy()

# (optional) check if the raw counts are copied correctly
try:
    assert_raw_counts(adata)
    print("adata.X contains raw counts.")
except ValueError as e:
    print(f"Warning: {e}")


# Step 3. compute Metacells
# First, as a rule of thumb, we follow the recommendation by the SEACells authors,
# which is assigning one metacell for every 75 single-cells.
## User defined parameters

## Core parameters 
n_SEACells = np.floor((adata.n_obs)/75)
build_kernel_on = 'X_lsi' # key in ad.obsm to use for computing metacells
                          # This would be replaced by 'X_pca' for RNA data

## Additional parameters
n_waypoint_eigs = 10 # Number of eigenvalues to consider when initializing metacells

# set up the model
model = SEACells.core.SEACells(adata, 
                  build_kernel_on=build_kernel_on, 
                  n_SEACells=n_SEACells, 
                  n_waypoint_eigs=n_waypoint_eigs,
                  convergence_epsilon = 1e-5)

# run the model
model.construct_kernel_matrix()
M = model.kernel_matrix
sns.clustermap(M.toarray()[:500,:500], ax=axs[0, 0])

# Initialize archetypes
model.initialize_archetypes()

# Plot the initilization to ensure they are spread across phenotypic space
SEACells.plot.plot_initialization(adata, model,plot_basis="X_umap.atac")

model.fit(min_iter=10, max_iter=100)
# Check for convergence 
model.plot_convergence()

# a histogram of the metacell sizes (number of cells per metacell)
axs[1, 0].hist(adata.obs.SEACell.value_counts())
axs[1, 0].xlabel("cell counts/metacell")
axs[1, 0].ylabel("counts")

# SEACElls model QC metric
plt.figure(figsize=(3,2))
sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[1, 1])
axs[1, 1].set_title(f'Non-trivial (> 0.1) assignments per cell')
axs[1, 1].set_xlabel('# Non-trivial SEACell Assignments')
axs[1, 1].set_ylabel('# Cells')
# plt.show()

plt.figure(figsize=(3,2))
b = np.partition(model.A_.T, -5)    
sns.heatmap(np.sort(b[:, -5:])[:, ::-1], cmap='viridis', vmin=0, ax=axs[2, 0])
axs[2, 0].set_title('Strength of top 5 strongest assignments')
axs[2, 0].set_xlabel('$n^{th}$ strongest assignment')
# plt.show()

# Step 4. Summarize the result (aggregating the count matrices by SEACells)
# summarize the data by SEACell (aggregating the count matrices by SEACells)
SEACell_ad = SEACells.core.summarize_by_SEACell(adata, SEACells_label='SEACell', summarize_layer='raw')
SEACell_ad

# (Alternative) summarize the data by soft SEACell (aggregating the count matrices by SEACells
SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(adata, model.A_, celltype_label='global_annotation',summarize_layer='raw', minimum_weight=0.05)
SEACell_soft_ad

# Step 5. visualize the results
# Plot the metacell assignments
SEACells.plot.plot_2D(adata, key='X_umap.atac', colour_metacells=False, ax=axs[2, 1])
SEACells.plot.plot_2D(adata, key='X_umap.atac', colour_metacells=True, ax=axs[3, 0])

# Plot the metacell sizes
SEACells.plot.plot_SEACell_sizes(adata, bins=5, ax=axs[3, 1])

# Step 6. Quantifying the results (celltype_purity, compactness, etc.)
# Compute the celltype purity
SEACell_purity = SEACells.evaluate.compute_celltype_purity(adata, 'global_annotation')

#plt.figure(figsize=(4,4))
sns.boxplot(data=SEACell_purity, y='global_annotation_purity', ax=axs[4, 0])
axs[4, 0].set_title('Celltype Purity')
sns.despine(ax=axs[4, 0])
plt.show()
plt.close()

# compute the compactness
compactness = SEACells.evaluate.compactness(adata, 'X_lsi')

sns.boxplot(data=compactness, y='compactness', ax=axs[4, 1])
axs[4, 1].set_title('Compactness')
sns.despine(ax=axs[4, 1])
# plt.close()

# compute the separation
separation = SEACells.evaluate.separation(adata, 'X_lsi',nth_nbr=1)

# plt.figure(figsize=(4,4))
sns.boxplot(data=separation, y='separation', ax=axs[4, 2])
axs[4, 1].set_title('Separation')
sns.despine(ax=axs[4, 1])

# Adjust spacing between plots
plt.tight_layout()

# Step 7. Save the results
adata.write_h5ad(output_path + filename)

# Save the entire figure
plt.savefig(figpath + 'combined_plots.pdf')
plt.savefig(figpath + 'combined_plots.png')

def assert_raw_counts(adata):
    # Check if the data is stored in a sparse matrix format
    if sp.issparse(adata.X):
        # Convert to dense format to work with numpy
        data = adata.X.todense()
    else:
        data = adata.X

    # Check if all values in the data are integers
    if not np.all(np.rint(data) == data):
        raise ValueError("adata.X does not contain raw counts (integer values).")