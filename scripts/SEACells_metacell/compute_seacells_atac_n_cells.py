# This script computes MetaCell for scATAC-seq data using SEACells (Persad et al., 2023)
# Note that this script should be run under "seacells" conda environment
# bash script to activate the conda environment
# module load anaconda
# conda activate seacells

# import libraries
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import SEACells
import os

# plotting modules
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

## Input arguments
# examples:
# input_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/TDR118_processed_peaks_merged.h5ad" 
# output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/"
# data_id = "TDR118_ATAC" # filename = f"{data_id}_SEACells.h5ad"
# annotation_class = "global_annotation"
# figpath = ""
# metadata_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/master_rna_atac_metadata.csv" 
# NOTE. we'll filter out the "low_quality_cells" from each timepoint/dataset, as they will not be helpful for computing the metacells

# Output(s): anndata object(s) with MetaCell scores (h5ad format)
# NOTE. We currently use only the "adata" object for downstream analyses, not the rest.
# 1) adata: original object with adata.obs["SEACell"] for SEACells assignments
# 2) (Optional) SEACell_ad: an anndata object with aggregated counts over SEACells
# 3) (Optional) SEACell_soft_ad: an anndata object with aggregated counts over soft SEACells
# 4) (Optional) SEACell_purity: a dataframe with celltype purity scores

# Parse command-line arguments
import argparse

# a syntax for running the python script as the main program (not in a module)
#if __name__ == "__main__":


# Set up argument parser
parser = argparse.ArgumentParser(description='Parse command-line arguments for the script.')

# Define the command-line arguments
parser.add_argument('input_path', type=str, help='A filepath to an input anndata object with scATAC-seq data (h5ad format)')
parser.add_argument('output_path', type=str, help='A filepath to save the output')
parser.add_argument('data_id', type=str, help='Name of the output file.')
parser.add_argument('annotation_class', type=str, help='Annotation class for the cell type assignment')
parser.add_argument('figpath', type=str, help='Path for the plots/figures')
# add the n_cells (for the number of cells per SEACells)
parser.add_argument('n_cells', type=int, help='Number of cells per SEACells')
#parser.add_argument('metadata_path', type=str, help='Path for the metadata file (adata.obs)')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
input_path = args.input_path
output_path = args.output_path
data_id = args.data_id
annotation_class = args.annotation_class
figpath = args.figpath
n_cells = args.n_cells
#metadata_path = args.metadata_path

# create the figpath if it doesn't exist yet
os.makedirs(figpath, exist_ok=True)

# Some plotting aesthetics
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100

# Figure setup (a large figure containing all plots together)
# Initialize the figure and subplots
fig, axs = plt.subplots(5, 2, figsize=(15, 20))  # Adjust the size as needed

# define utility functions
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

# SEACells.plot.plot_2D modified
def plot_2D_modified(
    ad,
    ax,
    key="X_umap",
    colour_metacells=True,
    title="Metacell Assignments",
    palette="Set2",
    SEACell_size=20,
    cell_size=10,
):
    """Plot 2D visualization of metacells using the embedding provided in 'key'.

    :param ad: annData containing 'Metacells' label in .obs
    :param ax: Axes object where the plot will be drawn
    :param key: (str) 2D embedding of data. Default: 'X_umap'
    :param colour_metacells: (bool) whether to colour cells by metacell assignment. Default: True
    :param title: (str) title for figure
    :param palette: (str) matplotlib colormap for metacells. Default: 'Set2'
    :param SEACell_size: (int) size of SEACell points
    :param cell_size: (int) size of cell points
    """
    umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs["SEACell"])
    umap["SEACell"] = umap["SEACell"].astype("category")
    mcs = umap.groupby("SEACell").mean().reset_index()

    if colour_metacells:
        sns.scatterplot(
            x=0, y=1, hue="SEACell", data=umap, s=cell_size, palette=sns.color_palette(palette), legend=None, ax=ax
        )
        sns.scatterplot(
            x=0,
            y=1,
            s=SEACell_size,
            hue="SEACell",
            data=mcs,
            palette=sns.color_palette(palette),
            edgecolor="black",
            linewidth=1.25,
            legend=None,
            ax=ax
        )
    else:
        sns.scatterplot(
            x=0, y=1, color="grey", data=umap, s=cell_size, legend=None, ax=ax
        )
        sns.scatterplot(
            x=0,
            y=1,
            s=SEACell_size,
            color="red",
            data=mcs,
            edgecolor="black",
            linewidth=1.25,
            legend=None,
            ax=ax
        )

    ax.set_xlabel(f"{key}-0")
    ax.set_ylabel(f"{key}-1")
    ax.set_title(title)
    ax.set_axis_off()

# SEACells.plot.plot_SEACell_sizes modified
def plot_SEACell_sizes_modified(
    ad,
    ax,
    save_as=None,
    show=True,
    title="Distribution of Metacell Sizes",
    bins=None,
    figsize=(5, 5),
):
    """Plot distribution of number of cells contained per metacell.

    :param ad: annData containing 'Metacells' label in .obs
    :param ax: Axes object where the plot will be drawn
    :param save_as: (str) path to which figure is saved. If None, figure is not saved.
    :param show: (bool) whether to show the plot
    :param title: (str) title of figure.
    :param bins: (int) number of bins for histogram
    :param figsize: (int,int) tuple of integers representing figure size
    :return: None.
    """
    assert "SEACell" in ad.obs, 'AnnData must contain "SEACell" in obs DataFrame.'
    label_df = ad.obs[["SEACell"]].reset_index()
    
    sns.histplot(label_df.groupby("SEACell").count().iloc[:, 0], bins=bins, ax=ax)
    sns.despine()
    ax.set_xlabel("Number of Cells per SEACell")
    ax.set_title(title)

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()
    return pd.DataFrame(label_df.groupby("SEACell").count().iloc[:, 0]).rename(
        columns={"index": "size"}
    )

# Step 1. load the data
# NOTE. make sure to check this section before running the script (file formats)
dataset_name = data_id.replace("reseq", "")

# # NOTE. The original adata object should contain the "X_lsi" embedding
# adata = sc.read_h5ad(input_path + f"{data_id}/{dataset_name}_processed_peaks_merged.h5ad")
adata = sc.read_h5ad(input_path + f"{dataset_name}_nmps_manual_annotation.h5ad")
print(adata)

# load the original adata ("X_lsi_integrated")
# import the master_metadata (cell annotations, cell_ids from the integrated object)
master_metadata = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/master_rna_atac_metadata.csv", index_col=0)

# import the "X_lsi_integrated" from the integrated object
lsi_integrated = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_lsi.csv", index_col=0)

# filter out the "low_quality_cells"
lsi_integrated = lsi_integrated[lsi_integrated.index.isin(master_metadata.index)]

# subset the metadata (per dataset)
metadata_sub = master_metadata[master_metadata.dataset==dataset_name]

# subset integrated_lsi (per dataset - using the indices from the above)
lsi_integrated_sub = lsi_integrated[lsi_integrated.index.isin(metadata_sub.index)]
lsi_integrated_sub = lsi_integrated_sub[lsi_integrated_sub.index.isin(adata.obs_names)]

# drop the 1st LSI component
lsi_integrated_sub_filtered = lsi_integrated_sub.drop("integratedlsi_1", axis=1)
lsi_integrated_sub_filtered

# Reorder lsi components to match adata.obs_names index
lsi_integrated_sub_filtered = lsi_integrated_sub_filtered.reindex(adata.obs.index)
lsi_integrated_sub_filtered.head()

# reformatg the adata.obs_names to match the original adata object
# adata.obs_names = [s.split('_')[0] for s in adata.obs_names]
# adata.obs.head()

# copy the "X_lsi" embedding from the original adata object
adata.obsm["X_lsi_integrated"] = lsi_integrated_sub_filtered.iloc[:,0:39].to_numpy()

# optional. cleaning up the adata.obs fields
# List of columns to keep (those not starting with "prediction")
columns_to_keep = [col for col in adata.obs.columns if not col.startswith("prediction")]

# Subset the adata.obs with the columns to keep
adata.obs = adata.obs[columns_to_keep]
print(adata)


# Step 2. pre-proecessing
# NOTE. Make sure that we have "raw" counts for SEACells (adata.X)
# NOTE. The Seurat/Signac pipeline results in a RDS file. Seurat-disk library's Convert function saves "data" to adata.X, and "counts" to adata.raw.X
# So, we will recover the raw counts to adata.X
# adata.X = adata.raw.X.copy()

# check the "raw" counts are saved in either layers or .X
if "counts" in adata.layers:
    adata.X = adata.layers["counts"]
else:
    adata.X = adata.raw.X.copy()

# check if the raw counts are copied correctly
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
n_SEACells = np.floor((adata.n_obs)/n_cells)
build_kernel_on = 'X_lsi_integrated' # key in ad.obsm to use for computing metacells
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

# plot the clustered heatmap of the kernel matrix
# sns.clustermap(M.toarray()[:500,:500], ax=axs[0, 0])
# Plot clustermap separately and extract the figure
# clustermap = sns.clustermap(M.toarray()[:500,:500])
# # Manually position the clustermap into the subplot layout
# clustermap_ax = clustermap.ax_heatmap
# clustermap_ax.set_position([axs[0, 0].get_position().x0, axs[0, 0].get_position().y0, axs[0, 0].get_position().width, axs[0, 0].get_position().height])
# axs[0, 0].remove()  # Remove the placeholder subplot

# Initialize archetypes
model.initialize_archetypes()

# Plot the initilization to ensure they are spread across phenotypic space
# NOTE. This embedding/basis choice should be the input argument
SEACells.plot.plot_initialization(adata, model,plot_basis="X_umap_aligned")

model.fit(min_iter=10, max_iter=100)
# Check for convergence 
model.plot_convergence()

# a histogram of the metacell sizes (number of cells per metacell)
axs[0, 1].hist(adata.obs.SEACell.value_counts())
axs[0, 1].set_xlabel("cell counts/metacell")
axs[0, 1].set_ylabel("counts")

# SEACElls model QC metric
# plt.figure(figsize=(3,2))
sns.distplot((model.A_.T > 0.1).sum(axis=1), kde=False, ax=axs[1, 0])
axs[1, 0].set_title(f'Non-trivial (> 0.1) assignments per cell')
axs[1, 0].set_xlabel('# Non-trivial SEACell Assignments')
axs[1, 0].set_ylabel('# Cells')
# plt.show()

# plt.figure(figsize=(3,2))
b = np.partition(model.A_.T, -5)    
sns.heatmap(np.sort(b[:, -5:])[:, ::-1], cmap='viridis', vmin=0, ax=axs[1, 1])
axs[1, 1].set_title('Strength of top 5 strongest assignments')
axs[1, 1].set_xlabel('$n^{th}$ strongest assignment')
# plt.show()

# Step 4. Summarize the result (aggregating the count matrices by SEACells)
# summarize the data by SEACell (aggregating the count matrices by SEACells)
# SEACell_ad = SEACells.core.summarize_by_SEACell(adata, SEACells_label='SEACell', summarize_layer='raw')
# SEACell_ad

# # (Alternative) summarize the data by soft SEACell (aggregating the count matrices by SEACells
# SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(adata, model.A_, celltype_label='global_annotation',summarize_layer='raw', minimum_weight=0.05)
# SEACell_soft_ad


# Step 5. visualize the results
# Plot the metacell assignments without coloring metacells
plot_2D_modified(adata, ax=axs[2, 0], key='X_umap_aligned', colour_metacells=False)
# Plot the metacell assignments with coloring metacells
plot_2D_modified(adata, ax=axs[2, 1], key='X_umap_aligned', colour_metacells=True)
# Plot the metacell sizes
plot_SEACell_sizes_modified(adata, ax=axs[3, 0], bins=5)


# # Step 6. Quantifying the results (celltype_purity, compactness, etc.)
# # Compute the celltype purity
# SEACell_purity = SEACells.evaluate.compute_celltype_purity(adata, annotation_class)

# #plt.figure(figsize=(4,4))
# sns.boxplot(data=SEACell_purity, y='global_annotation_purity', ax=axs[3, 1])
# axs[3, 1].set_title('Celltype Purity')
# sns.despine(ax=axs[3, 1])
# plt.show()
# plt.close()

# # compute the compactness
# compactness = SEACells.evaluate.compactness(adata, 'X_lsi')

# sns.boxplot(data=compactness, y='compactness', ax=axs[4, 0])
# axs[4, 0].set_title('Compactness')
# sns.despine(ax=axs[4, 0])
# # plt.close()

# # compute the separation
# separation = SEACells.evaluate.separation(adata, 'X_lsi',nth_nbr=1)

# # plt.figure(figsize=(4,4))
# sns.boxplot(data=separation, y='separation', ax=axs[4, 1])
# axs[4, 1].set_title('Separation')
# sns.despine(ax=axs[4, 1])
# Step 6. Quantifying the results (celltype_purity, compactness, etc.)
# Compute the celltype purity
SEACell_purity = SEACells.evaluate.compute_celltype_purity(adata, annotation_class)

sns.boxplot(data=SEACell_purity, y=f'{annotation_class}_purity', ax=axs[3, 1])
axs[3, 1].set_title('Celltype Purity')
sns.despine(ax=axs[3, 1])

# compute the compactness
compactness = SEACells.evaluate.compactness(adata, 'X_lsi_integrated')

sns.boxplot(data=compactness, y='compactness', ax=axs[4, 0])
axs[4, 0].set_title('Compactness')
sns.despine(ax=axs[4, 0])

# compute the separation
separation = SEACells.evaluate.separation(adata, 'X_lsi_integrated', nth_nbr=1)

sns.boxplot(data=separation, y='separation', ax=axs[4, 1])
axs[4, 1].set_title('Separation')
sns.despine(ax=axs[4, 1])

# Adjust spacing between plots using subplots_adjust
# plt.subplots_adjust(wspace=0.3, hspace=0.5)

# Adjust spacing between plots
fig.tight_layout()

# Step 7. Save the results
adata.write_h5ad(output_path + f"{dataset_name}_seacells_{annotation_class}_{n_cells}cells.h5ad")
# export the adata.obs
adata.obs.to_csv(output_path + f"{dataset_name}_seacells_obs_{annotation_class}_{n_cells}cells.csv")

# Save the entire figure
fig.savefig(figpath + f"combined_plots_{dataset_name}_{annotation_class}_{n_cells}cells.pdf")
fig.savefig(figpath + f"combined_plots_{dataset_name}_{annotation_class}_{n_cells}cells.png")