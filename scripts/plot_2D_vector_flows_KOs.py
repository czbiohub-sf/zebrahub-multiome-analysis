# A python script to generate a series of 2D vector flow plots for the list of KOs
# last updated: 08/30/2024
# conda activate celloracle_env

# Import public libraries
import os
from scipy.stats import binned_statistic_2d
from scipy.sparse import csr_matrix
from scipy import sparse
import scipy.sparse as sp
# from scipy.spatial.distance import cosine

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
# from tqdm.notebook import tqdm
import celloracle as co

# #plt.rcParams["font.family"] = "arial"
# plt.rcParams["figure.figsize"] = [5,5]
# # %config InlineBackend.figure_format = 'retina'
# plt.rcParams["savefig.dpi"] = 300
# plt.rcParams['pdf.fonttype']=42

import logging

# Set the logging level to WARN, filtering out informational messages
logging.getLogger().setLevel(logging.WARNING)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Set the default font to Arial
mpl.rcParams['font.family'] = 'Arial'

# If Arial is not available on your system, you might need to specify an alternative or ensure Arial is installed.
# On some systems, you might need to use 'font.sans-serif' as a fallback option:
# mpl.rcParams['font.sans-serif'] = 'Arial'

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number for editable text in PDFs
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')

# Plotting style function (run this before plotting the final figure)
def set_plotting_style():
    plt.style.use('seaborn-paper')
    plt.rc('axes', labelsize=12)
    plt.rc('axes', titlesize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=2)
    plt.rc('mathtext', fontset='stixsans', sf='sansserif')
    plt.rc('figure', figsize=[10,9])
    plt.rc('svg', fonttype='none')

    # Override any previously set font settings to ensure Arial is used
    plt.rc('font', family='Arial')

# define the list of datasets
list_datasets = ["TDR126","TDR127","TDR128","TDR118",
                "TDR119","TDR125","TDR124"]

# define the list of KOs
list_KO_genes = ["meox1","cdx4","tbx16","myod1","myf5","myog",
                "pax6a","pax6b","sox3","sox19","meis1a","meis1b",
                "hoxc3a","hoxc3b","meis2a","meis3","nr2f5","pax3b","rxraa"]

# define the utils function
# Color palette for cell types
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'fast_muscle': '#df4b9b',
    'neural_posterior': '#393b7f',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}

# computing the average of 2D transition vectors @Metacell level
def average_2D_trans_vecs_metacells(adata, metacell_col="SEACell", 
                                    basis='umap_aligned',key_added='WT'):
    X_umap = adata.obsm[f'X_{basis}']
    # Your cell-level 2D transition vectors
    # V_cell = adata.obsm[f'{key_added}_{basis}'] 
    V_cell = adata.obsm[f"{key_added}"]

    # metacells = adata.obs[metacell_col].cat.codes
    
    # Convert metacell column to categorical if it's not already
    if not pd.api.types.is_categorical_dtype(adata.obs[metacell_col]):
        metacells = pd.Categorical(adata.obs[metacell_col])
    else:
        metacells = adata.obs[metacell_col]
    # number of metacells    
    n_metacells = len(metacells.categories)
    
    # X_metacell is the average UMAP position of the metacells
    # V_metacell is the average transition vector of the metacells
    X_metacell = np.zeros((n_metacells, 2))
    V_metacell = np.zeros((n_metacells, 2))
    
    for i, category in enumerate(metacells.categories):
        mask = metacells == category
        X_metacell[i] = X_umap[mask].mean(axis=0)
        V_metacell[i] = V_cell[mask].mean(axis=0)
    
    return X_metacell, V_metacell

# plotting function for the averaged trans_vectors
# metacell_col = 'SEACell'  # Replace with your metacell column name
# basis="umap_aligned"
def plot_metacell_transitions(adata, X_metacell, V_metacell, data_id,
                                figpath=None,
                                metacell_col="SEACell", 
                                annotation_class="manual_annotation",
                                basis='umap_aligned', genotype="WT",
                                cell_type_color_dict = cell_type_color_dict,
                                cell_size=10, SEACell_size=20,
                                scale=1, arrow_scale=15, arrow_width=0.002, dpi=120):
    
    # create a figure object (matplotlib)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Prepare data for plotting
    umap_coords = pd.DataFrame(adata.obsm[f'X_{basis}'], columns=[0, 1], index=adata.obs_names)
    umap_data = umap_coords.join(adata.obs[[metacell_col, annotation_class]])
    umap_data = umap_data.rename(columns={annotation_class: 'celltype'})

    # Plot single cells
    sns.scatterplot(
        x=0, y=1, hue='celltype', data=umap_data, s=cell_size, 
        palette=cell_type_color_dict, legend=None, ax=ax, alpha=0.7
    )

    # Calculate most prevalent cell type for each metacell
    most_prevalent = adata.obs.groupby(metacell_col)[annotation_class].agg(lambda x: x.value_counts().idxmax())

    # Prepare metacell data
    mcs = umap_data.groupby(metacell_col).mean().reset_index()
    mcs['celltype'] = most_prevalent.values

    # Plot metacells
    sns.scatterplot(
        x=0, y=1, s=SEACell_size, hue='celltype', data=mcs,
        palette=cell_type_color_dict, edgecolor='black', linewidth=1.25,
        legend=None, ax=ax
    )

    # Plot transition vectors
    Q = ax.quiver(X_metacell[:, 0], X_metacell[:, 1], V_metacell[:, 0], V_metacell[:, 1],
                angles='xy', scale_units='xy', scale=1/arrow_scale, width=arrow_width,
                color='black', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # # Create custom legend
    # handles = [mpatches.Patch(color=color, label=label) 
    #         for label, color in cell_type_color_dict.items() 
    #         if label in umap_data['celltype'].unique()]
    # ax.legend(handles=handles, title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'Metacell Transitions on {basis.upper()}')
    plt.tight_layout()
    plt.grid(False)
    if figpath:
        plt.savefig(figpath + f"umap_{genotype}_metacell_aggre_trans_probs_{data_id}.png")
        plt.savefig(figpath + f"umap_{genotype}_metacell_aggre_trans_probs_{data_id}.pdf")
    # plt.show()
    # plt.show()
    return fig

# define the path to save the oracle objects
oracle_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"

for data_id in list_datasets:

    # define the path to save the figures
    figpath = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_NMPs_subsets_metacells/{data_id}/"
    # create the directory if not present already
    os.makedirs(figpath, exist_ok=True)

    # load the Oracle data (with cell-cell trans.probs from all in silico KOs)
    oracle = co.load_hdf5(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/14_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
    oracle

    # load the metacell(SEACell) information
    metacell = pd.read_csv(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/metacells/{data_id}_seacells_obs_manual_annotation_30cells.csv", index_col=0)

    # make a dict - keys=cell_id : values=SEACEll
    metacell_dict = metacell.SEACell.to_dict()

    # add the metacell information to the oracle.adata
    oracle.adata.obs["SEACell"] = oracle.adata.obs_names.map(metacell_dict)
    oracle.adata.obs.head()

    # perform in_silico_KO analysis
    list_KO_genes_filtered = [gene for gene in list_KO_genes if gene in oracle.active_regulatory_genes]
    print(list_KO_genes_filtered)

    # perform in silico KO for each gene/TF
    for KO_gene in list_KO_genes_filtered:
        # Enter perturbation conditions to simulate signal propagation after the perturbation.
        oracle.simulate_shift(perturb_condition={KO_gene: 0.0},
                            n_propagation=3)
        # Compute the cell-cell transition probabilities for each KO gene
        # Get transition probability
        oracle.estimate_transition_prob(n_neighbors=200,
                                        knn_random=True,
                                        sampled_fraction=1)

        # Calculate embedding
        oracle.calculate_embedding_shift(sigma_corr=0.05)

        # extract the cell-cell trans.probs
        trans_prob_KO = oracle.transition_prob
        # convert from dense to sparse matrix
        trans_prob_KO = sparse.csr_matrix(trans_prob_KO)

        # extract the 2D embedding shift (2D transition vectors, v_[i,sim])
        oracle.adata.obsm[f"{KO_gene}_KO_co"] = oracle.delta_embedding

        # average across the metacells
        genotype = f"{KO_gene}_KO_co"
        print(genotype)

        # average the 2D embedding and 2D transition vectors across "metacells"
        X_metacell, V_metacell = average_2D_trans_vecs_metacells(oracle.adata, 
                                                                metacell_col="SEACell", 
                                                                basis='umap_aligned',
                                                                key_added=genotype)

        # generate the plot and save it in the folder
        plot_metacell_transitions(oracle.adata, X_metacell, V_metacell, data_id=data_id,
                                    figpath=figpath,
                                    metacell_col="SEACell", 
                                    annotation_class="manual_annotation",
                                    basis='umap_aligned', genotype=genotype,
                                    arrow_scale=2.5)
        # plt.savefig(figpath + f"umap_{genotype}_metacell_avg_trans_vectors_{data_id}.png")
        # plt.savefig(figpath + f"umap_{genotype}_metacell_avg_trans_vectors_{data_id}.pdf")

        print(f"Plot generated for {KO_gene}")
    # save the oracle object
    oracle.to_hdf5(oracle_path + f"15_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
    print(f"Oracle object saved for {data_id}")

