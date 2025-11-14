# A python script to perform Markovian simulation of cell fates using cell-cell transition probabilities 
# source: https://github.com/morris-lab/CellOracle/blob/master/docs/notebooks/05_simulation/Gata1_KO_simulation_with_Paul_etal_2015_data.ipynb
# NOTE: Run this on "celloracle_env" conda environment.

# Inputs:
# 1. adata: an adata object to compute the initial cell states (csv file)
# 2. annotation: coarse-grained celltype classes (used for coarse-graining the cell-cell trans.probs)
# 3. trans_probs: cell-cell transition probabilities (csv file)
# 4. list_n_steps: a list of number of steps for the simulation

# Output:
# 1. df_final_dist: a dataframe containing the final distribution of cell types after the simulation

# Load the packages
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import scanpy as sc

from scipy.linalg import expm

import celloracle as co
import matplotlib.pyplot as plt
import seaborn as sns

# Load the inputs
# trans_probs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/sequencing_ver1/TDR118_cicero_output/in_silico_KO_trans_probs/WT_trans_prob.csv", index_col=0)
# trans_probs.head()


# Define a function to compute the markovian simulation
def compute_markovian_sim(adata, annotation, trans_probs, list_n_steps=[0, 0.1, 1, 10]):
    """
    Compute Markovian simulation of cell fates using cell-cell transition probabilities
    """
    # Extract cell type information from adata.obs
    cell_types = adata.obs[annotation]

    # Define the unique cell types
    unique_cell_types = cell_types.unique()

    # Count the frequency of each cell type
    celltype_counts = cell_types.value_counts(normalize=True)
    celltype_counts

    # Convert counts to a sorted numpy array that matches the order of unique_cell_types
    initial_dist = np.array([celltype_counts.get(ctype, 0) for ctype in unique_cell_types])
    initial_dist

    # Coarse-graining the cell-cell transition probabilities
    # Map each cell to its corresponding index
    unique_cell_types = np.unique(cell_types)
    type_to_index = {ctype: idx for idx, ctype in enumerate(unique_cell_types)}

    # Create a vector of these indices
    indices = cell_types.map(type_to_index).values

    # Initialize a matrix to store coarse-grained transition probabilities
    CG_trans_probs = np.zeros((len(unique_cell_types), len(unique_cell_types)))

    # convert the pandas dataframe into numpy arrays
    trans_prob_matrix = trans_probs.values  # Convert DataFrame to NumPy array for faster operations


    # For each cell type pair, sum the probabilities from trans_prob
    for i, ctype1 in enumerate(unique_cell_types):
        for j, ctype2 in enumerate(unique_cell_types):
            # Create masks for the rows and columns corresponding to each cell type
            mask_i = (indices == i)
            mask_j = (indices == j)

            # Aggregate transitions for these cell types
            CG_trans_probs[i, j] = np.sum(trans_prob_matrix[mask_i, :][:, mask_j])
            
    # Normalize the rows of CG_trans_probs to sum to 1
    CG_trans_probs = CG_trans_probs / CG_trans_probs.sum(axis=1, keepdims=True)

    list_markov_steps = [0.01, 0.1, 1, 10, 100]
    df_final_dist = pd.DataFrame()

    for markov_step in list_n_steps:
        P_t = expm(CG_trans_probs * markov_step)  # Matrix exponentiation at time t

        # Compute distribution at time t
        final_dist = np.dot(P_t, initial_dist)
        # Normalize the distribution to sum to 1
        final_dist /= np.sum(final_dist)
        df_final_dist[markov_step] = final_dist

    return df_final_dist

def simulate_genetic_perturbation(oracle, goi, save_folder=None):
    """
    Simulate the genetic perturbation of a gene of interest
    """
    # perform the in silico KO of the gene of interest
    oracle = simulate_KO(oracle, goi)

    fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

    scale = 20
    # Show quiver plot
    oracle.plot_quiver(scale=scale, ax=ax[0])
    ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

    # Show quiver plot that was calculated with randomized graph.
    oracle.plot_quiver_random(scale=scale, ax=ax[1])
    ax[1].set_title(f"Randomized simulation vector")
    plt.show()

    # n_grid = 40 is a good starting value.
    n_grid = 40 
    oracle.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

    min_mass = 30
    oracle.calculate_mass_filter(min_mass=min_mass, plot=True)

    # Plot vector field with cell cluster 
    fig, ax = plt.subplots(figsize=[8, 8])

    oracle.plot_cluster_whole(ax=ax, s=10)
    oracle.plot_simulation_flow_on_grid(scale=10, ax=ax, show_background=False)

    save_folder = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_NMPs_subsets/TDR118reseq/"
    plt.savefig(save_folder + f"umap_nmps_15somites_KO_{goi}.png")
    plt.savefig(save_folder + f"umap_nmps_15somites_KO_{goi}.pdf")
    # Load the transition probabilities


def simulate_KO(oracle_object, gene):
    oracle = oracle_object
    goi = gene
    
    # simulate the shift in gene expression
    oracle.simulate_shift(perturb_condition={goi: 0.0},
                          n_propagation=3)

    # Get transition probability
    oracle.estimate_transition_prob(n_neighbors=200,
                                    knn_random=True, 
                                    sampled_fraction=1)

    # Calculate embedding 
    oracle.calculate_embedding_shift(sigma_corr=0.05)
    
    return oracle

def extract_transition_probability(oracle):
    # Cell name
    cell_name = oracle.adata.obs.index
    
    # Get probability as numpy matrix
    transition_prob = oracle.transition_prob
    
    # Convert probability into data frame.
    df_transition_prob = pd.DataFrame(transition_prob, index=cell_name, columns=cell_name)
    df_transition_prob = df_transition_prob.rename_axis(index="From", columns="To")
    
    return df_transition_prob

# define the cell_types for the generate_colorandum function
cell_types = [
    'Epidermal', 'Lateral_Mesoderm', 'PSM', 'Neural_Posterior',
    'Neural_Anterior', 'Neural_Crest', 'Differentiating_Neurons',
    'Adaxial_Cells', 'Muscle', 'Somites', 'Endoderm', 'Notochord',
    'NMPs'
]
def generate_colorandum(oracle, annotation, cell_types):
    """
    Generate the colorandum for the cell types
    """
    # Generate "Set3" color palette
    set3_palette = sns.color_palette("Set3", n_colors=len(cell_types))

    # Suppose the light yellow is the 10th color in the palette (9th index, as indexing is 0-based)
    # and you want to replace it with teal color
    teal_color = (0.0, 0.5019607843137255, 0.5019607843137255)  # RGB for teal
    set3_palette[1] = teal_color  # Replace the light yellow with teal

    # Assign colors to cell types
    custom_palette = {cell_type: color for cell_type, color in zip(cell_types, set3_palette)}
    # Change the color for 'NMPs' to a dark blue
    custom_palette['NMPs'] = (0.12941176470588237, 0.4, 0.6745098039215687)  # Dark blue color

    # Get the order of categories as they appear in the dataset
    categories_in_order = oracle.adata.obs[annotation].cat.categories

    # Reorder your palette dictionary to match this order
    ordered_palette = {cat: custom_palette[cat] for cat in categories_in_order if cat in custom_palette}

    # Now assign this ordered palette back to the AnnData object
    annotation_color = annotation + "_color"
    oracle.adata.uns[annotation_color] = ordered_palette

    # Change the color palette in the Oracle object
    # Extract the cell type annotations from the AnnData object
    cell_types = oracle.adata.obs[annotation].values

    # Map each cell type to its corresponding color
    colors_for_cells = np.array([custom_palette[cell_type] for cell_type in cell_types])

    # Replace the colorandum in the oracle object
    oracle.colorandum = colors_for_cells

    return oracle