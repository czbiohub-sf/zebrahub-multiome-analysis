# A collection of python functions (modules) for simulating cell fate transitions based on the coarse-grained transition probabilities
# For Jupyter notebook, this can be run on 


# load the required libraries
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import scanpy as sc

from scipy.sparse import csr_matrix
from scipy.linalg import expm
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import celloracle as co

# inputs:
# adata_master_path: AnnData object with the following attributes (remember to import the master object with all timepoints)
# annotation: cell type annotation level (e.g. 'cell_type', 'cell_type1', 'cell_type2', ...)
# trans_probs: cell-cell transition probabilities (cell-by-cell matrix)

# import adata
# adata_master = sc.read_h5ad(adata_master_path)

# trans_probs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/sequencing_ver1/TDR118_cicero_output/in_silico_KO_trans_probs/WT_trans_prob.csv", index_col=0)
# trans_probs.head()

# Step 1. Extract the transition probabilities from the Oracle object
def compute_trans_probs(oracle, goi):
    """
    compute the transition probabilities from the Oracle object for given genes of interest (goi).
    
    Parameters:
        oracle: Oracle object containing the data.
        goi: Single gene (str) or list of genes (list of str) of interest.
    
    Returns:
        trans_probs: cell-cell transition probabilities matrix.
    """
    # extract the adata from the oracle object
    adata = oracle.adata

    # compute the transition probabilities for the given gene(s) of interest (goi)
    if not goi:
        perturb_condition = {}
    elif isinstance(goi, str):
        perturb_condition = {goi: 0.0}
    elif isinstance(goi, list):
        perturb_condition = {gene: 0.0 for gene in goi}
    else:
        raise ValueError("goi must be a string or a list of strings")
    # # extract the adata from the oracle object
    # adata = oracle.adata
    # adata

    # # compute the transition probabilities for a given gene (goi)
    # if goi=="":
    #     perturb_condition = {}
    # elif goi:
    #     perturb_condition = {goi: 0.0}
    
    # Enter perturbation conditions to simulate signal propagation after the perturbation.
    oracle.simulate_shift(perturb_condition=perturb_condition,
                        n_propagation=3)

    # estimate the cell-cell transition probability
    oracle.estimate_transition_prob(n_neighbors=200,
                                    knn_random=True,
                                    sampled_fraction=1)

    # Calculate embedding
    oracle.calculate_embedding_shift(sigma_corr=0.05)

    trans_probs = get_transition_matrix(oracle)

    return trans_probs


# Extract the transition probabilities
def get_transition_matrix(oracle_object):
    matrix = oracle.transition_prob.copy()
    dense_array = np.array(matrix, dtype=np.float32)
    trans_probs = dense_array
    # Convert dense array to a sparse matrix
    # sparse_matrix = csr_matrix(dense_array)
    #     adata.uns['transition_matrix'] = sparse_matrix
    return trans_probs

# Step 2. coarse-graining the cell-cell transition probabilities (macrostates/celltype level)
def coarse_graining_trans_probs(adata_master, data_key, 
                                annotatio_key="scANVI_zscape", trans_probs):
    """
    Coarse-grain the cell-cell transition probabilities to the cell type level.
    
    Parameters
    """

    # define the list of celltypes
    unique_cell_types = adata_master.obs[annotation_key].unique()

    # subset the adata_master into the subset for the given timepoint
    adata_sub = adata_master[adata_master.obs["dataset"]==data_key].copy()
    
    # Map each cell to its corresponding index
    cell_types = adata_sub.obs[annotation_key]
    type_to_index = {ctype: idx for idx, ctype in enumerate(unique_cell_types)}

    # Create a vector of these indices
    indices = cell_types.map(type_to_index).values

    # Initialize a matrix to store coarse-grained transition probabilities
    CG_trans_probs = np.zeros((len(unique_cell_types), len(unique_cell_types)))

    # For each cell type pair, sum the probabilities from trans_prob
    for i, ctype1 in enumerate(unique_cell_types):
        for j, ctype2 in enumerate(unique_cell_types):
            # Create masks for the rows and columns corresponding to each cell type
            mask_i = (indices == i)
            mask_j = (indices == j)

            # Aggregate transitions for these cell types
            CG_trans_probs[i, j] = np.sum(trans_probs[mask_i, :][:, mask_j])
        
    # Normalize the rows of CG_trans_probs to sum to 1
    row_sums = CG_trans_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Set zero sums to 1 to avoid division by zero
    CG_trans_probs = CG_trans_probs / row_sums
            
    return CG_trans_probs

# Function to compute initial distribution for a given timepoint or subset
def compute_celltype_dist(adata_sub, unique_cell_types, annotation_key):
    # Extract cell type information from the subset
    cell_types = adata_sub.obs[annotation_key]

    # Count the frequency of each cell type in the subset
    celltype_counts = cell_types.value_counts(normalize=True)

    # Convert counts to a sorted numpy array that matches the order of unique_cell_types
    celltype_dist = np.array([celltype_counts.get(ctype, 0) for ctype in unique_cell_types])

    return celltype_dist

# define the cost function
def cost_function(t, initial_dist, next_dist, CG_trans_probs):
    P_t = expm(CG_trans_probs * t)  # Matrix exponentiation at time t
    final_dist = np.dot(P_t, initial_dist)
    final_dist /= np.sum(final_dist)  # Normalize the distribution
    J_t = np.sum((final_dist - next_dist) ** 2)
    return J_t

# define the optimization function
def optimize_delta_time(initial_dist, next_dist, CG_trans_probs):
    # Initial guess for t
    t_initial_guess = np.array([1.0])  # Make sure it's an array

    # Minimize the cost function
    result = minimize(cost_function, t_initial_guess, args=(initial_dist, next_dist, CG_trans_probs), method='Nelder-Mead')
    
    # Optimal t
    optimal_t = result.x[0]
    print(f'Optimal t: {optimal_t}')

    return optimal_t




# # Function to compute the transition matrix for a given timepoint or subset
# def simulate_KO(oracle_object, gene):
#     oracle = oracle_object
#     goi = gene
    
#     # simulate the shift in gene expression
#     oracle.simulate_shift(perturb_condition={goi: 0.0},
#                           n_propagation=3)

#     # Get transition probability
#     oracle.estimate_transition_prob(n_neighbors=200,
#                                     knn_random=True, 
#                                     sampled_fraction=1)

#     # Calculate embedding 
#     oracle.calculate_embedding_shift(sigma_corr=0.05)
    
#     return oracle

