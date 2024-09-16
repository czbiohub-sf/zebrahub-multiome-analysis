# A python script to compute cosine similiarities between WT and KO samples for all in silico KOs
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

# define the utils functions
def compute_metacell_transitions(adata, trans_key="T_fwd_WT", metacell_key="SEACell"):
    # Get the cell-cell transition matrix
    T_cell = adata.obsp[trans_key]
    
    # Get metacell labels
    metacells = adata.obs[metacell_key]
    
    # Get unique metacells
    unique_metacells = metacells.unique()
    
    # Initialize the metacell transition matrix
    n_metacells = len(unique_metacells)
    T_metacell = np.zeros((n_metacells, n_metacells))
    
    # Create a mapping of metacell to cell indices
    metacell_to_indices = {mc: np.where(metacells == mc)[0] for mc in unique_metacells}
    
    # Compute metacell transitions
    for i, source_metacell in enumerate(unique_metacells):
        source_indices = metacell_to_indices[source_metacell]
        for j, target_metacell in enumerate(unique_metacells):
            target_indices = metacell_to_indices[target_metacell]
            
            # Extract the submatrix of transitions from source to target metacell
            submatrix = T_cell[source_indices][:, target_indices]
            
            # Sum all transitions and normalize by the number of source cells
            T_metacell[i, j] = submatrix.sum() / len(source_indices)
    
    # Create a DataFrame for easier interpretation
    T_metacell_df = pd.DataFrame(T_metacell, index=unique_metacells, columns=unique_metacells)
    
    # Normalize rows to sum to 1
    T_metacell_df = T_metacell_df.div(T_metacell_df.sum(axis=1), axis=0)
    
    return T_metacell_df


    # Process the Metacell information
    # Load the metacell data and transfer the annotation to the main adata object (oracle.adata)
    
    # Calculate most prevalent cell type for each metacell
    most_prevalent = oracle.adata.obs.groupby("SEACell")["manual_annotation"].agg(lambda x: x.value_counts().idxmax())
    most_prevalent


    # average the 2D embedding and 2D transition vectors across "metacells"
    trans_probs_metacell_WT = compute_metacell_transitions(oracle.adata, 
                                                        trans_key="T_fwd_WT", 
                                                        metacell_key="SEACell")


    # Initialize an empty DataFrame with celltypes as the index
    metacells = trans_probs_metacell_WT.index
    cosine_sim_df = pd.DataFrame(index=metacells)

    # Compute cosine similarities for each gene knockout
    for gene in oracle.active_regulatory_genes:
        # Compute transition probabilities for the current gene knockout
        trans_key = f"T_fwd_{gene}_KO"
        trans_probs_metacell_KO = compute_metacell_transitions(oracle.adata, trans_key=trans_key, 
                                                                metacell_key="SEACell")
        
        # Compute cosine similarities
        cosine_similarities = compute_row_cosine_similarities(trans_probs_metacell_WT, trans_probs_metacell_KO)
        
        # Add the cosine similarities as a new column to the DataFrame
        cosine_sim_df[gene] = cosine_similarities

    # Display the resulting DataFrame
    cosine_sim_df

    cosine_sim_df["celltype"] = cosine_sim_df.index.map(most_prevalent)


    # average the cosine similarities across cell types
    cosine_sim_df_avg = cosine_sim_df.groupby("celltype").mean()
    df_averaged = cosine_sim_df_avg.reset_index()

    # save to the master dictionary
    dict_cos_sim_mater[data_id] = df_averaged
