# A python script to compute cosine similiarities between WT and KO samples for all in silico KOs
# last updated: 08/30/2024
# conda activate celloracle_env

# Import public libraries
import os
from scipy.stats import binned_statistic_2d
from scipy.sparse import csr_matrix
from scipy import sparse
import scipy.sparse as sp
from scipy.spatial.distance import cosine

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
# from tqdm.notebook import tqdm
import celloracle as co


def extract_transition_matrix(adata, trans_key):
    """Extract cell-cell transition matrix from adata object."""
    return adata.obsp[trans_key]


def get_metacell_labels(adata, metacell_key):
    """Get metacell labels and unique metacells."""
    metacells = adata.obs[metacell_key]
    unique_metacells = metacells.unique()
    return metacells, unique_metacells


def create_metacell_mapping(metacells, unique_metacells):
    """Create a mapping of metacell to cell indices."""
    return {mc: np.where(metacells == mc)[0] for mc in unique_metacells}


def compute_metacell_transition_matrix(T_cell, unique_metacells, metacell_to_indices):
    """Compute the raw metacell transition matrix."""
    n_metacells = len(unique_metacells)
    T_metacell = np.zeros((n_metacells, n_metacells))
    
    for i, source_metacell in enumerate(unique_metacells):
        source_indices = metacell_to_indices[source_metacell]
        for j, target_metacell in enumerate(unique_metacells):
            target_indices = metacell_to_indices[target_metacell]
            
            # Extract the submatrix of transitions from source to target metacell
            submatrix = T_cell[source_indices][:, target_indices]
            
            # Sum all transitions and normalize by the number of source cells
            T_metacell[i, j] = submatrix.sum() / len(source_indices)
    
    return T_metacell


def normalize_transition_matrix(T_metacell, unique_metacells):
    """Convert to DataFrame and normalize rows to sum to 1."""
    T_metacell_df = pd.DataFrame(T_metacell, index=unique_metacells, columns=unique_metacells)
    T_metacell_df = T_metacell_df.div(T_metacell_df.sum(axis=1), axis=0)
    return T_metacell_df


def compute_metacell_transitions(adata, trans_key="T_fwd_WT", metacell_key="SEACell"):
    """Compute metacell transition probabilities from cell-cell transitions."""
    # Extract components
    T_cell = extract_transition_matrix(adata, trans_key)
    metacells, unique_metacells = get_metacell_labels(adata, metacell_key)
    metacell_to_indices = create_metacell_mapping(metacells, unique_metacells)
    
    # Compute and normalize transition matrix
    T_metacell = compute_metacell_transition_matrix(T_cell, unique_metacells, metacell_to_indices)
    T_metacell_df = normalize_transition_matrix(T_metacell, unique_metacells)
    
    return T_metacell_df


def compute_row_cosine_similarities(df1, df2):
    """Compute row-wise cosine similarities between two DataFrames."""
    cosine_similarities = []
    
    for idx in df1.index:
        if idx in df2.index:
            vec1 = df1.loc[idx].values
            vec2 = df2.loc[idx].values
            # Use 1 - cosine distance to get cosine similarity
            similarity = 1 - cosine(vec1, vec2)
            cosine_similarities.append(similarity)
        else:
            cosine_similarities.append(np.nan)
    
    return pd.Series(cosine_similarities, index=df1.index)


def get_most_prevalent_celltype(adata, metacell_key="SEACell", annotation_key="manual_annotation"):
    """Calculate most prevalent cell type for each metacell."""
    return adata.obs.groupby(metacell_key)[annotation_key].agg(
        lambda x: x.value_counts().idxmax()
    )


def compute_cosine_similarities_for_all_kos(oracle, metacell_key="SEACell"):
    """Compute cosine similarities between WT and all KO conditions."""
    # Compute WT transition probabilities
    trans_probs_metacell_WT = compute_metacell_transitions(
        oracle.adata, 
        trans_key="T_fwd_WT", 
        metacell_key=metacell_key
    )
    
    # Initialize results DataFrame
    metacells = trans_probs_metacell_WT.index
    cosine_sim_df = pd.DataFrame(index=metacells)
    
    # Compute similarities for each gene knockout
    for gene in oracle.active_regulatory_genes:
        trans_key = f"T_fwd_{gene}_KO"
        trans_probs_metacell_KO = compute_metacell_transitions(
            oracle.adata, 
            trans_key=trans_key, 
            metacell_key=metacell_key
        )
        
        cosine_similarities = compute_row_cosine_similarities(
            trans_probs_metacell_WT, 
            trans_probs_metacell_KO
        )
        
        cosine_sim_df[gene] = cosine_similarities
    
    return cosine_sim_df


def add_celltype_annotations(cosine_sim_df, oracle, metacell_key="SEACell", annotation_key="manual_annotation"):
    """Add cell type annotations to cosine similarity DataFrame."""
    most_prevalent = get_most_prevalent_celltype(oracle.adata, metacell_key, annotation_key)
    cosine_sim_df["celltype"] = cosine_sim_df.index.map(most_prevalent)
    return cosine_sim_df


def average_similarities_by_celltype(cosine_sim_df):
    """Average cosine similarities across cell types."""
    cosine_sim_df_avg = cosine_sim_df.groupby("celltype").mean()
    return cosine_sim_df_avg.reset_index()


def process_cosine_similarities(oracle, metacell_key="SEACell", annotation_key="manual_annotation"):
    """
    Main processing function to compute and process cosine similarities.
    
    Parameters:
    -----------
    oracle : celloracle object
        CellOracle object containing the data and regulatory genes
    metacell_key : str
        Key for metacell annotations in adata.obs
    annotation_key : str
        Key for cell type annotations in adata.obs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with averaged cosine similarities by cell type
    """
    # Compute cosine similarities
    cosine_sim_df = compute_cosine_similarities_for_all_kos(oracle, metacell_key)
    
    # Add cell type annotations
    cosine_sim_df = add_celltype_annotations(cosine_sim_df, oracle, metacell_key, annotation_key)
    
    # Average by cell type
    df_averaged = average_similarities_by_celltype(cosine_sim_df)
    
    return df_averaged


def save_results_to_dict(results_dict, data_id, df_averaged):
    """Save results to master dictionary."""
    results_dict[data_id] = df_averaged
    return results_dict


# Example usage (commented out - uncomment and modify as needed):
# if __name__ == "__main__":
#     # Initialize results dictionary
#     dict_cos_sim_master = {}
#     
#     # Process data (assuming oracle and data_id are defined)
#     # df_averaged = process_cosine_similarities(oracle)
#     # dict_cos_sim_master = save_results_to_dict(dict_cos_sim_master, data_id, df_averaged)
