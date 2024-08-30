# A collection of utility functions for in_silico knock-out analysis (CellOracle)
# Last updated: 08/20/2024
# sources:
# 1) CellRank: https://cellrank.readthedocs.io/en/stable/api/_autosummary/kernels/cellrank.kernels.PrecomputedKernel.html#cellrank.kernels.PrecomputedKernel.plot_projection

# Import necessary libraries
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.sparse import csr_matrix

# Import CellOracle library
import celloracle as co

# import CellRank modules
import cellrank as cr
import scvelo as scv

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
    T_fwd = adata.obsm[f'{key_added}_{basis}']  # Your cell-level transition vectors
    metacells = adata.obs[metacell_col].cat.codes
    n_metacells = len(adata.obs[metacell_col].cat.categories)
    
    # X_metacell is the average UMAP position of the metacells
    # V_metacell is the average transition vector of the metacells
    X_metacell = np.zeros((n_metacells, 2))
    V_metacell = np.zeros((n_metacells, 2))
    
    for i in range(n_metacells):
        mask = metacells == i
        X_metacell[i] = X_umap[mask].mean(axis=0)
        V_metacell[i] = T_fwd[mask].mean(axis=0)
    
    return X_metacell, V_metacell

# plotting function for the averaged trans_vectors
# metacell_col = 'SEACell'  # Replace with your metacell column name
# basis="umap_aligned"
def plot_metacell_transitions(adata, figpath, data_id, X_metacell, V_metacell,
                                metacell_col="SEACell", 
                                annotation_class="manual_annotation",
                                basis='umap_aligned', 
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
    ax.grid(False)

    # Create custom legend
    handles = [mpatches.Patch(color=color, label=label) 
            for label, color in cell_type_color_dict.items() 
            if label in umap_data['celltype'].unique()]
    ax.legend(handles=handles, title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'Metacell Transitions on {basis.upper()}')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(figpath + f"umap_WT_metacell_aggre_trans_probs_{data_id}.png")
    plt.savefig(figpath + f"umap_WT_metacell_aggre_trans_probs_{data_id}.pdf")
    # plt.show()
    return fig

# # A function to compute the gaussian kernel
# def gaussian_kernel(X: np.ndarray, mu: float=0, sigma: float=1) -> np.ndarray:
#     """Compute Gaussian kernel"""
#     return np.exp(-(X - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


# # A function to compute the cell-cell transition probabilities
# # metacell_col = "SEACell"
# # def metacell_transitions_weighted(adata, metacell_col, CG_trans_probs, 
# #                                   basis='umap_aligned', sigma_distance=1.0, sigma_transition=0.1):
# #     """compute 2D projection of metacell transitions with weighted vectors"""
# #     T = trans_probs_WT_pt
# #     X_umap = adata.obsm[f'X_{basis}']
    
# #     # Get metacell assignments
# #     metacells = adata.obs[metacell_col].cat.codes
# #     n_metacells = len(adata.obs[metacell_col].cat.categories)
    
# #     # Import the metacell-to-metacell transition probabilities (pre-computed and inputted to this function)
# #     # T_metacell = csr_matrix((T.ravel(), (np.repeat(metacells, len(metacells)), np.tile(metacells, len(metacells)))))
# #     # T_metacell = T_metacell.toarray()
# #     # T_metacell /= T_metacell.sum(axis=1, keepdims=True)
# #     T_metacell = CG_trans_probs
    
# #     # Compute average UMAP positions for metacells
# #     X_metacell = np.zeros((n_metacells, 2))
# #     for i in range(n_metacells):
# #         X_metacell[i] = X_umap[metacells == i].mean(axis=0)
    
# #     # # Compute 2D vectors for metacell transitions
# #     # V_metacell = np.zeros((n_metacells, 2))
# #     # for i in range(n_metacells):
# #     #     for j in range(n_metacells):
# #     #         if i != j:
# #     #             V_metacell[i] += T_metacell[i, j] * (X_metacell[j] - X_metacell[i])
    
# #     # Compute weighted 2D vectors for metacell transitions
# #     V_metacell = np.zeros((n_metacells, 2))
# #     for i in range(n_metacells):
# #        # Compute distance-based weights
# #         distances = np.linalg.norm(X_metacell - X_metacell[i], axis=1)
# #         distance_weights = gaussian_kernel(distances, mu=0, sigma=sigma_distance)
# #         distance_weights[i] = 0  # Exclude self-transitions
        
# #         # Compute transition probability weights
# #         trans_weights = gaussian_kernel(T_metacell[i], mu=T_metacell[i].max(), sigma=sigma_transition)
        
# #         # Combine weights
# #         combined_weights = distance_weights * trans_weights
        
# #         for j in range(n_metacells):
# #             if i != j:
# #                 V_metacell[i] += combined_weights[j] * (X_metacell[j] - X_metacell[i])
        
# #         # Normalize the vector
# #         V_metacell[i] /= combined_weights.sum()
    
# #     return X_metacell, V_metacell, T_metacell

# # computing the average of 2D transition vectors from single-cells @ Metacell level
# def metacell_transitions_from_cells(adata, metacell_col, basis='umap_aligned', KO_genes=None):
#     X_umap = adata.obsm[f'X_{basis}']
#     T_fwd = adata.obsm[f'T_fwd_{KO_genes}_KO_{basis}']  # Your cell-level transition vectors
#     metacells = adata.obs[metacell_col].cat.codes
#     n_metacells = len(adata.obs[metacell_col].cat.categories)
    
#     X_metacell = np.zeros((n_metacells, 2))
#     V_metacell = np.zeros((n_metacells, 2))
    
#     for i in range(n_metacells):
#         mask = metacells == i
#         X_metacell[i] = X_umap[mask].mean(axis=0)
#         V_metacell[i] = T_fwd[mask].mean(axis=0)
    
#     return X_metacell, V_metacell

# def plot_metacell_transitions(X_metacell, V_metacell, T_metacell, basis='umap_aligned', scale=1, cmap='viridis', arrow_size=15, dpi=120):
#     plt.figure(figsize=(6, 6), dpi=dpi)
    
#     # Plot metacells
#     plt.scatter(X_metacell[:, 0], X_metacell[:, 1], c='grey', alpha=0.5, s=50)
    
#     # Plot transition vectors
#     Q = plt.quiver(X_metacell[:, 0], X_metacell[:, 1], V_metacell[:, 0], V_metacell[:, 1],
#                    np.linalg.norm(V_metacell, axis=1), angles='xy', scale_units='xy',
#                    scale=scale, width=0.002, headwidth=3, headlength=5, headaxislength=4.5,
#                    alpha=0.8, cmap=cmap)
    
#     plt.colorbar(Q, label='Transition probability magnitude')
#     plt.title(f'Metacell Transitions on {basis.upper()}')
#     plt.xlabel(f'{basis.upper()} 1')
#     plt.ylabel(f'{basis.upper()} 2')
#     plt.show()

# # # Usage
# # metacell_col = 'SEACell'  # Replace with your metacell column name
# # X_metacell, V_metacell, T_metacell = metacell_transitions_weighted(adata, metacell_col, CG_trans_probs, 
# #                                                           sigma_distance=1, sigma_transition=0.5)
# # plot_metacell_transitions(X_metacell, V_metacell, T_metacell, scale=1)

# # A function to plot the cell-cell transition probability at single-cell level
# def plot_transition_probabilities(adata, basis='umap_aligned', n_samples=1000, n_grid=40, scale=1, cmap='coolwarm', arrow_size=15, dpi=120):
#     plt.figure(figsize=(12, 10), dpi=dpi)
    
#     # Get UMAP coordinates and transition vectors
#     X = adata.obsm[f'X_{basis}']
#     V = adata.obsm['T_fwd_umap_aligned']
    
#     # Subsample cells if needed
#     if n_samples and n_samples < adata.n_obs:
#         idx = np.random.choice(adata.n_obs, n_samples, replace=False)
#         X = X[idx]
#         V = V[idx]
    
#     # Compute vector magnitudes
#     magnitudes = np.linalg.norm(V, axis=1)
    
#     # Create grid
#     grid_x = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
#     grid_y = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)
    
#     # Compute grid-based averages
#     vx, _, _, _ = binned_statistic_2d(X[:, 0], X[:, 1], V[:, 0], statistic='mean', bins=[grid_x, grid_y])
#     vy, _, _, _ = binned_statistic_2d(X[:, 0], X[:, 1], V[:, 1], statistic='mean', bins=[grid_x, grid_y])
    
#     # Plot cells
#     plt.scatter(X[:, 0], X[:, 1], c='grey', alpha=0.5, s=10)
    
#     # Plot grid-based transitions
#     X_grid, Y_grid = np.meshgrid(grid_x[:-1] + np.diff(grid_x)/2, grid_y[:-1] + np.diff(grid_y)/2)
#     plt.quiver(X_grid, Y_grid, vx.T, vy.T, scale=scale*50, width=0.002, headwidth=3, headlength=5, headaxislength=4.5, alpha=0.8, color='black')
    
#     # Plot single-cell transitions
#     Q = plt.quiver(X[:, 0], X[:, 1], V[:, 0], V[:, 1], magnitudes, 
#                    angles='xy', scale_units='xy', scale=scale, 
#                    width=0.002, headwidth=3, headlength=5, headaxislength=4.5, 
#                    alpha=0.6, cmap=cmap)
    
#     plt.colorbar(Q, label='Transition probability magnitude')
#     plt.title(f'Cell-Cell Transition Probabilities on {basis.upper()}')
#     plt.xlabel(f'{basis.upper()} 1')
#     plt.ylabel(f'{basis.upper()} 2')
#     plt.show()

# # Usage
# # plot_transition_probabilities(adata, basis='umap_aligned', n_samples=5000, n_grid=40, scale=0.2, arrow_size=15, dpi=120)