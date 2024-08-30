#!/usr/bin/env python3
"""
In silico knock-out analysis for all genes in the zebrafish GRN using CellOracle.

This script performs systematic in silico knock-out simulations for genes in a Gene Regulatory Network (GRN),
computes cell-cell transition probabilities, and projects them into 2D space.

Author: Yang-Joon Kim (yang-joon.kim@czbiohub.org)
Last updated: 08/21/2024

Usage:
    python script_name.py --oracle_path PATH1 --adata_path PATH2 --figpath PATH3 --data_id ID --list_KO_genes GENES

Arguments:

"""
# conda environment: celloracle_env

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import glob
import os
import scipy as sp
from scipy import sparse
import cellrank as cr
import scvelo as scv

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import celloracle as co
from celloracle.applications import Oracle_development_module
co.__version__

# # Parse command-line argument
def parse_arguments():
    parser = argparse.ArgumentParser(description="In silico knock-out analysis using CellOracle")
    
    # Required arguments
    parser.add_argument('--adata_path', type=str, required=True, help="Path to the AnnData object")
    # parser.add_argument('--lsi_path', type=str, required=True, help="Path to the integrated LSI file")
    # parser.add_argument('--metacell_path', type=str, required=True, help="Path to the metacell data")
    parser.add_argument('--oracle_path', type=str, required=True, help="Path to save/load Oracle object")
    parser.add_argument('--data_id', type=str, required=True, help="Data identifier")
    parser.add_argument('--figpath', type=str, required=True, help="Path to save figures")
    
    # Optional arguments
    parser.add_argument('--annotation', type=str, default="manual_annotation", help="Cell-type annotation column name")
    parser.add_argument('--dim_reduce', type=str, default="X_umap_aligned", help="Dimensionality reduction method to use")
    parser.add_argument('--list_KO_genes', type=str, default="all", 
                        help="Comma-separated list of genes to knock out, or 'all' for all active regulatory genes")
    # parser.add_argument('--use_pseudotime', action='store_true', help="Use different pseudotime method")
    # parser.add_argument('--pseudotime_path', type=str, help="Path to pseudotime dataframe (if different from Oracle object)")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    adata_path = args.adata_path
    # lsi_path = args.lsi_path
    # metacell_path = args.metacell_path
    oracle_path = args.oracle_path
    data_id = args.data_id
    figpath = args.figpath
    list_KO_genes = args.list_KO_genes
    basis='umap_aligned'
    # annotation = args.annotation
    # dim_reduce = args.dim_reduce

    # Step 1. Load the datasets (adata and dim.reductions)
    # 1-1. Load the adata object
    adata = sc.read_h5ad(adata_path + f'{data_id}_nmps_manual_annotation.h5ad')

    # 1-2. "X_lsi_integrated" : LSI embedding of the integrated dataset (computed using rLSI in Seurat).
    # NOTE. there are 50 LSI components, but the first component is usually highly correlated with the sequencing depth, so we'll filter this out.
    lsi_integrated = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_lsi.csv", index_col=0)

    # subset the LSI embedding to the cells in the adata
    lsi_integrated_sub = lsi_integrated[lsi_integrated.index.isin(adata.obs_names)]

    # take the 2nd to 40th columns (dropping the 1st LSI component)
    lsi_integrated_sub_filtered = lsi_integrated_sub.iloc[:,1:40]

    # make sure that the indicies match between the lsi and adata
    lsi_integrated_sub_filtered = lsi_integrated_sub_filtered.reindex(adata.obs.index)

    # Add the LSI embedding to the adata
    adata.obsm["X_lsi_integrated"] = lsi_integrated_sub_filtered.to_numpy()

    # # # 1-3. Import the metacells
    # metadata = pd.read_csv(metacell_path + f"{data_id}_seacells_lsi_integrated_obs_manual_annotation.csv", index_col=0)
    # # aggregated metacells (metacells-by-genes)
    # ad_meta = sc.read_h5ad(metacell_path + f"{data_id}_seacells_aggre_lsi_integrated.h5ad")

    # adata.obs["SEACell"] = adata.obs_names.map(metadata["SEACell"])
    # adata.obs["SEACell"]

    # # compute the umap coordinantes for SEACells (by averaginng x, y coordinates among the metacells in 2d UMAP)
    # dim_reduce = "X_umap_aligned"
    # umap = pd.DataFrame(adata.obsm[dim_reduce]).set_index(adata.obs_names).join(adata.obs["SEACell"])
    # umap["SEACell"] = umap["SEACell"].astype("category")
    # mcs = umap.groupby("SEACell").mean().reset_index()

    # 1-4. Load the Oracle object
    # oracle_path = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/{data_id}/"
    oracle = co.load_hdf5(oracle_path + f"{data_id}/" + f'10_{data_id}_pseudotime.celloracle.oracle')

    # Step 2. (WT) Compute the cell-cell transition probabilities using
    # the Pseudotime (computed by pySlingshot)
    # First, transfer the "Pseudotime" from the Oracle.adata to the "adata" object
    adata.obs["Pseudotime"] = oracle.adata.obs["Pseudotime"]

    # 2-1. k-nn graph:computing the nearest-neighborhood using the "integrated_lsi"
    # NOTE. CellRank requires adata.obsp["connectivities"] to be present
    sc.pp.neighbors(adata, use_rep="X_lsi_integrated")

    # Use CellRank's "PseudotimeKernel" to compute the cell-cell transition probabilities
    # compute a transition matrix using the "PseudotimeKernel"
    pk = cr.kernels.PseudotimeKernel(adata, time_key="Pseudotime")
    pk.compute_transition_matrix()

    #extract the trans_probs for the WT (PseudotimeKernel)
    trans_probs_WT_pt = pk.transition_matrix

    # generate the 2D projection of the metacell transitions with weighted vectors
    # NOTE. this saves adata.obsm["{key_added}_{"basis"}"]
    # for the 2D projection of cell-cell trans.probs at single-cell level (cells-by-(x,y))
    # NOTE. this also saves adata.obsp["T_fwd_{"basis"}"] to save the cell-cell trans.probs
    pk.plot_projection(basis="umap_aligned", recompute=True, stream=False, scale=0.5, key_added="WT")
    # NOTE. this saves adata.obsm["{key_added}_{"basis"}"] to be averaged/smoothed later either at k-nn, or metacells

    # update the Oracle object with the "adata" (for pseudotime-derived cell-cell trans.probs and 2D projections)
    # oracle.adata = adata.copy()
    oracle.adata.obsp[f"T_fwd_WT"] = trans_probs_WT_pt
    oracle.adata.obsm[f"WT_{basis}"] = adata.obsm[f"WT_{basis}"]

    # Step 3. Smoothing the 2D projected transition vectors (cell-level) using Metacells
    # use the function above (utils)
    # X_metacell, V_metacell = average_2D_trans_vecs_metacells(adata, metacell_col="SEACell", 
    #                                                         basis='umap_aligned',key_added='WT')

    # # generate the plot for the metacell transitions
    # fig = plot_metacell_transitions(adata, figpath, data_id, X_metacell, V_metacell,
    #                                 metacell_col="SEACell", 
    #                                 annotation_class="manual_annotation",
    #                                 basis='umap_aligned', 
    #                                 cell_type_color_dict = cell_type_color_dict,
    #                                 cell_size=10, SEACell_size=20,
    #                                 scale=1, arrow_scale=15, arrow_width=0.002, dpi=120)

    # Step 4. (KO) Compute the cell-cell transition probabilities for each KO gene
    # Scheme: perform a systematic in silico KO for all TFs/genes,
    # and compute the cell-cell transition probabilities for each KO gene
    # and visualize the metacell transitions for each KO gene
    # For each KO gene, we will simulate the signal propagation after the perturbation
    # Also, for the cell-cell trans.probs, we will use the "PrecomputedKernel" in CellRank
    # to compute the 2D projection (trans.probs + k-nn graph)
    # NOTE. we will save the resulting cell-cell trans.probs and 2D vectors in oracle.adata
    # then, save the Oracle object as "hdf5" format
    # if list_KO_genes == "all":
    #     list_KO_genes = oracle.active_regulatory_genes
    # else:
    #     list_KO_genes = list_KO_genes.split(",")
    # Process list_KO_genes
    if args.list_KO_genes.lower() == "all":
        list_KO_genes = oracle.active_regulatory_genes
    else:
        # Split by comma and strip whitespace from each gene name
        list_KO_genes = [gene.strip() for gene in args.list_KO_genes.split(',')]


    for KO_gene in list_KO_genes:
        print(KO_gene)
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

        # add the trans_prob_KO to the adata.obsp["T_fwd_{KO_gene}_KO"] (cell-cell transition probs)
        adata.obsp[f"T_fwd_{KO_gene}_KO"] = trans_prob_KO

        # use CellRank's "PrecomputedKernel" to import the cell-cell transition probabilities
        kr_precomputed = cr.kernels.PrecomputedKernel(adata, obsp_key=f"T_fwd_{KO_gene}_KO")

        # comptue the 2D projection of the cell-cell trans.probs (at single-cell level)
        kr_precomputed.plot_projection(basis="umap_aligned", recompute=True, stream=False, scale=0.5, 
                                    connectivities=adata.obsp["connectivities"], key_added=f"{KO_gene}_KO")
        # NOTE. this saves adata.obsm[f"{KO_gene}_KO"}"] to be averaged/smoothed later either at k-nn, or metacells

        # # update the Oracle object with the "adata" (for pseudotime-derived cell-cell
        # oracle.adata = adata.copy()
        oracle.adata.obsp[f"T_fwd_{KO_gene}_KO"] = adata.obsp[f"T_fwd_{KO_gene}_KO"]
        oracle.adata.obsm[f"{KO_gene}_KO_{basis}"] = adata.obsm[f"{KO_gene}_KO_{basis}"]

        # Save the Oracle object
        oracle.to_hdf5(oracle_path + f"14_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
        print(f"Finished in silico KO simulation for {KO_gene}")

    print("Finished in silico KO simulation")

if __name__ == "__main__":
    main()