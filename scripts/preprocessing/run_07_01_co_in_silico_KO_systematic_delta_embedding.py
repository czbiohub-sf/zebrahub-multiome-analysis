#!/usr/bin/env python3
"""
In silico knock-out analysis for all genes in the zebrafish GRN using CellOracle.

This script performs systematic in silico knock-out simulations for genes in a Gene Regulatory Network (GRN),
computes cell-cell transition probabilities, and projects them into 2D space.

Author: Yang-Joon Kim (yang-joon.kim@czbiohub.org)
Last updated: 09/16/2024

Usage:
    python script_name.py --oracle_path PATH1 --adata_path PATH2 --data_id ID --list_KO_genes GENES
Arguments:

"""
# conda environment: celloracle_env

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import scipy as sp
from scipy import sparse
import cellrank as cr
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import celloracle as co
from celloracle.applications import Oracle_development_module
co.__version__

# Parse command-line argument
def parse_arguments():
    parser = argparse.ArgumentParser(description="In silico knock-out analysis using CellOracle")
    parser.add_argument('--oracle_path', type=str, required=True, help="Path to save/load Oracle object")
    parser.add_argument('--data_id', type=str, required=True, help="Data identifier")
    parser.add_argument('--list_KO_genes', type=str, default="all", help="Comma-separated list of genes to knock out, or 'all' for all active regulatory genes")
    return parser.parse_args()

def simulate_KO(oracle, KO_gene, data_id, basis):
    print(KO_gene)
    oracle.simulate_shift(perturb_condition={KO_gene: 0.0}, n_propagation=3)
    oracle.estimate_transition_prob(n_neighbors=200, knn_random=True, sampled_fraction=1)
    oracle.calculate_embedding_shift(sigma_corr=0.05)
    trans_prob_KO = sparse.csr_matrix(oracle.transition_prob)
    oracle.adata.obsp[f"T_fwd_{KO_gene}_KO"] = trans_prob_KO
    oracle.adata.obsm[f"{KO_gene}_KO_{basis}"] = oracle.delta_embedding
    return oracle
    print(f"Finished in silico KO simulation for {KO_gene}")

def main():
    args = parse_arguments()
    oracle_path = args.oracle_path
    data_id = args.data_id
    list_KO_genes = args.list_KO_genes

    # Load Oracle object
    oracle = co.load_hdf5(oracle_path + f"14_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
    adata = oracle.adata
    basis = 'umap_aligned'

    # Process list_KO_genes
    if args.list_KO_genes.lower() == "all":
        list_KO_genes = oracle.active_regulatory_genes
    else:
        list_KO_genes = [gene.strip() for gene in args.list_KO_genes.split(',')]

    # Perform KO simulations sequentially
    for KO_gene in list_KO_genes:
        oracle = simulate_KO(oracle, KO_gene, data_id, basis)

    print("Finished all in silico KO simulations")
    oracle.to_hdf5(oracle_path + f"14_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
    print("saved all in silico KO simulations")

if __name__ == "__main__":
    main()