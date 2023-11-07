# A python script to compute cluster(cell-type)-specific GRNs using scRNA-seq dataset and a base GRN.
# Modified from "02_GRN/04_Network_analysis/Network_analysis_data_id.ipynb" Jupyter notebook from CellOracle.

# Step 0. Import libraries
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import celloracle as co
co.__version__

import time


# Input Arguments:
# 1) output_path: filepath for the output
# 2) RNAdata_path: filepath for the adata (RNA)
# 3) baseGRN_path: filepath for the base GRN (parquet)
# 4) data_id: data_id
# 5) annotation: annotation class (cell-type annotation)
# 6) dim_reduce: dimensionality reduction embedding name

# Parse command-line arguments
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Filter and map data using CellOracle")
#output_path, RNAdata_path, baseGRN_path, data_id, annotation, dim_reduce
# Add command-line arguments
parser.add_argument('output_path', type=str, help="output filepath")
parser.add_argument('RNAdata_path', type=str, help="RNA data filepath")
parser.add_argument('baseGRN_path', type=str, help="base GRN filepath")
parser.add_argument('data_id', type=str, help="data_id")
parser.add_argument('annotation', type=str, help="celltype annotation class")
parser.add_argument('dim_reduce', type=str, help="dim.reduction embedding name")
#parser.add_argument('timepoints', type=list, help="a list of timepoints")

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments as attributes of the 'args' object
output_path = args.output_path
RNAdata_path = args.RNAdata_path
baseGRN_path = args.baseGRN_path
data_id = args.data_id
annotation = args.annotation
dim_reduce = args.dim_reduce
#timepoints = args.timepoints


# Define the function here
def compute_cluster_specific_GRNs(output_path, RNAdata_path, baseGRN_path, 
                                    data_id, annotation, dim_reduce):
    """
    A function to compute cluster(cell-type) specific GRNs using CellOracle.
    It uses a base GRN built from the previous script (04_XXX.py)
    Note that we can subset the scRNA-seq datasets in different ways - i.e. cell-type annotation, timepoints, condition, etc.
    # Optional: For multiple timepoints, we will add an option for splitting out the data for "timepoints" before subsetting for "cluster".
    # 
    
    Parameters:
        output_path: filepath for the output files (below)
        RNAdata_path: filepath for the RNA h5ad file 
        baseGRN_path: filepath for the base GRN
        data_id: identifier for the output dataframe file
        annotation: annotation class for the clusters (cell-types)
        dim_reduce: dim.reduction name

    
    Returns: 
        df: a dataframe of peaks-by-TFs (base GRN)

    Examples:
        filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/"
        RNAdata_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/ZF_atlas_v01/ZF_atlas_v01.h5ad"
        baseGRN_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/05_TDR118_base_GRN_dataframe.parquet"
        data_id = "TDR118"
        annotation: "global_annotation"
        dim_reduce: "umap.atac"
    """
    # create a folder for figures (Optional)
    # output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/baseGRN_CisBP_RNA_zebrahub/"
    # save_folder = "figures"
    # os.makedirs(output_path + save_folder, exist_ok=True)

    # Step 1. Load RNA data (scRNA-seq)
    # Load the zebrahub early timepoints
    adata = sc.read_h5ad(RNAdata_path)
    adata

    # Checking the adata object (Optional)
    # sc.pl.umap(adata, color = ["global_annotation", "timepoint"])

    # compute the highly variable genes to reduce the number of gene feature space (RNA feature space)
    if len(adata.var_names)>3000:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
        adata = adata[:,adata.var.highly_variable]
        adata
    else:
        pass

    # # Check data shape again
    # print("Number of cell : ", adata.shape[0])
    # print("Number of gene : ", adata.shape[1])

    # show meta data name in anndata
    print("metadata columns :", list(adata.obs.columns))

    # The annotation class - "global_annotation"
    print("cell types are:")
    print(adata.obs.global_annotation.unique())



    # Step 2. Load the base GRN (CisBP or scATAC-seq)
    if baseGRN_path=="CisBP_zebrafish":
        print("loading the CellOracle CisBP base GRN")
        baseGRN = co.data.load_zebrafish_promoter_base_GRN()
        baseGRN
    else:
        # # Load TF info which was made from our own zebrafish single-cell multiome dataset (15-somite)
        print("loading the custom base GRN")
        baseGRN = pd.read_parquet(baseGRN_path)

    # Check data
    print("checking the base GRN:")
    baseGRN.head()

    # Step 3. Compute the Oracle objects (if there are multiple timepoints, we can compute cell-type specific GRNs for all timepoints)
    # filepath for the output Oracle objects
    #output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/baseGRN_CisBP_RNA_zebrahub/"

    # We will assume that we have only one timepoint for now.
    # Instantiate Oracle object
    oracle = co.Oracle()
    
    # Step 3-1. Add the scRNA-seq (adata) to the Oracle object
    oracle.import_anndata_as_raw_count(adata=adata,
                                    cluster_column_name="global_annotation", # annotation
                                    embedding_name="X_umap.atac") # X_umap.cellranger.arc"
    
    # Step 3-2. Add the base GRN (TF info dataframe)
    oracle.import_TF_data(TF_info_matrix=baseGRN)
    
    # Step 3-3. KNN imputation
    # Perform PCA
    oracle.perform_PCA()
    
    # number of cells
    n_cell = oracle.adata.shape[0]
    print(f"cell number is :{n_cell}")
    
    # number of k
    k = int(0.025*n_cell)
    print(f"Auto-selected k is :{k}")
    
    # KNN imputation
    n_comps = 50
    oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                        b_maxl=k*4, n_jobs=8)

    # Save oracle object.output_filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/"
    oracle.to_hdf5(output_path + "06_"+ data_id + ".celloracle.oracle")


    # Step 4. Compute cluster-specific (cell-type specific) GRNs for each timepoint (using the Oracle objects computed in Step 3.)
    # Load the Oracle object for each timepoint
    oracle = co.load_hdf5(output_path + "06_"+ data_id + ".celloracle.oracle")
    # print the info
    info = oracle._generate_meta_data()
    info

    # Start measuring time
    start_time = time.time()

    # Calculate GRN for each population in "predicted.id" clustering unit.
    # This step may take long time (~ 1hour)
    links = oracle.get_links(cluster_name_for_GRN_unit="global_annotation", alpha=10,
                            verbose_level=10, test_mode=False, n_jobs=-1)
    
    # filter the GRN (2000 edges)
    links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
    
    # Calculate network scores. It takes several minutes.
    links.get_score(n_jobs=-1)
    
    # save the Links object (for all cell-types)
    links.to_hdf5(file_path=output_path + "08_"+ data_id + "_celltype_GRNs.celloracle.links")
    
    # 2) cell-type specific GRNs - save for each cell-type
    # create a directory for each timepoint
    os.makedirs(output_path + "07_" + data_id, exist_ok=True)
    GRN_output_path = output_path + "07_" + data_id
    
    # save cell-type specific GRNs to the timepoint-specific directories
    for cluster in links.links_dict.keys():
        # Set cluster name
        cluster = cluster

        # Save as csv
        links.links_dict[cluster].to_csv(GRN_output_path + "/" + f"raw_GRN_for_{cluster}.csv")
    
    # End measuring time
    end_time = time.time()

    # calculate and print the elapsed time (for one timepoint)
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

    return links



####### LINUX TERMINAL COMMANDS #######
compute_cluster_specific_GRNs(output_path, RNAdata_path, baseGRN_path, 
                                    data_id, annotation, dim_reduce)