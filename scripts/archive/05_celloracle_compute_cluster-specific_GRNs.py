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

# visualization settings
%config InlineBackend.figure_format = 'retina'
%matplotlib inline

plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 300

def compute_cluster_specific_GRNs(output_path, RNAdata_path, baseGRN_path, 
                                    data_id, annotation, dim_reduce):
                                    #multiple_timepoints = False):
    """
    A function to compute cluster(cell-type) specific GRNs using CellOracle.
    It uses a base GRN built from the previous script (04_XXX.py)
    Note that we can subset the scRNA-seq datasets in different ways - i.e. cell-type annotation, timepoints, condition, etc.
    # Optional: For multiple timepoints, we will add an option for splitting out the data for "timepoints" before subsetting for "cluster".
    # 
    
    Parameters:
        output_path: filepath for the output files (below)
        RNAdata_path: filename for the csv file with peaks mapped to the nearest TSS and filtered for high cicero co-accessibility scores.
        baseGRN_path: filepath for the directory where the output dataframe will be saved
        data_id: identifier for the output dataframe file
        annotation: annotation class for the clusters (cell-types)
        dim_reduce: dim.reduction name
        # [TBD]
        multiple_timepoints: Boolean for whether there are multiple time points or not. (True/False)
        timepoints: a list of timepoints
    
    Returns: 
        df: a dataframe of peaks-by-TFs (base GRN)

    Examples:
        filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/"
        RNAdata_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/ZF_atlas_v01/ZF_atlas_v01.h5ad"
        baseGRN_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/05_TDR118_base_GRN_dataframe.parquet"
        data_id = "TDR118"
        annotation: "global_annotation"
        dim_reduce: "X_umap"
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
        baseGRN = co.data.load_zebrafish_promoter_base_GRN()
        baseGRN
    else:
        # # Load TF info which was made from our own zebrafish single-cell multiome dataset (15-somite)
        baseGRN = pd.read_parquet(baseGRN_path)

    # Check data
    print("checking the base GRN:")
    baseGRN.head()

    # Step 3. Compute the Oracle objects (if there are multiple timepoints, we can compute cell-type specific GRNs for all timepoints)
    # filepath for the output Oracle objects
    #output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/baseGRN_CisBP_RNA_zebrahub/"

    # all timepoints (making this as a default for now)
    timepoints = adata.obs.timepoint.unique().to_list()
    timepoints

    for stage in timepoints:
        # Instantiate Oracle object
        oracle = co.Oracle()
        
        # subset the anndata for one timepoint
        adata_subset = adata[adata.obs.timepoint==stage]
        
        # Step 3-1. Add the scRNA-seq (adata) to the Oracle object
        oracle.import_anndata_as_raw_count(adata=adata_subset,
                                        cluster_column_name="global_annotation", # annotation
                                        embedding_name="X_umap") # X_umap.cellranger.arc"
        
        # Step 3-2. Add the base GRN (TF info dataframe)
        oracle.import_TF_data(TF_info_matrix=baseGRN)
        
        # Step 3-3. KNN imputation
        # Perform PCA
        oracle.perform_PCA()

    #     # Select important PCs
    #     plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
    #     n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
    #     plt.axvline(n_comps, c="k")
    #     print(n_comps)
    #     n_comps = min(n_comps, 50)
        
        # number of cells
        n_cell = oracle.adata.shape[0]
        print(f"cell number is :{n_cell}")
        
        # number of k
        k = int(0.025*n_cell)
        print(f"Auto-selected k is :{k}")
        
        # KNN imputation
        oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                            b_maxl=k*4, n_jobs=4)

        # Save oracle object.output_filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/"
        oracle.to_hdf5(output_path + "06_"+ stage + ".celloracle.oracle")
    
    # Step 4. Compute cluster-specific (cell-type specific) GRNs for each timepoint (using the Oracle objects computed in Step 3.)
    for stage in timepoints:
        # Load the Oracle object for each timepoint
        oracle = co.load_hdf5(output_path + "06_"+ stage + ".celloracle.oracle")
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
        links.to_hdf5(file_path=output_path + "08_"+ stage + "_celltype_GRNs.celloracle.links")
        
        # 2) cell-type specific GRNs - save for each cell-type
        # create a directory for each timepoint
        os.makedirs(output_path + "07_" + stage, exist_ok=True)
        GRN_output_path = output_path + "07_" + stage
        
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