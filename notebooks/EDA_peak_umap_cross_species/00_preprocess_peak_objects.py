# %% [markdown]
# # Preprocess peak objects
# 
# This notebook preprocesses the peak objects for the EDA_peak_umap_cross_species notebook.
# sc_rapids jupyter kernel is used. (with GPU acceleration)
# 
# %% Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

import cupy as cp
import rapids_singlecell as rsc

# %% Dataset 1. Argelaguet 2022 mouse peak objects
# 1) file paths for Argelaguet 2022 mouse peak objects
peak_objects_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/PeakMatrix_anndata.h5ad"

# 2) load the peak objects
peak_objects = sc.read_h5ad(peak_objects_path)

# %% 
# 3) inspect the peak objects
print(peak_objects.shape)
print(peak_objects.var.shape)
print(peak_objects.obs.shape)

# 4) inspect the peak objects celltype and stage
print(peak_objects.obs["celltype.mapped"].value_counts())
print(peak_objects.obs["stage"].value_counts())

# 5) inspect the peak objects
print(peak_objects.var.head())
print(peak_objects.obs.head())

# %% pseudobulk the peak objects to create peaks-by-pseudobulk (celltype-AND-stage) matrix
# %% pseudobulk the peak objects to create peaks-by-pseudobulk (celltype-AND-stage) matrix

def analyze_peaks_with_normalization(
    adata, 
    celltype_key='celltype.mapped', 
    timepoint_key='stage'
):
    """
    Pseudobulk cells-by-peaks matrix to pseudobulk-by-peaks with median scaling normalization.
    
    1) Compute each cell's total_counts (sum of peaks/reads).
    2) For each (celltype, timepoint) group, compute the total_coverage 
       = sum of total_counts from all cells in that group.
    3) Create pseudobulk by summing (func='sum') each group's cells for the peaks matrix.
    4) The common_scale_factor = median of all group_total_coverage.
    5) For each group g, normalized_pseudobulk = raw_pseudobulk * (common_scale_factor / group_total_coverage[g]).

    Parameters
    ----------
    adata : AnnData
        Cells-by-peaks matrix
    celltype_key : str
        Column name in adata.obs for celltype (default: 'celltype.mapped')
    timepoint_key : str
        Column name in adata.obs for developmental stage (default: 'stage')

    Returns
    -------
    adata_pseudo : an AnnData with:
        - .X = raw pseudobulk counts (summed)
        - layers['normalized'] = scaled pseudobulk counts (median normalized)
        - layers['log_norm'] = log1p transformed normalized counts
        - obs['total_coverage'] = group's raw coverage
        - obs['scale_factor'] = how much that group's coverage was scaled
        - obs['n_cells'] = number of cells in each pseudobulk group
        - obs['mean_depth'] = mean depth per cell in each group
        - uns['common_scale_factor'] = the median coverage used for scaling
    """
    import scipy.sparse as sp
    
    # 1) total_counts per cell
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1))

    # 2) total_coverage per group (sum of total_counts)
    group_total_coverage = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].sum()

    # 3) Pseudobulk by summing group cells
    ident_cols = [celltype_key, timepoint_key]
    adata_pseudo = sc.get.aggregate(adata, ident_cols, func='sum')
    # Copy the summed counts into .X
    adata_pseudo.X = adata_pseudo.layers["sum"].copy()

    # Split the new obs index (e.g. "Astro_E7.5") back into celltype/timepoint
    celltype_timepoint = pd.DataFrame({
        'celltype': ['_'.join(x.split('_')[:-1]) for x in adata_pseudo.obs.index],
        'timepoint': [x.split('_')[-1] for x in adata_pseudo.obs.index]
    }, index=adata_pseudo.obs.index)

    # Prepare for normalized counts
    X = adata_pseudo.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # 4) common_scale_factor = median of group_total_coverage
    common_scale_factor = np.median(group_total_coverage.values)

    # 5) Rescale each group's pseudobulk
    normalized_counts = np.zeros_like(X)
    coverage_list = []
    scale_factor_list = []

    for i, idx in enumerate(adata_pseudo.obs.index):
        ct = celltype_timepoint.loc[idx, 'celltype']
        tp = celltype_timepoint.loc[idx, 'timepoint']
        coverage_g = group_total_coverage[(ct, tp)]

        # Scale factor = common_scale_factor / group's total coverage
        scale_g = common_scale_factor / coverage_g
        normalized_counts[i, :] = X[i, :] * scale_g
        
        coverage_list.append(coverage_g)
        scale_factor_list.append(scale_g)

    # Store normalized counts in a new layer
    adata_pseudo.layers['normalized'] = normalized_counts

    # Record coverage and scaling info in .obs
    adata_pseudo.obs['total_coverage'] = coverage_list
    adata_pseudo.obs['scale_factor'] = scale_factor_list

    # Optionally, also store #cells and mean_depth
    group_ncells = adata.obs.groupby([celltype_key, timepoint_key]).size()
    group_mean_depth = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].mean()

    n_cells_list = []
    mean_depth_list = []
    for i, idx in enumerate(adata_pseudo.obs.index):
        ct = celltype_timepoint.loc[idx, 'celltype']
        tp = celltype_timepoint.loc[idx, 'timepoint']
        n_cells_list.append(group_ncells[(ct, tp)])
        mean_depth_list.append(group_mean_depth[(ct, tp)])
    adata_pseudo.obs['n_cells'] = n_cells_list
    adata_pseudo.obs['mean_depth'] = mean_depth_list

    # Save the "common" scale factor in .uns
    adata_pseudo.uns['common_scale_factor'] = common_scale_factor
    
    # Apply log1p transformation
    adata_pseudo.X = adata_pseudo.layers["normalized"].copy()
    sc.pp.log1p(adata_pseudo)
    adata_pseudo.layers["log_norm"] = adata_pseudo.X.copy()
    
    # Reset X back to raw pseudobulk counts
    adata_pseudo.X = adata_pseudo.layers["sum"].copy()

    return adata_pseudo


# %% Run the pseudobulking
adata_pseudo = analyze_peaks_with_normalization(
    peak_objects,
    celltype_key='celltype.mapped',
    timepoint_key='stage'
)

print(adata_pseudo)
print(f"Shape: {adata_pseudo.shape}")
print(f"Common scale factor: {adata_pseudo.uns['common_scale_factor']}")

# %% Inspect the results
print(adata_pseudo.obs[['total_coverage', 'scale_factor', 'n_cells', 'mean_depth']].head(10))

# %% Optional: Transpose to get peaks-by-pseudobulk for UMAP analysis
# This is useful if you want to embed peaks instead of pseudobulk groups
adata_pseudo.X = adata_pseudo.layers["log_norm"]
peaks_by_pseudobulk = adata_pseudo.copy().T
print(f"Transposed shape (peaks-by-pseudobulk): {peaks_by_pseudobulk.shape}")


## save the adata objects
adata_pseudo.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/pb_by_celltype_stage_peaks.h5ad")
peaks_by_pseudobulk.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage.h5ad")

# %% computing UMAP
# moves `.X` to the GPU
# rsc.get.anndata_to_GPU(peaks_by_pseudobulk) 

# Compute UMAP
# rsc.pp.scale(peaks_pb_norm) # scale is for each "features" to have similar importance...
rsc.pp.pca(peaks_by_pseudobulk, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_by_pseudobulk, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_by_pseudobulk, min_dist=0.3, random_state=42)

# plot the UMAP
sc.pl.umap(peaks_by_pseudobulk, color="chr")
# %% Optional: Save the pseudobulked object
# adata_pseudo.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/pseudobulk_by_celltype_stage.h5ad")