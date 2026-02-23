"""
Utility functions for pseudobulk analysis of peak data.
"""

import numpy as np
import pandas as pd
import scanpy as sc


def analyze_peaks_with_normalization(
    adata,
    celltype_key='annotation_ML_coarse',
    timepoint_key='dev_stage'
):
    """
    1) Compute each cell's total_counts (sum of peaks/reads).
    2) For each (celltype, timepoint) group, compute the total_coverage
       = sum of total_counts from all cells in that group.
    3) Create pseudobulk by summing (func='sum') each group's cells for the peaks matrix.
    4) The common_scale_factor = median of all group_total_coverage.
    5) For each group g, normalized_pseudobulk = raw_pseudobulk * (common_scale_factor / group_total_coverage[g]).

    Returns
    -------
    adata_pseudo : an AnnData with:
        - .X = raw pseudobulk counts
        - layers['normalized'] = scaled pseudobulk counts
        - obs['total_coverage'] = group's raw coverage
        - obs['scale_factor'] = how much that group's coverage was scaled
        - obs['n_cells'] and obs['mean_depth'] optionally stored as well
        - uns['common_scale_factor'] = the median coverage used for scaling
    """

    # 1) total_counts per cell
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1))

    # 2) total_coverage per group (sum of total_counts)
    group_total_coverage = adata.obs.groupby([celltype_key, timepoint_key])['total_counts'].sum()

    # 3) Pseudobulk by summing group cells
    ident_cols = [celltype_key, timepoint_key]
    adata_pseudo = sc.get.aggregate(adata, ident_cols, func='sum')
    # Copy the summed counts into .X
    adata_pseudo.X = adata_pseudo.layers["sum"].copy()

    # Split the new obs index (e.g. "Astro_dev_stage1") back into celltype/timepoint
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

    return adata_pseudo
