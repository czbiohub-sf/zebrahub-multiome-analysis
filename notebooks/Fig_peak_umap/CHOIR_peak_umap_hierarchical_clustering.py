# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Global R
#     language: R
#     name: r_4.3
# ---

# %% [markdown]
# ## Use CHOIR to find optimal resolution of clustering (in R)
#
# - last updated: 04/21/2025
# - https://www.choirclustering.com/articles/atlas_scale_data.html 

# %%
# load other libraries
# library(cicero)
library(Signac)
library(Seurat)
library(SeuratWrappers)
library(SeuratDisk) # to convert the h5ad to h5Seurat
library(dplyr)
library(Matrix)

.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(CHOIR))
# withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(cicero))

# %%
# import the counts and metadata from the h5ad (exported)
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_counts/"

obs_df <- read.csv(paste0(filepath, "/obs.csv"), row.names = 1)
var_df <- read.csv(paste0(filepath, "/var.csv"), row.names = 1)
mat <- readMM(paste0(filepath, "/sparse_matrix.mtx"))
mat <- t(mat)
# add the row and column names
colnames(mat) <- rownames(obs_df)
rownames(mat) <- rownames(var_df)

# %%
# Then create a Seurat object:
seurat <- CreateSeuratObject(counts = mat, meta.data = obs_df, assay="peaks_integrated")
seurat


# %%
# save the seurat object
saveRDS(seurat, file = paste0(filepath, "peaks_by_pseudobulks_leiden_0.4.rds"))

# %%
seurat@assays$peaks_integrated@data

# %%
# run CHOIR (build and prune trees)
seurat <- CHOIR(seurat, n_cores=8, use_assay="peaks_integrated", use_slot="data")
seurat

# %%

# %% [markdown]
# ## testing the atlas-scale CHOIR run
# - use subset to parallelize the hierarchical clustering
#

# %%
# Step 1. Generate parent clusters
seurat <- buildParentTree(seurat, use_assay="peaks_integrated", 
                         use_slot="data", n_cores=8,
                         cluster_params = list(algorithm = 4, group.singletons = TRUE))
seurat

# %%
# Step 2. Subset each parent cluster
# Suppose we look at unique parent clusters
parent_clusters <- unique(seurat$CHOIR_parent_clusters)

# For each parent cluster, subset the Seurat object
subtree_objects_list <- list()
for (pc in parent_clusters) {
  subtree_s <- subset(seurat, subset = CHOIR_parent_clusters == pc)
#   # If the subset is still too large (> 450k cells), do minimal splits
#   # (Your data has only 150 columns, so likely no need, but here’s how):
#   if (ncol(subtree_s) > 450000) {
#     # Identify subclusters at a starting resolution, then subset further
#     subtree_s <- subtree_s %>% 
#       FindVariableFeatures() %>% 
#       ScaleData() %>% 
#       RunPCA() %>%
#       FindNeighbors()
    
#     # The internal CHOIR function to guess a good starting resolution
#     starting_resolution <- CHOIR:::.getStartingResolution(
#       subtree_s@graphs$RNA_snn,
#       cluster_params = list(algorithm = 1, group.singletons = TRUE),
#       random_seed = 1, 
#       verbose = TRUE
#     )
    
#     subtree_s <- FindClusters(
#       subtree_s, 
#       resolution = starting_resolution[["starting_resolution"]]
#     )
    
#     # You might now subset again for each new cluster (0, 1, 2, etc.)
#     # e.g., subtree_s_0 <- subset(subtree_s, subset = seurat_clusters == 0)
#     # subtree_s_1 <- subset(subtree_s, subset = seurat_clusters == 1)
#     # ...
#     # Then store them in subtree_objects_list
#   }
  
  subtree_objects_list[[pc]] <- subtree_s
}

subtree_records_list <- list()

for (pc in names(subtree_objects_list)) {
  subtree_s <- subtree_objects_list[[pc]]
  
  if (ncol(subtree_s) > 1) {
    # Call CHOIR with the same downsampling_rate you used in buildParentTree
    subtree_s <- CHOIR(
      subtree_s,
      key = "CHOIR_subtree",
      downsampling_rate = seurat@misc$CHOIR$parameters$buildParentTree_parameters$downsampling_rate
      # Additional params as needed
    )
  } else {
    # If there's effectively 1 or 0 columns, set cluster label manually
    subtree_s@misc$CHOIR_subtree$clusters$CHOIR_clusters_0.05 <- data.frame(
      CellID = colnames(subtree_s),
      CHOIR_clusters_0.05 = 1,
      Record_cluster_label = "P0_L0_1"
    )
    subtree_s@misc$CHOIR_subtree$parameters <- seurat@misc$CHOIR$parameters
  }
  
  # Save back
  subtree_objects_list[[pc]] <- subtree_s
  subtree_records_list[[pc]] <- getRecords(subtree_s, key = "CHOIR_subtree")
}

# %%
# Step 4. Combine subtrees and standardize significance thresholds
# Run the CHOIR function combineTrees on the complete set of records extracted in step 2. In doing so, the significance threshold will be standardized across all clustering trees, yielding a final set of clusters.
seurat <- combineTrees(seurat,
                       subtree_list = subtree_records_list)
