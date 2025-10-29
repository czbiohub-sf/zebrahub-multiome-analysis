# This script computes the hierarchical clustering of the peaks-by-pseudobulk object using the CHOIR package
# reference: https://www.choirclustering.com/articles/atlas_scale_data.html
# NOTE. run this script using slurm (HPC) with the following modules:
# module load R/4.3

# Load the necessary libraries
library(Signac)
library(Seurat)
library(SeuratWrappers)
library(SeuratDisk) # to convert the h5ad to h5Seurat
library(dplyr)
library(Matrix)
# Load the CHOIR package
.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(CHOIR))

# Step 1. Import the counts and metadata from the h5ad (exported)
# define the filepath
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_counts/"
# import the counts and metadata from the h5ad (exported)
obs_df <- read.csv(paste0(filepath, "/obs.csv"), row.names = 1)
var_df <- read.csv(paste0(filepath, "/var.csv"), row.names = 1)
mat <- readMM(paste0(filepath, "/sparse_matrix.mtx"))
mat <- t(mat) # transpose the matrix (psuedobulk-by-peaks matrix)
# add the row and column names
colnames(mat) <- rownames(obs_df)
rownames(mat) <- rownames(var_df)

# Then create a Seurat object:
seurat <- CreateSeuratObject(counts = mat, meta.data = obs_df, assay = "peaks_integrated")
seurat
# save the seurat object
saveRDS(seurat, file = paste0(filepath, "peaks_by_pseudobulks_leiden_0.4.rds"))

# Step 2. Compute the hierarchical clustering of the peaks-by-pseudobulk object using the CHOIR package
# # load the seurat object
# seurat <- readRDS(paste0(filepath, "peaks_by_pseudobulks_leiden_0.4.rds"))

# compute the hierarchical clustering of the peaks-by-pseudobulk object using the CHOIR package
# seurat <- CHOIR(seurat, n_cores=64, use_assay="peaks_integrated", use_slot="data")

# Step 2-1. Generate parent clusters
# NOTE that the algorithm 4 is the leiden clustering algorithm (and we're grouping singletons)
seurat <- buildParentTree(seurat, use_assay="peaks_integrated", 
                         use_slot="data", n_cores=64,
                         cluster_params = list(algorithm = 4, group.singletons = TRUE))

# Step 2-2. Subset each parent cluster
# Subset each parent cluster and save it as a separate object.
# subtree_s <- subset(seurat, subset = CHOIR_parent_clusters == "P1")
parent_clusters <- unique(seurat$CHOIR_parent_clusters)

# Create a directory to store the subsets and job scripts
subset_dir <- file.path(filepath, "subtree_subsets")
dir.create(subset_dir, showWarnings = FALSE, recursive = TRUE)

# For each parent cluster, subset the Seurat object
subtree_objects_list <- list()
for (pc in parent_clusters) {
  subtree_s <- subset(seurat, subset = CHOIR_parent_clusters == pc)
#     # Add the parent cluster to the metadata
#   subtree_objects_list[[pc]] <- subtree_s

  # Save each subset to an RDS
  rds_name <- file.path(subset_dir, paste0("subtree_", pc, ".rds"))
  saveRDS(subtree_s, rds_name)
  message("Saved subset for parent cluster ", pc, " to ", rds_name)
}

# Step 2-3. Run CHOIR on each subsetted parent cluster
subtree_records_list <- list()
# Compute the child clusters for each parent cluster
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
# Step 2-4. Combine subtrees and standardize significance thresholds
# Run the CHOIR function combineTrees on the complete set of records extracted in step 2. In doing so, the significance threshold will be standardized across all clustering trees, yielding a final set of clusters.
seurat <- combineTrees(seurat,
                       subtree_list = subtree_records_list)

# save the seurat object
saveRDS(seurat, file = paste0(filepath, "peaks_by_pseudobulks_leiden_0.4_CHOIR.rds"))



