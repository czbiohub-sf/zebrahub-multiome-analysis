# Script to process individual CHOIR subset
library(Seurat)
.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(CHOIR))

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript process_CHOIR_subset.R <parent_cluster> <downsampling_rate>")
}
parent_cluster <- args[1]
downsampling_rate <- as.numeric(args[2])

# Define paths
base_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
filepath <- paste0(base_dir, "/data/annotated_data/objects_v2/CHOIR/")

# Load the subtree
subtree_s <- readRDS(paste0(filepath, "subtree_", parent_cluster, ".rds"))

# Process the subtree
if (ncol(subtree_s) > 1) {
  # Call CHOIR with the provided downsampling_rate
  subtree_s <- CHOIR(
    subtree_s,
    key = "CHOIR_subtree",
    downsampling_rate = downsampling_rate,
    n_cores = 8
  )
} else {
  # If there's effectively 1 or 0 columns, set cluster label manually
  subtree_s@misc$CHOIR_subtree$clusters$CHOIR_clusters_0.05 <- data.frame(
    CellID = colnames(subtree_s),
    CHOIR_clusters_0.05 = 1,
    Record_cluster_label = "P0_L0_1"
  )
  subtree_s@misc$CHOIR_subtree$parameters <- list(downsampling_rate = downsampling_rate)
}

# Save the processed subtree
saveRDS(subtree_s, file = paste0(filepath, "processed_subtree_", parent_cluster, ".rds"))

# Save just the records for later combining
records <- getRecords(subtree_s, key = "CHOIR_subtree")
saveRDS(records, file = paste0(filepath, "records_", parent_cluster, ".rds")) 