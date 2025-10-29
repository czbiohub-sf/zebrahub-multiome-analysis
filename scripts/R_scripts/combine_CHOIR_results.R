# Script to combine all processed CHOIR results
library(Seurat)
library(CHOIR)

# Define paths
base_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
filepath <- paste0(base_dir, "/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_counts/")

# Load the original Seurat object
seurat <- readRDS(paste0(filepath, "peaks_by_pseudobulks_leiden_0.4.rds"))
parent_clusters <- unique(seurat$CHOIR_parent_clusters)

# Wait for all jobs to complete (maximum 2 hours wait time)
wait_for_jobs <- function(filepath, parent_clusters, max_wait_time = 7200) {
  start_time <- Sys.time()
  all_files_present <- FALSE
  
  while (!all_files_present && (difftime(Sys.time(), start_time, units="secs") < max_wait_time)) {
    records_files <- file.path(filepath, paste0("records_", parent_clusters, ".rds"))
    existing_files <- sapply(records_files, file.exists)
    
    if (all(existing_files)) {
      all_files_present <- TRUE
      print("All CHOIR processing files are present!")
      break
    }
    
    print(paste("Waiting for jobs to complete... Found", sum(existing_files), "of", length(parent_clusters), "files"))
    Sys.sleep(60)
  }
  
  if (!all_files_present) {
    stop("Timeout waiting for CHOIR processing files")
  }
  
  return(records_files[existing_files])
}

# Wait for all files and get their paths
records_files <- wait_for_jobs(filepath, parent_clusters)

# Get all record files
subtree_records_list <- list()
for (pc in parent_clusters) {
  records_file <- paste0(filepath, "records_", pc, ".rds")
  subtree_records_list[[pc]] <- readRDS(records_file)
}

# Combine trees
seurat <- combineTrees(seurat, subtree_list = subtree_records_list)

# Save the final combined result
saveRDS(seurat, file = paste0(filepath, "peaks_by_pseudobulks_leiden_0.4_CHOIR.rds"))

# Clean up intermediate files (optional)
for (pc in parent_clusters) {
  file.remove(paste0(filepath, "subtree_", pc, ".rds"))
  file.remove(paste0(filepath, "records_", pc, ".rds"))
  file.remove(paste0(filepath, "processed_subtree_", pc, ".rds"))
} 