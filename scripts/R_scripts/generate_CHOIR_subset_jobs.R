# Script to generate SLURM jobs for CHOIR subset analysis
# Load required libraries
library(Seurat)
.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(CHOIR))

# Define paths
base_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
filepath <- paste0(base_dir, "/data/annotated_data/objects_v2/")
scripts_dir <- paste0(base_dir, "/zebrahub-multiome-analysis/scripts/R_scripts")
jobs_dir <- paste0(scripts_dir, "/choir_jobs")

# Create jobs directory if it doesn't exist
dir.create(jobs_dir, recursive = TRUE, showWarnings = FALSE)

# Load the Seurat object with parent clusters
seurat <- readRDS(paste0(filepath, "peaks_by_pseudobulks_leiden_0.4.rds"))
parent_clusters <- unique(seurat$CHOIR_parent_clusters)

# Get the downsampling rate from the parent object
downsampling_rate <- seurat@misc$CHOIR$parameters$buildParentTree_parameters$downsampling_rate

# Save parent clusters to separate RDS files
for (pc in parent_clusters) {
  subtree_s <- subset(seurat, subset = CHOIR_parent_clusters == pc)
  saveRDS(subtree_s, file = paste0(filepath, "subtree_", pc, ".rds"))
}

# Function to wait for all jobs to complete
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

# Generate SLURM job script for each parent cluster
for (pc in parent_clusters) {
  job_script <- paste0(jobs_dir, "/run_choir_", pc, ".sh")
  
  slurm_content <- sprintf('#!/bin/bash
#SBATCH --job-name=choir_%s
#SBATCH --output=%s/choir_%s_%%j.out
#SBATCH --error=%s/choir_%s_%%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

module load R/4.3

Rscript %s/process_CHOIR_subset.R %s %s
', 
    pc, jobs_dir, pc, jobs_dir, pc, scripts_dir, pc, downsampling_rate)
  
  # Write the SLURM script
  writeLines(slurm_content, job_script)
  
  # Make the script executable
  system(paste("chmod +x", job_script))
}

# Create a master script to submit all jobs and combine results
master_script <- paste0(jobs_dir, "/submit_all_choir_jobs.sh")
master_content <- c(
  "#!/bin/bash",
  paste0("cd ", jobs_dir),
  "# Submit all CHOIR jobs",
  paste("for script in run_choir_*.sh; do",
        "  sbatch $script",
        "  sleep 1",
        "done",
        sep = "\n"),
  "",
  "# Wait for all jobs to complete and combine results",
  paste0("Rscript ", scripts_dir, "/combine_CHOIR_results.R")
)

writeLines(master_content, master_script)
system(paste("chmod +x", master_script)) 