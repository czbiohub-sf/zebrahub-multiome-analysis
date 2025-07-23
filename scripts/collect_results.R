#!/usr/bin/env Rscript

# Standalone R script to collect and aggregate Cicero results from individual chromosome jobs
# Usage: Rscript collect_results.R seurat_object_path assay output_path data_id peaktype script_dir

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6) {
  stop("Usage: Rscript collect_results.R seurat_object_path assay output_path data_id peaktype script_dir")
}

seurat_object_path <- args[1]
assay <- args[2]
output_path <- args[3]
data_id <- args[4]
peaktype <- args[5]
script_dir <- args[6]

# Set library paths
.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(cicero))
library(Seurat)

cat("=== CICERO RESULTS COLLECTION ===\n")
cat("Collection started at:", as.character(Sys.time()), "\n\n")

# Print parameters
cat("=== PARAMETERS ===\n")
cat("Seurat object path:", seurat_object_path, "\n")
cat("Assay:", assay, "\n")
cat("Output path:", output_path, "\n")
cat("Data ID:", data_id, "\n")
cat("Peak type:", peaktype, "\n")
cat("Script directory:", script_dir, "\n\n")

# Define chromosomes (excluding MT)
chromosomes <- c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                "21", "22", "23", "24", "25")

# Validate input files
cat("=== VALIDATING INPUT FILES ===\n")
if (!file.exists(seurat_object_path)) {
  stop("Seurat object file not found: ", seurat_object_path)
}

if (!dir.exists(script_dir)) {
  stop("Script directory not found: ", script_dir)
}

if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
  cat("Created output directory:", output_path, "\n")
}

# Load Seurat object for peaks
cat("Loading Seurat object...\n")
seurat_object <- readRDS(seurat_object_path)
all_peaks <- row.names(seurat_object@assays[[assay]]@data)
cat("Total peaks in Seurat object:", length(all_peaks), "\n\n")

# Check for chromosome model files
cat("=== CHECKING CHROMOSOME MODEL FILES ===\n")
model_files <- file.path(script_dir, paste0("cicero_model_", chromosomes, ".rds"))
existing_files <- sapply(model_files, file.exists)

cat("Model file status:\n")
for (i in seq_along(chromosomes)) {
  status <- if (existing_files[i]) "FOUND" else "MISSING"
  cat(sprintf("  Chromosome %s: %s\n", chromosomes[i], status))
  if (existing_files[i]) {
    file_size <- file.info(model_files[i])$size
    cat(sprintf("    File size: %.2f MB\n", file_size / 1024^2))
  }
}
cat("\n")

# Check if any files exist
if (sum(existing_files) == 0) {
  stop("No cicero model files found in ", script_dir)
}

# Warn about missing files
if (sum(existing_files) < length(chromosomes)) {
  missing_chrs <- chromosomes[!existing_files]
  cat("WARNING: Missing model files for chromosomes:", paste(missing_chrs, collapse = ", "), "\n")
  cat("Proceeding with available chromosomes...\n\n")
}

# Load cicero models
cat("=== LOADING CICERO MODELS ===\n")
cat("Loading cicero models...\n")
cicero_models <- list()
connection_stats <- data.frame(
  chromosome = character(),
  n_connections = integer(),
  model_size_mb = numeric(),
  stringsAsFactors = FALSE
)

for (i in which(existing_files)) {
  chr_name <- chromosomes[i]
  cat("  Loading chromosome", chr_name, "...\n")
  
  tryCatch({
    model <- readRDS(model_files[i])
    cicero_models[[chr_name]] <- model
    
    # Get model statistics
    file_size <- file.info(model_files[i])$size / 1024^2
    
    # Try to get connection count
    n_connections <- 0
    if (is.list(model)) {
      tryCatch({
        temp_connections <- assemble_connections(model)
        n_connections <- nrow(temp_connections)
      }, error = function(e) {
        cat("    Could not count connections for chromosome", chr_name, "\n")
      })
    }
    
    connection_stats <- rbind(connection_stats, data.frame(
      chromosome = chr_name,
      n_connections = n_connections,
      model_size_mb = file_size,
      stringsAsFactors = FALSE
    ))
    
    cat("    Successfully loaded - Connections:", n_connections, ", Size:", round(file_size, 2), "MB\n")
  }, error = function(e) {
    cat("    ERROR loading chromosome", chr_name, ":", e$message, "\n")
  })
}
cat("\n")

# Filter out NULL values
cicero_models <- Filter(Negate(is.null), cicero_models)

if (length(cicero_models) == 0) {
  stop("No valid cicero models could be loaded")
}

# Assemble connections
cat("=== ASSEMBLING CONNECTIONS ===\n")
cat("Successfully loaded models for", length(cicero_models), "chromosomes\n")
cat("Assembling connections...\n")
start_time <- Sys.time()
all_connections <- do.call("rbind", lapply(cicero_models, assemble_connections))
end_time <- Sys.time()
cat("Connection assembly completed in:", round(difftime(end_time, start_time, units = "mins"), 2), "minutes\n")

# Log final connection statistics
total_connections <- nrow(all_connections)
cat("\n=== FINAL RESULTS SUMMARY ===\n")
cat("Total connections assembled:", total_connections, "\n")
if (total_connections > 0) {
  cat("Connection statistics:\n")
  if ("coaccess" %in% colnames(all_connections)) {
    cat("  - Mean co-accessibility score:", round(mean(all_connections$coaccess, na.rm = TRUE), 4), "\n")
    cat("  - Median co-accessibility score:", round(median(all_connections$coaccess, na.rm = TRUE), 4), "\n")
    cat("  - Max co-accessibility score:", round(max(all_connections$coaccess, na.rm = TRUE), 4), "\n")
  }
  cat("  - Connection data dimensions:", paste(dim(all_connections), collapse = "x"), "\n")
  cat("  - Column names:", paste(colnames(all_connections), collapse = ", "), "\n")
}

# Print per-chromosome statistics
cat("\nPer-chromosome connection summary:\n")
print(connection_stats)
cat("\n")

# Save results
cat("=== SAVING RESULTS ===\n")
peaks_file <- paste0(output_path, "01_", data_id, "_", peaktype, "_peaks.csv")
connections_file <- paste0(output_path, "02_", data_id, "_cicero_connections_", peaktype, "_peaks.csv")
stats_file <- paste0(output_path, "03_", data_id, "_cicero_stats_", peaktype, "_peaks.csv")

cat("Saving peaks file...\n")
write.csv(all_peaks, file = peaks_file, row.names = FALSE)
peaks_size <- file.info(peaks_file)$size / 1024^2
cat("  - Peaks file:", peaks_file, "\n")
cat("  - Peaks count:", length(all_peaks), "\n")
cat("  - File size:", round(peaks_size, 2), "MB\n\n")

cat("Saving connections file...\n")
write.csv(all_connections, file = connections_file, row.names = FALSE)
connections_size <- file.info(connections_file)$size / 1024^2
cat("  - Connections file:", connections_file, "\n")
cat("  - Connections count:", nrow(all_connections), "\n")
cat("  - File size:", round(connections_size, 2), "MB\n\n")

cat("Saving connection statistics...\n")
write.csv(connection_stats, file = stats_file, row.names = FALSE)
stats_size <- file.info(stats_file)$size / 1024^2
cat("  - Statistics file:", stats_file, "\n")
cat("  - File size:", round(stats_size, 2), "MB\n\n")

cat("=== ANALYSIS COMPLETED SUCCESSFULLY ===\n")
cat("Collection completed at:", as.character(Sys.time()), "\n")
cat("\nFinal summary:\n")
cat("  - Total peaks:", length(all_peaks), "\n")
cat("  - Total connections:", nrow(all_connections), "\n")
cat("  - Chromosomes processed:", length(cicero_models), "\n")
cat("  - Output files created: 3\n")
cat("\nOutput files:\n")
cat("  1.", peaks_file, "\n")
cat("  2.", connections_file, "\n")
cat("  3.", stats_file, "\n") 