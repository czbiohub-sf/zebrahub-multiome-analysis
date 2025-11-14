# An R script to run Cicero for a Seurat object with a ChromatinAssay object
# Note1: added the option to choose the peak profiles (i.e. peaks_celltype, peaks_bulk, peaks_merged)
# Note2: added the option to choose the dimensionality reduction (i.e. UMAP, UMAP.ATAC, PCA, etc.)
# Update (1/22/2024): I've added the parallelization option for the cicero computation.
# IMPROVED VERSION: Fixed major issues with library loading, job synchronization, and error handling

# Set up library paths and load libraries
.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(monocle3))
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(cicero))

# load other libraries
#library(cicero)
library(Signac)
library(Seurat)
library(SeuratWrappers)
library(GenomeInfoDb)

# Function to validate command line arguments
validate_args <- function(args) {
  if (length(args) != 7) {
    stop("Usage: Rscript run_02_compute_CCANS_cicero_parallelized.R seurat_object_path assay dim_reduced output_path data_id peaktype shell_script_dir")
  }
  
  # Check if seurat object file exists
  if (!file.exists(args[1])) {
    stop(paste("Seurat object file not found:", args[1]))
  }
  
  # Check if output directory exists or can be created
  if (!dir.exists(dirname(args[4]))) {
    stop(paste("Output directory does not exist:", dirname(args[4])))
  }
  
  return(TRUE)
}

# Parse and validate command-line arguments
args <- commandArgs(trailingOnly = TRUE)
validate_args(args)

seurat_object_path <- args[1]
assay <- args[2]
dim_reduced <- args[3]
output_path <- args[4] 
data_id <- args[5] 
peaktype <- args[6]
shell_script_dir <- args[7]

# Create script directory using absolute paths
script_dir <- file.path(shell_script_dir, "cicero_integrated_peaks")
dir.create(script_dir, showWarnings = FALSE, recursive = TRUE)

# Function to safely read RDS files
safe_readRDS <- function(file_path) {
  tryCatch({
    readRDS(file_path)
  }, error = function(e) {
    stop(paste("Error reading RDS file:", file_path, "-", e$message))
  })
}

# Import the Seurat object
print(paste("Loading Seurat object from:", seurat_object_path))
seurat_object <- safe_readRDS(seurat_object_path)

# Validate assay exists
if (!assay %in% names(seurat_object@assays)) {
  stop(paste("Assay", assay, "not found in Seurat object. Available assays:", 
             paste(names(seurat_object@assays), collapse = ", ")))
}

# Set default assay
DefaultAssay(seurat_object) <- assay
print(paste0("Default assay set to: ", assay))

# Convert to CellDataSet (CDS) format
seurat_object.cds <- as.cell_data_set(x = seurat_object)
print("CDS object created") 

# Validate dimensionality reduction
print("Available reduced dimensions:")
available_dims <- names(reducedDims(seurat_object.cds))
print(available_dims)

dim_reduced <- toupper(dim_reduced)
if (!dim_reduced %in% available_dims) {
  stop(paste("Dimensionality reduction", dim_reduced, "not found. Available:", 
             paste(available_dims, collapse = ", ")))
}

# Get the reduced coordinates
reduced_coords <- reducedDims(seurat_object.cds)[[dim_reduced]]
print(paste("Using dimensionality reduction:", dim_reduced))
print(paste("Dimensions:", paste(dim(reduced_coords), collapse = "x")))

# Create the cicero object
seurat_object.cicero <- make_cicero_cds(seurat_object.cds, reduced_coordinates = reduced_coords)
print("Cicero object created")

# Define the genomic length dataframe (GRCz11 chromosome lengths)
chr_lengths <- setNames(
    c(59578282, 59640629, 62628489, 78093715, 72500376, 60270060, 74282399,
      54304671, 56459846, 45420867, 45484837, 49182954, 52186027, 52660232,
      48040578, 55381981, 53969382, 51023478, 48449771, 55201332, 45934066,
      39133080, 46144548, 42173229, 37502051, 16596),
    c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
      "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
      "21", "22", "23", "24", "25", "MT")
)

chr_lengths <- setNames(as.integer(chr_lengths), names(chr_lengths))

# Assign chromosome lengths to seqinfo
seurat_object@assays[[assay]]@annotation@seqinfo@seqlengths <- chr_lengths

# Verify assignment
if (any(is.na(seurat_object@assays[[assay]]@annotation@seqinfo@seqlengths))) {
    stop("Seqlengths assignment failed - some lengths are NA")
}
print("Seqlengths successfully assigned")

# Create genome dataframe
genome.df <- data.frame(
    chr = names(chr_lengths),
    length = as.numeric(chr_lengths),
    stringsAsFactors = FALSE
)

# Save genome dataframe
genome_df_path <- file.path(script_dir, "genome_df.rds")
saveRDS(genome.df, file = genome_df_path)
print(paste("Genome dataframe saved to:", genome_df_path))

# Estimate distance parameter
print("Estimating distance parameter...")
distance_param <- estimate_distance_parameter(seurat_object.cicero, 
                                              genomic_coords = genome.df)
mean_distance_param <- mean(distance_param)
print(paste0("Mean distance parameter: ", mean_distance_param))

# Prepare chromosomes for processing
chromosomes <- unique(genome.df$chr)
chromosomes <- chromosomes[chromosomes != "MT"]  # Exclude mitochondrial chromosome
print(paste("Processing chromosomes:", paste(chromosomes, collapse = ", ")))

# Function to create SLURM job script
create_slurm_job <- function(current_chr, script_dir, mean_distance_param) {
  # Create chromosome-specific dataset
  chr_dataset <- subset(seurat_object.cicero, 
                        subset = current_chr == rowData(seurat_object.cicero)$chr)
  chr_dataset_path <- file.path(script_dir, paste0("seurat_chr_", current_chr, ".rds"))
  saveRDS(chr_dataset, file = chr_dataset_path)
  
  # Log the number of peaks for this chromosome
  n_peaks_chr <- nrow(chr_dataset)
  print(paste("Chromosome", current_chr, "has", n_peaks_chr, "peaks"))
  
  # Create SLURM script
  script_name <- file.path(script_dir, paste0("cicero_job_", current_chr, ".sh"))
  
  # Determine memory based on chromosome size
  chr_size <- chr_lengths[current_chr]
  memory_gb <- max(5, min(20, ceiling(chr_size / 10000000)))  # 5-20GB based on chr size
  
  script_content <- paste0(
    "#!/bin/bash\n",
    "#SBATCH --job-name=cicero_chr_", current_chr, "\n",
    "#SBATCH --output=", file.path(script_dir, paste0("cicero_chr_", current_chr, "_%j.out")), "\n",
    "#SBATCH --error=", file.path(script_dir, paste0("cicero_chr_", current_chr, "_%j.err")), "\n",
    "#SBATCH --time=12:00:00\n",
    "#SBATCH --mem=", memory_gb, "G\n",
    "#SBATCH --mail-type=FAIL\n",
    "#SBATCH --mail-user=yang-joon.kim@czbiohub.org\n",
    "module load R/4.3\n",
    "Rscript ", file.path(script_dir, paste0("run_cicero_chr_", current_chr, ".R")), "\n"
  )
  
  writeLines(script_content, script_name)
  
  # Create R script for this chromosome
  r_script_name <- file.path(script_dir, paste0("run_cicero_chr_", current_chr, ".R"))
  r_script_content <- paste0(
    "# Set library paths\n",
    ".libPaths('/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib')\n",
    "withr::with_libpaths(new = '/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib', library(monocle3))\n",
    "withr::with_libpaths(new = '/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib', library(cicero))\n",
    "library(Seurat)\n\n",
    "# Error handling function\n",
    "safe_operation <- function(expr, error_msg) {\n",
    "  tryCatch(expr, error = function(e) {\n",
    "    cat('ERROR:', error_msg, '\\n')\n",
    "    cat('Details:', e$message, '\\n')\n",
    "    quit(status = 1)\n",
    "  })\n",
    "}\n\n",
    "cat('=== CICERO ANALYSIS FOR CHROMOSOME ", current_chr, " ===\\n')\n",
    "cat('Job started at:', as.character(Sys.time()), '\\n')\n\n",
    "# Load data\n",
    "cat('Loading genome dataframe...\\n')\n",
    "genome.df <- safe_operation(\n",
    "  readRDS('", genome_df_path, "'),\n",
    "  'Failed to load genome dataframe'\n",
    ")\n\n",
    "cat('Loading chromosome dataset...\\n')\n",
    "chr_dataset <- safe_operation(\n",
    "  readRDS('", chr_dataset_path, "'),\n",
    "  'Failed to load chromosome dataset'\n",
    ")\n\n",
    "# Log dataset information\n",
    "n_peaks <- nrow(chr_dataset)\n",
    "n_cells <- ncol(chr_dataset)\n",
    "cat('Chromosome ", current_chr, " dataset loaded:\\n')\n",
    "cat('  - Number of peaks:', n_peaks, '\\n')\n",
    "cat('  - Number of cells:', n_cells, '\\n')\n",
    "cat('  - Chromosome length:', genome.df[genome.df$chr == '", current_chr, "', 'length'], 'bp\\n')\n\n",
    "cat('Generating cicero models...\\n')\n",
    "start_time <- Sys.time()\n",
    "cicero_model <- safe_operation(\n",
    "  generate_cicero_models(chr_dataset, \n",
    "                        distance_parameter = ", mean_distance_param, ",\n",
    "                        genomic_coords = genome.df),\n",
    "  'Failed to generate cicero models'\n",
    ")\n",
    "end_time <- Sys.time()\n",
    "cat('Cicero model generation completed in:', round(difftime(end_time, start_time, units = 'mins'), 2), 'minutes\\n')\n\n",
    "# Log model information\n",
    "if (is.list(cicero_model) && length(cicero_model) > 0) {\n",
    "  cat('Cicero model generated successfully\\n')\n",
    "  cat('  - Model type:', class(cicero_model), '\\n')\n",
    "  cat('  - Model length:', length(cicero_model), '\\n')\n",
    "} else {\n",
    "  cat('WARNING: Cicero model appears to be empty or invalid\\n')\n",
    "}\n\n",
    "cat('Saving cicero model...\\n')\n",
    "output_file <- '", file.path(script_dir, paste0("cicero_model_", current_chr, ".rds")), "'\n",
    "safe_operation(\n",
    "  saveRDS(cicero_model, file = output_file),\n",
    "  'Failed to save cicero model'\n",
    ")\n\n",
    "# Verify file was saved and get size\n",
    "if (file.exists(output_file)) {\n",
    "  file_size <- file.info(output_file)$size\n",
    "  cat('Model saved successfully:\\n')\n",
    "  cat('  - File:', output_file, '\\n')\n",
    "  cat('  - Size:', round(file_size / 1024^2, 2), 'MB\\n')\n",
    "} else {\n",
    "  cat('ERROR: Model file was not created\\n')\n",
    "}\n\n",
    "cat('Job completed at:', as.character(Sys.time()), '\\n')\n",
    "cat('=== CHROMOSOME ", current_chr, " ANALYSIS COMPLETED SUCCESSFULLY ===\\n')\n"
  )
  
  writeLines(r_script_content, r_script_name)
  
  return(script_name)
}

# Create and submit jobs for each chromosome
print("Creating and submitting SLURM jobs...")
print("=== CHROMOSOME PEAK SUMMARY ===")
job_ids <- c()
total_peaks <- 0

for (current_chr in chromosomes) {
  script_name <- create_slurm_job(current_chr, script_dir, mean_distance_param)
  
  # Count peaks for this chromosome (already printed in create_slurm_job)
  chr_dataset <- subset(seurat_object.cicero, 
                        subset = current_chr == rowData(seurat_object.cicero)$chr)
  n_peaks_chr <- nrow(chr_dataset)
  total_peaks <- total_peaks + n_peaks_chr
  
  # Submit job and capture job ID
  submission_result <- system(paste("sbatch", script_name), intern = TRUE)
  
  if (length(submission_result) > 0 && grepl("Submitted batch job", submission_result)) {
    job_id <- sub(".*Submitted batch job ([0-9]+).*", "\\1", submission_result)
    job_ids <- c(job_ids, job_id)
    print(paste("Submitted job", job_id, "for chromosome", current_chr))
  } else {
    warning(paste("Failed to submit job for chromosome", current_chr))
  }
}

print("=== SUBMISSION SUMMARY ===")
print(paste("Total peaks across all chromosomes:", total_peaks))
print(paste("Total jobs submitted:", length(job_ids)))
print(paste("Job IDs:", paste(job_ids, collapse = ", ")))

# Create a final collection script that depends on all chromosome jobs
if (length(job_ids) > 0) {
  collection_script <- file.path(script_dir, "collect_results.sh")
  dependency_str <- paste("--dependency=afterany:", paste(job_ids, collapse = ":"), sep = "")
  
  collection_content <- paste0(
    "#!/bin/bash\n",
    "#SBATCH --job-name=collect_cicero_results\n",
    "#SBATCH --output=", file.path(script_dir, "collect_results_%j.out"), "\n",
    "#SBATCH --error=", file.path(script_dir, "collect_results_%j.err"), "\n",
    "#SBATCH --time=6:00:00\n",
    "#SBATCH --mem=256G\n",
    "#SBATCH --mail-type=ALL\n",
    "#SBATCH --mail-user=yang-joon.kim@czbiohub.org\n",
    "module load R/4.3\n",
    "Rscript ", file.path(script_dir, "collect_results.R"), "\n"
  )
  
  writeLines(collection_content, collection_script)
  
  # Create R script for result collection
  collection_r_script <- file.path(script_dir, "collect_results.R")
  collection_r_content <- paste0(
    "# Set library paths\n",
    ".libPaths('/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib')\n",
    "withr::with_libpaths(new = '/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib', library(cicero))\n",
    "library(Seurat)\n\n",
    "cat('=== CICERO RESULTS COLLECTION ===\\n')\n",
    "cat('Collection started at:', as.character(Sys.time()), '\\n\\n')\n\n",
    "# Load parameters\n",
    "script_dir <- '", script_dir, "'\n",
    "chromosomes <- c('", paste(chromosomes, collapse = "', '"), "')\n",
    "output_path <- '", output_path, "'\n",
    "data_id <- '", data_id, "'\n",
    "peaktype <- '", peaktype, "'\n",
    "assay <- '", assay, "'\n\n",
    "# Load Seurat object for peaks\n",
    "cat('Loading Seurat object...\\n')\n",
    "seurat_object <- readRDS('", seurat_object_path, "')\n",
    "all_peaks <- row.names(seurat_object@assays[[assay]]@data)\n",
    "cat('Total peaks in Seurat object:', length(all_peaks), '\\n\\n')\n\n",
    "# Collect results\n",
    "cat('=== COLLECTING CICERO MODEL RESULTS ===\\n')\n",
    "model_files <- file.path(script_dir, paste0('cicero_model_', chromosomes, '.rds'))\n",
    "existing_files <- sapply(model_files, file.exists)\n\n",
    "cat('Model file status:\\n')\n",
    "for (i in seq_along(chromosomes)) {\n",
    "  status <- if (existing_files[i]) 'FOUND' else 'MISSING'\n",
    "  cat(sprintf('  Chromosome %s: %s\\n', chromosomes[i], status))\n",
    "  if (existing_files[i]) {\n",
    "    file_size <- file.info(model_files[i])$size\n",
    "    cat(sprintf('    File size: %.2f MB\\n', file_size / 1024^2))\n",
    "  }\n",
    "}\n",
    "cat('\\n')\n\n",
    "if (sum(existing_files) == 0) {\n",
    "  stop('No cicero model files found')\n",
    "}\n\n",
    "if (sum(existing_files) < length(chromosomes)) {\n",
    "  missing_chrs <- chromosomes[!existing_files]\n",
    "  cat('WARNING: Missing model files for chromosomes:', paste(missing_chrs, collapse = ', '), '\\n')\n",
    "  cat('Proceeding with available chromosomes...\\n\\n')\n",
    "}\n\n",
    "# Read successful models\n",
    "cat('Loading cicero models...\\n')\n",
    "cicero_models <- list()\n",
    "connection_stats <- data.frame(\n",
    "  chromosome = character(),\n",
    "  n_connections = integer(),\n",
    "  model_size_mb = numeric(),\n",
    "  stringsAsFactors = FALSE\n",
    ")\n\n",
    "for (i in which(existing_files)) {\n",
    "  chr_name <- chromosomes[i]\n",
    "  cat('  Loading chromosome', chr_name, '...\\n')\n",
    "  \n",
    "  tryCatch({\n",
    "    model <- readRDS(model_files[i])\n",
    "    cicero_models[[chr_name]] <- model\n",
    "    \n",
    "    # Get model statistics\n",
    "    file_size <- file.info(model_files[i])$size / 1024^2\n",
    "    \n",
    "    # Try to get connection count (this depends on the model structure)\n",
    "    n_connections <- 0\n",
    "    if (is.list(model)) {\n",
    "      # Try to assemble connections to count them\n",
    "      tryCatch({\n",
    "        temp_connections <- assemble_connections(model)\n",
    "        n_connections <- nrow(temp_connections)\n",
    "      }, error = function(e) {\n",
    "        cat('    Could not count connections for chromosome', chr_name, '\\n')\n",
    "      })\n",
    "    }\n",
    "    \n",
    "    connection_stats <- rbind(connection_stats, data.frame(\n",
    "      chromosome = chr_name,\n",
    "      n_connections = n_connections,\n",
    "      model_size_mb = file_size,\n",
    "      stringsAsFactors = FALSE\n",
    "    ))\n",
    "    \n",
    "    cat('    Successfully loaded - Connections:', n_connections, ', Size:', round(file_size, 2), 'MB\\n')\n",
    "  }, error = function(e) {\n",
    "    cat('    ERROR loading chromosome', chr_name, ':', e$message, '\\n')\n",
    "  })\n",
    "}\n",
    "cat('\\n')\n\n",
    "# Filter out NULL values\n",
    "cicero_models <- Filter(Negate(is.null), cicero_models)\n\n",
    "if (length(cicero_models) == 0) {\n",
    "  stop('No valid cicero models could be loaded')\n",
    "}\n\n",
    "cat('=== ASSEMBLING CONNECTIONS ===\\n')\n",
    "cat('Successfully loaded models for', length(cicero_models), 'chromosomes\\n')\n",
    "cat('Assembling connections...\\n')\n",
    "start_time <- Sys.time()\n",
    "all_connections <- do.call('rbind', lapply(cicero_models, assemble_connections))\n",
    "end_time <- Sys.time()\n",
    "cat('Connection assembly completed in:', round(difftime(end_time, start_time, units = 'mins'), 2), 'minutes\\n')\n\n",
    "# Log final connection statistics\n",
    "total_connections <- nrow(all_connections)\n",
    "cat('\\n=== FINAL RESULTS SUMMARY ===\\n')\n",
    "cat('Total connections assembled:', total_connections, '\\n')\n",
    "if (total_connections > 0) {\n",
    "  cat('Connection statistics:\\n')\n",
    "  if ('coaccess' %in% colnames(all_connections)) {\n",
    "    cat('  - Mean co-accessibility score:', round(mean(all_connections$coaccess, na.rm = TRUE), 4), '\\n')\n",
    "    cat('  - Median co-accessibility score:', round(median(all_connections$coaccess, na.rm = TRUE), 4), '\\n')\n",
    "    cat('  - Max co-accessibility score:', round(max(all_connections$coaccess, na.rm = TRUE), 4), '\\n')\n",
    "  }\n",
    "  cat('  - Connection data dimensions:', paste(dim(all_connections), collapse = 'x'), '\\n')\n",
    "  cat('  - Column names:', paste(colnames(all_connections), collapse = ', '), '\\n')\n",
    "}\n\n",
    "# Print per-chromosome statistics\n",
    "cat('\\nPer-chromosome connection summary:\\n')\n",
    "print(connection_stats)\n",
    "cat('\\n')\n\n",
    "# Save results\n",
    "cat('=== SAVING RESULTS ===\\n')\n",
    "peaks_file <- paste0(output_path, '01_', data_id, '_', peaktype, '_peaks.csv')\n",
    "connections_file <- paste0(output_path, '02_', data_id, '_cicero_connections_', peaktype, '_peaks.csv')\n\n",
    "cat('Saving peaks file...\\n')\n",
    "write.csv(all_peaks, file = peaks_file, row.names = FALSE)\n",
    "peaks_size <- file.info(peaks_file)$size / 1024^2\n",
    "cat('  - Peaks file:', peaks_file, '\\n')\n",
    "cat('  - Peaks count:', length(all_peaks), '\\n')\n",
    "cat('  - File size:', round(peaks_size, 2), 'MB\\n\\n')\n\n",
    "cat('Saving connections file...\\n')\n",
    "write.csv(all_connections, file = connections_file, row.names = FALSE)\n",
    "connections_size <- file.info(connections_file)$size / 1024^2\n",
    "cat('  - Connections file:', connections_file, '\\n')\n",
    "cat('  - Connections count:', nrow(all_connections), '\\n')\n",
    "cat('  - File size:', round(connections_size, 2), 'MB\\n\\n')\n\n",
    "# Save connection statistics\n",
    "stats_file <- paste0(output_path, '03_', data_id, '_cicero_stats_', peaktype, '_peaks.csv')\n",
    "write.csv(connection_stats, file = stats_file, row.names = FALSE)\n",
    "cat('Connection statistics saved to:', stats_file, '\\n\\n')\n\n",
    "cat('=== ANALYSIS COMPLETED SUCCESSFULLY ===\\n')\n",
    "cat('Collection completed at:', as.character(Sys.time()), '\\n')\n",
    "cat('\\nFinal summary:\\n')\n",
    "cat('  - Total peaks:', length(all_peaks), '\\n')\n",
    "cat('  - Total connections:', nrow(all_connections), '\\n')\n",
    "cat('  - Chromosomes processed:', length(cicero_models), '\\n')\n",
    "cat('  - Output files created: 3\\n')\n"
  )
  
  writeLines(collection_r_content, collection_r_script)
  
  # Submit collection job
  system(paste("sbatch", dependency_str, collection_script))
  print("Submitted result collection job")
}

print("All jobs submitted successfully!")
print(paste("Monitor progress in:", script_dir))
print("Job submission completed.") 

# Print final summary
print("=== FINAL SUMMARY ===")
print(paste("Dataset ID:", data_id))
print(paste("Peak type:", peaktype))
print(paste("Assay used:", assay))
print(paste("Dimensionality reduction:", dim_reduced))
print(paste("Total peaks processed:", total_peaks))
print(paste("Chromosomes processed:", length(chromosomes)))
print(paste("Jobs submitted:", length(job_ids)))
print(paste("Expected output files:"))
print(paste("  1. Peaks:", paste0(output_path, "01_", data_id, "_", peaktype, "_peaks.csv")))
print(paste("  2. Connections:", paste0(output_path, "02_", data_id, "_cicero_connections_", peaktype, "_peaks.csv")))
print(paste("  3. Statistics:", paste0(output_path, "03_", data_id, "_cicero_stats_", peaktype, "_peaks.csv")))
print("")
print("Monitor job progress with:")
print(paste("  squeue -u", Sys.getenv("USER")))
print(paste("  tail -f", file.path(script_dir, "collect_results_*.out")))
print("")
print("Individual chromosome logs available in:")
print(paste("  ", script_dir)) 