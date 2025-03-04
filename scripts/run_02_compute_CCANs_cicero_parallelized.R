# An R script to run Cicero for a Seurat object with a ChromatinAssay object
# Note1: added the option to choose the peak profiles (i.e. peaks_celltype, peaks_bulk, peaks_merged)
# Note2: added the option to choose the dimensionality reduction (i.e. UMAP, UMAP.ATAC, PCA, etc.)
# Update (1/22/2024): I've added the paralleization option for the cicero computation.

# Load the Cicero library from the local installation (trapnell lab branch for Signac implementation)
# library(remotes)
# library(devtools)
# install cicero
# withr::with_libpaths(new="/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", 
#                      install_github("cole-trapnell-lab/cicero-release", ref = "monocle3"))
# cicero
.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(monocle3))
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(cicero))

# load other libraries
#library(cicero)
library(Signac)
library(Seurat)
library(SeuratWrappers)
library(GenomeInfoDb)

# inputs:
# 1) seurat_object: a seurat object
# 2) assay: "ATAC", "peaks", etc. - a ChromatinAssay object generated with Signac using the best peak profiles
# 3) dim_reduced: "UMAP.ATAC", "UMAP", "PCA", etc. - a dimensionality reduction. 
# NOTE that this should be "capitalized" as as.cell_data_set capitalizes all dim.red fields 
# 4) output_path: path to save the output (peaks, and CCANs)
# 5) data_id: ID for the dataset, i.e. TDR118
# 6) peaktype: type of the peak profile. i.e. CRG_arc: Cellranger-arc peaks
# (i.e. peaks_celltype, peaks_bulk, peaks_joint)
# 7) shell_script_dir: path to save the shell scripts (for cicero parallelization)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 7) {
  stop("Usage: Rscript run_02_compute_CCANS_cicero_parallelized.R seurat_object_path assay dim_reduced output_path data_id peaktype shell_script_dir")
}

seurat_object_path <- args[1]
assay <- args[2]
dim_reduced <- args[3]
output_path <- args[4] 
data_id <- args[5] 
peaktype <- args[6]
shell_script_dir <- args[7]

# Example Input arguments:
# seurat_object <- readRDS(seurat_object_path)
# assay <- "ATAC"
# dim.reduced <- "UMAP.ATAC" ("integrated_lsi")
# output_path = "",
# data_id="TDR118",
# peaktype = "peaks_merged"
# shell_script_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_slurm_outputs/"

# Update the script_dir to include the cicero_integrated_peaks subdirectory
script_dir <- file.path(shell_script_dir, "cicero_integrated_peaks")
dir.create(script_dir, showWarnings = FALSE, recursive = TRUE)
setwd(script_dir)

# Import the Seurat object
seurat_object <- readRDS(seurat_object_path)

# print out the assays in the seurat object
#print(seurat_object@assays) 

# define the default assay
# We will pick which peak profiles we will use.
# ideally, we have defined the peak profiles 
# (and corresponding count matrices of cells-by-peaks) from the previous steps.
DefaultAssay(seurat_object) <- assay
print(paste0("default assay is ", assay))

# convert to CellDataSet (CDS) format
seurat_object.cds <- as.cell_data_set(x=seurat_object) # a function from SeuratWrappers
print("cds object created") 

# print out the available reduced dimensions
print("Available reduced dimensions:")
print(names(reducedDims(seurat_object.cds)))

# capitalize the dim_reduced, as the single.cell.experiment data format capitalizes all the fields
dim_reduced <- toupper(dim_reduced)
# Get the reduced coordinates
reduced_coords <- reducedDims(seurat_object.cds)[[dim_reduced]]
print("Class of reduced coordinates:")
print(class(reduced_coords))
print("Dimensions of reduced coordinates:")
print(dim(reduced_coords))


# create the cicero object
# default: we will use the ATAC.UMAP here for the sampling of the neighborhoods - as we'll treat this dataset as if we only had scATAC-seq.
# This is something we can ask Kenji Kamimoto/Samantha Morris later for their advice. (or compare the built GRNs from joint.UMAP vs ATAC.UMAP)
seurat_object.cicero <- make_cicero_cds(seurat_object.cds, reduced_coordinates = reduced_coords)
# seurat_object.cicero <- make_cicero_cds(seurat_object.cds, reduced_coordinates = reducedDims(seurat_object.cds)$UMAP.ATAC)
print("cicero object created")

# Define the genomic length dataframe (chromosome number ; length)
# GRCz11 chromosome lengths
chr_lengths <- setNames(
    c(59578282, 59640629, 62628489, 78093715, 72500376, 60270060, 74282399,
      54304671, 56459846, 45420867, 45484837, 49182954, 52186027, 52660232,
      48040578, 55381981, 53969382, 51023478, 48449771, 55201332, 45934066,
      39133080, 46144548, 42173229, 37502051, 16596),
    c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
      "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
      "21", "22", "23", "24", "25", "MT")
)
print(names(chr_lengths))  # Should return "1" "2" "3" ... "25" "MT"
# chr_lengths <- c(
#     "1" = 59578282,
#     "2" = 59640629,
#     "3" = 62628489,
#     "4" = 78093715,
#     "5" = 72500376,
#     "6" = 60270060,
#     "7" = 74282399,
#     "8" = 54304671,
#     "9" = 56459846,
#     "10" = 45420867,
#     "11" = 45484837,
#     "12" = 49182954,
#     "13" = 52186027,
#     "14" = 52660232,
#     "15" = 48040578,
#     "16" = 55381981,
#     "17" = 53969382,
#     "18" = 51023478,
#     "19" = 48449771,
#     "20" = 55201332,
#     "21" = 45934066,
#     "22" = 39133080,
#     "23" = 46144548,
#     "24" = 42173229,
#     "25" = 37502051,
#     "MT" = 16596
# )

# Convert values to integer while keeping names as strings
# chr_lengths <- as.integer(chr_lengths)  # This converts only the values to integer, keeps names as strings
chr_lengths <- setNames(as.integer(chr_lengths), names(chr_lengths))

# Assign the chromosome lengths to the seqinfo object
seurat_object@assays[[assay]]@annotation@seqinfo@seqlengths <- chr_lengths

# Verify the assignment worked
print(seurat_object@assays[[assay]]@annotation@seqinfo)

# After assigning seqlengths
if (any(is.na(seurat_object@assays[[assay]]@annotation@seqinfo@seqlengths))) {
    stop("Seqlengths assignment failed - some lengths are NA")
}
print("Seqlengths successfully assigned:")
print(seurat_object@assays[[assay]]@annotation@seqinfo)

# After setting the seqlengths, create genome.df directly
# genome.df <- data.frame(
#     chr = names(chr_lengths),
#     length = as.numeric(chr_lengths)  # convert to numeric to avoid integer overflow
# )
print(chr_lengths)
print(names(chr_lengths))
genome.df <- data.frame(
    chr = names(chr_lengths),
    length = as.numeric(chr_lengths),  # Ensure numeric values
    stringsAsFactors = FALSE  # Avoid factor conversion
)
# check the seqinfo
print(seurat_object@assays[[assay]]@annotation@seqinfo)

# Save for later use by parallel jobs
saveRDS(genome.df, file = paste0(script_dir, "genome_df.rds"))
print(genome.df)

# Perform CCAN computation
# get the chromosome sizes from the Seurat object
# genome <- seqlengths(seurat_object@assays[[assay]]@annotation)
# genome <- seqlengths(seurat_object@assays$ATAC@annotation)

# convert chromosome sizes to a dataframe
# genome.df <- data.frame("chr" = names(genome), "length" = genome)
# saveRDS(genome.df, file = paste0(script_dir, "genome_df.rds"))
# print(genome.df)

# load the genome.df
# genome.df <- readRDS(file = paste0("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_slurm_outputs/", "genome_df.rds"))
# print(genome.df)

# run cicero (parallelized)
# conns <- run_cicero(seurat_object.cicero, genomic_coords = genome.df, sample_num = 100)

# Step 1: Estimate the distance parameter on the whole dataset
distance_param <- estimate_distance_parameter(seurat_object.cicero, 
                                              genomic_coords = genome.df)

# averate the distance_param to get the final distance_param
mean_distance_param <- mean(distance_param)
print(paste0("mean distance parameter = ", mean_distance_param))

# Step 2: Generate Cicero models for each chromosome using Slurm
#chromosomes <- unique(seurat_object.cicero$chromosome)
chromosomes <- unique(genome.df$chr)
chromosomes <- chromosomes[chromosomes != "MT"]  # This should now work correctly
print(chromosomes)


# Loop through each chromosome and create a Slurm job
for (current_chr in chromosomes) {
  # Subset cell_data_set by chromosome
  chr_dataset <- subset(seurat_object.cicero, subset = current_chr == rowData(seurat_object.cicero)$chr)  
  # chr_dataset <- subset(seurat_object.cicero, chromosome == chr)
  saveRDS(chr_dataset, file=paste0(script_dir, "/seurat_chr_", current_chr, ".rds"))

  # Create a shell script for the Slurm job
  script_name <- paste0(script_dir, "cicero_job_", current_chr, ".sh")
  script_content <- paste(
    "#!/bin/bash\n",
    "#SBATCH --output=", script_dir, "cicero_chr_", current_chr, "_%j.out\n",
    "#SBATCH --error=", script_dir, "cicero_chr_", current_chr, "_%j.err\n",
    "#SBATCH --time=24:00:00\n",
    "#SBATCH --mem=5G\n",
    "#SBATCH --mail-type=FAIL\n", # reporting only in case of failure
    "#SBATCH --mail-user=yang-joon.kim@czbiohub.org\n",
    "module load R/4.3\n",
    "Rscript ", script_dir, "run_cicero_chr_", current_chr, ".R\n",
    sep=""
  )
  writeLines(script_content, script_name)

    # Create an R script for this chromosome
  r_script_name <- paste0(script_dir, "run_cicero_chr_", current_chr, ".R")
  r_script_content <- paste(
    "library(monocle3)\n",
    "library(cicero)\n",
    "library(Seurat)\n",
    "genome.df <- readRDS('", script_dir, "/genome_df.rds')\n",
    "chr_dataset <- readRDS('", script_dir, "/seurat_chr_", current_chr, ".rds')\n",
    "cicero_model <- generate_cicero_models(chr_dataset, distance_parameter = ", mean_distance_param, ", genomic_coords = genome.df)\n",
    "saveRDS(cicero_model, file='", script_dir, "cicero_model_", current_chr, ".rds')\n",
    sep=""
  )
  writeLines(r_script_content, r_script_name)

  # Submit the job to Slurm
  system(paste("sbatch", script_name))
}

# After submitting all Slurm jobs, add this waiting mechanism
# Step 3: Wait for all jobs to complete and collect results
wait_for_jobs <- function(script_dir, chromosomes, max_wait_time = 7200) { # 2 hours max wait
  start_time <- Sys.time()
  all_files_present <- FALSE
  
  while (!all_files_present && (difftime(Sys.time(), start_time, units="secs") < max_wait_time)) {
    # Check if all files exist
    model_files <- file.path(script_dir, paste0("cicero_model_", chromosomes, ".rds"))
    existing_files <- sapply(model_files, file.exists)
    
    # Add debugging information
    print("Looking for files in:")
    print(script_dir)
    print("Files found:")
    print(data.frame(
      file = basename(model_files),  # Only show filename for cleaner output
      exists = existing_files
    ))
    
    if (sum(existing_files) == length(chromosomes)) {
      all_files_present <- TRUE
      print("All cicero model files are present!")
      break
    }
    
    print(paste("Waiting for jobs to complete... Found", sum(existing_files), "of", length(chromosomes), "files"))
    Sys.sleep(60)  # Wait for 60 seconds before checking again
  }
  
  if (!all_files_present) {
    stop("Timeout waiting for cicero model files to be generated")
  }
  
  return(model_files[existing_files])  # Return paths of successfully generated files
}

# Call the wait function before proceeding to collect results
chromosomes <- chromosomes[chromosomes != "MT"]  # Exclude MT as before
model_files <- wait_for_jobs(script_dir, chromosomes)

# Now proceed with collecting results - use the files returned by wait_for_jobs
if (length(model_files) > 0) {
  # Read all the Cicero models
  cicero_models <- lapply(model_files, function(file) {
    tryCatch({
      readRDS(file)
    }, error = function(e) {
      warning(paste("Error reading file:", file, "-", e))
      NULL
    })
  })
  
  # Filter out NULL values in case some files couldn't be read
  cicero_models <- Filter(Negate(is.null), cicero_models)
  
  # Assemble connections from the per-chromosome models
  all_connections <- do.call("rbind", lapply(cicero_models, assemble_connections))
  print(paste("Successfully assembled connections from", length(cicero_models), "chromosome models"))
} else {
  stop("No cicero model files were found")
}

# Note: This script will submit Slurm jobs and exit. The actual Cicero processing will happen in the Slurm jobs.
# You'll need to collect and process the results once all jobs are com
print("CCANs computed")

# Return the CCAN results
# return(all_connections)

# saves the Cicero results 
# (1.all peaks as well as 2. pairwise cicero result)
# all_peaks <- row.names(seurat_object@assays$peaks_merged@data)
all_peaks <- row.names(seurat_object@assays[[assay]]@data)
#output_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data"
write.csv(x = all_peaks, file = paste0(output_path, "01_", data_id, "_",peaktype, "_peaks.csv"))
write.csv(x = all_connections, file = paste0(output_path, "02_", data_id, "_cicero_connections_",peaktype, "_peaks.csv"))