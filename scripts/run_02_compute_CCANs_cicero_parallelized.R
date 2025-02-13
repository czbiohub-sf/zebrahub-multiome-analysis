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

# Define the directory to save the shell scripts (for cicero parallelization)
#shell_script_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/cicero_slurm_outputs/"
script_dir <- shell_script_dir
dir.create(script_dir, showWarnings = FALSE, recursive = TRUE)
setwd(script_dir) # set as the temp working directory

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
chr_lengths <- c(
    "1" = 59578282,
    "2" = 59640629,
    "3" = 62628489,
    "4" = 78093715,
    "5" = 72500376,
    "6" = 60270060,
    "7" = 74282399,
    "8" = 54304671,
    "9" = 56459846,
    "10" = 45420867,
    "11" = 45484837,
    "12" = 49182954,
    "13" = 52186027,
    "14" = 52660232,
    "15" = 48040578,
    "16" = 55381981,
    "17" = 53969382,
    "18" = 51023478,
    "19" = 48449771,
    "20" = 55201332,
    "21" = 45934066,
    "22" = 39133080,
    "23" = 46144548,
    "24" = 42173229,
    "25" = 37502051,
    "MT" = 16596
)

# Convert values to integer while keeping names as strings
chr_lengths <- as.integer(chr_lengths)  # This converts only the values to integer, keeps names as strings

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
genome.df <- data.frame(
    chr = names(chr_lengths),
    length = as.numeric(chr_lengths)  # convert to numeric to avoid integer overflow
)

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

# Step 3: Collect the results from Slurm jobs (each chromosome)
# Assuming your RDS files are named in the format "cicero_model_<chromosome>.rds"
#chromosomes <- unique(seurat_object.cicero$chromosome) # List of chromosomes
model_files <- paste0(script_dir, "cicero_model_", chromosomes, ".rds")

# check if all 25 chromosomes are present (from individual slurm jobs)
existing_files <- sapply(model_files, file.exists)

if (sum(existing_files) != 25) {
    stop("Not all cicero_model files are present. Expected 25, found: ", sum(existing_files))
} else {
    print("All 25 cicero_model files are present. Proceeding to Step 3.")
}

# Read all the Cicero models
cicero_models <- lapply(model_files, function(file) {
  if (file.exists(file)) {
    readRDS(file)
  } else {
    warning(paste("File not found:", file))
    NULL
  }
})

# Filter out NULL values in case some files were not found
cicero_models <- Filter(Negate(is.null), cicero_models)

# Assemble connections from the per-chromosome models
all_connections <- do.call("rbind", lapply(cicero_models, assemble_connections))
# all_connections now contains the combined dataset of connections

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