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

# inputs:
# 1) seurat_object: a seurat object
# 2) assay: "ATAC", "peaks", etc. - a ChromatinAssay object generated with Signac using the best peak profiles
# 3) dim_reduced: "UMAP.ATAC", "UMAP", "PCA", etc. - a dimensionality reduction. 
# NOTE that this should be "capitalized" as as.cell_data_set capitalizes all dim.red fields 
# 4) output_path: path to save the output (peaks, and CCANs)
# 5) data_id: ID for the dataset, i.e. TDR118
# 6) peaktype: type of the peak profile. i.e. CRG_arc: Cellranger-arc peaks
# (i.e. peaks_celltype, peaks_bulk, peaks_joint)

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
# dim.reduced <- "UMAP.ATAC"
# output_path = "",
# data_id="TDR118",
# peaktype = "peaks_merged"
# shell_script_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/cicero_shell_scripts/"

# Define the directory to save the shell scripts (for cicero parallelization)
#shell_script_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/cicero_shell_scripts/"
script_dir <- shell_script_dir
dir.create(script_dir, showWarnings = FALSE, recursive = TRUE)

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

# conver to CellDataSet (CDS) format
seurat_object.cds <- as.cell_data_set(x=seurat_object) # a function from SeuratWrappers\
print("cds object created") 

# make the cicero object
# default: we will use the ATAC.UMAP here for the sampling of the neighborhoods - as we'll treat this dataset as if we only had scATAC-seq.
# This is something we can ask Kenji Kamimoto/Samantha Morris later for their advice. (or compare the built GRNs from joint.UMAP vs ATAC.UMAP)
seurat_object.cicero <- make_cicero_cds(seurat_object.cds, reduced_coordinates = reducedDims(seurat_object.cds)$UMAP.ATAC)
print("cicero object created")

# define the genomic length dataframe (chromosome number ; length)
df_seqinfo <- as.data.frame(seurat_object@assays$ATAC@seqinfo)
# zebrafish has 25 chromosomes
seurat_object@assays$ATAC@annotation@seqinfo@seqlengths <- df_seqinfo$seqlengths[1:26] 

# Perform CCAN computation
# get the chromosome sizes from the Seurat object
genome <- seqlengths(seurat_object@assays$ATAC@annotation)

# convert chromosome sizes to a dataframe
genome.df <- data.frame("chr" = names(genome), "length" = genome)
saveRDS(genome.df, file = paste0(script_dir, "genome_df.rds"))
print(genome.df)

# run cicero (parallelized)
# conns <- run_cicero(seurat_object.cicero, genomic_coords = genome.df, sample_num = 100)

# Step 1: Estimate the distance parameter on the whole dataset
distance_param <- estimate_distance_parameter(seurat_object.cicero, 
                                              genomic_coords = genome.df)

# averate the distance_param to get the final distance_param
mean_distance_param <- mean(distance_param)

# Step 2: Generate Cicero models for each chromosome using Slurm
#chromosomes <- unique(seurat_object.cicero$chromosome)
chromosomes <- unique(genome.df$chr)
chromosomes <- chromosomes[chromosomes != "MT"] # Remove mitochondrial chromosome
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
    "#SBATCH --time=01:00:00\n",
    "#SBATCH --mem=50G\n",
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
all_peaks <- row.names(seurat_object@assays$peaks_merged@data)
#output_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data"
write.csv(x = all_peaks, file = paste0(output_path, "01_", data_id, "_",peaktype, "_peaks.csv"))
write.csv(x = all_connections, file = paste0(output_path, "02_", data_id, "_cicero_connections_",peaktype, "_peaks.csv"))