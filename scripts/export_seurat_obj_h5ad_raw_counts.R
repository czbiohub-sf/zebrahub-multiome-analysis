# A script to export Seurat object to h5ad format with raw counts
# Inputs:
# Outputs:

# Command-line argument parsing
command_args <- commandArgs(trailingOnly = TRUE)
if (length(command_args) != 3) {
  cat("Usage: Rscript export_seurat_obj_h5ad_raw_counts.R input_dir_prefix output_dir assay\n")
  quit(status = 1)
}

input_dir_prefix <- command_args[1]
output_dir <- command_args[2]
assay <- command_args[3]

# load the libraries
suppressMessages(library(Seurat))
suppressMessages(library(Signac))
#library(Seurat)
#library(Signac)
library(SeuratData)
library(SeuratDisk)
library(Matrix)

# genome info
library(GenomeInfoDb)
library(ggplot2)
library(patchwork)
library(stringr)
library(BSgenome.Drerio.UCSC.danRer11)

print(R.version)
print(packageVersion("Seurat"))

# parallelization in Signac: https://stuartlab.org/signac/articles/future
library(future)
plan()

plan("multicore", workers = 20)
plan()