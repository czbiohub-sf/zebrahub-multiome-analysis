#' Export Seurat Assays to h5ad Format
#'
#' This function takes a Seurat object, specifies the assays to export, and saves them in h5ad format.
#'
#' @param input_dir_prefix Character string specifying the path to the input Seurat object in RDS format.
#' @param output_dir Character string specifying the directory where h5ad files will be saved.
#' @param assays_save Character vector specifying the assays to export.
#'
#' @return NULL
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' export_seurat_assays(input_dir_prefix = "/path/to/input_seurat.rds",
#'                      output_dir = "/path/to/output_directory",
#'                      assays_save = c("RNA", "ATAC"))
#' }
#'
#' @export

# Command-line argument parsing
command_args <- commandArgs(trailingOnly = TRUE)
if (length(command_args) != 3) {
  cat("Usage: Rscript export_seurat_assays.R input_dir_prefix output_dir assays_save\n")
  quit(status = 1)
}

input_dir_prefix <- command_args[1]
output_dir <- command_args[2]
assays_save <- strsplit(command_args[3], ",")[[1]]

# define the function
export_seurat_assays <- function(input_dir_prefix, output_dir, assays_save) {
  
  # Load required libraries
  library(Seurat)
  library(Signac)
  library(SeuratDisk)
  
  # Read the input Seurat object
  seurat <- readRDS(input_dir_prefix)
  
  # Loop through the specified assays
  for (assay in assays_save) {
    # Set the default assay
    DefaultAssay(seurat) <- assay
    print(seurat)
    
    # Save the object (assay)
    filename <- file.path(output_dir, paste0("data_id_processed_", assay, ".h5Seurat"))
    SaveH5Seurat(seurat, filename = filename, overwrite = TRUE)
    
    # Convert the h5Seurat to h5ad
    Convert(filename, dest = "h5ad")
  }
}

# Call the function with command-line arguments
# export_seurat_assays(input_dir_prefix, output_dir, assays_save)