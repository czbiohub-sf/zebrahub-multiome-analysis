# An R script to generate QC metric plots for Seurat object with scATAC-seq data

# Load the required libraries
library(Seurat)
library(Signac)
library(patchwork)
library(ggplot2)

# Define the path to save the figure/plots
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/QC_metrics/"

# Define the list of dataset names and their corresponding paths
dataset_names <- c("TDR126","TDR127","TDR128","TDR118reseq","TDR119reseq","TDR125reseq","TDR124reseq")
base_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"
# figpath <- "/path/to/your/figures/"  # Adjust the path as needed

# Create a loop to process each dataset
for (dataset in dataset_names) {
    # Construct the full path to the Seurat object
    # Remove "reseq" from the dataset name
    dataset_processed <- gsub("reseq", "", dataset)
    seurat_obj_path <- paste0(base_path, dataset, "/", dataset_processed, "_processed.RDS")

    # Load the Seurat object
    seurat_obj <- readRDS(seurat_obj_path)
    print(dataset_processed)
    print(seurat_obj)

    # make sure that the major identity is "orig.ident" for bulk peak profile
    Idents(seurat_obj) <- "orig.ident"

    # Run TSSEnrichment
    seurat_obj <- TSSEnrichment(object = seurat_obj, fast = FALSE)

    # Generate the TSS plot
    tss.plot <- TSSPlot(seurat_obj)

    # Print the TSS plot
    print(tss.plot)

    # Save the TSS plot
    ggsave(paste0(figpath, "TSS_enrichment_", dataset_processed, ".pdf"), plot = tss.plot, width = 8, height = 6)
    print("TSS enrichment plot saved")

    # Optionally, save the updated Seurat object if you want to keep the TSSEnrichment results
    # saveRDS(seurat_obj, paste0(base_path, dataset, "/", dataset_processed, "_processed_with_TSS.RDS"))
    # print("Seurat object with TSS enrichment saved")
}