# An R script to generate QC metric plots for Seurat object with scATAC-seq data

# Load the required libraries
library(Seurat)
library(Signac)
library(patchwork)
library(ggplot2)

# Define the path to save the figure/plots
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/QC_metrics/"

# Import the Seurat object
multiome <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_wnn_gene_activity_3d_umaps.rds")
multiome

# Filter out "low_quality_cells"
multiome <- subset(multiome, subset = annotation_ML_coarse != "low_quality_cells")


# # First, a series of histograms to show the distribution of the fragment lengths (for individual dataset)
# # NOTE. our gRanges object has the format of "1-start-end", not "chr1-start-end", so, we'd have to change the region argument
# plot1 <- FragmentHistogram(multiome, group.by = "dataset", region = "1-1-20000000")
# ggsave(paste0(filepath, "hist_fragment_lengths_datasets.pdf"), plot = plot1, width = 8, height = 12)
# print("histogram of fragment lengths for individual datasets is saved.")

# Second, a distribution of fragments near TSS (for individual dataset)
# compute TSS enrichment score per cell (need to re-run this as we merged the seurat objects from individual timepoints)
multiome <- TSSEnrichment(object = multiome, fast = FALSE)
plot2 <- TSSPlot(multiome, group.by = "dataset")
ggsave(paste0(filepath, "TSS_enrichment_distance_from_TSS.pdf"), plot = plot2, width = 8, height = 12)

# save the Seurat object so that we can generate the plots later on
saveRDS(multiome, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_wnn_gene_activity_3d_umaps_with_TSS_enrichment.rds")

# different group.by parameter (timepoint)
plot3 <- TSSPlot(multiome, group.by = "timepoint")
ggsave(paste0(filepath, "TSS_enrichment_distance_from_TSS_timepoint.pdf"), plot = plot3, width = 8, height = 12)
print("TSS enrichment score for individual datasets is saved.")

print("Done!")