# ## Notebook to link peaks to genes in multiome(RNA/ATAC) objects (Signac/ChromatinAssay)
# reference: https://stuartlab.org/signac/articles/pbmc_multiomic#linking-peaks-to-genes
# - Last updated: 01/14/2025
# - Author: Yang-Joon Kim

# - Step 1. load the multiome objects from all timepoints (6 timepoints, 2 replicates from 15-somites)
# - Step 2. link peaks to genes in the multiome objects

# load the libraries
suppressMessages(library(Seurat))
suppressMessages(library(Signac))
library(SeuratData)
library(SeuratDisk)
library(Matrix)

# genome info
library(GenomeInfoDb)
library(GenomicRanges)
library(Rsamtools)  # for FaFile()
library(ggplot2)
library(patchwork)
library(stringr)
library(BSgenome.Drerio.UCSC.danRer11)

print(R.version)
print(packageVersion("Seurat"))

# parallelization in Signac: https://stuartlab.org/signac/articles/future
library(future)
# plan("multicore", workers = 8)
plan("multisession", workers = 8)
# set the max memory size for the future
options(future.globals.maxSize = 2048 * 1024 ^ 3) # for Gb RAM

# path to your custom genome
custom_fa_path <- "/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/fasta/genome.fa"
GRCz11 <- FaFile(custom_fa_path)

# Load the multiome objects
seurat <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_ga_master.rds")
seurat

# List of assays to keep
assaysToKeep <- c("peaks_integrated", "RNA", "SCT")

# Function to remove other assays from a Seurat object
remove_unwanted_assays <- function(seuratObject, assaysToKeep) {
  allAssays <- names(seuratObject@assays)
  assaysToRemove <- setdiff(allAssays, assaysToKeep)
  
  for (assay in assaysToRemove) {
    seuratObject[[assay]] <- NULL
  }
  
  return(seuratObject)
}

# Apply the function to each of your Seurat objects
seurat <- remove_unwanted_assays(seurat, assaysToKeep)
print(seurat)
print("unnecessary assays removed")

# filter the peaks that go beyond the chromosome end
peak_to_remove <- c("3-62628283-62628504", "10-45419551-45420917")
# 1) Exclude the peak from the assay features
peak_assay <- seurat[["peaks_integrated"]]
peaks_to_keep <- setdiff(rownames(peak_assay), peak_to_remove)

# 2) Subset the ChromatinAssay using these features
seurat[["peaks_integrated"]] <- subset(seurat[["peaks_integrated"]], features = peaks_to_keep)
print(seurat)
print("peaks filtered for peaks beyond the chromosome end")

# # Load the list of 50K filtered peaks from the file
# filtered_peaks <- readLines("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig1_atlas_QC/peaks_hvp_50k.txt")
# filtered_peaks %>% head()

# # Subset the Seurat object to only include peaks in the filtered list
# seurat[["peaks_integrated"]] <- subset(seurat[["peaks_integrated"]], features = filtered_peaks)
# print(seurat)
# print("peaks filtered for 50K highly variable peaks")

# first compute the GC content for each peak
seurat <- RegionStats(seurat, genome = GRCz11)

# check the result of the RegionStats (there shouldn't be any NAs)
print(head(seurat[["peaks_integrated"]]@meta.features))

# compute the peak-gene links
seurat <- LinkPeaks(
  object            = seurat,
  peak.assay        = "peaks_integrated",
  expression.assay  = "SCT",
  distance          = 5e4,    # adjusted for 50kb distance limit
  min.cells         = 50,
  n_sample          = 50,
#   pvalue_cutoff     = 0.05,
#   score_cutoff      = 0.05,
  verbose           = TRUE
)
# pb$tick()  # Complete the progress bar

# Save the final result
# Save final object (with Links stored)
saveRDS(seurat, file = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_linked_peaks.rds")
print("Done!")

# Extract and save peak-gene links
links <- Links(seurat)
write.csv(as.data.frame(links), file = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/peak_gene_links_all_peaks.csv", row.names = FALSE)