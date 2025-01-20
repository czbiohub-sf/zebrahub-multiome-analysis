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
options(future.globals.maxSize = 512 * 1024 ^ 3) # for Gb RAM

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

# Load the list of 50K filtered peaks from the file
filtered_peaks <- readLines("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig1_atlas_QC/peaks_hvp_50k.txt")
filtered_peaks %>% head()

# Subset the Seurat object to only include peaks in the filtered list
seurat[["peaks_integrated"]] <- subset(seurat[["peaks_integrated"]], features = filtered_peaks)
print(seurat)
print("peaks filtered for 50K highly variable peaks")

# first compute the GC content for each peak
seurat <- RegionStats(seurat, genome = GRCz11)

# check the result of the RegionStats (there shouldn't be any NAs)
print(head(seurat[["peaks_integrated"]]@meta.features))

# Run LinkPeaks
# Some optional parameters to reduce time/memory:
# - distance = 2e5 or 1e5 (instead of 5e5)
# - n_sample = 100 or 200 (number of background peaks)
# - pvalue_cutoff = 0.01
# - min.cells = 30 (instead of 10)
# - genes.use = c(...) (subset of genes if you only care about certain genes)

# progress bar
# library(progress)
# pb <- progress_bar$new(
#   format = "  Linking Peaks [:bar] :percent | ETA: :eta",
#   total = 1,  # Single LinkPeaks run
#   clear = FALSE,
#   width = 60
# )
# pb$tick()  # Update progress at the start

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
saveRDS(seurat, file = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_50K_hvps_linked_peaks.rds")
print("Done!")

# Extract and save peak-gene links
links <- Links(seurat)
write.csv(as.data.frame(links), file = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/peak_gene_links.csv", row.names = FALSE)