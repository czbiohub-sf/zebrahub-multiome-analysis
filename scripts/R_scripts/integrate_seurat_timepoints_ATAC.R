# ## Notebook to integrate multiome(ATAC) objects (Seurat/Signac)
# reference: https://stuartlab.org/signac/0.2/articles/merging (note that this was from Signac 0.2.5, older version)
# [To-Do] we might want to replace the script with the recent version of the vignette (https://stuartlab.org/signac/articles/merging)

# - Last updated: 02/15/2024
# - Author: Yang-Joon Kim

# - Step 1. load the multiome objects from all timepoints (6 timepoints, 2 replicates from 15-somites)
# - Step 2. re-process the ATAC objects (merging peaks, re-computing the count matrices, then re-computing the PCA/LSI/SVD, seurat integration).
# - Step 3. integrate the ATAC objects using Seurat's integration method (CCA, rLSI, etc.)

# load the libraries
suppressMessages(library(Seurat))
suppressMessages(library(Signac))
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

plan("multicore", workers = 8)
plan()

# set the max memory size for the future
options(future.globals.maxSize = 512 * 1024 ^ 3) # for 512 Gb RAM

# # Step 1. Load the datasets
# # Load all seurat objects (replace the filepaths if you use this for your own datasets)
# TDR118 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/TDR118_processed.RDS")
# TDR119 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR119reseq/TDR119_processed.RDS")
# TDR124 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR124reseq/TDR124_processed.RDS")
# TDR125 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR125reseq/TDR125_processed.RDS")
# TDR126 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR126/TDR126_processed.RDS")
# TDR127 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR127/TDR127_processed.RDS")
# TDR128 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR128/TDR128_processed.RDS")

# # print("Loaded all the datasets")

# # List of assays to keep
# assaysToKeep <- c("peaks_merged", "RNA", "SCT")

# # Function to remove other assays from a Seurat object
# remove_unwanted_assays <- function(seuratObject, assaysToKeep) {
#   allAssays <- names(seuratObject@assays)
#   assaysToRemove <- setdiff(allAssays, assaysToKeep)
  
#   for (assay in assaysToRemove) {
#     seuratObject[[assay]] <- NULL
#   }
  
#   return(seuratObject)
# }

# # Apply the function to each of your Seurat objects
# TDR118 <- remove_unwanted_assays(TDR118, assaysToKeep)
# TDR119 <- remove_unwanted_assays(TDR119, assaysToKeep)
# TDR124 <- remove_unwanted_assays(TDR124, assaysToKeep)
# TDR125 <- remove_unwanted_assays(TDR125, assaysToKeep)
# TDR126 <- remove_unwanted_assays(TDR126, assaysToKeep)
# TDR127 <- remove_unwanted_assays(TDR127, assaysToKeep)
# TDR128 <- remove_unwanted_assays(TDR128, assaysToKeep)

# # Step 2. Re-process the ATAC objects
# # Make sure that we have the "peaks_merged" assay as the active assay in each Seurat object 
# (as "UnifyPeaks" function will merge the peaks/features from the active assay.)

# # Set "peaks_merged" as the default assay for each Seurat object
# DefaultAssay(TDR118) <- 'peaks_merged'
# DefaultAssay(TDR119) <- 'peaks_merged'
# DefaultAssay(TDR124) <- 'peaks_merged'
# DefaultAssay(TDR125) <- 'peaks_merged'
# DefaultAssay(TDR126) <- 'peaks_merged'
# DefaultAssay(TDR127) <- 'peaks_merged'
# DefaultAssay(TDR128) <- 'peaks_merged'

# # Step 2-1. creating a common peak set
# combined.peaks <- UnifyPeaks(object.list = list(TDR118,TDR119,TDR124,
#                                                 TDR125,TDR126,TDR127,
#                                                 TDR128), mode = "reduce")
# combined.peaks

# print("peaks were combined/merged")

# # Step 2-2. quantify peaks in each dataset (re-process the count matrices)
# # define a function to quantify the peaks
# recompute_count_matrices <- function(seurat_obj, combined.peaks, data_id){
#     new.counts <- FeatureMatrix(
#       fragments = Fragments(seurat_obj),
#       features = combined.peaks,
#       sep = c(":", "-"),
#       cells = colnames(seurat_obj)
#     )
#     seurat_obj[["peaks_integreated"]] <- CreateAssayObject(counts = new.counts)
#     # new_seurat_obj <- CreateSeuratObject(counts = new.counts, project = data_id,
#     #                                      assay = "peaks_integrated")
#     # add the data_id to the object
#     seurat_obj$dataset <- data_id
#     return(seurat_obj)
# }

# # recompute the count matrices for each dataset, save as "peaks_integrated" ChromatinAssay
# TDR118 <- recompute_count_matrices(TDR118, combined.peaks = combined.peaks, data_id="TDR118")
# TDR119 <- recompute_count_matrices(TDR119, combined.peaks = combined.peaks, data_id="TDR119")
# TDR124 <- recompute_count_matrices(TDR124, combined.peaks = combined.peaks, data_id="TDR124")
# TDR125 <- recompute_count_matrices(TDR125, combined.peaks = combined.peaks, data_id="TDR125")
# TDR126 <- recompute_count_matrices(TDR126, combined.peaks = combined.peaks, data_id="TDR126")
# TDR127 <- recompute_count_matrices(TDR127, combined.peaks = combined.peaks, data_id="TDR127")
# TDR128 <- recompute_count_matrices(TDR128, combined.peaks = combined.peaks, data_id="TDR128")

# # merge all datasets, adding a cell ID to make sure cell names are unique
# combined <- merge(x = TDR118, 
#                     y = list(TDR119, TDR124, TDR125, TDR126, TDR127, TDR128), 
#                     add.cell.ids = c("TDR118", "TDR119", "TDR124", "TDR125", "TDR126", "TDR127", "TDR128"))

# saveRDS(combined, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC.RDS")  # save the combined object
# print("Saved the combined object")

# # make sure to change to the assay containing common peaks
# DefaultAssay(combined) <- "peaks_integrated"
# combined <- RunTFIDF(combined)
# combined <- FindTopFeatures(combined, min.cutoff = 20)
# combined <- RunSVD(
#   combined,
#   reduction.key = 'LSI_',
#   reduction.name = 'lsi',
#   irlba.work = 400
# )
# combined <- RunUMAP(combined, dims = 2:30, reduction = 'lsi')

# # DimPlot(combined, group.by = 'dataset', pt.size = 0.1)

# saveRDS(combined, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC.RDS")  # save the combined object
# print("Saved the combined object with LSI and UMAP")

# Step 3. Integrate the ATAC objects (batch-correction)
# source: https://stuartlab.org/signac/articles/integrate_atac
# Loading the combined object (if we are resuming from here - it already has LSI and UMAP computed from the simple merge)
combined <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC.RDS")

# (NOT NEEDED IF WE START FROM THE TOP)
# Subset by dataset ID
TDR118 <- subset(combined, subset = dataset == "TDR118")
TDR119 <- subset(combined, subset = dataset == "TDR119")
TDR124 <- subset(combined, subset = dataset == "TDR124")
TDR125 <- subset(combined, subset = dataset == "TDR125")
TDR126 <- subset(combined, subset = dataset == "TDR126")
TDR127 <- subset(combined, subset = dataset == "TDR127")
TDR128 <- subset(combined, subset = dataset == "TDR128")

# find integration anchors
integration.anchors <- FindIntegrationAnchors(
  object.list = list(TDR118,TDR119,TDR124,TDR125,TDR126,TDR127,TDR128),
  anchor.features = rownames(TDR118), # all features/peaks (combined.peaks)
  reduction = "rlsi", # reciprocal LSI
  dims = 2:30
)
print("anchors computed")

# integrate LSI embeddings
integrated <- IntegrateEmbeddings(
  anchorset = integration.anchors,
  reductions = combined[["lsi"]],
  new.reduction.name = "integrated_lsi",
  dims.to.integrate = 1:30
)
print("integration completed")

# create a new UMAP using the integrated embeddings
integrated <- RunUMAP(integrated, reduction = "integrated_lsi", dims = 2:30)
# p2 <- DimPlot(integrated, group.by = "dataset")
saveRDS(integrated, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_rLSI.RDS")  # save the combined object
print("Saved the integrated object")