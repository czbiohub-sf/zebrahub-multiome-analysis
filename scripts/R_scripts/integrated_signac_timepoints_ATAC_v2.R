# ## Notebook to integrate multiome(ATAC) objects (Signac/ChromatinAssay)
# reference: https://stuartlab.org/signac/articles/merging#:~:text=The%20merge%20function%20defined%20in,object%20being%20merged%20become%20equivalent.
# - Last updated: 03/12/2024
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
library(GenomicRanges)
library(ggplot2)
library(patchwork)
library(stringr)
library(BSgenome.Drerio.UCSC.danRer11)

print(R.version)
print(packageVersion("Seurat"))

# parallelization in Signac: https://stuartlab.org/signac/articles/future
library(future)
plan("multicore", workers = 8)
# set the max memory size for the future
options(future.globals.maxSize = 800 * 1024 ^ 3) # for 512 Gb RAM

# Step 1. Load the datasets
# # Load all seurat objects (replace the filepaths if you use this for your own datasets)
# TDR118 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/TDR118_processed.RDS")
# TDR119 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR119reseq/TDR119_processed.RDS")
# TDR124 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR124reseq/TDR124_processed.RDS")
# TDR125 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR125reseq/TDR125_processed.RDS")
# TDR126 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR126/TDR126_processed.RDS")
# TDR127 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR127/TDR127_processed.RDS")
# TDR128 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR128/TDR128_processed.RDS")

# print("Loaded all the datasets")

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

# # Set "peaks_merged" as the default assay for each Seurat object
# DefaultAssay(TDR118) <- 'peaks_merged'
# DefaultAssay(TDR119) <- 'peaks_merged'
# DefaultAssay(TDR124) <- 'peaks_merged'
# DefaultAssay(TDR125) <- 'peaks_merged'
# DefaultAssay(TDR126) <- 'peaks_merged'
# DefaultAssay(TDR127) <- 'peaks_merged'
# DefaultAssay(TDR128) <- 'peaks_merged'

# # Step 2-1. creating a common peak set
# # # extract the genomic ranges of the peaks from each dataset
# # gr.TDR118 <- granges(TDR118)
# # gr.TDR119 <- granges(TDR119)
# # gr.TDR124 <- granges(TDR124)
# # gr.TDR125 <- granges(TDR125)
# # gr.TDR126 <- granges(TDR126)
# # gr.TDR127 <- granges(TDR127)
# # gr.TDR128 <- granges(TDR128)

# # # Create a unified set of peaks to quantify in each dataset
# # combined.peaks <- reduce(x = c(gr.TDR118, gr.TDR119, gr.TDR124, gr.TDR125, gr.TDR126, gr.TDR127, gr.TDR128))

# combined.peaks <- UnifyPeaks(object.list = list(TDR118,TDR119,TDR124,
#                                                 TDR125,TDR126,TDR127,
#                                                 TDR128), mode = "reduce")
# combined.peaks

# # Filter out bad peaks based on length
# # peakwidths <- width(combined.peaks)
# # combined.peaks <- combined.peaks[peakwidths  < 10000 & peakwidths > 20]
# # combined.peaks

# print("peaks were combined/merged")

# # Step 2-2. Compute the fragments object (this is to reformat the fragment files)
# # create fragment objects
# frags.TDR118 <- CreateFragmentObject(
#   path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118reseq/outs/atac_fragments.tsv.gz",
#   cells = colnames(TDR118)
# )
# frags.TDR119 <- CreateFragmentObject(
#   path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR119reseq/outs/atac_fragments.tsv.gz",
#   cells = colnames(TDR119)
# )
# frags.TDR124 <- CreateFragmentObject(
#   path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR124reseq/outs/atac_fragments.tsv.gz",
#   cells = colnames(TDR124)
# )
# frags.TDR125 <- CreateFragmentObject(
#   path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR125reseq/outs/atac_fragments.tsv.gz",
#   cells = colnames(TDR125)
# )
# frags.TDR126 <- CreateFragmentObject(
#   path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR126/outs/atac_fragments.tsv.gz",
#   cells = colnames(TDR126)
# )
# frags.TDR127 <- CreateFragmentObject(
#   path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR127/outs/atac_fragments.tsv.gz",
#   cells = colnames(TDR127)
# )
# frags.TDR128 <- CreateFragmentObject(
#   path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR128/outs/atac_fragments.tsv.gz",
#   cells = colnames(TDR128)
# )
# print("fragment objects created")

# # Step 2-3. quantify peaks in each dataset (re-process the count matrices)
# # define a function to quantify the peaks

# # recompute_count_matrices <- function(seurat_obj, combined.peaks, data_id){
# #     new.counts <- FeatureMatrix(
# #       fragments = Fragments(seurat_obj),
# #       features = combined.peaks,
# #       sep = c(":", "-"),
# #       cells = colnames(seurat_obj)
# #     )
# #     seurat_obj[["peaks_integreated"]] <- CreateChromatinAssay(counts = new.counts)

# #     # add the data_id to the object
# #     seurat_obj$dataset <- data_id
# #     return(seurat_obj)
# # }
# recompute_count_matrices <- function(seurat_obj, combined.peaks, fragments, data_id){
#     # recompute the count matrices using the unified peak set (combined.peaks)
#     new.counts <- FeatureMatrix(
#       fragments = fragments, #Fragments(seurat_obj),
#       features = combined.peaks,
#       #sep = c(":", "-"),
#       cells = colnames(seurat_obj)
#     )
#     #signac_obj <- CreateChromatinAssay(counts = new.counts)
#     signac_obj <- CreateChromatinAssay(counts = new.counts, 
#                                         fragments=fragments,
#                                         annotation=Annotation(seurat_obj)) # seurat_obj[["peaks_merged"]]@counts
#     print(signac_obj)
#     # add the fragment object to the seurat object
#     # Fragments(signac_obj) <- fragments

#     # add the signac object to the seurat object ("peaks_integrated" assay)
#     seurat_obj[["peaks_integrated"]] <- signac_obj
#     # add the data_id to the object
#     seurat_obj$dataset <- data_id
#     return(seurat_obj)
# }

# # recompute the count matrices for each dataset, save as "peaks_integrated" ChromatinAssay
# TDR118 <- recompute_count_matrices(TDR118, combined.peaks = combined.peaks, fragments=frags.TDR118, data_id="TDR118")
# TDR119 <- recompute_count_matrices(TDR119, combined.peaks = combined.peaks, fragments=frags.TDR119, data_id="TDR119")
# TDR124 <- recompute_count_matrices(TDR124, combined.peaks = combined.peaks, fragments=frags.TDR124, data_id="TDR124")
# TDR125 <- recompute_count_matrices(TDR125, combined.peaks = combined.peaks, fragments=frags.TDR125, data_id="TDR125")
# TDR126 <- recompute_count_matrices(TDR126, combined.peaks = combined.peaks, fragments=frags.TDR126, data_id="TDR126")
# TDR127 <- recompute_count_matrices(TDR127, combined.peaks = combined.peaks, fragments=frags.TDR127, data_id="TDR127")
# TDR128 <- recompute_count_matrices(TDR128, combined.peaks = combined.peaks, fragments=frags.TDR128, data_id="TDR128")

# save the individual re-processed objects (with "peaks_integrated" assay)
# saveRDS(TDR118, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118.RDS")  
# saveRDS(TDR119, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR119.RDS")
# saveRDS(TDR124, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR124.RDS")
# saveRDS(TDR125, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR125.RDS")
# saveRDS(TDR126, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR126.RDS")
# saveRDS(TDR127, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR127.RDS")
# saveRDS(TDR128, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR128.RDS")

# print("Saved the re-processed individual objects")

# Load all seurat objects (replace the filepaths if you use this for your own datasets)
TDR118 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118.RDS")
TDR119 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR119.RDS")
TDR124 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR124.RDS")
TDR125 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR125.RDS")
TDR126 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR126.RDS")
TDR127 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR127.RDS")
TDR128 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR128.RDS")

print("Loaded all re-processed objects")

# Now, we need to remove the "peaks_merged" assay as that'll cause some memory issue when we try "merge" functions.
# Set "peaks_integrated" as the default assay for each Seurat object
DefaultAssay(TDR118) <- 'peaks_integrated'
DefaultAssay(TDR119) <- 'peaks_integrated'
DefaultAssay(TDR124) <- 'peaks_integrated'
DefaultAssay(TDR125) <- 'peaks_integrated'
DefaultAssay(TDR126) <- 'peaks_integrated'
DefaultAssay(TDR127) <- 'peaks_integrated'
DefaultAssay(TDR128) <- 'peaks_integrated'

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
TDR118 <- remove_unwanted_assays(TDR118, assaysToKeep)
TDR119 <- remove_unwanted_assays(TDR119, assaysToKeep)
TDR124 <- remove_unwanted_assays(TDR124, assaysToKeep)
TDR125 <- remove_unwanted_assays(TDR125, assaysToKeep)
TDR126 <- remove_unwanted_assays(TDR126, assaysToKeep)
TDR127 <- remove_unwanted_assays(TDR127, assaysToKeep)
TDR128 <- remove_unwanted_assays(TDR128, assaysToKeep)

# # merge all datasets, adding a cell ID to make sure cell names are unique
combined <- merge(x = TDR118, 
                    y = list(TDR119, TDR124, TDR125, TDR126, TDR127, TDR128))#, 
                    #add.cell.ids = c("TDR118", "TDR119", "TDR124", "TDR125", "TDR126", "TDR127", "TDR128"))
# save the combined object
saveRDS(combined, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_v2.RDS")  
print("Saved the combined object")

# make sure to change to the assay containing common peaks
DefaultAssay(combined) <- "peaks_integrated"
combined <- RunTFIDF(combined)
combined <- FindTopFeatures(combined, min.cutoff = 20)
combined <- RunSVD(
  combined,
  reduction.key = 'LSI_',
  reduction.name = 'lsi',
  irlba.work = 400
)
combined <- RunUMAP(combined, dims = 2:30, reduction = 'lsi')

# DimPlot(combined, group.by = 'dataset', pt.size = 0.1)

saveRDS(combined, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_v2.RDS")  # save the combined object
print("Saved the combined object with LSI and UMAP")

# Step 3. Integrate the ATAC objects (batch-correction)
# source: https://stuartlab.org/signac/articles/integrate_atac
# Loading the combined object (if we are resuming from here - it already has LSI and UMAP computed from the simple merge)
# combined <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_v2.RDS")

# # (NOT NEEDED IF WE START FROM THE TOP)
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
saveRDS(integrated, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_rLSI_v2.RDS")  # save the combined object
print("Saved the integrated object")