# ## Notebook to integrate multiome objects (Seurat/Signac)

# - Last updated: 02/06/2024
# - Author: Yang-Joon Kim

# - Step 1. load the multiome objects from all timepoints (6 timepoints, 2 replicates from 15-somites)
# - Step 2. re-process the ATAC objects (merging peaks, re-computing the count matrices, then re-computing the PCA/LSI/SVD, seurat integration).
# - Step 3. integrate the RNA objects (seurat integration using rPCA or CCA)
# - Step 4. (optional - another notebook) alignUMAP for individual timepoints

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
options(future.globals.maxSize = 512 * 1024 ^ 3) # for  Gb RAM

# Step 1. Load the datasets
# Load all seurat objects (replace the filepaths if you use this for your own datasets)
TDR118 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/TDR118_processed.RDS")
TDR119 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR119reseq/TDR119_processed.RDS")
TDR124 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR124reseq/TDR124_processed.RDS")
TDR125 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR125reseq/TDR125_processed.RDS")
TDR126 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR126/TDR126_processed.RDS")
TDR127 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR127/TDR127_processed.RDS")
TDR128 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR128/TDR128_processed.RDS")

print("Loaded all the datasets")

# change the default assay to "SCT" (post-normalization and scaling)
DefaultAssay(TDR118) <- "SCT"
DefaultAssay(TDR119) <- "SCT"
DefaultAssay(TDR124) <- "SCT"
DefaultAssay(TDR125) <- "SCT"
DefaultAssay(TDR126) <- "SCT"
DefaultAssay(TDR127) <- "SCT"
DefaultAssay(TDR128) <- "SCT"

# Step 2. Integrate the RNA objects
print(paste("Integration ... "))
# 2-1. Load the list of objects
danio.list <- list(TDR118, TDR119, TDR124, TDR125, TDR126, TDR127, TDR128)


# 2-2. Find integration anchors and integrate data 
# Get the index for the "10hpf" entry
# index_10hpf <- which(names(danio.list) == "TDR126")
# index_10hpf

# # Get the index for the "24hpf" entry
# index_24hpf <- which(names(danio.list) == "TDR124")
# index_24hpf

# find the highly variable features
features <- SelectIntegrationFeatures(object.list = danio.list)

# find the integration anchors
danio.anchors <- FindIntegrationAnchors(object.list = danio.list, anchor.features = features,
                                       normalization.method = 'LogNormalize', #c("LogNormalize", "SCT"),
                                       dims = 1:30, # default 1:30
                                       k.anchor = 5, #default 5
                                       k.filter = 200, #default 200 for a query cell, If the anchor reference cell is found within the first k.filter (200) neighbors, then we retain this anchor.
                                       k.score = 30, # default 30: For each reference anchor cell, we determine its k.score (30) nearest within-dataset neighbors and its k.score nearest neighbors in the query dataset
                                       reduction = "rpca", # default cca, rpca should be faster 
                                       #reference = c(index_10hpf,index_24hpf) # the 10hpf and 24hpf timepoints as "references" that other datasets will be anchored against
                                       )
# Integreate the datasets using "anchors" computed above
seurat_combined <- IntegrateData(anchorset = danio.anchors, 
                                new.assay.name = "integrated",
                                dims=1:30,
                                k.weight = 30)

# 3. Generate an integrated embedding: run PCA on integrated (corrected) counts 
# specify that we will perform downstream analysis on the corrected data note that the
# original unmodified data still resides in the 'RNA' assay
DefaultAssay(seurat_combined) <- "integrated"

# Run the standard workflow for visualization and clustering
seurat_combined <- ScaleData(seurat_combined, verbose = FALSE)
seurat_combined <- RunPCA(seurat_combined, npcs = 100, verbose = FALSE)
print(paste("Runnning UMAP on the integrated PCA embedding..."))

# 4. UMAP on integrated embbedding 
seurat_combined <- RunUMAP(seurat_combined, reduction = "pca", dims = 1:30,
                             metric='euclidean',
                             n.neighbors = 30,
                             local.connectivity  =1, # 1 default
                             repulsion.strength = 1, # 1 default
                         )
seurat_combined <- FindNeighbors(seurat_combined, reduction = "pca", dims = 1:30)
seurat_combined <- FindClusters(seurat_combined, resolution = 0.5)

print(paste("plotting UMAP with different batch keys..."))
# Check the integrated UMAP
plot1 <- DimPlot(seurat_combined, dims = c(1, 2), group.by = "timepoint")
plot2 <- DimPlot(seurat_combined, dims = c(1, 2), group.by = "fish")
#plot3 <- DimPlot(seurat_combined, dims = c(1, 2), group.by = "leiden")
plot1 + plot2 #+ plot3

# 5. Export R object 
# name of the output file
saveRDS(seurat_combined, file = paste0("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/","integrated_RNA.rds")) 
print(paste("RDS object saved"))
