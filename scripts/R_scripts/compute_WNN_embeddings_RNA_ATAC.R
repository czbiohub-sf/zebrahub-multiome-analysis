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

plan("multicore", workers = 16)
plan()

# set the max memory size for the future
options(future.globals.maxSize = 1024 * 1024 ^ 3) # for  Gb RAM

# Step 1. Load the datasets
# Load both RNA and ATAC seurat objects (replace the filepaths if you use this for your own datasets)
# NOTE: "RNA" Seurat object has RNA and SCT assays, and "ATAC" Seurat object has "integrated_ATAC" assays.
integrated_RNA <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA.rds")
integrated_ATAC <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_rLSI_annotated.rds")

# First, ensure that the order of cells in both objects is indeed the same
# This step is critical because we will be replacing the cell IDs based on order
# If the order is not the same, this method will incorrectly align the data
# This check is a basic sanity check and not a full proof of order
RenameCells(integrated_RNA, new.names=colnames(integrated_ATAC[["peaks_integrated"]]))

# if(identical(rownames(integrated_RNA@meta.data), rownames(integrated_ATAC@meta.data))) {
#   print("The cell IDs and their order match between the two objects.")
# } else {
#   # If the order or names don't match, you'll need to investigate further before proceeding
#   print("The cell IDs or their order do not match between the two objects. Copying over the RNA cell nams to the ATAC object.")
#   # If they are the same, then proceed to set the cell names of integrated_ATAC to match integrated_RNA
#   # Set cell names for assays in integrated_ATAC
#   for(assay in c("RNA","SCT")) {
#     colnames(integrated_RNA[[assay]]) <- colnames(integrated_ATAC[["peaks_integrated"]])
#   }
#   # Set cell names for dimensional reductions in integrated_ATAC
#   for(reduction in names(integrated_RNA@reductions)) {
#     rownames(integrated_RNA@reductions[[reduction]]@cell.embeddings) <- colnames(integrated_ATAC[["peaks_integrated"]])
#   }
  
#   # Set cell names for meta.data in integrated_ATAC
#   rownames(integrated_RNA@meta.data) <- rownames(integrated_ATAC@meta.data)
# }

# Step 2. merge the ATAC modality into the RNA Seurat object
integrated_RNA[["peaks_integrated"]] <- integrated_ATAC[["peaks_integrated"]]

# copy over the dim.reductions
# transfer the dimensionality reductions from ATAC to RNA
integrated_RNA@reductions$umap.atac <- integrated_ATAC@reductions$umap
integrated_RNA@reductions$integrated_lsi <- integrated_ATAC@reductions$integrated_lsi

# rename the "integrated_pca" and "umap.rna" for RNA
# integrated_RNA@reductions$integrated_pca <- integrated_RNA@reductions$pca
integrated_RNA@reductions$umap.rna <- integrated_RNA@reductions$umap

# transfer the metadata from ATAC to RNA
# Assuming that the rownames of meta.data for both objects correspond to the same cells
# Combine the metadata, preferring data from integrated_RNA when overlap occurs
# This step requires careful handling to ensure that all data remains aligned
integrated_RNA@meta.data <- merge(integrated_RNA@meta.data, integrated_ATAC@meta.data, by="row.names", all=TRUE)

# The merge function adds a 'Row.names' column with the cell identifiers, set this as the rownames
rownames(integrated_RNA@meta.data) <- integrated_RNA@meta.data$Row.names

# Now, remove the 'Row.names' column as it's redundant
integrated_RNA@meta.data$Row.names <- NULL

# Step 3. Compute the weigted-nearest neighbors (WNN) embeddings for the integrated RNA and ATAC

# Identify multimodal neighbors. These will be stored in the neighbors slot, and can be accessed using bm[['weighted.nn']]
# The WNN graph can be accessed at bm[["wknn"]], and the SNN graph used for clustering at bm[["wsnn"]]
# Cell-specific modality weights can be accessed at bm$RNA.weight
integrated_RNA <- FindMultiModalNeighbors(
  integrated_RNA, reduction.list = list("integrated_pca", "integrated_lsi"), 
  dims.list = list(1:30, 2:40), modality.weight.name = "RNA.weight"
)

# compute the UMAP and clusters using WNNs
integrated_RNA <- RunUMAP(integrated_RNA, nn.name = "weighted.nn", n.neighbors = 30,
                          reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
integrated_RNA <- FindClusters(integrated_RNA, graph.name = "wsnn", algorithm = 4, verbose = FALSE)

# Step 4. Export R object 
# name of the output file
saveRDS(seurat_combined, file = paste0("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/","integrated_RNA_ATAC_WNN.rds")) 
print(paste("RDS object saved"))

# Step 5. Convert the Seurat object to h5Seurat and h5ad
setwd("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/")
SaveH5Seurat(integrated_ATAC, filename = "integrated_RNA_ATAC_WNN.h5Seurat", overwrite = TRUE)
Convert("integrated_RNA_ATAC_WNN.h5Seurat", dest = "h5ad", overwrite = TRUE)