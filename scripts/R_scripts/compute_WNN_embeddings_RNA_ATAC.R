# ## Notebook to compute weighted-nearest neighbors for the integrated multiome objects (Seurat/Signac)

# Last updated: 03/17/2024
# Author: Yang-Joon Kim

# NOTES:
# 1) Make sure to load R/4.3 module and also module anaconda, and activate the conda environment where leiden algorithm is installed (i.e., single-cell-base)

# Step 1. load the multiome objects - both RNA and ATAC  (integrated over 6 timepoints, 2 replicates from 15-somites)
# Step 2. copy over the dim.reduction and embeddings from one (RNA) to the other (ATAC)
# Step 3. compute the WNN and joint embeddings for the integrated RNA and ATAC

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
options(future.globals.maxSize = 512 * 1024 ^ 3) # for  512Gb RAM

# Step 1. Load the datasets
# Load both RNA and ATAC seurat objects (replace the filepaths if you use this for your own datasets)
# NOTE: "RNA" Seurat object has RNA and SCT assays, and "ATAC" Seurat object has "integrated_ATAC" assays.
integrated_RNA <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA.rds")
integrated_ATAC <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_rLSI_v2.RDS")

# NOTE: ensure that the order of cells in both objects is indeed the same
# This is important for the metadata to be aligned correctly

# First, we will transfer the "integrated" assay from RNA to ATAC (this is RNA integrated over time, 
# over which "integrated_pca" was computed). This is import for FindMultiModalNeighbors function
integrated_ATAC[["integrated"]] <- integrated_RNA[["integrated"]]
  
#   # # Set cell names for meta.data in integrated_ATAC
#   # rownames(integrated_RNA@meta.data) <- rownames(integrated_ATAC@meta.data)
# }

# transfer the dimensionality reductions from RNA to ATAC
integrated_ATAC@reductions$integrated_pca <- integrated_RNA@reductions$integrated_pca
integrated_ATAC@reductions$umap.rna <- integrated_RNA@reductions$umap

# rename the "umap.atac" for ATAC
integrated_ATAC@reductions$umap.atac <- integrated_ATAC@reductions$umap

# save the combined object (RNA and ATAC)
# there are three assays: (1) integrated_peaks, (2) RNA, (3) SCT
# the integrated_peaks assay is the ATAC-seq peaks, and RNA and SCT are the RNA-seq assays
saveRDS(integrated_ATAC, file = paste0("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/","integrated_RNA_ATAC.rds"))

# transfer the metadata from ATAC to RNA
# Assuming that the rownames of meta.data for both objects correspond to the same cells
# Combine the metadata, preferring data from integrated_RNA when overlap occurs
# This step requires careful handling to ensure that all data remains aligned
# integrated_ATAC@meta.data <- merge(integrated_RNA@meta.data, integrated_ATAC@meta.data, by="row.names", all=TRUE)

# The merge function adds a 'Row.names' column with the cell identifiers, set this as the rownames
# rownames(integrated_RNA@meta.data) <- integrated_RNA@meta.data$Row.names

# Now, remove the 'Row.names' column as it's redundant
# integrated_RNA@meta.data$Row.names <- NULL

# Step 3. Compute the weigted-nearest neighbors (WNN) embeddings for the integrated RNA and ATAC

# Identify multimodal neighbors. These will be stored in the neighbors slot, and can be accessed using integrated_ATAC[['weighted.nn']]
# The WNN graph can be accessed at integrated_ATAC[["wknn"]], and the SNN graph used for clustering at integrated_ATAC[["wsnn"]]
# Cell-specific modality weights can be accessed at integrated_ATAC$RNA.weight
integrated_ATAC <- FindMultiModalNeighbors(
  integrated_ATAC, reduction.list = list("integrated_pca", "integrated_lsi"), 
  dims.list = list(1:30, 2:40), modality.weight.name = "integrated.weight"
)

# compute the UMAP and clusters using WNNs
integrated_ATAC <- RunUMAP(integrated_ATAC, nn.name = "weighted.nn", n.neighbors = 30,
                          reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
integrated_ATAC <- FindClusters(integrated_ATAC, graph.name = "wsnn", algorithm = 4, verbose = FALSE)

# Step 4. Export R object 
# name of the output file
saveRDS(integrated_ATAC, file = paste0("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/","integrated_RNA_ATAC.rds")) 
print(paste("RDS object saved"))

# Step 5. Convert the Seurat object to h5Seurat and h5ad
# First, export the "peaks_integrated" counts layer to h5ad
setwd("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/")
SaveH5Seurat(integrated_ATAC, filename = "integrated_RNA_ATAC_counts_peaks_integrated.h5Seurat", overwrite = TRUE)
Convert("integrated_RNA_ATAC_counts_peaks_integrated.h5Seurat", dest = "h5ad", overwrite = TRUE)

# Next, export the "RNA" counts layer to h5ad
DefaultAssay(integrated_ATAC) <- "RNA"
SaveH5Seurat(integrated_ATAC, filename = "integrated_RNA_ATAC_counts_RNA.h5Seurat", overwrite = TRUE)
Convert("integrated_RNA_ATAC_counts_RNA.h5Seurat", dest = "h5ad", overwrite = TRUE)

# Step 6. Compute the Gene.Activity score using Signac (for "peaks_integrated" assay)
DefaultAssay(integrated_ATAC) <- "peaks_integrated"

# compute_gene_activity <- function(object=multiome){    
#     # DefaultAssay(multiome) <-"peaks_integrated"
#     # we use the Signac function "GeneActivity"
#     gene.activities <- GeneActivity(multiome)
#     # add the gene activity matrix to the Seurat object as a new assay and normalize it
#     multiome[['Gene.Activity']] <- CreateAssayObject(counts = gene.activities)
    
#     multiome <- NormalizeData(
#         object = multiome,
#         assay = 'Gene.Activity',
#         normalization.method = 'LogNormalize',
#         scale.factor = 10000
#     )
    
#     return(multiome)
#     }

# # compute the gene activity for the integrated multiome object
# integrated_ATAC <- compute_gene_activity(integrated_ATAC)

# save the resulting output to h5Seurat and h5ad

# This is just the compute_gene_activity function opened up
# we use the Signac function "GeneActivity"
gene.activities <- GeneActivity(integrated_ATAC)
# add the gene activity matrix to the Seurat object as a new assay and normalize it
integrated_ATAC[['Gene.Activity']] <- CreateAssayObject(counts = gene.activities)

integrated_ATAC <- NormalizeData(
    object = integrated_ATAC,
    assay = 'Gene.Activity',
    normalization.method = 'LogNormalize',
    scale.factor = 10000
)
DefaultAssay(integrated_ATAC) <- "Gene.Activity"

SaveH5Seurat(integrated_ATAC, filename = "integrated_RNA_ATAC_counts_gene_activity.h5Seurat", overwrite = TRUE)
Convert("integrated_RNA_ATAC_counts_gene_activity.h5Seurat", dest = "h5ad", overwrite = TRUE)