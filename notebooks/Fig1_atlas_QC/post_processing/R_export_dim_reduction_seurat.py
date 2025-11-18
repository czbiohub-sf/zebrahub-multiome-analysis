# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Global R
#     language: R
#     name: r_4.3
# ---

# %% [markdown]
# ## Exporting the dim.reductions from individual Seurat objects
#
# - last updated: 11/7/2024
#
# ### Goals:
# - export the dimensionality reduction from the individual objects
# - NOTE that the objects still has "low_quality" cells from Merlin's annotation, which needs to be filtered later.
#

# %%
suppressMessages(library(Signac))
suppressMessages(library(Seurat))
suppressMessages(library(GenomeInfoDb))

library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(patchwork)
library(stringr)
library(VennDiagram)
library(GenomicRanges)
library(Matrix)

# zebrafish genome
library(BSgenome.Drerio.UCSC.danRer11)

# %%
library(reticulate)
py_config()

# %%
library(reticulate)
py_config()
system("pip install leidenalg python-igraph")

# %%
# seurat <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_rLSI.RDS")
# seurat

# %%
seurat_obj <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC.rds")
seurat_obj

# %%

# %% [markdown]
# ## Step 0. export the WNN (neighborhoods)
#

# %%
seurat_obj@graphs$wsnn %>% head()

# %%
seurat_obj@neighbors$weighted.nn

# %%
# Extract the WSNN matrix
wsnn_matrix <- seurat_obj@graphs$wsnn

# Save as a Matrix Market format (efficient for sparse matrices)
output_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"
Matrix::writeMM(wsnn_matrix, file = paste0(output_dir, "wsnn_matrix.mtx"))

# Save row and column names (important for proper cell alignment)
cell_names <- colnames(seurat_obj)
writeLines(cell_names, paste0(output_dir, "cell_names.txt"))

# %%
# Optionally, save some metadata about the matrix for reference
metadata <- data.frame(
  n_cells = ncol(seurat_obj),
  n_nonzero = length(wsnn_matrix@x),
  density = length(wsnn_matrix@x) / (ncol(seurat_obj)^2),
  date = Sys.Date()
)
write.csv(metadata, file = paste0(output_dir, "wsnn_metadata.csv"), row.names = FALSE)

# %% [markdown]
# ## Step 1. exporting the raw counts in "peaks_integrated" assay (for scATAC-seq h5ad object)

# %%
# Extract the raw count matrix (sparse) from the 'peaks_integrated' assay
peak_counts <- seurat@assays$peaks_integrated@counts

# Optionally confirm it's a sparse matrix
class(peak_counts)
# Should be something like "dgCMatrix"

# %%
# path where we want to save the files
out_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/peaks_integrated"

# Write the matrix in Matrix Market format
writeMM(peak_counts, 
        file = file.path(out_dir, "peaks_integrated_counts.mtx")
       )

# Write feature names (row names) to file
write.table(
  rownames(peak_counts),
  file = file.path(out_dir, "peaks_integrated_features.tsv"),
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

# Write barcode names (column names) to file
write.table(
  colnames(peak_counts),
  file = file.path(out_dir, "peaks_integrated_barcodes.tsv"),
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

# %%

# %% [markdown]
# ## Step 2. re-computing the leiden clusters from the joint UMAP (cell umap)
# - last updated: 03/26/2025

# %%
seurat_obj

# %%
seurat_obj@meta.data %>% head()

# %%
DimPlot(seurat_obj, reduction = "wnn.umap", group.by="wsnn_res.0.8", label=TRUE, repel=TRUE)

# %%
seurat_obj$`wsnn_res.0.8` %>% head()

# %%
# recompute the leiden clustering with different resolutions for comparison (coarse and fine)
seurat_obj <- FindClusters(seurat_obj, graph.name = "wsnn", algorithm = 4, 
                           resolution=1, verbose = FALSE)
# save it to the meta.data
seurat_obj$`wsnn_res.1` <- Idents(seurat_obj)

# %%
# # recompute the leiden clustering with different resolutions for comparison (coarse and fine)
# seurat_obj <- FindClusters(seurat_obj, graph.name = "wsnn", algorithm = 4, 
#                            resolution=1, cluster.name="wsnn_res.1", verbose = FALSE)
# # save it to the meta.data
# seurat_obj$`wsnn_res.1` <- Idents(seurat_obj)

# # resolution=0.5
# seurat_obj <- FindClusters(seurat_obj, graph.name = "wsnn", algorithm = 4, 
#                            resolution=0.5, cluster.name="wsnn_res.0.5", verbose = FALSE)
# # save it to the meta.data
# seurat_obj$`wsnn_res.0.5` <- Idents(seurat_obj)

# # resolution=1.2
# seurat_obj <- FindClusters(seurat_obj, graph.name = "wsnn", algorithm = 4, 
#                            resolution=1.2, cluster.name="wsnn_res.1.2", verbose = FALSE)
# # save it to the meta.data
# seurat_obj$`wsnn_res.1.2` <- Idents(seurat_obj)

# %%

# %%

# %%

# %%

# %%
seurat_obj@assays[["peaks_integrated"]]@annotation@seqinfo@seqnames

# %%
seurat_obj[["peaks_integrated"]]

# %%
# step 1. add the genome annotation
# path to the GTF file
gff_path = "/hpc/reference/sequencing_alignment/alignment_references/"
gref_path = paste0(gff_path, "zebrafish_genome_GRCz11/genes/genes.gtf.gz")
gtf_zf <- rtracklayer::import(gref_path)

# make a gene.coord object
gene.coords.zf <- gtf_zf
# filter out the entries without the gene_name
gene.coords.zf <- gene.coords.zf[! is.na(gene.coords.zf$gene_name),]

# only keep the regions within standard chromosomes
gene.coords.zf <- keepStandardChromosomes(gene.coords.zf, pruning.mode = 'coarse')
# name the genome - GRCz11
genome(gene.coords.zf) <- 'GRCz11'

# copy the "gene_id" for the "tx_id" and "transcript_id" 
gene.coords.zf$tx_id <- gene.coords.zf$gene_id
gene.coords.zf$transcript_id <- gene.coords.zf$gene_id

# %%
# Extract seqinfo from the annotation
seqinfo_data <- seqinfo(gene.coords.zf)
seqinfo_data

# %% [markdown]
# ### test for the cds object creation

# %%
assay <- "peaks_integrated"
DefaultAssay(seurat_obj) <- assay
print(paste0("default assay is ", assay))

# %%
seurat_object.cds <- as.cell_data_set(x=seurat_obj)

# %%
seurat_object.cds

# %%
# print out the available reduced dimensions
print("Available reduced dimensions:")
print(names(reducedDims(seurat_object.cds)))
# Get the reduced coordinates
reduced_coords <- reducedDims(seurat_object.cds)[[dim_reduced]]
print("name of the dim.reduction")
print(dim_reduced)
print("Class of reduced coordinates:")
print(class(reduced_coords))
print("Dimensions of reduced coordinates:")
print(dim(reduced_coords))

# %% [markdown]
# ### validate the seurat object (whether it contains the relevant fields or not)
#

# %%
seurat_obj@assays$peaks_integrated

# %%
seqinfo

# %%
df_seqinfo <- as.data.frame(seurat_obj@assays[[assay]]@seqinfo)
df_seqinfo

# %%
seurat@assays[["peaks_integrated"]]@seqinfo

# %%

# %%

# %%

# %%
# data path
data_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"

# TDR118 (15-somites)
TDR118 <- readRDS(paste0(data_path, "TDR118reseq/TDR118_processed.RDS"))


# %%
TDR118@reductions

# %%
# Set up paths
data_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"  # Replace with your actual path
# output_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"   # Replace with where you want to save CSVs

# Create list of dataset IDs
dataset_ids <- c("TDR118", "TDR119", "TDR124", "TDR125", "TDR126", "TDR127", "TDR128")

# Function to handle the path based on whether it's a reseq dataset
get_dir_path <- function(id) {
  if (id %in% c("TDR126", "TDR127", "TDR128")) {
    return(paste0(data_path, id))
  } else {
    return(paste0(data_path, id, "reseq"))
  }
}

# Loop through datasets
for (id in dataset_ids) {
  # Construct path
  dir_path <- get_dir_path(id)
  seurat_path <- file.path(dir_path, paste0(id, "_processed.RDS"))
  
  # Load Seurat object
  print(paste("Processing", id))
  seurat_obj <- readRDS(seurat_path)
  
  # Export PCA
  if ("pca" %in% names(seurat_obj@reductions)) {
    pca_df <- as.data.frame(seurat_obj@reductions$pca@cell.embeddings)
    write.csv(pca_df, 
              file = file.path(dir_path, paste0(id, "_pca.csv")))
  }
  
  # Export LSI
  if ("lsi" %in% names(seurat_obj@reductions)) {
    lsi_df <- as.data.frame(seurat_obj@reductions$lsi@cell.embeddings)
    write.csv(lsi_df, 
              file = file.path(dir_path, paste0(id, "_lsi.csv")))
  }
  
  # Clean up to free memory
  rm(seurat_obj)
  gc()
  
  print(paste("Completed processing", id))
}

print("All datasets processed!")

# %% [markdown]
# ## Check the integrated Seurat object (counts)

# %%
# data path
data_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_wnn_gene_activity_3d_umaps.rds"

# TDR118 (15-somites)
# TDR118 <- readRDS(paste0(data_path, "TDR118reseq/TDR118_processed.RDS"))
integrated_seurat <- readRDS(data_path)
integrated_seurat

# %%
integrated_seurat@assays$peaks_integrated@counts
