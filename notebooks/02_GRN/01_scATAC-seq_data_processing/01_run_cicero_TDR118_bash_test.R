# Load the Cicero library from the local installation (trapnell lab branch for Signac implementation)
library(remotes)
library(devtools)
# install cicero
withr::with_libpaths(new="/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", 
                     install_github("cole-trapnell-lab/cicero-release", ref = "monocle3"))
# cicero
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(cicero))

# load other libraries
library(Signac)
library(Seurat)
library(SeuratWrappers)
#library(ggplot2)
#library(patchwork)

# inputs:
# seurat_object: a seurat object
# assay: "ATAC", "peaks", etc. - a ChromatinAssay object generated with Signac using the best peak profiles
# dim_reduced: "ATAC.UMAP", "UMAP", "PCA", etc. - a dimensionality reduction 

seurat_object <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/TDR118_processed.rds")
#assay <- "ATAC"
dim.reduced <- "ATAC.UMAP"

# print out the assays in the seurat object
#seurat_object@assays
#print() 

# define the default assay
# We will pick which peak profiles we will use.
# ideally, we have defined the peak profiles 
# (and corresponding count matrices of cells-by-peaks) from the previous steps.
DefaultAssay(seurat_object) <- "ATAC"

# conver to CellDataSet (CDS) format
seurat_object.cds <- as.cell_data_set(x=seurat_object) # a function from SeuratWrappers\

print("cds object created")
# make the cicero object
# default: we will use the ATAC.UMAP here for the sampling of the neighborhoods - as we'll treat this dataset as if we only had scATAC-seq.
# This is something we can ask Kenji Kamimoto/Samantha Morris later for their advice. (or compare the built GRNs from joint.UMAP vs ATAC.UMAP)
seurat_object.cicero <- make_cicero_cds(seurat_object.cds, reduced_coordinates = reducedDims(seurat_object.cds)$dim_reduced)
print("cicero object created")
# define the genomic length dataframe (chromosome number ; length)
df_seqinfo <- as.data.frame(seurat_object@assays$ATAC@seqinfo)
seurat_object@assays$ATAC@annotation@seqinfo@seqlengths <- df_seqinfo$seqlengths[1:26]

# seurat_object@assays$ATAC@annotation@seqinfo

# Perform CCAN computation
# get the chromosome sizes from the Seurat object
genome <- seqlengths(seurat_object@assays$ATAC@annotation)

# use chromosome 1 to save some time
# omit this step to run on the whole genome
# genome <- genome[1]

# convert chromosome sizes to a dataframe
genome.df <- data.frame("chr" = names(genome), "length" = genome)
print(genome.df)
# run cicero
conns <- run_cicero(seurat_object.cicero, genomic_coords = genome.df, sample_num = 100)
print("CCANs computed")
# Return the CCAN results
# ...
#all_peaks <- row.names(exprs(TDR118.cds))
all_peaks <- row.names(seurat_object@assays$ATAC@data)
output_folder <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data"
write.csv(x = all_peaks, file = paste0(output_folder, "/TDR118_CRG_arc_peaks.csv"))
write.csv(x = conns, file = paste0(output_folder, "/TDR118_cicero_connections_CRG_arc_peaks.csv"))