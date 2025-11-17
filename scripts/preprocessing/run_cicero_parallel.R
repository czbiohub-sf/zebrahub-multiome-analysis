# An R script to run Cicero for a Seurat object with a ChromatinAssay object
# Load the Cicero library from the local installation (trapnell lab branch for Signac implementation)
# library(remotes)
# library(devtools)
# install cicero
# withr::with_libpaths(new="/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", 
#                      install_github("cole-trapnell-lab/cicero-release", ref = "monocle3"))
# cicero (local installation - USE only if the global installation is broken)
#.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
#withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(monocle3))
# withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(cicero))

# load other libraries
library(cicero) # global installation
#withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(cicero))
library(Signac)
library(Seurat)
library(SeuratWrappers)

# inputs:
# 1) seurat_object: a seurat object
# 2) assay: "ATAC", "peaks", etc. - a ChromatinAssay object generated with Signac using the best peak profiles
# 3) dim_reduced: "UMAP.ATAC", "UMAP", "PCA", etc. - a dimensionality reduction. 
# NOTE that this should be "capitalized" as as.cell_data_set capitalizes all dim.red fields 
# 4) output_path: path to save the output (peaks, and CCANs)
# 5) data_id: ID for the dataset, i.e. TDR118
# 6) peaktype: type of the peak profile. i.e. CRG_arc: Cellranger-arc peaks
# (i.e. peaks_celltype, peaks_bulk, peaks_joint)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6) {
  stop("Usage: Rscript run_01_compute_CCANS_cicero.R seurat_object_path assay dim_reduced output_path data_id peaktype")
}

seurat_object_path <- args[1]
assay <- args[2]
dim_reduced <- args[3]
output_path <- args[4] 
data_id <- args[5] 
peaktype <- args[6]

# Example Input arguments:
# seurat_object <- readRDS(seurat_object_path)
# assay <- "ATAC"
# dim.reduced <- "umap.atac"
# output_path = "",
# data_id="TDR118",
# peaktype = "CRG_arc"

# capitalize the dim_reduced, as the single.cell.experiment data format capitalizes all the fields
dim_reduced <- toupper(dim_reduced)

# Step 1. Import the Seurat object
seurat_object <- readRDS(seurat_object_path)
# print out the assays in the seurat object
print("Available assays:")
print(names(seurat_object@assays))
print(paste("Working with assay:", assay))


# define the default assay
# We will pick which peak profiles we will use.
# ideally, we have defined the peak profiles 
# (and corresponding count matrices of cells-by-peaks) from the previous steps.
DefaultAssay(seurat_object) <- assay
print(paste0("default assay is ", assay))



# conver to CellDataSet (CDS) format
seurat_object.cds <- as.cell_data_set(x=seurat_object) # a function from SeuratWrappers\
print("cds object created") 

# print out the dim.reduced in the seurat object
print("Available reduced dimensions:")
print(names(reducedDims(seurat_object.cds)))
print(paste("Using reduced dimension:", dim_reduced))

# Step 2. make the cicero object
# default: we will use the ATAC.UMAP here for the sampling of the neighborhoods - as we'll treat this dataset as if we only had scATAC-seq.
# This is something we can ask Kenji Kamimoto/Samantha Morris later for their advice. (or compare the built GRNs from joint.UMAP vs ATAC.UMAP)
seurat_object.cicero <- make_cicero_cds(seurat_object.cds, reduced_coordinates = reducedDims(seurat_object.cds)$UMAP.ATAC)
print("cicero object created")

# check the structure of the assay
print("Checking assay structure:")
print(paste("Class of assay object:", class(seurat_object@assays[[assay]])))


# define the genomic length dataframe (chromosome number ; length)
# Check if seqinfo exists
if (is.null(seurat_object@assays[[assay]]@seqinfo)) {
    print("Warning: seqinfo is NULL")
    # Try to get seqinfo from annotation if available
    if (!is.null(seurat_object@assays[[assay]]@annotation)) {
        print("Attempting to get seqinfo from annotation")
        df_seqinfo <- as.data.frame(seurat_object@assays[[assay]]@annotation@seqinfo)
    } else {
        stop("Neither seqinfo nor annotation available in assay")
    }
} else {
    df_seqinfo <- as.data.frame(seurat_object@assays[[assay]]@seqinfo)
}

print("Chromosome lengths:")  # NEW: Added print
print(df_seqinfo$seqlengths[1:26])  # NEW: Added print
# df_seqinfo <- as.data.frame(seurat_object@assays[[assay]]@seqinfo)
# df_seqinfo <- as.data.frame(seurat_object@assays$ATAC@seqinfo)
# # zebrafish has 25 chromosomes and 1 MT chromosome
# seurat_object@assays$ATAC@annotation@seqinfo@seqlengths <- df_seqinfo$seqlengths[1:26] 

# create a dataframe for chromsomes and their lengths
# get the chromosome sizes from the Seurat object
# genome <- seqlengths(seurat_object@assays$ATAC@annotation)
genome <- seqlengths(seurat_object@assays[[assay]]@annotation)

# convert chromosome sizes to a dataframe
genome.df <- data.frame("chr" = names(genome), "length" = genome)
print("Genome dataframe created:")
print(head(genome.df))

# Step 3. Run Cicero (This part can be parallelized)

# run cicero
conns <- run_cicero(seurat_object.cicero, genomic_coords = genome.df, sample_num = 100)
print("CCANs computed")

# Return the CCAN results
# return(conns)

# saves the Cicero results 
# (1.all peaks as well as 2. pairwise cicero result)
all_peaks <- row.names(seurat_object@assays[[assay]]@data)
# all_peaks <- row.names(seurat_object@assays$ATAC@data)

#output_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data"
write.csv(x = all_peaks, file = paste0(output_path, "01_", data_id, "_",peaktype, "_peaks.csv"))
write.csv(x = conns, file = paste0(output_path, "02_", data_id, "_cicero_connections_",peaktype, "_peaks.csv"))