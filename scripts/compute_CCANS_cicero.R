# An R script to run Cicero for a Seurat object with a ChromatinAssay object

library(Signac)
library(Seurat)
library(SeuratWrappers)
#library(ggplot2)
#library(patchwork)

# inputs:
# seurat_object: a seurat object
# assay: "ATAC", "peaks", etc. - a ChromatinAssay object generated with Signac using the best peak profiles
# dim_reduced: "ATAC.UMAP", "UMAP", "PCA", etc. - a dimensionality reduction 

compute_CCANS_cicero <- function(seurat_object, assay, dim_reduced){
    # print out the assays in the seurat object
    #seurat_object@assays
    #print() 

    # define the default assay
    # We will pick which peak profiles we will use.
    # ideally, we have defined the peak profiles 
    # (and corresponding count matrices of cells-by-peaks) from the previous steps.
    DefaultAssay(seurat_object) <- assay

    # conver to CellDataSet (CDS) format
    seurat_object.cds <- as.cell_data_set(x=seurat_object) # a function from SeuratWrappers\

    # make the cicero object
    # default: we will use the ATAC.UMAP here for the sampling of the neighborhoods - as we'll treat this dataset as if we only had scATAC-seq.
    # This is something we can ask Kenji Kamimoto/Samantha Morris later for their advice. (or compare the built GRNs from joint.UMAP vs ATAC.UMAP)
    seurat_object.cicero <- make_cicero_cds(seurat_object.cds, reduced_coordinates = reducedDims(seurat_object.cds)$dim_reduced)


    # Perform CCAN computation
    # ...

    # Return the CCAN results
    # ...
}