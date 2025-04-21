# ## Notebook to integrate multiome(RNA and ATAC) objects (Seurat for RNA and Signac/ChromatinAssay for ATAC)
# ATAC merge reference: https://stuartlab.org/signac/articles/merging#:~:text=The%20merge%20function%20defined%20in,object%20being%20merged%20become%20equivalent.
# ATAC integration reference: 
# RNA integration reference:

# - Last updated: 03/14/2024
# - Author: Yang-Joon Kim

# Step 1. load the multiome objects from all timepoints (6 timepoints, 2 replicates from 15-somites)
# Step 2. re-process the ATAC objects (merging peaks, re-computing the count matrices, then re-computing the PCA/LSI/SVD, seurat integration).
# Step 3. merge the ATAC objects (as well as RNA/SCT assays) using "merge" function in Seurat
# Step 4. integrate the ATAC objects using Seurat's integration method (rLSI, etc.)
# Step 5. integrate the RNA/SCT objects using Seurat's integration method (rPCA, etc.) 

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

# make sure to install the necessary python packages for Seurat (leiden clustering)
library(reticulate)
# py_config()
system("pip install leidenalg python-igraph")

# parallelization in Signac: https://stuartlab.org/signac/articles/future
library(future)
plan("multicore", workers = 8)
# set the max memory size for the future
options(future.globals.maxSize = 512 * 1024 ^ 3) # for 512 Gb RAM

# import the seurat object
seurat_obj <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC.rds")
seurat_obj

# Step 2. re-compute the leiden clustering using the "wsnn", weighted nearest neighbor
# recompute the leiden clustering with different resolutions for comparison (coarse and fine)

# Refactor repeated FindClusters calls into a loop
resolutions <- c(0.5, 1, 1.2, 1.5, 2)

for (res in resolutions) {
    seurat_obj <- FindClusters(seurat_obj, graph.name = "wsnn", algorithm = 4, 
                                resolution=res, verbose = FALSE)
    seurat_obj[[paste0("wsnn_res_", res)]] <- Idents(seurat_obj)
    print(paste("Leiden clustering with resolution", res, "is done."))
}

print("Leiden clustering with wsnn is done.")

# save the object
saveRDS(seurat_obj, "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_wsnn_leiden.rds")

# Step 3. Visualize the leiden clustering results
# plot the UMAPs colored by the leiden clustering results
options(repr.plot.width=10, repr.plot.height=5)
plot_list <- list()
for (res in resolutions) {
    plot_list[[res]] <- DimPlot(seurat_obj, group.by = paste0("wsnn_res_", res), label = TRUE, repel = TRUE) + 
        ggtitle(paste("Leiden clustering with resolution", res))
}

plot_list[[1]] + plot_list[[2]] + plot_list[[3]] + plot_list[[4]] + plot_list[[5]]

# save the plot
ggsave("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/leiden_clustering_wsnn_resolutions.png", width=10, height=5)