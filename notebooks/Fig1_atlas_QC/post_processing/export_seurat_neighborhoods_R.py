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

# %%
suppressMessages(library(Seurat))
suppressMessages(library(Signac))
#library(Seurat)
#library(Signac)
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

# %%
getwd()

# %%
# import the Seurat object
# assays: RNA, integrated_peaks
# contains connectivities from each modality (RNA, integrated_peaks(ATAC), and WNN)
seurat <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_ga_master.rds")
seurat

# %%
# # import the individual object
# rna_integrated <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/")
# atac_integrated <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/")

# %%
# import the cell_ids from the filtered object
metadata <- read.csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/master_rna_atac_metadata.csv")
metadata %>% head()

# %%
# subset the Seurat object using "cell_ids" from the metadata (filtering out the low_quality cells)
cell_ids_to_keep <- metadata$X
# Subset the Seurat object
seurat_subset <- subset(seurat, cells = cell_ids_to_keep)
seurat_subset

# %%
seurat@graphs$wknn

# %%
# save the wknn (connectivities)
write.csv(seurat@graphs$wknn,
          "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/wknn.csv")   

# %%
# import the cell_ids from the filtered object (filtering out "low quality cells"

# then, export the 

# %%

                 # %%
                 write.csv(pbmc@neighbors$weighted.nn@nn.idx, 
          "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/neighbors_indices.csv")

write.csv(pbmc@neighbors$weighted.nn@nn.dist, 
          "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/distances.csv")

write.csv(pbmc@graphs$wknn,
          "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/connectivities")                                                                                                                                     

# %%
meta %>% head() 

# %%
# QC with the cell_ids from the neurips final object
cell_ids<-rownames(meta)
pbmc <- subset(pbmc, cells = cell_ids)

# %% [markdown]
# ## Sorting the order of cells using cell_ids alphabetically

# %%
# Get the order of the cell IDs
new_order <- order(rownames(pbmc@assays$RNA@data))

# Set the cell identities based on the new order
pbmc <- SetIdent(pbmc, value = new_order)
pbmc

# %%
pbmc@meta.data %>% head() 

# %%
pbmc

# %%
pbmc@meta.data %>% head() 

# %%
options(repr.plot.width=5, repr.plot.height=4)
options(repr.plot.res=300)
DimPlot(pbmc)

# %%
meta = read.csv('/mnt/ibm_lg/alejandro/neurIPS/cell_annotation_paper/data/metadata.csv')
row.names(meta) <- meta$X

# %%
meta %>% head() 

# %%
# filter for the cells that passed the neurips QC
common_cells <- intersect(colnames(pbmc),row.names(meta))

# %%
pbmc <- AddMetaData(pbmc, metadata = meta[common_cells,])

# %%
names(pbmc@meta.data)

# %%
pbmc

# %%
options(repr.plot.width=10, repr.plot.height=4)


DimPlot(pbmc, group.by = c('manual_annotation_fine', 'manual_annotation_coarse') ) 



# %%

options(repr.plot.width=9, repr.plot.height=8)


DimPlot(pbmc, group.by = c('cell_type_RNA', 'cell_type_ATAC') ,ncol = 1) 


# %%
options(repr.plot.width=8, repr.plot.height=6)
DimPlot(pbmc, group.by = c('cell_type') ,
        reduction="wnn.umap", ncol = 1, 
        label=TRUE, repel=TRUE) +  NoLegend() 


# %%
wnn.umap <- pbmc@reductions$wnn.umap


# %%
# save the Seurat object
saveRDS(pbmc, "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/s1d1_rna_atac_wnn_seurat.rds", 
        )

# %% [markdown]
# ## Checking the weighted nearest neighbors (wnn)

# %%
neighborhood_matrix = pbmc@neighbors$weighted.nn@nn.dist
neighborhood_index =  pbmc@neighbors$weighted.nn@nn.idx
pbmc@graphs

# %%
write.csv(pbmc@neighbors$weighted.nn@nn.idx, 
          "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/neighbors_indices.csv")

write.csv(pbmc@neighbors$weighted.nn@nn.dist, 
          "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/distances.csv")

write.csv(pbmc@graphs$wknn,
          "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/connectivities")

# %%
metadata = pbmc@meta.data
WNN_weights = subset(x=metadata, select = c(SCT.weight, ATAC.weight))
WNN_weights

# %%
write.csv(WNN_weights, 
          "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/wnn_weights.csv")

# %% [markdown]
# # Convert the Seurat object to h5ad object

# %%
library(SeuratDisk)

# %%
# Make sure that we have the raw counts in the @x field
# Note that this should be checked manually for now, but we can think of adding a unit test for this (whether the sum is integer or not.)
pbmc@assays$RNA@data@x <- pbmc@assays$RNA@counts@x

# Set the directory to the output_dir where we will save the h5 and h5ad files
output_dir = "/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/"
setwd(output_dir)

# Save as seurat h5
# RNA object
SaveH5Seurat(pbmc, overwrite = TRUE, filename = '/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/s1d1_rna_atac_wnn_seurat.h5Seurat')

#convert to h5ad, writes to disk
Convert("/mnt/ibm_lg/yangjoon.kim/excellxgene_tutorial_manuscript/data/neurips2021_multiome/s1d1_rna_atac_wnn_seurat.h5Seurat", dest = "h5ad")
