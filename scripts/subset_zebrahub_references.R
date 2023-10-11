# Subset the Zebrahub references (h5Seurat object) for each timepoint, for the annotation transfer.

# Load the libraries
library(Seurat)
library(SeuratDisk)

# set the working directory for where we want to save the h5Seurat objects
setwd("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/ZF_atlas_v01/")

# Load the Seurat object (6 timepoints)
zebrahub <- LoadH5Seurat("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/ZF_atlas_v01/ZF_atlas_v01.h5Seurat")

# print all timepoints
print(unique(zebrahub@meta.data$timepoint))

# subset for each timepoint and save it as a h5Seurat object
for (stage in unique(zebrahub@meta.data$timepoint)){
    print(stage)
    zebrahub_subset <- subset(zebrahub, timepoint==stage)
    filename = paste0("ZF_atlas_v01_", stage,".h5Seurat")
    SaveH5Seurat(zebrahub_subset, filename, overwrite=TRUE)
}
