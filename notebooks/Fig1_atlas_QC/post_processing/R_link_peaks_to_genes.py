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
# ## Link scATAC-seq peaks to genes using correlation
#
# - last updated: 1/15/2025
# - Reference: https://stuartlab.org/signac/reference/linkpeaks
# - Reference: https://stuartlab.org/signac/articles/pbmc_multiomic#linking-peaks-to-genes 
#
# ### Description: 
# Find peaks that are correlated with the expression of nearby genes. For each gene, this function computes the correlation coefficient between the gene expression and accessibility of each peak within a given distance from the gene TSS, and computes an expected correlation coefficient for each peak given the GC content, accessibility, and length of the peak. The expected coefficient values for the peak are then used to compute a z-score and p-value.

# %%
suppressMessages(library(Signac))
suppressMessages(library(Seurat))
suppressMessages(library(GenomeInfoDb))
library(Rsamtools)  # for FaFile()

library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(patchwork)
library(stringr)
library(VennDiagram)
library(GenomicRanges)

# zebrafish genome
library(BSgenome.Drerio.UCSC.danRer11)

# %%
seurat <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_wnn_gene_activity_3d_umaps.rds")
seurat

# %%
seurat

# %%
# List of assays to keep
assaysToKeep <- c("peaks_integrated", "RNA", "SCT")

# Function to remove other assays from a Seurat object
remove_unwanted_assays <- function(seuratObject, assaysToKeep) {
  allAssays <- names(seuratObject@assays)
  assaysToRemove <- setdiff(allAssays, assaysToKeep)
  
  for (assay in assaysToRemove) {
    seuratObject[[assay]] <- NULL
  }
  
  return(seuratObject)
}

# Apply the function to each of your Seurat objects
seurat <- remove_unwanted_assays(seurat, assaysToKeep)
seurat

# %%
# path to your custom genome
custom_fa_path <- "/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/fasta/genome.fa"
GRCz11 <- FaFile(custom_fa_path)

# %%

# %%
# Filepath to your CSV
file_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_75cells_per_metacell/TDR118reseq_seacells_obs_annotation_ML_coarse.csv"

# Load the CSV into a data frame
df <- read.csv(file_path)

# Check the column names to identify the relevant ones
colnames(df)

# Extract the columns for cell-to-metacell mapping (adjust column names based on the actual file structure)
cell_metacell_mapping <- df[, c("index", "SEACell")] 
cell_metacell_mapping %>% head()

# %%
seurat@meta.data %>% head()

# %%
seurat$dataset %>% unique()

# %%
# Full list of data IDs
data_ids <- c('TDR118', 'TDR119', 'TDR124', 'TDR125', 'TDR126', 'TDR127', 'TDR128')

# Base directory for the CSV files
base_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_75cells_per_metacell/"

# Initialize an empty list to store data frames
dataframes <- list()

# Loop through each data ID
for (data_id in data_ids) {
  # Construct possible file paths
  if (data_id %in% c('TDR118', 'TDR119', 'TDR124', 'TDR125')) {
    # For datasets with potential "reseq"
    file_path_reseq <- paste0(base_dir, data_id, "reseq_seacells_obs_annotation_ML_coarse.csv")
    file_path_normal <- paste0(base_dir, data_id, "_seacells_obs_annotation_ML_coarse.csv")
  } else {
    # For datasets without "reseq"
    file_path_reseq <- NULL
    file_path_normal <- paste0(base_dir, data_id, "_seacells_obs_annotation_ML_coarse.csv")
  }
  
  # Determine the correct file path to use
  if (!is.null(file_path_reseq) && file.exists(file_path_reseq)) {
    file_path <- file_path_reseq
  } else if (file.exists(file_path_normal)) {
    file_path <- file_path_normal
  } else {
    warning(paste("No file found for:", data_id))
    next
  }
  
  # Load the CSV file
  df <- read.csv(file_path)
  
  # Check if the required columns exist
  if (!("index" %in% colnames(df)) || !("SEACell" %in% colnames(df))) {
    stop(paste("Missing required columns in file:", file_path))
  }
  
  # Append the dataset suffix to the 'index' column to match Seurat's naming convention
  df$index <- paste0(df$index, "_", which(data_ids == data_id))  # Appends _1, _2, etc., based on dataset order
  
  # Append the dataset suffix to the 'SEACell' column for compatibility
  df$SEACell <- paste0(df$SEACell, "_", which(data_ids == data_id))
  
  # Add the dataset ID as a separate column for tracking
  df$data_id <- data_id
  
  # Append to the list of data frames
  dataframes[[data_id]] <- df[, c("index", "SEACell", "data_id")]
}

# Concatenate all data frames into one
concatenated_df <- do.call(rbind, dataframes)

# %%
concatenated_df %>% head()

# %%
# Save the concatenated mapping to a CSV file
write.csv(concatenated_df, "concatenated_cell_metacell_mapping.csv", row.names = FALSE)

# %%
# Load the concatenated mapping
mapping <- read.csv("concatenated_cell_metacell_mapping.csv")

# Check the structure of the mapping
head(mapping)

# Ensure the Seurat object has matching cell names
seurat_cells <- colnames(seurat)
mapping_cells <- mapping$index

# Find common cells between Seurat object and the mapping
common_cells <- intersect(seurat_cells, mapping_cells)

# Subset the mapping to include only cells present in the Seurat object
mapping_filtered <- mapping[mapping$index %in% common_cells, ]

# Reorder the mapping to match the Seurat object cell names
mapping_filtered <- mapping_filtered[match(seurat_cells, mapping_filtered$index), ]

# Add the SEACell column to the Seurat meta.data
seurat$SEACell <- mapping_filtered$SEACell

# Verify the updated meta.data
head(seurat@meta.data)

# %%
# Verify the updated meta.data
tail(seurat@meta.data)

# %%
seurat

# %%
## filter for the 50K highly variable peaks

# Load the list of filtered peaks from the file
filtered_peaks <- readLines("peaks_hvp_50k.txt")
filtered_peaks %>% head()

# %%
seurat

# %%
# Load the progress package
library(progress)

# %%
# Subset the Seurat object to only include peaks in the filtered list
seurat[["peaks_integrated"]] <- subset(seurat[["peaks_integrated"]], features = filtered_peaks)
seurat

# %%
# Aggregate the RNA assay by SEACell
rna_aggregated <- AverageExpression(
  object = seurat_obj,
  assay = "RNA",
  group.by = "SEACell",
  slot = "data"  # Use normalized data; change to "counts" for raw counts if needed
)

# Aggregate the SCT assay by SEACell
sct_aggregated <- AverageExpression(
  object = seurat_obj,
  assay = "SCT",
  group.by = "SEACell",
  slot = "data"
)

# Aggregate the peaks_integrated assay by SEACell
peaks_aggregated <- AverageExpression(
  object = seurat_obj,
  assay = "peaks_integrated",
  group.by = "SEACell",
  slot = "data"
)

# Save the aggregated data to CSV files for further analysis if needed
write.csv(as.data.frame(rna_aggregated$RNA), "RNA_aggregated_by_SEACell.csv", row.names = TRUE)
write.csv(as.data.frame(sct_aggregated$SCT), "SCT_aggregated_by_SEACell.csv", row.names = TRUE)
write.csv(as.data.frame(peaks_aggregated$peaks_integrated), "peaks_integrated_aggregated_by_SEACell.csv", row.names = TRUE)

# %%

# %%

# %%
seurat@assays$

# %%
# Remove out-of-bound peaks
seurat <- RemoveOutOfBoundPeaks(
  seurat_obj  = seurat,
  ref_seqinfo = ref_seqinfo,
  assay_name  = "peaks_integrated"
)

# Now run RegionStats
DefaultAssay(seurat) <- "peaks_integrated"
seurat <- RegionStats(
  object = seurat,
  genome = GRCz11
)

# %%

# %%

# %%

# %%
# data_ids <- c('TDR118''TDR119''TDR124''TDR125''TDR126''TDR127''TDR128')

# # Base directory for the CSV files
# base_dir <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_75cells_per_metacell/"

# # Initialize an empty list to store data frames
# dataframes <- list()

# # Loop through each data ID
# for (data_id in data_ids) {
#   # Construct the file path for the SEACell mapping
#   file_path <- paste0(base_dir, data_id, "_seacells_obs_annotation_ML_coarse.csv")
  
#   # Check if the file exists
#   if (!file.exists(file_path)) {
#     warning(paste("File not found:", file_path))
#     next
#   }
  
#   # Load the SEACell CSV
#   df <- read.csv(file_path)
  
#   # Check if the required columns exist
#   if (!("index" %in% colnames(df)) || !("SEACell" %in% colnames(df))) {
#     stop(paste("Missing required columns in file:", file_path))
#   }
  
#   # Append the dataset suffix to the 'index' column to match Seurat's naming convention
#   df$index <- paste0(df$index, "_", which(data_ids == data_id))  # Appends _1, _2, etc., based on dataset order
  
#   # Append the dataset suffix to the 'SEACell' column for compatibility
#   df$SEACell <- paste0(df$SEACell, "_", which(data_ids == data_id))
  
#   # Add the dataset ID as a separate column for tracking
#   df$data_id <- data_id
  
#   # Append to the list of data frames
#   dataframes[[data_id]] <- df[, c("index", "SEACell", "data_id")]
# }

# # Concatenate all data frames into one
# concatenated_df <- do.call(rbind, dataframes)

# # Save the concatenated mapping to a CSV file
# write.csv(concatenated_df, "concatenated_cell_metacell_mapping.csv", row.names = FALSE)

# # Display the first few rows of the concatenated data frame
# head(concatenated_df)

# %%

# %%

# %%
# # step 1. add the genome annotation
# # path to the GTF file
# gff_path = "/hpc/reference/sequencing_alignment/alignment_references/"
# gref_path = paste0(gff_path, "zebrafish_genome_GRCz11/genes/genes.gtf.gz")
# gtf_zf <- rtracklayer::import(gref_path)

# # make a gene.coord object
# gene.coords.zf <- gtf_zf
# # filter out the entries without the gene_name
# gene.coords.zf <- gene.coords.zf[! is.na(gene.coords.zf$gene_name),]

# # only keep the regions within standard chromosomes
# gene.coords.zf <- keepStandardChromosomes(gene.coords.zf, pruning.mode = 'coarse')
# # name the genome - GRCz11
# genome(gene.coords.zf) <- 'GRCz11'

# # copy the "gene_id" for the "tx_id" and "transcript_id" 
# gene.coords.zf$tx_id <- gene.coords.zf$gene_id
# gene.coords.zf$transcript_id <- gene.coords.zf$gene_id

# %%
# DefaultAssay(seurat) <- "peaks_integrated"

# # first compute the GC content for each peak
# seurat <- RegionStats(seurat, genome = BSgenome.Drerio.UCSC.danRer11)

# %%
# extract the current peaks
peak_ranges <- granges(seurat[["peaks_integrated"]])

# We compare the set of peak chroms to what's in the reference
peak_ranges <- dropSeqlevels(
  x = peak_ranges,
  value = setdiff(seqlevels(peak_ranges), seqlevels(GRCz11)),
  pruning.mode = "coarse"
)

# Trim out-of-bounds intervals(peaks)
peak_ranges <- trim(peak_ranges)
# make sure that the peaks have width (non-zero)
peak_ranges <- peak_ranges[width(peak_ranges) > 0]

seurat[["peaks_integrated"]] <- SetAssayData(
  object = seurat[["peaks_integrated"]],
  slot   = "ranges",
  new.data = peak_ranges
)

# %%
peak_ranges <- granges(seurat[["peaks_integrated"]])
offender <- peak_ranges[
  seqnames(peak_ranges) == "3" &
    start(peak_ranges) <= 62628504 &
    end(peak_ranges) >= 62628283
]
offender

# %%
peak_ranges <- peak_ranges[! (
  seqnames(peak_ranges) == "3" &
  start(peak_ranges) == 62628283 &
  end(peak_ranges)   == 62628504
)]

# %%
seurat[["peaks_integrated"]] <- SetAssayData(
  object = seurat[["peaks_integrated"]],
  slot   = "ranges",
  new.data = peak_ranges
)

# first compute the GC content for each peak
DefaultAssay(seurat) <- "peaks_integrated"
seurat <- RegionStats(seurat, genome = GRCz11)

# %%
peak_to_remove <- "3-62628283-62628504"
peak_assay <- seurat[["peaks_integrated"]]

# 1) Exclude the peak from the assay features
features_keep <- setdiff(rownames(peak_assay), peak_to_remove)

# 2) Subset the ChromatinAssay using these features
peak_assay_subset <- subset(
  x = peak_assay,
  features = features_keep
)

peak_assay_subset

# %%
# Check that names(peak_ranges) match rownames(peak_assay_subset)
all(names(peak_ranges) == rownames(peak_assay_subset))  # should be TRUE

peak_assay_subset <- SetAssayData(
  object  = peak_assay_subset,
  slot    = "ranges",
  new.data = peak_ranges
)

seurat[["peaks_integrated"]] <- peak_assay_subset

# %%
# standardize the peak naming convention (chr_num-start-end)

library(GenomicRanges)
library(GenomeInfoDb)

# Extract peaks from your ChromatinAssay
peaks_gr <- granges(seurat[["peaks_integrated"]])

# If your peaks look like "1", "2" and you need them to be "chr1", "chr2", ...
# rename each level to add "chr".
old_levels <- seqlevels(peaks_gr)
# old_levels is something like c("1", "2", "3", ...)
new_levels <- paste0("chr", old_levels)
peaks_gr <- renameSeqlevels(peaks_gr, new_levels)

# Now put the updated GRanges back into the assay
seurat[["peaks_integrated"]] <- SetAssayData(
  object = seurat[["peaks_integrated"]],
  slot   = "ranges",
  new.data = peaks_gr
)


# %%
# Extract peak ranges
peak_ranges <- granges(seurat[["peaks_integrated"]])

# Check actual seqlengths for chr3 in BSgenome
chr3_len <- seqlengths(BSgenome.Drerio.UCSC.danRer11)["chr3"]
chr3_len

# Subset just the chr3 peaks
chr3_peaks <- peak_ranges[seqnames(peak_ranges) == "chr3"]

# Find peaks that exceed this chromosome boundary
invalid_peaks <- chr3_peaks[end(chr3_peaks) > chr3_len]
invalid_peaks

# %%
# this code is now moved to an R script, for slurm submission

# %%

# %%

# %%

# %%

# %%

# %%
RemoveOutOfBoundPeaks <- function(
  seurat_obj,
  ref_seqinfo,
  assay_name = "peaks_integrated"
) {
  library(GenomicRanges)
  library(GenomeInfoDb)

  # 1) Extract the ChromatinAssay and its peaks (GRanges)
  peak_assay <- seurat_obj[[assay_name]]
  peak_ranges <- granges(peak_assay)

  # 2) Get a named vector of chromosome lengths
  #    (from a FaFile or DNAStringSet, e.g. seqinfo(GRCz11))
  chr_lens <- seqlengths(ref_seqinfo)

  # 3) Identify which peaks are truly "in bound":
  #    - On a chromosome that exists in ref_seqinfo
  #    - start >= 1
  #    - end <= chromosome length
  seqn   <- as.character(seqnames(peak_ranges))
  starts <- start(peak_ranges)
  ends   <- end(peak_ranges)

  in_ref <- seqn %in% names(chr_lens)
  in_lower_bound <- (starts >= 1)
  in_upper_bound <- ends <= chr_lens[seqn]  # references the length for each chromosome

  keep_vec <- in_ref & in_lower_bound & in_upper_bound

  # 4) Turn those valid intervals into the "peak names" used by the ChromatinAssay
  # Usually, Signac rownames look like "chr3:62628283-62628504" or "3:62628283-62628504"
  old_feature_names <- rownames(peak_assay)  # existing rownames in the assay
  # The new "kept" features (in-bound)
  features_keep <- old_feature_names[keep_vec]

  # 5) Subset the ChromatinAssay in one go
  #    This automatically updates the counts, meta.features, and ranges
  peak_assay_subset <- subset(
    x        = peak_assay,
    features = features_keep
  )

  # 6) Place it back into the Seurat object
  seurat_obj[[assay_name]] <- peak_assay_subset

  message(
    sum(!keep_vec), " out-of-bound peaks removed; ", 
    sum(keep_vec), " remain."
  )
  return(seurat_obj)
}

# %%
# Suppose GRCz11 is a FaFile or DNAStringSet with seqinfo(GRCz11) => valid zebrafish chromosome lengths
ref_seqinfo <- seqinfo(GRCz11)

# Remove out-of-bound peaks
seurat <- RemoveOutOfBoundPeaks(
  seurat_obj  = seurat,
  ref_seqinfo = ref_seqinfo,
  assay_name  = "peaks_integrated"
)

# Now run RegionStats
DefaultAssay(seurat) <- "peaks_integrated"
seurat <- RegionStats(
  object = seurat,
  genome = GRCz11
)

# %%
seurat[["peaks_integrated"]]

# %%
# save the intermediate object (without two peaks)
saveRDS(seurat, file="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_ga_master.rds")

# %%
seurat

# %% [markdown]
# ### Run RegionStats and compute the peak-gene linkage

# %%
DefaultAssay(seurat) <- "peaks_integrated"
seurat <- RegionStats(
  object = seurat,
  genome = GRCz11  # your FaFile or DNAStringSet
)

# %%
head(seurat[["peaks_integrated"]]@meta.features)

# %%

# %% [markdown]
# ## Check the result of the LinkPeaks

# %%
seurat <- readRDS("/hpc//projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data//01_Signac_processed/peak_gene_links.csv")

# %%

# %%

# %%

# %%
