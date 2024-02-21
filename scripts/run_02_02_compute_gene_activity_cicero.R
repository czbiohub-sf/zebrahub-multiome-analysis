# An R script to compute "cicero gene activity"
# This script is a part of the pipeline to analyze the multi-ome data from the zebrafish project.
# Last updated: 02/20/2024

# Load the Cicero library from the local installation (trapnell lab branch for Signac implementation)
# library(remotes)
# library(devtools)
# install cicero
# withr::with_libpaths(new="/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", 
#                      install_github("cole-trapnell-lab/cicero-release", ref = "monocle3"))
# cicero
.libPaths("/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib")
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(monocle3))
withr::with_libpaths(new = "/hpc/scratch/group.data.science/yangjoon.kim/.local/R_lib", library(cicero))

# load other libraries
#library(cicero)
library(Signac)
library(Seurat)
library(SeuratWrappers)
library(readr)
# library(dplyr)

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
  stop("Usage: Rscript run_02_02_compute_gene_activity_cicero.R seurat_object_path cicero_path gref_path assay data_id peaktype")
}
seurat_object_path <- args[1]
cicero_path <- args[2]
gref_path <- args[3]
assay <- args[3] # "peaks_merged" 
data_id <- args[4] # "TDR118reseq"
peaktype <- args[5] #"peaks_merged"

# Example Input arguments:
# seurat_object_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/TDR118_processed.RDS" 
# cicero_path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/02_cicero_processed/TDR118reseq_cicero/"
# gref_path <- "/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/genes/genes.gtf.gz"
# assay <- "peaks_merged" 
# data_id <- "TDR118reseq"
# peaktype <- "peaks_merged"


# Description of "Gene.Activity" computation in Cicero:
# requirements: (1) CDS object, (2) cicero connections, and (3) gene_annotation (GTF file)
# (3) gene_annotation (GTF) is needed to map the first exon of each gene as the "promoter" to compute the sum of the chromatin accessibilities for each gene (with co-access scores as weights)
# NOTE. CDS object is the Seurat object converted to the CellDataSet format

# Step 1. import a seurat object and convert it to CDS object
seurat_object <- readRDS(seurat_object_path)
DefaultAssay(seurat_object) <- assay
print(paste0("default assay is ", assay))

# conver to CellDataSet (CDS) format
seurat_object.cds <- as.cell_data_set(x=seurat_object) # a function from SeuratWrappers
print("cds object created") 

# Reformat the CDS object with feature-level metadata
# Step 1-1: Extract and process row names
site_names <- rownames(seurat_object.cds)
chr_bp_info <- strsplit(site_names, "-")
chr <- sapply(chr_bp_info, function(x) x[1])
bp1 <- sapply(chr_bp_info, function(x) x[2])
bp2 <- sapply(chr_bp_info, function(x) x[3])

# Step 1-2: Calculate num_cells_expressed
# Assuming 'counts' assay is used to calculate expression
counts_matrix <- counts(seurat_object.cds)
num_cells_expressed <- rowSums(counts_matrix > 0)

# Step 1-3: Update rowData
rowData(seurat_object.cds)$site_name <- site_names
rowData(seurat_object.cds)$chr <- chr
rowData(seurat_object.cds)$bp1 <- as.numeric(bp1)
rowData(seurat_object.cds)$bp2 <- as.numeric(bp2)
rowData(seurat_object.cds)$num_cells_expressed <- num_cells_expressed

# Verify the updated rowData
head(fData(seurat_object.cds))

# # define the genomic length dataframe (chromosome number ; length)
# df_seqinfo <- as.data.frame(seurat_object@assays$ATAC@seqinfo)
# # zebrafish has 25 chromosomes
# seurat_object@assays$ATAC@annotation@seqinfo@seqlengths <- df_seqinfo$seqlengths[1:26] 

# Step 2. import the cicero connections (result of run_cicero)
library(readr)
conns_filepath = paste0(cicero_path, "02_", data_id, "_cicero_connections_",peaktype, "_peaks.csv")
conns <- read_csv(conns_filepath, col_types = cols(.default = col_guess(), `...1` = col_skip()))

# # Step 3. import the gene annotation
# Step 3. import the gene annotation
gene_anno <- rtracklayer::readGFF(gref_path)
# check the gene_annotation dataframe
head(gene_anno)

# rename some columns to match requirements
gene_anno$chromosome <- paste0("chr", gene_anno$seqid)
gene_anno$gene <- gene_anno$gene_id
# gene_anno$transcript_id <- gene_anno$gene_id # rename the transcript_id using the gene_id
gene_anno$transcript <- gene_anno$transcript_id
gene_anno$symbol <- gene_anno$gene_name

# Step 4. compute CCANs 
#(CCANs: cis-Co-Accessibility Networks: a community/cluster of highly co-accessible peaks)
CCAN_assigns <- generate_ccans(conns)

head(CCAN_assigns)

# Step 5. compute the gene activity score using cicero results

# Step 5-1. reformat the gene_anno to anntoate the CDS for each gene. 
# If not annotated, we'll use the first exon

# Add a column for the pData table indicating the gene if a peak is a promoter ####
# Create a gene annotation set that only marks the transcription start sites of 
# the genes. We use this as a proxy for promoters.
# To do this we need the first exon of each transcript
pos <- subset(gene_anno, strand == "+")
pos <- pos[order(pos$start),] 
# remove all but the first exons per transcript
pos <- pos[!duplicated(pos$transcript),] 
# make a 1 base pair marker of the TSS
pos$end <- pos$start + 1 

neg <- subset(gene_anno, strand == "-")
neg <- neg[order(neg$start, decreasing = TRUE),] 
# remove all but the first exons per transcript
neg <- neg[!duplicated(neg$transcript),] 
neg$start <- neg$end - 1

gene_annotation_sub <- rbind(pos, neg)

# Make a subset of the TSS annotation columns containing just the coordinates 
# and the gene name
gene_annotation_sub <- gene_annotation_sub[,c("chromosome", "start", "end", "symbol")]

# Remove the 'chr' prefix from the 'chromosome' column in 'gene_annotation_sub'
gene_annotation_sub$chromosome <- gsub("chr", "", gene_annotation_sub$chr)

# Rename the gene symbol column to "gene"
names(gene_annotation_sub)[4] <- "gene"

# annotate the CDS object with the TSS/gene
seurat_object.cds <- annotate_cds_by_site(seurat_object.cds, gene_annotation_sub)

tail(fData(seurat_object.cds))

# Step 5-2.Generate gene activity scores

# Check if there are any NA values in the 'coaccess' column
anyNA_conns <- any(is.na(conns$coaccess))
print(paste("Are there any NA values in 'coaccess'? ", anyNA_conns))

# Check if there are any infinite values in the 'coaccess' column
anyInf_conns <- any(is.infinite(conns$coaccess))
print(paste("Are there any infinite values in 'coaccess'? ", anyInf_conns))

# Remove rows with NA or infinite values in 'coaccess'
conns_clean <- conns[!is.na(conns$coaccess) & !is.infinite(conns$coaccess), ]

# generate unnormalized gene activity matrix
unnorm_ga <- build_gene_activity_matrix(seurat_object.cds, conns_clean)

# remove any rows/columns with all zeroes
unnorm_ga <- unnorm_ga[!Matrix::rowSums(unnorm_ga) == 0, 
                       !Matrix::colSums(unnorm_ga) == 0]

# make a list of num_genes_expressed
num_genes <- pData(seurat_object.cds)$num_genes_expressed
names(num_genes) <- row.names(pData(seurat_object.cds))

# normalize
cicero_gene_activities <- normalize_gene_activities(unnorm_ga, num_genes)

# # if you had two datasets to normalize, you would pass both:
# # num_genes should then include all cells from both sets
# unnorm_ga2 <- unnorm_ga
# cicero_gene_activities <- normalize_gene_activities(list(unnorm_ga, unnorm_ga2), 
#                                                     num_genes)

# save the normalized cicero_gene_activities
gene.activity.path = paste0(cicero_path, "06_", data_id, "_gene_activities_",peaktype, "_peaks.csv")
write.csv(cicero_gene_activities, gene.activity.path, row.names=TRUE, col.names=TRUE)