# -*- coding: utf-8 -*-
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
# ## Coverage plot for integrated_ATAC datasets 
#
# - we will generate converage plots for some genes that change their chromatin accessibility over time (for all 6 timepoints, 7 datasets)
#
# - Last updated: 3/20/2024
# - Author: Yang-Joon Kim
#
# ### NOTE:
# - 1) We have the celltype annotation as "global_annotation". This was transferred from Zebrahub using Seurat's label transfer functions.
# - 2) As a result, there's "unassigned" cells, which were undetermined as the prediction confidence was lower than a threshold. 
# - 3) We'll have to manually curate our annotations and then re-run this notebook for a curated plots.
#
# (Reference: https://satijalab.org/seurat/reference/gettransferpredictions)

# %%
library(Seurat)
library(Signac)
library(patchwork)
library(ggplot2)
library(dplyr)


# %%
# Load the plotting module
source("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/utils/module_coverage_plot_viz.R")

# %%
# import the Seurat object
multiome <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_ATAC_rLSI.RDS")
multiome

# %%
# 1. Read the CSV file
# metadata_df <- read.csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_umap_3d_annotated.csv")
metadata_df <- read.csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/master_obj_obs.csv")

# Look at the structure to verify contents
head(metadata_df)
print(colnames(metadata_df))

# 2. Subset the Seurat object to keep only cells in the metadata
cells_to_keep <- metadata_df$X  # adjust column name if different
multiome_filtered <- subset(multiome, cells = cells_to_keep)

# %%
multiome_filtered

# %%

# %%
# 3. Add celltype annotations from the dataframe to the Seurat object
# First, make sure cell IDs match between metadata and Seurat object
rownames(metadata_df) <- metadata_df$X  # adjust column name if different

# # Add celltype annotation
# multiome_filtered$celltype <- metadata_df[Cells(multiome_filtered), "celltype"]  # adjust column name if different
multiome_filtered$tissue <- metadata_df[Cells(multiome_filtered), "tissue"]

# Verify the changes
print(table(multiome_filtered$tissue))

# %%
# 4. Map each celltype to a broader lineage via a dictionary
# Create a named character vector mapping each celltype to a lineage
# celltype_to_lineage <- c(
#   # Ectoderm
#   "epidermis"                 = "Epidermal",
#   "hatching_gland"            = "Epidermal",
#   "hindbrain"                 = "Neural",
#   "spinal_cord"               = "Neural",
#   "neural_optic"              = "Neural",
#   "neural_floor_plate"        = "Neural",
#   "neural_crest"              = "Neural",
#   "midbrain_hindbrain_boundary" = "Neural",
#   "neural_telencephalon"      = "Neural",
#   "differentiating_neurons"   = "Neural",
#   "neurons"                   = "Neural",
#   "optic_cup"                 = "Neural",
#   "neural_posterior"          = "Neural",
#   "enteric_neurons"           = "Neural",
#   "neural"                    = "Neural",

#   # Mesoderm
#   "PSM"                       = "Mesoderm",
#   "somites"                   = "Mesoderm",
#   "muscle"                    = "Mesoderm",
#   "fast_muscle"               = "Mesoderm",
#   "heart_myocardium"          = "Mesoderm",
#   "notochord"                 = "Mesoderm",
#   "NMPs"                      = "Mesoderm",
#   "lateral_plate_mesoderm"    = "Mesoderm",
#   "pronephros"                = "Mesoderm",
#   "pharyngeal_arches"         = "Mesoderm",
#   "floor_plate"               = "Mesoderm",  # or Neural – depends on your classification
#   "tail_bud"                  = "Mesoderm",

#   # Hematopoietic/Vascular
#   "hematopoietic_vasculature" = "Hematopoietic/Vascular",
#   "hemangioblasts"            = "Hematopoietic/Vascular",

#   # Endoderm
#   "endoderm"                  = "Endoderm",
#   "endocrine_pancreas"        = "Endoderm",

#   # Germline
#   "primordial_germ_cells"     = "Germline"
# )

# # Assign lineage by indexing the dictionary with each cell’s celltype
# multiome_filtered$lineage <- celltype_to_lineage[ multiome_filtered$celltype ]

# # If some cell types aren’t present in `celltype_to_lineage`, you’ll get NA;
# # optionally replace NA with "unassigned" or similar.
# multiome_filtered$lineage <- ifelse(
#   is.na(multiome_filtered$lineage),
#   "unassigned",
#   multiome_filtered$lineage
# )

# ###################################
# # 5. Verify the new lineage column
# ###################################
# table(multiome_filtered$lineage)

# %%
# Create a named character vector mapping each celltype to a lineage
celltype_to_lineage <- c(
  # CNS
  "neural"                      = "CNS",
  "neural_optic"                = "CNS",
  "neural_optic2"               = "CNS",
  "neural_posterior"            = "CNS",
  "neural_telencephalon"        = "CNS",
  "neurons"                     = "CNS",
  "hindbrain"                   = "CNS",
  "midbrain_hindbrain_boundary" = "CNS",
  "midbrain_hindbrain_boundary2" = "CNS",
  "optic_cup"                   = "CNS",
  "spinal_cord"                 = "CNS",
  "differentiating_neurons"      = "CNS",
  "floor_plate"                 = "CNS",
  "neural_floor_plate"          = "CNS",
  "enteric_neurons"             = "CNS",

  # Neural Crest
  "neural_crest"                = "Neural Crest",
  "neural_crest2"               = "Neural Crest",

  # Paraxial Mesoderm
  "somites"                     = "Paraxial Mesoderm",
  "fast_muscle"                 = "Paraxial Mesoderm",
  "muscle"                      = "Paraxial Mesoderm",
  "PSM"                         = "Paraxial Mesoderm",  # Presomitic mesoderm
  "floor_plate2"                = "Paraxial Mesoderm",
  "NMPs"                        = "Paraxial Mesoderm",  # Neuromesodermal progenitors
  "tail_bud"                    = "Paraxial Mesoderm",
  "notochord"                   = "Paraxial Mesoderm",

  # Lateral Mesoderm
  "lateral_plate_mesoderm"      = "Lateral Mesoderm",
  "heart_myocardium"            = "Lateral Mesoderm",
  "hematopoietic_vasculature"   = "Lateral Mesoderm",
  "pharyngeal_arches"           = "Lateral Mesoderm",
  "pronephros"                  = "Lateral Mesoderm",
  "pronephros2"                 = "Lateral Mesoderm",
  "hemangioblasts"              = "Lateral Mesoderm",
  "hatching_gland"              = "Lateral Mesoderm",

  # Endoderm
  "endoderm"                    = "Endoderm",
  "endocrine_pancreas"          = "Endoderm",

  # Epiderm
  "epidermis"                   = "Epiderm",
  "epidermis2"                  = "Epiderm",
  "epidermis3"                  = "Epiderm",
  "epidermis4"                  = "Epiderm",

  # Germline
  "primordial_germ_cells"       = "Germline"
)

# Assign lineage by indexing the dictionary with each cell’s celltype
seurat_obj$lineage <- celltype_to_lineage[seurat_obj$celltype ]

# Replace NA values with "unassigned" if any celltypes are not in the mapping
seurat_obj$lineage <- ifelse(
  is.na(seurat_obj$lineage),
  "unassigned",
  seurat_obj$lineage
)

# Verify the new lineage column
table(seurat_obj$lineage)


# %%
multiome_filtered@assays$peaks_integrated

# %%
# # Get existing fragments
# frags <- Fragments(multiome)

# # Keep only non-empty fragments
# non_empty_frags <- frags[c(1, 9, 17, 25, 33, 41, 49)]

# # Update the object with only non-empty fragments
# Fragments(multiome) <- non_empty_frags

# # Verify that we only have the fragments with cells
# Fragments(multiome)

# %%

# %%

# %%
# # log-normalize the RNA counts for plotting
# DefaultAssay(multiome) <- "RNA"
# multiome <- NormalizeData(multiome, normalization.method = "LogNormalize", scale.factor=10000)

# %%
# set the default assay back to "peaks_integrated"
DefaultAssay(multiome) <- "peaks_integrated"

# %% [markdown]
# ## Check for a couple of marker genes
#
# - tbxta, myf5, meox1, etc.

# %%
# # marker genes from ChatGPT
# early_genes <- c(
#   "sox19b",
#   "ntl",
#   "chd",
#   "gsc",
#   "eve1",
#   "snai1a",
#   "myod1",
#   "pou5f3",
#   "nanog",
#   "tbxta"
# )

# late_genes <- c(
#   "neurog1",
#   "pax2a",
#   "egr2b",
#   "myl7",
#   "dlx2a",
#   "foxi1",
#   "cldn15la",
#   "hand2",
#   "aldh1a2",
#   "fgf8"
# )

# %%
CoveragePlot(multiome, region = "tbxta")

# %% [markdown]
# ## Step 1. grouped.by datasets

# %%
# change the default "Idents" to "dataset" metadata (to group the fragments using "dataset")
Idents(object = multiome) <- "dataset"

# %%
# First, define your desired order
desired_order <- c("TDR126", "TDR127", "TDR128", "TDR118", "TDR119", "TDR125", "TDR124")

# Assuming multiome$dataset is a separate vector parallel to the identity classes in Seurat object
# Create a factor with levels in the order you want
new_idents <- factor(multiome$dataset, levels = desired_order)

# Assign these new ordered factors to the Seurat object
multiome <- Seurat::SetIdent(multiome, value = new_idents)

# %%
multiome

# %%
CoveragePlot(multiome, region = "myf5", 
             extend.upstream = 1000, extend.downstream = 1000)

# %%
CoveragePlot(multiome, region = "tbxta")

# %%
CoveragePlot(multiome, region = "myf5", 
             extend.upstream = 1000, extend.downstream = 1000)

# %%
CoveragePlot(multiome, region = "her1",
             extend.upstream = 1000, extend.downstream = 1000)

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## [DEPRECATED] Subset the seurat object for "celltype" - EDA on hox gene cluster
# - Also, add a new column for developmental stage ("dev_stage")
#

# %%
# Create a named vector for mapping dataset names to stages
stage_mapping <- c(
    "TDR126" = "10hpf",
    "TDR127" = "12hpf",
    "TDR128" = "14hpf",
    "TDR118" = "16hpf",
    "TDR119" = "16hpf",
    "TDR125" = "19hpf",
    "TDR124" = "24hpf"
)

# Add the stage information to both Seurat objects
multiome$dev_stage <- stage_mapping[multiome$dataset]
multiome_filtered$dev_stage <- stage_mapping[multiome_filtered$dataset]

# Convert to factor with correct order
stage_order <- c("10hpf", "12hpf", "14hpf", "16hpf", "19hpf", "24hpf")
multiome$dev_stage <- factor(multiome$dev_stage, levels = stage_order)
multiome_filtered$dev_stage <- factor(multiome_filtered$dev_stage, levels = stage_order)

# Verify the mapping
print("Original object stages:")
table(multiome$dataset, multiome$dev_stage)

print("\nFiltered object stages:")
table(multiome_filtered$dataset, multiome_filtered$dev_stage)

# %%
table(multiome_filtered$celltype)

# %%
# subset for specific celltype, then plot the temporal change of chromatin accessibility at the hoxc locus
nmps <- subset(multiome_filtered, celltype == "NMPs")
tail_bud <- subset(multiome_filtered, celltype == "tail_bud")
neural_posterior <- subset(multiome_filtered, celltype == "neural_posterior")
spinal_cord <- subset(multiome_filtered, celltype == "spinal_cord")
hindbrain <- subset(multiome_filtered, celltype == "hindbrain")

# %%
# subset for specific celltype, then plot the temporal change of chromatin accessibility at the hoxc locus
psm <- subset(multiome_filtered, celltype == "PSM")
somites <- subset(multiome_filtered, celltype == "somites")

# %%
# coverage plot for psm
CoveragePlot(psm, region = "hoxc8a", 
             extend.upstream = 100000, extend.downstream = 100000, ymax=490)

# %%
# coverage plot for somites
CoveragePlot(somites, region = "hoxc8a", 
             extend.upstream = 100000, extend.downstream = 100000, ymax=490)

# %%
# coverage plot for NMPs
CoveragePlot(nmps, region = "hoxc8a", 
             extend.upstream = 100000, extend.downstream = 100000, ymax=490)

# %%
# coverage plot for spinal_cord
CoveragePlot(spinal_cord, region = "hoxc8a", 
             extend.upstream = 100000, extend.downstream = 100000, ymax=490)

# %%
# coverage plot for neural_posterior
CoveragePlot(neural_posterior, region = "hoxc8a", 
             extend.upstream = 100000, extend.downstream = 100000, ymax=490)

# %%
# coverage plot for hindbrain
CoveragePlot(hindbrain, region = "hoxc8a", 
             extend.upstream = 100000, extend.downstream = 100000, ymax=490)

# %%
# coverage plot for tail_bud
CoveragePlot(tail_bud, region = "hoxc8a", 
             extend.upstream = 100000, extend.downstream = 100000, ymax=490)

# %%

# %%

# %%

# %% [markdown]
# ## Check the timecourse progression for one gene

# %%
gene = "myf5"

cov_plot <- CoveragePlot(multiome, region=gene, 
             extend.upstream = 1000, extend.downstream = 1000)

# expression of RNA
expr_plot <- ExpressionPlot(multiome, features = gene, assay = "RNA")

plot<-CombineTracks(
  plotlist = list(cov_plot),
  expression.plot = expr_plot,
  heights = c(8),
  widths = c(8, 1)
)

options(repr.plot.width = 8, repr.plot.height = 8, repr.plot.res = 300)
#     ggsave(paste0(filepath, "coverage_plot_", gene, "_allpeaks.png"), plot=plot, width=8, height=12)
return(plot)

# %%

# %% [markdown]
# ## Step 3. grouped.by tissues ("tissue")

# %%
# a function to generate the Coverage Plot with viridis colormap                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

# generate CoveragePlot using "group.by" parameter
generate_coverage_plots <- function(object = multiome, 
                                    gene=gene, 
                                    group.by="timepoint", 
                                    filepath=NULL,
                                    plot_viridis=FALSE,
                                    extend.up = 1000,
                                    extend.down = 1000,
                                    region.highlight=NULL,
                                    ymax=NULL){
    
    # Step 1. Check if gene exists in GTF file
    if (!gene %in% object@assays$peaks_integrated@annotation$gene_name) {
        cat("Gene", gene, "not found in GTF file. Skipping.\n")
        return(NULL)
      }
    
    # Step 2. we have to manually change the basic identity for Seurat
    Idents(object) <- group.by
    
    # if the group.by is "timepoint", then we'll re-order them
    if (group.by=="timepoint"){
        # First, define your desired order
        desired_order <- c("0 somites", "5 somites", "10 somites", "15 somites", "20 somites", "30 somites")

        # Create a factor with levels in the order you want
        new_idents <- factor(object$timepoint, levels = desired_order)

        # Assign these new ordered factors to the Seurat object
        object <- Seurat::SetIdent(object, value = new_idents)
    }
    # if the group.by is "timepoint", then we'll re-order them
    if (group.by=="global_annotation"){
        # First, define your desired order
        desired_order <- c("Epidermal","Differentiating_Neurons","Neural_Crest",
                           "Neural_Anterior","Neural_Posterior",
                           "NMPs","PSM","Somites","Muscle","Lateral_Mesoderm",
                           "Notochord","Endoderm","Adaxial_Cells","unassigned")

        # Create a factor with levels in the order you want
        new_idents <- factor(object$global_annotation, levels = desired_order)

        # Assign these new ordered factors to the Seurat object
        object <- Seurat::SetIdent(object, value = new_idents)
    }

    
    # Step 3. peak profile for the counts (grouped.by "group.by")
    cov_plot <- CoveragePlot(
        object = object, 
        region = gene,
        extend.downstream = extend.down, extend.upstream = extend.up,
        annotation = FALSE,
        peaks=FALSE, ymax=ymax, region.highlight=region.highlight,
    )

    
    # for gene/peak plots, we need to find the genomic locations as the old Signac doesn't take the gene name as an input argument.
    gene.coord <- LookupGeneCoords(object = object, gene = gene)
    gene.coord.df <- as.data.frame(gene.coord)
    
    # extract the chromosome number, start position and end position
    chromosome <- gene.coord.df$seqnames
    pos_start <- gene.coord.df$start
    pos_end <-gene.coord.df$end
    
    # compute the genomic region as "chromsome_number-start-end"
    genomic_region <- paste(chromosome, pos_start, pos_end, sep="-")
    
    # gene annotation
    gene_plot <- AnnotationPlot(
      object = object,
      region = genomic_region,
      extend.downstream = extend.down, extend.upstream=extend.up,
    )
    # gene_plot


#     # "peaks_integrated" peaks
#     peak_plot <- PeakPlot(
#       object = object,
#       region = genomic_region,
#       peaks=object@assays$peaks_integrated@ranges,
#       extend.downstream = extend.down, extend.upstream=extend.up,
#     ) + labs(y="peaks")


#     # expression of RNA
#     expr_plot <- ExpressionPlot(
#       object = object,
#       features = gene,
#       assay = "RNA"
#     )
    
    # add the viridis colormap if plot_viridis==TRUE
    if (plot_viridis){
        cov_plot <- cov_plot + scale_fill_viridis_d()
        expr_plot <- expr_plot + scale_fill_viridis_d()
    }

    plot<-CombineTracks(
      plotlist = list(cov_plot, gene_plot),
      # expression.plot = expr_plot,
      heights = c(10,2),
      widths = c(10, 1)
    )
    
    # options(repr.plot.width = 8, repr.plot.height = 12, repr.plot.res = 300)
#     ggsave(paste0(filepath, "coverage_plot_", gene, "_allpeaks.png"), plot=plot, width=8, height=12)
    return(plot)
    
}

# %%
# create a Coverage plot for "meox1", grouped by global_annotation
coverage_plot <- generate_coverage_plots(multiome_filtered, gene="meox1", 
                                         group.by = "tissue", plot_viridis = FALSE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# %%
# Open a PDF device
pdf("coverage_plot_meox1_timepoints.pdf", width = 5, height = 8)

# Draw the plot on the open device
coverage_plot

# Close the device to save the PDF file
dev.off()

# %%
# ggsave(filename = "coverage_plot_tissue_meox1.pdf", plot = coverage_plot, 
#        device)

# %%
goi = "nanos3"
# create a Coverage plot for goi, grouped by global_annotation
coverage_plot <- generate_coverage_plots(multiome_filtered, gene=goi, group.by = "tissue", plot_viridis = FALSE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# %%
goi = "nanos3"
# create a Coverage plot for goi, grouped by global_annotation
coverage_plot <- generate_coverage_plots(multiome_filtered, gene=goi, group.by = "lineage", plot_viridis = FALSE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# %%
goi = "hes6"
# create a Coverage plot for goi, grouped by global_annotation
coverage_plot <- generate_coverage_plots(multiome_filtered, gene=goi, group.by = "lineage", plot_viridis = FALSE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# %%
goi = "dnd1"
# create a Coverage plot for goi, grouped by global_annotation
coverage_plot <- generate_coverage_plots(multiome_filtered, gene=goi, group.by = "lineage", plot_viridis = FALSE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# %%

# %%
# Function to generate multiplexed coverage plots for a list of genes
generate_multiplex_coverage_plots <- function(object, 
                                              genes, 
                                              group.by = "tissue", 
                                              plot_viridis = FALSE, 
                                              extend.up = 1000, 
                                              extend.down = 1000, 
                                              region.highlight = NULL, 
                                              ymax = NULL) {
  
  # Initialize an empty list to store individual plots
  plot_list <- list()
  
  # Iterate over each gene in the provided list
  for (gene in genes) {
    
    # Check if gene exists in the GTF annotation
    if (!gene %in% object@assays$peaks_integrated@annotation$gene_name) {
      cat("Gene", gene, "not found in GTF file. Skipping.\n")
      next
    }
    
    # Set identity class based on grouping variable
    Idents(object) <- group.by
    
    # Generate coverage plot
    cov_plot <- CoveragePlot(
      object = object, 
      region = gene,
      extend.downstream = extend.down, 
      extend.upstream = extend.up,
      annotation = FALSE,
      peaks = FALSE, 
      ymax = ymax, 
      region.highlight = region.highlight
    )
    
    # Obtain genomic coordinates for the gene
    gene.coord <- LookupGeneCoords(object = object, gene = gene)
    gene.coord.df <- as.data.frame(gene.coord)
    
    # Extract chromosome number, start position, and end position
    chromosome <- gene.coord.df$seqnames
    pos_start <- gene.coord.df$start
    pos_end <- gene.coord.df$end
    
    # Compute the genomic region
    genomic_region <- paste(chromosome, pos_start, pos_end, sep = "-")
    
    # Generate gene annotation plot
    gene_plot <- AnnotationPlot(
      object = object,
      region = genomic_region,
      extend.downstream = extend.down, 
      extend.upstream = extend.up
    )
    
    # Apply viridis colormap if required
    if (plot_viridis) {
      cov_plot <- cov_plot + scale_fill_viridis_d()
    }
    
    # Combine coverage plot and gene annotation plot
    gene_combined_plot <- CombineTracks(
      plotlist = list(cov_plot, gene_plot),
      heights = c(10, 2)  # Adjust heights as needed
    )
    
    # Store the plot in the list
    plot_list[[gene]] <- gene_combined_plot
  }
  
  # Arrange all gene plots vertically in a single figure
  final_plot <- wrap_plots(plot_list, ncol = 1)
  
  return(final_plot)
}



# %%
# Example usage with a list of marker genes
marker_genes <- c("nova2", "ncam1a", "pax6a",
                  "fgfrl1b", "plxna2", "mcama",
                  "tp63", "epcam", "cdh1",
                  "nanos3", "dnd1", "tdrd7a",
                  "cdh11", "colec12", "adgra2",
                  "slc1a3a", "ednrab", "prex2", "pdgfra",
                  "fn1b","tbx16","msgn1","meox1")

# Generate the multiplexed coverage plot
multiplex_plot <- generate_multiplex_coverage_plots(
  object = multiome_filtered, 
  genes = marker_genes, 
  group.by = "tissue", 
  plot_viridis = FALSE
)

# Adjust plot theme for better visibility
multiplex_plot <- multiplex_plot + theme(
  text = element_text(size = 8),  # General text size
  axis.title = element_text(size = 12),  # Axis title size
  axis.text = element_text(size = 8),  # Axis text size
  legend.title = element_text(size = 8),  # Legend title size
  legend.text = element_text(size = 8)  # Legend text size
)

# Display the final plot
multiplex_plot

# %%

# %%
##############################
# Now produce a multi‐gene panel
##############################

# Say we have some markers of interest
genes_of_interest <- c(
"nanos3", "dnd1", "meox1"
)

# Loop over genes, collecting each coverage plot in a list
library(patchwork)  # for wrap_plots / +
plot_list <- lapply(genes_of_interest, function(g) {
  single_coverage_plot(
    object     = multiome_filtered,
    gene       = g,
    group.by   = "tissue",  # or "lineage" / "timepoint"
    extend.up  = 1000,
    extend.down= 1000
  )
})

# Combine them side by side
# (ncol = length(genes_of_interest) => one column per gene)
multi_coverage_figure <- wrap_plots(plot_list, ncol = length(genes_of_interest))marker_genesti_coverage_figure

# %%
# Remove y-axis from all but the first coverage plot
for (i in seq_along(plot_list)) {
  if (i > 1) {
    plot_list[[i]] <- plot_list[[i]] + theme(
      axis.text.y  = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y  = element_blank(),
      axis.title.y = element_blank()
    )
  }
}

# Finally, combine side-by-side
multi_coverage_figure <- wrap_plots(plot_list, ncol = length(genes_of_interest))
multi_coverage_figure

# %%
# Suppose we have this function that returns a CoveragePlot for a single gene:
# (similar to your 'generate_coverage_plots()', but simplified for brevity)
single_coverage_plot <- function(
  object,
  gene,
  group.by = "global_annotation",
  extend.up = 1000,
  extend.down = 1000
) {
  # 1. Set cluster identities
  Idents(object) <- group.by
  
  # 2. Coverage track
  cov_plot <- CoveragePlot(
    object = object,
    region = gene,
    extend.upstream = extend.up,
    extend.downstream = extend.down,
    annotation = FALSE,
    peaks = FALSE
  ) + ggtitle(gene) +
    theme(plot.title = element_text(hjust = 0.5))
  
  # # 3. Peak track
  # peak_plot <- PeakPlot(
  #   object  = object,
  #   region  = gene,
  #   peaks   = object@assays$peaks_integrated@ranges,
  #   extend.upstream = extend.up,
  #   extend.downstream = extend.down
  # ) + labs(y = "peaks")
  
  # 4. Gene annotation track
  gene_plot <- AnnotationPlot(
    object  = object,
    region  = gene,
    extend.upstream = extend.up,
    extend.downstream = extend.down
  )
  
#   # 5. Expression overlay
#   expr_plot <- ExpressionPlot(
#     object  = object,
#     features = gene,
#     assay    = "RNA"
#   )
  
  # Combine the coverage track and annotation tracks
  p <- CombineTracks(
    plotlist         = list(cov_plot, gene_plot),
    # expression.plot  = expr_plot,
    heights          = c(10,  2),
    widths           = c(8, 1)
  )
  
  return(p)
}

# %%
marker_genes <- c("nova2",
                  "fgfrl1b",
                  "tp63", "epcam",
                  "nanos3", "dnd1", 
                  "cdh11", 
                  "ednrab",
                  "tbx16")

# Loop over genes, collecting each coverage plot in a list
library(patchwork)  # for wrap_plots / +
plot_list <- lapply(marker_genes, function(g) {
  single_coverage_plot(
    object     = multiome_filtered,
    gene       = g,
    group.by   = "tissue",  # or "lineage" / "timepoint"
    extend.up  = 1000,
    extend.down= 1000
  )
})

# Combine them side by side
# (ncol = length(genes_of_interest) => one column per gene)
multi_coverage_figure <- wrap_plots(plot_list, ncol = length(marker_genes))
multi_coverage_figure

# %%
# Open a PDF device
pdf("coverage_plot_selected_tissue_markers.pdf", width = 30, height = 8)

# Draw the plot on the open device
multi_coverage_figure

# Close the device to save the PDF file
dev.off()

# %%
# # For each plot after the first column
# if (i > 1) {
#   plot_list[[i]] <- plot_list[[i]] +
#     scale_y_continuous(breaks = NULL) +
#     theme(
#       axis.text.y  = element_blank(),
#       axis.ticks.y = element_blank(),
#       axis.line.y  = element_blank(),
#       axis.title.y = element_blank(),
#       strip.text.y = element_blank()  # remove facet-label text
#     )
# }

# # Finally, combine side-by-side
# multi_coverage_figure <- wrap_plots(plot_list, ncol = length(genes_of_interest))
# multi_coverage_figure

# %% [markdown]
# ## Step 2. grouped.by timepoints
#
# - merge TDR118 and TDR119 as they are from the same timepoints (we'll use the meta.data slot for this)
# - generate a couple of CoveragePlots as pdf (vectorized images)
#
# - (optional) change the color palette to "viridis" to follow the timepoints used in UMAP (zebrahub)

# %%
# Define a mapping from dataset identifiers to timepoints
timepoint_mapping <- c("TDR126" = "0 somites", 
                       "TDR127" = "5 somites", 
                       "TDR128" = "10 somites",
                       "TDR118" = "15 somites",
                       "TDR119" = "15 somites",
                       "TDR125" = "20 somites",
                       "TDR124" = "30 somites")
timepoint_mapping

# create a new metadata column using the mapping dictionary above
multiome$timepoint <- timepoint_mapping[multiome$dataset]
multiome$timepoint %>% head

# set the default Ident as "timepoint"
Idents(multiome) <- "timepoint"

# %%
# First, define your desired order
desired_order <- c("0 somites", "5 somites", "10 somites", "15 somites", "20 somites", "30 somites")

# Create a factor with levels in the order you want
new_idents <- factor(multiome$timepoint, levels = desired_order)

# Assign these new ordered factors to the Seurat object
multiome <- Seurat::SetIdent(multiome, value = new_idents)

# %%
multiome$timepoint %>% unique

# %%
multiome$global_annotation %>% unique

# %%
# create a directory to save the plots
dir.create("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/ATAC_dynamics/")
setwd("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/ATAC_dynamics/")

# %%

# %%
# a function to generate the Coverage Plot with viridis colormap                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

# generate CoveragePlot using "group.by" parameter
generate_coverage_plots <- function(object = multiome, 
                                    gene=gene, 
                                    group.by="timepoint", 
                                    filepath=NULL,
                                    plot_viridis=FALSE,
                                    extend.up = 1000,
                                    extend.down = 1000,
                                    region.highlight=NULL,
                                    ymax=NULL){
    
    # Step 1. Check if gene exists in GTF file
    if (!gene %in% object@assays$peaks_integrated@annotation$gene_name) {
        cat("Gene", gene, "not found in GTF file. Skipping.\n")
        return(NULL)
      }
    
    # Step 2. we have to manually change the basic identity for Seurat
    Idents(object) <- group.by
    
    # if the group.by is "timepoint", then we'll re-order them
    if (group.by=="timepoint"){
        # First, define your desired order
        desired_order <- c("0 somites", "5 somites", "10 somites", "15 somites", "20 somites", "30 somites")

        # Create a factor with levels in the order you want
        new_idents <- factor(object$timepoint, levels = desired_order)

        # Assign these new ordered factors to the Seurat object
        object <- Seurat::SetIdent(object, value = new_idents)
    }
    # if the group.by is "timepoint", then we'll re-order them
    if (group.by=="global_annotation"){
        # First, define your desired order
        desired_order <- c("Epidermal","Differentiating_Neurons","Neural_Crest",
                           "Neural_Anterior","Neural_Posterior",
                           "NMPs","PSM","Somites","Muscle","Lateral_Mesoderm",
                           "Notochord","Endoderm","Adaxial_Cells","unassigned")

        # Create a factor with levels in the order you want
        new_idents <- factor(object$global_annotation, levels = desired_order)

        # Assign these new ordered factors to the Seurat object
        object <- Seurat::SetIdent(object, value = new_idents)
    }

    
    # Step 3. peak profile for the counts (grouped.by "group.by")
    cov_plot <- CoveragePlot(
        object = object, 
        region = gene,
        extend.downstream = extend.down, extend.upstream = extend.up,
        annotation = FALSE,
        peaks=FALSE, ymax=ymax, region.highlight=region.highlight,
    )

    
    # for gene/peak plots, we need to find the genomic locations as the old Signac doesn't take the gene name as an input argument.
    gene.coord <- LookupGeneCoords(object = object, gene = gene)
    gene.coord.df <- as.data.frame(gene.coord)
    
    # extract the chromosome number, start position and end position
    chromosome <- gene.coord.df$seqnames
    pos_start <- gene.coord.df$start
    pos_end <-gene.coord.df$end
    
    # compute the genomic region as "chromsome_number-start-end"
    genomic_region <- paste(chromosome, pos_start, pos_end, sep="-")
    
    # gene annotation
    gene_plot <- AnnotationPlot(
      object = object,
      region = genomic_region,
      extend.downstream = extend.down, extend.upstream=extend.up,
    )
    # gene_plot


    # "peaks_integrated" peaks
    peak_plot <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$peaks_integrated@ranges,
      extend.downstream = extend.down, extend.upstream=extend.up,
    ) + labs(y="peaks")


    # expression of RNA
    expr_plot <- ExpressionPlot(
      object = object,
      features = gene,
      assay = "RNA"
    )
    
    # add the viridis colormap if plot_viridis==TRUE
    if (plot_viridis){
        cov_plot <- cov_plot + scale_fill_viridis_d()
        expr_plot <- expr_plot + scale_fill_viridis_d()
    }

    plot<-CombineTracks(
      plotlist = list(cov_plot, peak_plot, gene_plot),
      expression.plot = expr_plot,
      heights = c(10,1,2),
      widths = c(10, 1)
    )
    
    # options(repr.plot.width = 8, repr.plot.height = 12, repr.plot.res = 300)
#     ggsave(paste0(filepath, "coverage_plot_", gene, "_allpeaks.png"), plot=plot, width=8, height=12)
    return(plot)
    
}

# %%
# test the function for "timepoint"
generate_coverage_plots(multiome, gene="her1", group.by = "timepoint", plot_viridis = TRUE)

# %%
# test the function for "global_annotation"
generate_coverage_plots(multiome, gene="her1", group.by = "global_annotation", plot_viridis = FALSE)

# %%
# create a Coverage plot for "myf5", grouped by timepoints
coverage_plot <- generate_coverage_plots(multiome, gene="myf5", group.by = "timepoint", plot_viridis = TRUE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# %%
# create a Coverage plot for "myf5", grouped by timepoints
coverage_plot <- generate_coverage_plots(multiome, gene="myf5", group.by = "timepoint", plot_viridis = TRUE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text

# Open a PDF device
pdf("coverage_plot_myf5_timepoints.pdf", width = 4, height = 4)

# Draw the plot on the open device
coverage_plot

# Close the device to save the PDF file
dev.off()

# %%
# create a Coverage plot for "her1", grouped by timepoints
coverage_plot <- generate_coverage_plots(multiome, gene="her1", group.by = "timepoint", plot_viridis = TRUE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text

# Open a PDF device
pdf("coverage_plot_her1_timepoints.pdf", width = 4, height = 4)

# Draw the plot on the open device
coverage_plot

# Close the device to save the PDF file
dev.off()

# %%
# create a Coverage plot for "hbbe1.1", grouped by timepoints
coverage_plot <- generate_coverage_plots(multiome, gene="hbbe1.1", 
                                         extend.up = 2000, extend.down = 2000,
                                         group.by = "timepoint", plot_viridis = TRUE)

coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot
# # Open a PDF device
# pdf("coverage_plot_myf5_timepoints.pdf", width = 10, height = 6)

# # Draw the plot on the open device
# print(coverage_plot)

# # Close the device to save the PDF file
# dev.off()

# %%
# create a Coverage plot for "myf5", grouped by global_annotation
coverage_plot <- generate_coverage_plots(multiome, gene="myf5", group.by = "global_annotation", plot_viridis = FALSE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# %%
# Open a PDF device
pdf("coverage_plot_myf5_global_annotation.pdf", width = 4, height = 4)

# Draw the plot on the open device
coverage_plot

# Close the device to save the PDF file
dev.off()

# %% [markdown]
# ### extended regions for co-accessible peak visualization
#
# - 

# %%
# create a Coverage plot for "myf5", grouped by timepoints, extended for all CCANs
coverage_plot <- generate_coverage_plots(multiome, gene="myf5", 
                                         extend.up = 110000, extend.down = 2000,
                                         group.by = "timepoint", plot_viridis = TRUE, ymax=3200)
# format the ggplot object
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# Open a PDF device
pdf("coverage_plot_myf5_timepoints_CCANs_extended.pdf", width = 8, height =6)

# Draw the plot on the open device
coverage_plot

# Close the device to save the PDF file
dev.off()

# %%
# highlight the co-accessible peaks manually
peaks_TSS_mapped_0somites <- read.csv(file = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/02_cicero_processed/TDR126_cicero/03_TDR126_processed_peak_file_danRer11.csv", row.names = 1)
peaks_TSS_mapped_0somites %>% head

peaks_TSS_mapped_5somites <- read.csv(file = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/02_cicero_processed/TDR127_cicero/03_TDR127_processed_peak_file_danRer11.csv", row.names = 1)
peaks_TSS_mapped_5somites %>% head

# %%
peaks_TSS_mapped_0somites[peaks_TSS_mapped_0somites$gene_short_name=="myf5",]

peaks_TSS_mapped_5somites[peaks_TSS_mapped_5somites$gene_short_name=="myf5",]

# %%
# highlight the highly co-accessible peaks from 5 somites stage
peaks_to_highlight <- peaks_TSS_mapped_5somites[peaks_TSS_mapped_5somites$gene_short_name=="myf5",]$peak_id
peaks_to_highlight

ranges.show <- StringToGRanges(peaks_to_highlight, sep = c("_","_"))
ranges.show


# %%
# create a Coverage plot for "myf5", grouped by timepoints
coverage_plot <- generate_coverage_plots(multiome, gene="myf5", 
                                         extend.up = 110000, extend.down = 2000,
                                         group.by = "timepoint", plot_viridis = TRUE, ymax=3200, region.highlight=ranges.show)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot


# %%
# create a Coverage plot for "myf5", grouped by timepoints
coverage_plot <- generate_coverage_plots(multiome, gene="myf5", 
                                         extend.up = 110000, extend.down = 2000,
                                         group.by = "timepoint", plot_viridis = TRUE, ymax=3200, region.highlight=ranges.show)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text

# Open a PDF device
pdf("coverage_plot_myf5_timepoints_CCANs_marked_extended.pdf", width = 4, height = 3)

# Draw the plot on the open device
coverage_plot

# Close the device to save the PDF file
dev.off()

# %%

# %%
1

# %%

# %%

# %% [markdown]
# ## Step 3. Subset for each timepoint, grouped by celltypes
#
# - subset the multiome object for each timepoint, then resolve the chromatin accessibility grouped.by celltypes (global_annotation)

# %%
# write a for loop for this
list_timepoints <- unique(multiome$timepoint)
list_timepoints

# %%
# subsetted_objects_timepoints <- 
# for (timepoint in list_timepoints) {
    
# }

# %%
# subset for 0 somites
somites_0 <- subset(multiome, subset=timepoint=="0 somites")

# %%
somites_0

# %%
# subset for 15 somites
somites_15 <- subset(multiome, subset=timepoint=="15 somites")
somites_15

# %%
# subset for 30 somites
somites_30 <- subset(multiome, subset=timepoint=="30 somites")
somites_30

# %%
# create a Coverage plot for "myf5", grouped by global_annotation
coverage_plot <- generate_coverage_plots(somites_0, gene="myf5", group.by = "global_annotation", 
                                         extend.down = 1000, extend.up = 1000, plot_viridis = FALSE)
coverage_plot <- coverage_plot + theme(text = element_text(size = 8),  # For overall text elements
                                       axis.title = element_text(size = 12),  # For axis titles
                                       axis.text = element_text(size = 8),  # For axis text
                                       legend.title = element_text(size = 8),  # For legend title
                                       legend.text = element_text(size = 8))  # For legend text
coverage_plot

# Open a PDF device
pdf("coverage_plot_0somites_celltypes_myf5.pdf", width = 6, height = 6)

# Draw the plot on the open device
coverage_plot

# Close the device to save the PDF file
dev.off()

# %%



# %%
# genereate a coverage plot grouped.by "cell-types"
Idents(somites_15) <- "global_annotation"

# coverage plot
coverage_plot <- CoveragePlot(somites_15, region = "myf5", extend.downstream = 1000, extend.upstream = 1000, expression.assay = "RNA")

# Open a PDF device
pdf("coverage_plot_15somites_celltypes_myf5.pdf", width = 10, height = 6)

# Draw the plot on the open device
print(coverage_plot)

# Close the device to save the PDF file
dev.off()

# %%
# genereate a coverage plot grouped.by "cell-types"
Idents(somites_15) <- "global_annotation"

# coverage plot
coverage_plot <- CoveragePlot(somites_15, region = "hbbe1.1", extend.downstream = 1000, extend.upstream = 1000, expression.assay = "RNA")

# Open a PDF device
pdf("coverage_plot_15somites_celltypes_hbbe1_1.pdf", width = 10, height = 6)

# Draw the plot on the open device
print(coverage_plot)

# Close the device to save the PDF file
dev.off()

# %%

# %%

# %%

# %% [markdown]
# ## Step 4. Subset for each celltype, grouped by timepoints
#
# - We will subset for each celltype ("global_annotation"), then check the ATAC dynamics grouped.by timepoints.
#
#

# %%
list_celltypes <- multiome$global_annotation %>% unique %>% c()
list_celltypes 

# %%
# First, define your desired order
desired_order <- c("0 somites", "5 somites", "10 somites", "15 somites", "20 somites", "30 somites")

# Create a factor with levels in the order you want
new_idents <- factor(multiome$timepoint, levels = desired_order)

# Assign these new ordered factors to the Seurat object
multiome <- Seurat::SetIdent(multiome, value = new_idents)

# %%
# subset the multiome object for each celltype
# Initialize an empty list to store the subsetted Seurat objects
subsetted_objects_celltypes <- list()

for (celltype in list_celltypes){
    # check the celltype
    print(celltype)
    # subset the multiome for that celltype
    subsetted_objects_celltypes[[celltype]] <- subset(multiome, subset=global_annotation ==celltype)
    
#     # Set timepoint as the identity for the subset
#     Idents(subsetted_objects_celltypes[[celltype]]) <- subsetted_objects_celltypes[[celltype]]$timepoint
}

# %%
# DimPlot(multiome, dims=c(1,2), group.by="global_annotation")
# P1 <- DimPlot(multiome, dims=c(1,2), group.by="nCount_ATAC")
# P2 <- DimPlot(multiome, dims=c(1,2), group.by="nCount_RNA")

# %%
# # check where the "unassigned" cells are in the global UMAP
# unassigned <- subset(multiome, subset=global_annotation=="unassigned")

# P1 <- DimPlot(unassigned, dims=c(1,2), group.by="global_annotation")
# # P2 <- DimPlot(unassigned, dims=c(1,2), group.by="nCount_ATAC")
# # P3 <- DimPlot(unassigned, dims=c(1,2), group.by="nCount_RNA")

# %%
# subset for 15 somites
NMPs <- subset(multiome, subset=global_annotation=="NMPs")
NMPs

# %%
# First, define your desired order
desired_order <- c("0 somites", "5 somites", "10 somites", "15 somites", "20 somites", "30 somites")

# Assuming multiome$dataset is a separate vector parallel to the identity classes in Seurat object
# Create a factor with levels in the order you want
new_idents <- factor(NMPs$timepoint, levels = desired_order)

# Assign these new ordered factors to the Seurat object
NMPs <- Seurat::SetIdent(NMPs, value = new_idents)

# %%
coverage_plot <- generate_coverage_plots(NMPs, gene="tbxta", 
                                         extend.up = 2000, extend.down = 2000,
                                         group.by = "timepoint", plot_viridis = TRUE)
coverage_plot

# %%
# subset for PSM
PSM <- subset(multiome, subset=global_annotation=="PSM")
PSM

# %%
# First, define your desired order
desired_order <- c("0 somites", "5 somites", "10 somites", "15 somites", "20 somites", "30 somites")

# Assuming multiome$dataset is a separate vector parallel to the identity classes in Seurat object
# Create a factor with levels in the order you want
new_idents <- factor(PSM$timepoint, levels = desired_order)

# Assign these new ordered factors to the Seurat object
PSM <- Seurat::SetIdent(PSM, value = new_idents)

# %%
coverage_plot <- generate_coverage_plots(PSM, gene="myf5", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
coverage_plot

# %%
coverage_plot <- generate_coverage_plots(PSM, gene="her1", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
coverage_plot

# %%
coverage_plot <- generate_coverage_plots(PSM, gene="myf5", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)

# Open a PDF device
pdf("coverage_plot_PSM_myf5.pdf", width = 10, height = 6)

# Draw the plot on the open device
print(coverage_plot)

# Close the device to save the PDF file
dev.off()

# %%
coverage_plot <- generate_coverage_plots(PSM, gene="her1", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
# Open a PDF device
pdf("coverage_plot_PSM_her1.pdf", width = 10, height = 6)

# Draw the plot on the open device
print(coverage_plot)

# Close the device to save the PDF file
dev.off()

# %%
multiome$global_annotation %>% unique

# %%
# subset for Differentiating_Neurons
Differentiating_Neurons <- subset(multiome, subset=global_annotation=="Differentiating_Neurons")
Differentiating_Neurons

# %%
# First, define your desired order
desired_order <- c("0 somites", "5 somites", "10 somites", "15 somites", "20 somites", "30 somites")

# Assuming multiome$dataset is a separate vector parallel to the identity classes in Seurat object
# Create a factor with levels in the order you want
new_idents <- factor(Differentiating_Neurons$timepoint, levels = desired_order)

# Assign these new ordered factors to the Seurat object
Differentiating_Neurons <- Seurat::SetIdent(Differentiating_Neurons, value = new_idents)

# %%
coverage_plot <- generate_coverage_plots(Differentiating_Neurons, gene="myf5", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
coverage_plot

# %%
# the same coverageplot but with the y_max from the "PSM" to scale things together
coverage_plot <- generate_coverage_plots(Differentiating_Neurons, gene="myf5", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE, ymax=780)
coverage_plot

# %%
coverage_plot <- generate_coverage_plots(Differentiating_Neurons, gene="her1", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
coverage_plot

# %%
# Differentiating_Neuron, "myf5"
coverage_plot <- generate_coverage_plots(Differentiating_Neurons, gene="myf5", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
# Open a PDF device
pdf("coverage_plot_Differentiating_Neurons_myf5.pdf", width = 10, height = 6)

# Draw the plot on the open device
print(coverage_plot)

# Close the device to save the PDF file
dev.off()

# %%
# Differentiating_Neuron, "myf5" (with y_max=max(Peaks|PSM))
coverage_plot <- generate_coverage_plots(Differentiating_Neurons, gene="myf5", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE, ymax=780)
# Open a PDF device
pdf("coverage_plot_Differentiating_Neurons_myf5_ymax_PSM.pdf", width = 10, height = 6)

# Draw the plot on the open device
print(coverage_plot)

# Close the device to save the PDF file
dev.off()

# %%

# %% [markdown]
# ### Muscle

# %%
# subset forMuscle
Muscle <- subset(multiome, subset=global_annotation=="Muscle")
Muscle

# %%
# First, define your desired order
desired_order <- c("0 somites", "5 somites", "10 somites", "15 somites", "20 somites", "30 somites")

# Assuming multiome$dataset is a separate vector parallel to the identity classes in Seurat object
# Create a factor with levels in the order you want
new_idents <- factor(Muscle$timepoint, levels = desired_order)

# Assign these new ordered factors to the Seurat object
Differentiating_Neurons <- Seurat::SetIdent(Muscle, value = new_idents)

# %%
coverage_plot <- generate_coverage_plots(Muscle, gene="myf5", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
coverage_plot

# %%
coverage_plot <- generate_coverage_plots(Muscle, gene="her1", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
coverage_plot

# %%
# Muscle, "myf5"
coverage_plot <- generate_coverage_plots(Muscle, gene="myf5", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
# Open a PDF device
pdf("coverage_plot_Muscle_myf5.pdf", width = 10, height = 6)

# Draw the plot on the open device
print(coverage_plot)

# Close the device to save the PDF file
dev.off()

# %%
# Muscle, "her1"
coverage_plot <- generate_coverage_plots(Muscle, gene="her1", 
                                         extend.up = 1000, extend.down = 1000,
                                         group.by = "timepoint", plot_viridis = TRUE)
# Open a PDF device
pdf("coverage_plot_Muscle_her1.pdf", width = 10, height = 6)

# Draw the plot on the open device
print(coverage_plot)

# Close the device to save the PDF file
dev.off()

# %%

# %%

# %%

# %%

# %% [markdown]
# ## generate Coverage Plots for a list of marker genes (grouped.by timepoints)

# %%
# a list of marker genes used to annotate cell-types in 15-somites stage (16hpf) - from Merlin
list_genes <- list("lrrc17","comp","ripply1","rx1","vsx2","tbx16","myf5",
                   "hes6","crestin","ednrab","dlx2a","cldni","cfl1l",
                   "fezf1","sox1b","foxg1a","olig3","hoxd4a","rxrga",
                   "gata5","myh7","tnnt2a",'pkd1b',"scg3","etv5a","pitx3",
                   "elavl3","stmn1b","sncb","myog","myl1","jam2a",
                   "prrx1","nid1b","cpox","gata1a","hbbe1","unc45b","ttn1",
                   "apobec2a","foxi3b","atp1b1b","fli1b","kdrl","anxa4",
                   "cldnc","cldn15a","tbx3b","loxl5b","emilin3a","sema3aa","irx7","vegfaa",
                   "ppl","krt17","icn2","osr1","hand2","shha","shhb","foxa2",
                   "cebpa","spi1b","myb","ctslb","surf4l","sec61a1l","mcf2lb",
                   "bricd5","etnk1","chd17","acy3")

list_genes <- unique(list_genes)

# %%
# Create a list to store the plot objects
plot_list <- list()

# Loop over 20 genes
for (gene in list_genes) {
  # Check if gene exists in GTF file
  if (!gene %in% object@assays$peaks_integrated@annotation$gene_name) {
    cat("Gene", gene, "not found in GTF file. Skipping.\n")
    return(NULL)
  }
  # Generate the coverage plot for the gene
  plot <- CoveragePlot(multiome, region=gene, extend.downstream = 1000, extend.upstream = 1000)
  
  # Add the plot object to the list
  plot_list[[gene]] <- plot
}

# Create a PDF file
pdf("coverage_plots_timepoints_marker_genes_ML.pdf")

# Loop over the plot list and save each plot to a separate page in the PDF
for (gene in list_genes) {
  plot <- plot_list[[gene]]
  print(plot)
}

# Close the PDF file
dev.off()

# %%
gene.coord <- LookupGeneCoords(object = multiome, gene = "myf5")
gene.coord

gene.coord.df <- as.data.frame(gene.coord)
gene.coord.df

chromosome <- gene.coord.df$seqnames
pos_start <- gene.coord.df$start
pos_end <-gene.coord.df$end

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
options(repr.plot.width = 8, repr.plot.height = 12, repr.plot.res = 300)

# generate CoveragePlot using "group.by" parameter
generate_coverage_plots <- function(object = multiome, 
                                    gene=gene, 
                                    group.by="dataset", 
                                    filepath=NULL){
    
      # Check if gene exists in GTF file
      if (!gene %in% object@assays$peaks_integrated@annotation$gene_name) {
        cat("Gene", gene, "not found in GTF file. Skipping.\n")
        return(NULL)
      }
    
#     # make sure that the major identity is "orig.ident" for bulk peak profile
#     Idents(object) <- "orig.ident"
#     # peak profile for the bulk counts
#     cov_plot_bulk <- CoveragePlot(
#       object = object,
#       region = gene,
#       annotation=FALSE,
#       peaks=FALSE
#       #ranges = peaks,
#       #ranges.title = "MACS2"
#     )

    # we have to manually change the basic identity for Seurat
    Idents(object) <- group.by
    
    # peak profile for the counts (cell-type, predicted.id)
    cov_plot_celltype <- CoveragePlot(
        object = object, 
        region = gene,
        annotation = FALSE,
        peaks=FALSE
    )
    
    # for gene/peak plots, we need to find the genomic locations as the old Signac doesn't take the gene name as an input argument.
    gene.coord <- LookupGeneCoords(object = object, gene = gene)
    gene.coord.df <- as.data.frame(gene.coord)
    
    # extract the chromosome number, start position and end position
    chromosome <- gene.coord.df$seqnames
    pos_start <- gene.coord.df$start
    pos_end <-gene.coord.df$end
    
    # compute the genomic region as "chromsome_number-start-end"
    genomic_region <- paste(chromosome, pos_start, pos_end, sep="-")
    
    # gene annotation
    gene_plot <- AnnotationPlot(
      object = object,
      region = genomic_region
    )
    # gene_plot


    # cellranger-arc peaks
    peak_plot_CRG <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$ATAC@ranges
    ) + labs(y="CRG-arc")
    # peak_plot

    # MACS2-bulk peaks
    peak_plot_bulk <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$peaks_bulk@ranges,
      color = "blue"
    )+ labs(y="bulk")

    # MACS2-cell-type-specific peaks
    peak_plot_celltype <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$peaks_celltype@ranges,
      color = "red"
    )+ labs(y="cell-type")

    # expression of RNA
    expr_plot <- ExpressionPlot(
      object = object,
      features = gene,
      assay = "RNA"
    )

    plot<-CombineTracks(
      plotlist = list(cov_plot_celltype, cov_plot_bulk, 
                      peak_plot_CRG, peak_plot_bulk, peak_plot_celltype, 
                      gene_plot),
      expression.plot = expr_plot,
      heights = c(10,3,1,1,1,2),
      widths = c(10, 1)
    )
    
    options(repr.plot.width = 8, repr.plot.height = 12, repr.plot.res = 300)
#     ggsave(paste0(filepath, "coverage_plot_", gene, "_allpeaks.png"), plot=plot, width=8, height=12)
    return(plot)
    
}

# 

# %%
generate_coverage_plots(object = TDR118, gene = "tbxta", 
                        filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/coverage_plots_TDR118/")

# %%

# %%
gene.coord <- LookupGeneCoords(object = TDR118, gene = "tbxta")
gene.coord

gene.coord.df <- as.data.frame(gene.coord)
gene.coord.df

# %%
gene.coord <- LookupGeneCoords(object = TDR118, gene = "tbxta")
gene.coord

gene.coord.df <- as.data.frame(gene.coord)
gene.coord.df

chromosome <- gene.coord.df$seqnames
pos_start <- gene.coord.df$start
pos_end <-gene.coord.df$end

# %%
c(chromosome,"-",pos_start,"-",pos_end)

# %%

# %%
gene = "tbxta"

# Compare the called peaks using a Coverage Plot
Idents(TDR118) <- "orig.ident"

cov_plot_bulk <- CoveragePlot(
  object = TDR118,
  region = "tbxta",
  annotation=FALSE,
  peaks=FALSE
  #ranges = peaks,
  #ranges.title = "MACS2"
)

# we have to manually change the basic identity for Seurat
Idents(TDR118) <- "predicted.id"
cov_plot_celltype <- CoveragePlot(
    object = TDR118, 
    region = "tbxta",
    annotation = FALSE,
    peaks=FALSE
)

# gene annotation
gene_plot <- AnnotationPlot(
  object = TDR118,
  region = "19-14187540-14191592"
)
# gene_plot


# cellranger-arc peaks
peak_plot_CRG <- PeakPlot(
  object = TDR118,
  region = "19-14187540-14191592",
  peaks=TDR118@assays$ATAC@ranges
) + labs(y="CRG-arc")
# peak_plot

# MACS2-bulk peaks
peak_plot_bulk <- PeakPlot(
  object = TDR118,
  region = "19-14187540-14191592",
  peaks=TDR118@assays$peaks_bulk@ranges,
  color = "blue"
)+ labs(y="bulk")

# MACS2-cell-type-specific peaks
peak_plot_celltype <- PeakPlot(
  object = TDR118,
  region = "19-14187540-14191592",
  peaks=TDR118@assays$peaks_celltype@ranges,
  color = "red"
)+ labs(y="cell-type")

# expression of RNA
expr_plot <- ExpressionPlot(
  object = TDR118,
  features = "tbxta",
  assay = "RNA"
)

# Patchwork
# cov_plot/
# (cov_plot_celltype | expr_plot)/
# gene_plot/
# peak_plot_CRG/
# peak_plot_bulk/
# peak_plot_celltype

CombineTracks(
  plotlist = list(cov_plot_celltype, cov_plot_bulk, 
                  peak_plot_CRG, peak_plot_bulk, peak_plot_celltype, 
                  gene_plot),
  expression.plot = expr_plot,
  heights = c(10,3,1,1,1,2),
  widths = c(10, 1)
)

# %%
options(repr.plot.width = 8, repr.plot.height = 12, repr.plot.res = 300)

generate_coverage_plots <- function(object, gene, filepath){
    
      # Check if gene exists in GTF file
      if (!gene %in% object@assays$ATAC@annotation$gene_name) {
        cat("Gene", gene, "not found in GTF file. Skipping.\n")
        return(NULL)
      }
    
    # make sure that the major identity is "orig.ident" for bulk peak profile
    Idents(object) <- "orig.ident"
    # peak profile for the bulk counts
    cov_plot_bulk <- CoveragePlot(
      object = object,
      region = gene,
      annotation=FALSE,
      peaks=FALSE
      #ranges = peaks,
      #ranges.title = "MACS2"
    )

    # we have to manually change the basic identity for Seurat
    Idents(object) <- "predicted.id"
    
    # peak profile for the counts (cell-type, predicted.id)
    cov_plot_celltype <- CoveragePlot(
        object = object, 
        region = gene,
        annotation = FALSE,
        peaks=FALSE
    )
    
    # for gene/peak plots, we need to find the genomic locations as the old Signac doesn't take the gene name as an input argument.
    gene.coord <- LookupGeneCoords(object = object, gene = gene)
    gene.coord.df <- as.data.frame(gene.coord)
    
    # extract the chromosome number, start position and end position
    chromosome <- gene.coord.df$seqnames
    pos_start <- gene.coord.df$start
    pos_end <-gene.coord.df$end
    
    # compute the genomic region as "chromsome_number-start-end"
    genomic_region <- paste(chromosome, pos_start, pos_end, sep="-")
    
    # gene annotation
    gene_plot <- AnnotationPlot(
      object = object,
      region = genomic_region
    )
    # gene_plot


    # cellranger-arc peaks
    peak_plot_CRG <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$ATAC@ranges
    ) + labs(y="CRG-arc")
    # peak_plot

    # MACS2-bulk peaks
    peak_plot_bulk <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$peaks_bulk@ranges,
      color = "blue"
    )+ labs(y="bulk")

    # MACS2-cell-type-specific peaks
    peak_plot_celltype <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$peaks_celltype@ranges,
      color = "red"
    )+ labs(y="cell-type")

    # expression of RNA
    expr_plot <- ExpressionPlot(
      object = object,
      features = gene,
      assay = "RNA"
    )

    plot<-CombineTracks(
      plotlist = list(cov_plot_celltype, cov_plot_bulk, 
                      peak_plot_CRG, peak_plot_bulk, peak_plot_celltype, 
                      gene_plot),
      expression.plot = expr_plot,
      heights = c(10,3,1,1,1,2),
      widths = c(10, 1)
    )
    
    options(repr.plot.width = 8, repr.plot.height = 12, repr.plot.res = 300)
#     ggsave(paste0(filepath, "coverage_plot_", gene, "_allpeaks.png"), plot=plot, width=8, height=12)
    return(plot)
    
}

# 

# %%
generate_coverage_plots(object = TDR118, gene = "tbxta", 
                        filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/coverage_plots_TDR118/")

# %%
list_genes <- list("lrrc17","comp","ripply1","rx1","vsx2","tbx16","myf5",
                   "hes6","crestin","ednrab","dlx2a","cldni","cfl1l",
                   "fezf1","sox1b","foxg1a","olig3","hoxd4a","rxrga",
                   "gata5","myh7","tnnt2a",'pkd1b',"scg3","etv5a","pitx3",
                   "elavl3","stmn1b","sncb","myog","myl1","jam2a",
                   "prrx1","nid1b","cpox","gata1a","hbbe1","unc45b","ttn1",
                   "apobec2a","foxi3b","atp1b1b","fli1b","kdrl","anxa4",
                   "cldnc","cldn15a","tbx3b","loxl5b","emilin3a","sema3aa","irx7","vegfaa",
                   "ppl","krt17","icn2","osr1","hand2","shha","shhb","foxa2",
                   "cebpa","spi1b","myb","ctslb","surf4l","sec61a1l","mcf2lb",
                   "bricd5","etnk1","chd17","acy3")
list_genes

# %%
list_genes <- unique(list_genes)
list_genes

# %%
# Create a list to store the plot objects
plot_list <- list()

# Loop over 20 genes
for (gene in list_genes) {
  # Generate the coverage plot for the gene
  plot <- generate_coverage_plots(TDR118, gene)
  
  # Add the plot object to the list
  plot_list[[gene]] <- plot
}

# Create a PDF file
pdf("coverage_plots_15somite_marker_genes.pdf")

# Loop over the plot list and save each plot to a separate page in the PDF
for (gene in list_genes) {
  plot <- plot_list[[gene]]
  print(plot)
}

# Close the PDF file
dev.off()

# %%
plot_list[1]

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## (Optional)
#
# - Check the regions with negative cicero co-accessibility scores
#

# %%
# import the cicero connections
cicero_connections <- read.csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/02_TDR118_cicero_connections_CRG_arc_peaks.csv")
head(cicero_connections)

# %%
sorted_cicero_connections <- cicero_connections[order(cicero_connections$coaccess, decreasing = FALSE), ]
head(sorted_cicero_connections)

# %%
sorted_cicero_connections[sorted_cicero_connections$coaccess < -0.6, ]

# %%
library(cicero)

# %%
ccans <- generate_ccans(cicero_connections)b

# %%
region <- sorted_cicero_connections[1, ]$Peak1
region

# # Input string
# input_string <- '24-17160375-17161077'

# %%
plot <- generate_coverage_plots(TDR118, "mllt10")
plot

# %%

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Some hypotheses on the pairs of peaks with highly negative co-accessibilities
#
# - one of the examples, mllt10 has two peaks on the intron regions that are highly negatively correlated (coaccess <-0.6). This might suggest that the pairs of peaks with strong negative co-accessibility might mean that they are within the same gene/transcriptional unit, thus they are not TSS and silencers, rather reporting different transcriptional "states".
#
# - Additional things to check: 
#
#     - whether the highly negatively co-accessible peaks are within the same gene (fraction of peak pairs that were from the same gene).
#     - If not, whether they are worth digging into.

# %%

# %%

# %%
# for gene/peak plots, we need to find the genomic locations as the old Signac doesn't take the gene name as an input argument.
# gene.coord <- LookupGeneCoords(object = object, gene = gene)
# gene.coord.df <- as.data.frame(gene.coord)

cicero_index = 1
region1 = sorted_cicero_connections[cicero_index, ]$Peak1
region2 = sorted_cicero_connections[cicero_index, ]$Peak2
#24-17160375-17161077	24-17168001-17168928

# # extract the chromosome number, start position and end position
# chromosome <- gene.coord.df$seqnames
# pos_start <- gene.coord.df$start
# pos_end <-gene.coord.df$end

# # compute the genomic region as "chromsome_number-start-end"
# genomic_region <- paste(chromosome, pos_start, pos_end, sep="-")

# gene annotation (peak1)
gene_plot1 <- AnnotationPlot(
  object = TDR118,
  region = region1
)
# gene annotation (peak2)
gene_plot2 <- AnnotationPlot(
  object = TDR118,
  region = region2
)



# cellranger-arc peaks
peak_plot_CRG <- PeakPlot(
  object = TDR118,
  region = "24-17160000-17169000",
  peaks=TDR118@assays$ATAC@ranges
) + labs(y="CRG-arc")
# peak_plot


gene_plot1/gene_plot2/peak_plot_CRG



# %% [markdown]
# ## (Deprecated) Hacking the Coverage Plot function

# %%
as.data.frame(TDR118@assays$ATAC@fragments[[1]])

# %%
fragments <- TDR118@assays$ATAC@fragments[[1]]
fragments

# %%
GetFragmentData(fragments, slot="path")

# %%
head(TDR118@assays$ATAC@fragments[[1]])

# %%
head(fragments)

# %%
class(fragments)
str(fragments)

# %%
frag_df <- data.frame(
  chrom = fragments@chrom,
  start = fragments@start,
  end = fragments@end,
  barcode = fragments@cells,
  readCount = fragments@readCount
)

head(frag_df)

# %%
frag_df <- as.data.frame(fragments)

head(frag_df[, c("chrom", "start", "end", "barcode", "readCount")])

# %%
frag_df <- data.frame(
  chrom = fragments@ranges@seqnames,
  start = start(fragments@ranges),
  end = end(fragments@ranges),
  barcode = fragments@cells,
  readCount = fragments@readCount
)

head(frag_df)

# %%
frag_df <- data.frame(
  chrom = fragments@regions@seqnames,
  start = start(fragments@regions),
  end = end(fragments@regions),
  barcode = fragments@cells,
  readCount = fragments@readCount
)

head(frag_df)
