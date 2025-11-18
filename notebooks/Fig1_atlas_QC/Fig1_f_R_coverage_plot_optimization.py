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
# # Notebook for coverage plot optimization
#
# - We want to generate coverage plots from Signac, but with our own style.
#
# More specific goals are below:
# - Use/Modify Signac's default functions to create customized coverage plots for different peak-calling algorithms (This will be used for the peak-calling method SI figure)

# %%
library(Seurat)
library(Signac)
library(patchwork)
library(ggplot2)

# %%
# Load the plotting module
source("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/utils/module_coverage_plot_viz.R")

# %% [markdown]
# ### sample: TDR118

# %%
# import the Seurat object (TDR118, 15-somites stage)
TDR118 <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/TDR118_processed.RDS")
TDR118

# %%
TDR118@assays$ATAC@fragments[[1]]@path

# %%
# Making sure that the Fragment files' location match as the one in the Seurat object
# Update the filepath for the "Fragment" object within the Seurat object
# NOTE that we need to update it manually for every assay
TDR118@assays$ATAC@fragments[[1]]@path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/TDR118reseq/outs/atac_fragments.tsv.gz"
TDR118@assays$peaks_celltype@fragments[[1]]@path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/TDR118reseq/outs/atac_fragments.tsv.gz"
TDR118@assays$peaks_bulk@fragments[[1]]@path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/TDR118reseq/outs/atac_fragments.tsv.gz"
TDR118@assays$peaks_merged@fragments[[1]]@path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/TDR118reseq/outs/atac_fragments.tsv.gz"

# %%
CoveragePlot(TDR118, region="tbxta")

# %%
CoveragePlot(multiome, region="tbxta")

# %% [markdown]
# ### Note that there are four types of peak profiles
#
# - ATAC: peaks called by Cellranger-arc
# - peaks_bulk: peaks called by MACS2 (Signac) for all cells
# - peaks_celltype: peaks called by MACS2 (Signac) for each celltype (pseudo-bulk)
# - peaks_merged: all three peaks merged by itrative overlap strategy (illustrated in ArchR: https://www.archrproject.com/bookdown/the-iterative-overlap-peak-merging-procedure.html). Here, peaks_celltype > peaks_bulk > ATAC order of significance.

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_calling_benchmark/"
coverage_plot(TDR118, gene = "tbxta", 
              filepath = figpath)

# %%

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_calling_benchmark/"
figpath

# %% [markdown]
# ## generation of coverage plots for a list of genes

# %%
# Use the functions defined in the module (module_coverage_plot_viz.R)
# list_genes <- list("lrrc17","comp","ripply1","rx1","vsx2","tbx16","myf5","cdx4",
#                  "hes6","crestin","ednrab","dlx2a","cldni","cfl1l",
#                   "fezf1","sox1b","foxg1a","olig3","hoxd4a","rxrga",
#                   "gata5","myh7","tnnt2a",'pkd1b',"scg3","etv5a","pitx3",
#                   "elavl3","stmn1b","sncb","myog","myl1","jam2a",
#                   "prrx1","nid1b","cpox","gata1a","hbbe1","unc45b","ttn1",
#                   "apobec2a","foxi3b","atp1b1b","fli1b","kdrl","anxa4",
#                   "cldnc","cldn15a","tbx3b","loxl5b","emilin3a","sema3aa","irx7","vegfaa",
#                   "ppl","krt17","icn2","osr1","hand2","shha","shhb","foxa2",
#                   "cebpa","spi1b","myb","ctslb","surf4l","sec61a1l","mcf2lb",
#                   "bricd5","etnk1","chd17","acy3","meis1a","en2a","pax6a")
list_genes <- list("tbx16","myf5","cdx4","hes6","gata5","myh7",
                   "tnnt2a",'pkd1b',"scg3","etv5a","pitx3",
                   "gata1a","hbbe1","irx7","meis1a",
                   "en2a","pax6a","rarga","noto","rps18")
make_coverage_plots(TDR118, list_genes, output_path = figpath)

# %%

# %% [markdown]
# ### sample: integrated object
#
# - last updated: 02/04/2025

# %%
# import the Seurat object (TDR118, 15-somites stage)
seurat <- readRDS("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_wnn_gene_activity_3d_umaps.rds")
seurat

# %%
seurat@assays$peaks_integrated@fragments

# %%
seurat@assays$peaks_integrated@fragments[[1]]@path

# %%
# Making sure that the Fragment files' location match as the one in the Seurat object
# Update the filepath for the "Fragment" object within the Seurat object
# NOTE that we need to update it manually for every assay
# TDR118@assays$ATAC@fragments[[1]]@path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/TDR118reseq/outs/atac_fragments.tsv.gz"
# TDR118@assays$peaks_celltype@fragments[[1]]@path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/TDR118reseq/outs/atac_fragments.tsv.gz"
# TDR118@assays$peaks_bulk@fragments[[1]]@path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/TDR118reseq/outs/atac_fragments.tsv.gz"
# TDR118@assays$peaks_merged@fragments[[1]]@path <- "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/TDR118reseq/outs/atac_fragments.tsv.gz"

# %%
seurat

# %%
CoveragePlot(seurat, region="hoxc1a")

# %%
CoveragePlot(multiome, region="tbxta")

# %% [markdown]
# ### Note that there are four types of peak profiles
#
# - ATAC: peaks called by Cellranger-arc
# - peaks_bulk: peaks called by MACS2 (Signac) for all cells
# - peaks_celltype: peaks called by MACS2 (Signac) for each celltype (pseudo-bulk)
# - peaks_merged: all three peaks merged by itrative overlap strategy (illustrated in ArchR: https://www.archrproject.com/bookdown/the-iterative-overlap-peak-merging-procedure.html). Here, peaks_celltype > peaks_bulk > ATAC order of significance.

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_calling_benchmark/"
coverage_plot(TDR118, gene = "tbxta", 
              filepath = figpath)

# %%

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_calling_benchmark/"
figpath

# %% [markdown]
# ## generation of coverage plots for a list of genes

# %%
# Use the functions defined in the module (module_coverage_plot_viz.R)
# list_genes <- list("lrrc17","comp","ripply1","rx1","vsx2","tbx16","myf5","cdx4",
#                  "hes6","crestin","ednrab","dlx2a","cldni","cfl1l",
#                   "fezf1","sox1b","foxg1a","olig3","hoxd4a","rxrga",
#                   "gata5","myh7","tnnt2a",'pkd1b',"scg3","etv5a","pitx3",
#                   "elavl3","stmn1b","sncb","myog","myl1","jam2a",
#                   "prrx1","nid1b","cpox","gata1a","hbbe1","unc45b","ttn1",
#                   "apobec2a","foxi3b","atp1b1b","fli1b","kdrl","anxa4",
#                   "cldnc","cldn15a","tbx3b","loxl5b","emilin3a","sema3aa","irx7","vegfaa",
#                   "ppl","krt17","icn2","osr1","hand2","shha","shhb","foxa2",
#                   "cebpa","spi1b","myb","ctslb","surf4l","sec61a1l","mcf2lb",
#                   "bricd5","etnk1","chd17","acy3","meis1a","en2a","pax6a")
list_genes <- list("tbx16","myf5","cdx4","hes6","gata5","myh7",
                   "tnnt2a",'pkd1b',"scg3","etv5a","pitx3",
                   "gata1a","hbbe1","irx7","meis1a",
                   "en2a","pax6a","rarga","noto","rps18")
make_coverage_plots(TDR118, list_genes, output_path = figpath)

# %%
