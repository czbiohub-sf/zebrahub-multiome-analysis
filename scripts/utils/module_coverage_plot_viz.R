# import the libraries
library(Seurat)
library(Signac)
library(patchwork)
library(ggplot2)

# example list of genes (marker genes used for 15somite stage):
# list_genes <- list("lrrc17","comp","ripply1","rx1","vsx2","tbx16","myf5",
#                  "hes6","crestin","ednrab","dlx2a","cldni","cfl1l",
#                   "fezf1","sox1b","foxg1a","olig3","hoxd4a","rxrga",
#                   "gata5","myh7","tnnt2a",'pkd1b',"scg3","etv5a","pitx3",
#                   "elavl3","stmn1b","sncb","myog","myl1","jam2a",
#                   "prrx1","nid1b","cpox","gata1a","hbbe1","unc45b","ttn1",
#                   "apobec2a","foxi3b","atp1b1b","fli1b","kdrl","anxa4",
#                   "cldnc","cldn15a","tbx3b","loxl5b","emilin3a","sema3aa","irx7","vegfaa",
#                   "ppl","krt17","icn2","osr1","hand2","shha","shhb","foxa2",
#                   "cebpa","spi1b","myb","ctslb","surf4l","sec61a1l","mcf2lb",
#                   "bricd5","etnk1","chd17","acy3")

#' Make a Series of Coverage Plots for Genes
#'
#' This function generates coverage plots for a given list of genes from a Seurat object and saves them as a single PDF file.
#' Each gene's coverage plot is saved on a separate page.
#'
#' @param object The file path for the RDS file containing a preprocessed Seurat object.
#' @param list_genes A vector of gene names for which coverage plots will be generated.
#' @param output_path The file path where the PDF will be saved.
#' @return Invisible null. The function is called for its side effect, which is creating a PDF file.
#' @examples
#' seurat_obj_path <- "path/to/your/seurat_object.rds"
#' genes_to_plot <- c("gene1", "gene2", "gene3")
#' output_pdf_path <- "path/to/output/"
#' make_coverage_plots(object = seurat_obj_path, list_genes = genes_to_plot, output_path = output_pdf_path)
#' @export
#'
make_coverage_plots <- function(object=object,
                                list_genes = c(), 
                                output_path = ""){
    # object <- readRDS(object)
    # Create a list to store the plot objects
    plot_list <- list()

    # Loop over 20 genes
    for (gene in list_genes) {
        # Generate the coverage plot for the gene
        plot <- coverage_plot(object, gene)
        
        # Add the plot object to the list
        plot_list[[gene]] <- plot
    }

    # Create a PDF file
    output_filepath <- paste0(output_path, "coverage_plots_marker_genes.pdf")
    pdf(output_filepath)

    # Loop over the plot list and save each plot to a separate page in the PDF
    for (gene in list_genes) {
    plot <- plot_list[[gene]]
    print(plot)
    }

    # Close the PDF file
    dev.off()

}


#' Generate a Coverage Plot for a Single Gene
#'
#' A sub-function used by `make_coverage_plots` to generate coverage plots for a single gene.
#' It uses a Seurat object to create coverage plots including bulk and cell-type-specific peak profiles.
#'
#' @param object A Seurat object.
#' @param gene The name of the gene to generate the plot for.
#' @param annotation_class A character vector for annotation class.
#' @param peak_profiles A character vector for peak profiles.
#' @param genomic_region Genomic region in "chrX-start-end" format, or NULL if not provided.
#' @param genomic_region_input Logical indicating whether genomic_region is provided.
#' @param filepath The file path where the plot will be saved.
#' @return A ggplot object representing the coverage plot for the specified gene or NULL if the gene is not found.
#' @examples
#' # Assume `object` is a preloaded Seurat object with necessary data
#' coverage_plot(object, "gene_name")
#' @export
#'
# a sub-function to generate a Coverage Plot (for one gene)
coverage_plot <- function(object, gene, 
                            annotation_class = "global_annotation",
                            peak_profiles = c("peak"),
                            genomic_region_input=FALSE,
                            genomic_region=NULL, filepath=NULL){
      # Check if gene exists in GTF file
      if (!gene %in% object@assays$ATAC@annotation$gene_name) {
        cat("Gene", gene, "not found in GTF file. Skipping.\n")
        return(NULL)
      }
    
    # make sure that the major identity is "orig.ident" for bulk peak profile
    Idents(object) <- "orig.ident"
    # mapped-reads profile for the bulk counts
    cov_plot_bulk <- CoveragePlot(
      object = object,
      region = gene,
      annotation=FALSE,
      peaks=FALSE
      #ranges = peaks,
      #ranges.title = "MACS2"
    )

    # we have to manually change the basic identity for Seurat
    Idents(object) <- annotation_class
    
    # mapped-reads profile for the counts (cell-type, global_annotation)
    cov_plot_celltype <- CoveragePlot(
        object = object, 
        region = gene,
        annotation = FALSE,
        peaks=FALSE
    )
    
    # Define the genomic region (either given by the user or automatically extracted from the gene name)
    if (!genomic_region_input) {
      # Lookup genomic coordinates for the gene
      gene.coord <- LookupGeneCoords(object = object, gene = gene)
      gene.coord.df <- as.data.frame(gene.coord)
      
      # Extract chromosome number, start position, and end position
      chromosome <- gene.coord.df$seqnames
      pos_start <- gene.coord.df$start
      pos_end <- gene.coord.df$end
      
      # Compute the genomic region as "chromosome_number-start-end"
      genomic_region <- paste(chromosome, pos_start, pos_end, sep = "-")
    }

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
      color = "blue")+ labs(y="bulk")

    # MACS2-cell-type-specific peaks
    peak_plot_celltype <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$peaks_celltype@ranges,
      color = "red")+ labs(y="cell-type")

    # MACS2-cell-type-specific peaks
    peak_plot_merged <- PeakPlot(
      object = object,
      region = genomic_region,
      peaks=object@assays$peaks_merged@ranges,
      color = "purple")+ labs(y="merged")

    # expression of RNA
    expr_plot <- ExpressionPlot(
      object = object,
      features = gene,
      assay = "RNA"
    )

    plot<-CombineTracks(
      plotlist = list(cov_plot_celltype, cov_plot_bulk, 
                      peak_plot_CRG, peak_plot_bulk, 
                      peak_plot_celltype, peak_plot_merged, 
                      gene_plot),
      expression.plot = expr_plot,
      heights = c(10,3,1,1,1,1,2),
      widths = c(10, 1)
    )
    
    options(repr.plot.width = 8, repr.plot.height = 12, repr.plot.res = 300)
  # Save the plot if a filepath is provided
  if (!is.null(filepath)) {
    ggsave(paste0(filepath, "coverage_plot_", gene, "_allpeaks.pdf"), plot = plot, width = 8, height = 12)
  }    
  # return the combined plot
    return(plot)
}
