# Utilities for basic QC pre-processing for single-cell multiome object

# This workflow is designed for processing one single-cell multiome dataset (one experiment)
# For integration of multiple "replicates", we will need additional steps.
# We recommend using RNA modalities to integrate multiple replicates, then perform joint peak-calling (either bulk, or cell-type specific)

# load the libraries
suppressMessages(library(Seurat))
suppressMessages(library(Seurat))
#library(Seurat)
#library(Signac)
library(SeuratData)
library(SeuratDisk)
# genome info
library(GenomeInfoDb)
library(ggplot2)
library(patchwork)
library(stringr)
library(BSgenome.Drerio.UCSC.danRer11)

print(R.version)
print(packageVersion("Seurat"))


# Input args
# raw_data_path: filepath for the cellranger-arc output files (h5 and fragment files)
# gref_path: filepath for the GTF file (used for Cellranger-arc alignment)
# 

# Output args
# Seurat: a seurat object containing multiple assays (i.e. RNA, ATAC, peaks_bulk, peaks_celltype, etc.)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6) {
  stop("Usage: Rscript bash_01_preprocess_multiome_object_signac.R raw_data_path gref_path reference annotation_class output_filepath data_id")
}

raw_data_path <- args[1]
gref_path <- args[2]
reference <- args[3]
annotation_class <- args[4] # Let's just use one annotation_class. (i.e. "global_annotation")
output_filepath <- args[5] # filepath for the output
data_id <- args[6]


# Sub-functions
# Step1. generate a Seurat object from the Cellranger-arc output (multiome)
generate_seurat_object <- function(raw_data_path ="" ,
                                    gref_path ="")
                                    { # nolint
    # step 0. load the RNA and ATAC data
    #raw_data_path = "/data/yangjoon.kim/bruno/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR119/outs/"
    counts <- Read10X_h5(paste0(raw_data_path, "filtered_feature_bc_matrix.h5"))
    fragpath <- paste0(raw_data_path, "atac_fragments.tsv.gz")

    # step 1. add the genome annotation
    # path to the GTF file
    gff_path = "/data/yangjoon.kim/bruno/projects/sequencing_alignment/alignment_references/"
    gref_path = paste0(gff_path, "zebrafish_genome_GRCz11/genes/genes.gtf.gz")
    gtf_zf <- rtracklayer::import(gref_path)

    # make a gene.coord object
    gene.coords.zf <- gtf_zf
    # filter out the entries without the gene_name
    gene.coords.zf <- gene.coords.zf[! is.na(gene.coords.zf$gene_name),]

    # only keep the regions within standard chromosomes
    gene.coords.zf <- keepStandardChromosomes(gene.coords.zf, pruning.mode = 'coarse')
    # name the genome - GRCz11
    genome(gene.coords.zf) <- 'GRCz11'

    # copy the "gene_id" for the "tx_id" and "transcript_id" 
    gene.coords.zf$tx_id <- gene.coords.zf$gene_id
    gene.coords.zf$transcript_id <- gene.coords.zf$gene_id

    # create a Seurat object containing the RNA assay
    multiome <- CreateSeuratObject(
    counts = counts$`Gene Expression`,
    assay = "RNA"
    )

    # create a ChromatinAssay object
    multiome[["ATAC"]] <- CreateChromatinAssay(
        counts = counts$Peaks,
        sep = c(":", "-"),
        genome = 'GRCz11', # we will manually add the genome version
        fragments = fragpath, # fragment file is saved using the filepath, so if the file is moved to another location, we need to update this.
        annotation = gene.coords.zf# we will manually add the genome annotation
    #     min.cells = 15
    )
    # return the Seurat object
    return(multiome)
}

# Step1-2. Filter out the low-quality cells.
# Note that we can be either very lenient or strict about the QC thresholds.
# We can just keep one QC standard for each time point, then perform additional QC after the cell-type annotation.
# REFER to the Zebrahub paper on how the QC step is described. 
filter_low_quality_cells <- function(object = multiome,
                                    nCount_ATAC_lower = 1000,
                                    nCount_ATAC_upper = 100000,
                                    nCount_RNA_lower = 1000,
                                    nCount_RNA_upper = 25000,
                                    nucleosome_signal_upper = 2,
                                    TSS_enrichment_lower = 1){
    DefaultAssay(multiome) <- "ATAC"

    multiome <- NucleosomeSignal(multiome)
    multiome <- TSSEnrichment(multiome)

    # filter out low quality cells
    # NOTE that we're not being meticulous about the thresholds here, especially the lower limits for RNA/ATAC fragment counts.

    # define the unfiltered object
    multiome_unfiltered <- multiome

    # filter out the low quality cells
    multiome <- subset(
    x = multiome,
    subset = nCount_ATAC < nCount_ATAC_upper &
        nCount_RNA < nCount_RNA_upper &
        nCount_ATAC > nCount_ATAC_lower &
        nCount_RNA > nCount_RNA_lower &
        nucleosome_signal < nucleosome_signal_upper &
        TSS.enrichment > TSS_enrichment_lower
    )

    # return the subsetted(QCed) Seurat object
    return(multiome)
    }

# Step2. Transfer the reference annotation to our dataset using the RNA anchors
# object: a Seurat object (RDS) from our multiome assay
# reference: a path to the reference Seurat object (RDS), RNA modality
# annotations: a list of annotation classes that we will transfer (i.e. coarse and fine)
transfer_reference_annotation_RNA_anchors <- function(object = multiome, 
                                            reference = reference,
                                            annotation_class = "global_annotation")
                                            {
    # To process the RNA (GEX) data from the Seurat object - SCTransform and RunPCA
    DefaultAssay(multiome) <- "RNA"
    multiome <- SCTransform(multiome)
    multiome <- RunPCA(multiome)

    # Import the reference Seurat object
    #reference <- readRDS(reference)
    reference <- LoadH5Seurat(reference)

    # preprocess the reference (so that we can compute the RNA anchors with our multiome dataset)
    reference <- SCTransform(reference)
    reference <- RunPCA(reference)

    # print the annotation labels to check
    print(unique(reference$global_annotation))

    # transfer cell type labels from reference to query
    transfer_anchors <- FindTransferAnchors(
        reference = reference, # reference object
        query = multiome, # our object of interest
        normalization.method = "SCT",
        reference.reduction = "pca",
        recompute.residuals = FALSE,
        dims = 1:50
    )

    #for (annotation in annotation_class){
    # transfer the label from "annotation" from a list of annotation classes ("annotation_class")
    predictions <- TransferData(
        anchorset = transfer_anchors, 
        refdata = reference$global_annotation,
        weight.reduction = multiome[['pca']],
        dims = 1:50
        )

    # add the predicted annotation to the main Seurat object
    # change the column name before adding the Metadata
    # rename the predictions_fine's colname
    colnames(predictions)[1] = "global_annotation"
    
    # Add the metadata (a dataframe whose column name is col.name)
    multiome <- AddMetaData(
        object = multiome,
        metadata = predictions,
        #col.name = "global_annotation", # use the annotation_class label from the reference
        )
    #}
    

    # set the cell identities to the cell type predictions
    Idents(multiome) <- "global_annotation"

    # return the Seurat object with the transferred annotations
    return(multiome)
    }

# Step3. Peak-calling (MACS2 - bulk, cell-type specific)
# perform MACS2(implemented in Signac) peak-calling for different group.by parameters (bulk, and cell-type specific)
call_MACS2_peaks_bulk_celltype <- function(object=multiome,
                                            annotation_class = "global_annotation"){

    # Change the default assay to "ATAC" for calling peaks
    DefaultAssay(multiome) <- "ATAC"
    
    # genome annotation (we need this to create a ChromatinAssay object)
    genome_annotation <- Annotation(multiome)
    # fragment path (path to the Fragment file)
    fragpath <- Fragments(multiome)[[1]]@path


    # call peaks for the bulk population
    peaks_bulk <- CallPeaks(
        object = multiome,
    )
    # remove peaks on nonstandard chromosomes and in genomic blacklist regions
    peaks_bulk <- keepStandardChromosomes(peaks_bulk, pruning.mode = "coarse")
    # peaks <- subsetByOverlaps(x = peaks, ranges = blacklist_hg38_unified, invert = TRUE)

    # quantify counts in each peak
    macs2_counts_bulk <- FeatureMatrix(
        fragments = Fragments(multiome),
        features = peaks_bulk,
        cells = colnames(multiome)
    )

    # create a new assay using the MACS2 peak set and add it to the Seurat object
    multiome[["peaks_bulk"]] <- CreateChromatinAssay(
        counts = macs2_counts_bulk,
        fragments = fragpath,
        annotation = genome_annotation
    )



    # call peaks using the annotations from the "annotation_class" (predicted labels from the reference)
    # We will default this to just one annotation_class for now, can be extended to multiple annotation_classes in future (for loop)
    # annotation <- "global_annotation"
    annotation <- annotation_class
    
    peaks <- CallPeaks(
        object = multiome,
        group.by = annotation
    )
    # remove peaks on nonstandard chromosomes and in genomic blacklist regions
    peaks <- keepStandardChromosomes(peaks, pruning.mode = "coarse")
    # peaks <- subsetByOverlaps(x = peaks, ranges = blacklist_hg38_unified, invert = TRUE)

    # quantify counts in each peak
    macs2_counts <- FeatureMatrix(
    fragments = Fragments(multiome),
    features = peaks,
    cells = colnames(multiome)
    )

    # create a new assay using the MACS2 peak set and add it to the Seurat object
    multiome[["peaks_celltype"]] <- CreateChromatinAssay(
        counts = macs2_counts,
        fragments = fragpath,
        annotation = genome_annotation
    )

    # return the Seurat object with updated ChromatinAssay objects
    return(multiome)
    }


# Step 4 (TBD). Compute the joint set of peaks (by merging cell-type specific peaks with bulk peaks)



# Step 5. Compute the dim.reduction and UMAPs - RNA, ATAC, and joint
# NOTE: We use PCA for RNA, and LSI for ATAC
# NOTE2: We will use the "peaks_celltype" ChromatinAssay object for computing the LSI/UMAP for the ATAC.
compute_embeddings <- function(object=multiome)
    {
    # check which embeddings are already present in the Seurat object (multiome)
    print(multiome@reductions)

    # RNA
    DefaultAssay(multiome) <-"RNA"
    multiome <- RunUMAP(multiome, reduction = "pca", dims = 1:50, reduction.name = "umap.rna")
    
    # ATAC
    DefaultAssay(multiome) <-"peaks_celltype"
    # preprocess the data (dim.reduction for UMAP)
    multiome <- FindTopFeatures(multiome, min.cutoff = 5)
    multiome <- RunTFIDF(multiome)
    multiome <- RunSVD(multiome)

    # UMAP
    # remove the first LSI since it's usually highly correlated to the sequencing-depth
    multiome <- RunUMAP(multiome, reduction = 'lsi', dims = c(2:40), assay = 'peaks_celltype', 
              reduction.name = 'umap.atac')

    # joint
    # Then, let's compute the weighted nearest neighbors and a joint embedding (UMAP)
    multiome <- FindMultiModalNeighbors(multiome,reduction.list = list("pca", "lsi"), dims.list = list(1:50, c(2:40)))
    multiome <- RunUMAP(multiome, nn.name = "weighted.nn", n.neighbors = 30, 
                        reduction.name = "umap.joint", reduction.key = "wnnUMAP_")
    # multiome <- FindClusters(multiome, graph.name = "wsnn", algorithm = 4, verbose = FALSE)

    # For plotting UMAPs, we can use the following command
    #DimPlot(multiome, label = TRUE, repel = TRUE, reduction = "umap.joint") + NoLegend()
    return(multiome)
    }

# Step 6 (Optional). Compute the Gene Activity (Signac)
# Signac: quantify the activity of each gene in the genome by simply 
# summing up the fragments intersecting the gene body and promoter region (2kb upstream)
# Then, we sum up the fragments for each gene, at each cell to construct a new count matrix, "Gene.Activity".
# This is all wrapped using "GeneActivity" in Signac
# Note that Cicero also has its own way of computing the Gene Activity count matrix.
compute_gene_activity <- function(object=multiome){
    
    DefaultAssay(multiome) <-"ATAC"
    # we use the Signac function "GeneActivity"
    gene.activities <- GeneActivity(multiome)
    # add the gene activity matrix to the Seurat object as a new assay and normalize it
    multiome[['Gene.Activity']] <- CreateAssayObject(counts = gene.activities)
    
    multiome <- NormalizeData(
        object = multiome,
        assay = 'Gene.Activity',
        normalization.method = 'LogNormalize',
        scale.factor = 10000
    )
    
    return(multiome)
    } 



##### THIS PART IS THE KEY COMMAND #####
##### ABOVE FUNCTIONS CAN BE WRAPPED INTO UTILITIES #####

# Execution on Linux Terminal

# # Load from here
# multiome <- readRDS(paste0(output_filepath,data_id,"_processed.RDS"))

# step 1. generate seurat object
multiome <- generate_seurat_object(raw_data_path, gref_path)
print("seurat object generated")

# step 2. basic QC
multiome <- filter_low_quality_cells(multiome)
saveRDS(object=multiome, file=paste0(output_filepath,data_id,"_processed.RDS"))
print("seurat object QCed, and saved")

# step 3. transfer the annotation (using RNA)
multiome <- transfer_reference_annotation_RNA_anchors(multiome, 
                                                        reference=reference, 
                                                        annotation_class=annotation_class) # nolint
saveRDS(object=multiome, file=paste0(output_filepath,data_id,"_processed.RDS"))
print("annotation transferred")


# step 4. peak-calling
multiome <- call_MACS2_peaks_bulk_celltype(multiome, annotation_class=annotation_class) # nolint
saveRDS(object=multiome, file=paste0(output_filepath,data_id,"_processed.RDS"))
print("peak-calling done")

multiome <- compute_embeddings(multiome)
print("embeddings computed")

# step 5. (Optional) Compute "Gene Activities"
multiome <- compute_gene_activity(multiome)
print("gene activity computed")

# step 6. save the RDS object
saveRDS(object=multiome, file=paste0(output_filepath,data_id,"_processed.RDS"))
print("seurat object saved")