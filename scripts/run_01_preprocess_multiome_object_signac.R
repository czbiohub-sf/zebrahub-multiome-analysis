# Utilities for basic QC pre-processing for single-cell multiome object

# This workflow is designed for processing one single-cell multiome dataset (one experiment)
# For integration of multiple "replicates", we will need additional steps.
# We recommend using RNA modalities to integrate multiple replicates, then perform joint peak-calling (either bulk, or cell-type specific)

#### MAKE SURE TO run "module load R/4.3 in HPC"

# load the libraries
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
reference <- args[3] # reference dataset with cell-type annotation
annotation_class <- args[4] # Let's just use one annotation_class. (i.e. "global_annotation")
output_filepath <- args[5] # filepath for the output
data_id <- args[6]
# assays_save <- args[7] # Assays in Seurat object that will be exported to h5ad objects
# assays_save <- list("RNA", "ATAC")

# Sub-functions
# Step1. generate a Seurat object from the Cellranger-arc output (multiome)
generate_seurat_object <- function(raw_data_path ="" ,
                                    gref_path ="")
                                    { # nolint
    # step 0. load the RNA and ATAC data
    #raw_data_path = "/data/yangjoon.kim/bruno/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR119/outs/"
    counts <- Read10X_h5(paste0(raw_data_path, "/filtered_feature_bc_matrix.h5"))
    fragpath <- paste0(raw_data_path, "/atac_fragments.tsv.gz")

    # step 1. add the genome annotation
    # path to the GTF file
    #gff_path = "/hpc/reference/sequencing_alignment/alignment_references/"
    #gref_path = paste0(gff_path, "zebrafish_genome_GRCz11/genes/genes.gtf.gz")
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
        cells = colnames(multiome),
        process_n=200000,
    )

    # create a new assay using the MACS2 peak set and add it to the Seurat object
    multiome[["peaks_bulk"]] <- CreateChromatinAssay(
        counts = macs2_counts_bulk,
        sep = c(":", "-"),
        fragments = fragpath,
        annotation = genome_annotation,
        genome = 'GRCz11', # we will manually add the genome version
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
    cells = colnames(multiome),
    process_n=200000,
    )

    # create a new assay using the MACS2 peak set and add it to the Seurat object
    multiome[["peaks_celltype"]] <- CreateChromatinAssay(
        counts = macs2_counts,
        sep = c(":", "-"),
        fragments = fragpath,
        annotation = genome_annotation,
        genome = 'GRCz11', # we will manually add the genome version
    )

    # return the Seurat object with updated ChromatinAssay objects
    return(multiome)
    }


# Step 4. Compute the joint set of peaks (by merging cell-type specific peaks with bulk, and CRG-arc peaks)
# Note. we use "iterative overlap merging described in ArchR paper. We assume that the importance of peaks is celltype>bulk>CRG-arc".
# Note2. We can also use the "union" of the peaks, but we will use the "iterative overlap merging" for now.
merge_peaks <- function(object=multiome){

    # Step 4-1. Extract non-overlapping (unique) peaks in peaks_bulk that are not in peaks_celltype
    # first, extract the peaks from the Seurat object (gRanges objects)
    peaks_celltype <- multiome@assays$peaks_celltype@ranges
    peaks_bulk <- multiome@assays$peaks_bulk@ranges
    peaks_CRG <- multiome@assays$ATAC@ranges

    # unique peaks in peaks_bulk compared to peaks_celltype
    non_overlapping_peaks_bulk <- extractNonOverlappingPeaks(peaks_bulk, peaks_celltype)
    non_overlapping_peaks_bulk

    # Merge the non-overlapping peaks with the celltype peaks
    peaks_merged <- c(peaks_celltype, non_overlapping_peaks_bulk)
    peaks_merged

    # Step 4-2. Extract non-overlapping (unique) peaks in peaks_CRG that are not in peaks_merged
    non_overlapping_peaks_CRG <- extractNonOverlappingPeaks(peaks_CRG, peaks_merged)
    non_overlapping_peaks_CRG

    peaks_merged <- c(peaks_merged, non_overlapping_peaks_CRG)
    peaks_merged

    # Change the default assay to "ATAC" for calling peaks
    DefaultAssay(multiome) <- "ATAC"

    # genome annotation (we need this to create a ChromatinAssay object)
    genome_annotation <- Annotation(multiome)
    # fragment path (path to the Fragment file)
    fragpath <- Fragments(multiome)[[1]]@path

    # remove peaks on nonstandard chromosomes and in genomic blacklist regions
    peaks_merged <- keepStandardChromosomes(peaks_merged, pruning.mode = "coarse")
    # peaks <- subsetByOverlaps(x = peaks, ranges = blacklist_hg38_unified, invert = TRUE)

    # quantify counts in each peak
    macs2_counts_merged <- FeatureMatrix(
        fragments = Fragments(multiome),
        features = peaks_merged,
        cells = colnames(multiome),
        process_n=200000,
    )

    # create a new assay using the MACS2 peak set and add it to the Seurat object
    multiome[["peaks_merged"]] <- CreateChromatinAssay(
        counts = macs2_counts_merged,
        sep = c(":", "-"),
        fragments = fragpath,
        annotation = genome_annotation,
        genome = 'GRCz11', # we will manually add the genome version
    )

    return(multiome)
}



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
    #DefaultAssay(multiome) <-"peaks_celltype"
    DefaultAssay(multiome) <-"peaks_merged"
    # preprocess the data (dim.reduction for UMAP)
    multiome <- FindTopFeatures(multiome, min.cutoff = 5)
    multiome <- RunTFIDF(multiome)
    multiome <- RunSVD(multiome)

    # UMAP
    # remove the first LSI since it's usually highly correlated to the sequencing-depth
    multiome <- RunUMAP(multiome, reduction = 'lsi', dims = c(2:40), assay = 'peaks_merged', 
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
    
    DefaultAssay(multiome) <-"peaks_merged"
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

# step 7. Convert Seurat object (multiple assays) into h5ad objects per assay
# define the function
export_seurat_assays <- function(object=multiome, 
                                output_dir, data_id, assays_save) {
  # Read the input Seurat object
  # seurat <- readRDS(input_dir_prefix)
  
  # Loop through the specified assays
  for (assay in assays_save) {
    # Set the default assay
    DefaultAssay(multiome) <- assay
    print(multiome)
    
    # Save the object (assay)
    filename <- file.path(output_dir, paste0(data_id, "_processed_", assay, ".h5Seurat"))
    SaveH5Seurat(multiome, filename = filename, overwrite = TRUE)
    
    # Convert the h5Seurat to h5ad
    Convert(filename, dest = "h5ad")
  }
}

# Define a function to generate unique filenames for each checkpoint
generate_filename <- function(base_path, data_id, suffix) {
  return(paste0(base_path, data_id, "_", suffix, ".RDS"))
}

# define a function that computes non-overlapping peaks between two gRanges objects
# this function computes the non-overlapping peaks in granges1 compared to granges2
extractNonOverlappingPeaks <- function(granges1, granges2) {
  # Find overlaps
  overlaps <- findOverlaps(granges1, granges2)

  # Get unique indices(peaks) of granges1 that overlap with granges2 peaks
  unique_overlaps <- unique(queryHits(overlaps))

  # Get all indices in granges1
  all_indices_granges1 <- seq_along(granges1)

  # Find indices in granges1 that are not in unique overlaps
  unique_indices_granges1 <- setdiff(all_indices_granges1, unique_overlaps)

  # Extract non-overlapping peaks from granges1
  non_overlapping_peaks <- granges1[unique_indices_granges1]

  return(non_overlapping_peaks)
}

##### THIS PART IS THE KEY COMMAND #####
##### ABOVE FUNCTIONS CAN BE WRAPPED INTO UTILITIES #####

# Execution on Linux Terminal

# # Load from here
# multiome <- readRDS(paste0(output_filepath,data_id,"_processed.RDS"))

# step 1. generate seurat object
multiome <- generate_seurat_object(raw_data_path, gref_path)
#saveRDS(object=multiome, file=generate_filename(output_filepath, data_id, "raw"))
print("seurat object generated")

# step 1-2. basic QC
multiome <- filter_low_quality_cells(multiome)
#saveRDS(object=multiome, file=paste0(output_filepath,data_id,"_processed.RDS"))
#saveRDS(object=multiome, file=generate_filename(output_filepath, data_id, "QCed"))
print("seurat object QCed")

# step 2. transfer the annotation (using RNA)
multiome <- transfer_reference_annotation_RNA_anchors(multiome, 
                                                        reference=reference, 
                                                        annotation_class=annotation_class) # nolint
#saveRDS(object=multiome, file=paste0(output_filepath,data_id,"_processed.RDS"))
#saveRDS(object=multiome, file=generate_filename(output_filepath, data_id, "annotated"))
print("annotation transferred")


# step 3. peak-calling
multiome <- call_MACS2_peaks_bulk_celltype(multiome, annotation_class=annotation_class) # nolint
#saveRDS(object=multiome, file=paste0(output_filepath,data_id,"_processed.RDS"))
#saveRDS(object=multiome, file=generate_filename(output_filepath, data_id, "peak_called"))
print("peak-calling done")

# step 4. merge the peaks (CRG, bulk, celltype)
multiome <- merge_peaks(multiome)
#saveRDS(object=multiome, file=generate_filename(output_filepath, data_id, "merged_peaks"))
print("ATAC peaks merged")

# step 5. compute embeddings
multiome <- compute_embeddings(multiome)
#saveRDS(object=multiome, file=generate_filename(output_filepath, data_id, "embeddings"))
print("embeddings computed")

# step 6. (Optional) Compute "Gene Activities"
multiome <- compute_gene_activity(multiome)
saveRDS(object=multiome, file=generate_filename(output_filepath, data_id, "gene_activity"))
print("gene activity computed")

# step 7. convert the RDS object to h5ad object (both RNA and ATAC)
# TBD: "assays_save" parameter should be defined at the very top
export_seurat_assays(object = multiome,
                    output_dir = output_filepath,
                    data_id = data_id,
                    assays_save= c("RNA", "peaks_merged"))

saveRDS(object=multiome, file=generate_filename(output_filepath, data_id, "processed"))                    
print("seurat object exported to h5ad objects per assay")