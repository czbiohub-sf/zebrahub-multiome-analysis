# This function is originally from SEACells (Dana Pe'er's lab, 2023)
# The original notebook grabs 10x alignment data for the scATAC-seq, and preprocessed for the following:
# 1) Create an Arrow file (ArchR object)
# 2) SVD, Clustering, UMAP
# 3) Gene score
# 4) Peak calling for NFR (Nucleosome Free Reads)
# 5) Export the results into csv files

#' This function takes scATAC-seq data, performs preprocessing steps, and generates an ArchR object 
#' # for downstream analysis as described in the SEACells script.
#'
#' @param n_threads Integer. The number of threads to use for processing (default: 1).
#' @param genome Character. The genome assembly used for alignment (e.g., "hg19", "mm10").
#' @param data_dir Character. The directory containing the ATAC fragments file.
#' @param inputFiles Character vector. The names of the fragment files.
#' @param proj_name Character. The name of the ArchR project.
#'
#' @return ArchRProject. The generated ArchR project object.
#'
#' @examples
#' archr_from_scATAC(n_threads = 1, genome = "hg19", data_dir = "<DIRECTORY>", inputFiles = c("fragment1.bam", "fragment2.bam"), proj_name = "my_project")
archr_from_scATAC <- function(n_threads = 10, genome, data_dir, inputFiles, proj_name) {
    library(ArchR)
    set.seed(1)

    # PARAMETERS
    n_threads <- n_threads # original was 1
    genome <- ""
    data_dir <- "<DIRECTORY CONTAIN THE ATAC FRAGMENTS FILE"
    inputFiles <- "FRAGMENT FILEs WITH NAME"
    proj_name <- "NAME OF ARCHR Project"

    # Configure (ADD ERROR MESSAGES IF NOT PROVIDED)
    # Set the number of threads
    addArchRThreads(threads = n_threads)
    # Set the genome assembly
    addArchRGenome(genome)


    # ################################################################################################
    # Arrow files and project 

    # Input files
    setwd(sprintf("%s/ArchR", data_dir))

    # Create Arrow files
    ArrowFiles <- createArrowFiles(
        inputFiles = inputFiles,
        sampleNames = names(inputFiles),
        filterTSS = 1, #Dont set this too high because you can always increase later
        filterFrags =3000, 
        addTileMat = TRUE,
        addGeneScoreMat = FALSE,
        excludeChr = c('chrM'),
        removeFilteredCells = TRUE
        )


    # Create project
    proj <- ArchRProject(
        ArrowFiles = ArrowFiles, 
        outputDirectory = proj_name,
        copyArrows = FALSE
        )


    # ################################################################################################
    # Preprocessing

    # SVD, Clustering, UMAP
    proj <- addIterativeLSI(ArchRProj = proj, useMatrix = "TileMatrix", 
                        name = "IterativeLSI", force=TRUE)

    # Gene scores with selected features
    # Artificial black list to exclude all non variable features
    chrs <- getChromSizes(proj)
    var_features <- proj@reducedDims[["IterativeLSI"]]$LSIFeatures
    var_features_gr <- GRanges(var_features$seqnames, IRanges(var_features$start, var_features$start + 500))
    blacklist <- setdiff(chrs, var_features_gr)
    proj <- addGeneScoreMatrix(proj, matrixName='GeneScoreMatrix', force=TRUE, blacklist=blacklist)



    # Peaks using NFR fragments
    proj <- addClusters(input = proj, reducedDims = "IterativeLSI")
    proj <- addGroupCoverages(proj, maxFragmentLength=147)
    proj <- addReproduciblePeakSet(proj)
    # Counts
    proj <- addPeakMatrix(proj, maxFragmentLength=147, ceiling=10^9)

    # Save 
    proj <- saveArchRProject(ArchRProj = proj)



    # ################################################################################################

    # Export
    dir.create(sprintf("%s/export", proj_name))
    write.csv(getReducedDims(proj), sprintf('%s/export/svd.csv', proj_name), quote=FALSE)
    write.csv(getCellColData(proj), sprintf('%s/export/cell_metadata.csv', proj_name), quote=FALSE)


    # Gene scores
    gene.scores <- getMatrixFromProject(proj)
    scores <- assays(gene.scores)[['GeneScoreMatrix']]
    scores <- as.matrix(scores)
    rownames(scores) <- rowData(gene.scores)$name
    write.csv(scores, sprintf('%s/export/gene_scores.csv', proj_name), quote=FALSE)



    # Peak counts
    peaks <- getPeakSet(proj)
    peak.counts <- getMatrixFromProject(proj, 'PeakMatrix')

    # Reorder peaks 
    # Chromosome order
    chr_order <- sort(seqlevels(peaks))
    reordered_features <- list()
    for(chr in chr_order)
        reordered_features[[chr]] = peaks[seqnames(peaks) == chr]
    reordered_features <- Reduce("c", reordered_features)    

    # Export counts
    dir.create(sprintf("%s/export/peak_counts", proj_name))
    counts <- assays(peak.counts)[['PeakMatrix']]
    writeMM(counts, sprintf('%s/export/peak_counts/counts.mtx', proj_name))
    write.csv(colnames(peak.counts), sprintf('%s/export/peak_counts/cells.csv', proj_name), quote=FALSE)
    names(reordered_features) <- sprintf("Peak%d", 1:length(reordered_features))
    write.csv(as.data.frame(reordered_features), sprintf('%s/export/peak_counts/peaks.csv', proj_name), quote=FALSE)

    # Return the ArchR object
    return(proj)
}