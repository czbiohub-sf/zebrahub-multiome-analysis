library(Signac)
library(Seurat)
library(GenomicRanges)

# Assuming `integrated_ATAC` is your existing Seurat object
integrated_ATAC <- readRDS("path/to/integrated_ATAC.rds")
# and `peaks_integrated` is the name of the assay containing integrated peaks

# Step 1: Extract the counts matrix
counts_matrix <- GetAssayData(integrated_ATAC, assay = "peaks_integrated", slot = "counts")

# Step 2: Correctly extract or define peak metadata (assuming it's not already in a GRanges object)
peak_coordinates <- rownames(counts_matrix)
seqnames <- gsub("^(.*):.*$", "\\1", peak_coordinates) # Extract chromosome names
starts <- as.integer(sub("^.+:(\\d+)-.*$", "\\1", peak_coordinates)) # Correctly extract start positions
ends <- as.integer(sub("^.+-(\\d+)$", "\\1", peak_coordinates)) # Correctly extract end positions

# Create the GRanges object with the corrected extraction
granges_peaks <- GRanges(
  seqnames = seqnames,
  ranges = IRanges(start = starts, end = ends)
)


# Step 3: Create the ChromatinAssay object
chromatin_assay <- CreateChromatinAssay(
  counts = counts_matrix,
  sep = c(":", "-"),
  genome = 'GRCz11', # Specify the appropriate genome
  #fragments = 'path/to/fragments.tsv.gz', # Specify the path to your fragments file, if available
  ranges = granges_peaks,
  #min.cells = 1, # Adjust based on your data
  #min.features = 1 # Adjust based on your data
)

# Add the list of Fragment files to the ChromatinAssay object
#chromatin_assay@fragments <- list.files("path/to/fragments", full.names = TRUE)

# Step 4: Create a new Seurat object or add to existing one
new_seurat_object <- CreateSeuratObject(counts = NULL)
new_seurat_object[['peaks']] <- chromatin_assay

# Now `new_seurat_object` contains a ChromatinAssay named 'peaks'
