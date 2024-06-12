# A R module to compare different peak profiles for Seurat objects (Signac/MACS2)
suppressMessages(library(Signac))
suppressMessages(library(Seurat))
suppressMessages(library(GenomeInfoDb))
library(ggplot2)
library(patchwork)
library(stringr)
# library(rtracklayer)
library(GenomicRanges)

# zebrafish genome
# library(BSgenome.Drerio.UCSC.danRer11)

# A function to extract non-overlapping peaks from two GRanges objects
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