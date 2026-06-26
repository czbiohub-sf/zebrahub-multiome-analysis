#!/usr/bin/env Rscript
# Capture the R environment used for the Zebrahub-Multiome R preprocessing
# (Signac / Seurat / Cicero / monocle3). Run once on the project's R:
#
#   module load R/4.3
#   Rscript environments/capture_R_sessioninfo.R > environments/R_sessionInfo.txt
#
# Commit the resulting environments/R_sessionInfo.txt so reviewers can see exact versions.

pkgs <- c("Signac", "Seurat", "SeuratObject", "cicero", "monocle3",
          "GenomicRanges", "GenomeInfoDb", "BSgenome", "rtracklayer",
          "CHOIR", "Matrix", "data.table", "dplyr", "ggplot2")

cat("# Zebrahub-Multiome — R session info (Signac/Cicero/Seurat preprocessing)\n")
cat("#", R.version.string, "\n")
cat("# Platform:", R.version$platform, "\n\n")

cat("Key package versions\n--------------------\n")
for (p in pkgs) {
  v <- tryCatch(as.character(packageVersion(p)), error = function(e) NA_character_)
  cat(sprintf("  %-15s %s\n", p, ifelse(is.na(v), "(not installed)", v)))
}

# Load the headline packages (ignore any that are absent) so sessionInfo() reports them.
invisible(lapply(c("Signac", "Seurat", "cicero", "monocle3"), function(p)
  suppressMessages(suppressWarnings(tryCatch(library(p, character.only = TRUE), error = function(e) NULL)))))

cat("\n--- sessionInfo() ---\n")
print(sessionInfo())
