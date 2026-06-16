# Methods: Peak Parts List — Celltype-Specific Regulatory Element Catalog

## Pseudobulk accessibility matrix

Single-cell ATAC-seq fragments from 94,562 cells across 32 cell types and 6 developmental timepoints (0, 5, 10, 15, 20, and 30 somites) were aggregated into a pseudobulk accessibility matrix. For each cell type × timepoint group, raw fragment counts were summed across all cells within the group for each of 640,830 consensus peaks (danRer11 genome assembly). Groups with fewer than 20 cells were flagged as unreliable and excluded from downstream analyses (14 of 190 groups, including all primordial germ cell timepoints). The pseudobulk counts were normalized using a median-scaling approach: for each group, a scale factor was computed as the ratio of the median total coverage across all groups to that group's total coverage, and counts were multiplied by this scale factor. The normalized counts were then log-transformed using log(1 + x). This produced a 640,830 × 190 log-normalized accessibility matrix.

## Celltype-level specificity z-scores

To identify peaks with celltype-specific accessibility independent of temporal dynamics, we computed celltype-level mean accessibility by averaging the log-normalized values across all reliable timepoints within each cell type, yielding a 640,830 × 31 matrix (excluding primordial germ cells). We then computed a leave-one-out z-score for each peak in each cell type:

z_i,c = (x_i,c − μ_i,−c) / σ_i,−c

where x_i,c is the celltype-level mean accessibility of peak i in cell type c, and μ_i,−c and σ_i,−c are the mean and standard deviation of peak i's accessibility across all cell types excluding c. A minimum standard deviation floor of 1 × 10⁻⁵ was applied to avoid division by zero. Higher z-scores indicate greater celltype specificity. For each cell type, the top 200 peaks ranked by z-score were selected as the celltype-specific regulatory element catalog.

## Tau specificity index

To quantify the overall celltype specificity of each peak as a single scalar, we computed the Tau index (Yanai et al., 2005):

τ_i = Σ_c (1 − x̂_i,c) / (N − 1)

where x̂_i,c = x_i,c / max_c(x_i,c) is the celltype-level mean accessibility of peak i normalized to its maximum across cell types, and N = 31 is the number of cell types. Tau ranges from 0 (uniformly accessible across all cell types) to 1 (accessible in only one cell type). Peaks with zero accessibility across all cell types were assigned τ = 0.

## TF motif scanning

Transcription factor binding motif occurrences within the top-200 peaks for each cell type (6,200 peaks total) were identified using FIMO (Grant et al., 2011) as implemented in pymemesuite. We scanned each peak's DNA sequence (extracted from the danRer11 reference genome using pysam) against 1,443 position weight matrices from the JASPAR 2022 CORE vertebrate collection (H12CORE). Motif hits were called at a significance threshold of p < 1 × 10⁻⁴ on both strands. Motif variants mapping to the same transcription factor (identified by parsing the JASPAR accession prefix) were collapsed, yielding 949 unique TFs.

To enable parallelization, FIMO scanning was performed as a SLURM array job with one task per cell type (31 tasks), each scanning 200 peaks against all 1,443 motifs. Results were merged into a unified binary hit matrix (6,200 peaks × 949 TFs) and a position-level table recording the exact genomic coordinates, strand, score, and p-value of each motif occurrence.

## Motif enrichment analysis

For each cell type, we tested whether each TF motif was enriched in that cell type's top-200 peaks relative to all other cell types' top-200 peaks pooled as background (~6,000 peaks). For each TF, we constructed a 2 × 2 contingency table:

|                | Motif present | Motif absent |
|----------------|---------------|--------------|
| Focal (200)    | a             | b            |
| Background (~6,000) | c        | d            |

where a motif was considered present in a peak if at least one hit (p < 10⁻⁴) was detected for any variant of that TF's PWM. Significance was assessed using Fisher's exact test (one-sided, greater). P-values were corrected for multiple testing within each cell type using the Benjamini-Hochberg procedure. TFs with FDR < 0.05 were considered significantly enriched.

To compare enrichment across cell types, we computed a cross-celltype enrichment z-score for each TF:

z_enrichment = (r_c − μ_r) / (σ_r + 0.02)

where r_c is the motif hit rate (fraction of peaks with a hit) in cell type c, and μ_r and σ_r are the mean and standard deviation of hit rates across all 31 cell types. The pseudocount of 0.02 in the denominator prevents extreme z-scores for TFs with near-uniform hit rates.

## Motif support classification

Each peak was classified as having "motif support" if it contained hits for 3 or more distinct significantly enriched TFs (FDR < 0.05 in that peak's cell type). Peaks lacking motif support may represent structural elements (e.g., CTCF boundaries), passively decompacted chromatin, or elements driven by TFs not represented in the JASPAR database, rather than TF-driven enhancers. This classification serves as a secondary filter for selecting candidate regulatory elements for synthetic enhancer design.

## RNA-ATAC concordance

To validate that ATAC-based peak specificity reflects gene regulatory activity, we constructed an RNA pseudobulk matrix from the matched single-cell RNA-seq data (94,562 cells × 32,057 genes) using the identical normalization pipeline as the ATAC pseudobulk: raw count summation per cell type × timepoint group, median-scale normalization, and log(1 + x) transformation. Celltype-level RNA z-scores were computed using the same leave-one-out formula, with a standard deviation floor of 0.5 to account for the zero-inflated nature of gene expression data (many genes have zero expression in most cell types, unlike ATAC peaks which have nonzero accessibility by definition). For each cell type's top ATAC peaks with a linked or nearest gene annotation, we compared the ATAC specificity z-score with the RNA expression z-score for the corresponding gene in the same cell type.

## Software

All analyses were performed in Python 3.10 using anndata (Wolf et al., 2018) for data management, pymemesuite for FIMO motif scanning, pysam for genome sequence extraction, scipy for statistical tests, and matplotlib/seaborn for visualization. SLURM was used for job scheduling on the HPC cluster.
