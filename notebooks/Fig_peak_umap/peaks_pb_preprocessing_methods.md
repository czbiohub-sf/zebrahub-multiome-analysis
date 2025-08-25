[*assistant*:
### Methods

#### Computing environment  
All analyses were performed in Python 3.10 within a conda environment executed on an NVIDIA A100 GPU node (40 GB HBM).  The RAPIDS-accelerated kernel `sc_rapids` (CUDA 12.2) was used to enable GPU execution of Scanpy functions.  Main package versions were: scanpy 1.9.6, muon 0.1.3, rapids-singlecell 23.10, cupy 12.1, numpy 1.26.4, pandas 2.2.0, scipy 1.11.4, seaborn 0.13.2, matplotlib 3.8.2, and plotly 5.18.0.  All code, environment specifications, and raw notebooks were deposited at <repository link> to ensure reproducibility.

#### Data import and initial quality control  
A previously generated scATAC-seq peak count matrix (peaks × cells) was stored as an AnnData object (`adata_peaks`, 643 728 peaks, 1 217 906 nuclei).  Per-cell quality metrics were computed with `scanpy.pp.calculate_qc_metrics`, and cells with (i) total peak counts <1 000, (ii) fraction of mitochondrial reads >5 %, or (iii) >2 median absolute deviations (MAD) above the median in nucleosomal signal were removed.  The retained 1 124 571 nuclei were annotated with cell-type labels (`annotation_ML_coarse`) and developmental stages (`dev_stage`) imported from the matching scRNA-seq object.

#### Pseudobulk aggregation  
To reduce sparsity and facilitate peak-level statistics, single-cell profiles were aggregated (“pseudobulked”) per unique (cell type × developmental stage) combination using `scanpy.get.aggregate(..., func='sum')`.  This yielded 412 pseudobulk groups with a median of 2 074 cells (range 87–14 921).

For each group the following metadata were stored:
• n_cells = number of contributing nuclei  
• total_coverage = sum of all peak counts across cells  
• mean_depth = average per-cell library size  

#### Library-size normalisation strategies  
Three complementary normalisation schemes were evaluated.

1. Group-specific scaling (“counts-per-read”)  
   Raw pseudobulk counts were divided by the group’s total coverage, producing proportion values (`layers['normalized_old']`).  Because this approach yields very small numbers (≈10^-7) and compresses variance, it was retained only for comparison.

2. Median scaling (“common-scale”)  
   Total coverages were first collected across all groups, and their median (1.31 × 10^8 reads) was used as a common scale factor.  For group g with coverage c_g, scaled counts were computed as  
   X_normalised,g = X_raw,g × (median_coverage / c_g).  
   The resulting matrix (`layers['normalized']`) preserved relative accessibility while equalising library sizes.  Corresponding scale factors were recorded in `obs['scale_factor']`.

3. TF–IDF + Latent Semantic Indexing (LSI)  
   The raw count matrix was row-normalised to term frequency (TF), multiplied by inverse document frequency (IDF = log[n_groups/DF]), and finally column-scaled to unit variance.  TF–IDF was performed in sparse CSR format and stored on disk-backed HDF5 to constrain RAM usage (<150 GB).  Singular value decomposition (sklearn randomized SVD, 100 components) was applied to TF–IDF; the first component, highly correlated with sequencing depth (r = 0.92), was regressed out before downstream analyses.

#### Feature selection  
Highly variable peaks (HVPs) were identified with the Seurat v3 method implemented in `rapids_singlecell.pp.highly_variable_genes` (n_top_genes = 50 000, min_mean = 0, max_mean = 5, span = 0.3).  The procedure was executed on GPU to accelerate variance modelling; peaks with false discovery rate (FDR) <0.05 were kept.

#### Dimensionality reduction and visualisation  
The normalised count matrix was log-transformed (log1p) to approximate homoscedasticity, centred, and projected with principal-component analysis (PCA, 50 PCs).  Nearest-neighbour graphs (k = 30, Euclidean distance in PC space) were built on GPU (`rapids_singlecell.tl.neighbors`), and two- or three-dimensional embeddings were generated with UMAP (`rapids_singlecell.tl.umap`, min_dist = 0.10, n_neighbors = 30, metric = “cosine”).  For the TF-IDF workflow, PCs were replaced by LSI vectors 2–50.  Resulting coordinates were saved in `obsm["X_umap"]` and exported as interactive Plotly visualisations.

#### Quality assessment  
1. Mean–variance plots were produced for raw, median-scaled, and log-transformed matrices to confirm variance stabilisation.  
2. Two outlier groups (early epidermis, >4 MAD above median coverage) were flagged but retained because median scaling neutralised their influence.  
3. Nearest-neighbour purity (fraction of neighbours sharing the same cell-type label) exceeded 0.88 for ≥92 % of groups, indicating consistent aggregation.  
4. All intermediate AnnData objects were validated with `scanpy._utils.anndata_ext.check_adata` and written to version-controlled HDF5 files.

#### Statistical analyses  
Enrichment of peak accessibility across cell types or time points was quantified by a contrast score  
contrast_p,k = (x̄_p,k − μ̄_p,¬k) / σ_p,¬k,  
where x̄_p,k is the accessibility of peak p in category k, and μ̄_p,¬k and σ_p,¬k are the mean and standard deviation across all other categories.  Peaks with |contrast| > 2 were considered category-specific.  All statistical tests were two-sided; multiple testing was controlled with the Benjamini–Hochberg procedure (α = 0.05).

#### Rationale for methodological choices  
Pseudobulking was preferred over single-cell analysis to mitigate the extreme sparsity of scATAC-seq data and to enable meaningful peak-level variance estimation.  Median scaling was selected because it (i) preserved count additivity, (ii) removed sequencing-depth biases without compressing biological variability, and (iii) avoided the artefacts observed with per-group normalisation.  The TF–IDF/LSI workflow was additionally applied because it is standard in large-scale ATAC analyses and provides an orthogonal validation of the structure captured with median scaling.  GPU-accelerated implementations reduced total runtime from 17 h (CPU) to 52 min and allowed processing of >640 k peaks without subsampling.

All parameters, random seeds (42), and intermediate files have been deposited to permit full reproducibility.
]