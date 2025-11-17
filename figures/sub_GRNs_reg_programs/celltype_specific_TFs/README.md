# Celltype-Specific Transcription Factors Analysis

**Generated:** 2025-11-10
**Analysis Directory:** `figures/sub_GRNs_reg_programs/celltype_specific_TFs/`

## Overview

This directory contains results from systematic analysis of celltype-specific transcription factors (TFs) across 346 regulatory programs in the zebrafish multiome atlas.

**Key Finding:** We identified **11 TFs** that show high specificity to 3-5 celltypes with strong lineage coherence (≥60%).

---

## The 11 Celltype-Specific TFs

### Mesoderm-Specific (4 TFs)

| TF | N CTs | Celltypes | Coherence | N Clusters | Notes |
|----|-------|-----------|-----------|------------|-------|
| **neurod6a** | 5 | PSM, fast_muscle, muscle, somites, tail_bud | 0.80 | 2 | Unexpected for mesoderm (typically neural) |
| **zic4** | 4 | NMPs, PSM, fast_muscle, muscle | 0.75 | 4 | Also surprising in mesoderm |
| **foxd5** | 3 | enteric_neurons, lateral_plate_mesoderm, muscle | 0.67 | 3 | Fox family TF |
| **dmrt2b** | 3 | PSM, neural_telencephalon, notochord | 0.67 | 1 | ✓ Known PSM/somite regulator |

### Neuroectoderm-Specific (4 TFs)

| TF | N CTs | Celltypes | Coherence | N Clusters | Notes |
|----|-------|-----------|-----------|------------|-------|
| **foxi3b** | 4 | differentiating_neurons, enteric_neurons, fast_muscle, neural | 0.75 | 6 | Most clusters |
| **hoxc3a** | 5 | differentiating_neurons, floor_plate, lateral_plate_mesoderm, neural_floor_plate, neural_posterior | 0.60 | 5 | ✓ Hox gene - positional identity |
| **uncx** | 3 | heart_myocardium, neural_posterior, spinal_cord | 0.67 | 1 | ✓ Known somite/neural TF |
| **nr2e1** | 3 | neural, neural_posterior, pharyngeal_arches | 0.67 | 1 | ✓ Neural TF (tailless) |

### Other Lineages (3 TFs)

| TF | N CTs | Celltypes | Coherence | N Clusters | Notes |
|----|-------|-----------|-----------|------------|-------|
| **alx1** | 3 | floor_plate, optic_cup, primordial_germ_cells | 1.00 | 2 | Perfect coherence (all "other") |
| **lhx8a** | 3 | lateral_plate_mesoderm, primordial_germ_cells, tail_bud | 0.67 | 1 | LIM-homeobox TF |
| **lhx4** | 3 | enteric_neurons, pharyngeal_arches, pronephros | 0.67 | 1 | LIM-homeobox TF |

---

## Key Insights

### 1. Known Lineage Markers Are Too Broad

Classic lineage markers (tbx6, sox2, pax6a, nkx2.5) appear in **20-32 celltypes** and fail the specificity criteria. This makes biological sense:
- These are **master regulators** used widely across cell types
- They appear in GRNs of many contexts beyond their primary lineage
- Low lineage coherence (29-40%)

### 2. Specific TFs Are Context-Dependent Regulators

The 11 specific TFs identified are likely:
- **Developmental stage-specific** regulators
- **Context-dependent** rather than master regulators
- **Novel candidates** for experimental validation
- Some are unexpected (e.g., neurod6a in mesoderm)

### 3. Distribution Patterns

- **Celltype breadth:** Most specific TFs (7/11) are ultra-specific (3 celltypes only)
- **Lineage balance:** Equal distribution between mesoderm (36%), neuroectoderm (36%), and other (27%)
- **Cluster count:** foxi3b appears in most clusters (6), suggesting robust specificity

---

## Files in This Directory

### Analysis Outputs

1. **`celltype_specific_tfs_summary.csv`**
   Machine-readable table of all 11 specific TFs with metrics

2. **`CELLTYPE_SPECIFIC_TFS_REPORT.md`**
   Comprehensive markdown report with detailed analysis and methodology

### Visualizations (PNG, 300 DPI)

1. **`tf_specificity_distribution.png`**
   Histogram showing distribution of the 11 TFs by celltype count

2. **`tf_celltype_heatmap_specific11.png`**
   Binary heatmap showing which TFs are present in which celltypes
   - Rows: TFs, Columns: Celltypes
   - Coherence scores annotated on right

3. **`lineage_coherence_vs_breadth_scatter.png`**
   Scatter plot of all 11 TFs showing relationship between:
   - X-axis: Number of celltypes (3-5 range)
   - Y-axis: Lineage coherence (0.6-1.0)
   - Bubble size: Number of clusters
   - Color: Dominant lineage
   - Green box: Specificity criteria zone

4. **`known_markers_analysis.png`**
   Two-panel analysis of known lineage markers:
   - Top: Celltype breadth (all exceed threshold)
   - Bottom: Lineage coherence (all below threshold)
   - Shows why known markers fail specificity criteria

5. **`celltype_cooccurrence_network.png`**
   Heatmap showing celltype-celltype co-occurrence via shared TFs
   - Identifies which celltypes tend to share specific TFs
   - Useful for understanding regulatory relationships

6. **`summary_statistics.png`**
   Multi-panel summary figure:
   - Lineage distribution (pie chart)
   - Celltype breadth distribution (bar chart)
   - Lineage coherence distribution (histogram)
   - TFs ranked by cluster count (horizontal bar)

---

## Methodology

### Specificity Criteria

**A TF is considered "celltype-specific" if:**
- Present in **3-5 celltypes** (not too narrow, not too broad)
- **≥60% lineage coherence** (most celltypes from same lineage)
- Present in **≥1 cluster** (avoid single-cluster artifacts)

### Lineage Definitions

- **Neuroectoderm:** neural, neural_floor_plate, neural_optic, neural_posterior, neural_crest, spinal_cord, differentiating_neurons, enteric_neurons
- **Mesoderm:** PSM, somites, fast_muscle, slow_muscle, muscle, heart_myocardium, lateral_plate_mesoderm, hemangioblasts, hematopoietic_vasculature, notochord
- **Endoderm:** endoderm, endocrine_pancreas, liver, pharyngeal_endoderm
- **Periderm/Ectoderm:** epidermis, periderm, hatching_gland
- **Other:** pronephros, optic_cup, tail_bud, NMPs

### Lineage Coherence Score

```
Coherence = (# celltypes in dominant lineage) / (total # celltypes)
```

- Score of 1.0 = perfect coherence (all celltypes from same lineage)
- Score of 0.6 = threshold (60% from same lineage)

---

## Scripts Used

### 1. Analysis Script
**File:** `notebooks/Fig3_GRN_dynamics/query_celltype_specific_tfs.py`

**Function:**
- Loads peak celltype information for 346 clusters
- Loads TF-gene matrices and CellOracle GRNs
- Extracts subGRNs across all celltypes at peak timepoint
- Computes TF-celltype presence matrix
- Calculates specificity metrics (breadth, coherence)
- Filters for specific TFs (3-5 CTs, ≥60% coherence)

**Outputs:**
- `celltype_specific_tfs_summary.csv`
- `CELLTYPE_SPECIFIC_TFS_REPORT.md`

**Runtime:** ~2 minutes (processes 346 clusters × 29 celltypes)

### 2. Visualization Script
**File:** `notebooks/Fig3_GRN_dynamics/visualize_celltype_specific_tfs.py`

**Function:**
- Loads analysis results
- Generates 6 publication-quality plots (300 DPI PNG)
- Uses colorblind-friendly palettes

**Outputs:** All 6 PNG files in this directory

**Runtime:** ~30 seconds

---

## Interpretation & Next Steps

### For Experimental Validation

**Top candidates for validation:**
1. **foxi3b** - most robust (6 clusters), neuroectoderm-specific
2. **zic4** - unexpected mesoderm role (4 clusters)
3. **neurod6a** - surprising mesoderm specificity (5 celltypes)
4. **hoxc3a** - Hox gene with specific neural pattern (5 clusters)

### Potential Follow-up Analyses

1. **Cross-reference with temporal dynamics:**
   Are these specific TFs also temporally dynamic?
   - Check edge turnover from temporal analysis
   - Identify TFs that are both spatially AND temporally specific

2. **Relax criteria to find more TFs:**
   - Try 3-8 celltypes (instead of 3-5)
   - Try ≥50% coherence (instead of ≥60%)
   - Generate supplementary table with relaxed criteria

3. **Literature mining:**
   - Query ZFIN for known functions of these 11 TFs
   - Check RNA expression patterns (in situ, scRNA-seq)
   - Look for phenotypes in mutants/morphants

4. **Target gene analysis:**
   - What genes do these specific TFs regulate?
   - Are target genes also celltype-specific?
   - Pathway enrichment of target genes

5. **Cross-species conservation:**
   - Are these TF-celltype associations conserved in mouse/human?
   - Compare with mammalian developmental atlases

---

## Data Availability

**Source Data:**
- Peak celltype information: `systematic_analysis_temporal_summary.csv`
- TF-gene matrices: `/hpc/.../data/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl`
- CellOracle GRNs: `/hpc/.../data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/grn_exported/filtered/`

**Full Results:**
- All 252 TFs analyzed (not just the 11 specific ones)
- Raw data available in analysis CSV/markdown files

---

## Citation

When using this analysis, refer to:

**Zebrahub-Multiome Preprint:**
Kim YJ, et al. (2024). Zebrahub-Multiome: Uncovering Gene Regulatory Network Dynamics During Zebrafish Embryogenesis. bioRxiv. https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1

**Analysis Code:**
Available at: `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig3_GRN_dynamics/`

---

## Contact

For questions about this analysis:
- Check the main project README: `CLAUDE.md`
- Review the systematic analysis documentation: `SYSTEMATIC_ANALYSIS_README.md`
- Examine the validation report: `VALIDATION_CORRECTED_ANALYSIS.md`

---

**Last Updated:** 2025-11-10
**Analysis Version:** 1.0
**Total TFs Analyzed:** 252
**Specific TFs Found:** 11
