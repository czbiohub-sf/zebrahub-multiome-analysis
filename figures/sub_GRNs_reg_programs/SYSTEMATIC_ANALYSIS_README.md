# Systematic SubGRN Analysis - Complete Documentation

**Generated:** 2025-11-05

## Overview

This directory contains comprehensive systematic analysis of **346 peak clusters** with TF-gene regulatory meshes, examining both **temporal dynamics** (changes across developmental timepoints) and **spatial dynamics** (changes across cell types).

## Analysis Scope

- **Total peak clusters:** 402
- **Clusters with TF-gene matrices:** 346 (analyzed)
- **Clusters without matrices:** 56 (documented only)
- **Timepoints analyzed:** 6 (0, 5, 10, 15, 20, 30 somites)
- **Celltype×timepoint combinations:** 189 GRNs

## Key Files

### Analysis Scripts

1. **`notebooks/Fig3_GRN_dynamics/analyze_all_clusters_systematic.py`**
   - Main analysis pipeline
   - Performs peak detection
   - Computes temporal and spatial dynamics
   - Generates summary CSV files

2. **`notebooks/Fig3_GRN_dynamics/generate_markdown_reports.py`**
   - Generates comprehensive markdown reports
   - Creates curated highlights

### Output Files

#### Summary Tables (CSV)

1. **`systematic_analysis_temporal_summary.csv`** (346 rows)
   - Cluster ID, peak location (celltype×timepoint)
   - Network size (nodes, edges, TF classification)
   - Active timepoints
   - Most dynamic TF and turnover score

2. **`systematic_analysis_spatial_summary.csv`** (346 rows)
   - Cluster ID, peak location
   - Number of celltypes with subGRN
   - Total TFs
   - Most ubiquitous and most specific TFs

#### Full Reports (Markdown)

3. **`temporal_subGRN_dynamics_all_clusters.md`** (34 KB)
   - Detailed temporal dynamics for all 346 clusters
   - Top 20 most dynamic clusters
   - Distribution by peak celltype
   - Per-cluster detailed reports (top 100)

4. **`spatial_celltype_subGRN_dynamics_all_clusters.md`** (23 KB)
   - Detailed spatial dynamics for all 346 clusters
   - Most ubiquitous TFs across clusters
   - Most celltype-specific TFs
   - Per-cluster detailed reports (top 100)

5. **`TOP_DYNAMIC_PROGRAMS_SUMMARY.md`** (3.3 KB)
   - **Curated highlights** of most dynamic programs
   - Top 20 temporal champions (highest edge turnover)
   - Top 20 spatial champions (broadest celltype activity)
   - Key insights and summary statistics

#### Reference Files

6. **`CLUSTERS_WITHOUT_MATRICES.txt`** (56 clusters)
   - List of clusters without TF-gene matrices
   - Cannot be analyzed for subGRN dynamics
   - Documented for completeness

## Analysis Details

### Temporal Dynamics

For each cluster at its **peak celltype**:
- Extract subGRNs across all 6 timepoints
- Track TF dynamics:
  - **Edge turnover**: gained/lost edges per TF
  - **Role switching**: TF-only ↔ TF&Target transitions
  - **Sign flipping**: activation ↔ repression changes
- Rank TFs by total dynamics score

**Key Metrics:**
- Total nodes and edges across timepoints
- TF classification (TF-only, Target-only, TF&Target)
- Active timepoints (out of 6)
- Top dynamic TFs per cluster

### Spatial Dynamics

For each cluster at its **peak timepoint**:
- Extract subGRNs across all celltypes
- Identify:
  - **Ubiquitous TFs**: present in many celltypes
  - **Celltype-specific TFs**: present in few celltypes
- Lineage analysis:
  - Mesodermal: NMPs → tail_bud → PSM → somites → fast_muscle
  - Neural: NMPs → spinal_cord → neural_posterior

**Key Metrics:**
- Number of celltypes with subGRN activity
- Total TFs across celltypes
- TF celltype specificity scores

## Top Findings

### Temporal Champions (highest edge turnover)

1. **Cluster 33_1** (neural_optic, 20 som): 160 nodes, 595 edges, sox13 (turnover=104)
2. **Cluster 26_8** (neural_floor_plate, 30 som): 163 nodes, 677 edges, sox13 (turnover=98)
3. **Cluster 21_6** (tail_bud, 10 som): 123 nodes, 370 edges, meox1 (turnover=82)

### Spatial Champions (broadest celltype activity)

1. **Cluster 14_0** (pronephros, 15 som): 32 celltypes, 35 TFs, en2b
2. **Cluster 14_1** (heart_myocardium, 15 som): 32 celltypes, 27 TFs, nr2f1b
3. **Cluster 17_7** (tail_bud, 10 som): 32 celltypes, 47 TFs, hoxb3a

### Peak Celltype Distribution

Most clusters peak in:
1. Primordial germ cells (79 clusters)
2. Enteric neurons (42 clusters)
3. Muscle (34 clusters)
4. Hatching gland (30 clusters)
5. Tail bud (25 clusters)

## Usage

### For Quick Overview

Read **`TOP_DYNAMIC_PROGRAMS_SUMMARY.md`** for curated highlights.

### For Specific Analysis

1. **Temporal analysis**: See `temporal_subGRN_dynamics_all_clusters.md`
2. **Spatial analysis**: See `spatial_celltype_subGRN_dynamics_all_clusters.md`

### For Data Analysis

Load CSV files for quantitative analysis:
```python
import pandas as pd

# Load temporal summary
temporal_df = pd.read_csv('systematic_analysis_temporal_summary.csv')

# Load spatial summary
spatial_df = pd.read_csv('systematic_analysis_spatial_summary.csv')

# Find most dynamic clusters
top_temporal = temporal_df.nlargest(20, 'top_dynamic_score')
top_spatial = spatial_df.nlargest(20, 'n_celltypes')
```

### For LLM Queries

Use the markdown files with LLMs to:
- Query specific TFs across clusters
- Identify regulatory programs in specific celltypes
- Compare temporal vs spatial dynamics
- Find lineage-divergent programs

Example queries:
- "Which clusters show sox13 as the most dynamic TF?"
- "What are the regulatory programs active in hemangioblasts?"
- "Which TFs are ubiquitous across many celltypes?"
- "Show me clusters with peak activity in tail_bud at 10 somites"

## Next Steps (Deferred to Step 4)

The following literature validation analyses were proposed but deferred:
- Use Web Research to validate TF roles
- Check if dynamics are novel or well-established
- Cross-reference zebrafish/human/mouse orthologs
- Identify potentially novel regulatory relationships

This can be performed on specific clusters of interest identified from the systematic analysis.

## Citation

When using this analysis, refer to:
**Zebrahub-Multiome: Uncovering Gene Regulatory Network Dynamics During Zebrafish Embryogenesis**
bioRxiv preprint: https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1

## Contact

For questions about this analysis, contact the Zebrahub-Multiome team.
