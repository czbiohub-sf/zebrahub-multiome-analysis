# SubGRN Analysis Module Documentation

**Author**: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
**Date**: 2025-01-13
**Purpose**: Modular analysis toolkit for extracting and analyzing sub-Gene Regulatory Networks (subGRNs) from zebrafish developmental multiome data

---

## Overview

This documentation describes the refactored SubGRN analysis modules created from the original 6,581-line notebook. The analysis workflow has been decomposed into **9 focused Python modules** organized in `/scripts/subGRN_utils/`, making the code more maintainable, reusable, and easier to understand.

### Module Location

All SubGRN analysis modules are organized in the `scripts/subGRN_utils/` package:
```
scripts/
└── subGRN_utils/
    ├── __init__.py                      # Package initialization
    ├── subgrn_data_loading.py           # Data loading utilities
    ├── subgrn_mesh_construction.py      # Mesh construction
    ├── subgrn_extraction.py             # SubGRN extraction
    ├── subgrn_analysis.py               # Temporal/spatial analysis
    ├── subgrn_visualization.py          # Visualization helpers
    ├── subgrn_temporal_dynamics.py      # Dynamics scoring
    ├── subgrn_similarity_analysis.py    # Similarity analysis
    ├── subgrn_enrichment.py             # TF enrichment
    └── subgrn_cluster_specificity.py    # Cluster specificity
```

### Key Concepts

- **SubGRN**: A subset of a full Gene Regulatory Network (GRN) that corresponds to a specific peak cluster's predicted TF-gene relationships
- **TF-Gene Mesh**: A binarized network representing potential regulatory relationships based on motif enrichment (TFs) and peak-gene links (targets)
- **Temporal Dynamics**: Quantification of how networks change over developmental timepoints
- **Similarity Blocks**: Groups of peak clusters with shared regulatory programs (similar enriched TFs)

---

## Module Structure

### 1. `subgrn_data_loading.py`

**Purpose**: Data loading and validation utilities

**Key Functions**:
- `load_grn_dict_pathlib(base_dir, grn_type)` - Load GRN dictionary from directory structure
- `load_peak_adata(file_path)` - Load peak accessibility AnnData object
- `load_motif_enrichment(file_path)` - Load motif enrichment z-scores
- `load_cluster_pseudobulk_accessibility(file_path)` - Load cluster accessibility matrix
- `validate_grn_dataframe(grn_df)` - Validate GRN structure
- `get_data_path(dataset_name)` - Get standard data paths

**Example Usage**:
```python
from subGRN_utils.subgrn_data_loading import load_grn_dict_pathlib, load_peak_adata

# Load GRNs
grn_dict = load_grn_dict_pathlib("/path/to/grns", grn_type="filtered")
# Returns: Dict[(celltype, timepoint), pd.DataFrame]

# Load peak data
adata = load_peak_adata("/path/to/peaks.h5ad")
```

---

### 2. `subgrn_mesh_construction.py`

**Purpose**: Build TF-gene mesh networks from peak clusters

**Key Functions**:
- `create_cluster_tf_matrix(clusters_tfs_dict)` - Binary cluster×TF matrix
- `create_cluster_gene_matrix(clusters_genes_dict)` - Binary cluster×gene matrix
- `create_all_cluster_meshes(clusters_tfs, clusters_genes)` - Generate all meshes
- `compute_mesh_statistics(cluster_meshes)` - Summary statistics
- `filter_motifs_by_threshold(clust_by_motifs, threshold)` - Binarize enrichment

**Example Usage**:
```python
from subGRN_utils.subgrn_mesh_construction import create_all_cluster_meshes, compute_mesh_statistics

# Create meshes
meshes = create_all_cluster_meshes(clusters_tfs_dict, clusters_genes_dict)
# Returns: Dict[cluster_id, pd.DataFrame (TFs × genes)]

# Get statistics
stats = compute_mesh_statistics(meshes)
print(f"Mean edges per cluster: {stats['n_edges'].mean():.1f}")
```

**Mesh Format**: Each mesh is a binary TF×gene matrix where 1 indicates a predicted regulatory relationship to be validated against actual GRNs.

---

### 3. `subgrn_extraction.py`

**Purpose**: Extract subGRNs from full GRNs using mesh predictions

**Key Functions**:
- `extract_subGRN_from_cluster(grn_df, mesh, cluster_id)` - Extract single subGRN
- `extract_all_cluster_subGRNs(grn_df, cluster_dict)` - Extract all subGRNs
- `extract_subgrn_for_celltype_timepoint(grn_dict, celltype, timepoint, pairs)` - Extract specific context
- `extract_subgrn_metrics(cluster_id, celltype, grn_dict, meshes, pairs)` - Comprehensive metrics
- `get_predicted_pairs_from_mesh(mesh_matrix)` - Convert mesh to pair set
- `count_subgrn_edges_per_timepoint(subgrns_dict)` - Temporal edge counts

**Example Usage**:
```python
from subGRN_utils.subgrn_extraction import extract_subgrn_metrics, get_predicted_pairs_from_mesh

# Get predicted pairs
predicted_pairs = get_predicted_pairs_from_mesh(cluster_mesh)

# Extract comprehensive metrics
metrics = extract_subgrn_metrics(
    cluster_id='26_11',
    celltype_of_interest='hemangioblasts',
    grn_dict=grn_dict,
    cluster_tf_gene_matrices=meshes,
    predicted_pairs=predicted_pairs
)

print(f"Total nodes: {metrics['total_nodes']}")
print(f"Total edges: {metrics['total_edges']}")
print(f"Complexity reduction: {metrics['complexity_reduction']:.1f}%")
```

**Key Metrics Returned**:
- Node counts (total, TF-only, target-only, dual)
- Edge counts across timepoints
- Complexity reduction vs original mesh
- SubGRN dataframes per timepoint

---

### 4. `subgrn_analysis.py`

**Purpose**: Temporal and spatial analysis of subGRNs

**Key Functions**:
- `analyze_single_timepoint(grn_dict, timepoint, predicted_pairs, cluster_id)` - Cross-celltype comparison
- `compare_celltypes_similarity(celltype_subgrns, predicted_pairs, timepoint)` - Jaccard similarity
- `compare_across_timepoints(grn_dict, predicted_pairs, cluster_id)` - Multi-timepoint analysis
- `track_celltype_across_time(timepoint_results, cluster_id)` - Temporal tracking
- `summarize_analysis(timepoint_results, temporal_tracking, cluster_id)` - Summary statistics
- `analyze_edge_types(grn_dict, predicted_pairs, celltype)` - Activation vs repression

**Example Usage**:
```python
from subGRN_utils.subgrn_analysis import compare_across_timepoints, track_celltype_across_time

# Analyze across all timepoints
tp_results = compare_across_timepoints(grn_dict, predicted_pairs, '26_11')

# Track specific celltype over time
tracking = track_celltype_across_time(tp_results, '26_11', plot=True)

# Shows how implementation rate and edge count change developmentally
```

**Analysis Outputs**:
- Implementation rates (what fraction of mesh is realized)
- Mean regulatory strength
- Jaccard similarity between celltypes
- Temporal evolution plots

---

### 5. `subgrn_visualization.py`

**Purpose**: Network visualization helper functions

**Key Functions**:
- `classify_nodes(subgrns)` - Classify as TF-only, target-only, or dual
- `get_node_colors(nodes, tf_only, target_only, dual)` - Color assignment
- `compute_edge_widths(edges, weights, max_width, min_width)` - Scale by strength
- `separate_edges_by_sign(subgrn, coef_column)` - Activation vs repression
- `create_legend_elements()` - Standard legend for plots
- `save_figure_publication_quality(fig, filename, dpi, formats)` - Multi-format export

**Example Usage**:
```python
from subGRN_utils.subgrn_visualization import classify_nodes, separate_edges_by_sign, create_legend_elements

# Classify nodes
tf_only, target_only, dual = classify_nodes(subgrns_by_timepoint)

# Separate edge types
pos_edges, neg_edges, weights = separate_edges_by_sign(subgrn_df, 'coef_mean')

# Create legend
legend = create_legend_elements()
```

**Note**: The full `plot_subgrns_over_time()` function (~500 lines) should be imported from the original notebook until fully refactored:

```python
from notebooks.Fig3_GRN_dynamics.EDA_extract_subGRN_reg_programs_Take2 import plot_subgrns_over_time
```

---

### 6. `subgrn_temporal_dynamics.py`

**Purpose**: Score and rank subGRNs by temporal dynamics

**Key Functions**:
- `gini_coefficient(values)` - Measure accessibility concentration
- `find_most_accessible_celltype(cluster_id, df_clusters_groups)` - Identify peak accessibility
- `compute_temporal_dynamics_score(cluster_id, celltype, grn_dict, meshes)` - Dynamic evolution score
- `rank_clusters_by_temporal_dynamics(df_clusters_groups, grn_dict, meshes)` - Comprehensive ranking

**Dynamics Score Formula**:
```
score = 0.4 × dev_tf_turnover + 0.3 × edge_turnover + 0.2 × tf_turnover + 0.1 × temporal_variance
```

**Score Components**:
- `dev_tf_turnover`: Fraction of TFs that appear over time (new developmental regulators)
- `edge_turnover`: Fraction of edges that change between timepoints
- `tf_turnover`: Overall TF composition changes
- `temporal_variance`: Variability in edge count over time

**Example Usage**:
```python
from subGRN_utils.subgrn_temporal_dynamics import rank_clusters_by_temporal_dynamics

# Rank all clusters
df_ranked = rank_clusters_by_temporal_dynamics(
    df_clusters_groups=accessibility_matrix,
    grn_dict=grn_dict,
    cluster_tf_gene_matrices=meshes,
    min_edges=5,
    min_timepoints=3,
    top_n=20
)

# Top candidate
top = df_ranked.iloc[0]
print(f"Cluster {top['cluster_id']}: dynamics={top['dynamics_score']:.3f}")
```

**Interpretation**:
- High scores (>0.5): Dynamic regulatory programs with significant temporal evolution
- Key for identifying developmental transitions and cell fate decisions

---

### 7. `subgrn_similarity_analysis.py`

**Purpose**: Identify clusters with shared regulatory programs

**Key Functions**:
- `cluster_similarity_analysis(clust_by_motifs, threshold)` - Pairwise Jaccard similarity
- `analyze_tf_sharing(cluster_tf_matrix)` - TF distribution across clusters
- `create_cluster_similarity_heatmap(matrix, top_n, feature_type)` - Hierarchical heatmap
- `analyze_similarity_distribution(cluster_tf_matrix)` - Threshold recommendations
- `find_dense_similarity_regions(matrix, min_threshold, avg_threshold)` - Block detection

**Example Usage**:
```python
from subGRN_utils.subgrn_similarity_analysis import find_dense_similarity_regions, analyze_tf_sharing

# Analyze TF sharing
tf_counts, top_tfs = analyze_tf_sharing(cluster_tf_matrix, savefig=True)

# Find dense similarity blocks
sim, names, linkage, blocks, details = find_dense_similarity_regions(
    cluster_feature_matrix=cluster_tf_matrix,
    min_similarity_threshold=0.15,
    average_similarity_threshold=0.35,
    min_block_size=4,
    savefig=True,
    filename="similarity_blocks.png"
)

print(f"Found {len(blocks)} high-quality similarity blocks")
```

**Block Detection Algorithm**:
1. Compute pairwise Jaccard similarity
2. Seed-and-grow: Start with highest similarity pairs
3. Add clusters that are similar to ALL block members
4. Filter by minimum and average similarity thresholds

---

### 8. `subgrn_enrichment.py`

**Purpose**: TF enrichment analysis in similarity blocks

**Key Functions**:
- `analyze_tf_enrichment_in_blocks(cluster_tf_matrix, blocks_data)` - Hypergeometric test
- `visualize_tf_enrichment(enrichment_results, top_n)` - Comprehensive plots
- `create_block_tf_summary(enrichment_results, blocks_data)` - Text summaries
- `find_shared_vs_specific_tfs(enrichment_results)` - Identify sharing patterns
- `create_enrichment_ranking_table(enrichment_results, output_file)` - Export rankings

**Example Usage**:
```python
from subGRN_utils.subgrn_enrichment import (
    analyze_tf_enrichment_in_blocks,
    visualize_tf_enrichment,
    find_shared_vs_specific_tfs
)

# Create block dictionary
blocks_data = {
    'Block1': ['0_0', '0_1', '0_2'],
    'Block2': ['1_0', '1_1', '1_2']
}

# Analyze enrichment
results = analyze_tf_enrichment_in_blocks(
    cluster_tf_matrix=cluster_tf_matrix,
    blocks_data=blocks_data,
    min_frequency=0.3,
    min_enrichment_ratio=1.5,
    statistical_test='hypergeometric'
)

# Visualize
visualize_tf_enrichment(results, top_n=10)

# Find shared vs specific
shared, specific = find_shared_vs_specific_tfs(results)
```

**Statistical Test**: Hypergeometric test assesses whether a TF is overrepresented in a block compared to background frequency across all clusters.

**Output**: For each block, returns top enriched TFs with:
- Frequency in block vs background
- Enrichment ratio
- P-value and combined score
- Highly specific TFs (high enrichment, low background)

---

### 9. `subgrn_cluster_specificity.py`

**Purpose**: Calculate cluster specificity based on accessibility

**Key Functions**:
- `calculate_cluster_specificity(df_clusters_groups, top_n)` - Multi-metric specificity
- `visualize_specificity_distribution(df_specificity)` - Distribution plots
- `identify_highly_specific_clusters(df_specificity, thresholds)` - Filter by criteria
- `annotate_block_clusters(block_name, cluster_list, info)` - Format annotations
- `analyze_specificity_by_celltype(df_clusters_groups, df_specificity)` - Celltype patterns
- `compare_specificity_across_blocks(df_specificity, blocks_data)` - Block comparison

**Specificity Metrics**:
1. **Specificity Score**: Fraction of signal in top N groups (0-1, higher = more specific)
2. **Fold Enrichment**: Ratio of top mean to rest mean
3. **Normalized Entropy**: Shannon entropy normalized by max entropy (0-1, lower = more specific)

**Example Usage**:
```python
from subGRN_utils.subgrn_cluster_specificity import (
    calculate_cluster_specificity,
    identify_highly_specific_clusters
)

# Calculate specificity
df_spec = calculate_cluster_specificity(accessibility_matrix, top_n=2)

# Identify highly specific clusters
high_spec = identify_highly_specific_clusters(
    df_spec,
    specificity_threshold=0.6,
    fold_enrichment_threshold=5.0
)

print(f"Found {len(high_spec)} highly specific clusters")
print(high_spec.head())
```

**Use Cases**:
- Identify lineage-specific regulatory programs
- Filter for clusters with focused activity
- Prioritize biologically interpretable subGRNs

---

## Complete Workflow Example

See `example_subgrn_analysis_workflow.py` for a comprehensive end-to-end example demonstrating:

1. Data loading
2. Mesh construction
3. SubGRN extraction
4. Temporal dynamics ranking
5. Similarity analysis and block detection
6. TF enrichment analysis
7. Cluster specificity calculation
8. Visualization preparation

---

## Data Requirements

### Input Data Files

1. **GRN Dictionary**:
   - Location: `processed_data/11_celloracle_grn_by_cell_types/filtered/`
   - Format: CSV files organized as `timepoint_XX/celltype.csv`
   - Columns: `source`, `target`, `coef_mean`, `coef_abs`, etc.

2. **Peak Accessibility Data**:
   - Location: `annotated_data/objects_v2/adata_atac_annotated_v2.h5ad`
   - Format: AnnData object with peak metadata
   - Required: `obs['leiden_unified']` for cluster IDs

3. **Motif Enrichment**:
   - Location: `processed_data/13_peak_umap_analysis/maelstrom_*/maelstrom.zscores.txt`
   - Format: Clusters (rows) × Motifs (columns) z-scores

4. **Cluster Accessibility**:
   - Location: `annotated_data/objects_v2/leiden_fine_by_pseudobulk.csv`
   - Format: Clusters (rows) × Pseudobulk groups (columns)

5. **TF and Gene Dictionaries**:
   - Format: Python pickle files
   - `clusters_tfs_dict`: {cluster_id: [TF list]}
   - `clusters_genes_dict`: {cluster_id: [gene list]}

### Output Files

All outputs saved to `figures/sub_GRNs_reg_programs/`:
- Temporal dynamics plots
- Similarity heatmaps
- TF enrichment visualizations
- Specificity distributions
- SubGRN network plots
- Ranking tables (CSV)

---

## Dependencies

```python
# Core scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Network analysis
import networkx as nx

# Statistical analysis
from scipy.stats import hypergeom, chi2_contingency
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import jaccard_score

# Single-cell analysis
import scanpy as sc

# File handling
from pathlib import Path
import pickle

# Type hints
from typing import Dict, List, Tuple, Set, Optional, Union
```

---

## Best Practices

### 1. Data Validation
Always validate input data structure:
```python
from subGRN_utils.subgrn_data_loading import validate_grn_dataframe

for (ct, tp), grn_df in grn_dict.items():
    if not validate_grn_dataframe(grn_df):
        print(f"Warning: Invalid GRN for {ct} at {tp}")
```

### 2. Logging
Enable logging to track analysis progress:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### 3. Memory Management
For large datasets, process clusters in batches:
```python
# Process in chunks
cluster_ids = list(meshes.keys())
batch_size = 50

for i in range(0, len(cluster_ids), batch_size):
    batch = cluster_ids[i:i+batch_size]
    # Process batch...
```

### 4. Parameter Tuning
Key parameters to adjust:
- **Temporal dynamics**: `min_edges`, `min_timepoints` (balance sensitivity vs noise)
- **Similarity blocks**: `min_similarity_threshold`, `average_similarity_threshold` (analyze distribution first)
- **Enrichment**: `min_frequency`, `min_enrichment_ratio` (adjust based on block size)
- **Specificity**: `specificity_threshold`, `fold_enrichment_threshold` (depends on data sparsity)

### 5. Reproducibility
Save analysis parameters and intermediate results:
```python
import json

params = {
    'min_edges': 5,
    'min_timepoints': 3,
    'similarity_threshold': 0.15,
    'enrichment_min_freq': 0.3
}

with open('analysis_params.json', 'w') as f:
    json.dump(params, f, indent=2)
```

---

## Troubleshooting

### Common Issues

**1. Missing clusters in matrices**
```python
# Check cluster overlap
available_clusters = set(cluster_tf_matrix.index)
mesh_clusters = set(meshes.keys())
missing = mesh_clusters - available_clusters
print(f"Missing clusters: {len(missing)}")
```

**2. Empty subGRNs**
- Check if celltype-timepoint combination exists in GRN dict
- Verify mesh has non-zero edges
- Lower `min_edges` threshold if too restrictive

**3. Memory errors with large datasets**
- Process clusters in batches
- Use `top_n_clusters` parameter in similarity functions
- Clear intermediate results: `del large_dataframe`

**4. Slow similarity computation**
- Pre-filter to top clusters by accessibility
- Use sparse matrix representations where possible
- Consider parallelization for independent computations

---

## Citation

If you use these modules in your analysis, please cite:

> Kim YJ, et al. (2024). Zebrahub-Multiome: Uncovering Gene Regulatory Network Dynamics During Zebrafish Embryogenesis. bioRxiv. doi: 10.1101/2024.10.18.618987

---

## Contact and Support

For questions or issues:
1. Check this documentation
2. Review the example workflow notebook
3. Examine function docstrings (`help(function_name)`)
4. Inspect the original notebook for additional context

**Module Author**: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
**Date**: 2025-01-13
**Version**: 1.0
