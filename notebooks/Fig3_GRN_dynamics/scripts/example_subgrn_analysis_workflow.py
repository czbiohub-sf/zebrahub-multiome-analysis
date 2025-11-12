# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example SubGRN Analysis Workflow
#
# This notebook demonstrates how to use the refactored SubGRN analysis modules
# for extracting and analyzing regulatory programs from peak clusters.
#
# **Author**: Generated from EDA_extract_subGRN_reg_programs_Take2.py refactoring
# **Date**: 2025-01-13
#
# ## Overview
#
# The analysis workflow consists of several steps:
# 1. Load data (GRNs, peaks, motif enrichment)
# 2. Construct TF-gene mesh networks for each peak cluster
# 3. Extract subGRNs by intersecting mesh with full GRNs
# 4. Analyze temporal dynamics and rank candidates
# 5. Perform similarity analysis and identify blocks
# 6. Analyze TF enrichment in similarity blocks
# 7. Calculate cluster specificity metrics
# 8. Visualize selected subGRNs

# %%
# Import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure plotting
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set_style('whitegrid')

# %% [markdown]
# ## Step 1: Import SubGRN Analysis Modules

# %%
# Add scripts directory to path if needed
import sys
sys.path.insert(0, '../../scripts')

# Import data loading
from subGRN_utils.subgrn_data_loading import (
    load_grn_dict_pathlib,
    load_peak_adata,
    load_motif_enrichment,
    load_cluster_pseudobulk_accessibility,
    get_data_path
)

# Import mesh construction
from subGRN_utils.subgrn_mesh_construction import (
    create_cluster_tf_matrix,
    create_cluster_gene_matrix,
    create_all_cluster_meshes,
    compute_mesh_statistics
)

# Import subGRN extraction
from subGRN_utils.subgrn_extraction import (
    extract_subGRN_from_cluster,
    extract_subgrn_metrics,
    get_predicted_pairs_from_mesh,
    count_subgrn_edges_per_timepoint
)

# Import temporal and spatial analysis
from subGRN_utils.subgrn_analysis import (
    analyze_single_timepoint,
    compare_celltypes_similarity,
    compare_across_timepoints,
    track_celltype_across_time,
    summarize_analysis
)

# Import temporal dynamics scoring
from subGRN_utils.subgrn_temporal_dynamics import (
    gini_coefficient,
    find_most_accessible_celltype,
    compute_temporal_dynamics_score,
    rank_clusters_by_temporal_dynamics
)

# Import similarity analysis
from subGRN_utils.subgrn_similarity_analysis import (
    cluster_similarity_analysis,
    analyze_tf_sharing,
    create_cluster_similarity_heatmap,
    analyze_similarity_distribution,
    find_dense_similarity_regions
)

# Import enrichment analysis
from subGRN_utils.subgrn_enrichment import (
    analyze_tf_enrichment_in_blocks,
    visualize_tf_enrichment,
    create_block_tf_summary,
    find_shared_vs_specific_tfs,
    create_enrichment_ranking_table
)

# Import cluster specificity
from subGRN_utils.subgrn_cluster_specificity import (
    calculate_cluster_specificity,
    visualize_specificity_distribution,
    identify_highly_specific_clusters,
    annotate_block_clusters
)

# Import visualization helpers
from subGRN_utils.subgrn_visualization import (
    classify_nodes,
    get_node_colors,
    separate_edges_by_sign,
    create_legend_elements,
    save_figure_publication_quality
)

logger.info("Successfully imported all SubGRN analysis modules!")

# %% [markdown]
# ## Step 2: Define Data Paths and Load Data

# %%
# Define base paths
BASE_DIR = Path("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data")
PROCESSED_DIR = BASE_DIR / "processed_data"
ANNOTATED_DIR = BASE_DIR / "annotated_data"
OUTPUT_DIR = Path("../../figures/sub_GRNs_reg_programs")

# Create output directory if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# %%
# Load GRN dictionary
grn_base_dir = PROCESSED_DIR / "11_celloracle_grn_by_cell_types"
grn_dict = load_grn_dict_pathlib(str(grn_base_dir), grn_type="filtered")

logger.info(f"Loaded {len(grn_dict)} GRNs")
logger.info(f"Available (celltype, timepoint) combinations: {len(grn_dict)}")

# Check available celltypes and timepoints
celltypes = sorted(set([ct for ct, tp in grn_dict.keys()]))
timepoints = sorted(set([tp for ct, tp in grn_dict.keys()]))

logger.info(f"Cell types: {len(celltypes)}")
logger.info(f"Timepoints: {timepoints}")

# %%
# Load peak data
peak_file = ANNOTATED_DIR / "objects_v2" / "adata_atac_annotated_v2.h5ad"
adata_peaks = load_peak_adata(str(peak_file))

logger.info(f"Loaded peak data: {adata_peaks.shape}")
logger.info(f"Peak clusters: {adata_peaks.obs['leiden_unified'].nunique()}")

# %%
# Load motif enrichment
motif_file = PROCESSED_DIR / "13_peak_umap_analysis" / "maelstrom_640K_leiden_unified_cisBP_ver2_Danio_rerio_output" / "maelstrom.zscores.txt"
clust_by_motifs = load_motif_enrichment(str(motif_file))

logger.info(f"Loaded motif enrichment: {clust_by_motifs.shape}")
logger.info(f"  {len(clust_by_motifs)} clusters × {len(clust_by_motifs.columns)} motifs")

# %%
# Load cluster-by-pseudobulk accessibility
accessibility_file = ANNOTATED_DIR / "objects_v2" / "leiden_fine_by_pseudobulk.csv"
df_clusters_groups = load_cluster_pseudobulk_accessibility(str(accessibility_file))

logger.info(f"Loaded accessibility matrix: {df_clusters_groups.shape}")
logger.info(f"  {len(df_clusters_groups)} clusters × {len(df_clusters_groups.columns)} groups")

# %% [markdown]
# ## Step 3: Construct TF-Gene Mesh Networks
#
# Create predicted TF-gene relationships for each peak cluster based on:
# - Enriched motifs (TFs)
# - Linked genes (from peak-gene associations)

# %%
# Load or create TF and gene dictionaries
# NOTE: These should be loaded from your actual data
# For this example, assume they exist as:
# - clusters_tfs_dict: {cluster_id: [list of TFs]}
# - clusters_genes_dict: {cluster_id: [list of genes]}

# Example loading (adjust paths as needed):
import pickle

mesh_dir = PROCESSED_DIR / "14_subGRN_analysis"

# Load TF dictionary
with open(mesh_dir / "clusters_tfs_dict.pkl", "rb") as f:
    clusters_tfs_dict = pickle.load(f)

# Load genes dictionary
with open(mesh_dir / "clusters_genes_dict.pkl", "rb") as f:
    clusters_genes_dict = pickle.load(f)

logger.info(f"Loaded TF dictionary: {len(clusters_tfs_dict)} clusters")
logger.info(f"Loaded genes dictionary: {len(clusters_genes_dict)} clusters")

# %%
# Create cluster-by-TF matrix
cluster_tf_matrix, all_tfs = create_cluster_tf_matrix(clusters_tfs_dict)

logger.info(f"Created cluster-TF matrix: {cluster_tf_matrix.shape}")
logger.info(f"Total unique TFs: {len(all_tfs)}")

# %%
# Create cluster-by-gene matrix
cluster_gene_matrix, all_genes = create_cluster_gene_matrix(clusters_genes_dict)

logger.info(f"Created cluster-gene matrix: {cluster_gene_matrix.shape}")
logger.info(f"Total unique genes: {len(all_genes)}")

# %%
# Create TF-gene mesh for all clusters
cluster_tf_gene_matrices = create_all_cluster_meshes(clusters_tfs_dict, clusters_genes_dict)

logger.info(f"Created {len(cluster_tf_gene_matrices)} TF-gene mesh networks")

# %%
# Compute mesh statistics
df_mesh_stats = compute_mesh_statistics(cluster_tf_gene_matrices)

logger.info("\nMesh statistics summary:")
logger.info(f"  Mean TFs per cluster: {df_mesh_stats['n_TFs'].mean():.1f}")
logger.info(f"  Mean genes per cluster: {df_mesh_stats['n_genes'].mean():.1f}")
logger.info(f"  Mean edges per cluster: {df_mesh_stats['n_edges'].mean():.1f}")

# Display top clusters by complexity
display(df_mesh_stats.sort_values('n_edges', ascending=False).head(10))

# %% [markdown]
# ## Step 4: Rank Clusters by Temporal Dynamics
#
# Identify biologically interesting subGRNs based on how dynamically they change
# over developmental time.

# %%
# Rank clusters by temporal dynamics
df_ranked = rank_clusters_by_temporal_dynamics(
    df_clusters_groups=df_clusters_groups,
    grn_dict=grn_dict,
    cluster_tf_gene_matrices=cluster_tf_gene_matrices,
    min_edges=5,
    min_timepoints=3,
    top_n=20
)

logger.info(f"\nTop 20 clusters by temporal dynamics:")
display(df_ranked)

# %%
# Visualize dynamics score distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Dynamics score distribution
ax = axes[0, 0]
ax.hist(df_ranked['dynamics_score'], bins=20, edgecolor='black', alpha=0.7)
ax.set_xlabel('Dynamics Score')
ax.set_ylabel('Count')
ax.set_title('Distribution of Temporal Dynamics Scores')
ax.axvline(df_ranked['dynamics_score'].median(), color='red', linestyle='--',
           label=f'Median: {df_ranked["dynamics_score"].median():.3f}')
ax.legend()

# Component scores
ax = axes[0, 1]
components = ['tf_turnover', 'edge_turnover', 'dev_tf_turnover', 'temporal_variance']
component_means = [df_ranked[c].mean() for c in components]
ax.bar(range(len(components)), component_means, color='steelblue', alpha=0.7)
ax.set_xticks(range(len(components)))
ax.set_xticklabels(['TF\nTurnover', 'Edge\nTurnover', 'Dev TF\nTurnover', 'Temporal\nVariance'])
ax.set_ylabel('Mean Score')
ax.set_title('Mean Component Scores')

# Dynamics vs edges
ax = axes[1, 0]
scatter = ax.scatter(df_ranked['max_edges'], df_ranked['dynamics_score'],
                    c=df_ranked['n_timepoints'], cmap='viridis', alpha=0.6, s=100)
ax.set_xlabel('Max Edges')
ax.set_ylabel('Dynamics Score')
ax.set_title('Dynamics Score vs Network Size')
plt.colorbar(scatter, ax=ax, label='N Timepoints')

# Gini coefficient vs dynamics
ax = axes[1, 1]
ax.scatter(df_ranked['gini_coefficient'], df_ranked['dynamics_score'], alpha=0.6, s=100)
ax.set_xlabel('Gini Coefficient (Accessibility Concentration)')
ax.set_ylabel('Dynamics Score')
ax.set_title('Dynamics vs Accessibility Specificity')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temporal_dynamics_overview.pdf", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 5: Analyze Top Candidate SubGRN
#
# Extract detailed metrics for the top-ranked cluster

# %%
# Select top candidate
top_cluster = df_ranked.iloc[0]
cluster_id = top_cluster['cluster_id']
celltype = top_cluster['celltype']

logger.info(f"\n=== Analyzing Top Candidate ===")
logger.info(f"Cluster: {cluster_id}")
logger.info(f"Cell type: {celltype}")
logger.info(f"Dynamics score: {top_cluster['dynamics_score']:.3f}")
logger.info(f"Accessibility: {top_cluster['accessibility']:.3f}")
logger.info(f"Gini coefficient: {top_cluster['gini_coefficient']:.3f}")

# %%
# Get predicted pairs from mesh
cluster_mesh = cluster_tf_gene_matrices[cluster_id]
predicted_pairs = get_predicted_pairs_from_mesh(cluster_mesh)

logger.info(f"\nMesh network:")
logger.info(f"  {len(cluster_mesh.index)} TFs × {len(cluster_mesh.columns)} genes")
logger.info(f"  {len(predicted_pairs)} predicted TF-target pairs")

# %%
# Extract comprehensive metrics
metrics = extract_subgrn_metrics(
    cluster_id=cluster_id,
    celltype_of_interest=celltype,
    grn_dict=grn_dict,
    cluster_tf_gene_matrices=cluster_tf_gene_matrices,
    predicted_pairs=predicted_pairs
)

logger.info(f"\nSubGRN composition:")
logger.info(f"  Total nodes: {metrics['total_nodes']}")
logger.info(f"  TFs: {metrics['total_tfs']}")
logger.info(f"  Targets: {metrics['total_targets']}")
logger.info(f"  Dual TF/Target nodes: {metrics['n_dual_nodes']}")
logger.info(f"  Total edges: {metrics['total_edges']}")
logger.info(f"  Complexity reduction: {metrics['complexity_reduction']:.1f}%")

# %%
# Count edges per timepoint
subgrns = metrics['subgrns_by_timepoint']
df_edge_counts = count_subgrn_edges_per_timepoint(subgrns)

# Plot temporal evolution
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_edge_counts['timepoint'], df_edge_counts['edge_count'],
        marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Developmental Timepoint')
ax.set_ylabel('Number of Edges')
ax.set_title(f'SubGRN Evolution: {cluster_id} - {celltype}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"subgrn_evolution_{cluster_id}_{celltype}.pdf")
plt.show()

# %% [markdown]
# ## Step 6: Similarity Analysis
#
# Identify clusters with shared regulatory programs

# %%
# Analyze TF sharing across clusters
tf_sharing_stats, most_shared_tfs = analyze_tf_sharing(
    cluster_tf_matrix,
    savefig=True,
    filename=str(OUTPUT_DIR / "tf_sharing_distribution.pdf")
)

# %%
# Analyze similarity distribution
stats, primary_cutoff, internal_cutoff = analyze_similarity_distribution(cluster_tf_matrix)

logger.info(f"\nRecommended thresholds:")
logger.info(f"  Primary cutoff: {primary_cutoff:.3f}")
logger.info(f"  Internal similarity: {internal_cutoff:.3f}")

# %%
# Find dense similarity regions
sim_matrix, cluster_names, linkage_info, dense_blocks, block_details = find_dense_similarity_regions(
    cluster_feature_matrix=cluster_tf_matrix,
    top_n_clusters=402,
    feature_type="TFs",
    min_similarity_threshold=0.15,
    average_similarity_threshold=0.35,
    min_block_size=4,
    max_block_size=100,
    savefig=True,
    filename=str(OUTPUT_DIR / "dense_similarity_regions.png"),
    hide_axis_labels=True,
    cmap="Blues",
    gamma=0.6,
    show_blocks=True
)

logger.info(f"\nFound {len(dense_blocks)} high-quality similarity blocks")

# %%
# Convert blocks to dictionary format for enrichment analysis
blocks_data = {}
for i, clusters in enumerate(dense_blocks):
    block_name = f"HQ{i+1}"
    blocks_data[block_name] = clusters

logger.info(f"Created block dictionary with {len(blocks_data)} blocks")

# %% [markdown]
# ## Step 7: TF Enrichment Analysis
#
# Identify block-defining transcription factors

# %%
# Analyze TF enrichment in blocks
enrichment_results = analyze_tf_enrichment_in_blocks(
    cluster_tf_matrix=cluster_tf_matrix,
    blocks_data=blocks_data,
    min_frequency=0.3,
    min_enrichment_ratio=1.5,
    max_tfs_per_block=15,
    statistical_test='hypergeometric'
)

logger.info(f"\nEnrichment analysis complete for {len(enrichment_results)} blocks")

# %%
# Visualize TF enrichment
visualize_tf_enrichment(enrichment_results, top_n=10)

# %%
# Create block-TF summary
block_summary = create_block_tf_summary(enrichment_results, blocks_data)

# %%
# Find shared vs specific TFs
shared_tfs, specific_tfs = find_shared_vs_specific_tfs(enrichment_results)

# %%
# Create comprehensive ranking table
df_enrichment_ranked = create_enrichment_ranking_table(
    enrichment_results,
    output_file=str(OUTPUT_DIR / "tf_enrichment_ranking.csv")
)

# %% [markdown]
# ## Step 8: Cluster Specificity Analysis

# %%
# Calculate cluster specificity
df_specificity = calculate_cluster_specificity(df_clusters_groups, top_n=2)

logger.info(f"\nCalculated specificity for {len(df_specificity)} clusters")

# %%
# Visualize specificity distribution
visualize_specificity_distribution(
    df_specificity,
    savefig=True,
    filename=str(OUTPUT_DIR / "cluster_specificity_distribution.pdf")
)

# %%
# Identify highly specific clusters
high_spec_clusters = identify_highly_specific_clusters(
    df_specificity,
    specificity_threshold=0.6,
    fold_enrichment_threshold=5.0,
    min_signal=0.01
)

logger.info(f"\nFound {len(high_spec_clusters)} highly specific clusters")
display(high_spec_clusters.head(10))

# %% [markdown]
# ## Step 9: Visualize Selected SubGRN
#
# Create network visualization for top candidate
#
# **Note**: The full `plot_subgrns_over_time()` function should be imported
# from the original notebook until fully refactored:
#
# ```python
# from notebooks.Fig3_GRN_dynamics.EDA_extract_subGRN_reg_programs_Take2 import plot_subgrns_over_time
# ```

# %%
# Extract subGRNs for visualization
subgrns_by_timepoint = metrics['subgrns_by_timepoint']

# Classify nodes
tf_only, target_only, tf_and_target = classify_nodes(subgrns_by_timepoint)

logger.info(f"\nNode classification:")
logger.info(f"  TF-only: {len(tf_only)}")
logger.info(f"  Target-only: {len(target_only)}")
logger.info(f"  Dual TF/Target: {len(tf_and_target)}")

# %%
# Create legend elements for any custom plots
legend = create_legend_elements()

logger.info("\nVisualization helpers ready!")
logger.info("Use plot_subgrns_over_time() from original notebook for full network plots")

# %% [markdown]
# ## Summary
#
# This workflow demonstrated:
# 1. ✅ Loading data using refactored modules
# 2. ✅ Creating TF-gene mesh networks
# 3. ✅ Ranking clusters by temporal dynamics
# 4. ✅ Extracting detailed subGRN metrics
# 5. ✅ Analyzing cluster similarity and identifying blocks
# 6. ✅ Performing TF enrichment analysis
# 7. ✅ Calculating cluster specificity
# 8. ✅ Preparing for network visualization
#
# All analysis functions are now modular and reusable!

# %%
logger.info("\n" + "="*60)
logger.info("WORKFLOW COMPLETE!")
logger.info("="*60)
logger.info(f"\nKey results saved to: {OUTPUT_DIR}")
logger.info("\nTop candidate:")
logger.info(f"  Cluster: {cluster_id}")
logger.info(f"  Cell type: {celltype}")
logger.info(f"  Dynamics score: {top_cluster['dynamics_score']:.3f}")
logger.info(f"  Total edges: {metrics['total_edges']}")
logger.info(f"  Complexity reduction: {metrics['complexity_reduction']:.1f}%")
