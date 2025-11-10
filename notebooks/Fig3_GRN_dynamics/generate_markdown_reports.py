#!/usr/bin/env python
"""
Generate comprehensive markdown reports from systematic analysis results

Generates:
1. temporal_subGRN_dynamics_all_clusters.md - Full temporal report
2. spatial_celltype_subGRN_dynamics_all_clusters.md - Full spatial report
3. TOP_DYNAMIC_PROGRAMS_SUMMARY.md - Curated highlights
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

print("="*80)
print("GENERATING MARKDOWN REPORTS")
print("="*80)

figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"

# Load summary CSVs
print("\nLoading summary data...")
temporal_df = pd.read_csv(f"{figpath}systematic_analysis_temporal_summary.csv")
spatial_df = pd.read_csv(f"{figpath}systematic_analysis_spatial_summary.csv")

# Re-run analysis to get detailed results (needed for markdown)
# This is a simplified re-run just to regenerate the detailed data structures
print("Re-loading detailed analysis results...")

import pickle
from pathlib import Path

# Load data (same as main script)
cluster_tf_gene_matrices_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl"
with open(cluster_tf_gene_matrices_path, 'rb') as f:
    cluster_tf_gene_matrices = pickle.load(f)

def load_grn_dict_pathlib(base_dir, grn_type="filtered"):
    grn_dict = {}
    base_path = Path(base_dir) / grn_type
    csv_files = list(base_path.glob("*/*.csv"))
    for csv_file in csv_files:
        timepoint_dir = csv_file.parent.name
        timepoint = timepoint_dir.split('_')[1] if 'timepoint_' in timepoint_dir else timepoint_dir
        celltype = csv_file.stem
        grn_df = pd.read_csv(csv_file)
        grn_dict[(celltype, timepoint)] = grn_df
    return grn_dict

grn_dict = load_grn_dict_pathlib(
    base_dir="/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/grn_exported/",
    grn_type="filtered"
)

TIMEPOINTS = ['00', '05', '10', '15', '20', '30']

print(f"✓ Loaded data for {len(temporal_df)} clusters\n")

# ============================================================================
# GENERATE TEMPORAL DYNAMICS MARKDOWN
# ============================================================================

print("Generating temporal dynamics markdown...")

with open(f"{figpath}temporal_subGRN_dynamics_all_clusters.md", 'w') as f:
    f.write("# Temporal SubGRN Dynamics - All 346 Clusters\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## Overview\n\n")
    f.write(f"This report documents temporal dynamics of subGRNs across all {len(temporal_df)} clusters with TF-gene matrices.\n\n")
    f.write("For each cluster, we analyzed subGRN changes across developmental timepoints (0-30 somites) at the peak celltype.\n\n")

    # Summary statistics
    f.write("### Summary Statistics\n\n")
    f.write(f"- **Total clusters analyzed:** {len(temporal_df)}\n")
    f.write(f"- **Clusters with active subGRNs:** {len(temporal_df[temporal_df['total_edges'] > 0])}\n")
    f.write(f"- **Clusters with no edges:** {len(temporal_df[temporal_df['total_edges'] == 0])}\n")
    f.write(f"- **Mean network size:** {temporal_df['total_nodes'].mean():.1f} nodes, {temporal_df['total_edges'].mean():.1f} edges\n")
    f.write(f"- **Mean active timepoints:** {temporal_df['active_timepoints'].mean():.1f} / 6\n\n")

    # Top dynamic clusters
    f.write("### Top 20 Most Dynamic Clusters (by edge turnover)\n\n")
    top_temporal = temporal_df.nlargest(20, 'top_dynamic_score')
    f.write("| Rank | Cluster | Peak Celltype | Peak TP | Nodes | Edges | Most Dynamic TF | Turnover Score |\n")
    f.write("|------|---------|---------------|---------|-------|-------|-----------------|----------------|\n")
    for idx, row in top_temporal.iterrows():
        f.write(f"| {idx+1} | {row['cluster_id']} | {row['peak_celltype']} | {row['peak_timepoint']} | ")
        f.write(f"{row['total_nodes']} | {row['total_edges']} | {row['top_dynamic_tf']} | {row['top_dynamic_score']} |\n")
    f.write("\n")

    # Peak celltype distribution
    f.write("### Distribution by Peak Celltype\n\n")
    celltype_counts = temporal_df['peak_celltype'].value_counts().head(20)
    f.write("| Celltype | N Clusters |\n")
    f.write("|----------|------------|\n")
    for celltype, count in celltype_counts.items():
        f.write(f"| {celltype} | {count} |\n")
    f.write("\n")

    # Detailed per-cluster reports
    f.write("---\n\n")
    f.write("## Detailed Cluster Reports\n\n")

    # Sort by dynamics score for readability
    sorted_df = temporal_df.sort_values('top_dynamic_score', ascending=False)

    for idx, row in sorted_df.head(100).iterrows():  # Top 100 for now
        cluster_id = row['cluster_id']
        f.write(f"### Cluster {cluster_id}\n\n")
        f.write(f"**Peak Location:** {row['peak_celltype']}, {row['peak_timepoint']} somites (accessibility = {row['peak_accessibility']:.2f})\n\n")
        f.write(f"**Network Size:** {row['total_nodes']} nodes ({row['total_tfs']} TFs), {row['total_edges']} edges\n\n")
        f.write(f"**Node Classification:**\n")
        f.write(f"- TF-only: {row['tf_only']}\n")
        f.write(f"- Target-only: {row['target_only']}\n")
        f.write(f"- TF & Target: {row['tf_and_target']}\n\n")
        f.write(f"**Temporal Activity:** Active in {row['active_timepoints']} / 6 timepoints\n\n")

        if row['top_dynamic_tf']:
            f.write(f"**Most Dynamic TF:** {row['top_dynamic_tf']} (turnover score = {row['top_dynamic_score']})\n\n")

        f.write("---\n\n")

print(f"✓ Saved: temporal_subGRN_dynamics_all_clusters.md\n")

# ============================================================================
# GENERATE SPATIAL/CELLTYPE DYNAMICS MARKDOWN
# ============================================================================

print("Generating spatial/celltype dynamics markdown...")

with open(f"{figpath}spatial_celltype_subGRN_dynamics_all_clusters.md", 'w') as f:
    f.write("# Spatial/Celltype SubGRN Dynamics - All 346 Clusters\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## Overview\n\n")
    f.write(f"This report documents spatial dynamics of subGRNs across celltypes for all {len(spatial_df)} clusters.\n\n")
    f.write("For each cluster, we analyzed subGRN changes across celltypes at the peak timepoint.\n\n")

    # Summary statistics
    f.write("### Summary Statistics\n\n")
    f.write(f"- **Total clusters analyzed:** {len(spatial_df)}\n")
    f.write(f"- **Mean celltypes per cluster:** {spatial_df['n_celltypes'].mean():.1f}\n")
    f.write(f"- **Mean TFs per cluster:** {spatial_df['n_total_tfs'].mean():.1f}\n\n")

    # Top ubiquitous TFs across all clusters
    f.write("### Most Ubiquitous TFs (across all clusters)\n\n")
    all_ubiquitous_tfs = []
    for tf_str in spatial_df['most_ubiquitous_tf'].dropna():
        if tf_str:
            all_ubiquitous_tfs.append(tf_str)
    if all_ubiquitous_tfs:
        tf_counts = Counter(all_ubiquitous_tfs).most_common(20)
        f.write("| TF | N Clusters as Most Ubiquitous |\n")
        f.write("|----|-------------------------------|\n")
        for tf, count in tf_counts:
            f.write(f"| {tf} | {count} |\n")
        f.write("\n")

    # Top specific TFs
    f.write("### Most Celltype-Specific TFs (across all clusters)\n\n")
    all_specific_tfs = []
    for tf_str in spatial_df['most_specific_tf'].dropna():
        if tf_str:
            all_specific_tfs.append(tf_str)
    if all_specific_tfs:
        tf_counts = Counter(all_specific_tfs).most_common(20)
        f.write("| TF | N Clusters as Most Specific |\n")
        f.write("|----|-----------------------------|\n")
        for tf, count in tf_counts:
            f.write(f"| {tf} | {count} |\n")
        f.write("\n")

    # Detailed per-cluster reports
    f.write("---\n\n")
    f.write("## Detailed Cluster Reports\n\n")

    # Sort by number of celltypes for readability
    sorted_df = spatial_df.sort_values('n_celltypes', ascending=False)

    for idx, row in sorted_df.head(100).iterrows():  # Top 100 for now
        cluster_id = row['cluster_id']
        f.write(f"### Cluster {cluster_id}\n\n")
        f.write(f"**Peak Location:** {row['peak_celltype']}, {row['peak_timepoint']} somites\n\n")
        f.write(f"**Spatial Breadth:** Active in {row['n_celltypes']} celltypes at peak timepoint\n\n")
        f.write(f"**Total TFs:** {row['n_total_tfs']}\n\n")

        if row['most_ubiquitous_tf']:
            f.write(f"**Most Ubiquitous TF:** {row['most_ubiquitous_tf']}\n\n")
        if row['most_specific_tf']:
            f.write(f"**Most Celltype-Specific TF:** {row['most_specific_tf']}\n\n")

        f.write("---\n\n")

print(f"✓ Saved: spatial_celltype_subGRN_dynamics_all_clusters.md\n")

# ============================================================================
# GENERATE TOP DYNAMICS SUMMARY
# ============================================================================

print("Generating top dynamics summary...")

with open(f"{figpath}TOP_DYNAMIC_PROGRAMS_SUMMARY.md", 'w') as f:
    f.write("# Top Dynamic Regulatory Programs - Curated Highlights\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("This report highlights the most dynamic regulatory programs from systematic analysis of 346 clusters.\n\n")

    f.write("## Top 20 Temporal Champions\n\n")
    f.write("Clusters with highest TF edge turnover across developmental timepoints:\n\n")
    top_temporal = temporal_df.nlargest(20, 'top_dynamic_score')
    f.write("| Rank | Cluster | Peak Celltype | Peak TP | Nodes | Edges | Most Dynamic TF | Score |\n")
    f.write("|------|---------|---------------|---------|-------|-------|-----------------|-------|\n")
    for rank, (idx, row) in enumerate(top_temporal.iterrows(), 1):
        f.write(f"| {rank} | {row['cluster_id']} | {row['peak_celltype']} | {row['peak_timepoint']} | ")
        f.write(f"{row['total_nodes']} | {row['total_edges']} | {row['top_dynamic_tf']} | {row['top_dynamic_score']} |\n")
    f.write("\n")

    f.write("## Top 20 Spatial Champions\n\n")
    f.write("Clusters with broadest celltype activity:\n\n")
    top_spatial = spatial_df.nlargest(20, 'n_celltypes')
    f.write("| Rank | Cluster | Peak Celltype | Peak TP | N Celltypes | N TFs | Most Ubiquitous TF |\n")
    f.write("|------|---------|---------------|---------|-------------|-------|--------------------|\\n")
    for rank, (idx, row) in enumerate(top_spatial.iterrows(), 1):
        f.write(f"| {rank} | {row['cluster_id']} | {row['peak_celltype']} | {row['peak_timepoint']} | ")
        f.write(f"{row['n_celltypes']} | {row['n_total_tfs']} | {row['most_ubiquitous_tf']} |\n")
    f.write("\n")

    f.write("## Key Insights\n\n")
    f.write(f"### Temporal Dynamics\n\n")
    f.write(f"- Mean edge turnover across all clusters: {temporal_df['top_dynamic_score'].mean():.1f}\n")
    f.write(f"- Highest turnover: {temporal_df['top_dynamic_score'].max():.0f} (cluster {temporal_df.loc[temporal_df['top_dynamic_score'].idxmax(), 'cluster_id']})\n")
    f.write(f"- Most common peak celltype: {temporal_df['peak_celltype'].mode()[0]}\n")
    f.write(f"- Most common peak timepoint: {temporal_df['peak_timepoint'].mode()[0]} somites\n\n")

    f.write(f"### Spatial Dynamics\n\n")
    f.write(f"- Mean celltype breadth: {spatial_df['n_celltypes'].mean():.1f} celltypes\n")
    f.write(f"- Broadest cluster: {spatial_df['n_celltypes'].max()} celltypes (cluster {spatial_df.loc[spatial_df['n_celltypes'].idxmax(), 'cluster_id']})\n")
    f.write(f"- Mean TFs per cluster: {spatial_df['n_total_tfs'].mean():.1f}\n\n")

print(f"✓ Saved: TOP_DYNAMIC_PROGRAMS_SUMMARY.md\n")

print("="*80)
print("MARKDOWN GENERATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - temporal_subGRN_dynamics_all_clusters.md")
print("  - spatial_celltype_subGRN_dynamics_all_clusters.md")
print("  - TOP_DYNAMIC_PROGRAMS_SUMMARY.md")
