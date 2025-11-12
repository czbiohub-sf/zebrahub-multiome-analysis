#!/usr/bin/env python
"""
Comprehensive Systematic SubGRN Analysis for All 346 Clusters

This script performs:
1. Peak detection for all 402 clusters
2. Temporal dynamics analysis (346 clusters × timepoints)
3. Spatial/celltype dynamics (346 clusters × celltypes)
4. Both lineage-based and all-celltype comparisons

Outputs:
- temporal_subGRN_dynamics_all_clusters.md
- spatial_celltype_subGRN_dynamics_all_clusters.md
- systematic_analysis_summary.csv
- TOP_DYNAMIC_PROGRAMS_SUMMARY.md
- CLUSTERS_WITHOUT_MATRICES.txt
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from datetime import datetime

print("="*80)
print("COMPREHENSIVE SYSTEMATIC SUBGRN ANALYSIS")
print("All 346 Clusters with TF-Gene Matrices")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Define paths
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
os.makedirs(figpath, exist_ok=True)

# Define pseudobulk groups to EXCLUDE (< 20 cells)
EXCLUDE_COLUMNS = [
    'NMPs_30somites',
    'epidermis_30somites',
    'fast_muscle_0somites',
    'hatching_gland_30somites',
    'muscle_30somites',
    'neural_10somites',
    'optic_cup_0somites',
    'primordial_germ_cells_0somites',
    'primordial_germ_cells_5somites',
    'primordial_germ_cells_10somites',
    'primordial_germ_cells_15somites',
    'primordial_germ_cells_20somites',
    'primordial_germ_cells_30somites',
    'tail_bud_30somites'
]

print(f"\n** EXCLUDING LOW-CELL PSEUDOBULK GROUPS **")
print(f"   Excluding {len(EXCLUDE_COLUMNS)} celltype×timepoint combinations (<20 cells)")
print(f"   Notably: ALL primordial_germ_cells timepoints excluded")

# Load data
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# 1. Load accessibility matrix (402 clusters)
print("\n1. Loading accessibility matrix...")
accessibility_df = pd.read_csv(
    "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_fine_by_pseudobulk.csv",
    index_col=0
)
print(f"   ✓ Loaded {len(accessibility_df)} peak clusters")
print(f"   ✓ Across {len(accessibility_df.columns)} celltype×timepoint combinations (before filtering)")

# Filter out excluded columns
columns_before = len(accessibility_df.columns)
accessibility_df = accessibility_df.drop(columns=EXCLUDE_COLUMNS, errors='ignore')
columns_after = len(accessibility_df.columns)
print(f"   ✓ Filtered to {columns_after} valid celltype×timepoint combinations (removed {columns_before - columns_after})")

# 2. Load TF-gene matrices (346 clusters)
print("\n2. Loading TF-gene matrices...")
cluster_tf_gene_matrices_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl"
with open(cluster_tf_gene_matrices_path, 'rb') as f:
    cluster_tf_gene_matrices = pickle.load(f)
print(f"   ✓ Loaded {len(cluster_tf_gene_matrices)} clusters with TF-gene matrices")

# 3. Load GRN dictionary (189 combinations)
print("\n3. Loading CellOracle GRNs...")
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
print(f"   ✓ Loaded {len(grn_dict)} celltype×timepoint GRN combinations")

# Identify clusters with/without matrices
all_clusters = set(accessibility_df.index)
clusters_with_matrices = set(cluster_tf_gene_matrices.keys())
clusters_without_matrices = all_clusters - clusters_with_matrices

print(f"\n** CLUSTER BREAKDOWN **")
print(f"   Total clusters: {len(all_clusters)}")
print(f"   With TF-gene matrices: {len(clusters_with_matrices)} (can analyze)")
print(f"   Without matrices: {len(clusters_without_matrices)} (documented only)")

# Document clusters without matrices
print(f"\n4. Documenting clusters without TF-gene matrices...")
with open(f"{figpath}CLUSTERS_WITHOUT_MATRICES.txt", 'w') as f:
    f.write("# Clusters Without TF-Gene Matrices\n")
    f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"#\n")
    f.write(f"# Total: {len(clusters_without_matrices)} clusters\n")
    f.write(f"# These clusters do not have predicted TF-gene regulatory meshes\n")
    f.write(f"# and therefore cannot be analyzed for subGRN dynamics.\n")
    f.write(f"#\n")
    f.write(f"# Possible reasons:\n")
    f.write(f"# - No enriched TF motifs detected\n")
    f.write(f"# - No linked genes identified\n")
    f.write(f"# - Did not pass quality filters during mesh construction\n")
    f.write(f"#\n\n")
    for cluster_id in sorted(clusters_without_matrices):
        f.write(f"{cluster_id}\n")
print(f"   ✓ Saved: CLUSTERS_WITHOUT_MATRICES.txt")

# Define lineages for spatial analysis
LINEAGES = {
    'mesodermal': ['NMPs', 'tail_bud', 'PSM', 'somites', 'fast_muscle'],
    'neural': ['NMPs', 'spinal_cord', 'neural_posterior']
}

TIMEPOINTS = ['00', '05', '10', '15', '20', '30']


# ============================================================================
# STEP 1: PEAK DETECTION
# ============================================================================

print("\n" + "="*80)
print("STEP 1: PEAK DETECTION FOR ALL 346 CLUSTERS")
print("="*80)

peak_info = {}

for cluster_id in sorted(clusters_with_matrices):
    if cluster_id not in accessibility_df.index:
        print(f"   WARNING: {cluster_id} not in accessibility matrix, skipping")
        continue

    # Get accessibility values for this cluster
    cluster_accessibility = accessibility_df.loc[cluster_id]

    # Find peak (max accessibility)
    peak_value = cluster_accessibility.max()
    peak_column = cluster_accessibility.idxmax()

    # Parse celltype and timepoint
    # Format: celltype_Nsomites, e.g., "NMPs_10somites"
    parts = peak_column.rsplit('_', 1)
    if len(parts) == 2:
        celltype = parts[0]
        timepoint_str = parts[1].replace('somites', '')
        timepoint = timepoint_str.zfill(2)  # Zero-pad to 2 digits
    else:
        print(f"   WARNING: Could not parse {peak_column}")
        continue

    peak_info[cluster_id] = {
        'peak_celltype': celltype,
        'peak_timepoint': timepoint,
        'peak_accessibility': peak_value,
        'peak_column': peak_column
    }

print(f"\n✓ Detected peaks for {len(peak_info)} clusters")
print(f"  Sample peaks:")
for i, (cluster_id, info) in enumerate(list(peak_info.items())[:5]):
    print(f"    {cluster_id}: {info['peak_celltype']} @ {info['peak_timepoint']} somites (acc={info['peak_accessibility']:.2f})")


# ============================================================================
# STEP 2: TEMPORAL DYNAMICS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: TEMPORAL DYNAMICS ANALYSIS")
print("="*80)
print("Analyzing subGRN changes across timepoints at peak celltype...")

temporal_results = []

for cluster_idx, (cluster_id, peak) in enumerate(peak_info.items(), 1):
    if cluster_idx % 50 == 0:
        print(f"  Progress: {cluster_idx}/{len(peak_info)} clusters...")

    peak_celltype = peak['peak_celltype']

    # Get predicted pairs for this cluster
    if cluster_id not in cluster_tf_gene_matrices:
        continue

    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = []
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.append((tf, gene))

    # Extract subGRNs across all timepoints for peak celltype
    subgrns = {}
    for timepoint in TIMEPOINTS:
        if (peak_celltype, timepoint) in grn_dict:
            grn_df = grn_dict[(peak_celltype, timepoint)]
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            subgrns[timepoint] = subgrn

    # Analyze temporal dynamics
    if len(subgrns) == 0:
        continue

    # Collect all nodes and edges
    all_nodes = set()
    all_edges = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_nodes.update(subgrn['source'])
            all_nodes.update(subgrn['target'])
            all_edges.update(zip(subgrn['source'], subgrn['target']))

    # Classify nodes globally
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])

    tf_only_nodes = all_sources - all_targets
    target_only_nodes = all_targets - all_sources
    tf_and_target_nodes = all_sources & all_targets

    # Track TF dynamics per timepoint
    tf_edge_turnover = {}  # TF -> total edge changes
    tf_role_switches = {}  # TF -> role change events
    tf_sign_flips = {}     # TF -> sign flip events

    for tf in all_sources:
        edge_changes = 0
        role_switches = []
        sign_flips = []

        prev_edges = set()
        prev_role = None
        prev_signs = {}  # target -> sign

        for timepoint in TIMEPOINTS:
            if timepoint not in subgrns or len(subgrns[timepoint]) == 0:
                continue

            subgrn = subgrns[timepoint]

            # Get current edges for this TF
            tf_edges = subgrn[subgrn['source'] == tf]
            current_edges = set(zip(tf_edges['source'], tf_edges['target']))

            # Edge turnover
            if prev_edges is not None:
                gained = current_edges - prev_edges
                lost = prev_edges - current_edges
                edge_changes += len(gained) + len(lost)

            # Role switching (TF-only <-> TF&Target)
            if tf in subgrn['source'].values:
                if tf in subgrn['target'].values:
                    current_role = 'TF&Target'
                else:
                    current_role = 'TF-only'
            else:
                current_role = None

            if prev_role is not None and current_role is not None and prev_role != current_role:
                role_switches.append((timepoint, f"{prev_role}→{current_role}"))

            # Sign flipping (activation <-> repression)
            current_signs = {}
            for _, row in tf_edges.iterrows():
                target = row['target']
                if 'coef_mean' in row and pd.notna(row['coef_mean']):
                    sign = 1 if row['coef_mean'] > 0 else -1
                elif 'coef' in row and pd.notna(row['coef']):
                    sign = 1 if row['coef'] > 0 else -1
                else:
                    sign = 1  # Default positive
                current_signs[target] = sign

            # Check for sign flips in edges to same targets
            for target in set(prev_signs.keys()) & set(current_signs.keys()):
                if prev_signs[target] != current_signs[target]:
                    sign_flips.append((timepoint, target, prev_signs[target], current_signs[target]))

            prev_edges = current_edges
            prev_role = current_role
            prev_signs = current_signs

        tf_edge_turnover[tf] = edge_changes
        tf_role_switches[tf] = role_switches
        tf_sign_flips[tf] = sign_flips

    # Rank TFs by dynamics
    top_dynamic_tfs = sorted(tf_edge_turnover.items(), key=lambda x: x[1], reverse=True)[:10]

    # Store results
    temporal_results.append({
        'cluster_id': cluster_id,
        'peak_celltype': peak_celltype,
        'peak_timepoint': peak['peak_timepoint'],
        'peak_accessibility': peak['peak_accessibility'],
        'total_nodes': len(all_nodes),
        'total_edges': len(all_edges),
        'total_tfs': len(all_sources),
        'tf_only_nodes': len(tf_only_nodes),
        'target_only_nodes': len(target_only_nodes),
        'tf_and_target_nodes': len(tf_and_target_nodes),
        'active_timepoints': len([tp for tp in TIMEPOINTS if tp in subgrns and len(subgrns[tp]) > 0]),
        'top_dynamic_tfs': [tf for tf, _ in top_dynamic_tfs],
        'top_dynamic_scores': [score for _, score in top_dynamic_tfs],
        'tf_edge_turnover': tf_edge_turnover,
        'tf_role_switches': tf_role_switches,
        'tf_sign_flips': tf_sign_flips,
        'subgrns_by_timepoint': subgrns
    })

print(f"\n✓ Analyzed temporal dynamics for {len(temporal_results)} clusters")

# Save temporal results to CSV
temporal_df = pd.DataFrame([{
    'cluster_id': r['cluster_id'],
    'peak_celltype': r['peak_celltype'],
    'peak_timepoint': r['peak_timepoint'],
    'peak_accessibility': r['peak_accessibility'],
    'total_nodes': r['total_nodes'],
    'total_edges': r['total_edges'],
    'total_tfs': r['total_tfs'],
    'tf_only': r['tf_only_nodes'],
    'target_only': r['target_only_nodes'],
    'tf_and_target': r['tf_and_target_nodes'],
    'active_timepoints': r['active_timepoints'],
    'top_dynamic_tf': r['top_dynamic_tfs'][0] if r['top_dynamic_tfs'] else '',
    'top_dynamic_score': r['top_dynamic_scores'][0] if r['top_dynamic_scores'] else 0
} for r in temporal_results])

temporal_df.to_csv(f"{figpath}systematic_analysis_temporal_summary.csv", index=False)
print(f"✓ Saved: systematic_analysis_temporal_summary.csv")


# ============================================================================
# STEP 3: SPATIAL/CELLTYPE DYNAMICS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: SPATIAL/CELLTYPE DYNAMICS ANALYSIS")
print("="*80)
print("Analyzing subGRN changes across celltypes at peak timepoint...")

spatial_results = []

for cluster_idx, (cluster_id, peak) in enumerate(peak_info.items(), 1):
    if cluster_idx % 50 == 0:
        print(f"  Progress: {cluster_idx}/{len(peak_info)} clusters...")

    peak_timepoint = peak['peak_timepoint']

    # Get predicted pairs
    if cluster_id not in cluster_tf_gene_matrices:
        continue

    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = []
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.append((tf, gene))

    # Extract subGRNs across all celltypes at peak timepoint
    subgrns_all_celltypes = {}
    for (celltype, timepoint), grn_df in grn_dict.items():
        if timepoint == peak_timepoint:
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            if len(found_pairs) > 0:
                mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
                subgrn = grn_df[mask].copy()
                subgrns_all_celltypes[celltype] = subgrn

    # Lineage-based analysis
    lineage_subgrns = {}
    for lineage_name, lineage_celltypes in LINEAGES.items():
        lineage_subgrns[lineage_name] = {}
        for celltype in lineage_celltypes:
            if celltype in subgrns_all_celltypes:
                lineage_subgrns[lineage_name][celltype] = subgrns_all_celltypes[celltype]

    # Analyze spatial dynamics
    all_tfs = set()
    for subgrn in subgrns_all_celltypes.values():
        if len(subgrn) > 0:
            all_tfs.update(subgrn['source'])

    # TF celltype specificity
    tf_celltype_presence = {}
    for tf in all_tfs:
        celltypes_with_tf = [ct for ct, subgrn in subgrns_all_celltypes.items()
                             if len(subgrn) > 0 and tf in subgrn['source'].values]
        tf_celltype_presence[tf] = celltypes_with_tf

    # Rank TFs by celltype specificity (ubiquitous vs specific)
    tf_specificity_scores = {tf: len(celltypes) for tf, celltypes in tf_celltype_presence.items()}
    ubiquitous_tfs = sorted([(tf, score) for tf, score in tf_specificity_scores.items()],
                           key=lambda x: x[1], reverse=True)[:10]
    specific_tfs = sorted([(tf, score) for tf, score in tf_specificity_scores.items()],
                         key=lambda x: x[1])[:10]

    # Store results
    spatial_results.append({
        'cluster_id': cluster_id,
        'peak_celltype': peak['peak_celltype'],
        'peak_timepoint': peak_timepoint,
        'n_celltypes_with_subgrn': len(subgrns_all_celltypes),
        'all_tfs': list(all_tfs),
        'ubiquitous_tfs': [tf for tf, _ in ubiquitous_tfs],
        'specific_tfs': [tf for tf, _ in specific_tfs],
        'tf_celltype_presence': tf_celltype_presence,
        'lineage_subgrns': lineage_subgrns,
        'all_celltypes_subgrns': subgrns_all_celltypes
    })

print(f"\n✓ Analyzed spatial dynamics for {len(spatial_results)} clusters")

# Save spatial results to CSV
spatial_df = pd.DataFrame([{
    'cluster_id': r['cluster_id'],
    'peak_celltype': r['peak_celltype'],
    'peak_timepoint': r['peak_timepoint'],
    'n_celltypes': r['n_celltypes_with_subgrn'],
    'n_total_tfs': len(r['all_tfs']),
    'most_ubiquitous_tf': r['ubiquitous_tfs'][0] if r['ubiquitous_tfs'] else '',
    'most_specific_tf': r['specific_tfs'][0] if r['specific_tfs'] else ''
} for r in spatial_results])

spatial_df.to_csv(f"{figpath}systematic_analysis_spatial_summary.csv", index=False)
print(f"✓ Saved: systematic_analysis_spatial_summary.csv")


# ============================================================================
# STEP 4: GENERATE MARKDOWN REPORTS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: GENERATING MARKDOWN REPORTS")
print("="*80)

# Will be implemented in next part of script
# This is a substantial amount of markdown generation
print("  (Markdown generation will be in separate script part)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nGenerated files:")
print(f"  - CLUSTERS_WITHOUT_MATRICES.txt ({len(clusters_without_matrices)} clusters)")
print(f"  - systematic_analysis_temporal_summary.csv ({len(temporal_results)} clusters)")
print(f"  - systematic_analysis_spatial_summary.csv ({len(spatial_results)} clusters)")
print(f"\nNext: Run markdown generation scripts to create full reports")
