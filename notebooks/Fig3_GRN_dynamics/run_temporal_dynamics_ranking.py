#!/usr/bin/env python
"""
Standalone script to run the temporal dynamics ranking pipeline
This extracts the relevant cells from the notebook and executes them
"""

import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
import os
import pickle
from tqdm import tqdm

# Define paths
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
os.makedirs(figpath, exist_ok=True)

print("="*80)
print("TEMPORAL DYNAMICS RANKING PIPELINE")
print("="*80)

# Load required data
print("\n1. Loading data...")

# Load the cluster-by-pseudobulk accessibility matrix
df_clusters_groups = pd.read_csv(
    "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_fine_by_pseudobulk.csv",
    index_col=0
)
print(f"   ✓ Loaded df_clusters_groups (raw): {df_clusters_groups.shape}")

# Filter out pseudobulk groups with < 20 cells
excluded_groups = [
    'NMPs_30somites', 'epidermis_30somites', 'fast_muscle_0somites',
    'hatching_gland_30somites', 'muscle_30somites', 'neural_10somites',
    'optic_cup_0somites', 'primordial_germ_cells_0somites',
    'primordial_germ_cells_5somites', 'primordial_germ_cells_10somites',
    'primordial_germ_cells_15somites', 'primordial_germ_cells_20somites',
    'primordial_germ_cells_30somites', 'tail_bud_30somites'
]

# Drop excluded columns
cols_to_drop = [col for col in excluded_groups if col in df_clusters_groups.columns]
df_clusters_groups = df_clusters_groups.drop(columns=cols_to_drop)
print(f"   ✓ Filtered out {len(cols_to_drop)} low-cell groups (< 20 cells)")
print(f"   ✓ Retained df_clusters_groups: {df_clusters_groups.shape}")

# Load the GRN dictionary
from pathlib import Path

def load_grn_dict_pathlib(base_dir="grn_exports", grn_type="filtered"):
    """Load GRN dictionary using pathlib"""
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
print(f"   ✓ Loaded GRN dictionary: {len(grn_dict)} celltype×timepoint combinations")

# Load cluster_tf_gene_matrices
cluster_tf_gene_matrices_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl"
with open(cluster_tf_gene_matrices_path, 'rb') as f:
    cluster_tf_gene_matrices = pickle.load(f)
print(f"   ✓ Loaded cluster_tf_gene_matrices: {len(cluster_tf_gene_matrices)} clusters")

# Define functions
print("\n2. Defining analysis functions...")

def gini_coefficient(values):
    """Calculate Gini coefficient (0=equal, 1=concentrated)"""
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

def find_most_accessible_celltype(cluster_id, df_clusters_groups, min_accessibility=0.01):
    """Find the celltype×timepoint with highest accessibility for a peak cluster."""
    values = df_clusters_groups.loc[cluster_id]
    values_filtered = values[values >= min_accessibility]

    if len(values_filtered) == 0:
        return None

    gini = gini_coefficient(values.values)
    best_group = values_filtered.idxmax()
    best_value = values_filtered.max()

    parts = best_group.rsplit('_', 1)
    if len(parts) == 2:
        celltype, timepoint = parts
    else:
        celltype = best_group
        timepoint = None

    top_5 = [(group, val) for group, val in
             values_filtered.sort_values(ascending=False).head(5).items()]

    return {
        'cluster_id': cluster_id,
        'best_group': best_group,
        'celltype': celltype,
        'timepoint': timepoint,
        'accessibility': best_value,
        'gini_coefficient': gini,
        'top_5_groups': top_5
    }

def compute_temporal_dynamics_score(cluster_id, celltype, grn_dict,
                                    cluster_tf_gene_matrices,
                                    developmental_tfs=None):
    """Compute how dynamically a subGRN changes over time for a specific celltype."""

    if developmental_tfs is None:
        developmental_tfs = {
            'sox2', 'sox3', 'sox9a', 'sox9b', 'sox10', 'sox19a', 'sox19b',
            'pax6a', 'pax6b', 'pax2a', 'pax8',
            'tbx6', 'tbx16', 'tbx5a', 'tbxa',
            'myod1', 'myf5', 'myog', 'myf6',
            'neurog1', 'neurod1', 'neurod4',
            'gata1a', 'gata2a', 'gata3', 'gata4', 'gata5', 'gata6',
            'hand2', 'hoxb1b', 'hoxa2b', 'hoxb5a',
            'foxa1', 'foxa2', 'foxa3',
            'nkx2.1a', 'nkx2.5', 'nkx6.1', 'nkx6.2',
            'olig2', 'ascl1a', 'msgn1', 'meox1', 'tcf21'
        }

    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = set()
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.add((tf, gene))

    timepoints = sorted([tp for (ct, tp) in grn_dict.keys() if ct == celltype])

    if len(timepoints) < 2:
        return None

    subgrns_by_timepoint = {}
    nodes_by_timepoint = {}
    edges_by_timepoint = {}
    tfs_by_timepoint = {}
    dev_tfs_by_timepoint = {}

    for timepoint in timepoints:
        if (celltype, timepoint) not in grn_dict:
            continue

        grn_df = grn_dict[(celltype, timepoint)]
        grn_pairs = set(zip(grn_df['source'], grn_df['target']))
        found_pairs = predicted_pairs & grn_pairs

        mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
        subgrn = grn_df[mask].copy()

        subgrns_by_timepoint[timepoint] = subgrn

        if len(subgrn) > 0:
            nodes = set(subgrn['source']) | set(subgrn['target'])
            edges = set(zip(subgrn['source'], subgrn['target']))
            tfs = set(subgrn['source'])
            dev_tfs = tfs & developmental_tfs

            nodes_by_timepoint[timepoint] = nodes
            edges_by_timepoint[timepoint] = edges
            tfs_by_timepoint[timepoint] = tfs
            dev_tfs_by_timepoint[timepoint] = dev_tfs
        else:
            nodes_by_timepoint[timepoint] = set()
            edges_by_timepoint[timepoint] = set()
            tfs_by_timepoint[timepoint] = set()
            dev_tfs_by_timepoint[timepoint] = set()

    all_nodes = set.union(*nodes_by_timepoint.values()) if nodes_by_timepoint else set()
    all_edges = set.union(*edges_by_timepoint.values()) if edges_by_timepoint else set()
    all_tfs = set.union(*tfs_by_timepoint.values()) if tfs_by_timepoint else set()
    all_dev_tfs = set.union(*dev_tfs_by_timepoint.values()) if dev_tfs_by_timepoint else set()

    if len(all_nodes) == 0:
        return None

    # Quality check: count timepoints with sufficient edges
    edge_counts_list = [len(edges) for edges in edges_by_timepoint.values()]
    timepoints_with_edges = sum(1 for count in edge_counts_list if count > 0)

    nodes_present_count = {node: sum(1 for tp_nodes in nodes_by_timepoint.values() if node in tp_nodes) for node in all_nodes}
    dynamic_nodes = sum(1 for count in nodes_present_count.values() if count < len(timepoints))
    node_turnover_rate = dynamic_nodes / len(all_nodes) if len(all_nodes) > 0 else 0

    edges_present_count = {edge: sum(1 for tp_edges in edges_by_timepoint.values() if edge in tp_edges) for edge in all_edges}
    dynamic_edges = sum(1 for count in edges_present_count.values() if count < len(timepoints))
    edge_turnover_rate = dynamic_edges / len(all_edges) if len(all_edges) > 0 else 0

    tfs_present_count = {tf: sum(1 for tp_tfs in tfs_by_timepoint.values() if tf in tp_tfs) for tf in all_tfs}
    dynamic_tfs = sum(1 for count in tfs_present_count.values() if count < len(timepoints))
    tf_turnover_rate = dynamic_tfs / len(all_tfs) if len(all_tfs) > 0 else 0

    dev_tfs_present_count = {tf: sum(1 for tp_dev_tfs in dev_tfs_by_timepoint.values() if tf in tp_dev_tfs) for tf in all_dev_tfs}
    dynamic_dev_tfs = sum(1 for count in dev_tfs_present_count.values() if count < len(timepoints))
    dev_tf_turnover_rate = (dynamic_dev_tfs / len(all_dev_tfs) if len(all_dev_tfs) > 0 else 0)

    edge_counts = [len(edges) for edges in edges_by_timepoint.values()]
    temporal_variance = np.std(edge_counts) / np.mean(edge_counts) if np.mean(edge_counts) > 0 else 0

    dynamics_score = (
        0.4 * dev_tf_turnover_rate +
        0.3 * edge_turnover_rate +
        0.2 * tf_turnover_rate +
        0.1 * temporal_variance
    )

    return {
        'cluster_id': cluster_id,
        'celltype': celltype,
        'n_timepoints': len(timepoints),
        'n_timepoints_with_edges': timepoints_with_edges,
        'n_total_nodes': len(all_nodes),
        'n_total_edges': len(all_edges),
        'n_total_tfs': len(all_tfs),
        'n_developmental_tfs': len(all_dev_tfs),
        'developmental_tfs_list': sorted(list(all_dev_tfs)),
        'node_turnover_rate': node_turnover_rate,
        'edge_turnover_rate': edge_turnover_rate,
        'tf_turnover_rate': tf_turnover_rate,
        'dev_tf_turnover_rate': dev_tf_turnover_rate,
        'temporal_variance': temporal_variance,
        'dynamics_score': dynamics_score,
        'edge_counts_per_timepoint': edge_counts_list,
        'subgrns_by_timepoint': subgrns_by_timepoint,
        'timepoints': timepoints
    }

def rank_clusters_by_temporal_dynamics(df_clusters_groups, grn_dict,
                                       cluster_tf_gene_matrices,
                                       min_accessibility=0.01,
                                       min_timepoints=3,
                                       min_edges_per_timepoint=5,
                                       min_timepoints_with_edges=3,
                                       output_csv="cluster_ranking_temporal_dynamics.csv"):
    """Full pipeline: rank all clusters by temporal dynamics.

    Parameters:
    - min_edges_per_timepoint: Minimum edges required at each timepoint (default: 5)
    - min_timepoints_with_edges: Minimum number of timepoints with >= min_edges (default: 3)
    """

    print(f"\n3. Starting analysis of {len(cluster_tf_gene_matrices)} clusters...")
    print(f"   Filters: min_edges_per_timepoint={min_edges_per_timepoint}, min_timepoints_with_edges={min_timepoints_with_edges}")

    results = []

    for cluster_id in tqdm(cluster_tf_gene_matrices.keys(), desc="Processing clusters"):
        access_info = find_most_accessible_celltype(cluster_id, df_clusters_groups, min_accessibility)

        if access_info is None:
            continue

        celltype = access_info['celltype']

        dynamics_info = compute_temporal_dynamics_score(
            cluster_id, celltype, grn_dict, cluster_tf_gene_matrices
        )

        if dynamics_info is None:
            continue

        if dynamics_info['n_timepoints'] < min_timepoints:
            continue

        # NEW FILTER: Check if enough timepoints have sufficient edges
        edge_counts = dynamics_info['edge_counts_per_timepoint']
        timepoints_with_sufficient_edges = sum(1 for count in edge_counts if count >= min_edges_per_timepoint)

        if timepoints_with_sufficient_edges < min_timepoints_with_edges:
            continue

        result = {**access_info, **dynamics_info}
        results.append(result)

    df_ranked = pd.DataFrame(results)
    df_ranked = df_ranked.sort_values('dynamics_score', ascending=False)

    df_ranked_export = df_ranked.drop(columns=['subgrns_by_timepoint', 'top_5_groups', 'edge_counts_per_timepoint'], errors='ignore')
    df_ranked_export.to_csv(output_csv, index=False)
    print(f"\n   ✓ Ranking table saved to: {output_csv}")

    print(f"\n{'='*80}")
    print(f"RANKING SUMMARY")
    print(f"{'='*80}")
    print(f"Total clusters analyzed: {len(df_ranked)}")
    print(f"Dynamics score range: {df_ranked['dynamics_score'].min():.3f} - {df_ranked['dynamics_score'].max():.3f}")
    print(f"Median dynamics score: {df_ranked['dynamics_score'].median():.3f}")
    print(f"\nTop 10 most dynamic clusters:")
    print(df_ranked[['cluster_id', 'celltype', 'dynamics_score',
                     'n_developmental_tfs', 'developmental_tfs_list']].head(10).to_string(index=False))

    return df_ranked

# Run the pipeline
print("\n" + "="*80)
df_ranked = rank_clusters_by_temporal_dynamics(
    df_clusters_groups=df_clusters_groups,
    grn_dict=grn_dict,
    cluster_tf_gene_matrices=cluster_tf_gene_matrices,
    min_accessibility=0.01,
    min_timepoints=3,
    min_edges_per_timepoint=10,
    min_timepoints_with_edges=3,
    output_csv=figpath + "cluster_ranking_temporal_dynamics.csv"
)

# Additional analysis
print(f"\n{'='*80}")
print("TOP 20 MOST DYNAMIC CLUSTERS (DETAILED)")
print(f"{'='*80}")
print(df_ranked[['cluster_id', 'celltype', 'dynamics_score',
                 'n_developmental_tfs', 'dev_tf_turnover_rate',
                 'edge_turnover_rate', 'developmental_tfs_list']].head(20).to_string())

print(f"\n{'='*80}")
print("CELL TYPE SPECIFIC ANALYSIS")
print(f"{'='*80}")

# Focus on developmental cell types of interest
target_celltypes = ['PSM', 'NMPs', 'tail_bud', 'neural_posterior', 'spinal_cord',
                   'somite', 'neural_floor_plate', 'notochord']

print(f"\nLooking for clusters in developmental cell types:")
print(f"Target cell types: {', '.join(target_celltypes)}")

# Filter for target cell types
df_target = df_ranked[df_ranked['celltype'].isin(target_celltypes)]

if len(df_target) > 0:
    print(f"\nFound {len(df_target)} clusters in target cell types:")
    print(df_target[['cluster_id', 'celltype', 'dynamics_score',
                     'n_developmental_tfs', 'n_total_edges',
                     'developmental_tfs_list']].to_string(index=False))

    # Group by celltype to see TF representation
    print(f"\n{'='*80}")
    print("TF REPRESENTATION BY CELLTYPE:")
    print(f"{'='*80}")
    for celltype in target_celltypes:
        ct_clusters = df_target[df_target['celltype'] == celltype]
        if len(ct_clusters) > 0:
            all_tfs = set()
            for tfs_list in ct_clusters['developmental_tfs_list']:
                all_tfs.update(tfs_list)
            print(f"\n{celltype}: {len(ct_clusters)} clusters")
            print(f"  Developmental TFs: {', '.join(sorted(all_tfs))}")
else:
    print(f"\nNo clusters found in target cell types with current filters.")
    print(f"Showing top celltypes in results:")
    celltype_counts = df_ranked['celltype'].value_counts().head(10)
    for ct, count in celltype_counts.items():
        print(f"  {ct}: {count} clusters")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"Results saved to: {figpath}cluster_ranking_temporal_dynamics.csv")
print(f"Use df_ranked to explore further or run visualizations.")
