#!/usr/bin/env python
"""
Find clusters with moderate complexity (5-10 nodes per timepoint) and clear temporal patterns
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("="*80)
print("FINDING MODERATE COMPLEXITY CLUSTERS WITH CLEAR TEMPORAL PATTERNS")
print("="*80)

# Load data
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
ranking_csv = figpath + "cluster_ranking_temporal_dynamics.csv"
df_ranked = pd.read_csv(ranking_csv)

# Load GRN dictionary to get detailed temporal info
def load_grn_dict_pathlib(base_dir="grn_exports", grn_type="filtered"):
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

cluster_tf_gene_matrices_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl"
with open(cluster_tf_gene_matrices_path, 'rb') as f:
    cluster_tf_gene_matrices = pickle.load(f)

print(f"\nLoaded {len(df_ranked)} ranked clusters")
print(f"Loaded {len(grn_dict)} GRN combinations")
print(f"Loaded {len(cluster_tf_gene_matrices)} cluster matrices")

# Function to compute detailed temporal metrics
def analyze_temporal_pattern(cluster_id, celltype, grn_dict, cluster_tf_gene_matrices):
    """Compute detailed temporal metrics for a cluster"""

    if cluster_id not in cluster_tf_gene_matrices:
        return None

    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = set()
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.add((tf, gene))

    # Get timepoints for this celltype
    timepoints = sorted([tp for (ct, tp) in grn_dict.keys() if ct == celltype])

    if len(timepoints) < 3:
        return None

    # Extract subGRNs per timepoint
    edge_counts = []
    node_counts = []
    subgrns = {}

    for timepoint in timepoints:
        if (celltype, timepoint) not in grn_dict:
            edge_counts.append(0)
            node_counts.append(0)
            continue

        grn_df = grn_dict[(celltype, timepoint)]
        grn_pairs = set(zip(grn_df['source'], grn_df['target']))
        found_pairs = predicted_pairs & grn_pairs

        mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
        subgrn = grn_df[mask].copy()

        subgrns[timepoint] = subgrn
        edge_counts.append(len(subgrn))

        if len(subgrn) > 0:
            nodes = set(subgrn['source']) | set(subgrn['target'])
            node_counts.append(len(nodes))
        else:
            node_counts.append(0)

    # Calculate temporal pattern metrics
    edge_counts_arr = np.array(edge_counts)
    node_counts_arr = np.array(node_counts)

    # Filter to timepoints with at least 10 edges
    valid_indices = edge_counts_arr >= 10
    if sum(valid_indices) < 3:
        return None

    valid_edge_counts = edge_counts_arr[valid_indices]
    valid_node_counts = node_counts_arr[valid_indices]

    # Check node count range (5-15 nodes per timepoint on average) - relaxed
    avg_nodes = np.mean(valid_node_counts)
    if avg_nodes < 5 or avg_nodes > 20:  # Relaxed upper bound
        return None

    # Detect temporal pattern
    # 1. Monotonic increasing
    increasing_score = np.sum(np.diff(valid_edge_counts) > 0) / (len(valid_edge_counts) - 1)

    # 2. Monotonic decreasing
    decreasing_score = np.sum(np.diff(valid_edge_counts) < 0) / (len(valid_edge_counts) - 1)

    # 3. Peak in middle (inverted-U)
    mid_idx = len(valid_edge_counts) // 2
    if len(valid_edge_counts) >= 4:
        peak_in_middle = (valid_edge_counts[mid_idx-1:mid_idx+2].max() ==
                         valid_edge_counts.max())
    else:
        peak_in_middle = False

    # 4. Valley in middle (U-shaped)
    if len(valid_edge_counts) >= 4:
        valley_in_middle = (valid_edge_counts[mid_idx-1:mid_idx+2].min() ==
                           valid_edge_counts.min())
    else:
        valley_in_middle = False

    # Determine pattern type - relaxed thresholds
    pattern_type = "complex"
    pattern_score = 0

    if increasing_score >= 0.6:  # Relaxed from 0.7
        pattern_type = "increasing"
        pattern_score = increasing_score
    elif decreasing_score >= 0.6:  # Relaxed from 0.7
        pattern_type = "decreasing"
        pattern_score = decreasing_score
    elif peak_in_middle:
        pattern_type = "peak"
        pattern_score = 0.8
    elif valley_in_middle:
        pattern_type = "valley"
        pattern_score = 0.7
    elif max(increasing_score, decreasing_score) >= 0.5:  # Partial trends
        if increasing_score > decreasing_score:
            pattern_type = "increasing"
            pattern_score = increasing_score
        else:
            pattern_type = "decreasing"
            pattern_score = decreasing_score

    return {
        'cluster_id': cluster_id,
        'celltype': celltype,
        'timepoints': timepoints,
        'edge_counts': edge_counts,
        'node_counts': node_counts,
        'avg_nodes': avg_nodes,
        'avg_edges': np.mean(valid_edge_counts),
        'pattern_type': pattern_type,
        'pattern_score': pattern_score,
        'increasing_score': increasing_score,
        'decreasing_score': decreasing_score,
        'peak_in_middle': peak_in_middle,
        'valley_in_middle': valley_in_middle,
        'n_valid_timepoints': sum(valid_indices)
    }

# Analyze all clusters
print("\n" + "="*80)
print("ANALYZING TEMPORAL PATTERNS...")
print("="*80)

results = []
for idx, row in df_ranked.iterrows():
    cluster_id = row['cluster_id']
    celltype = row['celltype']

    pattern_info = analyze_temporal_pattern(cluster_id, celltype, grn_dict, cluster_tf_gene_matrices)

    if pattern_info is not None:
        # Add ranking info
        pattern_info['dynamics_score'] = row['dynamics_score']
        pattern_info['n_developmental_tfs'] = row['n_developmental_tfs']
        pattern_info['developmental_tfs_list'] = row['developmental_tfs_list']
        results.append(pattern_info)

df_patterns = pd.DataFrame(results)

print(f"\nFound {len(df_patterns)} clusters with moderate complexity and clear patterns")

# Filter and rank by pattern clarity
df_patterns['pattern_clarity'] = df_patterns['pattern_score']

# Prioritize developmental celltypes
dev_celltypes = ['PSM', 'NMPs', 'tail_bud', 'neural_posterior', 'spinal_cord',
                'somite', 'neural_floor_plate', 'notochord', 'neural_crest',
                'enteric_neurons', 'differentiating_neurons']

df_patterns['is_dev_celltype'] = df_patterns['celltype'].isin(dev_celltypes)

# Sort by pattern clarity and developmental relevance
df_patterns = df_patterns.sort_values(
    ['is_dev_celltype', 'pattern_clarity', 'avg_nodes'],
    ascending=[False, False, True]
)

# Group by pattern type
print("\n" + "="*80)
print("PATTERN TYPE DISTRIBUTION:")
print("="*80)
for pattern_type in ['increasing', 'decreasing', 'peak', 'valley', 'complex']:
    count = len(df_patterns[df_patterns['pattern_type'] == pattern_type])
    print(f"{pattern_type.upper()}: {count} clusters")

# Show top candidates by pattern type
print("\n" + "="*80)
print("TOP CANDIDATES BY PATTERN TYPE:")
print("="*80)

for pattern_type in ['increasing', 'decreasing', 'peak', 'valley']:
    df_type = df_patterns[df_patterns['pattern_type'] == pattern_type]

    if len(df_type) > 0:
        print(f"\n{pattern_type.upper()} PATTERN (showing top 3):")
        print("-" * 80)

        for idx, row in df_type.head(3).iterrows():
            print(f"\nCluster {row['cluster_id']} - {row['celltype']}")
            print(f"  Avg nodes: {row['avg_nodes']:.1f}, Avg edges: {row['avg_edges']:.1f}")
            print(f"  Pattern score: {row['pattern_score']:.2f}")
            print(f"  Edge counts: {row['edge_counts']}")
            print(f"  Node counts: {row['node_counts']}")
            print(f"  Dev TFs ({row['n_developmental_tfs']}): {row['developmental_tfs_list']}")

# Select top 10 diverse candidates
print("\n" + "="*80)
print("TOP 10 DIVERSE CANDIDATES FOR MANUSCRIPT:")
print("="*80)

# Get 3 from each pattern type (increasing, decreasing, peak) + 1 valley
top_candidates = []

for pattern_type, n_select in [('peak', 3), ('increasing', 3), ('decreasing', 3), ('valley', 1)]:
    df_type = df_patterns[df_patterns['pattern_type'] == pattern_type]
    candidates = df_type.head(n_select)
    top_candidates.append(candidates)

df_top10 = pd.concat(top_candidates).head(10)

for idx, (i, row) in enumerate(df_top10.iterrows(), 1):
    print(f"\n{idx}. Cluster {row['cluster_id']} - {row['celltype']} ({row['pattern_type'].upper()})")
    print(f"   Dynamics score: {row['dynamics_score']:.3f}")
    print(f"   Avg nodes: {row['avg_nodes']:.1f}, Avg edges: {row['avg_edges']:.1f}")
    print(f"   Pattern score: {row['pattern_score']:.2f}")
    print(f"   Edge trajectory: {row['edge_counts']}")
    print(f"   Dev TFs ({row['n_developmental_tfs']}): {row['developmental_tfs_list']}")

# Save results
output_csv = figpath + "moderate_complexity_candidates.csv"
df_top10.to_csv(output_csv, index=False)
print(f"\n{'='*80}")
print(f"Results saved to: {output_csv}")
print(f"{'='*80}")
