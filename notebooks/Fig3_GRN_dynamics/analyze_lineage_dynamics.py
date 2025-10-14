#!/usr/bin/env python
"""
Analyze lineage dynamics: examine how subGRNs change across developmental lineages
at peak timepoints (rather than across time).

Focus on divergent patterns between neural and mesodermal lineages.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

print("="*80)
print("LINEAGE DYNAMICS ANALYSIS")
print("="*80)

# Load data
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
ranking_csv = figpath + "cluster_ranking_temporal_dynamics.csv"
df_ranked = pd.read_csv(ranking_csv)

# Load GRN dictionary
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

# Define developmental TFs
developmental_tf_families = [
    'sox', 'pax', 'tbx', 'myod', 'neurog', 'gata', 'hand', 'hox',
    'foxa', 'nkx', 'olig', 'ascl', 'msgn', 'meox', 'tcf21', 'neurod'
]

# Define lineage-specific TFs
neural_tfs = ['sox2', 'sox3', 'pax6a', 'pax6b', 'neurod1', 'neurod4',
              'olig2', 'ascl1a', 'ascl1b', 'sox10', 'sox19a', 'sox19b']
mesoderm_tfs = ['tbx6', 'tbx16', 'myod1', 'myf5', 'msgn1', 'meox1',
                'meox2a', 'meox2b', 'pax3a', 'pax3b', 'pax7a', 'pax7b']

def is_developmental_tf(tf_name, families=developmental_tf_families):
    """Check if TF belongs to developmental TF families"""
    return any(family in tf_name.lower() for family in families)

def get_lineage_tf_type(tf_name):
    """Classify TF as neural, mesoderm, or both"""
    is_neural = tf_name in neural_tfs
    is_mesoderm = tf_name in mesoderm_tfs

    if is_neural and is_mesoderm:
        return 'both'
    elif is_neural:
        return 'neural'
    elif is_mesoderm:
        return 'mesoderm'
    else:
        return 'other'

def analyze_lineage_dynamics(cluster_id, peak_timepoint, grn_dict, cluster_tf_gene_matrices):
    """
    For a given cluster at its peak timepoint, analyze subGRN dynamics along lineages.

    Returns dict with:
    - lineage_bias_score: (neural - mesoderm) / (neural + mesoderm) [-1 to 1]
    - neural_trajectory: metrics along neural lineage
    - mesoderm_trajectory: metrics along mesodermal lineage
    - trajectory_pattern: progressive/peaked/dipped/flat
    - developmental_tfs: list of dev TFs in network
    """

    # Define lineage cell types
    # Neural: terminal ← intermediate ← progenitor
    neural_lineage = ['neural_posterior', 'spinal_cord', 'NMPs']

    # Mesodermal: progenitor → intermediate → terminal
    mesoderm_lineage = ['NMPs', 'tail_bud', 'PSM', 'somites', 'fast_muscle']

    if cluster_id not in cluster_tf_gene_matrices:
        return None

    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = set()
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.add((tf, gene))

    # Extract subGRNs for each cell type in both lineages
    def extract_subgrn_metrics(celltype, timepoint):
        """Extract metrics for a single celltype at timepoint"""
        if (celltype, timepoint) not in grn_dict:
            return None

        grn_df = grn_dict[(celltype, timepoint)]
        grn_pairs = set(zip(grn_df['source'], grn_df['target']))
        found_pairs = predicted_pairs & grn_pairs

        if len(found_pairs) == 0:
            return None

        mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
        subgrn = grn_df[mask].copy()

        if len(subgrn) < 10:  # Min 10 edges threshold
            return None

        # Compute metrics
        edge_count = len(subgrn)
        edge_weight = subgrn['coef_mean'].abs().sum()  # Total absolute weight
        nodes = set(subgrn['source']) | set(subgrn['target'])
        node_count = len(nodes)

        # Developmental TF metrics
        tfs = set(subgrn['source'])
        dev_tfs = [tf for tf in tfs if is_developmental_tf(tf)]
        n_dev_tfs = len(dev_tfs)

        # Lineage-specific TF counts
        neural_tf_count = sum(1 for tf in tfs if tf in neural_tfs)
        mesoderm_tf_count = sum(1 for tf in tfs if tf in mesoderm_tfs)

        return {
            'celltype': celltype,
            'edge_count': edge_count,
            'edge_weight': edge_weight,
            'node_count': node_count,
            'n_dev_tfs': n_dev_tfs,
            'dev_tfs': dev_tfs,
            'neural_tf_count': neural_tf_count,
            'mesoderm_tf_count': mesoderm_tf_count,
            'subgrn': subgrn
        }

    # Extract metrics for all lineage cell types
    neural_metrics = []
    for celltype in neural_lineage:
        metrics = extract_subgrn_metrics(celltype, peak_timepoint)
        if metrics is not None:
            neural_metrics.append(metrics)

    mesoderm_metrics = []
    for celltype in mesoderm_lineage:
        metrics = extract_subgrn_metrics(celltype, peak_timepoint)
        if metrics is not None:
            mesoderm_metrics.append(metrics)

    # Need at least 2 cell types in each lineage
    if len(neural_metrics) < 2 or len(mesoderm_metrics) < 2:
        return None

    # Compute lineage strengths (using both edge count and edge weight)
    def compute_lineage_strength(metrics_list):
        """Compute average strength across lineage"""
        edge_counts = [m['edge_count'] for m in metrics_list]
        edge_weights = [m['edge_weight'] for m in metrics_list]

        # Normalize and combine
        avg_edges = np.mean(edge_counts)
        avg_weight = np.mean(edge_weights)

        return avg_edges * 0.5 + avg_weight * 0.5  # Equal weighting

    neural_strength = compute_lineage_strength(neural_metrics)
    mesoderm_strength = compute_lineage_strength(mesoderm_metrics)

    # Compute lineage bias score
    total_strength = neural_strength + mesoderm_strength
    if total_strength == 0:
        return None

    lineage_bias_score = (neural_strength - mesoderm_strength) / total_strength

    # Detect trajectory patterns
    def detect_trajectory_pattern(metrics_list):
        """Detect if network shows progressive strengthening/weakening"""
        edge_counts = np.array([m['edge_count'] for m in metrics_list])

        if len(edge_counts) < 3:
            return 'insufficient_data', 0

        # Check for monotonic trends
        diffs = np.diff(edge_counts)
        increasing_score = np.sum(diffs > 0) / len(diffs)
        decreasing_score = np.sum(diffs < 0) / len(diffs)

        if increasing_score >= 0.6:
            return 'progressive_strengthening', increasing_score
        elif decreasing_score >= 0.6:
            return 'progressive_weakening', decreasing_score
        else:
            # Check for peak/dip
            max_idx = np.argmax(edge_counts)
            min_idx = np.argmin(edge_counts)

            if 0 < max_idx < len(edge_counts) - 1:
                return 'peaked', 0.5
            elif 0 < min_idx < len(edge_counts) - 1:
                return 'dipped', 0.5
            else:
                return 'flat', 0

    neural_pattern, neural_clarity = detect_trajectory_pattern(neural_metrics)
    mesoderm_pattern, mesoderm_clarity = detect_trajectory_pattern(mesoderm_metrics)

    # Collect all developmental TFs across both lineages
    all_dev_tfs = set()
    neural_dev_tfs = set()
    mesoderm_dev_tfs = set()

    for m in neural_metrics:
        neural_dev_tfs.update(m['dev_tfs'])
        all_dev_tfs.update(m['dev_tfs'])

    for m in mesoderm_metrics:
        mesoderm_dev_tfs.update(m['dev_tfs'])
        all_dev_tfs.update(m['dev_tfs'])

    # Compute scoring components
    lineage_specificity = abs(lineage_bias_score)  # 0 to 1
    trajectory_clarity = (neural_clarity + mesoderm_clarity) / 2

    # Network complexity (prefer moderate size)
    avg_edges = np.mean([m['edge_count'] for m in neural_metrics + mesoderm_metrics])
    avg_nodes = np.mean([m['node_count'] for m in neural_metrics + mesoderm_metrics])

    if 10 <= avg_edges <= 30 and 10 <= avg_nodes <= 25:
        network_complexity = 1.0
    elif 5 <= avg_edges <= 50 and 5 <= avg_nodes <= 40:
        network_complexity = 0.7
    else:
        network_complexity = 0.3

    # Developmental TF enrichment
    n_all_dev_tfs = len(all_dev_tfs)
    if n_all_dev_tfs >= 5:
        dev_tf_enrichment = 1.0
    elif n_all_dev_tfs >= 3:
        dev_tf_enrichment = 0.7
    elif n_all_dev_tfs >= 1:
        dev_tf_enrichment = 0.4
    else:
        dev_tf_enrichment = 0

    # Final score (emphasize divergent patterns)
    lineage_dynamics_score = (
        0.35 * lineage_specificity +      # How lineage-specific?
        0.25 * trajectory_clarity +        # Clear pattern along lineage?
        0.20 * network_complexity +        # Sufficient edges/nodes?
        0.20 * dev_tf_enrichment          # Known TFs present?
    )

    return {
        'cluster_id': cluster_id,
        'peak_timepoint': peak_timepoint,
        'lineage_bias_score': lineage_bias_score,
        'lineage_specificity': lineage_specificity,
        'neural_strength': neural_strength,
        'mesoderm_strength': mesoderm_strength,
        'neural_pattern': neural_pattern,
        'mesoderm_pattern': mesoderm_pattern,
        'neural_clarity': neural_clarity,
        'mesoderm_clarity': mesoderm_clarity,
        'trajectory_clarity': trajectory_clarity,
        'avg_edges': avg_edges,
        'avg_nodes': avg_nodes,
        'network_complexity': network_complexity,
        'n_all_dev_tfs': n_all_dev_tfs,
        'n_neural_dev_tfs': len(neural_dev_tfs),
        'n_mesoderm_dev_tfs': len(mesoderm_dev_tfs),
        'all_dev_tfs': sorted(list(all_dev_tfs)),
        'neural_dev_tfs': sorted(list(neural_dev_tfs)),
        'mesoderm_dev_tfs': sorted(list(mesoderm_dev_tfs)),
        'dev_tf_enrichment': dev_tf_enrichment,
        'lineage_dynamics_score': lineage_dynamics_score,
        'n_neural_celltypes': len(neural_metrics),
        'n_mesoderm_celltypes': len(mesoderm_metrics),
        'neural_celltypes': [m['celltype'] for m in neural_metrics],
        'mesoderm_celltypes': [m['celltype'] for m in mesoderm_metrics],
        'neural_edge_counts': [m['edge_count'] for m in neural_metrics],
        'mesoderm_edge_counts': [m['edge_count'] for m in mesoderm_metrics],
        'neural_edge_weights': [m['edge_weight'] for m in neural_metrics],
        'mesoderm_edge_weights': [m['edge_weight'] for m in mesoderm_metrics]
    }

# Analyze all clusters
print("\n" + "="*80)
print("ANALYZING LINEAGE DYNAMICS...")
print("="*80)

results = []
for idx, row in df_ranked.iterrows():
    cluster_id = row['cluster_id']
    celltype = row['celltype']

    # Get peak timepoint from the temporal ranking
    # Use the celltype as a proxy, but we'll use all available timepoints
    # Find timepoint with maximum accessibility for this cluster

    # For now, try all available timepoints and pick the one with strongest network
    available_timepoints = sorted(set([tp for (ct, tp) in grn_dict.keys()]))

    best_result = None
    best_score = -1

    for timepoint in available_timepoints:
        result = analyze_lineage_dynamics(cluster_id, timepoint, grn_dict, cluster_tf_gene_matrices)

        if result is not None:
            if result['lineage_dynamics_score'] > best_score:
                best_score = result['lineage_dynamics_score']
                best_result = result

    if best_result is not None:
        # Add original ranking info
        best_result['original_celltype'] = celltype
        best_result['temporal_dynamics_score'] = row['dynamics_score']
        results.append(best_result)

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(df_ranked)} clusters...")

df_lineage = pd.DataFrame(results)

print(f"\nFound {len(df_lineage)} clusters with valid lineage dynamics")

# Sort by lineage dynamics score
df_lineage = df_lineage.sort_values('lineage_dynamics_score', ascending=False)

# Show distribution of lineage bias
print("\n" + "="*80)
print("LINEAGE BIAS DISTRIBUTION:")
print("="*80)

neural_biased = df_lineage[df_lineage['lineage_bias_score'] > 0.3]
mesoderm_biased = df_lineage[df_lineage['lineage_bias_score'] < -0.3]
balanced = df_lineage[(df_lineage['lineage_bias_score'] >= -0.3) &
                       (df_lineage['lineage_bias_score'] <= 0.3)]

print(f"Neural-biased (bias > 0.3): {len(neural_biased)} clusters")
print(f"Mesoderm-biased (bias < -0.3): {len(mesoderm_biased)} clusters")
print(f"Balanced (|bias| <= 0.3): {len(balanced)} clusters")

# Show top 10 candidates
print("\n" + "="*80)
print("TOP 10 CANDIDATES WITH STRONGEST LINEAGE DYNAMICS:")
print("="*80)

for idx, (i, row) in enumerate(df_lineage.head(10).iterrows(), 1):
    lineage_type = "NEURAL" if row['lineage_bias_score'] > 0 else "MESODERM"

    print(f"\n{idx}. Cluster {row['cluster_id']} - {row['original_celltype']} ({lineage_type}-biased)")
    print(f"   Peak timepoint: {row['peak_timepoint']}")
    print(f"   Lineage dynamics score: {row['lineage_dynamics_score']:.3f}")
    print(f"   Lineage bias: {row['lineage_bias_score']:.3f} ({lineage_type})")
    print(f"   Neural strength: {row['neural_strength']:.1f}, Pattern: {row['neural_pattern']}")
    print(f"   Mesoderm strength: {row['mesoderm_strength']:.1f}, Pattern: {row['mesoderm_pattern']}")
    print(f"   Avg edges: {row['avg_edges']:.1f}, Avg nodes: {row['avg_nodes']:.1f}")
    print(f"   Neural celltypes ({row['n_neural_celltypes']}): {row['neural_celltypes']}")
    print(f"   Mesoderm celltypes ({row['n_mesoderm_celltypes']}): {row['mesoderm_celltypes']}")
    print(f"   Neural edge counts: {row['neural_edge_counts']}")
    print(f"   Mesoderm edge counts: {row['mesoderm_edge_counts']}")
    print(f"   All dev TFs ({row['n_all_dev_tfs']}): {row['all_dev_tfs']}")
    print(f"   Neural TFs ({row['n_neural_dev_tfs']}): {row['neural_dev_tfs']}")
    print(f"   Mesoderm TFs ({row['n_mesoderm_dev_tfs']}): {row['mesoderm_dev_tfs']}")

# Show top 5 neural-biased and top 5 mesoderm-biased
print("\n" + "="*80)
print("TOP 5 NEURAL-BIASED CANDIDATES:")
print("="*80)

for idx, (i, row) in enumerate(df_lineage[df_lineage['lineage_bias_score'] > 0].head(5).iterrows(), 1):
    print(f"\n{idx}. Cluster {row['cluster_id']} - {row['original_celltype']}")
    print(f"   Lineage bias: {row['lineage_bias_score']:.3f}")
    print(f"   Score: {row['lineage_dynamics_score']:.3f}")
    print(f"   Neural: {row['neural_strength']:.1f} ({row['neural_pattern']})")
    print(f"   Mesoderm: {row['mesoderm_strength']:.1f} ({row['mesoderm_pattern']})")

print("\n" + "="*80)
print("TOP 5 MESODERM-BIASED CANDIDATES:")
print("="*80)

for idx, (i, row) in enumerate(df_lineage[df_lineage['lineage_bias_score'] < 0].head(5).iterrows(), 1):
    print(f"\n{idx}. Cluster {row['cluster_id']} - {row['original_celltype']}")
    print(f"   Lineage bias: {row['lineage_bias_score']:.3f}")
    print(f"   Score: {row['lineage_dynamics_score']:.3f}")
    print(f"   Neural: {row['neural_strength']:.1f} ({row['neural_pattern']})")
    print(f"   Mesoderm: {row['mesoderm_strength']:.1f} ({row['mesoderm_pattern']})")

# Save results
output_csv = figpath + "lineage_dynamics_ranking.csv"
df_lineage.to_csv(output_csv, index=False)
print(f"\n{'='*80}")
print(f"Results saved to: {output_csv}")
print(f"{'='*80}")
