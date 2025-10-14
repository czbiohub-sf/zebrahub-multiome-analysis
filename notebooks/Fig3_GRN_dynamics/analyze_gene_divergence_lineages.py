#!/usr/bin/env python
"""
Analyze progressive divergence of TFs and target genes across lineages.
Find clusters where TFs or target genes gradually diverge between neural and mesodermal lineages.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("="*80)
print("ANALYZING TF AND TARGET GENE DIVERGENCE ACROSS LINEAGES")
print("="*80)

# Load data
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
ranking_csv = figpath + "lineage_dynamics_ranking.csv"
df_lineage = pd.read_csv(ranking_csv)

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

print(f"\nLoaded {len(df_lineage)} clusters with lineage dynamics")
print(f"Loaded {len(grn_dict)} GRN combinations")

def analyze_gene_divergence(cluster_id, peak_timepoint, grn_dict, cluster_tf_gene_matrices):
    """
    Analyze how TFs and target genes diverge progressively across lineages.

    Returns metrics:
    - TF overlap across lineage (Jaccard similarity)
    - Target gene overlap across lineage
    - Progressive divergence score (how gradually sets change)
    """

    # Define lineages
    neural_lineage = ['neural_posterior', 'spinal_cord', 'NMPs']
    mesoderm_lineage = ['NMPs', 'tail_bud', 'PSM', 'somites', 'fast_muscle']

    if cluster_id not in cluster_tf_gene_matrices:
        return None

    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = set()
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.add((tf, gene))

    # Extract subGRNs for each lineage
    def extract_lineage_genes(lineage, timepoint):
        """Extract TFs and target genes for each celltype in lineage"""
        lineage_data = {}

        for celltype in lineage:
            if (celltype, timepoint) not in grn_dict:
                continue

            grn_df = grn_dict[(celltype, timepoint)]
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = predicted_pairs & grn_pairs

            if len(found_pairs) < 10:
                continue

            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()

            tfs = set(subgrn['source'])
            targets = set(subgrn['target'])

            lineage_data[celltype] = {
                'tfs': tfs,
                'targets': targets,
                'nodes': tfs | targets,
                'edges': len(subgrn)
            }

        return lineage_data

    neural_data = extract_lineage_genes(neural_lineage, peak_timepoint)
    mesoderm_data = extract_lineage_genes(mesoderm_lineage, peak_timepoint)

    if len(neural_data) < 2 or len(mesoderm_data) < 2:
        return None

    # Compute progressive divergence
    def compute_jaccard(set1, set2):
        """Jaccard similarity: |intersection| / |union|"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0

    # Get TF and target sets for each lineage
    neural_tfs_list = [data['tfs'] for data in neural_data.values()]
    neural_targets_list = [data['targets'] for data in neural_data.values()]
    mesoderm_tfs_list = [data['tfs'] for data in mesoderm_data.values()]
    mesoderm_targets_list = [data['targets'] for data in mesoderm_data.values()]

    # TF divergence: Compare TF sets between lineages at NMPs branch point
    # NMPs should be present in both lineages
    if 'NMPs' in neural_data and 'NMPs' in mesoderm_data:
        nmps_tf_overlap = compute_jaccard(neural_data['NMPs']['tfs'],
                                          mesoderm_data['NMPs']['tfs'])
        nmps_target_overlap = compute_jaccard(neural_data['NMPs']['targets'],
                                              mesoderm_data['NMPs']['targets'])
    else:
        nmps_tf_overlap = None
        nmps_target_overlap = None

    # Compute terminal divergence (most differentiated cells)
    neural_terminal = list(neural_data.keys())[0]  # neural_posterior
    mesoderm_terminal = list(mesoderm_data.keys())[-1]  # fast_muscle or last available

    terminal_tf_overlap = compute_jaccard(neural_data[neural_terminal]['tfs'],
                                          mesoderm_data[mesoderm_terminal]['tfs'])
    terminal_target_overlap = compute_jaccard(neural_data[neural_terminal]['targets'],
                                              mesoderm_data[mesoderm_terminal]['targets'])

    # Progressive divergence: Track overlap changes along lineages
    def compute_progressive_divergence(neural_list, mesoderm_list):
        """
        Compare neural lineage progression vs mesoderm lineage progression.
        High score = gradual divergence, Low score = sudden change
        """
        if len(neural_list) < 2 or len(mesoderm_list) < 2:
            return 0, []

        # Compute all pairwise overlaps
        overlaps = []
        for i, neural_set in enumerate(neural_list):
            for j, mesoderm_set in enumerate(mesoderm_list):
                overlap = compute_jaccard(neural_set, mesoderm_set)
                overlaps.append({
                    'neural_idx': i,
                    'mesoderm_idx': j,
                    'overlap': overlap
                })

        # Check if overlap decreases progressively
        # (from progenitor to terminal cells)
        if len(overlaps) > 0:
            overlaps_sorted = sorted(overlaps, key=lambda x: (x['neural_idx'], x['mesoderm_idx']))
            overlap_values = [o['overlap'] for o in overlaps_sorted]

            # Progressive divergence = negative correlation with distance from NMPs
            # Higher score if overlap decreases as cells differentiate
            avg_overlap = np.mean(overlap_values)
            overlap_std = np.std(overlap_values)

            return avg_overlap, overlap_values

        return 0, []

    tf_avg_overlap, tf_overlaps = compute_progressive_divergence(neural_tfs_list, mesoderm_tfs_list)
    target_avg_overlap, target_overlaps = compute_progressive_divergence(neural_targets_list, mesoderm_targets_list)

    # Divergence score: Lower overlap = higher divergence
    tf_divergence = 1 - tf_avg_overlap
    target_divergence = 1 - target_avg_overlap

    # Progressive divergence score: Check if divergence increases gradually
    def compute_progressive_score(neural_list, mesoderm_list):
        """Score based on whether sets become more different over lineage"""
        if len(neural_list) < 2 or len(mesoderm_list) < 2:
            return 0

        # Compare early (NMPs-adjacent) vs late (terminal)
        early_neural = neural_list[-1]  # NMPs is last in neural_lineage list
        early_mesoderm = mesoderm_list[0]  # NMPs is first in mesoderm_lineage list
        late_neural = neural_list[0]  # neural_posterior
        late_mesoderm = mesoderm_list[-1]  # fast_muscle

        early_overlap = compute_jaccard(early_neural, early_mesoderm)
        late_overlap = compute_jaccard(late_neural, late_mesoderm)

        # Progressive score: How much does overlap decrease?
        progressive_score = max(0, early_overlap - late_overlap)

        return progressive_score

    tf_progressive = compute_progressive_score(neural_tfs_list, mesoderm_tfs_list)
    target_progressive = compute_progressive_score(neural_targets_list, mesoderm_targets_list)

    # Compute lineage-specific genes
    all_neural_tfs = set()
    all_neural_targets = set()
    for data in neural_data.values():
        all_neural_tfs.update(data['tfs'])
        all_neural_targets.update(data['targets'])

    all_mesoderm_tfs = set()
    all_mesoderm_targets = set()
    for data in mesoderm_data.values():
        all_mesoderm_tfs.update(data['tfs'])
        all_mesoderm_targets.update(data['targets'])

    neural_specific_tfs = all_neural_tfs - all_mesoderm_tfs
    mesoderm_specific_tfs = all_mesoderm_tfs - all_neural_tfs
    shared_tfs = all_neural_tfs & all_mesoderm_tfs

    neural_specific_targets = all_neural_targets - all_mesoderm_targets
    mesoderm_specific_targets = all_mesoderm_targets - all_neural_targets
    shared_targets = all_neural_targets & all_mesoderm_targets

    return {
        'cluster_id': cluster_id,
        'peak_timepoint': peak_timepoint,
        'nmps_tf_overlap': nmps_tf_overlap,
        'nmps_target_overlap': nmps_target_overlap,
        'terminal_tf_overlap': terminal_tf_overlap,
        'terminal_target_overlap': terminal_target_overlap,
        'tf_avg_overlap': tf_avg_overlap,
        'target_avg_overlap': target_avg_overlap,
        'tf_divergence': tf_divergence,
        'target_divergence': target_divergence,
        'tf_progressive_score': tf_progressive,
        'target_progressive_score': target_progressive,
        'n_neural_specific_tfs': len(neural_specific_tfs),
        'n_mesoderm_specific_tfs': len(mesoderm_specific_tfs),
        'n_shared_tfs': len(shared_tfs),
        'n_neural_specific_targets': len(neural_specific_targets),
        'n_mesoderm_specific_targets': len(mesoderm_specific_targets),
        'n_shared_targets': len(shared_targets),
        'neural_specific_tfs': sorted(list(neural_specific_tfs)),
        'mesoderm_specific_tfs': sorted(list(mesoderm_specific_tfs)),
        'neural_specific_targets': sorted(list(neural_specific_targets)),
        'mesoderm_specific_targets': sorted(list(mesoderm_specific_targets))
    }

# Analyze all clusters
print("\n" + "="*80)
print("ANALYZING GENE DIVERGENCE...")
print("="*80)

results = []
for idx, row in df_lineage.iterrows():
    cluster_id = row['cluster_id']
    peak_timepoint = str(row['peak_timepoint']).zfill(2)

    divergence_info = analyze_gene_divergence(cluster_id, peak_timepoint, grn_dict, cluster_tf_gene_matrices)

    if divergence_info is not None:
        # Add original ranking info
        divergence_info['original_celltype'] = row['original_celltype']
        divergence_info['lineage_bias_score'] = row['lineage_bias_score']
        divergence_info['lineage_dynamics_score'] = row['lineage_dynamics_score']
        results.append(divergence_info)

df_divergence = pd.DataFrame(results)

print(f"\nAnalyzed {len(df_divergence)} clusters for gene divergence")

# Sort by progressive divergence (want gradual changes)
df_divergence['combined_progressive_score'] = (
    df_divergence['tf_progressive_score'] +
    df_divergence['target_progressive_score']
)

df_divergence = df_divergence.sort_values('combined_progressive_score', ascending=False)

# Show top candidates with progressive divergence
print("\n" + "="*80)
print("TOP 10 CLUSTERS WITH PROGRESSIVE GENE DIVERGENCE:")
print("="*80)

for idx, (i, row) in enumerate(df_divergence.head(10).iterrows(), 1):
    print(f"\n{idx}. Cluster {row['cluster_id']} - {row['original_celltype']}")
    print(f"   Peak timepoint: {row['peak_timepoint']} somites")
    print(f"   Lineage bias: {row['lineage_bias_score']:.3f}")
    print(f"   Combined progressive score: {row['combined_progressive_score']:.3f}")
    print(f"   TF progressive score: {row['tf_progressive_score']:.3f}")
    print(f"   Target progressive score: {row['target_progressive_score']:.3f}")
    print(f"   TF divergence: {row['tf_divergence']:.3f}")
    print(f"   Target divergence: {row['target_divergence']:.3f}")
    print(f"   Neural-specific TFs: {row['n_neural_specific_tfs']}")
    print(f"   Mesoderm-specific TFs: {row['n_mesoderm_specific_tfs']}")
    print(f"   Neural-specific targets: {row['n_neural_specific_targets']}")
    print(f"   Mesoderm-specific targets: {row['n_mesoderm_specific_targets']}")
    if row['n_neural_specific_tfs'] > 0:
        print(f"   Neural TFs: {row['neural_specific_tfs'][:5]}...")
    if row['n_mesoderm_specific_tfs'] > 0:
        print(f"   Mesoderm TFs: {row['mesoderm_specific_tfs'][:5]}...")

# Save results
output_csv = figpath + "lineage_gene_divergence_ranking.csv"
df_divergence.to_csv(output_csv, index=False)
print(f"\n{'='*80}")
print(f"Results saved to: {output_csv}")
print(f"{'='*80}")
