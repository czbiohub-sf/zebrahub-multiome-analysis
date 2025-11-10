#!/usr/bin/env python3
"""
Query Celltype-Specific TFs from Systematic Analysis
====================================================

Identifies transcription factors that show specificity to particular celltypes
or lineages (present in 3-5 celltypes, ideally within the same developmental lineage).

This script:
1. Recomputes TF-celltype presence across all clusters
2. Calculates lineage coherence scores
3. Identifies celltype-specific TFs
4. Generates CSV summary and markdown report

Usage:
    python query_celltype_specific_tfs.py

Outputs:
    - celltype_specific_tfs_summary.csv
    - CELLTYPE_SPECIFIC_TFS_REPORT.md
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_PATH = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
DATA_PATH = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data"
FIG_PATH = f"{BASE_PATH}/figures/sub_GRNs_reg_programs"

# Specificity parameters
MIN_CELLTYPES = 3      # Minimum celltypes for "specific"
MAX_CELLTYPES = 5      # Maximum celltypes for "specific"
MIN_LINEAGE_COHERENCE = 0.60  # Minimum % from dominant lineage
MIN_CLUSTERS = 1       # Minimum clusters to avoid artifacts

# ============================================================================
# LINEAGE GROUPINGS
# ============================================================================

LINEAGE_GROUPS = {
    'neuroectoderm': [
        'neural', 'neural_floor_plate', 'neural_optic', 'neural_posterior',
        'neural_crest', 'spinal_cord', 'differentiating_neurons', 'enteric_neurons'
    ],
    'mesoderm': [
        'PSM', 'somites', 'fast_muscle', 'slow_muscle', 'muscle',
        'heart_myocardium', 'lateral_plate_mesoderm', 'hemangioblasts',
        'hematopoietic_vasculature', 'notochord'
    ],
    'endoderm': [
        'endoderm', 'endocrine_pancreas', 'liver', 'pharyngeal_endoderm'
    ],
    'periderm_ectoderm': [
        'epidermis', 'periderm', 'hatching_gland'
    ],
    'other': [
        'pronephros', 'optic_cup', 'tail_bud', 'NMPs'
    ]
}

# Create reverse mapping: celltype -> lineage
CELLTYPE_TO_LINEAGE = {}
for lineage, celltypes in LINEAGE_GROUPS.items():
    for ct in celltypes:
        CELLTYPE_TO_LINEAGE[ct] = lineage

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_lineage(celltype):
    """Get lineage for a celltype."""
    return CELLTYPE_TO_LINEAGE.get(celltype, 'other')

def calculate_lineage_coherence(celltypes):
    """
    Calculate what fraction of celltypes belong to the dominant lineage.

    Returns:
        dominant_lineage: str
        coherence_score: float (0-1)
    """
    if not celltypes:
        return 'none', 0.0

    lineage_counts = defaultdict(int)
    for ct in celltypes:
        lineage = get_lineage(ct)
        lineage_counts[lineage] += 1

    dominant_lineage = max(lineage_counts.items(), key=lambda x: x[1])[0]
    coherence_score = lineage_counts[dominant_lineage] / len(celltypes)

    return dominant_lineage, coherence_score

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def load_all_data():
    """Load all required data: peak info, TF-gene matrices, and GRNs."""
    print("Loading all data...")

    # 1. Load temporal summary to get peak info
    temporal_df = pd.read_csv(f"{FIG_PATH}/systematic_analysis_temporal_summary.csv")

    # Create peak_info dictionary
    peak_info = {}
    for _, row in temporal_df.iterrows():
        peak_info[row['cluster_id']] = {
            'peak_celltype': row['peak_celltype'],
            'peak_timepoint': f"{int(row['peak_timepoint']):02d}"  # Format as '00', '05', etc.
        }

    print(f"   ✓ Loaded peak info for {len(peak_info)} clusters")

    # 2. Load TF-gene matrices (346 clusters)
    cluster_tf_gene_matrices_path = f"{DATA_PATH}/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl"
    with open(cluster_tf_gene_matrices_path, 'rb') as f:
        cluster_tf_gene_matrices = pickle.load(f)
    print(f"   ✓ Loaded {len(cluster_tf_gene_matrices)} TF-gene matrices")

    # 3. Load GRN dictionary
    def load_grn_dict_pathlib(base_dir, grn_type="filtered"):
        from pathlib import Path
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
        base_dir=f"{DATA_PATH}/processed_data/04_celloracle_celltype_GRNs_ML_annotation/grn_exported/",
        grn_type="filtered"
    )
    print(f"   ✓ Loaded {len(grn_dict)} celltype×timepoint GRN combinations")

    return peak_info, cluster_tf_gene_matrices, grn_dict

def build_tf_celltype_matrix(peak_info, cluster_tf_gene_matrices, grn_dict):
    """
    Build TF-celltype presence matrix across all clusters.

    For each cluster at its peak timepoint, extract subGRNs across all celltypes
    and track which TFs appear in which celltypes.

    Returns:
        tf_celltype_data: dict mapping TF -> list of (celltype, cluster_id) tuples
    """
    print("\nBuilding TF-celltype presence matrix...")

    # For each TF, track which celltypes it appears in
    tf_celltype_data = defaultdict(list)

    processed = 0
    for cluster_id, peak in peak_info.items():
        processed += 1
        if processed % 50 == 0:
            print(f"   Processing cluster {processed}/{len(peak_info)}...")

        peak_timepoint = peak['peak_timepoint']

        # Get predicted pairs for this cluster
        if cluster_id not in cluster_tf_gene_matrices:
            continue

        cluster_matrix = cluster_tf_gene_matrices[cluster_id]
        predicted_pairs = []
        for tf in cluster_matrix.index:
            for gene in cluster_matrix.columns:
                if cluster_matrix.loc[tf, gene] == 1:
                    predicted_pairs.append((tf, gene))

        # Extract subGRNs across all celltypes at peak timepoint
        for (celltype, timepoint), grn_df in grn_dict.items():
            if timepoint == peak_timepoint:
                # Find intersection of predicted pairs and GRN pairs
                grn_pairs = set(zip(grn_df['source'], grn_df['target']))
                found_pairs = set(predicted_pairs) & grn_pairs

                if len(found_pairs) > 0:
                    # Extract subGRN
                    mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
                    subgrn = grn_df[mask].copy()

                    if len(subgrn) > 0 and 'source' in subgrn.columns:
                        # Get all TFs (sources) in this subGRN
                        tfs = subgrn['source'].unique()

                        for tf in tfs:
                            tf_celltype_data[tf].append((celltype, cluster_id))

    print(f"   ✓ Found {len(tf_celltype_data)} unique TFs across all celltypes")

    return tf_celltype_data

def calculate_tf_specificity_metrics(tf_celltype_data):
    """
    Calculate specificity metrics for each TF.

    Returns DataFrame with columns:
        - tf_name
        - n_celltypes
        - celltypes_list
        - n_clusters
        - cluster_ids
        - dominant_lineage
        - lineage_coherence
    """
    print("\nCalculating specificity metrics...")

    tf_metrics = []

    for tf, celltype_cluster_list in tf_celltype_data.items():
        # Extract unique celltypes and clusters
        celltypes = list(set(ct for ct, _ in celltype_cluster_list))
        clusters = list(set(cid for _, cid in celltype_cluster_list))

        # Calculate lineage coherence
        dominant_lineage, coherence = calculate_lineage_coherence(celltypes)

        tf_metrics.append({
            'tf_name': tf,
            'n_celltypes': len(celltypes),
            'celltypes_list': ','.join(sorted(celltypes)),
            'n_clusters': len(clusters),
            'cluster_ids': ','.join(sorted(clusters)),
            'dominant_lineage': dominant_lineage,
            'lineage_coherence': coherence
        })

    df = pd.DataFrame(tf_metrics)
    print(f"   ✓ Computed metrics for {len(df)} TFs")

    return df

def filter_specific_tfs(tf_metrics_df):
    """
    Filter for celltype-specific TFs based on criteria.

    Criteria:
        - Present in MIN_CELLTYPES to MAX_CELLTYPES celltypes
        - Lineage coherence >= MIN_LINEAGE_COHERENCE
        - Present in >= MIN_CLUSTERS clusters
    """
    print(f"\nFiltering for celltype-specific TFs...")
    print(f"   Criteria: {MIN_CELLTYPES}-{MAX_CELLTYPES} celltypes, "
          f"≥{MIN_LINEAGE_COHERENCE:.0%} lineage coherence, "
          f"≥{MIN_CLUSTERS} clusters")

    specific_df = tf_metrics_df[
        (tf_metrics_df['n_celltypes'] >= MIN_CELLTYPES) &
        (tf_metrics_df['n_celltypes'] <= MAX_CELLTYPES) &
        (tf_metrics_df['lineage_coherence'] >= MIN_LINEAGE_COHERENCE) &
        (tf_metrics_df['n_clusters'] >= MIN_CLUSTERS)
    ].copy()

    # Sort by lineage coherence (desc), then n_celltypes (asc)
    specific_df = specific_df.sort_values(
        ['lineage_coherence', 'n_celltypes'],
        ascending=[False, True]
    )

    print(f"   ✓ Found {len(specific_df)} celltype-specific TFs")

    return specific_df

# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def write_csv_summary(specific_tfs_df, output_path):
    """Write CSV summary of celltype-specific TFs."""
    specific_tfs_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")

def write_markdown_report(specific_tfs_df, all_tfs_df, output_path):
    """Write comprehensive markdown report."""

    with open(output_path, 'w') as f:
        f.write("# Celltype-Specific Transcription Factors\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write("This report identifies transcription factors that show specificity to particular ")
        f.write("celltypes or developmental lineages.\n\n")
        f.write("**Definition of 'specific':** Present in 3-5 celltypes with high lineage coherence ")
        f.write(f"(≥{MIN_LINEAGE_COHERENCE:.0%} of celltypes from the same developmental lineage).\n\n")

        # Summary statistics
        f.write("### Summary Statistics\n\n")
        f.write(f"- **Total TFs analyzed:** {len(all_tfs_df)}\n")
        f.write(f"- **Celltype-specific TFs:** {len(specific_tfs_df)}\n")
        f.write(f"- **Ultra-specific (3 celltypes):** {len(specific_tfs_df[specific_tfs_df['n_celltypes'] == 3])}\n")
        f.write(f"- **Moderately specific (4-5 celltypes):** {len(specific_tfs_df[specific_tfs_df['n_celltypes'] >= 4])}\n\n")

        # Distribution by lineage
        f.write("### Distribution by Lineage\n\n")
        lineage_counts = specific_tfs_df['dominant_lineage'].value_counts()
        f.write("| Lineage | N Specific TFs |\n")
        f.write("|---------|----------------|\n")
        for lineage, count in lineage_counts.items():
            f.write(f"| {lineage} | {count} |\n")
        f.write("\n")

        # Top 50 most specific TFs
        f.write("## Top 50 Most Specific TFs\n\n")
        f.write("Ranked by lineage coherence (descending), then number of celltypes (ascending).\n\n")
        f.write("| Rank | TF | N CTs | Celltypes | Lineage | Coherence | N Clusters |\n")
        f.write("|------|-------|-------|-----------|---------|-----------|------------|\n")

        for idx, row in specific_tfs_df.head(50).iterrows():
            celltypes_abbrev = row['celltypes_list'][:60] + '...' if len(row['celltypes_list']) > 60 else row['celltypes_list']
            f.write(f"| {idx+1} | {row['tf_name']} | {row['n_celltypes']} | {celltypes_abbrev} | "
                   f"{row['dominant_lineage']} | {row['lineage_coherence']:.2f} | {row['n_clusters']} |\n")
        f.write("\n")

        # Detailed sections by lineage
        f.write("## Detailed Breakdown by Lineage\n\n")

        for lineage in ['neuroectoderm', 'mesoderm', 'endoderm', 'periderm_ectoderm', 'other']:
            lineage_tfs = specific_tfs_df[specific_tfs_df['dominant_lineage'] == lineage]

            if len(lineage_tfs) == 0:
                continue

            f.write(f"### {lineage.replace('_', ' ').title()}\n\n")
            f.write(f"**Total specific TFs:** {len(lineage_tfs)}\n\n")

            # Ultra-specific (3 celltypes)
            ultra = lineage_tfs[lineage_tfs['n_celltypes'] == 3]
            if len(ultra) > 0:
                f.write(f"#### Ultra-Specific (3 celltypes) - {len(ultra)} TFs\n\n")
                f.write("| TF | Celltypes | Coherence | N Clusters |\n")
                f.write("|----|-----------|-----------|------------|\n")
                for _, row in ultra.head(20).iterrows():
                    f.write(f"| {row['tf_name']} | {row['celltypes_list']} | "
                           f"{row['lineage_coherence']:.2f} | {row['n_clusters']} |\n")
                f.write("\n")

            # Moderately specific (4-5 celltypes)
            moderate = lineage_tfs[lineage_tfs['n_celltypes'] >= 4]
            if len(moderate) > 0:
                f.write(f"#### Moderately Specific (4-5 celltypes) - {len(moderate)} TFs\n\n")
                f.write("| TF | N CTs | Celltypes | Coherence | N Clusters |\n")
                f.write("|-------|-------|-----------|-----------|------------|\n")
                for _, row in moderate.head(20).iterrows():
                    celltypes_abbrev = row['celltypes_list'][:50] + '...' if len(row['celltypes_list']) > 50 else row['celltypes_list']
                    f.write(f"| {row['tf_name']} | {row['n_celltypes']} | {celltypes_abbrev} | "
                           f"{row['lineage_coherence']:.2f} | {row['n_clusters']} |\n")
                f.write("\n")

        # Known lineage markers section
        f.write("## Known Lineage Markers Validation\n\n")
        f.write("Checking if well-known lineage markers appear as expected:\n\n")

        known_markers = {
            'tbx6': 'mesoderm (PSM)',
            'tbxta': 'mesoderm (notochord)',
            'nkx2.5': 'mesoderm (heart)',
            'sox2': 'neuroectoderm',
            'pax6a': 'neuroectoderm (optic)',
            'foxa2': 'endoderm',
            'sox17': 'endoderm',
            'krt4': 'periderm/ectoderm'
        }

        f.write("| Marker | Expected Lineage | Found | N CTs | Celltypes | Coherence |\n")
        f.write("|--------|-----------------|-------|-------|-----------|----------|\n")

        for marker, expected in known_markers.items():
            match = all_tfs_df[all_tfs_df['tf_name'] == marker]
            if len(match) > 0:
                row = match.iloc[0]
                found = "✓" if row['n_celltypes'] <= MAX_CELLTYPES else f"✗ (too broad: {row['n_celltypes']} CTs)"
                celltypes_abbrev = row['celltypes_list'][:40] + '...' if len(row['celltypes_list']) > 40 else row['celltypes_list']
                f.write(f"| {marker} | {expected} | {found} | {row['n_celltypes']} | "
                       f"{celltypes_abbrev} | {row['lineage_coherence']:.2f} |\n")
            else:
                f.write(f"| {marker} | {expected} | Not found | - | - | - |\n")
        f.write("\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("### Lineage Groupings\n\n")
        for lineage, celltypes in LINEAGE_GROUPS.items():
            f.write(f"**{lineage.replace('_', ' ').title()}:** {', '.join(celltypes)}\n\n")

        f.write("### Specificity Criteria\n\n")
        f.write(f"- Present in {MIN_CELLTYPES}-{MAX_CELLTYPES} celltypes\n")
        f.write(f"- Lineage coherence ≥ {MIN_LINEAGE_COHERENCE:.0%}\n")
        f.write(f"- Present in ≥ {MIN_CLUSTERS} cluster(s)\n\n")

        f.write("### Lineage Coherence Score\n\n")
        f.write("Calculated as: (number of celltypes in dominant lineage) / (total celltypes)\n\n")
        f.write("A score of 1.0 means all celltypes belong to the same lineage (perfect coherence).\n")

    print(f"✓ Saved: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("CELLTYPE-SPECIFIC TF ANALYSIS")
    print("="*80)
    print()

    # Load data
    peak_info, cluster_tf_gene_matrices, grn_dict = load_all_data()

    # Build TF-celltype matrix
    tf_celltype_data = build_tf_celltype_matrix(peak_info, cluster_tf_gene_matrices, grn_dict)

    # Calculate metrics
    all_tfs_df = calculate_tf_specificity_metrics(tf_celltype_data)

    # Filter for specific TFs
    specific_tfs_df = filter_specific_tfs(all_tfs_df)

    # Generate outputs
    print("\nGenerating outputs...")
    write_csv_summary(
        specific_tfs_df,
        f"{FIG_PATH}/celltype_specific_tfs_summary.csv"
    )
    write_markdown_report(
        specific_tfs_df,
        all_tfs_df,
        f"{FIG_PATH}/CELLTYPE_SPECIFIC_TFS_REPORT.md"
    )

    # Summary statistics
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTotal TFs analyzed: {len(all_tfs_df)}")
    print(f"Celltype-specific TFs found: {len(specific_tfs_df)}")
    print(f"  - Ultra-specific (3 CTs): {len(specific_tfs_df[specific_tfs_df['n_celltypes'] == 3])}")
    print(f"  - Moderately specific (4-5 CTs): {len(specific_tfs_df[specific_tfs_df['n_celltypes'] >= 4])}")
    print(f"\nDistribution by lineage:")
    for lineage, count in specific_tfs_df['dominant_lineage'].value_counts().items():
        print(f"  - {lineage}: {count} TFs")
    print(f"\nGenerated files:")
    print(f"  - celltype_specific_tfs_summary.csv")
    print(f"  - CELLTYPE_SPECIFIC_TFS_REPORT.md")
    print()

if __name__ == "__main__":
    main()
