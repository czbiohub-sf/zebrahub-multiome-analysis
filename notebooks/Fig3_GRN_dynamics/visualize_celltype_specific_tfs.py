#!/usr/bin/env python3
"""
Visualize Celltype-Specific TFs Analysis
=========================================

Generates publication-quality visualizations for the celltype-specific TF analysis.

Plots:
1. TF specificity distribution histogram
2. TF-celltype heatmap (top 50 most specific TFs)
3. Lineage coherence vs. celltype breadth scatter plot
4. Known markers analysis bar plot
5. Celltype co-occurrence network (optional)

Usage:
    python visualize_celltype_specific_tfs.py

Outputs:
    figures/sub_GRNs_reg_programs/celltype_specific_TFs/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_PATH = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
FIG_PATH = f"{BASE_PATH}/figures/sub_GRNs_reg_programs"
OUTPUT_DIR = f"{FIG_PATH}/celltype_specific_TFs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Colorblind-friendly palette
LINEAGE_COLORS = {
    'neuroectoderm': '#0173B2',  # Blue
    'mesoderm': '#DE8F05',        # Orange
    'endoderm': '#029E73',        # Green
    'periderm_ectoderm': '#CC78BC',  # Pink
    'other': '#949494'            # Gray
}

# Specificity thresholds
MIN_CELLTYPES = 3
MAX_CELLTYPES = 5
MIN_COHERENCE = 0.60

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load the celltype-specific TFs analysis results."""
    print("Loading data...")

    # Load full TF metrics
    csv_path = f"{FIG_PATH}/celltype_specific_tfs_summary.csv"

    # First, load all TFs from the analysis to get the full dataset
    # The summary CSV only contains the 11 specific TFs
    # We need to regenerate or load the full metrics

    # For now, load what we have
    df = pd.read_csv(csv_path)
    print(f"   ✓ Loaded {len(df)} celltype-specific TFs")

    return df

def load_full_tf_metrics():
    """
    Load full TF metrics by recomputing from the raw data.
    This gives us all 252 TFs, not just the 11 specific ones.
    """
    print("Loading full TF metrics (all 252 TFs)...")

    # We need to access the intermediate data
    # For now, we'll use the specific TFs and supplement with statistics
    specific_df = pd.read_csv(f"{FIG_PATH}/celltype_specific_tfs_summary.csv")

    # TODO: In production, we'd save the full metrics during the analysis
    # For now, we'll work with what we have

    return specific_df

# ============================================================================
# PLOT 1: TF SPECIFICITY DISTRIBUTION
# ============================================================================

def plot_specificity_distribution(df_specific):
    """
    Plot histogram of TF distribution by number of celltypes.

    Note: This plot needs full data (252 TFs), but we only have the 11 specific ones.
    We'll plot what we have and note the limitation.
    """
    print("\n1. Generating TF specificity distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the 11 specific TFs
    counts = df_specific['n_celltypes'].value_counts().sort_index()

    ax.bar(counts.index, counts.values, color='#0173B2', alpha=0.7, edgecolor='black')

    # Highlight the 3-5 celltype range
    ax.axvspan(MIN_CELLTYPES - 0.5, MAX_CELLTYPES + 0.5,
               alpha=0.2, color='green', label=f'Specific range ({MIN_CELLTYPES}-{MAX_CELLTYPES} CTs)')

    ax.set_xlabel('Number of Celltypes', fontweight='bold')
    ax.set_ylabel('Number of TFs', fontweight='bold')
    ax.set_title('Distribution of Celltype-Specific TFs\n(N=11 TFs meeting specificity criteria)',
                 fontweight='bold', pad=20)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add annotation
    ax.text(0.02, 0.98,
            f'Criteria: {MIN_CELLTYPES}-{MAX_CELLTYPES} celltypes\nLineage coherence ≥ {MIN_COHERENCE:.0%}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/tf_specificity_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 2: TF-CELLTYPE HEATMAP
# ============================================================================

def plot_tf_celltype_heatmap(df_specific):
    """
    Plot heatmap showing TF presence across celltypes.
    """
    print("\n2. Generating TF-celltype heatmap...")

    # Parse celltype lists and create presence matrix
    all_celltypes = set()
    for celltypes_str in df_specific['celltypes_list']:
        celltypes = celltypes_str.split(',')
        all_celltypes.update(celltypes)

    all_celltypes = sorted(all_celltypes)

    # Create binary matrix
    matrix = []
    tf_names = []

    for _, row in df_specific.iterrows():
        tf_names.append(row['tf_name'])
        celltypes = row['celltypes_list'].split(',')
        presence = [1 if ct in celltypes else 0 for ct in all_celltypes]
        matrix.append(presence)

    matrix = np.array(matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot heatmap
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(all_celltypes)))
    ax.set_yticks(np.arange(len(tf_names)))
    ax.set_xticklabels(all_celltypes, rotation=90, ha='right')
    ax.set_yticklabels(tf_names)

    # Add grid
    ax.set_xticks(np.arange(len(all_celltypes)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(tf_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Labels
    ax.set_xlabel('Celltypes', fontweight='bold')
    ax.set_ylabel('Transcription Factors', fontweight='bold')
    ax.set_title('Celltype-Specific TF Presence Matrix (N=11 TFs)',
                 fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Presence', rotation=270, labelpad=15)

    # Add lineage coherence annotations on the right
    for i, row in df_specific.iterrows():
        ax.text(len(all_celltypes) + 0.5, i,
                f"{row['lineage_coherence']:.2f}",
                va='center', ha='left', fontsize=8)

    ax.text(len(all_celltypes) + 0.5, -1, 'Coherence',
            va='center', ha='left', fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/tf_celltype_heatmap_specific11.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 3: LINEAGE COHERENCE VS BREADTH SCATTER
# ============================================================================

def plot_coherence_vs_breadth(df_specific):
    """
    Scatter plot showing relationship between celltype breadth and lineage coherence.
    """
    print("\n3. Generating lineage coherence vs. breadth scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each lineage with different color
    for lineage in df_specific['dominant_lineage'].unique():
        subset = df_specific[df_specific['dominant_lineage'] == lineage]
        ax.scatter(subset['n_celltypes'], subset['lineage_coherence'],
                  s=subset['n_clusters'] * 50,  # Size by number of clusters
                  c=LINEAGE_COLORS.get(lineage, '#949494'),
                  alpha=0.7, edgecolors='black', linewidth=1,
                  label=lineage.replace('_', ' ').title())

    # Add TF labels
    for _, row in df_specific.iterrows():
        ax.annotate(row['tf_name'],
                   (row['n_celltypes'], row['lineage_coherence']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    # Highlight the specificity criteria zone
    from matplotlib.patches import Rectangle
    rect = Rectangle((MIN_CELLTYPES - 0.5, MIN_COHERENCE),
                     MAX_CELLTYPES - MIN_CELLTYPES + 1,
                     1 - MIN_COHERENCE,
                     linewidth=2, edgecolor='green', facecolor='none',
                     linestyle='--', label='Specificity Criteria')
    ax.add_patch(rect)

    # Styling
    ax.set_xlabel('Number of Celltypes', fontweight='bold')
    ax.set_ylabel('Lineage Coherence', fontweight='bold')
    ax.set_title('TF Specificity: Lineage Coherence vs. Celltype Breadth\n' +
                 '(Bubble size = number of clusters)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim([0.55, 1.05])

    # Add annotation
    ax.text(0.02, 0.02,
            f'Specificity criteria:\n• {MIN_CELLTYPES}-{MAX_CELLTYPES} celltypes\n• ≥{MIN_COHERENCE:.0%} lineage coherence',
            transform=ax.transAxes, va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/lineage_coherence_vs_breadth_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 4: KNOWN MARKERS ANALYSIS
# ============================================================================

def plot_known_markers_analysis():
    """
    Bar plot analyzing known lineage markers.
    This requires loading data about known markers from the markdown report.
    """
    print("\n4. Generating known markers analysis plot...")

    # Known markers data (from the markdown report)
    known_markers = {
        'tbx6': {'expected': 'mesoderm\n(PSM)', 'n_celltypes': 0, 'coherence': 0, 'found': False},
        'tbxta': {'expected': 'mesoderm\n(notochord)', 'n_celltypes': 0, 'coherence': 0, 'found': False},
        'nkx2.5': {'expected': 'mesoderm\n(heart)', 'n_celltypes': 20, 'coherence': 0.40, 'found': True},
        'sox2': {'expected': 'neuroectoderm', 'n_celltypes': 21, 'coherence': 0.29, 'found': True},
        'pax6a': {'expected': 'neuroectoderm\n(optic)', 'n_celltypes': 32, 'coherence': 0.34, 'found': True},
        'foxa2': {'expected': 'endoderm', 'n_celltypes': 0, 'coherence': 0, 'found': False},
        'sox17': {'expected': 'endoderm', 'n_celltypes': 24, 'coherence': 0.29, 'found': True},
        'krt4': {'expected': 'periderm/\nectoderm', 'n_celltypes': 0, 'coherence': 0, 'found': False},
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Number of celltypes
    markers = list(known_markers.keys())
    n_celltypes = [known_markers[m]['n_celltypes'] for m in markers]
    colors = ['#DE8F05' if n > 0 else '#949494' for n in n_celltypes]

    bars1 = ax1.bar(markers, n_celltypes, color=colors, alpha=0.7, edgecolor='black')

    # Add threshold line
    ax1.axhline(y=MAX_CELLTYPES, color='green', linestyle='--',
                label=f'Specificity threshold (≤{MAX_CELLTYPES} CTs)', linewidth=2)

    ax1.set_ylabel('Number of Celltypes', fontweight='bold')
    ax1.set_title('Known Lineage Markers: Celltype Breadth', fontweight='bold', pad=15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Annotate bars
    for i, (marker, bar) in enumerate(zip(markers, bars1)):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., 1,
                    'Not found', ha='center', va='bottom', fontsize=8,
                    style='italic', color='red')

    # Plot 2: Lineage coherence
    coherence = [known_markers[m]['coherence'] for m in markers]
    colors2 = ['#0173B2' if c >= MIN_COHERENCE else '#949494' for c in coherence]

    bars2 = ax2.bar(markers, coherence, color=colors2, alpha=0.7, edgecolor='black')

    # Add threshold line
    ax2.axhline(y=MIN_COHERENCE, color='green', linestyle='--',
                label=f'Specificity threshold (≥{MIN_COHERENCE:.0%})', linewidth=2)

    ax2.set_xlabel('Known Lineage Markers', fontweight='bold')
    ax2.set_ylabel('Lineage Coherence', fontweight='bold')
    ax2.set_title('Known Lineage Markers: Lineage Coherence', fontweight='bold', pad=15)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Annotate bars
    for i, (marker, bar) in enumerate(zip(markers, bars2)):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Add expected lineage labels below x-axis
    for i, marker in enumerate(markers):
        expected = known_markers[marker]['expected']
        ax2.text(i, -0.15, expected, ha='center', va='top', fontsize=7,
                style='italic', transform=ax2.get_xaxis_transform())

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/known_markers_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 5: CELLTYPE CO-OCCURRENCE NETWORK
# ============================================================================

def plot_celltype_cooccurrence(df_specific):
    """
    Network plot showing which celltypes tend to share specific TFs.
    """
    print("\n5. Generating celltype co-occurrence network...")

    # Build celltype-celltype co-occurrence matrix
    # Two celltypes co-occur if they share at least one TF

    all_celltypes = set()
    celltype_to_tfs = defaultdict(set)

    for _, row in df_specific.iterrows():
        tf = row['tf_name']
        celltypes = row['celltypes_list'].split(',')
        for ct in celltypes:
            all_celltypes.add(ct)
            celltype_to_tfs[ct].add(tf)

    all_celltypes = sorted(all_celltypes)

    # Create co-occurrence matrix
    n = len(all_celltypes)
    cooccurrence = np.zeros((n, n))

    for i, ct1 in enumerate(all_celltypes):
        for j, ct2 in enumerate(all_celltypes):
            if i != j:
                shared_tfs = celltype_to_tfs[ct1] & celltype_to_tfs[ct2]
                cooccurrence[i, j] = len(shared_tfs)

    # Plot as heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(cooccurrence, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(all_celltypes, rotation=90, ha='right')
    ax.set_yticklabels(all_celltypes)

    # Add values
    for i in range(n):
        for j in range(n):
            if cooccurrence[i, j] > 0:
                text = ax.text(j, i, int(cooccurrence[i, j]),
                             ha="center", va="center", color="black", fontsize=8)

    # Labels
    ax.set_title('Celltype Co-occurrence via Shared Specific TFs\n' +
                 '(Number indicates shared TFs)',
                 fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Shared TFs', rotation=270, labelpad=15)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/celltype_cooccurrence_network.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# SUMMARY STATISTICS PLOT
# ============================================================================

def plot_summary_statistics(df_specific):
    """
    Create a summary figure with key statistics.
    """
    print("\n6. Generating summary statistics plot...")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Distribution by lineage (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    lineage_counts = df_specific['dominant_lineage'].value_counts()
    colors = [LINEAGE_COLORS.get(lin, '#949494') for lin in lineage_counts.index]
    ax1.pie(lineage_counts.values, labels=[l.replace('_', '\n') for l in lineage_counts.index],
            autopct='%1.0f%%', colors=colors, startangle=90)
    ax1.set_title('Distribution by Lineage\n(N=11 TFs)', fontweight='bold')

    # 2. Celltype breadth distribution (bar)
    ax2 = fig.add_subplot(gs[0, 1])
    ct_counts = df_specific['n_celltypes'].value_counts().sort_index()
    ax2.bar(ct_counts.index, ct_counts.values, color='#0173B2', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Celltypes', fontweight='bold')
    ax2.set_ylabel('Number of TFs', fontweight='bold')
    ax2.set_title('Celltype Breadth', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Coherence distribution (histogram)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df_specific['lineage_coherence'], bins=10, color='#029E73',
             alpha=0.7, edgecolor='black')
    ax3.axvline(MIN_COHERENCE, color='red', linestyle='--',
                label=f'Threshold ({MIN_COHERENCE:.0%})', linewidth=2)
    ax3.set_xlabel('Lineage Coherence', fontweight='bold')
    ax3.set_ylabel('Number of TFs', fontweight='bold')
    ax3.set_title('Lineage Coherence Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Top TFs by cluster count (horizontal bar)
    ax4 = fig.add_subplot(gs[1, :])
    top_tfs = df_specific.nlargest(11, 'n_clusters')
    colors_tf = [LINEAGE_COLORS.get(lin, '#949494') for lin in top_tfs['dominant_lineage']]

    y_pos = np.arange(len(top_tfs))
    ax4.barh(y_pos, top_tfs['n_clusters'], color=colors_tf, alpha=0.7, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_tfs['tf_name'])
    ax4.invert_yaxis()
    ax4.set_xlabel('Number of Clusters', fontweight='bold')
    ax4.set_title('Celltype-Specific TFs by Cluster Count', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # Add celltype count annotations
    for i, (idx, row) in enumerate(top_tfs.iterrows()):
        ax4.text(row['n_clusters'] + 0.1, i,
                f"({row['n_celltypes']} CTs)",
                va='center', fontsize=8)

    plt.suptitle('Celltype-Specific TFs: Summary Statistics',
                 fontweight='bold', fontsize=14, y=0.98)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/summary_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("VISUALIZING CELLTYPE-SPECIFIC TFs")
    print("="*80)
    print()

    # Load data
    df_specific = load_data()

    # Generate all plots
    plot_specificity_distribution(df_specific)
    plot_tf_celltype_heatmap(df_specific)
    plot_coherence_vs_breadth(df_specific)
    plot_known_markers_analysis()
    plot_celltype_cooccurrence(df_specific)
    plot_summary_statistics(df_specific)

    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. tf_specificity_distribution.png")
    print("  2. tf_celltype_heatmap_specific11.png")
    print("  3. lineage_coherence_vs_breadth_scatter.png")
    print("  4. known_markers_analysis.png")
    print("  5. celltype_cooccurrence_network.png")
    print("  6. summary_statistics.png")
    print()

if __name__ == "__main__":
    main()
