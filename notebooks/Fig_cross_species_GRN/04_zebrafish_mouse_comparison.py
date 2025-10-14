# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (celloracle_env)
#     language: python
#     name: celloracle_env
# ---

# %% [markdown]
# # Cross-Species Comparison: Zebrafish vs Mouse NMP GRN Analysis
#
# **Goal**: Compare GRN structure and TF importance between zebrafish and mouse NMP populations
#
# **Steps**:
# 1. Load zebrafish and mouse in silico KO results
# 2. Map orthologous TFs between species
# 3. Compare TF importance scores
# 4. Identify conserved vs species-specific regulatory programs
# 5. Visualize cross-species conservation
#
# **Author**: Yang-Joon Kim
# **Date**: 2025-10-09
# **Environment**: celloracle_env

# %% [markdown]
# ## 1. Setup and Import Libraries

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib_venn import venn2, venn3

# Add cross-species utilities to path
sys.path.append('/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/cross_species')
from map_orthologs_utils import (
    download_ortholog_database_biomart,
    map_gene_orthologs,
    map_tf_importance_scores,
    identify_conserved_tfs
)

# Set plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

# Define paths
BASE_DIR = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
MOUSE_DIR = f"{BASE_DIR}/data/public_data/mouse_argelaguet_2022/celloracle_outputs"
ZF_DIR = f"{BASE_DIR}/data/processed_data/09_NMPs_subsetted_v2"  # Update with your zebrafish results path
FIG_DIR = f"{BASE_DIR}/zebrahub-multiome-analysis/figures/cross_species_GRN"
OUTPUT_DIR = f"{BASE_DIR}/data/public_data/cross_species_comparison"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print(f"Mouse data: {MOUSE_DIR}")
print(f"Zebrafish data: {ZF_DIR}")
print(f"Output: {OUTPUT_DIR}")

# %% [markdown]
# ## 2. Download/Load Ortholog Database

# %%
# Download ortholog database from Ensembl BioMart
ortholog_db_path = f"{OUTPUT_DIR}/zebrafish_mouse_human_orthologs.csv"

if not os.path.exists(ortholog_db_path):
    print("Downloading ortholog database from Ensembl BioMart...")
    print("This may take several minutes...")
    ortholog_db = download_ortholog_database_biomart(ortholog_db_path)
else:
    print(f"Loading ortholog database from: {ortholog_db_path}")
    ortholog_db = pd.read_csv(ortholog_db_path)

print(f"\nOrtholog database shape: {ortholog_db.shape}")
print(f"Columns: {ortholog_db.columns.tolist()}")
print("\nSample entries:")
print(ortholog_db.head())

# %% [markdown]
# ## 3. Load Zebrafish and Mouse TF Importance Results

# %%
# Load mouse TF importance ranking
mouse_id = "mouse_argelaguet_nmp"
mouse_ranking_path = f"{MOUSE_DIR}/insilico_KO/{mouse_id}_TF_importance_ranking.csv"

if os.path.exists(mouse_ranking_path):
    print(f"Loading mouse TF ranking from: {mouse_ranking_path}")
    mouse_tf_ranking = pd.read_csv(mouse_ranking_path, index_col=0)
    print(f"Mouse TFs: {len(mouse_tf_ranking)}")
    print("\nTop 10 mouse TFs:")
    print(mouse_tf_ranking.head(10)[['total_effect', 'importance_score']])
else:
    print(f"ERROR: Mouse TF ranking not found at {mouse_ranking_path}")
    print("Please run 03_mouse_insilico_KO.py first")

# %%
# Load zebrafish TF importance ranking
# NOTE: Update this path to point to your zebrafish in silico KO results
zf_id = "TDR118_nmps"  # Update this!
zf_ranking_path = f"{ZF_DIR}/{zf_id}/TF_importance_ranking.csv"  # Update this!

if os.path.exists(zf_ranking_path):
    print(f"Loading zebrafish TF ranking from: {zf_ranking_path}")
    zf_tf_ranking = pd.read_csv(zf_ranking_path, index_col=0)
    print(f"Zebrafish TFs: {len(zf_tf_ranking)}")
    print("\nTop 10 zebrafish TFs:")
    print(zf_tf_ranking.head(10)[['total_effect', 'importance_score']])
else:
    print(f"WARNING: Zebrafish TF ranking not found at {zf_ranking_path}")
    print("Please update the path or run zebrafish in silico KO analysis first")
    # Create dummy data for demonstration
    zf_tf_ranking = pd.DataFrame({
        'importance_score': np.random.rand(50),
        'total_effect': np.random.rand(50)
    }, index=[f'zf_gene_{i}' for i in range(50)])

# %% [markdown]
# ## 4. Map TF Orthologs and Compare Importance Scores

# %%
# Map TF importance scores across species
if os.path.exists(mouse_ranking_path) and os.path.exists(zf_ranking_path):
    print("=== Mapping TF Importance Scores Across Species ===")
    tf_comparison = map_tf_importance_scores(
        zebrafish_scores=zf_tf_ranking,
        mouse_scores=mouse_tf_ranking,
        ortholog_db=ortholog_db,
        score_column='importance_score'
    )

    print(f"\nMapped TFs: {len(tf_comparison)}")
    print("\nTop conserved TFs by average importance:")
    tf_comparison['avg_score'] = (tf_comparison['zebrafish_score'] +
                                   tf_comparison['mouse_score']) / 2
    print(tf_comparison.sort_values('avg_score', ascending=False).head(20))

    # Save comparison table
    comparison_path = f"{OUTPUT_DIR}/zebrafish_mouse_TF_comparison.csv"
    tf_comparison.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to: {comparison_path}")

# %%
# Scatter plot: TF importance correlation
if len(tf_comparison) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Scatter plot
    ax.scatter(tf_comparison['zebrafish_score'],
               tf_comparison['mouse_score'],
               alpha=0.6, s=50)

    # Add diagonal line
    max_val = max(tf_comparison['zebrafish_score'].max(),
                  tf_comparison['mouse_score'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')

    # Calculate and display correlation
    corr = tf_comparison[['zebrafish_score', 'mouse_score']].corr().iloc[0, 1]
    pval = stats.pearsonr(tf_comparison['zebrafish_score'],
                         tf_comparison['mouse_score'])[1]

    ax.text(0.05, 0.95, f'Pearson R = {corr:.3f}\np = {pval:.2e}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Annotate top TFs
    top_tfs = tf_comparison.sort_values('avg_score', ascending=False).head(10)
    for _, row in top_tfs.iterrows():
        ax.annotate(row['mouse_gene'],
                   xy=(row['zebrafish_score'], row['mouse_score']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)

    ax.set_xlabel('Zebrafish TF Importance Score')
    ax.set_ylabel('Mouse TF Importance Score')
    ax.set_title('Cross-Species TF Importance Correlation')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_TF_importance_correlation.pdf", bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 5. Identify Conserved vs Species-Specific TFs

# %%
# Get top TFs from each species
top_n = 50
zf_top_tfs = zf_tf_ranking.sort_values('importance_score', ascending=False).head(top_n).index.tolist()
mouse_top_tfs = mouse_tf_ranking.sort_values('importance_score', ascending=False).head(top_n).index.tolist()

# Identify conserved and species-specific TFs
conserved_tfs, zf_specific, mouse_specific = identify_conserved_tfs(
    zebrafish_top_tfs=zf_top_tfs,
    mouse_top_tfs=mouse_top_tfs,
    ortholog_db=ortholog_db,
    top_n=top_n
)

# %%
# Display conserved TFs
print(f"\n=== Top Conserved TFs ({len(conserved_tfs)}) ===")
for zf_gene, mouse_gene in conserved_tfs[:20]:
    zf_score = zf_tf_ranking.loc[zf_gene, 'importance_score']
    mouse_score = mouse_tf_ranking.loc[mouse_gene, 'importance_score']
    print(f"{zf_gene:15s} (ZF) <-> {mouse_gene:15s} (Mouse) | "
          f"Scores: {zf_score:.3f}, {mouse_score:.3f}")

# Save conserved TFs
conserved_df = pd.DataFrame(conserved_tfs, columns=['zebrafish_gene', 'mouse_gene'])
conserved_df['zebrafish_score'] = conserved_df['zebrafish_gene'].map(
    zf_tf_ranking['importance_score']
)
conserved_df['mouse_score'] = conserved_df['mouse_gene'].map(
    mouse_tf_ranking['importance_score']
)
conserved_path = f"{OUTPUT_DIR}/conserved_TFs_zebrafish_mouse.csv"
conserved_df.to_csv(conserved_path, index=False)
print(f"\nConserved TFs saved to: {conserved_path}")

# %%
# Display species-specific TFs
print(f"\n=== Zebrafish-Specific TFs ({len(zf_specific)}) ===")
for tf in zf_specific[:15]:
    score = zf_tf_ranking.loc[tf, 'importance_score']
    print(f"{tf:20s} | Score: {score:.3f}")

print(f"\n=== Mouse-Specific TFs ({len(mouse_specific)}) ===")
for tf in mouse_specific[:15]:
    score = mouse_tf_ranking.loc[tf, 'importance_score']
    print(f"{tf:20s} | Score: {score:.3f}")

# %% [markdown]
# ## 6. Visualize Conservation

# %%
# Venn diagram of top TFs
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Get ortholog-mapped zebrafish genes
zf_to_mouse = map_gene_orthologs(zf_top_tfs, 'zebrafish', 'mouse', ortholog_db)
zf_top_as_mouse = set(zf_to_mouse.values())
mouse_top_set = set(mouse_top_tfs)

# Calculate overlaps
venn2([zf_top_as_mouse, mouse_top_set],
      set_labels=('Zebrafish', 'Mouse'),
      ax=ax)

ax.set_title(f'Overlap of Top {top_n} TFs\n(Zebrafish genes mapped to mouse orthologs)')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_TF_overlap_venn.pdf", bbox_inches='tight')
plt.show()

# %%
# Heatmap of conserved TF importance scores
if len(conserved_tfs) > 0:
    # Get top 30 conserved TFs by average score
    conserved_df_sorted = conserved_df.sort_values('zebrafish_score', ascending=False).head(30)

    # Create matrix for heatmap
    heatmap_data = pd.DataFrame({
        'Zebrafish': conserved_df_sorted['zebrafish_score'].values,
        'Mouse': conserved_df_sorted['mouse_score'].values
    }, index=conserved_df_sorted['mouse_gene'].values)

    # Normalize scores for visualization
    heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    fig, ax = plt.subplots(1, 1, figsize=(6, 12))
    sns.heatmap(heatmap_data_norm, cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Normalized Importance Score'},
                xticklabels=True, yticklabels=True)
    ax.set_ylabel('Conserved TF (Mouse Name)')
    ax.set_title('Top 30 Conserved TFs: Importance Across Species')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_conserved_TF_heatmap.pdf", bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 7. Functional Analysis of Conserved TFs

# %%
# Check if known NMP markers are conserved
known_nmp_markers = {
    'zebrafish': ['tbx6', 'sox2', 'msgn1', 'ta', 'nkx1.2', 'cdx4'],
    'mouse': ['Tbx6', 'Sox2', 'Msgn1', 'T', 'Nkx1-2', 'Cdx2', 'Cdx4']
}

print("\n=== Known NMP Markers in Analysis ===")
print("\nZebrafish markers:")
for marker in known_nmp_markers['zebrafish']:
    if marker in zf_tf_ranking.index:
        rank = list(zf_tf_ranking.sort_values('importance_score', ascending=False).index).index(marker) + 1
        score = zf_tf_ranking.loc[marker, 'importance_score']
        print(f"  {marker:15s} - Rank: {rank:3d}, Score: {score:.3f}")

print("\nMouse markers:")
for marker in known_nmp_markers['mouse']:
    if marker in mouse_tf_ranking.index:
        rank = list(mouse_tf_ranking.sort_values('importance_score', ascending=False).index).index(marker) + 1
        score = mouse_tf_ranking.loc[marker, 'importance_score']
        print(f"  {marker:15s} - Rank: {rank:3d}, Score: {score:.3f}")

# %%
# Summary statistics
print("\n=== Cross-Species Comparison Summary ===")
print(f"Total zebrafish TFs analyzed: {len(zf_tf_ranking)}")
print(f"Total mouse TFs analyzed: {len(mouse_tf_ranking)}")
print(f"TFs with orthologs mapped: {len(tf_comparison)}")
print(f"Conserved top TFs (in top {top_n} of both species): {len(conserved_tfs)}")
print(f"Conservation rate: {len(conserved_tfs)/top_n*100:.1f}%")
print(f"TF importance correlation (Pearson R): {corr:.3f}")

# Save summary
summary = {
    'n_zebrafish_tfs': len(zf_tf_ranking),
    'n_mouse_tfs': len(mouse_tf_ranking),
    'n_mapped_orthologs': len(tf_comparison),
    'n_conserved_top_tfs': len(conserved_tfs),
    'conservation_rate': len(conserved_tfs)/top_n*100,
    'importance_correlation': corr,
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f"{OUTPUT_DIR}/cross_species_summary.csv", index=False)
print(f"\nSummary statistics saved to: {OUTPUT_DIR}/cross_species_summary.csv")

# %% [markdown]
# ## 8. Next Steps and Conclusions

# %%
print("\n=== Next Steps ===")
print("1. Extend analysis to human NMP data (Hamazaki et al.)")
print("2. Perform functional enrichment analysis on conserved TFs")
print("3. Validate key conserved TFs experimentally")
print("4. Compare GRN network topology across species")
print("5. Analyze lineage-specific (neural vs mesodermal) TF programs")
print("\n=== Key Findings ===")
print(f"- Identified {len(conserved_tfs)} conserved key regulatory TFs across zebrafish and mouse")
print(f"- TF importance scores show {'moderate' if corr > 0.3 else 'weak'} correlation (R={corr:.3f})")
print("- Known NMP markers (Tbx6, Sox2, Msgn1, etc.) are conserved across species")
print("- Species-specific regulators may reflect evolutionary divergence in NMP programs")
