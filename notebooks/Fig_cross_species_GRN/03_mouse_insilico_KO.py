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
# # Mouse NMP In Silico Knock-Out Analysis
#
# **Goal**: Perform systematic in silico knock-out experiments on mouse NMP populations to identify key regulatory TFs
#
# **Steps**:
# 1. Load computed GRNs and expression data
# 2. Define candidate TFs for perturbation
# 3. Simulate gene knock-outs using CellOracle
# 4. Quantify perturbation effects on cell fate decisions
# 5. Rank TFs by importance
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
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import celloracle as co

# Set plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white')
plt.rcParams['figure.figsize'] = (10, 8)

print(f"CellOracle version: {co.__version__}")

# Define paths
BASE_DIR = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
DATA_DIR = f"{BASE_DIR}/data/public_data/mouse_argelaguet_2022"
OUTPUT_DIR = f"{DATA_DIR}/celloracle_outputs"
KO_OUTPUT_DIR = f"{OUTPUT_DIR}/insilico_KO"
FIG_DIR = f"{BASE_DIR}/zebrahub-multiome-analysis/figures/cross_species_GRN"

# Create directories
os.makedirs(KO_OUTPUT_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {KO_OUTPUT_DIR}")
print(f"Figure directory: {FIG_DIR}")

# %% [markdown]
# ## 2. Load GRNs and Expression Data

# %%
# Define data identifiers (update based on your analysis)
data_id = "mouse_argelaguet_nmp"
celltype_col = "celltype"  # Update this!

# Load Links object (contains cell-type-specific GRNs)
links_path = f"{OUTPUT_DIR}/{data_id}.celloracle.links"
print(f"Loading Links object from: {links_path}")

if os.path.exists(links_path):
    links = co.load_hdf5(links_path)
    print("Links object loaded successfully!")
    print(f"Clusters with GRNs: {list(links.links_dict.keys())}")
else:
    print(f"ERROR: Links object not found at {links_path}")
    print("Please run 02_mouse_NMP_GRN_analysis.py first")

# %%
# Load AnnData with expression data
adata_path = f"{DATA_DIR}/mouse_nmp_subset.h5ad"
print(f"\nLoading AnnData from: {adata_path}")

if os.path.exists(adata_path):
    adata = sc.read_h5ad(adata_path)
    print(f"AnnData shape: {adata.shape}")
    print(f"Cell types: {adata.obs[celltype_col].unique()}")
else:
    print(f"ERROR: AnnData not found at {adata_path}")

# %% [markdown]
# ## 3. Define Candidate TFs for KO Simulation

# %%
# Option 1: Use known NMP marker TFs
nmp_marker_tfs = ['Tbx6', 'Sox2', 'Msgn1', 'T', 'Nkx1-2', 'Cdx2', 'Cdx4',
                  'Mesp1', 'Mesp2', 'Neurog2', 'Sox1', 'Pax6']

# Filter for TFs present in the data
available_marker_tfs = [tf for tf in nmp_marker_tfs if tf in adata.var_names]
print(f"=== NMP Marker TFs Available ===")
print(f"Available marker TFs ({len(available_marker_tfs)}): {available_marker_tfs}")

# %%
# Option 2: Use top TFs from GRN analysis
print("\n=== Top TFs from GRN ===")
links.get_network_score()
grn_scores = links.merged_score
top_tfs_grn = grn_scores.sort_values('degree_of_TF', ascending=False).head(30).index.tolist()
print(f"Top 30 TFs by degree: {top_tfs_grn[:10]}...")

# %%
# Combine both lists
ko_genes = list(set(available_marker_tfs + top_tfs_grn[:20]))
print(f"\n=== Final KO Gene List ===")
print(f"Total genes for KO simulation: {len(ko_genes)}")
print(f"Genes: {ko_genes}")

# Save KO gene list
ko_gene_list_path = f"{KO_OUTPUT_DIR}/{data_id}_KO_gene_list.txt"
with open(ko_gene_list_path, 'w') as f:
    f.write('\n'.join(ko_genes))
print(f"\nKO gene list saved to: {ko_gene_list_path}")

# %% [markdown]
# ## 4. Run In Silico KO Simulation
#
# We'll use the script: `scripts/cross_species/insilico_KO_cross_species.py`

# %%
# Construct the command
ko_genes_str = ','.join(ko_genes)
basis = "X_umap"  # Update based on available embeddings

cmd = f"""
python {BASE_DIR}/zebrahub-multiome-analysis/scripts/cross_species/insilico_KO_cross_species.py \\
    --oracle_path {links_path} \\
    --adata_path {adata_path} \\
    --output_path {KO_OUTPUT_DIR} \\
    --data_id {data_id} \\
    --list_KO_genes "{ko_genes_str}" \\
    --annotation {celltype_col} \\
    --basis {basis} \\
    --n_propagation 3 \\
    --n_jobs 4
"""

print("=== Command to run in silico KO analysis ===")
print(cmd)
print("\nNote: This will take time. Run this command in a terminal or submit as a job.")

# Uncomment to run directly (not recommended for large datasets):
# import subprocess
# subprocess.run(cmd, shell=True, check=True)

# %% [markdown]
# ## 5. Analyze Perturbation Results
#
# After running the KO simulation, load and analyze the results.

# %%
# Load perturbation scores
perturbation_score_path = f"{KO_OUTPUT_DIR}/{data_id}_perturbation_scores.csv"

if os.path.exists(perturbation_score_path):
    print(f"Loading perturbation scores from: {perturbation_score_path}")
    perturbation_scores = pd.read_csv(perturbation_score_path, index_col=0)
    print(f"Perturbation scores shape: {perturbation_scores.shape}")
    print("\n=== Top 20 Genes by Total Perturbation Effect ===")
    total_effect = perturbation_scores.abs().sum(axis=1).sort_values(ascending=False)
    print(total_effect.head(20))
else:
    print(f"Perturbation scores not found at: {perturbation_score_path}")
    print("Please run the in silico KO script first.")

# %%
# Visualize perturbation effects as heatmap
if os.path.exists(perturbation_score_path):
    # Top 20 genes
    top_genes = total_effect.head(20).index

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(perturbation_scores.loc[top_genes], cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'label': 'Perturbation Score'},
                xticklabels=True, yticklabels=True)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Knocked Out Gene')
    ax.set_title('Top 20 Gene KO Effects on Mouse NMP Populations')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_mouse_KO_effects_heatmap.pdf", bbox_inches='tight')
    plt.show()

# %%
# Identify cell-type-specific critical TFs
if os.path.exists(perturbation_score_path):
    print("\n=== Cell-Type-Specific Critical TFs ===")
    for celltype in perturbation_scores.columns:
        print(f"\n{celltype}:")
        top_tfs_celltype = perturbation_scores[celltype].abs().sort_values(ascending=False).head(10)
        print(top_tfs_celltype)

# %%
# Compare perturbation effects across cell types
if os.path.exists(perturbation_score_path):
    # Calculate variance of perturbation scores across cell types
    perturbation_variance = perturbation_scores.var(axis=1).sort_values(ascending=False)

    print("\n=== TFs with Cell-Type-Specific Effects (High Variance) ===")
    print(perturbation_variance.head(20))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    perturbation_variance.head(20).plot(kind='barh', ax=ax)
    ax.set_xlabel('Variance of Perturbation Score Across Cell Types')
    ax.set_ylabel('Gene')
    ax.set_title('Cell-Type Specificity of KO Effects')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_mouse_KO_celltype_specificity.pdf", bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 6. Rank TFs by Importance

# %%
if os.path.exists(perturbation_score_path):
    # Create TF ranking based on multiple metrics
    tf_ranking = pd.DataFrame({
        'total_effect': perturbation_scores.abs().sum(axis=1),
        'max_effect': perturbation_scores.abs().max(axis=1),
        'mean_effect': perturbation_scores.abs().mean(axis=1),
        'variance': perturbation_scores.var(axis=1),
    })

    # Add normalized scores
    for col in tf_ranking.columns:
        tf_ranking[f'{col}_norm'] = (tf_ranking[col] - tf_ranking[col].min()) / \
                                     (tf_ranking[col].max() - tf_ranking[col].min())

    # Compute composite importance score
    tf_ranking['importance_score'] = (
        tf_ranking['total_effect_norm'] * 0.4 +
        tf_ranking['max_effect_norm'] * 0.3 +
        tf_ranking['mean_effect_norm'] * 0.2 +
        tf_ranking['variance_norm'] * 0.1
    )

    # Sort by importance
    tf_ranking_sorted = tf_ranking.sort_values('importance_score', ascending=False)

    print("\n=== Top 20 TFs by Importance Score ===")
    print(tf_ranking_sorted[['total_effect', 'max_effect', 'mean_effect',
                              'variance', 'importance_score']].head(20))

    # Save ranking
    ranking_path = f"{KO_OUTPUT_DIR}/{data_id}_TF_importance_ranking.csv"
    tf_ranking_sorted.to_csv(ranking_path)
    print(f"\nTF importance ranking saved to: {ranking_path}")

    # Visualize top TFs
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    tf_ranking_sorted.head(20)['importance_score'].plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Transcription Factor')
    ax.set_title('Top 20 Most Important TFs in Mouse NMP Differentiation')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_mouse_TF_importance_ranking.pdf", bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 7. Summary and Next Steps

# %%
if os.path.exists(perturbation_score_path):
    print("=== Mouse In Silico KO Analysis Summary ===")
    print(f"Total genes simulated: {len(ko_genes)}")
    print(f"Cell types analyzed: {len(perturbation_scores.columns)}")
    print(f"\nTop 5 most important TFs:")
    print(tf_ranking_sorted.head(5)['importance_score'])

    print("\nKey NMP marker TFs in ranking:")
    for tf in available_marker_tfs:
        if tf in tf_ranking_sorted.index:
            rank = list(tf_ranking_sorted.index).index(tf) + 1
            score = tf_ranking_sorted.loc[tf, 'importance_score']
            print(f"  {tf}: Rank {rank}, Score {score:.3f}")

print("\nNext steps:")
print("1. Compare with zebrafish KO results: 04_zebrafish_mouse_comparison.py")
print("2. Validate top TFs experimentally")
print("3. Extend analysis to human NMP data")
