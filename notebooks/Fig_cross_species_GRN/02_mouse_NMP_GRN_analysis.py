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
# # Mouse NMP GRN Analysis using CellOracle
#
# **Goal**: Compute cell-type-specific GRNs for mouse NMP populations using CellOracle with default mouse base GRN
#
# **Steps**:
# 1. Load and subset NMP populations from Argelaguet data
# 2. Run CellOracle GRN computation with default mouse base GRN
# 3. Analyze GRN structure and identify key regulatory TFs
# 4. Visualize GRN networks
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
from celloracle.visualizations import visualize_heatmap

# Set plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white')
plt.rcParams['figure.figsize'] = (8, 6)

print(f"CellOracle version: {co.__version__}")

# Define paths
BASE_DIR = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
DATA_DIR = f"{BASE_DIR}/data/public_data/mouse_argelaguet_2022"
OUTPUT_DIR = f"{DATA_DIR}/celloracle_outputs"
FIG_DIR = f"{BASE_DIR}/zebrahub-multiome-analysis/figures/cross_species_GRN"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Figure directory: {FIG_DIR}")

# %% [markdown]
# ## 2. Load and Subset NMP Populations

# %%
# Load the full dataset
adata_path = f"{DATA_DIR}/anndata.h5ad"
print(f"Loading data from: {adata_path}")

adata = sc.read_h5ad(adata_path)
print(f"Original data shape: {adata.shape}")

# %%
# Explore cell-type annotations to identify NMP populations
# NOTE: Update these column names based on output from 01_explore_mouse_data.py
celltype_col = 'celltype'  # Update this!
stage_col = 'stage'  # Update this!

if celltype_col in adata.obs.columns:
    print(f"\nCell types in '{celltype_col}':")
    print(adata.obs[celltype_col].value_counts())

# %%
# Subset NMP-related populations
# Define NMP-related cell types (adjust based on actual annotations)
nmp_celltypes = [
    # Add actual celltype names from the data, e.g.:
    # 'Neuromesodermal progenitors',
    # 'Spinal cord',
    # 'Somites',
    # 'Paraxial mesoderm',
    # etc.
]

print("\n=== Subsetting NMP populations ===")
print(f"Selecting cell types: {nmp_celltypes}")

# Subset the data
adata_nmp = adata[adata.obs[celltype_col].isin(nmp_celltypes)].copy()
print(f"NMP subset shape: {adata_nmp.shape}")
print(f"NMP cell-type distribution:")
print(adata_nmp.obs[celltype_col].value_counts())

# %%
# Save the NMP subset
nmp_subset_path = f"{DATA_DIR}/mouse_nmp_subset.h5ad"
adata_nmp.write(nmp_subset_path)
print(f"NMP subset saved to: {nmp_subset_path}")

# %%
# Visualize NMP subset
if 'X_umap' in adata_nmp.obsm:
    sc.pl.umap(adata_nmp, color=[celltype_col], legend_loc='on data',
               legend_fontsize=8, save='_mouse_nmp_celltypes.pdf')

# %% [markdown]
# ## 3. Run CellOracle GRN Computation
#
# We'll use the script: `scripts/cross_species/compute_GRN_mouse_default_baseGRN.py`

# %%
# Define parameters for GRN computation
data_id = "mouse_argelaguet_nmp"
annotation = celltype_col
dim_reduce = "X_umap"  # Update based on available embeddings
base_grn_type = "mouse_scATAC_atlas"  # or "mouse_promoter"

# Construct the command
cmd = f"""
python {BASE_DIR}/zebrahub-multiome-analysis/scripts/cross_species/compute_GRN_mouse_default_baseGRN.py \\
    --output_path {OUTPUT_DIR} \\
    --adata_path {nmp_subset_path} \\
    --data_id {data_id} \\
    --annotation {annotation} \\
    --dim_reduce {dim_reduce} \\
    --base_grn_type {base_grn_type} \\
    --n_hvg 3000 \\
    --alpha 10 \\
    --n_jobs 8
"""

print("=== Command to run GRN computation ===")
print(cmd)
print("\nNote: This will take 1-2 hours. Run this command in a terminal or submit as a job.")

# Uncomment to run directly (not recommended for large datasets):
# import subprocess
# subprocess.run(cmd, shell=True, check=True)

# %% [markdown]
# ## 4. Load and Analyze Computed GRNs
#
# After running the GRN computation script above, load the results here.

# %%
# Load the Links object (contains cell-type-specific GRNs)
links_path = f"{OUTPUT_DIR}/{data_id}.celloracle.links"

if os.path.exists(links_path):
    print(f"Loading Links object from: {links_path}")
    links = co.load_hdf5(links_path)
    print("Links object loaded successfully!")
else:
    print(f"Links object not found at: {links_path}")
    print("Please run the GRN computation script first.")

# %%
# Display GRN scores
score_path = f"{OUTPUT_DIR}/{data_id}_GRN_scores.csv"

if os.path.exists(score_path):
    grn_scores = pd.read_csv(score_path, index_col=0)
    print("=== GRN Network Scores ===")
    print(grn_scores.head(20))

    # Plot GRN scores
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    grn_scores.sort_values('degree_of_TF', ascending=False).head(30).plot(
        y='degree_of_TF', kind='barh', ax=ax
    )
    ax.set_xlabel('Degree (number of target genes)')
    ax.set_ylabel('Transcription Factor')
    ax.set_title('Top TFs by Network Degree')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_mouse_top_TFs_by_degree.pdf", bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 5. Visualize Cell-Type-Specific GRNs

# %%
# Get available clusters
if os.path.exists(links_path):
    clusters = links.links_dict.keys()
    print(f"Cell types with computed GRNs: {list(clusters)}")

    # Visualize GRN for a specific cluster
    cluster_of_interest = list(clusters)[0]  # Change this to your cluster of interest
    print(f"\nVisualizing GRN for: {cluster_of_interest}")

    # Plot GRN network heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    links.plot_score_per_cluster(
        cluster=cluster_of_interest,
        min_score=0,
        ax=ax
    )
    plt.savefig(f"{FIG_DIR}/02_mouse_GRN_{cluster_of_interest}_heatmap.pdf",
                bbox_inches='tight')
    plt.show()

# %%
# Extract top TFs for each cell type
if os.path.exists(links_path):
    top_tfs_per_cluster = {}

    for cluster in links.links_dict.keys():
        # Get GRN for this cluster
        grn = links.links_dict[cluster]
        # Count degree for each TF
        tf_degree = grn.groupby('source').size().sort_values(ascending=False)
        top_tfs_per_cluster[cluster] = tf_degree.head(20)

    # Save top TFs
    top_tfs_df = pd.DataFrame(top_tfs_per_cluster)
    top_tfs_df.to_csv(f"{OUTPUT_DIR}/{data_id}_top_TFs_per_cluster.csv")
    print(f"Top TFs per cluster saved to: {OUTPUT_DIR}/{data_id}_top_TFs_per_cluster.csv")

    # Visualize as heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    sns.heatmap(top_tfs_df.fillna(0), cmap='viridis', ax=ax, cbar_kws={'label': 'Degree'})
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Transcription Factor')
    ax.set_title('Top TFs per Cell Type (by degree)')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_mouse_top_TFs_heatmap.pdf", bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 6. Identify NMP-Specific Regulatory TFs

# %%
# Check expression of known NMP markers in the GRN
nmp_marker_tfs = ['Tbx6', 'Sox2', 'Msgn1', 'T', 'Nkx1-2', 'Cdx2', 'Cdx4']

if os.path.exists(score_path):
    print("=== NMP Marker TFs in GRN ===")
    nmp_tfs_in_grn = grn_scores[grn_scores.index.isin(nmp_marker_tfs)]
    print(nmp_tfs_in_grn[['degree_of_TF', 'degree_of_target', 'betweenness_centrality']])

# %% [markdown]
# ## 7. Summary and Next Steps

# %%
print("=== Mouse NMP GRN Analysis Summary ===")
print(f"NMP subset cells: {adata_nmp.shape[0]:,}")
print(f"Genes used: {adata_nmp.shape[1]:,}")
if os.path.exists(links_path):
    print(f"Cell types with GRNs: {len(links.links_dict)}")
    print(f"Cell type names: {list(links.links_dict.keys())}")

print("\nNext steps:")
print("1. Perform in silico KO experiments: 03_mouse_insilico_KO.py")
print("2. Compare with zebrafish GRNs: 04_zebrafish_mouse_comparison.py")

# %%
# Save session info
session_info = {
    'data_path': adata_path,
    'nmp_subset_path': nmp_subset_path,
    'n_cells_total': adata.shape[0],
    'n_cells_nmp': adata_nmp.shape[0],
    'n_genes': adata_nmp.shape[1],
    'celltype_col': celltype_col,
    'nmp_celltypes': nmp_celltypes,
    'base_grn_type': base_grn_type,
}

session_df = pd.DataFrame([session_info])
session_df.to_csv(f"{OUTPUT_DIR}/{data_id}_session_info.csv", index=False)
print(f"\nSession info saved to: {OUTPUT_DIR}/{data_id}_session_info.csv")
