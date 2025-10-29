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
# # Explore Mouse Organogenesis Data (Argelaguet et al., 2022)
#
# **Goal**: Load and explore the pre-processed scRNA-seq data from Argelaguet et al. to:
# 1. Understand the data structure and cell-type annotations
# 2. Identify neuro-mesodermal progenitor (NMP) populations
# 3. Check data quality and available embeddings
# 4. Prepare data for CellOracle GRN analysis
#
# **Data source**:
# - Paper: Argelaguet, Reik, et al., bioRxiv, 2022
# - GEO: GSE205117
# - Pre-processed anndata: `/data/public_data/mouse_argelaguet_2022/anndata.h5ad`
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

# Set plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
plt.rcParams['figure.figsize'] = (6, 4)

# Define paths
BASE_DIR = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
DATA_DIR = f"{BASE_DIR}/data/public_data/mouse_argelaguet_2022"
REPO_DIR = f"{BASE_DIR}/external_repos/mouse_organogenesis_10x_multiome_publication"
FIG_DIR = f"{BASE_DIR}/zebrahub-multiome-analysis/figures/cross_species_GRN"

# Create figure directory
os.makedirs(FIG_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Figure directory: {FIG_DIR}")

# %% [markdown]
# ## 2. Load Data

# %%
# Load the pre-processed anndata object
adata_path = f"{DATA_DIR}/anndata.h5ad"

if not os.path.exists(adata_path):
    print(f"ERROR: Data file not found at {adata_path}")
    print("Please run: bash scripts/cross_species/download_argelaguet_data.sh")
else:
    print(f"Loading data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print("Data loaded successfully!")

# %%
# Display basic information about the dataset
print("=== Dataset Overview ===")
print(f"Number of cells: {adata.n_obs:,}")
print(f"Number of genes: {adata.n_vars:,}")
print(f"\nData shape: {adata.shape}")

# %%
# Check what's in the data
print("\n=== Available Data ===")
print(f"Observations (cells): {list(adata.obs.columns)}")
print(f"\nVariables (genes): {list(adata.var.columns)}")
print(f"\nEmbeddings: {list(adata.obsm.keys())}")
print(f"\nUnsupervised: {list(adata.uns.keys())}")

# %% [markdown]
# ## 3. Explore Cell-Type Annotations

# %%
# Check available annotation columns
annotation_cols = [col for col in adata.obs.columns if any(
    x in col.lower() for x in ['celltype', 'cluster', 'annotation', 'type', 'lineage']
)]

print("=== Potential Cell-Type Annotation Columns ===")
for col in annotation_cols:
    n_categories = adata.obs[col].nunique()
    print(f"\n{col}: {n_categories} categories")
    print(adata.obs[col].value_counts().head(10))

# %%
# Check developmental stage/timepoint information
stage_cols = [col for col in adata.obs.columns if any(
    x in col.lower() for x in ['stage', 'time', 'day', 'embryo']
)]

print("\n=== Developmental Stage/Timepoint Columns ===")
for col in stage_cols:
    n_categories = adata.obs[col].nunique()
    print(f"\n{col}: {n_categories} categories")
    print(adata.obs[col].value_counts())

# %% [markdown]
# ## 4. Visualize Cell-Type Composition

# %%
# Identify the main cell-type annotation column
# (This will need to be adjusted based on actual column names)
celltype_col = 'celltype'  # Update this based on output above

if celltype_col in adata.obs.columns:
    # Plot cell-type distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    celltype_counts = adata.obs[celltype_col].value_counts()
    celltype_counts.plot(kind='barh', ax=ax)
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('Cell type')
    ax.set_title('Cell-Type Distribution')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_celltype_distribution.pdf", bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 5. Identify NMP Populations

# %%
# Search for NMP-related cell types
# Common NMP markers: Tbx6, Sox2, Msgn1, T/Bra, Nkx1-2
nmp_keywords = ['nmp', 'neuromesodermal', 'progenitor', 'neural', 'mesoderm',
                'spinal', 'somite', 'paraxial', 'presomitic']

if celltype_col in adata.obs.columns:
    print("=== Potential NMP-Related Cell Types ===")
    for celltype in adata.obs[celltype_col].unique():
        if any(keyword in str(celltype).lower() for keyword in nmp_keywords):
            n_cells = (adata.obs[celltype_col] == celltype).sum()
            print(f"{celltype}: {n_cells:,} cells")

# %%
# Check expression of key NMP marker genes
nmp_markers = {
    'NMP': ['Tbx6', 'Sox2', 'Msgn1', 'T', 'Nkx1-2'],
    'Neural': ['Sox2', 'Sox1', 'Pax6', 'Neurog2'],
    'Mesoderm': ['Tbx6', 'Msgn1', 'Mesp1', 'Mesp2', 'Pdgfra']
}

print("\n=== NMP Marker Gene Availability ===")
for category, genes in nmp_markers.items():
    available = [g for g in genes if g in adata.var_names]
    print(f"{category}: {available}")

# %% [markdown]
# ## 6. Check Data Quality and Embeddings

# %%
# Check if data is raw counts or normalized
print("=== Data Type Check ===")
print(f"Data type: {adata.X.dtype}")
print(f"Min value: {adata.X.min()}")
print(f"Max value: {adata.X.max()}")
print(f"Mean value: {adata.X.mean():.2f}")

# Check if raw counts are stored separately
if 'raw_counts' in adata.layers:
    print("\nRaw counts available in .layers['raw_counts']")
elif adata.raw is not None:
    print("\nRaw counts available in .raw")
else:
    print("\nWARNING: Raw counts may not be available. Check data carefully!")

# %%
# Check available embeddings
if 'X_umap' in adata.obsm:
    print("UMAP embedding available!")

    # Plot UMAP
    if celltype_col in adata.obs.columns:
        sc.pl.umap(adata, color=celltype_col, legend_loc='on data',
                   legend_fontsize=6, save='_mouse_celltypes.pdf')

# %%
# QC metrics
qc_cols = [col for col in adata.obs.columns if any(
    x in col.lower() for x in ['n_counts', 'n_genes', 'percent_mito', 'qc']
)]

if qc_cols:
    print("\n=== QC Metrics ===")
    print(adata.obs[qc_cols].describe())

# %% [markdown]
# ## 7. Summary and Next Steps

# %%
print("=== Data Exploration Summary ===")
print(f"Total cells: {adata.n_obs:,}")
print(f"Total genes: {adata.n_vars:,}")
print(f"Cell-type annotation column: {celltype_col}")
print(f"Available embeddings: {list(adata.obsm.keys())}")
print("\nNext steps:")
print("1. Subset NMP populations based on cell-type annotations")
print("2. Ensure raw counts are available for CellOracle")
print("3. Proceed to: 02_mouse_NMP_GRN_analysis.py")

# %%
# Save a summary of the data structure
summary = {
    'n_cells': adata.n_obs,
    'n_genes': adata.n_vars,
    'celltype_col': celltype_col,
    'embeddings': list(adata.obsm.keys()),
    'obs_columns': list(adata.obs.columns),
    'var_columns': list(adata.var.columns),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f"{DATA_DIR}/data_summary.csv", index=False)
print(f"\nData summary saved to: {DATA_DIR}/data_summary.csv")
