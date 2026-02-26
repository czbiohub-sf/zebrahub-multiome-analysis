# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: sc_rapids
#     language: python
#     name: sc_rapids
# ---

# %% [markdown]
# ## Peak UMAP with CONCORD (All Peaks)
# - Comparing CONCORD vs PCA for peak dimensionality reduction
# - Using all ~640K peaks

# %%
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# CONCORD imports
from concord import Concord
from concord.utils.dim_reduction import run_umap

# rapids-singlecell for PCA comparison
import cupy as cp
import rapids_singlecell as rsc

# %%
# figure parameter setting
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')

# Define figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_v2/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %% [markdown]
# ## 1. Load preprocessed peak data (all peaks)

# %%
# Load the full peak object (all ~640K peaks)
peaks_pb = sc.read_h5ad(
    "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/"
    "objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_all_peaks.h5ad"
)
print(f"Loaded {peaks_pb.shape[0]} peaks x {peaks_pb.shape[1]} pseudobulk groups")

# %%
peaks_pb

# %% [markdown]
# ## 2. Compute PCA-based UMAP (baseline)

# %%
# Make a copy for PCA analysis
peaks_pca = peaks_pb.copy()

# Use log-normalized counts
peaks_pca.X = peaks_pca.layers["log_norm"].copy()
rsc.get.anndata_to_GPU(peaks_pca)

# Standard PCA workflow
rsc.pp.scale(peaks_pca)
rsc.pp.pca(peaks_pca, n_comps=100, use_highly_variable=False)
rsc.pp.neighbors(peaks_pca, n_neighbors=15, n_pcs=40)
rsc.tl.umap(peaks_pca, min_dist=0.3, random_state=42)

# %%
# Store PCA UMAP coordinates in the main object
peaks_pb.obsm["X_umap_pca"] = peaks_pca.obsm["X_umap"].copy()
peaks_pb.obsm["X_pca"] = peaks_pca.obsm["X_pca"].copy()

# %%
# Quick visualization of PCA UMAP
sc.pl.embedding(peaks_pb, basis="X_umap_pca", color="timepoint", title="PCA-based UMAP (timepoint)")

# %% [markdown]
# ## 3. Compute CONCORD-based UMAP

# %%
# Make a copy for CONCORD analysis
peaks_concord = peaks_pb.copy()

# Use log-normalized counts
peaks_concord.X = peaks_concord.layers["log_norm"].copy()

# Scale the data (same preprocessing as PCA)
sc.pp.scale(peaks_concord)

# %%
# Initialize CONCORD (naive mode - no batch correction)
model = Concord(
    adata=peaks_concord,
    domain_key=None,           # No batch correction for pseudobulked data
    latent_dim=100,            # Match PCA n_comps
    encoder_dims=[1000],       # Single hidden layer
    n_epochs=15,               # May need more epochs for larger data
    seed=42,
    device="cuda:0",
    verbose=True,
)

# Train and get embeddings
model.fit_transform(output_key="Concord")

# %%
# Compute UMAP with cosine distance (important for CONCORD)
run_umap(
    peaks_concord,
    source_key="Concord",
    result_key="X_umap",
    n_neighbors=15,
    min_dist=0.3,
    metric="cosine",
    random_state=42,
)

# %%
# Store CONCORD results in the main object
peaks_pb.obsm["X_umap_concord"] = peaks_concord.obsm["X_umap"].copy()
peaks_pb.obsm["X_concord"] = peaks_concord.obsm["Concord"].copy()

# %%
# Quick visualization of CONCORD UMAP
sc.pl.embedding(peaks_pb, basis="X_umap_concord", color="timepoint", title="CONCORD-based UMAP (timepoint)")

# %% [markdown]
# ## 4. Side-by-side comparison visualization

# %%
# Set up color palettes
timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(timepoint_order)))
timepoint_palette = dict(zip(timepoint_order, viridis_colors))

# Cell type color palette
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'differentiating_neurons': '#bebada',
    'endocrine_pancreas': '#fb8072',
    'endoderm': '#80b1d3',
    'enteric_neurons': '#fdb462',
    'epidermis': '#b3de69',
    'fast_muscle': '#df4b9b',
    'floor_plate': '#d9d9d9',
    'hatching_gland': '#bc80bd',
    'heart_myocardium': '#ccebc5',
    'hemangioblasts': '#ffed6f',
    'hematopoietic_vasculature': '#e41a1c',
    'hindbrain': '#377eb8',
    'lateral_plate_mesoderm': '#4daf4a',
    'midbrain_hindbrain_boundary': '#984ea3',
    'muscle': '#ff7f00',
    'neural': '#e6ab02',
    'neural_crest': '#a65628',
    'neural_floor_plate': '#66a61e',
    'neural_optic': '#999999',
    'neural_posterior': '#393b7f',
    'neural_telencephalon': '#fdcdac',
    'neurons': '#cbd5e8',
    'notochord': '#f4cae4',
    'optic_cup': '#c0c000',
    'pharyngeal_arches': '#fff2ae',
    'primordial_germ_cells': '#f1e2cc',
    'pronephros': '#cccccc',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}

# %%
# Comparison figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Row 1: Timepoint coloring
sc.pl.embedding(peaks_pb, basis="X_umap_pca", color="timepoint",
                palette=timepoint_palette, ax=axes[0, 0], show=False,
                title="PCA-based UMAP (timepoint)")
sc.pl.embedding(peaks_pb, basis="X_umap_concord", color="timepoint",
                palette=timepoint_palette, ax=axes[0, 1], show=False,
                title="CONCORD-based UMAP (timepoint)")

# Row 2: Celltype coloring
sc.pl.embedding(peaks_pb, basis="X_umap_pca", color="celltype",
                palette=cell_type_color_dict, ax=axes[1, 0], show=False,
                title="PCA-based UMAP (celltype)")
sc.pl.embedding(peaks_pb, basis="X_umap_concord", color="celltype",
                palette=cell_type_color_dict, ax=axes[1, 1], show=False,
                title="CONCORD-based UMAP (celltype)")

plt.tight_layout()
plt.savefig(f"{figpath}/pca_vs_concord_peak_umap_all_peaks_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Save results

# %%
# Save the object with both embeddings
peaks_pb.write_h5ad(
    "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/"
    "objects_v2/peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad"
)

# %%
print("Done! Saved object with both PCA and CONCORD embeddings.")
print(f"  - X_umap_pca: PCA-based UMAP coordinates")
print(f"  - X_umap_concord: CONCORD-based UMAP coordinates")
print(f"  - X_pca: PCA embeddings (100 components)")
print(f"  - X_concord: CONCORD embeddings (100 dimensions)")
