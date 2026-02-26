#!/usr/bin/env python
"""
Generate PCA vs CONCORD comparison plots without legends and with better margins.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

# Figure settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['pdf.fonttype'] = 42

# Define paths
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics_v2/"

# Load the data with both embeddings
print("Loading data...")
peaks_pb = sc.read_h5ad(
    "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/"
    "objects_v2/peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad"
)
print(f"Loaded {peaks_pb.shape[0]} peaks")

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

# Create comparison figure without legends
print("Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Row 1: Timepoint coloring
sc.pl.embedding(peaks_pb, basis="X_umap_pca", color="timepoint",
                palette=timepoint_palette, ax=axes[0, 0], show=False,
                title="PCA-based UMAP (timepoint)", legend_loc=None)
sc.pl.embedding(peaks_pb, basis="X_umap_concord", color="timepoint",
                palette=timepoint_palette, ax=axes[0, 1], show=False,
                title="CONCORD-based UMAP (timepoint)", legend_loc=None)

# Row 2: Celltype coloring
sc.pl.embedding(peaks_pb, basis="X_umap_pca", color="celltype",
                palette=cell_type_color_dict, ax=axes[1, 0], show=False,
                title="PCA-based UMAP (celltype)", legend_loc=None)
sc.pl.embedding(peaks_pb, basis="X_umap_concord", color="celltype",
                palette=cell_type_color_dict, ax=axes[1, 1], show=False,
                title="CONCORD-based UMAP (celltype)", legend_loc=None)

# Adjust layout with better margins
plt.tight_layout(pad=2.0)
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, wspace=0.25, hspace=0.25)

# Save
output_path = f"{figpath}/pca_vs_concord_peak_umap_all_peaks_comparison_no_legend.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to: {output_path}")

plt.close()
print("Done!")
