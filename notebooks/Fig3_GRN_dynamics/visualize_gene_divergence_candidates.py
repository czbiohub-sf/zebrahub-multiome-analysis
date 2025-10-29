#!/usr/bin/env python
"""
Visualize top gene divergence candidates with timepoint in filename
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import networkx as nx

print("="*80)
print("VISUALIZING GENE DIVERGENCE CANDIDATES")
print("="*80)

# Load results
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
divergence_csv = figpath + "lineage_gene_divergence_ranking.csv"
df_divergence = pd.read_csv(divergence_csv)

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

print(f"\nLoaded {len(df_divergence)} clusters with gene divergence analysis")
print(f"Loaded {len(grn_dict)} GRN combinations")
print(f"Loaded {len(cluster_tf_gene_matrices)} cluster matrices")

# Import visualization function from existing script
import sys
sys.path.insert(0, '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig3_GRN_dynamics')
from visualize_lineage_dynamics import visualize_lineage_subgrn

# Visualize top 10 gene divergence candidates
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS FOR TOP 10 GENE DIVERGENCE CANDIDATES...")
print("="*80)

for idx, (i, row) in enumerate(df_divergence.head(10).iterrows(), 1):
    cluster_id = row['cluster_id']
    peak_timepoint = row['peak_timepoint']
    original_celltype = row['original_celltype']
    lineage_bias = row['lineage_bias_score']
    progressive_score = row['combined_progressive_score']
    tf_prog = row['tf_progressive_score']
    target_prog = row['target_progressive_score']

    lineage_type = "neural" if lineage_bias > 0 else "mesoderm"

    print(f"\n{idx}. Generating visualization for cluster {cluster_id} - {original_celltype}")
    print(f"   Peak timepoint: {peak_timepoint} somites")
    print(f"   Lineage bias: {lineage_bias:.3f} ({lineage_type}-biased)")
    print(f"   Progressive score: {progressive_score:.3f} (TF:{tf_prog:.3f}, Target:{target_prog:.3f})")
    print(f"   Neural-specific TFs: {row['n_neural_specific_tfs']}, Mesoderm-specific: {row['n_mesoderm_specific_tfs']}")
    print(f"   Neural-specific targets: {row['n_neural_specific_targets']}, Mesoderm-specific: {row['n_mesoderm_specific_targets']}")

    fig = visualize_lineage_subgrn(cluster_id, peak_timepoint, grn_dict, cluster_tf_gene_matrices)

    if fig is not None:
        # Include timepoint in filename
        # Save PNG
        png_path = figpath + f"subGRN_divergence_{idx}_{cluster_id}_{original_celltype}_tp{peak_timepoint}.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"   Saved PNG: {png_path}")

        # Save PDF
        pdf_path = figpath + f"subGRN_divergence_{idx}_{cluster_id}_{original_celltype}_tp{peak_timepoint}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"   Saved PDF: {pdf_path}")

        plt.close(fig)
    else:
        print(f"   Skipped (insufficient data)")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
