#!/usr/bin/env python
"""
Visualize lineage dynamics: create side-by-side visualizations of neural vs mesodermal lineages
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import networkx as nx

print("="*80)
print("VISUALIZING LINEAGE DYNAMICS")
print("="*80)

# Load results
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
print(f"Loaded {len(cluster_tf_gene_matrices)} cluster matrices")

# Color scheme matching previous temporal analysis
# Node colors:
# - lightcoral: TF-only nodes
# - lightblue: Target-only nodes
# - orange: TF & Target nodes (dual role)
# Edge colors:
# - darkred: Activation (positive coef_mean)
# - darkblue: Repression (negative coef_mean)
# - Width: Scaled based on |coef_mean|

def visualize_lineage_subgrn(cluster_id, peak_timepoint, grn_dict, cluster_tf_gene_matrices, max_edge_width=2.0):
    """
    Create side-by-side visualization of neural vs mesodermal lineages with unified coordinates.
    Node colors: lightcoral (TF-only), lightblue (Target-only), orange (TF & Target)
    Edge colors: darkred (activation), darkblue (repression)
    Edge width: Scaled based on |coef_mean|
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
    def extract_lineage_subgrns(lineage, timepoint):
        """Extract subGRNs for all celltypes in lineage"""
        lineage_subgrns = {}

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
            lineage_subgrns[celltype] = subgrn

        return lineage_subgrns

    neural_subgrns = extract_lineage_subgrns(neural_lineage, peak_timepoint)
    mesoderm_subgrns = extract_lineage_subgrns(mesoderm_lineage, peak_timepoint)

    if len(neural_subgrns) < 2 or len(mesoderm_subgrns) < 2:
        return None

    # UNIFIED COORDINATE SYSTEM: Compute master layout using ALL nodes from BOTH lineages
    print(f"  Computing unified coordinate system across all {len(neural_subgrns) + len(mesoderm_subgrns)} subGRNs...")

    all_nodes = set()
    all_edges = set()
    all_sources = set()
    all_targets = set()

    # Collect all nodes and edges from BOTH lineages
    for subgrn in list(neural_subgrns.values()) + list(mesoderm_subgrns.values()):
        all_nodes.update(subgrn['source'])
        all_nodes.update(subgrn['target'])
        all_sources.update(subgrn['source'])
        all_targets.update(subgrn['target'])
        for _, row in subgrn.iterrows():
            all_edges.add((row['source'], row['target']))

    # Classify ALL nodes (TF-only, Target-only, or both)
    tf_only_nodes = all_sources - all_targets
    target_only_nodes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets

    print(f"  Master graph: {len(all_nodes)} nodes ({len(tf_only_nodes)} TF-only, {len(target_only_nodes)} target-only, {len(tf_target_nodes)} dual)")

    # Create master graph for unified layout
    master_G = nx.DiGraph()
    master_G.add_edges_from(all_edges)

    # Compute master layout (consistent across ALL panels)
    master_pos = nx.spring_layout(master_G, k=1.8, scale=1.8, iterations=100, seed=42)

    # Create figure with two rows (neural and mesoderm)
    n_neural = len(neural_subgrns)
    n_mesoderm = len(mesoderm_subgrns)
    n_cols = max(n_neural, n_mesoderm)

    fig = plt.figure(figsize=(5 * n_cols, 10))

    # Plot neural lineage (top row)
    for idx, celltype in enumerate(neural_lineage):
        if celltype not in neural_subgrns:
            continue

        subgrn = neural_subgrns[celltype]
        ax = plt.subplot(2, n_cols, idx + 1)

        # Create celltype-specific graph
        G = nx.DiGraph()
        present_nodes = set(subgrn['source']) | set(subgrn['target'])
        G.add_nodes_from(present_nodes)

        edge_weights = {}
        edge_signs = {}
        for _, row in subgrn.iterrows():
            G.add_edge(row['source'], row['target'])
            edge_weights[(row['source'], row['target'])] = abs(row['coef_mean'])
            edge_signs[(row['source'], row['target'])] = 1 if row['coef_mean'] > 0 else -1

        # Use master positions for consistency
        pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}

        # Classify nodes in this celltype
        current_tf_only = present_nodes & tf_only_nodes
        current_target_only = present_nodes & target_only_nodes
        current_tf_target = present_nodes & tf_target_nodes

        # Draw nodes with consistent colors
        if current_tf_only:
            nx.draw_networkx_nodes(G, pos, nodelist=list(current_tf_only),
                                  node_color='lightcoral', node_size=300, ax=ax, alpha=0.9)
        if current_target_only:
            nx.draw_networkx_nodes(G, pos, nodelist=list(current_target_only),
                                  node_color='lightblue', node_size=200, ax=ax, alpha=0.9)
        if current_tf_target:
            nx.draw_networkx_nodes(G, pos, nodelist=list(current_tf_target),
                                  node_color='orange', node_size=250, ax=ax, alpha=0.9)

        # Draw edges with color based on activation/repression and width based on strength
        if len(G.edges()) > 0:
            # Calculate scaled edge widths
            all_weights = [edge_weights.get((u, v), 0.1) for u, v in G.edges()]
            max_weight = max(all_weights) if all_weights else 0.1
            min_weight = min(all_weights) if all_weights else 0.1

            def scale_width(weight):
                if max_weight == min_weight:
                    return max_edge_width * 0.6
                normalized = (weight - min_weight) / (max_weight - min_weight)
                min_width = 0.3
                return min_width + normalized * (max_edge_width - min_width)

            # Separate positive (activation) and negative (repression) edges
            positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
            negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]

            # Draw activation edges in darkred
            if positive_edges:
                pos_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in positive_edges]
                pos_G = nx.DiGraph()
                pos_G.add_edges_from(positive_edges)
                nx.draw_networkx_edges(pos_G, pos, width=pos_widths,
                                      edge_color='darkred', alpha=0.8,
                                      arrowsize=15, arrowstyle='->', ax=ax)

            # Draw repression edges in darkblue
            if negative_edges:
                neg_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in negative_edges]
                neg_G = nx.DiGraph()
                neg_G.add_edges_from(negative_edges)
                nx.draw_networkx_edges(neg_G, pos, width=neg_widths,
                                      edge_color='darkblue', alpha=0.8,
                                      arrowsize=15, arrowstyle='->', ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title(f"{celltype}\n{len(G.edges())} edges, {len(G.nodes())} nodes",
                    fontsize=10, fontweight='bold')
        ax.axis('off')

    # Plot mesoderm lineage (bottom row)
    for idx, celltype in enumerate(mesoderm_lineage):
        if celltype not in mesoderm_subgrns:
            continue

        subgrn = mesoderm_subgrns[celltype]
        ax = plt.subplot(2, n_cols, n_cols + idx + 1)

        # Create celltype-specific graph
        G = nx.DiGraph()
        present_nodes = set(subgrn['source']) | set(subgrn['target'])
        G.add_nodes_from(present_nodes)

        edge_weights = {}
        edge_signs = {}
        for _, row in subgrn.iterrows():
            G.add_edge(row['source'], row['target'])
            edge_weights[(row['source'], row['target'])] = abs(row['coef_mean'])
            edge_signs[(row['source'], row['target'])] = 1 if row['coef_mean'] > 0 else -1

        # Use master positions for consistency
        pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}

        # Classify nodes in this celltype
        current_tf_only = present_nodes & tf_only_nodes
        current_target_only = present_nodes & target_only_nodes
        current_tf_target = present_nodes & tf_target_nodes

        # Draw nodes with consistent colors
        if current_tf_only:
            nx.draw_networkx_nodes(G, pos, nodelist=list(current_tf_only),
                                  node_color='lightcoral', node_size=300, ax=ax, alpha=0.9)
        if current_target_only:
            nx.draw_networkx_nodes(G, pos, nodelist=list(current_target_only),
                                  node_color='lightblue', node_size=200, ax=ax, alpha=0.9)
        if current_tf_target:
            nx.draw_networkx_nodes(G, pos, nodelist=list(current_tf_target),
                                  node_color='orange', node_size=250, ax=ax, alpha=0.9)

        # Draw edges with color based on activation/repression and width based on strength
        if len(G.edges()) > 0:
            # Calculate scaled edge widths
            all_weights = [edge_weights.get((u, v), 0.1) for u, v in G.edges()]
            max_weight = max(all_weights) if all_weights else 0.1
            min_weight = min(all_weights) if all_weights else 0.1

            def scale_width(weight):
                if max_weight == min_weight:
                    return max_edge_width * 0.6
                normalized = (weight - min_weight) / (max_weight - min_weight)
                min_width = 0.3
                return min_width + normalized * (max_edge_width - min_width)

            # Separate positive (activation) and negative (repression) edges
            positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
            negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]

            # Draw activation edges in darkred
            if positive_edges:
                pos_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in positive_edges]
                pos_G = nx.DiGraph()
                pos_G.add_edges_from(positive_edges)
                nx.draw_networkx_edges(pos_G, pos, width=pos_widths,
                                      edge_color='darkred', alpha=0.8,
                                      arrowsize=15, arrowstyle='->', ax=ax)

            # Draw repression edges in darkblue
            if negative_edges:
                neg_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in negative_edges]
                neg_G = nx.DiGraph()
                neg_G.add_edges_from(negative_edges)
                nx.draw_networkx_edges(neg_G, pos, width=neg_widths,
                                      edge_color='darkblue', alpha=0.8,
                                      arrowsize=15, arrowstyle='->', ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title(f"{celltype}\n{len(G.edges())} edges, {len(G.nodes())} nodes",
                    fontsize=10, fontweight='bold')
        ax.axis('off')

    # Add row labels
    fig.text(0.02, 0.75, 'NEURAL LINEAGE', rotation=90, fontsize=14,
            fontweight='bold', va='center', ha='center')
    fig.text(0.02, 0.25, 'MESODERM LINEAGE', rotation=90, fontsize=14,
            fontweight='bold', va='center', ha='center')

    # Add legend (updated to match temporal dynamics style)
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(color='lightcoral', label='TF only'),
        mpatches.Patch(color='orange', label='TF & Target'),
        mpatches.Patch(color='lightblue', label='Target only'),
        Line2D([0], [0], color='darkred', linewidth=2, label='Activation'),
        Line2D([0], [0], color='darkblue', linewidth=2, label='Repression')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    return fig

# Visualize top 10 candidates
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS FOR TOP 10 CANDIDATES...")
print("="*80)

for idx, (i, row) in enumerate(df_lineage.head(10).iterrows(), 1):
    cluster_id = row['cluster_id']
    peak_timepoint = str(row['peak_timepoint']).zfill(2)  # Convert to zero-padded string
    original_celltype = row['original_celltype']
    lineage_bias = row['lineage_bias_score']
    score = row['lineage_dynamics_score']

    lineage_type = "neural" if lineage_bias > 0 else "mesoderm"

    print(f"\n{idx}. Generating visualization for cluster {cluster_id} - {original_celltype}")
    print(f"   Lineage bias: {lineage_bias:.3f} ({lineage_type}-biased)")
    print(f"   Score: {score:.3f}")
    print(f"   Peak timepoint: {peak_timepoint} somites")

    fig = visualize_lineage_subgrn(cluster_id, peak_timepoint, grn_dict, cluster_tf_gene_matrices)

    if fig is not None:
        # Include timepoint in filename
        # Save PNG
        png_path = figpath + f"subGRN_lineage_{idx}_{cluster_id}_{original_celltype}_tp{peak_timepoint}.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"   Saved PNG: {png_path}")

        # Save PDF
        pdf_path = figpath + f"subGRN_lineage_{idx}_{cluster_id}_{original_celltype}_tp{peak_timepoint}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"   Saved PDF: {pdf_path}")

        plt.close(fig)
    else:
        print(f"   Skipped (insufficient data)")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
