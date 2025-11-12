#!/usr/bin/env python
"""
Generate NetworkX temporal evolution plots for selected developmental clusters
Outputs both PNG (for quick review) and PDF (for manuscript)
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("VISUALIZING SELECTED DEVELOPMENTAL CLUSTERS")
print("="*80)

# Define paths
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
os.makedirs(figpath, exist_ok=True)

# Load ranking results
print("\n1. Loading ranking results...")
ranking_csv = figpath + "cluster_ranking_temporal_dynamics.csv"
df_ranked = pd.read_csv(ranking_csv)
print(f"   ✓ Loaded {len(df_ranked)} ranked clusters")

# Load required data
print("\n2. Loading analysis data...")

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
print(f"   ✓ Loaded GRN dictionary: {len(grn_dict)} combinations")

# Load cluster_tf_gene_matrices
cluster_tf_gene_matrices_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/14_sub_GRN_reg_programs/cluster_tf_gene_matrices.pkl"
with open(cluster_tf_gene_matrices_path, 'rb') as f:
    cluster_tf_gene_matrices = pickle.load(f)
print(f"   ✓ Loaded cluster_tf_gene_matrices: {len(cluster_tf_gene_matrices)} clusters")

# Define the visualization function (extracted from notebook)
print("\n3. Defining visualization function...")

def plot_subgrns_over_time(grn_dict, predicted_pairs, cluster_id="26_8", celltype_of_interest="NMPs",
                          spring_k=1.2, layout_scale=1.5, max_labels=25, label_strategy="top_connected",
                          debug_labels=False, savefig=False, filename=None, max_edge_width=2.0,
                          node_size_scale=1.0, figsize=None):
    """
    Plot NetworkX diagrams for a celltype-specific subGRNs across all timepoints
    using a master GRN layout for consistent node positioning

    Edge colors: Dark Red = Activation, Dark Blue = Repression (based on coefficient sign)
    """
    # Get all timepoints where the celltype exists
    timepoints = []
    for (celltype, timepoint) in grn_dict.keys():
        if celltype == celltype_of_interest:
            timepoints.append(timepoint)

    timepoints = sorted(timepoints)
    print(f"Found {celltype_of_interest} at timepoints: {timepoints}")

    # Extract subGRNs for celltype at each timepoint
    subgrns = {}
    all_nodes = set()
    all_edges = set()

    for timepoint in timepoints:
        if (celltype_of_interest, timepoint) in grn_dict:
            grn_df = grn_dict[(celltype_of_interest, timepoint)]

            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs

            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()

            subgrns[timepoint] = subgrn
            print(f"Timepoint {timepoint}: {len(subgrn)} edges")

            # Collect all nodes and edges for master GRN
            if len(subgrn) > 0:
                all_nodes.update(subgrn['source'])
                all_nodes.update(subgrn['target'])
                all_edges.update(zip(subgrn['source'], subgrn['target']))

    print(f"Master GRN: {len(all_nodes)} total nodes, {len(all_edges)} total edges")

    # Create master GRN and compute layout
    master_G = nx.DiGraph()
    master_G.add_edges_from(all_edges)

    # Compute master layout based on network size
    n_master_nodes = len(master_G.nodes())
    n_master_edges = len(master_G.edges())

    print(f"Computing master layout for {n_master_nodes} nodes, {n_master_edges} edges...")

    # Choose layout algorithm based on master network properties
    if n_master_nodes < 30:
        master_pos = nx.circular_layout(master_G, scale=layout_scale*1.1)
    elif n_master_nodes < 80:
        master_pos = nx.spring_layout(master_G, k=spring_k*1.2, iterations=300, seed=42, scale=layout_scale*1.1)
    else:
        try:
            master_pos = nx.kamada_kawai_layout(master_G, scale=layout_scale*1.1)
        except:
            master_pos = nx.spring_layout(master_G, k=spring_k*1.3, iterations=350, seed=42, scale=layout_scale*1.1)

    # Calculate dynamic nodes
    print("Calculating node dynamics across timepoints...")
    node_edge_changes = {}
    node_presence = {}

    for node in all_nodes:
        presence_timepoints = []
        total_changes = 0
        prev_edges = set()

        for timepoint in timepoints:
            if timepoint in subgrns and len(subgrns[timepoint]) > 0:
                subgrn = subgrns[timepoint]
                current_edges = set()
                node_edges = subgrn[(subgrn['source'] == node) | (subgrn['target'] == node)]

                if len(node_edges) > 0:
                    presence_timepoints.append(timepoint)
                    for _, row in node_edges.iterrows():
                        current_edges.add((row['source'], row['target']))

                if prev_edges is not None:
                    gained_edges = current_edges - prev_edges
                    lost_edges = prev_edges - current_edges
                    total_changes += len(gained_edges) + len(lost_edges)

                prev_edges = current_edges

        node_edge_changes[node] = total_changes
        node_presence[node] = presence_timepoints

    # Calculate temporal dynamics scores
    node_temporal_dynamics = {}
    for node in all_nodes:
        presence_tp = node_presence[node]
        n_present = len(presence_tp)
        n_total = len(timepoints)

        transient_score = (n_total - n_present) * 10

        discontinuous_score = 0
        if n_present > 1:
            tp_indices = [timepoints.index(tp) for tp in presence_tp]
            expected_continuous = list(range(min(tp_indices), max(tp_indices) + 1))
            gaps = len(expected_continuous) - len(tp_indices)
            discontinuous_score = gaps * 15

        pattern_score = 0
        if n_present > 0:
            first_appearance = timepoints.index(presence_tp[0])
            last_appearance = timepoints.index(presence_tp[-1])

            if first_appearance <= 1 and last_appearance < n_total - 2:
                pattern_score += 20

            if first_appearance >= 2:
                pattern_score += 15

        edge_changes = node_edge_changes[node]
        total_dynamics = edge_changes + transient_score + discontinuous_score + pattern_score
        node_temporal_dynamics[node] = total_dynamics

    dynamic_nodes = sorted(node_temporal_dynamics.items(), key=lambda x: x[1], reverse=True)

    print(f"Top 10 most dynamic nodes:")
    for i, (node, score) in enumerate(dynamic_nodes[:10]):
        presence_tp = node_presence[node]
        edge_changes = node_edge_changes[node]
        print(f"  {i+1:2d}. {node}: score={score:.0f} (edges={edge_changes}) in {presence_tp}")

    # Get node classifications
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])

    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets

    # Create subplot layout
    n_timepoints = len(subgrns)
    if figsize is None:
        if n_timepoints <= 3:
            nrows, ncols = 1, n_timepoints
            figsize = (4*n_timepoints, 4)
        else:
            nrows, ncols = 2, 3
            figsize = (12, 8)
    else:
        if n_timepoints <= 3:
            nrows, ncols = 1, n_timepoints
        else:
            nrows, ncols = 2, 3

    # Calculate node sizes
    tf_node_size = int(400 * node_size_scale)
    target_node_size = int(250 * node_size_scale)
    tf_target_node_size = int(320 * node_size_scale)

    inactive_tf_size = int(200 * node_size_scale)
    inactive_target_size = int(120 * node_size_scale)
    inactive_tf_target_size = int(160 * node_size_scale)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_timepoints == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Plot each timepoint
    for i, (timepoint, subgrn) in enumerate(subgrns.items()):
        ax = axes[i]

        if len(subgrn) > 0:
            G = nx.DiGraph()
            edge_weights = {}
            edge_signs = {}

            sign_column = None
            for _, row in subgrn.iterrows():
                G.add_edge(row['source'], row['target'])

                if 'coef_mean' in row and pd.notna(row['coef_mean']):
                    coef_value = row['coef_mean']
                    edge_weights[(row['source'], row['target'])] = round(abs(coef_value), 4)
                    edge_signs[(row['source'], row['target'])] = 1 if coef_value > 0 else -1
                    sign_column = 'coef_mean'
                elif 'coef' in row and pd.notna(row['coef']):
                    coef_value = row['coef']
                    edge_weights[(row['source'], row['target'])] = round(abs(coef_value), 4)
                    edge_signs[(row['source'], row['target'])] = 1 if coef_value > 0 else -1
                    sign_column = 'coef'
                elif 'coefficient' in row and pd.notna(row['coefficient']):
                    coef_value = row['coefficient']
                    edge_weights[(row['source'], row['target'])] = round(abs(coef_value), 4)
                    edge_signs[(row['source'], row['target'])] = 1 if coef_value > 0 else -1
                    sign_column = 'coefficient'
                else:
                    edge_weights[(row['source'], row['target'])] = round(row.get('coef_abs', 0.1), 4)
                    edge_signs[(row['source'], row['target'])] = 1
                    sign_column = 'assumed_positive'

            if len(subgrn) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                print(f"Timepoint {timepoint}: {pos_count} activation, {neg_count} repression edges")

            pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}

            present_nodes = set(G.nodes())
            absent_nodes = all_nodes - present_nodes

            current_tf = present_nodes & tf_nodes
            current_targets = present_nodes & target_genes
            current_tf_targets = present_nodes & tf_target_nodes

            # Draw present nodes
            if current_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf),
                                      node_color='lightcoral', node_size=tf_node_size,
                                      ax=ax, alpha=0.9)
            if current_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_targets),
                                      node_color='lightblue', node_size=target_node_size,
                                      ax=ax, alpha=0.9)
            if current_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(current_tf_targets),
                                      node_color='orange', node_size=tf_target_node_size,
                                      ax=ax, alpha=0.9)

            # Draw absent nodes (ghosted)
            absent_tf = absent_nodes & tf_nodes
            absent_targets = absent_nodes & target_genes
            absent_tf_targets = absent_nodes & tf_target_nodes

            if absent_tf:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf),
                                      node_color='lightcoral', node_size=inactive_tf_size,
                                      ax=ax, alpha=0.15)
            if absent_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_targets),
                                      node_color='lightblue', node_size=inactive_target_size,
                                      ax=ax, alpha=0.15)
            if absent_tf_targets:
                nx.draw_networkx_nodes(master_G, master_pos, nodelist=list(absent_tf_targets),
                                      node_color='orange', node_size=inactive_tf_target_size,
                                      ax=ax, alpha=0.15)

            # Draw edges
            if len(G.edges()) > 0:
                all_weights = [edge_weights.get((u, v), 0.1) for u, v in G.edges()]
                max_weight = max(all_weights) if all_weights else 0.1
                min_weight = min(all_weights) if all_weights else 0.1

                def scale_width(weight):
                    if max_weight == min_weight:
                        return max_edge_width * 0.6
                    normalized = (weight - min_weight) / (max_weight - min_weight)
                    min_width = 0.3
                    return min_width + normalized * (max_edge_width - min_width)

                positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
                negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]

                if positive_edges:
                    pos_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in positive_edges]
                    pos_G = nx.DiGraph()
                    pos_G.add_edges_from(positive_edges)
                    nx.draw_networkx_edges(pos_G, pos, width=pos_widths,
                                          edge_color='darkred', alpha=0.8,
                                          arrowsize=15, arrowstyle='->', ax=ax)

                if negative_edges:
                    neg_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in negative_edges]
                    neg_G = nx.DiGraph()
                    neg_G.add_edges_from(negative_edges)
                    nx.draw_networkx_edges(neg_G, pos, width=neg_widths,
                                          edge_color='darkblue', alpha=0.8,
                                          arrowsize=15, arrowstyle='->', ax=ax)

            # Draw absent edges
            absent_edges = all_edges - set(G.edges())
            if absent_edges:
                valid_absent_edges = [(u, v) for u, v in absent_edges
                                    if u in master_pos and v in master_pos]
                if valid_absent_edges:
                    absent_G_filtered = nx.DiGraph()
                    absent_G_filtered.add_edges_from(valid_absent_edges)
                    nx.draw_networkx_edges(absent_G_filtered, master_pos,
                                          width=0.5, edge_color='gray', alpha=0.1,
                                          arrowsize=10, arrowstyle='->', ax=ax, style='dashed')

            # Label nodes
            node_degrees = dict(G.degree())

            if label_strategy == "all_tfs_plus_dynamic":
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes

                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)

                target_nodes_present = present_nodes & target_genes
                remaining_slots = max(5, max_labels - len(nodes_to_label))

                dynamic_targets = [(node, changes) for node, changes in dynamic_nodes
                                 if node in target_nodes_present and node not in nodes_to_label]

                for node, changes in dynamic_targets[:remaining_slots]:
                    nodes_to_label.append(node)
            else:
                nodes_to_label = list(present_nodes)

            nodes_to_label_filtered = [node for node in nodes_to_label if node in master_pos]
            label_pos = {node: master_pos[node] for node in nodes_to_label_filtered}

            labels_to_show = {node: node for node in nodes_to_label_filtered}
            nx.draw_networkx_labels(G, label_pos, labels=labels_to_show, font_size=8, font_weight='bold', ax=ax)

            print(f"Timepoint {timepoint}: Showing labels for {len(nodes_to_label_filtered)} out of {len(present_nodes)} nodes")

            # Set axis limits
            if master_pos:
                x_coords = [coord[0] for coord in master_pos.values()]
                y_coords = [coord[1] for coord in master_pos.values()]
                margin = 0.15
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

            edge_info = ""
            if len(G.edges()) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                edge_info = f"({pos_count} act, {neg_count} rep)"

            ax.set_title(f'{celltype_of_interest} - {timepoint}\n({len(G.edges())} edges, {len(G.nodes())} nodes) {edge_info}',
                        fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No edges\n{timepoint}',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'{celltype_of_interest} - {timepoint}', fontsize=10, fontweight='bold')

        ax.axis('off')

    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Title and legend
    fig.suptitle(f'{celltype_of_interest.replace("_", " ").title()} GRN Evolution - Cluster {cluster_id}\n({len(all_nodes)} nodes, {len(all_edges)} edges)',
                 fontsize=14, fontweight='bold', y=0.98)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                   markersize=10, label='TFs (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=8, label='Targets (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=9, label='TF&Target (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=6, label='Inactive', alpha=0.3),
        plt.Line2D([0], [0], color='darkred', linewidth=2, label='Activation', alpha=0.8),
        plt.Line2D([0], [0], color='darkblue', linewidth=2, label='Repression', alpha=0.8)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.93), fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Save figure
    if savefig:
        if filename is None:
            filename = f"{celltype_of_interest}_grn_temporal_{cluster_id}.png"

        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
        plt.close()
    else:
        plt.show()

    return subgrns, master_G, master_pos

# Selected clusters for visualization
selected_clusters = [
    {'cluster_id': '21_6', 'celltype': 'tail_bud', 'name': 'Tail Bud (370 edges, 12 TFs)'},
    {'cluster_id': '26_10', 'celltype': 'neural_floor_plate', 'name': 'Neural Floor Plate (267 edges, 6 Sox TFs)'},
    {'cluster_id': '21_4', 'celltype': 'spinal_cord', 'name': 'Spinal Cord (134 edges, 9 TFs)'}
]

print(f"\n4. Generating visualizations for {len(selected_clusters)} selected clusters...")
print("="*80)

for cluster_info in selected_clusters:
    cluster_id = cluster_info['cluster_id']
    celltype = cluster_info['celltype']

    # Get cluster info from ranking
    row = df_ranked[df_ranked['cluster_id'] == cluster_id].iloc[0]
    dynamics_score = row['dynamics_score']
    dev_tfs = eval(row['developmental_tfs_list']) if isinstance(row['developmental_tfs_list'], str) else row['developmental_tfs_list']

    print(f"\n{'='*80}")
    print(f"{cluster_info['name']}")
    print(f"Cluster {cluster_id} in {celltype}")
    print(f"Dynamics score: {dynamics_score:.3f}")
    print(f"Developmental TFs: {', '.join(dev_tfs)}")
    print(f"{'='*80}")

    # Get predicted pairs
    cluster_matrix = cluster_tf_gene_matrices[cluster_id]
    predicted_pairs = []
    for tf in cluster_matrix.index:
        for gene in cluster_matrix.columns:
            if cluster_matrix.loc[tf, gene] == 1:
                predicted_pairs.append((tf, gene))

    # Generate both PNG and PDF versions
    for file_format in ['png', 'pdf']:
        try:
            print(f"\nGenerating {file_format.upper()} for {cluster_id}/{celltype}...")

            subgrns, master_grn, master_pos = plot_subgrns_over_time(
                grn_dict=grn_dict,
                predicted_pairs=predicted_pairs,
                cluster_id=cluster_id,
                celltype_of_interest=celltype,
                spring_k=1.8,
                layout_scale=1.8,
                max_labels=50,
                label_strategy="all_tfs_plus_dynamic",
                debug_labels=False,
                savefig=True,
                filename=f"{figpath}subGRN_selected_{cluster_id}_{celltype}.{file_format}",
                max_edge_width=2.0,
                figsize=(15, 10),
                node_size_scale=0.5
            )
            print(f"   ✓ Saved: subGRN_selected_{cluster_id}_{celltype}.{file_format}")

        except Exception as e:
            print(f"   ✗ Error generating {file_format} for {cluster_id}/{celltype}: {e}")
            import traceback
            traceback.print_exc()
            continue

print(f"\n{'='*80}")
print("VISUALIZATION COMPLETE!")
print(f"{'='*80}")
print(f"Output directory: {figpath}")
print(f"Generated {len(selected_clusters) * 2} files ({len(selected_clusters)} PNG + {len(selected_clusters)} PDF)")
