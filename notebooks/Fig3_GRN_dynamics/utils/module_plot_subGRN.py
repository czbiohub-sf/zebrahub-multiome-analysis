import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union


def analyze_edge_types(grn_dict, predicted_pairs, celltype_of_interest="NMPs"):
    """
    Diagnostic function to analyze edge types (activation vs repression) in the raw data
    
    Parameters:
    -----------
    grn_dict : dict
        Dictionary mapping (celltype, timepoint) -> GRN DataFrame
    predicted_pairs : list
        List of (TF, target) pairs predicted by peak cluster
    celltype_of_interest : str
        Cell type to analyze
        
    Returns:
    --------
    None (prints analysis results)
    """
    print(f"\n=== Edge Type Analysis for {celltype_of_interest} ===")
    
    for (celltype, timepoint), grn_df in grn_dict.items():
        if celltype == celltype_of_interest:
            # Find intersection with predicted pairs
            grn_pairs = set(zip(grn_df['source'], grn_df['target']))
            found_pairs = set(predicted_pairs) & grn_pairs
            
            # Extract matching edges
            mask = grn_df.apply(lambda row: (row['source'], row['target']) in found_pairs, axis=1)
            subgrn = grn_df[mask].copy()
            
            if len(subgrn) > 0:
                print(f"\nTimepoint {timepoint}:")
                print(f"  Available columns: {list(subgrn.columns)}")
                
                # Check different possible coefficient columns
                for col in ['coef', 'coefficient', 'weight', 'coef_abs']:
                    if col in subgrn.columns:
                        values = subgrn[col]
                        pos_count = sum(values > 0)
                        neg_count = sum(values < 0)
                        zero_count = sum(values == 0)
                        print(f"  Column '{col}': {pos_count} positive, {neg_count} negative, {zero_count} zero")
                        print(f"    Range: {values.min():.4f} to {values.max():.4f}")
                        if col != 'coef_abs':  # Don't show examples for absolute values
                            print(f"    Sample values: {list(values.head())}")


def plot_subgrns_over_time(grn_dict, predicted_pairs, cluster_id="26_8", celltype_of_interest="NMPs", 
                          spring_k=1.2, layout_scale=1.5, max_labels=25, label_strategy="top_connected",
                          debug_labels=False, savefig=False, filename=None, max_edge_width=2.0,
                          node_size_scale=1.0, figsize=None):
    """
    Plot NetworkX diagrams for a celltype-specific subGRNs across all timepoints
    using a master GRN layout for consistent node positioning
    
    Edge colors: Dark Red = Activation, Dark Blue = Repression (based on coefficient sign)
    
    Parameters:
    -----------
    grn_dict : dict
        Dictionary mapping (celltype, timepoint) -> GRN DataFrame
    predicted_pairs : list
        List of (TF, target) pairs predicted by peak cluster
    cluster_id : str
        Identifier for the peak cluster
    celltype_of_interest : str
        Cell type to plot
    spring_k : float
        Spring constant for layout (higher = more spread out)
    layout_scale : float
        Overall scale of the layout (higher = bigger)
    max_labels : int
        Maximum number of labels to show
    label_strategy : str
        Strategy for labeling nodes: "top_connected", "tf_plus_top_targets", 
        "all_tfs_plus_dynamic", "all_tfs", "dynamic_only", "degree_threshold", or "all"
    debug_labels : bool
        Print debugging info for label positioning and edge types
    savefig : bool
        If True, save the figure to file instead of displaying
    filename : str, optional
        Path/filename for saving (e.g., 'grn_temporal.png', 'grn_temporal.pdf')
    max_edge_width : float
        Maximum edge thickness (default: 2.0, min will be 0.3)
    node_size_scale : float
        Scale factor for node sizes (default: 1.0, use 0.6 for smaller figures)
    figsize : tuple, optional
        Tuple (width, height) for figure size, None for auto-sizing
        
    Returns:
    --------
    tuple
        (subgrns, master_G, master_pos) - subGRNs dict, master graph, master positions
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
    
    # Choose layout algorithm based on master network properties - further increased spacing
    if n_master_nodes < 30:
        master_pos = nx.circular_layout(master_G, scale=layout_scale*1.1)
    elif n_master_nodes < 80:
        master_pos = nx.spring_layout(master_G, k=spring_k*1.2, iterations=300, seed=42, scale=layout_scale*1.1)
    else:
        try:
            master_pos = nx.kamada_kawai_layout(master_G, scale=layout_scale*1.1)
        except:
            master_pos = nx.spring_layout(master_G, k=spring_k*1.3, iterations=350, seed=42, scale=layout_scale*1.1)
    
    # Calculate dynamic nodes - both edge changes AND temporal presence patterns
    print("Calculating node dynamics across timepoints...")
    node_edge_changes = {}  # node -> total edge changes across time
    node_presence = {}      # node -> timepoints where node is present
    
    # Track when each node is present
    for node in all_nodes:
        presence_timepoints = []
        total_changes = 0
        prev_edges = set()
        
        for timepoint in timepoints:
            if timepoint in subgrns and len(subgrns[timepoint]) > 0:
                subgrn = subgrns[timepoint]
                # Get current edges for this node (both incoming and outgoing)
                current_edges = set()
                node_edges = subgrn[(subgrn['source'] == node) | (subgrn['target'] == node)]
                
                if len(node_edges) > 0:
                    presence_timepoints.append(timepoint)
                    for _, row in node_edges.iterrows():
                        current_edges.add((row['source'], row['target']))
                
                # Calculate changes from previous timepoint
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
        
        # Transient score: high for nodes present in few timepoints
        transient_score = (n_total - n_present) * 10  # Weight transience highly
        
        # Discontinuous score: high for nodes with gaps in presence
        discontinuous_score = 0
        if n_present > 1:
            # Check for gaps in temporal presence
            tp_indices = [timepoints.index(tp) for tp in presence_tp]
            expected_continuous = list(range(min(tp_indices), max(tp_indices) + 1))
            gaps = len(expected_continuous) - len(tp_indices)
            discontinuous_score = gaps * 15  # Weight discontinuity very highly
        
        # Early/late appearance patterns
        pattern_score = 0
        if n_present > 0:
            first_appearance = timepoints.index(presence_tp[0])
            last_appearance = timepoints.index(presence_tp[-1])
            
            # Early disappearance (appears early, disappears)
            if first_appearance <= 1 and last_appearance < n_total - 2:
                pattern_score += 20
            
            # Late appearance (appears later in development)
            if first_appearance >= 2:
                pattern_score += 15
        
        # Combine scores
        edge_changes = node_edge_changes[node]
        total_dynamics = edge_changes + transient_score + discontinuous_score + pattern_score
        node_temporal_dynamics[node] = total_dynamics
        
    # Get most dynamic nodes (combining edge and presence dynamics)
    dynamic_nodes = sorted(node_temporal_dynamics.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 15 most dynamic nodes (edge + temporal patterns):")
    for i, (node, score) in enumerate(dynamic_nodes[:15]):
        presence_tp = node_presence[node]
        edge_changes = node_edge_changes[node]
        temporal_score = score - edge_changes
        print(f"  {i+1:2d}. {node}: total={score:.0f} (edges={edge_changes}, temporal={temporal_score:.0f}) present in {presence_tp}")
    
    # Identify specific temporal patterns for debugging
    transient_nodes = [(node, tps) for node, tps in node_presence.items() 
                      if len(tps) <= 2 and len(tps) > 0]  # Present in ≤2 timepoints
    early_disappearing = [(node, tps) for node, tps in node_presence.items()
                         if len(tps) > 0 and timepoints.index(tps[-1]) <= 1]  # Last seen in first 2 timepoints
    
    print(f"\nTransient nodes (present ≤2 timepoints): {len(transient_nodes)}")
    for node, tps in transient_nodes[:5]:  # Show top 5
        print(f"  {node}: {tps}")
    
    print(f"\nEarly disappearing nodes: {len(early_disappearing)}")
    for node, tps in early_disappearing[:5]:  # Show top 5
        print(f"  {node}: {tps}")

    
    # Get node classifications for consistent coloring
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets
    
    # Create subplot layout with configurable figure size
    n_timepoints = len(subgrns)
    if figsize is None:
        # Default figure size calculation
        if n_timepoints <= 3:
            nrows, ncols = 1, n_timepoints
            figsize = (4*n_timepoints, 4)
        else:
            nrows, ncols = 2, 3
            figsize = (12, 8)
    else:
        # Use provided figure size and calculate grid
        if n_timepoints <= 3:
            nrows, ncols = 1, n_timepoints
        else:
            nrows, ncols = 2, 3
    
    # Calculate node sizes based on scale factor
    tf_node_size = int(400 * node_size_scale)
    target_node_size = int(250 * node_size_scale)
    tf_target_node_size = int(320 * node_size_scale)
    
    # Inactive node sizes (smaller)
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
    
    # Plot each timepoint using master layout
    for i, (timepoint, subgrn) in enumerate(subgrns.items()):
        ax = axes[i]
        
        if len(subgrn) > 0:
            # Create timepoint-specific graph
            G = nx.DiGraph()
            edge_weights = {}
            edge_signs = {}  # Track positive/negative interactions
            
            # Add edges with weights and signs
            sign_column = None
            for _, row in subgrn.iterrows():
                G.add_edge(row['source'], row['target'])
                
                # Use coef_mean for everything - absolute value for thickness, sign for color
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
                    # Fallback to coef_abs if no signed coefficient available
                    edge_weights[(row['source'], row['target'])] = round(row.get('coef_abs', 0.1), 4)
                    edge_signs[(row['source'], row['target'])] = 1
                    sign_column = 'assumed_positive'
            
            # Print edge type information
            if len(subgrn) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                print(f"Timepoint {timepoint}: {pos_count} activation, {neg_count} repression edges (using '{sign_column}' column)")
            
            # Use master positions, but only for nodes present in this timepoint
            pos = {node: master_pos[node] for node in G.nodes() if node in master_pos}
            
            # Draw all master nodes (both present and absent) for consistency
            present_nodes = set(G.nodes())
            absent_nodes = all_nodes - present_nodes
            
            # Classify nodes for this timepoint
            current_tf = present_nodes & tf_nodes
            current_targets = present_nodes & target_genes  
            current_tf_targets = present_nodes & tf_target_nodes
            
            # Draw present nodes with full opacity
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
            
            # Draw absent nodes with low opacity (ghosted)
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
            
            # Draw present edges with different colors for activation/repression
            if len(G.edges()) > 0:
                # Calculate scaled edge widths (max thickness = 2, min = 0.3)
                all_weights = [edge_weights.get((u, v), 0.1) for u, v in G.edges()]
                max_weight = max(all_weights) if all_weights else 0.1
                min_weight = min(all_weights) if all_weights else 0.1
                
                def scale_width(weight):
                    # Scale weights to range [0.3, max_edge_width]
                    if max_weight == min_weight:
                        return max_edge_width * 0.6  # Use 60% of max if all weights equal
                    normalized = (weight - min_weight) / (max_weight - min_weight)
                    min_width = 0.3
                    return min_width + normalized * (max_edge_width - min_width)
                
                # Separate positive and negative edges
                positive_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) > 0]
                negative_edges = [(u, v) for u, v in G.edges() if edge_signs.get((u, v), 1) < 0]
                
                # Draw positive edges (activation) in dark red
                if positive_edges:
                    pos_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in positive_edges]
                    pos_G = nx.DiGraph()
                    pos_G.add_edges_from(positive_edges)
                    nx.draw_networkx_edges(pos_G, pos, width=pos_widths, 
                                          edge_color='darkred', alpha=0.8,
                                          arrowsize=15, arrowstyle='->', ax=ax)
                
                # Draw negative edges (repression) in dark blue
                if negative_edges:
                    neg_widths = [scale_width(edge_weights.get((u, v), 0.1)) for u, v in negative_edges]
                    neg_G = nx.DiGraph()
                    neg_G.add_edges_from(negative_edges)
                    nx.draw_networkx_edges(neg_G, pos, width=neg_widths, 
                                          edge_color='darkblue', alpha=0.8,
                                          arrowsize=15, arrowstyle='->', ax=ax)
            
            # Draw absent edges with very low opacity
            absent_edges = all_edges - set(G.edges())
            if absent_edges:
                absent_G = nx.DiGraph()
                absent_G.add_edges_from(absent_edges)
                # Only draw absent edges if both nodes exist in master_pos
                valid_absent_edges = [(u, v) for u, v in absent_edges 
                                    if u in master_pos and v in master_pos]
                if valid_absent_edges:
                    absent_G_filtered = nx.DiGraph()
                    absent_G_filtered.add_edges_from(valid_absent_edges)
                    nx.draw_networkx_edges(absent_G_filtered, master_pos, 
                                          width=0.5, edge_color='gray', alpha=0.1,
                                          arrowsize=10, arrowstyle='->', ax=ax, style='dashed')
            
            # Selective labeling - configurable strategies
            node_degrees = dict(G.degree())
            
            if label_strategy == "top_connected":
                # Show labels for top N most connected nodes
                if len(node_degrees) > max_labels:
                    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
                    nodes_to_label = [node for node, degree in sorted_nodes[:max_labels]]
                else:
                    nodes_to_label = list(present_nodes)
                    
            elif label_strategy == "tf_plus_top_targets":
                # Always label TFs and TF-targets, plus top target genes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                target_nodes_present = present_nodes & target_genes
                
                # Always label TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add top connected target genes
                target_degrees = {node: degree for node, degree in node_degrees.items() 
                                if node in target_nodes_present}
                if target_degrees:
                    n_targets_to_add = max(5, max_labels - len(nodes_to_label))
                    top_targets = sorted(target_degrees.items(), key=lambda x: x[1], reverse=True)[:n_targets_to_add]
                    nodes_to_label.extend([node for node, degree in top_targets])
                    
            elif label_strategy == "all_tfs_plus_dynamic":
                # Label ALL TFs and TF-targets, plus most dynamic nodes
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                
                # Always label ALL TFs and TF-targets
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
                # Add most dynamic target genes that aren't already TFs
                target_nodes_present = present_nodes & target_genes
                remaining_slots = max(5, max_labels - len(nodes_to_label))
                
                # Get dynamic target genes (excluding those already labeled as TFs)
                dynamic_targets = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in target_nodes_present and node not in nodes_to_label]
                
                # Add top dynamic target genes
                for node, changes in dynamic_targets[:remaining_slots]:
                    nodes_to_label.append(node)
                    
            elif label_strategy == "all_tfs":
                # Label ALL transcription factors and TF-targets only
                tf_nodes_present = present_nodes & tf_nodes
                tf_target_nodes_present = present_nodes & tf_target_nodes
                nodes_to_label = list(tf_nodes_present | tf_target_nodes_present)
                
            elif label_strategy == "dynamic_only":
                # Label only the most dynamic nodes
                dynamic_present = [(node, changes) for node, changes in dynamic_nodes 
                                 if node in present_nodes]
                nodes_to_label = [node for node, changes in dynamic_present[:max_labels]]
                    
            elif label_strategy == "degree_threshold":
                # Label nodes with degree above threshold
                threshold = max(2, np.percentile(list(node_degrees.values()), 70))  # Top 30%
                nodes_to_label = [node for node, degree in node_degrees.items() if degree >= threshold]
                
            else:  # "all"
                nodes_to_label = list(present_nodes)
            
            # Draw labels only for selected nodes that exist in master_pos
            nodes_to_label_filtered = [node for node in nodes_to_label if node in master_pos]
            label_pos = {node: master_pos[node] for node in nodes_to_label_filtered}
            
            # Create labels dict for only the nodes we want to show
            labels_to_show = {node: node for node in nodes_to_label_filtered}
            nx.draw_networkx_labels(G, label_pos, labels=labels_to_show, font_size=8, font_weight='bold', ax=ax)
            
            print(f"Timepoint {timepoint}: Showing labels for {len(nodes_to_label_filtered)} out of {len(present_nodes)} nodes")
            
            # Count edge types for title
            if len(G.edges()) > 0:
                pos_count = sum(1 for sign in edge_signs.values() if sign > 0)
                neg_count = sum(1 for sign in edge_signs.values() if sign < 0)
                edge_info = f"({pos_count} activation, {neg_count} repression)"
            else:
                edge_info = ""
            
            # Set consistent axis limits based on master layout
            if master_pos:
                x_coords = [coord[0] for coord in master_pos.values()]
                y_coords = [coord[1] for coord in master_pos.values()]
                margin = 0.15
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}\n({len(G.edges())} edges, {len(G.nodes())} nodes) {edge_info}', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No edges found\nfor timepoint {timepoint}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'Celltype {celltype_of_interest} - Timepoint {timepoint}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title and legend
    fig.suptitle(f'{celltype_of_interest.replace("_", " ").title()} Regulatory Network Evolution - Cluster {cluster_id}\n(Master GRN: {len(all_nodes)} nodes, {len(all_edges)} edges)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Enhanced legend with edge types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Transcription Factors (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Target Genes (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=11, label='TF & Target (Active)', alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Inactive Nodes', alpha=0.3),
        plt.Line2D([0], [0], color='darkred', linewidth=3, label='Activation', alpha=0.8),
        plt.Line2D([0], [0], color='darkblue', linewidth=3, label='Repression', alpha=0.8),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Inactive Edges', alpha=0.3)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure if requested
    if savefig:
        if filename is None:
            # Generate default filename
            filename = f"{celltype_of_interest}_grn_temporal_{cluster_id}.png"
        
        # Save with high DPI for publication quality
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved as: {filename}")
    
    plt.show()
    
    return subgrns, master_G, master_pos


def compare_timepoints(grn_dict, predicted_pairs, timepoint1, timepoint2, 
                      cluster_id="26_8", celltype_of_interest="NMPs"):
    """
    Compare two specific timepoints side by side with master layout
    
    Parameters:
    -----------
    grn_dict : dict
        Dictionary mapping (celltype, timepoint) -> GRN DataFrame
    predicted_pairs : list
        List of (TF, target) pairs predicted by peak cluster
    timepoint1 : str
        First timepoint to compare
    timepoint2 : str
        Second timepoint to compare
    cluster_id : str
        Identifier for the peak cluster
    celltype_of_interest : str
        Cell type to compare
        
    Returns:
    --------
    None (displays comparison plot)
    """
    subgrns, master_G, master_pos = plot_subgrns_over_time(
        grn_dict, predicted_pairs, cluster_id, celltype_of_interest)
    
    # Create focused comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot logic for each timepoint would go here...
    # (Similar to above but focused on just two timepoints)
    
    plt.tight_layout()
    plt.show()


# Helper functions for temporal dynamics analysis
def calculate_temporal_dynamics(subgrns, timepoints, all_nodes):
    """
    Calculate temporal dynamics scores for nodes across timepoints
    
    Parameters:
    -----------
    subgrns : dict
        Dictionary mapping timepoint -> subGRN DataFrame
    timepoints : list
        Sorted list of timepoints
    all_nodes : set
        Set of all nodes across timepoints
        
    Returns:
    --------
    tuple
        (node_temporal_dynamics, node_presence, dynamic_nodes)
    """
    node_edge_changes = {}  # node -> total edge changes across time
    node_presence = {}      # node -> timepoints where node is present
    
    # Track when each node is present
    for node in all_nodes:
        presence_timepoints = []
        total_changes = 0
        prev_edges = set()
        
        for timepoint in timepoints:
            if timepoint in subgrns and len(subgrns[timepoint]) > 0:
                subgrn = subgrns[timepoint]
                # Get current edges for this node (both incoming and outgoing)
                current_edges = set()
                node_edges = subgrn[(subgrn['source'] == node) | (subgrn['target'] == node)]
                
                if len(node_edges) > 0:
                    presence_timepoints.append(timepoint)
                    for _, row in node_edges.iterrows():
                        current_edges.add((row['source'], row['target']))
                
                # Calculate changes from previous timepoint
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
        
        # Transient score: high for nodes present in few timepoints
        transient_score = (n_total - n_present) * 10  # Weight transience highly
        
        # Discontinuous score: high for nodes with gaps in presence
        discontinuous_score = 0
        if n_present > 1:
            # Check for gaps in temporal presence
            tp_indices = [timepoints.index(tp) for tp in presence_tp]
            expected_continuous = list(range(min(tp_indices), max(tp_indices) + 1))
            gaps = len(expected_continuous) - len(tp_indices)
            discontinuous_score = gaps * 15  # Weight discontinuity very highly
        
        # Early/late appearance patterns
        pattern_score = 0
        if n_present > 0:
            first_appearance = timepoints.index(presence_tp[0])
            last_appearance = timepoints.index(presence_tp[-1])
            
            # Early disappearance (appears early, disappears)
            if first_appearance <= 1 and last_appearance < n_total - 2:
                pattern_score += 20
            
            # Late appearance (appears later in development)
            if first_appearance >= 2:
                pattern_score += 15
        
        # Combine scores
        edge_changes = node_edge_changes[node]
        total_dynamics = edge_changes + transient_score + discontinuous_score + pattern_score
        node_temporal_dynamics[node] = total_dynamics
    
    # Get most dynamic nodes (combining edge and presence dynamics)
    dynamic_nodes = sorted(node_temporal_dynamics.items(), key=lambda x: x[1], reverse=True)
    
    return node_temporal_dynamics, node_presence, dynamic_nodes


def get_node_classifications(subgrns):
    """
    Classify nodes as TFs, targets, or TF-targets based on network roles
    
    Parameters:
    -----------
    subgrns : dict
        Dictionary mapping timepoint -> subGRN DataFrame
        
    Returns:
    --------
    tuple
        (tf_nodes, target_genes, tf_target_nodes)
    """
    all_sources = set()
    all_targets = set()
    for subgrn in subgrns.values():
        if len(subgrn) > 0:
            all_sources.update(subgrn['source'])
            all_targets.update(subgrn['target'])
    
    tf_nodes = all_sources - all_targets
    target_genes = all_targets - all_sources
    tf_target_nodes = all_sources & all_targets
    
    return tf_nodes, target_genes, tf_target_nodes 