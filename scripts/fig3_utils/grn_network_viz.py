"""
GRN Network Visualization Utilities

Functions for visualizing gene regulatory networks as heatmaps and
interactive network graphs using plotly and networkx.
"""

import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import networkx as nx


def get_pairs(tp=None, ct=None):
    """
    Get all TF-gene pairs from GRN data for a specific timepoint or cell type.

    Reads GRN CSV files and extracts unique regulatory pairs across either
    all cell types at a timepoint or all timepoints for a cell type.

    Parameters
    ----------
    tp : str, optional
        Timepoint to analyze (e.g., '10hpf', '12hpf')
        If provided, scans all cell types at this timepoint
    ct : str, optional
        Cell type to analyze (e.g., 'NMPs', 'PSM')
        If provided, scans all timepoints for this cell type

    Returns
    -------
    set of str
        Set of unique TF-gene pairs in format 'target_TF'

    Examples
    --------
    >>> # Get all pairs at 10hpf across all cell types
    >>> pairs_10hpf = get_pairs(tp='10hpf')
    >>>
    >>> # Get all pairs for PSM across all timepoints
    >>> pairs_psm = get_pairs(ct='PSM')

    Notes
    -----
    - Exactly one of tp or ct must be provided
    - Scans predefined cell types: neural_posterior, spinal_cord, NMPs,
      tail_bud, PSM, somites
    - Scans predefined timepoints: 10hpf, 12hpf, 14hpf, 16hpf, 19hpf, 24hpf
    """
    all_pairs = set()
    celltypes = ['neural_posterior', 'spinal_cord', 'NMPs', 'tail_bud', 'PSM', 'somites']
    timepoints = ['10hpf', '12hpf', '14hpf', '16hpf', '19hpf', '24hpf']

    if tp:
        for ct in celltypes:
            df = pd.read_csv(f'src/grn/data/tp/{tp}/{tp}_{ct}.csv', index_col=0)
            df = df.stack().reset_index()
            all_pairs.update(df['target'] + '_' + df['level_1'])
    elif ct:
        for tp in timepoints:
            df = pd.read_csv(f'src/grn/data/ct/{ct}/{ct}_{tp}.csv', index_col=0)
            df = df.stack().reset_index()
            all_pairs.update(df['target'] + '_' + df['level_1'])

    return all_pairs


def get_counts(all_pairs, tp_check, ct_check, drop):
    """
    Build count matrix of TF-gene pairs across conditions.

    Creates a DataFrame with TF-gene pairs as rows and either cell types
    or timepoints as columns, showing regulatory weights.

    Parameters
    ----------
    all_pairs : set of str
        Set of TF-gene pairs to include (from get_pairs())
    tp_check : str or None
        Timepoint to analyze (columns will be cell types)
    ct_check : str or None
        Cell type to analyze (columns will be timepoints)
    drop : int or None
        Minimum number of non-zero conditions required
        Pairs with more zeros than this threshold are dropped

    Returns
    -------
    pd.DataFrame
        Count matrix with pairs as rows, conditions as columns
        Values are regulatory weights (positive/negative)

    Examples
    --------
    >>> # Get counts for 10hpf across cell types
    >>> pairs = get_pairs(tp='10hpf')
    >>> counts = get_counts(pairs, tp_check='10hpf', ct_check=None, drop=2)
    >>>
    >>> # Get counts for PSM across timepoints
    >>> pairs = get_pairs(ct='PSM')
    >>> counts = get_counts(pairs, tp_check=None, ct_check='PSM', drop=None)

    Notes
    -----
    - Exactly one of tp_check or ct_check must be provided
    - Automatically filters out pairs that are zero across all conditions
    - drop parameter helps focus on ubiquitous regulatory interactions
    """
    celltypes = ['neural_posterior', 'spinal_cord', 'NMPs', 'tail_bud', 'PSM', 'somites']
    timepoints = ['10hpf', '12hpf', '14hpf', '16hpf', '19hpf', '24hpf']

    df_counts = pd.DataFrame(index=list(all_pairs))
    if tp_check:
        for ct in celltypes:
            df = pd.read_csv(f'src/grn/data/tp/{tp_check}/{tp_check}_{ct}.csv', index_col=0)
            df = df.stack().reset_index()
            df.index = df['target'] + '_' + df['level_1']
            df.drop(['target', 'level_1'], axis=1, inplace=True)
            df.columns = [ct]
            df_counts = df_counts.join(df, how='left')
    if ct_check:
        for tp in timepoints:
            df = pd.read_csv(f'src/grn/data/ct/{ct_check}/{ct_check}_{tp}.csv', index_col=0)
            df = df.stack().reset_index()
            df.index = df['target'] + '_' + df['level_1']
            df.drop(['target', 'level_1'], axis=1, inplace=True)
            df.columns = [tp]
            df_counts = df_counts.join(df, how='left')

    df_counts = df_counts.loc[df_counts.sum(axis=1) != 0]

    if drop:
        df_counts = df_counts.loc[(df_counts == 0).sum(axis=1) < drop]

    return df_counts


def cluster_counts(df_counts, row_order=None):
    """
    Hierarchically cluster TF-gene pairs by similarity pattern.

    Uses Ward linkage with Euclidean distance to order pairs with
    similar regulatory patterns together.

    Parameters
    ----------
    df_counts : pd.DataFrame
        Count matrix from get_counts()
    row_order : list of int, optional
        Pre-computed row ordering indices
        If None, performs hierarchical clustering

    Returns
    -------
    df_reordered : pd.DataFrame
        Reordered count matrix with similar pairs grouped
    row_order : list of int
        Row indices order (can be saved and reused)

    Examples
    --------
    >>> # Perform clustering
    >>> df_clustered, order = cluster_counts(df_counts, row_order=None)
    >>>
    >>> # Reuse the same order for another dataset
    >>> df_other_clustered, _ = cluster_counts(df_other, row_order=order)

    Notes
    -----
    - Clustering uses seaborn.clustermap with Ward method
    - Color scale fixed at vmin=-0.1, vmax=0.1 for regulatory weights
    """
    hmap = sns.clustermap(df_counts,
                            method='ward',
                            metric='euclidean',
                            standard_scale=None,
                            row_cluster=True,
                            vmax=0.1,
                            vmin=-0.1)
    if row_order:
        df_counts = df_counts.iloc[row_order]
        return df_counts, row_order
    else:
        df_counts = df_counts.iloc[hmap.dendrogram_row.reordered_ind]
        return df_counts, hmap.dendrogram_row.reordered_ind


def plot_counts(df_counts, tp, ct):
    """
    Create interactive heatmap of TF-gene regulatory weights.

    Generates a plotly heatmap visualization showing regulatory
    weights across cell types or timepoints.

    Parameters
    ----------
    df_counts : pd.DataFrame
        Clustered count matrix from cluster_counts()
    tp : str or None
        Timepoint label for title
    ct : str or None
        Cell type label for title

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive heatmap figure

    Examples
    --------
    >>> pairs = get_pairs(tp='10hpf')
    >>> counts = get_counts(pairs, tp_check='10hpf', ct_check=None, drop=2)
    >>> counts_clustered, _ = cluster_counts(counts)
    >>> fig = plot_counts(counts_clustered, tp='10hpf', ct=None)
    >>> fig.show()

    Notes
    -----
    - Color scale: 'balance' diverging colormap
    - Fixed scale: -0.1 to 0.1 for regulatory weights
    - Figure dimensions: 500px width, 1000px height
    """
    if tp:
        title = f'GRN at {tp}'
        xaxis = 'Cell types'
    if ct:
        title = f'GRN at {ct}'
        xaxis = 'Time points'

    # Create heatmap figure
    fig = go.Heatmap(
            z=df_counts.values,
            x=df_counts.columns,
            y=df_counts.index,
            colorscale='balance',
            zmin=-0.1, zmax=0.1
        )
    fig = go.Figure(fig)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title='Gene pairs',
        width=500,
        height=1000
    )

    return fig


def make_graph(gene_list, tf_list, edges, pairs):
    """
    Construct NetworkX graph from TF-gene regulatory interactions.

    Creates a weighted graph with genes and TFs as nodes, regulatory
    relationships as edges, and uses spring layout for positioning.

    Parameters
    ----------
    gene_list : list of str
        Target genes
    tf_list : list of str
        Transcription factors
    edges : list of float
        Regulatory weights (positive or negative)
    pairs : list of str
        Corresponding pair identifiers

    Returns
    -------
    networkx.Graph
        Graph with nodes and edges, positions stored in node attributes

    Examples
    --------
    >>> # Extract data from count matrix
    >>> ct_counts = df_counts['PSM']
    >>> ct_counts = ct_counts.loc[ct_counts != 0]
    >>> pairs = ct_counts.index.to_list()
    >>> gene_list = [p.split('_')[0] for p in pairs]
    >>> tf_list = [p.split('_')[1] for p in pairs]
    >>> edges = ct_counts.values.tolist()
    >>>
    >>> # Create graph
    >>> G = make_graph(gene_list, tf_list, edges, pairs)

    Notes
    -----
    - Uses spring layout with seed=42 for reproducibility
    - Node positions stored in node['pos'] attribute
    - Edge weights stored in edge['weight'] attribute
    """
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    G.add_nodes_from(tf_list)
    G.add_weighted_edges_from([(gene_list[i], tf_list[i], edges[i]) for i in range(len(pairs))])

    pos = nx.spring_layout(G, seed=42)
    for node, position in pos.items():
        G.nodes[node]['pos'] = position

    return G


def plot_edges(G):
    """
    Create plotly trace for network edges.

    Extracts edge coordinates from NetworkX graph and formats
    for plotly visualization.

    Parameters
    ----------
    G : networkx.Graph
        Graph from make_graph() with node positions

    Returns
    -------
    plotly.graph_objects.Scatter
        Scatter trace for drawing edges

    Examples
    --------
    >>> G = make_graph(gene_list, tf_list, edges, pairs)
    >>> edge_trace = plot_edges(G)
    >>> node_trace = plot_nodes(G)
    >>> fig = plot_network_graph(G, edge_trace, node_trace)

    Notes
    -----
    - Edges drawn as thin gray lines (width=0.3, color='#888')
    - Uses None to separate edge segments for plotly rendering
    """
    edge_x = []
    edge_y = []
    edge_weights = []

    # edges contains (u, v) pairs where u (edge[0]) is gene, v (edge[1]) is TF
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']  # gene node position
        x1, y1 = G.nodes[edge[1]]['pos']  # TF node position
        weight = G.edges[edge]['weight']  # edge weight

        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)

        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        edge_weights.append(weight)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.3, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    return edge_trace


def plot_nodes(G):
    """
    Create plotly trace for network nodes.

    Extracts node coordinates from NetworkX graph and formats
    for plotly visualization with color-coded connectivity.

    Parameters
    ----------
    G : networkx.Graph
        Graph from make_graph() with node positions

    Returns
    -------
    plotly.graph_objects.Scatter
        Scatter trace for drawing nodes

    Examples
    --------
    >>> G = make_graph(gene_list, tf_list, edges, pairs)
    >>> edge_trace = plot_edges(G)
    >>> node_trace = plot_nodes(G)
    >>> fig = plot_network_graph(G, edge_trace, node_trace)

    Notes
    -----
    - Nodes colored by degree (number of connections)
    - Colorscale: 'RdBu' (red to blue)
    - Node size: 10
    - Includes colorbar showing connection scale
    """
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                text='Node Connections',
                side='right'
                ),
                xanchor='left',
            ),
            line_width=2))

    return node_trace


def plot_network_graph(G, edge_trace, node_trace):
    """
    Create complete interactive network graph visualization.

    Combines edge and node traces into a complete plotly figure
    with node connectivity coloring and hover information.

    Parameters
    ----------
    G : networkx.Graph
        Graph from make_graph()
    edge_trace : plotly.graph_objects.Scatter
        Edge trace from plot_edges()
    node_trace : plotly.graph_objects.Scatter
        Node trace from plot_nodes()

    Returns
    -------
    plotly.graph_objects.Figure
        Complete interactive network visualization

    Examples
    --------
    >>> # Full workflow
    >>> pairs = get_pairs(tp='10hpf')
    >>> counts = get_counts(pairs, '10hpf', None, drop=2)
    >>>
    >>> # Extract PSM data
    >>> ct_counts = counts['PSM'].loc[counts['PSM'] != 0]
    >>> gene_list = [p.split('_')[0] for p in ct_counts.index]
    >>> tf_list = [p.split('_')[1] for p in ct_counts.index]
    >>> edges = ct_counts.values.tolist()
    >>>
    >>> # Create and plot
    >>> G = make_graph(gene_list, tf_list, edges, ct_counts.index.tolist())
    >>> edge_trace = plot_edges(G)
    >>> node_trace = plot_nodes(G)
    >>> fig = plot_network_graph(G, edge_trace, node_trace)
    >>> fig.show()

    Notes
    -----
    - Nodes colored by degree (number of connections)
    - Hover shows gene/TF name and connection count
    - Layout optimized for regulatory network visualization
    """
    node_adjacencies = []
    node_text = []
    node_names = list(G.nodes())
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{node_names[node]} connections: {str(len(adjacencies[1]))}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(
                    font=dict(
                        size=16
                    )
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=True))
                )

    return fig
