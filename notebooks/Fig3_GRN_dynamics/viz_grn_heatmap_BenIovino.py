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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()


def get_pairs(tp = None, ct = None):
    """
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


def cluster_counts(df_counts, row_order):
    """
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

timepoint = '10hpf'
celltype = None
drop = None
all_pairs = get_pairs(tp=timepoint, ct=celltype)
df_counts = get_counts(all_pairs, tp_check=timepoint, ct_check=celltype, drop=drop)
df_counts, row_order = cluster_counts(df_counts, row_order=None)
fig = plot_counts(df_counts, tp=timepoint, ct=celltype)

print(df_counts.shape)
fig

# %%
fig.write_image("my_plotly.pdf", width=1000, height=600, scale=50)

# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

timepoints = ['10hpf', '12hpf', '14hpf', '16hpf', '19hpf', '24hpf']
corr_mean, corr_std = [], []
sim_mean, sim_std = [], []

for tp in timepoints:
    timepoint = tp
    celltype = None
    all_pairs = get_pairs(tp=timepoint, ct=celltype)
    df_counts = get_counts(all_pairs, tp_check=timepoint, ct_check=celltype, drop=drop)
    #df_counts, row_order = cluster_counts(df_counts, row_order=None)

    # Calculate pearson correlation coefficient between all columns in df_counts
    corr = df_counts.corr()
    corr_mean.append(corr.values[np.triu_indices_from(corr, k=1)].mean())
    corr_std.append(corr.values[np.triu_indices_from(corr, k=1)].var())

    sim = cosine_similarity(df_counts.T)
    sim_df = pd.DataFrame(sim, index=df_counts.columns, columns=df_counts.columns)
    sim_mean.append(sim_df.values[np.triu_indices_from(sim_df, 1)].mean())
    sim_std.append(sim_df.values[np.triu_indices_from(sim_df, 1)].var())
    

# %%
# Graph correlation and similarity scores
fig = go.Figure()
fig.add_trace(go.Scatter(x=timepoints, y=corr_mean, error_y=dict(type='data', array=corr_std, visible=True), mode='lines', name='Pearson correlation'))
fig.add_trace(go.Scatter(x=timepoints, y=sim_mean, error_y=dict(type='data', array=sim_std, visible=True), mode='lines', name='Cosine similarity'))
fig.update_layout(
    title='GRN similarity over time',
    xaxis_title='Time points',
    yaxis_title='Metric',
    width=500,
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    xaxis=dict(showgrid=False, tickangle=45),     # Hide gridlines
    yaxis=dict(showgrid=False, range=[0, 0.7])      # Hide gridlines
)

# %%
# Convert timepoints to numerical indices for plotting
time_indices = np.arange(len(timepoints))

# Create the plot
plt.figure(figsize=(10, 8))

# Plot Pearson correlation
plt.errorbar(time_indices, corr_mean, yerr=corr_std, label='Pearson correlation', fmt='-o', capsize=3)

# Plot Cosine similarity
plt.errorbar(time_indices, sim_mean, yerr=sim_std, label='Cosine similarity', fmt='-o', capsize=3)

# Add labels, title, and legend
plt.title('GRN similarity over time', fontsize=14)
plt.xlabel('Time points', fontsize=12)
plt.ylabel('Metric', fontsize=12)

# Customize x-axis ticks
plt.xticks(time_indices, timepoints, rotation=45)

# Customize y-axis range
plt.ylim(0, 0.7)

# Remove gridlines
plt.grid(False)

# Transparent background
plt.gca().set_facecolor('none')

# Add legend
plt.legend()

# Save to pdf
plt.savefig('grn_similarity_timepoints.pdf')

# %%
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()


def get_pairs(tp = None, ct = None):
    """
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


def cluster_counts(df_counts):
    """
    """

    hmap = sns.clustermap(df_counts,
                            method='ward',
                            metric='euclidean',
                            standard_scale=None,
                            row_cluster=True,
                            vmax=0.1,
                            vmin=-0.1)
    df_counts = df_counts.iloc[hmap.dendrogram_row.reordered_ind]

    return df_counts


def plot_counts(df_counts, tp, ct):
    """
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


timepoint = None
celltype = 'NMPs'
all_pairs = get_pairs(tp=timepoint, ct=celltype)
df_counts = get_counts(all_pairs, tp_check=timepoint, ct_check=celltype, drop=None)
df_counts = cluster_counts(df_counts)
fig = plot_counts(df_counts, tp=timepoint, ct=celltype)

print(df_counts.shape)
fig


# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

celltypes = ['neural_posterior', 'spinal_cord', 'NMPs', 'tail_bud', 'PSM', 'somites']
corr_mean, corr_std = [], []
sim_mean, sim_std = [], []

for ct in celltypes:
    timepoint = None
    celltype = ct
    all_pairs = get_pairs(tp=timepoint, ct=celltype)
    df_counts = get_counts(all_pairs, tp_check=timepoint, ct_check=celltype, drop=None)
    #df_counts, row_order = cluster_counts(df_counts, row_order=None)

    # Calculate pearson correlation coefficient between all columns in df_counts
    corr = df_counts.corr()
    corr_mean.append(corr.values[np.triu_indices_from(corr, k=1)].mean())
    corr_std.append(corr.values[np.triu_indices_from(corr, k=1)].var())

    sim = cosine_similarity(df_counts.T)
    sim_df = pd.DataFrame(sim, index=df_counts.columns, columns=df_counts.columns)
    sim_mean.append(sim_df.values[np.triu_indices_from(sim_df, 1)].mean())
    sim_std.append(sim_df.values[np.triu_indices_from(sim_df, 1)].var())

# %%
# Graph correlation and similarity scores
fig = go.Figure()
fig.add_trace(go.Scatter(x=celltypes, y=corr_mean, error_y=dict(type='data',array=corr_std, visible=True), mode='markers', name='Pearson correlation'))
fig.add_trace(go.Scatter(x=celltypes, y=sim_mean, error_y=dict(type='data',array=sim_std, visible=True), mode='markers', name='Cosine similarity'))
fig.update_layout(
    title='GRN similarity over cell types',
    xaxis_title='Cell types',
    yaxis_title='Metric',
    width=500,
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    xaxis=dict(showgrid=False),     # Hide gridlines
    yaxis=dict(showgrid=False, range=[0, 0.7])      # Hide gridlines
)

# %%
# Convert timepoints to numerical indices for plotting
time_indices = np.arange(len(celltypes))

# Create the plot
plt.figure(figsize=(10, 8))

# Plot Pearson correlation
plt.errorbar(time_indices, corr_mean, yerr=corr_std, label='Pearson correlation', fmt='o', capsize=3)

# Plot Cosine similarity
plt.errorbar(time_indices, sim_mean, yerr=sim_std, label='Cosine similarity', fmt='o', capsize=3)

# Add labels, title, and legend
plt.title('GRN similarity over cell types', fontsize=14)
plt.xlabel('Time points', fontsize=12)
plt.ylabel('Metric', fontsize=12)

# Customize x-axis ticks
plt.xticks(time_indices, timepoints, rotation=45)

# Customize y-axis range
plt.ylim(0, 0.7)

# Remove gridlines
plt.grid(False)

# Transparent background
plt.gca().set_facecolor('none')

# Add legend
plt.legend()

plt.savefig('grn_similarity_celltypes.pdf')

# %% [markdown] vscode={"languageId": "plaintext"}
# # Network-X Graphs 

# %%
import networkx as nx
# %matplotlib inline

def make_graph(gene_list, tf_list, edges, pairs):
    """
    """

    G = nx.Graph()
    G.add_nodes_from(gene_list)
    G.add_nodes_from(tf_list)
    G.add_weighted_edges_from([(gene_list[i], tf_list[i], edges[i]) for i in range(len(pairs))])

    pos = nx.spring_layout(G, seed=42)
    for node, position in pos.items():
        G.nodes[node]['pos'] = position
    #nx.draw(G)

    return G


def plot_edges(G):
    """|
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
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
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


# %%
celltypes = ['neural_posterior', 'spinal_cord', 'NMPs', 'tail_bud', 'PSM', 'somites']
graphs = []
for cell in celltypes:

    # Filter out df_counts for cell type
    ct_counts = df_counts[cell]
    ct_counts = ct_counts.loc[ct_counts != 0]

    pairs = ct_counts.index.to_list()
    gene_list = [pair.split('_')[0] for pair in pairs]
    tf_list = [pair.split('_')[1] for pair in pairs]
    edges = ct_counts.values.tolist()

    # Make graph and filter out nodes with less than 2 connections
    G = make_graph(gene_list, tf_list, edges, pairs)
    #low_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree < 5]
    #G.remove_nodes_from(low_degree_nodes)
    #weak_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('weight', 1) < 0.05]
    #G.remove_edges_from(weak_edges)


    edge_trace = plot_edges(G)
    node_trace = plot_nodes(G)
    fig = plot_network_graph(G, edge_trace, node_trace)
    fig.update_layout(
        title=f'GRN at {cell}',
    )
    fig.show()
    graphs.append(G)

