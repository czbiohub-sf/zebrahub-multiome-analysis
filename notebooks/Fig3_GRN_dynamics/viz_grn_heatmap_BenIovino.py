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

# Import project-specific utilities
from scripts.fig3_utils.grn_network_viz import (
    get_pairs,
    get_counts,
    cluster_counts,
    plot_counts,
    make_graph,
    plot_edges,
    plot_nodes,
    plot_network_graph
)

# %matplotlib inline

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

