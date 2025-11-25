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
#     display_name: celloracle_env
#     language: python
#     name: celloracle_env
# ---

# %% [markdown]
# # EDA on the dynamics of GRN networks scores
#
# - Last updated: 2/14/2024
# - Author: Yang-Joon Kim
#
# Description/notes:
# - Exploratry data analysis on GRNs from different timepoints.
#
# - Analyses where we'd like to see how the GRN evolves over time/development.
#     - First, for the same cell-type (progenitor, or in intermediate fate), how does the GRN evolves over the developmental stages (real-time).
#     - [Dictys] Second, for the same dev stage, how does the GRN evolves over the developmental trajectories (mesoderm/neuroectoderm trajectories, for example). 
#     - From these analyses, can we learn a transient key driver genes/TFs that were unidentifiable from "static" GRNs?

# %%
# 0. Import

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# %%
import celloracle as co
co.__version__

# %%
# visualization settings
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 600

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/EDA_GRN_network_score_dynamics/"
os.makedirs(figpath, exist_ok=True)

# %% [markdown]
# ## Step 1. Import the GRNs (Links object)

# %%
# # import the GRNs (Links objects)
# TDR118_GRN = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/sequencing_ver1/TDR118_cicero_output/08_TDR118_celltype_GRNs.celloracle.links")
# TDR119_GRN = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/sequencing_ver1/TDR119_cicero_output/08_TDR119_celltype_GRNs.celloracle.links")

# %%
# Import all GRNs from 0-30 somites (6 timepoints)
# Note that we chose TDR118 for 16hpf/15somites stage (there's also TDR119)
GRN_0somites = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR126/08_TDR126_celltype_GRNs.celloracle.links")
GRN_5somites = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR127/08_TDR127_celltype_GRNs.celloracle.links")
GRN_10somites = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR128/08_TDR128_celltype_GRNs.celloracle.links")

GRN_15somites = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR118reseq/08_TDR118reseq_celltype_GRNs.celloracle.links")
GRN_20somites = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR125reseq/08_TDR125reseq_celltype_GRNs.celloracle.links")
GRN_30somites = co.load_hdf5("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/03_celloracle_celltype_GRNs/TDR124reseq/08_TDR124reseq_celltype_GRNs.celloracle.links")

# %%
# GRN_0somites.plot_scores_as_rank(cluster="NMPs", n_gene=30)
# GRN_5somites.plot_scores_as_rank(cluster="NMPs", n_gene=30)
# GRN_10somites.plot_scores_as_rank(cluster="NMPs", n_gene=30)
# GRN_15somites.plot_scores_as_rank(cluster="NMPs", n_gene=30)
# GRN_20somites.plot_scores_as_rank(cluster="NMPs", n_gene=30)
# GRN_30somites.plot_scores_as_rank(cluster="NMPs", n_gene=30)

# %%
# a function to plot the top 20-30 genes for a degree metric for N GRNs
list_GRNS = [GRN_0somites, GRN_5somites, GRN_10somites, GRN_15somites, GRN_20somites, GRN_30somites]

# define a degree metric to use
degree_metric = "degree_centrality_all"

# %%

# %%

# %%

# %% [markdown]
# # 7. Network analysis; Network score for each gene
# The Links class has many functions to visualize network score.
# See the documentation for the details of the functions.
#
# ## 7.1. Network score in each cluster
#

# %% [markdown]
# We have calculated several network scores using different centrality metrics.
# We can use the centrality score to identify key regulatory genes because centrality is one of the important indicators of network structure (https://en.wikipedia.org/wiki/Centrality). 
#
# Let's visualize genes with high network centrality.
#

# %%
oracle.adata.var_names

# %%
# Check cluster name
links.cluster

# %%
# Visualize top n-th genes that have high scores.
links.plot_scores_as_rank(cluster="NMPs", n_gene=30)#, 
                          #save="TDR118_15somite_GRN_coarse_celltypes/ranked_score")

# %% [markdown]
# ## 7.2. Network score comparison between two clusters
#

# %% [markdown]
# By comparing network scores between two clusters, we can analyze differences in GRN structure.

# %%
plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
links.plot_score_comparison_2D(value="eigenvector_centrality",
                               cluster1="NMPs", cluster2="PSM", 
                               percentile=98) #, save="TDR118_15somite_GRN_coarse_celltypes/score_comparison")

# %%

plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
links.plot_score_comparison_2D(value="betweenness_centrality",
                               cluster1="NMPs", cluster2="PSM", 
                               percentile=98) #, save="TDR118_15somite_GRN_coarse_celltypes/score_comparison")

# %%
plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
links.plot_score_comparison_2D(value="degree_centrality_all",
                               cluster1="NMPs", cluster2="PSM", 
                               percentile=98) #, save="TDR118_15somite_GRN_coarse_celltypes/score_comparison")

# %% [markdown]
# ## 7.3. Network score dynamics
# In the following session, we focus on how a gene's network score changes during the differentiation.
#
# Using Gata2, we will demonstrate how you can visualize networks scores for a single gene.
#
# Gata2 is known to play an essential role in the early MEP and GMP populations. .

# %%
# Visualize Gata2 network score dynamics
links.plot_score_per_cluster(goi="tbx16")#, save="TDR118_15somite_GRN_coarse_celltypes/network_score_per_gene/")

# %% [markdown]
# If a gene have no connections in a cluster, it is impossible to calculate network degree scores.
# Thus the scores will not be shown.
# For example, Cebpa have no connection in the erythloids clusters, and there is no degree scores for Cebpa in these clusters as follows.

# %%
links.plot_score_per_cluster(goi="pax6a")

# %% [markdown]
# You can check filtered network edge as follows.

# %%
cluster_name = "NMPs"
filtered_links_df = links.filtered_links[cluster_name]
filtered_links_df.head()

# %%
plt.hist(filtered_links_df.coef_mean, bins=20)

# %%

# %% [markdown]
# ## data exploration - with Alejandro on 7/20/2023

# %%
filtered_links_df.value_counts()

# %%
links.filtered_links.keys()

# %%
celltypes = links.filtered_links.keys()

plt.figure()

for celltype in celltypes:
    filtered_links_df = links.filtered_links[celltype]
    plt.hist(filtered_links_df["coef_mean"], bins=20, density=True, alpha=0.5, label=celltype)
# Draw a vertical line along the y-axis at x=0
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel("network weights")
plt.ylabel("frequency")
plt.legend()
plt.show()


# %%
plt.hist(filtered_links_df["coef_mean"], bins=20)
plt.xlabel("network weight (NMPs)")
plt.ylabel("occurences")

# %%
# scatter plot for coef_mean and -log p_value
plt.scatter(filtered_links_df["coef_mean"], filtered_links_df["-logp"])
plt.xlabel("coef_mean")
plt.ylabel("-log (p_value)")

# %%
filtered_links_df.sort_values("coef_mean", ascending=False)

# %% [markdown]
# You can confirm that there is no Cebpa connection in Ery_0 cluster.

# %%
filtered_links_df[filtered_links_df.source == "sox3"]

# %% [markdown]
# ## 7.4. Gene cartography analysis
#
# Gene cartography is a method for gene network analysis.
# The method classifies gene into several groups using the network module structure and connections.
# It provides us an insight about the role and regulatory mechanism for each gene. 
# For more information on gene cartography, please refer to the following paper (https://www.nature.com/articles/nature03288).
#
# The gene cartography will be calculated for the GRN in each cluster.
# Thus we can know how the gene cartography change by comparing the the score between clusters.

# %%
# Plot cartography as a scatter plot
links.plot_cartography_scatter_per_cluster(scatter=True,
                                           kde=False,
                                           gois=["hmga1a", "sox3", "twist1a"],
                                           auto_gene_annot=False,
                                           args_dot={"n_levels": 105},
                                           args_line={"c":"gray"}) #, save="TDR118_15somite_GRN_coarse_celltypes/cartography")

# %%
output_filepath

# %%
# # Plot the summary of cartography analysis
# links.plot_cartography_term(goi="sox3", save= output_filepath + "figures_danRer11/cartography")

# %%

# %% [markdown]
# # 8. Network analysis; network score distribution
#
# Next, we visualize the distribution of network score to get insight into the global trend of the GRNs.

# %% [markdown]
# ## 8.1. Distribution of network degree

# %%
plt.subplots_adjust(left=0.15, bottom=0.3)
plt.ylim([0,0.040])
links.plot_score_discributions(values=["degree_centrality_all"], method="boxplot")#, save="TDR118_15somite_GRN_coarse_celltypes")



# %%
plt.subplots_adjust(left=0.15, bottom=0.3)
plt.ylim([0, 0.40])
links.plot_score_discributions(values=["eigenvector_centrality"], method="boxplot")# , save="TDR118_15somite_GRN_coarse_celltypes")




# %% [markdown]
# ## 8.2. Distribution of netowrk entropy

# %%
plt.subplots_adjust(left=0.15, bottom=0.3)
links.plot_network_entropy_distributions() #(save="TDR118_15somite_GRN_coarse_celltypes")



# %% [markdown]
# Using the network scores, we could pick up cluster-specific key TFs.
# Gata2, Gata1, Klf1, E2f1, for example, are known to play an essential role in MEP, and these TFs showed high network score in our GRN.
#
# However, it is important to note that network analysis alone cannot shed light on the specific functions or roles these TFs play in cell fate determination. 
#
# In the next section, we will begin to investigate each TF’s contribution to cell fate by running GRN simulations

# %%

# %%
# extract filtered GRNs (filtered_links) as "dictionary"
dict_GRN_16hpf = GRN_16hpf.filtered_links
dict_GRN_16hpf

dict_GRN_19hpf = GRN_19hpf.filtered_links
dict_GRN_19hpf

dict_GRN_24hpf = GRN_24hpf.filtered_links
dict_GRN_24hpf

# %%
dict_GRN_16hpf.keys()

# %%
dict_GRN_19hpf.keys()

# %%
dict_GRN_24hpf.keys()

# %%
# Choose a cell-type of interest (that is present for all timepoints)
# Note that we will have to consider the edge case where the celltype is only transient for specific timepoints
ct = "PSM"

dict_GRN_16hpf[ct]

GRN_16hpf.merged_score[GRN_16hpf.merged_score["cluster"]==ct].sort_values("degree_centrality_all", ascending=False)

# %%
ct = "PSM"

GRN_19hpf.merged_score[GRN_19hpf.merged_score["cluster"]==ct].sort_values("degree_centrality_all", ascending=False)

# %%
ct = "PSM"

GRN_19hpf.merged_score[GRN_19hpf.merged_score["cluster"]==ct].sort_values("degree_centrality_all", ascending=False)

# %%
ct = "PSM"

GRN_24hpf.merged_score[GRN_24hpf.merged_score["cluster"]==ct].sort_values("degree_centrality_all", ascending=False)

# %%
# check the genes that appear/disappear over time within the GRNs
ct = "PSM"

# extract the GRNs for specific cell-types
df1 = dict_GRN_16hpf[ct]
df2 = dict_GRN_19hpf[ct]
df3 = dict_GRN_24hpf[ct]

# Extracting unique genes from each dataframe
genes_16hpf = set(df1['source']).union(set(df1['target']))
genes_19hpf = set(df2['source']).union(set(df2['target']))
genes_24hpf = set(df3['source']).union(set(df3['target']))


# %%
# Finding genes that appear or disappear
genes_appeared = (genes_19hpf.union(genes_24hpf)).difference(genes_16hpf)
genes_disappeared = genes_16hpf.difference(genes_19hpf.union(genes_24hpf))

print("Genes appeared:", genes_appeared)
print("Genes disappeared:", genes_disappeared)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
