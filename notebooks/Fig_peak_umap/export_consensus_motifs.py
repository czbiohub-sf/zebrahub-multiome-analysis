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
# ## export consensus sequences for all motifs
#
# - the goal here is to create a master dataframe of the motifs (to input in litemind queries)
# - For each "motif", use the PFM to compute the "consensus" sequence, then add it to the motif dataframe.
#
# - last updated: 7/9/2025

# %%
import pandas as pd
import numpy as np
# gimmemotifs

# %%
from gimmemotifs.motif import Motif, read_motifs

# Load motifs from the default database (all 5298 motifs)
motif_file = "/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm"  # Use the correct database file
motifs = read_motifs(motif_file)

# %%
print(f"the number of motifs is: {len(motifs)}")

# %% [markdown]
# ### Step 1. Check the differentially enriched motifs for "coarse" and "fine" clustering

# %%
# quick check: compare the output motifs from the "leiden_coarse" and "leiden_unified"
# first, coarse: "leiden_coarse"
clust_by_motifs_coarse = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_motifs_maelstrom.csv",
                                     index_col=0)
clust_by_motifs_coarse.head()

# second, fine: ("leiden_unified")
clust_by_motifs_fine = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_unified_cluster_by_motifs.csv",
                                   index_col=0)
clust_by_motifs_fine.head()

# %%
# print the number of motifs that are unique in "coarse" or "fine" clusters from the maelstrom runs
motifs_unique_coarse = set(clust_by_motifs_coarse.columns) - set(clust_by_motifs_fine.columns)
motifs_unique_fine = set(clust_by_motifs_fine.columns) - set(clust_by_motifs_coarse.columns)

print(f"motifs in coarse clusters: {len(clust_by_motifs_coarse.columns)}")
print(f"motifs that are unique in coarse clusters: {len(motifs_unique_coarse)}")

print(f"motifs in fine clusters: {len(clust_by_motifs_fine.columns)}")
print(f"motifs that are unique in fine clusters: {len(motifs_unique_fine)}")

# %% [markdown]
# ### Step 2. import the motif:TF dataframe for both "coarse" and "fine" clustering

# %% [markdown]
# #### 2-1. "coarse" motifs

# %%
# import the motifs post-gimme maelstrom
df_motif_info_coarse = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors.csv", index_col=0)
df_motif_info_coarse

# %%
df_motif_info_coarse.index

# %%
# extract the consensus sequence for each motif,then create a list
list_consensus_seqs = []

for motif_name in df_motif_info_coarse.index:
    selected_motif = next(m for m in motifs if m.id == motif_name)

    # Extract the Position Frequency Matrix (PFM) and Position Weight Matrix (PWM)
    pfm = selected_motif.to_pfm()  # Transpose to make it compatible with logomaker
    pwm = selected_motif.to_ppm()

    # consensus sequence
    # print(motif_name)
    # print(selected_motif.to_consensus())
    # print(pwm)
    list_consensus_seqs.append(selected_motif.to_consensus())
    

# add the consensus sequence as additional column
df_motif_info_coarse["consensus"] = list_consensus_seqs

df_motif_info_coarse.head()

# %%
# export the revised dataframe (motif for coarse)
df_motif_info_coarse.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors_consensus.csv")

# %% [markdown]
# #### 2-2. "fine" motifs
#
# - We will have to import the MaelstromResult first, then export the dataframe of motifs:factors for the "fine" clusters (differentially enriched motifs among the fine clusters)

# %%
# import the motif:factors dataframe (from fine clusters)
df_motif_info_fine = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_unified_cisBP_ver2_Danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors.csv",
                                 index_col=0)
df_motif_info_fine.head()

# %%
# extract the consensus sequence for each motif,then create a list
list_consensus_seqs = []

for motif_name in df_motif_info_fine.index:
    selected_motif = next(m for m in motifs if m.id == motif_name)

    # Extract the Position Frequency Matrix (PFM) and Position Weight Matrix (PWM)
    pfm = selected_motif.to_pfm()  # Transpose to make it compatible with logomaker
    pwm = selected_motif.to_ppm()

    # consensus sequence
    # print(motif_name)
    # print(selected_motif.to_consensus())
    # print(pwm)
    list_consensus_seqs.append(selected_motif.to_consensus())
    

# add the consensus sequence as additional column
df_motif_info_fine["consensus"] = list_consensus_seqs

df_motif_info_fine.head()

# %%
df_motif_info_fine.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_unified_cisBP_ver2_Danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors_fine_clusts_consensus.csv")

# %%

# %% [markdown]
# ### Check the clusters-by-motifs dataframes

# %%
df_clusters_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_motifs_maelstrom.csv", index_col=0)
df_clusters_motifs.head()

# %%
df_fine_clusters_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_unified_cluster_by_motifs.csv", index_col=0)
df_fine_clusters_motifs.head()
