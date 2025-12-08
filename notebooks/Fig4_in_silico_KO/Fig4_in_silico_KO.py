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
# ## Figure 4_in silico KO quantification/visualization
#
# - Last updated: 10/18/2024
# - Yang-Joon Kim
#
#
# - We have computed cell-cell trans.probs for WT and all in silico KO cases (roughly 240 genes/TFs).
# - We will use the Metacells computed separately, to (1) average the 2D transition vectors (single-cell level) across metacells, and (2) compute the metacell-metacell trans.probs, for "distance"/perturbation score computation.

# %%
# Import public libraries
import os
import sys

from scipy.stats import binned_statistic_2d
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm.notebook import tqdm

#import time

# %%
# Import CellOracle library
import celloracle as co
# from celloracle.applications import Oracle_development_module, Oracle_systematic_analysis_helper
co.__version__

# %%
# import CellRank modules
import cellrank as cr
import scvelo as scv

# %%
# Import project-specific utilities
from scripts.fig2_utils.plotting_utils import set_plotting_style
from scripts.fig4_utils.knockout_analysis import (
    compute_celltype_transitions,
    compute_metacell_transitions,
    get_top_genes_for_celltype
)
from scripts.fig4_utils.vector_field_utils import (
    average_2D_trans_vecs_metacells,
    plot_metacell_transitions,
    plot_metacell_cosine_sims
)
from scripts.fig4_utils.similarity_metrics import (
    compute_row_cosine_similarities,
    compute_row_euclidean_dist
)

# %% [markdown]
# ### Plotting parameter setting

# %%
#plt.rcParams["font.family"] = "arial"
plt.rcParams["figure.figsize"] = [5,5]
# %config InlineBackend.figure_format = 'retina'
plt.rcParams["savefig.dpi"] = 600
plt.rcParams['pdf.fonttype']=42

# %matplotlib inline

# %%
import logging

# Set the logging level to WARN, filtering out informational messages
logging.getLogger().setLevel(logging.WARNING)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Set the default font to Arial
mpl.rcParams['font.family'] = 'Arial'

# If Arial is not available on your system, you might need to specify an alternative or ensure Arial is installed.
# On some systems, you might need to use 'font.sans-serif' as a fallback option:
# mpl.rcParams['font.sans-serif'] = 'Arial'

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number for editable text in PDFs
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')

# Plotting style function (run this before plotting the final figure)
# %%
data_id = "TDR126"
figpath = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_NMPs_subsets_metacells/{data_id}/"

# create the directory if not present already
os.makedirs(figpath, exist_ok=True)

# %% [markdown]
# ### Step 1. Load the Oracle object with all cell-cell trans.probs

# %%
# import an oracle object
oracle = co.load_hdf5(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/14_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
oracle

# %% [markdown]
# ### Step 2. load the metacell information
#
# - NOTE that the Metacell computing parameters should be consistent across timepoints
# - We use "SEACells" method with 30cells/metacell.
#
#

# %%
metacell = pd.read_csv(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/metacells/{data_id}_seacells_obs_manual_annotation_30cells.csv", index_col=0)
metacell.head()

# make a dict - keys=cell_id : values=SEACEll
metacell_dict = metacell.SEACell.to_dict()

# %%
# add the metacell information to the oracle.adata
metacell_dict = metacell.SEACell.to_dict()
oracle.adata.obs["SEACell"] = oracle.adata.obs_names.map(metacell_dict)
oracle.adata.obs.head()

# %%
# Color palette for cell types
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'fast_muscle': '#df4b9b',
    'neural_posterior': '#393b7f',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}

# computing the average of 2D transition vectors @Metacell level
# %% [markdown]
# ### plotting the 2D projection of cell-cell transitions (pseudotime)
#
# - we have used three ways of computing the pseudotime
# - 1) pseudotime from each individual object
# - 2) pseudotime from "global (integrated" object - whole embryo
# - 3) pseudotime from "global (integrated" object - NMP trajectory subsetted

# %%
# WT ("local" pseudotime)
# average the 2D embedding and 2D transition vectors across "metacells"
X_metacell, V_metacell = average_2D_trans_vecs_metacells(oracle.adata, 
                                                         metacell_col="SEACell", 
                                                         basis='umap_aligned',
                                                         key_added='WT')

# generate the plot and save it in the folder
plot_metacell_transitions(oracle.adata, X_metacell, V_metacell,data_id="TDR126",
                            figpath=figpath,
                            metacell_col="SEACell", 
                            annotation_class="manual_annotation",
                            basis='umap_aligned', genotype="WT", arrow_scale=20)


# %%
# WT ("global" pseudotime)
# average the 2D embedding and 2D transition vectors across "metacells"
X_metacell, V_metacell = average_2D_trans_vecs_metacells(oracle.adata, 
                                                         metacell_col="SEACell", 
                                                         basis='umap_aligned',
                                                         key_added='WT_global')

# generate the plot and save it in the folder
plot_metacell_transitions(oracle.adata, X_metacell, V_metacell,data_id="TDR126",
                            figpath=figpath,
                            metacell_col="SEACell", 
                            annotation_class="manual_annotation",
                            basis='umap_aligned', genotype="WT_global", arrow_scale=20)

# %%
# WT ("global" pseudotime)
# average the 2D embedding and 2D transition vectors across "metacells"
X_metacell, V_metacell = average_2D_trans_vecs_metacells(oracle.adata, 
                                                         metacell_col="SEACell", 
                                                         basis='umap_aligned',
                                                         key_added='WT_global_nmps')

# generate the plot and save it in the folder
plot_metacell_transitions(oracle.adata, X_metacell, V_metacell,data_id="TDR126",
                            figpath=figpath,
                            metacell_col="SEACell", 
                            annotation_class="manual_annotation",
                            basis='umap_aligned', genotype="WT_global_nmps", arrow_scale=20)

# %% [markdown]
# ### step 2. 2D visualization of transition probabilities for the KOs

# %%
# Step 1: Extract the necessary data
transition_vectors = oracle.adata.obsm['transition_vectors']  # 2D vectors (n_cells x 2)
seacell_annotations = oracle.adata.obs['SEACell']  # SEACell annotations

# Step 2: Convert the transition vectors to a DataFrame for easier manipulation
transition_df = pd.DataFrame(transition_vectors, index=oracle.adata.obs_names, columns=['X_vector', 'Y_vector'])

# Step 3: Add SEACell annotations to the DataFrame
transition_df['SEACell'] = seacell_annotations

# Step 4: Group by SEACell and calculate the mean for each metacell
mean_transition_vectors = transition_df.groupby('SEACell')[['X_vector', 'Y_vector']].mean()

# Step 5: Store the averaged vectors in a new AnnData object or within the existing one
# Option 1: Store in the existing adata object
oracle.adata.obsm['metacell_transition_vectors'] = mean_transition_vectors.loc[seacell_annotations].values

# Option 2: Create a new AnnData object for metacells
import scanpy as sc

# Create a new AnnData object for metacells
metacell_adata = sc.AnnData(X=mean_transition_vectors.values)
metacell_adata.obs['SEACell'] = mean_transition_vectors.index

# %%
## repeat the above precedure for a gene KO
KO_gene = "ved"

genotype = f"{KO_gene}_KO"
print(genotype)

# average the 2D embedding and 2D transition vectors across "metacells"
X_metacell, V_metacell = average_2D_trans_vecs_metacells(oracle.adata, 
                                                         metacell_col="SEACell", 
                                                         basis='umap_aligned',
                                                         key_added=genotype)

# generate the plot and save it in the folder
plot_metacell_transitions(oracle.adata, X_metacell, V_metacell,data_id="TDR126",
                            figpath=figpath,
                            metacell_col="SEACell", 
                            annotation_class="manual_annotation",
                            basis='umap_aligned', genotype=genotype)

# %%
# generate the UMAP plots overlaid with 2D trans.vectors (averaged @metacell level)
for KO_gene in oracle.active_regulatory_genes:
    genotype = f"{KO_gene}_KO"

    # average the 2D embedding and 2D transition vectors across "metacells"
    X_metacell, V_metacell = average_2D_trans_vecs_metacells(oracle.adata, 
                                                             metacell_col="SEACell", 
                                                             basis='umap_aligned',
                                                             key_added=genotype)

    # generate the plot and save it in the folder
    plot_metacell_transitions(oracle.adata, X_metacell, V_metacell,data_id="TDR126",
                                figpath=figpath,
                                metacell_col="SEACell", 
                                annotation_class="manual_annotation",
                                basis='umap_aligned', genotype=genotype)

# %%
### make a for loop to generate the UMAP plots with transition vectors (metacell level)
list_datasets = ["TDR127","TDR128","TDR118","TDR119",
                 "TDR125","TDR124"]

for data_id in list_datasets:
    # define the figure path ({data_id} specific)
    figpath = f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_NMPs_subsets_metacells/{data_id}/"
    # create the directory if not present already
    os.makedirs(figpath, exist_ok=True)

    # load the Oracle object
    oracle = co.load_hdf5(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/14_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
    oracle
    
    # load the metacell info
    metacell = pd.read_csv(f"/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/metacells/{data_id}_seacells_obs_manual_annotation_30cells.csv", index_col=0)
    metacell.head()

    # make a dict - keys=cell_id : values=SEACEll
    metacell_dict = metacell.SEACell.to_dict()
    
    # add the metacell information to the oracle.adata
    oracle.adata.obs["SEACell"] = oracle.adata.obs_names.map(metacell_dict)
    oracle.adata.obs.head()
    
    # generate the UMAP plots overlaid with 2D trans.vectors (averaged @metacell level, WT)
    for genotype in ["WT","WT_global","WT_global_nmps"]:
        # average the 2D embedding and 2D transition vectors across "metacells"
        X_metacell, V_metacell = average_2D_trans_vecs_metacells(oracle.adata, 
                                                                 metacell_col="SEACell", 
                                                                 basis='umap_aligned',
                                                                 key_added=genotype)

        # generate the plot and save it in the folder
        plot_metacell_transitions(oracle.adata, X_metacell, V_metacell,
                                    data_id=data_id,
                                    figpath=figpath,
                                    metacell_col="SEACell", 
                                    annotation_class="manual_annotation",
                                    basis='umap_aligned', genotype=genotype)
    
    # generate the UMAP plots overlaid with 2D trans.vectors (averaged @metacell level)
    for KO_gene in oracle.active_regulatory_genes:
        genotype = f"{KO_gene}_KO"

        # average the 2D embedding and 2D transition vectors across "metacells"
        X_metacell, V_metacell = average_2D_trans_vecs_metacells(oracle.adata, 
                                                                 metacell_col="SEACell", 
                                                                 basis='umap_aligned',
                                                                 key_added=genotype)

        # generate the plot and save it in the folder
        plot_metacell_transitions(oracle.adata, X_metacell, V_metacell,
                                    data_id=data_id,
                                    figpath=figpath,
                                    metacell_col="SEACell", 
                                    annotation_class="manual_annotation",
                                    basis='umap_aligned', genotype=genotype)
        
    print(f"{data_id} is completed")

# %% [markdown]
# ### step 3. quantify the effect of perturbation
# ### 3-1. generate celltype-celltype trans.probs matrices
#
#
# - idea is illustrated in detail in the methods section.
#
# - we're quantifying the effect of perturbation using the cosine similarity between celltype-celltype transition probabilities between "WT", and "KO" cases.
#
#

# %%
oracle.adata.obsp["T_fwd_WT"]

# %%
# # Step 1: Create a mapping from SEACell to celltype
# seacell_to_celltype = df.set_index('SEACell')['celltype'].to_dict()

# # Step 2: Get unique celltypes
# celltypes = df['manual_annotation'].unique()

# # Step 3: Create an empty celltype transition matrix
# celltype_trans_probs = pd.DataFrame(0, index=celltypes, columns=celltypes)

# # Step 4: Fill the celltype transition matrix
# for i, source_metacell in enumerate(metacell_trans_probs.index):
#     source_celltype = seacell_to_celltype[source_metacell]
#     for j, target_metacell in enumerate(metacell_trans_probs.columns):
#         target_celltype = seacell_to_celltype[target_metacell]
#         celltype_trans_probs.loc[source_celltype, target_celltype] += metacell_trans_probs.iloc[i, j]

# # Step 5: Normalize the celltype transition matrix
# celltype_trans_probs = celltype_trans_probs.div(celltype_trans_probs.sum(axis=1), axis=0)


# %%
oracle.adata.obs.SEACell


# %%
celltype_transitions = compute_celltype_transitions(oracle.adata)
celltype_transitions
# # Visualize the result
# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(12, 10))
# sns.heatmap(celltype_transitions, annot=True, cmap='YlOrRd', fmt='.2f')
# plt.title('Celltype-to-Celltype Transition Probabilities')
# plt.tight_layout()
# plt.show()

# %%
np.sum(celltype_transitions,1)


# %%
# %%
# generate the UMAP plots overlaid with 2D trans.vectors (averaged @metacell level)

# average the 2D embedding and 2D transition vectors across "metacells"
trans_probs_ct_WT = compute_celltype_transitions(oracle.adata, trans_key="T_fwd_WT", 
                                 celltype_key="manual_annotation")

trans_probs_ct_KO = compute_celltype_transitions(oracle.adata, trans_key="T_fwd_meox1_KO", 
                                 celltype_key="manual_annotation")

cosine_similarities = compute_row_cosine_similarities(trans_probs_ct_WT, trans_probs_ct_KO)
cosine_similarities

# %%
# average the 2D embedding and 2D transition vectors across "metacells"
trans_probs_ct_WT = compute_celltype_transitions(oracle.adata, trans_key="T_fwd_WT", 
                                 celltype_key="manual_annotation")

# Initialize an empty DataFrame with celltypes as the index
celltypes = trans_probs_ct_WT.index
cosine_sim_df = pd.DataFrame(index=celltypes)

# Compute cosine similarities for each gene knockout
for gene in oracle.active_regulatory_genes:
    # Compute transition probabilities for the current gene knockout
    trans_key = f"T_fwd_{gene}_KO"
    trans_probs_ct_KO = compute_celltype_transitions(oracle.adata, trans_key=trans_key, 
                                                     celltype_key="manual_annotation")
    
    # Compute cosine similarities
    cosine_similarities = compute_row_cosine_similarities(trans_probs_ct_WT, trans_probs_ct_KO)
    
    # Add the cosine similarities as a new column to the DataFrame
    cosine_sim_df[gene] = cosine_similarities

# Display the resulting DataFrame
print(cosine_sim_df)

# %%
cosine_sim_df.loc["somites"].sort_values(ascending=False)


# %%
celltype = "fast_muscle"  # Replace with your celltype of interest
top_genes = get_top_genes_for_celltype(cosine_sim_df, celltype, n=20)

print(f"Top 20 genes with lowest cosine similarity for {celltype}:")
print(top_genes)

# %%
# Example usage:
celltype = "somites"  # Replace with your celltype of interest
top_genes = get_top_genes_for_celltype(cosine_sim_df, celltype, n=20)

print(f"Top 20 genes with lowest cosine similarity for {celltype}:")
print(top_genes)

# %%
# Example usage:
celltype = "spinal_cord"  # Replace with your celltype of interest
top_genes = get_top_genes_for_celltype(cosine_sim_df, celltype, n=20)

print(f"Top 20 genes with lowest cosine similarity for {celltype}:")
print(top_genes)


# %% [markdown]
# #### Conclusion
#
# When we coarse-grained everything at the celltype level, then the transition probabilities between WT and KO becomes very subtle - this could be problematic when we "score" the TFs for their strength of KO.

# %% [markdown]
# ## step 3. (Continued) quantify the effect of perturbation
# ### 3-2. generate metacell-metacell trans.probs matrices
#
#
# - idea is illustrated in detail in the methods section.
#
# - we're quantifying the effect of perturbation using the cosine similarity between celltype-celltype transition probabilities between "WT", and "KO" cases.
#

# %%
# %%
metacell_transitions

# %%
# plot the distribution (histogram) of metacell-metacell transition probabilities
plt.hist(metacell_transitions.values.flatten(),bins=20)
plt.yscale("log")
plt.xlabel("metacell-metacell trans.probs")
plt.ylabel("occurences")
plt.show()

# %%
# Calculate most prevalent cell type for each metacell
most_prevalent = oracle.adata.obs.groupby("SEACell")["manual_annotation"].agg(lambda x: x.value_counts().idxmax())
most_prevalent

# %%
most_prevalent.loc["SEACell-35"]

# %%
most_prevalent.index

# %%
most_prevalent.value_counts()

# %%
# compute the metacell-metacell transition probabilities for a couple of genotypes (WT and KO)
trans_probs_metacell_WT = compute_metacell_transitions(oracle.adata, trans_key="T_fwd_WT", metacell_key="SEACell")
trans_probs_metacell_meox1 = compute_metacell_transitions(oracle.adata, trans_key="T_fwd_meox1_KO", metacell_key="SEACell")



# %%
# average the 2D embedding and 2D transition vectors across "metacells"
trans_probs_metacell_WT = compute_metacell_transitions(oracle.adata, 
                                                       trans_key="T_fwd_WT", 
                                                       metacell_key="SEACell")


# Initialize an empty DataFrame with celltypes as the index
metacells = trans_probs_metacell_WT.index
cosine_sim_df = pd.DataFrame(index=metacells)

# Compute cosine similarities for each gene knockout
for gene in oracle.active_regulatory_genes:
    # Compute transition probabilities for the current gene knockout
    trans_key = f"T_fwd_{gene}_KO"
    trans_probs_metacell_KO = compute_metacell_transitions(oracle.adata, trans_key=trans_key, 
                                                             metacell_key="SEACell")
    
    # Compute cosine similarities
    cosine_similarities = compute_row_cosine_similarities(trans_probs_metacell_WT, trans_probs_metacell_KO)
    
    # Add the cosine similarities as a new column to the DataFrame
    cosine_sim_df[gene] = cosine_similarities

# Display the resulting DataFrame
cosine_sim_df

# %%
# Prepare data for plotting
adata = oracle.adata
basis = "umap_aligned"
metacell_col = "SEACell"
annotation_class = "manual_annotation"
cell_size = 10


umap_coords = pd.DataFrame(adata.obsm[f'X_{basis}'], columns=[0, 1], index=adata.obs_names)
umap_data = umap_coords.join(adata.obs[[metacell_col, annotation_class]])
umap_data = umap_data.rename(columns={annotation_class: 'celltype'})

# # Plot single cells
# sns.scatterplot(
#     x=0, y=1, hue='celltype', data=umap_data, s=cell_size, 
#     palette=cell_type_color_dict, legend=None, ax=ax, alpha=0.7
# )

# Calculate most prevalent cell type for each metacell
most_prevalent = adata.obs.groupby(metacell_col)[annotation_class].agg(lambda x: x.value_counts().idxmax())

# Prepare metacell data
mcs = umap_data.groupby(metacell_col).mean().reset_index()
mcs['celltype'] = most_prevalent.values

# %%
# mcs.set_index("SEACell", inplace=True)
mcs

# %%
cosine_sim_df

# %%
mcs_merged = pd.concat([mcs, cosine_sim_df], axis=1)
mcs_merged


# %%
# %%
plot_metacell_cosine_sims(adata, X_metacell, cosine_sim_df, gene="tbx16", vmin=0, vmax=1)

# %%
cosine_sim_df["celltype"] = cosine_sim_df.index.map(most_prevalent)
cosine_sim_df.head()

# %%
cosine_sim_df_avg = cosine_sim_df.groupby("celltype").mean()
df_averaged = cosine_sim_df_avg.reset_index()

df_averaged

# %%
df_averaged.drop("celltype", axis=1).mean(axis=0)

# %%
plt.hist(df_averaged.drop("celltype", axis=1).mean(axis=0))
plt.show()

# %%
df_avg_per_gene = df_averaged.drop("celltype", axis=1).mean(axis=0)
df_avg_per_gene

# %%
trans_probs_WT = oracle.adata.obsp["T_fwd_WT_global_nmps"]
trans_probs1 = oracle.adata.obsp["T_fwd_meox1_KO"]
trans_probs2 = oracle.adata.obsp["T_fwd_pax6b_KO"]

# %%
df_averaged.drop("celltype", axis=1).values.flatten()

# %%
plt.hist(df_averaged.drop("celltype", axis=1).values.flatten())
plt.show()

# %%
df_averaged.loc[:,"meis3"]

# %%
df_averaged_per_gene = df_averaged.mean(axis=0)
df_averaged_per_gene

# %% [markdown]
# ## Step 4. computng the cosine.similarity (metacell-level) between WT and KO
#
# - inspired by what we did for Sarah's RNA velocity paper

# %%
from tqdm import tqdm

list_datasets = ['TDR126','TDR127','TDR128','TDR118',
                 'TDR119','TDR125','TDR124']

dict_cos_sims = {}

oracle_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"

for data_id in tqdm(list_datasets, desc="Processing Datasets"):
    print(data_id)
    
    # load the oracle object
    oracle = co.load_hdf5(oracle_path + f"15_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
    
    # load the metacell info
    metacell = pd.read_csv(oracle_path + f"metacells/{data_id}_seacells_obs_manual_annotation_30cells.csv", index_col=0)
    
    # add the metacell information to the oracle.adata
    metacell_dict = metacell.SEACell.to_dict()
    oracle.adata.obs["SEACell"] = oracle.adata.obs_names.map(metacell_dict)
    oracle.adata.obs.head()
    
    # Calculate most prevalent cell type for each metacell
    most_prevalent = oracle.adata.obs.groupby("SEACell")["manual_annotation"].agg(lambda x: x.value_counts().idxmax())
    most_prevalent

    # average the 2D embedding and 2D transition vectors across "metacells"
    trans_probs_metacell_WT = compute_metacell_transitions(oracle.adata, 
                                                        trans_key="T_fwd_WT_global_nmps", 
                                                        metacell_key="SEACell")

    # Initialize an empty DataFrame with celltypes as the index
    metacells = trans_probs_metacell_WT.index
    cosine_sim_df = pd.DataFrame(index=metacells)
    
    # Compute cosine similarities for each gene knockout
    for gene in tqdm(oracle.active_regulatory_genes, desc=f"Processing Genes for {data_id}"):
        # Compute transition probabilities for the current gene knockout
        trans_key = f"T_fwd_{gene}_KO"
        trans_probs_metacell_KO = compute_metacell_transitions(oracle.adata, trans_key=trans_key, 
                                                                metacell_key="SEACell")
        
        # Compute cosine similarities
        cosine_similarities = compute_row_cosine_similarities(trans_probs_metacell_WT, trans_probs_metacell_KO)
        
        # Add the cosine similarities as a new column to the DataFrame
        cosine_sim_df[gene] = cosine_similarities

    # Display the resulting DataFrame (metacell-by-genes)
    cosine_sim_df["celltype"] = cosine_sim_df.index.map(most_prevalent)

    # average the cosine similarities across cell types
    cosine_sim_df_avg = cosine_sim_df.groupby("celltype").mean()
    df_averaged = cosine_sim_df_avg.reset_index()
    
    # save the dataframes
    cosine_sim_df.to_csv(oracle_path + f"{data_id}/cosine_similarity_df_metacells_{data_id}.csv")
    df_averaged.to_csv(oracle_path + f"{data_id}/cosine_similarity_df_averaged_{data_id}.csv")
    
    # save this into the master dictionary (dict)
    dict_cos_sims[data_id] = df_averaged
    # save the raw version (metacells-by-genes)
    dict_cos_sims[f"{data_id}_metacells"] = cosine_sim_df
    
    print(f"{data_id} is completed")

# %%

# %%
# Load the dictionary (resumption from here)

# Path to your data
oracle_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"

for data_id in list_datasets:
    print(f"Loading data for {data_id}")
    
    try:
        # Load the DataFrames from CSV files
        cosine_sim_df = pd.read_csv(
            oracle_path + f"{data_id}/cosine_similarity_df_metacells_{data_id}.csv", 
            index_col=0
        )
        df_averaged = pd.read_csv(
            oracle_path + f"{data_id}/cosine_similarity_df_averaged_{data_id}.csv", 
            index_col=0
        )
        
        # Store the DataFrames in the dictionary
        dict_cos_sims[data_id] = df_averaged
        dict_cos_sims[f"{data_id}_metacells"] = cosine_sim_df
        
        print(f"{data_id} data loaded successfully")
    except FileNotFoundError:
        print(f"Data files for {data_id} not found. Skipping this dataset.")

# %%
dict_cos_sims["TDR118"]

# %%
# subset the mesoderm celltypes
df_meso = dict_cos_sims["TDR118"][dict_cos_sims["TDR118"].celltype.isin(["NMPs","PSM","fast_muscle","somites"])]
df_meso

df_meso_avg = df_meso.median(axis=0)
df_meso_avg.sort_values(ascending=False)

# %%
# subset the mesoderm celltypes
df_meso = dict_cos_sims["TDR118"][dict_cos_sims["TDR118"].celltype.isin(["NMPs","PSM","fast_muscle","somites"])]
df_meso

df_meso_avg = df_meso.median(axis=0)
df_meso_avg.sort_values(ascending=False)

# subset the neuro-ectoderm celltypes
df_ne = dict_cos_sims["TDR118"][dict_cos_sims["TDR118"].celltype.isin(["NMPs","neural_posterior","spinal_cord"])]
df_ne

df_ne_avg = df_ne.median(axis=0)
df_ne_avg.sort_values(ascending=False)

# %%
# plt.hist(df_meso_avg.values, bins=20, alpha=0.5)
# plt.hist(df_ne_avg.values, bins=20, alpha=0.5)
# plt.legend(["mesoderm","neuro-ectoderm"])
# plt.xlabel("perturbation score")
# plt.ylabel("occurences")
# plt.grid(False)
# plt.show()

# %%
df_meso_avg.to_frame(name="meso")
df_ne_avg.to_frame(name="ne")


# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_NMPs_subsets_metacells/cosine_sims/"
os.makedirs(figpath, exist_ok=True)

# %%
dict_cos_sims[data_id]

# %%
list_datasets

for data_id in list_datasets:
    # subset the mesoderm celltypes
    df_meso = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["PSM","fast_muscle","somites"])]
    # compute the median across celltypes
    df_meso_avg = df_meso.median(axis=0)
    # compute the perturbation score (1-cos.similarity)
    df_meso_avg = 1 - df_meso_avg
    # df_meso_avg.sort_values(ascending=False)

# subset the neuro-ectoderm celltypes
    df_ne = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["neural_posterior","spinal_cord"])]
    # compute the median across celltypes
    df_ne_avg = df_ne.median(axis=0)
    # compute the perturbation score (1-cos.similarity)
    df_ne_avg = 1 - df_ne_avg
    # df_ne_avg.sort_values(ascending=False)
    
    df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))
    df_merged
    
    # Find top 5 genes for each axis
    top_meso = df_merged.nlargest(5, 'meso')
    top_ne = df_merged.nlargest(5, 'ne')

    # Combine top genes (in case there's overlap)
    top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()
    
    plt.scatter(df_merged["meso"], df_merged["ne"])
    # Add labels for top genes
    for idx, row in top_genes.iterrows():
        plt.annotate(idx, (row['meso'], row['ne']), 
                     xytext=(5, 5), textcoords='offset points', 
                     ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', alpha=0),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.xlabel("mesoderm")
    plt.ylabel("neuro-ectoderm")
    # plt.xlim([0.4, 0.85])
    # plt.ylim([0.4, 0.65])
    plt.title(f"{data_id}")
    plt.grid(False)
    # plt.savefig(figpath+f"scatter_cosine_sim_{data_id}.pdf")
    plt.show()

# %%
# List of datasets
list_datasets = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR119', 'TDR125', 'TDR124']

# List of genes to highlight
highlight_genes = ["meox1", "myog", "myod1", "pax6a"]

for data_id in list_datasets:
    # Subset the mesoderm cell types
    df_meso = dict_cos_sims[data_id][
        dict_cos_sims[data_id]['celltype'].isin(["PSM", "fast_muscle", "somites"])
    ]

    # Compute the median across cell types
    df_meso_avg = df_meso.median(axis=0)

    # Compute the perturbation score (1 - cosine similarity)
    df_meso_avg = 1 - df_meso_avg

    # Subset the neuro-ectoderm cell types
    df_ne = dict_cos_sims[data_id][
        dict_cos_sims[data_id]['celltype'].isin(["neural_posterior", "spinal_cord"])
    ]

    # Compute the median across cell types
    df_ne_avg = df_ne.median(axis=0)

    # Compute the perturbation score (1 - cosine similarity)
    df_ne_avg = 1 - df_ne_avg

    # Merge the mesoderm and neuro-ectoderm scores into a DataFrame
    df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))

    # Remove the 'celltype' row if it exists (since 'celltype' is not a gene)
    df_merged = df_merged.drop(index='celltype', errors='ignore')

    # Find top 5 genes for each axis
    top_meso = df_merged.nlargest(5, 'meso')
    top_ne = df_merged.nlargest(5, 'ne')

    # Combine top genes (in case there's overlap)
    top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()

    # Identify the highlight genes present in the data
    highlight_genes_in_data = df_merged.loc[df_merged.index.isin(highlight_genes)]

    # Begin plotting
    plt.figure(figsize=(8, 6))

    # Scatter plot for all genes
    plt.scatter(df_merged["meso"], df_merged["ne"], s=10, alpha=0.5, color='grey', label='Other Genes')

    # Scatter plot for highlight genes
    if not highlight_genes_in_data.empty:
        plt.scatter(
            highlight_genes_in_data["meso"],
            highlight_genes_in_data["ne"],
            s=50,
            color='red',
            label='Highlighted Genes'
        )

    # Annotate top genes
    for idx, row in top_genes.iterrows():
        plt.annotate(
            idx,
            (row['meso'], row['ne']),
            xytext=(5, 5),
            textcoords='offset points',
            ha='left',
            va='bottom',
            fontsize=9,
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

    # Annotate highlight genes
    for idx, row in highlight_genes_in_data.iterrows():
        plt.annotate(
            idx,
            (row['meso'], row['ne']),
            xytext=(-5, -5),
            textcoords='offset points',
            ha='right',
            va='top',
            fontsize=10,
            color='blue',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue', connectionstyle='arc3,rad=0')
        )

    plt.xlabel("Perturbation Score in Mesoderm", fontsize=12)
    plt.ylabel("Perturbation Score in Neuro-Ectoderm", fontsize=12)
    plt.title(f"Perturbation Scores for {data_id}", fontsize=14)
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# Set up the figure
import matplotlib.ticker as ticker

fig, axs = plt.subplots(1, 7, figsize=(35, 5))  # 1 row, 7 columns
fig.suptitle("Gene Perturbation Scores Across Datasets", fontsize=16)

# Initialize dictionaries to store data
meso_data = {}
ne_data = {}
# df_ps_meso = pd.DataFrame()
# df_ps_ne = pd.DataFrame()

for idx, data_id in enumerate(list_datasets):
    # subset the mesoderm celltypes
    df_meso = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["PSM","fast_muscle","somites"])]
    df_meso_avg = 1 - df_meso.median(axis=0)

    # subset the neuro-ectoderm celltypes
    df_ne = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["neural_posterior","spinal_cord"])]
    df_ne_avg = 1 - df_ne.median(axis=0)
    
    df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))
    
    # Find top 5 genes for each axis
    top_meso = df_merged.nlargest(5, 'meso')
    top_ne = df_merged.nlargest(5, 'ne')
    top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()
    
    # Plot on the corresponding subplot
    ax = axs[idx]
    ax.scatter(df_merged["meso"], df_merged["ne"], alpha=0.5)
    
    # Add labels for top genes
    for gene, row in top_genes.iterrows():
        # ax.annotate(gene, (row['meso'], row['ne']), 
        #              xytext=(5, 5), textcoords='offset points', 
        #              ha='left', va='bottom',
        #              bbox=dict(boxstyle='round,pad=0.5', alpha=0),
        #              arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        ax.text(row['meso'], row['ne'], gene, 
        fontsize=8, ha='left', va='bottom')
    
    ax.set_xlabel("Mesoderm")
    ax.set_ylabel("Neuro-ectoderm")
    ax.set_xlim([0.4, 0.825])
    ax.set_ylim([0.4, 0.62])
    # Set custom ticks for x and y axes
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticks([0.4, 0.5, 0.6])
    # Make ticks visible
    # ax.tick_params(axis='both', which='both', length=5, width=1, direction='out')
    ax.tick_params(axis='both', which='both', length=5, width=1, direction='out', colors='black', labelsize=10)
    # Ensure tick labels are not empty
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Show the tick lines (grid) if desired
    # ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
    ax.set_title(f"{data_id}")
    ax.grid(False)
    
    # Store the data
    meso_data[data_id] = df_meso_avg
    ne_data[data_id] = df_ne_avg

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
# plt.savefig(figpath + "scatter_median_perturbation_score_all_datasets_uniform_axes_scale.pdf", bbox_inches='tight')
plt.show()

# %%
fig, axs = plt.subplots(1, 7, figsize=(35, 5))  # 1 row, 7 columns
fig.suptitle("Gene Perturbation Scores Across Datasets", fontsize=16)

# Initialize dictionaries to store data
meso_data = {}
ne_data = {}

for idx, data_id in enumerate(list_datasets):
    # subset the mesoderm celltypes
    df_meso = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["PSM", "fast_muscle", "somites"])]
    df_meso_avg = 1 - df_meso.median(axis=0)

    # subset the neuro-ectoderm celltypes
    df_ne = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["neural_posterior", "spinal_cord"])]
    df_ne_avg = 1 - df_ne.median(axis=0)
    
    df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))
    
    # Find top 5 genes for each axis
    top_meso = df_merged.nlargest(5, 'meso')
    top_ne = df_merged.nlargest(5, 'ne')
    top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()
    
    # Plot on the corresponding subplot
    ax = axs[idx]
    ax.scatter(df_merged["meso"], df_merged["ne"], alpha=0.5)
    
    # Add labels for top genes
    for gene, row in top_genes.iterrows():
        ax.text(row['meso'], row['ne'], gene, fontsize=8, ha='left', va='bottom')
    
    ax.set_xlabel("Mesoderm")
    ax.set_ylabel("Neuro-ectoderm")
    ax.set_xlim([0.4, 0.825])
    ax.set_ylim([0.4, 0.62])
    
    # Set custom ticks for x and y axes
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticks([0.4, 0.5, 0.6])
    
    # Explicitly set tick labels
    ax.set_xticklabels([str(x) for x in [0.4, 0.5, 0.6, 0.7, 0.8]])
    ax.set_yticklabels([str(y) for y in [0.4, 0.5, 0.6]])
    
    # Ensure tick marks are visible
    ax.tick_params(axis='both', which='both', length=5, width=1, direction='out', colors='black', labelsize=10)
    
    # Ensure tick labels are not empty
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    ax.grid(False)
    ax.set_title(f"{data_id}")
    
    # Store the data
    meso_data[data_id] = df_meso_avg
    ne_data[data_id] = df_ne_avg

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
plt.show()

# %%
# Create DataFrames
df_meso_all = pd.DataFrame(meso_data)
df_ne_all = pd.DataFrame(ne_data)

# Clean up the rows with NaNs 
# For the mesoderm DataFrame
df_meso_all_clean = df_meso_all.dropna()
# For the neuro-ectoderm DataFrame
df_ne_all_clean = df_ne_all.dropna()

df_meso_all_clean.head()

# %%
# Set up the figure
fig, axs = plt.subplots(1, 7, figsize=(35, 5))  # 1 row, 7 columns
fig.suptitle("Gene Perturbation Scores Across Datasets", fontsize=16)

for idx, data_id in enumerate(list_datasets):
    # subset the mesoderm celltypes
    df_meso = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["PSM","fast_muscle","somites"])]
    df_meso_avg = 1 - df_meso.mean(axis=0)

    # subset the neuro-ectoderm celltypes
    df_ne = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["neural_posterior","spinal_cord"])]
    df_ne_avg = 1 - df_ne.mean(axis=0)
    
    df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))
    
    # Find top 5 genes for each axis
    top_meso = df_merged.nlargest(5, 'meso')
    top_ne = df_merged.nlargest(5, 'ne')
    top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()
    
    # Plot on the corresponding subplot
    ax = axs[idx]
    ax.scatter(df_merged["meso"], df_merged["ne"], alpha=0.5)
    
    # Add labels for top genes
    for gene, row in top_genes.iterrows():
        ax.annotate(gene, (row['meso'], row['ne']), 
                     xytext=(5, 5), textcoords='offset points', 
                     ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', alpha=0),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel("Mesoderm")
    ax.set_ylabel("Neuro-ectoderm")
    ax.set_xlim([0.45, 0.8])
    ax.set_ylim([0.45, 0.62])
    ax.set_title(f"{data_id}")
    ax.grid(False)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
plt.savefig(figpath + "scatter_mean_perturbation_score_all_datasets.pdf", bbox_inches='tight')
plt.show()

# %%
list_datasets

for data_id in list_datasets:
    # subset the mesoderm celltypes
    df_meso = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["PSM","fast_muscle","somites"])]
    # compute the median across celltypes
    df_meso_avg = df_meso.mean(axis=0)
    # compute the perturbation score (1-cos.similarity)
    df_meso_avg = 1 - df_meso_avg
    # df_meso_avg.sort_values(ascending=False)

# subset the neuro-ectoderm celltypes
    df_ne = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["neural_posterior","spinal_cord"])]
    # compute the median across celltypes
    df_ne_avg = df_ne.mean(axis=0)
    # compute the perturbation score (1-cos.similarity)
    df_ne_avg = 1 - df_ne_avg
    # df_ne_avg.sort_values(ascending=False)
    
    df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))
    df_merged
    
    # Find top 5 genes for each axis
    top_meso = df_merged.nlargest(5, 'meso')
    top_ne = df_merged.nlargest(5, 'ne')

    # Combine top genes (in case there's overlap)
    top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()
    
    plt.scatter(df_merged["meso"], df_merged["ne"])
    # Add labels for top genes
    for idx, row in top_genes.iterrows():
        plt.annotate(idx, (row['meso'], row['ne']), 
                     xytext=(5, 5), textcoords='offset points', 
                     ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', alpha=0),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.xlabel("mesoderm")
    plt.ylabel("neuro-ectoderm")
    # plt.xlim([0.4, 0.85])
    # plt.ylim([0.4, 0.65])
    plt.title(f"{data_id}")
    plt.grid(False)
    # plt.savefig(figpath+f"scatter_cosine_sim_{data_id}.pdf")
    plt.show()

# %%

# %% [markdown]
# ### Revisit on 9/16/2024

# %%
# import the dataframes
oracle_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"

list_datasets = ['TDR126','TDR127','TDR128','TDR118',
                 'TDR119','TDR125','TDR124']

dict_cos_sims = {}

for data_id in list_datasets:
    # cosine_sim_df = pd.read_csv(oracle_path + f"{data_id}/cosine_similarity_df_metacells_{data_id}.csv")
    df_averaged = pd.read_csv(oracle_path + f"{data_id}/cosine_similarity_df_averaged_{data_id}.csv", index_col=0)

    # save this into the master dictionary (dict)
    dict_cos_sims[data_id] = df_averaged


# %%
dict_marker_genes = {"TDR126":["ved","vox","hmga1a","cdx4"],
                     "TDR127":["cdx4","tbx16","sox5"],
                     "TDR128":["meis2a","pax6a","dmrt2a","mef2d","meis1a"],
                     "TDR118":["hoxb3a","gli2a"],
                     "TDR119":["hoxb3a","gli2a"],
                     "TDR125":["hoxb3a","meis1a","meis1b","rarga","hoxc3a","sox5","hoxc6b"],
                     "TDR124":["pax6a","hoxa4a","myog","myod1","foxd3","mef2d","meox1","hmga1a"]}

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_NMPs_subsets_metacells/cosine_sims/"

# %%
list_datasets


for data_id in list_datasets:
    # subset the mesoderm celltypes
    df_meso = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["PSM","fast_muscle","somites"])]
    # compute the median across celltypes
    df_meso_avg = df_meso.median(axis=0)
    # compute the perturbation score (1-cos.similarity)
    df_meso_avg = 1 - df_meso_avg
    # df_meso_avg.sort_values(ascending=False)

# subset the neuro-ectoderm celltypes
    df_ne = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["neural_posterior","spinal_cord"])]
    # compute the median across celltypes
    df_ne_avg = df_ne.median(axis=0)
    # compute the perturbation score (1-cos.similarity)
    df_ne_avg = 1 - df_ne_avg
    # df_ne_avg.sort_values(ascending=False)
    
    df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))
    df_merged
    
    # Find top 5 genes for each axis
    # top_meso = df_merged.nlargest(5, 'meso')
    # top_ne = df_merged.nlargest(5, 'ne')

    # Combine top genes (in case there's overlap)
    # top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()
    top_genes = dict_marker_genes[data_id]
    # Create a color palette with as many colors as there are genes to annotate
    gene_colors = sns.color_palette("Set2", len(top_genes))
    
    # Annotate the selected marker genes
    # for gene in top_genes:
    #     if gene in df_merged.index:  # Ensure the gene exists in the merged dataframe
    #         row = df_merged.loc[gene]
    #         plt.annotate(gene, (row['meso'], row['ne']), 
    #                      xytext=(5, 5), textcoords='offset points', 
    #                      ha='left', va='bottom',
    #                      bbox=dict(boxstyle='round,pad=0.5', alpha=0),
    #                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.scatter(df_merged["meso"], df_merged["ne"])
    
    # plt.scatter(df_merged["meso"], df_merged["ne"], color='lightgray', alpha=0.7, label="Other Genes")
    
    # # Color the points corresponding to the annotated genes
    # for i, gene in enumerate(top_genes):
    #     if gene in df_merged.index:  # Ensure the gene exists in the merged dataframe
    #         row = df_merged.loc[gene]
    #         # Scatter plot for each gene
    #         plt.scatter(row['meso'], row['ne'], color=gene_colors[i], label=gene, s=100, edgecolor='black')
    #         # Annotate the gene (remove the arrow, match the text color to the dot color)
    #         plt.annotate(gene, (row['meso'], row['ne']), 
    #                      xytext=(5, 5), textcoords='offset points', 
    #                      ha='left', va='bottom',
    #                      color=gene_colors[i],  # Set annotation text color to match dot color
    #                      bbox=dict(boxstyle='round,pad=0.5', alpha=0))  # No arrow props
        # Color the points corresponding to the annotated genes with smaller dots
    for i, gene in enumerate(top_genes):
        if gene in df_merged.index:  # Ensure the gene exists in the merged dataframe
            row = df_merged.loc[gene]
            # Scatter plot for each gene with smaller dots
            plt.scatter(row['meso'], row['ne'], color=gene_colors[i], label=gene, s=70, edgecolor='black')  # Smaller dots (s=70)
            
            # Dynamically adjust the text position to avoid overlap
            x_offset = 5 if i % 2 == 0 else -5  # Alternate x offset
            y_offset = 5 + (i % 3) * 5  # Vary y offset based on index
            
            # Annotate the gene (remove the arrow, match the text color to the dot color)
            plt.annotate(gene, (row['meso'], row['ne']), 
                         xytext=(x_offset, y_offset), textcoords='offset points', 
                         ha='left' if i % 2 == 0 else 'right', va='bottom',
                         color=gene_colors[i],  # Set annotation text color to match dot color
                         bbox=dict(boxstyle='round,pad=0.5', alpha=0))  # No arrow props
    
    

    plt.xlabel("mesoderm")
    plt.ylabel("neuro-ectoderm")
    plt.xlim([0.4, 0.85])
    plt.ylim([0.4, 0.65])
    plt.title(f"{data_id}")
    plt.grid(False)
    # plt.savefig(figpath+f"scatter_cosine_sim_{data_id}.pdf")
    plt.show()

# %%

# %%
df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))
df_merged

# %%
plt.scatter(df_merged["meso"], df_merged["ne"])

# Find top 5 genes for each axis
top_meso = df_merged.nlargest(5, 'meso')
top_ne = df_merged.nlargest(5, 'ne')

# Combine top genes (in case there's overlap)
top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()

# Add labels for top genes
for idx, row in top_genes.iterrows():
    plt.annotate(idx, (row['meso'], row['ne']), 
                 xytext=(5, 5), textcoords='offset points', 
                 ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', alpha=0),
                 arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.xlabel('Meso Score')
plt.ylabel('NE Score')
plt.title('Meso vs NE Scores with Top 5 Genes Labeled')
plt.grid(False)

plt.show()

# %%
df_merged.sort_values("ne",ascending=False)

# %%
df_averaged_temp = dict_cos_sims["TDR118"].drop("celltype", axis=1).median(axis=0)
df_averaged_temp.sort_values(ascending=False)

# %%
df_averaged_meso = dict_cos_sims["TDR118"].drop("celltype", axis=1).iloc[0:3,:].median(axis=0)
df_averaged_meso.sort_values(ascending=False)

# %%
df_averaged_endo = dict_cos_sims["TDR118"].drop("celltype", axis=1).iloc[3:,:].median(axis=0)
df_averaged_meso.sort_values(ascending=False)


# %% [markdown]
# ## Step 5. Euclidean distances

# %%
# %%
from tqdm import tqdm

list_datasets = ['TDR126','TDR127','TDR128','TDR118',
                 'TDR119','TDR125','TDR124']

dict_euclid_dist = {}

oracle_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"

for data_id in tqdm(list_datasets, desc="Processing Datasets"):
    print(data_id)
    
    # load the oracle object
    oracle = co.load_hdf5(oracle_path + f"15_{data_id}_in_silico_KO_trans_probs_added.celloracle.oracle")
    
    # load the metacell info
    metacell = pd.read_csv(oracle_path + f"metacells/{data_id}_seacells_obs_manual_annotation_30cells.csv", index_col=0)
    
    # add the metacell information to the oracle.adata
    metacell_dict = metacell.SEACell.to_dict()
    oracle.adata.obs["SEACell"] = oracle.adata.obs_names.map(metacell_dict)
    oracle.adata.obs.head()
    
    # Calculate most prevalent cell type for each metacell
    most_prevalent = oracle.adata.obs.groupby("SEACell")["manual_annotation"].agg(lambda x: x.value_counts().idxmax())
    most_prevalent

    # average the 2D embedding and 2D transition vectors across "metacells"
    trans_probs_metacell_WT = compute_metacell_transitions(oracle.adata, 
                                                        trans_key="T_fwd_WT_global_nmps", 
                                                        metacell_key="SEACell")

    # Initialize an empty DataFrame with celltypes as the index
    metacells = trans_probs_metacell_WT.index
    euclid_dist_df = pd.DataFrame(index=metacells)
    
    # Compute cosine similarities for each gene knockout
    for gene in tqdm(oracle.active_regulatory_genes, desc=f"Processing Genes for {data_id}"):
        # Compute transition probabilities for the current gene knockout
        trans_key = f"T_fwd_{gene}_KO"
        trans_probs_metacell_KO = compute_metacell_transitions(oracle.adata, trans_key=trans_key, 
                                                                metacell_key="SEACell")
        
        # Compute cosine similarities
        euclidean_distances = compute_row_euclidean_dist(trans_probs_metacell_WT, trans_probs_metacell_KO)
        
        # Add the cosine similarities as a new column to the DataFrame
        euclid_dist_df[gene] = euclidean_distances

    # Display the resulting DataFrame (metacell-by-genes)
    euclid_dist_df["celltype"] = euclid_dist_df.index.map(most_prevalent)

    # average the cosine similarities across cell types
    euclid_dist_df_avg = euclid_dist_df.groupby("celltype").mean()
    df_averaged = euclid_dist_df_avg.reset_index()
    
    # save the dataframes
    euclid_dist_df.to_csv(oracle_path + f"{data_id}/euclidean_dist_df_metacells_{data_id}.csv")
    df_averaged.to_csv(oracle_path + f"{data_id}/euclidean_dist_df_averaged_{data_id}.csv")
    
    # save this into the master dictionary (dict)
    dict_euclid_dist[data_id] = df_averaged
    # save the raw version (metacells-by-genes)
    dict_euclid_dist[f"{data_id}_metacells"] = euclid_dist_df
    
    print(f"{data_id} is completed")

# %%
# # import the dataframes
# oracle_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/09_NMPs_subsetted_v2/"
# list_datasets = ['TDR126','TDR127','TDR128','TDR118',
#                  'TDR119','TDR125','TDR124']

# dict_cos_sims = {}

# for data_id in list_datasets:
#     # cosine_sim_df = pd.read_csv(oracle_path + f"{data_id}/cosine_similarity_df_metacells_{data_id}.csv")
#     df_averaged = pd.read_csv(oracle_path + f"{data_id}/euclidean_dist_df_averaged_{data_id}.csv", index_col=0)

#     # save this into the master dictionary (dict)
#     dict_euclid_dist[data_id] = df_averaged


# %%
dict_euclid_dist["TDR126"]

# %%
dict_marker_genes = {"TDR126":["ved","vox","hmga1a","cdx4"],
                     "TDR127":["cdx4","tbx16","sox5"],
                     "TDR128":["meis2a","pax6a","dmrt2a","mef2d","meis1a"],
                     "TDR118":["hoxb3a","gli2a"],
                     "TDR119":["hoxb3a","gli2a"],
                     "TDR125":["hoxb3a","meis1a","meis1b","rarga","hoxc3a","sox5","hoxc6b"],
                     "TDR124":["pax6a","hoxa4a","myog","myod1","foxd3","mef2d"]}

# %%
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/in_silico_KO_NMPs_subsets_metacells/euclid_dist/"
os.makedirs(figpath, exist_ok=True)

# %%
list_datasets


for data_id in list_datasets:
    # subset the mesoderm celltypes
    df_meso = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["PSM","fast_muscle","somites"])]
    # compute the median across celltypes
    df_meso_avg = df_meso.median(axis=0)
    # compute the perturbation score (1-cos.similarity)
    df_meso_avg = 1 - df_meso_avg
    # df_meso_avg.sort_values(ascending=False)

# subset the neuro-ectoderm celltypes
    df_ne = dict_cos_sims[data_id][dict_cos_sims[data_id].celltype.isin(["neural_posterior","spinal_cord"])]
    # compute the median across celltypes
    df_ne_avg = df_ne.median(axis=0)
    # compute the perturbation score (1-cos.similarity)
    df_ne_avg = 1 - df_ne_avg
    # df_ne_avg.sort_values(ascending=False)
    
    df_merged = df_meso_avg.to_frame(name="meso").join(df_ne_avg.to_frame(name="ne"))
    df_merged
    
    # Find top 5 genes for each axis
    # top_meso = df_merged.nlargest(5, 'meso')
    # top_ne = df_merged.nlargest(5, 'ne')

    # Combine top genes (in case there's overlap)
    # top_genes = pd.concat([top_meso, top_ne]).drop_duplicates()
    top_genes = dict_marker_genes[data_id]
    # Create a color palette with as many colors as there are genes to annotate
    gene_colors = sns.color_palette("Set2", len(top_genes))
    
    # Annotate the selected marker genes
    # for gene in top_genes:
    #     if gene in df_merged.index:  # Ensure the gene exists in the merged dataframe
    #         row = df_merged.loc[gene]
    #         plt.annotate(gene, (row['meso'], row['ne']), 
    #                      xytext=(5, 5), textcoords='offset points', 
    #                      ha='left', va='bottom',
    #                      bbox=dict(boxstyle='round,pad=0.5', alpha=0),
    #                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.scatter(df_merged["meso"], df_merged["ne"])
    
    # plt.scatter(df_merged["meso"], df_merged["ne"], color='lightgray', alpha=0.7, label="Other Genes")
    
    # # Color the points corresponding to the annotated genes
    # for i, gene in enumerate(top_genes):
    #     if gene in df_merged.index:  # Ensure the gene exists in the merged dataframe
    #         row = df_merged.loc[gene]
    #         # Scatter plot for each gene
    #         plt.scatter(row['meso'], row['ne'], color=gene_colors[i], label=gene, s=100, edgecolor='black')
    #         # Annotate the gene (remove the arrow, match the text color to the dot color)
    #         plt.annotate(gene, (row['meso'], row['ne']), 
    #                      xytext=(5, 5), textcoords='offset points', 
    #                      ha='left', va='bottom',
    #                      color=gene_colors[i],  # Set annotation text color to match dot color
    #                      bbox=dict(boxstyle='round,pad=0.5', alpha=0))  # No arrow props
        # Color the points corresponding to the annotated genes with smaller dots
    for i, gene in enumerate(top_genes):
        if gene in df_merged.index:  # Ensure the gene exists in the merged dataframe
            row = df_merged.loc[gene]
            # Scatter plot for each gene with smaller dots
            plt.scatter(row['meso'], row['ne'], color=gene_colors[i], label=gene, s=70, edgecolor='black')  # Smaller dots (s=70)
            
            # Dynamically adjust the text position to avoid overlap
            x_offset = 5 if i % 2 == 0 else -5  # Alternate x offset
            y_offset = 5 + (i % 3) * 5  # Vary y offset based on index
            
            # Annotate the gene (remove the arrow, match the text color to the dot color)
            plt.annotate(gene, (row['meso'], row['ne']), 
                         xytext=(x_offset, y_offset), textcoords='offset points', 
                         ha='left' if i % 2 == 0 else 'right', va='bottom',
                         color=gene_colors[i],  # Set annotation text color to match dot color
                         bbox=dict(boxstyle='round,pad=0.5', alpha=0))  # No arrow props
    
    

    plt.xlabel("mesoderm")
    plt.ylabel("neuro-ectoderm")
    plt.xlim([0.4, 0.85])
    plt.ylim([0.4, 0.65])
    plt.title(f"{data_id}")
    plt.grid(False)
    plt.savefig(figpath+f"scatter_cosine_sim_{data_id}.pdf")
    plt.show()

# %%
