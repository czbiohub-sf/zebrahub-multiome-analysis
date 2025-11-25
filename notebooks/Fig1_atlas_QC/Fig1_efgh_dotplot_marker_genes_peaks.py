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
# # Notebook for marker genes/peaks - dot plots, heatmaps, etc.

# %%
import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# %%
# figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')
# Set default DPI for saved figures
mpl.rcParams['savefig.dpi'] = 600

# Import plotting utilities from refactored modules
from scripts.fig1_utils.plotting_utils import set_plotting_style


# %%
import logging
# Suppress INFO-level logs for the entire logger
logging.getLogger().setLevel(logging.WARNING)

# %%
# define the figure path
import os
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/umaps/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# load the master adata object (multiome, all timepoints, low-quality cells were filtered out)
# NOTE that this adata has "RNA" counts in the counts layer
adata = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
adata

# %% [markdown]
# ## Check the known markers from zebrafish biology experts

# %%
markers_15somites = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/table_marker_genes/marker_genes_15somites.csv", index_col=0)
markers_15somites.head()

# %%
markers_15somites.columns

# %%
list_markers = []

for col_name in ["marker genes", "Unnamed: 5", "Unnamed: 6"]:
    list_temp = markers_15somites[col_name].tolist()
    list_markers = list_markers + list_temp

list_unique_markers = np.unique(list_markers)
list_unique_markers

# %%
np.unique(list_markers)

# %%
markers_30somites = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/table_marker_genes/marker_genes_30somites.csv", index_col=0)
markers_30somites.head()

# %%
list_markers = []

for col_name in ["marker genes", "Unnamed: 7", "Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11"]:
    list_temp = markers_30somites[col_name].tolist()
    list_markers = list_markers + list_temp

list_unique_markers_30 = np.unique(list_markers)
list_unique_markers_30

# %%
master_list_markers = list_unique_markers.tolist() + list_unique_markers_30.tolist()
master_list_markers = np.unique(master_list_markers)
master_list_markers

# %%
## NOTE: some genes are missing from the adata.var_names (probably due to the Cellranger-arc mapping isoforms...)
missing_genes = ['acy3', 'apeob', 'chd17', 'hbbe1', 'nan', 'pmp22', 'prrx1', 'ptx3', 'sox2b', 'ttn1']

for gene in missing_genes:
    print("gene ", gene, " is ", gene in (adata.var_names))

# %%
master_list_markers = [gene for gene in master_list_markers if gene not in missing_genes]

# %% [markdown]
# ## Check the annotation levels
# - we will create a dictionary to mape the celll types to broader "tissues".

# %%
adata.obs["annotation_ML_coarse"].unique().to_list()

# %%
adata[adata.obs["annotation_ML_coarse"]=="floor_plate"].obs["annotation_ML"]

# %%
sc.pl.embedding(adata, basis="X_wnn.umap", color="annotation_ML")

# %%
sc.pl.embedding(adata, basis="X_wnn.umap", color="annotation_ML_coarse")

# Import cell type mapping utilities from refactored modules
from scripts.fig1_utils.cell_type_mappings import (
    CELLTYPE_TO_LINEAGE as celltype_to_lineage,
    map_celltype_to_tissue
)


# %%
# Create a new column in adata.obs for tissue groups
adata.obs["tissue"] = adata.obs["annotation_ML"].apply(
    lambda x: map_celltype_to_tissue(x, celltype_to_lineage)
)

# Verify the mapping worked
print(adata.obs["tissue"].value_counts())

# %%
tissue_colors = {
    'CNS': '#DAA520',                    # Golden/orange
    'Endoderm': '#6A5ACD',              # Blue/purple  
    'Epiderm': '#DC143C',               # Red
    'Germline': '#DA70D6',              # Magenta/orchid
    'Lateral Mesoderm': '#228B22',      # Forest green
    'Neural Crest': '#20B2AA',          # Light sea green/teal
    'Paraxial Mesoderm': '#4169E1'      # Royal blue
}

# %%
sc.pl.embedding(adata, basis="X_wnn.umap", color="tissue", 
                palette=celltype_colors, save="_tissue.png")

# %%
# First create your plot
sc.pl.embedding(adata, basis="X_wnn.umap", color="tissue", 
                palette=celltype_colors, show=False)

# Get the current figure and extract legend
fig = plt.gcf()
legend = fig.axes[0].get_legend()

# Create new figure with just the legend
fig_legend = plt.figure(figsize=(3, 4))
fig_legend.legend(legend.legendHandles, [t.get_text() for t in legend.get_texts()], 
                  loc='center', frameon=False)
plt.savefig(figpath + 'tissue_legend_only.pdf', dpi=600, bbox_inches='tight')
plt.show()

# %%
# sc.pl.embedding(adata, basis="X_wnn.umap", color="lineage")

# %%
# Define marker genes for each lineage
marker_genes_by_lineage = {
    'Epidermal': ['tp63', 'krt17', 'cdh1', 'tpma'],
    'Neural': ['sox2', 'sox1b', 'pax6a', 'vsx2', 'elavl3', 'olig2'],
    'Neural Crest': ['sox10', 'foxd3', 'tfap2a', 'mitfa', 'dlx2a'],
    'Mesoderm': ['tbxta', 'tbx16', 'meox1', 'myog', 'myf5', 'myl1'],
    'Hematopoietic/Vascular': ['spi1b', 'fli1b', 'gata1a', 'kdrl'],
    'Endoderm': ['foxa2', 'gata6', 'sox32', 'pdgfra'],
    'Germline': ['nanos3', 'vasa', 'dnd1', 'piwil1']
}

# Flatten the marker gene dictionary to get a list of all markers
all_markers = []
for genes in marker_genes_by_lineage.values():
    all_markers.extend(genes)


# %%
list_marker_genes = []

for gene in all_markers:
    if gene in adata.var_names:
        list_marker_genes.append(gene)
        
list_marker_genes

# %%
sc.pl.dotplot(adata, var_names = list_marker_genes, groupby="lineage")

# %%
# sc.pl.rank_genes_groups_dotplot(adata, n_genes=2)

# %%
sc.tl.rank_genes_groups(adata, groupby="lineage", method = "wilcoxon")

# %%
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5)

# %%
# # Define refined marker genes based on Wilcoxon results and literature
# # refined_marker_genes = {
# #     'Neural': ['sox2', 'neurog1', 'pax6a', 'gli3', 'chl1a'],
# #     'Germline': ['nanos3', 'tdrd7a', 'dnd1'],
# #     'Hematopoietic/Vascular': ['etv2', 'fli1a', 'lmo2', 'tal1'],
# #     'Mesoderm': ['tbx6', 'tbxta', 'msgn1', 'meox1'],
# #     'Endoderm': ['foxa3', 'onecut1', 'nr5a2', 'col2a1a'],
# #     'Epidermal': ['cdh1', 'epcam', 'krt4']
# # }
# refined_marker_genes = {
#     "Endoderm":["fgfrl1b","col2a1a","ptprfa","emid1","nr5a2","ism2a","pawr","mmp15b","foxa3","onecut1"],
#     "Epiderm":["cdh1","epcam","krt4"],
#     "Germline":['nanos3', 'tdrd7a', 'dnd1'],
#     'Hematopoietic/Vascular': ['etv2', 'fli1a', 'lmo2', 'tal1',"sox17"],
#     "Mesoderm":["msgn1","meox1","tbx6","tbxta","fgf8a","her1"],
#     "Neural":["pax6a","pax7a","pax6b","col18a1a", "en2b","znf536","gpm6aa","gli3","chl1a"],
#     # 'Neural': ['sox2', 'neurog1', 'pax6a', 'gli3', 'chl1a'],
#     # 'Germline': ['nanos3', 'tdrd7a', 'dnd1'],
#     # 'Hematopoietic/Vascular': ['etv2', 'fli1a', 'lmo2', 'tal1'],
#     # 'Mesoderm': ['tbx6', 'tbxta', 'msgn1', 'meox1'],
#     # 'Endoderm': ['foxa3', 'onecut1', 'nr5a2', 'col2a1a'],
#     # 'Epidermal': ['cdh1', 'epcam', 'krt4']
# }

# %%
# sc.pl.dotplot(adata, refined_marker_genes, groupby="lineage")

# %% [markdown]
# ## [updated] compute the marker genes from each tissue (wilcoxon)

# %%
sc.tl.rank_genes_groups(adata, groupby="tissue", method = "wilcoxon")

# %%
sc.pl.rank_genes_groups_dotplot(adata, n_genes=10, save="_tissues_wilcoxon.pdf")

# %%
refined_marker_genes = {
    "CNS": ["nova2", "ncam1a", "pax6a"],
    "Endoderm": ["fgfrl1b", "plxna2", "mcama"], #"gata6", "sox17", "sox32", "fgfrl1b", "onecut1", "hnf1ba", "nr5a2", "cdh17"],
    "Epiderm": ["tp63", "epcam", "cdh1"], #, "krt4", "krt8","cldnb", "krtt1c19e", "col17a1b"],
    "Germline": ["nanos3", "dnd1", "tdrd7a"], # "buc", "dazl", "piwil1"],
    "Lateral Mesoderm": ["cdh11", "colec12", "adgra2"],
    "Neural Crest": ["slc1a3a", "ednrab", "prex2", "pdgfra"],
    "Paraxial Mesoderm": ["fn1b","tbx16","msgn1","meox1"],#["myod1", "msgn1", "tbx6", "meox1", "myf5"],
}

# %%
marker_genes_ML = {
    "CNS": ["nova2", "sox2", "pax6b", "sox19a"],
    "Endoderm": ["foxa3", "anxa4"], #"gata6", "sox17", "sox32", "fgfrl1b", "onecut1", "hnf1ba", "nr5a2", "cdh17"],
    "Epiderm": ["tp63", "epcam", "cdh1"], #, "krt4", "krt8","cldnb", "krtt1c19e", "col17a1b"],
    "Germline": ["nanos3", "dnd1", "tdrd7a"], # "buc", "dazl", "piwil1"],
    "Lateral Mesoderm": ["cdh11", "colec12", "adgra2"],
    "Neural Crest": ["sox10", "ednrab", "crestin"],
    "Paraxial Mesoderm": ["tbxta","tbx16","msgn1","meox1"],#["myod1", "msgn1", "tbx6", "meox1", "myf5"],
}

# %%
sc.pl.dotplot(adata, var_names=marker_genes_ML, groupby="tissue")

# %%
# refined_marker_genes = {
#     "CNS": ["neurog1", "neurod4", "neurod1", "gpm6aa", "pax6a", "pax6b", "gli3", "en2b", "epha4a", "rfx4", "chl1a", "znf536"],
    
#     "Neural Crest": ["foxd3", "tfap2a", "sox10", "prdm1a", "id2a", "snai1b"],
    
#     "Paraxial Mesoderm": ["myod1", "msgn1", "tbx6", "meox1", "tbxta", "her1", "ripply1", "myf5"],
    
#     "Endoderm": ["foxa3", "gata6", "sox17", "sox32", "fgfrl1b", "onecut1", "hnf1ba", "nr5a2", "cdh17"],
    
#     "Epiderm": ["tp63", "krt4", "krt8", "epcam", "cdh1", "cldnb", "krtt1c19e", "col17a1b"],
    
#     "Germline": ["ddx4", "nanos3", "dnd1", "tdrd7a", "buc", "dazl", "piwil1"],
    
#     "Lateral Mesoderm": ["spry4", "tbx5a", "slc4a1a"]
# }

# %%
sc.pl.dotplot(adata, var_names = refined_marker_genes, groupby="tissue", save="tissue_selected_markers.pdf")

# %%
# import the gene activity matrix (ATAC)
adata_ga = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_gene_activity.h5ad")
adata_ga

# %%
adata_ga = adata_ga[adata_ga.obs_names.isin(adata.obs_names)]
adata_ga

# %%
adata_ga.obs_names.map(adata.obs["tissue"])

# %%
# # copy over the tissue annotation
adata_ga.obs["tissue"] = adata_ga.obs_names.map(adata.obs["tissue"])
adata_ga

# %%
sc.pl.embedding(adata_ga, basis="X_wnn.umap", color="tissue")

# %%
np.sum(np.expm1(adata_ga.X.todense()),1)

# %%
adata_ga.raw = adata_ga.copy()

# %%
sc.pl.dotplot(adata_ga, var_names = refined_marker_genes, groupby="tissue", save="_gene_activity_tissue_selected_markers.pdf")

# %%
# export the metadata (obs) - with tissue/lineage annotations
adata.obs.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/master_obj_obs.csv")
