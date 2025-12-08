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
#     display_name: seacells
#     language: python
#     name: seacells
# ---

# %% [markdown]
# ## [demo] aggregate the count matrices over metacells
#
# - SEACElls.core.summarize_by_SEACell(ad, SEACells_label="SEACell", celltype_label=None, summarize_layer="raw")
#
# - We will turn this notebook into a function so that we can call from a module

# %%
# import libraries
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import SEACells
import os

# plotting modules
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# %%
import sys
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/SEACells_metacell/")
from module_compute_seacells import * # import all functions

# Import from fig2_utils modules (refactored utilities)
from scripts.fig2_utils.metacell_utils import (
    aggregate_counts_multiome,
    compute_prevalent_celltype_per_metacell
)

# %%
# Some plotting aesthetics
# %matplotlib inline
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 300

# %%
# import logging

# # Set the logging level to WARN, filtering out informational messages
# logging.getLogger().setLevel(logging.WARNING)

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# # Set the default font to Arial
# mpl.rcParams['font.family'] = 'Arial'

# # If Arial is not available on your system, you might need to specify an alternative or ensure Arial is installed.
# # On some systems, you might need to use 'font.sans-serif' as a fallback option:
# # mpl.rcParams['font.sans-serif'] = 'Arial'

# # Editable text and proper LaTeX fonts in illustrator
# # matplotlib.rcParams['ps.useafm'] = True
# # Editable fonts. 42 is the magic number for editable text in PDFs
# mpl.rcParams['pdf.fonttype'] = 42
# sns.set(style='whitegrid', context='paper')

# # Plotting style function (run this before plotting the final figure)
# def set_plotting_style():
#     plt.style.use('seaborn-paper')
#     plt.rc('axes', labelsize=12)
#     plt.rc('axes', titlesize=12)
#     plt.rc('xtick', labelsize=10)
#     plt.rc('ytick', labelsize=10)
#     plt.rc('legend', fontsize=10)
#     plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
#     plt.rc('xtick.major', pad=2)
#     plt.rc('ytick.major', pad=2)
#     plt.rc('mathtext', fontset='stixsans', sf='sansserif')
#     plt.rc('figure', figsize=[10,9])
#     plt.rc('svg', fonttype='none')

#     # Override any previously set font settings to ensure Arial is used
#     plt.rc('font', family='Arial')

# %% [markdown]
# ### start with one object, then make a for loop

# %%
#plot_2D_modified arguments (filepaths, dim.reduction, annotation, etc.)
input_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_75cells_per_metacell_integrated_lsi/"
output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_30cells_per_metacell/"
data_id = "" # this will be looped into the for loop
annotation_class = "annotation_ML_coarse"
dim_reduction = "X_lsi"
#figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/seacells_individual_datasets_30cells/"
#os.makedirs(figpath, exist_ok=True)
# add the n_cells (for the number of cells per SEACells)
n_cells = 30

# %%
# the list of datasets
list_datasets = ["TDR126","TDR127","TDR128",
                 "TDR118reseq","TDR119reseq","TDR125reseq","TDR124reseq"]


# %% [markdown]
# **Note:** `aggregate_counts_multiome()` is now imported from scripts.fig2_utils.metacell_utils

# %% [markdown]
# ## NOTES
#
# The correlation between RNA and ATAC (gene.activity) was computed by the following steps:
#
# - 1) import integrated RNA/ATAC objects, respectively (ATAC has gene.activity)
# - 2) subset for each timepoint, import the SEACell information (metacells), then aggregate the counts (RNA/gene.activity)
# - 3) compute the pearson correlation coefficients between 

# %% [markdown]
# ## Step 1. Load the datasets and preprocess them

# %%
# Load the RNA and ATAC(gene.activity) master object, then subset for each timepoint/dataset
filepath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/'

# RNA master object (all timepoints/datasets)
adata_RNA = sc.read_h5ad(filepath + "integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
print(adata_RNA)

# ATAC (gene.acitivitiy) master object (all timepoints/datasets)
adata_ATAC = sc.read_h5ad(filepath + "integrated_RNA_ATAC_counts_gene_activity_raw_counts_master_filtered.h5ad")
print(adata_ATAC)

# %%
adata_rna = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")
adata_rna

# %%
adata_atac = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_gene_activity_raw_counts_master_filtered.h5ad")
adata_atac

# %%
adata_atac.obs_names.map(adata_rna.obs["annotation_ML_coarse"])

# %%
# transfer some annotations
adata_atac.obs["annotation_ML_coarse"] = adata_atac.obs_names.map(adata_rna.obs["annotation_ML_coarse"])
adata_atac.uns['annotation_ML_coarse_colors'] = adata_rna.uns['annotation_ML_coarse_colors']


# %% [markdown]
# ### test with one dataset

# %%
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"

# subset for individual dataset, and edit the adata.obs_names (to remove the extract "XXX_1")
data_id = "TDR118reseq"
sample_id = data_id.replace("reseq","")

# subset for each "dataset"
rna_ad = adata_rna[adata_rna.obs.dataset==sample_id]
atac_ad = adata_atac[adata_atac.obs.dataset==sample_id]

# reformat the adata.obs_names (to remove the additional index from f"XXXX_{index}")
rna_ad.obs_names = rna_ad.obs_names.str.rsplit('_', n=1).str[0]
rna_ad.obs_names

atac_ad.obs_names = atac_ad.obs_names.str.rsplit('_', n=1).str[0]
atac_ad.obs_names

# import the individual ATAC data to copy "X_lsi" ("X_svd") in adata.obsm field
# since SEACell aggregation requires "X_svd" embedding, we will add it (it's basically the same as X_lsi in our pipeline)
lsi = pd.read_csv(fiblepath + f"{data_id}/{sample_id}_lsi.csv", index_col=0)
lsi = lsi[lsi.index.isin(atac_ad.obs_names)]
atac_ad.obsm["X_svd"] = lsi.values

# make sure that the "adata.X" is the raw counts
rna_ad.X = rna_ad.layers["counts"].copy()
atac_ad.X = atac_ad.layers["counts"].copy()

print(rna_ad)
print(atac_ad)

# %%
sc.pl.embedding(atac_ad, basis="X_svd", color="annotation_ML_coarse")

# %%
rna_ad.var_names.intersection(atac_ad.var_names)

# %%
# Load the SEACells (adata.obs)
seacellpath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_30cells_per_metacell/'
# data_id = "TDR118reseq"
# sample_id = data_id.replace("reseq","")

# import the seacell metadata (dataframe)
df_seacells = pd.read_csv(seacellpath + f"{sample_id}_seacells.csv", index_col=0)
df_seacells

# create a dictionary of {"cell_id":"SEACell"}
dict_seacells = df_seacells["SEACell"].to_dict()

# transfer the "SEACell" information to the RNA and ATAC adata objects
rna_ad.obs["SEACell"] = rna_ad.obs_names.map(dict_seacells)
atac_ad.obs["SEACell"] = atac_ad.obs_names.map(dict_seacells)

# %%
rna_ad.X.todense()

# %%
atac_ad.X.todense()

# %%
# add a placeholder value of 0.5 for atac_ad_seacells.var["GC"] field 
# this is for SEACElls operations
atac_ad.var["GC"] = 0.5
atac_ad.var["GC"]

# %%
# # Compute the GC content of peaks for SEACells
# # SEACells.genescore.prepare_multiome_anndata()
# # source: https://github.com/dpeerlab/SEACells/issues/12

# import numpy as np
# import anndata as ad

# import genomepy
# from Bio.SeqUtils import gc_fraction

# # inputs:
# # 1) adata_atac: anndata for the scATAC-seq (cells-by-peaks)
# # output:
# def compute_GC_contents_scATAC(adata_atac, genome_name, provider):

#     # download genome from NCBI
#     genome_name = "GRCz11"
#     provider = "Ensembl"
#     genomepy.install_genome(name=genome_name, provider=provider, genome_dir = "../data")
#     genome = genomepy.Genome(name=genome_name, genome_dir="../data")
#     #genomepy.install_genome(name='GRCh38', provider='NCBI', genomes_dir = ''../data') # took about 9 min
#     #genome = genomepy.Genome(name = 'GRCh38', genomes_dir = '../data')

#     GC_content_list = []

#     for i, region in enumerate(adata_atac.var_names):
#         chromosome, start, end = region.split('-')
#         chromosome = chromosome[3:]

#         # get sequence
#         sequence = str(genome.get_seq(name = chromosome, start = int(start), end = int(end)))

#         # calculate GC content
#         GC_content = gc_fraction(str(sequence))
#         GC_content_list.append(GC_content)

#     # GC content ranges from 0% - 100%, should be 0 to 1
#     adata_atac.var['GC'] = GC_content_list
#     adata_atac.var.GC = adata_atac.var.GC/100

#     return adata_atac

# %%
# # To find the right genome name and provider
# for provided_genome in genomepy.search('GRCz11', provider=None):
#    print(provided_genome)

# # result:
# # ['GRCz11', 'Ensembl', 'GCA_000002035.4', 7955, True, 'Danio rerio', '2017-08-Ensembl/2018-04']
# # ['danRer11', 'UCSC', 'GCA_000002035.4', 7955, [True, True, True, False], 'Danio rerio', 'May 2017 (GRCz11/danRer11)']

# %%
# atac_ad_seacells = compute_GC_contents_scATAC(atac_ad_seacells, "GRCz11", "Ensembl")


# %% [markdown]
# # Preparation step
#

# %% [markdown]
# In the first step, we derive summarized ATAC and RNA SEACell metacells Anndata objects. Both the input single-cell RNA and ATAC anndata objects should contain raw, unnormalized data. SEACell results on ATAC data will be used for the summarization
#
# <b>Warning: </b> The ATAC and RNA single-cell Anndata objects should contain the same set of cells. Only the common cells will be used for downstream analyses.

# %%
print(rna_ad.shape[1])
print(atac_ad.shape[1])

# %%
atac_ad

# %%
atac_meta_ad, rna_meta_ad = SEACells.genescores.prepare_multiome_anndata(atac_ad, rna_ad, SEACells_label='SEACell')

# %% [markdown]
# The preparation step will generate summarized anndata objects for RNA and ATAC

# %%
atac_meta_ad

# %%
rna_meta_ad

# %% [markdown]
# ### summary
#
# rna_meta_ad, and atac_meta_ad are "aggregated" across metacells (SEACell), and the raw counts were saved in "adata.raw" slot, and the adata.X contains log-normalized (using sc.pp.normalize_total, which takes the median # of UMIs/cell across all cells in that dataset as the denominator).

# %%
sample_id

# %%
# save the aggregated adata objects
atac_meta_ad.write_h5ad(seacellpath + f"{data_id}/{sample_id}_ATAC_seacells_aggre.h5ad")
rna_meta_ad.write_h5ad(seacellpath + f"{data_id}/{sample_id}_RNA_seacells_aggre.h5ad")

# %%

# %%

# %% [markdown]
# ## make a for loop to aggreage the count matrices for each modality (rna, atac)
#
# - save the aggregated counts into metacells-by-features objects (rna_meta_ad, atac_meta_ad), with the cell-type annotation for metacell.

# %%
# reset the colors
# a color palette for the "coarse" grained celltype annotation ("annotation_ML_coarse")
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'differentiating_neurons': '#bebada',
    'endocrine_pancreas': '#fb8072',
    'endoderm': '#80b1d3',
    'enteric_neurons': '#fdb462',
    'epidermis': '#b3de69',
    'fast_muscle': '#df4b9b',
    'floor_plate': '#d9d9d9',
    'hatching_gland': '#bc80bd',
    'heart_myocardium': '#ccebc5',
    'hemangioblasts': '#ffed6f',
    'hematopoietic_vasculature': '#e41a1c',
    'hindbrain': '#377eb8',
    'lateral_plate_mesoderm': '#4daf4a',
    'midbrain_hindbrain_boundary': '#984ea3',
    'muscle': '#ff7f00',
    'neural': '#e6ab02',
    'neural_crest': '#a65628',
    'neural_floor_plate': '#66a61e',
    'neural_optic': '#999999',
    'neural_posterior': '#393b7f',
    'neural_telencephalon': '#fdcdac',
    'neurons': '#cbd5e8',
    'notochord': '#f4cae4',
    'optic_cup': '#c0c000',
    'pharyngeal_arches': '#fff2ae',
    'primordial_germ_cells': '#f1e2cc',
    'pronephros': '#cccccc',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}


# %% [markdown]
# **Note:** `compute_prevalent_celltype_per_metacell()` is now imported from scripts.fig2_utils.metacell_utils

# %%
filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"

output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_30cells_per_metacell/aggregated_counts/"
os.makedirs(output_path, exist_ok=True)

# Load the SEACells (adata.obs)
seacellpath = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/05_SEACells_processed/objects_30cells_per_metacell/'


# %%
# define the list of datasets
list_datasets = ["TDR126","TDR127","TDR128","TDR118reseq","TDR125reseq","TDR124reseq"]

# %%
for data_id in list_datasets:
    # subset for individual dataset, and edit the adata.obs_names (to remove the extract "XXX_1")
    # data_id = "TDR118reseq"
    sample_id = data_id.replace("reseq","")

    # subset for each "dataset"
    rna_ad = adata_rna[adata_rna.obs.dataset==sample_id]
    atac_ad = adata_atac[adata_atac.obs.dataset==sample_id]

    # reformat the adata.obs_names (to remove the additional index from f"XXXX_{index}")
    rna_ad.obs_names = rna_ad.obs_names.str.rsplit('_', n=1).str[0]
    rna_ad.obs_names

    atac_ad.obs_names = atac_ad.obs_names.str.rsplit('_', n=1).str[0]
    atac_ad.obs_names

    # import the individual ATAC data to copy "X_lsi" ("X_svd") in adata.obsm field
    # since SEACell aggregation requires "X_svd" embedding, we will add it (it's basically the same as X_lsi in our pipeline)
    lsi = pd.read_csv(filepath + f"{data_id}/{sample_id}_lsi.csv", index_col=0)
    lsi = lsi[lsi.index.isin(atac_ad.obs_names)]
    atac_ad.obsm["X_svd"] = lsi.values

    # make sure that the "adata.X" is the raw counts
    rna_ad.X = rna_ad.layers["counts"].copy()
    atac_ad.X = atac_ad.layers["counts"].copy()

    print(rna_ad)
    print(atac_ad)

    # import the seacell metadata (dataframe)
    df_seacells = pd.read_csv(seacellpath + f"{sample_id}_seacells.csv", index_col=0)
    df_seacells

    # create a dictionary of {"cell_id":"SEACell"}
    dict_seacells = df_seacells["SEACell"].to_dict()

    # transfer the "SEACell" information to the RNA and ATAC adata objects
    rna_ad.obs["SEACell"] = rna_ad.obs_names.map(dict_seacells)
    atac_ad.obs["SEACell"] = atac_ad.obs_names.map(dict_seacells)

    # add a placeholder value of 0.5 for atac_ad_seacells.var["GC"] field 
    # this is for SEACElls operations
    atac_ad.var["GC"] = 0.5
    atac_ad.var["GC"]

    # aggregate the raw counts (saves the normalized counts in the .X layer, and saves the "raw" layer for the raw counts for each metacell
    atac_meta_ad, rna_meta_ad = SEACells.genescores.prepare_multiome_anndata(atac_ad, rna_ad, SEACells_label='SEACell')

    # copy over the most prevalent celltype annotations
    prevalent_celltypes = compute_prevalent_celltype_per_metacell(rna_ad, celltype_key="annotation_ML_coarse", 
                                                                  metacell_key="SEACell")

    # Add the most prevalent cell type to the metacell dataframe
    atac_meta_ad.obs["celltype"] = atac_meta_ad.obs_names.map(prevalent_celltypes)
    rna_meta_ad.obs["celltype"] = rna_meta_ad.obs_names.map(prevalent_celltypes)

    # save the metacell-aggregated adata objects
    atac_meta_ad.write_h5ad(output_path + f"{sample_id}_atac_meta.h5ad")
    rna_meta_ad.write_h5ad(output_path + f"{sample_id}_rna_meta.h5ad")

# %%
rna_meta_ad

# %%
atac_meta_ad
