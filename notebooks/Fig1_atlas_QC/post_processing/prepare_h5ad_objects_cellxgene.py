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
#     display_name: Global single-cell-base
#     language: python
#     name: global-single-cell-base
# ---

# %% [markdown]
# ## Preparing/Cleaning up the h5ad objects for the Cellxgene instances (web portal)
#
# - last updated: 4/14/2025
# - Yang-Joon Kim (yang-joon.kim@czbiohub.org)
#
# Goals: we clean up the h5ad objects to generate cellxgene instnaces in the Zebrahub web portal. The datasets are the following, and are from "integrated" across all timepoints:
# - (1) RNA object (cells-by-genes), with 3 embeddings (UMAPs from RNA, ATAC, and joint)
# - (2) ATAC object (cells-by-gene.activity), with 3 embeddings
# - (3) joint object (cells-by-genes&gene.activity), with 3 embeddings
# - (4) ATAC object (cells-by-peaks), with ATAC embeddings and annotations (celltype and timepoints)
# - (5) subset the master objects into individual dataset for RNA and ATAC modalities (for downstream analyses)
#

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %% [markdown]
# ### import h5ad objects
#

# %%
adata_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/"

# RNA object
adata_RNA = sc.read_h5ad(adata_path + "integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad")

# ATAC object
adata_ATAC = sc.read_h5ad(adata_path + "integrated_RNA_ATAC_counts_gene_activity_raw_counts_master_filtered.h5ad")


# %% [markdown]
# ### (1) RNA object

# %%
adata_RNA

# %%
adata_RNA

# %%
print(np.median(adata_RNA.obs.n_genes_by_counts))
print(np.median(adata_RNA.obs.total_counts))

# %%
adata_ATAC

# %%
print(np.median(adata_ATAC.obs.n_peaks_by_counts_ATAC))
print(np.median(adata_ATAC.obs.n_counts_ATAC))

# %%
# Step 1. define the fields to keep
fields_RNA = ["dev_stage", "dataset", "annotation_ML", "annotation_ML_coarse"]

# clean up the obs fields (and also uns slot), and re-name them
adata_RNA.obs = adata_RNA.obs[fields_RNA]
adata_RNA

# rename the obsm fields
dict_names = {"dev_stage":"developmental_stage",
              "dataset":"dataset",
              "annotation_ML_coarse":"zebrafish_anatomy_ontology_class_coarse",
              "annotation_ML":"zebrafish_anatomy_ontology_class"}

# Step 2. Rename the columns in adata_RNA.obs using the dictionary
adata_RNA.obs = adata_RNA.obs.rename(columns=dict_names)

# Dictionary mapping dev_stage to real time (hpf)
dev_stage_to_hpf = {
    '0somites': '10hpf',
    '5somites': '12hpf',
    '10somites': '14hpf',
    '15somites': '16hpf',
    '20somites': '19hpf',
    '30somites': '24hpf'
}

adata_RNA.obs["timepoint"] = adata_RNA.obs.developmental_stage.map(dev_stage_to_hpf)
adata_RNA.obs.head()

# %%
adata_RNA

# %%
# Step 3: Rename obsm fields
obsm_rename_dict = {'X_umap.atac': 'X_umap_atac', 
                    'X_umap.rna': 'X_umap_rna', 
                    'X_wnn.umap': 'X_umap_joint'}

for old_key, new_key in obsm_rename_dict.items():
    adata_RNA.obsm[new_key] = adata_RNA.obsm.pop(old_key)

# %%
# Dictionary for renaming the relevant fields in adata_RNA.uns
uns_dict_names = {
    "annotation_ML_coarse_colors": "zebrafish_anatomy_ontology_class_coarse_colors",
    "annotation_ML_colors": "zebrafish_anatomy_ontology_class_colors",
    "dataset_colors": "dataset_colors"
}

# Step 2: Remove all fields from adata_RNA.uns except the first three and rename them
# Create a new dictionary with only the fields to keep
adata_RNA.uns = {new_key: adata_RNA.uns.pop(old_key) for old_key, new_key in uns_dict_names.items() if old_key in adata_RNA.uns}

# Step 3: Check the resulting adata_RNA.uns
adata_RNA

# %%
adata_RNA.var_names

# %%
# annotate the group of mitochondrial genes as "mt"m or "NC"
adata_RNA.var["mt"] = adata_RNA.var_names.str.startswith("mt-")
adata_RNA.var["nc"] = adata_RNA.var_names.str.startswith("NC-")

# recompute the QC metrics (for raw counts)
adata_RNA.X = adata_RNA.layers["counts"].copy()

sc.pp.calculate_qc_metrics(
    adata_RNA, qc_vars=["mt","nc"], percent_top=None, log1p=False, inplace=True
)

sc.pp.normalize_total(adata_RNA, target_sum=1e4)
sc.pp.log1p(adata_RNA)

# %%
sc.pl.embedding(adata_RNA, basis="X_umap_joint", color=["zebrafish_anatomy_ontology_class_coarse","meox1"])

# %%
adata_RNA

# %%
# save the object
adata_RNA.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/zebrahub_multiome_v1_release/zf_multiome_atlas_full_RNA_v1_release.h5ad")

# %% [markdown]
# ### (2) ATAC object

# %%
adata_ATAC

# %%
# # copy over the fields from adata_RNA
fields_to_copy = ['developmental_stage', 'zebrafish_anatomy_ontology_class', 'zebrafish_anatomy_ontology_class_coarse']

# Copy over the fields from adata_RNA to adata_ATAC
cell_ids = adata_RNA.obs_names
for field in fields_to_copy:
    adata_ATAC.obs.loc[cell_ids, field] = adata_RNA.obs.loc[cell_ids, field]

    
adata_ATAC

# %%
# Step 1. define the fields to keep
fields_ATAC = ['nucleosome_signal', 'nucleosome_percentile', 
               'TSS_enrichment', 'TSS_percentile',
               'nCount_peaks_integrated', 'nFeature_peaks_integrated',
               'developmental_stage', 'zebrafish_anatomy_ontology_class', 
               'zebrafish_anatomy_ontology_class_coarse']

# clean up the obs fields (and also uns slot), and re-name them
adata_ATAC.obs = adata_ATAC.obs[fields_ATAC]
adata_ATAC

# rename the obs fields
dict_names = {"nCount_peaks_integrated":"n_counts_ATAC",
              "nFeature_peaks_integrated":"n_peaks_by_counts_ATAC"}

# Step 2. Rename the columns in adata_RNA.obs using the dictionary
adata_ATAC.obs = adata_ATAC.obs.rename(columns=dict_names)

# Dictionary mapping dev_stage to real time (hpf)
dev_stage_to_hpf = {
    '0somites': '10hpf',
    '5somites': '12hpf',
    '10somites': '14hpf',
    '15somites': '16hpf',
    '20somites': '19hpf',
    '30somites': '24hpf'
}

adata_ATAC.obs["timepoint"] = adata_ATAC.obs.developmental_stage.map(dev_stage_to_hpf)
adata_ATAC.obs.head()

# %%
# Step 3: Rename obsm fields
obsm_rename_dict = {'X_umap.atac': 'X_umap_atac', 
                    'X_umap.rna': 'X_umap_rna', 
                    'X_wnn.umap': 'X_umap_joint'}

for old_key, new_key in obsm_rename_dict.items():
    adata_ATAC.obsm[new_key] = adata_ATAC.obsm.pop(old_key)
    
del adata_ATAC.obsm["X_umap"]

adata_ATAC

# %%
adata_ATAC.var_names

# %%
# annotate the group of mitochondrial genes as "mt"m or "NC"
adata_ATAC.var["mt"] = adata_ATAC.var_names.str.startswith("mt-")
# adata_ATAC.var["nc"] = adata_ATAC.var_names.str.startswith("NC-")

# recompute the QC metrics (for raw counts)
adata_ATAC.X = adata_ATAC.layers["counts"].copy()

sc.pp.calculate_qc_metrics(
    adata_ATAC, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
sc.pp.log1p(adata_ATAC)



# %%
sc.pl.embedding(adata_ATAC, basis="X_umap_joint", color=["zebrafish_anatomy_ontology_class_coarse","meox1"])

# %%
# save the adata_ATAC
adata_ATAC.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/zebrahub_multiome_v1_release/zf_multiome_atlas_full_ATAC_gene_activity_v1_release.h5ad")

# %% [markdown]
# ### (3) combining RNA and ATAC objects -
#
# - need to change the var_names ("-RNA", or "-ATAC")
# - concatenate the count matrices
# - keep the obs fields as the union
# - keep the obsm as the same
#

# %%
# resumption from here - loading the h5ad objects cleaned up in the above code cells
# RNA object
adata_RNA = sc.read_h5ad(adata_path + "zebrahub_multiome_v1_release/zf_multiome_atlas_full_RNA_v1_release.h5ad")

# ATAC object
adata_ATAC = sc.read_h5ad(adata_path + "zebrahub_multiome_v1_release/zf_multiome_atlas_full_ATAC_gene_activity_v1_release.h5ad")


# %%
adata_RNA

# %%
adata_ATAC

# %%
# change the var names in adata_RNA
adata_RNA.var_names = adata_RNA.var_names + "-RNA"
adata_RNA.var_names

# change the var names in adata_ATAC_genes
adata_ATAC.var_names = adata_ATAC.var_names + "-ATAC"
adata_ATAC.var_names

# %%
adata_RNA

# %%
import anndata as ad

# Put the assay name for adata.var just in case we need to differentiate where the marker comes from (usually it shoudl be obvious!)
adata_RNA.var['assay'] = 'RNA'
adata_ATAC.var['assay'] = 'ATAC'

# NOTE: the order of the adata.obs should matter in terms of concatenation (which is weird), 
# so we'd need to sort the orders of both adata objects before the concatenation (done in the previous step)

# concatenate the two adata objects
adata_joint = ad.concat([adata_RNA, adata_ATAC], axis=1, merge="first")

adata_joint

# %%
fields_to_remove = ['n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt', 'total_counts_nc', 'pct_counts_nc', 'nucleosome_signal', 'nucleosome_percentile', 'TSS_enrichment', 'TSS_percentile', 'n_counts_ATAC', 'n_peaks_by_counts_ATAC']
# Remove the fields from adata.obs if they exist
adata_joint.obs = adata_joint.obs.drop(columns=[field for field in fields_to_remove if field in adata_joint.obs.columns])

adata_joint

# %%
sc.pl.embedding(adata_joint, basis="X_umap_joint", color=["timepoint","zebrafish_anatomy_ontology_class_coarse"])

# %%
adata_joint.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/zebrahub_multiome_v1_release/zf_multiome_atlas_full_RNA_ATAC_concat_v1_release.h5ad")

# %%

# %% [markdown]
# ## Part 2. subset the integrated object into individual objects
#
# - For RNA, we'd want to make sure that the embeddings are from individual objects, not from the "integrated" object
# - For ATAC, we'd like to keep the embeddings (PC, LSI, and UMAPs) from individual objects, and just want to replace the count matrix to "gene.activity"

# %%

# %%
counts_ga = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/gene_activity_counts.csv", index_col=0)
counts_ga

# %%
adata_ga = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/TDR118reseq/TDR118_processed_peaks_merged.h5ad")
adata_ga

# %%
