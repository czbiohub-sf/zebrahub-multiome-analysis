# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: litemind_env
#     language: python
#     name: litemind_env
# ---

# %% [markdown]
# ## Exploration with litemind for a structured query for regulatory programs
#
# - last updated: 6/24/2025
#
# - DESCRIBE the goals here:
#     - 1) preprocess the structured input from the peak UMAP clusters for LLM queries (this should be scripted in the future)
#     - 2) Perform structured "query" using litemind (ChatGPT API, etc.)
#     - 3) (To-Do) Export the input query into a markdown file

# %%
# define the openAI API key
import os
# # Set the API key BEFORE importing and using litemind
# os.environ["OPENAI_API_KEY"] = ""
# For OpenAI
openai_api_key = os.environ.get('OPENAI_API_KEY')
print(f"OpenAI API key available: {'Yes' if openai_api_key else 'No'}")

# %%
# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from litemind import OpenAIApi

# Import utility functions from the new module
from module_litemind_query import (
    write_to_markdown,
    estimate_tokens,
    get_table_info,
    reduce_table_size,
    summarize_large_table,
    convert_clusters_genes_to_lists,
    process_cluster_data,
    check_model_feature,
    find_models_with_feature,
    get_best_model_for_analysis,
    print_model_capabilities
)

# %%
# load the data
# import the datasets
# (1) peak clusters-by-pseudobulk groups
df_clusters_groups = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_pseudobulk.csv",
                                 index_col=0)
# (2) peak clusters-by-associated genes
df_clusters_genes = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_assoc_genes.csv",
                                index_col=0)
# (3) peak clusters-by-TF motifs
df_clusters_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_motifs_maelstrom.csv",
                                 index_col=0)
# (4) motifs-by-factors
df_motif_info = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors.csv", index_col=0)

# parse the genes for each cluster (to reduce the size of the input table tokens)
cluster_genes_dict = convert_clusters_genes_to_lists(df_clusters_genes, method='nonzero')

# %%
# %% Create a report for each peak cluster
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

from pydantic import BaseModel
from typing import List, Optional
# Create or clear the markdown file
with open("peak_clusters_report.md", "w", encoding="utf-8") as f:
    f.write("# Peak Clusters Report\n\n")

# Check the size of input tables
print("=== INPUT TABLE ANALYSIS ===")
get_table_info(df_clusters_groups, "df_clusters_groups")
# get_table_info(cluster_genes_dict, "df_clusters_genes") 
get_table_info(df_clusters_motifs, "df_clusters_motifs")
get_table_info(df_motif_info, "df_motif_info")


for cluster_id in df_clusters_groups.index:
    print(f"\n=== Processing Cluster {cluster_id} ===")
    
    # Subset the data for the current cluster (cluster_id)
    df_clusters_groups_cluster, genes_text, df_clusters_motifs_cluster, cluster_tokens = process_cluster_data(
        cluster_id, df_clusters_groups, cluster_genes_dict, df_clusters_motifs, df_motif_info
    )
    print(f"Estimated tokens for cluster {cluster_id}: {cluster_tokens}")
    
    # Skip if too large (adjust threshold as needed)
    if cluster_tokens > 100000:  # Adjust this threshold
        print(f"Skipping cluster {cluster_id} - too large ({cluster_tokens} tokens)")
        continue


    # Initialize the OpenAI API
    api = OpenAIApi()
    # Create the agent, passing the toolset
    agent = Agent(api=api)
    # Add a system message
    agent.append_system_message("You are an expert developmental biologist specialized in vertebrate embryogenesis, especially zebrafish embryogenesis.")

    # Section 1: Context for the project
    message = Message()
    message.append_text("Here is the background for the project:")
    message.append_text("""
    - We are working on data from a study that generated a time-resolved single-cell multi-omic atlas of zebrafish embryogenesis by simultaneously profiling chromatin accessibility (scATAC-seq) and gene expression (scRNA-seq). 
    - Over 94,000 cells were sampled from pooled zebrafish embryos at six key developmental time points between 10 and 24 hours post-fertilization (hpf), covering the transition from gastrulation to early organogenesis. 
    - Single-nuclei dissociation was performed, followed by library preparation using the 10x Genomics Chromium Single Cell Multiome ATAC + Gene expression reagents, and sequencing was performed on the NovaSeq 6000 system. 
    - The resulting dataset includes paired chromatin accessibility (median 15,000 peaks per cell) and gene expression data (median 1,400 genes and 3,700 UMIs per cell) for each cell, allowing for integrated analysis of regulatory dynamics.
    """)
    message.append_text("Here is the context for the peak cluster annotation:")
    message.append_text("""
    - For scATAC-seq data, cells-by-peaks, we pseudo-bulked using celltype and timepoint as two identifiers (“celltype & timepoint” as columns).
    - The output, peaks-by-pseudobulk, was further processed for dimensionality reduction, UMAP, which gives the peak UMAP, with leiden clusters for "peaks" with similar accessibility patterns across cell types and timepoints.
    - The specific question that I have is to interpret/annotate each peak cluster from this. I’ll provide the following list of information as inputs from each peak cluster:
    - The peak cluster is annotated based on the peak accessibility data over cell types and timepoints.
    - The peak cluster is also annotated with the associated genes.
    - The peak cluster is also annotated with the enriched transcription factors.
    """)
    message.append_text(f"The peak cluster {cluster_id} is annotated based on the peak accessibility data over cell types and timepoints.")
    message.append_text(f"The peak cluster {cluster_id} is also annotated with the associated genes.")
    message.append_text(f"The peak cluster {cluster_id} is also annotated with the enriched transcription factors.")
    message.append_text("""
    - The peak cluster is annotated based on the peak accessibility data over cell types and timepoints.
    - The peak cluster is also annotated with the associated genes.
    - The peak cluster is also annotated with the enriched transcription factors.
    """)

    message.append_text("Here is the table of the peak cluster with its pseudobulk accessibility data over cell types and timepoints:")
    message.append_table(df_clusters_groups_cluster)
    message.append_text("Here is the list of associated genes for the peak cluster:")
    message.append_text(f"association is defined by first annotating the peaks with the genes whose RNA expression is highly correlated across cells, then secondly annotating the peaks with the genes whose gene body overlaps with the peak.")
    message.append_text(genes_text)
    message.append_text("Here is the table of enriched transcription factors for the peak cluster. Each element in the table is a z-scored enrichment score for each TF in the peak cluster:")
    message.append_table(df_clusters_motifs_cluster)
    message.append_text("Here is the table of TF motifs and their associated factors:")
    message.append_table(df_motif_info)
    # summarize the table and synthesize the information
    message.append_text(f"Please analyze the table and write a short summary of the peak cluster {cluster_id} in your own words. \n")

    # write the prompt
    prompt_str = message.to_markdown()
    # Save the prompt to a file
    with open(f"prompt_cluster_{cluster_id}.md", "w", encoding="utf-8") as f:
        f.write(prompt_str)

    # generate the response (write to the markdown file)
    try:
        response = agent(message)   
        write_to_markdown(filename = f"peak_clusters_report_v3.md", 
                            content = f"## Cluster {cluster_id}\n\n" + str(response))
        
        # print the response to the console
        print(f"## Cluster {cluster_id}\n\n" + str(response))
    except Exception as e:
        error_msg = f"Error processing cluster {cluster_id}: {str(e)}"
        print(error_msg)
        write_to_markdown(filename = f"peak_clusters_report_v3.md", 
                            content = f"## Cluster {cluster_id}\n\n{error_msg}")
        continue