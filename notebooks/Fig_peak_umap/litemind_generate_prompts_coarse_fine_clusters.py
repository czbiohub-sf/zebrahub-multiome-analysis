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

# %%
# import the necessary libraries
# from IPython.core.interactiveshell import interactiveShell
# interactiveShell.ast_node_interactivity = "all"
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
    print_model_capabilities,
    # get_common_cluster_ids
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

# Initialize the OpenAI API
api = OpenAIApi()

# Create prompts directory if it doesn't exist
prompts_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig_peak_umap/prompts"
os.makedirs(prompts_dir, exist_ok=True)

# Create or clear the markdown file
with open("peak_clusters_report_v4.md", "w", encoding="utf-8") as f:
    f.write("# Peak Clusters Report\n\n")

# Check the size of input tables
print("=== INPUT TABLE ANALYSIS ===")
get_table_info(df_clusters_groups, "df_clusters_groups")
# get_table_info(cluster_genes_dict, "df_clusters_genes") 
get_table_info(df_clusters_motifs, "df_clusters_motifs")
get_table_info(df_motif_info, "df_motif_info")


for cluster_id in df_clusters_groups.index:
# for cluster_id in [0,1]: # test with only two clusters
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

    # Create the agent, passing the toolset
    agent = Agent(api=api)
    # Add a system message
    agent.append_system_message("You are an expert developmental biologist specialized in vertebrate embryogenesis, especially zebrafish embryogenesis.")

    # Create streamlined message with reduced redundancy
    message = Message()
    
    # Background context - consolidated and concise
    message.append_text("## Project Background")
    message.append_text("""
    We generated a time-resolved single-cell multi-omic atlas of zebrafish embryogenesis, profiling both chromatin accessibility (scATAC-seq) and gene expression (scRNA-seq) from 94,000+ cells across six developmental timepoints (10-24 hpf, gastrulation to early organogenesis).
    
    **Analysis Pipeline:**
    1. Pseudo-bulked scATAC-seq data by celltype and timepoint, resulting in peaks-by-pseudobulk (celltype & timepoint) data
    2. Applied leiden clustering algorithm to identify peak clusters with similar accessibility patterns
    3. Each peak cluster represents co-accessible regulatory regions across celltypes and timepoints
    """)
    
    # Task context - clear and direct
    message.append_text(f"## Task: Annotate Peak Cluster {cluster_id}")
    message.append_text("""
    Analyze the provided data to characterize this peak cluster's biological function and developmental role. The cluster is annotated with:
    - **Accessibility patterns**: Pseudobulk data across cell types and timepoints
    - **Associated genes**: A list of genes whose RNA expression is significantly correlated with peak accessibility or overlapping with gene bodies  
    - **Enriched transcription factors**: Transcription factors with significant motif enrichment (z-scores)
    - **TF motif-factor mapping**: A table of transcription factor motifs and their associated transcription factors
    """)

    # Data sections - organized and labeled
    message.append_text("### 1. Accessibility Data (Pseudobulk by Cell Type & Timepoint)")
    message.append_table(df_clusters_groups_cluster)
    
    message.append_text("### 2. Associated Genes")
    message.append_text(genes_text)
    
    message.append_text("### 3. Enriched Transcription Factors (Z-scored Enrichment)")
    message.append_table(df_clusters_motifs_cluster)
    
    message.append_text("### 4. TF Motif-Factor Mapping")
    message.append_table(df_motif_info)
    
    # Clear analysis request
    message.append_text(f"""
    ### Analysis Request
    Please provide a concise biological interpretation of peak cluster {cluster_id}, addressing:
    1. **Temporal dynamics**: When is this cluster most active?
    2. **Cell type specificity**: Which cell types show highest accessibility?
    3. **Regulatory program**: What biological processes/pathways are likely regulated?
    4. **Key transcription factors**: Which TFs are driving this regulatory program?
    """)

    # write the prompt
    prompt_str = message.to_markdown()
    # Save the prompt to a file in the prompts directory
    with open(os.path.join(prompts_dir, f"prompt_cluster_{cluster_id}.md"), "w", encoding="utf-8") as f:
        f.write(prompt_str)

    # generate the response (write to the markdown file)
    try:
        response = agent(message)   
        write_to_markdown(filename = f"peak_clusters_report_v4.md", 
                            content = f"## Cluster {cluster_id}\n\n" + str(response))
        
        # print the response to the console
        print(f"## Cluster {cluster_id}\n\n" + str(response))
    except Exception as e:
        error_msg = f"Error processing cluster {cluster_id}: {str(e)}"
        print(error_msg)
        write_to_markdown(filename = f"peak_clusters_report_v4.md", 
                            content = f"## Cluster {cluster_id}\n\n{error_msg}")
        continue

# %% Section 2. Prompting for the fine-clusters
# load the data
# import the datasets
# (1) peak clusters-by-pseudobulk groups
df_clusters_groups = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_unified_by_pseudobulk.csv",
                                 index_col=0)
# (2) peak clusters-by-associated genes
df_clusters_genes = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_unified_by_assoc_genes.csv",
                                index_col=0)
# (3) peak clusters-by-TF motifs
df_clusters_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_unified_by_motifs_maelstrom.csv",
                                 index_col=0)
# (4) motifs-by-factors
df_motif_info = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors.csv", index_col=0)

# parse the genes for each cluster (to reduce the size of the input table tokens)
cluster_genes_dict = convert_clusters_genes_to_lists(df_clusters_genes, method='nonzero')

# %% Re-running the prompts for the fine-clusters
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures
from pydantic import BaseModel
from typing import List, Optional

# Initialize the OpenAI API
api = OpenAIApi()

# Create prompts directory if it doesn't exist
prompts_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/Fig_peak_umap/prompts/fine_clusters"
os.makedirs(prompts_dir, exist_ok=True)

# Create or clear the markdown file
with open("peak_clusters_report_fine.md", "w", encoding="utf-8") as f:
    f.write("# Peak Clusters Report\n\n")

# Check the size of input tables
print("=== INPUT TABLE ANALYSIS ===")
get_table_info(df_clusters_groups, "df_clusters_groups")
# get_table_info(cluster_genes_dict, "df_clusters_genes") 
get_table_info(df_clusters_motifs, "df_clusters_motifs")
get_table_info(df_motif_info, "df_motif_info")

# # Get common cluster IDs across all DataFrames
# common_cluster_ids = get_common_cluster_ids(df_clusters_groups, cluster_genes_dict, df_clusters_motifs)
# print(f"Processing {len(common_cluster_ids)} common fine clusters")

for cluster_id in df_clusters_groups.index:
# for cluster_id in [0,1]: # test with only two clusters
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

    # Create the agent, passing the toolset
    agent = Agent(api=api)
    # Add a system message
    agent.append_system_message("You are an expert developmental biologist specialized in vertebrate embryogenesis, especially zebrafish embryogenesis.")

    # Create streamlined message with reduced redundancy
    message = Message()
    
    # Background context - consolidated and concise
    message.append_text("## Project Background")
    message.append_text("""
    We generated a time-resolved single-cell multi-omic atlas of zebrafish embryogenesis, profiling both chromatin accessibility (scATAC-seq) and gene expression (scRNA-seq) from 94,000+ cells across six developmental timepoints (10-24 hpf, gastrulation to early organogenesis).
    
    **Analysis Pipeline:**
    1. Pseudo-bulked scATAC-seq data by celltype and timepoint, resulting in peaks-by-pseudobulk (celltype & timepoint) data
    2. Applied leiden clustering algorithm to identify peak clusters with similar accessibility patterns
    3. Each peak cluster represents co-accessible regulatory regions across celltypes and timepoints
    """)
    
    # Task context - clear and direct
    message.append_text(f"## Task: Annotate Peak Cluster {cluster_id}")
    message.append_text("""
    Analyze the provided data to characterize this peak cluster's biological function and developmental role. The cluster is annotated with:
    - **Accessibility patterns**: Pseudobulk data across cell types and timepoints
    - **Associated genes**: A list of genes whose RNA expression is significantly correlated with peak accessibility or overlapping with gene bodies  
    - **Enriched transcription factors**: Transcription factors with significant motif enrichment (z-scores)
    - **TF motif-factor mapping**: A table of transcription factor motifs and their associated transcription factors
    """)

    # Data sections - organized and labeled
    message.append_text("### 1. Accessibility Data (Pseudobulk by Cell Type & Timepoint)")
    message.append_table(df_clusters_groups_cluster)
    
    message.append_text("### 2. Associated Genes")
    message.append_text(genes_text)
    
    message.append_text("### 3. Enriched Transcription Factors (Z-scored Enrichment)")
    message.append_table(df_clusters_motifs_cluster)
    
    message.append_text("### 4. TF Motif-Factor Mapping")
    message.append_table(df_motif_info)
    
    # Clear analysis request
    message.append_text(f"""
    ### Analysis Request
    Please provide a concise biological interpretation of peak cluster {cluster_id}, addressing:
    1. **Temporal dynamics**: When is this cluster most active?
    2. **Cell type specificity**: Which cell types show highest accessibility?
    3. **Regulatory program**: What biological processes/pathways are likely regulated?
    4. **Key transcription factors**: Which TFs are driving this regulatory program?
    """)

    # write the prompt
    prompt_str = message.to_markdown()
    # Save the prompt to a file in the prompts directory
    with open(os.path.join(prompts_dir, f"prompt_cluster_{cluster_id}.md"), "w", encoding="utf-8") as f:
        f.write(prompt_str)

    # generate the response (write to the markdown file)
    try:
        response = agent(message)   
        write_to_markdown(filename = f"peak_clusters_report_fine.md", 
                            content = f"## Cluster {cluster_id}\n\n" + str(response))
        
        # print the response to the console
        print(f"## Cluster {cluster_id}\n\n" + str(response))
    except Exception as e:
        error_msg = f"Error processing cluster {cluster_id}: {str(e)}"
        print(error_msg)
        write_to_markdown(filename = f"peak_clusters_report_fine.md", 
                            content = f"## Cluster {cluster_id}\n\n{error_msg}")
        continue



