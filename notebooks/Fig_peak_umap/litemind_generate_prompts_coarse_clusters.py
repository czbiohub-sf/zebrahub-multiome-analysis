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
# - last updated: 6/14/2025
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

# parse the genes for each cluster
cluster_genes_dict = convert_clusters_genes_to_lists(df_clusters_genes, method='nonzero')
# %% [markdown]
# ## Step 1. Preprocess the structured input from the peak UMAP clusters for LLM queries

# # %% Example of using litemind to get the current date
# from litemind import OpenAIApi
# from litemind.agent.agent import Agent
# from litemind.agent.tools.toolset import ToolSet
# from datetime import datetime

# # Define a function to get the current date
# def get_current_date() -> str:
#     """Fetch the current date"""
#     return datetime.now().strftime("%Y-%m-%d")

# # Initialize the OpenAI API
# api = OpenAIApi()

# # Create a toolset and add the function tool
# toolset = ToolSet()
# toolset.add_function_tool(get_current_date)

# # Create the agent, passing the toolset
# agent = Agent(api=api, toolset=toolset)

# # Add a system message
# agent.append_system_message("You are a helpful assistant.")

# # Ask a question that requires the tool
# response = agent("What is the current date?")

# print("Agent with Tool Response:", response)
# # Expected output:
# # Agent with Tool Response: [*assistant*:
# # The current date is 2025-05-02.
# # ]

# %% example 2:


# class ClusterSummary(BaseModel):
#     cluster_id: int
#     expression_highlights: str
#     top_genes: List[str]
#     motif_enrichment: str
#     interpretation: str

# class AllClustersSummary(BaseModel):
#     clusters: List[ClusterSummary]

# # Create a system message:
# system_message = Message(role="system", text="""
# You are a bioinformatics assistant. 
# You will receive some structured data (in table form) about clusters of cells and their expression patterns, plus textual context.
# Please provide an insightful cluster-by-cluster summary and interpret the data accordingly.
# """)

# # Create a user message to hold the data and instructions
# user_message = Message(role="user")
# user_message.append_text("""
# We have three DataFrames describing clusters. 

# 1) cluster_id-by-groups: expression across celltype/timepoint combos
# 2) cluster_id-by-genes: gene counts per cluster
# 3) cluster_id-by-motifs: TF motif enrichment

# Please analyze them and produce a structured summary for each cluster.
# """)

# # Append the three DataFrames as tables:
# user_message.append_text("### clusters_vs_groups table:")
# user_message.append_table(df_clusters_groups)

# user_message.append_text("### clusters_vs_genes table:")
# user_message.append_table(df_clusters_genes)

# user_message.append_text("### clusters_vs_motifs table:")
# user_message.append_table(df_clusters_motifs)

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

print(f"\nTotal estimated tokens for all tables: \
    {estimate_tokens(df_clusters_groups.to_string()) + \
    estimate_tokens(cluster_genes_dict) + \
    estimate_tokens(df_clusters_motifs.to_string()) + \
    estimate_tokens(df_motif_info.to_string())}")

for cluster_id in df_clusters_groups.index:
    print(f"\n=== Processing Cluster {cluster_id} ===")
    
    # Use the new utility function to process cluster data
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

    # Section 1: Job Analysis
    message = Message()
    message.append_text("Here is the table of peak cluster with their pseudobulk expression data:")
    message.append_table(df_clusters_groups_cluster)
    message.append_text("Here is the table of peak cluster with their associated genes:")
    message.append_text(genes_text)
    message.append_text("Here is the table of peak cluster with their associated motifs:")
    message.append_table(df_clusters_motifs_cluster)
    
    # Only include motif info if not too large
    if estimate_tokens(df_motif_info.to_string()) < 50000:
        message.append_text("Here is the table of TF motifs and their associated factors:")
        message.append_table(df_motif_info)
    else:
        message.append_text("Note: TF motif information table was too large to include.")
    
    message.append_text("Please analyze the table and write a short summary of the peak clusters in your own words. \n")
    
    try:
        response = agent(message)   
        write_to_markdown(f"## Cluster {cluster_id}\n\n" + str(response))
        
        # print the response to the console
        print(f"## Cluster {cluster_id}\n\n" + str(response))
    except Exception as e:
        error_msg = f"Error processing cluster {cluster_id}: {str(e)}"
        print(error_msg)
        write_to_markdown(f"## Cluster {cluster_id}\n\n{error_msg}")
        continue

# %%
# Check the available models from OpenAIApi
from litemind.apis.model_features import ModelFeatures
api = OpenAIApi()
print("Available models:", api.list_models())

# Use utility functions to check for WebSearchTool feature
print("=== Checking specific models for WebSearchTool ===")
models_to_check = ["gpt-4o-search-preview", "o1-high", "gpt-4o", "gpt-4.1"]

for model in models_to_check:
    has_web_search = check_model_feature(api, model, ModelFeatures.WebSearchTool)
    print(f"{model}: WebSearchTool = {has_web_search}")

print()

# Find all models with WebSearchTool
web_search_models = find_models_with_feature(api, ModelFeatures.WebSearchTool)
print(f"Models with WebSearchTool: {web_search_models}")
print(f"Total: {len(web_search_models)}")
print()

# Get the best model for analysis
best_model = get_best_model_for_analysis(api, require_web_search=False)
best_web_search_model = get_best_model_for_analysis(api, require_web_search=True)

print(f"Best model for analysis: {best_model}")
print(f"Best model with web search: {best_web_search_model}")
print()

# Print detailed capabilities for a few key models
print_model_capabilities(api, models_to_check)

# %% Take 2: Enable the "search" mode
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures
from pydantic import BaseModel
from typing import List, Optional

# Create or clear the markdown file
with open("peak_clusters_report_with_search.md", "w", encoding="utf-8") as f:
    f.write("# Peak Clusters Report\n\n")

# Check the size of input tables
print("=== INPUT TABLE ANALYSIS ===")
get_table_info(df_clusters_groups, "df_clusters_groups")
# get_table_info(cluster_genes_dict, "df_clusters_genes") 
get_table_info(df_clusters_motifs, "df_clusters_motifs")
get_table_info(df_motif_info, "df_motif_info")

# Initialize the OpenAI API
api = OpenAIApi()

# create a toolset
toolset = ToolSet()
toolset.add_builtin_web_search_tool()

print(f"\nTotal estimated tokens for all tables: {estimate_tokens(df_clusters_groups.to_string()) + estimate_tokens(cluster_genes_dict) + estimate_tokens(df_clusters_motifs.to_string()) + estimate_tokens(df_motif_info.to_string())}")

for cluster_id in df_clusters_groups.index:
    print(f"\n=== Processing Cluster {cluster_id} ===")
    
    # Use the new utility function to process cluster data
    df_clusters_groups_cluster, genes_text, df_clusters_motifs_cluster, cluster_tokens = process_cluster_data(
        cluster_id, df_clusters_groups, cluster_genes_dict, df_clusters_motifs, df_motif_info
    )
    
    print(f"Estimated tokens for cluster {cluster_id}: {cluster_tokens}")
    
    # Skip if too large (adjust threshold as needed)
    if cluster_tokens > 100000:  # Adjust this threshold
        print(f"Skipping cluster {cluster_id} - too large ({cluster_tokens} tokens)")
        continue
    

    # Create the agent, passing the toolset
    agent = Agent(api=api, model_name="gpt-4.1", toolset=toolset)
    # Add a system message
    agent.append_system_message("You are an expert developmental biologist specialized in vertebrate embryogenesis, especially zebrafish embryogenesis.")

    # Section 1: Job Analysis
    message = Message()
    message.append_text("Here is the table of peak cluster with their pseudobulk expression data:")
    message.append_table(df_clusters_groups_cluster)
    message.append_text("Here is the table of peak cluster with their associated genes:")
    message.append_text(genes_text)
    message.append_text("Here is the table of peak cluster with their associated motifs:")
    message.append_table(df_clusters_motifs_cluster)
    
    # Only include motif info if not too large
    if estimate_tokens(df_motif_info.to_string()) < 50000:
        message.append_text("Here is the table of TF motifs and their associated factors:")
        message.append_table(df_motif_info)
    else:
        message.append_text("Note: TF motif information table was too large to include.")
    
    message.append_text("Please analyze the table and write a short summary of the peak clusters in your own words. \n")
    
    try:
        response = agent(message)
        write_to_markdown(f"## Cluster {cluster_id}\n\n" + str(response),
                          filename="peak_clusters_report_with_search.md")
        
        # print the response to the console
        print(f"## Cluster {cluster_id}\n\n" + str(response))
    except Exception as e:
        error_msg = f"Error processing cluster {cluster_id}: {str(e)}"
        print(error_msg)
        write_to_markdown(f"## Cluster {cluster_id}\n\n{error_msg}",
                          filename="peak_clusters_report_with_search.md")
        continue

# %%
# # import the necessary libraries from litemind
# from litemind.agent.messages.message import Message
# from litemind.apis.model_features import ModelFeatures
# from pydantic import BaseModel
# from typing import List, Optional

# # Create the CombinedApi instance (automatically picks best LLM if you want):
# api = OpenAIApi()

# # We can optionally choose a model that supports text + table usage:
# model_name = api.get_best_model([ModelFeatures.TextGeneration])
# print(model_name)

# # Create a system message:
# system_message = Message(role="system", text="""
# You are a bioinformatics assistant. 
# You will receive some structured data (in table form) about clusters of cells and their expression patterns, plus textual context.
# Please provide an insightful cluster-by-cluster summary and interpret the data accordingly.
# """)

# # Create a user message to hold the data and instructions
# user_message = Message(role="user")
# user_message.append_text("""
# We have three DataFrames describing clusters. 

# 1) cluster_id-by-groups: expression across celltype/timepoint combos
# 2) cluster_id-by-genes: gene counts per cluster
# 3) cluster_id-by-motifs: TF motif enrichment

# Please analyze them and produce a structured summary for each cluster.
# """)

# # Append the three DataFrames as tables:
# user_message.append_text("### clusters_vs_groups table:")
# user_message.append_table(df_clusters_groups)

# user_message.append_text("### clusters_vs_genes table:")
# user_message.append_table(df_clusters_genes)

# user_message.append_text("### clusters_vs_motifs table:")
# user_message.append_table(df_clusters_motifs)

# # user_message.append_text("### motifs information table:")
# # user_message.append_table(df_motif_info)

