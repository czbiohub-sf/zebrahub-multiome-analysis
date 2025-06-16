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

# %% [markdown]
# ## Step 1. Preprocess the structured input from the peak UMAP clusters for LLM queries

# %% Example of using litemind to get the current date
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from datetime import datetime

# Define a function to get the current date
def get_current_date() -> str:
    """Fetch the current date"""
    return datetime.now().strftime("%Y-%m-%d")

# Initialize the OpenAI API
api = OpenAIApi()

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_current_date)

# Create the agent, passing the toolset
agent = Agent(api=api, toolset=toolset)

# Add a system message
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires the tool
response = agent("What is the current date?")

print("Agent with Tool Response:", response)
# Expected output:
# Agent with Tool Response: [*assistant*:
# The current date is 2025-05-02.
# ]

# %% example 2:
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

from pydantic import BaseModel
from typing import List, Optional

class ClusterSummary(BaseModel):
    cluster_id: int
    expression_highlights: str
    top_genes: List[str]
    motif_enrichment: str
    interpretation: str

class AllClustersSummary(BaseModel):
    clusters: List[ClusterSummary]

# Create a system message:
system_message = Message(role="system", text="""
You are a bioinformatics assistant. 
You will receive some structured data (in table form) about clusters of cells and their expression patterns, plus textual context.
Please provide an insightful cluster-by-cluster summary and interpret the data accordingly.
""")

# Create a user message to hold the data and instructions
user_message = Message(role="user")
user_message.append_text("""
We have three DataFrames describing clusters. 

1) cluster_id-by-groups: expression across celltype/timepoint combos
2) cluster_id-by-genes: gene counts per cluster
3) cluster_id-by-motifs: TF motif enrichment

Please analyze them and produce a structured summary for each cluster.
""")

# Append the three DataFrames as tables:
user_message.append_text("### clusters_vs_groups table:")
user_message.append_table(df_clusters_groups)

user_message.append_text("### clusters_vs_genes table:")
user_message.append_table(df_clusters_genes)

user_message.append_text("### clusters_vs_motifs table:")
user_message.append_table(df_clusters_motifs)


# %% 
# Function to write content to a markdown file
def write_to_markdown(content, filename="peak_clusters_report.md", mode="a"):
    with open(filename, mode, encoding="utf-8") as f:
        f.write(content + "\n\n")
    return filename

# Function to estimate token count (rough approximation)
def estimate_tokens(text):
    """Rough estimation: ~4 characters per token for English text"""
    return len(str(text)) // 4

# Function to get table dimensions and sample
def get_table_info(df, name):
    """Get basic info about a dataframe"""
    print(f"\n{name} shape: {df.shape}")
    print(f"{name} columns: {list(df.columns)}")
    if len(df.columns) > 10:
        print(f"First 5 columns: {list(df.columns[:5])}")
        print(f"Last 5 columns: {list(df.columns[-5:])}")
    print(f"Estimated tokens for {name}: {estimate_tokens(df.to_string())}")
    return df.shape, estimate_tokens(df.to_string())

# Function to reduce table size if needed
def reduce_table_size(df, max_cols=50, max_rows=100):
    """Reduce table size by limiting columns and rows"""
    df_reduced = df.copy()
    
    # Limit columns
    if len(df.columns) > max_cols:
        # Keep first max_cols columns
        df_reduced = df_reduced.iloc[:, :max_cols]
        print(f"Reduced columns from {len(df.columns)} to {max_cols}")
    
    # Limit rows  
    if len(df) > max_rows:
        df_reduced = df_reduced.head(max_rows)
        print(f"Reduced rows from {len(df)} to {max_rows}")
        
    return df_reduced

# Function to create summary statistics for large tables
def summarize_large_table(df, name):
    """Create a summary of a large table instead of including the full table"""
    summary = f"""
    {name} Summary:
    - Shape: {df.shape}
    - Columns: {len(df.columns)}
    - Non-zero values: {(df != 0).sum().sum() if df.select_dtypes(include=[np.number]).shape[1] > 0 else 'N/A'}
    - Top 5 columns by variance: {df.var().nlargest(5).index.tolist() if df.select_dtypes(include=[np.number]).shape[1] > 0 else 'N/A'}
    """
    return summary

# %%
# Function to convert clusters-by-genes dataframe to a more compact representation
def convert_clusters_genes_to_lists(df_clusters_genes, method='nonzero', threshold=0, top_n=None):
    """
    Convert clusters-by-genes dataframe to a more compact representation
    
    Parameters:
    -----------
    df_clusters_genes : pd.DataFrame
        DataFrame with clusters as rows and genes as columns
    method : str
        'nonzero' - genes with non-zero values
        'threshold' - genes above a threshold value
        'top_n' - top N genes by value
    threshold : float
        Threshold value when method='threshold'
    top_n : int
        Number of top genes when method='top_n'
    
    Returns:
    --------
    dict : Dictionary mapping cluster_id to list of gene names
    """
    cluster_gene_lists = {}
    
    for cluster_id in df_clusters_genes.index:
        cluster_row = df_clusters_genes.loc[cluster_id]
        
        if method == 'nonzero':
            # Get genes with non-zero values
            active_genes = cluster_row[cluster_row > 0].index.tolist()
            
        elif method == 'threshold':
            # Get genes above threshold
            active_genes = cluster_row[cluster_row > threshold].index.tolist()
            
        elif method == 'top_n':
            # Get top N genes by value
            if top_n is None:
                top_n = 20  # default
            active_genes = cluster_row.nlargest(top_n).index.tolist()
            
        else:
            raise ValueError("method must be 'nonzero', 'threshold', or 'top_n'")
        
        cluster_gene_lists[cluster_id] = active_genes
    
    return cluster_gene_lists

# Example usage:
cluster_genes_dict = convert_clusters_genes_to_lists(df_clusters_genes, method='nonzero')

# # Or get top 20 genes per cluster:
# cluster_genes_dict = convert_clusters_genes_to_lists(df_clusters_genes, method='top_n', top_n=20)

# Print example
for cluster_id, genes in list(cluster_genes_dict.items())[:3]:  # First 3 clusters
    print(f"Cluster {cluster_id}: {len(genes)} genes")
    print(f"  Genes: {', '.join(genes[:10])}...")  # First 10 genes
    print()

# %%
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
    
    # subset the dataframes for the current cluster
    # Convert Series to DataFrame using .to_frame().T (transpose to make it a row)
    df_clusters_groups_cluster = df_clusters_groups.loc[cluster_id].to_frame().T
    df_clusters_genes_cluster = cluster_genes_dict[cluster_id]
    df_clusters_motifs_cluster = df_clusters_motifs.loc[cluster_id].to_frame().T

    # Convert df_clusters_genes_cluster to a string
    genes_text = ', '.join(df_clusters_genes_cluster)  # Convert list to string

    # Estimate tokens for this cluster's input
    cluster_tokens = (estimate_tokens(df_clusters_groups_cluster.to_string()) + 
                     estimate_tokens(df_clusters_genes_cluster) + 
                     estimate_tokens(df_clusters_motifs_cluster.to_string()) + 
                     estimate_tokens(df_motif_info.to_string()))
    
    print(f"Estimated tokens for cluster {cluster_id}: {cluster_tokens}")
    
    # Skip if too large (adjust threshold as needed)
    if cluster_tokens > 100000:  # Adjust this threshold
        print(f"Skipping cluster {cluster_id} - too large ({cluster_tokens} tokens)")
        continue

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
# # Peak Cluster Analysis with Direct DataFrame Input
# import pandas as pd
# import numpy as np
# from typing import Dict, Optional
# from litemind import OpenAIApi
# from litemind.agent.agent import Agent
# from litemind.agent.messages.message import Message
# from litemind.media.types.media_table import Table
# import tempfile
# import os
# %%
# import the necessary libraries from litemind
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures
from pydantic import BaseModel
from typing import List, Optional

# # define the Classes for the "summary" of each peak cluster
# class ClusterSummary(BaseModel):
#     cluster_id: int
#     expression_highlights: List[str]
#     top_genes: List[str]
#     motif_enrichment: List[str]
#     interpretation: List[str]

# class AllClustersSummary(BaseModel):
#     clusters: List[ClusterSummary]

# Create the CombinedApi instance (automatically picks best LLM if you want):
api = OpenAIApi()

# We can optionally choose a model that supports text + table usage:
model_name = api.get_best_model([ModelFeatures.TextGeneration])
print(model_name)

# Create a system message:
system_message = Message(role="system", text="""
You are a bioinformatics assistant. 
You will receive some structured data (in table form) about clusters of cells and their expression patterns, plus textual context.
Please provide an insightful cluster-by-cluster summary and interpret the data accordingly.
""")

# Create a user message to hold the data and instructions
user_message = Message(role="user")
user_message.append_text("""
We have three DataFrames describing clusters. 

1) cluster_id-by-groups: expression across celltype/timepoint combos
2) cluster_id-by-genes: gene counts per cluster
3) cluster_id-by-motifs: TF motif enrichment

Please analyze them and produce a structured summary for each cluster.
""")

# Append the three DataFrames as tables:
user_message.append_text("### clusters_vs_groups table:")
user_message.append_table(df_clusters_groups)

user_message.append_text("### clusters_vs_genes table:")
user_message.append_table(df_clusters_genes)

user_message.append_text("### clusters_vs_motifs table:")
user_message.append_table(df_clusters_motifs)

# user_message.append_text("### motifs information table:")
# user_message.append_table(df_motif_info)

