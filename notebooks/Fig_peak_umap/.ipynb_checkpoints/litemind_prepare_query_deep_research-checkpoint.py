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
# - last updated: 6/7/2025
#
# - DESCRIBE the goals here:
#     - 1) preprocess the structured input from the peak UMAP clusters for LLM queries (this should be scripted in the future)
#     - 2) Perform structured "query" using litemind (ChatGPT API, etc.)
#     - 3) (To-Do) Export the input query into a 

# %%
# define the openAI API key
import os
# # Set the API key BEFORE importing and using litemind
# os.environ["OPENAI_API_KEY"] = ""
# For OpenAI
openai_api_key = os.environ.get('OPENAI_API_KEY')
print(f"OpenAI API key available: {'Yes' if openai_api_key else 'No'}")

# %%
# from dotenv import load_dotenv
# load_dotenv()  # load variables from .env file

# import os
# api_key = os.environ.get('OPENAI_API_KEY')
# print(f"OpenAI API key available: {'Yes' if openai_api_key else 'No'}")

# %% [markdown]
# ## Step 1. preprocess the structured input from the peak UMAP object
# - scanpy/pandas to import and process the dataframe objects
#
# ### processing will be done in the following steps:
# - First, import the peaks-by-pseudobulk matrix (with leiden clusters in adata.obs["leiden"]
# - 1) context: a text blurb describing the data structure
# Then, for each peak cluster, compute the following dataframes:
# - 2) compute a vector for pseudobulk (averaging over peaks from each leiden cluster): a vector whose dimension is the celltype&timepoint
# - 3) compute genes-by-counts (genes associated with each peak)
# - 4) compute cluster-by-motif enrichment scores (computed by gimmemotifs maelstrom)
#
#

# %%
import scanpy as sc
import pandas as pd
import scipy
import numpy as np

# %%
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad")
adata_peaks

# %%
adata_peaks_ct_tp = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_wo_log_all_peaks_3d_umap.h5ad")
adata_peaks_ct_tp

# %%
# Checkpoint - save the master object
adata_peaks_ct_tp.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad")

# %%
# filter out the peaks (2 MT peaks and 2 peaks that extend beyond the chromosome end)
adata_peaks_ct_tp = adata_peaks_ct_tp[adata_peaks_ct_tp.obs_names.isin(adata_peaks.obs_names)]
adata_peaks_ct_tp

# %%
# # copy over the metadata
adata_peaks_ct_tp.obs = adata_peaks.obs.copy()
adata_peaks_ct_tp.obsm = adata_peaks.obsm.copy()

# %%
sc.pl.umap(adata_peaks_ct_tp, color=["accessibility_muscle","accessibility_epidermis",
                               "accessibility_optic_cup"], 
           vmin=0, vmax=500)

# %% [markdown]
# ### 1) a pseudobulk vector for each leiden cluster

# %%
adata_peaks_ct_tp

# %%
# convert the counts layer to "normalized"
adata_peaks_ct_tp.X = adata_peaks_ct_tp.layers["normalized"].copy()

# Get the unique leiden clusters
leiden_clusters = adata_peaks_ct_tp.obs["leiden_coarse"].unique()

# Create a pandas DataFrame to store the results
cluster_pseudobulk_df = pd.DataFrame(
    index=leiden_clusters,
    columns=adata_peaks_ct_tp.var_names
)

# Option 1: Using pandas groupby for vectorized operations
# This is often more memory-efficient for large datasets
pseudobulk_matrix = pd.DataFrame(
    adata_peaks_ct_tp.X.toarray() if scipy.sparse.issparse(adata_peaks_ct_tp.X) else adata_peaks_ct_tp.X,
    index=adata_peaks_ct_tp.obs["leiden_coarse"],
    columns=adata_peaks_ct_tp.var_names
)
cluster_pseudobulk_df = pseudobulk_matrix.groupby(level=0).mean()

cluster_pseudobulk_df.head()

# %%
# save the count matrix as csv file
cluster_pseudobulk_df.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_pseudobulk.csv")

# %% [markdown]
# ### 2) a dataframe of genes-by-counts for each leiden cluster

# %%
adata_peaks_ct_tp

# %%
# import the annotation ("associated_genes")
df_genes_anno = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/all_peaks_annotated.csv",
                            index_col=0)
df_genes_anno.head()

# %%
# # copy over the metadata ("linked_gene", etc.)
metadata_list = ['linked_gene', 'link_score', 'link_zscore', 'link_pvalue']
for col in metadata_list:
    adata_peaks_ct_tp.obs[col] = adata_peaks_ct_tp.obs_names.map(df_genes_anno[col])

adata_peaks_ct_tp.obs["linked_gene"]


# %%
# import the functions to annotate the peaks with "associated genes"
# this function uses "linked_genes" from Signac and "overlapping with gene body" based on GTF file
from utils_gene_annotate import *

# %%
# associate peaks to genes
# (1) use "linked_gene" if possible
# (2) use "gene_body_overlaps" as secondary
# (3) add NaN otherwise
adata_peaks_ct_tp = create_gene_associations(adata_peaks_ct_tp)
adata_peaks_ct_tp.obs.head()

# %%
adata_peaks_ct_tp.obs["associated_gene"]

# %%
# First, let's extract the relevant columns from adata_peaks_ct_tp.obs
peak_gene_data = adata_peaks_ct_tp.obs[["leiden_coarse", "associated_gene"]].copy()

# Some genes might have multiple entries, so we'll create a function to handle this
def process_gene_entries(gene_entry):
    if pd.isna(gene_entry):
        return []
    # Handle different possible formats of gene entries (comma-separated, list, etc.)
    if isinstance(gene_entry, str):
        return [g.strip() for g in gene_entry.split(',') if g.strip()]
    return [gene_entry]

# Create an expanded DataFrame with one row per peak-gene pair
peak_gene_pairs = []
for idx, row in peak_gene_data.iterrows():
    genes = process_gene_entries(row["associated_gene"])
    for gene in genes:
        peak_gene_pairs.append({
            "leiden": row["leiden_coarse"],
            "gene": gene
        })

# Convert to DataFrame
peak_gene_df = pd.DataFrame(peak_gene_pairs)

# If there are no valid gene associations, handle this case
if len(peak_gene_df) == 0:
    print("No gene associations found in the dataset")
    genes_by_counts = pd.DataFrame(index=adata_peaks_ct_tp.obs["leiden"].unique())
else:
    # Count occurrences of each gene in each leiden cluster
    gene_counts = peak_gene_df.groupby(["leiden", "gene"]).size().reset_index(name="count")
    
    # Pivot to create a leiden cluster x gene matrix
    genes_by_counts = gene_counts.pivot(index="leiden", columns="gene", values="count").fillna(0)
    
    # Ensure we have a row for every leiden cluster (even if no genes are associated)
    missing_clusters = set(adata_peaks_ct_tp.obs["leiden_coarse"].unique()) - set(genes_by_counts.index)
    for cluster in missing_clusters:
        genes_by_counts.loc[cluster] = 0
    
    # Sort the rows by leiden cluster
    genes_by_counts = genes_by_counts.loc[sorted(genes_by_counts.index)]

print(f"Created genes-by-counts matrix with shape: {genes_by_counts.shape}")

# %%
genes_by_counts.head()

# %%
# First, create a dictionary that maps each leiden cluster to its associated genes
leiden_to_genes = {}

for leiden in genes_by_counts.index:
    # Get the genes that have counts > 0 for this leiden cluster
    genes_present = genes_by_counts.loc[leiden]
    genes_present = genes_present[genes_present > 0].index.tolist()
    
    # Store in dictionary
    leiden_to_genes[leiden] = genes_present

# If you need a DataFrame with comma-separated gene lists
leiden_gene_lists_df = pd.DataFrame({
    'leiden': list(leiden_to_genes.keys()),
    'genes': [','.join(genes) for genes in leiden_to_genes.values()]
})
leiden_gene_lists_df.set_index('leiden', inplace=True)

# Print example
print(leiden_gene_lists_df.head())

# %%
genes_by_counts.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_assoc_genes.csv")

# %% [markdown]
# ### 3) cluster-by-motifs (enrichment score)

# %%
clust_by_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/peak_clusts_by_motifs_maelstrom.csv", index_col=0)
# Strip "z-score " from column names
clust_by_motifs.columns = clust_by_motifs.columns.str.replace('z-score ', '')
clust_by_motifs.head()

# %%
clust_by_motifs.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_motifs_maelstrom.csv")


# %% [markdown]
# ### 4) We need to import the TFs for each motif as a dictionary/df

# %%
info_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors.csv", index_col=0)
info_motifs.head()

# %%
info_motifs.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/info_cisBP_v2_danio_rerio_motif_factors.csv")

# %% [markdown]
# ## Step 2. generate a structured input
# - a collection of csv files (or dataframes)
# - query with structured output - revisit this with litemind.2025.6.8
#

# %%
from litemind import OpenAIApi
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.tools.function_tool import FunctionTool
from litemind.apis.model_features import ModelFeatures
from litemind.agent.react.react_agent import ReActAgent
from pandas import DataFrame

# create an agent that can take pandas df as the input
# 5. Using the Agent with a Code Block
api = OpenAIApi()
agent = ReActAgent(api=api)
agent.append_system_message("You are a code generation assistant.")
response = agent(
    "Write a python function that returns the factorial of a number.",
)
print("Agent with Code Block:", response)

# Create a small table with pandas:
table = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

user_message = Message(role="user")
user_message.append_text("Can you describe what you see in the table?")
user_message.append_table(table)

# Run agent:
response = agent(user_message)

print("Agent with Table:", response)


# api = OpenAIApi()

# # Using the ReAct Agent with a Custom Tool
# # Create the ReAct agent
# react_agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=3)

# # Add a system message
# react_agent.append_system_message(
#     "You are a helpful assistant that can answer questions and use tools."
# )

# # Ask a question
# response = react_agent("What is the current time?")

# # Print the response
# print("ReAct Agent Response:", response)

# %%

# %% [markdown]
# ## create a structured output (pydantic)
#

# %%
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


# %%
# (1) cluster_id-by-groups: e.g., expression across celltype-timepoint combos
df_clusters_groups = pd.DataFrame({
    'cluster_id': [1, 2, 3],
    'celltype_timepoint_1_expr': [10.2, 5.1, 0.0],
    'celltype_timepoint_2_expr': [8.3, 9.9, 1.2]
})

# (2) cluster_id-by-genes: e.g., gene counts or expression for relevant genes
df_clusters_genes = pd.DataFrame({
    'cluster_id': [1, 2, 3, 1, 2, 3],
    'gene': ['SOX2', 'SOX2', 'SOX2', 'MYC', 'MYC', 'MYC'],
    'count': [100, 80, 20, 200, 120, 5]
})

# (3) cluster_id-by-motifs: e.g., TF motif enrichment
df_clusters_motifs = pd.DataFrame({
    'cluster_id': [1, 2, 3],
    'motif': ['TF1_motif', 'TF2_motif', 'TF3_motif'],
    'enrichment_score': [3.2, 1.8, 0.5]
})

# %%
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

# Create the CombinedApi instance (automatically picks best LLM if you want):
api = OpenAIApi()

# We can optionally choose a model that supports text + table usage:
model_name = api.get_best_model([ModelFeatures.TextGeneration])

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

# %%
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.callbacks.print_callbacks import PrintCallbacks

# (Optional) set up callbacks for debugging/printing
callback_manager = CallbackManager()
callback_manager.add_callback(PrintCallbacks(
    print_text_generation=True,
    print_text_streaming=True
))

# Create the API with callbacks if you want to see debug info
api = OpenAIApi(callback_manager=callback_manager)

# We send system + user messages, and request a structured response
response = api.generate_text(
    messages=[system_message, user_message],
    model_name=model_name,
    response_format=AllClustersSummary  # We'll get a structured object
)

# Now 'response' should be a list. 
# In LiteMind’s structured output usage, response[0] might hold the structured object:
all_clusters_summary: AllClustersSummary = response[0][-1].content

# Access the data
for cluster_info in all_clusters_summary.clusters:
    print(f"Cluster: {cluster_info.cluster_id}")
    print(f"Expression highlights: {cluster_info.expression_highlights}")
    print(f"Top Genes: {', '.join(cluster_info.top_genes)}")
    print(f"Motif Enrichment: {cluster_info.motif_enrichment}")
    print(f"Interpretation: {cluster_info.interpretation}\n")

# %%

# %% [markdown]
# ## Example prompting

# %%
# redefine the variable names
df_clusters_groups = cluster_pseudobulk_df
df_clusters_genes = genes_by_counts
df_clusters_motifs = clust_by_motifs
df_motif_info = info_motifs

# %%
# chunk the data for token size issues
import math

def chunk_df_rows(df, chunk_size=100):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]

for idx, chunk in enumerate(chunk_df_rows(df_motif_info, chunk_size=100)):
    user_message.append_text(f"Motif info chunk {idx+1}:\n")
    user_message.append_table(chunk)

# %%
import tiktoken

def approximate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Returns the approximate number of tokens for a given text
    and model, using tiktoken.
    """
    encoder = tiktoken.encoding_for_model(model_name)
    return len(encoder.encode(text))

# Example usage:
csv_str = df_clusters_groups.to_csv(index=False)

num_tokens = approximate_tokens(csv_str, model_name="gpt-3.5-turbo")
byte_len = len(csv_str.encode("utf-8"))

print(f"Approx. {num_tokens} tokens, {byte_len} bytes in df_clusters_groups CSV.")

# %%
# Example usage:
csv_str = leiden_gene_lists_df.to_csv(index=False)

num_tokens = approximate_tokens(csv_str, model_name="gpt-4")
byte_len = len(csv_str.encode("utf-8"))

print(f"Approx. {num_tokens} tokens, {byte_len} bytes in df_clusters_groups CSV.")

# %%
df_clusters_genes

# %%
# Example usage:
csv_str = motif_factor_df.to_csv(index=False)

num_tokens = approximate_tokens(csv_str, model_name="gpt-4")
byte_len = len(csv_str.encode("utf-8"))

print(f"Approx. {num_tokens} tokens, {byte_len} bytes in df_clusters_groups CSV.")

# %%
# Example usage:
csv_str = df_motif_info.to_csv(index=False)

num_tokens = approximate_tokens(csv_str, model_name="gpt-4")
byte_len = len(csv_str.encode("utf-8"))

print(f"Approx. {num_tokens} tokens, {byte_len} bytes in df_clusters_groups CSV.")

# %%
motif_factor_df

# %%
# 1) Fill missing values with empty strings, so we don't get NaNs
df_motif_info["direct"] = df_motif_info["direct"].fillna("")
df_motif_info["indirect"] = df_motif_info["indirect"].fillna("")

def parse_factor_list(factor_str: str):
    """
    Splits a comma-separated string into a set of factor names,
    stripping whitespace.
    """
    if not factor_str.strip():
        return set()
    return {f.strip() for f in factor_str.split(",") if f.strip()}

# 2) Build a dictionary mapping each motif -> set of all factors (direct + indirect)
motif_to_factors = {}
for motif, row in df_motif_info.iterrows():
    direct_factors = parse_factor_list(row["direct"])
    indirect_factors = parse_factor_list(row["indirect"])
    all_factors = direct_factors.union(indirect_factors)
    motif_to_factors[motif] = all_factors

# 3) Gather all factor names across all motifs
all_factor_names = sorted(set().union(*motif_to_factors.values()))

# 4) Create a new DataFrame with rows = motifs, columns = factor names
motif_factor_df = pd.DataFrame(0, index=df_motif_info.index, columns=all_factor_names)

# 5) Fill in 1 where factor is associated with that motif
for motif, factors in motif_to_factors.items():
    motif_factor_df.loc[motif, list(factors)] = 1

# Now you have a wide, binary matrix of shape:
# (num_motifs) x (total_unique_factors).
print(motif_factor_df.head(5))

# %%
# Example usage:
for df in [df_clusters_groups, leiden_gene_lists_df, df_clusters_motifs, df_motif_info]:
    csv_str = df.to_csv(index=False)

    num_tokens = approximate_tokens(csv_str, model_name="gpt-4")
    byte_len = len(csv_str.encode("utf-8"))

    print(f"Approx. {num_tokens} tokens, {byte_len} bytes in df_clusters_groups CSV.")

# %%
import pandas as pd
from typing import List
from pydantic import BaseModel

# Suppose you have your four DataFrames already loaded:
#   genes_pseudobulk_df, genes_by_counts, clust_by_motifs, info_motifs
# For demonstration, we assume they're already defined.

# 1) Example data structure for final summary output
class ClusterSummary(BaseModel):
    cluster_id: str  # or int, depends on how your cluster IDs are labeled
    expression_highlights: str
    top_genes: List[str]
    motif_enrichment: str
    # Possibly you want to incorporate info_motifs (direct vs. indirect TF hits)
    motif_factor_insights: str

class AllClustersSummary(BaseModel):
    clusters: List[ClusterSummary]

# 2) Import LiteMind
# from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

# 3) Create a system message with general guidance
system_message = Message(role="system", text="""
You are a bioinformatics assistant. 
You will receive four DataFrames describing:
1) Cluster-level pseudobulk gene expression,
2) Per-cluster gene list,
3) Motif enrichment scores per cluster,
4) Motif factor info linking motifs to TF names.
Please integrate these pieces of data to provide a per-cluster summary:
- Summarize expression highlights (which cell states/timepoints are most expressed),
- Identify important genes,
- Discuss enriched motifs and relevant transcription factors,
- Provide an overall interpretation for each cluster.
Make the summary concise and domain-relevant.
""")

# 4) Create a user message that appends the data + instructions
user_message = Message(role="user", text="""
We have four DataFrames. Please analyze them and produce a structured summary
for each cluster, referencing any key insights from these tables:
""")

# Append each DataFrame (assuming the `append_table` method is available 
# and your model or API supports table blocks):
# user_message.append_table(df_clusters_groups, name="genes_pseudobulk_df")
# user_message.append_table(df_clusters_genes, name="genes_by_counts")
# user_message.append_table(df_clusters_motifs, name="clust_by_motifs")
# user_message.append_table(df_motif_info, name="info_motifs")
user_message.append_text("### clusters_vs_groups table:")
user_message.append_table(df_clusters_groups)

user_message.append_text("### cluster vs gene_list table:")
user_message.append_table(leiden_gene_lists_df)
# user_message.append_table(df_clusters_genes)
# user_message.append_table(leiden_gene_lists_df)
# for idx, chunk in enumerate(chunk_df_rows(leiden_gene_lists_df, chunk_size=100)):
#     user_message.append_text(f"gene list chunk {idx+1}:\n")
#     user_message.append_table(chunk)

user_message.append_text("### clusters_vs_motifs table:")
user_message.append_table(df_clusters_motifs.transpose())

user_message.append_text("### motifs information table:")
# user_message.append_table(df_motif_info)
user_message.append_table(df_motif_info)
# for idx, chunk in enumerate(chunk_df_rows(df_motif_info, chunk_size=100)):
#     user_message.append_text(f"motif info chunk {idx+1}:\n")
#     user_message.append_table(chunk)

# 5) Instantiate the CombinedApi and choose a model that supports table input
api = OpenAIApi()
model_name = api.get_best_model([ModelFeatures.TextGeneration])

# 6) Make the request, asking for Pydantic-validated structured output
response = api.generate_text(
    messages=[system_message, user_message],
    model_name=model_name,
    response_format=AllClustersSummary  # We want a structured object parsed into AllClustersSummary
)

# 7) Access the structured results
all_clusters_summary: AllClustersSummary = response[0][-1].content
for cluster in all_clusters_summary.clusters:
    print(f"Cluster: {cluster.cluster_id}")
    print(f"Expression highlights: {cluster.expression_highlights}")
    print(f"Top Genes: {', '.join(cluster.top_genes)}")
    print(f"Motif Enrichment: {cluster.motif_enrichment}")
    print(f"Motif-Factor Insights: {cluster.motif_factor_insights}")
    print("----")


# %%

# %%
import pandas as pd
from typing import List
from pydantic import BaseModel

# LiteMind / OpenAI API imports
from litemind.agent.messages.message import Message
from litemind.apis.providers.openai.openai_api import OpenAIApi
from litemind.apis.model_features import ModelFeatures

# tiktoken for token estimation
import tiktoken

#########################################################
# Function to estimate tokens for a given text + model
#########################################################
def approximate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Returns an approximate number of tokens for 'text'
    using the specified model's tokenizer via tiktoken.
    """
    encoder = tiktoken.encoding_for_model(model_name)
    return len(encoder.encode(text))

#########################################################
# Example Pydantic model for structured response
#########################################################
class ClusterSummary(BaseModel):
    cluster_id: str  # or int, depends on your data
    expression_highlights: str
    top_genes: List[str]
    motif_enrichment: str
    motif_factor_insights: str

class AllClustersSummary(BaseModel):
    clusters: List[ClusterSummary]

#########################################################
# Assume the following DataFrames are loaded:
#   1) df_clusters_groups
#   2) leiden_gene_lists_df (or df_clusters_genes)
#   3) df_clusters_motifs
#   4) df_motif_info
#
# Replace with your actual data-loading code.
#########################################################
# For demonstration, let's pretend these are loaded:
# df_clusters_groups = ...
# leiden_gene_lists_df = ...
# df_clusters_motifs = ...
# df_motif_info = ...

#########################################################
# 1) Convert each DataFrame to CSV and measure token usage
#########################################################
model_for_estimate = "gpt-3.5-turbo"  # or whichever model you plan to use

def df_token_estimate(df: pd.DataFrame, df_name: str):
    csv_str = df.to_csv(index=False)
    num_tokens = approximate_tokens(csv_str, model_name=model_for_estimate)
    print(f"DataFrame '{df_name}' => ~{num_tokens} tokens (CSV).")
    return csv_str, num_tokens

# Example usage:
csv_clusters_groups, tk_clusters_groups = df_token_estimate(df_clusters_groups, "df_clusters_groups")
csv_genes, tk_genes = df_token_estimate(leiden_gene_lists_df, "leiden_gene_lists_df")
csv_motifs, tk_motifs = df_token_estimate(df_clusters_motifs, "df_clusters_motifs")
csv_motif_info, tk_motif_info = df_token_estimate(df_motif_info, "df_motif_info")

#########################################################
# 2) Print a quick summary of token usage
#########################################################
total_tokens = tk_clusters_groups + tk_genes + tk_motifs + tk_motif_info
print("\nSummary of approximate token usage (CSV forms):")
print(f"  df_clusters_groups: {tk_clusters_groups}")
print(f"  leiden_gene_lists_df: {tk_genes}")
print(f"  df_clusters_motifs: {tk_motifs}")
print(f"  df_motif_info: {tk_motif_info}")
print(f"  TOTAL: {total_tokens} tokens")

#########################################################
# 3) If each DataFrame is small enough, you can append
#    them directly. Otherwise, chunk or summarize.
#########################################################
# Example of building messages and sending to OpenAI
system_message = Message(
    role="system",
    text="""
You are a bioinformatics assistant. You will receive multiple DataFrames...
Please provide a structured cluster-by-cluster summary.
"""
)

user_message = Message(
    role="user",
    text="We have four DataFrames. Please analyze them and produce a structured summary:"
)

# Suppose the CSV is not too large:
user_message.append_text("### clusters_vs_groups table:")
user_message.append_text(csv_clusters_groups)  # or user_message.append_table if you have it

user_message.append_text("### cluster vs gene_list table:")
user_message.append_text(csv_genes)

user_message.append_text("### clusters_vs_motifs table:")
user_message.append_text(csv_motifs)

user_message.append_text("### motifs information table:")
user_message.append_text(csv_motif_info)

#########################################################
# 4) Now call the OpenAI API if under limits
#########################################################
api = OpenAIApi()  # from litemind.apis.providers.openai.openai_api import OpenAIApi
model_name = api.get_best_model([ModelFeatures.TextGeneration])  # or "gpt-3.5-turbo"

# If the entire text is under the single-message ~1 MB limit and within total token context,
# you can attempt the generation:
response = api.generate_text(
    messages=[system_message, user_message],
    model_name=model_name,
    response_format=AllClustersSummary
)

all_clusters_summary: AllClustersSummary = response[0][-1].content
for cluster in all_clusters_summary.clusters:
    print(f"Cluster: {cluster.cluster_id}")
    print(f"Expression highlights: {cluster.expression_highlights}")
    print(f"Top Genes: {', '.join(cluster.top_genes)}")
    print(f"Motif Enrichment: {cluster.motif_enrichment}")
    print(f"Motif-Factor Insights: {cluster.motif_factor_insights}")
    print("----")


# %%
