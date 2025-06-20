#!/usr/bin/env python3
"""
Script to re-run cluster 0 processing that failed in the original run
"""

import os
import pandas as pd
import numpy as np
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.messages.message import Message

# Import utility functions from the module
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

def main():
    # Check OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    print(f"OpenAI API key available: {'Yes' if openai_api_key else 'No'}")
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Load the data
    print("Loading data...")
    df_clusters_groups = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_pseudobulk.csv",
                                     index_col=0)
    df_clusters_genes = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_assoc_genes.csv",
                                    index_col=0)
    df_clusters_motifs = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/leiden_by_motifs_maelstrom.csv",
                                     index_col=0)
    df_motif_info = pd.read_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_640K_leiden_coarse_cisBP_v2_danio_rerio_output/info_cisBP_v2_danio_rerio_motif_factors.csv", 
                               index_col=0)
    
    # Parse the genes for each cluster
    cluster_genes_dict = convert_clusters_genes_to_lists(df_clusters_genes, method='nonzero')
    
    # Initialize the OpenAI API and toolset
    api = OpenAIApi()
    toolset = ToolSet()
    toolset.add_builtin_web_search_tool()
    
    # Process only cluster 0
    cluster_id = 0
    
    print(f"\n=== Processing Cluster {cluster_id} (Re-run) ===")
    
    # Check if cluster 0 exists in the data
    if cluster_id not in df_clusters_groups.index:
        print(f"Error: Cluster {cluster_id} not found in data")
        return
    
    # Process cluster data
    df_clusters_groups_cluster, genes_text, df_clusters_motifs_cluster, cluster_tokens = process_cluster_data(
        cluster_id, df_clusters_groups, cluster_genes_dict, df_clusters_motifs, df_motif_info
    )
    
    print(f"Estimated tokens for cluster {cluster_id}: {cluster_tokens}")
    
    # Skip if too large (adjust threshold as needed)
    if cluster_tokens > 100000:
        print(f"Cluster {cluster_id} is too large ({cluster_tokens} tokens). Consider reducing data size.")
        return
    
    # Create the agent with web search capabilities
    try:
        agent = Agent(api=api, model_name="gpt-4.1", toolset=toolset)
    except Exception as e:
        print(f"Error creating agent with gpt-4.1, trying default model: {e}")
        agent = Agent(api=api, toolset=toolset)
    
    # Add system message
    agent.append_system_message("You are an expert developmental biologist specialized in vertebrate embryogenesis, especially zebrafish embryogenesis.")
    
    # Create the analysis message
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
    
    # Process the request with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to process cluster {cluster_id}")
            response = agent(message)
            
            # Success - write to markdown
            cluster_report = f"## Cluster {cluster_id} (Re-run)\n\n" + str(response) + "\n\n"
            
            # Write to a separate file for the re-run
            with open("cluster_0_rerun_report.md", "w", encoding="utf-8") as f:
                f.write("# Cluster 0 Re-run Report\n\n")
                f.write(cluster_report)
            
            # Also append to the main report file, replacing the error
            print("Writing successful result to files...")
            
            # Print the response to console
            print(f"\n{cluster_report}")
            
            print(f"Successfully processed cluster {cluster_id}!")
            return
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed for cluster {cluster_id}: {str(e)}"
            print(error_msg)
            
            if attempt == max_retries - 1:
                # Final attempt failed
                final_error = f"## Cluster {cluster_id} (Re-run Failed)\n\n{error_msg}\n\n"
                with open("cluster_0_rerun_report.md", "w", encoding="utf-8") as f:
                    f.write("# Cluster 0 Re-run Report\n\n")
                    f.write(final_error)
                print("All retry attempts failed.")
                return
            else:
                # Wait a bit before retry
                import time
                time.sleep(5)

if __name__ == "__main__":
    main() 