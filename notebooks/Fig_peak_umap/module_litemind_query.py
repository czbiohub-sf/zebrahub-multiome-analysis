"""
Utility functions for LiteMind-based analysis of peak clusters.

This module contains helper functions for processing data, estimating tokens,
managing table sizes, and generating reports for biological cluster analysis.

Author: Yan-Joon Kim
Last updated: 06/16/2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from litemind.apis.model_features import ModelFeatures


def write_to_markdown(content: str, filename: str = "peak_clusters_report.md", mode: str = "a") -> str:
    """
    Write content to a markdown file.
    
    Parameters:
    -----------
    content : str
        Content to write to the markdown file
    filename : str, default "peak_clusters_report.md"
        Name of the markdown file
    mode : str, default "a"
        File opening mode ('a' for append, 'w' for write)
    
    Returns:
    --------
    str : Filename that was written to
    """
    with open(filename, mode, encoding="utf-8") as f:
        f.write(content + "\n\n")
    return filename


def estimate_tokens(text) -> int:
    """
    Rough estimation of token count for text input.
    
    Parameters:
    -----------
    text : str or object
        Text or object to estimate tokens for
    
    Returns:
    --------
    int : Estimated number of tokens (~4 characters per token for English text)
    """
    return len(str(text)) // 4


def get_table_info(df: pd.DataFrame, name: str) -> tuple:
    """
    Get basic information about a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    name : str
        Name of the dataframe for display purposes
    
    Returns:
    --------
    tuple : (shape, estimated_tokens)
    """
    print(f"\n{name} shape: {df.shape}")
    print(f"{name} columns: {list(df.columns)}")
    if len(df.columns) > 10:
        print(f"First 5 columns: {list(df.columns[:5])}")
        print(f"Last 5 columns: {list(df.columns[-5:])}")
    estimated_tokens = estimate_tokens(df.to_string())
    print(f"Estimated tokens for {name}: {estimated_tokens}")
    return df.shape, estimated_tokens


def reduce_table_size(df: pd.DataFrame, max_cols: int = 50, max_rows: int = 100) -> pd.DataFrame:
    """
    Reduce table size by limiting columns and rows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to reduce
    max_cols : int, default 50
        Maximum number of columns to keep
    max_rows : int, default 100
        Maximum number of rows to keep
    
    Returns:
    --------
    pd.DataFrame : Reduced DataFrame
    """
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


def summarize_large_table(df: pd.DataFrame, name: str) -> str:
    """
    Create a summary of a large table instead of including the full table.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to summarize
    name : str
        Name of the dataframe for display purposes
    
    Returns:
    --------
    str : Summary text of the dataframe
    """
    summary = f"""
    {name} Summary:
    - Shape: {df.shape}
    - Columns: {len(df.columns)}
    - Non-zero values: {(df != 0).sum().sum() if df.select_dtypes(include=[np.number]).shape[1] > 0 else 'N/A'}
    - Top 5 columns by variance: {df.var().nlargest(5).index.tolist() if df.select_dtypes(include=[np.number]).shape[1] > 0 else 'N/A'}
    """
    return summary


def convert_clusters_genes_to_lists(df_clusters_genes: pd.DataFrame, 
                                  method: str = 'nonzero', 
                                  threshold: float = 0, 
                                  top_n: Optional[int] = None) -> Dict[int, List[str]]:
    """
    Convert clusters-by-genes dataframe to a more compact representation.
    
    Parameters:
    -----------
    df_clusters_genes : pd.DataFrame
        DataFrame with clusters as rows and genes as columns
    method : str, default 'nonzero'
        Method to use for gene selection:
        - 'nonzero': genes with non-zero values
        - 'threshold': genes above a threshold value
        - 'top_n': top N genes by value
    threshold : float, default 0
        Threshold value when method='threshold'
    top_n : int, optional
        Number of top genes when method='top_n' (default 20 if None)
    
    Returns:
    --------
    Dict[int, List[str]] : Dictionary mapping cluster_id to list of gene names
    
    Raises:
    -------
    ValueError : If method is not one of the supported options
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


def process_cluster_data(cluster_id, 
                        df_clusters_groups: pd.DataFrame,
                        cluster_genes_dict: Dict[int, List[str]],
                        df_clusters_motifs: pd.DataFrame,
                        df_motif_info: pd.DataFrame) -> tuple:
    """
    Process data for a single cluster and prepare it for LLM analysis.
    
    Parameters:
    -----------
    cluster_id : int
        ID of the cluster to process
    df_clusters_groups : pd.DataFrame
        DataFrame with cluster expression across groups
    cluster_genes_dict : Dict[int, List[str]]
        Dictionary mapping cluster IDs to gene lists
    df_clusters_motifs : pd.DataFrame
        DataFrame with cluster motif enrichment
    df_motif_info : pd.DataFrame
        DataFrame with motif information
    
    Returns:
    --------
    tuple : (cluster_groups_data, genes_text, cluster_motifs_data, estimated_tokens)
    """
    # Subset the dataframes for the current cluster
    df_clusters_groups_cluster = df_clusters_groups.loc[cluster_id].to_frame().T
    df_clusters_genes_cluster = cluster_genes_dict[cluster_id]
    df_clusters_motifs_cluster = df_clusters_motifs.loc[cluster_id].to_frame().T

    # Convert df_clusters_genes_cluster to a string
    genes_text = ', '.join(df_clusters_genes_cluster)

    # Estimate tokens for this cluster's input
    cluster_tokens = (estimate_tokens(df_clusters_groups_cluster.to_string()) + 
                     estimate_tokens(df_clusters_genes_cluster) + 
                     estimate_tokens(df_clusters_motifs_cluster.to_string()) + 
                     estimate_tokens(df_motif_info.to_string()))
    
    return df_clusters_groups_cluster, genes_text, df_clusters_motifs_cluster, cluster_tokens 


def check_model_feature(api, model_name: str, feature: ModelFeatures) -> bool:
    """
    Check if a specific model has a particular feature.
    
    Parameters:
    -----------
    api : OpenAIApi
        The LiteMind API instance
    model_name : str
        Name of the model to check
    feature : ModelFeatures
        The feature to check for (e.g., ModelFeatures.WebSearchTool)
    
    Returns:
    --------
    bool : True if the model has the feature, False otherwise
    """
    try:
        features = api.get_model_features(model_name)
        return feature in features
    except Exception as e:
        print(f"Error checking feature for {model_name}: {e}")
        return False


def find_models_with_feature(api, feature: ModelFeatures) -> List[str]:
    """
    Find all models that support a specific feature.
    
    Parameters:
    -----------
    api : OpenAIApi
        The LiteMind API instance
    feature : ModelFeatures
        The feature to search for (e.g., ModelFeatures.WebSearchTool)
    
    Returns:
    --------
    List[str] : List of model names that support the feature
    """
    models_with_feature = []
    for model in api.list_models():
        if check_model_feature(api, model, feature):
            models_with_feature.append(model)
    return models_with_feature


def get_best_model_for_analysis(api, require_web_search: bool = False) -> str:
    """
    Get the best model for biological data analysis.
    
    Parameters:
    -----------
    api : OpenAIApi
        The LiteMind API instance
    require_web_search : bool, default False
        Whether to require WebSearchTool capability
    
    Returns:
    --------
    str : Name of the best model for the analysis
    """
    required_features = [ModelFeatures.TextGeneration]
    if require_web_search:
        required_features.append(ModelFeatures.WebSearchTool)
    
    try:
        return api.get_best_model(required_features)
    except Exception as e:
        print(f"Error getting best model: {e}")
        # Fallback to a common model
        return "gpt-4o" if not require_web_search else "gpt-4o-search-preview"


def print_model_capabilities(api, models: Optional[List[str]] = None):
    """
    Print capabilities of specified models or all available models.
    
    Parameters:
    -----------
    api : OpenAIApi
        The LiteMind API instance
    models : List[str], optional
        List of model names to check. If None, checks all available models.
    """
    if models is None:
        models = api.list_models()
    
    print("=== Model Capabilities ===")
    for model in models:
        try:
            features = api.get_model_features(model)
            has_web_search = ModelFeatures.WebSearchTool in features
            has_text_gen = ModelFeatures.TextGeneration in features
            
            print(f"\n{model}:")
            print(f"  ✓ TextGeneration: {has_text_gen}")
            print(f"  ✓ WebSearchTool: {has_web_search}")
            print(f"  All features: {[f.name for f in features]}")
        except Exception as e:
            print(f"\n{model}: Error - {e}") 