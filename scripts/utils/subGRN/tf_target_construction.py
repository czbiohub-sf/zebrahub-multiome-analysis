# Peak Cluster to GRN Pipeline - TF-Target Construction Module
# Author: YangJoon Kim
# Date: 2025-06-25
# Description: Build putative TF-target gene relationships from motifs and genes

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
import os

def create_motif_tf_mapping(
    motif_database_path: Optional[str] = None,
    motif_tf_dict: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[str]]:
    """
    Create mapping from motifs to transcription factors.
    
    Parameters:
    -----------
    motif_database_path : str, optional
        Path to motif database file (tab-separated: motif_name, tf_name)
    motif_tf_dict : Dict[str, List[str]], optional
        Pre-computed motif-TF mapping dictionary
        
    Returns:
    --------
    motif_tf_dict : Dict[str, List[str]]
        {motif_name: [list_of_tfs]}
    """
    if motif_tf_dict is not None:
        return motif_tf_dict
    
    if motif_database_path is None:
        warnings.warn("No motif database provided. Using motif names as TF names.")
        return {}
    
    if not os.path.exists(motif_database_path):
        raise FileNotFoundError(f"Motif database file not found: {motif_database_path}")
    
    # Read motif database file
    try:
        # Assume tab-separated file with columns: motif_name, tf_name
        df = pd.read_csv(motif_database_path, sep='\t', header=0)
        
        if 'motif_name' not in df.columns or 'tf_name' not in df.columns:
            # Try alternative column names
            if len(df.columns) >= 2:
                df.columns = ['motif_name', 'tf_name'] + list(df.columns[2:])
            else:
                raise ValueError("Expected at least 2 columns in motif database file")
        
        # Create mapping dictionary
        motif_tf_mapping = {}
        for _, row in df.iterrows():
            motif = str(row['motif_name'])
            tf = str(row['tf_name'])
            
            if motif not in motif_tf_mapping:
                motif_tf_mapping[motif] = []
            motif_tf_mapping[motif].append(tf)
        
        print(f"Loaded motif-TF mapping for {len(motif_tf_mapping)} motifs")
        return motif_tf_mapping
        
    except Exception as e:
        raise ValueError(f"Error reading motif database file: {str(e)}")

def infer_motif_tf_mapping_from_names(motif_names: List[str]) -> Dict[str, List[str]]:
    """
    Infer TF names from motif names using common naming conventions.
    
    Parameters:
    -----------
    motif_names : List[str]
        List of motif names
        
    Returns:
    --------
    Dict[str, List[str]]
        {motif_name: [inferred_tf_names]}
    """
    motif_tf_mapping = {}
    
    for motif in motif_names:
        # Common patterns in motif naming
        tf_candidates = []
        
        # Pattern 1: Direct TF name (e.g., "SOX2", "NANOG")
        if motif.upper() in motif:
            tf_candidates.append(motif.upper())
        
        # Pattern 2: TF_motif format (e.g., "SOX2_1", "NANOG_MA0259.1")
        if '_' in motif:
            potential_tf = motif.split('_')[0].upper()
            tf_candidates.append(potential_tf)
        
        # Pattern 3: Remove common suffixes
        for suffix in ['.1', '.2', '_HUMAN', '_MOUSE', '_ZF']:
            if motif.endswith(suffix):
                potential_tf = motif.replace(suffix, '').upper()
                tf_candidates.append(potential_tf)
        
        # Remove duplicates and use motif name as fallback
        tf_candidates = list(set(tf_candidates)) if tf_candidates else [motif]
        motif_tf_mapping[motif] = tf_candidates
    
    return motif_tf_mapping

def extract_cluster_associated_genes(
    clusters_genes_df: pd.DataFrame,
    method: str = "correlation",
    correlation_threshold: float = 0.5,
    top_n_genes: Optional[int] = None,
    percentile_threshold: float = 75
) -> Dict[str, List[str]]:
    """
    Extract high-confidence associated genes for each cluster.
    
    Parameters:
    -----------
    clusters_genes_df : pd.DataFrame
        Clusters x genes matrix (correlation scores, binary, or expression)
    method : str
        Method to use: "correlation", "top_n", "percentile", or "binary"
    correlation_threshold : float
        Minimum correlation/association score (for correlation method)
    top_n_genes : int, optional
        Number of top genes to select per cluster (for top_n method)
    percentile_threshold : float
        Percentile threshold for gene selection (for percentile method)
        
    Returns:
    --------
    cluster_genes_dict : Dict[str, List[str]]
        {cluster_id: [list_of_associated_genes]}
    """
    cluster_genes = {}
    
    for cluster in clusters_genes_df.index:
        cluster_data = clusters_genes_df.loc[cluster]
        
        if method == "correlation":
            # Select genes above correlation threshold
            associated_genes = cluster_data[cluster_data >= correlation_threshold].index.tolist()
            
        elif method == "top_n":
            if top_n_genes is None:
                raise ValueError("top_n_genes must be specified for top_n method")
            # Select top N genes with highest values
            associated_genes = cluster_data.nlargest(top_n_genes).index.tolist()
            
        elif method == "percentile":
            # Select genes above percentile threshold
            threshold_value = np.percentile(cluster_data.values, percentile_threshold)
            associated_genes = cluster_data[cluster_data >= threshold_value].index.tolist()
            
        elif method == "binary":
            # Assume binary matrix, select genes with value > 0
            associated_genes = cluster_data[cluster_data > 0].index.tolist()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        cluster_genes[str(cluster)] = associated_genes
    
    # Print summary
    gene_counts = [len(genes) for genes in cluster_genes.values()]
    print(f"Extracted associated genes per cluster: mean={np.mean(gene_counts):.1f}, "
          f"median={np.median(gene_counts):.1f}, range={min(gene_counts)}-{max(gene_counts)}")
    
    return cluster_genes

def build_cluster_tf_target_matrix(
    cluster_differential_motifs: Dict[str, List[str]],
    cluster_associated_genes: Dict[str, List[str]], 
    motif_tf_mapping: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Build putative TF-target matrices for each cluster.
    
    Parameters:
    -----------
    cluster_differential_motifs : Dict[str, List[str]]
        {cluster_id: [differential_motifs]}
    cluster_associated_genes : Dict[str, List[str]]
        {cluster_id: [associated_genes]}
    motif_tf_mapping : Dict[str, List[str]]
        {motif_name: [tfs]}
        
    Returns:
    --------
    cluster_tf_target_matrices : Dict[str, pd.DataFrame]
        {cluster_id: TFs x target_genes binary matrix}
    """
    cluster_tf_target_matrices = {}
    
    for cluster in cluster_differential_motifs.keys():
        if cluster not in cluster_associated_genes:
            warnings.warn(f"Cluster {cluster} not found in associated genes")
            continue
        
        # Get differential motifs and associated genes for this cluster
        motifs = cluster_differential_motifs[cluster]
        genes = cluster_associated_genes[cluster]
        
        if not motifs or not genes:
            warnings.warn(f"No motifs or genes found for cluster {cluster}")
            continue
        
        # Extract TFs from motifs
        cluster_tfs = set()
        for motif in motifs:
            if motif in motif_tf_mapping:
                cluster_tfs.update(motif_tf_mapping[motif])
            else:
                # Use motif name as TF name if no mapping available
                cluster_tfs.add(motif)
        
        cluster_tfs = sorted(list(cluster_tfs))
        
        # Create binary TF-target matrix
        # All combinations of TFs and genes are considered potential targets
        tf_target_matrix = pd.DataFrame(
            1,  # All connections set to 1 (binary)
            index=cluster_tfs,
            columns=genes
        )
        
        cluster_tf_target_matrices[cluster] = tf_target_matrix
        
        print(f"Cluster {cluster}: {len(cluster_tfs)} TFs, {len(genes)} target genes")
    
    return cluster_tf_target_matrices

def refine_tf_target_relationships(
    tf_target_matrix: pd.DataFrame,
    expression_data: Optional[pd.DataFrame] = None,
    tf_expression_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Refine TF-target relationships based on expression data.
    
    Parameters:
    -----------
    tf_target_matrix : pd.DataFrame
        Binary TF-target matrix
    expression_data : pd.DataFrame, optional
        Expression data (TFs/genes x samples)
    tf_expression_threshold : float
        Minimum expression threshold for TFs
        
    Returns:
    --------
    pd.DataFrame
        Refined TF-target matrix
    """
    if expression_data is None:
        return tf_target_matrix
    
    refined_matrix = tf_target_matrix.copy()
    
    # Filter out TFs with low expression
    expressed_tfs = []
    for tf in tf_target_matrix.index:
        if tf in expression_data.index:
            if expression_data.loc[tf].mean() >= tf_expression_threshold:
                expressed_tfs.append(tf)
        else:
            # Keep TF if expression data not available
            expressed_tfs.append(tf)
    
    refined_matrix = refined_matrix.loc[expressed_tfs]
    
    print(f"Filtered TFs by expression: {len(tf_target_matrix)} -> {len(refined_matrix)}")
    
    return refined_matrix

def compute_tf_target_confidence_scores(
    cluster_tf_target_matrices: Dict[str, pd.DataFrame],
    clusters_motifs_df: pd.DataFrame,
    cluster_differential_motifs: Dict[str, List[str]],
    motif_tf_mapping: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Compute confidence scores for TF-target relationships based on motif enrichment.
    
    Parameters:
    -----------
    cluster_tf_target_matrices : Dict[str, pd.DataFrame]
        Binary TF-target matrices per cluster
    clusters_motifs_df : pd.DataFrame
        Clusters x motifs matrix with enrichment scores
    cluster_differential_motifs : Dict[str, List[str]]
        Differential motifs per cluster
    motif_tf_mapping : Dict[str, List[str]]
        Motif to TF mapping
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        TF-target matrices with confidence scores instead of binary values
    """
    confidence_matrices = {}
    
    for cluster, tf_target_matrix in cluster_tf_target_matrices.items():
        confidence_matrix = tf_target_matrix.copy().astype(float)
        
        # Get motif enrichment scores for this cluster
        if cluster in clusters_motifs_df.index:
            cluster_motif_scores = clusters_motifs_df.loc[cluster]
            
            for tf in tf_target_matrix.index:
                # Find motifs associated with this TF
                tf_motifs = []
                for motif, tfs in motif_tf_mapping.items():
                    if tf in tfs and motif in cluster_differential_motifs[cluster]:
                        tf_motifs.append(motif)
                
                if tf_motifs:
                    # Use mean motif enrichment score as confidence
                    tf_confidence = cluster_motif_scores[tf_motifs].mean()
                    confidence_matrix.loc[tf] *= tf_confidence
        
        confidence_matrices[cluster] = confidence_matrix
    
    return confidence_matrices

def get_tf_target_summary(
    cluster_tf_target_matrices: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create summary statistics for TF-target relationships across clusters.
    
    Parameters:
    -----------
    cluster_tf_target_matrices : Dict[str, pd.DataFrame]
        TF-target matrices per cluster
        
    Returns:
    --------
    pd.DataFrame
        Summary table with cluster, TF, and target gene statistics
    """
    summary_data = []
    
    for cluster, matrix in cluster_tf_target_matrices.items():
        n_tfs = len(matrix.index)
        n_targets = len(matrix.columns)
        n_edges = matrix.sum().sum() if hasattr(matrix.sum(), 'sum') else matrix.values.sum()
        
        summary_data.append({
            'cluster': cluster,
            'n_tfs': n_tfs,
            'n_target_genes': n_targets,
            'n_edges': n_edges,
            'edge_density': n_edges / (n_tfs * n_targets) if (n_tfs * n_targets) > 0 else 0
        })
    
    return pd.DataFrame(summary_data)

def save_tf_target_matrices(
    cluster_tf_target_matrices: Dict[str, pd.DataFrame],
    output_dir: str,
    file_prefix: str = "tf_target_matrix"
):
    """
    Save TF-target matrices to files.
    
    Parameters:
    -----------
    cluster_tf_target_matrices : Dict[str, pd.DataFrame]
        TF-target matrices per cluster
    output_dir : str
        Output directory
    file_prefix : str
        Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for cluster, matrix in cluster_tf_target_matrices.items():
        filename = f"{file_prefix}_cluster_{cluster}.csv"
        filepath = os.path.join(output_dir, filename)
        matrix.to_csv(filepath)
        print(f"Saved TF-target matrix for cluster {cluster} to {filepath}")

def load_tf_target_matrices(
    input_dir: str,
    file_prefix: str = "tf_target_matrix"
) -> Dict[str, pd.DataFrame]:
    """
    Load TF-target matrices from files.
    
    Parameters:
    -----------
    input_dir : str
        Input directory
    file_prefix : str
        Prefix of input files
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        TF-target matrices per cluster
    """
    cluster_tf_target_matrices = {}
    
    # Find all files matching the pattern
    import glob
    pattern = os.path.join(input_dir, f"{file_prefix}_cluster_*.csv")
    files = glob.glob(pattern)
    
    for filepath in files:
        # Extract cluster ID from filename
        filename = os.path.basename(filepath)
        cluster = filename.replace(f"{file_prefix}_cluster_", "").replace(".csv", "")
        
        # Load matrix
        matrix = pd.read_csv(filepath, index_col=0)
        cluster_tf_target_matrices[cluster] = matrix
        
        print(f"Loaded TF-target matrix for cluster {cluster}")
    
    return cluster_tf_target_matrices