"""
SubGRN Analysis Utilities

Modular toolkit for extracting and analyzing sub-Gene Regulatory Networks (subGRNs)
from zebrafish developmental multiome data.

Modules:
- data_loading: Data loading and validation utilities
- mesh_construction: TF-gene mesh network construction
- extraction: SubGRN extraction from full GRNs
- analysis: Temporal and spatial subGRN analysis
- visualization: Network visualization helper functions
- temporal_dynamics: Temporal dynamics scoring and ranking
- similarity_analysis: Cluster similarity and block detection
- enrichment: TF enrichment analysis in similarity blocks
- cluster_specificity: Cluster specificity metrics

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

__version__ = "1.0.0"

# Import key functions for convenient access
from .subgrn_data_loading import load_grn_dict_pathlib, load_peak_adata
from .subgrn_mesh_construction import create_all_cluster_meshes
from .subgrn_extraction import extract_subgrn_metrics
from .subgrn_temporal_dynamics import rank_clusters_by_temporal_dynamics
from .subgrn_similarity_analysis import find_dense_similarity_regions
from .subgrn_enrichment import analyze_tf_enrichment_in_blocks
from .subgrn_cluster_specificity import calculate_cluster_specificity

__all__ = [
    'load_grn_dict_pathlib',
    'load_peak_adata',
    'create_all_cluster_meshes',
    'extract_subgrn_metrics',
    'rank_clusters_by_temporal_dynamics',
    'find_dense_similarity_regions',
    'analyze_tf_enrichment_in_blocks',
    'calculate_cluster_specificity',
]
