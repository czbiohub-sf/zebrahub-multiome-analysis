# Peak Cluster to GRN Pipeline Package
# Author: YangJoon Kim
# Date: 2025-06-25
# Description: Package for extracting sub-GRNs from peak cluster analysis

"""
subGRN: Peak Cluster to Gene Regulatory Network Analysis Package

This package provides tools for analyzing peak clusters and extracting 
meaningful sub-GRNs (Gene Regulatory Networks) from larger CellOracle GRNs.

Main modules:
- data_processing: Handle input data and cluster-level aggregation
- differential_analysis: Identify differentially enriched motifs per cluster
- tf_target_construction: Build putative TF-target gene relationships
- grn_extraction: Extract meaningful sub-GRNs from large GRNs
- utilities: Helper functions and visualization tools
- main_pipeline: Main orchestration class for the complete workflow

Example usage:
    >>> from subGRN import PeakClusterGRNPipeline
    >>> from subGRN.utilities import create_default_config
    >>> 
    >>> config = create_default_config()
    >>> config['input_data']['peaks_motifs_path'] = 'path/to/peaks_motifs.h5ad'
    >>> config['input_data']['peaks_genes_path'] = 'path/to/peaks_genes.h5ad'
    >>> config['grn_extraction']['grn_path'] = 'path/to/celloracle_grns/'
    >>> 
    >>> pipeline = PeakClusterGRNPipeline(config)
    >>> results = pipeline.run_full_pipeline()
"""

__version__ = "1.0.0"
__author__ = "YangJoon Kim"
__email__ = "your.email@domain.com"  # Update with actual email

# Import main classes and functions for easy access
from .main_pipeline import PeakClusterGRNPipeline, create_example_config

# Import key functions from each module
from .data_processing import (
    aggregate_peaks_by_clusters,
    validate_input_data,
    get_cluster_peak_counts
)

from .differential_analysis import (
    compute_differential_motifs,
    compute_motif_enrichment_scores,
    plot_motif_heatmap
)

from .tf_target_construction import (
    create_motif_tf_mapping,
    extract_cluster_associated_genes,
    build_cluster_tf_target_matrix
)

from .grn_extraction import (
    load_celloracle_grn,
    extract_subgrn_from_putative,
    validate_subgrn_enrichment,
    merge_cluster_subgrns
)

from .utilities import (
    save_results,
    load_results,
    plot_cluster_motif_heatmap,
    plot_tf_target_network,
    compute_overlap_statistics,
    create_default_config
)

# Define what gets imported with "from subGRN import *"
__all__ = [
    # Main pipeline
    'PeakClusterGRNPipeline',
    'create_example_config',
    
    # Data processing
    'aggregate_peaks_by_clusters',
    'validate_input_data',
    'get_cluster_peak_counts',
    
    # Differential analysis
    'compute_differential_motifs',
    'compute_motif_enrichment_scores',
    
    # TF-target construction
    'create_motif_tf_mapping',
    'extract_cluster_associated_genes',
    'build_cluster_tf_target_matrix',
    
    # GRN extraction
    'load_celloracle_grn',
    'extract_subgrn_from_putative',
    'validate_subgrn_enrichment',
    'merge_cluster_subgrns',
    
    # Utilities
    'save_results',
    'load_results',
    'plot_cluster_motif_heatmap',
    'plot_tf_target_network',
    'compute_overlap_statistics',
    'create_default_config'
]

# Package metadata
__description__ = "Peak Cluster to Gene Regulatory Network Analysis Package"
__url__ = "https://github.com/your-username/zebrahub-multiome-analysis"  # Update with actual URL
__license__ = "MIT"

# Version info tuple
VERSION_INFO = tuple(map(int, __version__.split('.')))

def get_version():
    """Return the package version as a string."""
    return __version__

def get_info():
    """Return package information."""
    info = {
        'name': 'subGRN',
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'url': __url__,
        'license': __license__
    }
    return info

# Print info when package is imported
print(f"subGRN v{__version__} - Peak Cluster to GRN Analysis Package")
print(f"For documentation and examples, visit: {__url__}")

# Optional: Check for required dependencies
def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'pandas',
        'numpy', 
        'scanpy',
        'matplotlib',
        'seaborn',
        'networkx',
        'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Warning: Missing required packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
    
    return len(missing_packages) == 0

# Check dependencies on import (optional)
# check_dependencies()

# Module docstring for help()
if __name__ == '__main__':
    help(__name__)