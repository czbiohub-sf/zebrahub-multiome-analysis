# Peak Cluster to GRN Pipeline - Main Pipeline Module
# Author: YangJoon Kim
# Date: 2025-06-25
# Description: Orchestrate the entire peak cluster to GRN analysis workflow

import pandas as pd
import numpy as np
import scanpy as sc
from typing import Dict, List, Optional, Any
import os
import warnings
from datetime import datetime

# Import subGRN modules
from . import data_processing as dp
from . import differential_analysis as da
from . import tf_target_construction as ttc
from . import grn_extraction as ge
from . import utilities as utils

class PeakClusterGRNPipeline:
    """
    Main pipeline class to orchestrate peak cluster to GRN analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration parameters.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary with all parameters
        """
        self.config = config
        self.results = {}
        self.start_time = datetime.now()
        
        # Validate configuration
        if not utils.validate_config(config):
            raise ValueError("Invalid configuration")
        
        # Create output directory
        self.output_dir = config['output']['results_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Initialized PeakClusterGRNPipeline")
        print(f"Output directory: {self.output_dir}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing all analysis results
        """
        print("=" * 60)
        print("Starting Peak Cluster to GRN Analysis Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Data processing
            print("\nStep 1: Processing input data...")
            clusters_motifs_df, clusters_genes_df = self.process_input_data()
            
            # Step 2: Differential analysis  
            print("\nStep 2: Computing differential motifs...")
            differential_motifs = self.compute_differential_motifs(clusters_motifs_df)
            
            # Step 3: TF-target construction
            print("\nStep 3: Building TF-target relationships...")
            tf_target_matrices = self.build_tf_target_relationships(
                differential_motifs, clusters_genes_df
            )
            
            # Step 4: GRN extraction
            print("\nStep 4: Extracting sub-GRNs...")
            subgrns = self.extract_subgrns(tf_target_matrices)
            
            # Step 5: Validation and analysis
            print("\nStep 5: Validating results...")
            validation_results = self.validate_results(subgrns, tf_target_matrices)
            
            # Step 6: Generate visualizations
            print("\nStep 6: Generating visualizations...")
            self.generate_visualizations(
                clusters_motifs_df, 
                differential_motifs, 
                tf_target_matrices, 
                subgrns
            )
            
            # Step 7: Save results
            print("\nStep 7: Saving results...")
            self.save_all_results()
            
            # Generate summary report
            self.generate_summary_report()
            
            print(f"\nPipeline completed successfully!")
            print(f"Total runtime: {datetime.now() - self.start_time}")
            
            return self.results
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            raise e
    
    def process_input_data(self) -> tuple:
        """
        Step 1: Process input data.
        
        Returns:
        --------
        tuple
            (clusters_motifs_df, clusters_genes_df)
        """
        config = self.config['input_data']
        
        # Load data
        print(f"Loading peaks-motifs data from: {config['peaks_motifs_path']}")
        peaks_motifs_adata = sc.read_h5ad(config['peaks_motifs_path'])
        
        print(f"Loading peaks-genes data from: {config['peaks_genes_path']}")
        peaks_genes_adata = sc.read_h5ad(config['peaks_genes_path'])
        
        # Validate input data
        if not dp.validate_input_data(peaks_motifs_adata, peaks_genes_adata):
            raise ValueError("Input data validation failed")
        
        # Aggregate by clusters
        clusters_motifs_df, clusters_genes_df = dp.aggregate_peaks_by_clusters(
            peaks_motifs_adata, 
            peaks_genes_adata, 
            cluster_resolution=config['cluster_resolution']
        )
        
        # Compute statistics
        stats = dp.compute_cluster_statistics(clusters_motifs_df, clusters_genes_df)
        
        # Store results
        self.results['data_processing'] = {
            'clusters_motifs_df': clusters_motifs_df,
            'clusters_genes_df': clusters_genes_df,
            'statistics': stats,
            'clusters_motifs_shape': clusters_motifs_df.shape,
            'clusters_genes_shape': clusters_genes_df.shape
        }
        
        # Save intermediate results if requested
        if self.config['output']['save_intermediate']:
            utils.save_results(
                clusters_motifs_df, 
                os.path.join(self.output_dir, 'clusters_motifs.pkl'),
                format=self.config['output']['file_format']
            )
            utils.save_results(
                clusters_genes_df, 
                os.path.join(self.output_dir, 'clusters_genes.pkl'),
                format=self.config['output']['file_format']
            )
        
        return clusters_motifs_df, clusters_genes_df
    
    def compute_differential_motifs(self, clusters_motifs_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Step 2: Compute differential motifs.
        
        Parameters:
        -----------
        clusters_motifs_df : pd.DataFrame
            Clusters x motifs matrix
            
        Returns:
        --------
        Dict[str, List[str]]
            Differential motifs per cluster
        """
        config = self.config['differential_analysis']
        
        # Compute differential motifs
        differential_motifs = da.compute_differential_motifs(
            clusters_motifs_df,
            method=config['method'],
            top_n=config.get('top_n', 10),
            fold_change_threshold=config.get('fold_change_threshold', 2.0),
            pvalue_threshold=config.get('pvalue_threshold', 0.001),
            fdr_correction=config.get('fdr_correction', True)
        )
        
        # Compute enrichment scores
        enrichment_scores = da.compute_motif_enrichment_scores(clusters_motifs_df)
        
        # Create summary
        summary = da.summarize_differential_results(differential_motifs, clusters_motifs_df)
        
        # Store results
        self.results['differential_analysis'] = {
            'differential_motifs': differential_motifs,
            'enrichment_scores': enrichment_scores,
            'summary': summary,
            'n_clusters': len(differential_motifs),
            'total_differential_motifs': sum(len(motifs) for motifs in differential_motifs.values())
        }
        
        # Save intermediate results
        if self.config['output']['save_intermediate']:
            utils.save_results(
                differential_motifs, 
                os.path.join(self.output_dir, 'differential_motifs.pkl'),
                format=self.config['output']['file_format']
            )
            summary.to_csv(os.path.join(self.output_dir, 'differential_motifs_summary.csv'))
        
        return differential_motifs
    
    def build_tf_target_relationships(
        self, 
        differential_motifs: Dict[str, List[str]], 
        clusters_genes_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Step 3: Build TF-target relationships.
        
        Parameters:
        -----------
        differential_motifs : Dict[str, List[str]]
            Differential motifs per cluster
        clusters_genes_df : pd.DataFrame
            Clusters x genes matrix
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            TF-target matrices per cluster
        """
        config = self.config['tf_target_construction']
        
        # Create motif-TF mapping
        if config.get('motif_database_path'):
            motif_tf_mapping = ttc.create_motif_tf_mapping(
                motif_database_path=config['motif_database_path']
            )
        else:
            # Infer from motif names
            all_motifs = set()
            for motifs in differential_motifs.values():
                all_motifs.update(motifs)
            motif_tf_mapping = ttc.infer_motif_tf_mapping_from_names(list(all_motifs))
        
        # Extract cluster-associated genes
        cluster_associated_genes = ttc.extract_cluster_associated_genes(
            clusters_genes_df,
            method=config.get('gene_association_method', 'correlation'),
            correlation_threshold=config.get('correlation_threshold', 0.5),
            top_n_genes=config.get('top_n_genes', None)
        )
        
        # Build TF-target matrices
        tf_target_matrices = ttc.build_cluster_tf_target_matrix(
            differential_motifs,
            cluster_associated_genes,
            motif_tf_mapping
        )
        
        # Compute confidence scores
        if 'clusters_motifs_df' in self.results['data_processing']:
            clusters_motifs_df = self.results['data_processing']['clusters_motifs_df']
            confidence_matrices = ttc.compute_tf_target_confidence_scores(
                tf_target_matrices,
                clusters_motifs_df,
                differential_motifs,
                motif_tf_mapping
            )
        else:
            confidence_matrices = tf_target_matrices
        
        # Create summary
        summary = ttc.get_tf_target_summary(tf_target_matrices)
        
        # Store results
        self.results['tf_target_construction'] = {
            'tf_target_matrices': tf_target_matrices,
            'confidence_matrices': confidence_matrices,
            'motif_tf_mapping': motif_tf_mapping,
            'cluster_associated_genes': cluster_associated_genes,
            'summary': summary,
            'n_tf_target_matrices': len(tf_target_matrices)
        }
        
        # Save intermediate results
        if self.config['output']['save_intermediate']:
            ttc.save_tf_target_matrices(
                tf_target_matrices, 
                os.path.join(self.output_dir, 'tf_target_matrices'),
                'tf_target_matrix'
            )
            summary.to_csv(os.path.join(self.output_dir, 'tf_target_summary.csv'))
        
        return tf_target_matrices
    
    def extract_subgrns(self, tf_target_matrices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 4: Extract sub-GRNs.
        
        Parameters:
        -----------
        tf_target_matrices : Dict[str, pd.DataFrame]
            TF-target matrices per cluster
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Sub-GRNs per cluster
        """
        config = self.config['grn_extraction']
        
        # Load full GRN (assuming single GRN for now)
        full_grn_df = ge.load_celloracle_grn(
            grn_path=config['grn_path'],
            cell_type=config.get('cell_type', None),
            timepoint=config.get('timepoint', None)
        )
        
        # Extract sub-GRNs for each cluster
        cluster_subgrns = {}
        validation_results = {}
        
        for cluster, tf_target_matrix in tf_target_matrices.items():
            print(f"Extracting sub-GRN for cluster {cluster}...")
            
            # Extract sub-GRN
            subgrn = ge.extract_subgrn_from_putative(
                full_grn_df,
                tf_target_matrix,
                edge_strength_threshold=config.get('edge_strength_threshold', 0.1),
                keep_only_putative=config.get('keep_only_putative', True)
            )
            
            if not subgrn.empty:
                cluster_subgrns[cluster] = subgrn
                
                # Validate sub-GRN
                validation = ge.validate_subgrn_enrichment(
                    subgrn, full_grn_df, tf_target_matrix
                )
                validation_results[cluster] = validation
        
        # Merge cluster sub-GRNs if requested
        if config.get('merge_clusters', False):
            merged_grn = ge.merge_cluster_subgrns(
                cluster_subgrns,
                method=config.get('merge_method', 'union')
            )
            cluster_subgrns['merged'] = merged_grn
        
        # Store results
        self.results['grn_extraction'] = {
            'cluster_subgrns': cluster_subgrns,
            'validation_results': validation_results,
            'full_grn_shape': full_grn_df.shape,
            'n_subgrns': len(cluster_subgrns)
        }
        
        # Save intermediate results
        if self.config['output']['save_intermediate']:
            ge.save_subgrn_results(
                cluster_subgrns, 
                os.path.join(self.output_dir, 'subgrns'),
                'subgrn'
            )
        
        return cluster_subgrns
    
    def validate_results(
        self, 
        subgrns: Dict[str, pd.DataFrame], 
        tf_target_matrices: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Step 5: Validate results.
        
        Parameters:
        -----------
        subgrns : Dict[str, pd.DataFrame]
            Sub-GRNs per cluster
        tf_target_matrices : Dict[str, pd.DataFrame]
            TF-target matrices per cluster
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        validation_results = {}
        
        # Compute network statistics for each sub-GRN
        for cluster, subgrn in subgrns.items():
            if cluster == 'merged':
                continue  # Skip merged GRN for individual cluster validation
                
            network_stats = utils.compute_network_metrics(subgrn)
            validation_results[cluster] = network_stats
        
        # Compute overlap statistics between putative and extracted edges
        overlap_stats = {}
        for cluster in subgrns.keys():
            if cluster == 'merged' or cluster not in tf_target_matrices:
                continue
                
            # Extract edges
            putative_edges = utils.extract_edges_from_matrix(tf_target_matrices[cluster])
            extracted_edges = utils.extract_edges_from_matrix(subgrns[cluster])
            
            # Compute overlap
            overlap = utils.compute_overlap_statistics(extracted_edges, putative_edges)
            overlap_stats[cluster] = overlap
        
        # Store results
        self.results['validation'] = {
            'network_statistics': validation_results,
            'overlap_statistics': overlap_stats
        }
        
        return validation_results
    
    def generate_visualizations(
        self,
        clusters_motifs_df: pd.DataFrame,
        differential_motifs: Dict[str, List[str]],
        tf_target_matrices: Dict[str, pd.DataFrame],
        subgrns: Dict[str, pd.DataFrame]
    ):
        """
        Step 6: Generate visualizations.
        
        Parameters:
        -----------
        clusters_motifs_df : pd.DataFrame
            Clusters x motifs matrix
        differential_motifs : Dict[str, List[str]]
            Differential motifs per cluster
        tf_target_matrices : Dict[str, pd.DataFrame]
            TF-target matrices per cluster
        subgrns : Dict[str, pd.DataFrame]
            Sub-GRNs per cluster
        """
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            # Plot cluster-motif heatmap
            print("Generating cluster-motif heatmap...")
            utils.plot_cluster_motif_heatmap(
                clusters_motifs_df,
                output_path=os.path.join(viz_dir, 'cluster_motif_heatmap.png')
            )
            
            # Plot TF-target networks for selected clusters
            print("Generating TF-target network plots...")
            for i, (cluster, matrix) in enumerate(tf_target_matrices.items()):
                if i >= 3:  # Limit to first 3 clusters
                    break
                utils.plot_tf_target_network(
                    matrix,
                    output_path=os.path.join(viz_dir, f'tf_target_network_cluster_{cluster}.png')
                )
            
            # Plot cluster statistics
            if 'summary' in self.results['tf_target_construction']:
                print("Generating cluster statistics plot...")
                utils.plot_cluster_statistics(
                    self.results['tf_target_construction']['summary'],
                    output_path=os.path.join(viz_dir, 'cluster_statistics.png')
                )
            
        except Exception as e:
            warnings.warn(f"Error generating visualizations: {str(e)}")
    
    def save_all_results(self):
        """
        Step 7: Save all results.
        """
        # Save complete results
        utils.save_results(
            self.results,
            os.path.join(self.output_dir, 'complete_results.pkl'),
            format='pickle'
        )
        
        # Save configuration
        utils.save_results(
            self.config,
            os.path.join(self.output_dir, 'config.json'),
            format='json'
        )
        
        print(f"All results saved to: {self.output_dir}")
    
    def generate_summary_report(self):
        """
        Generate a summary report of the analysis.
        """
        report_path = os.path.join(self.output_dir, 'analysis_report.md')
        utils.create_summary_report(self.results, report_path)

    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Create pipeline instance from configuration file.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration JSON file
            
        Returns:
        --------
        PeakClusterGRNPipeline
            Pipeline instance
        """
        config = utils.load_config(config_path)
        return cls(config)
    
    @classmethod
    def create_example_config(cls, output_path: str):
        """
        Create an example configuration file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the example configuration
        """
        config = utils.create_default_config()
        utils.save_results(config, output_path, format='json')
        print(f"Example configuration saved to: {output_path}")


# =============================================================================
# Example Usage and Configuration
# =============================================================================

def create_example_config():
    """Create an example configuration for the pipeline."""
    config = {
        'input_data': {
            'peaks_motifs_path': 'data/peaks_motifs.h5ad',
            'peaks_genes_path': 'data/peaks_genes.h5ad',
            'cluster_resolution': 'coarse'
        },
        'differential_analysis': {
            'method': 'top_n',
            'top_n': 10,
            'fold_change_threshold': 2.0,
            'pvalue_threshold': 0.001,
            'fdr_correction': True
        },
        'tf_target_construction': {
            'motif_database_path': None,  # Will infer from motif names
            'gene_association_method': 'correlation',
            'correlation_threshold': 0.5,
            'top_n_genes': None
        },
        'grn_extraction': {
            'grn_path': 'data/celloracle_grns/',
            'cell_type': None,
            'timepoint': None,
            'edge_strength_threshold': 0.1,
            'keep_only_putative': True,
            'merge_clusters': False,
            'merge_method': 'union'
        },
        'output': {
            'results_dir': 'results/peak_cluster_grn_analysis/',
            'save_intermediate': True,
            'file_format': 'pickle'
        }
    }
    return config


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Peak Cluster to GRN Analysis Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--create-config', type=str, help='Create example config file at specified path')
    parser.add_argument('--peaks-motifs', type=str, help='Path to peaks-motifs h5ad file')
    parser.add_argument('--peaks-genes', type=str, help='Path to peaks-genes h5ad file')
    parser.add_argument('--grn-path', type=str, help='Path to CellOracle GRN files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_example_config()
        utils.save_results(config, args.create_config, format='json')
        print(f"Example configuration created at: {args.create_config}")
    
    elif args.config:
        # Run pipeline from config file
        pipeline = PeakClusterGRNPipeline.from_config_file(args.config)
        results = pipeline.run_full_pipeline()
    
    elif args.peaks_motifs and args.peaks_genes and args.grn_path:
        # Run pipeline with command-line arguments
        config = create_example_config()
        config['input_data']['peaks_motifs_path'] = args.peaks_motifs
        config['input_data']['peaks_genes_path'] = args.peaks_genes
        config['grn_extraction']['grn_path'] = args.grn_path
        config['output']['results_dir'] = args.output_dir
        
        pipeline = PeakClusterGRNPipeline(config)
        results = pipeline.run_full_pipeline()
    
    else:
        print("Please provide either --config or --create-config, or all of --peaks-motifs, --peaks-genes, and --grn-path")
        parser.print_help()