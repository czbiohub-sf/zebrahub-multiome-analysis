"""
Validation and Biological Interpretation for Chromatin Velocity

This module provides functions to validate chromatin velocity results
and interpret them in biological context.

Author: Generated for Zebrahub-Multiome analysis
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
import warnings


class ChromatinVelocityValidator:
    """
    Validation and interpretation class for chromatin velocity results.
    """
    
    def __init__(self, adata, peak_annotations: Optional[pd.DataFrame] = None):
        """
        Initialize validator.
        
        Parameters:
        -----------
        adata : AnnData
            AnnData object with computed chromatin velocity
        peak_annotations : pd.DataFrame, optional
            Peak annotations (gene associations, regulatory regions, etc.)
        """
        self.adata = adata
        self.peak_annotations = peak_annotations
        self.validation_results = {}
    
    def validate_velocity_consistency(self, 
                                    temporal_col: str = 'timepoint_order',
                                    return_details: bool = False) -> Dict:
        """
        Validate that velocity directions are consistent with temporal ordering.
        
        Parameters:
        -----------
        temporal_col : str
            Column name for temporal ordering
        return_details : bool
            Whether to return detailed results
            
        Returns:
        --------
        Dict with validation metrics
        """
        if 'velocity' not in self.adata.layers:
            raise ValueError("Velocity not computed")
        
        if temporal_col not in self.adata.var.columns:
            warnings.warn(f"{temporal_col} not found. Skipping temporal validation.")
            return {}
        
        print("Validating velocity temporal consistency...")
        
        # Get velocity and temporal data
        velocity = self.adata.layers['velocity']
        timepoints = self.adata.var[temporal_col].values
        
        # For each peak, check if velocity direction correlates with time progression
        consistency_scores = []
        
        for peak_idx in range(self.adata.n_obs):
            peak_velocity = velocity[peak_idx, :]
            
            # Skip if all velocities are zero
            if np.all(peak_velocity == 0):
                continue
            
            # Compute correlation between velocity and timepoint
            valid_mask = ~np.isnan(peak_velocity) & ~np.isnan(timepoints)
            if np.sum(valid_mask) > 2:
                corr, p_value = stats.pearsonr(timepoints[valid_mask], peak_velocity[valid_mask])
                consistency_scores.append({
                    'peak': self.adata.obs_names[peak_idx],
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        if not consistency_scores:
            warnings.warn("No valid peaks for temporal consistency analysis")
            return {}
        
        consistency_df = pd.DataFrame(consistency_scores)
        
        # Summary statistics
        results = {
            'n_peaks_analyzed': len(consistency_df),
            'mean_correlation': consistency_df['correlation'].mean(),
            'std_correlation': consistency_df['correlation'].std(),
            'fraction_positive_correlation': (consistency_df['correlation'] > 0).mean(),
            'fraction_significant': consistency_df['significant'].mean(),
            'strong_positive_correlation': (consistency_df['correlation'] > 0.5).mean(),
            'strong_negative_correlation': (consistency_df['correlation'] < -0.5).mean()
        }
        
        if return_details:
            results['detailed_scores'] = consistency_df
        
        self.validation_results['temporal_consistency'] = results
        
        print(f"Temporal consistency analysis complete:")
        print(f"  - Peaks analyzed: {results['n_peaks_analyzed']}")
        print(f"  - Mean correlation: {results['mean_correlation']:.3f}")
        print(f"  - Fraction with positive correlation: {results['fraction_positive_correlation']:.3f}")
        print(f"  - Fraction significant: {results['fraction_significant']:.3f}")
        
        return results
    
    def compare_with_rna_velocity(self, 
                                rna_velocity_adata,
                                peak_gene_mapping: Optional[Dict] = None,
                                correlation_method: str = 'pearson') -> Dict:
        """
        Compare chromatin velocity with RNA velocity from matched samples.
        
        Parameters:
        -----------
        rna_velocity_adata : AnnData
            RNA velocity AnnData object
        peak_gene_mapping : Dict, optional
            Mapping from peaks to genes
        correlation_method : str
            Correlation method ('pearson', 'spearman')
            
        Returns:
        --------
        Dict with comparison results
        """
        if 'velocity' not in self.adata.layers:
            raise ValueError("Chromatin velocity not computed")
        
        if 'velocity' not in rna_velocity_adata.layers:
            raise ValueError("RNA velocity not computed in provided data")
        
        print("Comparing chromatin and RNA velocity...")
        
        # Find common samples (pseudobulks)
        common_samples = list(set(self.adata.var_names) & set(rna_velocity_adata.var_names))
        
        if len(common_samples) == 0:
            warnings.warn("No common samples found between chromatin and RNA velocity")
            return {}
        
        print(f"Found {len(common_samples)} common samples")
        
        # Extract velocity data for common samples
        chrom_idx = [self.adata.var_names.get_loc(s) for s in common_samples]
        rna_idx = [rna_velocity_adata.var_names.get_loc(s) for s in common_samples]
        
        chrom_velocity = self.adata.layers['velocity'][:, chrom_idx]
        rna_velocity = rna_velocity_adata.layers['velocity'][:, rna_idx]
        
        # Compute correlations
        correlations = []
        
        if peak_gene_mapping is not None:
            # Use peak-gene mapping for specific comparisons
            for peak_name in self.adata.obs_names:
                if peak_name in peak_gene_mapping:
                    gene_name = peak_gene_mapping[peak_name]
                    if gene_name in rna_velocity_adata.obs_names:
                        peak_idx = self.adata.obs_names.get_loc(peak_name)
                        gene_idx = rna_velocity_adata.obs_names.get_loc(gene_name)
                        
                        peak_vel = chrom_velocity[peak_idx, :]
                        gene_vel = rna_velocity[gene_idx, :]
                        
                        if correlation_method == 'pearson':
                            corr, p_val = stats.pearsonr(peak_vel, gene_vel)
                        else:
                            corr, p_val = stats.spearmanr(peak_vel, gene_vel)
                        
                        correlations.append({
                            'peak': peak_name,
                            'gene': gene_name,
                            'correlation': corr,
                            'p_value': p_val
                        })
        else:
            # Global comparison using all data
            chrom_flat = chrom_velocity.flatten()
            rna_flat = rna_velocity.flatten()
            
            if correlation_method == 'pearson':
                global_corr, global_p = stats.pearsonr(chrom_flat, rna_flat)
            else:
                global_corr, global_p = stats.spearmanr(chrom_flat, rna_flat)
            
            correlations.append({
                'peak': 'global',
                'gene': 'global',
                'correlation': global_corr,
                'p_value': global_p
            })
        
        # Summary results
        if correlations:
            corr_df = pd.DataFrame(correlations)
            results = {
                'n_comparisons': len(corr_df),
                'mean_correlation': corr_df['correlation'].mean(),
                'std_correlation': corr_df['correlation'].std(),
                'significant_correlations': (corr_df['p_value'] < 0.05).sum(),
                'detailed_correlations': corr_df,
                'common_samples': common_samples
            }
        else:
            results = {'n_comparisons': 0, 'message': 'No valid comparisons found'}
        
        self.validation_results['rna_velocity_comparison'] = results
        
        if 'mean_correlation' in results:
            print(f"RNA velocity comparison complete:")
            print(f"  - Comparisons: {results['n_comparisons']}")
            print(f"  - Mean correlation: {results['mean_correlation']:.3f}")
            print(f"  - Significant correlations: {results['significant_correlations']}")
        
        return results
    
    def analyze_regulatory_program_coherence(self, 
                                           cluster_col: str = 'leiden_coarse',
                                           min_cluster_size: int = 50) -> Dict:
        """
        Analyze coherence of velocity within regulatory programs/clusters.
        
        Parameters:
        -----------
        cluster_col : str
            Column name for peak clusters
        min_cluster_size : int
            Minimum cluster size for analysis
            
        Returns:
        --------
        Dict with coherence analysis results
        """
        if 'velocity' not in self.adata.layers:
            raise ValueError("Velocity not computed")
        
        if cluster_col not in self.adata.obs.columns:
            raise ValueError(f"{cluster_col} not found in observations")
        
        print("Analyzing regulatory program coherence...")
        
        velocity = self.adata.layers['velocity']
        clusters = self.adata.obs[cluster_col]
        
        coherence_results = []
        
        for cluster_id in clusters.unique():
            cluster_mask = clusters == cluster_id
            cluster_size = cluster_mask.sum()
            
            if cluster_size < min_cluster_size:
                continue
            
            cluster_velocity = velocity[cluster_mask, :]
            
            # Compute pairwise cosine similarities within cluster
            if cluster_size > 1:
                # Flatten each peak's velocity vector
                velocities_norm = cluster_velocity / (np.linalg.norm(cluster_velocity, axis=1, keepdims=True) + 1e-8)
                
                # Compute pairwise cosine similarities
                pairwise_similarities = []
                for i in range(len(velocities_norm)):
                    for j in range(i+1, len(velocities_norm)):
                        sim = np.dot(velocities_norm[i], velocities_norm[j])
                        pairwise_similarities.append(sim)
                
                if pairwise_similarities:
                    mean_similarity = np.mean(pairwise_similarities)
                    std_similarity = np.std(pairwise_similarities)
                    
                    # Compute velocity magnitude statistics
                    velocity_magnitudes = np.sqrt((cluster_velocity**2).sum(axis=1))
                    mean_magnitude = np.mean(velocity_magnitudes)
                    std_magnitude = np.std(velocity_magnitudes)
                    
                    coherence_results.append({
                        'cluster': cluster_id,
                        'size': cluster_size,
                        'mean_cosine_similarity': mean_similarity,
                        'std_cosine_similarity': std_similarity,
                        'mean_velocity_magnitude': mean_magnitude,
                        'std_velocity_magnitude': std_magnitude,
                        'coherence_score': mean_similarity / (std_similarity + 1e-8)  # Higher = more coherent
                    })
        
        if not coherence_results:
            warnings.warn("No clusters met size criteria for coherence analysis")
            return {}
        
        coherence_df = pd.DataFrame(coherence_results)
        
        # Summary statistics
        results = {
            'n_clusters_analyzed': len(coherence_df),
            'mean_coherence_score': coherence_df['coherence_score'].mean(),
            'top_coherent_clusters': coherence_df.nlargest(5, 'coherence_score')[['cluster', 'coherence_score']].to_dict('records'),
            'cluster_details': coherence_df
        }
        
        self.validation_results['regulatory_coherence'] = results
        
        print(f"Regulatory coherence analysis complete:")
        print(f"  - Clusters analyzed: {results['n_clusters_analyzed']}")
        print(f"  - Mean coherence score: {results['mean_coherence_score']:.3f}")
        
        return results
    
    def identify_pioneer_peaks(self, 
                             velocity_threshold: float = 0.8,
                             accessibility_threshold: float = 0.2) -> List[str]:
        """
        Identify potential pioneer peaks with high velocity but low current accessibility.
        
        Parameters:
        -----------
        velocity_threshold : float
            Minimum velocity magnitude (percentile)
        accessibility_threshold : float
            Maximum current accessibility (percentile)
            
        Returns:
        --------
        List of pioneer peak names
        """
        if 'velocity' not in self.adata.layers:
            raise ValueError("Velocity not computed")
        
        print("Identifying potential pioneer peaks...")
        
        # Compute velocity magnitude
        velocity_magnitude = np.sqrt((self.adata.layers['velocity']**2).sum(axis=1))
        
        # Get current accessibility (mean across pseudobulks)
        current_accessibility = np.mean(self.adata.layers['spliced'], axis=1)
        
        # Define thresholds
        vel_thresh = np.percentile(velocity_magnitude, velocity_threshold * 100)
        acc_thresh = np.percentile(current_accessibility, accessibility_threshold * 100)
        
        # Identify pioneer peaks
        pioneer_mask = (velocity_magnitude >= vel_thresh) & (current_accessibility <= acc_thresh)
        pioneer_peaks = self.adata.obs_names[pioneer_mask].tolist()
        
        # Store in validation results
        self.validation_results['pioneer_peaks'] = {
            'n_pioneer_peaks': len(pioneer_peaks),
            'velocity_threshold': vel_thresh,
            'accessibility_threshold': acc_thresh,
            'pioneer_peak_names': pioneer_peaks
        }
        
        print(f"Identified {len(pioneer_peaks)} potential pioneer peaks")
        
        return pioneer_peaks
    
    def validate_coaccessibility_propagation(self, 
                                           coaccessibility_matrix: np.ndarray,
                                           sample_size: int = 1000) -> Dict:
        """
        Validate that co-accessibility propagation makes biological sense.
        
        Parameters:
        -----------
        coaccessibility_matrix : np.ndarray
            Original co-accessibility matrix
        sample_size : int
            Number of peaks to sample for validation
            
        Returns:
        --------
        Dict with validation results
        """
        if 'spliced' not in self.adata.layers or 'unspliced' not in self.adata.layers:
            raise ValueError("Spliced and unspliced layers not found")
        
        print("Validating co-accessibility propagation...")
        
        # Sample peaks for analysis
        n_peaks = min(sample_size, self.adata.n_obs)
        peak_indices = np.random.choice(self.adata.n_obs, n_peaks, replace=False)
        
        propagation_validation = []
        
        for peak_idx in peak_indices:
            # Get co-accessibility connections for this peak
            coaccess_scores = coaccessibility_matrix[peak_idx, :]
            connected_peaks = np.where(coaccess_scores > 0.1)[0]  # Threshold for connections
            
            if len(connected_peaks) == 0:
                continue
            
            # Compare direct accessibility vs propagated accessibility
            direct_accessibility = self.adata.layers['spliced'][peak_idx, :]
            propagated_accessibility = self.adata.layers['unspliced'][peak_idx, :]
            
            # Compute correlation with connected peaks' accessibility
            connected_accessibility = np.mean(self.adata.layers['spliced'][connected_peaks, :], axis=0)
            
            # Correlation between propagated and connected peaks' mean accessibility
            corr_propagated, p_prop = stats.pearsonr(propagated_accessibility, connected_accessibility)
            
            # Correlation between direct and connected peaks' mean accessibility
            corr_direct, p_direct = stats.pearsonr(direct_accessibility, connected_accessibility)
            
            propagation_validation.append({
                'peak_idx': peak_idx,
                'n_connections': len(connected_peaks),
                'correlation_propagated': corr_propagated,
                'correlation_direct': corr_direct,
                'p_value_propagated': p_prop,
                'p_value_direct': p_direct,
                'propagation_improvement': corr_propagated - corr_direct
            })
        
        if not propagation_validation:
            warnings.warn("No peaks with connections found for validation")
            return {}
        
        validation_df = pd.DataFrame(propagation_validation)
        
        results = {
            'n_peaks_validated': len(validation_df),
            'mean_propagation_correlation': validation_df['correlation_propagated'].mean(),
            'mean_direct_correlation': validation_df['correlation_direct'].mean(),
            'mean_improvement': validation_df['propagation_improvement'].mean(),
            'fraction_improved': (validation_df['propagation_improvement'] > 0).mean(),
            'validation_details': validation_df
        }
        
        self.validation_results['coaccessibility_propagation'] = results
        
        print(f"Co-accessibility propagation validation complete:")
        print(f"  - Peaks validated: {results['n_peaks_validated']}")
        print(f"  - Mean propagation correlation: {results['mean_propagation_correlation']:.3f}")
        print(f"  - Mean improvement: {results['mean_improvement']:.3f}")
        print(f"  - Fraction improved: {results['fraction_improved']:.3f}")
        
        return results
    
    def generate_validation_report(self, output_path: str):
        """
        Generate comprehensive validation report.
        
        Parameters:
        -----------
        output_path : str
            Path to save the validation report
        """
        report_sections = []
        
        report_sections.append("# Chromatin Velocity Validation Report\n")
        report_sections.append(f"Generated for dataset with {self.adata.n_obs} peaks and {self.adata.n_vars} pseudobulks\n\n")
        
        # Temporal consistency
        if 'temporal_consistency' in self.validation_results:
            tc = self.validation_results['temporal_consistency']
            report_sections.append("## Temporal Consistency Validation\n")
            report_sections.append(f"- Peaks analyzed: {tc['n_peaks_analyzed']}\n")
            report_sections.append(f"- Mean correlation with time: {tc['mean_correlation']:.3f}\n")
            report_sections.append(f"- Fraction with positive correlation: {tc['fraction_positive_correlation']:.3f}\n")
            report_sections.append(f"- Fraction significant (p<0.05): {tc['fraction_significant']:.3f}\n\n")
        
        # RNA velocity comparison
        if 'rna_velocity_comparison' in self.validation_results:
            rvc = self.validation_results['rna_velocity_comparison']
            report_sections.append("## RNA Velocity Comparison\n")
            if 'mean_correlation' in rvc:
                report_sections.append(f"- Comparisons made: {rvc['n_comparisons']}\n")
                report_sections.append(f"- Mean correlation: {rvc['mean_correlation']:.3f}\n")
                report_sections.append(f"- Significant correlations: {rvc['significant_correlations']}\n\n")
            else:
                report_sections.append("- No valid comparisons found\n\n")
        
        # Regulatory coherence
        if 'regulatory_coherence' in self.validation_results:
            rc = self.validation_results['regulatory_coherence']
            report_sections.append("## Regulatory Program Coherence\n")
            report_sections.append(f"- Clusters analyzed: {rc['n_clusters_analyzed']}\n")
            report_sections.append(f"- Mean coherence score: {rc['mean_coherence_score']:.3f}\n")
            report_sections.append("- Top coherent clusters:\n")
            for cluster_info in rc['top_coherent_clusters']:
                report_sections.append(f"  - Cluster {cluster_info['cluster']}: {cluster_info['coherence_score']:.3f}\n")
            report_sections.append("\n")
        
        # Pioneer peaks
        if 'pioneer_peaks' in self.validation_results:
            pp = self.validation_results['pioneer_peaks']
            report_sections.append("## Pioneer Peak Analysis\n")
            report_sections.append(f"- Pioneer peaks identified: {pp['n_pioneer_peaks']}\n")
            report_sections.append(f"- Velocity threshold: {pp['velocity_threshold']:.3f}\n")
            report_sections.append(f"- Accessibility threshold: {pp['accessibility_threshold']:.3f}\n\n")
        
        # Co-accessibility validation
        if 'coaccessibility_propagation' in self.validation_results:
            cap = self.validation_results['coaccessibility_propagation']
            report_sections.append("## Co-accessibility Propagation Validation\n")
            report_sections.append(f"- Peaks validated: {cap['n_peaks_validated']}\n")
            report_sections.append(f"- Mean propagation correlation: {cap['mean_propagation_correlation']:.3f}\n")
            report_sections.append(f"- Mean improvement over direct: {cap['mean_improvement']:.3f}\n")
            report_sections.append(f"- Fraction showing improvement: {cap['fraction_improved']:.3f}\n\n")
        
        # Write report
        with open(output_path, 'w') as f:
            f.writelines(report_sections)
        
        print(f"Validation report saved to {output_path}")


def run_comprehensive_validation(adata,
                                coaccessibility_matrix: Optional[np.ndarray] = None,
                                rna_velocity_adata = None,
                                peak_gene_mapping: Optional[Dict] = None,
                                output_dir: str = './validation/') -> ChromatinVelocityValidator:
    """
    Run comprehensive validation of chromatin velocity results.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData with computed chromatin velocity
    coaccessibility_matrix : np.ndarray, optional
        Original co-accessibility matrix
    rna_velocity_adata : AnnData, optional
        RNA velocity data for comparison
    peak_gene_mapping : Dict, optional
        Peak to gene mapping
    output_dir : str
        Directory to save validation outputs
        
    Returns:
    --------
    ChromatinVelocityValidator with all validation results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Comprehensive Chromatin Velocity Validation ===")
    
    # Initialize validator
    validator = ChromatinVelocityValidator(adata)
    
    # Run validations
    validator.validate_velocity_consistency()
    
    if rna_velocity_adata is not None:
        validator.compare_with_rna_velocity(rna_velocity_adata, peak_gene_mapping)
    
    validator.analyze_regulatory_program_coherence()
    validator.identify_pioneer_peaks()
    
    if coaccessibility_matrix is not None:
        validator.validate_coaccessibility_propagation(coaccessibility_matrix)
    
    # Generate report
    validator.generate_validation_report(f"{output_dir}/validation_report.md")
    
    print("=== Validation Complete ===")
    
    return validator