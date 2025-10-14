#!/usr/bin/env python3
"""
Peak UMAP Chromatin Velocity Visualizer

This module provides comprehensive visualization tools for overlaying chromatin velocity vectors
onto peak UMAP embeddings, enabling analysis of chromatin accessibility dynamics across
developmental timepoints in the zebrafish embryo.

Author: Zebrahub-Multiome Analysis Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
import warnings
from typing import Optional, List, Dict, Tuple, Union
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import colorcet as cc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PeakUMAPVelocityVisualizer:
    """
    Comprehensive visualization system for chromatin velocity vectors on peak UMAP embeddings.
    """
    
    def __init__(self, umap_coords_path: str, velocity_results_path: str, 
                 color_palette_path: Optional[str] = None):
        """
        Initialize the visualizer with UMAP coordinates and velocity results.
        
        Parameters:
        -----------
        umap_coords_path : str
            Path to CSV file containing peak UMAP coordinates
        velocity_results_path : str  
            Path to H5AD file containing chromatin velocity results
        color_palette_path : str, optional
            Path to color palette definitions
        """
        self.umap_coords_path = umap_coords_path
        self.velocity_results_path = velocity_results_path
        
        # Data containers
        self.umap_coords = None
        self.velocity_adata = None
        self.integrated_data = None
        
        # Visualization parameters
        self.figsize = (12, 10)
        self.dpi = 300
        self.arrow_scale = 0.1
        self.arrow_width = 0.001
        self.point_size = 1.0
        
        # Color palettes
        self.setup_color_palettes()
        
        # Load data
        self.load_data()
        self.integrate_data()
        
    def setup_color_palettes(self):
        """Set up color palettes for different visualization modes."""
        
        # Peak type colors
        self.peak_type_colors = {
            'Promoter': '#e41a1c',
            'Enhancer': '#377eb8', 
            'Intergenic': '#4daf4a',
            'Exonic': '#984ea3',
            'Intronic': '#ff7f00',
            'UTR': '#ffff33',
            'Unknown': '#a65628'
        }
        
        # Lineage colors (from existing codebase)
        self.lineage_colors = {
            'CNS': '#1f77b4',
            'Neural Crest': '#ff7f0e',
            'Paraxial Mesoderm': '#2ca02c',
            'Lateral Plate Mesoderm': '#d62728',
            'Intermediate Mesoderm': '#9467bd',
            'Endoderm': '#8c564b',
            'Ectoderm': '#e377c2',
            'Mesendoderm': '#7f7f7f',
            'Unknown': '#bcbd22'
        }
        
        # Timepoint colors
        self.timepoint_colors = {
            '0somites': '#440154',
            '5somites': '#31688e', 
            '10somites': '#26828e',
            '15somites': '#1f9e89',
            '20somites': '#6ece58',
            '30somites': '#fde725'
        }
        
        # Velocity magnitude colormap
        self.velocity_cmap = plt.cm.viridis
        
    def load_data(self):
        """Load UMAP coordinates and velocity results."""
        print("Loading peak UMAP coordinates and velocity results...")
        
        # Load UMAP coordinates
        if Path(self.umap_coords_path).exists():
            self.umap_coords = pd.read_csv(self.umap_coords_path)
            print(f"Loaded UMAP coordinates: {self.umap_coords.shape}")
            print(f"UMAP columns: {list(self.umap_coords.columns)}")
        else:
            raise FileNotFoundError(f"UMAP coordinates file not found: {self.umap_coords_path}")
        
        # Load velocity results  
        if Path(self.velocity_results_path).exists():
            self.velocity_adata = sc.read_h5ad(self.velocity_results_path)
            print(f"Loaded velocity results: {self.velocity_adata.shape}")
            print(f"Velocity layers: {list(self.velocity_adata.layers.keys())}")
        else:
            raise FileNotFoundError(f"Velocity results file not found: {self.velocity_results_path}")
            
    def integrate_data(self):
        """Integrate UMAP coordinates with velocity results using peak matching."""
        print("Integrating UMAP coordinates with velocity results...")
        
        # Extract peak identifiers from both datasets
        umap_peaks = set(self.umap_coords.index) if hasattr(self.umap_coords, 'index') else set()
        if not umap_peaks and 'peak' in self.umap_coords.columns:
            umap_peaks = set(self.umap_coords['peak'])
        elif not umap_peaks:
            # If no explicit peak column, use index
            umap_peaks = set(self.umap_coords.index)
            
        velocity_peaks = set(self.velocity_adata.obs_names)
        
        # Find overlapping peaks
        overlapping_peaks = umap_peaks.intersection(velocity_peaks)
        missing_in_umap = velocity_peaks - umap_peaks
        missing_in_velocity = umap_peaks - velocity_peaks
        
        print(f"Peak matching statistics:")
        print(f"  UMAP peaks: {len(umap_peaks)}")
        print(f"  Velocity peaks: {len(velocity_peaks)}")
        print(f"  Overlapping peaks: {len(overlapping_peaks)}")
        print(f"  Missing in UMAP: {len(missing_in_umap)}")
        print(f"  Missing in velocity: {len(missing_in_velocity)}")
        
        if len(overlapping_peaks) == 0:
            raise ValueError("No overlapping peaks found between UMAP and velocity data!")
            
        # Create integrated dataset with only overlapping peaks
        overlapping_peaks_list = list(overlapping_peaks)
        
        # Filter UMAP coordinates
        if 'peak' in self.umap_coords.columns:
            umap_filtered = self.umap_coords[self.umap_coords['peak'].isin(overlapping_peaks_list)]
        else:
            umap_filtered = self.umap_coords.loc[overlapping_peaks_list]
            
        # Filter velocity data
        velocity_filtered = self.velocity_adata[overlapping_peaks_list, :].copy()
        
        # Create integrated DataFrame
        self.integrated_data = umap_filtered.copy()
        
        # Add velocity information
        velocity_vectors = velocity_filtered.layers['velocity']
        velocity_magnitude = np.sqrt((velocity_vectors**2).sum(axis=1))
        
        # Compute mean velocity components (across pseudobulks)
        mean_velocity = velocity_vectors.mean(axis=1)
        velocity_x = np.real(mean_velocity)  # Use real part as x-component
        velocity_y = np.imag(mean_velocity) if np.iscomplexobj(mean_velocity) else np.zeros_like(velocity_x)  # y-component
        
        # If velocity is real, compute directional components from PCA or use temporal ordering
        if not np.iscomplexobj(mean_velocity):
            # Use temporal gradient as direction proxy
            timepoint_order = {'0somites': 0, '5somites': 1, '10somites': 2, 
                             '15somites': 3, '20somites': 4, '30somites': 5}
            
            # Compute temporal gradient for each peak
            velocity_temporal_gradient = []
            for peak_idx in range(velocity_vectors.shape[0]):
                peak_velocity_vector = velocity_vectors[peak_idx, :]
                # Simple approach: use first two principal components
                if len(peak_velocity_vector) >= 2:
                    velocity_x = peak_velocity_vector[0]
                    velocity_y = peak_velocity_vector[1] if len(peak_velocity_vector) > 1 else 0
                else:
                    velocity_x = peak_velocity_vector[0]
                    velocity_y = 0
                velocity_temporal_gradient.append([velocity_x, velocity_y])
            
            velocity_temporal_gradient = np.array(velocity_temporal_gradient)
            velocity_x = velocity_temporal_gradient[:, 0]
            velocity_y = velocity_temporal_gradient[:, 1]
        
        # Add velocity components to integrated data
        self.integrated_data['velocity_magnitude'] = velocity_magnitude
        self.integrated_data['velocity_x'] = velocity_x  
        self.integrated_data['velocity_y'] = velocity_y
        
        # Add peak metadata if available
        if 'peak_type' in self.umap_coords.columns:
            pass  # Already in integrated_data
        else:
            # Infer peak type from peak name if possible
            self.integrated_data['peak_type'] = self.infer_peak_types(overlapping_peaks_list)
            
        print(f"Integration complete! Final dataset shape: {self.integrated_data.shape}")
        
    def infer_peak_types(self, peak_names: List[str]) -> List[str]:
        """Infer peak types from peak names if not explicitly provided."""
        peak_types = []
        for peak in peak_names:
            # Simple heuristic based on peak naming conventions
            if 'promoter' in peak.lower():
                peak_types.append('Promoter')
            elif 'enhancer' in peak.lower():
                peak_types.append('Enhancer') 
            elif 'exon' in peak.lower():
                peak_types.append('Exonic')
            elif 'intron' in peak.lower():
                peak_types.append('Intronic')
            else:
                peak_types.append('Intergenic')  # Default
        return peak_types
        
    def plot_velocity_vectors_on_umap(self, 
                                    color_by: str = 'velocity_magnitude',
                                    arrow_scale: float = None,
                                    min_velocity_threshold: float = 0.1,
                                    max_arrows: int = 5000,
                                    figsize: Tuple[int, int] = None,
                                    save_path: Optional[str] = None,
                                    title: str = "Chromatin Velocity Vectors on Peak UMAP") -> plt.Figure:
        """
        Create the main visualization: velocity vectors overlaid on peak UMAP.
        
        Parameters:
        -----------
        color_by : str
            How to color points ('velocity_magnitude', 'peak_type', 'timepoint', etc.)
        arrow_scale : float, optional
            Scale factor for velocity arrows
        min_velocity_threshold : float
            Minimum velocity magnitude to show arrows
        max_arrows : int
            Maximum number of arrows to show (for performance)
        figsize : tuple, optional
            Figure size
        save_path : str, optional  
            Path to save the figure
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if self.integrated_data is None:
            raise ValueError("No integrated data available. Run integrate_data() first.")
            
        # Set parameters
        if figsize is None:
            figsize = self.figsize
        if arrow_scale is None:
            arrow_scale = self.arrow_scale
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Filter data for visualization
        plot_data = self.integrated_data.copy()
        
        # Filter by velocity threshold
        high_velocity_mask = plot_data['velocity_magnitude'] >= min_velocity_threshold
        plot_data_filtered = plot_data[high_velocity_mask]
        
        # Sample arrows if too many
        if len(plot_data_filtered) > max_arrows:
            plot_data_filtered = plot_data_filtered.sample(n=max_arrows, random_state=42)
            
        # Create base scatter plot
        self._create_base_scatter(ax, plot_data, color_by)
        
        # Add velocity arrows
        self._add_velocity_arrows(ax, plot_data_filtered, arrow_scale)
        
        # Customize plot
        ax.set_xlabel('UMAP 1', fontsize=14)
        ax.set_ylabel('UMAP 2', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Add colorbar if appropriate
        if color_by == 'velocity_magnitude':
            self._add_velocity_colorbar(fig, ax)
        elif color_by in ['peak_type', 'lineage', 'timepoint']:
            self._add_categorical_legend(fig, ax, color_by)
            
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
            
        return fig
        
    def _create_base_scatter(self, ax, data, color_by):
        """Create the base UMAP scatter plot."""
        
        if color_by == 'velocity_magnitude':
            # Color by velocity magnitude
            scatter = ax.scatter(data['UMAP_1'], data['UMAP_2'], 
                               c=data['velocity_magnitude'],
                               cmap=self.velocity_cmap, s=self.point_size, alpha=0.6)
            self.current_scatter = scatter
            
        elif color_by == 'peak_type':
            # Color by peak type
            for peak_type in data['peak_type'].unique():
                mask = data['peak_type'] == peak_type
                color = self.peak_type_colors.get(peak_type, '#808080')
                ax.scatter(data[mask]['UMAP_1'], data[mask]['UMAP_2'],
                          c=color, s=self.point_size, alpha=0.6, label=peak_type)
                          
        elif color_by == 'lineage' and 'lineage' in data.columns:
            # Color by lineage
            for lineage in data['lineage'].unique():
                mask = data['lineage'] == lineage  
                color = self.lineage_colors.get(lineage, '#808080')
                ax.scatter(data[mask]['UMAP_1'], data[mask]['UMAP_2'],
                          c=color, s=self.point_size, alpha=0.6, label=lineage)
                          
        elif color_by == 'timepoint' and 'timepoint' in data.columns:
            # Color by timepoint
            for timepoint in data['timepoint'].unique():
                mask = data['timepoint'] == timepoint
                color = self.timepoint_colors.get(timepoint, '#808080')
                ax.scatter(data[mask]['UMAP_1'], data[mask]['UMAP_2'],
                          c=color, s=self.point_size, alpha=0.6, label=timepoint)
        else:
            # Default: gray points
            ax.scatter(data['UMAP_1'], data['UMAP_2'], 
                      c='lightgray', s=self.point_size, alpha=0.6)
                      
    def _add_velocity_arrows(self, ax, data, arrow_scale):
        """Add velocity arrows to the plot."""
        
        # Create quiver plot
        quiver = ax.quiver(data['UMAP_1'], data['UMAP_2'],
                          data['velocity_x'], data['velocity_y'],
                          scale=1/arrow_scale, width=self.arrow_width,
                          color='black', alpha=0.7, zorder=3)
        
        # Add arrow scale reference
        self._add_arrow_scale_reference(ax, arrow_scale)
        
    def _add_arrow_scale_reference(self, ax, arrow_scale):
        """Add a reference arrow showing the scale."""
        
        # Position for reference arrow (bottom-right corner)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        ref_x = xlim[1] - 0.15 * (xlim[1] - xlim[0])
        ref_y = ylim[0] + 0.1 * (ylim[1] - ylim[0])
        
        # Reference velocity magnitude
        ref_velocity = 1000  # Arbitrary reference
        
        # Add reference arrow
        ax.arrow(ref_x, ref_y, ref_velocity * arrow_scale, 0,
                head_width=0.02 * (ylim[1] - ylim[0]),
                head_length=0.02 * (xlim[1] - xlim[0]),
                fc='black', ec='black')
        
        # Add text
        ax.text(ref_x, ref_y - 0.05 * (ylim[1] - ylim[0]), 
               f'Velocity = {ref_velocity}', ha='center', va='top', fontsize=10)
        
    def _add_velocity_colorbar(self, fig, ax):
        """Add colorbar for velocity magnitude."""
        if hasattr(self, 'current_scatter'):
            cbar = fig.colorbar(self.current_scatter, ax=ax, shrink=0.6)
            cbar.set_label('Velocity Magnitude', fontsize=12)
            
    def _add_categorical_legend(self, fig, ax, color_by):
        """Add legend for categorical coloring."""
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
    def create_velocity_heatmap_on_umap(self, 
                                      bin_size: int = 50,
                                      figsize: Tuple[int, int] = None,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing velocity magnitude across the UMAP space.
        
        Parameters:
        -----------
        bin_size : int
            Number of bins for the heatmap grid
        figsize : tuple, optional
            Figure size  
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Create 2D histogram of velocity magnitudes
        x = self.integrated_data['UMAP_1']
        y = self.integrated_data['UMAP_2'] 
        weights = self.integrated_data['velocity_magnitude']
        
        # Create binned heatmap
        hist, xbins, ybins = np.histogram2d(x, y, bins=bin_size, weights=weights)
        count_hist, _, _ = np.histogram2d(x, y, bins=bin_size)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = hist / count_hist
            heatmap[count_hist == 0] = 0
            
        # Plot heatmap
        im = ax.imshow(heatmap.T, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                      origin='lower', cmap=self.velocity_cmap, aspect='auto')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Mean Velocity Magnitude', fontsize=12)
        
        # Labels and title
        ax.set_xlabel('UMAP 1', fontsize=14)
        ax.set_ylabel('UMAP 2', fontsize=14)
        ax.set_title('Chromatin Velocity Heatmap on Peak UMAP', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
            
        return fig
        
    def analyze_high_velocity_peaks(self, top_n: int = 100) -> pd.DataFrame:
        """
        Identify and analyze peaks with highest velocity magnitudes.
        
        Parameters:
        -----------
        top_n : int
            Number of top velocity peaks to analyze
            
        Returns:
        --------
        pd.DataFrame
            Analysis results for high-velocity peaks
        """
        if self.integrated_data is None:
            raise ValueError("No integrated data available.")
            
        # Sort by velocity magnitude
        sorted_data = self.integrated_data.sort_values('velocity_magnitude', ascending=False)
        top_peaks = sorted_data.head(top_n)
        
        # Analyze characteristics
        analysis = {
            'mean_velocity': top_peaks['velocity_magnitude'].mean(),
            'std_velocity': top_peaks['velocity_magnitude'].std(),
            'min_velocity': top_peaks['velocity_magnitude'].min(),
            'max_velocity': top_peaks['velocity_magnitude'].max(),
        }
        
        # Peak type distribution
        if 'peak_type' in top_peaks.columns:
            peak_type_dist = top_peaks['peak_type'].value_counts()
            analysis['peak_type_distribution'] = peak_type_dist.to_dict()
            
        # Print summary
        print(f"Analysis of top {top_n} high-velocity peaks:")
        print(f"  Mean velocity magnitude: {analysis['mean_velocity']:.2f}")
        print(f"  Velocity range: [{analysis['min_velocity']:.2f}, {analysis['max_velocity']:.2f}]")
        
        if 'peak_type_distribution' in analysis:
            print("  Peak type distribution:")
            for peak_type, count in analysis['peak_type_distribution'].items():
                print(f"    {peak_type}: {count} ({count/top_n*100:.1f}%)")
                
        return top_peaks, analysis
        
    def create_comparative_plot(self, 
                              filter_column: str,
                              filter_values: List[str],
                              figsize: Tuple[int, int] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparative velocity plots for different categories.
        
        Parameters:
        -----------
        filter_column : str
            Column to filter by (e.g., 'peak_type', 'timepoint')
        filter_values : list
            Values to compare
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if figsize is None:
            figsize = (15, 5 * len(filter_values))
            
        fig, axes = plt.subplots(len(filter_values), 1, figsize=figsize, dpi=self.dpi)
        if len(filter_values) == 1:
            axes = [axes]
            
        for i, filter_value in enumerate(filter_values):
            ax = axes[i]
            
            # Filter data
            mask = self.integrated_data[filter_column] == filter_value
            filtered_data = self.integrated_data[mask]
            
            # Create subplot
            self._create_base_scatter(ax, filtered_data, 'velocity_magnitude')
            
            # Add velocity arrows for high-velocity peaks only
            high_vel_mask = filtered_data['velocity_magnitude'] >= filtered_data['velocity_magnitude'].quantile(0.8)
            high_vel_data = filtered_data[high_vel_mask]
            
            if len(high_vel_data) > 0:
                self._add_velocity_arrows(ax, high_vel_data, self.arrow_scale)
                
            ax.set_title(f'{filter_column}: {filter_value} (n={len(filtered_data)})', fontsize=14)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Comparative plot saved to: {save_path}")
            
        return fig

def create_example_visualization(umap_path: str, velocity_path: str, output_dir: str = "."):
    """
    Create example visualizations demonstrating the chromatin velocity analysis.
    
    Parameters:
    -----------
    umap_path : str
        Path to UMAP coordinates file
    velocity_path : str  
        Path to velocity results file
    output_dir : str
        Directory to save output figures
    """
    
    # Initialize visualizer
    print("Initializing Peak UMAP Velocity Visualizer...")
    visualizer = PeakUMAPVelocityVisualizer(umap_path, velocity_path)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Main velocity vector plot
    print("\n1. Creating main velocity vector visualization...")
    fig1 = visualizer.plot_velocity_vectors_on_umap(
        color_by='velocity_magnitude',
        title="Chromatin Velocity Vectors on Peak UMAP",
        save_path=f"{output_dir}/chromatin_velocity_vectors_umap.png"
    )
    plt.show()
    
    # 2. Velocity heatmap
    print("\n2. Creating velocity heatmap...")
    fig2 = visualizer.create_velocity_heatmap_on_umap(
        save_path=f"{output_dir}/chromatin_velocity_heatmap_umap.png"
    )
    plt.show()
    
    # 3. Analyze high-velocity peaks
    print("\n3. Analyzing high-velocity peaks...")
    top_peaks, analysis = visualizer.analyze_high_velocity_peaks(top_n=100)
    
    # Save high-velocity peaks
    top_peaks.to_csv(f"{output_dir}/high_velocity_peaks_top100.csv", index=False)
    
    # 4. Peak type comparison (if available)
    if 'peak_type' in visualizer.integrated_data.columns:
        print("\n4. Creating peak type comparison...")
        unique_peak_types = visualizer.integrated_data['peak_type'].unique()[:4]  # Top 4 types
        fig4 = visualizer.create_comparative_plot(
            filter_column='peak_type',
            filter_values=list(unique_peak_types),
            save_path=f"{output_dir}/chromatin_velocity_by_peak_type.png"
        )
        plt.show()
    
    print(f"\nVisualization complete! Results saved to: {output_dir}")
    print(f"Integration coverage: {len(visualizer.integrated_data)} peaks")
    
    return visualizer

if __name__ == "__main__":
    # Example usage with actual data paths
    umap_coords_path = "peak_umap_3d_annotated_v6.csv"
    velocity_results_path = "chromatin_velocity_results.h5ad"
    
    if Path(umap_coords_path).exists() and Path(velocity_results_path).exists():
        visualizer = create_example_visualization(
            umap_coords_path, 
            velocity_results_path,
            output_dir="chromatin_velocity_visualizations"
        )
    else:
        print("Data files not found. Please check the file paths.")
        print(f"Expected files:")
        print(f"  - {umap_coords_path}")
        print(f"  - {velocity_results_path}")