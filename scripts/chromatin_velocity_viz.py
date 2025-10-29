"""
Visualization Tools for Chromatin Velocity

This module provides advanced visualization functions for chromatin velocity analysis,
including custom plots for peak UMAP, temporal dynamics, and biological interpretation.

Author: Generated for Zebrahub-Multiome analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scanpy as sc
import scvelo as scv
from typing import Optional, List, Dict, Tuple, Union
import warnings


class ChromatinVelocityVisualizer:
    """
    Advanced visualization class for chromatin velocity analysis.
    """
    
    def __init__(self, adata, figsize: Tuple[int, int] = (8, 6), dpi: int = 300):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        adata : AnnData
            AnnData object with computed chromatin velocity
        figsize : Tuple[int, int]
            Default figure size
        dpi : int
            DPI for saved figures
        """
        self.adata = adata
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up plotting parameters
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        sns.set_style("whitegrid")
    
    def plot_peak_accessibility_dynamics(self, 
                                       peak_subset: Optional[List[str]] = None,
                                       pseudobulk_order: Optional[List[str]] = None,
                                       save: Optional[str] = None,
                                       **kwargs):
        """
        Plot accessibility dynamics for selected peaks across pseudobulks.
        
        Parameters:
        -----------
        peak_subset : List[str], optional
            Subset of peaks to plot
        pseudobulk_order : List[str], optional
            Order of pseudobulks for plotting
        save : str, optional
            Path to save figure
        **kwargs
            Additional matplotlib arguments
        """
        if peak_subset is None:
            # Select top variable peaks
            if 'highly_variable' in self.adata.var.columns:
                peak_subset = self.adata.var_names[self.adata.var['highly_variable']][:20]
            else:
                peak_subset = self.adata.obs_names[:20]
        
        if pseudobulk_order is None:
            pseudobulk_order = list(self.adata.var_names)
        
        # Extract data for selected peaks and pseudobulks
        subset_idx = [i for i, peak in enumerate(self.adata.obs_names) if peak in peak_subset]
        pb_idx = [i for i, pb in enumerate(self.adata.var_names) if pb in pseudobulk_order]
        
        accessibility_data = self.adata.X[subset_idx, :][:, pb_idx]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(accessibility_data, aspect='auto', cmap='viridis', **kwargs)
        
        # Set labels
        ax.set_yticks(range(len(subset_idx)))
        ax.set_yticklabels([self.adata.obs_names[i] for i in subset_idx], fontsize=8)
        ax.set_xticks(range(len(pb_idx)))
        ax.set_xticklabels([self.adata.var_names[i] for i in pb_idx], rotation=45, ha='right', fontsize=8)
        
        ax.set_xlabel('Pseudobulk samples')
        ax.set_ylabel('Peaks')
        ax.set_title('Peak Accessibility Dynamics')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Accessibility')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_velocity_magnitude_distribution(self, 
                                          color_by: Optional[str] = None,
                                          save: Optional[str] = None):
        """
        Plot distribution of velocity magnitudes across peaks.
        
        Parameters:
        -----------
        color_by : str, optional
            Color peaks by metadata (e.g., peak_type, celltype_max)
        save : str, optional
            Path to save figure
        """
        if 'velocity_length' not in self.adata.obs.columns:
            # Compute velocity length if not available
            if 'velocity' in self.adata.layers:
                velocity = self.adata.layers['velocity']
                self.adata.obs['velocity_length'] = np.sqrt((velocity**2).sum(axis=1))
            else:
                raise ValueError("Velocity not computed or velocity_length not available")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of velocity magnitudes
        axes[0].hist(self.adata.obs['velocity_length'], bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Velocity Magnitude')
        axes[0].set_ylabel('Number of Peaks')
        axes[0].set_title('Distribution of Velocity Magnitudes')
        
        # Violin plot by category if specified
        if color_by and color_by in self.adata.obs.columns:
            df_plot = pd.DataFrame({
                'velocity_length': self.adata.obs['velocity_length'],
                'category': self.adata.obs[color_by]
            })
            
            sns.violinplot(data=df_plot, x='category', y='velocity_length', ax=axes[1])
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
            axes[1].set_title(f'Velocity by {color_by}')
        else:
            # Box plot of all data
            axes[1].boxplot(self.adata.obs['velocity_length'])
            axes[1].set_ylabel('Velocity Magnitude')
            axes[1].set_title('Velocity Magnitude Distribution')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_spliced_unspliced_relationship(self, 
                                          n_peaks: int = 100,
                                          color_by: Optional[str] = None,
                                          save: Optional[str] = None):
        """
        Plot relationship between spliced and unspliced counts.
        
        Parameters:
        -----------
        n_peaks : int
            Number of peaks to sample for plotting
        color_by : str, optional
            Color points by metadata
        save : str, optional
            Path to save figure
        """
        if 'spliced' not in self.adata.layers or 'unspliced' not in self.adata.layers:
            raise ValueError("Spliced and unspliced layers not found")
        
        # Sample peaks for visualization
        n_peaks = min(n_peaks, self.adata.n_obs)
        peak_indices = np.random.choice(self.adata.n_obs, n_peaks, replace=False)
        
        spliced = self.adata.layers['spliced'][peak_indices, :].flatten()
        unspliced = self.adata.layers['unspliced'][peak_indices, :].flatten()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if color_by and color_by in self.adata.obs.columns:
            # Color by metadata (repeat for each pseudobulk)
            colors = np.repeat(self.adata.obs[color_by].values[peak_indices], self.adata.n_vars)
            scatter = ax.scatter(spliced, unspliced, c=colors, alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(spliced, unspliced, alpha=0.6, s=1)
        
        # Add diagonal line
        min_val = min(spliced.min(), unspliced.min())
        max_val = max(spliced.max(), unspliced.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel('Spliced (Current Accessibility)')
        ax.set_ylabel('Unspliced (Propagated Accessibility)')
        ax.set_title('Spliced vs Unspliced Accessibility')
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_velocity_confidence_umap(self, 
                                    basis: str = 'umap',
                                    save: Optional[str] = None):
        """
        Plot velocity confidence on UMAP.
        
        Parameters:
        -----------
        basis : str
            Embedding basis to use
        save : str, optional
            Path to save figure
        """
        if f'X_{basis}' not in self.adata.obsm:
            raise ValueError(f"Embedding {basis} not found in obsm")
        
        if 'velocity_confidence' not in self.adata.obs.columns:
            warnings.warn("Velocity confidence not computed. Computing now...")
            scv.tl.velocity_confidence(self.adata)
        
        sc.pl.embedding(
            self.adata,
            basis=basis,
            color='velocity_confidence',
            title='Velocity Confidence',
            save=save,
            color_map='viridis'
        )
    
    def plot_temporal_velocity_trends(self, 
                                    timepoint_col: str = 'timepoint_order',
                                    celltype_col: str = 'celltype',
                                    save: Optional[str] = None):
        """
        Plot velocity trends across timepoints for different cell types.
        
        Parameters:
        -----------
        timepoint_col : str
            Column name for timepoint in var
        celltype_col : str
            Column name for celltype in var
        save : str, optional
            Path to save figure
        """
        if timepoint_col not in self.adata.var.columns:
            raise ValueError(f"{timepoint_col} not found in var")
        
        # Compute velocity magnitude per pseudobulk
        if 'velocity' in self.adata.layers:
            velocity_mag = np.sqrt((self.adata.layers['velocity']**2).sum(axis=0))
            self.adata.var['velocity_magnitude'] = velocity_mag
        else:
            raise ValueError("Velocity layer not found")
        
        # Create plotting dataframe
        plot_df = self.adata.var[[timepoint_col, celltype_col, 'velocity_magnitude']].copy()
        plot_df = plot_df.dropna()
        
        if plot_df.empty:
            print("No data available for temporal plotting")
            return
        
        # Plot trends
        fig, ax = plt.subplots(figsize=(10, 6))
        
        celltypes = plot_df[celltype_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(celltypes)))
        
        for celltype, color in zip(celltypes, colors):
            celltype_data = plot_df[plot_df[celltype_col] == celltype]
            
            if len(celltype_data) > 1:
                # Sort by timepoint
                celltype_data = celltype_data.sort_values(timepoint_col)
                
                ax.plot(celltype_data[timepoint_col], celltype_data['velocity_magnitude'], 
                       marker='o', label=celltype, color=color, linewidth=2, markersize=6)
            else:
                ax.scatter(celltype_data[timepoint_col], celltype_data['velocity_magnitude'], 
                          label=celltype, color=color, s=50)
        
        ax.set_xlabel('Developmental Timepoint')
        ax.set_ylabel('Velocity Magnitude')
        ax.set_title('Chromatin Velocity Across Development')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_peak_type_velocity(self, 
                              peak_type_col: str = 'peak_type',
                              save: Optional[str] = None):
        """
        Plot velocity by peak type (promoter, enhancer, etc.).
        
        Parameters:
        -----------
        peak_type_col : str
            Column name for peak type in obs
        save : str, optional
            Path to save figure
        """
        if peak_type_col not in self.adata.obs.columns:
            raise ValueError(f"{peak_type_col} not found in obs")
        
        if 'velocity_length' not in self.adata.obs.columns:
            if 'velocity' in self.adata.layers:
                velocity = self.adata.layers['velocity']
                self.adata.obs['velocity_length'] = np.sqrt((velocity**2).sum(axis=1))
            else:
                raise ValueError("Velocity not computed")
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        df_plot = pd.DataFrame({
            'peak_type': self.adata.obs[peak_type_col],
            'velocity_length': self.adata.obs['velocity_length']
        })
        
        sns.violinplot(data=df_plot, x='peak_type', y='velocity_length', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Velocity Magnitude')
        ax.set_title('Velocity by Peak Type')
        
        # Add statistical annotation
        peak_types = df_plot['peak_type'].unique()
        for i, peak_type in enumerate(peak_types):
            subset = df_plot[df_plot['peak_type'] == peak_type]['velocity_length']
            mean_vel = subset.mean()
            ax.text(i, ax.get_ylim()[1] * 0.9, f'μ={mean_vel:.3f}', 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def create_velocity_summary_figure(self, 
                                     basis: str = 'umap',
                                     save: Optional[str] = None):
        """
        Create comprehensive summary figure with multiple velocity visualizations.
        
        Parameters:
        -----------
        basis : str
            Embedding basis to use
        save : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Define subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Velocity embedding
        ax1 = fig.add_subplot(gs[0, 0])
        scv.pl.velocity_embedding(self.adata, basis=basis, ax=ax1, show=False, title='Velocity Embedding')
        
        # 2. Velocity stream
        ax2 = fig.add_subplot(gs[0, 1])
        scv.pl.velocity_embedding_stream(self.adata, basis=basis, ax=ax2, show=False, title='Velocity Stream')
        
        # 3. Velocity confidence
        ax3 = fig.add_subplot(gs[0, 2])
        if 'velocity_confidence' in self.adata.obs:
            sc.pl.embedding(self.adata, basis=basis, color='velocity_confidence', ax=ax3, show=False, title='Velocity Confidence')
        
        # 4. Velocity magnitude distribution
        ax4 = fig.add_subplot(gs[1, 0])
        if 'velocity_length' not in self.adata.obs.columns and 'velocity' in self.adata.layers:
            velocity = self.adata.layers['velocity']
            self.adata.obs['velocity_length'] = np.sqrt((velocity**2).sum(axis=1))
        
        if 'velocity_length' in self.adata.obs.columns:
            ax4.hist(self.adata.obs['velocity_length'], bins=30, alpha=0.7)
            ax4.set_xlabel('Velocity Magnitude')
            ax4.set_ylabel('Number of Peaks')
            ax4.set_title('Velocity Distribution')
        
        # 5. Spliced vs Unspliced
        ax5 = fig.add_subplot(gs[1, 1])
        if 'spliced' in self.adata.layers and 'unspliced' in self.adata.layers:
            spliced_sample = self.adata.layers['spliced'][:1000, :].flatten()
            unspliced_sample = self.adata.layers['unspliced'][:1000, :].flatten()
            ax5.scatter(spliced_sample, unspliced_sample, alpha=0.5, s=1)
            ax5.set_xlabel('Spliced')
            ax5.set_ylabel('Unspliced')
            ax5.set_title('Spliced vs Unspliced')
        
        # 6. Peak type velocity (if available)
        ax6 = fig.add_subplot(gs[1, 2])
        if 'peak_type' in self.adata.obs.columns and 'velocity_length' in self.adata.obs.columns:
            df_plot = pd.DataFrame({
                'peak_type': self.adata.obs['peak_type'],
                'velocity_length': self.adata.obs['velocity_length']
            })
            sns.boxplot(data=df_plot, x='peak_type', y='velocity_length', ax=ax6)
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
            ax6.set_title('Velocity by Peak Type')
        
        # 7-9. Additional embedding views (if other metadata available)
        for i, ax in enumerate([fig.add_subplot(gs[2, j]) for j in range(3)]):
            meta_cols = [col for col in self.adata.obs.columns 
                        if col not in ['velocity_length', 'velocity_confidence'] 
                        and self.adata.obs[col].dtype in ['object', 'category']]
            
            if i < len(meta_cols):
                sc.pl.embedding(self.adata, basis=basis, color=meta_cols[i], ax=ax, show=False, title=f'Colored by {meta_cols[i]}')
        
        plt.suptitle('Chromatin Velocity Analysis Summary', fontsize=16, y=0.98)
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()


def create_publication_figure(adata, 
                            output_path: str,
                            figure_size: Tuple[int, int] = (12, 8),
                            dpi: int = 300):
    """
    Create publication-ready figure for chromatin velocity.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with computed velocity
    output_path : str
        Path to save the figure
    figure_size : Tuple[int, int]
        Figure size in inches
    dpi : int
        Resolution for saved figure
    """
    visualizer = ChromatinVelocityVisualizer(adata, figsize=figure_size, dpi=dpi)
    visualizer.create_velocity_summary_figure(save=output_path)