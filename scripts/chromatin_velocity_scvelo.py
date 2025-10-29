"""
scVelo Integration for Chromatin Velocity

This module provides functions to integrate chromatin velocity with scVelo
for velocity estimation, confidence calculation, and visualization.

Author: Generated for Zebrahub-Multiome analysis
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, List, Union
import warnings
from chromatin_velocity import ChromatinVelocity


class ChromatinVelocityAnalysis:
    """
    Analysis class for chromatin velocity using scVelo framework.
    """
    
    def __init__(self, chromatin_velocity: ChromatinVelocity):
        """
        Initialize with ChromatinVelocity object.
        
        Parameters:
        -----------
        chromatin_velocity : ChromatinVelocity
            Computed chromatin velocity object
        """
        self.cv = chromatin_velocity
        if self.cv.adata is None:
            raise ValueError("AnnData object not created. Call create_anndata_object() first.")
        
        self.adata = self.cv.adata.copy()
        self.velocity_computed = False
    
    def prepare_for_scvelo(self, 
                          min_shared_counts: int = 20,
                          min_counts_per_pseudobulk: int = 10,
                          filter_genes: bool = True):
        """
        Prepare AnnData object for scVelo analysis.
        
        Parameters:
        -----------
        min_shared_counts : int
            Minimum shared counts for filtering
        min_counts_per_pseudobulk : int  
            Minimum counts per pseudobulk for filtering
        filter_genes : bool
            Whether to filter pseudobulks (genes in scVelo terms)
        """
        print("Preparing data for scVelo analysis...")
        
        # In our case, pseudobulks are "genes" and peaks are "cells"
        # Filter pseudobulks (genes) based on expression
        if filter_genes:
            # Calculate basic statistics
            n_peaks_per_pseudobulk = (self.adata.X > 0).sum(axis=0)
            mean_accessibility_per_pseudobulk = np.array(self.adata.X.mean(axis=0)).flatten()
            
            # Filter criteria
            keep_pseudobulks = (
                (n_peaks_per_pseudobulk >= min_shared_counts) &
                (mean_accessibility_per_pseudobulk >= min_counts_per_pseudobulk)
            )
            
            print(f"Filtering pseudobulks: {keep_pseudobulks.sum()}/{len(keep_pseudobulks)} retained")
            
            if keep_pseudobulks.sum() > 0:
                self.adata = self.adata[:, keep_pseudobulks]
            else:
                warnings.warn("No pseudobulks passed filtering criteria. Using all pseudobulks.")
        
        # Set up scVelo annotations
        self.adata.var['highly_variable'] = True  # Treat all remaining pseudobulks as highly variable
        
        print(f"Prepared AnnData for scVelo: {self.adata.shape}")
    
    def compute_moments(self, n_neighbors: int = 30, n_pcs: int = 30):
        """
        Compute moments for velocity estimation.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors for moment calculation
        n_pcs : int
            Number of principal components
        """
        print("Computing moments...")
        
        # Compute PCA if not already present
        if 'X_pca' not in self.adata.obsm:
            sc.tl.pca(self.adata, n_comps=n_pcs)
        
        # Compute neighborhood graph if not present
        if 'neighbors' not in self.adata.uns:
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        # Compute moments
        scv.pp.moments(self.adata, n_neighbors=n_neighbors)
        
        print("Moments computed successfully")
    
    def estimate_velocity(self, 
                         mode: str = 'dynamical',
                         fit_likelihood: bool = True,
                         filter_genes: bool = False):
        """
        Estimate chromatin velocity using scVelo.
        
        Parameters:
        -----------
        mode : str
            Velocity estimation mode ('stochastic', 'dynamical', 'deterministic')
        fit_likelihood : bool
            Whether to fit likelihood for dynamical mode
        filter_genes : bool
            Whether to filter genes again during velocity estimation
        """
        print(f"Estimating velocity using {mode} mode...")
        
        # Set scVelo settings
        scv.settings.verbosity = 3
        
        if mode == 'dynamical':
            # Recover dynamics
            scv.tl.recover_dynamics(self.adata, fit_likelihood=fit_likelihood)
            # Estimate velocity
            scv.tl.velocity(self.adata, mode='dynamical')
        elif mode == 'stochastic':
            scv.tl.velocity(self.adata, mode='stochastic')
        elif mode == 'deterministic':
            scv.tl.velocity(self.adata, mode='deterministic') 
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Compute velocity graph
        scv.tl.velocity_graph(self.adata)
        
        self.velocity_computed = True
        print("Velocity estimation completed")
    
    def compute_velocity_confidence(self):
        """Compute velocity confidence metrics."""
        if not self.velocity_computed:
            raise ValueError("Velocity not computed. Call estimate_velocity() first.")
        
        print("Computing velocity confidence...")
        
        # Velocity confidence
        scv.tl.velocity_confidence(self.adata)
        
        # Velocity length and variance
        if 'velocity' in self.adata.layers:
            velocity = self.adata.layers['velocity']
            
            # Velocity magnitude
            self.adata.obs['velocity_length'] = np.sqrt((velocity**2).sum(axis=1))
            
            # Velocity variance across pseudobulks
            self.adata.obs['velocity_variance'] = np.var(velocity, axis=1)
        
        print("Velocity confidence computed")
    
    def identify_velocity_genes(self, 
                               min_likelihood: float = 0.1,
                               min_confidence: float = 0.75,
                               top_n: Optional[int] = None) -> List[str]:
        """
        Identify pseudobulks (cell-type/timepoint combinations) with high velocity.
        
        Parameters:
        -----------
        min_likelihood : float
            Minimum likelihood for dynamical mode
        min_confidence : float
            Minimum velocity confidence
        top_n : int, optional
            Return top N pseudobulks by velocity
        
        Returns:
        --------
        List of high-velocity pseudobulk names
        """
        if not self.velocity_computed:
            raise ValueError("Velocity not computed. Call estimate_velocity() first.")
        
        criteria = np.ones(self.adata.n_vars, dtype=bool)
        
        # Filter by likelihood if available
        if 'fit_likelihood' in self.adata.var:
            criteria &= (self.adata.var['fit_likelihood'] >= min_likelihood)
        
        # Filter by confidence if available  
        if 'velocity_confidence' in self.adata.var:
            criteria &= (self.adata.var['velocity_confidence'] >= min_confidence)
        
        velocity_pseudobulks = self.adata.var_names[criteria].tolist()
        
        if top_n is not None and len(velocity_pseudobulks) > top_n:
            # Sort by confidence or likelihood
            if 'velocity_confidence' in self.adata.var:
                scores = self.adata.var.loc[velocity_pseudobulks, 'velocity_confidence']
            elif 'fit_likelihood' in self.adata.var:
                scores = self.adata.var.loc[velocity_pseudobulks, 'fit_likelihood']
            else:
                # Use velocity magnitude as fallback
                velocity_mag = np.sqrt((self.adata.layers['velocity']**2).sum(axis=0))
                scores = pd.Series(velocity_mag, index=self.adata.var_names)
                scores = scores.loc[velocity_pseudobulks]
            
            top_indices = scores.nlargest(top_n).index
            velocity_pseudobulks = top_indices.tolist()
        
        print(f"Identified {len(velocity_pseudobulks)} high-velocity pseudobulks")
        return velocity_pseudobulks
    
    def plot_velocity_embedding(self, 
                               basis: str = 'umap',
                               color: Optional[Union[str, List[str]]] = None,
                               save: Optional[str] = None,
                               **kwargs):
        """
        Plot velocity on embedding.
        
        Parameters:
        -----------
        basis : str
            Embedding basis ('umap', 'pca', etc.)
        color : str or list
            Color peaks by metadata
        save : str, optional
            Path to save figure
        **kwargs
            Additional arguments for scv.pl.velocity_embedding
        """
        if not self.velocity_computed:
            raise ValueError("Velocity not computed. Call estimate_velocity() first.")
        
        scv.pl.velocity_embedding(
            self.adata, 
            basis=basis, 
            color=color,
            save=save,
            **kwargs
        )
    
    def plot_velocity_stream(self, 
                           basis: str = 'umap',
                           color: Optional[str] = None,
                           save: Optional[str] = None,
                           **kwargs):
        """
        Plot velocity as stream plot.
        
        Parameters:
        -----------
        basis : str
            Embedding basis
        color : str, optional
            Color peaks by metadata
        save : str, optional
            Path to save figure
        **kwargs
            Additional arguments for scv.pl.velocity_embedding_stream
        """
        if not self.velocity_computed:
            raise ValueError("Velocity not computed. Call estimate_velocity() first.")
        
        scv.pl.velocity_embedding_stream(
            self.adata,
            basis=basis,
            color=color,
            save=save,
            **kwargs
        )
    
    def plot_velocity_grid(self, 
                          basis: str = 'umap',
                          color: Optional[str] = None,
                          save: Optional[str] = None,
                          **kwargs):
        """
        Plot velocity as grid.
        
        Parameters:
        -----------
        basis : str
            Embedding basis
        color : str, optional
            Color peaks by metadata  
        save : str, optional
            Path to save figure
        **kwargs
            Additional arguments
        """
        if not self.velocity_computed:
            raise ValueError("Velocity not computed. Call estimate_velocity() first.")
        
        scv.pl.velocity_embedding_grid(
            self.adata,
            basis=basis,
            color=color,
            save=save,
            **kwargs
        )
    
    def analyze_temporal_patterns(self, 
                                 timepoint_col: str = 'timepoint_order',
                                 celltype_col: str = 'celltype') -> pd.DataFrame:
        """
        Analyze velocity patterns across timepoints and cell types.
        
        Parameters:
        -----------
        timepoint_col : str
            Column name for timepoint ordering in var
        celltype_col : str
            Column name for cell type in var
            
        Returns:
        --------
        DataFrame with velocity statistics per group
        """
        if not self.velocity_computed:
            raise ValueError("Velocity not computed. Call estimate_velocity() first.")
        
        if timepoint_col not in self.adata.var.columns:
            print(f"Warning: {timepoint_col} not found in var. Skipping temporal analysis.")
            return pd.DataFrame()
        
        results = []
        
        for celltype in self.adata.var[celltype_col].unique():
            celltype_mask = self.adata.var[celltype_col] == celltype
            celltype_pseudobulks = self.adata.var_names[celltype_mask]
            
            if len(celltype_pseudobulks) == 0:
                continue
            
            # Get velocity for this cell type
            celltype_velocity = self.adata[:, celltype_mask].layers['velocity']
            
            # Compute statistics
            mean_velocity = np.mean(celltype_velocity, axis=0)
            velocity_magnitude = np.sqrt((celltype_velocity**2).sum(axis=0))
            
            # Get timepoints for this cell type
            timepoints = self.adata.var.loc[celltype_pseudobulks, timepoint_col].values
            
            for i, (pb, tp, vel_mag) in enumerate(zip(celltype_pseudobulks, timepoints, velocity_magnitude)):
                results.append({
                    'celltype': celltype,
                    'pseudobulk': pb,
                    'timepoint': tp,
                    'velocity_magnitude': vel_mag,
                    'mean_velocity': mean_velocity[i]
                })
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            print("Temporal velocity analysis completed")
            print(f"Analyzed {len(results_df)} pseudobulk samples")
        
        return results_df


def run_chromatin_velocity_analysis(chromatin_velocity: ChromatinVelocity,
                                   mode: str = 'dynamical',
                                   n_neighbors: int = 30,
                                   min_confidence: float = 0.75,
                                   output_dir: str = './') -> ChromatinVelocityAnalysis:
    """
    Complete analysis pipeline for chromatin velocity.
    
    Parameters:
    -----------
    chromatin_velocity : ChromatinVelocity
        Computed chromatin velocity object
    mode : str
        scVelo mode for velocity estimation
    n_neighbors : int
        Number of neighbors for analysis
    min_confidence : float
        Minimum confidence for velocity genes
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    ChromatinVelocityAnalysis object with results
    """
    print("=== Chromatin Velocity Analysis Pipeline ===")
    
    # Initialize analysis object
    cva = ChromatinVelocityAnalysis(chromatin_velocity)
    
    # Prepare for scVelo
    cva.prepare_for_scvelo()
    
    # Compute moments
    cva.compute_moments(n_neighbors=n_neighbors)
    
    # Estimate velocity
    cva.estimate_velocity(mode=mode)
    
    # Compute confidence
    cva.compute_velocity_confidence()
    
    # Identify high-velocity pseudobulks
    velocity_pseudobulks = cva.identify_velocity_genes(min_confidence=min_confidence)
    
    # Generate summary plots
    if 'X_umap' in cva.adata.obsm:
        # Velocity embedding
        cva.plot_velocity_embedding(
            basis='umap',
            save=f'{output_dir}/chromatin_velocity_embedding.pdf'
        )
        
        # Velocity stream
        cva.plot_velocity_stream(
            basis='umap', 
            save=f'{output_dir}/chromatin_velocity_stream.pdf'
        )
        
        # Velocity grid
        cva.plot_velocity_grid(
            basis='umap',
            save=f'{output_dir}/chromatin_velocity_grid.pdf'
        )
    
    # Temporal analysis if metadata available
    if hasattr(chromatin_velocity, 'pseudobulk_metadata'):
        temporal_results = cva.analyze_temporal_patterns()
        if not temporal_results.empty:
            temporal_results.to_csv(f'{output_dir}/temporal_velocity_analysis.csv', index=False)
    
    # Save processed data
    cva.adata.write_h5ad(f'{output_dir}/chromatin_velocity_analysis.h5ad')
    
    print("=== Analysis Complete ===")
    return cva