"""
Chromatin Velocity Analysis Module

This module implements chromatin velocity calculation using co-accessibility propagation,
analogous to RNA velocity but for chromatin accessibility dynamics.

The core concept:
- "Spliced" analog: Current accessibility of each peak
- "Unspliced" analog: Weighted sum of co-accessible peaks' accessibility

Author: Generated for Zebrahub-Multiome analysis
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from typing import Optional, Tuple, Dict, List
import warnings
from tqdm import tqdm


class ChromatinVelocity:
    """
    Main class for chromatin velocity analysis using co-accessibility propagation.
    """
    
    def __init__(self, 
                 peaks_accessibility: np.ndarray,
                 peak_names: List[str],
                 pseudobulk_names: List[str],
                 coaccessibility_matrix: Optional[np.ndarray] = None,
                 peak_coordinates: Optional[pd.DataFrame] = None,
                 umap_coordinates: Optional[np.ndarray] = None):
        """
        Initialize ChromatinVelocity object.
        
        Parameters:
        -----------
        peaks_accessibility : np.ndarray
            Peak accessibility matrix (peaks x pseudobulks)
        peak_names : List[str]
            Names/IDs of peaks (e.g., "chr1-100-500")
        pseudobulk_names : List[str]
            Names of pseudobulk samples (e.g., "Neurons_15som", "PSM_20som")
        coaccessibility_matrix : np.ndarray, optional
            Peak-peak co-accessibility matrix from Cicero (peaks x peaks)
        peak_coordinates : pd.DataFrame, optional
            Genomic coordinates of peaks (columns: chr, start, end)
        umap_coordinates : np.ndarray, optional
            Peak UMAP coordinates (peaks x 2)
        """
        self.peaks_accessibility = peaks_accessibility
        self.peak_names = peak_names
        self.pseudobulk_names = pseudobulk_names
        self.coaccessibility_matrix = coaccessibility_matrix
        self.peak_coordinates = peak_coordinates
        self.umap_coordinates = umap_coordinates
        
        # Results storage
        self.spliced_counts = None
        self.unspliced_counts = None
        self.velocity = None
        self.adata = None
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data consistency."""
        n_peaks, n_pseudobulks = self.peaks_accessibility.shape
        
        if len(self.peak_names) != n_peaks:
            raise ValueError(f"Peak names length ({len(self.peak_names)}) doesn't match accessibility matrix rows ({n_peaks})")
        
        if len(self.pseudobulk_names) != n_pseudobulks:
            raise ValueError(f"Pseudobulk names length ({len(self.pseudobulk_names)}) doesn't match accessibility matrix columns ({n_pseudobulks})")
        
        if self.coaccessibility_matrix is not None:
            if self.coaccessibility_matrix.shape != (n_peaks, n_peaks):
                raise ValueError(f"Co-accessibility matrix shape {self.coaccessibility_matrix.shape} doesn't match peak count ({n_peaks}, {n_peaks})")
    
    def load_coaccessibility_matrix(self, 
                                   matrix_path: str,
                                   format: str = 'csv',
                                   threshold: float = 0.1,
                                   symmetric: bool = True):
        """
        Load co-accessibility matrix from file.
        
        Parameters:
        -----------
        matrix_path : str
            Path to co-accessibility matrix file
        format : str
            File format ('csv', 'tsv', 'npz', 'h5')
        threshold : float
            Minimum co-accessibility score to retain
        symmetric : bool
            Whether to make matrix symmetric (max of i,j and j,i)
        """
        print(f"Loading co-accessibility matrix from {matrix_path}")
        
        if format == 'csv':
            coaccess_df = pd.read_csv(matrix_path, index_col=0)
            self.coaccessibility_matrix = coaccess_df.values
        elif format == 'tsv':
            coaccess_df = pd.read_csv(matrix_path, sep='\t', index_col=0)
            self.coaccessibility_matrix = coaccess_df.values
        elif format == 'npz':
            self.coaccessibility_matrix = sp.load_npz(matrix_path).toarray()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Apply threshold
        self.coaccessibility_matrix[self.coaccessibility_matrix < threshold] = 0
        
        # Make symmetric if requested
        if symmetric:
            self.coaccessibility_matrix = np.maximum(
                self.coaccessibility_matrix, 
                self.coaccessibility_matrix.T
            )
        
        print(f"Loaded co-accessibility matrix: {self.coaccessibility_matrix.shape}")
        print(f"Non-zero entries: {np.count_nonzero(self.coaccessibility_matrix)}")
        print(f"Sparsity: {1 - np.count_nonzero(self.coaccessibility_matrix) / self.coaccessibility_matrix.size:.3f}")
    
    def compute_velocity_components(self, 
                                  normalize_accessibility: bool = True,
                                  min_coaccess_score: float = 0.1,
                                  max_connections: int = 100):
        """
        Compute 'spliced' and 'unspliced' analogs for chromatin velocity.
        
        Parameters:
        -----------
        normalize_accessibility : bool
            Whether to normalize accessibility values
        min_coaccess_score : float
            Minimum co-accessibility score to consider
        max_connections : int
            Maximum number of connections per peak to consider
        """
        if self.coaccessibility_matrix is None:
            raise ValueError("Co-accessibility matrix not loaded. Call load_coaccessibility_matrix() first.")
        
        print("Computing chromatin velocity components...")
        
        # Current accessibility ("spliced" analog)
        self.spliced_counts = self.peaks_accessibility.copy()
        
        if normalize_accessibility:
            # Log-normalize accessibility
            self.spliced_counts = np.log1p(self.spliced_counts)
        
        # Future potential ("unspliced" analog) via co-accessibility propagation
        self.unspliced_counts = np.zeros_like(self.spliced_counts)
        
        # For each peak, compute weighted sum of connected peaks' accessibility
        for i in tqdm(range(len(self.peak_names)), desc="Computing propagated accessibility"):
            # Get co-accessibility scores for peak i
            coaccess_scores = self.coaccessibility_matrix[i, :]
            
            # Filter by minimum score
            valid_connections = coaccess_scores >= min_coaccess_score
            
            # Limit to top connections if too many
            if np.sum(valid_connections) > max_connections:
                top_indices = np.argsort(coaccess_scores)[-max_connections:]
                valid_connections = np.zeros_like(valid_connections, dtype=bool)
                valid_connections[top_indices] = True
            
            if np.any(valid_connections):
                # Compute weighted sum of connected peaks' accessibility
                weights = coaccess_scores[valid_connections]
                connected_accessibility = self.spliced_counts[valid_connections, :]
                
                # Weighted average (normalized by sum of weights)
                self.unspliced_counts[i, :] = np.average(
                    connected_accessibility, 
                    weights=weights, 
                    axis=0
                )
            else:
                # If no connections, use own accessibility
                self.unspliced_counts[i, :] = self.spliced_counts[i, :]
        
        print(f"Computed velocity components for {len(self.peak_names)} peaks")
        print(f"Spliced range: [{self.spliced_counts.min():.3f}, {self.spliced_counts.max():.3f}]")
        print(f"Unspliced range: [{self.unspliced_counts.min():.3f}, {self.unspliced_counts.max():.3f}]")
    
    def add_temporal_ordering(self, 
                            timepoint_order: List[str],
                            celltype_grouping: Optional[Dict[str, List[str]]] = None):
        """
        Add temporal information to improve velocity directionality.
        
        Parameters:
        -----------
        timepoint_order : List[str]
            Ordered list of timepoints (e.g., ['0som', '5som', '10som', ...])
        celltype_grouping : Dict[str, List[str]], optional
            Grouping of related cell types for trajectory inference
        """
        # Parse timepoints from pseudobulk names
        pseudobulk_timepoints = []
        pseudobulk_celltypes = []
        
        for pb_name in self.pseudobulk_names:
            # Assuming format like "Neurons_15som" or "PSM_20som"
            parts = pb_name.split('_')
            if len(parts) >= 2:
                celltype = '_'.join(parts[:-1])
                timepoint = parts[-1]
            else:
                celltype = pb_name
                timepoint = 'unknown'
            
            pseudobulk_celltypes.append(celltype)
            pseudobulk_timepoints.append(timepoint)
        
        self.pseudobulk_metadata = pd.DataFrame({
            'pseudobulk': self.pseudobulk_names,
            'celltype': pseudobulk_celltypes,
            'timepoint': pseudobulk_timepoints
        })
        
        # Add timepoint ordering
        timepoint_to_order = {tp: i for i, tp in enumerate(timepoint_order)}
        self.pseudobulk_metadata['timepoint_order'] = [
            timepoint_to_order.get(tp, -1) for tp in pseudobulk_timepoints
        ]
        
        print(f"Added temporal ordering for {len(timepoint_order)} timepoints")
    
    def create_anndata_object(self, 
                            include_metadata: bool = True,
                            layer_names: Tuple[str, str] = ('spliced', 'unspliced')):
        """
        Create AnnData object compatible with scVelo.
        
        Parameters:
        -----------
        include_metadata : bool
            Whether to include peak annotations in obs
        layer_names : Tuple[str, str]
            Names for spliced and unspliced layers
        
        Returns:
        --------
        scanpy.AnnData
            AnnData object with peaks as observations, pseudobulks as variables
        """
        if self.spliced_counts is None or self.unspliced_counts is None:
            raise ValueError("Velocity components not computed. Call compute_velocity_components() first.")
        
        print("Creating AnnData object...")
        
        # Create AnnData with spliced counts as main X
        self.adata = sc.AnnData(
            X=self.spliced_counts,
            obs=pd.DataFrame(index=self.peak_names),
            var=pd.DataFrame(index=self.pseudobulk_names)
        )
        
        # Add layers
        self.adata.layers[layer_names[0]] = self.spliced_counts
        self.adata.layers[layer_names[1]] = self.unspliced_counts
        
        # Add UMAP coordinates if available
        if self.umap_coordinates is not None:
            self.adata.obsm['X_umap'] = self.umap_coordinates
        
        # Add peak metadata if available
        if include_metadata and self.peak_coordinates is not None:
            for col in self.peak_coordinates.columns:
                if len(self.peak_coordinates) == len(self.peak_names):
                    self.adata.obs[col] = self.peak_coordinates[col].values
        
        # Add pseudobulk metadata if available
        if hasattr(self, 'pseudobulk_metadata'):
            for col in self.pseudobulk_metadata.columns:
                if col != 'pseudobulk':
                    self.adata.var[col] = self.pseudobulk_metadata[col].values
        
        print(f"Created AnnData object: {self.adata.shape}")
        return self.adata
    
    def save_results(self, output_path: str, format: str = 'h5ad'):
        """
        Save results to file.
        
        Parameters:
        -----------
        output_path : str
            Path to save file
        format : str
            Output format ('h5ad', 'csv', 'npz')
        """
        if format == 'h5ad' and self.adata is not None:
            self.adata.write_h5ad(output_path)
        elif format == 'csv':
            # Save as separate CSV files
            base_path = output_path.replace('.csv', '')
            pd.DataFrame(self.spliced_counts, 
                        index=self.peak_names, 
                        columns=self.pseudobulk_names).to_csv(f"{base_path}_spliced.csv")
            pd.DataFrame(self.unspliced_counts, 
                        index=self.peak_names, 
                        columns=self.pseudobulk_names).to_csv(f"{base_path}_unspliced.csv")
        elif format == 'npz':
            np.savez_compressed(output_path,
                              spliced=self.spliced_counts,
                              unspliced=self.unspliced_counts,
                              peak_names=self.peak_names,
                              pseudobulk_names=self.pseudobulk_names)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved to {output_path}")


def load_data_from_anndata(adata_path: str, 
                          layer_name: str = 'normalized',
                          transpose: bool = True) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load peak accessibility data from AnnData object.
    
    Parameters:
    -----------
    adata_path : str
        Path to AnnData file
    layer_name : str
        Layer name to use for accessibility data
    transpose : bool
        Whether to transpose (if peaks are in var instead of obs)
    
    Returns:
    --------
    Tuple containing:
        - accessibility matrix (peaks x pseudobulks)
        - peak names
        - pseudobulk names
    """
    print(f"Loading data from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    if layer_name in adata.layers:
        accessibility = adata.layers[layer_name]
    else:
        accessibility = adata.X
    
    if transpose:
        accessibility = accessibility.T
        peak_names = list(adata.var.index)
        pseudobulk_names = list(adata.obs.index)
    else:
        peak_names = list(adata.obs.index)
        pseudobulk_names = list(adata.var.index)
    
    if sp.issparse(accessibility):
        accessibility = accessibility.toarray()
    
    print(f"Loaded accessibility matrix: {accessibility.shape}")
    print(f"Peaks: {len(peak_names)}, Pseudobulks: {len(pseudobulk_names)}")
    
    return accessibility, peak_names, pseudobulk_names


def compute_chromatin_velocity_pipeline(peaks_adata_path: str,
                                      coaccessibility_path: str,
                                      output_path: str,
                                      timepoint_order: List[str],
                                      coaccess_threshold: float = 0.1,
                                      max_connections: int = 100,
                                      normalize: bool = True) -> ChromatinVelocity:
    """
    Complete pipeline for chromatin velocity analysis.
    
    Parameters:
    -----------
    peaks_adata_path : str
        Path to peak accessibility AnnData file
    coaccessibility_path : str
        Path to co-accessibility matrix
    output_path : str
        Path to save results
    timepoint_order : List[str]
        Ordered list of developmental timepoints
    coaccess_threshold : float
        Threshold for co-accessibility scores
    max_connections : int
        Maximum connections per peak
    normalize : bool
        Whether to normalize accessibility values
    
    Returns:
    --------
    ChromatinVelocity object with computed results
    """
    print("=== Chromatin Velocity Pipeline ===")
    
    # 1. Load peak accessibility data
    accessibility, peak_names, pseudobulk_names = load_data_from_anndata(
        peaks_adata_path, layer_name='normalized', transpose=True
    )
    
    # 2. Initialize ChromatinVelocity object
    cv = ChromatinVelocity(
        peaks_accessibility=accessibility,
        peak_names=peak_names,
        pseudobulk_names=pseudobulk_names
    )
    
    # 3. Load co-accessibility matrix
    cv.load_coaccessibility_matrix(
        matrix_path=coaccessibility_path,
        format='csv',
        threshold=coaccess_threshold
    )
    
    # 4. Add temporal ordering
    cv.add_temporal_ordering(timepoint_order=timepoint_order)
    
    # 5. Compute velocity components
    cv.compute_velocity_components(
        normalize_accessibility=normalize,
        min_coaccess_score=coaccess_threshold,
        max_connections=max_connections
    )
    
    # 6. Create AnnData object
    cv.create_anndata_object()
    
    # 7. Save results
    cv.save_results(output_path, format='h5ad')
    
    print("=== Pipeline Complete ===")
    return cv