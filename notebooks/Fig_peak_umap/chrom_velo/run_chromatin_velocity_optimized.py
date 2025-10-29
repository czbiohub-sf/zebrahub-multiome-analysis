#!/usr/bin/env python3
"""
Memory-optimized chromatin velocity analysis using sparse matrices and efficient processing.
This script handles the large scale data (640K peaks) with limited memory.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
import gc
from scipy import sparse
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Add scripts directory to path
sys.path.append('scripts')
sys.path.append('notebooks/Fig_peak_umap')

print("=== Memory-Optimized Chromatin Velocity Analysis ===")
print("Python version:", sys.version)

try:
    import scanpy as sc
    print("scanpy imported successfully")
    import matplotlib.pyplot as plt
    print("matplotlib imported successfully")
    
    # Set up basic plotting parameters
    plt.rcParams['figure.dpi'] = 100
    sc.settings.verbosity = 2
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the required environment is activated")
    sys.exit(1)

def load_peak_accessibility_data_efficient(adata_path, layer_name='normalized', max_peaks=None):
    """Load peak accessibility data with optional peak filtering for memory efficiency."""
    print(f"Loading data from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    # Get the layer data
    if layer_name in adata.layers:
        accessibility = adata.layers[layer_name]
    else:
        accessibility = adata.X
    
    peak_names = list(adata.obs.index)
    pseudobulk_names = list(adata.var.index)
    
    # Filter peaks if requested
    if max_peaks and len(peak_names) > max_peaks:
        print(f"Filtering to top {max_peaks} most variable peaks for memory efficiency...")
        
        # Calculate peak variability
        if hasattr(accessibility, 'toarray'):
            accessibility_dense = accessibility.toarray()
        else:
            accessibility_dense = accessibility
        
        peak_variances = np.var(accessibility_dense, axis=1)
        top_peak_indices = np.argsort(peak_variances)[-max_peaks:]
        
        # Filter data
        accessibility = accessibility_dense[top_peak_indices, :]
        peak_names = [peak_names[i] for i in top_peak_indices]
        
        print(f"Filtered to {len(peak_names)} peaks")
    else:
        if hasattr(accessibility, 'toarray'):
            accessibility = accessibility.toarray()
    
    print(f"Loaded accessibility matrix: {accessibility.shape}")
    return accessibility, peak_names, pseudobulk_names, adata

def load_coaccessibility_connections_dict(matrix_path, 
                                        peak1_col='Peak1', 
                                        peak2_col='Peak2', 
                                        coaccess_col='coaccess',
                                        threshold=0.1):
    """Load co-accessibility connections as a dictionary for memory efficiency."""
    print(f"Loading co-accessibility connections from {matrix_path}")
    
    # Load the data
    df = pd.read_csv(matrix_path)
    print(f"Loaded {len(df)} co-accessibility pairs")
    
    # Filter by threshold
    df_filtered = df[df[coaccess_col] >= threshold].copy()
    print(f"Filtered to {len(df_filtered)} pairs (threshold >= {threshold})")
    
    # Create connections dictionary
    connections = {}
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Building connections"):
        peak1 = row[peak1_col]
        peak2 = row[peak2_col]
        score = row[coaccess_col]
        
        # Add both directions
        if peak1 not in connections:
            connections[peak1] = {}
        if peak2 not in connections:
            connections[peak2] = {}
        
        connections[peak1][peak2] = score
        if peak1 != peak2:  # Avoid duplicate self-loops
            connections[peak2][peak1] = score
    
    # Add self-loops
    all_peaks = list(connections.keys())
    for peak in all_peaks:
        connections[peak][peak] = 1.0
    
    print(f"Created connections dictionary for {len(all_peaks)} peaks")
    return connections, all_peaks

def compute_propagated_accessibility_efficient(accessibility_matrix, 
                                             peak_names,
                                             connections_dict,
                                             min_coaccess_score=0.1,
                                             max_connections=100):
    """Compute propagated accessibility using connections dictionary."""
    print("Computing propagated accessibility efficiently...")
    
    n_peaks, n_pseudobulks = accessibility_matrix.shape
    propagated = np.zeros_like(accessibility_matrix)
    
    peak_name_to_idx = {name: idx for idx, name in enumerate(peak_names)}
    
    for i, peak_name in enumerate(tqdm(peak_names, desc="Processing peaks")):
        if peak_name in connections_dict:
            connections = connections_dict[peak_name]
            
            # Filter and limit connections
            valid_connections = [(connected_peak, score) for connected_peak, score in connections.items() 
                               if score >= min_coaccess_score and connected_peak in peak_name_to_idx]
            
            if len(valid_connections) > max_connections:
                # Keep top connections
                valid_connections.sort(key=lambda x: x[1], reverse=True)
                valid_connections = valid_connections[:max_connections]
            
            if valid_connections:
                # Compute weighted average
                weights = []
                connected_accessibility = []
                
                for connected_peak, score in valid_connections:
                    connected_idx = peak_name_to_idx[connected_peak]
                    weights.append(score)
                    connected_accessibility.append(accessibility_matrix[connected_idx, :])
                
                weights = np.array(weights)
                connected_accessibility = np.array(connected_accessibility)
                
                # Weighted average
                propagated[i, :] = np.average(connected_accessibility, weights=weights, axis=0)
            else:
                # No connections, use own accessibility
                propagated[i, :] = accessibility_matrix[i, :]
        else:
            # Peak not in co-accessibility data, use own accessibility
            propagated[i, :] = accessibility_matrix[i, :]
    
    print(f"Propagated accessibility computed for {n_peaks} peaks")
    return propagated

def normalize_accessibility_data(accessibility_matrix, method='log1p'):
    """Normalize accessibility data."""
    print(f"Normalizing accessibility data using {method}")
    if method == 'log1p':
        return np.log1p(accessibility_matrix)
    elif method == 'zscore':
        return (accessibility_matrix - accessibility_matrix.mean(axis=1, keepdims=True)) / \
               (accessibility_matrix.std(axis=1, keepdims=True) + 1e-10)
    else:
        return accessibility_matrix

def compute_velocity_basic(spliced_counts, unspliced_counts):
    """Compute basic velocity as difference between unspliced and spliced."""
    return unspliced_counts - spliced_counts

def create_velocity_anndata_efficient(spliced_counts, unspliced_counts, velocity,
                                    peak_names, pseudobulk_names):
    """Create AnnData object for velocity analysis."""
    print("Creating velocity AnnData...")
    
    # Create AnnData with spliced as main X
    adata = sc.AnnData(
        X=spliced_counts,
        obs=pd.DataFrame(index=peak_names),
        var=pd.DataFrame(index=pseudobulk_names)
    )
    
    # Add layers
    adata.layers['spliced'] = spliced_counts
    adata.layers['unspliced'] = unspliced_counts
    adata.layers['velocity'] = velocity
    
    print(f"Created AnnData: {adata.shape}")
    return adata

def add_temporal_metadata_efficient(adata, pseudobulk_names):
    """Add temporal ordering metadata to AnnData."""
    print("Adding temporal metadata...")
    
    # Parse timepoints and celltypes from pseudobulk names
    pseudobulk_celltypes = []
    pseudobulk_timepoints = []
    
    for pb_name in pseudobulk_names:
        # Handle various naming conventions
        if '_' in pb_name:
            parts = pb_name.split('_')
            if len(parts) >= 2:
                celltype = '_'.join(parts[:-1])
                timepoint = parts[-1]
            else:
                celltype = pb_name
                timepoint = 'unknown'
        else:
            celltype = pb_name
            timepoint = 'unknown'
        
        pseudobulk_celltypes.append(celltype)
        pseudobulk_timepoints.append(timepoint)
    
    # Add to var (pseudobulks are variables)
    adata.var['celltype'] = pseudobulk_celltypes
    adata.var['timepoint'] = pseudobulk_timepoints
    
    # Extract numeric timepoints for ordering
    timepoint_orders = []
    for tp in pseudobulk_timepoints:
        try:
            # Extract numbers from timepoint strings (e.g., "15som" -> 15)
            numeric_part = ''.join(filter(str.isdigit, tp))
            if numeric_part:
                timepoint_orders.append(int(numeric_part))
            else:
                timepoint_orders.append(-1)
        except:
            timepoint_orders.append(-1)
    
    adata.var['timepoint_order'] = timepoint_orders
    
    print(f"Added temporal metadata")
    print(f"Unique timepoints: {set(pseudobulk_timepoints)}")
    print(f"Unique celltypes: {set(pseudobulk_celltypes)}")
    
    return adata

# Main execution with memory optimization
if __name__ == "__main__":
    print("\n=== Starting Memory-Optimized Analysis ===")
    
    # Define file paths
    accessibility_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad"
    coaccess_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv"
    
    # Check if files exist
    if not os.path.exists(accessibility_path):
        print(f"Error: Accessibility file not found at {accessibility_path}")
        sys.exit(1)
    
    if not os.path.exists(coaccess_path):
        print(f"Error: Co-accessibility file not found at {coaccess_path}")
        sys.exit(1)
    
    print("Both required files found!")
    
    try:
        # Parameters for memory efficiency
        MAX_PEAKS = 50000  # Limit to 50K most variable peaks
        COACCESS_THRESHOLD = 0.15  # Higher threshold to reduce connections
        MAX_CONNECTIONS = 50  # Fewer connections per peak
        
        print(f"\nMemory optimization parameters:")
        print(f"- Max peaks: {MAX_PEAKS}")
        print(f"- Co-accessibility threshold: {COACCESS_THRESHOLD}")
        print(f"- Max connections per peak: {MAX_CONNECTIONS}")
        
        # Step 1: Load accessibility data with filtering
        print(f"\nStep 1: Loading peak accessibility data (top {MAX_PEAKS} peaks)...")
        accessibility, peak_names, pseudobulk_names, original_adata = load_peak_accessibility_data_efficient(
            accessibility_path, layer_name='normalized', max_peaks=MAX_PEAKS
        )
        print(f"Loaded {len(peak_names)} peaks × {len(pseudobulk_names)} pseudobulks")
        
        # Force garbage collection
        del original_adata
        gc.collect()
        
        # Step 2: Load co-accessibility connections as dictionary
        print(f"\nStep 2: Loading co-accessibility connections (threshold >= {COACCESS_THRESHOLD})...")
        connections_dict, coaccess_peak_names = load_coaccessibility_connections_dict(
            coaccess_path,
            peak1_col='Peak1',
            peak2_col='Peak2',
            coaccess_col='coaccess',
            threshold=COACCESS_THRESHOLD
        )
        
        # Step 3: Normalize accessibility data
        print(f"\nStep 3: Normalizing accessibility data...")
        spliced_counts = normalize_accessibility_data(accessibility, method='log1p')
        
        # Step 4: Compute propagated accessibility
        print(f"\nStep 4: Computing propagated accessibility...")
        unspliced_counts = compute_propagated_accessibility_efficient(
            accessibility, peak_names, connections_dict,
            min_coaccess_score=COACCESS_THRESHOLD,
            max_connections=MAX_CONNECTIONS
        )
        
        # Step 5: Compute velocity
        print(f"\nStep 5: Computing velocity...")
        velocity = compute_velocity_basic(spliced_counts, unspliced_counts)
        
        # Step 6: Create AnnData object
        print(f"\nStep 6: Creating velocity AnnData...")
        adata_velocity = create_velocity_anndata_efficient(
            spliced_counts, unspliced_counts, velocity,
            peak_names, pseudobulk_names
        )
        
        # Step 7: Add temporal metadata
        print(f"\nStep 7: Adding temporal metadata...")
        adata_velocity = add_temporal_metadata_efficient(adata_velocity, pseudobulk_names)
        
        # Step 8: Compute summary statistics
        print("\n=== Analysis Summary ===")
        print(f"Final AnnData shape: {adata_velocity.shape}")
        print(f"Layers: {list(adata_velocity.layers.keys())}")
        
        # Velocity statistics
        velocity_magnitude = np.sqrt((velocity**2).sum(axis=1))
        print(f"\nVelocity Statistics:")
        print(f"- Mean magnitude: {velocity_magnitude.mean():.4f}")
        print(f"- Std magnitude: {velocity_magnitude.std():.4f}")
        print(f"- Range: [{velocity_magnitude.min():.4f}, {velocity_magnitude.max():.4f}]")
        
        # Peak coverage in co-accessibility
        peaks_with_connections = sum(1 for peak in peak_names if peak in connections_dict)
        coverage = peaks_with_connections / len(peak_names)
        print(f"\nCo-accessibility Coverage:")
        print(f"- Peaks with connections: {peaks_with_connections}/{len(peak_names)} ({coverage:.3f})")
        
        # Temporal information
        unique_timepoints = adata_velocity.var['timepoint'].unique()
        unique_celltypes = adata_velocity.var['celltype'].unique()
        print(f"\nTemporal Information:")
        print(f"- Unique timepoints: {len(unique_timepoints)} ({list(unique_timepoints)})")
        print(f"- Unique celltypes: {len(unique_celltypes)}")
        
        # Save results
        output_path = "chromatin_velocity_results.h5ad"
        print(f"\nSaving results to {output_path}...")
        adata_velocity.write(output_path)
        
        print("\n=== Analysis Complete! ===")
        print(f"Chromatin velocity analysis successful!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)