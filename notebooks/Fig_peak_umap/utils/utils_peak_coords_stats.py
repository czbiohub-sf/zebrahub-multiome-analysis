# A script to calculate the distance between peaks and their neighbors
# and to calculate the statistics of the distances
# This script is optimized for GPU acceleration (sc_rapids conda env)

import time
import numpy as np
import pandas as pd
from scipy import sparse
import cupy as cp
import cudf
from numba import cuda
import scanpy as sc
import rapids_singlecell as rsc

# GPU version
def get_peak_coordinates_df(adata, gpu=True):
    """Create a DataFrame with peak coordinates from adata.obs.
    
    Args:
        adata: AnnData object with 'chrom', 'start', 'end' in adata.obs
        gpu: Whether to convert to GPU DataFrame
        
    Returns:
        cudf.DataFrame or pd.DataFrame: DataFrame with columns ['chrom', 'start', 'end', 'center']
    """
    # Check if necessary columns exist in adata.obs
    required_cols = ['chrom', 'start', 'end']
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Required column '{col}' not found in adata.obs")
    
    # Create a copy of the relevant columns
    coord_df = adata.obs[required_cols].copy()
    
    # Convert categorical columns to string to avoid GPU conversion issues
    for col in coord_df.columns:
        if pd.api.types.is_categorical_dtype(coord_df[col]):
            print(f"Converting categorical column '{col}' to string type")
            coord_df[col] = coord_df[col].astype(str)
    
    # Ensure start and end are integers
    coord_df['start'] = coord_df['start'].astype(int)
    coord_df['end'] = coord_df['end'].astype(int)
    
    # Calculate center position
    coord_df['center'] = coord_df['start'] + (coord_df['end'] - coord_df['start']) // 2
    
    # Convert to GPU DataFrame if requested
    if gpu:
        try:
            gpu_df = cudf.DataFrame.from_pandas(coord_df)
            print("Successfully converted coordinates to GPU DataFrame")
            return gpu_df
        except Exception as e:
            print(f"Warning: Failed to convert to GPU DataFrame: {e}")
            print("Falling back to CPU DataFrame")
            return coord_df
    
    return coord_df


def get_nearest_neighbors_from_obsp_gpu(adata, n_neighbors=15, use_connectivities=True):
    """Get nearest neighbors for each peak from adata.obsp using GPU acceleration.
    
    Args:
        adata: AnnData object with 'connectivities' or 'distances' in obsp
        n_neighbors: Number of neighbors to retrieve
        use_connectivities: Whether to use 'connectivities' (True) or 'distances' (False)
        
    Returns:
        dict: Dictionary mapping each peak index to a list of its nearest neighbor indices
    """
    matrix_key = 'connectivities' if use_connectivities else 'distances'
    if matrix_key not in adata.obsp:
        raise ValueError(f"'{matrix_key}' not found in adata.obsp")
    
    # Get the sparse matrix
    matrix = adata.obsp[matrix_key]
    
    # Convert scipy sparse matrix to cupy sparse matrix
    try:
        # For csr_matrix
        if isinstance(matrix, sparse.csr_matrix):
            cu_matrix = cp.sparse.csr_matrix(
                (cp.array(matrix.data), cp.array(matrix.indices), cp.array(matrix.indptr)),
                shape=matrix.shape
            )
        # For csc_matrix
        elif isinstance(matrix, sparse.csc_matrix):
            cu_matrix = cp.sparse.csc_matrix(
                (cp.array(matrix.data), cp.array(matrix.indices), cp.array(matrix.indptr)),
                shape=matrix.shape
            )
        else:
            # Convert to CSR first if it's another format
            matrix_csr = matrix.tocsr()
            cu_matrix = cp.sparse.csr_matrix(
                (cp.array(matrix_csr.data), cp.array(matrix_csr.indices), cp.array(matrix_csr.indptr)),
                shape=matrix_csr.shape
            )
            
        print(f"Successfully converted {matrix_key} to cupy sparse matrix")
    except Exception as e:
        print(f"Warning: Failed to convert to cupy sparse matrix: {e}")
        print("Falling back to CPU-based processing")
        return get_nearest_neighbors_from_obsp_cpu(adata, n_neighbors, use_connectivities)
    
    # Initialize dictionary to store neighbors
    peak_neighbors = {}
    
    # Process in batches to avoid GPU memory issues
    batch_size = 1000  # Adjust based on your GPU memory
    n_peaks = adata.shape[0]
    
    for batch_start in range(0, n_peaks, batch_size):
        batch_end = min(batch_start + batch_size, n_peaks)
        print(f"Processing peaks {batch_start} to {batch_end-1}...")
        
        # Extract batch of rows from sparse matrix
        batch_matrix = cu_matrix[batch_start:batch_end].toarray()
        
        # Find top neighbors (excluding self)
        if use_connectivities:
            # For connectivities, higher values indicate stronger connections
            # Negating values to use argsort (which sorts ascending)
            neighbor_indices = cp.argsort(-batch_matrix, axis=1)[:, 1:n_neighbors+1]
        else:
            # For distances, lower values indicate closer neighbors
            # Skip the first column which is self (distance 0)
            neighbor_indices = cp.argsort(batch_matrix, axis=1)[:, 1:n_neighbors+1]
        
        # Transfer to CPU and convert to Python objects
        neighbor_indices_cpu = cp.asnumpy(neighbor_indices)
        
        # Store in dictionary
        for i, idx in enumerate(range(batch_start, batch_end)):
            peak_neighbors[idx] = neighbor_indices_cpu[i].tolist()
    
    return peak_neighbors


def get_nearest_neighbors_from_obsp_cpu(adata, n_neighbors=15, use_connectivities=True):
    """CPU fallback method for getting nearest neighbors."""
    matrix_key = 'connectivities' if use_connectivities else 'distances'
    matrix = adata.obsp[matrix_key]
    peak_neighbors = {}
    
    for i in range(matrix.shape[0]):
        row = matrix[i].toarray().flatten()
        
        if use_connectivities:
            neighbor_indices = np.argsort(row)[::-1][1:n_neighbors+1]
        else:
            neighbor_indices = np.argsort(row)[1:n_neighbors+1]
        
        peak_neighbors[i] = neighbor_indices.tolist()
    
    return peak_neighbors


@cuda.jit
def compute_distances_kernel(chroms, centers, peak_indices, neighbor_indices, 
                             results, diff_chrom_penalty):
    """CUDA kernel to compute distances between peaks and their neighbors.
    
    Args:
        chroms: Array of chromosome indices (converted to integers)
        centers: Array of peak centers
        peak_indices: Array of peak indices to process
        neighbor_indices: 2D array of neighbor indices for each peak
        results: Output array for distances
        diff_chrom_penalty: Penalty value for different chromosomes
    """
    # Get thread ID
    i = cuda.grid(1)
    
    # Check if thread is within bounds
    if i < peak_indices.shape[0]:
        peak_idx = peak_indices[i]
        peak_chrom = chroms[peak_idx]
        
        # Process each neighbor
        for j in range(neighbor_indices.shape[1]):
            neighbor_idx = neighbor_indices[i, j]
            neighbor_chrom = chroms[neighbor_idx]
            
            # Calculate distance
            if peak_chrom == neighbor_chrom:
                # Same chromosome - compute actual distance
                distance = abs(centers[peak_idx] - centers[neighbor_idx])
            else:
                # Different chromosome - use penalty
                distance = diff_chrom_penalty
            
            # Store result
            results[i, j] = distance


def compute_distances_gpu(coord_df, peaks_to_process, neighbors_dict, diff_chrom_penalty):
    """Compute distances between peaks and their neighbors using GPU.
    
    Args:
        coord_df: DataFrame with coordinates
        peaks_to_process: List of peak indices to process
        neighbors_dict: Dictionary mapping peak indices to their neighbor indices
        diff_chrom_penalty: Penalty for different chromosomes
        
    Returns:
        dict: Dictionary mapping each peak index to distances to its neighbors
    """
    # Convert chrom to numeric codes
    if isinstance(coord_df, cudf.DataFrame):
        # For GPU DataFrame
        unique_chroms = coord_df['chrom'].unique().to_pandas()
        chrom_to_idx = {chrom: idx for idx, chrom in enumerate(unique_chroms)}
        
        # Create a new column for numeric chromosome indices
        coord_df = coord_df.copy()
        chrom_idx_list = [chrom_to_idx[c] for c in coord_df['chrom'].to_pandas()]
        coord_df['chrom_idx'] = chrom_idx_list
        chrom_idx = coord_df['chrom_idx'].values.get()
        centers = coord_df['center'].values.get()
    else:
        # For CPU DataFrame
        unique_chroms = coord_df['chrom'].unique()
        chrom_to_idx = {chrom: idx for idx, chrom in enumerate(unique_chroms)}
        
        # Create a new column for numeric chromosome indices
        coord_df = coord_df.copy()
        coord_df['chrom_idx'] = coord_df['chrom'].map(chrom_to_idx)
        chrom_idx = coord_df['chrom_idx'].values
        centers = coord_df['center'].values
    
    # Convert penalty to numeric
    if diff_chrom_penalty == 'max':
        penalty_value = 52660232  # Median zebrafish chromosome length
    elif diff_chrom_penalty is None:
        penalty_value = -1  # Special value to indicate exclusion
    else:
        penalty_value = float(diff_chrom_penalty)
    
    # Prepare data for GPU
    peaks_array = np.array(peaks_to_process, dtype=np.int32)
    
    # Get max number of neighbors for consistent array shape
    max_neighbors = max(len(neighbors_dict[p]) for p in peaks_to_process)
    
    # Create array of neighbor indices (padded with -1 for consistency)
    neighbors_array = np.full((len(peaks_to_process), max_neighbors), -1, dtype=np.int32)
    for i, peak_idx in enumerate(peaks_to_process):
        neighbors = neighbors_dict[peak_idx]
        neighbors_array[i, :len(neighbors)] = neighbors
    
    # Create result array
    distances_array = np.zeros((len(peaks_to_process), max_neighbors), dtype=np.float32)
    
    # Copy arrays to GPU
    d_chroms = cuda.to_device(chrom_idx)
    d_centers = cuda.to_device(centers)
    d_peaks = cuda.to_device(peaks_array)
    d_neighbors = cuda.to_device(neighbors_array)
    d_results = cuda.to_device(distances_array)
    
    # Define grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (len(peaks_to_process) + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    compute_distances_kernel[blocks_per_grid, threads_per_block](
        d_chroms, d_centers, d_peaks, d_neighbors, d_results, penalty_value
    )
    
    # Copy results back to host
    distances_array = d_results.copy_to_host()
    
    # Convert to dictionary format
    distances_dict = {}
    for i, peak_idx in enumerate(peaks_to_process):
        # Get valid neighbors
        valid_neighbors = [neighbors_array[i, j] for j in range(max_neighbors) 
                          if neighbors_array[i, j] >= 0]
        
        # Get corresponding distances
        valid_distances = [distances_array[i, j] for j in range(max_neighbors) 
                          if neighbors_array[i, j] >= 0]
        
        # Filter out exclusions if using 'None' penalty
        if diff_chrom_penalty is None:
            valid_pairs = [(n, d) for n, d in zip(valid_neighbors, valid_distances) if d >= 0]
            valid_neighbors = [n for n, _ in valid_pairs]
            valid_distances = [d for _, d in valid_pairs]
        
        # Store results
        distances_dict[peak_idx] = {
            'neighbor_indices': valid_neighbors,
            'distances': valid_distances
        }
    
    return distances_dict



def calculate_neighbor_distance_stats_gpu(adata, neighbors_dict, coord_df, diff_chrom_penalty=None):
    """Calculate statistics of distances between each peak and its neighbors using GPU.
    
    Args:
        adata: AnnData object
        neighbors_dict: Dictionary mapping each peak index to its neighbor peak indices
        coord_df: DataFrame with peak coordinates
        diff_chrom_penalty: How to handle different chromosome distances
        
    Returns:
        pd.DataFrame: DataFrame with distance statistics for each peak
    """
    # Compute distances using GPU
    print("Computing distances on GPU...")
    peak_indices = list(neighbors_dict.keys())
    
    # Process in batches to avoid GPU memory issues
    batch_size = 5000  # Adjust based on your GPU memory
    results = {}
    
    for i in range(0, len(peak_indices), batch_size):
        batch_indices = peak_indices[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch_indices)} peaks)...")
        
        batch_neighbors = {idx: neighbors_dict[idx] for idx in batch_indices}
        distances_dict = compute_distances_gpu(coord_df, batch_indices, batch_neighbors, diff_chrom_penalty)
        results.update(distances_dict)
    
    # Calculate statistics
    print("Calculating statistics...")
    stats_dict = {}
    
    # Set penalty value for same-chromosome check
    if diff_chrom_penalty == 'max':
        penalty_value = 52660232  # Median zebrafish chromosome length
    else:
        penalty_value = float('inf')  # Use infinity for other penalty types
    
    # Get chromosome info for all peaks (necessary for purity calculation)
    if isinstance(coord_df, cudf.DataFrame):
        # For GPU DataFrame - convert to pandas for easier access
        chrom_series = coord_df['chrom'].to_pandas()
    else:
        # For CPU DataFrame
        chrom_series = coord_df['chrom']
    
    for peak_idx, data in results.items():
        peak_id = adata.obs_names[peak_idx]
        distances = data['distances']
        neighbor_indices = data['neighbor_indices']
        
        # Skip if no valid distances
        if not distances:
            stats_dict[peak_id] = {
                'mean_distance': np.nan,
                'median_distance': np.nan,
                'std_distance': np.nan,
                'min_distance': np.nan,
                'max_distance': np.nan,
                'n_valid_neighbors': 0,
                'chrom_purity': 0,
                'n_same_chrom_neighbors': 0,
                'n_total_neighbors': len(neighbors_dict[peak_idx]),
                'same_chrom_mean_distance': np.nan,
                'same_chrom_median_distance': np.nan,
                'same_chrom_std_distance': np.nan,
            }
            continue
        
        # Get chromosome info for calculating purity
        peak_chrom = chrom_series.iloc[peak_idx]
        neighbor_chroms = [chrom_series.iloc[n] for n in neighbor_indices]
        
        # Count same-chromosome neighbors
        same_chrom_neighbors = sum(1 for nc in neighbor_chroms if nc == peak_chrom)
        total_neighbors = len(neighbor_indices)
        chrom_purity = same_chrom_neighbors / total_neighbors if total_neighbors > 0 else 0
        
        # Identify same-chromosome distances
        same_chrom_distances = []
        for n_idx, dist in zip(neighbor_indices, distances):
            neighbor_chrom = chrom_series.iloc[n_idx]
            if neighbor_chrom == peak_chrom and (diff_chrom_penalty is None or dist != penalty_value):
                same_chrom_distances.append(dist)
        
        # Calculate statistics
        stats = {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances) if len(distances) > 1 else 0,
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'n_valid_neighbors': len(distances),
            'chrom_purity': chrom_purity,
            'n_same_chrom_neighbors': same_chrom_neighbors,
            'n_total_neighbors': total_neighbors,
        }
        
        # Add same-chromosome statistics
        if same_chrom_distances:
            stats.update({
                'same_chrom_mean_distance': np.mean(same_chrom_distances),
                'same_chrom_median_distance': np.median(same_chrom_distances),
                'same_chrom_std_distance': np.std(same_chrom_distances) if len(same_chrom_distances) > 1 else 0,
            })
        else:
            stats.update({
                'same_chrom_mean_distance': np.nan,
                'same_chrom_median_distance': np.nan,
                'same_chrom_std_distance': np.nan,
            })
        
        stats_dict[peak_id] = stats
    
    return pd.DataFrame.from_dict(stats_dict, orient='index')


def analyze_peak_neighbor_distances_gpu(adata, n_neighbors=15, use_connectivities=True, diff_chrom_penalty=None):
    """Analyze distances between peaks and their neighbors using GPU acceleration.
    
    Args:
        adata: AnnData object with peaks as observations
        n_neighbors: Number of neighbors to consider
        use_connectivities: Whether to use connectivities (True) or distances (False) matrix
        diff_chrom_penalty: How to handle different chromosome distances
        
    Returns:
        pd.DataFrame: DataFrame with distance statistics for each peak
    """
    start_time = time.time()
    
    print(f"Extracting peak coordinates from adata.obs...")
    coord_df = get_peak_coordinates_df(adata, gpu=True)
    
    print(f"Finding {n_neighbors} nearest neighbors for each peak from adata.obsp...")
    neighbors_dict = get_nearest_neighbors_from_obsp_gpu(adata, n_neighbors, use_connectivities)
    
    print("Calculating distance statistics using GPU...")
    if diff_chrom_penalty == 'max':
        print("Using median zebrafish chromosome length (52.7 Mb) as penalty for different chromosomes")
    
    stats_df = calculate_neighbor_distance_stats_gpu(adata, neighbors_dict, coord_df, diff_chrom_penalty)
    
    elapsed_time = time.time() - start_time
    print(f"Done! Processed {adata.n_obs} peaks in {elapsed_time:.2f} seconds")
    
    return stats_df