# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: sc_rapids
#     language: python
#     name: sc_rapids
# ---

# %% [markdown]
# # Chromatin Velocity Development Notebook
#
# Interactive development and testing notebook for chromatin velocity analysis.
# This notebook uses modular functions for step-by-step development and testing.

# %% [markdown]
# ## Setup and Imports

# %%
import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
# import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# rapids-singlecell
import cupy as cp
import rapids_singlecell as rsc

# Add scripts directory to path
sys.path.append('../scripts')

# Set up plotting
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")
sc.settings.verbosity = 2

print("Setup complete!")

# %% [markdown]
# ## 1. Data Loading Functions
# 
# Modular functions for loading different data types

# %%
def load_peak_accessibility_data(adata_path, layer_name='normalized', transpose=False):
    """Load peak accessibility data from AnnData file."""
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
    
    if hasattr(accessibility, 'toarray'):
        accessibility = accessibility.toarray()
    
    print(f"Loaded accessibility matrix: {accessibility.shape}")
    return accessibility, peak_names, pseudobulk_names, adata


# %%
def load_coaccessibility_matrix(matrix_path, format='csv', threshold=0.8, symmetric=True):
    """Load and preprocess co-accessibility matrix."""
    print(f"Loading co-accessibility matrix from {matrix_path}")
    
    if format == 'csv':
        coaccess_df = pd.read_csv(matrix_path, index_col=0)
        coaccess_matrix = coaccess_df.values
    elif format == 'tsv':
        coaccess_df = pd.read_csv(matrix_path, sep='\t', index_col=0)
        coaccess_matrix = coaccess_df.values
    elif format == 'npz':
        import scipy.sparse as sp
        coaccess_matrix = sp.load_npz(matrix_path).toarray()
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Apply threshold
    coaccess_matrix[coaccess_matrix < threshold] = 0
    
    # Make symmetric if requested
    if symmetric:
        coaccess_matrix = np.maximum(coaccess_matrix, coaccess_matrix.T)
    
    print(f"Co-accessibility matrix shape: {coaccess_matrix.shape}")
    print(f"Non-zero entries: {np.count_nonzero(coaccess_matrix)}")
    print(f"Sparsity: {1 - np.count_nonzero(coaccess_matrix) / coaccess_matrix.size:.3f}")
    
    return coaccess_matrix


# %%
def load_coaccessibility_matrix_longformat(matrix_path, 
                                         peak1_col='Peak1', 
                                         peak2_col='Peak2', 
                                         coaccess_col='coaccess',
                                         threshold=0.1, 
                                         symmetric=True,
                                         add_self_loops=True):
    """
    Load co-accessibility matrix from long format (Peak1, Peak2, coaccess) to peak-by-peak matrix.
    
    Parameters:
    -----------
    matrix_path : str
        Path to the co-accessibility file
    peak1_col : str
        Name of the first peak column
    peak2_col : str
        Name of the second peak column  
    coaccess_col : str
        Name of the co-accessibility score column
    threshold : float
        Minimum co-accessibility score to keep (default: 0.1)
    symmetric : bool
        Whether to make the matrix symmetric (default: True)
    add_self_loops : bool
        Whether to add self-loops with score 1.0 (default: True)
    
    Returns:
    --------
    coaccess_matrix : numpy.ndarray
        Peak-by-peak co-accessibility matrix
    peak_names : list
        List of peak names corresponding to matrix rows/columns
    """
    print(f"Loading co-accessibility matrix from {matrix_path}")
    
    # Load the data
    df = pd.read_csv(matrix_path)
    
    # Get column names (handle case where columns might be named differently)
    if peak1_col not in df.columns:
        # Try to find the peak columns automatically
        cols = df.columns.tolist()
        if len(cols) >= 3:
            peak1_col, peak2_col, coaccess_col = cols[:3]
            print(f"Auto-detected columns: {peak1_col}, {peak2_col}, {coaccess_col}")
        else:
            raise ValueError(f"Could not find {peak1_col} column in {df.columns}")
    
    print(f"Using columns: {peak1_col}, {peak2_col}, {coaccess_col}")
    
    # Filter by threshold
    df_filtered = df[df[coaccess_col] >= threshold].copy()
    print(f"Filtered from {len(df)} to {len(df_filtered)} pairs (threshold >= {threshold})")
    
    # Get unique peaks
    all_peaks = sorted(list(set(df_filtered[peak1_col].tolist() + df_filtered[peak2_col].tolist())))
    n_peaks = len(all_peaks)
    peak_to_idx = {peak: idx for idx, peak in enumerate(all_peaks)}
    
    print(f"Found {n_peaks} unique peaks")
    
    # Create the matrix
    coaccess_matrix = np.zeros((n_peaks, n_peaks))
    
    # Fill the matrix
    for _, row in df_filtered.iterrows():
        peak1_idx = peak_to_idx[row[peak1_col]]
        peak2_idx = peak_to_idx[row[peak2_col]]
        score = row[coaccess_col]
        
        # Add the score
        coaccess_matrix[peak1_idx, peak2_idx] = score
        
        # Make symmetric if requested
        if symmetric and peak1_idx != peak2_idx:
            coaccess_matrix[peak2_idx, peak1_idx] = score
    
    # Add self-loops if requested
    if add_self_loops:
        np.fill_diagonal(coaccess_matrix, 1.0)
        print("Added self-loops with score 1.0")
    
    print(f"Co-accessibility matrix shape: {coaccess_matrix.shape}")
    print(f"Non-zero entries: {np.count_nonzero(coaccess_matrix)}")
    print(f"Sparsity: {1 - np.count_nonzero(coaccess_matrix) / coaccess_matrix.size:.3f}")
    
    return coaccess_matrix, all_peaks


# %%
# Test data loading functions
# Update these paths with your actual data
test_adata_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad"

# Load a small sample to test
if os.path.exists(test_adata_path):
    accessibility, peak_names, pseudobulk_names, original_adata = load_peak_accessibility_data(test_adata_path)
    print(f"Test loading successful!")
    print(f"Peaks: {len(peak_names)}, Pseudobulks: {len(pseudobulk_names)}")
    print(f"First few peaks: {peak_names[:3]}")
    print(f"First few pseudobulks: {pseudobulk_names[:3]}")
else:
    print(f"Test file not found: {test_adata_path}")
    print("Will create mock data for testing")

# %%
# load the co-accessibility matrix
cicero_output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC/02_integrated_ATAC_cicero_connections_peaks_integrated.csv"
coaccess_matrix, peak_names_coaccess = load_coaccessibility_matrix_longformat(
    cicero_output_path, 
    peak1_col='Peak1', 
    peak2_col='Peak2', 
    coaccess_col='coaccess',
    threshold=0.1,  # Changed from 0.8 to 0.1 for more connections
    symmetric=True,
    add_self_loops=True
)


# %%
def test_longformat_function():
    """Test the longformat co-accessibility matrix function with sample data."""
    print("Testing longformat co-accessibility matrix function...")
    
    # Create sample data in the expected format
    sample_data = {
        'Peak1': ['1-14250900-14251605', '1-14250900-14251605', '1-14250900-14251605', '1-14251729-14252087'],
        'Peak2': ['1-14251729-14252087', '1-14252873-14253611', '1-14256357-14256643', '1-14252873-14253611'],
        'coaccess': [0.100097, 0.001212, 0.001017, 0.085]
    }
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        # Test the function
        matrix, peaks = load_coaccessibility_matrix_longformat(
            temp_path, 
            threshold=0.05,  # Low threshold to include most connections
            symmetric=True,
            add_self_loops=True
        )
        
        print(f"Sample matrix shape: {matrix.shape}")
        print(f"Peak names: {peaks}")
        print(f"Matrix diagonal (should be 1.0): {np.diag(matrix)}")
        print(f"Matrix symmetry check: {np.allclose(matrix, matrix.T)}")
        
        # Show a few entries
        print("\nSample matrix entries:")
        for i in range(min(3, len(peaks))):
            for j in range(min(3, len(peaks))):
                print(f"  {peaks[i]} <-> {peaks[j]}: {matrix[i, j]:.6f}")
        
        return matrix, peaks
        
    finally:
        # Clean up
        import os
        os.unlink(temp_path)

# Run the test
test_matrix, test_peaks = test_longformat_function()


# %%
def align_coaccessibility_with_accessibility(accessibility_peak_names, 
                                           coaccess_matrix, 
                                           coaccess_peak_names,
                                           fill_missing=0.0):
    """
    Align co-accessibility matrix with accessibility matrix peak order.
    
    Parameters:
    -----------
    accessibility_peak_names : list
        Peak names from accessibility matrix (desired order)
    coaccess_matrix : numpy.ndarray
        Co-accessibility matrix
    coaccess_peak_names : list
        Peak names from co-accessibility matrix (current order)
    fill_missing : float
        Value to fill for missing peaks (default: 0.0)
    
    Returns:
    --------
    aligned_matrix : numpy.ndarray
        Co-accessibility matrix aligned with accessibility peak order
    missing_peaks : list
        List of peaks in accessibility but not in co-accessibility
    extra_peaks : list
        List of peaks in co-accessibility but not in accessibility
    """
    print("Aligning co-accessibility matrix with accessibility matrix...")
    
    # Convert to sets for faster lookup
    accessibility_set = set(accessibility_peak_names)
    coaccess_set = set(coaccess_peak_names)
    
    # Find overlapping, missing, and extra peaks
    overlapping_peaks = accessibility_set.intersection(coaccess_set)
    missing_peaks = list(accessibility_set - coaccess_set)
    extra_peaks = list(coaccess_set - accessibility_set)
    
    print(f"Overlapping peaks: {len(overlapping_peaks)}")
    print(f"Missing peaks (in accessibility but not co-accessibility): {len(missing_peaks)}")
    print(f"Extra peaks (in co-accessibility but not accessibility): {len(extra_peaks)}")
    
    # Create mapping from old to new indices
    coaccess_peak_to_idx = {peak: idx for idx, peak in enumerate(coaccess_peak_names)}
    
    # Create aligned matrix
    n_accessibility_peaks = len(accessibility_peak_names)
    aligned_matrix = np.full((n_accessibility_peaks, n_accessibility_peaks), fill_missing)
    
    # Fill in the aligned matrix
    for i, peak_i in enumerate(accessibility_peak_names):
        for j, peak_j in enumerate(accessibility_peak_names):
            if peak_i in coaccess_peak_to_idx and peak_j in coaccess_peak_to_idx:
                old_i = coaccess_peak_to_idx[peak_i]
                old_j = coaccess_peak_to_idx[peak_j]
                aligned_matrix[i, j] = coaccess_matrix[old_i, old_j]
    
    # Add self-loops for missing peaks
    np.fill_diagonal(aligned_matrix, 1.0)
    
    print(f"Aligned matrix shape: {aligned_matrix.shape}")
    print(f"Alignment complete!")
    
    return aligned_matrix, missing_peaks, extra_peaks

# %% [markdown]
# ## 2. Core Velocity Computation Functions
#
# Modular functions for computing chromatin velocity components

# %%
def normalize_accessibility_data(accessibility_matrix, method='log1p'):
    """Normalize accessibility data."""
    if method == 'log1p':
        return np.log1p(accessibility_matrix)
    elif method == 'zscore':
        return (accessibility_matrix - accessibility_matrix.mean(axis=1, keepdims=True)) / accessibility_matrix.std(axis=1, keepdims=True)
    else:
        return accessibility_matrix


# %%
def compute_propagated_accessibility(accessibility_matrix, 
                                   coaccessibility_matrix,
                                   min_coaccess_score=0.1,
                                   max_connections=100):
    """Compute propagated accessibility (unspliced analog)."""
    print("Computing propagated accessibility...")
    
    n_peaks = accessibility_matrix.shape[0]
    propagated = np.zeros_like(accessibility_matrix)
    
    for i in tqdm(range(n_peaks), desc="Processing peaks"):
        # Get co-accessibility scores for peak i
        coaccess_scores = coaccessibility_matrix[i, :]
        
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
            connected_accessibility = accessibility_matrix[valid_connections, :]
            
            # Weighted average (normalized by sum of weights)
            propagated[i, :] = np.average(connected_accessibility, weights=weights, axis=0)
        else:
            # If no connections, use own accessibility
            propagated[i, :] = accessibility_matrix[i, :]
    
    print(f"Propagated accessibility computed for {n_peaks} peaks")
    return propagated


# %%
def compute_velocity_basic(spliced_counts, unspliced_counts):
    """Compute basic velocity as difference between unspliced and spliced."""
    return unspliced_counts - spliced_counts


# %%
# Test velocity computation with mock data
print("Testing velocity computation...")

# Create mock data for testing
n_peaks, n_pseudobulks = 100, 10
mock_accessibility = np.random.exponential(2, (n_peaks, n_pseudobulks))
mock_coaccess = np.random.exponential(0.1, (n_peaks, n_peaks))
np.fill_diagonal(mock_coaccess, 1.0)  # Self-accessibility = 1

# Normalize
mock_spliced = normalize_accessibility_data(mock_accessibility)

# Compute propagated
mock_unspliced = compute_propagated_accessibility(
    mock_accessibility, mock_coaccess, 
    min_coaccess_score=0.05, max_connections=20
)

# Compute velocity
mock_velocity = compute_velocity_basic(mock_spliced, mock_unspliced)

print(f"Mock velocity shape: {mock_velocity.shape}")
print(f"Velocity range: [{mock_velocity.min():.3f}, {mock_velocity.max():.3f}]")

# %% [markdown]
# ## 3. AnnData Creation and Metadata Functions

# %%
def create_velocity_anndata(spliced_counts, unspliced_counts, velocity,
                           peak_names, pseudobulk_names,
                           peak_metadata=None):
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
    
    # Add peak metadata if provided
    if peak_metadata is not None:
        for col in peak_metadata.columns:
            if len(peak_metadata) == len(peak_names):
                adata.obs[col] = peak_metadata[col].values
    
    print(f"Created AnnData: {adata.shape}")
    return adata


# %%
def add_temporal_metadata(adata, pseudobulk_names, timepoint_order):
    """Add temporal ordering metadata to AnnData."""
    print("Adding temporal metadata...")
    
    # Parse timepoints and celltypes from pseudobulk names
    pseudobulk_celltypes = []
    pseudobulk_timepoints = []
    
    for pb_name in pseudobulk_names:
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
    
    # Add to var (pseudobulks are variables)
    adata.var['celltype'] = pseudobulk_celltypes
    adata.var['timepoint'] = pseudobulk_timepoints
    
    # Add timepoint ordering
    timepoint_to_order = {tp: i for i, tp in enumerate(timepoint_order)}
    adata.var['timepoint_order'] = [timepoint_to_order.get(tp, -1) for tp in pseudobulk_timepoints]
    
    print(f"Added temporal metadata for {len(timepoint_order)} timepoints")
    return adata


# %%
# Test AnnData creation
print("Testing AnnData creation...")

mock_peak_names = [f"peak_{i}" for i in range(n_peaks)]
mock_pb_names = [f"Celltype{i%3}_{j}som" for i in range(n_pseudobulks) for j in [0, 5, 10, 15, 20]][:n_pseudobulks]

mock_adata = create_velocity_anndata(
    mock_spliced, mock_unspliced, mock_velocity,
    mock_peak_names, mock_pb_names
)

# Add temporal metadata
timepoint_order = ['0som', '5som', '10som', '15som', '20som']
mock_adata = add_temporal_metadata(mock_adata, mock_pb_names, timepoint_order)

print("AnnData creation successful!")
print(f"Layers: {list(mock_adata.layers.keys())}")
print(f"Var columns: {list(mock_adata.var.columns)}")

# %% [markdown]
# ## 4. Basic scVelo Integration Functions

# %%
def prepare_adata_for_scvelo(adata, min_shared_counts=20):
    """Prepare AnnData for scVelo analysis."""
    print("Preparing for scVelo...")
    
    # Calculate basic statistics
    n_peaks_per_pseudobulk = (adata.X > 0).sum(axis=0)
    mean_accessibility = np.array(adata.X.mean(axis=0)).flatten()
    
    # Filter pseudobulks based on criteria
    keep_pseudobulks = n_peaks_per_pseudobulk >= min_shared_counts
    
    print(f"Filtering: {keep_pseudobulks.sum()}/{len(keep_pseudobulks)} pseudobulks retained")
    
    if keep_pseudobulks.sum() > 0:
        adata_filtered = adata[:, keep_pseudobulks]
    else:
        warnings.warn("No pseudobulks passed filtering. Using all.")
        adata_filtered = adata
    
    # Set highly variable flag
    adata_filtered.var['highly_variable'] = True
    
    return adata_filtered


# %%
def compute_scvelo_moments(adata, n_neighbors=30, n_pcs=20):
    """Compute moments for scVelo."""
    print("Computing moments...")
    
    # PCA
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs)
    
    # Neighbors
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # Moments
    scv.pp.moments(adata, n_neighbors=n_neighbors)
    
    print("Moments computed")
    return adata


# %%
def estimate_velocity_scvelo(adata, mode='dynamical'):
    """Estimate velocity using scVelo."""
    print(f"Estimating velocity with {mode} mode...")
    
    scv.settings.verbosity = 2
    
    if mode == 'dynamical':
        scv.tl.recover_dynamics(adata)
        scv.tl.velocity(adata, mode='dynamical')
    else:
        scv.tl.velocity(adata, mode=mode)
    
    # Velocity graph
    scv.tl.velocity_graph(adata)
    
    print("Velocity estimation complete")
    return adata


# %%
# Test scVelo integration
print("Testing scVelo integration...")

# Prepare mock data
mock_adata_scvelo = prepare_adata_for_scvelo(mock_adata.copy())
mock_adata_scvelo = compute_scvelo_moments(mock_adata_scvelo, n_neighbors=5, n_pcs=5)

print("scVelo preparation successful!")
print(f"Final shape: {mock_adata_scvelo.shape}")

# %% [markdown]
# ## 5. Basic Visualization Functions

# %%
def plot_spliced_unspliced_relationship(adata, n_sample=1000, save=None):
    """Plot relationship between spliced and unspliced."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sample data for plotting
    n_sample = min(n_sample, adata.n_obs)
    sample_idx = np.random.choice(adata.n_obs, n_sample, replace=False)
    
    spliced = adata.layers['spliced'][sample_idx, :].flatten()
    unspliced = adata.layers['unspliced'][sample_idx, :].flatten()
    
    ax.scatter(spliced, unspliced, alpha=0.6, s=1)
    
    # Add diagonal
    min_val, max_val = min(spliced.min(), unspliced.min()), max(spliced.max(), unspliced.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    ax.set_xlabel('Spliced (Current Accessibility)')
    ax.set_ylabel('Unspliced (Propagated Accessibility)')
    ax.set_title('Spliced vs Unspliced Accessibility')
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()


# %%
def plot_velocity_distribution(adata, save=None):
    """Plot distribution of velocity magnitudes."""
    # Compute velocity magnitude
    velocity_mag = np.sqrt((adata.layers['velocity']**2).sum(axis=1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(velocity_mag, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Velocity Magnitude')
    ax1.set_ylabel('Number of Peaks')
    ax1.set_title('Velocity Magnitude Distribution')
    
    # Box plot
    ax2.boxplot(velocity_mag)
    ax2.set_ylabel('Velocity Magnitude')
    ax2.set_title('Velocity Magnitude')
    
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()
    
    return velocity_mag


# %%
# Test visualization functions
print("Testing visualization functions...")

plot_spliced_unspliced_relationship(mock_adata)
velocity_magnitudes = plot_velocity_distribution(mock_adata)

print(f"Velocity magnitude stats:")
print(f"  Mean: {velocity_magnitudes.mean():.3f}")
print(f"  Std: {velocity_magnitudes.std():.3f}")
print(f"  Range: [{velocity_magnitudes.min():.3f}, {velocity_magnitudes.max():.3f}]")

# %% [markdown]
# ## 6. Validation Functions

# %%
def validate_temporal_consistency(adata, temporal_col='timepoint_order'):
    """Validate velocity temporal consistency."""
    if temporal_col not in adata.var.columns:
        print(f"Warning: {temporal_col} not found")
        return {}
    
    velocity = adata.layers['velocity']
    timepoints = adata.var[temporal_col].values
    
    correlations = []
    for peak_idx in range(min(100, adata.n_obs)):  # Sample for testing
        peak_velocity = velocity[peak_idx, :]
        
        if np.all(peak_velocity == 0):
            continue
        
        valid_mask = ~np.isnan(peak_velocity) & ~np.isnan(timepoints)
        if np.sum(valid_mask) > 2:
            from scipy.stats import pearsonr
            corr, p_val = pearsonr(timepoints[valid_mask], peak_velocity[valid_mask])
            correlations.append(corr)
    
    if correlations:
        results = {
            'n_peaks': len(correlations),
            'mean_correlation': np.mean(correlations),
            'positive_fraction': np.mean(np.array(correlations) > 0)
        }
        print(f"Temporal validation: {results}")
        return results
    return {}


# %%
def identify_high_velocity_peaks(adata, top_n=20):
    """Identify peaks with highest velocity."""
    velocity_mag = np.sqrt((adata.layers['velocity']**2).sum(axis=1))
    top_indices = np.argsort(velocity_mag)[-top_n:]
    
    high_velocity_peaks = adata.obs_names[top_indices].tolist()
    
    print(f"Top {top_n} high-velocity peaks:")
    for i, peak in enumerate(high_velocity_peaks[-5:]):  # Show top 5
        print(f"  {peak}: {velocity_mag[top_indices[-(i+1)]]:.3f}")
    
    return high_velocity_peaks, velocity_mag[top_indices]


# %%
# Test validation functions
print("Testing validation functions...")

temporal_results = validate_temporal_consistency(mock_adata)
high_vel_peaks, high_vel_magnitudes = identify_high_velocity_peaks(mock_adata, top_n=10)

# %% [markdown]
# ## 7. Main Workflow Function

# %%
def run_chromatin_velocity_workflow(accessibility_matrix, 
                                  coaccessibility_matrix,
                                  peak_names, 
                                  pseudobulk_names,
                                  timepoint_order,
                                  normalize=True,
                                  coaccess_threshold=0.1,
                                  max_connections=100):
    """Complete modular workflow for chromatin velocity."""
    
    print("=== Starting Chromatin Velocity Workflow ===")
    
    # Step 1: Normalize accessibility
    if normalize:
        spliced_counts = normalize_accessibility_data(accessibility_matrix)
    else:
        spliced_counts = accessibility_matrix.copy()
    
    # Step 2: Compute propagated accessibility
    unspliced_counts = compute_propagated_accessibility(
        accessibility_matrix, coaccessibility_matrix,
        min_coaccess_score=coaccess_threshold,
        max_connections=max_connections
    )
    
    # Step 3: Compute velocity
    velocity = compute_velocity_basic(spliced_counts, unspliced_counts)
    
    # Step 4: Create AnnData
    adata = create_velocity_anndata(
        spliced_counts, unspliced_counts, velocity,
        peak_names, pseudobulk_names
    )
    
    # Step 5: Add temporal metadata
    adata = add_temporal_metadata(adata, pseudobulk_names, timepoint_order)
    
    # Step 6: Prepare for scVelo
    adata = prepare_adata_for_scvelo(adata)
    
    print("=== Workflow Complete ===")
    print(f"Final AnnData shape: {adata.shape}")
    
    return adata


# %%
# Test complete workflow
print("Testing complete workflow...")

test_result = run_chromatin_velocity_workflow(
    mock_accessibility, mock_coaccess,
    mock_peak_names, mock_pb_names,
    timepoint_order,
    normalize=True,
    coaccess_threshold=0.05,
    max_connections=20
)

print("Complete workflow test successful!")
print(f"Result: {test_result}")

# %% [markdown]
# ## 8. Ready for Real Data Testing
#
# Now we can test with real data when available

# %%
# Placeholder for real data testing
def test_with_real_data():
    """Test workflow with real data when paths are available."""
    
    # Real data paths (update these)
    real_adata_path = "/path/to/your/real/peaks_data.h5ad"
    real_coaccess_path = "/path/to/your/real/coaccessibility_matrix.csv"
    
    if os.path.exists(real_adata_path):
        print("Testing with real data...")
        
        # Load real data
        accessibility, peak_names, pseudobulk_names, original_adata = load_peak_accessibility_data(real_adata_path)
        
        # Load co-accessibility (if available)
        if os.path.exists(real_coaccess_path):
            coaccess_matrix = load_coaccessibility_matrix(real_coaccess_path)
            
            # Run workflow
            real_timepoints = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
            
            result = run_chromatin_velocity_workflow(
                accessibility, coaccess_matrix,
                peak_names, pseudobulk_names,
                real_timepoints
            )
            
            print("Real data test successful!")
            return result
        else:
            print("Co-accessibility matrix not found")
    else:
        print("Real data not available for testing")
    
    return None

# Uncomment to test with real data
# real_result = test_with_real_data()

# %% [markdown]
# ## Next Steps
#
# This modular notebook is ready for:
# 1. Testing individual functions with real data
# 2. Iterative development and debugging
# 3. Parameter optimization
# 4. Integration with scVelo for full velocity analysis
# 5. Validation and biological interpretation
#
# Each function can be tested and modified independently!

# %%
print("Chromatin velocity development notebook ready!")
print("All core functions tested with mock data.")
print("Ready for real data integration and iterative development.")

# %% [markdown]
# ## Function Summary
#
# **Data Loading:**
# - `load_peak_accessibility_data()`: Load peak accessibility matrix
# - `load_coaccessibility_matrix()`: Load and preprocess co-accessibility data
# - `load_coaccessibility_matrix_longformat()`: Load co-accessibility matrix from long format
#
# **Data Alignment:**
# - `align_coaccessibility_with_accessibility()`: Align co-accessibility matrix with accessibility peaks
#
# **Core Computation:**
# - `normalize_accessibility_data()`: Normalize accessibility values
# - `compute_propagated_accessibility()`: Compute co-accessibility propagation
# - `compute_velocity_basic()`: Compute velocity from spliced/unspliced
#
# **Data Structure:**
# - `create_velocity_anndata()`: Create AnnData for velocity analysis
# - `add_temporal_metadata()`: Add developmental timepoint information
#
# **scVelo Integration:**
# - `prepare_adata_for_scvelo()`: Prepare data for scVelo analysis
# - `compute_scvelo_moments()`: Compute neighborhood moments
# - `estimate_velocity_scvelo()`: Run scVelo velocity estimation
#
# **Visualization:**
# - `plot_spliced_unspliced_relationship()`: Plot spliced vs unspliced
# - `plot_velocity_distribution()`: Plot velocity magnitude distribution
#
# **Validation:**
# - `validate_temporal_consistency()`: Check velocity temporal coherence
# - `identify_high_velocity_peaks()`: Find peaks with highest velocity
#
# **Workflows:**
# - `run_chromatin_velocity_workflow()`: Complete modular pipeline
# - `run_chromatin_velocity_workflow_longformat()`: Complete pipeline with longformat co-accessibility
#
# **Testing:**
# - `test_longformat_function()`: Test longformat co-accessibility matrix loading

# %%
def run_chromatin_velocity_workflow_longformat(accessibility_path,
                                             coaccess_path,
                                             timepoint_order,
                                             accessibility_layer='normalized',
                                             peak1_col='Peak1',
                                             peak2_col='Peak2', 
                                             coaccess_col='coaccess',
                                             coaccess_threshold=0.1,
                                             normalize=True,
                                             max_connections=100):
    """
    Complete workflow for chromatin velocity with longformat co-accessibility matrix.
    
    Parameters:
    -----------
    accessibility_path : str
        Path to accessibility AnnData file
    coaccess_path : str
        Path to co-accessibility CSV file (Peak1, Peak2, coaccess format)
    timepoint_order : list
        List of timepoints in developmental order
    accessibility_layer : str
        Layer name in accessibility AnnData (default: 'normalized')
    peak1_col, peak2_col, coaccess_col : str
        Column names in co-accessibility file
    coaccess_threshold : float
        Minimum co-accessibility score to keep
    normalize : bool
        Whether to normalize accessibility data
    max_connections : int
        Maximum number of connections per peak
    
    Returns:
    --------
    adata : AnnData
        Velocity AnnData object ready for analysis
    """
    
    print("=== Starting Chromatin Velocity Workflow (Longformat) ===")
    
    # Step 1: Load accessibility data
    accessibility, peak_names, pseudobulk_names, original_adata = load_peak_accessibility_data(
        accessibility_path, layer_name=accessibility_layer
    )
    
    # Step 2: Load co-accessibility matrix (longformat)
    coaccess_matrix, coaccess_peak_names = load_coaccessibility_matrix_longformat(
        coaccess_path,
        peak1_col=peak1_col,
        peak2_col=peak2_col,
        coaccess_col=coaccess_col,
        threshold=coaccess_threshold,
        symmetric=True,
        add_self_loops=True
    )
    
    # Step 3: Align co-accessibility matrix with accessibility matrix
    aligned_coaccess_matrix, missing_peaks, extra_peaks = align_coaccessibility_with_accessibility(
        peak_names, coaccess_matrix, coaccess_peak_names, fill_missing=0.0
    )
    
    # Step 4: Run the main workflow
    adata = run_chromatin_velocity_workflow(
        accessibility, 
        aligned_coaccess_matrix,
        peak_names,
        pseudobulk_names,
        timepoint_order,
        normalize=normalize,
        coaccess_threshold=coaccess_threshold,
        max_connections=max_connections
    )
    
    # Step 5: Add alignment metadata
    adata.uns['alignment_info'] = {
        'missing_peaks': missing_peaks,
        'extra_peaks': extra_peaks,
        'n_missing': len(missing_peaks),
        'n_extra': len(extra_peaks),
        'alignment_coverage': (len(peak_names) - len(missing_peaks)) / len(peak_names)
    }
    
    print("=== Longformat Workflow Complete ===")
    print(f"Final AnnData shape: {adata.shape}")
    print(f"Alignment coverage: {adata.uns['alignment_info']['alignment_coverage']:.3f}")
    
    return adata

# %%
# Example usage with your data
def example_usage():
    """Example of how to use the new longformat workflow with your data."""
    
    # Your actual data paths
    accessibility_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad"
    coaccess_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC/02_integrated_ATAC_cicero_connections_peaks_integrated.csv"
    
    # Define your timepoint order
    timepoint_order = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']
    
    # Run the complete workflow
    if os.path.exists(accessibility_path) and os.path.exists(coaccess_path):
        print("Running complete chromatin velocity workflow...")
        
        adata = run_chromatin_velocity_workflow_longformat(
            accessibility_path=accessibility_path,
            coaccess_path=coaccess_path,
            timepoint_order=timepoint_order,
            accessibility_layer='normalized',
            peak1_col='Peak1',
            peak2_col='Peak2', 
            coaccess_col='coaccess',
            coaccess_threshold=0.1,  # Adjust as needed
            normalize=True,
            max_connections=100
        )
        
        print("Workflow completed successfully!")
        print(f"Result shape: {adata.shape}")
        print(f"Layers: {list(adata.layers.keys())}")
        
        # Print alignment information
        if 'alignment_info' in adata.uns:
            info = adata.uns['alignment_info']
            print(f"Alignment coverage: {info['alignment_coverage']:.3f}")
            print(f"Missing peaks: {info['n_missing']}")
            print(f"Extra peaks: {info['n_extra']}")
        
        return adata
    else:
        print("Data files not found. Please check the paths.")
        return None

# Uncomment to run with your actual data:
# result_adata = example_usage()

print("\n" + "="*60)
print("CHROMATIN VELOCITY NOTEBOOK READY!")
print("="*60)
print("\nNew functions added for longformat co-accessibility matrices:")
print("1. load_coaccessibility_matrix_longformat() - converts Peak1,Peak2,coaccess to matrix")
print("2. align_coaccessibility_with_accessibility() - aligns peak orders")
print("3. run_chromatin_velocity_workflow_longformat() - complete workflow")
print("\nYour data format (Peak1, Peak2, coaccess) is now supported!")
print("See example_usage() function for how to use with your data.")