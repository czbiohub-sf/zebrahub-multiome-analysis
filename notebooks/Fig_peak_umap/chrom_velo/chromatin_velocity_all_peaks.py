#!/usr/bin/env python3
"""
Chromatin Velocity Analysis - ALL PEAKS

Compute chromatin velocity for ALL 640,830 peaks using memory-efficient processing
and project into 2D UMAP embedding with RNA velocity-style visualization.

Uses single-cell-base environment with careful memory management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import warnings
import gc
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')

print("=== Chromatin Velocity Analysis - ALL PEAKS ===")
print("Computing velocity for ALL 640,830 peaks with memory-efficient processing...")

# Set scanpy settings for memory optimization
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=80, facecolor='white')

def load_data_all_peaks():
    """Load original AnnData and co-accessibility data for all peaks."""
    
    print("\n1. Loading original peaks-by-pseudobulk AnnData (ALL PEAKS)...")
    
    # Load original data with UMAP coordinates
    adata_path = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad'
    
    try:
        adata = sc.read_h5ad(adata_path)
        print(f"✓ Loaded original data: {adata.shape}")
        print(f"✓ UMAP coordinates available: {adata.obsm['X_umap'].shape}")
        print(f"✓ Data layers: {list(adata.layers.keys()) if adata.layers else 'X matrix only'}")
    except Exception as e:
        print(f"Error loading original data: {e}")
        return None, None
    
    print("\n2. Loading co-accessibility data...")
    
    # Load co-accessibility connections
    coaccess_path = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv'
    
    try:
        print("Loading co-accessibility CSV (this may take a few minutes)...")
        coaccess_df = pd.read_csv(coaccess_path)
        print(f"✓ Loaded co-accessibility data: {coaccess_df.shape}")
        print(f"✓ Columns: {list(coaccess_df.columns)}")
    except Exception as e:
        print(f"Error loading co-accessibility data: {e}")
        return None, None
    
    return adata, coaccess_df

def create_coaccessibility_dict_efficient(coaccess_df, 
                                         peak_names_set,
                                         peak1_col='Peak1', 
                                         peak2_col='Peak2', 
                                         coaccess_col='coaccess',
                                         threshold=0.1):
    """
    Create co-accessibility dictionary efficiently for all peaks.
    """
    print(f"\n3. Creating co-accessibility connections for ALL peaks...")
    print(f"   Threshold: {threshold}")
    
    # Filter by threshold first
    print("   Filtering by co-accessibility threshold...")
    df_filtered = coaccess_df[coaccess_df[coaccess_col] >= threshold].copy()
    print(f"   Filtered from {len(coaccess_df):,} to {len(df_filtered):,} pairs")
    
    # Filter to peaks that exist in our dataset
    print("   Filtering to peaks in dataset...")
    mask1 = df_filtered[peak1_col].isin(peak_names_set)
    mask2 = df_filtered[peak2_col].isin(peak_names_set)
    df_filtered = df_filtered[mask1 & mask2].copy()
    print(f"   Filtered to {len(df_filtered):,} pairs with peaks in dataset")
    
    # Create connections dictionary
    print("   Building connections dictionary...")
    connections = {}
    
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing connections"):
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
    
    # Add self-loops for all peaks
    print("   Adding self-loops...")
    for peak in tqdm(peak_names_set, desc="Adding self-loops"):
        if peak not in connections:
            connections[peak] = {}
        connections[peak][peak] = 1.0
    
    print(f"✓ Created connections for {len(connections):,} peaks")
    
    # Calculate connection statistics
    connection_counts = [len(conn) for conn in connections.values()]
    print(f"   Connection statistics:")
    print(f"     Mean connections per peak: {np.mean(connection_counts):.1f}")
    print(f"     Max connections per peak: {max(connection_counts):,}")
    print(f"     Peaks with connections: {len(connections):,}")
    
    return connections

def compute_propagated_accessibility_all_peaks(adata, 
                                             connections_dict,
                                             accessibility_layer='normalized',
                                             min_coaccess_score=0.1,
                                             max_connections=100,
                                             batch_size=10000):
    """
    Compute propagated accessibility for all peaks using batch processing.
    """
    print(f"\n4. Computing propagated accessibility for ALL peaks...")
    print(f"   Using layer: {accessibility_layer}")
    print(f"   Batch size: {batch_size:,} peaks")
    
    # Get accessibility data
    if accessibility_layer in adata.layers:
        accessibility_matrix = adata.layers[accessibility_layer]
    else:
        accessibility_matrix = adata.X
    
    if hasattr(accessibility_matrix, 'toarray'):
        print("   Converting sparse matrix to dense (this may take time)...")
        accessibility_matrix = accessibility_matrix.toarray()
    
    print(f"   Accessibility matrix shape: {accessibility_matrix.shape}")
    
    # Get peak names
    peak_names = list(adata.obs_names)
    peak_name_to_idx = {name: idx for idx, name in enumerate(peak_names)}
    
    n_peaks, n_pseudobulks = accessibility_matrix.shape
    propagated = np.zeros_like(accessibility_matrix)
    
    # Process in batches to manage memory
    n_batches = (n_peaks + batch_size - 1) // batch_size
    print(f"   Processing {n_peaks:,} peaks in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_peaks)
        batch_peak_indices = list(range(start_idx, end_idx))
        
        print(f"   Batch {batch_idx + 1}/{n_batches}: processing peaks {start_idx:,}-{end_idx-1:,}")
        
        for i in tqdm(batch_peak_indices, desc=f"Batch {batch_idx + 1} progress"):
            peak_name = peak_names[i]
            
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
        
        # Force garbage collection after each batch
        gc.collect()
    
    print(f"✓ Propagated accessibility computed for {n_peaks:,} peaks")
    return accessibility_matrix, propagated

def compute_velocity_all_peaks(spliced_counts, unspliced_counts):
    """Compute velocity for all peaks."""
    print("\n5. Computing velocity vectors for ALL peaks...")
    
    # Normalize data
    print("   Normalizing spliced (current) accessibility...")
    spliced_norm = np.log1p(spliced_counts)
    
    print("   Normalizing unspliced (propagated) accessibility...")  
    unspliced_norm = np.log1p(unspliced_counts)
    
    # Compute velocity
    print("   Computing velocity as difference...")
    velocity = unspliced_norm - spliced_norm
    
    # Compute velocity magnitude
    velocity_magnitude = np.sqrt((velocity**2).sum(axis=1))
    
    print(f"✓ Velocity computed for {len(velocity_magnitude):,} peaks")
    print(f"   Velocity magnitude range: [{velocity_magnitude.min():.3f}, {velocity_magnitude.max():.3f}]")
    print(f"   Mean velocity magnitude: {velocity_magnitude.mean():.3f}")
    
    return spliced_norm, unspliced_norm, velocity, velocity_magnitude

def create_velocity_anndata_all_peaks(adata_original, 
                                    spliced_counts, 
                                    unspliced_counts, 
                                    velocity,
                                    velocity_magnitude):
    """Create velocity AnnData for all peaks."""
    print("\n6. Creating velocity AnnData for ALL peaks...")
    
    # Create new AnnData with velocity information
    adata_velocity = adata_original.copy()
    
    # Add layers
    adata_velocity.layers['spliced'] = spliced_counts
    adata_velocity.layers['unspliced'] = unspliced_counts  
    adata_velocity.layers['velocity'] = velocity
    
    # Add velocity magnitude to obs
    adata_velocity.obs['velocity_magnitude'] = velocity_magnitude
    
    print(f"✓ Created velocity AnnData: {adata_velocity.shape}")
    print(f"   Layers: {list(adata_velocity.layers.keys())}")
    
    return adata_velocity

def compute_velocity_embedding_all_peaks(adata_velocity, n_neighbors=15):
    """
    Compute velocity embedding in UMAP space for all peaks using efficient neighborhoods.
    """
    print(f"\n7. Computing velocity embedding for ALL peaks...")
    print(f"   Using {n_neighbors} neighbors for smoothing")
    
    # Get UMAP coordinates and velocity data
    umap_coords = adata_velocity.obsm['X_umap']
    velocity_matrix = adata_velocity.layers['velocity']
    velocity_magnitude = adata_velocity.obs['velocity_magnitude'].values
    
    print(f"   UMAP coordinates: {umap_coords.shape}")
    print(f"   Velocity matrix: {velocity_matrix.shape}")
    
    # Compute neighborhoods in UMAP space (using fewer neighbors for all peaks)
    print("   Computing neighborhoods in UMAP space...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1)
    nbrs.fit(umap_coords)
    distances, indices = nbrs.kneighbors(umap_coords)
    
    print("   Creating connectivity matrix...")
    n_peaks = len(umap_coords)
    
    # Use a more memory-efficient approach for large datasets
    # Instead of creating full sparse matrix, compute smoothed values directly
    print("   Computing smoothed velocity vectors...")
    
    # Smooth velocity magnitude
    velocity_magnitude_smooth = np.zeros(n_peaks)
    for i in tqdm(range(n_peaks), desc="Smoothing magnitudes"):
        neighbor_indices = indices[i, 1:]  # Skip self
        neighbor_distances = distances[i, 1:]  # Skip self
        
        # Weight by inverse distance
        weights = 1.0 / (neighbor_distances + 1e-6)
        weights = weights / weights.sum()
        
        # Weighted average
        velocity_magnitude_smooth[i] = np.average(velocity_magnitude[neighbor_indices], weights=weights)
    
    # Compute velocity direction using PCA of velocity matrix
    print("   Computing velocity directions...")
    from sklearn.decomposition import PCA
    
    if velocity_matrix.shape[1] > 1:
        # Use incremental PCA for memory efficiency
        from sklearn.decomposition import IncrementalPCA
        pca = IncrementalPCA(n_components=2, batch_size=min(1000, velocity_matrix.shape[1]))
        velocity_2d_raw = pca.fit_transform(velocity_matrix)
        
        # Scale by magnitude
        velocity_2d_magnitude = np.sqrt((velocity_2d_raw**2).sum(axis=1))
        velocity_2d_magnitude[velocity_2d_magnitude == 0] = 1  # Avoid division by zero
        
        # Normalize and scale by actual velocity magnitude
        velocity_2d_norm = velocity_2d_raw / velocity_2d_magnitude[:, np.newaxis]
        velocity_umap_raw = velocity_2d_norm * velocity_magnitude_smooth[:, np.newaxis]
    else:
        # If only one pseudobulk, use UMAP gradient
        print("   Using UMAP coordinate gradient for direction...")
        center_x, center_y = np.median(umap_coords[:, 0]), np.median(umap_coords[:, 1])
        velocity_umap_raw = np.column_stack([
            (umap_coords[:, 0] - center_x) * velocity_magnitude_smooth * 0.01,
            (umap_coords[:, 1] - center_y) * velocity_magnitude_smooth * 0.01
        ])
    
    # Smooth velocity directions using neighborhoods
    print("   Smoothing velocity directions...")
    velocity_umap_x_smooth = np.zeros(n_peaks)
    velocity_umap_y_smooth = np.zeros(n_peaks)
    
    for i in tqdm(range(n_peaks), desc="Smoothing directions"):
        neighbor_indices = indices[i, 1:]  # Skip self
        neighbor_distances = distances[i, 1:]  # Skip self
        
        # Weight by inverse distance
        weights = 1.0 / (neighbor_distances + 1e-6)
        weights = weights / weights.sum()
        
        # Weighted average of directions
        velocity_umap_x_smooth[i] = np.average(velocity_umap_raw[neighbor_indices, 0], weights=weights)
        velocity_umap_y_smooth[i] = np.average(velocity_umap_raw[neighbor_indices, 1], weights=weights)
    
    velocity_umap = np.column_stack([velocity_umap_x_smooth, velocity_umap_y_smooth])
    
    print(f"✓ Velocity embedding computed: {velocity_umap.shape}")
    print(f"   Embedding range X: [{velocity_umap[:, 0].min():.3f}, {velocity_umap[:, 0].max():.3f}]")
    print(f"   Embedding range Y: [{velocity_umap[:, 1].min():.3f}, {velocity_umap[:, 1].max():.3f}]")
    
    # Store results in adata
    adata_velocity.obsm['velocity_umap'] = velocity_umap
    adata_velocity.obs['velocity_magnitude_smooth'] = velocity_magnitude_smooth
    
    return adata_velocity

def plot_velocity_all_peaks(adata_velocity, 
                           save_prefix="chromatin_velocity_all_peaks",
                           max_arrows=5000):
    """Create velocity plots for all peaks with sampling for visualization."""
    
    print(f"\n8. Creating velocity visualizations for ALL peaks...")
    
    # Get data
    umap_coords = adata_velocity.obsm['X_umap']
    velocity_umap = adata_velocity.obsm['velocity_umap']
    velocity_magnitude = adata_velocity.obs['velocity_magnitude_smooth'].values
    
    print(f"   Total peaks: {len(umap_coords):,}")
    print(f"   Velocity magnitude range: [{velocity_magnitude.min():.3f}, {velocity_magnitude.max():.3f}]")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: All peaks colored by velocity magnitude
    ax = axes[0, 0]
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                        c=velocity_magnitude, cmap='viridis', 
                        s=0.1, alpha=0.6, rasterized=True)
    ax.set_title(f'Velocity Magnitude - ALL {len(umap_coords):,} Peaks', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
    
    # Plot 2: High-velocity peaks with arrows
    ax = axes[0, 1]
    # Background: all peaks
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
              c='lightgray', s=0.05, alpha=0.3, rasterized=True)
    
    # Select high-velocity peaks for arrows
    velocity_threshold = np.percentile(velocity_magnitude, 80)
    high_velocity_mask = velocity_magnitude >= velocity_threshold
    high_velocity_indices = np.where(high_velocity_mask)[0]
    
    # Sample arrows if too many
    if len(high_velocity_indices) > max_arrows:
        arrow_indices = np.random.choice(high_velocity_indices, max_arrows, replace=False)
    else:
        arrow_indices = high_velocity_indices
    
    # Plot high-velocity peaks
    ax.scatter(umap_coords[arrow_indices, 0], umap_coords[arrow_indices, 1], 
              c=velocity_magnitude[arrow_indices], cmap='plasma', 
              s=2, alpha=0.8, zorder=2)
    
    # Add velocity arrows
    quiver = ax.quiver(umap_coords[arrow_indices, 0], umap_coords[arrow_indices, 1],
                      velocity_umap[arrow_indices, 0], velocity_umap[arrow_indices, 1],
                      angles='xy', scale_units='xy', scale=0.5,
                      width=0.002, color='red', alpha=0.8, zorder=3)
    
    ax.set_title(f'Velocity Vectors - {len(arrow_indices):,} High-Velocity Peaks', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Plot 3: Velocity field density
    ax = axes[1, 0]
    
    # Create 2D histogram of velocity magnitudes
    hist, xbins, ybins = np.histogram2d(umap_coords[:, 0], umap_coords[:, 1], 
                                       bins=100, weights=velocity_magnitude)
    count_hist, _, _ = np.histogram2d(umap_coords[:, 0], umap_coords[:, 1], bins=100)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap = hist / count_hist
        heatmap[count_hist == 0] = 0
    
    im = ax.imshow(heatmap.T, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                  origin='lower', cmap='plasma', aspect='auto')
    
    plt.colorbar(im, ax=ax, label='Mean Velocity Magnitude')
    ax.set_title('Velocity Density Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Plot 4: Statistics and summary
    ax = axes[1, 1]
    
    # Velocity magnitude histogram
    ax.hist(velocity_magnitude, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Velocity Magnitude')
    ax.set_ylabel('Number of Peaks')
    ax.set_title('Velocity Magnitude Distribution', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    
    # Add statistics text
    stats_text = f"""ALL PEAKS VELOCITY STATISTICS
    
Total Peaks: {len(umap_coords):,}
Arrows Shown: {len(arrow_indices):,}

Velocity Magnitude:
  Mean: {velocity_magnitude.mean():.3f}
  Median: {np.median(velocity_magnitude):.3f}
  Max: {velocity_magnitude.max():.3f}
  
Percentiles:
  90th: {np.percentile(velocity_magnitude, 90):.3f}
  95th: {np.percentile(velocity_magnitude, 95):.3f}
  99th: {np.percentile(velocity_magnitude, 99):.3f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save comprehensive plot
    save_path_comprehensive = f"{save_prefix}_comprehensive.png"
    plt.savefig(save_path_comprehensive, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive plot saved: {save_path_comprehensive}")
    plt.show()
    
    # Create focused publication plot
    print("   Creating focused publication plot...")
    fig_focused, ax = plt.subplots(figsize=(14, 12), dpi=150)
    
    # Background: all peaks with velocity coloring
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                        c=velocity_magnitude, cmap='viridis', 
                        s=1, alpha=0.7, rasterized=True)
    
    # High-velocity arrows (fewer for clarity)
    velocity_threshold_focused = np.percentile(velocity_magnitude, 85)
    high_vel_focused = velocity_magnitude >= velocity_threshold_focused
    focused_indices = np.where(high_vel_focused)[0]
    
    if len(focused_indices) > 3000:
        focused_indices = np.random.choice(focused_indices, 3000, replace=False)
    
    quiver = ax.quiver(umap_coords[focused_indices, 0], umap_coords[focused_indices, 1],
                      velocity_umap[focused_indices, 0], velocity_umap[focused_indices, 1],
                      angles='xy', scale_units='xy', scale=0.3,
                      width=0.003, headwidth=3, headlength=4,
                      color='red', alpha=0.9, zorder=3)
    
    # Styling
    ax.set_xlabel('UMAP 1', fontsize=18)
    ax.set_ylabel('UMAP 2', fontsize=18)
    ax.set_title(f'Chromatin Velocity - ALL {len(umap_coords):,} Peaks', 
                fontsize=20, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Velocity Magnitude', fontsize=16)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    # Save focused plot
    save_path_focused = f"{save_prefix}_focused.png"
    plt.savefig(save_path_focused, dpi=300, bbox_inches='tight')
    print(f"✓ Focused plot saved: {save_path_focused}")
    plt.show()
    
    return fig, fig_focused

def main():
    """Main execution function for all peaks velocity analysis."""
    
    print("Starting complete chromatin velocity analysis for ALL peaks...")
    
    # Step 1: Load data
    adata_original, coaccess_df = load_data_all_peaks()
    if adata_original is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Step 2: Create co-accessibility connections
    peak_names_set = set(adata_original.obs_names)
    connections_dict = create_coaccessibility_dict_efficient(
        coaccess_df, peak_names_set, threshold=0.1
    )
    
    # Clear co-accessibility dataframe to save memory
    del coaccess_df
    gc.collect()
    
    # Step 3: Compute propagated accessibility
    accessibility_matrix, propagated_matrix = compute_propagated_accessibility_all_peaks(
        adata_original, connections_dict, batch_size=5000  # Smaller batches for memory
    )
    
    # Clear connections dict to save memory
    del connections_dict
    gc.collect()
    
    # Step 4: Compute velocity
    spliced_norm, unspliced_norm, velocity, velocity_magnitude = compute_velocity_all_peaks(
        accessibility_matrix, propagated_matrix
    )
    
    # Clear large matrices to save memory
    del accessibility_matrix, propagated_matrix
    gc.collect()
    
    # Step 5: Create velocity AnnData
    adata_velocity = create_velocity_anndata_all_peaks(
        adata_original, spliced_norm, unspliced_norm, velocity, velocity_magnitude
    )
    
    # Clear individual arrays
    del spliced_norm, unspliced_norm, velocity
    gc.collect()
    
    # Step 6: Compute velocity embedding
    adata_velocity = compute_velocity_embedding_all_peaks(adata_velocity, n_neighbors=10)
    
    # Step 7: Create visualizations
    fig_comp, fig_focused = plot_velocity_all_peaks(adata_velocity)
    
    # Step 8: Save results
    output_path = "chromatin_velocity_all_peaks_integrated.h5ad"
    print(f"\n9. Saving integrated results...")
    adata_velocity.write(output_path)
    print(f"✓ All peaks velocity data saved: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ CHROMATIN VELOCITY ANALYSIS COMPLETE - ALL PEAKS!")
    print("="*80)
    print(f"Successfully processed ALL {adata_velocity.shape[0]:,} peaks")
    print(f"Generated files:")
    print(f"- chromatin_velocity_all_peaks_comprehensive.png")
    print(f"- chromatin_velocity_all_peaks_focused.png")
    print(f"- chromatin_velocity_all_peaks_integrated.h5ad")
    
    return adata_velocity

if __name__ == "__main__":
    # Run the complete analysis for all peaks
    result_adata = main()