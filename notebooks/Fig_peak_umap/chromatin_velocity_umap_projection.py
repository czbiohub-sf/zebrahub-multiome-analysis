#!/usr/bin/env python3
"""
Chromatin Velocity UMAP Projection

Project chromatin velocity vectors into the 2D UMAP embedding from the original AnnData object
using RNA velocity-style visualization with neighborhood-based smoothing.

Uses the single-cell-base environment with 16GB memory optimization.
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
warnings.filterwarnings('ignore')

print("=== Chromatin Velocity UMAP Projection ===")
print("Loading data and projecting velocity vectors into 2D UMAP embedding...")

# Set scanpy settings for memory optimization
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=80, facecolor='white')

def load_and_integrate_data():
    """Load original AnnData with UMAP and integrate with velocity results."""
    
    print("\n1. Loading original peaks-by-pseudobulk AnnData...")
    
    # Load original data with UMAP coordinates
    adata_path = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad'
    
    try:
        adata = sc.read_h5ad(adata_path)
        print(f"✓ Loaded original data: {adata.shape}")
        print(f"✓ UMAP coordinates available: {adata.obsm['X_umap'].shape}")
        print(f"✓ Neighborhood info: {adata.obsp['connectivities'].shape}")
    except Exception as e:
        print(f"Error loading original data: {e}")
        return None, None
    
    print("\n2. Loading chromatin velocity results...")
    
    # Load velocity results
    velocity_path = 'chromatin_velocity_results.h5ad'
    
    try:
        velocity_adata = sc.read_h5ad(velocity_path)
        print(f"✓ Loaded velocity data: {velocity_adata.shape}")
        print(f"✓ Velocity layers: {list(velocity_adata.layers.keys())}")
    except FileNotFoundError:
        print(f"Velocity results not found at {velocity_path}")
        print("Please ensure chromatin_velocity_results.h5ad is in the current directory")
        return None, None
    
    print("\n3. Integrating datasets...")
    
    # Find overlapping peaks
    original_peaks = set(adata.obs_names)
    velocity_peaks = set(velocity_adata.obs_names)
    overlapping_peaks = original_peaks.intersection(velocity_peaks)
    
    print(f"Original peaks: {len(original_peaks)}")
    print(f"Velocity peaks: {len(velocity_peaks)}")
    print(f"Overlapping peaks: {len(overlapping_peaks)}")
    
    if len(overlapping_peaks) == 0:
        print("ERROR: No overlapping peaks found!")
        return None, None
    
    # Filter to overlapping peaks
    overlapping_list = list(overlapping_peaks)
    adata_filtered = adata[overlapping_list, :].copy()
    velocity_filtered = velocity_adata[overlapping_list, :].copy()
    
    print(f"✓ Filtered to {len(overlapping_list)} overlapping peaks")
    
    return adata_filtered, velocity_filtered

def compute_velocity_embedding(adata, velocity_adata, n_neighbors=30):
    """
    Compute velocity embedding in UMAP space using neighborhood smoothing.
    Similar to scVelo's approach but for chromatin accessibility.
    """
    
    print(f"\n4. Computing velocity embedding with {n_neighbors} neighbors...")
    
    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']
    print(f"UMAP coordinates shape: {umap_coords.shape}")
    
    # Get velocity vectors from velocity_adata
    velocity_matrix = velocity_adata.layers['velocity']  # (n_peaks, n_pseudobulks)
    velocity_magnitude = np.sqrt((velocity_matrix**2).sum(axis=1))
    
    print(f"Velocity matrix shape: {velocity_matrix.shape}")
    print(f"Velocity magnitude range: [{velocity_magnitude.min():.3f}, {velocity_magnitude.max():.3f}]")
    
    # Always compute new neighborhoods for the filtered dataset
    print("Computing new neighborhoods in UMAP space for filtered data...")
    # Compute neighbors in UMAP space
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(umap_coords)
    distances, indices = nbrs.kneighbors(umap_coords)
    
    # Create connectivity matrix
    n_peaks = len(umap_coords)
    connectivities = sp.lil_matrix((n_peaks, n_peaks))
    for i in range(n_peaks):
        for j, neighbor_idx in enumerate(indices[i]):
            if j > 0:  # Skip self-connection
                weight = 1.0 / (distances[i, j] + 1e-6)
                connectivities[i, neighbor_idx] = weight
    connectivities = connectivities.tocsr()
    print(f"✓ Created connectivity matrix: {connectivities.shape}")
    
    # Smooth velocity vectors using neighborhoods
    print("Smoothing velocity vectors using neighborhood structure...")
    
    # Normalize connectivity matrix (row-wise normalization)
    connectivities_norm = connectivities.copy().astype(np.float64)
    
    # Get row sums 
    row_sums = np.array(connectivities_norm.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    
    # Normalize each row
    for i in range(connectivities_norm.shape[0]):
        start_idx = connectivities_norm.indptr[i]
        end_idx = connectivities_norm.indptr[i + 1]
        if end_idx > start_idx:  # If row has non-zero elements
            connectivities_norm.data[start_idx:end_idx] /= row_sums[i]
    
    # Smooth velocity magnitude using neighborhoods
    velocity_magnitude_smooth = connectivities_norm.dot(velocity_magnitude)
    
    # Compute velocity direction in UMAP space
    print("Computing velocity direction in UMAP space...")
    
    # Method 1: Use temporal progression if available
    # For now, use PCA of velocity matrix to get 2D direction
    from sklearn.decomposition import PCA
    
    if velocity_matrix.shape[1] > 1:
        pca = PCA(n_components=2)
        velocity_2d_raw = pca.fit_transform(velocity_matrix)
        
        # Scale by magnitude
        velocity_2d_magnitude = np.sqrt((velocity_2d_raw**2).sum(axis=1))
        velocity_2d_magnitude[velocity_2d_magnitude == 0] = 1  # Avoid division by zero
        
        # Normalize and scale by actual velocity magnitude
        velocity_2d_norm = velocity_2d_raw / velocity_2d_magnitude[:, np.newaxis]
        velocity_umap_raw = velocity_2d_norm * velocity_magnitude_smooth[:, np.newaxis]
        
    else:
        # If only one pseudobulk, create random directions
        angles = np.random.uniform(0, 2*np.pi, len(velocity_magnitude))
        velocity_umap_raw = np.column_stack([
            velocity_magnitude_smooth * np.cos(angles),
            velocity_magnitude_smooth * np.sin(angles)
        ])
    
    # Smooth velocity directions using neighborhoods
    velocity_umap_x = connectivities_norm.dot(velocity_umap_raw[:, 0])
    velocity_umap_y = connectivities_norm.dot(velocity_umap_raw[:, 1])
    
    velocity_umap = np.column_stack([velocity_umap_x, velocity_umap_y])
    
    print(f"✓ Velocity embedding computed: {velocity_umap.shape}")
    print(f"Velocity embedding range:")
    print(f"  X: [{velocity_umap[:, 0].min():.3f}, {velocity_umap[:, 0].max():.3f}]")
    print(f"  Y: [{velocity_umap[:, 1].min():.3f}, {velocity_umap[:, 1].max():.3f}]")
    
    # Store results in adata
    adata.obsm['velocity_umap'] = velocity_umap
    adata.obs['velocity_magnitude'] = velocity_magnitude_smooth
    adata.obs['velocity_magnitude_raw'] = velocity_magnitude
    
    return adata

def plot_velocity_embedding(adata, 
                           save_prefix="chromatin_velocity_umap",
                           min_mass=1.0, 
                           arrow_scale=1.0,
                           density=1.0):
    """
    Create RNA velocity-style plots of chromatin velocity in UMAP space.
    """
    
    print(f"\n5. Creating velocity visualization plots...")
    
    # Get data
    umap_coords = adata.obsm['X_umap']
    velocity_umap = adata.obsm['velocity_umap']
    velocity_magnitude = adata.obs['velocity_magnitude'].values
    
    # Filter for high-velocity peaks
    velocity_threshold = np.percentile(velocity_magnitude, 70)
    high_velocity_mask = velocity_magnitude >= velocity_threshold
    
    print(f"Showing {high_velocity_mask.sum()} high-velocity peaks (>70th percentile)")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Velocity magnitude heatmap
    ax = axes[0, 0]
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                        c=velocity_magnitude, cmap='viridis', 
                        s=1.5, alpha=0.7)
    ax.set_title('Chromatin Velocity Magnitude', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
    
    # Plot 2: Velocity vectors (arrows)
    ax = axes[0, 1]
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
              c='lightgray', s=0.5, alpha=0.3)
    
    # Sample arrows for visualization
    n_arrows = min(2000, high_velocity_mask.sum())
    if high_velocity_mask.sum() > n_arrows:
        arrow_indices = np.random.choice(np.where(high_velocity_mask)[0], n_arrows, replace=False)
    else:
        arrow_indices = np.where(high_velocity_mask)[0]
    
    quiver = ax.quiver(umap_coords[arrow_indices, 0], umap_coords[arrow_indices, 1],
                      velocity_umap[arrow_indices, 0], velocity_umap[arrow_indices, 1],
                      angles='xy', scale_units='xy', scale=1/arrow_scale,
                      width=0.002, color='red', alpha=0.6, zorder=3)
    
    ax.set_title(f'Velocity Vectors (n={len(arrow_indices)})', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Plot 3: Combined view
    ax = axes[1, 0]
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                        c=velocity_magnitude, cmap='plasma', 
                        s=2, alpha=0.6)
    
    # Add velocity streamlines for smooth flow visualization
    x_grid = np.linspace(umap_coords[:, 0].min(), umap_coords[:, 0].max(), 20)
    y_grid = np.linspace(umap_coords[:, 1].min(), umap_coords[:, 1].max(), 20)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate velocity field onto grid
    from scipy.interpolate import griddata
    
    try:
        U_grid = griddata((umap_coords[:, 0], umap_coords[:, 1]), velocity_umap[:, 0], 
                         (X_grid, Y_grid), method='linear', fill_value=0)
        V_grid = griddata((umap_coords[:, 0], umap_coords[:, 1]), velocity_umap[:, 1], 
                         (X_grid, Y_grid), method='linear', fill_value=0)
        
        # Create streamlines
        ax.streamplot(X_grid, Y_grid, U_grid, V_grid, 
                     density=density, color='white', linewidth=0.8, alpha=0.7)
                     
    except Exception as e:
        print(f"Note: Streamplot creation failed: {e}")
    
    ax.set_title('Velocity Field with Streamlines', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
    
    # Plot 4: Velocity divergence/convergence
    ax = axes[1, 1]
    
    # Compute velocity divergence (simplified)
    # Use finite differences on the interpolated field
    if 'U_grid' in locals():
        divergence = np.gradient(U_grid, axis=1) + np.gradient(V_grid, axis=0)
        
        im = ax.imshow(divergence, extent=[umap_coords[:, 0].min(), umap_coords[:, 0].max(),
                                         umap_coords[:, 1].min(), umap_coords[:, 1].max()],
                      origin='lower', cmap='RdBu_r', alpha=0.8)
        
        plt.colorbar(im, ax=ax, label='Velocity Divergence')
        ax.set_title('Velocity Divergence Field', fontsize=14, fontweight='bold')
    else:
        # Fallback: just show velocity magnitude with different colormap
        scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                            c=velocity_magnitude, cmap='coolwarm', 
                            s=2, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
        ax.set_title('Velocity Magnitude (Alt. View)', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    
    # Save plot
    save_path = f"{save_prefix}_embedding_plots.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Velocity embedding plots saved to: {save_path}")
    
    plt.show()
    
    return fig

def create_focused_velocity_plot(adata, save_path="chromatin_velocity_focused.png"):
    """Create a clean, focused velocity plot similar to RNA velocity papers."""
    
    print("\n6. Creating focused velocity plot...")
    
    # Get data
    umap_coords = adata.obsm['X_umap']
    velocity_umap = adata.obsm['velocity_umap']
    velocity_magnitude = adata.obs['velocity_magnitude'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # Background: all peaks with subtle coloring
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
              c=velocity_magnitude, cmap='viridis', 
              s=3, alpha=0.6, rasterized=True)
    
    # Velocity arrows for high-velocity peaks
    velocity_threshold = np.percentile(velocity_magnitude, 75)
    high_velocity_mask = velocity_magnitude >= velocity_threshold
    
    # Sample arrows to avoid overcrowding
    n_arrows = min(1500, high_velocity_mask.sum())
    if high_velocity_mask.sum() > n_arrows:
        arrow_indices = np.random.choice(np.where(high_velocity_mask)[0], n_arrows, replace=False)
    else:
        arrow_indices = np.where(high_velocity_mask)[0]
    
    # Add velocity arrows
    quiver = ax.quiver(umap_coords[arrow_indices, 0], umap_coords[arrow_indices, 1],
                      velocity_umap[arrow_indices, 0], velocity_umap[arrow_indices, 1],
                      angles='xy', scale_units='xy', scale=0.8,
                      width=0.003, headwidth=3, headlength=4,
                      color='red', alpha=0.8, zorder=3)
    
    # Styling
    ax.set_xlabel('UMAP 1', fontsize=16)
    ax.set_ylabel('UMAP 2', fontsize=16)
    ax.set_title('Chromatin Velocity Vectors in Peak UMAP Space', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.6)
    cbar.set_label('Velocity Magnitude', fontsize=14)
    
    # Add statistics
    stats_text = f"Peaks: {len(umap_coords):,}\\nArrows: {len(arrow_indices):,}\\nMean velocity: {velocity_magnitude.mean():.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Focused velocity plot saved to: {save_path}")
    
    plt.show()
    
    return fig

def main():
    """Main execution function."""
    
    # Load and integrate data
    adata, velocity_adata = load_and_integrate_data()
    if adata is None:
        print("Failed to load data. Exiting.")
        return
    
    # Compute velocity embedding
    adata = compute_velocity_embedding(adata, velocity_adata)
    
    # Create visualizations
    fig1 = plot_velocity_embedding(adata)
    fig2 = create_focused_velocity_plot(adata)
    
    # Save integrated data
    output_path = "chromatin_velocity_umap_integrated.h5ad"
    adata.write(output_path)
    print(f"\\n✓ Integrated data saved to: {output_path}")
    
    print("\\n" + "="*60)
    print("✓ CHROMATIN VELOCITY UMAP PROJECTION COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("- chromatin_velocity_umap_embedding_plots.png")  
    print("- chromatin_velocity_focused.png")
    print("- chromatin_velocity_umap_integrated.h5ad")
    
    return adata

if __name__ == "__main__":
    # Run the analysis
    result_adata = main()