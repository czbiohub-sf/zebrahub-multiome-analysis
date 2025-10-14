#!/usr/bin/env python3
"""
Improved Peak UMAP Chromatin Velocity Visualizer

Creates coherent velocity fields using proper neighborhood smoothing in UMAP space
and generates streamplot visualizations showing developmental flow patterns.

Author: Zebrahub-Multiome Analysis Pipeline  
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import warnings
import gc
from tqdm import tqdm
warnings.filterwarnings('ignore')

print("=== Improved Chromatin Velocity Analysis with Neighborhood Smoothing ===")

class ImprovedVelocityVisualizer:
    """
    Improved visualizer with proper UMAP neighborhood-based velocity smoothing.
    """
    
    def __init__(self, velocity_data_path):
        """Initialize with velocity data."""
        self.velocity_data_path = velocity_data_path
        self.adata = None
        self.umap_coords = None
        self.velocity_raw = None
        self.velocity_smoothed = None
        self.grid_velocity = None
        
    def load_velocity_data(self):
        """Load the pre-computed velocity data."""
        print(f"\n1. Loading velocity data from {self.velocity_data_path}...")
        
        try:
            self.adata = sc.read_h5ad(self.velocity_data_path)
            print(f"✓ Loaded velocity data: {self.adata.shape}")
            
            # Get UMAP coordinates
            self.umap_coords = self.adata.obsm['X_umap']
            print(f"✓ UMAP coordinates: {self.umap_coords.shape}")
            
            # Get velocity vectors (use the fast temporal approach)
            if 'velocity_umap_fast' in self.adata.obsm:
                self.velocity_raw = self.adata.obsm['velocity_umap_fast']
            elif 'velocity_temporal' in self.adata.layers:
                # If we only have temporal velocity, we need to project to UMAP
                velocity_temporal = self.adata.layers['velocity_temporal']
                print("   Projecting temporal velocity to UMAP space...")
                
                # Use PCA to get 2D velocity direction
                from sklearn.decomposition import TruncatedSVD
                if velocity_temporal.shape[1] > 2:
                    svd = TruncatedSVD(n_components=2, random_state=42)
                    velocity_2d = svd.fit_transform(velocity_temporal)
                else:
                    velocity_2d = velocity_temporal[:, :2]
                
                # Scale by velocity magnitude
                velocity_magnitude = self.adata.obs['velocity_magnitude'].values
                velocity_norm = np.sqrt((velocity_2d**2).sum(axis=1))
                velocity_norm[velocity_norm == 0] = 1
                
                velocity_2d_normed = velocity_2d / velocity_norm[:, np.newaxis]
                self.velocity_raw = velocity_2d_normed * velocity_magnitude[:, np.newaxis] * 0.1
            else:
                raise ValueError("No velocity data found in AnnData object")
            
            print(f"✓ Raw velocity vectors: {self.velocity_raw.shape}")
            print(f"   Velocity range: X[{self.velocity_raw[:, 0].min():.3f}, {self.velocity_raw[:, 0].max():.3f}], Y[{self.velocity_raw[:, 1].min():.3f}, {self.velocity_raw[:, 1].max():.3f}]")
            
        except Exception as e:
            print(f"Error loading velocity data: {e}")
            return False
        
        return True
    
    def compute_neighborhood_smoothed_velocity(self, k_neighbors=30, distance_weight=True):
        """
        Compute neighborhood-smoothed velocity using k-NN in UMAP space.
        """
        print(f"\n2. Computing neighborhood-smoothed velocity (k={k_neighbors})...")
        
        # Build k-NN tree in UMAP space
        print(f"   Building k-NN tree for {len(self.umap_coords):,} peaks...")
        tree = cKDTree(self.umap_coords)
        
        # Find neighbors for all points
        print(f"   Finding {k_neighbors} nearest neighbors...")
        distances, indices = tree.query(self.umap_coords, k=k_neighbors+1)  # +1 because it includes self
        
        # Remove self (first neighbor)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        print(f"   Computing smoothed velocity vectors...")
        smoothed_velocity = np.zeros_like(self.velocity_raw)
        
        for i in tqdm(range(len(self.umap_coords)), desc="Smoothing velocity"):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            if distance_weight:
                # Weight by inverse distance
                weights = 1.0 / (neighbor_distances + 1e-6)
                weights = weights / weights.sum()
            else:
                # Equal weights
                weights = np.ones(len(neighbor_indices)) / len(neighbor_indices)
            
            # Weighted average of neighbor velocities
            neighbor_velocities = self.velocity_raw[neighbor_indices]
            smoothed_velocity[i] = np.average(neighbor_velocities, axis=0, weights=weights)
        
        self.velocity_smoothed = smoothed_velocity
        
        print(f"✓ Smoothed velocity computed")
        print(f"   Smoothed range: X[{smoothed_velocity[:, 0].min():.3f}, {smoothed_velocity[:, 0].max():.3f}], Y[{smoothed_velocity[:, 1].min():.3f}, {smoothed_velocity[:, 1].max():.3f}]")
        
        # Compute smoothed magnitude
        smoothed_magnitude = np.sqrt((smoothed_velocity**2).sum(axis=1))
        self.adata.obs['velocity_magnitude_smoothed'] = smoothed_magnitude
        self.adata.obsm['velocity_umap_smoothed'] = smoothed_velocity
        
        return smoothed_velocity
    
    def interpolate_velocity_to_grid(self, grid_resolution=50):
        """
        Interpolate smoothed velocity onto a regular grid for streamplot.
        """
        print(f"\n3. Interpolating velocity to grid (resolution {grid_resolution}x{grid_resolution})...")
        
        if self.velocity_smoothed is None:
            print("   Error: No smoothed velocity available. Run compute_neighborhood_smoothed_velocity first.")
            return None
        
        # Create regular grid
        x_min, x_max = self.umap_coords[:, 0].min(), self.umap_coords[:, 0].max()
        y_min, y_max = self.umap_coords[:, 1].min(), self.umap_coords[:, 1].max()
        
        # Add padding
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        
        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, grid_resolution)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        print(f"   Grid bounds: X[{x_min-x_pad:.2f}, {x_max+x_pad:.2f}], Y[{y_min-y_pad:.2f}, {y_max+y_pad:.2f}]")
        
        # Interpolate velocity components
        print("   Interpolating velocity components...")
        
        try:
            # Use linear interpolation with nearest neighbor fallback
            U_grid = griddata(
                self.umap_coords, self.velocity_smoothed[:, 0],
                (X_grid, Y_grid), method='linear', fill_value=0
            )
            V_grid = griddata(
                self.umap_coords, self.velocity_smoothed[:, 1],
                (X_grid, Y_grid), method='linear', fill_value=0
            )
            
            # Fill any remaining NaN values with nearest neighbor
            mask_nan = np.isnan(U_grid) | np.isnan(V_grid)
            if np.any(mask_nan):
                print("   Filling NaN values with nearest neighbor interpolation...")
                U_grid_nn = griddata(
                    self.umap_coords, self.velocity_smoothed[:, 0],
                    (X_grid, Y_grid), method='nearest'
                )
                V_grid_nn = griddata(
                    self.umap_coords, self.velocity_smoothed[:, 1],
                    (X_grid, Y_grid), method='nearest'
                )
                
                U_grid[mask_nan] = U_grid_nn[mask_nan]
                V_grid[mask_nan] = V_grid_nn[mask_nan]
            
        except Exception as e:
            print(f"   Linear interpolation failed: {e}")
            print("   Falling back to nearest neighbor interpolation...")
            U_grid = griddata(
                self.umap_coords, self.velocity_smoothed[:, 0],
                (X_grid, Y_grid), method='nearest'
            )
            V_grid = griddata(
                self.umap_coords, self.velocity_smoothed[:, 1],
                (X_grid, Y_grid), method='nearest'
            )
        
        self.grid_velocity = {
            'X_grid': X_grid,
            'Y_grid': Y_grid,
            'U_grid': U_grid,
            'V_grid': V_grid,
            'speed_grid': np.sqrt(U_grid**2 + V_grid**2)
        }
        
        print(f"✓ Grid interpolation complete")
        print(f"   Grid velocity range: U[{U_grid.min():.3f}, {U_grid.max():.3f}], V[{V_grid.min():.3f}, {V_grid.max():.3f}]")
        
        return self.grid_velocity
    
    def create_streamplot_visualization(self, 
                                      save_prefix="chromatin_velocity_streamplot",
                                      streamline_density=2,
                                      arrow_density=0.01):
        """
        Create streamplot visualization with velocity field.
        """
        print(f"\n4. Creating streamplot visualizations...")
        
        if self.grid_velocity is None:
            print("   Error: No grid velocity available. Run interpolate_velocity_to_grid first.")
            return None
        
        # Extract grid data
        X_grid = self.grid_velocity['X_grid']
        Y_grid = self.grid_velocity['Y_grid']
        U_grid = self.grid_velocity['U_grid']
        V_grid = self.grid_velocity['V_grid']
        speed_grid = self.grid_velocity['speed_grid']
        
        # Get velocity magnitude for coloring
        velocity_magnitude = self.adata.obs['velocity_magnitude_smoothed'].values
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        
        # Plot 1: Streamplot with speed coloring
        ax = axes[0, 0]
        
        # Background: all peaks colored by velocity magnitude
        scatter = ax.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], 
                           c=velocity_magnitude, cmap='viridis', 
                           s=1, alpha=0.6, rasterized=True)
        
        # Streamlines
        strm = ax.streamplot(X_grid, Y_grid, U_grid, V_grid,
                           density=streamline_density, color=speed_grid,
                           cmap='plasma', linewidth=1.5, arrowsize=1.5,
                           arrowstyle='->')
        
        ax.set_title('Streamlines + Velocity Magnitude', fontsize=16, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        # Colorbar for background points
        cbar1 = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
        cbar1.set_label('Peak Velocity Magnitude', fontsize=12)
        
        # Plot 2: Clean streamlines only
        ax = axes[0, 1]
        
        # Light background
        ax.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], 
                  c='lightgray', s=0.3, alpha=0.4, rasterized=True)
        
        # Streamlines with speed coloring
        strm2 = ax.streamplot(X_grid, Y_grid, U_grid, V_grid,
                            density=streamline_density*1.5, color=speed_grid,
                            cmap='viridis', linewidth=2, arrowsize=2,
                            arrowstyle='->')
        
        ax.set_title('Clean Streamlines', fontsize=16, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        # Colorbar for streamlines
        cbar2 = plt.colorbar(strm2.lines, ax=ax, shrink=0.7)
        cbar2.set_label('Flow Speed', fontsize=12)
        
        # Plot 3: Vector field overlay
        ax = axes[1, 0]
        
        # Background
        ax.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], 
                  c=velocity_magnitude, cmap='plasma', 
                  s=1.5, alpha=0.7, rasterized=True)
        
        # Sample points for vector field
        n_sample = int(len(self.umap_coords) * arrow_density)
        sample_indices = np.random.choice(len(self.umap_coords), n_sample, replace=False)
        
        # High velocity mask for arrows
        high_vel_threshold = np.percentile(velocity_magnitude, 80)
        high_vel_mask = velocity_magnitude >= high_vel_threshold
        arrow_indices = sample_indices[high_vel_mask[sample_indices]]
        
        if len(arrow_indices) > 3000:
            arrow_indices = np.random.choice(arrow_indices, 3000, replace=False)
        
        # Plot arrows
        ax.quiver(self.umap_coords[arrow_indices, 0], self.umap_coords[arrow_indices, 1],
                 self.velocity_smoothed[arrow_indices, 0], self.velocity_smoothed[arrow_indices, 1],
                 angles='xy', scale_units='xy', scale=0.5,
                 width=0.003, color='red', alpha=0.8, zorder=3)
        
        ax.set_title(f'Vector Field ({len(arrow_indices):,} arrows)', fontsize=16, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        # Plot 4: Flow convergence/divergence
        ax = axes[1, 1]
        
        # Compute divergence (simplified)
        try:
            divergence = np.gradient(U_grid, axis=1) + np.gradient(V_grid, axis=0)
            
            # Background
            ax.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], 
                      c='lightgray', s=0.5, alpha=0.3, rasterized=True)
            
            # Divergence heatmap
            im = ax.imshow(divergence, extent=[X_grid.min(), X_grid.max(), Y_grid.min(), Y_grid.max()],
                         origin='lower', cmap='RdBu_r', alpha=0.8)
            
            plt.colorbar(im, ax=ax, shrink=0.7, label='Flow Divergence')
            ax.set_title('Flow Convergence/Divergence', fontsize=16, fontweight='bold')
            
        except Exception as e:
            print(f"   Divergence computation failed: {e}")
            # Fallback: velocity magnitude histogram
            ax.hist(velocity_magnitude[velocity_magnitude > 0], bins=100, alpha=0.7)
            ax.set_xlabel('Velocity Magnitude')
            ax.set_ylabel('Count')
            ax.set_title('Velocity Distribution', fontsize=16, fontweight='bold')
            ax.set_yscale('log')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        plt.tight_layout()
        
        # Save comprehensive plot
        save_path_comp = f"{save_prefix}_comprehensive.png"
        plt.savefig(save_path_comp, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Comprehensive streamplot saved: {save_path_comp}")
        plt.show()
        
        # Create focused publication plot
        print("   Creating focused streamplot...")
        fig_focused, ax = plt.subplots(figsize=(14, 12), dpi=150)
        
        # Background: all peaks
        scatter = ax.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], 
                           c=velocity_magnitude, cmap='viridis', 
                           s=2, alpha=0.7, rasterized=True)
        
        # Main streamlines
        strm_main = ax.streamplot(X_grid, Y_grid, U_grid, V_grid,
                                density=streamline_density, color='white',
                                linewidth=2.5, arrowsize=2.5, arrowstyle='->',
                                integration_direction='forward')
        
        # Styling
        ax.set_xlabel('UMAP 1', fontsize=18)
        ax.set_ylabel('UMAP 2', fontsize=18)
        ax.set_title(f'Chromatin Velocity Streamlines - ALL {len(self.umap_coords):,} Peaks', 
                    fontsize=20, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Velocity Magnitude', fontsize=16)
        
        # Statistics
        stats_text = f"""Flow Analysis:
Total peaks: {len(self.umap_coords):,}
Mean velocity: {velocity_magnitude.mean():.3f}
Flow coherence: Neighborhood smoothed
Streamline density: {streamline_density}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.9))
        
        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=14)
        
        plt.tight_layout()
        
        # Save focused plot
        save_path_focused = f"{save_prefix}_focused.png"
        plt.savefig(save_path_focused, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Focused streamplot saved: {save_path_focused}")
        plt.show()
        
        return fig, fig_focused
    
    def compare_smoothing_scales(self, k_values=[10, 30, 50, 100]):
        """
        Compare different smoothing scales to show multi-scale structure.
        """
        print(f"\n5. Comparing smoothing scales: {k_values}...")
        
        fig, axes = plt.subplots(2, len(k_values)//2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, k in enumerate(k_values):
            print(f"   Computing smoothing with k={k}...")
            
            # Compute smoothed velocity for this k
            tree = cKDTree(self.umap_coords)
            distances, indices = tree.query(self.umap_coords, k=k+1)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
            
            smoothed_velocity_k = np.zeros_like(self.velocity_raw)
            for j in range(len(self.umap_coords)):
                neighbor_indices = indices[j]
                neighbor_distances = distances[j]
                weights = 1.0 / (neighbor_distances + 1e-6)
                weights = weights / weights.sum()
                neighbor_velocities = self.velocity_raw[neighbor_indices]
                smoothed_velocity_k[j] = np.average(neighbor_velocities, axis=0, weights=weights)
            
            # Create quick visualization
            ax = axes[i]
            
            velocity_mag = np.sqrt((smoothed_velocity_k**2).sum(axis=1))
            
            # Background
            ax.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], 
                      c=velocity_mag, cmap='viridis', s=1, alpha=0.6, rasterized=True)
            
            # Sample arrows
            high_vel_mask = velocity_mag >= np.percentile(velocity_mag, 85)
            high_vel_indices = np.where(high_vel_mask)[0]
            
            if len(high_vel_indices) > 1000:
                arrow_indices = np.random.choice(high_vel_indices, 1000, replace=False)
            else:
                arrow_indices = high_vel_indices
            
            ax.quiver(self.umap_coords[arrow_indices, 0], self.umap_coords[arrow_indices, 1],
                     smoothed_velocity_k[arrow_indices, 0], smoothed_velocity_k[arrow_indices, 1],
                     angles='xy', scale_units='xy', scale=0.5,
                     width=0.003, color='red', alpha=0.8)
            
            ax.set_title(f'k = {k} neighbors', fontsize=14, fontweight='bold')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
        
        plt.tight_layout()
        
        save_path = "chromatin_velocity_multiscale_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Multi-scale comparison saved: {save_path}")
        plt.show()
        
        return fig

def main():
    """Main execution function."""
    
    print("Starting improved chromatin velocity analysis...")
    
    # Initialize visualizer
    velocity_data_path = "chromatin_velocity_all_peaks_fast.h5ad"
    visualizer = ImprovedVelocityVisualizer(velocity_data_path)
    
    # Load data
    if not visualizer.load_velocity_data():
        print("Failed to load velocity data. Exiting.")
        return None
    
    # Compute neighborhood-smoothed velocity with faster settings
    smoothed_velocity = visualizer.compute_neighborhood_smoothed_velocity(k_neighbors=20)
    
    # Interpolate to grid for streamplot with faster resolution
    grid_velocity = visualizer.interpolate_velocity_to_grid(grid_resolution=30)
    
    # Create streamplot visualizations
    fig_comp, fig_focused = visualizer.create_streamplot_visualization(
        streamline_density=1.5, arrow_density=0.005
    )
    
    # Skip multi-scale comparison for now to save time
    # fig_multiscale = visualizer.compare_smoothing_scales(k_values=[15, 30, 50, 100])
    
    # Save updated data
    output_path = "chromatin_velocity_all_peaks_streamplot.h5ad"
    visualizer.adata.write(output_path)
    print(f"\n✓ Updated data saved: {output_path}")
    
    print("\n" + "="*80)
    print("✓ IMPROVED CHROMATIN VELOCITY ANALYSIS COMPLETE!")
    print("="*80)
    print("Generated files:")
    print("- chromatin_velocity_streamplot_comprehensive.png")
    print("- chromatin_velocity_streamplot_focused.png")
    print("- chromatin_velocity_multiscale_comparison.png")
    print("- chromatin_velocity_all_peaks_streamplot.h5ad")
    
    return visualizer

if __name__ == "__main__":
    result_visualizer = main()