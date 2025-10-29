#!/usr/bin/env python3
"""
Improved Peak UMAP Chromatin Velocity Visualizer

Creates coherent velocity fields using pre-computed neighborhood connectivity from
adata.obsp['connectivities'] (computed in high-dimensional PCA space) and generates
streamplot visualizations showing developmental flow patterns.

This approach uses the TRUE peak similarity structure from the original high-dimensional
space rather than recomputing neighborhoods in the 2D UMAP projection.

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
    Improved visualizer using pre-computed connectivity matrix for velocity smoothing.

    Uses adata.obsp['connectivities'] from the original high-dimensional (PCA) space
    to capture true peak similarity structure, rather than recomputing neighborhoods
    in the 2D UMAP projection space.
    """
    
    def __init__(self, velocity_data_path, original_data_path=None):
        """Initialize with velocity data and optional original data with connectivity."""
        self.velocity_data_path = velocity_data_path
        self.original_data_path = original_data_path
        self.adata = None
        self.adata_original = None
        self.umap_coords = None
        self.velocity_raw = None
        self.velocity_smoothed = None
        self.grid_velocity = None
        self.connectivity_matrix = None
        
    def load_velocity_data(self):
        """Load the peak UMAP data and compute velocity if needed."""
        print(f"\n   Loading peak data from {self.velocity_data_path}...")

        try:
            self.adata = sc.read_h5ad(self.velocity_data_path)
            print(f"   ✓ Loaded peak data: {self.adata.shape}")

            # Get UMAP coordinates
            if 'X_umap' not in self.adata.obsm:
                print("   Error: No UMAP coordinates found in data")
                return False

            self.umap_coords = self.adata.obsm['X_umap']
            print(f"   ✓ UMAP coordinates: {self.umap_coords.shape}")

            # Check for connectivity matrix
            if 'connectivities' in self.adata.obsp:
                self.connectivity_matrix = self.adata.obsp['connectivities']
                print(f"   ✓ Found connectivity matrix: {self.connectivity_matrix.shape} with {self.connectivity_matrix.nnz:,} non-zero entries")
            else:
                print("   Warning: No connectivity matrix found in data")

            # Check if velocity already exists, otherwise compute it
            if 'velocity_umap' in self.adata.obsm:
                self.velocity_raw = self.adata.obsm['velocity_umap']
                print(f"   ✓ Found pre-computed velocity: {self.velocity_raw.shape}")
            elif 'velocity_temporal' in self.adata.layers:
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
                # Need to compute velocity from scratch
                print("   No velocity found. Computing chromatin velocity from temporal data...")
                if not self.compute_chromatin_velocity():
                    return False

            print(f"   ✓ Raw velocity vectors: {self.velocity_raw.shape}")
            print(f"   Velocity range: X[{self.velocity_raw[:, 0].min():.3f}, {self.velocity_raw[:, 0].max():.3f}], Y[{self.velocity_raw[:, 1].min():.3f}, {self.velocity_raw[:, 1].max():.3f}]")

        except Exception as e:
            print(f"   Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    def compute_chromatin_velocity(self):
        """Compute chromatin velocity from accessibility data across timepoints."""
        print("\n   Computing chromatin velocity...")

        # Check if we have timepoint information
        timepoint_col = None
        for col in ['timepoint', 'timepoint_numeric', 'somite_stage']:
            if col in self.adata.obs:
                timepoint_col = col
                break

        if timepoint_col is None:
            print("   Error: No timepoint information found")
            print(f"   Available columns: {self.adata.obs.columns.tolist()}")
            return False

        # Get timepoint info
        timepoints_raw = self.adata.obs[timepoint_col].values
        print(f"   Using timepoint column: '{timepoint_col}'")
        print(f"   Raw timepoints: {np.unique(timepoints_raw)}")

        # Convert to numeric if needed (e.g., "0somites" -> 0)
        if timepoints_raw.dtype == object or isinstance(timepoints_raw[0], str):
            # Extract numeric values from strings like "0somites", "5somites", etc.
            import re
            timepoints = np.array([int(re.search(r'\d+', str(t)).group()) if re.search(r'\d+', str(t)) else 0 for t in timepoints_raw])
            print(f"   Converted to numeric: {np.unique(timepoints)}")
        else:
            timepoints = timepoints_raw

        print(f"   Timepoints: {np.unique(timepoints)}")

        # Use accessibility data (X matrix)
        accessibility = self.adata.X
        if sp.issparse(accessibility):
            accessibility = accessibility.toarray()

        # Compute temporal derivative for each pseudobulk
        # Simple approach: velocity = future_state - current_state
        print("   Computing temporal derivatives...")

        velocity_matrix = np.zeros_like(accessibility)

        unique_timepoints = np.sort(np.unique(timepoints))

        for i, tp in enumerate(unique_timepoints[:-1]):
            # Find cells at current and next timepoint
            curr_mask = timepoints == tp
            next_mask = timepoints == unique_timepoints[i+1]

            if curr_mask.sum() > 0 and next_mask.sum() > 0:
                # Average accessibility at each timepoint
                curr_acc = accessibility[curr_mask].mean(axis=0)
                next_acc = accessibility[next_mask].mean(axis=0)

                # Velocity is the difference
                velocity_matrix[curr_mask] = next_acc - curr_acc

        # For last timepoint, use zero velocity
        last_mask = timepoints == unique_timepoints[-1]
        velocity_matrix[last_mask] = 0

        # Store as temporal velocity
        self.adata.layers['velocity_temporal'] = velocity_matrix

        # Compute velocity magnitude
        velocity_magnitude = np.sqrt((velocity_matrix**2).sum(axis=1))
        self.adata.obs['velocity_magnitude'] = velocity_magnitude

        print(f"   ✓ Computed velocity: mean={velocity_magnitude.mean():.3f}, max={velocity_magnitude.max():.3f}")

        # Project to 2D UMAP space using PCA
        print("   Projecting to 2D UMAP space...")
        from sklearn.decomposition import TruncatedSVD

        if velocity_matrix.shape[1] > 2:
            svd = TruncatedSVD(n_components=2, random_state=42)
            velocity_2d = svd.fit_transform(velocity_matrix)
        else:
            velocity_2d = velocity_matrix[:, :2]

        # Scale by magnitude
        velocity_norm = np.sqrt((velocity_2d**2).sum(axis=1))
        velocity_norm[velocity_norm == 0] = 1

        velocity_2d_normed = velocity_2d / velocity_norm[:, np.newaxis]
        self.velocity_raw = velocity_2d_normed * velocity_magnitude[:, np.newaxis] * 0.1

        self.adata.obsm['velocity_umap'] = self.velocity_raw

        print(f"   ✓ Velocity projection complete")

        return True
    
    def load_original_connectivity(self):
        """
        Load the original AnnData object with pre-computed connectivity matrix.
        """
        if self.original_data_path is None:
            print("   Warning: No original data path provided. Will attempt to load from velocity data.")
            # Check if connectivity is already in velocity data
            if 'connectivities' in self.adata.obsp:
                print("   ✓ Found connectivity matrix in velocity data")
                self.connectivity_matrix = self.adata.obsp['connectivities']
                return True
            else:
                print("   Error: No connectivity matrix found. Cannot proceed with smoothing.")
                return False

        print(f"\n   Loading original data with connectivity from {self.original_data_path}...")
        try:
            # Load only the connectivity matrix efficiently
            self.adata_original = sc.read_h5ad(self.original_data_path)

            if 'connectivities' not in self.adata_original.obsp:
                print("   Error: No 'connectivities' found in original data. Cannot proceed.")
                return False

            # Extract connectivity matrix
            self.connectivity_matrix = self.adata_original.obsp['connectivities']
            print(f"   ✓ Loaded connectivity matrix: {self.connectivity_matrix.shape} with {self.connectivity_matrix.nnz:,} non-zero entries")

            # Verify peak overlap
            original_peaks = self.adata_original.obs_names
            velocity_peaks = self.adata.obs_names
            overlap = set(original_peaks) & set(velocity_peaks)
            print(f"   ✓ Peak overlap: {len(overlap):,} / {len(velocity_peaks):,} ({100*len(overlap)/len(velocity_peaks):.1f}%)")

            # If not all peaks overlap, we need to subset the connectivity matrix
            if len(overlap) < len(velocity_peaks):
                print("   Subsetting connectivity matrix to match velocity data...")
                # Find indices of overlapping peaks in original data
                original_peak_to_idx = {peak: i for i, peak in enumerate(original_peaks)}
                velocity_indices_in_original = [original_peak_to_idx[peak] for peak in velocity_peaks if peak in original_peak_to_idx]

                # Subset connectivity matrix
                self.connectivity_matrix = self.connectivity_matrix[velocity_indices_in_original, :][:, velocity_indices_in_original]
                print(f"   ✓ Subsetted connectivity: {self.connectivity_matrix.shape}")

            return True

        except Exception as e:
            print(f"   Error loading original data: {e}")
            return False

    def compute_neighborhood_smoothed_velocity(self, use_precomputed=True):
        """
        Compute neighborhood-smoothed velocity using pre-computed connectivity matrix.

        This uses the connectivity matrix from adata.obsp['connectivities'] which captures
        the true peak similarity structure from high-dimensional (PCA) space, rather than
        recomputing neighborhoods in UMAP space.
        """
        print(f"\n2. Computing neighborhood-smoothed velocity using pre-computed connectivity...")

        # Load connectivity matrix if not already loaded
        if self.connectivity_matrix is None:
            if not self.load_original_connectivity():
                print("   Error: Failed to load connectivity matrix. Cannot proceed.")
                return None

        # Verify connectivity matrix matches velocity data
        if self.connectivity_matrix.shape[0] != len(self.velocity_raw):
            print(f"   Error: Connectivity matrix size {self.connectivity_matrix.shape} doesn't match velocity data {len(self.velocity_raw)}")
            return None

        print(f"   Using connectivity matrix: {self.connectivity_matrix.shape}")
        print(f"   Non-zero entries: {self.connectivity_matrix.nnz:,}")
        print(f"   Sparsity: {100 * (1 - self.connectivity_matrix.nnz / (self.connectivity_matrix.shape[0]**2)):.3f}%")

        # Convert to CSR format for efficient row operations
        if not sp.isspmatrix_csr(self.connectivity_matrix):
            print("   Converting connectivity to CSR format...")
            connectivity_csr = self.connectivity_matrix.tocsr()
        else:
            connectivity_csr = self.connectivity_matrix

        print(f"   Computing smoothed velocity vectors...")
        smoothed_velocity = np.zeros_like(self.velocity_raw)

        # Normalize rows to sum to 1 (weighted averaging)
        row_sums = np.array(connectivity_csr.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        # Apply sparse matrix multiplication for efficient weighted averaging
        # Each row of connectivity represents the weights for that peak's neighbors
        print("   Applying sparse matrix weighted averaging...")

        for dim in range(self.velocity_raw.shape[1]):
            # For each dimension of velocity, compute weighted average
            velocity_dim = self.velocity_raw[:, dim]

            # Weighted sum using sparse matrix multiplication
            weighted_sum = connectivity_csr.dot(velocity_dim)

            # Normalize by row sums to get weighted average
            smoothed_velocity[:, dim] = weighted_sum / row_sums

        self.velocity_smoothed = smoothed_velocity

        print(f"✓ Smoothed velocity computed using pre-computed neighborhoods")
        print(f"   Smoothed range: X[{smoothed_velocity[:, 0].min():.3f}, {smoothed_velocity[:, 0].max():.3f}], Y[{smoothed_velocity[:, 1].min():.3f}, {smoothed_velocity[:, 1].max():.3f}]")

        # Compute smoothed magnitude
        smoothed_magnitude = np.sqrt((smoothed_velocity**2).sum(axis=1))
        self.adata.obs['velocity_magnitude_smoothed'] = smoothed_magnitude
        self.adata.obsm['velocity_umap_smoothed'] = smoothed_velocity

        print(f"   Mean smoothed magnitude: {smoothed_magnitude.mean():.3f}")
        print(f"   Median smoothed magnitude: {np.median(smoothed_magnitude):.3f}")

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
    print("Using pre-computed connectivity from original high-dimensional space...")

    # Use the master peak UMAP object which already contains connectivity
    peak_umap_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad"

    print(f"\n1. Loading peak UMAP object from {peak_umap_path}...")

    # We'll compute velocity directly on this object, so we don't need separate velocity data
    visualizer = ImprovedVelocityVisualizer(peak_umap_path, original_data_path=None)

    # Load data and compute velocity if needed
    if not visualizer.load_velocity_data():
        print("Failed to load peak data. Exiting.")
        return None

    print("\n2. Computing neighborhood-smoothed velocity using pre-computed connectivity...")

    # Compute neighborhood-smoothed velocity using pre-computed connectivity
    smoothed_velocity = visualizer.compute_neighborhood_smoothed_velocity(use_precomputed=True)

    if smoothed_velocity is None:
        print("Failed to compute smoothed velocity. Exiting.")
        return None

    print("\n3. Interpolating velocity to grid for streamplot...")

    # Interpolate to grid for streamplot
    grid_velocity = visualizer.interpolate_velocity_to_grid(grid_resolution=40)

    print("\n4. Creating streamplot visualizations...")

    # Create streamplot visualizations
    fig_comp, fig_focused = visualizer.create_streamplot_visualization(
        streamline_density=2.0, arrow_density=0.01
    )

    # Save updated data
    output_path = "peak_umap_chromatin_velocity_connectivity_smoothed.h5ad"
    print(f"\n5. Saving results to {output_path}...")
    visualizer.adata.write(output_path)
    print(f"✓ Updated data saved: {output_path}")

    print("\n" + "="*80)
    print("✓ CHROMATIN VELOCITY ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("- chromatin_velocity_streamplot_comprehensive.png")
    print("- chromatin_velocity_streamplot_focused.png")
    print("- peak_umap_chromatin_velocity_connectivity_smoothed.h5ad")
    print("\n✓ Key: Velocity smoothing uses pre-computed connectivity from")
    print("  adata.obsp['connectivities'] (high-dimensional PCA space)")
    print("  rather than recomputing neighborhoods in UMAP space")

    return visualizer

if __name__ == "__main__":
    result_visualizer = main()