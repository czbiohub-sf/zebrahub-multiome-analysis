#!/usr/bin/env python3
"""
Chromatin Velocity Parameter Sweep

Performs parameter sweep over k-nearest neighbors for velocity smoothing
and applies 0.1x velocity scaling for better directionality visualization.

Generates:
1. Multi-panel comparison of different k values
2. Individual streamplots for each k
3. Final optimized visualization with proper scaling

Author: Zebrahub-Multiome Analysis Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc
import scipy.sparse as sp
from scipy.interpolate import griddata
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CHROMATIN VELOCITY PARAMETER SWEEP")
print("="*80)


class VelocityParameterSweep:
    """
    Parameter sweep for chromatin velocity visualization.
    """

    def __init__(self, data_path):
        """Initialize with data path."""
        self.data_path = data_path
        self.adata = None
        self.umap_coords = None
        self.velocity_raw = None
        self.connectivity_matrix = None
        self.velocity_scaled = None  # 0.1x scaling
        self.smoothed_velocities = {}  # Store results for each k

    def load_data(self):
        """Load peak data with connectivity and velocity."""
        print(f"\n1. Loading data from {self.data_path}...")

        try:
            self.adata = sc.read_h5ad(self.data_path)
            print(f"   ✓ Loaded: {self.adata.shape}")

            # Get UMAP
            self.umap_coords = self.adata.obsm['X_umap']
            print(f"   ✓ UMAP: {self.umap_coords.shape}")

            # Get raw velocity (before any smoothing)
            self.velocity_raw = self.adata.obsm['velocity_umap']
            print(f"   ✓ Raw velocity: {self.velocity_raw.shape}")
            print(f"     Magnitude: mean={np.sqrt((self.velocity_raw**2).sum(axis=1)).mean():.3f}")

            # Apply 0.1x scaling
            self.velocity_scaled = self.velocity_raw * 0.1
            print(f"   ✓ Applied 0.1x scaling")
            print(f"     Scaled magnitude: mean={np.sqrt((self.velocity_scaled**2).sum(axis=1)).mean():.3f}")

            # Get connectivity
            if 'connectivities' in self.adata.obsp:
                self.connectivity_matrix = self.adata.obsp['connectivities']
                print(f"   ✓ Connectivity: {self.connectivity_matrix.shape}")
                print(f"     Non-zero: {self.connectivity_matrix.nnz:,}")

                # Check neighbor info
                if 'neighbors' in self.adata.uns:
                    n_neighbors = self.adata.uns['neighbors']['params']['n_neighbors']
                    print(f"     Original n_neighbors: {n_neighbors}")
            else:
                print("   Error: No connectivity matrix found")
                return False

            return True

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def smooth_velocity_with_k_neighbors(self, k):
        """
        Smooth scaled velocity using top-k neighbors from connectivity matrix.

        FAST IMPLEMENTATION: Truncates connectivity matrix to top-k, then uses
        sparse matrix multiplication.

        Parameters:
        -----------
        k : int
            Number of nearest neighbors to use for smoothing

        Returns:
        --------
        smoothed_velocity : ndarray
            Smoothed velocity vectors (already scaled by 0.1)
        """
        print(f"\n   Computing smoothing with k={k} neighbors...")

        # Convert to CSR format
        if not sp.isspmatrix_csr(self.connectivity_matrix):
            connectivity_csr = self.connectivity_matrix.tocsr()
        else:
            connectivity_csr = self.connectivity_matrix

        # Create truncated connectivity matrix with only top-k per row
        print(f"     Truncating to top-k neighbors...")
        n_peaks = connectivity_csr.shape[0]

        # Build new sparse matrix with top-k per row
        rows = []
        cols = []
        data = []

        for i in tqdm(range(n_peaks), desc="Truncating", leave=False, disable=n_peaks>100000):
            row = connectivity_csr.getrow(i)
            neighbor_indices = row.indices
            neighbor_weights = row.data

            if len(neighbor_indices) == 0:
                # Add self-connection
                rows.append(i)
                cols.append(i)
                data.append(1.0)
            elif len(neighbor_indices) <= k:
                # Keep all
                rows.extend([i] * len(neighbor_indices))
                cols.extend(neighbor_indices.tolist())
                data.extend(neighbor_weights.tolist())
            else:
                # Keep top-k
                top_k_idx = np.argsort(neighbor_weights)[-k:]
                rows.extend([i] * k)
                cols.extend(neighbor_indices[top_k_idx].tolist())
                data.extend(neighbor_weights[top_k_idx].tolist())

        # Create truncated connectivity matrix
        print(f"     Building sparse matrix...")
        connectivity_k = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_peaks, n_peaks)
        )

        # Normalize rows to sum to 1
        row_sums = np.array(connectivity_k.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1

        # Apply smoothing via matrix multiplication (FAST!)
        print(f"     Applying weighted averaging...")
        smoothed_velocity = np.zeros_like(self.velocity_scaled)

        for dim in range(self.velocity_scaled.shape[1]):
            velocity_dim = self.velocity_scaled[:, dim]
            weighted_sum = connectivity_k.dot(velocity_dim)
            smoothed_velocity[:, dim] = weighted_sum / row_sums

        # Compute statistics
        magnitude = np.sqrt((smoothed_velocity**2).sum(axis=1))
        print(f"   ✓ k={k}: mean mag={magnitude.mean():.4f}, median={np.median(magnitude):.4f}, max={magnitude.max():.4f}")

        return smoothed_velocity

    def run_parameter_sweep(self, k_values=[5, 10, 15, 20, 30, 50]):
        """
        Run parameter sweep over different k values.
        """
        print(f"\n2. Running parameter sweep over k = {k_values}...")

        for k in k_values:
            smoothed_vel = self.smooth_velocity_with_k_neighbors(k)
            self.smoothed_velocities[k] = smoothed_vel

        print(f"\n   ✓ Completed smoothing for {len(k_values)} k values")

        return self.smoothed_velocities

    def interpolate_to_grid(self, velocity, grid_resolution=50):
        """Interpolate velocity to grid for streamplot."""

        # Create regular grid
        x_min, x_max = self.umap_coords[:, 0].min(), self.umap_coords[:, 0].max()
        y_min, y_max = self.umap_coords[:, 1].min(), self.umap_coords[:, 1].max()

        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05

        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, grid_resolution)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # Interpolate
        U_grid = griddata(
            self.umap_coords, velocity[:, 0],
            (X_grid, Y_grid), method='linear', fill_value=0
        )
        V_grid = griddata(
            self.umap_coords, velocity[:, 1],
            (X_grid, Y_grid), method='linear', fill_value=0
        )

        # Fill NaN
        mask_nan = np.isnan(U_grid) | np.isnan(V_grid)
        if np.any(mask_nan):
            U_grid_nn = griddata(
                self.umap_coords, velocity[:, 0],
                (X_grid, Y_grid), method='nearest'
            )
            V_grid_nn = griddata(
                self.umap_coords, velocity[:, 1],
                (X_grid, Y_grid), method='nearest'
            )
            U_grid[mask_nan] = U_grid_nn[mask_nan]
            V_grid[mask_nan] = V_grid_nn[mask_nan]

        return {
            'X_grid': X_grid,
            'Y_grid': Y_grid,
            'U_grid': U_grid,
            'V_grid': V_grid,
            'speed_grid': np.sqrt(U_grid**2 + V_grid**2)
        }

    def create_comparison_panel(self, k_values=[5, 10, 15, 20, 30, 50],
                                 streamline_density=2.0,
                                 grid_resolution=50):
        """
        Create multi-panel comparison of different k values.
        """
        print(f"\n3. Creating {len(k_values)}-panel comparison (density={streamline_density})...")

        # Create figure
        n_rows = 2
        n_cols = 3
        fig = plt.figure(figsize=(30, 18), dpi=150)
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

        for idx, k in enumerate(k_values):
            print(f"   Panel {idx+1}/{len(k_values)}: k={k}...")

            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            # Get smoothed velocity for this k
            velocity = self.smoothed_velocities[k]
            velocity_magnitude = np.sqrt((velocity**2).sum(axis=1))

            # Interpolate to grid
            grid_data = self.interpolate_to_grid(velocity, grid_resolution=grid_resolution)

            # Background scatter
            scatter = ax.scatter(
                self.umap_coords[:, 0], self.umap_coords[:, 1],
                c=velocity_magnitude, cmap='viridis',
                s=1, alpha=0.5, rasterized=True, vmin=0,
                vmax=np.percentile(velocity_magnitude, 99)
            )

            # Streamlines
            ax.streamplot(
                grid_data['X_grid'], grid_data['Y_grid'],
                grid_data['U_grid'], grid_data['V_grid'],
                density=streamline_density, color='white',
                linewidth=2, arrowsize=2, arrowstyle='->'
            )

            # Styling
            ax.set_title(f'k = {k} neighbors', fontsize=20, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=16)
            ax.set_ylabel('UMAP 2', fontsize=16)
            ax.tick_params(labelsize=14)

            # Add statistics
            stats_text = f"""Mean: {velocity_magnitude.mean():.4f}
Max: {velocity_magnitude.max():.4f}
Scale: 0.1x"""
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label('Velocity Magnitude', fontsize=14)
            cbar.ax.tick_params(labelsize=12)

        # Main title
        fig.suptitle(
            f'Chromatin Velocity: k-Neighbor Parameter Sweep (0.1x scaling)',
            fontsize=28, fontweight='bold', y=0.995
        )

        # Save
        save_path = "chromatin_velocity_k_neighbor_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n   ✓ Saved: {save_path}")

        return fig

    def create_individual_plots(self, k_values=[5, 10, 15, 20, 30, 50],
                                 streamline_density=2.0,
                                 grid_resolution=50):
        """
        Create individual high-res plots for each k value.
        """
        print(f"\n4. Creating individual plots for each k...")

        for k in k_values:
            print(f"   k={k}...")

            velocity = self.smoothed_velocities[k]
            velocity_magnitude = np.sqrt((velocity**2).sum(axis=1))

            # Interpolate to grid
            grid_data = self.interpolate_to_grid(velocity, grid_resolution=grid_resolution)

            # Create figure
            fig, ax = plt.subplots(figsize=(16, 14), dpi=150)

            # Background
            scatter = ax.scatter(
                self.umap_coords[:, 0], self.umap_coords[:, 1],
                c=velocity_magnitude, cmap='viridis',
                s=2, alpha=0.6, rasterized=True, vmin=0,
                vmax=np.percentile(velocity_magnitude, 99)
            )

            # Streamlines
            ax.streamplot(
                grid_data['X_grid'], grid_data['Y_grid'],
                grid_data['U_grid'], grid_data['V_grid'],
                density=streamline_density, color='white',
                linewidth=2.5, arrowsize=2.5, arrowstyle='->'
            )

            # Styling
            ax.set_xlabel('UMAP 1', fontsize=20)
            ax.set_ylabel('UMAP 2', fontsize=20)
            ax.set_title(
                f'Chromatin Velocity - k={k} neighbors (0.1x scale)',
                fontsize=24, fontweight='bold', pad=20
            )
            ax.tick_params(labelsize=16)

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label('Velocity Magnitude', fontsize=18)
            cbar.ax.tick_params(labelsize=14)

            # Statistics
            stats_text = f"""Connectivity-smoothed velocity
k = {k} neighbors
{len(self.umap_coords):,} peaks
Mean velocity: {velocity_magnitude.mean():.4f}
Max velocity: {velocity_magnitude.max():.4f}
Scaling: 0.1x"""

            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes, fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )

            plt.tight_layout()

            # Save
            save_path = f"chromatin_velocity_k{k}_scaled0.1x.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"     ✓ Saved: {save_path}")
            plt.close()

    def create_arrows_with_scaling(self, k=15, n_arrows=3000):
        """Create arrow plot with proper 0.1x scaling."""
        print(f"\n5. Creating arrow plot (k={k}, 0.1x scaling)...")

        velocity = self.smoothed_velocities[k]
        velocity_magnitude = np.sqrt((velocity**2).sum(axis=1))

        # Grid-based sampling
        x_min, x_max = self.umap_coords[:, 0].min(), self.umap_coords[:, 0].max()
        y_min, y_max = self.umap_coords[:, 1].min(), self.umap_coords[:, 1].max()

        n_grid = int(np.sqrt(n_arrows))
        x_bins = np.linspace(x_min, x_max, n_grid + 1)
        y_bins = np.linspace(y_min, y_max, n_grid + 1)

        arrow_indices = []
        for i in range(n_grid):
            for j in range(n_grid):
                mask = (
                    (self.umap_coords[:, 0] >= x_bins[i]) &
                    (self.umap_coords[:, 0] < x_bins[i+1]) &
                    (self.umap_coords[:, 1] >= y_bins[j]) &
                    (self.umap_coords[:, 1] < y_bins[j+1])
                )
                indices_in_cell = np.where(mask)[0]
                if len(indices_in_cell) > 0:
                    best_idx = indices_in_cell[np.argmax(velocity_magnitude[indices_in_cell])]
                    arrow_indices.append(best_idx)

        arrow_indices = np.array(arrow_indices)
        print(f"   Sampled {len(arrow_indices)} arrows")

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 14), dpi=150)

        # Background
        scatter = ax.scatter(
            self.umap_coords[:, 0], self.umap_coords[:, 1],
            c=velocity_magnitude, cmap='viridis',
            s=2, alpha=0.6, rasterized=True, vmin=0,
            vmax=np.percentile(velocity_magnitude, 99)
        )

        # Arrows with proper scaling
        ax.quiver(
            self.umap_coords[arrow_indices, 0],
            self.umap_coords[arrow_indices, 1],
            velocity[arrow_indices, 0],
            velocity[arrow_indices, 1],
            velocity_magnitude[arrow_indices],
            angles='xy', scale_units='xy', scale=0.04,  # 0.04 = 0.4 * 0.1
            width=0.003, cmap='plasma', alpha=0.9,
            clim=(0, np.percentile(velocity_magnitude, 99))
        )

        # Styling
        ax.set_xlabel('UMAP 1', fontsize=20)
        ax.set_ylabel('UMAP 2', fontsize=20)
        ax.set_title(
            f'Chromatin Velocity Arrows - k={k} (0.1x scale)',
            fontsize=24, fontweight='bold', pad=20
        )
        ax.tick_params(labelsize=16)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Velocity Magnitude', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        # Statistics
        stats_text = f"""k = {k} neighbors
{len(arrow_indices):,} arrows
Scaling: 0.1x
Mean velocity: {velocity_magnitude.mean():.4f}"""

        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

        plt.tight_layout()

        # Save
        save_path = f"chromatin_velocity_arrows_k{k}_scaled0.1x.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Saved: {save_path}")

        return fig


def main():
    """Main execution."""

    print("\nInitializing parameter sweep...")

    # Load data - use the file with pre-computed velocity
    data_path = "peak_umap_chromatin_velocity_connectivity_smoothed.h5ad"

    sweep = VelocityParameterSweep(data_path)

    if not sweep.load_data():
        print("Failed to load data. Exiting.")
        return None

    # Run parameter sweep
    k_values = [5, 10, 15, 20, 30, 50]
    sweep.run_parameter_sweep(k_values=k_values)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Create comparison panel
    sweep.create_comparison_panel(k_values=k_values, streamline_density=2.0)

    # Create individual plots
    sweep.create_individual_plots(k_values=k_values, streamline_density=2.0)

    # Create arrow plot with optimal k
    sweep.create_arrows_with_scaling(k=15, n_arrows=3000)

    print("\n" + "="*80)
    print("✓ PARAMETER SWEEP COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("- chromatin_velocity_k_neighbor_comparison.png (6-panel)")
    print("- chromatin_velocity_k{5,10,15,20,30,50}_scaled0.1x.png (6 individual plots)")
    print("- chromatin_velocity_arrows_k15_scaled0.1x.png")
    print("\n✓ All visualizations use 0.1x velocity scaling")
    print("✓ Parameter sweep: k = [5, 10, 15, 20, 30, 50] neighbors")

    return sweep


if __name__ == "__main__":
    result = main()
