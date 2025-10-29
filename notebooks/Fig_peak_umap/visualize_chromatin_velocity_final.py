#!/usr/bin/env python3
"""
Final Chromatin Velocity Visualization

Creates publication-ready visualizations of chromatin velocity on peak UMAP using:
1. Streamplot - continuous flow visualization
2. Arrow/quiver plots - discrete velocity vectors
3. Combined visualizations

Uses pre-computed velocity with connectivity-based smoothing from
adata.obsp['connectivities'] (high-dimensional PCA space neighborhoods).

Author: Zebrahub-Multiome Analysis Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CHROMATIN VELOCITY FINAL VISUALIZATION")
print("="*80)


class ChromatinVelocityVisualizer:
    """
    Visualizer for chromatin velocity using pre-computed connectivity-smoothed data.
    """

    def __init__(self, velocity_data_path):
        """Initialize with pre-computed velocity data."""
        self.velocity_data_path = velocity_data_path
        self.adata = None
        self.umap_coords = None
        self.velocity_smoothed = None
        self.velocity_magnitude = None
        self.grid_velocity = None

    def load_data(self):
        """Load pre-computed velocity data."""
        print(f"\n1. Loading pre-computed velocity data...")
        print(f"   Path: {self.velocity_data_path}")

        try:
            self.adata = sc.read_h5ad(self.velocity_data_path)
            print(f"   ✓ Loaded: {self.adata.shape}")

            # Get UMAP coordinates
            self.umap_coords = self.adata.obsm['X_umap']
            print(f"   ✓ UMAP: {self.umap_coords.shape}")

            # Get smoothed velocity
            if 'velocity_umap_smoothed' in self.adata.obsm:
                self.velocity_smoothed = self.adata.obsm['velocity_umap_smoothed']
            else:
                print("   Warning: Using raw velocity (not smoothed)")
                self.velocity_smoothed = self.adata.obsm['velocity_umap']

            print(f"   ✓ Velocity: {self.velocity_smoothed.shape}")

            # Get velocity magnitude
            if 'velocity_magnitude_smoothed' in self.adata.obs:
                self.velocity_magnitude = self.adata.obs['velocity_magnitude_smoothed'].values
            else:
                self.velocity_magnitude = self.adata.obs['velocity_magnitude'].values

            print(f"   ✓ Magnitude: mean={self.velocity_magnitude.mean():.3f}, median={np.median(self.velocity_magnitude):.3f}")

            # Check for timepoint info
            if 'timepoint' in self.adata.obs:
                timepoints = self.adata.obs['timepoint'].unique()
                print(f"   ✓ Timepoints: {sorted(timepoints)}")

            return True

        except Exception as e:
            print(f"   Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def interpolate_to_grid(self, grid_resolution=60):
        """Interpolate velocity to regular grid for streamplot."""
        print(f"\n2. Interpolating velocity to {grid_resolution}×{grid_resolution} grid...")

        # Create regular grid
        x_min, x_max = self.umap_coords[:, 0].min(), self.umap_coords[:, 0].max()
        y_min, y_max = self.umap_coords[:, 1].min(), self.umap_coords[:, 1].max()

        # Add padding
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05

        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, grid_resolution)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # Interpolate velocity components
        U_grid = griddata(
            self.umap_coords, self.velocity_smoothed[:, 0],
            (X_grid, Y_grid), method='linear', fill_value=0
        )
        V_grid = griddata(
            self.umap_coords, self.velocity_smoothed[:, 1],
            (X_grid, Y_grid), method='linear', fill_value=0
        )

        # Fill NaN with nearest neighbor
        mask_nan = np.isnan(U_grid) | np.isnan(V_grid)
        if np.any(mask_nan):
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

        self.grid_velocity = {
            'X_grid': X_grid,
            'Y_grid': Y_grid,
            'U_grid': U_grid,
            'V_grid': V_grid,
            'speed_grid': np.sqrt(U_grid**2 + V_grid**2)
        }

        print(f"   ✓ Grid interpolation complete")
        print(f"   Grid velocity: U[{U_grid.min():.3f}, {U_grid.max():.3f}], V[{V_grid.min():.3f}, {V_grid.max():.3f}]")

        return self.grid_velocity

    def sample_arrows_grid_based(self, n_arrows=5000):
        """Sample arrows evenly across UMAP space using grid."""
        print(f"\n   Grid-based sampling: {n_arrows} arrows...")

        # Create grid over UMAP space
        x_min, x_max = self.umap_coords[:, 0].min(), self.umap_coords[:, 0].max()
        y_min, y_max = self.umap_coords[:, 1].min(), self.umap_coords[:, 1].max()

        # Determine grid size
        n_grid = int(np.sqrt(n_arrows))
        x_bins = np.linspace(x_min, x_max, n_grid + 1)
        y_bins = np.linspace(y_min, y_max, n_grid + 1)

        # Sample one arrow per grid cell
        arrow_indices = []
        for i in range(n_grid):
            for j in range(n_grid):
                # Find peaks in this grid cell
                mask = (
                    (self.umap_coords[:, 0] >= x_bins[i]) &
                    (self.umap_coords[:, 0] < x_bins[i+1]) &
                    (self.umap_coords[:, 1] >= y_bins[j]) &
                    (self.umap_coords[:, 1] < y_bins[j+1])
                )

                indices_in_cell = np.where(mask)[0]
                if len(indices_in_cell) > 0:
                    # Pick the peak with highest velocity in this cell
                    best_idx = indices_in_cell[np.argmax(self.velocity_magnitude[indices_in_cell])]
                    arrow_indices.append(best_idx)

        arrow_indices = np.array(arrow_indices)
        print(f"   ✓ Sampled {len(arrow_indices)} arrows")

        return arrow_indices

    def sample_arrows_velocity_weighted(self, n_arrows=5000):
        """Sample arrows weighted by velocity magnitude."""
        print(f"\n   Velocity-weighted sampling: {n_arrows} arrows...")

        # Weight by velocity magnitude
        weights = self.velocity_magnitude.copy()
        weights = weights / weights.sum()

        # Sample with replacement to get desired number
        arrow_indices = np.random.choice(
            len(self.umap_coords),
            size=n_arrows,
            replace=False,
            p=weights
        )

        print(f"   ✓ Sampled {len(arrow_indices)} arrows")

        return arrow_indices

    def sample_arrows_adaptive(self, n_arrows=5000, min_distance=0.5):
        """Sample arrows with adaptive density - avoid overlaps."""
        print(f"\n   Adaptive sampling: {n_arrows} arrows (min_dist={min_distance})...")

        # Start with highest velocity peaks
        sorted_indices = np.argsort(self.velocity_magnitude)[::-1]

        arrow_indices = []
        arrow_coords = []

        for idx in sorted_indices:
            if len(arrow_indices) >= n_arrows:
                break

            coord = self.umap_coords[idx]

            # Check distance to existing arrows
            if len(arrow_coords) == 0:
                arrow_indices.append(idx)
                arrow_coords.append(coord)
            else:
                distances = np.sqrt(((np.array(arrow_coords) - coord)**2).sum(axis=1))
                if distances.min() > min_distance:
                    arrow_indices.append(idx)
                    arrow_coords.append(coord)

        arrow_indices = np.array(arrow_indices)
        print(f"   ✓ Sampled {len(arrow_indices)} arrows")

        return arrow_indices

    def create_streamplot_only(self, density=2.0, figsize=(16, 14)):
        """Create clean streamplot visualization."""
        print(f"\n3. Creating streamplot visualization (density={density})...")

        if self.grid_velocity is None:
            print("   Error: Grid velocity not computed")
            return None

        X_grid = self.grid_velocity['X_grid']
        Y_grid = self.grid_velocity['Y_grid']
        U_grid = self.grid_velocity['U_grid']
        V_grid = self.grid_velocity['V_grid']
        speed_grid = self.grid_velocity['speed_grid']

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        # Background: all peaks colored by velocity magnitude
        scatter = ax.scatter(
            self.umap_coords[:, 0], self.umap_coords[:, 1],
            c=self.velocity_magnitude, cmap='viridis',
            s=2, alpha=0.6, rasterized=True, vmin=0
        )

        # Streamlines
        strm = ax.streamplot(
            X_grid, Y_grid, U_grid, V_grid,
            density=density, color='white',
            linewidth=2.5, arrowsize=2.5,
            arrowstyle='->', integration_direction='forward'
        )

        # Styling
        ax.set_xlabel('UMAP 1', fontsize=20)
        ax.set_ylabel('UMAP 2', fontsize=20)
        ax.set_title(
            f'Chromatin Velocity Streamlines - {len(self.umap_coords):,} Peaks',
            fontsize=22, fontweight='bold', pad=20
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Velocity Magnitude', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        # Statistics
        stats_text = f"""Connectivity-smoothed velocity
{len(self.umap_coords):,} peaks
Mean velocity: {self.velocity_magnitude.mean():.2f}
Streamline density: {density}"""

        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=13,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

        ax.tick_params(labelsize=16)
        plt.tight_layout()

        # Save
        save_path = f"chromatin_velocity_streamplot_density{density:.1f}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Saved: {save_path}")

        return fig

    def create_arrows_only(self, n_arrows=5000, sampling='grid', figsize=(16, 14)):
        """Create arrow/quiver visualization."""
        print(f"\n4. Creating arrow visualization ({n_arrows} arrows, {sampling} sampling)...")

        # Sample arrows based on strategy
        if sampling == 'grid':
            arrow_indices = self.sample_arrows_grid_based(n_arrows)
        elif sampling == 'weighted':
            arrow_indices = self.sample_arrows_velocity_weighted(n_arrows)
        elif sampling == 'adaptive':
            arrow_indices = self.sample_arrows_adaptive(n_arrows, min_distance=0.4)
        else:
            # Random sampling
            arrow_indices = np.random.choice(len(self.umap_coords), n_arrows, replace=False)

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        # Background: all peaks
        scatter = ax.scatter(
            self.umap_coords[:, 0], self.umap_coords[:, 1],
            c=self.velocity_magnitude, cmap='viridis',
            s=2, alpha=0.6, rasterized=True, vmin=0
        )

        # Arrows
        ax.quiver(
            self.umap_coords[arrow_indices, 0],
            self.umap_coords[arrow_indices, 1],
            self.velocity_smoothed[arrow_indices, 0],
            self.velocity_smoothed[arrow_indices, 1],
            self.velocity_magnitude[arrow_indices],
            angles='xy', scale_units='xy', scale=0.4,
            width=0.003, cmap='plasma', alpha=0.9,
            clim=(0, np.percentile(self.velocity_magnitude, 99))
        )

        # Styling
        ax.set_xlabel('UMAP 1', fontsize=20)
        ax.set_ylabel('UMAP 2', fontsize=20)
        ax.set_title(
            f'Chromatin Velocity Arrows - {len(arrow_indices):,} vectors',
            fontsize=22, fontweight='bold', pad=20
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Velocity Magnitude', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        # Statistics
        stats_text = f"""Connectivity-smoothed velocity
Sampling: {sampling}
{len(arrow_indices):,} arrows
Mean velocity: {self.velocity_magnitude.mean():.2f}"""

        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=13,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

        ax.tick_params(labelsize=16)
        plt.tight_layout()

        # Save
        save_path = f"chromatin_velocity_arrows_{sampling}_{n_arrows}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Saved: {save_path}")

        return fig

    def create_combined_panel(self, streamline_density=2.0, n_arrows=3000):
        """Create 4-panel combined visualization."""
        print(f"\n5. Creating combined 4-panel visualization...")

        if self.grid_velocity is None:
            print("   Error: Grid velocity not computed")
            return None

        X_grid = self.grid_velocity['X_grid']
        Y_grid = self.grid_velocity['Y_grid']
        U_grid = self.grid_velocity['U_grid']
        V_grid = self.grid_velocity['V_grid']
        speed_grid = self.grid_velocity['speed_grid']

        # Create figure with 4 panels
        fig = plt.figure(figsize=(24, 20), dpi=150)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)

        # Panel 1: Streamplot only
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(
            self.umap_coords[:, 0], self.umap_coords[:, 1],
            c=self.velocity_magnitude, cmap='viridis',
            s=1, alpha=0.6, rasterized=True, vmin=0
        )
        ax1.streamplot(
            X_grid, Y_grid, U_grid, V_grid,
            density=streamline_density, color='white',
            linewidth=2, arrowsize=2, arrowstyle='->'
        )
        ax1.set_title('Streamplot', fontsize=18, fontweight='bold')
        ax1.set_xlabel('UMAP 1', fontsize=14)
        ax1.set_ylabel('UMAP 2', fontsize=14)
        plt.colorbar(scatter1, ax=ax1, shrink=0.7, label='Velocity Magnitude')

        # Panel 2: Arrows only (grid sampling)
        ax2 = fig.add_subplot(gs[0, 1])
        arrow_indices_grid = self.sample_arrows_grid_based(n_arrows)
        scatter2 = ax2.scatter(
            self.umap_coords[:, 0], self.umap_coords[:, 1],
            c=self.velocity_magnitude, cmap='viridis',
            s=1, alpha=0.6, rasterized=True, vmin=0
        )
        ax2.quiver(
            self.umap_coords[arrow_indices_grid, 0],
            self.umap_coords[arrow_indices_grid, 1],
            self.velocity_smoothed[arrow_indices_grid, 0],
            self.velocity_smoothed[arrow_indices_grid, 1],
            angles='xy', scale_units='xy', scale=0.4,
            width=0.003, color='red', alpha=0.8
        )
        ax2.set_title(f'Arrows (grid, n={len(arrow_indices_grid):,})', fontsize=18, fontweight='bold')
        ax2.set_xlabel('UMAP 1', fontsize=14)
        ax2.set_ylabel('UMAP 2', fontsize=14)
        plt.colorbar(scatter2, ax=ax2, shrink=0.7, label='Velocity Magnitude')

        # Panel 3: Combined streamplot + arrows
        ax3 = fig.add_subplot(gs[1, 0])
        arrow_indices_adaptive = self.sample_arrows_adaptive(n_arrows // 2, min_distance=0.6)
        scatter3 = ax3.scatter(
            self.umap_coords[:, 0], self.umap_coords[:, 1],
            c=self.velocity_magnitude, cmap='viridis',
            s=1, alpha=0.6, rasterized=True, vmin=0
        )
        ax3.streamplot(
            X_grid, Y_grid, U_grid, V_grid,
            density=streamline_density * 0.8, color='white',
            linewidth=1.5, arrowsize=1.5, arrowstyle='->'
        )
        ax3.quiver(
            self.umap_coords[arrow_indices_adaptive, 0],
            self.umap_coords[arrow_indices_adaptive, 1],
            self.velocity_smoothed[arrow_indices_adaptive, 0],
            self.velocity_smoothed[arrow_indices_adaptive, 1],
            angles='xy', scale_units='xy', scale=0.5,
            width=0.003, color='orange', alpha=0.9
        )
        ax3.set_title(f'Combined (n={len(arrow_indices_adaptive):,})', fontsize=18, fontweight='bold')
        ax3.set_xlabel('UMAP 1', fontsize=14)
        ax3.set_ylabel('UMAP 2', fontsize=14)
        plt.colorbar(scatter3, ax=ax3, shrink=0.7, label='Velocity Magnitude')

        # Panel 4: Velocity magnitude distribution by timepoint
        ax4 = fig.add_subplot(gs[1, 1])
        if 'timepoint' in self.adata.obs:
            timepoints = sorted(self.adata.obs['timepoint'].unique())
            for tp in timepoints:
                mask = self.adata.obs['timepoint'] == tp
                vel_mag = self.velocity_magnitude[mask]
                ax4.hist(vel_mag, bins=50, alpha=0.6, label=tp, density=True)
            ax4.legend(fontsize=12, title='Timepoint')
            ax4.set_xlabel('Velocity Magnitude', fontsize=14)
            ax4.set_ylabel('Density', fontsize=14)
            ax4.set_title('Velocity Distribution by Timepoint', fontsize=18, fontweight='bold')
            ax4.set_yscale('log')
        else:
            # Fallback: overall distribution
            ax4.hist(self.velocity_magnitude, bins=100, alpha=0.7, color='steelblue')
            ax4.set_xlabel('Velocity Magnitude', fontsize=14)
            ax4.set_ylabel('Count', fontsize=14)
            ax4.set_title('Velocity Magnitude Distribution', fontsize=18, fontweight='bold')
            ax4.set_yscale('log')

        # Main title
        fig.suptitle(
            f'Chromatin Velocity Visualization - {len(self.umap_coords):,} Peaks',
            fontsize=24, fontweight='bold', y=0.995
        )

        # Save
        save_path = "chromatin_velocity_combined_panel.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Saved: {save_path}")

        return fig


def main():
    """Main execution."""

    print("\nInitializing visualizer...")

    # Load pre-computed velocity data
    velocity_data_path = "peak_umap_chromatin_velocity_connectivity_smoothed.h5ad"
    visualizer = ChromatinVelocityVisualizer(velocity_data_path)

    if not visualizer.load_data():
        print("Failed to load data. Exiting.")
        return None

    # Interpolate to grid for streamplot
    visualizer.interpolate_to_grid(grid_resolution=60)

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Streamplot with different densities
    for density in [1.5, 2.0, 2.5]:
        visualizer.create_streamplot_only(density=density)

    # Arrows with different sampling strategies
    for sampling in ['grid', 'adaptive', 'weighted']:
        visualizer.create_arrows_only(n_arrows=5000, sampling=sampling)

    # Combined panel
    visualizer.create_combined_panel(streamline_density=2.0, n_arrows=3000)

    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("- chromatin_velocity_streamplot_density*.png (3 files)")
    print("- chromatin_velocity_arrows_*.png (3 files)")
    print("- chromatin_velocity_combined_panel.png")
    print("\n✓ All visualizations use connectivity-based smoothing")
    print("  from adata.obsp['connectivities'] (high-dimensional PCA space)")

    return visualizer


if __name__ == "__main__":
    result = main()
