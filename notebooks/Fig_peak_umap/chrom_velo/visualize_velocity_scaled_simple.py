#!/usr/bin/env python3
"""
Simple Chromatin Velocity Visualization with 0.1x Scaling

Takes pre-computed connectivity-smoothed velocity and applies 0.1x scaling
for better directionality visualization.

Author: Zebrahub-Multiome Analysis Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CHROMATIN VELOCITY VISUALIZATION (0.1x scaling)")
print("="*80)

# Load pre-computed velocity data
print("\n1. Loading pre-computed velocity data...")
adata = sc.read_h5ad("peak_umap_chromatin_velocity_connectivity_smoothed.h5ad")
print(f"   ✓ Loaded: {adata.shape}")

# Get data
umap_coords = adata.obsm['X_umap']
velocity_smoothed = adata.obsm['velocity_umap_smoothed']
velocity_magnitude = adata.obs['velocity_magnitude_smoothed'].values

print(f"   ✓ UMAP: {umap_coords.shape}")
print(f"   ✓ Velocity: {velocity_smoothed.shape}")
print(f"   Current magnitude: mean={velocity_magnitude.mean():.3f}, max={velocity_magnitude.max():.3f}")

# Apply 0.1x scaling
print("\n2. Applying 0.1x velocity scaling...")
velocity_scaled = velocity_smoothed * 0.1
velocity_mag_scaled = velocity_magnitude * 0.1

print(f"   ✓ Scaled magnitude: mean={velocity_mag_scaled.mean():.4f}, max={velocity_mag_scaled.max():.4f}")

# Interpolate to grid
print("\n3. Interpolating to grid for streamplot...")

def interpolate_to_grid(coords, velocity, grid_resolution=50):
    """Interpolate velocity to regular grid."""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, grid_resolution)
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    U_grid = griddata(coords, velocity[:, 0], (X_grid, Y_grid), method='linear', fill_value=0)
    V_grid = griddata(coords, velocity[:, 1], (X_grid, Y_grid), method='linear', fill_value=0)

    # Fill NaN
    mask_nan = np.isnan(U_grid) | np.isnan(V_grid)
    if np.any(mask_nan):
        U_grid_nn = griddata(coords, velocity[:, 0], (X_grid, Y_grid), method='nearest')
        V_grid_nn = griddata(coords, velocity[:, 1], (X_grid, Y_grid), method='nearest')
        U_grid[mask_nan] = U_grid_nn[mask_nan]
        V_grid[mask_nan] = V_grid_nn[mask_nan]

    return X_grid, Y_grid, U_grid, V_grid

X_grid, Y_grid, U_grid, V_grid = interpolate_to_grid(umap_coords, velocity_scaled, grid_resolution=50)
print(f"   ✓ Grid velocity: U[{U_grid.min():.4f}, {U_grid.max():.4f}], V[{V_grid.min():.4f}, {V_grid.max():.4f}]")

# Create visualizations
print("\n4. Generating visualizations...")
print("="*80)

# Visualization 1: Streamplot
print("\n   Creating streamplot...")
fig, ax = plt.subplots(figsize=(16, 14), dpi=150)

scatter = ax.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=velocity_mag_scaled, cmap='viridis',
    s=2, alpha=0.6, rasterized=True, vmin=0,
    vmax=np.percentile(velocity_mag_scaled, 99)
)

ax.streamplot(
    X_grid, Y_grid, U_grid, V_grid,
    density=2.0, color='white',
    linewidth=2.5, arrowsize=2.5, arrowstyle='->'
)

ax.set_xlabel('UMAP 1', fontsize=20)
ax.set_ylabel('UMAP 2', fontsize=20)
ax.set_title(
    f'Chromatin Velocity Streamlines (0.1x scale)',
    fontsize=24, fontweight='bold', pad=20
)
ax.tick_params(labelsize=16)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Velocity Magnitude (scaled)', fontsize=18)
cbar.ax.tick_params(labelsize=14)

stats_text = f"""Connectivity-smoothed velocity
{len(umap_coords):,} peaks
Scaling: 0.1x
Mean velocity: {velocity_mag_scaled.mean():.4f}
Max velocity: {velocity_mag_scaled.max():.4f}"""

ax.text(
    0.02, 0.98, stats_text,
    transform=ax.transAxes, fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
)

plt.tight_layout()
plt.savefig("chromatin_velocity_streamplot_scaled0.1x.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"   ✓ Saved: chromatin_velocity_streamplot_scaled0.1x.png")
plt.close()

# Visualization 2: Arrows (grid sampling)
print("\n   Creating arrow plot...")
fig, ax = plt.subplots(figsize=(16, 14), dpi=150)

scatter = ax.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=velocity_mag_scaled, cmap='viridis',
    s=2, alpha=0.6, rasterized=True, vmin=0,
    vmax=np.percentile(velocity_mag_scaled, 99)
)

# Grid-based sampling
n_arrows = 3000
x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
n_grid = int(np.sqrt(n_arrows))
x_bins = np.linspace(x_min, x_max, n_grid + 1)
y_bins = np.linspace(y_min, y_max, n_grid + 1)

arrow_indices = []
for i in range(n_grid):
    for j in range(n_grid):
        mask = (
            (umap_coords[:, 0] >= x_bins[i]) &
            (umap_coords[:, 0] < x_bins[i+1]) &
            (umap_coords[:, 1] >= y_bins[j]) &
            (umap_coords[:, 1] < y_bins[j+1])
        )
        indices_in_cell = np.where(mask)[0]
        if len(indices_in_cell) > 0:
            best_idx = indices_in_cell[np.argmax(velocity_mag_scaled[indices_in_cell])]
            arrow_indices.append(best_idx)

arrow_indices = np.array(arrow_indices)
print(f"   Sampled {len(arrow_indices)} arrows")

ax.quiver(
    umap_coords[arrow_indices, 0],
    umap_coords[arrow_indices, 1],
    velocity_scaled[arrow_indices, 0],
    velocity_scaled[arrow_indices, 1],
    velocity_mag_scaled[arrow_indices],
    angles='xy', scale_units='xy', scale=0.04,  # 0.04 = 0.4 * 0.1
    width=0.003, cmap='plasma', alpha=0.9,
    clim=(0, np.percentile(velocity_mag_scaled, 99))
)

ax.set_xlabel('UMAP 1', fontsize=20)
ax.set_ylabel('UMAP 2', fontsize=20)
ax.set_title(
    f'Chromatin Velocity Arrows (0.1x scale)',
    fontsize=24, fontweight='bold', pad=20
)
ax.tick_params(labelsize=16)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Velocity Magnitude (scaled)', fontsize=18)
cbar.ax.tick_params(labelsize=14)

stats_text = f"""Connectivity-smoothed velocity
{len(arrow_indices):,} arrows
Scaling: 0.1x
Mean velocity: {velocity_mag_scaled.mean():.4f}"""

ax.text(
    0.02, 0.98, stats_text,
    transform=ax.transAxes, fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
)

plt.tight_layout()
plt.savefig("chromatin_velocity_arrows_scaled0.1x.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"   ✓ Saved: chromatin_velocity_arrows_scaled0.1x.png")
plt.close()

# Visualization 3: Combined
print("\n   Creating combined plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 14), dpi=150)

# Left: Streamplot
scatter1 = ax1.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=velocity_mag_scaled, cmap='viridis',
    s=2, alpha=0.6, rasterized=True, vmin=0,
    vmax=np.percentile(velocity_mag_scaled, 99)
)
ax1.streamplot(
    X_grid, Y_grid, U_grid, V_grid,
    density=2.0, color='white',
    linewidth=2.5, arrowsize=2.5, arrowstyle='->'
)
ax1.set_xlabel('UMAP 1', fontsize=18)
ax1.set_ylabel('UMAP 2', fontsize=18)
ax1.set_title('Streamplot', fontsize=20, fontweight='bold')
ax1.tick_params(labelsize=14)
cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.7)
cbar1.set_label('Velocity Magnitude', fontsize=16)

# Right: Arrows
scatter2 = ax2.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=velocity_mag_scaled, cmap='viridis',
    s=2, alpha=0.6, rasterized=True, vmin=0,
    vmax=np.percentile(velocity_mag_scaled, 99)
)
ax2.quiver(
    umap_coords[arrow_indices, 0],
    umap_coords[arrow_indices, 1],
    velocity_scaled[arrow_indices, 0],
    velocity_scaled[arrow_indices, 1],
    velocity_mag_scaled[arrow_indices],
    angles='xy', scale_units='xy', scale=0.04,
    width=0.003, cmap='plasma', alpha=0.9,
    clim=(0, np.percentile(velocity_mag_scaled, 99))
)
ax2.set_xlabel('UMAP 1', fontsize=18)
ax2.set_ylabel('UMAP 2', fontsize=18)
ax2.set_title(f'Arrows ({len(arrow_indices):,} vectors)', fontsize=20, fontweight='bold')
ax2.tick_params(labelsize=14)
cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.7)
cbar2.set_label('Velocity Magnitude', fontsize=16)

fig.suptitle(
    f'Chromatin Velocity Visualization (0.1x scaling) - {len(umap_coords):,} Peaks',
    fontsize=26, fontweight='bold', y=0.98
)

plt.tight_layout()
plt.savefig("chromatin_velocity_combined_scaled0.1x.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"   ✓ Saved: chromatin_velocity_combined_scaled0.1x.png")
plt.close()

print("\n" + "="*80)
print("✓ VISUALIZATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("- chromatin_velocity_streamplot_scaled0.1x.png")
print("- chromatin_velocity_arrows_scaled0.1x.png")
print("- chromatin_velocity_combined_scaled0.1x.png")
print("\n✓ All visualizations use 0.1x velocity scaling for better directionality")
print("✓ Velocity smoothing uses pre-computed connectivity from")
print("  adata.obsp['connectivities'] (high-dimensional PCA space)")
