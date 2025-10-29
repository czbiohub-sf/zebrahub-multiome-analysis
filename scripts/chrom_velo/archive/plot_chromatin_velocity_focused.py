#!/usr/bin/env python3
"""
Focused Chromatin Velocity Visualization

Creates clean, publication-ready plots showing:
1. UMAP with cell type and timepoint coloring
2. Velocity streamplot overlaid on UMAP
3. Velocity magnitude analysis
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

def load_results():
    """Load pre-computed UMAP and velocity results"""
    print("Loading pre-computed results...")
    adata = sc.read_h5ad('./chromatin_velocity_umap_results.h5ad')
    print(f"✓ Loaded results: {adata.shape}")
    return adata

def create_focused_plots(adata):
    """Create focused visualization plots"""
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    umap_coords = adata.obsm['X_umap']
    velocity_umap = adata.obsm['velocity_umap']
    
    # Color palettes
    celltype_colors = dict(zip(adata.obs['celltype'].unique(), 
                              sns.color_palette("Set1", n_colors=len(adata.obs['celltype'].unique()))))
    
    timepoint_colors = dict(zip(adata.obs['timepoint'].unique(),
                               sns.color_palette("viridis", n_colors=len(adata.obs['timepoint'].unique()))))
    
    # 1. UMAP colored by celltype
    ax1 = axes[0, 0]
    for celltype in adata.obs['celltype'].unique():
        mask = adata.obs['celltype'] == celltype
        ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                   c=[celltype_colors[celltype]], label=celltype, 
                   s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax1.set_title('UMAP Embedding - Cell Types', fontsize=14, fontweight='bold')
    ax1.set_xlabel('UMAP 1', fontsize=12)
    ax1.set_ylabel('UMAP 2', fontsize=12)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # 2. UMAP colored by timepoint
    ax2 = axes[0, 1]
    for timepoint in sorted(adata.obs['timepoint'].unique()):
        mask = adata.obs['timepoint'] == timepoint
        ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                   c=[timepoint_colors[timepoint]], label=timepoint, 
                   s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax2.set_title('UMAP Embedding - Timepoints', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP 1', fontsize=12)
    ax2.set_ylabel('UMAP 2', fontsize=12)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # 3. Velocity streamplot
    ax3 = axes[1, 0]
    
    # Create grid for streamplot
    x_min, x_max = umap_coords[:, 0].min() - 0.5, umap_coords[:, 0].max() + 0.5
    y_min, y_max = umap_coords[:, 1].min() - 0.5, umap_coords[:, 1].max() + 0.5
    
    nx, ny = 25, 25
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate velocity to grid
    U = griddata(umap_coords, velocity_umap[:, 0], (X, Y), method='linear', fill_value=0)
    V = griddata(umap_coords, velocity_umap[:, 1], (X, Y), method='linear', fill_value=0)
    
    # Create streamplot
    strm = ax3.streamplot(X, Y, U, V, density=2.0, linewidth=1.5, arrowsize=2.0, 
                         color='darkblue', arrowstyle='->')
    
    # Overlay points colored by velocity magnitude
    velocity_mag = adata.obs['velocity_magnitude']
    scatter = ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                         c=velocity_mag, s=60, cmap='Reds', 
                         alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax3.set_title('Chromatin Velocity Streamplot', fontsize=14, fontweight='bold')
    ax3.set_xlabel('UMAP 1', fontsize=12)
    ax3.set_ylabel('UMAP 2', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for velocity magnitude
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Velocity Magnitude', fontsize=10)
    
    # 4. Velocity magnitude analysis
    ax4 = axes[1, 1]
    
    # Violin plot of velocity magnitude by celltype
    celltype_data = []
    celltype_labels = []
    for celltype in adata.obs['celltype'].unique():
        mask = adata.obs['celltype'] == celltype
        celltype_data.append(velocity_mag[mask])
        celltype_labels.append(celltype)
    
    parts = ax4.violinplot(celltype_data, positions=range(len(celltype_labels)), 
                          showmeans=True, showmedians=True)
    
    # Color the violin plots
    colors = [celltype_colors[ct] for ct in celltype_labels]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax4.set_title('Velocity Magnitude by Cell Type', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Cell Type', fontsize=12)
    ax4.set_ylabel('Velocity Magnitude', fontsize=12)
    ax4.set_xticks(range(len(celltype_labels)))
    ax4.set_xticklabels(celltype_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('./chromatin_velocity_focused_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('./chromatin_velocity_focused_analysis.pdf', bbox_inches='tight')
    
    print("✓ Focused visualizations saved:")
    print("  - chromatin_velocity_focused_analysis.png")
    print("  - chromatin_velocity_focused_analysis.pdf")
    
    return fig

def analyze_velocity_statistics(adata):
    """Detailed velocity statistics analysis"""
    
    print("\n" + "="*50)
    print("CHROMATIN VELOCITY ANALYSIS SUMMARY")
    print("="*50)
    
    velocity_mag = adata.obs['velocity_magnitude']
    
    # Overall statistics
    print(f"Overall Velocity Statistics:")
    print(f"  Mean magnitude: {velocity_mag.mean():.3f}")
    print(f"  Std deviation: {velocity_mag.std():.3f}")
    print(f"  Min magnitude: {velocity_mag.min():.3f}")
    print(f"  Max magnitude: {velocity_mag.max():.3f}")
    print(f"  Median: {velocity_mag.median():.3f}")
    
    # Cell type analysis
    print(f"\nVelocity by Cell Type:")
    celltype_stats = adata.obs.groupby('celltype')['velocity_magnitude'].agg(['mean', 'std', 'count'])
    celltype_stats_sorted = celltype_stats.sort_values('mean', ascending=False)
    
    for celltype, stats in celltype_stats_sorted.iterrows():
        print(f"  {celltype:12s}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={int(stats['count']):2d}")
    
    # Timepoint analysis
    print(f"\nVelocity by Timepoint:")
    timepoint_stats = adata.obs.groupby('timepoint')['velocity_magnitude'].agg(['mean', 'std', 'count'])
    
    # Try to sort timepoints numerically
    try:
        timepoint_stats['timepoint_num'] = timepoint_stats.index.str.extract(r'(\d+)').astype(float)
        timepoint_stats_sorted = timepoint_stats.sort_values('timepoint_num')
    except:
        timepoint_stats_sorted = timepoint_stats.sort_values('mean', ascending=False)
    
    for timepoint, stats in timepoint_stats_sorted.iterrows():
        print(f"  {timepoint:8s}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={int(stats['count']):2d}")
    
    # High velocity pseudobulks
    high_vel_threshold = np.percentile(velocity_mag, 75)
    high_vel_mask = velocity_mag > high_vel_threshold
    high_vel_data = adata.obs[high_vel_mask]
    
    print(f"\nHigh-Velocity Pseudobulks (>75th percentile = {high_vel_threshold:.3f}):")
    print(f"  Total count: {high_vel_mask.sum()}")
    print(f"  Cell types: {dict(high_vel_data['celltype'].value_counts())}")
    print(f"  Timepoints: {dict(high_vel_data['timepoint'].value_counts())}")
    
    # Velocity directionality
    velocity_umap = adata.obsm['velocity_umap']
    velocity_x_mean = velocity_umap[:, 0].mean()
    velocity_y_mean = velocity_umap[:, 1].mean()
    
    print(f"\nVelocity Directionality in UMAP Space:")
    print(f"  Mean X component: {velocity_x_mean:.3f}")
    print(f"  Mean Y component: {velocity_y_mean:.3f}")
    print(f"  Overall direction: {np.arctan2(velocity_y_mean, velocity_x_mean) * 180 / np.pi:.1f}°")
    
    return celltype_stats, timepoint_stats

def main():
    """Main analysis function"""
    
    print("="*60)
    print("FOCUSED CHROMATIN VELOCITY VISUALIZATION")
    print("="*60)
    
    # Load results
    adata = load_results()
    
    # Create visualizations
    fig = create_focused_plots(adata)
    
    # Detailed analysis
    celltype_stats, timepoint_stats = analyze_velocity_statistics(adata)
    
    print("\n" + "="*60)
    print("FOCUSED VISUALIZATION COMPLETE!")
    print("="*60)
    print("✅ High-quality plots generated")
    print("✅ Detailed statistics computed")
    print("✅ Results ready for publication/presentation")
    
    return adata, fig, celltype_stats, timepoint_stats

if __name__ == "__main__":
    adata, fig, celltype_stats, timepoint_stats = main()
    plt.show()