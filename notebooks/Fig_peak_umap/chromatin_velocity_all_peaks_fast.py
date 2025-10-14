#!/usr/bin/env python3
"""
Fast Chromatin Velocity Analysis - ALL PEAKS

Optimized version to compute chromatin velocity for ALL 640,830 peaks 
with faster processing and reduced memory usage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
import gc
from tqdm import tqdm
warnings.filterwarnings('ignore')

print("=== FAST Chromatin Velocity Analysis - ALL PEAKS ===")

def load_and_compute_velocity_fast():
    """Load data and compute velocity with optimized approach."""
    
    print("\n1. Loading original AnnData...")
    adata_path = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad'
    adata = sc.read_h5ad(adata_path)
    print(f"✓ Loaded: {adata.shape}")
    
    # Get accessibility data
    if 'normalized' in adata.layers:
        accessibility = adata.layers['normalized']
    else:
        accessibility = adata.X
    
    print(f"✓ Using accessibility layer: {'normalized' if 'normalized' in adata.layers else 'X'}")
    
    # Convert to dense if needed (more memory but faster computation)
    if hasattr(accessibility, 'toarray'):
        print("   Converting to dense matrix...")
        accessibility = accessibility.toarray()
    
    print(f"✓ Accessibility matrix: {accessibility.shape}")
    
    # Simple velocity computation using temporal smoothing
    print("\n2. Computing velocity using temporal approach...")
    
    # Get pseudobulk metadata for temporal ordering
    pseudobulk_names = list(adata.var_names)
    
    # Parse timepoints from pseudobulk names  
    timepoints = []
    for pb_name in pseudobulk_names:
        if 'somites' in pb_name:
            # Extract number before 'somites'
            parts = pb_name.split('_')
            for part in parts:
                if 'som' in part:
                    try:
                        tp_num = int(''.join(filter(str.isdigit, part)))
                        timepoints.append(tp_num)
                        break
                    except:
                        timepoints.append(0)
                        break
            else:
                timepoints.append(0)
        else:
            timepoints.append(0)
    
    timepoints = np.array(timepoints)
    print(f"   Parsed timepoints: {np.unique(timepoints)}")
    
    # Compute velocity as temporal derivative
    print("   Computing temporal derivatives...")
    
    # Sort by timepoint
    time_order = np.argsort(timepoints)
    sorted_accessibility = accessibility[:, time_order]
    sorted_times = timepoints[time_order]
    
    # Compute differences between consecutive timepoints
    velocity_temporal = np.zeros_like(accessibility)
    
    for i in range(len(sorted_times) - 1):
        dt = sorted_times[i+1] - sorted_times[i]
        if dt > 0:
            # Velocity as change per unit time
            dA = sorted_accessibility[:, i+1] - sorted_accessibility[:, i]
            velocity_temporal[:, time_order[i]] = dA / dt
    
    # Handle last timepoint
    if len(sorted_times) > 1:
        velocity_temporal[:, time_order[-1]] = velocity_temporal[:, time_order[-2]]
    
    # Compute velocity magnitude
    velocity_magnitude = np.sqrt((velocity_temporal**2).sum(axis=1))
    
    print(f"✓ Velocity computed for {len(velocity_magnitude):,} peaks")
    print(f"   Magnitude range: [{velocity_magnitude.min():.3f}, {velocity_magnitude.max():.3f}]")
    print(f"   Mean magnitude: {velocity_magnitude.mean():.3f}")
    
    # Add to AnnData
    adata.layers['velocity_temporal'] = velocity_temporal
    adata.obs['velocity_magnitude'] = velocity_magnitude
    
    return adata

def compute_umap_velocity_projection_fast(adata, n_neighbors=10):
    """Fast velocity projection using UMAP neighborhoods."""
    
    print("\n3. Computing UMAP velocity projection...")
    
    # Get data
    umap_coords = adata.obsm['X_umap']
    velocity_temporal = adata.layers['velocity_temporal']
    velocity_magnitude = adata.obs['velocity_magnitude'].values
    
    print(f"   UMAP coordinates: {umap_coords.shape}")
    print(f"   Velocity matrix: {velocity_temporal.shape}")
    
    # Use simple PCA to get 2D velocity direction
    print("   Computing velocity directions with PCA...")
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    
    # Use TruncatedSVD for sparse-friendly computation
    if velocity_temporal.shape[1] > 2:
        svd = TruncatedSVD(n_components=2, random_state=42)
        velocity_2d = svd.fit_transform(velocity_temporal)
    else:
        velocity_2d = velocity_temporal[:, :2]
    
    # Normalize directions and scale by magnitude
    velocity_norm = np.sqrt((velocity_2d**2).sum(axis=1))
    velocity_norm[velocity_norm == 0] = 1
    
    velocity_2d_normed = velocity_2d / velocity_norm[:, np.newaxis]
    velocity_umap = velocity_2d_normed * velocity_magnitude[:, np.newaxis] * 0.1  # Scale factor
    
    print(f"✓ Velocity UMAP projection: {velocity_umap.shape}")
    print(f"   Range X: [{velocity_umap[:, 0].min():.3f}, {velocity_umap[:, 0].max():.3f}]")
    print(f"   Range Y: [{velocity_umap[:, 1].min():.3f}, {velocity_umap[:, 1].max():.3f}]")
    
    # Add to AnnData
    adata.obsm['velocity_umap_fast'] = velocity_umap
    
    return adata

def plot_all_peaks_velocity_fast(adata, max_arrows=8000):
    """Create fast visualization for all peaks."""
    
    print("\n4. Creating velocity visualizations...")
    
    # Get data
    umap_coords = adata.obsm['X_umap']
    velocity_umap = adata.obsm['velocity_umap_fast']
    velocity_magnitude = adata.obs['velocity_magnitude'].values
    
    print(f"   Visualizing {len(umap_coords):,} peaks")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    
    # Plot 1: All peaks by velocity magnitude
    ax = axes[0, 0]
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                        c=velocity_magnitude, cmap='viridis', 
                        s=0.5, alpha=0.6, rasterized=True)
    ax.set_title(f'ALL {len(umap_coords):,} Peaks - Velocity Magnitude', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, shrink=0.7, label='Velocity Magnitude')
    
    # Plot 2: High-velocity peaks with arrows
    ax = axes[0, 1]
    
    # Background
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
              c='lightgray', s=0.1, alpha=0.3, rasterized=True)
    
    # Select high-velocity peaks
    velocity_threshold = np.percentile(velocity_magnitude, 85)
    high_vel_mask = velocity_magnitude >= velocity_threshold
    high_vel_indices = np.where(high_vel_mask)[0]
    
    print(f"   High-velocity peaks (>85th percentile): {len(high_vel_indices):,}")
    
    # Sample for arrows
    if len(high_vel_indices) > max_arrows:
        arrow_indices = np.random.choice(high_vel_indices, max_arrows, replace=False)
    else:
        arrow_indices = high_vel_indices
    
    # Plot high-velocity points
    ax.scatter(umap_coords[arrow_indices, 0], umap_coords[arrow_indices, 1], 
              c=velocity_magnitude[arrow_indices], cmap='plasma', 
              s=3, alpha=0.8, zorder=2)
    
    # Add arrows
    quiver = ax.quiver(umap_coords[arrow_indices, 0], umap_coords[arrow_indices, 1],
                      velocity_umap[arrow_indices, 0], velocity_umap[arrow_indices, 1],
                      angles='xy', scale_units='xy', scale=0.3,
                      width=0.002, color='red', alpha=0.8, zorder=3)
    
    ax.set_title(f'Velocity Vectors - {len(arrow_indices):,} Arrows', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Plot 3: Velocity magnitude histogram
    ax = axes[1, 0]
    ax.hist(velocity_magnitude[velocity_magnitude > 0], bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Velocity Magnitude')
    ax.set_ylabel('Number of Peaks')
    ax.set_title('Velocity Distribution', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    
    # Add percentile lines
    percentiles = [50, 75, 85, 90, 95, 99]
    colors = plt.cm.Set1(np.linspace(0, 1, len(percentiles)))
    for i, p in enumerate(percentiles):
        val = np.percentile(velocity_magnitude, p)
        ax.axvline(val, color=colors[i], linestyle='--', alpha=0.7, 
                  label=f'{p}th: {val:.3f}')
    ax.legend()
    
    # Plot 4: Velocity statistics by peak type
    ax = axes[1, 1]
    
    if 'peak_type' in adata.obs.columns:
        peak_types = adata.obs['peak_type'].unique()
        type_velocities = []
        type_names = []
        
        for ptype in peak_types:
            mask = adata.obs['peak_type'] == ptype
            type_vel = velocity_magnitude[mask]
            if len(type_vel) > 0:
                type_velocities.append(type_vel)
                type_names.append(f'{ptype}\\n(n={len(type_vel):,})')
        
        ax.boxplot(type_velocities, labels=type_names)
        ax.set_ylabel('Velocity Magnitude')
        ax.set_title('Velocity by Peak Type', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    else:
        # Alternative: velocity by chromosome
        ax.text(0.5, 0.5, 'Peak type information\\nnot available', 
               transform=ax.transAxes, ha='center', va='center', 
               fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax.set_title('Peak Type Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save comprehensive plot
    save_path = "chromatin_velocity_all_peaks_fast_comprehensive.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Comprehensive plot saved: {save_path}")
    plt.show()
    
    # Create focused publication plot
    print("   Creating focused plot...")
    fig_focused, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # All peaks background
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                        c=velocity_magnitude, cmap='viridis', 
                        s=1, alpha=0.7, rasterized=True)
    
    # Fewer arrows for clean visualization
    focused_threshold = np.percentile(velocity_magnitude, 90)
    focused_mask = velocity_magnitude >= focused_threshold
    focused_indices = np.where(focused_mask)[0]
    
    if len(focused_indices) > 3000:
        focused_indices = np.random.choice(focused_indices, 3000, replace=False)
    
    # Add arrows
    ax.quiver(umap_coords[focused_indices, 0], umap_coords[focused_indices, 1],
              velocity_umap[focused_indices, 0], velocity_umap[focused_indices, 1],
              angles='xy', scale_units='xy', scale=0.2,
              width=0.003, headwidth=3, headlength=4,
              color='red', alpha=0.9, zorder=3)
    
    # Styling
    ax.set_xlabel('UMAP 1', fontsize=16)
    ax.set_ylabel('UMAP 2', fontsize=16)
    ax.set_title(f'Chromatin Velocity - ALL {len(umap_coords):,} Peaks\\n({len(focused_indices):,} velocity arrows)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Velocity Magnitude', fontsize=14)
    
    # Statistics box
    stats_text = f"""Statistics:
Total peaks: {len(umap_coords):,}
Arrows shown: {len(focused_indices):,}
Mean velocity: {velocity_magnitude.mean():.3f}
Max velocity: {velocity_magnitude.max():.3f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9))
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save focused plot
    save_path_focused = "chromatin_velocity_all_peaks_fast_focused.png"
    plt.savefig(save_path_focused, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Focused plot saved: {save_path_focused}")
    plt.show()
    
    return fig, fig_focused

def main():
    """Main execution function."""
    
    print("Starting FAST chromatin velocity analysis for ALL peaks...")
    
    # Step 1: Load data and compute velocity
    adata = load_and_compute_velocity_fast()
    
    # Step 2: Project into UMAP
    adata = compute_umap_velocity_projection_fast(adata)
    
    # Step 3: Create visualizations
    fig_comp, fig_focused = plot_all_peaks_velocity_fast(adata)
    
    # Step 4: Save results
    output_path = "chromatin_velocity_all_peaks_fast.h5ad"
    print(f"\n5. Saving results...")
    adata.write(output_path)
    print(f"✓ Results saved: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ FAST CHROMATIN VELOCITY ANALYSIS COMPLETE - ALL PEAKS!")
    print("="*80)
    print(f"Successfully processed ALL {adata.shape[0]:,} peaks")
    print(f"")
    print("Generated files:")
    print("- chromatin_velocity_all_peaks_fast_comprehensive.png")  
    print("- chromatin_velocity_all_peaks_fast_focused.png")
    print("- chromatin_velocity_all_peaks_fast.h5ad")
    print("")
    print("Key results:")
    print(f"- Total peaks analyzed: {adata.shape[0]:,}")
    print(f"- Mean velocity magnitude: {adata.obs['velocity_magnitude'].mean():.3f}")
    print(f"- Max velocity magnitude: {adata.obs['velocity_magnitude'].max():.3f}")
    print(f"- Peaks with velocity > 0: {(adata.obs['velocity_magnitude'] > 0).sum():,}")
    
    return adata

if __name__ == "__main__":
    result_adata = main()