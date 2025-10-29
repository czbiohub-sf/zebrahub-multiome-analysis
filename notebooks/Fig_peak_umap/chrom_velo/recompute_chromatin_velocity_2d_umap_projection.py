#!/usr/bin/env python3
"""
Re-compute Chromatin Velocity Projection onto 2D UMAP
====================================================

This script re-computes the chromatin velocity projection onto the 2D UMAP
embedding for all peaks, handling potential errors and providing robust 
visualization.

Author: Claude Code
Date: 2025-10-08
Environment: single-cell-base
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
import os
from pathlib import Path
import pickle

warnings.filterwarnings('ignore')
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=80, facecolor='white')

print("=" * 60)
print("Re-computing Chromatin Velocity 2D UMAP Projection")
print("=" * 60)

class ChromatinVelocityUMAPProjector:
    """Class to handle chromatin velocity UMAP projection with error handling."""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.umap_coords = None
        self.velocity_data = None
        self.original_data = None
        self.results = {}
    
    def load_umap_coordinates(self, umap_csv_path=None):
        """Load 2D UMAP coordinates from CSV file."""
        if umap_csv_path is None:
            umap_csv_path = self.base_dir / "peak_umap_3d_annotated_v6.csv"
        
        print(f"\n1. Loading UMAP coordinates from {umap_csv_path}...")
        
        try:
            umap_df = pd.read_csv(umap_csv_path, index_col=0)
            print(f"✓ Loaded UMAP coordinates: {umap_df.shape}")
            print(f"✓ Columns: {list(umap_df.columns)}")
            
            # Extract 2D coordinates
            if 'UMAP_1' in umap_df.columns and 'UMAP_2' in umap_df.columns:
                self.umap_coords = umap_df[['UMAP_1', 'UMAP_2']].values
                self.peak_names = list(umap_df.index)
                print(f"✓ 2D UMAP coordinates shape: {self.umap_coords.shape}")
                print(f"✓ Peak names: {len(self.peak_names)}")
                
                # Store full metadata
                self.umap_metadata = umap_df
                return True
            else:
                print("✗ UMAP_1 and UMAP_2 columns not found")
                return False
                
        except Exception as e:
            print(f"✗ Error loading UMAP coordinates: {e}")
            return False
    
    def load_velocity_results(self, velocity_h5ad_path=None):
        """Load chromatin velocity results from h5ad file."""
        if velocity_h5ad_path is None:
            # Try multiple potential paths
            potential_paths = [
                self.base_dir / "chromatin_velocity_results.h5ad",
                self.base_dir / "chromatin_velocity_all_peaks_fast.h5ad",
                self.base_dir / "chromatin_velocity_umap_integrated.h5ad"
            ]
            
            velocity_h5ad_path = None
            for path in potential_paths:
                if path.exists():
                    velocity_h5ad_path = path
                    break
        
        if velocity_h5ad_path is None:
            print("✗ No velocity results file found")
            return False
            
        print(f"\n2. Loading velocity results from {velocity_h5ad_path}...")
        
        try:
            self.velocity_data = sc.read_h5ad(velocity_h5ad_path)
            print(f"✓ Loaded velocity data: {self.velocity_data.shape}")
            print(f"✓ Layers: {list(self.velocity_data.layers.keys()) if self.velocity_data.layers else 'None'}")
            print(f"✓ Obsm: {list(self.velocity_data.obsm.keys()) if self.velocity_data.obsm else 'None'}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading velocity results: {e}")
            return False
    
    def load_original_data(self, original_h5ad_path=None):
        """Load original peaks data if needed."""
        if original_h5ad_path is None:
            original_h5ad_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad"
        
        print(f"\n3. Loading original data from {original_h5ad_path}...")
        
        try:
            if os.path.exists(original_h5ad_path):
                self.original_data = sc.read_h5ad(original_h5ad_path)
                print(f"✓ Loaded original data: {self.original_data.shape}")
                return True
            else:
                print(f"✗ Original data file not found: {original_h5ad_path}")
                return False
                
        except Exception as e:
            print(f"✗ Error loading original data: {e}")
            return False
    
    def align_data(self):
        """Align velocity data with UMAP coordinates."""
        print("\n4. Aligning velocity data with UMAP coordinates...")
        
        if self.velocity_data is None or self.umap_coords is None:
            print("✗ Missing velocity data or UMAP coordinates")
            return False
        
        # Get peak names from velocity data
        velocity_peak_names = list(self.velocity_data.obs.index)
        umap_peak_names = self.peak_names
        
        print(f"✓ Velocity peaks: {len(velocity_peak_names)}")
        print(f"✓ UMAP peaks: {len(umap_peak_names)}")
        
        # Find common peaks
        common_peaks = list(set(velocity_peak_names) & set(umap_peak_names))
        print(f"✓ Common peaks: {len(common_peaks)}")
        
        if len(common_peaks) == 0:
            print("✗ No common peaks found between velocity and UMAP data")
            return False
        
        # Create index mappings
        velocity_indices = [velocity_peak_names.index(peak) for peak in common_peaks]
        umap_indices = [umap_peak_names.index(peak) for peak in common_peaks]
        
        # Align data
        self.aligned_velocity_data = self.velocity_data[velocity_indices, :]
        self.aligned_umap_coords = self.umap_coords[umap_indices, :]
        self.aligned_peak_names = common_peaks
        self.aligned_metadata = self.umap_metadata.loc[common_peaks]
        
        print(f"✓ Aligned data shape - Velocity: {self.aligned_velocity_data.shape}")
        print(f"✓ Aligned data shape - UMAP: {self.aligned_umap_coords.shape}")
        
        return True
    
    def compute_velocity_vectors_2d(self, method='difference', n_neighbors=30):
        """Compute velocity vectors projected onto 2D UMAP space."""
        print(f"\n5. Computing 2D velocity vectors using {method} method...")
        
        if not hasattr(self, 'aligned_velocity_data'):
            print("✗ Data not aligned. Run align_data() first.")
            return False
        
        try:
            # Method 1: Use velocity layer if available
            if 'velocity' in self.aligned_velocity_data.layers:
                print("✓ Using velocity layer from results")
                velocity_matrix = self.aligned_velocity_data.layers['velocity']
                
            elif 'spliced' in self.aligned_velocity_data.layers and 'unspliced' in self.aligned_velocity_data.layers:
                print("✓ Computing velocity from spliced/unspliced layers")
                velocity_matrix = (self.aligned_velocity_data.layers['unspliced'] - 
                                 self.aligned_velocity_data.layers['spliced'])
                
            else:
                print("✗ No suitable velocity data found in layers")
                return False
            
            # Convert to dense if sparse
            if sp.issparse(velocity_matrix):
                velocity_matrix = velocity_matrix.toarray()
            
            print(f"✓ Velocity matrix shape: {velocity_matrix.shape}")
            
            # Compute 2D velocity vectors using PCA-like projection
            n_peaks, n_pseudobulks = velocity_matrix.shape
            
            # Method: Use pseudobulk positions as temporal coordinates
            # and project velocity onto 2D UMAP space
            
            # Simple approach: compute velocity magnitude per peak
            velocity_magnitudes = np.sqrt((velocity_matrix**2).sum(axis=1))
            
            # Compute velocity direction in 2D UMAP space using neighbors
            print("✓ Computing 2D velocity vectors using neighborhood approach...")
            
            # Build neighbor graph in UMAP space
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
            nbrs.fit(self.aligned_umap_coords)
            
            # For each peak, compute velocity vector as weighted average 
            # of neighbor displacements
            velocity_vectors_2d = np.zeros((n_peaks, 2))
            
            for i in range(n_peaks):
                # Get neighbors
                distances, indices = nbrs.kneighbors([self.aligned_umap_coords[i]])
                neighbor_indices = indices[0]
                neighbor_distances = distances[0]
                
                # Compute velocity-weighted displacement
                if len(neighbor_indices) > 1:
                    # Get velocity magnitudes of neighbors
                    neighbor_velocities = velocity_magnitudes[neighbor_indices]
                    
                    # Compute displacements to neighbors
                    displacements = (self.aligned_umap_coords[neighbor_indices] - 
                                   self.aligned_umap_coords[i])
                    
                    # Weight displacements by velocity similarity
                    if np.sum(neighbor_velocities) > 0:
                        weights = neighbor_velocities / np.sum(neighbor_velocities)
                        velocity_vectors_2d[i] = np.average(displacements, weights=weights, axis=0)
                    
                # Scale by own velocity magnitude
                velocity_vectors_2d[i] *= velocity_magnitudes[i]
            
            self.velocity_vectors_2d = velocity_vectors_2d
            self.velocity_magnitudes = velocity_magnitudes
            
            print(f"✓ Computed 2D velocity vectors: {velocity_vectors_2d.shape}")
            print(f"✓ Velocity magnitude range: [{velocity_magnitudes.min():.3f}, {velocity_magnitudes.max():.3f}]")
            
            return True
            
        except Exception as e:
            print(f"✗ Error computing velocity vectors: {e}")
            return False
    
    def create_visualization(self, save_prefix="chromatin_velocity_2d_umap"):
        """Create comprehensive visualization of 2D velocity projection."""
        print("\n6. Creating visualizations...")
        
        if not hasattr(self, 'velocity_vectors_2d'):
            print("✗ Velocity vectors not computed. Run compute_velocity_vectors_2d() first.")
            return False
        
        try:
            # Set up the figure with multiple subplots
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Basic velocity stream plot
            ax1 = plt.subplot(2, 3, 1)
            self._plot_velocity_stream(ax1, title="Chromatin Velocity Stream Plot")
            
            # 2. Velocity magnitude heatmap
            ax2 = plt.subplot(2, 3, 2)
            self._plot_velocity_magnitude(ax2, title="Velocity Magnitude")
            
            # 3. Colored by cell type
            ax3 = plt.subplot(2, 3, 3)
            self._plot_colored_by_metadata(ax3, 'celltype', title="Colored by Cell Type")
            
            # 4. Colored by timepoint
            ax4 = plt.subplot(2, 3, 4)
            self._plot_colored_by_metadata(ax4, 'timepoint', title="Colored by Timepoint")
            
            # 5. Colored by lineage
            ax5 = plt.subplot(2, 3, 5)
            self._plot_colored_by_metadata(ax5, 'lineage', title="Colored by Lineage")
            
            # 6. Velocity quiver plot
            ax6 = plt.subplot(2, 3, 6)
            self._plot_velocity_quiver(ax6, title="Velocity Quiver Plot")
            
            plt.tight_layout()
            
            # Save the comprehensive figure
            save_path = f"{save_prefix}_comprehensive.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved comprehensive visualization: {save_path}")
            plt.show()
            
            # Create focused velocity plots
            self._create_focused_plots(save_prefix)
            
            return True
            
        except Exception as e:
            print(f"✗ Error creating visualization: {e}")
            return False
    
    def _plot_velocity_stream(self, ax, title="", subsample=1000):
        """Plot velocity as a stream plot."""
        # Subsample for clarity
        n_points = len(self.aligned_umap_coords)
        if n_points > subsample:
            indices = np.random.choice(n_points, subsample, replace=False)
            coords = self.aligned_umap_coords[indices]
            vectors = self.velocity_vectors_2d[indices]
        else:
            coords = self.aligned_umap_coords
            vectors = self.velocity_vectors_2d
        
        # Create streamplot
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Create grid
        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 20)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate velocity onto grid
        from scipy.interpolate import griddata
        
        U = griddata(coords, vectors[:, 0], (X, Y), method='linear', fill_value=0)
        V = griddata(coords, vectors[:, 1], (X, Y), method='linear', fill_value=0)
        
        # Plot streamlines
        ax.streamplot(X, Y, U, V, density=1, color='gray', alpha=0.6)
        
        # Overlay points
        ax.scatter(coords[:, 0], coords[:, 1], c=np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2), 
                  cmap='viridis', s=10, alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    def _plot_velocity_magnitude(self, ax, title=""):
        """Plot velocity magnitude as heatmap."""
        scatter = ax.scatter(self.aligned_umap_coords[:, 0], 
                           self.aligned_umap_coords[:, 1],
                           c=self.velocity_magnitudes,
                           cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    def _plot_colored_by_metadata(self, ax, column, title=""):
        """Plot UMAP colored by metadata column."""
        if column in self.aligned_metadata.columns:
            if self.aligned_metadata[column].dtype == 'object':
                # Categorical coloring
                unique_vals = self.aligned_metadata[column].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_vals)))
                
                for i, val in enumerate(unique_vals):
                    mask = self.aligned_metadata[column] == val
                    ax.scatter(self.aligned_umap_coords[mask, 0], 
                             self.aligned_umap_coords[mask, 1],
                             c=[colors[i]], label=val, s=10, alpha=0.7)
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                # Continuous coloring
                scatter = ax.scatter(self.aligned_umap_coords[:, 0], 
                                   self.aligned_umap_coords[:, 1],
                                   c=self.aligned_metadata[column],
                                   cmap='viridis', s=10, alpha=0.7)
                plt.colorbar(scatter, ax=ax, label=column)
        
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    def _plot_velocity_quiver(self, ax, title="", subsample=500, scale=1.0):
        """Plot velocity as quiver plot."""
        # Subsample for clarity
        n_points = len(self.aligned_umap_coords)
        if n_points > subsample:
            indices = np.random.choice(n_points, subsample, replace=False)
            coords = self.aligned_umap_coords[indices]
            vectors = self.velocity_vectors_2d[indices]
        else:
            coords = self.aligned_umap_coords
            vectors = self.velocity_vectors_2d
        
        # Plot background points
        ax.scatter(self.aligned_umap_coords[:, 0], 
                  self.aligned_umap_coords[:, 1],
                  c='lightgray', s=5, alpha=0.3)
        
        # Plot velocity arrows
        ax.quiver(coords[:, 0], coords[:, 1], 
                 vectors[:, 0] * scale, vectors[:, 1] * scale,
                 angles='xy', scale_units='xy', scale=1,
                 alpha=0.7, width=0.003)
        
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    def _create_focused_plots(self, save_prefix):
        """Create individual focused plots."""
        
        # High-quality stream plot
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_velocity_stream(ax, "Chromatin Velocity Stream Plot - High Resolution")
        plt.savefig(f"{save_prefix}_streamplot_focused.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved focused stream plot: {save_prefix}_streamplot_focused.png")
        plt.close()
        
        # High-quality magnitude plot
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_velocity_magnitude(ax, "Velocity Magnitude - High Resolution")
        plt.savefig(f"{save_prefix}_magnitude_focused.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved focused magnitude plot: {save_prefix}_magnitude_focused.png")
        plt.close()
    
    def save_results(self, save_path="chromatin_velocity_2d_umap_results.h5ad"):
        """Save results to AnnData format."""
        print(f"\n7. Saving results to {save_path}...")
        
        try:
            # Create new AnnData with aligned data
            result_adata = sc.AnnData(
                X=self.aligned_velocity_data.X,
                obs=pd.DataFrame(index=self.aligned_peak_names),
                var=self.aligned_velocity_data.var
            )
            
            # Add layers from original velocity data
            for layer in self.aligned_velocity_data.layers:
                result_adata.layers[layer] = self.aligned_velocity_data.layers[layer]
            
            # Add UMAP coordinates
            result_adata.obsm['X_umap'] = self.aligned_umap_coords
            
            # Add velocity vectors
            result_adata.obsm['velocity_umap'] = self.velocity_vectors_2d
            
            # Add metadata
            for col in self.aligned_metadata.columns:
                result_adata.obs[col] = self.aligned_metadata[col].values
            
            # Add velocity magnitudes
            result_adata.obs['velocity_magnitude'] = self.velocity_magnitudes
            
            # Save
            result_adata.write(save_path)
            print(f"✓ Saved results to {save_path}")
            
            # Also save as pickle for easy loading
            pickle_path = save_path.replace('.h5ad', '_full_results.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump({
                    'umap_coords': self.aligned_umap_coords,
                    'velocity_vectors_2d': self.velocity_vectors_2d,
                    'velocity_magnitudes': self.velocity_magnitudes,
                    'peak_names': self.aligned_peak_names,
                    'metadata': self.aligned_metadata
                }, f)
            print(f"✓ Saved full results to {pickle_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error saving results: {e}")
            return False

def main():
    """Main function to run the complete workflow."""
    
    print("Starting chromatin velocity 2D UMAP projection workflow...")
    
    # Initialize projector
    projector = ChromatinVelocityUMAPProjector(base_dir=".")
    
    # Step 1: Load UMAP coordinates
    if not projector.load_umap_coordinates():
        print("Failed to load UMAP coordinates. Exiting.")
        return False
    
    # Step 2: Load velocity results
    if not projector.load_velocity_results():
        print("Failed to load velocity results. Exiting.")
        return False
    
    # Step 3: Load original data (optional)
    projector.load_original_data()  # This is optional
    
    # Step 4: Align data
    if not projector.align_data():
        print("Failed to align data. Exiting.")
        return False
    
    # Step 5: Compute 2D velocity vectors
    if not projector.compute_velocity_vectors_2d():
        print("Failed to compute velocity vectors. Exiting.")
        return False
    
    # Step 6: Create visualizations
    if not projector.create_visualization():
        print("Failed to create visualizations. Exiting.")
        return False
    
    # Step 7: Save results
    if not projector.save_results():
        print("Failed to save results. Exiting.")
        return False
    
    print("\n" + "=" * 60)
    print("Chromatin Velocity 2D UMAP Projection COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("- chromatin_velocity_2d_umap_comprehensive.png")
    print("- chromatin_velocity_2d_umap_streamplot_focused.png") 
    print("- chromatin_velocity_2d_umap_magnitude_focused.png")
    print("- chromatin_velocity_2d_umap_results.h5ad")
    print("- chromatin_velocity_2d_umap_results_full_results.pkl")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("Workflow failed. Check error messages above.")
        exit(1)
    else:
        print("Workflow completed successfully!")