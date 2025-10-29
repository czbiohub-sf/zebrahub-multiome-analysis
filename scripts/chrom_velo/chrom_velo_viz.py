"""
Chromatin Velocity Visualization Module

Provides utilities for visualizing chromatin velocity in 2D space:
- Streamplots with velocity flow
- Arrow plots with velocity vectors
- Diagnostic plots for validation
- GPU-compatible (ensures arrays are converted to NumPy before plotting)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import scanpy as sc


class ChromatinVelocityVisualizer:
    """
    Visualize chromatin velocity in 2D UMAP space.
    GPU-compatible: Automatically converts CuPy arrays to NumPy before plotting.

    Attributes:
        adata: AnnData object with velocity results
        velocity_2d: (n_peaks, 2) velocity vectors in 2D
        umap_coords: (n_peaks, 2) UMAP coordinates
        figsize: Default figure size
    """

    def __init__(
        self,
        adata: sc.AnnData,
        velocity_2d: np.ndarray,
        umap_coords: np.ndarray,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Initialize visualizer.

        Args:
            adata: AnnData with peak information
            velocity_2d: (n_peaks, 2) velocity vectors
            umap_coords: (n_peaks, 2) UMAP coordinates
            figsize: Default figure size
        """
        self.adata = adata

        # Ensure arrays are NumPy (convert from GPU if needed)
        self.velocity_2d = self._to_numpy(velocity_2d)
        self.umap_coords = self._to_numpy(umap_coords)
        self.figsize = figsize

        # Precompute velocity magnitude
        self.velocity_magnitude = np.linalg.norm(self.velocity_2d, axis=1)

    @staticmethod
    def _to_numpy(arr):
        """Convert array to NumPy (from GPU if needed)."""
        try:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        except ImportError:
            pass
        return np.asarray(arr)

    def plot_streamplot(
        self,
        color_by: str = 'peak_type',
        velocity_scale: float = 0.1,
        density: float = 1.0,
        linewidth: float = 1.0,
        arrowsize: float = 1.0,
        cmap: str = 'viridis',
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ) -> plt.Axes:
        """
        Create streamplot showing velocity flow.

        Args:
            color_by: Column in adata.obs to color points by
            velocity_scale: Scale factor for velocity vectors (default 0.1)
            density: Streamline density (higher = more lines)
            linewidth: Width of streamlines
            arrowsize: Size of arrows on streamlines
            cmap: Colormap for points
            ax: Matplotlib axes (if None, creates new)
            title: Plot title

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Scale velocity vectors
        velocity_scaled = self.velocity_2d * velocity_scale

        # Create regular grid for streamplot
        x = self.umap_coords[:, 0]
        y = self.umap_coords[:, 1]
        u = velocity_scaled[:, 0]
        v = velocity_scaled[:, 1]

        # Create interpolation grid
        grid_resolution = 50
        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate velocity onto grid
        from scipy.interpolate import griddata
        Ui = griddata((x, y), u, (Xi, Yi), method='linear', fill_value=0)
        Vi = griddata((x, y), v, (Xi, Yi), method='linear', fill_value=0)

        # Create streamplot
        ax.streamplot(
            Xi, Yi, Ui, Vi,
            density=density,
            linewidth=linewidth,
            arrowsize=arrowsize,
            color='gray'
        )

        # Scatter plot colored by metadata
        if color_by in self.adata.obs.columns:
            if self.adata.obs[color_by].dtype.name == 'category':
                # Categorical coloring
                categories = self.adata.obs[color_by].cat.categories
                colors = plt.cm.get_cmap(cmap, len(categories))

                for i, cat in enumerate(categories):
                    mask = self.adata.obs[color_by] == cat
                    ax.scatter(
                        x[mask], y[mask],
                        c=[colors(i)],
                        label=cat,
                        s=5,
                        alpha=0.6
                    )
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                # Continuous coloring
                scatter = ax.scatter(
                    x, y,
                    c=self.adata.obs[color_by],
                    cmap=cmap,
                    s=5,
                    alpha=0.6
                )
                plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            # Default: color by velocity magnitude
            scatter = ax.scatter(
                x, y,
                c=self.velocity_magnitude,
                cmap=cmap,
                s=5,
                alpha=0.6
            )
            plt.colorbar(scatter, ax=ax, label='Velocity magnitude')

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(title or f'Chromatin Velocity Streamplot (colored by {color_by})')

        return ax

    def plot_arrows(
        self,
        color_by: str = 'peak_type',
        velocity_scale: float = 0.1,
        subsample: Optional[int] = None,
        min_velocity: float = 0.0,
        arrow_width: float = 0.003,
        cmap: str = 'viridis',
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ) -> plt.Axes:
        """
        Create arrow/quiver plot showing velocity vectors.

        Args:
            color_by: Column in adata.obs to color points by
            velocity_scale: Scale factor for velocity vectors
            subsample: Number of peaks to subsample (None = all)
            min_velocity: Minimum velocity magnitude to show
            arrow_width: Width of arrow shafts
            cmap: Colormap
            ax: Matplotlib axes
            title: Plot title

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Subsample if requested
        if subsample is not None and subsample < len(self.umap_coords):
            # Stratified sampling by velocity magnitude
            high_vel_mask = self.velocity_magnitude > np.percentile(self.velocity_magnitude, 75)
            n_high = min(subsample // 2, high_vel_mask.sum())
            n_low = subsample - n_high

            high_vel_idx = np.random.choice(
                np.where(high_vel_mask)[0],
                size=n_high,
                replace=False
            )
            low_vel_idx = np.random.choice(
                np.where(~high_vel_mask)[0],
                size=n_low,
                replace=False
            )
            sample_idx = np.concatenate([high_vel_idx, low_vel_idx])
        else:
            sample_idx = np.arange(len(self.umap_coords))

        # Filter by minimum velocity
        vel_mask = self.velocity_magnitude[sample_idx] >= min_velocity
        sample_idx = sample_idx[vel_mask]

        # Get data
        x = self.umap_coords[sample_idx, 0]
        y = self.umap_coords[sample_idx, 1]
        u = self.velocity_2d[sample_idx, 0] * velocity_scale
        v = self.velocity_2d[sample_idx, 1] * velocity_scale

        # Background scatter plot
        if color_by in self.adata.obs.columns:
            if self.adata.obs[color_by].dtype.name == 'category':
                categories = self.adata.obs[color_by].cat.categories
                colors_map = plt.cm.get_cmap(cmap, len(categories))

                for i, cat in enumerate(categories):
                    mask = self.adata.obs[color_by] == cat
                    ax.scatter(
                        self.umap_coords[mask, 0],
                        self.umap_coords[mask, 1],
                        c=[colors_map(i)],
                        label=cat,
                        s=3,
                        alpha=0.3
                    )
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                scatter = ax.scatter(
                    self.umap_coords[:, 0],
                    self.umap_coords[:, 1],
                    c=self.adata.obs[color_by],
                    cmap=cmap,
                    s=3,
                    alpha=0.3
                )
                plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(
                self.umap_coords[:, 0],
                self.umap_coords[:, 1],
                c='lightgray',
                s=3,
                alpha=0.3
            )

        # Quiver plot
        ax.quiver(
            x, y, u, v,
            angles='xy',
            scale_units='xy',
            scale=1,
            width=arrow_width,
            headwidth=3,
            headlength=5,
            color='red',
            alpha=0.7
        )

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(title or f'Chromatin Velocity Arrows (n={len(sample_idx)}, colored by {color_by})')

        return ax

    def plot_velocity_magnitude_distribution(
        self,
        groupby: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot distribution of velocity magnitudes.

        Args:
            groupby: Column in adata.obs to group by (e.g., 'peak_type')
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if groupby is not None and groupby in self.adata.obs.columns:
            # Grouped violin plot
            data = []
            labels = []
            for cat in self.adata.obs[groupby].cat.categories:
                mask = self.adata.obs[groupby] == cat
                data.append(self.velocity_magnitude[mask])
                labels.append(cat)

            parts = ax.violinplot(data, positions=range(len(labels)), showmeans=True)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Velocity magnitude')
            ax.set_title(f'Velocity Magnitude Distribution by {groupby}')
        else:
            # Simple histogram
            ax.hist(self.velocity_magnitude, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(
                self.velocity_magnitude.mean(),
                color='red',
                linestyle='--',
                label=f'Mean: {self.velocity_magnitude.mean():.4f}'
            )
            ax.axvline(
                np.median(self.velocity_magnitude),
                color='blue',
                linestyle='--',
                label=f'Median: {np.median(self.velocity_magnitude):.4f}'
            )
            ax.set_xlabel('Velocity magnitude')
            ax.set_ylabel('Count')
            ax.set_title('Velocity Magnitude Distribution')
            ax.legend()

        return ax

    def plot_velocity_comparison(
        self,
        temporal_velocity_2d: np.ndarray,
        regularized_velocity_2d: np.ndarray,
        subsample: int = 5000
    ) -> plt.Figure:
        """
        Compare temporal vs regularized velocity.

        Args:
            temporal_velocity_2d: (n_peaks, 2) temporal velocity vectors
            regularized_velocity_2d: (n_peaks, 2) regularized velocity vectors
            subsample: Number of peaks to plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Subsample
        if subsample < len(self.umap_coords):
            sample_idx = np.random.choice(len(self.umap_coords), size=subsample, replace=False)
        else:
            sample_idx = np.arange(len(self.umap_coords))

        x = self.umap_coords[sample_idx, 0]
        y = self.umap_coords[sample_idx, 1]

        # Plot 1: Temporal velocity
        temp_u = temporal_velocity_2d[sample_idx, 0] * 0.1
        temp_v = temporal_velocity_2d[sample_idx, 1] * 0.1
        axes[0].scatter(x, y, c='lightgray', s=3, alpha=0.5)
        axes[0].quiver(x, y, temp_u, temp_v, angles='xy', scale_units='xy', scale=1,
                       width=0.003, color='red', alpha=0.7)
        axes[0].set_title('Temporal Velocity')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')

        # Plot 2: Regularized velocity
        reg_u = regularized_velocity_2d[sample_idx, 0] * 0.1
        reg_v = regularized_velocity_2d[sample_idx, 1] * 0.1
        axes[1].scatter(x, y, c='lightgray', s=3, alpha=0.5)
        axes[1].quiver(x, y, reg_u, reg_v, angles='xy', scale_units='xy', scale=1,
                       width=0.003, color='blue', alpha=0.7)
        axes[1].set_title('Regularized Velocity (Co-accessibility)')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')

        # Plot 3: Magnitude comparison
        temp_mag = np.linalg.norm(temporal_velocity_2d[sample_idx], axis=1)
        reg_mag = np.linalg.norm(regularized_velocity_2d[sample_idx], axis=1)

        axes[2].scatter(temp_mag, reg_mag, s=5, alpha=0.5)
        axes[2].plot([0, max(temp_mag.max(), reg_mag.max())],
                     [0, max(temp_mag.max(), reg_mag.max())],
                     'r--', alpha=0.5, label='y=x')
        axes[2].set_xlabel('Temporal Velocity Magnitude')
        axes[2].set_ylabel('Regularized Velocity Magnitude')
        axes[2].set_title('Magnitude Comparison')
        axes[2].legend()

        correlation = np.corrcoef(temp_mag, reg_mag)[0, 1]
        axes[2].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                     transform=axes[2].transAxes, verticalalignment='top')

        plt.tight_layout()
        return fig

    def plot_comprehensive_summary(
        self,
        color_by: str = 'peak_type',
        velocity_scale: float = 0.1,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive 4-panel summary figure.

        Args:
            color_by: Column to color by
            velocity_scale: Velocity scaling factor
            output_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Panel 1: Streamplot
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_streamplot(
            color_by=color_by,
            velocity_scale=velocity_scale,
            density=1.5,
            ax=ax1,
            title=f'A. Velocity Streamplot (colored by {color_by})'
        )

        # Panel 2: Arrows
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_arrows(
            color_by=color_by,
            velocity_scale=velocity_scale,
            subsample=3000,
            min_velocity=0.1,
            ax=ax2,
            title=f'B. Velocity Arrows (n=3000, colored by {color_by})'
        )

        # Panel 3: Magnitude distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_velocity_magnitude_distribution(
            groupby=color_by if color_by in self.adata.obs.columns else None,
            ax=ax3
        )
        ax3.set_title('C. Velocity Magnitude Distribution')

        # Panel 4: Velocity colored by magnitude
        ax4 = fig.add_subplot(gs[1, 1])
        scatter = ax4.scatter(
            self.umap_coords[:, 0],
            self.umap_coords[:, 1],
            c=self.velocity_magnitude,
            cmap='viridis',
            s=5,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax4, label='Velocity magnitude')
        ax4.set_xlabel('UMAP 1')
        ax4.set_ylabel('UMAP 2')
        ax4.set_title('D. UMAP Colored by Velocity Magnitude')

        # Add summary statistics
        stats_text = (
            f"Summary Statistics:\n"
            f"  Total peaks: {len(self.velocity_magnitude):,}\n"
            f"  Mean velocity: {self.velocity_magnitude.mean():.4f}\n"
            f"  Median velocity: {np.median(self.velocity_magnitude):.4f}\n"
            f"  Std velocity: {self.velocity_magnitude.std():.4f}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_path}")

        return fig
