"""
Chromatin Velocity Core Computation Module

This module implements a hybrid approach for computing chromatin velocity:
1. Temporal tracking: Track the SAME peaks across timepoints
2. Co-accessibility regularization: Use regulatory context for smoothing
3. Optional GPU acceleration: CuPy/cuML for 10-50x speedup on large datasets

Key differences from old approach:
- OLD: Averaged across DIFFERENT peak sets at each timepoint (global trends only)
- NEW: Track SAME peaks over time (capture local, context-dependent dynamics)
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, Dict
import warnings
from tqdm import tqdm
import time


class ChromatinVelocityComputer:
    """
    Compute chromatin velocity using temporal derivatives and co-accessibility.

    Attributes:
        adata: AnnData object with peaks × pseudobulks
        coaccess_df: DataFrame with co-accessibility connections (from Cicero)
        timepoints: List of timepoint values (e.g., [0, 5, 10, 15, 20, 30])
        alpha: Weight for temporal vs co-accessibility regularization (0-1)
        use_gpu: Whether to use GPU acceleration (CuPy/cuML)
        xp: Array backend (cupy or numpy)
    """

    def __init__(
        self,
        adata: sc.AnnData,
        coaccess_df: Optional[pd.DataFrame] = None,
        alpha: float = 0.7,
        use_gpu: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the chromatin velocity computer.

        Args:
            adata: AnnData with peaks × pseudobulks, must have 'timepoint' in var
            coaccess_df: DataFrame with columns ['Peak1', 'Peak2', 'coaccess']
            alpha: Weight for temporal velocity (vs co-accessibility), default 0.7
            use_gpu: Use GPU acceleration if available (default False)
            verbose: Print progress messages (default True)
        """
        self.adata = adata
        self.coaccess_df = coaccess_df
        self.alpha = alpha
        self.verbose = verbose

        # GPU setup
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.xp = self._get_array_module()

        if self.verbose:
            if self.use_gpu:
                import cupy as cp
                print(f"✓ GPU acceleration enabled (CuPy {cp.__version__})")
                try:
                    device = cp.cuda.Device()
                    print(f"  Device ID: {device.id}")
                    mem_info = device.mem_info
                    print(f"  Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
                except Exception as e:
                    print(f"  GPU info: Available")
            else:
                if use_gpu:
                    print("⚠ GPU requested but not available, using CPU")
                else:
                    print("✓ Using CPU (NumPy)")

        # Extract timepoint information from pseudobulk names
        self.pseudobulk_timepoints = self._extract_pseudobulk_timepoints()
        self.unique_timepoints = np.sort(np.unique(self.pseudobulk_timepoints))

        # Storage for results
        self.temporal_velocity = None
        self.regularized_velocity = None
        self.velocity_2d = None

        # Timing statistics
        self.timing_stats = {}

    def _check_gpu_available(self) -> bool:
        """Check if GPU (CuPy) is available."""
        try:
            import cupy as cp
            # Test GPU accessibility
            _ = cp.array([1, 2, 3])
            return True
        except (ImportError, Exception) as e:
            if self.verbose:
                print(f"GPU check failed: {e}")
            return False

    def _get_array_module(self):
        """Get array module (cupy or numpy) based on GPU availability."""
        if self.use_gpu:
            import cupy as cp
            return cp
        else:
            return np

    def _to_numpy(self, arr):
        """Convert array to NumPy (from GPU if needed)."""
        if self.use_gpu:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        return np.asarray(arr)

    def _to_backend(self, arr):
        """Convert array to current backend (GPU or CPU)."""
        if self.use_gpu:
            import cupy as cp
            return cp.asarray(arr)
        else:
            return np.asarray(arr)

    def _extract_pseudobulk_timepoints(self) -> np.ndarray:
        """
        Extract timepoint values from pseudobulk names.

        Pseudobulk format: '{celltype_id}_{timepoint}' (e.g., '0_0somites')
        Returns array of timepoint integers (e.g., [0, 5, 10, ...])
        """
        timepoints = []
        for pb_name in self.adata.var_names:
            # Extract timepoint from name (e.g., '0_5somites' -> 5)
            tp_str = pb_name.split('_')[1].replace('somites', '')
            timepoints.append(int(tp_str))
        return np.array(timepoints)

    def compute_temporal_velocity(
        self,
        smoothing_factor: float = 0.5,
        subset_peaks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute temporal velocity by tracking same peaks across timepoints.

        For each peak:
        1. Extract accessibility trajectory across timepoints
        2. Fit smooth spline (handles 6 sparse timepoints)
        3. Compute derivative at each timepoint

        Args:
            smoothing_factor: Spline smoothing parameter (0-1), higher = smoother
            subset_peaks: Optional boolean mask for peak subset

        Returns:
            velocity_matrix: (n_peaks, n_pseudobulks) velocity values
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Computing temporal velocity for {self.adata.n_obs} peaks...")
            print(f"{'='*60}")

        if subset_peaks is None:
            subset_peaks = np.ones(self.adata.n_obs, dtype=bool)

        n_peaks = subset_peaks.sum()
        n_pseudobulks = self.adata.n_vars
        velocity_matrix = np.zeros((n_peaks, n_pseudobulks))

        # Get accessibility matrix (peaks × pseudobulks)
        if sparse.issparse(self.adata.X):
            X = self.adata.X.toarray()
        else:
            X = self.adata.X

        X_subset = X[subset_peaks, :]

        # Group pseudobulks by celltype (to track same celltype over time)
        celltype_groups = self._group_pseudobulks_by_celltype()

        if self.verbose:
            print(f"Processing {n_peaks} peaks across {len(celltype_groups)} celltypes...")

        # Compute velocity for each peak with progress bar
        iterator = tqdm(range(X_subset.shape[0]), desc="Computing velocity", disable=not self.verbose)

        for i in iterator:
            # For each celltype group, fit spline and compute derivative
            for celltype, pb_indices in celltype_groups.items():
                tp_values = self.pseudobulk_timepoints[pb_indices]
                acc_values = X_subset[i, pb_indices]

                # Skip if not enough timepoints
                if len(np.unique(tp_values)) < 3:
                    continue

                # Fit spline (CPU only - scipy doesn't have GPU version)
                try:
                    spline = UnivariateSpline(
                        tp_values,
                        acc_values,
                        s=smoothing_factor,
                        k=min(3, len(np.unique(tp_values)) - 1)
                    )

                    # Compute derivatives at each timepoint
                    derivatives = spline.derivative()(tp_values)
                    velocity_matrix[i, pb_indices] = derivatives

                except Exception as e:
                    # If spline fitting fails, use simple finite differences
                    velocity_matrix[i, pb_indices] = self._finite_difference(
                        tp_values, acc_values
                    )

        elapsed = time.time() - start_time
        self.timing_stats['temporal_velocity'] = elapsed

        if self.verbose:
            print(f"\n✓ Temporal velocity computed in {elapsed:.2f}s ({elapsed/n_peaks*1000:.2f} ms/peak)")
            print(f"  Mean magnitude: {np.abs(velocity_matrix).mean():.4f}")
            print(f"  Std magnitude: {np.abs(velocity_matrix).std():.4f}")

        self.temporal_velocity = velocity_matrix
        return velocity_matrix

    def _group_pseudobulks_by_celltype(self) -> Dict[str, np.ndarray]:
        """
        Group pseudobulk indices by celltype.

        Returns:
            dict mapping celltype_id -> array of pseudobulk indices
        """
        celltype_groups = {}
        for i, pb_name in enumerate(self.adata.var_names):
            celltype = pb_name.split('_')[0]
            if celltype not in celltype_groups:
                celltype_groups[celltype] = []
            celltype_groups[celltype].append(i)

        # Convert to numpy arrays
        for celltype in celltype_groups:
            celltype_groups[celltype] = np.array(celltype_groups[celltype])

        return celltype_groups

    def _finite_difference(self, timepoints: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Compute finite difference derivative as fallback.

        Args:
            timepoints: Array of timepoint values
            values: Array of accessibility values

        Returns:
            derivatives: Array of derivative values
        """
        # Sort by timepoint
        sort_idx = np.argsort(timepoints)
        tp_sorted = timepoints[sort_idx]
        val_sorted = values[sort_idx]

        derivatives = np.zeros_like(values)

        for i in range(len(tp_sorted)):
            if i == 0:
                # Forward difference
                derivatives[i] = (val_sorted[i+1] - val_sorted[i]) / (tp_sorted[i+1] - tp_sorted[i])
            elif i == len(tp_sorted) - 1:
                # Backward difference
                derivatives[i] = (val_sorted[i] - val_sorted[i-1]) / (tp_sorted[i] - tp_sorted[i-1])
            else:
                # Central difference
                derivatives[i] = (val_sorted[i+1] - val_sorted[i-1]) / (tp_sorted[i+1] - tp_sorted[i-1])

        # Unsort to match original order
        unsort_idx = np.argsort(sort_idx)
        return derivatives[unsort_idx]

    def compute_coaccessibility_regularization(
        self,
        min_coaccess_score: float = 0.5,
        max_connections: int = 100
    ) -> np.ndarray:
        """
        Regularize velocity using co-accessibility network.

        For each peak:
        1. Find co-accessible peaks (above threshold)
        2. Compute weighted average of their velocities
        3. Combine with temporal velocity: α*temporal + (1-α)*coaccess

        Args:
            min_coaccess_score: Minimum co-accessibility score threshold
            max_connections: Maximum number of connections per peak

        Returns:
            regularized_velocity: (n_peaks, n_pseudobulks) regularized velocities
        """
        if self.temporal_velocity is None:
            raise ValueError("Must compute temporal velocity first!")

        if self.coaccess_df is None:
            print("No co-accessibility data provided, skipping regularization")
            self.regularized_velocity = self.temporal_velocity.copy()
            return self.regularized_velocity

        print(f"Regularizing velocity using co-accessibility (threshold={min_coaccess_score})...")

        # Filter co-accessibility connections
        coaccess_filtered = self.coaccess_df[
            self.coaccess_df['coaccess'] >= min_coaccess_score
        ].copy()

        print(f"  Using {len(coaccess_filtered)} connections (from {len(self.coaccess_df)} total)")

        # Create peak name to index mapping
        peak_to_idx = {peak: i for i, peak in enumerate(self.adata.obs_names)}

        # Build co-accessibility graph
        coaccess_graph = {}
        for _, row in coaccess_filtered.iterrows():
            peak1, peak2, score = row['Peak1'], row['Peak2'], row['coaccess']

            if peak1 not in peak_to_idx or peak2 not in peak_to_idx:
                continue

            idx1, idx2 = peak_to_idx[peak1], peak_to_idx[peak2]

            if idx1 not in coaccess_graph:
                coaccess_graph[idx1] = []
            if idx2 not in coaccess_graph:
                coaccess_graph[idx2] = []

            coaccess_graph[idx1].append((idx2, score))
            coaccess_graph[idx2].append((idx1, score))

        # Regularize velocity for each peak
        regularized_velocity = np.zeros_like(self.temporal_velocity)

        for peak_idx in range(self.temporal_velocity.shape[0]):
            if (peak_idx + 1) % 10000 == 0:
                print(f"  Processed {peak_idx+1}/{self.temporal_velocity.shape[0]} peaks...")

            if peak_idx not in coaccess_graph:
                # No co-accessible peaks, use temporal velocity only
                regularized_velocity[peak_idx] = self.temporal_velocity[peak_idx]
                continue

            # Get co-accessible neighbors
            neighbors = coaccess_graph[peak_idx][:max_connections]
            neighbor_indices = [n[0] for n in neighbors]
            neighbor_weights = np.array([n[1] for n in neighbors])
            neighbor_weights /= neighbor_weights.sum()  # Normalize

            # Weighted average of neighbor velocities
            neighbor_velocity = np.zeros(self.temporal_velocity.shape[1])
            for n_idx, weight in zip(neighbor_indices, neighbor_weights):
                neighbor_velocity += weight * self.temporal_velocity[n_idx]

            # Combine temporal and co-accessibility
            regularized_velocity[peak_idx] = (
                self.alpha * self.temporal_velocity[peak_idx] +
                (1 - self.alpha) * neighbor_velocity
            )

        print(f"Regularization complete. Mean magnitude: {np.abs(regularized_velocity).mean():.4f}")
        self.regularized_velocity = regularized_velocity
        return regularized_velocity

    def project_to_2d(
        self,
        use_umap_coords: bool = True,
        velocity_key: str = 'regularized'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project velocity vectors to 2D UMAP space.

        Args:
            use_umap_coords: If True, use existing UMAP coordinates from adata.obsm
            velocity_key: Which velocity to project ('temporal' or 'regularized')

        Returns:
            velocity_2d: (n_peaks, 2) velocity vectors in 2D
            umap_coords: (n_peaks, 2) UMAP coordinates
        """
        if velocity_key == 'temporal' and self.temporal_velocity is None:
            raise ValueError("Temporal velocity not computed yet!")
        if velocity_key == 'regularized' and self.regularized_velocity is None:
            raise ValueError("Regularized velocity not computed yet!")

        velocity = self.temporal_velocity if velocity_key == 'temporal' else self.regularized_velocity

        print(f"Projecting {velocity_key} velocity to 2D...")

        # Get UMAP coordinates
        if use_umap_coords and 'X_umap' in self.adata.obsm:
            umap_coords = self.adata.obsm['X_umap']
            print(f"  Using existing UMAP coordinates")
        else:
            print(f"  Computing UMAP coordinates...")
            sc.pp.neighbors(self.adata, n_neighbors=15, use_rep='X')
            sc.tl.umap(self.adata)
            umap_coords = self.adata.obsm['X_umap']

        # Project velocity using local PCA
        # For each peak, find its neighbors in high-D space and compute local transformation
        velocity_2d = self._local_velocity_projection(velocity, umap_coords)

        self.velocity_2d = velocity_2d

        print(f"2D projection complete. Mean magnitude: {np.linalg.norm(velocity_2d, axis=1).mean():.4f}")
        return velocity_2d, umap_coords

    def _local_velocity_projection(
        self,
        velocity_hd: np.ndarray,
        coords_2d: np.ndarray,
        n_neighbors: int = 30
    ) -> np.ndarray:
        """
        Project high-dimensional velocity to 2D using local linear transformation.
        GPU-accelerated for K-NN search and distance computation.

        For each peak:
        1. Find its k-nearest neighbors in 2D space (GPU accelerated)
        2. Compute local linear map from high-D to 2D
        3. Apply transformation to velocity vector

        Args:
            velocity_hd: (n_peaks, n_pseudobulks) high-dimensional velocity
            coords_2d: (n_peaks, 2) 2D coordinates
            n_neighbors: Number of neighbors for local transformation

        Returns:
            velocity_2d: (n_peaks, 2) projected velocity vectors
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Projecting velocity to 2D (n_neighbors={n_neighbors})...")
            print(f"{'='*60}")

        n_peaks = velocity_hd.shape[0]
        velocity_2d = np.zeros((n_peaks, 2))

        # GPU-accelerated K-NN search if available
        if self.use_gpu:
            try:
                from cuml.neighbors import NearestNeighbors as cuNN
                import cupy as cp

                if self.verbose:
                    print(f"Using GPU-accelerated K-NN search (cuML)...")

                # Transfer data to GPU
                coords_2d_gpu = cp.asarray(coords_2d)

                # Fit K-NN model
                knn = cuNN(n_neighbors=n_neighbors, metric='euclidean')
                knn.fit(coords_2d_gpu)

                # Find neighbors for all peaks at once
                distances_gpu, indices_gpu = knn.kneighbors(coords_2d_gpu)

                # Transfer back to CPU for least squares (sklearn PCA is CPU-only)
                neighbor_indices = cp.asnumpy(indices_gpu)

                del coords_2d_gpu, distances_gpu, indices_gpu
                cp.get_default_memory_pool().free_all_blocks()

            except ImportError:
                if self.verbose:
                    print(f"cuML not available, using CPU K-NN search...")
                self.use_gpu = False  # Fall back to CPU
                neighbor_indices = self._compute_knn_cpu(coords_2d, n_neighbors)
        else:
            # CPU K-NN search
            if self.verbose:
                print(f"Using CPU K-NN search (scipy)...")
            neighbor_indices = self._compute_knn_cpu(coords_2d, n_neighbors)

        # Now project velocities using PCA and least squares (CPU)
        if self.use_gpu:
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                from sklearn.decomposition import PCA

        from sklearn.decomposition import PCA

        if self.verbose:
            print(f"Computing local projections for {n_peaks} peaks...")

        iterator = tqdm(range(n_peaks), desc="Projecting to 2D", disable=not self.verbose)

        for i in iterator:
            # Get neighbors for this peak
            neighbor_idx = neighbor_indices[i]

            # Get velocity vectors of neighbors in high-D
            neighbor_velocities = velocity_hd[neighbor_idx]

            # Get 2D displacement vectors from current peak to neighbors
            neighbor_displacements = coords_2d[neighbor_idx] - coords_2d[i]

            # Fit local linear transformation: high-D velocity -> 2D displacement
            # Use PCA to reduce dimensionality first
            if neighbor_velocities.shape[1] > 10:
                pca = PCA(n_components=min(10, n_neighbors - 1))
                neighbor_velocities_reduced = pca.fit_transform(neighbor_velocities)
                current_velocity_reduced = pca.transform(velocity_hd[i:i+1])
            else:
                neighbor_velocities_reduced = neighbor_velocities
                current_velocity_reduced = velocity_hd[i:i+1]

            # Solve least squares: neighbor_velocities_reduced @ W = neighbor_displacements
            try:
                W, residuals, rank, s = np.linalg.lstsq(
                    neighbor_velocities_reduced,
                    neighbor_displacements,
                    rcond=None
                )

                # Apply transformation to current peak's velocity
                velocity_2d[i] = current_velocity_reduced @ W
            except np.linalg.LinAlgError:
                # If singular, just use mean direction of neighbors
                velocity_2d[i] = neighbor_displacements.mean(axis=0)

        elapsed = time.time() - start_time
        self.timing_stats['projection_2d'] = elapsed

        if self.verbose:
            print(f"\n✓ 2D projection complete in {elapsed:.2f}s ({elapsed/n_peaks*1000:.2f} ms/peak)")
            print(f"  Mean magnitude: {np.linalg.norm(velocity_2d, axis=1).mean():.4f}")

        return velocity_2d

    def _compute_knn_cpu(self, coords_2d: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Compute K-NN on CPU using sklearn."""
        from sklearn.neighbors import NearestNeighbors

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
        knn.fit(coords_2d)
        distances, indices = knn.kneighbors(coords_2d)

        return indices

    def save_results(self, output_path: str):
        """
        Save computed velocity results to AnnData file.

        Args:
            output_path: Path to save .h5ad file
        """
        # Add velocity results to adata
        if self.temporal_velocity is not None:
            self.adata.layers['temporal_velocity'] = self.temporal_velocity

        if self.regularized_velocity is not None:
            self.adata.layers['regularized_velocity'] = self.regularized_velocity

        if self.velocity_2d is not None:
            self.adata.obsm['velocity_umap'] = self.velocity_2d

        # Save parameters and timing stats
        self.adata.uns['velocity_params'] = {
            'alpha': self.alpha,
            'n_timepoints': len(self.unique_timepoints),
            'timepoints': list(self.unique_timepoints),
            'use_gpu': self.use_gpu,
            'timing_stats': self.timing_stats
        }

        if self.verbose:
            print(f"\nSaving results to {output_path}...")
        self.adata.write(output_path)
        if self.verbose:
            print("✓ Done!")

    def print_timing_summary(self):
        """Print timing statistics summary."""
        if not self.timing_stats:
            print("No timing statistics available.")
            return

        print(f"\n{'='*60}")
        print("TIMING SUMMARY")
        print(f"{'='*60}")

        total_time = sum(self.timing_stats.values())

        for step, elapsed in self.timing_stats.items():
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"  {step:30s}: {elapsed:8.2f}s ({percentage:5.1f}%)")

        print(f"  {'='*40}")
        print(f"  {'TOTAL':30s}: {total_time:8.2f}s")
        print(f"{'='*60}")
