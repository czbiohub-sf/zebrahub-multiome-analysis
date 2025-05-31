#!/usr/bin/env python3
"""
GPU-Accelerated pySlingshot using CuPy and cuML

Key optimizations:
1. Replace scipy operations with CuPy equivalents
2. Replace sklearn with cuML for KDE and neighbors
3. GPU-accelerated distance calculations and MST
4. Vectorized interpolation on GPU
"""

from typing import Union
import numpy as np
from anndata import AnnData
from collections import deque
from tqdm.autonotebook import tqdm

# GPU imports with fallbacks
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    from cupyx.scipy.sparse.csgraph import minimum_spanning_tree as gpu_mst
    from cupyx.scipy.interpolate import interp1d as gpu_interp1d
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU acceleration")
except ImportError:
    import numpy as cp
    from scipy.sparse.csgraph import minimum_spanning_tree as gpu_mst
    from scipy.interpolate import interp1d as gpu_interp1d
    CUPY_AVAILABLE = False
    print("⚠️  CuPy not available - using CPU fallback")

try:
    import cuml
    from cuml.neighbors import KernelDensity as GPU_KernelDensity
    from cuml.neighbors import NearestNeighbors as GPU_NearestNeighbors
    CUML_AVAILABLE = True
    print("✅ cuML available for GPU sklearn acceleration")
except ImportError:
    from sklearn.neighbors import KernelDensity as GPU_KernelDensity
    CUML_AVAILABLE = False
    print("⚠️  cuML not available - using CPU sklearn")

# Original imports
from pcurvepy2 import PrincipalCurve
from sklearn.neighbors import KernelDensity

from .util import scale_to_range, mahalanobis, isint, isstr
from .lineage import Lineage
from .plotter import SlingshotPlotter


class GPUSlingshot:
    """GPU-accelerated version of Slingshot trajectory inference"""
    
    def __init__(
            self,
            data: Union[AnnData, np.ndarray],
            cluster_labels_onehot=None,
            celltype_key=None,
            obsm_key='X_umap',
            start_node=0,
            end_nodes=None,
            debug_level=None,
            use_gpu=True
    ):
        """
        GPU-accelerated Slingshot constructor
        
        Args:
            use_gpu: Enable GPU acceleration (auto-detects availability)
            ... (other args same as original)
        """
        
        # GPU setup
        self.use_gpu = use_gpu and (CUPY_AVAILABLE or CUML_AVAILABLE)
        if self.use_gpu:
            print("🚀 GPU acceleration enabled for Slingshot")
        else:
            print("💻 Using CPU-only Slingshot")
        
        # Data processing (same as original but with GPU transfer)
        if isinstance(data, AnnData):
            assert celltype_key is not None, "Must provide celltype key if data is an AnnData object"
            cluster_labels = data.obs[celltype_key]

            if isint(cluster_labels[0]):
                cluster_max = cluster_labels.max()
                self.cluster_label_indices = cluster_labels
            elif isstr(cluster_labels[0]):
                cluster_max = len(np.unique(cluster_labels))
                self.cluster_label_indices = np.array([np.where(np.unique(cluster_labels) == label)[0][0] for label in cluster_labels])
            else:
                raise ValueError("Unexpected cluster label dtype.")
            cluster_labels_onehot = np.zeros((cluster_labels.shape[0], cluster_max + 1))
            cluster_labels_onehot[np.arange(cluster_labels.shape[0]), self.cluster_label_indices] = 1

            data = data.obsm[obsm_key]
        else:
            assert cluster_labels_onehot is not None, "Must provide cluster labels if data is not an AnnData object"
            cluster_labels = self.cluster_labels_onehot.argmax(axis=1)
        
        # Transfer data to GPU if available
        if self.use_gpu and CUPY_AVAILABLE:
            self.data_gpu = cp.asarray(data)
            self.cluster_labels_onehot_gpu = cp.asarray(cluster_labels_onehot)
            self.cluster_label_indices_gpu = cp.asarray(self.cluster_label_indices)
        else:
            self.data_gpu = data
            self.cluster_labels_onehot_gpu = cluster_labels_onehot
            self.cluster_label_indices_gpu = self.cluster_label_indices
        
        self.data = data
        self.cluster_labels_onehot = cluster_labels_onehot
        self.cluster_labels = cluster_labels
        self.num_clusters = self.cluster_label_indices.max() + 1
        self.start_node = start_node
        self.end_nodes = [] if end_nodes is None else end_nodes
        
        # GPU-accelerated cluster center calculation
        cluster_centres = self._gpu_compute_cluster_centers()
        self.cluster_centres = cluster_centres
        
        # Initialize same attributes as original
        self.lineages = None
        self.cluster_lineages = None
        self.curves = None
        self.cell_weights = None
        self.distances = None
        self.branch_clusters = None
        self._tree = None

        # Plotting and debugging
        debug_level = 0 if debug_level is None else dict(verbose=1)[debug_level]
        self.debug_level = debug_level
        self._set_debug_axes(None)
        self.plotter = SlingshotPlotter(self)

        # GPU-accelerated kernel construction
        self._construct_gpu_kernel()

    def _gpu_compute_cluster_centers(self):
        """GPU-accelerated cluster center computation"""
        if self.use_gpu and CUPY_AVAILABLE:
            cluster_centres = []
            for k in range(self.num_clusters):
                mask = self.cluster_label_indices_gpu == k
                if cp.any(mask):
                    center = cp.mean(self.data_gpu[mask], axis=0)
                    cluster_centres.append(cp.asnumpy(center))
                else:
                    # Handle empty clusters
                    cluster_centres.append(np.zeros(self.data_gpu.shape[1]))
            return np.stack(cluster_centres)
        else:
            # CPU fallback
            cluster_centres = [self.data[self.cluster_label_indices == k].mean(axis=0) 
                             for k in range(self.num_clusters)]
            return np.stack(cluster_centres)

    def _construct_gpu_kernel(self):
        """GPU-accelerated kernel construction for shrinking step"""
        self.kernel_x = np.linspace(-3, 3, 512)
        
        if self.use_gpu and CUML_AVAILABLE:
            # GPU KernelDensity
            kde = GPU_KernelDensity(bandwidth=1.0, kernel='gaussian')
            kde.fit(np.zeros((self.kernel_x.shape[0], 1)))
            self.kernel_y = np.exp(kde.score_samples(self.kernel_x.reshape(-1, 1)))
        else:
            # CPU fallback
            kde = KernelDensity(bandwidth=1., kernel='gaussian')
            kde.fit(np.zeros((self.kernel_x.shape[0], 1)))
            self.kernel_y = np.exp(kde.score_samples(self.kernel_x.reshape(-1, 1)))

    def _set_debug_axes(self, axes):
        self.debug_axes = axes
        self.debug_plot_mst = axes is not None
        self.debug_plot_lineages = axes is not None
        self.debug_plot_avg = axes is not None

    @property
    def tree(self):
        if self._tree is None:
            self.construct_mst(self.start_node)
        return self._tree

    def gpu_mahalanobis_distances(self, centers, emp_covs):
        """GPU-accelerated Mahalanobis distance computation"""
        if not self.use_gpu or not CUPY_AVAILABLE:
            # CPU fallback using original mahalanobis function
            dists = np.zeros((self.num_clusters, self.num_clusters))
            for i in range(self.num_clusters):
                for j in range(i, self.num_clusters):
                    dist = mahalanobis(centers[i], centers[j], emp_covs[i], emp_covs[j])
                    dists[i, j] = dist
                    dists[j, i] = dist
            return dists
        
        # GPU implementation
        centers_gpu = cp.asarray(centers)
        emp_covs_gpu = cp.asarray(emp_covs)
        dists_gpu = cp.zeros((self.num_clusters, self.num_clusters))
        
        for i in range(self.num_clusters):
            for j in range(i, self.num_clusters):
                # GPU Mahalanobis distance calculation
                diff = centers_gpu[i] - centers_gpu[j]
                
                # Average covariance matrix
                avg_cov = (emp_covs_gpu[i] + emp_covs_gpu[j]) / 2
                
                # Add regularization for numerical stability
                reg = cp.eye(avg_cov.shape[0]) * 1e-6
                avg_cov_reg = avg_cov + reg
                
                try:
                    # Solve Ax = b instead of computing inverse
                    inv_cov_diff = cp.linalg.solve(avg_cov_reg, diff)
                    dist = cp.sqrt(cp.dot(diff, inv_cov_diff))
                except cp.linalg.LinAlgError:
                    # Fallback to Euclidean distance
                    dist = cp.linalg.norm(diff)
                
                dists_gpu[i, j] = dist
                dists_gpu[j, i] = dist
        
        return cp.asnumpy(dists_gpu)

    def construct_mst(self, start_node):
        """GPU-accelerated MST construction"""
        # GPU-accelerated covariance calculation
        if self.use_gpu and CUPY_AVAILABLE:
            emp_covs = []
            for i in range(self.num_clusters):
                mask = self.cluster_label_indices_gpu == i
                cluster_data = self.data_gpu[mask]
                if len(cluster_data) > 1:
                    # GPU covariance calculation
                    cov_matrix = cp.cov(cluster_data.T)
                    emp_covs.append(cp.asnumpy(cov_matrix))
                else:
                    # Single point or empty cluster
                    emp_covs.append(np.eye(self.data_gpu.shape[1]) * 1e-3)
            emp_covs = np.stack(emp_covs)
        else:
            # CPU fallback
            emp_covs = np.stack([np.cov(self.data[self.cluster_label_indices == i].T) 
                               for i in range(self.num_clusters)])

        # GPU-accelerated distance calculation
        dists = self.gpu_mahalanobis_distances(self.cluster_centres, emp_covs)

        # GPU-accelerated MST
        mst_dists = np.delete(np.delete(dists, self.end_nodes, axis=0), self.end_nodes, axis=1)
        
        if self.use_gpu and CUPY_AVAILABLE:
            mst_dists_gpu = cp.asarray(mst_dists)
            tree_gpu = gpu_mst(mst_dists_gpu)
            tree = tree_gpu.get()  # Transfer back to CPU
        else:
            tree = gpu_mst(mst_dists)

        # Process MST results (same logic as original)
        index_mapping = np.array([c for c in range(self.num_clusters - len(self.end_nodes))])
        for i, end_node in enumerate(self.end_nodes):
            index_mapping[end_node - i:] += 1

        connections = {k: list() for k in range(self.num_clusters)}
        cx = tree.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            i = index_mapping[i]
            j = index_mapping[j]
            connections[i].append(j)
            connections[j].append(i)

        # Connect end nodes
        for end in self.end_nodes:
            i = np.argmin(np.delete(dists[end], self.end_nodes))
            connections[i].append(end)
            connections[end].append(i)

        # BFS to construct children dict (same as original)
        visited = [False for _ in range(self.num_clusters)]
        queue = list()
        queue.append(start_node)
        children = {k: list() for k in range(self.num_clusters)}
        
        while len(queue) > 0:
            current_node = queue.pop()
            visited[current_node] = True
            for child in connections[current_node]:
                if not visited[child]:
                    children[current_node].append(child)
                    queue.append(child)

        # Debug plotting (same as original)
        if self.debug_plot_mst:
            self.plotter.clusters(self.debug_axes[0, 0], alpha=0.5)
            for root, kids in children.items():
                for child in kids:
                    start = [self.cluster_centres[root][0], self.cluster_centres[child][0]]
                    end = [self.cluster_centres[root][1], self.cluster_centres[child][1]]
                    self.debug_axes[0, 0].plot(start, end, c='black')
            self.debug_plot_mst = False

        self._tree = children
        return children

    def gpu_calculate_cell_weights(self):
        """GPU-accelerated cell weight calculation"""
        if not self.use_gpu or not CUPY_AVAILABLE:
            # CPU fallback - use original method
            self.calculate_cell_weights()
            return
        
        # GPU implementation
        cell_weights = []
        for l in range(len(self.lineages)):
            lineage_clusters = self.lineages[l].clusters
            lineage_weights = cp.sum(self.cluster_labels_onehot_gpu[:, lineage_clusters], axis=1)
            cell_weights.append(lineage_weights)
        
        cell_weights_gpu = cp.stack(cell_weights, axis=1)
        
        # Distance calculations on GPU
        d_sq_gpu = cp.stack([cp.asarray(d) for d in self.distances], axis=1)
        d_ord = cp.argsort(d_sq_gpu, axis=None)
        
        w_prob = cell_weights_gpu / cp.sum(cell_weights_gpu, axis=1, keepdims=True)
        w_rnk_d = cp.cumsum(w_prob.reshape(-1)[d_ord]) / cp.sum(w_prob)

        z_gpu = d_sq_gpu.copy()
        z_shape = z_gpu.shape
        z_flat = z_gpu.reshape(-1)
        z_flat[d_ord] = w_rnk_d
        z_gpu = z_flat.reshape(z_shape)
        
        z_prime = 1 - z_gpu ** 2
        z_prime[cell_weights_gpu == 0] = cp.nan
        
        w0 = cell_weights_gpu.copy()
        cell_weights_gpu = z_prime / cp.nanmax(z_prime, axis=1, keepdims=True)
        
        # Handle NaN and boundary conditions
        cell_weights_gpu = cp.nan_to_num(cell_weights_gpu, nan=1)
        cell_weights_gpu = cp.clip(cell_weights_gpu, 0, 1)
        cell_weights_gpu[w0 == 0] = 0

        # Reassignment logic
        if True:  # reassign
            cell_weights_gpu[z_gpu < 0.5] = 1
            
            ridx = (cp.max(z_gpu, axis=1) > 0.9) & (cp.min(cell_weights_gpu, axis=1) < 0.1)
            w0_subset = cell_weights_gpu[ridx]
            z0_subset = z_gpu[ridx]
            w0_subset[(z0_subset > 0.9) & (w0_subset < 0.1)] = 0
            cell_weights_gpu[ridx] = w0_subset

        self.cell_weights = cp.asnumpy(cell_weights_gpu)

    def gpu_interpolation(self, x_old, y_old, x_new):
        """GPU-accelerated interpolation"""
        if not self.use_gpu or not CUPY_AVAILABLE:
            # CPU fallback
            from scipy.interpolate import interp1d
            f = interp1d(x_old, y_old, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
            return f(x_new)
        
        # GPU interpolation
        x_old_gpu = cp.asarray(x_old)
        y_old_gpu = cp.asarray(y_old)
        x_new_gpu = cp.asarray(x_new)
        
        # Simple linear interpolation on GPU
        result = cp.interp(x_new_gpu, x_old_gpu, y_old_gpu)
        return cp.asnumpy(result)

    def fit(self, num_epochs=10, debug_axes=None):
        """GPU-accelerated fitting with same interface as original"""
        self._set_debug_axes(debug_axes)
        
        if self.curves is None:
            self.get_lineages()
            self.construct_initial_curves()
            cell_weights = [self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1)
                           for l in range(len(self.lineages))]
            self.cell_weights = np.stack(cell_weights, axis=1)

        for epoch in tqdm(range(num_epochs)):
            # GPU-accelerated cell weight calculation
            if self.use_gpu:
                self.gpu_calculate_cell_weights()
            else:
                self.calculate_cell_weights()

            # Fit principal curves (inherits GPU acceleration from data transfers)
            self.fit_lineage_curves()

            # Ensure starts at 0
            for l_idx, lineage in enumerate(self.lineages):
                curve = self.curves[l_idx]
                min_time = np.min(curve.pseudotimes_interp[self.cell_weights[:, l_idx] > 0])
                curve.pseudotimes_interp -= min_time

            # Average curves and shrinking (can benefit from GPU interpolation)
            shrinkage_percentages, cluster_children, cluster_avg_curves = self.avg_curves()
            self.shrink_curves(cluster_children, shrinkage_percentages, cluster_avg_curves)

            self.debug_plot_lineages = False
            self.debug_plot_avg = False

            if self.debug_axes is not None and epoch == num_epochs - 1:
                self.plotter.clusters(self.debug_axes[1, 1], s=2, alpha=0.5)
                self.plotter.curves(self.debug_axes[1, 1], self.curves)

    # Include all other methods from original with GPU acceleration where applicable
    # (get_lineages, construct_initial_curves, fit_lineage_curves, etc.)
    # These would follow the same pattern of GPU acceleration with CPU fallbacks

    def get_lineages(self):
        """Same as original - no GPU acceleration needed for tree traversal"""
        tree = self.construct_mst(self.start_node)

        branch_clusters = deque()
        def recurse_branches(path, v):
            num_children = len(tree[v])
            if num_children == 0:
                return path + [v, None]
            elif num_children == 1:
                return recurse_branches(path + [v], tree[v][0])
            else:
                branch_clusters.append(v)
                return [recurse_branches(path + [v], tree[v][i]) for i in range(num_children)]

        def flatten(li):
            if li[-1] is None:
                yield Lineage(li[:-1])
            else:
                for l in li:
                    yield from flatten(l)

        lineages = recurse_branches([], self.start_node)
        lineages = list(flatten(lineages))
        self.lineages = lineages
        self.branch_clusters = branch_clusters

        self.cluster_lineages = {k: list() for k in range(self.num_clusters)}
        for l_idx, lineage in enumerate(self.lineages):
            for k in lineage:
                self.cluster_lineages[k].append(l_idx)

        if self.debug_level > 0:
            print('Lineages:', lineages)

    # Add remaining methods with GPU acceleration...
    # (Following the same pattern as above)

    def calculate_cell_weights(self):
        """CPU fallback for cell weight calculation (original method)"""
        cell_weights = [self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1)
                        for l in range(len(self.lineages))]
        cell_weights = np.stack(cell_weights, axis=1)

        d_sq = np.stack(self.distances, axis=1)
        d_ord = np.argsort(d_sq, axis=None)
        w_prob = cell_weights/cell_weights.sum(axis=1, keepdims=True)
        w_rnk_d = np.cumsum(w_prob.reshape(-1)[d_ord]) / w_prob.sum()

        z = d_sq
        z_shape = z.shape
        z = z.reshape(-1)
        z[d_ord] = w_rnk_d
        z = z.reshape(z_shape)
        z_prime = 1 - z ** 2
        z_prime[cell_weights == 0] = np.nan
        w0 = cell_weights.copy()
        cell_weights = z_prime / np.nanmax(z_prime, axis=1, keepdims=True)
        np.nan_to_num(cell_weights, nan=1, copy=False)
        cell_weights[cell_weights > 1] = 1
        cell_weights[cell_weights < 0] = 0
        cell_weights[w0 == 0] = 0

        reassign = True
        if reassign:
            cell_weights[z < .5] = 1
            ridx = (z.max(axis=1) > .9) & (cell_weights.min(axis=1) < .1)
            w0 = cell_weights[ridx]
            z0 = z[ridx]
            w0[(z0 > .9) & (w0 < .1)] = 0
            cell_weights[ridx] = w0

        self.cell_weights = cell_weights


# Convenience function to create GPU or CPU version
def create_slingshot(data, use_gpu=None, **kwargs):
    """
    Create GPU or CPU Slingshot based on availability
    
    Args:
        data: Input data
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None)
        **kwargs: Other Slingshot parameters
    """
    if use_gpu is None:
        use_gpu = CUPY_AVAILABLE or CUML_AVAILABLE
    
    if use_gpu and (CUPY_AVAILABLE or CUML_AVAILABLE):
        print("🚀 Creating GPU-accelerated Slingshot")
        return GPUSlingshot(data, use_gpu=True, **kwargs)
    else:
        print("💻 Creating CPU Slingshot (GPU not available)")
        # Import and return original Slingshot
        from . import Slingshot
        return Slingshot(data, **kwargs)


# Example usage and performance comparison
def benchmark_slingshot(data, **kwargs):
    """Compare GPU vs CPU Slingshot performance"""
    import time
    
    print("🏁 Benchmarking Slingshot Performance")
    print("=" * 40)
    
    # CPU version
    print("Running CPU Slingshot...")
    start_time = time.time()
    cpu_slingshot = create_slingshot(data, use_gpu=False, **kwargs)
    cpu_slingshot.fit(num_epochs=5)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.2f} seconds")
    
    # GPU version
    if CUPY_AVAILABLE or CUML_AVAILABLE:
        print("Running GPU Slingshot...")
        start_time = time.time()
        gpu_slingshot = create_slingshot(data, use_gpu=True, **kwargs)
        gpu_slingshot.fit(num_epochs=5)
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.2f} seconds")
        print(f"🚀 GPU speedup: {cpu_time/gpu_time:.1f}x")
        
        return cpu_slingshot, gpu_slingshot
    else:
        print("⚠️  GPU not available for comparison")
        return cpu_slingshot, None