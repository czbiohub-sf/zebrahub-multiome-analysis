# Run this script to compute motif enrichment scores using GPU acceleration
# execute this within rapids environment (i.e., rapids_sc)
import numpy as np
import pandas as pd
import cudf
import cupy as cp
# from cupyx.scipy.stats import hypergeom as cupy_hypergeom
from scipy.stats import hypergeom as hypergeom
import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool
import time

class GPUMotifActivityPredictor:
    """Base class for GPU-accelerated motif activity prediction."""
    
    @staticmethod
    def create(method_name, **kwargs):
        """Factory method to create activity predictors."""
        methods = {
            "hypergeom": GPUHypergeomPredictor,
            "rf": GPURFPredictor,
        }
        
        if method_name.lower() not in methods:
            raise ValueError(f"Unknown method: {method_name}. Available methods: {list(methods.keys())}")
        
        return methods[method_name.lower()](**kwargs)
    
    @staticmethod
    def list_predictors():
        """List available predictors."""
        return ["hypergeom", "rf"]


class GPUHypergeomPredictor(GPUMotifActivityPredictor):
    """GPU-accelerated hypergeometric test for motif activity prediction."""
    
    def __init__(self, batch_size=1000, **kwargs):
        self.act_ = None
        self.pref_table = "count"
        self.ptype = "classification"
        self.batch_size = batch_size
        
    def fit(self, df_X, df_y):
        """
        Fit the predictor using hypergeometric test on GPU.
        
        Parameters
        ----------
        df_X : pandas.DataFrame
            Motif counts per region (peaks-by-motifs)
        df_y : pandas.DataFrame
            Cluster labels for each region
        
        Returns
        -------
        self
        """
        print("Starting GPU Hypergeometric test...")
        start_time = time.time()
        
        if not df_y.shape[0] == df_X.shape[0]:
            raise ValueError("Number of regions is not equal")
        if df_y.shape[1] != 1:
            raise ValueError("y needs to have 1 label column")

        # Check if motif table contains integer counts
        if set(df_X.dtypes) != {np.dtype(int)}:
            # Convert to binary counts (presence/absence)
            df_X = (df_X > 0).astype(int)
            print("Warning: Converting motif scores to binary presence/absence")

        # Get unique clusters
        clusters = df_y[df_y.columns[0]].unique()
        M = df_X.shape[0]  # Total number of regions
        
        # Initialize arrays for p-values
        pvals = np.ones((len(clusters), df_X.shape[1]))
        
        # Process in batches to avoid GPU memory issues
        motif_batches = [df_X.columns[i:i+self.batch_size] for i in range(0, df_X.shape[1], self.batch_size)]
        
        for batch_idx, motif_batch in enumerate(motif_batches):
            print(f"Processing batch {batch_idx+1}/{len(motif_batches)} with {len(motif_batch)} motifs")
            
            # Convert to cuDF for GPU processing
            gdf_X = cudf.DataFrame(df_X[motif_batch])
            gdf_y = cudf.DataFrame(df_y)
            
            for i, cluster in enumerate(clusters):
                # Get indices for regions in this cluster
                in_cluster = gdf_y.iloc[:, 0] == cluster
                
                # Calculate counts on GPU
                pos_true = gdf_X[in_cluster.values].astype(bool).sum().values_host
                pos_false = (~gdf_X[in_cluster.values].astype(bool)).sum().values_host
                neg_true = gdf_X[~in_cluster.values].astype(bool).sum().values_host
                
                batch_pvals = []
                for pt, pf, nt in zip(pos_true, pos_false, neg_true):
                    n = pt + nt      # Total regions with motif
                    N = pt + pf      # Total regions in cluster
                    x = pt - 1       # Successes - 1 for sf calculation
                    
                    # Use CPU hypergeom for now as cupy's implementation might be limited
                    p_val = hypergeom.sf(x, M, n, N)
                    batch_pvals.append(p_val)
                
                # Store p-values for this batch
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + len(motif_batch)
                pvals[i, batch_start:batch_end] = batch_pvals
        
        # Correct for multiple testing
        flat_pvals = pvals.flatten()
        flat_fdr = multipletests(flat_pvals, method="fdr_bh")[1]
        fdr = flat_fdr.reshape(pvals.shape)
        
        # Create output DataFrame
        self.act_ = pd.DataFrame(-np.log10(fdr.T), columns=clusters, index=df_X.columns)
        
        print(f"GPU Hypergeometric test completed in {time.time() - start_time:.2f} seconds")
        return self


class GPURFPredictor(GPUMotifActivityPredictor):
    """GPU-accelerated Random Forest for motif activity prediction."""
    
    def __init__(self, batch_size=1000, n_estimators=100, **kwargs):
        self.act_ = None
        self.pref_table = "score"
        self.ptype = "classification"
        self.batch_size = batch_size
        self.n_estimators = n_estimators
        
    def fit(self, df_X, df_y):
        """
        Fit the predictor using Random Forest on GPU.
        
        Parameters
        ----------
        df_X : pandas.DataFrame
            Motif scores per region (peaks-by-motifs)
        df_y : pandas.DataFrame
            Cluster labels for each region
        
        Returns
        -------
        self
        """
        print("Starting GPU Random Forest...")
        start_time = time.time()
        
        if not df_y.shape[0] == df_X.shape[0]:
            raise ValueError("Number of regions is not equal")
        if df_y.shape[1] != 1:
            raise ValueError("y needs to have 1 label column")

        # Get unique clusters and encode them
        clusters = df_y[df_y.columns[0]].unique()
        cluster_to_idx = {c: i for i, c in enumerate(clusters)}
        y_encoded = df_y.iloc[:, 0].map(cluster_to_idx).values
        
        # Initialize feature importance array
        importances = np.zeros((df_X.shape[1], len(clusters)))
        
        # Process in batches to avoid GPU memory issues
        motif_batches = [list(range(i, min(i+self.batch_size, df_X.shape[1]))) 
                          for i in range(0, df_X.shape[1], self.batch_size)]
        
        for batch_idx, col_indices in enumerate(motif_batches):
            print(f"Processing batch {batch_idx+1}/{len(motif_batches)} with {len(col_indices)} motifs")
            
            # Extract batch of features
            X_batch = df_X.iloc[:, col_indices].values
            
            # Convert to GPU arrays
            X_gpu = cp.array(X_batch)
            y_gpu = cp.array(y_encoded)
            
            # Train a Random Forest classifier for each class (one-vs-rest)
            for class_idx in range(len(clusters)):
                # Create binary target for this class
                binary_y = (y_gpu == class_idx).astype(cp.int32)
                
                # Train RF model on GPU
                clf = cuRF(n_estimators=self.n_estimators)
                clf.fit(X_gpu, binary_y)
                
                # Get feature importances
                batch_importances = clf.feature_importances_
                
                # Determine direction (positive/negative)
                # We need to compute this on CPU for now
                for i, col_idx in enumerate(col_indices):
                    class_mask = y_encoded == class_idx
                    if np.sum(class_mask) > 0 and np.sum(~class_mask) > 0:
                        # Calculate direction based on difference in distributions
                        pos_q75 = np.quantile(df_X.iloc[class_mask, col_idx], 0.75)
                        neg_q75 = np.quantile(df_X.iloc[~class_mask, col_idx], 0.75)
                        sign = 1 if pos_q75 >= neg_q75 else -1
                        importances[col_idx, class_idx] = batch_importances[i] * sign
                    else:
                        importances[col_idx, class_idx] = batch_importances[i]
        
        # Create output DataFrame
        self.act_ = pd.DataFrame(importances, columns=clusters, index=df_X.columns)
        
        print(f"GPU Random Forest completed in {time.time() - start_time:.2f} seconds")
        return self


def _combine_results(dfs, method="mean"):
    """
    Combine results from different methods.
    
    Parameters
    ----------
    dfs : dict
        Dictionary of DataFrames from different predictors
    method : str, optional
        How to combine results, 'mean' or 'max'
        
    Returns
    -------
    pandas.DataFrame
    """
    # Initialize with first DataFrame
    first_df = list(dfs.values())[0]
    combined = pd.DataFrame(0, index=first_df.index, columns=first_df.columns)
    
    # Z-score normalize each result
    for method_name, df in dfs.items():
        for col in df.columns:
            values = df[col].values
            mean = np.mean(values)
            std = np.std(values)
            if std > 0:
                df[col] = (values - mean) / std
    
    # Combine based on method
    if method == "mean":
        for df in dfs.values():
            combined += df
        combined /= len(dfs)
    elif method == "max":
        for df in dfs.values():
            combined = combined.combine(df, lambda x, y: x if abs(x) > abs(y) else y)
    
    return combined


def compute_gpu_motif_enrichment(
    peaks_motifs_matrix,
    cluster_labels,
    methods=None,
    batch_size=1000,
    combination_method="mean"
):
    """
    Compute motif enrichment scores using GPU acceleration.
    
    Parameters
    ----------
    peaks_motifs_matrix : pandas.DataFrame
        Matrix with motif scores for each peak
    cluster_labels : pandas.DataFrame or pandas.Series
        Cluster labels for each peak
    methods : list, optional
        List of activity prediction methods to use
    batch_size : int, optional
        Size of batches for GPU processing
    combination_method : str, optional
        How to combine results from different methods ("mean" or "max")
    
    Returns
    -------
    pandas.DataFrame
        Clusters-by-motifs matrix with enrichment scores
    """
    print(f"Starting GPU-accelerated motif enrichment analysis on {peaks_motifs_matrix.shape[0]} peaks and {peaks_motifs_matrix.shape[1]} motifs")
    start_time = time.time()
    
    if isinstance(cluster_labels, pd.Series):
        cluster_labels = pd.DataFrame(cluster_labels)
    
    # Ensure that indices match
    if not all(peaks_motifs_matrix.index == cluster_labels.index):
        raise ValueError("Peak indices in motif matrix and cluster labels must match")
    
    # Default methods if none specified
    if methods is None:
        methods = ["hypergeom", "rf"]
    
    # Run each method and collect results
    result_dfs = {}
    for method in methods:
        method_start = time.time()
        print(f"Running method: {method}")
        predictor = GPUMotifActivityPredictor.create(
            method, 
            batch_size=batch_size
        )
        
        # Choose appropriate matrix type (scores or binary counts)
        if predictor.pref_table == "count" and predictor.ptype == "classification":
            # For methods that need counts, convert scores to binary presence/absence
            X = (peaks_motifs_matrix > 0).astype(int)
        else:
            X = peaks_motifs_matrix
            
        # Fit the predictor
        predictor.fit(X, cluster_labels)
        
        # Store the result
        result_dfs[method] = predictor.act_
        print(f"Method {method} completed in {time.time() - method_start:.2f} seconds")
    
    # If only one method was used, return its result directly
    if len(methods) == 1:
        final_scores = result_dfs[methods[0]]
    else:
        # Combine results from multiple methods
        print("Combining results from multiple methods...")
        final_scores = _combine_results(result_dfs, method=combination_method)
    
    # Add percentage of peaks with motif per cluster
    print("Calculating motif frequency per cluster...")
    motif_presence = (peaks_motifs_matrix > 0).astype(int)
    
    # Process in batches for large datasets
    motif_batches = [motif_presence.columns[i:i+batch_size] 
                     for i in range(0, motif_presence.shape[1], batch_size)]
    
    freq_parts = []
    for batch in motif_batches:
        batch_presence = motif_presence[batch]
        batch_freq = batch_presence.join(cluster_labels).groupby(cluster_labels.columns[0]).mean() * 100
        freq_parts.append(batch_freq)
    
    freq_by_cluster = pd.concat(freq_parts, axis=1)
    freq_by_cluster_transposed = freq_by_cluster.T
    freq_by_cluster_transposed = freq_by_cluster_transposed.rename(
        columns={col: f"{col} % with motif" for col in freq_by_cluster_transposed.columns}
    )
    
    # Combine with enrichment scores
    final_result = pd.concat([final_scores, freq_by_cluster_transposed], axis=1)
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return final_result


# Placeholder for scipy.stats.hypergeom for CPU fallback
from scipy.stats import hypergeom

# Example usage:
"""
# 1. Load your peaks-by-motifs matrix
peaks_motifs = pd.read_csv("motif.score.txt.gz", index_col=0, sep="\t")

# 2. Load your cluster labels
clusters = pd.read_csv("cluster_labels.txt", index_col=0, sep="\t")

# 3. Compute the enrichment scores with GPU acceleration
enrichment_scores = compute_gpu_motif_enrichment(
    peaks_motifs, 
    clusters, 
    methods=["hypergeom", "rf"],
    batch_size=1000  # Adjust based on your GPU memory
)

# 4. Save the results
enrichment_scores.to_csv("motif_enrichment_scores.txt", sep="\t")
"""