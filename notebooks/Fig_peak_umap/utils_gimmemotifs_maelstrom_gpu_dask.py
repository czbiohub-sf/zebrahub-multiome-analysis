import numpy as np
import pandas as pd
import cudf
import cupy as cp
from scipy.stats import hypergeom
import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
from statsmodels.stats.multitest import multipletests
import time
import dask_cudf as dc
import dask.dataframe as dd
from dask.distributed import Client, progress

class DaskGPUMotifActivityPredictor:
    """Base class for Dask-GPU-accelerated motif activity prediction."""
    
    @staticmethod
    def create(method_name, **kwargs):
        """Factory method to create activity predictors."""
        methods = {
            "hypergeom": DaskGPUHypergeomPredictor,
            "rf": DaskGPURFPredictor,
        }
        
        if method_name.lower() not in methods:
            raise ValueError(f"Unknown method: {method_name}. Available methods: {list(methods.keys())}")
        
        return methods[method_name.lower()](**kwargs)
    
    @staticmethod
    def list_predictors():
        """List available predictors."""
        return ["hypergeom", "rf"]


class DaskGPUHypergeomPredictor(DaskGPUMotifActivityPredictor):
    """Dask-GPU-accelerated hypergeometric test for motif activity prediction."""
    
    def __init__(self, batch_size=1000, n_workers=1, **kwargs):
        self.act_ = None
        self.pref_table = "count"
        self.ptype = "classification"
        self.batch_size = batch_size
        self.n_workers = n_workers
        
    def fit(self, ddf_X, df_y):
        """
        Fit the predictor using hypergeometric test with Dask-GPU acceleration.
        
        Parameters
        ----------
        ddf_X : dask_cudf.DataFrame
            Motif scores per region (peaks-by-motifs)
        df_y : pandas.DataFrame or cudf.DataFrame
            Cluster labels for each region
        
        Returns
        -------
        self
        """
        print("Starting Dask-GPU Hypergeometric test...")
        start_time = time.time()
        
        # Convert cluster labels to cuDF if needed
        if isinstance(df_y, pd.DataFrame):
            df_y = cudf.DataFrame.from_pandas(df_y)
        
        # Get unique clusters
        clusters = df_y[df_y.columns[0]].unique().to_pandas().tolist()
        print(f"Processing {len(clusters)} clusters")
        
        # Get total number of regions
        M = len(ddf_X)
        print(f"Total regions: {M}")
        
        # Convert to binary counts (presence/absence) if needed
        # Note: This is a costly operation on a large dask_cudf dataframe
        print("Converting motif scores to binary presence/absence")
        ddf_X_binary = (ddf_X > 0).astype(int)
        
        # Get all column names
        all_columns = ddf_X.columns
        
        # Initialize results DataFrame
        pvals = np.ones((len(clusters), len(all_columns)))
        
        # Process in batches
        motif_batches = [all_columns[i:i+self.batch_size] for i in range(0, len(all_columns), self.batch_size)]
        
        for batch_idx, motif_batch in enumerate(motif_batches):
            print(f"Processing batch {batch_idx+1}/{len(motif_batches)} with {len(motif_batch)} motifs")
            
            # Create batch dataframe with selected motifs
            ddf_batch = ddf_X_binary[motif_batch]
            
            for i, cluster in enumerate(clusters):
                print(f"  Processing cluster {cluster}")
                
                # Create mask for regions in this cluster
                in_cluster = df_y.iloc[:, 0] == cluster
                
                # Convert to indices where condition is True
                cluster_indices = in_cluster.index[in_cluster].values_host
                
                # Process hypergeometric test for this cluster and batch
                batch_pvals = self._process_hypergeom_batch(
                    ddf_batch, 
                    cluster_indices, 
                    M, 
                    motif_batch
                )
                
                # Store p-values for this batch
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + len(motif_batch)
                pvals[i, batch_start:batch_end] = batch_pvals
        
        # Correct for multiple testing
        flat_pvals = pvals.flatten()
        flat_fdr = multipletests(flat_pvals, method="fdr_bh")[1]
        fdr = flat_fdr.reshape(pvals.shape)
        
        # Create output DataFrame
        self.act_ = pd.DataFrame(-np.log10(fdr.T), columns=clusters, index=all_columns)
        
        print(f"Dask-GPU Hypergeometric test completed in {time.time() - start_time:.2f} seconds")
        return self
    
    def _process_hypergeom_batch(self, ddf_batch, cluster_indices, M, motif_batch):
        """Process hypergeometric test for a batch of motifs for a single cluster."""
        
        # Extract regions in this cluster and not in this cluster
        try:
            # Using dask_cudf operations
            pos = ddf_batch.iloc[cluster_indices]
            neg = ddf_batch.drop(cluster_indices)
            
            # Calculate counts (trigger computation)
            pos_true = pos.astype(bool).sum().compute().values
            neg_true = neg.astype(bool).sum().compute().values
            pos_false = len(cluster_indices) - pos_true
            
            # Calculate p-values
            batch_pvals = []
            for pt, pf, nt in zip(pos_true, pos_false, neg_true):
                n = pt + nt      # Total regions with motif
                N = pt + pf      # Total regions in cluster
                x = pt - 1       # Successes - 1 for sf calculation
                
                # Use CPU hypergeom since cupy doesn't have hypergeom.sf
                p_val = hypergeom.sf(x, M, n, N)
                batch_pvals.append(p_val)
                
            return batch_pvals
            
        except Exception as e:
            print(f"Error in hypergeom calculation: {e}")
            # Return all ones (p-value of 1 means no significance)
            return [1.0] * len(motif_batch)


class DaskGPURFPredictor(DaskGPUMotifActivityPredictor):
    """Dask-GPU-accelerated Random Forest for motif activity prediction."""
    
    def __init__(self, batch_size=100, n_estimators=100, n_workers=1, **kwargs):
        self.act_ = None
        self.pref_table = "score"
        self.ptype = "classification"
        self.batch_size = batch_size
        self.n_estimators = n_estimators
        self.n_workers = n_workers
        
    def fit(self, ddf_X, df_y):
        """
        Fit the predictor using Random Forest on GPU with Dask.
        
        Parameters
        ----------
        ddf_X : dask_cudf.DataFrame
            Motif scores per region (peaks-by-motifs)
        df_y : pandas.DataFrame or cudf.DataFrame
            Cluster labels for each region
        
        Returns
        -------
        self
        """
        print("Starting Dask-GPU Random Forest...")
        start_time = time.time()
        
        # Convert cluster labels to cuDF if needed
        if isinstance(df_y, pd.DataFrame):
            df_y = cudf.DataFrame.from_pandas(df_y)
        
        # Get unique clusters and encode them
        clusters = df_y[df_y.columns[0]].unique().to_pandas().tolist()
        print(f"Processing {len(clusters)} clusters")
        
        # Create mapping from cluster names to indices
        cluster_to_idx = {c: i for i, c in enumerate(clusters)}
        
        # Get all column names
        all_columns = ddf_X.columns
        
        # Initialize feature importance array
        importances = np.zeros((len(all_columns), len(clusters)))
        
        # Warning: RF on very large datasets can be extremely memory-intensive and slow
        print("WARNING: Random Forest on large datasets can be very memory-intensive.")
        print("Consider using a smaller subset of motifs for testing.")
        
        # Process in small batches due to RF memory usage
        small_batch_size = min(self.batch_size, 100)  # Use a smaller batch size for RF
        motif_batches = [all_columns[i:i+small_batch_size] for i in range(0, len(all_columns), small_batch_size)]
        
        for batch_idx, motif_batch in enumerate(motif_batches):
            print(f"Processing batch {batch_idx+1}/{len(motif_batches)} with {len(motif_batch)} motifs")
            
            try:
                # For RF, we need to materialize a portion of the data
                # This is memory-intensive but necessary
                print("  Loading batch data into memory (this may take time)")
                X_batch = ddf_X[motif_batch].compute()
                
                # Encode cluster labels
                y_encoded = df_y.iloc[:, 0].map(cluster_to_idx).values
                
                # Train a Random Forest classifier for each class (one-vs-rest)
                for class_idx, cluster_name in enumerate(clusters):
                    print(f"  Training RF for cluster {cluster_name}")
                    
                    # Create binary target for this class
                    binary_y = (y_encoded == class_idx).astype(cp.int32)
                    
                    try:
                        # Train RF model on GPU
                        clf = cuRF(n_estimators=self.n_estimators)
                        clf.fit(X_batch, binary_y)
                        
                        # Get feature importances
                        batch_importances = clf.feature_importances_
                        
                        # Determine direction (positive/negative)
                        for i, motif_name in enumerate(motif_batch):
                            col_idx = list(all_columns).index(motif_name)
                            class_mask = y_encoded == class_idx
                            
                            if np.sum(class_mask) > 0 and np.sum(~class_mask) > 0:
                                # Calculate direction based on difference in distributions
                                pos_q75 = X_batch[class_mask, i].quantile(0.75)
                                neg_q75 = X_batch[~class_mask, i].quantile(0.75)
                                sign = 1 if pos_q75 >= neg_q75 else -1
                                importances[col_idx, class_idx] = batch_importances[i] * sign
                            else:
                                importances[col_idx, class_idx] = batch_importances[i]
                    except Exception as e:
                        print(f"    Error in RF for cluster {cluster_name}: {e}")
                        # Continue with other clusters
            
            except Exception as e:
                print(f"Error processing batch {batch_idx+1}: {e}")
                # Continue with other batches
        
        # Create output DataFrame
        self.act_ = pd.DataFrame(importances, columns=clusters, index=all_columns)
        
        print(f"Dask-GPU Random Forest completed in {time.time() - start_time:.2f} seconds")
        return self


def _combine_results(dfs, method="mean"):
    """Combine results from different methods."""
    # Initialize with first DataFrame
    first_df = list(dfs.values())[0]
    combined = pd.DataFrame(0, index=first_df.index, columns=first_df.columns)
    
    # Z-score normalize each result
    normalized_dfs = {}
    for method_name, df in dfs.items():
        normalized = df.copy()
        for col in df.columns:
            values = df[col].values
            mean = np.mean(values)
            std = np.std(values)
            if std > 0:
                normalized[col] = (values - mean) / std
        normalized_dfs[method_name] = normalized
    
    # Combine based on method
    if method == "mean":
        for df in normalized_dfs.values():
            combined += df
        combined /= len(normalized_dfs)
    elif method == "max":
        combined = None
        for df in normalized_dfs.values():
            if combined is None:
                combined = df.copy()
            else:
                for col in combined.columns:
                    combined[col] = np.sign(combined[col]) * np.maximum(
                        np.abs(combined[col]), np.abs(df[col])
                    )
    
    return combined


def calculate_motif_frequency(ddf_X, df_y, batch_size=1000):
    """
    Calculate the percentage of peaks with each motif per cluster.
    
    Parameters
    ----------
    ddf_X : dask_cudf.DataFrame
        Motif scores per region (peaks-by-motifs)
    df_y : pandas.DataFrame or cudf.DataFrame
        Cluster labels for each region
    batch_size : int
        Size of batches for processing
    
    Returns
    -------
    pandas.DataFrame
        Percentage of peaks with each motif per cluster
    """
    print("Calculating motif frequency per cluster...")
    start_time = time.time()
    
    # Convert to binary presence/absence
    ddf_presence = (ddf_X > 0).astype(int)
    
    # Get all column names and unique clusters
    all_columns = ddf_X.columns
    
    # Convert cluster labels to cuDF if needed
    if isinstance(df_y, pd.DataFrame):
        clusters_cudf = cudf.DataFrame.from_pandas(df_y)
    else:
        clusters_cudf = df_y
    
    unique_clusters = clusters_cudf[clusters_cudf.columns[0]].unique().to_pandas().tolist()
    
    # Initialize results DataFrame
    freq_by_cluster = pd.DataFrame(index=unique_clusters, columns=all_columns)
    
    # Process in batches
    motif_batches = [all_columns[i:i+batch_size] for i in range(0, len(all_columns), batch_size)]
    
    for batch_idx, motif_batch in enumerate(motif_batches):
        print(f"Processing frequency batch {batch_idx+1}/{len(motif_batches)}")
        
        # Create batch dataframe with selected motifs
        ddf_batch = ddf_presence[motif_batch]
        
        try:
            # Calculate presence per cluster for this batch
            batch_freq = {}
            
            for cluster in unique_clusters:
                # Create mask for regions in this cluster
                cluster_mask = clusters_cudf.iloc[:, 0] == cluster
                
                # Count total regions in this cluster
                n_regions = cluster_mask.sum()
                
                # Get regions in this cluster
                cluster_indices = cluster_mask.index[cluster_mask].values_host
                
                # Extract batch for this cluster
                cluster_batch = ddf_batch.iloc[cluster_indices]
                
                # Calculate percentage with motif
                presence_sum = cluster_batch.sum().compute()
                percentage = (presence_sum / n_regions * 100).to_pandas()
                
                batch_freq[cluster] = percentage
            
            # Convert batch results to DataFrame
            batch_df = pd.DataFrame(batch_freq).T
            
            # Update main results DataFrame
            for cluster in unique_clusters:
                for motif in motif_batch:
                    freq_by_cluster.loc[cluster, motif] = batch_df.loc[cluster, motif]
                    
        except Exception as e:
            print(f"Error processing frequency batch {batch_idx+1}: {e}")
            # Set missing values to 0
            for cluster in unique_clusters:
                for motif in motif_batch:
                    if pd.isna(freq_by_cluster.loc[cluster, motif]):
                        freq_by_cluster.loc[cluster, motif] = 0.0
    
    # Transpose to get motifs as rows
    freq_by_cluster_transposed = freq_by_cluster.T
    
    # Rename columns to indicate they are percentages
    freq_by_cluster_transposed = freq_by_cluster_transposed.rename(
        columns={col: f"{col} % with motif" for col in freq_by_cluster_transposed.columns}
    )
    
    print(f"Frequency calculation completed in {time.time() - start_time:.2f} seconds")
    return freq_by_cluster_transposed


def compute_dask_gpu_motif_enrichment(
    ddf_peaks_motifs,
    df_cluster_labels,
    methods=None,
    batch_size=1000,
    combination_method="mean",
    n_workers=1,
    client=None
):
    """
    Compute motif enrichment scores using Dask-GPU acceleration.
    
    Parameters
    ----------
    ddf_peaks_motifs : dask_cudf.DataFrame
        Matrix with motif scores for each peak
    df_cluster_labels : pandas.DataFrame or cudf.DataFrame
        Cluster labels for each peak
    methods : list, optional
        List of activity prediction methods to use
    batch_size : int, optional
        Size of batches for GPU processing
    combination_method : str, optional
        How to combine results from different methods ("mean" or "max")
    n_workers : int, optional
        Number of Dask workers to use
    client : dask.distributed.Client, optional
        Existing Dask client to use
    
    Returns
    -------
    pandas.DataFrame
        Clusters-by-motifs matrix with enrichment scores
    """
    print(f"Starting Dask-GPU-accelerated motif enrichment analysis")
    print(f"Dataset: {len(ddf_peaks_motifs)} peaks and {len(ddf_peaks_motifs.columns)} motifs")
    start_time = time.time()
    
    # Create or use Dask client
    local_client = False
    if client is None:
        print(f"Creating local Dask cluster with {n_workers} workers")
        client = Client(n_workers=n_workers)
        local_client = True
    
    try:
        # Default methods if none specified
        if methods is None:
            methods = ["hypergeom"]  # RF can be very slow for large datasets
        
        # Run each method and collect results
        result_dfs = {}
        for method in methods:
            method_start = time.time()
            print(f"Running method: {method}")
            predictor = DaskGPUMotifActivityPredictor.create(
                method, 
                batch_size=batch_size,
                n_workers=n_workers
            )
            
            # Fit the predictor
            predictor.fit(ddf_peaks_motifs, df_cluster_labels)
            
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
        
        # Calculate percentage of peaks with motif per cluster
        freq_scores = calculate_motif_frequency(
            ddf_peaks_motifs, 
            df_cluster_labels, 
            batch_size=batch_size
        )
        
        # Combine with enrichment scores
        final_result = pd.concat([final_scores, freq_scores], axis=1)
        
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return final_result
        
    finally:
        # Close the local client if we created it
        if local_client:
            client.close()


# Example usage:
"""
import pandas as pd
import cudf
import dask_cudf as dc
from pathlib import Path
import gzip
import shutil
from dask.distributed import Client

# Define the filepaths
motif_gz_path = "motif.score.txt.gz"
motif_txt_path = Path(motif_gz_path).with_suffix('')
cluster_path = "cluster_labels.txt"

# Decompress if needed
if not motif_txt_path.exists():
    print(f"Decompressing {motif_gz_path}...")
    with gzip.open(motif_gz_path, "rb") as f_in, open(motif_txt_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# Read the motif data using dask_cudf
print("Reading motif data with dask_cudf...")
ddf_peaks_motifs = dc.read_csv(
    motif_txt_path,
    sep="\t",
    chunksize="256MB",
)

# Set index if first column is the index
if ddf_peaks_motifs.columns[0] == 'Unnamed: 0':
    # Extract index column
    ddf_peaks_motifs = ddf_peaks_motifs.rename(columns={ddf_peaks_motifs.columns[0]: 'index'})
    ddf_peaks_motifs = ddf_peaks_motifs.set_index('index')

# Read cluster labels
print("Reading cluster labels...")
df_cluster_labels = pd.read_csv(cluster_path, sep='\t', index_col=0)

# Create a Dask client with 4 workers
client = Client(n_workers=4)

try:
    # Compute enrichment scores
    print("Computing enrichment scores...")
    enrichment_scores = compute_dask_gpu_motif_enrichment(
        ddf_peaks_motifs,
        df_cluster_labels,
        methods=["hypergeom"],
        batch_size=500,
        client=client
    )
    
    # Save results
    print("Saving results...")
    enrichment_scores.to_csv("motif_enrichment_scores.txt", sep="\t")
    
finally:
    # Close the client
    client.close()
"""