#!/usr/bin/env python3
"""
GPU-accelerated version of GimmeMotifs rank.py
Uses CuPy for massive speedup in rank aggregation operations
"""

import numpy as np
import pandas as pd

# GPU imports with fallbacks
try:
    import cupy as cp
    from cupyx.scipy.special import factorial as cp_factorial
    from cupyx.scipy.stats import norm as cp_norm
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU rank aggregation")
except ImportError:
    import numpy as cp
    from scipy.special import factorial as cp_factorial
    from scipy.stats import norm as cp_norm
    CUPY_AVAILABLE = False
    print("⚠️  CuPy not available - rank aggregation will use CPU")


def gpu_sumStuart(v, r):
    """GPU-accelerated Stuart sum calculation using CuPy"""
    if not CUPY_AVAILABLE:
        # CPU fallback
        k = len(v)
        l_k = np.arange(k)
        ones = (-1) ** l_k
        f = cp_factorial(l_k + 1)
        p = r ** (l_k + 1)
        return np.dot(ones, v[::-1] * p / f)
    
    # GPU computation
    v_gpu = cp.asarray(v)
    k = len(v_gpu)
    l_k = cp.arange(k)
    ones = (-1) ** l_k
    f = cp_factorial(l_k + 1)
    p = r ** (l_k + 1)
    
    result = cp.dot(ones, v_gpu[::-1] * p / f)
    return float(cp.asnumpy(result))


def gpu_qStuart(r):
    """GPU-accelerated Stuart q calculation using CuPy"""
    if not CUPY_AVAILABLE:
        # CPU fallback
        N = (~r.isnull()).sum().sum()
        v = np.ones(N + 1)
        for k in range(N):
            v[k + 1] = gpu_sumStuart(v[: k + 1], r[N - k - 1])
        return cp_factorial(N) * v[N]
    
    # GPU computation
    r_gpu = cp.asarray(r.values)
    N = int(cp.sum(~cp.isnan(r_gpu)))
    v = cp.ones(N + 1)
    
    for k in range(N):
        v[k + 1] = gpu_sumStuart(cp.asnumpy(v[: k + 1]), r.iloc[N - k - 1])
    
    result = cp_factorial(N) * v[N]
    return float(cp.asnumpy(result))


def gpu_rank_int(series, c=3.0 / 8, stochastic=True):
    """
    GPU-accelerated rank-based inverse normal transformation using CuPy
    Much faster for large series than scipy version
    """
    assert isinstance(series, pd.Series)
    
    if not CUPY_AVAILABLE:
        # CPU fallback - use original implementation
        return _rank_int_cpu(series, c, stochastic)
    
    # GPU acceleration
    orig_idx = series.index
    series = series.loc[~pd.isnull(series)]
    
    if len(series) == 0:
        return pd.Series(index=orig_idx, dtype=float)
    
    # Convert to GPU arrays
    values_gpu = cp.asarray(series.values)
    
    # GPU-accelerated ranking
    if stochastic:
        # Shuffle indices on GPU
        cp.random.seed(123)
        perm_idx = cp.random.permutation(len(values_gpu))
        values_shuffled = values_gpu[perm_idx]
        
        # GPU ranking (argsort is much faster on GPU for large arrays)
        rank_gpu = cp.argsort(cp.argsort(values_shuffled)) + 1
        
        # Restore original order
        rank_restored = cp.zeros_like(rank_gpu)
        rank_restored[perm_idx] = rank_gpu
        rank_gpu = rank_restored
    else:
        # Average ranking on GPU
        sorted_idx = cp.argsort(values_gpu)
        rank_gpu = cp.zeros_like(values_gpu, dtype=cp.float64)
        
        # Handle ties by averaging ranks
        unique_vals, inverse_idx, counts = cp.unique(values_gpu, return_inverse=True, return_counts=True)
        
        for i, (val, count) in enumerate(zip(unique_vals, counts)):
            mask = values_gpu == val
            positions = cp.where(mask)[0]
            avg_rank = cp.mean(cp.arange(len(positions)) + cp.sum(counts[:i]) + 1)
            rank_gpu[mask] = avg_rank
    
    # Convert rank to normal distribution on GPU
    n = len(rank_gpu)
    x_gpu = (rank_gpu - c) / (n - 2 * c + 1)
    
    # GPU-accelerated inverse normal transformation
    transformed_gpu = cp_norm.ppf(x_gpu)
    
    # Convert back to pandas Series
    transformed = pd.Series(cp.asnumpy(transformed_gpu), index=series.index)
    return transformed[orig_idx]


def _rank_int_cpu(series, c=3.0 / 8, stochastic=True):
    """CPU fallback for rank-based inverse normal transformation"""
    from scipy.stats import rankdata, norm
    
    orig_idx = series.index
    series = series.loc[~pd.isnull(series)]
    
    if len(series) == 0:
        return pd.Series(index=orig_idx, dtype=float)
    
    np.random.seed(123)
    
    if stochastic:
        series = series.loc[np.random.permutation(series.index)]
        rank = rankdata(series, method="ordinal")
    else:
        rank = rankdata(series, method="average")
    
    rank = pd.Series(rank, index=series.index)
    
    # Convert rank to normal distribution
    def rank_to_normal(rank_val, c, n):
        x = (rank_val - c) / (n - 2 * c + 1)
        return norm.ppf(x)
    
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    return transformed[orig_idx]


def gpu_rankagg_int(df):
    """
    GPU-accelerated inverse normal transform rank aggregation
    Uses CuPy for faster computation on large DataFrames
    """
    if not CUPY_AVAILABLE or df.shape[0] < 1000:
        # Use CPU for small datasets or when GPU unavailable
        df_int = df.apply(lambda x: gpu_rank_int(x))  # Still faster than original
        combined = (df_int.sum(1) / np.sqrt(df_int.shape[1])).to_frame()
        combined.columns = ["z-score"]
        return combined
    
    # GPU acceleration for large datasets
    print(f"🚀 GPU-accelerating rank aggregation for {df.shape[0]} x {df.shape[1]} matrix")
    
    # Convert DataFrame to GPU arrays
    df_gpu = cp.asarray(df.values)
    
    # GPU-accelerated rank computation for each column
    transformed_cols = []
    
    for col_idx in range(df_gpu.shape[1]):
        col_data = df_gpu[:, col_idx]
        
        # Remove NaN values
        valid_mask = ~cp.isnan(col_data)
        valid_data = col_data[valid_mask]
        
        if len(valid_data) == 0:
            transformed_cols.append(cp.full(df_gpu.shape[0], cp.nan))
            continue
        
        # GPU ranking
        sorted_idx = cp.argsort(valid_data)
        ranks = cp.empty_like(sorted_idx, dtype=cp.float64)
        ranks[sorted_idx] = cp.arange(1, len(valid_data) + 1)
        
        # Inverse normal transform on GPU
        c = 3.0 / 8
        n = len(valid_data)
        x = (ranks - c) / (n - 2 * c + 1)
        transformed_valid = cp_norm.ppf(x)
        
        # Place back into full array
        transformed_col = cp.full(df_gpu.shape[0], cp.nan)
        transformed_col[valid_mask] = transformed_valid
        transformed_cols.append(transformed_col)
    
    # Stack columns and compute Stouffer's method on GPU
    df_int_gpu = cp.stack(transformed_cols, axis=1)
    
    # Handle NaN values
    valid_counts = cp.sum(~cp.isnan(df_int_gpu), axis=1)
    valid_counts = cp.maximum(valid_counts, 1)  # Avoid division by zero
    
    # Sum and normalize (Stouffer's method)
    col_sums = cp.nansum(df_int_gpu, axis=1)
    combined_gpu = col_sums / cp.sqrt(valid_counts)
    
    # Convert back to pandas
    combined = pd.DataFrame(
        cp.asnumpy(combined_gpu), 
        index=df.index, 
        columns=["z-score"]
    )
    
    print("✅ GPU rank aggregation completed")
    return combined


def gpu_rankagg_stuart(df):
    """
    GPU-accelerated Stuart rank aggregation
    Uses CuPy for faster computation when possible
    """
    print(f"🚀 GPU Stuart rank aggregation for {df.shape[0]} x {df.shape[1]} matrix")
    
    if CUPY_AVAILABLE and df.shape[0] > 1000:
        # GPU acceleration for large datasets
        rmat = pd.DataFrame(index=df.iloc[:, 0])
        step = 1 / rmat.shape[0]
        
        # Convert rank matrices to GPU for faster sorting
        for col in df.columns:
            # Create rank matrix on GPU
            rank_array = cp.arange(step, 1 + step, step)
            
            # GPU-accelerated sorting and indexing
            sorted_indices = cp.argsort(cp.asarray(df[col].values))
            rmat[col] = cp.asnumpy(rank_array)[cp.asnumpy(sorted_indices)]
        
        # GPU-accelerated row-wise sorting
        rmat_gpu = cp.asarray(rmat.values)
        rmat_sorted_gpu = cp.sort(rmat_gpu, axis=1)
        rmat_sorted = pd.DataFrame(rmat_sorted_gpu, index=rmat.index)
        
        # Apply Stuart calculation (this part stays on CPU due to complexity)
        p = rmat_sorted.apply(gpu_qStuart, axis=1)
        
    else:
        # CPU fallback for smaller datasets
        rmat = pd.DataFrame(index=df.iloc[:, 0])
        step = 1 / rmat.shape[0]
        
        for col in df.columns:
            rmat[col] = pd.DataFrame(
                {col: np.arange(step, 1 + step, step)}, index=df[col]
            ).loc[rmat.index]
        
        rmat = rmat.apply(sorted, axis=1, result_type="expand")
        p = rmat.apply(gpu_qStuart, axis=1)
    
    print("✅ GPU Stuart rank aggregation completed")
    return pd.DataFrame({"score": p}, index=rmat.index)


def gpu_rankagg(df, method="int_stouffer", include_reverse=True, log_transform=True):
    """
    GPU-accelerated rank aggregation function
    
    Provides massive speedup for large datasets using CuPy
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with values to be ranked and aggregated
    method : str, optional
        Either "int_stouffer" or "stuart"
    include_reverse : bool, optional
        Include reverse rankings for Stuart method
    log_transform : bool, optional
        Apply log transform to Stuart results
        
    Returns
    -------
    pandas.DataFrame with aggregated ranks
    """
    
    method = method.lower()
    if method not in ["stuart", "int_stouffer"]:
        raise ValueError("Unknown method for rank aggregation")

    print(f"🚀 Starting GPU rank aggregation with method: {method}")
    
    if method == "stuart":
        # Stuart method with optional GPU acceleration
        df_asc = pd.DataFrame()
        df_desc = pd.DataFrame()
        
        for col in df.columns:
            # GPU-accelerated sampling and sorting if available
            if CUPY_AVAILABLE and len(df) > 10000:
                # Use GPU for large datasets
                values_gpu = cp.asarray(df[col].values)
                indices_gpu = cp.arange(len(values_gpu))
                
                # GPU shuffle
                cp.random.shuffle(indices_gpu)
                shuffled_values = values_gpu[indices_gpu]
                
                # GPU sort
                sort_idx_desc = cp.argsort(-shuffled_values)  # Descending
                sort_idx_asc = cp.argsort(shuffled_values)    # Ascending
                
                df_asc[col] = df.index[cp.asnumpy(indices_gpu[sort_idx_desc])].values
                if include_reverse:
                    df_desc[col] = df.index[cp.asnumpy(indices_gpu[sort_idx_asc])].values
                    
            else:
                # CPU fallback
                df_asc[col] = df.sample(frac=1).sort_values(col, ascending=False).index.values
                if include_reverse:
                    df_desc[col] = df.sample(frac=1).sort_values(col, ascending=True).index.values

        df_result = -np.log10(gpu_rankagg_stuart(df_asc)) if log_transform else gpu_rankagg_stuart(df_asc)
        
        if include_reverse:
            df_reverse = np.log10(gpu_rankagg_stuart(df_desc)) if log_transform else gpu_rankagg_stuart(df_desc)
            df_result += df_reverse

        return df_result
        
    elif method == "int_stouffer":
        # Inverse normal transform + Stouffer's method with GPU acceleration
        return gpu_rankagg_int(df)


# Backwards compatibility aliases
rankagg = gpu_rankagg
_rankagg_int = gpu_rankagg_int
_rankagg_stuart = gpu_rankagg_stuart
_rank_int = gpu_rank_int