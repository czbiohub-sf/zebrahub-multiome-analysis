"""
Module: tfidf_lsi_utils.py

Utility functions for TF-IDF normalization and LSI computation on sparse binary matrices (e.g., peaks × motifs in scATAC-seq).

Dependencies:
- numpy
- scipy.sparse
- anndata
- scikit-learn
- (optional, for GPU) cupy, cuml
"""

import numpy as np
from scipy import sparse
import time

try:
    import cupy as cp
    from cuml.decomposition import TruncatedSVD as cuTruncatedSVD
except ImportError:
    cp = None
    cuTruncatedSVD = None

from sklearn.decomposition import TruncatedSVD as skTruncatedSVD


def tfidf_normalize(X_binary, method='signac'):
    """
    Apply TF-IDF normalization to a binary matrix.
    See docstring in user prompt for details.
    """
    X = X_binary.astype(np.float64)
    n_peaks, n_motifs = X.shape

    if method == 'signac':
        row_sums = np.array(X.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        row_inv = sparse.diags(1.0 / row_sums)
        X_tf = row_inv @ X
        col_sums = np.array(X.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1
        idf = np.log1p(n_peaks / col_sums)
        X_tfidf = X_tf @ sparse.diags(idf)
        X_tfidf = X_tfidf.multiply(1e4)
        X_tfidf.data = np.log1p(X_tfidf.data)
    elif method == 'lsi':
        col_sums = np.array(X.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1
        idf = np.log1p(n_peaks / col_sums)
        X_tfidf = X @ sparse.diags(idf)
        X_tfidf.data = np.log1p(X_tfidf.data)
    X_tfidf = X_tfidf.tocsr()
    return X_tfidf, idf


def tfidf_within_species(adata):
    """
    Apply TF-IDF normalization separately within each species, then recombine.
    See docstring in user prompt for details.
    """
    adata.layers['binary'] = adata.X.copy()
    species_list = adata.obs['species'].unique().tolist()
    tfidf_blocks = []
    all_idf = {}
    for species in sorted(species_list):
        mask = (adata.obs['species'] == species).values
        n_species = mask.sum()
        X_species = adata.X[mask, :]
        col_sums = np.array(X_species.sum(axis=0)).flatten()
        prevalence = col_sums / n_species
        X_tfidf, idf = tfidf_normalize(X_species, method='signac')
        tfidf_blocks.append((mask, X_tfidf))
        all_idf[species] = idf
    X_combined = sparse.lil_matrix(adata.shape, dtype=np.float64)
    for mask, X_tfidf in tfidf_blocks:
        X_combined[mask, :] = X_tfidf
    adata.X = X_combined.tocsr()
    for species, idf in all_idf.items():
        adata.var[f'idf_{species}'] = idf
    return adata


def compute_lsi(adata, n_components=50, drop_first=True, use_gpu=True):
    """
    Compute Latent Semantic Indexing via Truncated SVD on the TF-IDF matrix.
    See docstring in user prompt for details.
    """
    t0 = time.time()
    n_compute = n_components + 1 if drop_first else n_components
    if use_gpu and cp is not None and cuTruncatedSVD is not None:
        try:
            if sparse.issparse(adata.X):
                X_dense = adata.X.toarray().astype(np.float32)
            else:
                X_dense = np.asarray(adata.X, dtype=np.float32)
            X_gpu = cp.asarray(X_dense)
            svd = cuTruncatedSVD(n_components=n_compute, random_state=42)
            X_lsi_gpu = svd.fit_transform(X_gpu)
            X_lsi = cp.asnumpy(X_lsi_gpu)
            explained_variance_ratio = cp.asnumpy(svd.explained_variance_ratio_)
            components = cp.asnumpy(svd.components_)
            del X_gpu, X_lsi_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            use_gpu = False
    if not use_gpu:
        svd = skTruncatedSVD(n_components=n_compute, random_state=42, algorithm='arpack')
        X_lsi = svd.fit_transform(adata.X.astype(np.float32))
        explained_variance_ratio = svd.explained_variance_ratio_
        components = svd.components_
    if drop_first:
        X_lsi = X_lsi[:, 1:]
        explained_variance_ratio = explained_variance_ratio[1:]
        components = components[1:]
    adata.obsm['X_lsi'] = X_lsi.astype(np.float32)
    adata.uns['lsi'] = {
        'variance_ratio': explained_variance_ratio,
        'cumulative_variance': np.cumsum(explained_variance_ratio),
        'components': components,
    }
    return adata
