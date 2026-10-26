"""Bounded-memory equivalent of specialization_flow.structural_flow.

CSR row blocks preserve every supplied atlas edge. They change allocation and
accumulation strategy only: no sampling, new neighbors, approximation, or forced
root directions are introduced. Designed for the full 640K-peak real atlas.
"""
import numpy as np
from scipy import sparse


def block_structural_flow(graph, specificity, embeddings=None, edge_tolerance=1e-3,
                          gradient_power=1., min_support=1, score_uncertainty=None,
                          edge_confidence_z=0., block_size=2048):
    """Return structural_flow-compatible results using CSR blocks/bincount.

    The edge scoring, empirical uncertainty margin, masks, and unit-direction
    projection match structural_flow. Floating accumulation can differ at round-
    off precision. No peak-by-peak dense matrix or all-edge coordinate array is
    created. Existing graph CSR buffers are reused when canonical.
    """
    if not sparse.issparse(graph):
        raise TypeError('Supply a sparse high-dimensional neighbor graph')
    score = np.asarray(specificity, dtype=float)
    if score.ndim != 1 or graph.shape != (len(score), len(score)):
        raise ValueError('Graph and specificity must share an ordered peak axis')
    if not np.isfinite(edge_tolerance) or edge_tolerance < 0:
        raise ValueError('edge_tolerance must be finite and nonnegative')
    if not np.isfinite(gradient_power) or gradient_power <= 0:
        raise ValueError('gradient_power must be finite and positive')
    if not isinstance(min_support, (int, np.integer)) or min_support < 1:
        raise ValueError('min_support must be a positive integer')
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    if not np.isfinite(edge_confidence_z) or edge_confidence_z < 0:
        raise ValueError('edge_confidence_z must be finite and nonnegative')
    n = len(score)
    if score_uncertainty is None:
        if edge_confidence_z > 0:
            raise ValueError('Positive edge_confidence_z requires score_uncertainty')
        uncertainty = None
        uncertainty_valid = np.ones(n, dtype=bool)
    else:
        uncertainty = np.asarray(score_uncertainty, dtype=float)
        if uncertainty.shape != score.shape:
            raise ValueError('Score uncertainty must have one value per peak')
        uncertainty_valid = np.isfinite(uncertainty) & (uncertainty >= 0)
    graph = graph.tocsr(copy=False)
    if not graph.has_canonical_format:
        graph = graph.copy()
        graph.sum_duplicates()
    if not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError('Graph weights must be finite and nonnegative')
    if embeddings is None:
        embeddings = {}
    elif isinstance(embeddings, np.ndarray):
        embeddings = {'embedding': embeddings}
    coordinates = {}
    for name, xy in embeddings.items():
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or len(xy) != n or not np.isfinite(xy).all():
            raise ValueError('Embeddings must be finite and match the peak axis')
        coordinates[name] = xy
    result = dict(support_count=np.zeros(n, dtype=np.int64), support_fraction=np.zeros(n),
                  active=np.zeros(n, dtype=bool), expected_specificity_gain=np.zeros(n),
                  vectors={k: np.zeros_like(xy) for k, xy in coordinates.items()},
                  displacement_vectors={k: np.zeros_like(xy) for k, xy in coordinates.items()},
                  coherence={k: np.zeros(n) for k in coordinates},
                  projection_support={k: np.zeros(n) for k in coordinates})
    out_ptr = np.zeros(n + 1, dtype=np.int64)
    data_parts, index_parts = [], []
    emitted = 0
    for start in range(0, n, block_size):
        stop = min(n, start + block_size); size = stop - start
        left, right = graph.indptr[start], graph.indptr[stop]
        row = np.repeat(np.arange(size), np.diff(graph.indptr[start:stop+1]))
        col = graph.indices[left:right]
        weight = graph.data[left:right].astype(float, copy=False)
        actual_edge = (col != row + start) & (weight > 0)
        base = np.bincount(row[actual_edge], weights=weight[actual_edge], minlength=size)
        gain = score[col] - score[row + start]
        threshold = np.full(len(row), edge_tolerance, dtype=float)
        supported = uncertainty_valid[col] & uncertainty_valid[row + start]
        if uncertainty is not None and edge_confidence_z > 0:
            threshold[supported] += edge_confidence_z * np.hypot(
                uncertainty[col[supported]], uncertainty[(row + start)[supported]])
        keep = actual_edge & supported & np.isfinite(gain) & (gain > threshold)
        count = np.bincount(row[keep], minlength=size)
        result['support_count'][start:stop] = count
        directed_base = np.bincount(row[keep], weights=weight[keep], minlength=size)
        result['support_fraction'][start:stop] = np.divide(directed_base, base, out=np.zeros(size), where=base > 0)
        keep &= count[row] >= min_support
        edge_row, edge_col = row[keep], col[keep]
        edge_gain = gain[keep]
        directed = weight[keep] * (edge_gain - threshold[keep]) ** gradient_power
        row_total = np.bincount(edge_row, weights=directed, minlength=size)
        probability = np.divide(directed, row_total[edge_row], out=np.zeros_like(directed), where=row_total[edge_row] > 0)
        nonzero = probability > 0
        edge_row, edge_col, edge_gain, probability = (
            edge_row[nonzero], edge_col[nonzero], edge_gain[nonzero], probability[nonzero])
        result['active'][start:stop] = row_total > 0
        result['expected_specificity_gain'][start:stop] = np.bincount(edge_row, weights=probability * edge_gain, minlength=size)
        row_count = np.bincount(edge_row, minlength=size)
        out_ptr[start+1:stop+1] = emitted + np.cumsum(row_count)
        emitted += len(probability)
        data_parts.append(probability)
        index_parts.append(edge_col.copy())
        for name, xy in coordinates.items():
            delta = xy[edge_col] - xy[edge_row + start]
            distance = np.linalg.norm(delta, axis=1)
            unit = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
            for d in range(xy.shape[1]):
                result['vectors'][name][start:stop, d] = np.bincount(edge_row, weights=probability * unit[:, d], minlength=size)
                result['displacement_vectors'][name][start:stop, d] = np.bincount(edge_row, weights=probability * delta[:, d], minlength=size)
            result['projection_support'][name][start:stop] = np.bincount(edge_row, weights=probability * (distance > 0), minlength=size)
            result['coherence'][name][start:stop] = np.linalg.norm(result['vectors'][name][start:stop], axis=1)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=float)
    index = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=graph.indices.dtype)
    result['probabilities'] = sparse.csr_matrix((data, index, out_ptr), shape=graph.shape)
    return result
