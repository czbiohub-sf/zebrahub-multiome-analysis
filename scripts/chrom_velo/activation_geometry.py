"""Descriptive distance to a reference set on an unchanged peak-neighbor graph.

These distances do not assign developmental time, orient biological trajectories,
or establish that Leiden regions are branches. Both metrics use the same original
positive edges; the weighted metric penalizes weak neighbor affinities.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra


def reference_distances(graph, reference, *, weight_floor=1e-6):
    """Return nearest-reference hop and affinity-cost distances plus components.

    Reference is a bool mask. Unreachable nodes remain +inf. A reference node has
    zero distance, including an isolated reference. Input must be an undirected
    positive affinity graph with affinities at most one; it is never modified.
    """
    graph = sparse.csr_matrix(graph, copy=True)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError('Graph must be square')
    reference = np.asarray(reference, dtype=bool)
    if reference.shape != (graph.shape[0],):
        raise ValueError('Reference mask must match graph rows')
    if not 0 < weight_floor <= 1:
        raise ValueError('weight_floor must lie in (0,1]')
    graph.sum_duplicates(); graph.eliminate_zeros(); graph.sort_indices()
    if np.any(~np.isfinite(graph.data)) or np.any(graph.data <= 0) or np.any(graph.data > 1):
        raise ValueError('Graph affinities must be finite, positive, and at most one')
    delta = graph - graph.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-7:
        raise ValueError('Reference distances require a symmetric atlas graph')
    n_components, component = connected_components(graph, directed=False)
    roots = np.flatnonzero(reference)
    if len(roots):
        hops = dijkstra(graph, directed=False, indices=roots, unweighted=True, min_only=True)
        costs = graph.copy(); costs.data = 1. - np.log(np.maximum(costs.data.astype(float), weight_floor))
        weighted = dijkstra(costs, directed=False, indices=roots, min_only=True)
    else:
        hops = np.full(graph.shape[0], np.inf)
        weighted = np.full(graph.shape[0], np.inf)
    root_counts = np.bincount(component[reference], minlength=n_components)
    if not np.array_equal(np.isfinite(hops), root_counts[component] > 0):
        raise RuntimeError('Shortest-path reachability disagrees with component references')
    if not np.array_equal(np.isfinite(weighted), np.isfinite(hops)):
        raise RuntimeError('Distance metrics disagree on reachability')
    return dict(distance_hops=hops, distance_weighted=weighted,
                component=component, n_components=n_components,
                component_reference_count=root_counts)


def region_graph_totals(graph, region_codes, n_regions=None, *, block_size=32768):
    """Aggregate directed CSR entries into region-by-region counts and affinity sums.

    For a symmetric graph, an undirected cross-region edge appears in each of its
    two ordered region pairs. No edges are added or graph weights changed.
    """
    graph = sparse.csr_matrix(graph)
    codes = np.asarray(region_codes, dtype=np.int64)
    if len(codes) != graph.shape[0] or np.any(codes < 0):
        raise ValueError('Region codes must be nonnegative and match graph rows')
    k = int(codes.max()+1) if n_regions is None else int(n_regions)
    if np.any(codes >= k):
        raise ValueError('Region code exceeds supplied region count')
    counts = np.zeros(k*k, dtype=np.int64); weights = np.zeros(k*k, dtype=float)
    for start in range(0, graph.shape[0], block_size):
        stop = min(start+block_size, graph.shape[0]); a=graph.indptr[start]; b=graph.indptr[stop]
        src = np.repeat(codes[start:stop], np.diff(graph.indptr[start:stop+1]))
        pairs = src*k + codes[graph.indices[a:b]]
        counts += np.bincount(pairs, minlength=k*k)
        weights += np.bincount(pairs, weights=graph.data[a:b], minlength=k*k)
    return counts.reshape(k,k), weights.reshape(k,k)


def select_reference_core(reference, region_codes, *, coverage=.9):
    """Smallest region set covering a given fraction of existing references.

    Ties resolve by region code. Selection uses reference counts only, never
    stage, accessibility change, or UMAP geometry. Returns subset mask and codes.
    """
    reference=np.asarray(reference,dtype=bool);codes=np.asarray(region_codes,dtype=int)
    if reference.shape!=codes.shape or reference.ndim!=1 or np.any(codes<0):
        raise ValueError('Reference and nonnegative region codes must have identical 1D shape')
    if not 0<coverage<=1:raise ValueError('coverage must lie in (0,1]')
    counts=np.bincount(codes[reference],minlength=int(codes.max())+1)
    order=np.lexsort((np.arange(len(counts)),-counts))
    if not reference.any():return reference.copy(),np.array([],dtype=int)
    number=int(np.searchsorted(np.cumsum(counts[order]),coverage*reference.sum(),side='left'))+1
    chosen=order[:number]
    return reference&np.isin(codes,chosen),chosen
