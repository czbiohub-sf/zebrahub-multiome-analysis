"""Root-free projection of stage-specific accessibility changes onto a fixed atlas.

The atlas points are genomic peaks, not moving particles. Edges connect different
peaks whose current accessibility profiles align with a source peak's fitted
change. No root, specificity score, inferred peak ancestry, or RNA model enters.
Unlike scVelo's cosine correlation, neither velocities nor state differences are
centered across celltypes: uniform opening/closing remains observable.
"""
import numpy as np
from scipy import sparse


def project_interval(graph, state, velocity, feature_valid=None, embeddings=None, *,
                     feature_weights=None, source_active=None, block_size=1024,
                     min_shared_features=5, min_shared_fraction=.7,
                     min_cosine=.2, cosine_power=1., min_neighbors=2,
                     min_coherence=.25, velocity_floor=0.,
                     state_difference_floor=1e-12, return_probabilities=True):
    """Project one interval using uncentered cosine on shared celltype features.

    State and velocity are matching peaks-by-celltypes matrices in the same
    feature scaling (velocity has an additional per-time unit). ``feature_valid``
    marks supported features; measured zero state/velocity entries remain valid.
    A source also requires finite velocity. A target only needs a finite state;
    its own velocity is irrelevant to serving as a state-space neighbor.

    Shared support must reach max(min_shared_features, ceil(C*fraction)), where C
    is the FULL supplied celltype panel, never the number of changing features.
    Optional positive feature_weights are fixed across peaks; equal biological
    celltype weights are the default. We never infer weights from cell counts.
    ``source_active`` optionally supplies an external fit/effect stability gate;
    it does not prohibit a filtered peak from being a target.

    Retain nonself existing edges with nonzero state displacement and shared
    velocity, cosine>0 and cosine>=min_cosine. Weights are graph_weight*cosine**p,
    row-normalized after requiring min_neighbors. No geometry baseline is
    subtracted: each raw arrow is the probability-weighted mean of unit embedding
    directions to retained neighbors. A separate baseline_vectors diagnostic
    reports the unfiltered graph's geometric tendency. Geometry can still bias
    a projected field; an arrow is not a uniquely identified transport velocity.

    vectors are dimensionless raw averages; coherence is their norm. Input
    velocity norm, weighted RMS and signed net rate preserve amplitude separately.
    display_vectors zero low-coherence/low-support rows; projection_valid supplies
    that display mask per embedding. active means adequate high-dimensional edge
    support; unprojectable distinguishes changing peaks lacking any valid supplied
    embedding projection (or lacking edge support if no embeddings were given).
    No suitable neighbor, not necessarily no biological change, can mean no arrow.

    Intermediate edge-feature arrays are bounded by CSR row blocks. The optional
    sparse probability output is O(retained edges); no dense peaks-by-peaks array
    is made. Coordinates, states, velocities, masks and input graph are unchanged.
    """
    if not sparse.issparse(graph):
        raise TypeError('graph must be sparse')
    state = np.asarray(state, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if state.ndim != 2 or state.shape != velocity.shape or state.shape[1] < 1:
        raise ValueError('state and velocity must have matching peaks-by-features shapes')
    n, c = state.shape
    if graph.shape != (n, n):
        raise ValueError('graph and matrices must share an ordered peak axis')
    for name, value in [('block_size', block_size), ('min_shared_features', min_shared_features),
                        ('min_neighbors', min_neighbors)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name, value in [('min_shared_fraction', min_shared_fraction), ('min_cosine', min_cosine),
                        ('min_coherence', min_coherence)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    for name, value in [('velocity_floor', velocity_floor), ('state_difference_floor', state_difference_floor)]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not np.isfinite(cosine_power) or cosine_power <= 0:
        raise ValueError('cosine_power must be finite and positive')
    mask = np.ones(state.shape, dtype=bool) if feature_valid is None else np.asarray(feature_valid, dtype=bool)
    if mask.shape != state.shape:
        raise ValueError('feature_valid must match state shape')
    weights = np.ones(c) if feature_weights is None else np.asarray(feature_weights, dtype=float)
    if weights.shape != (c,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('feature_weights must be one finite positive weight per feature')
    enabled = np.ones(n, dtype=bool) if source_active is None else np.asarray(source_active, dtype=bool)
    if enabled.shape != (n,):
        raise ValueError('source_active must have one flag per peak')
    state_valid = mask & np.isfinite(state)
    velocity_valid = state_valid & np.isfinite(velocity)
    feature_count = velocity_valid.sum(axis=1)
    required = max(int(min_shared_features), int(np.ceil(c * min_shared_fraction)))
    safe_velocity = np.where(velocity_valid, velocity, 0.)
    norm = np.sqrt(np.sum(safe_velocity**2 * weights, axis=1))
    weight_sum = np.sum(velocity_valid * weights, axis=1)
    source_supported = (feature_count >= required) & np.isfinite(norm)
    rms = np.sqrt(np.divide(norm**2, weight_sum, out=np.zeros(n), where=weight_sum > 0))
    net = np.divide(np.sum(safe_velocity * weights, axis=1), weight_sum,
                    out=np.full(n, np.nan), where=weight_sum > 0)
    changing = source_supported & enabled & (norm > velocity_floor)
    graph = graph.tocsr(copy=False)
    if not graph.has_canonical_format:
        graph = graph.copy(); graph.sum_duplicates()
    if not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError('graph weights must be finite and nonnegative')
    embeddings = {} if embeddings is None else embeddings
    coordinates = {}
    for name, xy in embeddings.items():
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or len(xy) != n or xy.shape[1] not in (2, 3) or not np.isfinite(xy).all():
            raise ValueError('embeddings must be finite, ordered peak-by-2 or peak-by-3 matrices')
        coordinates[name] = xy
    result = dict(source_supported=source_supported, stationary=source_supported & (norm <= velocity_floor),
        changing=changing, source_filtered=source_supported & ~enabled,
        source_feature_count=feature_count, velocity_norm=norm, velocity_rms=rms, net_rate=net,
        active=np.zeros(n, dtype=bool), support_count=np.zeros(n, dtype=np.int64),
        comparable_count=np.zeros(n, dtype=np.int64), support_fraction=np.zeros(n),
        mean_cosine=np.zeros(n), mean_shared_features=np.zeros(n),
        vectors={k:np.zeros_like(xy) for k, xy in coordinates.items()},
        baseline_vectors={k:np.zeros_like(xy) for k, xy in coordinates.items()},
        coherence={k:np.zeros(n) for k in coordinates},
        projection_support={k:np.zeros(n) for k in coordinates},
        projection_neighbor_count={k:np.zeros(n, dtype=np.int64) for k in coordinates})
    pointer = np.zeros(n+1, dtype=np.int64) if return_probabilities else None
    data_parts, index_parts, emitted = [], [], 0
    for start in range(0, n, block_size):
        stop = min(n, start+block_size); size = stop-start
        left, right = graph.indptr[start], graph.indptr[stop]
        row = np.repeat(np.arange(size), np.diff(graph.indptr[start:stop+1]))
        col = graph.indices[left:right]; src = row+start
        graph_weight = graph.data[left:right].astype(float, copy=False)
        actual = (src != col) & (graph_weight > 0)
        base = np.bincount(row[actual], weights=graph_weight[actual], minlength=size)
        shared = velocity_valid[src] & state_valid[col]
        count = shared.sum(axis=1)
        difference = np.where(shared, state[col]-state[src], 0.)
        v = np.where(shared, velocity[src], 0.)
        dnorm = np.sqrt(np.sum(difference**2 * weights, axis=1))
        vnorm = np.sqrt(np.sum(v**2 * weights, axis=1))
        denominator = dnorm * vnorm
        comparable = (actual & changing[src] & (count >= required)
                      & (dnorm > state_difference_floor) & (vnorm > 0) & np.isfinite(denominator))
        cosine = np.divide(np.sum(difference * v * weights, axis=1), denominator,
                           out=np.zeros(len(row)), where=comparable & (denominator > 0))
        cosine = np.clip(cosine, -1., 1.)
        keep = comparable & (cosine > 0) & (cosine >= min_cosine)
        support_count = np.bincount(row[keep], minlength=size)
        result['support_count'][start:stop] = support_count
        result['comparable_count'][start:stop] = np.bincount(row[comparable], minlength=size)
        selected_base = np.bincount(row[keep], weights=graph_weight[keep], minlength=size)
        result['support_fraction'][start:stop] = np.divide(selected_base, base, out=np.zeros(size), where=base > 0)
        keep &= support_count[row] >= min_neighbors
        r, j = row[keep], col[keep]
        directed = graph_weight[keep] * cosine[keep]**cosine_power
        total = np.bincount(r, weights=directed, minlength=size)
        probability = np.divide(directed, total[r], out=np.zeros_like(directed), where=total[r] > 0)
        positive = probability > 0
        r, j, probability = r[positive], j[positive], probability[positive]
        retained_cosine = cosine[keep][positive]; retained_shared = count[keep][positive]
        result['active'][start:stop] = total > 0
        result['mean_cosine'][start:stop] = np.bincount(r, weights=probability*retained_cosine, minlength=size)
        result['mean_shared_features'][start:stop] = np.bincount(r, weights=probability*retained_shared, minlength=size)
        if return_probabilities:
            pointer[start+1:stop+1] = emitted + np.cumsum(np.bincount(r, minlength=size))
            emitted += len(probability)
            data_parts.append(probability); index_parts.append(j.copy())
        for name, xy in coordinates.items():
            delta = xy[j]-xy[r+start]
            distance = np.linalg.norm(delta, axis=1)
            unit = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
            for dimension in range(xy.shape[1]):
                result['vectors'][name][start:stop, dimension] = np.bincount(r, weights=probability*unit[:, dimension], minlength=size)
            result['coherence'][name][start:stop] = np.linalg.norm(result['vectors'][name][start:stop], axis=1)
            result['projection_support'][name][start:stop] = np.bincount(r, weights=probability*(distance > 0), minlength=size)
            result['projection_neighbor_count'][name][start:stop] = np.bincount(r[distance > 0], minlength=size)
            delta_base = xy[col[actual]]-xy[src[actual]]
            distance_base = np.linalg.norm(delta_base, axis=1)
            unit_base = np.divide(delta_base, distance_base[:, None], out=np.zeros_like(delta_base), where=distance_base[:, None] > 0)
            base_probability = graph_weight[actual] / base[row[actual]]
            for dimension in range(xy.shape[1]):
                result['baseline_vectors'][name][start:stop, dimension] = np.bincount(row[actual], weights=base_probability*unit_base[:, dimension], minlength=size)
    result['projection_valid'] = {name: result['active'] & (result['coherence'][name] >= min_coherence)
        & (result['coherence'][name] > 1e-12) & (result['projection_neighbor_count'][name] >= min_neighbors)
        for name in coordinates}
    result['display_vectors'] = {name: np.where(result['projection_valid'][name][:, None], result['vectors'][name], 0.) for name in coordinates}
    projected = np.logical_or.reduce(list(result['projection_valid'].values())) if coordinates else result['active']
    result['unprojectable'] = changing & ~projected
    result['neighbor_unprojectable'] = changing & ~result['active']
    if return_probabilities:
        data = np.concatenate(data_parts) if data_parts else np.empty(0)
        indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=graph.indices.dtype)
        result['probabilities'] = sparse.csr_matrix((data, indices, pointer), shape=(n, n))
    result['metadata'] = dict(required_shared_features=required, n_features=c,
        min_shared_fraction=float(min_shared_fraction), min_shared_features=int(min_shared_features),
        min_cosine=float(min_cosine), cosine_power=float(cosine_power), min_neighbors=int(min_neighbors),
        min_coherence=float(min_coherence), velocity_floor=float(velocity_floor),
        state_difference_floor=float(state_difference_floor), block_size=int(block_size),
        cosine='uncentered weighted cosine on shared supported celltypes',
        shared_fraction_denominator='full supplied celltype panel', geometric_baseline_subtracted=False,
        vector_units='probability-weighted unit embedding directions; accessibility rates reported separately',
        interpretation='Local correspondence between fixed genomic peaks, not motion, ancestry, or identified transport')
    return result
