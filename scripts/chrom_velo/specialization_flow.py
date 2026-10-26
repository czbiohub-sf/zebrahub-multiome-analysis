"""Celltype specificity and structural directions on a fixed peak atlas.

The structural field orients an existing high-dimensional accessibility graph
by increasing specificity. It is an explicit ordering of *different peaks*, not
an estimate of peak motion, ancestry, transport, or future genomic positions.
Temporal specificity rates are separate measurements. Mapping a signed rate
onto a structural direction is an optional, explicitly assumed display rule.
"""

import numpy as np
from scipy import sparse
from scipy.special import xlogy
from sklearn.neighbors import NearestNeighbors


def _checked_times(times, n_stages):
    times = np.asarray(times, dtype=float)
    if (times.shape != (n_stages,) or not np.isfinite(times).all()
            or n_stages < 2 or np.any(np.diff(times) <= 0)):
        raise ValueError('Times must match stages and be finite, strictly increasing')
    return times


def _prepare_accessibility(accessibility, observed):
    a = np.asarray(accessibility, dtype=float)
    if a.ndim != 3 or a.shape[1] < 2 or a.shape[2] < 1:
        raise ValueError('Accessibility must have shape peaks x celltypes x stages')
    if observed is None:
        mask = np.ones(a.shape, dtype=bool)
    else:
        try:
            mask = np.broadcast_to(np.asarray(observed, dtype=bool), a.shape).copy()
        except ValueError as exc:
            raise ValueError('Observed mask must broadcast to accessibility shape') from exc
    # A missing celltype is excluded; its inaccessible-looking placeholder is
    # never counted as zero. Explicitly unobserved negatives are also ignored.
    mask &= np.isfinite(a)
    if np.any(a[mask] < 0):
        raise ValueError('Specificity requires nonnegative, unscaled accessibility')
    return a, mask


def _specificity_score(a, mask, signal_floor, pseudocount, shrinkage, min_celltypes, method):
    """Specificity along celltype axis 1; all other axes are preserved."""
    if method not in ('entropy', 'tau'):
        raise ValueError("method must be entropy or tau")
    if not np.isfinite(signal_floor) or signal_floor < 0:
        raise ValueError('signal_floor must be finite and nonnegative')
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError('pseudocount must be finite and nonnegative')
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError('shrinkage must lie in [0, 1]')
    if not isinstance(min_celltypes, (int, np.integer)) or min_celltypes < 2:
        raise ValueError('min_celltypes must be an integer >= 2')
    counts = mask.sum(axis=1)
    observed_a = np.where(mask, a, 0.)
    total = observed_a.sum(axis=1)
    mean = np.divide(total, counts, out=np.full(total.shape, np.nan), where=counts > 0)
    valid = (counts >= min_celltypes) & (total > 0) & (mean >= signal_floor)
    adjusted = np.where(mask, observed_a + pseudocount, 0.)
    denominator = adjusted.sum(axis=1, keepdims=True)
    probabilities = np.divide(adjusted, denominator, out=np.zeros_like(adjusted),
                              where=denominator > 0)
    uniform = np.divide(mask.astype(float), counts[:, None, :],
                        out=np.zeros_like(adjusted), where=counts[:, None, :] > 0)
    probabilities = (1. - shrinkage) * probabilities + shrinkage * uniform
    if method == 'entropy':
        entropy = -xlogy(probabilities, probabilities).sum(axis=1)
        max_entropy = np.log(np.maximum(counts, 2))
        raw_score = 1. - entropy / max_entropy
    else:
        # Tau uses relative accessibility compared with the most accessible
        # observed celltype. Normalizing by total first cancels from this ratio.
        # The same additive pseudocount/uniform shrinkage are applied explicitly
        # before max-normalization, keeping their stabilizing meaning inspectable.
        maximum = probabilities.max(axis=1, keepdims=True)
        relative = np.divide(probabilities, maximum, out=np.zeros_like(probabilities),
                             where=maximum > 0)
        raw_score = np.where(mask, 1. - relative, 0.).sum(axis=1) / np.maximum(counts - 1, 1)
    score = np.where(valid, np.clip(raw_score, 0., 1.), np.nan)
    return dict(score=score, valid=valid, mean_accessibility=mean, n_celltypes=counts)


def specificity_by_stage(accessibility, observed=None, signal_floor=0.02,
                         pseudocount=0., shrinkage=0., min_celltypes=2, method='entropy'):
    """Entropy or tau celltype specificity for each peak at each stage.

    ``accessibility`` is nonnegative (peaks, celltypes, stages), in the same
    normalization across conditions. NaNs and ``observed=False`` are missing.
    ``observed`` may have shape (celltypes, stages) or the full matrix shape.
    Each biological celltype is one category, irrespective of its cell count.

    Entropy specificity is 1 - H(p)/log(n), where p is accessibility divided by
    its sum across the n observed celltypes. Tau specificity is
    sum(1 - p/max(p))/(n - 1), counting only observed celltypes. Tau emphasizes
    restriction relative to the strongest celltype; entropy measures how the
    total accessibility is distributed. They need not rank programs identically.
    Constant broad accessibility gives 0 and accessibility confined to one
    celltype gives 1 for either method. With pseudocount=0 either method is
    invariant to multiplying accessibility by a positive constant, provided
    the unmodified signal still passes ``signal_floor``.

    ``signal_floor`` applies to mean observed accessibility *before* adding a
    pseudocount; an all-zero peak is always invalid. ``pseudocount`` is in input
    accessibility units. ``shrinkage`` mixes p with the observed uniform
    distribution and preserves scale invariance when pseudocount=0.

    Stage scores with different observed celltype sets are descriptive only;
    use temporal_specificity for paired, common-celltype temporal differences.
    """
    a, mask = _prepare_accessibility(accessibility, observed)
    return _specificity_score(a, mask, signal_floor, pseudocount, shrinkage, min_celltypes, method)


def temporal_specificity(accessibility, times, observed=None, signal_floor=0.02,
                         pseudocount=0., shrinkage=0., min_celltypes=2, method='entropy'):
    """Adjacent-stage specificity rates using matched observed celltypes.

    Both endpoint scores (entropy or tau) are recomputed on the intersection of observed
    celltypes for that peak and interval. No absent stage is bridged. A rate
    requires adequate signal at both endpoints and at least min_celltypes in
    common. Rates are NaN, not zero, when this requirement fails. There is no
    terminal-stage placeholder in the returned (peaks, stages - 1) rates.

    Returns stage summaries plus interval_rate, interval_valid, paired_start,
    paired_end, paired_celltypes, and dt. Positive rate means increasing
    restriction across the compared celltypes, not necessarily opening.
    """
    a, mask = _prepare_accessibility(accessibility, observed)
    times = _checked_times(times, a.shape[2])
    options = (signal_floor, pseudocount, shrinkage, min_celltypes, method)
    result = _specificity_score(a, mask, *options)
    common = mask[:, :, :-1] & mask[:, :, 1:]
    first = _specificity_score(a[:, :, :-1], common, *options)
    second = _specificity_score(a[:, :, 1:], common, *options)
    valid = first['valid'] & second['valid']
    dt = np.diff(times)
    result.update(
        interval_rate=np.where(valid, (second['score'] - first['score']) / dt, np.nan),
        interval_valid=valid,
        paired_start=first['score'],
        paired_end=second['score'],
        paired_celltypes=common.sum(axis=1),
        dt=dt,
    )
    return result


def aggregate_specificity(scores, times, valid=None, weighting='equal', late_power=2.):
    """Average stage specificity with an explicit, inspectable stage weighting.

    equal: equal contribution from each scheduled stage.
    duration (or trapezoid): trapezoidal integration weights / elapsed time.
    late: weights proportional to (1 + normalized_time)**late_power; default
          makes the final stage weigh four times as much as the first.
    terminal: only the final scheduled stage; a missing/invalid final stage
              yields an invalid aggregate, never a fallback to an earlier stage.

    Invalid stages are excluded and remaining weights renormalized per peak;
    effective_weight reports the retained fraction of the full weight. The
    aggregate is a descriptive specialization axis, not a developmental clock.
    With differing stage celltype support, compare matched-support aggregates
    separately rather than assuming all entropy normalizations are equivalent.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError('Scores must have shape peaks x stages')
    times = _checked_times(times, scores.shape[1])
    good = np.isfinite(scores)
    if valid is not None:
        if np.asarray(valid).shape != scores.shape:
            raise ValueError('Valid mask must match score shape')
        good &= np.asarray(valid, dtype=bool)
    if np.any((scores[good] < -1e-12) | (scores[good] > 1 + 1e-12)):
        raise ValueError('Specificity scores must lie in [0, 1]')
    if weighting == 'equal':
        weights = np.ones(len(times))
    elif weighting in ('duration', 'trapezoid'):
        dt = np.diff(times)
        weights = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    elif weighting == 'terminal':
        weights = np.zeros(len(times))
        weights[-1] = 1.
    elif weighting == 'late':
        if not np.isfinite(late_power) or late_power < 0:
            raise ValueError('late_power must be finite and nonnegative')
        weights = (1. + (times - times[0]) / (times[-1] - times[0])) ** late_power
    else:
        raise ValueError('weighting must be equal, duration, trapezoid, late, or terminal')
    weights /= weights.sum()
    retained = good @ weights
    aggregate = np.divide(np.where(good, scores, 0.) @ weights, retained,
                          out=np.full(len(scores), np.nan), where=retained > 0)
    return dict(aggregate=aggregate, valid=retained > 0,
                stage_weights=weights, effective_weight=retained)


def build_state_graph(states, n_neighbors=30, metric='euclidean'):
    """Symmetric, sparse neighbor weights from supplied accessibility/PCA states.

    This function never reads UMAP coordinates. A supplied atlas neighbor graph
    can instead go directly into structural_flow. Each row uses a Gaussian
    distance kernel with its furthest selected neighbor as bandwidth, then the
    graph is symmetrized by maximum. Self edges are excluded, including when
    duplicate state profiles tie at zero distance.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] < 1 or not np.isfinite(states).all():
        raise ValueError('States must be a finite peaks x features matrix')
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError('n_neighbors must be a positive integer')
    n = len(states)
    if n < 2:
        return sparse.csr_matrix((n, n), dtype=float)
    k = min(n_neighbors, n - 1)
    distances, indices = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(
        states).kneighbors(states)
    rows, cols, values = [], [], []
    for i, (ds, js) in enumerate(zip(distances, indices)):
        keep = js != i
        ds, js = ds[keep][:k], js[keep][:k]
        bandwidth = max(float(ds[-1]), np.finfo(float).eps)
        rows.extend([i] * len(js)); cols.extend(js)
        values.extend(np.exp(-(ds / bandwidth) ** 2))
    graph = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)
    graph.setdiag(0.)
    graph.eliminate_zeros()
    return graph


def structural_flow(graph, specificity, embeddings=None, edge_tolerance=1e-3,
                    gradient_power=1., min_support=1, score_uncertainty=None,
                    edge_confidence_z=0.):
    """Orient existing graph edges toward meaningfully higher specificity.

    Edge score = graph_weight * max(S_j - S_i - threshold_ij, 0)**gradient_power.
    By default threshold_ij is edge_tolerance. If score_uncertainty is supplied,
    threshold_ij = edge_tolerance + edge_confidence_z * sqrt(SE_i**2 + SE_j**2).
    This is an empirical uncertainty margin, not a formal confidence interval;
    it does not model dependence between peaks or graph/embedding uncertainty.
    Supplied uncertainty must have one value per peak. A vertex with nonfinite
    or negative uncertainty cannot send/receive transitions. Positive z requires
    uncertainty; z=0 with finite nonnegative uncertainties reproduces the default.
    Scores are normalized within each source row to directed probabilities.
    Rows with fewer than min_support qualifying neighbors have no transitions.
    Neither self edges, new graph connections, nor plateau directions are added.
    Invalid specificity scores cannot send or receive transitions.

    For each supplied embedding, ``vectors`` is the raw probability-weighted
    average of unit directions to selected neighbors; its norm (``coherence``)
    lies in [0, 1]. ``displacement_vectors`` averages unnormalized displacements.
    No display length scaling is applied. A zero vector can indicate opposing
    directions in that embedding even when graph transitions are supported.
    Embeddings determine the displayed direction, never the neighborhood or
    edge orientation. Coordinate dimensions may be 2 or 3 (or any positive d).

    support_count counts qualifying outgoing neighbors before min_support is
    applied. support_fraction is their fraction of the source's positive graph
    weight, including neighbors with invalid specificity in the denominator.
    It is a support diagnostic, not statistical confidence.
    """
    if not sparse.issparse(graph):
        raise TypeError('Supply a sparse high-dimensional neighbor graph')
    specificity = np.asarray(specificity, dtype=float)
    if specificity.ndim != 1 or graph.shape != (len(specificity), len(specificity)):
        raise ValueError('Graph and ordered specificity vector must align')
    if not np.isfinite(edge_tolerance) or edge_tolerance < 0:
        raise ValueError('edge_tolerance must be finite and nonnegative')
    if not np.isfinite(gradient_power) or gradient_power <= 0:
        raise ValueError('gradient_power must be finite and positive')
    if not isinstance(min_support, (int, np.integer)) or min_support < 1:
        raise ValueError('min_support must be a positive integer')
    if not np.isfinite(edge_confidence_z) or edge_confidence_z < 0:
        raise ValueError('edge_confidence_z must be finite and nonnegative')
    if score_uncertainty is None:
        if edge_confidence_z > 0:
            raise ValueError('Positive edge_confidence_z requires score_uncertainty')
        uncertainty_valid = np.ones(len(specificity), dtype=bool)
        uncertainty = None
    else:
        uncertainty = np.asarray(score_uncertainty, dtype=float)
        if uncertainty.shape != specificity.shape:
            raise ValueError('Score uncertainty must have one value per peak')
        uncertainty_valid = np.isfinite(uncertainty) & (uncertainty >= 0)
    graph = graph.tocsr(copy=True).astype(float)
    graph.sum_duplicates()
    if not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError('Graph weights must be finite and nonnegative')
    graph.setdiag(0.)
    graph.eliminate_zeros()
    base_support = np.asarray(graph.sum(axis=1)).ravel()
    edge = graph.tocoo()
    delta = specificity[edge.col] - specificity[edge.row]
    threshold = np.full(delta.shape, edge_tolerance, dtype=float)
    supported = uncertainty_valid[edge.row] & uncertainty_valid[edge.col]
    if uncertainty is not None and edge_confidence_z > 0:
        # hypot is stable for extreme magnitudes and invalid vertices are
        # excluded before accessing their uncertainty values.
        threshold[supported] += edge_confidence_z * np.hypot(
            uncertainty[edge.row[supported]], uncertainty[edge.col[supported]])
    keep = np.isfinite(delta) & supported & (delta > threshold)
    row, col, gain = edge.row[keep], edge.col[keep], delta[keep]
    weights = edge.data[keep] * (gain - threshold[keep]) ** gradient_power
    n = len(specificity)
    support_count = np.bincount(row, minlength=n)
    directed_base = np.bincount(row, weights=edge.data[keep], minlength=n)
    support_fraction = np.divide(directed_base, base_support,
                                 out=np.zeros(n), where=base_support > 0)
    active = support_count >= min_support
    selected = active[row]
    directed = sparse.csr_matrix((weights[selected], (row[selected], col[selected])),
                                 shape=graph.shape)
    row_total = np.asarray(directed.sum(axis=1)).ravel()
    inv_total = np.divide(1., row_total, out=np.zeros(n), where=row_total > 0)
    probabilities = (sparse.diags(inv_total) @ directed).tocsr()
    transitions = probabilities.tocoo()
    expected_gain = np.bincount(
        transitions.row,
        weights=transitions.data * (specificity[transitions.col] - specificity[transitions.row]),
        minlength=n)
    result = dict(probabilities=probabilities, support_count=support_count,
                  support_fraction=support_fraction, active=row_total > 0,
                  expected_specificity_gain=expected_gain,
                  vectors={}, coherence={}, displacement_vectors={}, projection_support={})
    if embeddings is None:
        embeddings = {}
    elif isinstance(embeddings, np.ndarray):
        embeddings = {'embedding': embeddings}
    for name, xy in embeddings.items():
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or len(xy) != n or not np.isfinite(xy).all():
            raise ValueError('Each embedding must be a finite aligned peaks x dimensions matrix')
        displacement = xy[transitions.col] - xy[transitions.row]
        distance = np.linalg.norm(displacement, axis=1)
        unit = np.divide(displacement, distance[:, None], out=np.zeros_like(displacement),
                         where=distance[:, None] > 0)
        vectors = np.zeros_like(xy)
        raw_displacement = np.zeros_like(xy)
        np.add.at(vectors, transitions.row, transitions.data[:, None] * unit)
        np.add.at(raw_displacement, transitions.row, transitions.data[:, None] * displacement)
        result['vectors'][name] = vectors
        result['displacement_vectors'][name] = raw_displacement
        result['coherence'][name] = np.linalg.norm(vectors, axis=1)
        result['projection_support'][name] = np.bincount(
            transitions.row, weights=transitions.data * (distance > 0), minlength=n)
    return result


def rate_guided_vectors(vectors, signed_rate, valid=None, min_coherence=0.,
                        coherence=None, rate_floor=0.):
    """Display signed temporal rates along a supplied structural direction.

    ASSUMPTION: increasing specificity is drawn outward along the atlas's
    aggregate-specificity direction; decreasing specificity is drawn inward.
    A scalar rate does not identify a unique spatial velocity. Returned arrows
    are this explicit visualization convention, not independently inferred
    transport. Unsupported/zero structural directions cannot acquire arrows.
    Length is abs(rate), before any external display scaling.
    """
    vectors = np.asarray(vectors, dtype=float)
    rates = np.asarray(signed_rate, dtype=float)
    if vectors.ndim != 2 or rates.shape != (len(vectors),):
        raise ValueError('Vectors and signed rate must have matching peak rows')
    if not np.isfinite(min_coherence) or not 0 <= min_coherence <= 1:
        raise ValueError('min_coherence must lie in [0, 1]')
    if not np.isfinite(rate_floor) or rate_floor < 0:
        raise ValueError('rate_floor must be finite and nonnegative')
    magnitude = np.linalg.norm(vectors, axis=1)
    score = magnitude if coherence is None else np.asarray(coherence, dtype=float)
    if score.shape != rates.shape:
        raise ValueError('Coherence must have one value per peak')
    good = (np.isfinite(vectors).all(axis=1) & np.isfinite(rates)
            & np.isfinite(score) & (score >= min_coherence) & (magnitude > 1e-12)
            & (np.abs(rates) > rate_floor))
    if valid is not None:
        if np.asarray(valid).shape != rates.shape:
            raise ValueError('Valid mask must have one value per peak')
        good &= np.asarray(valid, dtype=bool)
    output = np.zeros_like(vectors)
    output[good] = vectors[good] / magnitude[good, None] * rates[good, None]
    return output


def bidirectional_rate_guidance(upward_vectors, downward_vectors, signed_rate,
                                valid=None, upward_valid=None, downward_valid=None,
                                upward_coherence=None, downward_coherence=None,
                                min_coherence=0., rate_floor=0.):
    """Map rates onto supported uphill *or* downhill structural directions.

    Positive rates select ``upward_vectors`` from structural_flow(graph, S).
    Negative rates select ``downward_vectors`` from structural_flow(graph, -S).
    Each chosen vector is normalized, then multiplied by abs(rate). Thus a
    re-broadening peak at a local specificity maximum can point to its actual
    lower-specificity neighbors even though its uphill vector is zero. This is
    still an assumed scalar-to-spatial visualization, not inferred transport.

    ``valid`` gates all rates; ``upward_valid``/``downward_valid`` independently
    gate the corresponding structural fields (for example replicate direction
    agreement and neighbor support). Coherence gates are likewise directional.
    No supported field means no arrow. Zero or nonfinite rates stay zero.
    Inputs have shape (peaks, dimensions) and rates/masks have shape (peaks,).
    """
    upward_vectors = np.asarray(upward_vectors, dtype=float)
    downward_vectors = np.asarray(downward_vectors, dtype=float)
    rates = np.asarray(signed_rate, dtype=float)
    if upward_vectors.ndim != 2 or downward_vectors.shape != upward_vectors.shape:
        raise ValueError('Upward and downward vectors must have matching peaks x dimensions shapes')
    if rates.shape != (len(upward_vectors),):
        raise ValueError('Signed rate must have one value per peak')

    def checked_mask(mask):
        if mask is None:
            return np.ones(rates.shape, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != rates.shape:
            raise ValueError('Each validity mask must have one value per peak')
        return mask

    common = checked_mask(valid)
    upward_mask = common & checked_mask(upward_valid) & (rates > 0)
    downward_mask = common & checked_mask(downward_valid) & (rates < 0)
    positive = rate_guided_vectors(
        upward_vectors, np.abs(rates), valid=upward_mask,
        coherence=upward_coherence, min_coherence=min_coherence, rate_floor=rate_floor)
    negative = rate_guided_vectors(
        downward_vectors, np.abs(rates), valid=downward_mask,
        coherence=downward_coherence, min_coherence=min_coherence, rate_floor=rate_floor)
    return positive + negative
