"""Support-aware 2D interpolation for displaying precomputed peak directions.

This display layer does not estimate velocities or alter the peak atlas. It
averages unit directions, so streamline length/width is not biological speed.
Finite-radius support protects visible UMAP gaps; projected overlap between
unrelated high-dimensional neighborhoods remains a limitation.
"""
import numpy as np
from scipy.spatial import cKDTree


def build_stream_grid(xy, vectors, valid=None, grid_size=180, n_neighbors=50,
                      bandwidth_cells=.75, radius_cells=1.5, proximity_cells=.8,
                      min_neighbors=5, min_effective_neighbors=3.,
                      min_support_fraction=.2, min_coherence=.35):
    """Interpolate eligible unit vectors; retain ineligible peaks in support mass.

    A grid point needs nearby eligible peaks, sufficient effective sample size,
    enough eligible mass among all local peaks, and direction agreement. Both
    components are NaN outside supported grid nodes, so Matplotlib streamplot
    stops at masked interpolation cells. Counts denote peak support, not cells.
    """
    xy = np.asarray(xy, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or vectors.shape != xy.shape:
        raise ValueError('xy and vectors must both have shape (n_peaks, 2)')
    if len(xy) == 0:
        raise ValueError('At least one coordinate is required')
    for name, value in [('grid_size', grid_size), ('n_neighbors', n_neighbors), ('min_neighbors', min_neighbors)]:
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if grid_size < 3:
        raise ValueError('grid_size must be at least 3')
    for name, value in [('bandwidth_cells', bandwidth_cells), ('radius_cells', radius_cells),
                        ('proximity_cells', proximity_cells), ('min_effective_neighbors', min_effective_neighbors)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name, value in [('min_support_fraction', min_support_fraction), ('min_coherence', min_coherence)]:
        if not np.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f'{name} must be between zero and one')
    eligible = np.ones(len(xy), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if eligible.shape != (len(xy),):
        raise ValueError('valid must have one flag per peak')
    coordinate_valid = np.isfinite(xy).all(axis=1)
    if not coordinate_valid.any():
        raise ValueError('No finite coordinates')
    xy = xy[coordinate_valid]
    vectors = vectors[coordinate_valid]
    eligible = eligible[coordinate_valid].copy()
    length = np.linalg.norm(vectors, axis=1)
    eligible &= np.isfinite(vectors).all(axis=1) & (length > 1e-12)
    unit = np.divide(vectors, length[:, None], out=np.zeros_like(vectors),
                     where=eligible[:, None])
    lo = xy.min(axis=0); span = np.ptp(xy, axis=0)
    if not (span > 0).all():
        raise ValueError('Coordinates must span both dimensions')
    padding = .01 * span.max()
    lo = lo - padding; hi = xy.max(axis=0) + padding
    step = float(np.max(hi - lo) / (int(grid_size) - 1))
    axes = [lo[d] + np.arange(int(np.ceil((hi[d] - lo[d]) / step)) + 1, dtype=float) * step for d in range(2)]
    x, y = axes
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    k = min(int(n_neighbors), len(xy))
    distances, indices = cKDTree(xy).query(points, k=k, workers=1)
    if k == 1:
        distances = distances[:, None]; indices = indices[:, None]
    within = distances <= radius_cells * step
    weights = np.exp(-.5 * (distances / (bandwidth_cells * step)) ** 2) * within
    supported = eligible[indices] & within
    active_weights = weights * supported
    mass = active_weights.sum(axis=1)
    total_mass = weights.sum(axis=1)
    fraction = np.divide(mass, total_mass, out=np.zeros_like(mass), where=total_mass > 0)
    effective = np.divide(mass ** 2, (active_weights ** 2).sum(axis=1),
                          out=np.zeros_like(mass), where=(active_weights ** 2).sum(axis=1) > 0)
    mean = np.sum(active_weights[:, :, None] * unit[indices], axis=1)
    mean = np.divide(mean, mass[:, None], out=np.zeros_like(mean), where=mass[:, None] > 0)
    coherence = np.linalg.norm(mean, axis=1)
    count = supported.sum(axis=1)
    nearest = np.min(np.where(supported, distances, np.inf), axis=1)
    mask = ((count >= min_neighbors) & (effective >= min_effective_neighbors)
            & (fraction >= min_support_fraction) & (coherence >= min_coherence)
            & (coherence > 1e-12) & (nearest <= proximity_cells * step))
    shape = xx.shape
    out = dict(x=x, y=y, u=np.where(mask, mean[:, 0], np.nan).reshape(shape),
               v=np.where(mask, mean[:, 1], np.nan).reshape(shape), valid=mask.reshape(shape),
               coherence=coherence.reshape(shape), eligible_count=count.reshape(shape),
               effective_n=effective.reshape(shape), support_fraction=fraction.reshape(shape),
               nearest_eligible_distance=nearest.reshape(shape))
    out['metadata'] = dict(grid_size=int(grid_size), n_neighbors=int(n_neighbors),
        actual_neighbors=k, bandwidth_cells=float(bandwidth_cells), radius_cells=float(radius_cells),
        proximity_cells=float(proximity_cells), min_neighbors=int(min_neighbors),
        min_effective_neighbors=float(min_effective_neighbors), min_support_fraction=float(min_support_fraction),
        min_coherence=float(min_coherence), grid_step=step, grid_shape=list(shape),
        input_peaks=int(len(coordinate_valid)), finite_coordinate_peaks=int(len(xy)),
        eligible_peaks=int(eligible.sum()), supported_grid_nodes=int(mask.sum()),
        grid_nodes=int(mask.size), direction_normalized_before_smoothing=True,
        note='Display interpolation only; visible-gap support masks do not guarantee separation of distinct programs that overlap in 2D')
    return out
