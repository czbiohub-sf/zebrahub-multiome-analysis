"""Draw support-checked paths from Matplotlib's candidate streamlines.

Matplotlib can retain a final proposed vertex before its interpolation support
is checked. This display-only wrapper splits candidate polylines at unsupported
cells or local reversals, then draws fresh lines and support-checked arrowheads.
It never changes the input grid, per-peak vectors, or scientific support mask.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon


def _field(grid, points):
    """Bilinear field values and all-corner support for data-space positions."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    x, y = grid['x'], grid['y']
    dx, dy = x[1]-x[0], y[1]-y[0]
    fx, fy = (points[:, 0]-x[0])/dx, (points[:, 1]-y[0])/dy
    inside = (np.isfinite(points).all(axis=1) & (points[:, 0] >= x[0])
              & (points[:, 0] <= x[-1]) & (points[:, 1] >= y[0])
              & (points[:, 1] <= y[-1]))
    i = np.clip(np.searchsorted(x, points[:, 0], side='right')-1, 0, len(x)-1)
    j = np.clip(np.searchsorted(y, points[:, 1], side='right')-1, 0, len(y)-1)
    ii, jj = np.minimum(i+1, len(x)-1), np.minimum(j+1, len(y)-1)
    valid = grid['valid']
    supported = inside.copy()
    # At a grid boundary, keep a point only if every touching interpolation
    # cell is supported. This avoids accepting an endpoint solely because two
    # floating-point cell-index conventions round it to different sides.
    bx, by = np.rint(fx).astype(int), np.rint(fy).astype(int)
    on_x = np.abs(fx-bx) < 1e-9
    on_y = np.abs(fy-by) < 1e-9
    low_i = np.where(on_x, np.clip(bx-1, 0, len(x)-1), i)
    high_i = np.where(on_x, np.clip(bx, 0, len(x)-1), i)
    low_j = np.where(on_y, np.clip(by-1, 0, len(y)-1), j)
    high_j = np.where(on_y, np.clip(by, 0, len(y)-1), j)
    for ci in [low_i, high_i]:
        for cj in [low_j, high_j]:
            ni, nj = np.minimum(ci+1, len(x)-1), np.minimum(cj+1, len(y)-1)
            supported &= valid[cj,ci] & valid[cj,ni] & valid[nj,ci] & valid[nj,ni]
    a = np.divide(points[:,0]-x[i], x[ii]-x[i], out=np.zeros(len(points)), where=ii!=i)
    b = np.divide(points[:,1]-y[j], y[jj]-y[j], out=np.zeros(len(points)), where=jj!=j)
    a, b = np.clip(a,0,1), np.clip(b,0,1)
    values = []
    for name in ['u', 'v']:
        z = grid[name]
        values.append((1-b)*((1-a)*z[j, i]+a*z[j, ii]) + b*((1-a)*z[jj, i]+a*z[jj, ii]))
    return np.column_stack(values), supported


def _dense_path(path, grid, step):
    """Sample short pieces and insert every crossed x/y grid boundary."""
    output = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        delta = end-start
        length = np.linalg.norm(delta)
        if not np.isfinite(length) or length <= 1e-14:
            continue
        fractions = [np.linspace(0., 1., max(1, int(np.ceil(length/step)))+1)]
        for dimension, axis in enumerate([grid['x'], grid['y']]):
            if abs(delta[dimension]) > 1e-14:
                lower, upper = sorted([start[dimension], end[dimension]])
                lines = axis[np.searchsorted(axis, lower, side='right'):np.searchsorted(axis, upper, side='left')]
                fractions.append((lines-start[dimension])/delta[dimension])
        fraction = np.unique(np.concatenate(fractions))
        sampled = start+fraction[:, None]*delta
        output.extend(sampled[1:])
    return np.asarray(output)


def _clean_paths(candidates, grid, sample_step_fraction, min_direction_cosine):
    step = min(np.diff(grid['x']).min(), np.diff(grid['y']).min())*sample_step_fraction
    cleaned = []
    audit = dict(candidate_paths=len(candidates), candidate_vertices=sum(len(p) for p in candidates),
                 candidate_dense_segments=0, rejected_support_segments=0,
                 rejected_direction_segments=0, rejected_nonfinite_paths=0)
    for candidate in candidates:
        if not np.isfinite(candidate).all():
            audit['rejected_nonfinite_paths'] += 1
            continue
        dense = _dense_path(candidate, grid, step)
        if len(dense) < 2:
            continue
        start, end = dense[:-1], dense[1:]
        delta = end-start
        length = np.linalg.norm(delta, axis=1)
        nonzero = length > 1e-12*step
        support = nonzero.copy()
        direction = nonzero.copy()
        # Endpoints plus interior checks stop boundary overshoot and reversals.
        for fraction in [0., .25, .5, .75, 1.]:
            values, valid = _field(grid, start+fraction*delta)
            support &= valid
            denominator = length*np.linalg.norm(values, axis=1)
            cosine = np.divide(np.sum(delta*values, axis=1), denominator,
                               out=np.full(len(delta), -np.inf), where=denominator > 1e-15)
            direction &= np.isfinite(cosine) & (cosine >= min_direction_cosine)
        keep = support & direction
        audit['candidate_dense_segments'] += len(keep)
        audit['rejected_support_segments'] += int(np.sum(nonzero & ~support))
        audit['rejected_direction_segments'] += int(np.sum(support & ~direction))
        edges = np.diff(np.r_[False, keep, False].astype(np.int8))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            path = dense[lo:hi+1]
            if len(path) >= 2:
                cleaned.append(path)
    audit.update(clean_paths=len(cleaned), clean_vertices=sum(len(p) for p in cleaned))
    return cleaned, audit


def _point_along(path, distance):
    delta = np.diff(path, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    i = min(int(np.searchsorted(cumulative, distance, side='right')-1), len(delta)-1)
    i = max(i, 0)
    fraction = (distance-cumulative[i])/max(lengths[i], 1e-15)
    return path[i]+fraction*delta[i], delta[i]/max(lengths[i], 1e-15)


def _arrow_polygon(path, grid, arrowsize):
    """Use a small data-space head whose full boundary remains supported."""
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    cell = min(np.diff(grid['x']).min(), np.diff(grid['y']).min())
    head = min(.6*cell*arrowsize, .2*length)
    if head < .06*cell or arrowsize <= 0:
        return None
    for position in [.6, .45, .75, .3]:
        tip, tangent = _point_along(path, position*length)
        normal = np.array([-tangent[1], tangent[0]])
        base = tip-head*tangent
        triangle = np.array([tip, base+.34*head*normal, base-.34*head*normal])
        edge_points = _dense_path(np.vstack([triangle, triangle[0]]), grid, .05*cell)
        # Checking edge interiors catches a narrow masked cell at a triangle tip.
        edge_midpoints = (edge_points[:-1]+edge_points[1:])/2
        _, valid = _field(grid, np.vstack([edge_points, edge_midpoints, triangle.mean(axis=0)]))
        if valid.all():
            return triangle
    return None


def draw_supported_streams(ax, grid, **drawing):
    """Return cleaned paths, artists, and an audit of display-only pruning.

    The normal ``Axes.streamplot`` style arguments are accepted. This renderer
    expects a single line color and scalar linewidth. Extra wrapper controls:
    ``seed_limit`` (2000), ``seed`` (1729), ``sample_step_fraction`` (.1), and
    ``min_direction_cosine`` (.1). Explicit user start_points are replaced by
    deterministic valid-cell centers so disconnected supported islands are tried.
    Returned ``lines`` supports ordinary Matplotlib path effects; ``arrows`` are
    Polygon heads in data coordinates, independent of later figure resizing.
    """
    options = dict(drawing)
    seed_limit = int(options.pop('seed_limit', 2000))
    seed = int(options.pop('seed', 1729))
    fraction = float(options.pop('sample_step_fraction', .1))
    min_cosine = float(options.pop('min_direction_cosine', .1))
    if seed_limit < 1 or not 0 < fraction <= .1 or not 0 <= min_cosine <= 1:
        raise ValueError('Invalid seed limit, sample spacing, or direction threshold')
    g = {name:np.asarray(grid[name]) for name in ['x', 'y', 'u', 'v', 'valid']}
    g['valid'] = g['valid'].astype(bool)
    if any(len(g[axis]) < 2 or not (np.diff(g[axis]) > 0).all() for axis in ['x', 'y']):
        raise ValueError('Need increasing regular grid axes')
    if any(not np.allclose(np.diff(g[axis]), np.diff(g[axis])[0]) for axis in ['x','y']):
        raise ValueError('Streamline axes must be regularly spaced')
    shape = (len(g['y']), len(g['x']))
    if any(g[key].shape != shape for key in ['u','v','valid']):
        raise ValueError('Field and mask shapes must match the grid axes')
    if not np.isfinite(g['u'][g['valid']]).all() or not np.isfinite(g['v'][g['valid']]).all():
        raise ValueError('Supported grid values must be finite')
    options.setdefault('density', 3.)
    options.setdefault('minlength', .005)
    options.setdefault('maxlength', .65)
    options.setdefault('integration_direction', 'both')
    options.setdefault('broken_streamlines', True)
    color = options.get('color', '#152c40')
    width = options.get('linewidth', .8)
    if not isinstance(color, str) or not np.isscalar(width):
        raise ValueError('Clean stream rendering requires a single color and scalar linewidth')
    zorder = options.get('zorder', 4)
    valid = g['valid']
    cells = valid[:-1,:-1] & valid[:-1,1:] & valid[1:,:-1] & valid[1:,1:]
    iy, ix = np.where(cells)
    centers = np.column_stack([(g['x'][ix]+g['x'][ix+1])/2, (g['y'][iy]+g['y'][iy+1])/2])
    order = np.random.default_rng(seed).permutation(len(centers))[:seed_limit]
    centers = centers[order]
    candidates = []
    if len(centers):
        old_patches = {id(p) for p in ax.patches}
        old_collections = {id(c) for c in ax.collections}
        options['start_points'] = centers
        try:
            native = ax.streamplot(g['x'], g['y'],
                np.ma.array(g['u'], mask=~valid), np.ma.array(g['v'], mask=~valid), **options)
            candidates = [np.asarray(p).copy() for p in native.lines.get_segments() if len(p) >= 2]
        finally:
            # Matplotlib adds individual FancyArrowPatches; its returned arrow
            # collection alone is not sufficient to remove them from the axes.
            for patch in list(ax.patches):
                if id(patch) not in old_patches:
                    patch.remove()
            for collection in list(ax.collections):
                if id(collection) not in old_collections:
                    collection.remove()
    paths, audit = _clean_paths(candidates, g, fraction, min_cosine)
    lines = LineCollection(paths, colors=color, linewidths=float(width), zorder=zorder)
    ax.add_collection(lines, autolim=False)
    arrows = []
    for path in paths:
        triangle = _arrow_polygon(path, g, float(options.get('arrowsize', 1.)))
        if triangle is not None:
            patch = Polygon(triangle, closed=True, facecolor=color, edgecolor=color,
                            linewidth=0., zorder=zorder)
            ax.add_patch(patch); arrows.append(patch)
    audit.update(seed_policy='deterministic permutation of fully supported cell centers',
        seed=seed, seed_limit=seed_limit, candidate_seed_cells=int(cells.sum()),
        used_seed_points=len(centers), density=options['density'], minlength=options['minlength'],
        sample_step_fraction=fraction, min_direction_cosine=min_cosine,
        drawn_arrowheads=len(arrows), support_mask_unchanged=True,
        arrowhead_policy='small data-coordinate triangles; all edges checked against interpolation support')
    return dict(paths=paths, audit=audit, lines=lines, arrows=arrows)
