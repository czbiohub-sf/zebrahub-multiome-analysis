"""Independently audit exported Matplotlib streamline paths and support grids.

The verifier consumes saved artifacts rather than rerunning the smoothing model.
It samples inside every drawn line segment, checks the four interpolation nodes,
and compares ordered segment tangents with independently interpolated vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import label

DEFAULT_OUT = Path(__file__).resolve().parent / 'outputs/real_specialization_streams_v1'


def _interpolate(grid, points):
    """Return bilinear vectors and mask support, including domain boundaries."""
    x, y = grid['x'], grid['y']
    dx, dy = np.diff(x), np.diff(y)
    if not (len(x) >= 2 and len(y) >= 2 and (dx > 0).all() and (dy > 0).all()):
        raise ValueError('Grid axes must be increasing and nonempty')
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
        raise ValueError('Expected the regular grid used by streamplot')
    fx = (points[:, 0] - x[0]) / dx[0]
    fy = (points[:, 1] - y[0]) / dy[0]
    tolerance = 1e-7
    inside = ((fx >= -tolerance) & (fx <= len(x)-1+tolerance)
              & (fy >= -tolerance) & (fy <= len(y)-1+tolerance)
              & np.isfinite(points).all(axis=1))
    fx = np.clip(np.nan_to_num(fx), 0, len(x)-1)
    fy = np.clip(np.nan_to_num(fy), 0, len(y)-1)
    ix = np.clip(np.searchsorted(x, points[:,0], side='right')-1, 0, len(x)-1)
    iy = np.clip(np.searchsorted(y, points[:,1], side='right')-1, 0, len(y)-1)
    jx, jy = np.minimum(ix+1, len(x)-1), np.minimum(iy+1, len(y)-1)
    tx = np.divide(points[:,0]-x[ix], x[jx]-x[ix], out=np.zeros(len(points)), where=jx!=ix)
    ty = np.divide(points[:,1]-y[iy], y[jy]-y[iy], out=np.zeros(len(points)), where=jy!=iy)
    tx, ty = np.clip(tx,0,1), np.clip(ty,0,1)
    supported = (inside & grid['valid'][iy, ix] & grid['valid'][iy, jx]
                 & grid['valid'][jy, ix] & grid['valid'][jy, jx])
    weights = np.stack([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], axis=1)
    values = []
    for component in ['u', 'v']:
        field = grid[component]
        corners = np.stack([field[iy, ix], field[iy, jx], field[jy, ix], field[jy, jx]], axis=1)
        values.append(np.sum(corners * weights, axis=1))
    return np.column_stack(values), supported, inside



def _coverage(grid, samples, supported):
    valid = grid['valid']
    cells = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    components, count = label(cells)
    occupied = np.empty(0, dtype=int)
    covered = np.empty(0, dtype=int)
    if supported.any():
        points = samples[supported]
        ix = np.clip(np.floor((points[:, 0]-grid['x'][0])/np.diff(grid['x'])[0]).astype(int), 0, cells.shape[1]-1)
        iy = np.clip(np.floor((points[:, 1]-grid['y'][0])/np.diff(grid['y'])[0]).astype(int), 0, cells.shape[0]-1)
        occupied = np.unique(iy*cells.shape[1]+ix)
        covered = np.unique(components[iy, ix]); covered = covered[covered > 0]
    return dict(supported_interpolation_cells=int(cells.sum()), supported_cell_components=int(count),
                covered_interpolation_cells=len(occupied), covered_cell_components=len(covered))


def audit_paths(grid, points, offsets, sample_step_fraction=.1):
    points = np.asarray(points, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Path points must be N x 2')
    if offsets.ndim != 1 or not len(offsets) or offsets[0] != 0 or offsets[-1] != len(points) or (np.diff(offsets) < 2).any():
        raise ValueError('Path offsets must delimit polylines of at least two points')
    shape = (len(grid['y']), len(grid['x']))
    if any(grid[key].shape != shape for key in ['u', 'v', 'valid']):
        raise ValueError('Grid arrays do not match axis lengths')
    finite_field = np.isfinite(grid['u']) & np.isfinite(grid['v'])
    if not np.array_equal(finite_field, grid['valid']):
        raise ValueError('Finite field entries and saved support mask differ')
    starts, ends = [], []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        starts.append(points[lo:hi-1]); ends.append(points[lo+1:hi])
    if not starts:
        return dict(paths=0, vertices=0, segments=0, sampled_points=0,
                    unsupported_samples=0, out_of_domain_samples=0,
                    reversed_segments=0, passed=True, **_coverage(grid, np.empty((0,2)), np.empty(0,dtype=bool)))
    starts, ends = np.concatenate(starts), np.concatenate(ends)
    delta = ends-starts
    lengths = np.linalg.norm(delta, axis=1)
    nonzero = np.isfinite(lengths) & (lengths > 1e-12)
    step = min(np.diff(grid['x']).min(), np.diff(grid['y']).min()) * sample_step_fraction
    samples = []
    for a, b, length in zip(starts[nonzero], ends[nonzero], lengths[nonzero]):
        n = max(1, int(np.ceil(length/step)))
        samples.append(a + np.linspace(0, 1, n+1)[:, None]*(b-a))
    samples = np.concatenate(samples) if samples else np.empty((0, 2))
    _, supported, inside = _interpolate(grid, samples)
    midpoint = (starts[nonzero]+ends[nonzero])/2
    uv, midpoint_support, _ = _interpolate(grid, midpoint)
    denominator = np.linalg.norm(uv, axis=1)*lengths[nonzero]
    usable = midpoint_support & np.isfinite(denominator) & (denominator > 1e-12)
    cosine = np.sum(uv[usable]*delta[nonzero][usable], axis=1)/denominator[usable]
    reversed_count = int(np.sum(cosine < -1e-3))
    bad = ~supported
    result = dict(paths=len(offsets)-1, vertices=len(points), segments=len(starts),
        nonfinite_vertices=int((~np.isfinite(points).all(axis=1)).sum()),
        zero_length_segments=int(np.sum(lengths <= 1e-12)),
        sample_step_fraction=sample_step_fraction, sampled_points=len(samples),
        unsupported_samples=int(bad.sum()), out_of_domain_samples=int((~inside).sum()),
        unsupported_example_coordinates=samples[bad][:8].tolist(),
        tangent_comparisons=int(len(cosine)), reversed_segments=reversed_count,
        tangent_cosine_min=float(np.min(cosine)) if len(cosine) else None,
        tangent_cosine_median=float(np.median(cosine)) if len(cosine) else None,
        tangent_cosine_fraction_ge08=float(np.mean(cosine >= .8)) if len(cosine) else None,
        max_segment_length_in_grid_cells=float(np.max(lengths[nonzero])/step*sample_step_fraction) if nonzero.any() else 0.)
    result.update(_coverage(grid, samples, supported))
    result['passed'] = not (result['nonfinite_vertices'] or result['unsupported_samples'] or reversed_count)
    return result



def audit_arrowheads(grid, triangles, sample_step_fraction=.1):
    triangles = np.asarray(triangles, dtype=float)
    if triangles.size == 0:
        triangles = np.empty((0,3,2))
    if triangles.ndim != 3 or triangles.shape[1:] != (3,2):
        raise ValueError('Arrowheads must be A x 3 x 2 triangles')
    step = min(np.diff(grid['x']).min(), np.diff(grid['y']).min())*sample_step_fraction
    unsupported, reversed_count, samples_total = 0, 0, 0
    for triangle in triangles:
        if not np.isfinite(triangle).all():
            unsupported += 1
            continue
        largest = max(np.linalg.norm(triangle[i]-triangle[j]) for i in range(3) for j in range(i))
        n = max(1, int(np.ceil(largest/step)))
        weights = np.asarray([(i/n,j/n,1-(i+j)/n) for i in range(n+1) for j in range(n+1-i)])
        points = weights@triangle
        _, valid, _ = _interpolate(grid, points)
        samples_total += len(points)
        unsupported += int(not valid.all())
        tangent = triangle[0]-(triangle[1]+triangle[2])/2
        vector, supported, _ = _interpolate(grid, triangle.mean(axis=0)[None])
        denominator = np.linalg.norm(tangent)*np.linalg.norm(vector[0])
        if supported[0] and denominator > 1e-12:
            reversed_count += int(np.dot(tangent,vector[0])/denominator < -1e-3)
    return dict(arrowheads=len(triangles), arrowhead_sampled_points=samples_total,
                unsupported_arrowheads=unsupported, reversed_arrowheads=reversed_count,
                arrowheads_passed=not (unsupported or reversed_count))


def source_hashes(path, key):
    method, field = key.split('_', 1)
    with h5py.File(path, 'r') as f:
        if field == 'structural':
            vectors = f[f'obsm/{method}_up_umap2d'][:]
            mask = f[f'obs/{method}_up_umap2d_display'][:].astype(bool)
        else:
            index = int(field[1:])
            vectors = f[f'obsm/{method}_guided_umap2d'][:].reshape(-1, 5, 2)[:, index]
            mask = np.linalg.norm(vectors, axis=1) > 0
    return dict(source_vectors_sha256=hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest(),
                source_mask_sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest())


def self_test():
    x = y = np.arange(5., dtype=float)
    g = dict(x=x, y=y, u=np.ones((5,5)), v=np.zeros((5,5)), valid=np.ones((5,5), bool))
    points = np.array([[.25, 1.5], [3.75, 1.5]])
    assert audit_paths(g, points, [0, 2])['passed']
    g['valid'][1,2] = False; g['u'][1,2] = g['v'][1,2] = np.nan
    assert audit_paths(g, points, [0, 2])['unsupported_samples'] > 0
    g['valid'][:] = True; g['u'][:] = 1; g['v'][:] = 0
    assert audit_paths(g, points[::-1], [0, 2])['reversed_segments'] == 1
    assert audit_paths(g, np.array([[-.5, 1.5], [1., 1.5]]), [0,2])['out_of_domain_samples'] > 0
    print('Self-test passed: supported, hole-crossing, reversed, and out-of-domain paths')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--sample-step-fraction', type=float, default=.1)
    parser.add_argument('--skip-source-hashes', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not 0 < args.sample_step_fraction <= .25:
        raise ValueError('Sampling must use at most one quarter of a grid cell')
    out = args.output_dir
    manifest = json.loads((out/'streamplot_manifest.json').read_text())
    grids, hashes = {}, {}
    reports = []
    for panel in manifest['panels']:
        grid_file = panel['grid_file']
        if grid_file not in grids:
            with np.load(out/grid_file, allow_pickle=False) as saved:
                grid = {k:saved[k] for k in saved.files if k!='metadata_json'}
                metadata = json.loads(str(saved['metadata_json'].item()))
            if not args.skip_source_hashes:
                cache_key = (panel['model'], panel['field'])
                if cache_key not in hashes:
                    source = Path(manifest['source_files'][panel['model']]['path'])
                    hashes[cache_key] = source_hashes(source, panel['field'])
                for name, value in hashes[cache_key].items():
                    if metadata[name] != value: raise AssertionError(f'Source hash mismatch: {grid_file} {name}')
            grids[grid_file] = grid
        with np.load(out/panel['path_file'], allow_pickle=False) as saved:
            if str(saved['grid_file'].item()) != grid_file:
                raise AssertionError('Path archive references a different grid')
            report = audit_paths(grids[grid_file], saved['points'], saved['offsets'], args.sample_step_fraction)
            if 'arrowheads' in saved.files:
                report.update(audit_arrowheads(grids[grid_file], saved['arrowheads'], args.sample_step_fraction))
                report['passed'] &= report['arrowheads_passed']
        reports.append({**panel, **report})
        print(f"{panel['figure']} panel {panel['panel']}: {report['sampled_points']:,} samples; "
              f"{report['unsupported_samples']} unsupported; {report['reversed_segments']} reversed", flush=True)
    source_files_unchanged = True
    for info in manifest['source_files'].values():
        stat = Path(info['path']).stat()
        source_files_unchanged &= stat.st_size == info['bytes'] and stat.st_mtime_ns == info['mtime_ns']
    result = dict(passed=all(r['passed'] for r in reports) and source_files_unchanged,
        panels=len(reports), unique_grids=len(grids), source_hashes_checked=not args.skip_source_hashes,
        source_files_unchanged=bool(source_files_unchanged), sample_step_fraction=args.sample_step_fraction,
        total_sampled_points=sum(r['sampled_points'] for r in reports),
        total_unsupported_samples=sum(r['unsupported_samples'] for r in reports),
        total_reversed_segments=sum(r['reversed_segments'] for r in reports),
        total_arrowheads=sum(r.get('arrowheads',0) for r in reports),
        total_unsupported_arrowheads=sum(r.get('unsupported_arrowheads',0) for r in reports),
        total_reversed_arrowheads=sum(r.get('reversed_arrowheads',0) for r in reports), results=reports,
        interpretation='Numerical display audit only; support-mask compliance does not validate biological flow or prevent mixing programs overlapping in 2D.')
    destination = out/'verification'; destination.mkdir(exist_ok=True)
    (destination/'streamline_path_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}, indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
