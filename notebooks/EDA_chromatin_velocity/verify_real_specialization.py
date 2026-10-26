"""Independent, bounded-I/O checks of saved real-atlas analysis and viewer.

Checks source identity/coordinates exhaustively, score arithmetic independently
on sampled source rows, transitions exhaustively, and projection arithmetic on
sampled rows (including broad-reference peaks). It never loads the full input
peaks-by-pseudobulk matrix. Run after the H5AD and offline viewer are complete.
"""
import argparse
import base64
from datetime import datetime
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts/chrom_velo'))
from real_specialization import parse_condition_grid, to_condition_cube


def read_column(group, name):
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        return obj.asstr()[:] if obj.dtype.kind in 'OSU' else obj[:]
    categories = read_column(obj, 'categories')
    codes = obj['codes'][:]
    if np.any(codes < 0):
        raise ValueError(f'Missing category values in required metadata {name}')
    return categories[codes]


def scalar(obj):
    return obj.asstr()[()] if obj.dtype.kind in 'OSU' else obj[()]


def take(dataset, rows):
    rows = np.asarray(rows, dtype=int)
    unique, inverse = np.unique(rows, return_inverse=True)
    if len(unique) <= 1024:
        return dataset[unique][inverse]
    # h5py fancy indexing with 18K scattered rows has quadratic selection
    # overhead. Scan bounded contiguous chunks and retain only requested rows.
    parts = []
    for start in range((int(unique[0]) // 8192) * 8192, int(unique[-1]) + 1, 8192):
        stop = min(start + 8192, int(unique[-1]) + 1)
        left, right = np.searchsorted(unique, [start, stop])
        if right > left:
            parts.append(dataset[start:stop][unique[left:right] - start])
    return np.concatenate(parts, axis=0)[inverse]


def close(actual, expected, name, atol=2e-6, rtol=2e-5):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True, err_msg=name)


def manual_score(cube, observed, method):
    mask = np.broadcast_to(observed, cube.shape) & np.isfinite(cube)
    x = np.where(mask, cube, 0.)
    n = mask.sum(axis=1)
    total = x.sum(axis=1)
    valid = (n >= 2) & (total > 0)
    if method == 'tau':
        maximum = x.max(axis=1)
        relative = np.divide(x, maximum[:, None, :], out=np.zeros_like(x), where=maximum[:, None, :] > 0)
        values = np.where(mask, 1 - relative, 0.).sum(axis=1) / np.maximum(n - 1, 1)
    else:
        p = np.divide(x, total[:, None, :], out=np.zeros_like(x), where=total[:, None, :] > 0)
        positive = p > 0
        terms = np.zeros_like(p)
        terms[positive] = p[positive] * np.log(p[positive])
        values = 1 + terms.sum(axis=1) / np.log(np.maximum(n, 2))
    return np.where(valid, np.clip(values, 0., 1.), np.nan)


def check_arithmetic(saved, source, grid, rows, checks):
    info = saved['uns/real_specialization']
    depth_matched = 'depth_matching' in info
    minimum = float(scalar(info['parameters/min_total_counts']))
    min_cells = float(scalar(info['parameters/min_cells']))
    multiplier = 1e6 / float(scalar(source['uns/common_scale_factor']))
    source_normalized = take(source['layers/normalized'], rows)
    source_raw = take(source['layers/sum'], rows)
    expected_x = (source_normalized * multiplier).astype('float32')
    x = take(saved['X'], rows)
    close(x, expected_x, 'CPM from source normalization', atol=1e-6)
    cube = to_condition_cube(x, grid)
    raw = to_condition_cube(source_raw, grid)
    available = grid['present'] & (grid['n_cells'] >= min_cells)
    common = available.all(axis=1)
    stored_common = read_column(info, 'common_celltypes').astype(str)
    np.testing.assert_array_equal(stored_common, grid['celltypes'][common])
    np.testing.assert_array_equal(info['condition_valid'][:], available)
    assert common.sum() >= 2
    fixed_mask = np.broadcast_to(common[None, :, None], cube.shape)
    fixed_counts = np.where(fixed_mask, raw, 0.).sum(axis=1)
    fixed_mean = np.where(fixed_mask, cube, 0.).sum(axis=1) / common.sum()
    close(take(saved['obsm/fixed_raw_total'], rows), fixed_counts, 'Fixed-panel raw totals')
    close(take(saved['obsm/fixed_mean_cpm'], rows), fixed_mean, 'Fixed-panel mean CPM')
    times = grid['stages']
    for method in ('tau', 'entropy'):
        expected = manual_score(cube, fixed_mask, method)
        expected[fixed_counts < minimum] = np.nan
        prefix = f'{method}_uncorrected' if depth_matched else method
        close(take(saved[f'obsm/{prefix}_specificity'], rows), expected, f'{method}: original-depth fixed scores')
        expected_rate = np.diff(expected, axis=1) / np.diff(times)
        close(take(saved[f'obsm/{prefix}_rate'], rows), expected_rate, f'{method}: original-depth fixed interval rates')
        available_score = manual_score(cube, available[None], method)
        available_counts = np.where(available[None], raw, 0.).sum(axis=1)
        available_score[available_counts < minimum] = np.nan
        close(take(saved[f'obsm/{method}_available_specificity'], rows), available_score, f'{method}: available scores')
        pair = available[:, :-1] & available[:, 1:]
        first = manual_score(cube[:, :, :-1], pair[None], method)
        last = manual_score(cube[:, :, 1:], pair[None], method)
        count_first = np.where(pair[None], raw[:, :, :-1], 0.).sum(axis=1)
        count_last = np.where(pair[None], raw[:, :, 1:], 0.).sum(axis=1)
        expected_rate = (last - first) / np.diff(times)
        expected_rate[(count_first < minimum) | (count_last < minimum)] = np.nan
        close(take(saved[f'obsm/{method}_available_rate'], rows), expected_rate, f'{method}: paired available rates')
    opening = np.full(x.shape, np.nan)
    for c in range(len(grid['celltypes'])):
        for t in range(len(times) - 1):
            j, k = grid['column_index'][c, t:t+2]
            if j < 0 or k < 0 or not available[c, t:t+2].all():
                continue
            good = source_raw[:, j] + source_raw[:, k] >= minimum
            opening[good, j] = (x[good, k] - x[good, j]) / (times[t+1] - times[t])
    close(take(saved['layers/accessibility_rate'], rows), opening, 'Opening rates and unavailable/terminal masks')
    fixed_rates = np.diff(cube[:, common], axis=2) / np.diff(times)
    summary_valid = fixed_counts[:, :-1] + fixed_counts[:, 1:] >= minimum
    mean_rate = np.where(summary_valid, fixed_rates.mean(axis=1), np.nan)
    rms_rate = np.where(summary_valid, np.sqrt(np.mean(fixed_rates**2, axis=1)), np.nan)
    close(take(saved['obsm/fixed_accessibility_mean_rate'], rows), mean_rate, 'Fixed-celltype mean accessibility rate')
    close(take(saved['obsm/fixed_accessibility_rms_rate'], rows), rms_rate, 'Fixed-celltype RMS accessibility rate')
    all_intervals = summary_valid.all(axis=1)
    net = np.where(all_intervals, np.sum(mean_rate * np.diff(times), axis=1) / (times[-1]-times[0]), np.nan)
    magnitude = np.where(all_intervals, np.sum(rms_rate * np.diff(times), axis=1) / (times[-1]-times[0]), np.nan)
    close(take(saved['obs/net_accessibility_rate'], rows), net, 'Duration-weighted net accessibility rate')
    close(take(saved['obs/rms_accessibility_rate'], rows), magnitude, 'Duration-mean interval RMS accessibility rate')
    checks['independent_source_arithmetic'] = dict(passed=True, sampled_peaks=len(rows), fixed_celltypes=int(common.sum()),
                                                 terminal_available_celltypes=int(available[:, -1].sum()))


def check_depth_matched_arithmetic(saved, source, grid, rows, checks):
    """Recreate source-block random draws and use independent score formulas."""
    info = saved['uns/real_specialization/depth_matching']
    seed = int(scalar(info['seed']))
    reps = int(scalar(info['n_thinnings']))
    block_size = int(scalar(info['block_size']))
    minimum = float(scalar(info['min_total_counts']))
    min_cells = float(scalar(info['min_cells']))
    assert block_size > 0 and reps >= 1
    common = (grid['present'] & (grid['n_cells'] >= min_cells)).all(axis=1)
    assert common.sum() >= 2
    target = grid['total_coverage'][common].min(axis=1)
    probability = target[:, None] / grid['total_coverage'][common]
    np.testing.assert_array_equal(read_column(info, 'celltypes'), grid['celltypes'][common])
    close(info['target_coverage'][:], target, 'Depth targets from source coverage', atol=0, rtol=0)
    close(info['thinning_probabilities'][:], probability, 'Depth sampling probabilities', atol=0, rtol=0)
    starts = np.unique((rows // block_size) * block_size)
    sign_checks = dict(tau=0, entropy=0)
    actual_rows = 0
    for start in starts:
        stop = min(int(start) + block_size, source['layers/sum'].shape[0])
        selected = rows[(rows >= start) & (rows < stop)]
        local = selected - start
        raw = to_condition_cube(source['layers/sum'][start:stop], grid)[:, common]
        assert np.isfinite(raw).all() and (raw >= 0).all() and (raw == np.floor(raw)).all()
        raw = raw.astype(np.int64)
        draws = np.stack([np.random.default_rng(np.random.SeedSequence([seed, int(start), rep])).binomial(
            raw, probability[None]) for rep in range(reps)])
        totals = draws.sum(axis=2)
        close(take(saved['obsm/depth_matched_mean_raw_total'], selected), totals.mean(0)[local],
              'Depth-matched mean retained counts')
        close(take(saved['obsm/depth_matched_min_raw_total'], selected), totals.min(0)[local],
              'Depth-matched minimum retained counts')
        supported = ((totals >= minimum) & (totals > 0)).all(axis=0)
        for method in ('tau', 'entropy'):
            scores = np.stack([manual_score(d.astype(float) * (1e6 / target[None, :, None]),
                                           np.ones(d.shape, dtype=bool), method) for d in draws])
            mean = np.where(supported, scores.mean(axis=0), np.nan)
            rates = np.diff(mean, axis=1) / np.diff(grid['stages'])
            close(take(saved[f'obsm/{method}_specificity'], selected), mean[local], f'{method}: matched scores')
            close(take(saved[f'obsm/{method}_rate'], selected), rates[local], f'{method}: matched rates')
            draw_rates = np.diff(scores, axis=2) / np.diff(grid['stages'])
            interval_valid = supported[:, :-1] & supported[:, 1:]
            agreement = np.mean(np.sign(draw_rates) == np.sign(rates)[None], axis=0)
            agreement = np.where(interval_valid, agreement, np.nan)
            saved_agreement = take(saved[f'obsm/{method}_thinning_sign_agreement'], selected)
            np.testing.assert_array_equal(np.isfinite(saved_agreement), interval_valid[local])
            assert np.all((saved_agreement[np.isfinite(saved_agreement)] >= 0)
                          & (saved_agreement[np.isfinite(saved_agreement)] <= 1))
            # Equivalent entropy/tau formulas can differ at machine-zero rates.
            # Verify signs exactly only when every sign is numerically resolved.
            resolved = interval_valid & (np.abs(rates) > 1e-12) & (np.abs(draw_rates) > 1e-12).all(axis=0)
            close(saved_agreement[resolved[local]], agreement[local][resolved[local]],
                  f'{method}: thinning sign stability')
            sign_checks[method] += int(resolved[local].sum())
        actual_rows += len(raw)
    checks['independent_depth_matched_arithmetic'] = dict(passed=True, sampled_peaks=len(rows),
        recreated_blocks=len(starts), recreated_source_rows=actual_rows, seed=seed, n_thinnings=reps,
        block_size=block_size, sign_agreement_resolved_intervals=sign_checks,
        note='Independent binomial draws and direct tau/entropy formulas. Sign checks exclude machine-zero rates; thinning stability is not biological confidence.')


def check_full_support(saved, grid, checks):
    minimum = float(scalar(saved['uns/real_specialization/parameters/min_total_counts']))
    if 'depth_matching' in saved['uns/real_specialization']:
        raw = saved['obsm/depth_matched_min_raw_total'][:]
        valid = (raw >= minimum) & (raw > 0)
    else:
        raw = saved['obsm/fixed_raw_total'][:]
        mean = saved['obsm/fixed_mean_cpm'][:]
        valid = (raw >= minimum) & (mean > 0)
    for method in ('tau', 'entropy'):
        score = saved[f'obsm/{method}_specificity'][:]
        rate = saved[f'obsm/{method}_rate'][:]
        np.testing.assert_array_equal(np.isfinite(score), valid)
        np.testing.assert_array_equal(np.isfinite(rate), valid[:, :-1] & valid[:, 1:])
        close(rate, np.diff(score, axis=1) / np.diff(grid['stages']), f'{method}: all saved rates equal supported score differences')
    checks['full_primary_support_masks'] = dict(passed=True, valid_scores_by_stage=valid.sum(0).tolist(),
                                                valid_rates_by_interval=(valid[:, :-1] & valid[:, 1:]).sum(0).tolist())


def check_identity(saved, source, atlas, checks, block=16384):
    n = len(saved['obs'][saved['obs'].attrs['_index']])
    indices = [f['obs'][f['obs'].attrs['_index']] for f in (saved, source, atlas)]
    for dataset in indices:
        assert len(dataset) == n
    for start in range(0, n, block):
        sl = slice(start, min(n, start+block))
        labels = [ds.asstr()[sl] for ds in indices]
        np.testing.assert_array_equal(labels[0], labels[1])
        np.testing.assert_array_equal(labels[0], labels[2])
        for dim in (2, 3):
            out = saved[f'obsm/X_umap{dim}d'][sl]
            np.testing.assert_array_equal(out, source[f'obsm/X_umap_{dim}D'][sl])
            np.testing.assert_array_equal(out, atlas[f'obsm/X_umap_{dim}D'][sl])
    # Verify the copied graph against the actual published atlas graph, not its
    # potentially different copy inside the condition-state file.
    sg, ag = saved['obsp/atlas_connectivities'], atlas['obsp/connectivities']
    assert sg.attrs['encoding-type'] == ag.attrs['encoding-type'] == 'csr_matrix'
    np.testing.assert_array_equal(sg.attrs['shape'], ag.attrs['shape'])
    for name in ('indptr', 'indices', 'data'):
        assert sg[name].shape == ag[name].shape
        for start in range(0, len(sg[name]), 1000000):
            out = sg[name][start:start+1000000]
            expected = ag[name][start:start+1000000].astype(out.dtype)
            np.testing.assert_array_equal(out, expected)
    checks['full_source_identity_coordinates_and_graph'] = dict(passed=True, peaks=n,
        source_coordinates_exact=True, atlas_coordinates_exact=True, atlas_graph_exact_at_saved_precision=True)


def check_transitions_and_projections(saved, rows, checks):
    pars = saved['uns/real_specialization/parameters']
    tolerance = float(scalar(pars['edge_tolerance']))
    power = float(scalar(pars['gradient_power']))
    minimum = int(scalar(pars['min_neighbors']))
    base = saved['obsp/atlas_connectivities']
    base_ptr = base['indptr'][:]
    coordinates = {dim: saved[f'obsm/X_umap{dim}d'][:].astype(float) for dim in (2, 3)}
    n = len(coordinates[2])
    root_flags = read_column(saved['obs'], 'broad_reference').astype(bool)
    root_rows = np.flatnonzero(root_flags)[:64]
    sampled = np.unique(np.r_[rows[:128], root_rows])
    details = {}
    for method in ('tau', 'entropy'):
        score = saved[f'obsm/{method}_specificity'][:, -1].astype(float)
        for direction, sign in [('up', 1.), ('down', -1.)]:
            name = f'{method}_{direction}'
            transition = saved[f'obsp/{name}_transitions']
            pointer = transition['indptr'][:]
            signed_score = sign * score
            for start in range(0, n, 16384):
                stop = min(n, start+16384)
                left, right = pointer[start], pointer[stop]
                dst = transition['indices'][left:right]
                weights = transition['data'][left:right]
                src = np.repeat(np.arange(start, stop), np.diff(pointer[start:stop+1]))
                assert np.isfinite(weights).all() and (weights > 0).all()
                assert np.all(src != dst)
                gain = signed_score[dst] - signed_score[src]
                assert np.isfinite(gain).all() and np.all(gain > tolerance)
                total = np.bincount(src-start, weights=weights, minlength=stop-start)
                close(total[total > 0], np.ones(np.sum(total > 0)), f'{name}: transition probability sums')
            for i in sampled:
                left, right = base_ptr[i:i+2]
                neighbors = base['indices'][left:right]
                edge_weight = base['data'][left:right].astype(float)
                unique, inverse = np.unique(neighbors, return_inverse=True)
                if len(unique) != len(neighbors):
                    edge_weight = np.bincount(inverse, weights=edge_weight, minlength=len(unique))
                    neighbors = unique
                gain = signed_score[neighbors] - signed_score[i]
                keep = (neighbors != i) & np.isfinite(gain) & (gain > tolerance) & (edge_weight > 0)
                expected_cols = neighbors[keep]
                w = edge_weight[keep] * (gain[keep] - tolerance) ** power
                expected_count = len(w)
                saved_count = saved[f'obs/{name}_support'][i]
                assert expected_count == saved_count
                if expected_count < minimum:
                    expected_cols = expected_cols[:0]; w = w[:0]
                else:
                    w /= w.sum()
                left, right = pointer[i:i+2]
                cols = transition['indices'][left:right]
                probs = transition['data'][left:right]
                order = np.argsort(cols); expected_order = np.argsort(expected_cols)
                np.testing.assert_array_equal(cols[order], expected_cols[expected_order])
                close(probs[order], w[expected_order], f'{name}: original graph edge weights')
                for dim in (2, 3):
                    delta = coordinates[dim][expected_cols] - coordinates[dim][i]
                    norm = np.linalg.norm(delta, axis=1)
                    unit = np.divide(delta, norm[:, None], out=np.zeros_like(delta), where=norm[:, None] > 0)
                    projected = np.sum(w[:, None] * unit, axis=0)
                    close(saved[f'obsm/{name}_umap{dim}d'][i], projected, f'{name}: direct unit-neighbor projection')
                    close(saved[f'obs/{name}_umap{dim}d_coherence'][i], np.linalg.norm(projected), f'{name}: coherence')
                    all_delta = coordinates[dim][neighbors] - coordinates[dim][i]
                    all_norm = np.linalg.norm(all_delta, axis=1)
                    all_unit = np.divide(all_delta, all_norm[:, None], out=np.zeros_like(all_delta), where=all_norm[:, None] > 0)
                    comparisons = []
                    for alt_tolerance, alt_power in [(.005, 1.), (.02, 1.), (.01, 2.)]:
                        alt_keep = (neighbors != i) & np.isfinite(gain) & (gain > alt_tolerance) & (edge_weight > 0)
                        alt_vector = np.zeros(dim)
                        if alt_keep.sum() >= minimum:
                            alt_weights = edge_weight[alt_keep] * (gain[alt_keep] - alt_tolerance) ** alt_power
                            alt_weights /= alt_weights.sum()
                            alt_vector = np.sum(alt_weights[:, None] * all_unit[alt_keep], axis=0)
                        denominator = np.linalg.norm(projected) * np.linalg.norm(alt_vector)
                        comparisons.append(float(projected @ alt_vector / denominator) if denominator > 1e-12 else 0.)
                    stability = float(np.mean(comparisons))
                    close(saved[f'obs/{name}_umap{dim}d_stability'][i], stability, f'{name}: missing-as-zero parameter sensitivity')
                    display = (expected_count >= minimum and np.linalg.norm(projected) >= float(scalar(pars['min_coherence']))
                               and stability >= float(scalar(pars['min_parameter_cosine'])))
                    assert bool(saved[f'obs/{name}_umap{dim}d_display'][i]) == display
            details[name] = dict(edges=int(pointer[-1]), sampled_projection_rows=len(sampled))
        # Independently check every sampled signed guided vector, its gating,
        # and its speed before display normalization.
        rate = take(saved[f'obsm/{method}_rate'], sampled)
        floor = float(scalar(pars['rate_effect_floor']))
        stable = np.ones(rate.shape, dtype=bool)
        if 'depth_matching' in saved['uns/real_specialization']:
            depth = saved['uns/real_specialization/depth_matching']
            if 'min_thinning_sign_agreement' in depth:
                threshold = float(scalar(depth['min_thinning_sign_agreement']))
                stable = take(saved[f'obsm/{method}_thinning_sign_agreement'], sampled) >= threshold
        for dim in (2, 3):
            actual = take(saved[f'obsm/{method}_guided_umap{dim}d'], sampled).reshape(len(sampled), 5, dim)
            expected = np.zeros_like(actual)
            for direction, positive in [('up', True), ('down', False)]:
                vector = take(saved[f'obsm/{method}_{direction}_umap{dim}d'], sampled).astype(float)
                norm = np.linalg.norm(vector, axis=1)
                mask = take(saved[f'obs/{method}_{direction}_umap{dim}d_display'], sampled).astype(bool)
                sign = rate > 0 if positive else rate < 0
                eligible = mask[:, None] & sign & stable & np.isfinite(rate) & (np.abs(rate) > floor) & (norm[:, None] > 1e-12)
                unit = np.divide(vector, norm[:, None], out=np.zeros_like(vector), where=norm[:, None] > 0)
                candidate = unit[:, None] * np.abs(rate[:, :, None])
                expected[eligible] = candidate[eligible]
            close(actual, expected, f'{method}: sign-matched supported guidance')
    checks['transitions_projections_and_guidance'] = dict(passed=True, fields=details,
        broad_reference_rows_checked=len(root_rows), no_self_or_plateau_transitions=True,
        broad_reference_obeys_same_projection_formula=True)
    return coordinates


def decode(value):
    if isinstance(value, dict) and '__array__' in value:
        dtype = np.dtype('<' + value['__array__'])
        return np.frombuffer(base64.b64decode(value['data']), dtype=dtype).reshape(value['shape'])
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def check_viewer(saved, grid, coordinates, output, checks):
    path = output / 'real_specialization_explorer.html'
    html = path.read_text()
    marker = '<script>let PACKED='
    assert marker in html
    packed = html.split(marker, 1)[1].split(';</script>', 1)[0]
    payload = decode(json.loads(packed))
    rows = np.asarray([p['global_index'] for p in payload['peaks']], dtype=int)
    assert len(np.unique(rows)) == len(rows)
    labels = saved['obs'][saved['obs'].attrs['_index']].asstr()
    np.testing.assert_array_equal(np.array([p['peak'] for p in payload['peaks']]), take(labels, rows))
    assert payload['total_peaks'] == payload['analysis_peak_count'] == len(coordinates[2])
    index_table = pd.read_csv(output / 'viewer_peak_index.csv')
    np.testing.assert_array_equal(index_table['global_index'], rows)
    for dim in (2, 3):
        np.testing.assert_array_equal(payload[f'umap{dim}d'], coordinates[dim][rows].astype('float32'))
    access = to_condition_cube(take(saved['X'], rows), grid)
    observed = saved['uns/real_specialization/condition_valid'][:].astype(bool)
    access[:, ~observed] = np.nan
    close(payload['accessibility'], access.reshape(len(rows), -1), 'Viewer accessibility missing-condition masks')
    opening = to_condition_cube(take(saved['layers/accessibility_rate'], rows), grid)[:, :, :5]
    close(payload['opening_rate'], opening.reshape(len(rows), -1), 'Viewer opening rates')
    for method in ('tau', 'entropy'):
        model = payload['methods'][method]
        close(model['specificity'], take(saved[f'obsm/{method}_specificity'], rows), 'Viewer scores')
        close(model['specificity_rate'], take(saved[f'obsm/{method}_rate'], rows), 'Viewer scalar rates')
        fields = {f['id']: f for f in model['fields']}
        for dim in (2, 3):
            length = np.linalg.norm(np.ptp(coordinates[dim], axis=0)) * .008
            for name, source in [('structural', f'{method}_up_umap{dim}d'), ('temporal', f'{method}_guided_umap{dim}d')]:
                vector = take(saved[f'obsm/{source}'], rows)
                if name == 'temporal':
                    vector = vector.reshape(len(rows), 5, dim)
                norm = np.linalg.norm(vector, axis=-1, keepdims=True)
                expected = np.divide(vector, norm, out=np.zeros_like(vector), where=norm > 0) * length
                close(fields[name][f'vectors{dim}d'], expected, 'Viewer display-normalized vectors')
                mask = (norm[..., 0] > 0 if name == 'temporal' else
                        take(saved[f'obs/{method}_up_umap{dim}d_display'], rows).astype(bool))
                np.testing.assert_array_equal(fields[name][f'arrow_mask{dim}d'].astype(bool), mask)
    checks['offline_viewer_matches_saved_model'] = dict(passed=True, display_peaks=len(rows),
                                                       fixed_celltypes=len(payload['specificity_celltypes']))


def check_source_fingerprints(output, checks):
    provenance = json.loads((output/'input_provenance.json').read_text())
    checked = []
    for name, record in provenance['files'].items():
        if not record.get('exists'):
            continue
        stat = Path(record['path']).stat()
        assert stat.st_size == record['size_bytes'], f'Source file size changed: {name}'
        expected_mtime = datetime.fromisoformat(record['modified_utc']).timestamp()
        assert abs(stat.st_mtime - expected_mtime) < 1e-5, f'Source modification time changed: {name}'
        checked.append(name)
    checks['source_fingerprints_match_preanalysis_audit'] = dict(passed=True, files=checked,
        scope='File sizes and modification timestamps match the initial read-only audit; no full-file cryptographic hashes were computed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE/'outputs/real_specialization_v1')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--skip-viewer', action='store_true')
    args = parser.parse_args()
    output = args.output_dir
    checks = {}
    report = dict(status='running', checks=checks)
    try:
        check_source_fingerprints(output, checks)
        with h5py.File(output/'real_specialization_flow.h5ad') as saved:
            info = saved['uns/real_specialization']
            state_path = Path(scalar(info['state_path']))
            atlas_path = Path(scalar(info['atlas_path']))
            report.update(state_path=str(state_path), atlas_path=str(atlas_path))
            with h5py.File(state_path) as source, h5py.File(atlas_path) as atlas:
                print('Checking full source identity, coordinates, and atlas graph', flush=True)
                check_identity(saved, source, atlas, checks)
                columns = ['annotation_ML_coarse', 'dev_stage', 'n_cells', 'total_coverage']
                grid = parse_condition_grid(pd.DataFrame({k: read_column(source['var'], k) for k in columns}))
                n = saved['X'].shape[0]
                rows = np.sort(np.random.default_rng(791).choice(n, min(args.samples, n), replace=False))
                print('Checking independent source arithmetic and full support masks', flush=True)
                check_arithmetic(saved, source, grid, rows, checks)
                if 'depth_matching' in info:
                    print('Checking independently recreated depth-matched source blocks', flush=True)
                    check_depth_matched_arithmetic(saved, source, grid, rows, checks)
                check_full_support(saved, grid, checks)
            print('Checking transitions and projections', flush=True)
            coordinates = check_transitions_and_projections(saved, rows, checks)
            if not args.skip_viewer:
                print('Checking offline viewer arrays', flush=True)
                check_viewer(saved, grid, coordinates, output, checks)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output/'independent_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
