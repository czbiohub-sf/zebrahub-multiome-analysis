#!/usr/bin/env python3
"""Independent scientific audit of the small shared-weight Cicero forecast.

This harness owns no production outputs. Tests recompute quantities from source
arrays, use an independent linear solve, and exercise partner-specific synthetic
signal and future-value perturbations. Cell halves are sampling partitions, not
biological replicates; the fixed graph was learned using all stages.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import traceback
import time

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'outputs/ct_shared_weight_forecast_v1/independent_audit'
SOURCE = HERE / 'outputs/ct_cis_validation_v1/cis_split_input.h5'
EDGES = HERE / 'outputs/ct_cis_validation_v1/cis_directed_matched_control_edges.csv.gz'
SEED = 20261026


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, obj):
    Path(path).write_text(json.dumps(native(obj), indent=2, allow_nan=False) + '\n')


def signature(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def import_production(path):
    spec = importlib.util.spec_from_file_location('shared_weight_production_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source():
    with h5py.File(SOURCE, 'r') as h:
        result = {key: h[key][:] for key in h if h[key].dtype.kind != 'O'}
        for key in ('peak_ids', 'celltypes'):
            result[key] = h[key].asstr()[:]
    return result


def source_reconstruction(data):
    """Recompute normalized inputs and target rates directly from counts."""
    assert np.array_equal(data['stages'], [10, 12, 14, 16, 19, 24])
    assert len(data['celltypes']) == 21
    target = data['target_local_indices']
    assert np.array_equal(data['source_indices'][target], data['target_source_indices'])
    report = {}
    for half in ('a', 'b'):
        expected = data['raw_counts_' + half] * 1e6 / data['coverage_' + half][None]
        stored = data['observed_cpm_' + half].astype(float)
        assert np.allclose(stored, expected, rtol=1e-7, atol=1e-7)
        report[half] = {
            'max_cpm_absolute_float32_error': float(np.abs(stored - expected).max()),
            'n_source_nodes': len(stored),
            'n_targets': len(target),
            'finite': bool(np.isfinite(stored).all()),
        }
    return report


def independent_graphs(data, threshold=.25):
    """Construct sparse graphs independently from saved directed edge records."""
    edges = pd.read_csv(EDGES)
    edges = edges[edges.coaccess >= threshold].copy()
    source = data['source_indices']
    target = data['target_local_indices']
    lookup = {int(global_id): local for local, global_id in enumerate(source)}
    target_lookup = {int(source[local]): row for row, local in enumerate(target)}
    rows = np.asarray([target_lookup[int(v)] for v in edges.target], int)
    columns = np.asarray([lookup[int(v)] for v in edges.neighbor], int)
    controls = np.asarray([lookup[int(v)] for v in edges.null_neighbor], int)
    weights = edges.coaccess.to_numpy(float)
    mass = np.bincount(rows, weights=weights, minlength=len(target))
    normalized = weights / mass[rows]
    shape = (len(target), len(source))
    real = sparse.csr_matrix((normalized, (rows, columns)), shape=shape)
    null = sparse.csr_matrix((normalized, (rows, controls)), shape=shape)
    assert np.allclose(np.asarray(real.sum(1)).ravel()[mass > 0], 1)
    assert np.allclose(np.asarray(null.sum(1)).ravel()[mass > 0], 1)
    assert np.all(real[np.arange(len(target)), target].A1 == 0)
    chromosomes = np.asarray([s.split('-')[0] for s in data['peak_ids']])
    assert np.array_equal(chromosomes[target[rows]], chromosomes[columns])
    return real, null, edges, {
        'threshold': threshold, 'n_targets': len(target),
        'n_connected': int(np.count_nonzero(mass)), 'directed_edges': len(edges),
        'no_self_edges': True, 'row_normalized': True, 'cis_only': True,
    }


def partner_specific_problem(n_targets=900, n_celltypes=5):
    """Known future signal requiring nonuniform weighting of the partners.

Each target has two equally scored/equidistant partners. Their current states
are opposite around a common center. Independent previous partner rates make
W(x*r) informative while W*r alone lacks the differential component. No
future observations or trajectory fits enter the feature definition.
"""
    rng = np.random.default_rng(SEED)
    n_source = 3 * n_targets
    targets = np.arange(n_targets)
    pairs = np.column_stack((n_targets + np.arange(n_targets),
                             2 * n_targets + np.arange(n_targets)))
    rows = np.repeat(np.arange(n_targets), 2)
    columns = pairs.ravel()
    weights = np.full(2 * n_targets, .5)
    graph = sparse.csr_matrix((weights, (rows, columns)), shape=(n_targets, n_source))
    current_log = np.empty((n_source, n_celltypes))
    current_log[:n_targets] = 3 + rng.normal(0, .35, (n_targets, n_celltypes))
    polarity = rng.choice([-1., 1.], size=(n_targets, n_celltypes))
    current_log[pairs[:, 0]] = 3 + .6 * polarity
    current_log[pairs[:, 1]] = 3 - .6 * polarity
    current = np.expm1(current_log)
    previous_rate = rng.normal(0, .15, (n_source, n_celltypes))
    previous = current - 2 * previous_rate
    assert previous.min() > 0
    source = np.repeat(current[:, :, None], 6, axis=2)
    source[:, :, 1] = previous
    source[:, :, 0] = previous - .1 * rng.normal(size=previous.shape)
    intrinsic = .12 * previous_rate[targets]
    partner_signal = graph @ ((current_log - 3) * previous_rate)
    truth = intrinsic + 2 * partner_signal
    observed = truth + rng.normal(0, .02 * np.std(truth), truth.shape)
    order = rng.permutation(n_targets)
    train = order[:700]
    test = order[700:]
    assert not np.intersect1d(train, test).size
    return {'source': source, 'targets': targets, 'graph': graph,
            'current_log': current_log, 'previous_rate': previous_rate,
            'truth': truth, 'observed': observed, 'train': train, 'test': test,
            'rows': rows, 'columns': columns, 'weights': weights,
            'score': np.full(len(rows), .5),
            'distance': np.full(len(rows), 10000.),
            'stages': np.array([10, 12, 14, 16, 19, 24.])}


# Production-API adapters and output checks are appended once its interface is
# finalized; the harness never edits the forecast implementation.


def direct_ridge_solve_check(module):
    """Check Schur implementation against a separately assembled dense design."""
    rng = np.random.default_rng(SEED + 11)
    records = []
    for p in (0, 2, 6):
        n, c = 67, 4
        own = rng.normal(size=(n, c, 2))
        extra = rng.normal(size=(n, c, p))
        if p:
            extra[:, :, 0] += .7 * own[:, :, 0]
        y = (own * np.array([.5, -.3])[None, None]).sum(2) + rng.normal(size=(n, c))
        for penalty in module.LAMBDAS:
            fit = module.fit_shared_ridge(own, extra, y, penalty)
            # Independently recompute all means/scales, rather than using fit's.
            xm = own.mean(0); xs = own.std(0); xs[xs <= 1e-12] = 1
            gm = extra.mean(0); gs = np.sqrt(np.mean((extra-gm)**2, axis=(0, 1)))
            gs[gs <= 1e-12] = 1
            x = (own-xm)/xs; g = (extra-gm)/gs
            ym = y.mean(0)
            design = np.zeros((n*c, 2*c+p))
            for ct in range(c):
                row = np.arange(n)*c+ct
                design[row, 2*ct:2*ct+2] = x[:, ct]
            design[:, 2*c:] = g.reshape(n*c, p)
            ridge = np.concatenate((np.full(2*c, penalty/c), np.full(p, penalty)))
            expected = np.linalg.solve(design.T@design/(n*c)+np.diag(ridge),
                                       design.T@(y-ym).ravel()/(n*c))
            actual = np.concatenate((fit['own_beta'].ravel(), fit['shared_beta']))
            coef_error = float(np.max(np.abs(expected-actual)))
            assert np.allclose(actual, expected, atol=2e-11, rtol=2e-11)
            direct_pred = (design@expected).reshape(n, c)+ym
            pred_error = float(np.max(np.abs(module.predict_shared(fit, own, extra)-direct_pred)))
            assert pred_error < 2e-11
            records.append({'n_global_features': p, 'penalty': penalty,
                            'max_coefficient_error': coef_error,
                            'max_prediction_error': pred_error})
    return records


def partner_specific_production_check(module, out):
    p = partner_specific_problem()
    W = p['graph']
    d = {'observed': {'a': p['source'], 'b': p['source'].copy()},
         'stages': p['stages'], 'target_local_indices': p['targets'],
         'graphs': {'cis': {'W': W, 'Wd': W*(np.log1p(10000)/np.log1p(500000)), 'Ws': W*.5},
                    'distance': {'W': W, 'Wd': W*(np.log1p(10000)/np.log1p(500000)), 'Ws': W*.5}}}
    raw = module.build_raw_blocks(d, include_generic=False)['a'][2]
    gate = module.gate_scaler(raw['own'][p['train']])
    records = []
    predictions = {}
    for model in ('own_history', 'own_nonlinear', 'constant_cis', 'flexible_cis', 'partner_feature_removed'):
        production_model = 'flexible_cis' if model == 'partner_feature_removed' else model
        own, extra = module.feature_block(raw, production_model, gate)
        if model == 'partner_feature_removed':
            extra = extra.copy(); extra[:, :, 3] = 0
        fit = module.fit_shared_ridge(own[p['train']], extra[p['train']],
                                      p['observed'][p['train']], min(module.LAMBDAS))
        pred = module.predict_shared(fit, own[p['test']], extra[p['test']])
        predictions[model] = pred
        mse = float(np.mean((pred-p['truth'][p['test']])**2))
        records.append({'model': model, 'test_mse_vs_latent_truth': mse,
                        'penalty': min(module.LAMBDAS), 'shared_coefficients': fit['shared_beta']})
    losses = {row['model']: row['test_mse_vs_latent_truth'] for row in records}
    gain_constant = 1-losses['flexible_cis']/losses['constant_cis']
    gain_removed = 1-losses['flexible_cis']/losses['partner_feature_removed']
    assert gain_constant > .9, gain_constant
    assert gain_removed > .9, gain_removed
    # Verify the production partner feature against an explicit edge sum.
    own, extra = module.feature_block(raw, 'flexible_cis', gate)
    expected = W@(((p['current_log']-gate['mean'])/gate['scale'])*p['previous_rate'])
    feature_error = float(np.max(np.abs(extra[:, :, 3]-expected)))
    assert feature_error < 1e-11
    # Extra zero features must reproduce own-only ridge, not create information.
    zeros = np.zeros((*own.shape[:2], 6))
    f0 = module.fit_shared_ridge(own[p['train']], zeros[p['train']], p['observed'][p['train']], .1)
    empty = zeros[:, :, :0]
    fb = module.fit_shared_ridge(own[p['train']], empty[p['train']], p['observed'][p['train']], .1)
    equality = float(np.max(np.abs(module.predict_shared(f0, own[p['test']], zeros[p['test']])-
                                 module.predict_shared(fb, own[p['test']], empty[p['test']]))))
    assert equality < 1e-12
    # Scrambled training outcomes should not recover the planted held-out signal.
    rng = np.random.default_rng(SEED+88)
    shuffled = p['observed'][p['train']].copy()
    shuffled = shuffled[rng.permutation(len(shuffled))]
    fs = module.fit_shared_ridge(own[p['train']], extra[p['train']], shuffled, .1)
    ps = module.predict_shared(fs, own[p['test']], extra[p['test']])
    zero_mse = float(np.mean(p['truth'][p['test']]**2))
    shuffled_gain = 1-float(np.mean((ps-p['truth'][p['test']])**2))/zero_mse
    assert shuffled_gain < .1, shuffled_gain
    np.savez_compressed(out/'partner_specific_test_predictions.npz',
                        truth=p['truth'][p['test']], **predictions)
    pd.DataFrame([{k:v for k,v in row.items() if k!='shared_coefficients'} for row in records]).to_csv(out/'partner_specific_test_metrics.csv', index=False)
    return {'n_train_peaks': len(p['train']), 'n_test_peaks': len(p['test']),
            'n_celltypes': p['truth'].shape[1], 'equal_edge_score_and_distance': True,
            'gain_over_constant_graph': gain_constant,
            'gain_over_partner_feature_removed': gain_removed,
            'max_partner_feature_manual_error': feature_error,
            'zero_feature_prediction_difference': equality,
            'shuffled_training_gain_over_zero': shuffled_gain,
            'results': records,
            'scope': 'Strong planted partner-dependent effect checks implementation; not a power guarantee for biological effects.'}


def future_fold_invariance(module, d):
    """Alter every post-origin source/outcome value and rerun complete folds."""
    import copy
    records = []
    for origin, half, fold in ((2, 'a', 0), (3, 'b', 1), (4, 'a', 2)):
        start = time.time()
        baseline = module.run_outer_fold(d, half, origin, fold)
        changed = copy.deepcopy(d)
        rng = np.random.default_rng(SEED + origin)
        future = np.arange(len(d['stages'])) > origin
        for h in ('a', 'b'):
            shape = changed['observed'][h][:, :, future].shape
            changed['observed'][h][:, :, future] = rng.uniform(100, 10000, shape)
            shape = changed['selection_observed'][h][:, :, future].shape
            changed['selection_observed'][h][:, :, future] = rng.uniform(100, 10000, shape)
            changed['selection_counts'][h][:, :, future] = rng.integers(1000, 100000, shape)
        changed['selection_ncells'][:, future] = 0
        changed['raw'] = module.build_raw_blocks(changed)
        for h in ('a', 'b'):
            for t in range(1, origin+1):
                for key, value in d['raw'][h][t].items():
                    assert np.array_equal(value, changed['raw'][h][t][key]), (origin, h, t, key)
        counterfactual = module.run_outer_fold(changed, half, origin, fold)
        for key in ('train_indices', 'test_indices'):
            assert np.array_equal(baseline[key], counterfactual[key])
        for key in ('eligible', 'group_mask'):
            assert np.array_equal(baseline['selection'][key], counterfactual['selection'][key])
        for kind in ('scores', 'masks'):
            for key in baseline['selection'][kind]:
                assert np.array_equal(baseline['selection'][kind][key], counterfactual['selection'][kind][key])
        assert native(baseline['selection']['audit']) == native(counterfactual['selection']['audit'])
        maxdiff = 0.
        for model, original in baseline['models'].items():
            altered = counterfactual['models'][model]
            for key in ('prediction', 'unshrunk_prediction'):
                assert np.array_equal(original[key], altered[key]), (origin, model, key)
                maxdiff = max(maxdiff, float(np.max(np.abs(original[key]-altered[key]))))
            assert native(original['gate']) == native(altered['gate'])
            assert native(original['tuning']) == native(altered['tuning'])
            assert native(original['fit']) == native(altered['fit'])
        other = 'b' if half == 'a' else 'a'
        target = d['target_local_indices']
        old_label = d['observed'][other][target, :, origin+1]
        new_label = changed['observed'][other][target, :, origin+1]
        assert not np.array_equal(old_label, new_label)
        record = {'origin_hpf': float(d['stages'][origin]), 'source_half': half,
                  'outer_fold': fold, 'models_checked': list(module.MODELS),
                  'predictions_bitwise_invariant': True, 'tuning_bitwise_invariant': True,
                  'selection_bitwise_invariant': True, 'post_origin_labels_changed': True,
                  'max_prediction_difference': maxdiff, 'seconds': time.time()-start}
        records.append(record)
        print('Future perturbation passed:', record, flush=True)
    # A genuine all-zero target enforces the documented exact-tie convention.
    zero = dict(d); zero['observed'] = dict(d['observed'])
    zero['observed']['b'] = np.repeat(d['observed']['b'][:, :, :1], 6, axis=2)
    tie = module.run_outer_fold(zero, 'a', 2, 0, models=('own_history',))['models']['own_history']
    assert np.all(tie['tuning']['validation_loss'] == 0)
    assert tie['tuning']['shrinkage'] == 0
    assert tie['tuning']['penalty'] == max(module.LAMBDAS)
    assert np.all(tie['prediction'] == 0)
    return {'future_perturbations': records,
            'exact_zero_target_tie': {'penalty': tie['tuning']['penalty'],
                                     'shrinkage': tie['tuning']['shrinkage'],
                                     'all_losses_exact_zero': True}}


def manual_raw_blocks(d, real, null, edges):
    """Reconstruct every graph feature independently, including edge metadata."""
    source = d['source_indices']; target = d['target_local_indices']
    lookup = {int(v): i for i, v in enumerate(source)}
    target_lookup = {int(source[v]): i for i, v in enumerate(target)}
    rows = np.array([target_lookup[int(v)] for v in edges.target])
    score = edges.coaccess.to_numpy(float)
    weight = score/np.bincount(rows, weights=score, minlength=len(target))[rows]
    midpoint = np.array([(float(v.split('-')[1])+float(v.split('-')[2]))/2 for v in d['all_peak_ids']])
    graphs = {}
    for name, column, distance_name, W in (
            ('cis', 'neighbor', 'distance_bp', real),
            ('distance', 'null_neighbor', 'null_distance_bp', null)):
        cols = np.array([lookup[int(v)] for v in edges[column]])
        distance = np.abs(midpoint[target[rows]]-midpoint[cols])
        assert np.allclose(distance, edges[distance_name].to_numpy(float))
        wd = weight*np.clip(np.log1p(distance)/np.log1p(500000), 0, 1)
        graphs[name] = (W, sparse.csr_matrix((wd, (rows, cols)), shape=W.shape),
                        sparse.csr_matrix((weight*score, (rows, cols)), shape=W.shape))
    result = {}; max_error = 0.
    for half in ('a', 'b'):
        result[half] = {}
        for t in range(1, 5):
            state = d['observed'][half][:, :, t]
            rate = (state-d['observed'][half][:, :, t-1])/(d['stages'][t]-d['stages'][t-1])
            logs = np.log1p(state)
            block = {'own': np.stack([logs[target], rate[target]], axis=2)}
            for label, (W, WD, WS) in graphs.items():
                neighbor_rate = W@rate
                block[label] = np.stack([W@state-state[target], neighbor_rate,
                    logs[target]*neighbor_rate, W@(logs*rate), WD@rate, WS@rate], axis=2)
            # Reconstruct generic summaries from saved/current geometry only.
            neighbors = d['raw'][half][t]['generic_neighbor_indices']
            weights = d['raw'][half][t]['generic_weights']
            assert not np.any(neighbors == target[:, None])
            assert np.allclose(weights.sum(1), 1)
            wx = np.sum(state[neighbors]*weights[:, :, None], axis=1)
            wr = np.sum(rate[neighbors]*weights[:, :, None], axis=1)
            block['generic'] = np.stack([wx-state[target], wr], axis=2)
            for key, value in block.items():
                err = float(np.max(np.abs(value-d['raw'][half][t][key])))
                max_error = max(max_error, err)
                assert np.allclose(value, d['raw'][half][t][key], atol=1e-10, rtol=1e-12), (half,t,key,err)
            result[half][t] = block
    return result, {'max_reconstructed_feature_error': max_error,
                    'distance_metadata_matches_peak_coordinates': True,
                    'all_generic_self_edges_excluded': True}


def manual_features(raw, model, gate):
    own = raw['own']; r = own[:, :, 1]
    z = (own[:, :, 0]-gate['mean'])/gate['scale']
    if model == 'own_history':
        extra = np.empty((*own.shape[:2], 0))
    elif model == 'own_nonlinear':
        extra = np.stack([z*z, np.expm1(own[:, :, 0]), z*r, z*z*r,
                          r*r/(1+np.abs(r)), np.log1p(np.abs(r))], axis=2)
    elif model in ('constant_cis', 'generic_smoothing'):
        extra = raw['cis' if model == 'constant_cis' else 'generic'][:, :, :2].copy()
    else:
        extra = raw['cis' if model == 'flexible_cis' else 'distance'].copy()
        extra[:, :, 2] = (extra[:, :, 2]-gate['mean']*extra[:, :, 1])/gate['scale']
        extra[:, :, 3] = (extra[:, :, 3]-gate['mean']*extra[:, :, 1])/gate['scale']
    return own, extra


def saved_output_audit(module, d, manual, out):
    model_dir = module.OUT
    path = model_dir/'predictions.h5'
    metrics = pd.read_csv(model_dir/'metrics.csv')
    blocks = pd.read_csv(model_dir/'chromosome_losses.csv')
    pooled = pd.read_csv(model_dir/'pooled_metrics.csv')
    tuning = pd.read_csv(model_dir/'tuning_grid.csv')
    raw_source = load_source()
    selected_rows = 0
    for keys, group in tuning.groupby(['half','origin','fold','model']):
        chosen = group[group.selected]
        assert len(chosen) == 1
        expected = group.sort_values(['validation_squared_rate_error','shrinkage','penalty'],
                                     ascending=[True,True,False],kind='stable').iloc[0]
        assert chosen.iloc[0].penalty == expected.penalty
        assert chosen.iloc[0].shrinkage == expected.shrinkage
        selected_rows += 1
    cache = {}; forecast_checks = []; max_prediction_error = 0.; max_scaler_error = 0.
    with h5py.File(path, 'r') as h:
        assert np.array_equal(h['peak_ids'].asstr()[:], d['peak_ids'])
        assert np.array_equal(h['source_indices'][:], d['source_indices'][d['target_local_indices']])
        origins = h['origin_indices'][:]
        for half in ('a', 'b'):
            other = 'b' if half == 'a' else 'a'
            g = h['half_'+half]
            # Integer count reconstruction, including precisely the persisted float32 CPM representation.
            reconstructed_cpm = (raw_source['raw_counts_'+other]*1e6/raw_source['coverage_'+other][None]).astype('f4').astype(float)
            current = reconstructed_cpm[d['target_local_indices']]
            expected_y = np.stack([(current[:,:,t+1]-current[:,:,t])/(d['stages'][t+1]-d['stages'][t]) for t in origins],axis=2)
            saved_y = g['target_rate'][:]
            assert np.array_equal(saved_y, expected_y.astype('f4'))
            predictions = {m: g['predictions'][m][:].astype(float) for m in g['predictions']}
            masks = {c: g['cohorts'][c][:].astype(bool) for c in g['cohorts']}
            errors = {m:(pred-expected_y)**2 for m,pred in predictions.items()}
            cache[half] = {'truth':expected_y,'predictions':predictions,'masks':masks,'errors':errors}
            for ti,t in enumerate(origins):
                for fold in range(5):
                    fg = h[f'fits/half_{half}/origin_{t}/fold_{fold}']
                    train = fg['train_indices'][:]; test = fg['test_indices'][:]
                    assert not set(d['chromosome'][train]) & set(d['chromosome'][test])
                    assert np.array_equal(test,np.flatnonzero(d['fold']==fold))
                    train_own = np.concatenate([manual[half][j]['own'][train] for j in range(1,t)])
                    gate_ref = {'mean':float(train_own[:,:,0].mean()),'scale':max(float(train_own[:,:,0].std()),1e-12)}
                    for model in module.MODELS:
                        mg = fg[model]
                        gate = {'mean':float(mg.attrs['gate_mean']),'scale':float(mg.attrs['gate_scale'])}
                        assert gate == gate_ref
                        own,extra = manual_features(manual[half][t], model, gate)
                        fit = {key:mg[key][:] for key in ('own_mean','own_scale','shared_mean','shared_scale','target_mean','own_beta','shared_beta')}
                        train_extra=np.concatenate([manual_features(manual[half][j],model,gate)[1][train] for j in range(1,t)])
                        scaler_refs={'own_mean':train_own.mean(0),'own_scale':train_own.std(0),
                                     'shared_mean':train_extra.mean(0)}
                        scaler_refs['own_scale'][scaler_refs['own_scale']<=1e-12]=1
                        gs=np.sqrt(np.mean((train_extra-scaler_refs['shared_mean'])**2,axis=(0,1)));gs[gs<=1e-12]=1
                        scaler_refs['shared_scale']=gs
                        for key,ref in scaler_refs.items():
                            assert np.allclose(fit[key],ref,atol=1e-11,rtol=1e-12),(half,t,fold,model,key)
                            if ref.size:max_scaler_error=max(max_scaler_error,float(np.max(np.abs(fit[key]-ref))))
                        xx=(own[test]-fit['own_mean'])/fit['own_scale']
                        zz=(extra[test]-fit['shared_mean'])/fit['shared_scale']
                        raw=fit['target_mean'][None]+np.sum(xx*fit['own_beta'][None],axis=2)+np.sum(zz*fit['shared_beta'][None,None],axis=2)
                        expected=float(mg.attrs['shrinkage'])*raw
                        saved=predictions[model][test,:,ti]
                        err=float(np.max(np.abs(saved-expected)))
                        max_prediction_error=max(max_prediction_error,err)
                        assert np.allclose(saved,expected,rtol=1e-6,atol=3e-6),(half,t,fold,model,err)
                        stored_raw=g['unshrunk_predictions'][model][test,:,ti]
                        assert np.allclose(stored_raw,raw,rtol=1e-6,atol=3e-6)
                        if float(mg.attrs['shrinkage'])==0:assert np.all(saved==0)
                        forecast_checks.append({'half':half,'origin_index':int(t),'fold':fold,'model':model,
                                                'max_saved_prediction_roundoff':err})
    checks=[];max_relative=0.
    for row in metrics.itertuples():
        ti=list(module.ORIGINS).index(int(row.origin));c=cache[row.half];mask=c['masks'][row.cohort][:,ti]
        sse=float(c['errors'][row.model][mask,:,ti].sum())
        own=float(c['errors']['own_history'][mask,:,ti].sum())
        zero=float(c['errors']['persistence'][mask,:,ti].sum())
        dt=d['stages'][row.origin+1]-d['stages'][row.origin]
        assert int(mask.sum())==row.n_peaks and int(mask.sum())*len(d['celltypes'])==row.n_values
        for name,value in [('rate_sse',sse),('state_change_sse',sse*dt*dt),('own_history_sse',own),('persistence_sse',zero)]:
            actual=float(getattr(row,name));rel=abs(value-actual)/max(abs(actual),1e-12);max_relative=max(max_relative,rel)
            assert np.isclose(value,actual,rtol=1e-6,atol=1e-6),(row.half,row.origin,row.cohort,row.model,name,rel)
        checks.append({'half':row.half,'origin':row.origin,'cohort':row.cohort,'model':row.model,'recomputed_rate_sse':sse})
    for row in blocks.itertuples():
        ti=list(module.ORIGINS).index(int(row.origin));c=cache[row.half]
        mask=c['masks'][row.cohort][:,ti]&(d['chromosome']==str(row.chromosome))
        value=float(c['errors'][row.model][mask,:,ti].sum())
        assert int(mask.sum())==row.n_peaks
        assert np.isclose(value,row.rate_sse,rtol=1e-6,atol=1e-6)
    for row in pooled.itertuples():
        q=metrics[(metrics.cohort==row.cohort)&(metrics.model==row.model)]
        if str(row.interval)!='pooled':q=q[q.origin==int(row.interval)]
        value=float(q[row.error].sum())
        assert np.isclose(value,row.sse,rtol=1e-12,atol=1e-8)
        for baseline in ('persistence','own_history','own_nonlinear','constant_cis','distance_flexible','generic_smoothing'):
            z=metrics[(metrics.cohort==row.cohort)&(metrics.model==baseline)]
            if str(row.interval)!='pooled':z=z[z.origin==int(row.interval)]
            gain=1-value/z[row.error].sum()
            assert np.isclose(gain,getattr(row,'vs_'+baseline),atol=2e-12,rtol=2e-12)
            assert getattr(row,'vs_'+baseline+'_low')<=getattr(row,'vs_'+baseline+'_high')
    pd.DataFrame(forecast_checks).to_csv(out/'saved_prediction_reconstruction.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'independent_metric_reconstruction.csv',index=False)
    return {'n_saved_fits_reconstructed':len(forecast_checks),'n_metric_rows_recomputed':len(checks),
            'n_chromosome_loss_rows_recomputed':len(blocks),'n_pooled_rows_recomputed':len(pooled),
            'n_tuning_choices_verified':selected_rows,'max_prediction_float32_roundoff':max_prediction_error,
            'max_training_scaler_difference':max_scaler_error,'max_sse_relative_roundoff':max_relative,
            'raw_count_targets_match_exact_float32_representation':True,
            'train_and_test_chromosomes_disjoint':True,'zero_shrinkage_is_exact_persistence':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--skip-future',action='store_true',help='Run numerical/output checks only; report future check as not run.')
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    module=import_production(HERE/'ct_shared_weight_forecast.py')
    started=time.time();report={'status':'running','seed':SEED,'checks':{},'started_unix':started}
    sources=[SOURCE,EDGES,module.SELECT,HERE/'ct_shared_weight_forecast.py',HERE/'ct_variable_peak_forecast_selection.py']
    before=[signature(p) for p in sources]
    def check(name,fn):
        begin=time.time()
        try:
            value=fn();report['checks'][name]={'status':'passed','seconds':time.time()-begin,'details':value}
            print('PASS',name,round(time.time()-begin,2),'s',flush=True)
        except Exception as exc:
            report['checks'][name]={'status':'failed','error':repr(exc),'traceback':traceback.format_exc()}
            save_json(out/'independent_validation.json',report)
            raise
        save_json(out/'independent_validation.json',report)
        return value
    source=load_source();check('raw_source_reconstruction',lambda:source_reconstruction(source))
    real,null,edges,graph=independent_graphs(source);check('independent_graph_reconstruction',lambda:graph)
    check('dense_ridge_vs_schur',lambda:direct_ridge_solve_check(module))
    check('partner_specific_positive_and_null_controls',lambda:partner_specific_production_check(module,out))
    d=module.load_data()
    manual,feature_report=manual_raw_blocks(d,real,null,edges)
    check('manual_feature_reconstruction',lambda:feature_report)
    if not args.skip_future:check('whole_fold_future_invariance',lambda:future_fold_invariance(module,d))
    else:report['checks']['whole_fold_future_invariance']={'status':'not_run'}
    check('saved_predictions_and_losses',lambda:saved_output_audit(module,d,manual,out))
    assert before==[signature(p) for p in sources]
    report.update(status='passed' if not args.skip_future else 'partial_passed',seconds=time.time()-started,
                  sources_unchanged=True,sources=before,script_sha256=sha256(__file__),
                  audited_model_sha256=sha256(HERE/'ct_shared_weight_forecast.py'),
                  limitations=['Graph learned using all stages/cells; this audit does not make graph estimation prospective.',
                  'Cell halves quantify sampling reproducibility, not independent biological replication.',
                  'Strong synthetic recovery checks this planted mechanism, not power for subtle biological coupling.',
                  'Generic kNN graph uses current source states of held-out target identities; coefficient fitting is chromosome-held-out.'])
    save_json(out/'independent_validation.json',report)
    print('INDEPENDENT AUDIT',report['status'],'seconds',round(report['seconds'],1),flush=True)


if __name__=='__main__':
    main()
