"""Independent numerical audit of completed whole-trajectory peak-flow artifacts.

Run only after the writer closes trajectory_fits.h5. Model checks explicitly
refit each leave-one-stage-out training set; they do not call trajectory_modes.
Projection checks use per-row state differences and unit embedding directions;
they do not call trajectory_projection. Input matrices are read for sampled rows.
"""
from pathlib import Path
import argparse
import base64
import json
import time

import h5py
import numpy as np
from scipy import sparse

from verify_real_specialization import read_column, take, close

HERE = Path(__file__).resolve().parent


def say(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def independent_fit(y, exposure, counts, condition, times, params):
    """Small direct weighted fits and direct leave-one-out training refits."""
    valid = condition & np.isfinite(y)
    out = dict(model=-1, fitted=np.full(len(times), np.nan),
               rates=np.full(len(times)-1, np.nan), rate_se=np.full(len(times)-1, np.nan),
               rate_supported=np.zeros(len(times)-1, bool), mode=0, fit_valid=False,
               observed_count=int(valid.sum()), cv_error=np.nan, cv_improvement=np.nan,
               lack_of_fit=np.nan, raw_total=float(np.nansum(counts[condition])))
    if valid.sum() < params['min_observations']:
        return out
    weights = np.where(valid, exposure, 0.)
    weights /= weights.sum()
    vals = np.where(valid, y, 0.)
    variance = np.where(valid, (counts+.5)*(1e6/exposure)**2, 0.)
    q = (times-times[0])/(times[-1]-times[0])
    designs = [np.ones((len(times), 1)), np.column_stack([np.ones(len(times)), q])]
    for knot in (14., 16.):
        designs.append(np.column_stack([np.ones(len(times)), q,
            np.maximum(q-(knot-times[0])/(times[-1]-times[0]), 0.)]))
    center = weights @ vals
    scale = weights @ ((vals-center)**2)
    candidates = []
    for code, design in enumerate(designs):
        ok = np.linalg.matrix_rank(design[valid]) == design.shape[1]
        if code >= 2:
            knot = (14., 16.)[code-2]
            ok &= ((valid & (times <= knot)).sum() >= 2 and
                   (valid & (times >= knot)).sum() >= 2)
        # Only invert full-rank candidates; excluded candidates have infinite CV.
        if not ok:
            candidates.append(None)
            continue
        hat = design @ np.linalg.solve(design.T @ (weights[:, None]*design), design.T*weights)
        if np.any(np.diag(hat)[valid] >= 1.-1e-6):
            candidates.append(None)
            continue
        prediction = hat @ vals
        error = 0.
        for held in np.flatnonzero(valid):
            train = valid.copy(); train[held] = False
            rootw = np.sqrt(weights[train])
            coef = np.linalg.lstsq(design[train]*rootw[:, None], vals[train]*rootw, rcond=None)[0]
            loo = max(float(design[held] @ coef), 0.)
            error += weights[held]*(vals[held]-loo)**2
        score = error+params['complexity_penalty']*(design.shape[1]-1)*scale
        candidates.append((score, error, hat, prediction))
    chosen = int(np.argmin([c[0] if c is not None else np.inf for c in candidates]))
    score, error, hat, linear_prediction = candidates[chosen]
    prediction = np.maximum(linear_prediction, 0.)
    rates = np.diff(prediction)/np.diff(times)
    endpoints = valid[:-1] & valid[1:]
    se, boundary = [], []
    for j, dt in enumerate(np.diff(times)):
        row = (hat[j+1]-hat[j])/dt
        active = linear_prediction > 0
        clipped_row = (active[j+1]*hat[j+1]-active[j]*hat[j])/dt
        se.append(np.sqrt(max(np.dot(row*row, variance), np.dot(clipped_row*clipped_row, variance))))
        boundary.append(bool(np.any(np.abs(linear_prediction[j:j+2]) < 1e-10)))
    se = np.asarray(se)
    noise = weights @ variance
    residual = weights @ ((vals-prediction)**2)
    lof = residual/noise if noise > 0 else (np.inf if residual > 1e-12 else 0.)
    rates[~endpoints] = np.nan; se[~endpoints] = np.nan
    prediction[~valid] = np.nan
    supported = np.isfinite(rates) & (np.abs(rates) > np.maximum(params['rate_floor'], params['rate_z']*se)) & (lof <= params['lack_of_fit_threshold'])
    raw_total = out['raw_total']
    supported &= raw_total >= 20
    full_sign = np.sign(rates[np.isfinite(rates) & (np.abs(rates) > params['rate_floor'])])
    supported_sign = np.sign(rates[supported])
    if lof > params['lack_of_fit_threshold']:
        mode = 6
    elif not len(supported_sign):
        mode = 1
    elif len(set(full_sign)) > 1:
        mode = (4 if full_sign[0] > 0 else 5) if len(set(supported_sign)) == 2 and np.count_nonzero(np.diff(full_sign)) == 1 else 6
    else:
        mode = 2 if supported_sign[0] > 0 else 3
    if raw_total < 20:
        mode = 0
    base = candidates[0][1]
    out.update(model=chosen, fitted=prediction, rates=rates, rate_se=se,
               rate_supported=supported, mode=mode, fit_valid=raw_total >= 20,
               cv_error=error, cv_improvement=(base-error)/base if base > 1e-14 else 0.,
               lack_of_fit=lof, boundary=np.asarray(boundary))
    return out


def check_inputs_and_fits(f, source, atlas, params, rng, samples, checks):
    ids = f['peak_ids'].asstr()[:]
    for obj in (source, atlas):
        np.testing.assert_array_equal(ids, read_column(obj['obs'], obj['obs'].attrs['_index']).astype(str))
    coords = {}
    for name, original in [('umap2d', 'X_umap_2D'), ('umap3d', 'X_umap_3D')]:
        coords[name] = f[name][:]
        np.testing.assert_array_equal(coords[name], source['obsm'][original][:])
        np.testing.assert_array_equal(coords[name], atlas['obsm'][original][:])
    celltypes = f['celltypes'].asstr()[:]
    times = f['stages'][:]
    var = source['var']
    labels = read_column(var, 'annotation_ML_coarse').astype(str)
    somite_hpf = {'0somites': 10., '5somites': 12., '10somites': 14., '15somites': 16., '20somites': 19., '30somites': 24.}
    stages = np.asarray([somite_hpf[str(x)] if str(x) in somite_hpf else float(str(x).replace('hpf', '').strip()) for x in read_column(var, 'dev_stage')])
    cells = read_column(var, 'n_cells').astype(float)
    coverage = read_column(var, 'total_coverage').astype(float)
    indices = np.full((len(celltypes), len(times)), -1, int)
    for j, (label, stage) in enumerate(zip(labels, stages)):
        c = np.flatnonzero(celltypes == label); t = np.flatnonzero(times == stage)
        assert len(c) == len(t) == 1
        assert indices[c[0], t[0]] == -1
        indices[c[0], t[0]] = j
    condition = (indices >= 0)
    condition[condition] &= cells[indices[condition]] >= 20
    np.testing.assert_array_equal(f['column_index'][:], indices)
    np.testing.assert_array_equal(f['condition_valid'][:], condition)
    np.testing.assert_array_equal(f['common_celltypes'][:], condition.all(axis=1))
    exposures = np.full(indices.shape, np.nan)
    exposures[indices >= 0] = coverage[indices[indices >= 0]]
    close(f['coverage'][:], exposures, 'Coverage grid', rtol=0, atol=0)
    broad = f['broad_reference'][:].astype(bool)
    roots = np.flatnonzero(broad)
    rows = np.unique(np.r_[rng.choice(len(ids), min(samples, len(ids)), replace=False), rng.choice(roots, min(16, len(roots)), replace=False)])
    raw = take(source['layers/sum'], rows)
    normalized = take(source['layers/normalized'], rows)
    cube = np.full((len(rows),)+indices.shape, np.nan)
    counts = cube.copy()
    cube[:, indices >= 0] = normalized[:, indices[indices >= 0]] * (1e6/float(source['uns/common_scale_factor'][()]))
    counts[:, indices >= 0] = raw[:, indices[indices >= 0]]
    close(take(f['observed_cpm'], rows), cube, 'Observed CPM and absent-stage NaNs')
    output = {k:take(f[k], rows) for k in ['fitted_cpm', 'rates', 'rate_se', 'rate_supported', 'modes', 'models', 'fit_valid', 'cv_error', 'cv_improvement', 'lack_of_fit', 'raw_total', 'observed_count']}
    n_checked = 0; boundary_intervals = 0
    for i in range(len(rows)):
        for c in range(len(celltypes)):
            expected = independent_fit(cube[i,c], exposures[c], counts[i,c], condition[c], times, params)
            for name, key in [('model','models'), ('mode','modes'), ('fit_valid','fit_valid'), ('observed_count','observed_count')]:
                assert output[key][i,c] == expected[name], (int(rows[i]), str(celltypes[c]), name, output[key][i,c], expected[name])
            for name, key in [('fitted','fitted_cpm'),('rates','rates'),('cv_error','cv_error'),('cv_improvement','cv_improvement'),('lack_of_fit','lack_of_fit'),('raw_total','raw_total')]:
                close(output[key][i,c], expected[name], f'{rows[i]}/{celltypes[c]}: {key}', atol=2e-5, rtol=3e-5)
            # At exactly-zero predictions the clipped map is nondifferentiable;
            # do not assert an active-set derivative convention from roundoff.
            boundary = expected.get('boundary', np.zeros(len(times)-1, bool))
            boundary_intervals += int(boundary.sum())
            close(output['rate_se'][i,c][~boundary], expected['rate_se'][~boundary], 'Conditional clipped-rate SE', atol=2e-6, rtol=3e-5)
            np.testing.assert_array_equal(output['rate_supported'][i,c][~boundary], expected['rate_supported'][~boundary])
            n_checked += 1
    checks['identity'] = dict(passed=True, all_peak_ids=len(ids), all_coordinate_rows=len(ids), common_celltypes=int(condition.all(axis=1).sum()))
    checks['independent_fits'] = dict(passed=True, sampled_peaks=len(rows), sampled_peak_celltype_trajectories=n_checked,
        direct_leave_one_stage_out_refits=True, conditional_se_boundary_intervals_not_compared=boundary_intervals,
        broad_reference_peaks_in_sample=int(broad[rows].sum()))
    return coords, rows


def check_projections(f, atlas, coords, rng, rows, checks):
    original = atlas['obsp/connectivities']
    graph = sparse.csr_matrix((original['data'][:], original['indices'][:], original['indptr'][:]), shape=(len(f['peak_ids']),)*2)
    graph.sum_duplicates()
    params = json.loads(f.attrs['projection_parameters_json'])
    fit_params = json.loads(f.attrs['parameters_json'])
    common = f['common_celltypes'][:].astype(bool)
    C = int(common.sum()); needed = max(params['min_shared_features'], int(np.ceil(C*params['min_shared_fraction'])))
    exposure = f['coverage'][:][common]; dt = np.diff(f['stages'][:]); details = []
    for t in range(len(dt)):
        active = f[f'projection/{t}/active'][:].astype(bool)
        transition = f[f'projection/{t}/probabilities']
        probability_graph = sparse.csr_matrix((transition['data'][:], transition['indices'][:],
            transition['indptr'][:]), shape=graph.shape)
        assert np.isfinite(probability_graph.data).all() and np.all(probability_graph.data > 0)
        assert np.all(probability_graph.diagonal() == 0)
        assert probability_graph.multiply(graph.astype(bool)).nnz == probability_graph.nnz
        row_sums = np.asarray(probability_graph.sum(axis=1)).ravel()
        close(row_sums, active.astype(float), 'All transition rows normalize or remain empty', atol=2e-6, rtol=2e-6)
        assert np.all(np.diff(probability_graph.indptr)[active] >= params['min_neighbors'])
        for name in coords:
            coherence = f[f'projection/{t}/coherence/{name}'][:]
            count = f[f'projection/{t}/projection_neighbor_count/{name}'][:]
            expected_display = active & (coherence >= params['min_coherence']) & (coherence > 1e-12) & (count >= params['min_neighbors'])
            np.testing.assert_array_equal(f[f'projection/{t}/projection_valid/{name}'][:], expected_display)
        del probability_graph, row_sums
        pool = np.flatnonzero(active)
        sampled = np.unique(np.r_[rows, rng.choice(pool, min(48, len(pool)), replace=False)])
        neighbors = np.unique(np.concatenate([graph.indices[graph.indptr[i]:graph.indptr[i+1]] for i in sampled]))
        required_rows = np.union1d(sampled, neighbors)
        location = {int(i): j for j, i in enumerate(required_rows)}
        fitted = take(f['fitted_cpm'], required_rows)[:, common]
        rates = take(f['rates'], required_rows)[:, common, t]
        ses = take(f['rate_se'], required_rows)[:, common, t]
        lof = take(f['lack_of_fit'], required_rows)[:, common]
        rate_supported = take(f['rate_supported'], required_rows)[:, common, t]
        observed = take(f['observed_cpm'], required_rows)[:, common]
        totals = take(f['raw_total'], required_rows)[:, common]
        for kind in ('projection', 'raw_projection'):
            group = f[f'{kind}/{t}']
            if kind == 'projection':
                state = fitted[:, :, t]; vel = rates; se = ses
                valid = np.isfinite(state)&np.isfinite(vel)&np.isfinite(se)&(lof <= fit_params['lack_of_fit_threshold'])
                support = rate_supported
            else:
                state = observed[:, :, t]; vel = (observed[:,:,t+1]-state)/dt[t]
                var0 = (state*exposure[:,t]/1e6+.5)*(1e6/exposure[:,t])**2
                var1 = (observed[:,:,t+1]*exposure[:,t+1]/1e6+.5)*(1e6/exposure[:,t+1])**2
                se = np.sqrt(var0+var1)/dt[t]
                valid = np.isfinite(state)&np.isfinite(vel)
                support = (np.abs(vel)>np.maximum(fit_params['rate_floor'],fit_params['rate_z']*se))&(totals>=20)
            safe = np.where(valid, vel, 0.); noise = np.linalg.norm(np.where(valid, se, 0.),axis=1)
            magnitude = np.linalg.norm(safe,axis=1)
            snr = np.divide(magnitude,noise,out=np.zeros(len(noise)),where=noise>0)
            changing = (support&valid).any(axis=1)&(snr>=2.)&(valid.sum(axis=1)>=needed)&(magnitude>0)
            stored = {k:take(group[k], sampled) for k in ['changing','active','source_snr','support_count','mean_cosine','velocity_norm','net_rate']}
            for index, i in enumerate(sampled):
                a = location[int(i)]
                assert bool(stored['changing'][index]) == bool(changing[a])
                close(stored['source_snr'][index], snr[a], 'Source vector SNR', rtol=4e-5)
                close(stored['velocity_norm'][index], magnitude[a], 'Velocity amplitude', rtol=4e-5)
                close(stored['net_rate'][index], np.mean(vel[a,valid[a]]) if valid[a].any() else np.nan, 'Signed accessibility mean', rtol=4e-5)
                selected=[]; unnormalized=[]; cosines=[]
                left,right=graph.indptr[i:i+2]
                for j, weight in zip(graph.indices[left:right],graph.data[left:right]):
                    b=location[int(j)]; shared=valid[a]&valid[b]
                    if i==j or weight<=0 or not changing[a] or shared.sum()<needed:
                        continue
                    difference=state[b,shared].astype(float)-state[a,shared].astype(float)
                    v=vel[a,shared].astype(float)
                    norm=np.linalg.norm(difference)*np.linalg.norm(v)
                    if np.linalg.norm(difference)<=1e-12 or norm<=0:
                        continue
                    cs=float(np.clip(np.dot(v,difference)/norm,-1,1))
                    if cs>0 and cs>=params['min_cosine']:
                        selected.append(int(j));unnormalized.append(float(weight)*cs**params['cosine_power']);cosines.append(cs)
                assert stored['support_count'][index] == len(selected)
                is_active=len(selected)>=params['min_neighbors']
                assert bool(stored['active'][index]) == is_active
                if not is_active:
                    selected=[];unnormalized=[];cosines=[]
                selected=np.asarray(selected,int);weight=np.asarray(unnormalized,float)
                probability=weight/weight.sum() if len(weight) else weight
                close(stored['mean_cosine'][index], np.dot(probability,cosines), 'Probability-weighted cosine')
                if kind=='projection':
                    p=group['probabilities'];l,r=p['indptr'][i:i+2]
                    np.testing.assert_array_equal(p['indices'][l:r],selected)
                    close(p['data'][l:r],probability,'Directed transition probabilities',rtol=4e-5)
                for name,xy in coords.items():
                    delta=xy[selected].astype(float)-xy[i].astype(float)
                    lengths=np.linalg.norm(delta,axis=1)
                    unit=np.divide(delta,lengths[:,None],out=np.zeros_like(delta),where=lengths[:,None]>0)
                    vector=np.sum(probability[:,None]*unit,axis=0)
                    coherence=float(np.linalg.norm(vector))
                    display=is_active and coherence>=params['min_coherence'] and coherence>1e-12 and np.count_nonzero(lengths)>=(params['min_neighbors'])
                    close(group[f'vectors/{name}'][i],vector,'Unit embedding direction projection')
                    close(group[f'coherence/{name}'][i],coherence,'Embedding coherence')
                    assert bool(group[f'projection_valid/{name}'][i]) == display
                    close(group[f'display_vectors/{name}'][i],vector if display else np.zeros(xy.shape[1]),'Display gates')
            details.append(dict(interval=t,kind=kind,sampled_peaks=len(sampled),states_loaded=len(required_rows)))
    checks['independent_projection'] = dict(passed=True, fields=details, uses_existing_atlas_edges=True,
        no_root_or_specificity_in_manual_formula=True, amplitude_and_common_mode_preserved=True,
        exhaustive_transition_support_and_row_normalization=True)


def check_viewer(f, out, coords, checks):
    html=(out/'trajectory_peak_explorer.html').read_text()
    packed=html.split('const DATA=',1)[1].split(';\nconst A=',1)[0]
    data=json.loads(packed)
    rows=np.asarray([p['global_index'] for p in data['peaks']],int)
    assert len(np.unique(rows))==len(rows)
    assert data['total_peaks']==len(f['peak_ids'])
    np.testing.assert_array_equal([p['peak'] for p in data['peaks']],take(f['peak_ids'].asstr(),rows))
    arrays={k:np.frombuffer(base64.b64decode(v['base64']),dtype=v['dtype']).reshape(v['shape']) for k,v in data['arrays'].items()}
    for key,name in [('observed','observed_cpm'),('fitted','fitted_cpm'),('mode','modes'),('fitvalid','fit_valid')]:
        close(arrays[key],take(f[name],rows),'Viewer '+key,atol=0,rtol=0)
    np.testing.assert_array_equal(arrays['condition'],f['condition_valid'][:])
    for name,key in [('umap2d','xy'),('umap3d','xyz')]:
        np.testing.assert_array_equal(arrays[key],coords[name][rows].astype('f4'))
    for prefix,kind in [('fit','projection'),('raw','raw_projection')]:
        for dim in (2,3):
            for t in range(5):
                close(arrays[f'{prefix}v{dim}'][:,t],take(f[f'{kind}/{t}/display_vectors/umap{dim}d'],rows),'Viewer vectors',atol=0,rtol=0)
                np.testing.assert_array_equal(arrays[f'{prefix}m{dim}'][:,t],take(f[f'{kind}/{t}/projection_valid/umap{dim}d'],rows))
    for t in range(5):
        close(arrays['netrate'][:,t],take(f[f'projection/{t}/net_rate'],rows).astype('f4'),'Viewer signed rate',atol=0,rtol=0)
    checks['offline_viewer'] = dict(passed=True,all_displayed_peaks=len(rows),all_exported_curves_and_arrow_fields=True)



def check_anndata(f, out, checks):
    """Check the reusable AnnData against the primary saved model artifact."""
    with h5py.File(out/'trajectory_velocity.h5ad','r') as a:
        np.testing.assert_array_equal(read_column(a['obs'],a['obs'].attrs['_index']),f['peak_ids'].asstr()[:])
        common=f['common_celltypes'][:].astype(bool)
        np.testing.assert_array_equal(read_column(a['var'],a['var'].attrs['_index']),f['celltypes'].asstr()[:][common])
        for key,name in [('X_umap','umap2d'),('X_umap_2D','umap2d'),('X_umap_3D','umap3d')]:
            np.testing.assert_array_equal(a[f'obsm/{key}'][:],f[name][:])
        rows=np.sort(np.random.default_rng(932).choice(len(f['peak_ids']),96,replace=False))
        fitted=take(f['fitted_cpm'],rows)[:,common]
        observed=take(f['observed_cpm'],rows)[:,common]
        rates=take(f['rates'],rows)[:,common]
        se=take(f['rate_se'],rows)[:,common]
        support=take(f['rate_supported'],rows)[:,common]
        close(take(a['X'],rows),fitted[:,:,0],'AnnData X is fitted 10hpf state',atol=0,rtol=0)
        times=f['stages'][:].astype(int)
        for t,stage in enumerate(times):
            close(take(a[f'layers/state_{stage}hpf'],rows),fitted[:,:,t],'AnnData stage state',atol=0,rtol=0)
            close(take(a[f'layers/observed_{stage}hpf'],rows),observed[:,:,t],'AnnData observed CPM',atol=0,rtol=0)
        for t,(left,right) in enumerate(zip(times[:-1],times[1:])):
            key=f'{left}_{right}hpf';g=f[f'projection/{t}']
            for prefix,expected in [('velocity',rates[:,:,t]),('conditional_se',se[:,:,t]),('change_supported',support[:,:,t])]:
                close(take(a[f'layers/{prefix}_{key}'],rows),expected,'AnnData interval '+prefix,atol=0,rtol=0)
            for dim in ('2d','3d'):
                close(take(a[f'obsm/velocity_umap_{dim}_{key}'],rows),take(g[f'display_vectors/umap{dim}'],rows),'AnnData arrows',atol=0,rtol=0)
                np.testing.assert_array_equal(take(a[f'obs/projected_{dim}_{key}'],rows),take(g[f'projection_valid/umap{dim}'],rows))
            for name in ('net_rate','velocity_rms','source_feature_count','source_snr','changing','active'):
                close(take(a[f'obs/{name}_{key}'],rows),take(g[name],rows),'AnnData source diagnostic '+name,atol=0,rtol=0)
            for name in ('data','indices','indptr'):
                np.testing.assert_array_equal(a[f'obsp/transition_{key}/{name}'][:],g[f'probabilities/{name}'][:])
        close(take(a['layers/velocity'],rows),rates[:,:,0],'AnnData default interval velocity',atol=0,rtol=0)
        close(take(a['obsm/velocity_umap'],rows),take(f['projection/0/display_vectors/umap2d'],rows),'AnnData default arrows',atol=0,rtol=0)
        sources=json.loads(f.attrs['source_files_json'])
        with h5py.File(sources['atlas']['path'],'r') as atlas:
            for key in ('data','indices','indptr'):
                np.testing.assert_array_equal(a[f'obsp/connectivities/{key}'][:],atlas[f'obsp/connectivities/{key}'][:])
        checks['anndata_export']=dict(passed=True,all_peak_ids=len(f['peak_ids']),all_coordinates=True,
            sampled_layer_rows=len(rows),all_interval_transition_graphs=True,original_atlas_graph_equal=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs/trajectory_velocity_v1')
    parser.add_argument('--samples',type=int,default=96)
    parser.add_argument('--skip-viewer',action='store_true')
    parser.add_argument('--viewer-only',action='store_true',help='Append viewer verification to an already passed numerical audit')
    parser.add_argument('--export-only',action='store_true',help='Append AnnData export verification to an already passed numerical audit')
    args=parser.parse_args();checks={};rng=np.random.default_rng(7461)
    if args.viewer_only or args.export_only:
        path=args.output_dir/'independent_validation.json'
        report=json.loads(path.read_text())
        assert report['passed']
        with h5py.File(args.output_dir/'trajectory_fits.h5','r') as f:
            if args.viewer_only:
                coords={name:f[name][:] for name in ('umap2d','umap3d')}
                check_viewer(f,args.output_dir,coords,report['checks'])
            if args.export_only:
                check_anndata(f,args.output_dir,report['checks'])
        path.write_text(json.dumps(report,indent=2)+'\n')
        say('Independent artifact verification PASSED; appended to numerical audit')
        return
    with h5py.File(args.output_dir/'trajectory_fits.h5','r') as f:
        params=json.loads(f.attrs['parameters_json'])
        sources=json.loads(f.attrs['source_files_json'])
        for record in sources.values():
            stat=Path(record['path']).stat()
            assert stat.st_size==record['bytes'] and stat.st_mtime_ns==record['mtime_ns']
        checks['source_files_unchanged']=dict(passed=True,scope='Source sizes and modification timestamps, not full cryptographic hashes')
        with h5py.File(sources['state']['path'],'r') as source,h5py.File(sources['atlas']['path'],'r') as atlas:
            say('Checking source identity, normalization, and independently refitted models')
            coords,rows=check_inputs_and_fits(f,source,atlas,params,rng,args.samples,checks)
            say('Checking sampled fitted/raw graph edges and 2D/3D arrow arithmetic')
            check_projections(f,atlas,coords,rng,rows,checks)
        if not args.skip_viewer:
            say('Checking all exported viewer data against saved fields')
            check_viewer(f,args.output_dir,coords,checks)
    report=dict(passed=True,checks=checks,limitations=[
        'Fits and projections are independently checked on sampled peaks, not recomputed exhaustively.',
        'Model SE checks omit exact zero-clipping boundaries, where the active-set derivative is not unique.',
        'Passing numerical verification does not establish biological validity or replicate-based confidence.'])
    (args.output_dir/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    say('Independent verification PASSED')


if __name__=='__main__':
    main()
