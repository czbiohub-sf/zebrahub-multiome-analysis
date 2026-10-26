"""Independent read-only numerical audit of activation progression artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import h5py
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from real_specialization_benchmark import read_frame, read_graph

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/activation_progression_v1'
FITS = HERE / 'outputs/trajectory_velocity_v1/trajectory_fits.h5'
ATLAS = HERE.parents[2] / 'data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad'


def stamp(path):
    value = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view('u1')).hexdigest()


def check_shortest_paths(graph, reference, distances, *, weighted=False, block_size=32768):
    """Check shortest paths using all edges, without calling the producer solver.

    Positive edge costs plus zero values only at references and a matching
    minimum-cost predecessor certify shortest distances to the reference set.
    Unreachable components must be infinite; this is checked independently.
    """
    reference = np.asarray(reference, dtype=bool)
    d = np.asarray(distances, dtype=float)
    _, components = connected_components(graph, directed=False)
    has_reference = np.bincount(components[reference], minlength=components.max() + 1) > 0
    reachable = has_reference[components]
    assert np.array_equal(np.isfinite(d), reachable), 'Reachability differs from original graph components'
    assert not np.isnan(d).any(), 'NaN graph distance'
    assert np.array_equal(d == 0, reference), 'Distance zero must match exactly the reference mask'
    assert (d[reachable & ~reference] > 0).all(), 'Nonreference reachable nodes need positive distance'
    maximum_error = 0.
    for start in range(0, len(d), block_size):
        stop = min(start + block_size, len(d))
        a, b = graph.indptr[start], graph.indptr[stop]
        source = np.repeat(np.arange(stop - start), np.diff(graph.indptr[start:stop + 1]))
        cost = 1. - np.log(np.maximum(graph.data[a:b].astype(float), 1e-6)) if weighted else np.ones(b-a)
        candidate = d[graph.indices[a:b]] + cost
        best = np.full(stop-start, np.inf)
        np.minimum.at(best, source, candidate)
        use = reachable[start:stop] & ~reference[start:stop]
        err = np.abs(d[start:stop][use] - best[use])
        if len(err):
            maximum_error = max(maximum_error, float(err.max()))
        assert np.allclose(d[start:stop][use], best[use], rtol=3e-6, atol=3e-6), 'A nonreference distance lacks a shortest predecessor'
        edge_finite = np.isfinite(candidate) & np.isfinite(d[start:stop][source])
        assert np.all(d[start:stop][source][edge_finite] <= candidate[edge_finite] + 3e-6 * (1 + candidate[edge_finite])), 'Edge violates shortest-path inequality'
    return dict(reachable=int(reachable.sum()), unreachable=int((~reachable).sum()), reference=int(reference.sum()), maximum_bellman_residual=maximum_error, checked_directed_edges=int(graph.nnz))


def verify_geometry(out=OUT):
    paths = [ATLAS, FITS, out/'geometry.h5']
    if (out/'geometry_core_reference.h5').exists():
        paths.append(out/'geometry_core_reference.h5')
    before = [stamp(path) for path in paths]
    with h5py.File(ATLAS, 'r') as a, h5py.File(FITS, 'r') as f, h5py.File(out/'geometry.h5', 'r') as g:
        ids = f['peak_ids'].asstr()[:]
        obs = read_frame(a['obs'], ['leiden_coarse'])
        assert np.array_equal(ids, obs.index.astype(str).to_numpy())
        assert np.array_equal(ids, g['peak_ids'].asstr()[:])
        assert len(np.unique(ids)) == len(ids)
        for key, original in [('umap2d', 'X_umap_2D'), ('umap3d', 'X_umap_3D')]:
            assert np.array_equal(g[key][:], f[key][:])
            assert np.array_equal(g[key][:], a[f'obsm/{original}'][:])
        reference = g['broad_reference'][:].astype(bool)
        assert np.array_equal(reference, f['broad_reference'][:].astype(bool))
        labels = g['region_labels'].asstr()[:]
        codes = g['region_code'][:]
        assert np.array_equal(labels[codes], obs.leiden_coarse.astype(str).to_numpy())
        graph = read_graph(a['obsp/connectivities'])
        saved_summary = json.loads(g.attrs['summary_json'])
        for key in ['data', 'indices', 'indptr']:
            assert saved_summary['graph_hashes_sha256'][key] == digest(getattr(graph, key))
        checks = {
            'hops': check_shortest_paths(graph, reference, g['distance_hops'][:]),
            'weighted': check_shortest_paths(graph, reference, g['distance_weighted'][:], weighted=True),
        }
        counts, weights = g['region_edge_count'][:], g['region_edge_affinity'][:]
        assert np.array_equal(counts, counts.T)
        assert np.allclose(weights, weights.T, rtol=1e-8, atol=1e-8)
        assert counts.sum() == graph.nnz
        assert np.isclose(weights.sum(), graph.data.sum(dtype=float))
        primary_hops = g['distance_hops'][:]
    if (out/'geometry_core_reference.h5').exists():
        with h5py.File(out/'geometry_core_reference.h5', 'r') as core:
            assert np.array_equal(ids, core['peak_ids'].asstr()[:])
            mask = core['core_reference'][:].astype(bool)
            assert np.all(~mask | reference)
            assert mask.sum() >= .9 * reference.sum()
            summary = json.loads(core.attrs['summary_json'])
            selected = np.asarray(summary['selected_region_codes'])
            region_counts = np.bincount(codes[reference], minlength=len(labels))
            order = np.lexsort((np.arange(len(labels)), -region_counts))
            number = np.searchsorted(np.cumsum(region_counts[order]), .9 * reference.sum(), side='left') + 1
            assert np.array_equal(selected, order[:number])
            assert np.array_equal(mask, reference & np.isin(codes, selected))
            checks['core_hops'] = check_shortest_paths(graph, mask, core['distance_core_hops'][:])
            checks['core_weighted'] = check_shortest_paths(graph, mask, core['distance_core_weighted'][:], weighted=True)
            assert np.all(core['distance_core_hops'][:] >= primary_hops)
    after = [stamp(path) for path in paths]
    assert before == after, 'Source artifacts changed during read-only validation'
    result = dict(verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_peaks=len(ids),
                  n_regions=len(labels), source_files=before, source_unchanged=True,
                  exact_peak_order=True, exact_umap_coordinates=True,
                  broad_reference_identical=True, region_labels_identical=True,
                  original_graph_hashes_identical=True, region_graph_accounting=True,
                  distance_checks=checks, status='passed')
    (out/'verification_geometry.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result



def verify_aggregation(out=OUT, block_size=16384):
    """Reproduce every raw/fitted center and every timing mask in one block pass."""
    paths = [FITS, out/'activation_metrics.h5', out/'geometry.h5', out/'geometry_core_reference.h5']
    before = [stamp(path) for path in paths]
    with h5py.File(out/'geometry.h5', 'r') as g, h5py.File(out/'geometry_core_reference.h5', 'r') as core:
        distance = dict(hops=g['distance_hops'][:], weighted=g['distance_weighted'][:],
                        core_hops=core['distance_core_hops'][:], core_weighted=core['distance_core_weighted'][:])
        region = g['region_code'][:]; labels=g['region_labels'].asstr()[:]
        geo_ids = g['peak_ids'].asstr()[:]
    with h5py.File(FITS, 'r') as f, h5py.File(out/'activation_metrics.h5', 'r') as a:
        ids=f['peak_ids'].asstr()[:]; ct=f['celltypes'].asstr()[:]
        assert np.array_equal(ids, geo_ids) and np.array_equal(ids, a['peak_ids'].asstr()[:])
        assert np.array_equal(ct, a['celltypes'].asstr()[:])
        times=f['stages'][:]; condition=f['condition_valid'][:].astype(bool); common=f['common_celltypes'][:].astype(bool)
        assert np.array_equal(times, a['stages'][:]) and np.array_equal(common, a['common_celltypes'][:])
        assert np.array_equal(labels, a['region_labels'].asstr()[:])
        metrics=a['distance_metrics'].asstr()[:]; n,C,T=len(ids),len(ct),len(times); R=len(labels)
        assert C == 32 and common.sum() == 21
        params=json.loads(f.attrs['parameters_json']); timing=json.loads(a.attrs['timing_parameters_json'])
        numerators={kind:{metric:np.zeros((R,C,T)) for metric in metrics} for kind in ['observed','fitted']}
        denominators={kind:{metric:np.zeros((R,C,T)) for metric in metrics} for kind in ['observed','fitted']}
        peakcounts={kind:np.zeros((R,C), dtype=np.int64) for kind in ['observed','fitted']}
        max_timing_error=0.; resolved_counts={phase:np.zeros(C,dtype=np.int64) for phase in ['opening','closing']}
        duration=np.diff(times); midpoint=(times[:-1]+times[1:])/2
        for start in range(0,n,block_size):
            stop=min(n,start+block_size); nr=stop-start
            grouping=np.broadcast_to(region[start:stop,None,None]*(C*T)+np.arange(C*T).reshape(1,C,T),(nr,C,T)).ravel()
            y=f['observed_cpm'][start:stop].astype(float); fitted=f['fitted_cpm'][start:stop].astype(float)
            observed_stable=np.where(condition[None,:,:],np.isfinite(y),True).all(axis=-1)&condition.any(axis=-1)[None,:]
            fitted_stable=np.where(condition[None,:,:],np.isfinite(fitted),True).all(axis=-1)&condition.any(axis=-1)[None,:]
            root_reachable=np.isfinite(distance['hops'][start:stop])[:,None]
            for kind,values,stable in [('observed',y,observed_stable&root_reachable),('fitted',fitted,observed_stable&fitted_stable&root_reachable)]:
                for c in range(C):peakcounts[kind][:,c]+=np.bincount(region[start:stop][stable[:,c]],minlength=R)
                for metric in metrics:
                    d=distance[metric][start:stop].astype(float); fixed=stable&np.isfinite(d)[:,None]
                    weights=np.where(fixed[:,:,None]&condition[None,:,:],values,0.)
                    assert np.isfinite(weights).all() and (weights>=0).all()
                    denominators[kind][metric]+=np.bincount(grouping,weights=weights.ravel(),minlength=R*C*T).reshape(R,C,T)
                    numweights=weights*np.where(np.isfinite(d),d,0.)[:,None,None]
                    numerators[kind][metric]+=np.bincount(grouping,weights=numweights.ravel(),minlength=R*C*T).reshape(R,C,T)
            rates=f['rates'][start:stop].astype(float); support=f['rate_supported'][start:stop].astype(bool)
            eligible=f['fit_valid'][start:stop].astype(bool)&(f['lack_of_fit'][start:stop]<=params['lack_of_fit_threshold'])&common[None,:]
            for phase,sign in [('opening',1),('closing',-1)]:
                amplitude=np.clip(rates*sign,0,None); total=(amplitude*duration).sum(axis=-1)
                peak=amplitude.max(axis=-1); modulation=np.divide(peak-amplitude.min(axis=-1),peak,out=np.zeros_like(total),where=peak>1e-10)
                resolved=eligible&np.isfinite(rates).all(axis=-1)&(support&(rates*sign>0)).any(axis=-1)&(total>=timing['min_positive_change_cpm'])&(modulation>=timing['min_relative_rate_modulation'])
                center=np.divide((amplitude*duration*midpoint).sum(axis=-1),total,out=np.full_like(total,np.nan),where=total>1e-10)
                center[~resolved]=np.nan
                assert np.array_equal(resolved,a[f'{phase}_resolved'][start:stop])
                saved=a[f'{phase}_center_hpf'][start:stop]
                assert np.array_equal(np.isfinite(center),np.isfinite(saved))
                assert np.allclose(center,saved,rtol=1e-6,atol=1e-6,equal_nan=True)
                if resolved.any():max_timing_error=max(max_timing_error,float(np.max(np.abs(center[resolved]-saved[resolved]))))
                resolved_counts[phase]+=resolved.sum(axis=0)
            if start%(block_size*10)==0:print(f'Checked aggregation/timing {stop:,}/{n:,}',flush=True)
        errors={}
        for kind in ['observed','fitted']:
            expected=np.full((C,R+1,len(metrics),T),np.nan)
            for mi,metric in enumerate(metrics):
                numerator=numerators[kind][metric];denominator=denominators[kind][metric]
                values=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
                expected[:,:R,mi,:]=values.transpose(1,0,2)
                numerator=numerator.sum(axis=0);denominator=denominator.sum(axis=0)
                expected[:,R,mi,:]=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
            saved=a[f'{kind}_distance_centers'][:]
            assert np.array_equal(np.isfinite(expected),np.isfinite(saved))
            assert np.allclose(expected,saved,rtol=1e-10,atol=1e-10,equal_nan=True)
            errors[kind]=float(np.max(np.abs(expected[np.isfinite(expected)]-saved[np.isfinite(expected)])))
            assert np.isfinite(saved[common]).all(), 'Primary panel has missing temporal centers'
        table=pd.read_csv(out/'region_time_summary.csv')
        for c,name in enumerate(ct):
            for ti,time_hpf in enumerate(times):
                rows=table[(table.celltype==name)&(table.stage==time_hpf)].sort_values('region_code')
                assert len(rows)==R and (rows.condition_valid==condition[c,ti]).all()
                assert np.array_equal(rows.n_peaks,peakcounts['observed'][:,c])
                if not condition[c,ti]:
                    assert rows.observed_mean_cpm.isna().all();continue
                obs_sum=denominators['observed']['hops'][:,c,ti]
                np.testing.assert_allclose(rows.observed_mean_cpm,obs_sum/peakcounts['observed'][:,c],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(rows.accessibility_mass_fraction,obs_sum/obs_sum.sum(),rtol=1e-10,atol=1e-10)
        measured=a['observed_distance_centers'][:]
    assert before==[stamp(path) for path in paths]
    result=dict(status='passed',source_unchanged=True,source_files=before,n_peaks=n,
                exact_peak_celltype_time_alignment=True,all_distance_centers_recomputed=True,
                maximum_distance_center_errors=errors,all_opening_closing_masks_recomputed=True,
                maximum_timing_roundtrip_error_hpf=max_timing_error,
                all_region_mean_and_mass_fractions_recomputed=True,
                primary_celltype_panel_complete=True,
                opening_resolved_by_celltype=dict(zip(ct,resolved_counts['opening'].tolist())),
                closing_resolved_by_celltype=dict(zip(ct,resolved_counts['closing'].tolist())))
    (out/'verification_aggregation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def verify_temporal_controls(out=OUT):
    """Independent matrix calculation of all 720 raw-stage permutation controls."""
    from itertools import permutations
    with h5py.File(out/'activation_metrics.h5','r') as f:
        common=f['common_celltypes'][:].astype(bool);ct=f['celltypes'].asstr()[:][common];times=f['stages'][:]
        measured=f['observed_distance_centers'][:];metrics=f['distance_metrics'].asstr()[:]
        hops=int(np.flatnonzero(metrics=='hops')[0]);series=measured[common,-1,hops,:];regions=measured[common,:-1,hops,:]
    orderings=np.array(list(permutations(range(len(times)))))
    with np.load(out/'temporal_order_nulls.npz') as saved:
        assert np.array_equal(orderings,saved['permutations'])
        x=times-times.mean();xx=np.sum(x*x)
        def matrix_reference(y):
            centered=y-y.mean(axis=-1,keepdims=True)
            ss=np.sum(centered*centered,axis=-1)
            permuted=centered[:,orderings].transpose(1,0,2)
            cov=np.tensordot(permuted,x,axes=([-1],[0]))
            slopes=cov/xx
            corrs=np.divide(cov,np.sqrt(xx*ss)[None,:],out=np.zeros_like(cov),where=ss[None,:]>1e-24)
            return slopes,corrs
        slopes,corrs=matrix_reference(series)
        _,rcorrs=matrix_reference(regions.reshape(-1,len(times)))
        np.testing.assert_allclose(slopes,saved['null_ct_slopes'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(corrs,saved['null_ct_correlations'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(rcorrs.reshape(720,*regions.shape[:-1]),saved['null_region_correlations'],rtol=1e-10,atol=1e-10,equal_nan=True)
        np.testing.assert_allclose(slopes.mean(axis=1),saved['null_global_mean_slope'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose((corrs>1e-12).mean(axis=1),saved['null_global_fraction_positive'],rtol=1e-10,atol=1e-10)
    table=pd.read_csv(out/'outward_tests.csv')
    tail_columns=['p_outward','p_inward','p_two_sided','p_fwer_two_sided']
    assert table.loc[table.estimator=='fitted',tail_columns].isna().all().all(), 'Fitted sensitivity must not use naive postfit permutations'
    actual=table[(table.scope=='celltype')&(table.estimator=='observed')&(table.metric=='hops')].set_index('celltype').loc[ct]
    np.testing.assert_allclose(actual.slope,slopes[0],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(actual.correlation,corrs[0],rtol=1e-10,atol=1e-10)
    for col,expected in [('p_outward',(corrs>=corrs[0]-1e-12).mean(axis=0)),('p_inward',(corrs<=corrs[0]+1e-12).mean(axis=0)),('p_two_sided',(np.abs(corrs)>=np.abs(corrs[0])-1e-12).mean(axis=0)),('p_fwer_two_sided',(np.max(np.abs(corrs),axis=1)[:,None]>=np.abs(corrs[0])-1e-12).mean(axis=0))]:
        np.testing.assert_allclose(actual[col],expected,atol=1e-12,rtol=1e-12)
    result=dict(status='passed',n_shared_permutations=720,n_primary_celltypes=len(ct),n_region_celltype_series=int(np.prod(regions.shape[:-1])),all_primary_null_slopes_correlations_recomputed=True,all_primary_tail_fractions_recomputed=True,all_region_null_correlations_recomputed=True,global_mean_slope_and_positive_fraction_nulls_recomputed=True)
    (out/'verification_temporal_controls.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result



def verify_weight_sensitivity(out=OUT):
    """Check sampled fitted-center summaries and raw column-scale cancellation."""
    with h5py.File(out/'geometry.h5','r') as g, h5py.File(out/'geometry_core_reference.h5','r') as core:
        distance=dict(hops=g['distance_hops'][:],weighted=g['distance_weighted'][:],core_hops=core['distance_core_hops'][:],core_weighted=core['distance_core_weighted'][:])
    source_before=stamp(FITS)
    with np.load(out/'fit_weight_sample.npz') as sample:
        rows=sample['atlas_rows'];observed=sample['observed_cpm'];fits={key:sample[key+'_fitted'] for key in ['coverage','equal']}
    with h5py.File(FITS,'r') as f:
        times=f['stages'][:];ct=f['celltypes'].asstr()[:];common=f['common_celltypes'][:].astype(bool);exposure=f['coverage'][:]
        expected_rows=np.sort(np.random.default_rng(20261020).choice(len(f['peak_ids']),8192,replace=False))
        assert np.array_equal(rows,expected_rows)
        stored_observed=np.concatenate([f['observed_cpm'][rows[i:i+512]] for i in range(0,len(rows),512)],axis=0)
        stored_fitted=np.concatenate([f['fitted_cpm'][rows[i:i+512]] for i in range(0,len(rows),512)],axis=0)
        assert np.array_equal(stored_observed,observed,equal_nan=True)
        np.testing.assert_allclose(stored_fitted,fits['coverage'],rtol=1e-4,atol=2e-5,equal_nan=True)
    table=pd.read_csv(out/'fit_weight_progression_sensitivity.csv')
    max_scale_error=0.; max_summary_error=0.
    for c in np.flatnonzero(common):
        for metric,dall in distance.items():
            d=dall[rows];curves=[]
            for key in ['coverage','equal']:
                values=fits[key][:,c,:];fixed=np.isfinite(values).all(axis=-1)&np.isfinite(d)
                assert fixed.any()
                curves.append((values[fixed]*d[fixed,None]).sum(axis=0)/values[fixed].sum(axis=0))
            row=table[(table.celltype==ct[c])&(table.metric==metric)]
            assert len(row)==1;row=row.iloc[0]
            slopes=[np.polyfit(times,y,1)[0] for y in curves]
            for field,expected in [('coverage_slope',slopes[0]),('equal_slope',slopes[1]),('coverage_endpoint_shift',curves[0][-1]-curves[0][0]),('equal_endpoint_shift',curves[1][-1]-curves[1][0]),('max_distance_center_difference',np.max(np.abs(curves[0]-curves[1])))]:
                assert np.isclose(row[field],expected,rtol=1e-10,atol=1e-10)
                max_summary_error=max(max_summary_error,abs(row[field]-expected))
            assert row.same_slope_sign==(np.sign(slopes[0])==np.sign(slopes[1]))
            raw=observed[:,c,:];fixed=np.isfinite(raw).all(axis=-1)&np.isfinite(d)
            expected=(raw[fixed]*d[fixed,None]).sum(axis=0)/raw[fixed].sum(axis=0)
            # This is the exact columnwise conversion from CPM to count units;
            # no rounding or fresh normalization is introduced.
            count_units=raw[fixed]*exposure[c][None,:]/1e6
            actual=(count_units*d[fixed,None]).sum(axis=0)/count_units.sum(axis=0)
            assert np.allclose(expected,actual,rtol=1e-12,atol=1e-12)
            max_scale_error=max(max_scale_error,float(np.max(np.abs(expected-actual))))
    timing=pd.read_csv(out/'fit_weight_timing_sensitivity.csv')
    assert (timing.both_resolved<=timing.coverage_resolved).all()
    assert (timing.both_resolved<=timing.equal_resolved).all()
    assert (timing.median_absolute_timing_shift.dropna()>=0).all()
    assert (timing.timing_spearman.dropna().abs()<=1+1e-12).all()
    assert source_before==stamp(FITS)
    result=dict(status='passed',source_unchanged=True,n_uniform_sampled_peaks=len(rows),
                exact_sample_and_observed_values=True,primary_refit_matches_saved_states=True,
                all_fitted_distance_center_sensitivities_recomputed=True,
                maximum_summary_error=max_summary_error,
                measured_distance_centers_invariant_to_column_CPM_count_conversion=True,
                maximum_column_scale_cancellation_error=max_scale_error,
                timing_coverage_counts_self_consistent=True,
                scope='Centers and slopes independently reproduced; timing uncertainty/model-selection gates tested separately, not inferred from fitted state arrays alone')
    (out/'verification_weight_sensitivity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--stage', choices=['geometry','aggregation','controls','weights','all'], default='all')
    args = parser.parse_args()
    results={}
    if args.stage in ['geometry','all']:results['geometry']=verify_geometry(args.output_dir)
    if args.stage in ['aggregation','all']:results['aggregation']=verify_aggregation(args.output_dir)
    if args.stage in ['controls','all']:results['controls']=verify_temporal_controls(args.output_dir)
    if args.stage in ['weights','all']:results['weights']=verify_weight_sensitivity(args.output_dir)
    print(json.dumps(results, indent=2, allow_nan=False))
