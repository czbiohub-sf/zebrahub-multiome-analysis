"""Independent artifact audit of regional temporal order and fit-weight timing.

This verifier reconstructs changes directly from the input CSV, uses pandas ranks
rather than the producer's rank helper, and never calls producer timing functions.
"""
from pathlib import Path
import argparse
import hashlib
from itertools import permutations
import json
import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/activation_progression_v1'
FITS=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'


def fingerprint(path):
    path=Path(path);s=path.stat()
    return dict(path=str(path.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_regional(out=OUT):
    paths=[out/name for name in ['region_time_summary.csv','regional_activation_order.csv','regional_activation_order.json','regional_activation_order_nulls.npz','geometry.h5']]
    before=[fingerprint(p) for p in paths]
    with np.load(out/'regional_activation_order_nulls.npz') as z:
        saved={key:z[key] for key in z.files}
    table=pd.read_csv(out/'region_time_summary.csv');table=table[table.common_panel]
    ct=np.sort(table.celltype.unique());regions=np.sort(table.region_code.unique());times=np.sort(table.stage.unique())
    assert np.array_equal(ct,saved['celltypes']) and np.array_equal(times,saved['stage_times'])
    assert np.array_equal(regions,saved['region_codes'])
    ordered=table.set_index(['celltype','region_code','stage']).loc[pd.MultiIndex.from_product([ct,regions,times])]
    profiles=ordered.observed_mean_cpm.to_numpy().reshape(len(ct),len(regions),len(times))
    assert ordered.condition_valid.all() and np.isfinite(profiles).all()
    np.testing.assert_array_equal(profiles,saved['observed_profiles_cpm'])
    size=ordered.n_peaks.to_numpy().reshape(profiles.shape)
    assert np.all(size==size[0,:,0][None,:,None]);size=size[0,:,0]
    np.testing.assert_array_equal(size,saved['region_sizes'])
    with h5py.File(out/'geometry.h5','r') as g:
        codes=g['region_code'][:];d=g['distance_hops'][:]
        assert np.array_equal(g['region_labels'].asstr()[:],saved['region_labels'])
    assert np.isfinite(d).all()
    np.testing.assert_array_equal(np.bincount(codes,minlength=len(regions)),size)
    region_distance=np.array([np.mean(d[codes==r],dtype=float) for r in regions])
    np.testing.assert_allclose(region_distance,saved['regional_distance'],rtol=1e-12,atol=1e-12)
    orderings=np.array(list(permutations(range(6))))
    np.testing.assert_array_equal(orderings,saved['permutations'])
    amplitude=np.ptp(profiles,axis=-1)/profiles.mean(axis=-1)
    np.testing.assert_allclose(amplitude,saved['relative_amplitude'],rtol=1e-12,atol=1e-12)
    thresholds=saved['thresholds'];eligibility=np.array([(size[None,:]>=1000)&(amplitude>=h) for h in thresholds])
    np.testing.assert_array_equal(eligibility,saved['eligibility'])
    # Explicitly rebuild each shared snapshot ordering and both change phases.
    midpoint=(times[1:]+times[:-1])/2
    windows=np.full((len(orderings),len(ct),len(regions),2),np.nan)
    for p,order in enumerate(orderings):
        values=profiles[:,:,order]
        amplitude_permuted=np.ptp(values,axis=-1)/values.mean(axis=-1)
        for h,threshold in enumerate(thresholds):
            np.testing.assert_array_equal((size[None,:]>=1000)&(amplitude_permuted>=threshold),eligibility[h])
        delta=values[:,:,1:]-values[:,:,:-1]
        for phase,sign in enumerate([1,-1]):
            amount=np.clip(sign*delta,0,None)
            total=amount.sum(axis=-1)
            windows[p,:,:,phase]=np.divide((amount*midpoint).sum(axis=-1),total,out=np.full(total.shape,np.nan),where=total>0)
    np.testing.assert_allclose(windows[0],saved['window_centers'],rtol=1e-12,atol=1e-12,equal_nan=True)
    result=np.full(saved['spearman_nulls'].shape,np.nan);counts=np.zeros(result.shape,dtype=int)
    for h in range(len(thresholds)):
        for c in range(len(ct)):
            for phase in range(2):
                age=windows[:,c,:,phase];good=eligibility[h,c][None,:]&np.isfinite(age)
                # Pandas average ranks provide an independent path through ties.
                rank_x=pd.DataFrame(np.where(good,region_distance[None,:],np.nan)).rank(axis=1,method='average').to_numpy()
                rank_y=pd.DataFrame(np.where(good,age,np.nan)).rank(axis=1,method='average').to_numpy()
                count=good.sum(axis=1);counts[h,c,phase]=count
                centered_x=rank_x-np.nanmean(rank_x,axis=1)[:,None]
                centered_y=rank_y-np.nanmean(rank_y,axis=1)[:,None]
                numerator=np.nansum(centered_x*centered_y,axis=1)
                denominator=np.sqrt(np.nansum(centered_x**2,axis=1)*np.nansum(centered_y**2,axis=1))
                result[h,c,phase]=np.divide(numerator,denominator,out=np.full(len(orderings),np.nan),where=(count>=6)&(denominator>0))
    np.testing.assert_array_equal(counts,saved['phase_region_counts'])
    np.testing.assert_allclose(result,saved['spearman_nulls'],rtol=1e-12,atol=1e-12,equal_nan=True)
    maximum=np.nanmax(result[0],axis=(0,1))
    np.testing.assert_allclose(maximum,saved['primary_family_max_null'],rtol=1e-12,atol=1e-12)
    assert np.isfinite(result).all(), 'This completed dataset should have all720 valid orderings for every statistic'
    csv=pd.read_csv(out/'regional_activation_order.csv')
    for h,threshold in enumerate(thresholds):
        for c,name in enumerate(ct):
            for phase,label in enumerate(['opening','closing']):
                row=csv[(csv.celltype==name)&(csv.phase==label)&(csv.relative_amplitude_threshold==threshold)]
                assert len(row)==1;row=row.iloc[0];null=result[h,c,phase];observed=null[0]
                assert row.eligible_regions==eligibility[h,c].sum() and row.phase_regions==counts[h,c,phase,0]
                assert row.valid_permutations==720 and row.minimum_regions==6 and row.minimum_region_peaks==1000
                expected={'spearman':observed,'greater_tail_fraction':np.mean(null>=observed-1e-12),'null_median':np.median(null),'null_q05':np.quantile(null,.05),'null_q95':np.quantile(null,.95),'phase_regions_null_min':counts[h,c,phase].min(),'phase_regions_null_max':counts[h,c,phase].max()}
                for key,value in expected.items():assert np.isclose(row[key],value,rtol=1e-12,atol=1e-12),(name,label,key)
                if h==0:assert np.isclose(row.primary_family_max_tail_fraction,np.mean(maximum>=observed-1e-12),rtol=1e-12,atol=1e-12)
                else:assert np.isnan(row.primary_family_max_tail_fraction)
    summary=json.loads((out/'regional_activation_order.json').read_text())
    assert summary['raw_profile_csv_sha256']==sha(out/'region_time_summary.csv')
    assert np.isclose(summary['global_primary_observed_max'],maximum[0])
    assert np.isclose(summary['global_primary_max_tail_fraction'],np.mean(maximum>=maximum[0]-1e-12))
    primary=csv[csv.relative_amplitude_threshold==.1]
    assert summary['primary_nominal_tail_at_most_05']==int((primary.greater_tail_fraction<=.05).sum())
    assert summary['primary_family_max_tail_at_most_05']==int((primary.primary_family_max_tail_fraction<=.05).sum())
    assert before==[fingerprint(p) for p in paths]
    audit=dict(status='passed',source_files_unchanged=True,n_celltypes=len(ct),n_regions=len(regions),n_permutations=len(orderings),
               raw_profile_and_geometry_alignment=True,all_720_opening_closing_windows_recomputed=True,
               all_filters_temporal_order_invariant=True,all_null_correlations_and_phase_region_counts_recomputed=True,
               family_maximum_uses_all_42_primary_statistics=True,all_84_table_rows_recomputed=True,
               every_statistic_has_720_valid_permutations=True,global_max_correlation=float(maximum[0]),
               global_max_greater_tail_numerator=int((maximum>=maximum[0]-1e-12).sum()),
               global_max_greater_tail_fraction=float(np.mean(maximum>=maximum[0]-1e-12)),
               primary_individual_tails_at_most_05=int((primary.greater_tail_fraction<=.05).sum()),
               interpretation='No regional opening/closing chronology exceeds this shared-order reference; this does not exclude finer local programs or prove absence of a developmental wave.',
               source_sha256={p.name:sha(p) for p in paths[:-1]})
    (out/'verification_regional_activation_order.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    return audit


def verify_weight_timing(out=OUT):
    paths=[out/'fit_weight_sample.npz',out/'fit_weight_timing_sensitivity.csv',out/'activation_metrics.h5',FITS]
    before=[fingerprint(p) for p in paths]
    with np.load(out/'fit_weight_sample.npz') as z:
        saved={key:z[key] for key in z.files}
    with h5py.File(FITS,'r') as f,h5py.File(out/'activation_metrics.h5','r') as a:
        ct=f['celltypes'].asstr()[:];times=f['stages'][:];common=f['common_celltypes'][:].astype(bool)
        condition=f['condition_valid'][:].astype(bool);coverage=f['coverage'][:]
        params=json.loads(f.attrs['parameters_json']);timing=json.loads(a.attrs['timing_parameters_json'])
    y=saved['observed_cpm'];safe=np.where(np.isfinite(coverage)&(coverage>0),coverage,1.)
    counts=np.rint(y*safe[None,:,:]/1e6);count_supported=np.nansum(np.where(condition[None,:,:],counts,np.nan),axis=-1)>=20
    reconstructed={};max_error=0.
    for arm in ['coverage','equal']:
        rates=saved[arm+'_rates'];support=saved[arm+'_rate_supported'].astype(bool)
        np.testing.assert_allclose(np.diff(saved[arm+'_fitted'],axis=-1)/np.diff(times),rates,rtol=1e-10,atol=1e-10,equal_nan=True)
        quality=saved[arm+'_fit_valid'].astype(bool)&count_supported&(saved[arm+'_lack_of_fit']<=params['lack_of_fit_threshold'])&common[None,:]
        for phase,sign in [('opening',1),('closing',-1)]:
            speed=np.clip(sign*rates,0,None);increments=speed*np.diff(times);total=increments.sum(axis=-1)
            maximum=speed.max(axis=-1);modulation=np.divide(maximum-speed.min(axis=-1),maximum,out=np.zeros_like(total),where=maximum>1e-10)
            resolved=quality&np.isfinite(rates).all(axis=-1)&(support&(rates*sign>0)).any(axis=-1)&(total>=timing['min_positive_change_cpm'])&(modulation>=timing['min_relative_rate_modulation'])
            center=np.divide((increments*((times[1:]+times[:-1])/2)).sum(axis=-1),total,out=np.full(total.shape,np.nan),where=total>1e-10)
            center[~resolved]=np.nan;reconstructed[(arm,phase)]=center
            np.testing.assert_array_equal(resolved,saved[f'{arm}_{phase}_resolved'])
            np.testing.assert_allclose(total,saved[f'{arm}_{phase}_change_cpm'],rtol=1e-12,atol=1e-12,equal_nan=True)
            np.testing.assert_allclose(center,saved[f'{arm}_{phase}_center_hpf'],rtol=1e-12,atol=1e-12,equal_nan=True)
            if resolved.any():max_error=max(max_error,float(np.max(np.abs(center[resolved]-saved[f'{arm}_{phase}_center_hpf'][resolved]))))
    table=pd.read_csv(out/'fit_weight_timing_sensitivity.csv');selection=[]
    for c in np.flatnonzero(common):
        for phase in ['opening','closing']:
            x=reconstructed[('coverage',phase)][:,c];y=reconstructed[('equal',phase)][:,c]
            xp=np.isfinite(x);yp=np.isfinite(y);both=xp&yp;union=xp|yp
            row=table[(table.celltype==ct[c])&(table.phase==phase)];assert len(row)==1;row=row.iloc[0]
            assert row.coverage_resolved==xp.sum() and row.equal_resolved==yp.sum() and row.both_resolved==both.sum()
            shift=float(np.median(np.abs(x[both]-y[both]))) if both.any() else np.nan
            rho=float(spearmanr(x[both],y[both]).statistic) if both.sum()>2 and np.ptp(x[both])>1e-8 and np.ptp(y[both])>1e-8 else np.nan
            assert np.isclose(row.median_absolute_timing_shift,shift,rtol=1e-12,atol=1e-12,equal_nan=True)
            assert np.isclose(row.timing_spearman,rho,rtol=1e-12,atol=1e-12,equal_nan=True)
            selection.append(dict(celltype=ct[c],phase=phase,coverage_count=int(xp.sum()),equal_count=int(yp.sum()),paired_count=int(both.sum()),primary_retention=float(both.sum()/xp.sum()) if xp.any() else None,selection_jaccard=float(both.sum()/union.sum()) if union.any() else None))
    assert before==[fingerprint(p) for p in paths]
    result=dict(status='passed',source_files_unchanged=True,n_sampled_peaks=len(saved['atlas_rows']),n_celltypes=len(ct),
                rates_equal_actual_hour_fitted_differences=True,both_weight_arms_all_timing_masks_recomputed=True,
                both_weight_arms_all_timing_centers_recomputed=True,all42_timing_sensitivity_rows_recomputed=True,
                maximum_timing_difference_hpf=max_error,selection_overlap=selection,
                scope='Conditional timing/support arithmetic and saved tables checked; no biological replicate confidence inferred')
    (out/'verification_weight_timing.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--stage',choices=['regional','weights','all'],default='all');args=p.parse_args();result={}
    if args.stage in ['regional','all']:result['regional']=verify_regional(args.output_dir)
    if args.stage in ['weights','all']:result['weights']=verify_weight_timing(args.output_dir)
    print(json.dumps(result,indent=2,allow_nan=False))
