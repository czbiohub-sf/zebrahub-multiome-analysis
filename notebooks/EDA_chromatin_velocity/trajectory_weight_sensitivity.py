"""Read-only sampled comparison of coverage versus equal-stage fit weights.

Both arms retain identical linear CPM, observed/missing entries, conditional
count variances, model candidates, complexity penalty, and effect/support gates.
Only the relative least-squares/LOOCV weights change. This is a sensitivity
analysis, not an alternative normalization, retuning, or replacement atlas.
Run only after the producer has closed trajectory_fits.h5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/chrom_velo'))
from trajectory_modes import fit_trajectories,MODE_LABELS
from trajectory_classification import classify_supported_modes

DEFAULT_SOURCE=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'
DEFAULT_OUTPUT=HERE/'outputs/trajectory_velocity_v1/weight_sensitivity'


def _stat(path):
    stat=path.stat()
    return dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def _json(value):
    if isinstance(value,dict):return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    if isinstance(value,np.ndarray):return _json(value.tolist())
    if isinstance(value,np.generic):return _json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _take(dataset,indices,block_size):
    """Sorted bounded HDF5 point selections; never load the full atlas arrays."""
    if not len(indices):return np.empty((0,)+dataset.shape[1:],dtype=dataset.dtype)
    return np.concatenate([dataset[indices[j:j+block_size]]
                           for j in range(0,len(indices),block_size)],axis=0)


def _mean(values):
    return float(np.mean(values)) if np.size(values) else np.nan


def pair_metrics(a,b,mask=None):
    use=np.isfinite(a)&np.isfinite(b)
    if mask is not None:use &= mask
    x,y=np.asarray(a)[use].astype(float),np.asarray(b)[use].astype(float)
    difference=y-x
    result=dict(n=int(len(x)),rmse=float(np.sqrt(np.mean(difference**2))) if len(x) else np.nan,
                median_abs_difference=float(np.median(np.abs(difference))) if len(x) else np.nan,
                p90_abs_difference=float(np.quantile(np.abs(difference),.9)) if len(x) else np.nan,
                coverage_weighted_median=float(np.median(x)) if len(x) else np.nan,
                equal_weighted_median=float(np.median(y)) if len(x) else np.nan,
                coverage_weighted_positive_fraction=_mean(x>0),equal_weighted_positive_fraction=_mean(y>0))
    varying=len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0
    result['pearson']=float(np.corrcoef(x,y)[0,1]) if varying else np.nan
    result['spearman']=float(spearmanr(x,y).statistic) if varying else np.nan
    return result


def _fit(values,times,condition_valid,exposure,variance,count_supported,params,equal):
    weights=np.where(condition_valid,1. if equal else exposure,0.)
    result=fit_trajectories(values,times,observed=condition_valid,weights=weights,
                            measurement_variance=variance,**params)
    result['rate_supported'] &= count_supported[...,None]
    result['mode']=classify_supported_modes(result['rates'],result['rate_supported'],
        quality_valid=result['lack_of_fit']<=params['lack_of_fit_threshold'],rate_floor=params['rate_floor'])
    result['mode'][~result['fit_valid']|~count_supported]=0
    return result


def _source(result,panel,params):
    rate=result['rates'][:,panel]
    se=result['rate_se'][:,panel]
    state=result['fitted'][:,panel,:-1]
    valid=np.isfinite(state)&np.isfinite(rate)&np.isfinite(se)&(result['lack_of_fit'][:,panel]<=params['lack_of_fit_threshold'])[...,None]
    vectors=np.where(valid,rate,0.)
    magnitude=np.linalg.norm(vectors,axis=1)
    noise=np.linalg.norm(np.where(valid,se,0.),axis=1)
    snr=np.divide(magnitude,noise,out=np.zeros_like(magnitude),where=noise>0)
    active=(result['rate_supported'][:,panel]&valid).any(axis=1)&(snr>=2.)
    return dict(active=active,snr=snr,valid=valid,vectors=vectors)


def run(source=DEFAULT_SOURCE,output=DEFAULT_OUTPUT,n_peaks=8192,seed=20261020,read_block_size=512):
    if not 1<=n_peaks<=16384:raise ValueError('Use at most 16,384 uniformly sampled peaks')
    if read_block_size<1:raise ValueError('read_block_size must be positive')
    source,output=Path(source),Path(output)
    before=_stat(source)
    with h5py.File(source,'r') as f:
        n=len(f['peak_ids'])
        indices=np.sort(np.random.default_rng(seed).choice(n,min(n,n_peaks),replace=False))
        times=f['stages'][:].astype(float)
        exposure=f['coverage'][:].astype(float)
        condition_valid=f['condition_valid'][:].astype(bool)
        common=f['common_celltypes'][:].astype(bool)
        celltypes=f['celltypes'].asstr()[:]
        ids=_take(f['peak_ids'],indices,read_block_size).astype(str)
        params=json.loads(f.attrs['parameters_json'])
        projection_params=json.loads(f.attrs.get('projection_parameters_json','{"min_shared_features":5,"min_shared_fraction":0.7}'))
        # The saved values are float32 CPM. Reconstruct integer counts only for
        # the common conditional variance. Both arms use identical saved CPM.
        values=_take(f['observed_cpm'],indices,read_block_size).astype(float)
        stored={name:_take(f[name],indices,read_block_size) for name in
                ['fitted_cpm','rates','rate_se','rate_supported','modes','models','raw_total','lack_of_fit','fit_valid']}
    exposure_safe=np.where(np.isfinite(exposure)&(exposure>0),exposure,1.)
    approximate_counts=values*exposure_safe[None,:,:]/1e6
    counts=np.rint(approximate_counts)
    finite=np.isfinite(approximate_counts)&condition_valid[None,:,:]
    reconstruction_error=float(np.max(np.abs(approximate_counts[finite]-counts[finite])))
    # The known saved normalization should leave counts comfortably within their
    # nearest integer. A failure requires an explicit source-data audit.
    if reconstruction_error>.25:
        raise ValueError(f'Saved CPM cannot safely reconstruct raw counts: max rounding displacement {reconstruction_error}')
    variance=(counts+.5)*(1e6/exposure_safe[None,:,:])**2
    raw_total=np.nansum(np.where(condition_valid[None,:,:],counts,np.nan),axis=-1)
    count_supported=raw_total>=20
    if not np.array_equal(count_supported,stored['raw_total']>=20):
        raise AssertionError('Reconstructed count support differs from the primary run')
    weighted=_fit(values,times,condition_valid,exposure,variance,count_supported,params,False)
    equal=_fit(values,times,condition_valid,exposure,variance,count_supported,params,True)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'atlas_row':indices,'peak':ids}).to_csv(output/'sampled_peaks.csv',index=False)
    refit=[]
    for new,old in [('fitted','fitted_cpm'),('rates','rates'),('rate_se','rate_se')]:
        record=pair_metrics(stored[old],weighted[new])
        record.update(field=new,finite_mask_equal=bool(np.array_equal(np.isfinite(stored[old]),np.isfinite(weighted[new]))))
        valid=np.isfinite(stored[old])&np.isfinite(weighted[new])
        record['within_rtol1e_4_atol2e_5_fraction']=_mean(np.isclose(stored[old][valid],weighted[new][valid],rtol=1e-4,atol=2e-5))
        refit.append(record)
    pd.DataFrame(refit).to_csv(output/'primary_refit_check.csv',index=False)
    refit_discrete={name:_mean(weighted[new]==stored[old]) for name,new,old in
        [('mode_agreement','mode','modes'),('model_agreement','model','models'),('rate_support_agreement','rate_supported','rate_supported')]}
    stage_records=[];rate_records=[];mode_records=[];source_records=[];confusion_records=[]
    panels={'fixed_common_celltypes':common,'all_celltypes':np.ones(len(celltypes),bool)}
    for panel_name,panel in panels.items():
        for t,time in enumerate(times):
            row=pair_metrics(weighted['fitted'][:,panel,t],equal['fitted'][:,panel,t])
            row.update(panel=panel_name,time_hpf=time)
            stage_records.append(row)
        a,b=weighted['rates'][:,panel],equal['rates'][:,panel]
        sa,sb=weighted['rate_supported'][:,panel],equal['rate_supported'][:,panel]
        count_mask=count_supported[:,panel]
        for t in range(len(times)-1):
            for scope,mask in [('all_finite',None),('count_supported',count_mask),('both_supported_changes',sa[:,:,t]&sb[:,:,t])]:
                row=pair_metrics(a[:,:,t],b[:,:,t],mask)
                row.update(panel=panel_name,interval=f'{times[t]:g}-{times[t+1]:g}',scope=scope,
                           coverage_supported_changes=int(sa[:,:,t].sum()),equal_supported_changes=int(sb[:,:,t].sum()),
                           joint_supported_changes=int((sa[:,:,t]&sb[:,:,t]).sum()),
                           supported_change_jaccard=float((sa[:,:,t]&sb[:,:,t]).sum()/(sa[:,:,t]|sb[:,:,t]).sum()) if (sa[:,:,t]|sb[:,:,t]).any() else np.nan,
                           primary_change_retention=_mean(sb[:,:,t][sa[:,:,t]]),
                           joint_supported_sign_agreement=_mean((np.sign(a[:,:,t])==np.sign(b[:,:,t]))[sa[:,:,t]&sb[:,:,t]]))
                rate_records.append(row)
        ma,mb=weighted['mode'][:,panel],equal['mode'][:,panel]
        for scope,mask in [('all_trajectories',np.ones(ma.shape,bool)),('count_supported',count_mask),('either_change_mode',np.isin(ma,[2,3,4,5])|np.isin(mb,[2,3,4,5]))]:
            mode_records.append(dict(panel=panel_name,scope=scope,n=int(mask.sum()),mode_agreement=_mean((ma==mb)[mask]),model_agreement=_mean((weighted['model'][:,panel]==equal['model'][:,panel])[mask])))
        counts_confusion=np.zeros((7,7),int)
        np.add.at(counts_confusion,(ma[count_mask],mb[count_mask]),1)
        for i,j in np.ndindex(counts_confusion.shape):
            confusion_records.append(dict(panel=panel_name,coverage_mode=MODE_LABELS[i],equal_mode=MODE_LABELS[j],n=int(counts_confusion[i,j])))
        # This source-level calculation mirrors the pre-projection gate only;
        # it does not refit the neighbor graph or calculate new UMAP arrows.
        s1,s2=_source(weighted,panel,params),_source(equal,panel,params)
        for t in range(len(times)-1):
            shared=s1['valid'][:,:,t]&s2['valid'][:,:,t]
            va=np.where(shared,a[:,:,t],0.);vb=np.where(shared,b[:,:,t],0.)
            denominator=np.linalg.norm(va,axis=1)*np.linalg.norm(vb,axis=1)
            cosine=np.divide(np.sum(va*vb,axis=1),denominator,out=np.full(len(values),np.nan),where=denominator>1e-12)
            needed=max(int(projection_params['min_shared_features']),int(np.ceil(panel.sum()*projection_params['min_shared_fraction'])))
            both=s1['active'][:,t]&s2['active'][:,t]&np.isfinite(cosine)&(shared.sum(axis=1)>=needed)
            union=s1['active'][:,t]|s2['active'][:,t]
            source_records.append(dict(panel=panel_name,interval=f'{times[t]:g}-{times[t+1]:g}',n_sampled_peaks=len(values),coverage_active=int(s1['active'][:,t].sum()),equal_active=int(s2['active'][:,t].sum()),both_active=int((s1['active'][:,t]&s2['active'][:,t]).sum()),active_jaccard=float((s1['active'][:,t]&s2['active'][:,t]).sum()/union.sum()) if union.any() else np.nan,primary_source_retention=_mean(s2['active'][:,t][s1['active'][:,t]]),velocity_cosine_min_shared_features=needed,n_joint_active_velocity_comparisons=int(both.sum()),median_joint_active_velocity_cosine=float(np.median(cosine[both])) if both.any() else np.nan,joint_active_reversed_fraction=_mean(cosine[both]<0)))
    tables={'stage_state_comparison':stage_records,'interval_rate_comparison':rate_records,
            'trajectory_mode_comparison':mode_records,'mode_confusion':confusion_records,
            'source_activation_comparison':source_records}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    after=_stat(source)
    if before!=after:raise AssertionError('The source HDF5 changed during this read-only sensitivity run')
    summary=dict(source=before,source_unchanged=True,n_atlas_peaks=n,n_sampled_peaks=len(indices),sampling_seed=seed,
                 sampling='Uniform without replacement over the full atlas; indices sorted only for I/O',read_block_size=read_block_size,
                 celltypes=celltypes,common_celltypes=celltypes[common],times_hpf=times,parameters=params,velocity_comparison_feature_thresholds=projection_params,
                 changed='Relative fit/LOOCV weights: observed library coverage versus one per observed stage',
                 unchanged=['CPM values','observed/missing conditions','conditional count-variance estimates','model candidates and fixed knots','complexity penalty','effect/support thresholds','source SNR threshold'],
                 max_count_rounding_displacement=reconstruction_error,primary_refit_discrete_agreement=refit_discrete,
                 primary_refit_numeric=refit,mode_comparison=mode_records,
                 fixed_panel_interval_rates=[r for r in rate_records if r['panel']=='fixed_common_celltypes'],
                 fixed_panel_sources=[r for r in source_records if r['panel']=='fixed_common_celltypes'],
                 source_sha256={name:hashlib.sha256((ROOT/'scripts/chrom_velo'/name).read_bytes()).hexdigest() for name in ['trajectory_modes.py','trajectory_classification.py']},
                 interpretation=['Equal-stage weighting changes the fitting loss, not CPM normalization or sequencing depth.',
                                 'Coverage is a relative exposure weight, not an exact inverse-variance estimate when true means vary.',
                                 'Rate RMSE is disagreement between estimators; real ground truth is unknown.',
                                 'Jointly supported sign agreement is conditional on both fits selecting a change; retention and counts are reported separately.',
                                 'No primary model, graph, embedding, or arrow field is replaced.',
                                 'Conditional count diagnostics are not biological confidence intervals.'])
    (output/'weight_sensitivity_summary.json').write_text(json.dumps(_json(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps(_json(dict(output=str(output),n_peaks=len(indices),primary_refit=refit_discrete,mode_comparison=mode_records,fixed_panel_sources=summary['fixed_panel_sources'])),indent=2,allow_nan=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--sample-peaks',type=int,default=8192)
    parser.add_argument('--seed',type=int,default=20261020)
    parser.add_argument('--read-block-size',type=int,default=512)
    args=parser.parse_args()
    run(args.source,args.output_dir,args.sample_peaks,args.seed,args.read_block_size)


if __name__=='__main__':main()
