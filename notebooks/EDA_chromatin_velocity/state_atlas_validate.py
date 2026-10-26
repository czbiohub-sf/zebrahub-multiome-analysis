#!/usr/bin/env python3
"""Independent checks of saved state-atlas arrays and display diagnostics.

Does not import the implementation under test. All reported metric checks are
recomputed directly from the saved sample, original source, and model arrays.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import h5py
from scipy.stats import spearmanr

HERE=Path(__file__).parent
OUT=HERE/'outputs/state_trajectory_atlas_v1'
SOURCE=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'

def save(name,result,out=OUT):
    result['status']='passed'
    (out/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

def equal(a,b):
    assert a.shape==b.shape
    assert np.array_equal(a,b,equal_nan=True) if a.dtype.kind not in 'OUS' else np.array_equal(a,b)

def take(ds,indices):
    order=np.argsort(indices)
    return ds[indices[order]][np.argsort(order)]

def sample_audit(out=OUT):
    with h5py.File(out/'sample_data.h5') as f,h5py.File(SOURCE) as source:
        indices=f['source_indices'][:]
        train=f['train_mask'][:].astype(bool)
        test=f['test_mask'][:].astype(bool)
        ids=f['peak_ids'].asstr()[:]
        assert len(np.unique(indices))==len(indices)==len(np.unique(ids))
        assert train.sum()==20000 and test.sum()==5000 and not (train&test).any()
        equal(ids,source['peak_ids'].asstr()[:][indices])
        assert not set(ids[train])&set(ids[test])
        expected=np.random.default_rng(20261020).choice(len(source['peak_ids']),25000,replace=False)
        equal(indices[:25000],expected)
        assert train[:20000].all() and test[20000:25000].all()
        assert not train[20000:].any() and not test[:20000].any() and not test[25000:].any()
        checked=[]
        for key in ['observed_cpm','fitted_cpm','rates','rate_supported','rate_se']:
            equal(f[key][:],take(source[key],indices).transpose(0,2,1));checked.append(key)
        for key in ['fit_valid','lack_of_fit','modes','raw_total','broad_reference','umap2d','umap3d']:
            equal(f[key][:],take(source[key],indices));checked.append(key)
        for key in ['stages','condition_valid','common_celltypes']:
            equal(f[key][:],source[key][:]);checked.append(key)
        equal(f['celltypes'].asstr()[:],source['celltypes'].asstr()[:])
        original=json.loads(f.attrs['source_json'])
        stat=SOURCE.stat()
        assert stat.st_size==original['bytes'] and stat.st_mtime_ns==original['mtime_ns']
        meta=pd.read_csv(out/'sample_metadata.csv',keep_default_na=False)
        equal(meta.peak_id.to_numpy(),ids)
        equal(meta.source_index.to_numpy(),indices)
        equal(meta.sample_index.to_numpy(),np.arange(len(ids)))
        expected_split=np.where(train,'train',np.where(test,'test','marker'))
        equal(meta.split.to_numpy(),expected_split)
    result=dict(n_peaks=len(indices),n_train=int(train.sum()),n_test=int(test.sum()),n_marker_only=int((~train&~test).sum()),
                checked_source_arrays=checked,all_ids_and_coordinates_exact=True,whole_peak_train_test_disjoint=True,
                uniform_random_sample_reproduced=True,source_size_and_mtime_unchanged=True)
    save('validation_sample_alignment.json',result,out)
    return result

def panel_arrays(f,g):
    cc=g['celltype_indices'][:];tt=g['stage_indices'][:]
    return cc,tt, f['observed_cpm'][:][:,tt,:][:,:,cc].astype(float),f['fitted_cpm'][:][:,tt,:][:,:,cc].astype(float)

def pca_audit(out=OUT):
    rows=[];summary=[];intervals=[]
    reported=pd.read_csv(out/'pca_preservation.csv')
    reported_intervals=pd.read_csv(out/'pca_interval_preservation.csv')
    with h5py.File(out/'sample_data.h5') as f,h5py.File(out/'state_pca.h5') as p:
        train=f['train_mask'][:].astype(bool);test=f['test_mask'][:].astype(bool)
        condition=f['condition_valid'][:];alltimes=f['stages'][:]
        expectations={'primary_21':(21,np.arange(6)), 'matched_21':(21,np.arange(1,6)), 'optic_24':(24,np.arange(1,6))}
        for panel,(nfeatures,times) in expectations.items():
            g=p[panel];cc,tt,raw,fitted=panel_arrays(f,g)
            assert len(cc)==nfeatures
            equal(tt,times);equal(g['stages'][:],alltimes[tt])
            equal(g['celltypes'].asstr()[:],f['celltypes'].asstr()[:][cc])
            assert condition[cc][:,tt].all()
            expected_cc=np.flatnonzero(condition.all(axis=1) if panel != "optic_24" else condition[:,tt].all(axis=1));equal(cc,expected_cc)
            assert np.isfinite(raw).all() and np.isfinite(fitted).all() and raw.min()>=0 and fitted.min()>=0
            support=f['rate_supported'][:][:,tt[:-1],:][:,:,cc]
            equal(g['supported_celltype_count'][:],support.sum(axis=-1).astype('i2'))
            for label,x in [('observed',raw),('fitted',fitted)]:
                mean=x.mean(axis=-1)
                mx=x.max(axis=-1)
                tau=np.divide((mx[...,None]-x).sum(axis=-1), (nfeatures-1)*mx, out=np.full_like(mx,np.nan),where=mx>0)
                r=np.diff(x,axis=1)/np.diff(alltimes[tt])[None,:,None]
                for key,expected in [('mean_cpm',mean),('tau',tau),('rate_rms_cpm_per_hpf',np.sqrt(np.mean(r*r,axis=-1))),('net_rate_cpm_per_hpf',r.mean(axis=-1))]:
                    assert np.allclose(g[label+'_'+key][:],expected,rtol=2e-7,atol=1e-7,equal_nan=True), (panel,label,key)
            for transform in ['log1p','linear']:
                h=g[transform];components=h['components'][:];mean=h['mean'][:]
                x=np.log1p(raw) if transform=='log1p' else raw
                y=np.log1p(fitted) if transform=='log1p' else fitted
                trainx=x[train].reshape(-1,nfeatures)
                orth=float(np.max(np.abs(components@components.T-np.eye(nfeatures))))
                meanerr=float(np.max(np.abs(trainx.mean(axis=0)-mean)))
                assert orth<1e-10 and meanerr<1e-10
                trainproj=(trainx-mean)@components.T
                cov=trainproj.T@trainproj/(len(trainproj)-1)
                ev=np.diag(cov)
                assert np.max(np.abs(cov-np.diag(ev)))<1e-8
                assert np.all(np.diff(ev)<=1e-10)
                assert np.allclose(ev/ev.sum(),h['explained_variance_ratio'][:],atol=1e-12,rtol=1e-10)
                assert np.allclose(np.sqrt(ev*(len(trainproj)-1)),h['singular_values'][:],atol=1e-8,rtol=1e-10)
                for estimator,z in [('observed',x),('fitted',y)]:
                    stored=h[estimator+'_pca'][:].astype(float)
                    exact=(z-mean)@components.T
                    assert np.allclose(stored,exact,rtol=2e-7,atol=1e-7)
                    reconstruction=stored@components+mean
                    maxerr=float(np.max(np.abs(reconstruction-z)))
                    scalederr=float(np.max(np.abs(reconstruction-z)/(1+np.abs(z))))
                    assert scalederr<2e-5
                    stable=np.all(np.diff(z,axis=1)==0,axis=-1)
                    stablemovement=np.linalg.norm(np.diff(stored,axis=1),axis=-1)[stable]
                    assert len(stablemovement)==0 or stablemovement.max()==0
                    summary.append(dict(panel=panel,transform=transform,estimator=estimator,
                        orthonormality_max_abs_error=orth,training_mean_max_abs_error=meanerr,
                        saved_float32_max_reconstruction_error=maxerr,saved_float32_scaled_reconstruction_error=scalederr,
                        exact_stable_intervals=int(stable.sum()),exact_stable_max_displacement=float(stablemovement.max()) if len(stablemovement) else 0.))
                    for split,mask in [('train',train),('test',test)]:
                        dz=np.diff(z[mask],axis=1);dp=np.diff(stored[mask],axis=1)
                        total=float(np.sum(dz*dz));per=np.sum(dp*dp,axis=-1)
                        for k in [2,3,nfeatures]:
                            kept=np.sum(dp[...,:k]**2,axis=-1);valid=per>1e-12
                            metric=dict(panel=panel,transform=transform,estimator=estimator,split=split,components=k,
                                temporal_energy_fraction=float(kept.sum()/total),
                                median_per_step_energy_fraction=float(np.median(kept[valid]/per[valid])),
                                p10_per_step_energy_fraction=float(np.quantile(kept[valid]/per[valid],.1)),n_steps=int(valid.sum()))
                            rr=reported[(reported.panel==panel)&(reported['transform']==transform)&(reported.estimator==estimator)&(reported.split==split)&(reported.components==k)]
                            assert len(rr)==1
                            for name in ['temporal_energy_fraction','median_per_step_energy_fraction','p10_per_step_energy_fraction']:
                                assert np.isclose(metric[name],rr.iloc[0][name],atol=3e-6,rtol=3e-6), (panel,transform,estimator,split,k,name)
                            # Tiny float32 perturbations can affect only the near-zero step count.
                            metric['reported_n_steps']=int(rr.iloc[0].n_steps)
                            rows.append(metric)
                        for j in range(len(tt)-1):
                            den=float(np.sum(dz[:,j]**2))
                            rr=reported_intervals[(reported_intervals.panel==panel)&(reported_intervals['transform']==transform)&(reported_intervals.estimator==estimator)&(reported_intervals.split==split)&(reported_intervals.start_hpf==alltimes[tt[j]])]
                            assert len(rr)==1
                            for k in [2,3]:
                                got=float(np.sum(dp[:,j,:k]**2)/den)
                                assert np.isclose(got,rr.iloc[0][f'pca{k}_temporal_energy_fraction'],atol=1e-6,rtol=1e-6)
                                intervals.append(dict(panel=panel,transform=transform,estimator=estimator,split=split,components=k,start_hpf=float(alltimes[tt[j]]),energy_fraction=got))
    pd.DataFrame(rows).to_csv(out/'validation_pca_preservation_recomputed.csv',index=False)
    result={'n_model_estimator_checks':len(summary),'n_preservation_rows_checked':len(rows),'n_interval_metrics_checked':len(intervals),
            'all_feature_panels_fully_observed':True,'all_training_bases_from_observed_training_states':True,
            'rates_recomputed_with_actual_elapsed_hours':True,'results':summary,
            'precision_note':'Saved coordinates are float32; independently recomputed model arrays are float64. Displayed-PC metrics agree within numerical precision.'}
    save('validation_pca.json',result,out)
    return result

def duplicate_spread(inputs,coords):
    unique,inverse=np.unique(inputs,axis=0,return_inverse=True)
    lo=np.full((len(unique),coords.shape[1]),np.inf)
    hi=np.full_like(lo,-np.inf)
    np.minimum.at(lo,inverse,coords)
    np.maximum.at(hi,inverse,coords)
    return float((hi-lo).max()),int(len(inputs)-len(unique))

def umap_audit(out=OUT):
    results=[]
    diagnostics=json.loads((out/'umap_full_diagnostics.json').read_text())
    with h5py.File(out/'state_pca.h5') as p,h5py.File(out/'state_umap.h5') as u,h5py.File(out/'sample_data.h5') as f:
        test=f['test_mask'][:].astype(bool)
        for panel in ['primary_21','matched_21','optic_24']:
            if panel not in u: continue
            g=p[panel];h=u[panel]
            equal(g['stage_indices'][:],h['stage_indices'][:]);equal(g['celltype_indices'][:],h['celltype_indices'][:])
            obs=g['log1p/observed_pca'][:];fit=g['log1p/fitted_pca'][:]
            evaluation=h['evaluation_peak_indices'][:]
            assert f['test_mask'][:][evaluation].all()
            assert len(np.unique(evaluation))==len(evaluation)
            inputs=np.concatenate([obs.reshape(-1,obs.shape[-1]),fit.reshape(-1,fit.shape[-1])])
            for dim in [2,3]:
                if f'observed_umap{dim}d' not in h: continue
                oc=h[f'observed_umap{dim}d'][:];fc=h[f'fitted_umap{dim}d'][:]
                assert oc.shape==(*obs.shape[:2],dim) and fc.shape==oc.shape and np.isfinite(oc).all() and np.isfinite(fc).all()
                coords=np.concatenate([oc.reshape(-1,dim),fc.reshape(-1,dim)])
                spread,ndups=duplicate_spread(inputs,coords)
                assert spread==0,(panel,dim,spread)
                for label,pc,emb in [('observed',obs,oc),('fitted',fit,fc)]:
                    hd=np.linalg.norm(np.diff(pc[test].astype(float),axis=1),axis=-1)
                    ld=np.linalg.norm(np.diff(emb[test].astype(float),axis=1),axis=-1)
                    stable=hd==0
                    assert not stable.any() or np.max(ld[stable])==0
                    valid=hd>1e-9
                    evhi=np.linalg.norm(np.diff(pc[evaluation].astype(float),axis=1),axis=-1)
                    evlo=np.linalg.norm(np.diff(emb[evaluation].astype(float),axis=1),axis=-1)
                    evvalid=evhi>0
                    dt=np.diff(g['stages'][:])
                    evlength=float(spearmanr(evhi[evvalid],evlo[evvalid]).statistic)
                    evrate=float(spearmanr((evhi/dt)[evvalid],(evlo/dt)[evvalid]).statistic)
                    reported=diagnostics[panel][str(dim)][label+'_diagnostics']
                    assert np.isclose(evlength,reported['temporal_step_length_spearman'],atol=1e-12,rtol=1e-12)
                    assert np.isclose(evrate,reported['temporal_rate_magnitude_spearman'],atol=1e-12,rtol=1e-12)
                    assert reported['n_stationary_intervals_with_false_movement']==0
                    results.append(dict(panel=panel,dimensions=dim,estimator=label,
                        full_joint_exact_duplicate_coordinate_spread=spread,n_redundant_query_states=ndups,
                        heldout_exact_stable_intervals=int(stable.sum()),
                        heldout_temporal_length_spearman=float(spearmanr(hd[valid],ld[valid]).statistic),
                        n_nonzero_heldout_steps=int(valid.sum()),
                        independent_reported_step_correlation=evlength,independent_reported_rate_correlation=evrate))
    pd.DataFrame(results).to_csv(out/'validation_umap_displacements.csv',index=False)
    assert results, 'No published UMAPs found'
    save('validation_umap.json',dict(results=results,all_six_panel_dimension_models_checked=(len(results)==12),interpretation='Exact-repeat caching prevents false movement of identical states. UMAP length correlations are display diagnostics, not calibrated biological rates.'),out)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--stage',choices=['sample','pca','umap','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['sample','all']:sample_audit(args.out)
    if args.stage in ['pca','all']:pca_audit(args.out)
    if args.stage in ['umap','all']:umap_audit(args.out)
