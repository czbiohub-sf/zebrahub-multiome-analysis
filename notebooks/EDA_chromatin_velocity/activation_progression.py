"""Measure celltype-specific accessibility redistribution on the fixed peak atlas.

The primary statistic is the measured accessibility-weighted graph-distance
center on a fixed peak panel. Condition-wide scaling cancels. It is a spatial
allocation statistic, not absolute openness, transport, or developmental time.
Shared stage-label permutations keep the atlas and all cross-peak relationships
fixed. Their tail fractions are descriptive ordering references, not biological
replicate p-values. Fitted opening-window timing is a secondary descriptive view.
"""
from pathlib import Path
import argparse,sys,json,time
import h5py,numpy as np,pandas as pd
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/chrom_velo'))
from activation_statistics import temporal_permutation_test,activation_weight_sensitivity
from real_specialization_benchmark import write_json
from real_trajectory_velocity import PARAMS
from trajectory_weight_sensitivity import _take,_fit
SOURCE=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'
OUT=HERE/'outputs/activation_progression_v1'
TIMING=dict(min_positive_change_cpm=.5,min_relative_rate_modulation=.25)

def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def signature(p):
 s=p.stat();return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def timing_metrics(rates,support,times,quality):
 """Center of fitted opening/closing activity in supported, nonuniform trajectories; not onset.

 Uniform-rate lines have no resolved event timing. Positive increments weight
 interval midpoints, so unequal gaps do not overcount shorter intervals.
 """
 rates=np.asarray(rates,float);support=np.asarray(support,bool);dt=np.diff(times);mid=(times[:-1]+times[1:])/2
 result={}
 for label,sign in [('opening',1),('closing',-1)]:
  mag=np.maximum(sign*rates,0.);mass=mag*dt
  total=np.sum(mass,axis=-1);maximum=np.max(mag,axis=-1);minimum=np.min(mag,axis=-1)
  modulation=np.divide(maximum-minimum,maximum,out=np.zeros_like(total),where=maximum>1e-10)
  supported=(support&(sign*rates>0)).any(axis=-1)
  resolved=quality&np.isfinite(rates).all(axis=-1)&supported&(total>=TIMING['min_positive_change_cpm'])&(modulation>=TIMING['min_relative_rate_modulation'])
  center=np.divide(np.sum(mass*mid,axis=-1),total,out=np.full_like(total,np.nan),where=total>1e-10)
  center[~resolved]=np.nan
  result[label+'_center_hpf']=center;result[label+'_resolved']=resolved;result[label+'_change_cpm']=total
 return result

def distances(out):
 with h5py.File(out/'geometry.h5','r') as g:
  data={k:g[k][:] for k in ['distance_hops','distance_weighted','region_code','broad_reference']}
  data.update(peak_ids=g['peak_ids'].asstr()[:],region_labels=g['region_labels'].asstr()[:])
 result={'hops':data['distance_hops'],'weighted':data['distance_weighted']}
 core=out/'geometry_core_reference.h5'
 if core.exists():
  with h5py.File(core,'r') as g:
   assert np.array_equal(data['peak_ids'],g['peak_ids'].asstr()[:])
   result['core_hops']=g['distance_core_hops'][:];result['core_weighted']=g['distance_core_weighted'][:]
 return data,result

def aggregate(out):
 before=signature(SOURCE);geo,ds=distances(out);region=geo['region_code'];R=len(geo['region_labels']);n=len(region)
 times_rows=[];interval_rows=[];distance_rows=[];timing_rows=[];mode_rows=[];bin_rows=[]
 with h5py.File(SOURCE,'r') as f,h5py.File(out/'activation_metrics.h5','w') as target:
  ids=f['peak_ids'].asstr()[:];assert np.array_equal(ids,geo['peak_ids'])
  ct=f['celltypes'].asstr()[:];times=f['stages'][:];common=f['common_celltypes'][:].astype(bool);C=len(ct);T=len(times)
  for name,arr in [('peak_ids',ids),('celltypes',ct)]:target.create_dataset(name,data=arr.astype(object),dtype=h5py.string_dtype())
  target.create_dataset('stages',data=times);target.create_dataset('common_celltypes',data=common)
  for key in ['opening_center_hpf','closing_center_hpf']:target.create_dataset(key,shape=(n,C),dtype='f4',chunks=(4096,1),compression='lzf',fillvalue=np.nan)
  for key in ['opening_resolved','closing_resolved']:target.create_dataset(key,shape=(n,C),dtype='?',chunks=(4096,1),compression='lzf')
  measured=np.full((C,R+1,len(ds),T),np.nan);fitted=np.full_like(measured,np.nan)
  masks={metric:np.isfinite(d) for metric,d in ds.items()}
  condition=f['condition_valid'][:].astype(bool)
  for c,name in enumerate(ct):
   cv=condition[c];obs=f['observed_cpm'][:,c,:].astype(float);fit=f['fitted_cpm'][:,c,:].astype(float)
   rates=f['rates'][:,c,:].astype(float);supported=f['rate_supported'][:,c,:];quality=f['fit_valid'][:,c]&(f['lack_of_fit'][:,c]<=PARAMS['lack_of_fit_threshold'])
   stable_raw=np.isfinite(obs[:,cv]).all(axis=1) if cv.any() else np.zeros(n,bool)
   stable_fit=np.isfinite(fit[:,cv]).all(axis=1) if cv.any() else np.zeros(n,bool)
   rawmask=stable_raw&masks['hops'];fitmask=rawmask&stable_fit
   sizes=np.bincount(region[rawmask],minlength=R);fit_sizes=np.bincount(region[fitmask],minlength=R)
   all_modes=f['modes'][:,c]
   for r in range(R):
    hist=np.bincount(all_modes[region==r],minlength=7)
    mode_rows.extend(dict(celltype=name,region_code=r,region_label=geo['region_labels'][r],mode_code=k,peaks=int(v)) for k,v in enumerate(hist))
   for t,time_hpf in enumerate(times):
    if not cv[t]:
     times_rows.extend(dict(celltype=name,region_code=r,region_label=geo['region_labels'][r],stage=time_hpf,condition_valid=False,common_panel=bool(common[c]),n_peaks=int(sizes[r]),observed_mean_cpm=np.nan,fitted_mean_cpm=np.nan,observed_distance_center=np.nan,fitted_distance_center=np.nan,accessibility_mass_fraction=np.nan) for r in range(R))
     continue
    a=np.where(rawmask,obs[:,t],0.);b=np.where(fitmask,fit[:,t],0.)
    sums=np.bincount(region,weights=a,minlength=R);fsums=np.bincount(region,weights=b,minlength=R)
    centers={}
    for m,(metric,d) in enumerate(ds.items()):
     valid=rawmask&masks[metric];fvalid=fitmask&masks[metric]
     av=np.where(valid,obs[:,t],0.);bv=np.where(fvalid,fit[:,t],0.);dd=np.where(masks[metric],d,0.)
     den=np.bincount(region,weights=av,minlength=R);num=np.bincount(region,weights=av*dd,minlength=R)
     fden=np.bincount(region,weights=bv,minlength=R);fnum=np.bincount(region,weights=bv*dd,minlength=R)
     measured[c,:R,m,t]=np.divide(num,den,out=np.full(R,np.nan),where=den>0)
     fitted[c,:R,m,t]=np.divide(fnum,fden,out=np.full(R,np.nan),where=fden>0)
     measured[c,R,m,t]=num.sum()/den.sum() if den.sum()>0 else np.nan
     fitted[c,R,m,t]=fnum.sum()/fden.sum() if fden.sum()>0 else np.nan
     centers[metric]=(measured[c,:R,m,t],fitted[c,:R,m,t])
     distance_rows.append(dict(celltype=name,stage=time_hpf,metric=metric,primary_panel=bool(common[c]),n_peaks=int(valid.sum()),n_fitted_peaks=int(fvalid.sum()),observed_distance_center=measured[c,R,m,t],fitted_distance_center=fitted[c,R,m,t],observed_total_cpm=den.sum(),fitted_total_cpm=fden.sum()))
    for r in range(R):
     times_rows.append(dict(celltype=name,region_code=r,region_label=geo['region_labels'][r],stage=time_hpf,condition_valid=True,common_panel=bool(common[c]),n_peaks=int(sizes[r]),observed_mean_cpm=sums[r]/sizes[r] if sizes[r] else np.nan,fitted_mean_cpm=fsums[r]/fit_sizes[r] if fit_sizes[r] else np.nan,observed_distance_center=centers['hops'][0][r],fitted_distance_center=centers['hops'][1][r],accessibility_mass_fraction=sums[r]/sums.sum() if sums.sum()>0 else np.nan))
    hop=np.where(np.isfinite(geo['distance_hops']),geo['distance_hops'],-1).astype(int)
    for h in np.unique(hop[rawmask]):
     use=rawmask&(hop==h);bin_rows.append(dict(celltype=name,stage=time_hpf,distance_hops=int(h),peaks=int(use.sum()),observed_mean_cpm=float(np.mean(obs[use,t])),fitted_mean_cpm=float(np.nanmean(fit[use,t])),accessibility_mass_fraction=float(np.sum(obs[use,t])/sums.sum())))
   for t in range(T-1):
    valid=bool(cv[t]&cv[t+1]);rr=rates[:,t];finite=np.isfinite(rr)&quality
    op=supported[:,t]&(rr>0);cl=supported[:,t]&(rr<0)
    interval_rows.append(dict(celltype=name,start_hpf=times[t],end_hpf=times[t+1],valid_interval=valid,primary_panel=bool(common[c]),n_peaks=int(finite.sum()) if valid else 0,supported_opening=int(op.sum()) if valid else 0,supported_closing=int(cl.sum()) if valid else 0,mean_rate=float(np.mean(rr[finite])) if valid and finite.any() else np.nan,mean_positive_rate=float(np.mean(np.maximum(rr[finite],0))) if valid and finite.any() else np.nan,mean_negative_rate=float(np.mean(np.minimum(rr[finite],0))) if valid and finite.any() else np.nan))
   tm=timing_metrics(rates,supported,times,quality&common[c])
   for key in ['opening_center_hpf','closing_center_hpf','opening_resolved','closing_resolved']:target[key][:,c]=tm[key]
   for phase in ['opening','closing']:
    age=tm[phase+'_center_hpf']
    for r in list(range(R))+[-1]:
     use=np.isfinite(age)&np.isfinite(ds['hops']);use &= (region==r) if r>=0 else True
     meaningful=use.sum()>=30 and len(np.unique(ds['hops'][use]))>=3 and np.ptp(age[use])>1e-8
     corr=float(spearmanr(ds['hops'][use],age[use]).statistic) if meaningful else np.nan
     timing_rows.append(dict(celltype=name,phase=phase,region_code=r,region_label=geo['region_labels'][r] if r>=0 else 'all_regions',n_resolved=int(use.sum()),median_window_center_hpf=float(np.median(age[use])) if use.any() else np.nan,distance_timing_spearman=corr,primary_panel=bool(common[c]),inference='descriptive; spatially dependent peaks; no peak-independent p-value'))
   log(f'Aggregated {name} ({c+1}/{C})')
  target.create_dataset('observed_distance_centers',data=measured);target.create_dataset('fitted_distance_centers',data=fitted)
  target.create_dataset('distance_metrics',data=np.asarray(list(ds),object),dtype=h5py.string_dtype())
  target.create_dataset('region_labels',data=geo['region_labels'].astype(object),dtype=h5py.string_dtype())
  target.attrs['timing_parameters_json']=json.dumps(TIMING)
  target.attrs['timing_definition']='Center of positive/negative fitted increments over interval midpoints; only nonuniform supported profiles; not onset; only complete CTs compared'
  target.attrs['primary_statistic']='Observed CPM-weighted graph-distance center on fixed connected peak panel; common condition scaling cancels'
  target.attrs['distance_center_shape']='celltype x (regions plus final all-regions row) x metric x stage'
 for key,rows in [('region_time_summary',times_rows),('celltype_interval_summary',interval_rows),('celltype_distance_time',distance_rows),('opening_closing_timing',timing_rows),('region_mode_summary',mode_rows),('distance_bin_time_summary',bin_rows)]:pd.DataFrame(rows).to_csv(out/f'{key}.csv',index=False)
 assert before==signature(SOURCE)
 write_json(out/'aggregation_provenance.json',dict(source=before,source_unchanged=True,n_peaks=n,n_celltypes=C,n_regions=R,primary_celltypes=ct[common],metrics=list(ds),timing_parameters=TIMING,missingness='Fixed observed conditions per CT; primary temporal tests require six adequate stages; no peak panel changes across stages',normalization='CPM retained for levels; common multiplicative column scaling cancels in distance centers'))

def descriptive_time_statistics(values,times):
 x=np.asarray(times,float)-np.mean(times);a=np.asarray(values,float);center=a-a.mean(axis=-1,keepdims=True)
 cov=np.sum(center*x,axis=-1);slope=cov/np.sum(x*x)
 den=np.sqrt(np.sum(center*center,axis=-1)*np.sum(x*x));corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-12)
 return dict(slope=slope,correlation=corr,**{k:np.full(slope.shape,np.nan) for k in ['p_outward','p_inward','p_two_sided','p_fwer_two_sided']})

def temporal_tests(out):
 rows=[];global_results={}
 with h5py.File(out/'activation_metrics.h5','r') as f:
  times=f['stages'][:];ct=f['celltypes'].asstr()[:];common=f['common_celltypes'][:].astype(bool);regions=f['region_labels'].asstr()[:];metrics=f['distance_metrics'].asstr()[:]
  observed=f['observed_distance_centers'][:];fitted=f['fitted_distance_centers'][:]
 for kind,values in [('observed',observed),('fitted',fitted)]:
  for m,metric in enumerate(metrics):
   selected=values[common,-1,m,:]
   if not np.isfinite(selected).all():raise ValueError('Primary CT panel must have all six finite distance centers')
   if kind=='observed':
    test=temporal_permutation_test(selected,times,min_observations=6)
    global_results[f'{kind}_{metric}']=test['global_result']
   else:
    test=descriptive_time_statistics(selected,times)
    global_results[f'{kind}_{metric}']=dict(mean_correlation=float(np.mean(test['correlation'])),mean_slope=float(np.mean(test['slope'])),fraction_positive=float(np.mean(test['slope']>1e-12)),n_series=len(selected),interpretation='Descriptive sensitivity only; no postfit stage permutation because fits already use temporal ordering')
   for j,c in enumerate(np.flatnonzero(common)):
    row=dict(scope='celltype',celltype=ct[c],region_code=-1,region_label='all_regions',metric=metric,estimator=kind,primary_panel=True)
    for key in ['slope','correlation','p_outward','p_inward','p_two_sided','p_fwer_two_sided']:row[key]=float(test[key][j])
    row['endpoint_shift']=float(selected[j,-1]-selected[j,0]);rows.append(row)
   glob=temporal_permutation_test(np.mean(selected,axis=0)[None,:],times,min_observations=6) if kind=='observed' else descriptive_time_statistics(np.mean(selected,axis=0)[None,:],times)
   row=dict(scope='global',celltype='equal_weight_mean_of_21_CTs',region_code=-1,region_label='all_regions',metric=metric,estimator=kind,primary_panel=True)
   for key in ['slope','correlation','p_outward','p_inward','p_two_sided','p_fwer_two_sided']:row[key]=float(glob[key][0])
   row['endpoint_shift']=float(np.mean(selected[:,-1]-selected[:,0]));rows.append(row)
   if kind=='observed' and metric=='hops':
    # One shared permutation for all region/CT series; no independent peak null.
    rvalues=values[common,:-1,m,:];rt=temporal_permutation_test(rvalues,times,min_observations=6)
    for j,c in enumerate(np.flatnonzero(common)):
     for r,label in enumerate(regions):
      row=dict(scope='region_celltype',celltype=ct[c],region_code=r,region_label=label,metric=metric,estimator=kind,primary_panel=True)
      for key in ['slope','correlation','p_outward','p_inward','p_two_sided','p_fwer_two_sided']:row[key]=float(rt[key][j,r])
      row['endpoint_shift']=float(rvalues[j,r,-1]-rvalues[j,r,0]);rows.append(row)
    np.savez_compressed(out/'temporal_order_nulls.npz',permutations=test['permutation_indices'],null_ct_slopes=test['null_slope'],null_ct_correlations=test['null_correlation'],null_global_mean_slope=test['null_global_mean_slope'],null_global_fraction_positive=test['null_global_fraction_positive'],null_region_correlations=rt['null_correlation'])
 pd.DataFrame(rows).to_csv(out/'outward_tests.csv',index=False)
 write_json(out/'temporal_order_summary.json',dict(reference='All 720 shared stage-label permutations; atlas and cross-peak/celltype structure fixed. Conditional ordering reference, not replicate-based biological p-values.',times=times,global_results=global_results,primary='observed hops; complete 21-celltype panel; actual-hour slope',sensitivity='edge-cost distance, dense-core reference, and fitted versus observed states',limitations=['Atlas and broad references were built from these data; this is conditional analysis, not independent validation.','Six stage samples are not exchangeable biological replicates; stage-specific depth/noise and technical effects remain.','Uniform global opening leaves graph-distance centers unchanged.','Leiden regions are atlas neighborhoods, not validated developmental branches.']))
 log('Temporal order controls completed')

def weight_sensitivity(out):
 before=signature(SOURCE);geo,ds=distances(out)
 with h5py.File(SOURCE,'r') as f:
  ids=np.sort(np.random.default_rng(20261020).choice(len(f['peak_ids']),8192,replace=False));y=_take(f['observed_cpm'],ids,512).astype(float)
  cv=f['condition_valid'][:].astype(bool);common=f['common_celltypes'][:].astype(bool);times=f['stages'][:];ct=f['celltypes'].asstr()[:];ex=f['coverage'][:].astype(float)
 safe=np.where(np.isfinite(ex)&(ex>0),ex,1.);counts=np.rint(y*safe/1e6);var=(counts+.5)*(1e6/safe)**2;countok=np.nansum(np.where(cv,counts,np.nan),axis=-1)>=20
 fits=[_fit(y,times,cv,ex,var,countok,PARAMS,equal) for equal in [False,True]]
 rows=[];timing_rows=[];metrics={}
 for label,result in zip(['coverage','equal'],fits):
  quality=result['fit_valid']&countok&(result['lack_of_fit']<=PARAMS['lack_of_fit_threshold'])&common[None,:]
  metrics[label]=timing_metrics(result['rates'],result['rate_supported'],times,quality)
 for c in np.flatnonzero(common):
  for metric,dall in ds.items():
   d=dall[ids];centers=[]
   for label,result in zip(['coverage','equal'],fits):
    a=result['fitted'][:,c,:];valid=np.isfinite(a).all(axis=1)&np.isfinite(d);centers.append(np.sum(a[valid]*d[valid,None],axis=0)/a[valid].sum(axis=0))
   slopes=[float(temporal_permutation_test(v[None,:],times)['slope'][0]) for v in centers]
   rows.append(dict(celltype=ct[c],metric=metric,n_sampled_peaks=len(ids),coverage_slope=slopes[0],equal_slope=slopes[1],same_slope_sign=bool(np.sign(slopes[0])==np.sign(slopes[1])),coverage_endpoint_shift=float(centers[0][-1]-centers[0][0]),equal_endpoint_shift=float(centers[1][-1]-centers[1][0]),max_distance_center_difference=float(np.max(np.abs(centers[0]-centers[1])))))
  for phase in ['opening','closing']:
   a=metrics['coverage'][phase+'_center_hpf'][:,c];b=metrics['equal'][phase+'_center_hpf'][:,c];both=np.isfinite(a)&np.isfinite(b)
   timing_rows.append(dict(celltype=ct[c],phase=phase,coverage_resolved=int(np.isfinite(a).sum()),equal_resolved=int(np.isfinite(b).sum()),both_resolved=int(both.sum()),median_absolute_timing_shift=float(np.median(np.abs(a[both]-b[both]))) if both.any() else np.nan,timing_spearman=float(spearmanr(a[both],b[both]).statistic) if both.sum()>2 and np.ptp(a[both])>1e-8 and np.ptp(b[both])>1e-8 else np.nan))
 pd.DataFrame(rows).to_csv(out/'fit_weight_progression_sensitivity.csv',index=False);pd.DataFrame(timing_rows).to_csv(out/'fit_weight_timing_sensitivity.csv',index=False)
 archive=dict(atlas_rows=ids,observed_cpm=y,coverage_fitted=fits[0]['fitted'],equal_fitted=fits[1]['fitted'])
 for label,result in zip(['coverage','equal'],fits):
  for key in ['rates','rate_supported','lack_of_fit','fit_valid']:archive[label+'_'+key]=result[key]
  for key,value in metrics[label].items():archive[label+'_'+key]=value
 np.savez_compressed(out/'fit_weight_sample.npz',**archive)
 assert before==signature(SOURCE)
 write_json(out/'fit_weight_sensitivity_summary.json',dict(n_sampled_peaks=len(ids),seed=20261020,source_unchanged=True,parameters=PARAMS,timing_parameters=TIMING,primary_statistic='Measured centers have no fitting weights; this sensitivity compares fitted center summaries and resolved timing',hop_slope_sign_agreement=float(pd.DataFrame(rows).query('metric=="hops"').same_slope_sign.mean()),normalization='identical CPM/count variances in both fits'))
 log('Fit-weight progression sensitivity completed')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--stage',choices=['aggregate','test','weights','all'],default='all');a=p.parse_args();a.output_dir.mkdir(exist_ok=True,parents=True)
 if a.stage in ['aggregate','all']:aggregate(a.output_dir)
 if a.stage in ['test','all']:temporal_tests(a.output_dir)
 if a.stage in ['weights','all']:weight_sensitivity(a.output_dir)
if __name__=='__main__':main()
