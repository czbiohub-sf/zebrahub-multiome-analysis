#!/usr/bin/env python3
"""Independent numerical checks and prespecified forecast power controls.

Synthetic controls test the same low-dimensional estimator, not the biological
adequacy of the fixed Cicero graph. They do not tune real-data settings.
"""
from pathlib import Path
import argparse,json,sys
import h5py,numpy as np,pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_velocity_experiments_v1'
CIS=HERE/'outputs/ct_cis_forecast_v1'
FLOW=HERE/'outputs/ct_velocity_flow_sweep_v1'
sys.path.insert(0,str(HERE))
import ct_cis_forecast as forecast

def savejson(name,data):
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def controls():
 """Known neighbor-leading process and matched absence-of-signal process."""
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=[];numerical=[]
 empty_edges=pd.DataFrame({k:pd.Series(dtype='float64' if k=='coaccess' else 'int64') for k in ['coaccess','target','neighbor','null_neighbor']})
 noedge,control,connected,mass=forecast.graphs(empty_edges,np.arange(4),np.arange(2),.25)
 assert noedge.nnz==0 and control.nnz==0 and not connected.any() and np.all(mass==0)
 times=np.array([10.,12.,14.,16.,19.,24.]);dt=np.diff(times)
 n=1200;nc=6;chrom=1+np.arange(n)%25;fold=(chrom-1)%5
 for seed in (101,102,103):
  for scenario,gamma in [('neighbor_lead',.2),('no_neighbor_signal',0.)]:
   rng=np.random.default_rng(seed);own=np.empty((n,nc,6));lead=np.empty_like(own)
   own[:,:,0]=rng.uniform(10.,20.,size=(n,nc));past=rng.normal(0,.2,size=(n,nc))
   for t in range(5):
    lead[:,:,t]=own[:,:,t]+rng.normal(0,1.5,size=(n,nc))
    rate=.3*past+gamma*(lead[:,:,t]-own[:,:,t])+rng.normal(0,.03,size=(n,nc))
    own[:,:,t+1]=own[:,:,t]+dt[t]*rate;past=rate
   lead[:,:,5]=own[:,:,5]+rng.normal(0,1.5,size=(n,nc))
   assert own.min()>0 and lead.min()>0
   a=own+rng.normal(0,.05,size=own.shape);b=own+rng.normal(0,.05,size=own.shape)
   neighbor=lead+rng.normal(0,.05,size=lead.shape)
   previous=np.diff(a,axis=2)/dt;target=np.diff(b,axis=2)/dt;residual=neighbor-a
   perm=rng.permutation(n);shuffled=residual[perm]
   base=np.stack([a[:,:,1:5],previous[:,:,:4]],axis=-1)
   features={'own_history':base,'cis_history':np.concatenate([base,residual[:,:,1:5,None]],axis=-1),
             'shuffled_history':np.concatenate([base,shuffled[:,:,1:5,None]],axis=-1),
             'zero_neighbor':np.concatenate([base,np.zeros_like(residual[:,:,1:5,None])],axis=-1)}
   for t in (2,3,4):
    predictions={k:np.full((n,nc),np.nan) for k in features}
    for k in range(5):
     tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k);train_intervals=range(1,t)
     y=np.concatenate([target[tr,:,j] for j in train_intervals])
     assert not set(chrom[tr])&set(chrom[te])
     for name,feature in features.items():
      x=np.concatenate([feature[tr,:,j-1] for j in train_intervals]);z=feature[te,:,t-1]
      pred,fit=forecast.ridge_fit_predict(x,y,z);predictions[name][te]=pred
    y=target[:,:,t];ownerr=np.sum((predictions['own_history']-y)**2)
    zerodiff=float(np.max(abs(predictions['zero_neighbor']-predictions['own_history'])))
    assert zerodiff<1e-12
    numerical.append(zerodiff)
    for model,pred in predictions.items():
     loss=np.sum((pred-y)**2)
     rows.append(dict(seed=seed,scenario=scenario,model=model,interval=t,start_hpf=times[t],end_hpf=times[t+1],n_peaks=n,n_celltypes=nc,squared_error=loss,own_history_squared_error=ownerr,persistence_squared_error=np.sum(y*y),improvement_vs_own_history=1-loss/ownerr,improvement_vs_persistence=1-loss/np.sum(y*y)))
 d=pd.DataFrame(rows);OUT.mkdir(parents=True,exist_ok=True);d.to_csv(OUT/'forecast_synthetic_controls.csv',index=False)
 pooled=d.groupby(['scenario','model'])[['squared_error','own_history_squared_error','persistence_squared_error']].sum().reset_index()
 pooled['improvement_vs_own_history']=1-pooled.squared_error/pooled.own_history_squared_error
 pooled['improvement_vs_persistence']=1-pooled.squared_error/pooled.persistence_squared_error
 pooled.to_csv(OUT/'forecast_synthetic_control_summary.csv',index=False)
 power=float(pooled.query('scenario=="neighbor_lead" and model=="cis_history"').improvement_vs_own_history.iloc[0])
 null=float(pooled.query('scenario=="no_neighbor_signal" and model=="cis_history"').improvement_vs_own_history.iloc[0])
 shuffled=float(pooled.query('scenario=="neighbor_lead" and model=="shuffled_history"').improvement_vs_own_history.iloc[0])
 assert power>.5 and abs(null)<.05 and abs(shuffled)<.05,(power,null,shuffled)
 fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
 for ax,scenario,title in zip(axes,['neighbor_lead','no_neighbor_signal'],['Known neighbor-leading change','No neighbor-dependent change']):
  for model,color in [('cis_history','#187b89'),('shuffled_history','#916e9e'),('zero_neighbor','#777777')]:
   q=d[(d.scenario==scenario)&(d.model==model)].groupby('interval').improvement_vs_own_history
   mu=q.mean();lo=q.min();hi=q.max();ax.plot(range(3),100*mu,'-o',color=color,label=model.replace('_',' '));ax.fill_between(range(3),100*lo,100*hi,color=color,alpha=.12)
  ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['14→16','16→19','19→24']);ax.set_title(title);ax.set_xlabel('Held-out interval (hpf)');ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Rate-prediction error reduction vs own history (%)');axes[0].legend(fontsize=9)
 fig.suptitle('Forecast estimator recovers planted neighbor information',fontsize=14)
 fig.text(.5,.015,'Same ridge helper and λ=0.1; past-only rolling origins, chromosome folds, independent measurement halves.\nThree fixed seeds; bands span seeds. Gaussian synthetic measurement noise; this is not a real-data power guarantee.',ha='center',fontsize=9)
 fig.subplots_adjust(bottom=.24,top=.83,wspace=.16);fig.savefig(OUT/'01_forecast_positive_and_null_controls.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'01_forecast_positive_and_null_controls.pdf',bbox_inches='tight');plt.close(fig)
 report=dict(status='passed',seeds=[101,102,103],n_peaks=n,n_celltypes=nc,stages=times.tolist(),ridge_mean_loss_penalty=forecast.RIDGE_LAMBDA,training='Earlier intervals only, disjoint chromosome folds; A predictors and independent B Gaussian measurement outcomes',process='latent rate =0.3 * previous latent rate + gamma * (current neighbor latent accessibility-current own accessibility) + N(0,0.03^2); gamma=0.2 planted, gamma=0 null; leader residual iid N(0,1.5^2); observations N(latent,0.05^2)',positive_control_error_reduction=power,null_error_reduction=null,shuffled_positive_control_error_reduction=shuffled,zero_neighbor_max_prediction_difference=max(numerical),no_edge_graph_returns_no_connected_targets=True,prespecified_checks='Planted incremental error reduction>50%; absolute null/shuffled benefit<5%; adding an exactly zero neighbor feature preserves own-history predictions within1e-12',scope='Estimator power for one strong synthetic process; not independent validation of Cicero graph, biological replication, or comprehensive real-data power.')
 savejson('forecast_synthetic_control_validation.json',report);print(json.dumps(report,indent=2),flush=True)

def audit_forecast():
 with h5py.File(HERE/'outputs/ct_cis_validation_v1/cis_split_input.h5') as h:
  local_targets=h['target_local_indices'][:];source=h['source_indices'][:];ids=h['peak_ids'].asstr()[:];times=h['stages'][:]
  arr={label:h['observed_cpm_'+label][:].astype(float) for label in ['a','b']};ct=h['celltypes'].asstr()[:]
 dt=np.diff(times);target_ids=ids[local_targets];chrom=np.array([int(i.split('-')[0]) for i in target_ids]);fold=(chrom-1)%5
 coeff=pd.read_csv(CIS/'forecast_coefficients.csv');metrics=pd.read_csv(CIS/'forecast_metrics.csv');foldmeta=json.loads((CIS/'forecast_folds.json').read_text())
 for item in foldmeta:
  assert not set(item['train_chromosomes'])&set(item['test_chromosomes'])
  assert max(item['training_interval_indices'])<item['test_interval']
  assert item['maximum_training_outcome_hpf']<=item['start_hpf'] if 'start_hpf' in item else item['maximum_training_outcome_hpf']<=times[item['test_interval']]
 checks=[];metric_errors=[];target_errors=[];maxcoef=0.;maxpred=0.;maxmean=0.;maxscale=0.;graphchecks=0
 with h5py.File(CIS/'forecast_predictions.h5') as save:
  assert np.array_equal(save['peak_ids'].asstr()[:],target_ids)
  assert np.array_equal(save['fold'][:],fold)
  for threshold in [.25,.5]:
   W=sparse.load_npz(CIS/f'graph_cis_threshold_{threshold:g}.npz');N=sparse.load_npz(CIS/f'graph_distance_control_threshold_{threshold:g}.npz')
   assert W.shape==(len(local_targets),len(source))
   connected=np.flatnonzero(np.asarray(W.sum(axis=1)).ravel()>0)
   assert np.allclose(np.asarray(W.sum(axis=1)).ravel()[connected],1)
   assert np.all(W[np.arange(len(local_targets)),local_targets].A1==0)
   for i in connected[::max(1,len(connected)//12)][:12]:
    row=W.getrow(i);manual=np.zeros((len(ct),6))
    for j,w in zip(row.indices,row.data):manual+=w*arr['a'][j]
    assert np.allclose(manual,(row@arr['a'].reshape(len(source),-1)).reshape(len(ct),6),atol=1e-12)
    graphchecks+=1
   for sh,th in [('a','b'),('b','a')]:
    group=save[f'threshold_{threshold:g}/{sh}_to_{th}'];idx=group['connected_target_indices'][:];assert np.array_equal(idx,connected)
    x=arr[sh][local_targets];y=np.diff(arr[th][local_targets],axis=2)/dt;prev=np.diff(x,axis=2)/dt
    cis=(W@arr[sh].reshape(len(source),-1)).reshape(x.shape)-x
    distance=(N@arr[sh].reshape(len(source),-1)).reshape(x.shape)-x
    base=np.stack([x[:,:,1:5],prev[:,:,:4]],axis=-1)
    features={'own_history':base,'cis_history':np.concatenate([base,cis[:,:,1:5,None]],axis=-1),'distance_history':np.concatenate([base,distance[:,:,1:5,None]],axis=-1)}
    target_errors.append(float(np.max(abs(group['target_rate'][:]-y[idx][:,:,[2,3,4]]))))
    for ti,t in enumerate([2,3,4]):
     for model in forecast.MODELS:
      pred=group[model][:,:,ti].astype(float);yy=y[idx,:,t];b=group['own_history'][:,:,ti].astype(float)
      row=metrics[(metrics.threshold==threshold)&(metrics.split==sh+'_to_'+th)&(metrics.interval==t)&(metrics.model==model)].iloc[0]
      calculated=np.mean((pred-yy)**2);relative=abs(calculated-row.mse)/max(abs(row.mse),1e-12);metric_errors.append(relative)
      assert relative<2e-6
     # Reconstruct representative complete folds with sklearn Ridge and CSV.
     k=(t+int(threshold*100))%5;train=idx[fold[idx]!=k];test=idx[fold[idx]==k]
     assert not set(chrom[train])&set(chrom[test])
     pos=np.searchsorted(idx,test);yy=np.concatenate([y[train,:,j] for j in range(1,t)])
     for model,feature in features.items():
      xx=np.concatenate([feature[train,:,j-1] for j in range(1,t)]);zz=feature[test,:,t-1]
      cf=coeff[(coeff.threshold==threshold)&(coeff.split==sh+'_to_'+th)&(coeff.test_interval==t)&(coeff.fold==k)&(coeff.model==model)]
      for c in [0,7,20]:
       fns=['current_own_cpm','previous_own_rate']+([] if model=='own_history' else ['neighbor_current_residual'])
       table=cf[cf.celltype==ct[c]].set_index('feature').loc[fns]
       mu=xx[:,c].mean(axis=0);sd=xx[:,c].std(axis=0);sd=np.where(sd>1e-12,sd,1.)
       maxmean=max(maxmean,float(np.max(abs(mu-table.feature_mean))));maxscale=max(maxscale,float(np.max(abs(sd-table.feature_scale))))
       m=Ridge(alpha=forecast.RIDGE_LAMBDA*len(xx),fit_intercept=True,solver='svd').fit((xx[:,c]-mu)/sd,yy[:,c])
       pp=m.predict((zz[:,c]-mu)/sd);stored=group[model][pos,c,ti]
       maxcoef=max(maxcoef,float(np.max(abs(m.coef_-table.standardized_coefficient))))
       maxpred=max(maxpred,float(np.max(abs(pp-stored))))
       assert np.allclose(pp,stored,rtol=2e-6,atol=2e-6)
       direct=table.intercept.iloc[0]+zz[:,c]@table.coefficient.to_numpy()
       assert np.allclose(pp,direct,rtol=1e-9,atol=1e-9)
       checks.append(dict(threshold=threshold,split=sh+'_to_'+th,test_interval=t,fold=k,model=model,celltype=str(ct[c]),n_train=len(xx),n_test=len(zz)))
 assert maxmean<1e-9 and maxscale<1e-9 and maxcoef<1e-8
 pd.DataFrame(checks).to_csv(OUT/'forecast_independent_checked_models.csv',index=False)
 result=dict(status='passed',numerical_independence='Saved source arrays reconstructed; independent sklearn Ridge SVD fits against raw helper normal-equation implementation',model_celltype_folds_checked=len(checks),metric_rows_checked=len(metric_errors),chromosome_temporal_partitions_checked=len(foldmeta),manual_weighted_neighbor_rows=graphchecks,max_mean_error=maxmean,max_scale_error=maxscale,max_standardized_coefficient_error=maxcoef,max_saved_prediction_error=maxpred,max_saved_target_rate_error=max(target_errors),max_relative_mse_error=max(metric_errors),no_self_edges=True,rows_sum_one=True,correct_other_half_change_target=True,source_future_never_in_features=True)
 savejson('forecast_independent_audit.json',result);print(json.dumps(result,indent=2),flush=True)

def audit_lagged():
 directory=HERE/'outputs/ct_cis_forecast_lagged_v1'
 with h5py.File(HERE/'outputs/ct_cis_validation_v1/cis_split_input.h5') as h:
  targets=h['target_local_indices'][:];source=h['source_indices'][:];times=h['stages'][:];ct=h['celltypes'].asstr()[:]
  ids=h['peak_ids'].asstr()[:][targets];arr={s:h['observed_cpm_'+s][:].astype(float) for s in ['a','b']}
 dt=np.diff(times);chrom=np.array([int(i.split('-')[0]) for i in ids]);fold=(chrom-1)%5
 coeff=pd.read_csv(directory/'forecast_coefficients.csv');metrics=pd.read_csv(directory/'forecast_metrics.csv')
 checks=[];errors=[];maxcoef=maxpred=maxmom=0.;shufflechecks=0
 with h5py.File(directory/'forecast_predictions.h5') as h,h5py.File(CIS/'forecast_predictions.h5') as prior:
  for threshold in [.25,.5]:
   W=sparse.load_npz(CIS/f'graph_cis_threshold_{threshold:g}.npz');N=sparse.load_npz(CIS/f'graph_distance_control_threshold_{threshold:g}.npz')
   for sh,th in [('a','b'),('b','a')]:
    path=f'threshold_{threshold:g}/{sh}_to_{th}';g=h[path];idx=g['connected_target_indices'][:]
    x=arr[sh][targets];r=np.diff(arr[sh],axis=2)/dt;y=np.diff(arr[th][targets],axis=2)/dt
    residual=(W@arr[sh].reshape(len(source),-1)).reshape(x.shape)-x
    control=(N@arr[sh].reshape(len(source),-1)).reshape(x.shape)-x
    lag=np.zeros_like(x);lag[:,:,1:]=(W@r.reshape(len(source),-1)).reshape(len(targets),len(ct),5)
    nulllag=np.zeros_like(x);nulllag[:,:,1:]=(N@r.reshape(len(source),-1)).reshape(len(targets),len(ct),5)
    perm=g['shuffled_connected_row_indices'][:].astype(int)
    # Independently reconstruct the deterministic within-chromosome abundance bins.
    rng=np.random.default_rng(forecast.SEED);expected=np.empty_like(perm);ch=chrom[idx].astype(str)
    for t in range(6):
     for c in range(len(ct)):
      for cc in np.unique(ch):
       ix=np.flatnonzero(ch==cc);order=ix[np.argsort(x[idx[ix],c,t],kind='stable')]
       for part in np.array_split(order,min(5,max(1,len(order)//10))):expected[part,c,t]=rng.permutation(part)
    assert np.array_equal(perm,expected)
    assert np.array_equal(ch[perm],np.broadcast_to(ch[:,None,None],perm.shape))
    rshuf=np.zeros_like(x);lshuf=np.zeros_like(x)
    rshuf[idx]=np.take_along_axis(residual[idx],perm,axis=0);lshuf[idx]=np.take_along_axis(lag[idx],perm,axis=0)
    shufflechecks+=perm.size
    own=np.stack([x[:,:,1:5],r[targets,:,:4]],axis=-1)
    features={'own_history':own,'cis_current':np.concatenate([own,residual[:,:,1:5,None]],axis=-1),
      'cis_lagged_only':np.concatenate([own,lag[:,:,1:5,None]],axis=-1),
      'cis_current_plus_lagged':np.concatenate([own,residual[:,:,1:5,None],lag[:,:,1:5,None]],axis=-1),
      'distance_current_plus_lagged':np.concatenate([own,control[:,:,1:5,None],nulllag[:,:,1:5,None]],axis=-1),
      'shuffled_current_plus_lagged':np.concatenate([own,rshuf[:,:,1:5,None],lshuf[:,:,1:5,None]],axis=-1)}
    for new,old in [('own_history','own_history'),('cis_current','cis_history'),('target_rate','target_rate')]:assert np.array_equal(g[new][:],prior[path+'/'+old][:])
    for ti,t in enumerate([2,3,4]):
     for model in ['persistence']+list(features):
      yy=y[idx,:,t];p=g[model][:,:,ti].astype(float);metric=metrics[(metrics.threshold==threshold)&(metrics.split==sh+'_to_'+th)&(metrics.interval==t)&(metrics.model==model)].iloc[0]
      e=abs(np.mean((p-yy)**2)-metric.mse)/max(abs(metric.mse),1e-12);errors.append(e);assert e<2e-6
     k=(t+int(threshold*100))%5;train=idx[fold[idx]!=k];test=idx[fold[idx]==k];pos=np.searchsorted(idx,test)
     assert not set(chrom[train])&set(chrom[test]);yy=np.concatenate([y[train,:,j] for j in range(1,t)])
     for model,feature in features.items():
      xx=np.concatenate([feature[train,:,j-1] for j in range(1,t)]);zz=feature[test,:,t-1]
      cf=coeff[(coeff.threshold==threshold)&(coeff.split==sh+'_to_'+th)&(coeff.test_interval==t)&(coeff.fold==k)&(coeff.model==model)]
      fns=['current_own_cpm','previous_own_rate']
      if model not in ['own_history','cis_lagged_only']:fns+=['neighbor_current_residual']
      if model not in ['own_history','cis_current']:fns+=['neighbor_previous_rate']
      for c in [0,7,20]:
       table=cf[cf.celltype==ct[c]].set_index('feature').loc[fns];mu=xx[:,c].mean(axis=0);sd=xx[:,c].std(axis=0);sd=np.where(sd>1e-12,sd,1.)
       maxmom=max(maxmom,float(np.max(abs(mu-table.feature_mean))),float(np.max(abs(sd-table.feature_scale))))
       estimator=Ridge(alpha=forecast.RIDGE_LAMBDA*len(xx),fit_intercept=True,solver='svd').fit((xx[:,c]-mu)/sd,yy[:,c])
       pred=estimator.predict((zz[:,c]-mu)/sd);stored=g[model][pos,c,ti]
       maxcoef=max(maxcoef,float(np.max(abs(estimator.coef_-table.standardized_coefficient))))
       maxpred=max(maxpred,float(np.max(abs(pred-stored))))
       assert np.allclose(pred,stored,rtol=2e-6,atol=2e-6)
       checks.append(dict(threshold=threshold,split=sh+'_to_'+th,test_interval=t,fold=k,model=model,celltype=str(ct[c])))
 assert maxmom<1e-9 and maxcoef<1e-8
 pd.DataFrame(checks).to_csv(OUT/'lagged_independent_checked_models.csv',index=False)
 result=dict(status='passed',model_celltype_folds_checked=len(checks),metric_rows_checked=len(errors),joint_shuffle_assignments_checked=shufflechecks,max_training_moment_error=maxmom,max_standardized_coefficient_error=maxcoef,max_saved_prediction_error=maxpred,max_relative_mse_error=max(errors),primary_baselines_and_targets_bitwise_equal=True,lagged_features_use_previous_interval_only=True,joint_current_and_lagged_shuffle_preserves_donors=True,independent_estimator='sklearn Ridge with SVD solver and training-only standardization')
 savejson('lagged_forecast_independent_audit.json',result);print(json.dumps(result,indent=2),flush=True)

def audit_sweep_summary_and_distance():
 """Check every aggregate and verify shortest-hop distances by graph recurrence."""
 metrics=pd.read_csv(FLOW/'sweep_metrics.csv');errors=[]
 with h5py.File(FLOW/'sweep_vectors.h5') as f:
  broad=f['broad_reference'][:].astype(bool);reachable=np.isfinite(f['root_hops'][:]);cache={}
  for t in range(5):
   g=f[f'interval_{t}'];source=g['source_active'][:].astype(bool)&~broad&reachable
   common=source.copy()
   for k in [10,30,60,100]:
    for beta in [0,25,50]:common&=g[f'k{k}_b{beta:03d}/active'][:].astype(bool)
   cache[t]={'all_source_supported':source,'broad_reference_sources':g['source_active'][:].astype(bool)&broad&reachable,
     'gradual':source&g['gradual'][:].astype(bool),'common_active_all_settings':common,
     'gradual_common_active':common&g['gradual'][:].astype(bool)}
  for row in metrics.itertuples():
   group=f[f'interval_{row.interval}/k{row.k}_b{int(round(row.beta*100)):03d}'];h=group if row.control=='observed' else group[row.control]
   m=cache[row.interval][row.stratum];active=h['active'][:].astype(bool);ex=h['root_excess_delta'][:];finite=m&active&np.isfinite(ex)
   assert row.n_source==m.sum() and row.n_projectable==finite.sum()
   if finite.any():
    for field in ['root_delta','root_baseline_delta','root_excess_delta']:
     calc=float(h[field][:][finite].mean());saved=getattr(row,'mean_'+field);errors.append(abs(calc-saved))
   else:assert np.isnan(row.mean_root_excess_delta)
   calc=float(np.nansum(ex[m])/m.sum()) if m.any() else np.nan
   assert np.allclose(calc,row.all_source_mean_excess_zero_unprojectable,equal_nan=True,atol=1e-12)
   matched=m&active&group['active'][:].astype(bool)&np.isfinite(ex)&np.isfinite(group['root_excess_delta'][:])
   assert matched.sum()==row.n_paired_with_observed
   delta=ex[matched]-group['root_excess_delta'][:][matched]
   assert np.allclose(delta.mean() if len(delta) else np.nan,row.mean_excess_minus_observed_paired,equal_nan=True,atol=1e-12)
 # For unweighted undirected graphs, roots=0, |h(u)-h(v)|<=1 on every
 # edge, and each nonroot having a neighbor at h-1 prove exact graph hops.
 hops=np.load(FLOW/'full_reference_graph_hops.npy')
 graph=sparse.load_npz(HERE/'outputs/ct_velocity_reprojection_v1/new_graph_csr.npz').tocsr()
 with h5py.File(HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5') as f:broad=f['broad_reference'][:].astype(bool)
 assert np.isfinite(hops).all() and np.array_equal(hops,np.round(hops))
 assert np.array_equal(hops==0,broad)
 minimum_neighbor=np.full(len(hops),np.inf);maxjump=0.
 for start in range(0,len(hops),4096):
  end=min(start+4096,len(hops));ptr=graph.indptr[start:end+1];targets=graph.indices[ptr[0]:ptr[-1]]
  sources=np.repeat(np.arange(start,end),np.diff(ptr));jump=np.abs(hops[sources]-hops[targets]);maxjump=max(maxjump,float(jump.max()))
  np.minimum.at(minimum_neighbor,sources,hops[targets]);np.minimum.at(minimum_neighbor,targets,hops[sources])
 assert maxjump<=1 and np.array_equal(minimum_neighbor[~broad],hops[~broad]-1)
 result=dict(status='passed',aggregate_rows_checked=len(metrics),max_root_mean_error=max(errors),all_cohort_counts_exact=True,paired_control_cohorts_exact=True,full_graph_vertices=len(hops),directed_edge_entries_checked=graph.nnz,reference_vertices=int(broad.sum()),maximum_reference_hops=float(hops.max()),graph_distance_proof='All roots and only roots have hop0; every undirected edge changes hops by at most1; each nonroot has a neighbor at hop-1. These conditions independently prove shortest distances to the reference set.',max_edge_hop_difference=maxjump)
 savejson('sweep_summary_and_distance_audit.json',result);print(json.dumps(result,indent=2),flush=True)

def audit_sweep():
 # Audit stored row primitives without calling the sweep implementation.
 maxcos=maxp=maxw=maxdrift=maxvec=0.;checked=0;rows=[]
 with h5py.File(FLOW/'sweep_vectors.h5') as f,h5py.File(HERE/'outputs/ct_peak_atlas_v1/full_ct_peak_atlas.h5') as atlas:
  xy={name:atlas[name][:].astype(float) for name in ['umap2d','umap3d']}
  a=f['audit'];queries=a['query_rows'][:];sources=a['source_indices'][:];hops=a['reference_hops_all'][:]
  for name in a:
   if not name.startswith('interval_'):continue
   g=a[name];t=int(name.split('_')[1]);k=int(name.split('_k')[1]);allg=f[f'interval_{t}/k{k}_b000'];active=f[f'interval_{t}/source_active'][:]
   ix=g['candidate_indices'][:];dist=g['candidate_distances'][:];x=g['source_state'][:];z=g['target_state'][:];v=g['source_velocity'][:];sv=g['source_valid'][:];tv=g['target_valid'][:]
   for ri,(qr,source) in enumerate(zip(queries,sources)):
    cosine=np.zeros(k);comparable=np.zeros(k,bool);weight=np.exp(-dist[ri]/dist[ri,-1]) if dist[ri,-1]>0 else np.ones(k)
    for j in range(k):
     shared=sv[ri]&tv[ri,j]&np.isfinite(v[ri]);delta=z[ri,j,shared]-x[ri,shared];rate=v[ri,shared];den=np.linalg.norm(delta)*np.linalg.norm(rate)
     if shared.sum()>=15 and den>1e-12 and active[qr]:
      comparable[j]=True;cosine[j]=np.clip(np.dot(delta,rate)/den,-1,1)
    assert np.array_equal(comparable,g['comparable'][ri]);maxcos=max(maxcos,float(np.max(abs(cosine-g['cosine'][ri]))))
    # Production stores cosine float32 before probabilities: reproduce this precision.
    cosine=cosine.astype(np.float32);keep=comparable&(cosine>=.2)&(cosine>0)
    p=weight*np.where(keep,cosine,0).astype(float);p[:]=0 if keep.sum()<2 else p
    if p.sum()>0:p=p/p.sum()
    w=weight*comparable
    if w.sum()>0:w=w/w.sum()
    maxp=max(maxp,float(np.max(abs(p-g['P'][ri]))));maxw=max(maxw,float(np.max(abs(w-g['W'][ri]))))
    if p.sum()>0:
     dh=hops[ix[ri]]-hops[source];root=float(p@dh);base=float(w@dh)
     maxdrift=max(maxdrift,abs(root-float(allg['root_delta'][qr])),abs(base-float(allg['root_baseline_delta'][qr])),abs(root-base-float(allg['root_excess_delta'][qr])))
     for dim in [2,3]:
      delta=xy[f'umap{dim}d'][ix[ri]]-xy[f'umap{dim}d'][source];norm=np.linalg.norm(delta,axis=1)
      directions=np.divide(delta,norm[:,None],out=np.zeros_like(delta),where=norm[:,None]>1e-12).astype(np.float32)
      vec=p@directions;maxvec=max(maxvec,float(np.max(abs(vec-allg[f'vectors{dim}d'][qr]))))
    else:assert np.isnan(allg['root_delta'][qr])
    checked+=1
   rows.append(dict(interval=t,k=k,rows_checked=len(queries)))
 assert maxcos<1e-6 and maxp<2e-7 and maxw<2e-7 and maxdrift<2e-6 and maxvec<2e-6,(maxcos,maxp,maxw,maxdrift,maxvec)
 pd.DataFrame(rows).to_csv(OUT/'sweep_independent_checked_rows.csv',index=False)
 result=dict(status='passed',rows_checked=checked,max_cosine_error=maxcos,max_probability_error=maxp,max_baseline_weight_error=maxw,max_root_drift_error=maxdrift,max_vector_error=maxvec,reference_excluded_from_probabilities=True,nonuniform_differences_preserved_by_reusing_fitted_rates=True)
 savejson('sweep_independent_audit.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['controls','forecast','lagged','sweep','summary','all'],default='all');args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if args.stage in ['controls','all']:controls()
 if args.stage in ['forecast','all']:audit_forecast()
 if args.stage in ['lagged','all']:audit_lagged()
 if args.stage in ['sweep','all']:audit_sweep()
 if args.stage in ['summary','all']:audit_sweep_summary_and_distance()
