#!/usr/bin/env python3
"""Post-primary exploratory extension: do recent cis-neighbor changes lead a peak?

This extension was specified after the current-neighbor-state forecast failed.
All fixed penalty, fold, cell-half, and temporal-origin settings are retained.
No model is selected or tuned on the test metrics; every result is reported.
"""
from pathlib import Path
import json,time
import h5py,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ct_cis_forecast as base
OUT=base.HERE/'outputs/ct_cis_forecast_lagged_v1'
MODELS=['persistence','own_history','cis_current','cis_lagged_only','cis_current_plus_lagged','distance_current_plus_lagged','shuffled_current_plus_lagged']

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 paths=[base.SRC/'cis_split_input.h5',base.SRC/'cis_directed_matched_control_edges.csv.gz',base.OUT/'forecast_predictions.h5']
 before=[base.sig(p) for p in paths]
 with h5py.File(paths[0]) as f:
  source=f['source_indices'][:];targets=f['target_local_indices'][:];ids=f['peak_ids'].asstr()[:];cts=f['celltypes'].asstr()[:];times=f['stages'][:]
  arrays={h:f['observed_cpm_'+h][:].astype(float) for h in ('a','b')}
 edges=pd.read_csv(paths[1]);chrom=np.array([s.split('-')[0] for s in ids[targets]]);fold=np.array([(int(c)-1)%5 for c in chrom]);dt=np.diff(times);n=len(targets)
 rows=[];ctrows=[];blockrows=[];coeffrows=[];checks=[]
 temp=OUT/'forecast_predictions.pending.h5'
 with h5py.File(temp,'w') as save,h5py.File(paths[2]) as prior:
  for key,v in [('peak_ids',ids[targets]),('celltypes',cts)]:save.create_dataset(key,data=np.asarray(v,object),dtype=h5py.string_dtype())
  for key,v in [('source_indices',source[targets]),('fold',fold),('stages',times)]:save.create_dataset(key,data=v)
  for threshold in base.THRESHOLDS:
   W,N,connected,mass=base.graphs(edges,source,targets,threshold);idx=np.flatnonzero(connected);ch=chrom[idx];folds=fold[idx]
   for sh,th in [('a','b'),('b','a')]:
    start=time.time();x=arrays[sh][targets];rates=np.diff(arrays[sh],axis=2)/dt;ownprev=rates[targets];y=np.diff(arrays[th][targets],axis=2)/dt
    neighbor=(W@arrays[sh].reshape(len(source),-1)).reshape(x.shape);null=(N@arrays[sh].reshape(len(source),-1)).reshape(x.shape)
    residual=neighbor-x;control=null-x;residual[~connected]=0;control[~connected]=0
    lag=np.zeros_like(x);lag[:,:,1:]=(W@rates.reshape(len(source),-1)).reshape(n,len(cts),5)
    nulllag=np.zeros_like(x);nulllag[:,:,1:]=(N@rates.reshape(len(source),-1)).reshape(n,len(cts),5)
    shuffled,audit=base.shuffled_residual(residual[idx],x[idx],ch);shuflag,auditlag=base.shuffled_residual(lag[idx],x[idx],ch);assert audit==auditlag
    permutation_values=np.broadcast_to(np.arange(len(idx))[:,None,None],residual[idx].shape).copy();permutation,auditperm=base.shuffled_residual(permutation_values,x[idx],ch);assert auditperm==audit
    assert np.array_equal(shuffled,residual[idx][permutation,np.arange(len(cts))[None,:,None],np.arange(6)[None,None,:]])
    assert np.array_equal(shuflag,lag[idx][permutation,np.arange(len(cts))[None,:,None],np.arange(6)[None,None,:]])
    rshuf=np.zeros_like(x);rshuf[idx]=shuffled;lshuf=np.zeros_like(x);lshuf[idx]=shuflag
    own=np.stack([x[:,:,1:5],ownprev[:,:,:4]],axis=-1)
    features={'own_history':own,'cis_current':np.concatenate([own,residual[:,:,1:5,None]],axis=-1),'cis_lagged_only':np.concatenate([own,lag[:,:,1:5,None]],axis=-1),'cis_current_plus_lagged':np.concatenate([own,residual[:,:,1:5,None],lag[:,:,1:5,None]],axis=-1),'distance_current_plus_lagged':np.concatenate([own,control[:,:,1:5,None],nulllag[:,:,1:5,None]],axis=-1),'shuffled_current_plus_lagged':np.concatenate([own,rshuf[:,:,1:5,None],lshuf[:,:,1:5,None]],axis=-1)}
    names={'own_history':['current_own_cpm','previous_own_rate'],'cis_current':['current_own_cpm','previous_own_rate','neighbor_current_residual'],'cis_lagged_only':['current_own_cpm','previous_own_rate','neighbor_previous_rate']}
    for m in MODELS:
     if m not in names:names[m]=['current_own_cpm','previous_own_rate','neighbor_current_residual','neighbor_previous_rate']
    pred={m:np.full((n,len(cts),3),np.nan) for m in MODELS};pred['persistence'][idx]=0
    for ti,t in enumerate(base.TEST_INTERVALS):
     trainints=np.arange(1,t)
     for k in range(5):
      train=idx[folds!=k];test=idx[folds==k];assert not set(chrom[train])&set(chrom[test]);assert all(times[j+1]<=times[t] for j in trainints)
      yy=np.concatenate([y[train,:,j] for j in trainints])
      for model,feature in features.items():
       xx=np.concatenate([feature[train,:,j-1,:] for j in trainints]);zz=feature[test,:,t-1,:];pp,fit=base.ridge_fit_predict(xx,yy,zz);pred[model][test,:,ti]=pp
       for c,ct in enumerate(cts):
        for p,fn in enumerate(names[model]):coeffrows.append(dict(threshold=threshold,split=sh+'_to_'+th,test_interval=t,fold=k,model=model,celltype=ct,feature=fn,training_examples_per_celltype=len(xx),training_intervals=','.join(map(str,trainints)),train_last_observation_hpf=times[t],intercept=float(fit['intercept'][c]),feature_mean=float(fit['mean'][c,p]),feature_scale=float(fit['scale'][c,p]),coefficient=float(fit['coefficient'][c,p]),standardized_coefficient=float(fit['standardized_coefficient'][c,p])))
      checks.append(dict(threshold=threshold,split=sh+'_to_'+th,interval=t,fold=k,training_interval_indices=trainints.tolist(),train_chromosomes=sorted(set(chrom[train]),key=int),test_chromosomes=sorted(set(chrom[test]),key=int),maximum_training_outcome_hpf=float(times[t]),forecast_outcome_hpf=float(times[t+1])))
    path=f'threshold_{threshold:g}/{sh}_to_{th}';g=save.create_group(path);g.create_dataset('connected_target_indices',data=idx);g.create_dataset('shuffled_connected_row_indices',data=permutation.astype('i4'),compression='gzip')
    for k,v in [('target_rate',y[:,:,base.TEST_INTERVALS])]+list(pred.items()):g.create_dataset(k,data=v[idx].astype('f4'),compression='gzip')
    for new,old in [('own_history','own_history'),('cis_current','cis_history')]:assert np.array_equal(pred[new][idx].astype('f4'),prior[path+'/'+old][:]),f'Primary baseline mismatch: {new}'
    assert np.array_equal(y[idx][:,:,base.TEST_INTERVALS].astype('f4'),prior[path+'/target_rate'][:])
    for ti,t in enumerate(base.TEST_INTERVALS):
     yy=y[idx,:,t];baseline=pred['own_history'][idx,:,ti]
     for model in MODELS:
      pp=pred[model][idx,:,ti];assert np.isfinite(pp).all();meta=dict(threshold=threshold,split=sh+'_to_'+th,interval=t,start_hpf=times[t],end_hpf=times[t+1],model=model)
      rows.append({**meta,**base.stats(yy,pp,baseline)})
      for c,ct in enumerate(cts):
       ee=np.sum((pp[:,c]-yy[:,c])**2);bb=np.sum((baseline[:,c]-yy[:,c])**2);zero=np.sum(yy[:,c]**2)
       ctrows.append({**meta,'celltype':ct,'n_peaks':len(idx),'squared_error':ee,'own_history_squared_error':bb,'persistence_squared_error':zero,'improvement_vs_own_history':1-ee/bb,'improvement_vs_persistence':1-ee/zero})
      for cc in np.unique(ch):
       mask=ch==cc;blockrows.append({**meta,'chromosome':cc,'n_peaks':int(mask.sum()),'squared_error':float(np.sum((pp[mask]-yy[mask])**2))})
    print(f'Lag extension threshold {threshold:g} {sh}->{th} complete {time.time()-start:.1f}s',flush=True)
 temp.replace(OUT/'forecast_predictions.h5')
 for fn,data in [('forecast_metrics',rows),('forecast_celltype_metrics',ctrows),('forecast_chromosome_losses',blockrows),('forecast_coefficients',coeffrows)]:pd.DataFrame(data).to_csv(OUT/(fn+'.csv'),index=False)
 base.js(OUT/'forecast_folds.json',checks);pooled=bootstrap(pd.DataFrame(blockrows));pooled.to_csv(OUT/'forecast_pooled_metrics.csv',index=False);plot(pooled,pd.DataFrame(ctrows))
 assert before==[base.sig(p) for p in paths]
 validation=dict(status='passed',post_primary_exploratory_extension=True,reason='Current linked-peak level residual did not improve primary prediction; test whether recent linked-peak changes add temporal information.',models=MODELS,thresholds=list(base.THRESHOLDS),ridge_mean_squared_loss_lambda=base.RIDGE_LAMBDA,settings_unchanged_from_primary=True,primary_own_and_cis_current_predictions_bitwise_equal=True,primary_targets_bitwise_equal=True,no_future_predictors=True,predictors='Own current CPM and prior rate; optional current cis-neighbor CPM residual and weighted neighbor previous observed rate. All measurements in source cell half.',target='Other-cell-half (observed CPM next stage - observed CPM current stage) / dt',held_out_chromosomes=True,rolling_origins=True,no_model_selected_from_test_scores=True,source_and_primary_files_unchanged=True,source_signatures=before,limitations=['Post-primary exploratory extension; not a new independent confirmation.','Cicero graph learned from all cells and stages, so all tests condition on a transductive fixed graph.','Independent cell halves quantify sampling, not biological replication.','Only 4,525 connected peaks at score0.25,807 at0.5; do not generalize to full peak atlas.','Three heldout developmental transitions; bootstrap chromosome bands describe block sampling sensitivity.'])
 base.js(OUT/'forecast_validation.json',validation)
 print(pooled[(pooled.scope=='all_test_origins')][['threshold','model','improvement_vs_own_history','improvement_vs_cis_current','improvement_vs_persistence']].to_string(index=False),flush=True)

def bootstrap(block):
 rng=np.random.default_rng(base.SEED);rows=[]
 for threshold in base.THRESHOLDS:
  d=block[block.threshold==threshold]
  for scope in ['all_test_origins']+list(base.TEST_INTERVALS):
   q=d if scope=='all_test_origins' else d[d.interval==scope];agg=q.groupby(['model','chromosome']).squared_error.sum();chrom=sorted(q.chromosome.unique(),key=int);draw=rng.integers(len(chrom),size=(2000,len(chrom)));loss={m:agg.loc[m].reindex(chrom).to_numpy() for m in MODELS}
   for m in MODELS:
    row=dict(threshold=threshold,scope=str(scope),model=m,n_chromosome_blocks=len(chrom),squared_error=float(loss[m].sum()))
    for label,bm in [('own_history','own_history'),('cis_current','cis_current'),('persistence','persistence'),('distance','distance_current_plus_lagged'),('shuffled','shuffled_current_plus_lagged')]:
     ratio=1-loss[m].sum()/loss[bm].sum();samples=1-loss[m][draw].sum(axis=1)/loss[bm][draw].sum(axis=1);lo,hi=np.quantile(samples,[.025,.975]);row['improvement_vs_'+label]=ratio;row[label+'_low']=lo;row[label+'_high']=hi
    rows.append(row)
 return pd.DataFrame(rows)

def plot(pooled,ct):
 colors={'cis_current':'#7a7a7a','cis_lagged_only':'#377eb8','cis_current_plus_lagged':'#137d74','distance_current_plus_lagged':'#c38a3d','shuffled_current_plus_lagged':'#916e9e'}
 fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
 for ax,threshold in zip(axes,base.THRESHOLDS):
  for model,col in colors.items():
   z=pooled[(pooled.threshold==threshold)&(pooled.model==model)&(pooled.scope!='all_test_origins')].copy();z['interval']=z.scope.astype(int);z=z.sort_values('interval');ax.plot(range(3),100*z.improvement_vs_own_history,'-o',color=col,label=model.replace('_',' '));ax.fill_between(range(3),100*z.own_history_low,100*z.own_history_high,color=col,alpha=.1)
  ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['14→16','16→19','19→24']);ax.set_xlabel('Held-out developmental interval (hpf)');ax.set_title(f'Cicero ≥{threshold:g}');ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Change-prediction error reduction vs own history (%)');axes[0].legend(fontsize=8);fig.suptitle('Do recent changes in linked peaks forecast future accessibility changes?',fontsize=15);fig.text(.5,.01,'Post-primary exploratory extension; unchanged penalty, rolling origins, chromosome folds and independent cell-half outcomes.\nBands: chromosome-block bootstrap. All models reported; fixed graph learned from all cells/stages.',ha='center',fontsize=9);fig.subplots_adjust(top=.83,bottom=.23,wspace=.14);savefig(fig,'01_lagged_cis_forecast')
 # CT-specific paired scores are descriptive; no subtype selected for claims.
 q=ct[ct.threshold==.25].groupby(['model','celltype'])[['squared_error','own_history_squared_error']].sum();cts=sorted(ct.celltype.unique());mods=['cis_current','cis_lagged_only','cis_current_plus_lagged','distance_current_plus_lagged','shuffled_current_plus_lagged'];a=np.array([[100*(1-q.loc[(m,c),'squared_error']/q.loc[(m,c),'own_history_squared_error']) for m in mods] for c in cts]);limit=max(1,np.percentile(np.abs(a),98));fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(a,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit);ax.set_yticks(range(len(cts)),cts);ax.set_xticks(range(len(mods)),[m.replace('_','\n') for m in mods],fontsize=9);ax.set_title('Per-celltype forecast comparison · Cicero ≥0.25\nAll three held-out intervals and both cell-half directions');fig.colorbar(im,ax=ax,label='Error reduction vs own history (%)',shrink=.75);fig.text(.5,.01,'Descriptive, uncorrected celltype comparisons. No celltype selected for validation claims; no biological replicates.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);savefig(fig,'02_lagged_cis_celltypes')

def savefig(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
