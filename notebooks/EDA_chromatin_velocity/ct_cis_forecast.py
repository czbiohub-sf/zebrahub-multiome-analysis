#!/usr/bin/env python3
"""Conditional Cicero forecasts with temporal and chromosome holdouts.

No six-stage fitted values enter this experiment. Predictors use source-half
observations at or before an interval start. Outcomes are differences entirely
within the other cell half: (B[t+1]-B[t])/dt. The fixed Cicero graph was learned
from all cells and all stages, so this does NOT independently test graph learning.
"""
from pathlib import Path
import json,sys,time,hashlib
import h5py,numpy as np,pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
SRC=HERE/'outputs/ct_cis_validation_v1'
OUT=HERE/'outputs/ct_cis_forecast_v1'
SEED=20261022
RIDGE_LAMBDA=.1  # mean squared training loss + lambda * ||standardized slopes||^2
THRESHOLDS=(.25,.5)
TEST_INTERVALS=(2,3,4)
MODELS=('persistence','previous_slope','own_history','cis_history','distance_history','shuffled_history','diffusion_0.1','diffusion_0.3','diffusion_0.6','diffusion_1.0')

def sig(p):
 s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}

def js(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def ridge_fit_predict(x,y,z):
 """Separate regularized regression per CT; all standardization training-only."""
 mu=x.mean(axis=0);sd=x.std(axis=0);sd=np.where(sd>1e-12,sd,1.);ym=y.mean(axis=0)
 xx=(x-mu)/sd;zz=(z-mu)/sd;n=len(x)
 gram=np.einsum('ncp,ncq->cpq',xx,xx)/n
 rhs=np.einsum('ncp,nc->cp',xx,y-ym)/n
 beta=np.linalg.solve(gram+RIDGE_LAMBDA*np.eye(x.shape[-1])[None],rhs[...,None])[...,0]
 coef=beta/sd;intercept=ym-np.sum(mu*coef,axis=-1)
 pred=intercept[None]+np.sum(z*coef[None],axis=-1)
 assert np.allclose(pred,ym[None]+np.sum(zz*beta[None],axis=-1),rtol=1e-10,atol=1e-10)
 return pred,dict(mean=mu,scale=sd,standardized_coefficient=beta,coefficient=coef,intercept=intercept)

def shuffled_residual(residual,own,chrom):
 """Shuffle target row residuals within chromosome and source-current CPM bins.

This control preserves the residual distribution conditional on current source
abundance. It is a feature permutation, not a degree-preserving rewired graph.
Bins are defined without outcomes. Small bins can preserve original assignments.
"""
 rng=np.random.default_rng(SEED);out=np.empty_like(residual);changed=total=0
 for t in range(6):
  for c in range(own.shape[1]):
   for ch in np.unique(chrom):
    ix=np.flatnonzero(chrom==ch);order=ix[np.argsort(own[ix,c,t],kind='stable')]
    # At most five equal-count abundance bins; at least ten entries per bin.
    for part in np.array_split(order,min(5,max(1,len(order)//10))):
     perm=rng.permutation(part);out[part,c,t]=residual[perm,c,t]
     changed+=int(np.sum(part!=perm));total+=len(part)
 return out,dict(assignments=total,changed=changed,fraction_changed=changed/total)

def graphs(edges,source,targets,threshold):
 local=np.full(int(source.max())+1,-1,int);local[source]=np.arange(len(source))
 target_local_to_row=np.full(len(source),-1,int);target_local_to_row[targets]=np.arange(len(targets))
 e=edges[edges.coaccess>=threshold];r=target_local_to_row[local[e.target.to_numpy()]]
 w=e.coaccess.to_numpy();cols=local[e.neighbor.to_numpy()];null=local[e.null_neighbor.to_numpy()]
 assert np.all(r>=0) and np.all(cols>=0) and np.all(null>=0)
 mass=np.bincount(r,weights=w,minlength=len(targets)).astype(float);inverse=np.divide(1.,mass,out=np.zeros_like(mass),where=mass>0)
 W=sparse.diags(inverse)@sparse.csr_matrix((w,(r,cols)),shape=(len(targets),len(source)))
 N=sparse.diags(inverse)@sparse.csr_matrix((w,(r,null)),shape=W.shape)
 connected=mass>0
 assert np.allclose(np.asarray(W.sum(axis=1)).ravel()[connected],1)
 assert np.allclose(np.asarray(N.sum(axis=1)).ravel()[connected],1)
 assert np.all(W[np.arange(len(targets)),targets].A1==0)
 return W.tocsr(),N.tocsr(),connected,mass

def stats(y,p,base):
 e=(p-y)**2;be=(base-y)**2;zero=y*y
 den=np.linalg.norm(y,axis=1)*np.linalg.norm(p,axis=1);nz=den>1e-12
 cosine=np.divide(np.sum(y*p,axis=1),den,out=np.full(len(y),np.nan),where=nz)
 yy=y.ravel();pp=p.ravel();corr=float(spearmanr(yy,pp).statistic) if np.std(pp)>1e-12 else np.nan
 return dict(n_peaks=len(y),n_peak_celltype_values=y.size,mse=float(e.mean()),mae=float(np.abs(p-y).mean()),sum_squared_error=float(e.sum()),own_history_sum_squared_error=float(be.sum()),persistence_sum_squared_error=float(zero.sum()),improvement_vs_persistence=float(1-e.sum()/zero.sum()) if zero.sum()>0 else np.nan,improvement_vs_own_history=float(1-e.sum()/be.sum()) if be.sum()>0 else np.nan,mean_peak_vector_cosine=float(np.nanmean(cosine)) if nz.any() else np.nan,rate_spearman=corr)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 source_paths=[SRC/'cis_split_input.h5',SRC/'cis_directed_matched_control_edges.csv.gz']
 before=[sig(p) for p in source_paths]
 with h5py.File(source_paths[0],'r') as f:
  source=f['source_indices'][:];targets=f['target_local_indices'][:];ids=f['peak_ids'].asstr()[:];celltypes=f['celltypes'].asstr()[:];times=f['stages'][:]
  arrays={h:f['observed_cpm_'+h][:].astype(float) for h in ('a','b')}
  counts={h:f['raw_counts_'+h][:] for h in ('a','b')}
  coverage={h:f['coverage_'+h][:] for h in ('a','b')}
 edges=pd.read_csv(source_paths[1]);chrom=np.array([i.split('-')[0] for i in ids[targets]])
 assert all(np.isfinite(a).all() for a in arrays.values())
 assert len(celltypes)==21 and np.array_equal(times,[10,12,14,16,19,24])
 for h in arrays:assert np.allclose(arrays[h],counts[h]*1e6/coverage[h][None],rtol=1e-7,atol=1e-7)
 dt=np.diff(times);fold=np.array([(int(c)-1)%5 for c in chrom]);n=len(targets)
 pd.DataFrame(dict(source_index=source[targets],peak_id=ids[targets],chromosome=chrom,fold=fold)).to_csv(OUT/'chromosome_folds.csv',index=False)
 rows=[];ctrows=[];blockrows=[];fitrows=[];checks=[];shuffle_audit={};examples=None
 tmp=OUT/'forecast_predictions.pending.h5'
 with h5py.File(tmp,'w') as save:
  save.create_dataset('peak_ids',data=np.asarray(ids[targets],object),dtype=h5py.string_dtype());save.create_dataset('celltypes',data=np.asarray(celltypes,object),dtype=h5py.string_dtype());save.create_dataset('source_indices',data=source[targets]);save.create_dataset('fold',data=fold);save.create_dataset('stages',data=times)
  for threshold in THRESHOLDS:
   W,N,connected,mass=graphs(edges,source,targets,threshold)
   for label,G in [('cis',W),('distance_control',N)]:sparse.save_npz(OUT/f'graph_{label}_threshold_{threshold:g}.npz',G)
   idx=np.flatnonzero(connected);ch=chrom[idx];folds=fold[idx]
   for sourcehalf,targethalf in [('a','b'),('b','a')]:
    start=time.time();x=arrays[sourcehalf][targets];y=np.diff(arrays[targethalf][targets],axis=2)/dt
    prev=np.diff(x,axis=2)/dt
    neighbor=(W@arrays[sourcehalf].reshape(len(source),-1)).reshape(x.shape)
    null=(N@arrays[sourcehalf].reshape(len(source),-1)).reshape(x.shape)
    residual=neighbor-x;control=null-x;residual[~connected]=0;control[~connected]=0
    perm,audit=shuffled_residual(residual[idx],x[idx],ch);shuffle_audit[f'{threshold:g}_{sourcehalf}']=audit
    zperm=np.zeros_like(residual);zperm[idx]=perm
    # Own history is measured in the source half only, including at heldout origins.
    basefeatures=np.stack([x[:,:,1:5],prev[:,:,:4]],axis=-1)
    features={'own_history':basefeatures,'cis_history':np.concatenate([basefeatures,residual[:,:,1:5,None]],axis=-1),'distance_history':np.concatenate([basefeatures,control[:,:,1:5,None]],axis=-1),'shuffled_history':np.concatenate([basefeatures,zperm[:,:,1:5,None]],axis=-1)}
    pred={m:np.full((n,len(celltypes),3),np.nan) for m in MODELS}
    pred['persistence'][idx]=0;pred['previous_slope'][idx]=prev[idx][:,:,np.array(TEST_INTERVALS)-1]
    for beta in (.1,.3,.6,1.):pred[f'diffusion_{beta:.1f}'][idx]=beta*residual[idx][:,:,TEST_INTERVALS]/dt[np.array(TEST_INTERVALS)][None,None,:]
    for ti,t in enumerate(TEST_INTERVALS):
     train_intervals=np.arange(1,t)
     assert all(times[j+1]<=times[t] for j in train_intervals)
     for k in range(5):
      train=idx[folds!=k];test=idx[folds==k]
      assert not set(chrom[train])&set(chrom[test])
      yy=np.concatenate([y[train,:,j] for j in train_intervals])
      for model,feature in features.items():
       xx=np.concatenate([feature[train,:,j-1,:] for j in train_intervals]);zz=feature[test,:,t-1,:]
       pp,fit=ridge_fit_predict(xx,yy,zz);pred[model][test,:,ti]=pp
       feature_names=['current_own_cpm','previous_own_rate']+([] if model=='own_history' else ['neighbor_current_residual'])
       for c,ct in enumerate(celltypes):
        for p,fn in enumerate(feature_names):
         fitrows.append(dict(threshold=threshold,split=sourcehalf+'_to_'+targethalf,test_interval=t,start_hpf=times[t],end_hpf=times[t+1],fold=k,model=model,celltype=ct,feature=fn,training_examples_per_celltype=len(xx),training_intervals=','.join(map(str,train_intervals)),train_last_observation_hpf=times[t],intercept=float(fit['intercept'][c]),feature_mean=float(fit['mean'][c,p]),feature_scale=float(fit['scale'][c,p]),coefficient=float(fit['coefficient'][c,p]),standardized_coefficient=float(fit['standardized_coefficient'][c,p])))
       # A distinct direct linear solve checks one celltype for each fit.
       zz0=(xx[:,0]-fit['mean'][0])/fit['scale'][0];ym0=yy[:,0].mean();direct=np.linalg.solve(zz0.T@zz0+RIDGE_LAMBDA*len(xx)*np.eye(xx.shape[-1]),zz0.T@(yy[:,0]-ym0));assert np.allclose(direct,fit['standardized_coefficient'][0],atol=1e-9)
      checks.append(dict(threshold=threshold,split=sourcehalf+'_to_'+targethalf,test_interval=t,fold=k,n_train_peaks=len(train),n_test_peaks=len(test),training_interval_indices=train_intervals.tolist(),train_chromosomes=sorted(set(chrom[train]),key=int),test_chromosomes=sorted(set(chrom[test]),key=int),maximum_training_outcome_hpf=float(times[t]),test_outcome_hpf=float(times[t+1])))
    group=save.create_group(f'threshold_{threshold:g}/{sourcehalf}_to_{targethalf}')
    group.create_dataset('connected_target_indices',data=idx)
    for name,a in [('target_rate',y[:,:,TEST_INTERVALS])]+[(m,a) for m,a in pred.items()]:group.create_dataset(name,data=a[idx].astype('f4'),compression='gzip')
    for ti,t in enumerate(TEST_INTERVALS):
     yy=y[idx,:,t];base=pred['own_history'][idx,:,ti]
     for model in MODELS:
      pp=pred[model][idx,:,ti];assert np.isfinite(pp).all()
      meta=dict(threshold=threshold,split=sourcehalf+'_to_'+targethalf,interval=t,start_hpf=times[t],end_hpf=times[t+1],model=model)
      rows.append({**meta,**stats(yy,pp,base)})
      for c,ct in enumerate(celltypes):
       ee=np.sum((pp[:,c]-yy[:,c])**2);bb=np.sum((base[:,c]-yy[:,c])**2);zero=np.sum(yy[:,c]**2)
       ctrows.append({**meta,'celltype':ct,'n_peaks':len(idx),'squared_error':ee,'own_history_squared_error':bb,'persistence_squared_error':zero,'improvement_vs_own_history':1-ee/bb,'improvement_vs_persistence':1-ee/zero})
      for cc in np.unique(ch):
       mask=ch==cc;ee=(pp[mask]-yy[mask])**2;bb=(base[mask]-yy[mask])**2;zero=yy[mask]**2
       blockrows.append({**meta,'chromosome':cc,'n_peaks':int(mask.sum()),'squared_error':float(ee.sum()),'own_history_squared_error':float(bb.sum()),'persistence_squared_error':float(zero.sum())})
    if threshold==.25 and sourcehalf=='a':
     # Selection uses only source A at 10/12 hpf, never later target stages.
     early=prev[:,:,0];eligible=connected[:,None]&(counts['a'][targets,:,:2].sum(axis=2)>=20)
     selected=[];usedpeaks=set();usedct=set()
     for sign in (1,-1):
      order=np.argsort(np.where(eligible,sign*early,-np.inf).ravel())[::-1]
      for flat in order:
       i,c=np.unravel_index(flat,early.shape)
       if not eligible[i,c] or sign*early[i,c]<=0:continue
       if i in usedpeaks or c in usedct:continue
       selected.append((i,c));usedpeaks.add(i);usedct.add(c)
       if len(selected)==(2 if sign==1 else 4):break
     examples=dict(selected=selected,source=x,target=arrays['b'][targets],neighbor=neighbor,y=y,pred=pred,ids=ids[targets],cts=celltypes,times=times)
    print(f'Threshold {threshold:g} {sourcehalf}->{targethalf}: {len(idx)} connected peaks; {time.time()-start:.1f}s',flush=True)
 tmp.replace(OUT/'forecast_predictions.h5')
 pd.DataFrame(rows).to_csv(OUT/'forecast_metrics.csv',index=False);pd.DataFrame(ctrows).to_csv(OUT/'forecast_celltype_metrics.csv',index=False);pd.DataFrame(blockrows).to_csv(OUT/'forecast_chromosome_losses.csv',index=False);pd.DataFrame(fitrows).to_csv(OUT/'forecast_coefficients.csv',index=False)
 js(OUT/'forecast_folds.json',checks)
 pooled=bootstrap(pd.DataFrame(blockrows));pooled.to_csv(OUT/'forecast_pooled_metrics.csv',index=False)
 plot(pd.DataFrame(rows),pooled,pd.DataFrame(fitrows),examples)
 assert before==[sig(p) for p in source_paths]
 validation=dict(status='passed',seed=SEED,n_primary_peaks=4525,n_secondary_peaks=int(sum(connected)),n_celltypes=len(celltypes),n_test_origins=3,chromosome_folds=5,cell_half_directions=2,ridge_penalty_mean_squared_loss=RIDGE_LAMBDA,thresholds=list(THRESHOLDS),no_self_edges=True,rows_normalized=True,target_definition='(other-half observed CPM at t+1 minus other-half observed CPM at t) / actual dt; current source measurements never subtracted from target',predictor_definition='Source-half current own CPM, previous observed rate, and optional current neighbor CPM minus own CPM',no_six_stage_fits=True,training_standardization_only=True,all_training_outcomes_at_or_before_test_origin=True,training_and_test_chromosomes_disjoint=True,distinct_direct_linear_solve_verified=True,finite_predictions=True,no_future_selected_strata=True,graph_fixed_learned_all_cells=True,primary_forecast_scopes='Threshold-specific connected peaks; threshold groups differ and must not be ranked as the same population.',shuffle_audit=shuffle_audit,source_files_unchanged=True,source_signatures=before,limitations=['Fixed Cicero weights learned from all cells/stages, including temporal test stages; results are conditional on that graph and are not fully prospective graph validation.','Cell halves are conditional sampling partitions, not biological replicates.','Only 4,525/640,830 atlas peaks have edges at score >=0.25; these are selected connected regions, not a global atlas result.','Five chromosome folds protect model coefficient fitting, but fixed graph estimation already included all chromosomes.','Distance controls preserve chromosome, distance bins where available, edge draw count and row weight mass; not degree/abundance/indegree matched.','Feature-shuffle control uses source-current abundance bins within chromosome; it is not a rewired graph.','Only three rolling-origin test transitions; bootstrap chromosomes estimates block sampling sensitivity, not biological uncertainty.','Raw CPM rates are relative accessibility; changes in total accessibility cannot be distinguished from composition without external calibration.'])
 js(OUT/'forecast_validation.json',validation)
 print(pooled.query('scope=="all_test_origins" and model in ["own_history","cis_history","distance_history","shuffled_history"]')[['threshold','model','improvement_vs_own_history','improvement_vs_persistence','paired_own_low','paired_own_high']].to_string(index=False),flush=True)

def bootstrap(block):
 rows=[];rng=np.random.default_rng(SEED)
 for threshold in THRESHOLDS:
  d=block[block.threshold==threshold]
  for scope in ['all_test_origins']+list(TEST_INTERVALS):
   q=d if scope=='all_test_origins' else d[d.interval==scope]
   agg=q.groupby(['model','chromosome'])[['squared_error','own_history_squared_error','persistence_squared_error']].sum()
   chromosomes=sorted(q.chromosome.unique(),key=int);draw=rng.integers(len(chromosomes),size=(2000,len(chromosomes)))
   loss={m:agg.loc[m].reindex(chromosomes).squared_error.to_numpy() for m in MODELS}
   for model in MODELS:
    e=loss[model];b=loss['own_history'];z=loss['persistence'];sh=loss['shuffled_history'];dist=loss['distance_history']
    ci={}
    for name,den in [('own',b),('shuffle',sh),('distance',dist),('persistence',z)]:
     vals=1-e[draw].sum(axis=1)/den[draw].sum(axis=1)
     ci['paired_'+name+'_low'],ci['paired_'+name+'_high']=np.quantile(vals,[.025,.975])
    rows.append(dict(threshold=threshold,scope=str(scope),model=model,n_chromosome_blocks=len(chromosomes),squared_error=e.sum(),improvement_vs_own_history=1-e.sum()/b.sum(),improvement_vs_persistence=1-e.sum()/z.sum(),improvement_vs_shuffled=1-e.sum()/sh.sum(),improvement_vs_distance=1-e.sum()/dist.sum(),**ci))
 return pd.DataFrame(rows)

def plot(metrics,pooled,coeff,examples):
 colors={'own_history':'#6a6a6a','cis_history':'#187b89','distance_history':'#c38a3d','shuffled_history':'#916e9e'}
 fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 for ax,threshold in zip(axes,THRESHOLDS):
  for model,color in colors.items():
   vals=pooled[(pooled.threshold==threshold)&(pooled.model==model)&(pooled.scope!='all_test_origins')].copy();vals['ii']=vals.scope.astype(int);vals=vals.sort_values('ii')
   ax.plot(np.arange(3),100*vals.improvement_vs_own_history,color=color,marker='o',label=model.replace('_',' '));ax.fill_between(np.arange(3),100*vals.paired_own_low,100*vals.paired_own_high,color=color,alpha=.12)
  n=int(metrics[metrics.threshold==threshold].n_peaks.iloc[0]);ax.set_title(f'Cicero ≥{threshold:g}: {n:,} connected peaks');ax.set_xticks(range(3),['14→16','16→19','19→24']);ax.axhline(0,color='black',lw=.7);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Held-out developmental interval (hpf)')
 axes[0].set_ylabel('Change-prediction error reduction vs own history (%)');axes[0].legend(fontsize=9);fig.suptitle('Does current cis-neighbor accessibility predict the next change?',fontsize=15)
 fig.text(.5,.015,'Rolling temporal origins + held-out chromosomes + independent cell-half outcomes. Bands: chromosome-block bootstrap.\nFixed graph learned from all cells/stages: conditional graph test; no six-stage fitted predictors.',ha='center',fontsize=9);fig.subplots_adjust(top=.83,bottom=.23,wspace=.15);savefig(fig,'01_cis_future_change')
 fig,axes=plt.subplots(1,2,figsize=(12,5))
 names=['persistence','previous_slope','own_history','cis_history','distance_history','shuffled_history','diffusion_0.1','diffusion_0.3','diffusion_0.6','diffusion_1.0']
 for ax,threshold in zip(axes,THRESHOLDS):
  vals=pooled[(pooled.threshold==threshold)&(pooled.scope=='all_test_origins')].set_index('model').loc[names]
  xx=100*vals.improvement_vs_persistence.to_numpy();ax.barh(range(len(names)),xx,color=[colors.get(n,'#b3bdc1') for n in names]);lo=100*vals.paired_persistence_low.to_numpy();hi=100*vals.paired_persistence_high.to_numpy();ax.errorbar(xx,range(len(names)),xerr=[np.maximum(xx-lo,0),np.maximum(hi-xx,0)],fmt='none',ecolor='#333333',capsize=2,lw=.7);ax.set_yticks(range(len(names)),[n.replace('_',' ') for n in names]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Change-prediction error reduction vs persistence (%)');ax.set_title(f'Cicero ≥{threshold:g}');ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Forecast benchmarks across three held-out intervals',fontsize=15);fig.text(.5,.01,'Fixed diffusion β predicts β(Wx−x)/Δt; β is a sensitivity grid, not selected using test outcomes.\nPersistence predicts zero change. Own-history ridge uses source-half current accessibility and its previous rate.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.09,1,.93]);savefig(fig,'02_forecast_benchmarks')
 # Coefficients show whether fitted coupling resembles positive diffusion.
 q=coeff[(coeff.model=='cis_history')&(coeff.feature=='neighbor_current_residual')]
 fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
 for ax,threshold in zip(axes,THRESHOLDS):
  z=q[q.threshold==threshold]
  for k,t in enumerate(TEST_INTERVALS):
   v=z[z.test_interval==t].coefficient.to_numpy();ax.boxplot(v,positions=[k],widths=.5,showfliers=False);ax.text(k,.98,f'{100*np.mean(v>0):.0f}% positive',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9)
  ax.axhline(0,color='#777777',lw=.8);ax.set_xticks(range(3),['14→16','16→19','19→24']);ax.set_title(f'Cicero ≥{threshold:g}');ax.set_xlabel('Forecast interval (hpf)')
 axes[0].set_ylabel('Neighbor-residual coefficient (1/hour)');fig.suptitle('Does learned coupling point toward the cis-neighbor state?',fontsize=14);fig.text(.5,.01,'Distribution over 21 celltypes × 5 held-out chromosome folds × 2 cell-half directions; not independent replicates.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.94]);savefig(fig,'03_cis_coupling_coefficients')
 if examples:
  e=examples;fig,axes=plt.subplots(4,2,figsize=(13,13));erows=[]
  for row,(i,c) in enumerate(e['selected']):
   ax=axes[row,0];ax.plot(e['times'],e['source'][i,c],'-o',color='#555555',label='Own, source A');ax.plot(e['times'],e['target'][i,c],'-o',color='#c78137',label='Own, held-out half B');ax.plot(e['times'],e['neighbor'][i,c],'-o',color='#187b89',label='Cis weighted neighbors, A');ax.axvspan(14,24,color='#ededed',alpha=.5,zorder=-10);ax.set_title(f'{e["ids"][i]} · {e["cts"][c]}',fontsize=10);ax.set_ylabel('Observed CPM');ax.set_xticks(e['times']);ax.spines[['top','right']].set_visible(False)
   bx=axes[row,1];xx=np.arange(3);bx.plot(xx,e['y'][i,c,TEST_INTERVALS],'-o',color='#c78137',label='Held-out change, B');
   for model,col in [('own_history','#555555'),('cis_history','#187b89')]:bx.plot(xx,e['pred'][model][i,c],'-o',color=col,label=model.replace('_',' '))
   bx.axhline(0,color='black',lw=.6);bx.set_xticks(xx,['14→16','16→19','19→24']);bx.set_ylabel('CPM/hour');bx.spines[['top','right']].set_visible(False)
   for t in TEST_INTERVALS:erows.append(dict(peak_id=e['ids'][i],celltype=e['cts'][c],interval=t,selection='Two largest positive and two largest negative source-half first-interval rates, distinct peaks/celltypes, >=20 source counts across10/12hpf; no later outcomes used',observed_other_half_rate=e['y'][i,c,t],own_history_predicted_rate=e['pred']['own_history'][i,c,t-2],cis_predicted_rate=e['pred']['cis_history'][i,c,t-2]))
  axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8);fig.suptitle('Examples selected using only source-half 10/12 hpf measurements',fontsize=15);fig.text(.5,.01,'Selected early changes are illustrations, not a representative sample. Forecasts use past/current observations only; each target chromosome was held out from coefficient fitting.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.96]);savefig(fig,'04_training_selected_examples');pd.DataFrame(erows).to_csv(OUT/'training_selected_examples.csv',index=False)

def savefig(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
