#!/usr/bin/env python3
"""Post-primary sensitivity: select both training and tuning populations.

The frozen primary fits all eligible targets and only stratifies evaluation.
Here the same past-only selector also restricts fitting and inner validation.
This changes training AND tuning populations, not training alone. The entire
outer evaluation cohort, graph, model family and tuning grid remain fixed.
"""
from pathlib import Path
import time
import h5py,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ct_shared_weight_forecast as base
OUTROOT=base.ROOT/'model_selected_refit'
COHORTS=('total_variance_top20','gini_top20')

def tune_model_cohort(d,half,origin,outertrain,model,training_cohort):
 latest=origin-1;trainints=list(range(1,latest)) or [latest]
 losses=np.zeros((len(base.LAMBDAS),len(base.SHRINKAGES)));count=0;details=[]
 for inner in range(3):
  innertr=outertrain&(d['inner_fold']!=inner);innerva=outertrain&(d['inner_fold']==inner)
  selection=base.selector(d,half,latest,innertr);mask=selection['masks'][training_cohort]
  train=np.flatnonzero(innertr&mask);val=np.flatnonzero(innerva&mask)
  assert not set(d['chromosome'][train])&set(d['chromosome'][val]);assert len(train)>0 and len(val)>0
  own,extra,y,gate=base.get_training(d,half,trainints,train,model)
  testown,testextra=base.feature_block(d['raw'][half][latest],model,gate);testown=testown[val];testextra=testextra[val]
  other='b' if half=='a' else 'a';a=d['observed'][other][d['target_local_indices']];truth=(a[val,:,latest+1]-a[val,:,latest])/(d['stages'][latest+1]-d['stages'][latest])
  for li,penalty in enumerate(base.LAMBDAS):
   fit=base.fit_shared_ridge(own,extra,y,penalty);pred=base.predict_shared(fit,testown,testextra)
   for gi,shrink in enumerate(base.SHRINKAGES):losses[li,gi]+=np.sum((shrink*pred-truth)**2)
  count+=truth.size
  details.append(dict(inner_fold=inner,n_train_peaks=len(train),n_validation_peaks=len(val),training_intervals=trainints,validation_interval=latest,training_cohort=training_cohort,train_chromosomes=sorted(set(d['chromosome'][train]),key=int),validation_chromosomes=sorted(set(d['chromosome'][val]),key=int),gate=gate,selection_audit=selection['audit']))
 candidates=[(float(losses[li,gi]),float(shrink),-float(penalty),li,gi) for li,penalty in enumerate(base.LAMBDAS) for gi,shrink in enumerate(base.SHRINKAGES)]
 best=min(candidates);li,gi=best[-2:]
 return dict(penalty=float(base.LAMBDAS[li]),shrinkage=float(base.SHRINKAGES[gi]),validation_loss=losses,validation_n_values=count,inner_details=details,inner_temporal_holdout=latest>1,training_and_tuning_cohort=training_cohort)

def run_outer_fold_cohort(d,half,origin,fold,training_cohort,models=base.MODELS):
 trainmask=d['fold']!=fold;test=np.flatnonzero(d['fold']==fold);selection=base.selector(d,half,origin,trainmask);train=np.flatnonzero(trainmask&selection['masks'][training_cohort]);trainints=list(range(1,origin));answer={}
 assert not set(d['chromosome'][train])&set(d['chromosome'][test])
 for model in models:
  tuning=tune_model_cohort(d,half,origin,trainmask,model,training_cohort)
  own,extra,y,gate=base.get_training(d,half,trainints,train,model);fit=base.fit_shared_ridge(own,extra,y,tuning['penalty']);testown,testextra=base.feature_block(d['raw'][half][origin],model,gate)
  raw=base.predict_shared(fit,testown[test],testextra[test]);answer[model]=dict(prediction=tuning['shrinkage']*raw,unshrunk_prediction=raw,fit=fit,gate=gate,tuning=tuning)
 return dict(test_indices=test,train_indices=train,selection=selection,models=answer,training_intervals=trainints)

def main():
 OUTROOT.mkdir(parents=True,exist_ok=True)
 primary=base.OUT/'predictions.h5';signature=base.signature(primary)
 base.dump(OUTROOT/'exploratory_design.json',dict(post_primary_exploratory=True,reason='Primary all-eligible fit only stratified evaluation; this bounded extension tests the user-requested variable-target training combination.',cohorts=list(COHORTS),selection_fraction=.2,training_population='Chosen past-only selected subset among outertrainingchromosomes',tuning_population='Same chosen selection rule recomputed atinnerorigin usinginnertrainingchromcalibration; bothinnertraining andvalidation areselected',outer_test_masks='Bitwise identical to frozenprimary masks',models=list(base.MODELS),penalties=list(base.LAMBDAS),shrinkages=list(base.SHRINKAGES),no_test_tuning=True,all_neighbors_retained=True,not_independent_confirmation=True,comparison_interpretation='Joint change in training AND tuning populations; does not isolate a training-sample-only effect.'))
 d=base.load_data()
 for cohort in COHORTS:run_cohort(d,cohort,primary)
 assert signature==base.signature(primary)
 base.dump(OUTROOT/'validation.json',dict(status='passed',primary_unchanged=True,all_neighbors_retained=True,outer_cohorts_bitwise_identical=True,cohorts=list(COHORTS),post_primary_exploratory=True))

def run_cohort(d,cohort,primary):
 out=OUTROOT/cohort;out.mkdir(parents=True,exist_ok=True);models=['persistence']+list(base.MODELS);n=len(d['peak_ids']);C=len(d['celltypes']);rows=[];blocks=[];ctrows=[];tunerows=[];coeffrows=[];foldrows=[]
 with h5py.File(primary,'r') as old,h5py.File(out/'predictions.pending.h5','w') as h:
  for k in ['peak_ids','celltypes','chromosome','source_indices','fold','stages','origin_indices']:old.copy(k,h)
  h.attrs['axes']='Predictions,targets: peak,celltype,origin; masks: peak,origin';h.attrs['training_and_tuning_cohort']=cohort;h.attrs['post_primary_exploratory']=True
  for half in ('a','b'):
   other='b' if half=='a' else 'a';a=d['observed'][other][d['target_local_indices']];truth=np.stack([(a[:,:,t+1]-a[:,:,t])/(d['stages'][t+1]-d['stages'][t]) for t in base.ORIGINS],axis=2)
   assert np.array_equal(truth.astype('f4'),old[f'half_{half}/target_rate'][:])
   masks={k:old[f'half_{half}/cohorts/{k}'][:] for k in old[f'half_{half}/cohorts']}
   pred={m:np.zeros((n,C,3)) if m=='persistence' else np.full((n,C,3),np.nan) for m in models};rawpred={m:np.zeros((n,C,3)) if m=='persistence' else np.full((n,C,3),np.nan) for m in models}
   for ti,t in enumerate(base.ORIGINS):
    for fold in range(5):
     start=time.time();r=run_outer_fold_cohort(d,half,t,fold,cohort);train=r['train_indices'];test=r['test_indices'];sel=r['selection']
     for name,key in [('all_eligible','all'),('gini_top20','gini_top20'),('total_variance_top20','total_variance_top20')]:assert np.array_equal(sel['masks'][key][test],masks[name][test,ti])
     fg=h.require_group(f'fits/half_{half}/origin_{t}/fold_{fold}');fg.create_dataset('train_indices',data=train);fg.create_dataset('test_indices',data=test)
     foldrows.append(dict(half=half,origin=t,fold=fold,n_train=len(train),n_test=len(test),training_and_tuning_cohort=cohort,train_chromosomes=sorted(set(d['chromosome'][train]),key=int),test_chromosomes=sorted(set(d['chromosome'][test]),key=int),training_intervals=r['training_intervals'],test_origin_hpf=float(d['stages'][t]),test_outcome_hpf=float(d['stages'][t+1])))
     base.dump(out/f'selection_half_{half}_origin_{t}_fold_{fold}.json',sel['audit'])
     for model,item in r['models'].items():
      pred[model][test,:,ti]=item['prediction'];rawpred[model][test,:,ti]=item['unshrunk_prediction'];fit=item['fit'];mg=fg.require_group(model);base.write_fit(mg,fit,item['gate']);tune=item['tuning'];mg.attrs['shrinkage']=tune['shrinkage'];mg.create_dataset('inner_loss',data=tune['validation_loss']);base.dump(out/f'tuning_half_{half}_origin_{t}_fold_{fold}_{model}.json',tune)
      for li,p in enumerate(base.LAMBDAS):
       for si,s in enumerate(base.SHRINKAGES):tunerows.append(dict(half=half,origin=t,fold=fold,model=model,penalty=p,shrinkage=s,validation_squared_rate_error=tune['validation_loss'][li,si],n_validation_values=tune['validation_n_values'],selected=p==tune['penalty'] and s==tune['shrinkage'],inner_temporal_holdout=tune['inner_temporal_holdout']))
      for j,fn in enumerate(base.FEATURE_NAMES[model]):coeffrows.append(dict(half=half,origin=t,fold=fold,model=model,feature=fn,shared_across_all_celltypes=True,standardized_coefficient=fit['shared_beta'][j],unstandardized_coefficient=fit['shared_beta'][j]/fit['shared_scale'][j],final_shrinkage=tune['shrinkage'],penalty=tune['penalty'],gate_mean=item['gate']['mean'],gate_scale=item['gate']['scale']))
     base.log(f'Selected {cohort} {half} origin{d["stages"][t]:g} fold{fold}: {time.time()-start:.1f}s')
   hg=h.require_group('half_'+half);hg.create_dataset('target_rate',data=truth.astype('f4'),compression='gzip')
   for c,v in masks.items():hg.require_group('cohorts').create_dataset(c,data=v,compression='gzip')
   for m in models:
    assert np.isfinite(pred[m]).all();hg.require_group('predictions').create_dataset(m,data=pred[m].astype('f4'),compression='gzip');hg.require_group('unshrunk_predictions').create_dataset(m,data=rawpred[m].astype('f4'),compression='gzip')
   for ti,t in enumerate(base.ORIGINS):
    dt=d['stages'][t+1]-d['stages'][t];use=masks[cohort][:,ti];ix=np.flatnonzero(use)
    for m in models:
     err=(pred[m][:,:,ti]-truth[:,:,ti])**2;baseline=(pred['own_history'][:,:,ti]-truth[:,:,ti])**2;zero=truth[:,:,ti]**2;primaryerr=(old[f'half_{half}/predictions/{m}'][:,:,ti].astype(float)-truth[:,:,ti])**2
     meta=dict(half=half,origin=t,start_hpf=d['stages'][t],end_hpf=d['stages'][t+1],cohort=cohort,model=m)
     rows.append({**meta,'n_peaks':len(ix),'n_values':len(ix)*C,'rate_sse':float(err[ix].sum()),'state_change_sse':float(err[ix].sum()*dt*dt),'own_history_sse':float(baseline[ix].sum()),'persistence_sse':float(zero[ix].sum()),'primary_same_model_rate_sse':float(primaryerr[ix].sum()),'improvement_vs_own_history':float(1-err[ix].sum()/baseline[ix].sum()),'improvement_vs_persistence':float(1-err[ix].sum()/zero[ix].sum())})
     for chrom in np.unique(d['chromosome']):
      jj=np.flatnonzero(use&(d['chromosome']==chrom));blocks.append({**meta,'chromosome':chrom,'n_peaks':len(jj),'rate_sse':float(err[jj].sum()),'state_change_sse':float(err[jj].sum()*dt*dt),'primary_same_model_rate_sse':float(primaryerr[jj].sum()),'primary_same_model_state_change_sse':float(primaryerr[jj].sum()*dt*dt)})
     for c,name in enumerate(d['celltypes']):ctrows.append({**meta,'celltype':name,'n_peaks':len(ix),'rate_sse':float(err[ix,c].sum()),'state_change_sse':float(err[ix,c].sum()*dt*dt)})
 (out/'predictions.pending.h5').replace(out/'predictions.h5')
 for name,data in [('metrics',rows),('chromosome_losses',blocks),('celltype_losses',ctrows),('tuning_grid',tunerows),('shared_coefficients',coeffrows)]:pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
 base.dump(out/'outer_folds.json',foldrows);pooled=base.summarize(pd.DataFrame(blocks));pooled.to_csv(out/'pooled_metrics.csv',index=False);comparison=compare(pd.DataFrame(blocks));comparison.to_csv(out/'fit_strategy_comparison.csv',index=False);plot(out,cohort,pooled,comparison)
 base.dump(out/'validation.json',dict(status='passed',outer_cohort_masks_bitwise_identical_to_primary=True,target_rates_bitwise_identical_to_primary=True,training_and_tuning_cohort=cohort,all_graph_neighbors_retained=True,no_new_hyperparameter_options=True,post_primary_exploratory=True,source_files_unchanged=all(base.signature(Path(s['path']))==s for s in d['source_signatures'])))
 text=['SELECTED TRAINING AND TUNING SENSITIVITY',cohort,'','This is a post-primary exploratory extension. Both fitting and inner tuning use the selected population, with selectors recomputed from source-half past observations and training chromosomes at the applicable origin. Outer evaluation masks are exactly the primary masks. Models, graph, retained neighbors, penalty/shrinkage grid and fold assignments are unchanged. No model was selected from outer test scores.','',pooled[(pooled.interval=='pooled')&(pooled.error=='rate_sse')][['model','vs_persistence','vs_own_history','vs_own_nonlinear','vs_distance_flexible','vs_generic_smoothing']].to_string(index=False),'','JOINT TRAINING/TUNING CHANGE RELATIVE TO THE SAME PRIMARY MODEL',comparison[(comparison.interval=='pooled')&(comparison.error=='rate_sse')].to_string(index=False),'','This comparison does not isolate training-target selection from tuning-population selection. It is not independent confirmation of the primary result. All prior limitations (all-stage Cicero graph, conditional sampling halves, relative CPM, only4525connected targets and three forecast intervals) remain.']
 (out/'summary.txt').write_text('\n'.join(text)+'\n')

def compare(blocks):
 rng=np.random.default_rng(base.SEED);rows=[]
 for interval in ['pooled']+list(base.ORIGINS):
  q=blocks if interval=='pooled' else blocks[blocks.origin==interval];agg=q.groupby(['model','chromosome'])[['rate_sse','state_change_sse','primary_same_model_rate_sse','primary_same_model_state_change_sse']].sum();chrom=sorted(q.chromosome.unique(),key=int);draw=rng.integers(len(chrom),size=(2000,len(chrom)))
  for model in ['persistence']+list(base.MODELS):
   a=agg.loc[model].reindex(chrom)
   for error in ['rate_sse','state_change_sse']:
    new=a[error].to_numpy();old=a['primary_same_model_'+error].to_numpy();value=1-new.sum()/old.sum();lo,hi=np.quantile(1-new[draw].sum(axis=1)/old[draw].sum(axis=1),[.025,.975]);rows.append(dict(interval=str(interval),model=model,error=error,selected_training_sse=new.sum(),all_training_same_model_sse=old.sum(),improvement_selected_training_and_tuning_vs_all=value,low=lo,high=hi))
 return pd.DataFrame(rows)

def plot(out,cohort,pooled,comparison):
 colors={'own_history':'#777777','own_nonlinear':'#8f6b9b','constant_cis':'#78a4a4','flexible_cis':'#087e79','distance_flexible':'#c58b35','generic_smoothing':'#527fa8'}
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 for model in base.MODELS:
  p=pooled[(pooled.error=='rate_sse')&(pooled.interval!='pooled')&(pooled.model==model)].sort_values('interval');q=comparison[(comparison.error=='rate_sse')&(comparison.interval!='pooled')&(comparison.model==model)].sort_values('interval');x=np.arange(3)
  axes[0].plot(x,100*p.vs_persistence,'-o',color=colors[model],label=model.replace('_',' '),ms=4);axes[0].fill_between(x,100*p.vs_persistence_low,100*p.vs_persistence_high,color=colors[model],alpha=.08)
  axes[1].plot(x,100*q.improvement_selected_training_and_tuning_vs_all,'-o',color=colors[model],ms=4);axes[1].fill_between(x,100*q.low,100*q.high,color=colors[model],alpha=.08)
 for ax in axes:ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['14→16','16→19','19→24']);ax.set_xlabel('Forecast interval (hpf)');ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Squared-rate error reduction (%)');axes[0].set_title('Selected training/tuning vs persistence');axes[1].set_title('Selected vs all-eligible training/tuning');axes[0].legend(fontsize=7);fig.suptitle(cohort.replace('_',' ')+' · exploratory selected-target refit',fontsize=13);fig.text(.5,.01,'Both panels score exactly the same selected outer test peaks as the frozen primary experiment.\nBoth training AND inner tuning populations change; all model settings and graph neighbors remain fixed. Bands: chromosome-block bootstrap.',ha='center',fontsize=8);fig.subplots_adjust(top=.82,bottom=.23,wspace=.25);fig.savefig(out/'01_selected_refit.png',dpi=180,bbox_inches='tight');fig.savefig(out/'01_selected_refit.pdf',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
