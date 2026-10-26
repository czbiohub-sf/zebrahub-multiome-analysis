#!/usr/bin/env python3
"""Small shared signed Cicero operators, with nested past-only model selection.

Each graph model has at most six coefficients shared by all peaks AND celltypes.
The own-history baseline has celltype-specific intercept/current-state/prior-rate
coefficients. No coefficients are fitted separately for edges or individual peaks.
"""
from pathlib import Path
import json,time,hashlib
import h5py,numpy as np,pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ct_variable_peak_forecast_selection import fit_origin_selection
HERE=Path(__file__).resolve().parent
ROOT=HERE/'outputs/ct_shared_weight_forecast_v1'
OUT=ROOT/'model'
CIS=HERE/'outputs/ct_cis_validation_v1'
SELECT=ROOT/'selection_audit/target_32ct_split_input.h5'
SEED=20261026
ORIGINS=(2,3,4)
LAMBDAS=(.1,1.,10.)
SHRINKAGES=(0.,.25,.5,1.)
MODELS=('own_history','own_nonlinear','constant_cis','flexible_cis','distance_flexible','generic_smoothing')
FEATURE_NAMES={
 'own_history':[],
 'own_nonlinear':['z_squared','current_own_linear_cpm','z_times_own_previous_rate','z_squared_times_own_previous_rate','own_previous_rate_squared_over_one_plus_absolute','log1p_absolute_own_previous_rate'],
 'constant_cis':['neighbor_current_residual','neighbor_previous_rate'],
 'flexible_cis':['neighbor_current_residual','neighbor_previous_rate','recipient_state_times_neighbor_rate','partner_state_times_partner_rate','distance_times_partner_rate','score_times_partner_rate'],
 'distance_flexible':['neighbor_current_residual','neighbor_previous_rate','recipient_state_times_neighbor_rate','partner_state_times_partner_rate','distance_times_partner_rate','score_times_partner_rate'],
 'generic_smoothing':['neighbor_current_residual','neighbor_previous_rate']}

def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def signature(p):
 q=p.stat();return dict(path=str(p.resolve()),bytes=q.st_size,mtime_ns=q.st_mtime_ns)
def dump(p,d):
 def cv(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  raise TypeError(type(x).__name__)
 Path(p).write_text(json.dumps(d,indent=2,default=cv,allow_nan=False)+'\n')

def build_static_graphs(source,targets,edges):
 local=np.full(int(source.max())+1,-1,int);local[source]=np.arange(len(source))
 row=np.full(len(source),-1,int);row[targets]=np.arange(len(targets))
 e=edges[edges.coaccess>=.25];rr=row[local[e.target.to_numpy()]];score=e.coaccess.to_numpy(float)
 mass=np.bincount(rr,weights=score,minlength=len(targets));assert (mass>0).all()
 w=score/mass[rr];answer={}
 for name,col,dist in [('cis',local[e.neighbor.to_numpy()],e.distance_bp.to_numpy()),('distance',local[e.null_neighbor.to_numpy()],e.null_distance_bp.to_numpy())]:
  dd=np.clip(np.log1p(dist)/np.log1p(500000.),0,1)
  answer[name]={key:sparse.csr_matrix((value,(rr,col)),shape=(len(targets),len(source))) for key,value in [('W',w),('Wd',w*dd),('Ws',w*score)]}
  assert np.allclose(np.asarray(answer[name]['W'].sum(axis=1)).ravel(),1)
 return answer

def load_data():
 paths=[CIS/'cis_split_input.h5',CIS/'cis_directed_matched_control_edges.csv.gz',SELECT]
 d={'source_signatures':[signature(p) for p in paths]}
 with h5py.File(paths[0]) as f:
  for key in ['source_indices','target_local_indices','stages']:d[key]=f[key][:]
  d['all_peak_ids']=f['peak_ids'].asstr()[:];d['celltypes']=f['celltypes'].asstr()[:]
  d['observed']={h:f['observed_cpm_'+h][:].astype(float) for h in ('a','b')}
 with h5py.File(SELECT) as f:
  assert np.array_equal(f['source_indices'][:],d['source_indices'][d['target_local_indices']])
  d['selection_observed']={h:f['observed_cpm_'+h][:].astype(float) for h in ('a','b')}
  d['selection_counts']={h:f['raw_counts_'+h][:] for h in ('a','b')}
  d['selection_ncells']=f['n_cells_a'][:]+f['n_cells_b'][:]
 d['peak_ids']=d['all_peak_ids'][d['target_local_indices']];d['chromosome']=np.array([s.split('-')[0] for s in d['peak_ids']]);d['fold']=np.array([(int(s)-1)%5 for s in d['chromosome']]);d['inner_fold']=np.array([(int(s)-1)%3 for s in d['chromosome']])
 d['graphs']=build_static_graphs(d['source_indices'],d['target_local_indices'],pd.read_csv(paths[1]))
 d['raw']=build_raw_blocks(d)
 return d

def build_raw_blocks(d,include_generic=True):
 """Each raw block uses ONLY source observations at or before its origin.

A generic graph is recalculated at the current stage using all 11,334 available
peaks; no future-stage profile enters candidate selection or weights.
"""
 raw={};target=d['target_local_indices'];dt=np.diff(d['stages'])
 for half,array in d['observed'].items():
  raw[half]={}
  for t in range(1,5):
   x=array[:,:,t];r=(array[:,:,t]-array[:,:,t-1])/dt[t-1];lx=np.log1p(x);xt=x[target];rt=r[target];lt=lx[target]
   block={'own':np.stack([lt,rt],axis=-1)}
   for name,gs in d['graphs'].items():
    W=gs['W'];wr=W@r
    block[name]=np.stack([W@x-xt,wr,lt*wr,W@(lx*r),gs['Wd']@r,gs['Ws']@r],axis=-1)
   if include_generic:
    nn=NearestNeighbors(n_neighbors=31,algorithm='kd_tree',n_jobs=2).fit(lx)
    ds,ix=nn.kneighbors(lt);neighbors=np.empty((len(target),30),int);distance=np.empty((len(target),30))
    for i,j in enumerate(target):
     keep=ix[i]!=j;neighbors[i]=ix[i,keep][:30];distance[i]=ds[i,keep][:30]
    w=np.exp(-np.divide(distance,distance[:,-1:],out=np.zeros_like(distance),where=distance[:,-1:]>0));w/=w.sum(axis=1)[:,None]
    wx=np.einsum('nk,nkc->nc',w,x[neighbors]);wr=np.einsum('nk,nkc->nc',w,r[neighbors]);block['generic']=np.stack([wx-xt,wr],axis=-1)
    block['generic_neighbor_indices']=neighbors;block['generic_weights']=w
   raw[half][t]=block
  log('Prepared current/past-only graph features for half '+half)
 return raw

def gate_scaler(own):
 x=own[:,:,0];return dict(mean=float(x.mean()),scale=max(float(x.std()),1e-12))

def feature_block(raw,model,gate):
 """Return own[N,C,2] and at most six GLOBAL shared features[N,C,P]."""
 own=raw['own'];mu,sd=gate['mean'],gate['scale'];z=(own[:,:,0]-mu)/sd;r=own[:,:,1]
 if model=='own_history':extra=np.empty((*own.shape[:2],0))
 elif model=='own_nonlinear':extra=np.stack([z*z,np.expm1(own[:,:,0]),z*r,z*z*r,r*r/(1+np.abs(r)),np.log1p(np.abs(r))],axis=-1)
 elif model in ['constant_cis','generic_smoothing']:extra=raw['cis' if model=='constant_cis' else 'generic'][:,:,:2].copy()
 else:
  extra=raw['cis' if model=='flexible_cis' else 'distance'].copy()
  extra[:,:,2]=(extra[:,:,2]-mu*extra[:,:,1])/sd
  extra[:,:,3]=(extra[:,:,3]-mu*extra[:,:,1])/sd
 return own,extra

def fit_shared_ridge(own,extra,y,penalty):
 """Fit mean loss + λ(mean_CT ||own_CT||² + ||shared||²).

All centering/scaling is learned from training rows only. Shared features use
per-CT centering but ONE common scale per feature; resulting slopes remain shared
across CTs. CT intercepts are unpenalized. Schur complement avoids a dense design.
"""
 n,C,P=extra.shape;assert own.shape==(n,C,2) and y.shape==(n,C)
 xm=own.mean(axis=0);xs=own.std(axis=0);xs=np.where(xs>1e-12,xs,1.)
 gm=extra.mean(axis=0);gc=extra-gm;gs=np.sqrt(np.mean(gc*gc,axis=(0,1)));gs=np.where(gs>1e-12,gs,1.)
 x=(own-xm)/xs;g=gc/gs;ym=y.mean(axis=0);yy=y-ym
 A=np.einsum('ncp,ncq->cpq',x,x)/n+penalty*np.eye(2)[None]
 a=np.einsum('ncp,nc->cp',x,yy)/n
 if P:
  B=np.einsum('ncp,ncq->cpq',x,g)/n
  S=np.einsum('ncp,ncq->pq',g,g)/(n*C)+penalty*np.eye(P)
  s=np.einsum('ncp,nc->p',g,yy)/(n*C)
  ai=np.linalg.solve(A,a[...,None])[...,0];AB=np.linalg.solve(A,B)
  theta=np.linalg.solve(S-np.einsum('cpk,cpl->kl',B,AB)/C,s-np.einsum('cpk,cp->k',B,ai)/C)
  beta=np.linalg.solve(A,(a-np.einsum('cpk,k->cp',B,theta))[...,None])[...,0]
 else:theta=np.empty(0);beta=np.linalg.solve(A,a[...,None])[...,0]
 return dict(own_mean=xm,own_scale=xs,shared_mean=gm,shared_scale=gs,target_mean=ym,own_beta=beta,shared_beta=theta,penalty=float(penalty),n_train_rows=n)

def predict_shared(fit,own,extra):
 return fit['target_mean'][None]+np.einsum('ncp,cp->nc',(own-fit['own_mean'])/fit['own_scale'],fit['own_beta'])+np.einsum('ncp,p->nc',(extra-fit['shared_mean'])/fit['shared_scale'],fit['shared_beta'])

def selector(d,half,t,trainmask):
 return fit_origin_selection(d['selection_observed'][half],d['selection_counts'][half],d['stages'],d['selection_ncells'],float(d['stages'][t]),trainmask)

def get_training(d,half,intervals,train,model,gate=None):
 own=np.concatenate([d['raw'][half][t]['own'][train] for t in intervals])
 if gate is None:gate=gate_scaler(own)
 feature=[feature_block(d['raw'][half][t],model,gate)[1][train] for t in intervals]
 extra=np.concatenate(feature);other='b' if half=='a' else 'a';a=d['observed'][other][d['target_local_indices']];dt=np.diff(d['stages'])
 y=np.concatenate([(a[train,:,t+1]-a[train,:,t])/dt[t] for t in intervals])
 return own,extra,y,gate

def tune_model(d,half,origin,outertrain,model):
 """Inner chronological validation when possible, else past-interval chromosome CV."""
 latest=origin-1;trainints=list(range(1,latest)) or [latest]
 losses=np.zeros((len(LAMBDAS),len(SHRINKAGES)));count=0;details=[]
 for inner in range(3):
  innertr=outertrain&(d['inner_fold']!=inner);innerva=outertrain&(d['inner_fold']==inner)
  selection=selector(d,half,latest,innertr);train=np.flatnonzero(innertr&selection['eligible']);val=np.flatnonzero(innerva&selection['eligible'])
  assert not set(d['chromosome'][train])&set(d['chromosome'][val]);assert len(train)>0 and len(val)>0
  own,extra,y,gate=get_training(d,half,trainints,train,model)
  testown,testextra=feature_block(d['raw'][half][latest],model,gate);testown=testown[val];testextra=testextra[val]
  other='b' if half=='a' else 'a';a=d['observed'][other][d['target_local_indices']];truth=(a[val,:,latest+1]-a[val,:,latest])/(d['stages'][latest+1]-d['stages'][latest])
  for li,penalty in enumerate(LAMBDAS):
   fit=fit_shared_ridge(own,extra,y,penalty);pred=predict_shared(fit,testown,testextra)
   for gi,shrink in enumerate(SHRINKAGES):losses[li,gi]+=np.sum((shrink*pred-truth)**2)
  count+=truth.size
  details.append(dict(inner_fold=inner,n_train_peaks=len(train),n_validation_peaks=len(val),training_intervals=trainints,validation_interval=latest,train_chromosomes=sorted(set(d['chromosome'][train]),key=int),validation_chromosomes=sorted(set(d['chromosome'][val]),key=int),gate=gate))
 # Fixed deterministic ties favor zero/smaller forecasts, then stronger penalty.
 candidates=[(float(losses[li,gi]),float(shrink),-float(penalty),li,gi) for li,penalty in enumerate(LAMBDAS) for gi,shrink in enumerate(SHRINKAGES)]
 best=min(candidates);li,gi=best[-2:]
 return dict(penalty=float(LAMBDAS[li]),shrinkage=float(SHRINKAGES[gi]),validation_loss=losses,validation_n_values=count,inner_details=details,inner_temporal_holdout=latest>1)

def run_outer_fold(d,half,origin,fold,models=MODELS):
 """Auditable pure fold operation. Does not write outputs or use test outcomes."""
 trainmask=d['fold']!=fold;test=np.flatnonzero(d['fold']==fold);selection=selector(d,half,origin,trainmask);train=np.flatnonzero(trainmask&selection['eligible']);trainints=list(range(1,origin));answer={}
 assert not set(d['chromosome'][train])&set(d['chromosome'][test])
 for model in models:
  tuning=tune_model(d,half,origin,trainmask,model)
  own,extra,y,gate=get_training(d,half,trainints,train,model);fit=fit_shared_ridge(own,extra,y,tuning['penalty']);testown,testextra=feature_block(d['raw'][half][origin],model,gate)
  raw=predict_shared(fit,testown[test],testextra[test]);answer[model]=dict(prediction=tuning['shrinkage']*raw,unshrunk_prediction=raw,fit=fit,gate=gate,tuning=tuning)
 return dict(test_indices=test,train_indices=train,selection=selection,models=answer,training_intervals=trainints)

def write_fit(group,fit,gate):
 for k,v in fit.items():
  if np.isscalar(v):group.attrs[k]=v
  else:group.create_dataset(k,data=v)
 for k,v in gate.items():group.attrs['gate_'+k]=v

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 design=dict(seed=SEED,threshold=.25,n_shared_graph_coefficients=6,shared_across='All peaks AND all21 response celltypes',own_baseline='21CT-specific intercepts plus current log1p(CPM) and previous linear rate; no perpeak or peredge free coefficients',models=list(MODELS),graph_features=FEATURE_NAMES['flexible_cis'],edge_operator='For lagged rates: W_ij * (theta0 + theta_receiver*z_i + theta_partner*z_j + theta_distance*log1p(distance)/log500001 + theta_score*CiceroScore); z standardized by training target currentlogCPM; signed weights permitted. Separate theta_current*(Wx-x).',feature_scaling='All gates and scalers separately refitted within EACH inner/outer train split. Shared features perCT centered with one common scale; ownfeatures perCTscaled.',penalties=list(LAMBDAS),forecast_shrinkages=list(SHRINKAGES),penalty_objective='Mean squared residual across rows andCT + lambda*(mean overCT squared own slopes + squared shared slopes); unpenalized CTintercepts; global shrinkage multiplies entire final prediction including intercept.',outer_origins_hpf=[14,16,19],outer_chromosome_folds=5,inner_chromosome_folds=3,tuning='Latest completed interval is inner validation; all earlier history-available intervals train. For14hpf onlyone history-available interval exists: samepast12->14 interval crosschromosomeCV fallback. No outertest outcomes tune.',tie_rule='Minimum inner squared-rate loss; exact ties prefer smaller globalshrink then largerpenalty.',primary_target='Other-cell-half (next observedCPM-current observedCPM)/actualdt',response_panel='Existing21completeCTs; selector uses allreliablepast32CTxstage groups with pooled>=20cells',selection='Past source-half count>=20 eligibility; Gini and mean-adjustedtotalvariance top20 percent trainedonlyoutertrainingchroms. Fitonceonalleligible, scoreallconnected/allpasteligible/Gini/HVP; retainallneighbors.',graph_fixed_learned_all_cells=True,generic_degree_caveat='Generic smoothing uses30partners ratherthan Cicero median2; it is an additional smoothing baseline, not degree-matched. Distance control retainsoriginaledge draws/weights.',generic_graph='Current-stage log1p(CPM)21CT exactk30 among all11334availablepeaks, exclude self, exponential distanceweights; recomputedper sourcehalf/origin. Features currentmean residual andpreviousmeanrate.',future_state_error='Secondary pooled rate squared error multiplied by dt^2; uses other-half current as common evaluation anchor, not source-only absolute-level forecast.',no_test_model_selection=True)
 dump(OUT/'design_prespecified.json',design)
 d=load_data();graph_diagnostics=[]
 for graph_name,gs in d['graphs'].items():
  W=gs['W'];degree=np.diff(W.indptr);graph_diagnostics.append(dict(graph=graph_name,n_targets=W.shape[0],n_retained_neighbor_nodes=W.shape[1],distinct_edges=W.nnz,minimum_distinct_degree=int(degree.min()),median_distinct_degree=float(np.median(degree)),maximum_distinct_degree=int(degree.max()),row_mass_one=bool(np.allclose(np.asarray(W.sum(axis=1)).ravel(),1))))
 graph_diagnostics.append(dict(graph='generic_current_state',n_targets=len(d['peak_ids']),n_retained_neighbor_nodes=len(d['source_indices']),minimum_distinct_degree=30,median_distinct_degree=30,maximum_distinct_degree=30,scope='Different smoothing degree; distance graph is the matched-edge-draw comparison'))
 dump(OUT/'graph_degree_audit.json',graph_diagnostics)
 n=len(d['peak_ids']);C=len(d['celltypes']);models=['persistence']+list(MODELS);cohorts=['all_connected','all_eligible','gini_top20','total_variance_top20'];rows=[];blocks=[];ctrows=[];tunerows=[];coeffrows=[];foldrows=[];selectionrows=[]
 temp=OUT/'predictions.pending.h5'
 with h5py.File(temp,'w') as h:
  for key,v in [('peak_ids',d['peak_ids']),('celltypes',d['celltypes']),('chromosome',d['chromosome'])]:h.create_dataset(key,data=np.asarray(v,object),dtype=h5py.string_dtype())
  for key,v in [('source_indices',d['source_indices'][d['target_local_indices']]),('fold',d['fold']),('stages',d['stages']),('origin_indices',np.array(ORIGINS))]:h.create_dataset(key,data=v)
  h.attrs['axes']='Predictions,targets: peak,celltype,origin; masks: peak,origin; half_a means featuresA andoutcomeB'
  for half in ('a','b'):
   other='b' if half=='a' else 'a';a=d['observed'][other][d['target_local_indices']];truth=np.stack([(a[:,:,t+1]-a[:,:,t])/(d['stages'][t+1]-d['stages'][t]) for t in ORIGINS],axis=2)
   pred={m:np.zeros((n,C,3)) if m=='persistence' else np.full((n,C,3),np.nan) for m in models};rawpred={m:np.zeros((n,C,3)) if m=='persistence' else np.full((n,C,3),np.nan) for m in models}
   masks={c:np.ones((n,3),bool) if c=='all_connected' else np.zeros((n,3),bool) for c in cohorts}
   for ti,t in enumerate(ORIGINS):
    for fold in range(5):
     start=time.time();result=run_outer_fold(d,half,t,fold);test=result['test_indices'];train=result['train_indices'];sel=result['selection']
     for cohort,key in [('all_eligible','all'),('gini_top20','gini_top20'),('total_variance_top20','total_variance_top20')]:masks[cohort][test,ti]=sel['masks'][key][test]
     foldpath=f'fits/half_{half}/origin_{t}/fold_{fold}';fg=h.require_group(foldpath);fg.create_dataset('train_indices',data=train);fg.create_dataset('test_indices',data=test)
     dump(OUT/f'selection_half_{half}_origin_{t}_fold_{fold}.json',sel['audit'])
     for ii in test:
      selectionrows.append(dict(half=half,origin_index=t,origin_hpf=d['stages'][t],fold=fold,source_index=int(d['source_indices'][d['target_local_indices'][ii]]),peak_id=d['peak_ids'][ii],eligible=bool(sel['eligible'][ii]),gini_selected=bool(sel['masks']['gini_top20'][ii]),total_variance_selected=bool(sel['masks']['total_variance_top20'][ii]),**{k:float(v[ii]) for k,v in sel['scores'].items()}))
     foldrows.append(dict(half=half,origin=t,fold=fold,n_train=len(train),n_test=len(test),train_chromosomes=sorted(set(d['chromosome'][train]),key=int),test_chromosomes=sorted(set(d['chromosome'][test]),key=int),training_intervals=result['training_intervals'],test_origin_hpf=float(d['stages'][t]),test_outcome_hpf=float(d['stages'][t+1])))
     for model,item in result['models'].items():
      pred[model][test,:,ti]=item['prediction'];rawpred[model][test,:,ti]=item['unshrunk_prediction'];mg=fg.require_group(model);write_fit(mg,item['fit'],item['gate']);tune=item['tuning'];mg.attrs['shrinkage']=tune['shrinkage'];mg.create_dataset('inner_loss',data=tune['validation_loss']);dump(OUT/f'tuning_half_{half}_origin_{t}_fold_{fold}_{model}.json',tune)
      for li,pen in enumerate(LAMBDAS):
       for si,shr in enumerate(SHRINKAGES):tunerows.append(dict(half=half,origin=t,fold=fold,model=model,penalty=pen,shrinkage=shr,validation_squared_rate_error=tune['validation_loss'][li,si],n_validation_values=tune['validation_n_values'],selected=pen==tune['penalty'] and shr==tune['shrinkage'],inner_temporal_holdout=tune['inner_temporal_holdout']))
      fit=item['fit']
      for j,fn in enumerate(FEATURE_NAMES[model]):coeffrows.append(dict(half=half,origin=t,fold=fold,model=model,feature=fn,shared_across_all_celltypes=True,standardized_coefficient=fit['shared_beta'][j],unstandardized_coefficient=fit['shared_beta'][j]/fit['shared_scale'][j],final_shrinkage=tune['shrinkage'],penalty=tune['penalty'],gate_mean=item['gate']['mean'],gate_scale=item['gate']['scale']))
     log(f'{half} origin{d["stages"][t]:g} fold{fold}: {time.time()-start:.1f}s')
   hg=h.require_group('half_'+half);hg.create_dataset('target_rate',data=truth.astype('f4'),compression='gzip')
   for c,v in masks.items():hg.require_group('cohorts').create_dataset(c,data=v,compression='gzip')
   for m in models:
    assert np.isfinite(pred[m]).all();hg.require_group('predictions').create_dataset(m,data=pred[m].astype('f4'),compression='gzip');hg.require_group('unshrunk_predictions').create_dataset(m,data=rawpred[m].astype('f4'),compression='gzip')
   for ti,t in enumerate(ORIGINS):
    dt=d['stages'][t+1]-d['stages'][t]
    for cohort in cohorts:
     use=masks[cohort][:,ti];ix=np.flatnonzero(use)
     for m in models:
      err=(pred[m][:,:,ti]-truth[:,:,ti])**2;baseline=(pred['own_history'][:,:,ti]-truth[:,:,ti])**2;zero=truth[:,:,ti]**2
      meta=dict(half=half,origin=t,start_hpf=d['stages'][t],end_hpf=d['stages'][t+1],cohort=cohort,model=m)
      rows.append({**meta,'n_peaks':len(ix),'n_values':len(ix)*C,'rate_sse':float(err[ix].sum()),'state_change_sse':float(err[ix].sum()*dt*dt),'own_history_sse':float(baseline[ix].sum()),'persistence_sse':float(zero[ix].sum()),'improvement_vs_own_history':float(1-err[ix].sum()/baseline[ix].sum()),'improvement_vs_persistence':float(1-err[ix].sum()/zero[ix].sum())})
      for chrom in np.unique(d['chromosome']):
       jj=np.flatnonzero(use&(d['chromosome']==chrom));blocks.append({**meta,'chromosome':chrom,'n_peaks':len(jj),'rate_sse':float(err[jj].sum()),'state_change_sse':float(err[jj].sum()*dt*dt)})
      for c,name in enumerate(d['celltypes']):ctrows.append({**meta,'celltype':name,'n_peaks':len(ix),'rate_sse':float(err[ix,c].sum()),'state_change_sse':float(err[ix,c].sum()*dt*dt)})
 temp.replace(OUT/'predictions.h5')
 for name,data in [('metrics',rows),('chromosome_losses',blocks),('celltype_losses',ctrows),('tuning_grid',tunerows),('shared_coefficients',coeffrows),('cohort_scores',selectionrows)]:pd.DataFrame(data).to_csv(OUT/(name+'.csv.gz' if name=='cohort_scores' else name+'.csv'),index=False)
 dump(OUT/'outer_folds.json',foldrows);pooled=summarize(pd.DataFrame(blocks));pooled.to_csv(OUT/'pooled_metrics.csv',index=False);plots(pooled,pd.DataFrame(tunerows));summary(pooled)
 paths=[Path(x['path']) for x in d['source_signatures']];assert d['source_signatures']==[signature(p) for p in paths]
 dump(OUT/'validation.json',dict(status='passed',source_files_unchanged=True,source_signatures=d['source_signatures'],n_targets=n,n_response_celltypes=C,shared_graph_coefficients=6,all_predictions_finite=True,source_halves_independent_of_outcome_halves=True,outer_train_test_chromosomes_disjoint=True,no_future_features_or_tuning=True,selector_all_reliable_past_pseudobulks=True,all_graph_neighbors_retained=True,n_outer_folds=30,n_models_fitted_per_fold=len(MODELS),graph_learned_all_cells_and_stages=True,global_forecast_shrinkage_includes_zero=True))
 log('Shared-weight forecast pilot complete')

def summarize(blocks):
 rows=[];rng=np.random.default_rng(SEED)
 for cohort in blocks.cohort.unique():
  for interval in ['pooled']+list(ORIGINS):
   q=blocks[(blocks.cohort==cohort)&((blocks.origin==interval) if interval!='pooled' else True)]
   agg=q.groupby(['model','chromosome'])[['rate_sse','state_change_sse']].sum();chrom=sorted(q.chromosome.unique(),key=int);draw=rng.integers(len(chrom),size=(2000,len(chrom)))
   for error in ['rate_sse','state_change_sse']:
    losses={m:agg.loc[m].reindex(chrom)[error].to_numpy() for m in ['persistence']+list(MODELS)}
    for m,v in losses.items():
     item=dict(cohort=cohort,interval=str(interval),error=error,model=m,sse=float(v.sum()),n_chromosome_blocks=len(chrom))
     for baseline in ['persistence','own_history','own_nonlinear','constant_cis','distance_flexible','generic_smoothing']:
      b=losses[baseline];score=1-v.sum()/b.sum();lo,hi=np.quantile(1-v[draw].sum(axis=1)/b[draw].sum(axis=1),[.025,.975]);item['vs_'+baseline]=score;item['vs_'+baseline+'_low']=lo;item['vs_'+baseline+'_high']=hi
     rows.append(item)
 return pd.DataFrame(rows)

def plots(pooled,tuning):
 colors={'own_history':'#777777','own_nonlinear':'#8f6b9b','constant_cis':'#78a4a4','flexible_cis':'#087e79','distance_flexible':'#c58b35','generic_smoothing':'#527fa8'}
 fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
 for ax,cohort in zip(axes,['all_eligible','gini_top20','total_variance_top20']):
  q=pooled[(pooled.cohort==cohort)&(pooled.error=='rate_sse')&(pooled.interval!='pooled')]
  for m in MODELS:
   r=q[q.model==m].sort_values('interval');xx=np.arange(3);ax.plot(xx,100*r.vs_persistence,'-o',color=colors[m],label=m.replace('_',' '),ms=4);ax.fill_between(xx,100*r.vs_persistence_low,100*r.vs_persistence_high,color=colors[m],alpha=.08)
  ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['14→16','16→19','19→24']);ax.set_title(cohort.replace('_',' '));ax.set_xlabel('Forecast interval (hpf)');ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Squared-change error reduction vs persistence (%)');axes[0].legend(fontsize=7);fig.suptitle('Shared edge weights: held-out temporal and chromosome forecasts',fontsize=13);fig.text(.5,.015,'Models use past/current source-half observations; outcomes use the independent cell half. Shrinkage and penalty chosen on past inner validation.\nBands: chromosome-block sampling bootstrap; selected cohorts use training-calibrated past-only cutoffs. Fixed Cicero graph learned from all stages.\nNear-zero forecasts at19→24 reflect learned shrinkage toward persistence, not inferred biological stability.',ha='center',fontsize=8);fig.subplots_adjust(top=.82,bottom=.25,wspace=.18);savefig(fig,'01_shared_weight_forecast')
 fig,axes=plt.subplots(1,2,figsize=(10,4.8))
 for ax,error in zip(axes,['rate_sse','state_change_sse']):
  q=pooled[(pooled.model=='flexible_cis')&(pooled.interval=='pooled')&(pooled.error==error)]
  for j,c in enumerate(['all_eligible','gini_top20','total_variance_top20']):
   r=q[q.cohort==c].iloc[0];names=['own_history','own_nonlinear','constant_cis','distance_flexible','generic_smoothing'];v=np.array([r['vs_'+n] for n in names])*100;lo=np.array([r['vs_'+n+'_low'] for n in names])*100;hi=np.array([r['vs_'+n+'_high'] for n in names])*100;yy=np.arange(5)+(j-1)*.22;ax.errorbar(v,yy,xerr=[np.maximum(v-lo,0),np.maximum(hi-v,0)],fmt='o',ms=4,capsize=2,label=c.replace('_',' '))
  ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(5),[n.replace('_',' ') for n in names]);ax.invert_yaxis();ax.set_xlabel('Flexible Cis error reduction vs comparator (%)');ax.set_title('CPM/hour rate error' if error=='rate_sse' else 'Interval accessibility-change error');ax.spines[['top','right']].set_visible(False)
 axes[0].legend(fontsize=8);fig.suptitle('Does partner-dependent weighting add information?',fontsize=13);fig.text(.5,.01,'Same models trained on all eligible targets; subsets are evaluation strata. Interval-change error weights rate error by Δt².\nA future-state prediction anchored to other-half current accessibility has this same error; this is not an absolute-level source-only forecast.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.1,1,.94]);savefig(fig,'02_shared_weight_comparators')
 chosen=tuning[tuning.selected];fig,axes=plt.subplots(1,2,figsize=(10,4.4))
 for ax,key,values in [(axes[0],'shrinkage',SHRINKAGES),(axes[1],'penalty',LAMBDAS)]:
  count=pd.crosstab(chosen.model,chosen[key]).reindex(index=MODELS,columns=values,fill_value=0);im=ax.imshow(count.to_numpy(),aspect='auto',cmap='Blues',vmin=0,vmax=30);ax.set_yticks(range(len(MODELS)),[m.replace('_',' ') for m in MODELS]);ax.set_xticks(range(len(values)),list(values));ax.set_xlabel(key);ax.set_title('Chosen in 30 outer fits');
  for i in range(len(MODELS)):
   for j in range(len(values)):ax.text(j,i,str(count.iloc[i,j]),ha='center',va='center',fontsize=9)
 fig.suptitle('Past-only model selection can shrink the forecast to zero',fontsize=13);fig.tight_layout(rect=[0,0,1,.92]);savefig(fig,'03_selected_shrinkage_and_penalty')

def savefig(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def summary(pooled):
 q=pooled[(pooled.interval=='pooled')&(pooled.error=='rate_sse')]
 text=['SHARED SIGNED CICERO FORECAST PILOT','', 'All settings were fixed before this pilot; nested validation, not outer test outcomes, selected shrinkage/penalty. Graph estimation itself used all cells/stages, so this remains conditional validation.','', 'Main result: no consistent advantage of shared Cicero weights. Pooled flexible-Cis squared-rate error is6.10% higher than persistence on all eligible targets,27.55% higher in Gini-selected targets, and19.13% higher in mean-adjusted total-variance targets. The connected selected cohorts are evaluation strata of the same all-eligible model fit.', '', 'Temporal nuance: the total-variance cohort at14->16hpf has2.43% lower error than persistence (chromosome bootstrap0.63 to4.83%), but only0.29% incremental reduction versus own history (range-0.25 to0.84%) and0.16% versus equal-flex own nonlinear (range-0.60 to1.22%). This supports limited early-window predictability, without demonstrated additional Cis information. Failures concentrate at16->19hpf. At19->24hpf inner validation chooses zero forecasts in all10fits of most models and9/10 flexible-Cis fits; this is forecast abstention rather than biological stability.', '', 'The six shared graph coefficients are global across peaks and celltypes. CT-specific own-history slopes/intercepts are allowed; no edge-specific or peak-specific parameters are fitted. Partner current state can modify each partner previous-rate contribution separately.','',q[['cohort','model','vs_persistence','vs_own_history','vs_own_nonlinear','vs_constant_cis','vs_distance_flexible','vs_generic_smoothing']].to_string(index=False),'','Interpretation limits: cell halves are sampling partitions rather than biological replicates; bootstrap ranges describe chromosome-block sampling sensitivity. Only4525connected peaks enter this pilot. Gini/variable cohorts are evaluation strata of one all-eligible fit, not separately refitted models. Whole-course cell-count metadata define the fixed21CT response panel, while selector measurements use only each origin’s reliable past pseudobulks. Generic kNN uses only current source-half profiles, but is transductive across held-out peak identities. Predicted changes use relative CPM and do not distinguish compositional effects from changes in total accessibility.','', 'Future-level interpretation: state_change_sse is rate_sse×Δt², equivalently evaluating B_current+Δt*predicted_rate against B_next. B_current is an evaluation anchor only, never a predictor. The primary endpoint predicts change.','', 'Exact arrays, shared and own coefficients/scalers, nested choices, test/train chromosome membership, and cohort masks are saved in predictions.h5 and accompanying tables.']
 (OUT/'summary.txt').write_text('\n'.join(text)+'\n')
if __name__=='__main__':main()
