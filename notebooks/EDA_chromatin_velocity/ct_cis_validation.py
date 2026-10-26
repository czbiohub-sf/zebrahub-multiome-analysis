"""Audit and independently evaluate fixed cis-neighborhood smoothing.

The Cicero graph was learned previously from all cells. Cell-half tests therefore
assess conditional performance of that fixed graph, not fully held-out graph
learning or biological replication. Coverage is reported before selected-edge
analyses, because an isolated peak cannot benefit from cis smoothing.
"""
from pathlib import Path
import json,h5py,numpy as np,pandas as pd,time
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/ct_cis_validation_v1';SRC=HERE.parents[2]/'data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv';FITS=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'

def audit():
 OUT.mkdir(parents=True,exist_ok=True)
 with h5py.File(FITS) as f:ids=f['peak_ids'].asstr()[:]
 lookup=pd.Series(np.arange(len(ids)),index=ids);thresholds=[.05,.1,.25,.5,.8];totals={str(t):dict(raw_rows=0,aligned_rows=0) for t in thresholds};chunks=[];allchrom=set();nrows=0;nfinite=0;nonpositive=0;scoremax=-np.inf
 for d in pd.read_csv(SRC,chunksize=500000):
  nrows+=len(d);nfinite+=int(np.isfinite(d.coaccess).sum());nonpositive+=int((d.coaccess<=0).sum());scoremax=max(scoremax,float(d.coaccess.max()));allchrom.update(d.Peak1.str.split('-').str[0].unique())
  d=d[d.coaccess>=.05].copy();d['i']=d.Peak1.map(lookup);d['j']=d.Peak2.map(lookup)
  for t in thresholds:
   q=d[d.coaccess>=t];totals[str(t)]['raw_rows']+=len(q);totals[str(t)]['aligned_rows']+=int((q.i.notna()&q.j.notna()).sum())
  d=d.dropna(subset=['i','j']);d[['i','j']]=d[['i','j']].astype(int);chunks.append(d)
  print('Audited',nrows,flush=True)
 e=pd.concat(chunks,ignore_index=True);e['a']=np.minimum(e.i,e.j);e['b']=np.maximum(e.i,e.j);e=e[e.a!=e.b].groupby(['a','b'],as_index=False).coaccess.max()
 chrom=np.asarray([s.split('-')[0] for s in ids]);assert (chrom[e.a.to_numpy()]==chrom[e.b.to_numpy()]).all(),'Trans edges found'
 rows=[]
 for t in thresholds:
  q=e[e.coaccess>=t];nodes=np.unique(np.concatenate([q.a.to_numpy(),q.b.to_numpy()]));degree=np.bincount(np.concatenate([q.a.to_numpy(),q.b.to_numpy()]),minlength=len(ids));rows.append(dict(threshold=t,**totals[str(t)],undirected_edges=len(q),peaks_with_edges=len(nodes),fraction_all_peaks_with_edges=len(nodes)/len(ids),median_degree_connected=float(np.median(degree[nodes])) if len(nodes) else 0,chromosomes_connected=len(np.unique(chrom[nodes]))))
 e.to_csv(OUT/'aligned_positive_cis_edges.csv.gz',index=False);pd.DataFrame(rows).to_csv(OUT/'cis_graph_coverage.csv',index=False)
 (OUT/'cis_graph_provenance.json').write_text(json.dumps(dict(source=str(SRC),n_source_rows=nrows,n_finite_scores=nfinite,n_nonpositive_scores=nonpositive,maximum_score=scoremax,all_source_chromosomes=sorted(allchrom),n_atlas_peaks=len(ids),symmetrization='Undirected pair, maximum of duplicated reciprocal scores; diagonal excluded',cis_only_verified=True,prior_graph_uses_all_cells=True),indent=2)+'\n')
 print(pd.DataFrame(rows).to_string(index=False),flush=True)

def prepare_validation():
 import sys
 sys.path.insert(0,str(HERE));from real_input_provenance import read_column
 split=HERE/'outputs/ct_flow_noise_v1/cell_split_input.h5';raw=HERE.parents[2]/'data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad'
 with h5py.File(FITS) as f:
  ids=f['peak_ids'].asstr()[:];source_ct=f['celltypes'].asstr()[:];stages=f['stages'][:]
 edges=pd.read_csv(OUT/'aligned_positive_cis_edges.csv.gz');edges=edges[edges.coaccess>=.25].copy();targets=np.unique(np.concatenate([edges.a,edges.b])).astype(int)
 chrom=np.asarray([s.split('-')[0] for s in ids]);pos=np.asarray([(int(s.split('-')[1])+int(s.split('-')[2]))/2 for s in ids]);bins=np.array([0,10000,25000,50000,100000,250000,500001,np.inf]);rng=np.random.default_rng(20261021)
 ordered={c:np.flatnonzero(chrom==c)[np.argsort(pos[chrom==c])] for c in np.unique(chrom)}
 directed=pd.concat([edges.rename(columns={'a':'target','b':'neighbor'}),edges.rename(columns={'b':'target','a':'neighbor'})],ignore_index=True)
 neighbors={int(i):set(q.neighbor) for i,q in directed.groupby('target')};null=[];fallbacks=0
 for row in directed.itertuples():
  i=int(row.target);j=int(row.neighbor);distance=abs(pos[i]-pos[j]);bi=np.clip(np.searchsorted(bins,distance,side='right')-1,0,len(bins)-2);lo,hi=bins[bi:bi+2];pool=ordered[chrom[i]];pp=pos[pool]
  if np.isfinite(hi):
   cand=np.concatenate([pool[np.searchsorted(pp,pos[i]-hi,side='left'):np.searchsorted(pp,pos[i]-lo,side='right')],pool[np.searchsorted(pp,pos[i]+lo,side='left'):np.searchsorted(pp,pos[i]+hi,side='right')]])
  else:cand=pool[np.abs(pp-pos[i])>=lo]
  cand=np.asarray([v for v in cand if v!=i and v not in neighbors[i]],int)
  fallback=False
  if len(cand)==0:
   cand=np.asarray([v for v in pool if v!=i and v not in neighbors[i]],int);fallback=True;fallbacks+=1
  assert len(cand)>0;chosen=int(rng.choice(cand));null.append(chosen)
 directed['null_neighbor']=null;directed['distance_bp']=np.abs(pos[directed.target]-pos[directed.neighbor]);directed['null_distance_bp']=np.abs(pos[directed.target]-pos[directed.null_neighbor]);directed.to_csv(OUT/'cis_directed_matched_control_edges.csv.gz',index=False)
 needed=np.unique(np.concatenate([targets,directed.null_neighbor.to_numpy()])).astype(int);local={int(v):i for i,v in enumerate(needed)}
 with h5py.File(split) as s,h5py.File(raw) as f:
  half=s['cell_half'][:];assert np.array_equal(s['cell_ids'].asstr()[:],read_column(f['obs'],'_index'))
  allct=s['celltypes'].asstr()[:];ctidx=np.flatnonzero(s['condition_valid'][:].all(axis=1));ctmap={v:i for i,v in enumerate(allct)};tmap={'0somites':0,'5somites':1,'10somites':2,'15somites':3,'20somites':4,'30somites':5}
  group=np.asarray([ctmap[c]*6+tmap[t] for c,t in zip(read_column(f['obs'],'annotation_ML_coarse'),read_column(f['obs'],'dev_stage'))]);combo=group*2+half
  raw_index=pd.Index(read_column(f['var'],'_index')).get_indexer(ids[needed]);assert (raw_index>=0).all();counts=np.zeros((len(needed),32,6,2),np.int64);x=f['layers/counts'];ptr=x['indptr'][:];start=time.time()
  for k,i in enumerate(np.argsort(raw_index)):
   a,b=ptr[raw_index[i]:raw_index[i]+2];counts[i]=np.bincount(combo[x['indices'][a:b]],weights=x['data'][a:b],minlength=32*6*2).reshape(32,6,2).astype(np.int64)
   if (k+1)%3000==0:print('Cis count extraction',k+1,'/',len(needed),round(time.time()-start,1),'s',flush=True)
  coverage=np.stack([s['coverage_a'][:],s['coverage_b'][:]],axis=-1);cpm=counts[:,ctidx,:,:]*1e6/coverage[None,ctidx,:,:]
  # Exact checks for every overlap with the already-audited whole-peak sample.
  sampled={int(v):i for i,v in enumerate(s['source_indices'][:])};overlap=[(j,sampled[int(v)]) for j,v in enumerate(needed) if int(v) in sampled]
  for halfname,h in [('a',0),('b',1)]:
   allreference=s['raw_counts_'+halfname][:]
   for j,k in overlap:assert np.array_equal(counts[j,ctidx,:,h].T,allreference[k][:,ctidx])
  with h5py.File(OUT/'cis_split_input.h5','w') as out:
   for key,v in [('source_indices',needed),('target_source_indices',targets),('target_local_indices',np.asarray([local[int(v)] for v in targets])),('stages',stages),('celltype_indices',ctidx)]:out.create_dataset(key,data=v)
   out.create_dataset('peak_ids',data=np.asarray(ids[needed],object),dtype=h5py.string_dtype());out.create_dataset('celltypes',data=np.asarray(allct[ctidx],object),dtype=h5py.string_dtype())
   for label,h in [('a',0),('b',1)]:
    out.create_dataset('raw_counts_'+label,data=counts[:,ctidx,:,:][...,h],compression='gzip');out.create_dataset('observed_cpm_'+label,data=cpm[:,:,:,h].astype('f4'),compression='gzip');out.create_dataset('coverage_'+label,data=coverage[ctidx,:,h])
   out.attrs['axis_order']='peak,celltype,stage (matches temporal-fitting API)';out.attrs['fixed_graph_learned_all_cells']=True
 (OUT/'cis_sampling_validation.json').write_text(json.dumps(dict(status='passed',target_peaks=len(targets),all_required_peaks=len(needed),overlap_with_audited_sample=len(overlap),same_cell_partition=True,aligned_counts_match_overlap_exactly=True,control='Each directed edge draw retains its weight, chromosome, and distance bin where candidates exist; sampling with replacement preserves row weight mass and draw count, but can reduce distinct neighbor degree. Not indegree-preserving or abundance-matched.',distance_fallback_edges=fallbacks,total_directed_edges=len(directed),control_seed=20261021,scope='Selected connected peaks; graph learned on all cells, hence conditional fixed-graph performance, not fully independent graph learning'),indent=2)+'\n');print('Cis split input ready',len(needed),'peaks',flush=True)

def evaluate():
 import sys
 sys.path.insert(0,str(HERE.parents[1]/'scripts/chrom_velo'))
 from trajectory_modes import fit_trajectories
 from scipy import sparse
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 with h5py.File(OUT/'cis_split_input.h5') as f:
  ids=f['source_indices'][:];targets=f['target_local_indices'][:];times=f['stages'][:];arrays={h:f['observed_cpm_'+h][:].astype(float) for h in ['a','b']};counts={h:f['raw_counts_'+h][:] for h in ['a','b']};coverage={h:f['coverage_'+h][:] for h in ['a','b']}
 edges=pd.read_csv(OUT/'cis_directed_matched_control_edges.csv.gz');local=np.full(640830,-1,int);local[ids]=np.arange(len(ids));target_rows=np.full(len(ids),-1,int);target_rows[targets]=np.arange(len(targets));rr=target_rows[local[edges.target]];assert (rr>=0).all()
 params=dict(complexity_penalty=.2,rate_floor=.05,rate_z=3.5,lack_of_fit_threshold=4.,min_observations=4)
 fits={};supports={}
 for h in ['a','b']:
  cov=coverage[h];variance=(counts[h]+.5)*(1e6/cov[None])**2
  fit=fit_trajectories(arrays[h],times,observed=np.ones(cov.shape,bool),weights=cov,measurement_variance=variance,**params)
  fits[h]=fit['rates'];supports[h]=fit['rate_supported']&(counts[h].sum(axis=-1)>=20)[:,:,None]&(fit['lack_of_fit']<=4)[:,:,None]
 with h5py.File(OUT/'cis_fitted_rates.h5','w') as f:
  f.create_dataset('source_indices',data=ids);f.create_dataset('target_local_indices',data=targets)
  for h in ['a','b']:f.create_dataset(h+'_rates',data=fits[h],compression='gzip');f.create_dataset(h+'_support',data=supports[h],compression='gzip')
 rows=[];manual=[]
 for threshold in [.25,.5,.8]:
  good=edges.coaccess.to_numpy()>=threshold;weights=edges.coaccess.to_numpy()[good];r=rr[good];truecol=local[edges.neighbor.to_numpy()[good]];nullcol=local[edges.null_neighbor.to_numpy()[good]]
  degree=np.bincount(r,minlength=len(targets));mass=np.bincount(r,weights=weights,minlength=len(targets));selfweight=1/(1+mass)
  W=sparse.csr_matrix((np.concatenate([weights,np.ones(len(targets))]),(np.concatenate([r,np.arange(len(targets))]),np.concatenate([truecol,targets]))),shape=(len(targets),len(ids)))
  N=sparse.csr_matrix((np.concatenate([weights,np.ones(len(targets))]),(np.concatenate([r,np.arange(len(targets))]),np.concatenate([nullcol,targets]))),shape=W.shape)
  W=sparse.diags(selfweight)@W;N=sparse.diags(selfweight)@N;assert np.allclose(np.asarray(W.sum(axis=1)).ravel(),1);assert np.allclose(np.asarray(N.sum(axis=1)).ravel(),1)
  for source,target in [('a','b'),('b','a')]:
   y=np.diff(arrays[target][targets],axis=-1)/np.diff(times);v=fits[source];intrinsic=v[targets];smooth=(W@v.reshape(len(ids),-1)).reshape(intrinsic.shape);random=(N@v.reshape(len(ids),-1)).reshape(intrinsic.shape)
   for qi in np.linspace(0,len(targets)-1,17).astype(int):
    take=r==qi;expected=(v[targets[qi]]+(weights[take,None,None]*v[truecol[take]]).sum(axis=0))/(1+weights[take].sum());assert np.allclose(expected,smooth[qi],atol=1e-10)
   for alpha in [1.,.6,0.]:
    effective=alpha+(1-alpha)*selfweight
    predictions={'cis':alpha*intrinsic+(1-alpha)*smooth,'distance_control':alpha*intrinsic+(1-alpha)*random,'degree_matched_shrink':effective[:,None,None]*intrinsic+(1-effective[:,None,None])*intrinsic.mean(axis=0)}
    for name,pred in predictions.items():
     if alpha==1:assert np.array_equal(pred,intrinsic) or np.allclose(pred,intrinsic,atol=1e-15)
     for t in range(5):
      scopes={'all_connected_at_025':np.ones(len(targets),bool),'connected_at_current_threshold':degree>0,'supported_source_at_current_threshold':(degree>0)&supports[source][targets,:,t].any(axis=1)}
      for scope,mask in scopes.items():
       if not mask.any():continue
       yy=y[mask,:,t];pp=pred[mask,:,t];base=intrinsic[mask,:,t];err=np.sum((yy-pp)**2);baseerr=np.sum((yy-base)**2);den=np.linalg.norm(yy,axis=1)*np.linalg.norm(pp,axis=1);nz=den>1e-12;cs=np.divide(np.sum(yy*pp,axis=1),den,out=np.zeros(len(yy)),where=nz)
       rows.append(dict(threshold=threshold,alpha=alpha,method=name,split=source+'_to_'+target,scope=scope,interval=t,start_hpf=times[t],end_hpf=times[t+1],n_peaks=int(mask.sum()),mse_per_peak=err/mask.sum(),temporal_only_mse_per_peak=baseerr/mask.sum(),improvement_vs_temporal=1-err/baseerr if baseerr>0 else np.nan,improvement_vs_zero=1-err/np.sum(yy**2) if np.sum(yy**2)>0 else np.nan,mean_cosine=float(cs[nz].mean()) if nz.any() else np.nan,n_nonzero_vector_pairs=int(nz.sum())))
 df=pd.DataFrame(rows);df.to_csv(OUT/'cis_validation_metrics.csv',index=False)
 main=df.query('alpha==0.6 and scope=="connected_at_current_threshold"');fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
 for ax,threshold in zip(axes,[.25,.5,.8]):
  for method,color in [('cis','#1b7389'),('distance_control','#be7d33'),('degree_matched_shrink','#87718a')]:
   d=main.query('threshold==@threshold and method==@method');g=d.groupby('interval').improvement_vs_temporal;mu=g.mean();lo=g.min();hi=g.max();xx=np.arange(5)
   ax.plot(xx,100*mu,color=color,marker='o',label=method.replace('_',' '));ax.fill_between(xx,100*lo,100*hi,color=color,alpha=.15)
  n=int(main.query('threshold==@threshold').n_peaks.iloc[0]);ax.set_title(f'Cicero ≥{threshold:g} · {n:,} connected peaks');ax.set_xticks(np.arange(5),['10–12','12–14','14–16','16–19','19–24']);ax.axhline(0,color='k',lw=.7);ax.set_xlabel('Developmental interval (hpf)');ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Error reduction versus temporal fit alone (%)');axes[0].legend(fontsize=9);fig.suptitle('Does cis coupling improve independent-cell rate estimates?  α=0.6',fontsize=16)
 fig.text(.5,.015,'Source-half temporal fits predict raw rates in the other cell half; bands span the two split directions, not confidence intervals.\nGraph was learned from all cells; this is conditional fixed-graph validation. Connected peaks are a small selected part of the atlas.',ha='center',fontsize=10);fig.subplots_adjust(top=.82,bottom=.23,wspace=.18);fig.savefig(OUT/'01_cis_independent_cell_validation.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'01_cis_independent_cell_validation.pdf',bbox_inches='tight');plt.close(fig)
 cov=pd.read_csv(OUT/'cis_graph_coverage.csv');fig,ax=plt.subplots(figsize=(9,4));ax.bar(cov.threshold.astype(str),100*cov.fraction_all_peaks_with_edges,color='#277e91');ax.set_ylabel('Atlas peaks with ≥1 positive cis neighbor (%)');ax.set_xlabel('Cicero score cutoff');ax.set_title('Available cis graph reaches a small fraction of the peak atlas')
 for i,row in enumerate(cov.itertuples()):ax.text(i,100*row.fraction_all_peaks_with_edges+.04,f'{row.peaks_with_edges:,} peaks',ha='center',fontsize=9)
 fig.tight_layout();fig.savefig(OUT/'00_cis_graph_coverage.png',dpi=180);fig.savefig(OUT/'00_cis_graph_coverage.pdf');plt.close(fig)
 report=dict(status='passed',fitting_parameters=params,actual_time_differences=np.diff(times).tolist(),row_normalization_verified=True,manual_smoothing_checks=17*3*2,alpha1_recovers_intrinsic=True,independent_cell_partition=True,n_targets=int(len(targets)),scopes='Connected source graph at0.25, threshold-connected subsets and independently source-supported strata are reported separately.',limitations=['Cicero graph learned using all cells; not independent graph training.','Random control matches edge-draw count, row weight mass, chromosome and distance bins; replacement may reduce distinct degree. Not abundance or indegree matched.','Selected connected peaks, not a global atlas result.','Cell halves measure sampling reproducibility, not biological replication.'])
 (OUT/'cis_evaluation_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(main.groupby(['threshold','method']).improvement_vs_temporal.mean().to_string(),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','prepare','evaluate'],default='audit');a=p.parse_args();{'audit':audit,'prepare':prepare_validation,'evaluate':evaluate}[a.stage]()
