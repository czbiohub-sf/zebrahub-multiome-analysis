#!/usr/bin/env python3
"""Fixed-query sensitivity of retrospective chromatin correspondence arrows.

The reference enters diagnostics only, never candidate selection or probabilities.
All query peaks are a fixed random sample; candidate targets span the full atlas.
"""
from pathlib import Path
import argparse, json, os, sys, time
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_velocity_flow_sweep_v1'
SOURCE=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'
ATLAS=HERE/'outputs/ct_peak_atlas_v1/full_ct_peak_atlas.h5'
PREV=HERE/'outputs/ct_velocity_reprojection_v1'
SEED=20261022
KS=[10,30,60,100]
BETAS=[0.,.25,.5]
NQUERY=20000

def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def stat(p):
 s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def dump(p,d):
 def cv(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  raise TypeError(type(x).__name__)
 Path(p).write_text(json.dumps(d,indent=2,default=cv,allow_nan=False)+'\n')
def unit(a):
 n=np.linalg.norm(a,axis=-1,keepdims=True)
 return np.divide(a,n,out=np.zeros_like(a,dtype=float),where=n>1e-12)
def graph():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 with h5py.File(ATLAS,'r') as f:x=f['pca'][:]
 with h5py.File(SOURCE,'r') as f:broad=f['broad_reference'][:].astype(bool)
 n=len(x);rng=np.random.default_rng(SEED);q=np.sort(rng.choice(n,NQUERY,replace=False))
 oldidx=np.load(PREV/'new_graph_indices.npy',mmap_mode='r')
 olddist=np.load(PREV/'new_graph_distances.npy',mmap_mode='r')
 idx=np.empty((len(q),100),np.int32);dist=np.empty((len(q),100),np.float32)
 def block(begin):
  end=min(begin+128,len(q))
  for ii in range(begin,end):
   src=q[ii];direct=oldidx[src]
   pool=np.unique(np.concatenate([direct,oldidx[direct].ravel()]))
   pool=pool[pool!=src]
   if len(pool)<100:
    pool=np.unique(np.concatenate([pool,oldidx[pool].ravel()]))
    pool=pool[pool!=src]
   if len(pool)<100:pool=np.delete(np.arange(n,dtype=np.int32),src)
   dd=np.sqrt(np.sum((x[pool].astype(float)-x[src].astype(float))**2,axis=1))
   best=np.argpartition(dd,99)[:100];best=best[np.argsort(dd[best],kind='stable')]
   for refinement in range(3):
    search_count=min(300,len(pool))
    search_best=np.argpartition(dd,search_count-1)[:search_count]
    nearest=pool[search_best]
    pool=np.unique(np.concatenate([nearest,oldidx[nearest].ravel()]))
    pool=pool[pool!=src]
    dd=np.sqrt(np.sum((x[pool].astype(float)-x[src].astype(float))**2,axis=1))
    best=np.argpartition(dd,99)[:100];best=best[np.argsort(dd[best],kind='stable')]
   idx[ii]=pool[best];dist[ii]=dd[best]
  return end
 with ThreadPoolExecutor(max_workers=8) as ex:
  jobs=[ex.submit(block,b) for b in range(0,len(q),128)]
  for count,fut in enumerate(as_completed(jobs),1):
   fut.result()
   if count%40==0:log(f'Expanded candidate blocks {count}/{len(jobs)}')
 # k10/30 must reproduce the released graph exactly. Save those separately;
 # expanded k60/100 are an independently nested pair from the two-hop pool.
 np.savez_compressed(OUT/'query_neighbors.npz',source_indices=q,indices100=idx,distances100=dist,
                     indices30=np.asarray(oldidx[q]),distances30=np.asarray(olddist[q]))
 auditq=np.sort(rng.choice(len(q),96,replace=False));recalls={str(k):[] for k in KS}
 exact=np.empty((len(auditq),100),np.int32)
 for begin in range(0,len(auditq),8):
  qq=auditq[begin:begin+8];ds=cdist(x[q[qq]].astype(float),x.astype(float),'sqeuclidean')
  ds[np.arange(len(qq)),q[qq]]=np.inf
  nn=np.argpartition(ds,99,axis=1)[:,:100]
  nn=np.take_along_axis(nn,np.argsort(np.take_along_axis(ds,nn,axis=1),axis=1),axis=1)
  exact[begin:begin+len(qq)]=nn
  for row,qi in enumerate(qq):
   for k in KS:
    got=np.asarray(oldidx[q[qi],:k]) if k<=30 else idx[qi,:k]
    recalls[str(k)].append(len(set(got).intersection(nn[row,:k]))/k)
 exact_dist=np.sqrt(np.sum((x[exact].astype(float)-x[q[auditq],None,:].astype(float))**2,axis=2))
 stretch={}
 for kk in KS:
  candidate=olddist[q[auditq],kk-1] if kk<=30 else dist[auditq,kk-1]
  ratios=candidate/np.maximum(exact_dist[:,kk-1],1e-12)
  stretch[str(kk)]={name:float(value) for name,value in zip(['median','p95','max'],np.quantile(ratios,[.5,.95,1]))}
 log('Neighbor audit '+str({k:float(np.mean(v)) for k,v in recalls.items()}))
 g=sparse.load_npz(PREV/'new_graph_csr.npz').tocsr();g.data=np.ones(g.nnz,np.float32)
 log('Compute fixed undirected k30 multi-source reference graph hops')
 hops=dijkstra(g,directed=False,indices=np.flatnonzero(broad),unweighted=True,min_only=True)
 np.save(OUT/'full_reference_graph_hops.npy',hops.astype(np.float32))
 np.savez_compressed(OUT/'neighbor_audit.npz',query_sample_indices=auditq,source_indices=q[auditq],exact_neighbor_indices=exact)
 manifest={'seed':SEED,'n_full_peaks':n,'n_queries':len(q),'sample':'Uniform without replacement; fixed before rate or root inspection',
  'k':KS,'rate_neighbor_blend':BETAS,'k10_k30_candidates':'Released exact-refined full-atlas k30 neighbors, truncated',
  'k60_k100_candidates':'Direct/second-hop released k30 neighbors, then three best-first expansions through current nearest300; exact float64 PCA distances and reranking; small pools expand further or use full exact candidates',
  'neighbor_recall_mean':{k:float(np.mean(v)) for k,v in recalls.items()},'neighbor_recall_min':{k:float(np.min(v)) for k,v in recalls.items()},
  'neighbor_radius_stretch_vs_exact':stretch,'neighbor_audit_queries':len(auditq),'reference_n_peaks':int(broad.sum()),'reference_distance':'Undirected unweighted graph hops to nearest stored broad_reference peak on fixed released full-atlas k30 graph',
  'reference_distance_unreachable':int((~np.isfinite(hops)).sum()),'reference_hops_max':float(hops[np.isfinite(hops)].max()),
  'graph_seconds':time.monotonic()-start,'source_fit':stat(SOURCE),'source_atlas':stat(ATLAS)}
 dump(OUT/'design.json',manifest)
 if min(manifest['neighbor_recall_mean'].values())<.95:raise RuntimeError('Candidate recall too low; revise before proceeding')
 log('Graph stage complete')

def write(g,key,value):
 g.create_dataset(key,data=value,compression='lzf')

def geometry(q,idx,xy,pc,hops,broad):
 out={};source={**xy,'pca':pc}
 for name,coord in source.items():
  center=coord[broad].mean(axis=0,dtype=float);radial=unit(coord[q].astype(float)-center)
  edge=np.empty(idx.shape,np.float32)
  direction=None if name=='pca' else np.empty((*idx.shape,coord.shape[1]),np.float32)
  for start in range(0,len(q),256):
   end=min(start+256,len(q));u=unit(coord[idx[start:end]].astype(float)-coord[q[start:end],None,:].astype(float))
   edge[start:end]=np.sum(u*radial[start:end,None,:],axis=2)
   if direction is not None:direction[start:end]=u
  out[name+'_radial']=edge
  if direction is not None:out[name+'_directions']=direction
 out['root_hop_delta']=hops[idx]-hops[q,None]
 return out

def smooth_source(v,valid,q,idx30,d30):
 weight=np.exp(-np.divide(d30,d30[:,-1:],out=np.zeros_like(d30),where=d30[:,-1:]>0))
 result=np.empty((len(q),v.shape[1]),float)
 for start in range(0,len(q),512):
  end=min(start+512,len(q));ii=idx30[start:end];vv=valid[ii]&np.isfinite(v[ii]);ww=weight[start:end,:,None]*vv
  den=ww.sum(axis=1)
  avg=np.divide((ww*np.where(vv,v[ii],0)).sum(axis=1),den,out=v[q[start:end]].astype(float).copy(),where=den>0)
  result[start:end]=avg
 return result

def cosine_candidates(x,vq,valid,q,idx,changing):
 n,k=idx.shape;cs=np.zeros((n,k),np.float32);comparable=np.zeros((n,k),bool)
 for start in range(0,n,256):
  end=min(start+256,n);ii=idx[start:end]
  shared=valid[q[start:end],None,:]&valid[ii]&np.isfinite(vq[start:end,None,:])
  delta=np.where(shared,x[ii]-x[q[start:end],None,:],0.).astype(float)
  v=np.where(shared,vq[start:end,None,:],0.)
  den=np.linalg.norm(delta,axis=2)*np.linalg.norm(v,axis=2)
  ok=(shared.sum(axis=2)>=15)&(den>1e-12)&changing[start:end,None]
  cs[start:end]=np.clip(np.divide(np.sum(delta*v,axis=2),den,out=np.zeros_like(den),where=ok),-1,1)
  comparable[start:end]=ok
 return cs,comparable

def projection(cs,comparable,dist,geom,k):
 cs=cs[:,:k];comparable=comparable[:,:k];dist=dist[:,:k]
 w=np.exp(-np.divide(dist,dist[:,-1:],out=np.zeros_like(dist),where=dist[:,-1:]>0)).astype(float)
 base=w*comparable;den=base.sum(axis=1);base=np.divide(base,den[:,None],out=np.zeros_like(base),where=den[:,None]>0)
 keep=comparable&(cs>=.2)&(cs>0);count=keep.sum(axis=1)
 p=w*np.where(keep,cs,0);p[count<2]=0;total=p.sum(axis=1)
 p=np.divide(p,total[:,None],out=np.zeros_like(p),where=total[:,None]>0)
 active=total>0
 r={'active':active,'support_count':count,'comparable_count':comparable.sum(axis=1)}
 for dim,name in [(2,'umap2d'),(3,'umap3d')]:
  vec=np.einsum('ij,ijd->id',p,geom[name+'_directions'][:,:k,:],optimize=True)
  baseline=np.einsum('ij,ijd->id',base,geom[name+'_directions'][:,:k,:],optimize=True)
  coherence=np.linalg.norm(vec,axis=1)
  nz=np.linalg.norm(geom[name+'_directions'][:,:k,:],axis=2)>0
  r['vectors'+str(dim)+'d']=vec.astype(np.float32)
  r['baseline_vectors'+str(dim)+'d']=baseline.astype(np.float32)
  r['valid'+str(dim)+'d']=active&(coherence>=.25)&((keep&nz).sum(axis=1)>=2)
  r['coherence'+str(dim)+'d']=coherence.astype(np.float32)
 for name in ['pca','umap2d','umap3d']:
  edge=geom[name+'_radial'][:,:k]
  r[name+'_radial']=np.sum(p*edge,axis=1).astype(np.float32)
  r[name+'_baseline_radial']=np.sum(base*edge,axis=1).astype(np.float32)
  r[name+'_excess_radial']=np.where(active,r[name+'_radial']-r[name+'_baseline_radial'],np.nan).astype(np.float32)
 hd=geom['root_hop_delta'][:,:k];finite=np.isfinite(hd).all(axis=1)
 r['root_delta']=np.where(active&finite,np.sum(p*np.where(np.isfinite(hd),hd,0),axis=1),np.nan).astype(np.float32)
 r['root_baseline_delta']=np.where((den>0)&finite,np.sum(base*np.where(np.isfinite(hd),hd,0),axis=1),np.nan).astype(np.float32)
 r['root_excess_delta']=(r['root_delta']-r['root_baseline_delta']).astype(np.float32)
 return r,p,base

def permuted_velocity(v,active,strata,seed):
 rng=np.random.default_rng(seed);donor=np.arange(len(v))
 for label in np.unique(strata[active]):
  ix=np.flatnonzero(active&(strata==label));donor[ix]=rng.permutation(ix)
 return donor

def sweep():
 start=time.monotonic();before={'fits':stat(SOURCE),'atlas':stat(ATLAS)}
 nn=np.load(OUT/'query_neighbors.npz');q=nn['source_indices'];idx30=nn['indices30'];d30=nn['distances30'];idx100=nn['indices100'];d100=nn['distances100']
 hops=np.load(OUT/'full_reference_graph_hops.npy')
 oldidx=np.load(PREV/'new_graph_indices.npy',mmap_mode='r');olddist=np.load(PREV/'new_graph_distances.npy',mmap_mode='r')
 with h5py.File(ATLAS,'r') as a:
  xy={key:a[key][:] for key in ['umap2d','umap3d']};pc=a['pca'][:]
  mean=a['annotations/mean_cpm'][:];tau=a['annotations/ct_tau'][:]
 with h5py.File(SOURCE,'r') as f:
  broad=f['broad_reference'][:].astype(bool);common=f['common_celltypes'][:].astype(bool);stages=f['stages'][:]
  lof=f['lack_of_fit'][:][:,common];modes=f['modes'][:][:,common]
  suppall=f['rate_supported'][:][:,common,:].astype(bool)
  strict_ct=((modes==2)|(modes==3))&(suppall.sum(axis=2)>=3)
  no_transient=~np.isin(modes,[4,5]).any(axis=1)
  log('Cache fixed edge geometry')
  g30=geometry(q,idx30,xy,pc,hops,broad);g100=geometry(q,idx100,xy,pc,hops,broad)
  strata=np.searchsorted(np.quantile(mean,np.arange(.1,1,.1)),mean)*4+np.searchsorted(np.quantile(tau,[.25,.5,.75]),tau)
  tmp=OUT/'sweep_vectors.pending.h5'
  with h5py.File(tmp,'w') as out:
   for key,val in [('source_indices',q),('umap2d',xy['umap2d'][q]),('umap3d',xy['umap3d'][q]),('root_hops',hops[q]),('broad_reference',broad[q]),('stages',stages)]:write(out,key,val)
   out.create_dataset('peak_ids',data=f['peak_ids'].asstr()[:][q].astype(object),dtype=h5py.string_dtype())
   out.attrs['schema']='interval_t/kK_bBBB/{vectors2d,vectors3d,valid2d,valid3d,active,rate_net,root_delta,root_baseline_delta,root_excess_delta,pca_radial,pca_excess_radial}; reversal and permutation_0..2 subgroups at every beta'
   out.attrs['interpretation']='Retrospective profile correspondence among fixed peaks. No reference direction enters probabilities. Graph-distance diagnostics are not developmental ancestry.'
   audits=out.create_group('audit')
   audit_rows=np.arange(0,len(q),max(1,len(q)//32))[:32]
   write(audits,'query_rows',audit_rows);write(audits,'source_indices',q[audit_rows]);write(audits,'reference_hops_all',hops)
   for t in range(5):
    it=time.monotonic();log(f'Interval {stages[t]}->{stages[t+1]}')
    x=f['fitted_cpm'][:,:,t][:,common].astype(float);v=f['rates'][:,:,t][:,common].astype(float);se=f['rate_se'][:,:,t][:,common]
    valid=np.isfinite(x)&np.isfinite(v)&np.isfinite(se)&(lof<=4)
    activefull=f[f'projection/{t}/changing'][:].astype(bool);active=activefull[q]
    smooth=smooth_source(v,valid,q,idx30,d30)
    gradual=no_transient[q]&np.any(strict_ct[q]&suppall[q,:,t],axis=1)
    increasing=np.any((modes[q]==2)&strict_ct[q]&suppall[q,:,t],axis=1)
    decreasing=np.any((modes[q]==3)&strict_ct[q]&suppall[q,:,t],axis=1)
    group=out.create_group(f'interval_{t}')
    for key,val in [('source_active',active),('gradual',gradual),('gradual_increasing',increasing),('gradual_decreasing',decreasing),('intrinsic_net_rate',f[f'projection/{t}/net_rate'][:][q])]:write(group,key,val)
    for beta in BETAS:
     vq=(1-beta)*v[q]+beta*smooth
     cs30,comp30=cosine_candidates(x,vq,valid,q,idx30,active)
     cs100,comp100=cosine_candidates(x,vq,valid,q,idx100,active)
     net=np.divide(np.sum(np.where(valid[q],vq,0),axis=1),valid[q].sum(axis=1),out=np.full(len(q),np.nan),where=valid[q].sum(axis=1)>0)
     for k in KS:
      cs,comp,dist,geo,idx=(cs30,comp30,d30,g30,idx30) if k<=30 else (cs100,comp100,d100,g100,idx100)
      r,p,base=projection(cs,comp,dist,geo,k);g=group.create_group(f'k{k}_b{int(beta*100):03d}')
      for key,val in r.items():write(g,key,val)
      write(g,'rate_net',net.astype(np.float32))
      rev,_,_=projection(-cs,comp,dist,geo,k);rg=g.create_group('reversed')
      for key,val in rev.items():write(rg,key,val)
      if beta==0 and k in [30,100]:
       ag=audits.create_group(f'interval_{t}_k{k}')
       for key,val in [('candidate_indices',idx[audit_rows,:k]),('candidate_distances',dist[audit_rows,:k]),('cosine',cs[audit_rows,:k]),('comparable',comp[audit_rows,:k]),('P',p[audit_rows]),('W',base[audit_rows]),('source_velocity',vq[audit_rows]),('source_state',x[q[audit_rows]]),('target_state',x[idx[audit_rows,:k]]),('source_valid',valid[q[audit_rows]]),('target_valid',valid[idx[audit_rows,:k]])]:write(ag,key,val)
    for rep in range(3):
     donor=permuted_velocity(v,activefull,strata,SEED+1000+t*10+rep);dq=donor[q]
     donor_smooth=smooth_source(v,valid,dq,oldidx[dq],olddist[dq])
     for beta in BETAS:
      vq=(1-beta)*v[dq]+beta*donor_smooth
      cs30,comp30=cosine_candidates(x,vq,valid,q,idx30,active);cs100,comp100=cosine_candidates(x,vq,valid,q,idx100,active)
      for k in KS:
       cs,comp,dist,geo=(cs30,comp30,d30,g30) if k<=30 else (cs100,comp100,d100,g100)
       r,_,_=projection(cs,comp,dist,geo,k);g=group[f'k{k}_b{int(beta*100):03d}'].create_group(f'permutation_{rep}')
       for key,val in r.items():write(g,key,val)
     write(group,'permutation_'+str(rep)+'_donor_indices',donor[q])
    out.flush();log(f'Interval {t} complete in {time.monotonic()-it:.1f}s')
   out.attrs['seconds']=time.monotonic()-start
  tmp.replace(OUT/'sweep_vectors.h5')
 after={'fits':stat(SOURCE),'atlas':stat(ATLAS)};assert before==after
 manifest=json.loads((OUT/'design.json').read_text());manifest.update(sweep_seconds=time.monotonic()-start,
  source_files_unchanged=True,reference_used_in_probabilities=False,source_gates_unchanged=True,
  rate_smoothing='beta0,.25,.5 blending intrinsic with feature-valid distance-weighted k30 neighbor mean; same original fit and source gates; no cis graph',
  gradual_definition='At least one CT classified increasing/decreasing with >=3 supported intervals and support in the evaluated interval; no CT classified transient peak/trough; all21 projection dimensions retained',
  baseline='Row-normalized original distance weights over feature-comparable, nonzero-displacement candidate edges; removes cosine direction selection only',
  permutation='3 independent shuffles of whole21-CT blended velocity vectors at every beta (smoothing calculated before donor reassignment) among source-active peaks within full-atlas mean-accessibility decile × ct_tau quartile; source feature masks/gates retained',
  reference_diagnostics='Expected change in fixed k30 graph-hop distance under directed P, minus comparable graph-weight baseline; positive away, negative toward broad reference',
  projection_parameters={'min_shared':15,'min_cosine':.2,'min_neighbors':2,'display_min_coherence':.25},
  limitations=['No biological replicates or independent count realization in these projections','Whole-course atlas geometry and21-CT interval state differ in semantics','PCA/reference and UMAP arrows are descriptive correspondence, not transport','Only20000 fixed random queries evaluated; target candidates span all640830 peaks'])
 dump(OUT/'design.json',manifest);log('Sweep complete')

def summary():
 rows=[];stability=[]
 with h5py.File(OUT/'sweep_vectors.h5','r') as f:
  root=f['broad_reference'][:].astype(bool);reachable=np.isfinite(f['root_hops'][:]);stages=f['stages'][:]
  for t in range(5):
   g=f[f'interval_{t}'];source=g['source_active'][:].astype(bool)&~root&reachable
   keys=[f'k{k}_b{int(b*100):03d}' for k in KS for b in BETAS]
   common=source.copy()
   for key in keys:common&=g[key+'/active'][:].astype(bool)
   strata={'all_source_supported':source,'broad_reference_sources':g['source_active'][:].astype(bool)&root&reachable,'gradual':source&g['gradual'][:].astype(bool),
           'common_active_all_settings':common,'gradual_common_active':common&g['gradual'][:].astype(bool)}
   refvec=g['k30_b000/vectors2d'][:];refvalid=g['k30_b000/valid2d'][:].astype(bool)
   for key in keys:
    k=int(key.split('_')[0][1:]);beta=int(key.split('_')[1][1:])/100
    base=g[key];controls={'observed':base,'reversed':base['reversed']}
    controls.update({f'permutation_{rep}':base[f'permutation_{rep}'] for rep in range(3)})
    for control,h in controls.items():
     active=h['active'][:].astype(bool);valid2=h['valid2d'][:].astype(bool);v=h['vectors2d'][:]
     for label,m in strata.items():
      finite=m&active&np.isfinite(h['root_excess_delta'][:]);n=int(m.sum());nproj=int(finite.sum())
      def avg(name):
       a=h[name][:];return float(a[finite].mean()) if nproj else np.nan
      ex=h['root_excess_delta'][:]
      matched=m&active&base['active'][:].astype(bool)&np.isfinite(ex)&np.isfinite(base['root_excess_delta'][:])
      paired_difference=ex[matched]-base['root_excess_delta'][:][matched]
      row=dict(interval=t,start_hpf=stages[t],end_hpf=stages[t+1],k=k,beta=beta,control=control,stratum=label,
        n_source=n,n_projectable=nproj,n_paired_with_observed=int(matched.sum()),mean_excess_minus_observed_paired=float(paired_difference.mean()) if len(paired_difference) else np.nan,projectable_fraction=nproj/n if n else np.nan,n_display2d=int((m&valid2).sum()),
        mean_root_delta=avg('root_delta'),mean_root_baseline_delta=avg('root_baseline_delta'),mean_root_excess_delta=avg('root_excess_delta'),
        all_source_mean_excess_zero_unprojectable=float(np.nansum(ex[m])/n) if n else np.nan,
        outward_excess_fraction=float(np.mean(ex[finite]>1e-8)) if nproj else np.nan,toward_excess_fraction=float(np.mean(ex[finite]<-1e-8)) if nproj else np.nan,
        mean_pca_excess_radial=avg('pca_excess_radial'),mean_umap2d_excess_radial=avg('umap2d_excess_radial'))
      rows.append(row)
     if control=='observed':
      matched=source&valid2&refvalid;cos=np.sum(unit(v[matched])*unit(refvec[matched]),axis=1)
      stability.append(dict(interval=t,k=k,beta=beta,n_both_display=int(matched.sum()),median_cosine_vs_k30_unsmoothed=float(np.median(cos)) if len(cos) else np.nan,
                            reversed_fraction_vs_k30_unsmoothed=float(np.mean(cos<0)) if len(cos) else np.nan))
 pd.DataFrame(rows).to_csv(OUT/'sweep_metrics.csv',index=False)
 pd.DataFrame(stability).to_csv(OUT/'direction_stability.csv',index=False)
 log('Summary tables complete')


def validate():
 checks=[]
 with h5py.File(OUT/'sweep_vectors.h5','r') as h,h5py.File(PREV/'updated_projection.h5','r') as released,h5py.File(SOURCE,'r') as source:
  q=h['source_indices'][:]
  assert np.array_equal(h['peak_ids'].asstr()[:],source['peak_ids'].asstr()[:][q])
  max_error=0.;mask_differences=0
  for t in range(5):
   assert np.array_equal(h[f'interval_{t}/source_active'][:],source[f'projection/{t}/changing'][:][q])
   for dim in [2,3]:
    ours=h[f'interval_{t}/k30_b000/vectors{dim}d'][:]
    prior=released[f'projection/{t}/vectors/umap{dim}d'][:][q]
    err=float(np.max(abs(ours-prior)));max_error=max(max_error,err)
    diff=int(np.sum(h[f'interval_{t}/k30_b000/valid{dim}d'][:]!=released[f'projection/{t}/projection_valid/umap{dim}d'][:][q]))
    mask_differences+=diff
    checks.append(dict(interval=t,dimension=dim,max_absolute_vector_error=err,display_mask_differences=diff))
  assert max_error<2e-6,(max_error,checks)
  assert mask_differences==0,checks
  audit=h['audit'];rows=audit['query_rows'][:];hops=audit['reference_hops_all'][:];qall=h['source_indices'][:]
  errp=0.;errflux=0.;errw=0.;audited=0
  for t in range(5):
   for k in [30,100]:
    a=audit[f'interval_{t}_k{k}'];g=h[f'interval_{t}/k{k}_b000']
    for i,row in enumerate(rows):
     idx=a['candidate_indices'][i];distance=a['candidate_distances'][i].astype(float);v=a['source_velocity'][i];x=a['source_state'][i];target=a['target_state'][i]
     shared=a['source_valid'][i][None,:]&a['target_valid'][i]&np.isfinite(v)[None,:]
     xx=np.where(shared,target-x[None,:],0);vv=np.where(shared,v[None,:],0)
     den=np.linalg.norm(xx,axis=1)*np.linalg.norm(vv,axis=1)
     valid=(shared.sum(axis=1)>=15)&(den>1e-12)&bool(h[f'interval_{t}/source_active'][row])
     cs=np.divide(np.sum(xx*vv,axis=1),den,out=np.zeros(len(idx)),where=valid)
     weights=np.exp(-distance/distance[-1]) if distance[-1]>0 else np.ones(len(idx))
     keep=valid&(cs>=.2)&(cs>0)
     P=weights*np.where(keep,cs,0)
     if keep.sum()<2:P[:]=0
     if P.sum()>0:P/=P.sum()
     W=weights*valid
     if W.sum()>0:W/=W.sum()
     errp=max(errp,float(np.max(abs(P-a['P'][i]))));errw=max(errw,float(np.max(abs(W-a['W'][i]))))
     if P.sum()>0:
      ex=float(np.dot(P-W,hops[idx]-hops[qall[row]]));errflux=max(errflux,abs(ex-float(g['root_excess_delta'][row])))
     audited+=1
  assert errp<2e-6 and errw<2e-6 and errflux<2e-6,(errp,errw,errflux)
 result={'released_k30_unsmoothed_reproduction':checks,'max_vector_error':max_error,'display_mask_differences':mask_differences,
         'independent_audit_rows':audited,'max_probability_error':errp,'max_baseline_probability_error':errw,'max_root_excess_error':errflux,
         'source_gates_exact':True,'peak_index_exact':True}
 dump(OUT/'validation.json',result);log('Independent row calculations and released k30 reproduction passed')

def plots(selected_strata=None):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import TwoSlopeNorm
 df=pd.read_csv(OUT/'sweep_metrics.csv')
 for stratum,stem in [('all_source_supported','01_root_flow_sensitivity'),('gradual','02_gradual_root_flow_sensitivity'),('common_active_all_settings','03_common_cohort_sensitivity'),('gradual_common_active','04_gradual_common_cohort_sensitivity')]:
  if selected_strata is not None and stratum not in selected_strata:continue
  sub=df[df.stratum==stratum];obs=sub[sub.control=='observed'];null=sub[sub.control.str.startswith('permutation_')]
  values=[]
  for t in range(5):
   a=obs[obs.interval==t].pivot(index='beta',columns='k',values='mean_root_excess_delta').reindex(index=BETAS,columns=KS).values
   b=-null[null.interval==t].groupby(['beta','k']).mean(numeric_only=True)['mean_excess_minus_observed_paired'].unstack('k').reindex(index=BETAS,columns=KS).values
   raw=obs[obs.interval==t].pivot(index='beta',columns='k',values='mean_root_delta').reindex(index=BETAS,columns=KS).values
   values.append((raw,a,b))
  lim=max(np.nanmax(abs(np.stack([z for pair in values for z in pair]))),.01)
  fig,axs=plt.subplots(3,5,figsize=(18,10),layout='constrained')
  for t in range(5):
   for row in range(3):
    ax=axs[row,t];im=ax.imshow(values[t][row],cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(4),KS);ax.set_yticks(range(3),[str(b) for b in BETAS]);ax.set_xlabel('Candidate neighbors k')
    if t==0:ax.set_ylabel('Rate blend beta\n'+['Raw directed drift','P minus geometry baseline','Observed minus matched permutation'][row])
    for iy in range(3):
     for ix in range(4):
      val=values[t][row][iy,ix];ax.text(ix,iy,f'{val:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(val)>.6*lim else 'black')
    if row==0:
     panel_title=['10→12 hpf','12→14 hpf','14→16 hpf','16→19 hpf','19→24 hpf'][t]
     if 'common' in stratum:panel_title+=f"\nFixed n = {int(obs[obs.interval==t].n_source.iloc[0]):,}"
     ax.set_title(panel_title)
  fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8,label='Graph-hop change or difference (see rows; positive = more outward)')
  title={'all_source_supported':'All source-supported query peaks outside broad reference','gradual':'Strict gradual-trend query peaks outside broad reference','common_active_all_settings':'Sensitivity analysis: fixed query cohort projectable under all 12 settings','gradual_common_active':'Sensitivity analysis: strict gradual-trend subset of the fixed common cohort'}[stratum]
  subtitle='Fixed random 20,000 queries; reference never directs arrows. Differences in lower rows are not raw outward flow.'
  if 'common' in stratum:subtitle='Outside broad reference; cohort fixed across settings within each interval. Selected sensitivity cohort, not a population estimate.'
  fig.suptitle(title+'\n'+subtitle,fontsize=13)
  fig.savefig(OUT/(stem+'.png'),dpi=180);fig.savefig(OUT/(stem+'.pdf'));plt.close(fig)
 log('Sensitivity plots complete')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['graph','sweep','summary','all'],default='all');args=p.parse_args()
 if args.stage in ['graph','all']:graph()
 if args.stage in ['sweep','all']:sweep()
 if args.stage in ['summary','all']:
  summary();validate();plots()
