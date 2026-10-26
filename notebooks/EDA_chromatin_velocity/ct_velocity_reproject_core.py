#!/usr/bin/env python3
"""Reuse fitted linear accessibility rates on a new CT-stage fixed peak atlas.

The high-dimensional log-CPM PCA space defines candidate neighbors. Direction
alignment remains an uncentered cosine between linear CPM stage-state differences
and linear CPM/hour changes. This is a hybrid correspondence projection, not a
Jacobian pushforward through a 190-feature UMAP, and does not infer peak ancestry.
"""
from pathlib import Path
import argparse, gc, json, os, sys, time
import h5py
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_velocity_reprojection_v1'
FIT=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'
ATLAS=HERE/'outputs/ct_peak_atlas_v1/full_ct_peak_atlas.h5'
sys.path.insert(0,str(HERE))
from real_trajectory_velocity import PROJ, PARAMS, source_diagnostics, store_projection, write_graph
from trajectory_projection import project_interval
SEED=20261021


def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def stat(p):
 s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def dump(path,data):
 def convert(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  raise TypeError(type(x).__name__)
 Path(path).write_text(json.dumps(data,indent=2,default=convert,allow_nan=False)+'\n')


def graph_stage():
 from pynndescent import NNDescent
 from numba import set_num_threads
 set_num_threads(16)
 OUT.mkdir(parents=True,exist_ok=True)
 with h5py.File(ATLAS,'r') as h: x=h['pca'][:].astype(np.float32)
 n=len(x);assert x.shape==(640830,50) and np.isfinite(x).all()
 t=time.monotonic()
 log(f'Build all-peak PCA approximate kNN: {x.shape}, 16 threads')
 index=NNDescent(x,n_neighbors=40,metric='euclidean',n_trees=24,n_iters=12,
                 max_candidates=60,low_memory=True,n_jobs=16,random_state=SEED,
                 verbose=True,compressed=False)
 idx,dist=index.neighbor_graph
 # Remove self explicitly even when an approximate result omits self or has ties.
 row=np.arange(n)[:,None]
 d=np.where((idx>=0)&(idx!=row),dist,np.inf)
 order=np.argsort(d,axis=1,kind='stable')[:,:30]
 idx=np.take_along_axis(idx,order,axis=1).astype(np.int32)
 dist=np.take_along_axis(d,order,axis=1).astype(np.float32)
 assert np.isfinite(dist).all() and np.all(idx!=row) and np.all(idx>=0)
 assert np.all(np.diff(np.sort(idx,axis=1),axis=1)>0)
 elapsed=time.monotonic()-t
 np.save(OUT/'new_graph_indices.npy',idx);np.save(OUT/'new_graph_distances.npy',dist)
 del index;gc.collect()
 # Nonself duplicate-state distances are permitted; all-zero rows get weight1.
 scale=dist[:,-1:]
 weight=np.exp(-np.divide(dist,scale,out=np.zeros_like(dist),where=scale>0)).astype(np.float32)
 graph=sparse.csr_matrix((weight.ravel(),idx.ravel().copy(),np.arange(n+1,dtype=np.int64)*30),shape=(n,n))
 graph.sort_indices();sparse.save_npz(OUT/'new_graph_csr.npz',graph,compressed=True)
 # Independent exact brute-force audit, with float64 distances and own self mask.
 rng=np.random.default_rng(SEED);queries=np.sort(rng.choice(n,128,replace=False))
 exact=np.empty((len(queries),30),np.int32);recalls=[];distance_errors=[]
 for begin in range(0,len(queries),8):
  q=queries[begin:begin+8]
  ds=cdist(x[q].astype(float),x.astype(float),'sqeuclidean')
  ds[np.arange(len(q)),q]=np.inf
  near=np.argpartition(ds,29,axis=1)[:,:30]
  nearest=np.take_along_axis(near,np.argsort(np.take_along_axis(ds,near,axis=1),axis=1),axis=1)
  exact[begin:begin+len(q)]=nearest
  for ii,src in enumerate(q):
   recalls.append(len(set(idx[src]).intersection(nearest[ii]))/30)
   recomputed=np.sqrt(np.sum((x[idx[src]].astype(float)-x[src].astype(float))**2,axis=1))
   distance_errors.extend(abs(recomputed-dist[src]))
 log(f'Approximate kNN audit mean recall {np.mean(recalls):.5f}, min {np.min(recalls):.5f}')
 manifest={'n_peaks':n,'n_features':50,'k_nonself':30,'metric':'Euclidean in frozen 50-PC log1p(CPM) geometry',
  'algorithm':'PyNNDescent full-peak approximate kNN','n_neighbors_search':40,'n_trees':24,'n_iters':12,'max_candidates':60,
  'threads':16,'seed':SEED,'graph_seconds':elapsed,'row_weight':'exp(-distance / row kth-neighbor distance); all-zero distance rows weight1',
  'directed_graph':True,'symmetrized':False,'graph_nnz':graph.nnz,'original_umap_graph_reused':False,
  'landmark_geometry_note':'PCA/UMAP trained on20000 peaks; candidate graph rebuilt among all640830 projected PCA profiles.',
  'audit_queries':queries,'audit_neighbor30_recall':recalls,'audit_mean_recall':np.mean(recalls),'audit_min_recall':np.min(recalls),
  'audit_p05_recall':np.quantile(recalls,.05),'audit_max_distance_absolute_error':np.max(distance_errors),
  'audit_exact_method':'float64 scipy cdist against all peaks, explicit self exclusion, top30 selection',
  'all_rows_30_unique_nonself_neighbors':True,'all_weights_finite_positive':bool(np.isfinite(graph.data).all() and (graph.data>0).all()),
  'source_atlas':stat(ATLAS)}
 np.savez_compressed(OUT/'new_graph_exact_audit.npz',queries=queries,exact_neighbor_indices=exact,approximate_neighbor_indices=idx[queries],neighbor30_recall=np.asarray(recalls))
 dump(OUT/'new_graph_manifest.json',manifest)
 if manifest['audit_mean_recall']<.9: raise RuntimeError('Approximate graph recall below .9; increase search before projection')
 log(f'Candidate graph complete: {graph.nnz:,} edges')



def exact_graph_stage():
 """Recompute selected approximate-neighbor edge lengths before weighting."""
 import shutil
 start=time.monotonic()
 with h5py.File(ATLAS,'r') as h:x=h['pca'][:].astype(np.float64)
 idx=np.load(OUT/'new_graph_indices.npy');previous=np.load(OUT/'new_graph_distances.npy')
 manifest=json.loads((OUT/'new_graph_manifest.json').read_text())
 for filename in ['new_graph_distances.npy','new_graph_csr.npz','new_graph_manifest.json','new_graph_projection_validation.json']:
  source=OUT/filename
  if source.exists():shutil.copy2(source,OUT/filename.replace('new_graph_','new_graph_pre_exact_',1))
 old=OUT/'updated_projection.h5'
 if old.exists():old.replace(OUT/'new_graph_pre_exact_projection.h5')
 from concurrent.futures import ThreadPoolExecutor,as_completed
 distance=np.empty_like(previous);errors=[];relative=[]
 def one_block(begin):
  end=min(len(x),begin+4096)
  delta=x[idx[begin:end]]-x[begin:end,None,:]
  exact=np.sqrt(np.sum(delta*delta,axis=2))
  distance[begin:end]=exact.astype(np.float32)
  error=float(np.max(abs(exact-previous[begin:end])))
  relative_error=float(np.max(np.divide(abs(exact-previous[begin:end]),np.maximum(exact[:,-1:],1e-12))))
  return end,error,relative_error
 with ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(one_block,begin) for begin in range(0,len(x),4096)]
  for done,future in enumerate(as_completed(futures),1):
   end,error,relative_error=future.result();errors.append(error);relative.append(relative_error)
   if done==1 or done%32==0 or done==len(futures):log(f'Exact selected-edge blocks {done}/{len(futures)}')
 order=np.argsort(distance,axis=1,kind='stable')
 idx=np.take_along_axis(idx,order,axis=1)
 distance=np.take_along_axis(distance,order,axis=1)
 assert np.all(np.isfinite(distance)) and np.all(distance>=0)
 scale=distance[:,-1:]
 weight=np.exp(-np.divide(distance,scale,out=np.zeros_like(distance),where=scale>0)).astype(np.float32)
 graph=sparse.csr_matrix((weight.ravel(),idx.ravel().copy(),np.arange(len(x)+1,dtype=np.int64)*30),shape=(len(x),len(x)))
 graph.sort_indices()
 np.save(OUT/'new_graph_indices.npy',idx);np.save(OUT/'new_graph_distances.npy',distance)
 sparse.save_npz(OUT/'new_graph_csr.npz',graph,compressed=True)
 audit=np.load(OUT/'new_graph_exact_audit.npz');queries=audit['queries'];near=audit['exact_neighbor_indices']
 recalls=np.array([len(set(idx[src]).intersection(nn))/30 for src,nn in zip(queries,near)])
 exact=np.sqrt(np.sum((x[idx[queries]]-x[queries,None,:])**2,axis=2))
 graph_error=0.
 for src in queries:
  g=graph.getrow(src);recomputed=np.sqrt(np.sum((x[g.indices]-x[src])**2,axis=1)).astype(np.float32)
  expected=np.exp(-recomputed/recomputed.max()) if recomputed.max()>0 else np.ones(len(g.indices))
  graph_error=max(graph_error,float(np.max(abs(g.data-expected))))
 assert graph_error<1e-7
 manifest.update(exact_selected_edge_distances=True,exact_distance_dtype='float64 subtraction and squared sum; stored float32',
  exact_edge_recompute_seconds=time.monotonic()-start,corrected_approx_distance_max_absolute_error=max(errors),
  corrected_approx_distance_max_error_relative_to_row_radius=max(relative),
  audit_max_distance_absolute_error=float(np.max(abs(exact-distance[queries]))),
  audit_neighbor30_recall=recalls,audit_mean_recall=float(recalls.mean()),audit_min_recall=float(recalls.min()),
  audit_graph_weight_max_error=graph_error,
  audit_correction='Original distance diagnostic paired arrays after CSR index aliasing reordered only indices. Saved edge weights were correct. Final graph recomputes every selected edge distance in float64 and copies CSR indices to prevent aliasing.')
 dump(OUT/'new_graph_manifest.json',manifest)
 np.savez_compressed(OUT/'new_graph_exact_audit.npz',queries=queries,exact_neighbor_indices=near,approximate_neighbor_indices=idx[queries],neighbor30_recall=recalls)
 log(f'Exact edge update complete; original approximate max distance error {max(errors):.3g}, graph kernel audit {graph_error:.3g}')

def audit_rows(graph,x,v,valid,active,embeddings,result,seed):
 """Independent sparse-row reimplementation of cosine/filter/unit-vector rules."""
 rng=np.random.default_rng(seed)
 choices=[]
 for mask in [result['active'],result['changing']&~result['active'],~result['changing']]:
  eligible=np.flatnonzero(mask)
  if len(eligible):choices.extend(rng.choice(eligible,min(16,len(eligible)),replace=False).tolist())
 required=max(PROJ['min_shared_features'],int(np.ceil(x.shape[1]*PROJ['min_shared_fraction'])))
 maxp=0.;maxv=0.;checks=0
 for src in choices:
  begin,end=graph.indptr[src:src+2];target=graph.indices[begin:end];weights=graph.data[begin:end].astype(float)
  keep=[];unnormalized=[]
  for j,w in zip(target,weights):
   shared=valid[src]&np.isfinite(x[src])&np.isfinite(v[src])&valid[j]&np.isfinite(x[j])
   if src==j or w<=0 or shared.sum()<required or not active[src] or not result['changing'][src]:continue
   delta=x[j,shared].astype(float)-x[src,shared].astype(float);velocity=v[src,shared].astype(float)
   nd=np.linalg.norm(delta);nv=np.linalg.norm(velocity)
   if nd<=1e-12 or nv<=0:continue
   cs=float(np.clip(np.dot(delta,velocity)/(nd*nv),-1,1))
   if cs>0 and cs>=PROJ['min_cosine']:keep.append(int(j));unnormalized.append(w*cs**PROJ['cosine_power'])
  if len(keep)<PROJ['min_neighbors']:keep=[];unnormalized=[]
  pp=np.asarray(unnormalized,dtype=float)
  if len(pp):pp/=pp.sum()
  expected={j:p for j,p in zip(keep,pp)}
  actual=result['probabilities'].getrow(src)
  assert set(expected)==set(actual.indices),f'Edge selection differs at {src}'
  if len(pp):maxp=max(maxp,max(abs(expected[j]-p) for j,p in zip(actual.indices,actual.data)))
  for name,xy in embeddings.items():
   vec=np.zeros(xy.shape[1]);positive_count=0
   for j,p in zip(keep,pp):
    delta=xy[j].astype(float)-xy[src].astype(float);norm=np.linalg.norm(delta)
    if norm>0:vec+=p*delta/norm;positive_count+=1
   maxv=max(maxv,float(np.max(abs(vec-result['vectors'][name][src]))))
   expected_valid=len(pp)>0 and np.linalg.norm(vec)>=PROJ['min_coherence'] and np.linalg.norm(vec)>1e-12 and positive_count>=PROJ['min_neighbors']
   assert expected_valid==bool(result['projection_valid'][name][src])
  checks+=1
 assert maxp<1e-10 and maxv<1e-10,(maxp,maxv)
 return {'rows_checked':checks,'max_probability_error':maxp,'max_vector_error':maxv,'projection_masks_match':True}


def projection_stage():
 OUT.mkdir(parents=True,exist_ok=True)
 before={'fits':stat(FIT),'atlas':stat(ATLAS)}
 graph=sparse.load_npz(OUT/'new_graph_csr.npz').tocsr()
 manifest=json.loads((OUT/'new_graph_manifest.json').read_text())
 validations=[];timings=[]
 temp=OUT/'updated_projection.pending.h5'
 with h5py.File(FIT,'r') as f,h5py.File(ATLAS,'r') as a,h5py.File(temp,'w') as h:
  ids=f['peak_ids'].asstr()[:];assert np.array_equal(ids,a['peak_ids'].asstr()[:])
  assert graph.shape==(len(ids),len(ids))
  embeddings={name:a[name][:] for name in ['umap2d','umap3d']};embeddings['old_umap2d']=f['umap2d'][:]
  for key in ['peak_ids','celltypes','stages','common_celltypes','condition_valid']:f.copy(key,h)
  for key,xy in embeddings.items():h.create_dataset(key,data=xy,compression='lzf')
  write_graph(h.create_group('graph'),graph)
  common=f['common_celltypes'][:].astype(bool);lof=f['lack_of_fit'][:][:,common]
  common_names=f['celltypes'].asstr()[:][common]
  h.create_dataset('projection_celltypes',data=common_names.astype(object),dtype=h5py.string_dtype())
  h.create_group('projection')
  for t in range(5):
   start=time.monotonic()
   x=f['fitted_cpm'][:,:,t][:,common];v=f['rates'][:,:,t][:,common];se=f['rate_se'][:,:,t][:,common]
   supported=f['rate_supported'][:,:,t][:,common]
   valid=np.isfinite(x)&np.isfinite(v)&np.isfinite(se)&(lof<=PARAMS['lack_of_fit_threshold'])
   active,snr=source_diagnostics(v,se,valid,supported)
   log(f'Project interval{t}: {active.sum():,} supported sources')
   result=project_interval(graph,x,v,feature_valid=valid,embeddings=embeddings,source_active=active,block_size=2048,return_probabilities=True,**PROJ)
   # Numerical independence from the old graph: source and highdim magnitudes must be unchanged.
   old=f[f'projection/{t}']
   assert np.array_equal(result['changing'],old['changing'][:])
   assert np.allclose(result['net_rate'],old['net_rate'][:],atol=1e-7,rtol=1e-6,equal_nan=True)
   assert np.allclose(result['velocity_rms'],old['velocity_rms'][:],atol=1e-7,rtol=1e-6,equal_nan=True)
   rowsums=np.asarray(result['probabilities'].sum(axis=1)).ravel()
   assert np.allclose(rowsums[result['active']],1,atol=1e-12)
   assert np.all(rowsums[~result['active']]==0)
   audit=audit_rows(graph,x,v,valid,active,embeddings,result,SEED+t)
   audit.update(interval=t,active_rows=int(result['active'].sum()),source_active=int(active.sum()),probability_nnz=result['probabilities'].nnz,
     unchanged_changing_sources=True,unchanged_net_rate_and_velocity_rms=True,row_probability_sums_valid=True,
     projected_counts={k:int(mask.sum()) for k,mask in result['projection_valid'].items()})
   validations.append(audit)
   g=h['projection'].create_group(str(t));g.create_dataset('source_snr',data=snr.astype('f4'),compression='lzf');store_projection(g,result)
   elapsed=time.monotonic()-start;timings.append(elapsed)
   log(f'Interval{t} complete {elapsed:.1f}s, new2D {audit["projected_counts"]["umap2d"]:,}, new3D {audit["projected_counts"]["umap3d"]:,}')
   del result;gc.collect();h.flush()
  h.attrs['projection_parameters_json']=json.dumps(PROJ)
  h.attrs['graph_manifest_json']=json.dumps(manifest)
  h.attrs['source_files_json']=json.dumps(before)
  h.attrs['method']='Wholecourse log-CPM PCA neighbors; stage-specific linear CPM state and CPM/hour velocity cosine; probability-weighted unit embedding directions.'
  h.attrs['interpretation']='Local retrospective correspondence between fixed peak profiles; no forced root or peak transport. Not a differential pushforward of wholecourse states.'
  h.attrs['source_active_rule']=f.attrs['source_active_rule']
  h.attrs['rate_source']='Reused identical stored six-stage fitted accessibility and rates; no refitting, entropy weighting or coaccessibility smoothing.'
  h.attrs['direction_vector_units']='Dimensionless probability-weighted unit directions. CPM/hour magnitude and net rate stored separately.'
 after={'fits':stat(FIT),'atlas':stat(ATLAS)};assert before==after
 temp.replace(OUT/'updated_projection.h5')
 dump(OUT/'new_graph_projection_validation.json',{'checks_passed':True,'source_files_unchanged':before==after,'source_files':before,
      'n_peaks':len(ids),'intervals':validations,'interval_seconds':timings,
      'ordered_peak_ids_exact':True,'n_projection_celltypes':len(common_names),'projection_celltypes':common_names,
      'graph_mean_exact_neighbor30_recall':manifest['audit_mean_recall']})
 log('All updated projections saved and validated')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['graph','exact','project','all'],default='all');a=p.parse_args()
 if a.stage in ['graph','all']:graph_stage()
 if a.stage in ['exact','all']:exact_graph_stage()
 if a.stage in ['project','all']:projection_stage()

if __name__=='__main__':main()
