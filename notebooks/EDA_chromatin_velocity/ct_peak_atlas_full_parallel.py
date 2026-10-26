#!/usr/bin/env python3
"""Checkpointed independent fixed-boundary batches for landmark projection."""
from pathlib import Path
import argparse,json,os,time,concurrent.futures,multiprocessing as mp
import h5py,joblib,numpy as np
from ct_peak_atlas import OUT,log,dump
MODELS={}

def plan():
 cache=joblib.load(OUT/'models/ct_log_2d.joblib')['umap_cache']
 with h5py.File(OUT/'full_features.h5','r') as h:q=h['pca'][:]
 known=cache['reference_states'];unique,inverse=np.unique(np.concatenate([known,q]),axis=0,return_inverse=True);known_codes=inverse[:len(known)];seen=np.zeros(len(unique),bool);seen[known_codes]=True;unseen=np.flatnonzero(~seen)
 np.save(OUT/'full_unique_pca.npy',unique);np.savez(OUT/'full_transform_plan.npz',known_codes=known_codes,query_codes=inverse[len(known):],unseen_codes=unseen)
 dump(OUT/'full_transform_plan.json',{'query_peaks':len(q),'unique_vectors':len(unique),'known_vectors':len(known),'unseen_vectors':len(unseen),'chunk_size':30000,'chunk_boundaries':[[b,min(len(unseen),b+30000)] for b in range(0,len(unseen),30000)],'seeds':{'fit':20261021,'transform':20261021},'parallelism':'4 processes/model, two models concurrent,2threads/process','restart_note':'Original serial processes were terminated after initial batches; no saved model or sample coordinate was lost. Batches repeated to reduce elapsed time. All batches now atomically checkpointed.'})
 (OUT/'full_checkpoints').mkdir(exist_ok=True);log('Full transform plan ready')

def worker(args):
 dim,begin,end=args;checkpoint=OUT/'full_checkpoints'/f'{dim}d_{begin:06d}_{end:06d}.npy';timing=checkpoint.with_suffix('.json')
 if checkpoint.exists():
  a=np.load(checkpoint);assert a.shape==(end-begin,dim) and np.isfinite(a).all();return {'dimension':dim,'begin':begin,'end':end,'cached':True}
 if dim not in MODELS:MODELS[dim]=joblib.load(OUT/'models'/f'ct_log_{dim}d.joblib')['umap_cache']['model']
 model=MODELS[dim];unique=np.load(OUT/'full_unique_pca.npy',mmap_mode='r');unseen=np.load(OUT/'full_transform_plan.npz')['unseen_codes'];take=unseen[begin:end];t=time.monotonic();a=model.transform(unique[take]);sec=time.monotonic()-t;assert a.shape==(end-begin,dim) and np.isfinite(a).all()
 with open(str(checkpoint)+'.tmp','wb') as f:np.save(f,a)
 os.replace(str(checkpoint)+'.tmp',checkpoint);info={'dimension':dim,'begin':begin,'end':end,'seconds':sec,'pid':os.getpid(),'cached':False};dump(timing,info);return info

def run(dim):
 planinfo=json.loads((OUT/'full_transform_plan.json').read_text());start=time.monotonic();tasks=[(dim,b,e) for b,e in planinfo['chunk_boundaries']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
  for f in concurrent.futures.as_completed([pool.submit(worker,t) for t in tasks]):log(str(f.result()))
 cache=joblib.load(OUT/'models'/f'ct_log_{dim}d.joblib')['umap_cache'];p=np.load(OUT/'full_transform_plan.npz');coords=np.full((planinfo['unique_vectors'],dim),np.nan,np.float32);coords[p['known_codes']]=cache['reference_coordinates']
 for _,b,e in tasks:coords[p['unseen_codes'][b:e]]=np.load(OUT/'full_checkpoints'/f'{dim}d_{b:06d}_{e:06d}.npy')
 assert np.isfinite(coords).all();mapped=coords[p['query_codes']]
 with h5py.File(OUT/'full_features.h5','r') as h:idx=h['sample_source_indices'][:]
 expected=np.load(OUT/'coordinates'/f'ct_log_{dim}d.npz')['umap'];assert np.array_equal(mapped[idx],expected)
 np.save(OUT/f'full_umap{dim}d.npy',mapped);dump(OUT/f'full_umap{dim}d_validation.json',{'n_full_peaks':len(mapped),'n_dimensions':dim,'n_unique_training_and_full_query_vectors':planinfo['unique_vectors'],'n_new_transformed_vectors':planinfo['unseen_vectors'],'fit_training_peaks':20000,'sampled_coordinates_bitwise_preserved':True,'all_coordinates_finite':True,'elapsed_seconds':time.monotonic()-start,'parallel_workers':4,'threads_per_worker':2,'independent_chunk_size':30000,'method':'Frozen20k-landmarkPCA/UMAP; all25,056 sampledcoordinates exactcached. Same30kquerychunkboundaries andtransformseed as originalserialrun. Fullatlas wasnotrefit.'});log(f'{dim}D checkpointed projection complete')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','run']);p.add_argument('--dim',type=int,choices=[2,3]);a=p.parse_args();plan() if a.command=='plan' else run(a.dim)
