#!/usr/bin/env python3
"""Apply frozen 20k-landmark fixed-peak models to the complete peak set."""
from pathlib import Path
import argparse,json,time
import h5py,joblib,numpy as np
from ct_peak_atlas import OUT,DATA,strings,col,log,dump

def prepare():
 cache=joblib.load(OUT/'models/ct_log_2d.joblib');pca=cache['pca']
 with h5py.File(OUT/'aligned_features.h5','r') as s:
  idx=s['source_indices'][:];sampleids=s['peak_ids'].asstr()[:];group=s['ct_all/group'].asstr()[:];stage=s['ct_all/stage'].asstr()[:]
 sample=np.load(OUT/'coordinates/ct_log_2d.npz')['pca'];sample3=np.load(OUT/'coordinates/ct_log_3d.npz')['pca'];assert np.array_equal(sample,sample3)
 cts=np.unique(group);sts=['0somites','5somites','10somites','15somites','20somites','30somites'];times=np.array([10,12,14,16,19,24])
 source=DATA/'peaks_by_ct_tp_master.h5ad'
 with h5py.File(source,'r') as h,h5py.File(OUT/'full_features.h5','w') as dest:
  n=h['layers/sum'].shape[0];ids=h['obs/_index'].asstr()[:];assert np.array_equal(ids[idx],sampleids);depth=h['var/total_coverage'][:]
  dest.create_dataset('peak_ids',data=ids.astype(object),dtype=h5py.string_dtype(),compression='gzip');dest.create_dataset('sample_source_indices',data=idx)
  dest.create_dataset('pca',shape=(n,sample.shape[1]),dtype='f4',compression='gzip',chunks=(8192,sample.shape[1]));a=dest.create_group('annotations')
  for key in ['mean_cpm','ct_tau','stage_of_max_mean_cpm']:a.create_dataset(key,shape=(n,),dtype='f4',compression='gzip')
  for key in ['nearest_gene','leiden_coarse']:dest.create_dataset(key,data=col(h,'obs/'+key).astype(object),dtype=h5py.string_dtype(),compression='gzip')
  maxerr=0
  for begin in range(0,n,32768):
   end=min(n,begin+32768);raw=h['layers/sum'][begin:end].astype(np.float32);cpm=(raw/depth*1e6).astype(np.float32);x=np.log1p(cpm);pc=pca.transform(x).astype(np.float32)
   ix=np.flatnonzero((idx>=begin)&(idx<end));maxerr=max(maxerr,float(np.max(np.abs(pc[idx[ix]-begin]-sample[ix]))) if len(ix) else 0)
   pc[idx[ix]-begin]=sample[ix] # Preserve exact sampled PC rows across batch numerical roundoff.
   assert np.isfinite(pc).all();dest['pca'][begin:end]=pc
   means=np.column_stack([cpm[:,group==ct].mean(axis=1) for ct in cts]);maxi=means.max(axis=1);tau=(1-np.divide(means,maxi[:,None],out=np.zeros_like(means),where=maxi[:,None]>0)).sum(axis=1)/(len(cts)-1);tau[maxi==0]=0
   a['mean_cpm'][begin:end]=cpm.mean(axis=1);a['ct_tau'][begin:end]=tau;a['stage_of_max_mean_cpm'][begin:end]=times[np.column_stack([cpm[:,stage==st].mean(axis=1) for st in sts]).argmax(axis=1)]
  dest.attrs['sample_max_pca_difference_before_exact_preservation']=maxerr;dest.attrs['geometry']='Frozen PCA fit20,000 sample training peaks; 50components of log1p(full-library CPM).'
 log(f'Full PCA prepared: {n:,} peaks; sample maxfloat32roundoff={maxerr}')

def run(dim):
 name=f'ct_log_{dim}d';saved=joblib.load(OUT/'models'/f'{name}.joblib');cache=saved['umap_cache'];model=cache['model']
 with h5py.File(OUT/'full_features.h5','r') as h:queries=h['pca'][:];sampleidx=h['sample_source_indices'][:]
 known=cache['reference_states'];unique,inverse=np.unique(np.concatenate([known,queries]),axis=0,return_inverse=True);coordinates=np.full((len(unique),dim),np.nan,np.float32);coordinates[inverse[:len(known)]]=cache['reference_coordinates'];unseen=np.flatnonzero(~np.isfinite(coordinates).all(axis=1));log(f'{dim}Dfull: {len(queries):,}peaks, {len(unseen):,}unseenuniquevectors')
 start=time.monotonic();batches=[]
 for begin in range(0,len(unseen),30000):
  t=time.monotonic();take=unseen[begin:begin+30000];coordinates[take]=model.transform(unique[take]);seconds=time.monotonic()-t;batches.append(seconds);log(f'{dim}D: transformed{min(begin+30000,len(unseen)):,}/{len(unseen):,}, batch{seconds:.1f}s, total{time.monotonic()-start:.1f}s')
 assert np.isfinite(coordinates).all();mapped=coordinates[inverse[len(known):]]
 expected=np.load(OUT/'coordinates'/f'{name}.npz')['umap'];assert np.array_equal(mapped[sampleidx],expected)
 np.save(OUT/f'full_umap{dim}d.npy',mapped)
 info={'n_full_peaks':len(queries),'n_unique_training_and_full_query_vectors':len(unique),'n_new_transformed_vectors':len(unseen),'n_dimensions':dim,'fit_training_peaks':20000,'sampled_coordinates_bitwise_preserved':True,'all_coordinates_finite':True,'transform_seconds':time.monotonic()-start,'batch_seconds':batches,'method':'Frozen20k-landmark-trainedPCA/UMAP; existing25,056 samplecoordinates exactcached. Not refitted on full640,830 peaks.'};dump(OUT/f'full_umap{dim}d_validation.json',info)
 log(f'{dim}D full atlas complete')

def finish():
 with h5py.File(OUT/'full_features.h5','r') as src,h5py.File(OUT/'full_ct_peak_atlas.h5','w') as h:
  for key in src:src.copy(key,h)
  for key,value in src.attrs.items():h.attrs[key]=value
  for d in [2,3]:
   y=np.load(OUT/f'full_umap{d}d.npy');assert len(y)==len(h['peak_ids']) and np.isfinite(y).all();h.create_dataset(f'umap{d}d',data=y,compression='gzip');h.attrs[f'umap{d}d_method']='Frozen20,000-landmark-trainedmodel, fullpeakprojection with exact samplecache.'
 log('Consolidated full_ct_peak_atlas.h5')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','finish']);p.add_argument('--dim',type=int,choices=[2,3]);a=p.parse_args()
 {'prepare':prepare,'run':lambda:run(a.dim),'finish':finish}[a.command]()
