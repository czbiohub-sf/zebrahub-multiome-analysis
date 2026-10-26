"""Reproject saved temporal transitions onto the new fixed celltype peak atlas.

This file owns the coordinate-only control and figures. Updated PCA neighbors
and rate-aligned probabilities are computed by ct_velocity_reproject_core.py.
Source fits and both atlas coordinate sets remain read-only.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import h5py,numpy as np,pandas as pd
from scipy import sparse
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/ct_velocity_reprojection_v1';OLD=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5';ATLAS=HERE/'outputs/ct_peak_atlas_v1/full_ct_peak_atlas.h5'
sys.path.insert(0,str(HERE.parents[1]/'scripts/chrom_velo'))

def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist())+'\n')
def read_csr(g):return sparse.csr_matrix((g['data'][:],g['indices'][:],g['indptr'][:]),shape=tuple(g.attrs['shape']))
def write_array(g,key,a):g.create_dataset(key,data=a,compression='lzf')

def project_saved(P,coordinates,active):
 n=P.shape[0];results={}
 for key,y in coordinates.items():
  v=np.zeros(y.shape,float);count=np.zeros(n,np.int32);mass=np.zeros(n,float);reach=np.zeros(n,float)
  for start in range(0,n,8192):
   stop=min(start+8192,n);a,b=P.indptr[start],P.indptr[stop];row=np.repeat(np.arange(stop-start),np.diff(P.indptr[start:stop+1]));col=P.indices[a:b];w=P.data[a:b].astype(float)
   delta=y[col].astype(float)-y[start+row].astype(float);distance=np.linalg.norm(delta,axis=1);unit=np.divide(delta,distance[:,None],out=np.zeros_like(delta),where=distance[:,None]>0)
   for j in range(y.shape[1]):v[start:stop,j]=np.bincount(row,weights=w*unit[:,j],minlength=stop-start)
   count[start:stop]=np.bincount(row[distance>0],minlength=stop-start);mass[start:stop]=np.bincount(row,weights=w*(distance>0),minlength=stop-start);reach[start:stop]=np.bincount(row,weights=w*distance,minlength=stop-start)
  coherence=np.linalg.norm(v,axis=1);valid=active&(coherence>=.25)&(coherence>1e-12)&(count>=2)
  results[key]=dict(vectors=v.astype('f4'),coherence=coherence.astype('f4'),projection_valid=valid,display_vectors=np.where(valid[:,None],v,0).astype('f4'),projection_neighbor_count=count,projection_support=mass.astype('f4'),mean_edge_length=reach.astype('f4'))
 return results

def reuse():
 OUT.mkdir(exist_ok=True,parents=True);before={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in [OLD,ATLAS]};checks=[];counts=[]
 with h5py.File(OLD) as old,h5py.File(ATLAS) as atlas,h5py.File(OUT/'reused_projection.h5','w') as dest:
  ids=old['peak_ids'].asstr()[:];assert np.array_equal(ids,atlas['peak_ids'].asstr()[:]);n=len(ids)
  dest.create_dataset('peak_ids',data=ids.astype(object),dtype=h5py.string_dtype())
  for key in ['celltypes','stages','common_celltypes']:old.copy(key,dest)
  coords={k:atlas[k][:] for k in ['umap2d','umap3d']};coords['old_umap2d']=old['umap2d'][:]
  for k,a in coords.items():write_array(dest,k,a)
  parent=dest.create_group('projection')
  for t in range(5):
   source=old[f'projection/{t}'];g=parent.create_group(str(t));P=read_csr(source['probabilities']);assert P.shape==(n,n);assert np.isfinite(P.data).all() and (P.data>=0).all();active=source['active'][:]
   assert np.allclose(np.asarray(P.sum(axis=1)).ravel(),active.astype(float),atol=2e-7)
   for key,obj in source.items():
    if isinstance(obj,h5py.Dataset):source.copy(key,g)
   source.copy('probabilities',g);r=project_saved(P,coords,active)
   for d,fields in r.items():
    for k,v in fields.items():write_array(g.require_group(k),d,v)
   # Old-coordinate reconstruction must recover the previous arrows numerically.
   err=float(np.abs(r['old_umap2d']['vectors']-source['vectors/umap2d'][:]).max());assert err<2e-7,err
   changedmask=int((r['old_umap2d']['projection_valid']!=source['projection_valid/umap2d'][:]).sum());assert changedmask==0
   manual=0;candidate=np.flatnonzero(active);rng=np.random.default_rng(20261021+t)
   for i in rng.choice(candidate,min(32,len(candidate)),replace=False):
    j=P.indices[P.indptr[i]:P.indptr[i+1]];w=P.data[P.indptr[i]:P.indptr[i+1]].astype(float)
    for d in ['umap2d','umap3d']:
     delta=coords[d][j].astype(float)-coords[d][i].astype(float);length=np.linalg.norm(delta,axis=1);vv=np.sum(w[:,None]*np.divide(delta,length[:,None],out=np.zeros_like(delta),where=length[:,None]>0),axis=0);assert np.allclose(vv,r[d]['vectors'][i],atol=6e-8);manual+=1
   # Geometry-specific derived flags replace the previous geometry's flags.
   for key in ['unprojectable']:
    del g[key];write_array(g,key,source['changing'][:]&~(r['umap2d']['projection_valid']|r['umap3d']['projection_valid']))
   g.attrs['method']='Unchanged previous directed probabilities; recomputed unit directions and display gates in new coordinates.'
   checks.append(dict(interval=t,old_vector_max_absolute_error=err,old_mask_differences=changedmask,manual_rows_dimensions=manual,probabilities_reused_exactly=True))
   for dim in ['umap2d','umap3d','old_umap2d']:counts.append(dict(interval=t,embedding=dim,method='reused',changing=int(source['changing'][:].sum()),directed_sources=int(active.sum()),projected=int(r[dim]['projection_valid'].sum()),median_weighted_edge_length=float(np.median(r[dim]['mean_edge_length'][active]))))
   log(f'Reused weights interval {t}: {r["umap2d"]["projection_valid"].sum():,} new 2D arrows')
  dest.attrs['method']='coordinate-only control; old transition probabilities may be nonlocal on new atlas';dest.attrs['rates']='unchanged fitted linear CPM/hour, no biological refitting';dest.attrs['point_semantics']='fixed genomic peak, not a cell or moving physical entity'
 after={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in [OLD,ATLAS]};assert before==after
 dump(OUT/'reused_validation.json',dict(status='passed',n_peaks=n,source_files_unchanged=True,input_signatures=before,interval_checks=checks));pd.DataFrame(counts).to_csv(OUT/'reused_projection_coverage.csv',index=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['reuse']);a=p.parse_args();reuse()
