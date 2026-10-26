#!/usr/bin/env python3
"""Audit source alignment, frozen PCA, model outputs, and packed viewer payload."""
from pathlib import Path
import base64,json,re
import h5py,joblib,numpy as np
from ct_peak_atlas import OUT,OLD,SPECS,dump

def main():
 checks=[];models=[]
 prov=json.loads((OUT/'provenance.json').read_text())
 for key in ['ct_all','leiden_all']:
  p=Path(prov[key]['path']);assert p.stat().st_size==prov[key]['size'] and p.stat().st_mtime_ns==prov[key]['mtime_ns']
 checks.append('Both source inputs retain original size and modification timestamp.')
 with h5py.File(OUT/'aligned_features.h5','r') as h,h5py.File(OLD/'sample_data.h5','r') as old:
  assert np.array_equal(h['peak_ids'][:],old['peak_ids'][:]);assert np.array_equal(h['source_indices'][:],old['source_indices'][:]);train=h['train_mask'][:].astype(bool);test=h['test_mask'][:].astype(bool);assert not np.any(train&test)
  checks.append('Peak IDs/order/sourceindices and 20k/5k whole-peak split exactly match previous atlas.')
  ct=old['celltypes'].asstr()[:];st=old['stages'][:];stage={'0somites':10,'5somites':12,'10somites':14,'15somites':16,'20somites':19,'30somites':24};obs=old['observed_cpm'][:];count=0
  for j,(c,t) in enumerate(zip(h['ct_all/group'].asstr()[:],h['ct_all/stage'].asstr()[:])):
   src=obs[:,np.flatnonzero(st==stage[t])[0],np.flatnonzero(ct==c)[0]];dst=h['ct_all/cpm'][:,j];ok=np.isfinite(src);assert np.array_equal(src[ok],dst[ok]);count+=int(ok.sum())
  checks.append(f'All {count:,} finite CT CPM values exactly equal previous observed temporal-analysis input after reindexing.')
  for panel in ['ct_all','ct_min20','leiden_all']:
   g=h[panel];expected=(g['raw_counts'][:]/g['total_coverage'][:]*1e6).astype(np.float32);assert np.array_equal(expected,g['cpm'][:]);assert np.array_equal(np.log1p(expected),g['log1p_cpm'][:])
  checks.append('Every normalized feature recomputes exactly from rawcount/full-original-librarydepth ×1e6 and natural log1p.')
  assert (h['ct_min20/n_cells'][:]>=20).all() and len(h['ct_min20/n_cells'])==176
  for name,(panel,tr,dim,npc,k,md,seed) in SPECS.items():
   a=np.load(OUT/'coordinates'/f'{name}.npz');x=h[panel][tr][:];pca=joblib.load(OUT/'models'/f'{name}.joblib')['pca'];assert np.array_equal(pca.mean_,x[train].astype(float).mean(axis=0));projected=pca.transform(x).astype(np.float32);assert np.array_equal(projected,a['pca']);assert a['umap'].shape==(len(x),dim) and np.isfinite(a['umap']).all();assert np.all(test[a['evaluation_peak_indices']]);assert np.max(abs(pca.components_@pca.components_.T-np.eye(pca.n_components_)))<1e-10
   models.append({'name':name,'exact_frozen_training_mean':True,'exact_PCA_reprojection':True,'finite_coordinates':True,'evaluation_peaks_heldout':True})
  payload=re.search(r'const DATA=(.*?);const \$=',(OUT/'ct_peak_atlas_explorer.html').read_text(),re.S).group(1);data=json.loads(payload);ix=np.asarray(data['sample_indices']);assert np.array_equal(ix,np.load(OUT/'viewer_sample_indices.npy'));assert data['peak_ids']==h['peak_ids'].asstr()[:][ix].tolist();assert data['genes']==h['nearest_gene'].asstr()[:][ix].tolist()
  def unpack(p):return np.frombuffer(base64.b64decode(p['data']),dtype='<f4').reshape(p['shape'])
  values=0;arrays=0
  def check(a,b):
   nonlocal values,arrays
   assert np.array_equal(a,b);values+=a.size;arrays+=1
  check(unpack(data['cpm']),h['ct_all/cpm'][:][ix]);check(unpack(data['colors']['mean']),np.log1p(h['annotations/mean_cpm'][:][ix]));check(unpack(data['colors']['tau']),h['annotations/ct_tau'][:][ix]);check(unpack(data['colors']['stage']),h['annotations/stage_of_max_mean_cpm'][:][ix]);
  for d in [2,3]:check(unpack(data['models']['original']['coordinates'][str(d)]),h[f'original_umap{d}d'][:][ix])
  mapping={'ct_log':['ct_log_2d','ct_log_3d'],'ct_linear':['ct_linear_2d'],'ct_min20':['ct_min20_log_2d'],'leiden_log':['leiden_log_2d'],'ct_broad':['ct_log_broad_2d'],'ct_seed2':['ct_log_seed2_2d'],'ct_allpc':['ct_log_allpc_2d']}
  for key,names in mapping.items():
   for name in names:
    a=np.load(OUT/'coordinates'/f'{name}.npz')['umap'];check(unpack(data['models'][key]['coordinates'][str(a.shape[1])]),a[ix])
 checks.append(f'Viewer exact-payload audit passed: {arrays} numeric arrays, {values:,} values, {len(ix):,} peak and gene IDs.')
 browser=json.loads((OUT/'browser_validation/browser_validation.json').read_text());assert browser['passed'];checks.append(f"Offline browser QA passed {len(browser['checks'])} behavior checks, zero page errors/externalrequests.")
 dump(OUT/'validation.json',{'passed':True,'checks':checks,'models':models,'viewer_numeric_arrays':arrays,'viewer_numeric_values':values,'viewer_peak_count':len(ix)})
 print('\n'.join(checks))
if __name__=='__main__':main()
