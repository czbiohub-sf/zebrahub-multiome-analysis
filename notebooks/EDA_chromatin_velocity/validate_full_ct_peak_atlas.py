#!/usr/bin/env python3
"""Independent full-coverage and landmark-cache checks on saved full atlases."""
import json
import h5py,numpy as np
from ct_peak_atlas import OUT,DATA,dump

def main():
 summary={'passed':True,'checks':[],'dimensions':{}}
 with h5py.File(OUT/'full_ct_peak_atlas.h5','r') as h,h5py.File(DATA/'peaks_by_ct_tp_master.h5ad','r') as source,h5py.File(OUT/'aligned_features.h5','r') as sample:
  assert np.array_equal(h['peak_ids'][:],source['obs/_index'][:]);summary['n_peaks']=len(h['peak_ids']);summary['checks'].append('All640830 source peakIDs preserved exactly in originalorder.')
  idx=sample['source_indices'][:];assert np.array_equal(h['sample_source_indices'][:],idx)
  p=h['pca'][:];assert p.shape==(640830,50) and np.isfinite(p).all();order=np.argsort(idx);samp_pc=h['pca'][idx[order]][np.argsort(order)];expected=np.load(OUT/'coordinates/ct_log_2d.npz')['pca'];assert np.array_equal(samp_pc,expected)
  summary['sample_max_PCA_roundoff_before_preservation']=float(h.attrs['sample_max_pca_difference_before_exact_preservation'])
  for dim in [2,3]:
   x=h[f'umap{dim}d'][:];assert x.shape==(640830,dim) and np.isfinite(x).all();exp=np.load(OUT/'coordinates'/f'ct_log_{dim}d.npz')['umap'];assert np.array_equal(x[idx],exp)
   plan=np.load(OUT/'full_transform_plan.npz');pi=json.loads((OUT/'full_transform_plan.json').read_text());coords=np.full((pi['unique_vectors'],dim),np.nan,np.float32)
   import joblib
   cache=joblib.load(OUT/'models'/f'ct_log_{dim}d.joblib')['umap_cache'];coords[plan['known_codes']]=cache['reference_coordinates']
   for b,e in pi['chunk_boundaries']:
    part=np.load(OUT/'full_checkpoints'/f'{dim}d_{b:06d}_{e:06d}.npy');assert part.shape==(e-b,dim) and np.isfinite(part).all();coords[plan['unseen_codes'][b:e]]=part
   assert np.array_equal(x,coords[plan['query_codes']]);summary['dimensions'][str(dim)]={'coverage_fraction':1.0,'all_finite':True,'all25056_sample_coordinates_bitwise_equal':True,'exact_atomic_checkpoint_reassembly':True}
  for field in ['mean_cpm','ct_tau','stage_of_max_mean_cpm']:
   v=h['annotations'][field][:];assert np.isfinite(v).all();assert np.array_equal(v[idx],sample['annotations'][field][:])
  summary['checks'].append('Full annotations match sampled annotations exactly; allfullvalues finite.')
 prov=json.loads((OUT/'provenance.json').read_text())
 from pathlib import Path
 for name in ['ct_all','leiden_all']:
  f=Path(prov[name]['path']);assert f.stat().st_size==prov[name]['size'] and f.stat().st_mtime_ns==prov[name]['mtime_ns']
 summary['checks'].append('Both authoritative inputs stillunchanged.')
 dump(OUT/'validation_full_atlas.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
