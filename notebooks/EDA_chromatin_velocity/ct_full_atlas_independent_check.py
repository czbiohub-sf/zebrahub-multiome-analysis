"""Independent full-atlas alignment, source transformation and coordinate audit."""
from pathlib import Path
import json,h5py,joblib,numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/ct_peak_atlas_v1'
prov=json.loads((OUT/'provenance.json').read_text())
for key in ['ct_all','leiden_all']:
 p=Path(prov[key]['path']);s=p.stat();assert s.st_size==prov[key]['size'] and s.st_mtime_ns==prov[key]['mtime_ns']
rng=np.random.default_rng(20261022);check=np.sort(rng.choice(640830,2048,replace=False));pca=joblib.load(OUT/'models/ct_log_2d.joblib')['pca']
with h5py.File(prov['ct_all']['path']) as src,h5py.File(OUT/'full_ct_peak_atlas.h5') as f,h5py.File(OUT/'aligned_features.h5') as sample:
 ids=f['peak_ids'].asstr()[:];assert len(ids)==640830 and len(set(ids))==len(ids)
 assert np.array_equal(ids,src['obs/_index'].asstr()[:]);ix=f['sample_source_indices'][:]
 assert np.array_equal(ix,sample['source_indices'][:]);assert np.array_equal(ids[ix],sample['peak_ids'].asstr()[:])
 coords={}
 for dim in [2,3]:
  y=f[f'umap{dim}d'][:];assert y.shape==(640830,dim) and np.isfinite(y).all();assert np.array_equal(y,np.load(OUT/f'full_umap{dim}d.npy'))
  expected=np.load(OUT/f'coordinates/ct_log_{dim}d.npz')['umap'];assert np.array_equal(y[ix],expected);coords[str(dim)]={'shape':list(y.shape),'finite':True,'sample_bitwise_preserved':True,'npy_h5_exact_match':True}
 raw=src['layers/sum'][check].astype('f4');cpm=(raw/src['var/total_coverage'][:]*1e6).astype('f4');x=np.log1p(cpm)
 expected=(x.astype(float)-pca.mean_)@pca.components_.T
 actual=f['pca'][check];maxerr=float(np.abs(expected-actual).max());assert np.allclose(expected,actual,atol=1e-5,rtol=2e-6)
 assert np.allclose(cpm.mean(axis=1),f['annotations/mean_cpm'][check],atol=1e-6)
 for begin in range(0,len(ids),32768):assert np.isfinite(f['pca'][begin:begin+32768]).all()
 result=dict(status='passed',n_peaks=len(ids),peak_order_equals_source=True,unique_peak_ids=True,source_files_size_mtime_unchanged=True,coordinates=coords,independent_source_pca_rows=len(check),source_pca_max_absolute_error=maxerr,mean_cpm_reconstructed=True,pca_finite=True,model_training_peaks=20000,feature_count=190,pca_components=50,note='Frozen landmark-trained models; audit does not claim full-data model fitting or fine-layout biological validity.')
(OUT/'full_independent_array_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
