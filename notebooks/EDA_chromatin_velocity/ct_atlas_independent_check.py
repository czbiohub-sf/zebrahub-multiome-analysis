"""Independent array, basis and split checks for the rebuilt fixed peak atlas."""
from pathlib import Path
import json,h5py,numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent/'outputs/ct_peak_atlas_v1';OLD=OUT.parent/'state_trajectory_atlas_v1';SPLIT=OUT.parent/'ct_flow_noise_v1/cell_split_input.h5'

def main():
 results=[]
 with h5py.File(OUT/'aligned_features.h5') as f,h5py.File(OLD/'sample_data.h5') as old,h5py.File(SPLIT) as split:
  ids=f['peak_ids'].asstr()[:];assert np.array_equal(ids,old['peak_ids'].asstr()[:]);assert np.array_equal(ids,split['peak_ids'].asstr()[:]);train=f['train_mask'][:];test=f['test_mask'][:];assert train.sum()==20000 and test.sum()==5000 and not (train&test).any();assert np.array_equal(f['source_indices'][:],split['source_indices'][:])
  counts=split['raw_counts_a'][:]+split['raw_counts_b'][:];cov=split['coverage_a'][:]+split['coverage_b'][:];ct=list(split['celltypes'].asstr()[:]);stages={'0somites':0,'5somites':1,'10somites':2,'15somites':3,'20somites':4,'30somites':5};meta=pd.read_csv(OUT/'ct_all_features.csv')
  expected=np.stack([counts[:,stages[r.stage],ct.index(r.group)] for r in meta.itertuples()],axis=1);assert np.array_equal(expected,f['ct_all/raw_counts'][:]);assert np.array_equal(meta.total_coverage.to_numpy(),np.asarray([cov[ct.index(r.group),stages[r.stage]] for r in meta.itertuples()]))
  for panel in ['ct_all','ct_min20','leiden_all']:
   g=f[panel];x=g['raw_counts'][:].astype(float)*1e6/g['total_coverage'][:][None];assert np.allclose(x,g['cpm'][:],atol=1e-5,rtol=1e-6);assert np.allclose(np.log1p(g['cpm'][:]),g['log1p_cpm'][:],atol=1e-6)
  for p in sorted((OUT/'coordinates').glob('*.npz')):
   info=json.loads(p.with_suffix('.json').read_text());a=np.load(p);x=f[info['panel']][info['transform']][:].astype(float);assert a['umap'].shape==(len(ids),info['dimensions']) and np.isfinite(a['umap']).all();assert test[a['evaluation_peak_indices']].all()
   orth=np.max(np.abs(a['components']@a['components'].T-np.eye(len(a['components']))));assert orth<1e-10
   meanerr=np.max(np.abs(a['mean']-x[train].mean(axis=0)));assert meanerr<1e-10
   projection=(x-a['mean'])@a['components'].T;error=np.max(np.abs(projection-a['pca']));assert np.allclose(projection,a['pca'],atol=1e-4,rtol=1e-6)
   results.append(dict(model=p.stem,n_peaks=len(ids),dimensions=info['dimensions'],finite_coordinates=True,training_mean_error=float(meanerr),orthogonality_error=float(orth),max_float32_projection_error=float(error),heldout_indices_verified=True))
 result=dict(status='passed',all_peak_ids_aligned=True,all_celltype_raw_counts_reconstructed_from_cell_halves=True,no_train_test_overlap=True,normalization_verified=True,n_models=len(results),models=results)
 (OUT/'independent_array_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
