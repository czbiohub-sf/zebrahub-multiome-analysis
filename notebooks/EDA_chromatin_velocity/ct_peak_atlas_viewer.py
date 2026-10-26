#!/usr/bin/env python3
"""Offline explorer for fixed peak atlases with measured CT-stage curves."""
from pathlib import Path
import base64,json,hashlib
import h5py,numpy as np,pandas as pd
import plotly.offline
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/ct_peak_atlas_v1';OLD=HERE/'outputs/state_trajectory_atlas_v1'
def pack(a):
 a=np.asarray(a,dtype='<f4');return {'shape':list(a.shape),'data':base64.b64encode(a.tobytes()).decode()}
def build():
 with h5py.File(OUT/'aligned_features.h5','r') as h:
  n=len(h['peak_ids']);extra=np.flatnonzero(~h['train_mask'][:].astype(bool)&~h['test_mask'][:].astype(bool))
  reps=pd.read_csv(OLD/'representative_peaks.csv');selected=np.unique(np.r_[np.random.default_rng(20261021).choice(n,8000,replace=False),extra,reps['sample_index'].to_numpy()]);np.save(OUT/'viewer_sample_indices.npy',selected)
  cts=h['ct_all/group'].asstr()[:];st=h['ct_all/stage'].asstr()[:];names=sorted(set(cts));stages=['0somites','5somites','10somites','15somites','20somites','30somites'];lookup=[[-1]*6 for _ in names]
  for j,(c,t) in enumerate(zip(cts,st)):lookup[names.index(c)][stages.index(t)]=j
  labels=h['leiden_coarse'].asstr()[:];cat=sorted(set(labels));codes=np.array([cat.index(x) for x in labels],np.float32)
  data={'title':'Celltype × stage fixed peak atlas','sample_indices':selected.tolist(),'peak_ids':h['peak_ids'].asstr()[:][selected].tolist(),'genes':h['nearest_gene'].asstr()[:][selected].tolist(),'ct_names':names,'times':[10,12,14,16,19,24],'feature_lookup':lookup,'feature_ids':h['ct_all/feature_ids'].asstr()[:].tolist(),'n_cells':h['ct_all/n_cells'][:].tolist(),'low_cell_threshold':20,'cluster_names':cat,'n_full_sample_peaks':n,'cpm':pack(h['ct_all/cpm'][:][selected]),'colors':{'mean':pack(np.log1p(h['annotations/mean_cpm'][:][selected])),'tau':pack(h['annotations/ct_tau'][:][selected]),'stage':pack(h['annotations/stage_of_max_mean_cpm'][:][selected]),'cluster':pack(codes[selected])},'models':{},'examples':[{'label':r.label,'index':int(np.flatnonzero(selected==r.sample_index)[0])} for r in reps.itertuples()]}
  data['models']['original']={'label':'Original saved Leiden-stage atlas','coordinates':{'2':pack(h['original_umap2d'][:][selected]),'3':pack(h['original_umap3d'][:][selected])},'note':'Original full-data map; saved identically in both source files. Rebuilt maps use 20,000 training peaks.'}
 settings=[('ct_log','Celltype-stage • log1p(CPM)',['ct_log_2d','ct_log_3d']),('ct_linear','Celltype-stage • linear CPM',['ct_linear_2d']),('ct_min20','Celltype-stage • ≥20 cells per column',['ct_min20_log_2d']),('leiden_log','Rebuilt Leiden-stage • log1p(CPM)',['leiden_log_2d']),('ct_broad','Celltype-stage • broader UMAP',['ct_log_broad_2d']),('ct_seed2','Celltype-stage • second seed',['ct_log_seed2_2d']),('ct_allpc','Celltype-stage • all 190 PCs',['ct_log_allpc_2d'])]
 for key,label,files in settings:
  coords={};notes=[];diagnostics={}
  for file in files:
   a=np.load(OUT/'coordinates'/f'{file}.npz');dim=a['umap'].shape[1];coords[str(dim)]=pack(a['umap'][selected]);j=json.loads((OUT/'coordinates'/f'{file}.json').read_text());diagnostics[str(dim)]=j['heldout_umap_diagnostics'];notes.append(f"{dim}D: {j['n_features']} features, {j['n_pcs']} PCs ({100*j['pca_explained_variance_fraction']:.1f}% training variance), k={j['n_neighbors']}, min_dist={j['min_dist']}.")
  data['models'][key]={'label':label,'coordinates':coords,'diagnostics':diagnostics,'note':' '.join(notes)}
 template=(HERE/'ct_peak_atlas_viewer_template.html').read_text();payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/')
 html=template.replace('__PLOTLY__',plotly.offline.get_plotlyjs()).replace('__DATA__',payload);path=OUT/'ct_peak_atlas_explorer.html';path.write_text(html)
 manifest={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n_viewer_peaks':len(selected),'n_full_analysis_peaks':n,'n_features':190,'n_models':len(data['models']),'numeric_encoding':'Little-endian float32, exact to saved analysis arrays; base64 only, no numeric rounding.','selection':'Fixed random8000 plus all marker extras and six representative peaks. Search covers this subset; all25056 analysis coordinates stay available locally.'}
 (OUT/'viewer_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':build()
