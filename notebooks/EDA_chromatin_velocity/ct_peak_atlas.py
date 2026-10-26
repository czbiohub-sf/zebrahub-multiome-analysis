#!/usr/bin/env python3
"""Matched fixed-peak atlases from CT-stage versus Leiden-stage pseudobulks.

One point is one peak summarizing the entire developmental course. This is
separate from the shared peak-by-stage state atlas. Train/test are whole peaks;
feature normalization uses the whole original library totals, not sampled peaks.
"""
from pathlib import Path
import argparse, json, os, time
import h5py, joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from state_atlas_umap import exact_cached_embedding, neighborhood_diagnostics
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_peak_atlas_v1'
OLD=HERE/'outputs/state_trajectory_atlas_v1'
DATA=Path('/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2')
SEED=20261021
SPECS={
 'ct_log_2d':('ct_all','log1p_cpm',2,50,30,.3,SEED),
 'ct_log_3d':('ct_all','log1p_cpm',3,50,30,.3,SEED),
 'ct_linear_2d':('ct_all','cpm',2,50,30,.3,SEED),
 'ct_min20_log_2d':('ct_min20','log1p_cpm',2,50,30,.3,SEED),
 'leiden_log_2d':('leiden_all','log1p_cpm',2,50,30,.3,SEED),
 'ct_log_broad_2d':('ct_all','log1p_cpm',2,50,60,.6,SEED),
 'ct_log_seed2_2d':('ct_all','log1p_cpm',2,50,30,.3,SEED+1),
 'ct_log_allpc_2d':('ct_all','log1p_cpm',2,190,30,.3,SEED),
}
def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def dump(path,obj):
 def conv(v):
  if isinstance(v,np.ndarray): return v.tolist()
  if isinstance(v,np.generic): return v.item()
  raise TypeError(str(type(v)))
 Path(path).write_text(json.dumps(obj,indent=2,default=conv,allow_nan=False)+'\n')
def strings(a): return np.asarray([x.decode() if isinstance(x,bytes) else str(x) for x in a])
def col(h,key):
 o=h[key]
 if isinstance(o,h5py.Dataset):
  a=o[:];return strings(a) if a.dtype.kind in 'SO' else a
 cats=strings(o['categories'][:]);codes=o['codes'][:]
 return np.asarray([cats[i] if i>=0 else '' for i in codes])
def rows(ds,idx):
 out=np.empty((len(idx),ds.shape[1]),dtype=ds.dtype)
 for start in range(0,ds.shape[0],32768):
  select=np.flatnonzero((idx>=start)&(idx<start+32768))
  if len(select): out[select]=ds[start:start+32768][idx[select]-start]
 return out

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'models').mkdir(exist_ok=True);(OUT/'coordinates').mkdir(exist_ok=True)
 with h5py.File(OLD/'sample_data.h5','r') as s:
  ids=strings(s['peak_ids'][:]);idx=s['source_indices'][:];train=s['train_mask'][:].astype(bool);test=s['test_mask'][:].astype(bool)
 meta=pd.read_csv(OLD/'sample_metadata.csv')
 log('Reading and index-aligning raw pseudobulks')
 paths={'ct_all':DATA/'peaks_by_ct_tp_master.h5ad','leiden_all':DATA/'peaks_by_pb_annotated_master.h5ad'}
 states={};prov={}
 with h5py.File(OUT/'aligned_features.h5','w') as dest:
  for k,a in [('peak_ids',ids),('source_indices',idx),('train_mask',train),('test_mask',test)]:
   dest.create_dataset(k,data=a.astype(object) if a.dtype.kind=='U' else a,dtype=h5py.string_dtype() if a.dtype.kind=='U' else None)
  for name,path in paths.items():
   with h5py.File(path,'r') as h:
    origids=col(h,'obs/_index');assert np.array_equal(origids[idx],ids)
    raw=rows(h['layers/sum'],idx).astype(np.float32);depth=h['var/total_coverage'][:].astype(float);ncells=h['var/n_cells'][:]
    assert np.all(depth>0) and np.isfinite(raw).all() and np.all(raw>=0)
    cpm=(raw/depth*1e6).astype(np.float32);ln=np.log1p(cpm)
    features=col(h,'var/_index');stages=col(h,'var/dev_stage');group=col(h,'var/annotation_ML_coarse' if name=='ct_all' else 'var/leiden_0.4_merged')
    states[name]={'raw_counts':raw,'cpm':cpm,'log1p_cpm':ln,'feature_ids':features,'stage':stages,'group':group,'n_cells':ncells,'total_coverage':depth}
    prov[name]={'path':str(path),'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'n_full_peaks':len(origids),'n_features':len(features),'source_common_scale_factor':float(h['uns/common_scale_factor'][()]),'n_columns_below20cells':int((ncells<20).sum())}
    if name=='leiden_all':
     dest.create_dataset('original_umap2d',data=rows(h['obsm/X_umap_2D'],idx))
     dest.create_dataset('original_umap3d',data=rows(h['obsm/X_umap_3D'],idx))
     coarse=col(h,'obs/leiden_coarse')[idx];dest.create_dataset('leiden_coarse',data=coarse.astype(object),dtype=h5py.string_dtype())
     nearest=col(h,'obs/nearest_gene')[idx];dest.create_dataset('nearest_gene',data=nearest.astype(object),dtype=h5py.string_dtype())
  mask=states['ct_all']['n_cells']>=20
  states['ct_min20']={k:(a[:,mask] if a.ndim==2 else a[mask]) for k,a in states['ct_all'].items()}
  for name,d in states.items():
   g=dest.create_group(name)
   for k,a in d.items():
    g.create_dataset(k,data=a.astype(object) if a.dtype.kind=='U' else a,dtype=h5py.string_dtype() if a.dtype.kind=='U' else None,compression='gzip')
   pd.DataFrame({k:a for k,a in d.items() if a.ndim==1}).to_csv(OUT/f'{name}_features.csv',index=False)
  allct=states['ct_all'];ctnames=np.unique(allct['group'])
  means=np.column_stack([allct['cpm'][:,allct['group']==ct].mean(axis=1) for ct in ctnames])
  maximum=means.max(axis=1);tau=np.divide((1-np.divide(means,maximum[:,None],out=np.zeros_like(means),where=maximum[:,None]>0)).sum(axis=1),len(ctnames)-1)
  tau[maximum==0]=0
  annotation=dest.create_group('annotations');annotation.create_dataset('mean_cpm',data=allct['cpm'].mean(axis=1));annotation.create_dataset('ct_tau',data=tau)
  # Six stage summaries average present celltypes; these are descriptive colors.
  stagenames=np.array(['0somites','5somites','10somites','15somites','20somites','30somites'])
  bystage=np.column_stack([allct['cpm'][:,allct['stage']==st].mean(axis=1) for st in stagenames])
  annotation.create_dataset('stage_of_max_mean_cpm',data=np.array([10,12,14,16,19,24])[bystage.argmax(axis=1)])
  # Source matrix equivalence is tested rather than inferred from column names.
  a,b=states['ct_all'],states['leiden_all'];corr=np.corrcoef(a['raw_counts'].T,b['raw_counts'].T)[:190,190:];ii,jj=linear_sum_assignment(-corr)
  eq=np.array([np.array_equal(a['raw_counts'][:,i],b['raw_counts'][:,j]) for i,j in zip(ii,jj)])
  match=pd.DataFrame({'ct_feature':a['feature_ids'][ii],'leiden_feature':b['feature_ids'][jj],'raw_correlation':corr[ii,jj],'exact_equal_sample_counts':eq,'same_full_library_total':a['total_coverage'][ii]==b['total_coverage'][jj]})
  match.to_csv(OUT/'source_feature_matching.csv',index=False)
  pool=[]
  for st in ['0somites','5somites','10somites','15somites','20somites','30somites']:
   ma=a['stage']==st;mb=b['stage']==st
   pool.append({'stage':st,'ct_cells':int(a['n_cells'][ma].sum()),'leiden_cells':int(b['n_cells'][mb].sum()),'ct_depth':float(a['total_coverage'][ma].sum()),'leiden_depth':float(b['total_coverage'][mb].sum()),'sample_peak_stage_raw_totals_identical':bool(np.array_equal(a['raw_counts'][:,ma].sum(axis=1),b['raw_counts'][:,mb].sum(axis=1)))})
  dump(OUT/'source_stage_pool_comparison.json',pool)
  prov['source_matrix_comparison']={'sample_exactly_equal_columns_after_best_correlation_matching':int(eq.sum()),'n_features':190,'median_matched_raw_correlation':float(np.median(corr[ii,jj])),'minimum_matched_raw_correlation':float(np.min(corr[ii,jj])),'not_a_column_permutation':bool(not eq.all())}
  with h5py.File(paths['ct_all'],'r') as a,h5py.File(paths['leiden_all'],'r') as b:
   prov['source_saved_embeddings_identical']={key:bool(np.array_equal(a['obsm'][key][:],b['obsm'][key][:])) for key in ['X_pca','X_umap_2D','X_umap_3D']}
  dest.attrs['normalization']='CPM=raw_count/original_full_peak_library_total*1e6; optional log1p; no per-peak or feature variance scaling.'
  dest.attrs['row_semantics']='One fixed peak represented by its full observed course across existing pseudobulk columns.'
  dest.attrs['missing_combo_handling']='Absent source columns remain absent; min20 sensitivity removes14 measured but small pseudobulks. No imputation.'
 prov['sample']={'n_peaks':len(ids),'n_train':int(train.sum()),'n_test':int(test.sum()),'n_marker_extras':int((~train&~test).sum()),'same_sample_as_state_atlas':True}
 prov['normalization']='Rawcounts/whole_original_pseudobulk_depth*1e6. Log=natural log(1+CPM). No variance scaling.'
 prov['n_ct_min20_features']=int(mask.sum());dump(OUT/'provenance.json',prov)
 meta.to_csv(OUT/'sample_metadata.csv',index=False)
 log('Preparation complete')

def run(name):
 panel,transform,dim,npc,k,md,seed=SPECS[name]
 with h5py.File(OUT/'aligned_features.h5','r') as h:
  x=h[panel][transform][:];train=h['train_mask'][:].astype(bool);test=h['test_mask'][:].astype(bool)
 log(f'{name}: frozen PCA on observed training peaks, {x.shape}')
 pca=PCA(n_components=min(npc,x.shape[1]),svd_solver='full',whiten=False).fit(x[train].astype(float))
 coords=pca.transform(x).astype(np.float32)
 y,_,cache,info=exact_cached_embedding(coords[train],coords,dim,k,md,seed)
 evalindices=np.flatnonzero(test)[np.random.default_rng(SEED).choice(test.sum(),1500,replace=False)]
 diagnostic=neighborhood_diagnostics(x[evalindices],y[evalindices],maximum=1500,seed=SEED)
 pdiag=neighborhood_diagnostics(x[evalindices],coords[evalindices],maximum=1500,seed=SEED)
 pairs=pdist(x[evalindices[:800]].astype(float));pcpairs=pdist(coords[evalindices[:800]].astype(float))
 info.update(name=name,panel=panel,transform=transform,n_features=x.shape[1],n_pcs=coords.shape[1],pca_explained_variance_fraction=float(pca.explained_variance_ratio_.sum()),heldout_pca_pairwise_distance_spearman=float(spearmanr(pairs,pcpairs).statistic),heldout_umap_diagnostics=diagnostic,heldout_pca_diagnostics=pdiag,normalization='CPM based on full original library totals; optional natural log1p. Global feature centering in PCA; no whitening or variance scaling.')
 # Preserving all training and test rows permits later arrow projection on this fixed atlas.
 np.savez_compressed(OUT/'coordinates'/f'{name}.npz',umap=y,pca=coords,components=pca.components_,mean=pca.mean_,explained_variance_ratio=pca.explained_variance_ratio_,evaluation_peak_indices=evalindices)
 joblib.dump({'pca':pca,'umap_cache':cache,'panel':panel,'transform':transform},OUT/'models'/f'{name}.joblib',compress=3)
 dump(OUT/'coordinates'/f'{name}.json',info)
 log(f'{name}: completed')

def summary():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
 with h5py.File(OUT/'aligned_features.h5','r') as h:
  mean=h['annotations/mean_cpm'][:];tau=h['annotations/ct_tau'][:];coarse=strings(h['leiden_coarse'][:]);train=h['train_mask'][:].astype(bool);test=h['test_mask'][:].astype(bool);original=h['original_umap2d'][:];stagemax=h['annotations/stage_of_max_mean_cpm'][:]
 xy={};data=[]
 for name in SPECS:
  if not (OUT/'coordinates'/f'{name}.npz').exists(): continue
  a=np.load(OUT/'coordinates'/f'{name}.npz');xy[name]=a['umap'];j=json.loads((OUT/'coordinates'/f'{name}.json').read_text());data.append({'name':name,'n_features':j['n_features'],'n_pcs':j['n_pcs'],'pca_variance_fraction':j['pca_explained_variance_fraction'],**j['heldout_umap_diagnostics']})
 pd.DataFrame(data).to_csv(OUT/'model_diagnostics.csv',index=False)
 order=np.random.default_rng(SEED).permutation(len(mean));cats=np.unique(coarse);codes=np.searchsorted(cats,coarse)
 charts=[('original',original,'Original Leiden-stage atlas (saved)'),('leiden_log_2d',xy.get('leiden_log_2d'),'Rebuilt Leiden-stage • log1p(CPM)'),('ct_linear_2d',xy.get('ct_linear_2d'),'Celltype-stage • linear CPM'),('ct_log_2d',xy.get('ct_log_2d'),'Celltype-stage • log1p(CPM)'),('ct_min20_log_2d',xy.get('ct_min20_log_2d'),'Celltype-stage • ≥20 cells/column'),('ct_log_broad_2d',xy.get('ct_log_broad_2d'),'Celltype-stage • broader UMAP')]
 charts=[x for x in charts if x[1] is not None]
 fig,axes=plt.subplots(len(charts),3,figsize=(15,3.3*len(charts)),squeeze=False)
 for r,(name,y,title) in enumerate(charts):
  for c,(color,label,cmap) in enumerate([(np.log1p(mean),'Mean accessibility: log1p(CPM)','viridis'),(tau,'Celltype specificity τ','magma'),(codes,'Original coarse peak clusters','turbo')]):
   ax=axes[r,c];kw={'vmin':0,'vmax':1} if c==1 else {}
   sc=ax.scatter(y[order,0],y[order,1],c=color[order],cmap=cmap,s=2.5,alpha=.7,rasterized=True,linewidths=0,**kw);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title+'\n'+label,fontsize=10)
   if c<2: fig.colorbar(sc,ax=ax,shrink=.65)
 fig.suptitle('Fixed peak atlases: same 25,056 peaks; different pseudobulk features and transforms\nOne point = one peak across the whole course. Separate fits have arbitrary positions and orientations.',fontsize=14)
 fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'01_fixed_peak_atlas_comparison.png');fig.savefig(OUT/'01_fixed_peak_atlas_comparison.pdf');plt.close(fig)
 if 'ct_log_3d' in xy:
  y=xy['ct_log_3d'];fig=plt.figure(figsize=(16,5));colors=[(np.log1p(mean),'Mean accessibility: log1p(CPM)','viridis'),(tau,'Celltype specificity τ','magma'),(stagemax,'Stage of maximum mean CPM','viridis')]
  for p,(color,title,cmap) in enumerate(colors,1):
   ax=fig.add_subplot(1,3,p,projection='3d');sc=ax.scatter(y[order,0],y[order,1],y[order,2],c=color[order],cmap=cmap,s=2,alpha=.5,rasterized=True,linewidths=0);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]);fig.colorbar(sc,ax=ax,shrink=.55)
  fig.suptitle('Celltype × stage fixed-peak UMAP • log1p(CPM), 50 PCs, k=30, min_dist=0.3\nNo root or temporal direction imposed. Stage colors average available celltypes.',fontsize=13);fig.tight_layout();fig.savefig(OUT/'02_ct_peak_atlas_3d.png');fig.savefig(OUT/'02_ct_peak_atlas_3d.pdf');plt.close(fig)
 # Stability is evaluated on identical held-out rows and is not evidence of biology.
 stability=[]
 if 'ct_log_2d' in xy:
  ev=np.load(OUT/'coordinates/ct_log_2d.npz')['evaluation_peak_indices'];base=xy['ct_log_2d'][ev]
  nn=NearestNeighbors(n_neighbors=16).fit(base).kneighbors(base,return_distance=False)[:,1:]
  for name in ['ct_log_seed2_2d','ct_log_broad_2d','ct_log_allpc_2d','ct_min20_log_2d','ct_linear_2d','leiden_log_2d']:
   if name not in xy: continue
   y=xy[name][ev];n2=NearestNeighbors(n_neighbors=16).fit(y).kneighbors(y,return_distance=False)[:,1:]
   stability.append({'comparison':name,'n_heldout_peaks':len(ev),'neighbor15_overlap':float(np.mean([len(set(a)&set(b))/15 for a,b in zip(nn,n2)])),'pairwise_distance_spearman':float(spearmanr(pdist(base[:800]),pdist(y[:800])).statistic)})
 pd.DataFrame(stability).to_csv(OUT/'embedding_stability.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(15,5));d=pd.DataFrame(data);labels=d['name'].str.replace('_2d','',regex=False).str.replace('_',' ',regex=False)
 for ax,metric,title in zip(axes,['pca_variance_fraction','neighborhood_recall_mean','pairwise_distance_spearman'],['Training PCA variance retained','Held-out 15-neighbor recall','Held-out pairwise-distance rank correlation']):
  ax.barh(labels,d[metric]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(title,fontsize=11);ax.set_xlabel('Fraction / correlation')
 fig.suptitle('Embedding fidelity measured against each model’s own full feature space\nScores across feature representations describe fidelity, not which biological representation is best.',fontsize=13);fig.tight_layout(rect=(0,0,1,.9));fig.savefig(OUT/'03_embedding_diagnostics.png');fig.savefig(OUT/'03_embedding_diagnostics.pdf');plt.close(fig)
 # Standalone 3D HTML uses local Plotly to remain viewable offline.
 if 'ct_log_3d' in xy:
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots
  with h5py.File(OUT/'aligned_features.h5','r') as h: ids=strings(h['peak_ids'][:]);genes=strings(h['nearest_gene'][:])
  y=xy['ct_log_3d'];fig=go.Figure(go.Scatter3d(x=y[:,0],y=y[:,1],z=y[:,2],mode='markers',marker={'size':2,'color':tau,'colorscale':'Magma','opacity':.65,'colorbar':{'title':'CT specificity τ'}},customdata=np.column_stack([ids,genes,mean,tau,coarse]),hovertemplate='Peak %{customdata[0]}<br>Gene %{customdata[1]}<br>Mean CPM %{customdata[2]}<br>CT τ %{customdata[3]}<br>Old coarse cluster %{customdata[4]}<extra></extra>'))
  fig.update_layout(updatemenus=[{'buttons':[{'label':'CT specificity τ','method':'restyle','args':[{'marker.color':[tau],'marker.colorscale':'Magma','marker.colorbar.title':'CT specificity τ'}]},{'label':'Mean accessibility','method':'restyle','args':[{'marker.color':[np.log1p(mean)],'marker.colorscale':'Viridis','marker.colorbar.title':'log1p(mean CPM)'}]},{'label':'Stage of max accessibility','method':'restyle','args':[{'marker.color':[stagemax],'marker.colorscale':'Viridis','marker.colorbar.title':'Stage (hpf)'}]},{'label':'Original coarse clusters','method':'restyle','args':[{'marker.color':[codes],'marker.colorscale':'Turbo','marker.colorbar.title':'Original cluster code'}]}],'x':0,'y':1.07}])
  fig.update_layout(title='Fixed peak atlas: celltype × stage log1p(CPM) • one point = one peak<br><sup>No arrows or root direction imposed; separate from the peak × stage state atlas.</sup>',height=850,scene={'xaxis_title':'UMAP 1','yaxis_title':'UMAP 2','zaxis_title':'UMAP 3'})
  fig.write_html(OUT/'ct_peak_atlas_3d.html',include_plotlyjs=True,full_html=True)
 log('Summary figures and tables complete')

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','summary']);p.add_argument('--model',choices=list(SPECS));a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='run':run(a.model)
 else:summary()
if __name__=='__main__':main()
