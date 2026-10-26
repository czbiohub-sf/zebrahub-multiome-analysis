"""Prepare observed peak-stage states, whole-peak splits, and shared PCA models.

State geometry uses observed accessibility; fitted profiles use the same frozen
basis. Peak activity support is metadata, never an all-celltype entry filter.
"""
from pathlib import Path
import json,hashlib,time
import h5py,numpy as np,pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'
OUT=HERE/'outputs/state_trajectory_atlas_v1';SEED=20261020

def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def take(dataset,idx):
 order=np.argsort(idx);inverse=np.argsort(order)
 return dataset[np.asarray(idx)[order]][inverse]
def write_strings(f,key,values):f.create_dataset(key,data=np.asarray(values,object),dtype=h5py.string_dtype(),compression='gzip')
def signature(p):s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def specificity(x):
 maximum=x.max(axis=-1);den=x.shape[-1]-1
 return np.divide(np.sum(maximum[...,None]-x,axis=-1),den*maximum,out=np.full_like(maximum,np.nan),where=maximum>0)
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);before=signature(SOURCE)
 metadata=pd.read_csv(SOURCE.parent/'peak_metadata.csv.gz',index_col=0,keep_default_na=False)
 with h5py.File(SOURCE,'r') as f:
  allids=f['peak_ids'].asstr()[:];assert np.array_equal(allids,metadata.index.astype(str))
  rng=np.random.default_rng(SEED);random_ids=rng.choice(len(allids),25000,replace=False);selected=list(random_ids);seen=set(selected)
  genes=['slc4a1a','cdx1a','pax6a','pax6b','rx3','six3b','phox2bb','elavl3','neurog1','gata1a','sox10','foxd3']
  annotations=metadata[['nearest_gene','linked_gene']].astype(str).agg(' '.join,axis=1)
  markers={}
  for gene in genes:
   candidates=np.flatnonzero(annotations.str.contains(r'\b'+gene+r'\b',case=False,regex=True).to_numpy())
   chosen=candidates[np.linspace(0,len(candidates)-1,min(5,len(candidates))).astype(int)] if len(candidates) else []
   markers[gene]=allids[chosen].tolist()
   for i in chosen:
    if i not in seen:selected.append(int(i));seen.add(i)
  idx=np.asarray(selected);n=len(idx);train=np.arange(n)<20000;test=(np.arange(n)>=20000)&(np.arange(n)<25000)
  sub=metadata.iloc[idx].copy();sub.insert(0,'peak_id',allids[idx]);sub.insert(1,'source_index',idx);sub.insert(2,'sample_index',np.arange(n));sub['split']=np.where(train,'train',np.where(test,'test','marker'))
  sub.to_csv(OUT/'sample_metadata.csv',index=False)
  temporary=OUT/'sample_data.partial.h5'
  with h5py.File(temporary,'w') as dest:
   write_strings(dest,'peak_ids',allids[idx]);write_strings(dest,'celltypes',f['celltypes'].asstr()[:])
   for key,x in [('source_indices',idx),('train_mask',train),('test_mask',test),('stages',f['stages'][:]),('condition_valid',f['condition_valid'][:]),('common_celltypes',f['common_celltypes'][:])]:dest.create_dataset(key,data=x)
   for key in ['observed_cpm','fitted_cpm','rates','rate_supported','rate_se']:
    x=take(f[key],idx).transpose(0,2,1);dest.create_dataset(key,data=x,compression='gzip',compression_opts=1);log('Copied '+key)
   for key in ['fit_valid','lack_of_fit','modes','raw_total','broad_reference','umap2d','umap3d']:dest.create_dataset(key,data=take(f[key],idx),compression='gzip',compression_opts=1)
   dest.attrs['source_json']=json.dumps(before);dest.attrs['sampling_seed']=SEED
   dest.attrs['axis_order']='peak,stage,celltype (rates use adjacent intervals)'
   dest.attrs['interpretation']='Observed CPM geometry; fitted states retained independently of count support. No stage imputation or row normalization.'
  temporary.replace(OUT/'sample_data.h5')
 (OUT/'sampling_manifest.json').write_text(json.dumps(dict(n_source_peaks=len(allids),n_train_peaks=int(train.sum()),n_test_peaks=int(test.sum()),n_marker_only_peaks=int((~train&~test).sum()),seed=SEED,source_signature=before,marker_candidates=markers,selection='Uniform random whole peaks; train/test disjoint; extra marker peaks queried only',no_dynamic_support_filter_applied=True),indent=2)+'\n')
 assert before==signature(SOURCE);log('Sample ready: '+str(n)+' peaks')


def pca():
 rows=[];intervalrows=[]
 with h5py.File(OUT/'sample_data.h5','r') as f,h5py.File(OUT/'state_pca.partial.h5','w') as out:
  train=f['train_mask'][:].astype(bool);test=f['test_mask'][:].astype(bool);cv=f['condition_valid'][:];C=f['celltypes'].asstr()[:];T=f['stages'][:]
  panels={'primary_21':(np.flatnonzero(cv.all(axis=1)),np.arange(6)),'matched_21':(np.flatnonzero(cv.all(axis=1)),np.arange(1,6)),'optic_24':(np.flatnonzero(cv[:,1:].all(axis=1)),np.arange(1,6))}
  observed=f['observed_cpm'][:];fitted=f['fitted_cpm'][:];supported=f['rate_supported'][:]
  for name,(cc,tt) in panels.items():
   group=out.create_group(name);group.create_dataset('celltype_indices',data=cc);group.create_dataset('stage_indices',data=tt);write_strings(group,'celltypes',C[cc]);group.create_dataset('stages',data=T[tt])
   raw=observed[:,tt,:][:,:,cc].astype(float);fit=fitted[:,tt,:][:,:,cc].astype(float)
   assert np.isfinite(raw).all() and np.isfinite(fit).all() and raw.min()>=0 and fit.min()>=0
   for key,x in [('observed_mean_cpm',raw.mean(axis=-1)),('fitted_mean_cpm',fit.mean(axis=-1)),('observed_tau',specificity(raw)),('fitted_tau',specificity(fit))]:group.create_dataset(key,data=x.astype('f4'),compression='gzip')
   for label,x in [('observed',raw),('fitted',fit)]:
    rates=np.diff(x,axis=1)/np.diff(T[tt])[None,:,None];group.create_dataset(label+'_rate_rms_cpm_per_hpf',data=np.sqrt(np.mean(rates*rates,axis=-1)).astype('f4'),compression='gzip');group.create_dataset(label+'_net_rate_cpm_per_hpf',data=rates.mean(axis=-1).astype('f4'),compression='gzip')
   sup=supported[:,tt[:-1],:][:,:,cc];group.create_dataset('supported_celltype_count',data=sup.sum(axis=-1).astype('i2'),compression='gzip')
   for transform in ['log1p','linear']:
    x=np.log1p(raw) if transform=='log1p' else raw.copy();y=np.log1p(fit) if transform=='log1p' else fit.copy();d=x.shape[-1]
    model=PCA(n_components=d,svd_solver='full',whiten=False).fit(x[train].reshape(-1,d))
    # One transform call gives all observed/fitted states an identical affine map.
    coords=model.transform(np.concatenate([x.reshape(-1,d),y.reshape(-1,d)]));size=x.shape[0]*x.shape[1];obs_pc=coords[:size].reshape(x.shape);fit_pc=coords[size:].reshape(y.shape)
    sub=group.create_group(transform)
    for key,val in [('observed_pca',obs_pc.astype('f4')),('fitted_pca',fit_pc.astype('f4')),('components',model.components_),('mean',model.mean_),('explained_variance_ratio',model.explained_variance_ratio_),('singular_values',model.singular_values_)]:sub.create_dataset(key,data=val,compression='gzip')
    sub.attrs['training']='Observed states of 20,000 training peaks; all selected stages pooled; celltype columns centered globally; no whitening or feature variance scaling'
    for split,mask in [('train',train),('test',test)]:
     for estimator,z,pc in [('observed',x,obs_pc),('fitted',y,fit_pc)]:
      delta=np.diff(z[mask],axis=1);dpc=np.diff(pc[mask],axis=1);total=float(np.sum(delta*delta));per=np.sum(dpc*dpc,axis=-1)
      reconstruction=pc[mask]@model.components_+model.mean_;error=float(np.max(np.abs(reconstruction-z[mask])))
      for k in [2,3,d]:
       kept=np.sum(dpc[...,:k]**2,axis=-1);valid=per>1e-12
       rows.append(dict(panel=name,transform=transform,split=split,estimator=estimator,components=k,state_variance_fraction=float(model.explained_variance_ratio_[:k].sum()),temporal_energy_fraction=float(kept.sum()/total) if total>0 else np.nan,median_per_step_energy_fraction=float(np.median(kept[valid]/per[valid])),p10_per_step_energy_fraction=float(np.quantile(kept[valid]/per[valid],.1)),max_reconstruction_error_all_pcs=error,n_steps=int(valid.sum())))
      for j in range(len(tt)-1):
       den=float(np.sum(delta[:,j]**2));intervalrows.append(dict(panel=name,transform=transform,split=split,estimator=estimator,start_hpf=T[tt[j]],end_hpf=T[tt[j+1]],pca2_temporal_energy_fraction=float(np.sum(dpc[:,j,:2]**2)/den),pca3_temporal_energy_fraction=float(np.sum(dpc[:,j,:3]**2)/den)))
    log('PCA '+name+' '+transform)
  out.attrs['basis_fit']='Observed training peak states only';out.attrs['source_sample']=str(OUT/'sample_data.h5')
 (OUT/'state_pca.partial.h5').replace(OUT/'state_pca.h5')
 pd.DataFrame(rows).to_csv(OUT/'pca_preservation.csv',index=False);pd.DataFrame(intervalrows).to_csv(OUT/'pca_interval_preservation.csv',index=False)
 log('PCA ready')


def figures():
 metadata=pd.read_csv(OUT/'sample_metadata.csv',keep_default_na=False)
 with h5py.File(OUT/'sample_data.h5','r') as f,h5py.File(OUT/'state_pca.h5','r') as p:
  g=p['primary_21'];cc=g['celltype_indices'][:];t=g['stages'][:];train=f['train_mask'][:];test=f['test_mask'][:];obs=g['log1p/observed_pca'][:];fit=g['log1p/fitted_pca'][:];x=f['observed_cpm'][:][:,:,cc];y=f['fitted_cpm'][:][:,:,cc]
  support=f['rate_supported'][:][:,:,cc];rr=np.diff(y,axis=1)/np.diff(t)[None,:,None];quality=f['fit_valid'][:][:,cc]&(f['lack_of_fit'][:][:,cc]<=4)
  positive=(support&(rr>0)).sum(axis=(1,2));negative=(support&(rr<0)).sum(axis=(1,2));amount=np.linalg.norm(np.diff(y,axis=1),axis=(1,2));tau=g['fitted_tau'][:];mean=y.mean(-1);modes=f['modes'][:][:,cc];root=f['broad_reference'][:]
  candidates={
   'Small fitted change / broad reference':np.flatnonzero(test&root&(amount<=np.quantile(amount[root],.4))),
   'Opening-dominated example':np.flatnonzero(test&(positive>=5)&(positive>3*(negative+1))),
   'Closing-dominated example':np.flatnonzero(test&(negative>=5)&(negative>3*(positive+1))),
   'Transient peak in ≥1 celltype':np.flatnonzero(test&(modes==4).any(1)),
   'Transient trough in ≥1 celltype':np.flatnonzero(test&(modes==5).any(1)),
   'Increasing specificity example':np.flatnonzero(test&((tau[:,-1]-tau[:,0])>.2)&((positive+negative)>=5))}
  selected=[]
  for label,ids in candidates.items():
   if len(ids):
    ids=ids[np.argsort(amount[ids])];i=int(ids[len(ids)//2]);focal=np.flatnonzero(modes[i]==4) if 'Transient peak' in label else np.flatnonzero(modes[i]==5) if 'Transient trough' in label else np.arange(len(cc));focal=int(focal[np.argmax(np.ptp(y[i,:,focal],axis=-1))]) if len(focal) else 0;selected.append(dict(label=label,sample_index=i,peak_id=metadata.iloc[i].peak_id,nearest_gene=metadata.iloc[i].nearest_gene,focal_celltype=f['celltypes'].asstr()[cc[focal]],selection='Median fitted path amount among held-out candidates meeting the stated program filter'))
  pd.DataFrame(selected).to_csv(OUT/'representative_peaks.csv',index=False)
  rng=np.random.default_rng(17);background=rng.choice(np.flatnonzero(test),min(1200,test.sum()),replace=False);stagecolors=plt.cm.viridis(np.linspace(.05,.9,len(t)))
  fig,axes=plt.subplots(1,2,figsize=(14,6))
  for ax,k,title in [(axes[0],1,'PC1 versus PC2'),(axes[1],2,'PC1 versus PC3')]:
   for j,h in enumerate(t):ax.scatter(obs[background,j,0],obs[background,j,k],s=3,c=[stagecolors[j]],alpha=.25,label=f'{int(h)} hpf',rasterized=True)
   ax.set_xlabel('PC1');ax.set_ylabel(f'PC{k+1}');ax.set_title(title);ax.spines[['top','right']].set_visible(False)
  axes[0].legend(markerscale=3,fontsize=9);fig.suptitle('One observed accessibility-state space for all six stages',fontsize=17)
  fig.text(.5,.015,'Each point = one peak at one stage. Shown: 1,200 held-out peaks × six stages. Basis learned from observed log1p(CPM) of 20,000 training peaks.\nThe 21 celltype features are shared across stages. Stage labels and peak identity do not constrain the basis.',ha='center',fontsize=10)
  fig.subplots_adjust(bottom=.16,top=.88,wspace=.28);save(fig,'01_shared_pca_states')
  fig,axes=plt.subplots(2,3,figsize=(17,10))
  for ax,entry in zip(axes.flat,selected):
   i=entry['sample_index'];ax.plot(obs[i,:,0],obs[i,:,1],'--',c='.65',lw=1.1,label='Observed');ax.scatter(obs[i,:,0],obs[i,:,1],c=stagecolors,s=27,marker='o',zorder=3)
   ax.plot(fit[i,:,0],fit[i,:,1],'-',c='#243e53',lw=1.7,label='Fitted in same basis');ax.scatter(fit[i,:,0],fit[i,:,1],c=stagecolors,s=35,marker='s',edgecolor='#243e53',lw=.4,zorder=4)
   for j,h in enumerate(t):ax.annotate(str(int(h)),(fit[i,j,0],fit[i,j,1]),xytext=(5,4),textcoords='offset points',fontsize=8)
   for j in range(len(t)-1):
    if np.linalg.norm(fit[i,j+1,:2]-fit[i,j,:2])>1e-10:ax.annotate('',xy=fit[i,j+1,:2],xytext=fit[i,j,:2],arrowprops=dict(arrowstyle='->',color='#243e53',lw=1.1))
   ax.set_title(entry['label']+'\n'+str(entry['nearest_gene'])+' · '+entry['peak_id'],fontsize=10);ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.spines[['top','right']].set_visible(False)
  axes.flat[0].legend(fontsize=8);fig.suptitle('The same genomic peak traced through accessibility states',fontsize=17)
  fig.text(.5,.018,'Selected examples from held-out peaks; they do not estimate program prevalence. Numbers mark hpf; arrows connect sampled fitted states retrospectively.\nA PCA path can miss changes outside the displayed axes. Fitted smoothness and count-noise support are not biological-replicate validation.',ha='center',fontsize=10)
  fig.subplots_adjust(top=.88,bottom=.13,hspace=.43,wspace=.3);save(fig,'02_representative_pca_trajectories')
  # True 3D PCA trajectories with the same affine coordinate system.
  fig=plt.figure(figsize=(16,10))
  for a,entry in enumerate(selected):
   ax=fig.add_subplot(2,3,a+1,projection='3d');i=entry['sample_index'];ax.plot(*obs[i,:,:3].T,'--',c='.65');ax.plot(*fit[i,:,:3].T,c='#243e53');ax.scatter(*fit[i,:,:3].T,c=stagecolors,s=25)
   for j,h in enumerate(t):ax.text(*fit[i,j,:3],str(int(h)),fontsize=7)
   ax.set_title(entry['label'],fontsize=10);ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_zlabel('PC3')
  fig.suptitle('Shared 3D PCA · representative same-peak state trajectories',fontsize=16);fig.text(.5,.02,'Dashed: observed. Solid: fitted states transformed through the observed-data basis. Colors progress from 10 to 24 hpf; no refitting per stage.',ha='center',fontsize=10)
  fig.subplots_adjust(top=.91,bottom=.1,hspace=.25,wspace=.1);save(fig,'03_representative_pca_3d')
  fig,axes=plt.subplots(2,3,figsize=(17,8))
  for ax,entry in zip(axes.flat,selected):
   i=entry['sample_index'];ax.plot(t,mean[i],'o-',c='#176b86',label='Mean fitted CPM');ax.scatter(t,x[i].mean(-1),facecolors='none',edgecolors='#176b86',label='Mean observed CPM')
   twin=ax.twinx();twin.plot(t,tau[i],'s--',c='#a44d68',label='Fitted specificity (tau)');twin.set_ylim(0,1.05);twin.set_ylabel('Specificity tau',color='#a44d68');ax.set_xlabel('hpf');ax.set_ylabel('Mean CPM',color='#176b86');ax.set_title(entry['label'],fontsize=10)
  axes.flat[0].legend(fontsize=7,loc='best');fig.suptitle('Accessibility amount and celltype specificity are separate quantities',fontsize=17)
  fig.text(.5,.015,'Tau is calculated across the fixed 21-celltype panel; it is undefined when all values are zero and can be sensitive to low coverage.\nExamples are selected for illustration. Mean CPM is relative library-normalized accessibility, not absolute chromatin openness.',ha='center',fontsize=10)
  fig.subplots_adjust(top=.87,bottom=.15,hspace=.4,wspace=.48);save(fig,'04_accessibility_and_specificity')
 df=pd.read_csv(OUT/'pca_preservation.csv');sub=df.query('panel=="primary_21" and split=="test" and components in [2,3]');fig,axes=plt.subplots(1,2,figsize=(12,5))
 for ax,transform in zip(axes,['log1p','linear']):
  z=sub[sub['transform']==transform];labels=[f'{r.estimator}\n{r.components} PCs' for r in z.itertuples()];ax.bar(labels,100*z.temporal_energy_fraction,color=['#267e96','#5ea1b0','#b17998','#cb9cb4']);ax.set_ylim(0,105);ax.set_ylabel('Temporal displacement energy retained (%)');ax.set_title(transform+'(CPM)' if transform=='log1p' else 'Linear CPM')
  for j,r in enumerate(z.itertuples()):ax.text(j,100*r.temporal_energy_fraction+2,f'{100*r.temporal_energy_fraction:.1f}%',ha='center')
 fig.suptitle('How much temporal change is visible in two or three PCs?',fontsize=16);fig.text(.5,.015,'Held-out peaks; pooled squared displacement across intervals. The full set of 21 PCs preserves all transformed-space change.\nGlobal energy weights large changes more heavily; per-step and interval diagnostics are saved separately.',ha='center',fontsize=10);fig.subplots_adjust(top=.84,bottom=.23,wspace=.26);save(fig,'05_pca_change_preservation')
 log('PCA figures ready')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','prepare','pca','figures'],default='all');a=p.parse_args()
 if a.stage in ['all','prepare']:prepare()
 if a.stage in ['all','pca']:pca()
 if a.stage in ['all','figures']:figures()
