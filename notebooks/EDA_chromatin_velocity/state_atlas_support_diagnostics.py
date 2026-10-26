"""Report display fidelity by pre-existing temporal-support strata.

These overlapping strata describe conditional count-noise support, not biological
replicate confidence. They are not used to refit or select the embedding.
"""
from pathlib import Path
import numpy as np,pandas as pd,h5py,json
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'outputs/state_trajectory_atlas_v1'

def masks(f,cc,ids):
 count=f['rate_supported'][:][ids][:,:,cc].sum(-1);v=f['rates'][:][ids][:,:,cc];se=f['rate_se'][:][ids][:,:,cc]
 den=np.linalg.norm(se,axis=-1);snr=np.divide(np.linalg.norm(v,axis=-1),den,out=np.zeros_like(den),where=den>0)
 return {'All intervals':np.ones(count.shape,bool),'No supported celltype':count==0,'Any supported celltype':count>0,'At least 3 supported celltypes':count>=3,'Any support and vector SNR≥2':(count>0)&(snr>=2)}

def main():
 rows=[];umaprows=[]
 with h5py.File(OUT/'sample_data.h5','r') as f,h5py.File(OUT/'state_pca.h5','r') as p:
  cc=p['primary_21/celltype_indices'][:];tt=p['primary_21/stages'][:];test=np.flatnonzero(f['test_mask'][:]);strata=masks(f,cc,test)
  for estimator in ['observed','fitted']:
   pc=p['primary_21/log1p/'+estimator+'_pca'][:];delta=np.diff(pc[test],axis=1);den=np.sum(delta**2,axis=-1)
   for name,mask in strata.items():
    valid=mask&(den>1e-12)
    for k in [2,3]:
     num=np.sum(delta[...,:k]**2,axis=-1)
     rows.append(dict(estimator=estimator,stratum=name,components=k,n_intervals=int(mask.sum()),n_nonzero=int(valid.sum()),temporal_energy_fraction=float(num[mask].sum()/den[mask].sum()) if den[mask].sum()>0 else np.nan,median_step_fraction=float(np.median(num[valid]/den[valid])) if valid.any() else np.nan))
   for dim in [2,3]:
    path=OUT/f'umap_coordinates/primary_21_{dim}d.npz'
    if not path.exists():continue
    z=np.load(path);ids=z['evaluation_peak_indices'];panel=masks(f,cc,ids);high=np.linalg.norm(np.diff(pc[ids],axis=1),axis=-1);low=np.linalg.norm(np.diff(z[estimator][ids],axis=1),axis=-1)
    for name,mask in panel.items():
     valid=mask&(high>1e-10)
     umaprows.append(dict(estimator=estimator,stratum=name,dimensions=dim,n_intervals=int(mask.sum()),n_nonzero=int(valid.sum()),step_length_spearman=float(spearmanr(high[valid],low[valid]).statistic) if valid.sum()>=3 else np.nan,rate_length_spearman=float(spearmanr((high/np.diff(tt))[valid],(low/np.diff(tt))[valid]).statistic) if valid.sum()>=3 else np.nan,peak_cohort='UMAP engine evaluation_peak_indices',n_evaluation_peaks=len(ids)))
 pd.DataFrame(rows).to_csv(OUT/'support_stratified_pca.csv',index=False);pd.DataFrame(umaprows).to_csv(OUT/'support_stratified_umap.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(14,5))
 labels=list(strata)
 for ax,estimator in zip(axes,['observed','fitted']):
  sub=pd.DataFrame(rows).query('components==3 and estimator==@estimator').set_index('stratum').loc[labels]
  ax.barh(np.arange(len(sub)),100*sub.temporal_energy_fraction,color='#347d91' if estimator=='observed' else '#a77999');ax.set_yticks(np.arange(len(sub)),labels);ax.set_xlim(0,105);ax.set_xlabel('Temporal energy retained in 3 log-PCA axes (%)');ax.set_title(estimator.capitalize()+' states · held-out peaks');ax.spines[['top','right']].set_visible(False)
  for j,r in enumerate(sub.itertuples()):ax.text(100*r.temporal_energy_fraction+1,j,f'{100*r.temporal_energy_fraction:.1f}% (n={r.n_intervals:,})',va='center',fontsize=8)
 fig.suptitle('Does display fidelity improve for intervals with supported changes?',fontsize=16);fig.text(.5,.015,'Strata overlap. Support and vector signal-to-noise derive from the existing conditional count-noise model, not biological replication.\nNo support stratum changes the fitted coordinate system. Full high-dimensional states remain available.',ha='center',fontsize=10);fig.subplots_adjust(left=.22,right=.98,top=.86,bottom=.18,wspace=.75)
 fig.savefig(OUT/'07_support_stratified_preservation.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'07_support_stratified_preservation.pdf',bbox_inches='tight');plt.close(fig)
 if umaprows:
  ud=pd.DataFrame(umaprows);fig,axes=plt.subplots(1,2,figsize=(15,5.5),sharex=True)
  for ax,est in zip(axes,['observed','fitted']):
   for dim,color,shift in [(2,'#347d91',-.16),(3,'#a77999',.16)]:
    sub=ud.query('estimator==@est and dimensions==@dim').set_index('stratum').reindex(labels)
    if sub.step_length_spearman.isna().all():continue
    ax.barh(np.arange(len(labels))+shift,sub.step_length_spearman,height=.29,color=color,label=f'{dim}D')
    for j,r in enumerate(sub.itertuples()):ax.text(r.step_length_spearman+.015,j+shift,f'{r.step_length_spearman:.2f}',va='center',fontsize=8)
   ax.set_yticks(np.arange(len(labels)),[x.replace(' supported celltypes',' supported CTs').replace(' supported celltype',' supported CT') for x in labels]);ax.set_xlim(0,1);ax.set_xlabel('Full-space vs UMAP step-length rank correlation');ax.set_title(est.capitalize()+' states');ax.legend(loc='lower right');ax.spines[['top','right']].set_visible(False)
  fig.suptitle('Supported changes have better display fidelity, with remaining distortion',fontsize=16)
  fig.text(.5,.015,'Same 750 held-out peaks; overlapping strata; no support selection used to fit the layout. Support is conditional on the count-noise model.\nRequiring multiple changing celltypes would exclude real celltype-specific programs; this is a diagnostic, not a display filter.',ha='center',fontsize=10)
  fig.subplots_adjust(left=.22,right=.98,top=.86,bottom=.18,wspace=.8)
  fig.savefig(OUT/'08_support_stratified_umap.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'08_support_stratified_umap.pdf',bbox_inches='tight');plt.close(fig)
 (OUT/'support_strata_definition.json').write_text(json.dumps(dict(primary_panel='21 celltypes at six stages',pca_evaluation='All 5,000 held-out peaks',umap_evaluation='Same fixed 750 held-out peaks as the UMAP engine',vector_snr='L2 fitted rate / L2 conditional rate SE across the full panel',overlapping_strata=True,used_to_train_or_select_layout=False),indent=2)+'\n')
 print(pd.DataFrame(umaprows).to_string(index=False),flush=True)
if __name__=='__main__':main()
