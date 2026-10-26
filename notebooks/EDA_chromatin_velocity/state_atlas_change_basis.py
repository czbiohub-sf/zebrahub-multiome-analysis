"""Train-only linear display axes emphasizing measured temporal differences.

Uncentered SVD retains the mean change direction. This is a retrospective view
using known stage pairings, not a forecast or independent temporal-order test.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/state_trajectory_atlas_v1'

def main():
 rows=[]
 with h5py.File(OUT/'sample_data.h5','r') as f,h5py.File(OUT/'state_pca.h5','r') as p,h5py.File(OUT/'state_change_basis.h5','w') as dest:
  cc=p['primary_21/celltype_indices'][:];times=p['primary_21/stages'][:];train=f['train_mask'][:].astype(bool);test=f['test_mask'][:].astype(bool)
  raw=f['observed_cpm'][:][:,:,cc].astype(float);fit=f['fitted_cpm'][:][:,:,cc].astype(float);x=np.log1p(raw);y=np.log1p(fit);mu=x[train].mean(axis=(0,1));dx=np.diff(x,axis=1);dy=np.diff(y,axis=1);dt=np.diff(times)
  dest.create_dataset('celltype_indices',data=cc);dest.create_dataset('stages',data=times);dest.attrs['definition']='Uncentered SVD of observed adjacent-stage log1p(CPM) differences from training peaks only; state positions use fixed observed-training mean. Temporal pairings are used, with no root or direction constraint.'
  for label,changes in [('displacement',dx),('rate',dx/dt[None,:,None])]:
   _,s,components=np.linalg.svd(changes[train].reshape(-1,len(cc)),full_matrices=False);g=dest.create_group(label)
   coords_x=(x-mu)@components.T;coords_y=(y-mu)@components.T
   for key,value in [('components',components),('mean',mu),('singular_values',s),('observed_coords',coords_x.astype('f4')),('fitted_coords',coords_y.astype('f4'))]:g.create_dataset(key,data=value,compression='gzip')
   assert np.max(np.abs(components@components.T-np.eye(len(cc))))<1e-10
   for estimator,z,coord in [('observed',x,coords_x),('fitted',y,coords_y)]:
    reconstructed=coord@components+mu;assert np.max(np.abs(reconstructed-z))<1e-10
    a=np.diff(z[test],axis=1);b=np.diff(coord[test],axis=1)
    for measure in ['displacement','rate']:
     scale=1 if measure=='displacement' else dt[None,:,None];aa=a/scale;bb=b/scale;den=np.sum(aa**2,axis=-1);valid=den>1e-12
     for k in [2,3,len(cc)]:
      num=np.sum(bb[...,:k]**2,axis=-1)
      rows.append(dict(basis=label,estimator=estimator,measure=measure,components=k,pooled_energy_fraction=float(num.sum()/den.sum()),median_step_energy_fraction=float(np.median(num[valid]/den[valid]))))
  dest.attrs['fit_uses_time_pairings']=True;dest.attrs['train_peaks']=int(train.sum());dest.attrs['held_out_peaks']=int(test.sum())
 pd.DataFrame(rows).to_csv(OUT/'change_basis_preservation.csv',index=False)
 ordinary=pd.read_csv(OUT/'pca_preservation.csv').query('panel=="primary_21" and transform=="log1p" and split=="test" and components in [2,3]');new=pd.DataFrame(rows).query('measure=="displacement" and components in [2,3]')
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for ax,estimator in zip(axes,['observed','fitted']):
  labels=[];values=[]
  for basis in ['State PCA','Change SVD','Rate SVD']:
   for k in [2,3]:
    val=ordinary[(ordinary.estimator==estimator)&(ordinary.components==k)].iloc[0].temporal_energy_fraction if basis=='State PCA' else new[(new.estimator==estimator)&(new.components==k)&(new.basis==('displacement' if basis=='Change SVD' else 'rate'))].iloc[0].pooled_energy_fraction
    labels.append(basis+'\n'+str(k)+' axes');values.append(100*val)
  ax.bar(labels,values,color=['#8fa4b0','#668796','#4b9e9d','#247b7e','#bba5be','#977399']);ax.set_ylim(0,105);ax.set_ylabel('Held-out temporal-displacement energy retained (%)');ax.set_title(estimator.capitalize()+' accessibility states')
  for j,value in enumerate(values):ax.text(j,value+2,f'{value:.1f}',ha='center',fontsize=8)
 fig.suptitle('Can axes trained on changes preserve more temporal movement?',fontsize=16)
 fig.text(.5,.015,'All bases use observed training peaks only. Change/rate SVD uses known temporal pairings, without centering away the mean change.\nHeld-out geometric preservation is not biological validation. Compact views still lose changes; full 21-dimensional coordinates remain the reference.',ha='center',fontsize=10)
 fig.subplots_adjust(top=.84,bottom=.23,wspace=.28);fig.savefig(OUT/'06_change_focused_linear_axes.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'06_change_focused_linear_axes.pdf',bbox_inches='tight');plt.close(fig)
 (OUT/'change_basis_validation.json').write_text(json.dumps(dict(train_only=True,uncentered_differences=True,all_axes_orthonormal=True,all_coordinate_reconstruction_error_below_1e_10=True,scope='Retrospective linear display sensitivity; temporal pairings used. It can emphasize sampling noise as well as biological change.'),indent=2)+'\n')
 print(pd.DataFrame(rows).query('measure=="displacement" and components==3').to_string(index=False),flush=True)
if __name__=='__main__':main()
