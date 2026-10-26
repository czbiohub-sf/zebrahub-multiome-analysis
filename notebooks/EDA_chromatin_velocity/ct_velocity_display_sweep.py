"""Display-only smoothing sensitivity, independent of rate/neighbor experiments."""
from pathlib import Path
import sys,json
import h5py,numpy as np,pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts/chrom_velo'))
from streamline_field import build_stream_grid
from streamline_paths import draw_supported_streams
OUT=HERE/'outputs/ct_velocity_experiments_v1';PRIOR=HERE/'outputs/ct_velocity_reprojection_v1';FITS=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'

def dump(p,d):p.write_text(json.dumps(d,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist())+'\n')
def log(s):print(s,flush=True)

def main():
 OUT.mkdir(parents=True,exist_ok=True);folder=OUT/'display_grids';folder.mkdir(exist_ok=True)
 with h5py.File(FITS) as f:
  broad=f['broad_reference'][:].astype(bool);ct=f['common_celltypes'][:].astype(bool);m=f['modes'][:][:,ct];gradual=np.isin(m,[2,3]).any(axis=1);ids=f['peak_ids'].asstr()[:]
 with h5py.File(PRIOR/'updated_projection.h5') as f:
  assert np.array_equal(ids,f['peak_ids'].asstr()[:]);xy=f['umap2d'][:];stages=f['stages'][:];fields=[{key:f[f'projection/{t}/{key}'][:] for key in ['changing','net_rate']}|{'v':f[f'projection/{t}/display_vectors/umap2d'][:],'valid':f[f'projection/{t}/projection_valid/umap2d'][:]} for t in range(5)]
 n=len(xy);bg=np.sort(np.random.default_rng(20261022).choice(n,100000,replace=False));lim=json.loads((PRIOR/'figure_manifest.json').read_text())['rate_limit'];reference_tree=cKDTree(xy[broad]);rows=[];grids={};path_checks=[]
 presets=[(30,.5),(50,.75),(100,1.5)];intervals=[f'{int(a)} → {int(b)} hpf' for a,b in zip(stages[:-1],stages[1:])]
 fig,axes=plt.subplots(1,3,figsize=(15,5))
 for ax,title,mask,col in zip(axes,['Broad-accessibility reference','At least one increasing celltype trajectory','At least one decreasing celltype trajectory'],[broad,(m==2).any(axis=1),(m==3).any(axis=1)],['#d49820','#c74c42','#326ca8']):
  ax.scatter(*xy[bg].T,s=.5,c='#d6d9dc',alpha=.5,linewidths=0,rasterized=True);ss=bg[mask[bg]];ax.scatter(*xy[ss].T,s=1,c=col,alpha=.7,linewidths=0,rasterized=True);ax.set(title=f'{title}\n{mask.sum():,} peaks',aspect='equal',xticks=[],yticks=[])
 fig.suptitle('Fixed peak atlas: distributed reference and previously fitted trend classifications',fontsize=15);fig.text(.5,.02,'A peak can increase in one celltype and decrease in another. Six-stage classifications are retrospective; they are not forecast-selection criteria.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(OUT/'01_reference_and_gradual_peaks.png',dpi=180);fig.savefig(OUT/'01_reference_and_gradual_peaks.pdf');plt.close(fig)
 if '--reference-only' in sys.argv:return
 for t,a in enumerate(fields):
  baseline=build_stream_grid(xy,a['v'],valid=a['valid']);gridxy=np.stack(np.meshgrid(baseline['x'],baseline['y']),axis=-1).reshape(-1,2);rd,ri=reference_tree.query(gridxy);radial=gridxy-xy[broad][ri];radial=np.divide(radial,rd[:,None],out=np.zeros_like(radial),where=rd[:,None]>0);outside=rd>baseline['metadata']['grid_step']
  for nn in [30,50,100]:
   for bw in [.5,.75,1.5]:
    g=baseline if (nn,bw)==(50,.75) else build_stream_grid(xy,a['v'],valid=a['valid'],n_neighbors=nn,bandwidth_cells=bw,radius_cells=2*bw)
    assert np.array_equal(g['x'],baseline['x']) and np.array_equal(g['y'],baseline['y']);vv=np.c_[g['u'].ravel(),g['v'].ravel()];base=np.c_[baseline['u'].ravel(),baseline['v'].ravel()];ok=g['valid'].ravel();common=ok&baseline['valid'].ravel();den=np.linalg.norm(vv,axis=1)*np.linalg.norm(base,axis=1);cs=np.divide(np.sum(vv*base,axis=1),den,out=np.full(len(vv),np.nan),where=den>0);rr=np.sum(vv*radial,axis=1);use=ok&outside
    row=dict(interval=t,start_hpf=int(stages[t]),end_hpf=int(stages[t+1]),display_neighbors=nn,bandwidth_cells=bw,radius_cells=2*bw,eligible_peaks=int(a['valid'].sum()),supported_grid_nodes=int(ok.sum()),supported_fraction=float(ok.mean()),common_with_default=int(common.sum()),median_direction_cosine_to_default=float(np.nanmedian(cs[common])) if common.any() else None,fraction_direction_gt90_to_default=float(np.mean(cs[common]<0)) if common.any() else None,outside_reference_grid_nodes=int(use.sum()),display_mean_outward_component=float(np.mean(rr[use])) if use.any() else None,display_outward_fraction=float(np.mean(rr[use]>0)) if use.any() else None)
    rows.append(row);tag=f't{t}_n{nn}_bw{bw}';np.savez_compressed(folder/(tag+'.npz'),**{k:v for k,v in g.items() if k!='metadata'});dump(folder/(tag+'.json'),g['metadata'])
    if (nn,bw) in presets:grids[t,nn,bw]=g
  log(f'Display smoothing interval {t} complete')
 df=pd.DataFrame(rows);df.to_csv(OUT/'display_smoothing_metrics.csv',index=False)
 fig,axes=plt.subplots(3,5,figsize=(21,12))
 for row,(nn,bw) in enumerate(presets):
  for t,a in enumerate(fields):
   ax=axes[row,t];ax.scatter(*xy[bg].T,s=.4,c='#d5d8dc',alpha=.4,linewidths=0,rasterized=True);colored=bg[a['changing'][bg]&np.isfinite(a['net_rate'][bg])];art=ax.scatter(*xy[colored].T,s=.7,c=a['net_rate'][colored],cmap='RdBu_r',norm=TwoSlopeNorm(0,-lim,lim),linewidths=0,alpha=.75,rasterized=True);r=draw_supported_streams(ax,grids[t,nn,bw],density=2.5,linewidth=.7,arrowsize=.75,color='#1e3a47');path_checks.append(dict(interval=t,neighbors=nn,bandwidth=bw,**r['audit']));ax.set(aspect='equal',xticks=[],yticks=[])
   if row==0:ax.set_title(intervals[t])
   if t==0:ax.set_ylabel(f'{nn} display neighbors\nwidth {bw:g}, radius {2*bw:g} grid cells')
 fig.subplots_adjust(left=.065,right=.9,bottom=.085,top=.91,wspace=.08,hspace=.12);cax=fig.add_axes([.92,.28,.012,.42]);fig.colorbar(art,cax=cax,label='Mean fitted accessibility rate (CPM/hour)');fig.suptitle('Display smoothing changes the stream picture; per-peak velocities stay fixed',fontsize=17);fig.text(.5,.025,'Same full-atlas k=30 projected rates in every row. Finite-radius masks and direction agreement ≥0.35 remain active.\nGrid curves are display summaries; widths and path lengths do not measure biological speed or time.',ha='center',fontsize=10);fig.savefig(OUT/'02_display_smoothing_comparison.png',dpi=175);fig.savefig(OUT/'02_display_smoothing_comparison.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 for nn in [30,50,100]:
  for ax,col in zip(axes,['supported_grid_nodes','fraction_direction_gt90_to_default']):
   q=df[df.display_neighbors==nn].groupby('bandwidth_cells')[col];mu=q.mean();lo=q.min();hi=q.max();factor=100 if col.startswith('fraction') else 1;ax.plot(mu.index,factor*mu,marker='o',label=f'{nn} display neighbors');ax.fill_between(mu.index,factor*lo,factor*hi,alpha=.12)
 for ax in axes:ax.set_xlabel('Gaussian width (grid cells; radius = 2 × width)');ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Supported grid nodes');axes[1].set_ylabel('Directions >90° from default (%)\nAmong common supported grid nodes');axes[0].legend();fig.suptitle('Display-only sensitivity across all five intervals');fig.text(.5,.01,'Lines are interval means; bands span intervals, not confidence intervals. Default: 50 display neighbors, width 0.75.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(OUT/'03_display_smoothing_metrics.png',dpi=180);fig.savefig(OUT/'03_display_smoothing_metrics.pdf');plt.close(fig)
 dump(OUT/'display_sweep_validation.json',dict(status='passed',n_peaks=n,n_settings=9,n_intervals=5,n_broad_reference=int(broad.sum()),broad_reference_definition='Pre-existing whole-course broad-accessible set; distributed across atlas',n_any_gradual=int(gradual.sum()),same_rates_vectors_and_masks_for_all_display_settings=True,all_grid_coordinates_identical=True,finite_support_and_path_clipping=True,path_checks=path_checks,default=dict(n_neighbors=50,bandwidth_cells=.75,radius_cells=1.5),interpretation='Display parameter sensitivity only; neither rootward/outward statistics nor expanded support establish biological flow. Grid nodes are not independent observations.'))
 log('All display plots complete')

if __name__=='__main__':main()
