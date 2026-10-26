"""Fixed-coordinate arrow and support-masked stream views of reused fitted rates."""
from pathlib import Path
import json,sys
import h5py,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,Normalize
from scipy.interpolate import RegularGridInterpolator
from ct_velocity_reproject import OUT,OLD,ATLAS,read_csr,project_saved,dump,log
from real_specialization_report import choose_arrows
from streamline_field import build_stream_grid
from streamline_paths import draw_supported_streams
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def arrays(path):
 with h5py.File(path) as f:
  a={k:f[k][:] for k in ['umap2d','umap3d','old_umap2d','stages']};a['peak_ids']=f['peak_ids'].asstr()[:]
  for k in ['net_rate','velocity_rms','changing','active']:a[k]=np.stack([f[f'projection/{t}/{k}'][:] for t in range(5)],axis=1)
  for d in ['umap2d','umap3d','old_umap2d']:
   for k in ['vectors','projection_valid','display_vectors','coherence']:a[k+'_'+d]=np.stack([f[f'projection/{t}/{k}/{d}'][:] for t in range(5)],axis=1)
 return a

def decorate(ax,title):ax.set(aspect='equal',xticks=[],yticks=[]);ax.set_title(title,fontsize=11)
def save(fig,name,note):
 fig.text(.5,.015,note,ha='center',fontsize=9,color='#46515e');fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig);log('Saved '+name)
def draw(ax,a,t,rows,limit,d='umap2d',budget=230):
 xy=a[d];rate=a['net_rate'][:,t];good=a['changing'][:,t]&np.isfinite(rate)
 ax.scatter(*xy[rows].T,color='#cdd1d5',s=.45,alpha=.35,linewidths=0,rasterized=True)
 use=rows[good[rows]];art=ax.scatter(*xy[use].T,c=rate[use],cmap='RdBu_r',norm=TwoSlopeNorm(0,-limit,limit),s=.85,alpha=.85,linewidths=0,rasterized=True)
 ids=np.empty(0,int)
 if budget:
  vec=a['display_vectors_'+d][:,t];mask=a['projection_valid_'+d][:,t];ids=choose_arrows(xy,vec,mask,budget=budget);v=vec[ids];v=v/np.linalg.norm(v,axis=1)[:,None]*np.linalg.norm(np.ptp(xy,axis=0))*.012
  ax.quiver(*xy[ids].T,*v.T,color='#102c3c',angles='xy',scale_units='xy',scale=1,width=.0021,headwidth=3.5,headlength=4.7,zorder=4)
 return art,ids

def main():
 new=arrays(OUT/'updated_projection.h5');old=arrays(OUT/'reused_projection.h5');assert np.array_equal(new['peak_ids'],old['peak_ids']);assert np.array_equal(new['net_rate'],old['net_rate'],equal_nan=True);assert np.array_equal(new['velocity_rms'],old['velocity_rms'],equal_nan=True);assert np.array_equal(new['changing'],old['changing'])
 n=len(new['peak_ids']);rows=np.sort(np.random.default_rng(20261021).choice(n,120000,replace=False));shown=np.concatenate([np.abs(new['net_rate'][:,t][new['changing'][:,t]]) for t in range(5)]);limit=max(float(np.quantile(shown,.98)),1e-5);rmslimit=float(np.quantile(new['velocity_rms'][new['changing']],.98));intervals=[f'{int(l)} → {int(r)} hpf' for l,r in zip(new['stages'][:-1],new['stages'][1:])];selection={};coverage=[];agreement=[]
 with h5py.File(ATLAS) as f:mean=f['annotations/mean_cpm'][:]
 # Primary per-peak direction panel; no grid averaging alters the arrows.
 fig,axes=plt.subplots(2,3,figsize=(17,11));axes=axes.ravel()
 for t in range(5):
  art,ids=draw(axes[t],new,t,rows,limit);selection[f'primary_2d_t{t}']=ids;cnt=int(new['projection_valid_umap2d'][:,t].sum());decorate(axes[t],f'{intervals[t]} · {cnt:,} eligible peaks\n{len(ids)} directions displayed')
 axes[5].axis('off');axes[5].text(.02,.92,'New CT × stage peak UMAP\n\nOne point = one fixed genomic peak\nMap: 190 celltype × stage features\nVelocity: 21 celltypes per interval\n\nRed: net opening; blue: net closing\nGray: no supported-change display\nArrows favor coherent examples\nArrow lengths normalized for visibility\n\nNeighbors: new 50-PC log-CPM space\nRates and cosine alignment: linear CPM\nNo imposed root or direction',va='top',fontsize=12,linespacing=1.6);fig.colorbar(art,ax=axes[5],orientation='horizontal',fraction=.06,pad=.03,shrink=.85,label='Mean fitted rate (CPM / hour)')
 fig.suptitle('Previously fitted chromatin velocities reprojected onto the new celltype atlas',fontsize=17);fig.subplots_adjust(top=.91,bottom=.08,wspace=.2,hspace=.28);save(fig,'01_ct_umap_velocity_arrows','All 640,830 peaks evaluated; fixed 120,000-peak display background. Conditional count/fit gates are not biological confidence.\nArrows show local profile correspondence among peaks; arrow direction and opening/closing sign are different quantities.')
 # Old-map / old-weights-new-map / updated-neighbor comparison.
 fig,axes=plt.subplots(3,5,figsize=(22,12))
 for r,(a,d,label) in enumerate([(old,'old_umap2d','Original map + previous weights'),(old,'umap2d','New map + previous weights'),(new,'umap2d','New map + updated neighbors')]):
  for t in range(5):
   art,ids=draw(axes[r,t],a,t,rows,limit,d=d,budget=110);selection[f'comparison_r{r}_t{t}']=ids;decorate(axes[r,t],intervals[t] if r==0 else '')
   if t==0:axes[r,t].set_ylabel(label,fontsize=11)
 fig.colorbar(art,ax=axes.ravel().tolist(),shrink=.55,pad=.012,label='Mean fitted rate (CPM / hour)');fig.suptitle('Separate changing the map from changing the projection neighborhoods',fontsize=17);fig.subplots_adjust(top=.91,bottom=.08,hspace=.2,wspace=.1,right=.88);save(fig,'02_reprojection_controls','The fitted accessibility rates and source gates are identical in every row. Old edges may span distant regions in the new map.\nNew neighbors are selected in high-dimensional log-CPM PCA space, not by 2D proximity. Separate maps have arbitrary orientations.')
 # Display smoothing uses the same parameters as earlier reports, with finite-radius masks.
 folder=OUT/'stream_display';folder.mkdir(exist_ok=True);fig,axes=plt.subplots(2,3,figsize=(17,11));axes=axes.ravel();streaminfo=[]
 for t in range(5):
  art,_=draw(axes[t],new,t,rows,limit,budget=0);grid=build_stream_grid(new['umap2d'],new['display_vectors_umap2d'][:,t],valid=new['projection_valid_umap2d'][:,t]);render=draw_supported_streams(axes[t],grid,color='#123042',density=3,linewidth=.9,arrowsize=.85)
  offsets=np.r_[0,np.cumsum([len(p) for p in render['paths']])];points=np.concatenate(render['paths']) if render['paths'] else np.empty((0,2));heads=np.stack([p.get_xy() for p in render['arrows']]) if render['arrows'] else np.empty((0,4,2))
  np.savez_compressed(folder/f'interval_{t}_grid.npz',**{k:v for k,v in grid.items() if k!='metadata'});np.savez_compressed(folder/f'interval_{t}_paths.npz',points=points,offsets=offsets,arrowheads=heads)
  ids=np.flatnonzero(new['projection_valid_umap2d'][:,t]);ids=ids[np.linspace(0,len(ids)-1,min(10000,len(ids))).astype(int)] if len(ids) else ids
  uv=np.column_stack([RegularGridInterpolator((grid['y'],grid['x']),grid[k],bounds_error=False,fill_value=np.nan)(new['umap2d'][ids][:,::-1]) for k in ['u','v']]);v=new['display_vectors_umap2d'][ids,t];den=np.linalg.norm(uv,axis=1)*np.linalg.norm(v,axis=1);cs=np.sum(uv*v,axis=1)/np.maximum(den,1e-12);ok=np.isfinite(cs)&(den>0)
  item=dict(interval=t,**grid['metadata'],path_audit=render['audit'],native_comparable=int(ok.sum()),median_native_direction_cosine=float(np.median(cs[ok])) if ok.any() else None,native_opposing_fraction=float((cs[ok]<0).mean()) if ok.any() else None);streaminfo.append(item);decorate(axes[t],f'{intervals[t]}\n{grid["valid"].sum():,} supported grid nodes')
 axes[5].axis('off');axes[5].text(.02,.92,'Stream display\n\nSame per-peak directions as figure 01\nFinite-radius interpolation and gap masks\nMinimum 5 eligible nearby peaks\nMinimum local direction agreement 0.35\n\nLocal averaging can hide opposing arrows\nStream length is not developmental time\nStream width is not biological speed',va='top',fontsize=12,linespacing=1.7);fig.colorbar(art,ax=axes[5],orientation='horizontal',fraction=.06,pad=.03,shrink=.85,label='Mean fitted rate (CPM / hour)');fig.suptitle('Supported local stream display on the new celltype peak UMAP',fontsize=17);fig.subplots_adjust(top=.91,bottom=.08,wspace=.2,hspace=.28);save(fig,'03_ct_umap_velocity_streams','Streamlines summarize projected directions locally and are clipped at unsupported interpolation regions.\nThis descriptive view does not establish a global biological flow or trajectories of moving genomic peaks.');dump(OUT/'stream_validation.json',streaminfo)
 # 3D coverage: all interval fields, sampled dots and directions only for rendering.
 fig=plt.figure(figsize=(18,12));bg=rows[::2]
 for t in range(5):
  ax=fig.add_subplot(2,3,t+1,projection='3d');xyz=new['umap3d'];mask=new['changing'][:,t];ax.scatter(*xyz[bg].T,s=.5,color='#cbd0d4',alpha=.12,linewidths=0,rasterized=True);use=bg[mask[bg]];art=ax.scatter(*xyz[use].T,c=new['net_rate'][use,t],cmap='RdBu_r',norm=TwoSlopeNorm(0,-limit,limit),s=1,alpha=.6,linewidths=0,rasterized=True)
  vec=new['display_vectors_umap3d'][:,t];ids=choose_arrows(xyz,vec,new['projection_valid_umap3d'][:,t],budget=150);selection[f'primary_3d_t{t}']=ids;v=vec[ids]/np.linalg.norm(vec[ids],axis=1)[:,None]*np.linalg.norm(np.ptp(xyz,axis=0))*.025;ax.quiver(*xyz[ids].T,*v.T,color='#102c3c',length=1,normalize=False,arrow_length_ratio=.3,linewidth=.7);ax.set(xticks=[],yticks=[],zticks=[],title=intervals[t]);ax.view_init(elev=18,azim=48);ax.set_box_aspect(np.ptp(xyz,axis=0))
 ax=fig.add_subplot(2,3,6);ax.axis('off');ax.text(.02,.85,'Fixed 3D atlas\n\nAll 640,830 peaks evaluated\n60,000 background peaks displayed\nUp to 150 arrows per interval\n\nDirection lengths are display-normalized\nUse the HTML viewer to rotate the map',va='top',fontsize=12,linespacing=1.7);fig.colorbar(art,ax=ax,orientation='horizontal',fraction=.06,pad=.03,shrink=.85,label='Mean fitted rate (CPM / hour)');fig.suptitle('3D chromatin-velocity projection · same fitted rates, new PCA neighborhoods',fontsize=17);fig.subplots_adjust(top=.91,bottom=.08,wspace=.05,hspace=.1);save(fig,'04_ct_umap_velocity_3d','One point = one peak. All five developmental intervals are separate fields; no root or common radial direction is imposed.')
 # Magnitude and sign remain independent scientific scalars.
 fig,axes=plt.subplots(2,5,figsize=(22,8))
 for t in range(5):
  for r in range(2):
   ax=axes[r,t];xy=new['umap2d'];ax.scatter(*xy[rows].T,s=.4,color='#d4d7da',alpha=.35,linewidths=0,rasterized=True);use=rows[new['changing'][rows,t]];val=new['net_rate'][use,t] if r==0 else new['velocity_rms'][use,t];art=ax.scatter(*xy[use].T,c=val,s=.8,alpha=.85,linewidths=0,rasterized=True,cmap='RdBu_r' if r==0 else 'viridis',norm=TwoSlopeNorm(0,-limit,limit) if r==0 else Normalize(0,rmslimit));decorate(ax,intervals[t] if r==0 else '')
   if t==0:ax.set_ylabel('Signed mean rate' if r==0 else 'RMS rate magnitude',fontsize=12)
  if t==4:
   from matplotlib.cm import ScalarMappable
   fig.colorbar(ScalarMappable(norm=TwoSlopeNorm(0,-limit,limit),cmap='RdBu_r'),ax=axes[0].tolist(),shrink=.7,pad=.015,label='CPM / hour');fig.colorbar(art,ax=axes[1].tolist(),shrink=.7,pad=.015,label='CPM / hour')
 fig.suptitle('Opening/closing and change magnitude are unchanged by reprojection',fontsize=17);fig.subplots_adjust(top=.90,bottom=.1,wspace=.12,hspace=.16,right=.88);save(fig,'05_signed_rate_and_magnitude','Rates are relative accessibility in CPM/hour. A mean near zero can conceal opening in one celltype and closing in another.\nColor limits are fixed across intervals (98th percentile); gray points do not pass the source display gate.')
 # Quantify changes in geometry, not just their appearance.
 for t in range(5):
  for name,a in [('reused',old),('updated',new)]:
   for d in ['umap2d','umap3d']:
    keep=a['projection_valid_'+d][:,t];coverage.append(dict(method=name,interval=t,start_hpf=int(new['stages'][t]),end_hpf=int(new['stages'][t+1]),embedding=d,source_changing=int(a['changing'][:,t].sum()),directed_sources=int(a['active'][:,t].sum()),projected=int(keep.sum()),positive_net=int((keep&(a['net_rate'][:,t]>0)).sum()),negative_net=int((keep&(a['net_rate'][:,t]<0)).sum())))
  for d in ['umap2d','umap3d']:
   v=old['vectors_'+d][:,t];w=new['vectors_'+d][:,t];mask=old['projection_valid_'+d][:,t]&new['projection_valid_'+d][:,t];den=np.linalg.norm(v,axis=1)*np.linalg.norm(w,axis=1);cs=np.divide(np.sum(v*w,axis=1),den,out=np.zeros(n),where=den>0);agreement.append(dict(interval=t,embedding=d,comparable=int(mask.sum()),median_direction_cosine=float(np.median(cs[mask])),fraction_reversed=float((cs[mask]<0).mean())))
 pd.DataFrame(coverage).to_csv(OUT/'projection_coverage.csv',index=False);pd.DataFrame(agreement).to_csv(OUT/'projection_method_agreement.csv',index=False);np.savez_compressed(OUT/'plot_arrow_indices.npz',**selection)
 dump(OUT/'figure_manifest.json',dict(n_peaks=n,background2d=120000,background3d=60000,rate_limit=limit,rms_limit=rmslimit,all_method_net_rates_exactly_equal=True,all_method_rms_rates_exactly_equal=True,all_method_source_gates_exactly_equal=True,arrows='Highest-coherence eligible source per spatial bin, then farthest-point coverage; coherent display examples, not an unbiased sample. Unit-normalized direction, fixed display length.',rates='Equally weighted mean/RMS over valid fitted CT dimensions, linear CPM/hour; not absolute accessibility.',figures=[p.name for p in sorted(OUT.glob('0*.png'))]))
 log('All figures and diagnostic tables complete')

if __name__=='__main__':main()
