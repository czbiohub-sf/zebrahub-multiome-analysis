#!/usr/bin/env python3
"""Redraw a publication-width overview from saved chromatin-dynamics results.
No model fitting or parameter selection; source files are opened read-only.
"""
from pathlib import Path
import hashlib,json
import h5py,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from PIL import Image
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
BASE=HERE/'outputs';OUT=BASE/'chromatin_velocity_figures_170mm_v1';OUT.mkdir(exist_ok=True)
STEM='MAIN_temporal_chromatin_dynamics';W,H=170.,180.
SOURCES={'vectors':BASE/'ct_velocity_flow_sweep_v1/sweep_vectors.h5','sweep':BASE/'ct_velocity_flow_sweep_v1/sweep_metrics.csv','forecast':BASE/'ct_cis_forecast_lagged_v1/forecast_pooled_metrics.csv','color_scale':BASE/'ct_velocity_reprojection_v1/figure_manifest.json'}
source_state={k:{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for k,p in SOURCES.items()}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.labelsize':7.2,'axes.titlesize':8,'xtick.labelsize':6.8,'ytick.labelsize':7,'legend.fontsize':6.8,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','lines.linewidth':1.2})
fig=plt.figure(figsize=(W/25.4,H/25.4))
def axmm(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def label(letter,title,x,y):
 fig.text(x/W,y/H,letter,fontsize=10,weight='bold',va='bottom')
 fig.text((x+6)/W,y/H,title,fontsize=8,weight='bold',va='bottom')
def style(a):
 a.spines[['top','right']].set_visible(False);a.grid(axis='y',color='#e8e8e8',lw=.45,zorder=0);a.set_axisbelow(True)
def spatial(xy,mask,budget=110):
 rng=np.random.default_rng(20261022);valid=np.flatnonzero(mask);lo=xy.min(0);span=np.ptp(xy,axis=0)
 bins=np.floor((xy[valid]-lo)/np.maximum(span,1e-8)*35).astype(int);code=bins[:,0]*36+bins[:,1]
 order=rng.permutation(len(valid));_,first=np.unique(code[order],return_index=True);chosen=valid[order[first]]
 if len(chosen)>budget:chosen=chosen[rng.choice(len(chosen),budget,replace=False)]
 return np.sort(chosen)
fig.text(8/W,177/H,'Temporal chromatin dynamics and limits of velocity interpretation',fontsize=9,weight='bold',va='top')
label('a','Retrospective accessibility-to-direction workflow',4,166)
boxes=[('Peak profiles','190 celltype–stage\npseudobulks'),('Six-stage curve fits','10, 12, 14, 16,\n19 and 24 hpf'),('Five interval rates','Fitted change / Δt\n21 complete celltypes'),('Local correspondence','Fixed peak UMAP\nNo root orientation')]
for i,(title,detail) in enumerate(boxes):
 x=8+40*i
 rect=FancyBboxPatch((x/W,149/H), (35/W),(14/H),boxstyle='round,pad=0.002',transform=fig.transFigure,facecolor='#eef3f6',edgecolor='#b8c8d2',linewidth=.6);fig.add_artist(rect)
 fig.text((x+17.5)/W,160/H,title,ha='center',va='center',weight='bold',fontsize=7)
 fig.text((x+17.5)/W,154/H,detail,ha='center',va='center',fontsize=6.7,linespacing=1.2)
 if i<3:fig.add_artist(FancyArrowPatch(((x+35.5)/W,156/H),((x+39)/W,156/H),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=7,lw=.7,color='#526672'))
with h5py.File(SOURCES['vectors'],'r') as f:
 xy=f['umap2d'][:];source=f['source_indices'][:];ids=f['peak_ids'].asstr()[:];broad=f['broad_reference'][:].astype(bool)
 arrays={'xy':xy,'source_indices':source,'peak_ids':ids.astype(str),'broad_reference':broad}
 maps={}
 for t in [0,3]:
  it=f[f'interval_{t}'];g=it['k30_b000'];active=it['source_active'][:].astype(bool);valid=g['valid2d'][:].astype(bool)
  v=g['vectors2d'][:];rate=it['intrinsic_net_rate'][:];selected=spatial(xy,active);selected=selected[valid[selected]]
  maps[t]=(active,valid,v,rate,selected)
  for name,val in zip(['source_active','valid2d','vectors2d','net_rate','arrow_indices'],[active,valid,v,rate,selected]):arrays[f't{t}_{name}']=val
lim=json.loads(SOURCES['color_scale'].read_text())['rate_limit'];norm=TwoSlopeNorm(0,-lim,lim)
span=np.ptp(xy,axis=0);lower=xy.min(0)-span*.04;upper=xy.max(0)+span*.04;arrowlen=float(np.linalg.norm(span)*.023)
counts={}
for col,(letter,title,t) in enumerate([('b','Broad reference',None),('c','10–12 hpf',0),('d','16–19 hpf',3)]):
 x=5+56*col;a=axmm(x,89,48,50);label(letter,title,x-1,142)
 a.scatter(*xy.T,s=.25,c='#c9cfd3',alpha=.7,lw=0,rasterized=True)
 if t is None:
  a.scatter(*xy[broad].T,s=1.6,c='#c38b25',alpha=.9,lw=0,rasterized=True)
  fig.text((x+24)/W,84/H,'Broadly accessible reference',ha='center',fontsize=6.7,color='#986b18')
  fig.text((x+24)/W,80/H,f'{broad.sum():,} sampled / 20,905 total',ha='center',fontsize=6.5)
 else:
  active,valid,v,rate,selected=maps[t]
  art=a.scatter(*xy[active].T,s=.7,c=rate[active],cmap='RdBu_r',norm=norm,alpha=.88,lw=0,rasterized=True)
  vv=v[selected]/np.linalg.norm(v[selected],axis=1)[:,None]*arrowlen
  a.quiver(*xy[selected].T,*vv.T,angles='xy',scale_units='xy',scale=1,color='#21303b',width=.003,headwidth=3.5,headlength=4.3,minlength=0)
  counts[str(t)]={'source_supported':int(active.sum()),'displayable':int(valid.sum()),'sampled_arrows':len(selected)}
 a.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),aspect='equal',xticks=[],yticks=[]);a.spines[:].set_visible(False)
cb=fig.colorbar(art,cax=axmm(72,85,78,2.3),orientation='horizontal',extend='both');cb.set_ticks([-.8,0,.8]);cb.ax.tick_params(labelsize=6.5,pad=1,size=2);cb.set_label('Mean fitted accessibility change (CPM / hour)',fontsize=6.7,labelpad=1)
fig.text(114/W,74.5/H,'Closing ← 0 → Opening',ha='center',fontsize=6.5)
# Reference-distance diagnostic: fix projectability across all tested settings.
sweep=pd.read_csv(SOURCES['sweep']);q=sweep.query('control=="observed" and stratum=="common_active_all_settings" and k==30 and beta==0').sort_values('interval').copy()
assert len(q)==5
label('e','Apparent flow and graph bias',4,66)
a=axmm(18,20,61,39);style(a)
a.axhline(0,color='#888888',lw=.6,ls='--')
a.plot(range(5),q.mean_root_delta,'o-',ms=3,color='#237d8e',label='Rate-aligned field')
a.plot(range(5),q.mean_root_baseline_delta,'s--',ms=2.7,color='#7b8389',label='Unguided graph baseline')
a.set_xticks(range(5),['10–12','12–14','14–16','16–19','19–24'],rotation=25,ha='right')
a.set_ylabel('Change in reference distance\n(graph hops / step)',labelpad=3);a.set_xlabel('Interval (hpf)',labelpad=2);a.set_ylim(-.18,.025);a.set_yticks([-.15,-.10,-.05,0])
a.legend(loc='upper right',frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=6.5,handlelength=1.6,borderaxespad=.2)
fig.text(18/W,61/H,'Negative = toward broad reference',fontsize=6.5,color='#525a60')
# Forecasting benchmark: paired pooled comparison, existing chromosome bootstrap.
forecast=pd.read_csv(SOURCES['forecast']);models=['cis_current','cis_current_plus_lagged','distance_current_plus_lagged']
r=forecast.query('threshold==0.25 and scope=="all_test_origins"').set_index('model').loc[models].reset_index().copy()
label('f','Does Cicero improve forecasting?',87,66)
a=axmm(115,20,48,39);a.spines[['top','right','left']].set_visible(False);a.tick_params(axis='y',length=0,pad=3)
a.axvline(0,color='#888888',lw=.6,ls='--')
x=100*r.improvement_vs_own_history.to_numpy();lo=100*r.own_history_low.to_numpy();hi=100*r.own_history_high.to_numpy();ys=np.arange(3)[::-1]
for i,(y,color) in enumerate(zip(ys,['#357d8c','#357d8c','#868b90'])):a.errorbar(x[i],y,xerr=[[x[i]-lo[i]],[hi[i]-x[i]]],fmt='o',ms=3.5,color=color,capsize=2,lw=1.2)
a.set_yticks(ys,['Current cis','Current + lag','Matched control']);a.set_ylim(-.6,2.6);a.set_xlim(-2.4,.4);a.set_xticks([-2,-1,0]);a.set_xlabel('Error reduction vs own history (%)',fontsize=6.7,labelpad=3)
fig.text(115/W,61/H,'Cicero ≥ 0.25; 4,525 connected peaks',fontsize=6.5)
fig.text(139/W,9/H,'Positive = improved prediction',ha='center',fontsize=6.5,color='#525a60')
fig.text(8/W,3.5/H,'Fixed peak identities; arrows describe local profile correspondence, not genomic movement.',fontsize=6.7,color='#444c52')
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{STEM}.{ext}',dpi=600)
plt.close(fig)
np.savez_compressed(OUT/f'{STEM}_map_data.npz',**arrays)
q.to_csv(OUT/f'{STEM}_reference_distance.csv',index=False);r.to_csv(OUT/f'{STEM}_forecast.csv',index=False)
caption='''Temporal chromatin dynamics and limits of velocity interpretation. (a) Accessibility profiles across 190 celltype-by-stage pseudobulks define the shared peak atlas. Fits use all six developmental stages; interval rates are fitted consecutive-state differences divided by elapsed time. Projection uses 21 celltypes with adequate coverage at every stage, a fixed log1p-CPM atlas and linear-CPM interval states/rates. It is a retrospective local correspondence between peak profiles, not transport of a genomic locus or a derivative through the UMAP embedding. (b) The distributed broad-accessibility reference on the atlas. Reference membership requires low maximum celltype-specificity tau across stages (bottom 5%), above-median minimum-stage mean CPM and detection in at least 80% of celltypes at each stage; 20,905 peaks meet these conditions. All three maps display the same fixed random sample of 20,000 query peaks from the 640,830-peak atlas. (c,d) Representative interval fields at k = 30 and no additional neighbor-rate blending (beta = 0). Color is mean fitted rate over valid celltype dimensions (linear CPM/hour), with a shared clipped scale of ±0.82139; gray denotes peaks without source-supported change, not demonstrated biological stability. Arrows are sampled spatially without coherence ranking, filtered by the existing display mask and normalized to a fixed display length. Candidate targets are drawn from the full atlas. Opening/closing color and arrow direction are distinct quantities; averaging can cancel celltype-specific changes. No root direction is imposed. (e) Expected change in shortest graph distance to the broad reference, using the same fixed k = 30 undirected atlas graph. Negative values denote steps toward the reference. The rate-aligned field and unguided distance-weighted graph baseline both show an inward tendency. This panel uses source peaks outside the broad reference that are projectable under every tested k/beta setting within each interval: n = 4,117, 3,142, 2,601, 2,424 and 1,001, respectively, drawn from the fixed 20,000 queries. These are selected sensitivity cohorts, not population-wide estimates. Units are expected graph hops per correspondence step, not hops/hour. (f) Reduction in pooled next-change squared error relative to a model using each peak's own current accessibility and previous rate, on 4,525 Cicero-connected peaks at a 0.25 cutoff. Current cis adds the weighted neighbor-current-state residual; current + lag also adds weighted previous neighbor rates. Matched control uses both current and lagged features from a genomic-distance control graph instead of Cicero links. Evaluation uses rolling time splits, held-out chromosomes and targets from the other cell half. Bars are existing 95% chromosome-block bootstrap intervals, not biological-replicate confidence intervals. Negative point estimates provide no overall predictive gain; the current + lag interval includes zero. Persistence also outperforms own history (see supplementary figure). The Cicero graph was estimated using all stages, so this is a conditional fixed-graph test, not a fully prospective graph-learning evaluation. Cell halves do not supply biological replication.'''
(OUT/f'{STEM}_caption.txt').write_text(caption+'\n')
pdf=PdfReader(str(OUT/f'{STEM}.pdf')).pages[0];mm=[float(pdf.mediabox.width)*25.4/72,float(pdf.mediabox.height)*25.4/72]
assert np.allclose(mm,[W,H],atol=.001)
im=Image.open(OUT/f'{STEM}.png');assert im.width==int(W/25.4*600)
assert [int(n) for n in q.n_projectable]==[4117,3142,2601,2424,1001]
assert all(p.stat().st_size==source_state[k]['bytes'] and p.stat().st_mtime_ns==source_state[k]['mtime_ns'] for k,p in SOURCES.items())
manifest={'title':'Temporal chromatin dynamics and limits of velocity interpretation','width_mm':W,'height_mm':H,'pdf_dimensions_mm':mm,'png_pixels':list(im.size),'png_dpi':list(im.info['dpi']),'minimum_font_pt':6.5,'all_peaks':640830,'query_sample':len(xy),'reference_sample':int(broad.sum()),'reference_total':20905,'maps':counts,'rate_color_limit':lim,'display_arrow_length_umap_units':arrowlen,'arrow_sampling':'One random source per occupied spatial bin, capped at 110 before applying original display mask. No coherence ranking. Unit-normalized display.','common_cohort_counts':q.n_projectable.astype(int).tolist(),'source_files_unchanged':True,'sources':source_state,'script':str(Path(__file__).resolve())}
manifest['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.glob(STEM+'*') if p.suffix!='.json'}
(OUT/f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
preview=im.convert('RGB');preview.thumbnail((1500,1600));preview.save(OUT/f'{STEM}_preview.jpg',quality=92)
import subprocess,re,xml.etree.ElementTree as ET
bbox=Path('/tmp/chromatin_main_bbox.html')
subprocess.run(['pdftotext','-bbox',str(OUT/f'{STEM}.pdf'),str(bbox)],check=True)
ns={'h':'http://www.w3.org/1999/xhtml'};page=ET.parse(bbox).getroot().find('.//h:page',ns)
pw,ph=float(page.attrib['width']),float(page.attrib['height']);words=page.findall('.//h:word',ns)
clipped=[w.text for w in words if float(w.attrib['xMin'])<0 or float(w.attrib['yMin'])<0 or float(w.attrib['xMax'])>pw or float(w.attrib['yMax'])>ph]
assert not clipped,clipped
manifest['publication_preflight']={'pdf_words':len(words),'pdf_clipped_words':clipped,'svg_text_nodes':len(re.findall('<text',(OUT/f'{STEM}.svg').read_text())),'pdf_fonts':subprocess.check_output(['pdffonts',str(OUT/f'{STEM}.pdf')],text=True)}
(OUT/f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'dimensions_mm':mm,'png_pixels':im.size,'maps':counts,'preflight':manifest['publication_preflight']}))
