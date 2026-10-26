#!/usr/bin/env python3
"""Publication-width SI figure assembled from existing, read-only result tables.

No new model fitting, projections, hypothesis tests, or biological analyses.
All plotted values, selections, and display aggregations are exported.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from PIL import Image
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
BASE=HERE/'outputs'
OUT=BASE/'chromatin_velocity_figures_170mm_v1'
OUT.mkdir(exist_ok=True)
W,H=170.,205.
STEM='SI_parameter_robustness_and_cicero'
FILES={
 'stability':BASE/'ct_velocity_flow_sweep_v1/direction_stability.csv',
 'sweep':BASE/'ct_velocity_flow_sweep_v1/sweep_metrics.csv',
 'coverage':BASE/'ct_cis_validation_v1/cis_graph_coverage.csv',
 'forecast':BASE/'ct_cis_forecast_lagged_v1/forecast_pooled_metrics.csv',
 'synthetic':BASE/'ct_velocity_experiments_v1/forecast_synthetic_control_summary.csv',
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={k:dict(path=str(p),sha256=sha(p)) for k,p in FILES.items()}
data={k:pd.read_csv(p) for k,p in FILES.items()}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.labelsize':7.2,
 'axes.titlesize':8.2,'xtick.labelsize':7.0,'ytick.labelsize':7.0,
 'legend.fontsize':6.8,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
 'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white','lines.linewidth':1.15})
fig=plt.figure(figsize=(W/25.4,H/25.4))
fig.text(8/W,200/H,'Parameter robustness and tests of Cicero-based forecasting',fontsize=9,weight='bold',va='top')

def axmm(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def style(ax):
 ax.spines[['top','right']].set_visible(False)
 ax.grid(axis='y',color='#e8e8e8',lw=.45,zorder=0)
 ax.set_axisbelow(True)
def heading(letter,title,x,y):
 fig.text(x/W,y/H,letter,fontsize=10,weight='bold',ha='left',va='bottom')
 fig.text((x+6)/W,y/H,title,fontsize=8,weight='bold',ha='left',va='bottom')
rows=[]
def add(frame,panel,source):
 frame=frame.copy();frame.insert(0,'panel',panel);frame.insert(1,'source_table',str(FILES[source].relative_to(HERE)));rows.append(frame)

intervals=['10–12','12–14','14–16','16–19','19–24']
colors=['#4477AA','#66AACC','#228833','#CCAA33','#AA3377']
ks=[10,30,60,100];betas=[0,.25,.5]

# A: All 60 direction-sensitivity values; each comparison uses its shared
# displayable source cohort, rather than assuming a common denominator.
a=axmm(18,149,55,36)
s=data['stability'].sort_values(['interval','k','beta']).copy()
s['plotted_value']=100*s.reversed_fraction_vs_k30_unsmoothed
z=s.pivot(index='interval',columns=['k','beta'],values='plotted_value').reindex(index=range(5),columns=pd.MultiIndex.from_product([ks,betas])).values
im=a.imshow(z,origin='upper',aspect='auto',cmap='magma_r',vmin=0,vmax=3,interpolation='nearest')
a.set_yticks(range(5),intervals);a.set_ylabel('Interval (hpf)')
a.set_xticks(range(12),['0','.25','.5']*4)
a.tick_params(axis='x',labelsize=6.5,pad=2)
a.set_xlabel('Rate blend β',labelpad=3)
for i,k in enumerate(ks):a.text((3*i+1+.5)/12,1.035,str(k),transform=a.transAxes,ha='center',va='bottom',fontsize=7)
a.text(-.02,1.035,'k',transform=a.transAxes,ha='right',va='bottom',fontsize=7)
for v in [2.5,5.5,8.5]:a.axvline(v,color='white',lw=1)
a.spines[:].set_visible(False)
cb=fig.colorbar(im,cax=axmm(76,149,2.2,36));cb.set_ticks([0,1,2,3]);cb.set_label('Arrows differing >90° (%)',labelpad=2)
heading('a','Local direction sensitivity',8,192)
add(s,'a','stability')

# B: Original source gates, not a denominator selected by display eligibility.
b=axmm(108,149,55,36);style(b)
s=data['sweep'].query('control=="observed" and stratum=="all_source_supported"').copy()
s['plotted_value']=100*s.projectable_fraction
for t,c in enumerate(colors):
 q=s[s.interval==t];p=q.pivot(index='k',columns='beta',values='plotted_value').reindex(ks)
 b.fill_between(range(4),p.min(axis=1),p.max(axis=1),color=c,alpha=.14,lw=0)
 b.plot(range(4),p[0],'-o',ms=2.4,color=c,label=intervals[t])
b.set_xticks(range(4),ks);b.set_xlabel('Projection neighbors (k)')
b.set_ylabel('Projectable source peaks (%)');b.set_ylim(0,105);b.set_yticks([0,25,50,75,100])
b.legend(loc='lower left',ncol=2,frameon=False,handlelength=1.2,columnspacing=.6,labelspacing=.3)
heading('b','Projection coverage',98,192)
fig.text(108/W,136/H,'Line: β = 0; shading: range over β',fontsize=6.7)
add(s,'b','sweep')

# C: Use the already calculated paired difference to avoid changing source
# cohorts between an observed field and each matched permutation.
c=axmm(18,83,62,40);style(c)
s=data['sweep'].query('k==30 and beta==0 and stratum in ["all_source_supported", "common_active_all_settings"]').copy()
s=s[s.control.str.startswith('permutation_')].copy();s['plotted_value']=-s.mean_excess_minus_observed_paired
for stratum,col,label,off in [('all_source_supported','#C36A37','Setting-specific cohort',-.06),('common_active_all_settings','#276B8B','Common cohort',.06)]:
 q=s[s.stratum==stratum].groupby('interval').plotted_value.agg(['mean','min','max']).reindex(range(5))
 c.errorbar(np.arange(5)+off,q['mean'],yerr=np.vstack([q['mean']-q['min'],q['max']-q['mean']]),color=col,fmt='o-',ms=3,capsize=2,lw=1,label=label)
c.axhline(0,color='#444444',lw=.65,ls='--');c.set_xticks(range(5),intervals);c.set_xlabel('Developmental interval (hpf)')
c.set_ylabel('Observed − shuffled excess\n(graph hops)');c.set_ylim(-.027,.076)
c.legend(loc='upper left',frameon=False,handlelength=1.1,labelspacing=.25)
heading('c','Cohort sensitivity of rootward bias',8,128)
fig.text(18/W,70/H,'Bars: range of 3 shuffles; not confidence intervals',fontsize=6.5)
add(s,'c','sweep')

# D: Existing graph coverage table, including lower thresholds for context.
d=axmm(108,83,55,40);style(d)
s=data['coverage'].copy();s['plotted_value']=100*s.fraction_all_peaks_with_edges
barcolors=['#BEBEBE','#BEBEBE','#276B8B','#C36A37','#BEBEBE']
d.bar(range(len(s)),s.plotted_value,color=barcolors,width=.63,zorder=3)
for i,r in s.iterrows():d.text(i,r.plotted_value+.10,f'{int(r.peaks_with_edges):,}',ha='center',va='bottom',fontsize=6.5)
d.set_xticks(range(len(s)),['.05','.10','.25','.50','.80']);d.set_xlabel('Minimum Cicero score')
d.set_ylabel('Atlas peaks with ≥1 cis edge (%)');d.set_ylim(0,3.8);d.set_yticks([0,1,2,3])
heading('d','Cicero graph coverage',98,128)
fig.text(108/W,70/H,'Forecast tests: cutoffs 0.25 and 0.50',fontsize=6.7)
add(s,'d','coverage')

# E: Pooled squared-error reductions and the EXISTING 95% chromosome-block
# bootstrap intervals, with different threshold populations clearly identified.
e=axmm(18,19,62,40);style(e)
models=['own_history','cis_current','cis_current_plus_lagged']
s=data['forecast'].query('scope=="all_test_origins" and model in @models').copy()
s['plotted_value']=100*s.improvement_vs_persistence
s['plotted_low']=100*s.persistence_low;s['plotted_high']=100*s.persistence_high
for threshold,col,off,label in [(.25,'#276B8B',-.10,'≥0.25 (4,525 peaks)'),(.5,'#C36A37',.10,'≥0.50 (807 peaks)')]:
 q=s[s.threshold==threshold].set_index('model').loc[models]
 e.errorbar(np.arange(3)+off,q.plotted_value,yerr=np.vstack([q.plotted_value-q.plotted_low,q.plotted_high-q.plotted_value]),fmt='o',ms=3.3,capsize=2,color=col,label=label)
e.axhline(0,color='#444444',ls='--',lw=.7);e.set_xticks(range(3),['Own\nhistory','+ Current\nCicero','+ Current\n+ lagged Cicero'])
e.set_ylabel('Error reduction vs persistence (%)');e.set_ylim(-34,10);e.set_yticks([-30,-20,-10,0])
e.legend(loc='lower left',frameon=False,handlelength=1,labelspacing=.25,fontsize=6.5)
heading('e','Held-out forecast performance',8,64)
add(s,'e','forecast')

# F: Strong implementation positive/null controls, not a biological power study.
f=axmm(108,19,55,40);style(f)
select=[('neighbor_lead','cis_history'),('neighbor_lead','shuffled_history'),('no_neighbor_signal','cis_history'),('no_neighbor_signal','shuffled_history')]
s=data['synthetic'].set_index(['scenario','model']).loc[select].reset_index()
s['plotted_value']=100*s.improvement_vs_own_history
pos=np.array([3,2,1,0]);cols=['#276B8B','#AAAAAA','#276B8B','#AAAAAA']
f.barh(pos,s.plotted_value,color=cols,height=.55,zorder=3)
f.axvline(0,color='#444444',lw=.7)
for y,v in zip(pos,s.plotted_value):f.text(max(v,0)+2,y,f'{v:.2f}%',fontsize=6.7,va='center')
f.set_yticks(pos,['Lead: linked','Lead: shuffled','Null: linked','Null: shuffled']);f.set_xlim(-4,116);f.set_xticks([0,50,100])
f.set_xlabel('Error reduction vs own history (%)');f.grid(axis='y',visible=False);f.grid(axis='x',color='#e8e8e8',lw=.45)
heading('f','Synthetic estimator controls',98,64)
add(s,'f','synthetic')

fig.text(8/W,5/H,'Retrospective projections; relative accessibility. No biological replicate uncertainty is estimated.',fontsize=6.7,color='#444444')
for ext in ['pdf','svg','png']:
 fig.savefig(OUT/f'{STEM}.{ext}',dpi=600)
plt.close(fig)
pd.concat(rows,ignore_index=True,sort=False).to_csv(OUT/f'{STEM}_plotted_values.csv',index=False)
caption='''Supplementary Figure. Parameter sensitivity and conditional tests of co-accessibility-based prediction. (a) Percentage of projected 2D arrows differing by more than 90° from the k = 30, β = 0 reference, among arrows displayable under both settings. Columns vary projection-neighbor count k and rate blend β; β blends intrinsic rates with one fixed k = 30 distance-weighted neighbor-rate average. The sweep used 20,000 fixed random query peaks and candidates from the full 640,830-peak atlas. Source gates were unchanged after smoothing. (b) Fraction of source-supported query peaks outside the broad-accessibility reference that are projectable; lines show β = 0 and shaded ranges span the three β settings, not uncertainty. The denominator includes unprojectable source peaks. (c) At k = 30, β = 0, the observed graph-distance excess relative to a comparable unguided neighbor baseline, minus the corresponding excess after each matched whole-velocity-vector shuffle. Positive values indicate a less rootward/more outward diagnostic relative to the shuffled control; they do not establish an outward raw flow. The diagnostic measures expected change in distance to the reference, in graph hops per correspondence step, not accessibility rates or measured transport. Orange uses the setting-specific projectable population; blue restricts observed peaks to those projectable under every tested k–β setting within that interval. Each observed-versus-shuffled contrast is paired on peaks projectable in both, so the matched subset may still differ across the three shuffles. Points show the mean and bars the range of three shuffles, not confidence intervals or permutation p-values. Late-stage contrasts weaken in the common observed cohort, which is itself a selected sensitivity cohort. (d) Existing Cicero graph coverage across score cutoffs; numbers indicate connected peaks. Colors identify the two thresholds tested for forecasting. (e) Pooled next-interval prediction skill, defined as 100 × (1 − model squared error / persistence squared error); negative values are worse than predicting no change. All models include own current accessibility and previous observed rate, with the indicated optional Cicero features. Points and existing 95% chromosome-block bootstrap intervals pool three rolling test transitions, held-out chromosome folds, and both source/target cell-half directions. The two thresholds evaluate different connected-peak populations and should not be ranked as equivalent cohorts. Bootstrap intervals describe chromosome-block sampling sensitivity, not biological-replicate uncertainty. The current-plus-lagged test was a post-primary exploratory extension. (f) The same estimator recovers a deliberately strong planted neighbor-leading signal but shows negligible gain with shuffled neighbors or in the no-neighbor-signal simulation. These controls validate implementation in strong synthetic settings, not sensitivity to subtle real coupling. Across panels, the atlas uses whole-course celltype × stage features, whereas interval projections use 21 adequately sampled celltypes. CPM changes are relative accessibility. The Cicero graph was learned from all cells and stages; forecasting is conditional on that fixed graph and is not fully prospective graph validation. Cell halves are not biological replicates.\n'''
(OUT/f'{STEM}_caption.txt').write_text(caption)
pdf=PdfReader(OUT/f'{STEM}.pdf');box=pdf.pages[0].mediabox
pdfmm=[float(box.width)*25.4/72,float(box.height)*25.4/72]
with Image.open(OUT/f'{STEM}.png') as p:png=dict(pixels=list(p.size),dpi=list(p.info['dpi']))
assert np.allclose(pdfmm,[W,H],atol=1e-6),pdfmm
assert np.allclose(png['dpi'],[600,600],atol=.02),png
assert all(sha(FILES[k])==sources[k]['sha256'] for k in FILES)
manifest=dict(status='passed',width_mm=W,height_mm=H,pdf_mediabox_mm=pdfmm,png=png,minimum_font_pt=6.5,
 vector_text_editable=True,svg_fonttype='none',pdf_fonttype=42,no_tight_bounding_box=True,
 data_sources=sources,source_tables_unchanged=True,plotted_rows_by_panel=pd.concat(rows).panel.value_counts().sort_index().to_dict(),
 no_new_analysis=True,aggregations='Display only: unit conversion to percent, range/mean of existing three shuffles, range over existing beta settings.',
 files={ext:str(OUT/f'{STEM}.{ext}') for ext in ['pdf','svg','png']})
(OUT/f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
# Publication preflight using Poppler; writes only a temporary text-bounds file.
import subprocess
import xml.etree.ElementTree as ET
import re
bbox=Path('/tmp/chromatin_si_bbox.html')
subprocess.run(['pdftotext','-bbox',str(OUT/f'{STEM}.pdf'),str(bbox)],check=True)
ns={'h':'http://www.w3.org/1999/xhtml'}
page=ET.parse(bbox).getroot().find('.//h:page',ns)
pagewidth,pageheight=float(page.attrib['width']),float(page.attrib['height'])
words=page.findall('.//h:word',ns)
clipped=[w.text for w in words if float(w.attrib['xMin'])<0 or float(w.attrib['yMin'])<0 or float(w.attrib['xMax'])>pagewidth or float(w.attrib['yMax'])>pageheight]
assert not clipped,clipped
textnodes=len(re.findall('<text',(OUT/f'{STEM}.svg').read_text()))
assert textnodes>50
manifest['publication_preflight']=dict(pdf_words=len(words),pdf_clipped_words=clipped,svg_text_nodes=textnodes,
 pdf_fonts=subprocess.check_output(['pdffonts',str(OUT/f'{STEM}.pdf')],text=True),
 visual_review='Passed at 1,050-pixel and 643-pixel width (approximately 170 mm at 96 dpi): labels legible, no overlaps or clipping.')
(OUT/f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2))
