#!/usr/bin/env python3
"""Root-owned scientific display and delivery helpers for the shared-weight pilot."""
import os
from pathlib import Path
import argparse,json,hashlib,html,zipfile
import numpy as np,pandas as pd,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_shared_weight_forecast_v1'
FIG=OUT/'figures'
plt.rcParams.update({'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none'})

def dump(p,obj):
 p.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')

def save(fig,stem):
 for ext,kw in [('png',{'dpi':400}),('pdf',{}),('svg',{})]:fig.savefig(FIG/(stem+'.'+ext),**kw)
 plt.close(fig)

def coverage():
 FIG.mkdir(parents=True,exist_ok=True)
 src=HERE/'outputs/ct_flow_noise_v1/cell_split_input.h5'
 with h5py.File(src) as f:
  a=f['n_cells_a'][:];b=f['n_cells_b'][:];ct=f['celltypes'].asstr()[:];t=f['stages'][:];present=f['column_index'][:]>=0
 n=a+b;valid=n>=20
 assert np.all(np.minimum(a,b)[valid]>=10)
 rows=[dict(celltype=c,stage_hpf=int(t[j]),n_cells=int(n[i,j]),n_half_a=int(a[i,j]),n_half_b=int(b[i,j]),present=bool(present[i,j]),eligible=bool(valid[i,j])) for i,c in enumerate(ct) for j in range(6)]
 pd.DataFrame(rows).to_csv(FIG/'pseudobulk_cell_counts.csv',index=False)
 fig=plt.figure(figsize=(170/25.4,185/25.4));gs=fig.add_gridspec(2,2,width_ratios=[1.22,1],height_ratios=[1,1],left=.225,right=.97,bottom=.13,top=.84,wspace=.4,hspace=.48)
 ax=fig.add_subplot(gs[:,0]);im=ax.imshow(np.ma.masked_equal(n,0),aspect='auto',cmap='YlGnBu',norm=LogNorm(vmin=1,vmax=n.max()))
 ax.set_yticks(range(len(ct)),ct);ax.set_xticks(range(6),t.astype(int));ax.set_xlabel('Developmental stage (hpf)');ax.set_title('A  Cells per pseudobulk',loc='left',pad=11)
 for i,j in zip(*np.where(~valid)):
  ax.text(j,i,'×' if present[i,j] else '–',ha='center',va='center',color='#aa1721',fontsize=9,fontweight='bold')
 cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.035);cb.ax.tick_params(labelsize=6);cb.set_label('Cells (log scale)')
 ax=fig.add_subplot(gs[0,1]);xx=np.arange(6)
 ax.bar(xx,present.sum(axis=0),color='#d9dfe4',label='Observed');ax.bar(xx,valid.sum(axis=0),color='#237b87',label='≥20 cells')
 for j in xx:ax.text(j,valid[:,j].sum()+.5,str(valid[:,j].sum()),ha='center',fontsize=7)
 ax.set_xticks(xx,t.astype(int));ax.set_xlabel('Stage (hpf)');ax.set_ylabel('Celltype × stage groups');ax.set_ylim(0,38);ax.set_title('B  Group inclusion',loc='left',pad=11);ax.legend(frameon=False,loc='lower right',fontsize=6.5);ax.spines[['top','right']].set_visible(False)
 ax=fig.add_subplot(gs[1,1]);g=pd.read_csv(HERE/'outputs/ct_cis_validation_v1/cis_graph_coverage.csv')
 ax.bar(np.arange(len(g)),100*g.fraction_all_peaks_with_edges,color=['#aebdc4','#aebdc4','#237b87','#aebdc4','#aebdc4'])
 for k,row in enumerate(g.itertuples()):ax.text(k,100*row.fraction_all_peaks_with_edges+.06,f'{int(row.peaks_with_edges):,}',ha='center',fontsize=6,rotation=90,va='bottom')
 ax.set_xticks(range(len(g)),g.threshold.astype(str));ax.set_ylim(0,4.6);ax.set_ylabel('Atlas peaks with a partner (%)');ax.set_xlabel('Minimum Cicero score');ax.set_title('C  Graph coverage',loc='left',pad=11);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Reliable pseudobulks and the connected forecast cohort',x=.04,y=.98,ha='left',fontsize=11)
 fig.text(.04,.935,'Selection spans celltype differences and temporal changes.\nPseudobulks with fewer than 20 cells are omitted; missing groups remain missing.',fontsize=8,va='top')
 fig.text(.04,.07,'×  Observed but fewer than 20 cells;  –  absent group.\nAt score ≥0.25, Cicero connects 4,525 of 640,830 atlas peaks (0.71%).\nForecast responses use 21 complete celltypes; peak selection uses all reliable past groups.',fontsize=7,va='top')
 save(fig,'00_data_coverage')
 stats=dict(observed_groups=int(present.sum()),eligible_groups=int(valid.sum()),excluded_observed_groups=int((present&~valid).sum()),absent_groups=int((~present).sum()),minimum_cells_per_half_valid=int(np.minimum(a,b)[valid].min()),complete_celltypes=int(valid.all(axis=1).sum()),past_group_counts={str(int(t[j])):int(valid[:,:j+1].sum()) for j in [2,3,4]},figure_width_mm=170,figure_height_mm=185)
 dump(FIG/'coverage_summary.json',stats);print(json.dumps(stats,indent=2))



def atlas_cohorts(origin=16):
 """Display frozen retrospective vectors; selection alone changes source cohort."""
 FIG.mkdir(parents=True,exist_ok=True)
 sel=OUT/'selection_audit/origin_fold_selection.h5'
 names=['all','gini_top20','total_variance_top20'];labels=['All eligible connected peaks','Top20% Gini','Top20% adjusted total variance']
 with h5py.File(sel) as f:
  ids=f['peak_ids'].asstr()[:];source=f['source_indices'][:];fold=f['chromosome_fold'][:]
  masks={name:np.zeros(len(source),bool) for name in names}
  for k in range(5):
   g=f[f'a/origin_{origin}/fold_{k}'];use=fold==k
   for name in names:masks[name][use]=g[name][:][use]
 with h5py.File(HERE/'outputs/ct_peak_atlas_v1/full_ct_peak_atlas.h5') as f:
  xy=f['umap2d'][:];assert np.array_equal(f['peak_ids'].asstr()[:][source],ids)
 with h5py.File(HERE/'outputs/ct_velocity_reprojection_v1/updated_projection.h5') as f:
  assert np.array_equal(f['peak_ids'].asstr()[:][source],ids)
  stage=f['stages'][:];t=int(np.flatnonzero(stage==origin)[0]);g=f[f'projection/{t}']
  v=g['vectors/umap2d'][:][source];rate=g['net_rate'][:][source]
  valid=g['active'][:][source]&g['projection_valid/umap2d'][:][source]
 norm=np.linalg.norm(v,axis=1);valid&=np.isfinite(v).all(axis=1)&(norm>1e-12)
 # Prespecified deterministic source-id priority; no direction/magnitude ranking.
 priority=np.random.default_rng(20261026).permutation(len(source));bg=np.random.default_rng(20261026).choice(len(xy),50000,replace=False)
 lims=np.array([xy.min(axis=0),xy.max(axis=0)]);pad=.04*(lims[1]-lims[0]);arrow_length=.022*np.ptp(xy[:,0])
 fig,axes=plt.subplots(2,3,figsize=(170/25.4,145/25.4));fig.subplots_adjust(left=.025,right=.99,bottom=.19,top=.82,wspace=.035,hspace=.12)
 rows=[];plotted={}
 for c,(name,label) in enumerate(zip(names,labels)):
  mask=masks[name];keep=priority[mask[priority]&valid[priority]][:180];plotted[name]=source[keep]
  for row in range(2):
   ax=axes[row,c];ax.scatter(xy[bg,0],xy[bg,1],s=.3,c='#d3d9dc',alpha=.3,rasterized=True,edgecolors='none');ax.set_xlim(lims[0,0]-pad[0],lims[1,0]+pad[0]);ax.set_ylim(lims[0,1]-pad[1],lims[1,1]+pad[1]);ax.set_aspect('equal');ax.set_axis_off()
  axes[0,c].scatter(xy[source[mask],0],xy[source[mask],1],s=1.5,c=['#47596b','#925094','#d0842f'][c],alpha=.6,edgecolors='none',rasterized=True)
  axes[0,c].set_title(label.replace('Top20','Top 20')+'\n'+f'{mask.sum():,} sources',fontsize=7.3,pad=3)
  direction=np.divide(v[keep],norm[keep,None])*arrow_length
  colors=np.where(rate[keep]>0,'#c66b22',np.where(rate[keep]<0,'#277a9b','#777777'))
  axes[1,c].quiver(xy[source[keep],0],xy[source[keep],1],direction[:,0],direction[:,1],color=colors,angles='xy',scale_units='xy',scale=1,width=.004,headwidth=3.5,headlength=4.5,alpha=.8)
  axes[1,c].text(.5,-.03,f'{int((mask&valid).sum()):,} supported; {len(keep)} shown',ha='center',transform=axes[1,c].transAxes,fontsize=6.5)
  rows.append(dict(cohort=name,origin_hpf=origin,end_hpf=int(stage[t+1]),n_selected=int(mask.sum()),n_supported_projectable=int((mask&valid).sum()),n_arrows_shown=len(keep)))
 fig.suptitle('What peak selection changes on the existing atlas',x=.03,y=.98,ha='left',fontsize=11)
 fig.text(.03,.92,f'Selection: past source-half A through {origin} hpf, thresholds from other chromosomes.\nBottom row: existing retrospective {origin}→{int(stage[t+1])} hpf arrows on identical coordinates.',fontsize=8,va='top')
 fig.text(.03,.115,'Orange: net opening; blue: net closing. Arrow lengths are normalized for display.\nAt most 180 arrows per panel, chosen by the same fixed source-ID priority.\nThese reuse six-stage fitted rates and full-course UMAP neighbors; they are not forecasts.\nFiltering visible sources does not change neighbor partners or establish a global flow.',fontsize=7,va='top')
 save(fig,'01_variable_peak_fixed_atlas')
 pd.DataFrame(rows).to_csv(FIG/'atlas_cohort_display_counts.csv',index=False)
 np.savez_compressed(FIG/'atlas_cohort_plot_data.npz',source_indices=source,xy=xy[source],background_xy=xy[bg],vectors=v,net_rate=rate,valid=valid,**{name+'_mask':mask for name,mask in masks.items()},**{name+'_displayed_source_indices':p for name,p in plotted.items()})
 dump(FIG/'atlas_display_validation.json',dict(status='passed',peak_id_alignment=True,frozen_coordinates=True,reused_retrospective_vectors=True,new_forecasts_displayed=False,source_half='a',selection_origin_hpf=origin,selection_thresholds='Outer training chromosomes only; OOF target masks assembled across 5 folds',vector_source='ct_velocity_reprojection_v1/updated_projection.h5',max_arrows=180,arrow_length_umap_units=float(arrow_length),source_selection_rule='Fixed random priority over source IDs, independent of rates/outcomes',cohorts=rows,figure_width_mm=170,figure_height_mm=145))
 print(pd.DataFrame(rows).to_string(index=False))


def package_milestone(tag,title,summary,detailed,figures,extra_files=()):
 """Build portable HTML and an auditable attachment package; never send itself."""
 import base64
 mail=OUT/'email_package'/tag;mail.mkdir(parents=True,exist_ok=True)
 (mail/'high_level_summary.txt').write_text(summary+'\n')
 (mail/'detailed_summary.txt').write_text(detailed+'\n')
 sections=[]
 for p,caption in figures:
  b64=base64.b64encode(p.read_bytes()).decode()
  sections.append('<section><h2>'+html.escape(p.stem.replace('_',' '))+'</h2><img src="data:image/png;base64,'+b64+'" alt="'+html.escape(caption)+'"><p>'+html.escape(caption)+'</p></section>')
 paras=''.join('<p>'+html.escape(p)+'</p>' for p in summary.split('\n\n'))
 report='<!doctype html><html><head><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{max-width:1050px;margin:35px auto;padding:0 20px;font:16px/1.55 Arial;color:#20303b}h1{font-size:29px}h2{font-size:21px}img{max-width:100%}section{margin:40px 0}section p{font-size:14px}pre{white-space:pre-wrap;font:14px/1.5 Arial;background:#f2f5f6;padding:20px}</style></head><body><h1>'+html.escape(title)+'</h1><p>26 October 2026 · Chromatin dynamics</p>'+paras+''.join(sections)+'<h2>Detailed methods, results and interpretation</h2><pre>'+html.escape(detailed)+'</pre></body></html>'
 reportpath=mail/(tag+'_report.html');reportpath.write_text(report)
 body='<!doctype html><html><body style="font:15px/1.55 Arial,sans-serif;color:#20303b;max-width:950px"><p>Hello,</p><h2>'+html.escape(title)+'</h2>'+paras+'<p><b>Attached:</b> plots as PNGs, a self-contained HTML report with the high-level and detailed summaries, and a ZIP with PDFs, source tables, validation records and scripts. The original analysis inputs remain in the workspace.</p><p>Workspace report: '+html.escape(str(reportpath.resolve()))+'</p></body></html>'
 (mail/'email_body.html').write_text(body)
 archive=mail/(tag+'_analysis_package.zip')
 entries=[(mail/'high_level_summary.txt','high_level_summary.txt'),(mail/'detailed_summary.txt','detailed_summary.txt')]
 for p,caption in figures:
  pdf=p.with_suffix('.pdf')
  if pdf.exists():entries.append((pdf,'figures/'+pdf.name))
 for p in extra_files:
  p=Path(p)
  if p.exists():
   try:name=str(p.relative_to(OUT))
   except ValueError:name='scripts/'+p.name
   entries.append((p,name))
 entries=list({name:(p,name) for p,name in entries}.values())
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p,name in entries:z.write(p,name)
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 files=[(p,'image/png') for p,_ in figures]+[(reportpath,'text/html'),(archive,'application/zip')]
 attachments=[dict(path=str(p.resolve()),filename=p.name,mime_type=mime,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,mime in files]
 manifest=dict(to=os.environ.get('CHROMATIN_REPORT_RECIPIENT', ''),subject='Re: Real chromatin velocity: peak UMAPs, interactive atlas, and depth-matching interpretation',milestone=tag,attachments=attachments,binary_bytes=sum(p['bytes'] for p in attachments),archive_entries=len(entries),archive_test='passed')
 assert manifest['binary_bytes']<18_000_000
 dump(mail/'delivery_manifest.json',manifest)
 print(json.dumps(manifest,indent=2))
 return reportpath


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['coverage','atlas']);args=parser.parse_args()
 if args.stage=='coverage':coverage()
 elif args.stage=='atlas':atlas_cohorts()
