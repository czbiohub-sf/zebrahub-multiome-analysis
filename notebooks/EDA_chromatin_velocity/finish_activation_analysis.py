"""Make focused timing/robustness figures and a reviewable activation report."""
from pathlib import Path
import argparse,json,html,hashlib
import h5py,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from activation_progression import OUT,SOURCE,HERE,ROOT

def save(fig,out,stem):
 fig.savefig(out/(stem+'.png'),dpi=190,bbox_inches='tight',facecolor='white');fig.savefig(out/(stem+'.pdf'),bbox_inches='tight');plt.close(fig)

def figures(out):
 selected=['enteric_neurons','hematopoietic_vasculature','differentiating_neurons','spinal_cord']
 with h5py.File(out/'activation_metrics.h5','r') as f,h5py.File(out/'geometry.h5','r') as g:
  ct=f['celltypes'].asstr()[:];distance=g['distance_hops'][:]
  fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
  for ax,name in zip(axes.flat,selected):
   c=np.flatnonzero(ct==name)[0];age=f['opening_center_hpf'][:,c];use=np.isfinite(age)
   rng=np.random.default_rng(42);rows=np.flatnonzero(use);rows=rng.choice(rows,min(len(rows),4000),replace=False)
   ax.scatter(distance[rows]+rng.uniform(-.12,.12,len(rows)),age[rows],s=3,c='#659eb5',alpha=.14,rasterized=True)
   bins=[];middle=[];low=[];high=[]
   for d in np.unique(distance[use]):
    y=age[use&(distance==d)]
    if len(y)>=30:bins.append(d);middle.append(np.median(y));low.append(np.quantile(y,.25));high.append(np.quantile(y,.75))
   ax.fill_between(bins,low,high,color='#126d80',alpha=.18,label='Within-bin interquartile range');ax.plot(bins,middle,'o-',c='#126d80',lw=2,label='Within-bin median')
   rho=spearmanr(distance[use],age[use]).statistic
   ax.set_title(f'{name.replace("_"," ")}\nn={use.sum():,} resolved peaks; descriptive Spearman ρ={rho:.3f}')
   ax.set_xlabel('Graph hops from nearest broad reference');ax.set_ylabel('Center of fitted opening activity (hpf)');ax.spines[['top','right']].set_visible(False)
  axes.flat[0].legend(loc='upper left',fontsize=8)
  fig.suptitle('Opening schedules differ across celltypes',fontsize=17)
  fig.text(.5,.018,'Supported, nonuniform fitted trajectories only. Constant-rate rises have no resolved event age. Points jittered horizontally for display.\nShading is the distribution of peaks, not a confidence interval. These selected examples do not establish onset, peak movement, or a universal wave.',ha='center',fontsize=10)
  fig.subplots_adjust(top=.88,bottom=.13,hspace=.35,wspace=.19);save(fig,out,'09_opening_window_positions')
 tests=pd.read_csv(out/'outward_tests.csv');weights=pd.read_csv(out/'fit_weight_progression_sensitivity.csv')
 raw=tests.query('scope=="celltype" and estimator=="observed" and metric=="hops"').set_index('celltype')
 fit=tests.query('scope=="celltype" and estimator=="fitted" and metric=="hops"').set_index('celltype').loc[raw.index]
 w=weights.query('metric=="hops"').set_index('celltype').loc[raw.index]
 fig,axes=plt.subplots(1,3,figsize=(16,5))
 highlights={'enteric_neurons':'#a23c63','hematopoietic_vasculature':'#1b8892','differentiating_neurons':'#db8a26'}
 for ax,x,y,title,xlabel,ylabel in [(axes[0],raw.slope,fit.slope,'Observed versus fitted centers','Observed slope (hops/hpf)','Fitted slope (hops/hpf)'),(axes[1],w.coverage_slope,w.equal_slope,'Fitting-weight sensitivity · 8,192 peaks','Coverage-weighted fit slope (hops/hpf)','Equal-stage fit slope (hops/hpf)')]:
  ax.axhline(0,c='.6',lw=.8);ax.axvline(0,c='.6',lw=.8)
  lim=max(np.abs(np.r_[x,y]))*1.15;ax.plot([-lim,lim],[-lim,lim],ls=':',c='.6');ax.scatter(x,y,c='#788898',s=35)
  for name,color in highlights.items():ax.scatter(x[name],y[name],c=color,s=65,label=name.replace('_',' '))
  ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.spines[['top','right']].set_visible(False)
 axes[0].legend(fontsize=8,loc='upper left')
 labels=['Observed\npositive slope','Fitted\npositive slope','Both weight fits\nsame slope sign'];counts=[int((raw.slope>0).sum()),int((fit.slope>0).sum()),int(w.same_slope_sign.sum())]
 axes[2].bar(labels,counts,color=['#24798c','#a486a5','#888888']);axes[2].set_ylim(0,23);axes[2].set_ylabel('Celltypes (out of 21)');axes[2].set_title('Large trends survive; weak trends change')
 for i,n in enumerate(counts):axes[2].text(i,n+.4,f'{n}/21',ha='center')
 fig.text(.5,.012,'Primary ordering controls use observed data. Fitted curves are sensitivity summaries; no post-fit permutation test is used.\nWeight comparison uses a fixed uniform peak sample; neither CPM normalization nor atlas geometry changes.',ha='center',fontsize=9)
 fig.subplots_adjust(bottom=.20,wspace=.35);save(fig,out,'10_fit_weight_progression_sensitivity')
 # Export actual examples with measured/fitted values for direct follow-up.
 meta=pd.read_csv(HERE/'outputs/trajectory_velocity_v1/peak_metadata.csv.gz',index_col=0,keep_default_na=False)
 rows=[]
 with h5py.File(SOURCE,'r') as f,h5py.File(out/'activation_metrics.h5','r') as a,h5py.File(out/'geometry.h5','r') as g:
  ct=f['celltypes'].asstr()[:];ids=f['peak_ids'].asstr()[:];d=g['distance_hops'][:];regions=g['region_code'][:];rl=g['region_labels'].asstr()[:]
  for name in selected:
   c=np.flatnonzero(ct==name)[0];age=a['opening_center_hpf'][:,c];rr=f['rates'][:,c,:];change=np.sum(np.maximum(rr,0)*np.diff(f['stages'][:]),axis=1)
   idx=np.flatnonzero(np.isfinite(age));idx=idx[np.argsort(change[idx])[-50:]];idx.sort()
   for i in idx:
    row=dict(celltype=name,peak=ids[i],region=rl[regions[i]],distance_hops=d[i],opening_window_center_hpf=age[i],fitted_positive_change_cpm=change[i])
    for col in ['nearest_gene','linked_gene','peak_type_argelaguet']:row[col]=meta.iloc[i][col]
    for t,h in enumerate(f['stages'][:].astype(int)):row[f'observed_{h}hpf']=f['observed_cpm'][i,c,t];row[f'fitted_{h}hpf']=f['fitted_cpm'][i,c,t]
    rows.append(row)
 pd.DataFrame(rows).to_csv(out/'illustrative_peak_profiles.csv',index=False)

def report(out):
 tests=pd.read_csv(out/'outward_tests.csv');global_row=tests.query('scope=="global" and estimator=="observed" and metric=="hops"').iloc[0]
 ct=tests.query('scope=="celltype" and estimator=="observed" and metric=="hops"').sort_values('slope',ascending=False)
 timing=pd.read_csv(out/'opening_closing_timing.csv');weights=json.loads((out/'fit_weight_sensitivity_summary.json').read_text());geo=json.loads((out/'geometry_summary.json').read_text());ts=json.loads((out/'temporal_order_summary.json').read_text())
 row_table=ct[['celltype','slope','correlation','endpoint_shift','p_outward','p_fwer_two_sided']].copy();row_table.columns=['Celltype','Slope (hops/hpf)','Time correlation','10–24 shift (hops)','Outward ordering tail','Max-stat two-sided ordering tail']
 finding='Relative accessibility shifts toward more distant atlas regions in several celltypes, most clearly enteric-neuron and hematopoietic/vascular pseudobulks. A coordinated root-to-branch opening wave is not established.'
 detail=f"The observed mean accessibility-weighted graph distance increases by {global_row.endpoint_shift:.3f} hops from 10 to 24 hpf. Its actual-time correlation is {global_row.correlation:.3f}, but {round(global_row.p_outward*720)}/720 possible stage orderings are at least as outward (tail {global_row.p_outward:.3f}). Restricting the reference to its concentrated core weakens this ordering contrast (tail 0.104)."
 body=f'''<h1>Celltype-specific activation across the peak atlas</h1><p class="lead">{html.escape(finding)}</p><p>{html.escape(detail)}</p>
 <h2>What changed in this analysis</h2><p>The existing 640,830-peak atlas remains fixed. We show accessibility at each of six stages and opening/closing separately for each celltype. The primary progression statistic is measured accessibility-weighted graph distance from broadly accessible reference peaks:</p><p class="formula">distance center(c,t) = Σᵢ accessibility(i,c,t) × graph distance(i) / Σᵢ accessibility(i,c,t)</p><p>Primary temporal controls use the 21 celltypes with at least 20 cells at every stage; the remaining celltypes are descriptive. Missing combinations remain masked. The same peak panel is used at every stage. A common multiplicative scaling factor for a pseudobulk column cancels exactly; this statistic is identical from counts or CPM on that panel. It measures redistribution across the atlas. That redistribution can arise from programs farther from the atlas reference opening, nearer programs closing, or both. Uniform multiplicative opening everywhere leaves it unchanged. Coverage can still affect sampling noise and technical bias.</p>
 <h2>Results</h2><ul><li><b>Global trend is modest.</b> 20/21 measured CT slopes are positive, but those CTs share stages and are dependent. Under the shared stage-order reference, the tail for 20/21 positive slopes is 0.239. Counting them as 21 independent observations would overstate evidence.</li><li><b>Enteric neurons are the clearest convergent example.</b>Measured distance–time correlation 0.974, endpoint shift +0.448 hops; the fitted opening-window/distance association is 0.442 across 3,970 resolved peaks. The outward direction remains under weighted graph distances and the concentrated-core reference.</li><li><b>Hematopoietic/vascular pseudobulks also show a clear measured outward trend</b> (correlation 0.952), with a weaker opening-window association 0.166. Differentiating neurons show measured correlation 0.921 but much weaker timing association 0.070.</li><li><b>Opening schedules are heterogeneous.</b>Most celltype timing–distance associations are small or mixed; spinal cord is negative (−0.280). The data do not support universally later opening at greater distance.</li><li><b>Smoothing remains consequential.</b>Only 10/21 full fitted-center slopes are positive, versus 20/21 measured. Coverage versus equal-stage fits agree in slope sign for 13/21 CTs on the fixed 8,192-peak sample. The three highlighted candidates retain positive slopes under both weighting choices; weaker directions change.</li></ul>
 <h2>All primary celltype results</h2>{row_table.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}
 <p>Ordering-tail fractions compare actual time ordering to all 720 shared stage-label permutations. One permutation is applied across all series together, preserving cross-celltype and cross-region structure. The max-stat reference accounts for examining 21 CTs in that metric family. These are conditional, descriptive controls—not biological-replicate p-values. Geometry alternatives are sensitivity analyses, not separate discoveries.</p>
 <h2>How the reference and regions were defined</h2><p>The broad reference contains 20,905 peaks, concentrated in four Leiden regions but present in 29/36. Graph distance is measured to the nearest reference peak. We also used edge-affinity costs and a core-only reference retaining 19,178 peaks in regions 26, 33, 23, 11 (91.74% of references). Sparse peripheral reference peaks substantially shorten distance; median distance changes from 4 to 6 hops under the core definition. Neither distance is developmental pseudotime. Existing Leiden regions are descriptive atlas neighborhoods, not validated branchlets or lineages.</p>
 <h2>Timing means a window, not an onset</h2><p>For supported nonuniform fitted trajectories, opening-window center is the midpoint-weighted mean of all positive fitted accessibility increments; closing uses negative increments. Count, fit-quality and amplitude gates apply. Straight-line rises have no resolved event age because their midpoint would simply be the midpoint of the observation window. Stable references receive no artificial earliest time. Timing associations use selected, spatially dependent peaks and are descriptive; no independent-peak significance is claimed. Narrow pulses can be missed by the existing six-stage models.</p>
 <h2>What this does and does not establish</h2><p>The branch-like atlas can organize celltype-specific regulatory schedules. A stable accessible core can coexist with later activation of other programs. This does not imply that genomic peaks move, transform into one another, originate from the reference, or causally activate other peaks. CPM levels remain relative. The graph was constructed from these same pseudobulk profiles, so geometry-conditioned temporal ordering is not independent biological validation. Stage-specific technical effects and absent biological replication remain limitations. This pass uses all atlas peaks; it does not isolate promoter/intergenic enhancers or establish cis regulatory coupling. Region names do not confer enhancer function on genic or intronic accessibility.</p><p>The primary permutation analysis uses observed values. Fitted curves already incorporate chronological information and are used only for descriptive sensitivity; no invalid post-fit stage permutation fractions are reported.</p>
 <h2>Next biological checks</h2><p>Prioritize enteric-neuron and hematopoietic/vascular regions for inspection of individual observed curves and linked loci. Compare the region-level opening schedules, assess celltype composition and technical covariates, and use independent cells/replicates where available. Keep weak or weighting-sensitive regions as exploratory candidates.</p>
 <h2>Files</h2><ul><li><a href="activation_peak_explorer.html">Offline 2D/3D explorer</a>: 12,000 displayed peaks, 32 celltypes, six stages, observed/fitted states and curves, supported rates, reference and graph-distance overlays.</li><li>celltypes/: 32 individual PNG/PDF sheets, including missing-stage masks.</li><li>activation_metrics.h5: full-peak fitted timing masks and ages, every observed/fitted region/CT distance center.</li><li>geometry.h5 and geometry_core_reference.h5: original coordinates and aligned graph-distance arrays.</li><li>CSV/JSON tables: region/time profiles, interval changes, all ordering controls, timing summaries, sensitivity checks and provenance.</li><li>illustrative_peak_profiles.csv: 50 large fitted-opening examples from each of four displayed CTs, with measured/fitted values and annotations; selected examples, not prevalence estimates.</li></ul>
 <h2>Validation</h2><p>Independent audits check all peak identities, original coordinates, graph edges and distance equations; every measured/fitted distance center and timing mask; all 720 temporal permutations; sampled weighting comparisons; and exact viewer-array identity. Browser controls, missingness, 3D rotation and gene search are tested. These checks establish numerical and display consistency, not biological truth.</p><h2>Figures</h2>'''
 regional=out/'regional_activation_order.json'
 if regional.exists():
  regional_summary=json.loads(regional.read_text())
  section='<h2>Direct regional opening-order control</h2><p>This stricter test uses observed mean CPM curves of the 36 fixed atlas regions. Region size and relative-amplitude criteria are invariant to stage ordering. Opening/closing-window centers and their correlation with region graph distance are recomputed under all 720 shared orders.</p><p><b>No celltype-phase test has an outward ordering-tail fraction below 0.05.</b> The strongest correlation is 0.610 (endocrine-pancreas opening; tail 0.076), and the family-wide maximum has tail 0.394 across 42 celltype-phase tests. The threshold-zero sensitivity agrees. Enteric opening has correlation 0.173 (tail 0.243); hematopoietic/vascular opening −0.023 (tail 0.446). Thus, even our strongest redistribution candidates do not establish a coordinated outward opening sequence at this regional scale.</p><p>This does not rule out narrower subprograms, inward schedules, or local waves: region averaging can combine opposing peak trajectories. See regional_activation_order.csv and regional_activation_order.json. No fitted chronological curves enter this control.</p>'
  body=body.replace('<h2>Timing means a window, not an onset</h2>',section+'<h2>Timing means a window, not an onset</h2>')
 for png in sorted(out.glob('*.png')):
  body+=f'<section><h3>{html.escape(png.stem)}</h3><a href="{png.name}"><img src="{png.name}" alt="{html.escape(png.stem)}"></a></section>'
 body+='<h2>All celltype sheets</h2><ul>'+''.join(f'<li><a href="celltypes/{p.name}">{html.escape(p.stem)}</a></li>' for p in sorted((out/'celltypes').glob('*.png')))+'</ul>'
 style='body{max-width:1120px;margin:32px auto;padding:0 24px;font:16px/1.55 system-ui;color:#203246}h1,h2{line-height:1.2}h2{margin-top:36px}.lead{font-size:20px;font-weight:600}.formula{background:#edf4f6;padding:15px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:6px 9px;border-bottom:1px solid #ccd3da;text-align:right}a{color:#09678d}li{margin-bottom:8px}'
 (out/'analysis_summary.html').write_text('<!doctype html><meta charset="utf-8"><title>Activation progression across the peak atlas</title><style>'+style+'</style>'+body)
 summary=dict(main_finding=finding,global_observed_hops=global_row.to_dict(),n_peaks=640830,n_celltypes=32,primary_celltypes=21,global_positive_fraction_reference=ts['global_results']['observed_hops']['fraction_positive_p_outward'],weight_sensitivity=weights,regional_opening_order=json.loads(regional.read_text()),interpretation='Celltype-specific redistribution is promising; a coordinated outward activation wave is not established',source_scripts_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(HERE.glob('*activation*.py'))+list((ROOT/'scripts/chrom_velo').glob('activation*.py'))})
 (out/'analysis_conclusions.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
 email=f'''<p>Hello,</p><p>I completed the celltype-by-stage and graph-distance progression analysis, keeping the existing peak UMAP fixed.</p><p><b>{html.escape(finding)}</b></p><ul><li><b>Global:</b> the measured accessibility center shifts outward by 0.152 graph hops between 10 and 24 hpf, but the six-stage curve is not monotone. 48/720 possible stage orderings are at least as outward (tail 0.067); the concentrated-core sensitivity gives 0.104. Although 20/21 CT slopes are positive, these are dependent—the shared-order tail for that count is 0.239.</li><li><b>Most promising:</b> enteric-neuron pseudobulks (distance–time correlation 0.974; opening-window/distance association 0.442) and hematopoietic/vascular pseudobulks (0.952 and 0.166). Their measured outward patterns persist across both graph-distance and reference definitions. These are exploratory candidates, not established mechanisms.</li><li><b>Direct opening-order test:</b> no raw region-level opening/closing schedule exceeds the prespecified outward stage-order reference at tail below 0.05 (family-wide tail 0.394). Enteric opening has correlation 0.173 (tail 0.243); hematopoietic/vascular opening −0.023 (tail 0.446). This limits the stronger claim of a root-to-branch opening wave.</li><li><b>Other counterevidence:</b> many celltypes have weak/mixed timing patterns; spinal cord has negative opening-window association. Smoothing changes weaker progression directions. We should not claim one universal outward stream.</li><li><b>Depth question:</b> the primary center is Σaccessibility×distance /Σaccessibility. A common per-pseudobulk normalization factor cancels, so this measured statistic is identical from counts or CPM on the same peaks. Technical depth-dependent biases can still matter. Region CPM levels remain relative.</li></ul><p>The main PNGs and review summary are attached. All 32 celltype sheets, the offline 2D/3D viewer, full aligned arrays, tables, code, and validation records are saved at:<br><code>{html.escape(str(out))}</code></p><p>I will send the offline viewer in this thread as a separate ZIP. My interpretation is that celltype-specific redistribution is a promising lead. A shift in the accessibility center can reflect opening farther from the atlas reference, closing nearer to it, or both; it does not by itself identify a propagating wave. The ordering controls are conditional comparisons, not biological-replicate p-values. The accessible core can stay open while other programs activate; literal streamlines originating at the core are not required.</p><p>Best,<br>Codex</p>'''
 (out/'email_package').mkdir(exist_ok=True);(out/'email_package/email_body.html').write_text(email)
 print(finding)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['figures','report','all'],default='all');args=p.parse_args()
 if args.stage in ['figures','all']:figures(OUT)
 if args.stage in ['report','all']:report(OUT)
if __name__=='__main__':main()
