"""Expand figure 09 with matched-window opening, closing and measured accessibility.

Reuses existing fits and fixed graph. No refitting, imputation or graph changes.
Optic cup lacks an eligible 10 hpf pseudobulk; all comparisons use 12–24 hpf.
Activity ranking is supported peak prevalence, not total absolute accessibility.
"""
from pathlib import Path
import json,hashlib,html
import numpy as np,pandas as pd,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from activation_progression import OUT as BASE,SOURCE,PARAMS,timing_metrics,TIMING,signature
OUT=BASE/'graph_hop_profiles_v1'
MIN_BIN=30
COLORS={'opening':'#167d8d','closing':'#b45b39'}


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)


def format_ax(ax,ylabel):
    ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Graph hops from nearest broad reference');ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0,13,2));ax.set_xlim(-.3,12.3);ax.grid(axis='y',alpha=.15)


def timing_panel(ax,name,phase,data,distance,stats,small=False):
    age=data[name][phase+'_center_hpf'];good=np.isfinite(age)&np.isfinite(distance)
    ids=np.flatnonzero(good);rng=np.random.default_rng(1928);ids=rng.choice(ids,min(len(ids),4000),replace=False)
    color=COLORS[phase];ax.scatter(distance[ids]+rng.uniform(-.12,.12,len(ids)),age[ids],s=3,alpha=.12,c=color,rasterized=True)
    bins=[];med=[];lo=[];hi=[]
    for d in np.unique(distance[good]):
        y=age[good&(distance==d)]
        if len(y)>=MIN_BIN:bins.append(d);med.append(np.median(y));lo.append(np.quantile(y,.25));hi.append(np.quantile(y,.75))
    ax.fill_between(bins,lo,hi,color=color,alpha=.18);ax.plot(bins,med,'o-',color=color,ms=4,lw=1.8)
    row=stats[(stats.celltype==name)&(stats.phase==phase)].iloc[0]
    ax.set_title(f'{name.replace("_"," ")} · {phase}\nn={int(row.n_resolved):,}; descriptive ρ={row.spearman:.3f}',fontsize=10 if small else 12)
    format_ax(ax,f'Center of fitted {phase} activity (hpf)');ax.set_ylim(12,24);ax.set_yticks([12,15,18,21,24])


def accessibility_panel(ax,name,profiles,change=False,small=False):
    sub=profiles[profiles.celltype==name];stages=np.sort(sub.stage.unique());colors=plt.cm.viridis(np.linspace(.05,.88,len(stages)))
    for t,color in zip(stages,colors):
        z=sub[sub.stage==t].sort_values('distance_hops');y=z.log2_ratio_to_12hpf if change else z.observed_mean_cpm
        good=(z.peaks>=MIN_BIN)&np.isfinite(y)
        ax.plot(z.distance_hops[good],y[good],'o-',ms=3,lw=1.6,c=color,label=f'{int(t)} hpf')
    if change:
        ax.axhline(0,lw=.7,c='.4')
        low=sub[(sub.stage==12)&(sub.baseline_mean_cpm<.05)&(sub.peaks>=MIN_BIN)]
        if len(low):
            ax.axvspan(low.distance_hops.min()-.45,12.3,color='.75',alpha=.2,zorder=0)
            ax.text(.98,.97,'Gray: low baseline',transform=ax.transAxes,ha='right',va='top',fontsize=7,color='.4')
    else:ax.set_yscale('log')
    format_ax(ax,'log₂(mean CPM / mean CPM at 12 hpf)' if change else 'Observed mean accessibility (CPM; log scale)')
    ax.set_title(name.replace('_',' ')+(' · change from 12 hpf' if change else ' · observed accessibility'),fontsize=10 if small else 12)


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'celltypes').mkdir(exist_ok=True)
    before=signature(SOURCE);ranks=[];all_quality={}
    with h5py.File(SOURCE,'r') as f,h5py.File(BASE/'geometry.h5','r') as g:
        ids=f['peak_ids'].asstr()[:];assert np.array_equal(ids,g['peak_ids'].asstr()[:]);distance=g['distance_hops'][:]
        ct=f['celltypes'].asstr()[:];times=f['stages'][1:];dt=np.diff(times);cv=f['condition_valid'][:]
        assert np.array_equal(times,[12,14,16,19,24]);eligible=cv[:,1:].all(axis=1)
        for c in np.flatnonzero(eligible):
            name=ct[c];rr=f['rates'][:,c,1:].astype(float);ss=f['rate_supported'][:,c,1:]
            quality=f['fit_valid'][:,c]&(f['lack_of_fit'][:,c]<=PARAMS['lack_of_fit_threshold'])&np.isfinite(rr).all(axis=1)
            n=int(quality.sum());row=dict(celltype=name,n_eligible_peaks=n,full_six_stage_panel=bool(cv[c].all()))
            for phase,sign in [('opening',1),('closing',-1)]:
                active=quality[:,None]&ss&(sign*rr>0)
                row[phase+'_fraction']=float(np.sum(active.sum(axis=0)*dt)/(n*dt.sum()))
                row[phase+'_unique_peaks']=int(active.any(axis=1).sum())
                row[phase+'_supported_change_cpm']=float(np.sum(np.where(active,sign*rr,0)*dt))
            ranks.append(row);print('Ranked',name,flush=True)
        ranking=pd.DataFrame(ranks)
        for phase in ['opening','closing']:ranking[phase+'_rank']=ranking[phase+'_fraction'].rank(method='min',ascending=False).astype(int)
        top_open=ranking.nlargest(3,'opening_fraction').celltype.tolist();top_close=ranking.nlargest(3,'closing_fraction').celltype.tolist()
        selected=list(dict.fromkeys(['optic_cup']+top_open+top_close+['enteric_neurons']))
        ranking['selected']=ranking.celltype.isin(selected)
        ranking['selection_reason']=ranking.celltype.map(lambda c:'Requested optic cup' if c=='optic_cup' else '; '.join(([ 'Top 3 supported opening fraction'] if c in top_open else [])+(['Top 3 supported closing fraction'] if c in top_close else [])+(['Previous outward-progression comparison'] if c=='enteric_neurons' else [])))
        ranking.sort_values('opening_rank').to_csv(OUT/'celltype_activity_ranking.csv',index=False)
        data={};stats=[];binrows=[]
        with h5py.File(OUT/'matched_window_timing.h5','w') as out:
            out.create_dataset('peak_ids',data=ids.astype(object),dtype=h5py.string_dtype(),compression='gzip')
            out.create_dataset('distance_hops',data=distance,compression='gzip');out.create_dataset('stages',data=times)
            out.create_dataset('celltypes',data=np.asarray(selected,object),dtype=h5py.string_dtype())
            out.attrs['definition']='Existing fits; interval timing restricted to 12–24 hpf for every celltype. Support plus >=0.5 CPM phase amount and >=0.25 relative rate modulation. No onset or replicate uncertainty.'
            for name in selected:
                c=np.flatnonzero(ct==name)[0];rr=f['rates'][:,c,1:].astype(float);ss=f['rate_supported'][:,c,1:]
                quality=f['fit_valid'][:,c]&(f['lack_of_fit'][:,c]<=PARAMS['lack_of_fit_threshold'])
                tm=timing_metrics(rr,ss,times,quality);data[name]=tm;group=out.create_group(name)
                for phase in ['opening','closing']:
                    age=tm[phase+'_center_hpf'];group.create_dataset(phase+'_center_hpf',data=age.astype('f4'),compression='gzip')
                    good=np.isfinite(age)&np.isfinite(distance)
                    rho=float(spearmanr(distance[good],age[good]).statistic) if good.sum()>=30 and np.ptp(age[good])>0 else np.nan
                    stats.append(dict(celltype=name,phase=phase,n_resolved=int(good.sum()),spearman=rho,median_window_center_hpf=float(np.nanmedian(age)),start_hpf=12,end_hpf=24))
                    for d in np.unique(distance):
                        y=age[good&(distance==d)]
                        binrows.append(dict(celltype=name,phase=phase,distance_hops=int(d),n_resolved=len(y),displayed_summary=len(y)>=MIN_BIN,median=np.median(y) if len(y) else np.nan,q25=np.quantile(y,.25) if len(y) else np.nan,q75=np.quantile(y,.75) if len(y) else np.nan))
                print('Computed windows',name,flush=True)
    stats=pd.DataFrame(stats);stats.to_csv(OUT/'timing_summary.csv',index=False);pd.DataFrame(binrows).to_csv(OUT/'timing_by_hop.csv',index=False)
    profiles=pd.read_csv(BASE/'distance_bin_time_summary.csv');profiles=profiles[profiles.celltype.isin(selected)&profiles.stage.isin(times)].copy()
    baseline=profiles[profiles.stage==12][['celltype','distance_hops','observed_mean_cpm']].rename(columns={'observed_mean_cpm':'baseline_mean_cpm'})
    profiles=profiles.merge(baseline,on=['celltype','distance_hops'],validate='many_to_one')
    profiles['log2_ratio_to_12hpf']=np.where((profiles.observed_mean_cpm>0)&(profiles.baseline_mean_cpm>0),np.log2(profiles.observed_mean_cpm/profiles.baseline_mean_cpm),np.nan)
    profiles.to_csv(OUT/'observed_accessibility_by_hop.csv',index=False)
    footer='Matched 12–24 hpf window. Median lines and peak interquartile ranges use bins with ≥30 resolved peaks; shading is not uncertainty.\nSupported, nonuniform fitted trajectories only. Timing is a center of activity, not onset. Correlations are descriptive; no independent-peak p-values.'
    rows=int(np.ceil(len(selected)/4))
    for phase in ['opening','closing']:
        fig,axes=plt.subplots(rows,4,figsize=(20,rows*4.2),sharex=True,sharey=True)
        for ax,name in zip(axes.flat,selected):timing_panel(ax,name,phase,data,distance,stats,small=True)
        for ax in list(axes.flat)[len(selected):]:ax.axis('off')
        fig.suptitle(f'{phase.capitalize()} windows across graph distance · optic cup and dynamic celltypes',fontsize=18)
        fig.text(.5,.015,footer,ha='center',fontsize=10);fig.subplots_adjust(top=.87,bottom=.15,hspace=.45,wspace=.28)
        save(fig,'09b_opening_windows_selected' if phase=='opening' else '09c_closing_windows_selected')
    fig,axes=plt.subplots(rows,4,figsize=(20,rows*4.2),sharex=True)
    for ax,name in zip(axes.flat,selected):accessibility_panel(ax,name,profiles,change=True,small=True)
    for ax in list(axes.flat)[len(selected):]:ax.axis('off')
    axes.flat[0].legend(fontsize=8,ncol=2);fig.suptitle('Observed accessibility changes at each graph distance',fontsize=18)
    fig.text(.5,.015,'Every line uses the same peaks within each hop bin. Values are relative CPM; no pseudocount, no peak selection by dynamic mode.\nGray marks bins with baseline mean <0.05 CPM: large ratios can reflect small denominators. These relative changes do not imply peak movement.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.87,bottom=.15,hspace=.45,wspace=.33);save(fig,'09d_observed_accessibility_changes')
    for name in selected:
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        timing_panel(axes[0,0],name,'opening',data,distance,stats);timing_panel(axes[0,1],name,'closing',data,distance,stats)
        accessibility_panel(axes[1,0],name,profiles);accessibility_panel(axes[1,1],name,profiles,change=True);axes[1,0].legend(ncol=3,fontsize=9)
        rr=ranking.set_index('celltype').loc[name]
        fig.suptitle(f'{name.replace("_"," ")} · timing and accessibility over graph distance\nSupported activity: opening {rr.opening_fraction:.2%}; closing {rr.closing_fraction:.2%} of eligible peaks, averaged over 12–24 hpf',fontsize=15)
        note='Optic cup 10 hpf is below the cell-count threshold and excluded. ' if name=='optic_cup' else ''
        fig.text(.5,.012,note+'All panels use 12–24 hpf; the original fits are reused.\nTop: supported, nonuniform fitted trajectories; median ± interquartile range, not confidence. Bottom: observed mean CPM across all peaks in each bin.\nGray in ratio panel: baseline mean <0.05 CPM; ratios can be unstable. Accessibility is relative; a distance gradient alone does not establish flow.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.86,bottom=.14,hspace=.42,wspace=.3)
        save(fig,'09a_optic_cup_detail' if name=='optic_cup' else 'celltypes/'+name)
    # Direct extensions of original09 for CTs with all six eligible stages.
    full_selected=[name for name in selected if name!='optic_cup'];full_data={}
    full_stats=pd.read_csv(BASE/'opening_closing_timing.csv').query('region_code == -1').rename(columns={'distance_timing_spearman':'spearman'})
    with h5py.File(BASE/'activation_metrics.h5') as ff:
        full_ct=ff['celltypes'].asstr()[:]
        for name in full_selected:
            c=np.flatnonzero(full_ct==name)[0]
            assert ff['common_celltypes'][c]
            full_data[name]={phase+'_center_hpf':ff[phase+'_center_hpf'][:,c] for phase in ['opening','closing']}
    full_stats[full_stats.celltype.isin(full_selected)].to_csv(OUT/'timing_summary_full_10_24hpf.csv',index=False)
    for phase in ['opening','closing']:
        fig,axes=plt.subplots(2,4,figsize=(20,8.4),sharex=True,sharey=True)
        for ax,name in zip(axes.flat,full_selected):
            timing_panel(ax,name,phase,full_data,distance,full_stats,small=True);ax.set_ylim(10,24);ax.set_yticks([10,14,18,22])
        axes.flat[-1].axis('off');axes.flat[-1].text(.05,.7,'Optic cup is shown separately\nover 12–24 hpf.\n\nIts 10 hpf condition does not meet\nthe cell-count threshold.',transform=axes.flat[-1].transAxes,fontsize=12,va='top')
        fig.suptitle(f'Original figure09 extension · {phase} over the full 10–24 hpf window',fontsize=17)
        fig.text(.5,.015,'Same stored timing values and definitions as original figure09; these centers differ from the matched 12–24 hpf panels.\nSupported, nonuniform fitted trajectories only; median and peak interquartile range, not confidence. Correlations are descriptive.',ha='center',fontsize=10)
        fig.subplots_adjust(top=.87,bottom=.15,hspace=.45,wspace=.28)
        save(fig,'09f_full_window_opening' if phase=='opening' else '09g_full_window_closing')
    ordered=ranking.sort_values('closing_fraction',ascending=True);y=np.arange(len(ordered))
    fig,ax=plt.subplots(figsize=(10,9));ax.barh(y-.17,100*ordered.opening_fraction,height=.32,color=COLORS['opening'],label='Opening');ax.barh(y+.17,100*ordered.closing_fraction,height=.32,color=COLORS['closing'],label='Closing')
    ax.set_yticks(y,ordered.celltype.str.replace('_',' '));ax.set_xlabel('Duration-weighted supported peak fraction (%)');ax.legend();ax.spines[['top','right']].set_visible(False)
    for label,name in zip(ax.get_yticklabels(),ordered.celltype):label.set_fontweight('bold' if name in selected else 'normal')
    ax.set_title('Opening and closing activity across eligible celltypes\n12–24 hpf; bold labels included in the expanded panels')
    fig.text(.5,.012,'Ranked by supported peak prevalence among quality-eligible fits, with actual interval durations. A transient peak can contribute to both phases.\nDifferences reflect biological activity plus detection power and fit eligibility; this is not an absolute accessibility ranking.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.3,bottom=.12,top=.92);save(fig,'09e_activity_ranking')
    assert before==signature(SOURCE)
    manifest=dict(selected_celltypes=selected,top_opening=top_open,top_closing=top_close,window_hpf=times.tolist(),timing_parameters=TIMING,minimum_summary_bin=MIN_BIN,source_fits=before,source_unchanged=True,selection='Optic cup + top3 duration-weighted supported opening fractions + top3 closing fractions + enteric neurons as previous comparison; all eligible at every stage 12–24 hpf.',fit_reuse='Existing fits use all eligible stages for each celltype; timing amounts here use only adjacent intervals 12–24. No refitting.',interpretation='Timing and observed CPM are separate quantities. Local uncertainty is conditional count-noise, not biological replication.',figures=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(OUT.rglob('*.png'))])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    table=ranking[ranking.selected][['celltype','selection_reason','n_eligible_peaks','opening_fraction','closing_fraction']].copy();table[['opening_fraction','closing_fraction']]*=100
    table.columns=['Celltype','Reason included','Eligible peaks','Opening fraction (%)','Closing fraction (%)']
    body='<h1>Expanded graph-distance profiles</h1><p>Optic cup and celltypes with strong supported opening/closing activity, compared over <b>12–24 hpf</b>. Optic cup at 10 hpf is below the cell-count threshold. Existing fits and graph distances are reused.</p><p><b>Top timing panels:</b> y is the center of fitted opening/closing activity in hours. <b>Accessibility panels:</b> y is observed mean CPM or its log₂ ratio to 12 hpf. These answer different questions. An upward timing line means later activity farther from the atlas reference; it does not mean greater accessibility.</p>'+table.to_html(index=False,float_format=lambda x:f'{x:.3f}')
    body+='<p><b>Optic cup:</b> supported activity is 1.69% opening and 2.66% closing on average over the matched window. Its timing–distance correlations are weak (opening −0.087; closing −0.074). Opening and closing panels describe different, potentially overlapping peak sets; their medians do not establish that the same peaks open and then close.</p><p><b>Window sensitivity:</b> enteric opening correlation is 0.131 over 12–24 hpf versus 0.442 in the original 10–24 hpf analysis. The appended full-window panels reuse the original stored values for direct comparison. Limited hinge models can produce horizontal timing bands; these are not finely resolved event times.</p><p>Gray ratio-panel regions have a 12 hpf mean below 0.05 CPM. The data are retained, but large ratios there can be unstable. The falling overall accessibility gradient partly reflects the broad-reference definition; compare stage differences rather than interpreting that gradient alone as a flow.</p>'
    body+='<p>Selection uses the fraction of quality-eligible peaks with supported change, averaged with actual interval durations. It does not select celltypes for an outward correlation. Activity depends on sampling and fit eligibility; transient peaks can contribute to both phases. All displayed timing associations are descriptive. Incomplete stages stay excluded, and constant-rate trajectories do not receive artificial event ages.</p>'
    for p in sorted(OUT.glob('*.png'))+sorted((OUT/'celltypes').glob('*.png')):
        rel=p.relative_to(OUT);body+=f'<h2>{html.escape(p.stem)}</h2><a href="{rel}"><img src="{rel}"></a>'
    body+='<h2>Tables and arrays</h2><ul>'+''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(OUT.glob('*.csv')))+ '<li>matched_window_timing.h5 stores all peak-aligned window centers for these celltypes.</li><li>manifest.json records selection, sources, parameters, and figure checksums.</li></ul>'
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Expanded graph-distance profiles</title><style>body{max-width:1250px;margin:30px auto;padding:0 24px;font:16px/1.55 system-ui;color:#243448}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}table{border-collapse:collapse;font-size:14px}h2{margin-top:40px}</style>'+body)
    print(json.dumps({'selected':selected,'output':str(OUT),'timing':stats.to_dict('records')},indent=2),flush=True)

if __name__=='__main__':main()
