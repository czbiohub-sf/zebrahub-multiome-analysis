#!/usr/bin/env python3
"""Prespecified past-window peak selection and disjoint-window sampling agreement.

This is a selection/reproducibility pilot, not a forecast model. Half A at
10/12/14 hpf defines eligibility and selection; evaluation uses raw independent
cell-half changes at 16->19 and19->24 hpf. No future observation chooses peaks.
"""
from pathlib import Path
import hashlib,json,time
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/cis_variable_peak_design_v1/filter_pilot'
SOURCE=HERE/'outputs/ct_flow_noise_v1/cell_split_input.h5'
SEED=20261024
NCONTROLS=30
FRACTION=.20
METHODS=['accessibility_gini','adjusted_total_variance','adjusted_temporal_variance']
LABELS={'accessibility_gini':'Accessibility Gini','adjusted_total_variance':'Mean-adjusted total variance','adjusted_temporal_variance':'Mean-adjusted temporal variance'}
COLORS={'accessibility_gini':'#9b4e9e','adjusted_total_variance':'#d5802b','adjusted_temporal_variance':'#217c86'}


def dump(name,obj):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    (OUT/name).write_text(json.dumps(obj,indent=2,default=convert,allow_nan=False)+'\n')


def signature(path):
    st=path.stat();return dict(path=str(path.resolve()),bytes=st.st_size,mtime_ns=st.st_mtime_ns)


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def gini(x):
    assert (x>=0).all() and np.isfinite(x).all()
    y=np.sort(x,axis=1);n=y.shape[1];s=y.sum(axis=1)
    return np.divide((y*(2*np.arange(1,n+1)-n-1)).sum(axis=1),n*s,out=np.zeros(len(y)),where=s>0)


def bins_from_training(x,train,n):
    edges=np.unique(np.quantile(x[train],np.arange(1,n)/n))
    return np.searchsorted(edges,x,side='right'),edges


def adjusted_percentile(v,mean_bins,train):
    scores=np.full(len(v),np.nan);counts=[]
    for b in np.unique(mean_bins):
        reference=np.sort(v[train&(mean_bins==b)])
        assert len(reference)>0
        use=mean_bins==b
        lo=np.searchsorted(reference,v[use],side='left')
        hi=np.searchsorted(reference,v[use],side='right')
        scores[use]=(lo+hi)/(2*len(reference))
        counts.append(dict(mean_bin=int(b),n_training_reference=len(reference)))
    return scores,counts


def finite_mean(x):
    x=np.asarray(x);good=np.isfinite(x)
    return float(x[good].mean()) if good.any() else np.nan


def per_peak(a,b):
    na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1)
    valid=(na>1e-12)&(nb>1e-12)
    cos=np.divide((a*b).sum(axis=1),na*nb,out=np.full(len(a),np.nan),where=valid)
    both=(np.abs(a)>1e-12)&(np.abs(b)>1e-12)
    agree=(np.sign(a)==np.sign(b))&both
    pairs=both.sum(axis=1);agreements=agree.sum(axis=1)
    signs=np.divide(agreements,pairs,out=np.full(len(a),np.nan),where=pairs>0)
    return dict(cosine=np.clip(cos,-1,1),both_nonzero_vectors=valid,
                n_both_nonzero_ct=pairs,n_same_sign_ct=agreements,
                sign_agreement=signs,rate_energy_a=(a*a).sum(axis=1),rate_energy_b=(b*b).sum(axis=1))


def metrics(d,idx):
    c=d['cosine'][idx];nz=np.isfinite(c);pairs=d['n_both_nonzero_ct'][idx].sum()
    return dict(n_peaks=len(idx),n_nonzero_vector_pairs=int(nz.sum()),
                nonzero_vector_fraction=float(nz.mean()),
                mean_cosine=finite_mean(c),median_cosine=float(np.nanmedian(c)) if nz.any() else np.nan,
                positive_cosine_fraction=float(np.mean(c[nz]>0)) if nz.any() else np.nan,
                mean_per_peak_sign_agreement=finite_mean(d['sign_agreement'][idx]),
                pooled_ct_sign_agreement=float(d['n_same_sign_ct'][idx].sum()/pairs) if pairs else np.nan,
                nonzero_ct_pair_fraction=float(pairs/(len(idx)*21)),
                mean_rate_energy_a=float(d['rate_energy_a'][idx].mean()),
                mean_rate_energy_b=float(d['rate_energy_b'][idx].mean()))


def matched_controls(selected,test_eligible,strata,rng):
    target=np.flatnonzero(selected&test_eligible)
    pool=np.flatnonzero(~selected&test_eligible)
    draws=[[] for _ in range(NCONTROLS)];covered=[];rows=[]
    for s in np.unique(strata[target]):
        ix=target[strata[target]==s];cand=pool[strata[pool]==s]
        use=len(cand)>0;replace=len(cand)<len(ix)
        rows.append(dict(stratum=int(s),n_selected=len(ix),n_available_controls=len(cand),
                         matched=use,replacement=bool(replace and use)))
        if not use:continue
        covered.extend(ix.tolist())
        for r in range(NCONTROLS):draws[r].extend(rng.choice(cand,len(ix),replace=replace).tolist())
    covered=np.asarray(sorted(covered),int)
    draws=np.asarray(draws,int)
    assert len(covered)>0 and draws.shape==(NCONTROLS,len(covered))
    assert not np.intersect1d(draws.ravel(),target).size
    assert np.all(test_eligible[draws])
    for draw in draws:
        assert np.array_equal(np.bincount(strata[draw],minlength=50),np.bincount(strata[covered],minlength=50))
    return covered,draws,rows



def matched_controls_unique(selected,test_eligible,strata,rng):
    """Outcome-blind overlap sensitivity; each control appears at most once/draw."""
    target=np.flatnonzero(selected&test_eligible);pool=np.flatnonzero(~selected&test_eligible)
    chosen=[];draws=[[] for _ in range(NCONTROLS)]
    for label in np.unique(strata[target]):
        ix=target[strata[target]==label];cand=pool[strata[pool]==label];n=min(len(ix),len(cand))
        if not n:continue
        chosen.extend(rng.choice(ix,n,replace=False).tolist())
        for r in range(NCONTROLS):draws[r].extend(rng.choice(cand,n,replace=False).tolist())
    chosen=np.asarray(sorted(chosen),int);draws=np.asarray(draws,int)
    assert not np.intersect1d(draws.ravel(),target).size
    for draw in draws:
        assert len(np.unique(draw))==len(draw)
        assert np.array_equal(np.bincount(strata[draw],minlength=50),np.bincount(strata[chosen],minlength=50))
    return chosen,draws


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(seed=SEED,selection_half='A',selection_stages_hpf=[10,12,14],evaluation_intervals_hpf=[[16,19],[19,24]],
      panel='21 celltypes having >=20 pooled cells at all six stages; existing fixed panel',
      train_peaks=20000,test_peaks=5000,excluded_marker_extras=56,
      eligibility='At least20 raw counts summed over half-A,21 CTs and10/12/14hpf; finite values',
      selection_fraction=FRACTION,methods=METHODS,
      accessibility_gini='Gini over63 linear-CPM CT-stage values, finite-sample maximum62/63; zero-total defined0',
      total_variance='Population variance of log1p(CPM) across63 past CT-stage values',
      temporal_variance='Mean over21 CTs of population variance of log1p(CPM) over three past stages',
      mean_adjustment='Training-eligible half-A mean-CPM deciles; variance mapped to midrank empirical percentile within its training mean bin; global80th percentile of training adjusted scores defines top20%',
      gini_cutoff='80th percentile of eligible training Gini; strict score>cutoff for all methods; realized fractions reported',
      matching='Training-eligible mean-CPM deciles x detection quintiles, both from past half-A only; controls drawn from unselected eligible test peaks; exact bin counts; replacement only if fewer candidates than target; no candidate bin drops matching targets only, with coverage reported',
      controls=NCONTROLS,primary_endpoint='Mean cosine between raw21-CT A/B rate vectors in each evaluation interval, among both-nonzero vectors; nonzero coverage separately',
      secondary_endpoints=['Median cosine','fraction positive cosine','per-peak and pooled CT sign agreement on both-nonzero CT rates','nonzero CT and vector coverage','raw rate energy'],
      rate_units='CPM per hpf, divided by actual3h and5h intervals',
      no_smoothing=True,no_umap_refit=True,no_prediction_model=True,
      selection_validation_stage_overlap=False,
      uncertainty='Matched-control bands are5th–95th percentiles over30 random control samples, not confidence intervals or biological uncertainty',
      limitations=['Cell halves assess sampling reproducibility, not biological replication','A/B use complementary cells within the same samples, with stage/batch confounding retained','CPM is relative accessibility','Three-stage selection can miss later-emerging or transient programs','Matching controls measured past abundance/detection but does not remove all biological or technical differences','Future vector agreement is reproducibility of measured changes, not future prediction','Intervals share the19hpf evaluation stage and are not independent','Training/test split is random whole peaks, not independent chromosomes or genomic neighborhoods','The fixed21-celltype panel was defined using full-course cell-count coverage metadata; this is not a fully prospective study','Matching bins do not guarantee exact continuous covariate balance; balance diagnostics are reported','Finite-sample matching can need repeated control peaks; duplicates explicitly reported'])
    dump('design_prespecified.json',design)
    before=signature(SOURCE)
    with h5py.File(SOURCE,'r') as f:
        stage=f['stages'][:];ct=f['condition_valid'][:].all(axis=1);assert ct.sum()==21
        train=f['train_mask'][:].astype(bool);test=f['test_mask'][:].astype(bool)
        assert train.sum()==20000 and test.sum()==5000 and not (train&test).any()
        source=f['source_indices'][:];ids=f['peak_ids'].asstr()[:]
        names=f['celltypes'].asstr()[:][ct]
        # Read only the prespecified past window before defining any selection.
        past=f['observed_cpm_a'][:,:3,:][:,:,ct].astype(float)
        counts=f['raw_counts_a'][:,:3,:][:,:,ct]
        assert np.array_equal(stage[:3],[10,12,14]) and np.isfinite(past).all()
        flat=past.reshape(len(past),-1);log=np.log1p(past)
        mean=flat.mean(axis=1);detect=(counts>0).mean(axis=(1,2));totalcount=counts.sum(axis=(1,2))
        eligible=(totalcount>=20)&np.isfinite(flat).all(axis=1);tr=train&eligible;te=test&eligible
        mb,medges=bins_from_training(mean,tr,10);db,dedges=bins_from_training(detect,tr,5);strata=mb*5+db
        totalvar=np.var(log,axis=(1,2));temporalvar=np.var(log,axis=1).mean(axis=1)
        staticvar=np.var(log.mean(axis=1),axis=1)
        assert np.allclose(totalvar,temporalvar+staticvar,atol=1e-12)
        scores={'accessibility_gini':gini(flat)}
        for key,v in [('adjusted_total_variance',totalvar),('adjusted_temporal_variance',temporalvar)]:
            scores[key],_=adjusted_percentile(v,mb,tr)
        scores_df=pd.DataFrame(dict(source_index=source,peak_id=ids,split=np.where(train,'train',np.where(test,'test','excluded_marker')),eligible=eligible,
            past_mean_cpm=mean,past_detection_fraction=detect,past_total_count=totalcount,
            mean_bin=mb,detection_bin=db,matching_stratum=strata,
            total_log_variance=totalvar,temporal_log_variance=temporalvar,static_ct_log_variance=staticvar,
            temporal_fraction=np.divide(temporalvar,totalvar,out=np.zeros(len(mean)),where=totalvar>0),**scores))
        selected={};thresholdrows=[];matchingrows=[];matches={};draws={};unique_matches={};unique_draws={}
        for m,key in enumerate(METHODS):
            cut=float(np.quantile(scores[key][tr],1-FRACTION));selected[key]=eligible&(scores[key]>cut)
            scores_df[key+'_selected']=selected[key]
            thresholdrows.append(dict(method=key,cutoff=cut,n_training_eligible=int(tr.sum()),n_training_selected=int((selected[key]&train).sum()),
                                      n_test_eligible=int(te.sum()),n_test_selected=int((selected[key]&test).sum())))
            matches[key],draws[key],rows=matched_controls(selected[key],te,strata,np.random.default_rng(SEED+100*m))
            for row in rows:matchingrows.append(dict(method=key,**row))
            unique_matches[key],unique_draws[key]=matched_controls_unique(selected[key],te,strata,np.random.default_rng(SEED+10000+100*m))
        dump('common_support_sensitivity_design.json',dict(reason='Added after inspecting poor covariate overlap in the prespecified Gini comparison, before computing sensitivity outcomes; not part of original prespecified comparison',rule='Within each matching bin, randomly choose min(n_selected,n_available_unselected_controls) selected peaks once, using fixed seed. Draw equal-size controls without replacement. No outcome enters sampling.',selection_seed_base=SEED+10000))
        np.savez_compressed(OUT/'no_replacement_control_indices.npz',**{key:unique_draws[key] for key in METHODS},**{key+'_selected':unique_matches[key] for key in METHODS},source_indices=source)
        scores_df.to_csv(OUT/'selection_scores.csv.gz',index=False)
        pd.DataFrame(thresholdrows).to_csv(OUT/'selection_thresholds.csv',index=False)
        pd.DataFrame(matchingrows).to_csv(OUT/'matching_strata.csv',index=False)
        balance_rows=[]
        for mode,selmap,drawmap in [('primary',matches,draws),('no_replacement',unique_matches,unique_draws)]:
            for key in METHODS:
                for r,draw in enumerate(drawmap[key]):
                    for variable,values in [('log1p_past_mean_cpm',np.log1p(mean)),('past_detection_fraction',detect),('log1p_past_total_counts',np.log1p(totalcount))]:
                        xx=values[selmap[key]];yy=values[draw];sd=np.sqrt((np.var(xx)+np.var(yy))/2)
                        balance_rows.append(dict(comparison=mode,method=key,draw=r,variable=variable,selected_mean=float(xx.mean()),control_mean=float(yy.mean()),standardized_mean_difference=float((xx.mean()-yy.mean())/sd) if sd else 0.))
        pd.DataFrame(balance_rows).to_csv(OUT/'matching_continuous_covariate_balance.csv',index=False)
        np.savez_compressed(OUT/'matched_control_indices.npz',**{key:draws[key] for key in METHODS},**{key+'_selected':matches[key] for key in METHODS},source_indices=source)
        dump('selection_frozen.json',dict(training_eligible=int(tr.sum()),test_eligible=int(te.sum()),mean_bin_edges=medges,detection_bin_edges=dedges,thresholds=thresholdrows,
             selection_score_sha256=sha256(OUT/'selection_scores.csv.gz'),selection_stages=stage[:3],evaluation_not_yet_read=True))
        # Selections are now saved and fixed. Read held-out-stage outcomes.
        future_a=f['observed_cpm_a'][:,3:,:][:,:,ct].astype(float)
        future_b=f['observed_cpm_b'][:,3:,:][:,:,ct].astype(float)
    assert signature(SOURCE)==before
    assert np.array_equal(stage[3:],[16,19,24])
    assert np.isfinite(future_a).all() and np.isfinite(future_b).all()
    rate_a=np.diff(future_a,axis=1)/np.diff(stage[3:])[None,:,None]
    rate_b=np.diff(future_b,axis=1)/np.diff(stage[3:])[None,:,None]
    resultrows=[];controlrows=[];peakrows=[];unique_resultrows=[];unique_controlrows=[]
    for t in range(2):
        interval=f'{int(stage[t+3])}-{int(stage[t+4])}';d=per_peak(rate_a[:,t],rate_b[:,t]);meta=dict(interval=interval,start_hpf=stage[t+3],end_hpf=stage[t+4])
        resultrows.append(dict(**meta,method='all_eligible',cohort='all_eligible',**metrics(d,np.flatnonzero(te))))
        for key in METHODS:
            target=np.flatnonzero(te&selected[key])
            resultrows.append(dict(**meta,method=key,cohort='selected_all',**metrics(d,target)))
            resultrows.append(dict(**meta,method=key,cohort='selected_matchable',**metrics(d,matches[key])))
            for r,ix in enumerate(draws[key]):
                controlrows.append(dict(**meta,method=key,draw=r,n_unique_peaks=len(np.unique(ix)),duplicate_fraction=1-len(np.unique(ix))/len(ix),**metrics(d,ix)))
        for key in METHODS:
            unique_resultrows.append(dict(**meta,method=key,**metrics(d,unique_matches[key])))
            for r,ix in enumerate(unique_draws[key]):unique_controlrows.append(dict(**meta,method=key,draw=r,**metrics(d,ix)))
        for i in np.flatnonzero(test):
            peakrows.append(dict(source_index=int(source[i]),peak_id=ids[i],**meta,eligible=bool(eligible[i]),**{k:float(v[i]) for k,v in d.items()}))
    results=pd.DataFrame(resultrows);controls=pd.DataFrame(controlrows);perpeak=pd.DataFrame(peakrows)
    results.to_csv(OUT/'reproducibility_metrics.csv',index=False);controls.to_csv(OUT/'matched_control_metrics.csv',index=False)
    perpeak.to_csv(OUT/'heldout_peak_outcomes.csv.gz',index=False)
    comparisons=[]
    for row in results[results.cohort=='selected_matchable'].itertuples():
        cc=controls[(controls.method==row.method)&(controls.interval==row.interval)]
        comparison=dict(method=row.method,interval=row.interval,n_selected=row.n_peaks,
                        matching_coverage=len(matches[row.method])/int((te&selected[row.method]).sum()),
                        control_mean_duplicate_fraction=float(cc.duplicate_fraction.mean()),control_max_duplicate_fraction=float(cc.duplicate_fraction.max()))
        for metric in ['mean_cosine','mean_per_peak_sign_agreement','nonzero_vector_fraction','nonzero_ct_pair_fraction','mean_rate_energy_b']:
            value=float(getattr(row,metric));ref=cc[metric].to_numpy()
            comparison.update({metric+'_selected':value,metric+'_control_mean':float(ref.mean()),metric+'_control_p05':float(np.percentile(ref,5)),metric+'_control_p95':float(np.percentile(ref,95)),metric+'_excess':value-float(ref.mean())})
        comparisons.append(comparison)
    comp=pd.DataFrame(comparisons);comp.to_csv(OUT/'selected_vs_matched_controls.csv',index=False)
    unique_results=pd.DataFrame(unique_resultrows);unique_controls=pd.DataFrame(unique_controlrows)
    unique_results.to_csv(OUT/'no_replacement_selected_metrics.csv',index=False);unique_controls.to_csv(OUT/'no_replacement_control_metrics.csv',index=False)
    unique_comparisons=[]
    for row in unique_results.itertuples():
        cc=unique_controls[(unique_controls.method==row.method)&(unique_controls.interval==row.interval)]
        item=dict(method=row.method,interval=row.interval,n_selected=row.n_peaks,matching_coverage=row.n_peaks/int((te&selected[row.method]).sum()),control_mean_duplicate_fraction=0.)
        for metric in ['mean_cosine','mean_per_peak_sign_agreement','nonzero_vector_fraction','nonzero_ct_pair_fraction']:
            ref=cc[metric].to_numpy();item.update({metric+'_selected':float(getattr(row,metric)),metric+'_control_mean':float(ref.mean()),metric+'_control_p05':float(np.percentile(ref,5)),metric+'_control_p95':float(np.percentile(ref,95)),metric+'_excess':float(getattr(row,metric))-float(ref.mean())})
        unique_comparisons.append(item)
    unique_comp=pd.DataFrame(unique_comparisons);unique_comp.to_csv(OUT/'no_replacement_comparisons.csv',index=False)
    # These tests verify data separation and definitions, not a chosen outcome.
    assert gini(np.ones((1,63)))[0]==0
    assert np.isclose(gini(np.eye(1,63))[0],62/63)
    constant=np.broadcast_to(np.arange(21)[None,None,:],(2,3,21)).copy()
    assert np.var(constant,axis=1).mean(axis=1).max()==0
    perfect=per_peak(np.ones((2,21)),-np.ones((2,21)))
    assert np.allclose(perfect['cosine'],-1) and np.all(perfect['sign_agreement']==0)
    overlap=[]
    for a in METHODS:
        for b in METHODS:
            x=selected[a]&te;y=selected[b]&te
            overlap.append(dict(method_a=a,method_b=b,intersection=int((x&y).sum()),union=int((x|y).sum()),jaccard=float((x&y).sum()/(x|y).sum())))
    pd.DataFrame(overlap).to_csv(OUT/'selection_overlap.csv',index=False)
    context=[]
    for name,mask in [('all_eligible',te)]+[(key,selected[key]&te) for key in METHODS]:
        context.append(dict(method=name,n=int(mask.sum()),median_past_mean_cpm=float(np.median(mean[mask])),median_past_detection=float(np.median(detect[mask])),
                       median_temporal_fraction=float(np.median(scores_df.temporal_fraction.to_numpy()[mask])),
                       median_temporal_log_variance=float(np.median(temporalvar[mask])),median_total_log_variance=float(np.median(totalvar[mask]))))
    context=pd.DataFrame(context);context.to_csv(OUT/'selected_cohort_context.csv',index=False)
    make_plots(results,controls,comp,unique_comp,context,scores_df,test,eligible)
    write_methods(design,thresholdrows,comp,unique_comp,context,results)
    dump('results_summary.json',dict(status='complete',selection_window_hpf=[10,12,14],evaluation_windows_hpf=[[16,19],[19,24]],
        n_test=int(test.sum()),n_test_eligible=int(te.sum()),n_training_eligible=int(tr.sum()),celltypes=names,
        selection_thresholds=thresholdrows,comparisons=comparisons,no_replacement_sensitivity=unique_comparisons,cohort_context=context.to_dict('records'),matching_balance=pd.DataFrame(balance_rows).groupby(['comparison','method','variable']).standardized_mean_difference.mean().reset_index().to_dict('records'),
        interpretation='A past-window filtering comparison of later independent-cell raw-rate agreement; no prediction model fitted. Controls matched past mean accessibility and detection, and all future outcomes were excluded from selection.',
        design_fixed_before_future_outcomes=True))
    dump('validation.json',dict(status='passed',selection_and_evaluation_stage_sets_disjoint=True,training_and_test_peaks_disjoint=True,
        marker_extras_excluded=True,source_unchanged=signature(SOURCE)==before,n_celltypes=21,exact_matching_bin_counts=True,
        controls_exclude_selected=True,variance_decomposition_passed=True,gini_known_cases_passed=True,cosine_sign_known_cases_passed=True,
        no_model_or_hyperparameter_tuning=True))
    sources=[SOURCE,Path(__file__)]
    dump('provenance.json',dict(sources=[dict(**signature(p),sha256=sha256(p)) for p in sources],
        software=dict(numpy=np.__version__,pandas=pd.__version__,h5py=h5py.__version__,matplotlib=matplotlib.__version__)))
    print('No-replacement sensitivity',flush=True)
    print(unique_comp[['method','interval','n_selected','matching_coverage','mean_cosine_selected','mean_cosine_control_mean','mean_cosine_excess']].to_string(index=False),flush=True)
    print(comp[['method','interval','n_selected','matching_coverage','mean_cosine_selected','mean_cosine_control_mean','mean_cosine_excess']].to_string(index=False),flush=True)


def make_plots(results,controls,comp,unique_comp,context,scores,test,eligible):
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(10.5,7.4))
    for col,interval in enumerate(['16-19','19-24']):
        for row in range(2):
            metric='mean_cosine';label='Raw A/B rate-vector cosine';table=comp if row==0 else unique_comp
            ax=axs[row,col]
            baseline=results[(results.method=='all_eligible')&(results.interval==interval)][metric].iloc[0]
            ax.axvline(baseline,color='#a0a0a0',ls='--',lw=1,label='All eligible peaks')
            for y,key in enumerate(METHODS):
                r=table[(table.method==key)&(table.interval==interval)].iloc[0]
                selected=r[metric+'_selected'];mean=r[metric+'_control_mean'];lo=r[metric+'_control_p05'];hi=r[metric+'_control_p95']
                ax.plot([lo,hi],[y+.08,y+.08],c='#6c6c6c',lw=3,solid_capstyle='round')
                ax.scatter(mean,y+.08,s=28,color='#6c6c6c',zorder=3)
                ax.scatter(selected,y-.08,s=55,color=COLORS[key],zorder=4)
                annotation=f'n={int(r.n_selected):,}'
                if row==0:annotation+=f'; {100*r.control_mean_duplicate_fraction:.0f}% repeats'
                ax.text(.98,y+.04,annotation,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=7.5,color='#555555')
            ax.set_yticks(range(3),[LABELS[k] for k in METHODS] if col==0 else [])
            ax.set_ylim(2.55,-.55);ax.set_xlabel(label);ax.set_title(interval.replace('-','→')+' hpf | '+('All matchable peaks' if row==0 else 'No-replacement sensitivity'),fontsize=9)
            ax.grid(axis='x',alpha=.15)
            ax.set_xlim(-.02,.42)
    fig.suptitle('Does past-window peak filtering enrich reproducible later changes?',fontsize=13,y=.98)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#217c86',markeredgecolor='#217c86',label='Selected peaks'),
       Line2D([0],[0],marker='o',color='#6c6c6c',lw=3,label='Matched controls: mean and 5–95% range'),
       Line2D([0],[0],color='#a0a0a0',ls='--',label='All eligible peaks')],loc='upper center',bbox_to_anchor=(.55,.935),ncol=2,fontsize=8,frameon=False)
    fig.text(.5,.025,'Select using half A at 10/12/14 hpf; evaluate independent A/B changes at 16/19/24 hpf.\nTop 20% training thresholds; controls match past mean CPM and detection. Sampling reproducibility, not future prediction.\nLower row: outcome-blind common-support sensitivity; no repeated control peaks. Bands are control-draw ranges, not confidence intervals.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.265,right=.98,bottom=.17,top=.81,hspace=.65,wspace=.18)
    fig.savefig(OUT/'01_future_window_reproducibility.png',dpi=220);fig.savefig(OUT/'01_future_window_reproducibility.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,4.2))
    use=test&eligible
    rng=np.random.default_rng(SEED);ix=np.flatnonzero(use);ix=rng.choice(ix,min(3000,len(ix)),replace=False)
    ax=axs[0];sc=ax.scatter(scores.accessibility_gini.to_numpy()[ix],scores.temporal_fraction.to_numpy()[ix],c=np.log10(scores.past_mean_cpm.to_numpy()[ix]+1e-3),s=4,cmap='viridis',alpha=.6,rasterized=True,linewidths=0)
    ax.set_xlabel('Accessibility Gini (past 63 pseudobulks)');ax.set_ylabel('Temporal share of total log-CPM variance');ax.set_title('Specificity and dynamics differ')
    cb=fig.colorbar(sc,ax=ax,pad=.02,fraction=.05);cb.set_label('log10 mean CPM',fontsize=8)
    x=np.arange(3);cx=context.set_index('method')
    for j,key in enumerate(METHODS):
        axs[1].bar(j,cx.loc[key,'median_temporal_fraction'],color=COLORS[key]);axs[2].bar(j,cx.loc[key,'median_past_mean_cpm'],color=COLORS[key])
    for ax in axs[1:]:ax.set_xticks(x,['Gini','Total\nvariance','Temporal\nvariance']);ax.grid(axis='y',alpha=.15)
    axs[1].set_ylabel('Median temporal share');axs[1].set_title('Selected peaks: temporal contribution');axs[1].axhline(cx.loc['all_eligible','median_temporal_fraction'],ls='--',color='.5',lw=1)
    axs[2].set_ylabel('Median past CPM');axs[2].set_title('Abundance differs before matching');axs[2].axhline(cx.loc['all_eligible','median_past_mean_cpm'],ls='--',color='.5',lw=1)
    fig.suptitle('What do the three selection rules retain?',fontsize=13)
    fig.text(.5,.015,'Whole-pseudobulk variance = static celltype variance + within-celltype temporal variance (balanced 21×3 panel).\nDashed lines: all eligible held-out peaks. These are past-window descriptors; future outcomes do not define selection.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.065,right=.98,bottom=.26,top=.83,wspace=.50)
    fig.savefig(OUT/'02_selection_meaning_and_abundance.png',dpi=220);fig.savefig(OUT/'02_selection_meaning_and_abundance.pdf');plt.close(fig)


def write_methods(design,thresholds,comp,unique_comp,context,results):
    lines=['PAST-WINDOW VARIABLE-PEAK PILOT','',
      'Purpose: compare accessibility concentration (Gini), global pseudobulk variability, and within-celltype temporal variability as selectors for later reproducible raw changes. This is not a prediction model.',
      '',
      'Selection uses only half-A observations at10,12,14hpf. Evaluation uses A/B changes at16→19 and19→24hpf, with no shared stage observation between selection and evaluation. All21 CTs in the existing complete six-stage panel are retained. Whole-peak training/test splits are fixed (20,000/5,000);56 marker extras are excluded. Eligibility requires20 total half-A raw counts over the63 selection pseudobulks.',
      '',
      'Accessibility Gini is computed on63 nonnegative linear-CPM values. It measures concentration and is not temporal variance. The two variance statistics use log1p(CPM): total population variance across63 entries, or mean within-CT population variance overthree stages. Training-eligible mean-CPM deciles define mean bins. Each variance is converted to a midrank empirical percentile within its training mean bin. Strictly exceeding the training80th-percentile score defines selection; realized counts are reported.',
      '',
      'Controls come from unselected eligible held-out peaks, with exact counts in training-defined past-mean-CPM decile×detection-quintile strata. Thirty deterministic random draws are used. Sampling is without replacement when possible, with replacement where a stratum has insufficient candidates. Strata with no available control are excluded from the matched comparison, and matching coverage is reported. All-selected results remain in the metrics table. Control bands are5th–95th percentile variation across draws, not confidence intervals.',
      '',
      'Primary endpoint: mean cosine between21-dimensional raw CPM/hpf change vectors from cell halves A and B. Both vector norms must be nonzero. Secondary endpoints report nonzero-vector/CT coverage, sign agreement among CTs whose two rates are nonzero, and raw change energy. Rates use actual3h or5h gaps. No fitted rates, UMAP, smoothing, or model optimization are used.',
      '',
      'Interpretation: higher later A/B agreement would support enrichment for reproducibly changing peaks, after the stated matching. It would not establish cross-peak predictive information, causality, or a global flow. Cell halves share samples and do not constitute biological replication. Relative CPM, sample/batch confounding, selection on onlythree early stages, and remaining unmatched biological properties limit interpretation. Later-emerging programs can be missed. The two evaluation intervals share19hpf and should not be treated as independent replications.',
      '',
      'Common-support sensitivity (secondary, added after observing poor matching overlap): selected peaks are downsampled within each matching bin to the number of available unselected controls using a fixed random seed. This uses no outcome information. Controls are then sampled without replacement. The original comparison is retained. This sensitivity retains270 Gini-selected,941 total-variance-selected, and996 temporal-variance-selected peaks; it estimates a narrower population. Gini primary draws had about74% repeated control entries, so their apparent full-cohort matching coverage should not be mistaken for broad independent control support.',
      '',
      'Further limits: random whole-peak train/test splitting does not separate genomic neighborhoods. The21-celltype panel was selected from whole-course cell-count coverage metadata, although peak-level filtering uses only the early stages. This is not a fully prospective study. Exact matching-bin counts do not ensure exact continuous covariate balance; matching_continuous_covariate_balance.csv reports residual differences.',
      '',
      'Reproducibility: design_prespecified.json was written before outcome evaluation. selection_frozen.json records score hashes before future-stage outcomes were read. Input signatures and SHA256 hashes are recorded in provenance.json; source files are read-only.',
      '',
      'RESULTS',comp[['method','interval','n_selected','matching_coverage','mean_cosine_selected','mean_cosine_control_mean','mean_cosine_excess']].to_string(index=False),
      '', 'SECONDARY NO-REPLACEMENT COMMON-SUPPORT SENSITIVITY',unique_comp[['method','interval','n_selected','matching_coverage','mean_cosine_selected','mean_cosine_control_mean','mean_cosine_excess']].to_string(index=False),
      '', 'ALL-ELIGIBLE BASELINE',results[results.method=='all_eligible'][['interval','n_peaks','mean_cosine','nonzero_vector_fraction','nonzero_ct_pair_fraction','mean_per_peak_sign_agreement']].to_string(index=False),
      '', 'COHORT CONTEXT',context.to_string(index=False),
      '', 'Matching caveat: despite exact bin counts, Gini-selected peaks retained substantial within-bin detection imbalance (mean standardized difference approximately-0.81 in the original comparison and-0.41 in the no-replacement sensitivity). Gini effects cannot be cleanly separated from sparsity. Corresponding detection differences were approximately-0.11 for total variance and-0.04 for temporal variance. Mean-CPM balance was much closer. See the complete balance CSV.',
      '',
      'Interpretation: mean-adjusted total variance gave the largest point estimates of later vector agreement and its excess over abundance/detection-matched controls in both windows. Within-CT temporal variance also enriched agreement but did not outperform total variance. Gini selected substantially less accessible, sparser peaks and had poor overlap with unselected controls; its common-support sensitivity retained only25.7% of selected peaks. Absolute vector cosines remain modest. These results support a variable-peak enrichment strategy for reproducibility and motivate a separate test of model prediction; they do not establish a validated forecast or a global chromatin stream.']
    text='\n'.join(lines)+'\n'
    for before,after in [('at10','at 10'),('at16','at 16'),('All21','All 21'),(';56','; 56'),('requires20','requires 20'),('the63','the 63'),('on63','on 63'),('acrossthree','across three'),('overthree','over three'),('across63','across 63'),('training80th','training 80th'),('are5th','are 5th'),('between21','between 21'),('actual3h','actual 3 h'),('or5h','or 5 h'),('onlythree','only three'),('share19hpf','share 19 hpf')]:text=text.replace(before,after)
    (OUT/'methods_and_results.txt').write_text(text)


if __name__=='__main__':main()
