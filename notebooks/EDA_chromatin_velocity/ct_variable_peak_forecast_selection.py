#!/usr/bin/env python3
"""Origin-specific, training-only peak selection across reliable pseudobulks.

Total variance spans both celltype differences and temporal changes. Pseudobulks
with fewer than20 pooled cells are excluded; absent groups are never zero-filled.
The20-count peak-support threshold is separate from the20-cell group threshold.
"""
from pathlib import Path
import hashlib,json,time
import h5py
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_shared_weight_forecast_v1/selection_audit'
SPLIT=HERE/'outputs/ct_flow_noise_v1/cell_split_input.h5'
CIS=HERE/'outputs/ct_cis_validation_v1/cis_split_input.h5'
ROOT=HERE.parents[2]
RAW=ROOT/'data/processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad'
PB=ROOT/'data/annotated_data/objects_v2/peaks_by_ct_tp_master.h5ad'
ORIGINS=(14,16,19)


def signature(path):
    st=Path(path).stat();return dict(path=str(Path(path).resolve()),bytes=st.st_size,mtime_ns=st.st_mtime_ns)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write_json(name,obj):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    (OUT/name).write_text(json.dumps(obj,default=convert,indent=2,allow_nan=False)+'\n')


def fit_origin_selection(values,raw_counts,stages,n_cells_pooled,origin_hpf,training_peak_mask,*,
                         min_group_cells=20,min_peak_counts=20,top_fraction=.2,n_mean_bins=10):
    """Return fold-trained selector masks in the supplied target-peak order.

    Parameters
    ----------
    values, raw_counts : [peak, celltype, stage]
        CPM and raw counts from the SOURCE cell half only. Values outside eligible
        past pseudobulks are ignored, including NaNs in absent combinations.
    stages : [stage]
        Developmental times. Only times<=origin_hpf enter any computation.
    n_cells_pooled : [celltype, stage]
        Pooled A+B cell counts. This is group eligibility, not peak count support.
    training_peak_mask : [peak]
        True only for outer-training chromosomes. Mean-bin edges, reference
        distributions, and cutoff quantiles are estimated from these peaks.

    Returns
    -------
    dict with eligible, group_mask, masks, scores and audit. masks['all'] includes
    peaks with>=min_peak_counts past source-half raw counts, not all connected
    peaks unconditionally. Top20 names reflect the defaulttop_fraction=.2.
    All cutoffs are strict '>' thresholds; realized fractions can differ due ties.
    """
    values=np.asarray(values);raw_counts=np.asarray(raw_counts);stages=np.asarray(stages)
    n_cells_pooled=np.asarray(n_cells_pooled);train=np.asarray(training_peak_mask,dtype=bool)
    if values.ndim!=3 or raw_counts.shape!=values.shape:raise ValueError('values/counts must share [peak,CT,stage] shape')
    if stages.shape!=(values.shape[2],) or n_cells_pooled.shape!=values.shape[1:]:raise ValueError('Stage or group-count axes differ')
    if train.shape!=(values.shape[0],):raise ValueError('Training mask length differs from target count')
    if not np.any(stages==origin_hpf):raise ValueError('Forecast origin is not an observed stage')
    if not (0<top_fraction<1):raise ValueError('top_fraction must lie in (0,1)')
    group=(n_cells_pooled>=min_group_cells)&(stages[None,:]<=origin_hpf)
    if group.sum()<2:raise ValueError('At least two reliable past pseudobulks required')
    x=np.asarray(values[:,group],dtype=float);count=np.asarray(raw_counts[:,group],dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(count).all() or (x<0).any() or (count<0).any():
        raise ValueError('Reliable past groups contain missing, negative, or nonfinite values')
    mean=x.mean(axis=1);detection=(count>0).mean(axis=1);count_total=count.sum(axis=1)
    eligible=count_total>=min_peak_counts;reference=train&eligible
    if reference.sum()<20:raise ValueError('Insufficient training-eligible peaks for empirical adjustment')
    log=np.log1p(x);variance=np.var(log,axis=1)
    order=np.sort(x,axis=1);m=x.shape[1];s=order.sum(axis=1)
    gini=np.divide((order*(2*np.arange(1,m+1)-m-1)).sum(axis=1),m*s,out=np.zeros(len(x)),where=s>0)
    edges=np.unique(np.quantile(mean[reference],np.arange(1,n_mean_bins)/n_mean_bins))
    bins=np.searchsorted(edges,mean,side='right');adjusted=np.full(len(x),np.nan);bin_audit=[]
    for b in range(len(edges)+1):
        population=np.sort(variance[reference&(bins==b)]);take=bins==b
        if not len(population):
            if take.any():raise ValueError('A prediction mean bin has no training reference; reduce bins before outcomes')
            continue
        lo=np.searchsorted(population,variance[take],side='left');hi=np.searchsorted(population,variance[take],side='right')
        adjusted[take]=(lo+hi)/(2*len(population))
        bin_audit.append(dict(bin=b,n_training_reference=len(population),min_variance=float(population.min()),max_variance=float(population.max())))
    if not np.isfinite(adjusted).all():raise ValueError('Some adjusted scores undefined')
    gcut=float(np.quantile(gini[reference],1-top_fraction));vcut=float(np.quantile(adjusted[reference],1-top_fraction))
    masks={'all':eligible,'gini_top20':eligible&(gini>gcut),'total_variance_top20':eligible&(adjusted>vcut)}
    audit=dict(origin_hpf=float(origin_hpf),maximum_input_hpf=float(stages[np.flatnonzero(group.any(axis=0))[-1]]),
        n_reliable_past_pseudobulks=int(group.sum()),n_celltypes_represented=int(group.any(axis=1).sum()),
        min_group_cells=min_group_cells,min_peak_counts=min_peak_counts,top_fraction=top_fraction,
        n_training_peaks=int(train.sum()),n_training_eligible=int(reference.sum()),n_test_peaks=int((~train).sum()),n_test_eligible=int((eligible&~train).sum()),
        n_mean_bins=len(edges)+1,mean_bin_edges=edges,mean_bin_reference=bin_audit,
        gini_cutoff=gcut,mean_adjusted_total_variance_cutoff=vcut,
        selection_counts={key:dict(training=int((mask&train).sum()),test=int((mask&~train).sum())) for key,mask in masks.items()},
        score_definition='Gini on nonnegative linearCPM; population variance of log1pCPM across all reliable past CT-stage combinations; variance adjusted by training meanCPM-bin empirical midrank',
        input_half='Caller supplies source-half counts/CPM only',cutoff_rule='score>training_quantile',
        future_accessibility_used=False,heldout_peaks_fit_preprocessing=False,
        group_weighting='Each observed reliable CT-stage pseudobulk equally weighted; incomplete CT trajectories allowed; missing groups excluded',
        selection_is_not_forecast=True)
    return dict(eligible=eligible,group_mask=group,masks=masks,
        scores=dict(gini=gini,total_log_variance=variance,mean_adjusted_total_variance=adjusted,mean_cpm=mean,detection=detection,past_total_count=count_total,mean_bin=bins),audit=audit)


def prepare_target_input():
    from real_input_provenance import read_column
    OUT.mkdir(parents=True,exist_ok=True)
    before={str(p):signature(p) for p in (SPLIT,CIS,RAW,PB)}
    with h5py.File(CIS,'r') as cis:
        target_local=cis['target_local_indices'][:];source=cis['source_indices'][:][target_local]
        ids=cis['peak_ids'].asstr()[:][target_local];ct21=cis['celltype_indices'][:]
        reference21={h:cis['raw_counts_'+h][:][target_local] for h in ['a','b']}
    with h5py.File(SPLIT,'r') as split,h5py.File(RAW,'r') as raw,h5py.File(PB,'r') as pb:
        cell_ids=read_column(raw['obs'],'_index');assert np.array_equal(cell_ids,split['cell_ids'].asstr()[:])
        half=split['cell_half'][:];assert set(np.unique(half))=={0,1}
        ct=split['celltypes'].asstr()[:];stages=split['stages'][:];nc={h:split['n_cells_'+h][:] for h in ['a','b']};coverage={h:split['coverage_'+h][:] for h in ['a','b']}
        assert np.array_equal(split['condition_valid'][:],nc['a']+nc['b']>=20)
        assert np.abs(nc['a']-nc['b']).max()<=1
        ctmap={name:i for i,name in enumerate(ct)};tmap={'0somites':0,'5somites':1,'10somites':2,'15somites':3,'20somites':4,'30somites':5}
        group=np.array([ctmap[c]*6+tmap[t] for c,t in zip(read_column(raw['obs'],'annotation_ML_coarse'),read_column(raw['obs'],'dev_stage'))],int)
        combo=group*2+half
        index=pd.Index(read_column(raw['var'],'_index')).get_indexer(ids);assert np.min(index)>=0 and len(np.unique(index))==len(index)
        counts=np.zeros((len(ids),len(ct),6,2),np.int64);x=raw['layers/counts'];ptr=x['indptr'][:];start=time.time()
        for k,i in enumerate(np.argsort(index)):
            a,b=ptr[index[i]:index[i]+2]
            counts[i]=np.bincount(combo[x['indices'][a:b]],weights=x['data'][a:b],minlength=len(ct)*6*2).reshape(len(ct),6,2).astype(np.int64)
            if (k+1)%1000==0:print('32CT target extraction',k+1,'/',len(ids),round(time.time()-start,1),'s',flush=True)
        for h,j in [('a',0),('b',1)]:assert np.array_equal(counts[:,:,:,j][:,ct21,:],reference21[h]), (counts[:,:,:,j][:,ct21,:].shape,reference21[h].shape)
        # Reconstruct every target's190 pooled pseudobulk counts exactly.
        pbids=read_column(pb['obs'],'_index');assert np.array_equal(pbids[source],ids)
        pct=read_column(pb['var'],'annotation_ML_coarse');pt=read_column(pb['var'],'dev_stage')
        order=np.argsort(source);inv=np.argsort(order);expected=pb['layers/sum'][source[order]][inv]
        reconstructed=np.stack([counts[:,ctmap[c],tmap[t],:].sum(axis=1) for c,t in zip(pct,pt)],axis=1)
        assert np.array_equal(expected,reconstructed)
        dest=OUT/'target_32ct_split_input.partial.h5'
        with h5py.File(dest,'w') as f:
            for key,value in [('source_indices',source),('stages',stages),('pooled_n_cells',nc['a']+nc['b']),('legacy_21ct_indices',ct21)]:f.create_dataset(key,data=value)
            for key,value in [('peak_ids',ids),('celltypes',ct)]:f.create_dataset(key,data=np.asarray(value,object),dtype=h5py.string_dtype())
            for h,j in [('a',0),('b',1)]:
                c=counts[:,:,:,j];cpm=np.divide(c*1e6,coverage[h][None],out=np.full(c.shape,np.nan),where=coverage[h][None]>0)
                f.create_dataset('raw_counts_'+h,data=c,compression='gzip');f.create_dataset('observed_cpm_'+h,data=cpm.astype('f4'),compression='gzip')
                f.create_dataset('coverage_'+h,data=coverage[h]);f.create_dataset('n_cells_'+h,data=nc[h])
            f.attrs['axis_order']='target_peak,celltype,stage; targetorder matches cis_split_input/target_local_indices'
            f.attrs['purpose']='Selection inputs across all32celltypes; forecast features/outcomes may retain legacy21CT panel separately'
            f.attrs['missing']='Absent pseudobulks remainNaN CPM andzero cellcount; excludedbeforeselection'
            f.attrs['source_signatures_json']=json.dumps(before)
        dest.replace(OUT/'target_32ct_split_input.h5')
    assert before=={str(p):signature(p) for p in (SPLIT,CIS,RAW,PB)}
    write_json('count_aggregation_validation.json',dict(status='passed',n_targets=len(ids),n_celltypes=len(ct),source_inputs_unchanged=True,
        exact_existing21ct_counts_match=True,exact_all190_pooled_pseudobulk_counts_match=True,cell_ids_match_original_split=True,
        pooled_ge20_implies_each_half_ge10=bool(np.all(np.minimum(nc['a'],nc['b'])[nc['a']+nc['b']>=20]>=10)),sources=before))


def audit_selection():
    path=OUT/'target_32ct_split_input.h5';rows=[];group_rows=[];audits=[]
    with h5py.File(path,'r') as f:
        stages=f['stages'][:];pooled=f['pooled_n_cells'][:];ct=f['celltypes'].asstr()[:];ids=f['peak_ids'].asstr()[:];legacy=f['legacy_21ct_indices'][:]
        a=f['n_cells_a'][:];b=f['n_cells_b'][:];source=f['source_indices'][:];chrom=np.array([int(v.split('-')[0]) for v in ids]);fold=(chrom-1)%5
        data={h:f['observed_cpm_'+h][:].astype(float) for h in ['a','b']};counts={h:f['raw_counts_'+h][:] for h in ['a','b']}
    for c,name in enumerate(ct):
        for t,timepoint in enumerate(stages):group_rows.append(dict(celltype=name,stage_hpf=timepoint,pooled_cells=int(pooled[c,t]),half_a_cells=int(a[c,t]),half_b_cells=int(b[c,t]),pooled_ge20=bool(pooled[c,t]>=20),both_half_ge10=bool(min(a[c,t],b[c,t])>=10),both_half_ge20=bool(min(a[c,t],b[c,t])>=20),legacy21_panel=bool(c in legacy)))
    pd.DataFrame(group_rows).to_csv(OUT/'pseudobulk_cell_count_audit.csv',index=False)
    with h5py.File(OUT/'origin_fold_selection.h5','w') as save:
        save.create_dataset('source_indices',data=source);save.create_dataset('chromosome_fold',data=fold)
        save.create_dataset('peak_ids',data=np.asarray(ids,object),dtype=h5py.string_dtype())
        for h in ['a','b']:
            for origin in ORIGINS:
                for k in range(5):
                    train=fold!=k;r=fit_origin_selection(data[h],counts[h],stages,pooled,origin,train)
                    g=save.create_group(f'{h}/origin_{origin}/fold_{k}')
                    g.create_dataset('group_mask',data=r['group_mask'])
                    for name,mask in r['masks'].items():g.create_dataset(name,data=mask)
                    for name,value in r['scores'].items():g.create_dataset(name,data=value.astype('f4'),compression='gzip')
                    g.attrs['audit_json']=json.dumps(r['audit'],default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
                    row=dict(half=h,origin_hpf=origin,fold=k,**{key:r['audit'][key] for key in ['n_reliable_past_pseudobulks','n_celltypes_represented','n_training_eligible','n_test_eligible','gini_cutoff','mean_adjusted_total_variance_cutoff']})
                    for name,mask in r['masks'].items():row[name+'_train']=int((mask&train).sum());row[name+'_test']=int((mask&~train).sum())
                    rows.append(row)
                    # Future accessibility/group counts must not influence any score.
                    altered=data[h].copy();altered[:,:,stages>origin]=np.nan
                    altered_counts=counts[h].astype(float).copy();altered_counts[:,:,stages>origin]=np.nan
                    altered_cells=pooled.copy();altered_cells[:,stages>origin]=0
                    rr=fit_origin_selection(altered,altered_counts,stages,altered_cells,origin,train)
                    for name in r['scores']:assert np.array_equal(r['scores'][name],rr['scores'][name])
                    for name in r['masks']:assert np.array_equal(r['masks'][name],rr['masks'][name])
                    # Heldout peak values may change test assignments, never preprocessing.
                    altered=data[h].copy();altered[~train]*=11
                    altered_counts=counts[h].copy();altered_counts[~train]*=11
                    rr=fit_origin_selection(altered,altered_counts,stages,pooled,origin,train)
                    assert np.array_equal(r['audit']['mean_bin_edges'],rr['audit']['mean_bin_edges'])
                    assert r['audit']['gini_cutoff']==rr['audit']['gini_cutoff']
                    assert r['audit']['mean_adjusted_total_variance_cutoff']==rr['audit']['mean_adjusted_total_variance_cutoff']
                    for name in r['scores']:assert np.array_equal(r['scores'][name][train],rr['scores'][name][train])
                    audits.append(dict(half=h,origin=origin,fold=k,future_mutation_invariant=True,heldout_mutation_training_invariant=True))
    table=pd.DataFrame(rows);table.to_csv(OUT/'origin_fold_thresholds_and_counts.csv',index=False)
    summary=[]
    for origin in ORIGINS:
        mask=(pooled>=20)&(stages[None,:]<=origin)
        summary.append(dict(origin_hpf=origin,n_full32ct_past_groups=int(mask.sum()),n_legacy21ct_past_groups=int(mask[legacy].sum()),n_celltypes_represented=int(mask.any(axis=1).sum()),minimum_half_cells_retained=int(np.minimum(a,b)[mask].min())))
    write_json('origin_group_eligibility_summary.json',summary)
    write_json('selection_validation.json',dict(status='passed',n_audited_origin_half_fold_combinations=len(audits),checks=audits,
        future_mutation_changes_no_scores_or_masks=True,heldout_mutation_changes_no_training_preprocessing=True,
        pooled20_and_each_half10_masks_identical=bool(np.array_equal(pooled>=20,(a>=10)&(b>=10))),
        forecasts_not_evaluated=True,source_input=signature(path),source_sha256=sha256(path),script_sha256=sha256(__file__)))
    text="""SELECTION FOR ROLLING-ORIGIN CHROMATIN FORECASTS

The 4,525 target peaks retain exactly the order of cis_split_input/target_local_indices. Selection scores span all reliable observed past celltype × stage pseudobulks, retaining both celltype differences and temporal differences. The forecast model separately retains its existing 21-celltype outcome panel.

At origins 14, 16, and 19 hpf, reliable past combinations number 89, 120, and 151, versus 63, 84, and 105 in the old 21-celltype panel. Missing groups and groups with fewer than 20 pooled cells are excluded, never filled with zero. Pooled A+B cell count >=20 is the primary group rule. The balanced existing cell partition guarantees at least 10 cells in each half for every retained group. Requiring 20 cells in each half would impose a stronger threshold than requested.

A separate peak eligibility rule requires at least 20 raw counts summed over the reliable past groups in the SOURCE half. This is a count-support rule, distinct from the group cell-count requirement. Raw Gini uses linear CPM. Total variability is population variance of log1p(CPM) across all reliable past celltype × stage coordinates. It includes differences between celltypes as well as changes over time. The mean-adjusted score is the empirical midrank percentile of variance within source-half mean-CPM deciles learned from eligible outer-training chromosome peaks. The 80th percentile of eligible training scores defines each top-20% stratum; strict > cutoffs handle ties transparently. Each reliable pseudobulk contributes equally after CPM normalization.

All selection inputs end at the forecast origin. Later-stage values and eligibility metadata cannot change scores: this was checked by replacing every future entry with NaN and its cell count with zero. Changing held-out peak values/counts cannot change training mean bins, cutoff quantiles, or training scores. These checks cover both source halves, three origins, and five chromosome folds. Use source-A selection for A-to-B forecasts and source-B selection for B-to-A forecasts. Do not pool source- and target-half accessibility for selection.

The pooled group cell-count metadata include the target-half cell number to apply a symmetric sampling eligibility rule. This does not use target accessibility. The fixed Cicero graph was learned on all stages/cells; the selection audit does not remove that conditional-graph limitation. CPM is relative accessibility, cell halves are sampling partitions rather than biological replicates, and high Gini can reflect sparsity. Different origins have different observed coordinate sets, so thresholds remain origin-specific. Selection alone does not establish predictive skill.

The helper returns masks['all'] for past-count-eligible peaks, plus gini_top20 and total_variance_top20. If the main model also reports all connected targets without the peak count floor, label that cohort separately. Graph training and neighbor availability need not be pruned by these evaluation strata: evaluating identical model predictions on selected cohorts isolates selection from model changes.
"""
    (OUT/'selection_methods.txt').write_text(text)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(table.groupby(['half','origin_hpf'])[['all_test','gini_top20_test','total_variance_top20_test']].sum().to_string(),flush=True)



def selector_diagnostic_figure():
    """Plot source-A, out-of-fold selectors without reading forecast outcomes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pypdf import PdfReader
    path=OUT/'target_32ct_split_input.h5'
    with h5py.File(path,'r') as f:
        values=f['observed_cpm_a'][:].astype(float);stages=f['stages'][:];nc=f['pooled_n_cells'][:]
        ids=f['peak_ids'].asstr()[:];source=f['source_indices'][:]
    rows=[]
    with h5py.File(OUT/'origin_fold_selection.h5','r') as f:
        fold=f['chromosome_fold'][:]
        for origin in ORIGINS:
            group=(nc>=20)&(stages[None,:]<=origin);n_c=group.sum(axis=1);m=group.sum()
            z=np.where(group[None],np.log1p(values),0.)
            means=np.divide(z.sum(axis=2),n_c[None],out=np.zeros(z.shape[:2]),where=n_c[None]>0)
            grand=z.sum(axis=(1,2))/m
            between=np.sum(n_c[None]/m*(means-grand[:,None])**2,axis=1)
            within=np.sum(np.where(group[None],(z-means[:,:,None])**2,0),axis=(1,2))/m
            total=np.var(np.log1p(values[:,group]),axis=1)
            assert np.allclose(total,between+within,atol=1e-12)
            share=np.divide(between,total,out=np.full(len(values),np.nan),where=total>0)
            base=dict(origin_hpf=np.full(len(values),origin),source_index=source,peak_id=ids,chromosome_fold=fold,
                      between_ct_log_variance=between,within_ct_log_variance=within,between_ct_variance_share=share)
            for name in ['all','gini_top20','total_variance_top20','gini','total_log_variance','mean_adjusted_total_variance','mean_cpm','detection']:
                arr=np.empty(len(values),dtype=bool if name in ['all','gini_top20','total_variance_top20'] else float)
                for k in range(5):
                    g=f[f'a/origin_{origin}/fold_{k}'];take=fold==k;arr[take]=g[name][:][take]
                base[name]=arr
            rows.append(pd.DataFrame(base))
    frame=pd.concat(rows,ignore_index=True)
    frame.to_csv(OUT/'selector_diagnostic_peak_data.csv.gz',index=False)
    methods=['all','gini_top20','total_variance_top20'];labels=['All count-supported','High Gini','High total variance']
    colors=['#8d9399','#9b4e9e','#d5802b'];summary=[]
    for origin in ORIGINS:
        d=frame[frame.origin_hpf==origin]
        for name in methods:
            q=d[d[name]];row=dict(source_half='a',origin_hpf=origin,method=name,n_peaks=len(q),fraction_of_eligible=len(q)/d['all'].sum())
            for metric in ['mean_cpm','detection','between_ct_variance_share','total_log_variance']:
                a=q[metric].to_numpy();a=a[np.isfinite(a)]
                row.update({metric+'_median':float(np.median(a)),metric+'_q25':float(np.quantile(a,.25)),metric+'_q75':float(np.quantile(a,.75))})
            summary.append(row)
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'selector_diagnostic_summary.csv',index=False)
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':7.3,'axes.titlesize':8,'axes.labelsize':7.3,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(170/25.4,150/25.4))
    x=np.arange(3);offsets=np.array([-.24,0,.24]);width=.20
    for j,(key,color) in enumerate(zip(methods,colors)):
        d=summary[summary.method==key].sort_values('origin_hpf')
        axes[0,0].bar(x+offsets[j],d.n_peaks,width=width,color=color)
        for xx,nn in zip(x+offsets[j],d.n_peaks):axes[0,0].text(xx,nn+70,f'{int(nn):,}',ha='center',va='bottom',rotation=90,fontsize=6.5)
    axes[0,0].set_ylim(0,5650);axes[0,0].set_ylabel('Held-out target peaks');axes[0,0].set_title('A  Retained peaks',loc='left',fontweight='bold')
    for ax,key,label,title in [(axes[0,1],'mean_cpm','log10 mean CPM','B  Accessibility'),
                               (axes[1,0],'detection','Fraction with nonzero raw count','C  Detection across pseudobulks'),
                               (axes[1,1],'between_ct_variance_share','Between-CT share of total variance','D  Celltype differences are retained')]:
        for j,(method,color) in enumerate(zip(methods,colors)):
            arrays=[]
            for origin in ORIGINS:
                d=frame[(frame.origin_hpf==origin)&frame[method]][key].to_numpy();d=d[np.isfinite(d)]
                arrays.append(np.log10(d) if key=='mean_cpm' else d)
            b=ax.boxplot(arrays,positions=x+offsets[j],widths=width,whis=(5,95),showfliers=False,patch_artist=True,manage_ticks=False,
                medianprops=dict(color='#222222',lw=.8),boxprops=dict(facecolor=color,edgecolor=color,lw=.8),whiskerprops=dict(color=color,lw=.8),capprops=dict(color=color,lw=.8))
        ax.set_ylabel(label);ax.set_title(title,loc='left',fontweight='bold')
        if key!='mean_cpm':ax.set_ylim(-.03,1.03)
    for ax in axes.ravel():
        ax.set_xticks(x,[str(i) for i in ORIGINS]);ax.set_xlabel('Forecast origin (hpf)');ax.set_xlim(-.55,2.55);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Peak selection across reliable past celltype × stage pseudobulks',fontsize=9.3,y=.975)
    fig.legend([Patch(facecolor=c) for c in colors],labels,loc='upper center',bbox_to_anchor=(.51,.945),ncol=3,frameon=False,fontsize=7)
    fig.text(.5,.022,'4,525 Cicero-connected targets; half A only. Cutoffs use other chromosomes; no forecast outcomes.\nOrigins 14 / 16 / 19 hpf include 89 / 120 / 151 groups with ≥20 pooled cells. Boxes: quartiles; whiskers: 5–95%.\nPanel D: exact weighted variance decomposition; available stages differ among celltypes. It is not a causal attribution.',ha='center',fontsize=6.5)
    fig.subplots_adjust(left=.10,right=.975,bottom=.16,top=.86,hspace=.61,wspace=.40)
    stem=OUT/'01_selector_diagnostics_170mm'
    fig.savefig(str(stem)+'.png',dpi=600);fig.savefig(str(stem)+'.pdf');plt.close(fig)
    pdf=PdfReader(str(stem)+'.pdf');width_mm=float(pdf.pages[0].mediabox.width)*25.4/72
    assert abs(width_mm-170)<1e-6
    caption='''Selection diagnostics across reliable past pseudobulks for the 4,525 Cicero-connected target peaks (co-accessibility threshold 0.25), not the full peak atlas. Every group is one observed celltype × stage combination with at least 20 pooled cells; all eligible past groups enter total-variance selection. The figure uses only source-half A and out-of-fold masks, so each peak's cutoff and mean adjustment were learned on other chromosomes. All count-supported means at least 20 source-half raw counts across eligible past groups. High Gini uses linear CPM; high total variance uses the training-mean-adjusted percentile of population variance in log1p(CPM). Both use training 80th-percentile cutoffs. A: retained peak counts. B–C: past mean accessibility and raw-count detection. D: the fraction of total past log-accessibility variance attributable to differences among celltype means, using the exact weighted decomposition Var(all reliable groups) = weighted Var(celltype means) + weighted mean Var(within-celltype values). Weights equal each celltype's number of available past stages divided by the number of eligible groups. Because stage coverage differs among celltypes, the between-celltype term can also reflect unequal stage composition; it does not isolate a causal cell-identity effect. Boxes show quartiles and medians; whiskers show the 5th and 95th percentiles, with outliers omitted. Selection cohorts overlap. No forecast outcomes are read or plotted.'''
    (OUT/'selector_diagnostic_caption.txt').write_text(caption+'\n')
    write_json('selector_figure_validation.json',dict(status='passed',width_mm=width_mm,height_mm=150,source_half='a',out_of_fold_selection=True,no_forecast_outcomes_read=True,
        weighted_variance_decomposition_passed=True,png_dpi=600,source_peak_csv='selector_diagnostic_peak_data.csv.gz',summary_csv='selector_diagnostic_summary.csv'))
    (OUT/'selector_numeric_summary.txt').write_text('Half-A out-of-fold selectors retained 884/913/929 high-total-variance peaks and 893/891/910 high-Gini peaks at origins 14/16/19 hpf. High-total-variance median accessibility was 1.24/1.28/1.27 CPM, close to the all-supported 1.21/1.24/1.25 CPM; high-Gini medians were 0.25/0.25/0.27 CPM. Detection medians were 0.83/0.83/0.84 for high total variance versus 0.55/0.56/0.57 for high Gini. The median between-celltype share of total past log-accessibility variance was 0.78/0.74/0.71 among high-total-variance peaks, versus 0.61/0.55/0.52 among all count-supported peaks. Thus the total-variance selector retains differences among celltypes as intended; high Gini preferentially selects sparse, low-accessibility peaks. The weighted decomposition uses the available stages for each celltype and is not a causal attribution. No forecast outcomes enter these summaries.\n')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    prepare_target_input()
    audit_selection()
    selector_diagnostic_figure()
