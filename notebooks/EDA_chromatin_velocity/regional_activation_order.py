"""Conditional order test of regional opening/closing schedules on a fixed atlas.

A region's opening (closing) window center weights actual interval midpoints by
positive (negative) raw changes in mean CPM. No trajectory model is fitted.
All 720 shared stage permutations preserve the atlas and inter-region/celltype
profile covariance. Tail fractions describe this conditional ordering null;
they are not biological-replicate p-values. Regions are existing Leiden labels,
not validated branches. The atlas itself used all stage profiles.
"""
from pathlib import Path
import argparse,hashlib,itertools,json
import h5py,numpy as np,pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/activation_progression_v1'


def phase_window_center(profiles,times):
    """Return (...,2) opening/closing centers; incomplete or absent phase is NaN.

    Weights are change amounts, not per-hour rates; multiplying a whole profile
    by a positive factor leaves its timing unchanged. Six-stage completeness is
    required so missing endpoints cannot create or remove a change silently.
    """
    x=np.asarray(profiles,float);t=np.asarray(times,float)
    if x.shape[-1]!=len(t) or len(t)<2 or np.any(np.diff(t)<=0):
        raise ValueError('Profiles and increasing stage times must match')
    complete=np.isfinite(x).all(axis=-1);diff=np.diff(x,axis=-1)
    phase=np.stack((np.maximum(diff,0),np.maximum(-diff,0)),axis=-1)
    midpoint=(t[:-1]+t[1:])/2
    amount=phase.sum(axis=-2);num=np.sum(phase*midpoint[:,None],axis=-2)
    center=np.divide(num,amount,out=np.full_like(num,np.nan),where=amount>0)
    center[~complete]=np.nan
    return center


def row_spearman(distance,window,eligible,*,minimum_regions=6):
    """NaN-aware Spearman over the last axis for one or many orderings."""
    window=np.asarray(window,float);distance=np.broadcast_to(np.asarray(distance,float),window.shape)
    eligible=np.broadcast_to(np.asarray(eligible,bool),window.shape)
    good=eligible&np.isfinite(window)&np.isfinite(distance)
    rx=rankdata(np.where(good,distance,np.nan),axis=-1,nan_policy='omit')
    ry=rankdata(np.where(good,window,np.nan),axis=-1,nan_policy='omit')
    count=good.sum(axis=-1)
    mx=np.divide(np.nansum(rx,axis=-1),count,out=np.zeros(count.shape,float),where=count>0)
    my=np.divide(np.nansum(ry,axis=-1),count,out=np.zeros(count.shape,float),where=count>0)
    dx=np.where(good,rx-mx[...,None],0);dy=np.where(good,ry-my[...,None],0)
    den=np.sqrt(np.sum(dx*dx,axis=-1)*np.sum(dy*dy,axis=-1))
    result=np.divide(np.sum(dx*dy,axis=-1),den,out=np.full(count.shape,np.nan),where=(count>=minimum_regions)&(den>0))
    return result,count


def exact_tail(null,observed):
    null=np.asarray(null,float);good=np.isfinite(null)
    return float(np.mean(null[good]>=observed-1e-12)) if np.isfinite(observed) and good.any() else np.nan


def run(out):
    csv=out/'region_time_summary.csv';table=pd.read_csv(csv);table=table[table.common_panel].copy()
    times=np.sort(table.stage.unique());celltypes=np.sort(table.celltype.unique());region_codes=np.sort(table.region_code.unique())
    index=pd.MultiIndex.from_product([celltypes,region_codes,times],names=['celltype','region_code','stage'])
    if table.duplicated(['celltype','region_code','stage']).any():raise ValueError('Duplicated regional profile')
    ordered=table.set_index(['celltype','region_code','stage']).reindex(index)
    if ordered.observed_mean_cpm.isna().any() or not ordered.condition_valid.all():raise ValueError('The fixed celltype panel must contain all six measured stages')
    profiles=ordered.observed_mean_cpm.to_numpy().reshape(len(celltypes),len(region_codes),len(times))
    if np.any(profiles<0):raise ValueError('CPM must be nonnegative')
    sizes=ordered.n_peaks.to_numpy().reshape(profiles.shape)
    if not np.all(sizes==sizes[0,:,0][None,:,None]):raise ValueError('Region sizes vary across stage or celltype')
    sizes=sizes[0,:,0].astype(int)
    with h5py.File(out/'geometry.h5','r') as f:
        code=f['region_code'][:];distance=f['distance_hops'][:];labels=f['region_labels'].asstr()[:]
    if not np.array_equal(region_codes,np.arange(len(labels))):raise ValueError('Region labels do not align')
    if not np.array_equal(np.bincount(code,minlength=len(labels)),sizes):raise ValueError('Region sizes do not match geometry')
    regional_distance=np.bincount(code,weights=distance,minlength=len(labels))/sizes
    permutations=np.asarray(list(itertools.permutations(range(len(times)))),dtype='i1')
    if len(permutations)!=720 or not np.array_equal(permutations[0],np.arange(6)):raise ValueError('Expected all six-stage permutations with identity first')
    # C,R,P,T -> P,C,R,phase. The same column permutation is shared by every CT/region.
    windows=phase_window_center(profiles[...,permutations],times).transpose(2,0,1,3)
    mean=profiles.mean(axis=-1);amplitude=np.divide(np.ptp(profiles,axis=-1),mean,out=np.zeros_like(mean),where=mean>0)
    thresholds=np.array([.1,0.]);eligibility=np.stack([(sizes[None,:]>=1000)&(amplitude>=threshold)&np.isfinite(profiles).all(axis=-1) for threshold in thresholds])
    nulls=np.full((len(thresholds),len(celltypes),2,len(permutations)),np.nan);counts=np.zeros(nulls.shape,dtype='i2')
    for h in range(len(thresholds)):
        for c in range(len(celltypes)):
            for phase in range(2):
                nulls[h,c,phase],counts[h,c,phase]=row_spearman(regional_distance,windows[:,c,:,phase],eligibility[h,c],minimum_regions=6)
    # The prespecified primary family is 21 CTs × two phases, threshold0.10.
    # Every permutation uses the same available-statistic maximum rule.
    primary=nulls[0];finite=np.isfinite(primary)
    max_stat=np.max(np.where(finite,primary,-np.inf),axis=(0,1));max_stat[~finite.any(axis=(0,1))]=np.nan
    records=[]
    for h,threshold in enumerate(thresholds):
        for c,ct in enumerate(celltypes):
            for phase,name in enumerate(['opening','closing']):
                stat=nulls[h,c,phase];observed=stat[0]
                records.append(dict(celltype=ct,phase=name,relative_amplitude_threshold=float(threshold),minimum_region_peaks=1000,distance='region_mean_hops_to_any_broad_reference',eligible_regions=int(eligibility[h,c].sum()),phase_regions=int(counts[h,c,phase,0]),spearman=float(observed),greater_tail_fraction=exact_tail(stat,observed),primary_family_max_tail_fraction=exact_tail(max_stat,observed) if h==0 else np.nan,valid_permutations=int(np.isfinite(stat).sum()),minimum_regions=6,null_median=float(np.nanmedian(stat)),null_q05=float(np.nanquantile(stat,.05)),null_q95=float(np.nanquantile(stat,.95)),phase_regions_null_min=int(counts[h,c,phase].min()),phase_regions_null_max=int(counts[h,c,phase].max())))
    result=pd.DataFrame(records);result.to_csv(out/'regional_activation_order.csv',index=False)
    np.savez_compressed(out/'regional_activation_order_nulls.npz',permutations=permutations,celltypes=celltypes.astype(str),region_labels=labels.astype(str),region_codes=region_codes,region_sizes=sizes,stage_times=times,regional_distance=regional_distance,observed_profiles_cpm=profiles,relative_amplitude=amplitude,thresholds=thresholds,eligibility=eligibility,window_centers=windows[0],spearman_nulls=nulls,phase_region_counts=counts,primary_family_max_null=max_stat)
    primary_table=result[result.relative_amplitude_threshold==.1]
    strongest=primary_table.sort_values('spearman',ascending=False).head(8).to_dict('records')
    summary=dict(n_celltypes=len(celltypes),n_regions=len(labels),n_permutations=len(permutations),timepoints=times.tolist(),minimum_region_peaks=1000,minimum_regions_for_correlation=6,primary_relative_amplitude_threshold=.1,sensitivity_relative_amplitude_threshold=0.,phase_metric='Sum positive (or negative magnitude) adjacent raw regional mean CPM changes times actual interval midpoint, divided by summed changes. No fitting and no per-hour weighting.',selection='Region size>=1000 and six-stage range/mean>=threshold; selection is stage-order invariant. Complete six-stage profiles required.',distance='Unweighted mean across constituent peaks of nearest-broad-reference graph hop distance',null='All720 shared permutations of six measured-stage profiles; atlas graph and covariance across regions/celltypes fixed. Recompute change phases and rank correlations for every permutation. Undefined phases omitted identically; require at least6 available regions.',tail='Conditional one-sided greater tail, including identity and ties. Fractions are conditional on this atlas and are not biological-replicate p-values.',max_stat_family='42 primary CT×phase statistics at amplitude threshold0.10; each permutation uses maximum finite Spearman. This family correction excludes the threshold0 sensitivity.',global_primary_observed_max=float(max_stat[0]),global_primary_max_tail_fraction=exact_tail(max_stat,max_stat[0]),primary_nominal_tail_at_most_05=int((primary_table.greater_tail_fraction<=.05).sum()),primary_family_max_tail_at_most_05=int((primary_table.primary_family_max_tail_fraction<=.05).sum()),strongest_primary_correlations=strongest,raw_profile_csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),limitations=['Leiden atlas regions are not validated biological branches.','Region means pool heterogeneous peaks and may cancel opposing subprograms.','CPM is compositional relative accessibility, not absolute chromatin opening.','The existing atlas was built from all celltype-stage profiles; this conditional analysis is not independent lineage evidence.','A phase-window center is a center of change over five sampled intervals, not a precisely localized onset time.','No biological replicate uncertainty or region-size-adjusted precision is inferred.'])
    # Serialize NaNs as null for portable structured output.
    clean=json.loads(json.dumps(summary,default=lambda x:x.item() if isinstance(x,np.generic) else x,allow_nan=True).replace('NaN','null'))
    (out/'regional_activation_order.json').write_text(json.dumps(clean,indent=2,allow_nan=False)+'\n')
    plot(result,celltypes,max_stat,out)
    print(primary_table.sort_values('spearman',ascending=False).head(12).to_string(index=False));print('Global max:',max_stat[0],'tail:',exact_tail(max_stat,max_stat[0]),flush=True)


def plot(table,celltypes,max_stat,out):
    primary=table[table.relative_amplitude_threshold==.1];sensitivity=table[table.relative_amplitude_threshold==0]
    fig,axs=plt.subplots(1,3,figsize=(15,10),gridspec_kw={'width_ratios':[1.2,1.2,1.4]},constrained_layout=True)
    for ax,sub,title in [(axs[0],primary,'Primary: range/mean ≥0.10'),(axs[1],sensitivity,'Sensitivity: eligible regions\nNo amplitude cutoff')]:
        values=sub.pivot(index='celltype',columns='phase',values='spearman').reindex(celltypes)[['opening','closing']].to_numpy()
        p=sub.pivot(index='celltype',columns='phase',values='greater_tail_fraction').reindex(celltypes)[['opening','closing']].to_numpy()
        im=ax.imshow(values,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto');ax.set_xticks([0,1],['Opening','Closing']);ax.set_yticks(np.arange(len(celltypes)),celltypes if ax is axs[0] else ['']*len(celltypes));ax.set_title(title)
        for i in range(len(celltypes)):
            for j in range(2):
                ax.text(j,i,f'{values[i,j]:.2f}\ntail={p[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if abs(values[i,j])>.65 else '#222')
    fig.colorbar(im,ax=axs[:2],label='Spearman: region distance vs change-window center',fraction=.04)
    ax=axs[2];ax.hist(max_stat[np.isfinite(max_stat)],bins=30,color='#8e9fb3',edgecolor='white');ax.axvline(max_stat[0],color='#b23135',lw=2,label=f'Observed maximum={max_stat[0]:.3f}');ax.legend();ax.set_xlabel('Maximum correlation across42 CT×phase tests');ax.set_ylabel('Shared stage permutations');ax.set_title(f'Family maximum conditional null\nGreater tail={exact_tail(max_stat,max_stat[0]):.3f}')
    fig.suptitle('Regional opening/closing order on the fixed peak atlas\nTail denotes a conditional permutation fraction, not a biological-replicate p-value',fontsize=14)
    fig.savefig(out/'regional_activation_order.png',dpi=170);fig.savefig(out/'regional_activation_order.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUT);args=p.parse_args();run(args.output_dir)
