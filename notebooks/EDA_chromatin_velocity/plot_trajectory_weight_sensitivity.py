"""Render completed fit-weight tables and two bounded illustrative profiles.

Aggregate statistics are read from the completed 8,192-peak comparison. Only a
deterministic 128-peak subset is read again to select two illustrative profiles;
their equal-weight fits use the same frozen parameters. No atlas field is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trajectory_weight_sensitivity import DEFAULT_OUTPUT,_fit,_take,_stat,_json,MODE_LABELS


def illustrative_profiles(output,summary):
    source=Path(summary['source']['path'])
    if _stat(source)!=summary['source']:
        raise AssertionError('Source fingerprint differs from the completed sensitivity comparison')
    sample=pd.read_csv(output/'sampled_peaks.csv')
    rng=np.random.default_rng(20261021)
    selected=np.sort(rng.choice(sample.atlas_row.to_numpy(),min(128,len(sample)),replace=False))
    with h5py.File(source,'r') as f:
        values=_take(f['observed_cpm'],selected,128).astype(float)
        primary=_take(f['fitted_cpm'],selected,128)
        primary_mode=_take(f['modes'],selected,128)
        count_supported=_take(f['raw_total'],selected,128)>=20
        valid=f['condition_valid'][:].astype(bool)
        coverage=f['coverage'][:].astype(float)
        times=f['stages'][:].astype(float)
        common=f['common_celltypes'][:].astype(bool)
        ids=_take(f['peak_ids'],selected,128).astype(str)
        celltypes=f['celltypes'].asstr()[:]
        params=json.loads(f.attrs['parameters_json'])
    exposure=np.where(np.isfinite(coverage)&(coverage>0),coverage,1.)
    counts=np.rint(values*exposure[None,:,:]/1e6)
    variance=(counts+.5)*(1e6/exposure[None,:,:])**2
    alternative=_fit(values,times,valid,coverage,variance,count_supported,params,True)
    equal_mode=alternative['mode']
    difference=np.sqrt(np.mean((primary-alternative['fitted'])**2,axis=-1))
    available=common[None,:]&count_supported&np.isfinite(primary).all(axis=-1)&np.isfinite(alternative['fitted']).all(axis=-1)
    primary_change=np.isin(primary_mode,[2,3,4,5]);equal_change=np.isin(equal_mode,[2,3,4,5])
    candidates=[('Primary change; equal-weight no supported change',available&primary_change&(equal_mode==1)),
                ('Shared change; different fitted amplitude',available&(primary_mode==equal_mode)&primary_change&equal_change)]
    examples=[];rows=[]
    for label,mask in candidates:
        if examples:mask[examples[0]['subset_row']]=False
        if not mask.any():mask=available.copy()
        positions=np.argwhere(mask)
        if not len(positions):continue
        # A fixed upper-middle discrepancy quantile makes the example visible
        # without selecting the maximum discrepancy or implying representativeness.
        order=np.argsort(difference[tuple(positions.T)],kind='stable')
        i,c=positions[order[int(.75*(len(order)-1))]]
        example=dict(label=label,peak=ids[i],atlas_row=int(selected[i]),subset_row=int(i),
                     celltype=celltypes[c],celltype_index=int(c),coverage_mode=MODE_LABELS[int(primary_mode[i,c])],
                     equal_mode=MODE_LABELS[int(equal_mode[i,c])],curve_rmse_cpm=float(difference[i,c]),
                     candidate_trajectories=int(len(positions)),selection_quantile=.75)
        examples.append(example)
        for t,time in enumerate(times):
            rows.append(dict(example=len(examples),peak=ids[i],celltype=celltypes[c],time_hpf=time,
                             observed_cpm=values[i,c,t],coverage_fitted_cpm=primary[i,c,t],
                             equal_fitted_cpm=alternative['fitted'][i,c,t],coverage=coverage[c,t]))
    if _stat(source)!=summary['source']:raise AssertionError('Source changed while reading examples')
    pd.DataFrame(rows).to_csv(output/'illustrative_weighted_curves.csv',index=False)
    (output/'illustrative_profiles.json').write_text(json.dumps(_json(dict(
        candidate_peak_count=len(selected),candidate_selection_seed=20261021,
        candidates='Uniform subset of the original sampled peak IDs; examples selected by declared discrepancy quantile.',
        interpretation='Illustrative profiles, not representative frequencies; aggregate metrics use all 8,192 sampled peaks.',
        examples=examples,source_unchanged=True)),indent=2,allow_nan=False)+'\n')
    return pd.DataFrame(rows),examples


def plot(output=DEFAULT_OUTPUT):
    output=Path(output)
    summary=json.loads((output/'weight_sensitivity_summary.json').read_text())
    rates=pd.read_csv(output/'interval_rate_comparison.csv')
    rates=rates[(rates.panel=='fixed_common_celltypes')&(rates.scope=='both_supported_changes')]
    sources=pd.read_csv(output/'source_activation_comparison.csv')
    sources=sources[sources.panel=='fixed_common_celltypes']
    modes=pd.read_csv(output/'trajectory_mode_comparison.csv')
    modes=modes[modes.panel=='fixed_common_celltypes'].set_index('scope')
    curves,examples=illustrative_profiles(output,summary)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    blue,orange,gray='#007b98','#c47a20','#777777'
    x=np.arange(len(rates));labels=rates.interval.str.replace('-','→',regex=False).tolist()
    axes[0,0].bar(x,rates.joint_supported_sign_agreement,color=blue,width=.65)
    for j,row in enumerate(rates.itertuples()):
        axes[0,0].text(j,row.joint_supported_sign_agreement+.025,f'{row.joint_supported_sign_agreement:.2%}',ha='center',fontsize=8)
    axes[0,0].set_xticks(x,[f'{label}\nn={int(n):,}' for label,n in zip(labels,rates.joint_supported_changes)])
    axes[0,0].set(ylim=(0,1.12),ylabel='Fraction agreeing in sign',xlabel='Interval, hpf',
                  title='Signs of changes supported by both fits')
    axes[0,1].bar(x-.18,sources.primary_source_retention,.35,color=blue,label='Primary-source retention')
    axes[0,1].bar(x+.18,sources.active_jaccard,.35,color=orange,label='Shared / union of sources')
    axes[0,1].set_xticks(x,[f'{label}\nn={int(n):,}' for label,n in zip(labels,sources.coverage_active)])
    axes[0,1].set(ylim=(0,1),ylabel='Fraction of source-active peaks',xlabel='Interval, hpf; n = primary source count',title='Source inclusion changes with weighting')
    axes[0,1].legend(frameon=False,fontsize=8,loc='upper right')
    scopes=['all_trajectories','count_supported','either_change_mode']
    names=['All trajectories','Count-supported','Either fit labels\na change mode']
    values=modes.loc[scopes,'mode_agreement'].to_numpy()
    axes[0,2].bar(np.arange(3),values,color=[gray,blue,orange],width=.65)
    for j,value in enumerate(values):axes[0,2].text(j,value+.025,f'{value:.1%}',ha='center')
    axes[0,2].set_xticks(np.arange(3),[f'{name}\nn={int(n):,}' for name,n in zip(names,modes.loc[scopes,'n'])])
    axes[0,2].set(ylim=(0,1.12),ylabel='Exact mode agreement',title='High overall agreement hides changing cases')
    for ax,example in zip(axes[1,:2],examples):
        d=curves[curves.peak==example['peak']]
        ax.scatter(d.time_hpf,d.observed_cpm,c=gray,s=30,label='Measured CPM',zorder=3)
        ax.plot(d.time_hpf,d.coverage_fitted_cpm,color=blue,lw=2,label='Coverage-weighted fit')
        ax.plot(d.time_hpf,d.equal_fitted_cpm,color=orange,lw=2,label='Equal-stage fit')
        ax.set_xticks(d.time_hpf)
        ax.set(xlabel='hpf',ylabel='CPM',title=f"{example['peak']}\n{example['celltype'].replace('_',' ')}")
        ax.text(.02,.98,f"Coverage: {example['coverage_mode'].replace('_',' ')}\nEqual: {example['equal_mode'].replace('_',' ')}",transform=ax.transAxes,va='top',fontsize=8,
                bbox=dict(boxstyle='round',facecolor='white',alpha=.85,edgecolor='none'))
        ax.legend(frameon=False,fontsize=8,loc='lower right')
    axes[1,2].axis('off')
    axes[1,2].text(0,.98,
        f"{summary['n_sampled_peaks']:,} peaks sampled uniformly from the atlas\n21 fixed celltypes × six stages\n\n"
        "Both fits use identical CPM values, missingness,\nconditional count variances and frozen thresholds.\nOnly relative fitting/LOOCV weights differ.\n\n"
        "Sign agreement conditions on BOTH fits supporting\na change; source retention reports what drops out.\n\n"
        "Source-active means a supported celltype change\nand whole-peak signal/noise ≥2, before graph projection.\n\n"
        "Change modes: opening, closing, pulse or trough.\nUncertain and no-supported-change calls are separate.\n\n"
        "Two profiles are selected illustrations, not prevalence\nestimates. Low-count states remain in both fits.\n\n"
        "No normalization, graph, UMAP or primary field changed.\nThis is sensitivity analysis, not biological validation.",
        va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Shared change signs agree; inclusion and mode calls depend on fitting weights',fontsize=15)
    for suffix in ['png','pdf']:fig.savefig(output/f'01_fit_weight_sensitivity.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(_json(dict(figure=str(output/'01_fit_weight_sensitivity.png'),examples=examples)),indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    plot(parser.parse_args().output_dir)
