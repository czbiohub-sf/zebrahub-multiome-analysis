"""Static figures, offline display subset, and invariants for the real atlas."""
from pathlib import Path
import json
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/chrom_velo'))
from real_specialization import to_condition_cube


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    return value


def write_json(path,data):
    path.write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n')



def specificity_presentation(a):
    """Describe estimator provenance without changing scores or field arrays."""
    metadata = a.uns.get('real_specialization', {})
    matched = str(metadata.get('specificity_estimator', 'raw')) == 'depth_matched'
    if matched:
        thinning = metadata.get('depth_matching', {})
        n = int(thinning.get('n_thinnings', metadata.get('n_thinnings', 5)))
        seed = int(thinning.get('seed', metadata.get('seed', 4242)))
        agreement = thinning.get('min_thinning_sign_agreement')
        gate_note = ''
        gate_panel = ''
        gate_banner = ''
        if agreement is not None:
            required = int(np.ceil(float(agreement) * n - 1e-10))
            gate_note = (f'Displayed temporal arrows require agreement with the mean-rate sign in at least {required}/{n} thinnings. '
                         'This is a computational stability filter, not a biological confidence interval. Scalar colors retain the mean rates before this display filter. ')
            gate_panel = f'\nTemporal arrows: ≥{required}/{n} matching signs.\nComputational sign stability only.'
            gate_banner = f' Temporal arrows require ≥{required}/{n} matching signs.'
        return dict(
            estimator='depth_matched', title='Depth-matched specificity',
            structural_note=f'Specificity averaged over {n} technical count-thinnings; these are not biological replicates.',
            notice=(f'Exploratory depth-matched specificity. Counts are thinned to a shared target sequencing depth across stages within each celltype. '
                    f'Scores average {n} technical count-thinnings (seed {seed}); these are not biological replicates. '
                    'Depth matching reduces count support and does not remove every confounder. '+gate_note+
                    'Accessibility curves and opening/closing colors retain the original measured CPM values.'),
            figure_notice=f'Exploratory: {n} technical count-thinnings, not biological replicates.'+gate_banner,
            method_note=(f'Specificity alone is depth matched across stages within each celltype and averaged over {n} technical count-thinnings '
                         f'(seed {seed}). Every thinning must pass the pooled-count threshold at each endpoint. '+gate_note+'Original measured CPM accessibility and opening/closing rates are unchanged.'),
            panel_note=(f'Depth-matched specificity change\n\n{n} technical count-thinnings;\nnot biological replicates.\nLower count support after thinning.\nResidual confounding remains possible.'+gate_panel),
            support_title='Depth-matched specificity with count support',
        )
    return dict(
        estimator='raw', title='Raw specificity · depth-confounded',
        structural_note='Specificity uses original depth-unmatched counts.',
        notice=('Exploratory raw specificity: depth-confounded temporal overlay. Depth-only controls can generate apparent temporal specificity changes. '
                'These raw temporal arrows must not be interpreted as developmental flow. '
                'Accessibility curves and opening/closing colors show the original measured CPM values.'),
        figure_notice='Depth-confounded temporal overlay: raw specificity arrows are not evidence of developmental flow.',
        method_note='Specificity uses the original depth-unmatched counts. Depth-only controls reveal confounding of temporal specificity changes.',
        panel_note=('Raw specificity change\n\nDEPTH-CONFOUNDED OVERLAY\nDepth-only controls can generate\napparent temporal changes.\nNot evidence of developmental flow.'),
        support_title='Raw specificity with ≥20 pooled counts',
    )


def add_interpretation_banner(fig, presentation, y=.94, structural=False):
    text = ('Chosen orientation toward greater 24 hpf specificity; not evidence of developmental motion.'
            if structural else presentation['figure_notice'])
    fig.text(.5, y, text, ha='center', va='top',
             fontsize=10, color='#8a3e16', weight='semibold')


def choose_arrows(xy,vectors,mask,quality=None,budget=260):
    valid=np.asarray(mask)&np.isfinite(vectors).all(axis=1)&(np.linalg.norm(vectors,axis=1)>1e-10)
    ids=np.flatnonzero(valid)
    if len(ids)<=budget:return ids
    # Spatial coverage is a display choice, using only already-computed vectors.
    xy0=np.min(xy,axis=0);span=np.maximum(np.ptp(xy,axis=0),1e-10)
    bins=np.minimum((np.clip((xy[ids]-xy0)/span,0,1)*40).astype(int),39)
    keys=np.ravel_multi_index(bins.T,(40,)*xy.shape[1])
    q=np.linalg.norm(vectors[ids],axis=1) if quality is None else np.asarray(quality)[ids]
    order=np.lexsort((-q,keys));_,first=np.unique(keys[order],return_index=True)
    candidate=ids[order[first]]
    if len(candidate)<=budget:return candidate
    chosen=[int(candidate[np.argmax(q[order[first]])])]
    dist=np.full(len(candidate),np.inf)
    for _ in range(budget-1):
        dist=np.minimum(dist,np.sum((xy[candidate]-xy[chosen[-1]])**2,axis=1))
        chosen.append(int(candidate[np.argmax(dist)]))
    return np.asarray(chosen)


def display_vectors(v,coordinates):
    norm=np.linalg.norm(v,axis=-1,keepdims=True)
    length=np.linalg.norm(np.ptp(coordinates,axis=0))*.008
    return np.divide(v,norm,out=np.zeros_like(v),where=norm>0)*length


def plot_map(ax,a,background,values,title,vectors=None,mask=None,dim=2,rate=False,limit=None,unit=None,max_arrows=240):
    xy=a.obsm[f'X_umap{dim}d']; vals=np.asarray(values)
    good=np.isfinite(vals[background]);ids=background[good];missing=background[~good]
    ax.scatter(*xy[missing].T,c='#c7cbd1',s=.45 if dim==2 else .7,alpha=.35,rasterized=True,linewidths=0)
    if rate:
        limit=limit or max(float(np.nanquantile(np.abs(vals),.98)),1e-5)
        norm=TwoSlopeNorm(0,-limit,limit);cmap='RdBu_r'
    else:norm=Normalize(0,1);cmap='viridis'
    artist=ax.scatter(*xy[ids].T,c=vals[ids],s=.55 if dim==2 else .7,alpha=.65,
                      cmap=cmap,norm=norm,rasterized=True,linewidths=0)
    count=0
    if vectors is not None:
        chosen=choose_arrows(xy,vectors,mask,budget=max_arrows)
        shown=display_vectors(vectors[chosen],xy);count=len(chosen)
        if dim==2:
            # Static-only display scale makes shafts readable at publication size.
            shown = shown * 2
            ax.quiver(*xy[chosen].T,*shown.T,color='#1b2738',angles='xy',scale_units='xy',scale=1,
                      width=.0018,headwidth=3.8,headlength=5.2,zorder=4)
        else:
            ax.quiver(*xy[chosen].T,*shown.T,color='#14253c',normalize=False,length=1,
                      arrow_length_ratio=.34,linewidth=.65)
    if dim==2:
        ax.set_aspect('equal');ax.set(xticks=[],yticks=[],xlabel='Peak UMAP 1',ylabel='Peak UMAP 2')
    else:
        ax.set(xticks=[],yticks=[],zticks=[],xlabel='Peak UMAP 1',ylabel='Peak UMAP 2',zlabel='Peak UMAP 3')
        ax.view_init(elev=23,azim=38)
    ax.set_title(title+(f'\n{count} displayed directions' if vectors is not None else ''),fontsize=10)
    plt.colorbar(artist,ax=ax,shrink=.63,pad=.02,label=unit or ('Specificity / hpf' if rate else 'Specificity'))
    return count


def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',dpi=160,bbox_inches='tight')
    plt.close(fig)


def rate_limit(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    return max(float(np.quantile(np.abs(v),.98)),1e-5) if len(v) else 1.


def make_static_figures(a,grid,out):
    presentation=specificity_presentation(a)
    rng=np.random.default_rng(42)
    background=np.sort(rng.choice(a.n_obs,min(a.n_obs,120000),replace=False))
    times=np.asarray(grid['stages'])
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    for ax,method in zip(axes,['tau','entropy']):
        plot_map(ax,a,background,a.obsm[f'{method}_specificity'][:,-1],f'{method.capitalize()}: increasing restriction at 24 hpf',
                 a.obsm[f'{method}_up_umap2d'],a.obs[f'{method}_up_umap2d_display'].to_numpy())
    fig.suptitle(f"Structural reference paths · {presentation['title']} · existing 2D UMAP",fontsize=14)
    add_interpretation_banner(fig,presentation,structural=True)
    fig.text(.5,.01,'Fields computed on all 640,830 peaks; 120,000 points shown. Gray = insufficient count support. Arrow lengths are normalized.\n'+presentation['structural_note'],ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.91]);save(fig,out,'01_real_structural_arrows_2d')
    fig=plt.figure(figsize=(15,7))
    for i,method in enumerate(['tau','entropy']):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        plot_map(ax,a,background[::2],a.obsm[f'{method}_specificity'][:,-1],f'{method.capitalize()}: 24 hpf structural paths',
                 a.obsm[f'{method}_up_umap3d'],a.obs[f'{method}_up_umap3d_display'].to_numpy(),dim=3,max_arrows=180)
    fig.suptitle(f"Structural reference paths · {presentation['title']} · published 3D UMAP",fontsize=14)
    add_interpretation_banner(fig,presentation,structural=True)
    fig.text(.5,.01,'60,000 display points; model uses the full atlas. Increasing specificity is a chosen reference orientation.\n'+presentation['structural_note'],ha='center',fontsize=9)
    save(fig,out,'02_real_structural_arrows_3d')
    for method in ['tau','entropy']:
        rates=a.obsm[f'{method}_rate'];vectors=a.obsm[f'{method}_guided_umap2d'].reshape(a.n_obs,5,2)
        lim=rate_limit(rates);fig,axes=plt.subplots(2,3,figsize=(17,11))
        for t,ax in enumerate(axes.flat[:5]):
            v=vectors[:,t];mask=np.linalg.norm(v,axis=1)>0
            plot_map(ax,a,background,rates[:,t],f'{times[t]:g} → {times[t+1]:g} hpf',v,mask,rate=True,limit=lim,max_arrows=180)
        axes.flat[5].axis('off');axes.flat[5].text(.02,.85,
            presentation['panel_note']+'\n\nRed: increasing restriction\nBlue: decreasing restriction\nGray: insufficient count support\n\nDirections follow supported atlas paths.\nShared color range across all intervals.\nLengths are normalized, not physical speed.\n\nCelltype opening/closing is shown separately.\nNo biological confidence intervals inferred.',
            transform=axes.flat[5].transAxes,va='top',fontsize=11)
        fig.suptitle(f"{presentation['title']} · {method} change along atlas paths",fontsize=14)
        add_interpretation_banner(fig,presentation,y=.945)
        fig.tight_layout(rect=[0,0,1,.91]);save(fig,out,f'03_real_temporal_arrows_2d_{method}')
    selected_ct=['hematopoietic_vasculature','NMPs','enteric_neurons']
    selected_t=[0,3,4];opening=a.layers['accessibility_rate']
    fig,axes=plt.subplots(3,3,figsize=(16,15))
    for row,ct in enumerate(selected_ct):
        c=list(grid['celltypes']).index(ct)
        columns=[grid['column_index'][c,t] for t in range(5)]
        values=opening[:,[j for j in columns if j>=0]]
        lim=rate_limit(values)
        for col,t in enumerate(selected_t):
            source=grid['column_index'][c,t]
            v=opening[:,source] if source>=0 else np.full(a.n_obs,np.nan)
            plot_map(axes[row,col],a,background,v,f'{ct}\n{times[t]:g} → {times[t+1]:g} hpf',rate=True,limit=lim,unit='Relative accessibility (CPM / hpf)')
    fig.suptitle('Original measured CPM · within-celltype opening / closing',fontsize=14)
    fig.text(.5,.01,'Red = relative opening; blue = relative closing. Gray = absent/low-cell group or combined endpoint raw counts <20. NMPs at 24 hpf have 6 cells.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.03,1,.96]);save(fig,out,'04_real_celltype_opening_closing')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for method,color in [('tau','#087f8c'),('entropy','#b55b33')]:
        vals=np.isfinite(a.obsm[f'{method}_specificity']).mean(0)
        axes[0].plot(times,vals,'o-',label=method,color=color)
        v=a.obsm[f'{method}_guided_umap2d'].reshape(a.n_obs,5,2)
        axes[1].plot(times[:-1],np.mean(np.linalg.norm(v,axis=2)>0,axis=0),'o-',label=method,color=color)
        axes[2].hist(a.obs[f'{method}_up_umap2d_stability'],bins=45,range=(-1,1),histtype='step',density=True,label=method,color=color)
    axes[0].set(xlabel='Stage (hpf)',ylabel='Fraction of all peaks',title=presentation['support_title'],ylim=(0,1))
    axes[1].set(xlabel='Interval start (hpf)',ylabel='Fraction of all peaks',title='Temporal arrows passing display filters',ylim=(0,1))
    axes[2].set(xlabel='Mean cosine across parameter variants',ylabel='Density',title='Direction sensitivity (not confidence)')
    for ax in axes:ax.legend(frameon=False)
    fig.suptitle(presentation['title']+' · support and parameter sensitivity',fontsize=13)
    add_interpretation_banner(fig,presentation,y=.92)
    fig.tight_layout(rect=[0,0,1,.86]);save(fig,out,'05_real_support_and_sensitivity')


def make_viewer(a,grid,out):
    presentation=specificity_presentation(a)
    from real_specialization_viewer import export_real_viewer,select_display_peaks
    gene_targets=['slc4a1a','cdx1a','phox2bb','gata1a','tal1','sox10']
    include=np.zeros(a.n_obs,dtype=bool)
    for column in ['nearest_gene','linked_gene','associated_gene']:
        if column in a.obs:
            include |= a.obs[column].astype(str).str.contains('|'.join(r'\b'+g+r'\b' for g in gene_targets),regex=True).to_numpy()
    # Display sampling never changes the graph, rates, or projected directions.
    ids,selection=select_display_peaks(a.obsm['X_umap2d'],a.obsm['X_umap3d'],strata=a.obs['leiden_coarse'].astype(str).to_numpy(),
                                     include=np.flatnonzero(include),max_points=18000,seed=42)
    ids=np.asarray(ids,dtype=int)
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate display indices')
    opening=to_condition_cube(a.layers['accessibility_rate'][ids],grid)[:,:,:5]
    access=to_condition_cube(a.X[ids],grid)
    observed=np.asarray(a.uns['real_specialization']['condition_valid'],dtype=bool)
    # Preserve raw measured low-count curves in AnnData, but mask unsupported
    # pseudobulk groups in the viewer rather than present them as measured zeros.
    access[:,~observed]=np.nan
    methods={}
    for method in ['tau','entropy']:
        fields=[]
        structural=dict(id='structural',kind='structural',label='Increasing 24 hpf specificity',
            description='Existing atlas neighbors are oriented toward greater 24 hpf specificity across the same 21 celltypes. This is a chosen reference orientation among fixed peaks.')
        temporal=dict(id='temporal',kind='temporal',label='Temporal specificity change along paths',
            description=('Depth-matched' if presentation['estimator']=='depth_matched' else 'Depth-confounded raw')+' specificity rates select separate uphill or downhill neighbors. Spatial guidance is an explicit display assumption. Filters use pooled count support, effect size, coherence, and parameter sensitivity; they are not confidence intervals.')
        for dim in [2,3]:
            d=f'umap{dim}d';xy=a.obsm[f'X_{d}']
            up=a.obsm[f'{method}_up_{d}'][ids]
            structural[f'vectors{dim}d']=display_vectors(up,xy)
            for source,target in [('display','arrow_mask'),('coherence','coherence'),('stability','stability')]:
                structural[f'{target}{dim}d']=a.obs[f'{method}_up_{d}_{source}'].to_numpy()[ids]
            structural[f'support{dim}d']=a.obs[f'{method}_up_support'].to_numpy()[ids]
            guided=a.obsm[f'{method}_guided_{d}'][ids].reshape(len(ids),5,dim)
            temporal[f'vectors{dim}d']=display_vectors(guided,xy)
            temporal[f'arrow_mask{dim}d']=np.linalg.norm(guided,axis=2)>0
            positive=a.obsm[f'{method}_rate'][ids]>0
            for source,target in [('coherence','coherence'),('stability','stability')]:
                uv=a.obs[f'{method}_up_{d}_{source}'].to_numpy()[ids,None]
                dv=a.obs[f'{method}_down_{d}_{source}'].to_numpy()[ids,None]
                temporal[f'{target}{dim}d']=np.where(positive,uv,dv)
            temporal[f'support{dim}d']=np.where(positive,a.obs[f'{method}_up_support'].to_numpy()[ids,None],a.obs[f'{method}_down_support'].to_numpy()[ids,None])
        methods[method]=dict(label=method.capitalize(),description='Primary scores use the same 21 celltypes at all six stages. At least 20 pooled counts are required per endpoint. Both methods orient structural paths at 24 hpf. '+presentation['method_note'],
            specificity=a.obsm[f'{method}_specificity'][ids],specificity_rate=a.obsm[f'{method}_rate'][ids],fields=[structural,temporal],
            color_limits=dict(specificity_rate=rate_limit(a.obsm[f'{method}_rate'])))
    peaks=[]
    for i in ids:
        row=a.obs.iloc[i]
        peaks.append(dict(peak=str(a.obs_names[i]),global_index=int(i),display_forced=bool(include[i]),**{k:clean(v) for k,v in row.items() if k in ['nearest_gene','linked_gene','associated_gene','celltype','lineage','leiden_coarse','broad_reference']}))
    payload=dict(title='Real peak atlas · '+presentation['title'],
        interpretation_notice=presentation['notice'],specificity_estimator=presentation['estimator'],
        notes='All 640,830 peaks were analyzed using the published atlas neighbor graph and unchanged 2D/3D UMAPs. This browser shows a coverage-oriented subset, not an abundance sample. Scores and CPM rates come from the separate celltype × stage pseudobulk matrix. Relative accessibility changes are retrospective; no peak motion, ancestry, or biological confidence intervals are inferred. Broad-reference highlights are descriptive and do not force a root. Gray values are unsupported, not closed. '+presentation['method_note'],
        total_peaks=a.n_obs,analysis_peak_count=a.n_obs,celltypes=list(grid['celltypes']),stages=np.asarray(grid['stages']),
        peaks=peaks,umap2d=a.obsm['X_umap2d'][ids],umap3d=a.obsm['X_umap3d'][ids],accessibility=access.reshape(len(ids),-1),
        condition_n_cells=np.asarray(grid['n_cells']),condition_valid=observed,methods=methods,
        opening_rate=opening.reshape(len(ids),-1),color_limits=dict(opening_rate=rate_limit(a.layers['accessibility_rate']),accessibility=float(np.nanquantile(a.X,.99))),
        specificity_celltypes=list(a.uns['real_specialization']['common_celltypes']),
        default_method='entropy' if presentation['estimator']=='depth_matched' else 'tau',default_field='structural',default_dimension=2,
        default_interval=2,accessibility_units='CPM')
    export_real_viewer(payload,out/'real_specialization_explorer.html')
    pd.DataFrame({'display_row':np.arange(len(ids)),'global_index':ids,'peak':a.obs_names[ids]}).to_csv(out/'viewer_peak_index.csv',index=False)
    write_json(out/'viewer_selection.json',selection)
    # Validate actual export arrays before recording successful invariants.
    assert np.array_equal(payload['umap2d'], a.obsm['X_umap2d'][ids])
    assert np.array_equal(payload['umap3d'], a.obsm['X_umap3d'][ids])
    assert [p['peak'] for p in peaks] == list(a.obs_names[ids])
    for method in methods:
        assert np.allclose(methods[method]['specificity_rate'], a.obsm[f'{method}_rate'][ids], equal_nan=True)
        for field in methods[method]['fields']:
            for dim in [2,3]:
                lengths=np.linalg.norm(field[f'vectors{dim}d'],axis=-1)
                active=lengths>0
                expected=np.linalg.norm(np.ptp(a.obsm[f'X_umap{dim}d'],axis=0))*.008
                assert np.allclose(lengths[active],expected,rtol=1e-5,atol=1e-6)
                assert not np.any(np.asarray(field[f'arrow_mask{dim}d']) & ~active)
    assert np.isnan(access[:,~observed]).all()
    checks=dict(unique_display_indices=True,all_peak_model=True,n_model_peaks=a.n_obs,n_display_peaks=len(ids),
        display_coordinates_exact=True,display_vectors_normalized=True,condition_missing_count=int((~grid['present']).sum()),
        specificity_fixed_celltypes=len(payload['specificity_celltypes']),rate_units='CPM/hpf for opening; specificity/hpf for guided',
        no_biological_confidence_intervals=True,specificity_estimator=presentation['estimator'],
        measured_accessibility_and_opening_unchanged=True)
    write_json(out/'artifact_validation.json',checks)


def add_accessibility_summaries(a,grid,out):
    """Separate net accessibility change from the size of opposing CT changes."""
    from anndata.io import write_elem
    common=np.asarray(grid['present']) & (np.asarray(grid['n_cells'])>=20)
    common=common.all(axis=1)
    times=np.asarray(grid['stages']);dt=np.diff(times)
    mean_rate=np.full((a.n_obs,5),np.nan,dtype=np.float32)
    rms_rate=np.full((a.n_obs,5),np.nan,dtype=np.float32)
    raw=a.obsm['fixed_raw_total']
    for start in range(0,a.n_obs,512):
        stop=min(start+512,a.n_obs);sl=slice(start,stop)
        cube=to_condition_cube(a.X[sl],grid)[:,common,:]
        rate=np.diff(cube,axis=2)/dt
        supported=(raw[sl,:-1]+raw[sl,1:])>=20
        mean_rate[sl]=np.where(supported,np.mean(rate,axis=1),np.nan)
        rms_rate[sl]=np.where(supported,np.sqrt(np.mean(rate**2,axis=1)),np.nan)
    a.obsm['fixed_accessibility_mean_rate']=mean_rate
    a.obsm['fixed_accessibility_rms_rate']=rms_rate
    good=np.isfinite(mean_rate).all(axis=1)
    net=np.where(good,np.sum(mean_rate*dt,axis=1)/sum(dt),np.nan)
    magnitude=np.where(good,np.sum(rms_rate*dt,axis=1)/sum(dt),np.nan)
    a.obs['net_accessibility_rate']=net
    a.obs['rms_accessibility_rate']=magnitude
    a.uns['real_specialization']['accessibility_summary_definition']='Across fixed 21 celltypes: signed mean and RMS of CPM/hpf; combined pooled endpoint raw counts >=20. Whole-time summaries require all intervals and use duration weights. These are relative-accessibility magnitudes, not speed through UMAP.'
    rows=[]
    for cluster,ids in a.obs.groupby('leiden_coarse',observed=True).indices.items():
        for t in range(5):
            supported=np.isfinite(mean_rate[ids,t])
            rows.append(dict(leiden_coarse=str(cluster),start_hpf=times[t],end_hpf=times[t+1],
                n_peaks=len(ids),n_supported=int(supported.sum()),
                mean_net_cpm_per_hpf=float(np.mean(mean_rate[ids,t][supported])) if supported.any() else np.nan,
                mean_rms_cpm_per_hpf=float(np.mean(rms_rate[ids,t][supported])) if supported.any() else np.nan))
    pd.DataFrame(rows).to_csv(out/'cluster_accessibility_summary.csv',index=False)
    # Append only derived summaries to our new output, preserving its matrix and
    # the original source objects. AnnData's writer maintains H5AD encoding.
    with h5py.File(out/'real_specialization_flow.h5ad','r+') as f:
        for key in ['fixed_accessibility_mean_rate','fixed_accessibility_rms_rate']:
            write_elem(f['obsm'],key,a.obsm[key],dataset_kwargs={'compression':'gzip'})
        write_elem(f,'obs',a.obs,dataset_kwargs={'compression':'gzip'})
        write_elem(f['uns'],'real_specialization',a.uns['real_specialization'])
    rng=np.random.default_rng(42);background=np.sort(rng.choice(a.n_obs,min(a.n_obs,120000),replace=False))
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    plot_map(axes[0],a,background,net,'Net opening / closing over 10–24 hpf',rate=True,limit=rate_limit(net),unit='Mean CPM / hpf')
    xy=a.obsm['X_umap2d'];valid=np.isfinite(magnitude[background]);ids=background[valid]
    axes[1].scatter(*xy[background[~valid]].T,s=.45,color='#c7cbd1',alpha=.35,rasterized=True,linewidths=0)
    limit=rate_limit(magnitude)
    artist=axes[1].scatter(*xy[ids].T,c=magnitude[ids],s=.55,cmap='magma',vmin=0,vmax=limit,alpha=.65,rasterized=True,linewidths=0)
    axes[1].set(aspect='equal',xticks=[],yticks=[],xlabel='Peak UMAP 1',ylabel='Peak UMAP 2',title='Magnitude of accessibility change across celltypes')
    plt.colorbar(artist,ax=axes[1],shrink=.63,label='RMS CPM / hpf')
    fig.suptitle('Original measured CPM · signed net change and change magnitude',fontsize=14)
    fig.text(.5,.01,'Duration-weighted summaries across the fixed 21-celltype panel. Opposing changes can cancel in the signed mean; RMS retains them. Gray = unsupported.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.95]);save(fig,out,'06_real_accessibility_magnitude_and_net_change')


def render_report(a,grid,out):
    print('Adding accessibility magnitude and net-change summaries',flush=True)
    add_accessibility_summaries(a,grid,out)
    print('Rendering real-data figures',flush=True)
    make_static_figures(a,grid,out)
    print('Exporting offline real-data explorer',flush=True)
    make_viewer(a,grid,out)
