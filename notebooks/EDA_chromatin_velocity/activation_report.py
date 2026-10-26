"""Celltype × stage accessibility maps on a fixed peak atlas.

Display-only sampling; all numeric regional summaries are supplied from the full
atlas. No velocity is inferred from a celltype scalar, no new UMAP is fitted,
and graph distance is never labelled developmental pseudotime.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from real_specialization_viewer import select_display_peaks
from trajectory_report import _take

HERE = Path(__file__).resolve().parent
DEFAULT_FITS = HERE / 'outputs/trajectory_velocity_v1/trajectory_fits.h5'
DEFAULT_OUT = HERE / 'outputs/activation_progression_v1'


def decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def finite_limit(values, q=.99, floor=1e-4):
    finite = np.asarray(values)[np.isfinite(values)]
    return max(float(np.quantile(finite, q)), floor) if len(finite) else 1.


def save(fig, out, stem, caption, dpi=170):
    fig.text(.5, .01, caption, ha='center', va='bottom', fontsize=8, color='#485869')
    fig.savefig(out / (stem+'.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(out / (stem+'.pdf'), dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return dict(stem=stem, png=stem+'.png', pdf=stem+'.pdf', caption=caption)


def map_points(ax, xy, values, title, *, limit, rate=False, root=None, size=.52):
    finite = np.isfinite(values)
    ax.scatter(*xy[~finite].T, c='#c8cdd2', s=size, alpha=.40, linewidths=0, rasterized=True)
    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit) if rate else Normalize(0, limit)
    a = ax.scatter(*xy[finite].T, c=values[finite], s=(max(size, .95) if rate else size), alpha=.85, linewidths=0,
                   cmap='RdBu_r' if rate else 'viridis', norm=norm, rasterized=True)
    if root is not None:
        r = np.flatnonzero(root)
        # Keep contours legible; this subset denotes location, not root abundance.
        r = r[np.linspace(0, len(r)-1, min(180, len(r))).astype(int)] if len(r) else r
        ax.scatter(*xy[r].T, s=8, facecolors='none', edgecolors='#dc9e2b', linewidths=.55, zorder=3, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set(xticks=[], yticks=[], xlabel='', ylabel='')
    ax.set_aspect('equal')
    return a


def sheet(data, c, out):
    times, ct, xy = data['stages'], data['celltypes'][c], data['xy']
    states = np.log1p(data['fitted'][:, c]).copy()
    states[:, ~data['condition'][c]] = np.nan
    # Lack-of-fit does not erase measured raw curves but excludes misleading fit maps.
    good_fit = data['lof'][:, c] <= 4
    states[~good_fit] = np.nan
    rates = data['rates'][:, c].copy()
    rates[~data['supported'][:, c]] = np.nan
    state_lim = finite_limit(states)
    rate_lim = finite_limit(np.abs(rates), .98, .05)
    fig, axes = plt.subplots(2, 6, figsize=(20, 7.5))
    for t in range(6):
        condition = data['condition'][c, t]
        title = f'{times[t]:g} hpf' + ('' if condition else '\nmissing / <20 cells')
        a = map_points(axes[0, t], xy, states[:, t], title, limit=state_lim)
    for t in range(5):
        a_rate = map_points(axes[1, t], xy, rates[:, t], f'{times[t]:g} → {times[t+1]:g} hpf', limit=rate_lim, rate=True)
    axes[1, 5].axis('off')
    axes[1, 5].text(.03, .97, 'Top: fitted accessibility\nBottom: supported change\n\nRed: opening\nBlue: closing\nGray: missing / unsupported\n\nGray does not mean closed.\nFixed coordinates; each dot is a peak.\nColor scales shared across time\nwithin this celltype.', transform=axes[1, 5].transAxes, va='top', fontsize=9)
    cax = fig.add_axes([.17, .07, .24, .016]); fig.colorbar(a, cax=cax, orientation='horizontal', label='Fitted log(1 + CPM)')
    cax = fig.add_axes([.55, .07, .24, .016]); fig.colorbar(a_rate, cax=cax, orientation='horizontal', label='Supported fitted CPM / hpf')
    fig.suptitle(ct.replace('_',' ')+' · accessibility through development', fontsize=16)
    fig.subplots_adjust(top=.88, bottom=.15, wspace=.08, hspace=.24)
    stem = f'celltypes/{c:02d}_{re.sub(r"[^A-Za-z0-9_-]", "_", ct)}'
    item = save(fig, out, stem, f'{data["total"]:,} peaks fitted; {len(xy):,} displayed. Relative CPM, not absolute accessibility. Conditional count-noise support; no biological replicate confidence.')
    item.update(celltype=ct, state_limit=state_lim, rate_limit=rate_lim)
    return item


def overview(data, out):
    chosen = [ct for ct in ['NMPs','neural','hematopoietic_vasculature','enteric_neurons'] if ct in data['celltypes']]
    fig, axes = plt.subplots(len(chosen), 6, figsize=(20, 3.1*len(chosen)), squeeze=False)
    color_arts=[]
    for row, ct in enumerate(chosen):
        c = list(data['celltypes']).index(ct)
        values = np.log1p(data['fitted'][:, c]).copy()
        values[:, ~data['condition'][c]] = np.nan
        values[data['lof'][:, c] > 4] = np.nan
        limit = finite_limit(values)
        for t in range(6):
            art = map_points(axes[row,t], data['xy'], values[:,t], f'{data["stages"][t]:g} hpf'+('' if data['condition'][c,t] else ' · unavailable'), limit=limit)
        axes[row,0].set_ylabel(ct.replace('_',' '), fontsize=10, labelpad=10)
        color_arts.append(art)
    fig.suptitle('Where accessibility is used · celltype × developmental stage', fontsize=17, y=.99)
    fig.subplots_adjust(top=.94, bottom=.05, right=.90, wspace=.08, hspace=.22)
    for row,art in enumerate(color_arts):
        pos=axes[row,-1].get_position();cax=fig.add_axes([.92,pos.y0+.01,.009,pos.height-.02]);fig.colorbar(art,cax=cax,label='log(1 + CPM)')
    return save(fig, out, '01_celltype_stage_accessibility', 'Four illustrative celltypes; complete sheets provided for all 32. Each row shares a time-invariant scale; scales differ between celltypes. Gray = unavailable / fit flagged. Each point is a genomic peak.')


def rate_overview(data, out):
    chosen = [ct for ct in ['NMPs','neural','hematopoietic_vasculature','enteric_neurons'] if ct in data['celltypes']]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(18, 3.0*len(chosen)), squeeze=False)
    color_arts=[]
    for row, ct in enumerate(chosen):
        c = list(data['celltypes']).index(ct)
        values = data['rates'][:, c].copy(); values[~data['supported'][:,c]] = np.nan
        limit = finite_limit(np.abs(values), .98, .05)
        for t in range(5):
            art = map_points(axes[row,t], data['xy'], values[:,t], f'{data["stages"][t]:g} → {data["stages"][t+1]:g} hpf', limit=limit, rate=True)
        axes[row,0].set_ylabel(ct.replace('_',' '), fontsize=10, labelpad=10)
        color_arts.append(art)
    fig.suptitle('Where accessibility changes · supported opening and closing', fontsize=17, y=.99)
    fig.subplots_adjust(top=.94, bottom=.05, right=.90, wspace=.08, hspace=.22)
    for row,art in enumerate(color_arts):
        pos=axes[row,-1].get_position();cax=fig.add_axes([.92,pos.y0+.01,.009,pos.height-.02]);fig.colorbar(art,cax=cax,label='CPM / hpf')
    return save(fig, out, '02_celltype_interval_changes', 'Red = opening; blue = closing; gray = no supported rate / missing. No support is not evidence of stability. Each row shares its rate scale across intervals. No peak movement inferred.')


def geometry_figure(data, geometry, out):
    distance = geometry.get('graph_distance')
    if distance is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.1))
    vals = distance[data['ids']].copy()
    vals[~np.isfinite(vals)] = np.nan
    a = map_points(axes[0], data['xy'], vals, 'Distance from broad-accessibility reference', limit=finite_limit(vals, 1.), root=data['root'], size=.8)
    fig.colorbar(a, ax=axes[0], fraction=.04, label='Graph hops to nearest broad-reference peak')
    groups = data['metadata']['leiden_coarse'].astype(str).to_numpy()[data['ids']]
    labels = sorted(np.unique(groups), key=lambda x: int(x) if x.isdigit() else x)
    palette = plt.get_cmap('turbo')
    for j, group in enumerate(labels):
        use = groups == group
        axes[1].scatter(*data['xy'][use].T, s=.6, c=[palette(j/max(1,len(labels)-1))], linewidths=0, rasterized=True)
        center = np.median(data['xy'][use], axis=0)
        axes[1].text(*center, group, fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=.7, edgecolor='none', pad=1))
    axes[1].set(title='Existing coarse Leiden regions', xticks=[], yticks=[]); axes[1].set_aspect('equal')
    fig.suptitle('Atlas geometry is a reference for testing temporal organization', fontsize=15)
    fig.subplots_adjust(top=.88, bottom=.13, wspace=.2)
    return save(fig, out, '03_reference_and_atlas_regions', 'Gold rings locate broad-reference peaks. Graph distance is not developmental pseudotime. Coarse Leiden regions are atlas neighborhoods, not validated branchlets or lineages.')


def load_geometry(path, n, peak_ids):
    if path is None or not Path(path).exists():
        return {}
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            arrays = {k:f[k] for k in f.files}
    else:
        with h5py.File(path) as f:
            arrays = {k:f[k][:] for k in f if isinstance(f[k], h5py.Dataset)}
    if 'peak_ids' in arrays and not np.array_equal(decode(arrays['peak_ids']), peak_ids):
        raise ValueError('Geometry and fit peak ordering differ')
    for alias in ['distance','distance_from_reference','distance_hops','root_distance']:
        if 'graph_distance' not in arrays and alias in arrays:
            arrays['graph_distance'] = arrays[alias]
    if 'graph_distance' in arrays and arrays['graph_distance'].shape != (n,):
        raise ValueError('Graph distance has wrong shape')
    return arrays


def encode_array(a):
    a = np.asarray(a)
    dtype = 'uint8' if a.dtype.kind in 'bu' and a.dtype.itemsize == 1 else 'float32'
    b = np.ascontiguousarray(a, dtype=dtype)
    return dict(dtype=dtype, shape=list(b.shape), gzip_base64=base64.b64encode(gzip.compress(b.tobytes(), compresslevel=6, mtime=0)).decode())


def export_viewer(f, metadata, geometry, out, count=12000):
    from plotly.offline import get_plotlyjs
    n = len(f['peak_ids']); peak_ids = decode(f['peak_ids'][:]); celltypes = decode(f['celltypes'][:])
    xy, xyz = f['umap2d'][:], f['umap3d'][:]
    forced = np.zeros(n, bool)
    for col in ['nearest_gene','linked_gene','associated_gene']:
        if col in metadata:
            forced |= metadata[col].fillna('').astype(str).str.contains(r'\b(?:slc4a1a|cdx1a|phox2bb|gata1a|tal1|sox10)\b', regex=True).to_numpy()
    ids, selection = select_display_peaks(xy, xyz, strata=metadata['leiden_coarse'].astype(str), include=np.flatnonzero(forced), max_points=count, seed=1729)
    arrays = dict(xy=xy[ids], xyz=xyz[ids], observed=_take(f['observed_cpm'],ids), fitted=_take(f['fitted_cpm'],ids),
                  support=_take(f['rate_supported'],ids), mode=_take(f['modes'],ids).astype(np.uint8), fitvalid=_take(f['fit_valid'],ids),
                  lof=_take(f['lack_of_fit'],ids), condition=f['condition_valid'][:], root=f['broad_reference'][:][ids])
    if 'graph_distance' in geometry:
        dist = geometry['graph_distance'][ids].astype(float); dist[~np.isfinite(dist)] = np.nan; arrays['distance'] = dist
    peaks = []
    for i in ids:
        p = dict(peak=peak_ids[i], global_index=int(i))
        for key in ['nearest_gene','linked_gene','associated_gene','leiden_coarse','peak_type_argelaguet']:
            if key in metadata:
                p[key] = '' if pd.isna(metadata.iloc[i][key]) else str(metadata.iloc[i][key])
        peaks.append(p)
    payload = dict(total_peaks=n, display_peaks=len(ids), celltypes=celltypes.tolist(), stages=f['stages'][:].tolist(), peaks=peaks,
                   mode_labels=json.loads(f.attrs['mode_labels_json']), selection=selection,
                   arrays={k:encode_array(v) for k,v in arrays.items()})
    body = VIEWER_HTML.replace('__PLOTLY__',get_plotlyjs()).replace('__PAYLOAD__',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
    path = out / 'activation_peak_explorer.html'; path.write_text(body)
    np.save(out / 'viewer_display_indices.npy', ids)
    return dict(filename=path.name, bytes=path.stat().st_size, selection=selection)

def global_progression_figure(out):
    out=Path(out)
    table=pd.read_csv(out/'celltype_distance_time.csv')
    sub=table[(table.metric=='hops') & table.primary_panel.astype(bool)]
    pivot=sub.pivot(index='celltype',columns='stage',values='observed_distance_center')
    times=pivot.columns.to_numpy(float);values=pivot.to_numpy(float);mean=values.mean(axis=0)
    tests=pd.read_csv(out/'outward_tests.csv')
    test=tests[(tests.scope=='global')&(tests.metric=='hops')&(tests.estimator=='observed')].iloc[0]
    with np.load(out/'temporal_order_nulls.npz') as f:null=f['null_global_mean_slope']
    fig,axes=plt.subplots(1,3,figsize=(18,5.5))
    axes[0].plot(times,mean,'o-',color='#216b8b',lw=2,ms=6)
    axes[0].set(title='Global allocation across the fixed atlas',xlabel='Stage (hpf)',ylabel='Mean of celltype accessibility-weighted graph distance',xticks=times)
    axes[0].text(.04,.95,f'21 celltypes · equal weight\nEndpoint shift: +{test.endpoint_shift:.3f} hops',transform=axes[0].transAxes,va='top',fontsize=10)
    delta=values-values[:,[0]]
    for v in delta:axes[1].plot(times,v,color='#c0c8ce',lw=.7,alpha=.6)
    colors=['#b45d32','#216b8b','#786393','#448b68']
    for ct,color in zip(['enteric_neurons','hematopoietic_vasculature','floor_plate','endoderm'],colors):
        i=list(pivot.index).index(ct);axes[1].plot(times,delta[i],'o-',color=color,lw=1.8,ms=4,label=ct.replace('_',' '))
    axes[1].axhline(0,color='#81909b',lw=.7);axes[1].set(title='Celltypes follow different schedules',xlabel='Stage (hpf)',ylabel='Change from the same celltype at 10 hpf (hops)',xticks=times)
    axes[1].legend(fontsize=8,loc='upper left')
    axes[2].hist(null,bins=30,color='#b5c4ce',edgecolor='white')
    axes[2].axvline(test.slope,color='#aa4936',lw=2,label='Observed stage order')
    axes[2].set(title='Temporal-order reference · shared across all peaks',xlabel='Global slope (graph hops / hpf)',ylabel='Number of stage permutations')
    axes[2].text(.03,.95,f'All 720 stage orders\nOutward tail fraction: {test.p_outward:.3f}',transform=axes[2].transAxes,va='top',fontsize=10)
    axes[2].legend(fontsize=9,loc='lower right')
    for ax in axes[:2]:ax.grid(alpha=.15)
    fig.suptitle('Outward allocation is celltype-specific; a coherent global stream remains unproven',fontsize=15,y=.99)
    fig.subplots_adjust(left=.055,right=.985,top=.87,bottom=.18,wspace=.30)
    return save(fig,out,'08_global_ordering_controls','Observed CPM weights on a fixed peak panel and original graph. Global center is the equal mean of 21 celltype centers. Shared time-order controls preserve atlas geometry; they are not biological replicate tests.')

def global_observed_map(fits, out):
    out=Path(out)
    with h5py.File(fits) as f:
        ids=np.load(out/'figure_display_indices.npy')
        panel=f['common_celltypes'][:].astype(bool)
        observed=_take(f['observed_cpm'],ids)[:,panel]
        if not np.isfinite(observed).all():
            raise ValueError('Complete-panel observed accessibility unexpectedly has missing values')
        state=np.log1p(observed.mean(axis=1));xy=f['umap2d'][:][ids]
        times=f['stages'][:];root=f['broad_reference'][:][ids]
        fig,axes=plt.subplots(2,3,figsize=(15,10))
        limit=finite_limit(state)
        for t,ax in enumerate(axes.flat):
            art=map_points(ax,xy,state[:,t],f'{times[t]:g} hpf',limit=limit,root=root,size=.7)
        fig.subplots_adjust(top=.90,bottom=.10,left=.04,right=.90,hspace=.18,wspace=.10)
        cax=fig.add_axes([.93,.23,.014,.52]);fig.colorbar(art,cax=cax,label='log(1 + mean observed CPM)')
        fig.suptitle(f'Global observed accessibility · fixed {panel.sum()}-celltype panel',fontsize=16,y=.97)
        item=save(fig,out,'07_global_observed_stage_maps',f'Equal mean over {panel.sum()} celltypes with ≥20 cells at every stage; {len(ids):,} displayed peaks. Gold rings: broad-reference location. Shared temporal scale. Relative abundance, not absolute opening.')
        item['celltypes']=decode(f['celltypes'][:])[panel].tolist()
        (out/'global_observed_figure_manifest.json').write_text(json.dumps(item,indent=2))
        return item

def summary_figures(out):
    """Render full-atlas summaries supplied by the analysis, never display samples."""
    out=Path(out); items=[]
    path=out/'region_time_summary.csv'
    if path.exists():
        table=pd.read_csv(path)
        table['celltype']=table['celltype'].fillna('global')
        cts=[c for c in ['NMPs','neural','hematopoietic_vasculature','enteric_neurons'] if c in set(table.celltype)]
        region_order=None
        geo=out/'geometry_regions.csv'
        if geo.exists():
            g=pd.read_csv(geo)
            for col in ['median_distance_hops','distance_hops_median','median_hops','mean_distance_hops']:
                if col in g and 'region_code' in g:
                    region_order=g.sort_values(col).region_code.tolist();break
        fig,axes=plt.subplots(1,len(cts),figsize=(4.3*len(cts),11),squeeze=False)
        for ax,ct in zip(axes[0],cts):
            sub=table[table.celltype==ct].copy()
            sub.loc[~sub.condition_valid.astype(bool),'fitted_mean_cpm']=np.nan
            pivot=sub.pivot(index='region_code',columns='stage',values='fitted_mean_cpm')
            if region_order is not None:pivot=pivot.reindex([r for r in region_order if r in pivot.index])
            values=pivot.to_numpy(float)
            denom=np.nanmax(values,axis=1,keepdims=True);values=values/np.maximum(denom,1e-9)
            art=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
            ax.set_facecolor('#c8cdd2');ax.set_xticks(range(len(pivot.columns)),[f'{t:g}' for t in pivot.columns]);labels=sub.drop_duplicates('region_code').set_index('region_code').region_label.to_dict();ax.set_yticks(range(len(pivot)),[labels[r] for r in pivot.index])
            ax.set(title=ct.replace('_',' '),xlabel='Stage (hpf)',ylabel='Coarse atlas region')
            ax.tick_params(axis='y',labelsize=8)

        fig.suptitle('When existing atlas regions are used · celltype-specific schedules',fontsize=16,y=.99)
        fig.subplots_adjust(left=.045,right=.88,bottom=.07,top=.94,wspace=.27)
        cax=fig.add_axes([.91,.25,.012,.48]);fig.colorbar(art,cax=cax,label='Mean CPM / maximum across stages in the same region')
        items.append(save(fig,out,'04_region_accessibility_schedules','Full-atlas region means; each region is normalized to its own temporal maximum. This shows timing, not absolute activity differences between regions. Gray = missing / ineligible. Coarse regions are not validated branches.'))
    path=out/'celltype_interval_summary.csv'
    if path.exists():
        table=pd.read_csv(path)
        cts=list(dict.fromkeys(table.celltype));starts=sorted(table.start_hpf.unique())
        fig,axes=plt.subplots(1,2,figsize=(13,12))
        arts=[]
        for ax,key,title in zip(axes,['supported_opening','supported_closing'],['Supported opening','Supported closing']):
            sub=table.copy();sub['fraction']=sub[key]/sub['n_peaks']
            sub.loc[~sub.valid_interval.astype(bool),'fraction']=np.nan
            pivot=sub.pivot(index='celltype',columns='start_hpf',values='fraction').reindex(cts)
            # Shared scale makes opening and closing directly comparable.
            vmax=max(.01,float(np.nanquantile(np.r_[table.supported_opening/table.n_peaks,table.supported_closing/table.n_peaks],.99)))
            a=ax.imshow(np.ma.masked_invalid(pivot.to_numpy()),aspect='auto',cmap='magma',vmin=0,vmax=vmax,interpolation='nearest');arts.append(a)
            ends={r.start_hpf:r.end_hpf for r in table.itertuples()}
            ax.set_xticks(range(len(starts)),[f'{t:g}→{ends[t]:g}' for t in starts],rotation=30,ha='right');ax.set_yticks(range(len(cts)),[c.replace('_',' ') for c in cts]);ax.tick_params(axis='y',labelsize=8)
            ax.set(title=title,xlabel='Developmental interval (hpf)');ax.set_facecolor('#c8cdd2')

        fig.suptitle('Opening and closing schedules across celltypes',fontsize=16,y=.98)
        fig.subplots_adjust(left=.19,right=.87,top=.94,bottom=.08,wspace=.6)
        cax=fig.add_axes([.91,.23,.016,.52]);fig.colorbar(arts[0],cax=cax,label='Fraction of eligible fitted peaks with supported change')
        items.append(save(fig,out,'05_celltype_opening_closing_schedule','Full-atlas counts / eligible fitted peaks; shared color scale. Gray = unavailable interval. Conditional count-noise support can vary with coverage; absence of support does not prove absence of biological change.'))
    path=out/'outward_tests.csv'
    if path.exists():
        table=pd.read_csv(path)
        sub=table[(table.scope=='celltype') & (table.estimator=='observed')].copy()
        if len(sub):
            hops=sub[sub.metric=='hops'].set_index('celltype')
            weighted=sub[sub.metric=='weighted'].set_index('celltype')
            names=hops.sort_values('correlation').index.tolist()
            fig,ax=plt.subplots(figsize=(11,11))
            for y,ct in enumerate(names):
                a=hops.loc[ct,'correlation'];b=weighted.loc[ct,'correlation'] if ct in weighted.index else np.nan
                ax.plot([a,b],[y,y],color='#b9c4cd',lw=1)
                ax.scatter(a,y,c='#216b8b',s=27,zorder=3)
                if np.isfinite(b):ax.scatter(b,y,c='#bb7134',s=25,marker='s',zorder=3)
            ax.axvline(0,color='#7c8c99',lw=.8);ax.set_yticks(range(len(names)),[c.replace('_',' ') for c in names]);ax.tick_params(axis='y',labelsize=9)
            ax.set(xlabel='Correlation of developmental time with accessibility-weighted graph distance',title='Does accessibility shift outward from the broad reference?')
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([],[],color='#216b8b',marker='o',linestyle='',label='Graph hops'),Line2D([],[],color='#bb7134',marker='s',linestyle='',label='Weighted graph distance')],loc='lower right')
            ax.grid(axis='x',alpha=.15);fig.subplots_adjust(left=.28,bottom=.09,right=.96,top=.94)
            items.append(save(fig,out,'06_celltype_outward_progression','Full-atlas observed accessibility weights. Positive = later stages use peaks farther from the reference; negative = closer. Distance-metric sensitivity is shown. Significance and temporal-order controls are in outward_tests.csv.'))
    if (out/'celltype_distance_time.csv').exists() and (out/'temporal_order_nulls.npz').exists():
        items.append(global_progression_figure(out))
    (out/'activation_summary_figures_manifest.json').write_text(json.dumps(items,indent=2))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fits', type=Path, default=DEFAULT_FITS)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--geometry', type=Path)
    p.add_argument('--display-points', type=int, default=70000)
    p.add_argument('--viewer-points', type=int, default=12000)
    p.add_argument('--only-viewer', action='store_true')
    p.add_argument('--summaries-only', action='store_true')
    p.add_argument('--global-only', action='store_true')
    p.add_argument('--skip-viewer', action='store_true')
    args = p.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True); (out/'celltypes').mkdir(exist_ok=True)
    if args.global_only:
        print(json.dumps(global_observed_map(args.fits,out)),flush=True); return
    if args.summaries_only:
        print(json.dumps(summary_figures(out)),flush=True); return
    metadata = pd.read_csv(args.fits.parent/'peak_metadata.csv.gz', index_col=0)
    with h5py.File(args.fits) as f:
        peaks=decode(f['peak_ids'][:]); n=len(peaks)
        if not np.array_equal(metadata.index.astype(str).to_numpy(),peaks):
            raise ValueError('Metadata and fit peak ordering differ')
        geometry=load_geometry(args.geometry,n,peaks)
        manifest=dict(source_fits=str(args.fits.resolve()),geometry=str(args.geometry) if args.geometry else None,figures=[])
        if not args.only_viewer:
            ids,selection=select_display_peaks(f['umap2d'][:],f['umap3d'][:],strata=metadata['leiden_coarse'].astype(str),max_points=args.display_points,seed=1729)
            data=dict(ids=ids,total=n,xy=f['umap2d'][:][ids],stages=f['stages'][:],celltypes=decode(f['celltypes'][:]),
                      fitted=_take(f['fitted_cpm'],ids),rates=_take(f['rates'],ids),supported=_take(f['rate_supported'],ids),
                      condition=f['condition_valid'][:],lof=_take(f['lack_of_fit'],ids),metadata=metadata,root=f['broad_reference'][:][ids])
            manifest['selection']=selection; manifest['figures'] += [overview(data,out),rate_overview(data,out)]
            geoitem=geometry_figure(data,geometry,out)
            if geoitem: manifest['figures'].append(geoitem)
            for c in range(len(data['celltypes'])):
                manifest['figures'].append(sheet(data,c,out)); print('Rendered',data['celltypes'][c],flush=True)
            np.save(out/'figure_display_indices.npy',ids)
        if not args.skip_viewer:
            manifest['viewer']=export_viewer(f,metadata,geometry,out,args.viewer_points)
        manifest_path=out/('activation_viewer_manifest.json' if args.only_viewer else 'activation_report_manifest.json')
        manifest_path.write_text(json.dumps(manifest,indent=2))
        print(json.dumps(manifest.get('viewer',{})),flush=True)


VIEWER_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chromatin program usage through development</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#203246;font:14px system-ui,sans-serif}header{padding:20px 26px;background:white;border-bottom:1px solid #d7e0e8}h1{font-size:25px;margin:0 0 7px}p{margin:7px 0;max-width:145ch}.notice{font-size:12px;color:#655127}.controls{display:flex;gap:13px;flex-wrap:wrap;padding:14px 23px;background:#f8fafc}.controls label{display:flex;flex-direction:column;gap:4px;font-size:12px;font-weight:600}select,input,button{padding:6px 8px;border:1px solid #bcc9d5;border-radius:5px;background:white;font:inherit}button{align-self:end;cursor:pointer}label.check{flex-direction:row;align-items:center;gap:5px;align-self:end;padding-bottom:8px}.layout{display:grid;grid-template-columns:minmax(630px,1.65fr) minmax(350px,1fr);gap:14px;padding:12px 20px}.card{border:1px solid #d8e1e9;border-radius:8px;background:white;overflow:hidden}#atlas{height:720px}#curve{height:340px}#heatmap{height:350px}#status,#error{padding:8px 24px;font-size:13px}#error{color:#a21d1d}.caption,.selected{padding:11px 16px;font-size:12px;line-height:1.5;overflow-wrap:anywhere}.selected{font-weight:600}details{padding:13px 24px}summary{cursor:pointer;font-weight:600}#search{width:210px}@media(max-width:1030px){.layout{display:block}.card{margin-bottom:14px}#atlas{height:630px}}
</style></head><body><header><h1>Chromatin program usage through development</h1><p>Each point is a fixed genomic peak. Select a celltype and stage to see where accessibility is used, or an interval to see supported opening and closing.</p><p class="notice">CPM is relative accessibility. Gray does not mean closed: it denotes missing, low-cell, or unsupported estimates. Broad-reference peaks mark atlas geometry; graph distance is not developmental pseudotime. No peak transport or scalar-derived arrows are shown.</p></header>
<div class="controls"><label>Dimension<select id="dimension"><option value="2">2D</option><option value="3">3D · rotate</option></select></label><label>Celltype<select id="celltype"></select></label><label>Map<select id="map"><option value="fitted">Fitted accessibility</option><option value="observed">Observed accessibility</option><option value="rate">Fitted opening / closing rate</option><option value="distance">Distance from broad reference</option><option value="mode">Trajectory mode</option></select></label><label>Stage<select id="stage"></select></label><label>Interval<select id="interval"></select></label><label class="check"><input id="supported" type="checkbox" checked>Supported changes only</label><label class="check"><input id="root" type="checkbox">Broad-reference overlay</label><label>Find exported peak / gene<input id="search" placeholder="slc4a1a or peak coordinate"></label><button id="find">Find</button><button id="save">Save atlas PNG</button></div><div id="status">Loading embedded scientific arrays…</div><div id="error"></div><div class="layout"><section class="card"><div id="atlas"></div><div class="caption" id="caption"></div></section><section class="card"><div class="selected" id="selected"></div><div id="curve"></div><div id="heatmap"></div><div class="caption">Measured values remain available in curves, including low-cell observations. Fits connect measured stage estimates; six snapshots do not resolve narrow kinetics. Mode labels and rate support use conditional count-noise diagnostics, not biological replicate confidence.</div></section></div><details><summary>Methods and interpretation</summary><p>Fits use all available eligible stages at 10, 12, 14, 16, 19 and 24 hpf. Missing celltype–stage combinations are never filled with zero or bridged for interval rates. Rates are differences of fitted CPM divided by elapsed hours. Limited candidate models favor broad patterns and may miss true narrow pulses.</p><p>The searchable subset preserves spatial coverage and marker peaks; its density is not an abundance estimate. All numerical analyses use the full atlas. Existing coarse Leiden groups are atlas neighborhoods, not validated branchlets. Selecting a celltype changes the displayed accessibility data itself.</p><p>Default rate colors require stored support. Unchecking the support box shows all estimable rates from fits with sufficient observations and acceptable lack-of-fit, including low-count or unsupported changes; this is exploratory. State maps retain low-count estimates where fit quality is acceptable. Raw low-cell observations are shown in the curve but withheld from the atlas map.</p></details>
<script>__PLOTLY__</script><script>
const DATA=__PAYLOAD__,A={},N=DATA.display_peaks,C=DATA.celltypes.length,T=DATA.stages.length;let selected=0,busy=false,pending=false,lastSearch='',lastHit=-1;const $=id=>document.getElementById(id),finite=Number.isFinite,modeColors=['#c8cdd2','#69737e','#c55e2d','#2676a8','#8d5ca6','#238f85','#b7a8c5'];
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function options(id,vals){$(id).innerHTML=vals.map(([v,s])=>`<option value="${v}">${esc(s)}</option>`).join('')}
function state(){return {d:+$('dimension').value,c:+$('celltype').value,t:+$('stage').value,k:+$('interval').value,map:$('map').value,support:$('supported').checked,root:$('root').checked}}function curve(a,i,c){return Array.from(a.subarray((i*C+c)*T,(i*C+c+1)*T)).map(v=>finite(v)?v:null)}function xy(s,i){const a=s.d===3?A.xyz:A.xy;return Array.from(a.subarray(i*s.d,(i+1)*s.d))}
const limitCache={};function quantile(v,q,floor){v=v.filter(finite).sort((a,b)=>a-b);return Math.max(v.length?v[Math.floor(q*(v.length-1))]:1,floor)}
function limit(s){const key=s.map+':'+s.c;if(key in limitCache)return limitCache[key];let v=[];if(s.map==='distance')v=Array.from(A.distance||[]);else if(s.map==='rate'){for(let i=0;i<N;i++)for(let t=0;t<T-1;t++){let j=(i*C+s.c)*T+t,k=(i*C+s.c)*(T-1)+t;if(A.support[k])v.push(Math.abs((A.fitted[j+1]-A.fitted[j])/(DATA.stages[t+1]-DATA.stages[t])))}}else{const a=s.map==='observed'?A.observed:A.fitted;for(let i=0;i<N;i++)if(s.map==='observed'||A.lof[i*C+s.c]<=4)for(let t=0;t<T;t++)if(A.condition[s.c*T+t])v.push(Math.log1p(a[(i*C+s.c)*T+t]))}return limitCache[key]=quantile(v,s.map==='rate'?.98:.99,s.map==='rate'?.05:.0001)}
function values(s){let vals=new Array(N);for(let i=0;i<N;i++){const j=(i*C+s.c)*T+s.t;if(s.map==='distance'){vals[i]=A.distance?A.distance[i]:NaN;continue}if(s.map==='mode'){vals[i]=A.mode[i*C+s.c];continue}if(s.map==='rate'){const q=(i*C+s.c)*T+s.k,r=(i*C+s.c)*(T-1)+s.k;vals[i]=A.condition[s.c*T+s.k]&&A.condition[s.c*T+s.k+1]&&A.lof[i*C+s.c]<=4&&(!s.support||A.support[r])?(A.fitted[q+1]-A.fitted[q])/(DATA.stages[s.k+1]-DATA.stages[s.k]):NaN;continue}vals[i]=A.condition[s.c*T+s.t]&&(s.map==='observed'||A.lof[i*C+s.c]<=4)?Math.log1p((s.map==='observed'?A.observed:A.fitted)[j]):NaN}return vals}
function background(s){if(s.d===2)return Array.from({length:N},(_,i)=>i);const ids=[];for(let i=0;i<Math.min(7000,N);i++)ids.push(Math.floor(i*N/Math.min(7000,N)));return Array.from(new Set([...ids,selected]))}
async function render(){if(busy){pending=true;return}busy=true;try{const s=state(),vals=values(s),ids=background(s),good=ids.filter(i=>finite(vals[i])),missing=ids.filter(i=>!finite(vals[i])),lim=limit(s);$('stage').disabled=!['observed','fitted'].includes(s.map);$('interval').disabled=s.map!=='rate';$('supported').disabled=s.map!=='rate';const col=s.map==='rate'?[[0,'#2166ac'],[.25,'#67a9cf'],[.5,'#f7f7f7'],[.75,'#ef8a62'],[1,'#b2182b']]:'Viridis';function trace(ii,missing){const tr={type:s.d===3?'scatter3d':'scattergl',mode:'markers',x:ii.map(i=>xy(s,i)[0]),y:ii.map(i=>xy(s,i)[1]),customdata:ii,text:ii.map(i=>DATA.peaks[i].peak),hovertemplate:'%{text}<extra></extra>',showlegend:false,marker:{size:s.d===3?2.3:3,opacity:s.d===3?1:.8,color:missing?'#c8cdd2':s.map==='mode'?ii.map(i=>modeColors[vals[i]]):ii.map(i=>vals[i]),line:{width:0}}};if(s.d===3)tr.z=ii.map(i=>xy(s,i)[2]);if(!missing&&s.map!=='mode')Object.assign(tr.marker,{colorscale:col,cmin:s.map==='rate'?-lim:0,cmax:lim,colorbar:{title:{text:s.map==='rate'?'CPM / hpf':s.map==='distance'?'Graph hops':'log(1 + CPM)'},thickness:13,len:.7}});return tr}const traces=[trace(missing,true),trace(good,false)];if(s.root){let r=ids.filter(i=>A.root[i]);const tr=trace(r,false);tr.marker={size:s.d===3?4:7,color:'#e2aa3b',symbol:'circle-open',line:{width:1,color:'#a47009'}};tr.hoverinfo='skip';delete tr.hovertemplate;traces.push(tr)}const p=xy(s,selected),sel={type:s.d===3?'scatter3d':'scattergl',mode:'markers',x:[p[0]],y:[p[1]],marker:{size:8,color:'#f5c955',line:{width:2,color:'#173047'}},showlegend:false,hoverinfo:'skip'};if(s.d===3)sel.z=[p[2]];traces.push(sel);const label=s.map==='rate'?`${DATA.stages[s.k]} → ${DATA.stages[s.k+1]} hpf`:['fitted','observed'].includes(s.map)?`${DATA.stages[s.t]} hpf`:s.map==='distance'?'Geometry only':'All six stages';const title=`${DATA.celltypes[s.c].replaceAll('_',' ')} · ${label}`;await Plotly.react('atlas',traces,{title:{text:title,font:{size:17}},margin:{l:30,r:45,t:55,b:45},paper_bgcolor:'white',plot_bgcolor:'white',uirevision:'atlas'+s.d,xaxis:{title:'Peak UMAP 1',showgrid:false,zeroline:false},yaxis:{title:'Peak UMAP 2',showgrid:false,zeroline:false,scaleanchor:'x'},scene:{aspectmode:'data',xaxis:{title:'Peak UMAP 1'},yaxis:{title:'Peak UMAP 2'},zaxis:{title:'Peak UMAP 3'},camera:{eye:{x:1.4,y:1.4,z:1.1}}}},{responsive:true,displaylogo:false});$('status').textContent=`${DATA.total_peaks.toLocaleString()} analyzed peaks · ${N.toLocaleString()} searchable exported peaks · ${good.length.toLocaleString()} colored / ${ids.length.toLocaleString()} shown`;let caption=s.map==='rate'?`Red = opening; blue = closing. ${s.support?'Stored supported changes only.':'All estimable changes, including unsupported/low-count changes; exploratory.'} Fixed scale across intervals: ±${lim.toPrecision(3)} CPM/hpf.`:s.map==='distance'?'Shortest-path distance from the broad reference on the original graph. Not developmental pseudotime; unreachable vertices are gray.':s.map==='mode'?Object.entries(DATA.mode_labels).map(([k,v])=>`<span style="color:${modeColors[+k]}">■</span> ${esc(v.replaceAll('_',' '))}`).join(' · '):`Accessibility in log(1 + CPM). Scale is shared across stages within this celltype (0–${lim.toPrecision(3)}); scales may differ between celltypes. Colors saturate at the upper display limit.`;$('caption').innerHTML=caption+'<br>Gray = missing / low-cell / unsupported, not closed. Gold rings, when enabled, locate broad-reference peaks.';await renderCurves(s)}catch(e){$('error').textContent=e.stack||String(e);console.error(e)}finally{busy=false;if(pending){pending=false;await render()}}}
async function renderCurves(s){const p=DATA.peaks[selected],obs=curve(A.observed,selected,s.c),fit=curve(A.fitted,selected,s.c),mode=DATA.mode_labels[String(A.mode[selected*C+s.c])];$('selected').textContent=p.peak+' · '+([...new Set([p.nearest_gene,p.linked_gene,p.associated_gene].filter(x=>x&&x!=='nan'))].join(' / '))+' · region '+p.leiden_coarse+' · '+mode.replaceAll('_',' ');let tr=[{x:DATA.stages,y:obs,mode:'markers',type:'scatter',name:'Observed CPM',marker:{size:8,color:DATA.stages.map((_,t)=>A.condition[s.c*T+t]?'#17374f':'#aab1b9'),symbol:DATA.stages.map((_,t)=>A.condition[s.c*T+t]?'circle':'x')}}];if(fit.some(v=>v!==null))tr.push({x:DATA.stages,y:fit,mode:'lines+markers',type:'scatter',name:A.lof[selected*C+s.c]<=4?'Fitted stage values':'Flagged fit',connectgaps:false,line:{color:'#be7334',width:2,dash:A.lof[selected*C+s.c]<=4?'solid':'dot'}});await Plotly.react('curve',tr,{margin:{l:58,r:16,t:30,b:45},xaxis:{title:'Stage (hpf)',tickvals:DATA.stages},yaxis:{title:'Relative accessibility (CPM)',rangemode:'tozero'},legend:{orientation:'h',y:1.15},uirevision:'curve'+selected+':'+s.c},{responsive:true,displaylogo:false});await Plotly.react('heatmap',[{type:'heatmap',x:DATA.stages,y:DATA.celltypes,z:DATA.celltypes.map((_,c)=>curve(A.observed,selected,c).map(v=>v===null?null:Math.log1p(v))),colorscale:'Viridis',colorbar:{title:'log(1+CPM)',thickness:10},hovertemplate:'%{y}<br>%{x} hpf<br>%{z:.3g} log(1+CPM)<extra></extra>',hoverongaps:false}],{title:{text:'Selected peak · measured accessibility',font:{size:13}},margin:{l:155,r:50,t:35,b:40},xaxis:{title:'Stage (hpf)',tickvals:DATA.stages},yaxis:{tickfont:{size:8}},plot_bgcolor:'#c8cdd2'},{responsive:true,displaylogo:false})}
for(const id of ['dimension','celltype','map','stage','interval','supported','root'])$(id).addEventListener('change',render);$('find').onclick=()=>{const q=$('search').value.trim().toLowerCase();if(!q)return;const hits=DATA.peaks.flatMap((p,i)=>Object.values(p).some(v=>typeof v==='string'&&v.toLowerCase().includes(q))?[i]:[]);if(!hits.length){$('status').textContent='No matching peak in exported display subset; full data remain in the analysis files.';return}lastHit=q===lastSearch?(lastHit+1)%hits.length:0;lastSearch=q;selected=hits[lastHit];render()};$('search').onkeydown=e=>{if(e.key==='Enter')$('find').click()};$('save').onclick=()=>Plotly.downloadImage('atlas',{format:'png',filename:'chromatin_program_usage',width:1600,height:1100});
(async()=>{try{options('celltype',DATA.celltypes.map((c,i)=>[i,c.replaceAll('_',' ')]));options('stage',DATA.stages.map((t,i)=>[i,`${t} hpf`]));options('interval',DATA.stages.slice(0,-1).map((t,i)=>[i,`${t} → ${DATA.stages[i+1]} hpf`]));await Promise.all(Object.entries(DATA.arrays).map(async([k,x])=>{const bytes=Uint8Array.from(atob(x.gzip_base64),c=>c.charCodeAt(0));const buf=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();A[k]=x.dtype==='uint8'?new Uint8Array(buf):new Float32Array(buf)}));if(!A.distance)$('map').querySelector('option[value="distance"]').disabled=true;await render();$('atlas').on('plotly_click',ev=>{const i=ev.points[0]?.customdata;if(Number.isInteger(i)){selected=i;render()}});window.viewerReady=true;}catch(e){$('error').textContent=e.stack||String(e);console.error(e)}})();
</script></body></html>'''

if __name__ == '__main__':
    main()
