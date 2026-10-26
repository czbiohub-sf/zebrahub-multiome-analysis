#!/usr/bin/env python3
"""Joint peak-stage UMAPs with exact-repeat caching and predeclared diagnostics.

Models see accessibility vectors only. Stage labels and peak IDs are used only
for grouped train/test selection and diagnostic temporal connections.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
import umap

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/state_trajectory_atlas_v1'
SEED = 20261020
PANELS = ['primary_21', 'matched_21', 'optic_24']


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def write_json(path, value):
    def cvt(v):
        if isinstance(v, (np.integer, np.floating)): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        raise TypeError(type(v).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=cvt, allow_nan=False) + '\n')
    os.replace(tmp, path)


def finite_number(x):
    return float(x) if np.isfinite(x) else None


def exact_cached_embedding(training, queries, dimensions, neighbors=30,
                           min_dist=.3, seed=SEED, transform_batch=30000):
    """Embed unique training vectors; transform each unseen vector exactly once.

    Deduplication removes multiplicity weighting for repeated training states.
    Queries matching training vectors use their learned coordinates verbatim.
    This avoids artificial motion of exactly stationary accessibility states.
    """
    training = np.ascontiguousarray(training, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if not np.isfinite(training).all() or not np.isfinite(queries).all():
        raise ValueError('UMAP inputs must be finite; do not impute missing conditions.')
    train_unique, train_inverse = np.unique(training, axis=0, return_inverse=True)
    query_unique, query_inverse = np.unique(queries, axis=0, return_inverse=True)
    all_unique, combined_inverse = np.unique(
        np.concatenate([train_unique, query_unique]), axis=0, return_inverse=True)
    training_codes = combined_inverse[:len(train_unique)]
    query_codes = combined_inverse[len(train_unique):]
    all_coordinates = np.full((len(all_unique), dimensions), np.nan, dtype=np.float32)
    log(f'Fit {dimensions}D: {len(training):,} states -> {len(train_unique):,} unique; '
        f'k={neighbors}, min_dist={min_dist}, seed={seed}')
    start = time.monotonic()
    model = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist,
                      n_components=dimensions, metric='euclidean', random_state=seed,
                      transform_seed=seed, n_jobs=2, low_memory=True,
                      n_epochs=200, verbose=False)
    model.fit(train_unique)
    fit_seconds = time.monotonic() - start
    if not np.isfinite(model.embedding_).all():
        raise ValueError('Non-finite training UMAP coordinates.')
    all_coordinates[training_codes] = model.embedding_
    unseen = np.flatnonzero(~np.isfinite(all_coordinates).all(axis=1))
    log(f'Transform {len(unseen):,} distinct unseen states, training took {fit_seconds:.1f}s')
    start = time.monotonic()
    for begin in range(0, len(unseen), transform_batch):
        take = unseen[begin:begin + transform_batch]
        all_coordinates[take] = model.transform(all_unique[take])
        log(f'Transformed {min(begin + transform_batch, len(unseen)):,}/{len(unseen):,}')
    transform_seconds = time.monotonic() - start
    if not np.isfinite(all_coordinates).all():
        raise ValueError('Non-finite cached UMAP coordinates.')
    query_coordinates = all_coordinates[query_codes][query_inverse]
    train_coordinates = all_coordinates[training_codes][train_inverse]
    components, labels = connected_components(model.graph_, directed=False)
    component_sizes = np.bincount(labels)
    information = {
        'n_training_states': len(training), 'n_unique_training_states': len(train_unique),
        'n_query_states': len(queries), 'n_unique_query_states': len(query_unique),
        'n_unique_training_and_query_states': len(all_unique),
        'n_transformed_distinct_unseen_states': len(unseen),
        'dimensions': dimensions, 'n_neighbors': neighbors, 'min_dist': min_dist,
        'random_state': seed, 'transform_seed': seed, 'metric': 'euclidean',
        'n_epochs': 200, 'transform_batch_size': transform_batch,
        'umap_version': umap.__version__, 'numpy_version': np.__version__,
        'fit_seconds': fit_seconds, 'transform_seconds': transform_seconds,
        'graph_connected_components': int(components),
        'graph_largest_component_fraction': float(component_sizes.max()/len(train_unique)),
        'graph_isolated_states': int(np.count_nonzero(np.diff(model.graph_.tocsr().indptr)==0)),
        'identical_state_rule': 'Exact float32 PCA rows receive identical cached coordinates across observed/fitted and stages.',
        'duplicate_density_caveat': 'Exact repeated training vectors are fitted once; repetition does not increase their density weight.',
    }
    cache = {'model': model, 'reference_states': all_unique,
             'reference_coordinates': all_coordinates, 'information': information}
    return query_coordinates, train_coordinates, cache, information



def transform_with_cache(cache, pca_states, transform_batch=30000, update_cache=False):
    """Transform additional PCA states into a saved model, honoring exact repeats.

    Input states must use the saved panel's feature transform and frozen PCA.
    Returned rows preserve query order; optionally extend the reusable cache.
    """
    queries=np.ascontiguousarray(pca_states,dtype=np.float32)
    if queries.ndim!=2 or not np.isfinite(queries).all():
        raise ValueError('Supply a finite, two-dimensional PCA state array.')
    known=cache['reference_states']
    if queries.shape[1]!=known.shape[1]:
        raise ValueError('Query features do not match the saved PCA coordinate system.')
    unique, inverse=np.unique(np.concatenate([known,queries]),axis=0,return_inverse=True)
    coordinates=np.full((len(unique),cache['reference_coordinates'].shape[1]),np.nan,dtype=np.float32)
    coordinates[inverse[:len(known)]]=cache['reference_coordinates']
    unseen=np.flatnonzero(~np.isfinite(coordinates).all(axis=1))
    for begin in range(0,len(unseen),transform_batch):
        take=unseen[begin:begin+transform_batch]
        coordinates[take]=cache['model'].transform(unique[take])
    if not np.isfinite(coordinates).all():
        raise ValueError('Non-finite transformed coordinates.')
    if update_cache:
        cache['reference_states']=unique
        cache['reference_coordinates']=coordinates
    return coordinates[inverse[len(known):]]


def neighborhood_diagnostics(x, y, seed=SEED, maximum=1500, neighbors=15):
    """Within a fixed held-out state sample; exact repeated vectors omitted."""
    _, index = np.unique(x, axis=0, return_index=True)
    rng = np.random.default_rng(seed)
    if len(index) > maximum:
        index = rng.choice(index, maximum, replace=False)
    index = np.sort(index)
    x, y = x[index], y[index]
    k = min(neighbors, (len(x)-1)//3)
    if k < 2:
        return {'n_unique_evaluation_states': len(x), 'neighbors': k}
    nx = NearestNeighbors(n_neighbors=k+1, n_jobs=2).fit(x).kneighbors(x, return_distance=False)[:,1:]
    ny = NearestNeighbors(n_neighbors=k+1, n_jobs=2).fit(y).kneighbors(y, return_distance=False)[:,1:]
    overlap = np.asarray([len(set(a).intersection(b))/k for a,b in zip(nx,ny)])
    # Pairwise distance rank agreement has no physical-unit interpretation.
    take = np.arange(min(len(x), 800))
    high_dist = pdist(x[take]); low_dist = pdist(y[take])
    scale = np.dot(high_dist, low_dist)/max(np.dot(low_dist, low_dist), 1e-30)
    return {
        'n_unique_evaluation_states': len(x), 'neighbors': k,
        'neighborhood_recall_mean': float(overlap.mean()),
        'neighborhood_recall_median': float(np.median(overlap)),
        'trustworthiness': float(trustworthiness(x,y,n_neighbors=k)),
        'pairwise_distance_spearman': finite_number(spearmanr(high_dist,low_dist).statistic),
        'distance_stress_after_one_scale': float(np.sqrt(np.sum((high_dist-scale*low_dist)**2)/np.sum(high_dist**2))),
        'evaluation_scope': 'Held-out states only, exact repeated feature vectors omitted; neighbors recomputed within the diagnostic sample.',
    }


def temporal_diagnostics(x, y, stages):
    dx = np.diff(x.astype(np.float64), axis=1)
    dy = np.diff(y.astype(np.float64), axis=1)
    length_x = np.linalg.norm(dx, axis=-1)
    length_y = np.linalg.norm(dy, axis=-1)
    dt = np.diff(stages)
    moving = length_x > 0
    stationary = ~moving
    result = {
        'n_evaluation_peaks': len(x), 'n_intervals': int(length_x.size),
        'n_stationary_intervals': int(stationary.sum()),
        'n_stationary_intervals_with_false_movement': int(np.count_nonzero(length_y[stationary] != 0)),
        'temporal_step_length_spearman': finite_number(spearmanr(length_x[moving],length_y[moving]).statistic) if moving.sum()>2 else None,
        'temporal_rate_magnitude_spearman': finite_number(spearmanr((length_x/dt)[moving],(length_y/dt)[moving]).statistic) if moving.sum()>2 else None,
        'temporal_length_units': 'All-PC Euclidean distance in the chosen accessibility transform; embedding lengths are display distances.',
    }
    # Normalized display speed distortion; excludes exact stationary states.
    if moving.sum():
        unit_x = length_x[moving]/np.median(length_x[moving])
        unit_y = length_y[moving]/np.median(length_y[moving])
        result['median_absolute_log2_relative_step_distortion'] = finite_number(np.median(np.abs(np.log2(np.maximum(unit_y,1e-12)/unit_x))))
    return result


def get_panel_data(panel, transform):
    with h5py.File(OUT/'sample_data.h5','r') as h:
        train = h['train_mask'][:].astype(bool)
        test = h['test_mask'][:].astype(bool)
        stages = h['stages'][:]
    with h5py.File(OUT/'state_pca.h5','r') as h:
        g = h[panel]
        stage_indices = g['stage_indices'][:]
        celltype_indices = g['celltype_indices'][:]
        observed = g[transform]['observed_pca'][:]
        fitted = g[transform]['fitted_pca'][:]
    return observed, fitted, train, test, stages[stage_indices], stage_indices, celltype_indices


def pilot():
    """Predeclared small sensitivity grid; baseline is not picked aesthetically."""
    out = OUT/'umap_pilot'; out.mkdir(exist_ok=True,parents=True)
    settings = [
        ('baseline', 'log1p', 30, .3, SEED),
        ('local', 'log1p', 15, .1, SEED),
        ('broad', 'log1p', 60, .6, SEED),
        ('second_seed', 'log1p', 30, .3, SEED+1),
        ('linear_cpm', 'linear', 30, .3, SEED),
    ]
    all_metrics = []
    all_coordinates = {}
    rng = np.random.default_rng(SEED+12)
    selected_train = selected_test = None
    for name, transform, neighbors, min_dist, seed in settings:
        observed, fitted, train, test, stages, si, ci = get_panel_data('primary_21', transform)
        if selected_train is None:
            selected_train = np.sort(rng.choice(np.flatnonzero(train),min(3000,train.sum()),replace=False))
            selected_test = np.sort(rng.choice(np.flatnonzero(test),min(500,test.sum()),replace=False))
        xtrain = observed[selected_train].reshape(-1,observed.shape[-1])
        xtest = observed[selected_test]
        destination = out/f'{name}.npz'
        metadata = out/f'{name}.json'
        if destination.exists() and metadata.exists():
            coords = np.load(destination)['observed_umap2d']
            info = json.loads(metadata.read_text())
            log(f'Reusing completed pilot {name}')
        else:
            coords, _, cache, info = exact_cached_embedding(xtrain,xtest.reshape(-1,xtest.shape[-1]),2,neighbors,min_dist,seed)
            coords = coords.reshape(*xtest.shape[:2],2)
            info.update({'setting':name,'feature_transform':transform,
                         **neighborhood_diagnostics(xtest.reshape(-1,xtest.shape[-1]),coords.reshape(-1,2)),
                         **temporal_diagnostics(xtest,coords,stages)})
            np.savez_compressed(destination, observed_umap2d=coords,train_peak_indices=selected_train,
                                test_peak_indices=selected_test,stages=stages)
            write_json(metadata,info)
        all_metrics.append(info); all_coordinates[name] = coords
    baseline = all_coordinates['baseline'].reshape(-1,2)
    for row in all_metrics:
        y = all_coordinates[row['setting']].reshape(-1,2)
        row['distance_rank_agreement_with_baseline_layout'] = finite_number(spearmanr(pdist(baseline[:800]),pdist(y[:800])).statistic)
    pd.DataFrame(all_metrics).to_csv(OUT/'umap_pilot_diagnostics.csv',index=False)
    write_json(OUT/'umap_pilot_summary.json',{
        'predeclared_baseline': {'n_neighbors':30,'min_dist':.3,'random_state':SEED},
        'selection_rule': 'Keep the predeclared baseline; sensitivity settings diagnose distortion and stability, not visual appeal.',
        'peak_sampling': 'The same 3,000 whole training peaks and 500 held-out peaks in every pilot setting.',
        'diagnostics':all_metrics})
    plot_pilot(all_metrics,all_coordinates,stages)


def plot_pilot(metrics,coordinates,stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[m['setting'] for m in metrics]
    fig, axes=plt.subplots(2,5,figsize=(19,7.5),gridspec_kw={'height_ratios':[1.35,1]})
    palette=plt.get_cmap('viridis')(np.linspace(0,1,len(stages)))
    for a,name,info in zip(axes[0],names,metrics):
        z=coordinates[name]
        for t,color in enumerate(palette):
            a.scatter(z[:,t,0],z[:,t,1],s=3,c=[color],alpha=.3,rasterized=True)
        for idx in np.linspace(0,len(z)-1,18,dtype=int):
            a.plot(z[idx,:,0],z[idx,:,1],lw=.5,alpha=.4,c='.25')
        a.set_title(f"{name.replace('_',' ')}\nk={info['n_neighbors']}, min_dist={info['min_dist']}")
        a.set_xticks([]);a.set_yticks([])
    measures=[('trustworthiness','Neighborhood trustworthiness'),('neighborhood_recall_mean','15-neighbor overlap'),
              ('temporal_step_length_spearman','Temporal step length rank'),('temporal_rate_magnitude_spearman','Temporal rate rank'),
              ('distance_rank_agreement_with_baseline_layout','Distance rank vs baseline')]
    for a,(key,title) in zip(axes[1],measures):
        values=[m.get(key,np.nan) for m in metrics]
        a.bar(range(len(values)),values,color=['#4477AA','#66CCEE','#228833','#CCBB44','#EE6677'])
        a.set_xticks(range(len(names)),[x.replace('_','\n') for x in names],rotation=35,ha='right',fontsize=8)
        a.set_title(title,fontsize=10);a.set_ylim(min(0,min(values)-.03),1.05);a.grid(axis='y',alpha=.15)
    fig.suptitle('Joint state UMAP: predeclared sensitivity checks on held-out whole peaks',fontsize=15)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],marker='o',linestyle='',color=color,markersize=5) for color in palette],
               [f'{stage:g} hpf' for stage in stages],loc='upper center',bbox_to_anchor=(.5,.945),ncol=len(stages),frameon=False,fontsize=9)
    fig.text(.02,.012,'Same 3,000 training peaks and 500 held-out peaks across settings. Distances and arrows are display quantities; temporal order was not an embedding input.',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.89])
    for suffix in ['png','pdf']: fig.savefig(OUT/f'04_umap_parameter_sensitivity.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def final_panel(panel, dimensions=(2,3)):
    models = OUT/'umap_models';models.mkdir(exist_ok=True)
    partials = OUT/'umap_coordinates';partials.mkdir(exist_ok=True)
    observed, fitted, train, test, stages, si, ci = get_panel_data(panel,'log1p')
    xtrain=observed[train].reshape(-1,observed.shape[-1])
    combined=np.concatenate([observed.reshape(-1,observed.shape[-1]),fitted.reshape(-1,fitted.shape[-1])])
    rng=np.random.default_rng(SEED+44)
    evaluation=np.sort(rng.choice(np.flatnonzero(test),min(750,test.sum()),replace=False))
    metadata={}
    for dimension in dimensions:
        dest=partials/f'{panel}_{dimension}d.npz';meta=partials/f'{panel}_{dimension}d.json'
        if dest.exists() and meta.exists():
            log(f'Reusing completed {panel} {dimension}D');metadata[str(dimension)]=json.loads(meta.read_text());continue
        coordinates,_,cache,info=exact_cached_embedding(xtrain,combined,dimension)
        count=observed.shape[0]*observed.shape[1]
        yobserved=coordinates[:count].reshape(*observed.shape[:2],dimension)
        yfitted=coordinates[count:].reshape(*fitted.shape[:2],dimension)
        info.update({'panel':panel,'feature_transform':'log1p','n_evaluation_peaks':len(evaluation),
                     'observed_diagnostics': {**neighborhood_diagnostics(observed[evaluation].reshape(-1,observed.shape[-1]),yobserved[evaluation].reshape(-1,dimension)),
                                              **temporal_diagnostics(observed[evaluation],yobserved[evaluation],stages)},
                     'fitted_diagnostics': {**neighborhood_diagnostics(fitted[evaluation].reshape(-1,fitted.shape[-1]),yfitted[evaluation].reshape(-1,dimension)),
                                            **temporal_diagnostics(fitted[evaluation],yfitted[evaluation],stages)},
                     'parameters_fixed_before_visualization':True})
        np.savez_compressed(dest,observed=yobserved,fitted=yfitted,stage_indices=si,celltype_indices=ci,
                            evaluation_peak_indices=evaluation)
        cache.update({'panel':panel,'feature_transform':'log1p','stage_indices':si,'celltype_indices':ci})
        joblib.dump(cache,models/f'{panel}_{dimension}d.joblib',compress=3)
        write_json(meta,info);metadata[str(dimension)]=info
        log(f'Saved {panel} {dimension}D coordinates, model and diagnostics')
    consolidate()
    return metadata


def consolidate():
    # Independent panel jobs may finish together; publication is serialized.
    with (OUT/'.state_umap.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _consolidate_locked()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _consolidate_locked():
    """Atomically publish all complete dimensions without leaving a locked H5."""
    temporary=OUT/f'state_umap.h5.{os.getpid()}.tmp'
    manifest={}
    with h5py.File(temporary,'w') as output:
        output.attrs['meaning']='Each point is one peak at one stage. One observed-training UMAP per panel and dimension; fitted and held-out states transformed into the frozen map.'
        output.attrs['no_temporal_constraints']=True
        output.attrs['selection_rule']='Predeclared k=30, min_dist=0.3; no optimization for outward arrows.'
        for panel in PANELS:
            paths={dim:OUT/'umap_coordinates'/f'{panel}_{dim}d.npz' for dim in [2,3]}
            available=[dim for dim,p in paths.items() if p.exists() and p.with_suffix('.json').exists()]
            if not available: continue
            g=output.create_group(panel);manifest[panel]={}
            g.attrs['feature_transform']='log1p';g.attrs['n_neighbors']=30;g.attrs['min_dist']=.3
            g.attrs['seed']=SEED;g.attrs['metric']='euclidean';g.attrs['n_epochs']=200
            g.attrs['selection_rule']='Predeclared baseline; sensitivity diagnostics do not select by visual appearance.'
            for dim in available:
                with np.load(paths[dim]) as data:
                    for kind in ['observed','fitted']:
                        g.create_dataset(f'{kind}_umap{dim}d',data=data[kind],compression='gzip',shuffle=True)
                    if 'stage_indices' not in g:
                        for key in ['stage_indices','celltype_indices','evaluation_peak_indices']:
                            g.create_dataset(key,data=data[key])
                manifest[panel][str(dim)]=json.loads(paths[dim].with_suffix('.json').read_text())
    os.replace(temporary,OUT/'state_umap.h5')
    write_json(OUT/'umap_full_diagnostics.json',manifest)




def plot_full_diagnostics():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=OUT/'umap_full_diagnostics.json'
    if not path.exists(): return
    data=json.loads(path.read_text())
    entries=[(panel,dimension,data[panel][str(dimension)]) for panel in PANELS
             if panel in data and all(str(dim) in data[panel] for dim in [2,3])
             for dimension in [2,3]]
    if not entries: return
    labels={ 'primary_21':'21 celltypes, 6 stages', 'matched_21':'21 celltypes, 5 stages', 'optic_24':'24 celltypes, 5 stages'}
    ticklabels=[f"{labels[panel]}\n{dim}D" for panel,dim,_ in entries]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for column,source in enumerate(['observed','fitted']):
        for row,(metric,title,ylabel) in enumerate([
            ('temporal_step_length_spearman','Temporal step-length agreement','Spearman correlation'),
            ('neighborhood_recall_mean','Local accessibility-neighborhood overlap','Mean overlap of 15 neighbors')]):
            ax=axes[row,column]
            values=[entry[f'{source}_diagnostics'][metric] for _,_,entry in entries]
            colors=['#4477AA' if dim==2 else '#228833' for _,dim,_ in entries]
            ax.bar(np.arange(len(entries)),values,color=colors)
            for i,value in enumerate(values): ax.text(i,value+.025,f'{value:.3f}',ha='center',fontsize=10)
            ax.set_ylim(min(0,min(values)-.08),1.05)
            ax.set_xticks(np.arange(len(entries)),ticklabels,rotation=20,ha='right',fontsize=8)
            ax.set_title(f'{source.capitalize()} states: {title}',fontsize=12)
            ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Does the joint UMAP preserve measured accessibility-state changes?',fontsize=16)
    fig.text(.02,.015,'750 held-out peaks per panel. Neighborhoods are recomputed within up to 1,500 distinct held-out states.\nHigh-dimensional references use all PCs of log1p(CPM). UMAP lengths are display distances, not accessibility rates.',fontsize=10)
    fig.tight_layout(rect=[0,.075,1,.94])
    for suffix in ['png','pdf']: fig.savefig(OUT/f'06_umap_preservation_diagnostics.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def compare_panels():
    """Compare matching late-stage states without equating UMAP axis directions."""
    with h5py.File(OUT/'state_umap.h5','r') as h:
        if any(panel not in h or any(f'observed_umap{dim}d' not in h[panel] for dim in [2,3]) for panel in PANELS): return
        with h5py.File(OUT/'sample_data.h5','r') as sample:
            test=np.flatnonzero(sample['test_mask'][:].astype(bool))
        rng=np.random.default_rng(SEED+44)
        chosen=np.sort(rng.choice(test,min(750,len(test)),replace=False))
        coords={panel:{dim:h[panel][f'observed_umap{dim}d'][chosen] for dim in [2,3]} for panel in PANELS}
    original={}
    for panel in PANELS:
        observed,*_=get_panel_data(panel,'log1p')
        original[panel]=observed[chosen]
        if panel=='primary_21':
            original[panel]=original[panel][:,1:]
            coords[panel]={dim:z[:,1:] for dim,z in coords[panel].items()}
    rows=[]
    for left,right,meaning in [('primary_21','matched_21','Effect of excluding 10 hpf from model training, with identical 21 celltype features.'),
                               ('matched_21','optic_24','Effect of adding 3 adequately sampled late-stage celltype features, including optic cup.')]:
        xleft=original[left].reshape(-1,original[left].shape[-1])
        xright=original[right].reshape(-1,original[right].shape[-1])
        xl=pdist(xleft[:800]); xr=pdist(xright[:800])
        sl=np.linalg.norm(np.diff(original[left],axis=1),axis=-1).ravel()
        sr=np.linalg.norm(np.diff(original[right],axis=1),axis=-1).ravel()
        nonzero=(sl>0)&(sr>0)
        for dim in [2,3]:
            yl=coords[left][dim].reshape(-1,dim); yr=coords[right][dim].reshape(-1,dim)
            evaluation=np.sort(rng.choice(len(yl),min(1500,len(yl)),replace=False))
            a=NearestNeighbors(n_neighbors=16).fit(yl[evaluation]).kneighbors(yl[evaluation],return_distance=False)[:,1:]
            b=NearestNeighbors(n_neighbors=16).fit(yr[evaluation]).kneighbors(yr[evaluation],return_distance=False)[:,1:]
            rows.append({'left_panel':left,'right_panel':right,'dimensions':dim,'comparison':meaning,
                         'high_dimensional_distance_spearman':finite_number(spearmanr(xl,xr).statistic),
                         'high_dimensional_temporal_step_spearman':finite_number(spearmanr(sl[nonzero],sr[nonzero]).statistic),
                         'embedded_pairwise_distance_spearman':finite_number(spearmanr(pdist(yl[:800]),pdist(yr[:800])).statistic),
                         'embedded_neighborhood_overlap':float(np.mean([len(set(u).intersection(v))/15 for u,v in zip(a,b)])),
                         'n_matching_heldout_peaks':len(chosen),'matching_stages':'12,14,16,19,24'})
    pd.DataFrame(rows).to_csv(OUT/'umap_panel_comparison.csv',index=False)
    write_json(OUT/'umap_panel_comparison.json',{'interpretation':'Compare corresponding late-stage states and neighborhood geometry; independently learned UMAP axes have no shared physical direction.', 'comparisons':rows})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','full','all','consolidate'],default='all')
    parser.add_argument('--panels',nargs='+',choices=PANELS,default=PANELS)
    parser.add_argument('--dimensions',nargs='+',type=int,choices=[2,3],default=[2,3])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage in ['pilot','all']: pilot()
    if args.stage in ['full','all']:
        for panel in args.panels: final_panel(panel,args.dimensions)
        if len(args.dimensions)==2: compare_panels()
    if args.stage=='consolidate':
        consolidate()
        compare_panels()
    if args.stage in ['full','all','consolidate']:plot_full_diagnostics()

if __name__=='__main__':main()
